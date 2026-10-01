"""
Stufe A, v4: wie replay_v3, aber statt Zeiten aus EINEM Anker (Coburg) hochzurechnen,
werden fuer JEDEN Streckenpunkt die ECHTEN beobachteten Zeiten aus changes.jsonl
verwendet (die liegen fuer alle 4 realen Korridor-Stationen vor: Roedental,
Doerfles-Esbach, Coburg Nord, Coburg). Nur "Abzweig SFS" (kein echter Bahnhof)
wird weiter interpoliert.

Liest NUR aus ~/re19-watch/data/2026-09-24/. Importiert simulate_fixed.py (Kopie mit
Bugfix aus dem letzten Schritt) und re19watch.py NUR LESEND. Nichts im Original
veraendert.
"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

RE19_DIR = os.path.expanduser("~/re19-watch")
SCRATCHPAD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RE19_DIR)
sys.path.insert(0, SCRATCHPAD)

import simulate_fixed as simmod  # noqa: E402
sys.modules["simulate"] = simmod
from simulate_fixed import Train, Corridor, estimate  # noqa: E402
import re19watch as rw  # noqa: E402  (nur lesend importiert)

TZ = ZoneInfo("Europe/Berlin")
DAY_DIR = os.path.join(RE19_DIR, "data", "2026-09-24")

# Reale EVA je Korridorpunkt (Abzweig SFS hat keine eigene EVA -> wird interpoliert)
POINT_EVA = {
    "Rödental": "8004633",
    "Dörfles-Esbach": "8001484",
    "Coburg Nord": "8001334",
    "Coburg": "8001338",
}
COBURG_EVA = "8001338"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_iso(s):
    return datetime.fromisoformat(s) if s else None


def classify_path(path_list, cls_cfg):
    txt = " ".join(path_list or []).lower()
    if any(k.lower() in txt for k in cls_cfg["via_sfs"]):
        return "sfs"
    if any(k.lower() in txt for k in cls_cfg["via_werrabahn"]):
        return "werrabahn"
    return None


def run_day(day: str) -> list[dict]:
    day_dir = os.path.join(RE19_DIR, "data", day)
    with open(os.path.join(RE19_DIR, "config.toml"), "rb") as f:
        cfg = tomllib.load(f)
    corridor = rw.build_corridor(cfg["corridor"])
    coburg_pt = cfg["corridor"]["coburg_point"]
    junction_pt = cfg["corridor"]["junction_point"]
    wb_end = cfg["corridor"]["werrabahn_end"]
    cls_cfg = cfg["classify"]

    plan = load_jsonl(os.path.join(day_dir, "plan.jsonl"))
    changes = load_jsonl(os.path.join(day_dir, "changes.jsonl"))

    # plan[(trip_id, eva, kind)] -> planned dt ; changes[(trip_id, eva, kind)] -> sortierte Liste
    plan_by = {}
    for p in plan:
        plan_by[(p["trip_id"], p["eva"], p["kind"])] = parse_iso(p["planned"])

    chg_by = defaultdict(list)
    for c in changes:
        chg_by[(c["trip_id"], c["eva"], c["kind"])].append(c)
    for k in chg_by:
        chg_by[k].sort(key=lambda c: c["snapshot"])

    def prog_at(trip_id, eva, kind, t):
        best = None
        for c in chg_by.get((trip_id, eva, kind), []):
            sn = parse_iso(c["snapshot"])
            if sn <= t:
                best = c
            else:
                break
        return parse_iso(best["changed"]) if best and best.get("changed") else None

    def ist_at(trip_id, eva, kind):
        lst = chg_by.get((trip_id, eva, kind), [])
        if not lst:
            return None
        last = lst[-1]
        return parse_iso(last["changed"]) if last.get("changed") else None

    coburg_plan = {}
    for p in plan:
        if p["eva"] == COBURG_EVA:
            coburg_plan[(p["trip_id"], p["kind"])] = p

    trips = {}
    for (trip_id, kind), p in coburg_plan.items():
        route = classify_path(p.get("path"), cls_cfg)
        if route is None:
            continue
        trips[(trip_id, kind)] = {
            "route": route, "line": p.get("line") or p.get("category"),
            "category": p.get("category"), "number": p.get("number"),
            "planned": parse_iso(p["planned"]),
        }

    def priority_for(category, line):
        overrides = {rw.norm_line(k): v for k, v in cfg.get("priority", {}).get("lines", {}).items()}
        prod_map = {"ICE": "nationalExpress", "IC": "national", "EC": "national",
                    "RE": "regionalExpress", "ag": "regional", "STB": "regional", "Bus": "regional"}
        prod = prod_map.get(category, "regional")
        return overrides.get(rw.norm_line(line), rw.PRIORITY.get(prod, 0))

    def cls_for(category):
        return "fast" if category == "ICE" else "regional"

    def real_path_points(route, kind):
        """Reale Korridor-Stationen in Fahrtrichtung (ohne Abzweig SFS)."""
        if route == "sfs":
            pts = [coburg_pt, "Dörfles-Esbach", "Coburg Nord"]  # Coburg<->Dörfles-Esbach ueber Coburg Nord
            pts = [coburg_pt, "Coburg Nord", "Dörfles-Esbach"]
        else:
            pts = [coburg_pt, "Coburg Nord", "Dörfles-Esbach", wb_end]
        return pts if kind == "departure" else list(reversed(pts))

    def build_real_train(trip_id, kind, route, info, t, label, use_prog):
        """Baut ein Train-Objekt mit ECHTEN beobachteten Zeiten je Realstation.
        Fehlt eine Beobachtung, wird auf den Plan zurueckgefallen; 'Abzweig SFS'
        wird linear zwischen den beiden Nachbarpunkten interpoliert."""
        pts = real_path_points(route, kind)
        times = {}
        for i, pt in enumerate(pts):
            eva = POINT_EVA[pt]
            # An Zwischenstationen: Ankunft nehmen wenn letzter Punkt, sonst Abfahrt (bzw. Ankunft falls keine Abfahrt geloggt)
            is_last = i == len(pts) - 1
            prefer = "arrival" if is_last else "departure"
            fallback = "departure" if is_last else "arrival"
            for k in (prefer, fallback):
                pt_planned = plan_by.get((trip_id, eva, k))
                if pt_planned is None:
                    continue
                val = (prog_at(trip_id, eva, k, t) if use_prog else pt_planned) or pt_planned
                times[pt] = val
                break
        if len(times) < 2:
            return None
        full_path = corridor.path(pts[0], pts[-1])
        # Abzweig SFS interpolieren (liegt zwischen Coburg Nord und Doerfles-Esbach)
        if "Abzweig SFS" in full_path and "Abzweig SFS" not in times:
            i = full_path.index("Abzweig SFS")
            a, b = full_path[i - 1], full_path[i + 1]
            if a in times and b in times:
                seg_ab = corridor.run_min[corridor.key(a, "Abzweig SFS")]["regional"]
                seg_full = seg_ab + corridor.run_min[corridor.key("Abzweig SFS", b)]["regional"]
                frac = seg_ab / seg_full
                times["Abzweig SFS"] = times[a] + (times[b] - times[a]) * frac
        if any(p not in times for p in full_path):
            return None
        direction = 1 if corridor.points.index(pts[-1]) > corridor.points.index(pts[0]) else -1
        prio = priority_for(info["category"], info["line"])
        cls = cls_for(info["category"])
        return Train(label, prio, cls, full_path, times, direction, trip_id)

    results = []
    for (trip_id, kind), info in trips.items():
        if info["route"] != "werrabahn":
            continue
        planned_t = info["planned"]
        if planned_t is None:
            continue
        snaps = sorted({c["snapshot"] for c in chg_by.get((trip_id, COBURG_EVA, kind), [])})
        if not snaps:
            continue
        ist = ist_at(trip_id, COBURG_EVA, kind)
        if ist is None:
            continue

        for sn in snaps:
            t = parse_iso(sn)
            db_when = prog_at(trip_id, COBURG_EVA, kind, t) or planned_t

            others_plan, others_prog = [], []
            for (o_tid, o_kind), o_info in trips.items():
                if o_tid == trip_id:
                    continue
                o_planned = o_info["planned"]
                if o_planned is None or abs((o_planned - planned_t).total_seconds()) > 2 * 3600:
                    continue
                label = f"{o_info['line']} {o_info['number']}"
                tp = build_real_train(o_tid, o_kind, o_info["route"], o_info, t, label, False)
                tg = build_real_train(o_tid, o_kind, o_info["route"], o_info, t, label, True)
                if tp:
                    others_plan.append(tp)
                if tg:
                    others_prog.append(tg)

            me_plan = build_real_train(trip_id, kind, "werrabahn", info, planned_t, "MICH", False)
            me_prog = build_real_train(trip_id, kind, "werrabahn", info, t, "MICH", True)
            if me_plan is None or me_prog is None:
                continue
            try:
                est = estimate(me_plan, others_plan, me_prog, others_prog, corridor, coburg_pt, planned_t, db_when)
            except Exception:
                continue

            results.append({
                "day": day, "trip_id": trip_id, "kind": kind, "line": info["line"], "number": info["number"],
                "snapshot": sn, "planned": planned_t.isoformat(), "db_when": db_when.isoformat(),
                "model_estimate": est.estimate.isoformat(), "ist": ist.isoformat(),
                "extra_vs_db_min": est.extra_vs_db.total_seconds() / 60,
                "db_error_min": (db_when - ist).total_seconds() / 60,
                "model_error_min": (est.estimate - ist).total_seconds() / 60,
                "holds": len(est.holds),
            })

    return results


def main():
    days = ["2026-09-24", "2026-09-25", "2026-09-26"]
    all_results = []
    for day in days:
        r = run_day(day)
        print(f"{day}: {len(r)} ausgewertete (Zug,Snapshot)-Kombinationen")
        all_results.extend(r)

    out = os.path.join(SCRATCHPAD, "results_multiday.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1, default=str)
    print(f"\nGesamt: {len(all_results)} gespeichert nach", out)


if __name__ == "__main__":
    main()
