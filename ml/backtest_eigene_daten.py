"""
Ehrlicher Praxistest: Wie gut ist das Modell auf der EIGENEN Sammlung?

Das Modell wurde auf 2025er-Daten trainiert (altes Fahrplanjahr). Hier laeuft es
gegen die selbst gesammelten Tage aus September/Oktober 2026 - also aktuelles
Fahrplanjahr, komplett ausserhalb der Trainingsdaten.

Fuer JEDE RE19-Abfahrt in Doerfles-Esbach:
  - Vorhersagezeitpunkt T = planmaessige Abfahrt minus 15 min
  - Features aus dem Stand, der zu T bekannt war
  - Vergleich: Modell vs. DB-Prognose (Stand T) vs. tatsaechliche Abfahrt

Liest nur aus ~/re19-watch/data/, aendert dort nichts.
"""
import json
import os
import pathlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
TZ = ZoneInfo("Europe/Berlin")
RE19_DIR = os.path.expanduser("~/re19-watch")

DOERFLES = "8001484"
UPSTREAM = ["8013008", "8004325", "8004064", "8005122", "8004633"]
UP_NAME = {"8013008": "sonneberg", "8004325": "neustadt", "8004064": "moenchroeden",
           "8005122": "roedentalmitte", "8004633": "roedental"}
CONFLICT_WINDOW = 30
VOR_WINDOW = 45        # Suchfenster fuer den vorausfahrenden Zug (wie im Training)
LEAD_MIN = 15

FEATURES = [
    "db_delay_now",
    "up_sonneberg", "up_neustadt", "up_moenchroeden", "up_roedentalmitte", "up_roedental",
    "n_upstream_known", "last_known_delay", "max_upstream_delay", "upstream_trend",
    "conflict_max_delay", "conflict_mean_delay", "n_conflict_trips",
    "ice_max_delay", "n_ice_nearby", "conflict_gap_min", "ice_gap_min",
    "vorgaenger_delay", "vorgaenger_luecke_min",
    "hour", "minute_of_day", "dow", "month", "is_weekend",
]


def load_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def parse(s):
    return datetime.fromisoformat(s) if s else None


def build_day(day):
    plan = load_jsonl(f"{RE19_DIR}/data/{day}/plan.jsonl")
    changes = load_jsonl(f"{RE19_DIR}/data/{day}/changes.jsonl")
    if not plan or not changes:
        return []

    plan_by, meta = {}, {}
    for p in plan:
        plan_by[(p["trip_id"], p["eva"], p["kind"])] = parse(p.get("planned"))
        meta[p["trip_id"]] = {"line": p.get("line"), "number": p.get("number"),
                       "category": p.get("category")}

    chg = sorted((c for c in changes if c.get("changed")), key=lambda c: c["snapshot"])

    # letzter bekannter Stand ueberhaupt = Ist
    final = {}
    for c in chg:
        final[(c["trip_id"], c["eva"], c["kind"])] = (parse(c["changed"]), c.get("status"), parse(c["snapshot"]))

    def known_at(T):
        k = {}
        for c in chg:
            if parse(c["snapshot"]) > T:
                break
            k[(c["trip_id"], c["eva"], c["kind"])] = parse(c["changed"])
        return k

    rows = []
    targets = [(pl, t) for (t, eva, kind), pl in plan_by.items()
               if eva == DOERFLES and kind == "departure"
               and meta.get(t, {}).get("line") == "RE19" and pl]

    for sched_dep, trip in sorted(targets):
        ist_entry = final.get((trip, DOERFLES, "departure"))
        if not ist_entry or ist_entry[1] == "c":      # kein Ist oder ausgefallen
            continue
        y = (ist_entry[0] - sched_dep).total_seconds() / 60.0
        T = sched_dep - timedelta(minutes=LEAD_MIN)

        # Nur Ziele verwenden, die nach T und nach der prognostizierten Zeit
        # noch einmal als Echtzeitstand bestaetigt wurden.
        if ist_entry[2] <= T or ist_entry[2] < ist_entry[0] + timedelta(minutes=1):
            continue
        kn = known_at(T)

        def delay(tr, eva, kind):
            pl = plan_by.get((tr, eva, kind))
            v = kn.get((tr, eva, kind))
            return np.nan if (pl is None or v is None) else (v - pl).total_seconds() / 60.0

        r = {"db_delay_now": delay(trip, DOERFLES, "departure")}
        ups = []
        for eva in UPSTREAM:
            d = delay(trip, eva, "departure")
            if np.isnan(d):
                d = delay(trip, eva, "arrival")
            r["up_" + UP_NAME[eva]] = d
            ups.append(d)
        a = np.array(ups, float)
        valid = a[~np.isnan(a)]
        r["n_upstream_known"] = int(len(valid))
        r["last_known_delay"] = valid[-1] if len(valid) else np.nan
        r["max_upstream_delay"] = np.nanmax(a) if len(valid) else np.nan
        r["upstream_trend"] = (valid[-1] - valid[0]) if len(valid) else np.nan

        lo, hi = sched_dep - timedelta(minutes=CONFLICT_WINDOW), sched_dep + timedelta(minutes=CONFLICT_WINDOW)
        cd, ntr = {}, set()
        passages, ice_delays, ice_passages = [], [], []
        for (t2, e2, k2), pl2 in plan_by.items():
            if t2 == trip or not pl2 or not (lo <= pl2 <= hi):
                continue
            info = meta.get(t2, {})
            if info.get("line") not in ("RE28", "RE29") and info.get("category") != "ICE":
                continue
            ntr.add(t2)
            d = delay(t2, e2, k2)
            if not np.isnan(d):
                cd[t2] = d
                passage = pl2 + timedelta(minutes=d)
                passages.append(passage)
                if info.get("category") == "ICE":
                    ice_delays.append(d)
                    ice_passages.append(passage)
        r["conflict_max_delay"] = max(cd.values()) if cd else np.nan
        r["conflict_mean_delay"] = float(np.mean(list(cd.values()))) if cd else np.nan
        r["n_conflict_trips"] = len(ntr)
        r["ice_max_delay"] = max(ice_delays) if ice_delays else np.nan
        r["n_ice_nearby"] = len({t2 for t2 in ntr if meta.get(t2, {}).get("category") == "ICE"})
        r["conflict_gap_min"] = min(abs((p - sched_dep).total_seconds() / 60)
                                     for p in passages) if passages else np.nan
        r["ice_gap_min"] = min(abs((p - sched_dep).total_seconds() / 60)
                                for p in ice_passages) if ice_passages else np.nan

        # Vorausfahrender/kreuzender Zug - identisch zu ml/build_features.py
        vor = [(pl2, t2) for (t2, e2, k2), pl2 in plan_by.items()
               if e2 == DOERFLES and k2 == "departure" and t2 != trip and pl2
               and meta.get(t2, {}).get("line") in ("RE19", "RE28", "RE29")
               and sched_dep - timedelta(minutes=VOR_WINDOW) <= pl2 < sched_dep]
        if vor:
            pl_vor, t_vor = max(vor)
            r["vorgaenger_delay"] = delay(t_vor, DOERFLES, "departure")
            r["vorgaenger_luecke_min"] = (sched_dep - pl_vor).total_seconds() / 60
        else:
            r["vorgaenger_delay"] = np.nan
            r["vorgaenger_luecke_min"] = np.nan

        loc = sched_dep.astimezone(TZ)
        r.update(hour=loc.hour, minute_of_day=loc.hour * 60 + loc.minute,
                 dow=loc.weekday(), month=loc.month, is_weekend=int(loc.weekday() >= 5))
        r.update(day=day, trip=trip, number=meta.get(trip, {}).get("number"),
                 sched=loc, y_delay_min=y)
        rows.append(r)
    return rows


def main():
    days = sorted(d for d in os.listdir(f"{RE19_DIR}/data")
                  if d.startswith("2026-") and d >= "2026-09-24")
    allrows = []
    for d in days:
        allrows += build_day(d)
    df = pd.DataFrame(allrows)
    print(f"Tage: {days[0]} bis {days[-1]}  ({len(days)} Tage)")
    print(f"Auswertbare RE19-Abfahrten Doerfles: {len(df)}\n")
    if df.empty:
        return

    model = joblib.load(HERE / "model_werrabahn.joblib")
    df["pred"] = model.predict(df[FEATURES])

    y = df["y_delay_min"].to_numpy()
    pm = df["pred"].to_numpy()
    pb = df["db_delay_now"].to_numpy()
    ok = ~np.isnan(pb)

    def line(name, m):
        if m.sum() == 0:
            return
        mm = m & ok
        mae_model = np.abs(pm[mm] - y[mm]).mean()
        mae_db = np.abs(pb[mm] - y[mm]).mean()
        better = (np.abs(pm[mm] - y[mm]) < np.abs(pb[mm] - y[mm])).mean() * 100
        print(f"{name:<32} n={mm.sum():>4}  Modell={mae_model:5.2f} min  "
              f"DB={mae_db:5.2f} min  Modell besser: {better:4.1f}%")

    print("=== Praxistest auf eigenen Daten (aktuelles Fahrplanjahr) ===")
    line("Alle RE19-Abfahrten", np.ones(len(df), bool))
    line("Morgens 6-8 Uhr, Mo-Fr", ((df["hour"].isin([6, 7])) & (df["is_weekend"] == 0)).to_numpy())
    line("Ist-Verspaetung >= 5 min", (y >= 5))
    line("Ist-Verspaetung < 5 min", (y < 5))

    print("\n=== Die groessten Verspaetungen im Zeitraum ===")
    top = df.nlargest(8, "y_delay_min")[["day", "number", "sched", "db_delay_now", "pred", "y_delay_min"]]
    for _, r in top.iterrows():
        db = "  -  " if pd.isna(r["db_delay_now"]) else f"{r['db_delay_now']:+5.0f}"
        print(f"  {r['day']} {r['sched']:%H:%M} RE19 {r['number']:<6} "
              f"DB={db}  Modell={r['pred']:+5.1f}  IST={r['y_delay_min']:+5.0f} min")


if __name__ == "__main__":
    main()
