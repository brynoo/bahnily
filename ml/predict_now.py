"""
Live-Abfrage: "Wann faehrt mein RE19 in Doerfles-Esbach wirklich ab?"

Nutzt das auf den 2025er-Daten trainierte Modell, gefuettert mit dem AKTUELLEN
Stand aus der eigenen Sammlung (~/re19-watch/data/<datum>/). Liest dort nur,
aendert nichts.

Aufruf:
    python predict_now.py                 # jetzt, naechster RE19 ab Doerfles
    python predict_now.py --at 07:05      # so tun, als waere es 07:05 (heute)
    python predict_now.py --at 07:05 --date 2026-09-29
"""
import argparse
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

import pathlib
HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"


TZ = ZoneInfo("Europe/Berlin")
RE19_DIR = os.path.expanduser("~/re19-watch")
SCRATCH = str(HERE)

DOERFLES = "8001484"
UPSTREAM = ["8013008", "8004325", "8004064", "8005122", "8004633"]
UP_NAME = {"8013008": "sonneberg", "8004325": "neustadt", "8004064": "moenchroeden",
           "8005122": "roedentalmitte", "8004633": "roedental"}
CONFLICT_WINDOW = 30
VOR_WINDOW = 45        # Suchfenster fuer den vorausfahrenden Zug (wie im Training)

FEATURES = [
    "db_delay_now",
    "up_sonneberg", "up_neustadt", "up_moenchroeden", "up_roedentalmitte", "up_roedental",
    "n_upstream_known", "last_known_delay", "max_upstream_delay", "upstream_trend",
    "conflict_max_delay", "conflict_mean_delay", "n_conflict_trips",
    "ice_max_delay", "n_ice_nearby", "conflict_gap_min", "ice_gap_min",
    "vorgaenger_delay", "vorgaenger_luecke_min",
    "n_eng", "n_eng_vorrang", "gap_2nd",
    "hour", "minute_of_day", "dow", "month", "is_weekend",
]


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def parse(s):
    return datetime.fromisoformat(s) if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", help="Abfragezeitpunkt HH:MM (lokal), Default: jetzt")
    ap.add_argument("--date", help="Datum YYYY-MM-DD, Default: heute")
    args = ap.parse_args()

    day = args.date or datetime.now(TZ).strftime("%Y-%m-%d")
    if args.at:
        hh, mm = map(int, args.at.split(":"))
        T = datetime.strptime(day, "%Y-%m-%d").replace(hour=hh, minute=mm, tzinfo=TZ)
    else:
        T = datetime.now(TZ)

    plan = load_jsonl(f"{RE19_DIR}/data/{day}/plan.jsonl")
    changes = load_jsonl(f"{RE19_DIR}/data/{day}/changes.jsonl")
    if not plan:
        print(f"Keine Plandaten fuer {day} gefunden.")
        return

    # Planzeiten indexieren
    plan_by = {}
    meta = {}
    for p in plan:
        plan_by[(p["trip_id"], p["eva"], p["kind"])] = parse(p.get("planned"))
        meta[p["trip_id"]] = {"line": p.get("line"), "number": p.get("number"),
                              "category": p.get("category")}

    # Echtzeitstand je (trip, eva, kind), nur was bis T bekannt war
    known = {}
    for c in sorted(changes, key=lambda c: c["snapshot"]):
        if parse(c["snapshot"]) > T:
            continue
        if c.get("changed"):
            known[(c["trip_id"], c["eva"], c["kind"])] = parse(c["changed"])

    def delay_min(trip, eva, kind):
        pl = plan_by.get((trip, eva, kind))
        kn = known.get((trip, eva, kind))
        if pl is None or kn is None:
            return np.nan
        return (kn - pl).total_seconds() / 60.0

    # Kandidaten: RE19-Abfahrten in Doerfles, planmaessig nach T
    cands = []
    for (trip, eva, kind), pl in plan_by.items():
        if eva == DOERFLES and kind == "departure" and meta.get(trip, {}).get("line") == "RE19":
            if pl and pl >= T - timedelta(minutes=5):
                cands.append((pl, trip))
    if not cands:
        print(f"Kein RE19 ab Doerfles-Esbach nach {T:%H:%M} am {day} im Fahrplan gefunden.")
        return
    cands.sort()
    sched_dep, trip = cands[0]

    # --- Features bauen (identisch zum Training) ---
    row = {"db_delay_now": delay_min(trip, DOERFLES, "departure")}
    ups = []
    for eva in UPSTREAM:
        v = delay_min(trip, eva, "departure")
        if np.isnan(v):
            v = delay_min(trip, eva, "arrival")
        row["up_" + UP_NAME[eva]] = v
        ups.append(v)
    ups_arr = np.array(ups, dtype=float)
    row["n_upstream_known"] = int(np.sum(~np.isnan(ups_arr)))
    valid = ups_arr[~np.isnan(ups_arr)]
    row["last_known_delay"] = valid[-1] if len(valid) else np.nan
    row["max_upstream_delay"] = np.nanmax(ups_arr) if len(valid) else np.nan
    row["upstream_trend"] = (valid[-1] - valid[0]) if len(valid) else np.nan

    # Konfliktzuege RE28/RE29/ICE im Zeitfenster
    lo, hi = sched_dep - timedelta(minutes=CONFLICT_WINDOW), sched_dep + timedelta(minutes=CONFLICT_WINDOW)
    cdel, ntrips, ice_delays, ice_passages, passages = [], set(), [], [], []
    for (t2, eva2, kind2), pl2 in plan_by.items():
        if t2 == trip or not pl2 or not (lo <= pl2 <= hi):
            continue
        info = meta.get(t2, {})
        if info.get("line") not in ("RE28", "RE29") and info.get("category") != "ICE":
            continue
        ntrips.add(t2)
        d = delay_min(t2, eva2, kind2)
        if not np.isnan(d):
            cdel.append((t2, d))
            passage = pl2 + timedelta(minutes=d)
            passages.append(passage)
            if info.get("category") == "ICE":
                ice_delays.append(d)
                ice_passages.append(passage)
    if cdel:
        per_trip = pd.DataFrame(cdel, columns=["trip", "d"]).groupby("trip")["d"].last()
        row["conflict_max_delay"] = per_trip.max()
        row["conflict_mean_delay"] = per_trip.mean()
    else:
        row["conflict_max_delay"] = np.nan
        row["conflict_mean_delay"] = np.nan
    row["n_conflict_trips"] = len(ntrips)
    row["ice_max_delay"] = max(ice_delays) if ice_delays else np.nan
    row["n_ice_nearby"] = len({t2 for t2 in ntrips if meta.get(t2, {}).get("category") == "ICE"})
    row["conflict_gap_min"] = min(abs((p - sched_dep).total_seconds() / 60) for p in passages) if passages else np.nan
    row["ice_gap_min"] = min(abs((p - sched_dep).total_seconds() / 60) for p in ice_passages) if ice_passages else np.nan

    # Vorausfahrender/kreuzender Zug: letzte RE19/RE28/RE29-Abfahrt in Doerfles vor
    # meiner (Fenster VOR_WINDOW). Staerkstes Feature des Modells - das Signal kommt
    # vor allem von Gegenzuegen, die auf dem eingleisigen Abschnitt kreuzen muessen.
    vor = [(pl2, t2) for (t2, eva2, kind2), pl2 in plan_by.items()
           if eva2 == DOERFLES and kind2 == "departure" and t2 != trip and pl2
           and meta.get(t2, {}).get("line") in ("RE19", "RE28", "RE29")
           and sched_dep - timedelta(minutes=VOR_WINDOW) <= pl2 < sched_dep]
    if vor:
        pl_vor, t_vor = max(vor)
        row["vorgaenger_delay"] = delay_min(t_vor, DOERFLES, "departure")
        row["vorgaenger_luecke_min"] = (sched_dep - pl_vor).total_seconds() / 60
    else:
        row["vorgaenger_delay"] = np.nan
        row["vorgaenger_luecke_min"] = np.nan

    # Kaskade: Vorrangzuege im eigenen Belegungsfenster (identisch zu build_features.py)
    BELEG_VON, BELEG_BIS = -2.0, 8.0
    luecken, vorrang = [], []
    for (t2, e2, k2), pl2 in plan_by.items():
        if t2 == trip or not pl2:
            continue
        info2 = meta.get(t2, {})
        if info2.get("line") not in ("RE28", "RE29") and info2.get("category") != "ICE":
            continue
        d2 = delay_min(t2, e2, k2)
        if np.isnan(d2):
            continue
        g = ((pl2 + timedelta(minutes=d2)) - sched_dep).total_seconds() / 60
        luecken.append(g)
        vorrang.append(4 if info2.get("category") == "ICE" else (3 if info2.get("line") == "RE29" else 2))
    row["n_eng"] = sum(1 for g in luecken if BELEG_VON <= g <= BELEG_BIS)
    row["n_eng_vorrang"] = sum(1 for g, pr in zip(luecken, vorrang)
                               if BELEG_VON <= g <= BELEG_BIS and pr >= 3)
    row["gap_2nd"] = sorted(abs(g) for g in luecken)[1] if len(luecken) >= 2 else np.nan

    loc = sched_dep.astimezone(TZ)
    row["hour"] = loc.hour
    row["minute_of_day"] = loc.hour * 60 + loc.minute
    row["dow"] = loc.weekday()
    row["month"] = loc.month
    row["is_weekend"] = int(loc.weekday() >= 5)

    model = joblib.load(f"{SCRATCH}/model_werrabahn.joblib")
    X = pd.DataFrame([row])[FEATURES]
    pred_delay = float(model.predict(X)[0])
    pred_time = sched_dep + timedelta(minutes=pred_delay)

    db = row["db_delay_now"]
    print(f"\n  Abfrage um {T:%H:%M} Uhr am {day}")
    print(f"  Zug:            RE19 {meta.get(trip, {}).get('number', '?')} ab Doerfles-Esbach")
    print(f"  Planmaessig:    {sched_dep.astimezone(TZ):%H:%M}")
    print(f"  DB-Prognose:    {'(noch keine)' if np.isnan(db) else f'{(sched_dep + timedelta(minutes=db)).astimezone(TZ):%H:%M}  ({db:+.0f} min)'}")
    print(f"  MODELL:         {pred_time.astimezone(TZ):%H:%M}  ({pred_delay:+.1f} min)")
    print()
    print("  Was das Modell gerade sieht:")
    for eva in UPSTREAM:
        v = row["up_" + UP_NAME[eva]]
        print(f"    {UP_NAME[eva]:<16} {'-' if np.isnan(v) else f'{v:+.0f} min'}")
    cm = row["conflict_max_delay"]
    print(f"    Konfliktzuege    {row['n_conflict_trips']} Stueck, "
          f"max {'-' if np.isnan(cm) else f'{cm:+.0f} min'} verspaetet")
    print()


if __name__ == "__main__":
    main()
