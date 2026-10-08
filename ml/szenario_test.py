"""
Fiktives Szenario testen: verspaeteter ICE nach Berlin blockiert den eingleisigen
Abschnitt, waehrend der RE 19 von Roedental Richtung Doerfles-Esbach unterwegs ist.
Zusaetzlich faehrt der RE 29 (Vorrang vor RE 19) ueber dieselbe Einschleifung.

Verglichen werden die beiden Modelle, die das Projekt hat:
  A) simulate_fixed.py  - regelbasiert, rechnet die Gleisbelegung physikalisch aus
  B) model_werrabahn.joblib - gelernt aus den 2025er-Daten

Kein Netzwerkzugriff, keine echten Daten - rein konstruierte Faelle.
"""
import pathlib
import sys
import tomllib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "rule_based"))
from simulate_fixed import FAST, REGIONAL, Corridor, make_train, resolve  # noqa: E402

TZ = ZoneInfo("Europe/Berlin")
TAG = datetime(2026, 10, 5, tzinfo=TZ)          # fiktiver Montag

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


def build_corridor():
    with open(HERE.parent / "config.toml", "rb") as f:
        cc = tomllib.load(f)["corridor"]
    run, double = {}, set()
    pts = cc["points"]
    for s in cc["segment"]:
        a, b = (s["from"], s["to"]) if pts.index(s["from"]) < pts.index(s["to"]) else (s["to"], s["from"])
        run[(a, b)] = {REGIONAL: float(s["regional"]), FAST: float(s["fast"])}
        if s.get("double_track"):
            double.add((a, b))
    return Corridor(points=pts, run_min=run, crossing_points=set(cc["crossing_points"]),
                    double_track=double,
                    opposing_buffer_min=float(cc.get("opposing_buffer_min", 1.0)),
                    following_headway_min=float(cc.get("following_headway_min", 2.0)))


def t(hh, mm):
    return TAG.replace(hour=hh, minute=mm)


def regelbasiert(c, ice_delay, re29_delay):
    """Wie lange muss der RE 19 laut Gleisbelegungssimulation warten?"""
    # Mein Zug: RE 19, Roedental -> Coburg, planmaessig ab Doerfles-Esbach 07:20
    me = make_train(c, "RE19", 2, REGIONAL, "Rödental", "Coburg", "Dörfles-Esbach", t(7, 20))
    others = [
        # ICE nach Berlin: Coburg ab 07:12 planmaessig, ueber die Einschleifung nach Norden
        make_train(c, f"ICE nach Berlin (+{ice_delay})", 4, FAST,
                   "Coburg", "Abzweig SFS", "Coburg", t(7, 12) + timedelta(minutes=ice_delay)),
        # RE 29 nach Erfurt: Coburg ab 07:27, gleiche Einschleifung, Vorrang vor RE 19
        make_train(c, f"RE29 nach Erfurt (+{re29_delay})", 3, REGIONAL,
                   "Coburg", "Abzweig SFS", "Coburg", t(7, 27) + timedelta(minutes=re29_delay)),
    ]
    r = resolve(me, others, c)
    wartezeit = sum((h.wait.total_seconds() / 60 for h in r.holds), 0.0)
    gruende = [f"{cf.other.label} ({cf.kind})" for h in r.holds for cf in h.conflicts]
    ab = r.times["Dörfles-Esbach"]
    return wartezeit, ab, gruende


def gelernt(model, ice_delay, re29_delay, vorgaenger=0.0):
    """Was sagt das ML-Modell bei gleicher Lage?"""
    row = {
        "db_delay_now": 0.0,            # DB zeigt den RE 19 noch puenktlich
        "up_sonneberg": 0.0, "up_neustadt": 0.0, "up_moenchroeden": 0.0,
        "up_roedentalmitte": 0.0, "up_roedental": 0.0,
        "n_upstream_known": 5, "last_known_delay": 0.0,
        "max_upstream_delay": 0.0, "upstream_trend": 0.0,
        "conflict_max_delay": float(max(ice_delay, re29_delay)),
        "conflict_mean_delay": float((ice_delay + re29_delay) / 2),
        "n_conflict_trips": 2,
        "ice_max_delay": float(ice_delay), "n_ice_nearby": 1,
        # Abstand der Ist-Durchfahrt des Konfliktzugs zu meiner Soll-Abfahrt:
        # ICE faehrt planmaessig 07:12 in Coburg ab, meine Abfahrt ist 07:20.
        "conflict_gap_min": abs(-8 + ice_delay), "ice_gap_min": abs(-8 + ice_delay),
        # Vorausfahrender/kreuzender Zug: der RE19 der Gegenrichtung, planmaessig
        # 41 min vor meiner Abfahrt in Doerfles.
        "vorgaenger_delay": float(vorgaenger), "vorgaenger_luecke_min": 41.0,
        # Kaskade: Abstand der Ist-Durchfahrt zu meiner Soll-Abfahrt. ICE planmaessig
        # Coburg ab 07:12 -> Luecke = -8 + Verspaetung; RE 29 ab 07:27 (+3) -> +10,
        # liegt also ausserhalb des Belegungsfensters -2..+8.
        "n_eng": int(-2 <= -8 + ice_delay <= 8) + int(-2 <= 10 <= 8),
        "n_eng_vorrang": int(-2 <= -8 + ice_delay <= 8) + int(-2 <= 10 <= 8),
        "gap_2nd": max(abs(-8 + ice_delay), 10.0),
        "hour": 7, "minute_of_day": 7 * 60 + 20, "dow": 0, "month": 10, "is_weekend": 0,
    }
    return float(model.predict(pd.DataFrame([row])[FEATURES])[0])


def main():
    c = build_corridor()
    model = joblib.load(HERE / "model_werrabahn.joblib")

    print("Szenario: RE 19 planmaessig ab Doerfles-Esbach 07:20 Uhr, selbst puenktlich.")
    print("          ICE nach Berlin (Coburg ab 07:12) wird zunehmend verspaetet.")
    print("          RE 29 nach Erfurt (Coburg ab 07:27) faehrt mit +3 min.\n")
    print(f"{'ICE +min':>9} | {'Regelmodell':>28} | {'ML Vorg. 0':>12} | {'ML Vorg.+10':>12}")
    print(f"{'':>9} | {'Wartezeit  ->  Abfahrt':>28} | {'Prognose':>12} | {'Prognose':>12}")
    print("-" * 72)
    for ice_delay in [0, 3, 5, 8, 10, 12, 15, 20, 30]:
        wart, ab, gruende = regelbasiert(c, ice_delay, 3)
        rule = f"{wart:5.1f} min  ->  {ab.astimezone(TZ):%H:%M}"
        print(f"{ice_delay:>9} | {rule:>28} | {gelernt(model, ice_delay, 3, 0):+10.1f} min"
              f" | {gelernt(model, ice_delay, 3, 10):+10.1f} min")

    print("\n--- Detail: ICE +15 min, wer blockiert wen? ---")
    wart, ab, gruende = regelbasiert(c, 15, 3)
    print(f"Wartezeit {wart:.1f} min, Abfahrt Doerfles-Esbach {ab.astimezone(TZ):%H:%M}")
    for g in dict.fromkeys(gruende):
        print("  Konflikt mit:", g)

    print("\n--- Gegenprobe: nur RE 29 verspaetet, ICE puenktlich ---")
    print(f"{'RE29 +min':>9} | {'Regelmodell Wartezeit':>22} | {'ML-Modell':>12}")
    for d in [0, 5, 10, 20]:
        wart, ab, _ = regelbasiert(c, 0, d)
        ml = gelernt(model, 0, d)
        print(f"{d:>9} | {wart:>17.1f} min | {ml:+10.1f} min")


if __name__ == "__main__":
    main()
