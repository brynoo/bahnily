"""
Kombinierte Vorhersage: Regelmodell + gelerntes Modell.

Begruendung der Aufteilung (empirisch hergeleitet, siehe README):
  - Das REGELMODELL erkennt, OB ein Vorrangzug zeitlich in mein Fenster faellt.
    Das Konfliktfenster (ICE +8 bis +12 min) wurde an 553 realen Faellen aus 2025
    bestaetigt: dort springt die reale RE19-Verspaetung von 1 auf 7 Minuten Median.
  - Das Regelmodell schaetzt die HOEHE aber massiv zu hoch (18 min statt real ~7).
    Deshalb wird seine Minutenangabe verworfen und durch den empirischen Median
    aus den 2025er-Daten ersetzt.
  - Das ML-MODELL kann keine strukturellen Konflikte sehen (flache Antwort im
    Szenariotest), ist aber gut darin, eine BEREITS BESTEHENDE Verspaetung
    fortzuschreiben (bei Ist >= 5 min in 85 % der Faelle besser als die DB).

Ergebnis = max(ML-Prognose, eigene Verspaetung + erwarteter Konfliktzuschlag).
Das max() ist bewusst konservativ: wir wollen eher zu frueh warnen als zu spaet.
"""
import pathlib
import sys
import tomllib
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "rule_based"))
from simulate_fixed import FAST, REGIONAL, Corridor, make_train, resolve  # noqa: E402

# Empirische Kalibrierung: realer RE19-Verspaetungsmedian in Doerfles, aufgeschluesselt
# nach der tatsaechlichen Verspaetung des ICE im Konfliktfenster.
# Basis: 553 Faelle aus 2025 mit verifiziertem Ist-Wert. Nachrechnen: siehe README.
KALIBRIERUNG = [      # (ICE-Verspaetung bis, erwartete RE19-Verspaetung in min, n)
    (3,   1.0,  304),
    (5,   3.0,   81),
    (8,   4.0,   31),
    (12,  7.0,   37),
    (20,  1.0,   36),
    (999, 3.0,   64),
]


def erwarteter_zuschlag(ice_delay_min):
    """Empirisch beobachtete RE19-Verspaetung bei dieser ICE-Verspaetung."""
    for grenze, minuten, n in KALIBRIERUNG:
        if ice_delay_min <= grenze:
            return minuten, n
    return 0.0, 0


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


def regelcheck(corridor, sched_dep, konfliktzuege):
    """Erkennt das Regelmodell einen Konflikt? Gibt (ja/nein, Liste der Verursacher) zurueck.

    konfliktzuege: Liste von (label, prioritaet, klasse, coburg_abfahrt_ist)
    """
    me = make_train(corridor, "RE19", 2, REGIONAL, "Rödental", "Coburg", "Dörfles-Esbach", sched_dep)
    others = [make_train(corridor, lbl, prio, cls, "Coburg", "Abzweig SFS", "Coburg", ab)
              for lbl, prio, cls, ab in konfliktzuege]
    r = resolve(me, others, corridor)
    roh = sum(h.wait.total_seconds() / 60 for h in r.holds)
    verursacher = list(dict.fromkeys(cf.other.label for h in r.holds for cf in h.conflicts))
    return roh > 0.5, roh, verursacher


def kombiniere(ml_prognose, eigene_verspaetung, konflikt_erkannt, ice_delay):
    """Fuehrt beide Modelle zusammen."""
    if not konflikt_erkannt:
        return ml_prognose, "nur ML (kein Konflikt erkannt)", None
    zuschlag, n = erwarteter_zuschlag(ice_delay)
    regel_basiert = (eigene_verspaetung or 0.0) + zuschlag
    if regel_basiert > ml_prognose:
        return regel_basiert, f"Regelmodell (Konflikt, +{zuschlag:.0f} min aus n={n} Faellen)", zuschlag
    return ml_prognose, f"ML (hoeher als Konfliktzuschlag +{zuschlag:.0f})", zuschlag


def demo():
    """Zeigt die Pipeline am Szenario aus der Diskussion."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Europe/Berlin")
    TAG = datetime(2026, 10, 5, tzinfo=TZ)
    c = build_corridor()
    model = joblib.load(HERE / "model_werrabahn.joblib")
    FEATURES = [
        "db_delay_now", "up_sonneberg", "up_neustadt", "up_moenchroeden",
        "up_roedentalmitte", "up_roedental", "n_upstream_known", "last_known_delay",
        "max_upstream_delay", "upstream_trend", "conflict_max_delay", "conflict_mean_delay",
        "n_conflict_trips", "ice_max_delay", "n_ice_nearby", "conflict_gap_min", "ice_gap_min",
        "hour", "minute_of_day", "dow", "month", "is_weekend",
    ]
    sched = TAG.replace(hour=7, minute=20)

    print("Szenario: RE 19 ab Doerfles-Esbach 07:20, selbst puenktlich.")
    print("          ICE nach Berlin (Coburg ab 07:12 planmaessig) zunehmend verspaetet,")
    print("          RE 29 nach Erfurt (Coburg ab 07:27) mit +3 min.\n")
    print(f"{'ICE':>5} | {'Regel':>18} | {'ML':>7} | {'KOMBINIERT':>10} | {'real 2025':>9} | Entscheidung")
    print("-" * 94)
    for ice in [0, 3, 5, 8, 10, 12, 15, 20, 30]:
        zuege = [(f"ICE +{ice}", 4, FAST, TAG.replace(hour=7, minute=12) + timedelta(minutes=ice)),
                 ("RE29 +3", 3, REGIONAL, TAG.replace(hour=7, minute=27) + timedelta(minutes=3))]
        konflikt, roh, verurs = regelcheck(c, sched, zuege)

        row = {f: 0.0 for f in FEATURES}
        row.update(n_upstream_known=5, n_conflict_trips=2, n_ice_nearby=1,
                   hour=7, minute_of_day=440, dow=0, month=10, is_weekend=0,
                   conflict_max_delay=float(max(ice, 3)), conflict_mean_delay=float((ice + 3) / 2),
                   ice_max_delay=float(ice),
                   ice_gap_min=abs((TAG.replace(hour=7, minute=12) + timedelta(minutes=ice) - sched)
                                   .total_seconds() / 60),
                   conflict_gap_min=abs((TAG.replace(hour=7, minute=12) + timedelta(minutes=ice) - sched)
                                        .total_seconds() / 60))
        ml = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
        komb, grund, _ = kombiniere(ml, 0.0, konflikt, ice)
        real = erwarteter_zuschlag(ice)[0]
        regel_txt = f"{'JA' if konflikt else 'nein':>4} ({roh:4.1f} min)"
        print(f"{ice:>5} | {regel_txt:>18} | {ml:+6.1f} | {komb:+9.1f} | {real:>8.0f} | {grund}")

    print("\n'real 2025' = Median der tatsaechlichen RE19-Verspaetung in genau dieser Lage,")
    print("aus 553 beobachteten Faellen. Das ist die Messlatte, nicht eine Modellausgabe.")


if __name__ == "__main__":
    demo()
