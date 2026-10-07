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

# Empirische Kalibrierung aus 553 realen Faellen 2025 (verifizierte Ist-Werte):
# reale RE19-Verspaetung in Doerfles, aufgeschluesselt nach der tatsaechlichen
# Verspaetung des ICE im Konfliktfenster.
#
# WICHTIG - warum hier eine SPANNE und kein Einzelwert steht:
# Die Streuung innerhalb jeder Gruppe ist groesser als der Unterschied zwischen
# den Gruppen. Im kritischen Fenster (ICE +8..12) reicht die Realitaet von 0 bis
# 43 Minuten bei einem Median von 7. Eine Punktvorhersage waere Scheinpraezision:
# out-of-sample geprueft schlaegt die Tabelle eine konstante Vorhersage nur um
# 6 % (MAE 3,00 vs 3,20 min). Verlaesslich ist die ERKENNUNG des Risikofensters,
# nicht die Bezifferung.
KALIBRIERUNG = [   # (ICE-Verspaetung bis, p25, median, p75, p90, n)
    (3,   1.0, 1.0, 2.0,  5.0, 304),
    (5,   2.0, 3.0, 5.0,  7.0,  81),
    (8,   2.0, 4.0, 6.0,  7.0,  31),
    (12,  5.0, 7.0, 9.0, 12.0,  37),
    (20,  1.0, 1.0, 8.0, 10.0,  36),
    (999, 1.0, 3.0, 6.0,  8.0,  64),
]


def erwarteter_zuschlag(ice_delay_min):
    """Empirisch beobachtete RE19-Verspaetung: (p25, median, p75, p90, n)."""
    for grenze, p25, med, p75, p90, n in KALIBRIERUNG:
        if ice_delay_min <= grenze:
            return p25, med, p75, p90, n
    return 0.0, 0.0, 0.0, 0.0, 0


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
    """Fuehrt beide Modelle zusammen.

    Rueckgabe: (erwartet, untergrenze, obergrenze, risiko, begruendung)
    Die Grenzen sind p25/p90 aus den historischen Faellen - also keine
    Konfidenzintervalle im statistischen Sinn, sondern beobachtete Spannweiten.
    """
    eigen = eigene_verspaetung or 0.0
    if not konflikt_erkannt:
        return ml_prognose, ml_prognose, ml_prognose, "normal", "nur ML (kein Konflikt erkannt)"

    p25, med, p75, p90, n = erwarteter_zuschlag(ice_delay)
    erwartet = max(ml_prognose, eigen + med)
    unten = max(ml_prognose * 0.6, eigen + p25)
    oben = max(ml_prognose, eigen + p90)
    risiko = "ERHOEHT" if med >= 5 else "leicht erhoeht"
    quelle = "Regelmodell+Empirie" if eigen + med > ml_prognose else "ML"
    return erwartet, unten, oben, risiko, f"{quelle}, Konflikt erkannt (n={n} Vergleichsfaelle)"


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
        "vorgaenger_delay", "vorgaenger_luecke_min",
        "hour", "minute_of_day", "dow", "month", "is_weekend",
    ]
    sched = TAG.replace(hour=7, minute=20)

    print("Szenario: RE 19 ab Doerfles-Esbach 07:20, selbst puenktlich.")
    print("          ICE nach Berlin (Coburg ab 07:12 planmaessig) zunehmend verspaetet,")
    print("          RE 29 nach Erfurt (Coburg ab 07:27) mit +3 min.\n")
    print(f"{'ICE':>5} | {'Konflikt?':>10} | {'ML':>6} | {'KOMBINIERT (Spanne)':>22} | {'Risiko':>14} | real 2025")
    print("-" * 92)
    for ice in [0, 3, 5, 8, 10, 12, 15, 20, 30]:
        zuege = [(f"ICE +{ice}", 4, FAST, TAG.replace(hour=7, minute=12) + timedelta(minutes=ice)),
                 ("RE29 +3", 3, REGIONAL, TAG.replace(hour=7, minute=27) + timedelta(minutes=3))]
        konflikt, roh, verurs = regelcheck(c, sched, zuege)

        row = {f: 0.0 for f in FEATURES}
        # vorgaenger_luecke_min=0 waere unrealistisch (der Vorgaenger faehrt
        # planmaessig 41 min vor mir); vorgaenger_delay bleibt 0 = puenktlich.
        row.update(vorgaenger_luecke_min=41.0,
                   n_upstream_known=5, n_conflict_trips=2, n_ice_nearby=1,
                   hour=7, minute_of_day=440, dow=0, month=10, is_weekend=0,
                   conflict_max_delay=float(max(ice, 3)), conflict_mean_delay=float((ice + 3) / 2),
                   ice_max_delay=float(ice),
                   ice_gap_min=abs((TAG.replace(hour=7, minute=12) + timedelta(minutes=ice) - sched)
                                   .total_seconds() / 60),
                   conflict_gap_min=abs((TAG.replace(hour=7, minute=12) + timedelta(minutes=ice) - sched)
                                        .total_seconds() / 60))
        ml = float(model.predict(pd.DataFrame([row])[FEATURES])[0])
        erw, unten, oben, risiko, grund = kombiniere(ml, 0.0, konflikt, ice)
        real = erwarteter_zuschlag(ice)[1]
        spanne = f"{erw:+.1f}  ({unten:+.0f} bis {oben:+.0f})"
        print(f"{ice:>5} | {'JA' if konflikt else 'nein':>10} | {ml:+5.1f} | {spanne:>22} | "
              f"{risiko:>14} | {real:+.0f} min")

    print("\n'real 2025' = Median der tatsaechlichen RE19-Verspaetung in genau dieser Lage,")
    print("aus 553 beobachteten Faellen (verifizierte Ist-Werte).")
    print("Die Spanne ist p25-p90 derselben Faelle - KEIN statistisches Konfidenzintervall,")
    print("sondern die beobachtete Streuung. Im Fenster ICE +8..12 lagen real 0 bis 43 min.")


if __name__ == "__main__":
    demo()
