"""Test an den schwersten Verspaetungsfaellen der Werrabahn 2025.

Jeder Testtag wird aus dem Training ENTFERNT (leave-one-day-out), sonst waere es
eine Pruefung auf bereits gesehenen Daten.

Unterscheidet zwei Arten von Faellen, weil nur die erste ueberhaupt vorhersagbar ist:
  - zum Abfragezeitpunkt T (= Abfahrt - 15 min) war ein Signal >= 5 min sichtbar
    (DB-Prognose, Oberlauf, Konfliktzug, ICE oder vorausfahrender Zug)
  - um T war nichts zu sehen; die Verspaetung entstand erst danach

ACHTUNG bei der Interpretation: die hier gezeigten Faelle sind die EXTREME, also eine
bewusst verzerrte Stichprobe. Zaehlt man "wer lag naeher", gewinnt das Modell die
Mehrheit; im mittleren Fehler verliert es, weil es schon sichtbare grosse DB-Werte zur
Mitte hin daempft. Ob eine Untergrenze an der DB-Prognose besser waere, wurde auf allen
Daten geprueft: 2025 minimal dafuer (1,67 statt 1,69), die eigenen 2026-Daten minimal
dagegen (1,84 statt 1,82). Beides im Rauschen und nur ~1 % der Faelle betroffen,
deshalb bleibt es bei der direkten ML-Prognose. Aussagekraeftige Zahlen stehen in
README.md, nicht hier.

Aufruf: python schwere_faelle.py
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import pathlib
ML = str(pathlib.Path(__file__).resolve().parent)
FEATURES = ["db_delay_now","up_sonneberg","up_neustadt","up_moenchroeden","up_roedentalmitte",
  "up_roedental","n_upstream_known","last_known_delay","max_upstream_delay","upstream_trend",
  "conflict_max_delay","conflict_mean_delay","n_conflict_trips","ice_max_delay","n_ice_nearby",
  "conflict_gap_min","ice_gap_min","vorgaenger_delay","vorgaenger_luecke_min",
  "n_eng","n_eng_vorrang","gap_2nd",
  "hour","minute_of_day","dow","month","is_weekend"]

f = pd.read_parquet(f"{ML}/data/features_werrabahn.parquet")
f["lokal"] = f["sched_dep"].dt.tz_convert("Europe/Berlin")
f["tag"] = f["lokal"].dt.date
f["signal"] = f[["db_delay_now","max_upstream_delay","conflict_max_delay",
                 "ice_max_delay","vorgaenger_delay"]].max(axis=1)

def trainiere(tr):
    loc = tr["sched_dep"].dt.tz_convert("Europe/Berlin")
    w = np.where((loc.dt.hour.between(6,8)) & (loc.dt.dayofweek<5), 5.0, 1.0)
    m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05,
        max_depth=6, min_samples_leaf=20, l2_regularization=1.0, random_state=42)
    m.fit(tr[FEATURES], tr["y_delay_min"], sample_weight=w)
    return m

faelle = f[(f.y_delay_min >= 20) & (f.signal >= 5)].nlargest(14, "y_delay_min")
morgen = f[(f.lokal.dt.strftime('%H:%M')=='07:20') & (f.y_delay_min >= 14)]
auswahl = pd.concat([faelle, morgen]).drop_duplicates(subset=["trip_id"])

print(f"\nSchwere Faelle (>= 20 min) mit Signal um T, plus alle 07:20-Faelle >= 14 min:")
print(f"{'Tag':<12}{'Wochentag':<10}{'Soll':>6}{'DB@T':>6}{'Modell':>8}{'IST':>6}  wer naeher")
print("-"*62)
tr_db, tr_ml, n = 0.0, 0.0, 0
for tag, grp in auswahl.groupby("tag"):
    m = trainiere(f[f.tag != tag])          # Testtag raus
    p = m.predict(grp[FEATURES])
    for (_, r), pred in zip(grp.iterrows(), p):
        db = r.db_delay_now
        e_ml = abs(pred - r.y_delay_min)
        e_db = abs(db - r.y_delay_min) if pd.notna(db) else np.nan
        if pd.notna(e_db):
            tr_db += e_db; tr_ml += e_ml; n += 1
            wer = "Modell" if e_ml < e_db else "DB"
        else:
            wer = "(DB schwieg)"
        print(f"{str(tag):<12}{r.lokal.strftime('%a'):<10}{r.lokal:%H:%M}"
              f"{(f'{db:+.0f}' if pd.notna(db) else '  -'):>6}{pred:>+8.1f}{r.y_delay_min:>+6.0f}  {wer}")
print("-"*62)
print(f"Mittlerer absoluter Fehler ueber {n} vergleichbare Faelle:"
      f"  Modell {tr_ml/n:.2f} min   DB {tr_db/n:.2f} min")
