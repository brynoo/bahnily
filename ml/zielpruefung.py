"""Prueft die drei Ziele des Systems an den Daten, nicht an Behauptungen.

  1. Konfliktfaelle richtig treffen
  2. Im Alltag besser sein als der DB Navigator
  3. Nicht grundlos Alarm schlagen

Alles ueber fuenf gleitende Zweimonatsfenster: trainiert wird immer nur mit Daten VOR
dem jeweiligen Testfenster. Der Alarm wird mit der echten Produktionsschwelle aus
config.toml bewertet ([alarm] min_delay_min).

Aufruf: python zielpruefung.py
"""
from __future__ import annotations

import pathlib
import tomllib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = pathlib.Path(__file__).resolve().parent
CFG = tomllib.loads((HERE.parent / "config.toml").read_text(encoding="utf-8"))
SCHWELLE = int(CFG["alarm"].get("min_delay_min", 3))
RELEVANT = 5          # ab so viel Verspaetung ist ein Alarm fuer den Nutzer berechtigt

FEATURES = [
    "db_delay_now", "up_sonneberg", "up_neustadt", "up_moenchroeden", "up_roedentalmitte",
    "up_roedental", "n_upstream_known", "last_known_delay", "max_upstream_delay",
    "upstream_trend", "conflict_max_delay", "conflict_mean_delay", "n_conflict_trips",
    "ice_max_delay", "n_ice_nearby", "conflict_gap_min", "ice_gap_min",
    "vorgaenger_delay", "vorgaenger_luecke_min",
    "n_eng", "n_eng_vorrang", "gap_2nd",
    "hour", "minute_of_day", "dow", "month", "is_weekend",
]
FENSTER = ["2025-07-01", "2025-08-01", "2025-09-01", "2025-10-01", "2025-11-01"]


def _fit(tr: pd.DataFrame) -> HistGradientBoostingRegressor:
    loc = tr["sched_dep"].dt.tz_convert("Europe/Berlin")
    w = np.where((loc.dt.hour.between(6, 8)) & (loc.dt.dayofweek < 5), 5.0, 1.0)
    m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05,
                                      max_depth=6, min_samples_leaf=20, l2_regularization=1.0,
                                      random_state=42)
    m.fit(tr[FEATURES], tr["y_delay_min"], sample_weight=w)
    return m


def vorhersagen(f: pd.DataFrame) -> pd.DataFrame:
    teile = []
    for start in FENSTER:
        t0 = pd.Timestamp(start, tz="UTC")
        t1 = t0 + pd.DateOffset(months=2)
        tr = f[f["sched_dep"] < t0]
        te = f[(f["sched_dep"] >= t0) & (f["sched_dep"] < t1) & f["db_delay_now"].notna()].copy()
        if len(tr) < 500 or te.empty:
            continue
        te["pred"] = _fit(tr).predict(te[FEATURES])
        teile.append(te)
    d = pd.concat(teile)
    d["lokal"] = d["sched_dep"].dt.tz_convert("Europe/Berlin")
    return d


def ziel1(d: pd.DataFrame) -> None:
    print("=" * 72)
    print("ZIEL 1  Konfliktfaelle richtig treffen")
    print("=" * 72)
    konf = (d["ice_gap_min"] <= 15) & (d["ice_max_delay"] >= 10)
    vorg = d["vorgaenger_delay"] >= 8
    print(f"{'Lage':<40}{'n':>6}{'Modell':>9}{'DB':>8}")
    print("-" * 63)
    for nm, m in [("ICE nah (<=15 min) und >=10 min spaet", konf),
                  ("vorausfahrender Zug >= 8 min spaet", vorg),
                  ("eines von beiden", konf | vorg),
                  ("davon mit Ist >= 5 min", (konf | vorg) & (d["y_delay_min"] >= 5))]:
        m = m.to_numpy()
        if m.sum() < 5:
            continue
        print(f"{nm:<40}{m.sum():>6}"
              f"{np.abs(d['pred'][m] - d['y_delay_min'][m]).mean():>9.2f}"
              f"{np.abs(d['db_delay_now'][m] - d['y_delay_min'][m]).mean():>8.2f}")


def ziel2(d: pd.DataFrame) -> None:
    print()
    print("=" * 72)
    print("ZIEL 2  Im Alltag besser als der DB Navigator")
    print("=" * 72)
    s7 = (d["lokal"].dt.strftime("%H:%M") == "07:20").to_numpy()
    mo = ((d["lokal"].dt.hour.between(6, 8)) & (d["lokal"].dt.dayofweek < 5)).to_numpy()
    y, p, db = d["y_delay_min"].to_numpy(), d["pred"].to_numpy(), d["db_delay_now"].to_numpy()
    print(f"{'Slice':<40}{'n':>6}{'Modell':>9}{'DB':>8}{'besser':>9}")
    print("-" * 72)
    for nm, m in [("alle Abfahrten", np.ones(len(y), bool)),
                  (">>> der 07:20-Zug", s7),
                  ("morgens 6-8 h, Mo-Fr", mo),
                  ("Ist-Verspaetung >= 5 min", y >= 5),
                  ("Ist-Verspaetung < 5 min", y < 5)]:
        if m.sum() < 5:
            continue
        b = (np.abs(p[m] - y[m]) < np.abs(db[m] - y[m])).mean() * 100
        print(f"{nm:<40}{m.sum():>6}{np.abs(p[m]-y[m]).mean():>9.2f}"
              f"{np.abs(db[m]-y[m]).mean():>8.2f}{b:>8.0f}%")


def ziel3(d: pd.DataFrame) -> None:
    print()
    print("=" * 72)
    print(f"ZIEL 3  Nicht grundlos Alarm schlagen  (Schwelle {SCHWELLE} min, "
          f"berechtigt ab {RELEVANT} min)")
    print("=" * 72)
    y = d["y_delay_min"].to_numpy()
    echt = y >= RELEVANT
    print(f"{'Alarm ausgeloest durch':<34}{'Alarme':>8}{'Quote':>8}"
          f"{'davon zu Recht':>16}{'erkannt von allen':>19}")
    print("-" * 85)
    for nm, sig in [("Modell (so laeuft es)", d["pred"].to_numpy()),
                    ("DB-Prognose", d["db_delay_now"].to_numpy())]:
        a = sig >= SCHWELLE
        if not a.sum():
            continue
        print(f"{nm:<34}{a.sum():>8}{a.mean()*100:>7.1f}%"
              f"{(echt[a]).mean()*100:>15.1f}%{(a[echt]).mean()*100:>18.1f}%")
    # Zum Vergleich: was das entfernte Regelmodell-Veto angerichtet haette
    rp = HERE / "data" / "replay_2025.parquet"
    if rp.exists():
        r = pd.read_parquet(rp)[["trip_id", "rule_extra_min", "regel_delay"]]
        m = d.merge(r, on="trip_id", how="inner")
        if len(m) > 100:
            veto = np.maximum(m["pred"], m["regel_delay"]).to_numpy()
            e2 = (m["y_delay_min"] >= RELEVANT).to_numpy()
            a = veto >= SCHWELLE
            print(f"{'(frueheres Regelmodell-Veto)':<34}{a.sum():>8}{a.mean()*100:>7.1f}%"
                  f"{(e2[a]).mean()*100:>15.1f}%{(a[e2]).mean()*100:>18.1f}%")
    print("\n'davon zu Recht' = Anteil der Alarme, bei denen wirklich >= 5 min kamen.")
    print("'erkannt von allen' = Anteil der echten Verspaetungen, die einen Alarm ausloesten.")


def main() -> None:
    f = pd.read_parquet(HERE / "data" / "features_werrabahn.parquet")
    d = vorhersagen(f)
    print(f"Bewertet auf {len(d)} Abfahrten aus fuenf gleitenden Fenstern "
          f"({d['lokal'].min():%d.%m.%Y} - {d['lokal'].max():%d.%m.%Y})\n")
    ziel1(d)
    ziel2(d)
    ziel3(d)


if __name__ == "__main__":
    main()
