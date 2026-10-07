"""Schritt 1: Das physikalische Regelmodell ueber die 2025er-Daten laufen lassen.

Bisher war das Regelmodell nie gegen die Realitaet geprueft - im Livebetrieb hat es in
271 Laeufen nie angeschlagen, weil in dem Zeitraum kein ICE ins Konfliktfenster fiel.
Dieses Skript baut die Produktionslogik aus re19watch.run_check() aus den historischen
Daten nach und wertet sie auf dem ganzen Jahr aus.

Nachgebildet wird bewusst der PRODUKTIONSPFAD, nicht eine verbesserte Variante:
  - Gegenzuege werden wie dort aus der Tafel von COBURG genommen (+/- Fenster) und mit
    EINEM Anker (Coburg) ueber die Fahrzeiten hochgerechnet
  - zweimal simuliert (Soll + Prognose), gezaehlt wird nur die Differenz
  - alles nur mit dem Wissensstand zum Abfragezeitpunkt T = Soll-Abfahrt - 15 min

Die Richtung und der Laufweg kommen aus stop_sequence/initial_stop_id, weil das Parquet
keinen Zielbahnhof-Namen enthaelt (die Produktion nutzt dafuer classify() auf dem Zieltext).

Aufruf: python replay_2025.py [--limit N]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import tomllib

import re19watch as rw
from simulate import FAST, REGIONAL, estimate, make_train

CFG = tomllib.loads((HERE.parent / "config.toml").read_text(encoding="utf-8"))
CORR = rw.build_corridor(CFG["corridor"])
CC = CFG["corridor"]
COBURG_P, JUNCTION, WB_END = CC["coburg_point"], CC["junction_point"], CC["werrabahn_end"]

EVA = {"coburg": 8001338, "coburg_nord": 8001334, "doerfles": 8001484,
       "roedental": 8004633, "roedental_mitte": 8005122, "moenchroeden": 8004064,
       "neustadt": 8004325, "sonneberg": 8013008}
NORD_VON_COBURG = [EVA["coburg_nord"], EVA["doerfles"], EVA["roedental_mitte"],
                   EVA["roedental"], EVA["moenchroeden"], EVA["neustadt"], EVA["sonneberg"]]
LEAD_MIN = 15          # Abfragezeitpunkt wie im Training/Betrieb
FENSTER_VOR = int(CFG["api"].get("window_before_min", 90))
FENSTER_LAENGE = FENSTER_VOR + int(CFG["api"].get("window_after_min", 30))

PRODUKT = {"ICE": "nationalExpress", "IC": "national", "EC": "national",
           "RE": "regionalExpress", "RB": "regional", "S": "suburban"}


def produkt(category: str | None) -> str:
    return PRODUKT.get((category or "").upper(), "regionalExpress")


def vorrang(line: str | None, prod: str) -> int:
    """Wie rw.priority(), nur ohne Event-Objekt."""
    ovr = {rw.norm_line(k): v for k, v in CFG.get("priority", {}).get("lines", {}).items()}
    return ovr.get(rw.norm_line(line or ""), rw.PRIORITY.get(prod, 2))


def lade() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_parquet(HERE / "data" / "werrabahn_marschbahn_2025.parquet")
    w = d[d["strecke"] == "werrabahn"].copy()
    w["delay_min"] = w["delay"] / 60.0
    # Erstbeobachtung je Halt = Soll-Fahrplan
    erst = (w.sort_values("update_timestamp")
            .drop_duplicates(["trip_id", "stop_id", "is_arrival"], keep="first"))
    return w, erst


def laufwege(erst: pd.DataFrame) -> dict:
    """Je Fahrt: Haltreihenfolge an den Korridorstationen + Produkt/Linie."""
    info = {}
    for trip, g in erst.groupby("trip_id"):
        seq = dict(zip(g["stop_id"], g["stop_sequence"]))
        r = g.iloc[0]
        info[trip] = {"seq": seq, "line": r["line"], "cat": r["category"],
                      "start": r["initial_stop_id"]}
    return info


def zweig(trip_info: dict) -> str | None:
    """Faehrt die Fahrt ab Coburg ueber die Werrabahn oder ueber die SFS-Einschleifung?

    Werrabahn: die Fahrt haelt an mindestens einer Station noerdlich von Coburg.
    SFS: Fernverkehr, der in Coburg haelt, aber an keiner Werrabahn-Station - der kann
    den Korridor nur ueber die Einschleifung verlassen bzw. erreichen.
    """
    seq = trip_info["seq"]
    if any(e in seq for e in NORD_VON_COBURG):
        return "werrabahn"
    if produkt(trip_info["cat"]) in ("nationalExpress", "national"):
        return "sfs"
    return None


def richtung_nord(trip_info: dict) -> bool | None:
    """True, wenn die Fahrt in Coburg Richtung Norden weiterfaehrt."""
    seq = trip_info["seq"]
    cob = seq.get(EVA["coburg"])
    nord = [seq[e] for e in NORD_VON_COBURG if e in seq]
    if cob is not None and nord:
        return min(nord) > cob
    # Fernverkehr ohne Werrabahn-Halt: aus der Herkunft schliessen
    # (Sueden -> faehrt nach Norden weiter)
    SUED = {8000261.0, 8000284.0, 8103000.0, 8010255.0, 8000025.0, 8000228.0}
    s = trip_info.get("start")
    if pd.notna(s):
        return float(s) in SUED
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="nur die ersten N Zielzuege (zum Testen)")
    ap.add_argument("--out", default=str(HERE / "data" / "replay_2025.parquet"))
    args = ap.parse_args()

    w, erst = lade()
    info = laufwege(erst)
    print(f"Fahrten mit Laufweg: {len(info)}")

    # Beobachtungen: je (Fahrt, Halt, Art) die Verspaetungsmeldungen mit Zeitstempel
    obs = (w[["trip_id", "stop_id", "is_arrival", "update_timestamp", "delay_min"]]
           .dropna(subset=["delay_min"]).sort_values("update_timestamp"))
    obs_idx = {k: v for k, v in obs.groupby(["trip_id", "stop_id", "is_arrival"])}

    def delay_bei(trip, stop, is_arr, T):
        g = obs_idx.get((trip, stop, is_arr))
        if g is None:
            return 0.0
        g = g[g["update_timestamp"] <= T]
        return float(g["delay_min"].iloc[-1]) if len(g) else 0.0

    # Zielzuege: RE19 in meiner Fahrtrichtung (Coburg/Sueden) ab Doerfles
    ziele = pd.read_parquet(HERE / "data" / "features_werrabahn.parquet")
    ziele = ziele[["trip_id", "sched_dep", "T", "db_delay_now", "y_delay_min"]].copy()
    sued = {t: (richtung_nord(i) is False) for t, i in info.items()}
    ziele = ziele[ziele["trip_id"].map(sued).fillna(False)]
    ziele = ziele.sort_values("sched_dep")
    if args.limit:
        ziele = ziele.head(args.limit)
    print(f"Zielzuege (RE19 Richtung Coburg) : {len(ziele)}")

    # Coburger Halte als Quelle der Gegenzuege, wie in der Produktion
    cob = erst[erst["stop_id"] == EVA["coburg"]][
        ["trip_id", "is_arrival", "time_schedule"]].dropna(subset=["time_schedule"])
    cob = cob.sort_values("time_schedule").reset_index(drop=True)
    # Series.searchsorted statt np.searchsorted: time_schedule ist datetime64[us],
    # ein Vergleich gegen Timestamp.value (Nanosekunden) laege um Faktor 1000 daneben
    # und wuerde immer ein leeres Fenster liefern.
    cob_zeit = cob["time_schedule"]

    rows = []
    for n, (_, z) in enumerate(ziele.iterrows(), 1):
        if n % 500 == 0:
            print(f"  ... {n}/{len(ziele)}")
        planned, T = z["sched_dep"], z["T"]
        db_delay = z["db_delay_now"] if pd.notna(z["db_delay_now"]) else 0.0
        db_when = planned + timedelta(minutes=float(db_delay))

        lo = planned - timedelta(minutes=FENSTER_VOR)
        hi = lo + timedelta(minutes=FENSTER_LAENGE)
        i0 = int(cob_zeit.searchsorted(lo, side="left"))
        i1 = int(cob_zeit.searchsorted(hi, side="left"))
        fenster = cob.iloc[i0:i1]

        andere_plan, andere_prog = [], []
        for _, ev in fenster.iterrows():
            if ev["trip_id"] == z["trip_id"]:
                continue
            ti = info.get(ev["trip_id"])
            if ti is None:
                continue
            zw = zweig(ti)
            nordwaerts = richtung_nord(ti)
            if zw is None or nordwaerts is None:
                continue
            is_arr = bool(ev["is_arrival"])
            # verlaesst Coburg nach Norden = Abfahrt eines nordwaerts fahrenden Zuges
            if is_arr == nordwaerts:
                continue            # Richtung passt nicht zu Ankunft/Abfahrt -> faehrt nach Sueden
            far = JUNCTION if zw == "sfs" else WB_END
            start, end = (COBURG_P, far) if nordwaerts else (far, COBURG_P)
            prod = produkt(ti["cat"])
            cls = FAST if prod in ("nationalExpress", "national") else REGIONAL
            pr = vorrang(ti["line"], prod)
            label = f"{ti['line'] or ti['cat']} {'nach' if nordwaerts else 'aus'} Norden"
            d = delay_bei(ev["trip_id"], EVA["coburg"], is_arr, T)
            andere_plan.append(make_train(CORR, label, pr, cls, start, end, COBURG_P,
                                          ev["time_schedule"], ev["trip_id"]))
            andere_prog.append(make_train(CORR, label, pr, cls, start, end, COBURG_P,
                                          ev["time_schedule"] + timedelta(minutes=d), ev["trip_id"]))

        mt = CFG["my_train"]
        me_plan = make_train(CORR, "RE 19", vorrang("RE 19", "regionalExpress"), REGIONAL,
                             mt["from_point"], mt["to_point"], mt["point"], planned, z["trip_id"])
        me_prog = make_train(CORR, "RE 19", vorrang("RE 19", "regionalExpress"), REGIONAL,
                             mt["from_point"], mt["to_point"], mt["point"], db_when, z["trip_id"])
        est = estimate(me_plan, andere_plan, me_prog, andere_prog, CORR,
                       mt["point"], planned, db_when)
        rows.append({
            "trip_id": z["trip_id"], "sched_dep": planned,
            "db_delay_now": z["db_delay_now"], "y_delay_min": z["y_delay_min"],
            "regel_delay": (est.estimate - planned).total_seconds() / 60,
            "rule_extra_min": est.extra_vs_db.total_seconds() / 60,
            "baseline_min": est.baseline_error.total_seconds() / 60,
            "n_andere": len(andere_prog),
            "holds": "; ".join(f"{h.at}:{h.wait.total_seconds()/60:.0f}min" for h in est.holds),
        })

    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"\ngespeichert: {args.out}  ({len(out)} Zeilen)")
    print(f"Faelle mit erkannter Blockade (rule_extra_min > 0): "
          f"{(out.rule_extra_min > 0).sum()} ({(out.rule_extra_min > 0).mean()*100:.1f} %)")


if __name__ == "__main__":
    main()
