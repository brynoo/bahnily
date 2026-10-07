"""
Feature-Tabelle fuer das Werrabahn-Modell (Sonneberg - Coburg, RE19/RE28/RE29).

Zielgroesse (y): tatsaechliche Abfahrtsverspaetung des RE19 in Doerfles-Esbach (Minuten).
Vorhersagezeitpunkt T: LEAD_MIN Minuten vor der planmaessigen Abfahrt (Default 15 ->
bei planmaessig 07:20 also die Abfrage um 07:05).

Alle Features sind RELATIV (Verspaetungen in Minuten, Zaehler, Zeitabstaende) und
enthalten bewusst KEINE absoluten Zugnummern oder Soll-Uhrzeiten als Identitaet -
dadurch bleibt das Modell ueber einen Fahrplanwechsel hinweg gueltig.

Liest nur die gefilterte Parquet-Datei im Scratchpad. Aendert nichts am Original.
"""
import numpy as np
import pandas as pd

import pathlib
HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"


SRC = str(DATA / "werrabahn_marschbahn_2025.parquet")
OUT = str(DATA / "features_werrabahn.parquet")

LEAD_MIN = 15          # Abfrage so viele Minuten vor planmaessiger Abfahrt
CONFLICT_WINDOW = 30   # +/- Minuten um meine Soll-Abfahrt fuer Konfliktzuege
# Kandidaten werden deutlich weiter gefasst als das Konfliktfenster, weil ein stark
# verspaeteter Zug erst durch seine Verspaetung in mein Fenster rutscht. Beispiel aus
# den Daten: ICE 1604, Soll-Abfahrt Coburg 06:43, +37 min -> real 07:20, also genau im
# Weg des 07:20-Zuges, obwohl seine Soll-Zeit 37 min vor meiner liegt. Mit einem
# Soll-Filter von +/-30 min war dieser Zug - und damit alle drei 07:20-Konfliktfaelle
# des Jahres 2025 - unsichtbar. Muss zu window_before_min/window_after_min in
# config.toml passen, sonst sieht das Modell im Betrieb andere Zuege als beim Lernen.
KAND_VOR, KAND_NACH = 90, 30
VOR_WINDOW = 45        # so weit zurueck wird nach dem vorausfahrenden Zug gesucht

DOERFLES = 8001484
# Korridor Nord -> Sued; Upstream = alles noerdlich von Doerfles
UPSTREAM = [8013008, 8004325, 8004064, 8005122, 8004633]  # Sonneberg..Roedental
UP_NAME = {8013008: "sonneberg", 8004325: "neustadt", 8004064: "moenchroeden",
           8005122: "roedentalmitte", 8004633: "roedental"}


def main():
    df = pd.read_parquet(SRC)
    # ACHTUNG: ICE-Zuege haben als "line" eine Nummer (94, 1694, ...), nicht "ICE".
    # Sie muessen ueber category gefiltert werden, sonst fehlen sie komplett -
    # und gerade der ICE ueber die SFS-Einschleifung ist der wichtigste Vorrangzug.
    w = df[(df["strecke"] == "werrabahn")
           & ((df["line"].isin(["RE19", "RE28", "RE29"])) | (df["category"] == "ICE"))].copy()
    w["delay_min"] = w["delay"] / 60.0
    w = w[w["update_timestamp"].notna()].copy()
    w = w.sort_values("update_timestamp")
    print(f"Basis: {len(w)} Beobachtungen (RE19/RE28/RE29, Werrabahn)")

    # ---------------- Zielgroesse: finale Abfahrtsverspaetung RE19 in Doerfles ----------------
    tgt = w[(w["line"] == "RE19") & (w["stop_id"] == DOERFLES) & (~w["is_arrival"]) & (w["is_final"])]
    tgt = (tgt.sort_values(["trip_id", "update_timestamp"])
           .drop_duplicates("trip_id", keep="last")
           [["trip_id", "time_schedule", "delay_min", "is_cancelled", "update_timestamp", "time_real"]]
           .rename(columns={"delay_min": "y_delay_min", "time_schedule": "sched_dep",
                            "update_timestamp": "final_ts"})
           .copy())
    tgt = tgt[~tgt["is_cancelled"]].drop(columns=["is_cancelled"])
    tgt["T"] = tgt["sched_dep"] - pd.Timedelta(minutes=LEAD_MIN)
    n_roh = len(tgt)

    # --- DATENLECKAGE ENTFERNEN ---------------------------------------------
    # Problem: der Datensatz markiert den letzten empfangenen Echtzeitstand als
    # is_final und deklariert ihn als "tatsaechliche Zeit". In 59 % der Faelle
    # wurde dieser Stand aber ERFASST, BEVOR die angebliche Ist-Zeit eingetreten
    # war - es ist dann eine nie verifizierte Prognose, keine Beobachtung.
    # Kritisch wird das, wenn dieser finale Stand schon vor dem Vorhersage-
    # zeitpunkt T vorlag: dann IST die DB-Prognose die Zielgroesse (gemessen:
    # 13,5 % der Zeilen, dort stimmt db_delay_now zu 100 % exakt mit y ueberein).
    # Solche Zeilen sind wertlos: die DB hat per Definition null Fehler, was die
    # Baseline kuenstlich stark macht und den Modellvergleich verzerrt.
    leak = tgt["final_ts"] <= tgt["T"]
    tgt = tgt[~leak].copy()
    # Flag, ob der Zielwert nach Eintreten der Zeit bestaetigt wurde (echte Beobachtung)
    # Flag bleibt als SPALTE erhalten, wird aber NICHT zum Filtern benutzt.
    # Gemessen: Filtern auf verifizierte Zeilen verschlechtert das Modell deutlich
    # (07:20-Slot 2,53 -> 2,86 min MAE). Grund ist ein Selektionsbias - die DB
    # schickt bei verspaeteten Zuegen weiter Updates, puenktliche bekommen nie eine
    # Bestaetigung. Verifizierte Zeilen enthalten daher 22 % schwere Faelle statt
    # 13,6 %, und es gingen 58 % der Trainingsdaten verloren.
    tgt["ziel_verifiziert"] = (tgt["final_ts"] - tgt["time_real"]).dt.total_seconds() / 60 >= 1
    tgt = tgt.drop(columns=["final_ts", "time_real"])
    print(f"Zielereignisse roh: {n_roh}, nach Entfernen von {leak.sum()} geleakten Zeilen: {len(tgt)}")
    print(f"  davon Zielwert nachweislich beobachtet: {tgt['ziel_verifiziert'].sum()} "
          f"({tgt['ziel_verifiziert'].mean()*100:.1f} %) - nur als Information, kein Filter")

    # ---------------- Point-in-time: was war zum Zeitpunkt T bekannt? ----------------
    # (a) DB-eigene Prognose fuer genau diese Abfahrt, Stand T  -> Baseline
    obs_dep = w[(w["stop_id"] == DOERFLES) & (~w["is_arrival"])][
        ["trip_id", "update_timestamp", "delay_min"]
    ].sort_values("update_timestamp")
    base = pd.merge_asof(
        tgt[["trip_id", "T"]].sort_values("T"), obs_dep,
        left_on="T", right_on="update_timestamp", by="trip_id", direction="backward",
    ).rename(columns={"delay_min": "db_delay_now"})[["trip_id", "db_delay_now"]]
    tgt = tgt.merge(base, on="trip_id", how="left")

    # (b) Verspaetung an den Oberlauf-Stationen, Stand T
    for eva in UPSTREAM:
        obs = w[(w["stop_id"] == eva) & (~w["is_arrival"])][
            ["trip_id", "update_timestamp", "delay_min"]
        ].sort_values("update_timestamp")
        m = pd.merge_asof(
            tgt[["trip_id", "T"]].sort_values("T"), obs,
            left_on="T", right_on="update_timestamp", by="trip_id", direction="backward",
        )
        col = "up_" + UP_NAME[eva]
        tgt = tgt.merge(m[["trip_id", "delay_min"]].rename(columns={"delay_min": col}),
                        on="trip_id", how="left")

    up_cols = ["up_" + UP_NAME[e] for e in UPSTREAM]
    tgt["n_upstream_known"] = tgt[up_cols].notna().sum(axis=1)
    tgt["last_known_delay"] = tgt[up_cols].ffill(axis=1).iloc[:, -1]
    tgt["max_upstream_delay"] = tgt[up_cols].max(axis=1)
    # Trend: wie stark hat sich die Verspaetung im Oberlauf aufgebaut?
    tgt["upstream_trend"] = tgt[up_cols].ffill(axis=1).iloc[:, -1] - tgt[up_cols].bfill(axis=1).iloc[:, 0]

    # ---------------- Konfliktzuege (RE28 Gegenrichtung, RE29 gemeinsamer Abschnitt) ----------------
    # Konfliktzuege: RE28 (Gegenrichtung) + RE29 und ICE (beide ueber die
    # SFS-Einschleifung, teilen sich Coburg - Herzogsweg mit dem RE19).
    # ICE wird getrennt gefuehrt, weil er die hoechste Prioritaet hat.
    conf_src = w[(w["line"].isin(["RE28", "RE29"])) | (w["category"] == "ICE")][
        ["trip_id", "line", "category", "stop_id", "time_schedule", "update_timestamp", "delay_min"]
    ].copy()
    conf_src["date"] = conf_src["time_schedule"].dt.date
    tgt["date"] = tgt["sched_dep"].dt.date

    rows = []
    for d, g in tgt.groupby("date"):
        cday = conf_src[conf_src["date"] == d]
        if cday.empty:
            for _, r in g.iterrows():
                rows.append((r["trip_id"], np.nan, np.nan, 0,
                             np.nan, 0, np.nan, np.nan))
            continue
        for _, r in g.iterrows():
            lo = r["sched_dep"] - pd.Timedelta(minutes=KAND_VOR)
            hi = r["sched_dep"] + pd.Timedelta(minutes=KAND_NACH)
            weit = cday[(cday["time_schedule"] >= lo) & (cday["time_schedule"] <= hi)]
            kn = weit[weit["update_timestamp"] <= r["T"]]
            # Auswahl nach VORAUSSICHTLICHER Durchfahrt, nicht nach Soll-Zeit: genau so
            # faellt ein stark verspaeteter Zug in mein Fenster. Fuer Zuege ohne bekannte
            # Verspaetung gilt die Soll-Zeit.
            if len(kn):
                letzte = kn.sort_values("update_timestamp").groupby("trip_id").last()
                pass_t = letzte["time_schedule"] + pd.to_timedelta(letzte["delay_min"], unit="m")
                drin = pass_t[(pass_t >= r["sched_dep"] - pd.Timedelta(minutes=CONFLICT_WINDOW)) &
                              (pass_t <= r["sched_dep"] + pd.Timedelta(minutes=CONFLICT_WINDOW))].index
            else:
                drin = pd.Index([])
            ohne = weit[~weit["trip_id"].isin(kn["trip_id"])]
            ohne = ohne[(ohne["time_schedule"] >= r["sched_dep"] - pd.Timedelta(minutes=CONFLICT_WINDOW)) &
                        (ohne["time_schedule"] <= r["sched_dep"] + pd.Timedelta(minutes=CONFLICT_WINDOW))]
            near = weit[weit["trip_id"].isin(drin) | weit["trip_id"].isin(ohne["trip_id"])]
            known = near[near["update_timestamp"] <= r["T"]]
            ice_known = known[known["category"] == "ICE"]
            n_ice = near[near["category"] == "ICE"]["trip_id"].nunique()
            if known.empty:
                rows.append((r["trip_id"], np.nan, np.nan, near["trip_id"].nunique(),
                             np.nan, n_ice, np.nan, np.nan))
            else:
                last = known.sort_values("update_timestamp").groupby("trip_id").last()
                per_trip = last["delay_min"]
                ice_max = (ice_known.sort_values("update_timestamp")
                           .groupby("trip_id")["delay_min"].last().max()) if not ice_known.empty else np.nan

                # ENTSCHEIDENDES FEATURE: nicht die Verspaetung des Konfliktzugs zaehlt,
                # sondern ob er dadurch ZEITLICH IN MEIN FENSTER faellt. Gemessen als
                # Abstand zwischen seiner voraussichtlichen Ist-Durchfahrt und meiner
                # Soll-Abfahrt. Nahe null = Begegnung auf dem eingleisigen Abschnitt.
                real_pass = last["time_schedule"] + pd.to_timedelta(last["delay_min"], unit="m")
                gap = (real_pass - r["sched_dep"]).dt.total_seconds() / 60
                gap_min_abs = gap.abs().min() if len(gap) else np.nan
                ice_last = ice_known.sort_values("update_timestamp").groupby("trip_id").last()
                if len(ice_last):
                    ice_pass = ice_last["time_schedule"] + pd.to_timedelta(ice_last["delay_min"], unit="m")
                    ice_gap = ((ice_pass - r["sched_dep"]).dt.total_seconds() / 60)
                    ice_gap_abs = ice_gap.abs().min()
                else:
                    ice_gap_abs = np.nan
                rows.append((r["trip_id"], per_trip.max(), per_trip.mean(),
                             near["trip_id"].nunique(), ice_max, n_ice, gap_min_abs, ice_gap_abs))
    conf = pd.DataFrame(rows, columns=["trip_id", "conflict_max_delay", "conflict_mean_delay",
                                       "n_conflict_trips", "ice_max_delay", "n_ice_nearby",
                                       "conflict_gap_min", "ice_gap_min"])
    tgt = tgt.merge(conf, on="trip_id", how="left")

    # ---------------- Vorausfahrender Zug (gleiche Richtung) ----------------
    # Auf dem eingleisigen Abschnitt blockiert der Zug VOR mir unmittelbar meine
    # Fahrstrasse - "Verspaetung eines vorausfahrenden Zuges" (Code 43) ist der
    # zweithaeufigste Verspaetungsgrund in den RE19-Daten. Nur Werrabahn-Linien:
    # agilis/RB/STB/Bus stehen zwar auf den Tafeln von Coburg und Sonneberg,
    # fahren den Abschnitt Coburg-Sonneberg aber nicht. Der ICE haelt in Doerfles
    # nicht (er kommt ueber die SFS-Einschleifung) und steckt in den Konfliktfeatures.
    # Gemessen ueber 5 gleitende Zweimonatsfenster: Gesamt-MAE 1,73 -> 1,65 min,
    # 07:20-Slot 2,40 -> 2,28; vorgaenger_delay ist danach das staerkste Feature
    # ueberhaupt (Permutation Importance 0,24 gegenueber 0,13 fuer db_delay_now).
    #
    # Nachgemessen, welcher Zug das Signal traegt (ohne Richtungsfilter erfasst das
    # Feature beide): es ist fast ausschliesslich der GEGENZUG (Korrelation 0,26 bei
    # n=7477) und kaum der Zug in gleicher Richtung (0,09 bei n=1878). Passt zur
    # Strecke: gleiche Richtung fahrt im Stundentakt, da ist der Abstand nie knapp -
    # der Gegenzug muss aber kreuzen, und gekreuzt wird nur in Roedental und Coburg.
    # Eine Aufteilung in zwei getrennte Features (gleiche Richtung / Gegenrichtung)
    # plus ein Richtungsfeature wurde getestet und bringt nichts (1,64 statt 1,65;
    # 07:20 unveraendert 2,28), deshalb bleibt es bei dem einen kombinierten Feature.
    re_dep = w[(w["stop_id"] == DOERFLES) & (~w["is_arrival"])
               & (w["line"].isin(["RE19", "RE28", "RE29"]))]
    plan_dep = (re_dep.sort_values("update_timestamp")
                .drop_duplicates("trip_id", keep="first")[["trip_id", "time_schedule"]]
                .sort_values("time_schedule").reset_index(drop=True))
    obs_vor = {k: v for k, v in re_dep[["trip_id", "update_timestamp", "delay_min"]]
               .sort_values("update_timestamp").groupby("trip_id")}

    vor = []
    for _, r in tgt.iterrows():
        frueher = plan_dep[(plan_dep["time_schedule"] < r["sched_dep"]) &
                           (plan_dep["time_schedule"] >= r["sched_dep"] - pd.Timedelta(minutes=VOR_WINDOW))]
        if frueher.empty:
            vor.append((r["trip_id"], np.nan, np.nan))
            continue
        v = frueher.iloc[-1]
        b = obs_vor.get(v["trip_id"])
        delay = np.nan
        if b is not None:
            b = b[b["update_timestamp"] <= r["T"]]   # nur was zum Abfragezeitpunkt bekannt war
            if len(b):
                delay = b["delay_min"].iloc[-1]
        vor.append((r["trip_id"], delay,
                    (r["sched_dep"] - v["time_schedule"]).total_seconds() / 60))
    tgt = tgt.merge(pd.DataFrame(vor, columns=["trip_id", "vorgaenger_delay",
                                               "vorgaenger_luecke_min"]),
                    on="trip_id", how="left")

    # ---------------- Kalender / Kontext ----------------
    loc = tgt["sched_dep"].dt.tz_convert("Europe/Berlin")
    tgt["hour"] = loc.dt.hour
    tgt["minute_of_day"] = loc.dt.hour * 60 + loc.dt.minute
    tgt["dow"] = loc.dt.dayofweek
    tgt["month"] = loc.dt.month
    tgt["is_weekend"] = (tgt["dow"] >= 5).astype(int)
    tgt["after_fahrplanwechsel"] = (tgt["sched_dep"] >= pd.Timestamp("2025-12-15", tz="UTC")).astype(int)

    tgt = tgt.sort_values("sched_dep").reset_index(drop=True)
    tgt.to_parquet(OUT, index=False)
    print(f"\nFeature-Tabelle gespeichert: {OUT}")
    print(f"Zeilen: {len(tgt)}, Spalten: {len(tgt.columns)}")
    print(f"Zeitraum: {tgt['sched_dep'].min()} bis {tgt['sched_dep'].max()}")
    print("\nZielgroesse y_delay_min:")
    print(tgt["y_delay_min"].describe().to_string())
    print("\nFehlende Werte je Feature:")
    print(tgt.isna().sum()[tgt.isna().sum() > 0].to_string())


if __name__ == "__main__":
    main()
