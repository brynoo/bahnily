"""Exportiert die RE19-Fahrten Richtung Coburg/Nuernberg als CSV (eine Zeile je Fahrt).

Quelle ist die gefilterte Jahresdatei data/werrabahn_marschbahn_2025.parquet
(aus extract_2025.py). Geschrieben wird nach ~/Desktop, Trennzeichen Semikolon und
Komma als Dezimaltrennzeichen, damit Excel die Datei direkt richtig oeffnet.

Richtung: nur Fahrten, die in Coburg SPAETER halten als in Doerfles-Esbach - also
Nord -> Sued. Zuege der Gegenrichtung sind nicht enthalten.

Zeiten sind Ortszeit (Europe/Berlin). Soll = erste Beobachtung des Halts (Fahrplan),
Ist = letzter bekannter Stand. ACHTUNG: der letzte bekannte Stand ist bei einem Teil
der Halte noch eine Prognose und keine gemessene Ist-Zeit - die Quelle kennzeichnet das
nicht zuverlaessig (siehe ml/README.md).

Aufruf: python export_csv.py [--out PFAD]
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
TZ = "Europe/Berlin"

# Korridor Nord -> Sued
STATIONEN = [
    (8013008, "sonneberg", "Sonneberg(Thür)Hbf"),
    (8004325, "neustadt", "Neustadt(b Coburg)"),
    (8004064, "moenchroeden", "Mönchröden"),
    (8005122, "roedental_mitte", "Rödental Mitte"),
    (8004633, "roedental", "Rödental"),
    (8001484, "doerfles", "Dörfles-Esbach"),
    (8001334, "coburg_nord", "Coburg Nord"),
    (8001338, "coburg", "Coburg"),
]
DOERFLES, COBURG = 8001484, 8001338

# Verspaetungsgruende der DB (IRIS). Quelle: Travel::Status::DE::IRIS, %translation.
GRUENDE = {
    1: "Nähere Informationen in Kürze", 2: "Polizeieinsatz",
    3: "Feuerwehreinsatz auf der Strecke", 4: "Kurzfristiger Personalausfall",
    5: "Ärztliche Versorgung eines Fahrgastes", 6: "Betätigen der Notbremse",
    7: "Unbefugte Personen auf der Strecke", 8: "Notarzteinsatz auf der Strecke",
    9: "Streikauswirkungen", 10: "Tiere auf der Strecke", 11: "Unwetter",
    12: "Warten auf ein verspätetes Schiff", 13: "Pass- und Zollkontrolle",
    14: "Defekt am Bahnhof", 15: "Beeinträchtigung durch Vandalismus",
    16: "Entschärfung einer Fliegerbombe", 17: "Beschädigung einer Brücke",
    18: "Umgestürzter Baum auf der Strecke", 19: "Unfall an einem Bahnübergang",
    20: "Tiere im Gleis", 21: "Warten auf Anschlussreisende",
    22: "Witterungsbedingte Beeinträchtigungen", 23: "Betriebsstabilisierung",
    24: "Verspätung im Ausland", 25: "Bereitstellung weiterer Wagen",
    26: "Abhängen von Wagen", 27: "Technische Störung am Bus",
    28: "Gegenstände auf der Strecke", 29: "Ersatzverkehr mit Bus ist eingerichtet",
    30: "Personalausfall im Stellwerk", 31: "Bauarbeiten",
    32: "Längere Haltezeit am Bahnhof", 33: "Defekt an der Oberleitung",
    34: "Defekt an einem Signal", 35: "Streckensperrung",
    36: "Technische Störung am Zug", 37: "Kurzfristiger Fahrzeugausfall",
    38: "Defekt an der Strecke", 39: "Stau / Hohes Verkehrsaufkommen",
    40: "Defektes Stellwerk", 41: "Defekt an einem Bahnübergang",
    42: "Außerplanmäßige Geschwindigkeitsbeschränkung",
    43: "Verspätung eines vorausfahrenden Zuges",
    44: "Warten auf einen entgegenkommenden Zug",
    45: "Vorfahrt eines anderen Zuges", 46: "Vorfahrt eines anderen Zuges",
    47: "Verspätete Bereitstellung", 48: "Verspätung aus vorheriger Fahrt",
    49: "Kurzfristiger Personalausfall", 50: "Kurzfristige Erkrankung von Personal",
    51: "Verspätetes Personal aus vorheriger Fahrt", 52: "Streik",
    53: "Unwetterauswirkungen", 54: "Verfügbarkeit der Gleise derzeit eingeschränkt",
    55: "Technischer Defekt an einem anderen Zug", 56: "Laden der Antriebsbatterie",
    57: "Zusätzlicher Halt", 58: "Umleitung", 59: "Schnee und Eis",
    60: "Witterungsbedingt verminderte Geschwindigkeit", 61: "Defekte Tür",
    62: "Behobener Defekt am Zug", 63: "Technische Untersuchung am Zug",
    64: "Defekt an einer Weiche", 65: "Erdrutsch", 66: "Hochwasser",
    67: "Behördliche Maßnahme", 68: "Hohes Fahrgastaufkommen",
    69: "Zug verkehrt mit verminderter Geschwindigkeit",
    99: "Verzögerungen im Betriebsablauf",
}
WOCHENTAG = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path.home() / "Desktop" / "re19_sonneberg_coburg_2025.csv"))
    args = ap.parse_args()

    df = pd.read_parquet(HERE / "data" / "werrabahn_marschbahn_2025.parquet")
    w = df[(df["strecke"] == "werrabahn") & (df["line"] == "RE19")].copy()
    w["delay_min"] = w["delay"] / 60.0
    print(f"RE19-Zeilen gesamt: {len(w)}, Fahrten: {w['trip_id'].nunique()}")

    # Erste Beobachtung je Halt = Soll-Fahrplan, letzte = aktuellster Stand
    srt = w.sort_values("update_timestamp")
    erst = srt.drop_duplicates(["trip_id", "stop_id", "is_arrival"], keep="first")
    letzt = srt.drop_duplicates(["trip_id", "stop_id", "is_arrival"], keep="last")

    # Richtung: Coburg spaeter als Doerfles -> Nord nach Sued
    seq = erst.pivot_table(index="trip_id", columns="stop_id", values="stop_sequence", aggfunc="min")
    if COBURG not in seq.columns or DOERFLES not in seq.columns:
        raise SystemExit("Doerfles oder Coburg fehlen in den Daten")
    sued = (seq[COBURG] > seq[DOERFLES])
    sued = sued[sued.notna() & sued].index
    print(f"davon Richtung Coburg/Nürnberg: {len(sued)} Fahrten")

    e = erst[erst["trip_id"].isin(sued)]
    l = letzt[letzt["trip_id"].isin(sued)]

    # Grundgerüst: eine Zeile je Fahrt
    basis = (e.sort_values("stop_sequence").drop_duplicates("trip_id")
             [["trip_id", "number", "line", "initial_stop_id"]].set_index("trip_id"))

    # Soll-Abfahrt in Doerfles bestimmt Datum und Sortierung
    ank = e[(e["stop_id"] == DOERFLES) & (~e["is_arrival"])].set_index("trip_id")["time_schedule"]
    basis["soll_doerfles"] = ank
    basis = basis[basis["soll_doerfles"].notna()].copy()
    lok = basis["soll_doerfles"].dt.tz_convert(TZ)
    out = pd.DataFrame({
        "datum": lok.dt.strftime("%Y-%m-%d"),
        "wochentag": lok.dt.dayofweek.map(lambda i: WOCHENTAG[i]),
        "kw": lok.dt.isocalendar().week.astype("Int64"),
        "zug_nr": basis["number"],
        "linie": basis["line"],
    }, index=basis.index)

    # Je Station Soll, Ist und Verspätung
    for eva, kurz, _label in STATIONEN:
        arr = (eva == COBURG)          # Coburg ist Endstation -> Ankunft
        es = e[(e["stop_id"] == eva) & (e["is_arrival"] == arr)].set_index("trip_id")
        ls = l[(l["stop_id"] == eva) & (l["is_arrival"] == arr)].set_index("trip_id")
        soll = es["time_schedule"].reindex(out.index)
        ist = ls["time_real"].reindex(out.index)
        vsp = ls["delay_min"].reindex(out.index)
        out[f"{kurz}_soll"] = soll.dt.tz_convert(TZ).dt.strftime("%H:%M")
        out[f"{kurz}_ist"] = ist.dt.tz_convert(TZ).dt.strftime("%H:%M")
        out[f"{kurz}_versp_min"] = vsp.round(1)

    # Ausfall und Verspätungsgründe über alle Halte der Fahrt
    out["ausgefallen"] = (l.groupby("trip_id")["is_cancelled"].any()
                          .reindex(out.index).fillna(False).map({True: "ja", False: "nein"}))

    codes = {}
    for trip, g in w[w["trip_id"].isin(out.index)].groupby("trip_id"):
        s = set()
        for a in g["message_codes"].dropna():
            if a is not None and len(a):
                s.update(int(x) for x in a if int(x) > 0)
        codes[trip] = sorted(s)
    out["gruende_codes"] = pd.Series({k: ", ".join(map(str, v)) for k, v in codes.items()}).reindex(out.index)
    out["gruende_text"] = pd.Series(
        {k: "; ".join(GRUENDE.get(c, f"Code {c}") for c in v) for k, v in codes.items()}
    ).reindex(out.index)

    out = out.sort_values(["datum", "doerfles_soll"]).reset_index(drop=True)

    ziel = pathlib.Path(args.out).expanduser()
    out.to_csv(ziel, index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(f"\ngeschrieben: {ziel}")
    print(f"{len(out)} Zeilen, {len(out.columns)} Spalten, "
          f"{out['datum'].nunique()} Tage ({out['datum'].min()} bis {out['datum'].max()})")


if __name__ == "__main__":
    main()
