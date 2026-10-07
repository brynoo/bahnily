"""
Streamt alle Tagesdateien aus ~/Downloads/2025.tar (19 GB, NICHT voll entpacken!),
filtert jede Tagesdatei direkt im Speicher auf unsere 13 Werrabahn-/Marschbahn-
Stationen und haengt das Ergebnis an eine konsolidierte Parquet-Datei im Scratchpad an.
Das Original-Tar wird nur lesend geoeffnet (tarfile, Modus 'r'), nichts wird dort
veraendert oder geloescht.
"""
import io
import os
import tarfile
import pandas as pd

import pathlib
HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"


TAR_PATH = os.environ.get("TAR_PATH", str(pathlib.Path.home() / "Downloads" / "2025.tar"))
OUT_PATH = str(DATA / "werrabahn_marschbahn_2025.parquet")

WERRABAHN = [8013008, 8004325, 8004064, 8005122, 8004633, 8001484, 8001334, 8001338]
MARSCHBAHN = [8004343, 8003303, 8004093, 8003222, 8006369]
ALL_EVAS = set(WERRABAHN) | set(MARSCHBAHN)

LINE_NAME = {eva: "werrabahn" for eva in WERRABAHN}
LINE_NAME.update({eva: "marschbahn" for eva in MARSCHBAHN})


def main():
    chunks = []
    n_files = 0
    n_rows_total = 0
    with tarfile.open(TAR_PATH, "r") as tar:
        members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".parquet")]
        members.sort(key=lambda m: m.name)
        print(f"Gefundene Tagesdateien im Archiv: {len(members)}")
        for i, member in enumerate(members):
            f = tar.extractfile(member)
            if f is None:
                continue
            buf = io.BytesIO(f.read())
            df = pd.read_parquet(buf, columns=[
                "time_schedule", "time_real", "update_timestamp", "trip_id",
                "stop_id", "operator", "category", "number", "line",
                "is_regional", "is_final", "is_arrival", "is_cancelled",
                "delay", "dwell_time_schedule", "dwell_time_real",
                "stop_sequence", "message_codes", "platform_id_schedule",
                "platform_id_real", "initial_scheduled_departure", "initial_stop_id",
            ])
            sub = df[df["stop_id"].isin(ALL_EVAS)].copy()
            if len(sub):
                sub["strecke"] = sub["stop_id"].map(LINE_NAME)
                chunks.append(sub)
                n_rows_total += len(sub)
            n_files += 1
            if n_files % 30 == 0:
                print(f"  ... {n_files}/{len(members)} Tage verarbeitet, bisher {n_rows_total} Zeilen gefiltert")

    print(f"Fertig: {n_files} Tage, {n_rows_total} Zeilen insgesamt (Werrabahn+Marschbahn)")
    result = pd.concat(chunks, ignore_index=True)
    result.to_parquet(OUT_PATH, index=False)
    print(f"Gespeichert nach {OUT_PATH} ({len(result)} Zeilen, {result.memory_usage(deep=True).sum() / 1e6:.1f} MB im Speicher)")


if __name__ == "__main__":
    main()
