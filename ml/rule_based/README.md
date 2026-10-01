# Analyse des regelbasierten Modells

Hier liegt **kein** produktiver Code. Zwei Artefakte aus der Offline-Untersuchung von
`simulate.py` (Projektwurzel), die sonst verloren gegangen wären.

## `simulate_fixed.py` — Bugfix-Vorschlag, NICHT angewendet

Kopie von `simulate.py` mit einer einzigen Änderung in `_required_shift()`.

**Der Fehler:** Im Gegenzug-Zweig bildet das Original `min()`/`max()` über *alle*
gemeinsam genutzten eingleisigen Segmente und vergleicht diese beiden Riesenintervalle.
Zwei Züge, die sich auf einer Strecke mit mehreren Blockabschnitten begegnen, überlappen
in dieser Rechnung immer — auch wenn sie auf keinem *einzelnen* Segment zeitlich
kollidieren.

**Konkreter Fall (24.09.2026):** RE 19 „4905" (Ankunft Coburg) gegen RE 29 4900
(Abfahrt Coburg). RE 29 durchfährt drei gemeinsame Segmente zwischen 07:30 und 07:36,
mein Zug ist zu dem Zeitpunkt erst bei Abzweig SFS (07:24). Das Modell rechnete
`need = 07:36 + 1 min Puffer − 07:24 = 13 Minuten` — obwohl sich die Züge auf keinem
Segment überschnitten. Durch die iterative Verschiebung in `resolve()` eskalierte das
anschließend auf 20 Minuten Phantomverspätung. Real kam der Zug 4 Minuten zu spät an.

**Der Fix:** jedes gemeinsame Segment einzeln auf echte zeitliche Überlappung prüfen.

**Warum nicht angewendet:** Der Fix allein hat den Fall nicht gelöst (die Kaskade blieb,
nur mit anderen Zwischenschritten). Die eigentliche Hauptursache lag woanders — siehe unten.

## Die wichtigere Erkenntnis: Ein-Anker-Extrapolation

`make_train()` kennt nur *einen* Zeitanker (Coburg) und rechnet alle übrigen
Stationszeiten über die festen `run_min`-Werte aus `config.toml` hoch. Dabei liegen in
`changes.jsonl` längst **echte gemessene Zeiten** für Rödental, Dörfles-Esbach und
Coburg Nord vor, die komplett ignoriert werden. Die Extrapolationsfehler erzeugen
Schein-Überschneidungen zwischen Zügen, die real nie kollidiert sind.

Als das Offline-Replay stattdessen die echten Zwischenzeiten verwendete, verschwand der
Fehlalarm vollständig und der MAE fiel von 2,35 auf 1,94 min (DB-Prognose: 1,97 min).

**Empfehlung für `re19watch.py`:** `make_train()` / `build_others()` so erweitern, dass
vorhandene echte Zwischenstationszeiten genutzt werden statt reiner Extrapolation.
Das ist der größte Hebel am regelbasierten Modell.

## Weiterer offener Fund: Zeitfenster nach Planzeit

`TimetablesClient.board()` filtert Kandidatenzüge nach `ev.planned`, nicht nach der
aktuellen Prognose:

```python
if ev and sid not in seen and ev.planned and when <= ev.planned < end:
```

Ein Zug, der laut Fahrplan um 09:27 fährt, real aber 53 Minuten später (≈ 10:20) im
kritischen Fenster auftaucht, wird dadurch **nie geladen**. Nachgewiesen am 24.09.2026
mit RE 29 4904: dessen Planzeit lag außerhalb des Fensters jedes Werrabahn-Zuges
zwischen 10:20 und 10:41, obwohl er real genau dort unterwegs war. Betroffen sind
ausgerechnet die stark verspäteten Vorrangzüge — also genau der Fall, um den es geht.

## `replay_multiday.py` — Offline-Validierung

Spielt gesammelte Tage aus `~/re19-watch/data/` durch `simulate_fixed.py`, für alle
Werrabahn-Züge statt nur den eigenen. Erwartet `simulate_fixed.py` im selben Ordner und
die Originaldateien in `~/re19-watch` (nur lesend).

Ergebnis über 24.–26.09.2026 (498 Fälle): MAE Modell 2,08 min, DB-Prognose 2,10 min —
13× besser, 5× schlechter, 480× gleich.
