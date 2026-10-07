# Gelerntes Verspätungsmodell (Werrabahn)

Ergänzung zum regelbasierten `simulate.py` im Projektwurzelverzeichnis. Statt die Physik
von Hand zu programmieren, lernt dieses Modell aus historischen Daten, wie sich eine
Verspätung bis Dörfles-Esbach entwickelt.

**Zielgröße:** tatsächliche Abfahrtsverspätung des RE 19 in Dörfles-Esbach (Minuten).
**Vorhersagezeitpunkt:** 15 Minuten vor der planmäßigen Abfahrt (bei 07:20 also die Abfrage um 07:05).
**Streckenumfang:** Sonneberg – Coburg, Linien RE 19 / RE 28 / RE 29.

## Setup

```bash
cd ml
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Benutzung

```bash
# Nächster RE 19 ab Dörfles, Stand jetzt
.venv/bin/python3 predict_now.py

# So tun, als wäre es 07:05 Uhr
.venv/bin/python3 predict_now.py --at 07:05

# Rückblick auf einen bestimmten Tag
.venv/bin/python3 predict_now.py --at 07:05 --date 2026-09-29
```

`predict_now.py` liest den aktuellen Stand **nur lesend** aus `~/re19-watch/data/<datum>/`
(der laufenden Sammlung) und ändert dort nichts.

## Pipeline (nur nötig, wenn neu trainiert werden soll)

| Schritt | Skript | Ergebnis |
|---|---|---|
| 1 | `extract_2025.py` | streamt `~/Downloads/2025.tar` (19 GB, Mobilithek / Bahn-Vorhersage) und filtert auf die 13 Werrabahn-/Marschbahn-EVAs → `data/werrabahn_marschbahn_2025.parquet` (31 MB) |
| 2 | `build_features.py` | baut die Feature-Tabelle, 10 129 Trainingsbeispiele → `data/features_werrabahn.parquet` |
| 3 | `train_model.py` | trainiert und evaluiert → `model_werrabahn.joblib` |

Das Archiv wird **nie vollständig entpackt** (Platzbedarf), sondern Tagesdatei für
Tagesdatei im Speicher verarbeitet.

## Datenquelle

Mobilithek-Abo „Bahn-Vorhersage: Geparste deutschlandweite Verspätungsdaten"
(Angebot 938616012299546624), jährliche `.tar`-Archive mit täglichen Parquet-Dateien.
Lizenz ODbL, Quellenvermerk: Bahn-Vorhersage, Deutsche Bahn, OpenStreetMap, Trainline, DELFI.
Doku: https://bahnvorhersage.de/open-data/parsed-train-delays

Wichtig: `stop_id` in diesem Datensatz **ist** die EVA-Nummer, kein Umweg über eine
Haltestellentabelle nötig. Alle Zeitstempel sind UTC. `delay` ist in **Sekunden**.
`is_final == True` markiert den letzten Echtzeitstand, also die tatsächliche Zeit.

## Funktioniert es? — Kurzfassung

Praxistest auf der **eigenen Sammlung** (Sept/Okt 2026, aktuelles Fahrplanjahr,
195 auswertbare Abfahrten, nie im Training gesehen):

| | n | Modell | DB-Prognose | Modell besser |
|---|---|---|---|---|
| Alle Abfahrten | 195 | **2,00 min** | 2,16 min | 51,8 % |
| Morgens 6–8 h, Mo–Fr | 10 | **3,92 min** | 5,40 min | 70,0 % |
| Ist-Verspätung ≥ 5 min | 34 | **4,95 min** | 6,76 min | **85,3 %** |
| Ist-Verspätung < 5 min | 161 | 1,37 min | **1,19 min** | 44,7 % |

**Antwort: ja, aber mit klarem Profil.** Das Modell gewinnt dort, wo es darauf
ankommt — bei echten Verspätungen schlägt es die DB in 85 % der Fälle und senkt den
Fehler um 1,8 Minuten. An ruhigen Tagen (83 % aller Fahrten) ist es dagegen minimal
schlechter als die DB, weil es Verspätungen andeutet, die nicht kommen.

Wer nur ein Alarmsignal will, fährt mit einer Schwellenregel am besten: DB nehmen,
außer das Modell warnt vor ≥ 7 min. Das reduziert Fehlalarme stark, verpasst aber
auch einen Großteil der schlechten Tage — es ist ein Frühwarnsignal, kein Sicherheitsnetz.

Nachrechnen: `python backtest_eigene_daten.py`

## Kombinierte Pipeline (`kombiniert.py`)

Die beiden Modelle können unterschiedliche Dinge, deshalb werden sie kombiniert:

| | kann | kann nicht |
|---|---|---|
| **Regelmodell** (`simulate.py`) | erkennt, ob ein Vorrangzug zeitlich ins eigene Fenster fällt | die Höhe beziffern (sagt 18 min, real ~7) |
| **ML-Modell** | eine bestehende Verspätung fortschreiben (85 % besser als DB bei ≥5 min) | strukturelle Konflikte sehen (flache Antwort im Szenariotest) |

Ablauf: Regelmodell prüft auf Konflikt → falls ja, wird seine Minutenangabe **verworfen**
und durch den empirischen Median aus 553 realen 2025er-Fällen ersetzt → Ergebnis ist
`max(ML-Prognose, eigene Verspätung + Konfliktzuschlag)`.

### Empirische Validierung des Konfliktfensters

Aus 553 realen Fällen, in denen tatsächlich ein ICE im Konfliktfenster fuhr
(verifizierte Ist-Werte):

| ICE real verspätet | n | RE 19 Ist (Median) |
|---|---|---|
| pünktlich | 68 | 1 min |
| 1–3 min | 236 | 1 min |
| 3–5 min | 81 | 3 min |
| 5–8 min | 31 | 4 min |
| **8–12 min** | **37** | **7 min** ← Spitze |
| 12–20 min | 36 | 1 min |
| > 20 min | 64 | 3 min |

**Das Regelmodell traf das Fenster exakt.** Es meldet Konflikt bei ICE +8 bis +12 —
genau dort springt die reale Verspätung von 1 auf 7 Minuten und fällt danach wieder
zurück (der ICE ist dann so spät, dass der RE 19 längst in Coburg steht).

### Was die Pipeline kann und was nicht

Out-of-sample geprüft (Kalibrierung auf Jan–Sep 2025, Test auf Okt–Dez):

- **Erkennung: belastbar.** Jede Gruppe liegt 1–2 min neben dem tatsächlichen Wert,
  das Konfliktfenster ist auch in den ungesehenen Daten die Spitze.
- **Bezifferung: nicht belastbar.** MAE 3,00 min gegenüber 3,20 min für eine konstante
  Vorhersage von 2 Minuten — nur 6 % besser. Die Streuung innerhalb des kritischen
  Fensters (0 bis 43 min bei Median 7) ist größer als der Unterschied zwischen den
  Gruppen.

Deshalb gibt `kombiniert.py` eine **Spanne (p25–p90) und eine Risikostufe** aus statt
einer Punktzahl. Die Aussage lautet „heute ist ein Risikotag, rechne mit 5–12 Minuten",
nicht „du wirst 7 Minuten zu spät sein".

**Einschränkung des Szenariotests:** `szenario_test.py` und die Demo in `kombiniert.py`
füttern das ML-Modell mit konstruierten Feature-Vektoren (alle Oberlauf-Verspätungen
exakt 0), wie sie real selten vorkommen. Die ML-Spalte reagiert dort entsprechend
unruhig und sollte nicht überinterpretiert werden — der belastbare ML-Beleg sind die
Backtests auf echten Daten.

## ⚠ Datenleckage im Quelldatensatz (wichtig)

Der Mobilithek-Datensatz markiert den letzten empfangenen Echtzeitstand als
`is_final` und bezeichnet ihn in der Doku als „tatsächliche Zeit". **Das stimmt für
die Mehrheit der Zeilen nicht:** gemessen wurden 58,9 % der finalen Datenpunkte
erfasst, *bevor* die angebliche Ist-Zeit eingetreten war. Es sind also nie
verifizierte Prognosen, keine Beobachtungen.

Kritisch wird das, wenn dieser finale Stand schon vor dem Vorhersagezeitpunkt vorlag:
dann **ist** die DB-Prognose die Zielgröße. Betroffen waren 1 367 von 10 129 Zeilen
(13,5 %), dort stimmte `db_delay_now` zu 100 % exakt mit `y` überein.

`build_features.py` entfernt diese Zeilen jetzt und setzt zusätzlich das Flag
`ziel_verifiziert` (Zielwert erst nach Eintreten der Zeit bestätigt — trifft auf 42 %
der verbleibenden Zeilen zu). `backtest_eigene_daten.py` filtert analog.

Die Leckage hat die **DB-Baseline künstlich stark** gemacht, nicht das Modell: in
geleakten Zeilen hat die DB per Definition null Fehler. Nach der Bereinigung
verbessert sich der gemessene Modellvorteil in jedem einzelnen Slice.

**Konsequenz für die absoluten Zahlen:** Auf nachweislich beobachteten Zielwerten ist
die Aufgabe deutlich schwerer als die Rohdaten suggerieren (2,56 statt 1,90 min MAE).
Alle Zahlen unten sind nach Bereinigung.

## Ergebnisse (Test: Nov + Dez 2025, 1 616 Fälle, nie im Training gesehen)

| Slice | n | MAE Modell | MAE DB-Prognose | Modell besser |
|---|---|---|---|---|
| Gesamt | 1 616 | **2,03 min** | 2,36 min | 62,7 % |
| vor Fahrplanwechsel (< 15.12.) | 1 127 | **1,96 min** | 2,39 min | 65,9 % |
| nach Fahrplanwechsel (≥ 15.12.) | 489 | **2,19 min** | 2,27 min | 55,6 % |
| Morgens 6–8 h, Mo–Fr | 64 | **2,93 min** | 3,46 min | 60,3 % |
| Ist-Verspätung ≥ 5 min | 345 | **4,60 min** | 6,24 min | 84,9 % |

Wichtigste Features (Permutation Importance): `db_delay_now` (0,20), `up_roedental` (0,11),
`conflict_max_delay` (0,11). Dass die Verspätung der RE 28/RE 29-Konfliktzüge auf Platz 3
landet, stützt die Ausgangshypothese des Projekts empirisch.

### Einzelfall 29.09.2026 (Signalstörung, tatsächliche Abfahrt 08:10 = +50 min)

| Stand 07:05 | Vorhersage | Fehler |
|---|---|---|
| DB-Prognose | 07:35 (+15) | 35 min |
| Modell | 07:47 (+28) | 22 min |

Das Modell warnte 17 Minuten früher als die DB (die erst um 07:22 auf +42 korrigierte),
unterschätzte aber trotzdem deutlich. Ein Einzelfall, kein Beleg.

## Warum das den Fahrplanwechsel überlebt

Alle Features sind **relativ** (Verspätungen in Minuten, Zähler, Zeitabstände) — bewusst
keine absoluten Zugnummern oder Soll-Uhrzeiten als Identität. Dem Modell ist egal, ob der
Zug 07:20 oder 07:17 fährt; es rechnet mit „wie spät ist er gerade, wo steht der Gegenzug".
Empirisch bestätigt: der 07:20-Slot in Dörfles blieb über den Fahrplanwechsel am 15.12.2025
unverändert, andere Slots verschoben sich um eine Minute.

**Einschränkung:** Nach dem Wechsel schrumpft der Vorsprung (0,08 statt 0,43 min, 55,6 %
statt 65,9 % der Fälle besser) — bei nur 489 Testfällen aus 2,5 Wochen. Das Modell
funktioniert weiter, aber schwächer. Belastbar wird das erst mit mehreren Wochen eigener
Daten aus dem aktuellen Fahrplanjahr zum Nachtrainieren.

## Bekannte Schwächen

- Trainiert auf Normalbetrieb; große Störungen (29.09.2026: 3 h Signalstörung) werden
  systematisch unterschätzt.
- Zielgröße enthält Ausreißer bis −11 min (Zug „zu früh"), vermutlich Artefakte aus
  kurzfristigen Fahrplanänderungen. Nicht bereinigt.
- Nur Dörfles-Esbach als Zielpunkt, nur RE 19 als Zielzug.

## Vorausfahrender/kreuzender Zug (Oktober 2026)

Das Modell kannte als Konfliktzüge nur RE28, RE29 und ICE in einem ±30-min-Fenster,
aber nicht den Zug, der unmittelbar vor mir über den eingleisigen Abschnitt gefahren
ist – obwohl „Verspätung eines vorausfahrenden Zuges" (Code 43) der zweithäufigste
Verspätungsgrund in den RE19-Daten ist. Zwei neue Features schließen die Lücke:

- `vorgaenger_delay` – Verspätung der letzten RE19/RE28/RE29-Abfahrt in Dörfles-Esbach
  vor meiner (Fenster 45 min), Stand zum Abfragezeitpunkt T
- `vorgaenger_luecke_min` – planmäßiger Abstand zu diesem Zug

Gemessen über fünf gleitende Zweimonatsfenster (jeweils nur mit Daten vor dem Fenster
trainiert), sowie am Backtest auf eigenen Daten:

| Slice | vorher | nachher | DB-Prognose |
|---|---|---|---|
| Gesamt (2025, 5 Fenster, n=6816) | 1,73 | **1,65** | 2,05–2,36 |
| 07:20-Slot (n=228) | 2,40 | **2,28** | 3,24 |
| morgens 6–8 h Mo–Fr (n=727) | 2,54 | **2,40** | 3,46 |
| Ist ≥ 5 min (n=1229) | 5,27 | **4,89** | 6,24 |
| eigene Daten 2026 (n=133) | 1,78 | **1,66** | 1,92 |
| eigene Daten, Ist ≥ 5 min (n=22) | – | **4,68** | 6,64 |

`vorgaenger_delay` ist danach das stärkste Feature des Modells (Permutation Importance
0,26 gegenüber 0,13 für die DB-Prognose selbst).

**Woher das Signal kommt:** nachgemessen trägt es fast nur der *Gegenzug*
(Korrelation 0,26 bei n=7477), kaum der Zug in gleicher Richtung (0,09 bei n=1878).
Das passt zur Strecke – in gleicher Richtung fährt der Stundentakt, da wird der Abstand
nie knapp; der Gegenzug muss aber kreuzen, und gekreuzt werden kann nur in Rödental
und Coburg. Eine Aufspaltung in getrennte Features je Richtung plus ein explizites
Richtungsfeature wurde getestet und bringt nichts (1,64 statt 1,65), deshalb bleibt es
bei dem einen kombinierten Feature.

**Nicht einbezogen:** agilis, RB, STB und Bus. Die stehen auf den Tafeln von Coburg und
Sonneberg, fahren den Abschnitt Coburg–Sonneberg aber nicht. Sie als Konfliktzüge
mitzunehmen wurde getestet und verschlechtert das Modell messbar (07:20: 2,21 statt 2,16).

## Hinweis zum Prognose-Protokoll (predictions.jsonl)

Bis Oktober 2026 hat `collect.py` unter dem Schlüssel `model_estimate` das
**Regelmodell** (`est.estimate`) protokolliert, nicht das ML-Modell – `est.ml_estimate`
wurde gar nicht geschrieben. In Dateien aus diesem Zeitraum ist die ML-Prognose daher
nicht rekonstruierbar; sichtbar ist nur die fertige Systemausgabe (`expected_delay_min`),
in die das ML-Modell eingeht. Seit dem Fix heißen die Felder `rule_estimate` und
`ml_estimate`. Der Fehler fiel nicht auf, weil `test_simulate.py` die Feldnamen des
Protokolls nicht prüft.
