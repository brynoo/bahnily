# re19-watch

Jeden Werktag um 07:05 eine Telegram-Nachricht: Wird mein **RE 19 ab Dörfles-Esbach 07:20** auf dem eingleisigen Abschnitt Rödental – Coburg durch einen Zug mit Vorrang aufgehalten – *bevor* der DB Navigator es anzeigt?

**Ergebnis:** auf dem 07:20-Zug liegt die Schätzung bei 2,27 min mittlerem Fehler gegenüber
3,03 min für die DB-Prognose zum selben Zeitpunkt – gemessen an einem Jahr Echtdaten, bei
dem immer nur mit Daten *vor* dem Testzeitraum trainiert wurde. Bei Verspätungen ab 5 min
4,88 gegenüber 6,37. Die ausführlichen Zahlen und die Grenzen stehen weiter unten.

## Warum das funktionieren kann

Die DB-Prognose für den RE 19 reagiert erst, wenn der RE selbst später dran ist. Die Verspätung eines ICE, der gleich den eingleisigen Abschnitt Coburg – Einschleifung belegt, ist aber oft **schon bekannt**. re19-watch rechnet mit deinen Fahrzeiten aus, ob sich die beiden Belegungen überschneiden und wer warten muss.

Zweimal simuliert: einmal mit Soll-Zeiten, einmal mit Prognosen. Zusätzlich schätzt ein
ML-Modell die tatsächliche Verspätung aus DB-Prognose, Oberlauf, Konfliktzügen und dem
Zug, der unmittelbar vor mir über den eingleisigen Abschnitt fährt.

**Arbeitsteilung zwischen beiden Modellen** (so, wie die Messung sie ergeben hat):

- Die **Zahl** kommt allein aus dem ML-Modell, mit einer Plausibilitätsgrenze gegen
  Ausreißer. Das Regelmodell hat darauf keinen Einfluss mehr.
- Das **Regelmodell** liefert die Begründung – *welcher* Zug *wo* kreuzt. Das kann das
  ML-Modell nicht, und es ist der verständlichere Teil der Nachricht.
- Fehlt das ML-Modell oder eine seiner Abhängigkeiten, bleibt der Regelcheck als
  Rückfallebene aktiv.

Früher durfte das Regelmodell die Schätzung nach oben überstimmen. Ein Replay über das
ganze Jahr 2025 (`ml/replay_2025.py`, 4291 Zielzüge) hat gezeigt, dass das schadet: es
erkennt eine Blockade bei 13,7 % der Züge, davon sind **73 % Fehlalarme**, und mit Veto
wurde der 07:20-Slot auf 4,19 min MAE verschlechtert statt 2,47 – schlechter sogar als
die DB-Prognose. Auch auf die präzise Teilmenge begrenzt blieb es schädlich. Das Veto
ist deshalb entfernt.

## Einrichtung (ca. 15 min)

1. **Telegram-Bot**: In Telegram `@BotFather` → `/newbot` → Token kopieren. Dann deinem neuen Bot eine beliebige Nachricht schreiben und `https://api.telegram.org/bot<TOKEN>/getUpdates` im Browser öffnen → `"chat":{"id": ...}` ist deine Chat-ID.
2. **Repo**: Ordner als GitHub-Repo hochladen (privat reicht).
3. **Secrets**: Repo → Settings → Secrets and variables → Actions → `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `DB_CLIENT_ID`, `DB_API_KEY` anlegen. Token nie in Dateien committen.
4. **Testen**: Actions → „RE19 Morgen-Check" → *Run workflow* → `telegram-test` (nur Testnachricht), `mock` (kompletter Durchlauf mit erfundenem ICE-Konflikt) oder `force` (echter Check jetzt).

Lokal:

```bash
export DB_CLIENT_ID=… DB_API_KEY=… TELEGRAM_BOT_TOKEN=… TELEGRAM_CHAT_ID=…
pip install -r requirements.txt
python re19watch.py --telegram-test       # Testnachricht an Telegram
python re19watch.py --mock                # Durchlauf mit erfundenen Daten, schickt Nachricht
python re19watch.py --force --dry-run     # echter Check, nur Ausgabe (vergangene Stunden liefert die API nicht)
python test_simulate.py                   # Logik-Tests ohne Internet
```

Für den Hybridbetrieb müssen die ML-Abhängigkeiten installiert sein; das Modell liegt
unter `ml/model_werrabahn.joblib`. Fehlt das Modell oder eine ML-Abhängigkeit, läuft
der physikalische Regelcheck weiterhin als Fallback.

## Kalibrieren (das macht den Unterschied)

- `config.toml` → `[[corridor.segment]]`: deine gemessenen Fahrzeiten eintragen (Abfahrt A bis Abfahrt B).
- `crossing_points`: auf OpenRailwayMap prüfen, wo Züge sich wirklich kreuzen können, und wo genau die ICE-Einschleifung liegt.
- `predictions.csv` (bei lokalen Läufen): Spalte „tatsächlich" selbst füllen. Nach 2–3 Wochen siehst du, wie oft die Warnung stimmte.
- Kommt ein 🔧 Kalibrierhinweis, sieht das Modell schon im Soll-Fahrplan einen Konflikt → Puffer/Fahrzeiten zu streng.

## Ergebnisse

Trainiert auf einem Jahr deutschlandweiter Verspätungsdaten (2025, Mobilithek), bewertet
über fünf gleitende Zweimonatsfenster – jeweils **nur mit Daten vor dem Fenster trainiert**,
also ohne Blick in die Zukunft. Vergleichsmaßstab ist die DB-Prognose zum selben Zeitpunkt
(07:05, 15 Minuten vor Abfahrt). Fehlermaß ist der mittlere absolute Fehler in Minuten:

| Slice | Modell | DB-Prognose |
|---|---|---|
| Gesamt, 2025 (n=6816) | **1,65** | 2,05–2,36 |
| 07:20-Slot, der Anwendungsfall (n=228) | **2,28** | 3,24 |
| morgens 6–8 h Mo–Fr (n=727) | **2,40** | 3,46 |
| Fälle mit Ist-Verspätung ≥ 5 min (n=1229) | **4,89** | 6,24 |
| eigene Messungen 2026 (n=133) | **1,66** | 1,92 |
| davon Ist-Verspätung ≥ 5 min (n=22) | **4,68** | 6,64 |

Zusätzlicher Härtetest an den 15 störungsreichsten Tagen 2025, die aus dem Training
entfernt wurden: 3,26 gegenüber 3,72 min für die DB, in 68 % der Fälle näher dran.

Das stärkste einzelne Feature ist die Verspätung des Zuges, der kurz vor mir über den
eingleisigen Abschnitt fährt – stärker als die DB-Prognose selbst. Details, Methodik und
die verworfenen Varianten stehen in [`ml/README.md`](ml/README.md).

## Zielerreichung

Drei Ziele, gemessen mit `ml/zielpruefung.py` über fünf gleitende Zweimonatsfenster
(trainiert immer nur mit Daten *vor* dem Testfenster, n = 6465):

**1. Konfliktfälle richtig treffen** – mittlerer absoluter Fehler in Minuten:

| Lage | n | Modell | DB |
|---|---|---|---|
| ICE fährt nah (≤15 min) und ≥10 min verspätet | 334 | **2,71** | 3,21 |
| vorausfahrender Zug ≥8 min verspätet | 357 | **3,64** | 4,94 |
| davon mit tatsächlich ≥5 min Verspätung | 309 | **4,74** | 7,19 |

**2. Im Alltag besser als der DB Navigator:**

| Slice | n | Modell | DB | Modell näher |
|---|---|---|---|---|
| alle Abfahrten | 6465 | **1,69** | 2,22 | 70 % |
| **der 07:20-Zug** | 216 | **2,31** | 3,03 | 68 % |
| morgens 6–8 h Mo–Fr | 694 | **2,42** | 3,12 | 69 % |
| Ist ≥ 5 min | 1209 | **4,88** | 6,37 | 87 % |
| Ist < 5 min | 5256 | **0,95** | 1,27 | 66 % |
| eigene Messungen 2026 (n=185) | | **1,83** | 2,03 | 57 % |
| eigene Messungen, Ist ≥ 5 min (n=28) | | **4,95** | 6,82 | 93 % |

**3. Nicht grundlos Alarm schlagen** – der laute ntfy-Alarm ab 4 min erwarteter
Verspätung (die Telegram-Nachricht kommt ohnehin immer):

| Alarm ausgelöst durch | Alarme/Jahr | davon ≥5 min | erkennt von allen ≥5 min |
|---|---|---|---|
| **Modell ab 4 min** | 30 | **69,7 %** | **44,8 %** |
| DB-Prognose ab 4 min | 19 | 77,3 % | 31,6 % |
| früheres Regelmodell-Veto | 55 | 47,5 % | 51,2 % |

Bei gleicher Messlatte (»lag wirklich mindestens so viel Verspätung vor, wie der Alarm
behauptet«) liegt das Modell bei 85,1 %, der DB Navigator bei 85,6 % – gleich treffsicher
also, aber mit deutlich mehr erkannten Fällen. Die Schwelle 4 ist gemessen gewählt; bei 3
wären 30 % der Fehlalarme Züge mit 0–1 min Verspätung.

## Datenquellen und Lizenz

- **Live-Betrieb:** DB API Marketplace, API „Timetables" (IRIS), kostenlos mit Account.
- **Training:** „Geparste deutschlandweite Verspätungsdaten" über die
  [Mobilithek](https://mobilithek.info), bereitgestellt unter **ODbL**. Die Rohdaten sind
  nicht Teil dieses Repos. Das trainierte Modell (`ml/model_werrabahn.joblib`) ist daraus
  abgeleitet – wer es weiterverwendet, sollte die ODbL-Bedingungen der Quelle beachten.
- **Streckengeometrie** geprüft über [OpenRailwayMap](https://www.openrailwaymap.org)
  (OpenStreetMap, ODbL).
- Der Code selbst steht unter MIT, siehe [`LICENSE`](LICENSE).

## Grenzen

- Keine Stellwerks-/Blockdaten (nicht öffentlich) – es ist eine Heuristik.
- ICE halten nicht in Dörfles-Esbach, ihre Durchfahrt wird aus Coburg-Zeiten hochgerechnet.
- Nur dein Zug wird verschoben, Folgeverspätungen anderer Züge nicht.
- Was um 07:05 noch nicht messbar ist, kann das Modell nicht wissen: an den schlimmsten
  Tagen 2025 entstand etwa die Hälfte der schweren Verspätungen erst nach dem
  Abfragezeitpunkt. Dort liegt die Grenze nicht am Modell, sondern an der Information.
- GitHub-Cronjobs können sich einige Minuten verspäten.
- Daten: offizielle DB-API „Timetables“ (DB API Marketplace, kostenlos). db-vendo-client wird von der DB geblockt.

## Wie dieses Projekt entstanden ist

Gebaut mit KI-Unterstützung (Claude Code), und zwar offen so dokumentiert: die
Commit-Nachrichten tragen einen `Co-Authored-By`-Hinweis, und `CLAUDE.md` enthält den
Projektkontext, mit dem das Modell arbeitet. Wer die Historie liest, soll sehen, wie
gearbeitet wurde.

Was dabei aus der Strecken- und Betriebskenntnis kam und ohne das nichts davon funktioniert
hätte:

- **Die Streckenlogik.** Dass der Signalblock von Coburg durchgehend bis Rödental reicht und
  dort und in Coburg die einzigen Kreuzungsmöglichkeiten liegen – gegengeprüft auf
  OpenRailwayMap. Eine erste Modellversion hatte Coburg Nord fälschlich als zweigleisig
  angenommen; das wurde korrigiert.
- **Welche Züge überhaupt zählen.** agilis, RB, STB und Bus stehen auf den Tafeln von Coburg
  und Sonneberg, befahren den Abschnitt Coburg–Sonneberg aber nicht. Sie versuchsweise als
  Konfliktzüge aufzunehmen hat das Modell messbar verschlechtert (07:20: 2,21 statt 2,16).
- **Der Vorrang des RE 29** vor RE 19/RE 28 – eine Beobachtung aus dem Betrieb, die in keinem
  Datensatz steht und in `[priority.lines]` eingetragen ist.
- **Die Hypothese, dass die vorausfahrenden Züge das Problem sind.** Daraus wurde
  `vorgaenger_delay`, das stärkste Feature des Modells – stärker als die DB-Prognose selbst.
- **Die Frage nach der Kaskade** (der ICE wirft mich zurück, danach hält mich der RE 29 ein
  zweites Mal auf). Sie hat eine echte Lücke im ML-Modell aufgedeckt, die vorher niemandem
  aufgefallen war.
- **Das Beharren auf Überprüfung.** Mehrere Zwischenergebnisse, die plausibel klangen, haben
  einer Messung nicht standgehalten und wurden verworfen – darunter ein selbst eingebautes
  Veto des Regelmodells, das sich über ein ganzes Jahr gerechnet als schädlich erwies
  (73 % Fehlalarme), und eine Untergrenze an der DB-Prognose, die auf einem Datensatz besser
  und auf dem anderen schlechter war und deshalb nicht übernommen wurde.

Die Arbeitsweise ist in dem Sinne der eigentliche Inhalt: nichts behaupten, was nicht
gemessen ist, und eigene Ideen verwerfen, wenn die Daten dagegen sprechen.
