# re19-watch

Jeden Werktag um 07:05 eine Telegram-Nachricht: Wird mein **RE 19 ab Dörfles-Esbach 07:20** auf dem eingleisigen Abschnitt Rödental – Coburg durch einen Zug mit Vorrang aufgehalten – *bevor* der DB Navigator es anzeigt?

## Warum das funktionieren kann

Die DB-Prognose für den RE 19 reagiert erst, wenn der RE selbst später dran ist. Die Verspätung eines ICE, der gleich den eingleisigen Abschnitt Coburg – Einschleifung belegt, ist aber oft **schon bekannt**. re19-watch rechnet mit deinen Fahrzeiten aus, ob sich die beiden Belegungen überschneiden und wer warten muss.

Zweimal simuliert: einmal mit Soll-Zeiten, einmal mit Prognosen. Zusätzlich schätzt ein
ML-Modell die tatsächliche Verspätung aus DB-Prognose, Oberlauf, Konfliktzügen und dem
Zug, der unmittelbar vor mir über den eingleisigen Abschnitt fährt.

Für die ausgegebene Zahl wird das **ML-Modell direkt** verwendet, nicht als `max()` mit der
DB verrechnet – gemessen ist das die bessere Variante, weil das Modell auch korrekte
Abwärtskorrekturen liefert. Zwei Sicherungen bleiben: eine Plausibilitätsgrenze gegen
Ausreißer, und ein Veto des Regelmodells **nur nach oben** – hat die Simulation eine
konkrete Blockade auf dem eingleisigen Abschnitt berechnet, darf das ML-Modell nicht
darunter gehen (es kennt die Gleisbelegung nicht). Fehlt das Modell oder eine
ML-Abhängigkeit, läuft der physikalische Regelcheck als Rückfallebene.

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
