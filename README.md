# re19-watch

Jeden Werktag um ~06:50 und ~07:05 eine Telegram-Nachricht: Wird mein **RE 19 ab Dörfles-Esbach 07:20** auf dem eingleisigen Abschnitt Rödental – Coburg durch einen Zug mit Vorrang aufgehalten – *bevor* der DB Navigator es anzeigt?

## Warum das funktionieren kann

Die DB-Prognose für den RE 19 reagiert erst, wenn der RE selbst später dran ist. Die Verspätung eines ICE, der gleich den eingleisigen Abschnitt Coburg – Einschleifung belegt, ist aber oft **schon bekannt**. re19-watch rechnet mit deinen Fahrzeiten aus, ob sich die beiden Belegungen überschneiden und wer warten muss.

Zweimal simuliert: einmal mit Soll-Zeiten, einmal mit Prognosen. Gemeldet wird nur die Differenz, damit planmäßige Kreuzungen nicht als Verspätung auftauchen.

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

## Kalibrieren (das macht den Unterschied)

- `config.toml` → `[[corridor.segment]]`: deine gemessenen Fahrzeiten eintragen (Abfahrt A bis Abfahrt B).
- `crossing_points`: auf OpenRailwayMap prüfen, wo Züge sich wirklich kreuzen können, und wo genau die ICE-Einschleifung liegt.
- `predictions.csv` (bei lokalen Läufen): Spalte „tatsächlich" selbst füllen. Nach 2–3 Wochen siehst du, wie oft die Warnung stimmte.
- Kommt ein 🔧 Kalibrierhinweis, sieht das Modell schon im Soll-Fahrplan einen Konflikt → Puffer/Fahrzeiten zu streng.

## Grenzen

- Keine Stellwerks-/Blockdaten (nicht öffentlich) – es ist eine Heuristik.
- ICE halten nicht in Dörfles-Esbach, ihre Durchfahrt wird aus Coburg-Zeiten hochgerechnet.
- Nur dein Zug wird verschoben, Folgeverspätungen anderer Züge nicht.
- GitHub-Cronjobs können sich einige Minuten verspäten.
- Daten: offizielle DB-API „Timetables“ (DB API Marketplace, kostenlos). db-vendo-client wird von der DB geblockt.
