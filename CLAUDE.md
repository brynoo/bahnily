# re19-watch – Projektkontext

## Ziel
Morgendliche Telegram-Warnung (Mo–Fr, Fenster 06:45–07:19), ob der RE 19 ab Dörfles-Esbach 07:20
Richtung Nürnberg auf dem eingleisigen Abschnitt Rödental – Coburg von einem Zug mit Vorrang
(v. a. ICE über die Einschleifung der SFS Nürnberg–Erfurt) aufgehalten wird – bevor der
DB Navigator die Verspätung zeigt.

## Architektur
- `simulate.py` – reine Logik, kein Netzwerk. Korridor = Punktkette Nord→Süd mit Fahrzeiten je
  Klasse (regional inkl. Halt / fast). Gegenzüge dürfen eingleisige Segmente nicht gleichzeitig
  belegen (+Puffer), gleiche Richtung = Headway, kein Überholen. Vorrang: ICE > IC > RE > RB,
  bei Gleichstand wer zuerst kommt. Nur MEIN Zug wird verschoben, gewartet an Kreuzungspunkten.
  `estimate()` simuliert zweimal (Soll + Prognose) und meldet nur die Differenz.
- `re19watch.py` – holt Tafeln über die DB-API-Marketplace-API „Timetables“ (`TimetablesClient`:
  /plan je Stunde + /fchg, XML; Env `DB_CLIENT_ID`/`DB_API_KEY`; EVA Dörfles-Esbach 8001484,
  Coburg 8001338; vergangene Stunden liefert die API nicht). `DbClient` (db-vendo-client) bleibt
  als `provider = "vendo"`, ist aber von der DB geblockt. klassifiziert
  Coburger Züge per Ziel/Herkunft (`[classify]` in config.toml), baut Züge, formatiert Telegram-HTML.
  CLI: `--dry-run`, `--force`, `--gate-only`.
- `collect.py` – Datensammler fürs spätere ML-Modell (`re19watch.py --collect`, launchd alle 5 min):
  Rohdaten gzip in `data/<tag>/raw/`, `plan.jsonl`, `changes.jsonl` (nur Änderungen),
  `predictions.jsonl` (Modell vs. DB, 06:00–07:45 + Alarmlauf). Stationen in `[collect]`.
- Betrieb lokal: `install_local.sh` kopiert nach `~/re19-watch` (macOS sperrt Dokumente für launchd),
  Jobs `de.degaso.re19watch` (Mo–Fr 07:00) und `de.degaso.re19collect`; Secrets in `.env`.
  Alarm: Telegram + ntfy (`NTFY_TOPIC`, ab 3 min). CallMeBot verworfen (Spam-Sperre).
- Cloud (Oracle Always Free, Ubuntu): `deploy/push.sh ubuntu@IP` (Code+.env hoch, systemd-Timer
  `re19-collect` alle 5 min, `re19-alarm` Mo–Fr 07:00), `deploy/pull.sh` (Daten nach `cloud-data/`),
  `deploy/status.sh`. Sammlung Werrabahn + Marschbahn 24/7.
- `config.toml` – Streckenmodell, Fahrzeiten, Zug, Prüffenster.
- `.github/workflows/re19-morning.yml` (vorhanden) – Cron (UTC, Sommer+Winter doppelt, Gate filtert),
  Secrets `DB_CLIENT_ID`, `DB_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Geprüft (OSM/OpenRailwayMap + bahn.de, 23.09.2026)
- Werrabahn 5121: Rödental → Herzogsweg Abzw (= "Abzweig SFS", Verbindungskurve 5126 zur SFS
  bei Esbacher See Abzw) → Dörfles-Esbach → Coburg Nord → Coburg, durchgehend eingleisig;
  Kreuzen nur Rödental und Coburg. Abstände ca. 2,0 / 0,5 / 2,6 / 1,1 km.
- RE 19 Richtung Nürnberg ab Dörfles-Esbach 06:20 und 07:20 (nicht 07:05!), Coburg an 07:26.
- Relevant um 07:20: ICE 501 (Coburg an 07:03 aus Leipzig), RE 29 4900 → Erfurt (Coburg ab 07:27,
  über die Einschleifung). RE 29 hat laut Beobachtung Vorrang vor RE 19/28 (`[priority.lines]`).
- DB blockt db-vendo-client (dbnav/db: OPS_BLOCKED, dbweb: 403) → Timetables-API, läuft.

## Offene Annahmen
- ICE-Fahrzeiten im Korridor (`fast`) weiter geschätzt. Nicht direkt messbar, da ICE an
  keiner Werrabahn-Station halten. Summe Coburg → Abzweig SFS = 3,5 min für 4,2 km aus
  dem Stand ist plausibel, aber unbelegt.
- ERLEDIGT 10/2026: Die Regionalzeiten sind an 7.338 Fahrten aus den 2025er-Daten
  geprüft und stimmen exakt mit config.toml (Median Soll = Ist): Rödental→Dörfles 3,
  Dörfles→Coburg Nord 3, Coburg Nord→Coburg 2, gesamt Dörfles→Coburg 5 min.
  Streuung p10/p90 = 2/3 bzw. 3/3 min, also sehr stabil.

## Nächste Schritte
1. Kalibrieren über predictions.csv (tatsächliche Abfahrt manuell eintragen).
2. Zweiter Check um ~07:14 (gemessen MAE 1,66 statt 1,84) – näher an der Abfahrt, aber
   weniger Vorlauf zum Losgehen.
3. 2024er Trainingsdaten ergänzen (verdoppelt die Datenmenge).

## Konventionen
- Tests: `python test_simulate.py` (ohne Internet). Nach Logikänderungen immer laufen lassen.
- Kommentare und Nutzertexte auf Deutsch.
