# re19-watch

Hobbyprojekt. Ich fahre morgens mit dem RE 19 ab Dörfles-Esbach um 07:20 Richtung Nürnberg.

Mein Problem war: ich bin fast jeden Morgen zum Bahnhof gerannt, und der Zug kam dann
trotzdem zu spät. Der DB Navigator zeigt die Verspätung nämlich erst an, wenn mein Zug
selbst schon dran ist. Bis dahin stehe ich am Bahnsteig.

## Die Idee

Zwischen Rödental und Coburg ist die Strecke eingleisig. Kreuzen geht nur in Rödental und
in Coburg. Wenn also ein Zug mit Vorrang da gerade drauf ist, muss ich warten – egal ob
die DB das schon anzeigt oder nicht.

Und: dass dieser Zug Verspätung hat, **weiß man oft schon vorher**. Vor allem bei den ICE,
die über die Einschleifung zur Schnellfahrstrecke Nürnberg–Erfurt fahren.

Also prüfe ich um 07:05: Haben die Züge, die den eingleisigen Abschnitt vor mir belegen,
schon Verspätung – und fallen sie dadurch genau in mein Zeitfenster? Wenn ja, bekomme ich
eine Telegram-Nachricht und weiß, dass ich später losgehen kann.

## Wie es funktioniert

Zwei Teile:

- **Streckenmodell** (`simulate.py`): rechnet aus, wer wann auf welchem Abschnitt ist.
  Gegenzüge können nicht gleichzeitig drauf, Vorrang ist ICE > IC > RE > RB. Simuliert
  wird zweimal, mit Soll-Zeiten und mit den aktuellen Prognosen – gezählt wird nur die
  Differenz, sonst würde jede planmäßige Kreuzung als Verspätung gemeldet.
- **ML-Modell** (`ml/`): trainiert auf einem Jahr Verspätungsdaten von 2025. Es schätzt die
  tatsächliche Abfahrtsverspätung aus dem, was um 07:05 bekannt ist.

Die Zahl in der Nachricht kommt vom ML-Modell. Das Streckenmodell sagt dazu, *welcher* Zug
*wo* kreuzt – das kann das ML-Modell nicht.

## Ergebnisse

Gemessen auf einem Jahr Echtdaten. Trainiert wurde immer nur mit Daten *vor* dem
Testzeitraum, sonst wäre es geschummelt. Zahl = durchschnittlicher Fehler in Minuten:

| | mein Modell | DB-Prognose |
|---|---|---|
| **der 07:20-Zug** (n=216) | **2,27** | 3,03 |
| alle Abfahrten (n=6465) | 1,69 | 2,22 |
| wenn es wirklich ≥5 min wurden (n=1209) | 4,88 | 6,37 |
| eigene Messungen 2026 (n=189) | 1,87 | 2,07 |

Der laute Alarm geht ab 4 min los, das sind etwa 31 im Jahr. Bei 77 % davon waren es
wirklich mindestens 4 min. Die DB-Prognose ist da etwas treffsicherer (83 %), schlägt aber
nur 19 Mal im Jahr an und erkennt dadurch nur 32 % der Verspätungen ab 5 min – mein Modell
46 %. Mir ist das lieber so: ein Fehlalarm kostet mich nichts, ein verpasster Zug schon.

Nachrechnen: `python ml/zielpruefung.py`

## Was ich dabei gelernt habe

Das stärkste Feature ist nicht die DB-Prognose, sondern die Verspätung des Zuges, der kurz
vor mir über den Abschnitt fährt. Und zwar fast nur des **Gegenzuges** – in meiner Richtung
fährt der Stundentakt, da wird der Abstand nie knapp. Der Gegenzug muss aber kreuzen.

Ein Fehler, den ich lange nicht gesehen habe: die Bahnhofstafel wird nach Soll-Zeit
gefiltert. Alle drei Konfliktfälle meines Zuges in 2025 waren derselbe Zug, ICE 1604 mit
Soll-Abfahrt 06:43 und über 35 min Verspätung – real also mitten in meinem Weg, aber das
Zeitfenster fing erst 06:45 an. Der Zug war für mein Programm nie da.

Und: ich hatte eingebaut, dass das Streckenmodell die Schätzung nach oben überstimmen darf.
Über das ganze Jahr nachgerechnet war das schlechter als ohne – es schlägt bei 14 % der
Züge an, davon sind 73 % Fehlalarme. Ist rausgeflogen.

## Grenzen

- Etwa die Hälfte der schweren Verspätungen entsteht erst nach 07:05. Die kann man um 07:05
  nicht wissen.
- Bei den Konfliktfällen schätzt das Modell zu niedrig: sagt +4, wenn +11 kommt. Richtung
  stimmt, Höhe nicht.
- Keine Stellwerksdaten, die gibt es nicht öffentlich. Es ist eine Rechnung, keine Messung.
- Läuft bei mir auf einem Mac per launchd. Wenn der aus ist, kommt nichts.

## Benutzen

```bash
cp .env.example .env     # DB-API-Keys und Telegram-Token eintragen
pip install -r requirements.txt
python re19watch.py --mock      # Testlauf mit erfundenem ICE-Konflikt
python test_simulate.py         # Tests, ohne Internet
```

Fahrzeiten und Streckenmodell stehen in `config.toml`. Für eine andere Strecke muss man die
anpassen. Das ML-Modell neu zu trainieren braucht die Daten von der Mobilithek, siehe
[`ml/README.md`](ml/README.md).

## Daten

- Live: DB API Marketplace, API „Timetables" (kostenlos mit Account)
- Training: „Geparste deutschlandweite Verspätungsdaten" über die
  [Mobilithek](https://mobilithek.info), Lizenz ODbL. Quellenvermerk: Bahn-Vorhersage,
  Deutsche Bahn, OpenStreetMap, Trainline, DELFI
- Strecke gegengeprüft auf [OpenRailwayMap](https://www.openrailwaymap.org)

Code unter MIT, siehe [LICENSE](LICENSE).

## KI

Gebaut mit Claude Code, steht auch so in den Commits. Von mir kam die Streckenkenntnis:
dass der Signalblock durchgehend von Coburg bis Rödental reicht, dass agilis, RB und STB
nur auf den Tafeln stehen aber den Abschnitt nicht befahren, dass der RE 29 Vorrang vor dem
RE 19 hat, und die Idee mit den vorausfahrenden Zügen. `CLAUDE.md` ist der Projektkontext,
mit dem das Modell gearbeitet hat.
