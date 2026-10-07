#!/bin/zsh
# Kopiert den Check nach ~/re19-watch (außerhalb von "Dokumente", das macOS für
# Hintergrund-Jobs sperrt) und (re)lädt die launchd-Jobs (Check Mo–Fr 07:05, Sammler alle 5 min).
# Nach Code-/Config-Änderungen erneut ausführen.
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
DST="$HOME/re19-watch"
mkdir -p "$DST"
cp "$SRC"/{re19watch.py,simulate.py,collect.py,config.toml,requirements.txt,run.sh} "$DST"/
mkdir -p "$DST/ml"
cp "$SRC/ml/model_werrabahn.joblib" "$DST/ml/"
if [ ! -f "$SRC/.env" ]; then
  echo "Es fehlt $SRC/.env – bitte .env.example kopieren und ausfüllen:" >&2
  echo "  cp .env.example .env && \$EDITOR .env" >&2
  exit 1
fi
install -m 600 "$SRC/.env" "$DST/.env"
[ -d "$DST/.venv" ] || python3 -m venv "$DST/.venv"
"$DST/.venv/bin/pip" install -q -r "$DST/requirements.txt"
for job in re19watch re19collect; do
  # Die Plists im Repo enthalten den Platzhalter __INSTALL_DIR__, damit dort kein
  # fester Benutzerpfad steht und die Installation auf jedem Mac funktioniert.
  sed "s#__INSTALL_DIR__#$DST#g" "$SRC/launchd/de.degaso.$job.plist" > "$HOME/Library/LaunchAgents/de.degaso.$job.plist"
  launchctl bootout "gui/$(id -u)/de.degaso.$job" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/de.degaso.$job.plist"
done
echo "installiert in $DST – Log: $DST/re19watch.log"
