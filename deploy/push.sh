#!/bin/zsh
# Läuft auf dem MAC. Kopiert Code, config und .env auf die VM und richtet sie ein.
#   deploy/push.sh ubuntu@<IP>              Sammlung + Alarm 07:00 auf der VM
#   WITH_ALARM=0 deploy/push.sh ubuntu@<IP>  nur Sammlung (Alarm bleibt auf dem Mac)
# Nach jeder Code-/config-Änderung erneut ausführen. Gesammelte Daten auf der VM bleiben erhalten.
set -e
HOST="${1:?Aufruf: deploy/push.sh ubuntu@<IP>}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new)
"${SSH[@]}" "$HOST" "mkdir -p ~/re19-watch"
rsync -az -e "${SSH[*]}" \
  "$SRC"/{re19watch.py,simulate.py,collect.py,test_simulate.py,config.toml,requirements.txt,run.sh,.env} \
  "$SRC/deploy/server_setup.sh" "$HOST":re19-watch/
"${SSH[@]}" "$HOST" "cd ~/re19-watch && WITH_ALARM=${WITH_ALARM:-1} bash server_setup.sh"
