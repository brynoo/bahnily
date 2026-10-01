#!/bin/zsh
# Läuft auf dem MAC. Zeigt, ob auf der VM alles läuft.
HOST="${1:?Aufruf: deploy/status.sh ubuntu@<IP>}"
KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
ssh -i "$KEY" "$HOST" "systemctl list-timers 're19-*' --no-pager; echo; journalctl -u re19-collect --since '-20 min' --no-pager -o cat | tail -6; echo; du -sh ~/re19-watch/data ~/re19-watch/datamarschbahn; df -h ~ | tail -1"
