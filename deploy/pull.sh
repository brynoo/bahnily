#!/bin/zsh
# Läuft auf dem MAC. Holt die gesammelten Daten von der VM nach ./cloud-data/ (nur Neues).
set -e
HOST="${1:?Aufruf: deploy/pull.sh ubuntu@<IP>}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
mkdir -p "$SRC/cloud-data"
rsync -az --info=stats1 -e "ssh -i $KEY" "$HOST":re19-watch/{data,datamarschbahn} "$SRC/cloud-data/"
du -sh "$SRC/cloud-data"/*
