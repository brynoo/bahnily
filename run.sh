#!/bin/bash
# Startet den Morgen-Check mit den Zugangsdaten aus .env (für launchd/cron).
cd "$(dirname "$0")"
source ./.env
exec ./.venv/bin/python re19watch.py "$@"
