#!/bin/bash
# Läuft AUF DER VM (Ubuntu 22.04/24.04). Richtet Python, systemd-Timer und Zeitzone ein.
# Wird von deploy/push.sh automatisch aufgerufen.
set -euo pipefail
APP="$HOME/re19-watch"
cd "$APP"

sudo timedatectl set-timezone Europe/Berlin
if ! python3 -c "import venv, ensurepip" 2>/dev/null; then
  sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv
fi
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
chmod 600 .env
.venv/bin/python test_simulate.py > /dev/null && echo "Tests grün"

unit() { sudo tee "/etc/systemd/system/$1" > /dev/null; }

unit re19-collect.service <<U
[Unit]
Description=re19-watch Datensammlung (Werrabahn + Marschbahn)
After=network-online.target
[Service]
Type=oneshot
User=$USER
WorkingDirectory=$APP
ExecStart=$APP/run.sh --collect
TimeoutStartSec=240
U
unit re19-collect.timer <<U
[Unit]
Description=re19-watch Datensammlung alle 5 min
[Timer]
OnCalendar=*:0/5
Persistent=false
[Install]
WantedBy=timers.target
U

unit re19-alarm.service <<U
[Unit]
Description=re19-watch Morgen-Check RE 19
After=network-online.target
[Service]
Type=oneshot
User=$USER
WorkingDirectory=$APP
ExecStart=$APP/run.sh
U
unit re19-alarm.timer <<U
[Unit]
Description=re19-watch Morgen-Check Mo-Fr 07:00
[Timer]
OnCalendar=Mon..Fri 07:00 Europe/Berlin
[Install]
WantedBy=timers.target
U

sudo systemctl daemon-reload
sudo systemctl enable --now re19-collect.timer
if [ "${WITH_ALARM:-1}" = "1" ]; then sudo systemctl enable --now re19-alarm.timer; else sudo systemctl disable --now re19-alarm.timer 2>/dev/null || true; fi
systemctl list-timers 're19-*' --no-pager
