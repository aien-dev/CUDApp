#!/bin/bash
# GemmGeneral chip hold for the served-shape A/B (AB-PLAN.md): flock, 18091 /health and compute apps before and after,
# BOARD GPU-START/GPU-END, 14-minute cap. No new hold while the last anchored Main WINDOW-* line on BOARD is WINDOW-START,
# and none while 18091 is not "status":"ok". Bounded: 3 h of waiting, then NOT_RUN on BOARD.
# Pattern copied from handoffs/GemmTune/hold.sh.
# usage: hold.sh <label> <logdir> <command...>
set -u
label=$1; logdir=$2; shift 2
mkdir -p "$logdir"
B=/home/drakestapleton/handoffs/BOARD.md
window_open() { grep -E '^[0-9TZ:.-]+ Main: WINDOW-(START|END|ABORTED)' $B | tail -1 | grep -q 'WINDOW-START'; }
live_ok() { curl -s -m 5 127.0.0.1:18091/health | grep -q '"status":"ok"'; }
exec 9>/tmp/aien-gb10.lock
deadline=$(( $(date +%s) + 10800 ))
while :; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "$(date -u +%FT%TZ) GemmGeneral: NOT_RUN $label (window open or 18091 not ok for 3 h)" >> $B; exit 75
  fi
  if window_open || ! live_ok; then sleep 30; continue; fi
  if ! flock -w 900 9; then continue; fi
  if window_open || ! live_ok; then flock -u 9; sleep 30; continue; fi
  break
done
{
  echo "# $label start $(date -u +%FT%TZ) load $(cat /proc/loadavg)"
  echo "# health before: $(curl -s -m 5 127.0.0.1:18091/health)"
  echo "# compute apps before: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
} > "$logdir/hold.log"
echo "$(date -u +%FT%TZ) GemmGeneral: GPU-START $label (flock /tmp/aien-gb10.lock, <=14 min, beside live 35B; log $logdir)" >> $B
timeout 840 "$@" > "$logdir/run.log" 2>&1
rc=$?
{
  echo "# exit $rc end $(date -u +%FT%TZ)"
  echo "# health after: $(curl -s -m 5 127.0.0.1:18091/health)"
  echo "# compute apps after: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
} >> "$logdir/hold.log"
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | sort -u | tr '\n' ' ')
echo "$(date -u +%FT%TZ) GemmGeneral: GPU-END $label exit $rc; compute apps after: ${apps}; 18091 $(curl -s -m 5 127.0.0.1:18091/health | grep -o '"status":"[a-z]*"')" >> $B
exit $rc
