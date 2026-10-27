#!/usr/bin/env bash
# DecodeKernels chip hold (Train/PROTOCOL.md rules 1-8): flock -w 900, <=15 min, 18091 /health before/after,
# BOARD GPU-START/END, compute apps logged at start and end (must be 79557 only). Usage: hold.sh LABEL LOGDIR CMD...
set -u
label=$1; logdir=$2; shift 2
mkdir -p "$logdir"
B=/home/drakestapleton/handoffs/BOARD.md
ts() { date -u +%FT%TZ; }
exec 9>/tmp/aien-gb10.lock
if ! flock -w 900 9; then echo "$(ts) DecodeKernels: NOT_RUN $label (lock not free in 900 s)" >> $B; exit 75; fi
{ echo "head $(git -C "$PWD" rev-parse HEAD 2>/dev/null) dirty=$(git -C "$PWD" status --short 2>/dev/null | wc -l)"; echo "cwd $PWD"; echo "cmd $*"; } > "$logdir/hold.meta"
curl -s -m 5 127.0.0.1:18091/health > "$logdir/health-before.json"; echo >> "$logdir/health-before.json"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$logdir/apps-before.txt" 2>&1
echo "$(ts) DecodeKernels: GPU-START $label (flock /tmp/aien-gb10.lock, <=15 min, beside live 35B; log $logdir)" >> $B
timeout 840 "$@" > "$logdir/run.log" 2>&1; rc=$?
curl -s -m 5 127.0.0.1:18091/health > "$logdir/health-after.json"; echo >> "$logdir/health-after.json"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader > "$logdir/apps-after.txt" 2>&1
apps=$(cut -d, -f1 "$logdir/apps-after.txt" | tr -d ' ' | paste -sd' ')
st=$(grep -o '"status":"[a-z]*"' "$logdir/health-after.json" | head -1)
echo "$(ts) DecodeKernels: GPU-END $label exit $rc; compute apps after: $apps ; 18091 $st" >> $B
echo "rc=$rc apps=$apps $st"
exit $rc
