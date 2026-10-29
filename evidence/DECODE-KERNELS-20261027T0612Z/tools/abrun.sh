#!/bin/bash
# DIAGNOSTIC driver (DecodeKernels A/B, 2026-10-27; not product code). Usage: abrun.sh LOGDIR CMD...
# Runs a 18091 /health poller (CLOCK_REALTIME ns + in_flight, ~every 30 ms plus curl time) into LOGDIR/inflight.log for the
# whole life of CMD, so tag.py can drop every sample whose [t0, t1] overlaps in_flight > 0 or a poll gap > 200 ms.
set -u
D=$1; shift
mkdir -p "$D"
( while :; do
    t=$(date +%s%N); h=$(curl -s -m 0.5 127.0.0.1:18091/health)
    f=$(printf '%s' "$h" | grep -o '"in_flight":[0-9]*' | cut -d: -f2)
    echo "$t ${f:--1}"; sleep 0.03
  done ) > "$D/inflight.log" &
P=$!
sleep 0.2
"$@"
rc=$?
sleep 0.2
kill $P 2>/dev/null; wait $P 2>/dev/null
exit $rc
