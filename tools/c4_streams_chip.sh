#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# C4 streams on a GB10 (docs/OMEGA.md row C4): N fresh processes of build/gpu_streams_test, the test as imported from
# omega-mv, unchanged. Each run is a fresh session, fresh channels and fresh subcontexts, and does: calibration, overlap
# medians of 5 in both barrier modes, the serial red check, the event with the wait (both barrier modes) and the
# control without the wait. N runs give N overlap ratios per barrier mode (the distribution) and N red/green pairs.
#
#   make build/gpu_streams_test        (outside the lock)
#   tools/c4_streams_chip.sh OUTDIR [N] (takes /tmp/aien-gb10.lock itself, one hold for the set)
#
# Writes OUTDIR/run-NN.log (load average before, exit code after) and OUTDIR/summary.txt. Exit 0 only when all N passed.
set -eu
OUT=${1:?usage: tools/c4_streams_chip.sh OUTDIR [N]}
N=${2:-10}
BIN=${BIN:-build/gpu_streams_test}
LOCK=${LOCK:-/tmp/aien-gb10.lock}
[ -x "$BIN" ] || { echo "NOT_RUN: $BIN is not built" >&2; exit 2; }
mkdir -p "$OUT"
{
    echo "bin $(sha256sum "$BIN")"
    echo "driver $(cat /sys/module/nvidia/version 2>/dev/null || echo unknown)"
    echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/summary.txt"

exec 9>"$LOCK"
flock -w 900 9 || { echo "NOT_RUN: GB10 lock busy for 900 s" >&2; exit 2; }
i=1
while [ "$i" -le "$N" ]; do
    log="$OUT/run-$(printf %02d "$i").log"
    echo "loadavg $(cat /proc/loadavg)" > "$log"
    rc=0
    timeout 120 "$BIN" >> "$log" 2>&1 || rc=$?
    echo "exit $rc" >> "$log"
    echo "run $i exit $rc $(grep -E '^C4 streams chip' "$log" || echo 'no verdict')" >> "$OUT/summary.txt"
    i=$((i + 1))
done
flock -u 9
echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/summary.txt"
pass=$(grep -c '^run .* exit 0 C4 streams chip: PASS' "$OUT/summary.txt" || true)
echo "runs passing: $pass of $N" | tee -a "$OUT/summary.txt"
[ "$pass" -eq "$N" ]
