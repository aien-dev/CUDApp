#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# C5 sharing and recovery on a GB10 (docs/OMEGA.md row C5), one hold of the GB10 lock:
#   01 values   build/gpu_isolation_test --values-only: our two processes side by side, both right
#   02 mutant   build/gpu_isolation_test_mutant --values-only: red control, the values check must fail (exit 1)
#   03 hang     build/gpu_isolation_test (chip tier): never-ending kernel times out, latch refusal (red check),
#               recover, correct launches in the same process, stale command list refused (red check), two
#               processes, and a hang in one process beside a working one
#   04 cuda     build/c5_matrix c: our two processes beside a CUDA (PyTorch) process
#   05 pressure build/c5_matrix p: the same while the CUDA process holds 16 GiB of device memory
#   06 kill     build/c5_matrix k: our process SIGKILLed 500 ms into its never-ending launch beside a CUDA process
#   07 red      build/c5_matrix r: the CUDA process against a wrong expectation must exit 1
#   08 health   build/gpu_isolation_test --values-only again, after everything
# Kernel log lines added during the hold go to dmesg.txt; Xid / NVRM / watchdog lines to xid.txt.
#
#   tools/c5/build.sh                  (outside the lock, or the c5-sharing workflow artifact)
#   tools/c5/chip.sh OUTDIR            (takes /tmp/aien-gb10.lock itself)
# Exit 0 only when every step ended as expected.
set -u
OUT=${1:?usage: tools/c5/chip.sh OUTDIR}
B=${BUILD:-build}
PY=${PY:-$HOME/.venv/bin/python}
LOCK=${LOCK:-/tmp/aien-gb10.lock}
for f in "$B/gpu_isolation_test" "$B/gpu_isolation_test_mutant" "$B/c5_matrix"; do
    [ -x "$f" ] || { echo "NOT_RUN: $f is not built" >&2; exit 2; }
done
[ -x "$PY" ] || { echo "NOT_RUN: no Python at $PY for the CUDA tenant" >&2; exit 2; }
mkdir -p "$OUT"
S="$OUT/summary.txt"
{
    sha256sum "$B/gpu_isolation_test" "$B/gpu_isolation_test_mutant" "$B/c5_matrix"
    echo "driver $(cat /sys/module/nvidia/version 2>/dev/null || echo unknown)"
    echo "cuda tenant $("$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)' 2>&1 | tail -1)"
} > "$S"

exec 9>"$LOCK"
flock -w 900 9 || { echo "NOT_RUN: GB10 lock busy for 900 s" >&2; exit 2; }
echo "start $(date -u +%Y-%m-%dT%H:%M:%SZ) loadavg $(cat /proc/loadavg)" >> "$S"
base=$(sudo -n dmesg 2>/dev/null | wc -l)
fails=0
step() { # name want_rc command...
    name=$1; want=$2; shift 2
    log="$OUT/$name.log"
    { echo "loadavg $(cat /proc/loadavg)"; echo "cmd $*"; } > "$log"
    rc=0
    timeout -k 10 300 "$@" >> "$log" 2>&1 || rc=$?
    echo "exit $rc" >> "$log"
    v=$(grep -E '^C5 (isolation|matrix)' "$log" | tail -1)
    if [ "$rc" -eq "$want" ]; then r=ok; else r=FAIL; fails=$((fails + 1)); fi
    echo "$r $name exit $rc (want $want) ${v:-no verdict line}" >> "$S"
}
step 01-values   0 "$B/gpu_isolation_test" --values-only
step 02-mutant   1 "$B/gpu_isolation_test_mutant" --values-only
step 03-hang     0 "$B/gpu_isolation_test"
step 04-cuda     0 "$B/c5_matrix" "$(pwd)/$B/gpu_isolation_test" "$PY" c
step 05-pressure 0 "$B/c5_matrix" "$(pwd)/$B/gpu_isolation_test" "$PY" p
step 06-kill     0 "$B/c5_matrix" "$(pwd)/$B/gpu_isolation_test" "$PY" k
step 07-red      0 "$B/c5_matrix" "$(pwd)/$B/gpu_isolation_test" "$PY" r
# the red control counts only when the values check is what failed: wrong values reported, not a setup error
if grep -qE 'FAIL: process [12]: .* [1-9][0-9]* wrong values' "$OUT/02-mutant.log"; then
    echo "ok 02-mutant red because of wrong values" >> "$S"
else
    echo "FAIL 02-mutant did not fail on wrong values" >> "$S"; fails=$((fails + 1))
fi
step 08-health   0 "$B/gpu_isolation_test" --values-only
sudo -n dmesg -T 2>/dev/null | tail -n +"$((base + 1))" > "$OUT/dmesg.txt"
grep -E 'Xid|NVRM|watchdog|GPU has fallen|nvidia' "$OUT/dmesg.txt" > "$OUT/xid.txt" || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$OUT/compute-apps-after.txt" 2>&1 || true
flock -u 9
{
    echo "end $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "kernel log lines during the hold: $(wc -l < "$OUT/dmesg.txt"), Xid/NVRM/watchdog lines: $(wc -l < "$OUT/xid.txt")"
    echo "checks not as expected: $fails of 9 (8 steps and the mutant's cause)"
} >> "$S"
cat "$S"
[ "$fails" -eq 0 ]
