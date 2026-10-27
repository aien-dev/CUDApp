#!/bin/bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# tools/run_d7_same_bits_chip.sh JOB RUNS OUTDIR -- D7 on the GB10: run one job RUNS times, each a separate
# process, with no determinism setting, then compare the printed output hashes (tools/d7_same_bits_check.sh).
#
#   JOB   hello    build/hello_gpu                                c = a + b, 2^20 floats (no reduction; the D7 baseline)
#         gemv     build/gpu_bf16_gemv_test --reps 1 --no-evict    bf16 GEMV 151936 x 2048, the lm_head (per-lane FFMA chain,
#                                                                 then a SHFL.DOWN tree across the warp)
#         rmsnorm  build/gpu_elementwise_test                      rmsnorm 4x2048, 1x256, 2x8192 (SHFL.DOWN, BAR.SYNC and
#                                                                 shared memory across the CTA); the other ops print no hash
#         dattn    build/gpu_decode_attn_test --reps 1 --no-evict  decode attention 32 q / 4 kv heads x 128, contexts 1 .. 32768
#                                                                 (softmax max/sum reductions, the cross-split combine)
# Each binary prints "<label> output sha256 <hex>" for the chip output buffer of its first call (tests/out_sha.h).
#
# It opens the GPU, so the caller holds the shared lock:  flock -w 900 /tmp/aien-gb10.lock <this script> JOB RUNS OUTDIR
# Writes OUTDIR/JOB/run-NN.log (stdout+stderr of each run, headed by the command, UTC start, and followed by the exit
# code and UTC end), OUTDIR/JOB/env.txt (git HEAD, binary sha256, driver, the environment check) and
# OUTDIR/JOB/check.txt (the comparison). Exit 0 only when every run exited 0 and the checker says PASS.
set -u
JOB=${1:?usage: run_d7_same_bits_chip.sh JOB RUNS OUTDIR}; RUNS=${2:?RUNS}; OUT=${3:?OUTDIR}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
case "$JOB" in
    hello)   CMD=("$ROOT/build/hello_gpu") ;;
    gemv)    CMD=("$ROOT/build/gpu_bf16_gemv_test" --reps 1 --no-evict) ;;
    rmsnorm) CMD=("$ROOT/build/gpu_elementwise_test") ;;
    dattn)   CMD=("$ROOT/build/gpu_decode_attn_test" --reps 1 --no-evict) ;;
    *) echo "unknown job $JOB (hello|gemv|rmsnorm|dattn)" >&2; exit 2 ;;
esac
[ "$RUNS" -ge 2 ] 2>/dev/null || { echo "RUNS must be >= 2" >&2; exit 2; }
[ -x "${CMD[0]}" ] || { echo "NOT_RUN: ${CMD[0]} not built" >&2; exit 2; }
D="$OUT/$JOB"; mkdir -p "$D"
# No setting: refuse to run when any determinism or CUDA switch is in the environment.
SWITCHES=$(env | grep -E '^(CUBLAS_|CUDA_|CUDNN_|NVIDIA_TF32|TORCH_|PYTORCH_|OMEGA_.*DETERMIN)' || true)
{
    echo "job: $JOB"
    echo "command: ${CMD[*]}"
    echo "git HEAD: $(git -C "$ROOT" rev-parse HEAD)"
    echo "git status: $(git -C "$ROOT" status --porcelain | wc -l) changed files"
    echo "binary sha256: $(sha256sum "${CMD[0]}" | cut -d' ' -f1)"
    echo "driver: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1)"
    echo "kernel: $(uname -r)"
    echo "determinism/CUDA switches in the environment: ${SWITCHES:-none}"
} > "$D/env.txt"
[ -z "$SWITCHES" ] || { echo "NOT_RUN: a switch is set: $SWITCHES" >&2; exit 2; }
fails=0
for i in $(seq -w 1 "$RUNS"); do
    L="$D/run-$i.log"
    { echo "# command: ${CMD[*]}"; echo "# utc start: $(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"; } > "$L"
    "${CMD[@]}" >> "$L" 2>&1; rc=$?
    { echo "# exit: $rc"; echo "# utc end: $(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"; } >> "$L"
    [ "$rc" -eq 0 ] || fails=$((fails + 1))
    echo "run $i exit $rc: $(grep ' output sha256 ' "$L" | tr '\n' ';')"
done
"$ROOT/tools/d7_same_bits_check.sh" "$D"/run-*.log > "$D/check.txt"; crc=$?
cat "$D/check.txt"
echo "runs failed: $fails of $RUNS"
[ "$fails" -eq 0 ] && [ "$crc" -eq 0 ]
