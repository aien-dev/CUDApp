#!/bin/sh
# yardstick/cuda-vecadd/run.sh: the head-to-head, one window, one receipt.
#
#   run.sh OUTDIR
#
# Builds the CUDA half with nvcc, runs it, runs CUDA++'s hello_gpu on the same work, and writes both
# logs plus the two output hashes into OUTDIR. Run it inside a spark-window that holds the chip:
#
#   spark-window open cudapp-cuda-h2h --gb10 exclusive --quiet exclusive --wait
#   spark-window run sh yardstick/cuda-vecadd/run.sh evidence/cuda-head-to-head
#   spark-window close
#
# The CUDA++ side is built outside the window (build.sh is not part of this file): a build inside a
# measurement window spends the window on the compiler.
set -e
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$DIR/../.." && pwd)
OUT=${1:?usage: run.sh OUTDIR}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
HELLO=${HELLO:-$ROOT/build/hello_gpu}

mkdir -p "$OUT"
cd "$ROOT"

echo "== window ==" | tee "$OUT/window.txt"
{ date -u +"utc %Y-%m-%dT%H:%M:%SZ"; echo "window ${SPARK_WINDOW:-none}"; echo "dir ${SPARK_WINDOW_DIR:-none}"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true; } | tee -a "$OUT/window.txt"

echo "== build the CUDA half ==" | tee -a "$OUT/cuda.log"
$NVCC --version | tail -2 | tee -a "$OUT/cuda.log"
$NVCC -O2 -arch=sm_121 -o "$OUT/vecadd_cuda" "$DIR/vecadd.cu" 2>&1 | tee -a "$OUT/cuda.log"

echo "== CUDA's own stack, on the GB10 ==" | tee -a "$OUT/cuda.log"
"$OUT/vecadd_cuda" "$OUT/cuda_output.bin" 2>&1 | tee -a "$OUT/cuda.log"
sha256sum "$OUT/cuda_output.bin" | tee -a "$OUT/cuda.log"

echo "== CUDA++, the same work ==" | tee -a "$OUT/cudapp.log"
"$HELLO" 2>&1 | tee -a "$OUT/cudapp.log"

echo "== the two output hashes ==" | tee -a "$OUT/summary.txt"
sha256sum "$OUT/cuda_output.bin" >> "$OUT/summary.txt"
grep -h "^output sha256" "$OUT/cudapp.log" >> "$OUT/summary.txt" || true
grep -h "^work:" "$OUT/cuda.log" "$OUT/cudapp.log" >> "$OUT/summary.txt" || true
cat "$OUT/summary.txt"
