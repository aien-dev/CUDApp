#!/usr/bin/env bash
# Gate-2 cubin load matrix: C/libcuda oracle (CUDApp cubin_probe) vs MAX
# DeviceExternalFunction probe, same images, same ABI (16x16x8 matmul).
# Verification tool only. Run inside an authorized window:
#   spark-window run ~/handoffs/max-g2/run_matrix.sh
# Load-only first; an image is launched only if both loaders accept it.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
CPROBE=${CPROBE:-$HERE/cubin_probe}   # CUDApp#202 build, sha256 627b0e9c…
MPROBE=${MPROBE:-$HERE/max_cubin_probe}
MATRIX=${MATRIX:-$HERE/variants/MATRIX.tsv}
M=16 K=16 N=8

if [ -z "${SPARK_WINDOW:-}" ] || [ -z "${SPARK_WINDOW_DIR:-}" ]; then
  echo "refused: SPARK_WINDOW/SPARK_WINDOW_DIR not set; run via spark-window run" >&2
  exit 3
fi
OUT=$SPARK_WINDOW_DIR/max-g2
mkdir -p "$OUT"
LOG=$OUT/raw.log
SUM=$OUT/results.tsv

# MAX: no pooled device memory manager (shared chip with resident servers),
# synchronous mode, log every function load.
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE=0
export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY=false
export MODULAR_DEVICE_CONTEXT_SYNC_MODE=1
export MODULAR_DEVICE_CONTEXT_LOG_LOAD_FUNCTION=1

{
  echo "# window=$SPARK_WINDOW start=$(date -u +%FT%TZ) host=$(hostname)"
  echo "# driver: $(head -n1 /proc/driver/nvidia/version 2>/dev/null)"
  echo "# MemAvailable: $(awk '/MemAvailable/{print $2" kB"}' /proc/meminfo)"
  sha256sum "$CPROBE" "$MPROBE" "$MATRIX" "$0"
} >"$LOG" 2>&1

printf 'name\tfn\tmode\texpect\tsha_ok\tc_load\tc_run\tmax_load\tmax_run\n' >"$SUM"

step() { # label cmd... ; echo rc
  local label=$1; shift
  { echo "### $label: $*"; "$@"; echo "### rc=$?"; } >>"$LOG" 2>&1
  tail -n1 "$LOG" | sed 's/### rc=//'
}

tail -n +2 "$MATRIX" | while IFS=$'\t' read -r name fn mode expect sha isolates; do
  img=$HERE/variants/$name.cubin
  got=$(sha256sum "$img" | cut -d' ' -f1)
  ok=yes; [ "$got" = "$sha" ] || ok=NO
  echo "## $name fn=$fn mode=$mode expect=[$expect] sha=$got ok=$ok isolates=[$isolates]" >>"$LOG"
  cl=- cr=- ml=- mr=-
  if [ "$ok" != yes ]; then
    echo "## SKIP $name: sha mismatch" >>"$LOG"
  elif [ "$mode" = reject ]; then
    cl=$(step "$name C expect-reject" "$CPROBE" --expect-reject "$img" "$fn" $M $K $N)
    ml=$(step "$name MAX reject" "$MPROBE" "$img" "$fn" reject)
  else
    cl=$(step "$name C load-only" "$CPROBE" --load-only "$img" "$fn" $M $K $N)
    ml=$(step "$name MAX loadonly" "$MPROBE" "$img" "$fn" loadonly)
    if [ "$cl" = 0 ] && [ "$ml" = 0 ]; then
      cr=$(step "$name C run" "$CPROBE" "$img" "$fn" $M $K $N)
      mr=$(step "$name MAX run" "$MPROBE" "$img" "$fn" run)
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$fn" "$mode" "$expect" "$ok" "$cl" "$cr" "$ml" "$mr" >>"$SUM"
done

echo "# end=$(date -u +%FT%TZ) MemAvailable: $(awk '/MemAvailable/{print $2" kB"}' /proc/meminfo)" >>"$LOG"
cat "$SUM"
echo "raw: $LOG"
