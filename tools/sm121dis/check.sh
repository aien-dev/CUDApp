#!/bin/sh
# check.sh <sm121dis> <kernel-dir> <work-dir>: Omega's disassembler against nvdisasm (the oracle) on every <kernel-dir>/*.bin.
# Per kernel: PASS when every word's text is the same (spacing collapsed), else FAIL with the first differences.
# Exit 0 only when every kernel passes; 2 when nvdisasm or a kernel is missing. nvdisasm is a decoder used offline as
# the reference (docs/OMEGA.md C6); nothing it prints is used anywhere else.
set -u
DIS=$1; DIR=$2; WORK=$3
HERE=$(cd "$(dirname "$0")" && pwd)
NVDISASM=${NVDISASM:-/usr/local/cuda/bin/nvdisasm}
[ -x "$NVDISASM" ] || { echo "VERDICT NOT_RUN: $NVDISASM missing"; exit 2; }
ls "$DIR"/*.bin >/dev/null 2>&1 || { echo "VERDICT NOT_RUN: no kernels in $DIR"; exit 2; }
mkdir -p "$WORK"
echo "oracle: $("$NVDISASM" --version | grep -o "release [0-9.]*, V[0-9.]*")"
k=0; bad=0; words=0
for f in "$DIR"/*.bin; do
    b=$(basename "$f" .bin)
    "$NVDISASM" -b SM121 -hex "$f" > "$WORK/$b.nv" 2>&1 || { echo "[FAIL] $b: nvdisasm refused it"; head -3 "$WORK/$b.nv"; bad=$((bad + 1)); k=$((k + 1)); continue; }
    sh "$HERE/oracle_pairs.sh" "$WORK/$b.nv" > "$WORK/$b.want"
    "$DIS" --pairs "$f" | sed 's/[[:space:]]\{1,\}/ /g' > "$WORK/$b.got"
    n=$(wc -l < "$WORK/$b.want"); words=$((words + n))
    if [ "$n" -gt 0 ] && cmp -s "$WORK/$b.want" "$WORK/$b.got"; then echo "[PASS] $b: $n words"
    else echo "[FAIL] $b: $(diff "$WORK/$b.want" "$WORK/$b.got" | grep -c '^<') of $n words differ"; diff "$WORK/$b.want" "$WORK/$b.got" | head -6; bad=$((bad + 1)); fi
    k=$((k + 1))
done
echo "kernels: $k, words: $words, failed kernels: $bad"
[ $bad = 0 ] && { echo "VERDICT PASS"; exit 0; }
echo "VERDICT FAIL"; exit 1
