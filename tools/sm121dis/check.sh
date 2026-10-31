#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# check.sh <sm121dis> <kernel-dir> <work-dir>: Omega's disassembler against nvdisasm (the oracle) on every <kernel-dir>/*.bin.
# Per kernel: PASS when every word's text is the same (spacing collapsed), else FAIL with the first differences.
# Exit 0 only when every kernel passes; 2 when nvdisasm or a kernel is missing. nvdisasm is a decoder used offline as
# the reference (docs/OMEGA.md C6); nothing it prints is used anywhere else.
# Kernels are checked SM121DIS_JOBS at a time (default: every CPU this process may use); each writes
# <work-dir>/<name>.res and the report below reads them back in kernel order, so it is the same at any job count.
# A kernel without a result fails.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
NVDISASM=${NVDISASM:-/usr/local/cuda/bin/nvdisasm}

if [ "${1:-}" = --one ]; then
    # check.sh --one <sm121dis> <work-dir> <kernel.bin>: one kernel. First line of .res: PASS|FAIL <words>; the rest
    # is what the report prints for it. Written under a temporary name and renamed, so a cut-short check leaves none.
    DIS=$2; WORK=$3; f=$4
    b=$(basename "$f" .bin)
    if ! "$NVDISASM" -b SM121 -hex "$f" > "$WORK/$b.nv" 2>&1; then
        { echo "FAIL 0"; echo "[FAIL] $b: nvdisasm refused it"; head -3 "$WORK/$b.nv"; } > "$WORK/$b.res.tmp"
    else
        sh "$HERE/oracle_pairs.sh" "$WORK/$b.nv" > "$WORK/$b.want"
        n=$(wc -l < "$WORK/$b.want")
        # Capture the decoder directly: a pipeline would hide its exit status behind sed.
        rc=0
        "$DIS" --pairs "$f" > "$WORK/$b.raw" 2> "$WORK/$b.err" || rc=$?
        if [ "$rc" -ne 0 ]; then
            { echo "FAIL $n"; echo "[FAIL] $b: decoder exited $rc"; head -3 "$WORK/$b.err"; } > "$WORK/$b.res.tmp"
        elif ! sed 's/[[:space:]]\{1,\}/ /g' "$WORK/$b.raw" > "$WORK/$b.got"; then
            { echo "FAIL $n"; echo "[FAIL] $b: decoder output normalization failed"; } > "$WORK/$b.res.tmp"
        elif [ "$n" -gt 0 ] && cmp -s "$WORK/$b.want" "$WORK/$b.got"; then
            { echo "PASS $n"; echo "[PASS] $b: $n words"; } > "$WORK/$b.res.tmp"
        else
            { echo "FAIL $n"; echo "[FAIL] $b: $(diff "$WORK/$b.want" "$WORK/$b.got" | grep -c '^<') of $n words differ"
              diff "$WORK/$b.want" "$WORK/$b.got" | head -6; } > "$WORK/$b.res.tmp"
        fi
    fi
    mv "$WORK/$b.res.tmp" "$WORK/$b.res"
    exit 0
fi

DIS=$1; DIR=$2; WORK=$3
[ -x "$NVDISASM" ] || { echo "VERDICT NOT_RUN: $NVDISASM missing"; exit 2; }
ls "$DIR"/*.bin >/dev/null 2>&1 || { echo "VERDICT NOT_RUN: no kernels in $DIR"; exit 2; }
mkdir -p "$WORK"
# "$WORK/." so a work dir that is a symlink (to /dev/shm, say) is swept too; a stale PASS must never be read back.
find "$WORK/." -maxdepth 1 \( -name '*.res' -o -name '*.res.tmp' \) -exec rm -f {} +
echo "oracle: $("$NVDISASM" --version | grep -o "release [0-9.]*, V[0-9.]*")"
JOBS=${SM121DIS_JOBS:-$(nproc 2>/dev/null || echo 1)}
for f in "$DIR"/*.bin; do printf '%s\0' "$f"; done | xargs -0 -n 1 -P "$JOBS" sh "$HERE/check.sh" --one "$DIS" "$WORK"
k=0; bad=0; words=0
for f in "$DIR"/*.bin; do
    b=$(basename "$f" .bin)
    k=$((k + 1))
    if [ -f "$WORK/$b.res" ] && read -r st n < "$WORK/$b.res"; then
        sed 1d "$WORK/$b.res"
        words=$((words + n))
        [ "$st" = PASS ] || bad=$((bad + 1))
    else
        echo "[FAIL] $b: no result"; bad=$((bad + 1))
    fi
done
echo "kernels: $k, words: $words, failed kernels: $bad"
[ $bad = 0 ] && { echo "VERDICT PASS"; exit 0; }
echo "VERDICT FAIL"; exit 1
