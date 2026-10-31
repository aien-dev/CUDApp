#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Shape checks for `artifact-export matmul` against the matmul-bf16-v1 bounds. Opens no device.
#   K = 16 uses the single-tile kernel, whose CTA computes one 16 x 8 tile: grid must be (n/8, m/16).
#   K > 16 uses the looped kernel, whose CTA x walks column tiles x, x + grid_x, ...: 1 <= grid_x <= n/8.
#   Shapes off the 16 x 8 x 16 tiling are refused (exit 3).
# usage: artifact_export_shapes.sh EXPORTER OUTDIR
set -eu
exp=$1 out=$2
rm -rf "$out"; mkdir -p "$out"
cm=647116fc1ba1e1394982fc8ba181ce565293f160a6b56afc0f02b2128a675704
commit=0000000000000000000000000000000000000000
fail=0
grid() { sed -n 's/^grid=//p' "$out/mm_$1.manifest"; }
for s in 16x16x8 32x16x64 64x2048x512 16x32x8 8192x16384x8192; do
    IFS=x read -r m k n <<EOF
$s
EOF
    if ! "$exp" matmul --m "$m" --k "$k" --n "$n" --contract-id $cm --out "$out/mm_$s" --commit $commit >/dev/null; then
        echo "FAIL: in-contract matmul $s refused"; fail=1; continue
    fi
    g=$(grid "$s"); gx=${g%%,*}; rest=${g#*,}; gy=${rest%%,*}
    [ "$gy" = $((m / 16)) ] || { echo "FAIL: $s grid_y $gy != m/16"; fail=1; }
    if [ "$k" = 16 ]; then
        [ "$gx" = $((n / 8)) ] || { echo "FAIL: $s single-tile grid_x $gx != n/8 (columns left unwritten)"; fail=1; }
    elif [ "$gx" -lt 1 ] || [ "$gx" -gt $((n / 8)) ]; then
        echo "FAIL: $s looped grid_x $gx outside 1..n/8"; fail=1
    fi
done
for s in 16x24x8 24x16x8 16x16x12; do
    IFS=x read -r m k n <<EOF
$s
EOF
    rc=0; "$exp" matmul --m "$m" --k "$k" --n "$n" --contract-id $cm --out "$out/bad_$s" --commit $commit >/dev/null 2>&1 || rc=$?
    [ "$rc" = 3 ] || { echo "FAIL: off-tiling matmul $s exit $rc, want 3"; fail=1; }
done
[ $fail -eq 0 ] && echo "ARTIFACT EXPORT SHAPES PASS"
exit $fail
