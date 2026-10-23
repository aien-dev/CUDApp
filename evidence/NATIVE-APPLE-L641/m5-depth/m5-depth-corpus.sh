#!/bin/bash
# Host orchestration only; Omega writes inputs and parses binary containers.
set -euo pipefail
LAB=/Users/drakestapleton/.hermes/cache/scratch/amd-agx-native
OUT="$LAB/m5-depth-corpus"
METAL="$LAB/metal-mount/Metal.xctoolchain"
test ! -e "$OUT"
test -x "$METAL/usr/bin/metal"
test -x "$METAL/usr/bin/air-tt"
mkdir -m 700 "$OUT"
exec > >(tee "$OUT/acquisition.log") 2>&1
trap 'rc=$?; printf "ACQUISITION_EXIT=%s\n" "$rc" > "$OUT/exit-status.txt"; exit "$rc"' EXIT
"$METAL/usr/bin/metal" --version
ssh spark 'sha256sum ~/handoffs/native-apple-L641/apple-operand-input.new ~/handoffs/native-apple-L641/macho-extract.new ~/handoffs/native-apple-L641/macho-inner.new' > "$OUT/omega-tools.sha256"
cp "$LAB/k.mtlp-json" "$OUT/k.mtlp-json"
count=0
for depth in {1..64}; do
    printf "$(printf '\\%03o' "$depth")" | ssh -o ConnectTimeout=15 spark '~/handoffs/native-apple-L641/apple-operand-input.new' > "$OUT/k$depth.metal"
    test -s "$OUT/k$depth.metal"
    "$METAL/usr/bin/metal" -std=metal3.1 -target air64-apple-macosx26.0 -o "$OUT/k$depth.metallib" "$OUT/k$depth.metal" > "$OUT/k$depth-frontend.log" 2>&1
    test -s "$OUT/k$depth.metallib"
    for arch in applegpu_g16g applegpu_g17g; do
        stem="$OUT/$arch-k$depth"
        "$METAL/usr/bin/air-tt" -arch "$arch" -platform_version macos 26.0 26.0 -o "$stem.gpubin" "$OUT/k.mtlp-json" "$OUT/k$depth.metallib" > "$stem-compile.log" 2>&1
        test -s "$stem.gpubin"
        ssh -o ConnectTimeout=15 spark '~/handoffs/native-apple-L641/macho-extract.new' < "$stem.gpubin" > "$stem.text"
        ssh -o ConnectTimeout=15 spark '~/handoffs/native-apple-L641/macho-inner.new' < "$stem.gpubin" > "$stem.macho"
        test -s "$stem.text" && test -s "$stem.macho"
        "$METAL/usr/bin/air-objdump" --macho --syms "$stem.macho" > "$stem-symbols.txt"
        "$METAL/usr/bin/air-objdump" --macho --private-headers "$stem.macho" > "$stem-headers.txt"
        count=$((count+1))
        printf 'COMPILED_EXTRACTED %s depth=%s cases=%s\n' "$arch" "$depth" "$count"
    done
done
expected=$((64*2))
test "$count" -eq "$expected"
ssh spark 'sha256sum -c' < "$OUT/omega-tools.sha256"
(cd "$OUT" && shasum -a 256 -- *.metal *.metallib *.gpubin *.text *.macho *-symbols.txt *-headers.txt k.mtlp-json > SHA256SUMS)
printf 'ACQUISITION_COMPLETE cases=%s; ISA_NOT_YET_QUALIFIED; HARDWARE_NOT_RUN\n' "$count" | tee "$OUT/COMPLETE.txt"
