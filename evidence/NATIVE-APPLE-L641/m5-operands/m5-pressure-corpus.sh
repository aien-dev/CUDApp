#!/bin/bash
# Offline external compiler orchestration. Omega owns source/binary preparation.
set -euo pipefail
LAB=/Users/drakestapleton/.hermes/cache/scratch/amd-agx-native
OUT="$LAB/m5-pressure-corpus"
METAL="$LAB/metal-mount/Metal.xctoolchain"
test ! -e "$OUT"
mkdir -m 700 "$OUT"
exec > >(tee "$OUT/acquisition.log") 2>&1
trap 'rc=$?; printf "ACQUISITION_EXIT=%s\n" "$rc" > "$OUT/exit-status.txt"; exit "$rc"' EXIT
ssh spark 'sha256sum ~/handoffs/native-apple-L641/m5-pressure-input.new ~/handoffs/native-apple-L641/macho-extract.new ~/handoffs/native-apple-L641/macho-inner.new' > "$OUT/omega-tools.sha256"
count=0
for pressure in {0..31}; do
    printf "$(printf '\\%03o' "$pressure")" | ssh -o ConnectTimeout=15 spark '~/handoffs/native-apple-L641/m5-pressure-input.new' > "$OUT/p$pressure.metal"
    "$METAL/usr/bin/metal" -std=metal3.1 -target air64-apple-macosx26.0 -o "$OUT/p$pressure.metallib" "$OUT/p$pressure.metal" > "$OUT/p$pressure-frontend.log" 2>&1
    for arch in applegpu_g16g applegpu_g17g; do
        stem="$OUT/$arch-p$pressure"
        "$METAL/usr/bin/air-tt" -arch "$arch" -platform_version macos 26.0 26.0 -o "$stem.gpubin" "$LAB/k.mtlp-json" "$OUT/p$pressure.metallib" > "$stem-compile.log" 2>&1
        ssh -o ConnectTimeout=15 spark '~/handoffs/native-apple-L641/macho-extract.new' < "$stem.gpubin" > "$stem.text"
        ssh -o ConnectTimeout=15 spark '~/handoffs/native-apple-L641/macho-inner.new' < "$stem.gpubin" > "$stem.macho"
        test -s "$stem.text" && test -s "$stem.macho"
        "$METAL/usr/bin/air-objdump" --macho --syms "$stem.macho" > "$stem-symbols.txt"
        count=$((count+1))
        printf 'PRESSURE_COMPILED_EXTRACTED %s live-extra=%s cases=%s\n' "$arch" "$pressure" "$count"
    done
done
test "$count" -eq "$((32*2))"
ssh spark 'sha256sum -c' < "$OUT/omega-tools.sha256"
(cd "$OUT" && shasum -a 256 -- *.metal *.metallib *.gpubin *.text *.macho *-symbols.txt > SHA256SUMS)
printf 'PRESSURE_ACQUISITION_COMPLETE cases=%s; ISA_NOT_YET_QUALIFIED; HARDWARE_NOT_RUN\n' "$count" | tee "$OUT/COMPLETE.txt"
