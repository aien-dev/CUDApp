#!/bin/bash
# Host orchestration only; Omega emits inputs and extracts object contents.
set -euo pipefail
LAB=/Users/drakestapleton/.hermes/cache/scratch/amd-agx-native
OUT="$LAB/m5-index-corpus"
METAL="$LAB/metal-mount/Metal.xctoolchain"
test ! -e "$OUT"
mkdir -m 700 "$OUT"
exec > >(tee "$OUT/acquisition.log") 2>&1
trap 'rc=$?; printf "ACQUISITION_EXIT=%s\n" "$rc" > "$OUT/exit-status.txt"; exit "$rc"' EXIT
ssh spark 'sha256sum ~/handoffs/native-apple-L641/m5-index-input ~/handoffs/native-apple-L641/macho-extract.new ~/handoffs/native-apple-L641/macho-inner.new' > "$OUT/omega-tools.sha256"
count=0
for index in {0..63}; do
  printf "$(printf '\\%03o' "$index")" | ssh -o ConnectTimeout=15 spark '~/handoffs/native-apple-L641/m5-index-input' > "$OUT/i$index.metal"
  "$METAL/usr/bin/metal" -std=metal3.1 -target air64-apple-macosx26.0 -o "$OUT/i$index.metallib" "$OUT/i$index.metal" > "$OUT/i$index-frontend.log" 2>&1
  for arch in applegpu_g16g applegpu_g17g; do
    stem="$OUT/$arch-i$index"
    "$METAL/usr/bin/air-tt" -arch "$arch" -platform_version macos 26.0 26.0 -o "$stem.gpubin" "$LAB/k.mtlp-json" "$OUT/i$index.metallib" > "$stem-compile.log" 2>&1
    ssh -o ConnectTimeout=15 spark '~/handoffs/native-apple-L641/macho-extract.new' < "$stem.gpubin" > "$stem.text"
    ssh -o ConnectTimeout=15 spark '~/handoffs/native-apple-L641/macho-inner.new' < "$stem.gpubin" > "$stem.macho"
    test -s "$stem.text" && test -s "$stem.macho"
    "$METAL/usr/bin/air-objdump" --macho --syms "$stem.macho" > "$stem-symbols.txt"
    count=$((count+1))
    printf 'INDEX_COMPILED_EXTRACTED %s immediate=%s cases=%s\n' "$arch" "$index" "$count"
  done
done
test "$count" -eq "$((64*2))"
ssh spark 'sha256sum -c' < "$OUT/omega-tools.sha256"
(cd "$OUT" && shasum -a 256 -- *.metal *.metallib *.gpubin *.text *.macho *-symbols.txt > SHA256SUMS)
printf 'INDEX_ACQUISITION_COMPLETE cases=%s; ISA_NOT_YET_QUALIFIED; HARDWARE_NOT_RUN\n' "$count" | tee "$OUT/COMPLETE.txt"
