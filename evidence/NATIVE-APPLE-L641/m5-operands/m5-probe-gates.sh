#!/bin/bash
set -euo pipefail
repo="$HOME/workspace/cudapp-native-amd-agx"
compiler="$HOME/handoffs/quantum-development-L600/toolchain/build/bootstrap/omc"
cd "$repo"
out=$(mktemp -d "$HOME/handoffs/native-apple-L641/m5-probe-gates.XXXXXX")
printf 'PINNED_COMPILER_SHA256='
sha256sum "$compiler"
test "$(sha256sum "$compiler" | cut -d ' ' -f1)" = 191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9
for name in m5-live-input test-m5-live-input m5-pressure-input test-m5-pressure-input; do
  quiet-build "$compiler" < "native/gpulab/probes/$name.om" > "$out/$name"
  test -s "$out/$name"
  file "$out/$name"
  chmod +x "$out/$name"
done
printf '\006' | "$out/m5-live-input" > "$out/live6.metal"
"$out/test-m5-live-input" < "$out/live6.metal"
printf 'LIVE_SELECTOR_SIX_EXACT_SOURCE_PASS\n'
printf '\001' | "$out/m5-pressure-input" > "$out/pressure1.metal"
"$out/test-m5-pressure-input" < "$out/pressure1.metal"
printf 'PRESSURE_ONE_EXACT_SOURCE_PASS\n'
printf '\000' | "$out/m5-pressure-input" > "$out/pressure0.metal"
"$out/test-m5-live-input" < "$out/pressure0.metal"
printf 'PRESSURE_ZERO_LIVE_PARITY_PASS\n'
set +e
"$out/test-m5-pressure-input" < "$out/live6.metal"
negative=$?
set -e
test "$negative" -eq 1
printf 'PRESSURE_GOLDEN_REJECTS_MISSING_LOAD rc=%s\n' "$negative"
for name in m5-live-input m5-pressure-input; do
  for bytes in '' '\377' '\001\001'; do
    printf "$bytes" > "$out/invalid.bin"
    set +e
    "$out/$name" < "$out/invalid.bin" > "$out/invalid.out"
    rc=$?
    set -e
    test "$rc" -eq 65 && test ! -s "$out/invalid.out"
  done
  printf '%s EMPTY_RANGE_TRAILING_REFUSALS_PASS\n' "$name"
done
sha256sum native/gpulab/probes/{m5-live-input,test-m5-live-input,m5-pressure-input,test-m5-pressure-input}.om
printf 'M5_PROBE_TOOLS_PASS; BACKEND_NOT_IMPLEMENTED; HARDWARE_NOT_RUN\n'
