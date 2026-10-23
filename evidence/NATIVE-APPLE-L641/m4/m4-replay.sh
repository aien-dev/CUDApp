#!/bin/bash
set -euo pipefail
repo="$HOME/workspace/cudapp-native-amd-agx"
pin=${1:?source pin}
root="$HOME/handoffs/native-apple-L641/replay-m4-$pin"
compiler="$HOME/handoffs/quantum-development-L600/toolchain/build/bootstrap/omc"
test "$(sha256sum "$compiler" | cut -d ' ' -f1)" = 191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9
test ! -e "$root"
mkdir -p "$root"
git -C "$repo" archive "$pin" | tar -x -C "$root"
cd "$root"
evidence=evidence/NATIVE-APPLE-L641/m4
quiet-build make -f native/gpulab/Makefile test OMEGA_NATIVE_COMPILER="$compiler" > "$evidence/fresh-replay.log" 2>&1
out=build/native-gpulab-probes
mkdir -p "$out"
for name in apple-oracle-input apple-operand-input m5-roles-input; do
    quiet-build "$compiler" < "native/gpulab/probes/$name.om" > "$out/$name.new" 2> "$out/$name-compile.log"
    test -s "$out/$name.new"
    chmod +x "$out/$name.new"
    mv "$out/$name.new" "$out/$name"
done
for name in macho-test macho-extract macho-inner; do
    cp native/gpulab/probes/macho.om "$out/$name-source.om"
    tee -a "$out/$name-source.om" < "native/gpulab/probes/$name.om" > /dev/null
    quiet-build "$compiler" < "$out/$name-source.om" > "$out/$name.new" 2> "$out/$name-compile.log"
    test -s "$out/$name.new"
    chmod +x "$out/$name.new"
    mv "$out/$name.new" "$out/$name"
done
"$out/macho-test" > "$evidence/macho-fresh.log"
count=0
for archive in "$evidence"/compiler-corpus/*.gpubin; do
    "$out/macho-extract" < "$archive" > "$out/extracted.text"
    cmp "$out/extracted.text" "${archive%.gpubin}.text"
    count=$((count+1))
done
test "$count" -eq 24
printf 'FROZEN_COMPILER_CONTAINER_REPLAY cases=%s\n' "$count" > "$evidence/corpus-replay.log"
for name in apple-oracle-input apple-operand-input; do
    printf '\001' | "$out/$name" > "$out/$name-valid.metal"
    test -s "$out/$name-valid.metal"
    for bytes in '\000' '\101' '\001\001'; do
        printf "$bytes" > "$out/invalid.bin"
        set +e
        "$out/$name" < "$out/invalid.bin" > "$out/refused.out"
        rc=$?
        set -e
        test "$rc" -eq 65 && test ! -s "$out/refused.out"
    done
done
for id in 0 1 2 3 4 5; do
    printf "$(printf '\\%03o' "$id")" | "$out/m5-roles-input" > "$out/roles-$id.metal"
    test -s "$out/roles-$id.metal"
done
printf '\006' > "$out/invalid.bin"
set +e
"$out/m5-roles-input" < "$out/invalid.bin" > "$out/refused.out"
rc=$?
set -e
test "$rc" -eq 65 && test ! -s "$out/refused.out"
printf 'SOURCE_WRITER_VALID_AND_REFUSAL_GATES_PASS\n' > "$evidence/probe-writers.log"
sha256sum build/native-gpulab/{test-amd,test-agx,test-m3,test-m4,test-cross,test-families,emit-main,test-wire,test-m4-oracle} > "$evidence/native-binaries.sha256"
file build/native-gpulab/test-m4 "$out"/{apple-oracle-input,apple-operand-input,m5-roles-input,macho-test,macho-extract,macho-inner} > "$evidence/binary-types.log"
printf 'FRESH_M4_REPLAY_PASS source=%s\n' "$pin"
