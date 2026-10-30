#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Portable-backend host gate (docs/ARCHITECTURE.md "Portable backend"). Everything under test that
# makes a decision is seeded Omega: backend/targets.om (target table, code-object loader checks),
# backend/spirv/emit.om (SPIR-V kernel builder), backend/spirv/conform.om (host reference and bit
# compare). This script only coordinates and compares them with independent oracles:
#   spirv-val (SPIRV-Tools)            every emitted module, plus corrupted copies that must fail
#   llvm-mc + ld.lld + llvm-readelf     ORACLE gfx1151 code object (backend/amd/vadd-gfx1151.s)
#   hello_gpu --sim                     CUDA++'s own warp-simulator output digest for the same vadd
#   PoCL / Mesa lavapipe                CPU implementations that execute the SPIR-V (via the
#                                       migration-pending C launcher tools/multitarget/spirv-host.c)
# It opens no GPU: the launcher pins CPU-only drivers and fails if any nvidia path is mapped or open.
# Results are host EXERCISED at most; every GPU row stays hardware NOT_RUN.
#
# usage: run.sh /abs/path/to/pinned-omc
# Environment (absolute paths; a missing tool makes its checks NOT_RUN, never PASS):
#   SPIRV_VAL     spirv-val                       LLVM_BIN  dir with llvm-mc, llvm-readelf
#   LD_LLD        ld.lld                          HELLO_GPU build/hello_gpu (for --sim digest)
#   CUDAPP_MIGRATION_PENDING_SPIRV_HOST=1 builds and runs the C launcher (default 0: run NOT_RUN)
#   MT_CC (cc), MT_CFLAGS (extra -I for CL/ and vulkan/ headers), POCL_LIBRARY (absolute libpocl.so),
#   LVP_ICD (lavapipe manifest), MT_OUT (output dir to keep; default a new temp dir)
# Exit: 0 MULTITARGET PASS (NOT_RUN checks listed); 1 FAIL; 64 usage; 69 pin refused; 74 I/O;
# 77 NOT_RUN (not Linux AArch64).
set -u
LC_ALL=C
export LC_ALL

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != aarch64 ]; then
  echo 'NOT_RUN: the multitarget host gate needs Linux AArch64 and the pinned seeded omc' >&2
  exit 77
fi
if [ "$#" -ne 1 ]; then echo 'usage: run.sh /absolute/path/to/pinned-omc' >&2; exit 64; fi
omc=$1
case $omc in /*) ;; *) echo 'usage: omc path must be absolute' >&2; exit 64 ;; esac
# omc: omega-mv 369dbc96 seed/out/omc, equal to its committed seed/out/omc.sha256 (AGENTS.md pin).
omc_pin=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
digest() { d=$(sha256sum < "$1") || return 1; printf '%s\n' "${d%% *}"; }
[ "$(digest "$omc")" = "$omc_pin" ] || { echo "REFUSED: $omc digest differs from pin $omc_pin" >&2; exit 69; }

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd) || exit 74
out=${MT_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/cudapp-multitarget.XXXXXX")} || exit 74
mkdir -p "$out" || exit 74
echo "PIN omc $omc_pin"
echo "Outputs: $out"

npass=0; nfail=0; nnr=0
ok() { npass=$((npass + 1)); echo "PASS $*"; }
bad() { nfail=$((nfail + 1)); echo "FAIL $*"; }
notrun() { nnr=$((nnr + 1)); echo "NOT_RUN $*"; }
# expect RC NAME cmd...: the command must exit RC (stdout+stderr kept in $out/NAME.log)
expect() {
  want=$1; name=$2; shift 2
  "$@" > "$out/$name.log" 2>&1
  got=$?
  if [ "$got" -eq "$want" ]; then ok "$name exit=$got $(tail -n 1 "$out/$name.log")"; else bad "$name exit=$got want=$want $(tail -n 1 "$out/$name.log")"; fi
}
# poke FILE OFFSET OCTALBYTE: overwrite one byte in place
poke() { printf "\\$3" | dd of="$1" bs=1 seek="$2" conv=notrunc status=none; }

# 1. Compile the three Omega units with the pinned omc.
for u in targets:backend/targets.om emit:backend/spirv/emit.om conform:backend/spirv/conform.om; do
  name=${u%%:*}; src=$root/${u#*:}
  timeout 120 "$omc" < "$src" > "$out/$name" 2> "$out/$name.compile.err"
  rc=$?
  if [ "$rc" -ne 0 ] || [ -s "$out/$name.compile.err" ] || [ ! -s "$out/$name" ]; then
    cat "$out/$name.compile.err"; echo "FAIL compile $name exit $rc"; exit 1
  fi
  chmod 700 "$out/$name" || exit 74
  echo "COMPILE $name src=$(digest "$src") bin=$(digest "$out/$name")"
done
T=$out/targets; E=$out/emit; C=$out/conform

# 2. Target table: 22 rows; only nvidia-sm_121 SUPPORTED; every PLANNED row hardware NOT_RUN;
#    host rows say not-a-GPU. The same checker must reject a table with one NOT_RUN turned into PASS.
check_table() {
  rows=$(grep -v '^TARGET id vendor ' "$1" | grep -c '^TARGET [a-z]')
  [ "$rows" -eq 22 ] || return 1
  [ "$(grep -c ' SUPPORTED ' "$1")" -eq 1 ] || return 1
  grep -q '^TARGET nvidia-sm_121 .* SUPPORTED ' "$1" || return 1
  [ "$(grep -c ' PLANNED NOT_RUN ' "$1")" -eq 19 ] || return 1
  [ "$(grep -c ' HOST-ONLY(not-a-GPU) ' "$1")" -eq 2 ] || return 1
  return 0
}
"$T" doctor > "$out/doctor.txt" 2>&1 || bad "targets doctor exit $?"
if check_table "$out/doctor.txt"; then ok "table rows=22 supported=nvidia-sm_121 planned_not_run=19 host=2"; else bad table; fi
sed '0,/ PLANNED NOT_RUN /s// PLANNED PASS /' "$out/doctor.txt" > "$out/doctor-mutant.txt"
if check_table "$out/doctor-mutant.txt"; then bad "table checker accepted a row claiming PASS"; else ok "table-negative row claiming PASS rejected"; fi

# 3. Emit the SPIR-V modules (and the planted-wrong mutants) from Omega.
for k in vadd fsum16; do for a in cl vk; do
  "$E" $k $a > "$out/$k.$a.spv" || bad "emit $k $a"
  "$E" $k $a mutant > "$out/$k.$a.mutant.spv" || bad "emit $k $a mutant"
  echo "MODULE $k.$a.spv sha256=$(digest "$out/$k.$a.spv") bytes=$(wc -c < "$out/$k.$a.spv")"
done; done

# 4. spirv-val on every module; corrupted copies must fail it.
if [ -n "${SPIRV_VAL:-}" ] && [ -x "$SPIRV_VAL" ]; then
  echo "TOOL spirv-val $("$SPIRV_VAL" --version 2>&1 | head -n 1)"
  for k in vadd fsum16; do for m in "" .mutant; do
    expect 0 "spirv-val-$k.cl$m" "$SPIRV_VAL" --target-env opencl2.1 "$out/$k.cl$m.spv"
    expect 0 "spirv-val-$k.vk$m" "$SPIRV_VAL" --target-env vulkan1.1 "$out/$k.vk$m.spv"
  done; done
else
  notrun "spirv-val: SPIRV_VAL not set to an executable"
fi
head -c $(( $(wc -c < "$out/vadd.vk.spv") - 4 )) "$out/vadd.vk.spv" > "$out/vadd.vk.trunc.spv"
cp "$out/vadd.vk.spv" "$out/vadd.vk.badmagic.spv" && poke "$out/vadd.vk.badmagic.spv" 0 000
# first instruction's word count (high byte of word 5) set to 0x7f..: runs past the end
cp "$out/vadd.vk.spv" "$out/vadd.vk.overrun.spv" && poke "$out/vadd.vk.overrun.spv" 23 177
if [ -n "${SPIRV_VAL:-}" ] && [ -x "$SPIRV_VAL" ]; then
  for c in trunc badmagic overrun; do
    if "$SPIRV_VAL" --target-env vulkan1.1 "$out/vadd.vk.$c.spv" > "$out/spirv-val-$c.log" 2>&1; then
      bad "spirv-val accepted corrupted module ($c)"; else ok "spirv-val-negative $c rejected: $(head -n 1 "$out/spirv-val-$c.log")"; fi
  done
fi

# 5. Omega loader: accepts, and refuses every wrong format, corruption and entry.
for k in vadd fsum16; do
  expect 0 "load-cl-$k" "$T" load SPIRV_CL "$out/$k.cl.spv" $k
  expect 0 "load-vk-$k" "$T" load SPIRV_VK "$out/$k.vk.spv" $k
done
expect 21 load-neg-truncated "$T" load SPIRV_VK "$out/vadd.vk.trunc.spv"
expect 21 load-neg-overrun "$T" load SPIRV_VK "$out/vadd.vk.overrun.spv"
expect 20 load-neg-badmagic "$T" load SPIRV_VK "$out/vadd.vk.badmagic.spv"
expect 11 load-neg-aarch64 "$T" load AARCH64 "$out/vadd.vk.spv"
expect 22 load-neg-cl-as-vk "$T" load SPIRV_VK "$out/vadd.cl.spv"
expect 22 load-neg-vk-as-cl "$T" load SPIRV_CL "$out/vadd.vk.spv"
expect 23 load-neg-entry "$T" load SPIRV_VK "$out/vadd.vk.spv" nosuch
expect 23 load-neg-entry-prefix "$T" load SPIRV_CL "$out/fsum16.cl.spv" fsum
expect 11 load-neg-msl "$T" load MSL_SOURCE "$out/vadd.vk.spv"
expect 12 load-neg-cubin "$T" load CUBIN "$out/vadd.vk.spv"
expect 10 load-neg-unknown "$T" load PTX "$out/vadd.vk.spv"
expect 40 load-neg-spv-as-sm121 "$T" load SM121_RAW "$out/vadd.vk.spv"
head -c 32 /dev/zero > "$out/sm121-two.bin"
expect 0 load-sm121-shape "$T" load SM121_RAW "$out/sm121-two.bin"
head -c 20 /dev/zero > "$out/sm121-odd.bin"
expect 40 load-neg-sm121-odd "$T" load SM121_RAW "$out/sm121-odd.bin"

# 6. Host reference (conform.om) and its cross-check against hello_gpu --sim.
"$C" in vadd "$out/a.bin" "$out/b.bin" && "$C" in fsum16 "$out/x.bin" \
  && "$C" ref vadd "$out/vadd.ref" && "$C" ref fsum16 "$out/fsum16.ref" || bad "conform inputs/reference"
echo "REFERENCE vadd sha256=$(digest "$out/vadd.ref") fsum16 sha256=$(digest "$out/fsum16.ref")"
if [ -n "${HELLO_GPU:-}" ] && [ -x "$HELLO_GPU" ]; then
  hs=$("$HELLO_GPU" --sim | sed -n 's/^output sha256 //p')
  if [ "$hs" = "$(digest "$out/vadd.ref")" ]; then ok "reference = hello_gpu --sim output sha256 $hs"; else bad "reference $(digest "$out/vadd.ref") != hello_gpu --sim $hs"; fi
else
  notrun "hello_gpu --sim cross-check: HELLO_GPU not set"
fi
cp "$out/vadd.ref" "$out/vadd.planted.out" && poke "$out/vadd.planted.out" 4096 377
expect 1 conform-neg-planted "$C" check vadd "$out/vadd.planted.out"
expect 0 conform-self "$C" check vadd "$out/vadd.ref"

# 7. Execute on CPU implementations (migration-pending C launcher, default OFF).
if [ "${CUDAPP_MIGRATION_PENDING_SPIRV_HOST:-0}" != 1 ]; then
  notrun "spirv-host runs on PoCL and lavapipe: CUDAPP_MIGRATION_PENDING_SPIRV_HOST is not 1 (migration-pending C, default OFF)"
else
  cc=${MT_CC:-cc}
  pocl=${POCL_LIBRARY:-}; lvp=${LVP_ICD:-/usr/share/vulkan/icd.d/lvp_icd.json}
  build() { # NAME FORBID
    $cc -std=c11 -O2 -Wall -Wextra -Werror -DCUDAPP_MIGRATION_PENDING_SPIRV_HOST=1 \
      -DMT_POCL_LIBRARY="\"$pocl\"" -DMT_LVP_ICD="\"$lvp\"" -DMT_FORBID="\"$2\"" ${MT_CFLAGS:-} \
      "$root/tools/multitarget/spirv-host.c" -o "$out/$1" -l:libOpenCL.so.1 -l:libvulkan.so.1
  }
  if ! build spirv-host nvidia > "$out/spirv-host.build.log" 2>&1; then
    cat "$out/spirv-host.build.log"; bad "spirv-host build"
  else
    H=$out/spirv-host
    for a in cl vk; do
      case $a in cl) [ -n "$pocl" ] || { notrun "cl runs: POCL_LIBRARY not set"; continue; } ;; esac
      for m in "" .mutant; do
        "$H" $a "$out/vadd.$a$m.spv" vadd 1048576 "$out/a.bin" "$out/b.bin" "$out/vadd.$a$m.out" > "$out/run-vadd.$a$m.log" 2>&1 || bad "run vadd $a$m exit $?"
        "$H" $a "$out/fsum16.$a$m.spv" fsum16 65536 "$out/x.bin" "$out/fsum16.$a$m.out" > "$out/run-fsum16.$a$m.log" 2>&1 || bad "run fsum16 $a$m exit $?"
      done
      grep -h 'device=\|platform=' "$out/run-vadd.$a.log"
      expect 0 "conform-vadd-$a" "$C" check vadd "$out/vadd.$a.out"
      expect 0 "conform-fsum16-$a" "$C" check fsum16 "$out/fsum16.$a.out"
      expect 1 "conform-neg-vadd-$a-mutant" "$C" check vadd "$out/vadd.$a.mutant.out"
      expect 1 "conform-neg-fsum16-$a-mutant" "$C" check fsum16 "$out/fsum16.$a.mutant.out"
      for k in vadd fsum16; do
        if cmp -s "$out/$k.$a.out" "$out/$k.ref"; then ok "bytes $k.$a.out = $k.ref sha256=$(digest "$out/$k.ref")"; else bad "bytes $k.$a.out != $k.ref"; fi
      done
    done
    # Isolation guard control: a copy forbidding the CPU driver it must load has to exit 9.
    if build spirv-host-forbid-pocl libpocl > "$out/forbid.build.log" 2>&1 && build spirv-host-forbid-lvp vulkan_lvp >> "$out/forbid.build.log" 2>&1; then
      [ -n "$pocl" ] && expect 9 isolation-neg-cl "$out/spirv-host-forbid-pocl" cl "$out/vadd.cl.spv" vadd 1048576 "$out/a.bin" "$out/b.bin" "$out/iso.out"
      expect 9 isolation-neg-vk "$out/spirv-host-forbid-lvp" vk "$out/vadd.vk.spv" vadd 1048576 "$out/a.bin" "$out/b.bin" "$out/iso.out"
    else
      bad "isolation control build"
    fi
  fi
fi

# 8. AMDGPU_HSACO: ORACLE gfx1151 code object (llvm-mc + ld.lld), Omega loader validation and refusals.
mc=${LLVM_BIN:-/usr/lib/llvm-18/bin}/llvm-mc; re=${LLVM_BIN:-/usr/lib/llvm-18/bin}/llvm-readelf; lld=${LD_LLD:-}
if [ -x "$mc" ] && [ -x "$re" ] && [ -n "$lld" ] && [ -x "$lld" ]; then
  echo "TOOL llvm-mc $("$mc" --version | sed -n 's/.*LLVM version //p' | head -n 1); ld.lld $("$lld" --version)"
  hs=$out/vadd-gfx1151.hsaco
  if "$mc" -triple amdgcn-amd-amdhsa -mcpu=gfx1151 -filetype=obj "$root/backend/amd/vadd-gfx1151.s" -o "$out/vadd-gfx1151.o" \
     && "$lld" -shared "$out/vadd-gfx1151.o" -o "$hs"; then
    echo "HSACO ORACLE sha256=$(digest "$hs") bytes=$(wc -c < "$hs")"
    "$re" -h --notes --dyn-syms "$hs" > "$out/hsaco.readelf.txt" 2>&1
    if grep -q 'Flags: *0x4A, gfx1151' "$out/hsaco.readelf.txt" && grep -q 'amdhsa.target: *amdgcn-amd-amdhsa--gfx1151' "$out/hsaco.readelf.txt" \
       && grep -q 'OS/ABI: *AMDGPU - HSA' "$out/hsaco.readelf.txt" && grep -q ' vadd.kd$' "$out/hsaco.readelf.txt"; then
      ok "llvm-readelf: gfx1151 flags 0x4A, HSA OSABI, NT_AMDGPU_METADATA target gfx1151, vadd.kd"
    else bad "llvm-readelf did not show the expected gfx1151 code object"; fi
    expect 0 load-hsaco "$T" load AMDGPU_HSACO "$hs" gfx1151
    expect 33 load-neg-hsaco-wrong-gfx "$T" load AMDGPU_HSACO "$hs" gfx1100
    expect 37 load-neg-hsaco-unknown-gfx "$T" load AMDGPU_HSACO "$hs" gfx9999
    expect 2 load-neg-hsaco-no-gfx "$T" load AMDGPU_HSACO "$hs"
    expect 30 load-neg-spv-as-hsaco "$T" load AMDGPU_HSACO "$out/vadd.cl.spv" gfx1151
    expect 20 load-neg-hsaco-as-spv "$T" load SPIRV_VK "$hs"
    expect 40 load-neg-hsaco-as-sm121 "$T" load SM121_RAW "$hs"
    # Wrong note: the metadata target says gfx1150 while e_flags still says gfx1151.
    t=$(grep -obUa 'amdgcn-amd-amdhsa--gfx1151' "$hs" | head -n 1 | cut -d: -f1)
    cp "$hs" "$out/neg-note-target.hsaco" && poke "$out/neg-note-target.hsaco" $((t + 25)) 060
    if "$re" --notes "$out/neg-note-target.hsaco" 2>/dev/null | grep -q 'amdgcn-amd-amdhsa--gfx1150'; then
      ok "oracle sees the planted note target gfx1150"; else bad "planted note target not visible to llvm-readelf"; fi
    expect 36 load-neg-note-target "$T" load AMDGPU_HSACO "$out/neg-note-target.hsaco" gfx1151
    # Wrong note type: 32 (NT_AMDGPU_METADATA) becomes 33.
    n=$(grep -obUa 'AMDGPU' "$hs" | head -n 1 | cut -d: -f1)
    cp "$hs" "$out/neg-note-type.hsaco" && poke "$out/neg-note-type.hsaco" $((n - 4)) 041
    expect 34 load-neg-note-type "$T" load AMDGPU_HSACO "$out/neg-note-type.hsaco" gfx1151
    cp "$hs" "$out/neg-eflags.hsaco" && poke "$out/neg-eflags.hsaco" 48 101
    expect 33 load-neg-eflags "$T" load AMDGPU_HSACO "$out/neg-eflags.hsaco" gfx1151
    cp "$hs" "$out/neg-osabi.hsaco" && poke "$out/neg-osabi.hsaco" 7 000
    expect 31 load-neg-osabi "$T" load AMDGPU_HSACO "$out/neg-osabi.hsaco" gfx1151
    cp "$hs" "$out/neg-abi6.hsaco" && poke "$out/neg-abi6.hsaco" 8 004
    expect 31 load-neg-abi-v6 "$T" load AMDGPU_HSACO "$out/neg-abi6.hsaco" gfx1151
    head -c 600 "$hs" > "$out/neg-trunc.hsaco"
    expect 34 load-neg-hsaco-truncated "$T" load AMDGPU_HSACO "$out/neg-trunc.hsaco" gfx1151
  else
    bad "ORACLE hsaco assemble/link"
  fi
else
  notrun "AMDGPU_HSACO checks: llvm-mc/llvm-readelf under LLVM_BIN or LD_LLD missing"
fi

echo "hardware NOT_RUN: every GPU target except the GB10 (no AMD, Intel, Apple, Qualcomm, Arm, Imagination, Broadcom or Vortex device on this host; PoCL and lavapipe are CPUs)"
if [ "$nfail" -ne 0 ]; then echo "MULTITARGET FAIL pass=$npass fail=$nfail not_run=$nnr"; exit 1; fi
echo "MULTITARGET PASS pass=$npass fail=0 not_run=$nnr"
exit 0
