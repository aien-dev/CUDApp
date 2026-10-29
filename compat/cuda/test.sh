#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# CUDApp#184 C0 admission test: rebuild the fixture, admit it against both frozen profiles, and require
# every negative control to be refused for its own reason. Host only: no GPU, no libcuda, no execution
# of the artifact. Usage: compat/cuda/test.sh [WORKDIR]   (OMEGA_OMC overrides the pinned compiler path)
set -eu
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
w=${1:-$(mktemp -d)}
mkdir -p "$w"
w=$(cd "$w" && pwd)
OMC=${OMEGA_OMC:-$repo/../omega-mv/seed/out/omc}
pin=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
got=$(sha256sum "$OMC" | cut -d' ' -f1)
[ "$got" = "$pin" ] || { echo "omc pin mismatch: $got"; exit 2; }
cat "$repo/om/cache/digest.om" "$here/c0_check.om" > "$w/c0src.om"
"$OMC" < "$w/c0src.om" > "$w/c0_check"
chmod +x "$w/c0_check"
C="$w/c0_check"
"$here/build.sh" "$w/a" > "$w/build-a.log"
"$here/build.sh" "$w/b" > "$w/build-b.log"
cmp "$w/build-a.log" "$w/build-b.log" || { echo "FAIL rebuild not deterministic"; exit 1; }
cat "$w/build-a.log"
PA="$here/profile-c0-aarch64-v1.tsv"
PX="$here/profile-c0-x86_64-v1.tsv"
EA="$w/a/aarch64/vadd_drv"
EX="$w/a/x86_64/vadd_drv"
PTX="$w/a/vadd_u32.ptx"
fail=0
pass() { # name profile exe ptx runisa
	n=$1
	shift
	if "$C" check "$@" > "$w/$n.out"; then
		echo "PASS admit $n: $(cat "$w/$n.out")"
	else
		echo "FAIL admit $n rc=$?: $(cat "$w/$n.out")"
		fail=1
	fi
}
refuse() { # name reason profile exe ptx runisa
	n=$1
	why=$2
	shift 2
	rc=0
	"$C" check "$@" > "$w/$n.out" || rc=$?
	if [ "$rc" -eq 1 ] && grep -q "REFUSED $why" "$w/$n.out"; then
		echo "PASS refuse $n rc=1: $(tr '\n' ';' < "$w/$n.out")"
	else
		echo "FAIL refuse $n rc=$rc want $why: $(tr '\n' ';' < "$w/$n.out")"
		fail=1
	fi
}
sha() { sha256sum "$1" | cut -d' ' -f1; }
tab=$(printf '\t')
pass aarch64 "$PA" "$EA" "$PTX" aarch64
pass x86_64-static "$PX" "$EX" "$PTX" static
# 1 host ISA: an ARM64 profile is never compared on x86-64 and the reverse.
refuse isa-a-on-x86 host-isa "$PA" "$EA" "$PTX" x86_64
refuse isa-x86-on-a host-isa "$PX" "$EX" "$PTX" aarch64
# 2 wrong artifact: the x86-64 executable presented under the aarch64 profile, and an edited hash.
refuse x86-exe-as-a artifact-hash "$PA" "$EX" "$PTX" static
sed "s/$(sha "$EA")/$(sha "$EX")/" "$PA" > "$w/p-hash.tsv"
refuse edited-hash artifact-hash "$w/p-hash.tsv" "$EA" "$PTX" aarch64
# 3 profile drift: dropped import, added import, unsupported status, unknown schema, wrong libc floor.
grep -v "^import${tab}cuInit${tab}" "$PA" > "$w/p-drop.tsv"
refuse dropped-import "unlisted import cuInit" "$w/p-drop.tsv" "$EA" "$PTX" aarch64
{ cat "$PA"; printf 'import\tcuMemAllocManaged\tcuda\tsupported\tcuMemAllocManaged\n'; } > "$w/p-add.tsv"
refuse added-import "missing import cuMemAllocManaged" "$w/p-add.tsv" "$EA" "$PTX" aarch64
sed "s/^import${tab}cuInit${tab}cuda${tab}supported/import${tab}cuInit${tab}cuda${tab}unsupported/" "$PA" > "$w/p-unsup.tsv"
refuse unsupported-status unsupported-status "$w/p-unsup.tsv" "$EA" "$PTX" aarch64
sed "s/^schema${tab}AIEN_CUDA_COMPAT_PROFILE${tab}1\$/schema${tab}AIEN_CUDA_COMPAT_PROFILE${tab}2/" "$PA" > "$w/p-schema.tsv"
refuse schema-2 schema-mismatch "$w/p-schema.tsv" "$EA" "$PTX" aarch64
sed "s/^\(host${tab}.*${tab}\)GLIBC_2.34\$/\1GLIBC_2.17/" "$PA" > "$w/p-floor.tsv"
refuse libc-floor libc-floor "$w/p-floor.tsv" "$EA" "$PTX" aarch64
{ cat "$PA"; printf 'lookup\tabsent\tcuMod\n'; } > "$w/p-lookup.tsv"
refuse lookup-prefix dynamic-lookup "$w/p-lookup.tsv" "$EA" "$PTX" aarch64
# 4 missing dependency: the profile loses libc.so.6, and an executable linked without libcuda.so.1.
grep -v "^needed${tab}libc.so.6${tab}" "$PA" > "$w/p-noneed.tsv"
refuse profile-missing-needed "unlisted needed libc.so.6" "$w/p-noneed.tsv" "$EA" "$PTX" aarch64
mkdir -p "$w/nolib"
gcc -o "$w/nolib/vadd_drv" "$w/a/aarch64/vadd_drv.o" -Wl,--unresolved-symbols=ignore-all
sed "s/$(sha "$EA")/$(sha "$w/nolib/vadd_drv")/" "$PA" > "$w/p-nolib.tsv"
refuse exe-missing-libcuda "missing needed libcuda.so.1" "$w/p-nolib.tsv" "$w/nolib/vadd_drv" "$PTX" aarch64
# 5 unsupported PTX: a barrier and a two-statement line, each with the PTX hash re-pinned so only the op refuses.
sed 's/^\(\tret;\)$/\tbar.sync 0;\n\1/' "$PTX" > "$w/bar.ptx"
sed "s/$(sha "$PTX")/$(sha "$w/bar.ptx")/" "$PA" > "$w/p-bar.tsv"
refuse ptx-barrier "unsupported-op ptxop bar.sync" "$w/p-bar.tsv" "$EA" "$w/bar.ptx" aarch64
sed 's/^\(\tret;\)$/\tret; ret;/' "$PTX" > "$w/multi.ptx"
sed "s/$(sha "$PTX")/$(sha "$w/multi.ptx")/" "$PA" > "$w/p-multi.tsv"
refuse ptx-multi multi-statement "$w/p-multi.tsv" "$EA" "$w/multi.ptx" aarch64
if [ "$fail" -ne 0 ]; then
	echo "CUDA-C0-PROFILE-TEST FAIL"
	exit 1
fi
echo "CUDA-C0-PROFILE-TEST PASS"
