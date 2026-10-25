#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Copied from aien-sovereign-core scripts/zero-cuda-gate.sh at 14126c62 (written by aien-dev, Drake's account).
# Zero-CUDA gate (FB-1 cut 6): fails when a binary needs an NVIDIA CUDA library or
# imports a CUDA runtime/driver/cuBLAS/NVRTC symbol. AIEN reaches the GB10 only through
# the Omega engine (libomega_gpu.a, linked statically, no CUDA).
#
#   scripts/zero-cuda-gate.sh <binary>...      check the given binaries
#   scripts/zero-cuda-gate.sh --self-test      negative control: builds a stub library
#                                              and object that must FAIL the check
# Exit 0 = every binary clean, 1 = CUDA found, 2 = usage or tool error.
set -euo pipefail

LIB_RE='lib(cuda|cudart|cublas|cublasLt|nvrtc|nvJitLink|cudnn|nvidia-ptxjitcompiler)\.so'
SYM_RE='^(cu[A-Z][A-Za-z0-9_]*|cuda[A-Z][A-Za-z0-9_]*|cublas[A-Za-z0-9_]*|nvrtc[A-Z][A-Za-z0-9_]*|__cuda[A-Za-z0-9_]*)(@.*)?$'

check_one() {
	local bin=$1 bad=0
	if [ ! -f "$bin" ]; then
		echo "ZERO_CUDA ERROR $bin: no such file" >&2
		return 2
	fi
	local needed undef
	needed=$(readelf -d "$bin" 2>/dev/null | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p') || return 2
	undef=$(nm -D --undefined-only "$bin" 2>/dev/null | awk '{print $NF}') || return 2
	local hit
	hit=$(printf '%s\n' "$needed" | grep -E "$LIB_RE" || true)
	if [ -n "$hit" ]; then
		echo "ZERO_CUDA FAIL $bin needs: $(echo $hit)"
		bad=1
	fi
	hit=$(printf '%s\n' "$undef" | grep -E "$SYM_RE" || true)
	if [ -n "$hit" ]; then
		echo "ZERO_CUDA FAIL $bin imports: $(echo $hit | cut -c1-400)"
		bad=1
	fi
	if [ $bad -eq 0 ]; then
		echo "ZERO_CUDA PASS $bin needs=[$(echo $needed)]"
	fi
	return $bad
}

self_test() {
	local d
	d=$(mktemp -d)
	trap 'rm -rf "$d"' RETURN
	# A stub named like the CUDA runtime, and an object that calls cudaMalloc through it.
	printf 'int cudaMalloc(void **p, unsigned long n) { (void)p; (void)n; return 1; }\n' >"$d/stub.c"
	printf 'int cudaMalloc(void **p, unsigned long n);\nint probe(void) { void *p; return cudaMalloc(&p, 1); }\n' >"$d/probe.c"
	cc -shared -fPIC -o "$d/libcudart.so" -Wl,-soname,libcudart.so.13 "$d/stub.c"
	cc -shared -fPIC -o "$d/probe.so" "$d/probe.c" -L"$d" -Wl,--no-as-needed -lcudart
	printf 'int clean(void) { return 0; }\n' >"$d/clean.c"
	cc -shared -fPIC -o "$d/clean.so" "$d/clean.c"
	local out rc=0
	out=$(check_one "$d/probe.so") || rc=$?
	echo "$out"
	if [ $rc -ne 1 ] || ! grep -q 'needs: libcudart.so.13' <<<"$out" || ! grep -q 'imports: cudaMalloc' <<<"$out"; then
		echo "ZERO_CUDA SELF-TEST BROKEN: the CUDA probe was not caught (rc=$rc)" >&2
		return 2
	fi
	check_one "$d/clean.so" >/dev/null || {
		echo "ZERO_CUDA SELF-TEST BROKEN: a CUDA-free library was rejected" >&2
		return 2
	}
	echo "ZERO_CUDA SELF-TEST PASS (CUDA probe rejected, clean library accepted)"
}

if [ $# -eq 0 ]; then
	echo "usage: $0 <binary>... | --self-test" >&2
	exit 2
fi
if [ "$1" = "--self-test" ]; then
	self_test
	exit $?
fi
status=0
for b in "$@"; do
	check_one "$b" || { rc=$?; [ $rc -gt $status ] && status=$rc; }
done
exit $status
