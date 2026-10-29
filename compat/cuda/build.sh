#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Rebuild the C0 fixture executables for CUDApp#184: vadd_drv for aarch64 and x86_64.
# Usage: compat/cuda/build.sh OUTDIR
# Host only. Nothing here opens /dev/nvidia*, loads libcuda, or runs the executables.
# The link uses a link-only stub libcuda.so.1 generated from the object's own undefined cu* symbols
# (every symbol defined as 0, no code); it is a link-time placeholder, never a runtime library.
# cuda.h is read from the local CUDA install (CUDA_INC) and is never committed (PROVENANCE.md 3.3).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out=${1:?usage: build.sh OUTDIR}
CUDA_INC=${CUDA_INC:-/usr/local/cuda/include}
mkdir -p "$out"
out=$(cd "$out" && pwd)
cp "$here/fixture/vadd_u32.ptx" "$out/vadd_u32.ptx"
build_one() {
	isa=$1
	cc=$2
	d="$out/$isa"
	rm -rf "$d"
	mkdir -p "$d"
	cp "$here/fixture/vadd_drv.c" "$d/vadd_drv.c"
	(
		cd "$d"
		"$cc" -std=c11 -O2 -fno-ident -ffile-prefix-map="$d"=. -I"$CUDA_INC" -c vadd_drv.c -o vadd_drv.o
		defs=$(nm -u vadd_drv.o | awk '$2 ~ /^cu/ {printf " -Wl,--defsym,%s=0", $2}')
		printf '' | "$cc" -x c -c - -o empty.o
		# shellcheck disable=SC2086
		"$cc" -shared -nostdlib -o libcuda.so.1 -Wl,-soname,libcuda.so.1 empty.o $defs
		"$cc" -o vadd_drv vadd_drv.o -L. -l:libcuda.so.1
		rm -f empty.o vadd_drv.c
	)
	echo "built $isa $(sha256sum "$d/vadd_drv" | cut -d' ' -f1)"
}
build_one aarch64 "${CC_AARCH64:-gcc}"
build_one x86_64 "${CC_X86_64:-x86_64-linux-gnu-gcc}"
echo "built ptx $(sha256sum "$out/vadd_u32.ptx" | cut -d' ' -f1)"
