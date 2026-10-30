#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Revision and epoch vectors. On AArch64 and x86-64 they link the committed
# freestanding units (src/cudapp_freest_units.c, src/cudapp_freest_units_x86.c).
# On any other host they link the portable stand-in, and one seeded entry must
# refuse without writing its result.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT_DIR=${OUT_DIR:-build}
mkdir -p "$OUT_DIR"
host=${CUDAPP_HOST_ARCH:-$(uname -m)}
cc=${CC:-gcc}
case "$host" in
    aarch64|arm64) src=src/cudapp_freest_units.c; def= ;;
    x86_64) src=src/cudapp_freest_units_x86.c; def= ;;
    *) src=src/cudapp_binding_portable.c; def=-DBINDING_PORTABLE ;;
esac
"$cc" -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc $def \
    -o "$OUT_DIR/binding_portable_test" tests/omega/binding_portable_test.c "$src"
"$OUT_DIR/binding_portable_test"
