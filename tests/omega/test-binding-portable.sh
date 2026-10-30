#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Revision and epoch vectors. On AArch64 they link the committed freestanding
# units. On any other host they link the portable stand-in, and one seeded
# entry must refuse without writing its result.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT_DIR=${OUT_DIR:-build}
mkdir -p "$OUT_DIR"
host=$(uname -m)
cc=${CC:-gcc}
if [ "$host" = aarch64 ] || [ "$host" = arm64 ]; then
    src=src/cudapp_freest_units.c
    def=
else
    src=src/cudapp_binding_portable.c
    def=-DBINDING_PORTABLE
fi
"$cc" -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc $def \
    -o "$OUT_DIR/binding_portable_test" tests/omega/binding_portable_test.c "$src"
"$OUT_DIR/binding_portable_test"
