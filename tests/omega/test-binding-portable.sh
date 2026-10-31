#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Revision and epoch vectors on the generated freestanding units
# (om/units.sh, portable checked C). Every host links the same files.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT_DIR=${OUT_DIR:-build}
mkdir -p "$OUT_DIR"
cc=${CC:-gcc}
"$cc" -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc \
    -o "$OUT_DIR/binding_portable_test" tests/omega/binding_portable_test.c \
    src/cudapp_freest_revision.c src/cudapp_freest_epoch.c
"$OUT_DIR/binding_portable_test"
