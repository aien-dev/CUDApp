#!/bin/sh
# Drop the inverted-lifetime check and require the named assertion to fail.
# A mutant that still prints a passing inverted_lifetime_masks_overflow is a miss.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
src=tools/dagsim/dagsim.c
grep -q 'LIFETIME_ORDER_CHECK' "$src"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
sed '/LIFETIME_ORDER_CHECK/d' "$src" > "$tmp/dagsim.c"
grep -q 'LIFETIME_ORDER_CHECK' "$tmp/dagsim.c" && { echo "mutant: check still present"; exit 1; }
grep -q 'reject_inverted_lifetimes' "$tmp/dagsim.c" || { echo "mutant: deleted more than the call"; exit 1; }

CC=${CC:-cc}
# -Wno-unused-function: deleting the call leaves reject_inverted_lifetimes unreferenced.
# The mutant has to run. The kill is the named assertion, not a compile error.
cflags="-std=gnu11 -Wall -Wextra -Werror -Wno-unused-function -D_GNU_SOURCE -O2 -Isrc -Itools/dagsim"
$CC $cflags -c -o "$tmp/dagsim.o" "$tmp/dagsim.c"
$CC $cflags -c -o "$tmp/rx_costmodel.o" src/runtime/rx_costmodel.c
# CUDA++ has no src/omega_om_units.c; src/sha256.c is sha256_hash on every host (Makefile DS_SHA_SRC).
sha=src/sha256.c
$CC $cflags -o "$tmp/unit" tests/dagsim/dagsim_unit.c "$tmp/dagsim.o" "$tmp/rx_costmodel.o" "$sha" -Wl,--wrap=rx_cm_predict -lm
set +e
"$tmp/unit" > "$tmp/out" 2>&1
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "mutant survived: unit suite passed after dropping LIFETIME_ORDER_CHECK"
  exit 1
fi
if ! grep -q 'inverted_lifetime_masks_overflow' "$tmp/out"; then
  echo "mutant did not fail named assertion inverted_lifetime_masks_overflow"
  cat "$tmp/out"
  exit 1
fi
echo "mutant killed: inverted_lifetime_masks_overflow"
