#!/bin/sh
# Drop the unsupported-return and refused checks. Leave the p.known test in
# place. The named assertions must fail: a prediction can be marked known and
# still be a refusal.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
src=tools/dagsim/dagsim.c
grep -q 'PREDICT_REFUSAL_CHECK' "$src"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
sed '/PREDICT_REFUSAL_CHECK/s/st == RX_CM_ERR_UNSUPPORTED || p->refused || //' "$src" > "$tmp/dagsim.c"
grep -q 'PREDICT_REFUSAL_CHECK' "$tmp/dagsim.c" || { echo "mutant: deleted the check tag"; exit 1; }
grep -q 'RX_CM_ERR_UNSUPPORTED' "$tmp/dagsim.c" && { echo "mutant: unsupported check still present"; exit 1; }
grep -q 'predict_unusable' "$tmp/dagsim.c" || { echo "mutant: deleted the call"; exit 1; }

CC=${CC:-cc}
# -Wno-unused-parameter: dropping the use of st must not die at compile time.
# The kill is the named assertion, not a warning.
cflags="-std=gnu11 -Wall -Wextra -Werror -Wno-unused-parameter -D_GNU_SOURCE -O2 -Isrc -Itools/dagsim"
$CC $cflags -c -o "$tmp/dagsim.o" "$tmp/dagsim.c"
$CC $cflags -c -o "$tmp/rx_costmodel.o" src/runtime/rx_costmodel.c
# This tree has no src/omega_om_units.c. src/sha256.c is sha256_hash on every host.
# tests/dagsim/sha256_host.c aborts if a digest is requested.
sha=src/sha256.c
$CC $cflags -o "$tmp/unit" tests/dagsim/dagsim_unit.c "$tmp/dagsim.o" "$tmp/rx_costmodel.o" "$sha" -Wl,--wrap=rx_cm_predict -lm
set +e
"$tmp/unit" > "$tmp/out" 2>&1
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "mutant survived: unit suite passed after dropping the predict refusal check"
  exit 1
fi
if ! grep -q '^  FAIL .*predict_unsupported_despite_known' "$tmp/out"; then
  echo "mutant did not fail named assertion predict_unsupported_despite_known"
  cat "$tmp/out"
  exit 1
fi
if ! grep -q '^  FAIL .*predict_refused_despite_known' "$tmp/out"; then
  echo "mutant did not fail named assertion predict_refused_despite_known"
  cat "$tmp/out"
  exit 1
fi
echo "mutant killed: predict_unsupported_despite_known predict_refused_despite_known"
