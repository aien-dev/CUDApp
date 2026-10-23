#!/bin/sh
# Run the omega compiler harness against this repo's omega/slices.
# Usage: tests/omega/run.sh [FUZZ]  (default 120)
set -eu
W=$HOME/workspace/omega-osc-cubin       # the compiler harness build
R=$(cd "$(dirname "$0")/../.." && pwd)  # this repo checkout
FUZZ=${1:-120}
# Every slice must be present by name: a deleted file is a loud failure here,
# not a silently smaller keyed count in the harness summary.
SLICES="doorbell.osc sequencer.osc nvrm.osc va.osc open_plan.osc kmod.osc mesa.osc launch.osc json.osc"
for f in $SLICES; do
  if [ ! -f "$R/omega/slices/$f" ]; then
    find "$R"/tools -name "$f" | grep -q . || { echo "missing slice: $f" >&2; exit 1; }
  fi
done
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
mkdir -p "$T/slices"
ln -s "$W/tests/compiler/progs" "$T/progs"
ln -s "$W/tests/compiler/neg" "$T/neg"
for f in $SLICES; do
  src=$(find "$R"/omega/slices "$R"/tools -name "$f" | head -1)
  [ -n "$src" ] || { echo "missing slice: $f" >&2; exit 1; }
  cp "$src" "$T/slices/"
done
find "$R"/tools -name "*.osc" -exec cp {} "$T/slices/" \;
cd "$W"
./build/compiler/test_osc_compiler "$FUZZ" "$T"
