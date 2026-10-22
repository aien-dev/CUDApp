#!/bin/sh
# Run the omega compiler harness against this repo's omega/slices.
# Usage: tests/omega/run.sh [FUZZ]  (default 120)
set -eu
W=$HOME/workspace/omega-osc-cubin       # the compiler harness build
R=$(cd "$(dirname "$0")/../.." && pwd)  # this repo checkout
FUZZ=${1:-120}
T=$(mktemp -d)
mkdir -p "$T/slices"
ln -s "$W/tests/compiler/progs" "$T/progs"
ln -s "$W/tests/compiler/neg" "$T/neg"
cp "$R"/omega/slices/*.osc "$T/slices/"
cd "$W"
exec ./build/compiler/test_osc_compiler "$FUZZ" "$T"
