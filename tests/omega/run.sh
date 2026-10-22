#!/bin/sh
# Run the pinned Omega OSC compiler against this checkout's eight slices.
# Usage: tests/omega/run.sh [FUZZ [HARNESS_DIR]]
set -eu

PINNED_SOURCE=1accb93164ce68a2ac2e2c680acf54782f260d62
FUZZ=${1:-120}
W=${2:-${OMEGA_OSC_COMPILER_DIR:-}}
R=$(cd "$(dirname "$0")/../.." && pwd)
SLICES="doorbell.osc sequencer.osc nvrm.osc va.osc open_plan.osc kmod.osc mesa.osc launch.osc"

for f in $SLICES; do
  [ -f "$R/omega/slices/$f" ] || {
    echo "omega-slices: missing required slice $R/omega/slices/$f" >&2
    exit 1
  }
done

if [ -z "$W" ]; then
  echo "omega-slices: compiler harness not configured; set OMEGA_OSC_COMPILER_DIR to an Omega checkout" >&2
  echo "omega-slices: required source commit: $PINNED_SOURCE" >&2
  exit 2
fi
if [ ! -d "$W/.git" ] && [ ! -f "$W/.git" ]; then
  echo "omega-slices: harness directory is not a Git checkout: $W" >&2
  exit 2
fi
SOURCE=$(git -C "$W" rev-parse --verify HEAD 2>/dev/null || true)
if [ "$SOURCE" != "$PINNED_SOURCE" ]; then
  echo "omega-slices: wrong harness source at $W (expected $PINNED_SOURCE, found ${SOURCE:-unavailable})" >&2
  exit 2
fi
if [ -n "$(git -C "$W" status --porcelain)" ]; then
  echo "omega-slices: harness checkout is modified or has untracked source; use a clean $PINNED_SOURCE checkout" >&2
  exit 2
fi

BIN=$W/build/compiler/test_osc_compiler
if [ ! -x "$BIN" ]; then
  echo "omega-slices: compiler binary missing; build it with: make -C '$W' build/compiler/test_osc_compiler" >&2
  make -C "$W" build/compiler/test_osc_compiler
fi
BIN_SHA=$(sha256sum "$BIN" | awk '{print $1}')
printf 'omega harness source=%s binary_sha256=%s\n' "$SOURCE" "$BIN_SHA"

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT HUP INT TERM
mkdir -p "$T/slices"
ln -s "$W/tests/compiler/progs" "$T/progs"
ln -s "$W/tests/compiler/neg" "$T/neg"
for f in $SLICES; do cp "$R/omega/slices/$f" "$T/slices/"; done
cd "$W"
"$BIN" "$FUZZ" "$T"
