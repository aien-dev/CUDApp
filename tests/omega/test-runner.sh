#!/bin/sh
# Exercise runner refusal paths from an isolated checkout-shaped fixture.
set -eu
R=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT HUP INT TERM
fail() { echo "omega-runner-test: $*" >&2; exit 1; }

# A fresh checkout has no compiler harness configured.
if env -u OMEGA_OSC_COMPILER_DIR "$R/tests/omega/run.sh" >"$T/missing.log" 2>&1; then
  fail "missing harness unexpectedly passed"
fi
grep -q 'compiler harness not configured' "$T/missing.log" || fail "missing harness diagnostic"

# A missing named slice must be reported before a compiler checkout is needed.
mkdir -p "$T/missing/tests/omega" "$T/missing/omega/slices"
cp "$R/tests/omega/run.sh" "$T/missing/tests/omega/run.sh"
for f in doorbell.osc sequencer.osc nvrm.osc va.osc open_plan.osc kmod.osc mesa.osc; do
  : >"$T/missing/omega/slices/$f"
done
if "$T/missing/tests/omega/run.sh" >"$T/slice.log" 2>&1; then
  fail "missing named slice unexpectedly passed"
fi
grep -q 'missing required slice .*launch.osc' "$T/slice.log" || fail "missing-slice diagnostic"

# A present but unpinned checkout fails by its full source revision.
mkdir -p "$T/wrong/tests/omega" "$T/wrong/omega/slices" "$T/harness"
cp "$R/tests/omega/run.sh" "$T/wrong/tests/omega/run.sh"
cp "$R"/omega/slices/*.osc "$T/wrong/omega/slices/"
git -C "$T/harness" init -q
git -C "$T/harness" -c user.name=test -c user.email=test@example.invalid commit --allow-empty -qm wrong-pin
if OMEGA_OSC_COMPILER_DIR="$T/harness" "$T/wrong/tests/omega/run.sh" >"$T/version.log" 2>&1; then
  fail "wrong harness source unexpectedly passed"
fi
grep -q 'wrong harness source .*expected 1accb93164ce68a2ac2e2c680acf54782f260d62' "$T/version.log" || fail "wrong-version diagnostic"

echo "omega-runner-test: PASS (missing harness, missing slice, wrong source pin)"
