#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# The real mxmm entry must reject full-width dimensions before multiplication.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
: "${OMEGA_OMC:?set OMEGA_OMC to the pinned compiler}"
expected=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
actual=$(sha256sum < "$OMEGA_OMC" | cut -c1-64)
[ "$actual" = "$expected" ] || { echo 'NOT_RUN: model capacity compiler pin mismatch'; exit 3; }
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT HUP INT TERM
cat "$ROOT/native/models/fp8.om" "$ROOT/native/models/nvfp4.om" \
    "$ROOT/native/models/mxdot.om" "$ROOT/tests/models/mxdot-capacity.om" > "$T/input.om"
"$OMEGA_OMC" < "$T/input.om" > "$T/check"
chmod +x "$T/check"
check() {
    rc=0
    printf '%b' "$1" | "$T/check" || rc=$?
    [ "$rc" -eq "$2" ] || { echo "FAIL model capacity: $3 expected=$2 actual=$rc"; exit 1; }
}
check '\000' 0 output-capacity-boundary
check '\001' 105 full-width-output-product
check '\002' 105 full-width-right-row
check '\003' 105 full-width-left-row
check '\004' 0 zero-left-dimension
check '\005' 0 zero-right-dimension
check '\006' 105 output-over-capacity
check '\007' 105 scales-over-capacity
check '\010' 105 elements-over-capacity
echo 'MODEL CAPACITY PASS: full-width refusals, boundaries, zero dimensions and output values'
