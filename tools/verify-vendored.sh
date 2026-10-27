#!/bin/bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# verify-vendored: the vendored NVIDIA open headers are exactly the files THIRD_PARTY.md pins.
#   tools/verify-vendored.sh [DIR]      every file listed in DIR/SHA256SUMS matches, and DIR holds no unlisted file
#   tools/verify-vendored.sh --self-test   negative controls: a flipped byte and an extra file must both FAIL
# .crumb files are AIEN coordination metadata, not vendored content, and are not listed.
# Exit: 0 pass, 1 mismatch, 2 usage.
set -u
DEFAULT=$(cd "$(dirname "$0")/.." && pwd)/third_party/nvidia-open-580.173.02

verify() { # $1 = dir
    local d=$1 rc=0
    [ -f "$d/SHA256SUMS" ] || { echo "VENDORED FAIL $d: no SHA256SUMS"; return 1; }
    local bad; bad=$(cd "$d" && sha256sum -c --quiet SHA256SUMS 2>&1) || { echo "VENDORED FAIL $d: $bad"; rc=1; }
    local extra; extra=$(cd "$d" && LC_ALL=C comm -23 \
        <(find . -type f ! -name SHA256SUMS ! -name .crumb | LC_ALL=C sort) \
        <(awk '{print $2}' SHA256SUMS | LC_ALL=C sort))
    [ -z "$extra" ] || { echo "VENDORED FAIL $d: unlisted files: $(echo $extra)"; rc=1; }
    [ $rc -eq 0 ] && echo "VENDORED PASS $d ($(wc -l < "$d/SHA256SUMS") files match SHA256SUMS, none unlisted)"
    return $rc
}

if [ "${1:-}" = --self-test ]; then
    T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
    cp -r "$DEFAULT" "$T/flip" && cp -r "$DEFAULT" "$T/extra" || exit 1
    f=$(awk 'NR==1{print $2}' "$T/flip/SHA256SUMS")
    printf 'x' | dd of="$T/flip/$f" bs=1 seek=0 conv=notrunc status=none
    echo "planted" > "$T/extra/kernel-open/planted.h"
    verify "$T/flip" >/dev/null && { echo "VENDORED SELF-TEST FAIL: flipped byte accepted"; exit 1; }
    verify "$T/extra" >/dev/null && { echo "VENDORED SELF-TEST FAIL: unlisted file accepted"; exit 1; }
    echo "VENDORED SELF-TEST PASS (flipped byte and unlisted file rejected)"
    exit 0
fi
[ $# -le 1 ] || { sed -n '3,6p' "$0" >&2; exit 2; }
verify "${1:-$DEFAULT}"
