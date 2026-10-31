#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Invalid lists stop before command-list creation. No device is opened.
set -eu
binary=${1:?usage: grid_ab_exit_test.sh BINARY}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
for arms in '' 'bad' '0:0' '0:0,bad' '0:0x,0:0'; do
    rc=0
    "$binary" --grid-ab "$arms" >"$tmp/output" 2>&1 || rc=$?
    if [ "$rc" -ne 2 ]; then
        cat "$tmp/output"
        echo "FAIL: invalid grid list '$arms' returned $rc, expected 2" >&2
        exit 1
    fi
    if grep -q ': PASS' "$tmp/output"; then
        cat "$tmp/output"
        echo 'FAIL: rejected grid list printed PASS' >&2
        exit 1
    fi
done
echo 'grid-ab invalid-list exit checks: PASS'
