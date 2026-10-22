#!/bin/sh
# SPDX-License-Identifier: MIT
# Exercise only the host codegen/simulator tier. No GPU is opened.
set -u
B=${1:?gpu_isolation_test binary}
D=$(mktemp -d /tmp/cudapp-isolation-dump.XXXXXX) || exit 2
trap 'rm -f "$D/new.bin" "$D/keep.bin" "$D/link.bin" "$D/run.log"; rmdir "$D"' EXIT HUP INT TERM
umask 000
checks=0; failures=0
check() {
    checks=$((checks + 1))
    if "$@"; then echo "PASS isolation dump $checks"; else failures=$((failures + 1)); echo "FAIL isolation dump $checks"; fi
}
"$B" --host-only --dump-hang "$D/new.bin" > "$D/run.log" 2>&1; rc=$?
check test "$rc" -eq 0
check test "$(stat -c %a "$D/new.bin")" = 600
printf 'sentinel' > "$D/keep.bin"
before=$(sha256sum "$D/keep.bin" | cut -d ' ' -f1)
"$B" --host-only --dump-hang "$D/keep.bin" > "$D/run.log" 2>&1; rc=$?
check test "$rc" -eq 1
check test "$(sha256sum "$D/keep.bin" | cut -d ' ' -f1)" = "$before"
ln -s "$D/keep.bin" "$D/link.bin"
"$B" --host-only --dump-hang "$D/link.bin" > "$D/run.log" 2>&1; rc=$?
check test "$rc" -eq 1
check test "$(sha256sum "$D/keep.bin" | cut -d ' ' -f1)" = "$before"
printf 'isolation dump: %s checks, %s failures\n' "$checks" "$failures"
test "$failures" -eq 0
