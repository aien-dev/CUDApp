#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
mkdir -p "$t/scanner" "$t/bin" "$t/repo"
cp "$here/audit-scan.sh" "$t/scanner/audit-scan.sh"
printf '%s\n' SYNTHETIC_ONLY > "$t/scanner/rules-secrets.txt"
printf '%s\n' PRIVATE_ONLY > "$t/scanner/rules-private.txt"
git -C "$t/repo" init -q
git -C "$t/repo" config user.name test
git -C "$t/repo" config user.email test@example.invalid
printf '%s\n' benign > "$t/repo/input"
git -C "$t/repo" add input
git -C "$t/repo" commit -qm initial
real_grep=$(command -v grep)
printf '#!/bin/bash\nif [[ ${1-} == "${SCAN_FAIL_MODE-}" ]]; then exit 2; fi\nexec %q "$@"\n' "$real_grep" > "$t/bin/grep"
chmod +x "$t/bin/grep"
for phase in -naP -noaP -nP -noP; do
    rc=0
    PATH="$t/bin:$PATH" SCAN_FAIL_MODE="$phase" bash "$t/scanner/audit-scan.sh" "$t/repo" "$t/out" > "$t/log" 2>&1 || rc=$?
    if [[ $rc != 2 ]]; then
        echo "ERROR SELFTEST FAIL: grep phase $phase returned $rc, expected 2"
        exit 1
    fi
done
for rule in secrets private; do
    cp "$t/scanner/rules-$rule.txt" "$t/rule.save"
    printf '[\n' > "$t/scanner/rules-$rule.txt"
    rc=0
    bash "$t/scanner/audit-scan.sh" "$t/repo" "$t/out" > "$t/log" 2>&1 || rc=$?
    cp "$t/rule.save" "$t/scanner/rules-$rule.txt"
    [[ $rc == 2 ]] || { echo "ERROR SELFTEST FAIL: invalid $rule regex returned $rc"; exit 1; }
done
bash "$t/scanner/audit-scan.sh" "$t/repo" "$t/out" > "$t/log"
printf '\037\213broken' > "$t/repo/corrupt.gz"
git -C "$t/repo" add .
git -C "$t/repo" commit -qm corrupt
rc=0
bash "$t/scanner/audit-scan.sh" "$t/repo" "$t/out" > "$t/log" 2>&1 || rc=$?
[[ $rc == 2 && -s $t/out/unscanned.txt ]] || { echo 'ERROR SELFTEST FAIL: corrupt compression was not refused'; exit 1; }
printf '%s\n' SYNTHETIC_ONLY > "$t/repo/delimiter|file"
git -C "$t/repo" add .
git -C "$t/repo" commit -qm delimiter
rc=0
bash "$t/scanner/audit-scan.sh" "$t/repo" "$t/out" > "$t/log" 2>&1 || rc=$?
[[ $rc == 1 ]] || { echo "ERROR SELFTEST FAIL: delimiter-path secret returned $rc"; exit 1; }
grep -F 'delimiter|file:' "$t/out/secrets.txt" > /dev/null
echo 'ERROR SELFTEST PASS: scan errors refuse; clean input and delimiter-path detection preserved'
