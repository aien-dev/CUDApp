#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Host-only orchestration tests. All privilege, network and GPU-entry commands
# are mocked; no real user account, sudo, network request or device is used.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
script=${1:-$root/tools/runtime/clean-user-hello.sh}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/bin"
cat > "$work/bin/sudo" <<'MOCK'
#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$MOCK_CALLS"
case "$1" in
  useradd) exit "$MOCK_USERADD_RC" ;;
  -u) exit 0 ;; # Never execute the requested login shell.
  test) exit 1 ;; # No real trace was produced.
  pkill|userdel|rm) exit 0 ;; # Record only; never execute.
  *) echo UNEXPECTED_PRIVILEGE >> "$MOCK_CALLS"; echo "unexpected privileged command: $1" >&2; exit 99 ;;
esac
MOCK
cat > "$work/bin/curl" <<'MOCK'
#!/usr/bin/env bash
echo curl >> "$MOCK_CALLS"
exit 7 # Stop after account creation, before README/GPU execution.
MOCK
cat > "$work/bin/flock" <<'MOCK'
#!/usr/bin/env bash
echo UNEXPECTED_GPU_ENTRY >> "$MOCK_CALLS"
exit 99
MOCK
cat > "$work/bin/getent" <<'MOCK'
#!/usr/bin/env bash
exit 2 # The mock account is absent after successful mock deletion.
MOCK
cat > "$work/bin/date" <<'MOCK'
#!/usr/bin/env bash
printf '%s\n' 010203 # Make clock-based names collide deterministically.
MOCK
real_mktemp=$(command -v mktemp)
cat > "$work/bin/mktemp" <<'MOCK'
#!/usr/bin/env bash
if [ "${MOCK_MKTEMP_FAIL:-0}" = 1 ]; then exit 1; fi
exec "$REAL_MKTEMP" "$@"
MOCK
chmod +x "$work/bin/"*
pass=0; fail=0
check() {
  local name=$1; shift
  if "$@"; then echo "PASS $name"; pass=$((pass + 1))
  else echo "FAIL $name"; fail=$((fail + 1)); fi
}
no_account_cleanup() { ! grep -Eq '^(pkill|userdel) ' "$1"; }
no_privilege() { [ ! -s "$1" ]; }
run_case() {
  local name=$1 add_rc=$2 temp_fail=${3:-0}
  mkdir -p "$work/$name/tmp"
  : > "$work/$name/calls"
  set +e
  env PATH="$work/bin:/usr/bin:/bin" \
    REAL_MKTEMP="$real_mktemp" MOCK_CALLS="$work/$name/calls" \
    MOCK_USERADD_RC="$add_rc" MOCK_MKTEMP_FAIL="$temp_fail" \
    TMPDIR="$work/$name/tmp" CUDAPP_BOARD= \
    bash "$script" main "$work/$name/run.log" > "$work/$name/output" 2>&1
  rc=$?
  set -e
  check "$name-exit-3" test "$rc" -eq 3
  check "$name-never-enters-gpu" test "$(grep -c UNEXPECTED_GPU_ENTRY "$work/$name/calls" || true)" -eq 0
  check "$name-only-expected-privilege" test "$(grep -c UNEXPECTED_PRIVILEGE "$work/$name/calls" || true)" -eq 0
}

# useradd exit 9 means the candidate account already exists. Its processes and
# home belong to someone else; the cleanup must never issue deletion commands.
run_case collision 9
check existing-account-untouched no_account_cleanup "$work/collision/calls"
run_case failure 1
check failed-creation-untouched no_account_cleanup "$work/failure/calls"

# Successful creation followed by a network/setup failure must still remove
# precisely the owned temporary account, not leak it or use a different name.
run_case created 0
created=$(sed -n 's/^useradd -m -s \/bin\/bash //p' "$work/created/calls")
check created-name-nonempty test -n "$created"
check owned-process-cleanup grep -Fxq "pkill -u $created" "$work/created/calls"
check owned-account-cleanup grep -Fxq "userdel -r $created" "$work/created/calls"
check one-owned-account-deletion test "$(grep -c '^userdel ' "$work/created/calls" || true)" -eq 1

# Even with the same UTC clock value, simultaneous/repeated candidates differ.
run_case created_again 0
again=$(sed -n 's/^useradd -m -s \/bin\/bash //p' "$work/created_again/calls")
check distinct-candidate-names test "$created" != "$again"

# A missing temporary directory must abort before creating any account.
run_case temp_failure 0 1
check temp-failure-no-privilege no_privilege "$work/temp_failure/calls"

echo "clean-user cleanup tests: $pass passed, $fail failed (mocked commands only)"
[ "$fail" -eq 0 ]
