#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Shell orchestration only: fake compiler programs and deliberately synthetic decode rows.
# This does not compile Omega, decode GPU bytes, open a GPU or establish native execution.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/cudapp-runner-controls.XXXXXX")
trap 'rm -rf -- "$work"' EXIT HUP INT TERM
mkdir "$work/bin"
export RUNNER_CONTROL_LOG="$work/calls"

cat > "$work/bin/omc" <<'EOF'
#!/bin/sh
set -eu
src=$(cat)
case "$src" in
  *'global Aregs ulong'*) unit=address ;;
  *'fn gprepare to void'*) unit=guard ;;
  *'fn ballflags to bool'*)
    unit=body
    case "$src" in *'fn gbvec to void'*) : ;; *) exit 32 ;; esac ;;
  *'global Fails ulong'*) unit=emitter ;;
  *) unit=dump ;;
esac
printf 'compile %s\n' "$unit" >> "$RUNNER_CONTROL_LOG"
if [ "${CONTROL_STDERR:-}" = "$unit" ]; then echo 'synthetic compiler diagnostic' >&2; fi
# Emit executable bytes even when the compiler later fails. They must not run.
printf '#!/bin/sh\nprintf "run %s\\n" >> "$RUNNER_CONTROL_LOG"\n' "$unit"
printf '[ "${CONTROL_FAIL:-}" != "%s" ] || exit 23\n' "$unit"
if [ "$unit" = dump ]; then
  printf 'dd if=/dev/zero bs=640 count=1 2>/dev/null\n'
elif [ "$unit" = emitter ]; then
  printf 'printf "1 0\\n"\n'
fi
if [ "${CONTROL_COMPILE_FAIL:-}" = "$unit" ]; then exit 24; fi
EOF
cat > "$work/bin/nvdisasm" <<'EOF'
#!/bin/sh
set -eu
echo decode >> "$RUNNER_CONTROL_LOG"
[ "${CONTROL_DECODE_FAIL:-0}" = 0 ] || exit 25
# Only the runner's handwritten addressed expectations are meaningful here.
# Repeated ffff rows below are intentionally not a real instruction stream.
printf '%s\n' '/*0010*/ @P0 BRA 0x140 ;' '/*0070*/ MOV R6, 0x40 ;' \
  '/*0080*/ IMAD.WIDE.U32 R8, R6, 0x1, R4 ;' '/*01b0*/ MOV R6, 0x80 ;' \
  '/*01c0*/ IMAD.WIDE.U32 R8, R6, 0x1, R4 ;'
for pair in 'BRA 1' 'ISETP.GE.AND 2' 'LDC.64 6' 'LDCU.64 2' 'MOV 6' 'S2R 2' 'STG.E 16'; do
  set -- $pair
  n=$2
  while [ "$n" -gt 0 ]; do printf '/*ffff*/ %s ;\n' "$1"; n=$((n - 1)); done
done
EOF
chmod +x "$work/bin/omc" "$work/bin/nvdisasm"

run_case() {
  : > "$RUNNER_CONTROL_LOG"
  rc=0
  OMEGA_COMPILER="$work/bin/omc" PATH="$work/bin:$PATH" \
    sh "$root/tests/device-trace/run-emitter-checks.sh" > "$work/out" 2> "$work/err" || rc=$?
}
fail() { echo "runner control FAIL: $*" >&2; cat "$work/out" "$work/err" >&2; exit 1; }

run_case
[ "$rc" = 0 ] || fail 'positive shell orchestration'
for unit in emitter address guard body dump; do
  grep -qx "compile $unit" "$RUNNER_CONTROL_LOG" || fail "missing compile $unit"
  grep -qx "run $unit" "$RUNNER_CONTROL_LOG" || fail "missing run $unit"
done
grep -qx decode "$RUNNER_CONTROL_LOG" || fail 'missing decode'

for unit in emitter address guard body; do
  export CONTROL_FAIL="$unit"
  run_case
  [ "$rc" = 23 ] || fail "$unit failure was not propagated"
  if grep -qx decode "$RUNNER_CONTROL_LOG"; then fail "decode ran after $unit failed"; fi
done
unset CONTROL_FAIL

for unit in emitter address guard body dump; do
  export CONTROL_COMPILE_FAIL="$unit"
  run_case
  [ "$rc" != 0 ] || fail "$unit compiler failure was accepted"
  if grep -qx "run $unit" "$RUNNER_CONTROL_LOG"; then fail "$unit ran rejected compiler output"; fi
done
unset CONTROL_COMPILE_FAIL

for unit in emitter address guard body dump; do
  export CONTROL_STDERR="$unit"
  run_case
  [ "$rc" != 0 ] || fail "$unit compiler diagnostic was accepted"
  if grep -qx "run $unit" "$RUNNER_CONTROL_LOG"; then fail "$unit ran with compiler diagnostics"; fi
done
unset CONTROL_STDERR

export CONTROL_DECODE_FAIL=1
run_case
[ "$rc" = 25 ] || fail 'decoder failure was not propagated'
unset CONTROL_DECODE_FAIL
echo 'RUNNER ORCHESTRATION CONTROLS PASS (synthetic tools; no native or GPU execution)'
