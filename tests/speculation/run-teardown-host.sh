#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Disclosed development orchestration; test behavior is handwritten native Omega.
set -u
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != aarch64 ]; then
  echo 'NOT_RUN: teardown native-host test needs Linux AArch64 and the pinned native compiler' >&2
  exit 77
fi
if [ "$#" -ne 1 ]; then
  echo 'usage: run-teardown-host.sh /absolute/path/to/pinned-omc' >&2
  exit 64
fi
omc=$1
expected=9749f38841a55917d41924a68bce31463ec2339910bddd5e6cbf6ea4420551db
actual=$(sha256sum "$omc") || exit 74
actual=${actual%% *}
if [ "$actual" != "$expected" ]; then
  echo 'REFUSED: compiler digest differs from the d7390d88 native-device receipt' >&2
  exit 69
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd) || exit 74
out=$(mktemp -d "${TMPDIR:-/tmp}/cudapp-teardown-host.XXXXXX") || exit 74
echo "Native-host test outputs: $out"
base=5e037c7e3d0226abadaa10af9d44bc6339109b3f
git -C "$root" show "$base:native/nvrm/context.om" > "$out/before.om" || exit 74
cp "$root/native/nvrm/context.om" "$out/after.om" || exit 74
for arm in before after; do
  # Substitute only OS-effect verbs in a private unit. Preserve the state-machine code.
  awk '
    $2 == "=" && ($3 == "devcall" || $3 == "devctl" || $3 == "uvmctl") { print $1 " = tdriver"; next }
    $2 == "=" && $3 == "fopen" { print $1 " = tunexpected"; next }
    { for (i=1; i<=NF; i++) { if ($i == "mapdrop") $i="tdrop"; if ($i == "fclose") $i="tclose" }; print }
  ' "$out/$arm.om" > "$out/$arm-unit.om" || exit 74
  cat "$root/tests/speculation/teardown-host-test.om" >> "$out/$arm-unit.om" || exit 74
  # A new or differently shaped device effect must not silently escape the shim.
  awk '$1 != "note" { for (i=1; i<=NF; i++) if ($i == "devcall" || $i == "devctl" || $i == "uvmctl" || $i == "fopen" || $i == "fclose" || $i == "mapdrop") exit 1 }' "$out/$arm-unit.om" || {
    echo 'REFUSED: unbound OS effect in teardown fixture' >&2
    exit 69
  }
  timeout 30 "$omc" < "$out/$arm-unit.om" > "$out/$arm.elf" 2> "$out/$arm.compile.log"
  rc=$?
  if [ "$rc" -ne 0 ] || [ ! -s "$out/$arm.elf" ]; then
    echo "FAIL: $arm fixture did not compile successfully (exit $rc)" >&2
    exit 1
  fi
  chmod 700 "$out/$arm.elf" || exit 74
  timeout 10 "$out/$arm.elf" > "$out/$arm.stdout" 2> "$out/$arm.stderr"
  rc=$?
  printf '%s\n' "$rc" > "$out/$arm.exit" || exit 74
  if [ "$arm" = before ]; then
    if [ "$rc" -ne 1 ] || ! grep -q '^NATIVE_TEARDOWN_HOST_FAIL$' "$out/$arm.stderr"; then
      echo 'FAIL: original source did not fail by the intended test assertion' >&2
      exit 1
    fi
  elif [ "$rc" -ne 0 ] || ! grep -q '^NATIVE_TEARDOWN_HOST_PASS$' "$out/$arm.stdout"; then
    echo 'FAIL: repaired source did not pass the teardown fixture' >&2
    exit 1
  fi
done
sha256sum "$out"/*.om "$out"/*.elf || exit 74
echo 'PASS: original cleanup fails its assertion; repaired cleanup passes refusal, retry and close-once controls'
