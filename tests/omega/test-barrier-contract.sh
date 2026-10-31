#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# The production barrier is dsb sy on AArch64 and a real fence on other hosts.
# A raw dsb sy outside an __aarch64__ region fails this check. A fixture with
# an unguarded dsb sy is the negative control.
set -eu
R=$(cd "$(dirname "$0")/../.." && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT HUP INT TERM
fail() { echo "barrier-contract: $*" >&2; exit 1; }

# shellcheck disable=SC2016
scan() {
  awk '
    function in_aarch64(   i) {
      for (i = 1; i <= sp; i++) if (st[i]) return 1
      return 0
    }
    /^[[:space:]]*#[[:space:]]*if/ {
      sp++
      st[sp] = ($0 ~ /__aarch64__/) ? 1 : 0
      next
    }
    /^[[:space:]]*#[[:space:]]*else/ {
      if (sp >= 1) st[sp] = st[sp] ? 0 : 1
      next
    }
    /^[[:space:]]*#[[:space:]]*elif/ {
      if (sp >= 1) st[sp] = ($0 ~ /__aarch64__/) ? 1 : 0
      next
    }
    /^[[:space:]]*#[[:space:]]*endif/ {
      if (sp >= 1) sp--
      next
    }
    /__asm__ volatile\("dsb sy"/ {
      if (!in_aarch64()) {
        printf "%s:%d: dsb sy is outside #if defined(__aarch64__)\n", FILENAME, FNR
        bad = 1
      }
    }
    END { exit bad ? 1 : 0 }
  ' "$@"
}

hdr=$R/nvrm/cpu_barrier.h
[ -f "$hdr" ] || fail "missing $hdr"
awk '
  /#if defined\(__aarch64__\)/ { arm=1; other=0; next }
  /#else/ { if (arm) { arm=0; other=1 } next }
  /#endif/ { arm=0; other=0; next }
  arm && /dsb sy/ { saw_dsb=1 }
  other && /__atomic_thread_fence\(__ATOMIC_SEQ_CST\)/ { saw_fence=1 }
  other && /\(void\)0/ { saw_nop=1 }
  END {
    if (!saw_dsb) { print "AArch64 branch has no dsb sy"; exit 1 }
    if (!saw_fence) { print "portable branch has no seq_cst fence"; exit 1 }
    if (saw_nop) { print "portable branch contains (void)0"; exit 1 }
  }
' "$hdr" || fail "nvrm/cpu_barrier.h broke the production split"
grep -q 'nvrm_cpu_barrier();' "$R/src/omega_blackwell_engine.c" || fail "engine does not call nvrm_cpu_barrier"
grep -q 'nvrm_cpu_barrier();' "$R/src/omega_blackwell_submit.c" || fail "submit does not call nvrm_cpu_barrier"

set --
for dir in src nvrm m16; do
  for f in $(find "$R/$dir" -type f \( -name '*.c' -o -name '*.h' \)); do
    set -- "$@" "$f"
  done
done
scan "$@" || fail "tree has an unguarded dsb sy"

printf '%s\n' '__asm__ volatile("dsb sy" ::: "memory");' > "$T/unguarded.c"
if scan "$T/unguarded.c" >"$T/neg.log" 2>&1; then
  fail "negative control accepted an unguarded dsb sy"
fi
grep -q 'outside #if defined(__aarch64__)' "$T/neg.log" || fail "negative control diagnostic"
printf '%s\n' '#if defined(__aarch64__)' '__asm__ volatile("dsb sy" ::: "memory");' '#endif' > "$T/guarded.c"
scan "$T/guarded.c" || fail "guarded fixture was rejected"

cat > "$T/use.c" <<EOF
#include "cpu_barrier.h"
void use_barrier(void) { nvrm_cpu_barrier(); }
EOF
cc=${CC:-gcc}
"$cc" -std=gnu11 -Wall -Wextra -Werror -c -I"$R/nvrm" -o "$T/use.o" "$T/use.c"
objdump -d "$T/use.o" > "$T/use.s"
host=$(uname -m)
case "$host" in
  x86_64)
    grep -Eq 'mfence|[[:space:]]lock[[:space:]]' "$T/use.s" || fail "x86-64 object has no hardware fence"
    if grep -q 'dsb' "$T/use.s"; then fail "x86-64 object contains dsb"; fi
    echo "barrier codegen: PASS ($host seq_cst fence, no dsb)"
    ;;
  aarch64|arm64)
    grep -Eq 'dsb[[:space:]]+sy' "$T/use.s" || fail "AArch64 object has no dsb sy"
    echo "barrier codegen: PASS ($host dsb sy)"
    ;;
  *)
    echo "barrier codegen: NOT_RUN (host is $host; source contract still checked)"
    ;;
esac

if command -v aarch64-linux-gnu-gcc >/dev/null 2>&1 && command -v aarch64-linux-gnu-objdump >/dev/null 2>&1; then
  aarch64-linux-gnu-gcc -std=gnu11 -Wall -Wextra -Werror -c -I"$R/nvrm" -o "$T/use-arm.o" "$T/use.c"
  aarch64-linux-gnu-objdump -d "$T/use-arm.o" > "$T/use-arm.s"
  grep -Eq 'dsb[[:space:]]+sy' "$T/use-arm.s" || fail "AArch64 cross object has no dsb sy"
  echo "barrier codegen: PASS (aarch64-linux-gnu-gcc cross compile emits dsb sy)"
elif [ "$host" != aarch64 ] && [ "$host" != arm64 ]; then
  echo "AArch64 barrier codegen: NOT_RUN (aarch64-linux-gnu-gcc is not installed; source contract checked)"
fi

echo "barrier-contract: PASS"
