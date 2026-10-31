#!/bin/sh
# Offline checks for the device-side trace emitter (CUDApp #82 first slice).
#
# Two layers, deliberately separate:
#   1. emitter/address/guard/body-test.om -- host checks compiled by the pinned seed compiler.
#   2. nvdisasm 13.0 on the emitted words   -- an outside oracle. It is not part of the
#      delivered closure and no GPU is touched; it only confirms that the emitted words
#      decode to the intended instructions. Its listing stays in the temporary work
#      directory: AGENTS.md forbids committing NVIDIA-tool output, so the expectations
#      below are written by hand (opcode counts and the address-formation lines of the
#      #85 repair: MOV of the record offset, then IMAD.WIDE.U32 base + offset).
#
# Usage: OMEGA_COMPILER=/path/to/seed/out/omc sh tests/device-trace/run-emitter-checks.sh
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
compiler=${OMEGA_COMPILER:-}
[ -n "$compiler" ] && [ -x "$compiler" ] || {
  echo 'OMEGA_COMPILER must name the pinned seed compiler (seed/out/omc)' >&2
  exit 69
}
work=${TMPDIR:-/tmp}/cudapp-emitter-check.$$
mkdir -p "$work"
trap 'rm -rf -- "$work"' EXIT HUP INT TERM

echo "compiler $(sha256sum "$compiler" | cut -d' ' -f1)"

compile_unit() {
  src=$1
  out=$2
  if "$compiler" < "$src" > "$out.elf" 2> "$out.compile.stderr"; then
    :
  else
    rc=$?
    cat "$out.compile.stderr" >&2
    return "$rc"
  fi
  [ ! -s "$out.compile.stderr" ] || { cat "$out.compile.stderr" >&2; return 1; }
  chmod +x "$out.elf"
}

# 1. All semantic units must pass before any external decode is attempted.
for name in emitter address guard body; do
  : > "$work/$name.om"
  if [ "$name" = body ]; then
    cat "$root/native/blackwell/vector.om" >> "$work/$name.om"
  fi
  cat "$root/native/blackwell/trace-emit.om" "$root/tests/device-trace/$name-test.om" >> "$work/$name.om"
  compile_unit "$work/$name.om" "$work/$name"
  if "$work/$name.elf" > "$work/$name.out"; then
    :
  else
    rc=$?
    cat "$work/$name.out" >&2
    echo "host unit $name failed: exit $rc" >&2
    exit "$rc"
  fi
  if [ "$name" = emitter ]; then
    fails=$(awk '$2 == 1 { n++ } END { print n + 0 }' "$work/$name.out")
    checks=$(wc -l < "$work/$name.out")
    echo "host checks: $checks, failures: $fails"
    [ "$fails" = 0 ] || { cat "$work/$name.out" >&2; exit 1; }
  fi
  echo "host unit $name: PASS"
done

# 2. instruction decode of the two marker blocks (no kernel words between them)
cat > "$work/dump.om" <<'EOF'
global Ob uchar 1024
global Obn ulong 1
fn main to int
put gbinlen 0 0
put gtchecks 0 1
ok = gtewrap 1 4
if not ok
ret 3
end
for i ulong get gboutlen 0
c = get gbout i
put Ob get Obn 0 c
put Obn 0 add get Obn 0 1
end
r = write 1 Ob 0 get Obn 0
ret 0
end
EOF
cat "$root/native/blackwell/trace-emit.om" "$work/dump.om" > "$work/dump_full.om"
compile_unit "$work/dump_full.om" "$work/dump"
"$work/dump.elf" > "$work/words.bin"
echo "words: $(wc -c < "$work/words.bin") bytes, $(( $(wc -c < "$work/words.bin") / 16 )) instructions"
[ "$(wc -c < "$work/words.bin")" = 640 ] || { echo 'expected 40 instructions (640 bytes)' >&2; exit 1; }

nvdisasm --binary SM121 "$work/words.bin" > "$work/got.sass"
# Opcode histogram of the two marker blocks (20 instructions each).
awk '$1 ~ /^\/\*[0-9a-f]+\*\/$/ { op = $2; if (op ~ /^@/) op = $3; n[op]++ }
     END { for (o in n) print o, n[o] }' "$work/got.sass" | LC_ALL=C sort > "$work/ops.txt"
LC_ALL=C sort > "$work/ops.want" <<'EOF'
BRA 2
IMAD.WIDE.U32 2
ISETP.GE.AND 2
LDC.64 6
LDCU.64 2
MOV 8
S2R 2
STG.E 16
EOF
bad=0
if ! diff -u "$work/ops.want" "$work/ops.txt"; then
  echo 'opcode counts differ from the hand-written expectation' >&2
  bad=1
fi
# Record address = capture base (R4:R5) + record offset, full 64-bit width.
for want in '0010 @P0 BRA 0x140' '0070 MOV R6, 0x40' '0080 IMAD.WIDE.U32 R8, R6, 0x1, R4' \
            '01b0 MOV R6, 0x80' '01c0 IMAD.WIDE.U32 R8, R6, 0x1, R4'; do
  off=${want%% *}; ins=${want#* }
  got=$(awk -v o="/*$off*/" '$1 == o { $1 = ""; sub(/^ +/, ""); sub(/ *;$/, ""); print }' "$work/got.sass")
  if [ "$got" != "$ins" ]; then
    echo "offset 0x$off: want '$ins', got '$got'" >&2
    bad=1
  fi
done
[ "$bad" = 0 ] || exit 1
echo 'nvdisasm decode matches the hand-written expectations (40 instructions)'
