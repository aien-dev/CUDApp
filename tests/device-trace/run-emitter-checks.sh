#!/bin/sh
# Offline checks for the device-side trace emitter (CUDApp #82 first slice).
#
# Two layers, deliberately separate:
#   1. tests/device-trace/emitter-test.om  -- host checks compiled by the pinned seed compiler.
#   2. nvdisasm 13.0 on the emitted words   -- an outside oracle. It is not part of the
#      delivered closure and no GPU is touched; it only confirms that every emitted word
#      decodes to the intended instruction.
#
# Usage: OMEGA_COMPILER=/path/to/seed/out/omc sh tests/device-trace/run-emitter-checks.sh
#        add --update to rewrite expected.sass from the current emitter output.
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

# 1. host checks
cat "$root/native/blackwell/trace-emit.om" "$root/tests/device-trace/emitter-test.om" > "$work/test.om"
"$compiler" < "$work/test.om" > "$work/test.elf"
chmod +x "$work/test.elf"
"$work/test.elf" > "$work/test.out"
fails=$(awk '$2 == 1 { n++ } END { print n + 0 }' "$work/test.out")
checks=$(wc -l < "$work/test.out")
echo "host checks: $checks, failures: $fails"
[ "$fails" = 0 ] || { cat "$work/test.out" >&2; exit 1; }

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
"$compiler" < "$work/dump_full.om" > "$work/dump.elf"
chmod +x "$work/dump.elf"
"$work/dump.elf" > "$work/words.bin"
echo "words: $(wc -c < "$work/words.bin") bytes, $(( $(wc -c < "$work/words.bin") / 16 )) instructions"
[ "$(wc -c < "$work/words.bin")" = 640 ] || { echo 'expected 40 instructions (640 bytes)' >&2; exit 1; }

nvdisasm --binary SM121 "$work/words.bin" > "$work/got.sass"
if [ "${1:-}" = "--update" ]; then
  cp "$work/got.sass" "$root/tests/device-trace/expected.sass"
  echo 'expected.sass updated'
  exit 0
fi
if diff -u "$root/tests/device-trace/expected.sass" "$work/got.sass"; then
  echo 'nvdisasm decode matches expected.sass'
else
  echo 'nvdisasm decode differs from expected.sass' >&2
  exit 1
fi
