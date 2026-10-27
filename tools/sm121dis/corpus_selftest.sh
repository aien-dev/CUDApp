#!/bin/sh
# A missing or failing producer must invalidate the disassembler corpus.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/sm121dis-corpus-control.XXXXXX")
trap 'rm -rf "$WORK"' EXIT HUP INT TERM
mkdir "$WORK/bin" "$WORK/missing" "$WORK/failing"
rc=0
sh "$HERE/corpus.sh" "$WORK/bin" "$WORK/missing" > "$WORK/missing.log" 2>&1 || rc=$?
[ "$rc" -eq 127 ] && grep -q 'VERDICT FAIL: corpus producer failed:' "$WORK/missing.log" || {
    echo 'FAIL: a missing producer did not invalidate the corpus'; exit 1;
}
printf '#!/bin/sh\necho deliberate-producer-failure\nexit 42\n' > "$WORK/bin/gpu_matmul_api_test"
chmod +x "$WORK/bin/gpu_matmul_api_test"
rc=0
sh "$HERE/corpus.sh" "$WORK/bin" "$WORK/failing" > "$WORK/failing.log" 2>&1 || rc=$?
[ "$rc" -eq 42 ] && grep -q 'deliberate-producer-failure' "$WORK/failing.log" &&
    grep -q 'VERDICT FAIL: corpus producer failed:' "$WORK/failing.log" || {
    echo 'FAIL: a failing producer did not invalidate the corpus'; exit 1;
}
echo 'PASS: missing and failing corpus producers are refused'
