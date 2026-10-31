#!/bin/sh
# Host-only fixture: identical output cannot excuse a failed decoder process.
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/sm121dis-check-control.XXXXXX")
trap 'rm -rf "$WORK"' EXIT HUP INT TERM
mkdir "$WORK/kernels"
# The fixture decoders supply text, not real machine decoding or hardware evidence.
printf '\000' > "$WORK/kernels/a.bin"
printf '\000' > "$WORK/kernels/b.bin"
cat > "$WORK/oracle" <<'ORACLE'
#!/bin/sh
if [ "$1" = --version ]; then echo 'release 13.0, V13.0'; exit 0; fi
printf '/*0000*/ EXIT ; /* 0x0000000000000000 */\n/* 0x0000000000000000 */\n'
ORACLE
cat > "$WORK/decoder" <<'DECODER'
#!/bin/sh
printf '0000  0000000000000000\t0000000000000000 EXIT ;\n'
echo deliberate-decoder-diagnostic >&2
exit "${DECODER_RC:-0}"
DECODER
chmod +x "$WORK/oracle" "$WORK/decoder"
for jobs in 1 2; do
    for decoder_rc in 0 7 127; do
        rc=0
        NVDISASM="$WORK/oracle" SM121DIS_JOBS=$jobs DECODER_RC=$decoder_rc \
            sh "$HERE/check.sh" "$WORK/decoder" "$WORK/kernels" "$WORK/results" > "$WORK/log" 2>&1 || rc=$?
        if [ "$decoder_rc" -eq 0 ]; then
            [ "$rc" -eq 0 ] && grep -qx 'VERDICT PASS' "$WORK/log" &&
                grep -qx 'kernels: 2, words: 2, failed kernels: 0' "$WORK/log" || {
                cat "$WORK/log"; echo 'FAIL: successful decoder refused'; exit 1;
            }
        else
            [ "$rc" -eq 1 ] && grep -qx 'VERDICT FAIL' "$WORK/log" &&
                grep -qx 'kernels: 2, words: 2, failed kernels: 2' "$WORK/log" &&
                grep -q "decoder exited $decoder_rc" "$WORK/log" &&
                grep -q 'deliberate-decoder-diagnostic' "$WORK/log" &&
                ! grep -q '^\[PASS\]' "$WORK/log" || {
                cat "$WORK/log"; echo 'FAIL: nonzero decoder exit accepted or not diagnosed'; exit 1;
            }
        fi
    done
done
echo 'PASS: decoder exit controls (serial and parallel)'
