#!/bin/sh
# Offline smoke for exported native artifacts. Uses nvdisasm/cuobjdump only if
# present; they are independent readers, never build inputs. Opens no device.
set -eu
dir=${1:-build/artifact-test}
fail=0
for c in "$dir"/*.cubin; do
    [ -f "$c" ] || continue
    echo "== $c"
    m=${c%.cubin}.manifest
    if command -v nvdisasm >/dev/null 2>&1; then
        nvdisasm -c "$c" >/dev/null && echo "nvdisasm: PASS" || { echo "nvdisasm: FAIL"; fail=1; }
    else
        echo "nvdisasm: NOT_RUN (not installed)"
    fi
    if command -v cuobjdump >/dev/null 2>&1; then
        cuobjdump -res-usage "$c" >/dev/null && echo "cuobjdump: PASS" || { echo "cuobjdump: FAIL"; fail=1; }
        meta=$(cuobjdump -elf "$c" 2>/dev/null || true)
        regs=$(sed -n 's/^gpr_count=//p' "$m" 2>/dev/null || true)
        if [ -n "$regs" ] && ! printf '%s\n' "$meta" | grep -q "register count: $regs"; then
            echo "REGCOUNT does not match manifest"; fail=1
        fi
        if printf '%s\n' "$meta" | grep -q "EIATTR_EXIT_INSTR_OFFSETS"; then
            echo "fabricated EXIT_INSTR_OFFSETS present"; fail=1
        fi
        if ! printf '%s\n' "$meta" | grep -q "Ordinal : 0x0"; then
            echo "KPARAM ordinal 0 missing"; fail=1
        fi
        psize=$(printf '%s\n' "$meta" | awk '/EIATTR_CBANK_PARAM_SIZE/{f=1;next} f&&/Value:/{print $2;exit}')
        pcb=$(printf '%s\n' "$meta" | awk '/EIATTR_PARAM_CBANK/{f=1;next} f&&/Value:/{print $3;exit}')
        if [ -n "$psize" ] && [ -n "$pcb" ]; then
            hi=$(( (pcb >> 16) & 0xffff ))
            ps=$(( psize ))
            [ "$hi" -eq "$ps" ] || { echo "PARAM_CBANK size $hi != CBANK_PARAM_SIZE $ps"; fail=1; }
        else
            echo "parameter cbank metadata missing"; fail=1
        fi
        shnum=$(readelf -h "$c" 2>/dev/null | sed -n 's/.*Number of section headers:[[:space:]]*//p')
        [ "$shnum" = "11" ] || { echo "unexpected section count: $shnum"; fail=1; }
    else
        echo "cuobjdump: NOT_RUN (not installed)"
    fi
    [ -f "$m" ] || { echo "manifest missing: $m"; fail=1; continue; }
    id=$(sed -n 's/^artifact_id=//p' "$m")
    echo "artifact_id=$id"
    # Every listed role digest must equal the file next to the manifest.
    for role in cubin text qmd0 qmd1 cbank_args; do
        want=$(sed -n "s/^${role}_sha256=//p" "$m")
        case $role in
            cubin) f=${c%.cubin}.cubin ;;
            text) f=${c%.cubin}.text.bin ;;
            qmd0) f=${c%.cubin}.qmd0.bin ;;
            qmd1) f=${c%.cubin}.qmd1.bin ;;
            cbank_args) f=${c%.cubin}.cbank_args.bin ;;
        esac
        got=$(sha256sum "$f" | cut -d' ' -f1)
        [ "$want" = "$got" ] || { echo "digest mismatch $role"; fail=1; }
    done
done
# The frozen v1 bounds and ids are enforced by the exporter itself.
if [ -x ./build/artifact-export ]; then
    cm=647116fc1ba1e1394982fc8ba181ce565293f160a6b56afc0f02b2128a675704
    ca=003696e422bba7274877ba9e01a61076ea0f862f32760cbaa06c60f85a115359
    if ./build/artifact-export matmul --m 16 --k 16 --n 9000 --contract-id $cm --out /tmp/oob --commit 0000000000000000000000000000000000000000 2>/dev/null; then
        echo "out-of-contract n accepted"; fail=1
    fi
    if ./build/artifact-export matmul --m 16 --k 16 --n 8 --contract-id $ca --out /tmp/oob --commit 0000000000000000000000000000000000000000 2>/dev/null; then
        echo "wrong contract id accepted"; fail=1
    fi
    if ./build/artifact-export attention --q-heads 96 --kv-heads 3 --contract-id $ca --out /tmp/oob --commit 0000000000000000000000000000000000000000 2>/dev/null; then
        echo "out-of-contract heads accepted"; fail=1
    fi
fi
[ $fail -eq 0 ] && echo "ARTIFACT SMOKE PASS"
exit $fail
