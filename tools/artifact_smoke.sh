#!/bin/sh
# Offline smoke for exported native artifacts. Uses nvdisasm/cuobjdump only if
# present; they are independent readers, never build inputs. Opens no device.
set -eu
dir=${1:-build/artifact-test}
fail=0
for c in "$dir"/*.cubin; do
    [ -f "$c" ] || continue
    echo "== $c"
    if command -v nvdisasm >/dev/null 2>&1; then
        nvdisasm -c "$c" >/dev/null && echo "nvdisasm: PASS" || { echo "nvdisasm: FAIL"; fail=1; }
    else
        echo "nvdisasm: NOT_RUN (not installed)"
    fi
    if command -v cuobjdump >/dev/null 2>&1; then
        cuobjdump -res-usage "$c" >/dev/null && echo "cuobjdump: PASS" || { echo "cuobjdump: FAIL"; fail=1; }
    else
        echo "cuobjdump: NOT_RUN (not installed)"
    fi
    m=${c%.cubin}.manifest
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
[ $fail -eq 0 ] && echo "ARTIFACT SMOKE PASS"
exit $fail
