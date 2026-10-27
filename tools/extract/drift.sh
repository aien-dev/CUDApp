#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# drift.sh: fails when Omega's copy of a GPU file moved away from CUDA++ after the cutover.
#
#   tools/extract/drift.sh OMEGA_DIR OMEGA_REF [BASELINE]
#   tools/extract/drift.sh --self-test OMEGA_DIR OMEGA_REF [BASELINE]
#
# CUDA++ is the source of truth for the GPU stack (aien-dev/CUDApp); omega-mv's GPU files are superseded.
# BASELINE (default tools/extract/drift-baseline.tsv: omega_path, omega_blob) records Omega's blob of every
# GPU path at the cutover commit. A path DRIFTS when, at OMEGA_REF,
#   - Omega holds it, and
#   - Omega's blob differs from CUDA++'s blob at HEAD (map.sh status differs or omega-only), and
#   - Omega's blob is not the one recorded in BASELINE (a new path counts: it has no baseline blob).
# So the differences that existed at the cutover stay quiet, a GPU edit landed in Omega but not in CUDA++
# is red, and porting it to CUDA++ (same blob) turns it green again.
# Output: one "DRIFT path status omega_blob" line per drifting path, then "DRIFT-GATE: PASS|FAIL (...)".
# Exit 0 on PASS, 1 on drift, 2 on a broken setup.
#
# --self-test runs two negative controls that must both come out red, then the real check:
#   1. baseline control: the baseline blob of one cutover-time `differs` path is replaced; the gate must name it.
#   2. edit control: in a scratch clone of OMEGA_DIR a commit appends one line to a GPU source that is `same`
#      in both repos; the gate run against that commit must name exactly that file.
# A gate that stays green under either control is broken and the self-test exits 1.
set -euo pipefail
export LC_ALL=C

selftest=0
if [[ ${1:-} == --self-test ]]; then selftest=1; shift; fi
omega=${1:?omega dir}; oref=${2:?omega ref}
here=$(cd "$(dirname "$0")" && pwd)
baseline=${3:-$here/drift-baseline.tsv}
[[ -s $baseline ]] || { echo "DRIFT-GATE: NOT_RUN (baseline $baseline missing)"; exit 2; }

scratch=$(mktemp -d); trap 'rm -rf "$scratch"' EXIT

# check OMEGA_DIR OMEGA_REF BASELINE: prints DRIFT lines and the verdict; returns 0 PASS / 1 FAIL.
check() {
    local dir=$1 ref=$2 base=$3 map=$scratch/map.$RANDOM
    "$here/map.sh" "$dir" "$ref" HEAD > "$map"
    awk -F'\t' 'NR==FNR { if ($1 !~ /^#/) b[$1]=$2; next }
        FNR==1 { next }
        $4 != "-" && ($3 == "differs" || $3 == "omega-only") && b[$1] != $4 {
            printf "DRIFT\t%s\t%s\t%s\n", $1, $3, $4 }' "$base" "$map" > "$map.drift"
    cat "$map.drift"
    local n rows
    n=$(wc -l < "$map.drift"); rows=$(($(wc -l < "$map") - 1))
    if [[ $n -eq 0 ]]; then
        echo "DRIFT-GATE: PASS ($rows GPU paths, omega $(git -C "$dir" rev-parse --short=12 "$ref^{commit}"), none moved away from CUDA++ since the baseline)"
        return 0
    fi
    echo "DRIFT-GATE: FAIL ($n of $rows GPU paths changed in omega $(git -C "$dir" rev-parse --short=12 "$ref^{commit}") and not in CUDA++)"
    return 1
}

if [[ $selftest -eq 0 ]]; then
    check "$omega" "$oref" "$baseline" && exit 0 || exit 1
fi

fail=0
"$here/map.sh" "$omega" "$oref" HEAD > "$scratch/now"

# Control 1: baseline blob of one cutover-time `differs` path replaced.
victim=$(awk -F'\t' 'NR==FNR { if ($1 !~ /^#/) b[$1]=$2; next } FNR>1 && $3=="differs" && b[$1]==$4 { print $1; exit }' \
    "$baseline" "$scratch/now")
if [[ -z $victim ]]; then echo "control 1: NOT_RUN (no differs path recorded in the baseline)"; fail=1
else
    awk -F'\t' -v OFS='\t' -v v="$victim" '$1==v { $2="0000000000000000000000000000000000000000" } 1' "$baseline" > "$scratch/base1"
    out=$(check "$omega" "$oref" "$scratch/base1" || true)
    if grep -qxF "DRIFT	$victim	differs	$(awk -F'\t' -v v="$victim" '$1==v {print $4}' "$scratch/now")" <<< "$out" \
        && grep -q '^DRIFT-GATE: FAIL' <<< "$out"; then echo "control 1: red as required ($victim)"
    else echo "control 1: BROKEN, gate stayed green with a tampered baseline for $victim"; echo "$out"; fail=1; fi
fi

# Control 2: a real Omega commit that edits one GPU source CUDA++ holds byte-identically.
target=$(awk -F'\t' 'FNR>1 && $3=="same" && $1 ~ /^src\/omega_gpu_.*\.c$/ { print $1; exit }' "$scratch/now")
if [[ -z $target ]]; then echo "control 2: NOT_RUN (no same src/omega_gpu_*.c path)"; fail=1
else
    git clone -q --shared --no-checkout "$omega" "$scratch/om"
    git -C "$scratch/om" checkout -q --detach "$(git -C "$omega" rev-parse "$oref^{commit}")"
    echo "/* drift.sh self-test edit */" >> "$scratch/om/$target"
    git -C "$scratch/om" -c user.name=drift-selftest -c user.email=drift@selftest.invalid commit -q -am "drift self-test"
    out=$(check "$scratch/om" HEAD "$baseline" || true)
    if [[ $(grep -c '^DRIFT	' <<< "$out" || true) -ge 1 ]] && grep -q "^DRIFT	$target	differs	" <<< "$out" \
        && grep -q '^DRIFT-GATE: FAIL' <<< "$out"; then echo "control 2: red as required ($target edited in a scratch Omega commit)"
    else echo "control 2: BROKEN, gate stayed green after an Omega edit to $target"; echo "$out"; fail=1; fi
fi

echo "real check:"
check "$omega" "$oref" "$baseline" || fail=1
[[ $fail -eq 0 ]] && echo "SELF-TEST: PASS (both controls red, real check green)" || echo "SELF-TEST: FAIL"
exit "$fail"
