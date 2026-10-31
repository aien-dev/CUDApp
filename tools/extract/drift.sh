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
# Output: one "DRIFT path status omega_blob" line per drifting path, a "DIVERGENCE" summary of the whole map
# (blob compare; `differs` split into license-header-only and code), then "DRIFT-GATE: PASS|FAIL (...)".
# Exit 0 on PASS, 1 on drift, 2 on a broken setup.
#
# --self-test runs negative controls that must all come out red, then the real check:
#   1. baseline control: the baseline blob of one cutover-time `differs` path is replaced; the gate must name it.
#   2. edit control: in a scratch clone of OMEGA_DIR a commit appends one line to a GPU source whose Omega blob
#      is still the baseline one (preferring one `same` in both repos); the gate must name that file (other DRIFT
#      lines, from real drift at OMEGA_REF, may appear too; the real check below fails on those).
#   3. each Ampere numeric cutover source/header is changed in a scratch Omega commit and must be named.
# A gate that stays green under either control is broken and the self-test exits 1.
set -euo pipefail
export LC_ALL=C

selftest=0
if [[ ${1:-} == --self-test ]]; then selftest=1; shift; fi
omega=${1:?omega dir}; oref=${2:?omega ref}
here=$(cd "$(dirname "$0")" && pwd)
baseline=${3:-$here/drift-baseline.tsv}
[[ -f $baseline && -r $baseline && -s $baseline ]] || { echo "DRIFT-GATE: NOT_RUN (baseline $baseline is not a readable nonempty file)"; exit 2; }

scratch=$(mktemp -d); trap 'rm -rf "$scratch"' EXIT
cudapp=$(git -C "$here" rev-parse --show-toplevel)

# summary MAP OMEGA_DIR: one DIVERGENCE line. Shared = path present in both repos. A `differs` file is
# header-only when every changed line is a licence-header line (the SPDX/Apache boilerplate).
summary() {
    local map=$1 dir=$2 hdr=0 code=0 op cp st ob cb rc kind
    while IFS=$'\t' read -r op cp st ob cb; do
        [[ $st == differs ]] || continue
        # Process substitutions hide cat-file failures. Materialize both blobs and
        # the whole diff: status 1 means different, not a pipeline failure, and no
        # early-exit consumer can give diff a broken pipe.
        git -C "$dir" cat-file blob "$ob" > "$scratch/omega-blob" || return 2
        git -C "$cudapp" cat-file blob "$cb" > "$scratch/cudapp-blob" || return 2
        rc=0
        diff "$scratch/omega-blob" "$scratch/cudapp-blob" > "$scratch/changes" || rc=$?
        if [[ $rc -ne 1 ]]; then
            echo "summary: expected differing blobs for $op, diff returned $rc" >&2
            return 2
        fi
        kind=$(awk '/^[<>]/ { changed=1
                if (tolower($0) !~ /spdx|copyright|licen[cs]e|apache|llvm|applicable law|as is|you may not use|obtain a copy|distributed under|without warranties|limitations under|^[<>] *(\/\/|#|\/?\*|\*\/)? *$/) code=1 }
            END { print (changed && !code ? "header" : "code") }' "$scratch/changes") || return 2
        if [[ $kind == code ]]; then code=$((code + 1)); else hdr=$((hdr + 1)); fi
    done < "$map"
    awk -F'\t' -v hdr="$hdr" -v code="$code" 'FNR > 1 { n[$3]++ }
        END { sh = n["same"] + n["differs"]
              printf "DIVERGENCE: %d shared, %d same, %d differ (%d licence-header only, %d code); %d cudapp-only, %d omega-only\n",
                     sh, n["same"], n["differs"], hdr, code, n["cudapp-only"], n["omega-only"] }' "$map"
}
# check OMEGA_DIR OMEGA_REF BASELINE: returns 0 PASS / 1 FAIL / 2 broken setup.
check() {
    local dir=$1 ref=$2 base=$3 map=$scratch/map.$RANDOM
    "$here/map.sh" "$dir" "$ref" HEAD > "$map" || { echo "DRIFT-GATE: NOT_RUN (map failed)"; return 2; }
    awk -F'\t' 'NR==FNR { if ($1 !~ /^#/) b[$1]=$2; next }
        FNR==1 { next }
        $4 != "-" && ($3 == "differs" || $3 == "omega-only") && b[$1] != $4 {
            printf "DRIFT\t%s\t%s\t%s\n", $1, $3, $4 }' "$base" "$map" > "$map.drift" \
        || { echo "DRIFT-GATE: NOT_RUN (baseline comparison failed)"; return 2; }
    cat "$map.drift"
    summary "$map" "$dir" || { echo "DRIFT-GATE: NOT_RUN (summary failed)"; return 2; }
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
    if check "$omega" "$oref" "$baseline"; then exit 0; else exit $?; fi
fi

fail=0
"$here/map.sh" "$omega" "$oref" HEAD > "$scratch/now" || { echo "DRIFT-GATE: NOT_RUN (map failed)"; exit 2; }

# Control 1: baseline blob of one cutover-time `differs` path replaced.
victim=$(awk -F'\t' 'NR==FNR { if ($1 !~ /^#/) b[$1]=$2; next } FNR>1 && $3=="differs" && b[$1]==$4 { print $1; exit }' \
    "$baseline" "$scratch/now")
if [[ -z $victim ]]; then echo "control 1: NOT_RUN (no differs path recorded in the baseline)"; fail=1
else
    awk -F'\t' -v OFS='\t' -v v="$victim" '$1==v { $2="0000000000000000000000000000000000000000" } 1' "$baseline" > "$scratch/base1"
    rc=0; out=$(check "$omega" "$oref" "$scratch/base1") || rc=$?
    if [[ $rc -eq 2 ]]; then echo "$out"; exit 2; fi
    if grep -qxF "DRIFT	$victim	differs	$(awk -F'\t' -v v="$victim" '$1==v {print $4}' "$scratch/now")" <<< "$out" \
        && grep -q '^DRIFT-GATE: FAIL' <<< "$out"; then echo "control 1: red as required ($victim)"
    else echo "control 1: BROKEN, gate stayed green with a tampered baseline for $victim"; echo "$out"; fail=1; fi
fi

# Control 2: a real Omega commit that edits one GPU source still at its cutover blob.
target=$(awk -F'\t' 'NR==FNR { if ($1 !~ /^#/) b[$1]=$2; next }
    FNR>1 && $1 ~ /^src\/omega_gpu_.*\.c$/ && b[$1]==$4 && ($3=="same" || $3=="differs") { if ($3=="same") { print $1; exit } else if (!d) d=$1 }
    END { if (d) print d }' "$baseline" "$scratch/now" | head -1)
if [[ -z $target ]]; then echo "control 2: NOT_RUN (no src/omega_gpu_*.c still at its baseline blob)"; fail=1
else
    git clone -q --shared --no-checkout "$omega" "$scratch/om"
    git -C "$scratch/om" checkout -q --detach "$(git -C "$omega" rev-parse "$oref^{commit}")"
    echo "/* drift.sh self-test edit */" >> "$scratch/om/$target"
    git -C "$scratch/om" -c user.name=drift-selftest -c user.email=drift@selftest.invalid commit -q -am "drift self-test"
    rc=0; out=$(check "$scratch/om" HEAD "$baseline") || rc=$?
    if [[ $rc -eq 2 ]]; then echo "$out"; exit 2; fi
    if [[ $(grep -c '^DRIFT	' <<< "$out" || true) -ge 1 ]] && grep -q "^DRIFT	$target	differs	" <<< "$out" \
        && grep -q '^DRIFT-GATE: FAIL' <<< "$out"; then echo "control 2: red as required ($target edited in a scratch Omega commit)"
    else echo "control 2: BROKEN, gate stayed green after an Omega edit to $target"; echo "$out"; fail=1; fi
fi

# Control 3: pinning an intentional x86 fork must not hide subsequent edits to
# any retained Omega numeric CPU-oracle source or public header.
for numeric in src/omega_numeric.c src/omega_numeric.h src/omega_numeric_reduce.c src/omega_numeric_reduce.h; do
    if [[ ! -d $scratch/om/.git ]]; then
        git clone -q --shared --no-checkout "$omega" "$scratch/om"
        git -C "$scratch/om" checkout -q --detach "$(git -C "$omega" rev-parse "$oref^{commit}")"
    fi
    if ! git -C "$omega" cat-file -e "$oref:$numeric" 2>/dev/null; then
        echo "control 3: $numeric is not in $oref (fixture Omega); this row is not exercised"; continue
    fi
    echo "/* Ampere numeric drift mutant */" >> "$scratch/om/$numeric"
    git -C "$scratch/om" -c user.name=drift-selftest -c user.email=drift@selftest.invalid commit -q -am "numeric cutover drift self-test"
    out=$(check "$scratch/om" HEAD "$baseline" || true)
    blob=$(git -C "$scratch/om" rev-parse "HEAD:$numeric")
    if grep -qxF "DRIFT	$numeric	differs	$blob" <<< "$out" && grep -q '^DRIFT-GATE: FAIL' <<< "$out"; then
        echo "control 3: red as required ($numeric changed beyond its explicit cutover pin)"
    else
        echo "control 3: BROKEN, gate hid a numeric source/header change: $numeric"
        echo "$out"; fail=1
    fi
done

echo "real check:"
check "$omega" "$oref" "$baseline" || { rc=$?; [[ $rc -ne 2 ]] || exit 2; fail=1; }
[[ $fail -eq 0 ]] && echo "SELF-TEST: PASS (all controls red, real check green)" || echo "SELF-TEST: FAIL"
exit "$fail"
