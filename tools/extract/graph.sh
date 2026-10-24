#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# graph.sh: decide which Omega files belong to CUDA++ (the GPU half) from the
# include graph plus the header/source link rule.
#
#   tools/extract/graph.sh OMEGA_CHECKOUT OUT_DIR
#
# Inputs next to this script: stay.txt (Omega bridges the walk must not cross)
# and shared.txt (shared base kept in both repos).
#
# Seeds: src/omega_blackwell_* and src/omega_gpu_* (docs/OMEGA.md section 1:
# GPU toolchain, GPU runtime, libraries). From the seeds it follows every
# `#include "x"` and, for every header it reaches, the matching .c file (the link
# rule: a header without its body does not build). Files of the languages
# (src/compiler, src/checker2, tools/omegese) are never pulled across; if the GPU
# part needs one, it is written to conflicts.txt and the run fails.
#
# Tests: a file under tests/ moves when it includes at least one GPU file and
# everything it includes is GPU, test-local, or the driver client. Tests that
# also include Omega-only files stay in Omega (they are integration tests of the
# two halves) and are listed in tests-stay.txt.
#
# Output (OUT_DIR): gpu-src.txt, gpu-tests.txt, tests-stay.txt, consumers.txt
# (Omega files that include a GPU header and stay), driver-includes.txt,
# unresolved.txt, conflicts.txt, filemap.tsv (old path -> new path).
set -euo pipefail
export LC_ALL=C

root=${1:?omega checkout}
out=${2:?output dir}
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
cd "$root"
declare -A stay=() shared=()
while read -r p; do [[ -n $p && $p != \#* ]] && stay[$p]=1; done < "$here/stay.txt"
while read -r p; do [[ -n $p && $p != \#* ]] && shared[$p]=1; done < "$here/shared.txt"

git ls-files > "$out/.all"
declare -A tracked
while IFS= read -r f; do tracked[$f]=1; done < "$out/.all"

is_lang() { case $1 in src/compiler/*|src/checker2/*|tools/omegese/*) return 0;; esac; return 1; }
# Headers that come from the physics driver client or the NVIDIA open headers.
is_driver() {
    case $1 in nvrm.h|m16_native.h|m16_gpu_wait.h|nvos.h|nv-ioctl*.h|nv_escape.h|uvm_*.h|nv-unix-*.h|class/*|ctrl/*|alloc/*) return 0;; esac
    return 1
}

# includes_of FILE -> one resolved path per line, or "DRIVER:x" / "UNRESOLVED:x"
includes_of() {
    local f=$1 d inc
    d=$(dirname "$f")
    { grep -hoE '^[[:space:]]*#[[:space:]]*include[[:space:]]*"[^"]+"' "$f" 2>/dev/null || true; } |
        sed -E 's/.*"([^"]+)".*/\1/' | while IFS= read -r inc; do
        local c
        for c in "$d/$inc" "src/$inc" "$inc" "tests/$inc"; do
            c=$(realpath -m --relative-to=. "$c")
            if [[ -n ${tracked[$c]:-} ]]; then echo "$c"; continue 2; fi
        done
        if is_driver "$inc"; then echo "DRIVER:$inc"; else echo "UNRESOLVED:$f:$inc"; fi
    done
}

declare -A gpu=()
queue=()
for f in $(grep -E "^src/omega_(blackwell|gpu)_" "$out/.all"); do [[ -n ${stay[$f]:-} ]] && continue; gpu[$f]=1; queue+=("$f"); done

: > "$out/conflicts.txt"; : > "$out/unresolved.txt"; : > "$out/driver-includes.txt"
while ((${#queue[@]})); do
    f=${queue[0]}; queue=("${queue[@]:1}")
    cand=$(includes_of "$f")
    # link rule: header -> its .c
    if [[ $f == *.h ]]; then
        c=${f%.h}.c
        [[ -n ${tracked[$c]:-} ]] && cand+=$'\n'"$c"
    fi
    while IFS= read -r g; do
        [[ -z $g ]] && continue
        case $g in
            DRIVER:*) echo "$f ${g#DRIVER:}" >> "$out/driver-includes.txt"; continue;;
            UNRESOLVED:*) echo "${g#UNRESOLVED:}" >> "$out/unresolved.txt"; continue;;
        esac
        if is_lang "$g"; then echo "$f -> $g" >> "$out/conflicts.txt"; continue; fi
        if [[ -n ${stay[$g]:-} ]]; then echo "$f -> $g (stays in Omega)" >> "$out/conflicts.txt"; continue; fi
        if [[ -z ${gpu[$g]:-} ]]; then gpu[$g]=1; queue+=("$g"); fi
    done <<< "$cand"
done
# core = what CUDA++ owns; shared = the shared base it reached.
printf "%s\n" "${!gpu[@]}" | sort | while read -r p; do if [[ -n ${shared[$p]:-} ]]; then echo "$p"; fi; done > "$out/shared-src.txt"
printf "%s\n" "${!gpu[@]}" | sort | while read -r p; do if [[ -z ${shared[$p]:-} ]]; then echo "$p"; fi; done > "$out/gpu-src.txt"
declare -A core=()
while read -r p; do core[$p]=1; done < "$out/gpu-src.txt"

# Tests and consumers.
: > "$out/gpu-tests.txt"; : > "$out/tests-stay.txt"; : > "$out/consumers.txt"
while IFS= read -r f; do
    [[ $f == *.c || $f == *.h ]] || continue
    [[ -n ${gpu[$f]:-} || -n ${stay[$f]:-} ]] && continue
    incs=$(includes_of "$f")
    hit=0 other=""
    while IFS= read -r g; do
        [[ -z $g ]] && continue
        case $g in DRIVER:*|UNRESOLVED:*) continue;; esac
        if [[ -n ${core[$g]:-} ]]; then hit=1
        elif [[ -n ${shared[$g]:-} ]]; then :
        elif [[ $g == tests/* ]]; then :
        else other+=" $g"; fi
    done <<< "$incs"
    ((hit)) || continue
    if [[ $f == tests/* && -z $other ]]; then echo "$f" >> "$out/gpu-tests.txt"
    elif [[ $f == tests/* ]]; then echo "$f stays: includes$other" >> "$out/tests-stay.txt"
    else echo "$f" >> "$out/consumers.txt"; fi
done < "$out/.all"

# Test-local helpers: a test moves only if every tests/ file it includes is
# itself pure (includes only CUDA++ files, shared base, driver headers or other
# pure test files); the pure helpers come along. Others stay in Omega.
declare -A purity=()
pure() {
    local f=$1 g
    [[ -n ${purity[$f]:-} ]] && { [[ ${purity[$f]} == y ]]; return; }
    purity[$f]=n   # cycle guard
    while IFS= read -r g; do
        [[ -z $g ]] && continue
        case $g in DRIVER:*) continue;; UNRESOLVED:*) return 1;; esac
        [[ -n ${core[$g]:-} || -n ${shared[$g]:-} ]] && continue
        [[ $g == tests/* ]] && pure "$g" && continue
        return 1
    done <<< "$(includes_of "$f")"
    purity[$f]=y
}
: > "$out/.keep"
while IFS= read -r t; do
    if pure "$t"; then
        echo "$t" >> "$out/.keep"
        while IFS= read -r g; do [[ $g == tests/* ]] && echo "$g" >> "$out/.keep"; done <<< "$(includes_of "$t")"
    else
        echo "$t stays: includes an Omega-only test helper" >> "$out/tests-stay.txt"
    fi
done < "$out/gpu-tests.txt"
# a test helper header brings its body (tests/fake_m16_native.h -> .c)
while read -r t; do c=${t%.h}.c; if [[ $t == *.h && -n ${tracked[$c]:-} ]]; then echo "$c"; fi; done < "$out/.keep" >> "$out/.keep2"
cat "$out/.keep" "$out/.keep2" > "$out/gpu-tests.txt"; rm -f "$out/.keep" "$out/.keep2"
sort -u -o "$out/gpu-tests.txt" "$out/gpu-tests.txt"
sort -u -o "$out/driver-includes.txt" "$out/driver-includes.txt"

# Same relative paths on both sides: an Omega branch patch then applies to CUDA++
# unchanged (tools/extract/carry.sh). docs/NAMING.md holds the later rename plan.
grep -v "^#" "$here/extra-paths.txt" | grep . > "$out/extra-paths.txt"
cat "$out/gpu-src.txt" "$out/shared-src.txt" "$out/gpu-tests.txt" "$out/extra-paths.txt" > "$out/keep-paths.txt"
awk '{print $1"\t"$1}' "$out/keep-paths.txt" > "$out/filemap.tsv"
rm -f "$out/.all"

echo "gpu-src $(wc -l < "$out/gpu-src.txt")  shared $(wc -l < "$out/shared-src.txt")  gpu-tests $(wc -l < "$out/gpu-tests.txt")  tests-stay $(wc -l < "$out/tests-stay.txt")  consumers $(wc -l < "$out/consumers.txt")  unresolved $(wc -l < "$out/unresolved.txt")  conflicts $(wc -l < "$out/conflicts.txt")"
if [[ -s $out/conflicts.txt ]]; then echo "CONFLICT: the GPU part includes language or stay files (conflicts.txt)" >&2; exit 2; fi
