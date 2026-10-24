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

# carry.sh: bring the GPU half of an Omega branch over to CUDA++, with its whole history
# (merges and their conflict resolutions included).
#
#   tools/extract/carry.sh OMEGA_DIR OMEGA_REF NEW_BRANCH [EXTRA_PATHSPEC...]
#
# How: the branch is cut exactly like the original extraction (git fast-export limited to the
# CUDA++ paths, the same exclusions, into a scratch repository). The cut is deterministic, so
# every commit the branch shares with the Omega history CUDA++ was cut from comes out with the
# same id as here; the new commits sit on top. NEW_BRANCH = current HEAD merged with that cut,
# a normal three-way merge (it only has to reconcile what CUDA++ changed since, e.g. SPDX lines).
#
# Paths: every path in filemap.tsv, plus every src/omega_gpu_* and src/omega_blackwell_* file
# the branch adds (new GPU files), minus the Omega bridges in stay.txt, plus EXTRA_PATHSPEC.
# The Omega half of the same branch stays in Omega: rebase it on Omega's opencuda/consume and
# point cudapp.lock at NEW_BRANCH's tip.
set -euo pipefail
export LC_ALL=C

omega=${1:?omega dir}; ref=${2:?omega ref}; new=${3:?new branch}; shift 3
here=$(cd "$(dirname "$0")" && pwd)
top=$(git -C "$here" rev-parse --show-toplevel)
tip=$(git -C "$omega" rev-parse "$ref")
scratch=$(mktemp -d); trap 'rm -rf "$scratch"' EXIT

mapfile -t paths < <(cut -f1 "$here/filemap.tsv")
mapfile -t added < <(git -C "$omega" ls-tree -r --name-only "$tip" -- src | grep -E '^src/omega_(gpu|blackwell)_' |
    grep -vxF -f <(grep -v '^#' "$here/stay.txt" | grep .) | grep -vxF -f <(printf '%s\n' "${paths[@]}") || true)
paths+=("${added[@]}" "$@")
while IFS= read -r x; do [[ -n $x && $x != \#* ]] && paths+=(":(exclude)$x"); done < "$here/omega-exclude.txt"
((${#added[@]})) && echo "new GPU files on $ref: ${added[*]}"

git -C "$omega" update-ref refs/heads/cudapp-carry "$tip"
git init -q --bare "$scratch/cut.git"
git -C "$omega" fast-export --signed-tags=strip --tag-of-filtered-object=drop --reencode=yes \
    refs/heads/cudapp-carry -- -- "${paths[@]}" | git -C "$scratch/cut.git" fast-import --quiet
git -C "$omega" update-ref -d refs/heads/cudapp-carry
git -C "$top" fetch -q "$scratch/cut.git" "refs/heads/cudapp-carry:refs/carry/$new"
shared=$(git -C "$top" merge-base HEAD "refs/carry/$new" || true)
[[ -n $shared ]] || { echo "carry: the cut shares no commit with HEAD (was the branch cut with other paths?)" >&2; exit 1; }
echo "carry $ref ($(git -C "$omega" rev-parse --short "$tip")): $(git -C "$top" rev-list --count "$shared..refs/carry/$new") new commit(s) on top of $(git -C "$top" rev-parse --short "$shared")"

git -C "$top" checkout -q -b "$new"
if ! git -C "$top" merge -q --no-ff --no-edit "refs/carry/$new" \
    -m "Carry Omega $ref ($tip) into CUDA++ (tools/extract/carry.sh)"; then
    echo "CARRY CONFLICT on $new: resolve, then git commit (or git merge --abort)" >&2
    exit 1
fi
echo "carried onto $new = $(git -C "$top" rev-parse --short HEAD)"
