#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# carry.sh: bring the GPU half of an Omega branch over to CUDA++, commits and
# authors intact. Both repos use the same relative paths for the moved files
# (tools/extract/filemap.tsv), so Omega's patches apply here unchanged; git am -3
# falls back to a three-way merge where CUDA++ has moved on (the SPDX lines).
#
#   tools/extract/carry.sh OMEGA_DIR OMEGA_REF NEW_BRANCH [PATHSPEC...]
#
# OMEGA_DIR   an Omega clone (its origin/main is the base the branch forked from)
# OMEGA_REF   the branch to carry, e.g. origin/runtime/r2-hello
# NEW_BRANCH  created here from the current HEAD
# PATHSPEC    which paths to carry; default: every path in filemap.tsv. Pass
#             extra paths for files the branch adds (e.g. examples/ docs/runtime/).
#
# The Omega half of the same branch (everything else) stays in Omega: rebase it
# on Omega's opencuda/consume and point cudapp.lock at NEW_BRANCH's tip.
set -euo pipefail
export LC_ALL=C

omega=${1:?omega dir}; ref=${2:?omega ref}; new=${3:?new branch}; shift 3
here=$(cd "$(dirname "$0")" && pwd)
top=$(git -C "$here" rev-parse --show-toplevel)

paths=("$@")
if ((${#paths[@]} == 0)); then mapfile -t paths < <(cut -f1 "$here/filemap.tsv"); fi

base=$(git -C "$omega" merge-base origin/main "$ref")
n=$(git -C "$omega" rev-list --count "$base..$ref" -- "${paths[@]}")
echo "carry $ref: $n commit(s) touch CUDA++ paths since $(git -C "$omega" rev-parse --short "$base")"
((n > 0)) || { echo "nothing to carry"; exit 0; }

patch=$(mktemp); trap 'rm -f "$patch"' EXIT
git -C "$omega" format-patch -k --stdout "$base..$ref" -- "${paths[@]}" > "$patch"
git -C "$top" checkout -q -b "$new"
if ! git -C "$top" am -3 -k --committer-date-is-author-date "$patch"; then
    echo "CARRY CONFLICT on $new: resolve, then git am --continue (or git am --abort)" >&2
    exit 1
fi
echo "carried $n commit(s) onto $new = $(git -C "$top" rev-parse --short HEAD)"
