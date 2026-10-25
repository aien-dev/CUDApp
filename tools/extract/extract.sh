#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# extract.sh: cut the CUDA++ history out of Omega and physics with git's own
# tools (fast-export path-limited, fast-import into fresh repos). No git-filter-repo
# (it is Python; the project has none).
#
#   tools/extract/extract.sh WORK_DIR GRAPH_DIR [OMEGA_REV] [PHYSICS_REV]
#
# WORK_DIR gets: omega.git, physics.git (fresh bare clones from GitHub),
# omega-cut.git, physics-cut.git (the kept history), paths-*.txt (kept paths with
# every earlier name each file had), commit-map-*.tsv (old sha -> new sha) and
# tree-check-*.txt (every kept blob at the tip equals the original's).
# GRAPH_DIR is the output of graph.sh (keep-paths.txt).
set -euo pipefail
export LC_ALL=C

work=${1:?work dir}
graph=${2:?graph dir}
omega_rev=${3:-main}
physics_rev=${4:-main}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work"
work=$(cd "$work" && pwd)

clone() { # url dir
    if [[ -d $2 ]]; then git -C "$2" fetch -q origin '+refs/heads/*:refs/heads/*'; else git clone -q --bare "$1" "$2"; fi
}
clone git@github.com:aien-spark/omega.git "$work/omega.git"
clone git@github.com:aien-spark/physics.git "$work/physics.git"

# expand REPO REV PATHFILE OUT: add every earlier name of every kept file
expand() {
    local repo=$1 rev=$2 in=$3 outf=$4 p
    : > "$outf.new"
    while IFS= read -r p; do
        [[ -z $p || $p == \#* ]] && continue
        echo "$p" >> "$outf.new"
        if git -C "$repo" cat-file -e "$rev:$p" 2>/dev/null && [[ $(git -C "$repo" cat-file -t "$rev:$p") == blob ]]; then
            git -C "$repo" log --follow --name-only --format= "$rev" -- "$p" >> "$outf.new"
        fi
    done < "$in"
    # a --follow hit that still exists at the tip is a separate file (a copy), not a predecessor
    grep . "$outf.new" | sort -u | while IFS= read -r p; do
        if grep -qxF "$p" "$in" || ! git -C "$repo" cat-file -e "$rev:$p" 2>/dev/null; then echo "$p"; fi
    done > "$outf"; rm -f "$outf.new"
}

# cut REPO REV PATHFILE DEST NAME
# fast-export eats the first "--" itself (parse-options), so the paths need a second one.
cut_history() {
    local repo=$1 rev=$2 paths=$3 dest=$4 name=$5
    rm -rf "$dest"; git init -q --bare "$dest"
    # export a named ref: fast-export writes the ref name into the stream
    git -C "$repo" update-ref refs/heads/cudapp-cut "$rev"
    mapfile -t list < "$paths"
    local x
    while IFS= read -r x; do [[ -n $x && $x != \#* ]] && list+=(":(exclude)$x"); done < "$here/$name-exclude.txt"
    git -C "$repo" fast-export --signed-tags=strip --tag-of-filtered-object=drop \
        --export-marks="$work/marks-$name-export" --reencode=yes refs/heads/cudapp-cut -- -- "${list[@]}" |
        git -C "$dest" fast-import --quiet --export-marks="$work/marks-$name-import"
    # fast-export names the ref after REV; normalise to refs/heads/main
    local ref
    ref=$(git -C "$dest" for-each-ref --format='%(refname)' refs/heads | head -1)
    [[ $ref == refs/heads/main ]] || git -C "$dest" update-ref refs/heads/main "$ref"
    join <(sort "$work/marks-$name-export") <(sort "$work/marks-$name-import") |
        awk '{print $2"\t"$3}' > "$work/commit-map-$name.tsv"
}

# tree_check REPO REV DEST PATHFILE NAME: kept tip blobs must equal the originals
tree_check() {
    local repo=$1 rev=$2 dest=$3 paths=$4 name=$5 p a b bad=0
    : > "$work/tree-check-$name.txt"
    while IFS= read -r p; do
        [[ -z $p || $p == \#* ]] && continue
        a=$(git -C "$repo" ls-tree -r "$rev" -- "$p" | awk -v ex="$here/$name-exclude.txt" 'BEGIN {while ((getline l < ex) > 0) if (l !~ /^#/ && l != "") skip[l]=1} !($4 in skip) {print $3"\t"$4}' | sort)
        b=$(git -C "$dest" ls-tree -r main -- "$p" | awk '{print $3"\t"$4}' | sort)
        if [[ $a == "$b" && -n $a ]]; then echo "same $p" >> "$work/tree-check-$name.txt"
        else echo "DIFF $p" >> "$work/tree-check-$name.txt"; bad=1; fi
    done < "$paths"
    # nothing outside the kept paths may be in the cut tip
    git -C "$dest" ls-tree -r --name-only main > "$work/tip-files-$name.txt"
    while IFS= read -r f; do
        grep -qxF "$f" <(git -C "$repo" ls-tree -r --name-only "$rev" -- $(grep -v '^#' "$paths")) ||
            { echo "EXTRA $f" >> "$work/tree-check-$name.txt"; bad=1; }
    done < "$work/tip-files-$name.txt"
    echo "$name tree check: $(grep -c '^same' "$work/tree-check-$name.txt") same, $(grep -c -v '^same' "$work/tree-check-$name.txt" || true) not"
    return $bad
}

omega_tip=$(git -C "$work/omega.git" rev-parse "$omega_rev")
physics_tip=$(git -C "$work/physics.git" rev-parse "$physics_rev")
echo "omega $omega_rev = $omega_tip; physics $physics_rev = $physics_tip"

expand "$work/omega.git" "$omega_tip" "$graph/keep-paths.txt" "$work/paths-omega.txt"
grep -v '^#' "$here/physics-paths.txt" | grep . > "$work/keep-physics.txt"
expand "$work/physics.git" "$physics_tip" "$work/keep-physics.txt" "$work/paths-physics.txt"
# earlier names that are not kept names: listed for the audit (they carry history in)
comm -13 <(sort -u "$graph/keep-paths.txt") "$work/paths-omega.txt" > "$work/lineage-omega.txt" || true
comm -13 <(sort -u "$work/keep-physics.txt") "$work/paths-physics.txt" > "$work/lineage-physics.txt" || true
# the excluded binary must not come back through lineage
grep -vxF -f "$here/physics-exclude.txt" "$work/paths-physics.txt" > "$work/paths-physics.txt.k" || true
mv "$work/paths-physics.txt.k" "$work/paths-physics.txt"

cut_history "$work/omega.git" "$omega_tip" "$work/paths-omega.txt" "$work/omega-cut.git" omega
cut_history "$work/physics.git" "$physics_tip" "$work/paths-physics.txt" "$work/physics-cut.git" physics

rc=0
tree_check "$work/omega.git" "$omega_tip" "$work/omega-cut.git" "$graph/keep-paths.txt" omega || rc=1
tree_check "$work/physics.git" "$physics_tip" "$work/physics-cut.git" "$work/keep-physics.txt" physics || rc=1
echo "omega-cut $(git -C "$work/omega-cut.git" rev-list --count main) commits, physics-cut $(git -C "$work/physics-cut.git" rev-list --count main) commits"
exit $rc
