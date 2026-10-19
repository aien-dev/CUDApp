#!/usr/bin/env bash
# fetch-physics.sh [dest]
# Gets the "physics" repository (the GPU driver interface Omega builds on) at the exact commit
# pinned in physics.lock, into dest (default: build/physics inside this checkout), and checks it.
# Prints the folder on the last line of stdout. Safe to run again: if dest is already at the pinned
# commit it does nothing.  PHYSICS_URL overrides the source (default: the public repository).
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
dest=${1:-${OUT_DIR:-$here/build}/physics}
url=${PHYSICS_URL:-https://github.com/aien-dev/physics-mv}
[ -f "$here/physics.lock" ] || { echo "fetch-physics: physics.lock not found in $here" >&2; exit 1; }
want=$(tr -d '[:space:]' < "$here/physics.lock")
case $want in *[!0-9a-f]*|"") echo "fetch-physics: physics.lock is not a commit id: $want" >&2; exit 1 ;; esac
[ ${#want} -eq 40 ] || { echo "fetch-physics: physics.lock must hold a full 40-character commit id" >&2; exit 1; }

if [ -d "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" = "$want" ]; then
	echo "fetch-physics: $dest already at $want" >&2
	echo "$dest"; exit 0
fi
[ ! -e "$dest" ] || [ -d "$dest/.git" ] || { echo "fetch-physics: $dest exists and is not a git checkout; remove it or pass another folder" >&2; exit 1; }

mkdir -p "$dest"
[ -d "$dest/.git" ] || git -C "$dest" init -q
git -C "$dest" remote remove origin 2>/dev/null || true
git -C "$dest" remote add origin "$url"
# Fast path: fetch only the pinned commit. Fall back to a full fetch if the server refuses it by id.
if ! git -C "$dest" fetch -q --depth 1 origin "$want" 2>/dev/null; then
	echo "fetch-physics: fetching the pinned commit directly failed, fetching all branches" >&2
	git -C "$dest" fetch -q origin '+refs/heads/*:refs/remotes/origin/*'
fi
git -C "$dest" checkout -q --detach "$want"
got=$(git -C "$dest" rev-parse HEAD)
if [ "$got" != "$want" ]; then
	echo "fetch-physics: checked out $got but physics.lock pins $want" >&2; exit 1
fi
echo "fetch-physics: $dest at $got (matches physics.lock)" >&2
echo "$dest"
