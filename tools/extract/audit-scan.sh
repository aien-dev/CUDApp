#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# audit-scan.sh: secrets, private data and binaries in EVERY object of a repo's
# history (every blob ever reachable from any ref, plus every commit and tag
# message), not just the tip.
#
#   tools/extract/audit-scan.sh REPO OUT_DIR
#
# Writes OUT_DIR/{secrets,private,binaries,compressed,unscanned,messages}.txt and
# prints counts. Rules: rules-secrets.txt (credential shapes; any hit is a
# finding) and rules-private.txt (IPs, hostnames, emails, home paths; reviewed by
# hand). Binary blobs (any NUL byte) are listed in binaries.txt AND still scanned
# as bytes; gzip/bzip2/xz/zstd/zip blobs are also scanned decompressed (listed in
# compressed.txt). Exit 1 if any secret rule matches, 2 if a compressed blob could
# not be decompressed (unscanned.txt): an unread blob is not a clean blob.
# Run tools/extract/audit-selftest.sh first: it plants one of each secret shape
# (including inside a binary and a gzip blob) and fails if the scanner misses any.
set -euo pipefail
export LC_ALL=C

repo=${1:?repo}
out=${2:?out dir}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
# grep -P takes one pattern: join the rule lines as (?:a)|(?:b)|...
secret_re=$(grep -v "^#" "$here/rules-secrets.txt" | grep . | sed "s/.*/(?:&)/" | paste -sd"|")
private_re=$(grep -v "^#" "$here/rules-private.txt" | grep . | sed "s/.*/(?:&)/" | paste -sd"|")

git -C "$repo" rev-list --objects --all > "$tmp/objs"
cut -d' ' -f1 "$tmp/objs" | git -C "$repo" cat-file --batch-check='%(objectname) %(objecttype) %(objectsize)' > "$tmp/types"
# path for each blob (first name it was seen under)
awk 'NF>=2 {sha=$1; $1=""; sub(/^ /,""); if (!(sha in p)) p[sha]=$0} END {for (s in p) print s"\t"p[s]}' "$tmp/objs" | sort > "$tmp/paths"

: > "$out/secrets.txt"; : > "$out/private.txt"; : > "$out/binaries.txt"
: > "$out/compressed.txt"; : > "$out/unscanned.txt"
# scan FILE LABEL: secret and private rules over the raw bytes (-a: a NUL byte
# must not make grep skip the file or stop matching).
matches() {
    local rc=0
    grep "$@" > "$tmp/matches" || rc=$?
    if (( rc > 1 )); then
        echo "AUDIT INCOMPLETE: grep failed with status $rc" >&2
        exit 2
    fi
}
prefix() {
    local label=$1 line
    while IFS= read -r line; do
        printf '%s:%s\n' "$label" "$line"
    done
}
scan() {
    matches -naP -e "$secret_re" "$1"
    tr -d '\000' < "$tmp/matches" | cut -c1-200 | prefix "$sha $2" >> "$out/secrets.txt"
    matches -noaP -e "$private_re" "$1"
    tr -d '\000' < "$tmp/matches" | prefix "$2" >> "$out/private.txt"
}
mkdir -p "$tmp/b"
nblob=0
while read -r sha type size; do
    [[ $type == blob ]] || continue
    nblob=$((nblob + 1))
    path=$(awk -F'\t' -v s="$sha" '$1==s {print $2; exit}' "$tmp/paths")
    git -C "$repo" cat-file blob "$sha" > "$tmp/b/x"
    # Binary = contains a NUL byte. Counted with tr, not grep -P '\x00': GNU
    # grep 3.11 under LC_ALL=C never matches NUL that way, which left this list
    # empty on every run before 2026-10-27 (CUDApp#89).
    nul_bytes=$(tr -dc '\000' < "$tmp/b/x" | wc -c)
    if [[ $nul_bytes -gt 0 ]]; then
        echo "$sha $size $path" >> "$out/binaries.txt"
    fi
    scan "$tmp/b/x" "$path"
    magic=$(od -An -tx1 -N6 "$tmp/b/x" | tr -d ' \n')
    dec=
    case $magic in
        1f8b*) dec="gzip -dc" ;;
        425a68*) dec="bzip2 -dc" ;;
        fd377a585a00) dec="xz -dc" ;;
        28b52ffd*) dec="zstd -dcq" ;;
        504b0304*) dec="unzip -p" ;;
    esac
    if [[ -n $dec ]]; then
        echo "$sha $size $path ${dec%% *}" >> "$out/compressed.txt"
        # tar members inside come out concatenated; the byte scan covers them.
        if $dec "$tmp/b/x" > "$tmp/b/z" 2>/dev/null; then
            scan "$tmp/b/z" "$path(${dec%% *})"
        else
            echo "$sha $size $path ${dec%% *}" >> "$out/unscanned.txt"
        fi
    fi
done < "$tmp/types"

# commit and tag messages (and identities, for the record)
git -C "$repo" log --all --format='%H%n%B' > "$tmp/msgs"
git -C "$repo" for-each-ref refs/tags --format='%(objectname)%0a%(contents)' >> "$tmp/msgs"
matches -nP -e "$secret_re" "$tmp/msgs"
cut -c1-200 < "$tmp/matches" | prefix MSG >> "$out/secrets.txt"
matches -noP -e "$private_re" "$tmp/msgs"
prefix MSG < "$tmp/matches" >> "$out/private.txt"
cp "$tmp/msgs" "$out/messages.txt"

# private hits, grouped by the matched text (the reviewer reads this list)
cut -d: -f3- "$out/private.txt" | sort | uniq -c | sort -rn > "$out/private-summary.txt"

echo "blobs $nblob  binaries $(wc -l < "$out/binaries.txt")  compressed $(wc -l < "$out/compressed.txt") (unscanned $(wc -l < "$out/unscanned.txt"))  secret-hits $(wc -l < "$out/secrets.txt")  private-hits $(wc -l < "$out/private.txt") ($(wc -l < "$out/private-summary.txt") distinct)"
[[ ! -s $out/secrets.txt ]] || exit 1
[[ ! -s $out/unscanned.txt ]] || exit 2
