#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# audit-scan.sh: secrets, private data and binaries in EVERY object of a repo's
# history (every blob ever reachable from any ref, plus every commit and tag
# message), not just the tip.
#
#   tools/extract/audit-scan.sh REPO OUT_DIR
#
# Writes OUT_DIR/{secrets,private,binaries,messages}.txt and prints counts.
# Rules: rules-secrets.txt (credential shapes; any hit is a finding) and
# rules-private.txt (IPs, hostnames, emails, home paths; reviewed by hand).
# Exit 1 if any secret rule matches. Run tools/extract/audit-selftest.sh first:
# it plants one of each secret shape and fails if the scanner misses any.
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
mkdir -p "$tmp/b"
nblob=0
while read -r sha type size; do
    [[ $type == blob ]] || continue
    nblob=$((nblob + 1))
    path=$(awk -F'\t' -v s="$sha" '$1==s {print $2; exit}' "$tmp/paths")
    git -C "$repo" cat-file blob "$sha" > "$tmp/b/x"
    if grep -qP '\x00' "$tmp/b/x"; then
        echo "$sha $size $path" >> "$out/binaries.txt"
        continue
    fi
    grep -nP -e "$secret_re" "$tmp/b/x" 2>/dev/null | cut -c1-200 | sed "s|^|$sha $path:|" >> "$out/secrets.txt" || true
    grep -noP -e "$private_re" "$tmp/b/x" 2>/dev/null | sed "s|^|$path:|" >> "$out/private.txt" || true
done < "$tmp/types"

# commit and tag messages (and identities, for the record)
git -C "$repo" log --all --format='%H%n%B' > "$tmp/msgs"
git -C "$repo" for-each-ref refs/tags --format='%(objectname)%0a%(contents)' >> "$tmp/msgs"
grep -nP -e "$secret_re" "$tmp/msgs" | cut -c1-200 | sed 's|^|MSG:|' >> "$out/secrets.txt" || true
grep -noP -e "$private_re" "$tmp/msgs" | sed 's|^|MSG:|' >> "$out/private.txt" || true
cp "$tmp/msgs" "$out/messages.txt"

# private hits, grouped by the matched text (the reviewer reads this list)
cut -d: -f3- "$out/private.txt" | sort | uniq -c | sort -rn > "$out/private-summary.txt"

echo "blobs $nblob  binaries $(wc -l < "$out/binaries.txt")  secret-hits $(wc -l < "$out/secrets.txt")  private-hits $(wc -l < "$out/private.txt") ($(wc -l < "$out/private-summary.txt") distinct)"
[[ ! -s $out/secrets.txt ]]
