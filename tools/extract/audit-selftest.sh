#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# audit-selftest.sh: negative control for audit-scan.sh. Plants one fake secret
# per rule in a throwaway repo (one in a deleted file, one in a commit message,
# one in a blob with NUL bytes, one inside a gzip blob) and fails unless every
# planted one is found and the NUL blob is listed as binary.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
git init -q "$t/r"; cd "$t/r"
git config user.email t@example.invalid; git config user.name t
k=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij0123456789
# The fakes are put together at run time so no secret-shaped string sits in this file (outside scanners stay quiet).
b=-----BEGIN; jh=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9; ak=api_key
{
  echo "$b OPENSSH PRIVATE KEY-----"
  echo "AKIA${k:0:16}"
  echo "ghp_${k:0:36}"
  echo "github_pat_${k}${k:0:20}"
  echo "xo""xb-1234567890-abcdef"
  echo "sk-ant-api03-${k:0:30}"
  echo "sk-${k:0:40}"
  echo "AIza${k:0:35}"
  echo "tskey-auth-${k:0:24}"
  echo "glpat-${k:0:24}"
  echo "xai-${k}"
  echo "$jh.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop"
  echo "ssh-ed25519 AAAA${k}${k}"
  echo "$ak = \"abcdefghijklmnop1234\""
} > leak.txt
planted=$(wc -l < leak.txt)
git add leak.txt; git commit -qm "add"
git rm -q leak.txt; git commit -qm "remove; also npm_${k:0:36}"
planted=$((planted + 1))
# A binary blob (NUL bytes) and a compressed one: both must still be read.
printf 'bin\000\001\002\nAKIA%s\n\000tail\n' "${k:0:16}" > blob.bin
printf 'ghp_%s\n' "${k:4:36}" | gzip -9n > leak.gz
git add blob.bin leak.gz; git commit -qm "binary and gzip"
planted=$((planted + 2))
binsha=$(git rev-parse HEAD:blob.bin)
if "$here/audit-scan.sh" "$t/r" "$t/out" > "$t/log"; then echo "SELFTEST FAIL: scanner reported clean"; exit 1; fi
found=$(wc -l < "$t/out/secrets.txt")
if (( found < planted )); then echo "SELFTEST FAIL: planted $planted, found $found"; cat "$t/out/secrets.txt"; exit 1; fi
if ! grep -q "^$binsha " "$t/out/binaries.txt"; then echo "SELFTEST FAIL: NUL-byte blob not listed in binaries.txt"; exit 1; fi
if ! grep -q "leak.gz(gzip):" "$t/out/secrets.txt"; then echo "SELFTEST FAIL: secret inside gzip blob not found"; exit 1; fi
echo "SELFTEST PASS: planted $planted (one in a deleted file's history, one in a commit message, one in a NUL-byte blob, one gzipped), found $found; binary listed"
