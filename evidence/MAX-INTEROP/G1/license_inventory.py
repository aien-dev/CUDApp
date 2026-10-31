#!/usr/bin/env python3
"""File-level licence inventory of a git tree (from git objects, not the
working tree, which may be sparse). Throwaway diagnostic.
usage: license_inventory.py REPO REV OUT_TSV"""
import collections, hashlib, os, re, subprocess, sys

repo, rev, out = sys.argv[1:4]
spdx = re.compile(r"SPDX-License-Identifier:\s*([^\n*#]+)")
tree = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--format=%(objectmode) %(objectname) %(path)", rev],
                      capture_output=True, text=True, check=True).stdout.splitlines()
entries = [l.split(" ", 2) for l in tree]
lic = [p for _, _, p in entries if re.search(r"(^|/)(LICENSE|LICENCE|COPYING|MAX_LICENSE|NOTICE)[^/]*$", p)]
lic_dirs = collections.defaultdict(list)
for p in lic:
    lic_dirs[os.path.dirname(p)].append(p)

def nearest(p):
    d = os.path.dirname(p)
    while True:
        if d in lic_dirs:
            return ",".join(lic_dirs[d])
        if not d:
            return ""
        d = os.path.dirname(d)

def classify(b):
    if b.startswith(b"version https://git-lfs"):
        return "LFS-POINTER", ""
    t = b[:4096].decode("utf-8", "replace")
    m = spdx.search(t)
    if m:
        return "SPDX", m.group(1).strip().rstrip("-*/ ").strip()
    if "Apache License" in t and "LLVM" in t:
        return "HEADER", "Apache-2.0 WITH LLVM-exception"
    if re.search(r"MAX (Community )?License|MAX Platform", t):
        return "HEADER", "MAX License"
    if "Apache License" in t:
        return "HEADER", "Apache-2.0"
    if "MIT License" in t or "Permission is hereby granted, free of charge" in t:
        return "HEADER", "MIT"
    if b"\0" in b[:4096]:
        return "NONE", "binary"
    return "NONE", "no-header"

cat = subprocess.Popen(["git", "-C", repo, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
counts = collections.Counter()
with open(out, "w") as fo:
    fo.write("path\tmode\tblob_sha256\tkind\tlicence\tgoverning_licence_file\n")
    for mode, obj, path in entries:
        if mode == "160000":  # submodule
            fo.write(f"{path}\t{mode}\t\tSUBMODULE\t\t\n"); counts[("SUBMODULE", "", "")] += 1; continue
        cat.stdin.write((obj + "\n").encode()); cat.stdin.flush()
        hdr = cat.stdout.readline().split()
        n = int(hdr[2]); b = cat.stdout.read(n); cat.stdout.read(1)
        kind, l = classify(b)
        gov = nearest(path) if kind in ("NONE", "LFS-POINTER") else ""
        fo.write(f"{path}\t{mode}\t{hashlib.sha256(b).hexdigest()}\t{kind}\t{l}\t{gov}\n")
        counts[(kind, l, gov)] += 1
cat.stdin.close(); cat.wait()
print(f"repo={repo} rev={rev} files={len(entries)} licence_files={lic}")
for (k, l, g), c in counts.most_common():
    print(f"{c}\t{k}\t{l}\t{g}")
