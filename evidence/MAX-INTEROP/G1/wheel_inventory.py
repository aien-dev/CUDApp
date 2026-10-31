#!/usr/bin/env python3
"""File-level licence inventory of installed MAX/Mojo wheels, from RECORD.
Throwaway diagnostic. usage: wheel_inventory.py SITE_PACKAGES OUT_TSV"""
import collections, csv, glob, os, re, sys

site, out = sys.argv[1:3]
spdx = re.compile(r"SPDX-License-Identifier:\s*([^\n*#]+)")
counts = collections.Counter()
with open(out, "w") as fo:
    fo.write("dist\tversion\tdeclared_licence\tpath\trecord_hash\tsize\tfile_header_licence\n")
    for di in sorted(glob.glob(os.path.join(site, "*.dist-info"))):
        base = os.path.basename(di)
        if not re.match(r"(max|mojo|modular)", base):
            continue
        meta = open(os.path.join(di, "METADATA"), errors="replace").read()
        name = re.search(r"^Name: (.+)$", meta, re.M).group(1)
        ver = re.search(r"^Version: (.+)$", meta, re.M).group(1)
        dec = (re.search(r"^License(?:-Expression)?: (.+)$", meta, re.M) or [None, ""])[1]
        for row in csv.reader(open(os.path.join(di, "RECORD"))):
            path, h, size = (row + ["", ""])[:3]
            hdr = ""
            full = os.path.join(site, path)
            if os.path.isfile(full) and not path.endswith((".so", ".mojoc", ".mojopkg", ".pyc")) and "/bin/" not in path:
                try:
                    t = open(full, "rb").read(4096).decode("utf-8", "replace")
                    m = spdx.search(t)
                    if m:
                        hdr = m.group(1).strip().rstrip("-*/ ").strip()
                    elif "Apache License" in t and "LLVM" in t:
                        hdr = "Apache-2.0 WITH LLVM-exception"
                    elif "Apache License" in t:
                        hdr = "Apache-2.0"
                except OSError:
                    pass
            ext = os.path.splitext(path)[1] or "(none)"
            fo.write(f"{name}\t{ver}\t{dec}\t{path}\t{h}\t{size}\t{hdr}\n")
            counts[(name, ver, dec, hdr or "(no header)", ext if not hdr else "")] += 1
for k, c in sorted(counts.items()):
    if c >= 1:
        print(c, *k, sep="\t")
