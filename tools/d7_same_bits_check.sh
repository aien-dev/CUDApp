#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# tools/d7_same_bits_check.sh LOG... -- D7 (docs/issues/D7-same-bits-chip.md): do separate runs of one job
# print the same output hashes?
#
# Each LOG is the stdout of one run. Every line ending "[<label>] output sha256 <64 hex>" is collected (no label:
# "output", as hello_gpu prints it). For each label:
#   SAME <label> <hash> in N of N runs        the label appears exactly once in every log, same hash everywhere
#   MISMATCH <label> ...                      a log lacks it, has it twice, or two logs disagree (the hashes are listed)
# A log with no hash line at all is a MISMATCH too (a run that printed nothing proves nothing).
# Last line: "d7 same-bits: PASS (<labels> labels x <N> runs)" exit 0, or "d7 same-bits: MISMATCH" exit 1.
# Fewer than two logs: usage error, exit 2. Shell + awk, no Python.
set -eu
if [ "$#" -lt 2 ]; then echo "usage: $0 LOG LOG [LOG...]  (at least two runs)" >&2; exit 2; fi
for f in "$@"; do [ -r "$f" ] || { echo "cannot read $f" >&2; exit 2; }; done
awk -v nlogs="$#" '
FNR == 1 { file++; name[file] = FILENAME; lines[file] = 0 }
NF >= 3 && $(NF - 2) == "output" && $(NF - 1) == "sha256" && $NF ~ /^[0-9a-f]+$/ && length($NF) == 64 {
    lab = NF >= 4 ? $(NF - 3) : "output"   # hello_gpu prints a bare "output sha256 <hex>"
    if (!(lab in seen)) { seen[lab] = 1; order[++nl] = lab }
    cnt[lab, file]++; h[lab, file] = $NF; lines[file]++
}
END {
    bad = 0
    for (f = 1; f <= nlogs; f++) if (!lines[f]) { printf "MISMATCH %s: no output hash line\n", name[f]; bad = 1 }
    for (i = 1; i <= nl; i++) {
        lab = order[i]; ok = 1; ref = ""
        for (f = 1; f <= nlogs; f++) {
            if (cnt[lab, f] != 1) { ok = 0; continue }
            if (ref == "") ref = h[lab, f]; else if (h[lab, f] != ref) ok = 0
        }
        if (ok) { printf "SAME %s %s in %d of %d runs\n", lab, ref, nlogs, nlogs; continue }
        bad = 1; printf "MISMATCH %s\n", lab
        for (f = 1; f <= nlogs; f++) printf "  %s: %s\n", name[f], cnt[lab, f] == 1 ? h[lab, f] : (cnt[lab, f] ? "printed " cnt[lab, f] " times" : "missing")
    }
    if (bad || !nl) { print "d7 same-bits: MISMATCH"; exit 1 }
    printf "d7 same-bits: PASS (%d labels x %d runs)\n", nl, nlogs
}' "$@"
