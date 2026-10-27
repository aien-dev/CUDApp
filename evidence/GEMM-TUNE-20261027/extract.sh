#!/bin/bash
# Bookkeeping only: collect kept samples per cell as a Wolfram list (no arithmetic here; medians/ratios are in Wolfram).
# Usage: extract.sh <run.log>...  -> stdout: {{type,"MxNxK",lay,armA_main,armB_main,armA_mainpack,armB_mainpack},...} ; stderr: discard counts
# A sample is kept when rc=OK, wrong=0, if_before=0 and if_after=0 (AB-PLAN discard rule); others are counted as discarded.
awk '
/^sample / {
  delete f; for (i = 2; i <= NF; i++) { split($i, kv, "="); f[kv[1]] = kv[2] }
  cell = f["type"] "|" f["M"] "x" f["N"] "x" f["K"] "|" f["lay"]
  if (!(cell in seen)) { seen[cell] = 1; order[++n] = cell }
  if (f["rc"] != "OK" || f["wrong"] != "0") { bad[cell]++; badtot++; next }
  if (f["if_before"] != "0" || f["if_after"] != "0") { disc[cell]++; disctot++; next }
  a = f["arm"]; mp = f["main_ns"] "+" f["pack_ns"]
  m[cell, a] = m[cell, a] (m[cell, a] == "" ? "" : ",") f["main_ns"]
  p[cell, a] = p[cell, a] (p[cell, a] == "" ? "" : ",") mp
  kept[cell, a]++
}
END {
  printf "{"
  for (i = 1; i <= n; i++) {
    c = order[i]; split(c, q, "|")
    printf "%s{\"%s\",\"%s\",%s,{%s},{%s},{%s},{%s}}", (i > 1 ? "," : ""), q[1], q[2], q[3], m[c, "A"], m[c, "B"], p[c, "A"], p[c, "B"]
    printf "%s %s keptA=%d keptB=%d discarded=%d failed=%d\n", q[1], q[2], kept[c, "A"], kept[c, "B"], disc[c] + 0, bad[c] + 0 > "/dev/stderr"
  }
  printf "}\n"
  printf "TOTAL discarded=%d failed=%d cells=%d\n", disctot + 0, badtot + 0, n > "/dev/stderr"
}' "$@"
