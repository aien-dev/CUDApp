#!/usr/bin/env python3
# DIAGNOSTIC (DecodeKernels A/B, 2026-10-27; evidence tooling, not product code). Usage: tag.py RUN_LOG INFLIGHT_LOG OUT_PREFIX
# Tags every AB (router) / GAB (expert grid) JSON sample with the 18091 in_flight polls around it. A sample is CLEAN when
# a poll at or before t0 and a poll at or after t1 exist, every poll from the last one <= t0 to the first one >= t1 reads
# in_flight == 0, and no gap between consecutive polls in that span exceeds 200 ms. Writes OUT_PREFIX.tagged.jsonl (all
# samples + clean flag) and OUT_PREFIX.clean.tsv (key, value_ns) for the Wolfram medians; prints counts per key.
# It computes no statistic: medians and ratios are Wolfram's (AB-PLAN.md).
import bisect, collections, json, sys

run, infl, out = sys.argv[1:4]
polls = []
for l in open(infl):
    p = l.split()
    if len(p) == 2:
        polls.append((int(p[0]), int(p[1])))
polls.sort()
ts = [p[0] for p in polls]
GAP = 200_000_000


def clean(t0, t1):
    i = bisect.bisect_right(ts, t0) - 1
    j = bisect.bisect_left(ts, t1)
    if i < 0 or j >= len(ts):
        return False, "no bracketing poll"
    span = polls[i:j + 1]
    if any(f != 0 for _, f in span):
        return False, "in_flight>0"
    if any(b[0] - a[0] > GAP for a, b in zip(span, span[1:])):
        return False, "poll gap>200ms"
    return True, ""


n = collections.Counter(); c = collections.Counter(); why = collections.Counter(); wrong = 0
with open(out + ".tagged.jsonl", "w") as tj, open(out + ".clean.tsv", "w") as tv:
    for l in open(run):
        if l.startswith("AB {"):
            j = json.loads(l[3:]); key = "router arm%d ctas%d" % (j["arm"], j["ctas"]); val = j["chip_ns"]
        elif l.startswith("GAB {"):
            j = json.loads(l[4:]); key = "shape%d arm%d %d:%d" % (j["shape"], j["arm"], j["ctas"], j["warps"]); val = j["mean_ns"]
        else:
            continue
        ok, r = clean(j["t0_ns"], j["t1_ns"])
        if not j["exact"]:
            wrong += 1
        j["clean"] = ok; j["discard"] = r; j["key"] = key
        tj.write(json.dumps(j) + "\n")
        n[key] += 1
        if ok:
            c[key] += 1; tv.write("%s\t%d\n" % (key, val))
        else:
            why[r] += 1
tot = sum(n.values()); cl = sum(c.values())
print("samples %d clean %d discarded %d %s; not exact %d" % (tot, cl, tot - cl, dict(why), wrong))
for k in sorted(n):
    print("%-28s total %4d clean %4d" % (k, n[k], c[k]))
