#!/bin/sh
# run_cli.sh DAGSIM_BIN [OUT_DIR]
# Reproducible CLI example for omega-mv#583: pinned fixtures in, JSON and a
# Pareto table out, byte-identical on a second run, no GPU touched.
set -u
BIN=${1:?dagsim binary}
OUT=${2:-/tmp/dagsim-cli-$$}
F=tests/dagsim/fixtures
ASOF=2026-10-20
mkdir -p "$OUT"
fail=0
say() { echo "run_cli: $*"; }
check() { if [ "$1" -ne 0 ]; then say "FAIL $2"; fail=1; else say "ok   $2"; fi; }

"$BIN" eval $F/m1_spark_subset.oev --asof $ASOF > "$OUT/subset.json"; rc=$?
check $rc "eval exits 0 on the pinned M1 subset"
grep -q '"status": "OK"' "$OUT/subset.json"; check $? "status OK"
grep -q '"makespan_ns": {"nominal": 151756700,' "$OUT/subset.json"; check $? "serial makespan 20000 + 40736700 + 111000000 ns"
grep -q '"makespan_ge_bounds": true' "$OUT/subset.json"; check $? "makespan is above its bounds"
grep -q '"status": "UNMEASURED"' "$OUT/subset.json"; check $? "energy UNMEASURED without an instrument"
grep -q '"gpu_executed": false' "$OUT/subset.json"; check $? "offline: no GPU execution"
grep -q '"provenance": \["measured", "specified"\]' "$OUT/subset.json"; check $? "provenance list names measured and specified only"
"$BIN" eval $F/m1_spark_subset.oev --asof $ASOF > "$OUT/subset2.json"
cmp -s "$OUT/subset.json" "$OUT/subset2.json"; check $? "second run byte-identical"

"$BIN" eval $F/needs_arch244_trace.oev --asof $ASOF > "$OUT/unknown.json"; rc=$?
[ $rc -eq 1 ] ; check $? "unknown-trace fixture exits 1"
grep -q '"status": "REFUSED_UNKNOWN"' "$OUT/unknown.json"; check $? "unknown stays unknown (REFUSED_UNKNOWN)"
grep -q 'makespan' "$OUT/unknown.json"; [ $? -ne 0 ]; check $? "no makespan printed on refusal"

"$BIN" eval $F/m1_spark_subset.oev --asof 2026-12-01 > "$OUT/stale.json"
grep -q '"status": "REFUSED_STALE"' "$OUT/stale.json"; check $? "30-day calibration refused at 43 days"

"$BIN" eval $F/m1_synthetic_power.oev --asof $ASOF > "$OUT/syn.json"
grep -q '"status": "SYNTHETIC_SENSITIVITY_ONLY"' "$OUT/syn.json"; check $? "synthetic power is labelled, not ranked"
grep -q '"ranking_valid": false' "$OUT/syn.json"; check $? "energy ranking invalid on synthetic power"

"$BIN" eval $F/m1_spark_overlap.oev --asof $ASOF > "$OUT/overlap.json"
grep -q '"status": "OK"' "$OUT/overlap.json"; check $? "overlap fixture evaluates"
# two halves sharing a 65.4 GB/s pipe: 2 GB of traffic is 30.58 ms, each half's own busy
# time is 55.5 ms, so busy time binds and both halves end together at 40.7567 + 55.5 ms.
grep -q '"makespan_ns": {"nominal": 96256700,' "$OUT/overlap.json"; check $? "split copy: busy time binds, 40756700 + 55500000 ns"

"$BIN" pareto $F/m1_spark_subset.oev $F/m1_spark_overlap.oev $F/m1_synthetic_power.oev $F/needs_arch244_trace.oev --asof $ASOF > "$OUT/pareto.txt"; rc=$?
check $rc "pareto exits 0"
grep -q 'needs_arch244_trace.oev *REFUSED_UNKNOWN' "$OUT/pareto.txt"; check $? "refused candidate shown as refused in the table"
grep -q 'm1_spark_overlap.oev *OK .* yes$' "$OUT/pareto.txt"; check $? "faster candidate is on the front"
grep -q 'm1_spark_subset.oev *OK .* no$' "$OUT/pareto.txt"; check $? "dominated candidate is off the front"
grep -q 'Energy not measured on every row' "$OUT/pareto.txt"; check $? "pareto says energy is not ranked"

"$BIN" diff $F/mw_baseline.oev $F/mw_reuse.oev --asof $ASOF > "$OUT/diff_reuse.json"; rc=$?
check $rc "reuse diff OK"
grep -q '"reused": 200' "$OUT/diff_reuse.json"; check $? "200 ns of search moved to the reused ledger"
"$BIN" diff $F/mw_baseline.oev $F/mw_bad_elimination.oev --asof $ASOF > "$OUT/diff_bad.json"; rc=$?
[ $rc -eq 1 ]; check $? "unproven elimination exits 1"
grep -q '"status": "REFUSED_REMOVED_REQUIRED"' "$OUT/diff_bad.json"; check $? "unproven elimination refused"
"$BIN" diff $F/mw_baseline.oev $F/mw_baseline.oev --asof $ASOF > "$OUT/diff_same.json"
grep -q '"kept": 4, "added": 0, "removed": 0' "$OUT/diff_same.json"; check $? "no-change fixture adds no nodes"

# omega-mv#705: inverted lifetime must not come back as an OK capacity certificate.
"$BIN" eval $F/inverted_masks_overflow.oev --asof 2026-10-24 > "$OUT/inverted.json"; rc=$?
[ $rc -eq 1 ]; check $? "inverted lifetime exits 1"
grep -q '"status": "REFUSED_LIFETIME"' "$OUT/inverted.json"; check $? "inverted lifetime status"
grep -q 'first=C' "$OUT/inverted.json"; check $? "inverted reason names first=C"
grep -q 'last=A' "$OUT/inverted.json"; check $? "inverted reason names last=A"
grep -q 'makespan' "$OUT/inverted.json"; [ $? -ne 0 ]; check $? "no makespan on inverted refusal"
"$BIN" eval $F/inverted_lone.oev --asof 2026-10-24 > "$OUT/inverted_lone.json"; rc=$?
[ $rc -eq 1 ]; check $? "lone inverted lifetime exits 1"
grep -q '"status": "REFUSED_LIFETIME"' "$OUT/inverted_lone.json"; check $? "lone inverted is not peak 0"
"$BIN" eval $F/capacity_two_80.oev --asof 2026-10-24 > "$OUT/cap80.json"; rc=$?
[ $rc -eq 1 ]; check $? "two overlapping 80 B buffers exit 1"
grep -q '"status": "REFUSED_CAPACITY"' "$OUT/cap80.json"; check $? "overlap is REFUSED_CAPACITY"
"$BIN" eval $F/valid_disjoint_80.oev --asof 2026-10-24 > "$OUT/disjoint80.json"; rc=$?
check $rc "disjoint 80 B pair exits 0"
grep -q '"status": "OK"' "$OUT/disjoint80.json"; check $? "disjoint pair OK"
grep -q '"bytes": 80' "$OUT/disjoint80.json"; check $? "disjoint peak 80"
"$BIN" eval $F/valid_tie_80.oev --asof 2026-10-24 > "$OUT/tie80.json"; rc=$?
check $rc "completion/allocation tie exits 0"
grep -q '"status": "OK"' "$OUT/tie80.json"; check $? "tie OK"
grep -q '"bytes": 80' "$OUT/tie80.json"; check $? "tie peak 80"
"$BIN" pareto $F/inverted_masks_overflow.oev $F/valid_disjoint_80.oev --asof 2026-10-24 > "$OUT/pareto_life.txt"; rc=$?
check $rc "pareto with a refused lifetime exits 0"
grep -q 'inverted_masks_overflow.oev *REFUSED_LIFETIME' "$OUT/pareto_life.txt"; check $? "inverted row is refused, not ranked OK"
grep -q 'valid_disjoint_80.oev *OK' "$OUT/pareto_life.txt"; check $? "valid row still ranked"

say "outputs in $OUT"
exit $fail
