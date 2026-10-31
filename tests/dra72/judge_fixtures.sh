#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Runs the DRA-72 judge on the synthetic fixtures and checks each exit status (0 PASS, 1 FAIL, 4 MALFORMED or
# INCONCLUSIVE). Sequencing only: every verdict is the judge's.
J=${1:?usage: judge_fixtures.sh JUDGE}
F=$(dirname "$0")
bad=0
run() {
	want=$1; shift
	"$J" "$@"; got=$?
	if [ "$got" -ne "$want" ]; then echo "FIXTURE MISMATCH: want $want got $got: $*"; bad=1; fi
}
run 0 g2 "$F/cpu-pass.txt"
run 1 g2 "$F/cpu-fail.txt"
run 4 g2 "$F/cpu-malformed.txt"
run 0 g2 "$F/cpu-pass.txt" "$F/load-with-pass.txt"
run 4 g2 "$F/cpu-pass.txt" "$F/load-bad.txt"
run 0 g3 "$F/load-alone.txt" "$F/load-with-pass.txt" "$F/cpu-pass.txt"
run 1 g3 "$F/load-alone.txt" "$F/load-with-fail.txt" "$F/cpu-pass.txt"
run 4 g3 "$F/load-alone.txt" "$F/load-bad.txt" "$F/cpu-pass.txt"
run 4 g3 "$F/load-alone.txt" "$F/load-with-pass.txt" "$F/cpu-malformed.txt"
run 0 g3s "$F/gate-alone.txt" "$F/gate-with-pass.txt" "$F/cpu-pass.txt"
run 1 g3s "$F/gate-alone.txt" "$F/gate-with-fail.txt" "$F/cpu-pass.txt"
run 4 g3s "$F/gate-alone.txt" "$F/gate-with-gatefail.txt" "$F/cpu-pass.txt"
run 0 g4 "$F/hand-pass.txt"
run 1 g4 "$F/hand-fail.txt"
run 1 g4 "$F/hand-neg.txt"
run 4 g4 "$F/hand-malformed.txt"
run 4 g4 "$F/hand-sim.txt"
run 2 g9 "$F/hand-pass.txt"
run 5 g4 "$F/does-not-exist.txt"
[ "$bad" -eq 0 ] && echo "DRA72 JUDGE FIXTURES PASS"
exit "$bad"
