#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Host test for developer bar rows D3, D4, D7, D10 and D12 (docs/PAIN_POINTS.md), on hello_gpu --sim.
#   usage: tests/dx/hello_sim_test.sh build/hello_gpu
# No GPU: --sim runs the same kernel program on the host warp simulator (tests/bw_warp_sim.h).
#   D4  --sim checks all 1048576 answers bit for bit and passes
#   D3  --sim --bug=past-end (the host passes n+1 on the last launch, a classic off-by-one) fails
#       AT that launch and names the kernel, launch, instruction, block, thread, address, the buffer
#       and how far past its end, and the next command
#   D7  two runs print the same output hash
#   D10 two runs print the same machine-code hash, and the build time is printed
#   D12 the work line gives the exact bytes moved and the floor time at the measured ceiling
set -u
H=${1:?usage: hello_sim_test.sh path/to/hello_gpu}
pass=0; fail=0
want() { # name, text, pattern...
	n=$1; t=$2; shift 2; ok=1
	for p in "$@"; do printf '%s\n' "$t" | grep -qF -- "$p" || { echo "  missing: $p"; ok=0; }; done
	if [ $ok -eq 1 ]; then pass=$((pass+1)); echo "PASS $n"; else fail=$((fail+1)); echo "FAIL $n"; printf '%s\n' "$t" | sed 's/^/  | /'; fi
}
exitis() { # name, got, want
	if [ "$2" -eq "$3" ]; then pass=$((pass+1)); echo "PASS $1 (exit $2)"; else fail=$((fail+1)); echo "FAIL $1: exit $2, want $3"; fi
}

out1=$("$H" --sim 2>&1); rc1=$?
exitis sim-pass "$rc1" 0
want sim-checks "$out1" \
	'on the host simulator (no GPU)' \
	'checked 1048576 of 1048576 elements: 0 wrong, 0 never written' \
	'PASS'
want work-line "$out1" \
	'work: 12582912 bytes moved (8388608 read, 4194304 written), 1048576 floating-point operations' \
	'at the GB10'"'"'s measured 195.1 GB/s read ceiling this needs at least 64.5 us'
want hashes-printed "$out1" 'output sha256 ' 'machine code sha256 ' 'kernel built in '
out2=$("$H" --sim 2>&1)
h1=$(printf '%s\n' "$out1" | grep -E 'output sha256|machine code sha256'); h2=$(printf '%s\n' "$out2" | grep -E 'output sha256|machine code sha256')
if [ -n "$h1" ] && [ "$h1" = "$h2" ]; then pass=$((pass+1)); echo "PASS same-bits-twice"; else fail=$((fail+1)); echo "FAIL same-bits-twice"; echo "  run1: $h1"; echo "  run2: $h2"; fi
printf '%s\n' "$out1" | grep -q FAULT && { fail=$((fail+1)); echo "FAIL clean run printed FAULT"; }

bad=$("$H" --sim --bug=past-end 2>&1); rcb=$?
exitis fault-exit "$rcb" 1
want fault-report "$bad" \
	'FAULT at launch 8 of 8 of kernel vecadd (simulator): global load outside every buffer' \
	'instruction: ' 'LDG.E (4-byte load)' \
	'thread: block 1024, thread 0 (element 1048576 of the whole job)' \
	'address: 0x' '0 bytes past the end of buffer a = [' \
	'next: '
printf '%s\n' "$bad" | grep -q 'checked 1048576' && { fail=$((fail+1)); echo "FAIL fault run went on past the faulting launch"; }
# The address line must name the real end of a: parse "a = [lo, hi)" and the faulting address.
lo_hi=$(printf '%s\n' "$bad" | sed -n 's/.*buffer a = \[\(0x[0-9a-f]*\), \(0x[0-9a-f]*\)).*/\1 \2/p')
addr=$(printf '%s\n' "$bad" | sed -n 's/^ *address: \(0x[0-9a-f]*\).*/\1/p')
set -- $lo_hi
if [ -n "$addr" ] && [ $# -eq 2 ] && [ $(( addr )) -eq $(( $2 )) ] && [ $(( $2 - $1 )) -eq 4194304 ]; then pass=$((pass+1)); echo "PASS fault-address-is-end-of-a"
else fail=$((fail+1)); echo "FAIL fault-address-is-end-of-a (addr=$addr range=$lo_hi)"; fi

# --bug=short (the check's own negative control, also used on the GPU by `make hello-gpu-control`): the last
# launch is told one element too few, so exactly one answer is never written and the check must go red.
short=$("$H" --sim --bug=short 2>&1); rcs=$?
exitis short-exit "$rcs" 1
want short-report "$short" 'checked 1048576 of 1048576 elements: 1 wrong, 1 never written' 'first mismatch: c[1048575]' 'FAIL'
printf '%s\n' "$short" | grep -qx PASS && { fail=$((fail+1)); echo "FAIL short run printed PASS"; }
usage=$("$H" --bug=past-end 2>&1); rcu=$?
exitis past-end-refused-on-gpu "$rcu" 2

echo "hello --sim tests: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
