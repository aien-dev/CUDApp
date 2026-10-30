#!/bin/sh
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Host check of build/c5_matrix with stand-in processes (no GPU, no CUDA): each case must go green with good stand-ins
# and red with bad ones. This checks the orchestration (spawn, pipes, markers, group kill, reaping, verdicts) only.
set -eu
cd "$(dirname "$0")/../.."
M=$(pwd)/build/c5_matrix
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
cat > "$T/iso" <<'EOF'
#!/bin/sh
if [ "$1" = "--values-only" ]; then echo "C5 isolation values: PASS (stand-in)"; /bin/sleep 1; exit 0; fi
/bin/sh -c 'echo "[$$] ok: hang part: a command list made before the hang"; exec /bin/sleep 30' &
wait
EOF
printf '#!/bin/sh\necho "C5 isolation values: FAIL (stand-in)"; exit 1\n' > "$T/iso_bad"
printf '#!/bin/sh\n/bin/sleep 1; echo "[cuda] READY"; /bin/sleep 4; exit $5\n' > "$T/py"
printf '#!/bin/sh\n/bin/sleep 1; echo "[cuda] READY"; exit 0\n' > "$T/py_dies"
printf '#!/bin/sh\necho "no READY here"; exit 1\n' > "$T/py_noready"
chmod +x "$T"/*
fails=0
expect() { # want_rc case iso py
    rc=0; out=$(timeout 120 "$M" "$T/$3" "$T/$4" "$2" 2>&1) || rc=$?
    if [ "$rc" -eq "$1" ]; then echo "ok: case $2 iso=$3 py=$4 exit $rc"; else echo "FAIL: case $2 iso=$3 py=$4 exit $rc, want $1"; echo "$out"; fails=$((fails + 1)); fi
}
expect 0 c iso py
expect 0 p iso py
expect 0 k iso py
expect 0 r iso py
expect 1 c iso_bad py
expect 1 k iso_bad py
expect 1 c iso py_dies
expect 2 c iso py_noready
expect 2 x iso py
leftover=$(ps -eo args | grep -c "^/bin/sleep 30$" || true)
[ "$leftover" -eq 0 ] && echo "ok: no stand-in hang survives the group kill" || { echo "FAIL: $leftover stand-in hang(s) left"; fails=$((fails + 1)); }
[ "$fails" -eq 0 ] && echo "c5_matrix selftest: PASS" || { echo "c5_matrix selftest: FAIL ($fails)"; exit 1; }
