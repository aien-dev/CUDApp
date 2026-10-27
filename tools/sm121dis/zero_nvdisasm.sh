#!/bin/sh
# zero_nvdisasm.sh: the second half of bar C6 in docs/OMEGA.md, "nothing calls nvdisasm" (L386).
#
# Two checks, both must pass:
#   1. Static. No file under src/ tests/ tools/ mk/ or the Makefile names the CUDA toolkit directory or looks
#      up an nvdisasm binary. The one declared exception is the oracle harness of the 331-kernel comparison,
#      tools/sm121dis/check.sh and oracle_pairs.sh, which exists to compare Omega's decoder WITH nvdisasm.
#   2. Dynamic. Every listing check and provenance check runs (a) with a DECOY nvdisasm first on PATH that records
#      each call and fails, (b) with every CUDA directory removed from PATH, and (c) under strace -f tracing execve.
#      PASS only when every command exits 0, the decoy was never called, and no execve names a path containing
#      "nvdisasm". A test that quietly fell back to the toolkit would fail one of the three.
#
# Usage: PHYSICS_DIR=<pinned physics checkout> sh tools/sm121dis/zero_nvdisasm.sh [build-dir]
#        after: make PHYSICS_DIR=... build/sm121dis build/gpu_attention_test build/gpu_elementwise_test \
#                    build/gpu_fp8_api_test build/gpu_reconv_test build/test_omega_reduce_cpu
# Last line: "ZERO-NVDISASM: PASS (...)" or "ZERO-NVDISASM: FAIL (...)". Exit 0 only on PASS.
set -u
OUT=${1:-build}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/zero_nvdisasm.XXXXXX")
fail=0

command -v strace > /dev/null 2>&1 || { echo "ZERO-NVDISASM: NOT_RUN (strace missing)"; exit 2; }

# 1. static
hits=$(grep -rnE '/usr/local/cuda|command -v nvdisasm|which nvdisasm|cuda/bin/nvdisasm' src tests tools mk Makefile 2>/dev/null \
       | grep -vE '^tools/sm121dis/(check|oracle_pairs|zero_nvdisasm)\.sh:')
if [ -n "$hits" ]; then
    echo "static: FAIL, these lines name the CUDA toolkit or look up nvdisasm:"; echo "$hits" | head -20; fail=1
else
    echo "static: PASS, no file outside the oracle harness names the CUDA toolkit or looks up nvdisasm"
fi

# 2. dynamic
mkdir -p "$WORK/decoy"
cat > "$WORK/decoy/nvdisasm" <<'EOF'
#!/bin/sh
echo "DECOY nvdisasm called with: $*" >> "$ZERO_NVDISASM_MARK"
exit 1
EOF
chmod +x "$WORK/decoy/nvdisasm"
ZERO_NVDISASM_MARK=$WORK/decoy-calls; export ZERO_NVDISASM_MARK
cleanpath=$(printf '%s' "$PATH" | tr ':' '\n' | grep -vi 'cuda' | paste -sd: -)
n=0
run() {
    name=$1; shift
    n=$((n + 1))
    env PATH="$WORK/decoy:$cleanpath" strace -f -qq -e trace=execve -o "$WORK/$name.strace" "$@" > "$WORK/$name.out" 2>&1
    rc=$?
    ex=$(grep -cE 'execve\("[^"]*nvdisasm[^"]*"' "$WORK/$name.strace" 2>/dev/null)
    printf '%s: exit %s, execve naming nvdisasm: %s' "$name" "$rc" "$ex"
    if [ "$rc" -eq 0 ] && [ "$ex" -eq 0 ]; then echo "  ok"; else echo "  FAIL"; fail=1; fi
}
run attention_hd64  "$OUT/gpu_attention_test" --host-only
run attention_hd128 "$OUT/gpu_attention_test" --hd 128 --host-only
run elementwise     "$OUT/gpu_elementwise_test" --host-only
run fp8             "$OUT/gpu_fp8_api_test" --host-only
run reconv          "$OUT/gpu_reconv_test" --host-only
run divsqrt         sh tools/divsqrt_nvdisasm_check.sh
run ldst            sh tools/ldst_nvdisasm_check.sh
run reduce          make ${PHYSICS_DIR:+PHYSICS_DIR=$PHYSICS_DIR} test-numeric-reduce-nvdisasm
if [ "${2:-}" = "--red" ]; then
    # Red check of the gate itself: two planted shell-outs, one by name (the decoy must catch it) and one by the
    # toolkit's absolute path (strace must catch it). The gate must then say FAIL; a gate that stays green here is broken.
    run planted_by_name sh -c 'nvdisasm --version'
    run planted_by_path sh -c '/usr/local/cuda/bin/nvdisasm --version > /dev/null 2>&1; exit 0'
fi

calls=0
[ -f "$ZERO_NVDISASM_MARK" ] && calls=$(wc -l < "$ZERO_NVDISASM_MARK")
if [ "$calls" -ne 0 ]; then echo "decoy: FAIL, called $calls time(s):"; head -5 "$ZERO_NVDISASM_MARK"; fail=1
else echo "decoy: never called"; fi

if [ "$fail" -eq 0 ]; then
    echo "ZERO-NVDISASM: PASS ($n commands under strace, decoy calls 0, no execve of nvdisasm, static clean)"
    rm -rf "$WORK"; exit 0
fi
echo "ZERO-NVDISASM: FAIL (details in $WORK)"; exit 1
