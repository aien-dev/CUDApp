#!/bin/sh
# corpus.sh <build-out-dir> <kernel-dir>: collects every kernel the library builds, without a GPU, into <kernel-dir>.
# Each GPU test runs in its host-only (and simulator) modes with OMEGA_KERNEL_DUMP_DIR set, so every program the
# library encodes lands as <sha16>.bin (src/omega_blackwell_codegen.c omega_kernel_dump); the numeric kernels (DIV, SQRT,
# transcendentals, load/store, reductions) have their own encoder and their own --dump. Run from the repo root after
# `make OUT_DIR=<build-out-dir> sm121dis-corpus-bins` (mk/sm121dis.mk). Any failed producer invalidates the corpus.
set -u
OUT=$1; K=$2
mkdir -p "$K"; K=$(cd "$K" && pwd)
export OMEGA_KERNEL_DUMP_DIR="$K"
run() {
    printf '%s' "$*: "
    "$@" > "$K/producer.log" 2>&1; rc=$?
    echo "exit $rc, $(ls "$K"/*.bin 2>/dev/null | wc -l) kernels so far"
    if [ "$rc" -ne 0 ]; then
        cat "$K/producer.log" >&2
        echo "VERDICT FAIL: corpus producer failed: $*" >&2
        exit "$rc"
    fi
    rm -f "$K/producer.log"
}
for t in gpu_matmul_api_test gpu_elementwise_test gpu_reconv_test gpu_attention_test gpu_moe_window_test gpu_fp8_api_test \
         gpu_decode_attn_test gpu_router_test gpu_bf16_gemv_test gpu_tern_test gpu_fp8gemm_test gpu_tc_probe_test \
         gpu_tc_peak_test gpu_pf_mem_test gpu_qmma_probe_test gpu_cmdlist_test gpu_decode_graph_test; do
    run "$OUT/$t" --host-only
done
run "$OUT/gpu_attention_test" --hd 128 --host-only
for t in gpu_tcgemm_test gpu_prefill_attn_test gpu_pfgemm_test gpu_qgemm_test gpu_pfhost_test gpu_cmdlist_test; do
    run "$OUT/$t" --host-only; run "$OUT/$t" --sim
done
run "$OUT/gpu_gpr_boundary_test"
for t in gpu_scan_test gpu_gemm_test gpu_streams_test; do
    run "$OUT/$t" --host-only; run "$OUT/$t" --sim
done
run "$OUT/gpu_isolation_test" --host-only
mkdir -p "$K/numeric"
run build/test_omega_divsqrt_gb10_cpu --dump "$K/numeric"
run build/test_omega_ldst_gb10_cpu --dump "$K/numeric"
for f in "$K"/numeric/*.bin; do [ -f "$f" ] && mv "$f" "$K/num_$(basename "$f")"; done; rmdir "$K/numeric" 2>/dev/null
echo "corpus: $(ls "$K"/*.bin | wc -l) kernels in $K"
