# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Build rules for the GPU tests carried from aien-dev/omega-mv main e5b32567 (the rules are
# omega-mv's Makefile rules for the same files, unchanged apart from living here). Picked up by
# the `-include mk/*.mk` line of the Makefile, before it sets HOST_TESTS, which lists
# $(CARRIED_HOST_TESTS). The C3 tests (gpu_scan_test, gpu_gemm_test) have their rules in mk/c3.mk.
# Host-only targets open no device; the chip tiers run the same binaries with no flag, under the GB10 lock.

BW_ENCODE_CONTROL_TEST = $(OUT_DIR)/bw_encode_control_test
$(BW_ENCODE_CONTROL_TEST): tests/bw_encode_control_test.c src/omega_blackwell_codegen.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/bw_encode_control_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# Qwen3.8-27B decode attention, head_dim 256 (omega-mv#544).
GPU_DATTN256_TEST = $(OUT_DIR)/gpu_decode_attn256_test
$(GPU_DATTN256_TEST): tests/gpu_decode_attn256_test.c tests/bw_warp_sim_k.h src/omega_gpu_decode_attn256_api.h src/omega_blackwell_codegen_attn256.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_decode_attn256_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
GPU_DATTN256_MULTI_TEST = $(OUT_DIR)/gpu_dattn256_multi_test
$(GPU_DATTN256_MULTI_TEST): tests/gpu_dattn256_multi_test.c src/omega_gpu_decode_attn256_api.h src/omega_blackwell_codegen_attn256.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -Itests -o $@ tests/gpu_dattn256_multi_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# Qwen3.8-27B Gated DeltaNet decode step (omega#542).
GPU_GDN_TEST = $(OUT_DIR)/gpu_gdn_test
$(GPU_GDN_TEST): tests/gpu_gdn_test.c tests/bw_warp_sim_k.h src/omega_gpu_gdn_api.h src/omega_blackwell_codegen_gdn.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_gdn_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
GPU_GDN_MULTI_TEST = $(OUT_DIR)/gpu_gdn_multi_test
$(GPU_GDN_MULTI_TEST): tests/gpu_gdn_multi_test.c tests/bw_warp_sim_k.h src/omega_gpu_gdn_api.h src/omega_blackwell_codegen_gdn.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_gdn_multi_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# Qwen3.8-27B prefill attention, head_dim 256 (omega-mv#545).
GPU_PATTN256_TEST = $(OUT_DIR)/gpu_prefill_attn256_test
$(GPU_PATTN256_TEST): tests/gpu_prefill_attn256_test.c tests/bw_hmma_model.h tests/bw_warp_sim_pa.h src/omega_gpu_prefill_attn256_api.h src/omega_blackwell_codegen_pattn256.h src/omega_blackwell_codegen_attn256.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -DWS_MAX_THREADS=256 -Itests -o $@ tests/gpu_prefill_attn256_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
# Qwen3.6-35B-A3B flash-style tensor-core prefill attention (Linear DRA-85, P-DRA-4).
GPU_FATTN256_TEST = $(OUT_DIR)/gpu_prefill_fattn256_test
$(GPU_FATTN256_TEST): tests/gpu_prefill_fattn256_test.c tests/bw_hmma_model.h tests/bw_warp_sim_pa.h src/omega_gpu_prefill_fattn256_api.h src/omega_blackwell_codegen_fattn256.h src/omega_blackwell_codegen_attn256.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -DWS_MAX_THREADS=256 -Itests -o $@ tests/gpu_prefill_fattn256_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# Qwen3.8-27B decode graph (sc #642) and the exact two-pass lm_head (HUM L314).
GPU_Q38_GRAPH_TEST = $(OUT_DIR)/gpu_q38_graph_test
$(GPU_Q38_GRAPH_TEST): tests/gpu_q38_graph_test.c src/omega_gpu_q38_graph.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_q38_graph_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
GPU_EXACT_HEAD_TEST = $(OUT_DIR)/gpu_exact_head_test
$(GPU_EXACT_HEAD_TEST): tests/gpu_exact_head_test.c tests/bw_warp_sim_k.h src/omega_blackwell_codegen_dec.h src/omega_gpu_bf16_gemv_api.h src/omega_gpu_fp8_api.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_exact_head_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# C4 streams and C5 isolation (docs/OMEGA.md rows C4, C5 in omega-mv).
GPU_STREAMS_TEST = $(OUT_DIR)/gpu_streams_test
$(GPU_STREAMS_TEST): tests/gpu_streams_test.c tests/bw_warp_sim.h src/omega_gpu_cmdlist.h src/omega_bw_em.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_streams_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
GPU_ISOLATION_TEST = $(OUT_DIR)/gpu_isolation_test
$(GPU_ISOLATION_TEST): tests/gpu_isolation_test.c tests/bw_warp_sim.h src/omega_gpu_session.h src/omega_bw_em.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_isolation_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

.PHONY: test-bw-encode-control test-gpu-decode-attn256 test-gpu-gdn test-gpu-q38-graph-host test-gpu-streams \
	test-gpu-isolation test-gpu-prefill-fattn256 test-carried-bins
test-bw-encode-control: $(BW_ENCODE_CONTROL_TEST)
	./$(BW_ENCODE_CONTROL_TEST)
test-gpu-decode-attn256: $(GPU_DATTN256_TEST)
	./$(GPU_DATTN256_TEST) --host-only
	./$(GPU_DATTN256_TEST) --sim
test-gpu-gdn: $(GPU_GDN_TEST)
	./$(GPU_GDN_TEST) --host-only
	./$(GPU_GDN_TEST) --sim
test-gpu-q38-graph-host: $(GPU_Q38_GRAPH_TEST)
	./$(GPU_Q38_GRAPH_TEST) --host-only
test-gpu-streams: $(GPU_STREAMS_TEST)
	./$(GPU_STREAMS_TEST) --sim
test-gpu-isolation: $(GPU_ISOLATION_TEST)
	./$(GPU_ISOLATION_TEST) --host-only
test-gpu-prefill-fattn256: $(GPU_FATTN256_TEST)
	./$(GPU_FATTN256_TEST) --sim
# Chip-only binaries: built here so the host suite at least compiles them.
test-carried-bins: $(GPU_DATTN256_MULTI_TEST) $(GPU_GDN_MULTI_TEST) $(GPU_PATTN256_TEST) $(GPU_FATTN256_TEST) $(GPU_EXACT_HEAD_TEST)

CARRIED_HOST_TESTS = test-bw-encode-control test-gpu-decode-attn256 test-gpu-gdn test-gpu-q38-graph-host \
	test-gpu-streams test-gpu-isolation test-gpu-prefill-fattn256 test-carried-bins
