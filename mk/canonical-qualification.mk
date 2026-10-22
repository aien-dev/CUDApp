# SPDX-License-Identifier: MIT
# Existing qualified source carried from Omega for canonical integration.
# C4 streams (docs/OMEGA.md row C4): two queues of the same half-chip work side by side, an event (cmdlist wait_for)
# between them, both with red checks. test-gpu-streams is host + warp simulator; the chip tier is "$(GPU_STREAMS_TEST)"
# with no flag, under the GB10 lock (timings need an exclusive window).
.PHONY: test-gpu-streams
GPU_STREAMS_TEST = $(OUT_DIR)/gpu_streams_test
$(GPU_STREAMS_TEST): tests/gpu_streams_test.c tests/bw_warp_sim.h src/omega_gpu_cmdlist.h src/omega_bw_em.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_streams_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-streams: $(GPU_STREAMS_TEST)
	./$(GPU_STREAMS_TEST) --sim

# C5 isolation (docs/OMEGA.md row C5): a never-ending kernel times out, omega_gpu_session_recover, a correct launch in the
# same process; two processes on one GPU; a hang beside a working process. test-gpu-isolation is host + simulator;
# the chip tier is "$(GPU_ISOLATION_TEST)" with no flag, under the GB10 lock.
.PHONY: test-gpu-isolation
GPU_ISOLATION_TEST = $(OUT_DIR)/gpu_isolation_test
$(GPU_ISOLATION_TEST): tests/gpu_isolation_test.c tests/bw_warp_sim.h src/omega_gpu_session.h src/omega_bw_em.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_isolation_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-isolation: $(GPU_ISOLATION_TEST)
	./$(GPU_ISOLATION_TEST) --host-only


# C3 (docs/OMEGA.md): matrix multiply of any shape, BF16 and FP8 (e4m3), both layouts, on the tensor cores; bit-exact
# against the chip-measured tensor-core model. test-gpu-gemm: host refusals and codegen, then the warp simulator.
# The chip tier is "$(GPU_GEMM_TEST)" with no flag (>= 200 shapes up to 8192, speed vs the measured ceiling), under the GB10 lock.
GPU_GEMM_TEST = $(OUT_DIR)/gpu_gemm_test
$(GPU_GEMM_TEST): tests/gpu_gemm_test.c tests/bw_warp_sim_tc.h tests/bw_hmma_model.h tests/bw_qmma_model.h src/omega_gpu_gemm.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_gemm_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
.PHONY: test-gpu-gemm
test-gpu-gemm: $(GPU_GEMM_TEST)
	./$(GPU_GEMM_TEST) --sim

# sc #642 (2026-10-18): Qwen3.8-27B decode graph (one token = one command list), chip test vs the APIs one call at a time
.PHONY: test-gpu-q38-graph-host
GPU_Q38_GRAPH_TEST = $(OUT_DIR)/gpu_q38_graph_test
$(GPU_Q38_GRAPH_TEST): tests/gpu_q38_graph_test.c src/omega_gpu_q38_graph.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_q38_graph_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-q38-graph-host: $(GPU_Q38_GRAPH_TEST)
	./$(GPU_Q38_GRAPH_TEST) --host-only

# HUM L314 exact two-pass lm_head (XBOUND simulator proofs; the chip run comes with the export)
GPU_EXACT_HEAD_TEST = $(OUT_DIR)/gpu_exact_head_test
$(GPU_EXACT_HEAD_TEST): tests/gpu_exact_head_test.c tests/bw_warp_sim_k.h src/omega_blackwell_codegen_dec.h src/omega_gpu_bf16_gemv_api.h src/omega_gpu_fp8_api.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_exact_head_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm


.PHONY: test-canonical-qual-host
test-canonical-qual-host: test-gpu-streams test-gpu-isolation test-gpu-gemm test-gpu-q38-graph-host

