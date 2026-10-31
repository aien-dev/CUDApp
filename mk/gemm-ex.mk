# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# The general matrix multiply omega_gpu_gemm_ex (src/omega_gpu_gemm_ex.h): any M, N, K >= 1, op N/T, leading dimensions,
# alpha/beta, strided batches, f32/bf16/f16/e4m3/e5m2 inputs, fp8 scales, over the C3 kernels of src/omega_gpu_gemm.c.
# MIGRATION-PENDING C: src/omega_gpu_gemm_ex.c is new C (GpuStack PLAN rules), built only with the switch
# CUDAPP_MIGRATION_PENDING_GEMM_EX=1. The switch is ON by default because the file adds capability and changes no served
# path: the only library callers are the fp8 GEMV and grouped GEMM fallbacks, taken only for shapes those kernels refuse.
# With CUDAPP_MIGRATION_PENDING_GEMM_EX=0 the entry refuses every call (OMEGA_GPU_GEMM_UNSUPPORTED) and the fp8 calls
# return their old refusal codes. .om successor: src/omega_gpu_gemm_ex.om (aien-dev/CUDApp#144).
# Picked up by the `-include mk/*.mk` line of the Makefile, after CFLAGS and before HOST_TESTS.
CUDAPP_MIGRATION_PENDING_GEMM_EX ?= 1
CFLAGS += -DCUDAPP_MIGRATION_PENDING_GEMM_EX=$(CUDAPP_MIGRATION_PENDING_GEMM_EX)

GPU_GEMM_EX_TEST = $(OUT_DIR)/gpu_gemm_ex_test
$(GPU_GEMM_EX_TEST): tests/gpu_gemm_ex_test.c tests/bw_warp_sim_tc.h tests/bw_hmma_model.h tests/bw_qmma_model.h \
		src/omega_gpu_gemm_ex.h src/omega_gpu_gemm.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_gemm_ex_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# Host tiers: no device opened. The chip tier is the same binary with no flag, under the GB10 lock
# (flock -w 900 /tmp/aien-gb10.lock $(GPU_GEMM_EX_TEST) [--part I/N]); --list prints the predeclared sweep.
.PHONY: test-gpu-gemm-ex test-gpu-gemm-ex-sim
test-gpu-gemm-ex: $(GPU_GEMM_EX_TEST)
	./$(GPU_GEMM_EX_TEST) --host-only
test-gpu-gemm-ex-sim: $(GPU_GEMM_EX_TEST)
	./$(GPU_GEMM_EX_TEST) --sim

CARRIED_HOST_TESTS += test-gpu-gemm-ex test-gpu-gemm-ex-sim
