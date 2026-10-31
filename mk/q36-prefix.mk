# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# DRA-86: device-resident prefix snapshots of the hybrid Qwen3.6-35B (src/omega_gpu_q36_prefix.{c,h}). Host only: the
# test's copy / alloc / free callbacks work on a host arena. The off-by-one restore mutant (AIEN_OMEGA_Q36_PF_MUTANT=10)
# and the negative control (refusal checks removed) must each make the binary FAIL.
.PHONY: test-gpu-q36-prefix
GPU_Q36_PREFIX_TEST = $(OUT_DIR)/gpu_q36_prefix_test
$(GPU_Q36_PREFIX_TEST): tests/gpu_q36_prefix_test.c src/omega_gpu_q36_prefix.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_q36_prefix_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-q36-prefix: $(GPU_Q36_PREFIX_TEST)
	./$(GPU_Q36_PREFIX_TEST) --host-only
	! AIEN_OMEGA_Q36_PF_MUTANT=10 ./$(GPU_Q36_PREFIX_TEST) > $(OUT_DIR)/gpu_q36_prefix_mutant10.txt
	! ./$(GPU_Q36_PREFIX_TEST) --negative-control > $(OUT_DIR)/gpu_q36_prefix_negative.txt
	@echo "test-gpu-q36-prefix: mutant 10 and the negative control FAIL as required"
CARRIED_HOST_TESTS += test-gpu-q36-prefix
