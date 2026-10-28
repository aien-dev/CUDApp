# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Bar C3 (GpuStack PLAN): build rules for the C3 tests carried from omega-mv main e5b32567 (C3 work first landed on
# omega-mv c3/gemm a7d6463b9; src/omega_numeric* are the e5b32567 blobs).
# test-gpu-scan / test-gpu-gemm run the host and warp-simulator tiers; the chip tier is the binary with
# no flag, under `flock -w 900 /tmp/aien-gb10.lock`.
GPU_SCAN_TEST = $(OUT_DIR)/gpu_scan_test
$(GPU_SCAN_TEST): tests/gpu_scan_test.c tests/bw_warp_sim.h src/omega_gpu_scan.h src/omega_numeric_reduce.c src/omega_numeric.c $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_scan_test.c src/omega_numeric_reduce.c src/omega_numeric.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
GPU_GEMM_TEST = $(OUT_DIR)/gpu_gemm_test
$(GPU_GEMM_TEST): tests/gpu_gemm_test.c tests/bw_warp_sim_tc.h tests/bw_hmma_model.h tests/bw_qmma_model.h src/omega_gpu_gemm.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_gemm_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
GPU_TILELAB_TEST = $(OUT_DIR)/gpu_tilelab_test
$(GPU_TILELAB_TEST): tests/gpu_tilelab_test.c src/omega_gpu_gemm.h src/omega_gpu_session.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_tilelab_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
TILELAB_OMC ?= $(OMEGA_MV)/seed/out/omc
TILELAB_HOST_FAULT = $(OUT_DIR)/gpu_tilelab_host_fault
TILELAB_FAULT_GATE = $(OUT_DIR)/tilelab_host_fault
TILELAB_REPLAY = $(OUT_DIR)/tilelab_replay

# The fault binary compiles out device/session calls; the .om conductor invokes no shell.
$(TILELAB_HOST_FAULT): tests/gpu_tilelab_test.c src/omega_gpu_gemm.h src/omega_gpu_session.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -DTILELAB_HOST_FAULT -Itests -o $@ tests/gpu_tilelab_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
.PHONY: tilelab-compiler-check test-gpu-scan test-gpu-gemm test-tilelab-host tilelab-replay
tilelab-compiler-check:
	@test -x "$(TILELAB_OMC)" || { echo "NOT_RUN: set TILELAB_OMC to the tools/c5/omc.pin compiler"; exit 3; }
	@read repo rev unit hash < tools/c5/omc.pin; printf '%s  %s\n' "$$hash" "$(TILELAB_OMC)" | sha256sum -c -
$(TILELAB_FAULT_GATE): tools/tilelab_process.om tests/tilelab_host_fault.om | tilelab-compiler-check
	@mkdir -p $(OUT_DIR)
	cat tools/tilelab_process.om tests/tilelab_host_fault.om > $@.om
	$(TILELAB_OMC) < $@.om > $@.tmp
	chmod +x $@.tmp
	mv $@.tmp $@
$(TILELAB_REPLAY): tools/tilelab_process.om tools/tilelab_replay.om | tilelab-compiler-check
	@mkdir -p $(OUT_DIR)
	cat tools/tilelab_process.om tools/tilelab_replay.om > $@.om
	$(TILELAB_OMC) < $@.om > $@.tmp
	chmod +x $@.tmp
	mv $@.tmp $@
tilelab-replay: $(TILELAB_REPLAY)
test-gpu-scan: $(GPU_SCAN_TEST)
	./$(GPU_SCAN_TEST) --sim
test-gpu-gemm: $(GPU_GEMM_TEST)
	./$(GPU_GEMM_TEST) --sim
test-tilelab-host: $(GPU_TILELAB_TEST) $(TILELAB_HOST_FAULT) $(TILELAB_FAULT_GATE) $(TILELAB_REPLAY)
	./$(GPU_TILELAB_TEST) --host-only
	./$(TILELAB_FAULT_GATE) ./$(TILELAB_HOST_FAULT) ./$(TILELAB_REPLAY) "$$(mktemp -u -d $(OUT_DIR)/tilelab-host.XXXXXXXX)"
