# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# CUDA++ build. Plain make + gcc; no CUDA toolkit, no NVIDIA user-space library.
#
#   make            libomega_gpu.a (= libcudapp.a): GPU toolchain + runtime + libraries + driver client
#   make test       every host-only test (no GPU needed)
#   make doctor     is this machine ready? one line per check, each FAIL with its fix (no GPU opened)
#   make hello-gpu  run the doctor, then build examples/hello_gpu and run it (on a GB10)
#
# The driver client (nvrm/, m16/) and the NVIDIA open headers (third_party/) are in
# this repository, so PHYSICS_DIR is "." here. Omega is meant to build the same sources
# with PHYSICS_DIR pointing at a pinned CUDA++ checkout (cudapp.lock); that pin does not
# exist yet and Omega main still takes nvrm/m16 from physics-mv (#106, #107).
CC ?= gcc
.DEFAULT_GOAL := all
PHYSICS_DIR ?= .
OUT_DIR ?= build
GPU_PROD_TARGET ?= $(OUT_DIR)/libomega_gpu.a
NVIDIA_OPEN := $(PHYSICS_DIR)/third_party/nvidia-open-580.173.02
CFLAGS ?= -std=gnu11 -Wall -Wextra -Werror -MMD -MP -D_GNU_SOURCE -O2 -Isrc \
	-I$(PHYSICS_DIR)/m16 -I$(PHYSICS_DIR)/nvrm \
	-I$(NVIDIA_OPEN)/src/common/sdk/nvidia/inc \
	-I$(NVIDIA_OPEN)/kernel-open/common/inc \
	-I$(NVIDIA_OPEN)/kernel-open/nvidia-uvm \
	-I$(NVIDIA_OPEN)/src/nvidia/arch/nvalloc/unix/include
-include mk/*.mk

# sm_121 instruction encoder (CUDApp#79): om (default) = om/encoder/bw_enc_sm121.om through the generated binding
# src/omega_om_units.c; c = the migration-pending C encoder, the differential oracle; diff = both on every call,
# abort on any difference (mk/om-encoder.mk). Use a separate OUT_DIR per value.
CUDAPP_ENCODER ?= om
CUDAPP_ENCODER_DEFS_om :=
CUDAPP_ENCODER_DEFS_c := -DCUDAPP_MIGRATION_PENDING_C_ENCODER
CUDAPP_ENCODER_DEFS_diff := -DCUDAPP_MIGRATION_PENDING_ENCODER_DIFF
CUDAPP_ENCODER_DEFS = $(CUDAPP_ENCODER_DEFS_$(CUDAPP_ENCODER))
ifeq ($(filter om c diff,$(CUDAPP_ENCODER)),)
$(error CUDAPP_ENCODER must be om, c or diff)
endif

# Every C file of the GPU part, the shared base (tools/extract/shared.txt), the driver client and the generated
# binding of the seeded Omega units (om/units.sh).
SRCS := $(sort $(wildcard src/omega_blackwell_*.c src/omega_gpu_*.c)) \
	src/omega_cubin.c src/omega_native_artifact.c src/omega_sm121_disasm.c \
	src/sha256.c src/omega_canonical.c src/omega_core.c src/omega_vector.c src/omega_om_units.c \
	$(PHYSICS_DIR)/m16/m16_native.c $(PHYSICS_DIR)/nvrm/nvrm.c
OBJS = $(patsubst %.c,$(OUT_DIR)/%.o,$(notdir $(SRCS)))
DEPS = $(OBJS:.o=.d)
-include $(DEPS)
vpath %.c src $(PHYSICS_DIR)/m16 $(PHYSICS_DIR)/nvrm

$(OUT_DIR):
	mkdir -p $@
$(OUT_DIR)/%.o: %.c | $(OUT_DIR)
	$(CC) $(CFLAGS) $(CUDAPP_ENCODER_DEFS) -c $< -o $@

.PHONY: all libcudapp test clean
all: $(OUT_DIR)/libomega_gpu.a $(OUT_DIR)/libcudapp.a
$(OUT_DIR)/libcudapp.a: $(OUT_DIR)/libomega_gpu.a
	cp $< $@
libcudapp: $(OUT_DIR)/libcudapp.a
clean:
	rm -rf $(OUT_DIR) build

# FB-1 cut 1: native (no CUDA) matmul entry point for the inference stack.
# libomega_gpu.a = everything the tool links except its main; the Rust FFI crate
# (cut 2) links it. test-gpu-matmul-api runs the host-only refusals; the chip
# sweep is `./build/gpu_matmul_api_test --out receipt.json` through the heavy queue.
.PHONY: artifact-export test-native-artifact
# Offline handoff for one native realization (CUDApp #68 / #69). Opens no device.
ARTIFACT_EXPORT = $(OUT_DIR)/artifact-export
artifact-export: $(ARTIFACT_EXPORT)
ARTIFACT_TEST = $(OUT_DIR)/gpu_artifact_export_test
$(ARTIFACT_EXPORT): tools/artifact_export.c src/omega_native_artifact.c src/omega_cubin.c $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tools/artifact_export.c src/omega_native_artifact.c src/omega_cubin.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
$(ARTIFACT_TEST): tests/gpu_artifact_export_test.c src/omega_native_artifact.c src/omega_cubin.c $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/gpu_artifact_export_test.c src/omega_native_artifact.c src/omega_cubin.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
# cubin-probe is the only target that links NVIDIA's libcuda: an interop check that NVIDIA's own loader accepts
# CUDA++'s cubin. It is not in `make test`; zero-cuda-gate (in HOST_TESTS) checks hello_gpu and the API test link no CUDA.
CUDA_INC ?= /usr/local/cuda/include
CUDA_LIB ?= /usr/local/cuda/lib64
$(OUT_DIR)/cubin_probe: tools/interop/cubin_probe.c $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -I$(CUDA_INC) -o $@ tools/interop/cubin_probe.c $(OUT_DIR)/libomega_gpu.a -L$(CUDA_LIB) -lcuda -lpthread -lm
cubin-probe: $(OUT_DIR)/cubin_probe
# Artifacts are immutable (export refuses an existing file), so each run starts from an empty directory.
test-native-artifact: $(ARTIFACT_EXPORT) $(ARTIFACT_TEST)
	rm -rf $(OUT_DIR)/artifact-test
	./$(ARTIFACT_TEST) $(OUT_DIR)/artifact-test
# Pure artifact-format boundary: portable host test, no driver or device objects.
.PHONY: test-native-artifact-manifest
$(OUT_DIR)/native_artifact_manifest_test: tests/native_artifact_manifest_test.c src/omega_native_artifact.c src/omega_cubin.c src/omega_blackwell_realize.c src/sha256.c src/omega_om_units.c | $(OUT_DIR)
	$(CC) $(CFLAGS) -ffunction-sections -fdata-sections -Wl,--gc-sections -o $@ $^ -lpthread
test-native-artifact-manifest: $(OUT_DIR)/native_artifact_manifest_test
	./$(OUT_DIR)/native_artifact_manifest_test
.PHONY: libomega_gpu test-gpu-matmul-api
# The two gate-runner objects (omegatool's M17/M18/M19 and World gates) are NOT part of the
# library: nothing in the library or in the Rust FFI crate calls them (checked with nm), and
# they compile the physics checkout path into the object (-DOMEGA_PHYSICS_DIR), which made the
# archive digest depend on where physics was checked out. Without them libomega_gpu.a is
# byte-identical for any PHYSICS_DIR and any omega checkout directory.
GPU_API_OBJS = $(OBJS)
GPU_API_TEST = $(OUT_DIR)/gpu_matmul_api_test
GPU_R4_RECEIPT = $(OUT_DIR)/gpu_matmul_r4_receipt
$(OUT_DIR)/libomega_gpu.a: $(GPU_API_OBJS)
	ar rcs $@ $(GPU_API_OBJS)
libomega_gpu: $(OUT_DIR)/libomega_gpu.a
$(GPU_API_TEST): tests/gpu_matmul_api_test.c src/omega_gpu_matmul_api.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/gpu_matmul_api_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-matmul-api: $(GPU_API_TEST) $(GPU_R4_RECEIPT)
	./$(GPU_API_TEST) --host-only
	./$(GPU_R4_RECEIPT) --host-only

# omega #473 R4: the sealed CTA-budget receipt (full-output comparison at the training shapes,
# every budget, repeats, I42 stale-output check, negative controls). Host-only self-test above; the
# chip receipt is `quietlock run -- ./build/gpu_matmul_r4_receipt --out receipt.json`.
$(GPU_R4_RECEIPT): tests/gpu_matmul_r4_receipt.c src/omega_gpu_matmul_api.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/gpu_matmul_r4_receipt.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# FB-1 cut 4: native rmsnorm / rope / swiglu (+ EX2 and shared-exchange probes).
# test-gpu-elementwise is host-only (refusals, codegen, word fixtures, nvdisasm
# listing); the chip gate is `./build/gpu_elementwise_test --out receipt.json`
# through the heavy queue. -ffp-contract=off keeps the f32 oracle free of FMA.
.PHONY: test-gpu-elementwise
GPU_EW_TEST = $(OUT_DIR)/gpu_elementwise_test
$(GPU_EW_TEST): tests/gpu_elementwise_test.c src/omega_gpu_elementwise_api.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_elementwise_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-elementwise: $(GPU_EW_TEST)
	./$(GPU_EW_TEST) --host-only

# omega #308: structured warp reconvergence (BSSY/BSYNC regions). test-gpu-reconv is
# host-only (encoder goldens, region bookkeeping, refusals, nvdisasm listing, every probe
# kernel through the SIMT warp simulator tests/bw_warp_sim.h against the oracle, mutation
# controls); the chip gate is tools/run_gpu_reconv_chip.sh.
.PHONY: test-gpu-reconv
GPU_RECONV_TEST = $(OUT_DIR)/gpu_reconv_test
$(GPU_RECONV_TEST): tests/gpu_reconv_test.c tests/bw_warp_sim.h src/omega_gpu_elementwise_api.h src/omega_bw_reconv.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -Itests -o $@ tests/gpu_reconv_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-reconv: $(GPU_RECONV_TEST)
	./$(GPU_RECONV_TEST) --host-only

# FB-1 cut 5: native gqa_attention (f32 KV) and paged_attention (bf16 KV, + batch).
# head_dim 128 (Qwen3-4B): the same battery with --hd 128 through both host simulators; GB10 parity PASS 126/0
# at 7dd92dc (evidence/FB1-CUT5-HD128-7dd92dc).
# test-gpu-attention runs host-only (refusals, codegen, nvdisasm listing) and then the
# whole parity battery through the host IR simulator (--sim, no chip); the chip
# gate is `./build/gpu_attention_test --out receipt.json` through the heavy queue
# (tools/run_gpu_attention_chip.sh). -ffp-contract=off keeps the oracle free of FMA.
.PHONY: test-gpu-attention
GPU_ATTN_TEST = $(OUT_DIR)/gpu_attention_test
$(GPU_ATTN_TEST): tests/gpu_attention_test.c tests/bw_warp_sim.h src/omega_gpu_attention_api.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_attention_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-attention: $(GPU_ATTN_TEST)
	./$(GPU_ATTN_TEST) --host-only
	./$(GPU_ATTN_TEST) --sim
	./$(GPU_ATTN_TEST) --sweep --sim --out $(OUT_DIR)/gpu_attention_sweep_sim.json
	./$(GPU_ATTN_TEST) --hd 128 --host-only
	./$(GPU_ATTN_TEST) --hd 128 --sim
	./$(GPU_ATTN_TEST) --hd 128 --divergent --sim

# FB-1 cut 4b: fresh-process device-open probe (flake investigation, tools/probe_gpu_open.sh)
$(OUT_DIR)/gpu_session_probe: tests/gpu_session_probe.c src/omega_gpu_session.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/gpu_session_probe.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# OWN-30B cut 2: window matmul gate (tests/gpu_moe_window_test.c). test-gpu-moe-window runs the host-only
# refusals and the packed-window byte rule; the chip gate is `./build/gpu_moe_window_test --out r.json`
# under /tmp/aien-gb10.lock (equivalence to the whole-tensor columns, mutants, window timing).
GPU_MOE_TEST = $(OUT_DIR)/gpu_moe_window_test
$(GPU_MOE_TEST): tests/gpu_moe_window_test.c src/omega_gpu_moe_api.h src/omega_gpu_matmul_api.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/gpu_moe_window_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
.PHONY: test-gpu-moe-window
test-gpu-moe-window: $(GPU_MOE_TEST)
	./$(GPU_MOE_TEST) --host-only

# OWN-30B Cut 6 / Cut 5 (2026-10-13): FP8 e4m3 block-scaled GEMV (omega_gpu_fp8_api.h). test-gpu-fp8 is host-only
# (exhaustive conversion, pack refusals, codegen budgets, nvdisasm listing) then the warp-simulator battery
# (parity, grouped experts, mutants, both fragment orders); the chip gate is tools/run_gpu_fp8_chip.sh.
# -ffp-contract=off keeps the double oracle free of FMA.
.PHONY: test-gpu-fp8
GPU_FP8_TEST = $(OUT_DIR)/gpu_fp8_api_test
$(GPU_FP8_TEST): tests/gpu_fp8_api_test.c tests/bw_warp_sim.h src/omega_gpu_fp8_api.h src/omega_blackwell_codegen_moe.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_fp8_api_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-fp8: $(GPU_FP8_TEST)
	./$(GPU_FP8_TEST) --host-only
	./$(GPU_FP8_TEST) --sim

# omega #457 (2026-10-15): host rewrite of a GPU-cached buffer must be visible to the next launch. Chip only; the gate
# is tools/run_gpu_l2_stale_chip.sh.
GPU_L2_STALE_TEST = $(OUT_DIR)/gpu_l2_stale_test
$(GPU_L2_STALE_TEST): tests/gpu_l2_stale_test.c src/omega_gpu_fp8_api.h src/omega_gpu_session.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_l2_stale_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# OWN-30B cut 8 (2026-10-13): one-launch decode attention (omega_gpu_decode_attn_api.h). test-gpu-decode-attn is
# host-only + simulator; the chip gate is tools/run_gpu_decode_attn_chip.sh.
.PHONY: test-gpu-decode-attn
GPU_DATTN_TEST = $(OUT_DIR)/gpu_decode_attn_test
$(GPU_DATTN_TEST): tests/gpu_decode_attn_test.c tests/bw_warp_sim_k.h src/omega_gpu_decode_attn_api.h src/omega_blackwell_codegen_attn.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_decode_attn_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-decode-attn: $(GPU_DATTN_TEST)
	$(GPU_DATTN_TEST) --host-only
	$(GPU_DATTN_TEST) --sim

# OWN-30B cut 8 item 2 (2026-10-13): one-launch MoE router (omega_gpu_router_api.h). test-gpu-router is host-only
# (oracle, budgets) plus the warp simulator; the chip gate is tools/run_gpu_router_chip.sh under the GB10 lock.
.PHONY: test-gpu-router
GPU_ROUTER_TEST = $(OUT_DIR)/gpu_router_test
$(GPU_ROUTER_TEST): tests/gpu_router_test.c tests/bw_warp_sim_k.h src/omega_gpu_router_api.h src/omega_blackwell_codegen_router.h src/omega_gpu_fp8_api.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_router_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-router: $(GPU_ROUTER_TEST)
	./$(GPU_ROUTER_TEST) --host-only
	./$(GPU_ROUTER_TEST) --sim

# OWN-30B (2026-10-13): bf16 streaming GEMV, the lm_head (omega_gpu_bf16_gemv_api.h). test-gpu-bf16-gemv is host-only
# (emulation, budgets) plus the warp simulator (bit-identical to the host f32 emulation, mutants 1-3 caught); the chip
# gate is tools/run_gpu_bf16_gemv_chip.sh under the GB10 lock.
.PHONY: test-gpu-bf16-gemv
GPU_BF16_TEST = $(OUT_DIR)/gpu_bf16_gemv_test
$(GPU_BF16_TEST): tests/gpu_bf16_gemv_test.c tests/bw_warp_sim_k.h src/omega_gpu_bf16_gemv_api.h src/omega_blackwell_codegen_bf16.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_bf16_gemv_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-bf16-gemv: $(GPU_BF16_TEST)
	$(GPU_BF16_TEST) --host-only
	$(GPU_BF16_TEST) --sim

# The ternary GEMV host test (tests/gpu_tern_test.c) stays in Omega: its reference is Omega's ternary
# algebra oracle (src/algebra), which is not part of CUDA++. The library itself (omega_gpu_tern_api) is here.


# OWN-30B (2026-10-13): report-only lm_head study, fp8 vs bf16 argmax agreement on the real checkpoint (chip, GB10 lock).
$(OUT_DIR)/gpu_lmhead_fp8_study: tests/gpu_lmhead_fp8_study.c src/omega_gpu_bf16_gemv_api.h src/omega_gpu_fp8_api.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -O2 -o $@ tests/gpu_lmhead_fp8_study.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# OWN-30B prefill (2026-10-13): FP8 grouped GEMM (omega_gpu_fp8gemm_api.h). test-gpu-fp8gemm is host-only (emulation,
# permutation, budgets) plus the warp simulator (512 threads); the chip gate is tools/run_gpu_fp8gemm_chip.sh under the GB10 lock.
.PHONY: test-gpu-fp8gemm
GPU_FP8GEMM_TEST = $(OUT_DIR)/gpu_fp8gemm_test
$(GPU_FP8GEMM_TEST): tests/gpu_fp8gemm_test.c tests/bw_warp_sim_k.h src/omega_gpu_fp8gemm_api.h src/omega_blackwell_codegen_gemm.h src/omega_gpu_fp8_api.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -DWS_MAX_THREADS=512 -Itests -o $@ tests/gpu_fp8gemm_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-fp8gemm: $(GPU_FP8GEMM_TEST)
	$(GPU_FP8GEMM_TEST) --host-only
	$(GPU_FP8GEMM_TEST) --sim

# OWN-30B (2026-10-13): host-only audit that every launcher declares peak + 2 registers (the chip hung at peak + 1).
.PHONY: test-gpu-gpr-boundary
GPU_GPR_BOUNDARY_TEST = $(OUT_DIR)/gpu_gpr_boundary_test
$(GPU_GPR_BOUNDARY_TEST): tests/gpu_gpr_boundary_test.c src/omega_blackwell_codegen.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/gpu_gpr_boundary_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-gpr-boundary: $(GPU_GPR_BOUNDARY_TEST)
	$(GPU_GPR_BOUNDARY_TEST)

# OWN-30B T3 (2026-10-13): HMMA arithmetic probe (host model self-checks; chip run under the GB10 lock).
GPU_TC_PROBE_TEST = $(OUT_DIR)/gpu_tc_probe_test
$(GPU_TC_PROBE_TEST): tests/gpu_tc_probe_test.c tests/bw_hmma_model.h src/omega_blackwell_codegen_tc.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_tc_probe_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

GPU_TCGEMM_TEST = $(OUT_DIR)/gpu_tcgemm_test
$(GPU_TCGEMM_TEST): tests/gpu_tcgemm_test.c tests/bw_hmma_model.h tests/bw_warp_sim_tc.h src/omega_blackwell_codegen_tc.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -DWS_MAX_THREADS=512 -Itests -o $@ tests/gpu_tcgemm_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

GPU_TC_PEAK_TEST = $(OUT_DIR)/gpu_tc_peak_test
$(GPU_TC_PEAK_TEST): tests/gpu_tc_peak_test.c tests/bw_hmma_model.h tests/bw_warp_sim_tc.h src/omega_blackwell_codegen_tc.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -DWS_MAX_THREADS=512 -Itests -o $@ tests/gpu_tc_peak_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# OWN-30B L1 attention (2026-10-13): tensor-core causal prefill attention. --host-only and --sim are host proofs;
# the chip gate is tools/run_gpu_prefill_attn_chip.sh.
GPU_PATTN_TEST = $(OUT_DIR)/gpu_prefill_attn_test
$(GPU_PATTN_TEST): tests/gpu_prefill_attn_test.c tests/bw_hmma_model.h tests/bw_warp_sim_pa.h src/omega_gpu_prefill_attn_api.h src/omega_blackwell_codegen_pattn.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -DWS_MAX_THREADS=256 -Itests -o $@ tests/gpu_prefill_attn_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
# OWN-30B T3b (2026-10-13): device-resident prefill module tests (host sim first; chip parts under the GB10 lock)
GPU_PF_MEM_TEST = $(OUT_DIR)/gpu_pf_mem_test
$(GPU_PF_MEM_TEST): tests/gpu_pf_mem_test.c tests/bw_warp_sim_tc.h src/omega_gpu_prefill.h src/omega_blackwell_codegen_pf.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -DWS_MAX_THREADS=512 -Itests -o $@ tests/gpu_pf_mem_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
GPU_PFGEMM_TEST = $(OUT_DIR)/gpu_pfgemm_test
$(GPU_PFGEMM_TEST): tests/gpu_pfgemm_test.c tests/bw_hmma_model.h tests/bw_warp_sim_tc.h src/omega_gpu_prefill.h src/omega_blackwell_codegen_tc2.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -DWS_MAX_THREADS=512 -Itests -o $@ tests/gpu_pfgemm_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
GPU_QMMA_PROBE_TEST = $(OUT_DIR)/gpu_qmma_probe_test
$(GPU_QMMA_PROBE_TEST): tests/gpu_qmma_probe_test.c tests/bw_qmma_model.h src/omega_blackwell_codegen_qmma.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_qmma_probe_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
GPU_QGEMM_TEST = $(OUT_DIR)/gpu_qgemm_test
$(GPU_QGEMM_TEST): tests/gpu_qgemm_test.c tests/bw_qmma_model.h tests/bw_warp_sim_tc.h src/omega_gpu_prefill.h src/omega_blackwell_codegen_qg.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -DWS_MAX_THREADS=1024 -Itests -o $@ tests/gpu_qgemm_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
.PHONY: test-gemm-selectors
test-gemm-selectors: $(GPU_PFGEMM_TEST) $(GPU_QGEMM_TEST) $(GPU_FP8GEMM_TEST)
	python3 tests/gemm_selectors.py $(OUT_DIR)

GPU_PFHOST_TEST = $(OUT_DIR)/gpu_pfhost_test
$(GPU_PFHOST_TEST): tests/gpu_pfhost_test.c tests/bw_warp_sim_tc.h src/omega_gpu_pfhost.h src/omega_blackwell_codegen_pfhost.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -DWS_MAX_THREADS=1024 -Itests -o $@ tests/gpu_pfhost_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
# OWN-30B W-L2 (2026-10-13): command list (many launches, one submission) + decode glue kernels. test-gpu-cmdlist is
# host-only + warp simulator; the chip gate is tools/run_gpu_cmdlist_chip.sh under the GB10 lock.
.PHONY: test-gpu-cmdlist
GPU_CMDLIST_TEST = $(OUT_DIR)/gpu_cmdlist_test
$(GPU_CMDLIST_TEST): tests/gpu_cmdlist_test.c tests/bw_warp_sim_k.h src/omega_gpu_cmdlist.h src/omega_blackwell_codegen_dec.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_cmdlist_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-cmdlist: $(GPU_CMDLIST_TEST)
	./$(GPU_CMDLIST_TEST) --host-only
	./$(GPU_CMDLIST_TEST) --sim

# OWN-30B W-L2a (2026-10-13): one decode token as one command list (omega_gpu_decode_graph). test-gpu-decode-graph is
# host-only; the chip gate is tools/run_gpu_decode_graph_chip.sh under the GB10 lock.
.PHONY: test-gpu-decode-graph
GPU_DECODE_GRAPH_TEST = $(OUT_DIR)/gpu_decode_graph_test
$(GPU_DECODE_GRAPH_TEST): tests/gpu_decode_graph_test.c src/omega_gpu_decode_graph.h src/omega_gpu_cmdlist.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_decode_graph_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-decode-graph: $(GPU_DECODE_GRAPH_TEST)
	./$(GPU_DECODE_GRAPH_TEST) --host-only

# OWN-30B W-L2c (2026-10-14): per-launch fixed cost inside a command list (tests/gpu_launch_cost_test.c, chip only, GB10 lock).
GPU_LAUNCH_COST_TEST = $(OUT_DIR)/gpu_launch_cost_test
$(GPU_LAUNCH_COST_TEST): tests/gpu_launch_cost_test.c src/omega_gpu_fp8_api.h src/omega_gpu_cmdlist.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_launch_cost_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# OWN-30B W-L2d (2026-10-14): L2 reuse of FP8 GEMV weights between two launches of one list (tests/gpu_l2_reuse_test.c, chip only, GB10 lock).
GPU_L2_REUSE_TEST = $(OUT_DIR)/gpu_l2_reuse_test
$(GPU_L2_REUSE_TEST): tests/gpu_l2_reuse_test.c src/omega_gpu_fp8_api.h src/omega_gpu_cmdlist.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_l2_reuse_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# Every host-only check: no GPU, no device opened. The chip gates are listed in README.md.
HOST_TESTS = test-gpu-matmul-api test-gpu-elementwise test-gpu-reconv test-gpu-attention test-gpu-moe-window \
	test-gpu-fp8 test-gpu-decode-attn test-gpu-router test-gpu-bf16-gemv test-gpu-fp8gemm test-gpu-gpr-boundary \
	test-gpu-cmdlist test-gpu-decode-graph test-gpu-code-alloc test-gpu-engine-compile test-gpu-engine \
	test-gpu-engine-mutants test-gpu-engine-backend-compile test-gpu-engine-prod-build test-gpu-engine-sim \
	test-gpu-engine-backend-mutants test-gpu-engine-gb10-compile test-gpu-serving-alloc test-gpu-engine-freestanding \
	test-codegen-freestanding test-gpu-session-platform test-gpu-wait test-gpu-wait-mutants host-tests-m16 \
	test-runtime-api test-runtime-portability test-doctor test-hello-sim test-native-artifact-manifest test-gemm-selectors \
	test-native-artifact zero-cuda-gate verify-vendored test-swap-cli test-om-encoder test-om-cubin $(CARRIED_HOST_TESTS)
test: $(HOST_TESTS)
	@echo "CUDA++ HOST TESTS PASS ($(words $(HOST_TESTS)) targets)"

# The vendored NVIDIA headers match THIRD_PARTY.md's SHA256SUMS; the self-test (flipped byte, extra file) must FAIL.
.PHONY: verify-vendored test-swap-cli
verify-vendored:
	tools/verify-vendored.sh --self-test
	tools/verify-vendored.sh
# `cudapp swap` on a fake machine root: no root rights, no GPU, nothing outside a temp dir touched.
test-swap-cli:
	bash tests/swap/test_swap_cli.sh

# Zero-CUDA gate (tools/zero-cuda-gate.sh): the built programs need no NVIDIA CUDA library and import
# no CUDA symbol. The self-test is the negative control (a stub libcudart must FAIL).
.PHONY: zero-cuda-gate
zero-cuda-gate: $(HELLO_GPU) $(GPU_API_TEST)
	tools/zero-cuda-gate.sh --self-test
	tools/zero-cuda-gate.sh $(HELLO_GPU) $(GPU_API_TEST)
