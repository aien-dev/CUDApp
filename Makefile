# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# CUDA++ build. Plain make + gcc; no CUDA toolkit, no NVIDIA user-space library.
#
#   make            libomega_gpu.a (= libcudapp.a): GPU toolchain + runtime + libraries + driver client
#   make test       every host-only test (no GPU needed)
#   make doctor     is this machine ready? one line per check, each FAIL with its fix (no GPU opened);
#                   with OMC=/abs/pinned/omc it also lists every GPU target row (backend/targets.om)
#   make test-multitarget OMC=...  portable-backend host gate: SPIR-V on CPU drivers, HSACO checks
#   make hello-gpu  run the doctor, then build examples/hello_gpu and run it (on a GB10)
#
# The driver client (nvrm/, m16/) and the NVIDIA open headers (third_party/) are in
# this repository, so PHYSICS_DIR is "." here. Omega is meant to build the same sources
# with PHYSICS_DIR pointing at a pinned CUDA++ checkout (cudapp.lock); that pin does not
# exist yet and Omega main still takes nvrm/m16 from physics-mv (#106, #107).
.DEFAULT_GOAL := all
PHYSICS_DIR ?= .
# The seeded bindings src/cudapp_om_units.c and src/cudapp_freest_units.c are
# AArch64 machine code. A non-AArch64 host links src/cudapp_binding_portable.c
# instead and compiles the migration-pending C encoder and C cubin writer.
# That is the portable host path. It is not a run of the seeded units.
# CUDAPP_ENCODER=diff still requires the seeded binding and is refused here.
# On AArch64, CC stays make's built-in (CC ?= gcc does not replace it), so a
# GB10 `make` invokes the same compiler it did before this portability guard.
CUDAPP_HOST_ARCH := $(shell uname -m)
ifeq ($(filter aarch64 arm64,$(CUDAPP_HOST_ARCH)),)
# GNU make defines CC as cc before this file is read, and ?= does not replace
# that built-in. The documented x86-64 compiler is gcc. An explicit CC= on the
# command line or in the environment still wins.
ifeq ($(origin CC),default)
CC := gcc
endif
CUDAPP_HOST_IS_AARCH64 :=
CUDAPP_HOST_BINDING := src/cudapp_binding_portable.c
CUDAPP_HOST_FREEST_BINDING :=
CUDAPP_FS_BINDING := src/cudapp_binding_portable.c
CUDAPP_BINDING_SRCS := src/cudapp_binding_portable.c
CUDAPP_PORTABLE_DEFS := -DCUDAPP_MIGRATION_PENDING_C_ENCODER -DCUDAPP_MIGRATION_PENDING_C_CUBIN
CUDAPP_TARGET_ONLY_TESTS := test-om-encoder test-om-cubin test-gpu-native-runtime test-om-spec-encoder test-portable-receipts test-model-capacity
ifeq ($(CUDAPP_ENCODER),diff)
$(error CUDAPP_ENCODER=diff executes the AArch64 seeded binding; this host is $(CUDAPP_HOST_ARCH))
endif
$(info CUDA++ host $(CUDAPP_HOST_ARCH): seeded AArch64 binding is not assembled; C encoder and C cubin writer are the portable bodies; other seeded units refuse with status 34)
else
# Same assignment as before this branch. Make's built-in CC=cc wins, because ?=
# does not replace a variable make already defined.
CC ?= gcc
CUDAPP_HOST_IS_AARCH64 := 1
CUDAPP_HOST_BINDING := src/cudapp_om_units.c
CUDAPP_HOST_FREEST_BINDING := src/cudapp_freest_units.c
CUDAPP_FS_BINDING := src/cudapp_freest_units.c
CUDAPP_BINDING_SRCS := src/cudapp_om_units.c src/cudapp_freest_units.c
CUDAPP_PORTABLE_DEFS :=
CUDAPP_TARGET_ONLY_TESTS :=
endif
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
# src/cudapp_om_units.c; c = the migration-pending C encoder, the differential oracle; diff = both on every call,
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
# binding of the seeded Omega units (om/units.sh). src/omega_fp8_slot_table.c is the slot table under
# src/omega_gpu_fp8_slots.c (Expert Ribbon P1, omega #533).
SRCS := $(sort $(wildcard src/omega_blackwell_*.c src/omega_gpu_*.c)) src/omega_fp8_slot_table.c \
	src/omega_cubin.c src/omega_native_artifact.c src/omega_sm121_disasm.c \
	src/sha256.c src/omega_canonical.c src/omega_core.c src/omega_vector.c $(CUDAPP_BINDING_SRCS) \
	$(PHYSICS_DIR)/m16/m16_native.c $(PHYSICS_DIR)/nvrm/nvrm.c
OBJS = $(patsubst %.c,$(OUT_DIR)/%.o,$(notdir $(SRCS)))
DEPS = $(OBJS:.o=.d)
-include $(DEPS)
vpath %.c src $(PHYSICS_DIR)/m16 $(PHYSICS_DIR)/nvrm

$(OUT_DIR):
	mkdir -p $@
$(OUT_DIR)/%.o: %.c | $(OUT_DIR)
	$(CC) $(CFLAGS) $(CUDAPP_ENCODER_DEFS) $(CUDAPP_PORTABLE_DEFS) -c $< -o $@

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
# These two recipes compile omega_cubin.c on the command line, which overrides the
# archive member. On a non-AArch64 host that must be the C cubin writer
# (CUDAPP_PORTABLE_DEFS). On AArch64 the variable is empty and the .om packer stays.
$(ARTIFACT_EXPORT): tools/artifact_export.c src/omega_native_artifact.c src/omega_cubin.c $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) $(CUDAPP_PORTABLE_DEFS) -o $@ tools/artifact_export.c src/omega_native_artifact.c src/omega_cubin.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
$(ARTIFACT_TEST): tests/gpu_artifact_export_test.c src/omega_native_artifact.c src/omega_cubin.c $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) $(CUDAPP_PORTABLE_DEFS) -o $@ tests/gpu_artifact_export_test.c src/omega_native_artifact.c src/omega_cubin.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
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
$(OUT_DIR)/native_artifact_manifest_test: tests/native_artifact_manifest_test.c src/omega_native_artifact.c src/omega_cubin.c src/omega_blackwell_realize.c src/sha256.c $(CUDAPP_HOST_BINDING) | $(OUT_DIR)
	$(CC) $(CFLAGS) $(CUDAPP_PORTABLE_DEFS) -ffunction-sections -fdata-sections -Wl,--gc-sections -Wl,--wrap=fopen -o $@ $^ -lpthread
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
# General kernel (0.6.0): --general --sim runs the predeclared 318-config sweep (head_dim 1..256, MHA/GQA/MQA,
# causal/windows, strides, paged) and the general mutants on the host simulator; the chip run is
# `./build/gpu_attention_test --general --out g.json` under /tmp/aien-gb10.lock.
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
	./$(GPU_ATTN_TEST) --general --sim --out $(OUT_DIR)/gpu_attention_general_sim.json

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
	sh tests/grid_ab_exit_test.sh ./$(GPU_FP8_TEST)
	./$(GPU_FP8_TEST) --host-only
	./$(GPU_FP8_TEST) --sim

# omega #457 (2026-10-15): host rewrite of a GPU-cached buffer must be visible to the next launch. Chip only; the gate
# is tools/run_gpu_l2_stale_chip.sh.
GPU_L2_STALE_TEST = $(OUT_DIR)/gpu_l2_stale_test
$(GPU_L2_STALE_TEST): tests/gpu_l2_stale_test.c src/omega_gpu_fp8_api.h src/omega_gpu_session.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_l2_stale_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# Expert Ribbon P1 (aienix#4 / omega#533, 2026-10-18): versioned replaceable FP8 expert slots (omega_gpu_fp8_slots.h).
# test-gpu-fp8-slots is host-only (slot state machine, threaded race, mutant); the chip gate is omega's
# tools/run_gpu_fp8_slots_chip.sh under the GB10 lock (parity vs resident, churn, reader fence via a command list,
# negative control). That runner reads omega's physics.lock and is not carried here.
.PHONY: test-gpu-fp8-slots
GPU_FP8_SLOTS_TEST = $(OUT_DIR)/gpu_fp8_slots_test
$(GPU_FP8_SLOTS_TEST): tests/gpu_fp8_slots_test.c src/omega_gpu_fp8_slots.h src/omega_fp8_slot_table.h src/omega_gpu_fp8_api.h src/omega_gpu_cmdlist.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_fp8_slots_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-fp8-slots: $(GPU_FP8_SLOTS_TEST)
	./$(GPU_FP8_SLOTS_TEST) --host-only

# OWN-30B cut 8 (2026-10-13): one-launch decode attention (omega_gpu_decode_attn_api.h). test-gpu-decode-attn is
# host-only + simulator; the chip gate is tools/run_gpu_decode_attn_chip.sh.
.PHONY: test-gpu-decode-attn
GPU_DATTN_TEST = $(OUT_DIR)/gpu_decode_attn_test
$(GPU_DATTN_TEST): tests/gpu_decode_attn_test.c tests/bw_warp_sim_k.h src/omega_gpu_decode_attn_api.h src/omega_blackwell_codegen_attn.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_decode_attn_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-decode-attn: $(GPU_DATTN_TEST) test-gpu-decode-attn-maxpos
	$(GPU_DATTN_TEST) --host-only
	$(GPU_DATTN_TEST) --sim

# omega #535: the 128k position bound (OMEGA_GPU_DATTN_MAX_POS 131072) on a fake platform, no device. A sanitized copy
# of src/omega_gpu_decode_attn_api.c is linked ahead of the library, so cache writes, the RoPE fill and launch staging
# are ASan/UBSan checked against the fake's exact-size buffers. Each mutant breaks one bound in a copy of that file;
# the test must FAIL on it.
.PHONY: test-gpu-decode-attn-maxpos
GPU_DATTN_MAXPOS_TEST = $(OUT_DIR)/test_omega_gpu_decode_attn_maxpos
GPU_DATTN_MAXPOS_MUT = $(OUT_DIR)/dattn_maxpos_mut
GPU_DATTN_MAXPOS_SAN = -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all
$(GPU_DATTN_MAXPOS_TEST): tests/test_omega_gpu_decode_attn_maxpos.c src/omega_gpu_decode_attn_api.c src/omega_gpu_decode_attn_api.h src/omega_blackwell_codegen_attn.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(filter-out -MMD -MP -O2,$(CFLAGS)) $(GPU_DATTN_MAXPOS_SAN) -o $@ tests/test_omega_gpu_decode_attn_maxpos.c \
	  src/omega_gpu_decode_attn_api.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-decode-attn-maxpos: $(GPU_DATTN_MAXPOS_TEST)
	$(GPU_DATTN_MAXPOS_TEST)
	@mkdir -p $(GPU_DATTN_MAXPOS_MUT)
	@set -e; \
	tmut() { name=$$1; script=$$2; \
	  sed -e "$$script" src/omega_gpu_decode_attn_api.c > $(GPU_DATTN_MAXPOS_MUT)/$$name.c; \
	  if cmp -s src/omega_gpu_decode_attn_api.c $(GPU_DATTN_MAXPOS_MUT)/$$name.c; then echo "mutant $$name: pattern not found"; exit 1; fi; \
	  $(CC) $(filter-out -MMD -MP -O2,$(CFLAGS)) $(GPU_DATTN_MAXPOS_SAN) -o $(GPU_DATTN_MAXPOS_MUT)/$$name \
	    tests/test_omega_gpu_decode_attn_maxpos.c $(GPU_DATTN_MAXPOS_MUT)/$$name.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm; \
	  if $(GPU_DATTN_MAXPOS_MUT)/$$name > $(GPU_DATTN_MAXPOS_MUT)/$$name.log 2>&1; then echo "mutant $$name SURVIVED"; exit 1; fi; \
	  if ! grep -q '^FAIL ' $(GPU_DATTN_MAXPOS_MUT)/$$name.log; then \
	    echo "mutant $$name: no FAIL case (did not run?)"; cat $(GPU_DATTN_MAXPOS_MUT)/$$name.log; exit 1; fi; \
	  echo "mutant $$name killed"; }; \
	tmut cap-off-by-one 's/max_pos > OMEGA_GPU_DATTN_MAX_POS ||/max_pos >= OMEGA_GPU_DATTN_MAX_POS ||/'; \
	tmut cap-one-chunk-over 's/max_pos > OMEGA_GPU_DATTN_MAX_POS ||/max_pos > OMEGA_GPU_DATTN_MAX_POS + 256u ||/'; \
	tmut write-pos-bound 's/kv_head >= 4 || pos >= c->max_pos || count == 0/kv_head >= 4 || count == 0/'; \
	tmut write-count-bound 's/count > c->max_pos - pos ||/count > c->max_pos - pos + 1u ||/'; \
	tmut read-pos-bound 's/kv_head >= 4 || pos >= c->max_pos || !k/kv_head >= 4 || pos > c->max_pos || !k/'; \
	tmut step-pos-bound 's/layer >= c->layers || pos >= c->max_pos || (rec/layer >= c->layers || pos > c->max_pos || (rec/'; \
	tmut rope-table-short 's/(size_t)max_pos \* OMEGA_DATTN_ROPE_ROW_BYTES/(size_t)(max_pos - 8u) * OMEGA_DATTN_ROPE_ROW_BYTES/'; \
	echo "test-gpu-decode-attn-maxpos: 7 of 7 mutants killed"

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
# Sass#4 (2026-10-27): the generator's I2FP consumer rule (scoreboard or >= 19 cycles), host only, with its negative
# control (mutant 4, the pre-fix stall 6, must be refused). No reference oracle needed, so it lives here.
.PHONY: test-gpu-tern-sched
GPU_TERN_SCHED_TEST = $(OUT_DIR)/gpu_tern_sched_test
$(GPU_TERN_SCHED_TEST): tests/gpu_tern_sched_test.c src/omega_blackwell_codegen_tern.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/gpu_tern_sched_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-tern-sched: $(GPU_TERN_SCHED_TEST)
	$(GPU_TERN_SCHED_TEST)


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
# omega#456: the prefill tests above, host model and warp simulator only. Every run names --host-only or --sim, the two
# modes that never open the device (each test's main returns before its chip part); chip runs stay under the GB10 lock.
.PHONY: test-gpu-prefill-host
test-gpu-prefill-host: $(GPU_TC_PROBE_TEST) $(GPU_TCGEMM_TEST) $(GPU_PATTN_TEST) $(GPU_PF_MEM_TEST) $(GPU_PFGEMM_TEST) \
                       $(GPU_QMMA_PROBE_TEST) $(GPU_QGEMM_TEST) $(GPU_PFHOST_TEST)
	./$(GPU_TC_PROBE_TEST) --host-only
	./$(GPU_QMMA_PROBE_TEST) --host-only
	./$(GPU_PF_MEM_TEST) --host-only
	./$(GPU_TCGEMM_TEST) --host-only
	./$(GPU_TCGEMM_TEST) --sim
	./$(GPU_PATTN_TEST) --host-only
	./$(GPU_PATTN_TEST) --sim
	./$(GPU_PFGEMM_TEST) --host-only
	./$(GPU_PFGEMM_TEST) --sim
	./$(GPU_QGEMM_TEST) --host-only
	./$(GPU_QGEMM_TEST) --sim
	./$(GPU_PFHOST_TEST) --host-only
	./$(GPU_PFHOST_TEST) --sim
	@echo "test-gpu-prefill-host: PASS (8 prefill tests, host model + warp simulator, no device)"
# omega#456: tools/own30b_decode_profile.c is a link-only recorder built by hand on the Spark; compile it with the
# project's -Werror flags so a warning cannot land unseen.
.PHONY: check-own30b-decode-profile
check-own30b-decode-profile: tools/own30b_decode_profile.c
	@mkdir -p $(OUT_DIR)
	$(CC) $(CFLAGS) -c -o $(OUT_DIR)/own30b_decode_profile.o tools/own30b_decode_profile.c
	@echo "check-own30b-decode-profile: PASS (compiles with -Werror)"
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
GPU_DGRAPH_ABORT_TEST = $(OUT_DIR)/test_omega_gpu_decode_graph_abort
$(GPU_DECODE_GRAPH_TEST): tests/gpu_decode_graph_test.c src/omega_gpu_decode_graph.h src/omega_gpu_cmdlist.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_decode_graph_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-decode-graph: $(GPU_DECODE_GRAPH_TEST) $(GPU_DGRAPH_ABORT_TEST) test-gpu-decode-graph-abort-mutants test-gpu-decode-graph-timeline
	./$(GPU_DECODE_GRAPH_TEST) --host-only
	./$(GPU_DGRAPH_ABORT_TEST)

# omega #518: lock and token-state recovery of the split API (failed begin_token, abort_token, end_token after a failed
# record, latched session) on a fake platform whose lock refuses a second acquisition. Host only, opens no device.
# Each mutant breaks one rule in a copy of omega_gpu_decode_graph.c linked ahead of the library; the test must FAIL on it.
.PHONY: test-gpu-decode-graph-abort-mutants
GPU_DGRAPH_MUT_DIR = $(OUT_DIR)/dgraph_mut
$(GPU_DGRAPH_ABORT_TEST): tests/test_omega_gpu_decode_graph_abort.c src/omega_gpu_decode_graph.h src/omega_gpu_platform.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/test_omega_gpu_decode_graph_abort.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-decode-graph-abort-mutants: $(OUT_DIR)/libomega_gpu.a
	@mkdir -p $(GPU_DGRAPH_MUT_DIR)
	@set -e; killed=0; \
	dmut() { name=$$1; script=$$2; \
	  sed -e "$$script" src/omega_gpu_decode_graph.c > $(GPU_DGRAPH_MUT_DIR)/$$name.c; \
	  if cmp -s src/omega_gpu_decode_graph.c $(GPU_DGRAPH_MUT_DIR)/$$name.c; then echo "mutant $$name: pattern not found"; exit 1; fi; \
	  $(CC) $(filter-out -MMD -MP,$(CFLAGS)) -c -o $(GPU_DGRAPH_MUT_DIR)/$$name.o $(GPU_DGRAPH_MUT_DIR)/$$name.c; \
	  $(CC) $(filter-out -MMD -MP,$(CFLAGS)) -o $(GPU_DGRAPH_MUT_DIR)/$$name tests/test_omega_gpu_decode_graph_abort.c \
	    $(GPU_DGRAPH_MUT_DIR)/$$name.o $(OUT_DIR)/libomega_gpu.a -lpthread -lm; \
	  if ./$(GPU_DGRAPH_MUT_DIR)/$$name > $(GPU_DGRAPH_MUT_DIR)/$$name.log 2>&1; then echo "mutant $$name SURVIVED"; exit 1; fi; \
	  nfail=$$(grep -c '^FAIL' $(GPU_DGRAPH_MUT_DIR)/$$name.log || true); \
	  if [ "$$nfail" -eq 0 ]; then echo "mutant $$name: nonzero exit but no FAIL case (did not run?)"; cat $(GPU_DGRAPH_MUT_DIR)/$$name.log; exit 1; fi; \
	  echo "mutant $$name killed: $$nfail case(s) failed"; killed=$$((killed+1)); }; \
	dmut begin-fail-keeps-lock 's|    omega_gpu_session_unlock(); /\* begin_fail releases \*/|    ;|'; \
	dmut begin-fail-stays-in-token 's|    g->in_token = 0; g->failed = 1;|    g->failed = 1;|'; \
	dmut oneshot-old-bad-args-only 's|if (brc != OMEGA_GPU_DGRAPH_OK) {|if (brc == OMEGA_GPU_DGRAPH_BAD_ARGS) {|'; \
	dmut abort-keeps-lock 's|g->n_stamp = 0; /\* no timeline for an abandoned token \*/|&\n    return OMEGA_GPU_DGRAPH_OK;|'; \
	dmut abort-submits 's|^int omega_gpu_decode_graph_abort_token(OmegaGpuDecodeGraph \*g) {$$|&\n    if (g \&\& g->in_token) return omega_gpu_decode_graph_end_token(g, NULL, NULL);|'; \
	echo "test-gpu-decode-graph-abort-mutants: $$killed of 5 mutants killed"

# omega #456: the per-token timeline (set_stamps) never writes past stamp[]: 4096 stamped launches fill it, the 4097th
# (flush_each, one submission per launch) fails the token and leaves no timeline. Fake platform that completes every
# submission; no device. Each mutant breaks one rule in a copy of omega_gpu_decode_graph.c linked ahead of the library;
# the overflow case must FAIL on it.
.PHONY: test-gpu-decode-graph-timeline
GPU_DGRAPH_TL_TEST = $(OUT_DIR)/test_omega_gpu_decode_graph_timeline
GPU_DGRAPH_TL_MUT = $(OUT_DIR)/dgraph_tl_mut
$(GPU_DGRAPH_TL_TEST): tests/test_omega_gpu_decode_graph_timeline.c src/omega_gpu_decode_graph.h src/omega_gpu_platform.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/test_omega_gpu_decode_graph_timeline.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-decode-graph-timeline: $(GPU_DGRAPH_TL_TEST)
	./$(GPU_DGRAPH_TL_TEST)
	@mkdir -p $(GPU_DGRAPH_TL_MUT)
	@set -e; \
	tmut() { name=$$1; script=$$2; \
	  sed -e "$$script" src/omega_gpu_decode_graph.c > $(GPU_DGRAPH_TL_MUT)/$$name.c; \
	  if cmp -s src/omega_gpu_decode_graph.c $(GPU_DGRAPH_TL_MUT)/$$name.c; then echo "mutant $$name: pattern not found"; exit 1; fi; \
	  $(CC) $(filter-out -MMD -MP,$(CFLAGS)) -o $(GPU_DGRAPH_TL_MUT)/$$name tests/test_omega_gpu_decode_graph_timeline.c \
	    $(GPU_DGRAPH_TL_MUT)/$$name.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm; \
	  if ./$(GPU_DGRAPH_TL_MUT)/$$name > $(GPU_DGRAPH_TL_MUT)/$$name.log 2>&1; then echo "mutant $$name SURVIVED"; exit 1; fi; \
	  if ! grep -q '^FAIL one launch past the timeline' $(GPU_DGRAPH_TL_MUT)/$$name.log; then \
	    echo "mutant $$name: no FAIL case (did not run?)"; cat $(GPU_DGRAPH_TL_MUT)/$$name.log; exit 1; fi; \
	  echo "mutant $$name killed"; }; \
	tmut room-off-by-one 's|(uint32_t)(cnt - from) > room) {|(uint32_t)(cnt - from) > room + 1u) {|'; \
	tmut timeline-after-failure 's#if (!g->stamps || g->failed || #if (!g->stamps || #'; \
	echo "test-gpu-decode-graph-timeline: 2 of 2 mutants killed"

# OWN-30B W-L2c (2026-10-14): per-launch fixed cost inside a command list (tests/gpu_launch_cost_test.c, chip only, GB10 lock).
GPU_LAUNCH_COST_TEST = $(OUT_DIR)/gpu_launch_cost_test
$(GPU_LAUNCH_COST_TEST): tests/gpu_launch_cost_test.c src/omega_gpu_fp8_api.h src/omega_gpu_cmdlist.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_launch_cost_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# OWN-30B W-L2d (2026-10-14): L2 reuse of FP8 GEMV weights between two launches of one list (tests/gpu_l2_reuse_test.c, chip only, GB10 lock).
GPU_L2_REUSE_TEST = $(OUT_DIR)/gpu_l2_reuse_test
$(GPU_L2_REUSE_TEST): tests/gpu_l2_reuse_test.c src/omega_gpu_fp8_api.h src/omega_gpu_cmdlist.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ tests/gpu_l2_reuse_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

# The Blackwell QMD shared-memory field (word 36): the encoding of every size, the fixed SM-config
# fields beside it, and the refusal one byte over OMEGA_BW_QMD_MAX_SHARED_BYTES (a host check, no GPU).
.PHONY: test-gpu-qmd-shared test-gpu-qmd-shared-clean-build
QMD_SHARED_TEST = $(OUT_DIR)/test_omega_bw_qmd_shared
$(QMD_SHARED_TEST): tests/test_omega_bw_qmd_shared.c src/omega_blackwell_qmd.h src/omega_blackwell_qmd.c | $(OUT_DIR)
	$(CC) $(CFLAGS) -o $@ tests/test_omega_bw_qmd_shared.c src/omega_blackwell_qmd.c
test-gpu-qmd-shared: $(QMD_SHARED_TEST)
	$(QMD_SHARED_TEST)

# Build the same host-only test with an output directory that does not exist yet.
# The normal test target does not recurse through this regression target.
test-gpu-qmd-shared-clean-build:
	+@set -eu; out=$$(mktemp -d); trap 'rm -rf "$$out"' EXIT; \
		$(MAKE) --no-print-directory OUT_DIR="$$out/new" test-gpu-qmd-shared

# Fresh-checkout contract for issue 15. The slice runner and the barrier contract
# need no sibling compiler. The architecture guard runs on x86-64. On AArch64
# the target prints NOT_RUN and does not build the binary, so this check never
# asks the production path to open a device. test-omega-slices is the
# eight-slice gate and is not part of `make test`.
.PHONY: test-omega-slice-runner test-omega-arch-guard test-omega-barrier-contract test-omega-binding-portable test-omega-target-only test-omega-slices
ARCH_GUARD_TEST = $(OUT_DIR)/omega_blackwell_arch_guard_test
$(ARCH_GUARD_TEST): tests/test_omega_blackwell_arch_guard.c tests/fake_m16_native.c tests/fake_m16_native.h \
		$(GPU_ENGINE_SIM_SRCS) src/omega_gpu_wait.c | $(OUT_DIR)
	$(CC) -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -pthread -ffunction-sections -fdata-sections \
		$(CUDAPP_PORTABLE_DEFS) -Isrc -Itests $(GPU_ENGINE_PHYS_INC) -o $@ tests/test_omega_blackwell_arch_guard.c tests/fake_m16_native.c \
		$(GPU_ENGINE_SIM_SRCS) src/omega_gpu_wait.c -Wl,--gc-sections -lm
ifeq ($(CUDAPP_HOST_IS_AARCH64),1)
test-omega-arch-guard:
	OUT_DIR=$(OUT_DIR) sh tests/omega/test-arch-guard.sh
else
test-omega-arch-guard: $(ARCH_GUARD_TEST)
	OUT_DIR=$(OUT_DIR) sh tests/omega/test-arch-guard.sh $(ARCH_GUARD_TEST)
endif
test-omega-slice-runner:
	sh tests/omega/test-runner.sh
test-omega-barrier-contract:
	sh tests/omega/test-barrier-contract.sh
test-omega-binding-portable:
	CC=$(CC) OUT_DIR=$(OUT_DIR) sh tests/omega/test-binding-portable.sh
test-omega-target-only:
	sh tests/omega/test-target-only.sh
test-omega-slices:
	sh tests/omega/run.sh $(or $(OMEGA_SLICE_FUZZ),120)

# Every host-only check: no GPU, no device opened. The chip gates are listed in README.md.
# Portable commands and the AArch64-only rows are separated in docs/OMEGA_HOST_TESTS.md.
HOST_TESTS = test-drift test-gpu-matmul-api test-gpu-elementwise test-gpu-reconv test-gpu-attention test-gpu-moe-window \
	test-gpu-fp8 test-gpu-decode-attn test-gpu-router test-gpu-bf16-gemv test-gpu-fp8gemm test-gpu-tern-sched test-gpu-gpr-boundary \
	test-gpu-cmdlist test-gpu-decode-graph test-gpu-code-alloc test-gpu-engine-compile test-gpu-engine \
	test-gpu-engine-mutants test-gpu-engine-backend-compile test-gpu-engine-prod-build test-gpu-engine-sim \
	test-gpu-engine-backend-mutants test-gpu-engine-gb10-compile test-gpu-serving-alloc test-gpu-engine-freestanding \
	test-codegen-freestanding test-gpu-session-platform test-gpu-native-runtime test-gpu-wait test-gpu-wait-mutants host-tests-m16 \
	test-runtime-api test-runtime-portability test-doctor test-hello-sim test-native-artifact-manifest test-gemm-selectors \
	test-native-artifact zero-cuda-gate verify-vendored test-swap-cli test-om-encoder test-om-spec-encoder test-om-cubin \
	test-portable-receipts test-gpu-qmd-shared test-model-capacity test-gpu-qmd-shared-clean-build \
	test-omega-slice-runner test-omega-arch-guard test-omega-barrier-contract test-omega-binding-portable \
	test-gpu-imadhi-rc test-dagsim test-costmodel test-gpu-fp8-slots test-bw-hazard test-mufu-ftz $(CARRIED_HOST_TESTS)
ifneq ($(CUDAPP_TARGET_ONLY_TESTS),)
HOST_TESTS := $(filter-out $(CUDAPP_TARGET_ONLY_TESTS),$(HOST_TESTS)) test-omega-target-only
endif
test: $(HOST_TESTS)
	@echo "CUDA++ HOST TESTS PASS ($(words $(HOST_TESTS)) targets)"

.PHONY: test-model-capacity
ifeq ($(CUDAPP_HOST_IS_AARCH64),1)
test-model-capacity:
	OMEGA_OMC="$(OMEGA_OMC)" sh tests/models/capacity.sh
else
test-model-capacity:
	@echo "test-model-capacity: NOT_RUN (requires the pinned omc; not part of the portable host suite)"
	@exit 2
endif

.PHONY: test-drift
test-drift:
	bash tools/extract/test-drift.sh

# Original Git blobs are read-only verification oracles, not a replacement receipt authority.
.PHONY: test-portable-receipts
ifeq ($(CUDAPP_HOST_IS_AARCH64),1)
test-portable-receipts: test-om-units tests/portable_receipts.om tests/portable_receipts.tsv om/cache/digest.om
	@mkdir -p build/portable_receipts
	cat om/cache/digest.om tests/portable_receipts.om > build/portable_receipts.source.om
	"$(OMEGA_OMC)" < build/portable_receipts.source.om > build/portable_receipts_check
	chmod +x build/portable_receipts_check
	@set -eu; tab=$$(printf '\t'); \
	while IFS="$$tab" read -r path blob derived_sha; do \
		test -n "$$path" && test -n "$$blob" && test -n "$$derived_sha"; \
		printf '%s  %s\n' "$$derived_sha" "$$path" | sha256sum -c -; \
		GIT_NO_REPLACE_OBJECTS=1 git cat-file blob "$$blob" > build/portable_receipts/original; \
		./build/portable_receipts_check build/portable_receipts/original "$$path" "$$blob"; \
	done < tests/portable_receipts.tsv
else
test-portable-receipts:
	@echo "test-portable-receipts: NOT_RUN (requires the pinned omc; not part of the portable host suite)"
	@exit 2
endif

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

# omega #492: IMAD.HI.U32 with a register Rc. The chip reads Rc as a 64-bit register pair (nvcc 13.0
# sm_121 oracle, evidence/IMADHI-RC-ORACLE-nvcc13.0-20261016; chip probe pending), so the four warp
# simulators model an even Rc as the pair Rc+1:Rc and refuse an odd Rc, and replay the PR #487
# odd-column pack (it must come out wrong, as on the chip). One binary per simulator header.
.PHONY: test-gpu-imadhi-rc
GPU_IMADHI_SIMS = bw_warp_sim bw_warp_sim_tc bw_warp_sim_k bw_warp_sim_pa
GPU_IMADHI_TESTS = $(GPU_IMADHI_SIMS:%=$(OUT_DIR)/gpu_imadhi_rc_test_%)
$(OUT_DIR)/gpu_imadhi_rc_test_%: tests/gpu_imadhi_rc_test.c tests/%.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -Itests -DSIM_HEADER='"$*.h"' -o $@ tests/gpu_imadhi_rc_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-gpu-imadhi-rc: $(GPU_IMADHI_TESTS)
	@for t in $(GPU_IMADHI_TESTS); do ./$$t || exit 1; done

# omega-mv#554: static scoreboard hazard checker (RAW / WAR / WAW) over the encoded bytes of every kernel the generators
# build, after the launchers' patch steps. Host only: decoder vs encoder, the pre-#553 attend red receipt
# (tests/fixtures/bw_hazard), negative controls, then the sweep. test-bw-hazard-nvdisasm (the name it had when it
# called nvdisasm) adds the decoder cross-check against Omega's own sm_121 disassembler, in process, on every distinct
# kernel (bar C6: every check that called nvdisasm calls ours; test-sm121dis checks that disassembler against the oracle).
.PHONY: test-bw-hazard test-bw-hazard-nvdisasm
BW_HAZARD_TEST = $(OUT_DIR)/bw_hazard_test
$(BW_HAZARD_TEST): tests/bw_hazard_test.c src/omega_bw_hazard.c src/omega_bw_hazard.h src/omega_sm121_disasm.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/bw_hazard_test.c src/omega_bw_hazard.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
test-bw-hazard: $(BW_HAZARD_TEST)
	$(BW_HAZARD_TEST)
test-bw-hazard-nvdisasm: $(BW_HAZARD_TEST)
	$(BW_HAZARD_TEST) --disasm

# Offline resource-DAG evaluator, imported verbatim from closed PR #590
# (commit 3bd56731). Host arithmetic on the numbers written in the fixtures.
# Not a validated cost model and not a hardware measurement.
# src/omega_om_units.c is not in this tree. src/sha256.c is sha256_hash on every host.
# tests/dagsim/sha256_host.c is omega's non-AArch64 stub and aborts if a digest is requested.
# dagsim.o keeps the #590 (3bd56731) nm guard: no generation store, authority,
# world, or a call that maps or runs code.
.PHONY: test-dagsim
DS_OBJ = $(OUT_DIR)/dagsim.o
DS_UNIT = $(OUT_DIR)/dagsim_unit
DS_CLI = $(OUT_DIR)/dagsim
DS_SHA_SRC = src/sha256.c
RX_CM_OBJ = $(OUT_DIR)/rx_costmodel.o

$(RX_CM_OBJ): src/runtime/rx_costmodel.c src/runtime/rx_costmodel.h | $(OUT_DIR)
	$(CC) $(CFLAGS) -c -o $@ src/runtime/rx_costmodel.c
	@if nm -u $@ | grep -E 'rx_gen_|aienos_cap_|rx_caproot_|rx_capadmin_|rx_world_|mmap|mprotect|fork|exec|dlopen|system' ; then \
		echo "rx_costmodel.o references an operation a cost model must not have"; \
		rm -f $@; exit 1; fi

# omega OMEGA_EMPIRICAL_OPTIMIZER cost model, any host: synthetic measurements; fit, calibration, decisions, blob.
# The restore-op-clamp mutant (omega #934) must be killed by unsupported-op-not-aliased-to-matvec.
.PHONY: test-costmodel
RX_CM_UNIT = $(OUT_DIR)/rx_costmodel_unit
$(RX_CM_UNIT): tests/runtime/rx_costmodel_unit.c $(RX_CM_OBJ) $(DS_SHA_SRC) | $(OUT_DIR)
	$(CC) $(CFLAGS) -o $@ tests/runtime/rx_costmodel_unit.c $(RX_CM_OBJ) $(DS_SHA_SRC) -lm

test-costmodel: $(RX_CM_UNIT)
	./$(RX_CM_UNIT)
	@set -eu; dir=$(OUT_DIR)/costmodel-mutant; rm -rf $$dir; mkdir -p $$dir; \
	  sed "/CM704:refuse-unsupported-op/c\\    uint32_t op = f && f->op < RX_CM_OPS ? f->op : RX_CM_OPS - 1u; /* CM704:refuse-unsupported-op */" \
	    src/runtime/rx_costmodel.c > $$dir/rx_costmodel.c; \
	  if cmp -s src/runtime/rx_costmodel.c $$dir/rx_costmodel.c; then \
	    echo "mutant restore-op-clamp: sed changed nothing"; exit 1; fi; \
	  $(CC) $(CFLAGS) -c -o $$dir/rx_costmodel.o $$dir/rx_costmodel.c; \
	  $(CC) $(CFLAGS) -o $$dir/unit tests/runtime/rx_costmodel_unit.c $$dir/rx_costmodel.o $(DS_SHA_SRC) -lm; \
	  rc=0; $$dir/unit refuse > $$dir/out.txt 2> $$dir/err.txt || rc=$$?; \
	  if [ $$rc -eq 0 ] || ! grep -q unsupported-op-not-aliased-to-matvec $$dir/err.txt; then \
	    echo "mutant restore-op-clamp was not killed by unsupported-op-not-aliased-to-matvec (exit $$rc)"; \
	    cat $$dir/err.txt; exit 1; fi; \
	  echo "mutant restore-op-clamp: killed by unsupported-op-not-aliased-to-matvec (exit $$rc)"

$(DS_OBJ): tools/dagsim/dagsim.c tools/dagsim/dagsim.h src/runtime/rx_costmodel.h | $(OUT_DIR)
	$(CC) $(CFLAGS) -Itools/dagsim -c -o $@ tools/dagsim/dagsim.c
	@if nm -u $@ | grep -E 'rx_gen_|aienos_cap_|rx_caproot_|rx_capadmin_|rx_world_|mmap|mprotect|fork|exec|dlopen|system' ; then \
		echo "dagsim.o references an operation an offline evaluator must not have"; \
		rm -f $@; exit 1; fi

$(DS_UNIT): tests/dagsim/dagsim_unit.c $(DS_OBJ) $(RX_CM_OBJ) $(DS_SHA_SRC) | $(OUT_DIR)
	$(CC) $(CFLAGS) -Itools/dagsim -o $@ tests/dagsim/dagsim_unit.c $(DS_OBJ) $(RX_CM_OBJ) $(DS_SHA_SRC) -Wl,--wrap=rx_cm_predict -lm

$(DS_CLI): tools/dagsim/dagsim_cli.c $(DS_OBJ) $(RX_CM_OBJ) $(DS_SHA_SRC) | $(OUT_DIR)
	$(CC) $(CFLAGS) -Itools/dagsim -o $@ tools/dagsim/dagsim_cli.c $(DS_OBJ) $(RX_CM_OBJ) $(DS_SHA_SRC) -lm

test-dagsim: $(DS_UNIT) $(DS_CLI)
	./$(DS_UNIT)
	sh tests/dagsim/run_cli.sh $(DS_CLI) $(OUT_DIR)/dagsim-cli-out
	sh tests/dagsim/mutant_lifetime.sh
	sh tests/dagsim/mutant_predict.sh
