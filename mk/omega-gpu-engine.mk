# Omega GPU Engine (docs/numeric/OMEGA_GPU_ENGINE.md). Picked up by
# `-include mk/*.mk`. Host only: opens no device; needs no physics headers except the two A3b1 targets, not
# part of `all` or `test`.
#   test-gpu-engine-compile  header + core compile, enum names, refusals, no-op backend (A2/A3a)
#   test-gpu-engine          the engine core against a fake backend (every state and failure)
#   test-gpu-engine-backend-compile  syntax-checks src/omega_blackwell_engine.c (A3b1): -c only, no link, no device
#   test-gpu-engine-prod-build       production build of build/omegatool at the physics pin: links only, opens no device
#   test-gpu-engine-mutants  removes each protection from src/omega_gpu_engine.c in a scratch
#                            copy; the host test must fail each one (exit nonzero on a survivor)
#   test-gpu-engine-sim      (A3b2) the REAL engine core + Blackwell backend + vector wrapper against a simulated driver
#                            (tests/fake_m16_native.c); host only, no device, no physics C sources
#   test-gpu-engine-backend-mutants  (A3b2) one mutant per protection in the backend and the vector wrapper
#   test-gpu-engine-gb10-compile     (A3b2) compile-only check of tests/test_omega_gpu_engine_gb10.c (Phase B chip test, never run here)
.PHONY: test-gpu-engine-compile test-gpu-engine test-gpu-engine-mutants test-gpu-engine-backend-compile test-gpu-engine-prod-build test-gpu-engine-sim test-gpu-engine-backend-mutants test-gpu-engine-gb10-compile
build/test_omega_gpu_engine_compile: tests/test_omega_gpu_engine_compile.c src/omega_gpu_engine.c src/omega_gpu_engine.h
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -DOMEGA_NUMERIC_CPU_ONLY -o $@ tests/test_omega_gpu_engine_compile.c src/omega_gpu_engine.c
test-gpu-engine-compile: build/test_omega_gpu_engine_compile
	./build/test_omega_gpu_engine_compile

build/test_omega_gpu_engine: tests/test_omega_gpu_engine.c src/omega_gpu_engine.c src/omega_gpu_engine.h
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -DOMEGA_NUMERIC_CPU_ONLY -o $@ tests/test_omega_gpu_engine.c src/omega_gpu_engine.c
test-gpu-engine: build/test_omega_gpu_engine
	./build/test_omega_gpu_engine

test-gpu-engine-mutants:
	sh tools/gpu_engine_mutations.sh

# A3b1: the Blackwell backend includes physics headers (m16_native.h, nvrm.h), so it
# uses the same physics -I paths as the top-level Makefile CFLAGS (Makefile lines 9-14).
GPU_ENGINE_PHYS_INC = -I$(PHYSICS_DIR)/m16 -I$(PHYSICS_DIR)/nvrm \
	-I$(PHYSICS_DIR)/third_party/nvidia-open-580.173.02/src/common/sdk/nvidia/inc \
	-I$(PHYSICS_DIR)/third_party/nvidia-open-580.173.02/kernel-open/common/inc \
	-I$(PHYSICS_DIR)/third_party/nvidia-open-580.173.02/kernel-open/nvidia-uvm \
	-I$(PHYSICS_DIR)/third_party/nvidia-open-580.173.02/src/nvidia/arch/nvalloc/unix/include

build/omega_blackwell_engine_syntax.o: src/omega_blackwell_engine.c src/omega_blackwell_engine.h src/omega_gpu_engine.h
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -Isrc $(GPU_ENGINE_PHYS_INC) -c -o $@ src/omega_blackwell_engine.c
test-gpu-engine-backend-compile: build/omega_blackwell_engine_syntax.o

# Links only, opens no device. Uses the real m16_native and nvrm sources at the physics pin.
test-gpu-engine-prod-build:
	$(MAKE) build/omegatool

# A3b2: simulated-driver host test. Links the sources of the Makefile RX_COMPOSE_GPU_SRCS list (plus the
# three core files the vector builder needs) with tests/fake_m16_native.c in place of the physics
# m16_native.c and nvrm.c. Opens no device.
GPU_ENGINE_SIM_SRCS = src/omega_blackwell_submit.c src/omega_blackwell_engine.c src/omega_gpu_engine.c \
	src/omega_blackwell_matmul.c src/omega_blackwell_codegen.c src/omega_blackwell_encoder.c \
	src/omega_blackwell_qmd.c src/omega_blackwell_realize.c src/omega_vector.c src/omega_validate.c \
	src/omega_core.c src/omega_canonical.c src/sha256.c

build/test_omega_blackwell_engine: tests/test_omega_blackwell_engine.c tests/fake_m16_native.c tests/fake_m16_native.h \
	$(GPU_ENGINE_SIM_SRCS) src/omega_gpu_engine.h src/omega_blackwell_engine.h
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -pthread -ffunction-sections -fdata-sections \
		-Isrc -Itests $(GPU_ENGINE_PHYS_INC) -o $@ tests/test_omega_blackwell_engine.c tests/fake_m16_native.c \
		$(GPU_ENGINE_SIM_SRCS) -Wl,--gc-sections -lm
test-gpu-engine-sim: build/test_omega_blackwell_engine
	./build/test_omega_blackwell_engine

# One mutant per backend and vector-wrapper protection; the sim test must fail each one.
test-gpu-engine-backend-mutants:
	PHYSICS_DIR="$(PHYSICS_DIR)" sh tools/gpu_engine_backend_mutations.sh

# Phase B chip test, compile only here. Never run in Phase A.
build/test_omega_gpu_engine_gb10.o: tests/test_omega_gpu_engine_gb10.c src/omega_gpu_engine.h src/omega_blackwell_engine.h src/omega_blackwell_submit.h
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math -pthread -Isrc $(GPU_ENGINE_PHYS_INC) -c -o $@ tests/test_omega_gpu_engine_gb10.c
test-gpu-engine-gb10-compile: build/test_omega_gpu_engine_gb10.o

# Measurement (sc#277): driver allocations and frees made by the REAL session + matmul + attention
# APIs while serving a Qwen3-4B-shaped sequence, against the simulated driver. Host only, no device.
# Prints MEASURE lines; asserts only bookkeeping. Links no physics C sources, only its headers.
GPU_SERVE_ALLOC_SRCS = src/omega_gpu_session.c src/omega_gpu_matmul_api.c src/omega_gpu_attention_api.c \
	src/omega_gpu_wait.c src/omega_blackwell_codegen.c src/omega_blackwell_encoder.c src/omega_blackwell_qmd.c \
	src/omega_blackwell_matmul.c src/sha256.c
build/gpu_serving_alloc_count_test: tests/gpu_serving_alloc_count_test.c tests/fake_m16_native.c tests/fake_m16_native.h $(GPU_SERVE_ALLOC_SRCS)
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -pthread -ffp-contract=off -Isrc -Itests $(GPU_ENGINE_PHYS_INC) \
		-o $@ tests/gpu_serving_alloc_count_test.c tests/fake_m16_native.c $(GPU_SERVE_ALLOC_SRCS) -lm
.PHONY: test-gpu-serving-alloc
test-gpu-serving-alloc: build/gpu_serving_alloc_count_test
	./build/gpu_serving_alloc_count_test
