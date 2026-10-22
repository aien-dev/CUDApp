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
#   gpu-engine-freestanding          (native v1 P07 cut A) the portable GPU files built -ffreestanding with
#                                    OMEGA_GPU_NO_POSIX: no Linux header, no errno, no clock, no libc beyond mem*/str*
#   test-codegen-freestanding        (cut A.2) codegen hosted vs freestanding byte-identical, bounded code buffers
#   test-gpu-engine-freestanding     the engine contract test and the platform contract test linked against
#                                    those freestanding objects (host runner, opens no device)
.PHONY: test-gpu-engine-compile test-gpu-engine test-gpu-engine-mutants test-gpu-engine-backend-compile test-gpu-engine-prod-build test-gpu-engine-sim test-gpu-engine-backend-mutants test-gpu-engine-gb10-compile gpu-engine-freestanding test-gpu-engine-freestanding
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
GPU_SERVE_ALLOC_SRCS = src/omega_gpu_session.c src/omega_gpu_platform.c src/omega_gpu_platform_linux.c src/omega_gpu_fmt.c src/omega_gpu_matmul_api.c src/omega_gpu_attention_api.c src/omega_gpu_serving.c \
	src/omega_gpu_wait.c src/omega_blackwell_codegen.c src/omega_blackwell_encoder.c src/omega_blackwell_qmd.c \
	src/omega_blackwell_matmul.c src/sha256.c
build/gpu_serving_alloc_count_test: tests/gpu_serving_alloc_count_test.c tests/fake_m16_native.c tests/fake_m16_native.h $(GPU_SERVE_ALLOC_SRCS)
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -pthread -ffp-contract=off -Isrc -Itests $(GPU_ENGINE_PHYS_INC) \
		-o $@ tests/gpu_serving_alloc_count_test.c tests/fake_m16_native.c $(GPU_SERVE_ALLOC_SRCS) -lm
.PHONY: test-gpu-serving-alloc
test-gpu-serving-alloc: build/gpu_serving_alloc_count_test
	./build/gpu_serving_alloc_count_test plain
	./build/gpu_serving_alloc_count_test reserve
	./build/gpu_serving_alloc_count_test bounds
	./build/gpu_serving_alloc_count_test bounds-block
	./build/gpu_serving_alloc_count_test rollback
	./build/gpu_serving_alloc_count_test sweep
	./build/gpu_serving_alloc_count_test pin
	./build/gpu_serving_alloc_count_test seal-bare
	./build/gpu_serving_alloc_count_test update
	@if ./build/gpu_serving_alloc_count_test sweep-skip >build/gpu_serving_alloc_count_sweep_skip.log 2>&1; then echo "RED RUN PASSED: zero allocations over the row sweep do not need prepare + seal"; exit 1; fi
	@if [ "$$(grep -c "^FAIL" build/gpu_serving_alloc_count_sweep_skip.log)" -lt 1 ]; then echo "sweep-skip exited nonzero without a FAIL check (crash, not the red assertion)"; exit 1; fi
	@grep -E "^MEASURE sweep" build/gpu_serving_alloc_count_sweep_skip.log | sed "s/^/sweep-skip (control): /"
	@if ./build/gpu_serving_alloc_count_test reserve-skip >build/gpu_serving_alloc_count_skip.log 2>&1; then echo "RED RUN PASSED: the zero-allocation assertion does not need the reservation call"; exit 1; fi
	@if [ "$$(grep -c "^FAIL" build/gpu_serving_alloc_count_skip.log)" -lt 1 ]; then echo "reserve-skip exited nonzero without a FAIL check (crash, not the red assertion)"; exit 1; fi
	@echo "reserve-skip (no reservation call): fails as required ($$(grep -c "^FAIL" build/gpu_serving_alloc_count_skip.log) failing checks)"

# native v1 P07 cut A (aienos#286 cut A, aien-architecture#190): the freestanding link target.
# Each file below is compiled with no hosted headers available (-ffreestanding, -fno-builtin,
# OMEGA_GPU_NO_POSIX) and the partial link of all of them may leave only the mem*/str* symbols
# undefined. Any other undefined symbol (errno, clock_gettime, printf, malloc, pthread_*, nvrm_*,
# m16_*) fails the target. This proves the first-compute PATH links without Linux headers or
# runtime symbols; it proves nothing about device ownership (NOT_RUN until cut B runs on the Spark).
GPU_FS_DIR = build/gpu_fs
GPU_FS_CFLAGS = -std=gnu11 -O2 -Wall -Wextra -Werror -ffreestanding -fno-builtin -fno-stack-protector \
	-U_FORTIFY_SOURCE -DOMEGA_GPU_NO_POSIX -DOMEGA_NUMERIC_CPU_ONLY -Isrc
# P07 cut C (session part): the persistent session, the matmul and attention APIs, the serving reservation and the
# bounded wait join the link. Their machine comes from an installed OmegaGpuPlatform; the Linux reference
# (src/omega_gpu_platform_linux.c) is deliberately NOT in this list. Still outside it: the elementwise API, the
# numeric launchers, the vector executor, the resident runtime (docs/numeric/OMEGA_GPU_NATIVE_BOUNDARY.md).
GPU_FS_SRCS = src/omega_gpu_engine.c src/omega_gpu_platform.c src/omega_blackwell_encoder.c src/omega_blackwell_codegen.c \
	src/omega_blackwell_matmul.c src/omega_blackwell_qmd.c src/omega_blackwell_realize.c src/sha256.c \
	src/omega_gpu_session.c src/omega_gpu_fmt.c src/omega_gpu_matmul_api.c src/omega_gpu_attention_api.c \
	src/omega_gpu_serving.c src/omega_gpu_wait.c
GPU_FS_HDRS = src/omega_gpu_engine.h src/omega_gpu_platform.h src/omega_blackwell_codegen.h src/omega_blackwell_matmul.h \
	src/omega_gpu_session.h src/omega_gpu_fmt.h src/omega_gpu_matmul_api.h src/omega_gpu_attention_api.h \
	src/omega_gpu_serving.h src/omega_gpu_wait.h
GPU_FS_OBJS = $(patsubst src/%.c,$(GPU_FS_DIR)/%.o,$(GPU_FS_SRCS))
GPU_FS_LINK = $(GPU_FS_DIR)/omega_gpu_freestanding.o
GPU_FS_ALLOWED = memcpy|memmove|memset|memcmp|strlen|strcmp

$(GPU_FS_DIR)/%.o: src/%.c $(GPU_FS_HDRS)
	@mkdir -p $(GPU_FS_DIR)
	gcc $(GPU_FS_CFLAGS) -c -o $@ $<

$(GPU_FS_LINK): $(GPU_FS_OBJS)
	gcc -r -nostdlib -o $@ $(GPU_FS_OBJS)
	@bad=$$(nm -u $@ | awk '{print $$NF}' | grep -Ev '^(_GLOBAL_OFFSET_TABLE_|($(GPU_FS_ALLOWED)))$$' || true); \
	if [ -n "$$bad" ]; then echo "freestanding GPU link still needs: $$bad"; rm -f $@; exit 1; fi
	@echo "freestanding GPU link ($(words $(GPU_FS_SRCS)) files): only $$(nm -u $@ | awk '{print $$NF}' | grep -Ev '^_GLOBAL_OFFSET_TABLE_$$' | sort -u | tr '\n' ' ')undefined"
gpu-engine-freestanding: $(GPU_FS_LINK)

build/test_omega_gpu_engine_freestanding: tests/test_omega_gpu_engine.c $(GPU_FS_DIR)/omega_gpu_engine.o src/omega_gpu_engine.h
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -DOMEGA_NUMERIC_CPU_ONLY -DOMEGA_GPU_NO_POSIX \
		-o $@ tests/test_omega_gpu_engine.c $(GPU_FS_DIR)/omega_gpu_engine.o
build/test_omega_gpu_platform: tests/test_omega_gpu_platform.c $(GPU_FS_DIR)/omega_gpu_platform.o src/omega_gpu_platform.h
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -o $@ tests/test_omega_gpu_platform.c $(GPU_FS_DIR)/omega_gpu_platform.o
test-gpu-engine-freestanding: $(GPU_FS_LINK) build/test_omega_gpu_engine_freestanding build/test_omega_gpu_platform
	./build/test_omega_gpu_engine_freestanding
	./build/test_omega_gpu_platform

# native v1 P07 cut A.2 (aienos#286): code generation joins the freestanding link above with a caller-owned
# code buffer (omega_blackwell_kernel_encode_into, no allocator). The same test source is built hosted (the
# allocating generators) and freestanding (the *_prog generators into a caller buffer); their instruction
# stream dumps of every fixture must be byte-identical (cmp). The freestanding run then checks the bounded
# buffer contract, the golden-word fixture verifiers on the freestanding object and the bf16 rounding path on
# all 2^32 inputs. The allowlist control links a malloc caller into the freestanding object and requires the
# nm -u check to refuse it (a check that can never fail proves nothing).
.PHONY: test-codegen-freestanding test-gpu-fs-allowlist-control
CODEGEN_FS_TEST_DEPS = tests/test_omega_codegen_freestanding.c src/omega_blackwell_codegen.h src/omega_blackwell_matmul.h
build/test_codegen_hosted: $(CODEGEN_FS_TEST_DEPS) src/omega_blackwell_codegen.c src/omega_blackwell_matmul.c src/sha256.c
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -o $@ tests/test_omega_codegen_freestanding.c \
		src/omega_blackwell_codegen.c src/omega_blackwell_matmul.c src/sha256.c
build/test_codegen_freestanding: $(CODEGEN_FS_TEST_DEPS) $(GPU_FS_LINK)
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -DOMEGA_GPU_NO_POSIX -o $@ tests/test_omega_codegen_freestanding.c \
		$(GPU_FS_DIR)/omega_blackwell_codegen.o $(GPU_FS_DIR)/omega_blackwell_matmul.o $(GPU_FS_DIR)/sha256.o
test-codegen-freestanding: build/test_codegen_hosted build/test_codegen_freestanding test-gpu-fs-allowlist-control
	./build/test_codegen_hosted --dump > build/codegen_dump_hosted.txt
	./build/test_codegen_freestanding --dump > build/codegen_dump_freestanding.txt
	cmp build/codegen_dump_hosted.txt build/codegen_dump_freestanding.txt
	@echo "codegen dumps byte-identical: $$(grep -c '^fixture ' build/codegen_dump_hosted.txt) fixtures, $$(grep -v -E '^(fixture|digest) ' build/codegen_dump_hosted.txt | tr -d '\n' | wc -c | awk '{print $$1/2}') code bytes"
	./build/test_codegen_freestanding

test-gpu-fs-allowlist-control: $(GPU_FS_LINK)
	@mkdir -p $(GPU_FS_DIR)/control
	printf 'void *malloc(unsigned long);\nvoid *omega_gpu_fs_control(void) { return malloc(16); }\n' > $(GPU_FS_DIR)/control/canary.c
	gcc $(GPU_FS_CFLAGS) -c -o $(GPU_FS_DIR)/control/canary.o $(GPU_FS_DIR)/control/canary.c
	gcc -r -nostdlib -o $(GPU_FS_DIR)/control/with_canary.o $(GPU_FS_OBJS) $(GPU_FS_DIR)/control/canary.o
	@bad=$$(nm -u $(GPU_FS_DIR)/control/with_canary.o | awk '{print $$NF}' | grep -Ev '^(_GLOBAL_OFFSET_TABLE_|($(GPU_FS_ALLOWED)))$$' || true); \
	if [ "$$bad" != "malloc" ]; then echo "allowlist control: expected exactly malloc refused, got '$$bad'"; exit 1; fi; \
	echo "allowlist control: a malloc caller is refused by the nm -u check (as required)"

# native v1 P07 cut C, session part (aienos#286): the portable session over OmegaGpuPlatform.
#  - test_omega_gpu_session_platform links the FREESTANDING partial-link object and drives it through a fake
#    platform: install refusals, no default platform, typed GPU-uncached buffers, polling only, cancel before and
#    after submit, timeout (latch + park, nothing released), stale handle after close/reopen, reservation refusal.
#  - test_omega_gpu_platform_linux drives the Linux REFERENCE platform (reference only) on the simulated driver:
#    same driver calls in the same order as before cut C, its refusals, and a marker that never lands.
#  - the mutants prove the session test can fail: each one breaks one rule in a copy of the session and the test
#    must report FAIL for it.
.PHONY: test-gpu-session-platform test-gpu-session-mutants
build/test_omega_gpu_session_platform: tests/test_omega_gpu_session_platform.c $(GPU_FS_LINK)
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -DOMEGA_GPU_NO_POSIX -o $@ tests/test_omega_gpu_session_platform.c $(GPU_FS_LINK)
GPU_LINUX_REF_SRCS = src/omega_gpu_session.c src/omega_gpu_platform.c src/omega_gpu_platform_linux.c src/omega_gpu_fmt.c src/omega_blackwell_qmd.c
# omega #459: the command list over the reference (near-full ring with the #457 L2 invalidate prefix pending) needs the
# cached allocation, the command list and the codegen behind its nop kernel.
GPU_LINUX_REF_CMDLIST_SRCS = src/omega_gpu_mem_cached.c src/omega_gpu_cmdlist.c src/omega_blackwell_codegen.c src/omega_blackwell_encoder.c \
	src/omega_blackwell_matmul.c src/omega_blackwell_realize.c src/sha256.c
build/test_omega_gpu_platform_linux: tests/test_omega_gpu_platform_linux.c tests/fake_m16_native.c tests/fake_m16_native.h $(GPU_LINUX_REF_SRCS) $(GPU_FS_HDRS) src/omega_gpu_platform_linux.h \
		src/omega_gpu_cmdlist.h src/omega_gpu_mem_cached.h $(GPU_LINUX_REF_CMDLIST_SRCS)
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -pthread -Isrc -Itests $(GPU_ENGINE_PHYS_INC) \
		-o $@ tests/test_omega_gpu_platform_linux.c tests/fake_m16_native.c $(GPU_LINUX_REF_SRCS) $(GPU_LINUX_REF_CMDLIST_SRCS) -lm
test-gpu-session-platform: build/test_omega_gpu_session_platform build/test_omega_gpu_platform_linux test-gpu-session-mutants
	./build/test_omega_gpu_session_platform
	./build/test_omega_gpu_platform_linux

GPU_SESSION_MUT_DIR = build/gpu_session_mut
test-gpu-session-mutants: $(GPU_FS_LINK)
	@mkdir -p $(GPU_SESSION_MUT_DIR)
	@set -e; killed=0; \
	smut() { name=$$1; script=$$2; \
	  sed -e "$$script" src/omega_gpu_session.c > $(GPU_SESSION_MUT_DIR)/$$name.c; \
	  if cmp -s src/omega_gpu_session.c $(GPU_SESSION_MUT_DIR)/$$name.c; then echo "mutant $$name: pattern not found"; exit 1; fi; \
	  gcc $(GPU_FS_CFLAGS) -c -o $(GPU_SESSION_MUT_DIR)/$$name.o $(GPU_SESSION_MUT_DIR)/$$name.c; \
	  objs=$$(echo $(GPU_FS_OBJS) | tr ' ' '\n' | grep -v '/omega_gpu_session.o$$' | tr '\n' ' '); \
	  gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -DOMEGA_GPU_NO_POSIX -o $(GPU_SESSION_MUT_DIR)/$$name \
	    tests/test_omega_gpu_session_platform.c $$objs $(GPU_SESSION_MUT_DIR)/$$name.o; \
	  if ./$(GPU_SESSION_MUT_DIR)/$$name > $(GPU_SESSION_MUT_DIR)/$$name.log 2>&1; then echo "mutant $$name SURVIVED"; exit 1; fi; \
	  echo "mutant $$name killed: $$(grep -c '^FAIL' $(GPU_SESSION_MUT_DIR)/$$name.log) case(s) failed"; killed=$$((killed+1)); }; \
	smut release-when-latched 's|if (LOAD(g_blocked)) {$$|if (0) {|'; \
	smut ignore-cancel 's|if (LOAD(g_cancel)) return WAIT_CANCELLED;|;|g'; \
	smut no-latch-on-timeout 's|latch(w == WAIT_TIMED_OUT|omega_gpu_session_set_error(w == WAIT_TIMED_OUT|'; \
	smut stale-not-counted 's|else { BUMP(g_stats.stale_refusals, 1u); fail_plat("dma_free", rc); }|else {}|'; \
	smut marker-not-polled 's|RIGHTS_MARKER, PAGE|RIGHTS_DATA, PAGE|'; \
	smut close-while-latched 's|if (g_open \&\& !LOAD(g_blocked) \&\& p) {|if (g_open \&\& p) {|; s|if (!LOAD(g_blocked)) { /\* a hook|if (1) { /* a hook|'; \
	smut recover-keeps-latch 's|STORE(g_blocked, 0u);|;|'; \
	smut recover-without-close 's|const int prc = p->device_close(p->ctx, g_s.dev);|const int prc = OMEGA_GPU_PLAT_OK;|'; \
	smut recover-when-not-latched 's|const int latched = LOAD(g_blocked) \|\| LOAD(g_module_latched);|const int latched = 1;|'; \
	smut recover-ignores-teardown-fault 's|if (prc != OMEGA_GPU_PLAT_OK) {|if (0) {|'; \
	smut recover-ignores-module-latch 's|const int latched = LOAD(g_blocked) \|\| LOAD(g_module_latched);|const int latched = LOAD(g_blocked);|'; \
	echo "test-gpu-session-mutants: $$killed of 11 mutants killed"
