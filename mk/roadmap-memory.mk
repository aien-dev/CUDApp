# SPDX-License-Identifier: MIT
# Explicit host-only issue24 target; no default enabling or hardware execution.
.PHONY: test-roadmap-memory test-roadmap-memory-mutants
ROADMAP_MEMORY_SRCS = src/omega_gpu_admission.c src/omega_gpu_platform.c src/omega_gpu_platform_linux.c tests/fake_m16_native.c
ROADMAP_MEMORY_HEADERS = src/omega_gpu_admission.h src/omega_gpu_platform.h src/omega_gpu_platform_linux.h tests/fake_m16_native.h tests/admission_state_fixture.h
$(OUT_DIR)/test_omega_gpu_admission: tests/test_omega_gpu_admission.c $(ROADMAP_MEMORY_HEADERS) $(ROADMAP_MEMORY_SRCS) | $(OUT_DIR)
	$(CC) $(CFLAGS) -Itests -o $@ tests/test_omega_gpu_admission.c $(ROADMAP_MEMORY_SRCS) -Wl,--wrap=pwrite -lpthread
test-roadmap-memory: $(OUT_DIR)/test_omega_gpu_admission
	timeout 45 $<
test-roadmap-memory-mutants:
	tools/admission_mutations.sh "$(CC)" "$(CFLAGS)" "$(OUT_DIR)"
