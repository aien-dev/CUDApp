# SPDX-License-Identifier: AGPL-3.0-only
# Host-only tests of the m16 wait primitive (from the physics repository Makefile). No GPU.
M16_CFLAGS ?= -O2 -Wall -Wextra -Werror
BUILD := build/m16_gpu_wait

.PHONY: test-m16-gpu-wait mutants-m16-gpu-wait host-tests-m16

host-tests-m16: test-m16-gpu-wait mutants-m16-gpu-wait

$(BUILD)/test_real: tests/test_m16_gpu_wait.c m16/m16_gpu_wait.c m16/m16_gpu_wait.h
	@mkdir -p $(BUILD)
	$(CC) $(M16_CFLAGS) -I m16 tests/test_m16_gpu_wait.c m16/m16_gpu_wait.c -o $@

test-m16-gpu-wait: $(BUILD)/test_real
	$(BUILD)/test_real

# Three one-line compile-flag mutants of the REAL primitive. Each must make the
# host test fail, and fail in the scenario named for it. A surviving mutant
# fails this target.
#   STALL_NEVER_RESETS       killed by progress_resets_idle
#   PROGRESS_EXTENDS_HARD    killed by hard_deadline_with_progress
#   NO_BARRIER               killed by barrier_before_read
MUTANTS := STALL_NEVER_RESETS:progress_resets_idle \
           PROGRESS_EXTENDS_HARD:hard_deadline_with_progress \
           NO_BARRIER:barrier_before_read

mutants-m16-gpu-wait:
	@mkdir -p $(BUILD)
	@set -e; for m in $(MUTANTS); do \
	  flag=$${m%%:*}; scen=$${m##*:}; \
	  $(CC) $(M16_CFLAGS) -DM16_GPU_WAIT_MUT_$$flag -I m16 tests/test_m16_gpu_wait.c m16/m16_gpu_wait.c -o $(BUILD)/mut_$$flag; \
	  if $(BUILD)/mut_$$flag > $(BUILD)/mut_$$flag.log 2>&1; then echo "MUTANT $$flag SURVIVED (test passed)"; exit 1; fi; \
	  if ! grep -Eq "^real +$$scen +FAIL" $(BUILD)/mut_$$flag.log; then echo "MUTANT $$flag died but not in $$scen"; cat $(BUILD)/mut_$$flag.log; exit 1; fi; \
	  echo "MUTANT $$flag killed by $$scen"; \
	done
	@echo "M16_GPU_WAIT_MUTANTS PASS (3/3 killed)"
