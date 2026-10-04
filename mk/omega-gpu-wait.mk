# Shared GPU completion wait primitive (src/omega_gpu_wait.c): host unit test with
# fake memory and a fake clock, plus a naive wall-clock mutant that must be
# caught. Picked up by `-include mk/*.mk`. Host only: opens no device, needs no
# physics headers, takes no chip lock.
.PHONY: test-gpu-wait
build/test_omega_gpu_wait: tests/test_omega_gpu_wait.c src/omega_gpu_wait.c src/omega_gpu_wait.h
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -Isrc -o $@ tests/test_omega_gpu_wait.c src/omega_gpu_wait.c
test-gpu-wait: build/test_omega_gpu_wait
	./build/test_omega_gpu_wait

# Real-code mutants: src/omega_gpu_wait.c carries one-line compile-flag mutants.
# Each flagged build of the real primitive must FAIL the suite; the unflagged
# build must PASS. A mutant that does not compile is an error, not a kill.
GPU_WAIT_MUTANTS := OMEGA_GPU_WAIT_MUTANT_STALL_NO_RESET OMEGA_GPU_WAIT_MUTANT_PROGRESS_EXTENDS_HARD OMEGA_GPU_WAIT_MUTANT_NO_READ_BARRIER
.PHONY: test-gpu-wait-mutants
test-gpu-wait-mutants: build/test_omega_gpu_wait
	@./build/test_omega_gpu_wait >/dev/null || { echo "unflagged build FAILED"; exit 1; }
	@echo "unflagged: PASS"
	@set -e; for m in $(GPU_WAIT_MUTANTS); do \
	  gcc -std=gnu11 -O2 -Wall -Wextra -Werror -D_GNU_SOURCE -D$$m -Isrc -o build/test_omega_gpu_wait_$$m tests/test_omega_gpu_wait.c src/omega_gpu_wait.c; \
	  if ./build/test_omega_gpu_wait_$$m >build/test_omega_gpu_wait_$$m.log 2>&1; then echo "MUTANT SURVIVED: $$m"; exit 1; fi; \
	  echo "$$m: killed ($$(grep -c 'FAIL' build/test_omega_gpu_wait_$$m.log) failing checks)"; \
	done
	@echo "OMEGA_GPU_WAIT_MUTANTS PASS"
