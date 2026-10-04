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
