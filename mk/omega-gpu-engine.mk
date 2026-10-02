# Omega GPU Engine (docs/numeric/OMEGA_GPU_ENGINE.md), phase A2: compile check
# of the header and the stub. Picked up by `-include mk/*.mk`. Host only: opens
# no device, needs no physics headers, not part of `all` or `test`.
.PHONY: test-gpu-engine-compile
build/test_omega_gpu_engine_compile: tests/test_omega_gpu_engine_compile.c src/omega_gpu_engine.c src/omega_gpu_engine.h
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -DOMEGA_NUMERIC_CPU_ONLY -o $@ tests/test_omega_gpu_engine_compile.c src/omega_gpu_engine.c
test-gpu-engine-compile: build/test_omega_gpu_engine_compile
	./build/test_omega_gpu_engine_compile
