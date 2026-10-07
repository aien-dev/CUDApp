# GPU code buffer sizing (src/omega_gpu_code_alloc.h): the 2 KB instruction prefetch tail after every
# kernel (omega#323). Host only: no device, no physics headers. Three parts: the size rule, a NO_TAIL
# mutant that must fail it, and a source scan that refuses the page-only code sizing idioms the six
# variable-size upload sites and five fixed-size sites used before.
CODE_ALLOC_SCAN_SRC := src/*.c src/runtime/*.c
CODE_ALLOC_SCAN_RE := code_size \+ 0x[fF]{3}|session_alloc\([^,]*code_size|alloc[a-z_]*\([^,]*, *(0x[0-9a-fA-F]+|OMEGA_DS_MAX_CODE_BYTES), *&[a-z_>.-]*code_mem\)
.PHONY: test-gpu-code-alloc
build/test_omega_gpu_code_alloc: tests/test_omega_gpu_code_alloc.c src/omega_gpu_code_alloc.h
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -o $@ tests/test_omega_gpu_code_alloc.c
test-gpu-code-alloc: build/test_omega_gpu_code_alloc
	./build/test_omega_gpu_code_alloc
	@gcc -std=gnu11 -O2 -Wall -Wextra -Werror -DOMEGA_GPU_CODE_ALLOC_MUTANT_NO_TAIL -Isrc -o build/test_omega_gpu_code_alloc_no_tail tests/test_omega_gpu_code_alloc.c
	@if ./build/test_omega_gpu_code_alloc_no_tail >build/test_omega_gpu_code_alloc_no_tail.log 2>&1; then echo "MUTANT SURVIVED: NO_TAIL"; exit 1; fi
	@echo "NO_TAIL mutant: killed ($$(grep -c '^FAIL:' build/test_omega_gpu_code_alloc_no_tail.log) failing checks)"
	@if grep -nE '$(CODE_ALLOC_SCAN_RE)' $(CODE_ALLOC_SCAN_SRC); then echo "code buffer sized without the prefetch tail (use omega_gpu_code_alloc_bytes)"; exit 1; fi
	@echo "source scan: no page-only code sizing"
	@echo "OMEGA_GPU_CODE_ALLOC_GATE PASS"
