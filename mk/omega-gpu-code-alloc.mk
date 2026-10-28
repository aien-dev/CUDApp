# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# GPU code buffer sizing (src/omega_gpu_code_alloc.h): the 2 KB instruction prefetch tail after every
# kernel (omega#323). Host only: no device, no physics headers. Three parts: the size rule, a NO_TAIL
# mutant that must fail it, and a source scan that refuses the page-only code sizing idioms the six
# variable-size upload sites and five fixed-size sites used before.
CODE_ALLOC_SCAN_SRC := $(wildcard src/*.c src/runtime/*.c)
CODE_ALLOC_SCAN_RE := code_size \+ 0x[fF]{3}|session_alloc\([^,]*code_size|alloc[a-z_]*\([^,]*, *(0x[0-9a-fA-F]+|OMEGA_DS_MAX_CODE_BYTES), *&[a-z_>.-]*code_mem\)
.PHONY: test-gpu-code-alloc test-gpu-code-alloc-scan
build/test_omega_gpu_code_alloc: tests/test_omega_gpu_code_alloc.c src/omega_gpu_code_alloc.h
	@mkdir -p build
	gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -o $@ tests/test_omega_gpu_code_alloc.c
test-gpu-code-alloc: build/test_omega_gpu_code_alloc
	./build/test_omega_gpu_code_alloc
	@gcc -std=gnu11 -O2 -Wall -Wextra -Werror -DOMEGA_GPU_CODE_ALLOC_MUTANT_NO_TAIL -Isrc -o build/test_omega_gpu_code_alloc_no_tail tests/test_omega_gpu_code_alloc.c
	@if ./build/test_omega_gpu_code_alloc_no_tail >build/test_omega_gpu_code_alloc_no_tail.log 2>&1; then echo "MUTANT SURVIVED: NO_TAIL"; exit 1; fi
	@echo "NO_TAIL mutant: killed ($$(grep -c '^FAIL:' build/test_omega_gpu_code_alloc_no_tail.log) failing checks)"
	@$(MAKE) --no-print-directory test-gpu-code-alloc-scan
	@if $(MAKE) --no-print-directory test-gpu-code-alloc-scan CODE_ALLOC_SCAN_SRC=build/code-alloc-missing-input.c >build/code-alloc-missing.log 2>&1; then echo "source scan accepted a missing input"; exit 1; fi
	@grep -q '^source scan failed (grep exit 2)' build/code-alloc-missing.log
	@if $(MAKE) --no-print-directory test-gpu-code-alloc-scan CODE_ALLOC_SCAN_SRC= >build/code-alloc-empty.log 2>&1; then echo "source scan accepted an empty readset"; exit 1; fi
	@grep -q '^source scan has no inputs' build/code-alloc-empty.log
	@printf '%s\n' 'code_size + 0xfff' >build/code-alloc-forbidden.c
	@if $(MAKE) --no-print-directory test-gpu-code-alloc-scan CODE_ALLOC_SCAN_SRC=build/code-alloc-forbidden.c >build/code-alloc-forbidden.log 2>&1; then echo "source scan accepted forbidden sizing"; exit 1; fi
	@grep -q '^code buffer sized without the prefetch tail' build/code-alloc-forbidden.log
	@echo "OMEGA_GPU_CODE_ALLOC_GATE PASS"

# grep: 0 is a forbidden match, 1 is a clean scan, every other status is a scan failure.
test-gpu-code-alloc-scan:
	@test -n "$(strip $(CODE_ALLOC_SCAN_SRC))" || { echo "source scan has no inputs"; exit 2; }
	@rc=0; grep -nE '$(CODE_ALLOC_SCAN_RE)' $(CODE_ALLOC_SCAN_SRC) || rc=$$?; \
	case $$rc in \
	1) echo "source scan: no page-only code sizing" ;; \
	0) echo "code buffer sized without the prefetch tail (use omega_gpu_code_alloc_bytes)"; exit 1 ;; \
	*) echo "source scan failed (grep exit $$rc)"; exit 2 ;; \
	esac
