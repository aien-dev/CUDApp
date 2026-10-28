# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Include from Makefile. Hosted .om packer, no GPU; regenerate the checked
# portable TUs with om/units.sh after changing om/cubin/cubin_elf.om.
AMPERE_CUBIN_DIR = $(OUT_DIR)/ampere-cubin
AMPERE_NVDISASM ?= /usr/local/cuda/bin/nvdisasm
AMPERE_CUBIN_SOURCES = tests/ampere_cubin_test.c src/omega_cubin.c $(CUDAPP_OM_UNIT_SRCS)
AMPERE_CUBIN_DEPS = $(AMPERE_CUBIN_SOURCES) src/omega_cubin.h src/cudapp_native.h

.PHONY: test-ampere-cubin test-ampere-cubin-disasm
$(AMPERE_CUBIN_DIR)/host: $(AMPERE_CUBIN_DEPS)
	@mkdir -p $(AMPERE_CUBIN_DIR)
	$(CC) $(CFLAGS) -Isrc -pthread -o $@ $(AMPERE_CUBIN_SOURCES)

$(AMPERE_CUBIN_DIR)/diff: $(AMPERE_CUBIN_DEPS)
	@mkdir -p $(AMPERE_CUBIN_DIR)
	$(CC) $(CFLAGS) -Isrc -pthread -DCUDAPP_MIGRATION_PENDING_CUBIN_DIFF -o $@ $(AMPERE_CUBIN_SOURCES)

test-ampere-cubin: $(AMPERE_CUBIN_DIR)/host $(AMPERE_CUBIN_DIR)/diff
	$(AMPERE_CUBIN_DIR)/host $(AMPERE_CUBIN_DIR)/sm86.cubin
	$(AMPERE_CUBIN_DIR)/diff

test-ampere-cubin-disasm: test-ampere-cubin
	@test -x "$(AMPERE_NVDISASM)" || { echo 'NOT_RUN: missing offline nvdisasm'; exit 3; }
	"$(AMPERE_NVDISASM)" -c $(AMPERE_CUBIN_DIR)/sm86.cubin > $(AMPERE_CUBIN_DIR)/sm86.sass
	@grep -q 'EXIT' $(AMPERE_CUBIN_DIR)/sm86.sass || { echo 'FAIL: sm86 EXIT not decoded'; exit 1; }
	@echo 'AMPERE CUBIN NVDISASM PASS; device NOT_RUN'

# Independently runnable on a native x86 host before the bridge prerequisite.
.PHONY: test-ampere-numeric-host
$(AMPERE_CUBIN_DIR)/numeric-host: tests/ampere_numeric_host_test.c src/omega_numeric.c src/omega_numeric.h src/omega_numeric_reduce.c src/omega_numeric_reduce.h
	@mkdir -p $(AMPERE_CUBIN_DIR)
	$(CC) $(CFLAGS) -Isrc -ffp-contract=off -ffunction-sections -fdata-sections -Wl,--gc-sections -o $@ tests/ampere_numeric_host_test.c src/omega_numeric.c src/omega_numeric_reduce.c -lm

test-ampere-numeric-host: $(AMPERE_CUBIN_DIR)/numeric-host
	$(AMPERE_CUBIN_DIR)/numeric-host
