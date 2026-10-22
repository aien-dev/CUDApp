# SPDX-License-Identifier: MIT
NUMERIC_TEST = $(OUT_DIR)/numerical_contract_test
$(NUMERIC_TEST): tests/numerical_contract_test.c src/omega_gpu_numerical_contract.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ $< $(OUT_DIR)/libomega_gpu.a -lpthread -lm
.PHONY: test-numerical-contract
test-numerical-contract: $(NUMERIC_TEST)
	./$(NUMERIC_TEST)
	node --test tests/numerical_trace_test.mjs
# Include after HOST_TESTS is assigned by the root makefile through prerequisites.
test: test-numerical-contract
