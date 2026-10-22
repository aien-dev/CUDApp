# SPDX-License-Identifier: MIT
NUMERIC_TEST = $(OUT_DIR)/numerical_contract_test
$(NUMERIC_TEST): tests/numerical_contract_test.c src/omega_gpu_numerical_contract.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -o $@ $< $(OUT_DIR)/libomega_gpu.a -lpthread -lm
.PHONY: test-numerical-contract
test-numerical-contract: $(NUMERIC_TEST) $(OUT_DIR)/numerical_router_tie_test
	./$(NUMERIC_TEST)
	./$(OUT_DIR)/numerical_router_tie_test
	node --test tests/numerical_trace_test.mjs
# Include after HOST_TESTS is assigned by the root makefile through prerequisites.
test: test-numerical-contract
.PHONY: test-numerical-mutants
test-numerical-mutants: $(NUMERIC_TEST)
	@mkdir -p $(OUT_DIR)/numeric-mutants
	sed 's/if (rb \&\& (st || (q \& 1u)))/if (0 \&\& rb \&\& (st || (q \& 1u)))/' src/omega_gpu_fp8_api.c > $(OUT_DIR)/numeric-mutants/wrong-rounding.c
	@! cmp -s src/omega_gpu_fp8_api.c $(OUT_DIR)/numeric-mutants/wrong-rounding.c
	$(CC) $(CFLAGS) -ffp-contract=off -c $(OUT_DIR)/numeric-mutants/wrong-rounding.c -o $(OUT_DIR)/numeric-mutants/wrong-rounding.o
	$(CC) $(CFLAGS) -ffp-contract=off -o $(OUT_DIR)/numeric-mutants/wrong-rounding tests/numerical_contract_test.c $(OUT_DIR)/numeric-mutants/wrong-rounding.o $(OUT_DIR)/libomega_gpu.a -lpthread -lm
	@if ./$(OUT_DIR)/numeric-mutants/wrong-rounding >$(OUT_DIR)/numeric-mutants/wrong-rounding.log 2>&1; then echo 'FAIL wrong-rounding survived'; exit 1; fi
	@grep '^FAIL.*omega_gpu_fp8_e4m3_rne' $(OUT_DIR)/numeric-mutants/wrong-rounding.log
	@echo 'wrong-rounding production mutant killed'
	sed 's/else fminn(e, R(lc), R(lc), R(cj));/else fmaxx(e, R(lc), R(lc), R(cj));/' src/omega_blackwell_codegen_router.c > $(OUT_DIR)/numeric-mutants/wrong-tie.c
	@! cmp -s src/omega_blackwell_codegen_router.c $(OUT_DIR)/numeric-mutants/wrong-tie.c
	$(CC) $(CFLAGS) -ffp-contract=off -c $(OUT_DIR)/numeric-mutants/wrong-tie.c -o $(OUT_DIR)/numeric-mutants/wrong-tie.o
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $(OUT_DIR)/numeric-mutants/wrong-tie tests/numerical_router_tie_test.c $(OUT_DIR)/numeric-mutants/wrong-tie.o $(OUT_DIR)/libomega_gpu.a -lpthread -lm
	@if ./$(OUT_DIR)/numeric-mutants/wrong-tie >$(OUT_DIR)/numeric-mutants/wrong-tie.log 2>&1; then echo 'FAIL wrong-tie survived'; exit 1; fi
	@grep '^FAIL.*!badids' $(OUT_DIR)/numeric-mutants/wrong-tie.log
	@echo 'wrong-tie production mutant killed'
test-numerical-contract: test-numerical-mutants
NUMERIC_ROUTER_TEST = $(OUT_DIR)/numerical_router_tie_test
$(NUMERIC_ROUTER_TEST): tests/numerical_router_tie_test.c tests/bw_warp_sim_k.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ $< $(OUT_DIR)/libomega_gpu.a -lpthread -lm
