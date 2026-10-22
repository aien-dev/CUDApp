# C6 of docs/OMEGA.md: Omega's own sm_121 disassembler (src/omega_sm121_disasm.c, tools/sm121dis/).
#   make test-sm121dis   collect every kernel the library builds (no GPU), compare our text with nvdisasm's, word by word;
#                        then the red check: a decoder built with one deliberate fault must FAIL the same comparison.
# Picked up by the `-include mk/*.mk` line of the Makefile. The test binaries are named as files, not through the
# GPU_*_TEST variables: this file is read before the Makefile defines them.
SM121DIS = $(OUT_DIR)/sm121dis
SM121DIS_RED = $(OUT_DIR)/sm121dis_red
SM121DIS_K = $(OUT_DIR)/sm121dis-kernels

$(SM121DIS): tools/sm121dis/sm121dis.c src/omega_sm121_disasm.c src/omega_sm121_disasm.h
	@mkdir -p $(OUT_DIR)
	$(CC) -std=gnu11 -Wall -Wextra -Werror -O2 -Isrc -o $@ tools/sm121dis/sm121dis.c src/omega_sm121_disasm.c
$(SM121DIS_RED): tools/sm121dis/sm121dis.c src/omega_sm121_disasm.c src/omega_sm121_disasm.h
	@mkdir -p $(OUT_DIR)
	$(CC) -std=gnu11 -Wall -Wextra -Werror -O2 -DSM121DIS_MUTANT=1 -Isrc -o $@ tools/sm121dis/sm121dis.c src/omega_sm121_disasm.c

sm121dis-corpus-bins: $(addprefix $(OUT_DIR)/,gpu_matmul_api_test gpu_elementwise_test gpu_reconv_test gpu_attention_test \
	gpu_moe_window_test gpu_fp8_api_test gpu_decode_attn_test gpu_router_test gpu_bf16_gemv_test gpu_tern_test gpu_fp8gemm_test \
	gpu_tc_probe_test gpu_tc_peak_test gpu_pf_mem_test gpu_qmma_probe_test gpu_cmdlist_test gpu_decode_graph_test gpu_tcgemm_test \
	gpu_prefill_attn_test gpu_pfgemm_test gpu_qgemm_test gpu_pfhost_test gpu_gpr_boundary_test) \
	build/test_omega_divsqrt_gb10_cpu build/test_omega_ldst_gb10_cpu

.PHONY: test-sm121dis sm121dis-corpus-bins
test-sm121dis: $(SM121DIS) $(SM121DIS_RED) sm121dis-corpus-bins
	rm -rf $(SM121DIS_K) && sh tools/sm121dis/corpus.sh $(OUT_DIR) $(SM121DIS_K)
	sh tools/sm121dis/check.sh $(SM121DIS) $(SM121DIS_K) $(OUT_DIR)/sm121dis-work
	@if sh tools/sm121dis/check.sh $(SM121DIS_RED) $(SM121DIS_K) $(OUT_DIR)/sm121dis-red > $(OUT_DIR)/sm121dis-red.log 2>&1; then \
		echo "RED CHECK BROKEN: the faulty decoder passed"; exit 1; \
	else echo "red check: the faulty decoder FAILS the comparison, as it must ($$(grep -c '^\[FAIL\]' $(OUT_DIR)/sm121dis-red.log) kernels)"; fi

# The provenance checks (tools/divsqrt_nvdisasm_check.sh, tools/ldst_nvdisasm_check.sh, test-numeric-reduce-nvdisasm)
# call build/sm121dis by that name, whatever OUT_DIR is.
ifneq ($(OUT_DIR),build)
build/sm121dis: tools/sm121dis/sm121dis.c src/omega_sm121_disasm.c src/omega_sm121_disasm.h
	@mkdir -p build
	$(CC) -std=gnu11 -Wall -Wextra -Werror -O2 -Isrc -o $@ tools/sm121dis/sm121dis.c src/omega_sm121_disasm.c
endif
