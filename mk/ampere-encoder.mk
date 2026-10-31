# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Offline-only tests; never a device qualification. The parent Makefile adds
# $(AMPERE_CORPUS_SRCS) to SRCS. Use a distinct OUT_DIR when enabling capture.
AMPERE_ENCODER_TEST = $(OUT_DIR)/ampere_encoder_test
AMPERE_ORACLE_CORPUS = $(OUT_DIR)/ampere-oracle-corpus
CUDAPP_AMPERE_CORPUS ?= 0
ifeq ($(CUDAPP_AMPERE_CORPUS),1)
CFLAGS += -DCUDAPP_AMPERE_CORPUS
AMPERE_CORPUS_SRCS = tools/ampere_corpus.c
vpath ampere_corpus.c tools
endif

$(AMPERE_ENCODER_TEST): tests/test_omega_ampere_encoder.c src/omega_ampere_codegen.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/test_omega_ampere_encoder.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

$(OUT_DIR)/ampere_families: tools/ampere_families.c $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tools/ampere_families.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

$(OUT_DIR)/ampere_matmul_families: tests/test_omega_codegen_freestanding.c $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -o $@ tests/test_omega_codegen_freestanding.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

.PHONY: test-ampere-encoder test-ampere-encoder-oracle
test-ampere-encoder: $(AMPERE_ENCODER_TEST)
	./$(AMPERE_ENCODER_TEST)

# No recursive make/test. Existing producers execute serially in their documented
# host-only/simulator modes; raw-patched FP8 markers must be explicitly refused.
#   make OUT_DIR=build-ampere-oracle CUDAPP_AMPERE_CORPUS=1 test-ampere-encoder-oracle
test-ampere-encoder-oracle: $(AMPERE_ENCODER_TEST) $(OUT_DIR)/ampere_math_device \
		$(OUT_DIR)/ampere_families $(OUT_DIR)/ampere_matmul_families sm121dis-corpus-bins
	@test "$(CUDAPP_AMPERE_CORPUS)" = 1 || { echo 'use CUDAPP_AMPERE_CORPUS=1 and a fresh OUT_DIR' >&2; exit 1; }
	@rm -rf -- $(AMPERE_ORACLE_CORPUS)
	@mkdir -p $(AMPERE_ORACLE_CORPUS)/sm121
	OMEGA_AMPERE_CORPUS_DIR=$(abspath $(AMPERE_ORACLE_CORPUS)) ./$(AMPERE_ENCODER_TEST)
	OMEGA_AMPERE_CORPUS_DIR=$(abspath $(AMPERE_ORACLE_CORPUS)) ./$(OUT_DIR)/ampere_families
	OMEGA_AMPERE_CORPUS_DIR=$(abspath $(AMPERE_ORACLE_CORPUS)) ./$(OUT_DIR)/ampere_matmul_families --dump
	OMEGA_AMPERE_CORPUS_DIR=$(abspath $(AMPERE_ORACLE_CORPUS)) sh tools/sm121dis/corpus.sh $(OUT_DIR) $(AMPERE_ORACLE_CORPUS)/sm121
	OMEGA_AMPERE_CORPUS_DIR=$(abspath $(AMPERE_ORACLE_CORPUS)) ./$(OUT_DIR)/ampere_math_device --emit-only
	python3 tools/ampere_nvdisasm.py $(AMPERE_ORACLE_CORPUS)
