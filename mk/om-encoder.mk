# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# CUDApp#79: the sm_121 instruction encoder in seeded Omega (om/encoder/bw_enc_sm121.om), linked through the
# portable generated C units $(CUDAPP_BINDING_SRCS) that om/units.sh generates. The C encoder in
# src/omega_blackwell_codegen.c is migration-pending: the freestanding build and the oracle builds only.
#   make test-om-encoder          (in `make test`, no Omega tools needed)
#       1. the golden-word fixture verifiers (nvcc/nvdisasm oracles) through the .om encoder;
#       2. a build with CUDAPP_MIGRATION_PENDING_ENCODER_DIFF encodes every codegen fixture both ways and must log
#          only SAME lines; 3. its red check: the same build with one flipped output bit must abort.
#   make test-om-encoder-corpus   every kernel of the sm121dis corpus (tools/sm121dis/corpus.sh) built three ways:
#       .om (OUT_DIR), C oracle ($(OUT_DIR)-enc-c), diff ($(OUT_DIR)-enc-diff); the .om and C kernel sets must be
#       identical file for file, and the diff build must log no DIFFER line.
#   make test-om-units            the committed binding regenerates byte for byte from the pinned omc + bridge
#   make test-om-encoder-mutants  three mutant .om encoders, each compiled by the pinned omc, must go red
# The last two accept OMEGA_OMC and OMEGA_BRIDGE; their defaults retain the
# OMEGA_MV and OMEGA_BRIDGE_DIR checkout convention. Missing pins refuse.
OMEGA_MV ?= ../omega-mv
OMEGA_BRIDGE_DIR ?= ../omega-bridge
OMEGA_OMC ?= $(OMEGA_MV)/seed/out/omc
OMEGA_BRIDGE ?= $(OMEGA_BRIDGE_DIR)
OM_ENC_DIR = $(OUT_DIR)/om-encoder
OM_ENC_TEST_DEPS = tests/test_omega_codegen_freestanding.c src/omega_blackwell_codegen.h src/omega_blackwell_matmul.h
OM_ENC_SRCS = src/omega_blackwell_codegen.c src/omega_blackwell_matmul.c src/sha256.c
OM_ENC_BUILD = gcc -std=gnu11 -O2 -Wall -Wextra -Werror -pthread -Isrc

.PHONY: test-om-encoder test-om-encoder-corpus test-om-units test-om-encoder-mutants
$(OM_ENC_DIR)/codegen_om: $(OM_ENC_TEST_DEPS) $(OM_ENC_SRCS) $(CUDAPP_BINDING_SRCS) src/cudapp_native.h
	@mkdir -p $(OM_ENC_DIR)
	$(OM_ENC_BUILD) -o $@ tests/test_omega_codegen_freestanding.c $(OM_ENC_SRCS) $(CUDAPP_BINDING_SRCS)
$(OM_ENC_DIR)/codegen_diff: $(OM_ENC_TEST_DEPS) $(OM_ENC_SRCS) $(CUDAPP_BINDING_SRCS) src/cudapp_native.h
	@mkdir -p $(OM_ENC_DIR)
	$(OM_ENC_BUILD) -DCUDAPP_MIGRATION_PENDING_ENCODER_DIFF -o $@ tests/test_omega_codegen_freestanding.c \
		$(OM_ENC_SRCS) $(CUDAPP_BINDING_SRCS)
$(OM_ENC_DIR)/codegen_diff_red: $(OM_ENC_TEST_DEPS) $(OM_ENC_SRCS) $(CUDAPP_BINDING_SRCS) src/cudapp_native.h
	@mkdir -p $(OM_ENC_DIR)
	$(OM_ENC_BUILD) -DCUDAPP_MIGRATION_PENDING_ENCODER_DIFF -DCUDAPP_ENCODER_DIFF_SELFTEST -o $@ \
		tests/test_omega_codegen_freestanding.c $(OM_ENC_SRCS) $(CUDAPP_BINDING_SRCS)

test-om-encoder: $(OM_ENC_DIR)/codegen_om $(OM_ENC_DIR)/codegen_diff $(OM_ENC_DIR)/codegen_diff_red
	$(OM_ENC_DIR)/codegen_om --golden
	rm -f $(OM_ENC_DIR)/diff.log
	CUDAPP_ENCODER_DIFF_LOG=$(OM_ENC_DIR)/diff.log $(OM_ENC_DIR)/codegen_diff --golden
	CUDAPP_ENCODER_DIFF_LOG=$(OM_ENC_DIR)/diff.log $(OM_ENC_DIR)/codegen_diff --dump > $(OM_ENC_DIR)/dump_diff.txt
	@same=$$(grep -c ' SAME$$' $(OM_ENC_DIR)/diff.log); other=$$(grep -vc ' SAME$$' $(OM_ENC_DIR)/diff.log); \
	if [ "$$same" -eq 0 ] || [ "$$other" -ne 0 ]; then echo "FAIL encoder diff: $$same SAME, $$other other lines"; exit 1; fi; \
	echo "encoder diff: .om and C byte-identical on $$(grep -c '^insn ' $(OM_ENC_DIR)/diff.log) golden instructions and $$(grep -c '^program ' $(OM_ENC_DIR)/diff.log) fixture programs"
	@if CUDAPP_ENCODER_DIFF_LOG=$(OM_ENC_DIR)/diff_red.log $(OM_ENC_DIR)/codegen_diff_red --dump > /dev/null 2>&1; then \
		echo "RED CHECK BROKEN: a flipped bit passed the encoder diff"; exit 1; \
	else echo "red check: one flipped bit aborts the encoder diff, as it must"; fi

test-om-encoder-corpus:
	$(MAKE) OUT_DIR=$(OUT_DIR) CUDAPP_ENCODER=om sm121dis-corpus-bins
	$(MAKE) OUT_DIR=$(OUT_DIR)-enc-c CUDAPP_ENCODER=c sm121dis-corpus-bins
	$(MAKE) OUT_DIR=$(OUT_DIR)-enc-diff CUDAPP_ENCODER=diff sm121dis-corpus-bins
	rm -rf $(OM_ENC_DIR)/k-om $(OM_ENC_DIR)/k-c $(OM_ENC_DIR)/k-diff $(OM_ENC_DIR)/corpus-diff.log
	@mkdir -p $(OM_ENC_DIR)
	sh tools/sm121dis/corpus.sh $(OUT_DIR) $(OM_ENC_DIR)/k-om
	sh tools/sm121dis/corpus.sh $(OUT_DIR)-enc-c $(OM_ENC_DIR)/k-c
	CUDAPP_ENCODER_DIFF_LOG=$(abspath $(OM_ENC_DIR))/corpus-diff.log sh tools/sm121dis/corpus.sh $(OUT_DIR)-enc-diff $(OM_ENC_DIR)/k-diff
	diff -r $(OM_ENC_DIR)/k-om $(OM_ENC_DIR)/k-c
	diff -r $(OM_ENC_DIR)/k-om $(OM_ENC_DIR)/k-diff
	@if grep -v ' SAME$$' $(OM_ENC_DIR)/corpus-diff.log; then echo "FAIL corpus encoder diff"; exit 1; fi
	@echo "encoder corpus: $$(ls $(OM_ENC_DIR)/k-om/*.bin | wc -l) distinct kernels identical .om vs C; diff build $$(grep -c '^program ' $(OM_ENC_DIR)/corpus-diff.log) program encodes and $$(grep -c '^insn ' $(OM_ENC_DIR)/corpus-diff.log) single-instruction encodes, all SAME"

.PHONY: test-om-spec-encoder
ifeq ($(CUDAPP_OMC_HOST),1)
# On a non-AArch64 host the native AArch64 function-byte probe cannot execute; the script exits 3 (NOT_RUN) for
# it unless told the row is accepted as NOT_RUN, which this target does there and the log says so.
test-om-spec-encoder:
	OMEGA_BRIDGE="$(OMEGA_BRIDGE)" $(if $(CUDAPP_HOST_IS_AARCH64),,BW_SPEC_NATIVE_PROBE_NOT_RUN_OK=1) sh tools/test_bw_spec_encoder.sh
else
test-om-spec-encoder:
	@echo "test-om-spec-encoder: NOT_RUN (requires the pinned AArch64 omc and nvdisasm; set OMEGA_OMC to a pin that executes on this host; not part of the portable host suite)"
	@exit 2
endif

test-om-units:
	OMEGA_OMC="$(OMEGA_OMC)" OMEGA_BRIDGE="$(OMEGA_BRIDGE)" sh om/units.sh --check

# Each mutant is one edit of the .om source, compiled by the pinned omc into its own binding; the hosted
# fixture test linked with it must fail the golden words or differ from the C-encoded freestanding dump.
OM_ENC_MUTANTS = 's/^ret or base shl rb o 1 16$$/ret or base shl rb o 1 17/' \
	's/^p = 262144$$/p = 0/' \
	's/^const NOPC ulong 1032192$$/const NOPC ulong 1032193/'
test-om-encoder-mutants: build/test_codegen_freestanding
	@[ -x "$(OMEGA_OMC)" ] && [ -f "$(OMEGA_BRIDGE)/src/toc.om" ] && [ -f "$(OMEGA_BRIDGE)/src/freest.om" ] || { echo "NOT_RUN: set pinned OMEGA_OMC and OMEGA_BRIDGE"; exit 3; }
	@mkdir -p $(OM_ENC_DIR)
	./build/test_codegen_freestanding --dump > $(OM_ENC_DIR)/c-dump.txt
	@n=0; for m in $(OM_ENC_MUTANTS); do n=$$((n+1)); d=$(OM_ENC_DIR)/mutant$$n; rm -rf $$d; mkdir -p $$d; \
	sed "$$m" om/encoder/bw_enc_sm121.om > $$d/bw_enc_sm121.om; \
	if cmp -s $$d/bw_enc_sm121.om om/encoder/bw_enc_sm121.om; then echo "MUTANT $$n BROKEN: '$$m' changed nothing"; exit 1; fi; \
	OMEGA_UNITS=$$(pwd)/$$d/bw_enc_sm121.om OMEGA_UNITS_DEST=$$d OMEGA_OMC="$(OMEGA_OMC)" \
		OMEGA_BRIDGE="$(OMEGA_BRIDGE)" sh om/units.sh > /dev/null || exit 1; \
	$(OM_ENC_BUILD) -o $$d/codegen tests/test_omega_codegen_freestanding.c $(OM_ENC_SRCS) $$d/cudapp_om_bw_enc_sm121.c || exit 1; \
	g=0; $$d/codegen --golden > $$d/golden.log 2>&1 || g=1; \
	c=0; $$d/codegen --dump > $$d/dump.txt 2>&1; cmp -s $$d/dump.txt $(OM_ENC_DIR)/c-dump.txt || c=1; \
	if [ $$g = 0 ] && [ $$c = 0 ]; then echo "MUTANT $$n SURVIVED: '$$m'"; exit 1; fi; \
	echo "mutant $$n red ('$$m'): golden words $$( [ $$g = 1 ] && echo FAIL || echo pass ), dump vs C $$( [ $$c = 1 ] && echo DIFFERS || echo same )"; \
	done
