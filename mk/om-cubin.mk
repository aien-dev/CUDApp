# The sm_121 cubin packer in seeded Omega (om/cubin/cubin_elf.om), linked through the omega-bridge native binding
# src/omega_om_units.c that om/units.sh generates. The C writer in src/omega_cubin.c is migration-pending: the
# freestanding build and the oracle builds only (CUDAPP_MIGRATION_PENDING_C_CUBIN, CUDAPP_MIGRATION_PENDING_CUBIN_DIFF).
#   make test-om-cubin           (in `make test`, no Omega tools needed)
#       1. the cubin tests (gpu_artifact_export_test, native_artifact_manifest_test) through the .om packer;
#       2. the same two tests and the sweep tests/om_cubin_sweep_test.c built with CUDAPP_MIGRATION_PENDING_CUBIN_DIFF:
#          every call is packed both ways and aborts on any difference; the log must hold only SAME lines;
#       3. its red check: the sweep with one flipped image bit must abort.
#   make test-om-cubin-mutants   mutant .om packers, each compiled by the pinned omc, must turn the sweep red
#       (needs OMEGA_MV and OMEGA_BRIDGE_DIR as in mk/om-encoder.mk; without them exit 3, NOT_RUN, never PASS).
OM_CUBIN_DIR = $(OUT_DIR)/om-cubin
OM_CUBIN_BUILD = gcc -std=gnu11 -O2 -Wall -Wextra -Werror -pthread -Isrc
OM_CUBIN_DEPS = src/omega_cubin.c src/omega_cubin.h src/omega_om_units.c src/omega_native.h
OM_CUBIN_DIFF = -DCUDAPP_MIGRATION_PENDING_CUBIN_DIFF

.PHONY: test-om-cubin test-om-cubin-mutants
$(OM_CUBIN_DIR)/sweep_diff: tests/om_cubin_sweep_test.c $(OM_CUBIN_DEPS)
	@mkdir -p $(OM_CUBIN_DIR)
	$(OM_CUBIN_BUILD) $(OM_CUBIN_DIFF) -o $@ tests/om_cubin_sweep_test.c src/omega_cubin.c src/omega_om_units.c
$(OM_CUBIN_DIR)/sweep_diff_red: tests/om_cubin_sweep_test.c $(OM_CUBIN_DEPS)
	@mkdir -p $(OM_CUBIN_DIR)
	$(OM_CUBIN_BUILD) $(OM_CUBIN_DIFF) -DCUDAPP_CUBIN_DIFF_SELFTEST -o $@ tests/om_cubin_sweep_test.c src/omega_cubin.c \
		src/omega_om_units.c
$(OM_CUBIN_DIR)/artifact_diff: tests/gpu_artifact_export_test.c src/omega_native_artifact.c $(OM_CUBIN_DEPS) $(OUT_DIR)/libomega_gpu.a
	@mkdir -p $(OM_CUBIN_DIR)
	$(CC) $(CFLAGS) $(OM_CUBIN_DIFF) -o $@ tests/gpu_artifact_export_test.c src/omega_native_artifact.c src/omega_cubin.c \
		$(OUT_DIR)/libomega_gpu.a -lpthread -lm
$(OM_CUBIN_DIR)/manifest_diff: tests/native_artifact_manifest_test.c src/omega_native_artifact.c src/omega_blackwell_realize.c \
		src/sha256.c $(OM_CUBIN_DEPS)
	@mkdir -p $(OM_CUBIN_DIR)
	$(CC) $(CFLAGS) $(OM_CUBIN_DIFF) -ffunction-sections -fdata-sections -Wl,--gc-sections -o $@ \
		tests/native_artifact_manifest_test.c src/omega_native_artifact.c \
		src/omega_cubin.c src/omega_blackwell_realize.c src/sha256.c src/omega_om_units.c -lpthread

test-om-cubin: $(ARTIFACT_TEST) $(OUT_DIR)/native_artifact_manifest_test $(OM_CUBIN_DIR)/artifact_diff \
		$(OM_CUBIN_DIR)/manifest_diff $(OM_CUBIN_DIR)/sweep_diff $(OM_CUBIN_DIR)/sweep_diff_red
	rm -rf $(OM_CUBIN_DIR)/artifact-om $(OM_CUBIN_DIR)/artifact-diff $(OM_CUBIN_DIR)/diff.log
	./$(ARTIFACT_TEST) $(OM_CUBIN_DIR)/artifact-om
	./$(OUT_DIR)/native_artifact_manifest_test
	CUDAPP_CUBIN_DIFF_LOG=$(OM_CUBIN_DIR)/diff.log $(OM_CUBIN_DIR)/artifact_diff $(OM_CUBIN_DIR)/artifact-diff
	CUDAPP_CUBIN_DIFF_LOG=$(OM_CUBIN_DIR)/diff.log $(OM_CUBIN_DIR)/manifest_diff
	@a=$$(grep -c . $(OM_CUBIN_DIR)/diff.log); \
	CUDAPP_CUBIN_DIFF_LOG=$(OM_CUBIN_DIR)/diff.log $(OM_CUBIN_DIR)/sweep_diff || exit 1; \
	same=$$(grep -c ' SAME$$' $(OM_CUBIN_DIR)/diff.log); other=$$(grep -vc ' SAME$$' $(OM_CUBIN_DIR)/diff.log); \
	if [ "$$a" -eq 0 ] || [ "$$other" -ne 0 ]; then echo "FAIL cubin diff: $$a test calls, $$same SAME, $$other other lines"; exit 1; fi; \
	echo "cubin diff: .om and C packers identical (code, bytes, message) on $$same calls ($$a from the artifact tests, $$((same - a)) from the sweep; $$(grep -c '^cubin rc=0 ' $(OM_CUBIN_DIR)/diff.log) images)"
	@if $(OM_CUBIN_DIR)/sweep_diff_red > /dev/null 2>&1; then \
		echo "RED CHECK BROKEN: a flipped bit passed the cubin diff"; exit 1; \
	else echo "red check: one flipped image bit aborts the cubin diff, as it must"; fi

# Each mutant is one edit of the .om source, compiled by the pinned omc into its own binding; the sweep linked
# with it (diff mode) must abort or fail.
OM_CUBIN_MUTANTS = 's/^f = add f add 117 mul 3 n$$/f = add f add 118 mul 3 n/' \
	's/^const PBASE ulong 896$$/const PBASE ulong 892/' \
	's/^if gt get m 4 255$$/if gt get m 4 254/'
test-om-cubin-mutants:
	@[ -x $(OMEGA_MV)/seed/out/omc ] && [ -f $(OMEGA_BRIDGE_DIR)/src/native.om ] || { echo "NOT_RUN: set OMEGA_MV and OMEGA_BRIDGE_DIR"; exit 3; }
	@n=0; for m in $(OM_CUBIN_MUTANTS); do n=$$((n+1)); d=$(OM_CUBIN_DIR)/mutant$$n; rm -rf $$d; mkdir -p $$d; \
	sed "$$m" om/cubin/cubin_elf.om > $$d/cubin_elf.om; \
	if cmp -s $$d/cubin_elf.om om/cubin/cubin_elf.om; then echo "MUTANT $$n BROKEN: '$$m' changed nothing"; exit 1; fi; \
	OMEGA_UNITS=$$(pwd)/$$d/cubin_elf.om OMEGA_UNITS_DEST=$$d OMEGA_OMC=$(OMEGA_MV)/seed/out/omc \
		OMEGA_BRIDGE=$(OMEGA_BRIDGE_DIR) sh om/units.sh > /dev/null || exit 1; \
	$(OM_CUBIN_BUILD) $(OM_CUBIN_DIFF) -o $$d/sweep tests/om_cubin_sweep_test.c src/omega_cubin.c \
		$$d/omega_om_units.c || exit 1; \
	if $$d/sweep > $$d/sweep.log 2>&1; then echo "MUTANT $$n SURVIVED: '$$m'"; exit 1; fi; \
	echo "mutant $$n red ('$$m'): $$(grep -m1 -e 'CUBIN DIFF' -e FAIL $$d/sweep.log)"; \
	done
