# Optional native Omega host qualification; no GPU and no implicit compiler download.
OMEGA_NATIVE_COMPILER ?=
.PHONY: test-native-weightpack
test-native-weightpack:
	python3 native/weightpack/qualify.py --invalidate-only --build "$(OUT_DIR)/weightpack" --evidence "$(OUT_DIR)/weightpack/evidence"
	@test -n "$(OMEGA_NATIVE_COMPILER)" || { printf '%s\n' 'Set OMEGA_NATIVE_COMPILER to the pinned source-built omc' >&2; exit 64; }
	python3 native/weightpack/qualify.py --compiler "$(OMEGA_NATIVE_COMPILER)" --build "$(OUT_DIR)/weightpack" --evidence "$(OUT_DIR)/weightpack/evidence"
