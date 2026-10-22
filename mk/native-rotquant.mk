# Optional native Omega approximation experiment. No default-path or GPU changes.
OMEGA_NATIVE_COMPILER ?=
.PHONY: test-native-rotquant
test-native-rotquant:
	python3 native/rotquant/qualify_rotation.py --invalidate-only --build "$(OUT_DIR)/rotquant" --evidence "$(OUT_DIR)/rotquant/evidence"
	@test -n "$(OMEGA_NATIVE_COMPILER)" || { printf '%s\n' 'Set OMEGA_NATIVE_COMPILER to the pinned source-built omc' >&2; exit 64; }
	python3 native/rotquant/qualify_rotation.py --compiler "$(OMEGA_NATIVE_COMPILER)" --build "$(OUT_DIR)/rotquant" --evidence "$(OUT_DIR)/rotquant/evidence"
