# SPDX-License-Identifier: MIT
# Explicit host-native milestone, not part of the GPU library archive.
# The seed source may be any omega git clone that contains the pinned commit.
LIBRARY_OUT ?= $(OUT_DIR)/fft-rng-native
.PHONY: libraries-native test-libraries-native
libraries-native:
	@test -n "$(OMEGA_SEED_SOURCE)" || { printf '%s\n' 'Set OMEGA_SEED_SOURCE to an omega git clone (see docs/FFT_RNG.md).'; exit 2; }
	python3 tools/libraries/build.py --omega-source "$(OMEGA_SEED_SOURCE)" --out "$(LIBRARY_OUT)"
test-libraries-native: libraries-native
	python3 tests/libraries/test_philox.py "$(LIBRARY_OUT)/philox"
	python3 tests/libraries/test_binary32.py "$(LIBRARY_OUT)/binary32"
	python3 tests/libraries/test_sparse.py "$(LIBRARY_OUT)/sparse"
	python3 tests/libraries/test_fft.py "$(LIBRARY_OUT)/fft"
	@set -e; for b in arith potrf npp blas1 idct; do "$(LIBRARY_OUT)/$$b"; done
	python3 tests/libraries/mutants.py "$(LIBRARY_OUT)"
