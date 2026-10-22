# SPDX-License-Identifier: MIT
# Optional Omega-native host suite: requires the qualified Linux AArch64 seed omc.
# Kept separate from the C host suite so other hosts do not silently skip it.
.PHONY: test-csr-transpose
test-csr-transpose:
	@test -n "$(OMEGA_SEED_COMPILER)" || { printf '%s\n' 'Set OMEGA_SEED_COMPILER to the pinned seed-born omc. See docs/CSR_TRANSPOSE.md.' >&2; exit 2; }
	python3 tests/libraries/test_csr_transpose.py --compiler "$(OMEGA_SEED_COMPILER)" --out "$(OUT_DIR)/csr-transpose" --mutants
