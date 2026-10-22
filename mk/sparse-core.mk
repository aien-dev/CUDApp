# SPDX-License-Identifier: MIT
# Seed-native bounded sparse host library; intentionally separate from C suite.
.PHONY: test-sparse-core
test-sparse-core:
	@test -n "$(OMEGA_SEED_COMPILER)" || { printf '%s\n' 'Set OMEGA_SEED_COMPILER to the SHA-pinned Linux AArch64 omc (docs/CSR_TRANSPOSE.md).' >&2; exit 2; }
	python3 tests/libraries/test_sparse_core.py --compiler "$(OMEGA_SEED_COMPILER)" --out "$(OUT_DIR)/sparse-core" --mutants
