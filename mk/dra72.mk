# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# DRA-72 (P-DRA-4): CPU/GPU split feasibility benches. This fragment builds only the host tier that exists:
# cpubw (CPU bandwidth: plain read stream and a real FP8 e4m3 expert GEMV). The chip-tier benches (GPU-side wait on a
# CPU-written flag; FP8 GEMV load with a submission log) and the .om judge are not written yet; see the DRA-72 issue.
DRA72_CPUBW = $(OUT_DIR)/dra72_cpubw

# FMLAL (FEAT_FHM) is the f16 x f16 -> f32 product the expert GEMV needs; GB10 X925/A725 report asimdfhm.
$(DRA72_CPUBW): tools/dra72/cpubw.c
	@mkdir -p $(OUT_DIR)
	$(CC) -std=gnu11 -Wall -Wextra -Werror -O2 -march=armv8.2-a+fp16+fp16fml -ffp-contract=off -o $@ tools/dra72/cpubw.c -lpthread -lm

.PHONY: dra72 test-dra72-host
dra72: $(DRA72_CPUBW)
# Host tier, one core: decode check, its negative control (must fail), and a 2 s fp8gemv run on core 5.
test-dra72-host: $(DRA72_CPUBW)
	./$(DRA72_CPUBW) --check
	! ./$(DRA72_CPUBW) --check --mutate
	./$(DRA72_CPUBW) --mode fp8gemv --cores 5 --seconds 2 --mib 64 --out $(OUT_DIR)/dra72-host-fp8.txt
