# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# DRA-72 (P-DRA-4): CPU/GPU split feasibility benches.
#   dra72_cpubw            host: CPU bandwidth, plain read stream and a real FP8 e4m3 expert GEMV (tools/dra72/cpubw.c)
#   gpu_dra72_handoff_test chip: CPU -> GPU -> CPU hand-off through the host semaphore ACQUIRE/RELEASE (G4); --sim runs
#                          the same words through an in-process interpreter on the host
#   gpu_dra72_load_test    chip: timed FP8 GEMV load with a per-submission log (G2/G3)
#   dra72_judge            .om: reads those logs and decides G2/G3/G4 (tools/dra72/judge.om, pinned omc)
DRA72_CPUBW = $(OUT_DIR)/dra72_cpubw
DRA72_HANDOFF = $(OUT_DIR)/gpu_dra72_handoff_test
DRA72_LOAD = $(OUT_DIR)/gpu_dra72_load_test
DRA72_JUDGE = $(OUT_DIR)/dra72_judge
DRA72_OMC ?= $(OMEGA_MV)/seed/out/omc

# FMLAL (FEAT_FHM) is the f16 x f16 -> f32 product the expert GEMV needs; GB10 X925/A725 report asimdfhm.
$(DRA72_CPUBW): tools/dra72/cpubw.c
	@mkdir -p $(OUT_DIR)
	$(CC) -std=gnu11 -Wall -Wextra -Werror -O2 -march=armv8.2-a+fp16+fp16fml -ffp-contract=off -o $@ tools/dra72/cpubw.c -lpthread -lm
$(DRA72_HANDOFF): tests/gpu_dra72_handoff_test.c src/omega_gpu_session.h src/omega_blackwell_submit.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_dra72_handoff_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm
$(DRA72_LOAD): tests/gpu_dra72_load_test.c src/omega_gpu_fp8_api.h src/omega_gpu_cmdlist.h $(OUT_DIR)/libomega_gpu.a
	$(CC) $(CFLAGS) -ffp-contract=off -Itests -o $@ tests/gpu_dra72_load_test.c $(OUT_DIR)/libomega_gpu.a -lpthread -lm

.PHONY: dra72 dra72-compiler-check test-dra72-host
dra72-compiler-check:
	@test -x "$(DRA72_OMC)" || { echo "NOT_RUN: set DRA72_OMC to the tools/c5/omc.pin compiler"; exit 3; }
	@read repo rev unit hash < tools/c5/omc.pin; printf '%s  %s\n' "$$hash" "$(DRA72_OMC)" | sha256sum -c - >/dev/null || { echo "NOT_RUN: $(DRA72_OMC) is not the tools/c5/omc.pin compiler"; exit 3; }
$(DRA72_JUDGE): tools/dra72/judge.om | dra72-compiler-check
	@mkdir -p $(OUT_DIR)
	$(DRA72_OMC) < tools/dra72/judge.om > $@.tmp
	chmod +x $@.tmp
	mv $@.tmp $@
dra72: $(DRA72_CPUBW) $(DRA72_HANDOFF) $(DRA72_LOAD) $(DRA72_JUDGE)

# Host tier, one core (core 5 in the lane runs): cpubw decode check and its mutant (must fail), a 2 s fp8gemv run
# judged by G2 (the host number is far under 70 GB/s, so G2 must say FAIL: exit 1 is the expected result here), the
# hand-off protocol in --sim and its mutant (must fail; G4 on the sim log is INCONCLUSIVE, on the mutant log FAIL), a
# cpubw run stopped early by SIGTERM (the chip slices stop it that way; its log must still judge), and the judge's fixtures (tests/dra72/: pass, fail, malformed).
test-dra72-host: $(DRA72_CPUBW) $(DRA72_HANDOFF) $(DRA72_JUDGE)
	./$(DRA72_CPUBW) --check
	! ./$(DRA72_CPUBW) --check --mutate
	./$(DRA72_CPUBW) --mode fp8gemv --cores 5 --seconds 2 --mib 64 --out $(OUT_DIR)/dra72-host-fp8.txt
	./$(DRA72_JUDGE) g2 $(OUT_DIR)/dra72-host-fp8.txt; test $$? -eq 1
	./$(DRA72_HANDOFF) --sim --iters 2000 --chunk 500 --out $(OUT_DIR)/dra72-sim-handoff.txt
	! ./$(DRA72_HANDOFF) --sim --mutate --iters 2000 --chunk 500 --out $(OUT_DIR)/dra72-sim-handoff-mutant.txt
	./$(DRA72_JUDGE) g4 $(OUT_DIR)/dra72-sim-handoff.txt; test $$? -eq 4
	./$(DRA72_JUDGE) g4 $(OUT_DIR)/dra72-sim-handoff-mutant.txt; test $$? -eq 1
	./$(DRA72_CPUBW) --mode stream --cores 5 --seconds 30 --mib 64 --out $(OUT_DIR)/dra72-host-term.txt & p=$$!; sleep 1; kill -TERM $$p; wait $$p
	./$(DRA72_JUDGE) g2 $(OUT_DIR)/dra72-host-term.txt; test $$? -eq 1
	sh tests/dra72/judge_fixtures.sh ./$(DRA72_JUDGE)
