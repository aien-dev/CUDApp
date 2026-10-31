# GemmGeneral served-shape A/B: gemm_ex against the existing C3 entry (predeclared)

Written 2026-10-27 by GemmGeneral2 before any timed run. Committed to CUDApp branch `gemm-general/gemm-ex` and announced
on BOARD (with this file's sha256) before the hold starts. Nothing below is changed after the run.

## Question

At the nine served 35B GEMM shapes, does the general entry `omega_gpu_gemm_ex` (CUDApp#139) cost more wall time than the
existing entry `omega_gpu_gemm_ld` it calls, for the same host-memory inputs and outputs?

## Arms (one process, one build)

- Code under test: CUDApp `gemm-general/gemm-ex` at 90c3f89 (library code identical to fba0384, the commit of the chip
  receipts `evidence/GEMM-GENERAL-fba0384/`). `libomega_gpu.a` built with the Makefile CFLAGS
  (`-DCUDAPP_MIGRATION_PENDING_GEMM_EX=1`), sha256 6057d8ee13cb4d503e2edd92812af3a341e6c0ca043abb94bc418445dfeeb4be.
- Driver `gemm_ex_ab.c` sha256 eaec1adacd64eb718862cf4af5d835ba60a8a3120a684c27e76d03472d33f6bb, binary sha256
  3f1e9fe1ca0039f25c27a7b09d32b29747abde83d8f56faf825eda4a1dc1f8d6
  (`cc -std=gnu11 -Wall -Wextra -Werror -D_GNU_SOURCE -O2 -Isrc -DCUDAPP_MIGRATION_PENDING_GEMM_EX=1 -ffp-contract=off`).
- Arm A: `omega_gpu_gemm_ld(type, OMEGA_GPU_GEMM_B_COL, M, N, K, A, K, B, K, C, NULL)`.
- Arm B: `omega_gpu_gemm_ex` with op_a N, op_b T, the same type for A and B, C f32, alpha 1, beta 0, batch 1.
- Same host buffers, same input bits. A is M x K row-major, B is N rows of K (weights [out][in]).

## Cells (18)

Types bf16 and e4m3, times these shapes (M x N x K): 1x8192x2048 q+gate decode, 16x12288x2048 gdn in_proj b16,
512x8192x2048 q+gate p512, 512x2048x4096 o_proj p512, 2048x12288x2048 gdn in_proj p2048, 2048x2048x4096 out_proj p2048,
64x1024x2048 expert gate+up, 64x2048x512 expert down, 2048x16384x2048 lm_head slice. These are the nine GemmTune measured
(`handoffs/GemmTune/AB-PLAN.md`). Inputs: bf16 values k/128 for integer k in [-256, 256), e4m3 any non-NaN code, seeded
per shape.

## Run

- `hold.sh ab-1 <logdir> taskset -c 10-14 /tmp/gex2ab/gemm_ex_ab --rounds 20 --seed 20261027 --spin-us 300000`,
  one process, one hold (flock /tmp/aien-gb10.lock, at most 14 minutes, BOARD GPU-START/GPU-END, 18091 /health before and
  after, no hold while Main's last WINDOW line is WINDOW-START). Default nice.
- Round 0 is warm (labelled `warm=True`) and is not counted. Rounds 1..20 are timed (`warm=False`).
- Interleaving: each round runs all 18 cells in a seeded shuffled order; inside a cell the arm order is a seeded coin.
- One run. No re-run is chosen over another: if the run is cut short (timeout, hold refused), the rounds it finished are
  reported as they are and the missing ones are NOT_RUN.

## Correctness gate (any failure voids the run)

Every pair, warm included: both rc OK; C_B equal to C_A bit for bit; 64 sampled elements of C_A (the last element plus 63
seeded ones) within K 2^-22 sum|a b| of a double host reference; arm B's route NATIVE with host_fixed 0. Negative control
after the timed rounds: gemm_ex mutant 1 (drops the last k) on 64x2048x512 bf16 must be RED (bitwise differs and the
reference check fails on C_B while C_A passes). A green control fails the run.

## Kept pairs

A `warm=False` pair is kept unless its in_flight (the live 35B's /health) before or after the pair is non-zero or
unreadable (-1). A cell with fewer than 10 kept pairs is INCONCLUSIVE.

## Statistics (Wolfram, quoted with the query)

Per cell over kept pairs: median wall_a_ns, median wall_b_ns, r = median_B / median_A, min and max of each arm, the median
of the per-pair ratios wall_b / wall_a. Information only: Wilcoxon signed-rank test (`SignedRankTest`) of the per-pair
log ratios.

## Verdict per cell (fixed now)

- FASTER: r <= 0.95 and max_B < min_A.
- REGRESSION: r >= 1.05 and min_B > max_A. Reported as REGRESSION, whatever the cause.
- NO CHANGE: anything else.

## Lane gate

PASS if the correctness gate passes and every cell has r <= 1.10 (gemm_ex costs at most 10 % over the call it wraps at the
served shapes). Otherwise FAIL, with the failing cells listed; the measured overhead then goes into API.md section 9.8.

## NOT_RUN by design

- The tuned device-resident paths (fp8 GEMV `omega_gpu_fp8_gemv*`, grouped fp8 GEMM `omega_gpu_fp8gemm_*`, the prefill
  GEMMs `omega_gpu_pf_*`): they take device addresses and resident weights, so a host-memory arm against them would time
  uploads, not the GEMM. gemm_ex is not a serving path (API.md 9.8).
- Any served-model end-to-end timing: gemm_ex has no served caller.
