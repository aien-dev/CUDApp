# GemmTune A/B plan (frozen before any GemmTune timing)

Lane GemmTune, 2026-10-27. Question: is the general-shape GEMM of CUDApp (`src/omega_gpu_gemm.c`, C3, PR #112) faster on
branch `perf/gemm-tiles` than on `main` after #112 merges, per shape and type, and what fraction of the measured tensor-core
ceiling does each reach.

## Arms

- **A**: `aien-dev/CUDApp` `origin/main` at the merge of #112 (exact sha recorded in the receipt).
- **B**: `perf/gemm-tiles` head (exact sha recorded in the receipt), branched from that same main.
- Build, both arms, identical recipe in fresh clones: `nice -n 19 taskset -c 10-14 make -j5 build/libomega_gpu.a`, then the
  driver `gemm_ab.c` compiled with the Makefile's CFLAGS (`-std=gnu11 -Wall -Wextra -Werror -D_GNU_SOURCE -O2 -ffp-contract=off`)
  against that arm's own `tests/gpu_gemm_test.c` (fill, compare, host model) and `build/libomega_gpu.a`. `PHYSICS_DIR` as the
  tree's default (the arm's own `nvrm/`, `m16/`). Same gcc for both.
- Driver: `gemm_ab.c`, sha256 recorded below; copied unchanged into the evidence folder.

## Design

- Paired, interleaved: R = 7 rounds. In each round both arms run, one process per arm per round, arm order per round drawn
  from a fixed seed (`seed 20261027`, round r: A first when `(r * 2654435761 + 20261027) mod 2 == 0`, else B first; the order
  is printed in the log).
- Within a process: one warm-up call per type (65x63x100, excluded and labelled `warm`), then every (type, shape) of the set
  once, in an order shuffled by `(seed, round)` -- the same order for both arms in a round.
- Each timed call: `in_flight` of `127.0.0.1:18091/health` read right before and right after. A sample with either value
  `> 0` or unreadable (`-1`) is DISCARDED; the count of discarded samples is reported per cell and in total.
  If a cell keeps fewer than 5 samples after 7 rounds, up to 7 more full rounds (both arms, same seed rule, rounds 8..14)
  run; a cell still under 5 is reported INCONCLUSIVE.
- CPU: driver processes under `nice -n 19 taskset -c 10-14`. Load average recorded at the start of every process.
- Chip: every run under `flock -w 900 /tmp/aien-gb10.lock`, at most 15 minutes per hold, health and compute apps before and
  after (PROTOCOL non-negotiables). The live 35B on 18091 keeps serving; that is noise, handled by the in_flight rule.

## Shape set (both types, 17 shapes, 34 cells)

Serving shapes of the 35B (hidden 2048; weights [out][in], so layout 2 "A row B col"):
1x8192x2048 (q+gate decode), 16x12288x2048 (GDN in_proj, 16 rows), 512x8192x2048, 512x2048x4096 (o_proj),
2048x12288x2048, 2048x2048x4096 (out_proj), 64x1024x2048 (expert gate+up), 64x2048x512 (expert down),
2048x16384x2048 (one 16384-column slice of lm_head; the full 248320 is past OMEGA_GPU_GEMM_MAX_DIM).
The #112 ladder, layout 0 "A row B row": 1024^3, 2048^3, 4096^3, 8192^3, 3000x5000x700, 64x8192x8192, 8192x64x8192,
1000x999x1001.

## Correctness gate

- Every timed call is compared bit for bit with the host model of the tensor-core steps (`compare()` of
  `tests/gpu_gemm_test.c`: full up to 2^27 multiply-adds, else 4096 sampled elements plus the last row and column) and every
  output word must be written (POISON check in the library). One wrong or failed call voids the run for that arm.
- Arm B additionally passes, at its exact head: `make test-gpu-gemm` (host + sim), and on the chip the full
  `build/gpu_gemm_test` (208 shapes, both types, four layouts) with all 6 chip red checks caught (2 types x 3 mutants).

## Metrics

- Primary, per cell: MAIN kernel time `main_ns` (median, min..max of kept samples), TFLOP/s = 2 M N K / median main_ns.
- Secondary: `main_ns + pack_ns` (kernel time with the two pack kernels). Reference only (no verdict): `call_ns`.
- Ceiling fraction of arm B: TFLOP/s / ceiling, ceiling = best BF16 point of `build/gpu_tc_peak_test` and best E4M3 point of
  `build/gpu_qmma_probe_test` measured in the same GemmTune session if a hold allows, else the #112 values 120.58 / 238.65
  TFLOP/s, labelled as such.
- Every median, ratio and mean is computed in Wolfram (WolframLanguageEvaluator), query quoted next to the number.

## Verdict rule (per cell, on the primary metric; repeated on the secondary)

ratio r = median_B / median_A (times).
- FASTER: r <= 0.95 and max_B < min_A.
- REGRESSION: r >= 1.05 and min_B > max_A.
- NO CHANGE: otherwise.
Every REGRESSION is reported as REGRESSION, also on the secondary metric. No cell is dropped. Informational only: the
geometric mean of r over the 34 cells.

## No re-run selection

One declared run. A re-run only for a documented machine fault (hold timeout, health failure, a chip error unrelated to the
arms); both runs are then reported.

## Exploration (EXPLORATORY, never verdict-bearing)

Before the A/B, tile choices for B's defaults are explored on a 5-shape set (`--set explore`: 512x8192x2048 L2,
2048x12288x2048 L2, 2048^3 L0, 4096^3 L0, 64x8192x8192 L0) with `--cfg` overrides and the packed-scratch switch
`AIEN_OMEGA_GEMM_PACK_CACHED=0|1`, 2..3 passes each. These numbers choose B's defaults only; they are reported as
EXPLORATORY with their logs and are not compared against A.

## Driver

`gemm_ab.c` sha256 f6bb5b076bdc3162069edf35c5db574474c18c4c3541688a4f69521f8e1152ce

## Amendment 1 (2026-10-27, after exploration hold explore1, before any A/B timing)

The original text above hashed to 762a59a9997956614831d5e92d043a3a8433cc335ab84616ba64a97339625767 (AB-PLAN.sha256.v0).
Change: the driver accepts an optional sixth `--cfg` field `band` when compiled with `-DGEMM_AB_BAND` (exploration of
the banded MAIN grid on B-side trees only). The A/B runs pass no `--cfg`, so both arms run their library defaults, and
both arms compile the driver WITHOUT `-DGEMM_AB_BAND` (the same source compiles against main's header, checked against
1b5b4a6). Nothing else changes: shapes, rounds, discard rule, verdict rule, metrics.
New driver sha256 57aa161edb8d91d5f57d7503b61fd56561f8fdc370674568fd438e05c7080a01 (replaces f6bb5b07...).

## Amendment 2 (2026-10-27, documentation only, before any A/B timing)

Kernel identity (asked by Main after ExpertTensorCore/FINDINGS.md suspected the 6.79/14.05 figures came from a CUDA-core
FFMA kernel): the figures are from `build/gpu_gemm_test` at 7a43c9e (evidence/C3-GEMM/cudapp-20261027-7a43c9e/02-gemm-full.log),
which drives `omega_gpu_gemm` = `src/omega_gpu_gemm.c`, the kernel this plan measures. Read from the code at 1b5b4a6: its
MAIN kernel issues `BW_IR_HMMA_BF16` (`mma()`, line 52) for BF16 and the same op marked `OMEGA_QMMA_MARK` for E4M3, which
`omega_qmma_patch_code` rewrites to native QMMA (host test: "qmma patches 32/64" per E4M3 MAIN kernel, 0 for BF16). So both
types run on the tensor cores, not FFMA. It stages nothing in shared memory: the packs write operands in fragment order
and MAIN reads them with LDG.E.128 straight from L2/DRAM, so the 16 KB QMD shared-memory cap (finding 1 of FINDINGS.md) does
not bind this kernel; that cap and the T3 kernel (`omega_blackwell_codegen_tc.c`) are outside this A/B.
No change to arms, shapes, rounds, discard or verdict rules.

## Amendment 3 (2026-10-27 ~08:10Z, after the 7-round run, before any supplementary timing)

Finding from the 7-round run (receipts 03-ab-r*): `main_ns` is host wall time from submit to the completion marker, and
with the default spin of 0 the wait is the platform's 50 us sleep-poll. The driver ran at `nice -n 19` on cores 10-14 with
host load1 16-34, and many samples of both arms sit at 2.98-3.00 ms (or 3.0-4.1 ms) whatever the shape: delayed wake-ups,
not kernel time. Cells whose kernel is under a few ms are therefore not resolved by that run. Its verdicts stand as the
plan's result and are reported unchanged.

Supplementary run S (predeclared here, reported beside the main run, never replacing it):
- Same arms (A = main 20bd490, B = perf/gemm-tiles af9c854), same 34 cells, same 7 rounds, seed, arm-order rule, discard
  rule, metrics and verdict rule.
- Driver `gemm_ab.c` v2 (adds only `--spin-us`; v1 kept as gemm_ab.v1.c), built the same way in both arms, invoked with
  `--spin-us 300000` (the session spins on the uncached marker word for up to 300 ms before the sleep-poll, so every
  plan shape, the largest about 0.22 s, is timed by the spin) and at default nice (`taskset -c 10-14` kept), via
  `ab-s.sh`.
- Holds as before (under the lock, <= 14 min, health and compute apps before and after, no new hold while the last
  anchored `Main: WINDOW-*` line is WINDOW-START).
