# TILELAB 2026-10-28: native .om tile planner pick vs CUDA++ default, BF16 GEMM on GB10

These files are raw receipts, written by `tests/gpu_tilelab_test.c`. The test was built with:
`gcc -O2 -std=gnu11 -Wall -Isrc -ffp-contract=off tests/gpu_tilelab_test.c build/libomega_gpu.a -lpthread -lm`.

| artifact | sha256 |
|---|---|
| measured binary (gcc line above) | ce89dde689057a2210033c86b683ef9d58d9f049f44ff18132890d27a8c0f5c6 |
| tests/gpu_tilelab_test.c | b09e0af86fd72a10d67154c6252ee7ebc420187495adec0d94baced245daec28 |
| build/libomega_gpu.a (CUDApp main 5b91c6a) | b6f9e5a9ee96bf10086af0d973549fe5b19a978998071b43751d0d02b2ffd4cd |
| planner: omega-mv seed/tilelab/tilelab.om (3f3b6ba71), compiled by omc 2c824805... | binary 4707a15eb3b35b508793936185873a9849b82a69ccfbac630f55d432e3b96959 |

## Arms

- `default`: the existing CUDA++ configuration, `rt2 ct4 wm2 wn4 band8`.
- `plan`: `rt2 ct4 wm4 wn4 band8`, the configuration `tilelab 1 M N K` picks for every shape below.
- `t64x64`: `rt1 ct2 wm4 wn4 band8`, a fixed-tile control.
- `t64x128`: `rt1 ct4 wm4 wn4 band8`, a fixed-tile control.

## Correctness (all runs)

- Every arm was checked against an f64 oracle, with 0 wrong elements and a worst err/bound of 0.018.
- The three chip mutants were caught with 4095 wrong each, and the rt=3 and wm=3 refusals held.

## 02: quiet A/B (the result)

- Window `20261028T005158Z-c4bcea`: gb10 exclusive, quiet exclusive.
- The resident 35B was stopped for the run and restored afterwards (see BOARD.md, 2026-10-28T00:5xZ).
- 25 interleaved rounds per arm.

Statistics are from Wolfram Language, via the hosted evaluator over direct MCP HTTP (not mounted in the measuring session). The test is `MannWhitneyTest[{default, plan}]`.

| shape | default p50 ms | plan p50 ms | plan/default | p |
|---|---|---|---|---|
| 1024^3 | 0.2041 | 0.2041 | 1.000 | 0.65 |
| 4096^3 | 4.815 | 4.722 | 0.981 | 0.11 |
| 2048x8192x2048 | 2.753 | 2.955 | 1.073 | 5.6e-6 |
| 8192^3 | 31.89 | 27.20 | 0.853 | 1.3e-9 |

- The planner pick is faster at 8192^3: the plan has an interquartile range of 26.9 to 27.5 ms, while the default is bimodal, with Q3 at 40.9 ms.
- It is slower at 2048x8192x2048.
- It shows no resolvable difference at 1024^3 or 4096^3.
- The planner's cost model counts bytes only, so it cannot see whatever makes `wm4` lose at 2048x8192x2048. That is a known model gap and is not hidden.
- The fixed-tile controls rank as the byte model predicts: plan, then t64x128, then t64x64. This holds at 3 of 4 shapes; at 2048x8192x2048, t64x64 beat t64x128.

## 01a/01b: smoke runs only, not an A/B

- Window `20261028T004445Z-a319ce`: gb10 shared, quiet none.
- The 35B was serving during these runs (in_flight 1, GPU 96% busy).
- They are kept for correctness only. Do not read their timings as a result.
