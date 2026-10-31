# GEMM-TUNE-20261027: C3 general GEMM tile/raster tuning, paired A/B on the chip

Lane GemmTune, 2026-10-27. Arm A = `main` 20bd490 (merge of #112). Arm B = `perf/gemm-tiles` af9c854 (this branch's code
commit; this evidence directory is added on top and changes no code). Chip identity is not claimed.

## Change (arm B), all in `src/omega_gpu_gemm.{c,h}` and `tests/gpu_gemm_test.c`

1. Packed scratch is GPU-cached, persistent and grow-only (`packed_locked`). The packs write device-private fragments, so
   caching them is safe; A, B and C stay uncached. `AIEN_OMEGA_GEMM_PACK_CACHED=0` restores the uncached scratch (control).
2. Banded CTA raster for MAIN: `band` (0 or a power of two up to 32; 3 and 64 refused by `cfg_ok`, host-tested). With band > 1
   the grid is `(band*gy, ceil(gx/band))`, tile column = cy*band + (cx & (band-1)), row = cx >> log2(band), and CTAs past the
   tile columns EXIT (CTA-uniform; MAIN has no barrier). MAIN gets one more argument word (tile columns, `OMEGA_GPU_GEMM_ARGS_WORDS` 10).
3. `main_regs()` with `MAIN_REGS_DEFAULT 100` and the hardware GPR-limit opt-in, so rt 4 configurations are accepted.
4. Default configuration: rt 2, ct 4, wm 2, wn 4, band 8 (stall unchanged: 4 bf16, 2 e4m3). Chosen from the EXPLORATORY holds in
   `exploration/` (never verdict-bearing).

Kernel identity (AB-PLAN amendment 2): MAIN issues HMMA (bf16) and native QMMA (e4m3, via the QMMA patch); no shared memory
is staged, so the 16 KB QMD shared-memory cap does not bind this kernel.

## Gates at arm B (exact head af9c854)

| what | result | log |
|---|---|---|
| `make test-gpu-gemm` (simulator) | `C3 gemm sim: PASS (239 checks)` | `00-prep.log` |
| `build/gpu_gemm_test --host-only` | `C3 gemm host: PASS (209 checks)` (incl. band 3 / band 64 refused) | RESUME / rerun: one command |
| chip `build/gpu_gemm_test --ceiling 120.58,238.65`, beside the live 35B | `C3 gemm chip: PASS (218 checks)`, 208 shapes x both types x 4 layouts, 6/6 chip red checks caught (bf16 and e4m3 mutants 1-3, 4095 wrong each) | `01-gate/run.log`, `01-gate/hold.log` |
| chip gate again after `git merge origin/main` (#113, #119), head eba7c0f, library rebuilt with `make -B` | `C3 gemm chip: PASS (218 checks)`, 6/6 chip red checks caught, tc_peak and qmma_probe exit 0 | `05-gate-merged-eba7c0f/` |
| ceilings, 01-gate hold | best `gpu_tc_peak_test` 119.66 TFLOP/s (bf16), best `gpu_qmma_probe_test` 239.05 TFLOP/s (e4m3) | `01-gate/run.log` |
| control: `AIEN_OMEGA_GEMM_PACK_CACHED=0` | chip PASS 218; 4096^3 MAIN 18.77 ms bf16 / 9.80 ms e4m3 against 4.77 / 2.96 ms cached (the switch reproduces the #112 baseline) | `02-control-uncached/` |

Every hold: `flock -w 900 /tmp/aien-gb10.lock`, `timeout 840`; `hold.log` has 18091 health and compute apps before and after
(only pid 79557 every time, status ok).

## A/B (plan `AB-PLAN.md`, frozen before timing; sha256 history in `AB-PLAN.sha256*`)

Two runs, both reported. Primary metric `main_ns`, secondary `main_ns + pack_ns`. r = median_B / median_A. FASTER: r <= 0.95
and max_B < min_A. REGRESSION: r >= 1.05 and min_B > max_A. Otherwise NO CHANGE. Samples with 18091 `in_flight` > 0
before or after the call are discarded.

- **Main run** (plan as written, 7 rounds, driver at `nice -n 19`, spin 0): `03-ab-r*`. 476 samples, 25 discarded, 0 failed,
  every cell >= 5 kept. Verdicts on main_ns: 9 FASTER, 25 NO CHANGE, 0 REGRESSION; main+pack: 4 FASTER, 30 NO CHANGE,
  0 REGRESSION. Geometric mean of r: 0.759169 (main), 0.842637 (main+pack). Full table: `ab-wolfram-output.txt`.
  Finding: `main_ns` is host wall time from submit to marker. With spin 0 the wait is a 50 us sleep-poll, and at nice 19 with load1
  16-34 many samples of both arms sit at about 3.0 ms whatever the shape. Cells under a few ms are not resolved by this run.
- **Run S** (AB-PLAN amendment 3, declared before it ran: same arms, cells, rules; driver v2 `--spin-us 300000` at default
  nice): `04-abS-r*`. Rounds 1-7 left e4m3 512x8192x2048 with 4 kept B samples, so per plan line 26 rounds 8-14 ran too.
  952 samples, 114 discarded, 0 failed, every cell >= 10 kept. Verdicts on main_ns: **25 FASTER, 9 NO CHANGE, 0 REGRESSION**;
  main+pack: 14 FASTER, 20 NO CHANGE, 0 REGRESSION. Geometric mean of r: **0.3456** (main), 0.5647 (main+pack).
  Full table: `abS-wolfram-output.txt`.

All medians, ratios, TFLOP/s and ceiling fractions come from WolframLanguageEvaluator. The scripts are `ab-wolfram-main.wl`
and `ab-wolfram-S.wl`; their inputs are the sample lists `ab-cells.wl` / `abS-cells.wl`, extracted by `extract.sh` (bookkeeping
only). The key expressions:
`r = Median[b]/Median[a]`; `TFLOP/s = 2 M N K / Median[ns] / 1000`; `fraction = TFLOP/s / ceil` with
`ceil = <|"bf16" -> 119.66, "e4m3" -> 239.05|>`; `GeometricMean[N[Median[B]/Median[A]] & /@ data]`.

Table (run S columns first; main-run r and verdicts at the right; times in us; TFLOP/s on main_ns; B / ceiling uses this
session's ceilings 119.66 / 239.05):

| type | M x N x K | lay | S kept A/B | S med A us | S med B us | S r | S verdict | S TFLOP/s A | S TFLOP/s B | B / ceiling | S r (main+pack) | S verdict (main+pack) | main-run r | main-run verdict | main-run r (main+pack) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 1x8192x2048 | 2 | 12/13 | 255.9 | 98.5 | 0.385 | NO CHANGE | 0.13 | 0.34 | 0.0028 | 0.925 | NO CHANGE | 0.861 | NO CHANGE | 0.507 |
| bf16 | 16x12288x2048 | 2 | 12/13 | 461 | 260.7 | 0.566 | NO CHANGE | 1.75 | 3.09 | 0.0258 | 0.784 | NO CHANGE | 2.035 | NO CHANGE | 1.155 |
| bf16 | 64x1024x2048 | 2 | 13/12 | 111.9 | 29.8 | 0.266 | FASTER | 2.4 | 9.02 | 0.0754 | 1.52 | NO CHANGE | 0.599 | NO CHANGE | 0.671 |
| bf16 | 64x2048x512 | 2 | 12/13 | 75.8 | 24.3 | 0.321 | FASTER | 1.77 | 5.52 | 0.0461 | 0.805 | NO CHANGE | 2.377 | NO CHANGE | 1.354 |
| bf16 | 64x8192x8192 | 0 | 13/11 | 1457.3 | 721.5 | 0.495 | NO CHANGE | 5.89 | 11.91 | 0.0995 | 0.858 | NO CHANGE | 0.497 | NO CHANGE | 1.129 |
| bf16 | 512x2048x4096 | 2 | 12/13 | 1684.4 | 376.9 | 0.224 | FASTER | 5.1 | 22.79 | 0.1904 | 0.422 | FASTER | 1.337 | NO CHANGE | 1.66 |
| bf16 | 512x8192x2048 | 2 | 12/13 | 2869.2 | 849.3 | 0.296 | FASTER | 5.99 | 20.23 | 0.1691 | 0.417 | NO CHANGE | 0.954 | NO CHANGE | 0.903 |
| bf16 | 1000x999x1001 | 0 | 14/12 | 461.8 | 221 | 0.479 | FASTER | 4.33 | 9.05 | 0.0756 | 0.771 | NO CHANGE | 0.825 | NO CHANGE | 1.532 |
| bf16 | 1024x1024x1024 | 0 | 14/11 | 420.2 | 168.7 | 0.401 | NO CHANGE | 5.11 | 12.73 | 0.1064 | 0.883 | NO CHANGE | 2.012 | NO CHANGE | 0.727 |
| bf16 | 2048x2048x2048 | 0 | 12/12 | 3045.2 | 756.2 | 0.248 | NO CHANGE | 5.64 | 22.72 | 0.1899 | 0.355 | NO CHANGE | 0.412 | NO CHANGE | 0.699 |
| bf16 | 2048x2048x4096 | 2 | 11/12 | 6187.2 | 1108.5 | 0.179 | NO CHANGE | 5.55 | 31 | 0.259 | 0.251 | NO CHANGE | 0.231 | FASTER | 0.664 |
| bf16 | 2048x12288x2048 | 2 | 14/12 | 16447.3 | 4216.3 | 0.256 | FASTER | 6.27 | 24.45 | 0.2043 | 0.293 | FASTER | 0.266 | FASTER | 0.502 |
| bf16 | 2048x16384x2048 | 2 | 12/12 | 22270.5 | 5480.2 | 0.246 | FASTER | 6.17 | 25.08 | 0.2096 | 0.277 | FASTER | 0.261 | FASTER | 0.369 |
| bf16 | 3000x5000x700 | 0 | 13/12 | 4052 | 1838.2 | 0.454 | FASTER | 5.18 | 11.42 | 0.0955 | 0.51 | FASTER | 0.709 | NO CHANGE | 0.856 |
| bf16 | 4096x4096x4096 | 0 | 13/12 | 25170.3 | 4694.4 | 0.187 | FASTER | 5.46 | 29.28 | 0.2447 | 0.226 | FASTER | 0.192 | FASTER | 0.452 |
| bf16 | 8192x64x8192 | 0 | 12/13 | 1364.3 | 711 | 0.521 | FASTER | 6.3 | 12.08 | 0.101 | 0.837 | NO CHANGE | 1.785 | NO CHANGE | 1.445 |
| bf16 | 8192x8192x8192 | 0 | 12/12 | 222928.8 | 35308.6 | 0.158 | FASTER | 4.93 | 31.14 | 0.2602 | 0.176 | FASTER | 0.158 | FASTER | 0.186 |
| e4m3 | 1x8192x2048 | 2 | 12/13 | 127.9 | 32.2 | 0.252 | FASTER | 0.26 | 1.04 | 0.0044 | 0.783 | NO CHANGE | 0.873 | NO CHANGE | 0.492 |
| e4m3 | 16x12288x2048 | 2 | 12/12 | 239.1 | 107.6 | 0.45 | FASTER | 3.37 | 7.48 | 0.0313 | 0.896 | NO CHANGE | 1.919 | NO CHANGE | 1.073 |
| e4m3 | 64x1024x2048 | 2 | 12/13 | 75.2 | 22.4 | 0.298 | FASTER | 3.57 | 11.98 | 0.0501 | 1.118 | NO CHANGE | 1.643 | NO CHANGE | 0.397 |
| e4m3 | 64x2048x512 | 2 | 13/12 | 53.4 | 23.3 | 0.436 | NO CHANGE | 2.51 | 5.76 | 0.0241 | 1.266 | NO CHANGE | 3.927 | NO CHANGE | 3.713 |
| e4m3 | 64x8192x8192 | 0 | 10/13 | 694 | 382.8 | 0.552 | NO CHANGE | 12.38 | 22.44 | 0.0939 | 0.933 | NO CHANGE | 0.707 | NO CHANGE | 1.177 |
| e4m3 | 512x2048x4096 | 2 | 13/12 | 859.2 | 256.3 | 0.298 | FASTER | 10 | 33.52 | 0.1402 | 0.527 | FASTER | 0.352 | NO CHANGE | 0.931 |
| e4m3 | 512x8192x2048 | 2 | 12/11 | 1557.7 | 642 | 0.412 | FASTER | 11.03 | 26.76 | 0.1119 | 0.548 | NO CHANGE | 1.684 | NO CHANGE | 1.395 |
| e4m3 | 1000x999x1001 | 0 | 13/12 | 330 | 211.8 | 0.642 | NO CHANGE | 6.06 | 9.44 | 0.0395 | 1.118 | NO CHANGE | 1.741 | NO CHANGE | 0.949 |
| e4m3 | 1024x1024x1024 | 0 | 12/12 | 265.7 | 146.1 | 0.55 | FASTER | 8.08 | 14.7 | 0.0615 | 0.925 | NO CHANGE | 4.204 | NO CHANGE | 1.097 |
| e4m3 | 2048x2048x2048 | 0 | 13/12 | 1612 | 590.1 | 0.366 | FASTER | 10.66 | 29.11 | 0.1218 | 0.522 | FASTER | 1.105 | NO CHANGE | 3.894 |
| e4m3 | 2048x2048x4096 | 2 | 12/11 | 3020.6 | 778.9 | 0.258 | FASTER | 11.38 | 44.11 | 0.1845 | 0.374 | FASTER | 0.247 | NO CHANGE | 0.696 |
| e4m3 | 2048x12288x2048 | 2 | 13/12 | 8366.6 | 3257 | 0.389 | FASTER | 12.32 | 31.65 | 0.1324 | 0.425 | FASTER | 0.518 | FASTER | 1.252 |
| e4m3 | 2048x16384x2048 | 2 | 14/12 | 11225.1 | 4159.9 | 0.371 | FASTER | 12.24 | 33.04 | 0.1382 | 0.41 | FASTER | 0.404 | FASTER | 0.614 |
| e4m3 | 3000x5000x700 | 0 | 12/13 | 2765.2 | 1759.6 | 0.636 | FASTER | 7.59 | 11.93 | 0.0499 | 0.681 | FASTER | 1.002 | NO CHANGE | 0.955 |
| e4m3 | 4096x4096x4096 | 0 | 14/12 | 11345.3 | 2990.9 | 0.264 | FASTER | 12.11 | 45.95 | 0.1922 | 0.313 | FASTER | 0.286 | FASTER | 0.617 |
| e4m3 | 8192x64x8192 | 0 | 13/12 | 677 | 413.5 | 0.611 | FASTER | 12.69 | 20.77 | 0.0869 | 0.801 | NO CHANGE | 1.175 | NO CHANGE | 0.89 |
| e4m3 | 8192x8192x8192 | 0 | 12/11 | 106093.6 | 19344.8 | 0.182 | FASTER | 10.36 | 56.84 | 0.2378 | 0.203 | FASTER | 0.201 | FASTER | 0.244 |

Best ceiling fractions of arm B (run S): bf16 8192^3 31.14 TFLOP/s = 0.2602 of 119.66; e4m3 8192^3 56.84 TFLOP/s = 0.2378 of
239.05. Arm A, same cells: 4.93 and 10.36 TFLOP/s. The decode-shaped cells (M = 1, 16, 64) stay at 0.3-12 TFLOP/s:
they are bandwidth and launch bound, and a tile change does not move them toward the tensor ceiling.

Observed, not a verdict: on four small cells (bf16 64x1024x2048, e4m3 64x1024x2048, e4m3 64x2048x512, e4m3 1000x999x1001)
run S's main+pack median is higher in B (r 1.118-1.52; NO CHANGE by the rule, ranges overlap). `pack_ns` there is bimodal
(about 85-120 us or 250-300 us) in both arms, and B lands in the upper mode more often. [INFERENCE] The likely cause is the
session's L2_FLUSH_DIRTY tail, which now writes back the cached scratch lines the packs dirtied. That is not measured here.

## Not claimed

- No end-to-end serving claim: the 35B does not call this kernel.
- This is the CUDApp chip correctness tier plus a performance A/B. It is EXERCISED on the chip beside a live server, not a
  full hardware qualification.
- Timing is host wall time per launch (submit to marker), not GPU timestamps.

## Files

`prep.sh` (clones, branch, builds), `gate.sh`, `hold.sh` (lock, health, BOARD lines, window guard), `ab.sh` / `ab-s.sh`,
`gemm_ab.c` (driver v2) and `gemm_ab.v1.c` (main run), `perf-gemm-tiles.patch` (the code change as applied),
`exploration/` (EXPLORATORY holds explore1-3 and the 05:54Z dev-tree precheck).
