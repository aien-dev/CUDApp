# A/B ab-1 result: gemm_ex (B) against omega_gpu_gemm_ld (A), served shapes, GB10

Plan: `AB-PLAN.md` (committed 1da99c1 before the run; sha256 in `PLAN-SHA256SUMS`). One run, no re-runs, no selection.

- Hold: GPU-START 2026-10-27T08:22:44Z, GPU-END 08:23:52Z, `hold.sh ab-1` exit 0; 127.0.0.1:18091 /health ok before and after; compute apps before and after: only pid 79557 (`ab-1/hold.log`).
- Library: `libomega_gpu.a` built at 90c3f89 (sha256 6057d8ee13cb4d503e2edd92812af3a341e6c0ca043abb94bc418445dfeeb4be); driver binary sha256 3f1e9fe1ca0039f25c27a7b09d32b29747abde83d8f56faf825eda4a1dc1f8d6.
- Command: `hold.sh ab-1 <logdir> taskset -c 10-14 /tmp/gex2ab/gemm_ex_ab --rounds 20 --seed 20261027 --spin-us 300000`. Host load at start 15.39 (other agents).
- Correctness: 378 pairs, 0 gate FAIL, every pair bitwise=1. Negative control `mutant=1 cell="expert down"`: bitwise=0, ref_bad_b=59, verdict RED (expected). `run: PASS`.
- Timed rows: 360 warm=False pairs counted (20 per cell). Discarded for in_flight != 0 on 18091 (per plan): bf16 2048x16384x2048 x2, bf16 512x2048x4096, e4m3 16x12288x2048, bf16 2048x2048x4096, e4m3 2048x2048x4096. Kept pairs per cell: 18 to 20 (column n); every cell is above the plan's 10-pair floor.
- Statistics: `ab-1/stats.wl` evaluated by WolframLanguageEvaluator; full output `ab-1/wolfram-stats.txt`. r = Median[B]/Median[A]; p = SignedRankTest of the per-pair log ratios (information only, per plan).

## Verdict: gate FAIL (plan gate r <= 1.10 in every cell). 18/18 cells REGRESSION.

| cell | n | median A ms | median B ms | r | median paired B/A | A min-max ms | B min-max ms | verdict | p |
|---|---|---|---|---|---|---|---|---|---|
| e4m3 64x1024x2048 | 20 | 1.323 | 3.583 | 2.708 | 2.639 | 1.216-1.721 | 3.090-4.476 | REGRESSION | 0.0001 |
| bf16 16x12288x2048 | 20 | 20.576 | 60.467 | 2.939 | 2.945 | 19.621-22.638 | 57.724-78.011 | REGRESSION | 0.0001 |
| bf16 2048x12288x2048 | 20 | 85.601 | 228.112 | 2.665 | 2.653 | 83.200-93.735 | 220.453-238.201 | REGRESSION | 0.0001 |
| bf16 64x1024x2048 | 20 | 2.191 | 5.929 | 2.706 | 2.708 | 2.029-2.817 | 5.554-7.003 | REGRESSION | 0.0001 |
| e4m3 2048x2048x4096 | 19 | 17.416 | 56.576 | 3.248 | 3.248 | 17.105-18.108 | 51.952-60.309 | REGRESSION | 0.0001 |
| e4m3 512x8192x2048 | 20 | 16.546 | 47.877 | 2.894 | 2.877 | 15.875-19.575 | 43.463-60.204 | REGRESSION | 0.0001 |
| bf16 512x8192x2048 | 20 | 25.221 | 72.335 | 2.868 | 2.881 | 23.997-32.715 | 68.808-77.119 | REGRESSION | 0.0001 |
| bf16 2048x2048x4096 | 19 | 27.574 | 84.300 | 3.057 | 3.033 | 26.601-30.027 | 78.275-89.304 | REGRESSION | 0.0001 |
| e4m3 16x12288x2048 | 19 | 10.510 | 31.421 | 2.990 | 3.012 | 9.914-11.701 | 30.150-35.818 | REGRESSION | 0.0001 |
| e4m3 2048x16384x2048 | 20 | 85.910 | 229.599 | 2.673 | 2.671 | 83.570-94.662 | 221.490-236.192 | REGRESSION | 0.0001 |
| bf16 64x2048x512 | 20 | 1.563 | 3.654 | 2.338 | 2.456 | 1.332-1.872 | 3.300-4.509 | REGRESSION | 0.0001 |
| e4m3 1x8192x2048 | 20 | 6.830 | 21.031 | 3.079 | 3.042 | 6.417-7.456 | 19.315-22.682 | REGRESSION | 0.0001 |
| bf16 2048x16384x2048 | 18 | 113.308 | 293.618 | 2.591 | 2.581 | 110.237-126.012 | 280.347-297.927 | REGRESSION | 0.0002 |
| e4m3 64x2048x512 | 20 | 1.091 | 2.362 | 2.166 | 2.265 | 0.922-1.396 | 2.087-3.042 | REGRESSION | 0.0001 |
| bf16 512x2048x4096 | 19 | 12.152 | 36.575 | 3.010 | 3.053 | 11.629-13.458 | 34.259-52.837 | REGRESSION | 0.0001 |
| bf16 1x8192x2048 | 20 | 13.440 | 40.966 | 3.048 | 3.017 | 12.785-16.567 | 38.107-44.575 | REGRESSION | 0.0001 |
| e4m3 512x2048x4096 | 20 | 7.222 | 20.402 | 2.825 | 2.739 | 6.896-9.637 | 17.421-23.330 | REGRESSION | 0.0001 |
| e4m3 2048x12288x2048 | 20 | 66.261 | 180.277 | 2.721 | 2.737 | 64.025-73.710 | 173.630-198.688 | REGRESSION | 0.0001 |

Per the plan the measured overhead is recorded in API.md §9.8. Any change to gemm_ex that targets this overhead gets a new predeclared plan and run, reported beside this one; ab-1 stands as recorded.
