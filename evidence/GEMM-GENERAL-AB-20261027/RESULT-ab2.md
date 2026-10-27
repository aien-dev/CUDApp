# A/B ab-2 result: gemm_ex at 0d2b14a (B) against omega_gpu_gemm_ld (A), served shapes, GB10

Plan: `AB-PLAN-ab2.md` (committed df0e1ca before the run; sha256 in `PLAN-SHA256SUMS`). One run, no re-runs, no selection.
ab-1 (`RESULT.md`, 18/18 REGRESSION, gate FAIL) stands beside this.

- Chip tier of the new build first, own hold (GPU 08:45:18Z-08:45:29Z): `gemm_ex chip: PASS (901 checks)`, exit 0,
  18091 ok before and after (`ab-2/chip/`). A second compute app (pid 341161, BridgeAttach python) was on the GPU at that
  hold's start, gone at its end.
- Hold: GPU-START 2026-10-27T08:45:34Z, end 08:46:23Z, `hold.sh ab-2` exit 0; 18091 /health ok before and after; compute
  apps before and after: only pid 79557 (`ab-2/hold.log`). Host load at start 36.25 (other agents).
- Library sha256 3e960ea5a84ec283c93b77ea22880b1d7b284d4761bd3ff1280b2e3c5b68d866, driver binary sha256
  30874b38d19863babcf1f1993422a30d1e2684d94c524c24a19f4b841a81a8b3.
- Correctness: 378 pairs, 0 gate FAIL, every pair bitwise=1, route native, host_fixed 0, core_calls_b 1 per call.
  Negative control `mutant=1 cell="expert down"`: bitwise=0, ref_bad_b=59, RED (expected). `run: PASS`.
- Timed rows: 360 warm=False pairs; discarded for in_flight != 0 (per plan): e4m3 512x2048x4096, bf16 512x8192x2048, e4m3 64x2048x512, e4m3 2048x2048x4096, e4m3 512x2048x4096. Kept 18-20 per cell.
- Statistics: `ab-2/stats.wl` by WolframLanguageEvaluator, output `ab-2/wolfram-stats.txt`.

## Verdict: gate FAIL (r <= 1.10 in every cell), 5 cells over. 0/18 REGRESSION, 0/18 FASTER, 18/18 NO CHANGE.

Cells over 1.10: e4m3 64x1024x2048 (r 1.237); bf16 64x1024x2048 (r 1.169); e4m3 16x12288x2048 (r 1.101); e4m3 1x8192x2048 (r 1.101); e4m3 64x2048x512 (r 1.203). r over all cells 0.94-1.24 (ab-1: 2.17-3.25).

| cell | n | median A ms | median B ms | r | median paired B/A | A min-max ms | B min-max ms | verdict | p |
|---|---|---|---|---|---|---|---|---|---|
| e4m3 64x1024x2048 | 20 | 1.242 | 1.536 | 1.237 | 1.266 | 1.174-1.645 | 1.283-1.817 | NO CHANGE | 0.0105 |
| bf16 16x12288x2048 | 20 | 20.388 | 22.303 | 1.094 | 1.105 | 19.357-33.944 | 21.781-25.050 | NO CHANGE | 0.0018 |
| bf16 2048x12288x2048 | 20 | 84.511 | 86.909 | 1.028 | 1.027 | 83.253-88.353 | 84.593-94.543 | NO CHANGE | 0.0006 |
| bf16 64x1024x2048 | 20 | 2.110 | 2.467 | 1.169 | 1.201 | 1.966-2.509 | 2.210-2.786 | NO CHANGE | 0.0105 |
| e4m3 2048x2048x4096 | 19 | 17.069 | 17.876 | 1.047 | 1.032 | 16.583-17.915 | 17.249-20.798 | NO CHANGE | 0.0002 |
| e4m3 512x8192x2048 | 20 | 15.975 | 16.785 | 1.051 | 1.051 | 15.535-23.880 | 16.295-18.709 | NO CHANGE | 0.0018 |
| bf16 512x8192x2048 | 19 | 24.383 | 25.683 | 1.053 | 1.060 | 23.820-27.318 | 25.123-26.431 | NO CHANGE | 0.0004 |
| bf16 2048x2048x4096 | 20 | 26.848 | 28.172 | 1.049 | 1.053 | 26.235-43.196 | 27.438-37.394 | NO CHANGE | 0.0145 |
| e4m3 16x12288x2048 | 20 | 10.218 | 11.246 | 1.101 | 1.123 | 9.889-11.319 | 10.797-13.373 | NO CHANGE | 0.0001 |
| e4m3 2048x16384x2048 | 20 | 84.985 | 87.606 | 1.031 | 1.025 | 83.047-87.688 | 85.067-90.696 | NO CHANGE | 0.0001 |
| bf16 64x2048x512 | 20 | 1.544 | 1.453 | 0.941 | 0.928 | 1.266-2.431 | 1.356-1.863 | NO CHANGE | 0.4898 |
| e4m3 1x8192x2048 | 20 | 6.682 | 7.356 | 1.101 | 1.099 | 6.406-7.007 | 6.948-13.178 | NO CHANGE | 0.0001 |
| bf16 2048x16384x2048 | 20 | 111.636 | 114.141 | 1.022 | 1.026 | 110.040-116.082 | 112.364-123.355 | NO CHANGE | 0.0005 |
| e4m3 64x2048x512 | 19 | 1.010 | 1.216 | 1.203 | 1.303 | 0.879-1.509 | 0.954-1.513 | NO CHANGE | 0.1031 |
| bf16 512x2048x4096 | 20 | 11.852 | 12.622 | 1.065 | 1.068 | 11.343-12.331 | 12.118-13.657 | NO CHANGE | 0.0001 |
| bf16 1x8192x2048 | 20 | 13.060 | 14.282 | 1.094 | 1.108 | 12.482-14.040 | 13.946-17.658 | NO CHANGE | 0.0001 |
| e4m3 512x2048x4096 | 18 | 6.950 | 7.394 | 1.064 | 1.046 | 6.701-7.507 | 7.051-16.979 | NO CHANGE | 0.0011 |
| e4m3 2048x12288x2048 | 20 | 64.621 | 65.969 | 1.021 | 1.022 | 62.879-68.432 | 64.673-81.482 | NO CHANGE | 0.0038 |

The remaining overhead is largest in relative terms at the small cells (about 0.1-0.35 ms per call: the host inf/NaN scan
of A and B and the call's own buffers). Recorded in API.md section 9.8.
