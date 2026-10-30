AttnGeneral A/B plan v2 (predeclared before timing; 2026-10-27)

Why v2: v1 (AB-PLAN.md, sha256 1c1c8836...6cde) measured median B/A 1.061 over 10+10 clean runs (Wolfram Median;
MannWhitneyTest p 0.853), with per-run medians spanning 0.50-0.92 ms in both arms. v1's verdict rule (ratio <= 1.05) is
not met; v1 is reported as measured, not rewritten. v2 repeats with more runs to resolve the noise.

Same binaries, workload, discard rule and statistic as v1 (A = main c30c0be, B = feat/attn-general 4d51f91,
`--timing`, per-run median_ms, 18091 in_flight/gap discard). Changes: 40 pairs, alternating order, one hold.
Verdict: "no slower" if Wolfram Median(B)/Median(A) <= 1.05; otherwise reported as measured.
v1 and v2 are both reported; neither is dropped.
