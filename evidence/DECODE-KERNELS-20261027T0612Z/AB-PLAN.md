# DecodeKernels A/B plan (predeclared 2026-10-27, before any receipt timing)

Lane: DecodeKernels. Rows: CANDIDATES.md A2 (router grid) and A1 (expert grid). Status of everything below: PROPOSED until the
declared runs exist. Dev prechecks (precheck-a2-*, precheck-a2ab-*, precheck-a1-*) are NOT receipts and are excluded.

## Common protocol
- Binary: built from the PR head commit in a fresh clone under /home/drakestapleton/work/DecodeKernels/ (`nice -n 19 taskset -c 10-14 make`).
- Chip: one hold per declared run via /home/drakestapleton/handoffs/DecodeKernels/hold.sh (flock -w 900 /tmp/aien-gb10.lock,
  timeout 840 s, /health + compute apps before/after, BOARD GPU-START/END), beside the live 35B (pid 79557, untouched).
- Poller: /home/drakestapleton/handoffs/DecodeKernels/abrun.sh polls 127.0.0.1:18091/health (~30 ms + curl) for the whole run.
- Interleaving: the test binary shuffles arms (x shapes) x reps with the seeded LCG; the seed is fixed below. One declared
  run per row; no re-runs to pick a better result. A run that fails or is voided is reported as such, then re-run once
  with seed+1 and both are reported.
- Discard rule (tag.py, diagnostic): a sample is kept only if polls bracket [t0, t1], every poll from the last <= t0 to the
  first >= t1 shows in_flight == 0, and no poll gap in that span exceeds 200 ms. Discard counts are reported per arm.
- Minimum: >= 5 clean samples per arm (per shape). Fewer: that arm is NOT_RUN, not a loss or a win.
- Correctness voids: any sample with exact != 1 voids the whole run (the binaries stop at the first one).
- Statistics: medians of the clean samples and every ratio via Wolfram (WolframLanguageEvaluator `Median[...]`), queries
  quoted in the report. tag.py computes no statistic.
- Labels: kernel-level, chip-timed beside a serving 35B; per-step figures are MODELED (x40 layers). No tok/s claim.

## A2 router (perf/router-grid)
- Command: `build/gpu_router_test_q36 --ab "1,1,48,96" --reps 300 --seed 2710` (Q36 shape: 2048 hidden, 256 experts, top-8).
  Arm 0 and arm 1 are grid 1 (A/A); arm 2 = 48 CTAs, arm 3 = 96 CTAs. Sample value: chip_ns (one launch, GPU-timed).
- Bit-exact acceptance: every sample's top-8 ids and weights equal the grid-1 reference (exact == 1). Negative control:
  mutant 7 (merge disabled) at 48 and 96 must print "ids DIFFER" (caught) in the same run.
- A/A spread: d_AA = |M1 - M0| / M0. Verdict per split arm a: WIN iff exact and (M0 - Ma) / M0 > max(d_AA, 0.01);
  LOSS iff (Ma - M0) / M0 > max(d_AA, 0.01); else NO DIFFERENCE. The default changes only on WIN.
- Report: ms/layer (median) per arm and the MODELED step projection 40 x median.

## A1 expert grid (perf/expert-grid)
- Command: `build/gpu_fp8_api_test --grid-ab "0:0,0:0,4:0,5:0,8:0,11:0,16:0,11:24,16:16" --reps 40 --seed 2711`
  (CTAs per group : warps per CTA, 0 = default; default today = 6:32). Arms 0, 1 = default (A/A); 5:0 = 45 CTAs (the
  48-persistent single wave); 11:24 ~ 2 CTAs/SM at 24 warps; 16:16 = 144 CTAs of 512 threads (16x9@512).
- Shapes: 0 gate_up 9 of 144 x 1024 x 2048 (1 x row); 1 down 9 of 144 x 2048 x 512 (9 x rows); 2 o_proj 1 of 16 x
  2048 x 4096 at defaults in every arm (NEGATIVE CONTROL, must not move). 16 disjoint expert sets rotated (L2 defeated).
- Sample value: mean_ns = GPU-stamp span of a 32-launch PCAS list with WFI barriers / 32.
- Bit-exact acceptance: every list's outputs memcmp-equal arm 0's reference list; the reference list matches the double
  oracle (0 elements off); mutant 1 at arm 1 must differ (caught).
- A/A spread per shape s: d_AA(s) = |M1(s) - M0(s)| / M0(s). Combined per-layer score S(a) = M_a(gate_up) + M_a(down),
  d_AA(S) likewise.
- Verdict for arm a: WIN iff exact on all shapes, (S0 - Sa) / S0 > max(d_AA(S), 0.01), and neither expert shape is worse
  than arm 0 by more than max(d_AA(s), 0.01). Control: o_proj median of each arm within max(d_AA(o_proj), 0.02) of
  arm 0; a moved control voids the A1 verdict. The best WIN by (S0 - Sa) ships as the new grouped default (9 groups);
  no WIN: the default stays 6:32 and the PR carries only the sweep mode and its receipt.
- Report: per-shape medians, S, saving per layer and MODELED x40 step projection.
