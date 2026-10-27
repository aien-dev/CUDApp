# GemmGeneral served-shape A/B, run ab-2: gemm_ex after the host-overhead fix (predeclared)

Written 2026-10-27 by GemmGeneral2 before any ab-2 timed run. Committed to CUDApp branch `gemm-general/gemm-ex` and
announced on BOARD (with this file's sha256) before the hold starts. Nothing below is changed after the run. ab-1
(`AB-PLAN.md`, `RESULT.md`, `ab-1/`) stands as recorded: 18/18 REGRESSION, lane gate FAIL. ab-2 is a new run of new
library code, reported beside ab-1, never in place of it.

## What changed since ab-1

`src/omega_gpu_gemm_ex.c` only (same entry, same signature, same results):
- the inf/NaN scan of A and B reads each operand once with a 128-bit word test instead of a per-element decode;
- the pack buffers and the double sum are allocated only when a path uses them;
- one k part without block scales skips the host double sum (`0 + P`, the same value);
- native route, one k part, alpha s 1, beta 0, C f32 with ldc n, n <= 16384, finite inputs: the C3 kernel writes C in
  place, one call per 16384 rows, no 4096 host tile and no host epilogue (the header's bit-for-bit-C3 case).

Host-only profile of the entry with a no-op core (`/tmp/gex2prof/prof.c`, information only, not a gate): bf16
2048x16384x2048 151.7 ms -> 4.4 ms of host work per call; bf16 1x8192x2048 24.6 -> 1.2 ms.

## Arms, cells, run, correctness gate, kept pairs, statistics, verdict rule, lane gate

Exactly as `AB-PLAN.md` (same 18 cells, same driver source `gemm_ex_ab.c` sha256
eaec1adacd64eb718862cf4af5d835ba60a8a3120a684c27e76d03472d33f6bb, same seed 20261027, 20 timed rounds, same
`--spin-us 300000`, same hold script `hold.sh` sha256 05169150bbbbe7d45d3c3098ab03aa48242a034132a895f8af0a0bbbe3ab0329,
same correctness gate including the mutant-1 RED control, same verdict thresholds and the same lane gate r <= 1.10 in
every cell), with these inputs:

- Code under test: CUDApp `gemm-general/gemm-ex` at 0d2b14a (`gemm_ex: host overhead`), library built with the
  Makefile CFLAGS into `/tmp/gex2ab2`: `libomega_gpu.a` sha256
  3e960ea5a84ec283c93b77ea22880b1d7b284d4761bd3ff1280b2e3c5b68d866.
- Driver binary `/tmp/gex2ab2/gemm_ex_ab` (same compile line as ab-1, linked against that library) sha256
  30874b38d19863babcf1f1993422a30d1e2684d94c524c24a19f4b841a81a8b3.
- Command: `hold.sh ab-2 <logdir> taskset -c 10-14 /tmp/gex2ab2/gemm_ex_ab --rounds 20 --seed 20261027 --spin-us 300000`.
- Before the timed run, under its own hold, the chip tier of the new build: `/tmp/gex2ab2/gpu_gemm_ex_test` (no flag) must
  PASS; a chip FAIL voids ab-2 (NOT_RUN, reported).

One run. No re-run is chosen over another. If ab-2's lane gate fails, the measured per-cell overhead goes into API.md
section 9.8.
