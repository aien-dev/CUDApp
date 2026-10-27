# C3 receipt, part 1: sums, maximum, minimum and running totals of any length

Row C3 of `docs/OMEGA.md` asks for reductions and prefix sums (running totals) over every length from 1 to 2^28,
bit-exact against a host reference whose order of additions is frozen. The kernels come from Omega's own code
generator (`src/omega_gpu_scan.c`), with no CUDA. The order is written down in `src/omega_gpu_scan.h`; the reduction
order is the E1 contract (`docs/numeric/E1_REDUCTION_CONTRACT.md`).

| Log | What it checks | Result |
|---|---|---|
| 01-host-only | bad calls refused; all five kernels (and each planted fault) go through our code generator and encoder; the host reference gives exact answers on integer inputs and equals the E1 reduction reference bit for bit on mixed, signed-zero, NaN and infinite inputs; the frozen tree differs from a left-to-right sum | PASS, 133 checks |
| 02-sim | the real kernels in the warp simulator, both scheduling orders, every length 1 to 70 and the tile, block and grid-stride edges up to 270,000; three planted faults (a skipped tree step, a skipped scan step, a carry from the wrong tile) each caught | PASS, 139 checks |
| 03-chip-20261022-a7d6463b | every n from 1 to 2048, 2^k - 1, 2^k, 2^k + 1 up to 2^28, 64 random lengths, the three faults caught on the GB10 | PASS (8,665 checks), run 2026-10-22 07:03Z on `c3/gemm` a7d6463b (an older head, before this branch was rebased onto main 0d72560a); the kernel generator, its order contract and the test (`src/omega_gpu_scan.c`, `src/omega_gpu_scan.h`, `tests/gpu_scan_test.c`) are byte-identical between that head and this tip; `src/omega_numeric.c` and `src/omega_numeric_reduce.c` differ only by the licence header main added; `tests/bw_warp_sim.h` gained main's warp visit-order knob (simulator only, no chip code); binary sha256 c417f887 28d2ee77...; log recovered unpushed from the dead worker's worktree `omega-c5`-era sibling `omega-c3-scan` (durable copy `~/handoffs/evidence-L386-chip-20261022/`). Re-run on the train tip: row 04. |
| 04-chip-20261024-f16b56a58 | the same chip battery (every n to 2048, 2^k - 1 / 2^k / 2^k + 1 to 2^28, 64 random lengths, three faults caught) on the GB10 | PASS (8,665 checks), 2026-10-24 23:26Z on the L386 train tip f16b56a58 (c3/gemm over this branch over main 0d72560a), binary sha256 3400c91e 5d91d1a1... (byte-identical to this branch's own build), 39 s wall, spark-window 20261024T232617Z-b940cf (gb10 shared, value check), chip shared with the live 35B server on 18091 and two other /dev/nvidia0 holders. |

Command: `make test-gpu-scan` (host and simulator; `PHYSICS_DIR` must point at the physics checkout pinned in `physics.lock`). The chip run is `build/gpu_scan_test` with no flag, under the GB10 lock. 01 and 02 logs: 2026-10-24 23:10Z on this tip (c5cf3495b, binary sha256 3400c91e 5d91d1a1...), host 133 checks, simulator 139 checks.
