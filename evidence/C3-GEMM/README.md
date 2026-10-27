# C3 receipt, part 2: matrix multiply of any shape on the tensor cores

Row C3 of `docs/OMEGA.md` asks for matrix multiply (GEMM) of any shape in BF16 and FP8, both layouts, at least 200 shapes
from 1 to 8192, correct, with speed as a fraction of the measured tensor-core ceiling. The kernels come from Omega's own
code generator (`src/omega_gpu_gemm.c`), no CUDA: two pack kernels put A and B (row- or column-major) into the tensor
cores' order with zeros past the edges, one multiply kernel runs the tensor cores (HMMA for BF16, QMMA for FP8).

"Correct" here is bit for bit: the reference is a host model of the same tensor-core steps whose rounding was measured
on this chip (`tests/bw_hmma_model.h`, `tests/bw_qmma_model.h`, both exact on 262,144 of 262,144 chip outputs), and a
separate check shows that model stays within one part in a million of the exact product.

| Log | What it checks | Result |
|---|---|---|
| 01-host-only | bad calls refused, tile shapes past the register budget refused; every kernel (both types, five tile shapes, each planted fault) through our code generator and encoder; the model against exact double-precision sums | PASS, 136 checks |
| 02-sim | the real kernels in the warp simulator (tensor-core ops computed with the chip models): ten shapes over every edge, all four layouts, three tile shapes, both types, every output compared; three planted faults (a dropped k step, no zeros past the edges, swapped columns) each caught on all 4,095 outputs | PASS, 154 checks |
| cudapp-20261027-loaded | preliminary chip runs from b2ed663 plus a harness patch, host load1 16 to 24 | PASS (218 checks); superseded by the final-head row below |
| cudapp-20261027-7a43c9e | final head of CUDApp#112 (7a43c9e, clean tree): 00 build on E-cores 10-14 (make exit 0, binary sha256s in the log); 01 tensor-core ceilings measured in the same hold (best points BF16 120.58 TFLOPS, FP8 238.65 TFLOPS, 0 wrong values); 02 all 208 shapes up to 8192, both types, all four layouts, every sampled output bit-exact, six planted faults caught on chip (six `chip red check: ... is caught` lines), rebuilt after a device close and reopen | PASS (218 checks, `C3 gemm chip: PASS`, `## gemm full exit 0`), 2026-10-27 02:47:28Z to 02:52:11Z, one flock hold on /tmp/aien-gb10.lock (283 s), load1 3.0 to 6.3; 35B on 18091 stopped (BOARD 01:12Z), logs end with an empty compute-apps list. Speed is the fraction of the ceiling above: best BF16 0.0563 (2048x2048x2048 A row B row, 6.79 TFLOPS; WolframLanguageEvaluator `N[6.79/120.58] -> 0.05631`), best FP8 0.0589 (8192x64x8192 A row B row, 14.05 TFLOPS; `N[14.05/238.65] -> 0.05887`); 8192 cubed runs at 0.042 (BF16, `N[5.09/120.58] -> 0.04221`) and 0.046 (FP8, 11.05 TFLOPS; `N[11.05/238.65] -> 0.04630`). The kernel is correct everywhere and slow: about 5% of the ceiling. |
| cudapp-20261027-d884212/02-gemm-full | CUDApp#112 at d884212 (c3/gemm-scan merged with main c30c0be, the carry chain), fresh clone, clean tracked tree, gpu_gemm_test sha256 e5c2945730bf9941...: all 208 shapes, both types, four layouts, six planted faults (six `chip red check: ... is caught` lines) | PASS (218 checks, `C3 gemm chip: PASS`, `## gemm full exit 0`), 2026-10-27 06:43:40Z to 06:44:44Z under /tmp/aien-gb10.lock (TrainCarry2), BESIDE the live 35B on 18091 (health ok before and after; compute apps after: 79557 only), load1 10.2. Correctness receipt only: the ceilings were not re-measured (`--ceiling` reuses the 7a43c9e values) and the speed figures in this log were taken beside a serving model, so they are not quoted. |

Command: `make test-gpu-gemm` (host and simulator). The chip run is `build/gpu_gemm_test` with no flag, under the GB10 lock. 01 and 02 logs: 2026-10-24 23:11Z on the stacked tip (gemm layer over the scan layer over main 0d72560a; binary sha256 0a624c25 a68cef88...; `PHYSICS_DIR` must point at the checkout pinned in `physics.lock`), host 136 checks, simulator 154 checks.
This is the general kernel; the prefill-roof lane (L403) tunes it toward the ceiling.
