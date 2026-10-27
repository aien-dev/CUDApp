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
| (next) chip | 208 shapes up to 8192, both types, all four layouts; speed against 115.36 TFLOPS (BF16) and 237.52 TFLOPS (FP8) | waiting for a chip window |

Command: `make test-gpu-gemm` (host and simulator). The chip run is `build/gpu_gemm_test` with no flag, under the GB10 lock. 01 and 02 logs: 2026-10-24 23:11Z on the stacked tip (gemm layer over the scan layer over main 0d72560a; binary sha256 0a624c25 a68cef88...; `PHYSICS_DIR` must point at the checkout pinned in `physics.lock`), host 136 checks, simulator 154 checks.
This is the general kernel; the prefill-roof lane (L403) tunes it toward the ceiling.
