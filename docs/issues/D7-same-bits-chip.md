# D7, chip half: same inputs, same bits, on the GPU

Labels: D7

**The pain.** "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms." (PyTorch docs). Bitwise reproducibility in cuBLAS needs `CUBLAS_WORKSPACE_CONFIG` and costs speed or memory. `docs/PAIN_POINTS.md` section 2.6.

**What exists.** CUDA++ reductions add in a fixed order (`evidence/E1-CLOSURE/`). On the simulator, two runs of hello-gpu print the same output hash.

**Done when.** Two runs on the GB10 of hello-gpu and of one matrix-vector job print the same output hash, with no setting turned on; receipt under `evidence/`.

**Status: DONE on the GB10 (2026-10-27).** Receipt: `evidence/d7-same-bits-chip/2026-10-27/SUMMARY.md`.

The check went beyond the bar: 10 runs per job instead of two, each a separate process, with no determinism or CUDA variable in the environment. The runner refuses to start if any is set. Setup: driver 580.173.02, code at `0490d98`, and the resident 35B serving beside the runs. Command: `flock -w 900 /tmp/aien-gb10.lock tools/run_d7_same_bits_chip.sh JOB 10 OUTDIR`. The comparison is `tools/d7_same_bits_check.sh`.

| Job | Reduction | Labels | Result |
|---|---|---|---|
| hello (`build/hello_gpu`) | none (vector add) | output | `72d0c1b2…b6a6` in 10 of 10. This is the same hash as the simulator. |
| gemv (`build/gpu_bf16_gemv_test`) | bf16 151936 x 2048: FFMA chain, then a SHFL.DOWN tree | gemv_f32_151936x2048 | `ed380597…9e4b` in 10 of 10 |
| rmsnorm (`build/gpu_elementwise_test`) | sum of squares across the CTA | 4x2048, 1x256, 2x8192 | each label identical in 10 of 10 |
| dattn (`build/gpu_decode_attn_test`) | softmax max and sum; the combine across 1 to 128 KV splits | contexts 1, 7, 256, 301, 1024, 4096, 8192, 32768 | each label identical in 10 of 10 |

**Negative control.** The checker printed MISMATCH (exit 1) in three cases (`negative-control/checker-controls.txt`):
- A real chip run with one output element left unwritten (`hello_gpu_control --bug=short`, hash `ec24bd3e…`).
- A one-bit change to one run's hash.
- One run with a hash line missing.

On the unmodified runs it printed PASS.

**What is left (not claimed).**
- **Scan and sum-reduce** (`tests/gpu_scan_test.c`): NOT_RUN. They did not build on main when the receipt was taken (0490d98); #112 has since merged and they build on main (`mk/c3.mk`). Add a job line to the runner and take a chip receipt.
- **Other kernels.** No other kernels were hashed, including the fp8 grouped GEMM, prefill attention, GDN, MoE and the tensor-core GEMMs. Each one needs a single `out_sha_buf` line in its test (`tests/out_sha.h`) and a job in the runner.
- **Other GPUs and drivers.** Only one GB10 with one driver was checked.
- **Other inputs and other load.** Each job used one fixed input set. The runs did not test deliberately contended launches.
