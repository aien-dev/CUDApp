# motor

The motor is CUDA++ serving: the live 35B on port 18091, the stroke, chunked
prefill, and the FP8 grouped GEMM. This directory is its home in CUDA++.

What is here now:

- `../omega/slices/launch.osc` is the motor's submit scheme in Omega, copied
  from `omega_gpu_session_launch`: the method header `bw_mthd`, the 18
  compute-class setup words, the shader-cache invalidate, the constant-bank
  bind, and the QMD launch method 0x0318 with its two address words.

What is still in `~/workspace/omega-motor` on this box (C, to port):

- `src/omega_gpu_session.c`: session open, alloc, launch, retire.
- `src/omega_blackwell_submit.c`: pushbuffer build and submit.
- `src/omega_gpu_fp8gemm_api.c`: the FP8 GEMM the prefill runs.

New motor work lands here, in Omega, with a run line per behaviour. The gate
is `tests/omega/run.sh` at the repo root. Nothing here restarts the live
server or touches the GPU outside a scheduled window.
