# D8: measure what a CUDA++ process costs beyond its own buffers

Tracking: [#14](https://github.com/aien-dev/CUDApp/issues/14). Keep open until its exact-commit acceptance gate passes.

Labels: D8

**The pain.** "The CUDA tax, which is 4.6 GB approximately per each model on a DGX spark." (r/LocalLLaMA). `docs/PAIN_POINTS.md` section 2.7.

**What exists.** A CUDA++ process maps no CUDA library (Omega `evidence/WHAT-OMEGA-IS-20261021/live-serve-no-cuda.txt`).

**What is left.** Measure it: resident memory of `hello_gpu` before open, after open and after its buffers are allocated, and the GPU memory the driver reports for the process, so the overhead beyond the program's own buffers is a number.

**Done when.** `hello_gpu` prints its own overhead line, and a chip receipt records it.
