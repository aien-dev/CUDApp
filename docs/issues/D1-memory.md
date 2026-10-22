# D1: refuse an allocation before it hangs the machine

Tracking: [#8](https://github.com/aien-dev/CUDApp/issues/8). Keep open until its exact-commit acceptance gate passes.

Labels: pain-point, D1, good-first-issue

**The pain.** On the DGX Spark the CPU and GPU share one memory, and the machine has no swap. A memory spike does not end in an error message: "On unified memory a spike goes straight to a global OOM that hard hangs the whole box. No ssh, no ping, needs a power cycle." (github.com/sudoingX/dgx-spark-laguna, FINDINGS.md). `docs/PAIN_POINTS.md` section 2.1.

**What exists.** CUDA++ has no caching allocator: every buffer is an explicit allocation, so what a process holds is what it asked for. `make doctor` prints the shared memory and how much is free now.

**What is left.** In the Linux platform's allocation path (`src/omega_gpu_platform_linux.c`, hosted), refuse an allocation that would leave less than a reserve free on a unified-memory GPU. The refusal is a sentence: bytes asked, bytes free, the reserve, and what to do (free buffers, lower the batch, or set the reserve). The reserve has a default and one setting to change it.

**Done when.**
- A host test with a fake `/proc/meminfo` shows the refusal text and that a smaller allocation still succeeds (seen failing first).
- A negative control: with the check removed, the test fails.
- A chip run on the Spark prints the live budget at open.
