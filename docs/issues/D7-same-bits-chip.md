# D7, chip half: same inputs, same bits, on the GPU

Labels: D7

**The pain.** "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms." (PyTorch docs). Bitwise reproducibility in cuBLAS needs `CUBLAS_WORKSPACE_CONFIG` and costs speed or memory. `docs/PAIN_POINTS.md` section 2.6.

**What exists.** CUDA++ reductions add in a fixed order (`evidence/E1-CLOSURE/`). On the simulator, two runs of hello-gpu print the same output hash.

**What is left.** The same check on the chip, and the same check on a library kernel with a reduction (matrix-vector).

**Done when.** Two runs on the GB10 of hello-gpu and of one matrix-vector job print the same output hash, with no setting turned on; receipt under `evidence/`.
