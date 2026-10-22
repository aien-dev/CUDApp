# yardstick/cuda-vecadd: CUDA's own stack, on the same work

This is the other half of the head-to-head measurement. `vecadd.cu` does exactly what
`examples/hello_gpu/hello_gpu.c` does, on the same GB10, in the same measurement window:

- `c[i] = a[i] + b[i]` over 1048576 floats, with the same inputs (`a[i] = i * 0.25f`,
  `b[i] = 1.0f / (1 + i % 7)`), so the two output hashes must be the same bytes;
- eight launches of 131072 elements, 128 threads per block, 1024 blocks per launch;
- the same timing shape: each launch is timed on the GPU and waited for, and the eight are summed
  (`hello_gpu` gets that number from the session; here it is a CUDA event pair per launch);
- the same work line, against the same 195.1 GB/s read ceiling measured by CUDA++'s own probes.

Two more CUDA modes are printed because CUDA can do them: eight launches queued on one stream and
waited once, and eight empty launches (what a launch costs with no work behind it). That is the
honest decomposition of the gap: how much is the stack, and how much is waiting for each launch.

Nothing in CUDA++ links or calls this program, and the repository's Makefile does not build it. It
is a yardstick, not a dependency. Build it with nvcc 13.0 (`/usr/local/cuda/bin/nvcc -O2 -arch=sm_121`).

Run one window, one receipt:

```sh
make -j8 hello-gpu-build                     # outside the window: a build spends a window on the compiler
spark-window open cudapp-cuda-h2h --gb10 exclusive --quiet exclusive --wait
spark-window run sh yardstick/cuda-vecadd/run.sh evidence/cuda-head-to-head-<date>
spark-window close
```

`run.sh` writes `cuda.log`, `cudapp.log`, `window.txt` and `summary.txt` (both work lines and both
output hashes) into the evidence directory it is given. Commit that directory with the run.
