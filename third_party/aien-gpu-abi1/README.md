# aien-gpu compute ABI 1 (copied, unchanged)

`aien_gpu_uapi.h` is the user-space interface of AIEN's own GB10 kernel module, aien-gpu.ko (`/dev/aien-gpu`).
First-party AIEN material under `MIT OR GPL-2.0-only` (its SPDX line), not the CUDA++ licence.

- Source: aien-gpu repository (to be published as aien-dev/aien-gpu), commit
  `eab436b052d8cd92ec8d471bafe51e76c84b8b7b`, path `kmod/aien_gpu_uapi.h`.
- sha256 `1071e2a1338d258a654dc96f8c6d637db0e06fb4c47329cd73dc69fdf443eda7` (equal to `git show eab436b:kmod/aien_gpu_uapi.h`).

Never edit it here. A new ABI goes in a new `third_party/aien-gpu-abi<N>/` folder.
Used by `src/omega_gpu_platform_aien.c`.
