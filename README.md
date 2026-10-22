# CUDA++

**CUDA++ is the GPU half of the Omega runtime, under the MIT licence so anyone can use it.** You write a GPU program, CUDA++ turns it into the GPU's own machine code and runs it on an NVIDIA GPU, with no NVIDIA user-space software anywhere: no CUDA toolkit, no `nvcc`, no PTX, no `libcuda`, no cuBLAS.

It runs on the NVIDIA GB10 (the DGX Spark, compute capability sm_121) today. The 35-billion-parameter model AIEN serves on the Spark runs on this code: on 2026-10-21 the live server had only the C library, the maths library, `libgcc_s` and the loader mapped, and zero CUDA libraries (receipt in Omega: `evidence/WHAT-OMEGA-IS-20261021/live-serve-no-cuda.txt`).

CUDA++ keeps growing until every NVIDIA part on the Spark is replaced, and then does things CUDA does not:

- our own GPU driver replaces NVIDIA's kernel modules `nvidia.ko` and `nvidia-uvm.ko` (AIEN work item L332);
- our own disassembler replaces `nvdisasm`;
- only the GSP firmware stays NVIDIA's, for a physical reason: the GPU checks its signature and boots nothing else;
- beyond CUDA: owning the whole path from program to chip, so one submission per generated word, whole-model kernels that stay resident on the GPU, and zero-copy use of the memory the CPU and GPU share.

`ROADMAP.md` lists every NVIDIA piece on the Spark, what replaces it and how far that has got.

CUDA++ is not affiliated with or endorsed by NVIDIA; CUDA is NVIDIA's trademark.

## How it fits with Omega

Omega (github.com/aien-spark/omega, AGPL-3.0) is AIEN's language for all computing: the languages Omegese and OSC, their checker and the CPU backends. CUDA++ is the part of the Omega runtime that drives the GPU. Omega uses CUDA++ at a pinned commit (`cudapp.lock` in Omega), the same way it pins its other parts. The code was split out of Omega on 2026-10-21 with its full history; `docs/AUDIT.md` records how, and what was checked before it was published.

## What is in here

| Part | What it does | Where |
| --- | --- | --- |
| GPU toolchain | Writes sm_121 machine code (SASS) directly: a kernel builder with 59 instruction kinds, a register allocator, the instruction encoder, launch descriptors (QMD) | `src/omega_blackwell_codegen*.c`, `src/omega_blackwell_encoder.c`, `src/omega_blackwell_qmd.c` |
| GPU runtime | Opens the device, allocates memory, uploads code, launches kernels, records command lists, runs jobs. The job of CUDA's runtime and driver API | `src/omega_gpu_session.*`, `omega_gpu_cmdlist.*`, `omega_gpu_engine.*`, `omega_gpu_platform*` |
| Libraries | Matrix multiply and matrix-vector (BF16, FP8, 4-bit, ternary), attention for decode and prefill, elementwise, mixture-of-experts routing. The job of cuBLAS and cuDNN | `src/omega_gpu_*_api.*` |
| Driver client | Talks to NVIDIA's open kernel module through its ioctls; replaces `libcuda` | `nvrm/`, `m16/` |
| NVIDIA interface headers | The open kernel module's public interface definitions (MIT, unchanged) | `third_party/nvidia-open-580.173.02/` |
| Runtime API | The written, versioned API and its manifest check | `docs/runtime/API.md`, `src/omega_runtime_api.h` |

## CUDA, piece by piece

| CUDA piece | Its job | CUDA++ replacement | Status |
| --- | --- | --- | --- |
| CUDA C++ language | How you write a kernel | The kernel builder (C calls). Omega's languages reaching the GPU is Omega bar C1, built on this toolchain | PARTIAL: C calls only |
| nvcc, NVVM, PTX, ptxas | Compile the kernel to machine code | Our own sm_121 encoder and register allocator | PASS: every kernel Omega runs is encoded by it |
| libcuda (driver API) | Talk to the kernel driver | `nvrm` client, ioctls straight to the open kernel module | PASS (`evidence/m16-blackwell-native-path-requalification-receipt.json`) |
| libcudart (runtime API) | Device, memory, launch, streams | Session, command lists, job engine | PARTIAL: one device, one process; no streams or events yet; a timeout ends GPU use for the process |
| cuBLAS | Matrix multiply | Matmul, tensor-core GEMM, FP8 and BF16 libraries | PARTIAL: shaped for the models we run; no any-shape GEMM yet |
| cuDNN, FlashAttention | Attention and network layers | Decode and prefill attention | PARTIAL: model-shaped |
| CUB, Thrust | Reductions, scans, sorts | Frozen-order reductions (`evidence/E1-CLOSURE/`) | PARTIAL: no general scan or sort |
| cuFFT, cuRAND | FFT, random numbers | none yet | MISSING |
| cuobjdump, nvdisasm | Read machine code back | none yet: some host tests call NVIDIA's `nvdisasm` as a cross-check when it is installed | MISSING |
| Nsight | Measure | Ceiling probes: 115.14 TFLOPS tensor-core peak, 195.1 GB/s reads (`evidence/OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z/`) | PARTIAL: probes, no single tool |
| CUDA samples, docs | Your first kernel | `examples/hello_gpu`, `docs/runtime/` | PARTIAL: one example |
| `nvidia.ko`, `nvidia-uvm.ko` | Kernel driver | AIEN's own GB10 driver (L332), being built | MISSING |
| GSP firmware | Runs inside the GPU | stays NVIDIA's (signature-checked by the chip) | not replaceable |

PASS means a receipt exists on a named commit. Status words come from Omega's `docs/OMEGA.md`, section 2.

## Run your first kernel

You need: a DGX Spark (GB10) running Linux with NVIDIA's open kernel module loaded (version 580.173.02 was used here; `/dev/nvidia0` and `/dev/nvidia-uvm` must exist), plus `git`, `make` and `gcc`. Nothing from the CUDA toolkit is needed or used.

From an empty folder:

```bash
git clone https://github.com/aien-dev/CUDAplusplus
cd CUDAplusplus
make hello-gpu
```

`make hello-gpu` builds the library and `examples/hello_gpu`, then runs it. The program builds a vector-add kernel, adds two arrays of one million numbers on the GPU and checks every answer on the CPU. It prints `PASS` and exits 0 when all answers are right, exits 1 when any is wrong, and exits 2 when it cannot use the GPU. `examples/hello_gpu/hello_gpu.c` explains each of its five steps.

## Build and test without a GPU

```bash
make          # build/libomega_gpu.a (also as build/libcudapp.a)
make test     # every host-only check: encoder goldens, simulators, refusals, mutation tests
```

`make test` opens no device. The chip gates (`./build/gpu_*_test --out receipt.json`) need a GB10; each test's comment in the `Makefile` says how it is run.

## Names

The C names still carry the `omega_` prefix they had inside Omega, so Omega's work in progress carries over unchanged (`tools/extract/carry.sh`). `docs/NAMING.md` has the plan for the `cudapp` names.

## Licence

MIT, Copyright (c) 2026 Drake Stapleton (`LICENSE`). Third-party material and its notices: `THIRD_PARTY.md`.
