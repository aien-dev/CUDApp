<p align="center">
  <img src="docs/img/hero.svg" width="100%" alt="CUDA++: my own path from a program to an NVIDIA GPU. On the right, the 32 instructions of the first kernel CUDA++ ran, drawn bit by bit from src/omega_blackwell_encoder.c.">
</p>

# CUDA++

**CUDA++ is the GPU half of the Omega runtime, under the MIT licence so anyone can use it.** You write a GPU program, CUDA++ turns it into the GPU's own machine code and runs it on an NVIDIA GPU, with no NVIDIA user-space software anywhere: no CUDA toolkit, no `nvcc`, no PTX, no `libcuda`, no cuBLAS.

It runs on the NVIDIA GB10 (the DGX Spark, compute capability sm_121) today. The 35-billion-parameter model AIEN serves on the Spark runs on this code: on 2026-10-21 the live server had only the C library, the maths library, `libgcc_s` and the loader mapped, and zero CUDA libraries (receipt in Omega: `evidence/WHAT-OMEGA-IS-20261021/live-serve-no-cuda.txt`).

CUDA++ keeps growing until every NVIDIA part on the Spark is replaced, and then does things CUDA does not:

- my own GPU driver replaces NVIDIA's kernel modules `nvidia.ko` and `nvidia-uvm.ko` (AIEN work item L332);
- my own disassembler replaces `nvdisasm`;
- only the GSP firmware stays NVIDIA's, for a physical reason: the GPU checks its signature and boots nothing else;
- beyond CUDA: owning the whole path from program to chip, so one submission per generated word, whole-model kernels that stay resident on the GPU, and zero-copy use of the memory the CPU and GPU share.

CUDA++ is not affiliated with or endorsed by NVIDIA; CUDA is NVIDIA's trademark.

**On this page:** [the stack](#the-stack) · [how I did it](#how-i-did-it) · [the machine code](#what-the-machine-code-looks-like) · [the driver, part by part](#the-driver-part-by-part) · [one launch, start to finish](#one-launch-start-to-finish) · [CUDA, piece by piece](#cuda-piece-by-piece) · [speed](#speed-against-the-ceiling) · [run your first kernel](#run-your-first-kernel)

## The stack

<p align="center"><img src="docs/img/stack.svg" width="100%" alt="CUDA and CUDA++ layer by layer. The compiler, maths libraries, runtime and driver library of CUDA are closed and are replaced by CUDA++ code. NVIDIA's open kernel module is still used for now; the signed GSP firmware stays."></p>

Between a CUDA program and the GPU sit four closed NVIDIA layers: the compiler, the maths libraries, the runtime and the driver library. CUDA++ replaces all four with its own code. Under them are NVIDIA's kernel module, whose source NVIDIA publishes, and the firmware inside the GPU. CUDA++ talks to the kernel module directly; AIEN's own driver for the GB10 is being built to replace it.

## How I did it

<p align="center"><img src="docs/img/timeline.svg" width="100%" alt="Timeline: 2026-09-26 first native kernel; 2026-10-02 numerics closed on chip; 2026-10-13 ceilings measured; 2026-10-21 a 35B model served with no CUDA library."></p>

I built it alone on one DGX Spark in under four weeks, directing AI coding agents. NVIDIA publishes the source of its Linux kernel module, but not its compiler, its driver library or the GPU's machine code. So I skipped every closed layer:

1. **No libcuda.** I wrote my own client for the open kernel module's ioctls, from its MIT headers: client, device, address space, memory, command ring, compute object, doorbell, completion. [The driver, part by part](#the-driver-part-by-part) shows every piece.
2. **No nvcc, no PTX.** The GPU's machine code is undocumented, so I reverse engineered it: compiled small probe kernels of my own once with NVIDIA's compiler, read the machine code back with NVIDIA's disassembler, compared kernels that differ in one detail to find which bits mean what, cross-checked against Mesa's open NVIDIA compiler, and believed nothing until the chip ran it and gave the right answer. Arithmetic the documents do not pin down (the tensor cores) was measured on the chip until a model matched every output bit.
3. **My own encoder, runtime and libraries** on top: 59 instruction kinds, a register allocator, launch descriptors, command lists, matrix multiply, attention.

<p align="center"><img src="docs/img/reverse-engineering.svg" width="100%" alt="The method in eight steps. Learn: my own probe kernel, nvcc once offline, read the bytes back, change one detail. Prove: field hypothesis, my encoder writes it, the GB10 runs it, golden words in the tests. A wrong answer goes back to the hypothesis."></p>

No NVIDIA binary or listing is in this repository. The whole method, step by step with the files and receipts: [docs/HOW_I_DID_IT.md](docs/HOW_I_DID_IT.md).

### What the machine code looks like

Every GB10 instruction is 16 bytes. The low bits say what to do and to which registers; the top 23 bits tell the warp scheduler how long to wait before the next instruction and which earlier results to wait for. Here is the add from the first kernel, every bit accounted for:

<p align="center"><img src="docs/img/instruction.svg" width="100%" alt="IADD3 R9, PT, PT, R2, R5, RZ decoded: opcode 0x210, guard predicate PT, destination R9, sources R2, R5 and RZ, and the scheduling bits: stall 5 cycles, waits on scoreboard SB4."></p>

Those scheduling bits are what a compiler normally hides. Slow instructions (memory loads, constant loads, special registers) set one of six scoreboards; a later instruction that needs the result waits on it. Here is the whole first kernel with its waits drawn in:

<p align="center"><img src="docs/img/vecadd-scoreboards.svg" width="100%" alt="The 21 real instructions of vector add, with lines joining each instruction that sets a scoreboard to the later instruction that waits on it, and amber bars for fixed stall cycles."></p>

Getting one of these bits wrong does not fail loudly: the GPU reads a register before the load has landed and returns a wrong number. That is why every encoding has golden words in the tests and a chip run behind it. The figures on this page are drawn from the repository's own data by `tools/figures/make_figures.py`; the two above decode the words in `src/omega_blackwell_encoder.c`.

## The driver, part by part

`libcuda` is closed, but everything it does to the hardware goes through ioctls on three device files, and NVIDIA's open kernel module defines every one of those calls in its public headers. I copied 67 of those headers unchanged (`third_party/nvidia-open-580.173.02/`, MIT) and wrote my own client against them. This is the whole of it:

<p align="center"><img src="docs/img/driver-objects.svg" width="100%" alt="The driver client: the three device files, the tree of resource-manager objects it allocates from the root client down to the Blackwell compute object, and the unified-memory calls that bind the address space, each buffer and the channel."></p>

| Part | Where | What it does |
| --- | --- | --- |
| Open | `nvrm_open()` in `nvrm/nvrm.c` | Opens `/dev/nvidiactl` and `/dev/nvidia-uvm`, finds the GPU (`NV_ESC_CARD_INFO`), allocates the root client, device and subdevice, reads the class list to pick the Blackwell classes, reads the SM version and GPU UUID, maps the 64 KiB usermode page that holds the doorbell, and creates an address space that the unified-memory driver manages. |
| Memory | `nvrm_alloc()`, `nvrm_free()` | One system-memory object per buffer. The GB10 has no separate video memory, so each buffer is mapped at the **same address** for the CPU and the GPU: a pointer means the same thing on both sides and nothing is copied. Anything the CPU polls is allocated GPU-uncached (`nvrm_alloc_gpu_uncached()`). |
| Channel | `nvrm_channel()` | Channel group, context share, a command ring of 1,024 entries with its USERD control page, and the Blackwell compute object. Asks for the channel's work-submit token, registers the channel with the unified-memory driver and switches scheduling on. |
| Submit | `nvrm_enqueue()`, `nvrm_ring()` | Writes one 8-byte ring entry (pushbuffer address and length), advances `GPPut` with a full memory barrier, then rings the doorbell: a single 32-bit store of the token. No system call is made to launch work. |
| Wait | `m16/m16_gpu_wait.c` | The one completion wait: polls a word the GPU writes, with a hard deadline that is never extended and a stall deadline that restarts only on progress. |
| Qualification | `m16/m16_requalify.c`, `m16/m16_concurrent.c` | Proofs on the chip. Requalify queues work **without** ringing the doorbell and checks nothing happens for 100 ms, rings it, checks the marker turns to `0x16c0ffee`, and reads `/proc/self/maps` to prove `libcuda` was never loaded. Concurrent runs two channels and checks neither can fire the other. |
| Platform contract | `src/omega_gpu_platform.h` | What the runtime may ask of the machine: typed handles instead of file descriptors, explicit rights on every buffer, no silent fallback. The Linux client above is one implementation; AIEN's own driver plugs in at the same seam. |
| Session, command lists, engine | `src/omega_gpu_session.c`, `src/omega_gpu_cmdlist.c`, `src/omega_gpu_engine.c` | One device per process, kept open; many launches recorded into one submission; a job runner that returns success or exactly one named failure, and refuses all further GPU work if it cannot tell whether the GPU finished. |

What the headers do not say, I learned on the chip and wrote down as rules, each with a receipt or an issue behind it:

- **A word the CPU polls must be GPU-uncached.** Otherwise the GPU keeps the new value in its L2 cache and the CPU never sees it change. This stalled the work once, on 2026-10-02; the platform now refuses such a buffer.
- **New code needs a shader-cache invalidate before it runs.** The GPU's instruction cache does not see CPU writes; after a code address was reused, the chip ran the old instructions. Every launch that uploads code now starts with `INVALIDATE_SHADER_CACHES` (receipt FB1-CUT1B-4345406 in Omega).
- **Dependent launches need a wait-for-idle between them.** Without one, two grids on one channel may overlap. The order of the launch commands follows Mesa's NVK driver.
- **Blackwell's control page reports no read pointer.** The client counts completed ring entries itself and never overruns entries that have not been seen to finish.
- **Creating the compute object needs one free, physically contiguous 2 MiB block** for its context buffer. On a busy host it is briefly missing, so that one allocation is retried, bounded (omega#523).

Under the client: NVIDIA's open kernel module (`nvidia.ko`, `nvidia-uvm.ko`) and the GSP firmware inside the GPU. AIEN's own GB10 driver (L332) is being built to replace the two modules; the firmware stays NVIDIA's because the chip checks its signature. Every object and call in order, with line references: [docs/DRIVER.md](docs/DRIVER.md).

## One launch, start to finish

<p align="center"><img src="docs/img/launch-path.svg" width="100%" alt="The life of one kernel launch: the CPU builds the kernel, writes the launch descriptor, the methods and one ring entry into shared memory, and rings the doorbell; the GPU fetches, launches the grid and writes completion markers that the CPU polls."></p>

The CPU does five things, all of them plain memory writes except the last, which is a single store to a GPU register. The GPU then does the rest on its own: it fetches the ring entry, reads the commands, reads the launch descriptor (the QMD: grid size, block size, registers, shared memory, code address), runs the grid, and writes two completion markers with a cache flush between them. Results are already in memory the CPU can read. Recording many launches into one submission (`src/omega_gpu_cmdlist.c`) is how a whole generated word goes to the GPU at once.

## CUDA, piece by piece

<p align="center"><img src="docs/img/replacement-map.svg" width="100%" alt="Every NVIDIA piece on the Spark with its CUDA++ replacement and status."></p>

| CUDA piece | Its job | CUDA++ replacement | Status |
| --- | --- | --- | --- |
| CUDA C++ language | How you write a kernel | The kernel builder (C calls). Omega's languages reaching the GPU is Omega bar C1, built on this toolchain | PARTIAL: C calls only |
| nvcc, NVVM, PTX, ptxas | Compile the kernel to machine code | My own sm_121 encoder and register allocator | PASS: every kernel Omega runs is encoded by it |
| libcuda (driver API) | Talk to the kernel driver | `nvrm` client, ioctls straight to the open kernel module | PASS (`evidence/m16-blackwell-native-path-requalification-receipt.json`) |
| libcudart (runtime API) | Device, memory, launch, streams | Session, command lists, job engine | PARTIAL: one device, one process; no streams or events yet; a timeout ends GPU use for the process |
| cuBLAS | Matrix multiply | Matmul, tensor-core GEMM, FP8 and BF16 libraries | PARTIAL: shaped for the models AIEN runs; no any-shape GEMM yet |
| cuDNN, FlashAttention | Attention and network layers | Decode and prefill attention | PARTIAL: model-shaped |
| CUB, Thrust | Reductions, scans, sorts | Frozen-order reductions (`evidence/E1-CLOSURE/`) | PARTIAL: no general scan or sort |
| cuFFT, cuRAND | FFT, random numbers | none yet | MISSING |
| cuobjdump, nvdisasm | Read machine code back | My own sm_121 disassembler | BUILDING: on Omega branch `c6/disasm` (19eebe0e) it agrees with `nvdisasm` on 331 kernels; not in CUDA++ yet. Some host tests here still call `nvdisasm` as a cross-check when it is installed |
| Nsight | Measure | Ceiling probes: 115.14 TFLOPS tensor-core peak, 195.1 GB/s reads (`evidence/OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z/`) | PARTIAL: probes, no single tool |
| CUDA samples, docs | Your first kernel | `examples/hello_gpu`, `docs/runtime/` | PARTIAL: one example |
| `nvidia.ko`, `nvidia-uvm.ko` | Kernel driver | AIEN's own GB10 driver (L332), being built | BUILDING |
| GSP firmware | Runs inside the GPU | stays NVIDIA's (signature-checked by the chip) | not replaceable |

PASS means a receipt exists on a named commit. Status words come from Omega's `docs/OMEGA.md`, section 2. `ROADMAP.md` lists every NVIDIA piece on the Spark with its receipt, and the bars CUDA++ must pass before it says "replaces CUDA".

## Speed against the ceiling

<p align="center"><img src="docs/img/ceilings.svg" width="100%" alt="Roofline of the GB10 from my own probes: 115.14 TFLOPS tensor-core peak and 195.1 GB/s memory reads, meeting at 590 operations per byte."></p>

Every speed number is held against what this machine can physically do, measured by my own probe kernels. Generating a word of text reads each active weight once and does about one operation per byte, far left of the ridge, so its speed is set by memory, not by the tensor cores. The head-to-head measurement against llama.cpp and vLLM on this Spark is Omega bar P0 (AIEN L394).

## What is in here

| Part | What it does | Where |
| --- | --- | --- |
| GPU toolchain | Writes sm_121 machine code (SASS) directly: a kernel builder with 59 instruction kinds, a register allocator, the instruction encoder, launch descriptors (QMD) | `src/omega_blackwell_codegen*.c`, `src/omega_blackwell_encoder.c`, `src/omega_blackwell_qmd.c` |
| GPU runtime | Opens the device, allocates memory, uploads code, launches kernels, records command lists, runs jobs. The job of CUDA's runtime and driver API | `src/omega_gpu_session.*`, `omega_gpu_cmdlist.*`, `omega_gpu_engine.*`, `omega_gpu_platform*` |
| Libraries | Matrix multiply and matrix-vector (BF16, FP8, 4-bit, ternary), attention for decode and prefill, elementwise, mixture-of-experts routing. The job of cuBLAS and cuDNN | `src/omega_gpu_*_api.*` |
| Driver client | Talks to NVIDIA's open kernel module through its ioctls; replaces `libcuda` | `nvrm/`, `m16/` |
| NVIDIA interface headers | The open kernel module's public interface definitions (MIT, unchanged) | `third_party/nvidia-open-580.173.02/` |
| Runtime API | The written, versioned API and its manifest check | `docs/runtime/API.md`, `src/omega_runtime_api.h` |
| Figures | Every picture on this page, drawn from the repository's own data | `tools/figures/make_figures.py`, `docs/img/` |

## Run your first kernel

You need: a DGX Spark (GB10) running Linux with NVIDIA's open kernel module loaded (version 580.173.02 was used here; `/dev/nvidia0` and `/dev/nvidia-uvm` must exist), plus `git`, `make` and `gcc`. Nothing from the CUDA toolkit is needed or used.

From an empty folder:

```bash
git clone https://github.com/aien-dev/CUDApp
cd CUDApp
make hello-gpu
```

`make hello-gpu` builds the library and `examples/hello_gpu`, then runs it. The program builds a vector-add kernel, adds two arrays of one million numbers on the GPU and checks every answer on the CPU. It prints `PASS` and exits 0 when all answers are right, exits 1 when any is wrong, and exits 2 when it cannot use the GPU. `examples/hello_gpu/hello_gpu.c` explains each of its five steps.

## Build and test without a GPU

```bash
make          # build/libomega_gpu.a (also as build/libcudapp.a)
make test     # every host-only check: encoder goldens, simulators, refusals, mutation tests
python3 tools/figures/make_figures.py --check   # the figures on this page match the code they are drawn from
```

`make test` opens no device. The chip gates (`./build/gpu_*_test --out receipt.json`) need a GB10; each test's comment in the `Makefile` says how it is run.

## How it fits with Omega

[Omega](https://github.com/aien-dev/omega-mv) (AGPL-3.0) is AIEN's language for all computing: the languages Omegese and OSC, their checker and the CPU backends. CUDA++ is the part of the Omega runtime that drives the GPU. Omega uses CUDA++ at a pinned commit (`cudapp.lock` in Omega), the same way it pins its other parts. The code was split out of Omega on 2026-10-21 with its full history; `docs/AUDIT.md` records how, and what was checked before it was published.

## Names

The C names still carry the `omega_` prefix they had inside Omega, so Omega's work in progress carries over unchanged (`tools/extract/carry.sh`). `docs/NAMING.md` has the plan for the `cudapp` names.

## Licence

MIT, Copyright (c) 2026 Drake Stapleton (`LICENSE`). Third-party material and its notices: `THIRD_PARTY.md`.
