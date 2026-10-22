# CUDA++ roadmap: every NVIDIA piece on the Spark, and what replaces it

The goal is a DGX Spark with no NVIDIA software running on it except the GPU's own signed firmware, and a GPU stack that does things CUDA does not. Each row names its receipt; a row without a receipt is not done. Status words: PASS (receipt on a named commit), PARTIAL, MISSING, BUILDING (work exists on a branch, no receipt yet).

Sources: Omega `docs/OMEGA.md` sections 2 to 5 (branch `docs/what-omega-is`, 2026-10-21) and the own-driver work L332 (aienos/Aienux branch `feat/own-gpu-driver`).

## 1. NVIDIA software on the Spark today

| NVIDIA piece | What it does | CUDA++ replacement | Status | Receipt or where the work is |
| --- | --- | --- | --- | --- |
| `nvcc`, NVVM, PTX, `ptxas` | Compile kernels to machine code | sm_121 encoder + register allocator (`src/omega_blackwell_*`) | PASS | every kernel Omega runs is encoded by it (Omega `docs/OMEGA.md` s2) |
| `libcuda` | User-space driver API | `nvrm/` + `m16/` driver client | PASS | `evidence/m16-blackwell-native-path-requalification-receipt.json` |
| `libcudart` | Runtime API: device, memory, launch, streams | session, command lists, job engine | PARTIAL | one device and one process; streams and events are bar C4 |
| cuBLAS | Matrix multiply | matmul, tensor-core GEMM, FP8/BF16 libraries | PARTIAL | model shapes only; any-shape GEMM is bar C3 |
| cuDNN, FlashAttention | Attention, network layers | decode and prefill attention | PARTIAL | model-shaped |
| CUB, Thrust | Reductions, scans, sorts | frozen-order reductions | PARTIAL | `evidence/E1-CLOSURE/`; general scan and sort are bar C3 |
| cuFFT, cuRAND | FFT, random numbers | none yet | MISSING | |
| `cuobjdump`, `nvdisasm` | Read machine code back | our own sm_121 disassembler | MISSING | bar C6: agree with `nvdisasm` on every library kernel, then replace every call to it |
| Nsight | Profiling | ceiling probes | PARTIAL | `evidence/OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z/` (115.14 TFLOPS tensor-core peak, 195.1 GB/s reads), `evidence/OWN30B-LAUNCH-COST-20261014T0120Z/` |
| CUDA samples and docs | First kernel, API reference | `examples/hello_gpu`, `docs/runtime/API.md` | PARTIAL | bar C2: hello-GPU from an empty folder on the GB10 + the zero-CUDA gate |
| `nvidia.ko` | Kernel driver: GPU boot, memory, channels | AIEN's own GB10 driver (L332) | BUILDING | simulator gates PASS on `feat/own-gpu-driver` cd5eef7a (receipt `evidence/own-gpu-driver/rung1/gates-20261021T2158Z.txt` there); first chip window (window 0) prepared, not run |
| `nvidia-uvm.ko` | Unified memory, page tables, faults | same driver: GPU page tables, allocator, channels (L193 cut D, L332) | BUILDING | host simulator, 225 checks PASS (aienos `feat/gb10-gpu-mmu-sim` bcc56506) |
| `nvidia-modeset.ko`, `nvidia-drm.ko` | Display | not needed for compute; dropped when our driver takes the GPU | MISSING | |
| `nvidia-smi`, NVML | Status, clocks, power | not started | MISSING | |
| GSP firmware (`gsp_*.bin`) | Runs inside the GPU | stays NVIDIA's: the chip checks its signature | not replaceable | our driver loads NVIDIA's signed file from the system's firmware folder; CUDA++ never ships it |

## 2. Bars before we say "replaces CUDA"

From Omega `docs/OMEGA.md` section 3. Each needs a receipt on the exact commit; none is waived.

- **C1** Omega's languages compile kernels through checker2 to sm_121 and run on the GB10 (vector add, tiled GEMM, decode attention; a broken kernel is refused; one live 35B kernel replaced with identical golden tokens). The language part lives in Omega; the code generator it targets is here.
- **C2** a stranger runs hello-GPU from an empty folder with only the README commands, and the zero-CUDA gate passes on the binary; the runtime API is written down and versioned.
- **C3** any-shape GEMM in BF16 and FP8 (200+ shapes, 1 to 8192, each checked against a host reference, speed as a fraction of the measured tensor-core ceiling); reductions and prefix sums from 1 to 2^28 elements, bit-exact.
- **C4** streams and events: two queues finish in under 0.75 of the sum of their solo times; an event wait orders two jobs, checked by data.
- **C5** two processes share the GPU safely, and a kernel that never ends is timed out with the process still usable.
- **C6** our disassembler agrees with `nvdisasm` on every library kernel, and nothing calls `nvdisasm` any more.

## 3. More than CUDA

Things a CUDA stack does not do, because it does not own the whole path. Omega `docs/OMEGA.md` bar P4: at least one measured win from each before it is claimed.

| Item | Why CUDA cannot | Status |
| --- | --- | --- |
| One submission per generated word | CUDA launches go through the runtime and driver per kernel or per graph; we write the command list ourselves | PARTIAL: command lists exist (`src/omega_gpu_cmdlist.*`); extra launch in a list costs 24 to 31 us today with wait-for-idle (`evidence/OWN30B-LAUNCH-COST-20261014T0120Z/`) |
| Resident whole-model kernel | needs control of scheduling and memory below the CUDA API | MISSING (AIEN L272) |
| Zero-copy shared memory | the GB10's CPU and GPU share one memory; CUDA still stages through its own allocators | PARTIAL: weights shared between processes through the driver client (physics `feat/shared-mem` 82467f19, 1 GiB import in 171 ms; not yet in CUDA++ main) |
| Our own driver's launch path | CUDA's is fixed by NVIDIA | BUILDING (L332) |

## 4. Speed

Speed is reported against the physical ceiling of this machine, not only against yesterday. The table and the ceilings are in Omega `docs/OMEGA.md` section 5; the head-to-head measurement against llama.cpp and vLLM on this Spark is bar P0 (AIEN L394, running).
