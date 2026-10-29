# CUDA++ roadmap: every NVIDIA piece on the Spark, and what does its job

The goal is a DGX Spark where CUDA++ does the jobs of NVIDIA's user-space software, on top of NVIDIA's open kernel module and the GPU's own signed firmware, and a GPU stack that does things CUDA does not. GB10 support in the kernel goes to the upstream Linux drivers Nova and nouveau, not into a third driver. Each row names its receipt; a row without a receipt is not done. Status words: PASS (receipt on a named commit), PARTIAL, MISSING, BUILDING (work exists on a branch, no receipt yet), stays (NVIDIA's part remains in the path).

Sources: Omega `docs/OMEGA.md` sections 2 to 5 (branch `docs/what-omega-is`, 2026-10-21). The own-driver work L332 (aienux branch `feat/own-gpu-driver`, aienux#441) is a research harness, not a product.

## 1. NVIDIA software on the Spark today

| NVIDIA piece | What it does | CUDA++ replacement | Status | Receipt or where the work is |
| --- | --- | --- | --- | --- |
| `nvcc`, NVVM, PTX, `ptxas` | Compile kernels to machine code | sm_121 encoder + register allocator (`src/omega_blackwell_*`) | PASS | every GPU kernel this stack runs is encoded by it (Omega `docs/OMEGA.md` s2) |
| `libcuda` | User-space driver API | `nvrm/` + `m16/` driver client | PARTIAL | Implemented; imported M16 summary lacks its referenced raw logs. Current native qualification NOT_RUN; see README evidence limitation. |
| `libcudart` | Runtime API: device, memory, launch, streams | session, command lists, job engine | PARTIAL | one device and one process; streams and events are bar C4 |
| cuBLAS | Matrix multiply | matmul, tensor-core GEMM, FP8/BF16 libraries | PARTIAL | model shapes only; any-shape GEMM is bar C3 |
| cuDNN, FlashAttention | Attention, network layers | decode and prefill attention | PARTIAL | model-shaped |
| CUB, Thrust | Reductions, scans, sorts | frozen-order reductions | PARTIAL | `evidence/E1-CLOSURE/`; general scan and sort are bar C3 |
| cuFFT, cuRAND | FFT, random numbers | none yet | MISSING | |
| `cuobjdump`, `nvdisasm` | Read machine code back | my own sm_121 disassembler | BUILDING | bar C6: on Omega branch `c6/disasm` (19eebe0e, `evidence/C6-DISASM/` there) it agrees with `nvdisasm` on 331 kernels; not carried into CUDA++ yet, and some host tests here still call `nvdisasm` |
| Nsight | Profiling | ceiling probes | PARTIAL | `evidence/OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z/` (115.14 TFLOPS tensor-core peak, 195.1 GB/s reads), `evidence/OWN30B-LAUNCH-COST-20261014T0120Z/` |
| CUDA samples and docs | First kernel, API reference | `examples/hello_gpu`, `docs/runtime/API.md` | PARTIAL | bar C2: hello-GPU from an empty folder on the GB10 + the zero-CUDA gate |
| `nvidia.ko` | Kernel driver: GPU boot, memory, channels | none: NVIDIA's open kernel module stays; GB10 support goes upstream to Nova and nouveau | stays | research harness only: L332 simulator gates PASS on aienux `feat/own-gpu-driver` cd5eef7a (receipt `evidence/own-gpu-driver/rung1/gates-20261021T2158Z.txt` there); no chip run |
| `nvidia-uvm.ko` | Unified memory, page tables, faults | none: stays NVIDIA's open kernel module | stays | research harness only: host simulator, 225 checks PASS (aienux `feat/gb10-gpu-mmu-sim` bcc56506) |
| `nvidia-modeset.ko`, `nvidia-drm.ko` | Display | not needed for compute | not used by CUDA++ | |
| `nvidia-smi`, NVML | Status, clocks, power | not started | MISSING | |
| GSP firmware (`gsp_*.bin`) | Runs inside the GPU | stays NVIDIA's: the chip checks its signature | not replaceable | the kernel module loads NVIDIA's signed file from the system's firmware folder; CUDA++ never ships it |

## 2. Bars before I say "replaces CUDA"

From Omega `docs/OMEGA.md` section 3. Each needs a receipt on the exact commit; none is waived.

- **C1** Omega's languages compile kernels through checker2 to sm_121 and run on the GB10 (vector add, tiled GEMM, decode attention; a broken kernel is refused; one live 35B kernel replaced with identical golden tokens). The language part lives in Omega; the code generator it targets is here.
- **C2** a stranger runs hello-GPU from an empty folder with only the README commands, and the zero-CUDA gate passes on the binary; the runtime API is written down and versioned.
- **C3** any-shape GEMM in BF16 and FP8 (200+ shapes, 1 to 8192, each checked against a host reference, speed as a fraction of the measured tensor-core ceiling); reductions and prefix sums from 1 to 2^28 elements, bit-exact.
- **C4** streams and events: two queues finish in under 0.75 of the sum of their solo times; an event wait orders two jobs, checked by data.
- **C5** two processes share the GPU safely, and a kernel that never ends is timed out with the process still usable.
- **C6** my disassembler agrees with `nvdisasm` on every library kernel, and nothing calls `nvdisasm` any more.

## 3. More than CUDA

Things a CUDA stack does not do, because it does not own the whole path. Omega `docs/OMEGA.md` bar P4: at least one measured win from each before it is claimed.

| Item | Why CUDA cannot | Status |
| --- | --- | --- |
| One submission per generated word | CUDA launches go through the runtime and driver per kernel or per graph; I write the command list myself | PARTIAL: command lists exist (`src/omega_gpu_cmdlist.*`); extra launch in a list costs 24 to 31 us today with wait-for-idle (`evidence/OWN30B-LAUNCH-COST-20261014T0120Z/`) |
| Resident whole-model kernel | needs control of scheduling and memory below the CUDA API | MISSING (AIEN L272) |
| Zero-copy shared memory | the GB10's CPU and GPU share one memory; CUDA still stages through its own allocators | PARTIAL: weights shared between processes through the driver client (physics `feat/shared-mem` 82467f19, 1 GiB import in 171 ms; not yet in CUDA++ main) |

## 4. Speed

Speed is reported against the physical ceiling of this machine, not only against yesterday. The table and the ceilings are in Omega `docs/OMEGA.md` section 5; the head-to-head measurement against llama.cpp and vLLM on this Spark, same model and prompts, end to end, is bar P0 (AIEN L394; in progress, not finished, not published). Until it is published, no speed comparison with CUDA, llama.cpp, vLLM or TensorRT is claimed. Chip-only time per token and end-to-end time through the server are reported separately and never set beside each other or beside other projects' published end-to-end figures.

## 5. The developer bar: goals for being easier to use than CUDA

The research in `docs/PAIN_POINTS.md` ranks what developers hit with CUDA and its libraries, and why they stay. Each pain point becomes a row with a done-when anyone can run. The full table, with every row's mechanism and the counts behind its rank, is `docs/PAIN_POINTS.md` section 3; open rows have issue drafts in `docs/issues/`. The rows are developer-experience goals, not measured comparisons with CUDA, and none of them is a speed claim.

| Row | Aims to improve on CUDA by | Status |
| --- | --- | --- |
| D1 | Memory you can see, and a refusal before the machine hangs | DONE on the host; chip run open |
| D2 | One command, one version: `make doctor`, then `make hello-gpu` | DONE in doctor wrapper; no direct library admission guard |
| D3 | Faults named at the launch that caused them | DONE on the simulator; chip half open |
| D4 | The simulator first, from the same program: `hello_gpu --sim` | DONE |
| D5 | Builds in seconds, ships small | DONE (2.0 s, 1.06 MB) |
| D6 | A new chip works on day one or is refused with a sentence | DONE in doctor wrapper only; table row is not current-commit qualification |
| D7 | Same inputs, same bits, at no cost | simulator DONE; chip DONE for hello-gpu, bf16 GEMV, rmsnorm, decode attention (10 separate processes each, `evidence/d7-same-bits-chip/2026-10-27/`); other kernels not hashed; the "no cost" half is not measured |
| D8 | No per-process CUDA tax | PLANNED (chip window) |
| D9 | No licence that stops you | DONE (`evidence/dx/pain-points/licence.txt`) |
| D10 | No compiling at run time | DONE for hello-gpu |
| D11 | Open libraries with an easy default path | PLANNED (L386, Omega bar C3) |
| D12 | Work per answer, against the machine's own ceiling | simulator DONE; chip open |
| D13 | Our own disassembler, in plain text | PLANNED (L386, Omega bar C6) |
| D14 | Recover from a fault without restarting | PLANNED (L386, Omega bar C5) |
