# How I did it

This is the whole method, including the reverse engineering, so anyone can check it, repeat it on another NVIDIA GPU, or do better. Every step below points at code or receipts in this repository.

**Who:** me, Drake Stapleton, alone, directing AI coding agents as tools (Claude, Codex, Grok, Gemini; `docs/AUDIT.md` section 2). **Where:** one NVIDIA DGX Spark (GB10, sm_121, Linux, NVIDIA's open kernel module 580.173.02). **When:** the first native kernel ran on 2026-09-26; on 2026-10-21 a 35-billion-parameter model was serving on this code with zero CUDA libraries loaded.

## The idea

CUDA is several layers of NVIDIA software stacked on a GPU. Most of the layers are closed: the compiler (`nvcc`, `ptxas`), the driver library (`libcuda`) and the math libraries (cuBLAS, cuDNN). One layer is open: since 2022 NVIDIA publishes the source of its Linux kernel module (`open-gpu-kernel-modules`, MIT). The GPU itself runs whatever machine code it is handed.

So I skipped every closed layer. CUDA++ talks to the open kernel module directly and hands the GPU machine code I write myself. Nothing NVIDIA ships above the kernel module is used at run time.

```
   CUDA:     your code -> nvcc/ptxas -> libcudart -> libcuda -> nvidia.ko -> GPU
   CUDA++:   your code -> my encoder -------> my runtime -----> nvidia.ko -> GPU
```

## Step 1. Talk to the kernel module without libcuda

`libcuda` is a closed library, but everything it does to the hardware goes through ioctls on `/dev/nvidiactl`, `/dev/nvidia0` and `/dev/nvidia-uvm`, and the open kernel module defines every one of those calls in its public headers. I copied 67 of those headers unchanged (`third_party/nvidia-open-580.173.02/`, MIT) and wrote my own client against them (`nvrm/nvrm.c`, about 2,400 lines with `m16/`):

1. open a resource-manager client, then the device and subdevice objects (classes `cl0000`, `cl0080`, `cl2080`);
2. create a GPU virtual address space and allocate memory, mapped for both CPU and GPU;
3. create a channel: a command ring (GPFIFO) and its control page (USERD), bound to the compute engine;
4. create the compute object (Blackwell's compute class);
5. write commands into the ring and ring the doorbell, a single store from user space;
6. wait for completion by polling a memory word the GPU writes.

The headers say what each call takes. What they do not say I learned on the chip, and wrote down as rules with receipts. One example: a word the CPU polls while the GPU writes it must be mapped uncached for the GPU, or the CPU never sees it change. That stalled me once, on 2026-10-02, and the platform now refuses such a buffer (`docs/runtime/PORTABILITY.md`, `src/omega_gpu_platform.h`). The receipt that this path runs with no `libcuda` linked: `evidence/m16-blackwell-native-path-requalification-receipt.json`.

## Step 2. Learn the machine code (the reverse engineering)

NVIDIA does not publish the instruction encoding of its GPUs. Its public assembly language, PTX, is a virtual one; the real machine code (SASS) is undocumented. I worked it out from three sources and checked every result against the chip:

1. **NVIDIA's own tools as an offline oracle.** I wrote small CUDA C++ probe kernels of my own, compiled them once with `nvcc 13.0.88 -arch=sm_121 -cubin`, and printed the machine code with `cuobjdump -sass`. Each instruction comes out as 16 bytes next to its mnemonic. Comparing kernels that differ in one detail (a register number, an immediate, a predicate, a modifier) shows which bits encode what. The source code says where this was done: `src/omega_blackwell_codegen.h`, `src/omega_blackwell_codegen.c`, `src/omega_blackwell_codegen_tc.c` and `src/omega_blackwell_codegen_qmma.h` cite the nvcc output they match. These tools are never run by CUDA++ itself; no NVIDIA binary is stored in this repository, and the only stored NVIDIA-tool listing is the `nvdisasm` decode of our own trace-emitter words (`PROVENANCE.md` section 6; `docs/AUDIT.md` section 5).
2. **Mesa's open NVIDIA compiler.** Mesa's NAK compiler (MIT) encodes NVIDIA instructions for its Vulkan driver, so its source records many opcodes and bit positions (`nak/sm70_encode.rs`), instruction latencies (`nak/sm120_instr_latencies.rs`) and the launch descriptor layout (`nak/qmd.rs`). I cite it wherever I used a fact from it and copied no code (`THIRD_PARTY.md`). Where nvcc and NAK disagree for sm_121, I follow what nvcc emits and say so in a comment.
3. **The chip itself.** An encoding is only believed once the GPU runs it and gives the right answer. For arithmetic that is not exactly specified I ran probe kernels and fitted a model to the results: the tensor-core BF16 multiply-accumulate was characterised on 2,048 cases chosen to separate candidate models, until one model matched every output bit for bit (`tests/gpu_tc_probe_test.c`, `tests/bw_hmma_model.h`). The warp simulators the host tests run (`tests/bw_warp_sim*.h`) model the special-function unit the same way, flushing subnormals as the GB10 does.

What came out of this: my own encoder and kernel builder (`src/omega_blackwell_encoder.c`, `src/omega_blackwell_codegen*.c`), 59 instruction kinds on virtual registers, a register allocator, and the launch descriptor (QMD) builder. Each instruction kind has golden words in the tests; when NVIDIA's `nvdisasm` happens to be installed, some host tests also feed my bytes to it and check it prints the instruction I meant (`tests/gpu_reconv_test.c`). Replacing that cross-check with my own disassembler is on the roadmap.

The first kernel, vector add (`src/omega_blackwell_encoder.c`), is a fixed table of 32 instruction words checked bit for bit. The model's kernels are generated by the kernel builder.

## Step 3. Launch work like the driver does

A kernel launch on Blackwell is a short list of commands written into the channel ring: point at the code, hand over a launch descriptor (grid size, block size, registers, shared memory, constant buffer with the arguments), invalidate the right caches, fire. I took the order of those commands from Mesa's NVK Vulkan driver (`nvk_cmd_dispatch.c`, `nvk_cmd_buffer.c`) and the class method offsets from the open headers, built my own command lists (`src/omega_gpu_cmdlist.c`), and measured the result: about 24 to 31 microseconds per extra dependent launch (`evidence/OWN30B-LAUNCH-COST-20261014T0120Z/`).

## Step 4. Build up, checking everything

From there it is ordinary engineering with one rule: nothing counts until the chip agrees. Each piece has host tests (warp simulators, tensor-core models, refusal cases), mutation tests (a deliberately broken copy must fail), and a chip receipt on a named commit:

- 2026-09-26: vector add qualified on the GB10; the first tensor-core matrix multiply code (M18).
- 2026-10-02: numerics closed on the chip (reductions, nine transcendental functions over all 2^32 inputs, 148 load and store kernels; `evidence/E1-CLOSURE/`).
- 2026-10-13: tensor-core peak measured at 115.14 TFLOPS and memory reads at 195.1 GB/s by my own probes (`evidence/OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z/`).
- 2026-10-21: Qwen3.6-35B-A3B (a mixture-of-experts model, about 3B parameters active per token) served on this code with zero CUDA libraries in the process. Against an exact host reference, three of four prompts matched 32 of 32 token ids and the fourth matched 31 of 32, its one difference inside a near-tie band written before the run; the HTTP check matched 24 of 24.

## What I did not do

- I used no NVIDIA source code other than the open kernel module headers, and no NVIDIA binary. There is no `libcuda`, cubin, PTX or firmware in this repository or its history (`docs/AUDIT.md`).
- I did not take apart `libcuda`, `nvcc` or any other NVIDIA program. I ran NVIDIA's compiler and disassembler on my own small kernels and read what they printed.
- I did not touch the GPU's firmware. The GSP firmware is still NVIDIA's, loaded by NVIDIA's kernel module, and stays so: the chip checks its signature.
- I did not translate CUDA programs to run on other hardware. CUDA++ runs on NVIDIA GPUs, the ones people already paid for.

## Do it on your GPU

The same method works on any NVIDIA GPU the open kernel module supports. What changes per chip is listed in `docs/runtime/PORTABILITY.md`: the instruction encoding, the launch descriptor, the class numbers and the cache rules. Start with `examples/hello_gpu`, find the differences with the three sources above, and send the receipts.
