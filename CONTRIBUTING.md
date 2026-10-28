# Contributing to CUDA++

CUDA++ exists so that people can drop NVIDIA's closed software and still use the NVIDIA GPU they paid for. It gets there one receipt at a time, and anyone with a GPU can add one. This page says how.

## The one rule

Nothing counts until it runs and leaves a receipt. A receipt is the command you ran, the commit you ran it on, the machine, and what it printed, saved under `evidence/`. A pull request that says "should work" is welcome as a start; it is merged when its receipt is there.

A failing test means the work is wrong. Find the cause; do not skip the test, loosen it or rerun it until it passes once.

## What you need

For a host-only start, use Linux with `git`, `make`, a C compiler and the build tools described in [START_HERE.md](docs/START_HERE.md). The documented, fully qualified native GPU path is Linux on the GB10 (DGX Spark, sm_121) with NVIDIA's 580.173.02 kernel module. The CUDA toolkit is not needed for the normal runtime or host tests. A fresh x86-64 host checkout is not yet a validated target for the entire suite (see [#15](https://github.com/aien-dev/CUDApp/issues/15)).

```sh
git clone https://github.com/aien-dev/CUDApp.git
cd CUDApp
make
make test               # host checks; no GPU device opened
build/hello_gpu --sim   # simulator, not a hardware result
make doctor             # diagnostic; may report NOT READY on other hosts
```

The standalone grouped-MoE host acceptance is `sh tests/moe/run.sh`. Set
`OMEGA_OMC` and `OMEGA_BRIDGE` to the exact pins in `om/units.sh`; set
`QUANT_ROOT`, `CACHE_ROOT`, `COMPOSE_ROOT` and `PIPELINE_ROOT` to the same
canonical checkout under test. Its included graph oracle uses GNU11, matching
the production build's POSIX interfaces; strict ISO C11 alone does not expose
`strnlen`. The generated native caller remains compiled with strict C11.
The runner exercises index/admission, incumbent grouping, sparse rank and
refusal/mutant controls without opening a GPU. It is not whole-model evidence.

Only on a qualified GB10 when it is safe to use the GPU (never interrupt a live service):

```sh
make hello-gpu
```

If `make doctor` says your GPU is not qualified, report the device and driver information using the port-a-chip issue template rather than forcing a hardware run.

## Four ways to help, easiest first

### 1. Report a pain point, with a receipt

Something about CUDA or CUDA++ cost you time? Open an issue with the "Pain point" template. Paste the command you ran and its full output, and the output of `make doctor`. If it is a CUDA pain point CUDA++ does not answer yet, say what the better behaviour would be. `docs/PAIN_POINTS.md` is the list we work from; good reports get a row.

### 2. Port a chip ("do it on your GPU")

CUDA++ has run on one chip so far, the GB10 in the DGX Spark. Every other NVIDIA GPU needs a port, and the method is written down in `docs/HOW_I_DID_IT.md`. What changes from chip to chip is listed in `docs/runtime/PORTABILITY.md` section 5. In order:

1. Run `make doctor` and `make test` and keep both outputs.
2. Add a target row for your chip (PCI id, architecture, class numbers, launch descriptor version, memory model). Fields nobody has published are marked unknown until you measure them.
3. Check the encoder's instruction words on your chip: run `examples/hello_gpu` and the host tests' golden words against what your GPU does. Same family (for example another Blackwell part): mostly measuring. Different family: a new encoder.
4. When `make hello-gpu` prints PASS on your GPU, add your chip to the table in `src/omega_gpu_chip.c` with the path of your receipt. That row is what makes `make doctor` say READY for the next person with your GPU.

Open the issue first with the "Port a chip" template, even before anything works, so two people do not port the same chip.

### 3. Port a driver version

CUDA++ talks to NVIDIA's open kernel module through its public headers (`third_party/nvidia-open-580.173.02/`). A different module version may change those calls, so `make doctor` refuses a version CUDA++ was not built for. To add one: copy that release's headers from NVIDIA's `open-gpu-kernel-modules` (MIT) into `third_party/nvidia-open-<version>/`, point the Makefile at them, and run `make test` and `make hello-gpu`. Send the receipts.

### 4. Add or widen a library

CUDA++'s libraries (matrix multiply, attention, elementwise, mixture-of-experts routing) are shaped for the models AIEN runs. Any-shape GEMM, scans, sorts, FFT and random numbers are open (`ROADMAP.md`). A library change needs a host reference (the exact answer computed on the CPU), a simulator run where the simulator covers it, a mutation test (a deliberately broken copy must fail your test) and a chip receipt. Open an issue with the "Add a library" template first.

## Reverse engineering, translation and ports are welcome

The current first-party license is Apache-2.0 WITH LLVM-exception. You may read, disassemble, reverse engineer and translate CUDA++ output, including its machine code, and port CUDA++ to any hardware, subject to the applicable license and retained third-party notices. This is deliberate: CUDA's license restricts reverse engineering its output to target other hardware (`docs/PAIN_POINTS.md`, section 2.8); CUDA++ does not impose that restriction on its own output.

Do not send us NVIDIA's code or binaries. CUDA++ stores no NVIDIA binary, cubin, PTX or firmware. The only listing printed by an NVIDIA tool is a test pin of our own words, recorded in `PROVENANCE.md` section 6 (`docs/AUDIT.md`). If you used NVIDIA's compiler or disassembler on your own small kernels to learn an encoding, say so in the code comment, as the existing code does (`docs/HOW_I_DID_IT.md`, step 2).

## How a change gets in

1. Work on a branch named for what it does.
2. Write the test first and watch it fail for the reason you expect. Then make it pass. Include a negative control: a case that must fail, and does.
3. Run `make test` (the whole host suite, not only your test). Every failure in that run goes in your description by name, including ones you did not cause.
4. Put receipts under `evidence/<your-topic>/` with the commit they ran on.
5. Open the pull request. Say what it improves and where CUDA is still better. Do not claim it is faster than CUDA, llama.cpp, vLLM or TensorRT unless the pull request carries a same-machine, end-to-end measurement of both, with receipts; label chip-only times as chip-only.

## Writing

Plain English and short sentences. Say what a thing does, not how good it is. A number always comes with the command that produced it. Write the name as CUDA++, with the plus signs.

## Licence

By contributing, you agree that your contribution is licensed under Apache-2.0 WITH LLVM-exception as stated in `LICENSE`. If your contribution contains third-party material, identify it and its license before submission; do not submit material you are not authorized to contribute.
