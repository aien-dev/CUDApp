# CUDA's pain points, and what CUDA++ does about them

This is the research behind CUDA++'s developer bar (section 3). It asks two questions: what do people who use CUDA and its libraries actually run into, and why do they stay anyway? Then it turns the answers into rows anyone can check.

Written 2026-10-22 for AIEN ledger item L400. The license statements and receipt below describe the tree as it stood on that date; the current transition candidate is documented in `docs/LICENSING_TRANSITION.md`. Every outside claim has a numbered source at the end; every quote is attributed and was checked against the saved page text by a script (section 5). Where CUDA is better today, section 4 says so.

## 1. How this was done

Sources: GitHub issues (PyTorch, vLLM, FlashAttention, FlashInfer, Triton, mamba, causal-conv1d, onnxruntime, bitsandbytes, ZLUDA, ROCm), NVIDIA's developer forums, the PyTorch forum, Hacker News, Reddit (r/LocalLLaMA, r/CUDA, r/MachineLearning, r/hardware), Stack Overflow, project blogs, NVIDIA's own documents and its filings.

How often: each pain point has GitHub issue search counts (issues and pull requests whose text contains the phrase, all of GitHub, counted on 2026-10-22 between 00:18Z and 00:48Z). A count is a lower bound on how many people met the problem and wrote about it, not a count of people. Phrases overlap, so counts are not added up. The raw counts, the exact search URL of each and the time it ran are in `evidence/dx/pain-points/counts.json`; `tools/dx/recount.sh` runs them again.

How badly: one of three levels.

| Severity | Meaning |
| --- | --- |
| 3 | Work stops and the message points at the wrong place or the wrong cause, or the machine needs a power cycle. |
| 2 | Work stops with a clear cause, or it costs hours. |
| 1 | It costs time or money; work goes on. |

Rank = reach (the largest single count for that pain point) times severity. This is a rough order, and it is written down so anyone can disagree with a number instead of with a feeling.

Limits, said plainly: Reddit refused every automated read except its RSS feeds, so Reddit threads were read through RSS and carry no vote counts. Stack Overflow counts are small because most of these questions now live in GitHub issues and forums. Hacker News counts are story counts, not comment counts.

## 2. The pain points, most people hurt first

| Rank | Pain point | Reach (GitHub issues) | Severity | Score | CUDA++ row |
| --- | --- | --- | --- | --- | --- |
| 1 | Out of memory, and on the DGX Spark a frozen machine | 26,390 "CUDA out of memory" | 2 | 52,780 | D1 |
| 2 | The version and build matrix | 11,034 "TORCH_CUDA_ARCH_LIST" | 3 | 33,102 | D2 |
| 3 | Errors reported late, at the wrong call | 10,356 "CUDA_LAUNCH_BLOCKING" | 3 | 31,068 | D3, D4 |
| 4 | Builds take hours and ship gigabytes | 7,646 nvcc "compile time" | 2 | 15,292 | D5 |
| 5 | A new chip: "no kernel image is available" | 4,033 "no kernel image is available" | 3 | 12,099 | D6 |
| 6 | Different answers from the same inputs | 5,416 nondeterministic cuda | 2 | 10,832 | D7 |
| 7 | Memory and start-up cost of the CUDA context | 3,478 "CUDA context" memory overhead | 1 | 3,478 | D8 |
| 8 | Lock-in by licence | 2,902 ZLUDA | 1 | 2,902 | D9 |
| 9 | Compiling at run time, and hangs while it compiles | 517 "JIT compilation" hang cuda | 3 | 1,551 | D10 |
| 10 | Closed, model-shaped libraries | no single phrase | 2 | not scored | D11, D13 |
| 11 | The incentive: compute sold from data centers | not a bug report | 1 | not scored | D12 |

The DGX Spark is not a separate row: its problems are rows 1, 2, 5 and 9 at their worst. 8,268 GitHub issues mention "DGX Spark". It gets its own part in section 2.12 because CUDA++ runs on it today.

### 2.1 Out of memory, and on the Spark a frozen machine (rank 1)

**What happens.** A job asks for more GPU memory than is free and stops with "CUDA out of memory". Usually the cause is real: the model does not fit. Two things make it worse than it has to be. PyTorch's caching allocator keeps freed memory for reuse, so the numbers people look at are wrong: "However, the unused memory managed by the allocator will still show as if used in nvidia-smi."[46] Fragmentation can make a job fail that would fit, and the fix is a tuning knob: "This can reduce fragmentation and may allow some borderline workloads to complete without running out of memory."[46]

On the DGX Spark the CPU and GPU share one memory, and the failure is not an error message at all. A Spark owner wrote: "The Spark ships with zero swap. On unified memory a spike goes straight to a global OOM that hard hangs the whole box. No ssh, no ping, needs a power cycle."[2] On NVIDIA's forum: "I am experiencing a critical system-level freeze (hard-lock requiring power cycle) on the DGX Spark when running Gemma 4 at high memory utilization."[31] And: "The whole DGX Spark machine hangs completely—no mouse or keyboard response, nothing. no visual errors, no logs errors ."[32]

**How often.** 26,390 GitHub issues contain "CUDA out of memory" and 12,124 contain "CUDA error: out of memory". Stack Overflow has 1,501 questions. The Spark freeze threads have 1,058 and 1,202 views[31][32].

**Why people stay.** Running out of memory happens on every GPU stack. Nobody leaves CUDA over it.

**CUDA++'s answer (row D1).** CUDA++ has no caching allocator: every buffer is an explicit allocation you asked for, so what the process holds is what you allocated. What is missing is the part that matters on the Spark: refuse an allocation that would leave the machine without memory, with a sentence that says how much was asked, how much is free and what to do, before the kernel's out-of-memory killer or a hang. That is now built: the Linux platform refuses an allocation that would leave less than `CUDAPP_MEM_RESERVE_MIB` (default 4096) free, before the driver is called, and says so in one sentence. Status: DONE on the host (`evidence/dx/d1-memory/`); the chip run is open (`docs/issues/D1-memory.md`).

### 2.2 The version and build matrix (rank 2)

**What happens.** A CUDA program on Python sits on four version numbers that must agree: the kernel driver, the CUDA toolkit, the framework build (PyTorch's wheel is built for one CUDA version) and every extension built against both. A mismatch fails at import or later, often with a symbol name and no advice. Typical messages: "The detected CUDA version (12.0) mismatches the version that was used to compile PyTorch (11.7)" (the PyTorch forum thread with 10,462 views[40]), and "undefined symbol: _ZNK3c1010TensorImpl27throw_data_ptr_access_errorEv"[21]. Even finding the toolkit can fail: "However, CMake is unable to find the CUDA toolkit, even though it's installed and `nvcc` and `nvidia-smi` are working correctly."[94] The driver's "CUDA Version" is not the toolkit's version, which confuses people: "that value reflects the driver’s runtime capability , not the presence of the toolkit ."[35]

Extensions ship one wheel per combination, so the wheel name itself is the matrix: FlashAttention prints "Precompiled wheel not found. Building from source..."[23] and a Reddit guide explains that two wheels match one setup and differ only in the C++ ABI flag, "cxx11abiFALSE" or "cxx11abiTRUE"[89]. Its maintainer said: "I don't know a right solution that works for all setups, happy to hear suggestions."[20] A vLLM user: "Please note here that we can't use precompiled vLLM because `vllm-project/vllm` has not moved to the required torch and CUDA versions yet."[15] And when it goes wrong quietly: "This causes a nightly to be silently installed, which fails at runtime due to an `undefined-symbol` error because `_C.abi3.so` fails at runtime."[15]

Even the toolkit can change the answer. causal-conv1d records a "**numerical correctness bug** for sm_103 (B300): cubins built with CUDA 13.x nvcc for sm_103 produce wrong output"[5].

**How often.** 11,034 GitHub issues contain "TORCH_CUDA_ARCH_LIST" (the variable people set to build extensions for their GPU); 1,507 contain "CUDA driver version is insufficient"; 1,054 "not compatible with the current PyTorch installation"; 932 mention nvidia-smi's "CUDA Version" with a toolkit mismatch; 569 "detected CUDA version" mismatches. The Gaussian splatting install thread has 121 reactions[19], FlashAttention #246 has 84[20], vLLM #14452 has 44 reactions and 139 comments[15]. A forum post on the Spark: "getting a working CUDA toolkit (with nvcc ) turned into a trap maze."[35]

**Why people stay.** The matrix is the price of the libraries, and the libraries are the reason to be there. A Reddit user on the Spark: "The funniest part is paying the NVIDIA tax specifically for CUDA compatibility and then not getting it."[88] Other stacks have the same matrix or a worse one (section 2.11).

**CUDA++'s answer (row D2).** There is one version to match, and it is checked. CUDA++ uses no toolkit, no framework build and no extension build: the compiler, runtime and libraries are one repository at one commit, built by `make` with a C compiler. The only outside version is NVIDIA's kernel module, whose interface headers CUDA++ is built against (`third_party/nvidia-open-580.173.02/`). `make doctor` reads `/proc`, `/sys` and `/dev` (it opens no device) and prints one line per check; a wrong module version is refused with the reason and the fix, instead of failing later in a driver call. `make hello-gpu` runs the doctor first. Status: DONE (`evidence/dx/d2-doctor/`).

### 2.3 Errors reported late, at the wrong call (rank 3)

**What happens.** CUDA launches are asynchronous: "When you call a function that uses the GPU, the operations are enqueued to the particular device, but not necessarily executed until later."[46] So a bad memory access in one kernel is reported by some later, unrelated call: "an illegal memory access was encountered ... might be asynchronously reported at some other API call". vLLM's maintainers wrote: "In our experience the python stack traces for these types of exceptions are basically always incorrect and pretty worthless."[45] And: "Have you ever felt you are developing cuda kernels and your tests often run into illegal memory access (IMA for short) and you have no idea how to debug?"[45]

After such a fault the whole context is gone. NVIDIA's forum moderator: "Pretty much all asynchronous errors (these are errors that occur as a result of device code execution) are sticky."[47] A developer in that thread: "My goal is to determine whether or not restart the executable, otherwise I will restart it for every cuda-related error, but it is a pity."[47]

The tools for this are separate programs with their own gaps. compute-sanitizer "is hard to use in CI build farms that have a very strict network configuration"[48], and NVIDIA's own blog says the default mode misses bugs: "Compile-time instrumentation catches bugs that binary instrumentation misses, such as buffer overruns into neighboring allocated memory, while lowering runtime overhead in some cases by nearly an order of magnitude."[49]

**How often.** 10,356 GitHub issues contain "CUDA_LAUNCH_BLOCKING" (the variable people set to make errors land at the right call); 7,833 contain "illegal memory access was encountered"; 6,812 "might be asynchronously reported"; 6,991 "compute-sanitizer"; 4,232 "device-side assert triggered"; 2,708 "cuda-gdb"; 1,883 "cudaErrorIllegalAddress". Stack Overflow: 166 questions on illegal memory access, 114 on CUDA_LAUNCH_BLOCKING.

**Why people stay.** Every fast GPU stack runs work asynchronously, because waiting after each launch is slow. The debuggers exist, and people learn the environment variables.

**CUDA++'s answer (rows D3 and D4).** Two parts. First, in CUDA++ a launch waits for its own completion by default (`omega_gpu_session_launch` is synchronous; command lists batch work on purpose), so an error comes back from the call that caused it, not a later one. Second, the same kernel program runs on CUDA++'s host warp simulator from the same program with one flag, and a bad access there is reported with the kernel, the launch, the instruction, the block and thread, the address and which buffer it missed and by how many bytes, plus the next command to run. What is not built: reading the GPU's own fault record (address and unit) from the driver after a fault on the chip, and recovering without ending the process (that is Omega bar C5, ledger L386). Status: DONE on the simulator (`evidence/dx/d3-faults/`); the chip half is open (`docs/issues/D3-chip-faults.md`).

### 2.4 Builds take hours and ship gigabytes (rank 4)

**What happens.** CUDA libraries are compiled once per GPU architecture and shipped together, so builds are slow and packages are huge. On FlashAttention: "For reference I just compiled it on a 2018 Dell laptop (Quadro P1000 GPU) and the build time was about 15 hours."[52] Another user: "Without both changes it defaulted to just using a single compilation job for me, taking forever (I gave up after an hour)."[52] On Reddit: "This cut my build time from 1+ hours to just under 15 minutes. Thank you!"[89] Building PyTorch for an older GPU: "It keeps crashing with the nvcc error out of memory, even after I've allocated over 0.75TB of vRAM on my SSD."[93]

Sizes: "For example, recent manylinux2014 wheels for TensorFlow are 588 MB (wheels for 2.11.0 ), and for PyTorch those are 890 MB (wheels for 1.13.0 )."[53] PyPI's default limit is 100 MB per file[54], and projects wait in a queue for exceptions: "There are currently 34 other open limit request issues ahead of this one."[55] NVIDIA itself: "Docker images for ML and AI workloads are notoriously large, often exceeding 10 GB. Much of this comes from bundled GPU libraries for all possible architectures"[56]. On cuDNN, an NVIDIA employee: "We agree that the growth of the cuDNN DLL size is problematic, and we’re working on resolving this."[57] A user: "libcudnn.so.7 is over 200 MB in size!"[58] PyTorch had to drop older GPUs from its wheels: "Update: Option 2 is not a possible due to large binary size."[16]

**How often.** 7,646 GitHub issues match nvcc "compile time"; 73 contain FlashAttention's "Precompiled wheel not found".

**Why people stay.** Building once per architecture is what makes the code fast on each one; the cost lands on packagers and on whoever builds from source.

**CUDA++'s answer (row D5).** CUDA++ builds from a clean checkout with `make` in a time and size printed by the build receipt, with no per-architecture copies, because kernels are generated for the chip the program runs on, in microseconds, when the program builds them (row D10). This is not a like-for-like comparison: CUDA++'s libraries cover far less than cuBLAS and cuDNN (section 4). Status: DONE: a fresh clone builds the library in 2.0 s wall on the Spark (20 cores, other builds running), giving a 1,060,126-byte `libcudapp.a` that needs no CUDA symbol (`evidence/dx/pain-points/d5-build-from-empty.txt`).

### 2.5 A new chip: "no kernel image is available for execution on the device" (rank 5)

**What happens.** A new GPU arrives and the libraries have no machine code for it yet. The error names no library, no kernel and no fix. It happened with the RTX 3090: "I am a skilled user of pytorch-gpu, recently I purchased an RTX 3090 server, but the bug with pytorch 1.7 and RT 3090 makes me mad."[11] With the RTX 5090: "NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation."[14] With the 5070 Ti: "As a researcher, I purchased RTX 5070 Ti for deep learning workloads, but currently it cannot be used productively with DLC due to this gap."[13] "Users are forced to either build PyTorch from source or use experimental nightly builds with limited platform support."[12] Some patch it themselves: "PyTorch doesn’t support sm_120 or the RTX 5080 out of the box. So I patched it."[96]

On the DGX Spark (sm_121) it happens across the stack. mamba-ssm ships wheels for older chips only ("pip install mamba-ssm  # installs prebuilt wheel for sm_75-sm_90"[4]); causal-conv1d "does not ship prebuilt wheels for **sm_121** (GB10 / DGX Spark / Project DIGITS). Users must compile from source."[5] FlashInfer: "the mixed path still dies at routing time because a prebuilt, SM100-only `moe_utils` artifact takes precedence over the local JIT build."[6] FlashAttention: "FATAL: kernel built for sm80-sm100, but running on sm121"[9]. Triton's bundled assembler did not know the chip: "ptxas-blackwell fatal : Value ‘sm_121a’ is not defined for option ‘gpu-name’"[33], and "Setting TRITON_PTXAS_PATH doesn’t help - Triton ignores it in newer versions and uses its bundled ptxas which doesn’t support sm_121a."[34] NVIDIA's own model did not run on NVIDIA's own box: "Conclusion: Nemotron 3 Nano 30B currently cannot run on DGX Spark via SGLang due to Triton’s lack of sm_121a support."[34]

The workarounds can be worse than the error. In Triton's sm_121 issue: "Those \"fixes\" produce a kernel that loads and then computes wrong numbers, which is so much worse than a clean \"no kernel image\" crash."[8] And a stale cache: "a warm `~/.triton/cache` populated on an `sm_120` box (or an earlier toolchain) gets reused on `sm_121` and silently miscompiles instead of rebuilding."[8]

**How often.** 4,033 GitHub issues contain "no kernel image is available" (115 in PyTorch, 115 in vLLM, 20 in bitsandbytes, 18 in FlashInfer, 16 in FlashAttention); 600 contain "cudaErrorNoKernelImageForDevice"; 458 "sm_120 is not compatible"; 1,508 "ptxas fatal". 5,803 mention "sm_121" and 1,683 "sm_121a". PyTorch #31285 has 92 reactions and 78 comments[10].

**Why people stay.** The wait is weeks or months, and then it works. On the Spark forum: "It took several weeks of community effort here to get vllm working reasonably well, in both standalone and cluster configuration."[28] From the Spark Reddit thread: "Very frustrating compared to most every other GPU release that has zero day support from cuda/nvcc/pytorch/tensorflow."[88]

**CUDA++'s answer (row D6).** CUDA++ writes each chip's machine code itself, so it knows exactly which chips it has run on: a table of qualified chips, each with its receipt. A chip in the table works. Any other NVIDIA GPU is refused by `make doctor`, by name and PCI id, with what is missing (a target row, an encoder check, the launch descriptor and class numbers, a qualification run) and what to do (the porting steps in `docs/runtime/PORTABILITY.md` section 5). Nothing loads code built for a different chip, and nothing is cached across chips. CUDA is better here today in one way: it runs on many chips; CUDA++ runs on one (section 4). Status: DONE for the refusal (`evidence/dx/d2-doctor/fake-machines.txt`).

### 2.6 Different answers from the same inputs (rank 6)

**What happens.** Fast GPU code often adds numbers in whatever order threads finish, and floating-point addition depends on order. PyTorch's documentation: "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."[50] cuBLAS's documentation: "In that case, the results are not guaranteed to be bit-wise reproducible because atomics are used for the computation."[51] Getting the same bits needs an environment variable with a cost: "set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 (may limit overall performance) or :4096:8 (will increase library footprint in GPU memory by a"[51]

**How often.** 5,416 GitHub issues match nondeterministic cuda; 1,117 contain "CUBLAS_WORKSPACE_CONFIG"; 526 "use_deterministic_algorithms".

**Why people stay.** For training, small differences rarely matter, and speed does.

**CUDA++'s answer (row D7).** CUDA++'s reductions use a fixed order (`evidence/E1-CLOSURE/`), so the same inputs give the same bits without a switch and without a cost to turn off. The row is done when two runs of the same job give the same output hash on the simulator and on the chip. Status: DONE on the simulator (`evidence/dx/d3-faults/hello-sim-spark.txt`, the same output hash twice); the chip half is open.

### 2.7 Memory and start-up cost of the CUDA context (rank 7)

**What happens.** Every process that uses CUDA loads the CUDA libraries and creates a context, which costs memory per process. On a Spark: "The CUDA tax, which is 4.6 GB approximately per each model on a DGX spark."[91] Kernels shipped as PTX are compiled when the context starts: "This JIT compilation can cause delay at application start-up time (or more accurately, CUDA context creation time)."[59]

**How often.** 3,478 GitHub issues match "CUDA context" memory overhead.

**Why people stay.** It is a fixed cost per process, small next to a large model on a large GPU.

**CUDA++'s answer (row D8).** A CUDA++ process maps no CUDA library at all: on 2026-10-21 AIEN's live 35B server had only the C library, the maths library, `libgcc_s` and the loader mapped (Omega `evidence/WHAT-OMEGA-IS-20261021/live-serve-no-cuda.txt`). The row is done when the memory a CUDA++ process holds beyond its own buffers is measured and printed. Status: PLANNED (needs a chip window; section 3, D8).

### 2.8 Lock-in by licence (rank 8)

**What happens.** NVIDIA's CUDA licence forbids using CUDA's output on other hardware through translation: "You may not reverse engineer, decompile or disassemble any portion of the output generated using SDK elements for the purpose of translating such output artifacts to target a non-NVIDIA platform."[63] Tom's Hardware: "Nvidia has banned running CUDA-based software on other hardware platforms using translation layers in its licensing terms listed online since 2021, but the warning previously wasn't included in the documentation placed on a host system during the installation process."[64] It read the clause as aimed at ZLUDA: "The restriction appears to be designed to prevent initiatives like ZLUDA, which both Intel and AMD have recently participated, and, perhaps more critically, some Chinese GPU makers from utilizing CUDA code with translation layers."[64]

ZLUDA itself ("ZLUDA is a drop-in replacement for CUDA on non-NVIDIA GPUs."[67]) lost its code once, though not to NVIDIA: "AMD's legal team appears to have clawed back control of much of the ZLUDA project's code base."[65] The developer: "The code was released with AMD's approval through an email. AMD's legal department now says it's not legally binding, hence the rollback."[66] On r/LocalLLaMA: "I hope someone is working on hardware agnostic solutions, we need more GPUs, not less."[95] A user at the time: "But since its amd shooting its own foot this project will likely not survive this setback."[68]

**How often.** 2,902 GitHub issues mention ZLUDA; 89 Hacker News stories; the r/LocalLLaMA thread on the licence change is titled "CUDA Crackdown: NVIDIA's Licensing Update targets AMD and blocks ZLUDA".

**Why people stay.** The clause targets other hardware. People who stay on NVIDIA hardware are not affected by it, and that is the point: it keeps the software where the hardware is.

**CUDA++'s answer (row D9, as of 2026-10-22).** The then-current tree was MIT. Reverse engineering CUDA++'s output, translating it, and porting CUDA++ to any hardware, NVIDIA or not, were allowed and welcome. The current candidate license and preserved MIT snapshot are recorded in `docs/LICENSING_TRANSITION.md`. CUDA++ uses no NVIDIA licensed software at run time and stores no NVIDIA binary (`docs/AUDIT.md`). One thing stays NVIDIA's: the GPU's signed firmware, because the chip checks its signature. Status: DONE (receipt in section 3, D9).

### 2.9 Compiling at run time, and hangs while it compiles (rank 9)

**What happens.** Many libraries compile kernels the first time they are used: FlashInfer's default package "compiles/downloads kernels on first use"[60], Triton tunes and compiles at start ("Some users had expressed a desire to cache autotune results, both to speed up local iteration, and to avoid re-tuning when scaling up to large numbers of GPUs."[62]). This can hang: "Worker 0 completes compilation (172s) but workers 1-7 hang indefinitely."[61] On the Spark it took the GPU down: "A deep prefill forces a live JIT for a huge novel shape mid-inference, and that compile or kernel deadlocks the GPU on `sm_121a`."[2] Caches need care: "Insufficient cache size or a cache stored on a slow network share can cause long application start-up times"[59].

**How often.** 517 GitHub issues match "JIT compilation" hang cuda. Low count, high cost: the Spark report calls it "the finding that cost the most power cycles"[2].

**Why people stay.** Compiling for the exact shape is how these libraries get their speed.

**CUDA++'s answer (row D10).** CUDA++'s kernel builder runs in the program's own process, takes microseconds, has no cache directory and no outside compiler to call, and produces the same bytes every time for the same kernel. The row is done when hello-gpu prints its kernel build time and the hash of the machine code, the same on every run. Status: DONE for hello-gpu: built in tens of microseconds (22 and 23 us in the receipt), the same machine-code hash on every run (`evidence/dx/d3-faults/hello-sim-spark.txt`).

### 2.10 Closed, model-shaped libraries

**What happens.** cuBLAS, cuDNN and the layers above them are closed. When they are slow or wrong for a shape, nobody outside NVIDIA can see why. Modular's Chris Lattner: "PTX is only partially documented, constantly shifting between hardware generations, and effectively a black box for developers."[76] And: "CUDA doesn’t provide performance portability across NVIDIA’s own hardware generations, and the sheer scale of its libraries is a double-edged sword."[76] On the Spark, FlashInfer's own audit found "**SM121 falls through to cutlass/cudnn only.**"[7] Users find the hardware's features unreachable: "Just be careful when buying a DGX Spark. You are not really getting a modern CUDA experience."[88] And the playbooks: "many of their official playbooks are broken and can’t be run as-is on the latest versions."[88] Closed code also hides behaviour: a Reddit thread found NVIDIA's FP8 path ran faster when a kernel's name contained "cutlass", and one reply put the risk plainly: "Even worse, if someone accidentally created a kernel with "cutlass" in the name, the driver would apply optimizations that are not safe."[92]

**Why people stay.** The closed libraries are fast for the common shapes, and the easy path is the default: "With a design like Cuda where there is always a default easy path, and a complex but optional path."[77] tinygrad's pitch puts the flywheel plainly: "By being the dominant player, green team gets a whole ecosystem of developers to improve the developer experience for them."[78]

**CUDA++'s answer (rows D11 and D13).** Every CUDA++ library is source in this repository, generated by the same kernel builder you can call, with host tests and a simulator. What is missing is breadth: CUDA++'s matrix multiply and attention are shaped for the models AIEN runs; any-shape GEMM, scans and sorts are Omega bar C3, and our own disassembler is bar C6 (ledger L386). Status: PLANNED (L386).

### 2.11 Why people cannot leave

Three reasons come up again and again.

1. **The other stacks are worse today.** On ROCm: "If you want to know why everyone is going with Nvidia, there are a lot of them, but what they do at least well is they support they GPUs."[69] "I'm a long-time CUDA developer looking to explore ROCm and HIP development, but finding out which hardware even supports these tools is harder than it needs to be"[70]. In 2026, on a new AMD part: "We ran 93+ autonomous ML training experiments on a Strix Halo system (Radeon 8060S, gfx1151) using TheROCk nightly PyTorch and discovered 5 critical bf16 bugs, an undocumented 19x attention speedup, and several ecosystem gaps that prevent consumer AMD GPUs from being viable ML research platforms."[71] George Hotz: "AMD is structurally incapable of fixing these issues. They don't even have a 7900XTX in CI."[72] 17,774 GitHub issues contain ROCm "not supported".
2. **New work lands on CUDA first.** "On CUDA just runs of course with sliding window attention, ALiBi, and PagedAttention. ROCm fork? Nope."[74] "One of the biggest complaints we hear from the industry is \"we tried to port to X and we could never complete it\"."[79] And from a developer whose company tried: "i literally worked at a company where we dropped HIP support, and just went all in on nvidia."[95]
3. **The bar for an alternative is that it just works.** "you have to be able to pip install something and just have it work, reasonably fast, without crashing, and also it has to not interfere with 100 other weird poorly maintained ML library dependencies."[75] And people paid for the hardware: "To me it seems like you're paying more than twice the price mostly for CUDA compatibility."[44]

So CUDA++ does not ask anyone to change hardware. It runs on the NVIDIA GPU people already own, and it has to beat CUDA on that GPU, row by row.

### 2.12 The DGX Spark

The Spark is a small machine sold as a way into NVIDIA's data-center stack. A Reddit user: "You design and validate them locally on the spark, running the exact same software, then push to the data center full of Nvidia GPU racks."[90] Its owners hit rows 1, 2, 5 and 9 at once. One report sums it up: "The published Docker images do not carry sm_121a kernels, a missing environment variable silently breaks the NVFP4 MoE, and a deep-context request with speculative decoding will hard hang the entire machine."[1] "Without it the NVFP4 MoE CUTLASS kernels are not built for GB10 and you get CUDA error: no kernel image is available for execution on the device ."[1] On NVIDIA's forum: "How many times have you tried to run something that doesn’t recognize SM121 and fails?"[28] "As of today, FP4 is not properly utilized in current VLLM builds on our hardware, so you lose a lot of performance picking NVFP4 quants compared to AWQ 4-bit on"[29] (that thread has 16,907 views). An owner of two Sparks: "It was not marketed like an experimental dev kit where buyers should expect to spend months switching backends, testing builds, setting flags, and relying on community or hardcore fan fixes just to make a core feature work properly."[97] Simon Willison: "I eventually managed to get a PyTorch 2.7 wheel for CUDA on ARM, but failed to do so for 2.8."[41] One honest counterpoint from PyTorch's maintainer: "Every PyTorch build with CUDA 12.8 and sm_120 support already supports sm_121 (DGXSpark) as these architectures are binary compatible."[38] The Spark's memory bandwidth is also a hard physical limit no software fixes: "the unified memory is LPDDR5x, offering up to 273 GB/s , shared across both CPU and GPU."[42]

CUDA++ runs on the Spark today: AIEN serves a 35-billion-parameter model on it with no CUDA library loaded (README).

### 2.13 The incentive, and less computation

NVIDIA earns almost all its money from data centers. In its second quarter of fiscal 2027: "Data Center revenue of $89.0 billion, up 117% from a year ago"[85], out of "revenue for the second quarter ended July 26, 2026, of $96.2 billion"[85]: 92.5% of revenue. For fiscal 2025 the annual report gives data center revenue of $115,186 million out of $130,497 million[87], 88.3%. A company paid by the amount of compute it sells in data centers has no reason to make a small machine you own do the same work with less.

People notice. "Nvidia uses VRAM amount for market segmentation. They can't make a 128GB consumer card without cannibalizing their enterprise sales."[82] "Sure, Nvidia gives for \"free\" their software stack.... as long as you use it on their products."[90] The counter-argument is real too: "There is no math that will make local models come out ahead in $/intelligence/token."[83] and "At $20/month for a cloud AI subscription, you’re looking at almost 30 years of service for the same money."[84] People still want local compute for reasons money does not cover: "The point of local models is privacy, offline use, and maybe no guard rails."[83]

**CUDA++'s answer (row D12).** Measure the work, not the hardware. Every CUDA++ result prints the work it took: bytes moved, operations, and the time those need at this machine's own measured ceiling (195.1 GB/s reads and 115.14 TFLOPS tensor-core peak on the GB10, `evidence/OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z/`). Less work per answer is the goal, so the number is always in front of the developer. Status: DONE on the simulator (`evidence/dx/d3-faults/hello-sim-spark.txt`); the measured GPU time next to it needs a chip run.

## 3. The developer bar: D1 to D14

Each row has a done-when anyone can run. A row is DONE only with a receipt on a named commit. PLANNED rows name the ledger item that owns them. The aim is to beat CUDA, not to copy it.

| Row | Beats CUDA by | Done when | Status |
| --- | --- | --- | --- |
| D1 | Memory you can see, and a refusal before the machine hangs | An allocation that would leave less than the reserve free on a unified-memory GPU is refused with a sentence naming the bytes asked, the bytes free and what to do; a host test drives it with a fake `/proc/meminfo`; the chip run shows the live budget | DONE on the host (`evidence/dx/d1-memory/`); chip run open |
| D2 | One command, one version | From an empty folder with only git, make and gcc: `make doctor` prints READY or names each problem with its fix; `make hello-gpu` runs the doctor first; a wrong kernel-module version is refused with a sentence; host test of 9 fake machines | DONE (`evidence/dx/d2-doctor/`) |
| D3 | Faults named at the cause | A bad access is reported at the launch that caused it with the kernel, launch number, instruction, block, thread, address, the buffer it missed and by how much, and the next command; `hello_gpu --sim --bug=past-end` shows it; a host test checks every field | DONE on the simulator (`evidence/dx/d3-faults/`); chip half open |
| D4 | The simulator first, from the same program | `hello_gpu --sim` runs the same kernel program on the host warp simulator and checks every answer bit for bit, with no GPU | DONE (`evidence/dx/d3-faults/`) |
| D5 | Builds in seconds, ships small | Fresh-checkout build time and library size printed by a receipt on the Spark | DONE (`evidence/dx/pain-points/d5-build-from-empty.txt`) |
| D6 | A new chip works on day one or is refused with a sentence | `make doctor` names any unqualified NVIDIA GPU with what is missing and the porting steps; never a bare "no kernel image"; host test with an unknown PCI id | DONE (`evidence/dx/d2-doctor/fake-machines.txt`) |
| D7 | Same inputs, same bits, at no cost | Two runs of the same job give the same output hash, on the simulator and on the chip | simulator half DONE; chip half open |
| D8 | No per-process CUDA tax | The memory a CUDA++ process holds beyond its own buffers is measured and printed | PLANNED (needs a chip window) |
| D9 | No licence that stops you | At 2026-10-22, MIT and reverse-engineering/port rights; current candidate is Apache-2.0 WITH LLVM-exception with the exact MIT snapshot preserved (`docs/LICENSING_TRANSITION.md`) | Historical receipt `evidence/dx/pain-points/licence.txt` |
| D10 | No compiling at run time | hello-gpu prints its kernel build time and the hash of its machine code, identical across runs; no cache directory | DONE for hello-gpu |
| D11 | Open libraries with an easy default path | Any-shape GEMM, reductions, scans, sorts in source with host references | PLANNED (L386, Omega bar C3) |
| D12 | Work per answer, against the machine's own ceiling | Every example prints bytes moved, operations and the floor time at the measured ceiling; with a chip window, the measured time next to it | simulator half DONE; chip half open |
| D13 | Our own disassembler, plain text a person can read | Disassembler agrees with nvdisasm on every library kernel; nothing calls nvdisasm | PLANNED (L386, Omega bar C6) |
| D14 | Recover from a fault without restarting | A kernel that faults or never ends is stopped and the same process keeps working | PLANNED (L386, Omega bar C5) |

The swap story, replacing a developer's CUDA libraries and NVIDIA's kernel module with CUDA++'s, is ledger item L401. NVIDIA's kernel driver itself is being replaced by AIEN's own (L332, `ROADMAP.md`).

## 4. Where CUDA is better today

Said as plainly as the wins:

- **Chips.** CUDA runs on every NVIDIA GPU of the last decade. CUDA++ runs on one, the GB10. Row D6 makes the refusal honest; it does not add chips. Porting is the community's job as much as ours (`CONTRIBUTING.md`).
- **Libraries.** cuBLAS, cuDNN, NCCL, cuFFT, cuRAND, CUTLASS and the whole ecosystem above them. CUDA++ has matrix multiply and attention shaped for the models AIEN runs, and no FFT, random numbers or multi-GPU (`ROADMAP.md`).
- **Languages and frameworks.** PyTorch, JAX, Triton and every framework speak CUDA. CUDA++ today is a C kernel builder; Omega's languages reaching the GPU is Omega bar C1 (L386).
- **Asynchronous work.** CUDA's streams overlap independent work. CUDA++'s launches are synchronous by default and its command lists are the way to batch; streams and events are Omega bar C4.
- **Tools.** Nsight, cuda-gdb and compute-sanitizer are mature. CUDA++ has a simulator and receipts, and no profiler GUI.
- **Speed across the board.** CUDA++ is measured against this machine's ceiling, and the head-to-head with llama.cpp and vLLM on the Spark is AIEN item L394. Until it lands, no speed claim against CUDA is made here.

## 5. Checking this document

The citation ledger (`evidence/dx/pain-points/ledger.json`) maps each [n] to its URL and holds the verbatim quotes. Each quote was accepted only if it appears in the saved text of the page. The script and the command:

```
python3 sources.py --ledger evidence/dx/pain-points/ledger.json verify docs/PAIN_POINTS.md --evidence
```

`sources.py` is the grounded-citations script used by AIEN's agents; the copy in `tools/dx/sources.py` carries this candidate's Apache-2.0 WITH LLVM-exception notice.

## Sources

[1] https://github.com/sudoingX/dgx-spark-laguna — sudoingX/dgx-spark-laguna README
    > "The published Docker images do not carry sm_121a kernels, a missing environment variable silently breaks the NVFP4 MoE, and a deep-context request with speculative decoding will hard hang the entire machine."
    > "Without it the NVFP4 MoE CUTLASS kernels are not built for GB10 and you get CUDA error: no kernel image is available for execution on the device ."
[2] https://github.com/sudoingX/dgx-spark-laguna/blob/main/FINDINGS.md — dgx-spark-laguna FINDINGS.md
    > "The Spark ships with zero swap. On unified memory a spike goes straight to a global OOM that hard hangs the whole box. No ssh, no ping, needs a power cycle."
    > "A deep prefill forces a live JIT for a huge novel shape mid-inference, and that compile or kernel deadlocks the GPU on `sm_121a`."
    > "The published `vllm/vllm-openai:cu129-nightly` image resolves to `0.23.1rc1.dev`, not 0.25.1, and its kernels are not built for `sm_121a`."
[4] https://github.com/state-spaces/mamba/issues/1019 — No prebuilt wheels for sm_103 (B300) and sm_121 (GB10)
    > "pip install mamba-ssm  # installs prebuilt wheel for sm_75-sm_90"
[5] https://github.com/Dao-AILab/causal-conv1d/issues/119 — No prebuilt wheels for sm_121 (GB10 / DGX Spark)
    > "`causal-conv1d` does not ship prebuilt wheels for **sm_121** (GB10 / DGX Spark / Project DIGITS). Users must compile from source."
    > "Additionally, issue #105 documents a **numerical correctness bug** for sm_103 (B300): cubins built with CUDA 13.x nvcc for sm_103 produce wrong output (max_abs_diff ~0.4-0.6)."
[6] https://github.com/flashinfer-ai/flashinfer/issues/5238 — [Bug] SM121 (DGX Spark GB10): mixed mxfp8 x mxfp4 fused MoE dies in moe_sort routing with "no kernel image is available"
    > "the mixed path still dies at routing time because a prebuilt, SM100-only `moe_utils` artifact takes precedence over the local JIT build."
[7] https://github.com/flashinfer-ai/flashinfer/issues/3170 — DGX Spark (SM121) Current Support Audit
    > "**SM121 falls through to cutlass/cudnn only.**"
[8] https://github.com/triton-lang/triton/issues/10331 — Docs: bring-up recipe for Triton on NVIDIA Blackwell sm_121 (GB10 / DGX Spark / consumer Blackwell)
    > "I was unable to find an authoritative doc page or pinned issue with the correct bring-up steps, and the workarounds that surface in search results actively make things worse"
    > "Those "fixes" produce a kernel that loads and then computes wrong numbers, which is so much worse than a clean "no kernel image" crash."
    > "a warm `~/.triton/cache` populated on an `sm_120` box (or an earlier toolchain) gets reused on `sm_121` and silently miscompiles instead of rebuilding."
[9] https://github.com/Dao-AILab/flash-attention/issues/1969 — Support for compute capability 12.1 (sm121) - NVIDIA GB10 GPU
    > "Building from source fails with kernel mismatch: "FATAL: kernel built for sm80-sm100, but running on sm121""
    > "Minimum and Maximum cuda capability supported by this version of PyTorch is
    (8.0) - (12.0)"
[10] https://github.com/pytorch/pytorch/issues/31285 — Cuda error: no kernel image is available for execution on the device
    > "But starting from 1.3.1, we start to compile binaries only for arch 3.7 and up."
[11] https://github.com/pytorch/pytorch/issues/49161 — RuntimeError: CUDA error: no kernel image is available for execution on the driver, when use pytorch 1.7 on linux with R
    > "I am a skilled user of pytorch-gpu, recently I purchased an RTX 3090 server, but the bug with pytorch 1.7 and RT 3090 makes me mad."
[12] https://github.com/pytorch/pytorch/issues/159207 — Add official support for CUDA sm_120 (RTX 5090 / Blackwell architecture)
    > "Users are forced to either build PyTorch from source or use experimental nightly builds with limited platform support."
[13] https://github.com/pytorch/pytorch/issues/164342 — Official support for sm_120 (RTX 50-series / Blackwell) in stable PyTorch builds
    > "As a researcher, I purchased RTX 5070 Ti for deep learning workloads, but currently it cannot be used productively with DLC due to this gap."
[14] https://github.com/vllm-project/vllm/issues/13306 — [Feature]: Support for RTX 5090 (CUDA 12.8)
    > "NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation."
[15] https://github.com/vllm-project/vllm/issues/14452 — [Doc]: Steps to run vLLM on your RTX5080 or 5090!
    > "Please note here that we can't use precompiled vLLM because `vllm-project/vllm` has not moved to the required torch and CUDA versions yet."
    > "This causes a nightly to be silently installed, which fails at runtime due to an `undefined-symbol` error because `_C.abi3.so` fails at runtime."
[16] https://github.com/pytorch/pytorch/issues/157517 — [release 2.8-2.9] Delete support for Maxwell, Pascal, and Volta architectures for CUDA 12.8 and 12.9 builds
    > "Update: Option 2 is not a possible due to large binary size."
[19] https://github.com/graphdeco-inria/gaussian-splatting/issues/332 — Error installation of diff-gaussian-rasterization and found solution
    > "I followed the README and the video at https://www.youtube.com/watch?v=UXtuigy_wYc but i fell in a slump for a few hours due to persistent errors in installing as suggested."
[20] https://github.com/Dao-AILab/flash-attention/issues/246 — No Module Named 'torch'
    > "I don't know a right solution that works for all setups, happy to hear suggestions."
[21] https://github.com/Dao-AILab/flash-attention/issues/451 — Flash Attention 2.0 doesn't work: undefined symbol: _ZNK3c1010TensorImpl27throw_data_ptr_access_errorEv
    > "undefined symbol: _ZNK3c1010TensorImpl27throw_data_ptr_access_errorEv"
[23] https://github.com/Dao-AILab/flash-attention/issues/1799 — Problems with the installation of flash_attn with pytorch 2.8.0+cu128
    > "Precompiled wheel not found. Building from source..."
[28] https://forums.developer.nvidia.com/t/what-a-missed-opportunity-for-nvidia/353585 — What a missed opportunity for nvidia
    > "How many times have you tried to run something that doesn’t recognize SM121 and fails?"
    > "It took several weeks of community effort here to get vllm working reasonably well, in both standalone and cluster configuration."
[29] https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069 — PSA: State of FP4/NVFP4 Support for DGX Spark in VLLM
    > "As of today, FP4 is not properly utilized in current VLLM builds on our hardware, so you lose a lot of performance picking NVFP4 quants compared to AWQ 4-bit ones."
[31] https://forums.developer.nvidia.com/t/gemma-4-on-dgx-spark-gb10-system-freeze-at-80-utilization-sm-121-kernel-issues/366060 — Gemma 4 on DGX Spark (GB10): System Freeze at >80% Utilization & sm_121 Kernel Issues
    > "I am experiencing a critical system-level freeze (hard-lock requiring power cycle) on the DGX Spark when running Gemma 4 at high memory utilization."
    > "Official aarch64 images (e.g., vllm-openai:latest ) lacks sm_121 targets in their prebuilt binaries, defaulting to sm_120 , which causes decoding errors or crashes."
[32] https://forums.developer.nvidia.com/t/my-dgx-spark-hangs-is-this-normal/365930 — My DGX Spark Hangs ... is this normal?
    > "The whole DGX Spark machine hangs completely—no mouse or keyboard response, nothing. no visual errors, no logs errors ."
[33] https://forums.developer.nvidia.com/t/running-llama-3-1-8b-fp4-get-triton-error-value-sm-121a-is-not-defined-for-option-gpu-name/348808 — Running Llama-3.1-8B-FP4 get triton error. Value 'sm_121a' is not defined for option 'gpu-name'
    > "ptxas-blackwell fatal : Value ‘sm_121a’ is not defined for option ‘gpu-name’"
[34] https://forums.developer.nvidia.com/t/help-running-nemotron-3-nano-30b-a3b-fp8-on-dgx-spark-gb10/354771 — Help running Nemotron 3 Nano 30B-A3B-FP8 on DGX Spark (GB10)
    > "Conclusion: Nemotron 3 Nano 30B currently cannot run on DGX Spark via SGLang due to Triton’s lack of sm_121a support."
    > "Update 2: Setting TRITON_PTXAS_PATH doesn’t help - Triton ignores it in newer versions and uses its bundled ptxas which doesn’t support sm_121a."
[35] https://forums.developer.nvidia.com/t/dgx-spark-cuda-install-pitfalls-on-ubuntu-24-04-arm64-fixed/349881 — DGX Spark: CUDA Install Pitfalls on Ubuntu 24.04 (ARM64) – FIXED
    > "getting a working CUDA toolkit (with nvcc ) turned into a trap maze."
    > "that value reflects the driver’s runtime capability , not the presence of the toolkit ."
[38] https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744 — DGX Spark GB10 Cuda 13.0 Python 3.12 SM_121
    > "FATAL: kernel fmha_cutlassF_f16_aligned_64x128_rf_sm80 is for sm80-sm100, but was built for sm121"
    > "i simply cannot believe GB10 is that slow in Spark …."
    > "Every PyTorch build with CUDA 12.8 and sm_120 support already supports sm_121 (DGXSpark) as these architectures are binary compatible."
[40] https://discuss.pytorch.org/t/the-detected-cuda-version-12-0-mismatches-the-version-that-was-used-to-compile-pytorch-11-7/169586 — The detected CUDA version (12.0) mismatches the version that was used to compile PyTorch (11.7)
    > "The PyTorch binaries do not support CUDA 12 yet and you would have to build from source using a locally installed CUDA 12.0 toolkit."
[41] https://simonwillison.net/2025/Oct/14/nvidia-dgx-spark — NVIDIA DGX Spark: great hardware, early days for the ecosystem
    > "I eventually managed to get a PyTorch 2.7 wheel for CUDA on ARM, but failed to do so for 2.8."
    > "I have not yet got my head around the difference between CUDA 12 and 13. 13 appears to be very new, and a lot of the existing tutorials and libraries appear to expect 12."
[42] https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark — NVIDIA DGX Spark In-Depth Review: A New Standard for Local AI Inference
    > "However, the only downside of this machine lies in memory bandwidth, the unified memory is LPDDR5x, offering up to 273 GB/s , shared across both CPU and GPU."
[44] https://news.ycombinator.com/item?id=45575127 — NVIDIA DGX Spark In-Depth Review (HN)
    > "To me it seems like you're paying more than twice the price mostly for CUDA compatibility."
[45] https://vllm.ai/blog/2025-08-11-cuda-debugging — CUDA Core Dump: An Effective Tool to Debug Memory Access Issues and Beyond (vLLM blog)
    > "In our experience the python stack traces for these types of exceptions are basically always incorrect and pretty worthless."
    > "Many people launch CUDA kernels using the kernel<<<>>> syntax without adding error checking for the kernel launch status, for example, this code."
    > "Have you ever felt you are developing cuda kernels and your tests often run into illegal memory access (IMA for short) and you have no idea how to debug? We definitely felt this pain again and again while working on vLLM, a high-performance inference engine for LLM models."
[46] https://docs.pytorch.org/docs/2.14/notes/cuda.html — CUDA semantics — PyTorch docs
    > "When you call a function that uses the GPU, the operations are enqueued to the particular device, but not necessarily executed until later."
    > "However, the unused memory managed by the allocator will still show as if used in nvidia-smi."
    > "This can reduce fragmentation and may allow some borderline workloads to complete without running out of memory."
[47] https://forums.developer.nvidia.com/t/271625 — CUDA errors: determine "sticky-ness" (NVIDIA Developer Forums)
    > "Pretty much all asynchronous errors (these are errors that occur as a result of device code execution) are sticky."
    > "My goal is to determine whether or not restart the executable, otherwise I will restart it for every cuda-related error, but it is a pity."
[48] https://forums.developer.nvidia.com/t/225877 — Compute-sanitizer not quite a drop-in replacement of cuda-memcheck (NVIDIA Developer Forums)
    > "This is hard to use in CI build farms that have a very strict network configuration."
[49] https://developer.nvidia.com/blog/better-bug-detection-how-compile-time-instrumentation-for-compute-sanitizer-enhances-memory-safety — Better Bug Detection: How Compile-Time Instrumentation for Compute Sanitizer Enhances Memory Safety (NVIDIA Technical Bl
    > "Compile-time instrumentation catches bugs that binary instrumentation misses, such as buffer overruns into neighboring allocated memory, while lowering runtime overhead in some cases by nearly an order of magnitude."
[50] https://docs.pytorch.org/docs/2.14/notes/randomness.html — Reproducibility — PyTorch docs
    > "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."
[51] https://docs.nvidia.com/cuda/cublas/index.html — cuBLAS documentation — Results reproducibility
    > "set a debug environment variable CUBLAS_WORKSPACE_CONFIG to :16:8 (may limit overall performance) or :4096:8 (will increase library footprint in GPU memory by approximately 24MiB)."
    > "In that case, the results are not guaranteed to be bit-wise reproducible because atomics are used for the computation."
[52] https://github.com/Dao-AILab/flash-attention/issues/1038 — Build flash-attn takes a lot of time · Issue #1038 · Dao-AILab/flash-attention
    > "For reference I just compiled it on a 2018 Dell laptop (Quadro P1000 GPU) and the build time was about 15 hours."
    > "Without both changes it defaulted to just using a single compilation job for me, taking forever (I gave up after an hour)."
[53] https://pypackaging-native.github.io/key-issues/gpus — Packaging projects with GPU code — pypackaging-native
    > "GPU packages tend to result in very large wheels. This is mainly because compiled GPU libraries must support a number of architectures, leading to large binary sizes."
    > "For example, recent manylinux2014 wheels for TensorFlow are 588 MB (wheels for 2.11.0 ), and for PyTorch those are 890 MB (wheels for 1.13.0 )."
[54] https://docs.pypi.org/project-management/storage-limits — Storage Limits — PyPI Docs
    > "Limit for project 'sampleproject' is 100 MB."
[55] https://github.com/pypi/support/issues/10830 — File Limit Request: faiss-gpu - 250 MB · pypi/support #10830
    > "There are currently 34 other open limit request issues ahead of this one."
[56] https://developer.nvidia.com/blog/streamline-cuda-accelerated-python-install-and-packaging-workflows-with-wheel-variants — Streamline CUDA-Accelerated Python Install and Packaging Workflows with Wheel Variants (NVIDIA Technical Blog)
    > "Docker images for ML and AI workloads are notoriously large, often exceeding 10 GB. Much of this comes from bundled GPU libraries for all possible architectures."
[57] https://forums.developer.nvidia.com/t/196340 — Cudnn dll for cuda 11 are too big for redistribution (NVIDIA Developer Forums)
    > "We agree that the growth of the cuDNN DLL size is problematic, and we’re working on resolving this."
[58] https://forums.developer.nvidia.com/t/147271 — Any suggestions to reduce the size of libcudnn.so? (NVIDIA Developer Forums)
    > "Do you have any suggestions to remove unused symbols to reduce this shared library size for deployment? libcudnn.so.7 is over 200 MB in size!"
[59] https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching — CUDA Pro Tip: Understand Fat Binaries and JIT Caching (NVIDIA Technical Blog)
    > "This JIT compilation can cause delay at application start-up time (or more accurately, CUDA context creation time)."
    > "Insufficient cache size or a cache stored on a slow network share can cause long application start-up times"
[60] https://docs.flashinfer.ai/installation.html — Installation — FlashInfer documentation
    > "flashinfer-python: Core package that compiles/downloads kernels on first use"
[61] https://github.com/vllm-project/vllm/issues/41865 — FlashInfer GDN JIT Compilation Causes Multi-Worker Deadlock · vLLM #41865
    > "Worker 0 completes compilation (172s) but workers 1-7 hang indefinitely."
[62] https://github.com/triton-lang/triton/commit/1f983791498aab4f8f3a922ac8128a59ee3a328b — Cache autotune timings to disk (#6261) · triton-lang/triton
    > "Some users had expressed a desire to cache autotune results, both to speed up local iteration, and to avoid re-tuning when scaling up to large numbers of GPUs."
[63] https://docs.nvidia.com/cuda/eula/index.html — License Agreement for NVIDIA Software Development Kits — EULA
    > "You may not reverse engineer, decompile or disassemble any portion of the output generated using SDK elements for the purpose of translating such output artifacts to target a non-NVIDIA platform."
[64] https://www.tomshardware.com/pc-components/gpus/nvidia-bans-using-translation-layers-for-cuda-software-to-run-on-other-chips-new-restriction-apparently-targets-zluda-and-some-chinese-gpu-makers — Nvidia bans using translation layers for CUDA software — Tom's Hardware
    > "Nvidia has banned running CUDA-based software on other hardware platforms using translation layers in its licensing terms listed online since 2021, but the warning previously wasn't included in the documentation placed on a host system during the installation process."
    > "The restriction appears to be designed to prevent initiatives like ZLUDA, which both Intel and AMD have recently participated, and, perhaps more critically, some Chinese GPU makers from utilizing CUDA code with translation layer"
[65] https://www.theregister.com/software/2024/08/09/amd-lawyers-claw-back-cuda-compatibility-layer-zluda/1009658 — AMD lawyers claw back CUDA compatibility layer ZLUDA — The Register
    > "AMD's legal team appears to have clawed back control of much of the ZLUDA project's code base."
[66] https://www.tomshardware.com/pc-components/gpus/amd-asks-developer-to-take-down-open-source-zluda-dev-vows-to-rebuild-his-project — AMD asks developer to take down open source ZLUDA — Tom's Hardware
    > "The code was released with AMD's approval through an email. AMD's legal department now says it's not legally binding, hence the rollback."
[67] https://github.com/vosen/ZLUDA — vosen/ZLUDA README (GitHub)
    > "ZLUDA is a drop-in replacement for CUDA on non-NVIDIA GPUs."
[68] https://github.com/vosen/ZLUDA/discussions/261 — Code rollback · vosen/ZLUDA Discussion #261
    > "But since its amd shooting its own foot this project will likely not survive this setback."
[69] https://github.com/ROCm/legacy-rocm-build/issues/2308 — Support your GPUs for 8+ years, like Nvidia does, including gfx906 GPUs · ROCm #2308
    > "If you want to know why everyone is going with Nvidia, there are a lot of them, but what they do at least well is they support they GPUs."
[70] https://github.com/ROCm/legacy-rocm-build/issues/1714 — Which devices are even supported? (HIP/ROCm) · ROCm #1714
    > "I'm a long-time CUDA developer looking to explore ROCm and HIP development, but finding out which hardware even supports these tools is harder than it needs to be."
[71] https://github.com/ROCm/TheRock/issues/9198 — Strix Halo gfx1151: 93 ML experiments, 5 critical bf16 bugs · ROCm/TheRock #9198
    > "We ran 93+ autonomous ML training experiments on a Strix Halo system (Radeon 8060S, gfx1151) using TheROCk nightly PyTorch and discovered 5 critical bf16 bugs, an undocumented 19x attention speedup, and several ecosystem gaps that prevent consumer AMD GPUs from being viable ML research platforms."
[72] https://news.ycombinator.com/item?id=39888020 — Documentation for the AMD 7900XTX — Hacker News (comment by georgehotz)
    > "AMD is structurally incapable of fixing these issues. They don't even have a 7900XTX in CI."
[74] https://news.ycombinator.com/item?id=39801234 — CUDA Is Still a Giant Moat for Nvidia — Hacker News
    > "On CUDA just runs of course with sliding window attention, ALiBi, and PagedAttention. ROCm fork? Nope."
    > "AMD needs to be funding a CUDA shim that allows people to port stuff directly to their cards."
[75] https://news.ycombinator.com/item?id=40970560 — Run CUDA, unmodified, on AMD GPUs — Hacker News (1330 points)
    > "you have to be able to pip install something and just have it work, reasonably fast, without crashing, and also it has to not interfere with 100 other weird poorly maintained ML library dependencies."
[76] https://www.modular.com/blog/democratizing-ai-compute-part-4-cuda-is-the-incumbent-but-is-it-any-good — CUDA is the incumbent, but is it any good? (Democratizing AI Compute, Part 4) — Modular
    > "PTX is only partially documented, constantly shifting between hardware generations, and effectively a black box for developers."
    > "NIM also abstracts away underlying operations, increasing lock-in and limiting access to the low-level optimization and innovation key to CUDA's value proposition."
    > "CUDA doesn’t provide performance portability across NVIDIA’s own hardware generations, and the sheer scale of its libraries is a double-edged sword."
    > "For AI engineers who build on top of CUDA, it’s an essential tool, but one that comes with versioning headaches, opaque driver behavior, and deep platform dependence."
[77] https://news.ycombinator.com/item?id=48910260 — Alternative(s) to run CUDA on non-Nvidia hardware — Hacker News comment
    > "With a design like Cuda where there is always a default easy path, and a complex but optional path."
    > "That so many people have skill issues is the reason why Cuda trumps and will continue to trump Vulkan despite being vendor-locked."
[78] http://tinygrad.org/pitch.pdf — the tiny corp pitch deck (George Hotz, tinygrad.org)
    > "By being the dominant player, green team gets a whole ecosystem of developers to improve the developer experience for them."
[79] https://news.ycombinator.com/item?id=48903715 — Alternative(s) to run CUDA on non-Nvidia hardware — Hacker News
    > "One of the biggest complaints we hear from the industry is "we tried to port to X and we could never complete it"."
[82] https://news.ycombinator.com/item?id=45165571 — Intel Arc Pro B50 GPU Launched at $349 — Hacker News comment
    > "Nvidia uses VRAM amount for market segmentation. They can't make a 128GB consumer card without cannibalizing their enterprise sales."
[83] https://news.ycombinator.com/item?id=49151324 — Qwen3.8-Max — Hacker News comment
    > "There is no math that will make local models come out ahead in $/intelligence/token."
    > "The point of local models is privacy, offline use, and maybe no guard rails."
[84] https://news.ycombinator.com/item?id=48089658 — Running local models on an M4 with 24GB memory — Hacker News comment
    > "At $20/month for a cloud AI subscription, you’re looking at almost 30 years of service for the same money."
[85] https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-second-quarter-fiscal-2027 — NVIDIA Announces Financial Results for Second Quarter Fiscal 2027
    > "Data Center revenue of $89.0 billion, up 117% from a year ago"
    > "NVIDIA (NASDAQ: NVDA) today reported revenue for the second quarter ended July 26, 2026, of $96.2 billion"
[87] https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm — NVIDIA Form 10-K for fiscal year ended Jan 26, 2025 (SEC)
    > "Data Center revenue for fiscal year 2025 was up 142% from a year ago."
    > "Revenue for fiscal year 2025 was $130.5 billion, up 114% from a year ago."
[88] https://www.reddit.com/r/LocalLLaMA/comments/1r569eb/psa_nvidia_dgx_spark_has_terrible_cuda_software — r/LocalLLaMA: PSA: NVIDIA DGX Spark has terrible CUDA & software compatibility
    > "Just be careful when buying a DGX Spark. You are not really getting a modern CUDA experience."
    > "The funniest part is paying the NVIDIA tax specifically for CUDA compatibility and then not getting it."
    > "Very frustrating compared to most every other GPU release that has zero day support from cuda/nvcc/pytorch/tensorflow."
    > "many of their official playbooks are broken and can’t be run as-is on the latest versions."
    > "The sm121 fragmentation is the more interesting problem here."
[89] https://www.reddit.com/r/LocalLLaMA/comments/1no4ho1/some_things_i_learned_about_installing_flashattn — r/LocalLLaMA: Some things I learned about installing flash-attn
    > "This cut my build time from 1+ hours to just under 15 minutes. Thank you!"
    > "Usually the reason is a different torch. There's a whole issue on FA about people complaining."
[90] https://www.reddit.com/r/LocalLLaMA/comments/1oo6226/disappointed_by_dgx_spark — r/LocalLLaMA: Disappointed by dgx spark
    > "You design and validate them locally on the spark, running the exact same software, then push to the data center full of Nvidia GPU racks."
    > "Sure, Nvidia gives for "free" their software stack.... as long as you use it on their products."
[91] https://www.reddit.com/r/LocalLLaMA/comments/1sqera2/vllm_woes_in_spark — r/LocalLLaMA: VLLM woes in Spark
    > "The CUDA tax, which is 4.6 GB approximately per each model on a DGX spark."
[92] https://www.reddit.com/r/LocalLLaMA/comments/1lx62hd/nvidia_being_nvidia_fp8_is_150_tflops_faster_when — r/LocalLLaMA: Nvidia being Nvidia: FP8 is 150 Tflops faster when kernel name contain "cutlass"
    > "Even worse, if someone accidentally created a kernel with "cutlass" in the name, the driver would apply optimizations that are not safe."
[93] https://www.reddit.com/r/pytorch/comments/1l3uwk2/should_compiling_from_source_take_a_terabyte_of — r/pytorch: Should compiling from source take a terabyte of memory?
    > "It keeps crashing with the nvcc error out of memory, even after I've allocated over 0.75TB of vRAM on my SSD."
[94] https://www.reddit.com/r/LocalLLaMA/comments/1k903ov/llamacpp_cuda_setup_running_into_issues_is_it — r/LocalLLaMA: Llama.cpp CUDA Setup - Running into Issues - Is it Worth the Effort?
    > "However, CMake is unable to find the CUDA toolkit, even though it's installed and `nvcc` and `nvidia-smi` are working correctly."
[95] https://www.reddit.com/r/LocalLLaMA/comments/1b6gfjt/cuda_crackdown_nvidias_licensing_update_targets — r/LocalLLaMA: CUDA Crackdown: NVIDIA's Licensing Update targets AMD and blocks ZLUDA
    > "i literally worked at a company where we dropped HIP support, and just went all in on nvidia."
    > "I hope someone is working on hardware agnostic solutions, we need more GPUs, not less."
[96] https://www.reddit.com/r/CUDA/comments/1jhvtqm/patch_to_enable_pytorch_on_rtx_5080_cuda_128_sm — r/CUDA: Patch to enable PyTorch on RTX 5080 cuda 12.8 + sm_120 / Blackwell support
    > "PyTorch doesn’t support sm_120 or the RTX 5080 out of the box. So I patched it."
[97] https://www.reddit.com/r/LocalLLaMA/comments/1scf1x8/dont_buy_the_dgx_spark_nvfp4_still_missing_after — r/LocalLLaMA: Don't buy the DGX Spark: NVFP4 Still Missing After 6 Months
    > "It was not marketed like an experimental dev kit where buyers should expect to spend months switching backends, testing builds, setting flags, and relying on community or hardcore fan fixes just to make a core feature work properly."
