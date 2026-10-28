# Ampere device qualification plan

**RTX 3090 (GA102): NOT_RUN. RTX 3060 Ti (GA104): NOT_RUN.**
No RTX card is attached to the development Spark. Host, emulator, simulated-driver,
ELF inspection and nvdisasm results do not change either device status. This plan
is for a separate Linux x86_64 machine with an explicitly authorized GPU window.
It never stops a serving process, changes clocks, loads a module, or installs a driver.

## Prerequisites and exact target

NVIDIA lists both products at compute capability 8.6:
[CUDA GPUs](https://developer.nvidia.com/cuda/gpus.md).
The [3090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)
list 24 GB GDDR6X; [3060 Ti specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/)
list 8 GB GDDR6/GDDR6X. Do not use system RAM as the discrete allocation budget.

The initial direct-driver ABI is pinned to NVIDIA **580.173.02**, matching
`third_party/nvidia-open-580.173.02`. The
[pinned open-module README](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.173.02/README.md)
supports Linux x86_64 and Turing or later, including these cards. Open-module
support is not evidence that CUDA++'s direct RM/UVM path has run on them.
Do not bypass a mismatched-driver refusal. A different driver needs its own
reviewed header/ABI update before device execution.

The doctor accepts the requested GA102/GA104 PCI IDs: 2204 (3090), 2486 and
2489 (3060 Ti). Other products sold under a 3060 Ti name have distinct PCI IDs;
they are not silently treated as GA104. The actual PCI ID must be recorded.

Drake must provide the machine holding each card, its OS, driver version, and SSH
access. On that machine, obtain its PCI BDF with the read-only doctor or `lspci`.
`CUDAPP_GPU_BDF` selects that same exact device in the runtime. No device tests
are authorized on the development Spark by this plan.

## Build without a GPU

```sh
make all hello-gpu-build ampere-device-build
make test-ampere-plan
```

Use the portable generated-from-Omega units. Do not substitute the migration C
encoder, AArch64 machine-code trampolines, or a QEMU-AArch64 run for an x86 build.
Keep the source commit, converter pin, seeded compiler hash, generated provenance,
compiler version and full build/test logs with the run. The normal host suite is
`make test`; the Ampere-specific nvdisasm gate is:
`make OUT_DIR=build-ampere-oracle CUDAPP_AMPERE_CORPUS=1 test-ampere-encoder-oracle`.
It includes an intentionally corrupted encoding negative control that must fail.

## Read-only doctor

```sh
python3 tests/ampere/doctor.py --bdf 0000:01:00.0
```

Expected JSON: `preflight: PASS`, `machine: x86_64`, `open_module: true`, driver
`580.173.02`, exactly the selected GA102/GA104 GPU with `target: sm_86`, and
`device_execution: NOT_RUN`. It reads `/proc`, `/sys`, and node access bits only;
it never opens a GPU. PASS means the preconditions hold, not qualification.
A closed module, unknown card, missing node, ambiguous selection, or ABI mismatch
must exit 1. A `--root` filesystem fixture is always labeled **SIMULATED**.

## Authorized device run and receipts

Use a **new** evidence directory per card and source revision:

```sh
python3 tests/ampere/run.py --device --bdf 0000:01:00.0 \
  --output evidence/ampere-ga102-first-run --full-math
```

Run again on the GA104 machine with its BDF and another new output directory.
The runner refuses to overwrite existing receipts. `results.json` starts with
all checks `NOT_RUN`; only actually executed checks become PASS or FAIL. It
stops on a failure, preserving the failing log and leaving later checks NOT_RUN.
Without `--full-math`, `math-exhaustive` remains NOT_RUN even if smoke tests pass.

| Check | Expected output / criterion | GA102 | GA104 |
|---|---|---|---|
| doctor | Exact GPU, open module, pinned ABI; preflight PASS | NOT_RUN | NOT_RUN |
| hello | `checked 1048576 of 1048576 elements: 0 wrong, 0 never written`, exit 0 | NOT_RUN | NOT_RUN |
| hello negative control | One deliberately unwritten result, `1 wrong, 1 never written`, exit 1 | NOT_RUN | NOT_RUN |
| safe reserve refusal | `AMPERE MEMORY REFUSAL PASS`; no new RM allocation ioctl; free VRAM unchanged | NOT_RUN | NOT_RUN |
| math smoke | Four `SHARD PASS` lines; not exhaustive evidence | NOT_RUN | NOT_RUN |
| math negative control | Deliberately sign-corrupted first reciprocal result is caught, exit 1 | NOT_RUN | NOT_RUN |
| math exhaustive | Four `EXHAUSTIVE PASS` lines over all binary32 bit patterns, exit 0 | NOT_RUN | NOT_RUN |

### Vector add

```sh
CUDAPP_GPU_BDF=0000:01:00.0 build/hello_gpu --target=sm_86
CUDAPP_GPU_BDF=0000:01:00.0 build/hello_gpu_control --target=sm_86 --bug=short
```

The second command **must fail** with exit 1; no out-of-bounds access is involved.
Both programs verify every output element, print code/output hashes and synchronize
CPU/device visibility explicitly. Ampere output must not reuse GB10 bandwidth claims.
`--sim --target=sm_86` is an IR **SIMULATED** result, not SASS or device evidence.

### Exhaustive numerical checks

```sh
CUDAPP_GPU_BDF=0000:01:00.0 build/ampere_math_device --device
```

The verifier sweeps every binary32 input bit pattern for each emitted unary MUFU
function: reciprocal, reciprocal square root, base-two exponential and logarithm.
A bounded batch is uploaded/read back on every iteration; no full-domain buffer
is allocated. It checks NaN classification, infinities, signed zero and finite
error, against independent host long-double libm. NaN payload identity is not a
contract. A shard can be selected with `--start` and `--count`; it prints
`SHARD PASS`, never `EXHAUSTIVE PASS`. This is not an exhaustive binary-operand
Cartesian product or a claim about correctly rounded software divide/sqrt.

The intended non-FTZ bounds are NVIDIA PTX ISA 9.4 sections 9.7.3.13, 9.7.3.16,
9.7.3.20 and 9.7.3.21: reciprocal 1 ULP; reciprocal square root relative error
`2^-22.9`; logarithm absolute error `2^-22` for input in `(0.5,2)` and relative
error `2^-22` outside it; exponential 2 ULP. **Raw MUFU equivalence to these
PTX-level contracts, especially subnormal handling, is unqualified.** A failure
is a defect to fix in normalization/kernel generation, not permission to skip
inputs or enlarge a bound. A long-double libm comparison is a test oracle, not
a formal correctly-rounded reference proof.

Symbolic formula check used while writing this verifier:
`WolframLanguageEvaluator: {FullSimplify[1/Sqrt[x] == x^(-1/2), Assumptions->x>0], FullSimplify[Log[x]/Log[2] == Log[2,x], Assumptions->x>0], 2^(-22)} -> {True, True, 1/4194304}`.

Negative control (small, safe shard, still real device work):

```sh
CUDAPP_GPU_BDF=0000:01:00.0 build/ampere_math_device --device \
  --count 65536 --inject-output-error
```

It corrupts only the CPU-side readback after the GPU finishes. Expected:
`AMPERE MATH FAIL op=rcp input=00000000`, exit 1.
`--self-test` checks the host oracle's special-value/refusal controls without a GPU;
`--emit-only` emits all four kernels without opening a GPU.

### Safe memory refusal, not an OOM test

```sh
CUDAPP_GPU_BDF=0000:01:00.0 build/ampere_memory_refusal --device
```

The session opens first, then the test queries actual free and total VRAM, sets
`CUDAPP_GPU_MIN_FREE_BYTES` above total (therefore above free), and requests only a
small page. Even a concurrent free cannot turn this into a real allocation.
The request must be rejected by D1 **before any RM allocation ioctl**.
A link-time ioctl wrapper counts real allocation escapes independently of the
return code. A second RM heap query must report identical free and total VRAM.
Concurrent GPU users can change free VRAM and make this proof inconclusive/FAIL;
reserve an idle window rather than allocating pressure or relaxing the check.
Never allocate the card's capacity, exhaust UVM/system memory, or use a real OOM
as a refusal test. Host simulated-driver tests are explicitly SIMULATED.

## Recording qualification

Keep complete per-check logs and `results.json` with the exact source revision.
A doctor PASS, disassembler PASS, or x86 host PASS must never be copied into a
device result. No script modifies the qualified-device table. Only actual real-card
results with the required negative controls can support a separate reviewed
qualification change. Any failed check leaves the corresponding card unqualified.

## Experimental native Omega x86 oracle

This optional host-only probe does not change the production bridge-C route.
Use a checkout of omega-mv draft #923 at
`becf5c817bcb7e0846d12ebfc2be91042d8ad89a`; the probe verifies both that revision
and its tracked AArch64 compiler fingerprint before compiling anything.

```sh
python3 tests/ampere/native_omega_x86.py \
  --compiler-tree /path/to/pinned/omega-mv \
  --output build/ampere-native-x86
```

The probe compiles all four actual units to x86 and executes composed small-unit
callers under `qemu-x86_64`, comparing their output with native AArch64 and
requiring negative controls to fail. Label these results **EMULATED**, not native
x86 or device qualification. The cubin unit's full-size private-ABI buffers do
not fit Omega's local-array/stack limits; its native private-ABI execution remains
**NOT_RUN**, not a reduced-signature or fake-buffer PASS. The generated bridge-C
cubin tests exercise the full-size host boundary separately.
