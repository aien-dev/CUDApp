# RTX 3090 / RTX 3060 Ti Ampere port

**RTX 3090 (GA102): device NOT_RUN. RTX 3060 Ti (GA104): device NOT_RUN.**
This is host preparation, not a supported-hardware announcement. No RTX card is
attached to the development Spark, and this work must not submit to its GB10 or
interrupt its resident model. DONE below means the implementation or audit item
exists; exercised checks are recorded separately. A simulator, disassembler or
x86 emulator cannot qualify either GPU.

## Authoritative target and driver references

NVIDIA's [CUDA GPU table](https://developer.nvidia.com/cuda/gpus.md) places both
GeForce RTX 3090 and RTX 3060 Ti at compute capability **8.6 (sm_86)**. They are
not GB10/sm_121 and must not reuse its target tag. NVIDIA lists the
[3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)
as Ampere with **24 GB GDDR6X**, and the
[3060 Ti](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/)
as Ampere with **8 GB GDDR6/GDDR6X**, depending on the model. Physical capacity
is not an allocation budget: use the selected device's measured free VRAM.

NVIDIA's [580.173.02 open-module README](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.173.02/README.md)
supports Linux x86_64/aarch64 and Turing-or-newer GPUs, and lists these RTX 30
products. Thus “Ampere requires a closed kernel module” is false. That support
does **not** qualify CUDA++'s direct RM/UVM client. The client remains pinned to
vendored `third_party/nvidia-open-580.173.02`; another driver version is a
refusal, not an invitation to bypass the ABI check.

The local NVIDIA open-gpu-doc reference is revision
`9fdf5c4062007929d9f4e6cbad9c9771fe61b880`. Its
[`clc6c0.h:24-32`](https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clc6c0.h#L24-L32)
defines **AMPERE_COMPUTE_A = 0xc6c0**, whereas
[`clc7c0.h:24-32`](https://github.com/NVIDIA/open-gpu-doc/blob/9fdf5c4062007929d9f4e6cbad9c9771fe61b880/classes/compute/clc7c0.h#L24-L32)
defines **AMPERE_COMPUTE_B = 0xc7c0**. GA10x/sm_86 needs the latter, not the
GA100 COMPUTE_A class. Names and QMD layouts must not be inferred from a nearby
Blackwell header.

## Audit and implementation inventory

| Item | State | Implementation / boundary and source references |
|---|---|---|
| Explicit target selection | DONE | `src/omega_ampere_codegen.c:7` accepts exactly sm_86 or sm_121. Existing `omega_bw_encode_program` remains the explicit Blackwell API. `examples/hello_gpu/hello_gpu.c` accepts `--target=sm_86`; the session compares `OmegaGpuLaunch.target_sm` against the selected GPU before submission (`src/omega_gpu_session.c:536`). Zero retains the historical sm_121 contract; it is not silent cross-target selection. |
| Ampere instruction encoding | DONE | First-party implementation is `om/encoder/ampere_enc_sm86.om`, not a hand-written C fallback. Ampere argument constants start at 0x160 instead of Blackwell's 0x380. Target-specific differences include uniform constant loads, global-memory descriptor/address fields and atomic encodings. The portable binding is `src/cudapp_om_ampere_enc_sm86.c`; host marshaling is `src/omega_ampere_codegen.c`. |
| Offline instruction oracle | DONE | `mk/ampere-encoder.mk` captures emitted host-suite kernels, disassembles their sm_86 bytes with pinned nvdisasm 13.0.85, and checks decoded instructions/operands. Its corrupted-encoding negative control must fail. The build-only math kernels participate via `tests/ampere/math_device.c --emit-only`. CI uses a separate corpus output directory so baseline and instrumented objects cannot be mixed. |
| Blackwell-only instruction semantics | BLOCKED | Blackwell FP8 conversion / QMMA forms have no same-semantics sm_86 encoding. The sm_86 encoder refuses them rather than inventing bytes or silently executing a different operation. Existing Blackwell-specialized model/tensor paths are not made Ampere-capable by changing an ELF flag. A real Ampere algorithm lowering is required before those paths are enabled. |
| Cubin/ELF target | DONE | `om/cubin/cubin_elf.om:26` selects the parameter base; `:269` selects CUDA ELF OSABI/ABI and e_version; `:286` sets the sm_86 flags. `omega_cubin_pack_target` is the target-explicit host boundary in `src/omega_cubin.c`. The old sm_121 entry remains explicit/default-compatible. `mk/ampere-cubin.mk` checks ELF fields and parameter metadata, differential behavior, and offline nvdisasm loading. |
| Capability-mercury sections | DONE | The sm_86 writer emits no `.nv.capmerc.*` / `.nv.merc.*` sections. The Blackwell driver-loader requirement observed for sm_121 must not be generalized to Ampere. Successful offline ELF decoding is not evidence of successful `cuModuleLoadData` or direct RM launch; both RTX loader/device paths remain NOT_RUN. |
| Compute/FIFO classes and QMD | DONE | `src/omega_ampere_qmd.h:5` records C6C0 versus C7C0 and Ampere FIFO constants. `om/encoder/amp_runtime.om` implements target admission, QMD3.0 layout and bounded command admission. `src/omega_gpu_session.c:562` emits the Ampere QMD and `:599` its PCAS launch instead of Blackwell QMD5/inline packets. `nvrm/nvrm.c:69` requires the queried class list to contain the selected target's classes. Raw Blackwell packets on an Ampere device are refused before ring submission (`nvrm/nvrm.c:1055`). |
| Discrete allocation and copies | DONE | `nvrm/nvrm.c:631` selects local framebuffer memory on sm_86, RM maps it into GPU VA and maps its CPU BAR alias through real RM_MAP_MEMORY/mmap. Queue and host-polled completion storage remain system memory. `nvrm/nvrm.c:746` / `:755` provide live-identity/bounds-checked CPU→BAR and BAR→CPU transfers with ordering fences. The session may write/read that uncached BAR view directly; platform sync (`src/omega_gpu_platform_linux.c:358`) orders visibility rather than pretending to copy coherent host RAM. No copy engine is claimed. |
| Cached local mappings / large BAR availability | BLOCKED | Cached Ampere mappings are explicitly refused until their cache-maintenance contract is implemented. CPU BAR mapping availability and large-allocation behavior need the real driver/card configuration; failure is surfaced, not replaced by a fake host allocation. First-run tests use bounded allocations. |
| VRAM-aware D1 reserve | DONE | `nvrm/nvrm.c:637` queries the selected framebuffer heap. `src/omega_gpu_platform_linux.c:233` reads reserve policy and `:248` uses free VRAM, never host MemAvailable, for local allocations. `CUDAPP_GPU_MIN_FREE_BYTES` supports a precise safe refusal; the default MiB reserve remains documented by the runtime. The refusal precedes the driver allocation call. |
| Safe reserve refusal proof | DONE | `tests/ampere/memory_refusal.c:39` sets reserve above total VRAM (therefore above free VRAM even if other processes release memory), then requests only a small allocation. It requires refusal, zero intervening RM allocation ioctls and unchanged free VRAM. Simulated-driver coverage is separate from this real-device test. It never pressure-allocates VRAM. |
| Open-module and device selection doctor | DONE | `tests/ampere/doctor.py:79` verifies Linux x86_64, the exact pinned open module, requested PCI IDs, device nodes and an unambiguous BDF. `CUDAPP_GPU_BDF` selects that exact GPU in the driver client. `test_doctor.py` uses a fake filesystem and is always SIMULATED. No driver is opened by the doctor. |
| Portable Omega build provenance | IMPLEMENTED | `om/units.sh` covers all canonical hosted roots plus Ampere encoding/runtime and the separate revision/epoch freestanding profile. Checked portable generated translation units replace executable-code trampolines. `om/omega-bridge.lock`, per-unit identities and generated declarations retain exact composed-source SHA → C SHA provenance. Missing/mismatched converter, profile or seed refuses before installation. Final host acceptance is recorded separately below. |
| Native x86_64 host build | DONE | `.github/workflows/ampere-x86.yml:13` uses a native Ubuntu x86 runner for the full host build/tests and offline oracle. `src/omega_numeric.c:31` adds the x86 host-reference tier using scalar SSE/fused libm where appropriate; integer-defined conversions are not mislabeled as an independent conversion oracle. Local x86 emulation, if run, is labeled HOST_EMULATED separately. |
| Device test plan / runner | DONE | `tests/ampere/README.md`, `doctor.py`, `run.py`, `math_device.c`, and `memory_refusal.c` cover doctor, complete vector-add output, intentionally failing vector-add, exhaustive unary math, a numerical negative control and safe memory refusal. The runner starts each result at NOT_RUN, records real outputs and stops on failure. |
| GA102 / GA104 execution and qualification | BLOCKED | No RTX device is attached or authorized here. Both device states remain NOT_RUN. Need the actual host, OS, driver version, SSH access and a GPU window before any real execution. |

## Evidence and limits

The acceptance commands and status belong with the draft PR and its observed
logs. Merely adding a CI job does not mean it passed. `make test-ampere-plan`
contains only host/simulated checks; `make ampere-device-build` builds device
programs without executing them. The instruction oracle command is:

```sh
make OUT_DIR=build-ampere-oracle CUDAPP_AMPERE_CORPUS=1 test-ampere-encoder-oracle
```

The [device plan](../tests/ampere/README.md) keeps separate NOT_RUN entries for
every check on each card. Exhaustive math uses the intended NVIDIA PTX bounds,
not an invented tolerance or a claim that raw MUFU matches PTX normalization.
A failing subnormal case needs a real fix, not a weakened gate.

The runtime API minor version changes to 0.7 because appending the explicit
launch target grows `OmegaGpuLaunch`; rebuild callers against the new headers.
The ABI manifest and semver gate must agree. The existing sm_121 default, driver
pin and negative controls are not weakened for the port.

The portable converter is the pending
[omega-bridge draft #36](https://github.com/aien-dev/omega-bridge/pull/36);
the exact source commit is recorded in `om/omega-bridge.lock`. Its explicitly
selected seeded frontend uses omega e5b32567 compiler SHA-256
`2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f`.
The original 191b776 frontend gate remains unchanged and required. The expanded
route supports bounded canonical compositions and internal binary32 quant
operations; a separately checked scalar profile supplies import-free revision
and epoch without hosted trap machinery. Generated identities pin each
converter, frontend and compiler, not a moving branch name.

Experimental native x86 compiler work (#923) remains a distinct failing route,
not a build, generation or acceptance prerequisite here. Portable C is generated
by the seeded AArch64 compiler and compiled normally for the selected host.

The x86 numeric source/header divergence from Omega's retained CPU oracles is
intentional. Main authorized four explicit cutover blob pins, each with a
rationale, in `tools/extract/drift-baseline.tsv:2`; no wildcard exclusion is
used. `tools/extract/drift.sh:110` mutates each retained source/header and
requires a named drift failure, preserving future change detection.

The Ampere oracle also exposed the historical sm_121 `BW_IR_LDG_MMIO` mismatch:
its header promises MMIO.GPU but its old encoder emits EF
([issue #171](https://github.com/aien-dev/CUDApp/issues/171)).
The sm_86 expectation enforces the IR's MMIO.GPU contract, retaining every
operand and rejecting an EF mutant; it does not equate the cache policies.
The sm_121 correction is isolated in a separate PR and is not part of this
port. No disassembly result establishes cache coherence or device behavior.

SM86 `BRA` uses a byte-displacement field rather than SM121's split field.
The corpus oracle retains exact branch/join destinations and barrier registers;
wrong-target and wrong-barrier controls must fail. SM86 BSSY/BSYNC omit SM121's
printed `.RECONVERGENT` qualifier, not its reconvergence semantics.
Signed-field arithmetic is recorded with its Wolfram query and result in
`evidence/CUDAPP-AMPERE/wolfram-branch.txt`. FSUB's source negate bit is also
checked independently; FADD cannot satisfy the subtraction contract.
IADD3's discarded `PT` carry outputs are explicit on SM121 and implicit on
SM86. Only those two discarded outputs are omitted from the target-specific
expectation; an unexpected carry predicate must still fail the oracle.

## First real run: operator-provided access

Provide, for **each card**: which machine holds it, the machine's OS/distribution
and architecture, its exact NVIDIA driver version and open-module status, and
SSH host/account/access. Then select the actual PCI BDF and authorize the GPU
window on that machine. If a different driver is installed, stop at the doctor
and update/review the ABI pin; do not install or replace a driver implicitly.
Nothing in this preparation authorizes touching the Spark's live model.
