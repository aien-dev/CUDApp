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
| Portable Omega build provenance | IMPLEMENTED | `om/units.sh` covers all canonical hosted roots plus Ampere encoding/runtime and the separate revision/epoch/f16 freestanding profile. Checked portable generated translation units replace executable-code trampolines. `om/omega-bridge.lock`, per-unit identities and generated declarations retain exact composed-source SHA → C SHA provenance. Missing/mismatched converter, profile or seed refuses before installation. Final host acceptance is recorded separately below. |
| Native x86_64 host build | EXERCISED; host only | Source `592cee255eaba4b86c5a823ed071c2773a3d4f13` passed [native consumer run 40762515409](https://github.com/aien-dev/CUDApp/actions/runs/40762515409), executing PR merge `6a2b759333ce791663d0107949659c7a3cd887bb`: 58 host targets, 40 regenerated portable roots and the 2169-kernel offline oracle. The earlier `__truncsfhf2` link failure is repaired by import-free integer Omega conversion, not by weakening the gate or linking libgcc. `.github/workflows/ampere-x86.yml` separates EMULATED pinned ARM provenance from genuinely native x86 compiled-C consumers and offline oracles. This is not native x86 Omega compiler acceptance or RTX device qualification. |
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

Recovery host acceptance at `dbb1b5002fd7d96ff4695b2da4dcb3238db226c2`
used bridge `cb492a76348546dc1ffe5857492e0796cfcec232`, after preserving
the interrupted index at `3e4145d3` and integrating canonical `c6977b49`
plus the accepted capacity-refusal source/tests through `ca441c5a`.
`make -j2 OUT_DIR=build-ampere-final all test test-native-artifact
zero-cuda-gate ampere-device-build` exited zero on the AArch64 host:
`CUDA++ HOST TESTS PASS (57 targets)`. This includes byte-identical
regeneration of all 39 roots, real compose-capacity refusal, import-free
scalar linkage/allowlist controls, Ampere simulated ordering and VRAM
admission, numeric/cubin checks, and existing encoder/cubin differentials.
The five compiler/converter/frontend mutation controls each refused with
exit 3 and installed no output ([pin receipt](../evidence/CUDAPP-AMPERE/recovery-portable-pins.json)).
These are host results, not RTX execution or a native x86 compiler result.

The fresh instrumented producer run captured 773 SM121 kernels and emitted
the Ampere math corpus. Its 600-second make deadline expired during decoding;
the completed standalone comparison then exited zero: 2169 SM86 kernels,
every opcode and operand matched, 73 native-FP8 refusals. Register operand,
uniform destination, EF-versus-MMIO.GPU policy, subtraction sign, branch,
reconvergence join/barrier and carry-output mutants were refused.
See [exact oracle receipt](../evidence/CUDAPP-AMPERE/recovery-exact-oracle.json).
The legacy SM121 MMIO repair remains separate in issue171/draft172;
its unchanged 773-kernel and four-artifact impact receipts are not hardware
qualification and are not silently folded into Ampere's expected semantics.

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
operations; a separately checked scalar profile supplies import-free revision,
epoch and binary32-to-binary16 RNE conversion without hosted trap machinery.
Generated identities pin each converter, frontend and compiler, not a moving branch name.

The disposable x86 CI runner uses Ubuntu's verification-only
`qemu-user-static` package `8.2.2+ds-0ubuntu1.19`, amd64 package SHA256
`ec80ee896fe8f57f7420f4d4076164cb0fa613df6848568cdf5d1a2dbe2347cb`,
and AArch64 interpreter SHA256
`1152646a17920a81c618d62574ee8ae8b63243349957095d40212c2245f20aab`.
An ARM runner source-bootstraps baseline compiler
`191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`;
the x86 job checks that artifact and the seeded compiler before use.
Only their AArch64 ELF execution is registered, with exact magic, mask and
interpreter checks and unconditional removal of this run's handler afterward.
No Spark binfmt configuration is changed. These executions are EMULATED
provenance, not a native compiler result. Native x86 ELF consumers are built
with the runner's C compiler, checked with `readelf`, then exercised by the
unchanged acceptance targets. QEMU is neither linked into production nor an
implementation dependency; its intended Omega replacement is the separate
native x86 compiler, which still fails and is not assumed usable.
The native x86 consumer acceptance for source
`592cee255eaba4b86c5a823ed071c2773a3d4f13` completed successfully in
[run 40762515409](https://github.com/aien-dev/CUDApp/actions/runs/40762515409),
which checked out PR merge `6a2b759333ce791663d0107949659c7a3cd887bb`.
It reported 58 host targets, byte-identical regeneration of all 40 portable
roots, half-conversion boundary/sweep/mutant controls, the unchanged
freestanding import allowlist, and the 2169-kernel offline SM86 oracle.
The compiler/converter provenance executions remain EMULATED; the three
consumer ELF headers were checked as x86-64. Physical RTX execution remains
NOT_RUN. A separate CodeQL finding in the offline cubin fixture writer is
repaired by explicitly creating new output files with mode 0600, independent
of the caller's umask; this does not change the cubin bytes or production code.


`om/numeric/f16.om` implements the freestanding conversion; the C caller
only bitcasts the input and calls its checked scalar ABI. The hosted
`_Float16` cast remains the independent verification oracle, not production
freestanding code. `make test-ampere-f16-host` checks signed zero,
subnormals, ties-to-even, overflow, infinities and NaNs, rejects six
conversion mutants and compares a stepped raw-input sweep with that
native cast. The full host suite requires this target. The generated
unit compiles with no undefined symbols; the unchanged aggregate
freestanding allowlist continues to refuse unexpected runtime helpers.

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
