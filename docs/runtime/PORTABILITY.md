# Omega portability: what is tied to the chip, what is not

Slice R3 of the runtime roadmap (omega-mv#596, tracking #593). Written for a developer who has never seen this
repository and wants to know how much of Omega they would have to change to run it on another GPU, another operating
system, or no operating system. Every claim names the file it comes from. Where a source is missing the claim is marked
UNVERIFIED. Nothing here was run on a GPU; this slice is read-only analysis plus a host-side check.

Companion documents, not repeated here: `docs/numeric/OMEGA_GPU_NATIVE_BOUNDARY.md` (which driver call each part makes
today and which cut removes it) and `docs/numeric/OMEGA_GPU_ENGINE.md` (the engine contract).

## 1. Two separate questions

Porting Omega means answering two questions that are easy to mix up:

1. **Chip.** Does this file know something about the GB10 (Grace Blackwell, compute capability sm_121)? Machine-code
   words, launch-descriptor fields, class numbers, cache behaviour.
2. **Operating system and driver.** Does this file need Linux, the NVIDIA open driver, or a C library?

The table `docs/runtime/portability_table.tsv` answers both, one row per `src/omega_gpu_*`, `src/omega_blackwell_*` and
`src/omega_bw_*` file (102 rows):

| Column | Values | Meaning |
|---|---|---|
| `class` | `CHIP_SPECIFIC`, `PORTABLE` | `CHIP_SPECIFIC`: the file carries knowledge of the GB10 or of the driver interface that exposes it, so a different GPU or driver means reviewing or replacing it. `PORTABLE`: it carries neither. |
| `reason` | text | Why, with the evidence (an include list, a constant, a make target). |
| `os_binding` | `FREESTANDING_PROVEN`, `NO_HOSTED_INCLUDE`, `HOSTED_LIBC`, `LINUX_DRIVER` | `FREESTANDING_PROVEN`: a `.c` file in `GPU_FS_SRCS` (`mk/omega-gpu-engine.mk`), built by `make gpu-engine-freestanding` with no hosted headers. `NO_HOSTED_INCLUDE`: no hosted header by include scan only (a header, or a `.c` outside that build). `HOSTED_LIBC`: includes stdio, stdlib, time, errno, math or pthread. `LINUX_DRIVER`: includes `nvrm.h`, `m16_native.h` or `omega_gpu_platform_linux.h`. |

Totals at base commit a8ffd2b9: 87 files are `CHIP_SPECIFIC` and 15 are `PORTABLE`. By `os_binding`: 13 `FREESTANDING_PROVEN`,
77 `NO_HOSTED_INCLUDE`, 9 `HOSTED_LIBC`, 3 `LINUX_DRIVER`. The small `PORTABLE` count is the honest result: almost everything
in `src/omega_gpu_*` launches a Blackwell kernel. Command to recount: section 7.

`make test-runtime-portability` (host only, no compiler, no GPU) checks the table against the tree: every file has a row,
every row names a real file, a `PORTABLE` row's file includes no Linux or NVRM header and no Blackwell header, and the
`os_binding` column agrees with the includes and with the freestanding file list. It proves rows and includes agree. It
does not prove a reason is right; the reasons are human claims and each cites what to check. The include scan reads
direct `#include` lines only, and skips lines inside `#ifndef OMEGA_GPU_NO_POSIX` (the hosted-only branch) and
`#ifdef OMEGA_ATTN_DEBUG` (a debug trace).

## 2. What is tied to the chip

| Piece | Where | What is chip-specific about it |
|---|---|---|
| Machine-code encoder | `src/omega_blackwell_encoder.c`, `.h` | Packs instructions into 16-byte words (`OMEGA_BW_INSTRUCTION_BYTES`). A different instruction set needs a new encoder. |
| Kernel generators | `src/omega_blackwell_codegen*.c`, `src/omega_bw_*.h` | Every kernel (vector add, matmul, FP8 GEMV, attention, router, tensor-core GEMM, ternary GEMV) is generated as sm_121 instructions with scheduling rules proven on the chip (`src/omega_bw_em.h`). Some ops are patched in after encoding because the shared encoder lacks them (`src/omega_blackwell_codegen_qmma.h`, values from an nvcc sm_121 oracle). |
| Launch descriptor (QMD) | `src/omega_blackwell_qmd.h`, `.c` | A 96-word, 384-byte structure the GPU reads to start a kernel (`OMEGA_BW_QMD_BYTES`). Each field has an evidence tier: documented in NVIDIA's `clcdc0qmd.h`, observed on silicon, or reverse engineered. |
| Class numbers and setup words | `src/omega_gpu_session.c` (`SETUP_WORDS`), `src/omega_gpu_cmdlist.c`, `src/omega_blackwell_engine.c`, `src/omega_blackwell_submit.c` and seven more files | The pushbuffer words that select compute class `0xcec0` (Blackwell compute B) and set up the engine. The same 18-word block is copied into 11 files under `src/` (`grep -l 0x0000cec0 src/*.c src/runtime/*.c`). Method numbers such as `WAIT_FOR_IDLE 0x0110` and `SEND_PCAS_A 0x02b4` are in `src/omega_gpu_cmdlist.c`. The channel and user-mode classes (`clc96f.h`, `clc661.h`) are used by `physics/nvrm/nvrm.c`, outside this repository. |
| Cache rules | `src/omega_gpu_platform.c` (check at the cache-policy refusal), `src/omega_gpu_mem_cached.h`, `src/omega_gpu_cmdlist.h`, `src/omega_gpu_prefill.h` | Anything the CPU polls for a GPU write must be GPU-uncached, because GPU-cacheable system memory "will not be coherent with direct CPU mappings" (nvidia-open 580.173.02 `nvos.h`, quoted in `src/omega_gpu_platform.h`). Read-only weights may be GPU-cached only if the CPU wrote them once and never reads them. A command list ends with an L2 flush and a second marker on an uncached page. A chip with coherent memory could relax these rules. |
| Code buffer tail | `src/omega_gpu_code_alloc.h` | Code buffers stay mapped for 2 KB after the last instruction (`OMEGA_GPU_CODE_PREFETCH_TAIL`) because the instruction cache prefetches past it. The rule comes from Mesa NVK; that it was the cause of the GB10 fault in omega#323 is INFERRED until the chip confirms, per the header. |
| Target facts | `src/omega_bw_target.h` on branch `rig/gb20x` | Compute class, QMD version, shared-memory configuration, SM count and cache sizes in one table row per chip. Only the GB10 row is chip-proven. |

**A finding that corrects a simple reading of the architecture.** The session (`src/omega_gpu_session.c`), the command
list, and the matmul and attention APIs are free of the operating system: the first, the matmul and the attention API are
in the freestanding build. They are not free of the chip. They embed `SETUP_WORDS`, method numbers, QMD use and sm_121
stall counts. The platform contract hides the *driver*; it does not hide the *GPU*. The table marks those files
`CHIP_SPECIFIC` with `os_binding` `FREESTANDING_PROVEN` for that reason.

## 3. What is portable

| Piece | Where | Why it is portable |
|---|---|---|
| Platform contract, ABI 1.2 | `src/omega_gpu_platform.h`, `src/omega_gpu_platform.c` | Typed handles (index plus generation), separate CPU, GPU and bus address types, buffer kinds and rights, a service set negotiated up front, named refusals (`OmegaGpuPlatformRefusal`), no silent fallback. ABI 1.2 adds `submit_prefix_words`. Included in `make gpu-engine-freestanding`. |
| Engine state machine | `src/omega_gpu_engine.c`, `.h` | Open, validate, submit, verify, close, with a named failing step. Talks to a backend through a function table (`OmegaGpuBackend`). No chip knowledge. |
| Bounded wait | `src/omega_gpu_wait.c` | Clock and hooks come from the caller; the freestanding build has no default clock. |
| Serving reservation | `src/omega_gpu_serving.c` | Counts and bounds a session's allocations. |
| Text and host math | `src/omega_gpu_fmt.c`, `src/omega_gpu_hostmath.c` | A bounded formatter and libm-free log, exp, pow. |
| Public tensor entry points | `src/omega_gpu_matmul_api.h`, `src/omega_gpu_moe_api.h`, `src/omega_gpu_session.h` | Plain types in the signatures. The `.c` files behind the matmul and attention calls are chip-specific (section 2). |

What the freestanding build proves, and what it does not, is in `docs/numeric/OMEGA_GPU_NATIVE_BOUNDARY.md` section 1 and
section 4: the listed files link with only `memcpy`, `memmove`, `memset`, `memcmp`, `strlen` and `strcmp` undefined, and
nothing there opens a device. The table adds the other direction: which files are outside that build and why.

## 4. Three ways to reach the GPU

| | Linux open driver | WSL2 under Windows (`/dev/dxg`) | Native AIENOS |
|---|---|---|---|
| Status | Runs on the DGX Spark (GB10). The only backend with chip evidence. | Host-tested only. Nothing has touched a real `/dev/dxg` or an RTX 5070 (`docs/rig/dxg-backend.md` on `rig/gb20x`, section 5, status line). | Not built. The boot image that looks for the GPU has not been booted on hardware (aienos `docs/GB10_NATIVE_STATUS.md`). NOT_RUN. |
| How it talks | `/dev/nvidiactl`, `/dev/nvidia0`, `/dev/nvidia-uvm` through `physics/nvrm` and `physics/m16`. Omega rings the GPU itself through a channel it owns. | D3DKMT ioctls on `/dev/dxg` (Microsoft's `dxgkrnl`), which forwards to the Windows NVIDIA driver. No doorbell ioctl exists; each submit is an ioctl (`docs/WINDOWS-GPU-PATH.md` section 1, aien-rig branch `app/v0`). | AIENOS drives PCI, SMMU, firmware load, queues and doorbells itself. |
| Platform contract | `src/omega_gpu_platform_linux.c` implements it (reference only, hosted only). The vector executor `src/omega_blackwell_submit.c` and engine backend `src/omega_blackwell_engine.c` still call `m16_native_*` directly. | **Does not use it.** The branch defines its own interface, `OmegaSubmitOps` in `src/omega_gpu_submit_iface.h`, with an nvrm and a dxg implementation. No `OmegaGpuBackend` sits on it yet (`dxg-backend.md` section 5). | Designed around it: the contract's native owners per driver responsibility are listed in the boundary doc section 2. No native implementation exists (boundary doc section 6, "C, remaining"). |
| Biggest open question | none for the GB10 | NVIDIA's private driver data blobs and whether the Windows driver accepts Omega's own Blackwell method stream at all (`dxg-backend.md` section 4: "the largest open question"). `docs/WINDOWS-GPU-PATH.md` section 3 found no public project doing raw NVIDIA submission over `/dev/dxg`. | Everything the Linux driver does today: GSP firmware, memory, page tables, queues, interrupts, reset (boundary doc section 2, all rows NOT_RUN). |
| Chip code reused | all | The same encoder and QMD; the target table lists the RTX 5070 as sm_120, "THEORETICAL: built from public sources only, never run on the chip" (`src/omega_bw_target.h` on `rig/gb20x`). | The encoder and QMD builder are plain C with no Linux dependency (aienos `docs/GB10_NATIVE_DEPENDENCY_MAP.md`, plain-English summary, item 1); the freestanding build checks that for the files in `GPU_FS_SRCS`. |

Two caveats a reader should keep:

- The aienos dependency map is pinned to an older Omega commit (`2ae8677c`, its "Pins" table). Where it disagrees with the
  boundary doc, which tracks `omega_gpu_session.c` moving onto the platform contract, the boundary doc is newer.
- The `rig/gb20x` interface and the platform contract describe the same job (open a device, allocate, queue, submit,
  wait) with different shapes. The branch's own text does not mention the platform contract
  (`grep -i platform docs/rig/dxg-backend.md` finds nothing). Whether the two should be merged is a design question for
  the lane that owns that branch (ledger L024); this note does not decide it and did not edit the branch.

## 5. What a new GPU needs

For a GPU in the same family (another Blackwell part such as the RTX 5070, sm_120), the work is mostly *measuring*. For a
different family it is *rewriting*. In order:

1. **Identify the chip.** Add a target row: PCI ids, architecture and implementation ids, SM version as the driver
   reports it, compute and channel and user-mode class numbers, QMD version and shared-memory fields, SM count, L2 size,
   memory model (unified like GB10 or discrete VRAM), host barrier. The shape of this row is `OmegaBwTarget` in
   `src/omega_bw_target.h` on `rig/gb20x`; each field there carries an evidence grade, and fields nobody has published are
   `UNKNOWN` until measured on the card.
2. **Machine code.** Same family: check whether `src/omega_blackwell_encoder.c` output is valid on the new chip. For
   sm_120 versus sm_121 the branch records "no sm_120 vs sm_121 encoding split is public" (`omega_bw_target.c`, field
   `isa`, evidence INFERRED). Different family: a new encoder, new generators for every `omega_blackwell_codegen_*.c` kernel
   you need, and new emitter helpers (the whole `CHIP_SPECIFIC` block of the table).
3. **Launch descriptor.** Confirm or replace the QMD builder (`src/omega_blackwell_qmd.c`) and the constant-bank layout.
4. **Class numbers and setup words.** Replace the `0xcec0` setup block. It lives in 11 files; today a change means 11
   edits (`grep -l 0x0000cec0 src/*.c src/runtime/*.c`). Folding the copies into one is a cleanup this slice records but
   does not make.
5. **Cache and coherence rules.** Decide for the new memory model which buffers may be GPU-cached, which must be
   uncached, and what flush ends a command list. The contract refuses a cached host-polled buffer
   (`OMEGA_GPU_PLAT_REFUSED_CACHE_POLICY`); a chip with coherent memory would change that rule in one place
   (`src/omega_gpu_platform.c`) and in the cache notes of `mem_cached`, `cmdlist`, `prefill`.
6. **Code buffer tail.** Check whether the new chip also prefetches past the last instruction (`src/omega_gpu_code_alloc.h`).
7. **A platform implementation.** Provide the services in `OMEGA_GPU_SVC_REQUIRED_V1` (time, wait, lock, MMIO, DMA, sync,
   completion) and `OMEGA_GPU_SVC_REQUIRED_SESSION` (device, queue, clock, host memory), at ABI 1.2, and pass
   `omega_gpu_platform_check_for`. `tests/test_omega_gpu_platform.c` reaches every refusal; use it as the starting test.
   The Linux reference (`src/omega_gpu_platform_linux.c`) is the worked example.
8. **A backend.** Either the vector backend (`OmegaGpuBackend`, example `src/omega_blackwell_engine.c`) or the session path.
9. **Qualification on the chip.** The M17 and M18 gates (`src/omega_blackwell_gates.c`) and the poison-word vector add are
   the first correctness check; nothing is claimed for a chip until they pass there.
10. **Update the table.** `make test-runtime-portability` refuses a new `omega_gpu_*`, `omega_blackwell_*` or `omega_bw_*`
    file that has no row.

## 6. Limits of this note

- It reads source. No GPU, WSL2 machine or native boot was used; every "works" in the table of section 4 is a claim of the
  source it cites, and the WSL2 and native columns have no chip evidence at all.
- `class` is a judgement per file. The mechanical parts (includes, the freestanding list) are checked; whether a reason is
  right is not.
- The dxg and native documents live on other branches and repositories and can change. They were read from
  `origin/rig/gb20x` (`12e17891`), aien-rig `origin/app/v0`, and aienos `origin/main` on 2026-10-19.

## 7. Reproduce

```
make test-runtime-portability        # the check plus the negative controls; host only
awk -F'\t' '!/^#/ && $1!="file" {n[$2]++} END {for (c in n) print c, n[c]}' docs/runtime/portability_table.tsv
```

Evidence of the controls and of the freestanding build is in `evidence/RUNTIME-R3-PORT/`.
