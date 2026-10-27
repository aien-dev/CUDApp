# Omega Runtime API v0

Omega Runtime API version: 0.6.0

Status: first written version, 2026-10-19 (slice R1, omega-mv#594, tracking #593). Source of truth is the C headers in
`src/`. This file describes them; `docs/runtime/omega_runtime_api.manifest` lists them; `make test-runtime-api` checks the
list against the headers (a host check, no GPU). Review state: reviewed by Opus 5.5 against the headers and the code on 2026-10-22 (L386); its findings are fixed
in this version. The check proves the list and every quoted C signature match the headers; prose is checked by review.

The version number lives in one header, `src/omega_runtime_api.h` (`OMEGA_RUNTIME_API_VERSION_MAJOR`, `_MINOR`, `_PATCH`).

## 1. What this is

Omega is a GPU compute runtime that does not use CUDA. This file is the written contract for the part a program calls
directly: find and open the GPU, get memory, copy data in and out, run a kernel, run many kernels in one submission, run a
packaged job, and read errors. It covers what exists on the main branch today, on one machine type (GB10, the DGX Spark),
through the Linux open-driver backend. Nothing here claims more than that.

Compatibility preflight is **wrapper-only**: `make doctor` checks the kernel-module
version and PCI target, and `make hello-gpu` runs it first. Direct binaries and
library callers have no driver-version or PCI-ID admission guard at device open.
Callers must not bypass the preflight or the operator's GPU authorization procedure.
The M16 receipt imported here is a historical summary whose referenced raw logs
are absent; it is not current-commit qualification (see README evidence limitation).

How to read an entry:

| Tier | Meaning |
| --- | --- |
| **STABLE** | Used by other code in the project, covered by chip results, and kept compatible under the rule in section 2. Has its own `####` section below. |
| **EXPERIMENTAL** | Works and is listed, but may change in any minor version, because the design is still moving or only one use of it exists. Listed in tables with the reason. |
| **INTERNAL** | Exists in a header for the project's own libraries or tests. Not part of the API; not in the manifest; may change or disappear any time. Every such name in a manifest header is listed, with a reason, in `docs/runtime/omega_runtime_api.internal` (section 10), so nobody builds on it by accident. |

Every C signature in this file is copied from the header it names. `tools/runtime_api_check.sh` re-checks every quoted
signature against the headers on every run.

## 2. Version and compatibility rule

The version is MAJOR.MINOR.PATCH (semantic versioning), currently **0.6.0**. It is a property of the list of entries and
their behaviour, not of the library build. Changes so far:
- 0.6.0 (AttnGeneral): adds EXPERIMENTAL `omega_gpu_attention_general`, `OmegaGpuAttnGeneralDesc`,
  `OMEGA_GPU_ATTN_GENERAL_MAX_HEAD_DIM` (256), `OMEGA_GPU_ATTN_GENERAL_MAX_Q_HEADS` (1024) and `OMEGA_GPU_ATTN_WINDOW_NONE`:
  one attention kernel for every head_dim 1..256, any query-head count with any kv-head count dividing it (MHA, GQA,
  MQA), causal, non-causal and sliding-window masks, strided f32 or bf16 KV, any kv_len. The existing attention entries
  keep their tuned kernels for the shapes they took and now run the general kernel for the shapes they refused
  (head_dim other than 64 / 128, non-power-of-two GQA, more than 64 query heads, contexts past 4096); the paged
  entries do the same without the context split. Five test hooks and the general mutant ids go on the INTERNAL list.
- 0.5.0: adds EXPERIMENTAL `omega_gpu_cmdlist_submit_pair`, `omega_gpu_cmdlist_wait_for`,
  `OmegaGpuCmdListOpts.first_barrier_none`, `omega_gpu_session_extra_queue` (streams, C4) and `omega_gpu_session_recover`,
  `omega_gpu_session_recoveries` (recovery after a latch, C5), carried from omega-mv e5b32567 by CUDApp#121; two
  module-to-session hooks go on the INTERNAL list (section 10). Also adds EXPERIMENTAL `omega_gpu_fp8_grouped_ctas_rule`
  and `omega_gpu_fp8_grouped_ctas_auto` (section 9.7) and INTERNAL `OMEGA_GPU_SM_COUNT`.
- 0.4.0: `OMEGA_GPU_HANDLE_SLOTS` 1024 -> 4096 and `OMEGA_GPU_CMDLIST_MAX_LAUNCHES` 2048 -> 4096 (STABLE constants), and
  so `OmegaGpuHandleTable` (EXPERIMENTAL) grows to 69648 bytes: the values the live server (omega-mv 6de999da, carried by
  CUDApp#120) was built with. A MAJOR-class change, taken as MINOR under the major-version-zero rule below. Also adds the
  carried decode-path calls of section 9.7 (FP8 knobs and quantiser, record-into-command-list forms, router row batching)
  as EXPERIMENTAL, and six more names to the INTERNAL list.
- 0.3.0: adds the EXPERIMENTAL decode-path entries of section 9.7 and their result codes; no STABLE entry changed.
- 0.2.0: adds `OMEGA_RUNTIME_API_VERSION_PATCH`, lists `OMEGA_RUNTIME_API_VERSION` in the manifest and the INTERNAL list
  (section 10); no entry changed.

**A PATCH bump (0.5.0 to 0.5.1) is for:**
- a fix that changes no manifest entry: a STABLE function that did not do what this file says now does (the fix makes it
  match), or a correction to this file's prose.

**A MINOR bump (0.5 to 0.6) is allowed to:**
- add a function, a constant, an enum value at the end of an existing enum, or a new struct;
- add an EXPERIMENTAL entry, change or remove an EXPERIMENTAL entry, or move an entry from EXPERIMENTAL to STABLE;
- add a new failure name for the same call to report (callers must treat an unknown failure code or stage text as a failure).

**From 1.0 on it needs a MAJOR bump (1.x to 2.0) to:**
- remove or rename a STABLE function, struct, field or constant;
- change a STABLE function's argument or return types, a STABLE struct's size, field order or field offsets, or the value
  of a STABLE constant or enum value;
- change documented behaviour of a STABLE entry: what the lock covers, what a latch does, which buffers get parked, what
  a return value means;
- move a STABLE entry down to EXPERIMENTAL or INTERNAL;
- raise the platform contract's major number (section 9.2).

**Major version zero** (semantic versioning 2.0.0, item 4: "anything MAY change at any time"): while MAJOR is 0, a change
from the MAJOR list above needs a MINOR bump (0.5 to 0.6) plus its line in the change list above and in the manifest;
MAJOR is reserved for the first break after 1.0. It still never goes unbumped: a STABLE change with no bump fails the
check. STABLE in 0.x means "written down and checked", not "frozen". The first 1.0 will be cut only when section 8 (what
v0 does not have) has been looked at again.

How it is enforced (all host checks, no GPU; CI job `runtime-api`, `.github/workflows/runtime-api.yml`):
- `make test-runtime-api` (`tools/runtime_api_check.sh`): a header change that moves any listed entry fails until the
  manifest line is edited on purpose (`tools/runtime_api_check.sh --bless` refreshes the numbers; the diff is reviewed). A
  function, macro or enumerator added to a manifest header fails until it is listed in the manifest or in the INTERNAL list
  with a reason. The version macros, the manifest's `abi` line and consts, and the `Omega Runtime API version:` line in this
  file must agree.
- `make test-runtime-api-semver` (`tools/runtime_api_semver.sh`, base `origin/main` by default; CI uses the pull request's
  base): compares the manifest with the base's entry by entry and fails when the change needs a bigger bump than it has,
  or when the version goes down. Constants count as STABLE unless their manifest line carries `# EXPERIMENTAL`. While the
  base's MAJOR is 0, a MAJOR-class change passes with a MINOR bump (the rule above); once the base is 1.x it needs MAJOR.
- What stays a review duty: a change of documented behaviour with no entry change (the script cannot see prose), and the
  limits listed in section 11.

The platform contract has its own number (`OMEGA_GPU_PLATFORM_ABI_MAJOR` 1, `OMEGA_GPU_PLATFORM_ABI_MINOR` 2, set in
`src/omega_gpu_platform.h`). A platform minor bump needs at least a runtime MINOR bump; a platform major bump needs a runtime MAJOR bump.

## 3. How the pieces fit (plain English)

**Device and platform.** Omega talks to the GPU through a small "platform" layer: a table of services (a clock, a lock,
device-visible memory, a submission queue, a way to poll for completion). Today one platform exists, the Linux reference
(`src/omega_gpu_platform_linux.c`), which uses NVIDIA's open kernel driver. The contract is written so a second backend
(WSL2 on Windows, or AIENOS with no Linux driver) can fill the same table. A program normally never touches the platform:
the first call that needs the GPU installs the Linux reference by itself. There is **one device, one channel per process**.
There is no call to ask the device its name, SM count or memory size yet (section 8).

**Memory.** You ask the session for a buffer and get an `OmegaGpuMem`: a handle, a CPU pointer, a GPU address and a size.
Every ordinary buffer is GPU-uncached, because the CPU writes inputs the chip reads and polls markers the chip writes.
You make CPU writes visible to the GPU with `omega_gpu_session_sync_to_gpu` and GPU writes visible to the CPU with
`omega_gpu_session_sync_to_cpu`. Code (the machine code of a kernel) lives in a separate kind of buffer
(`omega_gpu_session_alloc_code`): the CPU writes it once, the GPU only executes it. Handles carry a generation number, so a
handle from a closed device or an already freed buffer is refused instead of silently reused. A buffer you mean to reuse and
grow, such as staging space, can be an `OmegaGpuScratch`. For read-only weights there is a GPU-cached allocation
(`omega_gpu_session_alloc_cached`, EXPERIMENTAL). There is no async copy and no unified memory in v0.

**Launch.** One kernel launch is an `OmegaGpuLaunch`: where the code is, how many registers, the block and grid size, the
argument words, shared memory, and a timeout. `omega_gpu_session_launch` submits it and waits for the chip to finish (it
is synchronous). The wait is by polling a completion word, never by an interrupt.

**Command lists.** `omega_gpu_cmdlist_*` records many launches and submits them as one unit with one completion wait. By
default each launch is preceded by a barrier so dependent launches cannot overlap; a launch can be marked independent.
EXPERIMENTAL: streams, events and dependency-ordered execution are being built on top of it (omega-mv#580), so its options
and statistics will grow.

**Module load.** There is no module or fat-binary loader in v0, and no PTX or JIT: the program brings the machine code.
Loading a kernel is `omega_gpu_session_alloc_code` (section 5) plus a CPU copy of the code into `code.cpu`; the launch then
names `code.va`. The project's own kernels come from its encoder (`src/omega_blackwell_*`, INTERNAL), and
`examples/hello_gpu` shows the whole path. A loader that reads a packaged module is section 8 work.

**Streams and events.** EXPERIMENTAL since 0.5.0, and only on command lists: `omega_gpu_session_extra_queue` opens a
second queue (behind `AIEN_OMEGA_GPU_CHANNELS=2`), `omega_gpu_cmdlist_submit_pair` submits two lists on two queues before
waiting for either, `omega_gpu_cmdlist_wait_for` makes one list wait for a launch of the other, and
`OmegaGpuCmdListOpts.first_barrier_none` drops the leading barrier. There is no stream or event object, and
`omega_gpu_session_launch` still returns only when the chip has finished. Chip qualification is CUDApp#113; the `.om`
successor is tracked in CUDApp#115.

**Job engine.** `omega_gpu_execute` runs one already-built job (today only a 1-D vector job: two inputs, one output) through
named states and returns success or exactly one named failure. EXPERIMENTAL.

**Errors, the latch and parked buffers.**
- A failing call returns a non-zero value and names its stage in text: `omega_gpu_session_last_error()`. A device-open failure
  carries the driver's own text. Do not read a bare return code as the cause.
- If the chip's completion cannot be confirmed (a completion wait ran out of time, the completion poll was refused, or a launch was cancelled after submit),
  the work may still be running and still be reading or writing your buffers. The process then **latches**:
  `omega_gpu_session_is_blocked()` returns 1 for the rest of the process, every later launch is refused, close and reopen are
  refused, and nothing is returned to the platform.
- After the latch, every buffer freed through `omega_gpu_session_free` is **parked**: kept allocated, counted by
  `omega_gpu_session_parked`, never released. Handles stay valid for inspection (`omega_gpu_session_is_parked`).
- Today there is no way back from a latch except ending the process. Recovery is open work (omega-mv#518).

**Threads.** One process-wide lock (the platform LOCK service, lock id `OMEGA_GPU_SESSION_LOCK_ID`, which is 0). A call
sequence that touches the device (open, alloc, launch, readback) is done between `omega_gpu_session_lock()` and
`omega_gpu_session_unlock()`. Threads therefore take turns on the one channel. The Linux reference lock is a plain
(non-recursive) pthread mutex (`locks[id]`, `src/omega_gpu_platform_linux.c:29` and `:119`), so taking it twice from one thread
deadlocks. These take the lock themselves and must be called without it: `omega_gpu_session_close`, `omega_gpu_cmdlist_create`,
`omega_gpu_cmdlist_free`, and every library call of section 9.5 (matmul, elementwise, attention, serving).
`omega_gpu_session_cancel` may be called from any thread without the lock (header). `omega_gpu_session_is_blocked` reads the latch with an atomic load (`src/omega_gpu_session.c:157`), so it is safe without the lock too.
`omega_gpu_session_last_error` returns a pointer into one process-wide text buffer: read it under the lock, before the next call.

### 3.1 A minimal sequence (sketch)

```c example
omega_gpu_session_lock();
OmegaGpuSession *s = omega_gpu_session_open();         /* NULL: see omega_gpu_session_last_error() */
OmegaGpuMem in, code;
omega_gpu_session_alloc(bytes, &in);                   /* GPU-uncached data buffer */
omega_gpu_session_alloc_code(code_size, &code);        /* write the machine code to code.cpu */
omega_gpu_session_sync_to_gpu(&in);
OmegaGpuLaunch L = { .code_va = code.va, .timeout_ms = 1000, /* gpr_count, threads_*, grid_*, args, n_args ... */ };
uint64_t elapsed_ns = 0; uint32_t marker = 0;     /* launch adds to elapsed_ns: start it at 0 */
int rc = omega_gpu_session_launch(&L, &elapsed_ns, &marker);
omega_gpu_session_sync_to_cpu(&in);
omega_gpu_session_free(&in);
omega_gpu_session_unlock();
```

This is a shape sketch, not a worked example (the worked example is slice R2).

## 4. Session: device, lock and state

#### `omega_gpu_session_lock`
STABLE: used by every library API and chip-proven; the lock model is the documented one.
```c
void omega_gpu_session_lock(void);
```
Takes the process-wide session lock (platform LOCK service, id `OMEGA_GPU_SESSION_LOCK_ID`). Does not touch the device.
With no platform installed it does nothing, and the open that follows refuses.

#### `omega_gpu_session_unlock`
STABLE: pairs with the lock.
```c
void omega_gpu_session_unlock(void);
```

#### `omega_gpu_session_open`
STABLE: used by every library API and chip-proven. Treat the returned pointer as "the device is open"; the fields of
`OmegaGpuSession` are EXPERIMENTAL (the session's own launch buffers) and a caller needs none of them.
```c
OmegaGpuSession *omega_gpu_session_open(void);
```
Caller holds the lock. Opens the device on first use. Returns NULL when the process is latched, no platform is installed or the
open failed; the stage is in `omega_gpu_session_last_error()`.

#### `omega_gpu_session_is_open`
STABLE: simple state query.
```c
bool omega_gpu_session_is_open(void);
```

#### `omega_gpu_session_is_blocked`
STABLE: the latch query that every caller needs.
```c
int omega_gpu_session_is_blocked(void);
```
Returns 1 once an uncertain completion has latched this process, else 0. Any thread.

#### `omega_gpu_session_close`
STABLE: the only way to close the device; its refusal after a latch is part of the error model.
```c
void omega_gpu_session_close(void);
```
Closes the device and frees every session buffer. **Takes the lock itself.** Refused (a no-op) after an uncertain completion.
The next open reopens; every handle of the closed device is stale from then on.

#### `omega_gpu_session_open_count`
STABLE: diagnostic with a fixed meaning.
```c
uint32_t omega_gpu_session_open_count(void);
```
Device opens so far in this process (a persistent session shows 1).

#### `omega_gpu_session_set_spin_us`
STABLE: used by sovereign-core's binding; fixed meaning and limit.
```c
int omega_gpu_session_set_spin_us(uint32_t us);
```
Process-wide marker-poll spin window, used by every launch whose own `spin_us` is 0. 0 (the default) keeps the platform's
sleep-poll. Returns 0 on success, -1 above `OMEGA_GPU_SESSION_MAX_SPIN_US` (setting unchanged).

#### `omega_gpu_session_spin_us`
STABLE: reads the setting above.
```c
uint32_t omega_gpu_session_spin_us(void);
```

## 5. Session: memory

All allocation calls need the caller to hold the lock with the device open.

#### `OmegaGpuMem`
STABLE: the buffer type every API passes around.
A device buffer: a typed handle, its kind, the CPU view, the GPU virtual address and the size. A `cpu` of NULL means there is no CPU view.

| Field | Meaning (from `src/omega_gpu_session.h`) |
| --- | --- |
| `handle` | `OmegaGpuBufferHandle`: index and generation |
| `kind` | `OMEGA_GPU_HANDLE_BUF_DATA`, `_BUF_CODE` or `_BUF_QUEUE` |
| `cpu` | CPU view |
| `va` | GPU virtual address |
| `size` | bytes of both views |

#### `OmegaGpuBufferHandle`
STABLE: a field of `OmegaGpuMem`, so its layout is part of it.
`{ uint32_t index; uint32_t generation; }`. Generation 0 is never issued, so a zeroed handle means "none".
A handle is valid only while its slot generation matches; a closed handle is refused.

#### `OmegaGpuDeviceHandle`
STABLE: same shape and rule as the buffer handle; named in the platform contract.
`{ uint32_t index; uint32_t generation; }`.

#### `OmegaGpuQueueHandle`
STABLE: same shape and rule as the buffer handle; named in the platform contract.
`{ uint32_t index; uint32_t generation; }`.

#### `omega_gpu_session_alloc`
STABLE: the ordinary buffer call, used by every library.
```c
int omega_gpu_session_alloc(size_t bytes, OmegaGpuMem *out);
```
A GPU-uncached DATA buffer, CPU and GPU read/write, rounded up to a page. Returns 0 on success; non-zero with the stage named on
failure (the driver can refuse, for example with no memory; the refusal is counted in `OmegaGpuAllocStats.alloc_failures`).

#### `omega_gpu_session_alloc_code`
STABLE: the only way to place kernel code; chip-proven.
```c
int omega_gpu_session_alloc_code(size_t code_size, OmegaGpuMem *out);
```
A CODE buffer (CPU writes, GPU executes, GPU never writes) for `code_size` bytes plus the 2 KB instruction-prefetch tail
(`src/omega_gpu_code_alloc.h`), the tail zero-filled. The caller writes the code to `out->cpu`. A size of 0 still gets one page (`omega_gpu_code_alloc_bytes` rounds up to at least one page).

#### `omega_gpu_session_free`
STABLE: the release call; its latch behaviour is part of the error model.
```c
void omega_gpu_session_free(OmegaGpuMem *m);
```
Releases `m` to the platform and zeroes `*m`. **Latched: `m` is parked instead (never released).** A stale handle (from a
closed device) or a closed session is refused: nothing is released, the refusal is counted
(`OmegaGpuAllocStats.stale_refusals`) and `*m` is zeroed.

#### `omega_gpu_session_sync_to_gpu`
STABLE: the visibility rule for CPU writes.
```c
int omega_gpu_session_sync_to_gpu(const OmegaGpuMem *m);
```
Makes CPU writes to the whole buffer visible to the device (platform SYNC). Returns 0, also for a buffer with no CPU view;
-1 with the stage named when refused or when there is no device.

#### `omega_gpu_session_sync_to_cpu`
STABLE: the visibility rule for GPU writes.
```c
int omega_gpu_session_sync_to_cpu(const OmegaGpuMem *m);
```
Makes device writes to the whole buffer visible to the CPU. Same return values as `sync_to_gpu`.

#### `OmegaGpuScratch`
STABLE: the staging-buffer type the library APIs use.
A caller-owned staging buffer that grows to its high-water mark: `mem` (an `OmegaGpuMem`), `cap` (usable bytes), `high_water`
(largest request so far) and `reserved` (a serving reservation is active, see `omega_gpu_session_scratch_reserve`).
Zero-initialise it before first use. This is required: `omega_gpu_session_scratch` reads `reserved`, `cap`, `mem.cpu` and
`high_water` before it writes them (`src/omega_gpu_session.c:290-296`).

#### `omega_gpu_session_scratch`
STABLE: used by every library API.
```c
int omega_gpu_session_scratch(OmegaGpuScratch *s, size_t bytes);
```
Makes `s->mem` at least `bytes` long (reallocated when it grows; **contents are not kept**). Returns 0 or -1; with a
reservation active, a request past the reserved size returns `OMEGA_GPU_SESSION_SCRATCH_RESERVED` (-2) and allocates nothing.

#### `omega_gpu_session_scratch_free`
STABLE: the matching release.
```c
void omega_gpu_session_scratch_free(OmegaGpuScratch *s);
```

#### `OmegaGpuAllocStats`
STABLE: used by sovereign-core's binding to show that serving asks the driver for nothing.
Counters of driver allocations made through `omega_gpu_session_alloc` since the process started: `allocs`, `alloc_bytes`
(page-rounded), `alloc_failures`, `frees`, `stale_refusals`. Covers the session layer only.

#### `omega_gpu_session_alloc_stats`
STABLE: reads the counters above.
```c
void omega_gpu_session_alloc_stats(OmegaGpuAllocStats *out);
```

## 6. Session: launch

#### `OmegaGpuLaunch`
STABLE: the launch descriptor; chip-proven by every library.
One launch: a kernel already resident at `code_va`, argument words for constant bank 0 at 0x380 (buffer addresses, then
parameters), and the grid.

Argument layout and the checks `omega_gpu_session_launch` makes (`src/omega_gpu_session.c:433-446`):
- a 64-bit GPU address takes two words, the low word first, then the high word;
- `n_args` is 1 to `OMEGA_GPU_SESSION_MAX_ARGS`, `args` is not NULL, and `threads_x`, `grid_x` and `grid_y` are not 0 (a
  `threads_y` of 0 means 1); otherwise the launch is refused with "launch arguments" and nothing runs;
- `gpr_count` is rounded up to a multiple of 8, and to at least 64;
- `elapsed_ns` is added to, not set: start it at 0.
What the kernel itself may read in constant bank 0 besides these words, and where GB10 machine code comes from, is slice
R2 (`examples/hello_gpu`, README "Run your first kernel").

| Field | Meaning (from `src/omega_gpu_session.h`) |
| --- | --- |
| `code_va` | GPU address of the kernel code |
| `gpr_count` | registers the QMD (the launch descriptor the hardware reads) declares; multiple of 8, at least 64 in practice |
| `threads_x`, `threads_y` | block size |
| `grid_x`, `grid_y` | grid size |
| `num_elements` | QMD config num_elements (kept per API for identical words) |
| `args`, `n_args` | `n_args` words written at constant bank + 0x380; `n_args` at most `OMEGA_GPU_SESSION_MAX_ARGS` (32) |
| `timeout_ms` | completion poll limit, applied to each of the launch's two markers in turn (so a launch can wait up to twice this); past it the completion is uncertain and **the process latches**. 0 is refused with "launch arguments" before anything is written or submitted |
| `shared_bytes` | shared memory per block; 0 = the default (1024) |
| `spin_us` | spin-poll window before the platform's sleep-poll; 0 = the process-wide setting |

#### `omega_gpu_session_launch`
STABLE: the single-kernel launch; chip-proven.
```c
int omega_gpu_session_launch(const OmegaGpuLaunch *L, uint64_t *elapsed_ns, uint32_t *marker_out);
```
Caller holds the lock, device open, not latched. Submits one launch and polls for both completion markers, so it returns when
the chip is done. 0 on success; -1 with the stage named on failure; `OMEGA_GPU_SESSION_CANCELLED` (-3) when a cancel request
was seen. A marker timeout latches the process. `elapsed_ns` (submit to second marker) is added to when the pointer is given;
`marker_out` receives the first marker word observed.

## 7. Session: errors, the latch, parked buffers

#### `omega_gpu_session_last_error`
STABLE: the error-model entry point.
```c
const char *omega_gpu_session_last_error(void);
```
The stage of the most recent failure, `""` if none. Shared by every API. Points into one process-wide buffer: read it under
the lock, before the next call.

#### `OmegaGpuParkStats`
STABLE: the parked-buffer report.
`buffers` (parked, listed plus untracked), `untracked` (parked past the listed `OMEGA_GPU_SESSION_PARK_MAX` = 512) and
`bytes`.

#### `omega_gpu_session_parked`
STABLE: how a caller sees what the latch kept.
```c
void omega_gpu_session_parked(OmegaGpuParkStats *out);
```
Buffers parked after an uncertain completion: kept allocated, never returned to the platform. The session's own launch
buffers (ring, constant bank, marker, QMD page) are parked at the latch and every later free parks too.

#### `omega_gpu_session_is_parked`
STABLE: per-handle form of the above.
```c
int omega_gpu_session_is_parked(OmegaGpuBufferHandle h);
```
1 if this handle is parked and listed.

## 8. Not in v0 (so nobody assumes it)

Each line is a gap named in `docs/runtime/ROADMAP.md` section 1 (on branch `runtime/roadmap`, not yet merged); none of it is in the manifest.
- A device query (name, SM count, memory): there is no API call for it.
- More than one device in a process (a second channel in the same group is `omega_gpu_session_extra_queue`, EXPERIMENTAL); sharing the GPU between processes (a second channel failed under load: omega-mv#523).
- Recovery after a latch beyond `omega_gpu_session_recover` (EXPERIMENTAL since 0.5.0: throw the device away and open a
  new one in the same process; a driver teardown fault still ends GPU use for the process).
- Async copy, stream and event objects, host callbacks (0.5.0 has only two-queue submission and cross-queue waits on command lists, EXPERIMENTAL).
- Cached read-write buffers; unified memory.
- Launches with more than the 32 argument words, or an asynchronous single launch.
- A second real backend (WSL2 `/dev/dxg` is on branch `rig/gb20x`; native AIENOS is planned), so the platform contract has only been exercised by one.
- A runtime function that reports the version of the linked library (the version is a compile-time macro only).

## 9. EXPERIMENTAL entries

Listed in the manifest, may change in any minor version. Signatures are in the manifest and the headers; the names are
checked there. The reason is why each is not STABLE yet.

### 9.1 Session extras (`src/omega_gpu_session.h`, `src/omega_gpu_mem_cached.h`)

| Entry | Reason it is EXPERIMENTAL |
| --- | --- |
| `omega_gpu_session_alloc_cached` | GPU-cached memory is correct only for write-once, never-polled, GPU-read-only weights; the API cannot enforce that (the caller must). |
| `omega_gpu_session_scratch_reserve`, `omega_gpu_session_scratch_unreserve` | Serving-daemon feature (no driver call after warm-up); one user. |
| `omega_gpu_session_cancel` | A cancel after submit cannot stop the chip: it latches the process. Semantics may tighten when recovery (#518) lands. |
| `omega_gpu_session_extra_queue` | A second GPFIFO channel in the same group, only with `AIEN_OMEGA_GPU_CHANNELS` > 1 at open; one backend, chip qualification in CUDApp#113. |
| `omega_gpu_session_recover`, `omega_gpu_session_recoveries` | Recovery after a latch (omega-mv#518, C5): closes and reopens the device in-process; every old handle and list goes stale. Chip qualification in CUDApp#119. |
| `omega_gpu_session_set_platform`, `omega_gpu_session_platform`, `omega_gpu_session_submit_prefix_words`, `omega_gpu_session_completion_mode`, `omega_gpu_session_now_ns` | Tied to the platform contract, which has one real backend. |
| `omega_gpu_host_alloc`, `omega_gpu_host_calloc`, `omega_gpu_host_free` | CPU-only memory from the platform HOST_MEM service; the freestanding build needs them, hosted code can use malloc. |
| `OmegaGpuSession` | The session's own launch buffers; a caller needs none of its fields. |

### 9.2 Platform contract (`src/omega_gpu_platform.h`, ABI 1.2)

Types: `OmegaGpuPlatform` (the service table), `OmegaGpuPlatformCaps`, `OmegaGpuBufferDesc`, `OmegaCpuView`, `OmegaGpuVa`,
`OmegaGpuBusAddr`, `OmegaGpuHandleTable`. Functions: `omega_gpu_platform_refusal_name`, `omega_gpu_platform_negotiate`,
`omega_gpu_platform_check`, `omega_gpu_platform_check_for`, `omega_gpu_platform_check_buffer`, `omega_gpu_handles_init`,
`omega_gpu_handles_open`, `omega_gpu_handles_check`, `omega_gpu_handles_close`. Enums and constants: the handle kinds,
rights (`OMEGA_GPU_RIGHT_*`), cache policy, services (`OMEGA_GPU_SVC_*`) and refusal names (`OMEGA_GPU_PLAT_*`).

Reason: the contract is written for three backends but only the Linux reference has run; the second backend (`rig/gb20x`) and
the native one will likely force changes. What it already promises (from the header): resources are typed handles with a
generation; every byte range carries a length and rights; anything the CPU polls is `OMEGA_GPU_RIGHT_HOST_POLLED` and must be
GPU-uncached; polling a word is never claimed as interrupt support; a missing service is refused, never emulated.

### 9.3 Command lists (`src/omega_gpu_cmdlist.h`)

Types: `OmegaGpuCmdListOpts`, `OmegaGpuCmdListStats`, the opaque `OmegaGpuCmdList`. Functions: `omega_gpu_cmdlist_create`
(takes the lock itself), `omega_gpu_cmdlist_free`, `omega_gpu_cmdlist_reset`, `omega_gpu_cmdlist_count`,
`omega_gpu_cmdlist_add`, `omega_gpu_cmdlist_mark_independent`, `omega_gpu_cmdlist_submit`, `omega_gpu_cmdlist_submit_pair`,
`omega_gpu_cmdlist_wait_for`, `omega_gpu_cmdlist_is_latched`, `omega_gpu_cmdlist_stamps`, `omega_gpu_cmdlist_barrier_name`.
Up to `OMEGA_GPU_CMDLIST_MAX_LAUNCHES` (4096) launches per list.

Reason: omega-mv#580 is building pipelined and dependency-ordered execution on this module, so the options, statistics
and barrier modes will grow. The measured extra cost of each launch in a list is about 30 us with a wait-for-idle barrier
(`evidence/OWN30B-LAUNCH-COST-20261014T0120Z/r1/receipt.json`, verdict FAIL on 4 of 11 checks). A timeout latches the list
and then the session. The `OMEGA_GPU_CMDLIST_BARRIER_NONE` mode and `skip_launch` option are test-only.

### 9.4 Job engine (`src/omega_gpu_engine.h`)

Types: `OmegaGpuJob`, `OmegaGpuResult`, `OmegaGpuTimeouts`, `OmegaGpuBackend`, `OmegaGpuBackendDiag`. Functions:
`omega_gpu_execute`, `omega_gpu_engine_set_backend`, `omega_gpu_engine_is_blocked`, `omega_gpu_engine_set_clock`,
`omega_gpu_engine_failure_name`, `omega_gpu_engine_state_name`, `omega_gpu_engine_wait_name`, `omega_gpu_engine_step_name`. One failure name per run
(`OMEGA_GPU_ENGINE_*`), states `INITIAL` to `SUCCESS`, the failed step recorded, and an uncertain completion blocks every
later `omega_gpu_execute` in the process with `UNCERTAIN_COMPLETION_BLOCKED`.

Reason: accepts one job shape (`OMEGA_GPU_LAYOUT_VECTOR_1D`); #580 widens it; the chip receipt for the vector-add job is on an
unmerged branch (`hive/DEEP2-A3-gpu-engine-impl`). Note: the status paragraph at the top of `omega_gpu_engine.h` says only a
fake backend exists; `src/omega_blackwell_engine.c` is on main and `src/omega_blackwell_submit.c` installs it, so that
paragraph is out of date (whether it is chip-qualified on main is UNVERIFIED here).

### 9.5 Libraries (matmul, elementwise, attention, serving)

The entries carried over from aien-sovereign-core's manifest: `src/omega_gpu_matmul_api.h`, `omega_gpu_elementwise_api.h`,
`omega_gpu_attention_api.h`, `omega_gpu_serving.h` (the functions and structs after the `libraries` comment in the
manifest). Reason: shapes and formats follow one model (Qwen3-30B); there is no general any-shape GEMM yet
(`docs/runtime/ROADMAP.md` 1.3, branch `runtime/roadmap`). Attention is the exception since 0.6.0: `omega_gpu_attention_general` (declared after the carried block) takes every head_dim 1..256 and any MHA / GQA / MQA head split, and the carried attention entries fall back to it for the shapes their tuned kernels refuse. They are the surface sovereign-core's Rust binding uses today. They are not described entry by entry in v0;
their headers are the description. One value differs from sovereign-core's pinned copy: `OMEGA_GPU_MATMUL_MAX_CTAS` is 1024 in
the header (changed by omega #473, commit 98cecadc), 64 in their manifest. Omega's manifest is canonical; R12 brings theirs level.


### 9.6 Library entries (manifest spelling)

Types are the manifest's Rust spelling (`*const`, `*mut`, `u32`, ...); the headers hold the C. Every one is EXPERIMENTAL except the three STABLE session entries noted in section 4 and 5.

| Entry | Signature |
| --- | --- |
| `OmegaGpuMatmulInfo` | struct, 160 bytes (fields in the manifest) |
| `OmegaGpuServingBounds` | struct, 44 bytes (fields in the manifest) |
| `OmegaGpuEwInfo` | struct, 120 bytes (fields in the manifest) |
| `OmegaGpuAttnInfo` | struct, 176 bytes (fields in the manifest) |
| `OmegaGpuKvLayout` | struct, 64 bytes (fields in the manifest) |
| `OmegaGpuAttnGeneralDesc` | struct, 64 bytes (fields in the manifest) |
| `omega_gpu_matmul_bf16` | `fn(u32, u32, u32, *const u16, *const u16, *mut f32, *mut OmegaGpuMatmulInfo) -> i32` |
| `omega_gpu_matmul_f32` | `fn(u32, u32, u32, *const f32, *const f32, *mut f32, *mut OmegaGpuMatmulInfo) -> i32` |
| `omega_gpu_matmul_rc_name` | `fn(i32) -> *const c_char` |
| `omega_gpu_matmul_cache_clear` | `fn() -> ()` |
| `omega_gpu_tensor_upload_bf16` | `fn(u32, u32, *const u16, *mut *mut OmegaGpuTensor) -> i32` |
| `omega_gpu_tensor_upload_f32` | `fn(u32, u32, *const f32, *mut *mut OmegaGpuTensor) -> i32` |
| `omega_gpu_tensor_free` | `fn(*mut OmegaGpuTensor) -> ()` |
| `omega_gpu_tensor_shape` | `fn(*const OmegaGpuTensor, *mut u32, *mut u32) -> ()` |
| `omega_gpu_matmul_resident_bf16` | `fn(u32, *const u16, *const OmegaGpuTensor, *mut f32, *mut OmegaGpuMatmulInfo) -> i32` |
| `omega_gpu_matmul_resident_f32` | `fn(u32, *const f32, *const OmegaGpuTensor, *mut f32, *mut OmegaGpuMatmulInfo) -> i32` |
| `omega_gpu_matmul_last_error` | `fn() -> *const c_char` |
| `omega_gpu_matmul_is_blocked` | `fn() -> i32` |
| `omega_gpu_device_close` | `fn() -> ()` |
| `omega_gpu_matmul_set_cta_budget` | `fn(u32) -> ()` |
| `omega_gpu_matmul_cta_budget` | `fn() -> u32` |
| `omega_gpu_reserve_serving` | `fn(*const OmegaGpuServingBounds) -> i32` |
| `omega_gpu_serving_release` | `fn() -> ()` |
| `omega_gpu_matmul_prepare` | `fn(u32, u32, u32) -> i32` |
| `omega_gpu_serving_seal` | `fn() -> ()` |
| `omega_gpu_rmsnorm_f32` | `fn(u32, u32, *const f32, *const f32, f32, *mut f32, *mut OmegaGpuEwInfo) -> i32` |
| `omega_gpu_rope_f32` | `fn(u32, u32, *const f32, *const f32, *const f32, *mut f32, *mut OmegaGpuEwInfo) -> i32` |
| `omega_gpu_swiglu_f32` | `fn(u32, *const f32, *const f32, *mut f32, *mut OmegaGpuEwInfo) -> i32` |
| `omega_gpu_elementwise_rc_name` | `fn(i32) -> *const c_char` |
| `omega_gpu_gqa_attention_f32` | `fn(*const f32, *const f32, *const f32, u32, u32, u32, u32, *mut f32, *mut OmegaGpuAttnInfo) -> i32` |
| `omega_gpu_gqa_attention_rows_f32` | `fn(*const f32, *const f32, *const f32, u32, u32, u32, u32, u32, *mut f32, *mut OmegaGpuAttnInfo) -> i32` |
| `omega_gpu_attention_pool_alloc` | `fn(usize, *mut *mut c_void) -> i32` |
| `omega_gpu_attention_pool_free` | `fn() -> ()` |
| `omega_gpu_paged_attention_bf16` | `fn(*const f32, *const c_void, *const OmegaGpuKvLayout, *const u32, u32, u32, u32, u32, u32, u32, *mut f32, *mut OmegaGpuAttnInfo) -> i32` |
| `omega_gpu_paged_attention_batch_bf16` | `fn(*const f32, *const c_void, *const OmegaGpuKvLayout, *const i32, *const i32, u32, u32, u32, u32, u32, u32, *mut f32, *mut OmegaGpuAttnInfo) -> i32` |
| `omega_gpu_attention_rc_name` | `fn(i32) -> *const c_char` |
| `omega_gpu_attention_last_error` | `fn() -> *const c_char` |
| `omega_gpu_attention_general` | `fn(*const OmegaGpuAttnGeneralDesc, *const f32, *const c_void, *const c_void, *mut f32, *mut OmegaGpuAttnInfo) -> i32` |

### 9.7 Decode path (ADR 0043 task P2.2)

The kernel APIs the serving decode and prefill paths call, listed so the check sees any change to them. All are
EXPERIMENTAL for the same three reasons: each serves one model (Qwen3-30B-A3B, the OWN-30B lane), each is proven on one
chip (sm_121), and each is tied to its kernel generator in `src/omega_blackwell_codegen_*.h`, which ADR 0043 moves behind
a backend. Signatures and struct layouts are in the manifest. They were added in 0.3.0, a MINOR bump (section 2).
Their result codes (`*_OK`, `*_BAD_ARGS`, `*_TOO_LARGE`, `*_CODEGEN_FAIL`, `*_CHIP_FAIL`, `*_NAN_WEIGHT`, `*_BAD_TRIT`) are
EXPERIMENTAL constants in the manifest; the other constants these headers define (shape limits, kernel slot ids, mode
values, poison patterns) are in the INTERNAL list.

| Header | Entries | Reason it is EXPERIMENTAL |
| --- | --- | --- |
| `src/omega_gpu_fp8_api.h` | `OmegaGpuFp8Info`, `omega_gpu_fp8_tensor_upload`, `omega_gpu_fp8_tensor_free`, `omega_gpu_fp8_tensor_shape`, `omega_gpu_fp8_gemv_f32`, `omega_gpu_fp8_gemv_bf16`, `omega_gpu_fp8_gemv_grouped_f32`, `omega_gpu_fp8_gemv_grouped_dev`, `omega_gpu_fp8_gemv_grouped_record`, `omega_gpu_fp8_tensor_entry`, `omega_gpu_fp8_prepare`, `omega_gpu_fp8_seal`, `omega_gpu_fp8_rc_name`, `omega_gpu_fp8_last_error`, `omega_gpu_fp8_cache_clear`, `omega_gpu_fp8_set_shape`, `omega_gpu_fp8_set_depth`, `omega_gpu_fp8_set_cta_budget`, `omega_gpu_fp8_set_weight_cache`, `omega_gpu_fp8_set_variant`, `omega_gpu_fp8_variant`, `omega_gpu_fp8_weight_cache`, `omega_gpu_fp8_set_gpr_pad`, `omega_gpu_fp8_shape_for`, `omega_gpu_fp8_set_bigk`, `omega_gpu_fp8_bigk`, `omega_gpu_fp8_set_coal`, `omega_gpu_fp8_coal`, `omega_gpu_fp8_set_deep`, `omega_gpu_fp8_deep`, `omega_gpu_fp8_set_sched`, `omega_gpu_fp8_sched`, `omega_gpu_fp8_set_wfirst`, `omega_gpu_fp8_wfirst`, `omega_gpu_fp8_quantize_bf16`, `omega_gpu_fp8_e4m3_rne`, `omega_gpu_fp8_grouped_ctas_rule`, `omega_gpu_fp8_grouped_ctas_auto` | FP8 e4m3 block-scaled GEMV for the OWN-30B lane (Qwen3-30B-A3B): decode GEMV and grouped experts; depth and weight-cache knobs are tuned on one chip. |
| `src/omega_gpu_fp8gemm_api.h` | `OmegaGpuFp8GemmInfo`, `omega_gpu_fp8gemm_f32`, `omega_gpu_fp8gemm_bf16`, `omega_gpu_fp8gemm_prepare`, `omega_gpu_fp8gemm_rc_name`, `omega_gpu_fp8gemm_last_error`, `omega_gpu_fp8gemm_cache_clear`, `omega_gpu_fp8gemm_shape_for`, `omega_gpu_fp8gemm_set_shape`, `omega_gpu_fp8gemm_set_cta_budget`, `omega_gpu_fp8gemm_prepare_shape`, `omega_gpu_fp8gemm_cta_budget_for`, `omega_gpu_fp8gemm_record` | FP8 e4m3 grouped GEMM for OWN-30B prefill; one model's shapes. |
| `src/omega_gpu_bf16_gemv_api.h` | `OmegaGpuBf16Info`, `omega_gpu_bf16_tensor_upload`, `omega_gpu_bf16_tensor_free`, `omega_gpu_bf16_tensor_shape`, `omega_gpu_bf16_gemv_f32`, `omega_gpu_bf16_gemv_bf16`, `omega_gpu_bf16_gemv_record`, `omega_gpu_bf16_prepare`, `omega_gpu_bf16_rc_name`, `omega_gpu_bf16_last_error`, `omega_gpu_bf16_cache_clear`, `omega_gpu_bf16_shape_for`, `omega_gpu_bf16_set_shape`, `omega_gpu_bf16_set_cta_budget`, `omega_gpu_bf16_gemv_mask_record`, `omega_gpu_bf16_gemv_rows_record`, `omega_gpu_bf16_gemv_xbatch_record`, `omega_gpu_bf16_prepare_mask`, `omega_gpu_bf16_prepare_xbatch` | bf16 streaming GEMV for the OWN-30B lm_head (vocab 151936 x hidden 2048); one shape. |
| `src/omega_gpu_decode_attn_api.h` | `OmegaGpuDecodeAttnInfo`, `OmegaGpuDecodeAttnCacheView`, `omega_gpu_decode_attn_cache_alloc`, `omega_gpu_decode_attn_cache_free`, `omega_gpu_decode_attn_set_norms`, `omega_gpu_decode_attn`, `omega_gpu_decode_attn_record`, `omega_gpu_decode_attn_counters_check`, `omega_gpu_decode_attn_cache_write`, `omega_gpu_decode_attn_cache_read`, `omega_gpu_decode_attn_cache_view`, `omega_gpu_decode_attn_last_error`, `omega_gpu_decode_attn_rc_name`, `omega_gpu_decode_attn_set_depth`, `omega_gpu_decode_attn_set_per_split` | One-launch decode attention for Qwen3-30B-A3B with its own KV cache; the cache view hands out device addresses. |
| `src/omega_gpu_router_api.h` | `OmegaGpuRouterInfo`, `OmegaGpuRouterDevOut`, `omega_gpu_router_alloc`, `omega_gpu_router_free`, `omega_gpu_router_set_layer`, `omega_gpu_router_set_directory`, `omega_gpu_router_set_directory_fp8`, `omega_gpu_router`, `omega_gpu_router_device_outputs`, `omega_gpu_router_record`, `omega_gpu_router_read_back`, `omega_gpu_router_last_error`, `omega_gpu_router_rc_name`, `omega_gpu_router_set_shape`, `omega_gpu_router_alloc_shape`, `omega_gpu_router_layout_bytes`, `omega_gpu_router_prepare_rows`, `omega_gpu_router_record_to`, `omega_gpu_router_record_rows` | One-launch MoE router for Qwen3-30B-A3B; the device-output path writes device addresses. |
| `src/omega_gpu_tern_api.h` | `OmegaGpuTernSegments`, `OmegaGpuTernInfo`, `omega_gpu_tern_tensor_upload`, `omega_gpu_tern_tensor_free`, `omega_gpu_tern_tensor_shape`, `omega_gpu_tern_tensor_bytes`, `omega_gpu_tern_gemv_i32`, `omega_gpu_tern_gemv_f32`, `omega_gpu_tern_gemv_record`, `omega_gpu_tern_rc_name`, `omega_gpu_tern_last_error`, `omega_gpu_tern_cache_clear`, `omega_gpu_tern_pack`, `omega_gpu_tern_padded_rows`, `omega_gpu_tern_segment_bounds`, `omega_gpu_tern_shape_for`, `omega_gpu_tern_set_layout`, `omega_gpu_tern_set_shape`, `omega_gpu_tern_set_cta_budget`, `omega_gpu_tern_set_weight_cache` | Ternary engine stage T0 (W2A8 ternary GEMV); the packing layout is still being designed. |
| `src/omega_gpu_tcgemm_api.h` | `OmegaGpuTcGemmInfo`, `omega_gpu_tcgemm_f32`, `omega_gpu_tcgemm_prepare`, `omega_gpu_tcgemm_rc_name`, `omega_gpu_tcgemm_last_error`, `omega_gpu_tcgemm_cache_clear`, `omega_gpu_tcgemm_set_shape` | OWN-30B T3 cut 1: FP8 grouped GEMM on the tensor cores; one model's shapes. |
| `src/omega_gpu_prefill_attn_api.h` | `OmegaGpuPrefillAttnInfo`, `OmegaGpuPrefillAttnMirrorView`, `omega_gpu_prefill_attn`, `omega_gpu_prefill_attn_release`, `omega_gpu_prefill_attn_dev`, `omega_gpu_prefill_attn_set_warps`, `omega_gpu_prefill_attn_set_fused`, `omega_gpu_prefill_attn_set_kv_mirror`, `omega_gpu_prefill_attn_kv_mirror_view`, `omega_gpu_prefill_attn_kv_mirror_invalidate`, `omega_gpu_prefill_attn_kv_mirror_forget`, `omega_gpu_prefill_attn_kv_mirror_available` | Causal prefill attention for Qwen3-30B-A3B, with an optional KV mirror; one model. |
| `src/omega_gpu_moe_api.h` | `omega_gpu_moe_tensor_upload_bf16`, `omega_gpu_moe_tensor_upload_f32`, `omega_gpu_moe_tensor_free`, `omega_gpu_moe_tensor_shape`, `omega_gpu_matmul_resident_window_f32`, `omega_gpu_matmul_resident_window_bf16`, `omega_gpu_matmul_resident_windows_f32`, `omega_gpu_moe_reserve`, `omega_gpu_moe_prepare`, `omega_gpu_moe_seal`, `omega_gpu_moe_unreserve`, `omega_gpu_moe_cache_clear`, `omega_gpu_moe_set_cta_budget`, `omega_gpu_moe_cta_budget`, `omega_gpu_moe_window_byte_offset` | Column windows over a resident bf16 expert tensor (OWN-30B); tied to the serving pool. |
| `src/omega_gpu_decode_graph.h` | `OmegaGpuDecodeGraphConfig`, `OmegaGpuDecodeGraphStats`, `OmegaGpuDecodeGraphVas`, `omega_gpu_decode_graph_create`, `omega_gpu_decode_graph_free`, `omega_gpu_decode_graph_set_layer`, `omega_gpu_decode_graph_set_final_norm`, `omega_gpu_decode_graph_token`, `omega_gpu_decode_graph_begin_token`, `omega_gpu_decode_graph_record_layer`, `omega_gpu_decode_graph_record_final`, `omega_gpu_decode_graph_end_token`, `omega_gpu_decode_graph_record_res`, `omega_gpu_decode_graph_record_swiglu`, `omega_gpu_decode_graph_list`, `omega_gpu_decode_graph_vas`, `omega_gpu_decode_graph_read`, `omega_gpu_decode_graph_set_stamps`, `omega_gpu_decode_graph_set_barrier`, `omega_gpu_decode_graph_barrier`, `omega_gpu_decode_graph_stamps`, `omega_gpu_decode_graph_kind_name`, `omega_gpu_decode_graph_set_flush_each`, `omega_gpu_decode_graph_set_prefetch`, `omega_gpu_decode_graph_prefetch` | One Qwen3-30B-A3B decode token as one command list; scheduling, barrier and prefetch modes are still measured. |
| `src/omega_gpu_prefill.h` | `OmegaPfWeights`, `omega_gpu_pf_rc_name`, `omega_gpu_pf_alloc_dev`, `omega_gpu_pf_alloc_host`, `omega_gpu_pf_free`, `omega_gpu_pf_upload`, `omega_gpu_pf_download`, `omega_gpu_pf_copy_va`, `omega_gpu_pf_stats`, `omega_gpu_pf_stats_reset`, `omega_gpu_pf_weights_upload`, `omega_gpu_pf_weights_free`, `omega_gpu_pf_weights_host_form`, `omega_gpu_pf_gemm_bm`, `omega_gpu_pf_gemm_tiles`, `omega_gpu_pf_gemm`, `omega_gpu_pf_gemm_set_shape`, `omega_gpu_pf_quant`, `omega_gpu_pf_qgemm_bm`, `omega_gpu_pf_qgemm_tiles`, `omega_gpu_pf_qgemm_tiles_va`, `omega_gpu_pf_qgemm`, `omega_gpu_pf_qgemm_set_shape`, `omega_gpu_pf_quant_record`, `omega_gpu_pf_qgemm_record` | Device-resident one-pass prefill for OWN-30B; arguments carry raw device addresses. |
| `src/omega_gpu_pfhost.h` | `omega_gpu_pf_router_rows`, `omega_gpu_pf_router_packed_bytes`, `omega_gpu_pf_router_pack`, `omega_gpu_pf_router_logits`, `omega_gpu_pf_moe_combine`, `omega_gpu_pf_router_set_rows` | Whole-prompt prefill router logits and MoE combine for OWN-30B; arguments carry raw device addresses. |

## 10. INTERNAL (not part of the API)

Not in the manifest; may change or vanish without notice. The complete list of such names in the manifest headers, each
with its reason, is `docs/runtime/omega_runtime_api.internal` (192 names in 0.6.0: test hooks, kernel builders, encoder
probes, serving-daemon internals and launch limits; the 25 serving internals and launch limits are marked "candidate
EXPERIMENTAL", the ones a caller is most likely to want promoted). `make test-runtime-api` fails on a name in a manifest header that is in neither file. Notable ones:
- `omega_gpu_session_set_error` (library code names its failure stage through it).
- `omega_gpu_session_on_close` (lets a library drop its resident kernels before the session closes; up to 8 hooks).
- `omega_gpu_engine_test_reset_block`, `omega_gpu_engine_test_force_block`, `omega_gpu_engine_test_set_clock`: TEST ONLY. A blocked process must stay blocked because the GPU may still be running.
- Take or return a kernel-generator type, which is backend (ADR 0043): `omega_gpu_fp8_depth_for`, `omega_gpu_fp8_kernel_fits`, `omega_gpu_fp8gemm_kernel_fits`, `omega_gpu_bf16_kernel_fits`, `omega_gpu_tern_kernel_fits`, `omega_gpu_tcgemm_spec_for`, `omega_gpu_prefill_attn_encode`.
- TEST ONLY (mutant and fault-injection setters in the decode-path headers): `omega_gpu_fp8_test_set_mutant`, `omega_gpu_fp8_test_set_no_fold`, `omega_gpu_fp8gemm_test_set_mutant`, `omega_gpu_bf16_test_set_mutant`, `omega_gpu_decode_attn_test_set_mutant`, `omega_gpu_router_test_set_mutant`, `omega_gpu_tern_test_set_mutant`, `omega_gpu_tcgemm_test_set_mutant`, `omega_gpu_prefill_attn_test_set_mutant`, `omega_gpu_prefill_attn_test_set_mirror_per_call`, `omega_gpu_prefill_attn_test_launch_ns`, `omega_gpu_moe_test_set_mutant`, `omega_gpu_moe_test_set_window_skew`, `omega_gpu_moe_test_a_row0_hits`, `omega_gpu_decode_graph_test_set_mutant`, `omega_gpu_decode_graph_test_set_barrier_none`, `omega_gpu_decode_graph_test_set_prefetch_red`, `omega_gpu_pf_gemm_test_set_mutant`, `omega_gpu_pf_quant_test_set_mutant`, `omega_gpu_pf_qgemm_test_set_mutant`, `omega_gpu_pf_router_test_set_mutant`, `omega_gpu_pf_combine_test_set_mutant`.
- Host reference and packing helpers in the decode-path headers (not `omega_gpu_` entries): `omega_fp8_e4m3_ref`, `omega_fp8_e4m3_kernel_bits`, `omega_fp8_e4m3_is_nan`, `omega_fp8_pack_scan_nan`, `omega_fp8_pack_scales`, `omega_fp8_pack_x_f32`, `omega_fp8_pack_x_bf16`, `omega_fp8_gemv_oracle`, `omega_fp8gemm_permute_x`, `omega_dattn_bf16_rne`, `omega_dattn_bf16_f32`, `omega_dattn_rope_row`.
- Launch helpers whose caller already holds the session lock: `omega_gpu_pf_kernel_locked`, `omega_gpu_pf_launch_locked`, `omega_gpu_pf_open_locked`.
- Everything under `src/omega_gpu_*` and `src/omega_blackwell_*` in headers the manifest does not name (kernel builder, encoder, per-model APIs, launch helpers).

## 11. Checking this file

`make test-runtime-api` runs `tools/runtime_api_check.sh` on the real headers and then `tools/runtime_api_negative.sh`. The
negative script breaks a copy of the headers (or this file) in fourteen ways and requires the check to fail each time: a
removed function, a missing MAJOR and a missing PATCH version macro, a changed function signature, a changed struct, a
changed constant, a version line in this file that disagrees with the macro, a quoted signature in this file that no header
has, a quoted signature that only appears inside a longer one (`int32_t` inside `uint32_t`), a STABLE entry with no section
here, an added undeclared function, an added undeclared macro, an added undeclared enumerator, and a stale INTERNAL entry.
It then breaks a copy of the manifest five ways and requires `tools/runtime_api_semver.sh` to fail each time (a removed
STABLE function with no bump on a 0.x base, a removed STABLE function with only a MINOR bump on a base already at 1.x, a
changed STABLE signature with no bump, an added function with no bump, a version that goes down) and to pass two correct
changes (a removed STABLE function with a MINOR bump on a 0.x base; an added function with a MINOR bump).
`make test-runtime-api-semver` compares the real manifest with the base branch's. None of it needs a GPU or a physics
checkout. It checks: every manifest function against its header with exact types; sizes, offsets and constants against a
compiled probe; the version macros against the manifest and this file; every public name in the manifest headers against
the manifest and the INTERNAL list; that every STABLE entry has a `####` section here and every EXPERIMENTAL one is named
here; that every C signature quoted in this file appears verbatim in a header; and the size of the version bump. It does
not check that the prose is true. A struct field added inside padding is not seen, nor a field that changes type but keeps
its size, nor a constant value quoted in prose, nor a new type (struct or typedef) that no function uses yet, nor a
function-like macro.
