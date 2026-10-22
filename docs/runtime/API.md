# Omega Runtime API v0

Omega Runtime API version: 0.1

Status: first written version, 2026-10-19 (slice R1, omega-mv#594, tracking #593). Source of truth is the C headers in
`src/`. This file describes them; `docs/runtime/omega_runtime_api.manifest` lists them; `make test-runtime-api` checks the
list against the headers (a host check, no GPU). Review state: reviewed by Opus 5.5 against the headers and the code on 2026-10-22 (L386); its findings are fixed
in this version. The check proves the list and every quoted C signature match the headers; prose is checked by review.

The version number lives in one header, `src/omega_runtime_api.h` (`OMEGA_RUNTIME_API_VERSION_MAJOR`, `_MINOR`).

## 1. What this is

Omega is a GPU compute runtime that does not use CUDA. This file is the written contract for the part a program calls
directly: find and open the GPU, get memory, copy data in and out, run a kernel, run many kernels in one submission, run a
packaged job, and read errors. It covers what exists on the main branch today, on one machine type (GB10, the DGX Spark),
through the Linux open-driver backend. Nothing here claims more than that.

How to read an entry:

| Tier | Meaning |
| --- | --- |
| **STABLE** | Used by other code in the project, covered by chip results, and kept compatible under the rule in section 2. Has its own `####` section below. |
| **EXPERIMENTAL** | Works and is listed, but may change in any minor version, because the design is still moving or only one use of it exists. Listed in tables with the reason. |
| **INTERNAL** | Exists in a header for the project's own libraries or tests. Not part of the API; not in the manifest; may change or disappear any time. Listed in section 9 so nobody builds on it by accident. |

Every C signature in this file is copied from the header it names. `tools/runtime_api_check.sh` re-checks every quoted
signature against the headers on every run.

## 2. Version and compatibility rule

The version is MAJOR.MINOR, currently **0.1**. It is a property of the list of entries and their behaviour, not of the
library build.

**A MINOR bump (0.1 to 0.2) is allowed to:**
- add a function, a constant, an enum value at the end of an existing enum, or a new struct;
- add an EXPERIMENTAL entry, change or remove an EXPERIMENTAL entry, or move an entry from EXPERIMENTAL to STABLE;
- fix a STABLE function that did not do what this file says (the fix makes it match);
- add a new failure name for the same call to report (callers must treat an unknown failure code or stage text as a failure).

**It needs a MAJOR bump (0.x to 1.0, then 1.x to 2.0) to:**
- remove or rename a STABLE function, struct, field or constant;
- change a STABLE function's argument or return types, a STABLE struct's size, field order or field offsets, or the value
  of a STABLE constant or enum value;
- change documented behaviour of a STABLE entry: what the lock covers, what a latch does, which buffers get parked, what
  a return value means;
- move a STABLE entry down to EXPERIMENTAL or INTERNAL;
- raise the platform contract's major number (section 9.2).

MAJOR 0 means "the first written version", not "anything goes": the rule above already binds STABLE entries in 0.x. What
0 adds is a warning: the project may promote entries to STABLE freely until 1.0 (moving a STABLE entry down still needs a
MAJOR bump, as above), and the first 1.0 will be cut only when
section 8 (what v0 does not have) has been looked at again.

How it is enforced: a header change that moves any listed entry fails `make test-runtime-api` until the manifest line is
edited on purpose (`tools/runtime_api_check.sh --bless` refreshes the numbers; the diff is reviewed). A reviewer then decides
MINOR or MAJOR using the rule above and edits the version macros and the `Omega Runtime API version:` line in this file;
the check fails if the three disagree. The check cannot tell whether the bump was the right size. That is a review duty.

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
| `timeout_ms` | completion poll limit, applied to each of the launch's two markers in turn (so a launch can wait up to twice this); past it the completion is uncertain and **the process latches**. Must not be 0: 0 runs out at once and latches |
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
- More than one device or channel in a process; sharing the GPU between processes (a second channel failed under load: omega-mv#523).
- Recovery after a latch (omega-mv#518): a single timeout ends GPU use for the whole process.
- Async copy, streams, events, host callbacks (omega-mv#580 is building ordering on command lists).
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
`omega_gpu_cmdlist_add`, `omega_gpu_cmdlist_mark_independent`, `omega_gpu_cmdlist_submit`, `omega_gpu_cmdlist_is_latched`,
`omega_gpu_cmdlist_stamps`, `omega_gpu_cmdlist_barrier_name`. Up to `OMEGA_GPU_CMDLIST_MAX_LAUNCHES` (2048) launches per list.

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
(`docs/runtime/ROADMAP.md` 1.3, branch `runtime/roadmap`). They are the surface sovereign-core's Rust binding uses today. They are not described entry by entry in v0;
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

## 10. INTERNAL (not part of the API)

Not in the manifest; may change or vanish without notice.
- `omega_gpu_session_set_error` (library code names its failure stage through it).
- `omega_gpu_session_on_close` (lets a library drop its resident kernels before the session closes; up to 8 hooks).
- `omega_gpu_engine_test_reset_block`, `omega_gpu_engine_test_force_block`, `omega_gpu_engine_test_set_clock`: TEST ONLY. A blocked process must stay blocked because the GPU may still be running.
- Everything else under `src/omega_gpu_*` and `src/omega_blackwell_*` not named above (kernel builder, encoder, per-model APIs, launch helpers).

## 11. Checking this file

`make test-runtime-api` runs `tools/runtime_api_check.sh` on the real headers and then `tools/runtime_api_negative.sh`, which breaks a
copy of the headers (or this file) in nine ways and requires the check to fail each time: a removed function, a missing version macro, a
changed function signature, a changed struct, a changed constant, a version line in this file that disagrees with the macro, a quoted signature in this file that no header has, a quoted
signature that only appears inside a longer one (`int32_t` inside `uint32_t`), and a STABLE entry with no section here. It needs no GPU and no physics checkout. It checks: every manifest function against its header
with exact types; sizes, offsets and constants against a compiled probe; the version macros against the manifest and this file;
that every STABLE entry has a `####` section here and every EXPERIMENTAL one is named here; and that every C signature quoted
in this file appears verbatim in a header. It does not check that the prose is true. A struct field added inside padding is not seen, nor a field that changes type but keeps its size, nor a constant
value quoted in prose.
