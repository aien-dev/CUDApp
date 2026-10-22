#ifndef OMEGA_GPU_SESSION_H
#define OMEGA_GPU_SESSION_H
/*
 * omega_gpu_session: the ONE persistent device for the native GPU APIs
 * (FB-1 cut 4b, 2026-10-04). Shared by omega_gpu_matmul_api,
 * omega_gpu_elementwise_api and omega_gpu_attention_api.
 *
 * Before this cut the matmul API kept its own device open for the process while
 * the elementwise and attention APIs opened a client, device, VA space and
 * channel for EVERY call and closed them again (~40 ms per call against ~0.1 ms
 * of chip time; TinyLlama decode makes ~3500 elementwise calls per 32 tokens,
 * sovereign-core cut 3c: 0.207 tok/s). Now:
 *  - one RM client / device / channel per process, opened on the first launch of
 *    any API and kept until omega_gpu_session_close();
 *  - one pushbuffer, constant bank, marker page and QMD page, reused per launch;
 *  - staging buffers owned by the callers grow to a high-water mark and are reused
 *    (omega_gpu_session_scratch);
 *  - every buffer is GPU-uncached (nvos.h NVOS32_ATTR2_GPU_CACHEABLE, read
 *    2026-10-04: "For system memory this will not be coherent with direct CPU
 *    mappings"; the default for system memory is uncached), because the host
 *    writes inputs the chip then reads, and polls the markers the chip writes;
 *  - every launch keeps the chip-proven sequence: shader-cache invalidate
 *    (clcec0.h NVCEC0_INVALIDATE_SHADER_CACHES, cut 1b finding), constant bank
 *    upload, QMD0 / QMD1, first marker, L2_FLUSH_DIRTY, second marker on the
 *    uncached page (C3 tail, omega#225/#230), GPFIFO retire after completion.
 *  - an uncertain completion (marker wait timed out) latches the process: nothing
 *    is freed or closed (the chip may still be running) and every later launch
 *    of every API is refused, as the engine does (omega_gpu_engine.h).
 *
 * Threading: one process-wide mutex. A caller takes omega_gpu_session_lock(),
 * does its whole call (kernel lookup, staging, launches, readback) and unlocks.
 * Calls from several threads therefore serialise on the device; that is the
 * documented model (one channel, one GPFIFO).
 *
 * Diagnostics (cut 4b flake investigation): every failure names its stage
 * (omega_gpu_session_last_error) and an open failure carries the driver text
 * (Nvrm.err: RM class, ioctl errno, RM status) so a first-call failure is never
 * reported as a bare CHIP_FAIL again.
 *
 * Portability (native v1 P07 cut C, session part; aienos#286, docs/numeric/OMEGA_GPU_NATIVE_BOUNDARY.md):
 * this file and omega_gpu_session.c no longer name a Linux driver type or call one. Everything the session
 * needs from the machine goes through the installed OmegaGpuPlatform (src/omega_gpu_platform.h, ABI 1.1,
 * service set OMEGA_GPU_SVC_REQUIRED_SESSION): device open/close, typed DMA buffers with explicit rights,
 * kind and GPU-uncached policy, the submission queue, POLLING completion (poll_word; the session never
 * claims or waits on an interrupt), explicit CPU<->GPU sync, the nanosecond clock, the session lock and
 * CPU-only host memory. omega_gpu_session_set_platform installs one; the hosted build installs the Linux
 * reference (src/omega_gpu_platform_linux.c, reference only) on first use when none was installed; the
 * freestanding build (OMEGA_GPU_NO_POSIX) has no default and refuses until a platform is installed.
 * Buffers are OmegaGpuMem: a typed handle (index + generation) plus the CPU view and the GPU VA as two
 * distinct fields; no bus or physical address reaches the APIs. A handle from a closed device is stale and
 * every use of it is refused.
 *
 * Uncertain completion parks, it never releases: after a marker poll timed out or a launch was cancelled
 * after submit, the device may still be reading or writing any session buffer, so every buffer the session
 * or an API frees from then on is PARKED (kept allocated, counted in omega_gpu_session_parked) and never
 * returned to the platform; close and reopen are refused; the latch holds for the life of the process.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "omega_gpu_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A device buffer: typed handle, kind, CPU view (cpu, size bytes) and GPU VA (va). cpu NULL = none. */
typedef struct {
    OmegaGpuBufferHandle handle;
    uint32_t kind;  /* OMEGA_GPU_HANDLE_BUF_DATA, _BUF_CODE or _BUF_QUEUE */
    void *cpu;      /* CPU view (OmegaCpuView.ptr) */
    uint64_t va;    /* GPU virtual address (OmegaGpuVa.va) */
    uint64_t size;  /* bytes of both views */
} OmegaGpuMem;

typedef struct {
    OmegaGpuDeviceHandle dev;
    OmegaGpuQueueHandle queue;
    OmegaGpuMem pb, cbank, marker, qmd; /* per-launch scratch, owned by the session (pb is the queue ring) */
    uint32_t opens;                     /* device opens in this process */
    uint64_t last_open_ns;              /* wall time of the last successful open */
} OmegaGpuSession;

/* A caller-owned staging buffer that grows to its high-water mark. */
typedef struct {
    OmegaGpuMem mem;
    size_t cap;        /* usable bytes */
    size_t high_water; /* largest request so far */
    bool reserved;     /* serving reservation active: a request past cap is refused, never allocated (omega_gpu_session_scratch_reserve) */
} OmegaGpuScratch;

/* One launch: a kernel already resident at code_va, the argument words for
 * constant bank 0 at 0x380 (buffer addresses then parameters), the grid. */
typedef struct {
    uint64_t code_va;
    uint32_t gpr_count;          /* registers the QMD declares (multiple of 8, >= 64 in practice) */
    uint32_t threads_x, threads_y;
    uint32_t grid_x, grid_y;
    size_t num_elements;         /* QMD config num_elements (kept per API for identical words) */
    const uint32_t *args;        /* n_args words written at cbank + 0x380 */
    uint32_t n_args;             /* <= OMEGA_GPU_SESSION_MAX_ARGS */
    uint64_t timeout_ms;         /* marker poll; past it the completion is uncertain and the process latches */
    uint32_t shared_bytes;       /* shared memory per CTA; 0 = the QMD default (1024) */
    uint32_t spin_us;            /* poll the marker without sleeping for up to this long, then the platform's
                                    poll_word (Linux reference: 50 us sleep-poll) until timeout_ms; 0 = the process-wide setting
                                    (omega_gpu_session_set_spin_us) */
} OmegaGpuLaunch;
#define OMEGA_GPU_SESSION_MAX_SPIN_US 1000000u
#define OMEGA_GPU_SESSION_MAX_ARGS 32u
/* poll_word is called in slices of at most this many ms so a cancel request is seen within one slice */
#define OMEGA_GPU_SESSION_POLL_SLICE_MS 1u

/* ---- platform (ABI 1.2) ---- */

/* Installs the platform the session runs on. Negotiates caps against OMEGA_GPU_SVC_REQUIRED_SESSION and checks the
 * table (omega_gpu_platform_check_for). OMEGA_GPU_PLAT_OK, or the refusal; REFUSED_BUSY while the device is open,
 * the lock is held or the process is latched. NULL p uninstalls (refused the same way). Install once, at start. */
int omega_gpu_session_set_platform(const OmegaGpuPlatform *p, const OmegaGpuPlatformCaps *caps);
/* The installed platform (the hosted build installs the Linux reference here when none was installed); NULL in the
 * freestanding build until omega_gpu_session_set_platform. */
const OmegaGpuPlatform *omega_gpu_session_platform(void);
/* The installed platform's caps.submit_prefix_words: words queue_submit may put ahead of the caller's in the ring, so a
 * caller that fills the ring budgets ring words minus this (omega #457 L2 invalidate prefix). 0 with no platform. */
uint32_t omega_gpu_session_submit_prefix_words(void);

/* Completion is observed by polling a GPU-uncached word; the session never claims interrupt support. */
typedef enum { OMEGA_GPU_COMPLETION_POLLING = 1 } OmegaGpuCompletionMode;
OmegaGpuCompletionMode omega_gpu_session_completion_mode(void);

/* Platform clock in ns (CLOCK_NS); 0 without a platform. */
uint64_t omega_gpu_session_now_ns(void);

/* CPU-only memory for API tables and staging, from the platform HOST_MEM service (Linux reference: malloc/free).
 * NULL without a platform or on refusal; 0 bytes gives a unique 1-byte block (malloc(0) behaviour); calloc also
 * refuses n * size overflow. free(NULL) is a no-op. */
void *omega_gpu_host_alloc(size_t bytes);
void *omega_gpu_host_calloc(size_t n, size_t size);
void omega_gpu_host_free(void *p);

/* ---- lock, open, state ---- */

/* Take the session lock (platform LOCK service, lock id OMEGA_GPU_SESSION_LOCK_ID). Does not touch the device.
 * Without a platform it is a no-op and the open that follows refuses. */
#define OMEGA_GPU_SESSION_LOCK_ID 0u
void omega_gpu_session_lock(void);
void omega_gpu_session_unlock(void);

/* Caller holds the lock. Opens the device on first use. NULL when the process is
 * latched, no platform is installed or the open failed (stage in omega_gpu_session_last_error). */
OmegaGpuSession *omega_gpu_session_open(void);
bool omega_gpu_session_is_open(void);

/* 1 once an uncertain completion has latched this process. */
int omega_gpu_session_is_blocked(void);

/* Process-wide marker-poll spin window, used by every launch whose own spin_us is 0
 * (the matmul, elementwise and attention APIs leave it 0). 0 (the default) keeps the
 * poll_word sleep-poll (Linux reference: 50 us, which Linux timer slack rounds to about 102 us
 * per launch on GB10, omega#328). 0 on success, -1 above OMEGA_GPU_SESSION_MAX_SPIN_US (setting unchanged). */
int omega_gpu_session_set_spin_us(uint32_t us);
uint32_t omega_gpu_session_spin_us(void);

/* ---- buffers ---- */

/* Caller holds the lock, device open. GPU-uncached DATA buffer (CPU and GPU read/write), page rounded. */
int omega_gpu_session_alloc(size_t bytes, OmegaGpuMem *out);
/* Caller holds the lock, device open. A CODE buffer (CPU writes, GPU executes, GPU never writes) for code_size
 * bytes with the 2 KB instruction prefetch tail (omega_gpu_code_alloc.h), the tail zero-filled; the caller writes
 * the code to out->cpu. */
int omega_gpu_session_alloc_code(size_t code_size, OmegaGpuMem *out);
/* Releases m to the platform and zeroes *m. Latched: m is PARKED instead (never released). A handle from a closed
 * device (stale) or a closed session is refused: nothing is released, the refusal is counted
 * (OmegaGpuAllocStats.stale_refusals) and *m is zeroed. */
void omega_gpu_session_free(OmegaGpuMem *m);

/* Explicit visibility of a whole buffer (platform SYNC): CPU writes before the device reads them, device writes
 * before the CPU reads them (Linux reference: dsb sy, the barrier the APIs issued themselves before cut C).
 * 0 also for a buffer with no CPU view (nothing to sync); -1 with the stage named when refused or no device. */
int omega_gpu_session_sync_to_gpu(const OmegaGpuMem *m);
int omega_gpu_session_sync_to_cpu(const OmegaGpuMem *m);

/* Caller holds the lock, device open. Makes s->mem at least `bytes` long
 * (reallocated when it grows; contents are not kept). */
int omega_gpu_session_scratch(OmegaGpuScratch *s, size_t bytes);
/* Caller holds the lock, device open. Allocates s->mem for `bytes` now and marks it reserved: from then on
 * omega_gpu_session_scratch(s, n) with n <= the reserved capacity never touches the driver, and with n past it
 * returns OMEGA_GPU_SESSION_SCRATCH_RESERVED (-2) with a clear error and no allocation. 0 on success, -1 on
 * allocation failure (not reserved then). Cleared by omega_gpu_session_scratch_free (device close) or _unreserve. */
#define OMEGA_GPU_SESSION_SCRATCH_RESERVED (-2)
int omega_gpu_session_scratch_reserve(OmegaGpuScratch *s, size_t bytes);
void omega_gpu_session_scratch_unreserve(OmegaGpuScratch *s);
void omega_gpu_session_scratch_free(OmegaGpuScratch *s);

/* ---- launch, cancel ---- */

/* Caller holds the lock, device open, not latched. Submits one launch and polls
 * for both markers. 0 on success; -1 with the stage named on failure. A marker
 * timeout latches the process (uncertain completion). elapsed_ns (submit to second
 * marker) and the observed first marker word are added / stored when the pointers are given.
 * OMEGA_GPU_SESSION_CANCELLED: a cancel request was seen. Before submit nothing was submitted and the
 * session stays usable; after submit the completion is uncertain and the process latches. */
#define OMEGA_GPU_SESSION_CANCELLED (-3)
int omega_gpu_session_launch(const OmegaGpuLaunch *L, uint64_t *elapsed_ns, uint32_t *marker_out);

/* Asks the launch in progress (or the next one) to stop. Any thread, no lock. Polling cannot stop the device: a
 * launch already submitted is not cancelled on the chip, its completion becomes uncertain (latch + park). The
 * request is consumed by the launch that sees it. */
void omega_gpu_session_cancel(void);

/* Stage of the most recent failure ("" if none). Shared by every API. */
const char *omega_gpu_session_last_error(void);
void omega_gpu_session_set_error(const char *stage);

/* Caller holds the lock. Register a function the session calls (lock held, device
 * still open) right before it closes, so an API can drop its resident kernels and
 * scratch. Up to 8; the same function is registered once. (Taking the lock inside
 * deadlocked the first call of every API on 2026-10-04: the APIs register from
 * their open path, which already holds it.) */
int omega_gpu_session_on_close(void (*hook)(void));

/* Close the device and free every session buffer. Takes the lock itself.
 * Refused (no-op) after an uncertain completion. The next open reopens; every handle of the
 * closed device is stale from then on. */
void omega_gpu_session_close(void);

/* Device opens so far in this process (diagnostics; a persistent session shows 1). */
uint32_t omega_gpu_session_open_count(void);

/* Driver allocations made through omega_gpu_session_alloc (every GPU API buffer, tensor, scratch growth and kernel
 * code buffer) since the process started, for measurement: a serving daemon reads it after warm-up and per request
 * to show that serving asks the driver for nothing. Covers this session layer only, not other engines. */
typedef struct {
    uint64_t allocs;          /* successful driver allocations */
    uint64_t alloc_bytes;     /* their page-rounded bytes */
    uint64_t alloc_failures;  /* driver refusals (e.g. NV_ERR_NO_MEMORY) */
    uint64_t frees;           /* driver frees */
    uint64_t stale_refusals;  /* frees refused because the handle was stale (closed device or double free) */
} OmegaGpuAllocStats;
void omega_gpu_session_alloc_stats(OmegaGpuAllocStats *out);

/* Buffers parked after an uncertain completion: kept allocated, never returned to the platform. The session's own
 * launch buffers (ring, constant bank, marker, QMD page) are parked at the latch; every later free parks too.
 * The first OMEGA_GPU_SESSION_PARK_MAX are listed by handle; past that they are still kept, counted in untracked. */
#define OMEGA_GPU_SESSION_PARK_MAX 512u
typedef struct {
    uint32_t buffers;   /* parked buffers (listed + untracked) */
    uint32_t untracked; /* parked past the list (still never released) */
    uint64_t bytes;     /* their bytes */
} OmegaGpuParkStats;
void omega_gpu_session_parked(OmegaGpuParkStats *out);
/* 1 if this handle is parked (listed). */
int omega_gpu_session_is_parked(OmegaGpuBufferHandle h);

#ifdef __cplusplus
}
#endif
#endif
