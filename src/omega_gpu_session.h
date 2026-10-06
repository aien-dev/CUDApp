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
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "m16_native.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    M16NativeContext ctx;
    NvrmMem pb, cbank, marker, qmd; /* per-launch scratch, owned by the session */
    uint32_t opens;                 /* device opens in this process */
    uint64_t last_open_ns;          /* wall time of the last successful open */
} OmegaGpuSession;

/* A caller-owned staging buffer that grows to its high-water mark. */
typedef struct {
    NvrmMem mem;
    size_t cap;        /* usable bytes */
    size_t high_water; /* largest request so far */
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
    uint64_t timeout_ms;         /* marker wait; past it the process latches */
    uint32_t shared_bytes;       /* shared memory per CTA; 0 = the QMD default (1024) */
    uint32_t spin_us;            /* poll the marker without sleeping for up to this long, then the usual
                                    50 us sleep-poll until timeout_ms; 0 = sleep-poll only (old behavior) */
} OmegaGpuLaunch;
#define OMEGA_GPU_SESSION_MAX_SPIN_US 1000000u
#define OMEGA_GPU_SESSION_MAX_ARGS 32u

/* Take the session lock. Never fails; does not touch the device. */
void omega_gpu_session_lock(void);
void omega_gpu_session_unlock(void);

/* Caller holds the lock. Opens the device on first use. NULL when the process is
 * latched or the open failed (stage in omega_gpu_session_last_error). */
OmegaGpuSession *omega_gpu_session_open(void);
bool omega_gpu_session_is_open(void);

/* 1 once an uncertain completion has latched this process. */
int omega_gpu_session_is_blocked(void);

/* Caller holds the lock, device open. GPU-uncached allocation, page rounded. */
int omega_gpu_session_alloc(size_t bytes, NvrmMem *out);
void omega_gpu_session_free(NvrmMem *m); /* no-op when latched or closed; zeroes *m */

/* Caller holds the lock, device open. Makes s->mem at least `bytes` long
 * (reallocated when it grows; contents are not kept). */
int omega_gpu_session_scratch(OmegaGpuScratch *s, size_t bytes);
void omega_gpu_session_scratch_free(OmegaGpuScratch *s);

/* Caller holds the lock, device open, not latched. Submits one launch and waits
 * for both markers. 0 on success; -1 with the stage named on failure. A marker
 * timeout latches the process. elapsed_ns (submit to second marker) and the
 * observed first marker word are added / stored when the pointers are given. */
int omega_gpu_session_launch(const OmegaGpuLaunch *L, uint64_t *elapsed_ns, uint32_t *marker_out);

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
 * Refused (no-op) after an uncertain completion. The next open reopens. */
void omega_gpu_session_close(void);

/* Device opens so far in this process (diagnostics; a persistent session shows 1). */
uint32_t omega_gpu_session_open_count(void);

#ifdef __cplusplus
}
#endif
#endif
