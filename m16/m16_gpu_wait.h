#ifndef M16_GPU_WAIT_H
#define M16_GPU_WAIT_H

/* Shared completion wait for words the GPU writes to host-visible memory.
 *
 * Pure C, no NVRM dependency, so the host unit test can drive it with fake
 * memory and a fake clock.
 *
 * Two wait kinds, never one ambiguous wait:
 *   m16_gpu_wait_fixed     the word must equal `expected` exactly. A value
 *                            that overshoots is NOT success.
 *   m16_gpu_wait_sequence  the word is a free-running u32 counter (for
 *                            example dispatch ids). Success when it has reached
 *                            or passed `target` in serial-number order, i.e.
 *                            (int32_t)(observed - target) >= 0, so it is
 *                            correct across the 0xffffffff -> 0 wrap. Valid
 *                            while the counter is less than 2^31 ahead.
 *
 * Two bounds, both on CLOCK_MONOTONIC in nanoseconds:
 *   total_timeout_ms     hard deadline from the start; never extended. A device
 *                        that advances occasionally still hits it (TIMEOUT).
 *   progress_timeout_ms  stall deadline; restarts only when a watched word
 *                        (word, marker2, snapshot words) changes (STALLED).
 *
 * Memory visibility: a barrier (dsb sy on AArch64, a full fence elsewhere) runs
 * before every read of the watched words and once more, after the success
 * observation, before returning PASS. After PASS, the CPU loads issued by the
 * caller are ordered after the marker load. That is ordering only. It does not
 * make a cached CPU mapping coherent with GPU writes and it does not flush the
 * GPU L2: the marker page must still be GPU-uncached and the producing work must
 * flush (WFI marker, L2_FLUSH_DIRTY mem-op, second WFI marker; see
 * m16/m16_concurrent.c). The timer does not fix visibility.
 *
 * Same contract as omega's src/omega_gpu_wait.h (omega 6fa4d97, PR omega#232).
 * Physics cannot include omega code, so this is the canonical copy; omega's
 * becomes a thin wrapper in a later cut. m16_native_wait_marker(_ge) is not
 * converted yet (follow-up).
 *
 * If cfg->marker2 is set, PASS also requires *marker2 == marker2_want (read
 * output only after both). */

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    M16_GPU_WAIT_FIXED = 1,
    M16_GPU_WAIT_SEQUENCE = 2
} m16_gpu_wait_kind_t;

typedef enum {
    M16_GPU_WAIT_PASS = 0,
    M16_GPU_WAIT_TIMEOUT = 1,       /* hard total deadline */
    M16_GPU_WAIT_STALLED = 2,       /* no progress for progress_timeout_ms */
    M16_GPU_WAIT_INVALID_STATE = 3, /* cfg->valid rejected an observed value */
    M16_GPU_WAIT_ABORTED = 4        /* cfg->abort asked to stop */
} m16_gpu_wait_result_t;

typedef struct {
    m16_gpu_wait_kind_t wait_kind;
    m16_gpu_wait_result_t result;
    uint32_t expected;       /* fixed value or sequence target */
    uint32_t last_observed;  /* last value read from the watched word */
    uint32_t last_marker2;   /* last value read from marker2 (0 when unused) */
    uint32_t snap[2];        /* last values from the snapshot callback */
    uint64_t start_ns;
    uint64_t elapsed_ns;
    uint64_t last_progress_ns; /* time of the last observed change (start if none) */
    uint64_t progress_count;   /* observed changes after the first read */
    uint64_t polls;
} m16_gpu_wait_report_t;

typedef struct {
    uint32_t total_timeout_ms;    /* 0 means the default, 600000 */
    uint32_t progress_timeout_ms; /* 0 means the default, 5000 */
    volatile uint32_t *marker2;
    uint32_t marker2_want;
    /* Optional: fill two extra words that count as progress when they change
     * (for example GP_PUT and a semaphore). Called once per poll. */
    void (*snapshot)(void *ctx, uint32_t out[2]);
    void *snapshot_ctx;
    /* Optional: return false to declare the observed word out of order. */
    bool (*valid)(void *ctx, uint32_t observed);
    /* Optional: return true to stop the wait (ABORTED). */
    bool (*abort)(void *ctx);
    /* Test hooks. NULL means the real ones (CLOCK_MONOTONIC, dsb, yield).
     * hook_ctx is passed to now_ns, barrier, relax, valid and abort. */
    uint64_t (*now_ns)(void *ctx);
    void (*barrier)(void *ctx);
    void (*relax)(void *ctx);
    void *hook_ctx;
} m16_gpu_wait_cfg_t;

#define M16_GPU_WAIT_DEFAULT_TOTAL_MS 600000u
#define M16_GPU_WAIT_DEFAULT_PROGRESS_MS 5000u

/* Both return true on PASS. rpt and cfg may be NULL (cfg NULL means defaults). */
bool m16_gpu_wait_fixed(volatile uint32_t *word, uint32_t expected,
                          m16_gpu_wait_report_t *rpt, const m16_gpu_wait_cfg_t *cfg);
bool m16_gpu_wait_sequence(volatile uint32_t *word, uint32_t target,
                             m16_gpu_wait_report_t *rpt, const m16_gpu_wait_cfg_t *cfg);

/* Failure class of a failed wait, for tagged test output (additive, not in omega's copy):
 *   "wrong_marker"  INVALID_STATE; or, for a FIXED wait, the word holds a nonzero
 *                   value other than the expected one; or marker2 holds a nonzero
 *                   value other than marker2_want. (A SEQUENCE word below its
 *                   target is progress, never wrong.) Independent of TIMEOUT vs STALLED.
 *   "late_progress" something moved (progress_count > 0, or the word already held
 *                   its value but marker2 never arrived) yet the wait did not finish
 *                   (hard TIMEOUT while still moving, or stalled after moving)
 *   "no_progress"   word and marker2 never left 0
 * Call only on a report whose result is not PASS. marker2_want is 0 when unused. */
const char *m16_gpu_wait_failure_class(const m16_gpu_wait_report_t *r, uint32_t marker2_want);

const char *m16_gpu_wait_result_name(m16_gpu_wait_result_t r);
const char *m16_gpu_wait_kind_name(m16_gpu_wait_kind_t k);

#endif
