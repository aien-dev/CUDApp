#ifndef OMEGA_GPU_CMDLIST_H
#define OMEGA_GPU_CMDLIST_H
/*
 * omega_gpu_cmdlist: many kernel launches in ONE submission (OWN-30B W-L2, decode launch collapse, 2026-10-13).
 *
 * Why: omega_gpu_session_launch submits one kernel per GPFIFO entry and polls two markers before it returns, so a
 * Qwen3-30B decode token (289 launches) pays the submit + completion round trip 289 times. A command list records N
 * launches, then submits them as one method stream with one completion (the C3 tail: marker, L2_FLUSH_DIRTY, second
 * marker on the uncached page) at the end.
 *
 * What each launch gets (all buffers from omega_gpu_session_alloc, GPU-uncached; nvos.h:1117 read 2026-10-13):
 *   - its own constant bank 0 (1 KB: the 224 driver words patched for this bank, arguments at 0x380), CPU-written;
 *   - its own QMD slot (the session's QMD1 words with this launch's bank and semaphore), CPU-written;
 *   - its own release semaphore word, set to INTERMEDIATE_INIT (5) by the CPU and released to DONE (6) by the QMD when
 *     the grid completes. submit() checks every word: a launch that never ran is caught by name (index).
 *
 * Ordering between launches (the race this module must not create): Mesa NVK, the working open driver for this class,
 * serialises dependent dispatches with WAIT_FOR_IDLE (NVCEC0 0x0110) or INVALIDATE_SHADER_CACHES ("This is also
 * implicitly a WFI", nvk_cmd_buffer.c:633); without one, consecutive grids of one channel may overlap. Every launch
 * here is preceded by WAIT_FOR_IDLE and the session's INVALIDATE_SHADER_CACHES_ALL (instruction, data, constant).
 * OMEGA_GPU_CMDLIST_BARRIER_NONE exists only as the negative control of the chip test.
 *
 * Launch methods (opts.mode):
 *   PCAS   QMD in memory, SEND_PCAS_A (qmd >> 8) + SEND_SIGNALING_PCAS2_B INVALIDATE_COPY_SCHEDULE (clcec0.h 0x02b4 /
 *          0x02c0; the NVK dispatch sequence, nvk_cmd_dispatch.c), no dependent QMD: 8 words per launch, a whole
 *          decode token fits one 64 KB ring.
 *   INLINE the session's own chip-proven words per launch (QMD0 null grid + QMD1 inline), bank from memory: ~200 words
 *          per launch, so the list is cut into several submissions, each waited before the next (the ring is copied to
 *          offset 0 on every submit).
 *
 * Threading: the caller holds omega_gpu_session_lock() around add / submit (one channel, as for every API).
 * Uncertain completion: a marker timeout latches this module and then forces the session latch (a 1 ms session
 * launch queued behind the stuck list times out and latches the process), so no API keeps launching on a channel
 * that may still be busy.
 */
#include "omega_gpu_session.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_GPU_CMDLIST_MAX_LAUNCHES 2048u
#define OMEGA_GPU_CMDLIST_BANK_BYTES 0x400u
#define OMEGA_GPU_CMDLIST_QMD_BYTES 0x400u /* QMD1 at +0, QMD0 (INLINE mode) at +0x200 */

typedef enum { OMEGA_GPU_CMDLIST_PCAS = 0, OMEGA_GPU_CMDLIST_INLINE = 1 } OmegaGpuCmdListMode;
typedef enum { OMEGA_GPU_CMDLIST_BARRIER_WFI = 0, OMEGA_GPU_CMDLIST_BARRIER_NONE = 1 /* TEST ONLY: negative control */ } OmegaGpuCmdListBarrier;

typedef struct {
    OmegaGpuCmdListMode mode;
    OmegaGpuCmdListBarrier barrier;
    uint64_t timeout_ms;  /* per submission; 0 = 20000 */
    uint32_t spin_us;     /* marker spin window before the platform poll; 0 = the session's process-wide setting */
    int skip_launch;      /* TEST ONLY: >= 1 drops launch (skip_launch - 1) from the method stream (its slot stays
                             recorded), so the per-launch semaphore check must fail. 0 = off. */
} OmegaGpuCmdListOpts;

typedef struct {
    uint32_t launches;     /* recorded launches submitted */
    uint32_t submits;      /* GPFIFO entries (1 in PCAS mode below the ring size) */
    uint32_t words;        /* method words over all submissions */
    uint64_t elapsed_ns;   /* first submit to last second marker */
    uint64_t host_ns;      /* building the method stream (CPU) */
    uint32_t sems_done;    /* launch semaphores found DONE */
    int32_t first_missing; /* index of the first launch whose semaphore was not DONE, -1 if none */
} OmegaGpuCmdListStats;

typedef struct OmegaGpuCmdList OmegaGpuCmdList;

/* Takes the session lock itself. Allocates the slots for up to cap launches. 0 ok, -1 (session error names it). */
int omega_gpu_cmdlist_create(uint32_t cap, OmegaGpuCmdList **out);
void omega_gpu_cmdlist_free(OmegaGpuCmdList *cl);
/* Caller holds the lock. Empties the list (slots are rewritten by the next adds). */
void omega_gpu_cmdlist_reset(OmegaGpuCmdList *cl);
uint32_t omega_gpu_cmdlist_count(const OmegaGpuCmdList *cl);
/* Caller holds the lock. Writes the launch's bank and QMD into the next slot. The launch's code must stay resident
 * until submit returns. Returns the launch index, or -1 (full, bad arguments, QMD refused). L->timeout_ms and
 * L->spin_us are ignored (the list has one completion). */
int omega_gpu_cmdlist_add(OmegaGpuCmdList *cl, const OmegaGpuLaunch *L);
/* Caller holds the lock, device open, not latched. Submits every recorded launch in order, waits for the completion,
 * checks every launch semaphore. 0 ok; -1 with the stage named (omega_gpu_session_last_error). */
int omega_gpu_cmdlist_submit(OmegaGpuCmdList *cl, const OmegaGpuCmdListOpts *opts, OmegaGpuCmdListStats *st);
/* 1 once a timeout latched this module. */
int omega_gpu_cmdlist_is_latched(void);

#ifdef __cplusplus
}
#endif
#endif
