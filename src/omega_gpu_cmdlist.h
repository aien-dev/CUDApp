/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
 *   (completion markers: the session marker page at 0x100, the one page the platform lets the host poll)
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

#define OMEGA_GPU_CMDLIST_MAX_LAUNCHES 4096u
#define OMEGA_GPU_CMDLIST_BANK_BYTES 0x400u
#define OMEGA_GPU_CMDLIST_QMD_BYTES 0x400u /* QMD1 at +0, QMD0 (INLINE mode) at +0x200 */

typedef enum { OMEGA_GPU_CMDLIST_PCAS = 0, OMEGA_GPU_CMDLIST_INLINE = 1 } OmegaGpuCmdListMode;
/* Barrier before every launch after the first of a submission (the first always gets the full WFI + invalidate ALL):
 *   WFI        WAIT_FOR_IDLE + INVALIDATE_SHADER_CACHES ALL (instruction, data, constant). The default since cut 1b.
 *   NONE       nothing (TEST ONLY: negative control; consecutive grids may overlap).
 *   WFI_ONLY   WAIT_FOR_IDLE, no cache invalidate (W-L2c launch-cost study: isolates the drain from the invalidate).
 *   WFI_DATA   WAIT_FOR_IDLE + INVALIDATE_SHADER_CACHES {data} (keeps the instruction and constant caches warm).
 *   FLUSH_DATA INVALIDATE_SHADER_CACHES {data, flush_data} alone, which NVK uses for a storage-write to storage-read
 *              dependency between dispatches ("This is also implicitly a WFI", nvk_cmd_buffer.c:633). */
typedef enum { OMEGA_GPU_CMDLIST_BARRIER_WFI = 0, OMEGA_GPU_CMDLIST_BARRIER_NONE = 1 /* TEST ONLY: negative control */,
               OMEGA_GPU_CMDLIST_BARRIER_WFI_ONLY = 2, OMEGA_GPU_CMDLIST_BARRIER_WFI_DATA = 3, OMEGA_GPU_CMDLIST_BARRIER_FLUSH_DATA = 4 } OmegaGpuCmdListBarrier;
const char *omega_gpu_cmdlist_barrier_name(OmegaGpuCmdListBarrier b);

typedef struct {
    OmegaGpuCmdListMode mode;
    OmegaGpuCmdListBarrier barrier;
    uint64_t timeout_ms;  /* per submission; 0 = 20000 */
    uint32_t spin_us;     /* marker spin window before the platform poll; 0 = the session's process-wide setting */
    int stamps;           /* 1: a GPU timestamp report before the first launch and after every launch (after its WAIT_FOR_IDLE), read
                             back by omega_gpu_cmdlist_stamps. Costs 6 words + the report per launch. 0 = off. */
    int skip_launch;      /* TEST ONLY: >= 1 drops launch (skip_launch - 1) from the method stream (its slot stays
                             recorded), so the per-launch semaphore check must fail. 0 = off. */
    int first_barrier_none; /* C4 streams: no WAIT_FOR_IDLE + invalidate before launch 0 (the CPU-written code, banks and QMDs
                             must not sit in a stale cache line: fresh slots or unchanged contents). A WAIT_FOR_IDLE may also
                             wait for grids of another queue of the group; the streams test measures both. 0 = the barrier. */
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
/* Caller holds the lock. Marks the most recently added launch as sharing NO data with the launch before it (neither reads
 * what the other writes): submit emits no barrier before it, so the two grids may overlap on the chip (W-L2d: a weight
 * prefetch beside the latency-bound attention). The launch after it gets the normal barrier, which waits for both.
 * Ignored for the first launch of a submission. 0 ok, -1 (no launch recorded). */
int omega_gpu_cmdlist_mark_independent(OmegaGpuCmdList *cl);
/* Caller holds the lock. Writes the launch's bank and QMD into the next slot. The launch's code must stay resident
 * until submit returns. Returns the launch index, or -1 (full, bad arguments, QMD refused). L->timeout_ms and
 * L->spin_us are ignored (the list has one completion). */
int omega_gpu_cmdlist_add(OmegaGpuCmdList *cl, const OmegaGpuLaunch *L);
/* Caller holds the lock, device open, not latched. Submits every recorded launch in order, waits for the completion,
 * checks every launch semaphore. 0 ok; -1 with the stage named (omega_gpu_session_last_error). */
int omega_gpu_cmdlist_submit(OmegaGpuCmdList *cl, const OmegaGpuCmdListOpts *opts, OmegaGpuCmdListStats *st);
/* L136 (roadblock R3): caller holds the lock. List a on the session queue and list b on extra queue 1
 * (omega_gpu_session_extra_queue; needs AIEN_OMEGA_GPU_CHANNELS >= 2 at device open), both submitted before either is
 * waited, so the two may run at the same time. The lists must share no written buffer. PCAS mode, no stamps, each list in
 * one submission. Both stats get the pair wall time (first submit to the last marker). 0 ok; -1 with the stage named;
 * a marker timeout latches the session. */
int omega_gpu_cmdlist_submit_pair(OmegaGpuCmdList *a, OmegaGpuCmdList *b, const OmegaGpuCmdListOpts *opts, OmegaGpuCmdListStats *sa, OmegaGpuCmdListStats *sb);
/* C4 events. Caller holds the lock. The next launch added to cl waits, on the chip, until launch `index` of list `other`
 * (already added there) has completed: a host semaphore ACQUIRE on that launch's release semaphore (DONE), emitted
 * before the launch's barrier. Meant for two lists on two queues (submit_pair); a wait on a list that never runs
 * stalls the queue until the marker timeout latches the session. 0 ok, -1 (no such launch). */
int omega_gpu_cmdlist_wait_for(OmegaGpuCmdList *cl, const OmegaGpuCmdList *other, uint32_t index);
/* 1 once a timeout latched this module, until omega_gpu_session_recover (C5). */
int omega_gpu_cmdlist_is_latched(void);

/* After a submit with opts.stamps = 1: out[0] = GPU timestamp (ns, PTIMER) before launch 0, out[i + 1] = after launch i completed
 * (taken after its WAIT_FOR_IDLE). Returns the count written (launches + 1) or -1 (stamps off, slot unwritten, cap too small). */
int omega_gpu_cmdlist_stamps(const OmegaGpuCmdList *cl, uint64_t *out, uint32_t cap);

#ifdef __cplusplus
}
#endif
#endif
