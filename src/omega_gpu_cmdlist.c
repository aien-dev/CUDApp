/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * omega_gpu_cmdlist.c: many launches in one submission (see the header).
 *
 * The per-launch words reuse the session's chip-proven pieces (omega_gpu_session.c, read 2026-10-13): SETUP_WORDS,
 * INVALIDATE_SHADER_CACHES_ALL, the QMD0/QMD1 builders, the C3 tail (first marker with RELEASE_WFI, L2_FLUSH_DIRTY
 * mem-op, second marker on the uncached page) and the poll loop. What is new: banks and QMDs live in per-launch slots
 * written by the CPU, WAIT_FOR_IDLE sits between launches, and the PCAS launch method.
 */
#include "omega_gpu_cmdlist.h"
#include "omega_gpu_cmdlist_native.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_submit.h"
#include "omega_gpu_native_runtime.h"
#include <string.h>

#define PAGE 0x1000ull
#define RING_WORDS (0x10000u / 4u)         /* the session ring: 64 KB */
#define DEFAULT_TIMEOUT_MS 20000ull
#define MTHD_WAIT_FOR_IDLE 0x0110u         /* clcec0.h NVCEC0_WAIT_FOR_IDLE */
/* Completion markers live on the SESSION marker page (OmegaGpuSession.marker, allocated by the session with
 * OMEGA_GPU_RIGHT_HOST_POLLED, GPU-uncached): the platform refuses poll_word on any buffer without that right
 * (omega_gpu_platform_linux.c ref_poll_word, chip run cmdlist-r1 2026-10-13: "marker poll refused"), and the session
 * exposes no polled allocation. The session uses bytes 0..0x1f of the page; this module uses 0x100..0x11f. */
#define MARKER_OFF 0x100ull
#define MTHD_SEND_PCAS_A 0x02b4u           /* clcec0.h NVCEC0_SEND_PCAS_A, QMD_ADDRESS_SHIFTED8 */
#define MTHD_SEND_SIGNALING_PCAS2_B 0x02c0u /* clcec0.h, PCAS_ACTION 3:0 */
#define PCAS_ACTION_INVALIDATE_COPY_SCHEDULE 3u

static inline uint32_t bw_mthd(uint32_t subc, uint32_t mthd, uint32_t count) {
    return (2u << 28) | (count << 16) | (subc << 13) | (mthd >> 2);
}
/* Same words as omega_gpu_session.c / omega_blackwell_submit.c SETUP_WORDS (compute class setup). */
static const uint32_t SETUP_WORDS[18] = {
    0x20012061, 0x0000cec0, 0x20012092, 0x00000001, 0x200120a8, 0x0000000f, 0x2001255d, 0x00000003,
    0x2001255e, 0x20000000, 0x2001255f, 0x000fffff, 0x20012557, 0x00000003, 0x20012558, 0x22000000,
    0x20012559, 0x00000000,
};

typedef struct {
    OmegaGpuNativeCache *cache;
    uint64_t state[16];
    uint64_t code_va; /* successful add() intent, independently checked against actual QMD in .om */
    uint32_t n_args; /* exact successful add() fact, including incumbent entries */
    uint8_t intent_seal[32]; /* generated normalized bank/QMD binding at native add */
} NativeRef;
struct OmegaGpuCmdList {
    uint32_t cap, n;
    OmegaGpuMem banks, qmds, sems, scratch;
    OmegaGpuMem stamps;   /* (cap + 1) x 16 B: compute-class REPORT_SEMAPHORE four-word reports {u64 payload, u64 timestamp}
                             (mesa clcec0.h 0x0158..0x0168, nvk_query.h), slot 0 = submit start, slot i + 1 = launch i done */
    uint32_t *words;      /* host method stream */
    uint32_t words_cap;
    OmegaGpuMem nop_code; /* EXIT kernel used to force the session latch after a timeout */
    uint32_t nop_gpr;
    uint8_t *indep;       /* per launch: 1 = shares no data with the launch before it (mark_independent): no barrier before it */
    int stamped;          /* the last submit had opts.stamps (the accessor refuses stale stamps from an earlier submit) */
    uint64_t *wait_va;    /* per launch: semaphore word of another list's launch to wait for before it (wait_for), 0 = none */
    uint64_t pending_wait;/* wait_for recorded, taken by the next add */
    uint32_t nwaits;      /* launches with a wait (6 more words each) */
    const OmegaGpuCmdList *wait_other; /* the one list this list waits on (it must be the other list of the pair) */
    uint32_t wait_max;    /* highest launch index of wait_other waited on */
    uint32_t gen;         /* omega_gpu_session_recoveries() at create: a list from before a recovery is stale (C5) */
    uint64_t epoch;        /* normal close/reopen also invalidates device handles */
    uint64_t record_generation; /* advanced only by generated policy on safe reset */
    NativeRef *native;    /* per-launch exact code ownership, metadata only */
};

static uint64_t next_record_generation(uint64_t previous) {
    uint64_t next = 0;
    return omega_native_revision(&next, previous) == OMEGA_OK ? next : 0;
}

static uint32_t g_latched; /* 1 + omega_gpu_session_recoveries() at the latch; 0 never latched (C5: a recovery lifts it) */

int omega_gpu_cmdlist_is_latched(void) { return __atomic_load_n(&g_latched, __ATOMIC_ACQUIRE) == omega_gpu_session_recoveries() + 1u ? 1 : 0; }
uint32_t omega_gpu_cmdlist_count(const OmegaGpuCmdList *cl) { return cl ? cl->n : 0; }
static int native_event(NativeRef *ref, unsigned event) {
    if (!ref->cache) return 0;
    uint64_t result = 0;
    return omega_native_event(&result, ref->cache->pool, 528, ref->state, 16,
                              omega_gpu_session_epoch(), event) == OMEGA_OK && result == 0 ? 0 : -1;
}
static int native_events(OmegaGpuCmdList *cl, uint32_t first, uint32_t end, unsigned event) {
    for (uint32_t i = first; i < end; i++)
        if (native_event(&cl->native[i], event) != 0) return -1;
    return 0;
}
static int native_begin(OmegaGpuCmdList *cl, uint32_t first, uint32_t end) {
    uint32_t i;
    for (i = first; i < end; i++) if (native_event(&cl->native[i], 1) != 0) break;
    if (i == end) return 0;
    (void)native_events(cl, first, i, 4); /* none of this chunk was submitted */
    omega_gpu_session_set_error("cmdlist: native code lease refused before submit");
    return -1;
}
void omega_gpu_cmdlist_reset(OmegaGpuCmdList *cl) {
    if (!cl) return;
    if (native_events(cl, 0, cl->n, 5) != 0) {
        omega_gpu_session_set_error("cmdlist reset: uncertain native flight still owns list buffers"); return;
    }
    const uint64_t next = next_record_generation(cl->record_generation);
    if (!next) {
        omega_gpu_session_set_error("cmdlist reset: record generation exhausted"); return;
    }
    (void)native_events(cl, 0, cl->n, 3);
    memset(cl->native, 0, (size_t)cl->cap * sizeof *cl->native);
    cl->n = 0; cl->pending_wait = 0; cl->nwaits = 0; cl->wait_other = NULL; cl->wait_max = 0;
    cl->record_generation = next;
}
/* C5: every handle of a list made before the latest session recovery belongs to the closed device */
static int stale(const OmegaGpuCmdList *cl) {
    if (cl->gen == omega_gpu_session_recoveries() && cl->epoch == omega_gpu_session_epoch()) return 0;
    omega_gpu_session_set_error("cmdlist: made before a session recovery (its buffers are gone): create a new list");
    return 1;
}
int omega_gpu_cmdlist_owner(const OmegaGpuCmdList *cl, OmegaGpuCmdListOwner *out) {
    if (!cl || !out || stale(cl)) return -1;
    out->banks = cl->banks; out->qmds = cl->qmds; out->semaphores = cl->sems;
    out->scratch = cl->scratch; out->stamps = cl->stamps;
    out->record_generation = cl->record_generation;
    out->capacity = cl->cap; out->count = cl->n;
    return 0;
}
int omega_gpu_cmdlist_entry(const OmegaGpuCmdList *cl, uint32_t index, OmegaGpuCmdListEntry *out) {
    if (!cl || !out || index >= cl->n || stale(cl)) return -1;
    out->bank = (const uint8_t *)cl->banks.cpu + (size_t)index * OMEGA_GPU_CMDLIST_BANK_BYTES;
    out->qmd_slot = (const uint8_t *)cl->qmds.cpu + (size_t)index * OMEGA_GPU_CMDLIST_QMD_BYTES;
    out->intent_seal = cl->native[index].intent_seal;
    out->code_va = cl->native[index].code_va;
    out->pending_wait = cl->wait_va[index]; out->record_generation = cl->record_generation;
    out->ordinal = index; out->count = cl->n; out->n_args = cl->native[index].n_args;
    out->independent = cl->indep[index];
    return 0;
}
int omega_gpu_cmdlist_wait_for(OmegaGpuCmdList *cl, const OmegaGpuCmdList *other, uint32_t index) {
    if (!cl || !other || cl == other || index >= other->n) { omega_gpu_session_set_error("cmdlist wait_for: another list and one of its launches"); return -1; }
    if (stale(cl) || stale(other)) return -1;
    if (cl->wait_other && cl->wait_other != other) { omega_gpu_session_set_error("cmdlist wait_for: a list waits on one other list only"); return -1; }
    cl->wait_other = other;
    if (index > cl->wait_max) cl->wait_max = index;
    cl->pending_wait = other->sems.va + (uint64_t)index * 16u;
    return 0;
}
int omega_gpu_cmdlist_mark_independent(OmegaGpuCmdList *cl) {
    if (!cl || cl->n == 0) { omega_gpu_session_set_error("cmdlist mark_independent: no launch recorded"); return -1; }
    cl->indep[cl->n - 1] = 1;
    return 0;
}

static int build_nop(OmegaGpuCmdList *cl) {
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) return -1;
    omega_bw_ir_init(prog);
    BlackwellIRInsn ex; memset(&ex, 0, sizeof ex);
    ex.op = BW_IR_EXIT; ex.dst_vreg = ex.src1_vreg = ex.src2_vreg = ex.src3_vreg = ex.ureg = -1;
    ex.control = (5u << 9) | (7u << 14) | (7u << 17);
    BlackwellIRInsn self = ex; self.op = BW_IR_BRA; self.control = 0x000fc000u;
    int rc = (omega_bw_ir_append(prog, &ex) < 0 || omega_bw_ir_append(prog, &self) < 0 || omega_bw_regalloc_solve(prog) != 0) ? -1 : 0;
    size_t need = rc == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (need == 0 || omega_gpu_session_alloc_code(need, &cl->nop_code) != 0) { omega_gpu_host_free(prog); return -1; }
    OmegaBlackwellKernel k;
    rc = omega_blackwell_kernel_encode_into(prog, cl->nop_code.cpu, need, &k);
    omega_gpu_host_free(prog);
    if (rc != 0 || omega_gpu_session_sync_to_gpu(&cl->nop_code) != 0) return -1;
    cl->nop_gpr = omega_bw_gpr_declared(k.gpr_count);
    if (cl->nop_gpr < 64u) cl->nop_gpr = 64u;
    return 0;
}

int omega_gpu_cmdlist_create(uint32_t cap, OmegaGpuCmdList **out) {
    if (!out || cap == 0 || cap > OMEGA_GPU_CMDLIST_MAX_LAUNCHES) { omega_gpu_session_set_error("cmdlist create: arguments"); return -1; }
    *out = NULL;
    OmegaGpuCmdList *cl = omega_gpu_host_calloc(1, sizeof *cl);
    if (!cl) return -1;
    cl->cap = cap;
    cl->gen = omega_gpu_session_recoveries();
    cl->epoch = omega_gpu_session_epoch();
    cl->record_generation = next_record_generation(0);
    if (!cl->record_generation) { omega_gpu_host_free(cl); return -1; }
    cl->words_cap = RING_WORDS;
    cl->words = omega_gpu_host_alloc((size_t)cl->words_cap * 4u);
    cl->indep = omega_gpu_host_calloc(cap, 1);
    cl->wait_va = omega_gpu_host_calloc(cap, sizeof *cl->wait_va);
    cl->native = omega_gpu_host_calloc(cap, sizeof *cl->native);
    int rc = cl->words && cl->indep && cl->wait_va && cl->native ? 0 : -1;
    omega_gpu_session_lock();
    if (rc == 0 && !omega_gpu_session_open()) rc = -1;
    if (rc == 0 && (omega_gpu_session_alloc((size_t)cap * OMEGA_GPU_CMDLIST_BANK_BYTES, &cl->banks) != 0 ||
                    omega_gpu_session_alloc((size_t)cap * OMEGA_GPU_CMDLIST_QMD_BYTES, &cl->qmds) != 0 ||
                    omega_gpu_session_alloc((size_t)cap * 16u, &cl->sems) != 0 ||
                    omega_gpu_session_alloc((size_t)(cap + 1u) * 16u, &cl->stamps) != 0 ||
                    omega_gpu_session_alloc(0x4000, &cl->scratch) != 0))
        rc = -1;
    if (rc == 0 && build_nop(cl) != 0) { omega_gpu_session_set_error("cmdlist create: nop kernel"); rc = -1; }
    if (rc == 0) {
        memset(cl->scratch.cpu, 0, 0x4000);
        if (omega_gpu_session_sync_to_gpu(&cl->scratch) != 0) rc = -1;
    }
    omega_gpu_session_unlock();
    if (rc != 0) { omega_gpu_cmdlist_free(cl); return -1; }
    *out = cl;
    return 0;
}

void omega_gpu_cmdlist_free(OmegaGpuCmdList *cl) {
    if (!cl) return;
    omega_gpu_session_lock();
    if (cl->native) (void)native_events(cl, 0, cl->n, 3); /* timeout retains pool flight refs until device teardown */
    OmegaGpuMem *m[] = { &cl->banks, &cl->qmds, &cl->sems, &cl->stamps, &cl->scratch, &cl->nop_code };
    for (unsigned i = 0; i < sizeof m / sizeof m[0]; i++) if (m[i]->cpu) omega_gpu_session_free(m[i]);
    omega_gpu_session_unlock();
    omega_gpu_host_free(cl->words);
    omega_gpu_host_free(cl->indep);
    omega_gpu_host_free(cl->wait_va);
    omega_gpu_host_free(cl->native);
    omega_gpu_host_free(cl);
}

static uint32_t round8(uint32_t v) { return (v + 7u) / 8u * 8u; }

static int add_recorded(OmegaGpuCmdList *cl, const OmegaGpuLaunch *L, NativeRef *pending) {
    if (!cl || !L || !L->args || L->n_args == 0 || L->n_args > OMEGA_GPU_SESSION_MAX_ARGS || L->threads_x == 0 || L->grid_x == 0 || L->grid_y == 0 ||
        L->code_va == 0) { omega_gpu_session_set_error("cmdlist add: arguments"); return -1; }
    if (stale(cl)) return -1;
    if (cl->n >= cl->cap) { omega_gpu_session_set_error("cmdlist add: list full"); return -1; }
    const uint32_t i = cl->n;
    const uint32_t ty = L->threads_y ? L->threads_y : 1;
    const uint64_t bank_va = cl->banks.va + (uint64_t)i * OMEGA_GPU_CMDLIST_BANK_BYTES;
    uint8_t *bank = (uint8_t *)cl->banks.cpu + (size_t)i * OMEGA_GPU_CMDLIST_BANK_BYTES;
    uint32_t drv[OMEGA_BW_CBANK_DRIVER_WORDS];
    omega_blackwell_build_cbank_driver_2d(drv, bank_va, L->threads_x, ty, L->grid_x, L->grid_y);
    memcpy(bank, drv, sizeof drv);
    memset(bank + sizeof drv, 0, 0x380 - sizeof drv);
    memcpy(bank + 0x380, L->args, (size_t)L->n_args * 4u);

    const uint64_t q1_va = cl->qmds.va + (uint64_t)i * OMEGA_GPU_CMDLIST_QMD_BYTES, q0_va = q1_va + 0x200;
    const uint64_t sem_va = cl->sems.va + (uint64_t)i * 16u;
    uint32_t gpr = L->gpr_count > 64u ? round8(L->gpr_count) : 64u;
    OmegaBlackwellQmdConfig cfg = {
        .code_va = L->code_va, .cbank_va = bank_va, .scratch_va = cl->scratch.va, .sem_va = sem_va,
        .qmd0_va = q0_va, .qmd1_va = q1_va, .num_elements = (uint32_t)L->num_elements,
        .threads_per_block = L->threads_x, .grid_width = L->grid_x * L->grid_y, .threads_x = L->threads_x, .threads_y = ty,
        .grid_x = L->grid_x, .grid_y = L->grid_y, .gpr_count = gpr, .shared_bytes = L->shared_bytes,
    };
    uint32_t q1[OMEGA_BW_QMD_WORDS], q0[OMEGA_BW_QMD_WORDS];
    if (omega_blackwell_build_qmd1(q1, &cfg) != 0 || omega_blackwell_verify_qmd_invariants(q1) != 0) { omega_gpu_session_set_error("cmdlist add: qmd"); return -1; }
    omega_blackwell_build_qmd0(q0, q0_va, q1_va);
    uint8_t *qs = (uint8_t *)cl->qmds.cpu + (size_t)i * OMEGA_GPU_CMDLIST_QMD_BYTES;
    memcpy(qs, q1, sizeof q1);
    memcpy(qs + 0x200, q0, sizeof q0);
    if (pending) {
        uint64_t result = 0;
        if (omega_native_mcseal(&result, bank, OMEGA_GPU_CMDLIST_BANK_BYTES,
            qs, OMEGA_GPU_CMDLIST_QMD_BYTES, L->n_args, pending->intent_seal, 32) != OMEGA_OK ||
            result != 0) return -1;
        pending->n_args = L->n_args;
        pending->code_va = L->code_va;
        cl->native[i] = *pending;
    } else {
        cl->native[i].n_args = L->n_args;
        cl->native[i].code_va = L->code_va;
    }
    cl->indep[cl->n] = 0;
    cl->wait_va[cl->n] = cl->pending_wait;
    if (cl->pending_wait) cl->nwaits++;
    cl->pending_wait = 0;
    cl->n++;
    return (int)i;
}
int omega_gpu_cmdlist_add(OmegaGpuCmdList *cl, const OmegaGpuLaunch *L) {
    return add_recorded(cl, L, NULL);
}

int omega_gpu_cmdlist_add_native(OmegaGpuCmdList *cl, const OmegaGpuLaunch *L, const OmegaGpuNativeLease *lease) {
    if (!cl || !lease || !lease->cache || cl->n >= cl->cap || stale(cl)) return -1;
    NativeRef pending; memset(&pending, 0, sizeof pending);
    pending.cache = lease->cache;
    pending.state[0] = 1; pending.state[1] = lease->slot;
    pending.state[2] = lease->revision; pending.state[3] = lease->epoch;
    if (native_event(&pending, 0) != 0) return -1;
    int index = add_recorded(cl, L, &pending);
    if (index < 0) { (void)native_event(&pending, 3); return -1; }
    return index;
}

#define WAIT_OK 0
#define WAIT_TIMED_OUT 2
#define WAIT_REFUSED 3

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

static int wait_word(const OmegaGpuPlatform *p, const OmegaGpuMem *marker, uint64_t off, uint32_t want, uint64_t timeout_ms, uint32_t spin_us) {
    off += MARKER_OFF;
    volatile uint32_t *m = (volatile uint32_t *)((uint8_t *)marker->cpu + off);
    if (spin_us) {
        const uint64_t end = p->time_now_ns(p->ctx) + (uint64_t)spin_us * 1000u;
        do { if ((int32_t)(*m - want) >= 0) return WAIT_OK; cpu_relax(); } while (p->time_now_ns(p->ctx) < end);
    }
    uint64_t waited = 0;
    for (;;) {
        uint32_t seen = 0;
        int rc = p->poll_word(p->ctx, marker->handle, off, want, 1, &seen);
        if (rc == OMEGA_GPU_PLAT_OK) return WAIT_OK;
        if (rc != OMEGA_GPU_POLL_TIMED_OUT) return WAIT_REFUSED;
        if (++waited >= timeout_ms) return WAIT_TIMED_OUT;
    }
}

/* Uncertain completion: latch this module, then hand the session a 1 ms launch. The stuck list is unretired, so the
 * platform refuses that submission REFUSED_BUSY before it writes the ring or rings a doorbell (caps.max_inflight 1,
 * CudaAuditFable Q6), and the session latches the whole process (parks every buffer, refuses every API). The launch
 * only writes session-owned cbank, QMD and marker bytes 0..0x1f, none of which the list uses. */
static void force_latch(OmegaGpuCmdList *cl, const char *why) {
    __atomic_store_n(&g_latched, omega_gpu_session_recoveries() + 1u, __ATOMIC_RELEASE);
    uint32_t arg = 0;
    OmegaGpuLaunch L = { .code_va = cl->nop_code.va, .gpr_count = cl->nop_gpr, .threads_x = 32, .threads_y = 1, .grid_x = 1, .grid_y = 1,
                         .num_elements = 1, .args = &arg, .n_args = 1, .timeout_ms = 1 };
    (void)omega_gpu_session_launch(&L, NULL, NULL);
    omega_gpu_session_note_module_latch(); /* C5: so omega_gpu_session_recover lifts it even when the check launch completed */
    omega_gpu_session_set_error(why);
}

static size_t emit_tail(uint32_t *pb, size_t n, uint64_t marker_va) {
    pb[n++] = bw_mthd(0, 0x005c, 5);
    pb[n++] = (uint32_t)marker_va; pb[n++] = (uint32_t)(marker_va >> 32);
    pb[n++] = OMEGA_BW_MARKER_COMPLETION_PAYLOAD; pb[n++] = 0; pb[n++] = 0x1 | (1u << 20);
    pb[n++] = bw_mthd(0, 0x0028, 4); pb[n++] = 0; pb[n++] = 0; pb[n++] = 0; pb[n++] = (0x10u << 27);
    pb[n++] = bw_mthd(0, 0x005c, 5);
    pb[n++] = (uint32_t)(marker_va + 0x10); pb[n++] = (uint32_t)((marker_va + 0x10) >> 32);
    pb[n++] = OMEGA_BW_MARKER2_PAYLOAD; pb[n++] = 0; pb[n++] = 0x1 | (1u << 20);
    return n;
}
#define TAIL_WORDS 17u

/* Compute-class four-word report {u64 payload = slot, u64 GPU timestamp} to stamps slot `slot` (mesa clcec0.h: PAYLOAD_LOWER 0x0158,
 * PAYLOAD_UPPER 0x015c, ADDRESS_LOWER 0x0160, ADDRESS_UPPER 0x0164, EXECUTE 0x0168 = OPERATION_RELEASE 0 | STRUCTURE_SIZE_FOUR_WORDS 0).
 * Ordered after the preceding WAIT_FOR_IDLE, like nvk_query_pool.c timestamp queries. */
#define MTHD_REPORT_SEMAPHORE_PAYLOAD_LOWER 0x0158u
#define STAMP_WORDS 6u
/* INVALIDATE_SHADER_CACHES (NVCEC0 0x021c) fields, clc6c0.h:222-236 (same layout on Blackwell): INSTRUCTION 0:0, FLUSH_DATA 2:2, DATA 4:4, CONSTANT 12:12 */
#define INVALIDATE_DATA (1u << 4)
#define INVALIDATE_FLUSH_DATA (1u << 2)
static size_t emit_stamp(uint32_t *pb, size_t n, uint64_t stamps_va, uint32_t slot) {
    const uint64_t va = stamps_va + (uint64_t)slot * 16u;
    pb[n++] = bw_mthd(1, MTHD_REPORT_SEMAPHORE_PAYLOAD_LOWER, 5);
    pb[n++] = slot; pb[n++] = 0; pb[n++] = (uint32_t)va; pb[n++] = (uint32_t)(va >> 32); pb[n++] = 0;
    return n;
}

static size_t emit_inline_qmd(uint32_t *pb, size_t n, uint64_t va, const uint32_t *words) {
    pb[n++] = (98u << 16) | (1u << 13) | (0x0318 >> 2) | (2u << 28);
    pb[n++] = (1u << 30) | (uint32_t)((va >> 40) & 0x1ff); pb[n++] = (uint32_t)(va >> 8);
    memcpy(&pb[n], words, 96 * 4); return n + 96;
}

/* Finalize ALL retained slots before observation, synchronization or any queue
 * commit. INLINE consumes the same bytes, including after PCAS reuse. */
static void prepare_qmd_mode(OmegaGpuCmdList *cl, const OmegaGpuCmdListOpts *op) {
    for (uint32_t i = 0; i < cl->n; i++) {
        uint32_t *q1 = (uint32_t *)((uint8_t *)cl->qmds.cpu + (size_t)i * OMEGA_GPU_CMDLIST_QMD_BYTES);
        const uint64_t q1_va = cl->qmds.va + (uint64_t)i * OMEGA_GPU_CMDLIST_QMD_BYTES;
        q1[10] = op->mode == OMEGA_GPU_CMDLIST_INLINE ? 0x00190000u : 0;
        q1[12] = op->mode == OMEGA_GPU_CMDLIST_INLINE ? (uint32_t)((q1_va + 0x200) >> 8) : 0;
    }
}
/* One submission's method stream for launches [done,end), followed by completion. */
static size_t emit_chunk(OmegaGpuCmdList *cl, const OmegaGpuCmdListOpts *op, uint32_t done, uint32_t end, uint64_t marker_va) {
    const OmegaGpuCmdListOpts o = *op;
    uint32_t *pb = cl->words;
    size_t n = 0;
    memcpy(pb, SETUP_WORDS, sizeof SETUP_WORDS); n += 18;
    for (uint32_t i = done; i < end; i++) {
        if (cl->wait_va[i]) {
            /* wait_for: host semaphore ACQUIRE (clc96f.h SEM_ADDR_LO 0x5c .. SEM_EXECUTE 0x6c, OPERATION_ACQUIRE 0, 32-bit
             * payload; the omega-mv#580 ORDER_SEM words, chip-proven): this queue stops here until the other list's launch
             * released DONE. Before the barrier, so the launch's cache invalidate comes after the wait. */
            const uint64_t va = cl->wait_va[i];
            pb[n++] = bw_mthd(0, 0x005c, 5); pb[n++] = (uint32_t)va; pb[n++] = (uint32_t)(va >> 32);
            pb[n++] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE; pb[n++] = 0; pb[n++] = 0;
        }
        if (o.skip_launch && i == (uint32_t)(o.skip_launch - 1)) continue;
        /* the first launch of a submission gets the full barrier (CPU-written QMDs, code and banks before it), unless
         * opts.first_barrier_none (C4 streams measurement: a WAIT_FOR_IDLE may also wait for another queue's grids) */
        const int indep = i != done && cl->indep[i]; /* marked independent: no barrier, overlaps the launch before it */
        /* a launch that waited (wait_for) always gets the full barrier: its invalidate must come after the wait */
        const OmegaGpuCmdListBarrier b = cl->wait_va[i] ? OMEGA_GPU_CMDLIST_BARRIER_WFI
                                       : i == done ? (i == 0 && o.first_barrier_none ? OMEGA_GPU_CMDLIST_BARRIER_NONE : OMEGA_GPU_CMDLIST_BARRIER_WFI)
                                       : indep ? OMEGA_GPU_CMDLIST_BARRIER_NONE : o.barrier;
        if (indep && o.stamps) n = emit_stamp(pb, n, cl->stamps.va, i); /* no WFI: slot i is the front-end time, launch i - 1 may still run */
        if (b != OMEGA_GPU_CMDLIST_BARRIER_NONE) {
            if (b != OMEGA_GPU_CMDLIST_BARRIER_FLUSH_DATA) { pb[n++] = bw_mthd(1, MTHD_WAIT_FOR_IDLE, 1); pb[n++] = 0; }
            else { pb[n++] = bw_mthd(1, OMEGA_BW_MTHD_INVALIDATE_SHADER_CACHES, 1); pb[n++] = INVALIDATE_DATA | INVALIDATE_FLUSH_DATA; } /* implicit WFI (NVK) */
            if (o.stamps) n = emit_stamp(pb, n, cl->stamps.va, i); /* slot i: launch i - 1 done (slot `done`: submit start) */
            if (b == OMEGA_GPU_CMDLIST_BARRIER_WFI) { pb[n++] = bw_mthd(1, OMEGA_BW_MTHD_INVALIDATE_SHADER_CACHES, 1); pb[n++] = OMEGA_BW_INVALIDATE_SHADER_CACHES_ALL; }
            else if (b == OMEGA_GPU_CMDLIST_BARRIER_WFI_DATA) { pb[n++] = bw_mthd(1, OMEGA_BW_MTHD_INVALIDATE_SHADER_CACHES, 1); pb[n++] = INVALIDATE_DATA; }
        }
        const uint64_t q1_va = cl->qmds.va + (uint64_t)i * OMEGA_GPU_CMDLIST_QMD_BYTES;
        if (o.mode == OMEGA_GPU_CMDLIST_INLINE) {
            const uint8_t *qs = (const uint8_t *)cl->qmds.cpu + (size_t)i * OMEGA_GPU_CMDLIST_QMD_BYTES;
            n = emit_inline_qmd(pb, n, q1_va + 0x200, (const uint32_t *)(qs + 0x200));
            n = emit_inline_qmd(pb, n, q1_va, (const uint32_t *)qs);
        } else {
            pb[n++] = bw_mthd(1, MTHD_SEND_PCAS_A, 1); pb[n++] = (uint32_t)(q1_va >> 8);
            pb[n++] = bw_mthd(1, MTHD_SEND_SIGNALING_PCAS2_B, 1); pb[n++] = PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
        }
    }
    if (o.stamps) {
        pb[n++] = bw_mthd(1, MTHD_WAIT_FOR_IDLE, 1); pb[n++] = 0;
        n = emit_stamp(pb, n, cl->stamps.va, end); /* slot end: the last launch done */
    }
    n = emit_tail(pb, n, marker_va);
    return n;
}

static int submit_observed(OmegaGpuCmdList *cl, const OmegaGpuCmdListOpts *opts, OmegaGpuCmdListStats *st,
                           OmegaGpuCmdListObserver observer, void *context) {
    OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o);
    if (opts) o = *opts;
    if (st) { memset(st, 0, sizeof *st); st->first_missing = -1; }
    const OmegaGpuPlatform *p = omega_gpu_session_platform();
    if (!cl || cl->n == 0 || !p) { omega_gpu_session_set_error("cmdlist submit: empty list or no platform"); return -1; }
    if (stale(cl)) return -1;
    if (cl->nwaits) { omega_gpu_session_set_error("cmdlist submit: a list with waits (wait_for) runs only in a pair (submit_pair)"); return -1; }
    if (omega_gpu_cmdlist_is_latched() || omega_gpu_session_is_blocked()) { omega_gpu_session_set_error("cmdlist submit: latched"); return -1; }
    OmegaGpuSession *s = omega_gpu_session_open();
    if (!s) return -1;
    if (!s->marker.cpu || s->marker.size < MARKER_OFF + 0x20) { omega_gpu_session_set_error("cmdlist submit: session marker page"); return -1; }
    const OmegaGpuMem *marker = &s->marker;
    const uint64_t marker_va = marker->va + MARKER_OFF;
    const uint64_t timeout = o.timeout_ms ? o.timeout_ms : DEFAULT_TIMEOUT_MS;
    const uint32_t spin = o.spin_us ? o.spin_us : omega_gpu_session_spin_us();
    prepare_qmd_mode(cl, &o);
    if (observer) {
        const OmegaGpuCmdListConsumed consumed = { .options = o, .timeout_ms = timeout, .spin_us = spin };
        if (observer(cl, &consumed, context) != 0) {
            omega_gpu_session_set_error("cmdlist submit: actual native closure refused before queue commit"); return -1;
        }
    }
    const uint64_t th0 = p->time_now_ns(p->ctx);

    /* semaphores INIT, markers 0, everything the CPU wrote made visible */
    uint32_t *sem = cl->sems.cpu;
    for (uint32_t i = 0; i < cl->n; i++) sem[4 * i] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT;
    volatile uint32_t *mk = (volatile uint32_t *)((uint8_t *)marker->cpu + MARKER_OFF);
    mk[0] = 0; mk[4] = 0;
    if (p->sync_cpu_to_gpu(p->ctx, cl->banks.handle, 0, (uint64_t)cl->n * OMEGA_GPU_CMDLIST_BANK_BYTES) != OMEGA_GPU_PLAT_OK ||
        p->sync_cpu_to_gpu(p->ctx, cl->qmds.handle, 0, (uint64_t)cl->n * OMEGA_GPU_CMDLIST_QMD_BYTES) != OMEGA_GPU_PLAT_OK ||
        p->sync_cpu_to_gpu(p->ctx, cl->sems.handle, 0, (uint64_t)cl->n * 16u) != OMEGA_GPU_PLAT_OK ||
        p->sync_cpu_to_gpu(p->ctx, marker->handle, MARKER_OFF, 0x20) != OMEGA_GPU_PLAT_OK) {
        omega_gpu_session_set_error("cmdlist submit: sync_cpu_to_gpu"); return -1;
    }

    const uint32_t per_launch = (o.mode == OMEGA_GPU_CMDLIST_INLINE ? 4u + 2u * 99u : 8u) + (o.stamps ? STAMP_WORDS : 0u) + (cl->nwaits ? 6u : 0u);
    /* The platform may put its own words ahead of ours in the ring (the Linux reference: the omega #457 L2 invalidate
     * after a host write to a GPU-cached buffer). Budget what it publishes, or a near-full submission is refused. */
    const uint32_t prefix = omega_gpu_session_submit_prefix_words();
    const uint32_t avail = cl->words_cap - 18u - TAIL_WORDS - 2u - STAMP_WORDS;
    if (prefix > avail || avail - prefix < per_launch) { omega_gpu_session_set_error("cmdlist submit: the platform submit prefix leaves no room for a launch"); return -1; }
    uint64_t *stamp_cpu = cl->stamps.cpu;
    cl->stamped = o.stamps ? 1 : 0;
    if (o.stamps) {
        memset(stamp_cpu, 0xff, (size_t)(cl->n + 1u) * 16u); /* poisoned: a slot the chip never wrote reads all-ones */
        if (p->sync_cpu_to_gpu(p->ctx, cl->stamps.handle, 0, (uint64_t)(cl->n + 1u) * 16u) != OMEGA_GPU_PLAT_OK) {
            omega_gpu_session_set_error("cmdlist submit: sync stamps"); return -1;
        }
    }
    const uint32_t room = (avail - prefix) / per_launch;
    uint64_t elapsed = 0, host = p->time_now_ns(p->ctx) - th0;
    uint32_t done = 0, submits = 0, words = 0;
    while (done < cl->n) {
        const uint64_t tb = p->time_now_ns(p->ctx);
        const uint32_t end = (cl->n - done > room) ? done + room : cl->n;
        uint32_t *pb = cl->words;
        size_t n = emit_chunk(cl, &o, done, end, marker_va);
        mk[0] = 0; mk[4] = 0;
        (void)p->sync_cpu_to_gpu(p->ctx, marker->handle, MARKER_OFF, 0x20);
        const uint64_t t0 = p->time_now_ns(p->ctx);
        host += t0 - tb;
        if (native_begin(cl, done, end) != 0) return -1;
        const int qrc = p->queue_submit(p->ctx, s->queue, pb, (uint32_t)n);
        if (qrc == OMEGA_GPU_PLAT_REFUSED_BUSY) { force_latch(cl, "cmdlist: an earlier submission is still in flight (latched)"); return -1; }
        if (qrc != OMEGA_GPU_PLAT_OK) {
            (void)native_events(cl, done, end, 4);
            omega_gpu_session_set_error("cmdlist submit: queue_submit refused"); return -1; /* refused: nothing in flight */
        }
        int w = wait_word(p, marker, 0, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, timeout, spin);
        if (w == WAIT_OK) w = wait_word(p, marker, 0x10, OMEGA_BW_MARKER2_PAYLOAD, timeout, spin);
        if (w != WAIT_OK) { force_latch(cl, w == WAIT_TIMED_OUT ? "cmdlist: marker wait timed out (latched)" : "cmdlist: marker poll refused (latched)"); return -1; }
        (void)p->sync_gpu_to_cpu(p->ctx, marker->handle, MARKER_OFF, 0x20);
        elapsed += p->time_now_ns(p->ctx) - t0;
        (void)p->queue_retire(p->ctx, s->queue);
        (void)native_events(cl, done, end, 2); /* both authoritative completion markers observed */
        submits++; words += (uint32_t)n; done = end;
    }
    (void)p->sync_gpu_to_cpu(p->ctx, cl->sems.handle, 0, (uint64_t)cl->n * 16u);
    if (o.stamps) (void)p->sync_gpu_to_cpu(p->ctx, cl->stamps.handle, 0, (uint64_t)(cl->n + 1u) * 16u);
    uint32_t ok = 0; int32_t missing = -1;
    for (uint32_t i = 0; i < cl->n; i++) {
        if (sem[4 * i] == OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE) ok++;
        else if (missing < 0) missing = (int32_t)i;
    }
    if (st) { st->launches = cl->n; st->submits = submits; st->words = words; st->elapsed_ns = elapsed; st->host_ns = host; st->sems_done = ok; st->first_missing = missing; }
    if (missing >= 0) { omega_gpu_session_set_error("cmdlist: a launch semaphore is not DONE (launch did not run)"); return -1; }
    return 0;
}
int omega_gpu_cmdlist_submit(OmegaGpuCmdList *cl, const OmegaGpuCmdListOpts *opts, OmegaGpuCmdListStats *st) {
    return submit_observed(cl, opts, st, NULL, NULL);
}
int omega_gpu_cmdlist_submit_observed(OmegaGpuCmdList *cl, const OmegaGpuCmdListOpts *opts, OmegaGpuCmdListStats *st,
                                     OmegaGpuCmdListObserver observer, void *context) {
    if (!observer) return -1;
    return submit_observed(cl, opts, st, observer, context);
}

/* L136: list a on the session queue and list b on extra queue 1, both in flight before either is waited. Each list gets
 * the same prefix as submit (semaphores INIT, banks / QMDs / semaphores synced) and the same words (emit_chunk); b's
 * completion markers sit at MARKER_OFF + PAIR_B_MARKER on the same uncached page. */
#define PAIR_B_MARKER 0x80ull
static int pair_prep(const OmegaGpuPlatform *p, OmegaGpuCmdList *cl) {
    uint32_t *sem = cl->sems.cpu;
    for (uint32_t i = 0; i < cl->n; i++) sem[4 * i] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT;
    cl->stamped = 0;
    return p->sync_cpu_to_gpu(p->ctx, cl->banks.handle, 0, (uint64_t)cl->n * OMEGA_GPU_CMDLIST_BANK_BYTES) == OMEGA_GPU_PLAT_OK &&
           p->sync_cpu_to_gpu(p->ctx, cl->sems.handle, 0, (uint64_t)cl->n * 16u) == OMEGA_GPU_PLAT_OK ? 0 : -1;
}

static int pair_check(const OmegaGpuPlatform *p, OmegaGpuCmdList *cl, uint32_t nwords, uint64_t elapsed, uint64_t host, OmegaGpuCmdListStats *st) {
    (void)p->sync_gpu_to_cpu(p->ctx, cl->sems.handle, 0, (uint64_t)cl->n * 16u);
    const uint32_t *sem = cl->sems.cpu;
    uint32_t ok = 0; int32_t missing = -1;
    for (uint32_t i = 0; i < cl->n; i++) {
        if (sem[4 * i] == OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE) ok++;
        else if (missing < 0) missing = (int32_t)i;
    }
    if (st) { st->launches = cl->n; st->submits = 1; st->words = nwords; st->elapsed_ns = elapsed; st->host_ns = host; st->sems_done = ok; st->first_missing = missing; }
    if (missing >= 0) { omega_gpu_session_set_error("cmdlist pair: a launch semaphore is not DONE (launch did not run)"); return -1; }
    return 0;
}

/* Uncertain completion of a pair: queue 1 may still run, so a nop on queue 0 proves nothing; latch the session directly. */
static void pair_latch(const char *why) {
    __atomic_store_n(&g_latched, omega_gpu_session_recoveries() + 1u, __ATOMIC_RELEASE);
    omega_gpu_session_latch_uncertain(why);
}

int omega_gpu_cmdlist_submit_pair(OmegaGpuCmdList *a, OmegaGpuCmdList *b, const OmegaGpuCmdListOpts *opts, OmegaGpuCmdListStats *sa, OmegaGpuCmdListStats *sb) {
    OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o);
    if (opts) o = *opts;
    if (sa) { memset(sa, 0, sizeof *sa); sa->first_missing = -1; }
    if (sb) { memset(sb, 0, sizeof *sb); sb->first_missing = -1; }
    const OmegaGpuPlatform *p = omega_gpu_session_platform();
    if (!a || !b || a == b || a->n == 0 || b->n == 0 || !p) { omega_gpu_session_set_error("cmdlist pair: two distinct non-empty lists and a platform"); return -1; }
    if (stale(a) || stale(b)) return -1;
    for (int k = 0; k < 2; k++) { /* a wait must point into the other list of this pair, at a launch it still has */
        const OmegaGpuCmdList *me = k ? b : a, *peer = k ? a : b;
        if (me->nwaits && (me->wait_other != peer || me->wait_max >= peer->n)) {
            omega_gpu_session_set_error("cmdlist pair: a wait_for points outside the other list of this pair"); return -1;
        }
    }
    if (o.mode != OMEGA_GPU_CMDLIST_PCAS || o.stamps || o.skip_launch) { omega_gpu_session_set_error("cmdlist pair: PCAS mode only, no stamps, no skip"); return -1; }
    if (omega_gpu_cmdlist_is_latched() || omega_gpu_session_is_blocked()) { omega_gpu_session_set_error("cmdlist pair: latched"); return -1; }
    OmegaGpuSession *s = omega_gpu_session_open();
    if (!s) return -1;
    if (!s->marker.cpu || s->marker.size < MARKER_OFF + PAIR_B_MARKER + 0x20) { omega_gpu_session_set_error("cmdlist pair: session marker page"); return -1; }
    OmegaGpuQueueHandle q1;
    if (omega_gpu_session_extra_queue(1, &q1) != 0) return -1; /* names the refusing stage */
    const OmegaGpuMem *marker = &s->marker;
    const uint64_t marker_va = marker->va + MARKER_OFF;
    const uint64_t timeout = o.timeout_ms ? o.timeout_ms : DEFAULT_TIMEOUT_MS;
    const uint32_t spin = o.spin_us ? o.spin_us : omega_gpu_session_spin_us();
    const uint64_t th0 = p->time_now_ns(p->ctx);
    /* one submission per list: both must fit their ring with the platform prefix (a q38 token is ~650 launches x 8 words) */
    const uint32_t prefix = omega_gpu_session_submit_prefix_words();
    const OmegaGpuCmdList *L[2] = { a, b };
    for (int k = 0; k < 2; k++) {
        const uint32_t avail = L[k]->words_cap - 18u - TAIL_WORDS - 2u - STAMP_WORDS;
        if (prefix > avail || (avail - prefix) / (L[k]->nwaits ? 14u : 8u) < L[k]->n) { omega_gpu_session_set_error("cmdlist pair: a list does not fit one submission"); return -1; }
    }
    prepare_qmd_mode(a, &o); prepare_qmd_mode(b, &o);
    if (pair_prep(p, a) || pair_prep(p, b)) { omega_gpu_session_set_error("cmdlist pair: sync_cpu_to_gpu"); return -1; }
    const size_t na = emit_chunk(a, &o, 0, a->n, marker_va), nb = emit_chunk(b, &o, 0, b->n, marker_va + PAIR_B_MARKER);
    volatile uint32_t *mk = (volatile uint32_t *)((uint8_t *)marker->cpu + MARKER_OFF);
    mk[0] = 0; mk[4] = 0; mk[PAIR_B_MARKER / 4] = 0; mk[PAIR_B_MARKER / 4 + 4] = 0;
    if (p->sync_cpu_to_gpu(p->ctx, marker->handle, MARKER_OFF, PAIR_B_MARKER + 0x20) != OMEGA_GPU_PLAT_OK) {
        omega_gpu_session_set_error("cmdlist pair: sync qmd / marker"); return -1;
    }
    const uint64_t t0 = p->time_now_ns(p->ctx);
    const uint64_t host = t0 - th0;
    if (native_begin(a, 0, a->n) != 0) return -1;
    if (native_begin(b, 0, b->n) != 0) { (void)native_events(a, 0, a->n, 4); return -1; }
    const int aqrc = p->queue_submit(p->ctx, s->queue, a->words, (uint32_t)na);
    if (aqrc != OMEGA_GPU_PLAT_OK) {
        if (aqrc == OMEGA_GPU_PLAT_REFUSED_BUSY) {
            pair_latch("cmdlist pair: uncertain earlier submission (latched)");
        } else {
            (void)native_events(a, 0, a->n, 4);
            (void)native_events(b, 0, b->n, 4);
        }
        omega_gpu_session_set_error("cmdlist pair: queue_submit (queue 0) refused"); return -1;
    }
    const int bqrc = p->queue_submit(p->ctx, q1, b->words, (uint32_t)nb);
    const int b_ok = bqrc == OMEGA_GPU_PLAT_OK;
    if (!b_ok && bqrc != OMEGA_GPU_PLAT_REFUSED_BUSY) (void)native_events(b, 0, b->n, 4);
    int w = wait_word(p, marker, 0, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, timeout, spin);
    if (w == WAIT_OK) w = wait_word(p, marker, 0x10, OMEGA_BW_MARKER2_PAYLOAD, timeout, spin);
    if (w == WAIT_OK && b_ok) w = wait_word(p, marker, PAIR_B_MARKER, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, timeout, spin);
    if (w == WAIT_OK && b_ok) w = wait_word(p, marker, PAIR_B_MARKER + 0x10, OMEGA_BW_MARKER2_PAYLOAD, timeout, spin);
    if (w != WAIT_OK) { pair_latch(w == WAIT_TIMED_OUT ? "cmdlist pair: marker wait timed out (latched)" : "cmdlist pair: marker poll refused (latched)"); return -1; }
    (void)p->sync_gpu_to_cpu(p->ctx, marker->handle, MARKER_OFF, PAIR_B_MARKER + 0x20);
    const uint64_t elapsed = p->time_now_ns(p->ctx) - t0;
    (void)p->queue_retire(p->ctx, s->queue);
    (void)native_events(a, 0, a->n, 2);
    if (!b_ok) {
        if (bqrc == OMEGA_GPU_PLAT_REFUSED_BUSY) pair_latch("cmdlist pair: queue 1 uncertain earlier flight (latched)");
        omega_gpu_session_set_error("cmdlist pair: queue_submit (queue 1) refused"); return -1;
    }
    (void)p->queue_retire(p->ctx, q1);
    (void)native_events(b, 0, b->n, 2);
    const int ra = pair_check(p, a, (uint32_t)na, elapsed, host, sa), rb = pair_check(p, b, (uint32_t)nb, elapsed, host, sb);
    return ra == 0 && rb == 0 ? 0 : -1;
}

int omega_gpu_cmdlist_stamps(const OmegaGpuCmdList *cl, uint64_t *out, uint32_t cap) {
    if (!cl || !out || !cl->stamps.cpu) { omega_gpu_session_set_error("cmdlist stamps: arguments"); return -1; }
    if (!cl->stamped) { omega_gpu_session_set_error("cmdlist stamps: the last submit had stamps off"); return -1; }
    const uint32_t want = cl->n + 1u;
    if (cap < want) { omega_gpu_session_set_error("cmdlist stamps: out too small"); return -1; }
    const uint64_t *s = cl->stamps.cpu;
    for (uint32_t i = 0; i < want; i++) {
        if (s[2 * i] != (uint64_t)i || s[2 * i + 1] == UINT64_MAX) { omega_gpu_session_set_error("cmdlist stamps: a slot was not written (stamps off, or the launch did not run)"); return -1; }
        out[i] = s[2 * i + 1];
    }
    return (int)want;
}

const char *omega_gpu_cmdlist_barrier_name(OmegaGpuCmdListBarrier b) {
    switch (b) {
    case OMEGA_GPU_CMDLIST_BARRIER_WFI: return "wfi+inval_all";
    case OMEGA_GPU_CMDLIST_BARRIER_NONE: return "none";
    case OMEGA_GPU_CMDLIST_BARRIER_WFI_ONLY: return "wfi_only";
    case OMEGA_GPU_CMDLIST_BARRIER_WFI_DATA: return "wfi+inval_data";
    case OMEGA_GPU_CMDLIST_BARRIER_FLUSH_DATA: return "flush_data";
    }
    return "?";
}
