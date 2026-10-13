/*
 * omega_gpu_cmdlist.c: many launches in one submission (see the header).
 *
 * The per-launch words reuse the session's chip-proven pieces (omega_gpu_session.c, read 2026-10-13): SETUP_WORDS,
 * INVALIDATE_SHADER_CACHES_ALL, the QMD0/QMD1 builders, the C3 tail (first marker with RELEASE_WFI, L2_FLUSH_DIRTY
 * mem-op, second marker on the uncached page) and the poll loop. What is new: banks and QMDs live in per-launch slots
 * written by the CPU, WAIT_FOR_IDLE sits between launches, and the PCAS launch method.
 */
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_submit.h"
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

struct OmegaGpuCmdList {
    uint32_t cap, n;
    OmegaGpuMem banks, qmds, sems, scratch;
    uint32_t *words;      /* host method stream */
    uint32_t words_cap;
    OmegaGpuMem nop_code; /* EXIT kernel used to force the session latch after a timeout */
    uint32_t nop_gpr;
};

static uint32_t g_latched;

int omega_gpu_cmdlist_is_latched(void) { return __atomic_load_n(&g_latched, __ATOMIC_ACQUIRE) ? 1 : 0; }
uint32_t omega_gpu_cmdlist_count(const OmegaGpuCmdList *cl) { return cl ? cl->n : 0; }
void omega_gpu_cmdlist_reset(OmegaGpuCmdList *cl) { if (cl) cl->n = 0; }

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
    cl->words_cap = RING_WORDS;
    cl->words = omega_gpu_host_alloc((size_t)cl->words_cap * 4u);
    int rc = cl->words ? 0 : -1;
    omega_gpu_session_lock();
    if (rc == 0 && !omega_gpu_session_open()) rc = -1;
    if (rc == 0 && (omega_gpu_session_alloc((size_t)cap * OMEGA_GPU_CMDLIST_BANK_BYTES, &cl->banks) != 0 ||
                    omega_gpu_session_alloc((size_t)cap * OMEGA_GPU_CMDLIST_QMD_BYTES, &cl->qmds) != 0 ||
                    omega_gpu_session_alloc((size_t)cap * 16u, &cl->sems) != 0 ||
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
    OmegaGpuMem *m[] = { &cl->banks, &cl->qmds, &cl->sems, &cl->scratch, &cl->nop_code };
    for (unsigned i = 0; i < sizeof m / sizeof m[0]; i++) if (m[i]->cpu) omega_gpu_session_free(m[i]);
    omega_gpu_session_unlock();
    omega_gpu_host_free(cl->words);
    omega_gpu_host_free(cl);
}

static uint32_t round8(uint32_t v) { return (v + 7u) / 8u * 8u; }

int omega_gpu_cmdlist_add(OmegaGpuCmdList *cl, const OmegaGpuLaunch *L) {
    if (!cl || !L || !L->args || L->n_args == 0 || L->n_args > OMEGA_GPU_SESSION_MAX_ARGS || L->threads_x == 0 || L->grid_x == 0 || L->grid_y == 0 ||
        L->code_va == 0) { omega_gpu_session_set_error("cmdlist add: arguments"); return -1; }
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
    cl->n++;
    return (int)i;
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

/* Uncertain completion: latch this module, then queue a 1 ms session launch behind the stuck list. If the channel is
 * still busy it times out and the session latches the whole process (parks every buffer, refuses every API). */
static void force_latch(OmegaGpuCmdList *cl, const char *why) {
    __atomic_store_n(&g_latched, 1u, __ATOMIC_RELEASE);
    uint32_t arg = 0;
    OmegaGpuLaunch L = { .code_va = cl->nop_code.va, .gpr_count = cl->nop_gpr, .threads_x = 32, .threads_y = 1, .grid_x = 1, .grid_y = 1,
                         .num_elements = 1, .args = &arg, .n_args = 1, .timeout_ms = 1 };
    (void)omega_gpu_session_launch(&L, NULL, NULL);
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

static size_t emit_inline_qmd(uint32_t *pb, size_t n, uint64_t va, const uint32_t *words) {
    pb[n++] = (98u << 16) | (1u << 13) | (0x0318 >> 2) | (2u << 28);
    pb[n++] = (1u << 30) | (uint32_t)((va >> 40) & 0x1ff); pb[n++] = (uint32_t)(va >> 8);
    memcpy(&pb[n], words, 96 * 4); return n + 96;
}

int omega_gpu_cmdlist_submit(OmegaGpuCmdList *cl, const OmegaGpuCmdListOpts *opts, OmegaGpuCmdListStats *st) {
    OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o);
    if (opts) o = *opts;
    if (st) { memset(st, 0, sizeof *st); st->first_missing = -1; }
    const OmegaGpuPlatform *p = omega_gpu_session_platform();
    if (!cl || cl->n == 0 || !p) { omega_gpu_session_set_error("cmdlist submit: empty list or no platform"); return -1; }
    if (omega_gpu_cmdlist_is_latched() || omega_gpu_session_is_blocked()) { omega_gpu_session_set_error("cmdlist submit: latched"); return -1; }
    OmegaGpuSession *s = omega_gpu_session_open();
    if (!s) return -1;
    if (!s->marker.cpu || s->marker.size < MARKER_OFF + 0x20) { omega_gpu_session_set_error("cmdlist submit: session marker page"); return -1; }
    const OmegaGpuMem *marker = &s->marker;
    const uint64_t marker_va = marker->va + MARKER_OFF;
    const uint64_t timeout = o.timeout_ms ? o.timeout_ms : DEFAULT_TIMEOUT_MS;
    const uint32_t spin = o.spin_us ? o.spin_us : omega_gpu_session_spin_us();
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

    const uint32_t per_launch = o.mode == OMEGA_GPU_CMDLIST_INLINE ? 4u + 2u * 99u : 8u;
    const uint32_t room = (cl->words_cap - 18u - TAIL_WORDS) / per_launch;
    uint64_t elapsed = 0, host = p->time_now_ns(p->ctx) - th0;
    uint32_t done = 0, submits = 0, words = 0;
    while (done < cl->n) {
        const uint64_t tb = p->time_now_ns(p->ctx);
        uint32_t *pb = cl->words;
        size_t n = 0;
        memcpy(pb, SETUP_WORDS, sizeof SETUP_WORDS); n += 18;
        const uint32_t end = (cl->n - done > room) ? done + room : cl->n;
        for (uint32_t i = done; i < end; i++) {
            if (o.skip_launch && i == (uint32_t)(o.skip_launch - 1)) continue;
            const bool barrier = o.barrier == OMEGA_GPU_CMDLIST_BARRIER_WFI || i == done;
            if (barrier) {
                pb[n++] = bw_mthd(1, MTHD_WAIT_FOR_IDLE, 1); pb[n++] = 0;
                pb[n++] = bw_mthd(1, OMEGA_BW_MTHD_INVALIDATE_SHADER_CACHES, 1); pb[n++] = OMEGA_BW_INVALIDATE_SHADER_CACHES_ALL;
            }
            const uint64_t q1_va = cl->qmds.va + (uint64_t)i * OMEGA_GPU_CMDLIST_QMD_BYTES;
            if (o.mode == OMEGA_GPU_CMDLIST_INLINE) {
                const uint8_t *qs = (const uint8_t *)cl->qmds.cpu + (size_t)i * OMEGA_GPU_CMDLIST_QMD_BYTES;
                uint32_t q1[OMEGA_BW_QMD_WORDS];
                memcpy(q1, qs, sizeof q1);
                q1[10] = 0x00190000u; q1[12] = (uint32_t)((q1_va + 0x200) >> 8); /* the session pair: QMD1 decrements QMD0 */
                n = emit_inline_qmd(pb, n, q1_va + 0x200, (const uint32_t *)(qs + 0x200));
                n = emit_inline_qmd(pb, n, q1_va, q1);
            } else {
                /* standalone grid: no dependent QMD (NVK builds Blackwell QMDs without one, nak/qmd.rs Qmd5_0) */
                uint32_t *q1 = (uint32_t *)((uint8_t *)cl->qmds.cpu + (size_t)i * OMEGA_GPU_CMDLIST_QMD_BYTES);
                q1[10] = 0; q1[12] = 0;
                pb[n++] = bw_mthd(1, MTHD_SEND_PCAS_A, 1); pb[n++] = (uint32_t)(q1_va >> 8);
                pb[n++] = bw_mthd(1, MTHD_SEND_SIGNALING_PCAS2_B, 1); pb[n++] = PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
            }
        }
        n = emit_tail(pb, n, marker_va);
        if (o.mode != OMEGA_GPU_CMDLIST_INLINE &&
            p->sync_cpu_to_gpu(p->ctx, cl->qmds.handle, 0, (uint64_t)cl->n * OMEGA_GPU_CMDLIST_QMD_BYTES) != OMEGA_GPU_PLAT_OK) {
            omega_gpu_session_set_error("cmdlist submit: sync qmd"); return -1;
        }
        mk[0] = 0; mk[4] = 0;
        (void)p->sync_cpu_to_gpu(p->ctx, marker->handle, MARKER_OFF, 0x20);
        const uint64_t t0 = p->time_now_ns(p->ctx);
        host += t0 - tb;
        if (p->queue_submit(p->ctx, s->queue, pb, (uint32_t)n) != OMEGA_GPU_PLAT_OK) {
            omega_gpu_session_set_error("cmdlist submit: queue_submit refused"); return -1; /* refused: nothing in flight */
        }
        int w = wait_word(p, marker, 0, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, timeout, spin);
        if (w == WAIT_OK) w = wait_word(p, marker, 0x10, OMEGA_BW_MARKER2_PAYLOAD, timeout, spin);
        if (w != WAIT_OK) { force_latch(cl, w == WAIT_TIMED_OUT ? "cmdlist: marker wait timed out (latched)" : "cmdlist: marker poll refused (latched)"); return -1; }
        (void)p->sync_gpu_to_cpu(p->ctx, marker->handle, MARKER_OFF, 0x20);
        elapsed += p->time_now_ns(p->ctx) - t0;
        (void)p->queue_retire(p->ctx, s->queue);
        submits++; words += (uint32_t)n; done = end;
    }
    (void)p->sync_gpu_to_cpu(p->ctx, cl->sems.handle, 0, (uint64_t)cl->n * 16u);
    uint32_t ok = 0; int32_t missing = -1;
    for (uint32_t i = 0; i < cl->n; i++) {
        if (sem[4 * i] == OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE) ok++;
        else if (missing < 0) missing = (int32_t)i;
    }
    if (st) { st->launches = cl->n; st->submits = submits; st->words = words; st->elapsed_ns = elapsed; st->host_ns = host; st->sems_done = ok; st->first_missing = missing; }
    if (missing >= 0) { omega_gpu_session_set_error("cmdlist: a launch semaphore is not DONE (launch did not run)"); return -1; }
    return 0;
}
