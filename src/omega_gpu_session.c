/*
 * omega_gpu_session.c: FB-1 cut 4b. See the header for the contract.
 *
 * The launch words are the ones omega_gpu_matmul_api.c (cut 1b, chip PASS
 * FB1-CUT1B-4345406) and omega_gpu_elementwise_api.c (cut 4, FB1-CUT4-0d0241a)
 * both emitted; the two were identical except for the argument word count, which
 * is now a parameter. Nothing about the sequence changed in this cut.
 */
#include "omega_gpu_session.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_submit.h"
#include "nvrm.h"
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define PAGE 0x1000ull

/* Same words as src/omega_blackwell_submit.c SETUP_WORDS (compute class setup). */
static const uint32_t SETUP_WORDS[18] = {
    0x20012061, 0x0000cec0, 0x20012092, 0x00000001, 0x200120a8, 0x0000000f, 0x2001255d, 0x00000003,
    0x2001255e, 0x20000000, 0x2001255f, 0x000fffff, 0x20012557, 0x00000003, 0x20012558, 0x22000000,
    0x20012559, 0x00000000,
};

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static OmegaGpuSession g_s;
static bool g_open, g_blocked;
static char g_err[320] = "";
static void (*g_hooks[8])(void);
static unsigned g_nhooks;

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static size_t page_up(size_t b) { b = (b + PAGE - 1) & ~(PAGE - 1); return b < PAGE ? PAGE : b; }

const char *omega_gpu_session_last_error(void) { return g_err; }
void omega_gpu_session_set_error(const char *stage) {
    snprintf(g_err, sizeof g_err, "%s", stage ? stage : "");
}
static void fail_rm(const char *stage, const Nvrm *rm) {
    /* the driver text (RM class / ioctl errno / RM status, nvrm.c fail()) travels with the stage */
    snprintf(g_err, sizeof g_err, "%s: %s", stage, rm && rm->err[0] ? rm->err : "(no driver text)");
}

void omega_gpu_session_lock(void) { pthread_mutex_lock(&g_mu); }
void omega_gpu_session_unlock(void) { pthread_mutex_unlock(&g_mu); }
bool omega_gpu_session_is_open(void) { return g_open; }
int omega_gpu_session_is_blocked(void) { return g_blocked ? 1 : 0; }

int omega_gpu_session_alloc(size_t bytes, NvrmMem *out) {
    if (!g_open || g_blocked) { omega_gpu_session_set_error("alloc without open device"); return -1; }
    if (nvrm_alloc_gpu_uncached(&g_s.ctx.rm, page_up(bytes), out) != 0) { fail_rm("nvrm_alloc_gpu_uncached", &g_s.ctx.rm); memset(out, 0, sizeof *out); return -1; }
    return 0;
}

void omega_gpu_session_free(NvrmMem *m) {
    if (m->cpu && g_open && !g_blocked) (void)nvrm_free(&g_s.ctx.rm, m);
    memset(m, 0, sizeof *m);
}

int omega_gpu_session_scratch(OmegaGpuScratch *s, size_t bytes) {
    if (bytes > s->high_water) s->high_water = bytes;
    if (s->cap >= bytes && s->mem.cpu) return 0;
    omega_gpu_session_free(&s->mem);
    s->cap = 0;
    if (omega_gpu_session_alloc(s->high_water, &s->mem) != 0) return -1;
    s->cap = (size_t)s->mem.size;
    return 0;
}

void omega_gpu_session_scratch_free(OmegaGpuScratch *s) {
    omega_gpu_session_free(&s->mem);
    s->cap = 0;
}

static void free_session_buffers(void) {
    omega_gpu_session_free(&g_s.qmd); omega_gpu_session_free(&g_s.marker);
    omega_gpu_session_free(&g_s.cbank); omega_gpu_session_free(&g_s.pb);
}

OmegaGpuSession *omega_gpu_session_open(void) {
    if (g_blocked) { omega_gpu_session_set_error("process latched"); return NULL; }
    if (g_open) return &g_s;
    uint64_t t0 = now_ns();
    memset(&g_s.ctx, 0, sizeof g_s.ctx);
    /* nvrm_open directly (m16_native_open is memset + nvrm_open + nvrm_close on
     * failure) so the driver's own failure text survives into the stage name. */
    if (nvrm_open(&g_s.ctx.rm) != 0) { fail_rm("nvrm_open", &g_s.ctx.rm); (void)nvrm_close(&g_s.ctx.rm); return NULL; }
    if (m16_native_create_channel(&g_s.ctx) != 0) { fail_rm("m16_native_create_channel", &g_s.ctx.rm); (void)m16_native_close(&g_s.ctx); return NULL; }
    g_open = true; /* so alloc / free work during the rest of the open */
    if (omega_gpu_session_alloc(0x10000, &g_s.pb) || omega_gpu_session_alloc(PAGE, &g_s.cbank) ||
        omega_gpu_session_alloc(PAGE, &g_s.marker) || omega_gpu_session_alloc(0x10000, &g_s.qmd)) {
        char keep[sizeof g_err]; memcpy(keep, g_err, sizeof keep);
        free_session_buffers();
        g_open = false;
        (void)m16_native_close(&g_s.ctx);
        snprintf(g_err, sizeof g_err, "session scratch alloc (%s)", keep);
        return NULL;
    }
    g_s.ctx.pb_mem = g_s.pb; /* m16_native_enqueue_methods copies the method stream here */
    g_s.opens++;
    g_s.last_open_ns = now_ns() - t0;
    return &g_s;
}

int omega_gpu_session_on_close(void (*hook)(void)) { /* caller holds the lock (the APIs register from their open path) */
    if (!hook) return -1;
    int rc = 0;
    bool seen = false;
    for (unsigned i = 0; i < g_nhooks; i++) if (g_hooks[i] == hook) seen = true;
    if (!seen) { if (g_nhooks < sizeof g_hooks / sizeof g_hooks[0]) g_hooks[g_nhooks++] = hook; else rc = -1; }
    return rc;
}

void omega_gpu_session_close(void) {
    pthread_mutex_lock(&g_mu);
    if (g_open && !g_blocked) {
        for (unsigned i = 0; i < g_nhooks; i++) g_hooks[i]();
        free_session_buffers();
        (void)m16_native_close(&g_s.ctx);
        g_open = false;
    }
    pthread_mutex_unlock(&g_mu);
}

static uint32_t round_up(uint32_t v, uint32_t q) { return (v + q - 1) / q * q; }

int omega_gpu_session_launch(const OmegaGpuLaunch *L, uint64_t *elapsed_ns, uint32_t *marker_out) {
    if (!g_open || g_blocked) { omega_gpu_session_set_error(g_blocked ? "process latched" : "launch without open device"); return -1; }
    if (!L || !L->args || L->n_args == 0 || L->n_args > OMEGA_GPU_SESSION_MAX_ARGS || L->threads_x == 0 || L->grid_x == 0 || L->grid_y == 0) {
        omega_gpu_session_set_error("launch arguments"); return -1;
    }
    const uint32_t threads_y = L->threads_y ? L->threads_y : 1;
    uint32_t cbank_data[OMEGA_BW_CBANK_DRIVER_WORDS];
    omega_blackwell_build_cbank_driver_2d(cbank_data, g_s.cbank.va, L->threads_x, threads_y, L->grid_x, L->grid_y);
    memcpy(g_s.cbank.cpu, cbank_data, sizeof cbank_data);
    memcpy((uint8_t *)g_s.cbank.cpu + 0x380, L->args, (size_t)L->n_args * 4u);

    const uint64_t qmd0_va = g_s.qmd.va, qmd1_va = g_s.qmd.va + 0x1000, sem_va = g_s.qmd.va + 0x2000, scratch_va = g_s.qmd.va + 0x4000;
    uint32_t gpr = L->gpr_count > 64u ? round_up(L->gpr_count, 8u) : 64u;
    OmegaBlackwellQmdConfig cfg = {
        .code_va = L->code_va, .cbank_va = g_s.cbank.va, .scratch_va = scratch_va, .sem_va = sem_va,
        .qmd0_va = qmd0_va, .qmd1_va = qmd1_va, .num_elements = L->num_elements,
        .threads_per_block = L->threads_x, .grid_width = L->grid_x * L->grid_y, .threads_x = L->threads_x, .threads_y = threads_y,
        .grid_x = L->grid_x, .grid_y = L->grid_y, .gpr_count = gpr,
    };
    uint32_t qmd0_words[OMEGA_BW_QMD_WORDS], qmd1_words[OMEGA_BW_QMD_WORDS];
    omega_blackwell_build_qmd0(qmd0_words, qmd0_va, qmd1_va);
    omega_blackwell_build_qmd1(qmd1_words, &cfg);
    if (omega_blackwell_verify_qmd_invariants(qmd1_words) != 0) { omega_gpu_session_set_error("qmd invariants"); return -1; }
    memcpy(g_s.qmd.cpu, qmd0_words, sizeof qmd0_words);
    memcpy((uint8_t *)g_s.qmd.cpu + 0x1000, qmd1_words, sizeof qmd1_words);

    volatile uint32_t *hsem = (volatile uint32_t *)((uint8_t *)g_s.qmd.cpu + 0x2000);
    volatile uint32_t *hmarker = (volatile uint32_t *)g_s.marker.cpu;
    volatile uint32_t *hmarker2 = (volatile uint32_t *)((uint8_t *)g_s.marker.cpu + 0x10);
    *hsem = 0; *hmarker = 0; *hmarker2 = 0;
    __asm__ volatile("dsb sy" ::: "memory");

    uint32_t pb[1024];
    size_t n = 0;
    memcpy(&pb[n], SETUP_WORDS, sizeof SETUP_WORDS); n += sizeof SETUP_WORDS / 4;
    /* Invalidate the SM instruction, constant and data caches before this launch
     * (cut 1b finding: a reused code address served stale instructions; clcec0.h
     * NVCEC0_INVALIDATE_SHADER_CACHES 0x021c, bits INSTRUCTION 0, DATA 4, CONSTANT 12). */
    pb[n++] = nvrm_mthd(1, OMEGA_BW_MTHD_INVALIDATE_SHADER_CACHES, 1); pb[n++] = OMEGA_BW_INVALIDATE_SHADER_CACHES_ALL;
    /* constant bank 0: driver words, then our arguments at 0x380 (inline upload) */
    pb[n++] = nvrm_mthd(1, 0x0188, 2); pb[n++] = (uint32_t)(g_s.cbank.va >> 32); pb[n++] = (uint32_t)g_s.cbank.va;
    pb[n++] = nvrm_mthd(1, 0x0180, 2); pb[n++] = 0x00000380; pb[n++] = 0x00000001;
    pb[n++] = nvrm_mthd(1, 0x01b0, 1); pb[n++] = 0x00000041;
    pb[n++] = (224u << 16) | (1u << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[n], cbank_data, 224 * 4); n += 224;
    pb[n++] = nvrm_mthd(1, 0x0188, 2); pb[n++] = (uint32_t)((g_s.cbank.va + 0x380) >> 32); pb[n++] = (uint32_t)(g_s.cbank.va + 0x380);
    pb[n++] = nvrm_mthd(1, 0x0180, 2); pb[n++] = L->n_args * 4u; pb[n++] = 0x00000001;
    pb[n++] = nvrm_mthd(1, 0x01b0, 1); pb[n++] = 0x00000041;
    pb[n++] = (L->n_args << 16) | (1u << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[n], L->args, (size_t)L->n_args * 4u); n += L->n_args;
    /* QMD0 (null grid, dependence 2) */
    pb[n++] = (98u << 16) | (1u << 13) | (0x0318 >> 2) | (2u << 28);
    pb[n++] = (1u << 30) | (uint32_t)((qmd0_va >> 40) & 0x1ff); pb[n++] = (uint32_t)(qmd0_va >> 8);
    memcpy(&pb[n], qmd0_words, 96 * 4); n += 96;
    /* intermediate semaphore init */
    pb[n++] = nvrm_mthd(1, 0x0188, 2); pb[n++] = (uint32_t)(sem_va >> 32); pb[n++] = (uint32_t)sem_va;
    pb[n++] = nvrm_mthd(1, 0x0180, 2); pb[n++] = 0x00000004; pb[n++] = 0x00000001;
    pb[n++] = nvrm_mthd(1, 0x01b0, 1); pb[n++] = 0x00000041;
    pb[n++] = (1u << 16) | (1u << 13) | (0x01b4 >> 2) | (6u << 28);
    pb[n++] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT;
    /* QMD1 (the kernel) */
    pb[n++] = (98u << 16) | (1u << 13) | (0x0318 >> 2) | (2u << 28);
    pb[n++] = (1u << 30) | (uint32_t)((qmd1_va >> 40) & 0x1ff); pb[n++] = (uint32_t)(qmd1_va >> 8);
    memcpy(&pb[n], qmd1_words, 96 * 4); n += 96;
    /* first marker */
    pb[n++] = nvrm_mthd(0, 0x005c, 5);
    pb[n++] = (uint32_t)g_s.marker.va; pb[n++] = (uint32_t)(g_s.marker.va >> 32);
    pb[n++] = OMEGA_BW_MARKER_COMPLETION_PAYLOAD; pb[n++] = 0; pb[n++] = 0x1 | (1u << 20);
    /* C3 tail: L2_FLUSH_DIRTY, then the second release on the uncached marker page */
    pb[n++] = nvrm_mthd(0, 0x0028, 4); pb[n++] = 0; pb[n++] = 0; pb[n++] = 0; pb[n++] = (0x10u << 27);
    pb[n++] = nvrm_mthd(0, 0x005c, 5);
    pb[n++] = (uint32_t)(g_s.marker.va + 0x10); pb[n++] = (uint32_t)((g_s.marker.va + 0x10) >> 32);
    pb[n++] = OMEGA_BW_MARKER2_PAYLOAD; pb[n++] = 0; pb[n++] = 0x1 | (1u << 20);

    uint64_t t0 = now_ns();
    if (m16_native_submit_methods(&g_s.ctx, pb, n) != 0) { fail_rm("submit_methods", &g_s.ctx.rm); return -1; }
    if (m16_native_wait_marker(hmarker, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, L->timeout_ms) != 0 ||
        m16_native_wait_marker(hmarker2, OMEGA_BW_MARKER2_PAYLOAD, L->timeout_ms) != 0) {
        g_blocked = true; /* uncertain completion: keep everything, refuse every later call */
        omega_gpu_session_set_error("marker wait timed out (latched)");
        return -1;
    }
    __asm__ volatile("dsb sy" ::: "memory");
    uint64_t t1 = now_ns();
    nvrm_retire(&g_s.ctx.rm, g_s.ctx.rm.put); /* both releases landed: the GPFIFO entry is consumed */
    if (*hsem != OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE) { omega_gpu_session_set_error("intermediate semaphore not DONE"); return -1; }
    if (elapsed_ns) *elapsed_ns += t1 - t0;
    if (marker_out) *marker_out = *hmarker;
    return 0;
}

uint32_t omega_gpu_session_open_count(void) { return g_s.opens; }
