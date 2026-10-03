/*
 * Omega GPU Engine, Blackwell GB10 backend (cut A3b1). See
 * src/omega_blackwell_engine.h and docs/numeric/OMEGA_GPU_ENGINE.md.
 *
 * This file is the backend function table (OmegaGpuBackend) that the engine
 * core in src/omega_gpu_engine.c drives. The core owns the state machine, the
 * poison comparison, the uncertain-completion block and the cleanup order.
 * This file owns only the device calls. Each callback below says which lines of
 * the launcher it was lifted from. "SUB@676f16f" means
 * omega_blackwell_execute_vector in src/omega_blackwell_submit.c at commit
 * 676f16f (the file before this cut). "ASTRA" means the first cut by GPT-6
 * Astra, src/omega_blackwell_engine.c at commit 21392ee on branch
 * fix/omega-gpu-engine-first-cut.
 *
 * Roles. SCRATCH is the backend's own and holds four device buffers: the large
 * pushbuffer (0x10000), the driver/argument constant bank (0x1000), the marker
 * page (0x1000) and the QMD page (0x10000), the same sizes as SUB@676f16f
 * lines 50, 59, 63, 64. PROGRAM, INPUT_A, INPUT_B and OUTPUT are page rounded
 * with a 0x1000 minimum (SUB@676f16f lines 55-56, 58-62).
 *
 * L2 flush. The core has no separate callback for it, so build() emits the L2
 * flush and the second release marker (payload 0x46464646 at marker page + 0x10)
 * inside the pushbuffer unless the job sets NO_C3, exactly as divsqrt does in
 * src/omega_numeric_divsqrt_gb10.c:1145-1152. The wait_marker2 callback waits
 * for that marker. The CPU side of the visibility rule (dsb sy before readback)
 * is the barrier callback (divsqrt line 1171).
 *
 * One run at a time. The context is one static object because a retained run
 * (uncertain completion, nothing freed, nothing closed) must keep its device
 * memory reachable for the life of the process, and because the native context
 * is about 225 KiB (Nvrm in physics/nvrm/nvrm.h). A second open while a run is
 * active fails with DEVICE_OPEN.
 *
 * NOT_RUN: this file has not been compiled or executed by its author.
 */
#include "omega_blackwell_engine.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_submit.h"
#include "m16_native.h"

#include <errno.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define BE_PB_CAP 1024u            /* pushbuffer words, SUB@676f16f line 131 (uint32_t pb[1024]) */
#define BE_PAGE 0x1000u
#define BE_PB_BYTES 0x10000u       /* SUB@676f16f line 50 */
#define BE_QMD_BYTES 0x10000u      /* SUB@676f16f line 64 */
#define BE_SEM_OFFSET 0x2000u      /* sem_va = qmd page + 0x2000, SUB@676f16f line 99 */
#define BE_MARKER2_OFFSET 0x10u    /* divsqrt line 1113 and 1152 */
#define BE_MARKER2_PAYLOAD 0x46464646u /* divsqrt line 1152 */

typedef struct {
    M16NativeContext m;
    NvrmMem pb;                               /* SCRATCH part 1: large pushbuffer */
    NvrmMem cbank;                            /* SCRATCH part 2 */
    NvrmMem marker;                           /* SCRATCH part 3 */
    NvrmMem qmd;                              /* SCRATCH part 4 (QMD0, QMD1, semaphore) */
    NvrmMem buf[OMEGA_GPU_BUF_ROLE_COUNT];    /* PROGRAM, INPUT_A, INPUT_B, OUTPUT (SCRATCH slot unused) */
    int native_open;                          /* m16_native_open was attempted this run */
    int pb_replaced;                          /* m.pb_mem now points at the large pushbuffer */
    int c3;                                   /* L2 flush and second marker are in the pushbuffer */
    volatile uint32_t *hmarker;               /* NULL until build zeroes it; NULL again once SCRATCH is freed */
    volatile uint32_t *hmarker2;
    volatile uint32_t *hsem;
    uint32_t pb_words[BE_PB_CAP];
    size_t pb_len;
    int pb_overflow;
    int last_rc;                              /* raw driver code of the most recent callback */
    char text[OMEGA_GPU_ENGINE_DRV_TEXT_BYTES]; /* backend's own message, "" when none */
    OmegaBlackwellEngineRunInfo info;
} BeCtx;

static BeCtx g_be;
static atomic_flag g_busy = ATOMIC_FLAG_INIT;

static uint64_t now_ns(void)
{
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) return 0u;
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* Bounded copy that never reads past a NUL and always terminates. */
static void copy_text(char *dst, size_t cap, const char *src)
{
    size_t i = 0;
    if (cap == 0) return;
    while (i + 1u < cap && src[i] != '\0') {
        dst[i] = src[i];
        i++;
    }
    dst[i] = '\0';
}

static void note(BeCtx *c, const char *msg)
{
    copy_text(c->text, sizeof c->text, msg);
}

/* Start of every callback that can fail: forget the previous callback's code and text. */
static void begin(BeCtx *c)
{
    c->last_rc = 0;
    c->text[0] = '\0';
}

/* ---- allocation helpers ------------------------------------------------ */

/* Allocates and checks one device buffer. The post-allocation checks (CPU
 * mapping present, size at least what was asked, 256-byte aligned GPU address)
 * come from ASTRA checked_alloc. Returns 0 or non-zero. */
static int alloc_checked(BeCtx *c, uint64_t bytes, NvrmMem *m)
{
    int rc;
    memset(m, 0, sizeof *m);
    /* GB10: the CPU rewrites pushbuffers and reads markers/outputs; GPU-cacheable system memory is not
     * coherent with CPU mappings (nvos.h:1117-1118). All engine buffers are GPU-uncached until the
     * allocator default is flipped (hive idea I36). */
    rc = nvrm_alloc_gpu_uncached(&c->m.rm, bytes, m);
    if (rc != 0) {
        c->last_rc = rc;
        memset(m, 0, sizeof *m);
        return rc;
    }
    if (!m->cpu || m->size < bytes || (m->va & 255u) != 0) {
        note(c, "allocation failed its own checks (CPU mapping, size or 256-byte alignment)");
        (void)nvrm_free(&c->m.rm, m);
        memset(m, 0, sizeof *m);
        return -1;
    }
    return 0;
}

/* Page rounding with a 0x1000 minimum, SUB@676f16f lines 55-56. Non-zero if bytes is 0 or overflows. */
static int round_pages(size_t bytes, uint64_t *out)
{
    uint64_t v;
    if (bytes == 0 || (uint64_t)bytes > UINT64_MAX - (BE_PAGE - 1u)) return -1;
    v = ((uint64_t)bytes + (BE_PAGE - 1u)) & ~(uint64_t)(BE_PAGE - 1u);
    if (v < BE_PAGE) v = BE_PAGE;
    *out = v;
    return 0;
}

static NvrmMem *role_mem(BeCtx *c, int role)
{
    if (role < (int)OMEGA_GPU_BUF_PROGRAM || role >= (int)OMEGA_GPU_BUF_ROLE_COUNT) return NULL;
    return &c->buf[role];
}

/* Frees the four SCRATCH buffers, every one even if an earlier one fails.
 * Returns 0, or the first non-zero nvrm_free code. */
static int release_scratch(BeCtx *c)
{
    NvrmMem *parts[4];
    int first = 0;
    size_t i;
    parts[0] = &c->qmd;
    parts[1] = &c->marker;
    parts[2] = &c->cbank;
    parts[3] = &c->pb;
    /* The marker and semaphore words live in these buffers: stop reading them first. */
    c->hmarker = NULL;
    c->hmarker2 = NULL;
    c->hsem = NULL;
    for (i = 0; i < 4u; i++) {
        int rc = nvrm_free(&c->m.rm, parts[i]); /* a zeroed NvrmMem is a no-op (nvrm.h) */
        if (rc != 0 && first == 0) first = rc;
    }
    if (c->pb_replaced) {
        memset(&c->m.pb_mem, 0, sizeof c->m.pb_mem);
        c->pb_replaced = 0;
    }
    return first;
}

/* ---- pushbuffer helpers ------------------------------------------------ */

static void pb_put(BeCtx *c, uint32_t w)
{
    if (c->pb_len < BE_PB_CAP) c->pb_words[c->pb_len++] = w;
    else c->pb_overflow = 1;
}

static void pb_copy(BeCtx *c, const uint32_t *w, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) pb_put(c, w[i]);
}

/* Compute class and engine setup methods, SUB@676f16f lines 12-16 (also in divsqrt as SETUP). */
static const uint32_t BE_SETUP_WORDS[18] = {
    0x20012061, 0x0000cec0, 0x20012092, 0x00000001, 0x200120a8, 0x0000000f, 0x2001255d, 0x00000003,
    0x2001255e, 0x20000000, 0x2001255f, 0x000fffff, 0x20012557, 0x00000003, 0x20012558, 0x22000000,
    0x20012559, 0x00000000,
};

/* ---- callbacks --------------------------------------------------------- */

/* Opens the RM client. Resets every bit of the context first so an open
 * failure never reports values from an earlier run (ASTRA did memset(&ctx) at
 * the same point). The busy flag stays set until be_close. */
static int be_open_device(void *ctx)
{
    BeCtx *c = (BeCtx *)ctx;
    int rc;
    if (atomic_flag_test_and_set(&g_busy)) return -1; /* another run is active; touch nothing */
    memset(c, 0, sizeof *c);
    c->native_open = 1;
    rc = m16_native_open(&c->m); /* SUB@676f16f line 45 */
    if (rc != 0) {
        c->last_rc = rc;
        atomic_flag_clear(&g_busy); /* native open already cleaned up after itself (m16_native.c:15-21) */
        return rc;
    }
    return 0;
}

static int be_create_channel(void *ctx)
{
    BeCtx *c = (BeCtx *)ctx;
    int rc;
    begin(c);
    rc = m16_native_create_channel(&c->m); /* SUB@676f16f line 46 */
    c->last_rc = rc;
    return rc;
}

static int be_alloc(void *ctx, int role, size_t bytes)
{
    BeCtx *c = (BeCtx *)ctx;
    NvrmMem *m;
    uint64_t sz;
    begin(c);
    if (role == (int)OMEGA_GPU_BUF_SCRATCH) {
        if (alloc_checked(c, BE_PB_BYTES, &c->pb) != 0 ||
            alloc_checked(c, BE_PAGE, &c->cbank) != 0 ||
            alloc_checked(c, BE_PAGE, &c->marker) != 0 ||
            alloc_checked(c, BE_QMD_BYTES, &c->qmd) != 0) {
            int rc = c->last_rc ? c->last_rc : -1;
            int saved_errno = errno; /* the core reads errno right after this callback */
            char keep[OMEGA_GPU_ENGINE_DRV_TEXT_BYTES];
            copy_text(keep, sizeof keep, c->text);
            (void)release_scratch(c); /* the core does not free a role whose alloc failed */
            copy_text(c->text, sizeof c->text, keep);
            c->last_rc = rc;
            errno = saved_errno;
            return rc;
        }
        c->m.pb_mem = c->pb; /* the unified method stream needs the large ring, SUB@676f16f lines 49-51 */
        c->pb_replaced = 1;
        return 0;
    }
    m = role_mem(c, role);
    if (!m) {
        note(c, "alloc: unknown buffer role");
        return -1;
    }
    if (round_pages(bytes, &sz) != 0) {
        note(c, "alloc: byte count is zero or too large");
        return -1;
    }
    return alloc_checked(c, sz, m);
}

static int be_copy_in(void *ctx, int role, const void *src, size_t bytes)
{
    BeCtx *c = (BeCtx *)ctx;
    NvrmMem *m;
    begin(c);
    m = role_mem(c, role);
    if (!m || !m->cpu || !src || (uint64_t)bytes > m->size) {
        note(c, "copy_in: bad role, buffer or length");
        return -1;
    }
    memcpy(m->cpu, src, bytes); /* SUB@676f16f lines 74-76 (code via the encoder, inputs via memcpy) */
    return 0;
}

static int be_fill_poison(void *ctx, const uint32_t *words, size_t count)
{
    BeCtx *c = (BeCtx *)ctx;
    NvrmMem *out = &c->buf[OMEGA_GPU_BUF_OUTPUT];
    begin(c);
    if (!out->cpu || !words || (uint64_t)count * 4u > out->size) {
        note(c, "fill_poison: no output buffer or length too large");
        return -1;
    }
    memcpy(out->cpu, words, count * 4u); /* SUB@676f16f lines 77-80: pre-fill the device output */
    return 0;
}

/* Descriptors, QMDs and the whole pushbuffer. Lifted from SUB@676f16f lines
 * 83-194 and, for the C3 tail, divsqrt lines 1145-1152. Zeroes the marker and
 * semaphore words and issues dsb sy (SUB@676f16f lines 124-128). Does not submit. */
static int be_build(void *ctx, const OmegaGpuJob *job)
{
    BeCtx *c = (BeCtx *)ctx;
    const NvrmMem *code = &c->buf[OMEGA_GPU_BUF_PROGRAM];
    const NvrmMem *a = &c->buf[OMEGA_GPU_BUF_INPUT_A];
    const NvrmMem *b = &c->buf[OMEGA_GPU_BUF_INPUT_B];
    const NvrmMem *out = &c->buf[OMEGA_GPU_BUF_OUTPUT];
    uint32_t cbank_data[OMEGA_BW_CBANK_DRIVER_WORDS];
    uint32_t cbank_args[OMEGA_BW_CBANK_ARGS_WORDS];
    uint32_t qmd0_words[OMEGA_BW_QMD_WORDS];
    uint32_t qmd1_words[OMEGA_BW_QMD_WORDS];
    OmegaBlackwellQmdConfig cfg;
    uint64_t qmd0_va, qmd1_va, sem_va, marker_va, marker2_va, need;
    uint32_t n;
    int rc;

    begin(c);
    if (!job || job->layout != (int)OMEGA_GPU_LAYOUT_VECTOR_1D) {
        note(c, "build: only the VECTOR_1D layout is supported");
        return -1;
    }
    n = job->element_count;
    /* (n + 63) / 64 must not overflow; ASTRA refused n above UINT32_MAX - 63 as well. */
    if (n == 0 || n > UINT32_MAX - 63u) {
        note(c, "build: element count is zero or too large");
        return -1;
    }
    need = (uint64_t)n * 4u;
    if (!code->cpu || !a->cpu || !b->cpu || !out->cpu || !c->pb.cpu || !c->cbank.cpu ||
        !c->marker.cpu || !c->qmd.cpu || a->size < need || b->size < need || out->size < need) {
        note(c, "build: a buffer is missing or too small");
        return -1;
    }
    c->c3 = (job->flags & OMEGA_GPU_ENGINE_FLAG_NO_C3) == 0;

    qmd0_va = c->qmd.va;
    qmd1_va = c->qmd.va + 0x1000;
    sem_va = c->qmd.va + BE_SEM_OFFSET;
    marker_va = c->marker.va;
    marker2_va = c->marker.va + BE_MARKER2_OFFSET;

    memset(&cfg, 0, sizeof cfg);
    cfg.code_va = code->va;
    cfg.cbank_va = c->cbank.va;
    cfg.scratch_va = c->qmd.va + 0x4000; /* SUB@676f16f line 100 */
    cfg.sem_va = sem_va;
    cfg.qmd0_va = qmd0_va;
    cfg.qmd1_va = qmd1_va;
    cfg.num_elements = n;
    cfg.threads_per_block = 64;
    cfg.grid_width = (n + 63u) / 64u;

    /* The builders return 0 on success (src/omega_blackwell_qmd.c); SUB@676f16f
     * ignored their codes, ASTRA checked them. Checked here. */
    rc = omega_blackwell_build_cbank_driver(cbank_data, c->cbank.va);
    if (rc == 0) rc = omega_blackwell_build_cbank_args(cbank_args, a->va, b->va, out->va, n);
    if (rc == 0) rc = omega_blackwell_build_qmd0(qmd0_words, qmd0_va, qmd1_va);
    if (rc == 0) rc = omega_blackwell_build_qmd1(qmd1_words, &cfg);
    if (rc == 0) rc = omega_blackwell_verify_qmd_invariants(qmd1_words); /* SUB@676f16f line 118 */
    if (rc != 0) {
        c->last_rc = rc;
        note(c, "build: a descriptor builder or the QMD invariant check refused");
        return rc;
    }
    /* Extra QMD1 checks from ASTRA: release enable (word 9), the release
     * semaphore address and the DONE value (words 15-17), and the memory
     * barrier flags (word 19). The words come from src/omega_blackwell_qmd.c
     * lines 81, 93-98. */
    if (qmd1_words[9] != 3u || qmd1_words[15] != (uint32_t)sem_va ||
        qmd1_words[16] != (uint32_t)(sem_va >> 32) ||
        qmd1_words[17] != OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE ||
        qmd1_words[19] != 0x81810000u) {
        note(c, "build: QMD1 release words do not match the expected layout");
        return -1;
    }

    memcpy(c->cbank.cpu, cbank_data, sizeof cbank_data);                       /* SUB@676f16f line 112 */
    memcpy((uint8_t *)c->cbank.cpu + 0x380, cbank_args, sizeof cbank_args);    /* line 113 */
    memcpy(c->qmd.cpu, qmd0_words, sizeof qmd0_words);                         /* line 120 */
    memcpy((uint8_t *)c->qmd.cpu + 0x1000, qmd1_words, sizeof qmd1_words);     /* line 121 */

    /* Initialise the synchronisation words in coherent memory (lines 124-128;
     * divsqrt also zeroes the second marker, line 1113). Then dsb sy so the
     * zeros are visible to the chip before the doorbell. */
    c->hsem = (volatile uint32_t *)((uint8_t *)c->qmd.cpu + BE_SEM_OFFSET);
    c->hmarker = (volatile uint32_t *)c->marker.cpu;
    c->hmarker2 = (volatile uint32_t *)((uint8_t *)c->marker.cpu + BE_MARKER2_OFFSET);
    *c->hsem = 0;
    *c->hmarker = 0;
    *c->hmarker2 = 0;
    if (!c->c3) c->hmarker2 = NULL; /* never waited for, never reported */
    __asm__ volatile("dsb sy" ::: "memory");

    /* ---- unified pushbuffer, SUB@676f16f lines 130-195 ---- */
    c->pb_len = 0;
    c->pb_overflow = 0;

    /* 1. setup words (18) */
    pb_copy(c, BE_SETUP_WORDS, 18u);

    /* 2. driver constant bank upload (224 words by DMA) */
    pb_put(c, nvrm_mthd(1, 0x0188, 2));
    pb_put(c, (uint32_t)(c->cbank.va >> 32));
    pb_put(c, (uint32_t)c->cbank.va);
    pb_put(c, nvrm_mthd(1, 0x0180, 2));
    pb_put(c, 0x00000380);
    pb_put(c, 0x00000001);
    pb_put(c, nvrm_mthd(1, 0x01b0, 1));
    pb_put(c, 0x00000041);
    pb_put(c, (uint32_t)((224 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28)));
    pb_copy(c, cbank_data, 224u);

    /* 3. kernel argument upload (7 words by DMA) */
    pb_put(c, nvrm_mthd(1, 0x0188, 2));
    pb_put(c, (uint32_t)((c->cbank.va + 0x380) >> 32));
    pb_put(c, (uint32_t)(c->cbank.va + 0x380));
    pb_put(c, nvrm_mthd(1, 0x0180, 2));
    pb_put(c, 0x0000001c);
    pb_put(c, 0x00000001);
    pb_put(c, nvrm_mthd(1, 0x01b0, 1));
    pb_put(c, 0x00000041);
    pb_put(c, (uint32_t)((7 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28)));
    pb_copy(c, cbank_args, 7u);

    /* 4. inline QMD 0 (2 address words + 96 QMD words) */
    pb_put(c, (uint32_t)((98 << 16) | (1 << 13) | (0x0318 >> 2) | (2u << 28)));
    pb_put(c, (1u << 30) | (uint32_t)((qmd0_va >> 40) & 0x1ff));
    pb_put(c, (uint32_t)(qmd0_va >> 8));
    pb_copy(c, qmd0_words, 96u);

    /* 5. intermediate semaphore initial value (one word by DMA) */
    pb_put(c, nvrm_mthd(1, 0x0188, 2));
    pb_put(c, (uint32_t)(sem_va >> 32));
    pb_put(c, (uint32_t)sem_va);
    pb_put(c, nvrm_mthd(1, 0x0180, 2));
    pb_put(c, 0x00000004);
    pb_put(c, 0x00000001);
    pb_put(c, nvrm_mthd(1, 0x01b0, 1));
    pb_put(c, 0x00000041);
    pb_put(c, (uint32_t)((1 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28)));
    pb_put(c, OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT);

    /* 6. inline QMD 1 */
    pb_put(c, (uint32_t)((98 << 16) | (1 << 13) | (0x0318 >> 2) | (2u << 28)));
    pb_put(c, (1u << 30) | (uint32_t)((qmd1_va >> 40) & 0x1ff));
    pb_put(c, (uint32_t)(qmd1_va >> 8));
    pb_copy(c, qmd1_words, 96u);

    /* 7. subchannel 0 completion release (RELEASE | WFI) */
    pb_put(c, nvrm_mthd(0, 0x005c, 5));
    pb_put(c, (uint32_t)marker_va);
    pb_put(c, (uint32_t)(marker_va >> 32));
    pb_put(c, OMEGA_BW_MARKER_COMPLETION_PAYLOAD);
    pb_put(c, 0);
    pb_put(c, 0x1u | (1u << 20));

    /* 8. C3, divsqrt lines 1145-1152 (NVC96F_MEM_OP_A..D = 0x28..0x34, operation
     * L2_FLUSH_DIRTY 0x10 in bits 31:27; third_party/nvidia-open-580.173.02/src/
     * common/sdk/nvidia/inc/class/clc96f.h:36-73): flush the L2, then a second
     * WFI release at marker page + 0x10 that the host waits for before readback. */
    if (c->c3) {
        pb_put(c, nvrm_mthd(0, 0x0028, 4));
        pb_put(c, 0);
        pb_put(c, 0);
        pb_put(c, 0);
        pb_put(c, 0x10u << 27);
        pb_put(c, nvrm_mthd(0, 0x005c, 5));
        pb_put(c, (uint32_t)marker2_va);
        pb_put(c, (uint32_t)(marker2_va >> 32));
        pb_put(c, BE_MARKER2_PAYLOAD);
        pb_put(c, 0);
        pb_put(c, 0x1u | (1u << 20));
    }

    if (c->pb_overflow) {
        note(c, "build: pushbuffer exceeded its word capacity");
        return -1;
    }
    return 0;
}

/* Submit methods to the GPFIFO ring and ring the doorbell, SUB@676f16f lines
 * 198-200. m16_native_submit_methods rings only after the enqueue succeeded
 * (physics/m16/m16_native.c:76-84), so a non-zero return means the doorbell
 * was not rung: a known failure, not an uncertain completion. */
static int be_submit(void *ctx)
{
    BeCtx *c = (BeCtx *)ctx;
    int rc;
    begin(c);
    if (c->pb_len == 0) {
        note(c, "submit: nothing was built");
        return -1;
    }
    c->info.launch_ns = now_ns(); /* SUB@676f16f line 199 */
    rc = m16_native_submit_methods(&c->m, c->pb_words, c->pb_len);
    c->last_rc = rc;
    return rc;
}

/* Waits for one coherent word. m16_native_wait_marker is a ">=" serial compare
 * (physics/m16/m16_native.c:93-108), so an overshoot also returns 0; ASTRA
 * checked_alloc/checked_wait therefore re-checked equality, kept here and
 * reported as UNKNOWN_STATE (the device wrote a value we cannot classify).
 * Returns 0, OMEGA_GPU_BACKEND_TIMEOUT, OMEGA_GPU_BACKEND_UNKNOWN_STATE or a
 * driver fault. *snap always receives the last value read. */
static int wait_word(BeCtx *c, volatile uint32_t *w, uint32_t want, uint64_t ms, uint32_t *snap)
{
    int rc;
    begin(c);
    if (!w) {
        note(c, "wait: the word is not mapped (wait before build or after free)");
        return -2;
    }
    rc = m16_native_wait_marker(w, want, ms);
    *snap = *w;
    if (rc != 0) {
        c->last_rc = rc;
        return rc == -1 ? OMEGA_GPU_BACKEND_TIMEOUT : -2;
    }
    if (*snap != want) {
        (void)snprintf(c->text, sizeof c->text, "observed 0x%08x, wanted 0x%08x", *snap, want);
        return OMEGA_GPU_BACKEND_UNKNOWN_STATE;
    }
    return 0;
}

static int be_wait_marker(void *ctx, uint64_t timeout_ms)
{
    BeCtx *c = (BeCtx *)ctx;
    int rc = wait_word(c, c->hmarker, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, timeout_ms, &c->info.marker);
    if (rc == 0) c->info.marker_done_ns = now_ns(); /* SUB@676f16f line 204 */
    return rc;
}

static int be_wait_marker2(void *ctx, uint64_t timeout_ms)
{
    BeCtx *c = (BeCtx *)ctx;
    return wait_word(c, c->hmarker2, BE_MARKER2_PAYLOAD, timeout_ms, &c->info.marker2);
}

/* The QMD's own release semaphore, written after the grid finishes (divsqrt
 * lines 1163-1169). SUB@676f16f only compared it once (line 207); now it is waited for. */
static int be_wait_release_semaphore(void *ctx, uint64_t timeout_ms)
{
    BeCtx *c = (BeCtx *)ctx;
    return wait_word(c, c->hsem, OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE, timeout_ms,
                     &c->info.semaphore);
}

/* dsb sy before readback, divsqrt line 1171. Completion is already known here. */
static int be_barrier(void *ctx, uint64_t timeout_ms)
{
    BeCtx *c = (BeCtx *)ctx;
    (void)timeout_ms;
    begin(c);
    __asm__ volatile("dsb sy" ::: "memory");
    return 0;
}

static int be_copy_out(void *ctx, void *dst, size_t bytes)
{
    BeCtx *c = (BeCtx *)ctx;
    const NvrmMem *out = &c->buf[OMEGA_GPU_BUF_OUTPUT];
    begin(c);
    if (!out->cpu || !dst || (uint64_t)bytes > out->size) {
        note(c, "copy_out: no output buffer or length too large");
        return -1;
    }
    memcpy(dst, out->cpu, bytes); /* SUB@676f16f line 210 */
    return 0;
}

/* Driver code, driver text and the sync words as they are right now. Words are
 * read only while SCRATCH is still mapped (hmarker and friends are NULL before
 * build and after SCRATCH is freed), so this is safe during cleanup. */
static void be_diagnostics(void *ctx, OmegaGpuBackendDiag *d)
{
    BeCtx *c = (BeCtx *)ctx;
    d->drv_rc = c->last_rc;
    if (c->text[0] != '\0') copy_text(d->drv_text, sizeof d->drv_text, c->text);
    else if (c->native_open) copy_text(d->drv_text, sizeof d->drv_text, c->m.rm.err);
    if (c->hmarker) {
        d->marker = *c->hmarker;
        d->sync_valid |= OMEGA_GPU_SYNC_MARKER;
    }
    if (c->hmarker2) {
        d->marker2 = *c->hmarker2;
        d->sync_valid |= OMEGA_GPU_SYNC_MARKER2;
    }
    if (c->hsem) {
        d->semaphore[0] = *c->hsem;
        d->sync_valid |= OMEGA_GPU_SYNC_SEMAPHORE;
    }
}

/* Real nvrm_free per buffer. SUB@676f16f never freed (m16_native_close frees
 * everything left, nvrm.c:757-766); nvrm_free before close is already used by
 * src/omega_accelerator_world.c (A1b scout, section 4). DIVERGENCE, chip
 * behaviour of free-then-close for this launcher is UNVERIFIED (confidence 70%). */
static int be_free_buf(void *ctx, int role)
{
    BeCtx *c = (BeCtx *)ctx;
    NvrmMem *m;
    int rc;
    begin(c);
    if (role == (int)OMEGA_GPU_BUF_SCRATCH) {
        rc = release_scratch(c);
        c->last_rc = rc;
        return rc;
    }
    m = role_mem(c, role);
    if (!m) {
        note(c, "free: unknown buffer role");
        return -1;
    }
    rc = nvrm_free(&c->m.rm, m);
    c->last_rc = rc;
    return rc;
}

static int be_close(void *ctx)
{
    BeCtx *c = (BeCtx *)ctx;
    int rc;
    begin(c);
    c->hmarker = NULL;
    c->hmarker2 = NULL;
    c->hsem = NULL;
    rc = m16_native_close(&c->m); /* SUB@676f16f line 225 */
    c->last_rc = rc;
    atomic_flag_clear(&g_busy);
    return rc;
}

static const OmegaGpuBackend g_backend = {
    &g_be,
    be_open_device,
    be_create_channel,
    be_alloc,
    be_copy_in,
    be_fill_poison,
    be_build,
    be_submit,
    be_wait_marker,
    be_wait_marker2,
    be_wait_release_semaphore,
    be_barrier,
    be_copy_out,
    be_diagnostics,
    be_free_buf,
    be_close,
};

const OmegaGpuBackend *omega_blackwell_engine_backend(void)
{
    return &g_backend;
}

void omega_blackwell_engine_run_info(OmegaBlackwellEngineRunInfo *out)
{
    if (out) *out = g_be.info;
}
