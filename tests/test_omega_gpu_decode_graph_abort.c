/* test_omega_gpu_decode_graph_abort: lock and token-state recovery of the decode graph's split API (omega #518) on a
 * FAKE OmegaGpuPlatform. Host program, links libomega_gpu.a with the fake installed first; opens no device, touches
 * no GPU.
 *
 * The fake's session lock REFUSES a second acquisition while held (counted in fk.double_locks) instead of blocking,
 * so a lock left held by a failed call shows up as a count, not a hung test. The fake device never runs a kernel: a
 * submitted list's markers never land (the cmdlist times out on the fake clock), and the graph's token slot is written
 * by the "device" at the gpu->cpu sync only when a case arms fk.dev_token (a token the device produced).
 *
 * Contract under test (omega_gpu_decode_graph.h): a failed begin_token returns with the session lock released and the
 * graph out of the token (no end_token owed); abort_token drops a begun token without submitting anything; end_token
 * after a failed record submits nothing and releases the lock; an uncertain completion latches the process and every
 * later begin fails without a retry or a submission. After each failure: set_final_norm, read, free and the next
 * valid token neither deadlock nor see a stale token state.
 *
 * Each case runs in its own child process (a latch lasts for the life of the process).
 */
#include "omega_gpu_decode_graph.h"
#include "omega_gpu_session.h"
#include "omega_gpu_platform.h"
#include "omega_blackwell_codegen_dec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int case_ok;
#define EXPECT(c) do { if (!(c)) { case_ok = 0; printf("  expect failed at line %d: %s\n", __LINE__, #c); } } while (0)

/* ---- the fake platform ---- */
#define NB OMEGA_GPU_HANDLE_SLOTS
#define VA_BASE 0x100000000ull
#define VA_STRIDE 0x1000000ull
static struct {
    OmegaGpuHandleTable t;
    void *mem[NB];
    OmegaGpuBufferDesc desc[NB];
    OmegaGpuDeviceHandle dev;
    OmegaGpuQueueHandle q;
    OmegaGpuBufferHandle ring;
    uint64_t ns;
    unsigned locks_held, double_locks, bad_unlocks;
    unsigned submits, dev_opens;
    int fail_open, fail_c2g;  /* refuse device open / every cpu->gpu visibility sync */
    uint64_t dev_token_va;    /* nonzero: the device wrote dev_token there (seen at the gpu->cpu sync) */
    uint32_t dev_token;
} fk;

static uint64_t f_now_ns(void *c) { (void)c; fk.ns += 1000u; return fk.ns; }
static uint64_t f_now_ms(void *c) { return f_now_ns(c) / 1000000u; }
static int f_wait(void *c, uint64_t ms) { (void)c; fk.ns += ms * 1000000u; return 0; }
static int f_lock(void *c, uint32_t id) {
    (void)c;
    if (id != OMEGA_GPU_SESSION_LOCK_ID) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    if (fk.locks_held) { fk.double_locks++; return OMEGA_GPU_PLAT_REFUSED_BUSY; } /* a real lock would block here forever */
    fk.locks_held = 1;
    return 0;
}
static int f_unlock(void *c, uint32_t id) {
    (void)c;
    if (id != OMEGA_GPU_SESSION_LOCK_ID || !fk.locks_held) { fk.bad_unlocks++; return OMEGA_GPU_PLAT_REFUSED_ARGS; }
    fk.locks_held = 0;
    return 0;
}
static const char *f_diag(void *c) { (void)c; return fk.fail_open ? "fake device: open refused" : ""; }

static int f_dev_open(void *c, uint32_t ord, OmegaGpuDeviceHandle *out) {
    (void)c;
    if (fk.fail_open) return OMEGA_GPU_PLAT_REFUSED_DEVICE;
    if (ord != 0 || fk.dev.generation) return OMEGA_GPU_PLAT_REFUSED_LIMIT;
    int rc = omega_gpu_handles_open(&fk.t, OMEGA_GPU_HANDLE_DEVICE, 0, 1, 0, &out->index, &out->generation);
    if (rc == 0) { fk.dev = *out; fk.dev_opens++; }
    return rc;
}
static int f_dev_close(void *c, OmegaGpuDeviceHandle d) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    if (rc) return rc;
    /* every handle of the device goes stale; the host memory stays mapped (leaked) so a graph that outlives the device
     * writes into its own stale buffer, not freed memory: this test checks the lock, not use after close */
    for (uint32_t i = 0; i < NB; i++)
        if (fk.t.live[i] && i != d.index) { (void)omega_gpu_handles_close(&fk.t, i, fk.t.generation[i], fk.t.kind[i]); fk.mem[i] = NULL; }
    (void)omega_gpu_handles_close(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    memset(&fk.dev, 0, sizeof fk.dev); memset(&fk.q, 0, sizeof fk.q);
    return 0;
}

static OmegaGpuPlatformCaps g_caps = {
    .abi_major = OMEGA_GPU_PLATFORM_ABI_MAJOR, .abi_minor = OMEGA_GPU_PLATFORM_ABI_MINOR,
    .services = OMEGA_GPU_SVC_REQUIRED_SESSION, .max_devices = 1, .max_buffers = 1000, .max_queues = 1, .max_inflight = 1,
    .max_buffer_bytes = 1ull << 30, .max_total_bytes = 1ull << 32,
};

static int f_alloc(void *c, OmegaGpuDeviceHandle d, const OmegaGpuBufferDesc *desc, OmegaGpuBufferHandle *o,
                   OmegaCpuView *cpu, OmegaGpuVa *g, OmegaGpuBusAddr *b) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    if (!rc) rc = omega_gpu_platform_check_buffer(desc, &g_caps);
    if (!rc) rc = omega_gpu_handles_open(&fk.t, desc->kind, desc->bytes, 1000, 1ull << 32, &o->index, &o->generation);
    if (rc) return rc;
    void *m = calloc(1, desc->bytes);
    if (!m) { (void)omega_gpu_handles_close(&fk.t, o->index, o->generation, desc->kind); return OMEGA_GPU_PLAT_REFUSED_LIMIT; }
    fk.mem[o->index] = m; fk.desc[o->index] = *desc;
    cpu->ptr = m; cpu->len = desc->bytes; g->va = VA_BASE + (uint64_t)o->index * VA_STRIDE; b->iova = 0;
    return 0;
}
static int f_free(void *c, OmegaGpuBufferHandle h) {
    (void)c;
    if (h.index >= NB) return OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE;
    int rc = omega_gpu_handles_close(&fk.t, h.index, h.generation, fk.t.kind[h.index]);
    if (rc) return rc;
    free(fk.mem[h.index]); fk.mem[h.index] = NULL;
    return 0;
}
static int buf_live(OmegaGpuBufferHandle h) { return h.index < NB && fk.t.live[h.index] && fk.t.generation[h.index] == h.generation; }
static void *va_to_cpu(uint64_t va) {
    for (uint32_t i = 0; i < NB; i++) {
        if (!fk.mem[i] || !fk.t.live[i]) continue;
        const uint64_t base = VA_BASE + (uint64_t)i * VA_STRIDE;
        if (va >= base && va < base + fk.desc[i].bytes) return (uint8_t *)fk.mem[i] + (va - base);
    }
    return NULL;
}
static int sync_check(OmegaGpuBufferHandle h, uint64_t off, uint64_t len) {
    if (!buf_live(h)) return OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE;
    if (len == 0 || off > fk.desc[h.index].bytes || len > fk.desc[h.index].bytes - off) return OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH;
    return 0;
}
static int f_c2g(void *c, OmegaGpuBufferHandle h, uint64_t off, uint64_t len) {
    (void)c;
    if (fk.fail_c2g) return OMEGA_GPU_PLAT_REFUSED_DEVICE;
    return sync_check(h, off, len);
}
static int f_g2c(void *c, OmegaGpuBufferHandle h, uint64_t off, uint64_t len) {
    (void)c;
    int rc = sync_check(h, off, len);
    if (rc == 0 && fk.dev_token_va) { uint32_t *t = va_to_cpu(fk.dev_token_va); if (t) t[0] = fk.dev_token; }
    return rc;
}
static int f_poll(void *c, OmegaGpuBufferHandle h, uint64_t off, uint32_t want, uint64_t ms, uint32_t *seen) {
    (void)c;
    if (!buf_live(h)) return OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE;
    if (!(fk.desc[h.index].rights & OMEGA_GPU_RIGHT_HOST_POLLED)) return OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS;
    const volatile uint32_t *w = (const volatile uint32_t *)((uint8_t *)fk.mem[h.index] + off);
    *seen = *w;
    if ((int32_t)(*w - want) >= 0) return 0;
    fk.ns += ms * 1000000u;
    return OMEGA_GPU_POLL_TIMED_OUT;
}
static int f_qopen(void *c, OmegaGpuDeviceHandle d, OmegaGpuBufferHandle ring, OmegaGpuQueueHandle *o) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    if (!rc && (!buf_live(ring) || fk.t.kind[ring.index] != OMEGA_GPU_HANDLE_BUF_QUEUE)) rc = OMEGA_GPU_PLAT_REFUSED_BAD_KIND;
    if (!rc) rc = omega_gpu_handles_open(&fk.t, OMEGA_GPU_HANDLE_QUEUE, 0, 1, 0, &o->index, &o->generation);
    if (!rc) { fk.q = *o; fk.ring = ring; }
    return rc;
}
static int f_qclose(void *c, OmegaGpuQueueHandle q) {
    (void)c;
    int rc = omega_gpu_handles_close(&fk.t, q.index, q.generation, OMEGA_GPU_HANDLE_QUEUE);
    if (!rc) memset(&fk.q, 0, sizeof fk.q);
    return rc;
}
/* The fake device accepts the words and runs nothing: no marker, semaphore or output ever lands. */
static int f_submit(void *c, OmegaGpuQueueHandle q, const uint32_t *w, uint32_t n) {
    (void)c; (void)w;
    int rc = omega_gpu_handles_check(&fk.t, q.index, q.generation, OMEGA_GPU_HANDLE_QUEUE);
    if (rc) return rc;
    if ((uint64_t)n * 4u > fk.desc[fk.ring.index].bytes) return OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH;
    fk.submits++;
    return 0;
}
static int f_retire(void *c, OmegaGpuQueueHandle q) { (void)c; (void)q; return 0; }
static void *f_halloc(void *c, size_t b) { (void)c; return malloc(b); }
static void f_hfree(void *c, void *p) { (void)c; free(p); }

static OmegaGpuPlatform g_plat = {
    .ctx = &fk, .time_now_ms = f_now_ms, .wait_ms = f_wait, .lock = f_lock, .unlock = f_unlock,
    .dma_alloc = f_alloc, .dma_free = f_free, .sync_cpu_to_gpu = f_c2g, .sync_gpu_to_cpu = f_g2c, .poll_word = f_poll,
    .device_open = f_dev_open, .device_close = f_dev_close, .diag_text = f_diag,
    .queue_open = f_qopen, .queue_close = f_qclose, .queue_submit = f_submit, .queue_retire = f_retire,
    .time_now_ns = f_now_ns, .host_alloc = f_halloc, .host_free = f_hfree,
};

/* ---- a one-layer graph on the fake ---- */
#define LAYERS 1u
#define VOCAB 256u
#define MAX_POS 256u
typedef struct {
    OmegaGpuDecodeAttnCache *attn;
    OmegaGpuRouter *router;
    OmegaGpuBf16Tensor *lm;
    OmegaGpuDecodeGraph *g;
} World;

static int world_build(World *w) {
    memset(w, 0, sizeof *w);
    static uint16_t lm_w[VOCAB * OMEGA_GPU_DGRAPH_HIDDEN]; /* zeros: finite bf16 */
    if (omega_gpu_decode_attn_cache_alloc(LAYERS, MAX_POS, 1e6, &w->attn) != 0) return -1;
    if (omega_gpu_router_alloc(LAYERS, &w->router) != 0) return -1;
    if (omega_gpu_bf16_tensor_upload(VOCAB, OMEGA_GPU_DGRAPH_HIDDEN, lm_w, &w->lm) != 0) return -1;
    OmegaGpuDecodeGraphConfig cfg; memset(&cfg, 0, sizeof cfg);
    cfg.layers = LAYERS; cfg.vocab = VOCAB; cfg.eps = 1e-6f; cfg.norm_topk = 1;
    cfg.router = w->router; cfg.attn = w->attn; cfg.lm_head = w->lm;
    return omega_gpu_decode_graph_create(&cfg, &w->g) == OMEGA_GPU_DGRAPH_OK ? 0 : -1;
}
static void world_free(World *w) {
    omega_gpu_decode_graph_free(w->g);
    omega_gpu_bf16_tensor_free(w->lm);
    omega_gpu_router_free(w->router);
    omega_gpu_decode_attn_cache_free(w->attn);
    memset(w, 0, sizeof *w);
}

static float g_x[OMEGA_GPU_DGRAPH_HIDDEN], g_norm[OMEGA_GPU_DGRAPH_HIDDEN];

/* The postcondition of every failed or finished call: lock free, never taken twice, never released twice. */
#define EXPECT_LOCK_FREE() do { EXPECT(fk.locks_held == 0); EXPECT(fk.double_locks == 0); EXPECT(fk.bad_unlocks == 0); } while (0)

/* Not in a token any more: the between-token calls run (BAD_ARGS here would mean in_token stuck). */
static void expect_between_tokens(World *w) {
    EXPECT(omega_gpu_decode_graph_set_final_norm(w->g, g_norm) == OMEGA_GPU_DGRAPH_OK);
    EXPECT_LOCK_FREE();
    float x[OMEGA_GPU_DGRAPH_HIDDEN];
    EXPECT(omega_gpu_decode_graph_read(w->g, x, NULL, NULL) == OMEGA_GPU_DGRAPH_OK);
    EXPECT_LOCK_FREE();
}

/* A valid split-API token: begin, nothing recorded, end; the fake device "wrote" the token. */
static void expect_valid_token(World *w, uint32_t want) {
    OmegaGpuDecodeGraphVas v; omega_gpu_decode_graph_vas(w->g, &v);
    fk.dev_token_va = v.token; fk.dev_token = want;
    const unsigned submits = fk.submits;
    EXPECT(omega_gpu_decode_graph_begin_token(w->g, g_x) == OMEGA_GPU_DGRAPH_OK);
    EXPECT(fk.locks_held == 1);
    uint32_t tok = 0xdeadu;
    OmegaGpuDecodeGraphStats st; memset(&st, 0xff, sizeof st);
    EXPECT(omega_gpu_decode_graph_end_token(w->g, &tok, &st) == OMEGA_GPU_DGRAPH_OK);
    EXPECT(tok == want && st.launches == 0 && st.submits == 0);
    EXPECT(fk.submits == submits); /* an empty list submits nothing */
    EXPECT_LOCK_FREE();
    fk.dev_token_va = 0;
}

static int install(void) {
    memset(&fk, 0, sizeof fk); omega_gpu_handles_init(&fk.t);
    for (unsigned i = 0; i < OMEGA_GPU_DGRAPH_HIDDEN; i++) { g_x[i] = (float)(i % 7) - 3.0f; g_norm[i] = 1.0f; }
    return omega_gpu_session_set_platform(&g_plat, &g_caps);
}

/* Visibility sync (cpu->gpu of x and the poisoned token) refused: split and one-shot paths, repeated, then recovery. */
static void c_sync_fails(void) {
    World w;
    if (install() != OMEGA_GPU_PLAT_OK) { EXPECT(!"fake platform installs"); return; } /* never fall back to the real driver */
    if (world_build(&w) != 0) { EXPECT(!"world builds on the fake"); return; }
    EXPECT_LOCK_FREE();
    const unsigned submits = fk.submits;
    fk.fail_c2g = 1;
    for (int rep = 0; rep < 3; rep++) { /* same outcome every time */
        EXPECT(omega_gpu_decode_graph_begin_token(w.g, g_x) == OMEGA_GPU_DGRAPH_CHIP_FAIL);
        EXPECT_LOCK_FREE();
    }
    /* nothing owed after a failed begin: end and abort see no token */
    uint32_t tok = 7;
    EXPECT(omega_gpu_decode_graph_end_token(w.g, &tok, NULL) == OMEGA_GPU_DGRAPH_BAD_ARGS && tok == 7);
    EXPECT(omega_gpu_decode_graph_abort_token(w.g) == OMEGA_GPU_DGRAPH_BAD_ARGS);
    EXPECT(omega_gpu_decode_graph_record_final(w.g) == OMEGA_GPU_DGRAPH_BAD_ARGS);
    EXPECT_LOCK_FREE();
    /* one-shot: the begin failure comes back as CHIP_FAIL with zeroed stats, lock free */
    OmegaGpuDecodeGraphStats st; memset(&st, 0xff, sizeof st);
    tok = 7;
    EXPECT(omega_gpu_decode_graph_token(w.g, g_x, 0, &tok, &st) == OMEGA_GPU_DGRAPH_CHIP_FAIL);
    EXPECT(tok == 7 && st.launches == 0 && st.submits == 0 && st.total_ns == 0);
    EXPECT_LOCK_FREE();
    EXPECT(fk.submits == submits); /* nothing reached the device, nothing retried */
    EXPECT(!omega_gpu_session_is_blocked());
    fk.fail_c2g = 0;
    expect_between_tokens(&w);
    expect_valid_token(&w, 42u);
    expect_valid_token(&w, 43u); /* and again */
    world_free(&w);
    EXPECT_LOCK_FREE();
}

/* Device open refused: the session was closed (every handle stale), the reopen in begin_token fails. */
static void c_open_fails(void) {
    World w;
    if (install() != OMEGA_GPU_PLAT_OK) { EXPECT(!"fake platform installs"); return; } /* never fall back to the real driver */
    if (world_build(&w) != 0) { EXPECT(!"world builds on the fake"); return; }
    omega_gpu_session_close();
    EXPECT_LOCK_FREE();
    fk.fail_open = 1;
    for (int rep = 0; rep < 3; rep++) {
        EXPECT(omega_gpu_decode_graph_begin_token(w.g, g_x) == OMEGA_GPU_DGRAPH_CHIP_FAIL);
        EXPECT_LOCK_FREE();
    }
    EXPECT(omega_gpu_decode_graph_token(w.g, g_x, 0, NULL, NULL) == OMEGA_GPU_DGRAPH_CHIP_FAIL);
    EXPECT_LOCK_FREE();
    EXPECT(omega_gpu_decode_graph_end_token(w.g, NULL, NULL) == OMEGA_GPU_DGRAPH_BAD_ARGS);
    /* the graph's buffers belong to the closed device: set_final_norm and read reach the stale sync and refuse (not
     * BAD_ARGS: the graph is out of the token), nothing hangs */
    EXPECT(omega_gpu_decode_graph_set_final_norm(w.g, g_norm) == OMEGA_GPU_DGRAPH_CHIP_FAIL);
    EXPECT(omega_gpu_decode_graph_read(w.g, NULL, NULL, NULL) == OMEGA_GPU_DGRAPH_CHIP_FAIL);
    EXPECT_LOCK_FREE();
    fk.fail_open = 0;
    /* reopen succeeds; the old graph's buffer is stale, so its begin fails at the visibility sync, lock free */
    EXPECT(omega_gpu_decode_graph_begin_token(w.g, g_x) == OMEGA_GPU_DGRAPH_CHIP_FAIL);
    EXPECT_LOCK_FREE();
    world_free(&w);
    EXPECT_LOCK_FREE();
    /* a fresh graph on the reopened device runs a valid token */
    if (world_build(&w) != 0) { EXPECT(!"world rebuilds after the reopen"); return; }
    expect_valid_token(&w, 5u);
    world_free(&w);
    EXPECT_LOCK_FREE();
}

/* The caller abandons a begun token (cancellation): nothing submitted, lock released, the next token runs. */
static void c_abort(void) {
    World w;
    if (install() != OMEGA_GPU_PLAT_OK) { EXPECT(!"fake platform installs"); return; } /* never fall back to the real driver */
    if (world_build(&w) != 0) { EXPECT(!"world builds on the fake"); return; }
    OmegaGpuDecodeGraphVas v; omega_gpu_decode_graph_vas(w.g, &v);
    const unsigned submits = fk.submits;
    /* abort after a recorded launch */
    EXPECT(omega_gpu_decode_graph_begin_token(w.g, g_x) == OMEGA_GPU_DGRAPH_OK);
    EXPECT(omega_gpu_decode_graph_record_res(w.g, OMEGA_DEC_RES_NORM, v.x, 0, 0, v.x, v.xq, 1e-6f, 1.0f) == OMEGA_GPU_DGRAPH_OK);
    EXPECT(omega_gpu_decode_graph_abort_token(w.g) == OMEGA_GPU_DGRAPH_OK);
    EXPECT_LOCK_FREE();
    EXPECT(fk.submits == submits);
    EXPECT(omega_gpu_decode_graph_abort_token(w.g) == OMEGA_GPU_DGRAPH_BAD_ARGS); /* only once */
    EXPECT(omega_gpu_decode_graph_end_token(w.g, NULL, NULL) == OMEGA_GPU_DGRAPH_BAD_ARGS);
    EXPECT_LOCK_FREE();
    /* abort after a failed record */
    EXPECT(omega_gpu_decode_graph_begin_token(w.g, g_x) == OMEGA_GPU_DGRAPH_OK);
    EXPECT(omega_gpu_decode_graph_record_res(w.g, 0xffu, v.x, 0, 0, 0, v.xq, 1e-6f, 1.0f) == OMEGA_GPU_DGRAPH_BAD_ARGS);
    EXPECT(omega_gpu_decode_graph_abort_token(w.g) == OMEGA_GPU_DGRAPH_OK);
    EXPECT_LOCK_FREE();
    EXPECT(fk.submits == submits);
    /* abort with nothing begun */
    EXPECT(omega_gpu_decode_graph_abort_token(NULL) == OMEGA_GPU_DGRAPH_BAD_ARGS);
    expect_between_tokens(&w);
    expect_valid_token(&w, 9u);
    world_free(&w);
    EXPECT_LOCK_FREE();
}

/* end_token after a failed record: no submit, the failure returned, lock released, the next token runs. */
static void c_end_after_record_fail(void) {
    World w;
    if (install() != OMEGA_GPU_PLAT_OK) { EXPECT(!"fake platform installs"); return; } /* never fall back to the real driver */
    if (world_build(&w) != 0) { EXPECT(!"world builds on the fake"); return; }
    OmegaGpuDecodeGraphVas v; omega_gpu_decode_graph_vas(w.g, &v);
    const unsigned submits = fk.submits;
    EXPECT(omega_gpu_decode_graph_begin_token(w.g, g_x) == OMEGA_GPU_DGRAPH_OK);
    EXPECT(omega_gpu_decode_graph_record_res(w.g, OMEGA_DEC_RES_NORM, v.x, 0, 0, v.x, v.xq, 1e-6f, 1.0f) == OMEGA_GPU_DGRAPH_OK);
    EXPECT(omega_gpu_decode_graph_record_layer(w.g, 0, 0) == OMEGA_GPU_DGRAPH_BAD_ARGS); /* layer 0 never set */
    EXPECT(omega_gpu_decode_graph_record_final(w.g) == OMEGA_GPU_DGRAPH_CHIP_FAIL);     /* failed sticks */
    uint32_t tok = 7;
    EXPECT(omega_gpu_decode_graph_end_token(w.g, &tok, NULL) == OMEGA_GPU_DGRAPH_CHIP_FAIL && tok == 7);
    EXPECT(fk.submits == submits);
    EXPECT_LOCK_FREE();
    /* one-shot with an unset layer: the same, deterministically */
    for (int rep = 0; rep < 2; rep++) {
        EXPECT(omega_gpu_decode_graph_token(w.g, g_x, 0, &tok, NULL) == OMEGA_GPU_DGRAPH_CHIP_FAIL && tok == 7);
        EXPECT(fk.submits == submits);
        EXPECT_LOCK_FREE();
    }
    expect_between_tokens(&w);
    expect_valid_token(&w, 11u);
    world_free(&w);
    EXPECT_LOCK_FREE();
}

/* An uncertain completion: the submitted list never completes, the process latches. end_token reports the failure (never
 * a token); every later begin fails at once with the lock released, submits nothing and retries nothing. */
static void c_latched(void) {
    World w;
    if (install() != OMEGA_GPU_PLAT_OK) { EXPECT(!"fake platform installs"); return; } /* never fall back to the real driver */
    if (world_build(&w) != 0) { EXPECT(!"world builds on the fake"); return; }
    OmegaGpuDecodeGraphVas v; omega_gpu_decode_graph_vas(w.g, &v);
    fk.dev_token_va = v.token; fk.dev_token = 3u; /* even a token in the slot must not be claimed after a timeout */
    EXPECT(omega_gpu_decode_graph_begin_token(w.g, g_x) == OMEGA_GPU_DGRAPH_OK);
    EXPECT(omega_gpu_decode_graph_record_res(w.g, OMEGA_DEC_RES_NORM, v.x, 0, 0, v.x, v.xq, 1e-6f, 1.0f) == OMEGA_GPU_DGRAPH_OK);
    uint32_t tok = 7;
    EXPECT(omega_gpu_decode_graph_end_token(w.g, &tok, NULL) == OMEGA_GPU_DGRAPH_CHIP_FAIL && tok == 7);
    EXPECT(fk.submits >= 1);
    EXPECT(omega_gpu_session_is_blocked());
    EXPECT_LOCK_FREE();
    const unsigned submits = fk.submits, opens = fk.dev_opens;
    for (int rep = 0; rep < 3; rep++) {
        EXPECT(omega_gpu_decode_graph_begin_token(w.g, g_x) == OMEGA_GPU_DGRAPH_CHIP_FAIL);
        EXPECT_LOCK_FREE();
        EXPECT(omega_gpu_decode_graph_token(w.g, g_x, 0, &tok, NULL) == OMEGA_GPU_DGRAPH_CHIP_FAIL && tok == 7);
        EXPECT_LOCK_FREE();
    }
    EXPECT(fk.submits == submits && fk.dev_opens == opens); /* no retry, no reopen */
    EXPECT(omega_gpu_decode_graph_end_token(w.g, NULL, NULL) == OMEGA_GPU_DGRAPH_BAD_ARGS);
    EXPECT(omega_gpu_decode_graph_abort_token(w.g) == OMEGA_GPU_DGRAPH_BAD_ARGS);
    EXPECT(omega_gpu_decode_graph_set_final_norm(w.g, g_norm) == OMEGA_GPU_DGRAPH_OK);
    (void)omega_gpu_decode_graph_read(w.g, NULL, NULL, NULL); /* latched: refused or parked, but it returns */
    EXPECT_LOCK_FREE();
    world_free(&w);
    EXPECT_LOCK_FREE();
    EXPECT(omega_gpu_session_is_blocked());
}

typedef struct { const char *name; void (*fn)(void); } Case;
static const Case CASES[] = {
    { "visibility sync refused at begin: split + one-shot recover", c_sync_fails },
    { "device open refused at begin: lock free, reopen recovers", c_open_fails },
    { "abort_token: cancel a begun token, nothing submitted", c_abort },
    { "end_token after a failed record: no submit, lock free", c_end_after_record_fail },
    { "latched session: begin fails at once, no retry", c_latched },
};

int main(int argc, char **argv) {
    const unsigned n = sizeof CASES / sizeof CASES[0];
    if (argc == 2) { /* one case in this process (debugging): its index */
        const unsigned i = (unsigned)atoi(argv[1]);
        if (i >= n) return 2;
        case_ok = 1; CASES[i].fn();
        printf("%s %s\n", case_ok ? "ok  " : "FAIL", CASES[i].name);
        return case_ok ? 0 : 1;
    }
    unsigned failed = 0;
    for (unsigned i = 0; i < n; i++) {
        fflush(stdout);
        pid_t pid = fork();
        if (pid == 0) { case_ok = 1; CASES[i].fn(); fflush(stdout); _exit(case_ok ? 0 : 1); }
        int status = 0;
        int ok = pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        printf("%s %s\n", ok ? "ok  " : "FAIL", CASES[i].name);
        if (!ok) failed++;
    }
    printf("test_omega_gpu_decode_graph_abort: %u of %u cases passed: %s\n", n - failed, n, failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
