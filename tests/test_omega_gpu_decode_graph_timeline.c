/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* test_omega_gpu_decode_graph_timeline: the decode graph's per-token timeline never writes past stamp[] (omega #456).
 * Host program on a FAKE OmegaGpuPlatform that completes every submission (markers, launch semaphores, stamp reports);
 * links libomega_gpu.a with the fake installed first; opens no device, touches no GPU.
 *
 * The timeline holds OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1 stamps. Batched, a token's list never holds more launches than
 * that. With flush_each (one submission per launch) the list is emptied after every launch, so nothing in the command
 * list bounds the number of launches a token records: before #456 the 4097th stamped launch wrote stamp[4097], past the
 * end of the graph. Contract under test: 4096 launches fill the timeline exactly (stamps() returns 4096); one more fails
 * the token (CHIP_FAIL, timeline refused, lock released) and the next token runs normally.
 *
 * Each case runs in its own child process.
 */
#include "omega_gpu_decode_graph.h"
#include "omega_gpu_session.h"
#include "omega_gpu_platform.h"
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_submit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int case_ok;
#define EXPECT(c) do { if (!(c)) { case_ok = 0; printf("  expect failed at line %d: %s\n", __LINE__, #c); } } while (0)

/* ---- the fake platform: every submitted word stream "runs" ---- */
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
    unsigned locks_held, double_locks, bad_unlocks, submits;
    uint64_t dev_token_va; /* nonzero: the device wrote dev_token there (seen at the gpu->cpu sync) */
    uint32_t dev_token;
} fk;

static uint64_t f_now_ns(void *c) { (void)c; fk.ns += 1000u; return fk.ns; }
static uint64_t f_now_ms(void *c) { return f_now_ns(c) / 1000000u; }
static int f_wait(void *c, uint64_t ms) { (void)c; fk.ns += ms * 1000000u; return 0; }
static int f_lock(void *c, uint32_t id) {
    (void)c;
    if (id != OMEGA_GPU_SESSION_LOCK_ID) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    if (fk.locks_held) { fk.double_locks++; return OMEGA_GPU_PLAT_REFUSED_BUSY; } /* a real lock would block forever */
    fk.locks_held = 1;
    return 0;
}
static int f_unlock(void *c, uint32_t id) {
    (void)c;
    if (id != OMEGA_GPU_SESSION_LOCK_ID || !fk.locks_held) { fk.bad_unlocks++; return OMEGA_GPU_PLAT_REFUSED_ARGS; }
    fk.locks_held = 0;
    return 0;
}
static const char *f_diag(void *c) { (void)c; return ""; }
static int f_dev_open(void *c, uint32_t ord, OmegaGpuDeviceHandle *out) {
    (void)c;
    if (ord != 0 || fk.dev.generation) return OMEGA_GPU_PLAT_REFUSED_LIMIT;
    int rc = omega_gpu_handles_open(&fk.t, OMEGA_GPU_HANDLE_DEVICE, 0, 1, 0, &out->index, &out->generation);
    if (rc == 0) fk.dev = *out;
    return rc;
}
static int f_dev_close(void *c, OmegaGpuDeviceHandle d) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    if (rc) return rc;
    (void)omega_gpu_handles_close(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    memset(&fk.dev, 0, sizeof fk.dev);
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
static int f_c2g(void *c, OmegaGpuBufferHandle h, uint64_t off, uint64_t len) { (void)c; return sync_check(h, off, len); }
/* gpu->cpu: a command list's launch semaphores (16-byte slots, every one still INIT) read DONE: the launches ran.
 * Only a range whose every 16-byte slot holds INIT is taken for a semaphore array. */
static int f_g2c(void *c, OmegaGpuBufferHandle h, uint64_t off, uint64_t len) {
    (void)c;
    int rc = sync_check(h, off, len);
    if (rc) return rc;
    if (len % 16u == 0) {
        uint32_t *w = (uint32_t *)((uint8_t *)fk.mem[h.index] + off);
        uint64_t i = 0, n = len / 16u;
        while (i < n && w[4 * i] == OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT) i++;
        if (i == n) for (i = 0; i < n; i++) w[4 * i] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE;
    }
    if (fk.dev_token_va) { uint32_t *t = va_to_cpu(fk.dev_token_va); if (t) t[0] = fk.dev_token; }
    return 0;
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
static uint32_t mthd(uint32_t subc, uint32_t m, uint32_t count) { return (2u << 28) | (count << 16) | (subc << 13) | (m >> 2); }
/* The fake device runs the stream: semaphore releases (0x005c: the completion markers) write their payload, compute
 * report semaphores (0x0158, four-word: the stamps) write {payload, timestamp}. */
static int f_submit(void *c, OmegaGpuQueueHandle q, const uint32_t *w, uint32_t n) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, q.index, q.generation, OMEGA_GPU_HANDLE_QUEUE);
    if (rc) return rc;
    if ((uint64_t)n * 4u > fk.desc[fk.ring.index].bytes) return OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH;
    fk.submits++;
    for (uint32_t i = 0; i + 5 < n; i++) {
        if (w[i] == mthd(0, 0x005c, 5)) {
            uint32_t *m = va_to_cpu((uint64_t)w[i + 1] | ((uint64_t)w[i + 2] << 32));
            if (m) *m = w[i + 3];
            i += 5;
        } else if (w[i] == mthd(1, 0x0158, 5)) {
            uint64_t *s = va_to_cpu((uint64_t)w[i + 3] | ((uint64_t)w[i + 4] << 32));
            if (s) { s[0] = (uint64_t)w[i + 1] | ((uint64_t)w[i + 2] << 32); s[1] = (fk.ns += 1000u); }
            i += 5;
        }
    }
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
    if (omega_gpu_decode_graph_create(&cfg, &w->g) != OMEGA_GPU_DGRAPH_OK) return -1;
    omega_gpu_decode_graph_set_stamps(w->g, 1);
    omega_gpu_decode_graph_set_flush_each(w->g, 1); /* one submission per launch: the list never bounds the token */
    return 0;
}

static float g_x[OMEGA_GPU_DGRAPH_HIDDEN];
#define EXPECT_LOCK_FREE() do { EXPECT(fk.locks_held == 0); EXPECT(fk.double_locks == 0); EXPECT(fk.bad_unlocks == 0); } while (0)

static int install(void) {
    memset(&fk, 0, sizeof fk); omega_gpu_handles_init(&fk.t);
    for (unsigned i = 0; i < OMEGA_GPU_DGRAPH_HIDDEN; i++) g_x[i] = (float)(i % 7) - 3.0f;
    return omega_gpu_session_set_platform(&g_plat, &g_caps);
}

/* One token of `launches` SWIGLU glue launches (each submitted on its own); returns end_token's code, fills *refused
 * with the first record that did not return OK (launches when every record did). */
static int token_of(World *w, uint32_t launches, uint32_t want, uint32_t *refused, uint32_t *tok) {
    OmegaGpuDecodeGraphVas v; omega_gpu_decode_graph_vas(w->g, &v);
    fk.dev_token_va = v.token; fk.dev_token = want;
    *refused = launches;
    if (omega_gpu_decode_graph_begin_token(w->g, g_x) != OMEGA_GPU_DGRAPH_OK) return -1000;
    for (uint32_t i = 0; i < launches; i++)
        if (omega_gpu_decode_graph_record_swiglu(w->g, v.gu, v.act, 1.0f) != OMEGA_GPU_DGRAPH_OK && *refused == launches) *refused = i;
    const int rc = omega_gpu_decode_graph_end_token(w->g, tok, NULL);
    fk.dev_token_va = 0;
    return rc;
}

static uint64_t g_st[OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 2];
static uint8_t g_kd[OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 2];

/* MAX_LAUNCHES stamped launches: the timeline is exactly full and reads back whole. */
static void c_full(void) {
    World w;
    if (install() != OMEGA_GPU_PLAT_OK) { EXPECT(!"fake platform installs"); return; } /* never fall back to the real driver */
    if (world_build(&w) != 0) { EXPECT(!"world builds on the fake"); return; }
    uint32_t refused = 0, tok = 0;
    EXPECT(token_of(&w, OMEGA_GPU_CMDLIST_MAX_LAUNCHES, 7, &refused, &tok) == OMEGA_GPU_DGRAPH_OK);
    EXPECT(refused == OMEGA_GPU_CMDLIST_MAX_LAUNCHES && tok == 7);
    EXPECT(fk.submits == OMEGA_GPU_CMDLIST_MAX_LAUNCHES);
    EXPECT_LOCK_FREE();
    EXPECT(omega_gpu_decode_graph_stamps(w.g, g_st, g_kd, OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 2) == (int)OMEGA_GPU_CMDLIST_MAX_LAUNCHES);
    for (uint32_t i = 0; i < OMEGA_GPU_CMDLIST_MAX_LAUNCHES; i++) if (!(g_st[i + 1] > g_st[i])) { EXPECT(!"stamps increase"); break; }
}

/* One launch more than the timeline holds: that record fails the token, nothing is written past stamp[], the lock is
 * released at end_token, no timeline is offered, and the next (small) token runs and reads back normally. */
static void c_overflow(void) {
    World w;
    if (install() != OMEGA_GPU_PLAT_OK) { EXPECT(!"fake platform installs"); return; }
    if (world_build(&w) != 0) { EXPECT(!"world builds on the fake"); return; }
    uint32_t refused = 0, tok = 0xdeadu;
    EXPECT(token_of(&w, OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1u, 7, &refused, &tok) == OMEGA_GPU_DGRAPH_CHIP_FAIL);
    EXPECT(refused == OMEGA_GPU_CMDLIST_MAX_LAUNCHES); /* the 4097th launch, the first that has no stamp slot */
    EXPECT(tok == 0xdeadu);
    EXPECT(strstr(omega_gpu_session_last_error(), "timeline full") != NULL);
    EXPECT_LOCK_FREE();
    EXPECT(omega_gpu_decode_graph_stamps(w.g, g_st, g_kd, OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 2) == OMEGA_GPU_DGRAPH_BAD_ARGS);
    EXPECT(token_of(&w, 3, 9, &refused, &tok) == OMEGA_GPU_DGRAPH_OK);
    EXPECT(refused == 3 && tok == 9);
    EXPECT_LOCK_FREE();
    EXPECT(omega_gpu_decode_graph_stamps(w.g, g_st, g_kd, OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 2) == 3);
}

static const struct { const char *name; void (*fn)(void); } CASES[] = {
    { "timeline exactly full: 4096 stamped launches read back", c_full },
    { "one launch past the timeline: token fails, no write past stamp[]", c_overflow },
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
    printf("test_omega_gpu_decode_graph_timeline: %u of %u cases passed: %s\n", n - failed, n, failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
