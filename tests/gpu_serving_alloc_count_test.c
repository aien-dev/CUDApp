/*
 * gpu_serving_alloc_count_test: MEASUREMENT of driver (RM) allocations and frees made by the real
 * native GPU stack (session + matmul API + attention API) while it "serves" a Qwen3-4B-shaped
 * sequence, on the simulated driver of tests/fake_m16_native.c. Host only, opens no device.
 *
 * Why: sovereign-core design note docs/design/gb10-weight-ownership.md (sc#277, sc PR #292) argues
 * that omega still asks the driver for memory while serving, and that one refused request faults
 * the GPU session for the rest of the process. This test counts those requests.
 *
 * What it asserts: only bookkeeping that is true today (the session opens once, the fake saw every
 * request, no double free, nothing latched, the call mix is what the report says it is). It does
 * NOT assert the allocation counts: they are printed as MEASURE lines. A later change that reserves
 * serving memory up front would lower them and this test would still pass; the numbers in the PR
 * that made it are the record of today's behaviour.
 *
 * What it does not model: the kernels. The fake runs in FK_KERNEL_SKIP mode, so launches are
 * counted and complete, but no matmul or attention result is computed. The matmul API therefore
 * reports its own poison check (CHIP_FAIL) and the attention API reports UNWRITTEN; those return
 * codes are expected here and are not errors of the allocation path. Elementwise kernels (rmsnorm,
 * rope, swiglu) are not exercised; the design note's audit says they stop allocating after the
 * first full-size call.
 *
 * Shapes (Qwen3-4B): hidden 2560, 32 q heads, 8 kv heads, head_dim 128, intermediate 9728,
 * vocab 151936, 36 layers. One resident tensor per distinct weight shape stands in for the 36
 * layers (resident-weight allocations are a load-time cost, counted in phase LOAD only).
 * Call pattern per layer follows crates/aien-inference-abi/src/transformer_backend.rs:
 *   decode (m = 1):  q, k, v, attention(seq), o, gate, up, down ; after the layers lm_head (m = 1)
 *   prefill chunk:   the same with m = chunk rows (<= 128), attention once per token at seq =
 *                    offset + t + 1, lm_head with m = 1 after the chunk
 * CTA budget 256 (the daemon value in the audit).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fake_m16_native.h"
#include "omega_gpu_attention_api.h"
#include "omega_gpu_matmul_api.h"
#include "omega_gpu_session.h"

/* The two driver symbols the session needs that the fake (written for the vector engine) lacks. */
int nvrm_open(Nvrm *rm) { (void)rm; fk.device_open = 1; return 0; }
int nvrm_close(Nvrm *rm) { (void)rm; return 0; }
void nvrm_retire(Nvrm *rm, uint32_t upto) { (void)rm; (void)upto; }

#define HID 2560u
#define QDIM 4096u
#define KVDIM 1024u
#define INTER 9728u
#define VOCAB 151936u
#define LAYERS 36u
#define QH 32u
#define KVH 8u
#define HD 128u
#define CHUNK 128u

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* ---- the weights: one tensor per distinct shape ---- */
static OmegaGpuTensor *W_Q, *W_K, *W_O, *W_G, *W_D, *W_LM;

static OmegaGpuTensor *upload(uint32_t k, uint32_t n) {
    uint16_t *z = calloc((size_t)k * n, sizeof *z);
    OmegaGpuTensor *t = NULL;
    CHECK(z != NULL, "calloc %ux%u", k, n);
    int rc = z ? omega_gpu_tensor_upload_bf16(k, n, z, &t) : -1;
    CHECK(rc == OMEGA_GPU_MATMUL_OK, "upload %ux%u rc=%d %s", k, n, rc, omega_gpu_matmul_last_error());
    free(z);
    return t;
}

/* ---- per-phase counters ---- */
typedef struct {
    const char *name;
    int allocs, frees;
    uint64_t alloc_bytes, free_bytes;
    int mm_calls, mm_miss, mm_remiss, mm_other_allocs; /* code-cache misses; misses on an already-seen shape; scratch growth */
    int at_calls, at_allocs;
    int a0, f0;                                        /* log positions at phase start */
} Phase;

typedef struct { uint32_t kp, np, gx; } Key;
static Key g_seen[64];
static int g_nseen;
static int seen(uint32_t kp, uint32_t np, uint32_t gx) {
    for (int i = 0; i < g_nseen; i++) if (g_seen[i].kp == kp && g_seen[i].np == np && g_seen[i].gx == gx) return 1;
    if (g_nseen < 64) { g_seen[g_nseen].kp = kp; g_seen[g_nseen].np = np; g_seen[g_nseen].gx = gx; g_nseen++; }
    return 0;
}

static Phase *P; /* current phase */
static int g_rc_mm_ok, g_rc_mm_chipfail, g_rc_at_ok, g_rc_at_unwritten, g_rc_bad;

static float *g_x, *g_y;      /* activation in / result out, sized for CHUNK x INTER and CHUNK x VOCAB-free */
static float *g_logits;       /* 1 x VOCAB */
static float *g_k, *g_v;      /* host KV for the attention call, 4096 x 1024 f32 each */
static float g_q[QDIM], g_attn[QDIM];

static void mm(uint32_t m, const OmegaGpuTensor *w, float *out) {
    uint32_t k, n;
    omega_gpu_tensor_shape(w, &k, &n);
    int a_before = fk.alloc_calls;
    OmegaGpuMatmulInfo info;
    int rc = omega_gpu_matmul_resident_f32(m, g_x, w, out, &info);
    int made = fk.alloc_calls - a_before;
    P->mm_calls++;
    if (rc == OMEGA_GPU_MATMUL_OK) g_rc_mm_ok++;
    else if (rc == OMEGA_GPU_MATMUL_CHIP_FAIL) g_rc_mm_chipfail++;
    else { g_rc_bad++; printf("unexpected matmul rc=%d (%s)\n", rc, omega_gpu_matmul_rc_name(rc)); }
    int miss = info.kernel_cache_hit ? 0 : 1; /* at most one miss per launch; m <= 128 rows is one launch here */
    if (miss) {
        P->mm_miss++;
        if (seen(info.padded_k, info.padded_n, info.grid_x)) P->mm_remiss++;
    } else {
        (void)seen(info.padded_k, info.padded_n, info.grid_x);
    }
    P->mm_other_allocs += made - miss;
}

static void attn(uint32_t seq) {
    int a_before = fk.alloc_calls;
    OmegaGpuAttnInfo info;
    int rc = omega_gpu_gqa_attention_f32(g_q, g_k, g_v, seq, QH, KVH, HD, g_attn, &info);
    P->at_calls++;
    P->at_allocs += fk.alloc_calls - a_before;
    if (rc == OMEGA_GPU_ATTN_OK) g_rc_at_ok++;
    else if (rc == OMEGA_GPU_ATTN_UNWRITTEN) g_rc_at_unwritten++;
    else { g_rc_bad++; printf("unexpected attention rc=%d (%s)\n", rc, omega_gpu_attention_rc_name(rc)); }
}

/* One transformer step over `rows` rows of a sequence that already holds `offset` tokens. */
static void step(uint32_t rows, uint32_t offset) {
    for (uint32_t l = 0; l < LAYERS; l++) {
        mm(rows, W_Q, g_y); mm(rows, W_K, g_y); mm(rows, W_K, g_y);
        for (uint32_t t = 0; t < rows; t++) attn(offset + t + 1);
        mm(rows, W_O, g_y); mm(rows, W_G, g_y); mm(rows, W_G, g_y); mm(rows, W_D, g_y);
    }
    mm(1, W_LM, g_logits);
}

static void begin(Phase *p, const char *name) {
    memset(p, 0, sizeof *p);
    p->name = name;
    p->a0 = fk.alloc_size_n;
    p->f0 = fk.free_size_n;
    p->allocs = fk.alloc_calls;   /* start values, turned into deltas by end() */
    p->frees = fk.free_calls;
    P = p;
}

static void sizes(const char *kind, const uint64_t *log, int from, int to) {
    /* distinct sizes with counts, in order of first appearance, at most 6 shown */
    uint64_t seen_sz[256]; int cnt[256]; int ns = 0;
    for (int i = from; i < to && i < FK_SIZE_LOG; i++) {
        int j;
        for (j = 0; j < ns; j++) if (seen_sz[j] == log[i]) break;
        if (j == ns && ns < 256) { seen_sz[ns] = log[i]; cnt[ns] = 0; ns++; }
        if (j < 256) cnt[j]++;
    }
    printf("MEASURE   %s sizes (%d distinct):", kind, ns);
    for (int j = 0; j < ns && j < 6; j++) printf(" %llux%d", (unsigned long long)seen_sz[j], cnt[j]);
    if (ns > 6) printf(" ... last %llux%d", (unsigned long long)seen_sz[ns - 1], cnt[ns - 1]);
    printf("\n");
}

static void end(Phase *p, int sized) {
    int a1 = fk.alloc_size_n, f1 = fk.free_size_n;
    p->allocs = fk.alloc_calls - p->allocs;
    p->frees = fk.free_calls - p->frees;
    for (int i = p->a0; i < a1 && i < FK_SIZE_LOG; i++) p->alloc_bytes += fk.alloc_size_log[i];
    for (int i = p->f0; i < f1 && i < FK_SIZE_LOG; i++) p->free_bytes += fk.free_size_log[i];
    printf("MEASURE phase %-22s allocs %4d  frees %4d  alloc_bytes %12llu  matmul_calls %5d (code-cache misses %d, of which on a shape seen before %d, other scratch growth %d)  attention_calls %5d (allocs %d)\n",
           p->name, p->allocs, p->frees, (unsigned long long)p->alloc_bytes, p->mm_calls, p->mm_miss, p->mm_remiss, p->mm_other_allocs,
           p->at_calls, p->at_allocs);
    if (sized && p->allocs) { sizes("alloc", fk.alloc_size_log, p->a0, a1); sizes("free ", fk.free_size_log, p->f0, f1); }
    CHECK(a1 <= FK_SIZE_LOG && f1 <= FK_SIZE_LOG, "size log overflow (%d allocs, %d frees)", a1, f1);
}

int main(void) {
    fake_reset();
    fk.kernel_mode = FK_KERNEL_SKIP;
    fk.alloc_max = 1ull << 31;
    omega_gpu_matmul_set_cta_budget(256);

    g_x = calloc((size_t)CHUNK * INTER, sizeof *g_x);
    g_y = calloc((size_t)CHUNK * INTER, sizeof *g_y);
    g_logits = calloc(VOCAB, sizeof *g_logits);
    g_k = calloc((size_t)4096 * KVDIM, sizeof *g_k);
    g_v = calloc((size_t)4096 * KVDIM, sizeof *g_v);
    CHECK(g_x && g_y && g_logits && g_k && g_v, "host buffers");
    if (g_fail) return 1;

    Phase ph[8];
    int np = 0;

    begin(&ph[np], "LOAD (session+6 weights)");
    W_Q = upload(HID, QDIM); W_K = upload(HID, KVDIM); W_O = upload(QDIM, HID);
    W_G = upload(HID, INTER); W_D = upload(INTER, HID); W_LM = upload(HID, VOCAB);
    end(&ph[np], 1); np++;
    CHECK(omega_gpu_session_open_count() == 1, "session opened %u times", omega_gpu_session_open_count());

    /* WARM-UP: one decode step at context 1. Everything after this phase is "serving". */
    begin(&ph[np], "WARMUP decode ctx=1");
    step(1, 0);
    end(&ph[np], 1); np++;
    const int after_warm_allocs = fk.alloc_calls, after_warm_frees = fk.free_calls;

    uint32_t ctx = 0;
    const uint32_t N1 = 100, D1 = 64, N2 = 200, D2 = 64;

    begin(&ph[np], "PREFILL-A 100 tokens");
    for (uint32_t off = 0; off < N1; off += CHUNK) { uint32_t r = N1 - off < CHUNK ? N1 - off : CHUNK; step(r, ctx + off); }
    ctx += N1;
    end(&ph[np], 1); np++;

    begin(&ph[np], "DECODE-A 64 tokens");
    for (uint32_t i = 0; i < D1; i++) { step(1, ctx); ctx++; }
    end(&ph[np], 1); np++;

    begin(&ph[np], "PREFILL-B 200 tokens");
    for (uint32_t off = 0; off < N2; off += CHUNK) { uint32_t r = N2 - off < CHUNK ? N2 - off : CHUNK; step(r, ctx + off); }
    ctx += N2;
    end(&ph[np], 1); np++;

    begin(&ph[np], "DECODE-B 64 tokens");
    for (uint32_t i = 0; i < D2; i++) { step(1, ctx); ctx++; }
    end(&ph[np], 1); np++;

    const int serve_allocs = fk.alloc_calls - after_warm_allocs, serve_frees = fk.free_calls - after_warm_frees;
    printf("MEASURE AFTER WARM-UP: %d driver allocations, %d driver frees over %u tokens of context (final ctx %u)\n",
           serve_allocs, serve_frees, ctx, ctx);
    printf("MEASURE distinct matmul kernel shapes seen (kp,np,grid_x): %d (cache slots: 8)\n", g_nseen);
    printf("MEASURE live driver buffers at end: %d ; session opens: %u ; launches reaching the fake: %d\n",
           fk.live, omega_gpu_session_open_count(), fk.kernel_ran);
    printf("MEASURE return codes: matmul ok %d, matmul poison-check CHIP_FAIL %d, attention ok %d, attention UNWRITTEN %d, other %d\n",
           g_rc_mm_ok, g_rc_mm_chipfail, g_rc_at_ok, g_rc_at_unwritten, g_rc_bad);

    /* ---- what is true today and stays true ---- */
    CHECK(g_rc_bad == 0, "%d calls returned a code other than the expected poison-check ones", g_rc_bad);
    CHECK(!omega_gpu_matmul_is_blocked() && !omega_gpu_session_is_blocked(), "session latched");
    CHECK(fk.double_free == 0, "double free %d", fk.double_free);
    CHECK(fk.pb_parse_error == 0, "pushbuffer parse error");
    CHECK(fk.kernel_ran == ph[1].mm_calls + ph[1].at_calls + ph[2].mm_calls + ph[2].at_calls + ph[3].mm_calls + ph[3].at_calls +
                           ph[4].mm_calls + ph[4].at_calls + ph[5].mm_calls + ph[5].at_calls,
          "launches %d != calls made", fk.kernel_ran);
    CHECK(fk.live < FK_LIVE_MAX, "fake live table full (%d)", fk.live);
    /* every real allocation is either live or was freed */
    CHECK(fk.alloc_calls - fk.free_calls >= fk.live, "alloc/free/live inconsistent: %d - %d < %d", fk.alloc_calls, fk.free_calls, fk.live);

    /* teardown: tensors first, then the device (the on-close hooks free the API scratch) */
    omega_gpu_tensor_free(W_Q); omega_gpu_tensor_free(W_K); omega_gpu_tensor_free(W_O);
    omega_gpu_tensor_free(W_G); omega_gpu_tensor_free(W_D); omega_gpu_tensor_free(W_LM);
    omega_gpu_device_close();
    CHECK(fk.live_at_close == 0, "leak: %d driver buffers live at close", fk.live_at_close);
    printf("MEASURE leaked at close: %d\n", fk.live_at_close);

    printf("gpu_serving_alloc_count_test: %s (%d failed checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
