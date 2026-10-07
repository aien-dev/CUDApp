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
#include "omega_gpu_serving.h"
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

static const OmegaGpuServingBounds QWEN3_4B_BOUNDS = {
    .max_context = 4096, .max_seqs = 1, .num_q_heads = QH, .num_kv_heads = KVH, .head_dim = HD, .kv_block_size = 16,
    .max_rows = CHUNK, .max_k = INTER, .max_n = INTER, .max_n_one_row = VOCAB, .kernel_slots = 32,
};

/* Bounds mode: a tiny reservation, then calls inside it (no driver traffic) and past it (refused loudly). */
static int paged_call(uint32_t bs, uint32_t ctx);
static int attn_ran(int rc);

static int bounds_mode(void) {
    OmegaGpuServingBounds b = QWEN3_4B_BOUNDS;
    b.max_context = 256; b.max_rows = 32; b.max_k = HID; b.max_n = QDIM; b.max_n_one_row = 0;
    Phase ph; begin(&ph, "BOUNDS"); /* sets P for mm()/attn() */
    W_Q = upload(HID, QDIM);
    attn(1); mm(1, W_Q, g_y);                                   /* warm kernels at small size */
    CHECK(omega_gpu_reserve_serving(&b) == 0, "reserve failed: %s", omega_gpu_matmul_last_error());
    mm(1, W_Q, g_y); mm(17, W_Q, g_y); mm(32, W_Q, g_y); attn(256); /* warm each shape once (reserving a deeper kernel cache restarts it empty) */
    const int a0 = fk.alloc_calls, f0 = fk.free_calls;
    mm(1, W_Q, g_y); mm(17, W_Q, g_y); mm(32, W_Q, g_y); attn(1); attn(100); attn(256);
    CHECK(fk.alloc_calls == a0 && fk.free_calls == f0, "calls inside the bounds touched the driver (%d allocs, %d frees)", fk.alloc_calls - a0, fk.free_calls - f0);
    OmegaGpuMatmulInfo info;
    int rc = omega_gpu_matmul_resident_f32(33, g_x, W_Q, g_y, &info);          /* 33 rows -> 48 padded: past 32 */
    CHECK(rc == OMEGA_GPU_MATMUL_TOO_LARGE, "33 rows past a 32-row reservation: rc=%d (%s)", rc, omega_gpu_matmul_rc_name(rc));
    CHECK(strstr(omega_gpu_matmul_last_error(), "serving reservation exceeded") != NULL, "matmul error text: \"%s\"", omega_gpu_matmul_last_error());
    rc = omega_gpu_gqa_attention_f32(g_q, g_k, g_v, 257, QH, KVH, HD, g_attn, NULL);
    CHECK(rc == OMEGA_GPU_ATTN_TOO_LARGE, "context 257 past a 256 reservation: rc=%d (%s)", rc, omega_gpu_attention_rc_name(rc));
    CHECK(strstr(omega_gpu_attention_last_error(), "serving reservation exceeded") != NULL, "attention error text: \"%s\"", omega_gpu_attention_last_error());
    CHECK(fk.alloc_calls == a0 && fk.free_calls == f0, "a refused call touched the driver (%d allocs, %d frees)", fk.alloc_calls - a0, fk.free_calls - f0);
    CHECK(!omega_gpu_matmul_is_blocked() && !omega_gpu_session_is_blocked(), "a refusal latched the session");
    mm(32, W_Q, g_y); attn(256);                                /* still serves after a refusal */
    CHECK(fk.alloc_calls == a0, "serving after a refusal allocated");
    /* paged bf16 attention (the sovereign-core path): 512 tokens = 32 blocks of 16 = exactly the reserved bytes */
    (void)paged_call(16, 16);                                   /* build the paged kernel once */
    const int p0 = fk.alloc_calls, pf0 = fk.free_calls;
    rc = paged_call(16, 16);  CHECK(attn_ran(rc), "paged 16 tokens in bounds: rc=%d (%s)", rc, omega_gpu_attention_last_error());
    rc = paged_call(16, 512); CHECK(attn_ran(rc), "paged 512 tokens (exactly the bound) refused: rc=%d (%s)", rc, omega_gpu_attention_last_error());
    CHECK(fk.alloc_calls == p0 && fk.free_calls == pf0, "in-bounds paged calls touched the driver (%d allocs, %d frees)", fk.alloc_calls - p0, fk.free_calls - pf0);
    rc = paged_call(16, 513); CHECK(rc == OMEGA_GPU_ATTN_TOO_LARGE, "paged 513 tokens (bound+1): rc=%d", rc);
    CHECK(strstr(omega_gpu_attention_last_error(), "serving reservation exceeded") != NULL, "paged error text: \"%s\"", omega_gpu_attention_last_error());
    CHECK(fk.alloc_calls == p0 && fk.free_calls == pf0, "a refused paged call touched the driver");
    omega_gpu_serving_release();                                /* lifted: growth is allowed again */
    rc = omega_gpu_gqa_attention_f32(g_q, g_k, g_v, 257, QH, KVH, HD, g_attn, NULL);
    CHECK(rc == OMEGA_GPU_ATTN_OK || rc == OMEGA_GPU_ATTN_UNWRITTEN, "after release, context 257 rc=%d", rc);
    CHECK(fk.alloc_calls > a0, "after release, growth did not allocate");
    printf("MEASURE bounds: refused calls made 0 driver allocations; 33 rows -> TOO_LARGE, ctx 257 -> TOO_LARGE; error \"%s\"\n", "serving reservation exceeded: ...");
    omega_gpu_tensor_free(W_Q);
    omega_gpu_device_close();
    CHECK(fk.live_at_close == 0, "leak: %d", fk.live_at_close);
    printf("gpu_serving_alloc_count_test bounds: %s (%d failed checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

/* Paged bf16 attention call over `ctx` tokens of a one-layer pool of `bs`-token blocks (the path sovereign-core uses). */
static int paged_call(uint32_t bs, uint32_t ctx) {
    const uint32_t nb = (ctx + bs - 1) / bs;
    OmegaGpuKvLayout ly;
    ly.head_stride_bytes = (uint64_t)HD * 2; ly.token_stride_bytes = ly.head_stride_bytes * KVH;
    ly.kv_plane_stride_bytes = ly.token_stride_bytes * bs; ly.layer_stride_bytes = 2 * ly.kv_plane_stride_bytes;
    ly.block_stride_bytes = ly.layer_stride_bytes; ly.pool_bytes = ly.block_stride_bytes * nb;
    ly.num_blocks = nb; ly.num_layers = 1; ly.block_size = bs;
    uint8_t *pool = calloc(1, (size_t)ly.pool_bytes);
    uint32_t *ids = calloc(nb, sizeof *ids);
    CHECK(pool && ids, "paged host buffers");
    int rc = OMEGA_GPU_ATTN_BAD_ARGS;
    if (pool && ids) {
        for (uint32_t i = 0; i < nb; i++) ids[i] = i;
        rc = omega_gpu_paged_attention_bf16(g_q, pool, &ly, ids, nb, ctx, 0, QH, KVH, HD, g_attn, NULL);
    }
    free(pool); free(ids);
    return rc;
}
static int attn_ran(int rc) { return rc == OMEGA_GPU_ATTN_OK || rc == OMEGA_GPU_ATTN_UNWRITTEN; }

/* Under half a block of context: the paged path stages a whole block, more than the f32 formula (review item 1). */
static int bounds_block_mode(void) {
    OmegaGpuServingBounds b = QWEN3_4B_BOUNDS;
    b.max_context = 16; b.kv_block_size = 64; b.max_rows = 16; b.max_k = HID; b.max_n = QDIM; b.max_n_one_row = 0;
    Phase ph; begin(&ph, "BOUNDS-BLOCK");
    W_Q = upload(HID, QDIM);
    CHECK(omega_gpu_reserve_serving(&b) == 0, "reserve failed: %s", omega_gpu_matmul_last_error());
    int rc = paged_call(64, 16);                               /* first paged call builds the kernel (one code buffer); no pre-reserve call, it would raise the pool high-water mark */
    CHECK(attn_ran(rc), "paged call of 16 tokens in one 64-token block refused: rc=%d (%s)", rc, omega_gpu_attention_last_error());
    const int a0 = fk.alloc_calls, f0 = fk.free_calls;
    rc = paged_call(64, 64);
    CHECK(attn_ran(rc), "paged call of exactly one block (the byte bound) refused: rc=%d (%s)", rc, omega_gpu_attention_last_error());
    CHECK(fk.alloc_calls == a0 && fk.free_calls == f0, "in-bounds paged calls touched the driver");
    rc = paged_call(64, 65);
    CHECK(rc == OMEGA_GPU_ATTN_TOO_LARGE, "two blocks past a one-block reservation: rc=%d", rc);
    CHECK(fk.alloc_calls == a0 && fk.free_calls == f0, "refused paged call touched the driver");
    printf("MEASURE bounds-block: 16 and 64 tokens (one 64-token block) served with 0 driver traffic; 65 tokens -> TOO_LARGE\n");
    omega_gpu_tensor_free(W_Q);
    omega_gpu_device_close();
    CHECK(fk.live_at_close == 0, "leak: %d", fk.live_at_close);
    printf("gpu_serving_alloc_count_test bounds-block: %s (%d failed checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

/* A reserve that fails part way must leave nothing flagged (review item 2). Sweep the failing driver request over every
 * allocation the reserve makes (pool, q, out, table, activation, result), on a freshly opened device each time; the
 * control (no injected failure) must leave the reservation active. */
static int rollback_mode(void) {
    OmegaGpuServingBounds b = QWEN3_4B_BOUNDS;
    b.max_context = 256; b.kv_block_size = 16; b.max_rows = 32; b.max_k = HID; b.max_n = QDIM; b.max_n_one_row = 0; b.kernel_slots = 0;
    Phase ph; begin(&ph, "ROLLBACK");
    int failed_reserves = 0;
    /* n = 100 first: the un-injected control, before the probes of the failing runs raise the buffers' high-water marks */
    const int sweep[] = { 100, 1, 2, 3, 4, 5, 6 };
    for (unsigned si = 0; si < sizeof sweep / sizeof sweep[0]; si++) {
        const int n = sweep[si];
        W_Q = upload(HID, QDIM);                                /* opens the session (its own allocations come first) */
        fk.alloc_mode = FK_ALLOC_FAIL;
        fk.alloc_fail_nth = fk.alloc_calls + n;
        int rc = omega_gpu_reserve_serving(&b);
        fk.alloc_fail_nth = 0;
        /* probes that need more than the reservation: refused only if something is still flagged */
        /* each probe is larger than every earlier one, so a buffer that is still flagged is refused even though the high-water mark carries over */
        int ra = omega_gpu_gqa_attention_f32(g_q, g_k, g_v, si ? 300 + 500 * si : 257, QH, KVH, HD, g_attn, NULL);
        int refused_attn = ra == OMEGA_GPU_ATTN_TOO_LARGE && strstr(omega_gpu_attention_last_error(), "serving reservation exceeded") != NULL;
        OmegaGpuMatmulInfo info;
        int rm = omega_gpu_matmul_resident_f32(si ? 32 + 16 * si : 48, g_x, W_Q, g_y, &info);
        int refused_mm = rm == OMEGA_GPU_MATMUL_TOO_LARGE && strstr(omega_gpu_matmul_last_error(), "serving reservation exceeded") != NULL;
        if (rc != 0) {
            failed_reserves++;
            CHECK(!refused_attn && !refused_mm, "failed reserve (injected fail on its request %d, rc=%d) left a reservation flagged: attention %d matmul %d", n, rc, refused_attn, refused_mm);
        } else {
            CHECK(refused_attn && refused_mm, "control (request %d never failed, reserve ok): the reservation should be active, attention %d matmul %d", n, refused_attn, refused_mm);
        }
        omega_gpu_tensor_free(W_Q);
        omega_gpu_device_close();
    }
    CHECK(failed_reserves >= 6, "only %d of the sweep's reserves failed (expected the 6 buffer requests)", failed_reserves);
    CHECK(fk.live_at_close == 0, "leak: %d", fk.live_at_close);
    printf("MEASURE rollback: %d injected failures during reserve, none left anything flagged; the un-injected control stayed reserved\n", failed_reserves);
    printf("gpu_serving_alloc_count_test rollback: %s (%d failed checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

/* mode: "plain" (default) = today's measurement, no reservation; "reserve" = reserve after warm-up and
 * ASSERT zero driver traffic while serving; "reserve-skip" = the same asserts without the call (must FAIL: the red run);
 * "bounds" = refusal behaviour. */
int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "plain";
    const int do_reserve = strcmp(mode, "reserve") == 0, assert_zero = do_reserve || strcmp(mode, "reserve-skip") == 0;
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

    Phase ph[9];
    int np = 0;
    if (strcmp(mode, "bounds") == 0) return bounds_mode();
    if (strcmp(mode, "bounds-block") == 0) return bounds_block_mode();
    if (strcmp(mode, "rollback") == 0) return rollback_mode();

    begin(&ph[np], "LOAD (session+6 weights)");
    W_Q = upload(HID, QDIM); W_K = upload(HID, KVDIM); W_O = upload(QDIM, HID);
    W_G = upload(HID, INTER); W_D = upload(INTER, HID); W_LM = upload(HID, VOCAB);
    end(&ph[np], 1); np++;
    CHECK(omega_gpu_session_open_count() == 1, "session opened %u times", omega_gpu_session_open_count());

    /* WARM-UP: one decode step at context 1. Everything after this phase is "serving". */
    begin(&ph[np], "WARMUP decode ctx=1");
    step(1, 0);
    end(&ph[np], 1); np++;
    if (assert_zero) {
        /* a daemon's startup: reserve once after loading and a first request, then cover each row-count class
         * (1 decode row, a full chunk, the 100 and 72 row remainders) so every matmul shape has been built once */
        if (do_reserve) CHECK(omega_gpu_reserve_serving(&QWEN3_4B_BOUNDS) == 0, "reserve failed: %s", omega_gpu_matmul_last_error());
        begin(&ph[np], "WARMUP row classes");
        step(128, 0); step(100, 0); step(72, 0); step(1, 0);
        end(&ph[np], 0); np++;
    }
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
    printf("MEASURE distinct matmul kernel shapes seen (kp,np,grid_x): %d (cache slots: %d)\n", g_nseen, do_reserve ? 32 : 8);
    printf("MEASURE live driver buffers at end: %d ; session opens: %u ; launches reaching the fake: %d\n",
           fk.live, omega_gpu_session_open_count(), fk.kernel_ran);
    printf("MEASURE return codes: matmul ok %d, matmul poison-check CHIP_FAIL %d, attention ok %d, attention UNWRITTEN %d, other %d\n",
           g_rc_mm_ok, g_rc_mm_chipfail, g_rc_at_ok, g_rc_at_unwritten, g_rc_bad);

    /* ---- what is true today and stays true ---- */
    CHECK(g_rc_bad == 0, "%d calls returned a code other than the expected poison-check ones", g_rc_bad);
    CHECK(!omega_gpu_matmul_is_blocked() && !omega_gpu_session_is_blocked(), "session latched");
    CHECK(fk.double_free == 0, "double free %d", fk.double_free);
    CHECK(fk.pb_parse_error == 0, "pushbuffer parse error");
    int calls_made = 0;
    for (int i = 1; i < np; i++) calls_made += ph[i].mm_calls + ph[i].at_calls;
    CHECK(fk.kernel_ran == calls_made, "launches %d != calls made %d", fk.kernel_ran, calls_made);
    if (assert_zero) {
        /* the point of the reservation: nothing reached the driver while serving inside the bounds */
        CHECK(serve_allocs == 0 && serve_frees == 0, "serving made %d driver allocations and %d frees after warm-up; the reservation promises none", serve_allocs, serve_frees);
        for (int i = np - 4; i < np; i++) CHECK(ph[i].allocs == 0 && ph[i].frees == 0, "phase %s: %d allocs, %d frees", ph[i].name, ph[i].allocs, ph[i].frees);
    }
    CHECK(fk.live < FK_LIVE_MAX, "fake live table full (%d)", fk.live);
    /* every real allocation is either live or was freed */
    CHECK(fk.alloc_calls - fk.free_calls >= fk.live, "alloc/free/live inconsistent: %d - %d < %d", fk.alloc_calls, fk.free_calls, fk.live);

    /* teardown: tensors first, then the device (the on-close hooks free the API scratch) */
    omega_gpu_tensor_free(W_Q); omega_gpu_tensor_free(W_K); omega_gpu_tensor_free(W_O);
    omega_gpu_tensor_free(W_G); omega_gpu_tensor_free(W_D); omega_gpu_tensor_free(W_LM);
    omega_gpu_device_close();
    CHECK(fk.live_at_close == 0, "leak: %d driver buffers live at close", fk.live_at_close);
    printf("MEASURE leaked at close: %d\n", fk.live_at_close);

    printf("gpu_serving_alloc_count_test %s: %s (%d failed checks)\n", mode, g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
