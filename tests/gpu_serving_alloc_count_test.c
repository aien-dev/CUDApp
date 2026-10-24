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
#include "nvrm.h" /* the session reaches the fake driver through the Linux reference platform (P07 cut C) */
#include "omega_blackwell_codegen.h" /* omega_matmul_frag_block_shape: the kernel cache key */
#include "omega_blackwell_matmul.h" /* omega_fp32_to_bf16 (update mode) */
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

/* The matmul API's kernel cache key: (kp, np, grid_x, rt, ct). rt and ct follow from the call's padded shape
 * (omega_matmul_frag_block_shape, the rule the API's launch_shape uses); since the blocked prefill kernel (#407)
 * two calls with the same grid_x can need different kernels. */
typedef struct { uint32_t kp, np, gx, rt, ct; } Key;
static Key g_seen[128];
static int g_nseen;
static int seen(const OmegaGpuMatmulInfo *in) {
    Key k = { in->padded_k, in->padded_n, in->grid_x, 0, 0 };
    omega_matmul_frag_block_shape(in->padded_m / 16u, in->padded_n / 8u, &k.rt, &k.ct);
    for (int i = 0; i < g_nseen; i++)
        if (g_seen[i].kp == k.kp && g_seen[i].np == k.np && g_seen[i].gx == k.gx && g_seen[i].rt == k.rt && g_seen[i].ct == k.ct) return 1;
    if (g_nseen < 128) g_seen[g_nseen++] = k;
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
        if (seen(&info)) P->mm_remiss++;
    } else {
        (void)seen(&info);
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
    /* build each paged kernel the checked calls use: the split count is part of the kernel (cut F/J, own_log2split),
     * 16 tokens run split 0 and 512 tokens split 2, and the attention kernel cache is outside the reservation
     * (omega_gpu_serving.h, "Not covered") */
    (void)paged_call(16, 16); (void)paged_call(16, 512);
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
/* ---- sweep: the sovereign-core daemon's bounds (sc#311, sc#277). Every row count 1..256 over every weight
 * shape is a distinct (kp, np, grid_x) for some rows, because grid_x is baked into the kernel and depends on
 * the row tiles. prepare = 1: prepare each (rows, shape) and seal, then serving must make no driver call and an
 * unprepared shape is refused. prepare = 0 (control, kernel_slots 32 as before): the cache wraps and serving
 * allocates, which this mode must report as a failure. */
#define SWEEP_ROWS 256u
static int sweep_mode(int prepare) {
    W_Q = upload(HID, QDIM); W_K = upload(HID, KVDIM); W_O = upload(QDIM, HID);
    W_G = upload(HID, INTER); W_D = upload(INTER, HID); W_LM = upload(HID, VOCAB);
    OmegaGpuTensor *odd = upload(HID, 512); /* a shape the server never prepares */
    OmegaGpuTensor *w[6] = {W_Q, W_K, W_O, W_G, W_D, W_LM};
    float *x = calloc((size_t)SWEEP_ROWS * INTER, sizeof *x), *y = calloc((size_t)SWEEP_ROWS * VOCAB, sizeof *y);
    CHECK(x && y, "sweep buffers");
    if (g_fail) return 1;
    OmegaGpuServingBounds b = QWEN3_4B_BOUNDS;
    b.max_rows = SWEEP_ROWS; b.max_n = VOCAB; b.max_n_one_row = VOCAB; b.kernel_slots = prepare ? 128 : 32;
    CHECK(omega_gpu_reserve_serving(&b) == 0, "reserve failed: %s", omega_gpu_matmul_last_error());
    int prepared = -1; /* code buffers the prepare pass made: one per distinct kernel */
    if (prepare) {
        int a0 = fk.alloc_calls;
        for (unsigned i = 0; i < 6; i++) {
            uint32_t k, n;
            omega_gpu_tensor_shape(w[i], &k, &n);
            for (uint32_t m = 1; m <= SWEEP_ROWS; m++) {
                int rc = omega_gpu_matmul_prepare(m, k, n);
                CHECK(rc == OMEGA_GPU_MATMUL_OK, "prepare m=%u %ux%u rc=%d %s", m, k, n, rc, omega_gpu_matmul_last_error());
            }
        }
        prepared = fk.alloc_calls - a0;
        printf("MEASURE sweep prepare: %d driver allocations (one code buffer per distinct kernel)\n", prepared);
        omega_gpu_serving_seal();
    }
    g_nseen = 0;
    OmegaGpuAllocStats st0, st1;
    omega_gpu_session_alloc_stats(&st0);
    int a0 = fk.alloc_calls, too_large = 0, other = 0;
    for (unsigned pass = 0; pass < 2; pass++)
        for (uint32_t m = 1; m <= SWEEP_ROWS; m++)
            for (unsigned i = 0; i < 6; i++) {
                OmegaGpuMatmulInfo info;
                int rc = omega_gpu_matmul_resident_f32(m, x, w[i], y, &info);
                if (rc == OMEGA_GPU_MATMUL_TOO_LARGE) too_large++;
                else if (rc != OMEGA_GPU_MATMUL_OK && rc != OMEGA_GPU_MATMUL_CHIP_FAIL) other++; /* CHIP_FAIL = fake kernel skips writes */
                if (rc != OMEGA_GPU_MATMUL_TOO_LARGE) (void)seen(&info);
            }
    int made = fk.alloc_calls - a0;
    omega_gpu_session_alloc_stats(&st1);
    CHECK((int)(st1.allocs - st0.allocs) == made, "session alloc stats %llu != driver %d",
          (unsigned long long)(st1.allocs - st0.allocs), made);
    printf("MEASURE sweep serving (rows 1..%u x 6 shapes, twice): %d driver allocations, %d distinct kernels, %d refused, %d other\n",
           SWEEP_ROWS, made, g_nseen, too_large, other);
    CHECK(made == 0, "serving made %d driver allocations", made);
    CHECK(too_large == 0 && other == 0, "serving refused %d, other rc %d", too_large, other);
    /* measured: 95 before the blocked prefill kernel (#407), 83 with it (keyed with rt, ct; lm_head stays
     * on the single-tile kernel, its rt > 1 store offset does not fit the 24-bit immediate) */
    CHECK(g_nseen == 83, "distinct kernels %d, expected 83 for Qwen3-4B at CTA budget 256", g_nseen);
    CHECK(prepared < 0 || prepared == g_nseen, "prepare built %d kernels, serving used %d distinct", prepared, g_nseen);
    if (prepare) {
        int a1 = fk.alloc_calls;
        OmegaGpuMatmulInfo info;
        int rc = omega_gpu_matmul_resident_f32(1, x, odd, y, &info);
        CHECK(rc == OMEGA_GPU_MATMUL_TOO_LARGE && strstr(omega_gpu_matmul_last_error(), "not prepared"),
              "unprepared shape rc=%d %s", rc, omega_gpu_matmul_last_error());
        CHECK(fk.alloc_calls == a1, "unprepared shape asked the driver for memory");
        omega_gpu_serving_release();
        rc = omega_gpu_matmul_resident_f32(1, x, odd, y, &info);
        CHECK(rc != OMEGA_GPU_MATMUL_TOO_LARGE, "release did not lift the seal: %s", omega_gpu_matmul_last_error());
    }
    free(x); free(y);
    printf("gpu_serving_alloc_count_test sweep%s: %s (%d failed checks)\n", prepare ? "" : "-skip", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

/* ---- pin: a full cache of prepared kernels is never evicted. 8 slots, 8 prepared kernels (W_Q at 16..144 rows:
 * 9 row counts; since the blocked prefill kernel (#407) 64 and 128 rows share one kernel, rt 4 x ct 4 at grid_x 128),
 * then an unprepared call BEFORE any seal must be refused with no driver call, and every prepared kernel must still hit. */
static int pin_mode(void) {
    W_Q = upload(HID, QDIM); W_K = upload(HID, KVDIM);
    float *x = calloc((size_t)144 * HID, sizeof *x), *y = calloc((size_t)144 * QDIM, sizeof *y);
    CHECK(x && y, "pin buffers");
    if (g_fail) return 1;
    OmegaGpuServingBounds b = QWEN3_4B_BOUNDS;
    b.max_rows = 144; b.kernel_slots = 8;
    CHECK(omega_gpu_reserve_serving(&b) == 0, "reserve failed: %s", omega_gpu_matmul_last_error());
    const int ap = fk.alloc_calls;
    for (uint32_t m = 16; m <= 144; m += 16) {
        int rc = omega_gpu_matmul_prepare(m, HID, QDIM);
        CHECK(rc == OMEGA_GPU_MATMUL_OK, "prepare m=%u rc=%d %s", m, rc, omega_gpu_matmul_last_error());
    }
    CHECK(fk.alloc_calls - ap == 8, "premise: prepare built %d kernels, the cache holds 8", fk.alloc_calls - ap);
    int a0 = fk.alloc_calls;
    OmegaGpuMatmulInfo info;
    int rc = omega_gpu_matmul_resident_f32(1, x, W_K, y, &info);
    CHECK(rc == OMEGA_GPU_MATMUL_TOO_LARGE && strstr(omega_gpu_matmul_last_error(), "prepared kernel"),
          "unprepared call with a full pinned cache: rc=%d %s", rc, omega_gpu_matmul_last_error());
    CHECK(fk.alloc_calls == a0, "refused call asked the driver for memory");
    for (uint32_t m = 16; m <= 144; m += 16) {
        rc = omega_gpu_matmul_resident_f32(m, x, W_Q, y, &info);
        CHECK(rc != OMEGA_GPU_MATMUL_TOO_LARGE && info.kernel_cache_hit, "prepared m=%u evicted (rc=%d hit=%d)", m, rc, info.kernel_cache_hit);
    }
    CHECK(fk.alloc_calls == a0, "prepared kernels rebuilt: %d driver allocations", fk.alloc_calls - a0);
    free(x); free(y);
    printf("gpu_serving_alloc_count_test pin: %s (%d failed checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}


/* Sealed without a serving reservation: the scratch buffers could still grow, so every matmul call
 * must be refused by name before any driver allocation (review of omega#338). */
static int seal_bare_mode(void) {
    W_Q = upload(HID, QDIM);
    float *x = calloc((size_t)144 * HID, sizeof *x), *y = calloc((size_t)144 * QDIM, sizeof *y);
    CHECK(x && y, "seal-bare buffers");
    if (g_fail) return 1;
    int rc = omega_gpu_matmul_prepare(16, HID, QDIM);
    CHECK(rc == OMEGA_GPU_MATMUL_OK, "prepare m=16 rc=%d %s", rc, omega_gpu_matmul_last_error());
    omega_gpu_matmul_seal(1);
    int a0 = fk.alloc_calls;
    OmegaGpuMatmulInfo info;
    rc = omega_gpu_matmul_resident_f32(144, x, W_Q, y, &info);
    CHECK(rc == OMEGA_GPU_MATMUL_TOO_LARGE && strstr(omega_gpu_matmul_last_error(), "without a scratch reservation"),
          "sealed, unreserved, larger call: rc=%d %s", rc, omega_gpu_matmul_last_error());
    CHECK(fk.alloc_calls == a0, "sealed call without a reservation asked the driver for memory (%d)", fk.alloc_calls - a0);
    omega_gpu_matmul_seal(0);
    rc = omega_gpu_matmul_resident_f32(16, x, W_Q, y, &info);
    CHECK(rc != OMEGA_GPU_MATMUL_TOO_LARGE, "unsealed call still refused: rc=%d %s", rc, omega_gpu_matmul_last_error());
    free(x); free(y);
    printf("gpu_serving_alloc_count_test seal-bare: %s (%d failed checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
/* update: omega_gpu_tensor_update_* re-packs new values into the tensor's own device buffer. The fake's device
 * memory is host memory with GPU address = CPU address (fk.alloc_va), so the buffer is compared word for word
 * with omega_matmul_frag_pack_b of the new values. Two shapes: 72 x 1100 (k pads to 80; n is not a multiple of
 * 8 and spans three 64-tile chunks) and 4096 x 200 (the f32 staging holds 8 tiles at this k: four chunks, the
 * last ragged). Per shape: the upload's one driver allocation is the GPU-uncached one, update_f32, update_bf16
 * back, update while sealed, no driver allocation or free, shape-mismatch refusals write nothing. Then NULL
 * refusals and stale handles (closed device, then a reopened one). */
static int upd_same(const uint16_t *d, const uint16_t *want, size_t cnt, const char *what) {
    size_t diff = 0;
    for (size_t i = 0; i < cnt; i++) diff += d[i] != want[i];
    CHECK(diff == 0, "%s: %zu of %zu packed words differ from omega_matmul_frag_pack_b", what, diff, cnt);
    return diff == 0;
}
static OmegaGpuTensor *update_shape(uint32_t K, uint32_t N, uint16_t **b1_out, float **b2_out) {
    const uint32_t KP = (K + 15u) / 16u * 16u, NT = (N + 7u) / 8u;
    const size_t cnt = (size_t)K * N, packed = (size_t)KP * NT * 8u;
    uint16_t *b1 = malloc(cnt * sizeof *b1), *b2h = malloc(cnt * sizeof *b2h), *want = malloc(packed * sizeof *want);
    float *b2 = malloc(cnt * sizeof *b2);
    CHECK(b1 && b2 && b2h && want, "update buffers %ux%u", K, N);
    if (!b1 || !b2 || !b2h || !want) return NULL;
    uint32_t s = 0x5eedu ^ K ^ (N << 16);
    for (size_t i = 0; i < cnt; i++) {
        s = s * 1664525u + 1013904223u;
        b1[i] = omega_fp32_to_bf16(((float)(s >> 8) / 16777216.0f) - 0.5f);
        b2[i] = ((float)(s % 2000003u) / 1000001.0f) - 1.0f + 1e-7f * (float)(i % 7u); /* low bits that rounding must drop */
    }
    /* edge values for the update path's rounding: NaNs with payloads and sign, infinities, zeros, subnormals,
     * exact ties (odd and even), the largest finite value (rounds to infinity) */
    static const uint32_t EDGE[] = {0x7fc00001u, 0xff800001u, 0x7fbfffffu, 0x7f800000u, 0xff800000u, 0x00000000u, 0x80000000u,
                                    0x00000001u, 0x807fffffu, 0x00018000u, 0x3f808000u, 0x3f818000u, 0xbf808000u, 0x7f7fffffu, 0xff7fffffu};
    for (size_t e = 0; e < sizeof EDGE / sizeof EDGE[0]; e++) for (size_t r = 0; r < 3; r++) {
        const size_t i = r * (K / 3u) * N + (e * 73u + r) % N; /* rows 0, k/3, 2k/3; columns across the chunks */
        memcpy(&b2[i], &EDGE[e], 4);
    }
    for (size_t i = 0; i < cnt; i++) b2h[i] = omega_fp32_to_bf16(b2[i]);

    OmegaGpuTensor *t = NULL;
    const int a0 = fk.alloc_calls, u0 = fk.uncached_calls;
    int rc = omega_gpu_tensor_upload_bf16(K, N, b1, &t);
    CHECK(rc == OMEGA_GPU_MATMUL_OK && fk.alloc_calls == a0 + 1 && a0 < 16, "premise %ux%u: upload rc=%d made %d allocations (index %d)", K, N, rc, fk.alloc_calls - a0, a0);
    CHECK(fk.uncached_calls == u0 + 1, "%ux%u: the tensor's driver allocation was not the GPU-uncached one", K, N);
    if (rc != OMEGA_GPU_MATMUL_OK || fk.alloc_calls != a0 + 1 || a0 >= 16) return NULL;
    const uint16_t *d = (const uint16_t *)(uintptr_t)fk.alloc_va[a0];
    omega_matmul_frag_pack_b(b1, K, N, KP, 0, NT, want);
    upd_same(d, want, packed, "premise: upload");

    const int a1 = fk.alloc_calls, f1 = fk.free_calls;
    CHECK(omega_gpu_tensor_update_f32(t, K + 1u, N, b2) == OMEGA_GPU_MATMUL_BAD_ARGS, "update_f32 with k + 1 not refused");
    CHECK(omega_gpu_tensor_update_bf16(t, K, N - 1u, b1) == OMEGA_GPU_MATMUL_BAD_ARGS, "update_bf16 with n - 1 not refused");
    CHECK(omega_gpu_tensor_update_f32(t, N, K, b2) == OMEGA_GPU_MATMUL_BAD_ARGS, "update_f32 with k and n swapped not refused");
    upd_same(d, want, packed, "shape-mismatch refusals left the buffer");
    rc = omega_gpu_tensor_update_f32(t, K, N, b2);
    CHECK(rc == OMEGA_GPU_MATMUL_OK, "update_f32 %ux%u rc=%d %s", K, N, rc, omega_gpu_matmul_last_error());
    omega_matmul_frag_pack_b(b2h, K, N, KP, 0, NT, want);
    upd_same(d, want, packed, "update_f32");
    rc = omega_gpu_tensor_update_bf16(t, K, N, b1);
    CHECK(rc == OMEGA_GPU_MATMUL_OK, "update_bf16 %ux%u rc=%d %s", K, N, rc, omega_gpu_matmul_last_error());
    omega_matmul_frag_pack_b(b1, K, N, KP, 0, NT, want);
    upd_same(d, want, packed, "update_bf16 back");
    omega_gpu_matmul_seal(1);
    rc = omega_gpu_tensor_update_f32(t, K, N, b2);
    omega_gpu_matmul_seal(0);
    CHECK(rc == OMEGA_GPU_MATMUL_OK, "update while sealed rc=%d %s", rc, omega_gpu_matmul_last_error());
    omega_matmul_frag_pack_b(b2h, K, N, KP, 0, NT, want);
    upd_same(d, want, packed, "update_f32 while sealed");
    CHECK(fk.alloc_calls == a1 && fk.free_calls == f1, "updates touched the driver (%d allocs, %d frees)", fk.alloc_calls - a1, fk.free_calls - f1);
    uint32_t k2 = 0, n2 = 0;
    omega_gpu_tensor_shape(t, &k2, &n2);
    CHECK(k2 == K && n2 == N, "shape changed by update: %ux%u", k2, n2);
    free(b2h); free(want);
    *b1_out = b1; *b2_out = b2;
    return t;
}
static int update_mode(void) {
    OmegaGpuTensor *warm = upload(16, 8); /* opens the session, so each next upload is exactly one driver allocation */
    uint16_t *b1 = NULL, *b1b = NULL; float *b2 = NULL, *b2b = NULL;
    OmegaGpuTensor *t = update_shape(72, 1100, &b1, &b2);
    OmegaGpuTensor *t2 = update_shape(4096, 200, &b1b, &b2b);
    if (!t || !t2) return 1;

    CHECK(omega_gpu_tensor_update_f32(NULL, 72, 1100, b2) == OMEGA_GPU_MATMUL_BAD_ARGS, "update_f32 NULL tensor not refused");
    CHECK(omega_gpu_tensor_update_bf16(NULL, 72, 1100, b1) == OMEGA_GPU_MATMUL_BAD_ARGS, "update_bf16 NULL tensor not refused");
    CHECK(omega_gpu_tensor_update_f32(t, 72, 1100, NULL) == OMEGA_GPU_MATMUL_BAD_ARGS, "update_f32 NULL values not refused");
    CHECK(omega_gpu_tensor_update_bf16(t, 72, 1100, NULL) == OMEGA_GPU_MATMUL_BAD_ARGS, "update_bf16 NULL values not refused");

    /* a handle whose device was closed is stale: refused before anything is written (its memory is gone) */
    omega_gpu_device_close();
    const int a2 = fk.alloc_calls;
    int rc = omega_gpu_tensor_update_f32(t, 72, 1100, b2);
    CHECK(rc == OMEGA_GPU_MATMUL_CHIP_FAIL, "update_f32 of a stale handle rc=%d (want CHIP_FAIL)", rc);
    rc = omega_gpu_tensor_update_bf16(t, 72, 1100, b1);
    CHECK(rc == OMEGA_GPU_MATMUL_CHIP_FAIL, "update_bf16 of a stale handle rc=%d (want CHIP_FAIL)", rc);
    CHECK(fk.alloc_calls == a2, "stale update asked the driver for memory");
    OmegaGpuTensor *reopen = upload(16, 8); /* a new device: the old handle stays stale */
    rc = omega_gpu_tensor_update_f32(t, 72, 1100, b2);
    CHECK(rc == OMEGA_GPU_MATMUL_CHIP_FAIL, "update_f32 of a handle from a previous device rc=%d (want CHIP_FAIL)", rc);
    omega_gpu_tensor_free(reopen);
    omega_gpu_tensor_free(t);
    omega_gpu_tensor_free(t2);
    omega_gpu_tensor_free(warm);
    omega_gpu_device_close();
    free(b1); free(b2); free(b1b); free(b2b);
    printf("gpu_serving_alloc_count_test update: %s (%d failed checks)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}

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
    if (strcmp(mode, "sweep") == 0) return sweep_mode(1);
    if (strcmp(mode, "pin") == 0) return pin_mode();
    if (strcmp(mode, "seal-bare") == 0) return seal_bare_mode();
    if (strcmp(mode, "update") == 0) return update_mode();
    if (strcmp(mode, "sweep-skip") == 0) return sweep_mode(0);

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
