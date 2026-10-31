/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* DRA-83 (P-DRA-4 G2) host test of the prefill chunk: the row bound split (legacy 128, tensor-core 8192), the
 * AIEN_OMEGA_Q36_PF_CHUNK knob, the buffer layout bytes (Wolfram-computed constants, with and without the DRA-82 QMMA
 * regions), a refused allocation, the per-chunk streamed count (33,617,346,560 B for a 40-layer chunk touching every
 * expert), TEST mutant 7 (one chunk's expert segment skipped: red) against the baseline (green), and the grid-y split of
 * the expert row launches on the legacy and the QMMA expert path.
 * Host only: the canonical graph source is compiled in (with omega_gpu_pfq_api.c and the QMMA code generator's spec
 * checks linked), the session / command-list / GEMM / prefill-kernel entry points it reaches are stubbed below, function
 * sections + linker GC drop the rest. No device node or GPU library is opened.
 * Build with -DPFCHUNK_BASE against a tree without DRA-83 (the negative control): the same checks then run on the old
 * entry points and must fail. */
#define _GNU_SOURCE 1
#include <string.h>
#include <stdlib.h>
#include "../src/omega_gpu_q36_graph.c"
#include <stdio.h>
#include "../src/omega_gpu_prefill.h"

static int fails, checks;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* ---- stubs: the session, command list and GEMM entry points the included graph reaches ---- */
static int st_alloc, st_free, st_refuse_dev, st_last_ok_n;
static size_t st_host_bytes, st_dev_bytes;
void omega_gpu_session_lock(void) {}
void omega_gpu_session_unlock(void) {}
OmegaGpuSession *omega_gpu_session_open(void) { static char s[64]; return (OmegaGpuSession *)(void *)s; }
static char st_err[256];
void omega_gpu_session_set_error(const char *e) { snprintf(st_err, sizeof st_err, "%s", e ? e : ""); }
static int st_do_alloc(size_t b, OmegaGpuMem *m, int dev) {
    if (dev && st_refuse_dev) return -1;
    memset(m, 0, sizeof *m);
    m->cpu = calloc(1, b ? b : 1); m->va = 0x100000000ull + (uint64_t)st_alloc * 0x10000000ull; m->size = b;
    st_alloc++; st_last_ok_n++;
    if (dev) st_dev_bytes = b; else st_host_bytes = b;
    return m->cpu ? 0 : -1;
}
static int st_dev_next;
int omega_gpu_session_alloc(size_t b, OmegaGpuMem *m) { const int dev = st_dev_next; st_dev_next = !st_dev_next; return st_do_alloc(b, m, dev); }
int omega_gpu_session_alloc_cached(size_t b, OmegaGpuMem *m) { st_dev_next = 0; return st_do_alloc(b, m, 1); }
void omega_gpu_session_free(OmegaGpuMem *m) { if (m && m->cpu) { free(m->cpu); st_free++; } if (m) memset(m, 0, sizeof *m); }
uint32_t omega_native_revision(uint64_t *next, uint64_t cur) { *next = cur + 1u; return OMEGA_OK; }
int omega_gpu_fp8gemm_prepare_shape(uint32_t n, uint32_t a, uint32_t b, int c) { (void)n; (void)a; (void)b; (void)c; return 0; }
int omega_gpu_router_prepare_rows(OmegaGpuRouter *r, uint32_t xs, uint32_t lbs) { (void)r; (void)xs; (void)lbs; return OMEGA_GPU_ROUTER_OK; }
void omega_gpu_decode_attn256_test_set_multi_mutant(int on) { (void)on; }
int omega_gpu_fp8_tensor_entry(const OmegaGpuFp8Tensor *t, uint32_t expert, uint64_t *w, uint64_t *s) {
    (void)t; *w = (uint64_t)expert * 256u + 4096u; *s = *w + 128u; return 0;
}
/* the record path of pf_record_experts: every launch is captured (pf = 1: a QMMA quant/gemm launch) */
typedef struct { uint32_t gx, gy; int pf; uint64_t a0, a1; } Cap;
static Cap cap_l[4096]; static uint32_t cap_n;
static void cap_add(uint32_t gx, uint32_t gy, int pf, const uint32_t *a) {
    if (cap_n < 4096) { cap_l[cap_n] = (Cap){ .gx = gx, .gy = gy, .pf = pf, .a0 = (uint64_t)a[0] | (uint64_t)a[1] << 32,
                                              .a1 = (uint64_t)a[2] | (uint64_t)a[3] << 32 }; }
    cap_n++;
}
int omega_gpu_cmdlist_add(OmegaGpuCmdList *cl, const OmegaGpuLaunch *L) { (void)cl; cap_add(L->grid_x, L->grid_y, 0, L->args); return 0; }
uint32_t omega_gpu_cmdlist_count(const OmegaGpuCmdList *cl) { (void)cl; return cap_n; }
int omega_gpu_pf_open_locked(void) { return OMEGA_GPU_PF_OK; }
int omega_gpu_pf_kernel_locked(const OmegaPfKey *key, OmegaPfGen gen, OmegaPfKernel *out) { (void)key; (void)gen; memset(out, 0, sizeof *out); return OMEGA_GPU_PF_OK; }
int omega_gpu_pf_record_locked(struct OmegaGpuCmdList *cl, const OmegaPfKernel *k, uint32_t threads, uint32_t grid_x, uint32_t grid_y,
                               uint32_t shared_bytes, const uint32_t *args, uint32_t n_args) {
    (void)cl; (void)k; (void)threads; (void)shared_bytes; (void)n_args; cap_add(grid_x, grid_y, 1, args); return OMEGA_GPU_PF_OK;
}
int omega_gpu_fp8gemm_record(struct OmegaGpuCmdList *cl, const OmegaGpuFp8Tensor *t, uint32_t rows, uint32_t segs, uint64_t seg, uint64_t xp, uint64_t y,
                             uint64_t xm, uint64_t ym, uint32_t mt, uint32_t f, OmegaGpuFp8GemmInfo *p) {
    (void)cl; (void)t; (void)rows; (void)segs; (void)seg; (void)xp; (void)y; (void)xm; (void)ym; (void)mt; (void)f; (void)p; return 0;
}

/* ---- helpers ---- */
#define LBS 16896u   /* al(16800): the 35B router layout (shared slot) */
#define IDS_OFF 16384u
static void env(const char *k, const char *v) { if (v) setenv(k, v, 1); else unsetenv(k); }
static void env_clear(void) {
    const char *k[] = { "AIEN_OMEGA_Q36_PF_QMMA", "AIEN_OMEGA_Q36_PF_CHUNK", "AIEN_OMEGA_Q36_PF_GDN", "AIEN_OMEGA_Q36_PF_ATTN",
                        "AIEN_OMEGA_Q36_PF_MUTANT", "AIEN_OMEGA_Q36_CN_ROWS", "AIEN_OMEGA_Q36_BUF_CACHED" };
    for (unsigned i = 0; i < sizeof k / sizeof *k; i++) unsetenv(k[i]);
    g_cn_rows = -1;
}
static OmegaGpuQ36Graph *graph_new(void) {
    OmegaGpuQ36Graph *g = calloc(1, sizeof *g);
    g->lo.lb_stride = LBS; g->cfg.layers = 40;
    g->pf.gdn_multi = g->pf.attn_multi = 1; /* prefill_alloc's defaults: o on the QMMA path in every layer */
    for (uint32_t l = 0; l < 40; l++) g->layer[l].attn = (l % 4u) == 3u; /* the 35B: full attention every 4th layer */
    return g;
}
static void graph_free(OmegaGpuQ36Graph *g) { omega_gpu_session_free(&g->pf.hb); omega_gpu_session_free(&g->pf.db); free(g); }
/* host-only pf buffers for pf_group (no session): the QMMA layout (a superset of the legacy one) */
static void pf_host(OmegaGpuQ36Graph *g, uint32_t P) {
#ifndef PFCHUNK_BASE
    size_t ho = 0, dof = 0;
    const uint32_t tcg = omega_pfq_tile_cap(P * NG, 257u), tcd = omega_pfq_tile_cap(P, 1u);
    pf_layout(g, P, LBS, 1, 40u, tcg, tcd, &ho, &dof);
    g->pf.hb.cpu = calloc(1, ho); g->pf.cap = P; g->pf.qtg_cap = tcg; g->pf.qtd_cap = tcd;
#else
    /* the base tree has no layout helper: its prefill_alloc, which refuses more than 128 rows */
    const int rc = omega_gpu_q36_graph_prefill_alloc(g, P);
    CHECK(rc == OMEGA_GPU_DGRAPH_OK, "base prefill_alloc(%u) rc %d", P, rc);
    if (rc) (void)omega_gpu_q36_graph_prefill_alloc(g, 128u);
    g->pf.mutant = 0;
#endif
}
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s >> 8; }
/* 8 distinct routed picks per token; full = 1 makes every expert appear (token t also picks t % 256) */
static void picks(OmegaGpuQ36Graph *g, uint32_t n, uint32_t seed, int full) {
    uint32_t s = seed;
    for (uint32_t t = 0; t < n; t++) {
        uint32_t *id = pf_hc(g, g->pf.rlb + (size_t)t * LBS + IDS_OFF);
        for (uint32_t j = 0; j < 8; j++) {
            uint32_t e;
            for (;;) {
                e = (full && j == 0) ? t % 256u : lcg(&s) % 256u;
                int dup = 0;
                for (uint32_t k = 0; k < j; k++) dup |= id[k] == e;
                if (!dup) break;
                full = full && j != 0;
            }
            id[j] = e;
        }
    }
}
/* The MoE rows the segment tables compute: y[ymap[p]] = f(expert, xmap[p]) for each segment's rows (the shared expert
 * from the second table); then every (token, slot) row must hold f(its pick, token). Returns the rows that do not. */
static uint32_t moe_rows_wrong(OmegaGpuQ36Graph *g, uint32_t n, uint32_t segs) {
    const uint32_t R = n * NG;
    uint32_t *y = malloc(R * sizeof *y);
    for (uint32_t r = 0; r < R; r++) y[r] = 0xffffffffu;
    const Seg32 *sg = pf_hc(g, g->pf.sgu), *sd = pf_hc(g, g->pf.sdn);
    const uint32_t *xm = pf_hc(g, g->pf.xmap), *ym = pf_hc(g, g->pf.ymap);
    for (uint32_t j = 0; j < segs; j++) {
        const uint32_t e = (uint32_t)((sg[j].w_va - 4096u) / 256u);
        if (sd[j].w_va != sg[j].w_va || sd[j].cnt != sg[j].cnt) { free(y); return R; }
        for (uint32_t p = sg[j].row0; p < sg[j].row0 + sg[j].cnt; p++) y[ym[p]] = e * 65536u + xm[p];
    }
    const Seg32 *g2 = pf_hc(g, g->pf.sgu2);
    const uint32_t *xm2 = pf_hc(g, g->pf.xmap2), *ym2 = pf_hc(g, g->pf.ymap2);
    for (uint32_t p = 0; p < g2->cnt; p++) y[ym2[p]] = 256u * 65536u + xm2[p];
    uint32_t wrong = 0;
    for (uint32_t t = 0; t < n; t++) {
        const uint32_t *id = pf_hc(g, g->pf.rlb + (size_t)t * LBS + IDS_OFF);
        for (uint32_t s = 0; s < 8; s++) wrong += y[t * NG + s] != id[s] * 65536u + t;
        wrong += y[t * NG + 8] != 256u * 65536u + t;
    }
    free(y);
    return wrong;
}

#define EX ((uint64_t)3145728u)              /* Wolfram: 3*512*2048 */
#define G2_CHUNK ((uint64_t)33617346560ull)  /* Wolfram: 40*257*3*512*2048 + 2048*(30*(12288+4096)+10*(9216+4096)) */

#ifndef PFCHUNK_BASE
#define PF_GROUP(g, l, n, skip) pf_group(g, l, n, IDS_OFF, skip)
#define SKIP(g, l, pos0) pf_skip_segment(g, l, pos0)
#else
/* the base tree: pf_group has no skip argument and no mutant 7; the chunk knob and the TC bound do not exist */
#define PF_GROUP(g, l, n, skip) ((void)(skip), pf_group(g, l, n, IDS_OFF))
#define SKIP(g, l, pos0) ((g)->pf.mutant == 7 && (pos0) == 0 && (l) == 0)
#define pf_count_dense(g, l, n) ((g)->pf.streamed += (uint64_t)(((n) + 7u) / 8u) * H * (((g)->layer[l].attn ? OMEGA_GPU_Q36_ATT_MIX : OMEGA_GPU_Q36_GDN_MIX) + OMEGA_GPU_Q36_MIXER_OUT))
#endif

/* One chunk through the host half of the prefill (dense count + GROUP per layer) for 40 layers: the streamed bytes and the
 * MoE rows the tables leave wrong. The G2 gate: bytes == G2_CHUNK when every expert is touched, and 0 wrong rows. */
static void chunk(OmegaGpuQ36Graph *g, uint32_t n, uint32_t pos0, uint64_t *bytes, uint32_t *wrong) {
    const uint64_t s0 = g->pf.streamed;
    *wrong = 0;
    for (uint32_t l = 0; l < 40; l++) {
        picks(g, n, 1000u + l, 1);
        pf_count_dense(g, l, n);
        const uint32_t segs = PF_GROUP(g, l, n, SKIP(g, l, pos0));
        *wrong += moe_rows_wrong(g, n, segs);
    }
    *bytes = g->pf.streamed - s0;
}

static void test_bound_and_knob(void) {
#ifndef PFCHUNK_BASE
    uint32_t c = 0;
    env_clear();
    CHECK(omega_gpu_q36_graph_prefill_rows_max() == 128u, "legacy bound %u", omega_gpu_q36_graph_prefill_rows_max());
    CHECK(omega_gpu_q36_graph_prefill_chunk(&c) == OMEGA_GPU_DGRAPH_OK && c == 128u, "default chunk %u", c);
    env("AIEN_OMEGA_Q36_PF_CHUNK", "2048"); c = 7;
    CHECK(omega_gpu_q36_graph_prefill_chunk(&c) == OMEGA_GPU_DGRAPH_BAD_ARGS && c == 7, "2048 refused without QMMA");
    env("AIEN_OMEGA_Q36_PF_CHUNK", "64");
    CHECK(omega_gpu_q36_graph_prefill_chunk(&c) == OMEGA_GPU_DGRAPH_OK && c == 64u, "64 legacy");
    env("AIEN_OMEGA_Q36_PF_QMMA", "1");
    CHECK(omega_gpu_q36_graph_prefill_rows_max() == 8192u, "TC bound");
    const char *ok[] = { "1", "128", "2048", "4096", "8192" }; const uint32_t okv[] = { 1, 128, 2048, 4096, 8192 };
    for (unsigned i = 0; i < 5; i++) {
        env("AIEN_OMEGA_Q36_PF_CHUNK", ok[i]); c = 0;
        CHECK(omega_gpu_q36_graph_prefill_chunk(&c) == OMEGA_GPU_DGRAPH_OK && c == okv[i], "chunk %s -> %u", ok[i], c);
    }
    const char *bad[] = { "8193", "0", "", "x", "12x", "-5", "99999999999", " 64" };
    for (unsigned i = 0; i < 8; i++) {
        env("AIEN_OMEGA_Q36_PF_CHUNK", bad[i]); c = 7;
        CHECK(omega_gpu_q36_graph_prefill_chunk(&c) == OMEGA_GPU_DGRAPH_BAD_ARGS && c == 7, "chunk '%s' must be refused", bad[i]);
    }
    const char *qbad[] = { "0", "10", "yes", "" };
    for (unsigned i = 0; i < 4; i++) { env("AIEN_OMEGA_Q36_PF_QMMA", qbad[i]); CHECK(omega_gpu_q36_graph_prefill_rows_max() == 128u, "QMMA '%s' is off", qbad[i]); }
    CHECK(omega_gpu_q36_graph_prefill_chunk(NULL) == OMEGA_GPU_DGRAPH_BAD_ARGS, "NULL chunk");
    printf("bound+knob: legacy 128, TC 8192, chunk default 128, 5 accepted, 8 refused, 4 QMMA spellings off\n");
#else
    printf("bound+knob: not in the base tree (prefill_rows_max / prefill_chunk absent)\n");
    checks++; fails++;
#endif
}

static void test_layout_and_alloc(void) {
    /* Wolfram (layout of pf_layout, lbs 16896, al = 256-byte rounding): {cap, host, device} without QMMA; qx the DRA-82 QMMA
     * regions added with AIEN_OMEGA_Q36_PF_QMMA=1: 2 al((9 cap + 64) 4) + 2 al(tcg 32) + al(40 2 tcd 32) host,
     * al(4608 cap) + al(4608 cap / 128 4) device, tcg = 257 + 9 cap / 64, tcd = 1 + cap / 64 */
    static const uint64_t want[4][3] = { { 128, 3239424ull, 38584320ull }, { 2048, 51561984ull, 613171200ull },
                                         { 4096, 103106048ull, 1226063872ull }, { 8192, 206194176ull, 2451849216ull } };
    static const uint64_t qx[4][2] = { { 35328ull, 608256ull }, { 267776ull, 9732096ull }, { 515584ull, 19464192ull }, { 1011200ull, 38928384ull } };
#ifndef PFCHUNK_BASE
    env_clear();
    for (unsigned i = 0; i < 4; i++) {
        uint64_t hb = 0, db = 0;
        const int rc = omega_gpu_q36_graph_prefill_bytes((uint32_t)want[i][0], LBS, 40u, &hb, &db);
        CHECK(i == 0 ? rc == OMEGA_GPU_DGRAPH_OK && hb == want[i][1] && db == want[i][2] : rc == OMEGA_GPU_DGRAPH_BAD_ARGS,
              "layout cap %llu without QMMA: rc %d host %llu dev %llu", (unsigned long long)want[i][0], rc, (unsigned long long)hb, (unsigned long long)db);
    }
    env("AIEN_OMEGA_Q36_PF_QMMA", "1");
    for (unsigned i = 0; i < 4; i++) {
        uint64_t hb = 0, db = 0;
        CHECK(omega_gpu_q36_graph_prefill_bytes((uint32_t)want[i][0], LBS, 40u, &hb, &db) == OMEGA_GPU_DGRAPH_OK && hb == want[i][1] + qx[i][0]
              && db == want[i][2] + qx[i][1], "layout cap %llu QMMA: host %llu dev %llu", (unsigned long long)want[i][0], (unsigned long long)hb,
              (unsigned long long)db);
    }
    uint64_t hb = 0, db = 0;
    CHECK(omega_gpu_q36_graph_prefill_bytes(8193u, LBS, 40u, &hb, &db) == OMEGA_GPU_DGRAPH_BAD_ARGS, "bytes 8193");
    CHECK(omega_gpu_q36_graph_prefill_bytes(0u, LBS, 40u, &hb, &db) == OMEGA_GPU_DGRAPH_BAD_ARGS, "bytes 0");
#endif
    env_clear();
    OmegaGpuQ36Graph *g = graph_new();
    st_alloc = st_free = 0;
    CHECK(omega_gpu_q36_graph_prefill_alloc(g, 129u) == OMEGA_GPU_DGRAPH_BAD_ARGS && st_alloc == 0, "129 without QMMA");
    st_host_bytes = st_dev_bytes = 0;
    CHECK(omega_gpu_q36_graph_prefill_alloc(g, 128u) == OMEGA_GPU_DGRAPH_OK && omega_gpu_q36_graph_prefill_cap(g) == 128u, "128 legacy");
    CHECK(st_host_bytes == want[0][1] && st_dev_bytes == want[0][2], "128 asked host %zu dev %zu", st_host_bytes, st_dev_bytes);
    env("AIEN_OMEGA_Q36_PF_QMMA", "1");
    st_host_bytes = st_dev_bytes = 0;
    CHECK(omega_gpu_q36_graph_prefill_alloc(g, 8192u) == OMEGA_GPU_DGRAPH_OK && omega_gpu_q36_graph_prefill_cap(g) == 8192u, "8192 with QMMA");
    CHECK(st_host_bytes == want[3][1] + qx[3][0] && st_dev_bytes == want[3][2] + qx[3][1], "8192 asked host %zu dev %zu", st_host_bytes, st_dev_bytes);
    CHECK(st_free == 2, "the 128 buffers freed on growth (%d)", st_free);
#ifndef PFCHUNK_BASE
    CHECK(g->pf.qmma == 1, "qmma latched");
#endif
    CHECK(omega_gpu_q36_graph_prefill_alloc(g, 8193u) == OMEGA_GPU_DGRAPH_BAD_ARGS, "8193 with QMMA");
    graph_free(g);
    /* above 128 the per-token fallbacks and the native MoE are refused; at 128 they stay allowed */
    const char *fk[] = { "AIEN_OMEGA_Q36_PF_GDN", "AIEN_OMEGA_Q36_PF_ATTN", "AIEN_OMEGA_Q36_CN_ROWS", "AIEN_OMEGA_Q36_PF_MUTANT", "AIEN_OMEGA_Q36_PF_MUTANT" };
    const char *fv[] = { "0", "0", "0", "1", "3" };
    for (unsigned i = 0; i < 5; i++) {
        env_clear(); env("AIEN_OMEGA_Q36_PF_QMMA", "1"); env(fk[i], fv[i]);
        g = graph_new(); st_alloc = 0;
        CHECK(omega_gpu_q36_graph_prefill_alloc(g, 4096u) == OMEGA_GPU_DGRAPH_BAD_ARGS && st_alloc == 0, "4096 with %s=%s refused", fk[i], fv[i]);
        CHECK(omega_gpu_q36_graph_prefill_alloc(g, 128u) == OMEGA_GPU_DGRAPH_OK, "128 with %s=%s allowed", fk[i], fv[i]);
        graph_free(g);
    }
    env_clear(); env("AIEN_OMEGA_Q36_PF_QMMA", "1");
    g = graph_new(); g->native = (OmegaGpuQ36NativeGraph *)(uintptr_t)1;
    CHECK(omega_gpu_q36_graph_prefill_alloc(g, 4096u) == OMEGA_GPU_DGRAPH_BAD_ARGS, "4096 refused with the native MoE");
    g->native = NULL; graph_free(g);
    /* the session refuses the device buffer: CHIP_FAIL, cap 0, nothing left allocated */
    g = graph_new(); st_alloc = st_free = 0; st_refuse_dev = 1;
    const int rc = omega_gpu_q36_graph_prefill_alloc(g, 4096u);
    st_refuse_dev = 0;
    CHECK(rc == OMEGA_GPU_DGRAPH_CHIP_FAIL && omega_gpu_q36_graph_prefill_cap(g) == 0 && !g->pf.hb.cpu && !g->pf.db.cpu && st_alloc == st_free,
          "refused allocation: rc %d cap %u hb %p db %p alloc %d free %d", rc, omega_gpu_q36_graph_prefill_cap(g), g->pf.hb.cpu, g->pf.db.cpu, st_alloc, st_free);
    graph_free(g);
    env_clear();
    printf("layout+alloc: bytes at 128/2048/4096/8192 match Wolfram; 8192 needs QMMA; fallbacks/native refused above 128; refusal frees\n");
}

static void test_streamed_and_mutant(void) {
    env_clear();
    OmegaGpuQ36Graph *g = graph_new();
    /* legacy chunk (128 rows): the per-pass count, unchanged */
    pf_host(g, 4096u);
    picks(g, 128u, 7u, 0);
    uint32_t cnt[256] = {0};
    for (uint32_t t = 0; t < 128u; t++) { const uint32_t *id = pf_hc(g, g->pf.rlb + (size_t)t * LBS + IDS_OFF); for (int s = 0; s < 8; s++) cnt[id[s]]++; }
    uint64_t legacy = (uint64_t)((128u + 3u) / 4u) * EX;
    for (int e = 0; e < 256; e++) legacy += (uint64_t)((cnt[e] + 3u) / 4u) * EX;
    uint64_t s0 = g->pf.streamed;
    (void)PF_GROUP(g, 0, 128u, 0);
    CHECK(g->pf.streamed - s0 == legacy, "legacy expert count %llu want %llu", (unsigned long long)(g->pf.streamed - s0), (unsigned long long)legacy);
#ifndef PFCHUNK_BASE
    CHECK(g->pf.passes == legacy, "legacy passes == streamed");
#endif
    /* the tensor-core chunk: every touched expert once, the shared once, the dense once per layer */
    g->pf.streamed = 0;
#ifndef PFCHUNK_BASE
    g->pf.passes = 0;
#endif
    const uint32_t N = g->pf.cap; /* 4096 (the base tree: 128) */
#ifndef PFCHUNK_BASE
    g->pf.qmma = 1;
#endif
    uint64_t bytes = 0; uint32_t wrong = 0;
    chunk(g, N, 0u, &bytes, &wrong);
    CHECK(bytes == G2_CHUNK, "G2 chunk bytes %llu want %llu (Wolfram)", (unsigned long long)bytes, (unsigned long long)G2_CHUNK);
    CHECK(wrong == 0, "baseline MoE rows wrong %u", wrong);
    printf("G2 baseline (4096-token chunk, 40 layers): streamed %llu B (want %llu), wrong rows %u -> %s\n", (unsigned long long)bytes,
           (unsigned long long)G2_CHUNK, wrong, bytes == G2_CHUNK && wrong == 0 ? "GREEN" : "RED");
#ifndef PFCHUNK_BASE
    printf("  passes as launched (one weight pass per 64-row QMMA tile): %llu B\n", (unsigned long long)g->pf.passes);
#endif
    /* mutant 7: the chunk at pos0 0 loses one expert segment (layer 0, highest id): the gate goes red; a later chunk is clean */
    g->pf.mutant = 7;
    CHECK(SKIP(g, 0, 0) && !SKIP(g, 1, 0) && !SKIP(g, 0, 4096), "mutant 7 fires at layer 0 of the pos0-0 chunk only");
    chunk(g, N, 0u, &bytes, &wrong);
    const int red = bytes != G2_CHUNK || wrong != 0;
    CHECK(red, "mutant 7 must be red (bytes %llu wrong %u)", (unsigned long long)bytes, wrong);
    CHECK(bytes == G2_CHUNK - EX && wrong > 0, "mutant 7: one expert short (%llu) and its rows unwritten (%u)", (unsigned long long)(G2_CHUNK - bytes), wrong);
    printf("G2 mutant 7: streamed %llu B (short %llu), wrong rows %u -> %s\n", (unsigned long long)bytes, (unsigned long long)(G2_CHUNK - bytes), wrong,
           red ? "RED" : "GREEN");
    chunk(g, N, N, &bytes, &wrong);
    CHECK(bytes == G2_CHUNK && wrong == 0, "mutant 7 leaves the second chunk alone");
    g->pf.mutant = 0;
    free(g->pf.hb.cpu); g->pf.hb.cpu = NULL; free(g);
}

static void test_grid_split(void) {
#ifndef PFCHUNK_BASE
    /* the expert rows of an 8192-token chunk (73728) exceed one launch's grid y: SWIGLU (and on the legacy path the ACT PERM)
     * split at 65535 rows, the pieces' row pointers advanced by r0 rows; a 128-token chunk stays one launch each. The QMMA
     * path (DRA-82) has no PERM: quant(n x' rows), gemm(gate_up tiles), SWIGLU, quant(R act rows), gemm(down tiles). */
    static const uint32_t ns[4] = { 128u, 8192u, 128u, 8192u };
    for (unsigned i = 0; i < 4; i++) {
        const int qm = i >= 2;
        env_clear(); env("AIEN_OMEGA_Q36_PF_QMMA", "1");
        OmegaGpuQ36Graph *g = graph_new();
        pf_host(g, ns[i]);
        g->pf.qmma = qm;
        g->cl = (OmegaGpuCmdList *)(uintptr_t)1; /* a handle the stubbed record calls never dereference (the QMMA record refuses NULL) */
        const uint32_t n = ns[i], R = n * NG;
        (void)PF_GROUP(g, 0, n, 0);
        cap_n = 0;
        const int rc = pf_record_experts(g, 0, 0, n, 1, 0, 0);
        uint32_t sw = 0, pa = 0, maxy = 0, rows_sw = 0, rows_pa = 0, pfn = 0; int ptr_ok = 1, pf_ok = 1;
        const uint64_t guN = pf_d(g, g->pf.guN), actN = pf_d(g, g->pf.actN), actp = pf_d(g, g->pf.actp);
        for (uint32_t k = 0; k < cap_n && k < 4096; k++) {
            if (cap_l[k].gy > maxy) maxy = cap_l[k].gy;
            if (cap_l[k].pf) { /* quant: grid x = rows; gemm: grid y = the expert tiles */
                const uint32_t want_gx = pfn == 0 ? n : R;
                pf_ok &= (pfn & 1u) ? cap_l[k].gy == g->pf.qtg_n : cap_l[k].gx == want_gx;
                pfn++;
            } else if (cap_l[k].a0 >= guN && cap_l[k].a0 < actN && cap_l[k].gy > n) { /* SWIGLU: gu, act */
                ptr_ok &= cap_l[k].a0 == guN + (uint64_t)rows_sw * 2u * EI * 4u && cap_l[k].a1 == actN + (uint64_t)rows_sw * EI * 4u;
                rows_sw += cap_l[k].gy; sw++;
            } else if (cap_l[k].a0 >= actN && cap_l[k].a0 < actp && cap_l[k].gy > n) { /* PERM ACT: in, tab */
                ptr_ok &= cap_l[k].a0 == actN + (uint64_t)rows_pa * EI * 4u;
                rows_pa += cap_l[k].gy; pa++;
            }
        }
        const uint32_t nsw = (R + 65534u) / 65535u;
        CHECK(rc == OMEGA_GPU_DGRAPH_OK && maxy <= 65535u && rows_sw == R && ptr_ok && sw == nsw
              && (qm ? pa == 0 && pfn == 4 && pf_ok && g->pf.qtg_n > 0 : rows_pa == R && pa == sw && pfn == 0),
              "%s n %u: rc %d maxy %u swiglu %u launches/%u rows, perm %u/%u, ptrs %d, qmma launches %u ok %d", qm ? "qmma" : "legacy", n, rc, maxy,
              sw, rows_sw, pa, rows_pa, ptr_ok, pfn, pf_ok);
        printf("grid split %s n=%u: R=%u -> swiglu %u launch(es), perm %u launch(es), qmma %u launch(es) (%u expert tiles), max grid y %u\n",
               qm ? "qmma" : "legacy", n, R, sw, pa, pfn, g->pf.qtg_n, maxy);
        free(g->pf.hb.cpu); free(g);
    }
    env_clear();
#endif
}

int main(void) {
    test_bound_and_knob();
    test_layout_and_alloc();
    test_streamed_and_mutant();
    test_grid_split();
    printf("gpu_q36_pfchunk_test: %d/%d checks pass%s\n", checks - fails, checks, fails ? "" : " (PASS)");
    return fails ? 1 : 0;
}
