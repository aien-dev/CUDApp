/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* gpu_pfq_test.c (DRA-82, CUDApp #220): the tiled mma.sync FP8 prefill path, omega_gpu_pfq_api.h.
 *   --host-only  tiler, launch plans, the pre-DRA-82 kernel words (digest of CUDApp main 4bbccf0), the x64 quantiser
 *                identity, the C reference against a double oracle
 *   --sim        the planned kernels and arguments in the warp simulator: quantiser row stride, row map (rowidx), output
 *                map (ymap), x64 scales, segments longer than a tile, both issue orders; bit-identical to the C reference,
 *                untouched output rows left alone; AIEN_OMEGA_PFQ_MUTANT 1..4 each caught
 *   --chip       (inside a GB10 window only) the Qwen3.6 expert shapes of one prefill chunk through a command list:
 *                quantiser and GEMM outputs bit-identical to the reference, mutants caught, per-launch GPU time
 * AIEN_OMEGA_PFQ_MUTANT set in the environment applies to every planned launch, so a clean case then FAILS: the
 * end-to-end check of the knob. Verdict line last: "GPU_PFQ: PASS|FAIL". */
#include "omega_gpu_pfq_api.h"
#include "omega_gpu_prefill.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_cmdlist.h"
#include "bw_warp_sim_tc.h"
#include "pfq_legacy_words.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* printed by tests/pfq_legacy_words.h built against CUDApp main 4bbccf0 (evidence/DRA-82-pfq/golden-main-4bbccf0.txt) */
#define PFQ_LEGACY_PROGRAMS 264
#define PFQ_LEGACY_DIGEST 0x0dd37a9ee7562178ull
#define PAD OMEGA_PFQ_MAP_PAD

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static int env_mutant;

/* the quantiser's header semantics (tests/gpu_qgemm_test.c pq_ref; mutant 2 = c 0) */
static void pq_ref(const float *x, uint32_t k, uint8_t *q, float *s, int mutant) {
    for (uint32_t b = 0; b < k / 128; b++) {
        float m = ldexpf(1.0f, -100);
        for (uint32_t j = 0; j < 128; j++) m = fmaxf(m, fabsf(x[128 * b + j]));
        uint32_t mb = f2u(m), E = mb >> 23, M = mb & 0x7fffff, c = mutant == 2 ? 0 : (M + 0x1fffff) >> 23;
        s[b] = u2f((E + c - 8) << 23);
        const float inv = u2f((262 - E - c) << 23);
        for (uint32_t j = 0; j < 128; j++) q[128 * b + j] = qm_e4m3_rne_sat(x[128 * b + j] * inv);
    }
}
static float rand_act(void) {
    float v = ((float)(rnd() & 0xffffff) / 16777216.0f) * 2.0f - 1.0f;
    int r = rnd() % 32; if (r == 0) v *= 300.0f; else if (r == 1) v *= 1e-3f; else if (r == 2) v = 0.0f;
    return v;
}
static uint8_t rand_e4m3(void) { for (;;) { uint8_t c = (uint8_t)rnd(); if ((c & 0x7f) != 0x7f) return c; } }

/* ---------------- one grouped expert GEMM problem ---------------- */
typedef struct {
    uint32_t E, n, k, KB, M, XR, YR, indexed, ymapped, x64;
    uint8_t *w; uint16_t *sbf; float *swf;  /* [E][n][k] e4m3, scale_inv bf16, GEMV-tensor scales (f32 x 2^56) */
    float *x;                               /* XR true activation rows */
    uint8_t *xq; float *xs, *xs64;          /* XR + PAD rows: e4m3, true scales, the scales of 2^64 x */
    uint32_t *eid, *ridx, *ymap;            /* M + PAD entries */
    uint32_t *counts; OmegaPfqTile *seg;    /* one segment per expert, in order (zero counts kept) */
    uint32_t x64_mismatch;                  /* q(2^64 x) != q(x) or s(2^64 x) != 2^64 s(x) outside all-zero blocks */
} Prob;

static void prob_make(Prob *P, uint32_t E, uint32_t n, uint32_t k, const uint32_t *counts, uint32_t XR, uint32_t indexed, uint32_t ymapped, uint32_t YR) {
    memset(P, 0, sizeof *P);
    uint32_t M = 0; for (uint32_t e = 0; e < E; e++) M += counts[e];
    P->E = E; P->n = n; P->k = k; P->KB = k / 128; P->M = M; P->indexed = indexed; P->ymapped = ymapped;
    P->XR = indexed ? XR : M; P->YR = ymapped ? YR : M;
    const size_t wn = (size_t)E * n * k, sn = (size_t)E * (n / 128) * P->KB, XA = (size_t)P->XR + PAD;
    P->w = malloc(wn); P->sbf = malloc(sn * 2); P->swf = malloc(sn * 4);
    for (size_t i = 0; i < wn; i++) P->w[i] = rand_e4m3();
    for (size_t i = 0; i < sn; i++) { float s = ldexpf(1.0f + (float)(rnd() & 127) / 128.0f, -(int)(rnd() % 12) - 4); P->sbf[i] = (uint16_t)(f2u(s) >> 16); }
    if (omega_fp8_pack_scales(P->sbf, sn, P->swf, 0) != 0) { printf("scale pack refused\n"); exit(2); }
    P->x = calloc(XA * k, 4); P->xq = calloc(XA * k, 1); P->xs = calloc(XA * P->KB, 4); P->xs64 = calloc(XA * P->KB, 4);
    for (size_t i = 0; i < (size_t)P->XR * k; i++) P->x[i] = rand_act();
    if (P->XR > 3) memset(P->x + 3 * (size_t)k, 0, k * 4); /* an all-zero row: the amax floor */
    if (P->XR > 5) P->x[5 * (size_t)k + 77] = 3.0e4f;       /* an outlier */
    float *x64 = malloc(k * 4); uint8_t *q64 = malloc(k);
    for (size_t r = 0; r < XA; r++) {
        pq_ref(P->x + r * k, k, P->xq + r * k, P->xs + r * P->KB, 0);
        for (uint32_t j = 0; j < k; j++) x64[j] = ldexpf(P->x[r * k + j], 64);
        pq_ref(x64, k, q64, P->xs64 + r * P->KB, 0);
        for (uint32_t b = 0; b < P->KB; b++) {
            int zero = 1; for (uint32_t j = 0; j < 128; j++) if (P->x[r * k + 128 * b + j] != 0.0f) zero = 0;
            if (memcmp(q64 + 128 * b, P->xq + r * k + 128 * b, 128)) P->x64_mismatch++;
            else if (!zero && f2u(P->xs64[r * P->KB + b]) != f2u(ldexpf(P->xs[r * P->KB + b], 64))) P->x64_mismatch++;
        }
    }
    free(x64); free(q64);
    P->eid = calloc(M + PAD, 4); P->ridx = calloc(M + PAD, 4); P->ymap = calloc(M + PAD, 4);
    P->counts = malloc(E * 4); memcpy(P->counts, counts, E * 4);
    P->seg = calloc(E, sizeof *P->seg);
    uint32_t row = 0;
    for (uint32_t e = 0; e < E; e++) {
        P->seg[e].w_va = (uintptr_t)(P->w + (size_t)e * n * k); P->seg[e].s_va = (uintptr_t)(P->swf + (size_t)e * (n / 128) * P->KB);
        P->seg[e].row0 = row; P->seg[e].cnt = counts[e];
        for (uint32_t q = 0; q < counts[e]; q++) P->eid[row + q] = e;
        row += counts[e];
    }
    for (uint32_t p = 0; p < M; p++) P->ridx[p] = indexed ? rnd() % P->XR : p;
    uint32_t *perm = malloc((size_t)P->YR * 4); for (uint32_t i = 0; i < P->YR; i++) perm[i] = i;
    for (uint32_t i = P->YR; i > 1; i--) { uint32_t j = rnd() % i, t = perm[i - 1]; perm[i - 1] = perm[j]; perm[j] = t; }
    for (uint32_t p = 0; p < M; p++) P->ymap[p] = ymapped ? perm[p] : p;
    free(perm);
}
static void prob_free(Prob *P) {
    free(P->w); free(P->sbf); free(P->swf); free(P->x); free(P->xq); free(P->xs); free(P->xs64);
    free(P->eid); free(P->ridx); free(P->ymap); free(P->counts); free(P->seg);
}

/* the kernel's arithmetic for routed row p, output column c (weight row c), with the TRUE activation scales */
static const QmmaModel GB10Q = { 32, 26, 0, 2, 0 }; /* chip-exact, evidence OWN30B-L1G-QMMA-PFGEMM-c742aa13 */
static float qref(const Prob *P, uint32_t p, uint32_t c) {
    const uint32_t k = P->k, KB = P->KB, e = P->eid[p], s = P->ridx[p];
    const uint8_t *xr = P->xq + (size_t)s * k, *wr = P->w + ((size_t)e * P->n + c) * k;
    const float *swt = P->swf + ((size_t)e * (P->n / 128) + c / 128) * KB;
    float acc = 0.0f;
    for (uint32_t kb = 0; kb < KB; kb++) {
        const float sc = P->xs[(size_t)s * KB + kb] * (swt[kb] * 0x1p-56f);
        for (uint32_t h = 0; h < 2; h++) {
            float T = 0.0f;
            for (uint32_t v = 0; v < 2; v++) {
                float av[32], bv[32];
                for (uint32_t t = 0; t < 4; t++) for (uint32_t j = 0; j < 4; j++) {
                    const uint32_t o = 128 * kb + 64 * h + 16 * t + 8 * v + j;
                    av[4 * t + j] = qm_e4m3(xr[o]); bv[4 * t + j] = qm_e4m3(wr[o]);
                    av[16 + 4 * t + j] = qm_e4m3(xr[o + 4]); bv[16 + 4 * t + j] = qm_e4m3(wr[o + 4]);
                }
                T = qm_dot32(&GB10Q, av, bv, T);
            }
            acc = fmaf(T, sc, acc);
        }
    }
    return acc;
}
/* double oracle: original bf16 weight scales, dequantised activations */
static double qoracle(const Prob *P, uint32_t p, uint32_t c, double *mag) {
    const uint32_t k = P->k, KB = P->KB, e = P->eid[p], s = P->ridx[p];
    double acc = 0, m = 0;
    for (uint32_t j = 0; j < k; j++) {
        const double sw = u2f((uint32_t)P->sbf[((size_t)e * (P->n / 128) + c / 128) * KB + j / 128] << 16);
        const double t = (double)qm_e4m3(P->w[((size_t)e * P->n + c) * k + j]) * sw * qm_e4m3(P->xq[(size_t)s * k + j]) * P->xs[(size_t)s * KB + j / 128];
        acc += t; m += fabs(t);
    }
    *mag = m; return acc;
}
/* expected y (YR x n f32): routed rows at their map rows, every other row the 0xee fill */
static uint32_t *expect_y(const Prob *P) {
    uint32_t *y = malloc((size_t)P->YR * P->n * 4); memset(y, 0xee, (size_t)P->YR * P->n * 4);
    for (uint32_t p = 0; p < P->M; p++) for (uint32_t c = 0; c < P->n; c++) y[(size_t)P->ymap[p] * P->n + c] = f2u(qref(P, p, c));
    return y;
}
static size_t ydiff(const Prob *P, const uint32_t *got, const uint32_t *want) {
    size_t bad = 0;
    for (size_t i = 0; i < (size_t)P->YR * P->n; i++) if (got[i] != want[i]) { if (getenv("PFDBG") && bad < 6) printf("  y[%zu][%zu] got %08x want %08x\n", i / P->n, i % P->n, got[i], want[i]); bad++; }
    return bad;
}

/* ---------------- host ---------------- */
static int tiler_check(const OmegaPfqTile *seg, uint32_t nseg, uint32_t rows, const OmegaPfqTile *t, uint32_t nt) {
    int bad = 0; uint8_t *seen = calloc(rows + 1, 1);
    for (uint32_t i = 0; i < nt; i++) {
        if (t[i].cnt == 0 || t[i].cnt > OMEGA_PFQ_BM || t[i].pad) bad++;
        const OmegaPfqTile *s = NULL;
        for (uint32_t j = 0; j < nseg; j++) if (seg[j].cnt && t[i].row0 >= seg[j].row0 && t[i].row0 < seg[j].row0 + seg[j].cnt) s = &seg[j];
        if (!s || t[i].w_va != s->w_va || t[i].s_va != s->s_va || t[i].row0 + t[i].cnt > s->row0 + s->cnt) { bad++; continue; }
        for (uint32_t r = 0; r < t[i].cnt; r++) seen[t[i].row0 + r]++;
    }
    for (uint32_t r = 0; r < rows; r++) if (seen[r] != 1) bad++;
    free(seen);
    return bad;
}
static int host_tiler(void) {
    static const uint32_t cnt[] = { 0, 1, 63, 64, 65, 130, 7 };
    const uint32_t ns = sizeof cnt / sizeof cnt[0];
    OmegaPfqTile seg[7], t[32]; uint32_t row = 0, want = 0;
    for (uint32_t i = 0; i < ns; i++) { memset(&seg[i], 0, sizeof seg[i]); seg[i].w_va = 0x1000u * (i + 1); seg[i].s_va = 0x100000u * (i + 1); seg[i].row0 = row; seg[i].cnt = cnt[i]; row += cnt[i]; want += (cnt[i] + 63) / 64; }
    const uint32_t cap = omega_pfq_tile_cap(row, ns), nt = omega_pfq_tiles(seg, ns, t, 32);
    const int bad = tiler_check(seg, ns, row, t, nt);
    const int over = omega_pfq_tiles(seg, ns, t, want - 1) != 0, null = omega_pfq_tiles(NULL, ns, t, 32) != 0;
    const int ok = nt == want && cap >= want && !bad && !over && !null;
    printf("%s tiler: segments {0,1,63,64,65,130,7} -> %u tiles (want %u, cap %u), %d coverage errors, cap-1 %s, NULL %s\n", ok ? "PASS" : "FAIL",
           nt, want, cap, bad, over ? "ACCEPTED" : "refused", null ? "ACCEPTED" : "refused");
    return ok ? 0 : 1;
}
static uint64_t words_hash(int qg, const void *spec) {
    BlackwellIRProgram *prog = calloc(1, sizeof *prog); uint8_t *code = malloc(1u << 20); uint64_t h = 0xcbf29ce484222325ull;
    const int cg = qg ? omega_blackwell_codegen_qgemm_prog(spec, prog) : omega_blackwell_codegen_pq_prog(spec, prog);
    if (cg != 0 || pfq_encode_hash(prog, code, 1u << 20, &h, 0) != 0) h = 0;
    free(prog); free(code); return h;
}
static int host_words(void) {
    uint64_t h = 0; const int n = pfq_legacy_digest(&h);
    const int same = n == PFQ_LEGACY_PROGRAMS && h == PFQ_LEGACY_DIGEST;
    printf("%s pre-DRA-82 qg/pq kernels: %d programs, digest 0x%016llx (main 4bbccf0: %d, 0x%016llx)\n", same ? "PASS" : "FAIL",
           n, (unsigned long long)h, PFQ_LEGACY_PROGRAMS, (unsigned long long)PFQ_LEGACY_DIGEST);
    /* negative control: the new fields do change the words, so the digest above would see them leak into legacy specs */
    OmegaQgSpec b; memset(&b, 0, sizeof b); b.k = 2048; b.rt = 4; b.warps = 8; b.epi = OMEGA_QG_EPI_F32; b.n = 1024; b.wait = 45;
    OmegaQgSpec y = b, x = b; y.ymap = 1; x.x64 = 1;
    OmegaPqSpec p; memset(&p, 0, sizeof p); p.k = 2048; OmegaPqSpec ps = p; ps.stride = 2048 + 68;
    const uint64_t hb = words_hash(1, &b), hy = words_hash(1, &y), hx = words_hash(1, &x), hp = words_hash(0, &p), hs = words_hash(0, &ps);
    const int ctl = hb && hy && hx && hp && hs && hb != hy && hb != hx && hy != hx && hp != hs;
    printf("%s negative control: qg ymap=1 / x64=1 and pq stride change the encoded words (%016llx %016llx %016llx | %016llx %016llx)\n", ctl ? "PASS" : "FAIL",
           (unsigned long long)hb, (unsigned long long)hy, (unsigned long long)hx, (unsigned long long)hp, (unsigned long long)hs);
    return (same ? 0 : 1) + (ctl ? 0 : 1);
}
static int host_plans(void) {
    int bad = 0; OmegaQgSpec s; OmegaPqSpec q; OmegaPfqLaunch L;
    OmegaPfqGemm c = { .n = 1024, .k = 2048, .x64 = 1, .ntiles = 21, .tiles_va = 0x10, .rowidx_va = 0x20, .xq_va = 0x30, .xs_va = 0x40, .ymap_va = 0x1234567890ull, .y_va = 0x50 };
    if (omega_pfq_gemm_plan(&c, &s, &L) != OMEGA_GPU_PF_OK || L.nargs != OMEGA_QG_ARGS_WORDS_YMAP || L.args[12] != 0x34567890u || L.args[13] != 0x12u ||
        L.args[10] != 1024 || L.args[11] != 0 || L.args[6] != 0x50 || L.args[8] != 0x40 || L.threads != 256 || L.grid_x != 1024 / omega_qg_span(&s) ||
        L.grid_y != 21 || L.shared != omega_qg_shared_bytes(&s) || !s.ymap || !s.x64 || s.direct || s.epi != OMEGA_QG_EPI_F32) { printf("  gemm plan (ymap) wrong\n"); bad++; }
    c.ymap_va = 0; c.rowidx_va = 0; c.x64 = 0;
    if (omega_pfq_gemm_plan(&c, &s, &L) != OMEGA_GPU_PF_OK || L.nargs != OMEGA_QG_ARGS_WORDS || s.ymap || !s.direct || s.x64) { printf("  gemm plan (plain) wrong\n"); bad++; }
    OmegaPfqGemm r = c; r.n = 1000; if (omega_pfq_gemm_plan(&r, &s, &L) != OMEGA_GPU_PF_BAD_ARGS) { printf("  n 1000 accepted\n"); bad++; }
    r = c; r.x64 = 2; if (omega_pfq_gemm_plan(&r, &s, &L) != OMEGA_GPU_PF_BAD_ARGS) { printf("  x64 2 accepted\n"); bad++; }
    r = c; r.ntiles = 70000; if (omega_pfq_gemm_plan(&r, &s, &L) != OMEGA_GPU_PF_TOO_LARGE) { printf("  70000 tiles accepted\n"); bad++; }
    OmegaPfqQuant u = { .k = 2048, .rows = 128, .stride = 2048 + 1032, .x_va = 1, .q_va = 2, .s_va = 3 };
    if (omega_pfq_quant_plan(&u, &q, &L) != OMEGA_GPU_PF_OK || q.stride != 3080 || L.threads != 512 || L.grid_x != 128 || L.grid_y != 1 || L.nargs != 6) { printf("  quant plan wrong\n"); bad++; }
    u.k = 5120; u.stride = 0;
    if (omega_pfq_quant_plan(&u, &q, &L) != OMEGA_GPU_PF_OK || q.groups != 20 || L.threads != 640 || L.grid_y != 2) { printf("  quant plan (split row) wrong\n"); bad++; }
    u.k = 2048; u.stride = 2040; if (omega_pfq_quant_plan(&u, &q, &L) != OMEGA_GPU_PF_BAD_ARGS) { printf("  stride < k accepted\n"); bad++; }
    u.stride = 2050; if (omega_pfq_quant_plan(&u, &q, &L) != OMEGA_GPU_PF_BAD_ARGS) { printf("  stride 2050 accepted\n"); bad++; }
    printf("%s launch plans: argument words, grids, refusals (%d wrong)\n", bad ? "FAIL" : "PASS", bad);
    return bad ? 1 : 0;
}
static int host_ref(void) {
    static const uint32_t cnt[] = { 30, 0, 9, 70 };
    Prob P; prob_make(&P, 4, 256, 384, cnt, 40, 1, 1, 120);
    double worst = 0;
    for (uint32_t p = 0; p < P.M; p += 3) for (uint32_t c = 0; c < P.n; c++) { double m, o = qoracle(&P, p, c, &m); double d = fabs(qref(&P, p, c) - o) / (m + 1e-30); if (d > worst) worst = d; }
    const int ok = worst < 1e-6;
    printf("%s C reference vs double oracle: worst error %.3g of the sum of |terms|\n", ok ? "PASS" : "FAIL", worst);
    const int x64ok = P.x64_mismatch == 0;
    printf("%s x64 quantiser identity: q(2^64 x) == q(x), s(2^64 x) == 2^64 s(x) on %u rows x %u blocks (%u differ; all-zero blocks keep the floor scale, q 0)\n",
           x64ok ? "PASS" : "FAIL", P.XR + PAD, P.KB, P.x64_mismatch);
    /* INFO (no gate): what the activation quantiser costs against the incumbent GEMV's unquantised f32 activations, on these
     * synthetic rows (uniform +-1 with 1/32 x300 outliers, 1/32 x1e-3, 1/32 zeros; NOT model activations) */
    double wa = 0, se = 0, sy = 0;
    for (uint32_t p = 0; p < P.M; p += 3)
        for (uint32_t c = 0; c < P.n; c++) {
            const uint32_t e = P.eid[p], s = P.ridx[p];
            double a = 0, m = 0;
            for (uint32_t j = 0; j < P.k; j++) {
                const double sw = u2f((uint32_t)P.sbf[((size_t)e * (P.n / 128) + c / 128) * P.KB + j / 128] << 16);
                const double t = (double)qm_e4m3(P.w[((size_t)e * P.n + c) * P.k + j]) * sw * P.x[(size_t)s * P.k + j];
                a += t; m += fabs(t);
            }
            const double d = fabs((double)qref(&P, p, c) - a);
            if (m > 0 && d / m > wa) wa = d / m;
            se += d * d; sy += a * a;
        }
    printf("INFO activation e4m3 (per row, per 128-col block, power-of-two scale) vs f32 activations, synthetic rows k %u: worst |dy| %.3g of the sum of |terms|, rms |dy|/rms |y| %.3g\n",
           P.k, wa, sy > 0 ? sqrt(se / sy) : 0.0);
    prob_free(&P);
    return (ok ? 0 : 1) + (x64ok ? 0 : 1);
}

/* ---------------- warp simulator ---------------- */
static int sim_quant(uint32_t k, uint32_t stride, uint32_t rows, int x64, size_t *bad, char *msg, size_t msgn) {
    const uint32_t XS = stride ? stride : k, KB = k / 128;
    float *x = malloc((size_t)rows * XS * 4), *xin = malloc((size_t)rows * XS * 4);
    for (size_t i = 0; i < (size_t)rows * XS; i++) x[i] = (i % XS) < k ? rand_act() : 1e30f; /* the gap columns must never be read */
    if (rows > 3) for (uint32_t j = 0; j < k; j++) x[3 * (size_t)XS + j] = 0.0f;
    for (size_t i = 0; i < (size_t)rows * XS; i++) xin[i] = (x64 && (i % XS) < k) ? ldexpf(x[i], 64) : x[i];
    uint8_t *q = malloc((size_t)rows * k), *rq = malloc(k); float *s = malloc((size_t)rows * KB * 4), *rs = malloc(KB * 4);
    memset(q, 0xee, (size_t)rows * k); memset(s, 0xee, (size_t)rows * KB * 4);
    OmegaPfqQuant u = { .k = k, .rows = rows, .stride = stride, .x_va = (uintptr_t)xin, .q_va = (uintptr_t)q, .s_va = (uintptr_t)s };
    OmegaPqSpec sp; OmegaPfqLaunch L; int rc = -1;
    static BlackwellIRProgram prog;
    if (omega_pfq_quant_plan(&u, &sp, &L) != OMEGA_GPU_PF_OK) snprintf(msg, msgn, "plan refused");
    else if (omega_blackwell_codegen_pq_prog(&sp, &prog) != 0) snprintf(msg, msgn, "codegen refused");
    else {
        static WsSim ws; memset(&ws, 0, sizeof ws);
        ws.p = &prog; ws.threads = L.threads;
        for (uint32_t i = 0; i < L.nargs; i++) ws.cbank[0x380 / 4 + i] = L.args[i];
        ws.ranges[0].lo = (uintptr_t)xin; ws.ranges[0].hi = (uintptr_t)(xin + (size_t)rows * XS);
        ws.ranges[1].lo = (uintptr_t)q; ws.ranges[1].hi = (uintptr_t)(q + (size_t)rows * k);
        ws.ranges[2].lo = (uintptr_t)s; ws.ranges[2].hi = (uintptr_t)(s + (size_t)rows * KB);
        ws.nranges = 3;
        rc = ws_run(&ws, L.grid_x, L.grid_y) ? -1 : 0;
        if (rc) snprintf(msg, msgn, "simulator %.150s", ws.msg);
    }
    *bad = 0;
    if (!rc) for (uint32_t r = 0; r < rows; r++) {
        pq_ref(xin + (size_t)r * XS, k, rq, rs, 0);
        for (uint32_t j = 0; j < k; j++) if (rq[j] != q[(size_t)r * k + j]) (*bad)++;
        for (uint32_t b = 0; b < KB; b++) if (f2u(rs[b]) != f2u(s[(size_t)r * KB + b])) (*bad)++;
    }
    free(x); free(xin); free(q); free(rq); free(s); free(rs);
    return rc;
}
/* plans, generates and simulates one gemm of P; y gets YR x n f32 */
static int sim_gemm(const Prob *P, int x64, int order, uint32_t *y, char *msg, size_t msgn) {
    const uint32_t cap = omega_pfq_tile_cap(P->M, P->E);
    OmegaPfqTile *t = calloc(cap + 1, sizeof *t);
    const uint32_t nt = omega_pfq_tiles(P->seg, P->E, t, cap);
    const float *xs = x64 ? P->xs64 : P->xs;
    memset(y, 0xee, (size_t)P->YR * P->n * 4);
    OmegaPfqGemm c = { .n = P->n, .k = P->k, .x64 = (uint32_t)x64, .ntiles = nt, .tiles_va = (uintptr_t)t, .rowidx_va = P->indexed ? (uintptr_t)P->ridx : 0,
                       .xq_va = (uintptr_t)P->xq, .xs_va = (uintptr_t)xs, .ymap_va = P->ymapped ? (uintptr_t)P->ymap : 0, .y_va = (uintptr_t)y };
    OmegaQgSpec sp; OmegaPfqLaunch L; int rc = -1;
    static BlackwellIRProgram prog;
    if (!nt) snprintf(msg, msgn, "tiler refused");
    else if (omega_pfq_gemm_plan(&c, &sp, &L) != OMEGA_GPU_PF_OK) snprintf(msg, msgn, "plan refused");
    else if (omega_blackwell_codegen_qgemm_prog(&sp, &prog) != 0) snprintf(msg, msgn, "codegen refused");
    else {
        uint8_t *smem = malloc(L.shared);
        static WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = L.threads; s.smem = smem; s.smem_bytes = L.shared; s.order = order;
        for (uint32_t i = 0; i < L.nargs; i++) s.cbank[0x380 / 4 + i] = L.args[i];
        const size_t XA = (size_t)P->XR + PAD;
        s.ranges[0].lo = (uintptr_t)t; s.ranges[0].hi = (uintptr_t)(t + nt);
        s.ranges[1].lo = (uintptr_t)P->ridx; s.ranges[1].hi = (uintptr_t)(P->ridx + P->M + PAD);
        s.ranges[2].lo = (uintptr_t)P->xq; s.ranges[2].hi = (uintptr_t)(P->xq + XA * P->k);
        s.ranges[3].lo = (uintptr_t)y; s.ranges[3].hi = (uintptr_t)(y + (size_t)P->YR * P->n);
        s.ranges[4].lo = (uintptr_t)P->w; s.ranges[4].hi = (uintptr_t)(P->w + (size_t)P->E * P->n * P->k);
        s.ranges[5].lo = (uintptr_t)P->swf; s.ranges[5].hi = (uintptr_t)(P->swf + (size_t)P->E * (P->n / 128) * P->KB);
        s.ranges[6].lo = (uintptr_t)xs; s.ranges[6].hi = (uintptr_t)(xs + XA * P->KB);
        s.ranges[7].lo = (uintptr_t)P->ymap; s.ranges[7].hi = (uintptr_t)(P->ymap + P->M + PAD);
        s.nranges = 8;
        rc = ws_run(&s, L.grid_x, L.grid_y) ? -1 : 0;
        if (rc) snprintf(msg, msgn, "simulator %.150s", s.msg);
        free(smem);
    }
    free(t);
    return rc;
}
/* the q36 prefill's expert GEMMs as omega_gpu_q36_graph.c pf_group lays them out: T tokens x 8 routed picks (distinct experts
 * of E-1) sorted by expert, then the shared expert (segment E-1) at rows T*8 .. with x' row t and y row t*9+8. gate_up reads
 * x' row = token through rowidx; down reads act row = its own y row (rowidx == ymap). */
static void q36_layout(Prob *P, uint32_t T, uint32_t E, uint32_t n, uint32_t k, int down) {
    uint32_t *pick = malloc((size_t)T * 8 * 4), *cnt = calloc(E, 4);
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < 8; s++) {
            uint32_t e; int dup;
            do { e = rnd() % (E - 1); dup = 0; for (uint32_t j = 0; j < s; j++) dup |= pick[t * 8 + j] == e; } while (dup);
            pick[t * 8 + s] = e; cnt[e]++;
        }
    cnt[E - 1] = T;
    prob_make(P, E, n, k, cnt, down ? T * 9 : T, 1, 1, T * 9);
    uint32_t *start = calloc(E, 4); for (uint32_t e = 1; e < E; e++) start[e] = start[e - 1] + cnt[e - 1];
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t s = 0; s < 8; s++) { const uint32_t p = start[pick[t * 8 + s]]++; P->ymap[p] = t * 9 + s; P->ridx[p] = t; }
    for (uint32_t t = 0; t < T; t++) { P->ymap[T * 8 + t] = t * 9 + 8; P->ridx[T * 8 + t] = t; }
    if (down) for (uint32_t p = 0; p < P->M; p++) P->ridx[p] = P->ymap[p];
    free(pick); free(cnt); free(start);
}
static int sim(void) {
    int fail = 0; char msg[200];
    static const struct { uint32_t k, stride, rows; int x64; } qc[] = { { 2048, 2048 + 68, 9, 1 }, { 2048, 0, 9, 0 }, { 512, 512, 11, 1 }, { 5120, 5120 + 128, 5, 1 },
                                                                        { 4096, 0, 9, 1 } /* the q36 o input */ };
    for (size_t i = 0; i < sizeof qc / sizeof qc[0]; i++) {
        size_t bad = 0; const int rc = sim_quant(qc[i].k, qc[i].stride, qc[i].rows, qc[i].x64, &bad, msg, sizeof msg);
        printf("sim quant k %u stride %u rows %u %s: %s, %zu differ\n", qc[i].k, qc[i].stride, qc[i].rows, qc[i].x64 ? "2^64 x" : "x", rc ? msg : "ran", bad);
        if (rc || bad) fail++;
    }
    {
        omega_gpu_pfq_test_set_mutant(4);
        size_t bad = 0; const int rc = sim_quant(2048, 2048 + 68, 9, 1, &bad, msg, sizeof msg);
        omega_gpu_pfq_test_set_mutant(env_mutant);
        printf("sim quant mutant 4 (c = 0): %s\n", rc ? msg : bad ? "caught" : "NOT CAUGHT");
        if (rc || !bad) fail++;
    }
    static const uint32_t ca[] = { 130, 0, 1, 64, 65, 7 }, cb[] = { 65, 63 }, cc[] = { 20, 64, 100 };
    static const struct { const char *name; const uint32_t *cnt; uint32_t E, n, k, XR, indexed, ymapped, extra; int x64; } gc[] = {
        { "A indexed+ymap+x64", ca, 6, 256, 384, 90, 1, 1, 37, 1 },
        { "B direct+ymap", cb, 2, 256, 256, 0, 0, 1, 5, 0 },
        { "C indexed+x64", cc, 3, 256, 128, 50, 1, 0, 0, 1 },
    };
    for (size_t i = 0; i < sizeof gc / sizeof gc[0]; i++) {
        uint32_t M = 0; for (uint32_t e = 0; e < gc[i].E; e++) M += gc[i].cnt[e];
        Prob P; prob_make(&P, gc[i].E, gc[i].n, gc[i].k, gc[i].cnt, gc[i].XR, gc[i].indexed, gc[i].ymapped, M + gc[i].extra);
        uint32_t *want = expect_y(&P), *y = malloc((size_t)P.YR * P.n * 4), *y0 = malloc((size_t)P.YR * P.n * 4);
        for (int order = 0; order < 2; order++) {
            const int rc = sim_gemm(&P, gc[i].x64, order, y, msg, sizeof msg);
            const size_t bad = rc ? 0 : ydiff(&P, y, want);
            printf("sim gemm %s (M %u n %u k %u, %u y rows) order %d: %s, %zu words differ\n", gc[i].name, P.M, P.n, P.k, P.YR, order, rc ? msg : "ran", bad);
            if (rc || bad) fail++;
        }
        if (gc[i].x64) { /* the same problem with true scales and x64 0: the same bits */
            const int rc = sim_gemm(&P, 0, 0, y0, msg, sizeof msg), rc1 = sim_gemm(&P, 1, 0, y, msg, sizeof msg);
            const int same = !rc && !rc1 && !memcmp(y0, y, (size_t)P.YR * P.n * 4);
            printf("sim gemm %s: x64 on 2^64 scales == x64 off on true scales: %s\n", gc[i].name, same ? "identical" : "DIFFER");
            if (!same) fail++;
        }
        if (i == 0) for (int m = 1; m <= 3; m++) {
            omega_gpu_pfq_test_set_mutant(m);
            const int rc = sim_gemm(&P, 1, 0, y, msg, sizeof msg);
            omega_gpu_pfq_test_set_mutant(env_mutant);
            const size_t bad = rc ? 0 : ydiff(&P, y, want);
            printf("sim gemm mutant %d (%s): %s\n", m, m == 1 ? "tiler row skew" : m == 2 ? "wrong scale block" : "tile i^1", rc ? msg : bad ? "caught" : "NOT CAUGHT");
            if (rc || !bad) fail++;
        }
        free(want); free(y); free(y0); prob_free(&P);
    }
    /* the q36 shapes: gate_up (n 1024, k 2048) and down (n 2048, k 512) over 12 tokens x (8 routed of 10 + shared); the dense
     * mix|o (direct rows, x64) at k 2048 and k 4096 with n cut to 256 (n only multiplies the grid) */
    static const struct { const char *name; uint32_t T, E, n, k; int down; } q36[] = {
        { "q36 gate_up routed+shared", 12, 11, 1024, 2048, 0 }, { "q36 down routed+shared", 12, 11, 2048, 512, 1 },
        { "q36 down 72 tokens (segments > 64 rows, n cut to 256)", 72, 11, 256, 512, 1 },
    };
    for (size_t i = 0; i < sizeof q36 / sizeof q36[0]; i++) {
        Prob P; q36_layout(&P, q36[i].T, q36[i].E, q36[i].n, q36[i].k, q36[i].down);
        uint32_t *want = expect_y(&P), *y = malloc((size_t)P.YR * P.n * 4);
        const int rc = sim_gemm(&P, 1, 0, y, msg, sizeof msg);
        const size_t bad = rc ? 0 : ydiff(&P, y, want);
        printf("sim gemm %s (M %u n %u k %u, %u y rows): %s, %zu words differ\n", q36[i].name, P.M, P.n, P.k, P.YR, rc ? msg : "ran", bad);
        if (rc || bad) fail++;
        if (i == 0) { /* negative control: the shared expert's y rows dropped from the map (pointed at routed row 0's slot) */
            for (uint32_t t = 0; t < q36[i].T; t++) P.ymap[q36[i].T * 8 + t] = P.ymap[0];
            const int rc2 = sim_gemm(&P, 1, 0, y, msg, sizeof msg);
            const size_t bad2 = rc2 ? 0 : ydiff(&P, y, want);
            printf("sim gemm %s negative control (shared rows mis-mapped): %s\n", q36[i].name, rc2 ? msg : bad2 ? "caught" : "NOT CAUGHT");
            if (rc2 || !bad2) fail++;
        }
        free(want); free(y); prob_free(&P);
    }
    static const uint32_t cd[] = { 9 };
    static const uint32_t dk[] = { 2048, 4096 };
    for (size_t i = 0; i < 2; i++) {
        Prob P; prob_make(&P, 1, 256, dk[i], cd, 0, 0, 0, 0);
        uint32_t *want = expect_y(&P), *y = malloc((size_t)P.YR * P.n * 4);
        const int rc = sim_gemm(&P, 1, 0, y, msg, sizeof msg);
        const size_t bad = rc ? 0 : ydiff(&P, y, want);
        printf("sim gemm q36 dense (M %u n %u k %u): %s, %zu words differ\n", P.M, P.n, P.k, rc ? msg : "ran", bad);
        if (rc || bad) fail++;
        free(want); free(y); prob_free(&P);
    }
    return fail;
}

/* ---------------- chip: one Qwen3.6 prefill chunk's routed expert GEMMs ---------------- */
typedef struct { OmegaGpuMem xf, xq, xs, y, tab, ridx, ymap; OmegaGpuFp8Tensor *wt; uint32_t nt, stride; } Dev;
static void dev_free(Dev *d) {
    omega_gpu_fp8_tensor_free(d->wt); omega_gpu_pf_free(&d->xf); omega_gpu_pf_free(&d->xq); omega_gpu_pf_free(&d->xs);
    omega_gpu_pf_free(&d->y); omega_gpu_pf_free(&d->tab); omega_gpu_pf_free(&d->ridx); omega_gpu_pf_free(&d->ymap);
}
static size_t r16(size_t b) { return (b + 15) & ~(size_t)15; }
static int dev_setup(const Prob *P, uint32_t stride, Dev *d) {
    memset(d, 0, sizeof *d); d->stride = stride;
    const size_t XA = (size_t)P->XR + PAD;
    int rc = omega_gpu_fp8_tensor_upload(P->E, P->n, P->k, P->w, P->sbf, &d->wt) == 0 ? 0 : OMEGA_GPU_PF_CHIP_FAIL;
    float *xf = malloc((size_t)P->XR * stride * 4);
    for (size_t r = 0; r < P->XR; r++) for (uint32_t j = 0; j < stride; j++) xf[r * stride + j] = j < P->k ? ldexpf(P->x[r * P->k + j], 64) : 1e30f;
    if (!rc) rc = omega_gpu_pf_alloc_dev(r16((size_t)P->XR * stride * 4), &d->xf);
    if (!rc) rc = omega_gpu_pf_alloc_dev(r16(XA * P->k), &d->xq);
    if (!rc) rc = omega_gpu_pf_alloc_dev(r16(XA * P->KB * 4), &d->xs);
    if (!rc) rc = omega_gpu_pf_alloc_dev(r16((size_t)P->YR * P->n * 4), &d->y);
    if (!rc) rc = omega_gpu_pf_upload(&d->xf, 0, xf, (size_t)P->XR * stride * 4 & ~(size_t)15);
    free(xf);
    const uint32_t cap = omega_pfq_tile_cap(P->M, P->E);
    if (!rc) rc = omega_gpu_pf_alloc_host((size_t)cap * sizeof(OmegaPfqTile), &d->tab);
    if (!rc) rc = omega_gpu_pf_alloc_host((size_t)(P->M + PAD) * 4, &d->ridx);
    if (!rc) rc = omega_gpu_pf_alloc_host((size_t)(P->M + PAD) * 4, &d->ymap);
    if (rc) return rc;
    OmegaPfqTile *seg = malloc(P->E * sizeof *seg); memcpy(seg, P->seg, P->E * sizeof *seg);
    for (uint32_t e = 0; e < P->E; e++) if (omega_gpu_fp8_tensor_entry(d->wt, e, &seg[e].w_va, &seg[e].s_va) != 0) rc = OMEGA_GPU_PF_CHIP_FAIL;
    d->nt = rc ? 0 : omega_pfq_tiles(seg, P->E, d->tab.cpu, cap);
    free(seg);
    memcpy(d->ridx.cpu, P->ridx, (size_t)(P->M + PAD) * 4); memcpy(d->ymap.cpu, P->ymap, (size_t)(P->M + PAD) * 4);
    omega_gpu_session_lock();
    if (omega_gpu_session_sync_to_gpu(&d->tab) || omega_gpu_session_sync_to_gpu(&d->ridx) || omega_gpu_session_sync_to_gpu(&d->ymap)) rc = OMEGA_GPU_PF_CHIP_FAIL;
    omega_gpu_session_unlock();
    return rc ? rc : d->nt ? 0 : OMEGA_GPU_PF_BAD_ARGS;
}
static int dev_record(OmegaGpuCmdList *cl, const Prob *P, const Dev *d) {
    const OmegaPfqQuant u = { .k = P->k, .rows = P->XR, .stride = d->stride, .x_va = d->xf.va, .q_va = d->xq.va, .s_va = d->xs.va };
    const OmegaPfqGemm c = { .n = P->n, .k = P->k, .x64 = 1, .ntiles = d->nt, .tiles_va = d->tab.va, .rowidx_va = d->ridx.va, .xq_va = d->xq.va,
                             .xs_va = d->xs.va, .ymap_va = d->ymap.va, .y_va = d->y.va };
    int rc = omega_gpu_pfq_quant_record_locked(cl, &u);
    if (!rc) rc = omega_gpu_pfq_gemm_record_locked(cl, &c);
    return rc;
}
/* runs both GEMMs once in one list; ns[0..3] = quant, gemm, quant, gemm GPU time */
static int chip_run(OmegaGpuCmdList *cl, const Prob *P1, const Dev *d1, const Prob *P2, const Dev *d2, uint64_t ns[4]) {
    omega_gpu_session_lock();
    omega_gpu_cmdlist_reset(cl);
    int rc = dev_record(cl, P1, d1);
    if (!rc) rc = dev_record(cl, P2, d2);
    OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.stamps = 1;
    OmegaGpuCmdListStats st;
    if (!rc && omega_gpu_cmdlist_submit(cl, &o, &st) != 0) rc = OMEGA_GPU_PF_CHIP_FAIL;
    uint64_t ts[5] = { 0 };
    if (!rc && omega_gpu_cmdlist_stamps(cl, ts, 5) != 5) rc = OMEGA_GPU_PF_CHIP_FAIL;
    omega_gpu_session_unlock();
    for (int i = 0; i < 4; i++) ns[i] = ts[i + 1] - ts[i];
    return rc;
}
/* quantiser and gemm outputs against the reference: xq/xs whole, y on sampled routed rows plus every untouched row */
static size_t chip_check(const Prob *P, const Dev *d, int rows, size_t *qbad) {
    const size_t XR = P->XR, yb = (size_t)P->YR * P->n * 4;
    uint8_t *xq = malloc(r16(XR * P->k)); float *xs = malloc(r16(XR * P->KB * 4)); uint32_t *y = malloc(r16(yb));
    *qbad = 0; size_t bad = 0;
    if (omega_gpu_pf_download(&d->xq, 0, xq, r16(XR * P->k)) || omega_gpu_pf_download(&d->xs, 0, xs, r16(XR * P->KB * 4)) || omega_gpu_pf_download(&d->y, 0, y, r16(yb))) { *qbad = 1; bad = 1; }
    else {
        for (size_t i = 0; i < XR * P->k; i++) if (xq[i] != P->xq[i]) (*qbad)++;
        for (size_t i = 0; i < XR * P->KB; i++) if (f2u(xs[i]) != f2u(P->xs64[i])) (*qbad)++;
        uint8_t *hit = calloc(P->YR, 1); for (uint32_t p = 0; p < P->M; p++) hit[P->ymap[p]] = 1;
        for (uint32_t r = 0; r < P->YR; r++) if (!hit[r]) for (uint32_t c = 0; c < P->n; c++) if (y[(size_t)r * P->n + c] != 0) bad++;
        free(hit);
        for (int q = 0; q < rows; q++) {
            const uint32_t p = q == 0 ? 0 : q == 1 ? P->M - 1 : rnd() % P->M;
            for (uint32_t c = 0; c < P->n; c++) if (y[(size_t)P->ymap[p] * P->n + c] != f2u(qref(P, p, c))) bad++;
        }
    }
    free(xq); free(xs); free(y);
    return bad;
}
static int chip(uint32_t tokens, int reps, int rows, uint32_t experts) {
    const uint32_t E = experts, T = tokens, NG = 9, R = T * NG; /* 8 routed + the shared slot per token, as the q36 graph lays out actN */
    uint32_t *cnt = calloc(E, 4), *tok_e = malloc((size_t)T * 8 * 4);
    for (uint32_t t = 0; t < T; t++) for (uint32_t s = 0; s < 8; s++) {
        uint32_t e; int dup;
        do { e = rnd() % E; dup = 0; for (uint32_t j = 0; j < s; j++) if (tok_e[t * 8 + j] == e) dup = 1; } while (dup);
        tok_e[t * 8 + s] = e; cnt[e]++;
    }
    uint32_t touched = 0; for (uint32_t e = 0; e < E; e++) touched += cnt[e] != 0;
    Prob P1, P2;
    prob_make(&P1, E, 1024, 2048, cnt, T, 1, 1, R);  /* gate|up: x' rows (2^64), routed by token, out to gu row t * 9 + s */
    prob_make(&P2, E, 2048, 512, cnt, R, 1, 1, R);   /* down: act rows t * 9 + s in and out */
    { /* the q36 graph's maps (pf_group): rows grouped by expert, xmap = token, ymap = token * 9 + slot */
        uint32_t *fill = calloc(E, 4), *base = calloc(E, 4);
        for (uint32_t e = 1; e < E; e++) base[e] = base[e - 1] + cnt[e - 1];
        for (uint32_t t = 0; t < T; t++) for (uint32_t s = 0; s < 8; s++) {
            const uint32_t e = tok_e[t * 8 + s], p = base[e] + fill[e]++;
            P1.ridx[p] = t; P1.ymap[p] = t * NG + s; P2.ridx[p] = t * NG + s; P2.ymap[p] = t * NG + s;
        }
        free(fill); free(base);
    }
    Dev d1, d2; int fail = 0;
    int rc = dev_setup(&P1, 2048 + 64, &d1);
    if (!rc) rc = dev_setup(&P2, 0, &d2);
    OmegaGpuCmdList *cl = NULL;
    if (!rc && omega_gpu_cmdlist_create(8, &cl) != 0) rc = OMEGA_GPU_PF_CHIP_FAIL;
    if (!rc) {
        omega_gpu_pfq_quant_prepare(2048, 2048 + 64); omega_gpu_pfq_quant_prepare(512, 0);
        omega_gpu_pfq_gemm_prepare(1024, 2048, 1, 1, 1); omega_gpu_pfq_gemm_prepare(2048, 512, 1, 1, 1);
    }
    uint64_t ns[4][32]; if (reps > 32) reps = 32; if (reps < 1) reps = 1;
    for (int r = 0; r < reps && !rc; r++) { uint64_t t4[4]; rc = chip_run(cl, &P1, &d1, &P2, &d2, t4); for (int i = 0; i < 4; i++) ns[i][r] = t4[i]; }
    if (rc) { printf("chip setup/run: %s (%s)\n", omega_gpu_pf_rc_name(rc), omega_gpu_session_last_error()); fail++; }
    else {
        size_t q1, q2; const size_t b1 = chip_check(&P1, &d1, rows, &q1), b2 = chip_check(&P2, &d2, rows, &q2);
        printf("chip gate_up (T %u, M %u, %u experts touched): quantiser %zu differ, y %zu words differ on %d sampled rows + untouched rows\n", T, P1.M, touched, q1, b1, rows);
        printf("chip down    (R %u rows):                     quantiser %zu differ, y %zu words differ\n", R, q2, b2);
        if (q1 || q2 || b1 || b2) fail++;
        static const char *nm[4] = { "quant x' k 2048", "gemm gate_up n 1024 k 2048", "quant act k 512", "gemm down n 2048 k 512" };
        const double fl[4] = { 0, 2.0 * P1.M * 1024 * 2048, 0, 2.0 * P2.M * 2048 * 512 }, wb[4] = { 0, (double)touched * 1024 * 2048, 0, (double)touched * 2048 * 512 };
        for (int i = 0; i < 4; i++) {
            qsort(ns[i], reps, 8, cmp_u64);
            const double med = (double)ns[i][reps / 2];
            printf("PFQ %-28s median %.3f us (min %.3f)", nm[i], med / 1e3, ns[i][0] / 1e3);
            if (fl[i] > 0) printf(", %.2f TFLOPS, %.1f GB/s of distinct weights", fl[i] / med / 1e3, wb[i] / med);
            printf("\n");
        }
        for (int m = 1; m <= 4; m++) {
            omega_gpu_pfq_test_set_mutant(m);
            uint64_t t4[4]; int mrc = 0;
            if (m == 1) { /* the tiler mutant changes the table: rebuild it */
                OmegaPfqTile *seg = malloc(E * sizeof *seg); memcpy(seg, P1.seg, E * sizeof *seg);
                for (uint32_t e = 0; e < E; e++) (void)omega_gpu_fp8_tensor_entry(d1.wt, e, &seg[e].w_va, &seg[e].s_va);
                d1.nt = omega_pfq_tiles(seg, E, d1.tab.cpu, omega_pfq_tile_cap(P1.M, E)); free(seg);
                omega_gpu_session_lock(); (void)omega_gpu_session_sync_to_gpu(&d1.tab); omega_gpu_session_unlock();
            }
            (void)omega_gpu_pf_upload(&d1.y, 0, calloc(1, r16((size_t)P1.YR * P1.n * 4)), r16((size_t)P1.YR * P1.n * 4));
            mrc = chip_run(cl, &P1, &d1, &P2, &d2, t4);
            omega_gpu_pfq_test_set_mutant(env_mutant);
            size_t mq1 = 0, mq2 = 0; const size_t mb = mrc ? 0 : chip_check(&P1, &d1, rows > 8 ? 8 : rows, &mq1) + chip_check(&P2, &d2, rows > 8 ? 8 : rows, &mq2);
            printf("chip mutant %d: %s\n", m, mrc ? "run failed" : (mb || mq1 || mq2) ? "caught" : "NOT CAUGHT");
            if (mrc || !(mb || mq1 || mq2)) fail++;
            if (m == 1) {
                OmegaPfqTile *seg = malloc(E * sizeof *seg); memcpy(seg, P1.seg, E * sizeof *seg);
                for (uint32_t e = 0; e < E; e++) (void)omega_gpu_fp8_tensor_entry(d1.wt, e, &seg[e].w_va, &seg[e].s_va);
                d1.nt = omega_pfq_tiles(seg, E, d1.tab.cpu, omega_pfq_tile_cap(P1.M, E)); free(seg);
                omega_gpu_session_lock(); (void)omega_gpu_session_sync_to_gpu(&d1.tab); omega_gpu_session_unlock();
            }
        }
    }
    if (cl) omega_gpu_cmdlist_free(cl);
    dev_free(&d1); dev_free(&d2); prob_free(&P1); prob_free(&P2); free(cnt); free(tok_e);
    return fail;
}

int main(int argc, char **argv) {
    int host = 1, simm = 1, chipm = 0, reps = 9, rows = 32; uint32_t tokens = 128, experts = 256;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) simm = 0;
        else if (!strcmp(argv[i], "--sim")) host = 0;
        else if (!strcmp(argv[i], "--chip")) { chipm = 1; host = 0; simm = 0; }
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) experts = (uint32_t)atoi(argv[++i]);
        else { fprintf(stderr, "GPU_PFQ: INVALID argument %s\n", argv[i]); return 2; }
    }
    if (tokens == 0 || tokens > 4096 || experts < 8 || experts > 256 || rows < 2) { fprintf(stderr, "GPU_PFQ: INVALID --tokens/--experts/--rows\n"); return 2; }
    env_mutant = omega_gpu_pfq_mutant();
    printf("AIEN_OMEGA_PFQ_MUTANT=%d%s\n", env_mutant, env_mutant ? " (every planned launch is the mutant: clean cases must FAIL)" : "");
    int fail = 0, cases = 0;
    if (host) { fail += host_tiler(); fail += host_words(); fail += host_plans(); fail += host_ref(); cases++; }
    if (simm) { fail += sim(); cases++; }
    if (chipm) { fail += chip(tokens, reps, rows, experts); cases++; }
    if (!cases) { fprintf(stderr, "GPU_PFQ: NOT_RUN (no mode selected)\n"); return 2; }
    printf("GPU_PFQ: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
