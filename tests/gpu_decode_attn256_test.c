/*
 * gpu_decode_attn256_test: the one-launch decode attention for Qwen3.8-27B (omega-mv#544, omega_gpu_decode_attn256_api.h).
 *   --host-only   host pieces: RoPE rows (partial: pairs >= rotary_dim/2 are identity), the permutation (bijection, and
 *                 "full rotate-half on the permuted head with the partial table == the model's rotate-half within the
 *                 first rotary_dim dims", double), reference sanity (context 1 returns v_new), codegen budget per depth
 *                 and mutant, refusals. No device.
 *   --sim         the kernel through tests/bw_warp_sim_k.h (no chip): contexts 1, 7, 256, 257, 301, 701 (1..3 splits,
 *                 the cross-split combine, 2 chunks per CTA), rotary_dim 64 and 256, both fragment orders, every mutant
 *                 caught at context 301.
 *   (default)     the chip: parity against the double reference at contexts 1, 7, 256, 301 (sharp q), 1024, 4096, 8192,
 *                 32768 (and 131072 with --max-pos 131072), the appended K/V row read back and compared, second call
 *                 identical, out_scale 2^64 path, timing medians with the host caches evicted, every mutant caught at
 *                 context 301, JSON receipt with --out.
 * The reference is the model's maths in the model's dim order (crates/aien-inference-abi/src/qwen38.rs `attention`:
 * zero-centred RMSNorm x * rsqrt(mean + eps) * (1 + w), RoPE on the first rotary_dim dims with pairs (j, j + rotary_dim/2),
 * scale 1/sqrt(256), causal softmax over the context). The test permutes q, k and the norm weights with
 * omega_dattn256_perm_vec before handing them to the kernel, stores history K rows permuted, and un-permutes nothing on
 * the way back (V and the output are in model order). The gate (sigmoid) is outside the kernel and outside this test.
 * Tolerance: |out - ref| <= 3e-3 * max|v| + 1e-6 per element. Host and simulator passes are never chip results.
 */
#include "omega_gpu_decode_attn256_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_hostmath.h"
#include "bw_warp_sim_k.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static const double THETA = 10000000.0; /* Qwen3.8-27B rope_theta */
static const float EPS = 1e-6f;
#define HD 256u
/* Model shape under test: 0 = the 27B (24 q / 4 KV heads, 6 q per CTA row); -DDATTN_TEST_Q36 builds Qwen3.6-35B-A3B (16 / 2 / 4,
 * two CTA rows per KV head; sc L306). */
#ifdef DATTN_TEST_Q36
#define QH 16u
#define KVH 2u
#define QPC 4u
#else
#define QH 24u
#define KVH 4u
#define QPC 6u
#endif
#define SPEC_SHAPE , .q_heads = QH, .kv_heads = KVH, .q_per_cta = QPC
#define VY (QH / QPC)                 /* CTA rows */
#define QW ((QH + 2u * KVH) * HD)     /* 8192 */
#define OW (QH * HD)                  /* 6144 */
#define QEND (QH * HD)                /* 6144: k block starts here */
#define KEND (QEND + KVH * HD)        /* 7168: v block starts here */

/* one layer's random state in MODEL order: qkv, history rows (bf16) for 4 KV heads, zero-centred norm weights w */
typedef struct {
    uint32_t max_pos, pos, rotary;
    float qkv[QW], qn[HD], kn[HD];
    uint16_t *kc, *vc;   /* [4][max_pos][256] model order (history 0..pos-1 filled; the rest zero) */
    float vmax;
} Case;
static void case_make(Case *c, uint32_t max_pos, uint32_t pos, uint32_t rotary, uint32_t seed, float qscale) {
    uint32_t s = seed;
    c->max_pos = max_pos; c->pos = pos; c->rotary = rotary;
    for (size_t i = 0; i < QW; i++) c->qkv[i] = frand(&s, -1.0f, 1.0f) * (i < QEND ? qscale : 1.0f);
    for (unsigned i = 0; i < HD; i++) { c->qn[i] = frand(&s, -0.2f, 0.2f); c->kn[i] = frand(&s, -0.2f, 0.2f); }
    size_t words = (size_t)KVH * max_pos * HD;
    c->kc = calloc(words, 2); c->vc = calloc(words, 2);
    c->vmax = 1.0f;
    for (unsigned g = 0; g < KVH; g++) for (uint32_t t = 0; t < pos; t++) for (unsigned i = 0; i < HD; i++) {
        size_t o = ((size_t)g * max_pos + t) * HD + i;
        c->kc[o] = omega_dattn_bf16_rne(frand(&s, -1.0f, 1.0f)); c->vc[o] = omega_dattn_bf16_rne(frand(&s, -1.0f, 1.0f));
    }
}
static void case_free(Case *c) { free(c->kc); free(c->vc); }
/* what the kernel sees: q and k heads permuted, v as is; norms 1 + w permuted; K history rows permuted */
static void case_kernel_inputs(const Case *c, float *qkv_k, float *qn_k, float *kn_k) {
    for (unsigned h = 0; h < QH + KVH; h++) omega_dattn256_perm_vec(c->qkv + h * HD, qkv_k + h * HD, c->rotary);
    memcpy(qkv_k + KEND, c->qkv + KEND, (size_t)KVH * HD * 4);
    float t[HD];
    for (unsigned i = 0; i < HD; i++) t[i] = 1.0f + c->qn[i];
    omega_dattn256_perm_vec(t, qn_k, c->rotary);
    for (unsigned i = 0; i < HD; i++) t[i] = 1.0f + c->kn[i];
    omega_dattn256_perm_vec(t, kn_k, c->rotary);
}
static uint16_t *case_kc_perm(const Case *c) {
    size_t rows = (size_t)KVH * c->max_pos;
    uint16_t *p = calloc(rows * HD, 2);
    for (size_t r = 0; r < rows; r++) omega_dattn256_perm_vec_bf16(c->kc + r * HD, p + r * HD, c->rotary);
    return p;
}

/* ---- double-precision oracle (test-owned: libm) in MODEL order ---- */
static void norm_rope(double *x, const float *w, float eps, uint32_t rotary, uint32_t pos, double theta, int rope) {
    double ss = 0; for (unsigned i = 0; i < HD; i++) ss += x[i] * x[i];
    double r = 1.0 / sqrt(ss / (double)HD + (double)eps);
    for (unsigned i = 0; i < HD; i++) x[i] = x[i] * r * (1.0 + (double)w[i]);
    if (!rope) return;
    const unsigned half = rotary / 2;
    for (unsigned j = 0; j < half; j++) {
        double inv = 1.0 / pow(theta, (double)(2 * j) / (double)rotary), ang = (double)pos * inv, cs = cos(ang), sn = sin(ang);
        double x1 = x[j], x2 = x[j + half];
        x[j] = x1 * cs - x2 * sn; x[j + half] = x2 * cs + x1 * sn;
    }
}
/* Reference of the whole layer step. kc/vc: this layer's bf16 planes [4][max_pos][256] (model order) holding positions
 * 0..pos-1; out: [24*256] doubles (already * out_scale); k_new/v_new: the bf16 rows (model order) the kernel must have
 * appended at pos ([4][256] each). The reference rounds its k_new/v_new to bf16 and attends over the rounded values. */
static void reference(const float *qkv, const uint16_t *kc, const uint16_t *vc, uint32_t max_pos, uint32_t pos, uint32_t rotary,
                      const float *q_w, const float *k_w, float eps, double theta, float out_scale, double *out, uint16_t *k_new, uint16_t *v_new) {
    const double qscale = 1.0 / sqrt((double)HD);
    for (unsigned gk = 0; gk < KVH; gk++) {
        double k[HD];
        for (unsigned i = 0; i < HD; i++) k[i] = qkv[QEND + gk * HD + i];
        norm_rope(k, k_w, eps, rotary, pos, theta, 1);
        for (unsigned i = 0; i < HD; i++) { k_new[gk * HD + i] = omega_dattn_bf16_rne((float)k[i]); v_new[gk * HD + i] = omega_dattn_bf16_rne(qkv[KEND + gk * HD + i]); }
    }
    double *p = calloc((size_t)pos + 1, sizeof(double));
    for (unsigned h = 0; h < QH; h++) {
        unsigned gk = h / (QH / KVH);
        double q[HD];
        for (unsigned i = 0; i < HD; i++) q[i] = qkv[h * HD + i];
        norm_rope(q, q_w, eps, rotary, pos, theta, 1);
        double mx = -1e300;
        for (uint32_t t = 0; t <= pos; t++) {
            const uint16_t *kr = t == pos ? k_new + gk * HD : kc + ((size_t)gk * max_pos + t) * HD;
            double s = 0; for (unsigned i = 0; i < HD; i++) s += q[i] * (double)omega_dattn_bf16_f32(kr[i]);
            s *= qscale; p[t] = s; if (s > mx) mx = s;
        }
        double l = 0; for (uint32_t t = 0; t <= pos; t++) { p[t] = exp(p[t] - mx); l += p[t]; }
        double o[HD]; memset(o, 0, sizeof o);
        for (uint32_t t = 0; t <= pos; t++) {
            const uint16_t *vr = t == pos ? v_new + gk * HD : vc + ((size_t)gk * max_pos + t) * HD;
            for (unsigned i = 0; i < HD; i++) o[i] += p[t] * (double)omega_dattn_bf16_f32(vr[i]);
        }
        for (unsigned i = 0; i < HD; i++) out[h * HD + i] = o[i] / l * (double)out_scale;
    }
    free(p);
}

/* compares out (f32 [6144]) against the reference; returns the number of elements outside the tolerance and the worst ratio */
static size_t parity(const Case *c, const float *out, float out_scale, double *worst, uint16_t *k_new, uint16_t *v_new) {
    double *ref = malloc(OW * sizeof(double));
    reference(c->qkv, c->kc, c->vc, c->max_pos, c->pos, c->rotary, c->qn, c->kn, EPS, THETA, out_scale, ref, k_new, v_new);
    double tol = 3e-3 * c->vmax * out_scale + 1e-6 * out_scale; size_t bad = 0; *worst = 0;
    for (size_t i = 0; i < OW; i++) { double d = fabs((double)out[i] - ref[i]); if (d / tol > *worst) *worst = d / tol; if (d > tol) bad++; }
    free(ref);
    return bad;
}
/* appended rows (chip rows: K permuted, V model order) against the reference rows (model order): v exact, k within one
 * bf16 ulp on at most 32 elements over the 4 heads */
static int rows_ok(const uint16_t *k_chip, const uint16_t *v_chip, const uint16_t *k_ref, const uint16_t *v_ref, uint32_t rotary, char *why, size_t n) {
    unsigned kflips = 0;
    for (unsigned g = 0; g < KVH; g++) for (unsigned i = 0; i < HD; i++) {
        unsigned ci = g * HD + i, ri = g * HD + omega_dattn256_perm_slot(i, rotary);
        if (v_chip[ci] != v_ref[ci]) { snprintf(why, n, "v[%u] chip %04x ref %04x", ci, v_chip[ci], v_ref[ci]); return 0; }
        if (k_chip[ci] != k_ref[ri]) { int d = (int)k_chip[ci] - (int)k_ref[ri]; if (d < -1 || d > 1) { snprintf(why, n, "k slot %u (model dim %u) chip %04x ref %04x", ci, ri, k_chip[ci], k_ref[ri]); return 0; } kflips++; }
    }
    if (kflips > 32) { snprintf(why, n, "%u one-ulp k flips", kflips); return 0; }
    snprintf(why, n, "ok (%u one-ulp k flips)", kflips);
    return 1;
}

/* ------------------------------------------------------------- host only */
static void perm_checks(void) {
    static const uint32_t rots[] = { 64, 256, 2, 128, 32 };
    for (unsigned ri = 0; ri < sizeof rots / sizeof rots[0]; ri++) {
        uint32_t R = rots[ri], P = R / 2; uint8_t seen[HD]; memset(seen, 0, sizeof seen); int bij = 1;
        for (uint32_t i = 0; i < HD; i++) { uint32_t m = omega_dattn256_perm_slot(i, R); if (m >= HD || seen[m]) bij = 0; else seen[m] = 1; }
        CHECK(bij, "perm rotary %u is a bijection", R);
        int pairs = 1; for (uint32_t i = 0; i < P; i++) pairs &= omega_dattn256_perm_slot(i, R) == i && omega_dattn256_perm_slot(i + 128, R) == i + P;
        CHECK(pairs, "perm rotary %u: slots (i, i+128), i < %u hold the model pairs (i, i+%u)", R, P, P);
        if (R == 256) { int id = 1; for (uint32_t i = 0; i < HD; i++) id &= omega_dattn256_perm_slot(i, R) == i; CHECK(id, "perm rotary 256 is the identity"); }
        /* the kernel's rotate-half on the permuted head with the partial table == the model's rope, in double */
        uint32_t s = 99 + R; double maxd = 0;
        for (unsigned trial = 0; trial < 50; trial++) {
            uint32_t pos = trial < 3 ? trial : lcg(&s) % 131072u;
            double x[HD], y_model[HD], xp[HD], yp[HD], y_back[HD]; float row[HD];
            for (unsigned i = 0; i < HD; i++) x[i] = frand(&s, -2.0f, 2.0f);
            memcpy(y_model, x, sizeof x);
            { float w0[HD]; for (unsigned i = 0; i < HD; i++) w0[i] = 0.0f; /* rope only: norm with w = 0 then undo the scale */
              double ss = 0; for (unsigned i = 0; i < HD; i++) ss += x[i] * x[i]; double r = 1.0 / sqrt(ss / HD + (double)EPS);
              norm_rope(y_model, w0, EPS, R, pos, THETA, 1); for (unsigned i = 0; i < HD; i++) y_model[i] /= r; }
            omega_dattn256_rope_row(pos, THETA, R, row);
            for (unsigned i = 0; i < HD; i++) xp[i] = x[omega_dattn256_perm_slot(i, R)];
            for (unsigned i = 0; i < 128; i++) { double cs = row[i], sn = row[128 + i]; yp[i] = xp[i] * cs - xp[i + 128] * sn; yp[i + 128] = xp[i + 128] * cs + xp[i] * sn; }
            for (unsigned i = 0; i < HD; i++) y_back[omega_dattn256_perm_slot(i, R)] = yp[i];
            for (unsigned i = 0; i < HD; i++) { double d = fabs(y_back[i] - y_model[i]); if (d > maxd) maxd = d; }
        }
        CHECK(maxd < 1e-5, "perm rotary %u: rotate-half on the permuted head == model rope (max abs diff %.3g; f32 table)", R, maxd);
        printf("perm rotary %3u: bijection, pairs, rope-by-permutation max abs diff %.3g over 50 positions\n", R, maxd);
    }
}
static void host_only(void) {
    float row[HD]; omega_dattn256_rope_row(0, THETA, 64, row);
    int ok = 1; for (unsigned i = 0; i < 128; i++) ok &= row[i] == 1.0f && row[128 + i] == 0.0f;
    CHECK(ok, "rope row 0 is cos 1 sin 0");
    omega_dattn256_rope_row(100, THETA, 64, row);
    CHECK(fabs((double)row[0] - cos(100.0)) < 1e-6 && fabs((double)row[128] - sin(100.0)) < 1e-6, "rope row 100 pair 0 is cos/sin(100)");
    CHECK(fabs((double)row[31] - cos(100.0 * pow(THETA, -62.0 / 64.0))) < 1e-6, "rope row 100 pair 31 uses theta^(-62/64)");
    ok = 1; for (unsigned i = 32; i < 128; i++) ok &= row[i] == 1.0f && row[128 + i] == 0.0f;
    CHECK(ok, "rope row 100: pairs 32..127 are identity for rotary_dim 64");
    omega_dattn256_rope_row(100, THETA, 256, row);
    CHECK(fabs((double)row[127] - cos(100.0 * pow(THETA, -254.0 / 256.0))) < 1e-6, "rope row 100 rotary 256 pair 127 uses theta^(-254/256)");
    perm_checks();
    /* reference sanity: context 1 returns v_new (rounded), whatever q and k */
    { Case c; case_make(&c, 256, 0, 64, 11, 1.0f); double *ref = malloc(OW * sizeof(double)); uint16_t kn[KVH * HD], vn[KVH * HD];
      reference(c.qkv, c.kc, c.vc, 256, 0, 64, c.qn, c.kn, EPS, THETA, 1.0f, ref, kn, vn);
      int good = 1; for (unsigned h = 0; h < QH && good; h++) for (unsigned i = 0; i < HD; i++) if (ref[h * HD + i] != (double)omega_dattn_bf16_f32(vn[(h / (QH / KVH)) * HD + i])) { good = 0; break; }
      CHECK(good, "reference: context 1 returns the rounded v_new of the head's KV group");
      free(ref); case_free(&c); }
    { Case c; case_make(&c, 256, 8, 64, 12, 1.0f);
      double *ref = malloc(OW * sizeof(double)); uint16_t kn[KVH * HD], vn[KVH * HD];
      reference(c.qkv, c.kc, c.vc, 256, 8, 64, c.qn, c.kn, EPS, THETA, 1.0f, ref, kn, vn);
      double ss = 0; for (unsigned i = 0; i < HD; i++) { double v = omega_dattn_bf16_f32(kn[i]); ss += v * v; }
      CHECK(ss / HD > 0.6 && ss / HD < 1.5, "reference: k_new has rms about 1 after RMSNorm (%.3f)", ss / HD);
      double os = 0; for (unsigned i = 0; i < OW; i++) os += fabs(ref[i]);
      CHECK(os / OW < 0.6, "reference: a 9-token softmax average of U(-1,1) rows is small (%.3f)", os / OW);
      free(ref); case_free(&c); }
    /* codegen budget */
    for (uint32_t d = 1; d <= 4; d++) for (int m = 0; m <= 5; m++) {
        OmegaDecodeAttn256Spec s = { .depth = d, .mutant = m SPEC_SHAPE }; uint32_t n = 0, gp = 0;
        int rc = omega_decode_attn256_prog_stats(&s, &n, &gp);
        CHECK(rc == 0 && n <= 1024 && gp <= 126, "codegen depth %u mutant %d: rc %d insns %u gprs %u", d, m, rc, n, gp);
        if (rc == 0 && m == 0) printf("codegen depth %u: %u instructions, %u registers (declared %u, 256 threads -> %u of 65536)\n", d, n, gp, omega_bw_gpr_declared(gp), 256 * omega_bw_gpr_declared(gp));
    }
    { OmegaDecodeAttn256Spec s = { .depth = 0 SPEC_SHAPE }; CHECK(omega_decode_attn256_prog_stats(&s, NULL, NULL) != 0, "depth 0 refused");
      s.depth = 5; CHECK(omega_decode_attn256_prog_stats(&s, NULL, NULL) != 0, "depth 5 refused");
      s.depth = 1; s.mutant = 6; CHECK(omega_decode_attn256_prog_stats(&s, NULL, NULL) != 0, "mutant 6 refused");
      s.mutant = 0; s.q_heads = 24; s.kv_heads = 4; s.q_per_cta = 5; CHECK(omega_decode_attn256_prog_stats(&s, NULL, NULL) != 0, "q_per_cta not dividing q heads refused");
      s.q_heads = 24; s.kv_heads = 2; s.q_per_cta = 4; CHECK(omega_decode_attn256_prog_stats(&s, NULL, NULL) != 0, "three CTA rows per KV head refused");
      s.q_heads = 16; s.kv_heads = 2; s.q_per_cta = 8; CHECK(omega_decode_attn256_prog_stats(&s, NULL, NULL) != 0, "more than 6 q heads per CTA row refused"); }
    CHECK(omega_gpu_decode_attn256_cache_alloc_shape(0, 256, THETA, 64, QH, KVH, QPC, NULL) == OMEGA_GPU_DATTN_BAD_ARGS, "NULL out refused");
    { OmegaGpuDecodeAttn256Cache *c = (OmegaGpuDecodeAttn256Cache *)1;
      CHECK(omega_gpu_decode_attn256_cache_alloc_shape(1, 100, THETA, 64, QH, KVH, QPC, &c) == OMEGA_GPU_DATTN_BAD_ARGS && c == NULL, "max_pos not a multiple of 256 refused before any device work");
      c = (OmegaGpuDecodeAttn256Cache *)1;
      CHECK(omega_gpu_decode_attn256_cache_alloc_shape(1, 256, THETA, 63, QH, KVH, QPC, &c) == OMEGA_GPU_DATTN_BAD_ARGS && c == NULL, "odd rotary_dim refused");
      c = (OmegaGpuDecodeAttn256Cache *)1;
      CHECK(omega_gpu_decode_attn256_cache_alloc_shape(1, 256, THETA, 258, QH, KVH, QPC, &c) == OMEGA_GPU_DATTN_BAD_ARGS && c == NULL, "rotary_dim > 256 refused");
      c = (OmegaGpuDecodeAttn256Cache *)1;
      CHECK(omega_gpu_decode_attn256_cache_alloc_shape(1, 131072 + 256, THETA, 64, QH, KVH, QPC, &c) == OMEGA_GPU_DATTN_BAD_ARGS && c == NULL, "max_pos > 131072 refused"); }
    CHECK(OMEGA_DATTN256_SHARED_BYTES <= 16384u, "shared memory within the QMD cap (%u)", OMEGA_DATTN256_SHARED_BYTES);
    CHECK(OMEGA_DATTN256_SM_FLAG + 16u == OMEGA_DATTN256_SHARED_BYTES && OMEGA_DATTN256_SM_S + 8192u <= OMEGA_DATTN256_SM_M && OMEGA_DATTN256_SM_M + 288u <= OMEGA_DATTN256_SM_L && OMEGA_DATTN256_SM_L + 256u <= OMEGA_DATTN256_SM_FLAG, "shared map does not overlap");
}

/* ------------------------------------------------------------- simulator */
static uint64_t g_sim_issues;
static int sim_point(uint32_t max_pos, uint32_t pos, uint32_t rotary, uint32_t depth, int mutant, uint32_t per_split, float qscale, uint32_t seed, char *msg, size_t n, double *worst, size_t *bad, char *rows, size_t rn) {
    static BlackwellIRProgram prog;
    OmegaDecodeAttn256Spec spec = { .depth = depth, .mutant = mutant SPEC_SHAPE };
    if (omega_blackwell_codegen_decode_attn256_prog(&spec, &prog) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    Case c; case_make(&c, max_pos, pos, rotary, seed, qscale);
    uint16_t *kcp = case_kc_perm(&c);
    const uint32_t splits = (pos + per_split) / per_split;
    size_t plane = (size_t)KVH * max_pos * OMEGA_DATTN256_ROW_BYTES + OMEGA_DATTN256_PLANE_SLACK, hist = (size_t)KVH * max_pos * OMEGA_DATTN256_ROW_BYTES;
    uint8_t *kv = aligned_alloc(16, 2 * plane); memset(kv, 0, 2 * plane);
    float *rope = aligned_alloc(16, (size_t)max_pos * OMEGA_DATTN256_ROPE_ROW_BYTES); for (uint32_t p = 0; p < max_pos; p++) omega_dattn256_rope_row(p, THETA, rotary, rope + (size_t)p * 256);
    float *norms = aligned_alloc(16, 2048);
    float *qkv = aligned_alloc(16, sizeof c.qkv);
    case_kernel_inputs(&c, qkv, norms, norms + HD);
    float *out = aligned_alloc(16, OW * 4), *copy = malloc(OW * 4);
    float *part = aligned_alloc(16, (size_t)VY * splits * OMEGA_DATTN256_PARTIAL_WORDS * 4);
    uint32_t *cnt = aligned_alloc(16, 16);
    int err = 0;
    for (int order = 0; order < 2 && !err; order++) {
        WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = 256; s.order = order;
        static uint8_t smem[OMEGA_DATTN256_SHARED_BYTES]; s.smem = smem; s.smem_bytes = OMEGA_DATTN256_SHARED_BYTES;
        memset(cnt, 0, 16); for (unsigned i = 0; i < OW; i++) out[i] = -999.0f;
        memcpy(kv, kcp, hist); memcpy(kv + plane, c.vc, hist); /* fresh history (the kernel appends) */
        uint64_t A[9] = { (uintptr_t)qkv, (uintptr_t)kv, (uintptr_t)(kv + plane), (uintptr_t)rope, (uintptr_t)norms, (uintptr_t)(norms + HD), (uintptr_t)out, (uintptr_t)part, (uintptr_t)cnt };
        for (unsigned i = 0; i < 9; i++) { s.cbank[(0x380 + 8 * i) / 4] = (uint32_t)A[i]; s.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
        float fe = EPS, fo = 1.0f, fq = (float)((1.0 / 16.0) * 1.4426950408889634);
        memcpy(&s.cbank[OMEGA_DATTN256_ARG_EPS / 4], &fe, 4); memcpy(&s.cbank[OMEGA_DATTN256_ARG_OUT_SCALE / 4], &fo, 4); memcpy(&s.cbank[OMEGA_DATTN256_ARG_QK_SCALE / 4], &fq, 4);
        s.cbank[OMEGA_DATTN256_ARG_POS / 4] = pos; s.cbank[OMEGA_DATTN256_ARG_PER_SPLIT / 4] = per_split; s.cbank[OMEGA_DATTN256_ARG_SPLITS / 4] = splits; s.cbank[OMEGA_DATTN256_ARG_KV_STRIDE / 4] = max_pos * OMEGA_DATTN256_ROW_BYTES;
        s.nranges = 7;
        s.ranges[0].lo = (uintptr_t)qkv; s.ranges[0].hi = (uintptr_t)qkv + sizeof c.qkv;
        s.ranges[1].lo = (uintptr_t)kv; s.ranges[1].hi = (uintptr_t)kv + 2 * plane;
        s.ranges[2].lo = (uintptr_t)rope; s.ranges[2].hi = (uintptr_t)rope + (size_t)max_pos * OMEGA_DATTN256_ROPE_ROW_BYTES;
        s.ranges[3].lo = (uintptr_t)norms; s.ranges[3].hi = (uintptr_t)norms + 2048;
        s.ranges[4].lo = (uintptr_t)out; s.ranges[4].hi = (uintptr_t)out + OW * 4;
        s.ranges[5].lo = (uintptr_t)part; s.ranges[5].hi = (uintptr_t)part + (size_t)VY * splits * OMEGA_DATTN256_PARTIAL_WORDS * 4;
        s.ranges[6].lo = (uintptr_t)cnt; s.ranges[6].hi = (uintptr_t)cnt + 16;
        err = ws_run(&s, splits, VY);
        if (err) { snprintf(msg, n, "order %d: %s (pc %d)", order, s.msg, s.err_pc); break; }
        g_sim_issues += s.warp_issues;
        size_t unwritten = 0; for (unsigned i = 0; i < OW; i++) if (out[i] == -999.0f) unwritten++;
        if (unwritten) { err = -1; snprintf(msg, n, "%zu output words never written", unwritten); break; }
        if (cnt[0] | cnt[1] | cnt[2] | cnt[3]) { err = -1; snprintf(msg, n, "counters not reset: %u %u %u %u", cnt[0], cnt[1], cnt[2], cnt[3]); break; }
        if (order == 0) memcpy(copy, out, OW * 4);
        else if (memcmp(copy, out, OW * 4) != 0) { err = -1; snprintf(msg, n, "the fragment order changed the result"); }
    }
    if (!err) {
        uint16_t kn[KVH * HD], vn[KVH * HD];
        *bad = parity(&c, out, 1.0f, worst, kn, vn);
        uint16_t kchip[KVH * HD], vchip[KVH * HD];
        for (unsigned g = 0; g < KVH; g++) { memcpy(kchip + g * HD, kv + ((size_t)g * max_pos + pos) * OMEGA_DATTN256_ROW_BYTES, OMEGA_DATTN256_ROW_BYTES); memcpy(vchip + g * HD, kv + plane + ((size_t)g * max_pos + pos) * OMEGA_DATTN256_ROW_BYTES, OMEGA_DATTN256_ROW_BYTES); }
        if (!rows_ok(kchip, vchip, kn, vn, rotary, rows, rn)) *bad += 1000;
        snprintf(msg, n, "bad=%zu worst=%.3f rows: %s", *bad, *worst, rows);
    }
    free(kv); free(rope); free(norms); free(qkv); free(out); free(copy); free(part); free(cnt); free(kcp); case_free(&c);
    return err;
}
static void sim_battery(void) {
    char msg[300], rows[160]; double worst; size_t bad;
    struct { uint32_t max_pos, pos, rotary, depth, per; float qscale; const char *name; } pts[] = {
        { 256, 0, 64, 1, 256, 1.0f, "context 1" },
        { 256, 6, 64, 2, 256, 1.0f, "context 7" },
        { 256, 6, 64, 4, 256, 4.0f, "context 7, sharp q (x4)" },
        { 256, 6, 256, 4, 256, 1.0f, "context 7, full rope (rotary 256)" },
        { 256, 255, 64, 4, 256, 1.0f, "context 256 (full chunk, 1 split)" },
        { 512, 256, 64, 4, 256, 1.0f, "context 257 (2 splits, last holds one position)" },
        { 512, 300, 64, 3, 256, 1.0f, "context 301 (2 splits)" },
        { 1024, 700, 64, 4, 256, 1.0f, "context 701 (3 splits)" },
        { 1024, 700, 64, 2, 512, 1.0f, "context 701 (per_split 512: 2 chunks per CTA)" },
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        int err = sim_point(pts[i].max_pos, pts[i].pos, pts[i].rotary, pts[i].depth, 0, pts[i].per, pts[i].qscale, 100 + (uint32_t)i, msg, sizeof msg, &worst, &bad, rows, sizeof rows);
        CHECK(err == 0 && bad == 0, "sim %s depth %u: %s", pts[i].name, pts[i].depth, msg);
        printf("sim %-50s depth %u: %s %s\n", pts[i].name, pts[i].depth, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
    }
    for (int m = 1; m <= 5; m++) {
        int err = sim_point(512, 300, 64, 2, m, 256, 1.0f, 200, msg, sizeof msg, &worst, &bad, rows, sizeof rows);
        int caught = err != 0 || bad > 0;
        CHECK(caught, "sim mutant %d NOT caught (%s)", m, msg);
        printf("sim mutant %d at context 301: %s (%s)\n", m, caught ? "caught" : "MISSED", msg);
    }
    printf("sim: %" PRIu64 " warp issues in total\n", g_sim_issues);
}

/* ------------------------------------------------------------- chip */
static int g_evict = 1; static const char *g_only; static uint32_t g_maxp = 32768;
static void evict_host_caches(void) {
    static uint8_t *buf; const size_t sz = 512u << 20; static uint8_t pass;
    if (!buf) buf = malloc(sz);
    if (!buf) return;
    memset(buf, ++pass, sz);
    volatile uint64_t sum = 0; for (size_t i = 0; i < sz; i += 64) sum += buf[i];
}
static int fill_history(OmegaGpuDecodeAttn256Cache *cache, const Case *c, const uint16_t *kcp) {
    if (c->pos == 0) return OMEGA_GPU_DATTN_OK;
    for (unsigned g = 0; g < KVH; g++) {
        int rc = omega_gpu_decode_attn256_cache_write(cache, 0, g, 0, c->pos, kcp + (size_t)g * c->max_pos * HD, c->vc + (size_t)g * c->max_pos * HD);
        if (rc != OMEGA_GPU_DATTN_OK) return rc;
    }
    return OMEGA_GPU_DATTN_OK;
}
static int chip_run(const char *out_path, uint32_t reps) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot write %s\n", out_path); return 1; }
    if (out) fprintf(out, "{\"receipt\":\"OMEGA_GPU_DECODE_ATTN256_V1\",\"reps\":%u,\"evict\":%d,\"max_pos\":%u,\"rotary_dim\":64,\"points\":[", reps, g_evict, g_maxp);
    const uint32_t MAXP = g_maxp;
    OmegaGpuDecodeAttn256Cache *cache = NULL;
    double t0 = now_ms();
    int rc = omega_gpu_decode_attn256_cache_alloc_shape(1, MAXP, THETA, 64, QH, KVH, QPC, &cache);
    printf("cache alloc (1 layer, %u positions, %.0f MiB): rc=%s %.1f ms (%s)\n", MAXP, (double)MAXP * 2 * KVH * 512 / 1048576.0, omega_gpu_decode_attn_rc_name(rc), now_ms() - t0, rc ? omega_gpu_decode_attn_last_error() : "");
    CHECK(rc == OMEGA_GPU_DATTN_OK, "cache alloc");
    if (rc != OMEGA_GPU_DATTN_OK) { if (out) fprintf(out, "],\"verdict\":\"OMEGA_GPU_DECODE_ATTN256_FAIL\"}\n"); return 1; }
    struct { uint32_t pos; float qscale; const char *name; } pts[] = {
        { 0, 1.0f, "context 1" }, { 6, 1.0f, "context 7" }, { 255, 1.0f, "context 256" }, { 300, 4.0f, "context 301 sharp q" },
        { 1023, 1.0f, "context 1024" }, { 4095, 1.0f, "context 4096" }, { 8191, 1.0f, "context 8192" }, { 32767, 1.0f, "context 32768" },
        { 131071, 1.0f, "context 131072" },
    };
    int first = 1;
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        if (pts[i].pos >= MAXP) continue;
        if (g_only && !strstr(pts[i].name, g_only)) continue;
        Case c; case_make(&c, MAXP, pts[i].pos, 64, 500 + (uint32_t)i, pts[i].qscale);
        uint16_t *kcp = case_kc_perm(&c);
        float *qkv_k = malloc(QW * 4), qn_k[HD], kn_k[HD];
        case_kernel_inputs(&c, qkv_k, qn_k, kn_k);
        CHECK(omega_gpu_decode_attn256_set_norms(cache, 0, qn_k, kn_k) == OMEGA_GPU_DATTN_OK, "set norms");
        t0 = now_ms();
        rc = fill_history(cache, &c, kcp);
        double fill_ms = now_ms() - t0;
        float *y = malloc(OW * 4), *y2 = malloc(OW * 4);
        OmegaGpuDecodeAttn256Info info; memset(&info, 0, sizeof info);
        if (rc == OMEGA_GPU_DATTN_OK) rc = omega_gpu_decode_attn256(cache, 0, c.pos, qkv_k, EPS, 1.0f, y, &info);
        size_t bad = 0; double worst = 1e9; uint16_t kn[KVH * HD], vn[KVH * HD], kchip[KVH * HD], vchip[KVH * HD]; char rows[160] = "n/a";
        if (rc == OMEGA_GPU_DATTN_OK) {
            bad = parity(&c, y, 1.0f, &worst, kn, vn);
            for (unsigned g = 0; g < KVH; g++) if (omega_gpu_decode_attn256_cache_read(cache, 0, g, c.pos, kchip + g * HD, vchip + g * HD) != OMEGA_GPU_DATTN_OK) bad += 1000;
            if (!rows_ok(kchip, vchip, kn, vn, 64, rows, sizeof rows)) bad += 1000;
        }
        int ok = rc == OMEGA_GPU_DATTN_OK && bad == 0;
        int same = 0; OmegaGpuDecodeAttn256Info i2; memset(&i2, 0, sizeof i2);
        if (ok) { int r2 = omega_gpu_decode_attn256(cache, 0, c.pos, qkv_k, EPS, 1.0f, y2, &i2); same = r2 == OMEGA_GPU_DATTN_OK && memcmp(y, y2, OW * 4) == 0 && i2.kernel_cache_hit; }
        CHECK(ok && same, "chip %s: rc=%s bad=%zu worst=%.3f rows=%s second-call-identical=%d (%s)", pts[i].name, omega_gpu_decode_attn_rc_name(rc), bad, worst, rows, same, omega_gpu_decode_attn_last_error());
        if (ok) {
            float *ys = malloc(OW * 4); int r3 = omega_gpu_decode_attn256(cache, 0, c.pos, qkv_k, EPS, 18446744073709551616.0f, ys, NULL);
            int good = r3 == OMEGA_GPU_DATTN_OK; double wr = 0;
            for (unsigned k = 0; k < OW && good; k++) { double e = fabs((double)ys[k] / 18446744073709551616.0 - (double)y[k]); double r = e / (fabs((double)y[k]) + 1e-6); if (r > wr) wr = r; }
            CHECK(good && wr < 1e-5, "chip %s: out_scale 2^64 matches the unscaled output (worst rel %.2e)", pts[i].name, wr);
            free(ys);
        }
        uint64_t *el = calloc(reps, sizeof *el), *cl = calloc(reps, sizeof *cl); uint32_t done = 0;
        if (g_evict) evict_host_caches();
        for (uint32_t r = 0; r < reps && ok; r++) { OmegaGpuDecodeAttn256Info ti; if (omega_gpu_decode_attn256(cache, 0, c.pos, qkv_k, EPS, 1.0f, y2, &ti) != OMEGA_GPU_DATTN_OK) break; el[r] = ti.elapsed_ns; cl[r] = ti.call_ns; done++; }
        uint64_t med_el = 0, med_cl = 0; double gbs = 0;
        if (done) { qsort(el, done, sizeof *el, cmp_u64); qsort(cl, done, sizeof *cl, cmp_u64); med_el = el[done / 2]; med_cl = cl[done / 2]; gbs = med_el ? (double)info.kv_bytes / (double)med_el : 0; }
        printf("chip %-24s rc=%-9s bad=%zu worst=%.3f rows=%s grid=%ux%u per_split=%u depth=%u kv_bytes=%" PRIu64 " fill=%.1fms chip_med=%.1fus (%.1f GB/s) call_med=%.1fus reps=%u\n",
               pts[i].name, omega_gpu_decode_attn_rc_name(rc), bad, worst, rows, info.grid_x, info.grid_y, info.per_split, info.depth, info.kv_bytes, fill_ms, med_el / 1e3, gbs, med_cl / 1e3, done);
        if (out) fprintf(out, "%s{\"name\":\"%s\",\"pos\":%u,\"rc\":\"%s\",\"bad\":%zu,\"worst_of_tol\":%.3f,\"rows\":\"%s\",\"second_call_identical\":%d,\"grid_x\":%u,\"grid_y\":%u,\"per_split\":%u,\"depth\":%u,"
                         "\"kv_bytes\":%" PRIu64 ",\"fill_ms\":%.1f,\"chip_median_ns\":%" PRIu64 ",\"gb_per_s\":%.1f,\"call_median_ns\":%" PRIu64 ",\"reps\":%u}",
                         first ? "" : ",", pts[i].name, c.pos, omega_gpu_decode_attn_rc_name(rc), bad, worst, rows, same, info.grid_x, info.grid_y, info.per_split, info.depth, info.kv_bytes, fill_ms, med_el, gbs, med_cl, done);
        first = 0;
        free(el); free(cl);
        if (ok && c.pos == 300) {
            for (int m = 1; m <= 5; m++) {
                omega_gpu_decode_attn256_test_set_mutant(m);
                (void)fill_history(cache, &c, kcp); /* mutant 3 must not inherit the row the honest kernel wrote */
                uint16_t zk[HD] = { 0 }, zv[HD] = { 0 };
                for (unsigned g = 0; g < KVH; g++) (void)omega_gpu_decode_attn256_cache_write(cache, 0, g, c.pos, 1, zk, zv);
                int r = omega_gpu_decode_attn256(cache, 0, c.pos, qkv_k, EPS, 1.0f, y2, NULL);
                size_t mb = 0; double mw = 0; uint16_t k2[KVH * HD], v2[KVH * HD];
                if (r == OMEGA_GPU_DATTN_OK) { mb = parity(&c, y2, 1.0f, &mw, k2, v2); }
                int caught = r != OMEGA_GPU_DATTN_OK || mb > 0;
                if (m == 4 && info.grid_x == 1) { printf("chip %-24s mutant 4: inert (1 split, the combine is not exercised)\n", pts[i].name); omega_gpu_decode_attn256_test_set_mutant(0); continue; }
                CHECK(caught, "chip mutant %d NOT caught", m);
                printf("chip %-24s mutant %d: %s (rc=%s bad=%zu worst=%.3f)\n", pts[i].name, m, caught ? "caught" : "MISSED", omega_gpu_decode_attn_rc_name(r), mb, mw);
                if (out) fprintf(out, ",{\"name\":\"%s mutant %d\",\"caught\":%d,\"rc\":\"%s\",\"bad\":%zu,\"worst_of_tol\":%.3f}", pts[i].name, m, caught, omega_gpu_decode_attn_rc_name(r), mb, mw);
                omega_gpu_decode_attn256_test_set_mutant(0);
            }
        }
        free(y); free(y2); free(qkv_k); free(kcp); case_free(&c);
        if (omega_gpu_session_is_blocked()) { printf("session latched: stopping\n"); break; }
    }
    omega_gpu_decode_attn256_cache_free(cache);
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"device_opens\":%u,\"verdict\":\"%s\"}\n", g_checks, g_failed, omega_gpu_session_open_count(), g_failed == 0 ? "OMEGA_GPU_DECODE_ATTN256_PASS" : "OMEGA_GPU_DECODE_ATTN256_FAIL"); fclose(out); }
    return 0;
}

/* --q-heads Q --kv-heads K --head-dim D: any geometry other than 24 / 4 / 256 is REFUSED (receipt verdict "REFUSED", exit 3). */
static int geometry_refusal(uint32_t qh, uint32_t kvh, uint32_t hd, const char *out_path) {
    if (qh == QH && kvh == KVH && hd == HD) return 0;
    printf("REFUSED: decode attention 256 at %u query / %u KV heads, head dim %u (built for %u / %u / %u)\n", qh, kvh, hd, QH, KVH, HD);
    if (out_path) { FILE *f = fopen(out_path, "w"); if (f) { fprintf(f, "{\"test\":\"gpu_decode_attn256_test\",\"q_heads\":%u,\"kv_heads\":%u,\"head_dim\":%u,\"verdict\":\"REFUSED\"}\n", qh, kvh, hd); fclose(f); } }
    return 3;
}

int main(int argc, char **argv) {
    int host = 0, sim = 0; const char *out_path = NULL; uint32_t reps = 20, depth = 0, per = 0, spin = 2000; int sim_pos = -1; uint32_t sim_max = 256;
    uint32_t q_heads = QH, kv_heads = KVH, head_dim = HD;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) depth = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--per-split") == 0 && i + 1 < argc) per = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--spin-us") == 0 && i + 1 < argc) spin = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-evict") == 0) g_evict = 0;
        else if (strcmp(argv[i], "--sim-pos") == 0 && i + 1 < argc) sim_pos = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sim-max") == 0 && i + 1 < argc) sim_max = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) g_only = argv[++i];
        else if (strcmp(argv[i], "--max-pos") == 0 && i + 1 < argc) g_maxp = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--q-heads") == 0 && i + 1 < argc) q_heads = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv-heads") == 0 && i + 1 < argc) kv_heads = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = (uint32_t)atoi(argv[++i]);
        else { printf("unknown argument %s\n", argv[i]); return 2; }
    }
    { int g = geometry_refusal(q_heads, kv_heads, head_dim, out_path); if (g) return g; }
    if (host) host_only();
    if (sim && sim_pos >= 0 && sim_max <= (uint32_t)sim_pos) sim_max = (uint32_t)sim_pos + 1u;
    if (sim && sim_pos >= 0) { char msg[300], rows[160]; double worst; size_t bad; int err = sim_point(sim_max, (uint32_t)sim_pos, 64, depth ? depth : 2, 0, per ? per : 256, 1.0f, 777, msg, sizeof msg, &worst, &bad, rows, sizeof rows); printf("sim pos %d: %s %s\n", sim_pos, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg); return err || bad; }
    if (sim) sim_battery();
    if (!host && !sim) {
        if (depth) omega_gpu_decode_attn256_set_depth(depth);
        if (per) omega_gpu_decode_attn256_set_per_split(per);
        (void)omega_gpu_session_set_spin_us(spin);
        chip_run(out_path, reps ? reps : 1);
        omega_gpu_session_close();
    }
    printf("gpu_decode_attn256_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
