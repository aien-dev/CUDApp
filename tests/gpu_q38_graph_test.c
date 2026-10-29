/*
 * gpu_q38_graph_test: one Qwen3.8-27B decode token as ONE command list (omega_gpu_q38_graph.h, sc #642).
 *   --host-only   the gather table against the attention API's permutation, and the create / set_layer refusals that
 *                 need no device beyond the session
 *   (default)     chip, caller holds the GB10 lock. A synthetic Qwen3.8-shaped model on random weights: hidden 5120,
 *                 intermediate 17408, vocab 248320; --layers N (2) and --mask M (0x2: bit l = layer l is attention), so by
 *                 default one GDN layer then one attention layer: both kinds and the handover between them in every token.
 *                 Tokens at positions 0, 1, 2:
 *                   A  baseline = the APIs one call at a time (FP8 / bf16 GEMV, GDN step, hd256 attention), host glue in
 *                      C: rmsnorm f64, the gather table, sigmoid gate expf, SwiGLU expf, argmax strict >; its own GDN
 *                      state and KV cache
 *                   C  main = the same token recorded as one list (omega_gpu_q38_graph_token)
 *                   B  the same list submitted one launch per submission (flush_each)
 *                 Checks per token: B == C bit for bit (x, logits, token); A vs C token identical and logits within
 *                 tolerance; x within tolerance. Negative controls: graph mutants 1..4 (SWIGLU, GATHER, GATE, post-norm
 *                 weight) must each fail the A-vs-C comparison. Then --reps timed tokens (median).
 * Verdict: exit 0 when every CHECK passed.
 */
#include "omega_gpu_q38_graph.h"
#include "omega_gpu_cmdlist.h"
#include "omega_gpu_session.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H OMEGA_GPU_Q38_HIDDEN
#define I OMEGA_GPU_Q38_INTER
#define EPS 1e-6f
#define MAX_POS 256u
#define TOKENS 3u
#define SCALE_2P64 18446744073709551616.0f
#define SCALE_2M64 5.42101086242752217e-20f
/* A vs C: |d| <= TOL_REL * max|ref|. The differences: RMSNorm f32 FFMA + MUFU.RSQ vs f64 host, SwiGLU / gate MUFU EX2 /
 * RCP vs expf (~1e-7 relative each) through 2 layers and 5120- to 17408-term sums; 1e-4 is ~100x that. */
#define TOL_REL 1e-4

static int g_checks, g_failed;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } else { printf("ok: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }
static uint16_t bf16_rne(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t r = x + 0x7fffu + ((x >> 16) & 1u); return (uint16_t)(r >> 16); }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static uint64_t median_u64(uint64_t *v, unsigned n) { qsort(v, n, sizeof *v, cmp_u64); return v[n / 2]; }

static void rmsnorm(const float *x, const float *w, float *out) {
    double sq = 0; for (unsigned i = 0; i < H; i++) sq += (double)x[i] * x[i];
    float s = (float)(1.0 / sqrt(sq / H + (double)EPS));
    for (unsigned i = 0; i < H; i++) out[i] = x[i] * s * w[i];
}
static uint32_t argmax(const float *v, uint32_t n) { uint32_t b = 0; for (uint32_t i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }
static double maxabs(const float *v, size_t n) { double m = 0; for (size_t i = 0; i < n; i++) if (fabs(v[i]) > m) m = fabs(v[i]); return m; }
static double maxdiff(const float *a, const float *b, size_t n) { double m = 0; for (size_t i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > m || d != d) m = d != d ? 1e30 : d; } return m; }

static int fp8_random(uint32_t seed, uint32_t n, uint32_t k, OmegaGpuFp8Tensor **out) {
    size_t wb = (size_t)n * k, sb = (size_t)(n / 128) * (k / 128);
    uint8_t *w = malloc(wb); uint16_t *s = malloc(sb * 2);
    if (!w || !s) { free(w); free(s); return -1; }
    uint32_t st = seed;
    for (size_t i = 0; i < wb; i += 4) { uint32_t r = lcg(&st); for (unsigned j = 0; j < 4 && i + j < wb; j++) w[i + j] = (uint8_t)((r >> (8 * j)) & 0xbfu); } /* |v| <= 1.875: never a NaN code */
    const float sc = 1.5f / sqrtf((float)k); /* outputs O(1) for O(1) inputs */
    for (size_t i = 0; i < sb; i++) s[i] = bf16_rne(frand(&st, 0.5f * sc, sc));
    int rc = omega_gpu_fp8_tensor_upload(1, n, k, w, s, out);
    free(w); free(s);
    return rc;
}
static int bf16_random(uint32_t seed, uint32_t n, uint32_t k, float lo, float hi, float mul, OmegaGpuBf16Tensor **out) {
    uint16_t *w = malloc((size_t)n * k * 2);
    if (!w) return -1;
    uint32_t st = seed;
    for (size_t i = 0; i < (size_t)n * k; i++) w[i] = bf16_rne(frand(&st, lo, hi) * mul);
    int rc = omega_gpu_bf16_tensor_upload(n, k, w, out);
    free(w);
    return rc;
}

/* ---- the synthetic model ---- */
typedef struct {
    uint32_t layers, vocab, n_gdn, n_attn;
    uint64_t mask;
    struct { OmegaGpuFp8Tensor *mix, *o, *gu, *dn; OmegaGpuBf16Tensor *ab; float in_norm[H], post_norm[H]; } L[8];
    OmegaGpuBf16Tensor *lm;
    OmegaGpuFp8Tensor *lm8;      /* --fp8-head: the same lm_head quantised by omega_gpu_fp8_quantize_bf16 */
    uint8_t *lm8_w; uint16_t *lm8_s; /* its host copy (the sampled-row host check) */
    float fnorm[H];
    float x0[TOKENS][H];
    /* the two state copies: [0] the graph's, [1] the baseline's */
    OmegaGpuGdn *gdn[2];
    OmegaGpuDecodeAttn256Cache *attn[2];
} Model;
static int g_fp8_head; /* --fp8-head: the graph and the baseline use the FP8-quantised lm_head */

static int state_alloc(Model *m, int which, uint32_t seed) {
    uint32_t s = seed;
    if (omega_gpu_gdn_alloc(m->n_gdn ? m->n_gdn : 1, &m->gdn[which]) != 0) { printf("gdn alloc: %s\n", omega_gpu_gdn_last_error()); return -1; }
    float *conv = malloc((size_t)OMEGA_GPU_Q38_GDN_QKV * 4 * 4), a_log[48], dtb[48], nw[128];
    if (!conv) return -1;
    for (uint32_t l = 0; l < (m->n_gdn ? m->n_gdn : 1); l++) {
        for (size_t i = 0; i < (size_t)OMEGA_GPU_Q38_GDN_QKV * 4; i++) conv[i] = frand(&s, -0.6f, 0.6f);
        for (unsigned h = 0; h < 48; h++) { a_log[h] = frand(&s, -1.5f, 1.5f); dtb[h] = frand(&s, -3.0f, 1.0f); }
        for (unsigned i = 0; i < 128; i++) nw[i] = frand(&s, 0.8f, 1.2f);
        if (omega_gpu_gdn_set_layer(m->gdn[which], l, conv, a_log, dtb, nw) != 0) { printf("gdn set: %s\n", omega_gpu_gdn_last_error()); free(conv); return -1; }
    }
    free(conv);
    if (omega_gpu_decode_attn256_cache_alloc(m->n_attn ? m->n_attn : 1, MAX_POS, 10000000.0, OMEGA_GPU_Q38_ROTARY, &m->attn[which]) != 0) { printf("attn alloc: %s\n", omega_gpu_decode_attn_last_error()); return -1; }
    for (uint32_t l = 0; l < (m->n_attn ? m->n_attn : 1); l++) {
        float qn[256], kn[256], qp[256], kp[256];
        for (unsigned i = 0; i < 256; i++) { qn[i] = frand(&s, 0.8f, 1.2f); kn[i] = frand(&s, 0.8f, 1.2f); }
        omega_dattn256_perm_vec(qn, qp, OMEGA_GPU_Q38_ROTARY); omega_dattn256_perm_vec(kn, kp, OMEGA_GPU_Q38_ROTARY);
        if (omega_gpu_decode_attn256_set_norms(m->attn[which], l, qp, kp) != 0) return -1;
    }
    return 0;
}

static int model_build(Model *m, uint32_t layers, uint64_t mask, uint32_t vocab) {
    memset(m, 0, sizeof *m); m->layers = layers; m->mask = mask; m->vocab = vocab;
    uint32_t s = 4321;
    for (uint32_t l = 0; l < layers; l++) {
        const int attn = (int)((mask >> l) & 1u);
        if (attn) m->n_attn++; else m->n_gdn++;
        if (fp8_random(100 + l, attn ? OMEGA_GPU_Q38_ATT_MIX : OMEGA_GPU_Q38_GDN_MIX, H, &m->L[l].mix) || fp8_random(200 + l, H, OMEGA_GPU_Q38_MIXER_OUT, &m->L[l].o) ||
            fp8_random(300 + l, 2 * I, H, &m->L[l].gu) || fp8_random(400 + l, H, I, &m->L[l].dn)) { printf("fp8 upload: %s\n", omega_gpu_fp8_last_error()); return -1; }
        if (!attn && bf16_random(500 + l, OMEGA_GPU_Q38_AB, H, -0.03f, 0.03f, SCALE_2M64, &m->L[l].ab)) { printf("bf16 upload: %s\n", omega_gpu_bf16_last_error()); return -1; }
        for (unsigned i = 0; i < H; i++) { m->L[l].in_norm[i] = frand(&s, 0.8f, 1.2f); m->L[l].post_norm[i] = frand(&s, 0.8f, 1.2f); }
    }
    if (bf16_random(600, vocab, H, -0.05f, 0.05f, 1.0f, &m->lm)) { printf("lm_head upload: %s\n", omega_gpu_bf16_last_error()); return -1; }
    if (g_fp8_head) {
        uint16_t *w = malloc((size_t)vocab * H * 2); m->lm8_w = malloc((size_t)vocab * H); m->lm8_s = malloc((size_t)(vocab / 128) * (H / 128) * 2);
        if (!w || !m->lm8_w || !m->lm8_s) return -1;
        uint32_t st = 600; for (size_t i = 0; i < (size_t)vocab * H; i++) w[i] = bf16_rne(frand(&st, -0.05f, 0.05f)); /* bf16_random(600, ...)'s values */
        int qrc = omega_gpu_fp8_quantize_bf16(vocab, H, w, m->lm8_w, m->lm8_s); free(w);
        if (qrc || omega_gpu_fp8_tensor_upload(1, vocab, H, m->lm8_w, m->lm8_s, &m->lm8)) { printf("fp8 lm_head: %s\n", omega_gpu_fp8_last_error()); return -1; }
    }
    for (unsigned i = 0; i < H; i++) m->fnorm[i] = frand(&s, 0.8f, 1.2f);
    for (unsigned t = 0; t < TOKENS; t++) for (unsigned i = 0; i < H; i++) m->x0[t][i] = frand(&s, -1.0f, 1.0f);
    if (state_alloc(m, 0, 777) || state_alloc(m, 1, 777)) return -1;
    return 0;
}

static int state_reset(Model *m) {
    /* GDN state to zero; the KV rows at the test's positions are rewritten by every token */
    return omega_gpu_gdn_reset(m->gdn[0]) || omega_gpu_gdn_reset(m->gdn[1]);
}

/* ---- A: the APIs one call at a time ---- */
typedef struct { float x[H], logits[248320]; uint32_t token; } Out;

static int baseline_token(Model *m, const float *x0, uint32_t pos, Out *o) {
    static float xn[H], xs[H], mix[OMEGA_GPU_Q38_GDN_MIX], ab[OMEGA_GPU_Q38_AB], agg[OMEGA_GPU_Q38_ATT_MIX], att[OMEGA_GPU_Q38_MIXER_OUT], y[H], gu[2 * I], act[I];
    static uint32_t tab[OMEGA_GPU_Q38_ATT_MIX];
    omega_q38_gather_table(tab);
    memcpy(o->x, x0, sizeof o->x);
    uint32_t ng = 0, na = 0;
    for (uint32_t l = 0; l < m->layers; l++) {
        rmsnorm(o->x, m->L[l].in_norm, xn);
        { int r_ = omega_gpu_fp8_gemv_f32(m->L[l].mix, 0, xn, mix, NULL); if (r_ != 0) { printf("fp8 gemv rc %d: %s\n", r_, omega_gpu_fp8_last_error()); return -1; } }
        if (!((m->mask >> l) & 1u)) {
            for (unsigned i = 0; i < H; i++) xs[i] = xn[i] * SCALE_2P64; /* the a|b rows are uploaded * 2^-64 */
            { int r_ = omega_gpu_bf16_gemv_f32(m->L[l].ab, xs, ab, NULL); if (r_ != 0) { printf("bf16 gemv rc %d: %s\n", r_, omega_gpu_bf16_last_error()); return -1; } }
            if (omega_gpu_gdn_step(m->gdn[1], ng++, mix, mix + OMEGA_GPU_Q38_GDN_QKV, ab, EPS, EPS, 1.0f, 1.0f, att, NULL) != 0) { printf("gdn step: %s\n", omega_gpu_gdn_last_error()); return -1; }
        } else {
            for (unsigned n = 0; n < OMEGA_GPU_Q38_ATT_MIX; n++) agg[n] = mix[tab[n]];
            if (omega_gpu_decode_attn256(m->attn[1], na++, pos, agg, EPS, 1.0f, att, NULL) != 0) { printf("attn: %s\n", omega_gpu_decode_attn_last_error()); return -1; }
            for (unsigned n = 0; n < OMEGA_GPU_Q38_MIXER_OUT; n++) att[n] = att[n] * (1.0f / (1.0f + expf(-agg[OMEGA_GPU_Q38_ATT_QKV + n])));
        }
        { int r_ = omega_gpu_fp8_gemv_f32(m->L[l].o, 0, att, y, NULL); if (r_ != 0) { printf("fp8 gemv rc %d: %s\n", r_, omega_gpu_fp8_last_error()); return -1; } }
        for (unsigned i = 0; i < H; i++) o->x[i] = o->x[i] + y[i];
        rmsnorm(o->x, m->L[l].post_norm, xn);
        { int r_ = omega_gpu_fp8_gemv_f32(m->L[l].gu, 0, xn, gu, NULL); if (r_ != 0) { printf("fp8 gemv rc %d: %s\n", r_, omega_gpu_fp8_last_error()); return -1; } }
        for (unsigned j = 0; j < I; j++) { float v = gu[j]; act[j] = v / (1.0f + expf(-v)) * gu[I + j]; }
        { int r_ = omega_gpu_fp8_gemv_f32(m->L[l].dn, 0, act, y, NULL); if (r_ != 0) { printf("fp8 gemv rc %d: %s\n", r_, omega_gpu_fp8_last_error()); return -1; } }
        for (unsigned i = 0; i < H; i++) o->x[i] = o->x[i] + y[i];
    }
    rmsnorm(o->x, m->fnorm, xn);
    if (m->lm8) {
        int r_ = omega_gpu_fp8_gemv_f32(m->lm8, 0, xn, o->logits, NULL); if (r_ != 0) { printf("fp8 head rc %d: %s\n", r_, omega_gpu_fp8_last_error()); return -1; }
        double worst = 0, ref_max = 0; /* every 97th row against the dequantised host product */
        for (uint32_t r = 0; r < m->vocab; r += 97) {
            double acc = 0;
            for (uint32_t c = 0; c < H; c++) { uint8_t q = m->lm8_w[(size_t)r * H + c]; unsigned e = (q >> 3) & 15u, mm = q & 7u; double v = e ? (1.0 + mm / 8.0) * ldexp(1.0, (int)e - 7) : mm / 8.0 * ldexp(1.0, -6); if (q & 0x80u) v = -v;
                uint32_t sb = (uint32_t)m->lm8_s[(r / 128) * (H / 128) + c / 128] << 16; float sf; memcpy(&sf, &sb, 4); acc += v * sf * xn[c]; }
            double d = fabs(acc - o->logits[r]); if (d > worst) worst = d; if (fabs(acc) > ref_max) ref_max = fabs(acc);
        }
        CHECK(worst <= 1e-4 * ref_max + 1e-6, "fp8 head: sampled rows max diff %.3g vs host dequantised (max %.3g)", worst, ref_max);
    } else
    { int r_ = omega_gpu_bf16_gemv_f32(m->lm, xn, o->logits, NULL); if (r_ != 0) { printf("bf16 gemv rc %d: %s\n", r_, omega_gpu_bf16_last_error()); return -1; } }
    o->token = argmax(o->logits, m->vocab);
    return 0;
}

static int graph_build(Model *m, OmegaGpuQ38Graph **out) {
    OmegaGpuQ38GraphConfig c; memset(&c, 0, sizeof c);
    c.layers = m->layers; c.attn_mask = m->mask; c.vocab = m->vocab; c.eps = EPS; c.eps_l2 = EPS;
    c.gdn = m->gdn[0]; c.attn = m->attn[0]; c.lm_head = m->lm;
    int rc = omega_gpu_q38_graph_create(&c, out);
    if (rc != 0) { printf("graph create %d: %s\n", rc, omega_gpu_session_last_error()); return -1; }
    for (uint32_t l = 0; l < m->layers; l++)
        if ((rc = omega_gpu_q38_graph_set_layer(*out, l, m->L[l].mix, m->L[l].ab, m->L[l].o, m->L[l].gu, m->L[l].dn, m->L[l].in_norm, m->L[l].post_norm)) != 0) { printf("set_layer %u: %d\n", l, rc); return -1; }
    if (omega_gpu_q38_graph_set_final_norm(*out, m->fnorm) != 0) return -1;
    if (m->lm8 && omega_gpu_q38_graph_set_lm_head_fp8(*out, m->lm8) != 0) { printf("set_lm_head_fp8 refused\n"); return -1; }
    return 0;
}

static int graph_token(OmegaGpuQ38Graph *g, Model *m, const float *x0, uint32_t pos, Out *o, OmegaGpuDecodeGraphStats *st) {
    int rc = omega_gpu_q38_graph_token(g, x0, pos, &o->token, st);
    if (rc != 0) { printf("graph token rc %d: %s\n", rc, omega_gpu_session_last_error()); return -1; }
    (void)m;
    return omega_gpu_q38_graph_read(g, o->x, NULL, o->logits) != 0 ? -1 : 0;
}

static int host_only(void) {
    static uint32_t tab[OMEGA_GPU_Q38_ATT_MIX];
    omega_q38_gather_table(tab);
    int bad = 0;
    for (uint32_t h = 0; h < 24; h++) for (uint32_t i = 0; i < 256; i++) if (tab[h * 256 + i] != h * 512 + omega_dattn256_perm_slot(i, 64)) bad++;
    for (uint32_t h = 0; h < 4; h++) for (uint32_t i = 0; i < 256; i++) if (tab[6144 + h * 256 + i] != 12288 + h * 256 + omega_dattn256_perm_slot(i, 64)) bad++;
    for (uint32_t i = 0; i < 1024; i++) if (tab[7168 + i] != 13312 + i) bad++;
    for (uint32_t h = 0; h < 24; h++) for (uint32_t d = 0; d < 256; d++) if (tab[8192 + h * 256 + d] != h * 512 + 256 + d) bad++;
    CHECK(bad == 0, "gather table: q, k permuted, v and gate in place (%d wrong)", bad);
    static uint8_t seen[OMEGA_GPU_Q38_ATT_MIX];
    int dup = 0; for (uint32_t n = 0; n < OMEGA_GPU_Q38_ATT_MIX; n++) { if (tab[n] >= OMEGA_GPU_Q38_ATT_MIX || seen[tab[n]]) dup++; else seen[tab[n]] = 1; }
    CHECK(dup == 0, "gather table is a permutation of the 14336 mix values (%d repeats or out of range)", dup);
    OmegaGpuQ38Graph *g = NULL;
    OmegaGpuQ38GraphConfig c; memset(&c, 0, sizeof c);
    CHECK(omega_gpu_q38_graph_create(&c, &g) == OMEGA_GPU_DGRAPH_BAD_ARGS && !g, "create refuses an empty config");
    CHECK(omega_gpu_q38_graph_create(NULL, &g) == OMEGA_GPU_DGRAPH_BAD_ARGS, "create refuses NULL");
    return g_failed ? 1 : 0;
}

static uint32_t tl_reps = 1; /* --tl-reps: stamped tokens for the timeline (more than 1: the first is dropped, medians printed) */

static int chip_run(uint32_t layers, uint64_t mask, uint32_t reps) {
    Model m;
    const uint32_t vocab = 248320u;
    if (model_build(&m, layers, mask, vocab) != 0) { CHECK(0, "model build"); return 1; }
    printf("model: %u layers (%u GDN, %u attention, mask 0x%" PRIx64 "), vocab %u\n", layers, m.n_gdn, m.n_attn, mask, vocab);
    static Out A[TOKENS], C[TOKENS], B, M;
    /* A first: it also warms every kernel the record paths need */
    for (uint32_t t = 0; t < TOKENS; t++) if (baseline_token(&m, m.x0[t], t, &A[t]) != 0) { CHECK(0, "baseline token %u", t); return 1; }
    OmegaGpuQ38Graph *g = NULL;
    if (graph_build(&m, &g) != 0) { CHECK(0, "graph build"); return 1; }
    OmegaGpuDecodeGraphStats st;
    for (uint32_t t = 0; t < TOKENS; t++) {
        if (graph_token(g, &m, m.x0[t], t, &C[t], &st) != 0) { CHECK(0, "graph token %u", t); return 1; }
        const double ref = maxabs(A[t].logits, vocab), dl = maxdiff(A[t].logits, C[t].logits, vocab);
        const double xref = maxabs(A[t].x, H), dx = maxdiff(A[t].x, C[t].x, H);
        CHECK(C[t].token == A[t].token, "token %u: graph %u == baseline %u", t, C[t].token, A[t].token);
        CHECK(dl <= TOL_REL * ref, "token %u: logits max diff %.3g <= %.0e * %.3g", t, dl, TOL_REL, ref);
        CHECK(dx <= TOL_REL * xref, "token %u: x max diff %.3g <= %.0e * %.3g", t, dx, TOL_REL, xref);
        CHECK(st.launches == m.n_gdn * 9u + m.n_attn * 10u + 4u && st.submits == 1u, "token %u: %u launches in %u submit", t, st.launches, st.submits);
    }
    /* B: flush_each over the same three tokens from a reset state, bit-identical to C */
    if (state_reset(&m) != 0) { CHECK(0, "state reset"); return 1; }
    omega_gpu_q38_graph_set_flush_each(g, 1);
    int bsame = 1;
    for (uint32_t t = 0; t < TOKENS; t++) {
        if (graph_token(g, &m, m.x0[t], t, &B, &st) != 0) { CHECK(0, "flush_each token %u", t); return 1; }
        if (memcmp(B.x, C[t].x, sizeof B.x) || memcmp(B.logits, C[t].logits, vocab * 4u) || B.token != C[t].token) bsame = 0;
    }
    omega_gpu_q38_graph_set_flush_each(g, 0);
    CHECK(bsame, "flush_each == one list, bit for bit, all %u tokens", TOKENS);
    /* read_buf: x from the host-visible buffer; gu and act from the GPU-cached one (a chip copy into the staging region first).
     * act = silu(gu[c]) * gu[I + c] * 2^64 (the last layer's SwiGLU) ties the two copies together. */
    {
        static float rx[H], rgu[2 * I], ract[I], rgu100[100];
        const int r1 = omega_gpu_q38_graph_read_buf(g, OMEGA_GPU_Q38_BUF_X, rx, H), r2 = omega_gpu_q38_graph_read_buf(g, OMEGA_GPU_Q38_BUF_GU, rgu, 2 * I);
        const int r3 = omega_gpu_q38_graph_read_buf(g, OMEGA_GPU_Q38_BUF_ACT, ract, I), r4 = omega_gpu_q38_graph_read_buf(g, OMEGA_GPU_Q38_BUF_GU, rgu100, 100);
        CHECK(r1 == 0 && r2 == 0 && r3 == 0 && r4 == 0, "read_buf x, gu, act, gu[0..100): rc %d %d %d %d", r1, r2, r3, r4);
        CHECK(memcmp(rx, B.x, sizeof rx) == 0, "read_buf x == read x, bit for bit");
        CHECK(memcmp(rgu100, rgu, sizeof rgu100) == 0, "read_buf of 100 values == the first 100 of the whole row");
        double worst = 0, amax = 0;
        for (uint32_t c = 0; c < I; c++) {
            const double v = rgu[c], want = v / (1.0 + exp(-v)) * (double)rgu[I + c] * 0x1p64;
            if (fabs(want) > amax) amax = fabs(want);
            if (fabs(want - ract[c]) > worst) worst = fabs(want - ract[c]);
        }
        CHECK(amax > 0 && worst <= 1e-5 * amax, "read_buf act == silu(gu) * up * 2^64 from read_buf gu: worst %.3g of %.3g", worst, amax);
    }
    /* negative controls: each graph mutant must fail the A comparison on some token */
    const char *mname[5] = { "", "swiglu up half -1", "gather index +1", "gate dropped", "post-norm weight dropped" };
    for (int mu = 1; mu <= 4; mu++) {
        if (state_reset(&m) != 0) return 1;
        omega_gpu_q38_graph_test_set_mutant(g, mu);
        int caught = 0;
        for (uint32_t t = 0; t < TOKENS; t++) {
            if (graph_token(g, &m, m.x0[t], t, &M, &st) != 0) { caught = 1; break; }
            if (maxdiff(A[t].x, M.x, H) > TOL_REL * maxabs(A[t].x, H)) caught = 1;
        }
        CHECK(caught, "mutant %d (%s) caught", mu, mname[mu]);
    }
    omega_gpu_q38_graph_test_set_mutant(g, 0);
    /* timing: reps tokens at position 2 over the reset state (the time, not the values, is measured here) */
    uint64_t *chip = calloc(reps, 8), *tot = calloc(reps, 8);
    if (!chip || !tot) return 1;
    for (uint32_t r = 0; r < reps; r++) {
        if (state_reset(&m) != 0 || graph_token(g, &m, m.x0[0], 2, &M, &st) != 0) { CHECK(0, "timed token %u", r); return 1; }
        chip[r] = st.chip_ns; tot[r] = st.total_ns;
    }
    printf("timing: %u reps, median chip %.1f us, median total %.1f us per token (%u layers)\n", reps, median_u64(chip, reps) / 1e3, median_u64(tot, reps) / 1e3, layers);
    free(chip); free(tot);
    /* timeline: tl_reps stamped tokens (first one dropped when there are more), median GPU time per launch kind, and the
     * weight bytes each kind streams so the line reads as GB/s (activations and state are not counted) */
    {
        static uint64_t stp[OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1]; static uint8_t kd[OMEGA_GPU_CMDLIST_MAX_LAUNCHES];
        uint64_t wb[OMEGA_GPU_Q38_K_COUNT] = { 0 }; uint32_t e, n8, k8;
        for (uint32_t l = 0; l < m.layers; l++) {
            const OmegaGpuFp8Tensor *f[4] = { m.L[l].mix, m.L[l].o, m.L[l].gu, m.L[l].dn };
            const unsigned fk[4] = { OMEGA_GPU_Q38_K_MIX, OMEGA_GPU_Q38_K_O, OMEGA_GPU_Q38_K_GATE_UP, OMEGA_GPU_Q38_K_DOWN };
            for (int j = 0; j < 4; j++) if (f[j]) { omega_gpu_fp8_tensor_shape(f[j], &e, &n8, &k8); wb[fk[j]] += (uint64_t)n8 * k8; }
            if (m.L[l].ab) { omega_gpu_bf16_tensor_shape(m.L[l].ab, &n8, &k8); wb[OMEGA_GPU_Q38_K_AB] += 2ull * n8 * k8; }
        }
        omega_gpu_bf16_tensor_shape(m.lm, &n8, &k8); wb[OMEGA_GPU_Q38_K_LM_HEAD] = 2ull * n8 * k8;
        if (m.lm8) wb[OMEGA_GPU_Q38_K_LM_HEAD] /= 2u; /* FP8 head: one byte per weight */
        const uint32_t tr = tl_reps > 1 ? tl_reps - 1 : 1;
        uint64_t *ns = calloc((size_t)tr * (OMEGA_GPU_Q38_K_COUNT + 1), 8), col[256];
        uint32_t cnt[OMEGA_GPU_Q38_K_COUNT] = { 0 };
        if (!ns || tr > 256) { CHECK(0, "timeline buffers (tl-reps <= 257)"); return 1; }
        omega_gpu_q38_graph_set_stamps(g, 1);
        for (uint32_t r = 0; r < tl_reps; r++) {
            if (state_reset(&m) != 0 || graph_token(g, &m, m.x0[0], 2, &M, &st) != 0) { CHECK(0, "stamped token %u", r); return 1; }
            int n = omega_gpu_q38_graph_stamps(g, stp, kd, OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1);
            CHECK(n == (int)st.launches, "timeline %u: %d stamped launches", r, n);
            if (tl_reps > 1 && r == 0) continue;
            uint64_t *row = ns + (size_t)(tl_reps > 1 ? r - 1 : 0) * (OMEGA_GPU_Q38_K_COUNT + 1);
            memset(cnt, 0, sizeof cnt);
            for (int i = 0; i < n; i++) { row[kd[i]] += stp[i + 1] - stp[i]; cnt[kd[i]]++; }
            if (n > 0) row[OMEGA_GPU_Q38_K_COUNT] = stp[n] - stp[0];
        }
        omega_gpu_q38_graph_set_stamps(g, 0);
        for (unsigned k = 0; k <= OMEGA_GPU_Q38_K_COUNT; k++) {
            if (k < OMEGA_GPU_Q38_K_COUNT && !cnt[k]) continue;
            uint64_t lo = UINT64_MAX, hi = 0;
            for (uint32_t r = 0; r < tr; r++) { col[r] = ns[(size_t)r * (OMEGA_GPU_Q38_K_COUNT + 1) + k]; if (col[r] < lo) lo = col[r]; if (col[r] > hi) hi = col[r]; }
            const double med = (double)median_u64(col, tr);
            if (k == OMEGA_GPU_Q38_K_COUNT) { printf("timeline span %.1f us (median of %u, min %.1f max %.1f)\n", med / 1e3, tr, lo / 1e3, hi / 1e3); break; }
            printf("timeline %-8s %3u launches %9.1f us (min %.1f max %.1f)", omega_gpu_q38_graph_kind_name((uint8_t)k), cnt[k], med / 1e3, lo / 1e3, hi / 1e3);
            if (wb[k]) printf(" %12" PRIu64 " weight B %6.1f GB/s", wb[k], (double)wb[k] / med);
            printf("\n");
        }
        free(ns);
    }
    omega_gpu_q38_graph_free(g);
    return g_failed ? 1 : 0;
}

int main(int argc, char **argv) {
    int host = 0; uint32_t layers = 2, reps = 10; uint64_t mask = 0x2;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--mask") == 0 && i + 1 < argc) mask = strtoull(argv[++i], NULL, 0);
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--tl-reps") == 0 && i + 1 < argc) tl_reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--fp8-head") == 0) g_fp8_head = 1;
        /* tuning knobs for sweeps (process-wide, before any kernel is built): FP8 GEMV lane_bytes,rows,warps / CTA budget,
         * bf16 GEMV (lm_head, a|b) rows,warps,depth / CTA budget */
        else if (strcmp(argv[i], "--fp8-shape") == 0 && i + 1 < argc) { unsigned a, b, c; if (sscanf(argv[++i], "%u,%u,%u", &a, &b, &c) != 3) return 2; omega_gpu_fp8_set_shape(a, b, c); }
        else if (strcmp(argv[i], "--fp8-ctas") == 0 && i + 1 < argc) omega_gpu_fp8_set_cta_budget((uint32_t)atoi(argv[++i]));
        else if (strcmp(argv[i], "--bf16-shape") == 0 && i + 1 < argc) { unsigned a, b, c; if (sscanf(argv[++i], "%u,%u,%u", &a, &b, &c) != 3) return 2; omega_gpu_bf16_set_shape(a, b, c); }
        else if (strcmp(argv[i], "--bf16-ctas") == 0 && i + 1 < argc) omega_gpu_bf16_set_cta_budget((uint32_t)atoi(argv[++i]));
        else { printf("usage: %s [--host-only] [--layers N (<= 8)] [--mask M] [--reps R] [--tl-reps T (<= 257)]\n", argv[0]); return 2; }
    }
    if (layers == 0 || layers > 8 || reps == 0 || tl_reps == 0 || tl_reps > 257) { printf("layers 1..8, reps >= 1, tl-reps 1..257\n"); return 2; }
    int rc = host ? host_only() : chip_run(layers, mask, reps);
    printf("gpu_q38_graph_test%s: %d checks, %d failed: %s\n", host ? " (host-only)" : "", g_checks, g_failed, rc ? "FAIL" : "PASS");
    return rc;
}
