/*
 * gpu_decode_graph_test: OWN-30B W-L2a, one Qwen3-30B-A3B decode token as ONE command list (omega_gpu_decode_graph.h).
 *   --host-only   glue-kernel specs the graph builds (codegen, no device) and the record entry points' argument refusals
 *   (default)     chip, caller holds the GB10 lock. A synthetic Qwen3-30B-A3B-shaped step on random weights (hidden 2048,
 *                 128 experts top-8, intermediate 768, head_dim 128, 32/4 heads; --layers N (2), --vocab V (151936),
 *                 pos 300 over a random KV prefix: 2 attention splits):
 *                   A  baseline = today's path: the four APIs one call at a time, host glue in C mirroring sc
 *                      qwen3_moe_omega.rs step() (rmsnorm f64, x += o, device router, gate_up from the router's table,
 *                      SwiGLU expf, down, combine, final rmsnorm, lm_head, argmax strict >)
 *                   C  main = the same step recorded as one list (omega_gpu_decode_graph_token)
 *                   B  the same list submitted one launch per submission (flush_each): batching comparison
 *                 Checks: C deterministic over the reps; B == C bit for bit (x, hidden, logits, token); A vs C token
 *                 identical and logits within tolerance; negative controls (SWIGLU mutant, COMBINE mutant) must fail
 *                 the A-vs-C comparison. Per-token medians over --reps (20). --out receipt.json
 *   --red         the C path with the wrong-wiring mutant 3 (attention at scale 1, not 2^64): this run MUST FAIL
 *                 (red-before-green receipt).
 * Verdict: exit 0 when every CHECK passed.
 */
#include "omega_gpu_decode_graph.h"
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_codegen_dec.h"
#include "omega_gpu_session.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H 2048u
#define QKV 5120u
#define ATT 4096u
#define NE 128u
#define TK 8u
#define I 768u
#define EPS 1e-6f
#define POS 300u
#define MAX_POS 512u
#define SCALE_2P64 18446744073709551616.0f
/* A vs C logits: |dl| <= TOL_REL * max|l_A|. Sources of difference (header): RMSNorm f32 FFMA + MUFU.RSQ vs f64 host,
 * SwiGLU MUFU EX2/RCP vs expf (~1e-7 relative each), carried through 2 layers of random weights and 2048-term sums;
 * 1e-4 is ~100x that. Mutants move logits by O(1e-1..1) of max|l|. */
#define TOL_REL 1e-4

static int g_checks, g_failed;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } else { printf("ok: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }
static uint16_t bf16_rne(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t r = x + 0x7fffu + ((x >> 16) & 1u); return (uint16_t)(r >> 16); }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static uint64_t median_u64(uint64_t *v, unsigned n) { qsort(v, n, sizeof *v, cmp_u64); return v[n / 2]; }

/* ---- host glue: the Rust step()'s operations ---- */
static void rmsnorm(const float *x, const float *w, float *out) { /* tensor.rs rmsnorm */
    double sq = 0; for (unsigned i = 0; i < H; i++) sq += (double)x[i] * x[i];
    float s = (float)(1.0 / sqrt(sq / H + (double)EPS));
    for (unsigned i = 0; i < H; i++) out[i] = x[i] * s * w[i];
}
static void swiglu(const float *gu, float *act) { /* qwen3_moe_omega.rs swiglu: g / (1 + exp(-g)) * up */
    for (unsigned g = 0; g < TK; g++) for (unsigned j = 0; j < I; j++) {
        float v = gu[g * 2 * I + j];
        act[g * I + j] = v / (1.0f + expf(-v)) * gu[g * 2 * I + I + j];
    }
}
static uint32_t argmax(const float *v, uint32_t n) { uint32_t b = 0; for (uint32_t i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }

/* ---- the synthetic model ---- */
typedef struct {
    uint32_t layers, vocab;
    OmegaGpuFp8Tensor *qkv, *o, *gu, *dn; /* shared by every layer (memory); norms, router and KV differ per layer */
    OmegaGpuRouter *router;
    OmegaGpuDecodeAttnCache *attn;
    OmegaGpuBf16Tensor *lm;
    float *in_norm, *fnorm, *x0;
} Model;

static int fp8_random(uint32_t seed, uint32_t e, uint32_t n, uint32_t k, OmegaGpuFp8Tensor **out) {
    size_t wb = (size_t)e * n * k, sb = (size_t)e * (n / 128) * (k / 128);
    uint8_t *w = malloc(wb); uint16_t *s = malloc(sb * 2);
    if (!w || !s) { free(w); free(s); return -1; }
    uint32_t st = seed;
    for (size_t i = 0; i < wb; i += 4) { uint32_t r = lcg(&st); for (unsigned j = 0; j < 4 && i + j < wb; j++) w[i + j] = (uint8_t)((r >> (8 * j)) & 0xbfu); } /* |v| <= 1.875: never a NaN code */
    for (size_t i = 0; i < sb; i++) s[i] = bf16_rne(frand(&st, 0.02f, 0.04f));
    int rc = omega_gpu_fp8_tensor_upload(e, n, k, w, s, out);
    free(w); free(s);
    return rc;
}

static int model_build(Model *m, uint32_t layers, uint32_t vocab) {
    memset(m, 0, sizeof *m); m->layers = layers; m->vocab = vocab;
    uint32_t s = 1234;
    if (fp8_random(11, 1, QKV, H, &m->qkv) || fp8_random(12, 1, H, ATT, &m->o) || fp8_random(13, NE, 2 * I, H, &m->gu) || fp8_random(14, NE, H, I, &m->dn)) { printf("fp8 upload: %s\n", omega_gpu_fp8_last_error()); return -1; }
    if (omega_gpu_router_alloc(layers, &m->router) != 0) { printf("router alloc: %s\n", omega_gpu_router_last_error()); return -1; }
    uint16_t *gate = malloc((size_t)NE * H * 2); float *nw = malloc(H * 4);
    m->in_norm = malloc((size_t)layers * H * 4); m->fnorm = malloc(H * 4); m->x0 = malloc(H * 4);
    if (!gate || !nw || !m->in_norm || !m->fnorm || !m->x0) return -1;
    for (uint32_t l = 0; l < layers; l++) {
        for (size_t i = 0; i < (size_t)NE * H; i++) gate[i] = bf16_rne(frand(&s, -0.05f, 0.05f));
        for (unsigned i = 0; i < H; i++) nw[i] = frand(&s, 0.8f, 1.2f);
        if (omega_gpu_router_set_layer(m->router, l, gate, nw) != 0 || omega_gpu_router_set_directory_fp8(m->router, l, m->gu, m->dn) != 0) { printf("router layer: %s\n", omega_gpu_router_last_error()); return -1; }
        for (unsigned i = 0; i < H; i++) m->in_norm[(size_t)l * H + i] = frand(&s, 0.8f, 1.2f);
    }
    free(gate); free(nw);
    for (unsigned i = 0; i < H; i++) { m->fnorm[i] = frand(&s, 0.8f, 1.2f); m->x0[i] = frand(&s, -1.0f, 1.0f); }
    if (omega_gpu_decode_attn_cache_alloc(layers, MAX_POS, 1000000.0, &m->attn) != 0) { printf("attn alloc: %s\n", omega_gpu_decode_attn_last_error()); return -1; }
    float qn[128], kn[128]; uint16_t *kr = malloc((size_t)POS * 128 * 2), *vr = malloc((size_t)POS * 128 * 2);
    if (!kr || !vr) return -1;
    for (uint32_t l = 0; l < layers; l++) {
        for (unsigned i = 0; i < 128; i++) { qn[i] = frand(&s, 0.8f, 1.2f); kn[i] = frand(&s, 0.8f, 1.2f); }
        if (omega_gpu_decode_attn_set_norms(m->attn, l, qn, kn) != 0) return -1;
        for (uint32_t h = 0; h < 4; h++) {
            for (size_t i = 0; i < (size_t)POS * 128; i++) { kr[i] = bf16_rne(frand(&s, -1, 1)); vr[i] = bf16_rne(frand(&s, -1, 1)); }
            if (omega_gpu_decode_attn_cache_write(m->attn, l, h, 0, POS, kr, vr) != 0) { printf("kv write: %s\n", omega_gpu_decode_attn_last_error()); return -1; }
        }
    }
    free(kr); free(vr);
    uint16_t *lw = malloc((size_t)vocab * H * 2);
    if (!lw) return -1;
    for (size_t i = 0; i < (size_t)vocab * H; i++) lw[i] = bf16_rne(frand(&s, -0.05f, 0.05f));
    int rc = omega_gpu_bf16_tensor_upload(vocab, H, lw, &m->lm);
    free(lw);
    if (rc != 0) { printf("lm_head upload: %s\n", omega_gpu_bf16_last_error()); return -1; }
    return 0;
}

/* A: today's path, one API call at a time, host glue */
static int step_host(const Model *m, float *x, float *hidden, float *logits, uint32_t *tok, uint32_t *launches) {
    static float xn[H], qkv[QKV], att[ATT], op[H], gu[TK * 2 * I], act[TK * I], dn[TK * H], mo[H];
    uint32_t ids[TK], zero = 0; float wts[TK]; uint64_t xout = 0, tgu = 0, tdn = 0;
    memcpy(x, m->x0, H * 4); *launches = 0;
    for (uint32_t l = 0; l < m->layers; l++) {
        rmsnorm(x, m->in_norm + (size_t)l * H, xn);
        if (omega_gpu_fp8_gemv_grouped_f32(m->qkv, 1, &zero, xn, 1, qkv, NULL)) return printf("A qkv: %s\n", omega_gpu_fp8_last_error()), -1;
        if (omega_gpu_decode_attn(m->attn, l, POS, qkv, EPS, 1.0f, att, NULL)) return printf("A attn: %s\n", omega_gpu_decode_attn_last_error()), -1;
        if (omega_gpu_fp8_gemv_grouped_f32(m->o, 1, &zero, att, 1, op, NULL)) return printf("A o: %s\n", omega_gpu_fp8_last_error()), -1;
        for (unsigned i = 0; i < H; i++) x[i] += op[i];
        if (omega_gpu_router(m->router, l, x, EPS, 1, SCALE_2P64, ids, wts, NULL)) return printf("A router: %s\n", omega_gpu_router_last_error()), -1;
        omega_gpu_router_device_outputs(m->router, &xout, &tgu, &tdn);
        if (omega_gpu_fp8_gemv_grouped_dev(m->gu, TK, tgu, xout, gu, NULL)) return printf("A gate_up: %s\n", omega_gpu_fp8_last_error()), -1;
        swiglu(gu, act);
        if (omega_gpu_fp8_gemv_grouped_f32(m->dn, TK, ids, act, TK, dn, NULL)) return printf("A down: %s\n", omega_gpu_fp8_last_error()), -1;
        for (unsigned i = 0; i < H; i++) mo[i] = 0.0f;
        for (unsigned g = 0; g < TK; g++) for (unsigned i = 0; i < H; i++) mo[i] += wts[g] * dn[g * H + i];
        for (unsigned i = 0; i < H; i++) x[i] += mo[i];
        *launches += 6;
    }
    rmsnorm(x, m->fnorm, hidden);
    if (omega_gpu_bf16_gemv_f32(m->lm, hidden, logits, NULL)) return printf("A lm_head: %s\n", omega_gpu_bf16_last_error()), -1;
    *launches += 1;
    *tok = argmax(logits, m->vocab);
    return 0;
}

typedef struct { float x[H], hidden[H]; float *logits; uint32_t tok; } Out;
static int out_alloc(Out *o, uint32_t vocab) { memset(o, 0, sizeof *o); o->logits = malloc((size_t)vocab * 4); return o->logits ? 0 : -1; }

static int run_graph(OmegaGpuDecodeGraph *gr, const Model *m, Out *o, OmegaGpuDecodeGraphStats *st) {
    int rc = omega_gpu_decode_graph_token(gr, m->x0, POS, &o->tok, st);
    if (rc != 0) { printf("graph token rc %d: %s\n", rc, omega_gpu_session_last_error()); return rc; }
    return omega_gpu_decode_graph_read(gr, o->x, o->hidden, o->logits);
}

typedef struct { double max_abs, scale; int tok_same; } Cmp;
static Cmp compare(const Out *a, const Out *c, uint32_t vocab) {
    Cmp r = { 0, 0, a->tok == c->tok };
    for (uint32_t i = 0; i < vocab; i++) { double d = fabs((double)a->logits[i] - c->logits[i]); if (d > r.max_abs) r.max_abs = d; if (fabs(a->logits[i]) > r.scale) r.scale = fabs(a->logits[i]); }
    return r;
}
static int cmp_pass(Cmp c) { return c.tok_same && c.max_abs <= TOL_REL * c.scale; }
static double max_rel(const float *a, const float *b, unsigned n) { double mx = 0, sc = 0; for (unsigned i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > mx) mx = d; if (fabs(a[i]) > sc) sc = fabs(a[i]); } return sc > 0 ? mx / sc : mx; }

static int host_only(void) {
    static const struct { OmegaDecKind kind; uint32_t mode; int mutant; const char *name; } sp[] = {
        { OMEGA_DEC_RES, OMEGA_DEC_RES_NORM, 0, "RES NORM" }, { OMEGA_DEC_RES, OMEGA_DEC_RES_ADD, 0, "RES ADD" },
        { OMEGA_DEC_RES, OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM, 0, "RES COMBINE|NORM" }, { OMEGA_DEC_SWIGLU, 0, 0, "SWIGLU" },
        { OMEGA_DEC_ARGMAX, 0, 0, "ARGMAX identity" }, { OMEGA_DEC_ARGMAX, 1, 0, "ARGMAX indexed" },
        { OMEGA_DEC_SWIGLU, 0, 1, "SWIGLU mutant 1" }, { OMEGA_DEC_RES, OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM, 2, "RES COMBINE|NORM mutant 2" },
    };
    for (unsigned i = 0; i < sizeof sp / sizeof sp[0]; i++) {
        OmegaDecSpec s; memset(&s, 0, sizeof s); s.kind = sp[i].kind; s.mode = sp[i].mode; s.mutant = sp[i].mutant;
        uint32_t ins = 0, gpr = 0; int rc = omega_dec_prog_stats(&s, &ins, &gpr);
        CHECK(rc == 0 && ins > 0, "%s builds (%u instructions, peak %u registers)", sp[i].name, ins, gpr);
    }
    OmegaDecSpec bad; memset(&bad, 0, sizeof bad); bad.kind = OMEGA_DEC_RES; bad.mode = OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_COMBINE;
    CHECK(omega_dec_prog_stats(&bad, NULL, NULL) != 0, "RES ADD|COMBINE refused by codegen");
    /* record entry points refuse a missing list or missing device addresses before touching the device */
    CHECK(omega_gpu_fp8_gemv_grouped_record(NULL, NULL, 1, 1, 1, 1, 1) == OMEGA_GPU_FP8_BAD_ARGS, "fp8 record without a list: BAD_ARGS");
    CHECK(omega_gpu_bf16_gemv_record(NULL, NULL, 1, 1) == OMEGA_GPU_BF16_BAD_ARGS, "bf16 record without a list: BAD_ARGS");
    CHECK(omega_gpu_router_record(NULL, NULL, 0, 1, EPS, 1, 1.0f, NULL) == OMEGA_GPU_ROUTER_BAD_ARGS, "router record without a list: BAD_ARGS");
    CHECK(omega_gpu_decode_attn_record(NULL, NULL, 0, 0, 1, EPS, 1.0f, 1) == OMEGA_GPU_DATTN_BAD_ARGS, "decode attn record without a list: BAD_ARGS");
    OmegaGpuCmdList *fake = (OmegaGpuCmdList *)(uintptr_t)16; /* never dereferenced: the argument checks come first */
    CHECK(omega_gpu_fp8_gemv_grouped_record(fake, NULL, 1, 0, 1, 1, 1) == OMEGA_GPU_FP8_BAD_ARGS, "fp8 record with tab_va 0: BAD_ARGS");
    CHECK(omega_gpu_bf16_gemv_record(fake, NULL, 0, 1) == OMEGA_GPU_BF16_BAD_ARGS, "bf16 record with x_va 0: BAD_ARGS");
    CHECK(omega_gpu_router_record(fake, NULL, 0, 0, EPS, 1, 1.0f, NULL) == OMEGA_GPU_ROUTER_BAD_ARGS, "router record with a NULL router: BAD_ARGS");
    CHECK(omega_gpu_decode_attn_record(fake, NULL, 0, 0, 0, EPS, 1.0f, 1) == OMEGA_GPU_DATTN_BAD_ARGS, "decode attn record with a NULL cache: BAD_ARGS");
    OmegaGpuDecodeGraphConfig cfg; memset(&cfg, 0, sizeof cfg);
    OmegaGpuDecodeGraph *gr = NULL;
    CHECK(omega_gpu_decode_graph_create(&cfg, &gr) == OMEGA_GPU_DGRAPH_BAD_ARGS && !gr, "graph create with an empty config: BAD_ARGS");
    return g_failed ? 1 : 0;
}

static int chip_run(uint32_t layers, uint32_t vocab, uint32_t reps, int red, int cached, const char *out_path) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out) fprintf(out, "{\"layers\":%u,\"vocab\":%u,\"pos\":%u,\"reps\":%u,\"red\":%d,\"cached_weights\":%d,", layers, vocab, POS, reps, red, cached);
    omega_gpu_fp8_set_weight_cache(cached); /* one variable: the FP8 tensors uploaded below live GPU-cached or not */
    Model m;
    uint64_t t0 = omega_gpu_session_now_ns();
    if (model_build(&m, layers, vocab) != 0) { CHECK(0, "synthetic model build"); return 1; }
    printf("model built in %.1f s (layers %u, vocab %u, pos %u)\n", (omega_gpu_session_now_ns() - t0) / 1e9, layers, vocab, POS);
    OmegaGpuDecodeGraphConfig cfg = { .layers = layers, .vocab = vocab, .eps = EPS, .norm_topk = 1, .router = m.router, .attn = m.attn, .lm_head = m.lm };
    OmegaGpuDecodeGraph *gr = NULL;
    int rc = omega_gpu_decode_graph_create(&cfg, &gr);
    if (rc != 0) { CHECK(0, "graph create rc %d: %s", rc, omega_gpu_session_last_error()); return 1; }
    for (uint32_t l = 0; l < layers; l++)
        if (omega_gpu_decode_graph_set_layer(gr, l, m.qkv, m.o, m.gu, m.dn, m.in_norm + (size_t)l * H) != 0) { CHECK(0, "set_layer %u", l); return 1; }
    if (omega_gpu_decode_graph_set_final_norm(gr, m.fnorm) != 0) { CHECK(0, "set_final_norm"); return 1; }

    Out A, A0, C, C0, B; uint64_t *ta = calloc(reps, 8), *tc = calloc(reps, 8), *tb = calloc(reps, 8), *cc = calloc(reps, 8), *cb = calloc(reps, 8), *rcn = calloc(reps, 8);
    if (out_alloc(&A, vocab) || out_alloc(&A0, vocab) || out_alloc(&C, vocab) || out_alloc(&C0, vocab) || out_alloc(&B, vocab) || !ta || !tc || !tb || !cc || !cb || !rcn) return 1;

    /* A: baseline first (also fills every API's kernel cache, so no record has to build) */
    uint32_t la = 0; int a_ok = 1, a_det = 1;
    for (uint32_t r = 0; r < reps && a_ok; r++) {
        uint64_t s = omega_gpu_session_now_ns();
        if (step_host(&m, A.x, A.hidden, A.logits, &A.tok, &la) != 0) { a_ok = 0; break; }
        ta[r] = omega_gpu_session_now_ns() - s;
        if (r == 0) { memcpy(A0.x, A.x, sizeof A.x); memcpy(A0.hidden, A.hidden, sizeof A.hidden); memcpy(A0.logits, A.logits, (size_t)vocab * 4); A0.tok = A.tok; }
        else if (A.tok != A0.tok || memcmp(A.logits, A0.logits, (size_t)vocab * 4)) a_det = 0;
    }
    CHECK(a_ok, "A (baseline, host glue) %u reps ran", reps);
    if (!a_ok) return 1;
    CHECK(a_det, "A deterministic over %u reps (token %u)", reps, A0.tok);
    /* top-2 gap of the baseline: how much margin the argmax identity has */
    uint32_t t1 = A0.tok, t2 = t1 == 0 ? 1 : 0; for (uint32_t i = 0; i < vocab; i++) if (i != t1 && A0.logits[i] > A0.logits[t2]) t2 = i;
    printf("A: token %u logit %.6f, runner-up %u logit %.6f (gap %.3g)\n", t1, A0.logits[t1], t2, A0.logits[t2], (double)A0.logits[t1] - A0.logits[t2]);

    if (red) omega_gpu_decode_graph_test_set_mutant(gr, 3);
    /* C: one list per token */
    OmegaGpuDecodeGraphStats st; memset(&st, 0, sizeof st); uint32_t lc = 0, sc = 0; int c_ok = 1, c_det = 1;
    for (uint32_t r = 0; r < reps && c_ok; r++) {
        if (run_graph(gr, &m, &C, &st) != 0) { c_ok = 0; break; }
        tc[r] = st.total_ns; cc[r] = st.chip_ns; rcn[r] = st.record_ns; lc = st.launches; sc = st.submits;
        if (r == 0) { C0 = C; C0.logits = malloc((size_t)vocab * 4); memcpy(C0.logits, C.logits, (size_t)vocab * 4); }
        else if (C.tok != C0.tok || memcmp(C.logits, C0.logits, (size_t)vocab * 4) || memcmp(C.x, C0.x, sizeof C.x)) c_det = 0;
    }
    CHECK(c_ok, "C (one list per token) %u reps ran: %u launches in %u submission(s)", reps, lc, sc);
    if (!c_ok) return 1;
    CHECK(c_det, "C deterministic over %u reps (token %u)", reps, C0.tok);
    Cmp ac = compare(&A0, &C0, vocab);
    double xr = max_rel(A0.x, C0.x, H), hr = max_rel(A0.hidden, C0.hidden, H);
    printf("A vs C: token %u vs %u, logits max |d| %.3g (scale %.3g, rel %.3g, tol %.0e), x max rel %.3g, hidden max rel %.3g\n",
           A0.tok, C0.tok, ac.max_abs, ac.scale, ac.max_abs / ac.scale, TOL_REL, xr, hr);
    CHECK(ac.tok_same, "argmax token identical: A %u, C %u", A0.tok, C0.tok);
    CHECK(ac.max_abs <= TOL_REL * ac.scale, "logits A vs C within %.0e of max|logit| (max |d| %.3g, scale %.3g)", TOL_REL, ac.max_abs, ac.scale);
    if (out) fprintf(out, "\"A\":{\"token\":%u,\"runner_up\":%u,\"gap\":%.6g,\"launches\":%u},\"C\":{\"token\":%u,\"launches\":%u,\"submits\":%u},\"A_vs_C\":{\"tok_same\":%d,\"logit_max_abs\":%.6g,\"logit_scale\":%.6g,\"x_max_rel\":%.6g,\"hidden_max_rel\":%.6g,\"tol_rel\":%g},",
                     t1, t2, (double)A0.logits[t1] - A0.logits[t2], la, C0.tok, lc, sc, ac.tok_same, ac.max_abs, ac.scale, xr, hr, TOL_REL);

    if (!red) {
        /* B: same list, one launch per submission: bit-identical to C */
        omega_gpu_decode_graph_set_flush_each(gr, 1);
        uint32_t lb = 0, sb = 0; int b_ok = 1, b_same = 1;
        for (uint32_t r = 0; r < reps && b_ok; r++) {
            if (run_graph(gr, &m, &B, &st) != 0) { b_ok = 0; break; }
            tb[r] = st.total_ns; cb[r] = st.chip_ns; lb = st.launches; sb = st.submits;
            if (B.tok != C0.tok || memcmp(B.logits, C0.logits, (size_t)vocab * 4) || memcmp(B.x, C0.x, sizeof B.x) || memcmp(B.hidden, C0.hidden, sizeof B.hidden)) b_same = 0;
        }
        omega_gpu_decode_graph_set_flush_each(gr, 0);
        CHECK(b_ok, "B (one launch per submission) %u reps ran: %u launches in %u submissions", reps, lb, sb);
        CHECK(b_ok && b_same, "B bit-identical to C (x, hidden, logits, token) in every rep");

        /* negative controls: a mutant glue kernel in C must fail the A-vs-C comparison */
        for (int mut = 1; mut <= 2; mut++) {
            omega_gpu_decode_graph_test_set_mutant(gr, mut);
            Out M; out_alloc(&M, vocab);
            int mrc = run_graph(gr, &m, &M, &st);
            Cmp mc = compare(&A0, &M, vocab);
            CHECK(mrc == 0 && !cmp_pass(mc), "negative control: %s mutant fails the comparison (token %u vs %u, max |d| %.3g = %.3g of scale)",
                  mut == 1 ? "SWIGLU" : "COMBINE", M.tok, A0.tok, mc.max_abs, mc.max_abs / mc.scale);
            if (out) fprintf(out, "\"mutant%d\":{\"rc\":%d,\"token\":%u,\"logit_max_abs\":%.6g,\"caught\":%d},", mut, mrc, M.tok, mc.max_abs, mrc == 0 && !cmp_pass(mc));
            free(M.logits);
        }
        omega_gpu_decode_graph_test_set_mutant(gr, 0);
        /* after the mutants the real list must still agree (state not poisoned) */
        int prc = run_graph(gr, &m, &C, &st);
        CHECK(prc == 0 && C.tok == C0.tok && !memcmp(C.logits, C0.logits, (size_t)vocab * 4), "C after the mutants bit-identical to the first C");

        if (out) { uint64_t h = 1469598103934665603ull; const unsigned char *b = (const unsigned char *)C0.logits; for (size_t i = 0; i < (size_t)vocab * 4; i++) { h ^= b[i]; h *= 1099511628211ull; }
                   fprintf(out, "\"C0_token\":%u,\"C0_logits_fnv\":\"%016" PRIx64 "\",", C0.tok, h); }
        /* timeline instrument: stamps on -> one timestamp per launch, monotonic, span within the chip time; token unchanged; off -> refuses */
        {
            static uint64_t sts[OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1]; static uint8_t kinds[OMEGA_GPU_CMDLIST_MAX_LAUNCHES];
            omega_gpu_decode_graph_set_stamps(gr, 1);
            Out S; out_alloc(&S, vocab);
            int src_ = run_graph(gr, &m, &S, &st);
            int n = omega_gpu_decode_graph_stamps(gr, sts, kinds, OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1);
            int mono = 1; uint64_t per_kind[OMEGA_GPU_DGRAPH_K_COUNT] = { 0 }; uint32_t cnt_kind[OMEGA_GPU_DGRAPH_K_COUNT] = { 0 };
            for (int i = 0; i < n; i++) { if (sts[i + 1] < sts[i]) mono = 0; if (kinds[i] < OMEGA_GPU_DGRAPH_K_COUNT) { per_kind[kinds[i]] += sts[i + 1] - sts[i]; cnt_kind[kinds[i]]++; } }
            const uint64_t span = n > 0 ? sts[n] - sts[0] : 0;
            CHECK(src_ == 0 && n == (int)st.launches && mono && span > 0 && span <= st.chip_ns && S.tok == C0.tok && !memcmp(S.logits, C0.logits, (size_t)vocab * 4),
                  "timeline: rc %d, %d stamps for %u launches, monotonic %d, span %.1f us <= chip %.1f us, token and logits unchanged", src_, n, st.launches, mono, span / 1e3, st.chip_ns / 1e3);
            printf("TIMELINE (%u layers, pos %u):", layers, POS);
            for (unsigned k = 0; k < OMEGA_GPU_DGRAPH_K_COUNT; k++) if (cnt_kind[k]) printf(" %s x%u %.1f us", omega_gpu_decode_graph_kind_name((uint8_t)k), cnt_kind[k], per_kind[k] / 1e3);
            printf("\n");
            if (out) { fprintf(out, "\"timeline\":{\"launches\":%d,\"span_ns\":%" PRIu64 ",\"chip_ns\":%" PRIu64 ",\"per_kind_ns\":{", n, span, st.chip_ns);
                       int first = 1; for (unsigned k = 0; k < OMEGA_GPU_DGRAPH_K_COUNT; k++) if (cnt_kind[k]) { fprintf(out, "%s\"%s\":[%u,%" PRIu64 "]", first ? "" : ",", omega_gpu_decode_graph_kind_name((uint8_t)k), cnt_kind[k], per_kind[k]); first = 0; }
                       fprintf(out, "}},"); }
            omega_gpu_decode_graph_set_stamps(gr, 0);
            int orc = run_graph(gr, &m, &S, &st);
            int n2 = omega_gpu_decode_graph_stamps(gr, sts, kinds, OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1);
            CHECK(orc == 0 && n2 < 0, "timeline off: token rc %d, stamps accessor refuses (%d)", orc, n2);
            free(S.logits);
        }
        uint64_t ma = median_u64(ta, reps), mc_ = median_u64(tc, reps), mb = median_u64(tb, reps), mcc = median_u64(cc, reps), mcb = median_u64(cb, reps), mr = median_u64(rcn, reps);
        printf("PER TOKEN (median of %u, %u layers): A host-glue %u launches %.1f us | C one list %u launches %.1f us (chip %.1f us, record %.1f us) | B per-launch submits %u launches %.1f us (chip sum %.1f us)\n",
               reps, layers, la, ma / 1e3, lc, mc_ / 1e3, mcc / 1e3, mr / 1e3, lb, mb / 1e3, mcb / 1e3);
        if (out) fprintf(out, "\"median_ns\":{\"A_total\":%" PRIu64 ",\"C_total\":%" PRIu64 ",\"C_chip\":%" PRIu64 ",\"C_record\":%" PRIu64 ",\"B_total\":%" PRIu64 ",\"B_chip\":%" PRIu64 "},\"B\":{\"launches\":%u,\"submits\":%u,\"identical\":%d},",
                         ma, mc_, mcc, mr, mb, mcb, lb, sb, b_ok && b_same);
    }
    omega_gpu_decode_graph_free(gr);
    if (out) { fprintf(out, "\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS"); fclose(out); }
    return g_failed ? 1 : 0;
}

int main(int argc, char **argv) {
    int host = 0, red = 0, cached = 0, variant = 0, barrier = -1, prefetch = 0, prefetch_red = 0; uint32_t cta_budget = 0, sh_lane = 0, sh_rows = 0, sh_warps = 0; uint32_t reps = 20, layers = 2, vocab = 151936; const char *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--red") == 0) red = 1;
        else if (strcmp(argv[i], "--cached-weights") == 0) cached = 1;
        else if (strcmp(argv[i], "--prefetch") == 0) prefetch = 1;          /* W-L2d: o_proj weight prefetch beside the attention */
        else if (strcmp(argv[i], "--prefetch-red") == 0) prefetch = prefetch_red = 1; /* its red: o_proj unordered, MUST FAIL */
        else if (strcmp(argv[i], "--cta-budget") == 0 && i + 1 < argc) cta_budget = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) variant = atoi(argv[++i]); /* FP8 GEMV generator: 1 (default) or 2 (W-L2c) */
        else if (strcmp(argv[i], "--barrier") == 0 && i + 1 < argc) barrier = atoi(argv[++i]); /* per-launch barrier mode (W-L2c): 0 wfi+inval_all, 2 wfi_only, 3 wfi+inval_data, 4 flush_data */
        else if (strcmp(argv[i], "--shape") == 0 && i + 3 < argc) { sh_lane = (uint32_t)atoi(argv[++i]); sh_rows = (uint32_t)atoi(argv[++i]); sh_warps = (uint32_t)atoi(argv[++i]); }
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) vocab = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out = argv[++i];
        else { printf("usage: %s [--host-only] [--red] [--reps N] [--layers N] [--vocab V] [--out receipt.json]\n", argv[0]); return 2; }
    }
    if (reps == 0) reps = 1;
    if (layers == 0 || layers > 8) layers = 2;
    if (variant) omega_gpu_fp8_set_variant(variant);
    omega_gpu_decode_graph_set_prefetch(prefetch); omega_gpu_decode_graph_test_set_prefetch_red(prefetch_red);
    if (barrier == 1) omega_gpu_decode_graph_test_set_barrier_none(); /* negative control: this run MUST FAIL */
    else if (barrier >= 0) omega_gpu_decode_graph_set_barrier(barrier);
    if (cta_budget) omega_gpu_fp8_set_cta_budget(cta_budget);               /* one variable per run (timeline sweeps) */
    if (sh_lane) omega_gpu_fp8_set_shape(sh_lane, sh_rows, sh_warps);
    printf("dgraph prefetch %d (red %d)\n", omega_gpu_decode_graph_prefetch(), prefetch_red);
    printf("fp8 knobs: variant %d barrier %d (%s) cta_budget %u shape %u/%u/%u cached %d\n", omega_gpu_fp8_variant(), omega_gpu_decode_graph_barrier(), omega_gpu_cmdlist_barrier_name((OmegaGpuCmdListBarrier)omega_gpu_decode_graph_barrier()), cta_budget, sh_lane, sh_rows, sh_warps, cached);
    int rc = host ? host_only() : chip_run(layers, vocab, reps, red, cached, out);
    printf("%s: %d checks, %d failed: %s\n", host ? "host-only" : red ? "chip-red" : "chip", g_checks, g_failed, rc ? "FAIL" : "PASS");
    return rc;
}
