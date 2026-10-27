/* gpu_gemm_ex_test.c -- the general matrix multiply omega_gpu_gemm_ex (src/omega_gpu_gemm_ex.h): any M, N, K >= 1, op N/T,
 * leading dimensions, alpha/beta, strided batches, f32/bf16/f16/e4m3/e5m2 inputs, f32/bf16/f16 output, fp8 scales.
 *
 *   --host-only  no device: the refusals, then every route through a host model of the core (the chip-exact tensor-core
 *                models of tests/bw_hmma_model.h / bw_qmma_model.h, sequential fmaf for the FFMA reference) on a seeded
 *                case list plus the edge cases; the four planted mutants of the general path must go red; the fp8 block-rows
 *                fallback (omega_gpu_gemm_ex_fp8_block_rows) on the shapes the fp8 kernels refuse, mutant 5 red
 *   --sim        the same with the real C3 kernels (bf16, f16, e4m3 HMMA/QMMA and the FFMA reference) in the warp
 *                simulator on small cases, bit for bit against the host-model core; the FFMA and f16 kernel mutants red
 *   (no flag)    the GB10: the predeclared sweep (--list prints it; --quick: the first 40; --part I/N: every N-th case
 *                from I), the subnormal probes, the edges, the mutants, the fp8 fallback through the real fp8 GEMV and
 *                grouped GEMM APIs (refused-before shapes answer, served shapes stay on the kernels), chip core == model
 *                core bit for bit where the model is cheap enough
 *
 * The check of every output element (predeclared, printed by --list): the double reference
 *   ref = alpha s * sum_k a b f + beta c_old         (s: tensor scales, f: block scales)
 * and   |got - ref| <= |alpha s| ((S + 4) 2^-22 sum|a b f| + 2^-124 sum (|a| + |b|) |f| + k 2^-140 + F sum|a b f|)
 *                      + 2^-50 |beta c_old| + u_c |ref| + t_c
 * with S the fp32 accumulation steps (ceil(k / 16), ceil(k / 32) for e4m3, 9 ceil(k / 16) for bf16x3, k for FFMA, plus the
 * number of k parts), F = 2^-7 for FAST_BF16 (both operands rounded to bf16) else 0, u_c / t_c = 2^-24 / 2^-149 (f32),
 * 2^-8 / 2^-133 (bf16), 2^-11 / 2^-24 (f16). A non-finite ref must come back as the same class (NaN, or inf of the same
 * sign); an inf result is accepted for a finite ref only when |ref| + bound reaches the largest finite output.
 * Untouched C (padding between ldc and n, gaps between batches) must keep its bytes. With beta = 0 C is preset to NaN.
 * Last line: "gemm_ex <tier>: PASS (<checks> checks)" or "... FAIL (<failed> of <checks>)". */
#include "omega_gpu_gemm_ex.h"
#include "omega_gpu_gemm.h"
#include "omega_gpu_session.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_fp8gemm_api.h"
#include "bw_warp_sim_tc.h"

#include <inttypes.h>
#include <math.h>

static int g_checks, g_failed;
static void check(int ok, const char *what) {
    g_checks++;
    if (!ok) { g_failed++; printf("FAIL: %s\n", what); }
}

static const HmmaModel HM = { 16, 2, 0, 2, 0 };  /* chip-exact, evidence OWN30B-T3-HMMA-MODEL */
static const QmmaModel QM = { 32, 26, 0, 2, 0 }; /* chip-exact, evidence OWN30B-L1G-QMMA-PFGEMM-c742aa13 */

static uint64_t g_rng = 1;
static uint32_t rnd(void) { g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(g_rng >> 33); }
static uint32_t t_f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float t_u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static const char *DTN[5] = { "f32", "bf16", "f16", "e4m3", "e5m2" };
static const char *RTN[6] = { "none", "native", "widen", "bf16x3", "fast_bf16", "ffma" };
static uint32_t dsz(unsigned t) { return t == OMEGA_GEMM_DT_F32 ? 4u : t <= OMEGA_GEMM_DT_F16 ? 2u : 1u; }

/* an element of a stored array as a float (decoders independent of src/omega_gpu_gemm_ex.c) */
static float tval(unsigned t, const void *p, uint64_t i) {
    switch (t) {
    case OMEGA_GEMM_DT_F32: { float f; memcpy(&f, (const uint8_t *)p + i * 4u, 4); return f; }
    case OMEGA_GEMM_DT_BF16: return t_u2f((uint32_t)((const uint16_t *)p)[i] << 16);
    case OMEGA_GEMM_DT_F16: return ws_h2f(((const uint16_t *)p)[i]);
    case OMEGA_GEMM_DT_E4M3: { uint8_t c = ((const uint8_t *)p)[i]; return (c & 0x7fu) == 0x7fu ? NAN : qm_e4m3(c); }
    default: return ws_h2f((uint16_t)(((const uint8_t *)p)[i] << 8));
    }
}
static void tput(unsigned t, void *p, uint64_t i, uint32_t bits) {
    if (t == OMEGA_GEMM_DT_F32) memcpy((uint8_t *)p + i * 4u, &bits, 4);
    else if (t <= OMEGA_GEMM_DT_F16) ((uint16_t *)p)[i] = (uint16_t)bits;
    else ((uint8_t *)p)[i] = (uint8_t)bits;
}
/* random finite values: magnitudes 2^-8 .. 2^9 (e4m3: every finite code), 1 in 61 zero */
static uint32_t rnd_bits(unsigned t) {
    uint32_t r = rnd(), s = (r >> 8) & 1u, e = (r >> 9) % 17u;
    if (t != OMEGA_GEMM_DT_E4M3 && r % 61u == 0) return 0;
    switch (t) {
    case OMEGA_GEMM_DT_F32: return s << 31 | (119u + e) << 23 | (rnd() & 0x7fffffu);
    case OMEGA_GEMM_DT_BF16: return s << 15 | (119u + e) << 7 | (rnd() & 0x7fu);
    case OMEGA_GEMM_DT_F16: return s << 15 | (7u + e) << 10 | (rnd() & 0x3ffu);
    case OMEGA_GEMM_DT_E4M3: { uint32_t c = rnd() & 0xffu; return (c & 0x7fu) == 0x7fu ? c ^ 1u : c; }
    default: return s << 7 | (7u + e) << 2 | (rnd() & 3u);
    }
}
static uint32_t nan_bits(unsigned t) { return t == OMEGA_GEMM_DT_F32 ? 0x7fc00000u : t == OMEGA_GEMM_DT_BF16 ? 0x7fc0u : 0x7e00u; }
static uint32_t inf_bits(unsigned t, int neg) {
    return t == OMEGA_GEMM_DT_F32 ? (neg ? 0xff800000u : 0x7f800000u) : t == OMEGA_GEMM_DT_BF16 ? (neg ? 0xff80u : 0x7f80u) : (neg ? 0xfc00u : 0x7c00u);
}

/* ------------------------------------------------------------------ the cores */

/* the host model of the C3 core: the tensor-core steps of tests/gpu_gemm_test.c, f16 with the bf16 model (same HMMA
 * datapath, offline oracle only), the FFMA reference as one fmaf per k from +0 */
static int model_core(void *ctx, OmegaGpuGemmType t, unsigned lay, uint32_t M, uint32_t N, uint32_t K, const void *A, uint64_t lda,
                      const void *B, uint64_t ldb, float *C) {
    (void)ctx;
    const int acol = (lay & OMEGA_GPU_GEMM_A_COL) != 0, bcol = (lay & OMEGA_GPU_GEMM_B_COL) != 0;
    const uint64_t ea = lda ? lda : acol ? M : K, eb = ldb ? ldb : bcol ? K : N;
    const unsigned dt = t == OMEGA_GPU_GEMM_BF16 ? OMEGA_GEMM_DT_BF16 : t == OMEGA_GPU_GEMM_F16 ? OMEGA_GEMM_DT_F16
                      : t == OMEGA_GPU_GEMM_E4M3 ? OMEGA_GEMM_DT_E4M3 : OMEGA_GEMM_DT_F32;
    const uint32_t ks = t == OMEGA_GPU_GEMM_E4M3 ? 32u : 16u;
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++) {
            float acc = 0.0f;
            if (t == OMEGA_GPU_GEMM_F32_FFMA) {
                for (uint32_t k = 0; k < K; k++) acc = fmaf(tval(dt, A, (uint64_t)m * ea + k), tval(dt, B, (uint64_t)k * eb + n), acc);
            } else
                for (uint32_t k0 = 0; k0 < K; k0 += ks) {
                    float av[32], bv[32];
                    for (uint32_t k = 0; k < ks; k++) {
                        uint64_t kk = k0 + k;
                        if (kk >= K) { av[k] = bv[k] = 0.0f; continue; }
                        av[k] = tval(dt, A, acol ? kk * ea + m : (uint64_t)m * ea + kk);
                        bv[k] = tval(dt, B, bcol ? (uint64_t)n * eb + kk : kk * eb + n);
                    }
                    acc = ks == 16u ? hm_dot16(&HM, av, bv, acc) : qm_dot32(&QM, av, bv, acc);
                }
            C[(uint64_t)m * N + n] = acc;
        }
    return OMEGA_GPU_GEMM_OK;
}

/* the real kernels in the warp simulator */
typedef struct {
    BlackwellIRProgram *prog[4][OMEGA_GEMM_K_COUNT];
    int mutant[4];
    WsRange ranges[5];
    int nr;
    char err[300];
    uint64_t calls;
} SimCtx;

static int sim_launch(void *ctx, OmegaGemmKind k, const uint32_t a[OMEGA_GPU_GEMM_ARGS_WORDS], uint32_t gx, uint32_t gy, uint32_t threads) {
    SimCtx *c = ctx;
    static WsSim s;
    memset(&s, 0, sizeof s);
    s.p = c->prog[c->nr >> 8][k]; s.threads = threads;
    for (unsigned i = 0; i < OMEGA_GPU_GEMM_ARGS_WORDS; i++) s.cbank[0x380 / 4 + i] = a[i];
    memcpy(s.ranges, c->ranges, sizeof c->ranges); s.nranges = c->nr & 0xff;
    if (ws_run(&s, gx, gy) != 0) { snprintf(c->err, sizeof c->err, "kernel %d: %s at pc %d", (int)k, s.msg, s.err_pc); return -1; }
    return 0;
}
static void sim_free(SimCtx *c) {
    for (int t = 0; t < 4; t++) for (int k = 0; k < OMEGA_GEMM_K_COUNT; k++) { free(c->prog[t][k]); c->prog[t][k] = NULL; }
}
static int sim_core(void *ctx, OmegaGpuGemmType t, unsigned lay, uint32_t M, uint32_t N, uint32_t K, const void *A, uint64_t lda,
                    const void *B, uint64_t ldb, float *C) {
    SimCtx *c = ctx;
    OmegaGemmCfg cfg;
    omega_gpu_gemm_default_cfg(t, &cfg); cfg.mutant = c->mutant[t];
    const int ffma = t == OMEGA_GPU_GEMM_F32_FFMA;
    if (!c->prog[t][OMEGA_GEMM_K_MAIN])
        for (int k = ffma ? OMEGA_GEMM_K_MAIN : 0; k < OMEGA_GEMM_K_COUNT; k++) {
            c->prog[t][k] = calloc(1, sizeof *c->prog[t][k]);
            if (!c->prog[t][k] || omega_gpu_gemm_codegen(&cfg, (OmegaGemmKind)k, c->prog[t][k]) != 0) return OMEGA_GPU_GEMM_CODEGEN_FAIL;
        }
    OmegaGemmPlan p;
    uint64_t sa = 0, sb = 0;
    if (omega_gpu_gemm_plan(&cfg, M, N, K, &p) != 0 || omega_gpu_gemm_span(lay, M, N, K, lda, ldb, &sa, &sb) != 0) return OMEGA_GPU_GEMM_BAD_ARGS;
    const uint32_t es = t == OMEGA_GPU_GEMM_E4M3 ? 1u : ffma ? 4u : 2u;
    uint8_t *ap = malloc(p.ap_bytes + 1), *bp = malloc(p.bp_bytes + 1);
    memset(ap, 0xa5, p.ap_bytes); memset(bp, 0x5a, p.bp_bytes);
    for (uint64_t i = 0; i < (uint64_t)M * N; i++) C[i] = t_u2f(0x7fa00badu);
    c->ranges[0] = (WsRange){ (uintptr_t)A, (uintptr_t)A + sa * es + 1 };
    c->ranges[1] = (WsRange){ (uintptr_t)B, (uintptr_t)B + sb * es + 1 };
    c->ranges[2] = (WsRange){ (uintptr_t)C, (uintptr_t)(C + (uint64_t)M * N) };
    c->ranges[3] = (WsRange){ (uintptr_t)ap, (uintptr_t)ap + p.ap_bytes };
    c->ranges[4] = (WsRange){ (uintptr_t)bp, (uintptr_t)bp + p.bp_bytes };
    c->nr = (int)t << 8 | (ffma ? 3 : 5);
    int rc = omega_gpu_gemm_drive_ld(&cfg, lay, M, N, K, (uintptr_t)A, lda, (uintptr_t)B, ldb, (uintptr_t)ap, (uintptr_t)bp, (uintptr_t)C, sim_launch, c);
    for (uint64_t i = 0; rc == 0 && i < (uint64_t)M * N; i++) if (t_f2u(C[i]) == 0x7fa00badu) rc = OMEGA_GPU_GEMM_CHIP_FAIL;
    if (rc) printf("sim: %s %s\n", omega_gpu_gemm_rc_name(rc), c->err);
    c->calls++;
    free(ap); free(bp);
    return rc;
}

/* ------------------------------------------------------------------ cases */

typedef struct {
    unsigned at, bt, ct, opa, opb, compute, scale;
    uint32_t m, n, k, batch, bm, bn, bk;
    uint32_t pa, pb, pc;          /* leading-dimension padding */
    int bcast_a, bcast_b;         /* stride 0 (with batch > 1) */
    float alpha, beta;
    uint64_t seed;
    int edge;                     /* 0 random; 1 subnormal 2^-126 / 2^-110; 2 inf/NaN; 3 near FLT_MAX; 4 e4m3/e5m2 NaN, inf */
} Case;

typedef struct {
    Case cs;
    OmegaGemmExArgs x;
    void *A, *B, *C, *C0;
    float *sa, *sb;
    uint64_t ea, eb, ec, nsa, nsb;  /* elements */
} Run;

static uint64_t span(uint64_t rows, uint64_t cols, uint64_t ld) { return (rows - 1u) * ld + cols; }

static void run_free(Run *r) { free(r->A); free(r->B); free(r->C); free(r->C0); free(r->sa); free(r->sb); memset(r, 0, sizeof *r); }

/* builds the operands and the arguments of a case */
static int run_make(Run *r, const Case *cs) {
    memset(r, 0, sizeof *r);
    r->cs = *cs;
    OmegaGemmExArgs *x = &r->x;
    x->op_a = cs->opa; x->op_b = cs->opb; x->m = cs->m; x->n = cs->n; x->k = cs->k; x->batch = cs->batch;
    x->a_type = cs->at; x->b_type = cs->bt; x->c_type = cs->ct; x->alpha = cs->alpha; x->beta = cs->beta;
    x->compute = cs->compute; x->scale = cs->scale; x->block_m = cs->bm; x->block_n = cs->bn; x->block_k = cs->bk;
    const uint64_t ar = cs->opa ? cs->k : cs->m, ac = cs->opa ? cs->m : cs->k, br = cs->opb ? cs->n : cs->k, bc = cs->opb ? cs->k : cs->n;
    x->lda = ac + cs->pa; x->ldb = bc + cs->pb; x->ldc = cs->n + cs->pc;
    const uint64_t sa = span(ar, ac, x->lda), sb = span(br, bc, x->ldb), sc = span(cs->m, cs->n, x->ldc);
    g_rng = cs->seed * 0x9e3779b97f4a7c15ull + 1;
    x->stride_a = cs->batch > 1 && cs->bcast_a ? 0 : sa + rnd() % 5u;
    x->stride_b = cs->batch > 1 && cs->bcast_b ? 0 : sb + rnd() % 5u;
    x->stride_c = sc + rnd() % 5u;
    r->ea = (cs->batch - 1u) * x->stride_a + sa; r->eb = (cs->batch - 1u) * x->stride_b + sb; r->ec = (cs->batch - 1u) * x->stride_c + sc;
    r->A = malloc(r->ea * dsz(cs->at) + 8); r->B = malloc(r->eb * dsz(cs->bt) + 8);
    r->C = malloc(r->ec * dsz(cs->ct) + 8); r->C0 = malloc(r->ec * dsz(cs->ct) + 8);
    if (!r->A || !r->B || !r->C || !r->C0) return -1;
    for (uint64_t i = 0; i < r->ea; i++) tput(cs->at, r->A, i, rnd_bits(cs->at));
    for (uint64_t i = 0; i < r->eb; i++) tput(cs->bt, r->B, i, rnd_bits(cs->bt));
    for (uint64_t i = 0; i < r->ec; i++) {
        uint32_t v = cs->beta != 0.0f ? rnd_bits(cs->ct == OMEGA_GEMM_DT_F32 ? OMEGA_GEMM_DT_F32 : cs->ct) : nan_bits(cs->ct);
        tput(cs->ct, r->C, i, v);
    }
    /* the edges */
    if (cs->edge == 1) { /* values near 2^-126 and 2^-110 in A, ordinary B */
        for (uint64_t i = 0; i < r->ea; i++) {
            uint32_t e = rnd() & 1u ? 1u : 17u; /* biased exponent 1 (2^-126) or 17 (2^-110) */
            tput(OMEGA_GEMM_DT_F32, r->A, i, (rnd() & 1u) << 31 | e << 23 | (rnd() & 0x7fffffu));
        }
    } else if (cs->edge == 2) { /* a few inf and NaN in A and B */
        for (int q = 0; q < 3; q++) {
            tput(cs->at, r->A, rnd() % r->ea, q == 2 ? nan_bits(cs->at) | 1u : inf_bits(cs->at, q == 1));
            tput(cs->bt, r->B, rnd() % r->eb, inf_bits(cs->bt, 0));
        }
    } else if (cs->edge == 3) { /* A near FLT_MAX / k, B in [0.5, 1) */
        for (uint64_t i = 0; i < r->ea; i++) tput(OMEGA_GEMM_DT_F32, r->A, i, (rnd() & 1u) << 31 | (254u - 5u) << 23 | (rnd() & 0x7fffffu));
        for (uint64_t i = 0; i < r->eb; i++) tput(OMEGA_GEMM_DT_F32, r->B, i, (rnd() & 1u) << 31 | 126u << 23 | (rnd() & 0x7fffffu));
    } else if (cs->edge == 4) { /* e4m3 NaN codes, e5m2 inf and NaN */
        tput(cs->at, r->A, rnd() % r->ea, cs->at == OMEGA_GEMM_DT_E4M3 ? 0x7fu : 0x7cu);
        tput(cs->bt, r->B, rnd() % r->eb, cs->bt == OMEGA_GEMM_DT_E4M3 ? 0xffu : 0xfdu);
    }
    memcpy(r->C0, r->C, r->ec * dsz(cs->ct));
    if (cs->scale == OMEGA_GEMM_SCALE_TENSOR) { r->nsa = r->nsb = 1; }
    else if (cs->scale == OMEGA_GEMM_SCALE_BLOCK) {
        const uint64_t kb = (cs->k + cs->bk - 1u) / cs->bk;
        r->nsa = (cs->m + cs->bm - 1u) / cs->bm * kb; r->nsb = kb * ((cs->n + cs->bn - 1u) / cs->bn);
    }
    if (r->nsa) {
        x->scale_stride_a = cs->batch > 1 && !cs->bcast_a ? r->nsa : 0; x->scale_stride_b = cs->batch > 1 && !cs->bcast_b ? r->nsb : 0;
        uint64_t na = r->nsa * (x->scale_stride_a ? cs->batch : 1u), nb = r->nsb * (x->scale_stride_b ? cs->batch : 1u);
        r->sa = malloc(na * 4); r->sb = malloc(nb * 4);
        for (uint64_t i = 0; i < na; i++) r->sa[i] = ldexpf(1.0f + (float)(rnd() % 8u) / 8.0f, (int)(rnd() % 9u) - 4);
        for (uint64_t i = 0; i < nb; i++) r->sb[i] = ldexpf(1.0f + (float)(rnd() % 8u) / 8.0f, (int)(rnd() % 9u) - 4);
        x->scale_a = r->sa; x->scale_b = r->sb;
    }
    x->a = r->A; x->b = r->B; x->c = r->C;
    return 0;
}

static double f_of(const Run *r, uint32_t b, uint32_t i, uint32_t kk, uint32_t j) {
    const Case *cs = &r->cs;
    if (cs->scale != OMEGA_GEMM_SCALE_BLOCK) return 1.0;
    const uint64_t kb = (cs->k + cs->bk - 1u) / cs->bk, nb = (cs->n + cs->bn - 1u) / cs->bn;
    const float *sa = r->sa + b * r->x.scale_stride_a, *sb = r->sb + b * r->x.scale_stride_b;
    return (double)sa[(i / cs->bm) * kb + kk / cs->bk] * (double)sb[(kk / cs->bk) * nb + j / cs->bn];
}

static uint64_t steps(const Case *cs) {
    const uint64_t k = cs->k, parts = (k + 1807u) / 1808u + (k + 16383u) / 16384u + (cs->scale == OMEGA_GEMM_SCALE_BLOCK ? (k + cs->bk - 1u) / cs->bk : 0u);
    if (cs->compute == OMEGA_GEMM_EX_F32_FFMA_REF) return k + parts;
    const int f32ish = cs->at == OMEGA_GEMM_DT_F32 || cs->bt == OMEGA_GEMM_DT_F32 ||
                       (cs->at == OMEGA_GEMM_DT_BF16 && cs->bt == OMEGA_GEMM_DT_F16) || (cs->at == OMEGA_GEMM_DT_F16 && cs->bt == OMEGA_GEMM_DT_BF16);
    if (f32ish && cs->compute != OMEGA_GEMM_EX_F32_FAST_BF16) return 9u * ((k + 15u) / 16u + parts) + parts;
    if (cs->at == OMEGA_GEMM_DT_E4M3 && cs->bt == OMEGA_GEMM_DT_E4M3) return (k + 31u) / 32u + parts;
    return (k + 15u) / 16u + parts;
}
static int is_fast(const Case *cs) {
    return cs->compute == OMEGA_GEMM_EX_F32_FAST_BF16 &&
           (cs->at == OMEGA_GEMM_DT_F32 || cs->bt == OMEGA_GEMM_DT_F32 || (cs->at == OMEGA_GEMM_DT_BF16 && cs->bt == OMEGA_GEMM_DT_F16) ||
            (cs->at == OMEGA_GEMM_DT_F16 && cs->bt == OMEGA_GEMM_DT_BF16));
}

typedef struct { uint64_t checked, bad, untouched_bad; double max_rel; char first[300]; } Verdict;

static void one_elem(const Run *r, uint32_t b, uint32_t i, uint32_t j, Verdict *v) {
    const Case *cs = &r->cs;
    const OmegaGemmExArgs *x = &r->x;
    const uint8_t *A = (const uint8_t *)r->A + b * x->stride_a * dsz(cs->at), *B = (const uint8_t *)r->B + b * x->stride_b * dsz(cs->bt);
    double s = 0, sab = 0, sapb = 0;
    for (uint32_t kk = 0; kk < cs->k; kk++) {
        double a = tval(cs->at, A, cs->opa ? (uint64_t)kk * x->lda + i : (uint64_t)i * x->lda + kk);
        double bb = tval(cs->bt, B, cs->opb ? (uint64_t)j * x->ldb + kk : (uint64_t)kk * x->ldb + j);
        double f = f_of(r, b, i, kk, j);
        s += a * bb * f; sab += fabs(a * bb * f); sapb += (fabs(a) + fabs(bb)) * fabs(f);
    }
    double al = cs->alpha;
    if (cs->scale == OMEGA_GEMM_SCALE_TENSOR) al *= (double)r->sa[b * x->scale_stride_a] * (double)r->sb[b * x->scale_stride_b];
    const uint64_t ci = b * x->stride_c + (uint64_t)i * x->ldc + j;
    const double old = cs->beta != 0.0f ? tval(cs->ct, r->C0, ci) : 0.0;
    const double ref = (cs->alpha != 0.0f ? al * s : 0.0) + (double)cs->beta * old;
    const double got = tval(cs->ct, r->C, ci);
    const double uc = cs->ct == OMEGA_GEMM_DT_F32 ? 0x1p-24 : cs->ct == OMEGA_GEMM_DT_BF16 ? 0x1p-8 : 0x1p-11;
    const double tc = cs->ct == OMEGA_GEMM_DT_F32 ? 0x1p-149 : cs->ct == OMEGA_GEMM_DT_BF16 ? 0x1p-133 : 0x1p-24;
    const double maxc = cs->ct == OMEGA_GEMM_DT_F32 ? 3.4028234663852886e38 : cs->ct == OMEGA_GEMM_DT_BF16 ? 3.3895313892515355e38 : 65504.0;
    double bound = (cs->alpha != 0.0f ? fabs(al) * (((double)steps(cs) + 4.0) * 0x1p-22 * sab + 0x1p-124 * sapb + cs->k * 0x1p-140 +
                    (is_fast(cs) ? 0x1p-7 * sab : 0.0)) : 0.0) + 0x1p-50 * fabs((double)cs->beta * old) + uc * fabs(ref) + tc;
    int ok;
    if (isnan(ref)) ok = isnan(got);
    else if (isinf(ref)) ok = got == ref;
    else if (isinf(got)) ok = fabs(ref) + bound >= maxc && (got > 0) == (ref > 0);
    else ok = !isnan(got) && fabs(got - ref) <= bound;
    if (ok && isfinite(ref) && isfinite(got) && sab > 0 && cs->alpha != 0.0f) {
        double rel = fabs(got - ref) / (fabs(al) * sab * 0x1p-24);
        if (rel > v->max_rel) v->max_rel = rel;
    }
    v->checked++;
    if (!ok) {
        if (!v->bad) snprintf(v->first, sizeof v->first, "batch %u C[%u][%u] = %.9g, ref %.9g, bound %.3g", b, i, j, got, ref, bound);
        v->bad++;
    }
}

/* every element when m n k batch <= 2^22, else 512 samples per batch plus the four corners */
static Verdict verify(const Run *r) {
    Verdict v; memset(&v, 0, sizeof v);
    const Case *cs = &r->cs;
    const OmegaGemmExArgs *x = &r->x;
    const uint32_t es = dsz(cs->ct);
    /* untouched C: everything outside the m x n windows */
    uint8_t *mark = calloc(r->ec, 1);
    for (uint32_t b = 0; b < cs->batch; b++)
        for (uint32_t i = 0; i < cs->m; i++) memset(mark + b * x->stride_c + (uint64_t)i * x->ldc, 1, cs->n);
    for (uint64_t e = 0; e < r->ec; e++)
        if (!mark[e] && memcmp((uint8_t *)r->C + e * es, (uint8_t *)r->C0 + e * es, es)) v.untouched_bad++;
    free(mark);
    const int full = (uint64_t)cs->m * cs->n * cs->k * cs->batch <= (1ull << 22);
    uint64_t saved = g_rng;
    g_rng = cs->seed + 99;
    for (uint32_t b = 0; b < cs->batch; b++) {
        if (full) { for (uint32_t i = 0; i < cs->m; i++) for (uint32_t j = 0; j < cs->n; j++) one_elem(r, b, i, j, &v); continue; }
        for (int q = 0; q < 512; q++) one_elem(r, b, rnd() % cs->m, rnd() % cs->n, &v);
        one_elem(r, b, 0, 0, &v); one_elem(r, b, cs->m - 1, 0, &v); one_elem(r, b, 0, cs->n - 1, &v); one_elem(r, b, cs->m - 1, cs->n - 1, &v);
    }
    g_rng = saved;
    return v;
}

static void case_name(const Case *c, char *w, size_t n) {
    snprintf(w, n, "%s%s x %s%s -> %s %ux%ux%u batch %u%s%s pad %u/%u/%u alpha %g beta %g%s%s%s edge %d",
             DTN[c->at], c->opa ? "^T" : "", DTN[c->bt], c->opb ? "^T" : "", DTN[c->ct], c->m, c->n, c->k, c->batch,
             c->bcast_a ? " A-bcast" : "", c->bcast_b ? " B-bcast" : "", c->pa, c->pb, c->pc, (double)c->alpha, (double)c->beta,
             c->compute == 1 ? " FAST_BF16" : c->compute == 2 ? " FFMA_REF" : "", c->scale == 1 ? " tensor-scale" : "",
             c->scale == 2 ? " block-scale" : "", c->edge);
    if (c->scale == 2) { size_t l = strlen(w); snprintf(w + l, n - l, " blocks %u/%u/%u", c->bm, c->bn, c->bk); }
}

/* runs one case through gemm_ex and checks it; returns the wrong elements (or 1 on a refusal); *route_out the route */
static uint64_t run_case(const Case *cs, int quiet, unsigned *route_out, double *rel_out, uint64_t *calls_out) {
    char w[1400], nm[400];
    Run r;
    case_name(cs, nm, sizeof nm);
    if (run_make(&r, cs) != 0) { snprintf(w, sizeof w, "%s: host allocation", nm); check(0, w); run_free(&r); return 1; }
    OmegaGemmExInfo info; memset(&info, 0, sizeof info);
    r.x.info = &info;
    int rc = omega_gpu_gemm_ex(&r.x);
    Verdict v; memset(&v, 0, sizeof v);
    if (rc == 0) v = verify(&r);
    snprintf(w, sizeof w, "%s: %s route %s core calls %" PRIu64 " host-fixed %" PRIu64 ", %" PRIu64 " checked, %" PRIu64 " wrong, %" PRIu64
             " untouched changed, max err %.2f x 2^-24 sum|ab|%s%s%s%s",
             nm, omega_gpu_gemm_rc_name(rc), RTN[info.route < 6 ? info.route : 0], info.core_calls, info.host_fixed, v.checked, v.bad,
             v.untouched_bad, v.max_rel, v.bad ? "; first " : "", v.bad ? v.first : "", rc ? ": " : "", rc ? omega_gpu_session_last_error() : "");
    if (!quiet) { check(rc == 0 && v.bad == 0 && v.untouched_bad == 0, w); if (rc == 0 && !v.bad && !v.untouched_bad) printf("ok: %s\n", w); }
    else printf("%s\n", w);
    if (route_out) *route_out = info.route;
    if (rel_out) *rel_out = v.max_rel;
    if (calls_out) *calls_out = info.core_calls;
    run_free(&r);
    return rc ? 1 : v.bad + v.untouched_bad;
}

/* the same case through two cores (the chip or the simulator, then the host model): bit for bit */
static void same_as_model(const Case *cs, OmegaGemmExCoreFn fn, void *ctx, const char *who) {
    char w[900], nm[400];
    Run r1, r2;
    case_name(cs, nm, sizeof nm);
    if (run_make(&r1, cs) || run_make(&r2, cs)) { check(0, "host allocation"); run_free(&r1); run_free(&r2); return; }
    omega_gpu_gemm_ex_set_backend(fn, ctx);
    int rc1 = omega_gpu_gemm_ex(&r1.x);
    omega_gpu_gemm_ex_set_backend(model_core, NULL);
    int rc2 = omega_gpu_gemm_ex(&r2.x);
    omega_gpu_gemm_ex_set_backend(fn, ctx);
    uint64_t diff = 0, first = UINT64_MAX;
    const uint32_t es = dsz(cs->ct);
    for (uint64_t e = 0; e < r1.ec; e++) {
        uint32_t u1 = 0, u2 = 0;
        memcpy(&u1, (uint8_t *)r1.C + e * es, es); memcpy(&u2, (uint8_t *)r2.C + e * es, es);
        if (u1 != u2 && !(isnan(tval(cs->ct, r1.C, e)) && isnan(tval(cs->ct, r2.C, e)))) { if (!diff) first = e; diff++; }
    }
    snprintf(w, sizeof w, "%s core == host-model core, bit for bit: %s: %s / %s, %" PRIu64 " differ (first element %" PRIu64 ")", who, nm,
             omega_gpu_gemm_rc_name(rc1), omega_gpu_gemm_rc_name(rc2), diff, first == UINT64_MAX ? 0 : first);
    check(rc1 == 0 && rc2 == 0 && diff == 0, w);
    if (rc1 == 0 && rc2 == 0 && diff == 0) printf("ok: %s\n", w);
    run_free(&r1); run_free(&r2);
}

/* ------------------------------------------------------------------ the predeclared sweep */

static const uint32_t DIMS[] = { 1, 2, 3, 5, 7, 8, 13, 15, 16, 17, 31, 32, 33, 63, 64, 65, 97, 127, 128, 129, 251, 255, 256, 257, 509, 511, 513, 1000, 1021, 1023, 1025 };
#define NDIMS (sizeof DIMS / sizeof DIMS[0])
#define SWEEP_SEED 20261027u
#define SWEEP_N 600u

static uint32_t pick_dim(uint32_t cap) { uint32_t d; do d = DIMS[rnd() % NDIMS]; while (d > cap); return d; }

/* case i of the sweep: 13 route classes in turn, the dimensions, ops, padding, batch, alpha/beta, output from the seed */
static Case sweep_case(uint32_t i) {
    Case c; memset(&c, 0, sizeof c);
    g_rng = (uint64_t)SWEEP_SEED * 1000003u + i;
    rnd();
    static const unsigned cls[13][4] = { /* a, b, compute, scale */
        { 1, 1, 0, 0 }, { 2, 2, 0, 0 }, { 3, 3, 0, 0 }, { 4, 4, 0, 0 }, { 1, 3, 0, 0 }, { 2, 4, 0, 0 }, { 0, 0, 0, 0 },
        { 0, 0, 1, 0 }, { 0, 1, 0, 0 }, { 1, 2, 0, 0 }, { 0, 0, 2, 0 }, { 3, 3, 0, 2 }, { 3, 4, 0, 1 } };
    const unsigned k3 = i % 13u;
    c.at = cls[k3][0]; c.bt = cls[k3][1]; c.compute = cls[k3][2]; c.scale = cls[k3][3];
    const int x3 = k3 >= 6 && k3 <= 9; /* bf16x3 / fast: the host splits every element */
    c.m = pick_dim(1025); c.n = pick_dim(1025); c.k = pick_dim(1025);
    const uint32_t sp = rnd() % 20u;
    if (sp == 0) { c.k = x3 ? (uint32_t[]){ 1807, 1808, 1809, 3617 }[rnd() % 4u] : (uint32_t[]){ 16383, 16384, 16385, 20000 }[rnd() % 4u]; c.m = pick_dim(65); c.n = pick_dim(65); }
    else if (sp == 1) { if (rnd() & 1u) c.m = (uint32_t[]){ 4095, 4096, 4097, 5000 }[rnd() % 4u]; else c.n = (uint32_t[]){ 4095, 4096, 4097, 5000 }[rnd() % 4u];
                        if (c.m > 1025) c.n = pick_dim(33); else c.m = pick_dim(33); c.k = pick_dim(257); }
    else if (sp == 2) { c.m = c.n = c.k = 1; }
    c.opa = rnd() & 1u; c.opb = rnd() & 1u;
    c.pa = rnd() % 3u ? 0 : 1 + rnd() % 7u; c.pb = rnd() % 3u ? 0 : 1 + rnd() % 7u; c.pc = rnd() % 3u ? 0 : 1 + rnd() % 7u;
    c.batch = rnd() % 4u ? 1u : 2u + rnd() % 2u;
    if ((uint64_t)c.m * c.n * c.k * c.batch > (1ull << 30)) c.batch = 1;
    c.bcast_a = c.batch > 1 && rnd() % 4u == 0; c.bcast_b = c.batch > 1 && rnd() % 4u == 0;
    static const float AL[4] = { 1.0f, -0.5f, 2.75f, 1.0f }, BE[4] = { 0.0f, 1.0f, -1.5f, 0.0f };
    c.alpha = AL[rnd() % 4u]; c.beta = BE[rnd() % 4u];
    const uint32_t co = rnd() % 5u;
    c.ct = co < 3u ? OMEGA_GEMM_DT_F32 : co == 3u ? OMEGA_GEMM_DT_BF16 : OMEGA_GEMM_DT_F16;
    if (c.ct == OMEGA_GEMM_DT_F16) c.alpha = ldexpf(c.alpha, -18 - (int)ceilf(log2f((float)c.k)));
    if (c.scale == OMEGA_GEMM_SCALE_BLOCK) {
        static const uint32_t BS[5] = { 1, 16, 100, 128, 7 };
        c.bm = BS[rnd() % 5u]; c.bn = BS[rnd() % 5u]; c.bk = (uint32_t[]){ 16, 32, 100, 128 }[rnd() % 4u];
        if (c.ct == OMEGA_GEMM_DT_F16) c.alpha = ldexpf(c.alpha, -8);
    }
    c.seed = (uint64_t)SWEEP_SEED * 7919u + i;
    return c;
}

static void list_sweep(void) {
    char nm[400];
    printf("gemm_ex predeclared sweep: seed %u, %u cases; check per element as stated in tests/gpu_gemm_ex_test.c lines 14-21\n", SWEEP_SEED, SWEEP_N);
    for (uint32_t i = 0; i < SWEEP_N; i++) { Case c = sweep_case(i); case_name(&c, nm, sizeof nm); printf("%u %s\n", i, nm); }
}

/* the edge cases and fixed cases of the host tiers */
static Case mk(unsigned at, unsigned bt, unsigned ct, uint32_t m, uint32_t n, uint32_t k) {
    Case c; memset(&c, 0, sizeof c);
    c.at = at; c.bt = bt; c.ct = ct; c.m = m; c.n = n; c.k = k; c.batch = 1; c.alpha = 1.0f; c.seed = (uint64_t)m * 131 + n * 17 + k * 3 + at * 5 + bt;
    return c;
}
static void edge_cases(Case *e, int *ne) {
    int n = 0;
    Case c;
    c = mk(0, 0, 0, 33, 17, 70); c.edge = 1; e[n++] = c;                          /* bf16x3: A at 2^-126 and 2^-110 */
    c = mk(0, 0, 0, 33, 17, 70); c.edge = 1; c.compute = 2; e[n++] = c;           /* the same through the FFMA reference */
    c = mk(0, 0, 0, 20, 19, 40); c.edge = 2; c.beta = 1.0f; e[n++] = c;           /* inf and NaN propagate */
    c = mk(1, 1, 0, 20, 19, 40); c.edge = 2; c.opa = 1; e[n++] = c;               /* the same, bf16 native */
    c = mk(2, 2, 2, 20, 19, 40); c.edge = 2; c.alpha = 0x1p-12f; e[n++] = c;      /* f16 native */
    c = mk(0, 0, 0, 16, 16, 24); c.edge = 3; e[n++] = c;                          /* near FLT_MAX: bf16x3 */
    c = mk(0, 0, 0, 16, 16, 24); c.edge = 3; c.compute = 1; e[n++] = c;           /* and FAST_BF16 */
    c = mk(3, 3, 0, 18, 21, 70); c.edge = 4; e[n++] = c;                          /* e4m3 NaN codes */
    c = mk(4, 4, 0, 18, 21, 70); c.edge = 4; e[n++] = c;                          /* e5m2 inf and NaN */
    c = mk(1, 1, 0, 9, 9, 9); c.alpha = 0.0f; c.beta = 2.0f; e[n++] = c;          /* alpha 0: C = beta C */
    c = mk(1, 1, 1, 9, 9, 9); c.beta = 0.0f; e[n++] = c;                          /* beta 0 with NaN in C: never read */
    c = mk(1, 1, 0, 7, 5, 3); c.batch = 3; c.bcast_a = 1; c.pc = 2; e[n++] = c;   /* broadcast A over a batch */
    c = mk(3, 3, 0, 40, 30, 100); c.scale = 2; c.bm = 7; c.bn = 16; c.bk = 32; e[n++] = c;
    c = mk(3, 4, 1, 40, 30, 100); c.scale = 1; c.batch = 2; e[n++] = c;
    c = mk(1, 1, 0, 3, 5, 16385); e[n++] = c;                                     /* k past one core call */
    c = mk(0, 0, 0, 3, 5, 1809); e[n++] = c;                                      /* bf16x3 k past one part */
    c = mk(2, 2, 0, 33, 31, 47); c.opa = 1; c.opb = 1; c.pa = 3; c.pb = 5; c.pc = 1; e[n++] = c;
    *ne = n;
}

/* the four planted mutants of the general path; each must make a case go red */
static void ex_mutants(const char *tier) {
    char w[300];
    Case cs[5];
    cs[1] = mk(1, 1, 0, 24, 24, 24);                                              /* 1: the last k element dropped */
    cs[2] = mk(1, 1, 0, 24, 24, 24); cs[2].opb = 1;                               /* 2: op(B) read as N */
    cs[3] = mk(3, 3, 0, 24, 24, 96); cs[3].scale = 2; cs[3].bm = 8; cs[3].bn = 8; cs[3].bk = 32; /* 3: the next k block's scale */
    cs[4] = mk(1, 1, 0, 24, 24, 24); cs[4].beta = 1.0f;                           /* 4: beta ignored */
    for (int mu = 1; mu <= 4; mu++) {
        uint64_t clean = run_case(&cs[mu], 1, NULL, NULL, NULL);
        omega_gpu_gemm_ex_test_set_mutant(mu);
        uint64_t bad = run_case(&cs[mu], 1, NULL, NULL, NULL);
        omega_gpu_gemm_ex_test_set_mutant(0);
        snprintf(w, sizeof w, "%s red check: gemm_ex mutant %d is caught (%" PRIu64 " wrong; %" PRIu64 " without the mutant)", tier, mu, bad, clean);
        check(bad > 0 && clean == 0, w);
        printf("%s\n", w);
    }
}

/* ------------------------------------------------------------------ the fp8 grouped/MoE fallback */

/* Block-scaled fp8 weights (n x k e4m3, NaN codes excluded) with bf16-valued scales (exact in the dequant), x f32 or bf16.
 * Check of y[r][j] (predeclared): the double reference ref = sum_c x e4m3(w) s, and
 *   |got - ref| <= (9 ceil(k / 16) + 8) 2^-22 sum|x e4m3(w) s| + 2^-24 |ref| + k 2^-140
 * (the bf16x3 bound of the header with the k parts folded into the + 8). */
typedef struct { uint32_t n, k, block, m; } Fp8Shape;
static uint8_t fp8_code(void) { uint8_t b; do b = (uint8_t)rnd(); while (omega_fp8_e4m3_is_nan(b)); return b; }
static float fp8_scale(void) { return ldexpf(1.0f + (float)(rnd() % 8u) / 8.0f, -(int)(rnd() % 4u)); } /* 4 significant bits */
static uint16_t bf16_bits(float f) { return (uint16_t)(t_f2u(f) >> 16); } /* exact for the values above */

/* wrong elements of y (m x n) against the reference; x row r = xrow[r] of x (rows of k) */
static uint64_t fp8_verify(const uint8_t *w, const float *s, double s_mul, uint32_t n, uint32_t k, uint32_t block, const float *x32,
                           const uint16_t *x16, const uint32_t *xrow, const uint32_t *wexp, uint32_t m, const float *y, double *worst) {
    const uint32_t kb = (k + block - 1u) / block, nb = (n + block - 1u) / block;
    uint64_t bad = 0;
    for (uint32_t r = 0; r < m; r++) {
        const uint32_t xr = xrow ? xrow[r] : r, e = wexp ? wexp[r] : 0u;
        for (uint32_t j = 0; j < n; j++) {
            double ref = 0, mag = 0;
            for (uint32_t c = 0; c < k; c++) {
                const double xv = x32 ? (double)x32[(size_t)xr * k + c] : (double)t_u2f((uint32_t)x16[(size_t)xr * k + c] << 16);
                const double t = xv * (double)omega_fp8_e4m3_ref(w[((size_t)e * n + j) * k + c]) *
                                 (double)s[(size_t)e * nb * kb + (size_t)(j / block) * kb + c / block] * s_mul;
                ref += t; mag += fabs(t);
            }
            const double got = (double)y[(size_t)r * n + j];
            const double bound = (9.0 * ((k + 15u) / 16u) + 8.0) * 0x1p-22 * mag + 0x1p-24 * fabs(ref) + k * 0x1p-140;
            const double err = fabs(got - ref);
            if (!(err <= bound)) bad++;
            if (mag > 0 && worst && err / (mag * 0x1p-24) > *worst) *worst = err / (mag * 0x1p-24);
        }
    }
    return bad;
}

static void fp8_fill(uint8_t *w, float *s, uint16_t *s16, size_t nw, size_t ns, float *x32, uint16_t *x16, size_t nx) {
    for (size_t i = 0; i < nw; i++) w[i] = fp8_code();
    for (size_t i = 0; i < ns; i++) { s[i] = fp8_scale(); if (s16) s16[i] = bf16_bits(s[i]); }
    for (size_t i = 0; i < nx; i++) {
        const float v = (float)((int)(rnd() % 2001u) - 1000) / 256.0f;
        if (x32) x32[i] = v; else x16[i] = bf16_bits(v);
    }
}

/* the raw entry omega_gpu_gemm_ex_fp8_block_rows on the installed core: refused-before shapes, row independence, mutant 5 */
static void fp8_rows_tier(const char *tier) {
    static const Fp8Shape sh[] = {
        { 130, 200, 128, 3 }, { 128, 128, 128, 1 }, { 64, 384, 128, 2 }, { 33, 640, 128, 2 }, { 17, 1280, 128, 2 },
        { 9, 1792, 128, 1 }, { 8, 2304, 128, 2 }, { 5, 2560, 128, 1 }, { 4, 3584, 128, 2 }, { 2, 16512, 128, 1 },
        { 3, 3, 1, 2 }, { 20, 50, 7, 3 }, { 1, 1, 128, 1 } };
    char w[300];
    for (size_t i = 0; i < sizeof sh / sizeof sh[0]; i++) {
        const Fp8Shape c = sh[i];
        const int use16 = (int)(i & 1u);
        g_rng = 0x5eed0000u + i;
        const size_t nb = (c.n + c.block - 1u) / c.block, kb = (c.k + c.block - 1u) / c.block;
        uint8_t *wt = malloc((size_t)c.n * c.k); float *s = malloc(nb * kb * 4u);
        float *x32 = use16 ? NULL : malloc((size_t)c.m * c.k * 4u); uint16_t *x16 = use16 ? malloc((size_t)c.m * c.k * 2u) : NULL;
        float *y = malloc((size_t)c.m * c.n * 4u), *y1 = malloc((size_t)c.n * 4u);
        fp8_fill(wt, s, NULL, (size_t)c.n * c.k, nb * kb, x32, x16, (size_t)c.m * c.k);
        const double s_mul = 0x1p-3;
        OmegaGemmExInfo info;
        int rc = omega_gpu_gemm_ex_fp8_block_rows(wt, s, s_mul, c.n, c.k, c.block, x32, x16, c.m, y, 0, &info);
        double worst = 0;
        uint64_t bad = rc ? 1u : fp8_verify(wt, s, s_mul, c.n, c.k, c.block, x32, x16, NULL, NULL, c.m, y, &worst);
        uint32_t same = 0;
        for (uint32_t r = 0; r < c.m && rc == 0; r++) {
            int r1 = omega_gpu_gemm_ex_fp8_block_rows(wt, s, s_mul, c.n, c.k, c.block, x32 ? x32 + (size_t)r * c.k : NULL,
                                                      x16 ? x16 + (size_t)r * c.k : NULL, 1, y1, 0, NULL);
            if (r1 == 0 && !memcmp(y1, y + (size_t)r * c.n, (size_t)c.n * 4u)) same++;
        }
        snprintf(w, sizeof w, "%s fp8 block rows n=%u k=%u block=%u m=%u x %s: rc %s, %" PRIu64 " wrong, worst %.2f x 2^-24 sum|t|, %u/%u rows equal to the m=1 call, route %s",
                 tier, c.n, c.k, c.block, c.m, use16 ? "bf16" : "f32", omega_gpu_gemm_rc_name(rc), bad, worst, same, c.m, RTN[info.route < 6 ? info.route : 0]);
        check(rc == 0 && bad == 0 && same == c.m && info.route == OMEGA_GEMM_ROUTE_BF16X3, w);
        printf("%s\n", w);
        if (i == 0) { /* mutant 5: the next k block's scale; k = 200 has 2 blocks */
            omega_gpu_gemm_ex_test_set_mutant(5);
            rc = omega_gpu_gemm_ex_fp8_block_rows(wt, s, s_mul, c.n, c.k, c.block, x32, x16, c.m, y, 0, NULL);
            omega_gpu_gemm_ex_test_set_mutant(0);
            bad = rc ? 1u : fp8_verify(wt, s, s_mul, c.n, c.k, c.block, x32, x16, NULL, NULL, c.m, y, NULL);
            snprintf(w, sizeof w, "%s red check: gemm_ex mutant 5 (fp8 block rows, next k block's scale) is caught (%" PRIu64 " wrong)", tier, bad);
            check(bad > 0, w);
            printf("%s\n", w);
        }
        free(wt); free(s); free(x32); free(x16); free(y); free(y1);
    }
    float yy[4]; uint8_t w1 = 0x38; float s1 = 1.0f, x1 = 1.0f;
    check(omega_gpu_gemm_ex_fp8_block_rows(&w1, &s1, 1.0, 1, 1, 128, &x1, (const uint16_t *)&x1, 1, yy, 0, NULL) == OMEGA_GPU_GEMM_BAD_ARGS, "refusal: fp8 block rows with both x32 and x16");
    check(omega_gpu_gemm_ex_fp8_block_rows(&w1, &s1, 1.0, 1, 1, 0, &x1, NULL, 1, yy, 0, NULL) == OMEGA_GPU_GEMM_BAD_ARGS, "refusal: fp8 block rows with block 0");
    check(omega_gpu_gemm_ex_fp8_block_rows(&w1, &s1, 1.0, 2, 1, 128, &x1, NULL, 1, yy, 1, NULL) == OMEGA_GPU_GEMM_BAD_ARGS, "refusal: fp8 block rows with ldy 1 < n 2");
}

/* the real fp8 APIs on the chip: shapes the kernels refuse now answer through the fallback (info.general 1) within the
 * bound; a served-domain shape stays on the kernel (info.general 0, grouped GEMM == GEMV bit for bit as before) */
static void fp8_api_chip(void) {
    char w[300];
    static const struct { uint32_t experts, n, k; } T[] = { { 3, 256, 128 }, { 2, 128, 384 }, { 2, 256, 2048 } };
    for (size_t ti = 0; ti < sizeof T / sizeof T[0]; ti++) {
        const uint32_t E = T[ti].experts, n = T[ti].n, k = T[ti].k, nb = n / 128u, kb = k / 128u;
        const int served = k == 2048u;
        g_rng = 0xf8a90000u + ti;
        const uint32_t G = served ? 8u : 65u, M = served ? 8u : 1030u, XR = 16u;
        uint8_t *wt = malloc((size_t)E * n * k); float *s = malloc((size_t)E * nb * kb * 4u); uint16_t *s16 = malloc((size_t)E * nb * kb * 2u);
        float *x = malloc((size_t)XR * k * 4u), *yg = malloc((size_t)M * n * 4u), *yv = malloc((size_t)n * 4u);
        uint32_t *ids = malloc(M * 4u), *trow = malloc(M * 4u);
        fp8_fill(wt, s, s16, (size_t)E * n * k, (size_t)E * nb * kb, x, NULL, (size_t)XR * k);
        OmegaGpuFp8Tensor *t = NULL;
        int rc = omega_gpu_fp8_tensor_upload(E, n, k, wt, s16, &t);
        snprintf(w, sizeof w, "chip fp8 upload experts=%u n=%u k=%u: %s", E, n, k, omega_gpu_fp8_rc_name(rc));
        check(rc == OMEGA_GPU_FP8_OK, w);
        if (rc != OMEGA_GPU_FP8_OK) { free(wt); free(s); free(s16); free(x); free(yg); free(yv); free(ids); free(trow); continue; }
        /* GEMV, one expert */
        OmegaGpuFp8Info vi;
        rc = omega_gpu_fp8_gemv_f32(t, E - 1u, x, yg, &vi);
        uint32_t one = E - 1u;
        uint64_t bad = rc ? 1u : fp8_verify(wt, s, 1.0, n, k, 128, x, NULL, NULL, &one, 1, yg, NULL);
        snprintf(w, sizeof w, "chip fp8 gemv n=%u k=%u: rc %s, general %u (want %u), %" PRIu64 " wrong", n, k, omega_gpu_fp8_rc_name(rc), vi.general, served ? 0u : 1u, bad);
        check(rc == 0 && bad == 0 && vi.general == (served ? 0u : 1u), w);
        printf("%s\n", w);
        /* grouped GEMV: 65 groups (past the table of 64) or 8, one x row per group */
        for (uint32_t gi = 0; gi < G; gi++) { ids[gi] = rnd() % E; trow[gi] = gi % XR; }
        float *xg = malloc((size_t)G * k * 4u);
        for (uint32_t gi = 0; gi < G; gi++) memcpy(xg + (size_t)gi * k, x + (size_t)trow[gi] * k, (size_t)k * 4u);
        rc = omega_gpu_fp8_gemv_grouped_f32(t, G, ids, xg, G, yg, &vi);
        bad = rc ? 1u : fp8_verify(wt, s, 1.0, n, k, 128, xg, NULL, NULL, ids, G, yg, NULL);
        snprintf(w, sizeof w, "chip fp8 grouped gemv groups=%u n=%u k=%u: rc %s, general %u (want %u), %" PRIu64 " wrong", G, n, k, omega_gpu_fp8_rc_name(rc), vi.general, served ? 0u : 1u, bad);
        check(rc == 0 && bad == 0 && vi.general == (served ? 0u : 1u), w);
        printf("%s\n", w);
        free(xg);
        /* grouped GEMM: 1030 rows (past 1024) or 8, ids sorted */
        for (uint32_t i = 0; i < M; i++) { ids[i] = (uint32_t)((uint64_t)i * E / M); trow[i] = rnd() % XR; }
        OmegaGpuFp8GemmInfo gi;
        rc = omega_gpu_fp8gemm_f32(t, M, ids, trow, x, XR, yg, &gi);
        bad = rc ? 1u : fp8_verify(wt, s, 1.0, n, k, 128, x, NULL, trow, ids, M, yg, NULL);
        uint32_t same = 0;
        if (rc == 0 && served) /* the kernel's promise: bit-identical to the GEMV once per row */
            for (uint32_t i = 0; i < M; i++)
                if (omega_gpu_fp8_gemv_f32(t, ids[i], x + (size_t)trow[i] * k, yv, NULL) == 0 && !memcmp(yv, yg + (size_t)i * n, (size_t)n * 4u)) same++;
        snprintf(w, sizeof w, "chip fp8 grouped gemm M=%u n=%u k=%u: rc %s, general %u (want %u), %" PRIu64 " wrong%s", M, n, k, omega_gpu_fp8gemm_rc_name(rc), gi.general, served ? 0u : 1u, bad,
                 served ? (same == M ? ", every row == the GEMV" : ", rows differ from the GEMV") : "");
        check(rc == 0 && bad == 0 && gi.general == (served ? 0u : 1u) && (!served || same == M), w);
        printf("%s\n", w);
        if (!served && ti == 1) { /* mutant 5 through the API path (k = 384: three k blocks) */
            omega_gpu_gemm_ex_test_set_mutant(5);
            rc = omega_gpu_fp8gemm_f32(t, M, ids, trow, x, XR, yg, &gi);
            omega_gpu_gemm_ex_test_set_mutant(0);
            bad = rc ? 0u : fp8_verify(wt, s, 1.0, n, k, 128, x, NULL, trow, ids, M, yg, NULL);
            snprintf(w, sizeof w, "chip red check: gemm_ex mutant 5 through omega_gpu_fp8gemm_f32 (k=%u) is caught (rc %s, %" PRIu64 " wrong)", k, omega_gpu_fp8gemm_rc_name(rc), bad);
            check(rc == 0 && bad > 0, w);
            printf("%s\n", w);
        }
        omega_gpu_fp8_tensor_free(t);
        free(wt); free(s); free(s16); free(x); free(yg); free(yv); free(ids); free(trow);
    }
}

/* ------------------------------------------------------------------ tiers */

static int refuses(OmegaGemmExArgs x, int want, const char *what) {
    char w[200];
    int rc = omega_gpu_gemm_ex(&x);
    snprintf(w, sizeof w, "refusal: %s -> %s (want %s)", what, omega_gpu_gemm_rc_name(rc), omega_gpu_gemm_rc_name(want));
    check(rc == want, w);
    return rc == want;
}

static void refusals(void) {
    static float a[64], b[64], c[64];
    static float nanv = NAN;
    OmegaGemmExArgs x; memset(&x, 0, sizeof x);
    x.m = x.n = x.k = 4; x.a = a; x.b = b; x.c = c; x.alpha = 1.0f;
    x.a_type = x.b_type = x.c_type = OMEGA_GEMM_DT_BF16;
    check(omega_gpu_gemm_ex(NULL) == OMEGA_GPU_GEMM_BAD_ARGS, "refusal: NULL args");
    OmegaGemmExArgs y;
    y = x; y.c = NULL; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "C NULL");
    y = x; y.a = NULL; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "A NULL with alpha 1");
    y = x; y.m = 0; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "m 0");
    y = x; y.k = 0; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "k 0");
    y = x; y.op_a = 2; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "op_a 2");
    y = x; y.a_type = 5; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "a_type 5");
    y = x; y.c_type = 5; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "c_type 5");
    y = x; y.compute = 3; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "compute 3");
    y = x; y.scale = 3; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "scale 3");
    y = x; y.lda = 3; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "lda 3 < k 4");
    y = x; y.op_b = 1; y.ldb = 3; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "op_b T, ldb 3 < k 4");
    y = x; y.ldc = 2; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "ldc 2 < n 4");
    y = x; y.batch = 2; y.stride_a = y.stride_b = 0; y.stride_c = 15; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "C batches overlap (stride_c 15 < 16)");
    y = x; y.c = a; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "C is A");
    y = x; y.c = (uint8_t *)b + 6; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "C overlaps B");
    y = x; y.lda = 1ull << 60; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "lda 2^60");
    y = x; y.batch = 3; y.stride_c = 1ull << 62; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "stride_c 2^62");
    y = x; y.a_type = y.b_type = OMEGA_GEMM_DT_E4M3; y.scale = OMEGA_GEMM_SCALE_TENSOR; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "tensor scale without scales");
    y = x; y.scale = OMEGA_GEMM_SCALE_TENSOR; y.scale_a = &nanv; y.scale_b = b; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "NaN tensor scale");
    y = x; y.scale = OMEGA_GEMM_SCALE_BLOCK; y.scale_a = b; y.scale_b = b; y.block_m = y.block_n = 1; refuses(y, OMEGA_GPU_GEMM_BAD_ARGS, "block_k 0");
    y = x; y.m = OMEGA_GPU_GEMM_EX_MAX_DIM + 1u; refuses(y, OMEGA_GPU_GEMM_TOO_LARGE, "m 2^31");
    y = x; y.batch = OMEGA_GPU_GEMM_EX_MAX_BATCH + 1u; refuses(y, OMEGA_GPU_GEMM_TOO_LARGE, "batch 2^31");
    y = x; y.c_type = OMEGA_GEMM_DT_E4M3; refuses(y, OMEGA_GPU_GEMM_UNSUPPORTED, "e4m3 output");
    y = x; y.c_type = OMEGA_GEMM_DT_E5M2; refuses(y, OMEGA_GPU_GEMM_UNSUPPORTED, "e5m2 output");
    /* alpha 0: A and B may be NULL (not read) */
    y = x; y.alpha = 0.0f; y.a = y.b = NULL; y.beta = 0.0f;
    check(omega_gpu_gemm_ex(&y) == OMEGA_GPU_GEMM_OK, "alpha 0 with A, B NULL: accepted (A and B not read)");
}

static void route_table(void) {
    /* the route of every input pair (contract table of omega_gpu_gemm_ex.h) through the model core, 3x3x3 */
    static const unsigned want[5][5] = { /* compute DEFAULT */
        { OMEGA_GEMM_ROUTE_BF16X3, OMEGA_GEMM_ROUTE_BF16X3, OMEGA_GEMM_ROUTE_BF16X3, OMEGA_GEMM_ROUTE_BF16X3, OMEGA_GEMM_ROUTE_BF16X3 },
        { OMEGA_GEMM_ROUTE_BF16X3, OMEGA_GEMM_ROUTE_NATIVE, OMEGA_GEMM_ROUTE_BF16X3, OMEGA_GEMM_ROUTE_WIDEN, OMEGA_GEMM_ROUTE_WIDEN },
        { OMEGA_GEMM_ROUTE_BF16X3, OMEGA_GEMM_ROUTE_BF16X3, OMEGA_GEMM_ROUTE_NATIVE, OMEGA_GEMM_ROUTE_WIDEN, OMEGA_GEMM_ROUTE_WIDEN },
        { OMEGA_GEMM_ROUTE_BF16X3, OMEGA_GEMM_ROUTE_WIDEN, OMEGA_GEMM_ROUTE_WIDEN, OMEGA_GEMM_ROUTE_NATIVE, OMEGA_GEMM_ROUTE_WIDEN },
        { OMEGA_GEMM_ROUTE_BF16X3, OMEGA_GEMM_ROUTE_WIDEN, OMEGA_GEMM_ROUTE_WIDEN, OMEGA_GEMM_ROUTE_WIDEN, OMEGA_GEMM_ROUTE_WIDEN } };
    char w[200];
    for (unsigned a = 0; a < 5; a++)
        for (unsigned b = 0; b < 5; b++) {
            Case c = mk(a, b, 0, 3, 3, 3);
            unsigned route = 0;
            uint64_t bad = run_case(&c, 1, &route, NULL, NULL);
            snprintf(w, sizeof w, "route %s x %s: %s (want %s), %" PRIu64 " wrong", DTN[a], DTN[b], RTN[route], RTN[want[a][b]], bad);
            check(route == want[a][b] && bad == 0, w);
        }
}

static void host_tier(void) {
    char w[200];
    omega_gpu_gemm_ex_set_backend(model_core, NULL);
    refusals();
    route_table();
    Case e[32]; int ne = 0;
    edge_cases(e, &ne);
    for (int i = 0; i < ne; i++) run_case(&e[i], 0, NULL, NULL, NULL);
    /* every 12th sweep case with small dimensions through the model core */
    int ran = 0;
    for (uint32_t i = 0; i < SWEEP_N; i++) {
        Case c = sweep_case(i);
        if ((uint64_t)c.m * c.n * c.k * c.batch > (1ull << 21)) continue;
        if (i % 3u) continue;
        run_case(&c, 0, NULL, NULL, NULL); ran++;
    }
    snprintf(w, sizeof w, "host model: %d small sweep cases", ran);
    check(ran >= 40, w);
    printf("%s\n", w);
    ex_mutants("host");
    fp8_rows_tier("host");
}

static void sim_tier(void) {
    char w[300];
    SimCtx sc; memset(&sc, 0, sizeof sc);
    static const Case base[] = {
        { 1, 1, 0, 0, 0, 0, 0, 17, 9, 33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 11, 0 },
        { 1, 1, 0, 1, 1, 0, 0, 13, 20, 40, 2, 0, 0, 0, 3, 2, 1, 0, 0, 1.0f, 1.0f, 12, 0 },
        { 2, 2, 0, 0, 1, 0, 0, 19, 11, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 13, 0 },
        { 2, 2, 2, 1, 0, 0, 0, 9, 24, 17, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0x1p-14f, 0.0f, 14, 0 },
        { 3, 3, 0, 0, 0, 0, 0, 17, 9, 70, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 15, 0 },
        { 3, 4, 1, 1, 0, 0, 1, 10, 12, 45, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 16, 0 },
        { 4, 4, 0, 0, 0, 0, 0, 7, 13, 21, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 17, 0 },
        { 0, 0, 0, 0, 1, 0, 0, 5, 6, 23, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 18, 0 },
        { 0, 0, 0, 0, 0, 1, 0, 6, 5, 33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 19, 0 },
        { 0, 0, 0, 1, 1, 2, 0, 9, 130, 37, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, -1.5f, 20, 0 },
        { 0, 0, 0, 0, 0, 2, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 21, 0 },
        { 3, 3, 0, 0, 0, 0, 2, 12, 12, 64, 1, 5, 16, 32, 0, 0, 0, 0, 0, 1.0f, 0.0f, 22, 0 } };
    omega_gpu_gemm_ex_set_backend(sim_core, &sc);
    for (size_t i = 0; i < sizeof base / sizeof base[0]; i++) { run_case(&base[i], 0, NULL, NULL, NULL); same_as_model(&base[i], sim_core, &sc, "simulator"); }
    /* the kernel mutants: FFMA 1, f16 1-3 */
    static const struct { unsigned t; int mu; Case c; } km[] = {
        { OMEGA_GPU_GEMM_F32_FFMA, 1, { 0, 0, 0, 0, 0, 2, 0, 5, 7, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 31, 0 } },
        { OMEGA_GPU_GEMM_F16, 1, { 2, 2, 0, 0, 0, 0, 0, 17, 9, 33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 32, 0 } },
        { OMEGA_GPU_GEMM_F16, 2, { 2, 2, 0, 0, 0, 0, 0, 17, 9, 33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 33, 0 } },
        { OMEGA_GPU_GEMM_F16, 3, { 2, 2, 0, 0, 0, 0, 0, 17, 9, 33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 34, 0 } } };
    for (size_t i = 0; i < sizeof km / sizeof km[0]; i++) {
        sim_free(&sc); sc.mutant[km[i].t] = km[i].mu;
        uint64_t bad = run_case(&km[i].c, 1, NULL, NULL, NULL);
        sim_free(&sc); sc.mutant[km[i].t] = 0;
        snprintf(w, sizeof w, "sim red check: kernel %s mutant %d is caught (%" PRIu64 " wrong)", km[i].t == OMEGA_GPU_GEMM_F16 ? "f16" : "ffma", km[i].mu, bad);
        check(bad > 0, w);
        printf("%s\n", w);
    }
    sim_free(&sc);
    omega_gpu_gemm_ex_set_backend(NULL, NULL);
}

/* one 1x1x1 product through the native core: does the tensor core keep a subnormal input? */
static void subnormal_probe(OmegaGemmExCoreFn fn, void *ctx, unsigned dt, uint16_t a, uint16_t b, const char *what) {
    char w[300];
    uint16_t A = a, B = b; float C1 = 0, C2 = 0;
    OmegaGemmExArgs x; memset(&x, 0, sizeof x);
    x.m = x.n = x.k = 1; x.a = &A; x.b = &B; x.a_type = x.b_type = dt; x.c_type = OMEGA_GEMM_DT_F32; x.alpha = 1.0f;
    x.c = &C1; omega_gpu_gemm_ex_set_backend(fn, ctx); int rc1 = omega_gpu_gemm_ex(&x);
    x.c = &C2; omega_gpu_gemm_ex_set_backend(model_core, NULL); int rc2 = omega_gpu_gemm_ex(&x);
    omega_gpu_gemm_ex_set_backend(fn, ctx);
    const double exact = (double)tval(dt, &A, 0) * (double)tval(dt, &B, 0);
    snprintf(w, sizeof w, "subnormal probe %s: core %a (%s), model %a, exact %a: the tensor core %s the subnormal input", what, (double)C1,
             omega_gpu_gemm_rc_name(rc1), (double)C2, exact, C1 == 0.0f ? "FLUSHES" : (double)C1 == exact ? "KEEPS" : "changes");
    check(rc1 == 0 && rc2 == 0 && t_f2u(C1) == t_f2u(C2), w);
    printf("%s\n", w);
}

static void chip_tier(int quick, uint32_t part, uint32_t parts) {
    char w[300];
    omega_gpu_gemm_ex_set_backend(NULL, NULL);
    if (part == 0) {
        subnormal_probe(NULL, NULL, OMEGA_GEMM_DT_BF16, 0x0001u, 0x7180u, "bf16 2^-133 x 2^100");
        subnormal_probe(NULL, NULL, OMEGA_GEMM_DT_F16, 0x0001u, 0x6400u, "f16 2^-24 x 2^10");
        Case e[32]; int ne = 0;
        edge_cases(e, &ne);
        for (int i = 0; i < ne; i++) run_case(&e[i], 0, NULL, NULL, NULL);
        ex_mutants("chip");
        fp8_rows_tier("chip");
        fp8_api_chip();
        /* the kernel mutants on the chip */
        static const struct { unsigned t; int mu; Case c; } km[] = {
            { OMEGA_GPU_GEMM_F32_FFMA, 1, { 0, 0, 0, 0, 0, 2, 0, 65, 130, 33, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 41, 0 } },
            { OMEGA_GPU_GEMM_F16, 1, { 2, 2, 0, 0, 0, 0, 0, 65, 63, 100, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 42, 0 } },
            { OMEGA_GPU_GEMM_F16, 2, { 2, 2, 0, 0, 0, 0, 0, 65, 63, 100, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 43, 0 } },
            { OMEGA_GPU_GEMM_F16, 3, { 2, 2, 0, 0, 0, 0, 0, 65, 63, 100, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, 0.0f, 44, 0 } } };
        for (size_t i = 0; i < sizeof km / sizeof km[0]; i++) {
            OmegaGemmCfg cfg; omega_gpu_gemm_default_cfg((OmegaGpuGemmType)km[i].t, &cfg); cfg.mutant = km[i].mu;
            omega_gpu_gemm_test_set_cfg((OmegaGpuGemmType)km[i].t, &cfg);
            uint64_t bad = run_case(&km[i].c, 1, NULL, NULL, NULL);
            omega_gpu_gemm_test_set_cfg((OmegaGpuGemmType)km[i].t, NULL);
            snprintf(w, sizeof w, "chip red check: kernel %s mutant %d is caught (%" PRIu64 " wrong)", km[i].t == OMEGA_GPU_GEMM_F16 ? "f16" : "ffma", km[i].mu, bad);
            check(bad > 0, w);
            printf("%s\n", w);
        }
    }
    /* the sweep */
    const uint32_t n = quick ? 40u : SWEEP_N;
    uint32_t ran = 0, exact = 0;
    double worst[6] = { 0 };
    int before = g_failed;
    for (uint32_t i = part; i < n; i += parts) {
        Case c = sweep_case(i);
        unsigned route = 0; double rel = 0; uint64_t calls = 0;
        run_case(&c, 0, &route, &rel, &calls);
        if (route < 6 && rel > worst[route]) worst[route] = rel;
        ran++;
        /* chip core == model core bit for bit where the model is cheap (the model computes every k of every element) */
        uint64_t kcore = route == OMEGA_GEMM_ROUTE_BF16X3 ? 9u * (uint64_t)c.k : c.k;
        if ((uint64_t)c.m * c.n * kcore * c.batch <= (1ull << 24) && i % 2u == 0) { same_as_model(&c, NULL, NULL, "chip"); exact++; }
    }
    snprintf(w, sizeof w, "sweep part %u/%u: %u cases (%u also bit for bit against the model core): %s", part, parts, ran, exact, g_failed == before ? "PASS" : "FAIL");
    printf("%s\n", w);
    for (int r = 1; r < 6; r++) printf("route %s: largest |got - ref| = %.2f x 2^-24 |alpha s| sum|a b f|\n", RTN[r], worst[r]);
    omega_gpu_session_close();
}

int main(int argc, char **argv) {
    const char *tier = "chip";
    int quick = 0;
    uint32_t part = 0, parts = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) tier = "host";
        else if (!strcmp(argv[i], "--sim")) tier = "sim";
        else if (!strcmp(argv[i], "--list")) { list_sweep(); return 0; }
        else if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--part") && i + 1 < argc && sscanf(argv[i + 1], "%u/%u", &part, &parts) == 2 && parts && part < parts) i++;
        else { fprintf(stderr, "usage: %s [--host-only | --sim | --list | --quick | --part I/N]\n", argv[0]); return 2; }
    }
    if (!strcmp(tier, "host")) host_tier();
    else if (!strcmp(tier, "sim")) { host_tier(); sim_tier(); }
    else chip_tier(quick, part, parts);
    if (g_failed) printf("gemm_ex %s: FAIL (%d of %d)\n", tier, g_failed, g_checks);
    else printf("gemm_ex %s: PASS (%d checks)\n", tier, g_checks);
    return g_failed ? 1 : 0;
}
