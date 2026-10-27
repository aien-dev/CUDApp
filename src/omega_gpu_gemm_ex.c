/* omega_gpu_gemm_ex.c -- the general matrix multiply over the C3 kernels (contract: omega_gpu_gemm_ex.h).
 * Migration-pending C: switch CUDAPP_MIGRATION_PENDING_GEMM_EX (mk/gemm-ex.mk), successor src/omega_gpu_gemm_ex.om. */
#include "omega_gpu_gemm_ex.h"
#include "omega_gpu_session.h"

#include <string.h>

#ifndef CUDAPP_MIGRATION_PENDING_GEMM_EX
#define CUDAPP_MIGRATION_PENDING_GEMM_EX 1
#endif
#if CUDAPP_MIGRATION_PENDING_GEMM_EX

#define TILE 4096u        /* rows / columns of C per host tile */
#define KMAX 16384u       /* k per C3 call (OMEGA_GPU_GEMM_MAX_DIM) */
#define KMAX3 1808u       /* k per bf16x3 call: 9 planes of roundup(k, 16) <= 16384 */

static int chip_core(void *ctx, OmegaGpuGemmType t, unsigned layout, uint32_t M, uint32_t N, uint32_t K, const void *A, uint64_t lda,
                     const void *B, uint64_t ldb, float *C) {
    (void)ctx;
    return omega_gpu_gemm_ld(t, layout, M, N, K, A, lda, B, ldb, C, NULL);
}
static OmegaGemmExCoreFn g_core = chip_core;
static void *g_core_ctx;
static int g_mutant;

void omega_gpu_gemm_ex_set_backend(OmegaGemmExCoreFn fn, void *ctx) { g_core = fn ? fn : chip_core; g_core_ctx = fn ? ctx : NULL; }
void omega_gpu_gemm_ex_test_set_mutant(int mutant) { g_mutant = mutant; }

/* ------------------------------------------------------------------ number formats */

static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static int fin(float f) { return (f2u(f) & 0x7f800000u) != 0x7f800000u; }
static double dabs(double v) { return v < 0 ? -v : v; }

static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31u, m = h & 0x3ffu;
    if (e == 31u) return u2f(s | 0x7f800000u | m << 13);
    if (e == 0u) { float v = (float)m * (1.0f / 16777216.0f); return s ? -v : v; } /* m 2^-24, exact */
    return u2f(s | (e + 112u) << 23 | m << 13);
}
static float e4m3f(uint8_t b) {
    uint32_t e = (b >> 3) & 15u, m = b & 7u;
    if (e == 15u && m == 7u) return u2f(0x7fc00000u);
    float v = e == 0u ? (float)m * (1.0f / 512.0f) : u2f((e + 120u) << 23 | m << 20);
    return b & 0x80u ? -v : v;
}
static uint32_t esize(unsigned t) { return t == OMEGA_GEMM_DT_F32 ? 4u : t <= OMEGA_GEMM_DT_F16 ? 2u : 1u; }
static float ldf(unsigned t, const void *p, uint64_t i) {
    switch (t) {
    case OMEGA_GEMM_DT_F32: { float f; memcpy(&f, (const uint8_t *)p + i * 4u, 4); return f; }
    case OMEGA_GEMM_DT_BF16: return u2f((uint32_t)((const uint16_t *)p)[i] << 16);
    case OMEGA_GEMM_DT_F16: return h2f(((const uint16_t *)p)[i]);
    case OMEGA_GEMM_DT_E4M3: return e4m3f(((const uint8_t *)p)[i]);
    default: return h2f((uint16_t)(((const uint8_t *)p)[i] << 8)); /* an e5m2 byte is the top byte of an f16 */
    }
}
static uint16_t f2bf_rne(float f) {
    uint32_t u = f2u(f);
    if (!fin(f)) return (uint16_t)(u >> 16 | ((u & 0x7fffffu) ? 0x40u : 0u));
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static uint16_t f2h_rne(float f) {
    uint32_t u = f2u(f), s = (u >> 16) & 0x8000u, a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) return (uint16_t)(s | 0x7c00u | (a > 0x7f800000u ? 0x200u : 0u));
    if (a >= 0x477ff000u) return (uint16_t)(s | 0x7c00u);      /* >= 65520 rounds past the f16 maximum */
    if (a < 0x38800000u) {                                     /* below 2^-14: an f16 subnormal (unit 2^-24) or zero */
        if (a < 0x33000000u) return (uint16_t)s;               /* below 2^-25 */
        uint32_t mant = (a & 0x7fffffu) | 0x800000u, sh = 126u - (a >> 23);
        uint32_t q = mant >> sh, rem = mant & ((1u << sh) - 1u), half = 1u << (sh - 1u);
        if (rem > half || (rem == half && (q & 1u))) q++;
        return (uint16_t)(s | q);
    }
    uint32_t r = a + 0xfffu + ((a >> 13) & 1u);
    return (uint16_t)(s | (((r >> 23) - 112u) << 10 | ((r >> 13) & 0x3ffu)));
}
/* double to float rounding to odd: one more rounding to bf16 or f16 is then the correct rounding of v */
static float d2f_odd(double v) {
    float f = (float)v;
    if (v != v || !fin(f) || (double)f == v) return f;
    uint32_t u = f2u(f);
    if (u & 1u) return f;
    return u2f(dabs(v) > dabs((double)f) ? u + 1u : u - 1u);
}
/* exact split of a finite f32 into three bf16 planes by truncation: f = hi + mid + lo, except the parts below 2^-133 */
static void split3(float f, uint16_t p[3]) {
    if (!fin(f)) { p[0] = p[1] = p[2] = 0; return; }
    uint32_t u = f2u(f);
    p[0] = (uint16_t)(u >> 16);
    float r1 = f - u2f(u & 0xffff0000u);
    uint32_t u1 = f2u(r1);
    p[1] = (uint16_t)(u1 >> 16);
    p[2] = (uint16_t)(f2u(r1 - u2f(u1 & 0xffff0000u)) >> 16);
}
/* plane products in k order, smallest first: (lo,lo) (lo,mid) (mid,lo) (lo,hi) (mid,mid) (hi,lo) (mid,hi) (hi,mid) (hi,hi) */
static const uint8_t TA[9] = { 2, 2, 1, 2, 1, 0, 1, 0, 0 }, TB[9] = { 2, 1, 2, 0, 1, 2, 0, 1, 0 };

/* ------------------------------------------------------------------ the call */

typedef struct {
    OmegaGemmExArgs x;
    uint64_t sa_r, sa_c, sb_c;           /* block-scale grid: rows, k blocks, column blocks */
    const float *sa, *sb;                /* this batch's scales */
    const uint8_t *A, *B;                /* this batch's operands */
    unsigned route;
    OmegaGpuGemmType core;
    uint64_t kmax;
} Call;

static uint64_t a_at(const Call *c, uint64_t i, uint64_t kk) { return c->x.op_a ? kk * c->x.lda + i : i * c->x.lda + kk; }
static uint64_t b_at(const Call *c, uint64_t kk, uint64_t j) { return c->x.op_b ? j * c->x.ldb + kk : kk * c->x.ldb + j; }
static float aval(const Call *c, uint64_t i, uint64_t kk) { return ldf(c->x.a_type, c->A, a_at(c, i, kk)); }
static float bval(const Call *c, uint64_t kk, uint64_t j) { return ldf(c->x.b_type, c->B, b_at(c, kk, j)); }
static int bad_a(const Call *c, uint64_t i, uint64_t kk) {
    if (c->x.a_type == OMEGA_GEMM_DT_E4M3) return (c->A[a_at(c, i, kk)] & 0x7fu) == 0x7fu;
    return !fin(aval(c, i, kk));
}
static int bad_b(const Call *c, uint64_t kk, uint64_t j) {
    if (c->x.b_type == OMEGA_GEMM_DT_E4M3) return (c->B[b_at(c, kk, j)] & 0x7fu) == 0x7fu;
    return !fin(bval(c, kk, j));
}
static double block_factor(const Call *c, uint64_t i, uint64_t kb, uint64_t j) {
    if (c->x.scale != OMEGA_GEMM_SCALE_BLOCK) return 1.0;
    return (double)c->sa[(i / c->x.block_m) * c->sa_c + kb] * (double)c->sb[kb * c->sb_c + j / c->x.block_n];
}

/* stored rows x cols with leading dimension ld; 0 when ld is short or the span passes 2^56 */
static int span_of(uint64_t rows, uint64_t cols, uint64_t ld, uint64_t *span) {
    if (ld < cols || (rows - 1u) > (1ull << 56) / ld) return 0;
    *span = (rows - 1u) * ld + cols;
    return 1;
}
static int extent(uint64_t span, uint64_t stride, uint64_t batch, uint64_t *ext) {
    uint64_t t;
    if (__builtin_mul_overflow(batch - 1u, stride, &t) || __builtin_add_overflow(t, span, ext) || *ext > (1ull << 58)) return 0;
    return 1;
}
static int overlap(const void *p, uint64_t pbytes, const void *q, uint64_t qbytes) {
    uintptr_t a = (uintptr_t)p, b = (uintptr_t)q;
    return a < b + qbytes && b < a + pbytes;
}
static int scales_finite(const float *s, uint64_t count, uint64_t stride, uint64_t batch) {
    for (uint64_t b = 0; b < (stride ? batch : 1u); b++)
        for (uint64_t i = 0; i < count; i++) if (!fin(s[b * stride + i])) return 0;
    return 1;
}

static int validate(Call *c, const OmegaGemmExArgs *in) {
    c->x = *in;
    OmegaGemmExArgs *x = &c->x;
    if (!x->batch) x->batch = 1;
    if (x->op_a > 1u || x->op_b > 1u || x->a_type > OMEGA_GEMM_DT_E5M2 || x->b_type > OMEGA_GEMM_DT_E5M2 ||
        x->c_type > OMEGA_GEMM_DT_E5M2 || x->compute > OMEGA_GEMM_EX_F32_FFMA_REF || x->scale > OMEGA_GEMM_SCALE_BLOCK || !x->c)
        return OMEGA_GPU_GEMM_BAD_ARGS;
    if (!x->m || !x->n || !x->k) return OMEGA_GPU_GEMM_BAD_ARGS;
    if (x->m > OMEGA_GPU_GEMM_EX_MAX_DIM || x->n > OMEGA_GPU_GEMM_EX_MAX_DIM || x->k > OMEGA_GPU_GEMM_EX_MAX_DIM ||
        x->batch > OMEGA_GPU_GEMM_EX_MAX_BATCH) return OMEGA_GPU_GEMM_TOO_LARGE;
    if (x->c_type > OMEGA_GEMM_DT_F16) return OMEGA_GPU_GEMM_UNSUPPORTED;
    const uint64_t ar = x->op_a ? x->k : x->m, ac = x->op_a ? x->m : x->k;
    const uint64_t br = x->op_b ? x->n : x->k, bc = x->op_b ? x->k : x->n;
    if (!x->lda) x->lda = ac;
    if (!x->ldb) x->ldb = bc;
    if (!x->ldc) x->ldc = x->n;
    uint64_t sa, sb, sc, ea, eb, ec;
    if (!span_of(ar, ac, x->lda, &sa) || !span_of(br, bc, x->ldb, &sb) || !span_of(x->m, x->n, x->ldc, &sc)) return OMEGA_GPU_GEMM_BAD_ARGS;
    if (x->batch > 1u && x->stride_c < sc) return OMEGA_GPU_GEMM_BAD_ARGS; /* C batches would overlap */
    if (!extent(sa, x->stride_a, x->batch, &ea) || !extent(sb, x->stride_b, x->batch, &eb) || !extent(sc, x->stride_c, x->batch, &ec))
        return OMEGA_GPU_GEMM_BAD_ARGS;
    const uint64_t cbytes = ec * esize(x->c_type);
    if (x->alpha != 0.0f) {
        if (!x->a || !x->b) return OMEGA_GPU_GEMM_BAD_ARGS;
        if (overlap(x->c, cbytes, x->a, ea * esize(x->a_type)) || overlap(x->c, cbytes, x->b, eb * esize(x->b_type))) return OMEGA_GPU_GEMM_BAD_ARGS;
    }
    if (x->scale == OMEGA_GEMM_SCALE_TENSOR) {
        if (!x->scale_a || !x->scale_b || !scales_finite(x->scale_a, 1, x->scale_stride_a, x->batch) ||
            !scales_finite(x->scale_b, 1, x->scale_stride_b, x->batch)) return OMEGA_GPU_GEMM_BAD_ARGS;
    } else if (x->scale == OMEGA_GEMM_SCALE_BLOCK) {
        if (!x->scale_a || !x->scale_b || !x->block_m || !x->block_n || !x->block_k) return OMEGA_GPU_GEMM_BAD_ARGS;
        c->sa_r = (x->m + x->block_m - 1u) / x->block_m; c->sa_c = (x->k + x->block_k - 1u) / x->block_k;
        c->sb_c = (x->n + x->block_n - 1u) / x->block_n;
        if (!scales_finite(x->scale_a, c->sa_r * c->sa_c, x->scale_stride_a, x->batch) ||
            !scales_finite(x->scale_b, c->sa_c * c->sb_c, x->scale_stride_b, x->batch)) return OMEGA_GPU_GEMM_BAD_ARGS;
    }
    /* the route */
    const unsigned ta = x->a_type, tb = x->b_type;
    const int bf_ok = ta != OMEGA_GEMM_DT_F32 && ta != OMEGA_GEMM_DT_F16 && tb != OMEGA_GEMM_DT_F32 && tb != OMEGA_GEMM_DT_F16;
    const int h_ok = ta != OMEGA_GEMM_DT_F32 && ta != OMEGA_GEMM_DT_BF16 && tb != OMEGA_GEMM_DT_F32 && tb != OMEGA_GEMM_DT_BF16;
    c->kmax = KMAX;
    if (x->compute == OMEGA_GEMM_EX_F32_FFMA_REF) { c->route = OMEGA_GEMM_ROUTE_FFMA; c->core = OMEGA_GPU_GEMM_F32_FFMA; }
    else if (ta == tb && ta == OMEGA_GEMM_DT_BF16) { c->route = OMEGA_GEMM_ROUTE_NATIVE; c->core = OMEGA_GPU_GEMM_BF16; }
    else if (ta == tb && ta == OMEGA_GEMM_DT_F16) { c->route = OMEGA_GEMM_ROUTE_NATIVE; c->core = OMEGA_GPU_GEMM_F16; }
    else if (ta == tb && ta == OMEGA_GEMM_DT_E4M3) { c->route = OMEGA_GEMM_ROUTE_NATIVE; c->core = OMEGA_GPU_GEMM_E4M3; }
    else if (bf_ok) { c->route = OMEGA_GEMM_ROUTE_WIDEN; c->core = OMEGA_GPU_GEMM_BF16; }
    else if (h_ok) { c->route = OMEGA_GEMM_ROUTE_WIDEN; c->core = OMEGA_GPU_GEMM_F16; }
    else if (x->compute == OMEGA_GEMM_EX_F32_FAST_BF16) { c->route = OMEGA_GEMM_ROUTE_FAST_BF16; c->core = OMEGA_GPU_GEMM_BF16; }
    else { c->route = OMEGA_GEMM_ROUTE_BF16X3; c->core = OMEGA_GPU_GEMM_BF16; c->kmax = KMAX3; }
    if (g_mutant == 2) x->op_b = OMEGA_GEMM_OP_N;
    return OMEGA_GPU_GEMM_OK;
}

/* the operand buffers of one k part: A' rows of op(A), B' columns of op(B) (B' row-major k x n for FFMA) */
typedef struct { uint8_t *a, *b; float *p; double *acc; uint8_t *rowbad, *colbad; } Bufs;

static uint32_t core_esize(OmegaGpuGemmType t) { return t == OMEGA_GPU_GEMM_E4M3 ? 1u : t == OMEGA_GPU_GEMM_F32_FFMA ? 4u : 2u; }
static uint32_t round16(uint32_t v) { return (v + 15u) & ~15u; }

/* encode an exactly representable finite value in the core type */
static void put_core(OmegaGpuGemmType t, uint8_t *dst, uint64_t i, float f) {
    if (!fin(f)) f = 0.0f;
    if (t == OMEGA_GPU_GEMM_F32_FFMA) memcpy(dst + i * 4u, &f, 4);
    else if (t == OMEGA_GPU_GEMM_F16) ((uint16_t *)dst)[i] = f2h_rne(f);
    else ((uint16_t *)dst)[i] = (uint16_t)(f2u(f) >> 16);
}

/* one k part [k0, k0 + kc) of the tile [m0, m0 + mc) x [n0, n0 + nc) into P (dense mc x nc) */
static int run_part(Call *c, Bufs *bf, int any_bad, uint64_t m0, uint32_t mc, uint64_t n0, uint32_t nc, uint64_t k0, uint32_t kc,
                    uint64_t *calls) {
    const OmegaGemmExArgs *x = &c->x;
    const OmegaGpuGemmType t = c->core;
    const unsigned layout = (x->op_a ? OMEGA_GPU_GEMM_A_COL : 0u) | (x->op_b ? OMEGA_GPU_GEMM_B_COL : 0u);
    int rc;
    (*calls)++;
    if (c->route == OMEGA_GEMM_ROUTE_NATIVE && !any_bad && omega_gpu_gemm_span(layout, mc, nc, kc, x->lda, x->ldb, NULL, NULL) == 0) {
        const uint64_t es = esize(x->a_type);
        const uint8_t *pa = c->A + (x->op_a ? k0 * x->lda + m0 : m0 * x->lda + k0) * es;
        const uint8_t *pb = c->B + (x->op_b ? n0 * x->ldb + k0 : k0 * x->ldb + n0) * es;
        return g_core(g_core_ctx, t, layout, mc, nc, kc, pa, x->lda, pb, x->ldb, bf->p);
    }
    if (c->route == OMEGA_GEMM_ROUTE_BF16X3) {
        const uint32_t kp = round16(kc), K = 9u * kp;
        uint16_t *A2 = (uint16_t *)bf->a, *B2 = (uint16_t *)bf->b;
        for (uint32_t i = 0; i < mc; i++) {
            uint16_t *row = A2 + (uint64_t)i * K;
            for (uint32_t kk = 0; kk < kp; kk++) {
                uint16_t p[3] = { 0, 0, 0 };
                if (kk < kc) split3(aval(c, m0 + i, k0 + kk), p);
                for (unsigned q = 0; q < 9u; q++) row[q * kp + kk] = p[TA[q]];
            }
        }
        for (uint32_t j = 0; j < nc; j++) {
            uint16_t *col = B2 + (uint64_t)j * K;
            for (uint32_t kk = 0; kk < kp; kk++) {
                uint16_t p[3] = { 0, 0, 0 };
                if (kk < kc) split3(bval(c, k0 + kk, n0 + j), p);
                for (unsigned q = 0; q < 9u; q++) col[q * kp + kk] = p[TB[q]];
            }
        }
        return g_core(g_core_ctx, OMEGA_GPU_GEMM_BF16, OMEGA_GPU_GEMM_B_COL, mc, nc, K, A2, 0, B2, 0, bf->p);
    }
    if (c->route == OMEGA_GEMM_ROUTE_FAST_BF16) {
        uint16_t *A2 = (uint16_t *)bf->a, *B2 = (uint16_t *)bf->b;
        for (uint32_t i = 0; i < mc; i++)
            for (uint32_t kk = 0; kk < kc; kk++) { float f = aval(c, m0 + i, k0 + kk); A2[(uint64_t)i * kc + kk] = fin(f) ? f2bf_rne(f) : 0; }
        for (uint32_t j = 0; j < nc; j++)
            for (uint32_t kk = 0; kk < kc; kk++) { float f = bval(c, k0 + kk, n0 + j); B2[(uint64_t)j * kc + kk] = fin(f) ? f2bf_rne(f) : 0; }
        return g_core(g_core_ctx, OMEGA_GPU_GEMM_BF16, OMEGA_GPU_GEMM_B_COL, mc, nc, kc, A2, 0, B2, 0, bf->p);
    }
    if (c->route == OMEGA_GEMM_ROUTE_FFMA) {
        for (uint32_t i = 0; i < mc; i++)
            for (uint32_t kk = 0; kk < kc; kk++) put_core(t, bf->a, (uint64_t)i * kc + kk, aval(c, m0 + i, k0 + kk));
        for (uint32_t kk = 0; kk < kc; kk++)
            for (uint32_t j = 0; j < nc; j++) put_core(t, bf->b, (uint64_t)kk * nc + j, bval(c, k0 + kk, n0 + j));
        return g_core(g_core_ctx, t, 0u, mc, nc, kc, bf->a, 0, bf->b, 0, bf->p);
    }
    /* WIDEN, or NATIVE with non-finite inputs or a span past 2^32: dense copies in the core type, non-finite as 0 */
    if (t == OMEGA_GPU_GEMM_E4M3) {
        for (uint32_t i = 0; i < mc; i++)
            for (uint32_t kk = 0; kk < kc; kk++) { uint8_t v = c->A[a_at(c, m0 + i, k0 + kk)]; bf->a[(uint64_t)i * kc + kk] = (v & 0x7fu) == 0x7fu ? 0 : v; }
        for (uint32_t j = 0; j < nc; j++)
            for (uint32_t kk = 0; kk < kc; kk++) { uint8_t v = c->B[b_at(c, k0 + kk, n0 + j)]; bf->b[(uint64_t)j * kc + kk] = (v & 0x7fu) == 0x7fu ? 0 : v; }
    } else {
        for (uint32_t i = 0; i < mc; i++)
            for (uint32_t kk = 0; kk < kc; kk++) put_core(t, bf->a, (uint64_t)i * kc + kk, aval(c, m0 + i, k0 + kk));
        for (uint32_t j = 0; j < nc; j++)
            for (uint32_t kk = 0; kk < kc; kk++) put_core(t, bf->b, (uint64_t)j * kc + kk, bval(c, k0 + kk, n0 + j));
    }
    rc = g_core(g_core_ctx, t, OMEGA_GPU_GEMM_B_COL, mc, nc, kc, bf->a, 0, bf->b, 0, bf->p);
    return rc;
}

static double host_dot(const Call *c, uint64_t i, uint64_t j) {
    double s = 0.0;
    for (uint64_t kk = 0; kk < c->x.k; kk++)
        s += (double)aval(c, i, kk) * (double)bval(c, kk, j) * block_factor(c, i, kk / (c->x.block_k ? c->x.block_k : 1u), j);
    return s;
}

static double load_c(unsigned t, const void *p, uint64_t i) { return (double)ldf(t, p, i); }
static void store_c(unsigned t, void *p, uint64_t i, double v) {
    if (t == OMEGA_GEMM_DT_F32) { float f = (float)v; memcpy((uint8_t *)p + i * 4u, &f, 4); }
    else if (t == OMEGA_GEMM_DT_BF16) ((uint16_t *)p)[i] = f2bf_rne(d2f_odd(v));
    else ((uint16_t *)p)[i] = f2h_rne(d2f_odd(v));
}

int omega_gpu_gemm_ex(const OmegaGemmExArgs *args) {
    if (!args) return OMEGA_GPU_GEMM_BAD_ARGS;
    Call c;
    memset(&c, 0, sizeof c);
    int rc = validate(&c, args);
    if (rc != OMEGA_GPU_GEMM_OK) return rc;
    const OmegaGemmExArgs *x = &c.x;
    OmegaGemmExInfo info;
    memset(&info, 0, sizeof info);
    const int compute = x->alpha != 0.0f;
    const double beta = g_mutant == 4 ? 0.0 : (double)x->beta;
    const uint64_t tm = x->m < TILE ? x->m : TILE, tn = x->n < TILE ? x->n : TILE;
    Bufs bf;
    memset(&bf, 0, sizeof bf);
    if (compute) {
        info.route = c.route; info.core_type = c.core;
        const uint64_t kc = x->k < c.kmax ? x->k : c.kmax;
        const uint64_t kcore = c.route == OMEGA_GEMM_ROUTE_BF16X3 ? 9u * round16((uint32_t)kc) : kc;
        const uint64_t es = core_esize(c.core);
        bf.a = omega_gpu_host_calloc(tm * kcore + 16u, es);
        bf.b = omega_gpu_host_calloc(tn * kcore + 16u, es);
        bf.p = omega_gpu_host_calloc(tm * tn, sizeof(float));
        bf.acc = omega_gpu_host_calloc(tm * tn, sizeof(double));
        bf.rowbad = omega_gpu_host_calloc(x->m, 1);
        bf.colbad = omega_gpu_host_calloc(x->n, 1);
        if (!bf.a || !bf.b || !bf.p || !bf.acc || !bf.rowbad || !bf.colbad) { rc = OMEGA_GPU_GEMM_CHIP_FAIL; goto out; }
    }
    for (uint64_t b = 0; b < x->batch; b++) {
        uint8_t *C = (uint8_t *)x->c + b * x->stride_c * esize(x->c_type);
        double alpha = (double)x->alpha;
        int any_bad = 0;
        if (compute) {
            c.A = (const uint8_t *)x->a + b * x->stride_a * esize(x->a_type);
            c.B = (const uint8_t *)x->b + b * x->stride_b * esize(x->b_type);
            if (x->scale != OMEGA_GEMM_SCALE_NONE) { c.sa = x->scale_a + b * x->scale_stride_a; c.sb = x->scale_b + b * x->scale_stride_b; }
            if (x->scale == OMEGA_GEMM_SCALE_TENSOR) alpha *= (double)c.sa[0] * (double)c.sb[0];
            memset(bf.rowbad, 0, x->m); memset(bf.colbad, 0, x->n);
            for (uint64_t i = 0; i < x->m; i++) for (uint64_t kk = 0; kk < x->k; kk++) if (bad_a(&c, i, kk)) { bf.rowbad[i] = 1; any_bad = 1; break; }
            for (uint64_t j = 0; j < x->n; j++) for (uint64_t kk = 0; kk < x->k; kk++) if (bad_b(&c, kk, j)) { bf.colbad[j] = 1; any_bad = 1; break; }
        }
        for (uint64_t m0 = 0; m0 < x->m; m0 += TILE)
            for (uint64_t n0 = 0; n0 < x->n; n0 += TILE) {
                const uint32_t mc = (uint32_t)(x->m - m0 < TILE ? x->m - m0 : TILE), nc = (uint32_t)(x->n - n0 < TILE ? x->n - n0 : TILE);
                if (compute) {
                    memset(bf.acc, 0, (uint64_t)mc * nc * sizeof(double));
                    for (uint64_t k0 = 0; k0 < x->k;) {
                        uint64_t kc = x->k - k0 < c.kmax ? x->k - k0 : c.kmax;
                        if (x->scale == OMEGA_GEMM_SCALE_BLOCK && kc > x->block_k - k0 % x->block_k) kc = x->block_k - k0 % x->block_k;
                        uint64_t ku = kc;
                        if (g_mutant == 1 && k0 + kc == x->k) ku = kc - 1u;
                        if (ku) {
                            rc = run_part(&c, &bf, any_bad, m0, mc, n0, nc, k0, (uint32_t)ku, &info.core_calls);
                            if (rc != OMEGA_GPU_GEMM_OK) goto out;
                            uint64_t kb = x->scale == OMEGA_GEMM_SCALE_BLOCK ? k0 / x->block_k : 0;
                            if (g_mutant == 3 && x->scale == OMEGA_GEMM_SCALE_BLOCK) kb = (kb + 1u) % c.sa_c;
                            for (uint32_t i = 0; i < mc; i++)
                                for (uint32_t j = 0; j < nc; j++)
                                    bf.acc[(uint64_t)i * nc + j] += (double)bf.p[(uint64_t)i * nc + j] * block_factor(&c, m0 + i, kb, n0 + j);
                        }
                        k0 += kc;
                    }
                }
                for (uint32_t i = 0; i < mc; i++)
                    for (uint32_t j = 0; j < nc; j++) {
                        const uint64_t ci = (m0 + i) * x->ldc + n0 + j;
                        double v = 0.0;
                        if (compute) {
                            double a = bf.acc[(uint64_t)i * nc + j];
                            if (any_bad && (bf.rowbad[m0 + i] || bf.colbad[n0 + j])) { a = host_dot(&c, m0 + i, n0 + j); info.host_fixed++; }
                            v = alpha * a;
                        }
                        if (beta != 0.0) v += beta * load_c(x->c_type, C, ci);
                        store_c(x->c_type, C, ci, v);
                    }
            }
    }
out:
    omega_gpu_host_free(bf.a); omega_gpu_host_free(bf.b); omega_gpu_host_free(bf.p); omega_gpu_host_free(bf.acc);
    omega_gpu_host_free(bf.rowbad); omega_gpu_host_free(bf.colbad);
    if (x->info) *x->info = info;
    return rc;
}

/* ------------------------------------------------------------------ the fp8 grouped/MoE fallback */

#define FP8_SLAB_BYTES (32ull << 20) /* dequantized weight rows per gemm_ex call */

int omega_gpu_gemm_ex_fp8_block_rows(const uint8_t *w, const float *scale, double scale_mul, uint32_t n, uint32_t k, uint32_t block,
                                     const float *x32, const uint16_t *x16, uint32_t m, float *y, uint64_t ldy, OmegaGemmExInfo *info) {
    if (info) memset(info, 0, sizeof *info);
    if (!w || !scale || !y || (!x32) == (!x16) || n == 0 || k == 0 || m == 0 || block == 0 || (ldy && ldy < n)) return OMEGA_GPU_GEMM_BAD_ARGS;
    if (!ldy) ldy = n;
    const uint64_t kb = (k + (uint64_t)block - 1u) / block;
    uint64_t rows = FP8_SLAB_BYTES / ((uint64_t)k * 4u);
    if (rows == 0) rows = 1;
    if (rows > n) rows = n;
    float *slab = omega_gpu_host_calloc((size_t)(rows * k), sizeof *slab);
    if (!slab) return OMEGA_GPU_GEMM_CHIP_FAIL;
    int rc = OMEGA_GPU_GEMM_OK;
    for (uint64_t j0 = 0; j0 < n && rc == OMEGA_GPU_GEMM_OK; j0 += rows) {
        const uint64_t nr = n - j0 < rows ? n - j0 : rows;
        for (uint64_t j = 0; j < nr; j++) {
            const uint8_t *wr = w + (j0 + j) * k;
            const float *sr = scale + ((j0 + j) / block) * kb;
            float *dr = slab + j * k;
            for (uint64_t c = 0; c < k; c++) {
                uint64_t cb = c / block;
                if (g_mutant == 5) cb = (cb + 1u) % kb;
                dr[c] = (float)((double)e4m3f(wr[c]) * (double)sr[cb] * scale_mul); /* one rounding */
            }
        }
        OmegaGemmExInfo pi;
        OmegaGemmExArgs a = {
            .op_a = OMEGA_GEMM_OP_N, .op_b = OMEGA_GEMM_OP_T, .m = m, .n = nr, .k = k, .batch = 1,
            .a_type = x32 ? OMEGA_GEMM_DT_F32 : OMEGA_GEMM_DT_BF16, .b_type = OMEGA_GEMM_DT_F32, .c_type = OMEGA_GEMM_DT_F32,
            .a = x32 ? (const void *)x32 : (const void *)x16, .lda = k, .b = slab, .ldb = k, .c = y + j0, .ldc = ldy,
            .alpha = 1.0f, .beta = 0.0f, .info = &pi,
        };
        rc = omega_gpu_gemm_ex(&a);
        if (info) { info->route = pi.route; info->core_type = pi.core_type; info->core_calls += pi.core_calls; info->host_fixed += pi.host_fixed; }
    }
    omega_gpu_host_free(slab);
    return rc;
}

#else /* the switch off: the entry refuses every call */
void omega_gpu_gemm_ex_set_backend(OmegaGemmExCoreFn fn, void *ctx) { (void)fn; (void)ctx; }
void omega_gpu_gemm_ex_test_set_mutant(int mutant) { (void)mutant; }
int omega_gpu_gemm_ex(const OmegaGemmExArgs *args) { (void)args; return OMEGA_GPU_GEMM_UNSUPPORTED; }
int omega_gpu_gemm_ex_fp8_block_rows(const uint8_t *w, const float *scale, double scale_mul, uint32_t n, uint32_t k, uint32_t block,
                                     const float *x32, const uint16_t *x16, uint32_t m, float *y, uint64_t ldy, OmegaGemmExInfo *info) {
    (void)w; (void)scale; (void)scale_mul; (void)n; (void)k; (void)block; (void)x32; (void)x16; (void)m; (void)y; (void)ldy;
    if (info) memset(info, 0, sizeof *info);
    return OMEGA_GPU_GEMM_UNSUPPORTED;
}
#endif
