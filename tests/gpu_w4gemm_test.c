/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * gpu_w4gemm_test: the NVFP4 W4A16 grouped GEMV/GEMM generator (omega_blackwell_codegen_w4.h), P-DRA-4 DRA-71. Host only.
 *   --host-only   kernel-order emulation vs a double oracle, permutation bijection, codegen budgets, register fit, refusals
 *   --sim         warp simulator (build with -DWS_MAX_THREADS=512): bit-identical to the emulation in both fragment
 *                 orders; multi-segment, multi-pass, row clamping, empty segments, rowmap; mutants 1..6 caught
 *   --real F...   .w4ref fixtures dumped by aien-sovereign-core-mv nvfp4.rs from the real NVFP4 checkpoint: the simulated
 *                 kernel at N = 1 bit-identical to the emulation on every row, emulation vs the f64 oracle <= 1e-5 of the
 *                 term scale, the wrong-scale mutant (2) red on the same bytes
 * Verdict line last: "gpu_w4gemm_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_blackwell_codegen_w4.h"
#include "bw_warp_sim_k.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }
static float f32_of_bits(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static const double E2M1[8] = { 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
static double e2m1_d(unsigned n) { return (n & 8u ? -1.0 : 1.0) * E2M1[n & 7u]; }
static double e4m3_d(uint8_t b) { unsigned ex = (b >> 3) & 15u, mn = b & 7u; double v = ex ? ldexp(1.0 + mn / 8.0, (int)ex - 7) : ldexp(mn / 8.0, -6); return b & 0x80u ? -v : v; }
/* a positive finite e4m3 scale byte (never NaN 0x7f, never 0) */
static uint8_t rand_scale(uint32_t *s) { uint8_t b = (uint8_t)((lcg(s) >> 24) & 0x7fu); if (b == 0x7f) b = 0x7e; if (b == 0) b = 1; return b; }
static unsigned nib(const uint8_t *row, uint32_t c) { uint8_t b = row[c / 2u]; return c & 1u ? b >> 4 : b & 15u; }

/* the kernel's exact f32 result for one weight row: codes k/2 bytes, scales k/16 bytes, gp = G' (global * 2^62), xp the
 * permuted 2^64-scaled row. mutant: the generator's mutants 1, 2, 5, 6 (3, 4 concern rows and segments) */
static float emu_row(const uint8_t *codes, const uint8_t *sc, float gp, const float *xp, uint32_t k, uint32_t lb, int mutant) {
    const uint32_t el = 2u * lb, steps = k / (32u * el), bpl = lb / 8u;
    float a[32];
    for (unsigned L = 0; L < 32; L++) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < steps; s++) {
            const uint32_t c0 = s * 32u * el + L * el;
            for (uint32_t b = 0; b < bpl; b++) {
                float dot = 0.0f;
                for (uint32_t ww = 0; ww < 2; ww++) {
                    const uint32_t word = 2u * b + ww;
                    for (unsigned j = 0; j < 8; j++) {
                        const unsigned jj = mutant == 5 ? j ^ 1u : j, n = nib(codes, c0 + 8u * word + jj);
                        const float f = f32_of_bits(((n & 7u) << 22) | ((n & 8u) << 28));
                        const uint32_t q = 2u * word + j / 4u;
                        dot = fmaf(f, xp[s * 32u * el + (q * 32u + L) * 4u + j % 4u], dot);
                    }
                }
                uint32_t blk = c0 / 16u + b;
                if (mutant == 2) blk = lb == 16 ? c0 / 16u + (b ^ 1u) : (L & 3u ? blk - 1u : ~0u);
                const uint8_t sb = blk == ~0u ? 0 : sc[blk];
                const float scf = f32_of_bits(((uint32_t)(sb & 0x7fu) << 20) | ((uint32_t)(sb & 0x80u) << 24)) * f32_of_bits(0x7B800000u);
                acc = fmaf(dot, mutant == 1 ? 1.0f : scf, acc);
            }
        }
        a[L] = acc;
    }
    for (unsigned o = 16; o >= 1; o /= 2) { float b[32]; for (unsigned l = 0; l < 32; l++) b[l] = a[l + o < 32 ? l + o : l]; for (unsigned l = 0; l < 32; l++) a[l] += b[l]; }
    return mutant == 6 ? a[0] : fmaf(a[0], gp, 0.0f);
}
/* double oracle of one row: y and the term scale sum |w x| */
static void oracle_row(const uint8_t *codes, const uint8_t *sc, double g, const float *x, uint32_t k, double *y, double *ts) {
    double s = 0, t = 0;
    for (uint32_t c = 0; c < k; c++) { double w = e2m1_d(nib(codes, c)) * e4m3_d(sc[c / 16u]) * g; s += w * x[c]; t += fabs(w * x[c]); }
    *y = s; *ts = t;
}

typedef struct { uint64_t w, s; uint32_t row0, cnt; uint64_t g; } Seg;
_Static_assert(sizeof(Seg) == OMEGA_W4GEMM_SEG_BYTES, "segment layout");

/* One simulator run. nseg weight tables of n rows x k (codes, scales, G'); counts[g] rows of x each; x rows: xrows
 * natural-order source rows. Returns 0 and the count of outputs differing from the emulation in *bad. */
typedef struct { uint32_t k, n, lb, mt, w, grid; uint32_t counts[4]; uint32_t nseg; } SimPt;
typedef struct { const uint8_t *codes, *sc; const float *gp, *x; uint32_t xrows; } SimData;
static int sim_run(const SimPt *p, const SimData *D, int mutant, int rowmap, int empty_at, uint64_t budget, char *msg, size_t msgn, size_t *bad, int *moved) {
    static BlackwellIRProgram prog;
    OmegaW4GemmSpec spec = { p->k, p->lb, p->mt, p->w, mutant, (uint32_t)rowmap };
    if (omega_blackwell_codegen_w4gemm_prog(&spec, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    const uint32_t n = p->n, k = p->k;
    uint32_t M = 0; for (uint32_t g = 0; g < p->nseg; g++) M += p->counts[g];
    uint32_t ids[64], trow[64]; { uint32_t i = 0; for (uint32_t g = 0; g < p->nseg; g++) for (uint32_t c = 0; c < p->counts[g]; c++, i++) { ids[i] = g; trow[i] = (i * 5u + 1u) % D->xrows; } }
    const size_t wrow = k / 2u, srow = k / 16u;
    float *xperm = malloc((size_t)D->xrows * k * 4);
    for (uint32_t r = 0; r < D->xrows; r++) omega_w4gemm_permute_x(D->x + (size_t)r * k, k, p->lb, xperm + (size_t)r * k);
    float *yexp = malloc((size_t)M * n * 4);
    for (uint32_t i = 0; i < M; i++) for (uint32_t r = 0; r < n; r++) {
        const size_t row = (size_t)ids[i] * n + r;
        yexp[(size_t)i * n + r] = emu_row(D->codes + row * wrow, D->sc + row * srow, D->gp[row], xperm + (size_t)trow[i] * k, k, p->lb, mutant == 3 || mutant == 4 ? 0 : mutant);
    }
    const size_t wbytes = (size_t)p->nseg * n * wrow, sbytes = (size_t)p->nseg * n * srow, gbytes = (size_t)p->nseg * n * 4;
    uint8_t *wd = aligned_alloc(16, (wbytes + 15) & ~(size_t)15); memcpy(wd, D->codes, wbytes);
    uint8_t *sd = aligned_alloc(16, (sbytes + 15) & ~(size_t)15); memcpy(sd, D->sc, sbytes);
    float *gd = aligned_alloc(16, (gbytes + 15) & ~(size_t)15); memcpy(gd, D->gp, gbytes);
    const uint32_t xrows = rowmap ? D->xrows : M + p->mt;
    float *xp = aligned_alloc(16, ((size_t)xrows * k * 4 + 15) & ~(size_t)15); memset(xp, 0, (size_t)xrows * k * 4);
    if (rowmap) memcpy(xp, xperm, (size_t)D->xrows * k * 4);
    else for (uint32_t i = 0; i < M; i++) memcpy(xp + (size_t)i * k, xperm + (size_t)trow[i] * k, (size_t)k * 4);
    uint32_t *xmap = aligned_alloc(16, 256 * 4), *ymap = aligned_alloc(16, 256 * 4);
    memset(xmap, 0, 256 * 4); memset(ymap, 0, 256 * 4);
    { uint32_t s = 7u * M + 3u; for (uint32_t i = 0; i < M; i++) { xmap[i] = trow[i]; ymap[i] = i; }
      if (rowmap) for (uint32_t i = M; i > 1; i--) { uint32_t j = lcg(&s) % i, t = ymap[i - 1]; ymap[i - 1] = ymap[j]; ymap[j] = t; } }
    float *yexp2 = malloc((size_t)M * n * 4);
    for (uint32_t i = 0; i < M; i++) memcpy(yexp2 + (size_t)ymap[i] * n, yexp + (size_t)i * n, (size_t)n * 4);
    Seg *seg = aligned_alloc(16, 64 * sizeof(Seg)); memset(seg, 0, 64 * sizeof(Seg));
    uint32_t segs = 0;
    { uint32_t r0 = 0;
      for (uint32_t g = 0; g <= p->nseg; g++) {
          if ((int)segs == empty_at) { seg[segs].w = (uintptr_t)wd; seg[segs].s = (uintptr_t)sd; seg[segs].g = (uintptr_t)gd; seg[segs].row0 = r0; seg[segs].cnt = 0; segs++; }
          if (g == p->nseg) break;
          seg[segs].w = (uintptr_t)(wd + (size_t)g * n * wrow); seg[segs].s = (uintptr_t)(sd + (size_t)g * n * srow); seg[segs].g = (uintptr_t)(gd + (size_t)g * n);
          seg[segs].row0 = r0; seg[segs].cnt = p->counts[g]; r0 += p->counts[g]; segs++;
      } }
    float *y = aligned_alloc(16, ((size_t)M * n * 4 + 15) & ~(size_t)15);
    const uint32_t shared = omega_w4gemm_shared_bytes(&spec);
    uint8_t *smem = aligned_alloc(16, shared);
    float *ycopy = NULL; int err = 0;
    for (int order = 0; order < 2 && !err; order++) {
        WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = 32u * p->w; s.order = order; s.smem = smem; s.smem_bytes = shared; s.step_budget = budget;
        for (size_t i = 0; i < (size_t)M * n; i++) y[i] = -999.0f;
        uint64_t A[5] = { (uintptr_t)seg, (uintptr_t)xp, (uintptr_t)y, (uintptr_t)xmap, (uintptr_t)ymap };
        const uint32_t at[5] = { OMEGA_GEMM_ARG_SEG, OMEGA_GEMM_ARG_X, OMEGA_GEMM_ARG_Y, OMEGA_GEMM_ARG_XMAP, OMEGA_GEMM_ARG_YMAP };
        for (unsigned i = 0; i < 5; i++) { s.cbank[at[i] / 4] = (uint32_t)A[i]; s.cbank[at[i] / 4 + 1] = (uint32_t)(A[i] >> 32); }
        s.cbank[OMEGA_GEMM_ARG_N / 4] = n; s.cbank[OMEGA_GEMM_ARG_STRIDE / 4] = p->grid * p->w;
        s.nranges = 8;
        s.ranges[0].lo = (uintptr_t)seg; s.ranges[0].hi = (uintptr_t)seg + 64 * sizeof(Seg);
        s.ranges[1].lo = (uintptr_t)wd; s.ranges[1].hi = (uintptr_t)wd + wbytes;
        s.ranges[2].lo = (uintptr_t)sd; s.ranges[2].hi = (uintptr_t)sd + sbytes;
        s.ranges[3].lo = (uintptr_t)xp; s.ranges[3].hi = (uintptr_t)xp + (size_t)xrows * k * 4;
        s.ranges[4].lo = (uintptr_t)y; s.ranges[4].hi = (uintptr_t)y + (size_t)M * n * 4;
        s.ranges[5].lo = (uintptr_t)xmap; s.ranges[5].hi = (uintptr_t)xmap + (M + p->mt) * 4;
        s.ranges[6].lo = (uintptr_t)ymap; s.ranges[6].hi = (uintptr_t)ymap + (M + p->mt) * 4;
        s.ranges[7].lo = (uintptr_t)gd; s.ranges[7].hi = (uintptr_t)gd + gbytes;
        err = ws_run(&s, p->grid, segs);
        if (err) { snprintf(msg, msgn, "order %d: %s (pc %d)", order, s.msg, s.err_pc); break; }
        size_t unwritten = 0; for (size_t i = 0; i < (size_t)M * n; i++) if (y[i] == -999.0f) unwritten++;
        if (unwritten) { err = -1; snprintf(msg, msgn, "%zu output words never written", unwritten); break; }
        if (order == 0) { ycopy = malloc((size_t)M * n * 4); memcpy(ycopy, y, (size_t)M * n * 4); }
        else if (memcmp(ycopy, y, (size_t)M * n * 4) != 0) { err = -1; snprintf(msg, msgn, "the fragment order changed the result"); }
    }
    if (!err) {
        size_t b = 0; for (size_t i = 0; i < (size_t)M * n; i++) if (memcmp(&yexp2[i], &y[i], 4) != 0) b++;
        *bad = b; if (moved) *moved = memcmp(yexp, y, (size_t)M * n * 4) != 0;
        snprintf(msg, msgn, "%zu of %zu outputs differ from the emulation%s", b, (size_t)M * n, b ? "" : " (bit-identical)");
    }
    free(ycopy); free(wd); free(sd); free(gd); free(xp); free(xperm); free(seg); free(y); free(smem); free(yexp); free(yexp2); free(xmap); free(ymap);
    return err;
}

typedef struct { uint8_t *codes, *sc; float *gp, *x; double *g; uint32_t rows, k, xrows; } Rand;
static void rand_make(Rand *R, uint32_t rows, uint32_t k, uint32_t xrows, uint32_t seed) {
    uint32_t s = seed;
    R->rows = rows; R->k = k; R->xrows = xrows;
    R->codes = malloc((size_t)rows * k / 2); R->sc = malloc((size_t)rows * k / 16); R->gp = malloc((size_t)rows * 4); R->g = malloc((size_t)rows * 8);
    R->x = malloc((size_t)xrows * k * 4);
    for (size_t i = 0; i < (size_t)rows * k / 2; i++) R->codes[i] = (uint8_t)(lcg(&s) >> 24);
    for (size_t i = 0; i < (size_t)rows * k / 16; i++) R->sc[i] = rand_scale(&s);
    for (uint32_t r = 0; r < rows; r++) { float g = ldexpf(frand(&s, 1.0f, 2.0f), -14 - (int)(lcg(&s) >> 29)); R->g[r] = g; R->gp[r] = g * 4611686018427387904.0f; }
    for (size_t i = 0; i < (size_t)xrows * k; i++) R->x[i] = frand(&s, -2.0f, 2.0f);
}
static void rand_free(Rand *R) { free(R->codes); free(R->sc); free(R->gp); free(R->g); free(R->x); }

static void host_only(void) {
    printf("host-only: emulation, permutation, budgets\n");
    { const uint32_t ks[2] = { 2048, 512 }, lbs[2] = { 16, 8 };
      for (int t = 0; t < 2; t++) {
          Rand R; rand_make(&R, 64, ks[t], 1, 11 + t); float *xp = malloc(ks[t] * 4); omega_w4gemm_permute_x(R.x, ks[t], lbs[t], xp);
          double worst = 0;
          for (uint32_t r = 0; r < 64; r++) {
              double yo, ts; oracle_row(R.codes + (size_t)r * ks[t] / 2, R.sc + (size_t)r * ks[t] / 16, R.g[r], R.x, ks[t], &yo, &ts);
              float e = emu_row(R.codes + (size_t)r * ks[t] / 2, R.sc + (size_t)r * ks[t] / 16, R.gp[r], xp, ks[t], lbs[t], 0);
              double d = fabs((double)e - yo) / (ts + 1e-300); if (d > worst) worst = d;
          }
          CHECK(worst < 1e-5, "k %u emulation vs the double oracle: worst %.2e of the term scale", ks[t], worst);
          printf("k %4u lane %2u: emulation vs double oracle, worst %.2e of the term scale\n", ks[t], lbs[t], worst);
          free(xp); rand_free(&R);
      } }
    { const uint32_t ks[2] = { 2048, 512 }, lbs[2] = { 16, 8 };
      for (int t = 0; t < 2; t++) {
          float x[2048], p[2048]; for (unsigned i = 0; i < ks[t]; i++) x[i] = (float)(i + 1);
          omega_w4gemm_permute_x(x, ks[t], lbs[t], p);
          char seen[2048] = { 0 }; int ok = 1;
          for (unsigned i = 0; i < ks[t]; i++) { double v = (double)p[i] / 18446744073709551616.0; if (v < 1 || v > ks[t] || seen[(int)v - 1]++) ok = 0; }
          CHECK(ok, "permutation of a %u-row (lane %u) is a bijection", ks[t], lbs[t]);
      }
      /* lane 16: column c = L*32 + q*4 + e of step 0 sits at (q*32 + L)*4 + e: c = 5*32 + 13 -> L 5, q 3, e 1 */
      float x[2048], p[2048]; for (unsigned i = 0; i < 2048; i++) x[i] = (float)(i + 1);
      omega_w4gemm_permute_x(x, 2048, 16, p);
      CHECK(p[(3 * 32 + 5) * 4 + 1] / 18446744073709551616.0f == x[173], "column 173 sits at lane 5 quad 3 element 1"); }
    struct { uint32_t k, lb, mt, w, rm; } shp[] = { { 2048, 16, 1, 16, 0 }, { 2048, 16, 4, 16, 0 }, { 2048, 16, 1, 16, 1 }, { 2048, 16, 2, 16, 1 },
        { 512, 8, 1, 16, 0 }, { 512, 8, 4, 16, 1 }, { 512, 8, 8, 16, 0 }, { 2048, 8, 8, 16, 0 }, { 4096, 16, 4, 8, 0 }, { 2048, 16, 1, 8, 0 } };
    for (unsigned i = 0; i < sizeof shp / sizeof shp[0]; i++) for (int m = 0; m <= OMEGA_W4GEMM_MUTANTS; m++) {
        OmegaW4GemmSpec s = { shp[i].k, shp[i].lb, shp[i].mt, shp[i].w, m, shp[i].rm }; uint32_t n = 0, gp = 0;
        int rc = omega_w4gemm_prog_stats(&s, &n, &gp);
        const uint32_t decl = ((gp + 2 + 7) / 8) * 8;
        CHECK(rc == 0 && n <= 1024 && 32u * s.warps * decl <= 65536u, "codegen k %u lb %u mt %u warps %u rowmap %u mutant %d: rc %d insns %u gprs %u", s.k, s.lane_bytes, s.mt, s.warps, s.rowmap, m, rc, n, gp);
        if (m == 0 && rc == 0) printf("codegen k %4u lane %2u mt %u warps %2u rowmap %u: %u instructions, %u registers (declared %u, register file %u of 65536), shared %u B\n", s.k, s.lane_bytes, s.mt, s.warps, s.rowmap, n, gp, decl, 32u * s.warps * decl, omega_w4gemm_shared_bytes(&s));
    }
    { OmegaW4GemmSpec s = { 2048, 16, 8, 16, 0, 0 }; CHECK(omega_w4gemm_prog_stats(&s, NULL, NULL) != 0, "mt 8 at lane 16 refused (32 KB of shared memory)"); }
    { OmegaW4GemmSpec s = { 512, 16, 1, 16, 0, 0 }; CHECK(omega_w4gemm_prog_stats(&s, NULL, NULL) != 0, "k 512 at lane 16 refused (half a step)"); }
    { OmegaW4GemmSpec s = { 2048, 16, 3, 16, 0, 0 }; CHECK(omega_w4gemm_prog_stats(&s, NULL, NULL) != 0, "mt 3 refused"); }
    { OmegaW4GemmSpec s = { 2048, 16, 4, 16, 0, 1 }; CHECK(omega_w4gemm_prog_stats(&s, NULL, NULL) != 0, "rowmap at lane 16 mt 4 refused (register allocation)"); }
    { OmegaW4GemmSpec s = { 2048, 16, 1, 16, 7, 0 }; CHECK(omega_w4gemm_prog_stats(&s, NULL, NULL) != 0, "mutant 7 refused"); }
    { OmegaW4GemmSpec s = { 512, 8, 8, 16, 0, 1 }; CHECK(omega_w4gemm_prog_stats(&s, NULL, NULL) != 0, "rowmap at mt 8 refused"); }
}

static void sim_battery(void) {
    const SimPt pts[] = {
        { 2048, 40, 16, 4, 16, 2, { 1, 5, 6 }, 3 },  /* lane 16, KS 2, two passes in segment 3, row clamping */
        { 512, 70, 8, 8, 16, 2, { 3, 9 }, 2 },        /* lane 8, KS 1 (down rows of 256 B), mt 8, two passes */
        { 4096, 20, 16, 4, 8, 1, { 2, 3 }, 2 },       /* KS 4: two iterations with the tail prefetch */
        { 2048, 24, 16, 1, 16, 2, { 1, 1, 1 }, 3 },   /* N = 1 decode form: mt 1 */
        { 1536, 18, 8, 2, 8, 1, { 2, 2 }, 2 },        /* lane 8 KS 3 (odd, unrolled) */
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        Rand R; rand_make(&R, pts[i].nseg * pts[i].n, pts[i].k, 16, 31 + (uint32_t)i);
        SimData D = { R.codes, R.sc, R.gp, R.x, 16 };
        char msg[300]; size_t bad = 1;
        int err = sim_run(&pts[i], &D, 0, 0, -1, 0, msg, sizeof msg, &bad, NULL);
        CHECK(!err && bad == 0, "sim k %u n %u lane %u mt %u warps %u: %s", pts[i].k, pts[i].n, pts[i].lb, pts[i].mt, pts[i].w, msg);
        printf("sim k %4u n %2u lane %2u mt %u warps %2u grid %u: %s %s\n", pts[i].k, pts[i].n, pts[i].lb, pts[i].mt, pts[i].w, pts[i].grid, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
        if (pts[i].mt <= 2 || (pts[i].lb == 8 && pts[i].mt <= 4)) { /* rowmap at lane 16 mt 4 exceeds the register allocator: refused */
            int moved = 0; bad = 1;
            err = sim_run(&pts[i], &D, 0, 1, 1, 0, msg, sizeof msg, &bad, &moved);
            CHECK(!err && bad == 0 && moved, "sim rowmap + empty segment k %u lane %u mt %u: %s (moved %d)", pts[i].k, pts[i].lb, pts[i].mt, msg, moved);
            printf("sim rowmap + empty segment at index 1, k %4u lane %2u mt %u: %s %s\n", pts[i].k, pts[i].lb, pts[i].mt, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
        }
        if (i < 2) for (int m = 1; m <= OMEGA_W4GEMM_MUTANTS; m++) {
            /* RED: each mutant against the CORRECT emulation (mutant 0 expectation), so a caught mutant differs */
            SimPt q = pts[i]; size_t b2 = 0; char m2[300];
            static BlackwellIRProgram dummy; (void)dummy;
            int e2;
            if (m == 4) { e2 = sim_run(&q, &D, 4, 0, 1, 2000000ull, m2, sizeof m2, &b2, NULL);
                CHECK(e2 != 0 && (strstr(m2, "step budget") || strstr(m2, "outside")), "sim mutant 4 (no empty-segment guard) must be caught: %s", m2);
                printf("sim lane %2u mutant 4: %s %s\n", q.lb, e2 ? "caught" : "NOT CAUGHT", m2); continue; }
            /* the mutant's own emulation must match the mutated kernel bit for bit (mutants 1, 2, 5, 6), and differ from mutant 0 */
            e2 = sim_run(&q, &D, m, 0, -1, 0, m2, sizeof m2, &b2, NULL);
            if (m == 3) { CHECK(!e2 && b2 != 0, "sim mutant 3 (neighbour x row) must be caught: %s", m2); printf("sim lane %2u mutant 3: %s %s\n", q.lb, !e2 && b2 ? "caught" : "NOT CAUGHT", m2); continue; }
            CHECK(!e2 && b2 == 0, "sim mutant %d matches its own emulation: %s", m, m2);
            /* and the mutant's output differs from the true result */
            Rand *Rp = &R; size_t diff = 0; float *xp = malloc(q.k * 4); omega_w4gemm_permute_x(Rp->x + 1u * q.k, q.k, q.lb, xp);
            for (uint32_t r = 0; r < q.n; r++) { float y0 = emu_row(Rp->codes + (size_t)r * q.k / 2, Rp->sc + (size_t)r * q.k / 16, Rp->gp[r], xp, q.k, q.lb, 0), y1 = emu_row(Rp->codes + (size_t)r * q.k / 2, Rp->sc + (size_t)r * q.k / 16, Rp->gp[r], xp, q.k, q.lb, m); if (memcmp(&y0, &y1, 4)) diff++; }
            free(xp);
            CHECK(diff == q.n, "mutant %d changes every row (%zu of %u)", m, diff, q.n);
            printf("sim lane %2u mutant %d: kernel = its emulation (%s), differs from the true result on %zu of %u rows: %s\n", q.lb, m, m2, diff, q.n, diff == q.n ? "caught" : "NOT CAUGHT");
        }
        rand_free(&R);
    }
}

/* .w4ref: u32 rows, u32 k, f32 global, codes rows*k/2, scales rows*k/16, x k f32, y rows f64, term scale rows f64 */
static void real_file(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { CHECK(0, "cannot open %s", path); return; }
    uint32_t hdr[2]; float g; int ok = fread(hdr, 4, 2, f) == 2 && fread(&g, 4, 1, f) == 1;
    const uint32_t rows = hdr[0], k = hdr[1];
    uint8_t *codes = malloc((size_t)rows * k / 2), *sc = malloc((size_t)rows * k / 16); float *x = malloc((size_t)k * 4);
    double *yo = malloc((size_t)rows * 8), *ts = malloc((size_t)rows * 8);
    ok = ok && fread(codes, 1, (size_t)rows * k / 2, f) == (size_t)rows * k / 2 && fread(sc, 1, (size_t)rows * k / 16, f) == (size_t)rows * k / 16
        && fread(x, 4, k, f) == k && fread(yo, 8, rows, f) == rows && fread(ts, 8, rows, f) == rows;
    fclose(f);
    CHECK(ok, "%s: short read", path);
    if (ok) {
        const uint32_t lb = k % 1024u == 0 ? 16u : 8u;
        float *gp = malloc((size_t)rows * 4); for (uint32_t r = 0; r < rows; r++) gp[r] = g * 4611686018427387904.0f;
        float *xp = malloc((size_t)k * 4); omega_w4gemm_permute_x(x, k, lb, xp);
        double worst = 0; for (uint32_t r = 0; r < rows; r++) { float e = emu_row(codes + (size_t)r * k / 2, sc + (size_t)r * k / 16, gp[r], xp, k, lb, 0); double d = fabs((double)e - yo[r]) / (ts[r] + 1e-300); if (d > worst) worst = d; }
        CHECK(worst < 1e-5, "%s: emulation vs the sovereign f64 oracle %.2e of the term scale", path, worst);
        SimPt p = { k, rows, lb, 1, 16, 8, { 1 }, 1 };
        SimData D = { codes, sc, gp, x, 1 };
        char msg[300]; size_t bad = 1;
        int err = sim_run(&p, &D, 0, 0, -1, 0, msg, sizeof msg, &bad, NULL);
        CHECK(!err && bad == 0, "%s: simulated kernel at N = 1: %s", path, msg);
        /* a row can hide the wrong-scale mutant only where every swapped scale equals the true one: at lane 16 the
         * mutant swaps blocks 2p and 2p + 1, so a row whose block pairs all carry equal bytes is blind; at lane 8 a lane
         * with L % 4 == 0 reads byte 0 of a shifted word (scale 0), so every row with a nonzero term is sensitive */
        size_t diff = 0, sens = 0, miss = 0;
        for (uint32_t r = 0; r < rows; r++) {
            const uint8_t *s = sc + (size_t)r * k / 16; int sensitive = lb == 8;
            for (uint32_t b = 0; b + 1 < k / 16 && !sensitive; b += 2) if (s[b] != s[b + 1]) sensitive = 1;
            float y0 = emu_row(codes + (size_t)r * k / 2, s, gp[r], xp, k, lb, 0), y2 = emu_row(codes + (size_t)r * k / 2, s, gp[r], xp, k, lb, 2);
            const int red = memcmp(&y0, &y2, 4) != 0;
            diff += red; sens += sensitive; miss += sensitive && !red;
        }
        size_t bad2 = 1; char msg2[300]; int err2 = sim_run(&p, &D, 2, 0, -1, 0, msg2, sizeof msg2, &bad2, NULL);
        CHECK(!err2 && bad2 == 0 && miss == 0 && diff == sens, "%s: wrong-scale mutant: kernel = its emulation (%s), red on %zu of %zu scale-sensitive rows (%u rows)", path, msg2, diff, sens, rows);
        printf("real %s: [%u][%u] lane %u: emulation vs f64 oracle %.2e of the term scale; kernel %s; wrong-scale mutant red on %zu of %u rows (%zu rows have a scale the mutant can change)\n", path, rows, k, lb, worst, msg, diff, rows, sens);
        free(gp); free(xp);
    }
    free(codes); free(sc); free(x); free(yo); free(ts);
}

int main(int argc, char **argv) {
    int host = 0, sim = 0, real = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) host = 1;
        else if (!strcmp(argv[i], "--sim")) sim = 1;
        else if (!strcmp(argv[i], "--real")) real = i + 1;
    }
    if (!host && !sim && !real) host = sim = 1;
    if (host) host_only();
    if (sim) sim_battery();
    if (real) for (int i = real; i < argc && argv[i][0] != '-'; i++) real_file(argv[i]);
    printf("gpu_w4gemm_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
