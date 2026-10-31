/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * q36_dec_sim_test: warp-simulator checks of the decode glue kernels at the Qwen3.6-35B-A3B MoE shapes (sc L306):
 *   RES COMBINE|NORM with 9 groups (8 routed experts + the shared expert in slot 8): x += sum_g w[g] * d[g], then
 *   y = RMSNorm(x) * nw * scale, against a double oracle; mutants 2 (expert 7 skipped) and 3 (shared slot skipped)
 *   must be caught. SWIGLU at width 512 over 9 groups (grid 2 x 9) against a double oracle; mutant 1 caught.
 *   GROUP (L312 v2, the stroke): N heads x 9 slots of expert ids to the grouped-GEMM segment tables and row maps, byte
 *   for byte against a host sort (heads 1, 2, 3, 8; all heads alike); mutants 1 (no tie-break), 2 (cnt short), 3 (empties
 *   unwritten) caught.
 *   COMBINE|NORM in the rows form (L356, one launch for N heads): bit-identical to the one-head program per head (1, 3,
 *   8 heads); mutant 3 caught.
 *   GATE in the rows form (L312 lever 7, the attention gate of N heads in one launch): bit-identical to the one-row program
 *   per head (1, 3, 8 heads); mutants 1 (gate dropped) and 2 (row 0's gate for every row) caught.
 * Host only (no device). Verdict line last: "q36_dec_sim_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_blackwell_codegen_dec.h"
#include "bw_warp_sim_k.h"
#include "omega_gpu_q36_graph.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }

#define H 2048u
#define G 9u
#define I 512u
static const float EPS = 1e-6f;
static void put64(WsSim *s, uint32_t off, const void *p) { uintptr_t v = (uintptr_t)p; s->cbank[off / 4] = (uint32_t)v; s->cbank[off / 4 + 1] = (uint32_t)((uint64_t)v >> 32); }

/* worst relative error of x and y against the oracle, or -1 on a simulator error */
static double combine_point(int mutant, uint32_t seed, float scale, char *msg, size_t n) {
    static BlackwellIRProgram prog;
    OmegaDecSpec sp = { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM, .mutant = mutant, .groups = G };
    if (omega_blackwell_codegen_dec_prog(&sp, &prog) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    float *x = aligned_alloc(16, H * 4), *d = aligned_alloc(16, (size_t)G * H * 4), *w = aligned_alloc(16, 64), *nw = aligned_alloc(16, H * 4), *y = aligned_alloc(16, H * 4);
    double *xr = malloc(H * sizeof(double)), *yr = malloc(H * sizeof(double));
    uint32_t s = seed;
    for (unsigned i = 0; i < H; i++) { x[i] = frand(&s, -2.0f, 2.0f); nw[i] = frand(&s, 0.8f, 1.2f); y[i] = -999.0f; }
    for (size_t i = 0; i < (size_t)G * H; i++) d[i] = frand(&s, -1.0f, 1.0f);
    for (unsigned g = 0; g < 16; g++) w[g] = g < G ? frand(&s, 0.05f, 0.4f) : 0.0f;
    for (unsigned i = 0; i < H; i++) { double m = 0; for (unsigned g = 0; g < G; g++) m += (double)w[g] * d[(size_t)g * H + i]; xr[i] = (double)x[i] + m; }
    double ss = 0; for (unsigned i = 0; i < H; i++) ss += xr[i] * xr[i];
    const double r = 1.0 / sqrt(ss / H + (double)EPS);
    for (unsigned i = 0; i < H; i++) yr[i] = xr[i] * r * nw[i] * scale;
    WsSim sim; memset(&sim, 0, sizeof sim);
    static uint8_t smem[OMEGA_DEC_SHARED_BYTES];
    sim.p = &prog; sim.threads = 256; sim.smem = smem; sim.smem_bytes = OMEGA_DEC_SHARED_BYTES;
    put64(&sim, OMEGA_DEC_RES_X, x); put64(&sim, OMEGA_DEC_RES_A, d); put64(&sim, OMEGA_DEC_RES_W, w); put64(&sim, OMEGA_DEC_RES_NW, nw); put64(&sim, OMEGA_DEC_RES_Y, y);
    memcpy(&sim.cbank[OMEGA_DEC_RES_EPS / 4], &EPS, 4); memcpy(&sim.cbank[OMEGA_DEC_RES_SCALE / 4], &scale, 4);
    sim.nranges = 5;
    sim.ranges[0] = (WsRange){ (uintptr_t)x, (uintptr_t)x + H * 4 }; sim.ranges[1] = (WsRange){ (uintptr_t)d, (uintptr_t)d + (size_t)G * H * 4 };
    sim.ranges[2] = (WsRange){ (uintptr_t)w, (uintptr_t)w + 64 }; sim.ranges[3] = (WsRange){ (uintptr_t)nw, (uintptr_t)nw + H * 4 };
    sim.ranges[4] = (WsRange){ (uintptr_t)y, (uintptr_t)y + H * 4 };
    double worst = -1;
    if (ws_run(&sim, 1, 1) != 0) snprintf(msg, n, "sim: %.150s (pc %d)", sim.msg, sim.err_pc);
    else {
        worst = 0;
        double xmax = 0, ymax = 0; for (unsigned i = 0; i < H; i++) { if (fabs(xr[i]) > xmax) xmax = fabs(xr[i]); if (fabs(yr[i]) > ymax) ymax = fabs(yr[i]); }
        for (unsigned i = 0; i < H; i++) {
            double ex = fabs((double)x[i] - xr[i]) / xmax, ey = fabs((double)y[i] - yr[i]) / ymax;
            if (ex > worst) worst = ex;
            if (ey > worst) worst = ey;
        }
        snprintf(msg, n, "x and y worst rel %.2e", worst);
    }
    free(x); free(d); free(w); free(nw); free(y); free(xr); free(yr);
    return worst;
}

/* L356: COMBINE|NORM in the rows form (one launch for N heads) against the one-head program run on each head's slice:
 * x and y must be bit-identical (the same arithmetic per row); the weights move by a 256-byte stride, the down outputs by
 * 9 x 2048 floats per head. Returns the number of differing floats, or -1 on a simulator error. */
static long rows_point(int mutant, uint32_t heads, uint32_t seed, char *msg, size_t n) {
    static BlackwellIRProgram p1, pr;
    const uint32_t WS = 256u; /* bytes between heads' weights (the router buffer stride is 256-aligned) */
    OmegaDecSpec s1 = { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM, .groups = G };
    OmegaDecSpec sr = { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM | OMEGA_DEC_RES_ROWS, .groups = G, .stride = WS, .mutant = mutant };
    if (omega_blackwell_codegen_dec_prog(&s1, &p1) != 0 || omega_blackwell_codegen_dec_prog(&sr, &pr) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    const size_t NX = (size_t)heads * H, ND = (size_t)heads * G * H, NW = (size_t)heads * WS / 4u;
    float *x0 = aligned_alloc(16, NX * 4), *x1 = aligned_alloc(16, NX * 4), *xr = aligned_alloc(16, NX * 4), *d = aligned_alloc(16, ND * 4);
    float *w = aligned_alloc(16, NW * 4), *nw = aligned_alloc(16, H * 4), *y1 = aligned_alloc(16, NX * 4), *yr = aligned_alloc(16, NX * 4);
    uint32_t s = seed;
    for (size_t i = 0; i < NX; i++) { x0[i] = frand(&s, -2.0f, 2.0f); y1[i] = yr[i] = -999.0f; }
    for (unsigned i = 0; i < H; i++) nw[i] = frand(&s, 0.8f, 1.2f);
    for (size_t i = 0; i < ND; i++) d[i] = frand(&s, -1.0f, 1.0f);
    for (size_t i = 0; i < NW; i++) w[i] = frand(&s, 0.05f, 0.4f);
    memcpy(x1, x0, NX * 4); memcpy(xr, x0, NX * 4);
    const float scale = 18446744073709551616.0f;
    static uint8_t smem[OMEGA_DEC_SHARED_BYTES];
    long bad = 0;
    for (uint32_t h = 0; h <= heads && bad >= 0; h++) { /* h < heads: the one-head program on head h; h == heads: the rows program */
        WsSim sim; memset(&sim, 0, sizeof sim);
        const int rows = h == heads;
        sim.p = rows ? &pr : &p1; sim.threads = 256; sim.smem = smem; sim.smem_bytes = OMEGA_DEC_SHARED_BYTES;
        float *xx = rows ? xr : x1 + (size_t)h * H, *yy = rows ? yr : y1 + (size_t)h * H;
        put64(&sim, OMEGA_DEC_RES_X, xx); put64(&sim, OMEGA_DEC_RES_A, rows ? d : d + (size_t)h * G * H);
        put64(&sim, OMEGA_DEC_RES_W, rows ? w : w + (size_t)h * WS / 4u); put64(&sim, OMEGA_DEC_RES_NW, nw); put64(&sim, OMEGA_DEC_RES_Y, yy);
        memcpy(&sim.cbank[OMEGA_DEC_RES_EPS / 4], &EPS, 4); memcpy(&sim.cbank[OMEGA_DEC_RES_SCALE / 4], &scale, 4);
        sim.nranges = 5;
        sim.ranges[0] = (WsRange){ (uintptr_t)xx, (uintptr_t)xx + (rows ? NX : H) * 4 };
        sim.ranges[1] = (WsRange){ (uintptr_t)d, (uintptr_t)d + ND * 4 };
        sim.ranges[2] = (WsRange){ (uintptr_t)w, (uintptr_t)w + NW * 4 };
        sim.ranges[3] = (WsRange){ (uintptr_t)nw, (uintptr_t)nw + H * 4 };
        sim.ranges[4] = (WsRange){ (uintptr_t)yy, (uintptr_t)yy + (rows ? NX : H) * 4 };
        if (ws_run(&sim, 1, rows ? heads : 1) != 0) { snprintf(msg, n, "sim (%s): %.150s (pc %d)", rows ? "rows" : "one head", sim.msg, sim.err_pc); bad = -1; }
    }
    if (bad == 0) {
        for (size_t i = 0; i < NX; i++) bad += memcmp(&x1[i], &xr[i], 4) != 0;
        for (size_t i = 0; i < NX; i++) bad += memcmp(&y1[i], &yr[i], 4) != 0;
        snprintf(msg, n, "%u heads: %ld of %zu x and y floats differ from the one-head program", heads, bad, 2 * NX);
    }
    free(x0); free(x1); free(xr); free(d); free(w); free(nw); free(y1); free(yr);
    return bad;
}

/* L312 lever 7: GATE in the rows form (one launch for N heads) against the one-row program run on each head's slice: out must
 * be bit-identical per row; a and out move by width (= count) floats per row, g by the attention mix row (9216 floats). Returns the
 * number of differing floats, or -1 on a simulator error. */
static long gate_rows_point(int mutant, uint32_t heads, uint32_t seed, char *msg, size_t n) {
    static BlackwellIRProgram p1, pr;
    const uint32_t C = 4096u, GS = 9216u; /* MIXER_OUT, ATT_MIX */
    OmegaDecSpec s1 = { .kind = OMEGA_DEC_GATE };
    OmegaDecSpec sr = { .kind = OMEGA_DEC_GATE, .mode = 1, .width = C, .stride = GS, .mutant = mutant }; /* L369's rows layout */
    if (omega_blackwell_codegen_dec_prog(&s1, &p1) != 0 || omega_blackwell_codegen_dec_prog(&sr, &pr) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    const size_t NA = (size_t)heads * C, NG = (size_t)heads * GS;
    float *a = aligned_alloc(16, NA * 4), *o1 = aligned_alloc(16, NA * 4), *orr = aligned_alloc(16, NA * 4), *gg = aligned_alloc(16, NG * 4);
    uint32_t s = seed;
    for (size_t i = 0; i < NA; i++) { a[i] = frand(&s, -2.0f, 2.0f); o1[i] = orr[i] = -999.0f; }
    for (size_t i = 0; i < NG; i++) gg[i] = frand(&s, -4.0f, 4.0f);
    static uint8_t smem[OMEGA_DEC_SHARED_BYTES];
    long bad = 0;
    for (uint32_t h = 0; h <= heads && bad >= 0; h++) { /* h < heads: the one-row program on head h; h == heads: the rows program */
        WsSim sim; memset(&sim, 0, sizeof sim);
        const int rows = h == heads;
        sim.p = rows ? &pr : &p1; sim.threads = 256; sim.smem = smem; sim.smem_bytes = OMEGA_DEC_SHARED_BYTES;
        float *aa = rows ? a : a + (size_t)h * C, *oo = rows ? orr : o1 + (size_t)h * C, *g0 = rows ? gg : gg + (size_t)h * GS;
        put64(&sim, OMEGA_DEC_GT_A, aa); put64(&sim, OMEGA_DEC_GT_G, g0 + 5120u); put64(&sim, OMEGA_DEC_GT_OUT, oo);
        sim.cbank[OMEGA_DEC_GT_N / 4] = C;
        sim.nranges = 3;
        sim.ranges[0] = (WsRange){ (uintptr_t)aa, (uintptr_t)aa + (rows ? NA : C) * 4 };
        sim.ranges[1] = (WsRange){ (uintptr_t)gg, (uintptr_t)gg + NG * 4 };
        sim.ranges[2] = (WsRange){ (uintptr_t)oo, (uintptr_t)oo + (rows ? NA : C) * 4 };
        if (ws_run(&sim, C / 256u, rows ? heads : 1) != 0) { snprintf(msg, n, "sim (%s): %.150s (pc %d)", rows ? "rows" : "one row", sim.msg, sim.err_pc); bad = -1; }
    }
    if (bad == 0) {
        for (size_t i = 0; i < NA; i++) bad += memcmp(&o1[i], &orr[i], 4) != 0;
        snprintf(msg, n, "%u heads: %ld of %zu gate outputs differ from the one-row program", heads, bad, NA);
    }
    free(a); free(o1); free(orr); free(gg);
    return bad;
}

static double swiglu_point(int mutant, uint32_t seed, char *msg, size_t n) {
    static BlackwellIRProgram prog;
    OmegaDecSpec sp = { .kind = OMEGA_DEC_SWIGLU, .mutant = mutant, .width = I };
    if (omega_blackwell_codegen_dec_prog(&sp, &prog) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    float *gu = aligned_alloc(16, (size_t)G * 2 * I * 4), *act = aligned_alloc(16, (size_t)G * I * 4);
    const float scale = 18446744073709551616.0f; /* 2^64: the FP8 GEMV x' row */
    uint32_t s = seed;
    for (size_t i = 0; i < (size_t)G * 2 * I; i++) gu[i] = frand(&s, -4.0f, 4.0f);
    for (size_t i = 0; i < (size_t)G * I; i++) act[i] = -999.0f;
    WsSim sim; memset(&sim, 0, sizeof sim);
    static uint8_t smem[OMEGA_DEC_SHARED_BYTES];
    sim.p = &prog; sim.threads = 256; sim.smem = smem; sim.smem_bytes = OMEGA_DEC_SHARED_BYTES;
    put64(&sim, OMEGA_DEC_SW_GU, gu); put64(&sim, OMEGA_DEC_SW_ACT, act); memcpy(&sim.cbank[OMEGA_DEC_SW_SCALE / 4], &scale, 4);
    sim.nranges = 2;
    sim.ranges[0] = (WsRange){ (uintptr_t)gu, (uintptr_t)gu + (size_t)G * 2 * I * 4 }; sim.ranges[1] = (WsRange){ (uintptr_t)act, (uintptr_t)act + (size_t)G * I * 4 };
    double worst = -1;
    if (ws_run(&sim, I / 256u, G) != 0) snprintf(msg, n, "sim: %.150s (pc %d)", sim.msg, sim.err_pc);
    else {
        worst = 0;
        for (unsigned g = 0; g < G; g++) for (unsigned c = 0; c < I; c++) {
            const double v = gu[(size_t)g * 2 * I + c], u = gu[(size_t)g * 2 * I + I + c];
            const double ref = v / (1.0 + exp(-v)) * u * scale, got = act[(size_t)g * I + c];
            const double e = fabs(got - ref) / (fabs(ref) + 1e-3 * scale);
            if (e > worst) worst = e;
        }
        snprintf(msg, n, "act worst rel %.2e", worst);
    }
    free(gu); free(act);
    return worst;
}

/* GROUP (L312 v2): N heads x 9 slots of expert ids -> the two segment tables and the two maps, byte for byte against a
 * host sort. all_same: every head picks the same 8 experts (9 segments of cnt N: the heaviest overlap). */
typedef struct { uint64_t w, s; uint32_t row0, cnt, pad[2]; } Seg;
static uint64_t fake_w(uint32_t id) { return 0x100000ull + (uint64_t)id * 0x1000ull; }
static uint64_t fake_s(uint32_t id) { return 0x900000ull + (uint64_t)id * 0x100ull; }
static int group_point(int mutant, uint32_t heads, int all_same, uint32_t seed, char *msg, size_t n) {
    static BlackwellIRProgram prog;
    OmegaDecSpec sp = { .kind = OMEGA_DEC_GROUP, .mutant = mutant, .groups = G };
    if (omega_blackwell_codegen_dec_prog(&sp, &prog) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    const uint32_t R = heads * G, STRIDE = 4096u, PAD = OMEGA_DEC_GR_PAD;
    uint8_t *lb = aligned_alloc(16, (size_t)heads * STRIDE); memset(lb, 0, (size_t)heads * STRIDE);
    Seg *sgu = aligned_alloc(16, (size_t)R * sizeof(Seg)), *sdn = aligned_alloc(16, (size_t)R * sizeof(Seg));
    uint32_t *xm = aligned_alloc(16, (size_t)(R + PAD) * 4), *ym = aligned_alloc(16, (size_t)(R + PAD) * 4);
    memset(sgu, 0xEE, (size_t)R * sizeof(Seg)); memset(sdn, 0xEE, (size_t)R * sizeof(Seg)); memset(xm, 0xEE, (size_t)(R + PAD) * 4); memset(ym, 0xEE, (size_t)(R + PAD) * 4);
    uint32_t s = seed, ids[16 * G];
    for (uint32_t h = 0; h < heads; h++) {
        uint32_t *row = ids + h * G;
        if (all_same && h) memcpy(row, ids, G * 4);
        else {
            for (uint32_t j = 0; j < 8; j++) { int dup; do { row[j] = lcg(&s) % 256u; dup = 0; for (uint32_t q = 0; q < j; q++) if (row[q] == row[j]) dup = 1; } while (dup); }
            row[8] = 256u; /* the shared expert */
        }
        uint32_t *idp = (uint32_t *)(lb + (size_t)h * STRIDE); uint64_t *tg = (uint64_t *)(lb + (size_t)h * STRIDE + 1024), *td = (uint64_t *)(lb + (size_t)h * STRIDE + 2048);
        for (uint32_t j = 0; j < G; j++) { idp[j] = row[j]; tg[2 * j] = fake_w(row[j]); tg[2 * j + 1] = fake_s(row[j]); td[2 * j] = fake_w(row[j]) + 7u; td[2 * j + 1] = fake_s(row[j]) + 3u; }
    }
    /* host reference */
    Seg *egu = calloc(R, sizeof(Seg)), *edn = calloc(R, sizeof(Seg)); uint32_t *exm = calloc(R + PAD, 4), *eym = calloc(R + PAD, 4);
    uint32_t rank[16 * G];
    for (uint32_t t = 0; t < R; t++) { uint32_t p = 0; for (uint32_t q = 0; q < R; q++) if (ids[q] * 128u + q < ids[t] * 128u + t) p++; rank[t] = p; exm[p] = t / G; eym[p] = t; }
    uint32_t D = 0;
    for (uint32_t t = 0; t < R; t++) {
        int first = 1; uint32_t cnt = 0, below = 0;
        for (uint32_t q = 0; q < R; q++) { if (ids[q] == ids[t]) { cnt++; if (q < t) first = 0; } }
        if (!first) continue;
        for (uint32_t q = 0; q < R; q++) { int fq = 1; for (uint32_t z = 0; z < q; z++) if (ids[z] == ids[q]) fq = 0; if (fq && ids[q] < ids[t]) below++; }
        egu[below] = (Seg){ fake_w(ids[t]), fake_s(ids[t]), rank[t], cnt, { 0, 0 } };
        edn[below] = (Seg){ fake_w(ids[t]) + 7u, fake_s(ids[t]) + 3u, rank[t], cnt, { 0, 0 } };
        D++;
    }
    WsSim sim; memset(&sim, 0, sizeof sim);
    static uint8_t smem[OMEGA_DEC_GR_SHARED_BYTES];
    sim.p = &prog; sim.threads = 256; sim.smem = smem; sim.smem_bytes = OMEGA_DEC_GR_SHARED_BYTES;
    put64(&sim, OMEGA_DEC_GR_IDS, lb); put64(&sim, OMEGA_DEC_GR_TGU, lb + 1024); put64(&sim, OMEGA_DEC_GR_TDN, lb + 2048);
    put64(&sim, OMEGA_DEC_GR_SGU, sgu); put64(&sim, OMEGA_DEC_GR_SDN, sdn); put64(&sim, OMEGA_DEC_GR_XMAP, xm); put64(&sim, OMEGA_DEC_GR_YMAP, ym);
    sim.cbank[OMEGA_DEC_GR_STRIDE / 4] = STRIDE; sim.cbank[OMEGA_DEC_GR_HEADS / 4] = heads;
    sim.nranges = 5;
    sim.ranges[0] = (WsRange){ (uintptr_t)lb, (uintptr_t)lb + (size_t)heads * STRIDE };
    sim.ranges[1] = (WsRange){ (uintptr_t)sgu, (uintptr_t)sgu + (size_t)R * sizeof(Seg) }; sim.ranges[2] = (WsRange){ (uintptr_t)sdn, (uintptr_t)sdn + (size_t)R * sizeof(Seg) };
    sim.ranges[3] = (WsRange){ (uintptr_t)xm, (uintptr_t)xm + (size_t)(R + PAD) * 4 }; sim.ranges[4] = (WsRange){ (uintptr_t)ym, (uintptr_t)ym + (size_t)(R + PAD) * 4 };
    int bad = -1;
    if (ws_run(&sim, 1, 1) != 0) snprintf(msg, n, "sim: %.150s (pc %d)", sim.msg, sim.err_pc);
    else {
        int bg = memcmp(sgu, egu, (size_t)R * sizeof(Seg)) != 0, bd = memcmp(sdn, edn, (size_t)R * sizeof(Seg)) != 0;
        int bx = memcmp(xm, exm, (size_t)(R + PAD) * 4) != 0, by = memcmp(ym, eym, (size_t)(R + PAD) * 4) != 0;
        bad = bg + bd + bx + by;
        snprintf(msg, n, "heads %u rows %u distinct %u: seg_gu %s, seg_dn %s, xmap %s, ymap %s", heads, R, D, bg ? "WRONG" : "ok", bd ? "WRONG" : "ok", bx ? "WRONG" : "ok", by ? "WRONG" : "ok");
    }
    free(lb); free(sgu); free(sdn); free(xm); free(ym); free(egu); free(edn); free(exm); free(eym);
    return bad;
}

int main(void) {
    char msg[300];
    for (uint32_t gg = 4; gg <= 8; gg += 4) /* lever 1/6 lm_head groups: every N heads covered once, no row past N-1 */
        for (uint32_t n = 1; n <= 8; n++) {
            uint32_t cov = 0, bad = 0, ng = 0;
            for (uint32_t h0 = 0; h0 < n; h0 += gg) {
                const uint32_t xb = omega_gpu_q36_head_group(n, gg, h0);
                if (xb == 0 || xb > gg || h0 + xb > n || h0 != cov) bad++;
                cov = h0 + xb; ng++;
            }
            CHECK(!bad && cov == n && ng == (n + gg - 1) / gg && omega_gpu_q36_head_group(n, gg, n) == 0,
                  "lm_head groups N %u G %u: %u groups cover %u heads, %u bad", n, gg, ng, cov, bad);
        }
    printf("lm_head group plan: N 1..8 x G {4,8} checked\n");
    { OmegaDecSpec sp = { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_NORM, .groups = 9 }; CHECK(omega_dec_prog_stats(&sp, NULL, NULL) != 0, "groups 9 without COMBINE refused"); }
    { OmegaDecSpec sp = { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_COMBINE, .groups = 7 }; CHECK(omega_dec_prog_stats(&sp, NULL, NULL) != 0, "groups 7 refused"); }
    { OmegaDecSpec sp = { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_COMBINE, .mutant = 3 }; CHECK(omega_dec_prog_stats(&sp, NULL, NULL) != 0, "mutant 3 without the shared slot refused"); }
    const struct { uint32_t seed; float scale; } pts[] = { { 1, 1.0f }, { 2, 18446744073709551616.0f }, { 3, 1.0f } };
    for (unsigned i = 0; i < 3; i++) {
        double w = combine_point(0, pts[i].seed, pts[i].scale, msg, sizeof msg);
        CHECK(w >= 0 && w < 2e-6, "COMBINE|NORM 9 groups seed %u: %s", pts[i].seed, msg);
        printf("COMBINE|NORM 9 groups seed %u scale %s: %s\n", pts[i].seed, pts[i].scale > 2.0f ? "2^64" : "1", msg);
    }
    for (int m = 2; m <= 3; m++) {
        double w = combine_point(m, 7, 1.0f, msg, sizeof msg);
        CHECK(w > 1e-3, "COMBINE mutant %d must be caught: %s", m, msg);
        printf("COMBINE mutant %d: %s (%s)\n", m, w > 1e-3 ? "caught" : "NOT CAUGHT", msg);
    }
    for (uint32_t seed = 11; seed <= 12; seed++) {
        double w = swiglu_point(0, seed, msg, sizeof msg);
        CHECK(w >= 0 && w < 1e-5, "SWIGLU 512 x 9 seed %u: %s", seed, msg);
        printf("SWIGLU width 512 x 9 groups seed %u: %s\n", seed, msg);
    }
    { double w = swiglu_point(1, 13, msg, sizeof msg); CHECK(w > 1e-3, "SWIGLU mutant 1 must be caught: %s", msg); printf("SWIGLU mutant 1: %s (%s)\n", w > 1e-3 ? "caught" : "NOT CAUGHT", msg); }
    { OmegaDecSpec sp = { .kind = OMEGA_DEC_GROUP, .groups = 17 }; CHECK(omega_dec_prog_stats(&sp, NULL, NULL) != 0, "GROUP 17 slots refused"); }
    { OmegaDecSpec sp = { .kind = OMEGA_DEC_GROUP, .groups = G }; uint32_t ins = 0, gp = 0; CHECK(omega_dec_prog_stats(&sp, &ins, &gp) == 0, "GROUP codegen"); printf("GROUP 9 slots: %u instructions, %u registers\n", ins, gp); }
    const struct { uint32_t heads; int same; uint32_t seed; } gpts[] = { { 8, 0, 21 }, { 8, 1, 22 }, { 1, 0, 23 }, { 3, 0, 24 }, { 8, 0, 25 }, { 2, 1, 26 } };
    for (unsigned i = 0; i < sizeof gpts / sizeof gpts[0]; i++) {
        int b = group_point(0, gpts[i].heads, gpts[i].same, gpts[i].seed, msg, sizeof msg);
        CHECK(b == 0, "GROUP heads %u%s seed %u: %s", gpts[i].heads, gpts[i].same ? " all-same" : "", gpts[i].seed, msg);
        printf("GROUP%s seed %u: %s\n", gpts[i].same ? " all-same" : "", gpts[i].seed, msg);
    }
    for (int m = 1; m <= 3; m++) {
        int b = group_point(m, 8, 0, 27, msg, sizeof msg);
        CHECK(b != 0, "GROUP mutant %d must be caught: %s", m, msg);
        printf("GROUP mutant %d: %s (%s)\n", m, b != 0 ? "caught" : "NOT CAUGHT", msg);
    }
    { OmegaDecSpec sp = { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM | OMEGA_DEC_RES_ROWS, .groups = G, .stride = 8 };
      CHECK(omega_dec_prog_stats(&sp, NULL, NULL) != 0, "COMBINE rows with a weights stride not a multiple of 16 refused"); }
    for (uint32_t hh = 1; hh <= 8; hh += hh < 3 ? 2 : 5) {
        long b = rows_point(0, hh, 40 + hh, msg, sizeof msg);
        CHECK(b == 0, "COMBINE|NORM rows: %s", msg);
        printf("COMBINE|NORM rows (L356): %s\n", msg);
    }
    { long b = rows_point(3, 8, 49, msg, sizeof msg); CHECK(b > 0, "COMBINE rows mutant 3 must be caught: %s", msg); printf("COMBINE rows mutant 3: %s (%s)\n", b > 0 ? "caught" : "NOT CAUGHT", msg); }
    { OmegaDecSpec sp = { .kind = OMEGA_DEC_GATE, .mode = 1 }; CHECK(omega_dec_prog_stats(&sp, NULL, NULL) != 0, "GATE rows without a g stride refused"); }
    { OmegaDecSpec sp = { .kind = OMEGA_DEC_GATE, .mode = 2, .stride = 9216 }; CHECK(omega_dec_prog_stats(&sp, NULL, NULL) != 0, "GATE mode 2 refused"); }
    for (uint32_t hh = 1; hh <= 8; hh += hh < 3 ? 2 : 5) {
        long b = gate_rows_point(0, hh, 60 + hh, msg, sizeof msg);
        CHECK(b == 0, "GATE rows: %s", msg);
        printf("GATE rows (L312 lever 7): %s\n", msg);
    }
    for (int m = 1; m <= 2; m++) { long b = gate_rows_point(m, 3, 70 + m, msg, sizeof msg); CHECK(b > 0, "GATE rows mutant %d must be caught: %s", m, msg); printf("GATE rows mutant %d: %s (%s)\n", m, b > 0 ? "caught" : "NOT CAUGHT", msg); }
    printf("q36_dec_sim_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
