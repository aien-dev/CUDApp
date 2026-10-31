/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * gpu_fp8_head_test: the all-heads FP8 lm_head kernel (omega_blackwell_codegen_fp8head.h, aien-dev/CUDApp#217,
 * P-DRA-4 DRA-69). Host and warp-simulator only; nothing here is a chip result.
 *
 *   ./gpu_fp8_head_test --host-only   spec refusals, instruction / register budgets for every accepted shape
 *   ./gpu_fp8_head_test --sim         warp simulator:
 *       1. bitwise parity: for xb 1..8 and rows 1, 2, row b of the fp8head output is memcmp-identical to the
 *          legacy single-x FP8 GEMV (omega_blackwell_codegen_moe.c, lane_bytes 16) run on x row b alone;
 *       2. oracle: every output within TOL of the double oracle (omega_fp8_gemv_oracle), relative to sum |terms|;
 *       3. mutants 1 (scale pointer stuck at k-block 0), 2 (neighbouring 128-row scale block), 3 (every x row reads
 *          row 0): each must break the oracle check and the bitwise parity.
 *   FP8HEAD_TEST_FORCE_MUTANT=m       negative control of the test itself: builds every "good" kernel of the
 *                                     battery with mutant m, so the parity and oracle checks must go red.
 * Exit 0 only when every check passes. -ffp-contract=off keeps the double oracle free of FMA.
 */
#include "omega_blackwell_codegen_fp8head.h"
#include "omega_blackwell_codegen_moe.h"
#include "omega_gpu_fp8_api.h"
#include "bw_warp_sim.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-5
static int g_checks, g_failed, g_force_mutant;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }
static uint8_t rand_e4m3(uint32_t *s) { uint8_t b = (uint8_t)(lcg(s) >> 24); if (omega_fp8_e4m3_is_nan(b)) b = (uint8_t)(b - 1); return b; }

typedef struct { uint32_t n, k; uint8_t *w; uint16_t *s; float *sf; } Head;
static void head_make(Head *H, uint32_t n, uint32_t k, uint32_t seed) {
    H->n = n; H->k = k;
    size_t wb = (size_t)n * k, sc = (size_t)(n / 128) * (k / 128);
    H->w = aligned_alloc(64, (wb + 63) & ~(size_t)63); H->s = malloc(sc * 2);
    H->sf = aligned_alloc(16, (sc * 4 + 15) & ~(size_t)15);
    uint32_t st = seed;
    for (size_t i = 0; i < wb; i++) H->w[i] = rand_e4m3(&st);
    for (size_t i = 0; i < sc; i++) H->s[i] = omega_fp32_to_bf16(frand(&st, 0.001f, 0.1f) * ((lcg(&st) & 1) ? 1.0f : -1.0f));
    if (omega_fp8_pack_scales(H->s, sc, H->sf, 0) != 0) { printf("FAIL: scale pack\n"); exit(2); }
}
static void head_free(Head *H) { free(H->w); free(H->s); free(H->sf); }

/* fp8head on the simulator, both fragment orders (must agree). xp: xb packed rows (x * 2^64). y: xb rows of n. */
static int sim_head(const OmegaFp8HeadSpec *spec, const Head *H, const float *xp, float *y, uint32_t ctas, char *msg, size_t ml) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_fp8head_prog(spec, &prog) != 0) { snprintf(msg, ml, "codegen refused"); return -1; }
    const uint32_t n = H->n, k = H->k, RG = n / spec->rows, xb = spec->xb;
    size_t ny = (size_t)xb * n;
    float *yd = aligned_alloc(16, (ny * 4 + 15) & ~(size_t)15), *first = malloc(ny * 4);
    uint32_t grid_x = (RG + spec->warps - 1) / spec->warps; if (grid_x > ctas) grid_x = ctas; if (!grid_x) grid_x = 1;
    int err = 0;
    for (int run = 0; run < 2 && !err; run++) {
        WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = 32 * spec->warps; s.order = run;
        uint64_t wa = (uintptr_t)H->w, sa = (uintptr_t)H->sf, xa = (uintptr_t)xp, ya = (uintptr_t)yd;
        s.cbank[OMEGA_FP8HEAD_ARG_W / 4] = (uint32_t)wa; s.cbank[OMEGA_FP8HEAD_ARG_W / 4 + 1] = (uint32_t)(wa >> 32);
        s.cbank[OMEGA_FP8HEAD_ARG_S / 4] = (uint32_t)sa; s.cbank[OMEGA_FP8HEAD_ARG_S / 4 + 1] = (uint32_t)(sa >> 32);
        s.cbank[OMEGA_FP8HEAD_ARG_X / 4] = (uint32_t)xa; s.cbank[OMEGA_FP8HEAD_ARG_X / 4 + 1] = (uint32_t)(xa >> 32);
        s.cbank[OMEGA_FP8HEAD_ARG_Y / 4] = (uint32_t)ya; s.cbank[OMEGA_FP8HEAD_ARG_Y / 4 + 1] = (uint32_t)(ya >> 32);
        s.cbank[OMEGA_FP8HEAD_ARG_RG / 4] = RG; s.cbank[OMEGA_FP8HEAD_ARG_YSTRIDE / 4] = n * 4;
        s.cbank[OMEGA_FP8HEAD_ARG_STRIDE / 4] = grid_x * spec->warps;
        s.nranges = 4; /* every global access must land in a declared buffer */
        s.ranges[0].lo = wa; s.ranges[0].hi = wa + (size_t)n * k;
        s.ranges[1].lo = sa; s.ranges[1].hi = sa + (size_t)(n / 128) * (k / 128) * 4;
        s.ranges[2].lo = xa; s.ranges[2].hi = xa + (size_t)xb * k * 4;
        s.ranges[3].lo = ya; s.ranges[3].hi = ya + ny * 4;
        for (size_t i = 0; i < ny; i++) yd[i] = -999.0f;
        err = ws_run(&s, grid_x, 1);
        if (err) { snprintf(msg, ml, "run %d: %s", run, s.msg); break; }
        size_t unwritten = 0; for (size_t i = 0; i < ny; i++) unwritten += yd[i] == -999.0f;
        if (unwritten) { err = -1; snprintf(msg, ml, "%zu output words never written", unwritten); break; }
        if (run == 0) memcpy(first, yd, ny * 4);
        else if (memcmp(first, yd, ny * 4)) { err = -1; snprintf(msg, ml, "fragment order changed the result"); }
    }
    if (!err) memcpy(y, yd, ny * 4);
    free(yd); free(first);
    return err;
}

/* legacy single-x FP8 GEMV (lane_bytes 16, one expert) for one packed x row */
static int sim_legacy(const Head *H, uint32_t rows, const float *xp, float *y, char *msg, size_t ml) {
    static BlackwellIRProgram prog;
    OmegaFp8GemvSpec sp = { .k = H->k, .lane_bytes = 16, .rows = rows, .warps = 8 };
    if (omega_blackwell_codegen_fp8_gemv_prog(&sp, &prog) != 0) { snprintf(msg, ml, "legacy codegen refused"); return -1; }
    const uint32_t n = H->n, RG = n / rows;
    uint64_t tab[2] = { (uintptr_t)H->w, (uintptr_t)H->sf };
    float *yd = aligned_alloc(16, ((size_t)n * 4 + 15) & ~(size_t)15);
    uint32_t grid_x = (RG + 7) / 8; if (grid_x > 4) grid_x = 4;
    WsSim s; memset(&s, 0, sizeof s);
    static uint8_t smem[OMEGA_FP8_MAX_K * 4 + 4096];
    s.p = &prog; s.threads = 256; s.smem = smem; s.smem_bytes = omega_fp8_gemv_shared_bytes(&sp);
    uint64_t ta = (uintptr_t)tab, xa = (uintptr_t)xp, ya = (uintptr_t)yd;
    s.cbank[OMEGA_FP8_ARG_TAB / 4] = (uint32_t)ta; s.cbank[OMEGA_FP8_ARG_TAB / 4 + 1] = (uint32_t)(ta >> 32);
    s.cbank[OMEGA_FP8_ARG_X / 4] = (uint32_t)xa; s.cbank[OMEGA_FP8_ARG_X / 4 + 1] = (uint32_t)(xa >> 32);
    s.cbank[OMEGA_FP8_ARG_Y / 4] = (uint32_t)ya; s.cbank[OMEGA_FP8_ARG_Y / 4 + 1] = (uint32_t)(ya >> 32);
    s.cbank[OMEGA_FP8_ARG_RG / 4] = RG; s.cbank[OMEGA_FP8_ARG_XSTRIDE / 4] = 0;
    s.cbank[OMEGA_FP8_ARG_YSTRIDE / 4] = n * 4; s.cbank[OMEGA_FP8_ARG_STRIDE / 4] = grid_x * 8;
    s.nranges = 5;
    s.ranges[0].lo = (uintptr_t)H->w; s.ranges[0].hi = (uintptr_t)H->w + (size_t)n * H->k;
    s.ranges[1].lo = (uintptr_t)H->sf; s.ranges[1].hi = (uintptr_t)H->sf + (size_t)(n / 128) * (H->k / 128) * 4;
    s.ranges[2].lo = xa; s.ranges[2].hi = xa + (size_t)H->k * 4;
    s.ranges[3].lo = ya; s.ranges[3].hi = ya + (size_t)n * 4;
    s.ranges[4].lo = ta; s.ranges[4].hi = ta + sizeof tab;
    int err = ws_run(&s, grid_x, 1);
    if (err) snprintf(msg, ml, "legacy: %s", s.msg); else memcpy(y, yd, (size_t)n * 4);
    free(yd);
    return err;
}

/* worst relative error against the double oracle; *bad = outputs at or over TOL (or NaN) */
static double oracle_check(const Head *H, uint32_t xb, const float *x, const float *y, size_t *bad) {
    double worst = 0; size_t nb = 0;
    double *ref = malloc(H->n * sizeof *ref), *mag = malloc(H->n * sizeof *mag);
    for (uint32_t b = 0; b < xb; b++) {
        omega_fp8_gemv_oracle(H->w, H->s, H->n, H->k, x + (size_t)b * H->k, ref, mag);
        for (uint32_t r = 0; r < H->n; r++) {
            float v = y[(size_t)b * H->n + r];
            double rel = fabs(ref[r] - (double)v) / (mag[r] > 1e-30 ? mag[r] : 1e-30);
            if (rel > worst) worst = rel;
            nb += rel >= TOL || isnan(v);
        }
    }
    free(ref); free(mag);
    *bad = nb;
    return worst;
}

static void host_only(void) {
    uint32_t built = 0, max_insns = 0, max_gprs = 0;
    for (uint32_t rows = 1; rows <= 2; rows++)
        for (uint32_t xb = 1; xb <= OMEGA_FP8HEAD_MAX_XB; xb++)
            for (int m = 0; m <= 3; m++) {
                OmegaFp8HeadSpec sp = { .k = 2048, .rows = rows, .warps = 8, .xb = xb, .mutant = m };
                uint32_t ni = 0, ng = 0;
                int rc = omega_fp8head_prog_stats(&sp, &ni, &ng);
                CHECK(rc == 0 && ni <= BW_MAX_IR_INSNS, "k 2048 rows %u xb %u mutant %d: rc %d, %u insns", rows, xb, m, rc, ni);
                if (!rc) { built++; if (ni > max_insns) max_insns = ni; if (ng > max_gprs) max_gprs = ng; }
                if (m == 0 && (xb == 1 || xb == 8)) printf("host k 2048 rows %u xb %u: %u insns, %u gprs\n", rows, xb, ni, ng);
            }
    const OmegaFp8HeadSpec refuse[] = {
        { .k = 2000, .rows = 1, .warps = 8, .xb = 1 },  /* k not a multiple of 512 */
        { .k = 2048, .rows = 3, .warps = 8, .xb = 1 },  /* rows must divide 128 and be <= 2 */
        { .k = 2048, .rows = 1, .warps = 0, .xb = 1 },  /* warps 0 */
        { .k = 2048, .rows = 1, .warps = 33, .xb = 1 }, /* warps > 32 */
        { .k = 2048, .rows = 1, .warps = 8, .xb = 0 },  /* xb 0 */
        { .k = 2048, .rows = 1, .warps = 8, .xb = 9 },  /* xb > OMEGA_FP8HEAD_MAX_XB */
        { .k = 2048, .rows = 1, .warps = 8, .xb = 1, .mutant = 4 },
    };
    for (size_t i = 0; i < sizeof refuse / sizeof refuse[0]; i++)
        CHECK(!omega_fp8head_spec_ok(&refuse[i]) && omega_fp8head_prog_stats(&refuse[i], NULL, NULL) != 0, "refusal case %zu accepted", i);
    printf("host: %u kernels built (rows 1-2, xb 1-8, mutants 0-3), max %u insns / %u gprs, %zu refusals checked\n", built, max_insns,
           max_gprs, sizeof refuse / sizeof refuse[0]);
}

static void sim_battery(void) {
    char msg[320];
    const uint32_t n = 256, k = 1024; /* 2 row blocks x 2 k-blocks per 512 step x 2 steps: every mutant has something to break */
    Head H; head_make(&H, n, k, 0x69u);
    float *x = malloc((size_t)OMEGA_FP8HEAD_MAX_XB * k * 4), *xp = aligned_alloc(16, (size_t)OMEGA_FP8HEAD_MAX_XB * k * 4);
    float *y = malloc((size_t)OMEGA_FP8HEAD_MAX_XB * n * 4), *yl = malloc((size_t)OMEGA_FP8HEAD_MAX_XB * n * 4);
    uint32_t st = 0x217u;
    for (size_t i = 0; i < (size_t)OMEGA_FP8HEAD_MAX_XB * k; i++) x[i] = frand(&st, -1.0f, 1.0f);
    for (uint32_t b = 0; b < OMEGA_FP8HEAD_MAX_XB; b++) omega_fp8_pack_x_f32(x + (size_t)b * k, k, xp + (size_t)b * k);
    for (uint32_t rows = 1; rows <= 2; rows++) {
        /* legacy reference rows once per row count */
        int lrc = 0;
        for (uint32_t b = 0; b < OMEGA_FP8HEAD_MAX_XB && !lrc; b++) lrc = sim_legacy(&H, rows, xp + (size_t)b * k, yl + (size_t)b * n, msg, sizeof msg);
        CHECK(lrc == 0, "rows %u legacy sim: %s", rows, msg);
        if (lrc) continue;
        size_t lbad; double lw = oracle_check(&H, OMEGA_FP8HEAD_MAX_XB, x, yl, &lbad);
        CHECK(lbad == 0, "rows %u legacy vs oracle: %zu bad, worst %.3e", rows, lbad, lw);
        for (uint32_t xb = 1; xb <= OMEGA_FP8HEAD_MAX_XB; xb++) {
            OmegaFp8HeadSpec sp = { .k = k, .rows = rows, .warps = 4, .xb = xb, .mutant = g_force_mutant };
            int rc = sim_head(&sp, &H, xp, y, 3, msg, sizeof msg); /* 3 CTAs x 4 warps < RG: the grid-stride loop runs */
            CHECK(rc == 0, "rows %u xb %u sim: %s", rows, xb, msg);
            if (rc) continue;
            size_t diff = 0; for (size_t i = 0; i < (size_t)xb * n; i++) diff += memcmp(&y[i], &yl[i], 4) != 0;
            size_t bad; double w = oracle_check(&H, xb, x, y, &bad);
            CHECK(diff == 0, "rows %u xb %u: %zu of %zu words differ bitwise from the legacy kernel", rows, xb, diff, (size_t)xb * n);
            CHECK(bad == 0, "rows %u xb %u: %zu of %zu outputs over TOL (worst %.3e)", rows, xb, bad, (size_t)xb * n, w);
            printf("sim rows %u xb %u: bitwise diff %zu/%zu, oracle bad %zu worst_rel %.3e\n", rows, xb, diff, (size_t)xb * n, bad, w);
        }
        /* mutants at xb 8: each must break both the oracle check and the bitwise parity */
        for (int m = 1; m <= 3; m++) {
            OmegaFp8HeadSpec sp = { .k = k, .rows = rows, .warps = 4, .xb = 8, .mutant = m };
            int rc = sim_head(&sp, &H, xp, y, 3, msg, sizeof msg);
            size_t diff = 0, bad = 0; double w = 0;
            if (!rc) { for (size_t i = 0; i < (size_t)8 * n; i++) diff += memcmp(&y[i], &yl[i], 4) != 0; w = oracle_check(&H, 8, x, y, &bad); }
            int caught = rc != 0 || (bad > 0 && diff > 0);
            CHECK(caught, "rows %u mutant %d NOT caught (diff %zu bad %zu)", rows, m, diff, bad);
            printf("sim rows %u xb 8 mutant %d: %s (rc %d, bitwise diff %zu/%u, oracle bad %zu worst_rel %.3e)\n", rows, m,
                   caught ? "caught" : "MISSED", rc, diff, 8 * n, bad, w);
        }
    }
    free(x); free(xp); free(y); free(yl); head_free(&H);
}

int main(int argc, char **argv) {
    int sim = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--sim")) sim = 1;
        else if (!strcmp(argv[i], "--host-only")) sim = 0;
        else { fprintf(stderr, "usage: %s --host-only | --sim\n", argv[0]); return 2; }
    }
    const char *fm = getenv("FP8HEAD_TEST_FORCE_MUTANT");
    if (fm) g_force_mutant = atoi(fm);
    if (g_force_mutant) printf("FP8HEAD_TEST_FORCE_MUTANT=%d: every battery kernel is mutant %d (negative control)\n", g_force_mutant, g_force_mutant);
    host_only();
    if (sim) sim_battery();
    printf("%s: %d checks, %d failed\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
