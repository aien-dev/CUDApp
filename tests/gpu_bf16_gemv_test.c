/*
 * gpu_bf16_gemv_test: the bf16 streaming GEMV (omega_gpu_bf16_gemv_api.h), OWN-30B lm_head.
 *   --host-only   emulation vs double oracle, codegen budgets and refusals (no device)
 *   --sim         warp-simulator parity at 8 warps (both fragment orders, rows 1/2/4, depths, k 512/2048, multi-CTA
 *                 with the row-group loop): bit-identical to the host f32 emulation; mutants 1-3 caught
 *   (default)     the chip: all 151936 logits bit-identical to the host emulation and within tolerance of the double
 *                 oracle, second call bit-identical, bf16-x entry, mutants 1-3 caught on a small tensor, timing
 *                 (--reps N, --n N, --rows R, --warps W, --depth D, --ctas C, --no-evict, --spin-us, --out receipt.json)
 * Verdict line last: "gpu_bf16_gemv_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_gpu_bf16_gemv_api.h"
#include "omega_gpu_session.h"
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
static uint16_t bf16_rne(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t r = x + 0x7fffu + ((x >> 16) & 1u); return (uint16_t)(r >> 16); }
static float bf16_f32(uint16_t b) { uint32_t x = (uint32_t)b << 16; float f; memcpy(&f, &x, 4); return f; }

/* The kernel's exact f32 result for one row: per lane one FFMA chain (step, word, low half, high half), then the
 * SHFL.DOWN tree 16, 8, 4, 2, 1 (lane L adds lane L + o's value; lane 0 is the result). */
static float emu_row(const uint16_t *w, const float *x, uint32_t k) {
    float a[32];
    for (unsigned lane = 0; lane < 32; lane++) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < k / 256u; s++) {
            const uint16_t *wl = w + s * 256u + lane * 8u; const float *xl = x + s * 256u + lane * 8u;
            for (unsigned i = 0; i < 8; i++) acc = fmaf(bf16_f32(wl[i]), xl[i], acc);
        }
        a[lane] = acc;
    }
    for (unsigned o = 16; o >= 1; o /= 2) { float b[32]; for (unsigned l = 0; l < 32; l++) b[l] = a[l + o < 32 ? l + o : l]; for (unsigned l = 0; l < 32; l++) a[l] += b[l]; }
    return a[0];
}
static void oracle_row(const uint16_t *w, const float *x, uint32_t k, double *y, double *ys) {
    double s = 0, m = 0;
    for (uint32_t c = 0; c < k; c++) { double t = (double)bf16_f32(w[c]) * x[c]; s += t; m += fabs(t); }
    *y = s; *ys = m;
}
static uint16_t *make_w(size_t count, uint32_t seed) {
    uint16_t *w = malloc(count * 2); uint32_t s = seed;
    for (size_t i = 0; i < count; i++) w[i] = bf16_rne(frand(&s, -0.06f, 0.06f));
    return w;
}
static float *make_x(uint32_t k, uint32_t seed) {
    float *x = malloc((size_t)k * 4); uint32_t s = seed;
    for (uint32_t i = 0; i < k; i++) x[i] = frand(&s, -2.0f, 2.0f);
    return x;
}
/* bit-identical compare of n outputs against the emulation; returns the number of differing rows */
static size_t compare_emu(const uint16_t *w, const float *x, uint32_t n, uint32_t k, const float *y, size_t *first) {
    size_t bad = 0; *first = (size_t)-1;
    for (uint32_t r = 0; r < n; r++) {
        float e = emu_row(w + (size_t)r * k, x, k);
        if (memcmp(&e, &y[r], 4) != 0) { if (!bad) *first = r; bad++; }
    }
    return bad;
}

static void host_only(void) {
    printf("host-only: emulation, oracle, budgets\n");
    { uint16_t *w = make_w(2048, 5); float *x = make_x(2048, 6);
      double yo, ys; oracle_row(w, x, 2048, &yo, &ys);
      float e = emu_row(w, x, 2048);
      CHECK(fabs((double)e - yo) <= 1e-6 * ys, "emulation within 1e-6 of the double oracle (%.9g vs %.9g, scale %.3g)", (double)e, yo, ys);
      printf("emulation vs oracle: abs %.3e, scale %.3f\n", fabs((double)e - yo), ys);
      /* a known hand case: w = 1 on the first 256 columns, x = c: sum 0..255 = 32640 */
      uint16_t hw[256]; float hx[256]; for (unsigned i = 0; i < 256; i++) { hw[i] = 0x3f80; hx[i] = (float)i; }
      CHECK(emu_row(hw, hx, 256) == 32640.0f, "hand case sum 0..255 = 32640");
      free(w); free(x); }
    uint32_t ws[] = { 4, 8, 16, 32 }, rs[] = { 1, 2, 4 };
    for (unsigned i = 0; i < 4; i++) for (unsigned j = 0; j < 3; j++) for (uint32_t d = 1; d <= 8; d *= 2) for (int m = 0; m <= 3; m++) {
        OmegaBf16GemvSpec s = { 2048, rs[j], ws[i], d, m, 0, 0, 0 };
        if (rs[j] * d > 16u) continue;
        uint32_t n = 0, g = 0; int rc = omega_bf16_gemv_prog_stats(&s, &n, &g);
        uint32_t decl = ((g + 1 + 7) / 8) * 8;
        CHECK(rc == 0 && n <= 1024, "codegen k2048 rows %u warps %u depth %u mutant %d: rc %d insns %u gprs %u", rs[j], ws[i], d, m, rc, n, g);
        if (m == 0 && rc == 0 && rs[j] == 1 && d == 8) printf("codegen rows %u warps %2u depth %u: %u instructions, %u registers (declared %u, register file %u of 65536)%s\n", rs[j], ws[i], d, n, g, decl, 32u * ws[i] * decl, 32u * ws[i] * decl <= 65536u ? "" : " DOES NOT FIT");
    }
    { OmegaBf16GemvSpec s = { 2048, 3, 8, 8, 0, 0, 0, 0 }; CHECK(omega_bf16_gemv_prog_stats(&s, NULL, NULL) != 0, "rows 3 refused"); }
    { OmegaBf16GemvSpec s = { 2000, 1, 8, 8, 0, 0, 0, 0 }; CHECK(omega_bf16_gemv_prog_stats(&s, NULL, NULL) != 0, "k 2000 refused"); }
    { OmegaBf16GemvSpec s = { 2048, 1, 8, 3, 0, 0, 0, 0 }; CHECK(omega_bf16_gemv_prog_stats(&s, NULL, NULL) != 0, "depth 3 refused"); }
    { OmegaBf16GemvSpec s = { 2048, 1, 8, 8, 4, 0, 0, 0 }; CHECK(omega_bf16_gemv_prog_stats(&s, NULL, NULL) != 0, "mutant 4 refused"); }
    { /* the XBATCH form (L312 lever 1): the 35B lm_head shape, 4 x rows per CTA, 32 KiB of shared memory */
      OmegaBf16GemvSpec s = { 2048, 1, 16, 8, 0, 249856, 4, 0 }; uint32_t ni = 0, gg = 0; int rc = omega_bf16_gemv_prog_stats(&s, &ni, &gg);
      uint32_t decl = ((gg + 1 + 7) / 8) * 8;
      CHECK(rc == 0 && omega_bf16_gemv_shared_bytes(&s) == 32768u && 32u * 16u * decl <= 65536u, "codegen XBATCH 4 k 2048 warps 16 depth 8: rc %d insns %u gprs %u shared %u", rc, ni, gg, omega_bf16_gemv_shared_bytes(&s));
      printf("codegen XBATCH 4 k 2048 warps 16 depth 8: %u instructions, %u registers (declared %u), shared %u bytes\n", ni, gg, decl, omega_bf16_gemv_shared_bytes(&s)); }
    { OmegaBf16GemvSpec s = { 2048, 1, 16, 8, 0, 249856, 5, 0 }; CHECK(omega_bf16_gemv_prog_stats(&s, NULL, NULL) != 0, "xbatch 5 refused"); }
    { OmegaBf16GemvSpec s = { 2048, 1, 16, 8, 0, 0, 2, 0 }; CHECK(omega_bf16_gemv_prog_stats(&s, NULL, NULL) != 0, "xbatch without ystride refused"); }
    { OmegaBf16GemvSpec s = { 4096, 1, 16, 16, 0, 249856, 4, 0 }; CHECK(omega_bf16_gemv_prog_stats(&s, NULL, NULL) != 0, "xbatch 4 x k 4096 (64 KiB shared) refused"); }
    { OmegaBf16GemvSpec s = { 2048, 1, 8, 8, 0, 249856, 4, 0 }; CHECK(omega_bf16_gemv_prog_stats(&s, NULL, NULL) != 0, "xbatch 4 x k 2048 at 8 warps (more than 16 staging rounds) refused"); }
    { OmegaBf16GemvSpec s = { 2048, 1, 16, 8, 4, 249856, 1, 0 }; CHECK(omega_bf16_gemv_prog_stats(&s, NULL, NULL) != 0, "mutant 4 at xbatch 1 refused"); }
    CHECK(OMEGA_BF16_ARGS_WORDS <= OMEGA_GPU_SESSION_MAX_ARGS, "argument words fit the constant bank window");
    uint16_t bad[256]; memset(bad, 0, sizeof bad); bad[7] = 0x7f80;
    OmegaGpuBf16Tensor *t = NULL;
    CHECK(omega_gpu_bf16_tensor_upload(1, 256, bad, &t) == OMEGA_GPU_BF16_NAN_WEIGHT && !t, "an infinite bf16 weight is refused before any device use");
    CHECK(omega_gpu_bf16_tensor_upload(1, 200, bad, &t) == OMEGA_GPU_BF16_BAD_ARGS, "k 200 refused at upload");
}

/* ------------------------------------------------------------- simulator */
static uint64_t g_sim_issues;
static int sim_point(uint32_t k, uint32_t n, uint32_t rows, uint32_t warps, uint32_t depth, uint32_t grid_x, int mutant, uint32_t seed, char *msg, size_t msgn, size_t *bad) {
    static BlackwellIRProgram prog;
    OmegaBf16GemvSpec spec = { k, rows, warps, depth, mutant, 0, 0, 0 };
    if (omega_blackwell_codegen_bf16_gemv_prog(&spec, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    uint16_t *w = aligned_alloc(16, (size_t)n * k * 2); { uint16_t *t = make_w((size_t)n * k, seed); memcpy(w, t, (size_t)n * k * 2); free(t); }
    float *x = aligned_alloc(16, (size_t)k * 4); { float *t = make_x(k, seed + 1); memcpy(x, t, (size_t)k * 4); free(t); }
    float *y = aligned_alloc(16, ((size_t)n * 4 + 15) & ~15ull);
    uint8_t *smem = aligned_alloc(16, omega_bf16_gemv_shared_bytes(&spec));
    float first[1] = { 0 }; float *ycopy = NULL; int err = 0;
    for (int order = 0; order < 2 && !err; order++) {
        WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = 32u * warps; s.order = order; s.smem = smem; s.smem_bytes = omega_bf16_gemv_shared_bytes(&spec);
        for (uint32_t i = 0; i < n; i++) y[i] = -999.0f;
        uint64_t A[3] = { (uintptr_t)w, (uintptr_t)x, (uintptr_t)y };
        for (unsigned i = 0; i < 3; i++) { s.cbank[(OMEGA_BF16_ARG_W + 8 * i) / 4] = (uint32_t)A[i]; s.cbank[(OMEGA_BF16_ARG_W + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
        s.cbank[OMEGA_BF16_ARG_RG / 4] = n / rows; s.cbank[OMEGA_BF16_ARG_STRIDE / 4] = grid_x * warps;
        s.nranges = 3;
        s.ranges[0].lo = (uintptr_t)w; s.ranges[0].hi = (uintptr_t)w + (size_t)n * k * 2;
        s.ranges[1].lo = (uintptr_t)x; s.ranges[1].hi = (uintptr_t)x + (size_t)k * 4;
        s.ranges[2].lo = (uintptr_t)y; s.ranges[2].hi = (uintptr_t)y + (size_t)n * 4;
        err = ws_run(&s, grid_x, 1);
        if (err) { snprintf(msg, msgn, "order %d: %s (pc %d)", order, s.msg, s.err_pc); break; }
        g_sim_issues += s.warp_issues;
        size_t unwritten = 0; for (uint32_t i = 0; i < n; i++) if (y[i] == -999.0f) unwritten++;
        if (unwritten) { err = -1; snprintf(msg, msgn, "%zu output words never written", unwritten); break; }
        if (order == 0) { ycopy = malloc((size_t)n * 4); memcpy(ycopy, y, (size_t)n * 4); }
        else if (memcmp(ycopy, y, (size_t)n * 4) != 0) { err = -1; snprintf(msg, msgn, "the fragment order changed the result"); }
    }
    (void)first;
    if (!err) { size_t f; *bad = compare_emu(w, x, n, k, y, &f); snprintf(msg, msgn, "%zu of %u rows differ from the emulation%s", *bad, n, *bad ? "" : " (bit-identical)"); }
    free(ycopy); free(w); free(x); free(y); free(smem);
    return err;
}
/* ROWS form (spec.ystride, sc #642 prefill; xb 0): grid_y = T token rows; row r reads x + r * k and writes y + r * ystride.
 * XBATCH form (spec.xbatch = xb = T, L312 lever 1): one CTA row stages all T x rows and every weight read feeds them all.
 * Each row must equal the one-row emulation of its own x, and the ystride - n gap after every row stays untouched. */
static int sim_multi_point(uint32_t k, uint32_t n, uint32_t T, uint32_t xb, uint32_t ystride, uint32_t grid_x, int mutant, uint32_t seed, char *msg, size_t msgn, size_t *bad) {
    static BlackwellIRProgram prog;
    const uint32_t rows = 1, warps = 8, depth = 4; /* the host simulator runs up to 8 warps, so k (xb * k for XBATCH) <= 4096 here */
    OmegaBf16GemvSpec spec = { k, rows, warps, depth, mutant, ystride, xb, 0 };
    if (omega_blackwell_codegen_bf16_gemv_prog(&spec, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    uint16_t *w = aligned_alloc(16, (size_t)n * k * 2); { uint16_t *t = make_w((size_t)n * k, seed); memcpy(w, t, (size_t)n * k * 2); free(t); }
    float *x = aligned_alloc(16, (size_t)T * k * 4);
    for (uint32_t r = 0; r < T; r++) { float *t = make_x(k, seed + 1 + r); memcpy(x + (size_t)r * k, t, (size_t)k * 4); free(t); }
    const size_t ywords = (size_t)T * ystride;
    float *y = aligned_alloc(16, (ywords * 4 + 15) & ~15ull);
    uint8_t *smem = aligned_alloc(16, omega_bf16_gemv_shared_bytes(&spec));
    for (size_t i = 0; i < ywords; i++) y[i] = -999.0f;
    WsSim s; memset(&s, 0, sizeof s);
    s.p = &prog; s.threads = 32u * warps; s.smem = smem; s.smem_bytes = omega_bf16_gemv_shared_bytes(&spec);
    uint64_t A[3] = { (uintptr_t)w, (uintptr_t)x, (uintptr_t)y };
    for (unsigned i = 0; i < 3; i++) { s.cbank[(OMEGA_BF16_ARG_W + 8 * i) / 4] = (uint32_t)A[i]; s.cbank[(OMEGA_BF16_ARG_W + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
    s.cbank[OMEGA_BF16_ARG_RG / 4] = n / rows; s.cbank[OMEGA_BF16_ARG_STRIDE / 4] = grid_x * warps;
    s.nranges = 3;
    s.ranges[0].lo = (uintptr_t)w; s.ranges[0].hi = (uintptr_t)w + (size_t)n * k * 2;
    s.ranges[1].lo = (uintptr_t)x; s.ranges[1].hi = (uintptr_t)x + (size_t)T * k * 4;
    s.ranges[2].lo = (uintptr_t)y; s.ranges[2].hi = (uintptr_t)y + ywords * 4;
    int err = ws_run(&s, grid_x, xb > 1u ? 1u : T);
    if (err) snprintf(msg, msgn, "%s (pc %d)", s.msg, s.err_pc);
    *bad = 0;
    size_t gap_written = 0;
    for (uint32_t r = 0; !err && r < T; r++) {
        size_t f; *bad += compare_emu(w, x + (size_t)r * k, n, k, y + (size_t)r * ystride, &f);
        for (uint32_t i = n; i < ystride; i++) if (y[(size_t)r * ystride + i] != -999.0f) gap_written++;
    }
    if (!err && gap_written) { err = -1; snprintf(msg, msgn, "%zu words written inside the ystride gap", gap_written); }
    else if (!err) snprintf(msg, msgn, "%zu of %u rows x %u tokens differ from the emulation%s", *bad, n, T, *bad ? "" : " (bit-identical)");
    free(w); free(x); free(y); free(smem);
    return err;
}
/* MASK form (HUM L314 exact head): rows with hi >= tau must equal the plain kernel's row bit for bit (the emulation),
 * every other row must hold -inf. hi covers below / equal / above tau and -0 against tau = +0 (a candidate). */
static int sim_mask_point(uint32_t k, uint32_t n, uint32_t warps, uint32_t depth, uint32_t grid_x, int mutant, uint32_t seed,
                          char *msg, size_t msgn, size_t *bad, size_t *cands) {
    static BlackwellIRProgram prog;
    OmegaBf16GemvSpec spec = { k, 1, warps, depth, mutant, 0, 0, 1 };
    if (omega_blackwell_codegen_bf16_gemv_prog(&spec, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    uint16_t *w = aligned_alloc(16, (size_t)n * k * 2); { uint16_t *t = make_w((size_t)n * k, seed); memcpy(w, t, (size_t)n * k * 2); free(t); }
    float *x = aligned_alloc(16, (size_t)k * 4); { float *t = make_x(k, seed + 1); memcpy(x, t, (size_t)k * 4); free(t); }
    float *y = aligned_alloc(16, ((size_t)n * 4 + 15) & ~15ull);
    float *hi = aligned_alloc(16, ((size_t)n * 4 + 15) & ~15ull);
    float *tau = aligned_alloc(16, 16);
    uint8_t *smem = aligned_alloc(16, omega_bf16_gemv_shared_bytes(&spec));
    const float T = seed & 1 ? 0.0f : 0.25f;
    tau[0] = T;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t h = (i * 2654435761u) ^ seed;
        hi[i] = (h % 3 == 0) ? T + 0.5f : (h % 3 == 1) ? T - 0.5f : (h % 5 == 0) ? T : T - 1e-6f;
    }
    hi[1] = T; hi[2] = nextafterf(T, -1.0f);
    if (T == 0.0f) hi[3] = -0.0f; /* -0 >= +0 numerically: a candidate */
    *cands = 0;
    for (uint32_t i = 0; i < n; i++) *cands += hi[i] >= T;
    int err = 0;
    for (int order = 0; order < 2 && !err; order++) {
        WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = 32u * warps; s.order = order; s.smem = smem; s.smem_bytes = omega_bf16_gemv_shared_bytes(&spec);
        for (uint32_t i = 0; i < n; i++) y[i] = -999.0f;
        uint64_t A[5] = { (uintptr_t)w, (uintptr_t)x, (uintptr_t)y, 0, (uintptr_t)hi };
        for (unsigned i = 0; i < 3; i++) { s.cbank[(OMEGA_BF16_ARG_W + 8 * i) / 4] = (uint32_t)A[i]; s.cbank[(OMEGA_BF16_ARG_W + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
        s.cbank[OMEGA_BF16_ARG_HI / 4] = (uint32_t)A[4]; s.cbank[OMEGA_BF16_ARG_HI / 4 + 1] = (uint32_t)(A[4] >> 32);
        s.cbank[OMEGA_BF16_ARG_TAU / 4] = (uint32_t)(uintptr_t)tau; s.cbank[OMEGA_BF16_ARG_TAU / 4 + 1] = (uint32_t)((uint64_t)(uintptr_t)tau >> 32);
        s.cbank[OMEGA_BF16_ARG_RG / 4] = n; s.cbank[OMEGA_BF16_ARG_STRIDE / 4] = grid_x * warps;
        s.nranges = 5;
        s.ranges[0].lo = (uintptr_t)w; s.ranges[0].hi = (uintptr_t)w + (size_t)n * k * 2;
        s.ranges[1].lo = (uintptr_t)x; s.ranges[1].hi = (uintptr_t)x + (size_t)k * 4;
        s.ranges[2].lo = (uintptr_t)y; s.ranges[2].hi = (uintptr_t)y + (size_t)n * 4;
        s.ranges[3].lo = (uintptr_t)hi; s.ranges[3].hi = (uintptr_t)hi + (size_t)n * 4;
        s.ranges[4].lo = (uintptr_t)tau; s.ranges[4].hi = (uintptr_t)tau + 4;
        err = ws_run(&s, grid_x, 1);
        if (err) { snprintf(msg, msgn, "order %d: %s (pc %d)", order, s.msg, s.err_pc); break; }
        g_sim_issues += s.warp_issues;
        size_t b = 0;
        for (uint32_t i = 0; i < n; i++) {
            float want = hi[i] >= T ? emu_row(w + (size_t)i * k, x, k) : -INFINITY;
            b += memcmp(&want, &y[i], 4) != 0;
        }
        *bad = b;
        snprintf(msg, msgn, "%zu candidates of %u rows, %zu rows wrong%s", *cands, n, b, b ? "" : " (candidates bit-identical, the rest -inf)");
    }
    free(w); free(x); free(y); free(hi); free(tau); free(smem);
    return err;
}
static void sim_battery(void) {
    {   /* MASK form (L314): two thresholds (0 with a -0 row, 0.25), two grids, and mutant 4 must be caught */
        static const struct { uint32_t k, n, warps, depth, grid, seed; } mp[] = { { 2048, 100, 8, 8, 2, 7 }, { 2048, 96, 8, 4, 3, 10 }, { 512, 70, 4, 2, 2, 13 } };
        for (size_t i = 0; i < sizeof mp / sizeof mp[0]; i++) {
            char msg[300]; size_t bad = 1, c = 0;
            int err = sim_mask_point(mp[i].k, mp[i].n, mp[i].warps, mp[i].depth, mp[i].grid, 0, mp[i].seed, msg, sizeof msg, &bad, &c);
            CHECK(!err && bad == 0 && c > 0 && c < mp[i].n, "sim MASK k %u n %u: %s", mp[i].k, mp[i].n, msg);
            printf("sim MASK k %u n %u warps %u depth %u grid %u: %s %s\n", mp[i].k, mp[i].n, mp[i].warps, mp[i].depth, mp[i].grid, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
        }
        char msg[300]; size_t bad = 0, c = 0;
        int err = sim_mask_point(2048, 64, 8, 4, 2, 4, 7, msg, sizeof msg, &bad, &c);
        CHECK(!err && bad != 0, "sim MASK mutant 4 must be caught: %s", msg);
        printf("sim MASK mutant 4: %s %s\n", err ? "ERROR" : bad ? "caught" : "NOT CAUGHT", msg);
        OmegaBf16GemvSpec r2 = { 2048, 2, 8, 4, 0, 0, 0, 1 }, ys = { 2048, 1, 8, 4, 0, 64, 0, 1 }, m4 = { 2048, 1, 8, 4, 4, 0, 0, 0 };
        CHECK(omega_bf16_gemv_prog_stats(&r2, NULL, NULL) != 0 && omega_bf16_gemv_prog_stats(&ys, NULL, NULL) != 0 &&
              omega_bf16_gemv_prog_stats(&m4, NULL, NULL) != 0, "MASK refuses rows 2, ystride, and mutant 4 without MASK");
    }
    {   /* ROWS form: 3 token rows, ystride n + 4 (the gap must stay untouched); and ystride == n as the graph uses it */
        static const struct { uint32_t k, n, T, ystride, grid; } rp[] = { { 2048, 96, 3, 100, 2 }, { 4096, 96, 4, 96, 3 } };
        for (size_t i = 0; i < sizeof rp / sizeof rp[0]; i++) {
            char msg[300]; size_t bad = 1;
            int err = sim_multi_point(rp[i].k, rp[i].n, rp[i].T, 0, rp[i].ystride, rp[i].grid, 0, 91 + (uint32_t)i, msg, sizeof msg, &bad);
            CHECK(!err && bad == 0, "sim ROWS k %u n %u T %u ystride %u: %s", rp[i].k, rp[i].n, rp[i].T, rp[i].ystride, msg);
            printf("sim ROWS k %u n %u T %u ystride %u: %s %s\n", rp[i].k, rp[i].n, rp[i].T, rp[i].ystride, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
        }
    }
    { /* XBATCH form: every weight read feeds xb x rows; mutant 4 (every row reads x row 0) must be caught */
        struct { uint32_t k, n, xb, ystride, grid; } xp[] = { { 1024, 96, 2, 100, 2 }, { 1024, 96, 4, 96, 3 }, { 2048, 64, 2, 70, 2 }, { 1024, 40, 3, 50, 2 } };
        for (size_t i = 0; i < sizeof xp / sizeof xp[0]; i++) {
            char msg[300]; size_t bad = 1;
            int err = sim_multi_point(xp[i].k, xp[i].n, xp[i].xb, xp[i].xb, xp[i].ystride, xp[i].grid, 0, 61 + (uint32_t)i, msg, sizeof msg, &bad);
            CHECK(!err && bad == 0, "sim XBATCH k %u n %u xb %u ystride %u: %s", xp[i].k, xp[i].n, xp[i].xb, xp[i].ystride, msg);
            printf("sim XBATCH k %u n %u xb %u ystride %u: %s %s\n", xp[i].k, xp[i].n, xp[i].xb, xp[i].ystride, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
        }
        char msg[300]; size_t bad = 0;
        int err = sim_multi_point(1024, 64, 2, 2, 64, 2, 4, 79, msg, sizeof msg, &bad);
        CHECK(!err && bad != 0, "sim XBATCH mutant 4 must be caught: %s", msg);
        printf("sim XBATCH mutant 4: %s %s\n", err ? "ERROR" : bad ? "caught" : "NOT CAUGHT", msg);
    }
    struct { uint32_t k, n, rows, warps, depth, grid; } pts[] = {
        { 2048, 100, 1, 8, 8, 2 }, { 2048, 100, 1, 8, 4, 3 }, { 2048, 64, 1, 8, 1, 2 }, { 2048, 96, 2, 8, 4, 2 }, { 2048, 96, 4, 8, 4, 1 },
        { 512, 70, 1, 8, 2, 2 }, { 512, 70, 2, 4, 1, 3 }, { 4096, 40, 1, 8, 16, 2 },
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        char msg[300]; size_t bad = 1;
        int err = sim_point(pts[i].k, pts[i].n, pts[i].rows, pts[i].warps, pts[i].depth, pts[i].grid, 0, 31 + (uint32_t)i, msg, sizeof msg, &bad);
        CHECK(!err && bad == 0, "sim k %u n %u rows %u warps %u depth %u grid %u: %s", pts[i].k, pts[i].n, pts[i].rows, pts[i].warps, pts[i].depth, pts[i].grid, msg);
        printf("sim k %u n %u rows %u warps %u depth %u grid %u: %s %s\n", pts[i].k, pts[i].n, pts[i].rows, pts[i].warps, pts[i].depth, pts[i].grid, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
    }
    for (int m = 1; m <= 3; m++) {
        char msg[300]; size_t bad = 0;
        int err = sim_point(2048, 64, 1, 8, 4, 2, m, 77, msg, sizeof msg, &bad);
        CHECK(!err && bad != 0, "sim mutant %d must be caught: %s", m, msg);
        printf("sim mutant %d: %s %s\n", m, err ? "ERROR" : bad ? "caught" : "NOT CAUGHT", msg);
    }
    printf("sim: %" PRIu64 " warp issues in total\n", g_sim_issues);
}

/* ------------------------------------------------------------- chip */
static int g_evict = 1;
static void evict_host_caches(void) {
    static uint8_t *buf; const size_t sz = 512u << 20; static uint8_t pass;
    if (!buf) buf = malloc(sz);
    if (!buf) return;
    memset(buf, ++pass, sz);
    volatile uint64_t sum = 0; for (size_t i = 0; i < sz; i += 64) sum += buf[i];
}
static double med(uint64_t *v, uint32_t n) { qsort(v, n, 8, cmp_u64); return (double)v[n / 2]; }

static int chip_run(const char *out_path, uint32_t reps, uint32_t N, uint32_t K) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot write %s\n", out_path); return 1; }
    if (out) fprintf(out, "{\"receipt\":\"OMEGA_GPU_BF16_GEMV_V1\",\"n\":%u,\"k\":%u,\"reps\":%u,\"evict\":%d,\"spin_us\":%u,\"points\":[", N, K, reps, g_evict, omega_gpu_session_spin_us());
    uint16_t *w = make_w((size_t)N * K, 90001); float *x = make_x(K, 90002);
    OmegaGpuBf16Tensor *t = NULL;
    double t0 = now_ms();
    int rc = omega_gpu_bf16_tensor_upload(N, K, w, &t);
    printf("upload %u x %u bf16 (%.0f MB): rc=%s %.0f ms (%s)\n", N, K, (double)N * K * 2 / 1e6, omega_gpu_bf16_rc_name(rc), now_ms() - t0, rc ? omega_gpu_bf16_last_error() : "");
    CHECK(rc == 0, "upload");
    if (rc) { if (out) { fprintf(out, "],\"verdict\":\"FAIL\"}\n"); fclose(out); } return 1; }
    float *y = malloc((size_t)N * 4), *y2 = malloc((size_t)N * 4); OmegaGpuBf16Info info;
    rc = omega_gpu_bf16_gemv_f32(t, x, y, &info);
    CHECK(rc == 0, "gemv: %s %s", omega_gpu_bf16_rc_name(rc), rc ? omega_gpu_bf16_last_error() : "");
    if (rc == 0) {
        size_t first; size_t bad = compare_emu(w, x, N, K, y, &first);
        CHECK(bad == 0, "chip vs host emulation: %zu of %u rows differ (first %zd)", bad, N, (ssize_t)first);
        double worst = 0, wy = 0; size_t arg_o = 0, arg_c = 0; double bo = -1e300; float bc = -1e30f;
        for (uint32_t r = 0; r < N; r++) {
            double yo, ys; oracle_row(w + (size_t)r * K, x, K, &yo, &ys);
            double d = fabs((double)y[r] - yo) / (ys + 1e-30); if (d > worst) { worst = d; wy = yo; }
            if (yo > bo) { bo = yo; arg_o = r; }
            if (y[r] > bc) { bc = y[r]; arg_c = r; }
        }
        CHECK(worst < 1e-5, "chip vs double oracle: worst rel-to-scale %.2e", worst);
        printf("parity: %zu of %u rows differ from the f32 emulation (bit compare); worst error vs double oracle %.2e of the term scale (y %.4g); argmax chip %zu oracle %zu\n", bad, N, worst, wy, arg_c, arg_o);
        if (out) fprintf(out, "{\"point\":\"parity\",\"rows_bad\":%zu,\"worst_rel\":%.3e,\"argmax_chip\":%zu,\"argmax_oracle\":%zu}", bad, worst, arg_c, arg_o);
        rc = omega_gpu_bf16_gemv_f32(t, x, y2, NULL);
        CHECK(rc == 0 && memcmp(y, y2, (size_t)N * 4) == 0, "second call bit-identical");
        printf("second call: %s\n", rc == 0 && memcmp(y, y2, (size_t)N * 4) == 0 ? "bit-identical" : "DIFFERS");
        uint16_t *xb = malloc(K * 2); float *xf = malloc(K * 4);
        for (uint32_t i = 0; i < K; i++) { xb[i] = bf16_rne(x[i]); xf[i] = bf16_f32(xb[i]); }
        rc = omega_gpu_bf16_gemv_bf16(t, xb, y2, NULL);
        size_t b2 = 0, f2; if (rc == 0) b2 = compare_emu(w, xf, N, K, y2, &f2);
        CHECK(rc == 0 && b2 == 0, "bf16-x entry equals the emulation on the rounded x (%zu rows differ)", b2);
        printf("bf16-x entry: %s\n", rc == 0 && b2 == 0 ? "bit-identical to the emulation" : "MISMATCH");
        free(xb); free(xf);
    }
    /* timing */
    {
        uint64_t *ce = malloc(reps * 8), *cc = malloc(reps * 8); uint32_t done = 0; int ok = 1;
        for (uint32_t i = 0; i < reps && ok; i++) {
            if (g_evict) evict_host_caches();
            if (omega_gpu_bf16_gemv_f32(t, x, y2, &info) != 0) { ok = 0; break; }
            ce[done] = info.elapsed_ns; cc[done] = info.call_ns; done++;
        }
        CHECK(done == reps, "timing loop completed %u of %u", done, reps);
        if (done) {
            double mc = med(ce, done), ma = med(cc, done);
            printf("timing (medians of %u, evict %d): rows %u warps %u depth %u ctas %u: chip %.1f us = %.1f GB/s on %.0f MB; call %.1f us = %.1f GB/s\n",
                   done, g_evict, info.rows_per_warp, info.warps_per_cta, info.depth, info.grid_x, mc / 1e3, (double)info.weight_bytes / mc, (double)info.weight_bytes / 1e6, ma / 1e3, (double)info.weight_bytes / ma);
            if (out) fprintf(out, ",{\"point\":\"timing\",\"rows\":%u,\"warps\":%u,\"depth\":%u,\"ctas\":%u,\"reps\":%u,\"chip_median_ns\":%.0f,\"call_median_ns\":%.0f,\"weight_bytes\":%" PRIu64 ",\"chip_gbps\":%.2f,\"call_gbps\":%.2f,\"marker\":%u}",
                             info.rows_per_warp, info.warps_per_cta, info.depth, info.grid_x, done, mc, ma, info.weight_bytes, (double)info.weight_bytes / mc, (double)info.weight_bytes / ma, info.completion_marker);
        }
        free(ce); free(cc);
    }
    omega_gpu_bf16_tensor_free(t);
    /* mutants on a small tensor */
    {
        const uint32_t MN = 4096; OmegaGpuBf16Tensor *ts = NULL;
        rc = omega_gpu_bf16_tensor_upload(MN, K, w, &ts);
        CHECK(rc == 0, "small tensor upload");
        if (rc == 0) {
            float *ys = malloc(MN * 4);
            rc = omega_gpu_bf16_gemv_f32(ts, x, ys, NULL); size_t f; size_t b0 = rc == 0 ? compare_emu(w, x, MN, K, ys, &f) : (size_t)-1;
            CHECK(b0 == 0, "small tensor unmutated parity (%zu rows differ)", b0);
            for (int m = 1; m <= 3; m++) {
                omega_gpu_bf16_test_set_mutant(m);
                rc = omega_gpu_bf16_gemv_f32(ts, x, ys, NULL);
                size_t b = rc == 0 ? compare_emu(w, x, MN, K, ys, &f) : (size_t)-1;
                CHECK(b != 0, "chip mutant %d must be caught (rc %s)", m, omega_gpu_bf16_rc_name(rc));
                printf("chip mutant %d: %s (%zu of %u rows differ)\n", m, b ? "caught" : "NOT CAUGHT", b == (size_t)-1 ? (size_t)MN : b, MN);
                if (out) fprintf(out, ",{\"point\":\"mutant %d\",\"caught\":%d}", m, b != 0);
            }
            omega_gpu_bf16_test_set_mutant(0);
            free(ys);
        }
        omega_gpu_bf16_tensor_free(ts);
    }
    free(w); free(x); free(y); free(y2);
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS"); fclose(out); }
    return g_failed ? 1 : 0;
}

int main(int argc, char **argv) {
    int host = 0, sim = 0; const char *out_path = NULL; uint32_t reps = 0, rows = 0, warps = 0, depth = 0, ctas = 0, spin = 2000, N = 151936, K = 2048;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) rows = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--warps") == 0 && i + 1 < argc) warps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) depth = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctas") == 0 && i + 1 < argc) ctas = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) N = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) K = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--spin-us") == 0 && i + 1 < argc) spin = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-evict") == 0) g_evict = 0;
        else { printf("unknown argument %s\n", argv[i]); return 2; }
    }
    if (host) host_only();
    if (sim) sim_battery();
    if (!host && !sim) {
        if (rows || warps || depth) omega_gpu_bf16_set_shape(rows, warps, depth);
        if (ctas) omega_gpu_bf16_set_cta_budget(ctas);
        (void)omega_gpu_session_set_spin_us(spin);
        chip_run(out_path, reps ? reps : 1, N, K);
        omega_gpu_session_close();
    }
    printf("gpu_bf16_gemv_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
