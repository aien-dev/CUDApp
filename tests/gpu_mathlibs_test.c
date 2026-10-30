/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* gpu_mathlibs_test.c -- CUDA audit rows 29 (cuFFT-class) and 30 (cuRAND-class): the GB10 xoshiro256** generator
 * (src/omega_gpu_mathlibs_api.h, kernels from om/mathlibs/mathlibs_ir.om) and the batched f32 complex FFT.
 *
 *   --host-only  no device: the C reference generator against the aien-math fixture (792 rows), refusals, every
 *                kernel through the unit, the allocator and the .om encoder, digest and listing checks, the
 *                twiddle table, and the normal transform's f32 model swept over every one of its 2^23 inputs
 *                against an f64 erfinv (Newton on libm erf/erfc) within the Wolfram bound 2.87e-6
 *   --sim        --host-only plus the real kernels through the warp simulator (tests/bw_warp_sim.h): raw and
 *                uniform outputs bit-exact against the reference, normal outputs bit-exact against the f32
 *                model and within the bound, FFT N = 4..2048 forward and inverse within the Wolfram normwise
 *                tolerance of an f64 DFT under three warp orders; every mutant must be caught
 *   (no flag)    the GB10: 4096 streams x 17 at offset 0 and x 16 at offset 1000 for bases 0, 1729, 2^64-4
 *                against the fixture and the reference, uniform and normal runs, a 131072-stream run, the FFT at
 *                every N both ways, a batch of 2050 (three launches); every mutant must be caught.
 *                --out FILE writes the receipt lines there as well.
 *   --fixture F  the aien-math fixture (default tests/fixtures/mathlibs_rng_aienmath.txt)
 * Last line: "mathlibs <tier>: PASS (<checks> checks)" or "mathlibs <tier>: FAIL (<failed> of <checks>)". */
#include "omega_gpu_mathlibs_api.h"
#include "omega_gpu_om_ir.h"
#include "omega_sm121_disasm.h"
#define WS_MAX_THREADS 512 /* the FFT's n / 4 threads at n = 2048 */
#include "bw_warp_sim.h"
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_failed;
static FILE *g_rcpt;

static void check(int ok, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void check(int ok, const char *fmt, ...) {
    g_checks++;
    if (ok) return;
    g_failed++;
    va_list ap;
    va_start(ap, fmt);
    printf("FAIL: ");
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
}

static void receipt(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void receipt(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    if (g_rcpt) {
        va_start(ap, fmt);
        vfprintf(g_rcpt, fmt, ap);
        va_end(ap);
        fprintf(g_rcpt, "\n");
    }
}

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

/* ------------------------------------------------------------------ reference generator */

#define GOLDEN 0x9E3779B97F4A7C15ull
typedef struct { uint64_t s[4]; } Xo;
static uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static void xo_seed(Xo *x, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += GOLDEN;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        x->s[i] = z ^ (z >> 31);
    }
}
static uint64_t xo_next(Xo *x) {
    uint64_t *s = x->s, r = rotl64(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl64(s[3], 45);
    return r;
}
/* out[j * ns + g] = output offset + j of the stream seeded with seed + stream0 + g */
static void ref_raw(uint64_t seed, uint64_t stream0, uint32_t ns, uint32_t offset, uint32_t per, uint64_t *out) {
    for (uint32_t g = 0; g < ns; g++) {
        Xo x;
        xo_seed(&x, seed + stream0 + g);
        for (uint32_t i = 0; i < offset; i++) (void)xo_next(&x);
        for (uint32_t j = 0; j < per; j++) out[(size_t)j * ns + g] = xo_next(&x);
    }
}

/* ------------------------------------------------------------------ normal transform: f32 model and f64 truth */

/* f32 constants, bit patterns from evidence/MATHLIBS-RNG-FFT-2026-10-27/wolfram-constants.txt section 2 */
static const uint32_t GC_BITS[9] = { 0x3fc02e2f, 0x3e7c8f63, 0xbb88e4ef, 0xbaa45408, 0x396532db,
                                     0xb6935ac1, 0xb66c7357, 0x34b84b36, 0x32f16588 };
static const uint32_t GT_BITS[9] = { 0x40354f7e, 0x3f8036db, 0x3c1aa57e, 0xbbf9c5d7, 0x3bbc127b,
                                     0xbb70bde7, 0x3ab0dc72, 0x38d3b56b, 0xb951f09b };
#define NLN2_BITS 0xbf317218u
#define RT2_BITS 0x3fb504f3u
#define NORMAL_TOL 2.87e-6

static float uniform_of(uint64_t r) { return (float)(uint32_t)(r >> 40) * 0x1p-24f; }

/* The kernel's operation order with exact LG2 and RSQ (what the simulator executes): A = (r >> 41) * 2 + 1. */
static float normal_model(uint32_t A) {
    uint32_t b = 16777216u - A;
    int32_t xi = (int32_t)(2u * A - 16777216u);
    float q = (float)(int32_t)A * (float)(int32_t)b;
    q = q * 0x1p-46f;
    q = log2f(q);
    float w = q * u2f(NLN2_BITS);
    float t = w - 2.5f, pc = u2f(GC_BITS[8]);
    for (int k = 7; k >= 0; k--) pc = fmaf(pc, t, u2f(GC_BITS[k]));
    t = w * (1.0f / sqrtf(w));
    t = t - 3.0f;
    float pt = u2f(GT_BITS[8]);
    for (int k = 7; k >= 0; k--) pt = fmaf(pt, t, u2f(GT_BITS[k]));
    uint32_t mask = (uint32_t)0 - (f2u(w - 5.0f) >> 31);
    float p = u2f((mask & f2u(pc)) | (~mask & f2u(pt)));
    float x = (float)xi * 0x1p-24f;
    return p * x * u2f(RT2_BITS);
}

/* sqrt(2) erfinv(y), y = (2A - 2^24) / 2^24: Winitzki's start, Newton on erf (|y| <= 1/2) or erfc (tail) */
static double normal_ref(uint32_t A) {
    double y = ((double)(2.0 * A) - 16777216.0) * 0x1p-24;
    double omy2 = (double)A * (double)(16777216u - A) * 0x1p-46; /* 1 - y^2, exact */
    double c = 1.0 - fabs(y);                                    /* exact */
    const double a = 0.147, l = log(omy2), u = 2.0 / (M_PI * a) + l / 2.0;
    double x = sqrt(sqrt(u * u - l / a) - u);
    for (int it = 0; it < 12; it++) {
        double d = M_2_SQRTPI * exp(-x * x), step = fabs(y) > 0.5 ? -(erfc(x) - c) / d : (erf(x) - fabs(y)) / d;
        x -= step;
        if (fabs(step) <= 1e-17 * x) break;
    }
    return copysign(M_SQRT2 * x, y);
}
static uint32_t normal_A(uint64_t r) { return (uint32_t)(r >> 41) * 2u + 1u; }
static double rel_err(float z, double ref) { return fabs((double)z - ref) / fabs(ref); }

/* ------------------------------------------------------------------ FFT reference */

static const double FFT_TOL[12] = { 0, 0, 1.19209296e-7, 4.66610727e-7, 5.26215405e-7, 8.73616990e-7, 9.33221726e-7,
                                    1.28062354e-6, 1.34022856e-6, 1.68763126e-6, 1.74723836e-6, 2.09464702e-6 };
static int lg2u(uint32_t n) { int k = 0; while ((1u << k) < n) k++; return k; }

/* f64 DFT of one transform, twiddles exp(-+2 pi i (l m mod n) / n) from libm */
static void dft64(const float *x, double *y, uint32_t n, int inverse, const double *cs, const double *sn) {
    for (uint32_t m = 0; m < n; m++) {
        double re = 0, im = 0;
        for (uint32_t l = 0; l < n; l++) {
            uint32_t k = (uint32_t)(((uint64_t)l * m) % n);
            double c = cs[k], s = inverse ? sn[k] : -sn[k], xr = x[2 * l], xm = x[2 * l + 1];
            re += xr * c - xm * s;
            im += xr * s + xm * c;
        }
        y[2 * m] = re; y[2 * m + 1] = im;
    }
}
/* worst normwise ||y - ref|| / ||ref|| over the batch */
static double fft_err(const float *in, const float *out, uint32_t n, uint32_t batch, int inverse) {
    double *cs = malloc(n * sizeof *cs), *sn = malloc(n * sizeof *sn), *ref = malloc(2 * n * sizeof *ref), worst = 0;
    for (uint32_t k = 0; k < n; k++) { cs[k] = cos(2.0 * M_PI * k / n); sn[k] = sin(2.0 * M_PI * k / n); }
    for (uint32_t b = 0; b < batch; b++) {
        dft64(in + (size_t)2 * n * b, ref, n, inverse, cs, sn);
        double num = 0, den = 0;
        for (uint32_t i = 0; i < 2 * n; i++) {
            double d = (double)out[(size_t)2 * n * b + i] - ref[i];
            num += d * d; den += ref[i] * ref[i];
        }
        double e = sqrt(num / den);
        if (!(e <= worst)) worst = e; /* a NaN stays */
    }
    free(cs); free(sn); free(ref);
    return worst;
}
static uint64_t g_lcg = 0x243F6A8885A308D3ull;
static float rnd_pm1(void) {
    g_lcg = g_lcg * 6364136223846793005ull + 1442695040888963407ull;
    return (float)(int32_t)(g_lcg >> 40 & 0xffffff) * 0x1p-23f - 1.0f;
}
static void fill_rand(float *x, size_t n) { for (size_t i = 0; i < n; i++) x[i] = rnd_pm1(); }

/* ------------------------------------------------------------------ fixture */

typedef struct { uint64_t seed; uint32_t pos; uint64_t val; } FixRow;
static FixRow g_fix[1024];
static size_t g_nfix;
static int load_fixture(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    while (fgets(line, sizeof line, f) && g_nfix < sizeof g_fix / sizeof g_fix[0]) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char *e1, *e2, *e3;
        uint64_t s = strtoull(line, &e1, 10);
        unsigned long p = strtoul(e1, &e2, 10);
        uint64_t v = strtoull(e2, &e3, 10);
        if (e1 == line || e2 == e1 || e3 == e2) { fclose(f); return -2; }
        g_fix[g_nfix++] = (FixRow){ s, (uint32_t)p, v };
    }
    fclose(f);
    return 0;
}
static const uint64_t BASES[3] = { 0, 1729, 0xFFFFFFFFFFFFFFFCull };

/* fixture rows with seed = base + g (g < ns) and offset <= pos < offset + per, checked against out; g_fix_hit marks
 * every row compared (runs overlap, so the distinct count is what says the whole fixture was covered) */
static unsigned char g_fix_hit[1024];
static size_t fixture_compare(const uint64_t *out, uint64_t base, uint32_t ns, uint32_t offset, uint32_t per,
                              size_t *bad) {
    size_t used = 0;
    for (size_t i = 0; i < g_nfix; i++) {
        uint64_t g = g_fix[i].seed - base;
        if (g >= ns || g_fix[i].pos < offset || g_fix[i].pos >= offset + per) continue;
        used++;
        g_fix_hit[i] = 1;
        if (out[(size_t)(g_fix[i].pos - offset) * ns + g] != g_fix[i].val) (*bad)++;
    }
    return used;
}
static size_t fixture_distinct(void) {
    size_t n = 0;
    for (size_t i = 0; i < g_nfix; i++) n += g_fix_hit[i];
    return n;
}

/* ------------------------------------------------------------------ host tier */

static int build(uint64_t kind, uint64_t a, uint64_t b, uint64_t c, uint32_t threads, BlackwellIRProgram *prog,
                 uint8_t *code, OmegaBlackwellKernel *k) {
    uint32_t declared = 0;
    uint64_t emit_err = 0;
    return omega_gpu_om_ir_build(kind, a, b, c, threads, prog, code, OMEGA_BW_CODE_CAP_MAX, k, &declared, &emit_err);
}

static char *listing_of(const OmegaBlackwellKernel *k) {
    char *buf = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    if (!f) return NULL;
    int rc = omega_sm121_listing(k->code, k->code_size, f);
    fclose(f);
    if (rc != 0) {
        printf("listing: %d words match no known form\n", rc);
        for (char *l = buf; l && (l = strstr(l, "??")); l++) {
            char *e = strchr(l, '\n');
            printf("  %.*s\n", e ? (int)(e - l) : 60, l);
        }
        free(buf);
        return NULL;
    }
    return buf;
}

static void host_fixture(void) {
    check(g_nfix == 792, "fixture rows %zu, expected 792", g_nfix);
    size_t bad = 0;
    for (size_t i = 0; i < g_nfix; i++) {
        Xo x;
        xo_seed(&x, g_fix[i].seed);
        for (uint32_t p = 0; p < g_fix[i].pos; p++) (void)xo_next(&x);
        if (xo_next(&x) != g_fix[i].val) bad++;
    }
    check(bad == 0, "C reference differs from aien-math in %zu of %zu fixture rows", bad, g_nfix);
    receipt("host rng_reference_vs_aienmath rows=%zu mismatches=%zu", g_nfix, bad);
}

static void host_refusals(void) {
    uint64_t buf[4];
    float fb[64], fo[64];
    check(omega_gpu_rng_generate(OMEGA_GPU_RNG_U64, 0, 0, 0, 0, 1, buf, NULL) == OMEGA_GPU_ML_BAD_ARGS, "rng ns=0");
    check(omega_gpu_rng_generate(OMEGA_GPU_RNG_U64, 0, 0, 1, 0, 0, buf, NULL) == OMEGA_GPU_ML_BAD_ARGS, "rng per=0");
    check(omega_gpu_rng_generate(OMEGA_GPU_RNG_U64, 0, 0, 1, 0, 1, NULL, NULL) == OMEGA_GPU_ML_BAD_ARGS, "rng out NULL");
    check(omega_gpu_rng_generate((OmegaGpuRngKind)3, 0, 0, 1, 0, 1, buf, NULL) == OMEGA_GPU_ML_BAD_ARGS, "rng kind 3");
    check(omega_gpu_rng_generate(OMEGA_GPU_RNG_U64, 0, 0, OMEGA_GPU_RNG_MAX_STREAMS + 1, 0, 1, buf, NULL) ==
              OMEGA_GPU_ML_TOO_LARGE, "rng streams past 131072");
    check(omega_gpu_rng_generate(OMEGA_GPU_RNG_U64, 0, 0, OMEGA_GPU_RNG_MAX_STREAMS, 0, 2049, buf, NULL) ==
              OMEGA_GPU_ML_TOO_LARGE, "rng outputs past 2^28");
    check(omega_gpu_rng_codegen((OmegaGpuRngKind)3, NULL) == OMEGA_GPU_ML_BAD_ARGS, "rng codegen kind 3");
    const uint32_t badn[] = { 0, 1, 2, 3, 6, 1000, 4096, 8192 };
    for (size_t i = 0; i < sizeof badn / sizeof badn[0]; i++) {
        check(omega_gpu_fft_c2c(fb, fo, badn[i], 1, 0, NULL) == OMEGA_GPU_ML_BAD_ARGS, "fft n=%u", badn[i]);
        check(omega_gpu_fft_codegen(badn[i], 0, NULL) == OMEGA_GPU_ML_BAD_ARGS, "fft codegen n=%u", badn[i]);
    }
    check(omega_gpu_fft_c2c(fb, fo, 4, 0, 0, NULL) == OMEGA_GPU_ML_BAD_ARGS, "fft batch=0");
    check(omega_gpu_fft_c2c(fb, fo, 4, 1, 2, NULL) == OMEGA_GPU_ML_BAD_ARGS, "fft inverse=2");
    check(omega_gpu_fft_c2c(NULL, fo, 4, 1, 0, NULL) == OMEGA_GPU_ML_BAD_ARGS, "fft in NULL");
    check(omega_gpu_fft_c2c(fb, NULL, 4, 1, 0, NULL) == OMEGA_GPU_ML_BAD_ARGS, "fft out NULL");
    check(omega_gpu_fft_c2c(fb, fb + 4, 4, 1, 0, NULL) == OMEGA_GPU_ML_BAD_ARGS, "fft overlapping in/out");
    check(omega_gpu_fft_c2c(fb, fo, 2048, 65537, 0, NULL) == OMEGA_GPU_ML_TOO_LARGE, "fft past 2^28 floats");
    check(omega_gpu_fft_twiddles(8, 0, NULL) == OMEGA_GPU_ML_BAD_ARGS, "twiddles NULL");
    /* the unit's own refusals */
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    uint8_t *code = malloc(OMEGA_BW_CODE_CAP_MAX);
    OmegaBlackwellKernel k;
    const uint64_t refuse[][4] = { { 0, 3, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 3 }, { 1, 1, 0, 0 }, { 1, 12, 0, 0 },
                                   { 1, 4, 2, 0 }, { 1, 4, 0, 3 }, { 2, 1, 0, 0 }, { 2, 4, 0, 1 }, { 3, 0, 0, 0 } };
    for (size_t i = 0; i < sizeof refuse / sizeof refuse[0]; i++)
        check(build(refuse[i][0], refuse[i][1], refuse[i][2], refuse[i][3], 128, prog, code, &k) == OMEGA_OM_IR_REFUSED,
              "unit accepted kind %" PRIu64 " a %" PRIu64 " b %" PRIu64 " c %" PRIu64, refuse[i][0], refuse[i][1],
              refuse[i][2], refuse[i][3]);
    free(prog);
    free(code);
}

static void host_codegen(void) {
    OmegaGpuMathInfo a, b;
    uint8_t base_rng[3][32];
    omega_gpu_mathlibs_test_set_mutant(0, 0);
    for (int kind = 0; kind < 3; kind++) {
        int rc = omega_gpu_rng_codegen((OmegaGpuRngKind)kind, &a);
        check(rc == OMEGA_GPU_ML_OK, "rng codegen kind %d rc %d (%s)", kind, rc, omega_gpu_mathlibs_last_error());
        rc = omega_gpu_rng_codegen((OmegaGpuRngKind)kind, &b);
        check(rc == OMEGA_GPU_ML_OK && b.kernel_cache_hit && !memcmp(a.code_digest, b.code_digest, 32),
              "rng kind %d: second codegen not a cache hit with the same digest", kind);
        memcpy(base_rng[kind], a.code_digest, 32);
        receipt("host rng_codegen kind=%d insns=%u gpr_peak=%u gpr_declared=%u", kind, a.insn_count, a.gpr_peak,
                a.gpr_declared);
    }
    uint8_t base_fft[12][2][32];
    for (int lg = 2; lg <= 11; lg++)
        for (int inv = 0; inv < 2; inv++) {
            int rc = omega_gpu_fft_codegen(1u << lg, inv, &a);
            check(rc == OMEGA_GPU_ML_OK, "fft codegen n=%u inv=%d rc %d (%s)", 1u << lg, inv, rc,
                  omega_gpu_mathlibs_last_error());
            check(a.threads_per_cta == (1u << lg) / 4, "fft n=%u threads %u", 1u << lg, a.threads_per_cta);
            memcpy(base_fft[lg][inv], a.code_digest, 32);
            receipt("host fft_codegen n=%u inverse=%d insns=%u gpr_peak=%u gpr_declared=%u", 1u << lg, inv,
                    a.insn_count, a.gpr_peak, a.gpr_declared);
        }
    for (int lg = 2; lg <= 11; lg++)
        check(memcmp(base_fft[lg][0], base_fft[lg][1], 32) != 0, "fft n=%u forward and inverse share code", 1u << lg);
    /* mutants change the code they claim to change, and only that */
    for (int lvl = 1; lvl <= 2; lvl++) {
        omega_gpu_mathlibs_test_set_mutant(1, lvl);
        for (int kind = 0; kind < 3; kind++) {
            check(omega_gpu_rng_codegen((OmegaGpuRngKind)kind, &a) == OMEGA_GPU_ML_OK, "rng mutant %d kind %d", lvl, kind);
            int differs = memcmp(a.code_digest, base_rng[kind], 32) != 0;
            int want = lvl == 1 || kind == OMEGA_GPU_RNG_NORMAL;
            check(differs == want, "rng mutant %d kind %d: digest %s", lvl, kind, differs ? "differs" : "unchanged");
        }
        omega_gpu_mathlibs_test_set_mutant(2, lvl);
        for (int lg = 2; lg <= 11; lg++) {
            check(omega_gpu_fft_codegen(1u << lg, 0, &a) == OMEGA_GPU_ML_OK, "fft mutant %d n=%u", lvl, 1u << lg);
            int differs = memcmp(a.code_digest, base_fft[lg][0], 32) != 0;
            int want = lvl == 1 || lg >= 3;
            check(differs == want, "fft mutant %d n=%u: digest %s", lvl, 1u << lg, differs ? "differs" : "unchanged");
        }
    }
    omega_gpu_mathlibs_test_set_mutant(0, 0);
    /* our own disassembler finds the instructions the encoder was asked for */
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    uint8_t *code = malloc(OMEGA_BW_CODE_CAP_MAX);
    OmegaBlackwellKernel k;
    const struct { uint64_t kind, a, b; uint32_t threads; const char *need[4]; } want[] = {
        { 0, 0, 0, 128, { "STG.E", "+0x4]", "IMAD.WIDE", NULL } },
        { 0, 2, 0, 128, { "MUFU.LG2", "MUFU.RSQ", "I2FP.F32.S32", "LOP3.LUT" } },
        { 1, 11, 1, 512, { "BAR.SYNC", "LDS", "STS", "LDG.E.64" } },
    };
    for (size_t i = 0; i < sizeof want / sizeof want[0]; i++) {
        int rc = build(want[i].kind, want[i].a, want[i].b, 0, want[i].threads, prog, code, &k);
        check(rc == OMEGA_OM_IR_OK, "build kind %" PRIu64 " a %" PRIu64 ": %s", want[i].kind, want[i].a,
              omega_gpu_om_ir_rc_name(rc));
        if (rc != OMEGA_OM_IR_OK) continue;
        char *l = listing_of(&k);
        check(l != NULL, "listing of kind %" PRIu64 " a %" PRIu64 " failed", want[i].kind, want[i].a);
        for (int j = 0; l && j < 4 && want[i].need[j]; j++)
            check(strstr(l, want[i].need[j]) != NULL, "kind %" PRIu64 " a %" PRIu64 ": listing lacks %s", want[i].kind,
                  want[i].a, want[i].need[j]);
        free(l);
    }
    free(prog);
    free(code);
}

static void host_twiddles(void) {
    static float tw[2 * 2048];
    size_t bad = 0, total = 0;
    for (int lg = 2; lg <= 11; lg++)
        for (int inv = 0; inv < 2; inv++) {
            uint32_t n = 1u << lg;
            check(omega_gpu_fft_twiddles(n, inv, tw) == OMEGA_GPU_ML_OK, "twiddles n=%u", n);
            for (uint32_t m = 0; m < n; m++) {
                double c = cos(2.0 * M_PI * m / n), s = (inv ? 1.0 : -1.0) * sin(2.0 * M_PI * m / n);
                /* within half an f32 ulp of the libm value (plus the libm error) */
                double tc = tw[2 * m], ts = tw[2 * m + 1];
                double uc = fabs(nextafterf(tw[2 * m], INFINITY) - tw[2 * m]);
                double us = fabs(nextafterf(tw[2 * m + 1], INFINITY) - tw[2 * m + 1]);
                if (fabs(tc - c) > 0.5 * uc + 1e-15 || fabs(ts - s) > 0.5 * us + 1e-15) bad++;
                total++;
            }
        }
    check(bad == 0, "%zu of %zu twiddles not the correctly rounded f32 of w^m", bad, total);
    receipt("host fft_twiddles entries=%zu off_by_more_than_half_ulp=%zu", total, bad);
}

static void host_normal_sweep(void) {
    double worst = 0;
    uint32_t worst_a = 0;
    size_t over = 0, central = 0;
    for (uint32_t A = 1; A < 16777216u; A += 2) {
        float z = normal_model(A);
        double ref = normal_ref(A), e = rel_err(z, ref);
        if (!(e <= NORMAL_TOL)) over++;
        if (!(e <= worst)) { worst = e; worst_a = A; }
        float w = -log2f((float)A * (float)(16777216u - A) * 0x1p-46f) * 0.69314718f;
        central += w < 5.0f;
    }
    check(over == 0, "normal f32 model: %zu of 2^23 inputs past %.3g (worst %.4g at A=%u)", over, NORMAL_TOL, worst,
          worst_a);
    receipt("host normal_model_sweep inputs=8388608 over_tol=%zu worst_rel=%.6e at_A=%u tol=%.3g central~%zu", over,
            worst, worst_a, NORMAL_TOL, central);
    /* the f64 reference itself against WolframLanguageEvaluator, 2026-10-27:
     * Table[{A, N[Sqrt[2] InverseErf[(2 A - 2^24)/2^24], 20]}, {A, {1, 8388607, 8388609, 16777215, 12345679, 5000001}}] */
    const struct { uint32_t A; double z; } wl[] = {
        { 1u, -5.2947040848545980574 },
        { 8388607u, -1.4940668789333163667e-7 },
        { 8388609u, 1.4940668789333163667e-7 },
        { 16777215u, 5.2947040848545980574 },
        { 12345679u, 0.63063321267521096414 },
        { 5000001u, -0.53009427681848114495 },
    };
    for (size_t i = 0; i < sizeof wl / sizeof wl[0]; i++) {
        double r = normal_ref(wl[i].A);
        check(fabs(r - wl[i].z) <= 1e-13 * fabs(wl[i].z), "normal_ref(A=%u) = %.17g, Wolfram %.17g", wl[i].A, r, wl[i].z);
    }
}

static void host_tier(void) {
    host_fixture();
    host_refusals();
    host_codegen();
    host_twiddles();
    host_normal_sweep();
}

/* ------------------------------------------------------------------ simulator tier */

static BlackwellIRProgram *g_prog;
static uint8_t *g_code;
static WsSim g_sim;

static int sim_prog(uint64_t kind, uint64_t a, uint64_t b, uint64_t mut, uint32_t threads) {
    OmegaBlackwellKernel k;
    int rc = build(kind, a, b, mut, threads, g_prog, g_code, &k);
    check(rc == OMEGA_OM_IR_OK, "sim build kind %" PRIu64 " a %" PRIu64 " b %" PRIu64 " mut %" PRIu64 ": %s", kind, a, b,
          mut, omega_gpu_om_ir_rc_name(rc));
    return rc == OMEGA_OM_IR_OK ? 0 : -1;
}

static void put64(uint32_t *w, uint64_t v) { w[0] = (uint32_t)v; w[1] = (uint32_t)(v >> 32); }

/* the API's argument block (src/omega_gpu_mathlibs_api.c) against host memory */
static int sim_rng(int kind, int mut, uint64_t seed, uint64_t stream0, uint32_t ns, uint32_t offset, uint32_t per,
                   void *out) {
    if (sim_prog(0, (uint64_t)kind, 0, (uint64_t)mut, OMEGA_GPU_RNG_THREADS) != 0) return -1;
    size_t bytes = (size_t)ns * per * (kind == 0 ? 8u : 4u);
    memset(out, 0xa5, bytes);
    memset(&g_sim, 0, sizeof g_sim);
    g_sim.p = g_prog;
    g_sim.threads = OMEGA_GPU_RNG_THREADS;
    uint32_t *cb = &g_sim.cbank[0x380 / 4];
    put64(&cb[0], (uint64_t)(uintptr_t)out);
    uint64_t x = seed + stream0;
    for (int i = 0; i < 4; i++) { x += GOLDEN; put64(&cb[2 + 2 * i], x); }
    cb[10] = offset; cb[11] = per; cb[12] = ns;
    g_sim.ranges[0] = (WsRange){ (uintptr_t)out, (uintptr_t)out + bytes };
    g_sim.nranges = 1;
    int rc = ws_run(&g_sim, (ns + OMEGA_GPU_RNG_THREADS - 1) / OMEGA_GPU_RNG_THREADS, 1);
    check(rc == 0, "sim rng kind %d: %s", kind, g_sim.msg);
    return rc;
}

static int sim_fft(uint32_t n, int inverse, int mut, uint32_t sched, const float *in, float *out, uint32_t batch) {
    int lg = lg2u(n);
    if (sim_prog(1, (uint64_t)lg, (uint64_t)inverse, (uint64_t)mut, n / 4) != 0) return -1;
    static float tw[2 * 2048];
    static uint8_t smem[8 * 2048];
    if (omega_gpu_fft_twiddles(n, inverse, tw) != OMEGA_GPU_ML_OK) return -1;
    memset(out, 0xa5, (size_t)8 * n * batch);
    memset(&g_sim, 0, sizeof g_sim);
    g_sim.p = g_prog;
    g_sim.threads = n / 4;
    g_sim.smem = smem;
    g_sim.smem_bytes = 8 * n;
    g_sim.warp_sched = sched;
    uint32_t *cb = &g_sim.cbank[0x380 / 4];
    put64(&cb[0], (uint64_t)(uintptr_t)in);
    put64(&cb[2], (uint64_t)(uintptr_t)out);
    put64(&cb[4], (uint64_t)(uintptr_t)tw);
    g_sim.ranges[0] = (WsRange){ (uintptr_t)in, (uintptr_t)in + (size_t)8 * n * batch };
    g_sim.ranges[1] = (WsRange){ (uintptr_t)out, (uintptr_t)out + (size_t)8 * n * batch };
    g_sim.ranges[2] = (WsRange){ (uintptr_t)tw, (uintptr_t)tw + (size_t)8 * n };
    g_sim.nranges = 3;
    int rc = ws_run(&g_sim, batch, 1);
    check(rc == 0, "sim fft n=%u inv=%d mut=%d: %s", n, inverse, mut, g_sim.msg);
    return rc;
}

static void sim_rng_tier(void) {
    const uint32_t ns = 130; /* two CTAs, the second with 2 live threads and 126 that exit */
    static uint64_t got[130 * 17], ref[130 * 17];
    static float fgot[130 * 17];
    const struct { uint64_t seed, stream0; uint32_t offset, per; } runs[] = {
        { 0, 0, 0, 17 }, { 0, 0, 1000, 16 }, { 0xFFFFFFFFFFFFFFFCull, 0, 0, 17 }, { 0xFFFFFFFFFFFFFFFCull, 0, 1000, 16 },
        { 1729, 0, 0, 17 }, { 1729, 0xFFFFFFFFFFFFFF80ull, 3, 5 },
    };
    size_t fix_used = 0, fix_bad = 0;
    memset(g_fix_hit, 0, sizeof g_fix_hit);
    for (size_t r = 0; r < sizeof runs / sizeof runs[0]; r++) {
        uint32_t cnt = ns * runs[r].per;
        ref_raw(runs[r].seed, runs[r].stream0, ns, runs[r].offset, runs[r].per, ref);
        if (sim_rng(0, 0, runs[r].seed, runs[r].stream0, ns, runs[r].offset, runs[r].per, got) == 0) {
            size_t bad = 0;
            for (uint32_t i = 0; i < cnt; i++) bad += got[i] != ref[i];
            check(bad == 0, "sim raw seed %" PRIu64 " offset %u: %zu of %u differ", runs[r].seed, runs[r].offset, bad, cnt);
            if (runs[r].stream0 == 0) fix_used += fixture_compare(got, runs[r].seed, ns, runs[r].offset, runs[r].per, &fix_bad);
            receipt("sim rng_raw seed=%" PRIu64 " stream0=%" PRIu64 " ns=%u offset=%u per=%u mismatches=%zu", runs[r].seed,
                    runs[r].stream0, ns, runs[r].offset, runs[r].per, bad);
        }
        if (runs[r].offset == 1000) continue; /* uniform and normal share the skip loop */
        if (sim_rng(1, 0, runs[r].seed, runs[r].stream0, ns, runs[r].offset, runs[r].per, fgot) == 0) {
            size_t bad = 0;
            for (uint32_t i = 0; i < cnt; i++) bad += f2u(fgot[i]) != f2u(uniform_of(ref[i]));
            check(bad == 0, "sim uniform seed %" PRIu64 ": %zu of %u differ", runs[r].seed, bad, cnt);
            receipt("sim rng_uniform seed=%" PRIu64 " n=%u mismatches=%zu", runs[r].seed, cnt, bad);
        }
        if (sim_rng(2, 0, runs[r].seed, runs[r].stream0, ns, runs[r].offset, runs[r].per, fgot) == 0) {
            size_t bad = 0, over = 0;
            double worst = 0;
            for (uint32_t i = 0; i < cnt; i++) {
                uint32_t A = normal_A(ref[i]);
                bad += f2u(fgot[i]) != f2u(normal_model(A));
                double e = rel_err(fgot[i], normal_ref(A));
                over += !(e <= NORMAL_TOL);
                if (!(e <= worst)) worst = e;
            }
            check(bad == 0, "sim normal seed %" PRIu64 ": %zu of %u differ from the f32 model", runs[r].seed, bad, cnt);
            check(over == 0, "sim normal seed %" PRIu64 ": %zu of %u past the bound", runs[r].seed, over, cnt);
            receipt("sim rng_normal seed=%" PRIu64 " n=%u model_mismatches=%zu over_tol=%zu worst_rel=%.4e", runs[r].seed,
                    cnt, bad, over, worst);
        }
    }
    check(fix_used > 0 && fix_bad == 0, "sim: %zu of %zu fixture comparisons differ", fix_bad, fix_used);
    receipt("sim rng_fixture rows_covered=%zu of %zu comparisons=%zu mismatches=%zu", fixture_distinct(), g_nfix,
            fix_used, fix_bad);
    /* mutants: rotation 8 breaks the raw stream; GC0 = 1.5 breaks the normal bound */
    ref_raw(0, 0, ns, 0, 17, ref);
    if (sim_rng(0, 1, 0, 0, ns, 0, 17, got) == 0) {
        size_t bad = 0;
        for (uint32_t i = 0; i < ns * 17; i++) bad += got[i] != ref[i];
        check(bad > 0, "sim rng mutant 1 not caught");
        receipt("sim rng_mutant1 mismatches=%zu of %u (must be > 0)", bad, ns * 17);
    }
    if (sim_rng(2, 2, 0, 0, ns, 0, 17, fgot) == 0) {
        size_t over = 0;
        for (uint32_t i = 0; i < ns * 17; i++) over += !(rel_err(fgot[i], normal_ref(normal_A(ref[i]))) <= NORMAL_TOL);
        check(over > 0, "sim rng mutant 2 not caught");
        receipt("sim rng_mutant2 over_tol=%zu of %u (must be > 0)", over, ns * 17);
    }
}

static void sim_fft_tier(void) {
    const uint32_t batch = 3;
    float *in = malloc((size_t)8 * 2048 * batch), *out = malloc((size_t)8 * 2048 * batch),
          *out2 = malloc((size_t)8 * 2048 * batch);
    const uint32_t sched[3] = { 0, 1, 0x9E3779B9u };
    for (int lg = 2; lg <= 11; lg++)
        for (int inv = 0; inv < 2; inv++) {
            uint32_t n = 1u << lg;
            fill_rand(in, (size_t)2 * n * batch);
            double e = -1;
            int same = 1;
            for (int s = 0; s < 3; s++) {
                if (sim_fft(n, inv, 0, sched[s], in, s ? out2 : out, batch) != 0) { same = 0; break; }
                if (s == 0) e = fft_err(in, out, n, batch, inv);
                else same &= !memcmp(out, out2, (size_t)8 * n * batch);
            }
            check(e >= 0 && e <= FFT_TOL[lg], "sim fft n=%u inv=%d: error %.4e, tolerance %.4e", n, inv, e, FFT_TOL[lg]);
            check(same, "sim fft n=%u inv=%d: output depends on the warp order", n, inv);
            receipt("sim fft n=%u inverse=%d batch=%u normwise_err=%.4e tol=%.4e warp_orders_identical=%d", n, inv, batch, e,
                    FFT_TOL[lg], same);
        }
    /* mutants: direction flip at every N, dropped twiddles at N >= 8 (N = 4 has no twiddled stage) */
    const struct { uint32_t n; int mut; int caught; } mm[] = { { 4, 1, 1 }, { 64, 1, 1 }, { 2048, 1, 1 },
                                                              { 8, 2, 1 }, { 16, 2, 1 }, { 2048, 2, 1 }, { 4, 2, 0 } };
    for (size_t i = 0; i < sizeof mm / sizeof mm[0]; i++) {
        uint32_t n = mm[i].n;
        int lg = lg2u(n);
        fill_rand(in, (size_t)2 * n * batch);
        if (sim_fft(n, 0, mm[i].mut, 0, in, out, batch) != 0) continue;
        double e = fft_err(in, out, n, batch, 0);
        int caught = !(e <= FFT_TOL[lg]);
        check(caught == mm[i].caught, "sim fft mutant %d n=%u: error %.4e, %s", mm[i].mut, n, e,
              caught ? "caught" : "not caught");
        receipt("sim fft_mutant%d n=%u normwise_err=%.4e caught=%d expected=%d", mm[i].mut, n, e, caught, mm[i].caught);
    }
    free(in); free(out); free(out2);
}

static void sim_tier(void) {
    g_prog = calloc(1, sizeof *g_prog);
    g_code = malloc(OMEGA_BW_CODE_CAP_MAX);
    sim_rng_tier();
    sim_fft_tier();
    free(g_prog);
    free(g_code);
}

/* ------------------------------------------------------------------ chip tier */

static void chip_rng(void) {
    const uint32_t ns = 4096;
    uint64_t *got = malloc((size_t)ns * 17 * 8), *ref = malloc((size_t)ns * 17 * 8);
    float *fgot = malloc((size_t)ns * 17 * 4);
    OmegaGpuMathInfo info;
    size_t fix_used = 0, fix_bad = 0;
    memset(g_fix_hit, 0, sizeof g_fix_hit);
    for (int b = 0; b < 3; b++)
        for (int o = 0; o < 2; o++) {
            uint32_t offset = o ? 1000 : 0, per = o ? 16 : 17, cnt = ns * per;
            ref_raw(BASES[b], 0, ns, offset, per, ref);
            int rc = omega_gpu_rng_generate(OMEGA_GPU_RNG_U64, BASES[b], 0, ns, offset, per, got, &info);
            check(rc == OMEGA_GPU_ML_OK, "chip raw base %" PRIu64 " offset %u: rc %d (%s)", BASES[b], offset, rc,
                  omega_gpu_mathlibs_last_error());
            if (rc != OMEGA_GPU_ML_OK) continue;
            size_t bad = 0;
            for (uint32_t i = 0; i < cnt; i++) bad += got[i] != ref[i];
            fix_used += fixture_compare(got, BASES[b], ns, offset, per, &fix_bad);
            check(bad == 0, "chip raw base %" PRIu64 " offset %u: %zu of %u differ", BASES[b], offset, bad, cnt);
            receipt("chip rng_raw base=%" PRIu64 " ns=%u offset=%u per=%u mismatches=%zu unwritten=%" PRIu64
                    " elapsed_ns=%" PRIu64 " marker=%u",
                    BASES[b], ns, offset, per, bad, info.unwritten_words, info.elapsed_ns, info.completion_marker);
        }
    size_t fix_distinct = fixture_distinct();
    check(fix_distinct == g_nfix && fix_bad == 0, "chip fixture: %zu of %zu rows covered, %zu of %zu comparisons differ",
          fix_distinct, g_nfix, fix_bad, fix_used);
    receipt("chip rng_fixture rows_covered=%zu of %zu comparisons=%zu mismatches=%zu", fix_distinct, g_nfix, fix_used,
            fix_bad);
    /* a wrapping stream0 */
    ref_raw(0x123, 0xFFFFFFFFFFFFFF9Cull, 300, 7, 5, ref);
    if (omega_gpu_rng_generate(OMEGA_GPU_RNG_U64, 0x123, 0xFFFFFFFFFFFFFF9Cull, 300, 7, 5, got, &info) == OMEGA_GPU_ML_OK) {
        size_t bad = 0;
        for (uint32_t i = 0; i < 1500; i++) bad += got[i] != ref[i];
        check(bad == 0, "chip raw wrapping stream0: %zu differ", bad);
        receipt("chip rng_raw seed=291 stream0=2^64-100 ns=300 offset=7 per=5 mismatches=%zu", bad);
    } else check(0, "chip raw wrapping stream0 failed (%s)", omega_gpu_mathlibs_last_error());
    /* uniform and normal at base 1729 */
    ref_raw(1729, 0, ns, 0, 17, ref);
    if (omega_gpu_rng_generate(OMEGA_GPU_RNG_UNIFORM, 1729, 0, ns, 0, 17, fgot, &info) == OMEGA_GPU_ML_OK) {
        size_t bad = 0;
        for (uint32_t i = 0; i < ns * 17; i++) bad += f2u(fgot[i]) != f2u(uniform_of(ref[i]));
        check(bad == 0, "chip uniform: %zu differ", bad);
        receipt("chip rng_uniform base=1729 n=%u mismatches=%zu", ns * 17, bad);
    } else check(0, "chip uniform failed (%s)", omega_gpu_mathlibs_last_error());
    if (omega_gpu_rng_generate(OMEGA_GPU_RNG_NORMAL, 1729, 0, ns, 0, 17, fgot, &info) == OMEGA_GPU_ML_OK) {
        size_t over = 0, differ = 0;
        double worst = 0;
        for (uint32_t i = 0; i < ns * 17; i++) {
            uint32_t A = normal_A(ref[i]);
            double e = rel_err(fgot[i], normal_ref(A));
            over += !(e <= NORMAL_TOL);
            if (!(e <= worst)) worst = e;
            differ += f2u(fgot[i]) != f2u(normal_model(A));
        }
        check(over == 0, "chip normal: %zu past the bound (worst %.4e)", over, worst);
        receipt("chip rng_normal base=1729 n=%u over_tol=%zu worst_rel=%.6e tol=%.3g differ_from_exact_mufu_model=%zu",
                ns * 17, over, worst, NORMAL_TOL, differ);
    } else check(0, "chip normal failed (%s)", omega_gpu_mathlibs_last_error());
    free(got); free(ref); free(fgot);
    /* the full launch: 131072 streams x 64 normals, and the same as uniforms */
    const uint32_t big = OMEGA_GPU_RNG_MAX_STREAMS, per = 64;
    size_t cnt = (size_t)big * per;
    uint64_t *r2 = malloc(cnt * 8);
    float *f2 = malloc(cnt * 4);
    ref_raw(42, 0, big, 0, per, r2);
    for (int kind = 1; kind <= 2; kind++) {
        int rc = omega_gpu_rng_generate((OmegaGpuRngKind)kind, 42, 0, big, 0, per, f2, &info);
        check(rc == OMEGA_GPU_ML_OK, "chip big kind %d rc %d (%s)", kind, rc, omega_gpu_mathlibs_last_error());
        if (rc != OMEGA_GPU_ML_OK) continue;
        size_t bad = 0;
        double worst = 0;
        for (size_t i = 0; i < cnt; i++) {
            if (kind == 1) { bad += f2u(f2[i]) != f2u(uniform_of(r2[i])); continue; }
            double e = rel_err(f2[i], normal_ref(normal_A(r2[i])));
            bad += !(e <= NORMAL_TOL);
            if (!(e <= worst)) worst = e;
        }
        check(bad == 0, "chip big kind %d: %zu bad", kind, bad);
        receipt("chip rng_big kind=%d ns=%u per=%u n=%zu bad=%zu worst_rel=%.6e elapsed_ns=%" PRIu64 " ctas=%u marker=%u",
                kind, big, per, cnt, bad, worst, info.elapsed_ns, info.ctas_last_launch, info.completion_marker);
    }
    free(r2); free(f2);
    /* mutants */
    uint64_t *mg = malloc(256 * 17 * 8), *mr = malloc(256 * 17 * 8);
    float *mf = malloc(256 * 17 * 4);
    ref_raw(0, 0, 256, 0, 17, mr);
    omega_gpu_mathlibs_test_set_mutant(1, 1);
    if (omega_gpu_rng_generate(OMEGA_GPU_RNG_U64, 0, 0, 256, 0, 17, mg, NULL) == OMEGA_GPU_ML_OK) {
        size_t bad = 0;
        for (uint32_t i = 0; i < 256 * 17; i++) bad += mg[i] != mr[i];
        check(bad > 0, "chip rng mutant 1 not caught");
        receipt("chip rng_mutant1 mismatches=%zu of %u (must be > 0)", bad, 256 * 17);
    } else check(0, "chip rng mutant 1 run failed (%s)", omega_gpu_mathlibs_last_error());
    omega_gpu_mathlibs_test_set_mutant(1, 2);
    if (omega_gpu_rng_generate(OMEGA_GPU_RNG_NORMAL, 0, 0, 256, 0, 17, mf, NULL) == OMEGA_GPU_ML_OK) {
        size_t over = 0;
        for (uint32_t i = 0; i < 256 * 17; i++) over += !(rel_err(mf[i], normal_ref(normal_A(mr[i]))) <= NORMAL_TOL);
        check(over > 0, "chip rng mutant 2 not caught");
        receipt("chip rng_mutant2 over_tol=%zu of %u (must be > 0)", over, 256 * 17);
    } else check(0, "chip rng mutant 2 run failed (%s)", omega_gpu_mathlibs_last_error());
    omega_gpu_mathlibs_test_set_mutant(0, 0);
    free(mg); free(mr); free(mf);
}

static void chip_fft(void) {
    const uint32_t batch = 8;
    size_t cap = (size_t)2 * 256 * 2050;
    float *in = malloc(cap * 4), *out = malloc(cap * 4);
    OmegaGpuMathInfo info;
    for (int lg = 2; lg <= 11; lg++)
        for (int inv = 0; inv < 2; inv++) {
            uint32_t n = 1u << lg;
            fill_rand(in, (size_t)2 * n * batch);
            int rc = omega_gpu_fft_c2c(in, out, n, batch, inv, &info);
            check(rc == OMEGA_GPU_ML_OK, "chip fft n=%u inv=%d rc %d (%s)", n, inv, rc, omega_gpu_mathlibs_last_error());
            if (rc != OMEGA_GPU_ML_OK) continue;
            double e = fft_err(in, out, n, batch, inv);
            check(e <= FFT_TOL[lg], "chip fft n=%u inv=%d: error %.4e, tolerance %.4e", n, inv, e, FFT_TOL[lg]);
            receipt("chip fft n=%u inverse=%d batch=%u normwise_err=%.4e tol=%.4e unwritten=%" PRIu64 " elapsed_ns=%" PRIu64,
                    n, inv, batch, e, FFT_TOL[lg], info.unwritten_words, info.elapsed_ns);
        }
    /* a batch past one launch: 2050 transforms = 1024 + 1024 + 2 CTAs */
    fill_rand(in, cap);
    int rc = omega_gpu_fft_c2c(in, out, 256, 2050, 0, &info);
    check(rc == OMEGA_GPU_ML_OK, "chip fft batch 2050 rc %d (%s)", rc, omega_gpu_mathlibs_last_error());
    if (rc == OMEGA_GPU_ML_OK) {
        double e = fft_err(in, out, 256, 2050, 0);
        check(e <= FFT_TOL[8], "chip fft batch 2050: error %.4e", e);
        check(info.chip_calls == 3 && info.ctas_last_launch == 2, "chip fft batch 2050: %" PRIu64 " launches, last %u CTAs",
              info.chip_calls, info.ctas_last_launch);
        receipt("chip fft n=256 batch=2050 launches=%" PRIu64 " last_ctas=%u normwise_err=%.4e tol=%.4e elapsed_ns=%" PRIu64,
                info.chip_calls, info.ctas_last_launch, e, FFT_TOL[8], info.elapsed_ns);
    }
    const struct { uint32_t n; int mut; int caught; } mm[] = { { 64, 1, 1 }, { 2048, 1, 1 }, { 16, 2, 1 }, { 4, 2, 0 } };
    for (size_t i = 0; i < sizeof mm / sizeof mm[0]; i++) {
        uint32_t n = mm[i].n;
        omega_gpu_mathlibs_test_set_mutant(2, mm[i].mut);
        fill_rand(in, (size_t)2 * n * batch);
        rc = omega_gpu_fft_c2c(in, out, n, batch, 0, NULL);
        check(rc == OMEGA_GPU_ML_OK, "chip fft mutant %d n=%u rc %d", mm[i].mut, n, rc);
        if (rc != OMEGA_GPU_ML_OK) continue;
        double e = fft_err(in, out, n, batch, 0);
        int caught = !(e <= FFT_TOL[lg2u(n)]);
        check(caught == mm[i].caught, "chip fft mutant %d n=%u: error %.4e", mm[i].mut, n, e);
        receipt("chip fft_mutant%d n=%u normwise_err=%.4e caught=%d expected=%d", mm[i].mut, n, e, caught, mm[i].caught);
    }
    omega_gpu_mathlibs_test_set_mutant(0, 0);
    free(in); free(out);
}

int main(int argc, char **argv) {
    const char *tier = "chip", *fixture = "tests/fixtures/mathlibs_rng_aienmath.txt", *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) tier = "host";
        else if (!strcmp(argv[i], "--sim")) tier = "sim";
        else if (!strcmp(argv[i], "--fixture") && i + 1 < argc) fixture = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else { fprintf(stderr, "usage: %s [--host-only | --sim] [--fixture F] [--out FILE]\n", argv[0]); return 2; }
    }
    if (out && !(g_rcpt = fopen(out, "w"))) { fprintf(stderr, "cannot write %s\n", out); return 2; }
    if (load_fixture(fixture) != 0) { fprintf(stderr, "cannot read fixture %s\n", fixture); return 2; }
    receipt("mathlibs tier=%s fixture=%s", tier, fixture);
    host_tier();
    if (!strcmp(tier, "sim")) sim_tier();
    else if (!strcmp(tier, "chip")) { chip_rng(); chip_fft(); }
    if (g_failed) receipt("mathlibs %s: FAIL (%d of %d)", tier, g_failed, g_checks);
    else receipt("mathlibs %s: PASS (%d checks)", tier, g_checks);
    if (g_rcpt) fclose(g_rcpt);
    return g_failed ? 1 : 0;
}
