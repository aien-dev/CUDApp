/* gemm_ex_ab.c -- GemmGeneral served-shape A/B timing driver (AB-PLAN.md in this directory).
 * Measurement harness only, not product code. One process, one build (CUDApp gemm-general/gemm-ex at the plan's sha):
 *   arm A = omega_gpu_gemm_ld   (src/omega_gpu_gemm.c, C3: the existing entry for these shapes)
 *   arm B = omega_gpu_gemm_ex   (src/omega_gpu_gemm_ex.c: the general entry, same shapes, same inputs)
 * Weights are stored [out][in] (A row-major M x K, B N rows of K): C3 layout OMEGA_GPU_GEMM_B_COL, gemm_ex op_a N, op_b T.
 * Round 0 runs every cell through both arms and is labelled warm=1 (kernel builds, first touch); rounds 1..R are timed.
 * In every round the cell order is a seeded shuffle and the arm order inside a cell a seeded coin.
 * Per pair: wall_ns of each call (CLOCK_MONOTONIC around the call), rc, B's route and core_calls, the live server's
 * in_flight before and after the pair, and the correctness gate: C_B == C_A bit for bit, plus 64 sampled elements of C_A
 * within K * 2^-22 * sum|a b| of a double host reference.
 * After the timed rounds, the negative control: gemm_ex mutant 1 (drops the last k) on one cell must fail both the bitwise
 * check and the reference check; a control that stays green fails the run.
 *   gemm_ex_ab --rounds R [--seed S] [--spin-us US]
 * Why C: it links the C library under test; no .om binding of either entry exists. Copied unchanged into the evidence. */
#include "omega_gpu_gemm.h"
#include "omega_gpu_gemm_ex.h"
#include "omega_gpu_session.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int in_flight(void) {
    FILE *f = popen("curl -s -m 2 127.0.0.1:18091/health", "r");
    if (!f) return -1;
    char buf[1024]; size_t n = fread(buf, 1, sizeof buf - 1, f); buf[n] = 0;
    pclose(f);
    const char *p = strstr(buf, "\"in_flight\"");
    if (!p || !(p = strchr(p, ':'))) return -1;
    return atoi(p + 1);
}

static uint64_t g_rng;
static uint32_t rnd(void) { g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(g_rng >> 33); }
static uint64_t now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }

static float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static float e4m3f(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    const float v = e ? ldexpf(1.0f + (float)m / 8.0f, e - 7) : ldexpf((float)m / 8.0f, -6);
    return s ? -v : v;
}
/* finite values: bf16 in [-2, 2) with 8 significant bits; e4m3 any code but the NaNs 0x7f / 0xff */
static void fill(int fp8, void *p, uint64_t n, uint64_t seed) {
    g_rng = seed * 2862933555777941757ull + 3037000493ull;
    for (uint64_t i = 0; i < n; i++) {
        if (fp8) { uint8_t b; do b = (uint8_t)rnd(); while ((b & 0x7fu) == 0x7fu); ((uint8_t *)p)[i] = b; }
        else { float f = ldexpf((float)(int32_t)(rnd() % 512u) - 256.0f, -7); uint32_t u; memcpy(&u, &f, 4); ((uint16_t *)p)[i] = (uint16_t)(u >> 16); }
    }
}
static float val(int fp8, const void *p, uint64_t i) { return fp8 ? e4m3f(((const uint8_t *)p)[i]) : bf2f(((const uint16_t *)p)[i]); }

typedef struct { uint32_t M, N, K; const char *tag; } Shape;
/* serving shapes of the live 35B (hidden 2048), the same nine GemmTune measured (handoffs/GemmTune/AB-PLAN.md) */
static const Shape SH[] = {
    { 1, 8192, 2048, "q+gate decode" },   { 16, 12288, 2048, "gdn in_proj b16" }, { 512, 8192, 2048, "q+gate p512" },
    { 512, 2048, 4096, "o_proj p512" },   { 2048, 12288, 2048, "gdn in_proj p2048" }, { 2048, 2048, 4096, "out_proj p2048" },
    { 64, 1024, 2048, "expert gate+up" }, { 64, 2048, 512, "expert down" },      { 2048, 16384, 2048, "lm_head slice" },
};
#define NS (sizeof SH / sizeof SH[0])

static uint64_t ref_bad_of(int fp8, const Shape *s, const void *A, const void *B, const float *C, uint64_t seed) {
    uint64_t bad = 0;
    g_rng = seed;
    for (int e = 0; e < 64; e++) {
        const uint32_t i = e == 0 ? s->M - 1u : rnd() % s->M, j = e == 0 ? s->N - 1u : rnd() % s->N;
        double d = 0.0, mag = 0.0;
        for (uint32_t kk = 0; kk < s->K; kk++) {
            const double p = (double)val(fp8, A, (uint64_t)i * s->K + kk) * (double)val(fp8, B, (uint64_t)j * s->K + kk);
            d += p; mag += fabs(p);
        }
        const double got = (double)C[(uint64_t)i * s->N + j];
        if (!(fabs(got - d) <= (double)s->K * ldexp(1.0, -22) * mag + 1e-30)) bad++;
    }
    return bad;
}

int main(int argc, char **argv) {
    unsigned rounds = 0; uint64_t seed = 20261027;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--spin-us") && i + 1 < argc) {
            unsigned us = (unsigned)strtoul(argv[++i], NULL, 10);
            if (omega_gpu_session_set_spin_us(us) != 0) { fprintf(stderr, "bad --spin-us %u\n", us); return 2; }
            printf("# spin_us %u\n", us);
        } else { fprintf(stderr, "usage: %s --rounds R [--seed S] [--spin-us US]\n", argv[0]); return 2; }
    }
    if (!rounds) { fprintf(stderr, "--rounds R (>= 1) required\n"); return 2; }
    const size_t amax = (size_t)2048 * 4096 * 2, bmax = (size_t)16384 * 2048 * 2, cmax = (size_t)2048 * 16384;
    void *A = malloc(amax), *B = malloc(bmax); float *CA = malloc(cmax * 4), *CB = malloc(cmax * 4);
    if (!A || !B || !CA || !CB) { printf("NOT_RUN: host allocation\n"); return 2; }
    memset(A, 0, amax); memset(B, 0, bmax); memset(CA, 0, cmax * 4); memset(CB, 0, cmax * 4); /* first touch outside timing */
    const size_t n = 2 * NS;
    int bad_total = 0;
    for (unsigned round = 0; round <= rounds; round++) {
        size_t ord[2 * NS];
        for (size_t i = 0; i < n; i++) ord[i] = i;
        uint64_t r = seed * 1000003u + round;
        for (size_t i = n - 1; i > 0; i--) { r = r * 6364136223846793005ull + 1442695040888963407ull; size_t j = (size_t)(r >> 33) % (i + 1); size_t t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
        for (size_t q = 0; q < n; q++) {
            const int fp8 = (int)(ord[q] / NS);
            const Shape *s = &SH[ord[q] % NS];
            const OmegaGpuGemmType t = fp8 ? OMEGA_GPU_GEMM_E4M3 : OMEGA_GPU_GEMM_BF16;
            fill(fp8, A, (uint64_t)s->M * s->K, (uint64_t)s->M * 3 + s->N * 5 + s->K * 7 + (uint64_t)fp8);
            fill(fp8, B, (uint64_t)s->N * s->K, (uint64_t)s->M * 11 + s->N * 13 + s->K * 17 + (uint64_t)fp8);
            const size_t cn = (size_t)s->M * s->N;
            memset(CA, 0xff, cn * 4); memset(CB, 0xff, cn * 4); /* NaN poison: an unwritten element fails the gate */
            r = r * 6364136223846793005ull + 1442695040888963407ull;
            const int a_first = (int)((r >> 40) & 1u);
            OmegaGemmExInfo xi; memset(&xi, 0, sizeof xi);
            OmegaGemmExArgs x; memset(&x, 0, sizeof x);
            x.op_a = OMEGA_GEMM_OP_N; x.op_b = OMEGA_GEMM_OP_T; x.m = s->M; x.n = s->N; x.k = s->K; x.batch = 1;
            x.a_type = x.b_type = fp8 ? OMEGA_GEMM_DT_E4M3 : OMEGA_GEMM_DT_BF16; x.c_type = OMEGA_GEMM_DT_F32;
            x.a = A; x.lda = s->K; x.b = B; x.ldb = s->K; x.c = CB; x.ldc = s->N; x.alpha = 1.0f; x.beta = 0.0f; x.info = &xi;
            const int if0 = in_flight();
            int rca = -99, rcb = -99; uint64_t wa = 0, wb = 0;
            for (int leg = 0; leg < 2; leg++) {
                if ((leg == 0) == (a_first != 0)) {
                    const uint64_t t0 = now_ns();
                    rca = omega_gpu_gemm_ld(t, OMEGA_GPU_GEMM_B_COL, s->M, s->N, s->K, A, s->K, B, s->K, CA, NULL);
                    wa = now_ns() - t0;
                } else {
                    const uint64_t t0 = now_ns();
                    rcb = omega_gpu_gemm_ex(&x);
                    wb = now_ns() - t0;
                }
            }
            const int if1 = in_flight();
            /* correctness gate */
            const int same = rca == 0 && rcb == 0 && memcmp(CA, CB, cn * 4) == 0;
            const uint64_t ref_bad = rca == 0 ? ref_bad_of(fp8, s, A, B, CA, seed ^ ((uint64_t)round << 32) ^ q) : 64u;
            const int ok = same && ref_bad == 0 && xi.route == OMEGA_GEMM_ROUTE_NATIVE && xi.host_fixed == 0;
            bad_total += !ok;
            printf("pair round=%u warm=%s seq=%zu type=%s M=%u N=%u K=%u tag=\"%s\" first=%s rc_a=%s rc_b=%s wall_a_ns=%" PRIu64
                   " wall_b_ns=%" PRIu64 " route_b=%u core_calls_b=%" PRIu64 " host_fixed_b=%" PRIu64 " bitwise=%d ref_bad=%" PRIu64
                   " gate=%s if_before=%d if_after=%d\n",
                   round, round == 0 ? "True" : "False", q, fp8 ? "e4m3" : "bf16", s->M, s->N, s->K, s->tag, a_first ? "A" : "B",
                   omega_gpu_gemm_rc_name(rca), omega_gpu_gemm_rc_name(rcb), wa, wb, xi.route, xi.core_calls, xi.host_fixed, same,
                   ref_bad, ok ? "PASS" : "FAIL", if0, if1);
            fflush(stdout);
        }
    }
    {   /* negative control: expert down, bf16, mutant 1 */
        const Shape *s = &SH[7];
        fill(0, A, (uint64_t)s->M * s->K, 1); fill(0, B, (uint64_t)s->N * s->K, 2);
        const size_t cn = (size_t)s->M * s->N;
        memset(CA, 0xff, cn * 4); memset(CB, 0xff, cn * 4);
        const int rca = omega_gpu_gemm_ld(OMEGA_GPU_GEMM_BF16, OMEGA_GPU_GEMM_B_COL, s->M, s->N, s->K, A, s->K, B, s->K, CA, NULL);
        OmegaGemmExInfo xi; memset(&xi, 0, sizeof xi);
        OmegaGemmExArgs x; memset(&x, 0, sizeof x);
        x.op_a = OMEGA_GEMM_OP_N; x.op_b = OMEGA_GEMM_OP_T; x.m = s->M; x.n = s->N; x.k = s->K; x.batch = 1;
        x.a_type = x.b_type = OMEGA_GEMM_DT_BF16; x.c_type = OMEGA_GEMM_DT_F32;
        x.a = A; x.lda = s->K; x.b = B; x.ldb = s->K; x.c = CB; x.ldc = s->N; x.alpha = 1.0f; x.beta = 0.0f; x.info = &xi;
        omega_gpu_gemm_ex_test_set_mutant(1);
        const int rcb = omega_gpu_gemm_ex(&x);
        omega_gpu_gemm_ex_test_set_mutant(0);
        const int same = memcmp(CA, CB, cn * 4) == 0;
        const uint64_t rb = rcb == 0 ? ref_bad_of(0, s, A, B, CB, 3) : 64u;
        const uint64_t ra = rca == 0 ? ref_bad_of(0, s, A, B, CA, 3) : 64u;
        const int red = rca == 0 && ra == 0 && rcb == 0 && !same && rb > 0;
        printf("control mutant=1 cell=\"%s\" rc_a=%s rc_b=%s bitwise=%d ref_bad_a=%" PRIu64 " ref_bad_b=%" PRIu64 " verdict=%s\n",
               s->tag, omega_gpu_gemm_rc_name(rca), omega_gpu_gemm_rc_name(rcb), same, ra, rb, red ? "RED (expected)" : "GREEN (control failed)");
        if (!red) bad_total++;
    }
    printf("run: %s (%d pairs failed the correctness gate of %zu)\n", bad_total ? "FAIL" : "PASS", bad_total, n * (rounds + 1u));
    free(A); free(B); free(CA); free(CB);
    return bad_total ? 1 : 0;
}
