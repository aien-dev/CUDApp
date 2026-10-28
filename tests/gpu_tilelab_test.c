/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * gpu_tilelab_test: TileLab chip driver (aien-architecture-mv#347 T2/T3). A/B of GEMM tile schedules on the GB10.
 * The schedules (rt ct wm wn band) come from the command line; the native .om planner (omega-mv seed/tilelab/tilelab.om)
 * chose them. This file contains no planner: it only runs the cfgs it is given through omega_gpu_gemm_test_set_cfg.
 *   --host-only   refusals and oracle self-tests, no device
 *   (default)     chip (caller holds flock /tmp/aien-gb10.lock): per shape and cfg, one verified run against an f64 oracle,
 *                 then interleaved timed rounds; red controls; JSON receipt with every raw sample.
 *   --shapes "M,N,K;..."  --cfg "name:rt,ct,wm,wn,band" (repeatable; "default" is always arm 0)  --rounds N  --out FILE
 * Oracle: f64 sum of bf16 products (inputs decoded arithmetically). Bound: |c - ref| <= (K/16 + 2) 2^-22 sum|a b| per element
 * (one tensor-core step per 16 values of k, each step rounded at most 2^-23 relative to the running partial sum, doubled for slack).
 * Large shapes check the first 32 rows and 32 spread rows (all columns), small shapes every element.
 */
#include "omega_gpu_gemm.h"
#include "omega_gpu_session.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCFG 16
#define MAXSH 16
#define MAXR 64
typedef struct { char name[40]; OmegaGemmCfg c; } Arm;
static Arm arms[MAXCFG]; static int narms;
static uint64_t rng = 0x243f6a8885a308d3ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static uint16_t rbf(void) { float f = ((float)(rnd() & 0xffff) / 32768.0f) - 1.0f; uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static double bf(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double pct(double *v, int n, double p) { qsort(v, n, sizeof *v, cmpd); int i = (int)ceil(p * n) - 1; return v[i < 0 ? 0 : i]; }

/* returns number of wrong elements; *worst = max (|c-ref| / bound) */
static uint64_t oracle(uint32_t M, uint32_t N, uint32_t K, const uint16_t *A, const uint16_t *B, const float *C, double *worst, uint64_t *checked) {
    uint64_t bad = 0, ck = 0; *worst = 0;
    int full = (double)M * N * K <= 2.2e9;
    for (uint32_t r = 0; r < M; r++) {
        if (!full && !(r < 32 || r % (M / 32 + 1) == 0)) continue;
        for (uint32_t c = 0; c < N; c++) {
            double s = 0, a = 0;
            for (uint32_t k = 0; k < K; k++) { double p = bf(A[(size_t)r * K + k]) * bf(B[(size_t)k * N + c]); s += p; a += fabs(p); }
            double bound = ((double)(K / 16 + 2)) * ldexp(1.0, -22) * a + 1e-30, e = fabs((double)C[(size_t)r * N + c] - s);
            if (!(e <= bound)) bad++;                       /* NaN counts as wrong */
            if (e / bound > *worst || e != e) *worst = e != e ? INFINITY : e / bound;
            ck++;
        }
    }
    *checked = ck; return bad;
}

static int parse_shape(const char *s, uint32_t sh[][3], int *n) {
    *n = 0;
    while (*s && *n < MAXSH) { unsigned m, nn, k; int used = 0; if (sscanf(s, "%u,%u,%u%n", &m, &nn, &k, &used) < 3 || !m || !nn || !k) return -1; sh[*n][0] = m; sh[*n][1] = nn; sh[*n][2] = k; (*n)++; s += used; if (*s == ';') s++; }
    return *n ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *out = NULL, *shapes = "1024,1024,1024"; int host = 0, rounds = 7;
    uint32_t sh[MAXSH][3]; int nsh;
    omega_gpu_gemm_default_cfg(OMEGA_GPU_GEMM_BF16, &arms[0].c); strcpy(arms[0].name, "default"); narms = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) host = 1;
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--shapes") && i + 1 < argc) shapes = argv[++i];
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) { rounds = atoi(argv[++i]); if (rounds < 1 || rounds > MAXR) { fprintf(stderr, "rounds 1..%d\n", MAXR); return 2; } }
        else if (!strcmp(argv[i], "--cfg") && i + 1 < argc && narms < MAXCFG) {
            Arm *a = &arms[narms]; unsigned rt, ct, wm, wn, band; char nm[40];
            if (sscanf(argv[++i], "%39[^:]:%u,%u,%u,%u,%u", nm, &rt, &ct, &wm, &wn, &band) != 6) { fprintf(stderr, "bad --cfg\n"); return 2; }
            omega_gpu_gemm_default_cfg(OMEGA_GPU_GEMM_BF16, &a->c); a->c.rt = rt; a->c.ct = ct; a->c.wm = wm; a->c.wn = wn; a->c.band = band;
            strcpy(a->name, nm); narms++;
        } else { fprintf(stderr, "usage: %s [--host-only] [--shapes S] [--cfg name:rt,ct,wm,wn,band]... [--rounds N] [--out FILE]\n", argv[0]); return 2; }
    }
    if (parse_shape(shapes, sh, &nsh)) { fprintf(stderr, "bad --shapes\n"); return 2; }
    int fails = 0;
    /* refusals: an arm the API refuses is reported, never run */
    int ok[MAXCFG];
    for (int a = 0; a < narms; a++) { ok[a] = omega_gpu_gemm_cfg_ok(&arms[a].c); printf("arm %s rt=%u ct=%u wm=%u wn=%u band=%u: %s\n", arms[a].name, arms[a].c.rt, arms[a].c.ct, arms[a].c.wm, arms[a].c.wn, arms[a].c.band, ok[a] ? "accepted" : "REFUSED"); }
    { OmegaGemmCfg bad = arms[0].c; bad.rt = 3; int r1 = omega_gpu_gemm_cfg_ok(&bad); bad = arms[0].c; bad.wm = 3; int r2 = omega_gpu_gemm_cfg_ok(&bad);
      printf("red: rt=3 refused=%s, wm=3 refused=%s\n", !r1 ? "yes" : "NO", !r2 ? "yes" : "NO"); if (r1 || r2) fails++; }
    { /* oracle self-test: one element moved by 1e-3 relative must be flagged */
      uint16_t A[16 * 32], B[32 * 8]; float C[16 * 8]; double w; uint64_t ck;
      for (int i = 0; i < 16 * 32; i++) A[i] = rbf();
        for (int i = 0; i < 32 * 8; i++) B[i] = rbf();
      for (int r = 0; r < 16; r++) for (int c = 0; c < 8; c++) { double s = 0; for (int k = 0; k < 32; k++) s += bf(A[r * 32 + k]) * bf(B[k * 8 + c]); C[r * 8 + c] = (float)s; }
      uint64_t good = oracle(16, 8, 32, A, B, C, &w, &ck); C[37] *= 1.001f; C[37] += 0.01f; uint64_t bad = oracle(16, 8, 32, A, B, C, &w, &ck);
      printf("oracle self-test: exact result %" PRIu64 " wrong, planted error %" PRIu64 " wrong\n", good, bad); if (good || !bad) fails++; }
    if (host) { printf("TILELAB_CHIP host: %s\n", fails ? "FAIL" : "PASS"); return fails ? 1 : 0; }

    FILE *js = out ? fopen(out, "w") : NULL; if (out && !js) { perror(out); return 2; }
    if (js) fprintf(js, "{\"tool\":\"gpu_tilelab_test\",\"rounds\":%d,\"cells\":[", rounds);
    int firstcell = 1;
    for (int s = 0; s < nsh; s++) {
        uint32_t M = sh[s][0], N = sh[s][1], K = sh[s][2];
        uint16_t *A = malloc((size_t)M * K * 2), *B = malloc((size_t)K * N * 2); float *C = malloc((size_t)M * N * 4);
        if (!A || !B || !C) { printf("NOT_RUN: host allocation\n"); return 2; }
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rbf();
        for (size_t i = 0; i < (size_t)K * N; i++) B[i] = rbf();
        double samp[MAXCFG][MAXR]; int ns[MAXCFG] = {0}; double worst[MAXCFG]; uint64_t wrong[MAXCFG], chk[MAXCFG]; uint32_t gpr[MAXCFG]; double packms[MAXCFG];
        memset(wrong, 0, sizeof wrong);
        for (int a = 0; a < narms; a++) {
            if (!ok[a]) continue;
            omega_gpu_gemm_test_set_cfg(OMEGA_GPU_GEMM_BF16, &arms[a].c);
            OmegaGpuGemmInfo info; memset(C, 0xff, (size_t)M * N * 4);
            int rc = omega_gpu_gemm(OMEGA_GPU_GEMM_BF16, 0, M, N, K, A, B, C, &info);
            if (rc) { printf("%ux%ux%u %s: chip rc %s: %s\n", M, N, K, arms[a].name, omega_gpu_gemm_rc_name(rc), omega_gpu_session_last_error()); ok[a] = 0; fails++; continue; }
            wrong[a] = oracle(M, N, K, A, B, C, &worst[a], &chk[a]); gpr[a] = info.gpr[OMEGA_GEMM_K_MAIN]; packms[a] = info.pack_ns / 1e6;
            if (wrong[a]) fails++;
        }
        for (int r = 0; r < rounds; r++)
            for (int j = 0; j < narms; j++) {
                int a = (j + r) % narms; if (!ok[a] || wrong[a]) continue;
                omega_gpu_gemm_test_set_cfg(OMEGA_GPU_GEMM_BF16, &arms[a].c);
                OmegaGpuGemmInfo info; omega_gpu_gemm(OMEGA_GPU_GEMM_BF16, 0, M, N, K, A, B, C, &info); /* warm: builds the kernels */
                if (omega_gpu_gemm(OMEGA_GPU_GEMM_BF16, 0, M, N, K, A, B, C, &info)) { fails++; continue; }
                samp[a][ns[a]++] = info.main_ns / 1e6;
            }
        for (int a = 0; a < narms; a++) {
            const Arm *m = &arms[a];
            if (!ok[a] || wrong[a] || !ns[a]) { printf("%ux%ux%u %-12s %s\n", M, N, K, m->name, ok[a] ? "WRONG RESULT" : "NOT_RUN/REFUSED"); continue; }
            double sc[MAXR]; memcpy(sc, samp[a], ns[a] * sizeof(double));
            double p50 = pct(sc, ns[a], 0.5), p95 = pct(sc, ns[a], 0.95), tf = 2.0 * M * N * K / (p50 * 1e-3) / 1e12;
            printf("%ux%ux%u %-12s main p50 %.4f ms p95 %.4f ms %.2f TFLOPS gpr %u worst err/bound %.3f (%" PRIu64 " checked)\n", M, N, K, m->name, p50, p95, tf, gpr[a], worst[a], chk[a]);
            if (js) {
                fprintf(js, "%s{\"m\":%u,\"n\":%u,\"k\":%u,\"arm\":\"%s\",\"rt\":%u,\"ct\":%u,\"wm\":%u,\"wn\":%u,\"band\":%u,\"gpr_main\":%u,\"p50_ms\":%.6f,\"p95_ms\":%.6f,\"tflops_p50\":%.4f,\"pack_ms_first\":%.6f,\"worst_err_over_bound\":%.6f,\"checked\":%" PRIu64 ",\"wrong\":%" PRIu64 ",\"samples_ms\":[", firstcell ? "" : ",", M, N, K, m->name, m->c.rt, m->c.ct, m->c.wm, m->c.wn, m->c.band, gpr[a], p50, p95, tf, packms[a], worst[a], chk[a], wrong[a]);
                for (int i = 0; i < ns[a]; i++) fprintf(js, "%s%.6f", i ? "," : "", samp[a][i]);
                fprintf(js, "]}"); firstcell = 0;
            }
        }
        free(A); free(B); free(C);
    }
    /* red controls on the chip: the three API mutants must produce wrong results under the oracle */
    int caught = 0;
    for (int mut = 1; mut <= 3; mut++) {
        OmegaGemmCfg c = arms[0].c; c.mutant = mut; omega_gpu_gemm_test_set_cfg(OMEGA_GPU_GEMM_BF16, &c);
        uint16_t A[65 * 100], B[100 * 63]; float C[65 * 63]; double w; uint64_t ck;
        for (int i = 0; i < 65 * 100; i++) A[i] = rbf();
        for (int i = 0; i < 100 * 63; i++) B[i] = rbf();
        int rc = omega_gpu_gemm(OMEGA_GPU_GEMM_BF16, 0, 65, 63, 100, A, B, C, NULL);
        uint64_t bad = rc ? 1 : oracle(65, 63, 100, A, B, C, &w, &ck);
        printf("red: mutant %d %s (%" PRIu64 " wrong)\n", mut, bad ? "CAUGHT" : "NOT CAUGHT", bad); caught += bad > 0;
    }
    if (caught != 3) fails++;
    omega_gpu_gemm_test_set_cfg(OMEGA_GPU_GEMM_BF16, NULL);
    if (js) { fprintf(js, "],\"red_mutants_caught\":%d,\"fails\":%d}\n", caught, fails); fclose(js); }
    printf("TILELAB_CHIP: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
