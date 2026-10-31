/* gemm_ab.c -- GemmTune A/B timing driver for omega_gpu_gemm (CUDApp C3 general GEMM).
 * Measurement harness only, not product code: it is compiled against each arm's own tests/gpu_gemm_test.c (fill, compare,
 * the host model) and build/libomega_gpu.a, so arm A and arm B run the same inputs through the same checker.
 * One process = one pass of one arm over the shape set, in a seeded order. Every timed call is compared against the host
 * model (compare(): full for shapes up to 2^27 multiply-adds, else 4096 sampled elements plus the last row and column) and
 * records the live server's in_flight from 127.0.0.1:18091/health before and after.
 *   gemm_ab --arm A|B --round R --seed S [--set ab|explore] [--cfg bf16|e4m3 rt,ct,wm,wn,stall[,band]]...
 *   (",band" only when built with -DGEMM_AB_BAND against a tree whose OmegaGemmCfg has the band field)
 * Why C: it must link the C library under test and reuse the C checker verbatim; no .om binding of omega_gpu_gemm exists.
 * It lives in the evidence folder as part of the receipt. */
#define main gemm_test_main
#include "gpu_gemm_test.c"
#undef main

static int in_flight(void) {
    FILE *f = popen("curl -s -m 2 127.0.0.1:18091/health", "r");
    if (!f) return -1;
    char buf[1024]; size_t n = fread(buf, 1, sizeof buf - 1, f); buf[n] = 0;
    pclose(f);
    const char *p = strstr(buf, "\"in_flight\"");
    if (!p || !(p = strchr(p, ':'))) return -1;
    return atoi(p + 1);
}

typedef struct { uint32_t M, N, K; unsigned lay; const char *tag; } Shape;
static const Shape AB[] = {
    /* serving shapes of the 35B (hidden 2048), A = activations row-major, B = weights [out][in] = column-major */
    { 1, 8192, 2048, 2, "q+gate decode" },   { 16, 12288, 2048, 2, "gdn in_proj b16" }, { 512, 8192, 2048, 2, "q+gate p512" },
    { 512, 2048, 4096, 2, "o_proj p512" },   { 2048, 12288, 2048, 2, "gdn in_proj p2048" }, { 2048, 2048, 4096, 2, "out_proj p2048" },
    { 64, 1024, 2048, 2, "expert gate+up" }, { 64, 2048, 512, 2, "expert down" },      { 2048, 16384, 2048, 2, "lm_head slice" },
    /* the #112 ladder, A row B row as in evidence/C3-GEMM */
    { 1024, 1024, 1024, 0, "ladder" }, { 2048, 2048, 2048, 0, "ladder" }, { 4096, 4096, 4096, 0, "ladder" }, { 8192, 8192, 8192, 0, "ladder" },
    { 3000, 5000, 700, 0, "ladder" },  { 64, 8192, 8192, 0, "ladder" },   { 8192, 64, 8192, 0, "ladder" },   { 1000, 999, 1001, 0, "ladder" },
};
static const Shape EXPLORE[] = {
    { 512, 8192, 2048, 2, "q+gate p512" }, { 2048, 12288, 2048, 2, "gdn in_proj p2048" }, { 2048, 2048, 2048, 0, "ladder" },
    { 4096, 4096, 4096, 0, "ladder" },     { 64, 8192, 8192, 0, "ladder" },
};

int main(int argc, char **argv) {
    const char *arm = "?"; unsigned round = 0; uint64_t seed = 20261027; int explore = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--arm") && i + 1 < argc) arm = argv[++i];
        else if (!strcmp(argv[i], "--round") && i + 1 < argc) round = (unsigned)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--set") && i + 1 < argc) { i++; explore = !strcmp(argv[i], "explore"); }
        else if (!strcmp(argv[i], "--cfg") && i + 2 < argc) {
            OmegaGemmCfg c; memset(&c, 0, sizeof c);
            c.type = !strcmp(argv[i + 1], "bf16") ? OMEGA_GPU_GEMM_BF16 : OMEGA_GPU_GEMM_E4M3;
            unsigned band = 0;
            int nf = sscanf(argv[i + 2], "%u,%u,%u,%u,%u,%u", &c.rt, &c.ct, &c.wm, &c.wn, &c.stall, &band);
#ifdef GEMM_AB_BAND
            c.band = band;
#else
            if (nf == 6) nf = -1;
#endif
            if ((nf != 5 && nf != 6) || omega_gpu_gemm_test_set_cfg(c.type, &c) != 0) {
                fprintf(stderr, "bad --cfg %s %s\n", argv[i + 1], argv[i + 2]); return 2;
            }
            i += 2;
        } else { fprintf(stderr, "usage: %s --arm A|B --round R [--seed S] [--set ab|explore] [--cfg bf16|e4m3 rt,ct,wm,wn,stall]\n", argv[0]); return 2; }
    }
    const Shape *sh = explore ? EXPLORE : AB;
    const size_t ns = explore ? sizeof EXPLORE / sizeof EXPLORE[0] : sizeof AB / sizeof AB[0];
    const size_t big = (size_t)8192 * 8192 * 2;
    void *A = malloc(big), *B = malloc(big); float *C = malloc((size_t)8192 * 16384 * 4);
    if (!A || !B || !C) { printf("NOT_RUN: host allocation\n"); return 2; }
    /* order: every (type, shape) once, Fisher-Yates with a generator seeded by (seed, round, arm) -- same order for A and B in a round */
    size_t n = 2 * ns, ord[64];
    for (size_t i = 0; i < n; i++) ord[i] = i;
    uint64_t r = seed * 1000003u + round;
    for (size_t i = n - 1; i > 0; i--) { r = r * 6364136223846793005ull + 1442695040888963407ull; size_t j = (size_t)(r >> 33) % (i + 1); size_t t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
    /* warm-up per type (kernel build + first launch), reported and excluded */
    for (int t = 0; t < 2; t++) {
        fill((OmegaGpuGemmType)t, A, 65 * 100, 77); fill((OmegaGpuGemmType)t, B, 100 * 63, 78);
        OmegaGpuGemmInfo info; int rc = omega_gpu_gemm((OmegaGpuGemmType)t, 0, 65, 63, 100, A, B, C, &info);
        printf("warm arm=%s type=%s rc=%s gpr_main=%u\n", arm, tname((OmegaGpuGemmType)t), omega_gpu_gemm_rc_name(rc), info.gpr[OMEGA_GEMM_K_MAIN]);
    }
    int bad_total = 0;
    for (size_t q = 0; q < n; q++) {
        const OmegaGpuGemmType t = (OmegaGpuGemmType)(ord[q] / ns);
        const Shape *s = &sh[ord[q] % ns];
        fill(t, A, (uint64_t)s->M * s->K, (uint64_t)s->M * 3 + s->N * 5 + s->K * 7 + s->lay);
        fill(t, B, (uint64_t)s->K * s->N, (uint64_t)s->M * 11 + s->N * 13 + s->K * 17 + s->lay);
        const int if0 = in_flight();
        OmegaGpuGemmInfo info; memset(&info, 0, sizeof info);
        int rc = omega_gpu_gemm(t, s->lay, s->M, s->N, s->K, A, B, C, &info);
        const int if1 = in_flight();
        char first[200] = ""; uint64_t checked = 0, bad = rc == 0 ? compare(t, s->lay, s->M, s->N, s->K, A, B, C, &checked, first, sizeof first) : 1;
        bad_total += rc != 0 || bad != 0;
        printf("sample arm=%s round=%u seq=%zu type=%s lay=%u M=%u N=%u K=%u tag=\"%s\" rc=%s checked=%" PRIu64 " wrong=%" PRIu64 " main_ns=%" PRIu64
               " pack_ns=%" PRIu64 " call_ns=%" PRIu64 " gpr_main=%u if_before=%d if_after=%d%s%s\n",
               arm, round, q, tname(t), s->lay, s->M, s->N, s->K, s->tag, omega_gpu_gemm_rc_name(rc), checked, bad, info.main_ns, info.pack_ns,
               info.call_ns, info.gpr[OMEGA_GEMM_K_MAIN], if0, if1, bad ? " first=" : "", bad ? first : "");
        fflush(stdout);
    }
    printf("pass arm=%s round=%u: %s (%d wrong or failed of %zu)\n", arm, round, bad_total ? "FAIL" : "PASS", bad_total, n);
    free(A); free(B); free(C);
    return bad_total ? 1 : 0;
}
