/* gpu_elementwise_test: FB-1 cut 4 gate for omega_gpu_elementwise_api.
 *
 *   ./gpu_elementwise_test --host-only          refusals, codegen, word fixtures, register budget,
 *                                               nvdisasm listing check when /usr/local/cuda/bin/nvdisasm exists
 *   ./gpu_elementwise_test --out receipt.json   chip gate: every op against the host oracle, then its mutant
 *                                               (a deliberately wrong kernel) which the same check must catch
 *   ./gpu_elementwise_test --timing --out t.json cut 4b gate: persistent device, 100 calls each of rmsnorm 1x2048
 *                                               and swiglu 5632 under 0.5 ms median, one device open per process
 *
 * Oracles re-state aien-sovereign-core crates/aien-inference-abi/src/tensor.rs (main 5fdab70) in C:
 * rmsnorm sums in f64, rope computes cos/sin in f64 and casts to f32 before f32 math, swiglu is f64.
 * This file is compiled with -ffp-contract=off so the f32 oracle math has no hidden fused multiply-add.
 *
 * Tolerances (why): EX2 1e-6 rel (PTX ex2.approx.f32: 2^-22.5 rel, we allow 2x). XCHG exact.
 * RMSNORM 1e-5 rel + 1e-6 abs (f32 sum of up to 16384 squares against an f64 sum, plus RSQ seed with one
 * Newton step; the mutant moves the scale by about 40 percent). ROPE exact (same f32 ops, same order).
 * SWIGLU 1e-5 rel + 1e-6 abs (EX2 and RCP approximations, each about 2^-22 rel; the mutant flips the sign
 * of the exponent, which is wrong by a factor of e^(2g)).
 */
#include "omega_gpu_elementwise_api.h"
#include "omega_blackwell_codegen.h"
#include "omega_gpu_session.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }

/* ----------------------------------------------------------------- oracles */
static void oracle_rmsnorm(const float *x, const float *w, float eps, float *out, size_t dim) {
    double ss = 0.0;
    for (size_t i = 0; i < dim; i++) ss += (double)x[i] * (double)x[i];
    double mean = ss / (double)dim;
    float scale = (float)(1.0 / sqrt(mean + (double)eps));
    for (size_t i = 0; i < dim; i++) { float t = x[i] * scale; out[i] = t * w[i]; }
}
static void oracle_rope_tables(size_t head_dim, size_t pos, float theta, float *cos_half, float *sin_half) {
    size_t half = head_dim / 2;
    for (size_t i = 0; i < half; i++) {
        double exponent = (double)(2 * i) / (double)head_dim;
        double freq = 1.0 / pow((double)theta, exponent);
        double rot = (double)pos * freq;
        sin_half[i] = (float)sin(rot);
        cos_half[i] = (float)cos(rot);
    }
}
static void oracle_rope(const float *v, size_t heads, size_t head_dim, const float *cos_half, const float *sin_half, float *out) {
    size_t half = head_dim / 2;
    for (size_t h = 0; h < heads; h++)
        for (size_t i = 0; i < half; i++) {
            size_t i0 = h * head_dim + i, i1 = i0 + half;
            float q0 = v[i0], q1 = v[i1], c = cos_half[i], s = sin_half[i];
            float a = q0 * c, b = q1 * s; out[i0] = a - b;
            float d = q0 * s, e = q1 * c; out[i1] = d + e;
        }
}
static void oracle_swiglu(const float *g, const float *u, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) { double gg = g[i]; double silu = gg / (1.0 + exp(-gg)); out[i] = (float)(silu * (double)u[i]); }
}

/* |got - want| <= rel*|want| + abs; returns the number of violations and the worst scaled error */
static size_t compare(const float *got, const float *want, size_t n, double rel, double abs_tol, double *worst) {
    size_t bad = 0; *worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)want[i]);
        double lim = rel * fabs((double)want[i]) + abs_tol;
        double scaled = lim > 0 ? d / lim : (d > 0 ? INFINITY : 0.0);
        if (scaled > *worst) *worst = scaled;
        if (!(d <= lim)) bad++; /* NaN counts as bad */
    }
    return bad;
}

/* ------------------------------------------------------------- host only */
static int nvdisasm_check(OmegaGpuEwOp op, const char *name, const char *const *must, int n_must) {
    const char *nv = "/usr/local/cuda/bin/nvdisasm";
    if (access(nv, X_OK) != 0) { printf("nvdisasm: not present, listing check skipped for %s\n", name); return 0; }
    OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
    if (omega_gpu_elementwise_codegen(op, NULL, 0.0f, &k) != OMEGA_GPU_EW_OK) return -1;
    char bin[64], lst[64], cmd[256];
    snprintf(bin, sizeof bin, "/tmp/omega_ew_%s_%d.bin", name, (int)getpid());
    snprintf(lst, sizeof lst, "/tmp/omega_ew_%s_%d.sass", name, (int)getpid());
    FILE *f = fopen(bin, "wb"); if (!f) { free(k.code); return -1; }
    fwrite(k.code, 1, k.code_size, f); fclose(f);
    snprintf(cmd, sizeof cmd, "%s -b SM121 %s > %s 2>&1", nv, bin, lst);
    int rc = system(cmd);
    free(k.code);
    FILE *l = fopen(lst, "r"); if (!l) return -1;
    char line[512]; int seen[16] = {0}; int bad = 0, insns = 0;
    while (fgets(line, sizeof line, l)) {
        if (!strstr(line, "/*0")) continue;
        insns++;
        if (strstr(line, "?") || strstr(line, "error")) bad++;
        for (int i = 0; i < n_must; i++) if (strstr(line, must[i])) seen[i] = 1;
    }
    fclose(l);
    int miss = 0;
    for (int i = 0; i < n_must; i++) if (!seen[i]) { printf("  %s: nvdisasm listing lacks %s\n", name, must[i]); miss++; }
    printf("nvdisasm %s: rc=%d insns=%d undecodable=%d missing=%d (%s)\n", name, rc, insns, bad, miss, lst);
    unlink(bin);
    return (rc == 0 && bad == 0 && miss == 0) ? 0 : -1;
}

static void host_only(void) {
    CHECK(omega_blackwell_verify_codegen_fixtures_fb1cut4() == 0, "word fixtures MUFU_EX2 / BAR_SYNC / LDS32 / STS32 (rc %d)", omega_blackwell_verify_codegen_fixtures_fb1cut4());
    float a[256] = {0}, b[256] = {0}, c[256] = {0};
    CHECK(omega_gpu_rmsnorm_f32(0, 128, a, b, 1e-5f, c, NULL) == OMEGA_GPU_EW_BAD_ARGS, "rows=0 refused");
    CHECK(omega_gpu_rmsnorm_f32(1, 100, a, b, 1e-5f, c, NULL) == OMEGA_GPU_EW_BAD_ARGS, "dim not a multiple of 128 refused");
    CHECK(omega_gpu_rmsnorm_f32(1, OMEGA_GPU_EW_MAX_DIM + 128, a, b, 1e-5f, c, NULL) == OMEGA_GPU_EW_TOO_LARGE, "dim over limit refused");
    CHECK(omega_gpu_rope_f32(1, 63, a, b, b, c, NULL) == OMEGA_GPU_EW_BAD_ARGS, "odd head_dim refused");
    CHECK(omega_gpu_rope_f32(1, 64, NULL, b, b, c, NULL) == OMEGA_GPU_EW_BAD_ARGS, "NULL vector refused");
    CHECK(omega_gpu_swiglu_f32(0, a, b, c, NULL) == OMEGA_GPU_EW_BAD_ARGS, "n=0 refused");
    CHECK(omega_gpu_shared_xchg_u32(100, (uint32_t *)a, (uint32_t *)c, NULL) == OMEGA_GPU_EW_BAD_ARGS, "xchg n not a multiple of 128 refused");
    CHECK(strcmp(omega_gpu_elementwise_rc_name(OMEGA_GPU_EW_UNWRITTEN), "UNWRITTEN") == 0, "rc name");

    struct { OmegaGpuEwOp op; const char *name; const char *must[6]; int n; } ops[] = {
        { OMEGA_GPU_EW_EX2, "ex2", { "MUFU.EX2", "@P0 EXIT", "STG.E" }, 3 },
        { OMEGA_GPU_EW_XCHG, "xchg", { "STS [R", "BAR.SYNC", "LDS R" }, 3 },
        { OMEGA_GPU_EW_RMSNORM, "rmsnorm", { "SHFL.DOWN", "BAR.SYNC", "MUFU.RSQ", "@!P0 BRA", "STS [R", "LDS R" }, 6 },
        { OMEGA_GPU_EW_ROPE, "rope", { "FADD", "FMUL", "STG.E" }, 3 },
        { OMEGA_GPU_EW_SWIGLU, "swiglu", { "MUFU.EX2", "MUFU.RCP", "@P0 EXIT" }, 3 },
    };
    for (size_t i = 0; i < sizeof ops / sizeof ops[0]; i++) {
        OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
        int rc = omega_gpu_elementwise_codegen(ops[i].op, NULL, 0.0f, &k);
        CHECK(rc == OMEGA_GPU_EW_OK && k.code_size > 0 && k.gpr_count <= 64, "%s kernel generated (rc %s, %zu insns, %u gprs)", ops[i].name, omega_gpu_elementwise_rc_name(rc), k.insn_count, k.gpr_count);
        printf("codegen %s: %zu insns, %u gprs, %u ugprs\n", ops[i].name, k.insn_count, k.gpr_count, k.uniform_gpr_count);
        uint8_t digest[32]; memcpy(digest, k.code_digest, 32);
        free(k.code);
        /* the mutant must be a different kernel */
        omega_gpu_elementwise_test_set_mutant((int)ops[i].op);
        memset(&k, 0, sizeof k);
        CHECK(omega_gpu_elementwise_codegen(ops[i].op, NULL, 0.0f, &k) == OMEGA_GPU_EW_OK && memcmp(digest, k.code_digest, 32) != 0, "%s mutant kernel differs", ops[i].name);
        free(k.code);
        omega_gpu_elementwise_test_set_mutant(0);
        CHECK(nvdisasm_check(ops[i].op, ops[i].name, ops[i].must, ops[i].n) == 0, "%s nvdisasm listing decodes and shows the expected instructions", ops[i].name);
    }
}

/* ------------------------------------------------------------- chip gate */
typedef struct { const char *name; size_t n; int rc; size_t bad; double worst; int pass; int mutant_rc; size_t mutant_bad; int mutant_caught; uint64_t chip_ns; uint32_t calls; uint32_t unwritten; } Case;
static Case g_cases[16]; static int g_ncases;

static void record(Case c, FILE *out) {
    g_cases[g_ncases++] = c;
    CHECK(c.pass, "%s rc=%s violations=%zu worst=%g", c.name, omega_gpu_elementwise_rc_name(c.rc), c.bad, c.worst);
    CHECK(c.mutant_caught, "%s mutant caught (rc=%s violations=%zu)", c.name, omega_gpu_elementwise_rc_name(c.mutant_rc), c.mutant_bad);
    printf("%s %s n=%zu rc=%s violations=%zu worst_scaled_err=%g chip_ns=%" PRIu64 " calls=%u unwritten=%u | mutant rc=%s violations=%zu %s\n",
           c.pass && c.mutant_caught ? "PASS" : "FAIL", c.name, c.n, omega_gpu_elementwise_rc_name(c.rc), c.bad, c.worst, c.chip_ns, c.calls, c.unwritten,
           omega_gpu_elementwise_rc_name(c.mutant_rc), c.mutant_bad, c.mutant_caught ? "CAUGHT" : "NOT CAUGHT");
    if (out) fprintf(out, "%s{\"case\":\"%s\",\"n\":%zu,\"rc\":\"%s\",\"violations\":%zu,\"worst_scaled_err\":%g,\"pass\":%s,\"chip_elapsed_ns\":%" PRIu64 ",\"chip_calls\":%u,\"unwritten_words\":%u,\"mutant_rc\":\"%s\",\"mutant_violations\":%zu,\"mutant_caught\":%s}",
                     g_ncases > 1 ? "," : "", c.name, c.n, omega_gpu_elementwise_rc_name(c.rc), c.bad, c.worst, c.pass ? "true" : "false", c.chip_ns, c.calls, c.unwritten,
                     omega_gpu_elementwise_rc_name(c.mutant_rc), c.mutant_bad, c.mutant_caught ? "true" : "false");
}

static int chip(const char *out_path) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_ELEMENTWISE_V1\",\"tolerances\":{\"ex2_rel\":1e-6,\"xchg\":\"exact\",\"rmsnorm_rel\":1e-5,\"rmsnorm_abs\":1e-6,\"rope\":\"exact\",\"swiglu_rel\":1e-5,\"swiglu_abs\":1e-6},\"cases\":[");
    uint32_t seed = 0xfb1c0004u;
    OmegaGpuEwInfo info;

    /* 1. EX2 probe, with a negative control on the comparison itself */
    {
        size_t n = 4096; float *x = malloc(n * 4), *got = malloc(n * 4), *want = malloc(n * 4), *wrong = malloc(n * 4);
        for (size_t i = 0; i < n; i++) { x[i] = frand(&seed, -20.0f, 20.0f); want[i] = (float)exp2((double)x[i]); wrong[i] = (float)exp2((double)x[i] + 1.0); }
        Case c = { .name = "ex2_probe", .n = n };
        c.rc = omega_gpu_ex2_f32((uint32_t)n, x, got, &info);
        c.bad = compare(got, want, n, 1e-6, 0.0, &c.worst);
        double w2; size_t neg = compare(got, wrong, n, 1e-6, 0.0, &w2);
        CHECK(neg > n / 2, "ex2 negative control: the check rejects a wrong expectation (%zu of %zu flagged)", neg, n);
        c.pass = c.rc == OMEGA_GPU_EW_OK && c.bad == 0; c.chip_ns = info.elapsed_ns; c.calls = info.chip_calls; c.unwritten = info.unwritten_words;
        omega_gpu_elementwise_test_set_mutant(OMEGA_GPU_EW_EX2);
        c.mutant_rc = omega_gpu_ex2_f32((uint32_t)n, x, got, &info);
        c.mutant_bad = compare(got, want, n, 1e-6, 0.0, &w2);
        c.mutant_caught = c.mutant_rc != OMEGA_GPU_EW_OK || c.mutant_bad > 0;
        omega_gpu_elementwise_test_set_mutant(0);
        record(c, out); free(x); free(got); free(want); free(wrong);
    }
    /* 2. shared-memory exchange probe (STS, BAR.SYNC, LDS) */
    {
        size_t n = 4096; uint32_t *x = malloc(n * 4), *got = malloc(n * 4);
        for (size_t i = 0; i < n; i++) x[i] = lcg(&seed);
        Case c = { .name = "shared_xchg_probe", .n = n };
        c.rc = omega_gpu_shared_xchg_u32((uint32_t)n, x, got, &info);
        for (size_t i = 0; i < n; i++) if (got[i] != x[i ^ (OMEGA_GPU_EW_THREADS - 1)]) c.bad++;
        c.pass = c.rc == OMEGA_GPU_EW_OK && c.bad == 0; c.chip_ns = info.elapsed_ns; c.calls = info.chip_calls; c.unwritten = info.unwritten_words;
        omega_gpu_elementwise_test_set_mutant(OMEGA_GPU_EW_XCHG);
        c.mutant_rc = omega_gpu_shared_xchg_u32((uint32_t)n, x, got, &info);
        for (size_t i = 0; i < n; i++) if (got[i] != x[i ^ (OMEGA_GPU_EW_THREADS - 1)]) c.mutant_bad++;
        c.mutant_caught = c.mutant_rc != OMEGA_GPU_EW_OK || c.mutant_bad > 0;
        omega_gpu_elementwise_test_set_mutant(0);
        record(c, out); free(x); free(got);
    }
    /* 3. rmsnorm at the model hidden sizes (TinyLlama and Llama-3.2-1B: 2048), plus small and large */
    { size_t dims[3] = { 2048, 256, 8192 }; size_t rowss[3] = { 4, 1, 2 };
      for (int t = 0; t < 3; t++) {
        size_t dim = dims[t], rows = rowss[t], n = rows * dim; float eps = 1e-5f;
        float *x = malloc(n * 4), *w = malloc(dim * 4), *got = malloc(n * 4), *want = malloc(n * 4), *again = malloc(n * 4);
        for (size_t i = 0; i < n; i++) x[i] = frand(&seed, -3.0f, 3.0f);
        for (size_t i = 0; i < dim; i++) w[i] = frand(&seed, 0.5f, 1.5f);
        for (size_t r = 0; r < rows; r++) oracle_rmsnorm(x + r * dim, w, eps, want + r * dim, dim);
        char name[64]; snprintf(name, sizeof name, "rmsnorm_%zux%zu", rows, dim);
        Case c = { .name = strdup(name), .n = n };
        c.rc = omega_gpu_rmsnorm_f32((uint32_t)rows, (uint32_t)dim, x, w, eps, got, &info);
        c.bad = compare(got, want, n, 1e-5, 1e-6, &c.worst);
        c.pass = c.rc == OMEGA_GPU_EW_OK && c.bad == 0; c.chip_ns = info.elapsed_ns; c.calls = info.chip_calls; c.unwritten = info.unwritten_words;
        OmegaGpuEwInfo i2; int rc2 = omega_gpu_rmsnorm_f32((uint32_t)rows, (uint32_t)dim, x, w, eps, again, &i2);
        CHECK(rc2 == OMEGA_GPU_EW_OK && i2.kernel_cache_hit && memcmp(got, again, n * 4) == 0, "%s repeat run (cache hit) is bit-identical", name);
        omega_gpu_elementwise_test_set_mutant(OMEGA_GPU_EW_RMSNORM);
        double w2; c.mutant_rc = omega_gpu_rmsnorm_f32((uint32_t)rows, (uint32_t)dim, x, w, eps, got, &info);
        c.mutant_bad = compare(got, want, n, 1e-5, 1e-6, &w2);
        c.mutant_caught = c.mutant_rc != OMEGA_GPU_EW_OK || c.mutant_bad > 0;
        omega_gpu_elementwise_test_set_mutant(0);
        record(c, out); free(x); free(w); free(got); free(want); free(again);
      } }
    /* 4. rope: TinyLlama (32 q heads, 4 kv heads, head_dim 64, theta 10000) and Llama-3.2-1B (32/8, 64, 500000) */
    { struct { size_t heads, hd, pos; float theta; } rs[4] = { { 32, 64, 17, 10000.0f }, { 4, 64, 17, 10000.0f }, { 32, 64, 1023, 500000.0f }, { 8, 64, 1023, 500000.0f } };
      for (int t = 0; t < 4; t++) {
        size_t heads = rs[t].heads, hd = rs[t].hd, n = heads * hd, half = hd / 2;
        float *v = malloc(n * 4), *got = malloc(n * 4), *want = malloc(n * 4), *ch = malloc(half * 4), *sh = malloc(half * 4);
        for (size_t i = 0; i < n; i++) v[i] = frand(&seed, -4.0f, 4.0f);
        oracle_rope_tables(hd, rs[t].pos, rs[t].theta, ch, sh);
        oracle_rope(v, heads, hd, ch, sh, want);
        char name[64]; snprintf(name, sizeof name, "rope_%zuh_%zud_pos%zu_theta%g", heads, hd, rs[t].pos, rs[t].theta);
        Case c = { .name = strdup(name), .n = n };
        c.rc = omega_gpu_rope_f32((uint32_t)heads, (uint32_t)hd, v, ch, sh, got, &info);
        c.bad = compare(got, want, n, 0.0, 0.0, &c.worst);
        c.pass = c.rc == OMEGA_GPU_EW_OK && c.bad == 0; c.chip_ns = info.elapsed_ns; c.calls = info.chip_calls; c.unwritten = info.unwritten_words;
        omega_gpu_elementwise_test_set_mutant(OMEGA_GPU_EW_ROPE);
        double w2; c.mutant_rc = omega_gpu_rope_f32((uint32_t)heads, (uint32_t)hd, v, ch, sh, got, &info);
        c.mutant_bad = compare(got, want, n, 0.0, 0.0, &w2);
        c.mutant_caught = c.mutant_rc != OMEGA_GPU_EW_OK || c.mutant_bad > 0;
        omega_gpu_elementwise_test_set_mutant(0);
        record(c, out); free(v); free(got); free(want); free(ch); free(sh);
      } }
    /* 5. swiglu at the model ffn sizes (TinyLlama 5632, Llama-3.2-1B 8192) and an odd tail */
    { size_t ns[3] = { 5632, 8192, 1000 };
      for (int t = 0; t < 3; t++) {
        size_t n = ns[t];
        float *g = malloc(n * 4), *u = malloc(n * 4), *got = malloc(n * 4), *want = malloc(n * 4);
        for (size_t i = 0; i < n; i++) { g[i] = frand(&seed, -12.0f, 12.0f); u[i] = frand(&seed, -2.0f, 2.0f); }
        oracle_swiglu(g, u, want, n);
        char name[64]; snprintf(name, sizeof name, "swiglu_%zu", n);
        Case c = { .name = strdup(name), .n = n };
        c.rc = omega_gpu_swiglu_f32((uint32_t)n, g, u, got, &info);
        c.bad = compare(got, want, n, 1e-5, 1e-6, &c.worst);
        c.pass = c.rc == OMEGA_GPU_EW_OK && c.bad == 0; c.chip_ns = info.elapsed_ns; c.calls = info.chip_calls; c.unwritten = info.unwritten_words;
        omega_gpu_elementwise_test_set_mutant(OMEGA_GPU_EW_SWIGLU);
        double w2; c.mutant_rc = omega_gpu_swiglu_f32((uint32_t)n, g, u, got, &info);
        c.mutant_bad = compare(got, want, n, 1e-5, 1e-6, &w2);
        c.mutant_caught = c.mutant_rc != OMEGA_GPU_EW_OK || c.mutant_bad > 0;
        omega_gpu_elementwise_test_set_mutant(0);
        record(c, out); free(g); free(u); free(got); free(want);
      } }
    /* 6. stale-SM-instruction-cache guard (cut 1b finding, omega #256): 10 distinct kernels
     * (5 ops + 5 mutants) in rotation, two rounds. Every launch uploads code into a freshly
     * allocated page, so a kernel routinely lands on the code address its predecessor used,
     * and 10 keys evict the 8-slot kernel cache. Each clean launch must match the oracle and
     * each mutant must still be caught; a stale icache would make a mutant pass or a clean
     * kernel fail. Red was observed by cut 1b on matmul before its invalidate (receipt
     * FB1-CUT1B-f629798); this build carries the invalidate, so the case is a guard. */
    {
        enum { RN = 256, RH = 4, RD = 64 };
        float *x = malloc(RN * 4), *w = malloc(RN * 4), *u = malloc(RN * 4), *got = malloc(RN * 4), *want = malloc(RN * 4);
        float *ch = malloc(RD / 2 * 4), *sh = malloc(RD / 2 * 4);
        uint32_t *xu = malloc(RN * 4), *gu = malloc(RN * 4);
        for (size_t i = 0; i < RN; i++) { x[i] = frand(&seed, -3.0f, 3.0f); w[i] = frand(&seed, 0.5f, 1.5f); u[i] = frand(&seed, -2.0f, 2.0f); xu[i] = lcg(&seed); }
        oracle_rope_tables(RD, 17, 10000.0f, ch, sh);
        Case c = { .name = "icache_reuse_rotation", .n = 20 };
        c.rc = OMEGA_GPU_EW_OK; c.mutant_rc = OMEGA_GPU_EW_OK; c.mutant_caught = 1;
        const int order[10] = { OMEGA_GPU_EW_EX2, -OMEGA_GPU_EW_SWIGLU, OMEGA_GPU_EW_XCHG, -OMEGA_GPU_EW_EX2, OMEGA_GPU_EW_RMSNORM,
                                -OMEGA_GPU_EW_XCHG, OMEGA_GPU_EW_SWIGLU, -OMEGA_GPU_EW_ROPE, OMEGA_GPU_EW_ROPE, -OMEGA_GPU_EW_RMSNORM };
        for (int step = 0; step < 20; step++) {
            int round = step / 10, k = step % 10;
            int mut = order[k] < 0, op = mut ? -order[k] : order[k];
            omega_gpu_elementwise_test_set_mutant(mut ? op : 0);
            int rc; size_t bad = 0; double wst;
            switch (op) {
            case OMEGA_GPU_EW_EX2:
                for (size_t i = 0; i < RN; i++) want[i] = (float)exp2((double)x[i]);
                rc = omega_gpu_ex2_f32(RN, x, got, &info); bad = compare(got, want, RN, 1e-6, 0.0, &wst); break;
            case OMEGA_GPU_EW_XCHG:
                rc = omega_gpu_shared_xchg_u32(RN, xu, gu, &info);
                for (size_t i = 0; i < RN; i++) { if (gu[i] != xu[i ^ (OMEGA_GPU_EW_THREADS - 1)]) bad++; }
                break;
            case OMEGA_GPU_EW_RMSNORM:
                oracle_rmsnorm(x, w, 1e-5f, want, RN);
                rc = omega_gpu_rmsnorm_f32(1, RN, x, w, 1e-5f, got, &info); bad = compare(got, want, RN, 1e-5, 1e-6, &wst); break;
            case OMEGA_GPU_EW_ROPE:
                oracle_rope(x, RH, RD, ch, sh, want);
                rc = omega_gpu_rope_f32(RH, RD, x, ch, sh, got, &info); bad = compare(got, want, RH * RD, 0.0, 0.0, &wst); break;
            default:
                oracle_swiglu(x, u, want, RN);
                rc = omega_gpu_swiglu_f32(RN, x, u, got, &info); bad = compare(got, want, RN, 1e-5, 1e-6, &wst); break;
            }
            omega_gpu_elementwise_test_set_mutant(0);
            int ok = rc == OMEGA_GPU_EW_OK && bad == 0;
            if (!mut && !ok) { c.bad++; if (rc != OMEGA_GPU_EW_OK) c.rc = rc; printf("  rotation %d/%d: clean op %d FAILED rc=%s violations=%zu\n", round, k, op, omega_gpu_elementwise_rc_name(rc), bad); }
            if (mut && ok) { c.mutant_bad++; c.mutant_caught = 0; printf("  rotation %d/%d: mutant op %d NOT caught\n", round, k, op); }
            c.chip_ns += info.elapsed_ns; c.calls += info.chip_calls; c.unwritten += info.unwritten_words;
        }
        c.pass = c.bad == 0 && c.rc == OMEGA_GPU_EW_OK;
        record(c, out);
        free(x); free(w); free(u); free(got); free(want); free(ch); free(sh); free(xu); free(gu);
    }
    /* cut 4b: the whole battery (clean and mutant kernels, cache clears) ran on ONE device open */
    CHECK(omega_gpu_session_open_count() == 1, "device opened once for the battery (opens=%u)", omega_gpu_session_open_count());
    printf("device_opens=%u\n", omega_gpu_session_open_count());
    if (out) {
        fprintf(out, "],\"device_opens\":%u,\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", omega_gpu_session_open_count(), g_checks, g_failed, g_failed == 0 ? "OMEGA_GPU_ELEMENTWISE_PASS" : "OMEGA_GPU_ELEMENTWISE_FAIL");
        fclose(out);
    }
    return 0;
}

/* ------------------------------------------------------------ timing gate (cut 4b) */
/* The persistent device: 100 calls each of rmsnorm 1x2048 and swiglu 5632 (the TinyLlama
 * decode shapes), every call checked against the oracle, median wall time per call under
 * 0.5 ms (cut 4 opened and closed the device per call: ~40 ms). The device must have been
 * opened exactly once in this process. */
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int timing(const char *out_path) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_ELEMENTWISE_TIMING_V1\",\"gate_median_ms\":0.5,\"calls\":100,\"cases\":[");
    uint32_t seed = 0xfb1c004bu;
    enum { N = 100 };
    for (int t = 0; t < 2; t++) {
        const char *name = t == 0 ? "rmsnorm_1x2048" : "swiglu_5632";
        size_t n = t == 0 ? 2048 : 5632;
        float *a = malloc(n * 4), *b = malloc(n * 4), *got = malloc(n * 4), *want = malloc(n * 4);
        for (size_t i = 0; i < n; i++) { a[i] = frand(&seed, -3.0f, 3.0f); b[i] = t == 0 ? frand(&seed, 0.5f, 1.5f) : frand(&seed, -2.0f, 2.0f); }
        if (t == 0) oracle_rmsnorm(a, b, 1e-5f, want, n); else oracle_swiglu(a, b, want, n);
        double samples[N]; int ok_calls = 0; size_t bad_total = 0; double worst = 0; uint64_t chip_sum = 0; int rc_last = 0;
        OmegaGpuEwInfo info;
        /* warm-up: first call opens the device and generates the kernel (not timed) */
        int rc0 = t == 0 ? omega_gpu_rmsnorm_f32(1, (uint32_t)n, a, b, 1e-5f, got, &info) : omega_gpu_swiglu_f32((uint32_t)n, a, b, got, &info);
        double first_ms = info.call_ns / 1e6;
        CHECK(rc0 == OMEGA_GPU_EW_OK, "%s first call rc=%s err=\"%s\"", name, omega_gpu_elementwise_rc_name(rc0), omega_gpu_elementwise_last_error());
        for (int i = 0; i < N; i++) {
            double t0 = now_ms();
            int rc = t == 0 ? omega_gpu_rmsnorm_f32(1, (uint32_t)n, a, b, 1e-5f, got, &info) : omega_gpu_swiglu_f32((uint32_t)n, a, b, got, &info);
            samples[i] = now_ms() - t0;
            double w; size_t bad = compare(got, want, n, 1e-5, 1e-6, &w);
            if (w > worst) worst = w;
            bad_total += bad; chip_sum += info.elapsed_ns; rc_last = rc;
            if (rc == OMEGA_GPU_EW_OK && bad == 0) ok_calls++;
        }
        qsort(samples, N, sizeof samples[0], cmp_d);
        double median = samples[N / 2], p90 = samples[N * 9 / 10], mn = samples[0], mx = samples[N - 1];
        uint32_t opens = omega_gpu_session_open_count();
        int pass = ok_calls == N && bad_total == 0 && median < 0.5 && opens == 1;
        CHECK(pass, "timing %s ok_calls=%d/%d violations=%zu median_ms=%.3f opens=%u last_rc=%s err=\"%s\"", name, ok_calls, N, bad_total, median, opens, omega_gpu_elementwise_rc_name(rc_last), omega_gpu_elementwise_last_error());
        printf("%s timing %s first_call_ms=%.3f ok_calls=%d/%d violations=%zu worst_scaled_err=%g median_ms=%.3f p90_ms=%.3f min_ms=%.3f max_ms=%.3f chip_us_mean=%.1f device_opens=%u\n",
               pass ? "PASS" : "FAIL", name, first_ms, ok_calls, N, bad_total, worst, median, p90, mn, mx, chip_sum / 1e3 / N, opens);
        if (out) fprintf(out, "%s{\"case\":\"%s\",\"first_call_ms\":%.3f,\"ok_calls\":%d,\"violations\":%zu,\"worst_scaled_err\":%g,\"median_ms\":%.4f,\"p90_ms\":%.4f,\"min_ms\":%.4f,\"max_ms\":%.4f,\"chip_us_mean\":%.1f,\"device_opens\":%u,\"pass\":%s}",
                         t ? "," : "", name, first_ms, ok_calls, bad_total, worst, median, p90, mn, mx, chip_sum / 1e3 / N, opens, pass ? "true" : "false");
        free(a); free(b); free(got); free(want);
    }
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed, g_failed == 0 ? "OMEGA_GPU_ELEMENTWISE_TIMING_PASS" : "OMEGA_GPU_ELEMENTWISE_TIMING_FAIL"); fclose(out); }
    return 0;
}

int main(int argc, char **argv) {
    const char *out_path = NULL; int host = 0, timing_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--timing") == 0) timing_mode = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    }
    if (host) host_only();
    else if (timing_mode) { if (timing(out_path) != 0) return 2; }
    else if (chip(out_path) != 0) return 2;
    printf("%s: %d checks, %d failed\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
