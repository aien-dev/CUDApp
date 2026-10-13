/* gpu_qgemm_test.c (OWN-30B L1 GEMM, W-L1G, 2026-10-14): the native FP8 prefill path (omega_blackwell_codegen_qg.h).
 *   --host-only  reference self-checks
 *   --sim        kernels in the warp simulator bit-identical to the C reference; mutants caught (a simulator error is a FAIL)
 *   --chip       (under flock /tmp/aien-gb10.lock) through the device API: bit-identical to the reference, mutants caught,
 *                medians at --tokens
 * Verdict line last: "GPU_QGEMM: PASS|FAIL". */
#include "omega_blackwell_codegen_qg.h"
#include "omega_gpu_prefill.h"
#include "bw_warp_sim_tc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x243f6a8885a308d3ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }

/* ---------------- quantiser reference (header semantics) ---------------- */
static void pq_ref(const float *x, uint32_t k, uint8_t *q, float *s, int mutant) {
    for (uint32_t b = 0; b < k / 128; b++) {
        float m = ldexpf(1.0f, -100);
        if (mutant == 1) { /* lane-local amax: lane l sees elements 4l..4l+3 only (scale per lane; the reference keeps lane 31's) */
            for (uint32_t l = 0; l < 32; l++) { float ml = ldexpf(1.0f, -100); for (int j = 0; j < 4; j++) ml = fmaxf(ml, fabsf(x[128 * b + 4 * l + j])); if (l == 31) m = ml; }
        } else for (uint32_t j = 0; j < 128; j++) m = fmaxf(m, fabsf(x[128 * b + j]));
        uint32_t mb = f2u(m), E = mb >> 23, M = mb & 0x7fffff, c = mutant == 2 ? 0 : (M + 0x1fffff) >> 23;
        float scale = u2f((E + c - 8) << 23), inv = u2f((262 - E - c) << 23);
        s[b] = scale;
        for (uint32_t j = 0; j < 128; j++) q[128 * b + j] = qm_e4m3_rne_sat(x[128 * b + j] * inv);
    }
}
static float rand_act(void) {
    float v = ((float)(rnd() & 0xffffff) / 16777216.0f) * 2.0f - 1.0f;
    int r = rnd() % 32; if (r == 0) v *= 300.0f; else if (r == 1) v *= 1e-3f; else if (r == 2) v = 0.0f;
    return v;
}
static uint16_t bf16_rn(float f) { uint32_t u = f2u(f); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
/* rows x k activations; row 3 all zero, row 5 one huge outlier; bf16 copy when in_type is BF16 */
static void make_x(uint32_t rows, uint32_t k, uint32_t in_type, float *x, uint16_t *xb) {
    for (size_t i = 0; i < (size_t)rows * k; i++) x[i] = rand_act();
    if (rows > 3) memset(x + 3 * (size_t)k, 0, k * 4);
    if (rows > 5) x[5 * (size_t)k + 77] = 3.0e4f;
    if (in_type == OMEGA_QG_IN_BF16) for (size_t i = 0; i < (size_t)rows * k; i++) { xb[i] = bf16_rn(x[i]); x[i] = u2f((uint32_t)xb[i] << 16); }
}
static size_t pq_compare(const float *x, uint32_t rows, uint32_t k, const uint8_t *q, const float *s, int mutant) {
    size_t bad = 0; uint8_t *rq = malloc(k); float *rs = malloc(k / 128 * 4);
    for (uint32_t r = 0; r < rows; r++) {
        pq_ref(x + (size_t)r * k, k, rq, rs, mutant);
        for (uint32_t j = 0; j < k; j++) if (rq[j] != q[(size_t)r * k + j]) bad++;
        for (uint32_t b = 0; b < k / 128; b++) if (f2u(rs[b]) != f2u(s[(size_t)r * (k / 128) + b])) bad++;
    }
    free(rq); free(rs); return bad;
}

static int pq_host(void) {
    int fail = 0;
    /* scale choice: amax / scale in (224, 448] for normal amax */
    for (int i = 0; i < 100000; i++) {
        float x[128]; for (int j = 0; j < 128; j++) x[j] = rand_act() * ldexpf(1.0f, (int)(rnd() % 40) - 20);
        uint8_t q[128]; float s[1]; pq_ref(x, 128, q, s, 0);
        float m = 0; for (int j = 0; j < 128; j++) m = fmaxf(m, fabsf(x[j]));
        if (m > 0 && !(m / s[0] <= 448.0f && m / s[0] > 224.0f)) { if (fail < 3) printf("scale out of range: amax %g scale %g\n", m, s[0]); fail++; }
        double err = 0; for (int j = 0; j < 128; j++) { double d = fabs((double)qm_e4m3(q[j]) * s[0] - x[j]); if (d > err) err = d; }
        if (m > 0 && err > m * (1.0 / 16 + 1e-6)) { if (fail < 3) printf("quantisation error %g vs amax %g\n", err, m); fail++; }
    }
    printf("quantiser reference: %s\n", fail ? "FAIL" : "PASS (amax/scale in (224, 448], error within half an e4m3 ulp)");
    return fail;
}

static int pq_sim_one(uint32_t k, uint32_t in_type, uint32_t rows, int mutant, size_t *bad_out, char *msg, size_t msgn) {
    OmegaPqSpec sp = { k, in_type, mutant };
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_pq_prog(&sp, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    float *x = malloc((size_t)rows * k * 4); uint16_t *xb = malloc((size_t)rows * k * 2);
    uint8_t *q = malloc((size_t)rows * k); float *s = malloc((size_t)rows * (k / 128) * 4);
    make_x(rows, k, in_type, x, xb);
    memset(q, 0xee, (size_t)rows * k); memset(s, 0xee, (size_t)rows * (k / 128) * 4);
    static WsSim ws; memset(&ws, 0, sizeof ws);
    ws.p = &prog; ws.threads = 32u * (k / 128);
    const void *xin = in_type == OMEGA_QG_IN_F32 ? (const void *)x : (const void *)xb;
    uint64_t a[3] = { (uintptr_t)xin, (uintptr_t)q, (uintptr_t)s };
    for (int i = 0; i < 3; i++) { ws.cbank[(0x380 + 8 * i) / 4] = (uint32_t)a[i]; ws.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(a[i] >> 32); }
    ws.ranges[0].lo = (uintptr_t)xin; ws.ranges[0].hi = (uintptr_t)xin + (size_t)rows * k * (in_type == OMEGA_QG_IN_F32 ? 4 : 2);
    ws.ranges[1].lo = (uintptr_t)q; ws.ranges[1].hi = (uintptr_t)q + (size_t)rows * k;
    ws.ranges[2].lo = (uintptr_t)s; ws.ranges[2].hi = (uintptr_t)s + (size_t)rows * (k / 128) * 4;
    ws.nranges = 3;
    int err = ws_run(&ws, rows, 1);
    if (err) snprintf(msg, msgn, "simulator %.150s", ws.msg);
    *bad_out = err ? 0 : pq_compare(x, rows, k, q, s, 0);
    free(x); free(xb); free(q); free(s);
    return err ? -1 : 0;
}
static int pq_sim(void) {
    int fail = 0; char msg[200];
    static const struct { uint32_t k, in_type; } cs[] = { { 2048, OMEGA_QG_IN_F32 }, { 768, OMEGA_QG_IN_BF16 }, { 4096, OMEGA_QG_IN_F32 }, { 128, OMEGA_QG_IN_BF16 } };
    for (size_t c = 0; c < sizeof cs / sizeof cs[0]; c++) {
        size_t bad = 0; int rc = pq_sim_one(cs[c].k, cs[c].in_type, 9, 0, &bad, msg, sizeof msg);
        printf("sim quantiser k %u %s: %s, %zu differ\n", cs[c].k, cs[c].in_type ? "bf16" : "f32", rc ? msg : "ran", bad);
        if (rc || bad) fail++;
    }
    for (int m = 1; m <= 2; m++) {
        size_t bad = 0; int rc = pq_sim_one(2048, OMEGA_QG_IN_F32, 9, m, &bad, msg, sizeof msg);
        printf("sim quantiser mutant %d: %s\n", m, rc ? msg : bad ? "caught" : "NOT CAUGHT");
        if (rc || !bad) fail++;
    }
    return fail;
}

static int pq_chip(uint32_t tokens, int reps) {
    int fail = 0;
    static const struct { uint32_t k, in_type, rows_mul; } cs[] = { { 2048, OMEGA_QG_IN_F32, 1 }, { 768, OMEGA_QG_IN_BF16, 8 } };
    for (size_t c = 0; c < 2; c++) {
        const uint32_t k = cs[c].k, rows = tokens * cs[c].rows_mul, eb = cs[c].in_type == OMEGA_QG_IN_F32 ? 4 : 2;
        float *x = malloc((size_t)rows * k * 4); uint16_t *xb = malloc((size_t)rows * k * 2);
        uint8_t *q = malloc((size_t)rows * k); float *s = malloc((size_t)rows * (k / 128) * 4);
        make_x(rows, k, cs[c].in_type, x, xb);
        OmegaGpuMem xd = { 0 }, qd = { 0 }, sd = { 0 };
        const size_t xbytes = (size_t)rows * k * eb, qbytes = (size_t)rows * k, sbytes = ((size_t)rows * (k / 128) * 4 + 15) & ~(size_t)15;
        int rc = omega_gpu_pf_alloc_dev(xbytes, &xd);
        if (!rc) rc = omega_gpu_pf_alloc_dev(qbytes, &qd);
        if (!rc) rc = omega_gpu_pf_alloc_dev(sbytes, &sd);
        if (!rc) rc = omega_gpu_pf_upload(&xd, 0, cs[c].in_type == OMEGA_QG_IN_F32 ? (const void *)x : (const void *)xb, xbytes);
        uint64_t ns[32] = { 0 }; if (reps > 32) reps = 32;
        for (int r = 0; r < reps && !rc; r++) rc = omega_gpu_pf_quant(k, cs[c].in_type, rows, xd.va, qd.va, sd.va, &ns[r]);
        if (!rc) rc = omega_gpu_pf_download(&qd, 0, q, qbytes);
        if (!rc) { float *st = malloc(sbytes); rc = omega_gpu_pf_download(&sd, 0, st, sbytes); memcpy(s, st, (size_t)rows * (k / 128) * 4); free(st); }
        if (rc) { printf("quantiser chip k %u: %s (%s)\n", k, omega_gpu_pf_rc_name(rc), omega_gpu_session_last_error()); fail++; }
        else {
            size_t bad = pq_compare(x, rows, k, q, s, 0);
            qsort(ns, reps, 8, cmp_u64);
            printf("QUANT chip k %u %s rows %u: %zu differ from the reference; median %.3f ms (%.0f GB/s moved)\n", k, cs[c].in_type ? "bf16" : "f32", rows, bad,
                   ns[reps / 2] / 1e6, (double)(xbytes + qbytes + sbytes) / ns[reps / 2]);
            if (bad) fail++;
            omega_gpu_pf_quant_test_set_mutant(1);
            int mrc = omega_gpu_pf_quant(k, cs[c].in_type, rows, xd.va, qd.va, sd.va, NULL);
            omega_gpu_pf_quant_test_set_mutant(0);
            if (!mrc) mrc = omega_gpu_pf_download(&qd, 0, q, qbytes);
            size_t mb = mrc ? 0 : pq_compare(x, rows > 64 ? 64 : rows, k, q, s, 0);
            printf("QUANT chip k %u mutant 1: %s\n", k, mrc ? "launch failed" : mb ? "caught" : "NOT CAUGHT");
            if (mrc || !mb) fail++;
        }
        omega_gpu_pf_free(&xd); omega_gpu_pf_free(&qd); omega_gpu_pf_free(&sd);
        free(x); free(xb); free(q); free(s);
    }
    return fail;
}

int main(int argc, char **argv) {
    int host = 1, sim = 1, chip = 0, reps = 7; uint32_t tokens = 3600;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) sim = 0;
        else if (!strcmp(argv[i], "--sim")) host = 0;
        else if (!strcmp(argv[i], "--chip")) { chip = 1; host = 0; sim = 0; }
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    }
    int fail = 0;
    if (host) fail += pq_host();
    if (sim) fail += pq_sim();
    if (chip) fail += pq_chip(tokens, reps);
    printf("GPU_QGEMM: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
