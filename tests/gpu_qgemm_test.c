/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* gpu_qgemm_test.c (OWN-30B L1 GEMM, W-L1G, 2026-10-14): the native FP8 prefill path (omega_blackwell_codegen_qg.h).
 *   --host-only  reference self-checks
 *   --sim        kernels in the warp simulator bit-identical to the C reference; mutants caught (a simulator error is a FAIL)
 *   --chip       (under flock /tmp/aien-gb10.lock) through the device API: bit-identical to the reference, mutants caught,
 *                medians at --tokens
 * Verdict line last: "GPU_QGEMM: PASS|FAIL". */
#include "omega_blackwell_codegen_qg.h"
#include "omega_gpu_prefill.h"
#include "omega_gpu_fp8_api.h"
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

/* ---------------- grouped GEMM on QMMA ---------------- */
static const QmmaModel GB10Q = { 32, 26, 0, 2, 0 }; /* chip-exact, evidence OWN30B-L1G-QMMA-PFGEMM-c742aa13 */
typedef struct {
    uint32_t E, n, k, M, xr, layout, direct, pad;
    uint8_t *w, *xq;       /* weights [E][n][k] (checkpoint order, read in place), quantised activations (xr + pad rows) */
    uint16_t *sbf;         /* scale_inv bf16 */
    float *swf, *xs, *x;   /* GEMV-tensor scale table [E][n/128][KB] f32 * 2^56, activation scales (xr + pad rows), true activations */
    uint32_t *eid, *src, *counts;
} QProb;
static uint8_t rand_e4m3(void) { for (;;) { uint8_t c = (uint8_t)rnd(); if ((c & 0x7f) != 0x7f) return c; } }
static void qprob_make(QProb *p, uint32_t E, uint32_t n, uint32_t k, uint32_t M, uint32_t xr, uint32_t layout, uint32_t direct, uint32_t pad) {
    memset(p, 0, sizeof *p);
    p->E = E; p->n = n; p->k = k; p->M = M; p->xr = direct ? M : xr; p->layout = layout; p->direct = direct; p->pad = pad;
    const uint32_t KB = k / 128, XR = p->xr + pad;
    const size_t wn = (size_t)E * n * k, sn = (size_t)E * (n / 128) * KB;
    p->w = malloc(wn); p->sbf = malloc(sn * 2); p->swf = malloc(sn * 4);
    p->x = calloc((size_t)XR * k, 4); p->xq = calloc((size_t)XR * k, 1); p->xs = malloc((size_t)XR * KB * 4);
    p->eid = malloc(M * 4); p->src = malloc(M * 4); p->counts = calloc(E, 4);
    for (size_t i = 0; i < wn; i++) p->w[i] = rand_e4m3();
    for (size_t i = 0; i < sn; i++) { float s = ldexpf(1.0f + (float)(rnd() & 127) / 128.0f, -(int)(rnd() % 12) - 4); p->sbf[i] = (uint16_t)(f2u(s) >> 16); }
    if (omega_fp8_pack_scales(p->sbf, sn, p->swf, 0) != 0) { printf("scale pack refused\n"); exit(2); }
    for (size_t i = 0; i < (size_t)p->xr * k; i++) p->x[i] = rand_act();
    if (p->xr > 3) memset(p->x + 3 * (size_t)k, 0, k * 4);
    for (uint32_t r = 0; r < XR; r++) pq_ref(p->x + (size_t)r * k, k, p->xq + (size_t)r * k, p->xs + (size_t)r * KB, 0);
    uint32_t i = 0;
    for (uint32_t e = 0; e < E && i < M; e++) { uint32_t c = e + 1 == E ? M - i : (uint32_t)((uint64_t)(rnd() % 2000) * 2 * M / (2000ull * E)); if (c > M - i) c = M - i; for (uint32_t q = 0; q < c; q++) p->eid[i + q] = e; p->counts[e] = c; i += c; }
    for (uint32_t q = 0; q < M; q++) p->src[q] = direct ? q : rnd() % p->xr;
}
static void qprob_free(QProb *p) { free(p->w); free(p->xq); free(p->sbf); free(p->swf); free(p->xs); free(p->x); free(p->eid); free(p->src); free(p->counts); }

/* the kernel's arithmetic for output row i, physical column c */
static float qemu_phys(const QProb *p, uint32_t i, uint32_t c) {
    const uint32_t k = p->k, KB = k / 128, e = p->eid[i], s = p->src[i];
    /* physical column c -> weight row: F32 c; SWIGLU gate 8G + h % 8 (h < 8) or up I + 8G + h % 8, G = c / 16, h = c % 16 */
    const uint32_t wrow = p->layout == OMEGA_GPU_PF_PLAIN ? c : 8 * (c / 16) + c % 8 + (c % 16 >= 8 ? p->n / 2 : 0);
    const uint8_t *xr = p->xq + (size_t)s * k, *wr = p->w + ((size_t)e * p->n + wrow) * k;
    const float *swt = p->swf + ((size_t)e * (p->n / 128) + wrow / 128) * KB;
    float acc = 0.0f;
    for (uint32_t kb = 0; kb < KB; kb++) {
        const float sc = p->xs[(size_t)s * KB + kb] * (swt[kb] * 0x1p-56f);
        for (uint32_t h = 0; h < 2; h++) {
            float T = 0.0f;
            for (uint32_t v = 0; v < 2; v++) {
                float av[32], bv[32];
                for (uint32_t t = 0; t < 4; t++) for (uint32_t j = 0; j < 4; j++) {
                    const uint32_t o = 128 * kb + 64 * h + 16 * t + 8 * v + j;
                    av[4 * t + j] = qm_e4m3(xr[o]); bv[4 * t + j] = qm_e4m3(wr[o]);
                    av[16 + 4 * t + j] = qm_e4m3(xr[o + 4]); bv[16 + 4 * t + j] = qm_e4m3(wr[o + 4]);
                }
                T = qm_dot32(&GB10Q, av, bv, T);
            }
            acc = fmaf(T, sc, acc);
        }
    }
    return acc;
}
static uint16_t qswiglu_bits(float g, float u) {
    float t = fminf(g * u2f(0xbfb8aa3bu), 125.0f), ex = exp2f(t), d = ex + 1.0f, r = 1.0f / d, o = g * r;
    o = o * u;
    return bf16_rn(o);
}
static void qemu_row(const QProb *p, uint32_t i, float *yf, uint16_t *ya) {
    if (p->layout == OMEGA_GPU_PF_PLAIN) { for (uint32_t c = 0; c < p->n; c++) yf[c] = qemu_phys(p, i, c); return; }
    for (uint32_t a = 0; a < p->n / 2; a++) { uint32_t G = a / 8, m = a % 8; ya[a] = qswiglu_bits(qemu_phys(p, i, 16 * G + m), qemu_phys(p, i, 16 * G + 8 + m)); }
}
/* double oracle on the ORIGINAL weights and the dequantised activations: logical output col (PLAIN) or gate/up of index a */
static double qoracle(const QProb *p, uint32_t i, uint32_t col, double *mag) {
    const uint32_t k = p->k, KB = k / 128, e = p->eid[i], s = p->src[i];
    double acc = 0, m = 0;
    for (uint32_t j = 0; j < k; j++) {
        const double sw = u2f((uint32_t)p->sbf[((size_t)e * (p->n / 128) + col / 128) * KB + j / 128] << 16);
        const double t = (double)qm_e4m3(p->w[((size_t)e * p->n + col) * k + j]) * sw * qm_e4m3(p->xq[(size_t)s * k + j]) * p->xs[(size_t)s * KB + j / 128];
        acc += t; m += fabs(t);
    }
    *mag = m; return acc;
}

typedef struct { uint64_t w_va, s_va; uint32_t row0, cnt; uint64_t pad; } QTile;
static uint32_t qmake_tiles(const QProb *p, uint32_t bm, QTile *t) {
    uint32_t nt = 0, row = 0; const uint32_t KB = p->k / 128;
    for (uint32_t e = 0; e < p->E; e++) { for (uint32_t r = 0; r < p->counts[e]; r += bm) {
        t[nt].w_va = (uintptr_t)(p->w + (size_t)e * p->n * p->k); t[nt].s_va = (uintptr_t)(p->swf + (size_t)e * (p->n / 128) * KB);
        t[nt].row0 = row + r; t[nt].cnt = p->counts[e] - r < bm ? p->counts[e] - r : bm; t[nt].pad = 0; nt++; } row += p->counts[e]; }
    return nt;
}
static size_t qcompare(const QProb *p, const void *y, const uint32_t *rows, uint32_t nrows, int tol_ulp, size_t *ulp1) {
    size_t bad = 0; if (ulp1) *ulp1 = 0;
    float *yf = malloc(p->n * 4); uint16_t *ya = malloc(p->n);
    for (uint32_t q = 0; q < nrows; q++) {
        const uint32_t i = rows[q]; qemu_row(p, i, yf, ya);
        if (p->layout == OMEGA_GPU_PF_PLAIN) { const uint32_t *g = (const uint32_t *)y + (size_t)i * p->n; for (uint32_t c = 0; c < p->n; c++) if (g[c] != f2u(yf[c])) { if (getenv("PFDBG") && bad < 6) printf("  row %u col %u got %.9g want %.9g\n", i, c, u2f(g[c]), yf[c]); bad++; } }
        else { const uint16_t *g = (const uint16_t *)y + (size_t)i * (p->n / 2);
            for (uint32_t c = 0; c < p->n / 2; c++) if (g[c] != ya[c]) { int d = abs((int)g[c] - (int)ya[c]); if (d <= tol_ulp && ((g[c] ^ ya[c]) & 0x8000) == 0) { if (ulp1) (*ulp1)++; } else { if (getenv("PFDBG") && bad < 12) { const uint32_t G = c / 8, mm = c % 8; printf("  row %u act %u got %04x want %04x (g %.9g u %.9g)\n", i, c, g[c], ya[c], qemu_phys(p, i, 16 * G + mm), qemu_phys(p, i, 16 * G + 8 + mm)); } bad++; } } }
    }
    free(yf); free(ya);
    return bad;
}

static int qg_host(void) {
    int fail = 0;
    for (uint32_t layout = 0; layout < 3; layout += 2) {
        QProb p; qprob_make(&p, 3, 256, 384, 40, 17, layout, 0, 0);
        double worst = 0;
        for (uint32_t i = 0; i < p.M; i += 3) {
            if (layout == OMEGA_GPU_PF_PLAIN) for (uint32_t c = 0; c < p.n; c++) { double m, o = qoracle(&p, i, c, &m); double d = fabs(qemu_phys(&p, i, c) - o) / (m + 1e-30); if (d > worst) worst = d; }
            else for (uint32_t a = 0; a < p.n / 2; a++) { uint32_t G = a / 8, mm = a % 8; double m1, m2;
                double og = qoracle(&p, i, a, &m1), ou = qoracle(&p, i, p.n / 2 + a, &m2);
                double dg = fabs(qemu_phys(&p, i, 16 * G + mm) - og) / (m1 + 1e-30), du = fabs(qemu_phys(&p, i, 16 * G + 8 + mm) - ou) / (m2 + 1e-30);
                if (dg > worst) worst = dg;
                if (du > worst) worst = du; }
        }
        const int ok = worst < 1e-6;
        printf("qgemm emulation vs double oracle (%s): worst error %.3g of the sum of |terms| %s\n", layout ? "swiglu_nat" : "plain", worst, ok ? "PASS" : "FAIL");
        if (!ok) fail++;
        qprob_free(&p);
    }
    return fail;
}

static int qg_sim_run(const QProb *p, const OmegaQgSpec *sp, int order, void *y, char *msg, size_t msgn) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_qgemm_prog(sp, &prog) != 0) { snprintf(msg, msgn, "codegen refused (spec_ok %d, insns %zu)", omega_qg_spec_ok(sp), prog.count); return -1; }
    const uint32_t bm = 16u * sp->rt, span = omega_qg_span(sp), KB = p->k / 128;
    QTile *tiles = calloc(p->M + 1, sizeof *tiles);
    const uint32_t nt = qmake_tiles(p, bm, tiles);
    uint32_t *ridx = malloc((p->M + bm) * 4); for (uint32_t i = 0; i < p->M + bm; i++) ridx[i] = i < p->M ? p->src[i] : 0;
    const size_t ob = p->layout == OMEGA_GPU_PF_PLAIN ? 4 : 1, yb = (size_t)(p->M + bm) * p->n * ob;
    memset(y, 0xee, yb);
    uint8_t *smem = malloc(omega_qg_shared_bytes(sp));
    static WsSim s; memset(&s, 0, sizeof s);
    s.p = &prog; s.threads = 32u * sp->warps; s.smem = smem; s.smem_bytes = omega_qg_shared_bytes(sp); s.order = order;
    const uint64_t a[5] = { (uintptr_t)tiles, sp->direct ? 0 : (uintptr_t)ridx, (uintptr_t)p->xq, (uintptr_t)y, (uintptr_t)p->xs };
    for (int i = 0; i < 5; i++) { s.cbank[(0x380 + 8 * i) / 4] = (uint32_t)a[i]; s.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(a[i] >> 32); }
    s.cbank[0x3a8 / 4] = p->n;
    const size_t XR = p->xr + p->pad;
    s.ranges[0].lo = (uintptr_t)tiles; s.ranges[0].hi = (uintptr_t)(tiles + nt);
    s.ranges[1].lo = (uintptr_t)ridx; s.ranges[1].hi = (uintptr_t)(ridx + p->M + bm);
    s.ranges[2].lo = (uintptr_t)p->xq; s.ranges[2].hi = (uintptr_t)(p->xq + XR * p->k);
    s.ranges[3].lo = (uintptr_t)y; s.ranges[3].hi = (uintptr_t)y + (size_t)p->M * p->n * ob;
    s.ranges[4].lo = (uintptr_t)p->w; s.ranges[4].hi = (uintptr_t)(p->w + (size_t)p->E * p->n * p->k);
    s.ranges[5].lo = (uintptr_t)p->swf; s.ranges[5].hi = (uintptr_t)(p->swf + (size_t)p->E * (p->n / 128) * KB);
    s.ranges[6].lo = (uintptr_t)p->xs; s.ranges[6].hi = (uintptr_t)(p->xs + XR * KB);
    s.nranges = 7;
    const int err = ws_run(&s, p->n / span, nt);
    if (err) snprintf(msg, msgn, "simulator %.150s", s.msg);
    free(tiles); free(ridx); free(smem);
    return err ? -1 : 0;
}
static int qg_sim(void) {
    int fail = 0; char msg[200];
    static const struct { uint32_t E, n, k, M, xr, layout, direct, rt, warps; } cs[] = {
        { 3, 256, 384, 300, 120, OMEGA_GPU_PF_SWIGLU_NAT, 0, 4, 8 },
        { 3, 256, 256, 200, 0, OMEGA_GPU_PF_PLAIN, 1, 4, 8 },
        { 2, 256, 128, 150, 64, OMEGA_GPU_PF_PLAIN, 0, 4, 8 },
        { 2, 512, 1024, 140, 0, OMEGA_GPU_PF_SWIGLU_NAT, 1, 4, 8 },
    };
    for (size_t c = 0; c < sizeof cs / sizeof cs[0]; c++) {
        const uint32_t bm = 16 * cs[c].rt;
        QProb p; qprob_make(&p, cs[c].E, cs[c].n, cs[c].k, cs[c].M, cs[c].xr, cs[c].layout, cs[c].direct, bm);
        OmegaQgSpec sp = { cs[c].k, cs[c].rt, cs[c].warps, cs[c].layout == OMEGA_GPU_PF_PLAIN ? OMEGA_QG_EPI_F32 : OMEGA_QG_EPI_SWIGLU, cs[c].direct, cs[c].n, 45, 0 };
        uint32_t *rows = malloc(p.M * 4); for (uint32_t i = 0; i < p.M; i++) rows[i] = i;
        void *y = malloc((size_t)(p.M + bm) * p.n * 4);
        for (int order = 0; order < 2; order++) {
            const int rc = qg_sim_run(&p, &sp, order, y, msg, sizeof msg);
            size_t ulp1 = 0, bad = rc ? 0 : qcompare(&p, y, rows, p.M, 0, &ulp1);
            printf("sim qgemm case %zu (n %u k %u M %u %s %s rt %u warps %u) order %d: %s, %zu differ\n", c, p.n, p.k, p.M, cs[c].layout ? "swiglu" : "f32",
                   cs[c].direct ? "direct" : "indexed", cs[c].rt, cs[c].warps, order, rc ? msg : "ran", bad);
            if (rc || bad) fail++;
        }
        if (c == 0 || c == 1) for (int m = 1; m <= 4; m++) {
            OmegaQgSpec ms = sp; ms.mutant = m;
            const int rc = qg_sim_run(&p, &ms, 0, y, msg, sizeof msg);
            const size_t bad = rc ? 0 : qcompare(&p, y, rows, p.M, 0, NULL);
            printf("sim qgemm case %zu mutant %d: %s\n", c, m, rc ? msg : bad ? "caught" : "NOT CAUGHT");
            if (rc || !bad) fail++;
        }
        free(rows); free(y); qprob_free(&p);
    }
    return fail;
}

static int g_cached; /* 1: weights in a GPU-cached device copy (GEMV layout), 0: read in place from the GEMV tensor */
static int qg_chip_shape(const char *name, uint32_t tokens, int reps, int check_rows, int mutants) {
    uint32_t E, n, k, M, xr, layout, direct;
    if (!strcmp(name, "gateup")) { E = 128; n = 1536; k = 2048; M = tokens * 8; xr = tokens; layout = OMEGA_GPU_PF_SWIGLU_NAT; direct = 0; }
    else if (!strcmp(name, "down")) { E = 128; n = 2048; k = 768; M = tokens * 8; xr = 0; layout = OMEGA_GPU_PF_PLAIN; direct = 1; }
    else if (!strcmp(name, "qkv")) { E = 1; n = 5120; k = 2048; M = tokens; xr = tokens; layout = OMEGA_GPU_PF_PLAIN; direct = 0; }
    else if (!strcmp(name, "o")) { E = 1; n = 2048; k = 4096; M = tokens; xr = tokens; layout = OMEGA_GPU_PF_PLAIN; direct = 0; }
    else { printf("unknown shape %s\n", name); return 1; }
    const uint32_t bm = omega_gpu_pf_qgemm_bm(), KB = k / 128;
    QProb p; qprob_make(&p, E, n, k, M, xr, layout, direct, bm);
    int fail = 0, rc;
    OmegaGpuFp8Tensor *wt = NULL; OmegaGpuMem xd = { 0 }, sd = { 0 }, yd = { 0 }, tab = { 0 }, ridx = { 0 }, wc = { 0 }, sc = { 0 };
    const size_t XR = p.xr + bm, ob = layout == OMEGA_GPU_PF_PLAIN ? 4 : 1;
    rc = omega_gpu_fp8_tensor_upload(E, n, k, p.w, p.sbf, &wt) == 0 ? 0 : OMEGA_GPU_PF_CHIP_FAIL; /* the GEMV tensor, read in place */
    const size_t xb = (XR * k + 15) & ~(size_t)15, sb = (XR * KB * 4 + 15) & ~(size_t)15, yb = (size_t)(M + bm) * n * ob;
    if (!rc) rc = omega_gpu_pf_alloc_dev(xb, &xd);
    if (!rc) rc = omega_gpu_pf_alloc_dev(sb, &sd);
    if (!rc) rc = omega_gpu_pf_alloc_dev(yb, &yd);
    if (!rc) rc = omega_gpu_pf_upload(&xd, 0, p.xq, XR * k & ~(size_t)15);
    if (!rc) rc = omega_gpu_pf_upload(&sd, 0, p.xs, XR * KB * 4 & ~(size_t)15);
    const uint32_t maxt = M / bm + E + 1;
    if (!rc) rc = omega_gpu_pf_alloc_host((size_t)maxt * 32, &tab);
    if (!rc) rc = omega_gpu_pf_alloc_host((size_t)(M + bm) * 4, &ridx);
    if (rc) { printf("%s setup: %s (%s)\n", name, omega_gpu_pf_rc_name(rc), omega_gpu_session_last_error()); qprob_free(&p); return 1; }
    uint32_t nt = 0;
    if (!g_cached) nt = omega_gpu_pf_qgemm_tiles(wt, p.counts, tab.cpu, maxt);
    else {
        const size_t wbytes = (size_t)E * n * k, sbytes = ((size_t)E * (n / 128) * KB * 4 + 15) & ~(size_t)15;
        float *sst = calloc(1, sbytes); memcpy(sst, p.swf, (size_t)E * (n / 128) * KB * 4);
        if (omega_gpu_pf_alloc_dev(wbytes, &wc) || omega_gpu_pf_alloc_dev(sbytes, &sc) || omega_gpu_pf_upload(&wc, 0, p.w, wbytes) || omega_gpu_pf_upload(&sc, 0, sst, sbytes)) { printf("%s cached copy failed\n", name); free(sst); return 1; }
        free(sst);
        uint64_t *wv = malloc(E * 8), *sv = malloc(E * 8);
        for (uint32_t e = 0; e < E; e++) { wv[e] = wc.va + (uint64_t)e * n * k; sv[e] = sc.va + (uint64_t)e * (n / 128) * KB * 4; }
        nt = omega_gpu_pf_qgemm_tiles_va(E, wv, sv, p.counts, tab.cpu, maxt); free(wv); free(sv);
    }
    if (!nt) { printf("%s tile table failed\n", name); return 1; }
    { uint32_t *ri = ridx.cpu; for (uint32_t i = 0; i < M + bm; i++) ri[i] = i < M ? p.src[i] : 0; }
    omega_gpu_session_lock(); (void)omega_gpu_session_sync_to_gpu(&tab); (void)omega_gpu_session_sync_to_gpu(&ridx); omega_gpu_session_unlock();
    const uint64_t rva = direct ? 0 : ridx.va;
    uint8_t *y1 = malloc(yb), *y2 = calloc(1, yb);
    uint64_t ns[32] = { 0 };
    if (reps > 32) reps = 32;
    for (int r = 0; r < reps && !rc; r++) rc = omega_gpu_pf_qgemm(n, k, layout, tab.va, nt, rva, xd.va, sd.va, yd.va, &ns[r]);
    if (!rc) rc = omega_gpu_pf_download(&yd, 0, y1, yb & ~(size_t)15);
    if (!rc) rc = omega_gpu_pf_upload(&yd, 0, y2, yb & ~(size_t)15); /* zero the output, run once more: identical result required */
    uint64_t ns2 = 0;
    if (!rc) rc = omega_gpu_pf_qgemm(n, k, layout, tab.va, nt, rva, xd.va, sd.va, yd.va, &ns2);
    if (!rc) rc = omega_gpu_pf_download(&yd, 0, y2, yb & ~(size_t)15);
    if (rc) { printf("%s run: %s (%s)\n", name, omega_gpu_pf_rc_name(rc), omega_gpu_session_last_error()); fail++; }
    else {
        const size_t outb = (size_t)M * n * ob;
        printf("%s second launch identical: %s\n", name, memcmp(y1, y2, outb) ? "NO" : "yes");
        if (memcmp(y1, y2, outb)) fail++;
        uint32_t *rows = malloc(check_rows * 4); for (int q = 0; q < check_rows; q++) rows[q] = q == 0 ? 0 : q == 1 ? M - 1 : rnd() % M;
        size_t ulp1 = 0; const size_t bad = qcompare(&p, y1, rows, check_rows, layout == OMEGA_GPU_PF_PLAIN ? 0 : 1, &ulp1);
        printf("%s bit-identity on %d sampled rows: %zu differ", name, check_rows, bad);
        if (layout != OMEGA_GPU_PF_PLAIN) printf(" (beyond 1 bf16 ulp; %zu within 1 ulp of %zu)", ulp1, (size_t)check_rows * n / 2);
        printf("\n");
        if (bad) fail++;
        qsort(ns, reps, 8, cmp_u64);
        const double fl = 2.0 * M * n * k, med = (double)ns[reps / 2];
        printf("QGEMM%s %s tokens %u M %u n %u k %u tiles %u: median %.3f ms (min %.3f), %.1f TFLOPS, weights %.0f MB -> %.0f GB/s of distinct weights\n", g_cached ? "-cached" : "-inplace", name, tokens, M, n, k, nt,
               med / 1e6, ns[0] / 1e6, fl / med / 1e3, (double)E * n * k / 1e6, (double)E * n * k / med);
        if (mutants) for (int m = 1; m <= 4; m++) {
            omega_gpu_pf_qgemm_test_set_mutant(m);
            int mrc = omega_gpu_pf_qgemm(n, k, layout, tab.va, nt, rva, xd.va, sd.va, yd.va, NULL);
            omega_gpu_pf_qgemm_test_set_mutant(0);
            if (!mrc) mrc = omega_gpu_pf_download(&yd, 0, y2, yb & ~(size_t)15);
            const size_t mb = mrc ? 0 : qcompare(&p, y2, rows, check_rows > 8 ? 8 : check_rows, layout == OMEGA_GPU_PF_PLAIN ? 0 : 1, NULL);
            printf("%s mutant %d: %s\n", name, m, mrc ? "launch failed" : mb ? "caught" : "NOT CAUGHT");
            if (mrc || !mb) fail++;
        }
        free(rows);
    }
    free(y1); free(y2);
    omega_gpu_fp8_tensor_free(wt); omega_gpu_pf_free(&wc); omega_gpu_pf_free(&sc); omega_gpu_pf_free(&xd); omega_gpu_pf_free(&sd); omega_gpu_pf_free(&yd); omega_gpu_pf_free(&tab); omega_gpu_pf_free(&ridx);
    qprob_free(&p);
    return fail;
}

int main(int argc, char **argv) {
    int host = 1, sim = 1, chip = 0, reps = 7, rows = 64; uint32_t tokens = 3600, rt = 0, warps = 0, wait = 0, waitset = 0; const char *only = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) sim = 0;
        else if (!strcmp(argv[i], "--sim")) host = 0;
        else if (!strcmp(argv[i], "--chip")) { chip = 1; host = 0; sim = 0; }
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shape") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--cached")) g_cached = 1;
        else if (!strcmp(argv[i], "--rt") && i + 1 < argc) rt = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warps") && i + 1 < argc) warps = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--wait") && i + 1 < argc) { wait = (uint32_t)atoi(argv[++i]); waitset = 1; }
    }
    int fail = 0;
    if (host) { fail += pq_host(); fail += qg_host(); }
    if (sim) { fail += pq_sim(); fail += qg_sim(); }
    if (chip) {
        if (rt || warps || waitset) omega_gpu_pf_qgemm_set_shape(rt ? rt : 4, warps ? warps : 8, waitset ? wait : 45);
        if (!only) fail += pq_chip(tokens, reps);
        static const char *shapes[] = { "gateup", "down", "qkv", "o" };
        for (int s = 0; s < 4; s++) if (!only || !strcmp(only, shapes[s])) fail += qg_chip_shape(shapes[s], tokens, reps, rows, !only || s < 2);
    }
    printf("GPU_QGEMM: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
