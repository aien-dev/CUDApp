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

/*
 * gpu_prefill_attn_test: the tensor-core causal prefill attention (omega_gpu_prefill_attn_api.h, OWN-30B L1 attention).
 *   --host-only   codegen budgets (instructions <= 1024, declared registers <= 128, placeholders patched against the
 *                 oracle words), refusals, reference sanity. No device.
 *   --sim         the kernel through tests/bw_warp_sim_pa.h (chip-exact HMMA model; EX2/RSQ/RCP exact): several
 *                 (pos0, m) points incl. two chunks, warps 4 and 8, both fragment orders. Gates: appended K/V rows
 *                 bit-equal to a host emulation of the decode kernel's operation order; outputs within the tolerance
 *                 of a double reference over the cache contents; mutants 1-4 caught.
 *   (default)     the chip: parity against the double reference; appended K/V rows bit-identical to m calls of the
 *                 decode kernel on a twin cache (gate 2) and its outputs within the tolerance; second call identical;
 *                 out_scale 2^64; mutants 1-4 caught; timing medians (prefill vs the per-token decode loop).
 * Tolerance: |out - ref| <= 1e-2 * max|v| * out_scale per element. Derivation: q' and P are rounded to bf16 (relative
 * 2^-9 each). P rounding moves each weight by <= 2^-9 relative while l sums the f32 p: <= 2^-9 * max|v| = 0.0020.
 * q' rounding moves a score (log2 units) by <= 2^-9 * sum|q'_i k_i|; for the normed rows here (|q'|, |k| ~ sqrt(128)
 * with random signs) that is ~2^-9 * sqrt(128) * log2(e)/sqrt(128) * ... ~ 3e-3 relative on the weights in the worst
 * measured case; EX2/RSQ on the chip add ~2^-22. 1e-2 covers the sum with margin; the measured worst ratio is printed
 * and the mutants miss by far more than the tolerance.
 * Host and simulator passes are never chip results.
 */
#include "omega_gpu_prefill_attn_api.h"
#include "omega_gpu_session.h"
#include "omega_blackwell_codegen_tc.h"
#include "bw_warp_sim_pa.h"
#if defined(__has_include)
#if __has_include("omega_gpu_prefill.h")
#include "omega_gpu_prefill.h"   /* W-L1G device-resident module (integration builds): GPU-cached buffers for the dev check */
#define PATTN_HAVE_PF 1
#endif
#endif
#ifndef PATTN_HAVE_PF
#define PATTN_HAVE_PF 0
#endif
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
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static const double THETA = 10000000.0;
static const float EPS = 1e-6f;
#define QKVW OMEGA_PATTN_QKV_WORDS
#define OUTW OMEGA_PATTN_OUT_WORDS

/* one layer's random state */
typedef struct {
    uint32_t max_pos, pos0, m;
    float *qkv;            /* [m][5120] */
    float qn[128], kn[128];
    uint16_t *kc, *vc;     /* [4][max_pos][128]: history 0..pos0-1 */
} Case;
static void case_make(Case *c, uint32_t max_pos, uint32_t pos0, uint32_t m, uint32_t seed, float qscale) {
    uint32_t s = seed;
    c->max_pos = max_pos; c->pos0 = pos0; c->m = m;
    c->qkv = malloc((size_t)m * QKVW * 4);
    for (size_t t = 0; t < m; t++) for (size_t i = 0; i < QKVW; i++) c->qkv[t * QKVW + i] = frand(&s, -1.0f, 1.0f) * (i < 4096 ? qscale : 1.0f);
    for (unsigned i = 0; i < 128; i++) { c->qn[i] = frand(&s, 0.8f, 1.2f); c->kn[i] = frand(&s, 0.8f, 1.2f); }
    size_t words = (size_t)4 * max_pos * 128;
    c->kc = calloc(words, 2); c->vc = calloc(words, 2);
    for (unsigned g = 0; g < 4; g++) for (uint32_t t = 0; t < pos0; t++) for (unsigned i = 0; i < 128; i++) {
        size_t o = ((size_t)g * max_pos + t) * 128 + i;
        c->kc[o] = omega_dattn_bf16_rne(frand(&s, -1.0f, 1.0f)); c->vc[o] = omega_dattn_bf16_rne(frand(&s, -1.0f, 1.0f));
    }
}
static void case_free(Case *c) { free(c->qkv); free(c->kc); free(c->vc); }

/* ---- host emulation of the decode kernel's K/V row (float, the kernel's operation order; RSQ exact as in the simulator) */
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static void emu_rows(const float *qkv_row, const float *kn, float eps, uint32_t pos, double theta, uint16_t *krow /*[4][128]*/, uint16_t *vrow) {
    float rope[128]; omega_dattn_rope_row(pos, theta, rope);
    for (unsigned gk = 0; gk < 4; gk++) {
        float x[128], sq[32];
        for (unsigned i = 0; i < 128; i++) x[i] = qkv_row[4096 + gk * 128 + i];
        for (unsigned L = 0; L < 32; L++) { float a = x[4 * L] * x[4 * L]; for (unsigned i = 1; i < 4; i++) a = fmaf(x[4 * L + i], x[4 * L + i], a); sq[L] = a; }
        for (unsigned o = 16; o >= 1; o /= 2) { float nx[32]; for (unsigned L = 0; L < 32; L++) nx[L] = sq[L] + (L + o < 32 ? sq[L + o] : sq[L]); memcpy(sq, nx, sizeof sq); }
        float rs = 1.0f / sqrtf(fmaf(sq[0], u2f(0x3c000000u), eps));
        for (unsigned i = 0; i < 128; i++) { x[i] = x[i] * rs; x[i] = x[i] * kn[i]; }
        float y[128];
        for (unsigned i = 0; i < 128; i++) {
            unsigned j = i % 64; float cs = rope[j], sn = rope[64 + j]; float part = x[i < 64 ? i + 64 : i - 64];
            float ts = x[i] * cs; y[i] = fmaf(part, i < 64 ? -sn : sn, ts);
        }
        for (unsigned i = 0; i < 128; i++) { krow[gk * 128 + i] = omega_dattn_bf16_rne(y[i]); vrow[gk * 128 + i] = omega_dattn_bf16_rne(qkv_row[4096 + 512 + gk * 128 + i]); }
    }
}

/* ---- double reference over the cache contents (kc/vc hold positions 0 .. pos0+m-1 after the call) */
static void norm_rope_d(double *x, const float *w, float eps, uint32_t pos) {
    double ss = 0; for (unsigned i = 0; i < 128; i++) ss += x[i] * x[i];
    double r = 1.0 / sqrt(ss / 128.0 + (double)eps);
    for (unsigned i = 0; i < 128; i++) x[i] = x[i] * r * (double)w[i];
    double y[128];
    for (unsigned i = 0; i < 64; i++) {
        double ang = (double)pos * pow(THETA, -(double)(2 * i) / 128.0), cs = cos(ang), sn = sin(ang);
        y[i] = x[i] * cs - x[i + 64] * sn; y[i + 64] = x[i + 64] * cs + x[i] * sn;
    }
    memcpy(x, y, sizeof y);
}
static void reference(const Case *c, const uint16_t *kc, const uint16_t *vc, float out_scale, double *out /*[m][4096]*/) {
    double *p = malloc(((size_t)c->pos0 + c->m) * sizeof(double));
    for (uint32_t t = 0; t < c->m; t++) {
        const uint32_t pos = c->pos0 + t;
        for (unsigned h = 0; h < 32; h++) {
            unsigned gk = h / 8; double q[128];
            for (unsigned i = 0; i < 128; i++) q[i] = c->qkv[(size_t)t * QKVW + h * 128 + i];
            norm_rope_d(q, c->qn, EPS, pos);
            double mx = -1e300;
            for (uint32_t k = 0; k <= pos; k++) {
                const uint16_t *kr = kc + ((size_t)gk * c->max_pos + k) * 128;
                double s = 0; for (unsigned i = 0; i < 128; i++) s += q[i] * (double)omega_dattn_bf16_f32(kr[i]);
                s /= sqrt(128.0); p[k] = s; if (s > mx) mx = s;
            }
            double l = 0, o[128]; memset(o, 0, sizeof o);
            for (uint32_t k = 0; k <= pos; k++) { p[k] = exp(p[k] - mx); l += p[k]; }
            for (uint32_t k = 0; k <= pos; k++) { const uint16_t *vr = vc + ((size_t)gk * c->max_pos + k) * 128; for (unsigned i = 0; i < 128; i++) o[i] += p[k] * (double)omega_dattn_bf16_f32(vr[i]); }
            for (unsigned i = 0; i < 128; i++) out[(size_t)t * OUTW + h * 128 + i] = o[i] / l * (double)out_scale;
        }
    }
    free(p);
}
static size_t parity(const Case *c, const uint16_t *kc, const uint16_t *vc, const float *out, float out_scale, double *worst) {
    double *ref = malloc((size_t)c->m * OUTW * sizeof(double));
    reference(c, kc, vc, out_scale, ref);
    double tol = 1e-2 * out_scale; size_t bad = 0; *worst = 0;
    for (size_t i = 0; i < (size_t)c->m * OUTW; i++) { double d = fabs((double)out[i] - ref[i]); if (!(d <= tol)) bad++; if (d / tol > *worst || d != d) *worst = d != d ? 1e9 : d / tol; }
    free(ref);
    return bad;
}
/* appended rows (cache contents at pos0 .. pos0+m-1) against the host emulation: count of differing bf16 words and the
 * largest difference in ulps */
static void rows_vs_emu(const Case *c, const uint16_t *kc, const uint16_t *vc, size_t *kdiff, size_t *vdiff, int *maxulp) {
    *kdiff = *vdiff = 0; *maxulp = 0;
    uint16_t kr[512], vr[512];
    for (uint32_t t = 0; t < c->m; t++) {
        emu_rows(c->qkv + (size_t)t * QKVW, c->kn, EPS, c->pos0 + t, THETA, kr, vr);
        for (unsigned gk = 0; gk < 4; gk++) for (unsigned i = 0; i < 128; i++) {
            size_t o = ((size_t)gk * c->max_pos + c->pos0 + t) * 128 + i;
            if (kc[o] != kr[gk * 128 + i]) { (*kdiff)++; int d = abs((int)kc[o] - (int)kr[gk * 128 + i]); if (d > *maxulp) *maxulp = d; }
            if (vc[o] != vr[gk * 128 + i]) (*vdiff)++;
        }
    }
}

/* ------------------------------------------------------------- host only */
static void host_only(void) {
    char msg[200];
    CHECK(omega_tc_patch_selftest(msg, sizeof msg) == 0, "placeholder self-test: %s", msg);
    for (uint32_t w = 4; w <= 8; w += 4) for (uint32_t md = 0; md <= 2; md++) for (int m = 0; m <= 4; m++) {
        OmegaPrefillAttnSpec s = { w, m, md, 0 }; uint32_t n = 0, gp = 0, ph = 0;
        int rc = omega_prefill_attn_prog_stats(&s, &n, &gp, &ph);
        CHECK(rc == 0 && n <= 1024 && omega_bw_gpr_declared(gp) <= 128, "codegen warps %u mode %u mutant %d: rc %d insns %u gprs %u", w, md, m, rc, n, gp);
        static uint8_t code[1024 * 16]; uint32_t kg = 0; int np = -1;
        size_t sz = omega_gpu_prefill_attn_encode(&s, code, sizeof code, &kg, &np);
        CHECK(sz > 0 && np == (int)ph, "encode+patch warps %u mode %u mutant %d: size %zu patched %d of %u", w, md, m, sz, np, ph);
        if (rc == 0 && m == 0) printf("codegen warps %u mode %u: %u instructions, %u registers (declared %u, %u threads -> %u of 65536), %u placeholders patched\n",
                                      w, md, n, gp, omega_bw_gpr_declared(gp), 32 * w, 32 * w * omega_bw_gpr_declared(gp), ph);
    }
    { OmegaPrefillAttnSpec s = { 2, 0, 0, 0 }; CHECK(omega_prefill_attn_prog_stats(&s, NULL, NULL, NULL) != 0, "warps 2 refused");
      s.warps = 8; s.mutant = 5; CHECK(omega_prefill_attn_prog_stats(&s, NULL, NULL, NULL) != 0, "mutant 5 refused");
      s.mutant = 0; s.mode = 3; CHECK(omega_prefill_attn_prog_stats(&s, NULL, NULL, NULL) != 0, "mode 3 refused");
      s.mode = OMEGA_PATTN_MODE_ATTEND; s.mirror = 1; CHECK(omega_prefill_attn_prog_stats(&s, NULL, NULL, NULL) != 0, "mirror outside APPEND refused");
      s.mode = OMEGA_PATTN_MODE_APPEND; uint32_t n = 0, gp = 0, ph = 0;
      CHECK(omega_prefill_attn_prog_stats(&s, &n, &gp, &ph) == 0 && n <= 1024 && omega_bw_gpr_declared(gp) <= 128, "APPEND + mirror in budget: %u insns %u gprs", n, gp);
      printf("codegen warps 8 APPEND + mirror: %u instructions, %u registers\n", n, gp); }
    CHECK(omega_gpu_prefill_attn(NULL, 0, 0, 1, NULL, EPS, 1.0f, NULL, NULL) == OMEGA_GPU_DATTN_BAD_ARGS, "NULL cache refused");
    CHECK(OMEGA_PATTN_SHARED_BYTES <= 16384u, "shared memory within the QMD cap");
    /* reference sanity: a single token at position 0 returns its own v row (rounded) */
    { Case c; case_make(&c, 256, 0, 1, 5, 1.0f); uint16_t kr[512], vr[512];
      emu_rows(c.qkv, c.kn, EPS, 0, THETA, kr, vr);
      for (unsigned gk = 0; gk < 4; gk++) { memcpy(c.kc + (size_t)gk * 256 * 128, kr + gk * 128, 256); memcpy(c.vc + (size_t)gk * 256 * 128, vr + gk * 128, 256); }
      double *ref = malloc(OUTW * sizeof(double)); reference(&c, c.kc, c.vc, 1.0f, ref);
      int good = 1; for (unsigned h = 0; h < 32 && good; h++) for (unsigned i = 0; i < 128; i++) if (ref[h * 128 + i] != (double)omega_dattn_bf16_f32(vr[(h / 8) * 128 + i])) { good = 0; break; }
      CHECK(good, "reference: one token at position 0 returns the rounded v row of its KV group");
      free(ref); case_free(&c); }
}

/* ------------------------------------------------------------- simulator */
static uint64_t g_sim_issues;
/* mirror planes for the simulated APPEND (write) / ATTEND (read) launches */
static struct { uint8_t *k, *v; uint32_t kvs; size_t plane; int write, read; } g_sm;
/* one simulated launch of prog over tokens p0 .. p0+mc-1 whose rows are qbuf / obuf (host memory) */
static int sim_launch(const BlackwellIRProgram *prog, uint32_t W, const Case *c, uint8_t *kv, size_t plane, const float *rope, const float *norms,
                      float out_scale, float *qbuf, float *obuf, uint32_t p0, uint32_t mc, int order, char *msg, size_t n) {
    static uint8_t smem[OMEGA_PATTN_SHARED_BYTES];
    WsSim s; memset(&s, 0, sizeof s);
    s.p = prog; s.threads = 32 * W; s.order = order; s.smem = smem; s.smem_bytes = sizeof smem;
    uint64_t A[7] = { (uintptr_t)qbuf, (uintptr_t)kv, (uintptr_t)(kv + plane), (uintptr_t)rope, (uintptr_t)norms, (uintptr_t)(norms + 128), (uintptr_t)obuf };
    for (unsigned i = 0; i < 7; i++) { s.cbank[(0x380 + 8 * i) / 4] = (uint32_t)A[i]; s.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
    float fe = EPS, fo = out_scale, fq = (float)((1.0 / sqrt(128.0)) * 1.4426950408889634);
    s.cbank[OMEGA_PATTN_ARG_POS0 / 4] = p0; s.cbank[OMEGA_PATTN_ARG_MC / 4] = mc; s.cbank[OMEGA_PATTN_ARG_KV_STRIDE / 4] = c->max_pos * 256;
    memcpy(&s.cbank[OMEGA_PATTN_ARG_EPS / 4], &fe, 4); memcpy(&s.cbank[OMEGA_PATTN_ARG_OUT_SCALE / 4], &fo, 4); memcpy(&s.cbank[OMEGA_PATTN_ARG_QK_SCALE / 4], &fq, 4);
    s.nranges = 5;
    s.ranges[0].lo = (uintptr_t)qbuf; s.ranges[0].hi = (uintptr_t)qbuf + (size_t)mc * QKVW * 4;
    s.ranges[1].lo = (uintptr_t)kv; s.ranges[1].hi = (uintptr_t)kv + 2 * plane;
    s.ranges[2].lo = (uintptr_t)rope; s.ranges[2].hi = (uintptr_t)rope + (size_t)c->max_pos * 512;
    s.ranges[3].lo = (uintptr_t)norms; s.ranges[3].hi = (uintptr_t)norms + 1024;
    s.ranges[4].lo = (uintptr_t)obuf; s.ranges[4].hi = (uintptr_t)obuf + (size_t)mc * OUTW * 4;
    if (g_sm.write) {
        uint64_t k2 = (uintptr_t)g_sm.k, v2 = (uintptr_t)g_sm.v;
        s.cbank[OMEGA_PATTN_ARG_KC2 / 4] = (uint32_t)k2; s.cbank[OMEGA_PATTN_ARG_KC2 / 4 + 1] = (uint32_t)(k2 >> 32);
        s.cbank[OMEGA_PATTN_ARG_VC2 / 4] = (uint32_t)v2; s.cbank[OMEGA_PATTN_ARG_VC2 / 4 + 1] = (uint32_t)(v2 >> 32);
        s.cbank[OMEGA_PATTN_ARG_KV_STRIDE2 / 4] = g_sm.kvs;
        s.ranges[5].lo = (uintptr_t)g_sm.k; s.ranges[5].hi = (uintptr_t)g_sm.k + 2 * g_sm.plane; s.nranges = 6;
    }
    if (g_sm.read) {   /* ATTEND reads only the mirror: the cache range is replaced, so a stray cache read faults */
        uint64_t k2 = (uintptr_t)g_sm.k, v2 = (uintptr_t)g_sm.v;
        s.cbank[OMEGA_PATTN_ARG_KC / 4] = (uint32_t)k2; s.cbank[OMEGA_PATTN_ARG_KC / 4 + 1] = (uint32_t)(k2 >> 32);
        s.cbank[OMEGA_PATTN_ARG_VC / 4] = (uint32_t)v2; s.cbank[OMEGA_PATTN_ARG_VC / 4 + 1] = (uint32_t)(v2 >> 32);
        s.cbank[OMEGA_PATTN_ARG_KV_STRIDE / 4] = g_sm.kvs;
        s.ranges[1].lo = (uintptr_t)g_sm.k; s.ranges[1].hi = (uintptr_t)g_sm.k + 2 * g_sm.plane;
    }
    int err = ws_run(&s, (mc + W - 1) / W, 4);
    if (err) { snprintf(msg, n, "launch at %u: %s (pc %d)", p0, s.msg, s.err_pc); return -1; }
    g_sim_issues += s.warp_issues;
    return 0;
}
/* runs the API's launch plan through the simulator on host buffers; kv: [K plane | V plane] holding the history.
 * progs[mode]; split: one APPEND + one ATTEND launch over all m tokens, else fused 64-token chunks */
static int sim_call(BlackwellIRProgram *const progs[4], int split, uint32_t W, const Case *c, uint8_t *kv, size_t plane, const float *rope, const float *norms,
                    float out_scale, float *out, int order, char *msg, size_t n) {
    float *qbuf = aligned_alloc(16, (size_t)c->m * QKVW * 4), *obuf = aligned_alloc(16, (size_t)c->m * OUTW * 4);
    memcpy(qbuf, c->qkv, (size_t)c->m * QKVW * 4);
    for (size_t i = 0; i < (size_t)c->m * OUTW; i++) obuf[i] = -999.0f;
    int err = 0;
    if (split) {
        uint8_t *mir = NULL;
        if (split == 2) {   /* the API's mirror plan: history rows copied in, APPEND writes both, ATTEND reads the mirror */
            const uint32_t P = c->pos0 + c->m, Pr = (P + 31) & ~31u;
            g_sm.kvs = Pr * 256; g_sm.plane = (size_t)4 * g_sm.kvs;
            mir = aligned_alloc(16, 2 * g_sm.plane); memset(mir, 0, 2 * g_sm.plane);
            g_sm.k = mir; g_sm.v = mir + g_sm.plane;
            for (unsigned h = 0; h < 4; h++) {
                memcpy(g_sm.k + (size_t)h * g_sm.kvs, kv + (size_t)h * c->max_pos * 256, (size_t)c->pos0 * 256);
                memcpy(g_sm.v + (size_t)h * g_sm.kvs, kv + plane + (size_t)h * c->max_pos * 256, (size_t)c->pos0 * 256);
            }
            g_sm.write = 1;
        }
        err = sim_launch(progs[split == 2 ? 3 : OMEGA_PATTN_MODE_APPEND], W, c, kv, plane, rope, norms, out_scale, qbuf, obuf, c->pos0, c->m, order, msg, n);
        /* APPEND writes no output */
        for (size_t i = 0; i < (size_t)c->m * OUTW && !err; i++) if (obuf[i] != -999.0f) { err = -1; snprintf(msg, n, "APPEND launch wrote output word %zu", i); }
        if (split == 2 && !err) for (unsigned h = 0; h < 4 && !err; h++) {   /* the mirror holds exactly the cache rows 0 .. P-1 */
            const size_t P = (size_t)c->pos0 + c->m;
            if (memcmp(g_sm.k + (size_t)h * g_sm.kvs, kv + (size_t)h * c->max_pos * 256, P * 256) ||
                memcmp(g_sm.v + (size_t)h * g_sm.kvs, kv + plane + (size_t)h * c->max_pos * 256, P * 256)) { err = -1; snprintf(msg, n, "mirror rows of head %u differ from the cache", h); }
        }
        g_sm.write = 0; g_sm.read = split == 2;
        if (!err) err = sim_launch(progs[OMEGA_PATTN_MODE_ATTEND], W, c, kv, plane, rope, norms, out_scale, qbuf, obuf, c->pos0, c->m, order, msg, n);
        g_sm.read = 0; free(mir);
    } else {
        for (uint32_t done = 0; done < c->m && !err; done += 64) {
            const uint32_t mc = c->m - done < 64 ? c->m - done : 64;
            err = sim_launch(progs[OMEGA_PATTN_MODE_FUSED], W, c, kv, plane, rope, norms, out_scale, qbuf + (size_t)done * QKVW, obuf + (size_t)done * OUTW,
                             c->pos0 + done, mc, order, msg, n);
        }
    }
    if (!err) {
        size_t unwritten = 0; for (size_t i = 0; i < (size_t)c->m * OUTW; i++) if (obuf[i] == -999.0f) unwritten++;
        if (unwritten) { snprintf(msg, n, "%zu output words never written", unwritten); err = -1; }
        else memcpy(out, obuf, (size_t)c->m * OUTW * 4);
    }
    free(qbuf); free(obuf);
    return err;
}
static int sim_point(uint32_t W, int mutant, int split, uint32_t max_pos, uint32_t pos0, uint32_t m, float qscale, uint32_t seed, int orders,
                     char *msg, size_t n, double *worst, size_t *bad) {
    static BlackwellIRProgram p0, p1, p2, p3;
    BlackwellIRProgram *const progs[4] = { &p0, &p1, &p2, &p3 };
    for (uint32_t md = 0; md < 4; md++) {
        OmegaPrefillAttnSpec spec = { W, mutant, md == 3 ? OMEGA_PATTN_MODE_APPEND : md, md == 3 };
        if (omega_blackwell_codegen_prefill_attn_prog(&spec, progs[md]) != 0) { snprintf(msg, n, "codegen refused (mode %u)", md); return -1; }
    }
    Case c; case_make(&c, max_pos, pos0, m, seed, qscale);
    size_t plane = (size_t)4 * max_pos * 256 + OMEGA_DATTN_PLANE_SLACK;
    uint8_t *kv = aligned_alloc(16, 2 * plane);
    float *rope = aligned_alloc(16, (size_t)max_pos * 512); for (uint32_t p = 0; p < max_pos; p++) omega_dattn_rope_row(p, THETA, rope + (size_t)p * 128);
    float *norms = aligned_alloc(16, 1024); memcpy(norms, c.qn, 512); memcpy(norms + 128, c.kn, 512);
    float *out = malloc((size_t)m * OUTW * 4), *copy = malloc((size_t)m * OUTW * 4);
    int err = 0;
    for (int order = 0; order < orders && !err; order++) {
        memset(kv, 0, 2 * plane);
        memcpy(kv, c.kc, (size_t)4 * max_pos * 256); memcpy(kv + plane, c.vc, (size_t)4 * max_pos * 256);
        err = sim_call(progs, split, W, &c, kv, plane, rope, norms, 1.0f, out, order, msg, n);
        if (!err && order == 0) memcpy(copy, out, (size_t)m * OUTW * 4);
        else if (!err && memcmp(copy, out, (size_t)m * OUTW * 4) != 0) { err = -1; snprintf(msg, n, "the fragment order changed the result"); }
    }
    if (!err) {
        size_t kd, vd; int mu;
        rows_vs_emu(&c, (const uint16_t *)kv, (const uint16_t *)(kv + plane), &kd, &vd, &mu);
        *bad = parity(&c, (const uint16_t *)kv, (const uint16_t *)(kv + plane), out, 1.0f, worst);
        if (kd || vd) *bad += 1000000;
        snprintf(msg, n, "bad=%zu worst=%.3f rows: k %zu v %zu words differ from the emulation (max %d ulp)", *bad, *worst, kd, vd, mu);
    }
    free(kv); free(rope); free(norms); free(out); free(copy); case_free(&c);
    return err;
}
static void sim_battery(void) {
    char msg[300]; double worst; size_t bad;
    struct { uint32_t w, max_pos, pos0, m; float qs; int orders, split; const char *name; } pts[] = {
        { 8, 256, 0, 1, 1.0f, 2, 0, "one token at position 0" },
        { 8, 256, 0, 5, 1.0f, 2, 0, "5 tokens from 0 (partial CTA)" },
        { 4, 256, 37, 19, 1.0f, 2, 0, "19 tokens from 37, warps 4" },
        { 8, 256, 37, 19, 4.0f, 1, 0, "19 tokens from 37, sharp q (x4)" },
        { 8, 256, 100, 64, 1.0f, 1, 0, "64 tokens from 100 (full chunk)" },
        { 8, 512, 0, 70, 1.0f, 1, 0, "70 tokens from 0 (two chunks)" },
        { 4, 512, 200, 33, 1.0f, 1, 0, "33 tokens from 200, warps 4" },
        { 8, 256, 0, 1, 1.0f, 2, 1, "split: one token at position 0" },
        { 8, 256, 0, 5, 1.0f, 2, 1, "split: 5 tokens from 0" },
        { 4, 256, 37, 19, 1.0f, 2, 1, "split: 19 tokens from 37, warps 4" },
        { 8, 512, 0, 70, 1.0f, 1, 1, "split: 70 tokens from 0" },
        { 8, 512, 37, 141, 4.0f, 1, 1, "split: 141 tokens from 37, sharp q" },
        { 8, 256, 0, 5, 1.0f, 2, 2, "mirror: 5 tokens from 0" },
        { 4, 256, 37, 19, 1.0f, 2, 2, "mirror: 19 tokens from 37, warps 4" },
        { 8, 512, 37, 141, 4.0f, 1, 2, "mirror: 141 tokens from 37, sharp q" },
        { 8, 512, 300, 45, 1.0f, 1, 2, "mirror: 45 tokens from 300" },
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        int err = sim_point(pts[i].w, 0, pts[i].split, pts[i].max_pos, pts[i].pos0, pts[i].m, pts[i].qs, 100 + (uint32_t)i, pts[i].orders, msg, sizeof msg, &worst, &bad);
        CHECK(err == 0 && bad == 0, "sim %s: %s", pts[i].name, msg);
        printf("sim %-40s W=%u: %s %s\n", pts[i].name, pts[i].w, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
        fflush(stdout);
    }
    for (int split = 0; split <= 2; split++) for (int m = 1; m <= 4; m++) {
        int err = sim_point(8, m, split, 256, 40, 24, 1.0f, 200, 1, msg, sizeof msg, &worst, &bad);
        int caught = err != 0 || bad > 0;
        CHECK(caught, "sim %s mutant %d NOT caught (%s)", split == 2 ? "mirror" : split ? "split" : "fused", m, msg);
        printf("sim %s mutant %d at 24 tokens from 40: %s (%s)\n", split == 2 ? "mirror" : split ? "split" : "fused", m, caught ? "caught" : "MISSED", msg);
    }
    printf("sim: %" PRIu64 " warp issues in total\n", g_sim_issues);
}

/* ------------------------------------------------------------- chip */
static FILE *g_json;
static int g_warm;  /* untimed calls before each timing series (--warm N) */
static int g_cached;  /* --cached: device-entry checks use omega_gpu_pf_alloc_dev buffers (needs PATTN_HAVE_PF) */
static int g_mirror_on = 1;
static int g_fused;  /* --fused: the cut-1 fused path, which never builds a K/V mirror */
/* the mirror contract: forget the cache's K/V mirrors before freeing it */
static void cache_free(OmegaGpuDecodeAttnCache *c) { omega_gpu_prefill_attn_kv_mirror_forget(c); omega_gpu_decode_attn_cache_free(c); }
static int cache_with_history(const Case *c, OmegaGpuDecodeAttnCache **out) {
    int rc = omega_gpu_decode_attn_cache_alloc(1, c->max_pos, THETA, out);
    if (rc != OMEGA_GPU_DATTN_OK) return rc;
    rc = omega_gpu_decode_attn_set_norms(*out, 0, c->qn, c->kn);
    for (unsigned g = 0; g < 4 && rc == OMEGA_GPU_DATTN_OK && c->pos0; g++)
        rc = omega_gpu_decode_attn_cache_write(*out, 0, g, 0, c->pos0, c->kc + (size_t)g * c->max_pos * 128, c->vc + (size_t)g * c->max_pos * 128);
    return rc;
}
/* reads rows pos0 .. pos0+m-1 of every KV head into the case-shaped arrays kc/vc (history copied from the case) */
static int read_rows(OmegaGpuDecodeAttnCache *cc, const Case *c, uint16_t *kc, uint16_t *vc) {
    memcpy(kc, c->kc, (size_t)4 * c->max_pos * 256); memcpy(vc, c->vc, (size_t)4 * c->max_pos * 256);
    for (unsigned g = 0; g < 4; g++) for (uint32_t t = 0; t < c->m; t++) {
        size_t o = ((size_t)g * c->max_pos + c->pos0 + t) * 128;
        if (omega_gpu_decode_attn_cache_read(cc, 0, g, c->pos0 + t, kc + o, vc + o) != OMEGA_GPU_DATTN_OK) return -1;
    }
    return 0;
}
static int chip_point(uint32_t max_pos, uint32_t pos0, uint32_t m, float qscale, uint32_t seed, int with_decode, float out_scale, int reps) {
    Case c; case_make(&c, max_pos, pos0, m, seed, qscale);
    OmegaGpuDecodeAttnCache *pc = NULL, *dc = NULL;
    float *out = malloc((size_t)m * OUTW * 4), *out2 = malloc((size_t)m * OUTW * 4), *dout = malloc((size_t)m * OUTW * 4);
    uint16_t *kc = malloc((size_t)4 * max_pos * 256), *vc = malloc((size_t)4 * max_pos * 256);
    uint16_t *dk = malloc((size_t)4 * max_pos * 256), *dv = malloc((size_t)4 * max_pos * 256);
    int ok = 1;
    OmegaGpuPrefillAttnInfo info;
    int rc = cache_with_history(&c, &pc);
    if (rc == OMEGA_GPU_DATTN_OK) rc = omega_gpu_prefill_attn(pc, 0, pos0, m, c.qkv, EPS, out_scale, out, &info);
    CHECK(rc == OMEGA_GPU_DATTN_OK, "chip prefill pos0 %u m %u: rc %s (%s)", pos0, m, omega_gpu_decode_attn_rc_name(rc), omega_gpu_decode_attn_last_error());
    if (rc != OMEGA_GPU_DATTN_OK) { ok = 0; goto done; }
    CHECK(read_rows(pc, &c, kc, vc) == 0, "read rows");
    double worst; size_t bad = parity(&c, kc, vc, out, out_scale, &worst);
    size_t kd, vd; int mu; rows_vs_emu(&c, kc, vc, &kd, &vd, &mu);
    CHECK(bad == 0, "chip pos0 %u m %u: %zu outputs outside the tolerance (worst %.3f)", pos0, m, bad, worst);
    CHECK(vd == 0 && mu <= 1, "chip pos0 %u m %u: rows vs host emulation: k %zu differ (max %d ulp), v %zu differ", pos0, m, kd, mu, vd);
    printf("chip pos0 %5u m %5u: parity bad=%zu worst=%.3f; rows vs emulation k %zu (max %d ulp) v %zu; chunks %u grid %ux%u x%u thr; chip %.3f ms call %.3f ms\n",
           pos0, m, bad, worst, kd, mu, vd, info.chunks, info.grid_x, info.grid_y, info.threads, info.elapsed_ns / 1e6, info.call_ns / 1e6);
    ok &= bad == 0 && vd == 0 && mu <= 1;
    /* second call: same rows rewritten, output bit-identical */
    rc = omega_gpu_prefill_attn(pc, 0, pos0, m, c.qkv, EPS, out_scale, out2, NULL);
    int same = rc == OMEGA_GPU_DATTN_OK && memcmp(out, out2, (size_t)m * OUTW * 4) == 0;
    CHECK(same, "chip pos0 %u m %u: second call not bit-identical", pos0, m);
    ok &= same;
    double pre_med = 0, dec_ms = 0;
    if (reps > 0) {
        uint64_t *t = malloc(sizeof(uint64_t) * reps), *tc = malloc(sizeof(uint64_t) * reps);
        static uint64_t lns[256], worst_l[256]; uint32_t worst_n = 0; uint64_t worst_t = 0;
        for (int r = 0; r < g_warm; r++) omega_gpu_prefill_attn(pc, 0, pos0, m, c.qkv, EPS, out_scale, out2, NULL);
        printf("  reps chip ms (in order, %d untimed warm calls first):", g_warm);
        for (int r = 0; r < reps; r++) {
            omega_gpu_prefill_attn(pc, 0, pos0, m, c.qkv, EPS, out_scale, out2, &info); t[r] = info.elapsed_ns; tc[r] = info.call_ns;
            uint32_t nl = omega_gpu_prefill_attn_test_launch_ns(lns, 256);
            printf(" %.3f", t[r] / 1e6);
            if (t[r] > worst_t) { worst_t = t[r]; worst_n = nl < 256 ? nl : 256; memcpy(worst_l, lns, worst_n * sizeof *lns); }
        }
        printf("\n  slowest rep, per launch ms (%u launches):", worst_n);
        for (uint32_t i = 0; i < worst_n; i++) printf(" %.3f", worst_l[i] / 1e6);
        printf("\n");
        qsort(t, reps, sizeof *t, cmp_u64); qsort(tc, reps, sizeof *tc, cmp_u64);
        pre_med = t[reps / 2] / 1e6;
        double flops = 0; for (uint32_t i = 0; i < m; i++) flops += 4.0 * 32 * 128 * (pos0 + i + 1.0);
        printf("  timing prefill pos0 %u m %u: chip median %.3f ms (min %.3f) call median %.3f ms; %.2f useful TFLOPS on chip time\n",
               pos0, m, pre_med, t[0] / 1e6, tc[reps / 2] / 1e6, flops / (pre_med * 1e-3) / 1e12);
        if (g_json) fprintf(g_json, "  {\"pos0\": %u, \"m\": %u, \"prefill_chip_ms_median\": %.4f, \"prefill_call_ms_median\": %.4f, \"useful_tflops\": %.3f},\n", pos0, m, pre_med, tc[reps / 2] / 1e6, flops / (pre_med * 1e-3) / 1e12);
        free(t); free(tc);
    }
    if (with_decode) {
        /* gate 2: m decode steps on a twin cache: rows bit-identical, outputs within the tolerance of each other */
        rc = cache_with_history(&c, &dc);
        uint64_t chip = 0, call = 0; OmegaGpuDecodeAttnInfo di;
        for (uint32_t i = 0; i < m && rc == OMEGA_GPU_DATTN_OK; i++) { rc = omega_gpu_decode_attn(dc, 0, pos0 + i, c.qkv + (size_t)i * QKVW, EPS, out_scale, dout + (size_t)i * OUTW, &di); chip += di.elapsed_ns; call += di.call_ns; }
        CHECK(rc == OMEGA_GPU_DATTN_OK, "decode twin: rc %s (%s)", omega_gpu_decode_attn_rc_name(rc), omega_gpu_decode_attn_last_error());
        if (rc == OMEGA_GPU_DATTN_OK && read_rows(dc, &c, dk, dv) == 0) {
            size_t rk = 0, rv = 0;
            for (unsigned g = 0; g < 4; g++) for (uint32_t t2 = 0; t2 < m; t2++) for (unsigned i = 0; i < 128; i++) {
                size_t o = ((size_t)g * max_pos + pos0 + t2) * 128 + i; rk += kc[o] != dk[o]; rv += vc[o] != dv[o]; }
            double dw = 0; size_t dbad = 0, tol_n = 0;
            for (size_t i = 0; i < (size_t)m * OUTW; i++) { double d = fabs((double)out[i] - (double)dout[i]) / (1e-2 * out_scale); if (d > dw) dw = d; if (d > 1.0) dbad++; tol_n++; }
            CHECK(rk == 0 && rv == 0, "gate 2: prefill rows differ from the decode kernel's rows: k %zu v %zu", rk, rv);
            CHECK(dbad == 0, "gate 2: prefill vs decode outputs: %zu outside the tolerance (worst %.3f)", dbad, dw);
            dec_ms = chip / 1e6;
            printf("  gate 2 vs %u decode steps: rows k %zu v %zu differ; outputs worst %.3f of tol; decode loop chip %.3f ms call %.3f ms%s\n",
                   m, rk, rv, dw, chip / 1e6, call / 1e6, pre_med > 0 ? "" : "");
            if (pre_med > 0) printf("  speed: prefill %.3f ms vs decode loop %.3f ms chip time: %.1fx\n", pre_med, dec_ms, dec_ms / pre_med);
            if (g_json) fprintf(g_json, "  {\"pos0\": %u, \"m\": %u, \"gate2_rows_k_diff\": %zu, \"gate2_rows_v_diff\": %zu, \"gate2_out_worst\": %.4f, \"decode_loop_chip_ms\": %.4f, \"decode_loop_call_ms\": %.4f},\n", pos0, m, rk, rv, dw, chip / 1e6, call / 1e6);
            ok &= rk == 0 && rv == 0 && dbad == 0;
        } else ok = 0;
    }
done:
    if (pc) cache_free(pc);
    if (dc) cache_free(dc);
    free(out); free(out2); free(dout); free(kc); free(vc); free(dk); free(dv); case_free(&c);
    return ok;
}
/* device-resident entry: qkv and out in session buffers (GPU VAs); output and rows must equal the host entry's bit for bit */
static int dev_buf(size_t bytes, OmegaGpuMem *m) {
    omega_gpu_session_lock();
    int rc = omega_gpu_session_open() ? omega_gpu_session_alloc(bytes, m) : -1;
    omega_gpu_session_unlock();
    return rc;
}
static int chip_dev(uint32_t max_pos, uint32_t pos0, uint32_t m, uint32_t seed, int reps) {
    Case c; case_make(&c, max_pos, pos0, m, seed, 1.0f);
    OmegaGpuDecodeAttnCache *hc = NULL, *dc = NULL;
    float *hout = malloc((size_t)m * OUTW * 4);
    uint16_t *hk = malloc((size_t)4 * max_pos * 256), *hv = malloc((size_t)4 * max_pos * 256);
    uint16_t *dk = malloc((size_t)4 * max_pos * 256), *dv = malloc((size_t)4 * max_pos * 256);
    OmegaGpuMem q, o; memset(&q, 0, sizeof q); memset(&o, 0, sizeof o);
    float *odh = NULL;
    int ok = 1;
    OmegaGpuPrefillAttnInfo info, dinfo;
    int rc = cache_with_history(&c, &hc);
    if (rc == OMEGA_GPU_DATTN_OK) rc = cache_with_history(&c, &dc);
    if (rc == OMEGA_GPU_DATTN_OK) rc = omega_gpu_prefill_attn(hc, 0, pos0, m, c.qkv, EPS, 1.0f, hout, &info);
    CHECK(rc == OMEGA_GPU_DATTN_OK, "dev point %u+%u: host entry rc %s", pos0, m, omega_gpu_decode_attn_rc_name(rc));
    if (rc != OMEGA_GPU_DATTN_OK) { ok = 0; goto done; }
#if PATTN_HAVE_PF
    if (g_cached) {   /* W-L1G's GPU-cached device-private buffers, filled and read only through GPU copies */
        CHECK(omega_gpu_pf_alloc_dev((size_t)m * QKVW * 4, &q) == OMEGA_GPU_PF_OK && omega_gpu_pf_alloc_dev((size_t)m * OUTW * 4, &o) == OMEGA_GPU_PF_OK, "dev point: cached device buffers");
        if (!q.va || !o.va) { ok = 0; goto done; }
        odh = malloc((size_t)m * OUTW * 4);
        for (size_t i = 0; i < (size_t)m * OUTW; i++) odh[i] = -999.0f;
        CHECK(omega_gpu_pf_upload(&q, 0, c.qkv, (size_t)m * QKVW * 4) == OMEGA_GPU_PF_OK && omega_gpu_pf_upload(&o, 0, odh, (size_t)m * OUTW * 4) == OMEGA_GPU_PF_OK, "dev point: upload through the copy kernel");
    } else
#endif
    {
    CHECK(dev_buf((size_t)m * QKVW * 4, &q) == 0 && dev_buf((size_t)m * OUTW * 4, &o) == 0, "dev point: session buffers");
    if (!q.cpu || !o.cpu) { ok = 0; goto done; }
    memcpy(q.cpu, c.qkv, (size_t)m * QKVW * 4);
    float *op = o.cpu; for (size_t i = 0; i < (size_t)m * OUTW; i++) op[i] = -999.0f;
    omega_gpu_session_lock(); omega_gpu_session_sync_to_gpu(&q); omega_gpu_session_sync_to_gpu(&o); omega_gpu_session_unlock();
    }
    /* negative control: misaligned and zero VAs are refused before any launch */
    CHECK(omega_gpu_prefill_attn_dev(dc, 0, pos0, m, q.va + 4, EPS, 1.0f, o.va, NULL) == OMEGA_GPU_DATTN_BAD_ARGS, "dev: misaligned qkv VA refused");
    CHECK(omega_gpu_prefill_attn_dev(dc, 0, pos0, m, q.va, EPS, 1.0f, 0, NULL) == OMEGA_GPU_DATTN_BAD_ARGS, "dev: zero out VA refused");
    rc = omega_gpu_prefill_attn_dev(dc, 0, pos0, m, q.va, EPS, 1.0f, o.va, &dinfo);
    CHECK(rc == OMEGA_GPU_DATTN_OK, "dev point %u+%u: rc %s (%s)", pos0, m, omega_gpu_decode_attn_rc_name(rc), omega_gpu_decode_attn_last_error());
    if (rc != OMEGA_GPU_DATTN_OK) { ok = 0; goto done; }
    float *od;
#if PATTN_HAVE_PF
    if (g_cached) { CHECK(omega_gpu_pf_download(&o, 0, odh, (size_t)m * OUTW * 4) == OMEGA_GPU_PF_OK, "dev point: download through the copy kernel"); od = odh; }
    else
#endif
    { omega_gpu_session_lock(); omega_gpu_session_sync_to_cpu(&o); omega_gpu_session_unlock(); od = o.cpu; }
    size_t unw = 0, diff = 0;
    for (size_t i = 0; i < (size_t)m * OUTW; i++) { if (od[i] == -999.0f) unw++; else if (memcmp(&od[i], &hout[i], 4)) diff++; }
    CHECK(read_rows(hc, &c, hk, hv) == 0 && read_rows(dc, &c, dk, dv) == 0, "dev point: read rows");
    size_t rk = 0, rv = 0;
    for (size_t i = 0; i < (size_t)4 * max_pos * 128; i++) { rk += hk[i] != dk[i]; rv += hv[i] != dv[i]; }
    CHECK(unw == 0 && diff == 0, "dev point %u+%u: %zu outputs unwritten, %zu differ from the host entry", pos0, m, unw, diff);
    CHECK(rk == 0 && rv == 0, "dev point %u+%u: cache rows differ from the host entry: k %zu v %zu", pos0, m, rk, rv);
    ok &= unw == 0 && diff == 0 && rk == 0 && rv == 0;
    printf("chip dev%s pos0 %5u m %5u: outputs vs host entry: %zu unwritten, %zu differ; rows k %zu v %zu differ; launches %u grid %ux%u; chip %.3f ms call %.3f ms\n",
           g_cached ? " (cached buffers)" : "", pos0, m, unw, diff, rk, rv, dinfo.chunks, dinfo.grid_x, dinfo.grid_y, dinfo.elapsed_ns / 1e6, dinfo.call_ns / 1e6);
    if (reps > 0) {
        uint64_t *t = malloc(sizeof(uint64_t) * reps), *tc = malloc(sizeof(uint64_t) * reps);
        for (int r = 0; r < g_warm; r++) omega_gpu_prefill_attn_dev(dc, 0, pos0, m, q.va, EPS, 1.0f, o.va, NULL);
        printf("  dev reps chip ms:");
        for (int r = 0; r < reps; r++) { omega_gpu_prefill_attn_dev(dc, 0, pos0, m, q.va, EPS, 1.0f, o.va, &dinfo); t[r] = dinfo.elapsed_ns; tc[r] = dinfo.call_ns; printf(" %.3f", t[r] / 1e6); }
        qsort(t, reps, sizeof *t, cmp_u64); qsort(tc, reps, sizeof *tc, cmp_u64);
        double flops = 0; for (uint32_t i = 0; i < m; i++) flops += 4.0 * 32 * 128 * (pos0 + i + 1.0);
        printf("\n  timing dev pos0 %u m %u: chip median %.3f ms (min %.3f) call median %.3f ms; %.2f useful TFLOPS on chip time\n",
               pos0, m, t[reps / 2] / 1e6, t[0] / 1e6, tc[reps / 2] / 1e6, flops / (t[reps / 2] * 1e-9) / 1e12);
        if (g_json) fprintf(g_json, "  {\"dev\": 1, \"pos0\": %u, \"m\": %u, \"chip_ms_median\": %.4f, \"chip_ms_min\": %.4f, \"call_ms_median\": %.4f, \"out_diff\": %zu, \"rows_diff\": %zu},\n",
                            pos0, m, t[reps / 2] / 1e6, t[0] / 1e6, tc[reps / 2] / 1e6, diff, rk + rv);
        free(t); free(tc);
    }
done:
#if PATTN_HAVE_PF
    if (g_cached) { if (q.va) omega_gpu_pf_free(&q); if (o.va) omega_gpu_pf_free(&o); } else
#endif
    { omega_gpu_session_lock(); if (q.cpu) omega_gpu_session_free(&q); if (o.cpu) omega_gpu_session_free(&o); omega_gpu_session_unlock(); }
    free(odh);
    if (hc) cache_free(hc);
    if (dc) cache_free(dc);
    free(hout); free(hk); free(hv); free(dk); free(dv); case_free(&c);
    return ok;
}
/* persistent mirror (cut 5): prefill 300, decode 5, prefill 64, then a foreign rewrite of rows 0..99 and prefill 32.
 * Cache X runs with the mirror, cache Y without (cut 3 path, the reference). Outputs must be bit-identical, except
 * the negative control: the rewrite WITHOUT invalidate must make X differ (the stale mirror is visible), and after
 * invalidate X must match again. */
static int chip_continuation(void) {
    if (!omega_gpu_prefill_attn_kv_mirror_available() || !g_mirror_on) { printf("chip continuation: skipped (no mirror in this build or --no-mirror)\n"); return 1; }
    /* the mirror lives on the split path (APPEND + ATTEND): the API calls mirror_get only when not fused, so the fused
     * path never builds one and the check has nothing to compare there. The default (split) run covers it. */
    if (g_fused) { printf("chip continuation: skipped (--fused: the fused path builds no K/V mirror; the split run covers it)\n"); return 1; }
    const uint32_t MP = 1024, NA = 300, ND = 5, NB = 64, NC = 32, N = NA + ND + NB + NC;
    Case c, junk; case_make(&c, MP, 0, N, 21, 1.0f); case_make(&junk, MP, 100, 1, 22, 1.0f);
    OmegaGpuDecodeAttnCache *x = NULL, *y = NULL;
    float *ox = malloc((size_t)N * OUTW * 4), *oy = malloc((size_t)N * OUTW * 4), *ox2 = malloc((size_t)NC * OUTW * 4);
    int ok = 1, rc = cache_with_history(&c, &x);
    if (rc == OMEGA_GPU_DATTN_OK) rc = cache_with_history(&c, &y);
    OmegaGpuDecodeAttnInfo di;
    OmegaGpuDecodeAttnCache *cs[2] = { x, y };
    float *os[2] = { ox, oy };
    for (int k = 0; k < 2 && rc == OMEGA_GPU_DATTN_OK; k++) {
        omega_gpu_prefill_attn_set_kv_mirror(k == 0);
        const float *q = c.qkv; float *o = os[k];
        rc = omega_gpu_prefill_attn(cs[k], 0, 0, NA, q, EPS, 1.0f, o, NULL);
        for (uint32_t i = 0; i < ND && rc == OMEGA_GPU_DATTN_OK; i++)
            rc = omega_gpu_decode_attn(cs[k], 0, NA + i, q + (size_t)(NA + i) * QKVW, EPS, 1.0f, o + (size_t)(NA + i) * OUTW, &di);
        if (rc == OMEGA_GPU_DATTN_OK) rc = omega_gpu_prefill_attn(cs[k], 0, NA + ND, NB, q + (size_t)(NA + ND) * QKVW, EPS, 1.0f, o + (size_t)(NA + ND) * OUTW, NULL);
    }
    CHECK(rc == OMEGA_GPU_DATTN_OK, "continuation: rc %s (%s)", omega_gpu_decode_attn_rc_name(rc), omega_gpu_decode_attn_last_error());
    if (rc != OMEGA_GPU_DATTN_OK) { ok = 0; goto done; }
    OmegaGpuPrefillAttnMirrorView mv;
    int vrc = omega_gpu_prefill_attn_kv_mirror_view(x, 0, &mv);
    CHECK(vrc == OMEGA_GPU_DATTN_OK && mv.valid == NA + ND + NB && mv.kv_stride == MP * 256u, "continuation: mirror view valid %u (want %u) stride %u", mv.valid, NA + ND + NB, mv.kv_stride);
    size_t d1 = 0; for (size_t i = 0; i < (size_t)(NA + ND + NB) * OUTW; i++) d1 += memcmp(&ox[i], &oy[i], 4) != 0;
    CHECK(d1 == 0, "continuation: %zu outputs differ, mirror vs no mirror", d1);
    /* foreign writer: rows 0..99 of head 0 rewritten in both caches */
    for (int k = 0; k < 2; k++) omega_gpu_decode_attn_cache_write(cs[k], 0, 0, 0, 100, junk.kc, junk.vc);
    const size_t offC = (size_t)(NA + ND + NB);
    omega_gpu_prefill_attn_set_kv_mirror(0);
    rc = omega_gpu_prefill_attn(y, 0, offC, NC, c.qkv + offC * QKVW, EPS, 1.0f, oy + offC * OUTW, NULL);
    omega_gpu_prefill_attn_set_kv_mirror(1);
    if (rc == OMEGA_GPU_DATTN_OK) rc = omega_gpu_prefill_attn(x, 0, offC, NC, c.qkv + offC * QKVW, EPS, 1.0f, ox2, NULL);   /* no invalidate */
    size_t stale = 0; for (size_t i = 0; i < (size_t)NC * OUTW; i++) stale += memcmp(&ox2[i], &oy[offC * OUTW + i], 4) != 0;
    omega_gpu_prefill_attn_kv_mirror_invalidate(x, UINT32_MAX, 0);
    if (rc == OMEGA_GPU_DATTN_OK) rc = omega_gpu_prefill_attn(x, 0, offC, NC, c.qkv + offC * QKVW, EPS, 1.0f, ox + offC * OUTW, NULL);
    size_t d2 = 0; for (size_t i = 0; i < (size_t)NC * OUTW; i++) d2 += memcmp(&ox[offC * OUTW + i], &oy[offC * OUTW + i], 4) != 0;
    CHECK(rc == OMEGA_GPU_DATTN_OK, "continuation C: rc %s", omega_gpu_decode_attn_rc_name(rc));
    CHECK(stale > 0, "negative control: a foreign rewrite without invalidate must show (stale mirror), %zu outputs differ", stale);
    CHECK(d2 == 0, "after invalidate: %zu outputs differ, mirror vs no mirror", d2);
    printf("chip continuation (prefill %u, decode %u, prefill %u, rewrite, prefill %u): mirror valid %u; outputs differ mirror vs none: %zu; "
           "stale control %zu differ (must be > 0); after invalidate %zu differ\n", NA, ND, NB, NC, mv.valid, d1, stale, d2);
    ok &= d1 == 0 && stale > 0 && d2 == 0 && vrc == OMEGA_GPU_DATTN_OK;
done:
    omega_gpu_prefill_attn_set_kv_mirror(1);
    if (x) cache_free(x);
    if (y) cache_free(y);
    free(ox); free(oy); free(ox2); case_free(&c); case_free(&junk);
    return ok;
}
static void chip_mutants(void) {
    for (int mu = 1; mu <= 4; mu++) {
        omega_gpu_prefill_attn_test_set_mutant(mu);
        Case c; case_make(&c, 256, 40, 24, 200, 1.0f);
        OmegaGpuDecodeAttnCache *pc = NULL;
        float *out = malloc((size_t)24 * OUTW * 4);
        uint16_t *kc = malloc((size_t)4 * 256 * 256), *vc = malloc((size_t)4 * 256 * 256);
        int rc = cache_with_history(&c, &pc);
        if (rc == OMEGA_GPU_DATTN_OK) rc = omega_gpu_prefill_attn(pc, 0, 40, 24, c.qkv, EPS, 1.0f, out, NULL);
        int caught = rc != OMEGA_GPU_DATTN_OK;
        double worst = 0; size_t bad = 0, kd = 0, vd = 0; int mx = 0;
        if (!caught && read_rows(pc, &c, kc, vc) == 0) {
            bad = parity(&c, kc, vc, out, 1.0f, &worst);
            rows_vs_emu(&c, kc, vc, &kd, &vd, &mx);
            caught = bad > 0 || vd > 0 || mx > 1;
        }
        CHECK(caught, "chip mutant %d NOT caught", mu);
        printf("chip mutant %d: %s (rc %s, bad %zu, worst %.2f, rows k %zu v %zu)\n", mu, caught ? "caught" : "MISSED", omega_gpu_decode_attn_rc_name(rc), bad, worst, kd, vd);
        if (pc) cache_free(pc);
        free(out); free(kc); free(vc); case_free(&c);
    }
    omega_gpu_prefill_attn_test_set_mutant(0);
}

int main(int argc, char **argv) {
    int host = 0, sim = 0, reps = 9, quick = 0, fused = 0, timing_only = 0; const char *json = NULL; uint32_t warps = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) host = 1;
        else if (!strcmp(argv[i], "--sim")) sim = 1;
        else if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warps") && i + 1 < argc) warps = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) json = argv[++i];
        else if (!strcmp(argv[i], "--fused")) fused = g_fused = 1;
        else if (!strcmp(argv[i], "--no-mirror")) { omega_gpu_prefill_attn_set_kv_mirror(0); g_mirror_on = 0; }
        else if (!strcmp(argv[i], "--cached") && PATTN_HAVE_PF) g_cached = 1;
        else if (!strcmp(argv[i], "--mirror-per-call")) omega_gpu_prefill_attn_test_set_mirror_per_call(1);
        else if (!strcmp(argv[i], "--timing-only")) timing_only = 1;
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc) g_warm = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--host-only | --sim | [--quick] [--reps N] [--warps 4|8] [--fused] [--no-mirror] [--mirror-per-call] [--cached] [--warm N] [--timing-only] [--out receipt.json]]\n", argv[0]); return 2; }
    }
    host_only();
    if (sim) sim_battery();
    if (!host && !sim) {
        if (json) { g_json = fopen(json, "w"); if (g_json) fprintf(g_json, "{\"test\": \"gpu_prefill_attn\", \"points\": [\n"); }
        if (warps) omega_gpu_prefill_attn_set_warps(warps);
        int ok = 1;
        if (fused) omega_gpu_prefill_attn_set_fused(1);
        printf("path: %s, warps %u, warm %d, dev buffers %s\n", fused ? "fused 64-token chunks" : g_mirror_on && omega_gpu_prefill_attn_kv_mirror_available() ? "split APPEND + ATTEND, ATTEND reads the GPU-cached K/V mirror" : "split APPEND + ATTEND, ATTEND reads the uncached cache", warps ? warps : OMEGA_GPU_PATTN_WARPS, g_warm, g_cached ? "GPU-cached (pf_alloc_dev)" : "session (uncached)");
        if (timing_only) {
            ok &= chip_point(4096, 0, 2048, 1.0f, 8, 0, 1.0f, reps);
            ok &= chip_dev(4096, 0, 2048, 8, reps);
            ok &= chip_dev(8192, 4096, 1500, 12, reps);
            ok &= chip_point(4096, 4032, 64, 1.0f, 7, 0, 1.0f, reps);
            ok &= chip_dev(8192, 4096, 64, 13, reps);
        } else {
        ok &= chip_point(256, 0, 1, 1.0f, 1, 1, 1.0f, 0);
        ok &= chip_point(256, 37, 19, 1.0f, 2, 1, 1.0f, 0);
        ok &= chip_point(256, 0, 64, 1.0f, 3, 1, 1.0f, reps);
        ok &= chip_point(512, 100, 70, 4.0f, 4, 1, 1.0f, 0);
        ok &= chip_point(512, 200, 64, 1.0f, 5, 1, 18446744073709551616.0f, 0);
        if (!quick) {
            ok &= chip_point(1024, 0, 512, 1.0f, 6, 1, 1.0f, reps);
            ok &= chip_point(4096, 4032, 64, 1.0f, 7, 1, 1.0f, reps);
            ok &= chip_point(4096, 0, 2048, 1.0f, 8, 0, 1.0f, reps);
        }
        ok &= chip_dev(512, 100, 70, 11, 0);
        if (!quick) { ok &= chip_dev(4096, 0, 2048, 8, reps); ok &= chip_dev(8192, 4096, 1500, 12, reps); }
        ok &= chip_continuation();
        chip_mutants();
        }
        (void)ok;
        if (g_json) { fprintf(g_json, "  {}\n], \"checks\": %d, \"failed\": %d}\n", g_checks, g_failed); fclose(g_json); }
    }
    printf("gpu_prefill_attn_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
