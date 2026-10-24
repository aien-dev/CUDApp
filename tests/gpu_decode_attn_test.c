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
 * gpu_decode_attn_test: the one-launch decode attention (OWN-30B cut 8, omega_gpu_decode_attn_api.h).
 *   --host-only   host pieces: bf16 rounding, RoPE row, reference sanity (context 1 returns v_new, a one-hot
 *                 key wins), codegen budget per depth and mutant, refusals. No device.
 *   --sim         the kernel through tests/bw_warp_sim_k.h (no chip): contexts 1, 7, 256, 257, 301, 701 (1..3
 *                 splits, the cross-split combine), both fragment orders, every mutant caught at context 301.
 *   (default)     the chip: parity against the double reference at contexts 1, 7, 256, 1024, 4096, 8192, 32768
 *                 (the appended K/V row read back and compared), second call identical, timing medians with
 *                 the host caches evicted, every mutant caught at context 301, JSON receipt with --out.
 * Tolerance: |out - ref| <= 3e-3 * max|v| + 1e-6 per element (MUFU.EX2 / RSQ are 2^-22-ish, bf16 rounding of the
 * appended K/V can flip one ulp against the f32 chip path). The mutants miss by 0.1 .. 1.0 of that scale.
 * --q-heads Q --kv-heads K --head-dim D: any geometry other than 32 / 4 / 128 is REFUSED with reasons (exit 3, receipt
 * verdict "REFUSED"); e.g. the BitNet b1.58 layout 20 / 5 / 128 (ternary engine T0).
 * Host and simulator passes are never chip results.
 */
#include "omega_gpu_decode_attn_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_hostmath.h"
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
static const double THETA = 10000000.0; /* config.json rope_theta */
static const float EPS = 1e-6f;

/* one layer's random state: qkv, history rows (bf16) for 4 KV heads, norms */
typedef struct {
    uint32_t max_pos, pos;
    float qkv[OMEGA_GPU_DATTN_QKV_WORDS], qn[128], kn[128];
    uint16_t *kc, *vc;   /* [4][max_pos][128] (history 0..pos-1 filled; the rest zero) */
    float vmax;
} Case;
static void case_make(Case *c, uint32_t max_pos, uint32_t pos, uint32_t seed, float qscale) {
    uint32_t s = seed;
    c->max_pos = max_pos; c->pos = pos;
    for (size_t i = 0; i < OMEGA_GPU_DATTN_QKV_WORDS; i++) c->qkv[i] = frand(&s, -1.0f, 1.0f) * (i < 4096 ? qscale : 1.0f);
    for (unsigned i = 0; i < 128; i++) { c->qn[i] = frand(&s, 0.8f, 1.2f); c->kn[i] = frand(&s, 0.8f, 1.2f); }
    size_t words = (size_t)4 * max_pos * 128;
    c->kc = calloc(words, 2); c->vc = calloc(words, 2);
    c->vmax = 1.0f;
    for (unsigned g = 0; g < 4; g++) for (uint32_t t = 0; t < pos; t++) for (unsigned i = 0; i < 128; i++) {
        size_t o = ((size_t)g * max_pos + t) * 128 + i;
        c->kc[o] = omega_dattn_bf16_rne(frand(&s, -1.0f, 1.0f)); c->vc[o] = omega_dattn_bf16_rne(frand(&s, -1.0f, 1.0f));
    }
}
static void case_free(Case *c) { free(c->kc); free(c->vc); }

/* ---- double-precision oracle (test-owned: it uses libm, which the library does not link) ----
 * RMSNorm + RoPE of one head (double), in place; cs: cos[64], sn: sin[64] */
static void norm_rope(double *x, const float *w, float eps, const double *cs, const double *sn, int rope) {
    double ss = 0; for (unsigned i = 0; i < 128; i++) ss += x[i] * x[i];
    double r = 1.0 / sqrt(ss / 128.0 + (double)eps);
    for (unsigned i = 0; i < 128; i++) x[i] = x[i] * r * (double)w[i];
    if (!rope) return;
    double y[128];
    for (unsigned i = 0; i < 64; i++) { y[i] = x[i] * cs[i] - x[i + 64] * sn[i]; y[i + 64] = x[i + 64] * cs[i] + x[i] * sn[i]; }
    memcpy(x, y, sizeof y);
}
/* Reference of the whole layer step. kc/vc: this layer's bf16 planes [4][max_pos][128] holding positions 0..pos-1;
 * out: [32*128] doubles (already * out_scale); k_new/v_new: the bf16 rows the kernel must have appended at pos
 * ([4][128] each). The reference rounds its k_new/v_new to bf16 and attends over the rounded values. */
static void omega_decode_attn_reference(const float *qkv, const uint16_t *kc, const uint16_t *vc, uint32_t max_pos, uint32_t pos,
                                        const float *q_norm, const float *k_norm, float eps, double rope_theta, float out_scale,
                                        double *out, uint16_t *k_new, uint16_t *v_new) {
    double cs[64], sn[64];
    for (unsigned i = 0; i < 64; i++) { double ang = (double)pos * pow(rope_theta, -(double)(2 * i) / 128.0); cs[i] = cos(ang); sn[i] = sin(ang); }
    const double qscale = 1.0 / sqrt(128.0);
    for (unsigned gk = 0; gk < 4; gk++) {
        double k[128];
        for (unsigned i = 0; i < 128; i++) k[i] = qkv[4096 + gk * 128 + i];
        norm_rope(k, k_norm, eps, cs, sn, 1);
        for (unsigned i = 0; i < 128; i++) { k_new[gk * 128 + i] = omega_dattn_bf16_rne((float)k[i]); v_new[gk * 128 + i] = omega_dattn_bf16_rne(qkv[4096 + 512 + gk * 128 + i]); }
    }
    double *p = calloc((size_t)pos + 1, sizeof(double));
    for (unsigned h = 0; h < 32; h++) {
        unsigned gk = h / 8;
        double q[128];
        for (unsigned i = 0; i < 128; i++) q[i] = qkv[h * 128 + i];
        norm_rope(q, q_norm, eps, cs, sn, 1);
        double mx = -1e300;
        for (uint32_t t = 0; t <= pos; t++) {
            const uint16_t *kr = t == pos ? k_new + gk * 128 : kc + ((size_t)gk * max_pos + t) * 128;
            double s = 0; for (unsigned i = 0; i < 128; i++) s += q[i] * (double)omega_dattn_bf16_f32(kr[i]);
            s *= qscale; p[t] = s; if (s > mx) mx = s;
        }
        double l = 0; for (uint32_t t = 0; t <= pos; t++) { p[t] = exp(p[t] - mx); l += p[t]; }
        double o[128]; memset(o, 0, sizeof o);
        for (uint32_t t = 0; t <= pos; t++) {
            const uint16_t *vr = t == pos ? v_new + gk * 128 : vc + ((size_t)gk * max_pos + t) * 128;
            for (unsigned i = 0; i < 128; i++) o[i] += p[t] * (double)omega_dattn_bf16_f32(vr[i]);
        }
        for (unsigned i = 0; i < 128; i++) out[h * 128 + i] = o[i] / l * (double)out_scale;
    }
    free(p);
}

/* the library's libm-free host math against libm: RoPE rows bit-equal or within one f32 ulp, double values close */
static void host_math_vs_libm(void) {
    uint32_t s = 77; unsigned flips = 0, worse = 0; double maxd = 0;
    static const uint32_t fixed[] = { 0, 1, 2, 3, 100, 255, 256, 4095, 4096, 8191, 32767, 32768, 65535 };
    for (unsigned n = 0; n < 1000 + sizeof fixed / sizeof fixed[0]; n++) {
        uint32_t pos = n < sizeof fixed / sizeof fixed[0] ? fixed[n] : lcg(&s) % 65536u;
        float row[128]; omega_dattn_rope_row(pos, THETA, row);
        for (unsigned i = 0; i < 64; i++) {
            double ang = (double)pos * pow(THETA, -(double)(2 * i) / 128.0);
            float c = (float)cos(ang), sn = (float)sin(ang);
            uint32_t a, b; memcpy(&a, &row[i], 4); memcpy(&b, &c, 4);
            int dc = (int)a - (int)b; memcpy(&a, &row[64 + i], 4); memcpy(&b, &sn, 4); int ds = (int)a - (int)b;
            if (dc < -1 || dc > 1 || ds < -1 || ds > 1) worse++; else if (dc || ds) flips++;
        }
    }
    CHECK(worse == 0, "rope rows vs libm: %u entries differ by more than one f32 ulp", worse);
    printf("rope rows vs libm: 1013 positions x 64 dims, %u one-ulp differences, %u worse\n", flips, worse);
    for (unsigned n = 0; n < 200000; n++) {
        double a = ((double)(lcg(&s) % 2000001u) - 1000000.0) * (n & 1 ? 1.0 : 1e-6), hs, hc;
        omega_hostmath_sincos(a, &hs, &hc);
        double d = fabs(hs - sin(a)); if (d > maxd) maxd = d; d = fabs(hc - cos(a)); if (d > maxd) maxd = d;
    }
    CHECK(maxd < 1e-14, "hostmath sincos vs libm over |a| <= 1e6: max abs diff %.3g", maxd);
    double maxr = 0;
    for (unsigned n = 0; n < 200000; n++) {
        double b = 1e-6 + (double)(lcg(&s) % 100000000u) * 1e-1, e = ((double)(lcg(&s) % 20001u) - 10000.0) / 10000.0 * 4.0;
        double h = omega_hostmath_pow(b, e), l = pow(b, e), r = fabs(h - l) / l; if (r > maxr) maxr = r;
        h = omega_hostmath_exp(e * 20.0); l = exp(e * 20.0); r = fabs(h - l) / l; if (r > maxr) maxr = r;
        h = omega_hostmath_log(b); l = log(b); r = fabs(h - l) / (1.0 + fabs(l)); if (r > maxr) maxr = r;
    }
    CHECK(maxr < 1e-13, "hostmath pow/exp/log vs libm: max rel diff %.3g", maxr);
    printf("hostmath vs libm: sincos max abs %.3g, pow/exp/log max rel %.3g\n", maxd, maxr);
    CHECK(omega_hostmath_exp(0.0) == 1.0 && omega_hostmath_log(1.0) == 0.0 && omega_hostmath_pow(THETA, 0.0) == 1.0, "hostmath identities");
    CHECK(omega_hostmath_log(-1.0) != omega_hostmath_log(-1.0) && omega_hostmath_exp(-800.0) == 0.0, "hostmath log(-1) is NaN, exp(-800) is 0");
}

/* compares out (f32 [4096]) against the reference; returns the number of elements outside the tolerance and the worst ratio */
static size_t parity(const Case *c, const float *out, float out_scale, double *worst, uint16_t *k_new, uint16_t *v_new) {
    double *ref = malloc(4096 * sizeof(double));
    omega_decode_attn_reference(c->qkv, c->kc, c->vc, c->max_pos, c->pos, c->qn, c->kn, EPS, THETA, out_scale, ref, k_new, v_new);
    double tol = 3e-3 * c->vmax * out_scale + 1e-6 * out_scale; size_t bad = 0; *worst = 0;
    for (size_t i = 0; i < 4096; i++) { double d = fabs((double)out[i] - ref[i]); if (d / tol > *worst) *worst = d / tol; if (d > tol) bad++; }
    free(ref);
    return bad;
}
/* appended rows: v exact, k within one bf16 ulp on at most 4 elements per head */
static int rows_ok(const uint16_t *k_chip, const uint16_t *v_chip, const uint16_t *k_ref, const uint16_t *v_ref, char *why, size_t n) {
    unsigned kflips = 0;
    for (unsigned i = 0; i < 512; i++) {
        if (v_chip[i] != v_ref[i]) { snprintf(why, n, "v[%u] chip %04x ref %04x", i, v_chip[i], v_ref[i]); return 0; }
        if (k_chip[i] != k_ref[i]) { int d = (int)k_chip[i] - (int)k_ref[i]; if (d < -1 || d > 1) { snprintf(why, n, "k[%u] chip %04x ref %04x", i, k_chip[i], k_ref[i]); return 0; } kflips++; }
    }
    if (kflips > 16) { snprintf(why, n, "%u one-ulp k flips", kflips); return 0; }
    snprintf(why, n, "ok (%u one-ulp k flips)", kflips);
    return 1;
}

/* ------------------------------------------------------------- host only */
static void host_only(void) {
    CHECK(omega_dattn_bf16_rne(1.0f) == 0x3f80, "bf16 of 1.0");
    CHECK(omega_dattn_bf16_rne(1.00390625f) == 0x3f80, "bf16 rounds 1 + 2^-8 to even (down)");
    CHECK(omega_dattn_bf16_rne(1.01171875f) == 0x3f82, "bf16 rounds 1 + 3 * 2^-8 to even (up)");
    CHECK(omega_dattn_bf16_rne(-2.5f) == 0xc020, "bf16 of -2.5");
    float row[128]; omega_dattn_rope_row(0, THETA, row);
    int ok = 1; for (unsigned i = 0; i < 64; i++) ok &= row[i] == 1.0f && row[64 + i] == 0.0f;
    CHECK(ok, "rope row 0 is cos 1 sin 0");
    omega_dattn_rope_row(100, THETA, row);
    CHECK(fabs((double)row[0] - cos(100.0)) < 1e-6 && fabs((double)row[64] - sin(100.0)) < 1e-6, "rope row 100 dim 0 is cos/sin(100)");
    CHECK(fabs((double)row[63] - cos(100.0 * pow(THETA, -126.0 / 128.0))) < 1e-6, "rope row 100 dim 63 uses theta^(-126/128)");
    host_math_vs_libm();
    /* reference sanity: context 1 returns v_new (rounded), whatever q and k */
    { Case c; case_make(&c, 256, 0, 11, 1.0f); double *ref = malloc(4096 * sizeof(double)); uint16_t kn[512], vn[512];
      omega_decode_attn_reference(c.qkv, c.kc, c.vc, 256, 0, c.qn, c.kn, EPS, THETA, 1.0f, ref, kn, vn);
      int good = 1; for (unsigned h = 0; h < 32 && good; h++) for (unsigned i = 0; i < 128; i++) if (ref[h * 128 + i] != (double)omega_dattn_bf16_f32(vn[(h / 8) * 128 + i])) { good = 0; break; }
      CHECK(good, "reference: context 1 returns the rounded v_new of the head's KV group");
      /* a dominant key: history row 3 equal to the roped-normed q direction of head 0 scaled up -> output ~ v row 3 */
      free(ref); case_free(&c); }
    { Case c; case_make(&c, 256, 8, 12, 1.0f);
      double *ref = malloc(4096 * sizeof(double)); uint16_t kn[512], vn[512];
      omega_decode_attn_reference(c.qkv, c.kc, c.vc, 256, 8, c.qn, c.kn, EPS, THETA, 1.0f, ref, kn, vn);
      /* k_new is unit-rms-ish: rms of the k_norm-weighted row around 1 */
      double ss = 0; for (unsigned i = 0; i < 128; i++) { double v = omega_dattn_bf16_f32(kn[i]); ss += v * v; }
      CHECK(ss / 128.0 > 0.6 && ss / 128.0 < 1.5, "reference: k_new has rms about 1 after RMSNorm (%.3f)", ss / 128.0);
      double os = 0; for (unsigned i = 0; i < 4096; i++) os += fabs(ref[i]);
      CHECK(os / 4096.0 < 0.6, "reference: a 9-token softmax average of U(-1,1) rows is small (%.3f)", os / 4096.0);
      free(ref); case_free(&c); }
    /* codegen budget */
    for (uint32_t d = 1; d <= 4; d++) for (int m = 0; m <= 4; m++) {
        OmegaDecodeAttnSpec s = { d, m }; uint32_t n = 0, gp = 0;
        int rc = omega_decode_attn_prog_stats(&s, &n, &gp);
        CHECK(rc == 0 && n <= 1024 && gp <= 128, "codegen depth %u mutant %d: rc %d insns %u gprs %u", d, m, rc, n, gp);
        if (rc == 0 && m == 0) printf("codegen depth %u: %u instructions, %u registers (declared %u, 256 threads -> %u of 65536)\n", d, n, gp, ((gp + 8) / 8) * 8, 256 * ((gp + 8) / 8) * 8);
    }
    { OmegaDecodeAttnSpec s = { 0, 0 }; CHECK(omega_decode_attn_prog_stats(&s, NULL, NULL) != 0, "depth 0 refused");
      s.depth = 5; CHECK(omega_decode_attn_prog_stats(&s, NULL, NULL) != 0, "depth 5 refused");
      s.depth = 1; s.mutant = 5; CHECK(omega_decode_attn_prog_stats(&s, NULL, NULL) != 0, "mutant 5 refused"); }
    CHECK(omega_gpu_decode_attn_cache_alloc(0, 256, THETA, NULL) == OMEGA_GPU_DATTN_BAD_ARGS, "NULL out refused");
    { OmegaGpuDecodeAttnCache *c = (OmegaGpuDecodeAttnCache *)1;
      CHECK(omega_gpu_decode_attn_cache_alloc(1, 100, THETA, &c) == OMEGA_GPU_DATTN_BAD_ARGS && c == NULL, "max_pos not a multiple of 256 refused before any device work"); }
    CHECK(OMEGA_DATTN_SHARED_BYTES <= 16384u, "shared memory within the QMD cap (%u)", OMEGA_DATTN_SHARED_BYTES);
}

/* ------------------------------------------------------------- simulator */
static uint64_t g_sim_issues;
static int sim_point(uint32_t max_pos, uint32_t pos, uint32_t depth, int mutant, uint32_t per_split, float qscale, uint32_t seed, char *msg, size_t n, double *worst, size_t *bad, char *rows, size_t rn) {
    static BlackwellIRProgram prog;
    OmegaDecodeAttnSpec spec = { depth, mutant };
    if (omega_blackwell_codegen_decode_attn_prog(&spec, &prog) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    Case c; case_make(&c, max_pos, pos, seed, qscale);
    const uint32_t splits = (pos + per_split) / per_split;
    size_t plane = (size_t)4 * max_pos * 256 + OMEGA_DATTN_PLANE_SLACK;
    uint8_t *kv = aligned_alloc(16, 2 * plane); memset(kv, 0, 2 * plane);
    memcpy(kv, c.kc, (size_t)4 * max_pos * 256); memcpy(kv + plane, c.vc, (size_t)4 * max_pos * 256);
    float *rope = aligned_alloc(16, (size_t)max_pos * 512); for (uint32_t p = 0; p < max_pos; p++) omega_dattn_rope_row(p, THETA, rope + (size_t)p * 128);
    float *norms = aligned_alloc(16, 1024); memcpy(norms, c.qn, 512); memcpy(norms + 128, c.kn, 512);
    float *qkv = aligned_alloc(16, sizeof c.qkv); memcpy(qkv, c.qkv, sizeof c.qkv);
    float *out = aligned_alloc(16, 4096 * 4), *copy = malloc(4096 * 4);
    float *part = aligned_alloc(16, (size_t)4 * splits * OMEGA_DATTN_PARTIAL_WORDS * 4);
    uint32_t *cnt = aligned_alloc(16, 16);
    int err = 0;
    for (int order = 0; order < 2 && !err; order++) {
        WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = 256; s.order = order;
        static uint8_t smem[OMEGA_DATTN_SHARED_BYTES]; s.smem = smem; s.smem_bytes = OMEGA_DATTN_SHARED_BYTES;
        memset(cnt, 0, 16); for (unsigned i = 0; i < 4096; i++) out[i] = -999.0f;
        memcpy(kv, c.kc, (size_t)4 * max_pos * 256); memcpy(kv + plane, c.vc, (size_t)4 * max_pos * 256); /* fresh history (the kernel appends) */
        uint64_t A[9] = { (uintptr_t)qkv, (uintptr_t)kv, (uintptr_t)(kv + plane), (uintptr_t)rope, (uintptr_t)norms, (uintptr_t)(norms + 128), (uintptr_t)out, (uintptr_t)part, (uintptr_t)cnt };
        for (unsigned i = 0; i < 9; i++) { s.cbank[(0x380 + 8 * i) / 4] = (uint32_t)A[i]; s.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
        float fe = EPS, fo = 1.0f, fq = (float)((1.0 / sqrt(128.0)) * 1.4426950408889634);
        memcpy(&s.cbank[OMEGA_DATTN_ARG_EPS / 4], &fe, 4); memcpy(&s.cbank[OMEGA_DATTN_ARG_OUT_SCALE / 4], &fo, 4); memcpy(&s.cbank[OMEGA_DATTN_ARG_QK_SCALE / 4], &fq, 4);
        s.cbank[OMEGA_DATTN_ARG_POS / 4] = pos; s.cbank[OMEGA_DATTN_ARG_PER_SPLIT / 4] = per_split; s.cbank[OMEGA_DATTN_ARG_SPLITS / 4] = splits; s.cbank[OMEGA_DATTN_ARG_KV_STRIDE / 4] = max_pos * 256;
        s.nranges = 7;
        s.ranges[0].lo = (uintptr_t)qkv; s.ranges[0].hi = (uintptr_t)qkv + sizeof c.qkv;
        s.ranges[1].lo = (uintptr_t)kv; s.ranges[1].hi = (uintptr_t)kv + 2 * plane;
        s.ranges[2].lo = (uintptr_t)rope; s.ranges[2].hi = (uintptr_t)rope + (size_t)max_pos * 512;
        s.ranges[3].lo = (uintptr_t)norms; s.ranges[3].hi = (uintptr_t)norms + 1024;
        s.ranges[4].lo = (uintptr_t)out; s.ranges[4].hi = (uintptr_t)out + 4096 * 4;
        s.ranges[5].lo = (uintptr_t)part; s.ranges[5].hi = (uintptr_t)part + (size_t)4 * splits * OMEGA_DATTN_PARTIAL_WORDS * 4;
        s.ranges[6].lo = (uintptr_t)cnt; s.ranges[6].hi = (uintptr_t)cnt + 16;
        err = ws_run(&s, splits, 4);
        if (err) { snprintf(msg, n, "order %d: %s (pc %d)", order, s.msg, s.err_pc); break; }
        g_sim_issues += s.warp_issues;
        size_t unwritten = 0; for (unsigned i = 0; i < 4096; i++) if (out[i] == -999.0f) unwritten++;
        if (unwritten) { err = -1; snprintf(msg, n, "%zu output words never written", unwritten); break; }
        if (cnt[0] | cnt[1] | cnt[2] | cnt[3]) { err = -1; snprintf(msg, n, "counters not reset: %u %u %u %u", cnt[0], cnt[1], cnt[2], cnt[3]); break; }
        if (order == 0) memcpy(copy, out, 4096 * 4);
        else if (memcmp(copy, out, 4096 * 4) != 0) { err = -1; snprintf(msg, n, "the fragment order changed the result"); }
    }
    if (!err) {
        uint16_t kn[512], vn[512];
        *bad = parity(&c, out, 1.0f, worst, kn, vn);
        uint16_t kchip[512], vchip[512];
        for (unsigned g = 0; g < 4; g++) { memcpy(kchip + g * 128, kv + ((size_t)g * max_pos + pos) * 256, 256); memcpy(vchip + g * 128, kv + plane + ((size_t)g * max_pos + pos) * 256, 256); }
        if (!rows_ok(kchip, vchip, kn, vn, rows, rn)) *bad += 1000;
        if (getenv("DATTN_DUMP")) { for (int dr = -1; dr <= 1; dr++) { if ((int)pos + dr < 0) continue; const uint16_t *r = (const uint16_t *)(kv + plane + ((size_t)0 * max_pos + pos + dr) * 256); printf("  v row %d:", (int)pos + dr); for (unsigned i = 0; i < 12; i++) printf(" %04x", r[i]); printf(" ... ref:"); for (unsigned i = 0; i < 12; i++) printf(" %04x", vn[i]); printf("\n"); }
            const uint16_t *r = (const uint16_t *)(kv + ((size_t)0 * max_pos + pos) * 256); printf("  k row %u:", pos); for (unsigned i = 0; i < 12; i++) printf(" %04x", r[i]); printf(" ... ref:"); for (unsigned i = 0; i < 12; i++) printf(" %04x", kn[i]); printf("\n"); }
        snprintf(msg, n, "bad=%zu worst=%.3f rows: %s", *bad, *worst, rows);
    }
    free(kv); free(rope); free(norms); free(qkv); free(out); free(copy); free(part); free(cnt); case_free(&c);
    return err;
}
static void sim_battery(void) {
    char msg[300], rows[120]; double worst; size_t bad;
    struct { uint32_t max_pos, pos, depth, per; float qscale; const char *name; } pts[] = {
        { 256, 0, 1, 256, 1.0f, "context 1" },
        { 256, 6, 2, 256, 1.0f, "context 7" },
        { 256, 6, 4, 256, 4.0f, "context 7, sharp q (x4)" },
        { 256, 255, 4, 256, 1.0f, "context 256 (full chunk, 1 split)" },
        { 512, 256, 4, 256, 1.0f, "context 257 (2 splits, last holds one position)" },
        { 512, 300, 3, 256, 1.0f, "context 301 (2 splits)" },
        { 1024, 700, 4, 256, 1.0f, "context 701 (3 splits)" },
        { 1024, 700, 2, 512, 1.0f, "context 701 (per_split 512: 2 chunks per CTA)" },
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        int err = sim_point(pts[i].max_pos, pts[i].pos, pts[i].depth, 0, pts[i].per, pts[i].qscale, 100 + (uint32_t)i, msg, sizeof msg, &worst, &bad, rows, sizeof rows);
        CHECK(err == 0 && bad == 0, "sim %s depth %u: %s", pts[i].name, pts[i].depth, msg);
        printf("sim %-50s depth %u: %s %s\n", pts[i].name, pts[i].depth, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
    }
    for (int m = 1; m <= 4; m++) {
        int err = sim_point(512, 300, 2, m, 256, 1.0f, 200, msg, sizeof msg, &worst, &bad, rows, sizeof rows);
        int caught = err != 0 || bad > 0;
        CHECK(caught, "sim mutant %d NOT caught (%s)", m, msg);
        printf("sim mutant %d at context 301: %s (%s)\n", m, caught ? "caught" : "MISSED", msg);
    }
    printf("sim: %" PRIu64 " warp issues in total\n", g_sim_issues);
}

/* ------------------------------------------------------------- chip */
static int g_evict = 1; static const char *g_only;
static void evict_host_caches(void) {
    static uint8_t *buf; const size_t sz = 512u << 20; static uint8_t pass;
    if (!buf) buf = malloc(sz);
    if (!buf) return;
    memset(buf, ++pass, sz);
    volatile uint64_t sum = 0; for (size_t i = 0; i < sz; i += 64) sum += buf[i];
}
static int fill_history(OmegaGpuDecodeAttnCache *cache, const Case *c) {
    if (c->pos == 0) return OMEGA_GPU_DATTN_OK;
    for (unsigned g = 0; g < 4; g++) {
        int rc = omega_gpu_decode_attn_cache_write(cache, 0, g, 0, c->pos, c->kc + (size_t)g * c->max_pos * 128, c->vc + (size_t)g * c->max_pos * 128);
        if (rc != OMEGA_GPU_DATTN_OK) return rc;
    }
    return OMEGA_GPU_DATTN_OK;
}
static int chip_run(const char *out_path, uint32_t reps) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot write %s\n", out_path); return 1; }
    if (out) fprintf(out, "{\"receipt\":\"OMEGA_GPU_DECODE_ATTN_V1\",\"reps\":%u,\"evict\":%d,\"points\":[", reps, g_evict);
    const uint32_t MAXP = 32768;
    OmegaGpuDecodeAttnCache *cache = NULL;
    double t0 = now_ms();
    int rc = omega_gpu_decode_attn_cache_alloc(1, MAXP, THETA, &cache);
    printf("cache alloc (1 layer, %u positions): rc=%s %.1f ms (%s)\n", MAXP, omega_gpu_decode_attn_rc_name(rc), now_ms() - t0, rc ? omega_gpu_decode_attn_last_error() : "");
    CHECK(rc == OMEGA_GPU_DATTN_OK, "cache alloc");
    if (rc != OMEGA_GPU_DATTN_OK) { if (out) fprintf(out, "],\"verdict\":\"FAIL\"}\n"); return 1; }
    struct { uint32_t pos; float qscale; const char *name; } pts[] = {
        { 0, 1.0f, "context 1" }, { 6, 1.0f, "context 7" }, { 255, 1.0f, "context 256" }, { 300, 4.0f, "context 301 sharp q" },
        { 1023, 1.0f, "context 1024" }, { 4095, 1.0f, "context 4096" }, { 8191, 1.0f, "context 8192" }, { 32767, 1.0f, "context 32768" },
    };
    int first = 1;
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        if (g_only && !strstr(pts[i].name, g_only)) continue;
        Case c; case_make(&c, MAXP, pts[i].pos, 500 + (uint32_t)i, pts[i].qscale);
        CHECK(omega_gpu_decode_attn_set_norms(cache, 0, c.qn, c.kn) == OMEGA_GPU_DATTN_OK, "set norms");
        t0 = now_ms();
        rc = fill_history(cache, &c);
        double fill_ms = now_ms() - t0;
        float *y = malloc(4096 * 4), *y2 = malloc(4096 * 4);
        OmegaGpuDecodeAttnInfo info; memset(&info, 0, sizeof info);
        if (rc == OMEGA_GPU_DATTN_OK) rc = omega_gpu_decode_attn(cache, 0, c.pos, c.qkv, EPS, 1.0f, y, &info);
        size_t bad = 0; double worst = 1e9; uint16_t kn[512], vn[512], kchip[512], vchip[512]; char rows[120] = "n/a";
        if (rc == OMEGA_GPU_DATTN_OK) {
            bad = parity(&c, y, 1.0f, &worst, kn, vn);
            for (unsigned g = 0; g < 4; g++) if (omega_gpu_decode_attn_cache_read(cache, 0, g, c.pos, kchip + g * 128, vchip + g * 128) != OMEGA_GPU_DATTN_OK) bad += 1000;
            if (!rows_ok(kchip, vchip, kn, vn, rows, sizeof rows)) bad += 1000;
        }
        int ok = rc == OMEGA_GPU_DATTN_OK && bad == 0;
        int same = 0; OmegaGpuDecodeAttnInfo i2; memset(&i2, 0, sizeof i2);
        if (ok) { int r2 = omega_gpu_decode_attn(cache, 0, c.pos, c.qkv, EPS, 1.0f, y2, &i2); same = r2 == OMEGA_GPU_DATTN_OK && memcmp(y, y2, 4096 * 4) == 0 && i2.kernel_cache_hit; }
        CHECK(ok && same, "chip %s: rc=%s bad=%zu worst=%.3f rows=%s second-call-identical=%d (%s)", pts[i].name, omega_gpu_decode_attn_rc_name(rc), bad, worst, rows, same, omega_gpu_decode_attn_last_error());
        /* the out_scale path (2^64 for the FP8 x' row): bit-exact scaling is not required, a relative check is */
        if (ok) {
            float *ys = malloc(4096 * 4); int r3 = omega_gpu_decode_attn(cache, 0, c.pos, c.qkv, EPS, 18446744073709551616.0f, ys, NULL);
            int good = r3 == OMEGA_GPU_DATTN_OK; double wr = 0;
            for (unsigned k = 0; k < 4096 && good; k++) { double e = fabs((double)ys[k] / 18446744073709551616.0 - (double)y[k]); double r = e / (fabs((double)y[k]) + 1e-6); if (r > wr) wr = r; }
            CHECK(good && wr < 1e-5, "chip %s: out_scale 2^64 matches the unscaled output (worst rel %.2e)", pts[i].name, wr);
            free(ys);
        }
        uint64_t *el = calloc(reps, sizeof *el), *cl = calloc(reps, sizeof *cl); uint32_t done = 0;
        if (g_evict) evict_host_caches();
        for (uint32_t r = 0; r < reps && ok; r++) { OmegaGpuDecodeAttnInfo ti; if (omega_gpu_decode_attn(cache, 0, c.pos, c.qkv, EPS, 1.0f, y2, &ti) != OMEGA_GPU_DATTN_OK) break; el[r] = ti.elapsed_ns; cl[r] = ti.call_ns; done++; }
        uint64_t med_el = 0, med_cl = 0; double gbs = 0;
        if (done) { qsort(el, done, sizeof *el, cmp_u64); qsort(cl, done, sizeof *cl, cmp_u64); med_el = el[done / 2]; med_cl = cl[done / 2]; gbs = med_el ? (double)info.kv_bytes / (double)med_el : 0; }
        printf("chip %-24s rc=%-9s bad=%zu worst=%.3f rows=%s grid=%ux%u per_split=%u depth=%u kv_bytes=%" PRIu64 " fill=%.1fms chip_med=%.1fus (%.1f GB/s) call_med=%.1fus reps=%u\n",
               pts[i].name, omega_gpu_decode_attn_rc_name(rc), bad, worst, rows, info.grid_x, info.grid_y, info.per_split, info.depth, info.kv_bytes, fill_ms, med_el / 1e3, gbs, med_cl / 1e3, done);
        if (out) fprintf(out, "%s{\"name\":\"%s\",\"pos\":%u,\"rc\":\"%s\",\"bad\":%zu,\"worst_of_tol\":%.3f,\"rows\":\"%s\",\"second_call_identical\":%d,\"grid_x\":%u,\"grid_y\":%u,\"per_split\":%u,\"depth\":%u,"
                         "\"kv_bytes\":%" PRIu64 ",\"fill_ms\":%.1f,\"chip_median_ns\":%" PRIu64 ",\"gb_per_s\":%.1f,\"call_median_ns\":%" PRIu64 ",\"reps\":%u}",
                         first ? "" : ",", pts[i].name, c.pos, omega_gpu_decode_attn_rc_name(rc), bad, worst, rows, same, info.grid_x, info.grid_y, info.per_split, info.depth, info.kv_bytes, fill_ms, med_el, gbs, med_cl, done);
        first = 0;
        free(el); free(cl);
        if (ok && c.pos == 300) {
            for (int m = 1; m <= 4; m++) {
                omega_gpu_decode_attn_test_set_mutant(m);
                (void)fill_history(cache, &c); /* mutant 3 must not inherit the row the honest kernel wrote */
                uint16_t zk[128] = { 0 }, zv[128] = { 0 }; (void)omega_gpu_decode_attn_cache_write(cache, 0, 0, c.pos, 1, zk, zv);
                for (unsigned g = 1; g < 4; g++) (void)omega_gpu_decode_attn_cache_write(cache, 0, g, c.pos, 1, zk, zv);
                int r = omega_gpu_decode_attn(cache, 0, c.pos, c.qkv, EPS, 1.0f, y2, NULL);
                size_t mb = 0; double mw = 0; uint16_t k2[512], v2[512];
                if (r == OMEGA_GPU_DATTN_OK) { mb = parity(&c, y2, 1.0f, &mw, k2, v2); }
                int caught = r != OMEGA_GPU_DATTN_OK || mb > 0;
                if (m == 4 && info.grid_x == 1) { printf("chip %-24s mutant 4: inert (1 split, the combine is not exercised)\n", pts[i].name); omega_gpu_decode_attn_test_set_mutant(0); continue; }
                CHECK(caught, "chip mutant %d NOT caught", m);
                printf("chip %-24s mutant %d: %s (rc=%s bad=%zu worst=%.3f)\n", pts[i].name, m, caught ? "caught" : "MISSED", omega_gpu_decode_attn_rc_name(r), mb, mw);
                if (out) fprintf(out, ",{\"name\":\"%s mutant %d\",\"caught\":%d,\"rc\":\"%s\",\"bad\":%zu,\"worst_of_tol\":%.3f}", pts[i].name, m, caught, omega_gpu_decode_attn_rc_name(r), mb, mw);
                omega_gpu_decode_attn_test_set_mutant(0);
            }
        }
        free(y); free(y2); case_free(&c);
        if (omega_gpu_session_is_blocked()) { printf("session latched: stopping\n"); break; }
    }
    omega_gpu_decode_attn_cache_free(cache);
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"device_opens\":%u,\"verdict\":\"%s\"}\n", g_checks, g_failed, omega_gpu_session_open_count(), g_failed == 0 ? "OMEGA_GPU_DECODE_ATTN_PASS" : "OMEGA_GPU_DECODE_ATTN_FAIL"); fclose(out); }
    return 0;
}

/* --q-heads Q --kv-heads K --head-dim D (ternary engine T0, 2026-10-15): asks for an attention geometry. The kernel is
 * built for one geometry only (OMEGA_GPU_DATTN_Q_HEADS 32, OMEGA_GPU_DATTN_KV_HEADS 4, head dim 128, per-head q_norm /
 * k_norm before RoPE). Any other geometry is REFUSED here, with the reasons recorded (receipt verdict "REFUSED", exit 3),
 * instead of running a kernel that would read the wrong q / k / v slices. */
static int geometry_refusal(uint32_t qh, uint32_t kvh, uint32_t hd, const char *out_path) {
    if (qh == OMEGA_GPU_DATTN_Q_HEADS && kvh == OMEGA_GPU_DATTN_KV_HEADS && hd == 128u) return 0;
    char why[4][200]; int nw = 0;
    if (qh != OMEGA_GPU_DATTN_Q_HEADS || kvh != OMEGA_GPU_DATTN_KV_HEADS)
        snprintf(why[nw++], sizeof why[0], "head layout fixed at %u query / %u KV heads (%u per KV head); asked %u / %u (%u per KV head)",
                 OMEGA_GPU_DATTN_Q_HEADS, OMEGA_GPU_DATTN_KV_HEADS, OMEGA_GPU_DATTN_Q_HEADS / OMEGA_GPU_DATTN_KV_HEADS, qh, kvh, kvh ? qh / kvh : 0);
    snprintf(why[nw++], sizeof why[0], "fused q|k|v input fixed at %u words; the asked layout needs %u", OMEGA_GPU_DATTN_QKV_WORDS, (qh + 2u * kvh) * hd);
    snprintf(why[nw++], sizeof why[0], "per-head q_norm / k_norm is built in (Qwen3); a model without QK norm needs a kernel without it");
    if (hd != 128u) snprintf(why[nw++], sizeof why[0], "head dim fixed at 128; asked %u", hd);
    printf("REFUSED: decode attention at %u query / %u KV heads, head dim %u\n", qh, kvh, hd);
    for (int i = 0; i < nw; i++) printf("  reason %d: %s\n", i + 1, why[i]);
    printf("  needed: a decode attention kernel for this geometry (ternary engine design KT3)\n");
    if (out_path) {
        FILE *f = fopen(out_path, "w");
        if (f) {
            fprintf(f, "{\"test\":\"gpu_decode_attn_test\",\"q_heads\":%u,\"kv_heads\":%u,\"head_dim\":%u,\"kernel_q_heads\":%u,\"kernel_kv_heads\":%u,\"reasons\":[",
                    qh, kvh, hd, OMEGA_GPU_DATTN_Q_HEADS, OMEGA_GPU_DATTN_KV_HEADS);
            for (int i = 0; i < nw; i++) fprintf(f, "%s\"%s\"", i ? "," : "", why[i]);
            fprintf(f, "],\"needed\":\"KT3 decode attention for this geometry\",\"verdict\":\"REFUSED\"}\n");
            fclose(f);
        }
    }
    printf("gpu_decode_attn_test: REFUSED (geometry %u/%u/%u not built)\n", qh, kvh, hd);
    return 3;
}

int main(int argc, char **argv) {
    int host = 0, sim = 0; const char *out_path = NULL; uint32_t reps = 20, depth = 0, per = 0, spin = 2000; int sim_pos = -1; uint32_t sim_max = 256;
    uint32_t q_heads = OMEGA_GPU_DATTN_Q_HEADS, kv_heads = OMEGA_GPU_DATTN_KV_HEADS, head_dim = 128;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) depth = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--per-split") == 0 && i + 1 < argc) per = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--spin-us") == 0 && i + 1 < argc) spin = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-evict") == 0) g_evict = 0;
        else if (strcmp(argv[i], "--sim-pos") == 0 && i + 1 < argc) sim_pos = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sim-max") == 0 && i + 1 < argc) sim_max = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) g_only = argv[++i];
        else if (strcmp(argv[i], "--q-heads") == 0 && i + 1 < argc) q_heads = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv-heads") == 0 && i + 1 < argc) kv_heads = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) head_dim = (uint32_t)atoi(argv[++i]);
        else { printf("unknown argument %s\n", argv[i]); return 2; }
    }
    { int g = geometry_refusal(q_heads, kv_heads, head_dim, out_path); if (g) return g; }
    if (host) host_only();
    if (sim && sim_pos >= 0 && sim_max <= (uint32_t)sim_pos) sim_max = (uint32_t)sim_pos + 1u;
    if (sim && sim_pos >= 0) { char msg[300], rows[120]; double worst; size_t bad; int err = sim_point(sim_max, (uint32_t)sim_pos, depth ? depth : 2, 0, per ? per : 256, 1.0f, 777, msg, sizeof msg, &worst, &bad, rows, sizeof rows); printf("sim pos %d: %s %s\n", sim_pos, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg); return err || bad; }
    if (sim) sim_battery();
    if (!host && !sim) {
        if (depth) omega_gpu_decode_attn_set_depth(depth);
        if (per) omega_gpu_decode_attn_set_per_split(per);
        (void)omega_gpu_session_set_spin_us(spin);
        chip_run(out_path, reps ? reps : 1);
        omega_gpu_session_close();
    }
    printf("gpu_decode_attn_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
