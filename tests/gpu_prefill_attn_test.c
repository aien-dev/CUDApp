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
    for (uint32_t w = 4; w <= 8; w += 4) for (int m = 0; m <= 4; m++) {
        OmegaPrefillAttnSpec s = { w, m }; uint32_t n = 0, gp = 0, ph = 0;
        int rc = omega_prefill_attn_prog_stats(&s, &n, &gp, &ph);
        CHECK(rc == 0 && n <= 1024 && omega_bw_gpr_declared(gp) <= 128, "codegen warps %u mutant %d: rc %d insns %u gprs %u", w, m, rc, n, gp);
        static uint8_t code[1024 * 16]; uint32_t kg = 0; int np = -1;
        size_t sz = omega_gpu_prefill_attn_encode(&s, code, sizeof code, &kg, &np);
        CHECK(sz > 0 && np == (int)ph, "encode+patch warps %u mutant %d: size %zu patched %d of %u", w, m, sz, np, ph);
        if (rc == 0 && m == 0) printf("codegen warps %u: %u instructions, %u registers (declared %u, %u threads -> %u of 65536), %u placeholders patched\n",
                                      w, n, gp, omega_bw_gpr_declared(gp), 32 * w, 32 * w * omega_bw_gpr_declared(gp), ph);
    }
    { OmegaPrefillAttnSpec s = { 2, 0 }; CHECK(omega_prefill_attn_prog_stats(&s, NULL, NULL, NULL) != 0, "warps 2 refused");
      s.warps = 8; s.mutant = 5; CHECK(omega_prefill_attn_prog_stats(&s, NULL, NULL, NULL) != 0, "mutant 5 refused"); }
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
/* runs the API's chunk loop through the simulator on host buffers; kv: [K plane | V plane] holding the history */
static int sim_call(const BlackwellIRProgram *prog, uint32_t W, const Case *c, uint8_t *kv, size_t plane, const float *rope, const float *norms,
                    float out_scale, float *out, int order, char *msg, size_t n) {
    static float qbuf[OMEGA_PATTN_CHUNK * QKVW], obuf[OMEGA_PATTN_CHUNK * OUTW];
    static uint8_t smem[OMEGA_PATTN_SHARED_BYTES];
    const uint32_t chunks = (c->m + 63) / 64;
    for (uint32_t ch = 0; ch < chunks; ch++) {
        const uint32_t p0 = c->pos0 + ch * 64, mc = c->m - ch * 64 < 64 ? c->m - ch * 64 : 64;
        memset(qbuf, 0, sizeof qbuf);
        memcpy(qbuf, c->qkv + (size_t)ch * 64 * QKVW, (size_t)mc * QKVW * 4);
        for (size_t i = 0; i < (size_t)64 * OUTW; i++) obuf[i] = -999.0f;
        WsSim s; memset(&s, 0, sizeof s);
        s.p = prog; s.threads = 32 * W; s.order = order; s.smem = smem; s.smem_bytes = sizeof smem;
        uint64_t A[7] = { (uintptr_t)qbuf, (uintptr_t)kv, (uintptr_t)(kv + plane), (uintptr_t)rope, (uintptr_t)norms, (uintptr_t)(norms + 128), (uintptr_t)obuf };
        for (unsigned i = 0; i < 7; i++) { s.cbank[(0x380 + 8 * i) / 4] = (uint32_t)A[i]; s.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
        float fe = EPS, fo = out_scale, fq = (float)((1.0 / sqrt(128.0)) * 1.4426950408889634);
        s.cbank[OMEGA_PATTN_ARG_POS0 / 4] = p0; s.cbank[OMEGA_PATTN_ARG_MC / 4] = mc; s.cbank[OMEGA_PATTN_ARG_KV_STRIDE / 4] = c->max_pos * 256;
        memcpy(&s.cbank[OMEGA_PATTN_ARG_EPS / 4], &fe, 4); memcpy(&s.cbank[OMEGA_PATTN_ARG_OUT_SCALE / 4], &fo, 4); memcpy(&s.cbank[OMEGA_PATTN_ARG_QK_SCALE / 4], &fq, 4);
        s.nranges = 5;
        s.ranges[0].lo = (uintptr_t)qbuf; s.ranges[0].hi = (uintptr_t)qbuf + sizeof qbuf;
        s.ranges[1].lo = (uintptr_t)kv; s.ranges[1].hi = (uintptr_t)kv + 2 * plane;
        s.ranges[2].lo = (uintptr_t)rope; s.ranges[2].hi = (uintptr_t)rope + (size_t)c->max_pos * 512;
        s.ranges[3].lo = (uintptr_t)norms; s.ranges[3].hi = (uintptr_t)norms + 1024;
        s.ranges[4].lo = (uintptr_t)obuf; s.ranges[4].hi = (uintptr_t)obuf + sizeof obuf;
        int err = ws_run(&s, (mc + W - 1) / W, 4);
        if (err) { snprintf(msg, n, "chunk %u: %s (pc %d)", ch, s.msg, s.err_pc); return -1; }
        g_sim_issues += s.warp_issues;
        size_t unwritten = 0; for (size_t i = 0; i < (size_t)mc * OUTW; i++) if (obuf[i] == -999.0f) unwritten++;
        if (unwritten) { snprintf(msg, n, "chunk %u: %zu output words never written", ch, unwritten); return -1; }
        memcpy(out + (size_t)ch * 64 * OUTW, obuf, (size_t)mc * OUTW * 4);
    }
    return 0;
}
static int sim_point(uint32_t W, int mutant, uint32_t max_pos, uint32_t pos0, uint32_t m, float qscale, uint32_t seed, int orders,
                     char *msg, size_t n, double *worst, size_t *bad) {
    static BlackwellIRProgram prog;
    OmegaPrefillAttnSpec spec = { W, mutant };
    if (omega_blackwell_codegen_prefill_attn_prog(&spec, &prog) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
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
        err = sim_call(&prog, W, &c, kv, plane, rope, norms, 1.0f, out, order, msg, n);
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
    struct { uint32_t w, max_pos, pos0, m; float qs; int orders; const char *name; } pts[] = {
        { 8, 256, 0, 1, 1.0f, 2, "one token at position 0" },
        { 8, 256, 0, 5, 1.0f, 2, "5 tokens from 0 (partial CTA)" },
        { 4, 256, 37, 19, 1.0f, 2, "19 tokens from 37, warps 4" },
        { 8, 256, 37, 19, 4.0f, 1, "19 tokens from 37, sharp q (x4)" },
        { 8, 256, 100, 64, 1.0f, 1, "64 tokens from 100 (full chunk)" },
        { 8, 512, 0, 70, 1.0f, 1, "70 tokens from 0 (two chunks)" },
        { 4, 512, 200, 33, 1.0f, 1, "33 tokens from 200, warps 4" },
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        int err = sim_point(pts[i].w, 0, pts[i].max_pos, pts[i].pos0, pts[i].m, pts[i].qs, 100 + (uint32_t)i, pts[i].orders, msg, sizeof msg, &worst, &bad);
        CHECK(err == 0 && bad == 0, "sim %s: %s", pts[i].name, msg);
        printf("sim %-40s W=%u: %s %s\n", pts[i].name, pts[i].w, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
        fflush(stdout);
    }
    for (int m = 1; m <= 4; m++) {
        int err = sim_point(8, m, 256, 40, 24, 1.0f, 200, 1, msg, sizeof msg, &worst, &bad);
        int caught = err != 0 || bad > 0;
        CHECK(caught, "sim mutant %d NOT caught (%s)", m, msg);
        printf("sim mutant %d at 24 tokens from 40: %s (%s)\n", m, caught ? "caught" : "MISSED", msg);
    }
    printf("sim: %" PRIu64 " warp issues in total\n", g_sim_issues);
}

/* ------------------------------------------------------------- chip */
static FILE *g_json;
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
        for (int r = 0; r < reps; r++) { omega_gpu_prefill_attn(pc, 0, pos0, m, c.qkv, EPS, out_scale, out2, &info); t[r] = info.elapsed_ns; tc[r] = info.call_ns; }
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
    if (pc) omega_gpu_decode_attn_cache_free(pc);
    if (dc) omega_gpu_decode_attn_cache_free(dc);
    free(out); free(out2); free(dout); free(kc); free(vc); free(dk); free(dv); case_free(&c);
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
        if (pc) omega_gpu_decode_attn_cache_free(pc);
        free(out); free(kc); free(vc); case_free(&c);
    }
    omega_gpu_prefill_attn_test_set_mutant(0);
}

int main(int argc, char **argv) {
    int host = 0, sim = 0, reps = 9, quick = 0; const char *json = NULL; uint32_t warps = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) host = 1;
        else if (!strcmp(argv[i], "--sim")) sim = 1;
        else if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warps") && i + 1 < argc) warps = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) json = argv[++i];
        else { fprintf(stderr, "usage: %s [--host-only | --sim | [--quick] [--reps N] [--warps 4|8] [--out receipt.json]]\n", argv[0]); return 2; }
    }
    host_only();
    if (sim) sim_battery();
    if (!host && !sim) {
        if (json) { g_json = fopen(json, "w"); if (g_json) fprintf(g_json, "{\"test\": \"gpu_prefill_attn\", \"points\": [\n"); }
        if (warps) omega_gpu_prefill_attn_set_warps(warps);
        int ok = 1;
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
        chip_mutants();
        (void)ok;
        if (g_json) { fprintf(g_json, "  {}\n], \"checks\": %d, \"failed\": %d}\n", g_checks, g_failed); fclose(g_json); }
    }
    printf("gpu_prefill_attn_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
