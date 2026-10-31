/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * gpu_prefill_fattn256_test: the flash-style tensor-core prefill attention for the Qwen3.6-35B-A3B full-attention layers
 * (Linear DRA-85, omega_gpu_prefill_fattn256_api.h), on the decode attention's cache (omega_gpu_decode_attn256_api.h)
 * in the q36 graph's shape (16 q / 2 KV heads, q_per_cta 4) and strides (q|k|v rows 9216 f32 apart: the output-gate
 * words follow and must not be read; output rows ostride bytes apart).
 *   --host-only   codegen budgets of both kernels (instructions <= 1024, registers within the opt-in limit and the
 *                 256-thread register file, attend placeholders patched), refusals (spec, shape, prep_xstride, API
 *                 arguments), layout constants, reference sanity. No device.
 *   --sim         both launches through tests/bw_warp_sim_pa.h (no chip, no shared memory given to the attend: a shared
 *                 access faults): (pos0, m) points with m = 1, odd m, a 16-token CTA boundary, odd starts, history, a 256
 *                 boundary, full rope, warps past m; both fragment orders on the small points; --poison SEED starts
 *                 every register as junk. Gates: appended K/V rows bit-equal to m runs of the DECODE program on a twin
 *                 cache (the same simulator), outputs within the tolerance of a double reference, the padding words of
 *                 every output row untouched, prep mutants 1 and 3 and attend mutants 1 (no causal mask) and 2 (no
 *                 rescale) caught. The q' scratch starts as NaN so a slot the prep never wrote shows up.
 *   (default)     the chip: parity (every token up to 512, 8 sampled tokens above), rows bit-identical to m decode calls
 *                 on a twin cache and the next decode step bit-identical on both caches, second call identical, out_scale
 *                 2^20 exact, device entry == host entry, mutants caught, timing medians of 2048-token chunks at contexts
 *                 2k, 8k, 32k (and 128k with --max-pos 131072) and with --long the 8192-token chunks ending at 8k and 32k;
 *                 JSON receipt with --out. Chip runs belong to the DRA-85 slices (P-DRA-4 hive chip/DRA-85/).
 * Reference and tolerance as gpu_prefill_attn256_test.c: model dim order (zero-centred RMSNorm, RoPE pairs (j, j +
 * rotary/2), q.k / 16, causal softmax), q' rounded to bf16 as the prep does, its own bf16 K/V rows;
 * |out - ref| <= 5e-3 * max|v| * out_scale per element (P rounded to bf16 for the tensor cores, <= 2^-9 * max|v|).
 * The output is not bit-identical to the decode steps (another summation order); host and simulator passes are never
 * chip results.
 */
#include "omega_gpu_prefill_fattn256_api.h"
#include "omega_gpu_session.h"
#include "omega_blackwell_codegen_tc.h"
#include "omega_blackwell_codegen_pattn256.h"
#include "bw_warp_sim_pa.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "test_out_file.h"

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static const double THETA = 10000000.0; /* rope_theta */
static const float EPS = 1e-6f;
#define HD 256u
/* Qwen3.6-35B-A3B full-attention layer: 16 q / 2 KV heads, the q36 graph's cache (omega_gpu_q36_graph.h: 16, 2, 4) */
#define QH 16u
#define KVH 2u
#define QPC 4u
#define VY (QH / QPC)                 /* decode / prep CTA rows */
#define SPEC_SHAPE , .q_heads = QH, .kv_heads = KVH, .q_per_cta = QPC
#define ATT_SHAPE , .q_heads = QH, .kv_heads = KVH
#define QW ((QH + 2u * KVH) * HD)     /* 5120: q | k | v */
#define XSW 9216u                     /* f32 per q36 mixer row (OMEGA_GPU_Q36_ATT_MIX: q | k | v | output gate) */
#define XS (XSW * 4u)                 /* the q36 graph's xstride */
#define OW (QH * HD)                  /* 4096 */
#define QFP (QH * 1024u)              /* q' bytes per token pair */
#define QEND (QH * HD)
#define KEND (QEND + KVH * HD)
#define RB OMEGA_DATTN256_ROW_BYTES
static const float QKS = (float)((1.0 / 16.0) * 1.4426950408889634);
#define TOL_REL 5e-3

/* one layer's random state in MODEL order: m qkv rows, norm weights w (zero-centred), history rows 0..pos0-1 */
typedef struct {
    uint32_t max_pos, pos0, m, rotary;
    float *qkv;                 /* [m][XSW]: q | k | v, then the gate words the kernels must not read */
    float qn[HD], kn[HD];
    uint16_t *kc, *vc;          /* [4][max_pos][256] model order */
} Case;
static void case_make(Case *c, uint32_t max_pos, uint32_t pos0, uint32_t m, uint32_t rotary, uint32_t seed, float qscale) {
    uint32_t s = seed;
    c->max_pos = max_pos; c->pos0 = pos0; c->m = m; c->rotary = rotary;
    c->qkv = malloc((size_t)m * XS);
    for (size_t t = 0; t < m; t++) for (size_t i = 0; i < XSW; i++) c->qkv[t * XSW + i] = i < QW ? frand(&s, -1.0f, 1.0f) * (i < QEND ? qscale : 1.0f) : 1e30f;
    for (unsigned i = 0; i < HD; i++) { c->qn[i] = frand(&s, -0.2f, 0.2f); c->kn[i] = frand(&s, -0.2f, 0.2f); }
    size_t words = (size_t)KVH * max_pos * HD;
    c->kc = calloc(words, 2); c->vc = calloc(words, 2);
    for (unsigned g = 0; g < KVH; g++) for (uint32_t t = 0; t < pos0; t++) for (unsigned i = 0; i < HD; i++) {
        size_t o = ((size_t)g * max_pos + t) * HD + i;
        c->kc[o] = omega_dattn_bf16_rne(frand(&s, -1.0f, 1.0f)); c->vc[o] = omega_dattn_bf16_rne(frand(&s, -1.0f, 1.0f));
    }
}
static void case_free(Case *c) { free(c->qkv); free(c->kc); free(c->vc); }
/* the kernel's inputs: q and k heads permuted, v and the gate words as is; norms 1 + w permuted */
static void case_kernel_qkv(const Case *c, float *qkv_k) {
    for (size_t t = 0; t < c->m; t++) {
        const float *src = c->qkv + t * XSW; float *dst = qkv_k + t * XSW;
        for (unsigned h = 0; h < QH + KVH; h++) omega_dattn256_perm_vec(src + h * HD, dst + h * HD, c->rotary);
        memcpy(dst + KEND, src + KEND, (size_t)(XSW - KEND) * 4);
    }
}
static void case_kernel_norms(const Case *c, float *qn_k, float *kn_k) {
    float t[HD];
    for (unsigned i = 0; i < HD; i++) t[i] = 1.0f + c->qn[i];
    omega_dattn256_perm_vec(t, qn_k, c->rotary);
    for (unsigned i = 0; i < HD; i++) t[i] = 1.0f + c->kn[i];
    omega_dattn256_perm_vec(t, kn_k, c->rotary);
}
/* K history rows permuted, V as is, into a [K plane | V plane] buffer laid out like the cache (plane bytes apart) */
static void case_history_into(const Case *c, uint8_t *kv, size_t plane) {
    for (unsigned g = 0; g < KVH; g++) for (uint32_t t = 0; t < c->pos0; t++) {
        size_t o = ((size_t)g * c->max_pos + t);
        omega_dattn256_perm_vec_bf16(c->kc + o * HD, (uint16_t *)(kv + o * RB), c->rotary);
        memcpy(kv + plane + o * RB, c->vc + o * HD, RB);
    }
}

/* ---- double-precision oracle (test-owned: libm) in MODEL order ---- */
static void norm_rope(double *x, const float *w, float eps, uint32_t rotary, uint32_t pos) {
    double ss = 0; for (unsigned i = 0; i < HD; i++) ss += x[i] * x[i];
    double r = 1.0 / sqrt(ss / (double)HD + (double)eps);
    for (unsigned i = 0; i < HD; i++) x[i] = x[i] * r * (1.0 + (double)w[i]);
    const unsigned half = rotary / 2;
    for (unsigned j = 0; j < half; j++) {
        double inv = 1.0 / pow(THETA, (double)(2 * j) / (double)rotary), ang = (double)pos * inv, cs = cos(ang), sn = sin(ang);
        double x1 = x[j], x2 = x[j + half];
        x[j] = x1 * cs - x2 * sn; x[j + half] = x2 * cs + x1 * sn;
    }
}
/* the reference K/V rows (model order, bf16) of every chunk token: kn/vn [m][4][256] */
static void ref_rows(const Case *c, uint16_t *kn, uint16_t *vn) {
    for (size_t t = 0; t < c->m; t++) for (unsigned g = 0; g < KVH; g++) {
        const float *row = c->qkv + t * XSW; double k[HD];
        for (unsigned i = 0; i < HD; i++) k[i] = row[QEND + g * HD + i];
        norm_rope(k, c->kn, EPS, c->rotary, c->pos0 + (uint32_t)t);
        for (unsigned i = 0; i < HD; i++) { kn[(t * KVH + g) * HD + i] = omega_dattn_bf16_rne((float)k[i]); vn[(t * KVH + g) * HD + i] = omega_dattn_bf16_rne(row[KEND + g * HD + i]); }
    }
}
/* reference output of chunk token t: out[4096] (* out_scale) */
static void ref_token(const Case *c, const uint16_t *kn, const uint16_t *vn, size_t t, float out_scale, double *out) {
    const uint32_t pos = c->pos0 + (uint32_t)t;
    double *p = malloc(((size_t)pos + 1) * sizeof(double));
    for (unsigned h = 0; h < QH; h++) {
        const unsigned g = h / (QH / KVH);
        double q[HD]; float qb[HD];
        for (unsigned i = 0; i < HD; i++) q[i] = c->qkv[t * XSW + h * HD + i];
        norm_rope(q, c->qn, EPS, c->rotary, pos);
        for (unsigned i = 0; i < HD; i++) qb[i] = omega_dattn_bf16_f32(omega_dattn_bf16_rne((float)(q[i] * (double)QKS)));
        double mx = -1e300;
        for (uint32_t j = 0; j <= pos; j++) {
            const uint16_t *kr = j < c->pos0 ? c->kc + ((size_t)g * c->max_pos + j) * HD : kn + ((size_t)(j - c->pos0) * KVH + g) * HD;
            double s = 0; for (unsigned i = 0; i < HD; i++) s += (double)qb[i] * (double)omega_dattn_bf16_f32(kr[i]);
            p[j] = s; if (s > mx) mx = s;
        }
        double l = 0, o[HD]; memset(o, 0, sizeof o);
        for (uint32_t j = 0; j <= pos; j++) {
            const double w = exp2(p[j] - mx); l += w;
            const uint16_t *vr = j < c->pos0 ? c->vc + ((size_t)g * c->max_pos + j) * HD : vn + ((size_t)(j - c->pos0) * KVH + g) * HD;
            for (unsigned i = 0; i < HD; i++) o[i] += w * (double)omega_dattn_bf16_f32(vr[i]);
        }
        for (unsigned i = 0; i < HD; i++) out[h * HD + i] = o[i] / l * (double)out_scale;
    }
    free(p);
}
/* the tokens parity checks: all up to 512, else first 2, last 2 and 4 spread */
static size_t parity_tokens(uint32_t m, uint32_t *tok) {
    if (m <= 512) { for (uint32_t t = 0; t < m; t++) tok[t] = t; return m; }
    const uint32_t pick[8] = { 0, 1, m / 5, (2 * m) / 5, (3 * m) / 5, (4 * m) / 5, m - 2, m - 1 };
    memcpy(tok, pick, sizeof pick);
    return 8;
}
/* out: rows osw f32 apart */
static size_t parity(const Case *c, const uint16_t *kn, const uint16_t *vn, const float *out, size_t osw, float out_scale, double *worst) {
    uint32_t *tok = malloc((size_t)(c->m > 8 ? c->m : 8) * 4); size_t nt = parity_tokens(c->m, tok);
    double *ref = malloc(OW * sizeof(double)), tol = TOL_REL * out_scale + 1e-6 * out_scale; size_t bad = 0; *worst = 0;
    for (size_t k = 0; k < nt; k++) {
        ref_token(c, kn, vn, tok[k], out_scale, ref);
        for (size_t i = 0; i < OW; i++) { double d = fabs((double)out[(size_t)tok[k] * osw + i] - ref[i]); if (!(d <= tol)) bad++; if (d / tol > *worst || d != d) *worst = d != d ? INFINITY : d / tol; }
    }
    free(ref); free(tok);
    return bad;
}
/* the rows the kernel appended (K permuted, V model order; [4][256] per token at kchip/vchip + t * 1024 words) against
 * the reference rows: v exact, k as below */
static float bf16f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
/* K rows: chip == ref, or one bf16 ulp apart (at most 32 per token), or, for a value left small by the rotation's
 * cancellation (x1 cos - x2 sin), within 8 f32 roundings of the head row's largest |k| (8 * 2^-24 * rowmax): the
 * chip rotates in f32, the oracle in double, and a 3-ulp gap at 7e-6 in a row of O(1) values is f32 rounding, not a
 * kernel error (seen at 30720+2048 and 129024+2048 on 2026-10-18; the rows were bit-equal to the decode program's). */
static int rows_ok(const Case *c, const uint16_t *kchip, const uint16_t *vchip, const uint16_t *kn, const uint16_t *vn, char *why, size_t n) {
    unsigned flips = 0, worst_tok = 0, small = 0;
    for (size_t t = 0; t < c->m; t++) {
        unsigned tf = 0;
        for (unsigned g = 0; g < KVH; g++) {
            float rowmax = 0;
            for (unsigned i = 0; i < HD; i++) { float a = fabsf(bf16f(kn[(t * KVH + g) * HD + i])); if (a > rowmax) rowmax = a; }
            for (unsigned i = 0; i < HD; i++) {
                size_t ci = (t * KVH + g) * HD + i, ri = (t * KVH + g) * HD + omega_dattn256_perm_slot(i, c->rotary);
                if (vchip[ci] != vn[ci]) { snprintf(why, n, "token %zu v[%u][%u] chip %04x ref %04x", t, g, i, vchip[ci], vn[ci]); return 0; }
                if (kchip[ci] == kn[ri]) continue;
                int d = (int)kchip[ci] - (int)kn[ri];
                if (d >= -1 && d <= 1) { tf++; continue; }
                if (fabsf(bf16f(kchip[ci]) - bf16f(kn[ri])) <= 8.0f * 0x1p-24f * rowmax) { small++; continue; }
                snprintf(why, n, "token %zu k[%u] slot %u chip %04x ref %04x (row max %.3g)", t, g, i, kchip[ci], kn[ri], (double)rowmax);
                return 0;
            }
        }
        if (tf > 32) { snprintf(why, n, "token %zu: %u one-ulp k flips", t, tf); return 0; }
        if (tf > worst_tok) worst_tok = tf;
        flips += tf;
    }
    snprintf(why, n, "ok (%u one-ulp k flips, worst token %u; %u small-value k gaps within 8 f32 roundings of the row max)", flips, worst_tok, small);
    return 1;
}
/* ------------------------------------------------------------- host only */
static void host_only(void) {
    uint32_t att_ph = 0;
    for (int m = 0; m <= 2; m++) {
        OmegaPrefillFattn256Spec s = { .mutant = m ATT_SHAPE }; uint32_t n = 0, gp = 0, ph = 0;
        int rc = omega_prefill_fattn256_prog_stats(&s, &n, &gp, &ph);
        const uint32_t dec = omega_bw_gpr_declared(gp);
        CHECK(rc == 0 && n <= BW_MAX_IR_INSNS && gp <= BW_PHYS_GPR_HW_LIMIT && 256u * dec <= 65536u && ph > 0,
              "attend codegen mutant %d: rc %d insns %u gprs %u (declared %u) placeholders %u", m, rc, n, gp, dec, ph);
        if (rc == 0 && m == 0) { att_ph = ph; printf("attend: %u instructions, %u registers (declared %u, 256 threads -> %u of 65536), %u placeholders\n", n, gp, dec, 256 * dec, ph); }
    }
    { OmegaPrefillFattn256Spec s = { .mutant = 3 ATT_SHAPE }; CHECK(omega_prefill_fattn256_prog_stats(&s, NULL, NULL, NULL) != 0, "attend mutant 3 refused");
      s.mutant = -1; CHECK(omega_prefill_fattn256_prog_stats(&s, NULL, NULL, NULL) != 0, "attend mutant -1 refused");
      OmegaPrefillFattn256Spec b1 = { .q_heads = 16, .kv_heads = 3 }, b2 = { .q_heads = 18, .kv_heads = 2 }, ok27 = { .q_heads = 24, .kv_heads = 4 }, dflt = { 0, 0, 0 };
      CHECK(!omega_fattn256_spec_ok(&b1) && !omega_fattn256_spec_ok(&b2), "attend shapes 16/3 (not a multiple) and 18/2 (group 9 > 8) refused");
      OmegaFattn256Shape sh; uint32_t n27 = 0, n36 = 0;
      CHECK(omega_fattn256_spec_ok(&ok27) && omega_fattn256_shape(&dflt, &sh) == 0 && sh.qh == 16 && sh.kvh == 2 && sh.group == 8 && sh.out_words == OW && sh.qf_pair_bytes == QFP,
            "attend default shape is the q36 16 / 2 (group 8, %u out words, %u q' bytes per pair); the 27B 24 / 4 builds", OW, QFP);
      CHECK(omega_prefill_fattn256_prog_stats(&ok27, &n27, NULL, NULL) == 0 && omega_prefill_fattn256_prog_stats(&dflt, &n36, NULL, NULL) == 0 && n27 > 0 && n36 > 0, "27B and default shapes build"); }
    static const int pm[3] = { 0, 1, 3 };
    for (unsigned k = 0; k < 3; k++) {
        OmegaDecodeAttn256Spec s = { .depth = OMEGA_GPU_DATTN256_DEPTH, .mutant = pm[k], .prep = 1, .prep_xstride = XS SPEC_SHAPE }; uint32_t n = 0, gp = 0;
        int rc = omega_decode_attn256_prog_stats(&s, &n, &gp);
        CHECK(rc == 0 && n <= BW_MAX_IR_INSNS && gp <= BW_PHYS_GPR_MAX - 2, "prep (xstride %u) codegen mutant %d: rc %d insns %u gprs %u", XS, pm[k], rc, n, gp);
        if (rc == 0 && k == 0) printf("prep (xstride %u): %u instructions, %u registers\n", XS, n, gp);
    }
    { OmegaDecodeAttn256Spec s = { .depth = OMEGA_GPU_DATTN256_DEPTH, .prep = 1, .prep_xstride = QW * 4u - 16u SPEC_SHAPE };
      CHECK(omega_decode_attn256_prog_stats(&s, NULL, NULL) != 0, "prep_xstride below a q|k|v row refused");
      s.prep_xstride = XS + 4u; CHECK(omega_decode_attn256_prog_stats(&s, NULL, NULL) != 0, "prep_xstride off 16 bytes refused");
      s.prep_xstride = XS; s.prep = 0; CHECK(omega_decode_attn256_prog_stats(&s, NULL, NULL) != 0, "prep_xstride without prep refused (decode unchanged)");
      s.prep = 1; s.multi = 1; CHECK(omega_decode_attn256_prog_stats(&s, NULL, NULL) != 0, "prep_xstride with multi refused (the multi path has its own stride)"); }
    static uint8_t code[1 << 17];
    for (unsigned k = 0; k < 2; k++) {
        uint32_t gp = 0; int np = -1; size_t n = omega_gpu_prefill_fattn256_encode(k, QH, KVH, QPC, XS, code, sizeof code, &gp, &np);
        CHECK(n > 0 && (k == 0 ? np == 0 : (np > 0 && (uint32_t)np == att_ph)), "encode %s: %zu bytes, gprs %u, %d placeholders patched (expect %u)", k ? "attend" : "prep", n, gp, np, k ? att_ph : 0u);
    }
    CHECK(omega_gpu_prefill_fattn256_encode(2, 0, 0, 0, 0, code, sizeof code, NULL, NULL) == 0, "encode kind 2 refused");
    CHECK(omega_gpu_prefill_fattn256_encode(1, 0, 0, 0, 0, code, 64, NULL, NULL) == 0, "encode into a too-small buffer refused");
    CHECK(omega_gpu_prefill_fattn256_encode(1, 18, 2, 0, 0, code, sizeof code, NULL, NULL) == 0, "encode of a refused shape returns 0");
    CHECK(OMEGA_FATTN256_SHARED_BYTES == 0u && OMEGA_FATTN256_ARGS_WORDS * 4u == OMEGA_FATTN256_ARG_OSTRIDE + 4u - 0x380u, "attend: no shared memory; %u argument words end at OSTRIDE", OMEGA_FATTN256_ARGS_WORDS);
    CHECK(OMEGA_PATTN256_QF_OFF_SHAPE(KVH, QH / KVH, 1, 0, 0, 0, 0) == QFP && OMEGA_PATTN256_QF_OFF_SHAPE(KVH, QH / KVH, 0, KVH - 1, QH / KVH - 1, 15, 3) + 16u == QFP, "QF layout: one token pair is %u bytes, densely packed", QFP);
    CHECK(omega_gpu_prefill_fattn256(NULL, 0, 0, 1, (const float *)code, XS, EPS, 1.0f, (float *)code, OW * 4u, NULL) == OMEGA_GPU_DATTN_BAD_ARGS, "NULL cache refused");
    CHECK(omega_gpu_prefill_fattn256_record(NULL, NULL, 0, 0, 1, 16, XS, EPS, 1.0f, 16, OW * 4u) == OMEGA_GPU_DATTN_BAD_ARGS, "record without a command list refused");
    CHECK(omega_gpu_prefill_fattn256_reserve_for(NULL, 16, XS) == OMEGA_GPU_DATTN_BAD_ARGS, "reserve without a cache refused before any device work");
    /* reference sanity: one token at position 0 returns its own rounded v */
    { Case c; case_make(&c, 256, 0, 1, 64, 11, 1.0f); uint16_t kn[KVH * HD], vn[KVH * HD]; double ref[OW];
      ref_rows(&c, kn, vn); ref_token(&c, kn, vn, 0, 1.0f, ref);
      int good = 1; for (unsigned h = 0; h < QH; h++) for (unsigned i = 0; i < HD; i++) good &= fabs(ref[h * HD + i] - (double)omega_dattn_bf16_f32(vn[(h / (QH / KVH)) * HD + i])) < 1e-12;
      CHECK(good, "reference: context 1 returns the rounded v of the head's KV group");
      case_free(&c); }
}
/* ------------------------------------------------------------- simulator */
static uint64_t g_sim_issues;
static uint32_t g_poison;   /* --poison SEED: registers start as junk in every simulated CTA (tests/bw_warp_sim_pa.h reg_poison) */
typedef struct {
    const Case *c;
    uint8_t *kv; size_t plane;
    float *rope, *norms, *qkv_k;
} SimBufs;
static void cb64(WsSim *s, unsigned off, const void *p) { uint64_t a = (uintptr_t)p; s->cbank[off / 4] = (uint32_t)a; s->cbank[off / 4 + 1] = (uint32_t)(a >> 32); }
static void cbf(WsSim *s, unsigned off, float f) { memcpy(&s->cbank[off / 4], &f, 4); }
/* the decode program (prep 0 or 1) at position pos (prep: tokens pos .. pos+grid_x-1, qkv rows from qkv_k) */
static int sim_decode_launch(const BlackwellIRProgram *prog, const SimBufs *b, uint32_t pos, uint32_t grid_x, const float *qkv_k, size_t qkv_bytes,
                             void *out, size_t out_bytes, float *part, size_t part_bytes, uint32_t *cnt, uint32_t splits, int order, char *msg, size_t n) {
    static uint8_t smem[OMEGA_DATTN256_SHARED_BYTES];
    WsSim s; memset(&s, 0, sizeof s);
    s.p = prog; s.threads = 256; s.order = order; s.reg_poison = g_poison; s.smem = smem; s.smem_bytes = sizeof smem;
    cb64(&s, OMEGA_DATTN256_ARG_QKV, qkv_k); cb64(&s, OMEGA_DATTN256_ARG_KC, b->kv); cb64(&s, OMEGA_DATTN256_ARG_VC, b->kv + b->plane);
    cb64(&s, OMEGA_DATTN256_ARG_ROPE, b->rope); cb64(&s, OMEGA_DATTN256_ARG_QN, b->norms); cb64(&s, OMEGA_DATTN256_ARG_KN, b->norms + HD);
    cb64(&s, OMEGA_DATTN256_ARG_OUT, out); cb64(&s, OMEGA_DATTN256_ARG_PART, part); cb64(&s, OMEGA_DATTN256_ARG_CNT, cnt);
    s.cbank[OMEGA_DATTN256_ARG_POS / 4] = pos; s.cbank[OMEGA_DATTN256_ARG_PER_SPLIT / 4] = 256; s.cbank[OMEGA_DATTN256_ARG_SPLITS / 4] = splits;
    s.cbank[OMEGA_DATTN256_ARG_KV_STRIDE / 4] = b->c->max_pos * RB;
    cbf(&s, OMEGA_DATTN256_ARG_EPS, EPS); cbf(&s, OMEGA_DATTN256_ARG_OUT_SCALE, 1.0f); cbf(&s, OMEGA_DATTN256_ARG_QK_SCALE, QKS);
    s.nranges = 5;
    s.ranges[0].lo = (uintptr_t)qkv_k; s.ranges[0].hi = (uintptr_t)qkv_k + qkv_bytes;
    s.ranges[1].lo = (uintptr_t)b->kv; s.ranges[1].hi = (uintptr_t)b->kv + 2 * b->plane;
    s.ranges[2].lo = (uintptr_t)b->rope; s.ranges[2].hi = (uintptr_t)b->rope + (size_t)b->c->max_pos * OMEGA_DATTN256_ROPE_ROW_BYTES;
    s.ranges[3].lo = (uintptr_t)b->norms; s.ranges[3].hi = (uintptr_t)b->norms + 2048;
    s.ranges[4].lo = (uintptr_t)out; s.ranges[4].hi = (uintptr_t)out + out_bytes;
    if (part) {   /* prep never touches the partials or counters: no range, so a stray access faults */
        s.ranges[5].lo = (uintptr_t)part; s.ranges[5].hi = (uintptr_t)part + part_bytes;
        s.ranges[6].lo = (uintptr_t)cnt; s.ranges[6].hi = (uintptr_t)cnt + 16; s.nranges = 7;
    }
    int err = ws_run(&s, grid_x, VY);
    if (err) { snprintf(msg, n, "%s launch at %u: %s (pc %d)", part ? "decode" : "prep", pos, s.msg, s.err_pc); return -1; }
    g_sim_issues += s.warp_issues;
    return 0;
}
/* the fattn attend: grid (ceil(m / 16), KVH), no shared memory (smem NULL: any shared access faults) */
static int sim_attend_launch(const BlackwellIRProgram *prog, const SimBufs *b, const uint8_t *qf, size_t qf_bytes, float *out, size_t osw, int order, char *msg, size_t n) {
    const Case *c = b->c;
    WsSim s; memset(&s, 0, sizeof s);
    s.p = prog; s.threads = OMEGA_FATTN256_THREADS; s.order = order; s.reg_poison = g_poison; s.smem = NULL; s.smem_bytes = OMEGA_FATTN256_SHARED_BYTES;
    cb64(&s, OMEGA_FATTN256_ARG_QF, qf); cb64(&s, OMEGA_FATTN256_ARG_KC, b->kv); cb64(&s, OMEGA_FATTN256_ARG_VC, b->kv + b->plane); cb64(&s, OMEGA_FATTN256_ARG_OUT, out);
    s.cbank[OMEGA_FATTN256_ARG_POS0 / 4] = c->pos0; s.cbank[OMEGA_FATTN256_ARG_MC / 4] = c->m; s.cbank[OMEGA_FATTN256_ARG_KV_STRIDE / 4] = c->max_pos * RB;
    s.cbank[OMEGA_FATTN256_ARG_OSTRIDE / 4] = (uint32_t)(osw * 4);
    cbf(&s, OMEGA_FATTN256_ARG_OUT_SCALE, 1.0f);
    s.nranges = 3;
    s.ranges[0].lo = (uintptr_t)qf; s.ranges[0].hi = (uintptr_t)qf + qf_bytes;
    s.ranges[1].lo = (uintptr_t)b->kv; s.ranges[1].hi = (uintptr_t)b->kv + 2 * b->plane;
    s.ranges[2].lo = (uintptr_t)out; s.ranges[2].hi = (uintptr_t)out + (size_t)c->m * osw * 4;
    int err = ws_run(&s, (c->m + OMEGA_FATTN256_TOKENS - 1) / OMEGA_FATTN256_TOKENS, KVH);
    if (err) { snprintf(msg, n, "attend: %s (pc %d)", s.msg, s.err_pc); return -1; }
    g_sim_issues += s.warp_issues;
    return 0;
}
/* output rows OSW_SIM f32 apart: 16 padding words per row that the attend must leave alone */
#define OSW_SIM (OW + 16u)
static int sim_point(uint32_t max_pos, uint32_t pos0, uint32_t m, uint32_t rotary, float qscale, uint32_t seed, int orders, int prep_mut, int att_mut,
                     char *msg, size_t n, double *worst, size_t *bad) {
    static BlackwellIRProgram pprog, aprog, dprog;
    OmegaDecodeAttn256Spec ps = { .depth = OMEGA_GPU_DATTN256_DEPTH, .mutant = prep_mut, .prep = 1, .prep_xstride = XS SPEC_SHAPE }, ds = { .depth = OMEGA_GPU_DATTN256_DEPTH SPEC_SHAPE };
    OmegaPrefillFattn256Spec as = { .mutant = att_mut ATT_SHAPE };
    if (omega_blackwell_codegen_decode_attn256_prog(&ps, &pprog) != 0 || omega_blackwell_codegen_prefill_fattn256_prog(&as, &aprog) != 0 ||
        omega_blackwell_codegen_decode_attn256_prog(&ds, &dprog) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    Case c; case_make(&c, max_pos, pos0, m, rotary, seed, qscale);
    SimBufs b = { .c = &c };
    b.plane = (size_t)KVH * max_pos * RB + OMEGA_DATTN256_PLANE_SLACK;
    b.kv = aligned_alloc(16, 2 * b.plane);
    uint8_t *twin = aligned_alloc(16, 2 * b.plane);
    b.rope = aligned_alloc(16, (size_t)max_pos * OMEGA_DATTN256_ROPE_ROW_BYTES);
    for (uint32_t p = 0; p < max_pos; p++) omega_dattn256_rope_row(p, THETA, rotary, b.rope + (size_t)p * 256);
    b.norms = aligned_alloc(16, 2048); case_kernel_norms(&c, b.norms, b.norms + HD);
    b.qkv_k = aligned_alloc(16, (size_t)m * XS); case_kernel_qkv(&c, b.qkv_k);
    const size_t qf_bytes = (size_t)((m + 1) / 2) * QFP, ob = (size_t)m * OSW_SIM * 4;
    uint8_t *qf = aligned_alloc(16, qf_bytes);
    float *out = aligned_alloc(16, ob), *copy = malloc(ob);
    int err = 0;
    for (int order = 0; order < orders && !err; order++) {
        memset(b.kv, 0, 2 * b.plane); case_history_into(&c, b.kv, b.plane);
        memset(qf, 0xff, qf_bytes);   /* NaN bf16 pairs: a slot the prep never wrote poisons its row */
        for (size_t i = 0; i < (size_t)m * OSW_SIM; i++) out[i] = -999.0f;
        err = sim_decode_launch(&pprog, &b, pos0, m, b.qkv_k, (size_t)m * XS, qf, qf_bytes, NULL, 0, NULL, 0, order, msg, n);
        if (!err) err = sim_attend_launch(&aprog, &b, qf, qf_bytes, out, OSW_SIM, order, msg, n);
        if (err) break;
        size_t unwritten = 0, pad = 0;
        for (size_t t = 0; t < m; t++) for (size_t i = 0; i < OSW_SIM; i++) { const int u = out[t * OSW_SIM + i] == -999.0f; if (i < OW) unwritten += u; else pad += !u; }
        if (unwritten || pad) { err = -1; snprintf(msg, n, "%zu output words never written, %zu padding words written", unwritten, pad); break; }
        if (order == 0) memcpy(copy, out, ob);
        else if (memcmp(copy, out, ob) != 0) { err = -1; snprintf(msg, n, "the fragment order changed the result"); }
    }
    if (!err) {
        /* twin: m decode steps of the decode program from the same history */
        memset(twin, 0, 2 * b.plane); case_history_into(&c, twin, b.plane);
        SimBufs tb = b; tb.kv = twin;
        float *dout = aligned_alloc(16, OW * 4);
        const uint32_t smax = (pos0 + m + 255) / 256;
        float *part = aligned_alloc(16, (size_t)VY * smax * OMEGA_DATTN256_PARTIAL_WORDS * 4);
        uint32_t *cnt = aligned_alloc(16, 16);
        for (uint32_t t = 0; t < m && !err; t++) {
            const uint32_t pos = pos0 + t, splits = (pos + 256) / 256;
            memset(cnt, 0, 16);
            err = sim_decode_launch(&dprog, &tb, pos, splits, b.qkv_k + (size_t)t * XSW, QW * 4, dout, OW * 4, part, (size_t)VY * splits * OMEGA_DATTN256_PARTIAL_WORDS * 4, cnt, splits, 0, msg, n);
        }
        free(dout); free(part); free(cnt);
        size_t rowdiff = 0;
        for (unsigned h = 0; h < 2 * KVH && !err; h++) {
            const size_t o = (h >= KVH ? b.plane : 0) + (size_t)(h % KVH) * max_pos * RB;
            for (uint32_t t = 0; t < pos0 + m; t++) rowdiff += memcmp(b.kv + o + (size_t)t * RB, twin + o + (size_t)t * RB, RB) != 0;
        }
        uint16_t *kn = malloc((size_t)m * KVH * HD * 2), *vn = malloc((size_t)m * KVH * HD * 2), *kch = malloc((size_t)m * KVH * HD * 2), *vch = malloc((size_t)m * KVH * HD * 2);
        ref_rows(&c, kn, vn);
        for (uint32_t t = 0; t < m; t++) for (unsigned g = 0; g < KVH; g++) {
            const size_t o = ((size_t)g * max_pos + pos0 + t) * RB;
            memcpy(kch + ((size_t)t * KVH + g) * HD, b.kv + o, RB); memcpy(vch + ((size_t)t * KVH + g) * HD, b.kv + b.plane + o, RB);
        }
        char why[160];
        if (!err) {
            int rok = rows_ok(&c, kch, vch, kn, vn, why, sizeof why);
            *bad = parity(&c, kn, vn, out, OSW_SIM, 1.0f, worst);
            if (rowdiff) *bad += 1000000;
            if (!rok) *bad += 2000000;
            snprintf(msg, n, "bad=%zu worst=%.3f rows vs decode twin: %zu differ; vs reference: %s", *bad, *worst, rowdiff, why);
        }
        free(kn); free(vn); free(kch); free(vch);
    }
    free(b.kv); free(twin); free(b.rope); free(b.norms); free(b.qkv_k); free(qf); free(out); free(copy); case_free(&c);
    return err;
}
static void sim_battery(void) {
    char msg[400]; double worst; size_t bad;
    struct { uint32_t max_pos, pos0, m, rot; float qs; int orders; const char *name; } pts[] = {
        { 256, 0, 1, 64, 1.0f, 2, "one token at position 0" },
        { 256, 0, 16, 64, 1.0f, 2, "16 tokens from 0 (one full CTA)" },
        { 256, 0, 17, 64, 1.0f, 2, "17 tokens from 0 (CTA boundary, odd m)" },
        { 256, 0, 37, 64, 1.0f, 2, "37 tokens from 0 (odd m, partial CTA)" },
        { 256, 37, 19, 64, 1.0f, 2, "19 tokens from 37 (odd start)" },
        { 256, 100, 19, 64, 4.0f, 1, "19 tokens from 100, sharp q (x4)" },
        { 512, 250, 40, 64, 1.0f, 1, "40 tokens from 250 (crosses 256)" },
        { 256, 0, 24, 256, 1.0f, 1, "24 tokens from 0, full rope (rotary 256)" },
        { 512, 300, 50, 64, 1.0f, 1, "50 tokens from 300 (history > 256, warps past m)" },
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        double t0 = now_ms();
        int err = sim_point(pts[i].max_pos, pts[i].pos0, pts[i].m, pts[i].rot, pts[i].qs, 100 + (uint32_t)i, pts[i].orders, 0, 0, msg, sizeof msg, &worst, &bad);
        CHECK(err == 0 && bad == 0, "sim %s: %s", pts[i].name, msg);
        printf("sim %-48s %s %s (%.1f s)\n", pts[i].name, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg, (now_ms() - t0) / 1e3);
        fflush(stdout);
    }
    static const struct { int pm, am; } mut[] = { { 1, 0 }, { 3, 0 }, { 0, 1 }, { 0, 2 } };
    for (size_t i = 0; i < sizeof mut / sizeof mut[0]; i++) {
        int err = sim_point(256, 40, 24, 64, 1.0f, 200, 1, mut[i].pm, mut[i].am, msg, sizeof msg, &worst, &bad);
        int caught = err != 0 || bad > 0;
        CHECK(caught, "sim %s mutant %d NOT caught (%s)", mut[i].pm ? "prep" : "attend", mut[i].pm ? mut[i].pm : mut[i].am, msg);
        printf("sim %s mutant %d at 24 tokens from 40: %s (%s)\n", mut[i].pm ? "prep" : "attend", mut[i].pm ? mut[i].pm : mut[i].am, caught ? "caught" : "MISSED", msg);
        fflush(stdout);
    }
    printf("sim: %" PRIu64 " warp issues in total\n", g_sim_issues);
}

/* ------------------------------------------------------------- chip */
static FILE *g_json;
static int g_json_first = 1;
static int cache_with_history(const Case *c, OmegaGpuDecodeAttn256Cache **out) {
    int rc = omega_gpu_decode_attn256_cache_alloc_shape(1, c->max_pos, THETA, c->rotary, QH, KVH, QPC, out);
    if (rc != OMEGA_GPU_DATTN_OK) return rc;
    float qn_k[HD], kn_k[HD]; case_kernel_norms(c, qn_k, kn_k);
    rc = omega_gpu_decode_attn256_set_norms(*out, 0, qn_k, kn_k);
    if (rc != OMEGA_GPU_DATTN_OK || c->pos0 == 0) return rc;
    uint16_t *kp = malloc((size_t)c->pos0 * RB);
    for (unsigned g = 0; g < KVH && rc == OMEGA_GPU_DATTN_OK; g++) {
        for (uint32_t t = 0; t < c->pos0; t++) omega_dattn256_perm_vec_bf16(c->kc + ((size_t)g * c->max_pos + t) * HD, kp + (size_t)t * HD, c->rotary);
        rc = omega_gpu_decode_attn256_cache_write(*out, 0, g, 0, c->pos0, kp, c->vc + (size_t)g * c->max_pos * HD);
    }
    free(kp);
    return rc;
}
/* the chunk's appended rows, [m][4][256] each (K permuted as stored) */
static int read_rows(OmegaGpuDecodeAttn256Cache *cc, const Case *c, uint16_t *k, uint16_t *v) {
    for (uint32_t t = 0; t < c->m; t++) for (unsigned g = 0; g < KVH; g++) {
        int rc = omega_gpu_decode_attn256_cache_read(cc, 0, g, c->pos0 + t, k + ((size_t)t * KVH + g) * HD, v + ((size_t)t * KVH + g) * HD);
        if (rc != OMEGA_GPU_DATTN_OK) return rc;
    }
    return OMEGA_GPU_DATTN_OK;
}
static uint64_t median_u64(uint64_t *v, int n) { qsort(v, (size_t)n, sizeof *v, cmp_u64); return v[n / 2]; }
/* osw: f32 between output rows (OW = the q36 graph's MOS) */
static int chip_point(uint32_t max_pos, uint32_t pos0, uint32_t m, float qscale, uint32_t seed, int twin, int reps, uint32_t osw) {
    printf("-- chip %u tokens from %u (max_pos %u, ostride %u)\n", m, pos0, max_pos, osw * 4); fflush(stdout);
    Case c; case_make(&c, max_pos, pos0, m, 64, seed, qscale);
    float *qkv_k = malloc((size_t)m * XS); case_kernel_qkv(&c, qkv_k);
    const size_t ob = (size_t)m * osw * 4;
    float *out = malloc(ob), *out2 = malloc(ob);
    const size_t rw = (size_t)m * KVH * HD;
    uint16_t *kn = malloc(rw * 2), *vn = malloc(rw * 2), *kch = malloc(rw * 2), *vch = malloc(rw * 2), *ktw = malloc(rw * 2), *vtw = malloc(rw * 2);
    OmegaGpuDecodeAttn256Cache *cc = NULL, *tc = NULL;
    OmegaGpuPrefillFattn256Info info;
    int ok = 1; char why[200]; double worst = 0; size_t bad = 0;
    int rc = cache_with_history(&c, &cc);
    CHECK(rc == OMEGA_GPU_DATTN_OK, "cache with history: %s %s", omega_gpu_decode_attn_rc_name(rc), rc ? omega_gpu_decode_attn_last_error() : "");
    if (rc != OMEGA_GPU_DATTN_OK) { ok = 0; goto done; }
    for (size_t i = 0; i < (size_t)m * osw; i++) out[i] = -999.0f;
    rc = omega_gpu_prefill_fattn256(cc, 0, pos0, m, qkv_k, XS, EPS, 1.0f, out, osw * 4, &info);
    CHECK(rc == OMEGA_GPU_DATTN_OK, "prefill %u+%u: %s %s", pos0, m, omega_gpu_decode_attn_rc_name(rc), rc ? omega_gpu_decode_attn_last_error() : "");
    if (rc != OMEGA_GPU_DATTN_OK) { ok = 0; goto done; }
    printf("   prep %u insns %u regs, attend %u insns %u regs; first call %.3f ms (prep %.3f, attend %.3f)\n", info.insns[0], info.gprs[0], info.insns[1], info.gprs[1],
           info.call_ns / 1e6, info.elapsed_ns[0] / 1e6, info.elapsed_ns[1] / 1e6);
    ref_rows(&c, kn, vn);
    rc = read_rows(cc, &c, kch, vch);
    int rok = rc == OMEGA_GPU_DATTN_OK && rows_ok(&c, kch, vch, kn, vn, why, sizeof why);
    CHECK(rok, "%u+%u: appended rows vs reference: %s", pos0, m, rc ? "read failed" : why);
    double t0 = now_ms();
    bad = parity(&c, kn, vn, out, osw, 1.0f, &worst);
    { size_t pad = 0; for (size_t t = 0; t < m; t++) for (size_t i = OW; i < osw; i++) pad += out[t * osw + i] != -999.0f;
      CHECK(pad == 0, "%u+%u: %zu output padding words written", pos0, m, pad); bad += pad; }
    CHECK(bad == 0, "%u+%u: parity: %zu elements outside the tolerance (worst ratio %.3f)", pos0, m, bad, worst);
    printf("   parity: %zu bad, worst ratio %.3f of the tolerance (%.1f s reference); rows: %s\n", bad, worst, (now_ms() - t0) / 1e3, rok ? why : "FAIL");
    ok &= rok && bad == 0;
    /* second call over the same positions: the same rows are rewritten, the same output */
    memcpy(out2, out, ob);
    rc = omega_gpu_prefill_fattn256(cc, 0, pos0, m, qkv_k, XS, EPS, 1.0f, out2, osw * 4, &info);
    CHECK(rc == OMEGA_GPU_DATTN_OK && memcmp(out, out2, ob) == 0 && info.kernel_cache_hits == 2, "%u+%u: second call bit-identical with both kernels cached (rc %d hits %u)", pos0, m, rc, info.kernel_cache_hits);
    /* out_scale 2^20: exact */
    rc = omega_gpu_prefill_fattn256(cc, 0, pos0, m, qkv_k, XS, EPS, 1048576.0f, out2, osw * 4, NULL);
    { size_t diff = 0; for (size_t t = 0; t < m; t++) for (size_t i = 0; i < OW; i++) diff += out2[t * osw + i] != out[t * osw + i] * 1048576.0f;
      CHECK(rc == OMEGA_GPU_DATTN_OK && diff == 0, "%u+%u: out_scale 2^20 is exact (%zu words differ)", pos0, m, diff); }
    /* device entry == host entry */
    {
        OmegaGpuMem q, o; memset(&q, 0, sizeof q); memset(&o, 0, sizeof o);
        omega_gpu_session_lock();
        int arc = omega_gpu_session_open() && omega_gpu_session_alloc((size_t)m * XS, &q) == 0 && omega_gpu_session_alloc(ob, &o) == 0 ? 0 : -1;
        if (arc == 0) { memcpy(q.cpu, qkv_k, (size_t)m * XS); float *od = o.cpu; for (size_t i = 0; i < (size_t)m * osw; i++) od[i] = -999.0f;
                        arc = omega_gpu_session_sync_to_gpu(&q) | omega_gpu_session_sync_to_gpu(&o); }
        omega_gpu_session_unlock();
        rc = arc == 0 ? omega_gpu_prefill_fattn256_dev(cc, 0, pos0, m, q.va, XS, EPS, 1.0f, o.va, osw * 4, NULL) : OMEGA_GPU_DATTN_CHIP_FAIL;
        omega_gpu_session_lock();
        int same = rc == OMEGA_GPU_DATTN_OK && omega_gpu_session_sync_to_cpu(&o) == 0 && memcmp(o.cpu, out, ob) == 0;
        if (q.cpu) omega_gpu_session_free(&q);
        if (o.cpu) omega_gpu_session_free(&o);
        omega_gpu_session_unlock();
        CHECK(same, "%u+%u: device entry bit-identical to the host entry (rc %s)", pos0, m, omega_gpu_decode_attn_rc_name(rc));
    }
    /* twin: m decode calls from the same history; rows bit-identical, then the next decode step on both caches */
    if (twin) {
        rc = cache_with_history(&c, &tc);
        float *dout = malloc(OW * 4);
        t0 = now_ms();
        for (uint32_t t = 0; t < m && rc == OMEGA_GPU_DATTN_OK; t++) rc = omega_gpu_decode_attn256(tc, 0, pos0 + t, qkv_k + (size_t)t * XSW, EPS, 1.0f, dout, NULL);
        const double dec_ms = now_ms() - t0;
        CHECK(rc == OMEGA_GPU_DATTN_OK, "%u+%u: twin decode loop: %s %s", pos0, m, omega_gpu_decode_attn_rc_name(rc), rc ? omega_gpu_decode_attn_last_error() : "");
        if (rc == OMEGA_GPU_DATTN_OK) rc = read_rows(tc, &c, ktw, vtw);
        size_t kd = 0, vd = 0;
        for (size_t i = 0; i < rw && rc == OMEGA_GPU_DATTN_OK; i++) { kd += kch[i] != ktw[i]; vd += vch[i] != vtw[i]; }
        CHECK(rc == OMEGA_GPU_DATTN_OK && kd == 0 && vd == 0, "%u+%u: rows bit-identical to %u decode calls (k %zu, v %zu words differ)", pos0, m, m, kd, vd);
        printf("   twin: %u decode calls %.1f ms; rows k %zu v %zu words differ\n", m, dec_ms, kd, vd);
        if (pos0 + m < max_pos && rc == OMEGA_GPU_DATTN_OK) {
            float *nq = malloc(QW * 4), *o1 = malloc(OW * 4), *o2 = malloc(OW * 4); uint32_t s = seed ^ 0x5a5a5a5au;
            for (unsigned h = 0; h < QH + KVH; h++) { float raw[HD]; for (unsigned i = 0; i < HD; i++) raw[i] = frand(&s, -1.0f, 1.0f); omega_dattn256_perm_vec(raw, nq + h * HD, 64); }
            for (unsigned i = KEND; i < QW; i++) nq[i] = frand(&s, -1.0f, 1.0f);
            int r1 = omega_gpu_decode_attn256(cc, 0, pos0 + m, nq, EPS, 1.0f, o1, NULL), r2 = omega_gpu_decode_attn256(tc, 0, pos0 + m, nq, EPS, 1.0f, o2, NULL);
            CHECK(r1 == OMEGA_GPU_DATTN_OK && r2 == OMEGA_GPU_DATTN_OK && memcmp(o1, o2, OW * 4) == 0, "%u+%u: next decode step bit-identical after prefill and after the decode loop", pos0, m);
            free(nq); free(o1); free(o2);
        }
        free(dout);
    }
    /* timing: reps calls on the same positions (rows rewritten with the same bits) */
    if (reps > 0) {
        uint64_t *pp = malloc((size_t)reps * 8), *aa = malloc((size_t)reps * 8), *cl = malloc((size_t)reps * 8);
        for (int r = 0; r < reps && rc == OMEGA_GPU_DATTN_OK; r++) {
            rc = omega_gpu_prefill_fattn256(cc, 0, pos0, m, qkv_k, XS, EPS, 1.0f, out2, osw * 4, &info);
            pp[r] = info.elapsed_ns[0]; aa[r] = info.elapsed_ns[1]; cl[r] = info.call_ns;
        }
        CHECK(rc == OMEGA_GPU_DATTN_OK, "%u+%u: timing calls", pos0, m);
        if (rc == OMEGA_GPU_DATTN_OK) {
            const uint64_t mp = median_u64(pp, reps), ma = median_u64(aa, reps), mc = median_u64(cl, reps);
            /* useful work: QK^T and PV over the causal context, 2 flops per multiply-add, 16 heads x 256 dims */
            double flops = 0, kvb = 0;
            for (uint32_t t = 0; t < m; t++) flops += 4.0 * (double)(pos0 + t + 1) * QH * HD;
            for (uint32_t cx = 0; cx * OMEGA_FATTN256_TOKENS < m; cx++) {   /* per CTA: the rows of its last warp (L1 serves the rest) */
                uint32_t nk = (cx + 1) * OMEGA_FATTN256_TOKENS; if (nk > m) nk = m;
                kvb += (double)KVH * (((pos0 + nk - 1) >> 4) + 1) * 16.0 * 2 * RB;
            }
            printf("   timing (median of %d): prep %.3f ms, attend %.3f ms, call %.3f ms -> %.0f tokens/s; attend %.2f TFLOPS useful, %.1f GB/s K/V streamed\n",
                   reps, mp / 1e6, ma / 1e6, mc / 1e6, m / ((mp + ma) / 1e9), flops / (double)ma / 1e3, kvb / (double)ma);
            if (g_json) {
                fprintf(g_json, "%s  {\"pos0\": %u, \"m\": %u, \"max_pos\": %u, \"reps\": %d, \"prep_ns\": %" PRIu64 ", \"attend_ns\": %" PRIu64 ", \"call_ns\": %" PRIu64 ", \"attend_useful_flops\": %.0f, \"attend_kv_bytes\": %.0f, \"parity_bad\": %zu, \"parity_worst_ratio\": %.4f}",
                        g_json_first ? "" : ",\n", pos0, m, max_pos, reps, mp, ma, mc, flops, kvb, bad, worst);
                g_json_first = 0;
            }
        }
        free(pp); free(aa); free(cl);
    }
done:
    if (tc) omega_gpu_decode_attn256_cache_free(tc);
    if (cc) omega_gpu_decode_attn256_cache_free(cc);
    free(qkv_k); free(out); free(out2); free(kn); free(vn); free(kch); free(vch); free(ktw); free(vtw); case_free(&c);
    return ok;
}
static void chip_mutants(void) {
    static const struct { int pm, am; } mut[] = { { 1, 0 }, { 3, 0 }, { 0, 1 }, { 0, 2 } };
    for (size_t i = 0; i < sizeof mut / sizeof mut[0]; i++) {
        omega_gpu_prefill_fattn256_test_set_mutant(mut[i].pm, mut[i].am);
        Case c; case_make(&c, 256, 40, 24, 64, 300, 1.0f);
        float *qkv_k = malloc((size_t)c.m * XS); case_kernel_qkv(&c, qkv_k);
        float *out = malloc((size_t)c.m * OW * 4);
        uint16_t *kn = malloc((size_t)c.m * KVH * HD * 2), *vn = malloc((size_t)c.m * KVH * HD * 2), *kch = malloc((size_t)c.m * KVH * HD * 2), *vch = malloc((size_t)c.m * KVH * HD * 2);
        OmegaGpuDecodeAttn256Cache *cc = NULL;
        int rc = cache_with_history(&c, &cc);
        if (rc == OMEGA_GPU_DATTN_OK) rc = omega_gpu_prefill_fattn256(cc, 0, c.pos0, c.m, qkv_k, XS, EPS, 1.0f, out, OW * 4u, NULL);
        double worst = 0; size_t bad = 0; char why[200] = "";
        int rok = 1;
        if (rc == OMEGA_GPU_DATTN_OK) {
            ref_rows(&c, kn, vn);
            rok = read_rows(cc, &c, kch, vch) == OMEGA_GPU_DATTN_OK && rows_ok(&c, kch, vch, kn, vn, why, sizeof why);
            bad = parity(&c, kn, vn, out, OW, 1.0f, &worst);
        }
        const int caught = rc != OMEGA_GPU_DATTN_OK || bad > 0 || !rok;
        CHECK(caught, "chip %s mutant %d NOT caught", mut[i].pm ? "prep" : "attend", mut[i].pm ? mut[i].pm : mut[i].am);
        printf("chip %s mutant %d: %s (rc %s, %zu bad, worst %.1f, rows %s)\n", mut[i].pm ? "prep" : "attend", mut[i].pm ? mut[i].pm : mut[i].am, caught ? "caught" : "MISSED",
               omega_gpu_decode_attn_rc_name(rc), bad, worst, rok ? "ok" : "differ");
        if (cc) omega_gpu_decode_attn256_cache_free(cc);
        free(qkv_k); free(out); free(kn); free(vn); free(kch); free(vch); case_free(&c);
    }
    omega_gpu_prefill_fattn256_test_set_mutant(0, 0);
}

int main(int argc, char **argv) {
    int host = 0, sim = 0, reps = 9, quick = 0, timing_only = 0, longp = 0; const char *json = NULL; uint32_t maxp = 32768;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) host = 1;
        else if (!strcmp(argv[i], "--sim")) sim = 1;
        else if (!strcmp(argv[i], "--poison") && i + 1 < argc) g_poison = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (!strcmp(argv[i], "--sim-point") && i + 3 < argc) {   /* one simulator point: MAX_POS POS0 M */
            uint32_t a = (uint32_t)atoi(argv[i + 1]), b = (uint32_t)atoi(argv[i + 2]), m = (uint32_t)atoi(argv[i + 3]); i += 3;
            char msg[400]; double worst = 0; size_t bad = 0; double t0 = now_ms();
            int err = sim_point(a, b, m, 64, 1.0f, 77, 1, 0, 0, msg, sizeof msg, &worst, &bad);
            CHECK(err == 0 && bad == 0, "sim point %u+%u (max_pos %u): %s", b, m, a, msg);
            printf("sim point %u tokens from %u (max_pos %u): %s %s (%.1f s)\n", m, b, a, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg, (now_ms() - t0) / 1e3);
            host = 1; }
        else if (!strcmp(argv[i], "--quick")) quick = 1;
        else if (!strcmp(argv[i], "--timing-only")) timing_only = 1;
        else if (!strcmp(argv[i], "--long")) longp = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-pos") && i + 1 < argc) maxp = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) json = argv[++i];
        else { fprintf(stderr, "usage: %s [--host-only | --sim | [--quick] [--timing-only] [--long] [--reps N] [--max-pos 32768|131072] [--out receipt.json]]\n", argv[0]); return 2; }
    }
    if (reps < 1) reps = 1;
    host_only();
    if (sim) sim_battery();
    if (!host && !sim) {
        if (json) { g_json = test_out_open(json); if (g_json) fprintf(g_json, "{\"test\": \"gpu_prefill_fattn256\", \"shape\": \"q36 16/2\", \"xstride\": %u, \"rotary_dim\": 64, \"points\": [\n", XS); }
        if (!timing_only) {
            chip_point(256, 0, 1, 1.0f, 1, 1, 0, OW);
            chip_point(256, 0, 37, 1.0f, 2, 1, 0, OW + 16u);
            chip_point(512, 100, 19, 1.0f, 3, 1, 0, OW);
            chip_point(512, 37, 141, 4.0f, 4, 1, 0, OW);
            chip_mutants();
        }
        if (!quick) {
            chip_point(4096, 0, 2048, 1.0f, 8, !timing_only, reps, OW);
            chip_point(8192, 6144, 2048, 1.0f, 9, !timing_only, reps, OW);
            chip_point(32768, 30720, 2048, 1.0f, 10, !timing_only, reps, OW);
            if (maxp >= 131072) chip_point(131072, 129024, 2048, 1.0f, 11, !timing_only, reps, OW);
        }
        if (longp) {   /* G3 (DRA-85): 8192-token chunks ending at 8k and 32k */
            chip_point(8192, 0, 8192, 1.0f, 12, 0, reps, OW);
            chip_point(32768, 24576, 8192, 1.0f, 13, 0, reps, OW);
        }
        if (g_json) { fprintf(g_json, "\n], \"checks\": %d, \"failed\": %d}\n", g_checks, g_failed); fclose(g_json); }
    }
    printf("gpu_prefill_fattn256_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
