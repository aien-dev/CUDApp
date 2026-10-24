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
 * gpu_router_test: the one-launch MoE router (omega_gpu_router_api.h), OWN-30B cut 8 item 2.
 *   --host-only   oracle sanity, codegen budgets and refusals, the libm-free pieces (no device)
 *   --sim         warp-simulator parity at 8 warps (both fragment orders), norm_topk on and off, mutants 1-4 caught
 *   (default)     the chip: parity against the double oracle on random hidden states (two layers, norm_topk on and
 *                 off), second call bit-identical, mutants 1-4 caught, the end-to-end: the grouped FP8 gate+up GEMV
 *                 reading the router-written group table and x' row equals the host-table call bit for bit, chip
 *                 medians (--reps N, --warps W, --depth D, --no-evict, --spin-us, --out receipt.json)
 * Verdict line last: "gpu_router_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_gpu_router_api.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_session.h"
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
static uint16_t bf16_rne(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t r = x + 0x7fffu + ((x >> 16) & 1u); return (uint16_t)(r >> 16); }
static float bf16_f32(uint16_t b) { uint32_t x = (uint32_t)b << 16; float f; memcpy(&f, &x, 4); return f; }
static const float EPS = 1e-6f;
#define H OMEGA_ROUTER_HIDDEN
/* -DROUTER_TEST_Q36: Qwen3.6-35B-A3B (256 routed experts + the shared expert gate row and slot; sc L306) */
#ifdef ROUTER_TEST_Q36
#define NE 256u
#define SHX 1u
#define SPEC_SHAPE , .experts = NE, .shared = SHX
#define GN_E2E 1024u
#define DK_E2E 512u
#else
#define NE OMEGA_ROUTER_EXPERTS
#define SHX 0u
#define SPEC_SHAPE
#define GN_E2E 1536u
#define DK_E2E 768u
#endif
#define NR (NE + SHX)                 /* gate rows */
#define TK OMEGA_ROUTER_TOPK
#define TS (TK + SHX)                 /* output slots */
#define MUT_MAX (SHX ? 5 : 4)
#define MARGIN_FLOOR 1e-4 /* an id mismatch at an oracle margin (8th - 9th logit) below this is recorded, not failed */

typedef struct {
    float x[H], nw[H];
    uint16_t *gate; /* [NR][2048] bf16 (row NE = the shared expert gate when SHX) */
    double xn[H], logits[NR], w[TS], margin;
    uint32_t ids[TS];
} Case;

/* the double oracle from c->x, c->nw, c->gate: xn, logits, ids (descending, lowest index on ties), weights, margin */
static void oracle_fill(Case *c, int norm_topk) {
    double ss = 0; for (unsigned i = 0; i < H; i++) ss += (double)c->x[i] * c->x[i];
    double r = 1.0 / sqrt(ss / (double)H + (double)EPS);
    for (unsigned i = 0; i < H; i++) c->xn[i] = (double)c->x[i] * r * (double)c->nw[i];
    double mx = -1e300;
    for (unsigned e = 0; e < NR; e++) { double l = 0; for (unsigned k = 0; k < H; k++) l += (double)bf16_f32(c->gate[(size_t)e * H + k]) * c->xn[k]; c->logits[e] = l; if (e < NE && l > mx) mx = l; }
    double p[NE], sum = 0; for (unsigned e = 0; e < NE; e++) { p[e] = exp(c->logits[e] - mx); sum += p[e]; }
    for (unsigned e = 0; e < NE; e++) p[e] /= sum;
    int taken[NE]; memset(taken, 0, sizeof taken);
    double s8 = 0;
    for (unsigned r8 = 0; r8 <= TK; r8++) { /* 9 rounds: the 9th gives the margin */
        int best = -1; for (unsigned e = 0; e < NE; e++) if (!taken[e] && (best < 0 || c->logits[e] > c->logits[best])) best = (int)e;
        if (r8 < TK) { c->ids[r8] = (uint32_t)best; taken[best] = 1; s8 += p[best]; c->w[r8] = p[best]; }
        else c->margin = c->logits[c->ids[TK - 1]] - c->logits[best];
    }
    if (norm_topk) for (unsigned i = 0; i < TK; i++) c->w[i] /= s8;
    if (SHX) { c->ids[TK] = NE; c->w[TK] = 1.0 / (1.0 + exp(-c->logits[NE])); } /* the shared slot: never in the top-8 */
}
/* shared_scale: the shared gate row (SHX) is multiplied by it (bf16): a huge row must leave the top-8 untouched */
static void case_make_shared(Case *c, uint32_t seed, float xscale, int norm_topk, float shared_scale) {
    uint32_t s = seed;
    c->gate = malloc((size_t)NR * H * 2);
    for (unsigned i = 0; i < H; i++) { c->x[i] = frand(&s, -2.0f, 2.0f) * xscale; c->nw[i] = frand(&s, 0.8f, 1.2f); }
    for (size_t i = 0; i < (size_t)NR * H; i++) c->gate[i] = bf16_rne(frand(&s, -0.05f, 0.05f) * (i >= (size_t)NE * H ? shared_scale : 1.0f));
    oracle_fill(c, norm_topk);
}
static void case_make(Case *c, uint32_t seed, float xscale, int norm_topk) { case_make_shared(c, seed, xscale, norm_topk, 1.0f); }
static void case_free(Case *c) { free(c->gate); }

/* outputs of one run (chip or simulator) */
typedef struct { uint32_t ids[TS]; float w[TS], xout[H]; uint64_t tgu[TS * 2], tdn[TS * 2]; } Out;

/* compares an Out against the oracle; returns a bitmask of what differed (1 ids, 2 weights, 4 x', 8 tables); fills msg */
static int compare(const Case *c, const Out *o, float out_scale, const uint64_t *dgu, const uint64_t *ddn, char *msg, size_t n, int *margin_case) {
    int bad = 0; *margin_case = 0;
    int ids_eq = memcmp(c->ids, o->ids, sizeof c->ids) == 0;
    if (!ids_eq) { if (c->margin < MARGIN_FLOOR) *margin_case = 1; else bad |= 1; }
    double ww = 0, wx = 0;
    if (ids_eq) for (unsigned i = 0; i < TS; i++) { double d = fabs((double)o->w[i] - c->w[i]); if (d > ww) ww = d; }
    if (ww > 1e-5) bad |= 2;
    double xmax = 0; for (unsigned i = 0; i < H; i++) if (fabs(c->xn[i]) > xmax) xmax = fabs(c->xn[i]);
    for (unsigned i = 0; i < H; i++) { double d = fabs((double)o->xout[i] / (double)out_scale - c->xn[i]); if (d > wx) wx = d; }
    if (wx > 1e-5 * xmax) bad |= 4;
    if (ids_eq) for (unsigned g = 0; g < TS; g++) {
        uint32_t id = o->ids[g];
        if (o->tgu[2 * g] != dgu[2 * id] || o->tgu[2 * g + 1] != dgu[2 * id + 1] || o->tdn[2 * g] != ddn[2 * id] || o->tdn[2 * g + 1] != ddn[2 * id + 1]) bad |= 8;
    }
    snprintf(msg, n, "ids %s%s, w max abs %.2e, x' max abs %.2e (of %.2f), tables %s, margin %.2e", ids_eq ? "equal" : "DIFFER", *margin_case ? " (margin case)" : "", ww, wx, xmax, (bad & 8) ? "WRONG" : "ok", c->margin);
    return bad;
}

/* ------------------------------------------------------------- host only */
static void host_only(void) {
    { Case c; case_make(&c, 1, 1.0f, 1);
      double s = 0; for (unsigned i = 0; i < TK; i++) s += c.w[i];
      CHECK(fabs(s - 1.0) < 1e-12, "oracle: normalised weights sum to 1 (%.15f)", s);
      int desc = 1; for (unsigned i = 1; i < TK; i++) desc &= c.logits[c.ids[i - 1]] >= c.logits[c.ids[i]];
      CHECK(desc, "oracle: ids are in descending logit order");
      CHECK(c.margin > 0, "oracle: margin positive (%.3e)", c.margin);
      double rms = 0; for (unsigned i = 0; i < H; i++) rms += c.xn[i] * c.xn[i] / (c.nw[i] * c.nw[i]); rms = sqrt(rms / H);
      CHECK(fabs(rms - 1.0) < 1e-3, "oracle: RMSNorm output has unit rms before the weight (%.6f)", rms);
      case_free(&c); }
    { Case c; case_make(&c, 2, 1.0f, 0);
      double s = 0; for (unsigned i = 0; i < TK; i++) s += c.w[i];
      CHECK(s < 1.0 && s > 0.0, "oracle: unnormalised weights sum below 1 (%.4f)", s);
      case_free(&c); }
    /* codegen budgets: every production shape fits the register file (32 * warps * declared <= 65536) and the IR limit */
    uint32_t ws[] = { 8, 16, 32 }, ds[] = { 1, 2, 4, 8 };
    for (unsigned i = 0; i < 3; i++) for (unsigned j = 0; j < 4; j++) for (int m = 0; m <= MUT_MAX; m++) {
        OmegaRouterSpec s = { .warps = ws[i], .depth = ds[j], .mutant = m SPEC_SHAPE }; uint32_t n = 0, g = 0;
        int rc = omega_router_prog_stats(&s, &n, &g);
        uint32_t decl = ((g + 1 + 7) / 8) * 8;
        CHECK(rc == 0 && n <= 1024 && 32u * ws[i] * decl <= 65536u, "codegen warps %u depth %u mutant %d: rc %d insns %u gprs %u declared %u", ws[i], ds[j], m, rc, n, g, decl);
        if (m == 0 && rc == 0) printf("codegen warps %2u depth %u: %u instructions, %u registers (declared %u, register file %u of 65536)\n", ws[i], ds[j], n, g, decl, 32u * ws[i] * decl);
    }
    { OmegaRouterSpec s = { .warps = 4, .depth = 8 SPEC_SHAPE }; CHECK(omega_router_prog_stats(&s, NULL, NULL) != 0, "warps 4 refused"); }
    { OmegaRouterSpec s = { .warps = 8, .depth = 3 SPEC_SHAPE }; CHECK(omega_router_prog_stats(&s, NULL, NULL) != 0, "depth 3 refused"); }
    { OmegaRouterSpec s = { .warps = 8, .depth = 8, .mutant = MUT_MAX + 1 SPEC_SHAPE }; CHECK(omega_router_prog_stats(&s, NULL, NULL) != 0, "mutant %d refused", MUT_MAX + 1); }
    { OmegaRouterSpec s = { .warps = 8, .depth = 8, .experts = 192 }; CHECK(omega_router_prog_stats(&s, NULL, NULL) != 0, "192 experts refused"); }
    { OmegaRouterSpec s = { .warps = 8, .depth = 8, .shared = 2 }; CHECK(omega_router_prog_stats(&s, NULL, NULL) != 0, "shared 2 refused"); }
    /* ROWS form (lever 2) at the production shapes: fits the register file; the strides are checked */
    for (unsigned i = 0; i < 3; i++) for (unsigned j = 0; j < 4; j++) {
        OmegaRouterSpec s = { .warps = ws[i], .depth = ds[j], .xstride = 8192, .lbstride = 16896 SPEC_SHAPE }; uint32_t n = 0, g = 0;
        int rc = omega_router_prog_stats(&s, &n, &g);
        uint32_t decl = ((g + 1 + 7) / 8) * 8;
        CHECK(rc == 0 && n <= 1024 && 32u * ws[i] * decl <= 65536u, "codegen ROWS warps %u depth %u: rc %d insns %u gprs %u declared %u", ws[i], ds[j], rc, n, g, decl);
        if (rc == 0) printf("codegen ROWS warps %2u depth %u: %u instructions, %u registers (declared %u)\n", ws[i], ds[j], n, g, decl);
    }
    { OmegaRouterSpec s = { .warps = 8, .depth = 8, .xstride = 8192 SPEC_SHAPE }; CHECK(omega_router_prog_stats(&s, NULL, NULL) != 0, "xstride without lbstride refused"); }
    { OmegaRouterSpec s = { .warps = 8, .depth = 8, .lbstride = 16896 SPEC_SHAPE }; CHECK(omega_router_prog_stats(&s, NULL, NULL) != 0, "lbstride without xstride refused"); }
    { OmegaRouterSpec s = { .warps = 8, .depth = 8, .xstride = 4096, .lbstride = 16896 SPEC_SHAPE }; CHECK(omega_router_prog_stats(&s, NULL, NULL) != 0, "xstride below one row refused"); }
    { OmegaRouterSpec s = { .warps = 8, .depth = 8, .xstride = 8192, .lbstride = 16900 SPEC_SHAPE }; CHECK(omega_router_prog_stats(&s, NULL, NULL) != 0, "unaligned lbstride refused"); }
    { OmegaRouterSpec s = { .warps = 8, .depth = 8, .mutant = 6 SPEC_SHAPE }; CHECK(omega_router_prog_stats(&s, NULL, NULL) != 0, "mutant 6 without the ROWS form refused"); }
    CHECK(OMEGA_ROUTER_ARGS_WORDS <= OMEGA_GPU_SESSION_MAX_ARGS, "argument words fit the constant bank window");
}

/* ------------------------------------------------------------- simulator */
static uint64_t g_sim_issues;
/* one "device" block: x | xout | ids | wts | tab_gu | tab_dn | dir_gu | dir_dn | norm; the gate row block apart */
enum { B_X = 0, B_XOUT = 8192, B_IDS = 16384, B_WTS = 16448, B_TGU = 16512, B_TDN = 16672, B_DGU = 16832, B_DDN = B_DGU + NR * 16,
       B_NORM = B_DDN + NR * 16, B_BYTES = B_NORM + 8192 };
static void fake_dirs(uint64_t *dgu, uint64_t *ddn) { for (unsigned e = 0; e < NR; e++) { dgu[2 * e] = 0x100000000ull + e * 16; dgu[2 * e + 1] = 0x200000000ull + e * 16 + 8; ddn[2 * e] = 0x300000000ull + e * 32; ddn[2 * e + 1] = 0x400000000ull + e * 32 + 16; } }
static int sim_point(uint32_t depth, int mutant, int norm_topk, float out_scale, uint32_t seed, float shared_scale, char *msg, size_t n, int *bad, int *margin_case) {
    static BlackwellIRProgram prog;
    OmegaRouterSpec spec = { .warps = 8, .depth = depth, .mutant = mutant SPEC_SHAPE };
    OmegaRouterSmem sm;
    if (omega_blackwell_codegen_router_prog(&spec, &prog) != 0 || omega_router_smem(&spec, &sm) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    Case c; case_make_shared(&c, seed, 1.0f, norm_topk, shared_scale);
    uint8_t *blk = aligned_alloc(16, B_BYTES); memset(blk, 0, B_BYTES);
    uint16_t *gate = aligned_alloc(16, (size_t)NR * H * 2); memcpy(gate, c.gate, (size_t)NR * H * 2);
    memcpy(blk + B_X, c.x, sizeof c.x); memcpy(blk + B_NORM, c.nw, sizeof c.nw);
    uint64_t dgu[NR * 2], ddn[NR * 2]; fake_dirs(dgu, ddn);
    memcpy(blk + B_DGU, dgu, sizeof dgu); memcpy(blk + B_DDN, ddn, sizeof ddn);
    Out o, copy; int err = 0;
    for (int order = 0; order < 2 && !err; order++) {
        WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = 256; s.order = order;
        static uint8_t smem[16384]; s.smem = smem; s.smem_bytes = sm.total;
        float *xo = (float *)(blk + B_XOUT), *wd = (float *)(blk + B_WTS); uint32_t *idd = (uint32_t *)(blk + B_IDS);
        for (unsigned i = 0; i < H; i++) xo[i] = -999.0f;
        for (unsigned i = 0; i < TS; i++) { wd[i] = -999.0f; idd[i] = 0xffffffffu; }
        memset(blk + B_TGU, 0, B_DGU - B_TGU);
        uint64_t A[10] = { (uintptr_t)(blk + B_X), (uintptr_t)gate, (uintptr_t)(blk + B_NORM), (uintptr_t)(blk + B_XOUT), (uintptr_t)(blk + B_IDS),
                           (uintptr_t)(blk + B_WTS), (uintptr_t)(blk + B_DGU), (uintptr_t)(blk + B_DDN), (uintptr_t)(blk + B_TGU), (uintptr_t)(blk + B_TDN) };
        for (unsigned i = 0; i < 10; i++) { s.cbank[(0x380 + 8 * i) / 4] = (uint32_t)A[i]; s.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
        float fe = EPS, fo = out_scale;
        memcpy(&s.cbank[OMEGA_ROUTER_ARG_EPS / 4], &fe, 4); memcpy(&s.cbank[OMEGA_ROUTER_ARG_OUT_SCALE / 4], &fo, 4);
        s.cbank[OMEGA_ROUTER_ARG_NORM_TOPK / 4] = norm_topk ? 1u : 0u;
        s.nranges = 2;
        s.ranges[0].lo = (uintptr_t)blk; s.ranges[0].hi = (uintptr_t)blk + B_BYTES;
        s.ranges[1].lo = (uintptr_t)gate; s.ranges[1].hi = (uintptr_t)gate + (size_t)NR * H * 2;
        err = ws_run(&s, 1, 1);
        if (err) { snprintf(msg, n, "order %d: %s (pc %d)", order, s.msg, s.err_pc); break; }
        g_sim_issues += s.warp_issues;
        size_t unwritten = 0; for (unsigned i = 0; i < H; i++) if (xo[i] == -999.0f) unwritten++;
        for (unsigned i = 0; i < TS; i++) if (wd[i] == -999.0f || idd[i] == 0xffffffffu) unwritten++;
        if (unwritten) { err = -1; snprintf(msg, n, "%zu output words never written", unwritten); break; }
        memcpy(o.ids, idd, sizeof o.ids); memcpy(o.w, wd, sizeof o.w); memcpy(o.xout, xo, sizeof o.xout); memcpy(o.tgu, blk + B_TGU, TS * 16); memcpy(o.tdn, blk + B_TDN, TS * 16);
        if (order == 0) copy = o;
        else if (memcmp(&copy, &o, sizeof o) != 0) { err = -1; snprintf(msg, n, "the fragment order changed the result"); }
    }
    if (!err) *bad = compare(&c, &o, out_scale, dgu, ddn, msg, n, margin_case);
    free(blk); free(gate); case_free(&c);
    return err;
}
/* ROWS form (L312 lever 2): one launch of `rows` CTAs routes `rows` residual rows of one layer (one gate, one norm, one
 * pair of directories); CTA i reads x row i (stride XST_SIM, padded past 8192 so a wrong stride shows) and writes block i
 * (stride B_BYTES). Every row's outputs must be bit-identical to the plain kernel run on that row alone, and both
 * fragment orders must agree. Mutant 6 (every CTA reads row 0's x) must be caught. */
#define XST_SIM (H * 4u + 64u)
static void sim_args(WsSim *s, BlackwellIRProgram *prog, const OmegaRouterSmem *sm, int order, const uint64_t A[10], int norm_topk, float out_scale) {
    static uint8_t smem[16384];
    memset(s, 0, sizeof *s);
    s->p = prog; s->threads = 256; s->order = order; s->smem = smem; s->smem_bytes = sm->total;
    for (unsigned i = 0; i < 10; i++) { s->cbank[(0x380 + 8 * i) / 4] = (uint32_t)A[i]; s->cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
    float fe = EPS, fo = out_scale;
    memcpy(&s->cbank[OMEGA_ROUTER_ARG_EPS / 4], &fe, 4); memcpy(&s->cbank[OMEGA_ROUTER_ARG_OUT_SCALE / 4], &fo, 4);
    s->cbank[OMEGA_ROUTER_ARG_NORM_TOPK / 4] = norm_topk ? 1u : 0u;
}
static void poison_block(uint8_t *b) {
    float *xo = (float *)(b + B_XOUT), *wd = (float *)(b + B_WTS); uint32_t *idd = (uint32_t *)(b + B_IDS);
    for (unsigned i = 0; i < H; i++) xo[i] = -999.0f;
    for (unsigned i = 0; i < TS; i++) { wd[i] = -999.0f; idd[i] = 0xffffffffu; }
    memset(b + B_TGU, 0, B_DGU - B_TGU);
}
static size_t block_out(const uint8_t *b, Out *o) {
    const float *xo = (const float *)(b + B_XOUT), *wd = (const float *)(b + B_WTS); const uint32_t *idd = (const uint32_t *)(b + B_IDS);
    size_t unwritten = 0; for (unsigned i = 0; i < H; i++) if (xo[i] == -999.0f) unwritten++;
    for (unsigned i = 0; i < TS; i++) if (wd[i] == -999.0f || idd[i] == 0xffffffffu) unwritten++;
    memcpy(o->ids, idd, sizeof o->ids); memcpy(o->w, wd, sizeof o->w); memcpy(o->xout, xo, sizeof o->xout); memcpy(o->tgu, b + B_TGU, TS * 16); memcpy(o->tdn, b + B_TDN, TS * 16);
    return unwritten;
}
static int sim_rows_point(uint32_t depth, int mutant, uint32_t rows, uint32_t seed, char *msg, size_t n, int *bad) {
    static BlackwellIRProgram prog, plain;
    OmegaRouterSpec spec = { .warps = 8, .depth = depth, .mutant = mutant, .xstride = XST_SIM, .lbstride = B_BYTES SPEC_SHAPE };
    OmegaRouterSpec pspec = { .warps = 8, .depth = depth SPEC_SHAPE };
    OmegaRouterSmem sm;
    *bad = 0;
    if (omega_blackwell_codegen_router_prog(&spec, &prog) != 0 || omega_blackwell_codegen_router_prog(&pspec, &plain) != 0 || omega_router_smem(&spec, &sm) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    Case *c = calloc(rows, sizeof *c);
    case_make(&c[0], seed, 1.0f, 1);
    for (uint32_t i = 1; i < rows; i++) { /* one layer: row 0's gate and norm, row i's own x */
        c[i] = c[0]; uint32_t s = seed + 1000u * i;
        for (unsigned k = 0; k < H; k++) c[i].x[k] = frand(&s, -2.0f, 2.0f);
        oracle_fill(&c[i], 1);
    }
    uint8_t *blk = aligned_alloc(16, (size_t)B_BYTES * rows); memset(blk, 0, (size_t)B_BYTES * rows);
    uint8_t *xs = aligned_alloc(16, (size_t)XST_SIM * rows); memset(xs, 0, (size_t)XST_SIM * rows);
    uint16_t *gate = aligned_alloc(16, (size_t)NR * H * 2); memcpy(gate, c[0].gate, (size_t)NR * H * 2);
    for (uint32_t i = 0; i < rows; i++) memcpy(xs + (size_t)i * XST_SIM, c[i].x, sizeof c[i].x);
    memcpy(blk + B_NORM, c[0].nw, sizeof c[0].nw);
    uint64_t dgu[NR * 2], ddn[NR * 2]; fake_dirs(dgu, ddn);
    memcpy(blk + B_DGU, dgu, sizeof dgu); memcpy(blk + B_DDN, ddn, sizeof ddn);
    Out *o = calloc(rows, sizeof *o), *copy = calloc(rows, sizeof *copy); int err = 0;
    uint64_t A[10] = { (uintptr_t)xs, (uintptr_t)gate, (uintptr_t)(blk + B_NORM), (uintptr_t)(blk + B_XOUT), (uintptr_t)(blk + B_IDS),
                       (uintptr_t)(blk + B_WTS), (uintptr_t)(blk + B_DGU), (uintptr_t)(blk + B_DDN), (uintptr_t)(blk + B_TGU), (uintptr_t)(blk + B_TDN) };
    for (int order = 0; order < 2 && !err; order++) {
        WsSim s; sim_args(&s, &prog, &sm, order, A, 1, 1.0f);
        s.nranges = 3;
        s.ranges[0].lo = (uintptr_t)blk; s.ranges[0].hi = (uintptr_t)blk + (size_t)B_BYTES * rows;
        s.ranges[1].lo = (uintptr_t)gate; s.ranges[1].hi = (uintptr_t)gate + (size_t)NR * H * 2;
        s.ranges[2].lo = (uintptr_t)xs; s.ranges[2].hi = (uintptr_t)xs + (size_t)XST_SIM * rows;
        for (uint32_t i = 0; i < rows; i++) poison_block(blk + (size_t)i * B_BYTES);
        err = ws_run(&s, 1, rows);
        if (err) { snprintf(msg, n, "rows order %d: %s (pc %d)", order, s.msg, s.err_pc); break; }
        g_sim_issues += s.warp_issues;
        for (uint32_t i = 0; i < rows && !err; i++) {
            size_t unwritten = block_out(blk + (size_t)i * B_BYTES, &o[i]);
            if (unwritten) { err = -1; snprintf(msg, n, "row %u: %zu output words never written", i, unwritten); }
        }
        if (err) break;
        if (order == 0) memcpy(copy, o, (size_t)rows * sizeof *o);
        else if (memcmp(copy, o, (size_t)rows * sizeof *o) != 0) { err = -1; snprintf(msg, n, "the fragment order changed a row"); }
    }
    /* every row against the plain kernel on its own x (bit for bit) and against the oracle */
    uint32_t rows_differ = 0; int worst = 0; char sub[200] = "";
    for (uint32_t i = 0; i < rows && !err; i++) {
        A[0] = (uintptr_t)(xs + (size_t)i * XST_SIM);
        WsSim s; sim_args(&s, &plain, &sm, 0, A, 1, 1.0f);
        s.nranges = 3;
        s.ranges[0].lo = (uintptr_t)blk; s.ranges[0].hi = (uintptr_t)blk + B_BYTES;
        s.ranges[1].lo = (uintptr_t)gate; s.ranges[1].hi = (uintptr_t)gate + (size_t)NR * H * 2;
        s.ranges[2].lo = (uintptr_t)xs; s.ranges[2].hi = (uintptr_t)xs + (size_t)XST_SIM * rows;
        poison_block(blk);
        err = ws_run(&s, 1, 1);
        if (err) { snprintf(msg, n, "plain row %u: %s (pc %d)", i, s.msg, s.err_pc); break; }
        Out p; if (block_out(blk, &p)) { err = -1; snprintf(msg, n, "plain row %u: output words never written", i); break; }
        if (memcmp(&p, &o[i], sizeof p) != 0) rows_differ++;
        int mc = 0, b = compare(&c[i], &o[i], 1.0f, dgu, ddn, sub, sizeof sub, &mc);
        if (b) worst |= b;
    }
    if (!err) {
        *bad = worst | (rows_differ ? 16 : 0);
        snprintf(msg, n, "%u rows: %u differ from the plain kernel, oracle mask %d (last: %s)", rows, rows_differ, worst, sub);
    }
    free(blk); free(xs); free(gate); free(o); free(copy); case_free(&c[0]); free(c);
    return err;
}
static void sim_battery(void) {
    struct { uint32_t depth; int norm_topk; float oscale; uint32_t seed; const char *name; } pts[] = {
        { 8, 1, 1.0f, 11, "depth 8 norm" }, { 8, 1, 1.0f, 12, "depth 8 norm seed 12" }, { 8, 0, 1.0f, 13, "depth 8 no-norm" },
        { 4, 1, 1.0f, 14, "depth 4 norm" }, { 1, 1, 1.0f, 15, "depth 1 norm" }, { 2, 1, 18446744073709551616.0f, 16, "depth 2 norm out_scale 2^64" },
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        char msg[300]; int bad = 0, mc = 0;
        int err = sim_point(pts[i].depth, 0, pts[i].norm_topk, pts[i].oscale, pts[i].seed, 1.0f, msg, sizeof msg, &bad, &mc);
        CHECK(!err && bad == 0, "sim %s: %s", pts[i].name, msg);
        printf("sim %s: %s %s\n", pts[i].name, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
    }
    if (SHX) {   /* the shared gate row scaled up 1000x: if it entered the top-8 or the softmax, ids / weights would differ */
        char msg[300]; int bad = 0, mc = 0;
        int err = sim_point(8, 0, 1, 1.0f, 17, 1000.0f, msg, sizeof msg, &bad, &mc);
        CHECK(!err && bad == 0, "sim shared row x1000: %s", msg);
        printf("sim shared row x1000: %s %s\n", err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
    }
    for (int m = 1; m <= MUT_MAX; m++) {
        char msg[300]; int bad = 0, mc = 0;
        int err = sim_point(8, m, 1, 1.0f, 21, 1.0f, msg, sizeof msg, &bad, &mc);
        CHECK(!err && bad != 0, "sim mutant %d must be caught: %s", m, msg);
        printf("sim mutant %d: %s (bad mask %d) %s\n", m, err ? "ERROR" : bad ? "caught" : "NOT CAUGHT", bad, msg);
    }
    { /* ROWS form (lever 2): rows 2, 3 and 8 bit-identical to the plain kernel per row; mutant 6 caught at rows 2 */
        struct { uint32_t depth, rows, seed; } rp[] = { { 8, 2, 31 }, { 4, 3, 32 }, { 8, 8, 33 }, { 1, 2, 34 } };
        for (size_t i = 0; i < sizeof rp / sizeof rp[0]; i++) {
            char msg[300]; int bad = 0;
            int err = sim_rows_point(rp[i].depth, 0, rp[i].rows, rp[i].seed, msg, sizeof msg, &bad);
            CHECK(!err && bad == 0, "sim rows %u depth %u: %s", rp[i].rows, rp[i].depth, msg);
            printf("sim rows %u depth %u: %s %s\n", rp[i].rows, rp[i].depth, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
        }
        char msg[300]; int bad = 0;
        int err = sim_rows_point(8, 6, 2, 35, msg, sizeof msg, &bad);
        CHECK(!err && bad != 0, "sim rows mutant 6 must be caught: %s", msg);
        printf("sim rows mutant 6: %s (bad mask %d) %s\n", err ? "ERROR" : bad ? "caught" : "NOT CAUGHT", bad, msg);
    }
    printf("sim: %" PRIu64 " warp issues in total\n", g_sim_issues);
}

/* ------------------------------------------------------------- chip */
static int g_evict = 1;
static void evict_host_caches(void) {
    static uint8_t *buf; const size_t sz = 512u << 20; static uint8_t pass;
    if (!buf) buf = malloc(sz);
    if (!buf) return;
    memset(buf, ++pass, sz);
    volatile uint64_t sum = 0; for (size_t i = 0; i < sz; i += 64) sum += buf[i];
}
static int chip_call(OmegaGpuRouter *r, uint32_t layer, const Case *c, int norm_topk, float out_scale, Out *o, OmegaGpuRouterInfo *info) {
    int rc = omega_gpu_router(r, layer, c->x, EPS, norm_topk, out_scale, o->ids, o->w, info);
    if (rc != OMEGA_GPU_ROUTER_OK) return rc;
    return omega_gpu_router_read_back(r, o->xout, o->tgu, o->tdn);
}
static int chip_run(const char *out_path, uint32_t reps) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot write %s\n", out_path); return 1; }
    if (out) fprintf(out, "{\"receipt\":\"OMEGA_GPU_ROUTER_V1\",\"reps\":%u,\"evict\":%d,\"spin_us\":%u,\"points\":[", reps, g_evict, omega_gpu_session_spin_us());
    OmegaGpuRouter *r = NULL;
    double t0 = now_ms();
    int rc = omega_gpu_router_alloc_shape(2, NE, SHX, &r);
    printf("router alloc (2 layers): rc=%s %.1f ms (%s)\n", omega_gpu_router_rc_name(rc), now_ms() - t0, rc ? omega_gpu_router_last_error() : "");
    CHECK(rc == OMEGA_GPU_ROUTER_OK, "router alloc");
    if (rc) { if (out) fprintf(out, "],\"verdict\":\"FAIL\"}\n"); return 1; }
    uint64_t dgu[NR * 2], ddn[NR * 2]; fake_dirs(dgu, ddn);
    CHECK(omega_gpu_router_set_directory(r, 0, dgu, ddn) == OMEGA_GPU_ROUTER_OK, "set directory layer 0");
    int first = 1, parity_fail = 0, margin_cases = 0;
    /* parity: layer 0, 24 cases norm on, 8 off, 4 with out_scale 2^64; then the same gate on layer 1 (addressing) */
    Case fixed; case_make(&fixed, 1000, 1.0f, 1);
    CHECK(omega_gpu_router_set_layer(r, 0, fixed.gate, fixed.nw) == OMEGA_GPU_ROUTER_OK, "set layer 0");
    for (unsigned i = 0; i < 36; i++) {
        int norm = i < 24 || i >= 32; float os = i >= 32 ? 18446744073709551616.0f : 1.0f;
        /* layer 0 holds `fixed`'s router weights and norm: a fresh x per case, the oracle on fixed.gate / fixed.nw */
        Case c; uint32_t s = 2000 + i; const float xs = i % 3 == 0 ? 0.01f : i % 3 == 1 ? 1.0f : 40.0f;
        for (unsigned k = 0; k < H; k++) c.x[k] = frand(&s, -2.0f, 2.0f) * xs;
        memcpy(c.nw, fixed.nw, sizeof c.nw); c.gate = fixed.gate;
        oracle_fill(&c, norm);
        Out o; OmegaGpuRouterInfo info;
        rc = chip_call(r, 0, &c, norm, os, &o, &info);
        char msg[300]; int bad = -1, mc = 0;
        if (rc == OMEGA_GPU_ROUTER_OK) bad = compare(&c, &o, os, dgu, ddn, msg, sizeof msg, &mc);
        else snprintf(msg, sizeof msg, "rc %s: %s", omega_gpu_router_rc_name(rc), omega_gpu_router_last_error());
        if (bad) parity_fail++;
        if (mc) margin_cases++;
        if (bad || i < 3 || i >= 24) printf("parity %u (norm %d, out_scale %s, x scale %s): %s%s\n", i, norm, os > 2.0f ? "2^64" : "1", i % 3 == 0 ? "0.01" : i % 3 == 1 ? "1" : "40", bad ? "MISMATCH " : "ok ", msg);
        if (out) { fprintf(out, "%s{\"point\":\"parity %u\",\"norm_topk\":%d,\"bad_mask\":%d,\"margin_case\":%d,\"chip_ns\":%" PRIu64 "}", first ? "" : ",", i, norm, bad, mc, info.elapsed_ns); first = 0; }
        if (i == 0 && rc == OMEGA_GPU_ROUTER_OK) {
            Out o2; rc = chip_call(r, 0, &c, norm, os, &o2, NULL);
            CHECK(rc == OMEGA_GPU_ROUTER_OK && memcmp(&o, &o2, sizeof o) == 0, "second call bit-identical (ids, weights, x', tables)");
            printf("second call: %s\n", rc == OMEGA_GPU_ROUTER_OK && memcmp(&o, &o2, sizeof o) == 0 ? "bit-identical" : "DIFFERS");
        }
    }
    CHECK(parity_fail == 0, "chip parity: %d of 36 cases mismatch", parity_fail);
    printf("chip parity: %d of 36 mismatches, %d margin cases (recorded, oracle margin below %.0e)\n", parity_fail, margin_cases, MARGIN_FLOOR);
    /* mutants on a fixed case, layer 0 */
    for (int m = 1; m <= MUT_MAX; m++) {
        omega_gpu_router_test_set_mutant(m);
        Out o; char msg[300]; int bad = -1, mc = 0;
        rc = chip_call(r, 0, &fixed, 1, 1.0f, &o, NULL);
        if (rc == OMEGA_GPU_ROUTER_OK) bad = compare(&fixed, &o, 1.0f, dgu, ddn, msg, sizeof msg, &mc);
        else snprintf(msg, sizeof msg, "rc %s: %s", omega_gpu_router_rc_name(rc), omega_gpu_router_last_error());
        CHECK(bad != 0, "chip mutant %d must be caught: %s", m, msg);
        printf("chip mutant %d: %s (bad mask %d) %s\n", m, bad ? "caught" : "NOT CAUGHT", bad, msg);
        if (out) fprintf(out, ",{\"point\":\"mutant %d\",\"caught\":%d,\"bad_mask\":%d}", m, bad != 0, bad);
    }
    omega_gpu_router_test_set_mutant(0);
    /* end to end on layer 1: real FP8 gate+up shape (128 x 1536 x 2048) and a small down tensor; the router's device
     * table + x' feed omega_gpu_fp8_gemv_grouped_dev; the host-table call with the chip's ids and x'/2^64 must match bit for bit */
    {
        const uint32_t GN = GN_E2E, GK = 2048, DN = 256, DK = DK_E2E;
        size_t gw = (size_t)NR * GN * GK, dw = (size_t)NR * DN * DK;
        uint8_t *w1 = malloc(gw), *w2 = malloc(dw);
        uint16_t *s1 = malloc((size_t)NR * (GN / 128) * (GK / 128) * 2), *s2 = malloc((size_t)NR * (DN / 128) * (DK / 128) * 2);
        uint32_t s = 4242;
        for (size_t i = 0; i < gw; i++) { uint8_t b = (uint8_t)(lcg(&s) >> 24); if ((b & 0x7f) == 0x7f) b ^= 1; w1[i] = b; }
        for (size_t i = 0; i < dw; i++) { uint8_t b = (uint8_t)(lcg(&s) >> 24); if ((b & 0x7f) == 0x7f) b ^= 1; w2[i] = b; }
        for (size_t i = 0; i < (size_t)NR * (GN / 128) * (GK / 128); i++) s1[i] = bf16_rne(frand(&s, 0.5f, 2.0f));
        for (size_t i = 0; i < (size_t)NR * (DN / 128) * (DK / 128); i++) s2[i] = bf16_rne(frand(&s, 0.5f, 2.0f));
        OmegaGpuFp8Tensor *tg = NULL, *td = NULL;
        t0 = now_ms();
        int r1 = omega_gpu_fp8_tensor_upload(NR, GN, GK, w1, s1, &tg), r2 = omega_gpu_fp8_tensor_upload(NR, DN, DK, w2, s2, &td);
        printf("fp8 upload gate+up %zu MB + down %zu MB: rc %s / %s, %.0f ms\n", gw >> 20, dw >> 20, omega_gpu_fp8_rc_name(r1), omega_gpu_fp8_rc_name(r2), now_ms() - t0);
        CHECK(r1 == 0 && r2 == 0, "fp8 uploads for the end-to-end");
        if (r1 == 0 && r2 == 0) {
            Case c; case_make(&c, 3000, 1.0f, 1);
            CHECK(omega_gpu_router_set_layer(r, 1, c.gate, c.nw) == OMEGA_GPU_ROUTER_OK, "set layer 1");
            CHECK(omega_gpu_router_set_directory_fp8(r, 1, tg, td) == OMEGA_GPU_ROUTER_OK, "set directory layer 1 from the fp8 tensors");
            uint64_t rgu[NR * 2], rdn[NR * 2];
            for (unsigned e = 0; e < NR; e++) { omega_gpu_fp8_tensor_entry(tg, e, &rgu[2 * e], &rgu[2 * e + 1]); omega_gpu_fp8_tensor_entry(td, e, &rdn[2 * e], &rdn[2 * e + 1]); }
            Out o; OmegaGpuRouterInfo info; char msg[300]; int mc = 0;
            const float OS = 18446744073709551616.0f;
            rc = chip_call(r, 1, &c, 1, OS, &o, &info);
            int bad = rc == 0 ? compare(&c, &o, OS, rgu, rdn, msg, sizeof msg, &mc) : -1;
            CHECK(bad == 0, "e2e router on layer 1 with real directories: %s", rc ? omega_gpu_router_last_error() : msg);
            printf("e2e router layer 1: %s %s\n", bad ? "MISMATCH" : "ok", rc ? omega_gpu_router_last_error() : msg);
            uint64_t xva, tgva, tdva; omega_gpu_router_device_outputs(r, &xva, &tgva, &tdva);
            float *ydev = malloc((size_t)TS * GN * 4), *yhost = malloc((size_t)TS * GN * 4), *xh = malloc(H * 4);
            for (unsigned k = 0; k < H; k++) xh[k] = o.xout[k] / OS; /* exact: a power of two */
            OmegaGpuFp8Info fi, fh;
            int rd = omega_gpu_fp8_gemv_grouped_dev(tg, TS, tgva, xva, ydev, &fi);
            int rh = omega_gpu_fp8_gemv_grouped_f32(tg, TS, o.ids, xh, 1, yhost, &fh);
            CHECK(rd == 0 && rh == 0, "grouped gate+up launches: dev %s host %s", omega_gpu_fp8_rc_name(rd), omega_gpu_fp8_rc_name(rh));
            int same = rd == 0 && rh == 0 && memcmp(ydev, yhost, (size_t)TS * GN * 4) == 0;
            CHECK(same, "grouped gate+up from the router-written table and x' equals the host-table call bit for bit");
            printf("e2e grouped gate+up (8 experts, %u x %u): device table/x' vs host table: %s; chip %.1f us vs %.1f us, call %.1f us vs %.1f us\n", GN, GK, same ? "bit-identical" : "DIFFER",
                   fi.elapsed_ns / 1e3, fh.elapsed_ns / 1e3, fi.call_ns / 1e3, fh.call_ns / 1e3);
            /* one group against the FP8 host oracle (the grouped kernel is proven elsewhere; this pins the plumbing) */
            if (rd == 0) {
                double *yo = malloc(GN * sizeof(double)), *ys = malloc(GN * sizeof(double));
                uint32_t id = o.ids[3];
                omega_fp8_gemv_oracle(w1 + (size_t)id * GN * GK, s1 + (size_t)id * (GN / 128) * (GK / 128), GN, GK, xh, yo, ys);
                double worst = 0; for (unsigned i = 0; i < GN; i++) { double d = fabs((double)ydev[3 * GN + i] - yo[i]) / (ys[i] + 1e-30); if (d > worst) worst = d; }
                CHECK(worst < 1e-5, "e2e group 3 (expert %u) vs the FP8 host oracle: worst rel %.2e", id, worst);
                printf("e2e group 3 vs fp8 oracle: worst rel %.2e\n", worst);
                free(yo); free(ys);
            }
            /* sequence timing: router then grouped gate+up (dev), medians */
            uint64_t *er = malloc(reps * 8), *ec = malloc(reps * 8), *eg = malloc(reps * 8), *es = malloc(reps * 8);
            uint32_t done = 0;
            for (uint32_t i = 0; i < reps; i++) {
                if (g_evict) evict_host_caches();
                double a = now_ms();
                if (omega_gpu_router(r, 1, c.x, EPS, 1, OS, o.ids, o.w, &info) != 0) break;
                if (omega_gpu_fp8_gemv_grouped_dev(tg, TS, tgva, xva, ydev, &fi) != 0) break;
                es[done] = (uint64_t)((now_ms() - a) * 1e6);
                er[done] = info.elapsed_ns; ec[done] = info.call_ns; eg[done] = fi.elapsed_ns; done++;
            }
            CHECK(done == reps, "timing loop completed %u of %u", done, reps);
            if (done) {
                qsort(er, done, 8, cmp_u64); qsort(ec, done, 8, cmp_u64); qsort(eg, done, 8, cmp_u64); qsort(es, done, 8, cmp_u64);
                printf("timing (medians of %u, evict %d, warps %u depth %u): router chip %.1f us, router call %.1f us, grouped gate+up dev chip %.1f us, router+gate+up sequence %.1f us\n",
                       done, g_evict, info.warps, info.depth, er[done / 2] / 1e3, ec[done / 2] / 1e3, eg[done / 2] / 1e3, es[done / 2] / 1e3);
                if (out) fprintf(out, ",{\"point\":\"timing\",\"warps\":%u,\"depth\":%u,\"reps\":%u,\"router_chip_median_ns\":%" PRIu64 ",\"router_call_median_ns\":%" PRIu64 ",\"gateup_dev_chip_median_ns\":%" PRIu64 ",\"sequence_median_ns\":%" PRIu64 ",\"router_bytes\":%" PRIu64 ",\"marker\":%u}",
                                 info.warps, info.depth, done, er[done / 2], ec[done / 2], eg[done / 2], es[done / 2], info.weight_bytes, info.completion_marker);
            }
            free(er); free(ec); free(eg); free(es); free(ydev); free(yhost); free(xh);
            case_free(&c);
        }
        if (tg) omega_gpu_fp8_tensor_free(tg);
        if (td) omega_gpu_fp8_tensor_free(td);
        free(w1); free(w2); free(s1); free(s2);
    }
    case_free(&fixed);
    omega_gpu_router_free(r);
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS"); fclose(out); }
    return g_failed ? 1 : 0;
}

int main(int argc, char **argv) {
    int host = 0, sim = 0; const char *out_path = NULL; uint32_t reps = 0, warps = 0, depth = 0, spin = 2000;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--warps") == 0 && i + 1 < argc) warps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) depth = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--spin-us") == 0 && i + 1 < argc) spin = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-evict") == 0) g_evict = 0;
        else { printf("unknown argument %s\n", argv[i]); return 2; }
    }
    if (host) host_only();
    if (sim) sim_battery();
    if (!host && !sim) {
        if (warps || depth) omega_gpu_router_set_shape(warps, depth);
        (void)omega_gpu_session_set_spin_us(spin);
        chip_run(out_path, reps ? reps : 1);
        omega_gpu_session_close();
    }
    printf("gpu_router_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
