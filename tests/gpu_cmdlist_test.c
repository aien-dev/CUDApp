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
 * gpu_cmdlist_test: OWN-30B W-L2 decode launch collapse (omega_gpu_cmdlist.h + omega_blackwell_codegen_dec.h).
 *   --host-only   codegen budgets and refusals of the four glue kernels (no device)
 *   --sim         warp-simulator parity of AXPY chain, RES (ADD / COMBINE / NORM, scale 1 and 2^64), SWIGLU, ARGMAX
 *                 (two stages, tie case) against the host reference; mutants caught
 *   (default)     chip, caller holds the GB10 lock:
 *                 baseline: L dependent AXPY launches one omega_gpu_session_launch at a time
 *                 main: the same L launches recorded once and submitted once (PCAS) and in INLINE mode; x bit-identical
 *                 to the baseline, every launch semaphore DONE; negative control skip_launch (must fail, first_missing
 *                 named, x short by one a); BARRIER_NONE recorded (informational); per-launch medians (--reps N);
 *                 glue kernels in one list: RES ADD|NORM, SWIGLU, RES COMBINE|NORM, ARGMAX x2 against the host
 *                 reference; mutants caught on chip. --out receipt.json
 * Verdict: exit 0 when every CHECK passed.
 */
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_codegen_dec.h"
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
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static uint64_t median_u64(uint64_t *v, unsigned n) { qsort(v, n, sizeof *v, cmp_u64); return v[n / 2]; }

#define H OMEGA_DEC_HIDDEN
#define TK OMEGA_DEC_TOPK
#define INTER OMEGA_DEC_INTER
#define VOCAB 151936u
#define AM_CTAS ((VOCAB + OMEGA_DEC_ARGMAX_SPAN - 1) / OMEGA_DEC_ARGMAX_SPAN) /* 75 */
#define AM_PAD (AM_CTAS * OMEGA_DEC_ARGMAX_SPAN)                                 /* 153600 */
#define EPS 1e-6f
#define SCALE_2P64 18446744073709551616.0f

/* ---- host references (float, the kernel's operation order where it matters; -ffp-contract=off) ---- */
static void ref_res(uint32_t mode, float *x, const float *a, const float *w, const float *nw, float *y, float eps, float scale) {
    if (mode & OMEGA_DEC_RES_ADD) for (unsigned i = 0; i < H; i++) x[i] = x[i] + a[i];
    if (mode & OMEGA_DEC_RES_COMBINE) for (unsigned i = 0; i < H; i++) {
        float m = w[0] * a[i];
        for (unsigned g = 1; g < TK; g++) { float t = w[g] * a[g * H + i]; m = m + t; }
        x[i] = x[i] + m;
    }
    if (mode & OMEGA_DEC_RES_NORM) {
        double sq = 0; for (unsigned i = 0; i < H; i++) sq += (double)x[i] * x[i];
        float rs = (float)(1.0 / sqrt(sq / H + (double)eps));
        for (unsigned i = 0; i < H; i++) { float v = x[i] * rs; v = v * nw[i]; y[i] = v * scale; }
    }
}
static void ref_swiglu(const float *gu, float *act, float scale) {
    for (unsigned g = 0; g < TK; g++) for (unsigned c = 0; c < INTER; c++) {
        float v = gu[g * 2 * INTER + c], u = gu[g * 2 * INTER + INTER + c];
        float s = v / (1.0f + expf(-v));
        act[g * INTER + c] = s * u * scale;
    }
}
static uint32_t ref_argmax(const float *v, uint32_t n) { uint32_t b = 0; for (uint32_t i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }
static int close_rel(float a, float b, float rel, float abs_) { float d = fabsf(a - b); return d <= abs_ || d <= rel * fabsf(b); }

/* ---- one glue-kernel case: buffers in plain host memory (sim) or in device buffers (chip) ---- */
typedef struct {
    float *x, *a8, *w, *nw, *y, *gu, *act, *logits, *ov; uint32_t *oi;
    float x_ref[H], y_ref[H], act_ref[TK * INTER]; uint32_t am_ref;
} Glue;
static void glue_fill(Glue *g, uint32_t seed, int tie) {
    uint32_t s = seed;
    for (unsigned i = 0; i < H; i++) { g->x[i] = frand(&s, -2, 2); g->nw[i] = frand(&s, 0.5f, 1.5f); }
    for (unsigned i = 0; i < TK * H; i++) g->a8[i] = frand(&s, -1, 1);
    for (unsigned i = 0; i < TK; i++) g->w[i] = frand(&s, 0, 0.3f);
    for (unsigned i = 0; i < TK * 2 * INTER; i++) g->gu[i] = frand(&s, -4, 4);
    for (unsigned i = 0; i < AM_PAD; i++) g->logits[i] = i < VOCAB ? frand(&s, -50, 50) : -1e30f;
    if (tie) { g->logits[1007] = 60.0f; g->logits[90007] = 60.0f; } /* lowest index wins: 1007 (a j == 7 slot) */
    for (unsigned i = 0; i < OMEGA_DEC_ARGMAX_SPAN; i++) { g->ov[i] = -1e30f; g->oi[i] = 0; }
    g->am_ref = ref_argmax(g->logits, VOCAB);
}

/* ---- kernels ---- */
typedef struct { BlackwellIRProgram prog; OmegaGpuMem code; uint32_t gpr; int built; } Kern;
static Kern K_axpy, K_res_an, K_res_cn, K_sw, K_am0, K_am1, K_res_m1, K_res_m2, K_res_cn_m1, K_sw_m1, K_am_m1, K_axpy_big;

static int kern_prog(Kern *k, const OmegaDecSpec *sp) { k->built = 0; return omega_blackwell_codegen_dec_prog(sp, &k->prog); }
static int kern_upload(Kern *k) {
    size_t need = omega_bw_program_code_bytes(&k->prog);
    if (need == 0 || omega_gpu_session_alloc_code(need, &k->code) != 0) return -1;
    OmegaBlackwellKernel kk;
    if (omega_blackwell_kernel_encode_into(&k->prog, k->code.cpu, need, &kk) != 0 || omega_gpu_session_sync_to_gpu(&k->code) != 0) return -1;
    k->gpr = omega_bw_gpr_declared(kk.gpr_count);
    if (k->gpr < 64u) k->gpr = 64u;
    k->built = 1;
    return 0;
}
static void put64(uint32_t *a, unsigned off, uint64_t v) { a[(off - 0x380) / 4] = (uint32_t)v; a[(off - 0x380) / 4 + 1] = (uint32_t)(v >> 32); }
static void putf(uint32_t *a, unsigned off, float f) { memcpy(&a[(off - 0x380) / 4], &f, 4); }

#define AXPY_STRIDE 2048u          /* small launch: 8 CTAs x 256 threads, one element each */
#define AXPY_BIG_STRIDE 65536u     /* heavy launch for the no-barrier control: 256 CTAs x 16 iterations */
#define AXPY_BIG_ITERS 16u

static int specs_build(void) {
    OmegaDecSpec s;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_AXPY; s.stride = AXPY_STRIDE; if (kern_prog(&K_axpy, &s)) return -1;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_AXPY; s.stride = AXPY_BIG_STRIDE; if (kern_prog(&K_axpy_big, &s)) return -1;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_RES; s.mode = OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_NORM; if (kern_prog(&K_res_an, &s)) return -1;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_RES; s.mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM; if (kern_prog(&K_res_cn, &s)) return -1;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_RES; s.mode = OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_NORM; s.mutant = 1; if (kern_prog(&K_res_m1, &s)) return -1;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_RES; s.mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM; s.mutant = 2; if (kern_prog(&K_res_m2, &s)) return -1;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_RES; s.mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM; s.mutant = 1; if (kern_prog(&K_res_cn_m1, &s)) return -1;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_SWIGLU; if (kern_prog(&K_sw, &s)) return -1;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_SWIGLU; s.mutant = 1; if (kern_prog(&K_sw_m1, &s)) return -1;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_ARGMAX; s.mode = 0; if (kern_prog(&K_am0, &s)) return -1;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_ARGMAX; s.mode = 1; if (kern_prog(&K_am1, &s)) return -1;
    memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_ARGMAX; s.mode = 0; s.mutant = 1; if (kern_prog(&K_am_m1, &s)) return -1;
    return 0;
}

/* ---- host-only ---- */
static int host_only(void) {
    const struct { const char *name; OmegaDecSpec s; } cases[] = {
        { "axpy", { .kind = OMEGA_DEC_AXPY, .stride = AXPY_STRIDE } },
        { "res add|norm", { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_NORM } },
        { "res combine|norm", { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM } },
        { "res add", { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_ADD } },
        { "swiglu", { .kind = OMEGA_DEC_SWIGLU } },
        { "argmax", { .kind = OMEGA_DEC_ARGMAX } },
        { "argmax indexed", { .kind = OMEGA_DEC_ARGMAX, .mode = 1 } },
    };
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        uint32_t insns = 0, gprs = 0;
        int rc = omega_dec_prog_stats(&cases[i].s, &insns, &gprs);
        CHECK(rc == 0, "codegen %s", cases[i].name);
        CHECK(gprs <= 200, "%s gprs %u within budget", cases[i].name, gprs);
        printf("codegen %s: %u insns, peak %u gprs, declared %u\n", cases[i].name, insns, gprs, omega_bw_gpr_declared(gprs));
    }
    OmegaDecSpec bad[] = { { .kind = OMEGA_DEC_AXPY }, { .kind = OMEGA_DEC_AXPY, .stride = 100 }, { .kind = OMEGA_DEC_RES },
                           { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_COMBINE }, { .kind = (OmegaDecKind)0 }, { .kind = OMEGA_DEC_RES, .mode = 8 } };
    for (unsigned i = 0; i < sizeof bad / sizeof bad[0]; i++) CHECK(omega_dec_prog_stats(&bad[i], NULL, NULL) != 0, "refusal %u", i);
    /* widths (sc #642): 5120 and 8192 build; not a multiple of 1024, above 8192, COMBINE off 2048, SWIGLU off 256 refuse */
    OmegaDecSpec w; memset(&w, 0, sizeof w); w.kind = OMEGA_DEC_RES; w.mode = OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_NORM;
    w.width = 5120; CHECK(omega_dec_prog_stats(&w, NULL, NULL) == 0, "codegen res width 5120");
    w.width = 8192; CHECK(omega_dec_prog_stats(&w, NULL, NULL) == 0, "codegen res width 8192");
    w.width = 3000; CHECK(omega_dec_prog_stats(&w, NULL, NULL) != 0, "refusal res width 3000");
    w.width = 9216; CHECK(omega_dec_prog_stats(&w, NULL, NULL) != 0, "refusal res width 9216");
    w.width = 5120; w.mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM; CHECK(omega_dec_prog_stats(&w, NULL, NULL) != 0, "refusal combine width 5120");
    memset(&w, 0, sizeof w); w.kind = OMEGA_DEC_SWIGLU; w.width = 17408; CHECK(omega_dec_prog_stats(&w, NULL, NULL) == 0, "codegen swiglu width 17408");
    w.width = 17409; CHECK(omega_dec_prog_stats(&w, NULL, NULL) != 0, "refusal swiglu width 17409");
    return g_failed ? 1 : 0;
}

/* ---- simulator ---- */
static int sim_one(Kern *k, const uint32_t *args, unsigned n_args, uint32_t gx, uint32_t gy, WsRange *ranges, int nr, const char *what) {
    static uint8_t smem[OMEGA_DEC_SHARED_BYTES];
    WsSim s; memset(&s, 0, sizeof s);
    s.p = &k->prog; s.threads = OMEGA_DEC_THREADS; s.smem = smem; s.smem_bytes = sizeof smem;
    memcpy(&s.cbank[0x380 / 4], args, n_args * 4u);
    s.nranges = nr; for (int i = 0; i < nr; i++) s.ranges[i] = ranges[i];
    int err = ws_run(&s, gx, gy);
    CHECK(err == 0, "sim %s: %s (pc %d)", what, s.msg, s.err_pc);
    return err;
}
static int sim_mode(void) {
    if (specs_build()) { printf("FAIL: codegen\n"); return 1; }
    /* AXPY chain: 3 launches x += a over 2048 */
    {
        static float x[AXPY_STRIDE], a[AXPY_STRIDE], xr[AXPY_STRIDE];
        uint32_t s = 7; for (unsigned i = 0; i < AXPY_STRIDE; i++) { x[i] = frand(&s, -1, 1); a[i] = frand(&s, -1, 1); xr[i] = x[i]; }
        uint32_t args[OMEGA_DEC_AXPY_WORDS] = { 0 }; put64(args, OMEGA_DEC_AXPY_X, (uintptr_t)x); put64(args, OMEGA_DEC_AXPY_A, (uintptr_t)a); args[(OMEGA_DEC_AXPY_ITERS - 0x380) / 4] = 1;
        WsRange r[2] = { { (uintptr_t)x, (uintptr_t)x + sizeof x }, { (uintptr_t)a, (uintptr_t)a + sizeof a } };
        for (int l = 0; l < 3; l++) { if (sim_one(&K_axpy, args, OMEGA_DEC_AXPY_WORDS, AXPY_STRIDE / OMEGA_DEC_THREADS, 1, r, 2, "axpy")) return 1; for (unsigned i = 0; i < AXPY_STRIDE; i++) xr[i] = xr[i] + a[i]; }
        CHECK(memcmp(x, xr, sizeof x) == 0, "sim axpy chain bit-identical");
    }
    /* glue kernels */
    static float x[H], a8[TK * H], w[TK], nw[H], y[H], gu[TK * 2 * INTER], act[TK * INTER], logits[AM_PAD], ov[OMEGA_DEC_ARGMAX_SPAN];
    static uint32_t oi[OMEGA_DEC_ARGMAX_SPAN];
    Glue g = { x, a8, w, nw, y, gu, act, logits, ov, oi, { 0 }, { 0 }, { 0 }, 0 };
    WsRange rr[6] = { { (uintptr_t)x, (uintptr_t)x + sizeof x }, { (uintptr_t)a8, (uintptr_t)a8 + sizeof a8 }, { (uintptr_t)w, (uintptr_t)w + sizeof w },
                      { (uintptr_t)nw, (uintptr_t)nw + sizeof nw }, { (uintptr_t)y, (uintptr_t)y + sizeof y }, { (uintptr_t)gu, (uintptr_t)gu + sizeof gu } };
    const float scales[2] = { 1.0f, SCALE_2P64 };
    for (int sc = 0; sc < 2; sc++) for (int mode = 0; mode < 2; mode++) for (int mut = 0; mut < 2; mut++) {
        glue_fill(&g, 100 + sc, 0);
        memcpy(g.x_ref, x, sizeof x);
        const uint32_t M = mode == 0 ? (OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_NORM) : (OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM);
        ref_res(M, g.x_ref, a8, w, nw, g.y_ref, EPS, scales[sc]);
        uint32_t args[OMEGA_DEC_RES_WORDS] = { 0 };
        put64(args, OMEGA_DEC_RES_X, (uintptr_t)x); put64(args, OMEGA_DEC_RES_A, (uintptr_t)a8); put64(args, OMEGA_DEC_RES_W, (uintptr_t)w);
        put64(args, OMEGA_DEC_RES_NW, (uintptr_t)nw); put64(args, OMEGA_DEC_RES_Y, (uintptr_t)y); putf(args, OMEGA_DEC_RES_EPS, EPS); putf(args, OMEGA_DEC_RES_SCALE, scales[sc]);
        Kern *k = mut ? (mode == 0 ? &K_res_m1 : &K_res_m2) : (mode == 0 ? &K_res_an : &K_res_cn);
        if (sim_one(k, args, OMEGA_DEC_RES_WORDS, 1, 1, rr, 5, "res")) return 1;
        int xbad = memcmp(x, g.x_ref, sizeof x) != 0, ybad = 0;
        for (unsigned i = 0; i < H; i++) if (!close_rel(y[i], g.y_ref[i], 2e-5f, 1e-6f)) ybad++;
        if (!mut) { CHECK(!xbad, "sim res mode %u scale %d: x bit-identical", M, sc); CHECK(ybad == 0, "sim res mode %u scale %d: y within 2e-5 (%d bad)", M, sc, ybad); }
        else CHECK(xbad || ybad, "sim res mutant (mode %u) caught", M);
        printf("sim res mode %u scale %s mutant %d: x %s, y %d/%u outside tolerance\n", M, sc ? "2^64" : "1", mut, xbad ? "DIFFERS" : "identical", ybad, H);
    }
    for (int mut = 0; mut < 2; mut++) {
        glue_fill(&g, 200, 0);
        ref_swiglu(gu, g.act_ref, 1.0f);
        uint32_t args[OMEGA_DEC_SW_WORDS] = { 0 }; put64(args, OMEGA_DEC_SW_GU, (uintptr_t)gu); put64(args, OMEGA_DEC_SW_ACT, (uintptr_t)act); putf(args, OMEGA_DEC_SW_SCALE, 1.0f);
        WsRange r2[2] = { rr[5], { (uintptr_t)act, (uintptr_t)act + sizeof act } };
        if (sim_one(mut ? &K_sw_m1 : &K_sw, args, OMEGA_DEC_SW_WORDS, INTER / OMEGA_DEC_THREADS, TK, r2, 2, "swiglu")) return 1;
        int bad = 0; for (unsigned i = 0; i < TK * INTER; i++) if (!close_rel(act[i], g.act_ref[i], 5e-5f, 1e-6f)) bad++;
        if (!mut) CHECK(bad == 0, "sim swiglu within 5e-5 (%d bad)", bad); else CHECK(bad > 0, "sim swiglu mutant caught");
        printf("sim swiglu mutant %d: %d/%u outside tolerance\n", mut, bad, TK * INTER);
    }
    for (int tie = 0; tie < 2; tie++) for (int mut = 0; mut < 2; mut++) {
        glue_fill(&g, 300 + tie, tie);
        uint32_t a0[OMEGA_DEC_AM_WORDS] = { 0 }, a1[OMEGA_DEC_AM_WORDS] = { 0 };
        static float ov2[1]; static uint32_t oi2[1];
        put64(a0, OMEGA_DEC_AM_V, (uintptr_t)logits); put64(a0, OMEGA_DEC_AM_I, 0); put64(a0, OMEGA_DEC_AM_OV, (uintptr_t)ov); put64(a0, OMEGA_DEC_AM_OI, (uintptr_t)oi);
        put64(a1, OMEGA_DEC_AM_V, (uintptr_t)ov); put64(a1, OMEGA_DEC_AM_I, (uintptr_t)oi); put64(a1, OMEGA_DEC_AM_OV, (uintptr_t)ov2); put64(a1, OMEGA_DEC_AM_OI, (uintptr_t)oi2);
        WsRange r3[4] = { { (uintptr_t)logits, (uintptr_t)logits + sizeof logits }, { (uintptr_t)ov, (uintptr_t)ov + sizeof ov }, { (uintptr_t)oi, (uintptr_t)oi + sizeof oi },
                          { (uintptr_t)ov2, (uintptr_t)ov2 + 4 } };
        WsRange r4[5] = { r3[0], r3[1], r3[2], r3[3], { (uintptr_t)oi2, (uintptr_t)oi2 + 4 } };
        if (sim_one(mut ? &K_am_m1 : &K_am0, a0, OMEGA_DEC_AM_WORDS, AM_CTAS, 1, r4, 5, "argmax stage 1")) return 1;
        if (sim_one(&K_am1, a1, OMEGA_DEC_AM_WORDS, 1, 1, r4, 5, "argmax stage 2")) return 1;
        if (!mut) CHECK(oi2[0] == g.am_ref && ov2[0] == logits[g.am_ref], "sim argmax tie %d: %u (ref %u)", tie, oi2[0], g.am_ref);
        else if (tie) CHECK(oi2[0] != g.am_ref, "sim argmax mutant caught (max at a j == 7 slot)");
        printf("sim argmax tie %d mutant %d: index %u value %g (ref %u)\n", tie, mut, oi2[0], ov2[0], g.am_ref);
    }
    return g_failed ? 1 : 0;
}

/* ---- Qwen3.8-27B widths (sc #642): RES over 5120, SWIGLU over one dense 17408 row ---- */
#define WH 5120u
#define WI 17408u
static int sim_wide(void) {
    static Kern kw[4];
    static float x[WH], xr[WH], a[WH], nw[WH], y[WH], yr[WH], gu[2 * WI], act[WI], actr[WI];
    const uint32_t modes[3] = { OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_NORM, OMEGA_DEC_RES_NORM, OMEGA_DEC_RES_ADD };
    for (int mi = 0; mi < 3; mi++) for (int mut = 0; mut < 2; mut++) {
        if (mut && !(modes[mi] & OMEGA_DEC_RES_NORM)) continue;
        OmegaDecSpec s; memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_RES; s.mode = modes[mi]; s.mutant = mut; s.width = WH;
        if (kern_prog(&kw[0], &s)) { CHECK(0, "codegen res width %u mode %u", WH, modes[mi]); return 1; }
        uint32_t sd = 300 + mi;
        for (unsigned i = 0; i < WH; i++) { x[i] = frand(&sd, -2, 2); a[i] = frand(&sd, -1, 1); nw[i] = frand(&sd, 0.5f, 1.5f); y[i] = -7.0f; }
        memcpy(xr, x, sizeof x);
        /* ref_res at width WH */
        if (modes[mi] & OMEGA_DEC_RES_ADD) for (unsigned i = 0; i < WH; i++) xr[i] = xr[i] + a[i];
        if (modes[mi] & OMEGA_DEC_RES_NORM) {
            double sq = 0; for (unsigned i = 0; i < WH; i++) sq += (double)xr[i] * xr[i];
            float rs = (float)(1.0 / sqrt(sq / WH + (double)EPS));
            for (unsigned i = 0; i < WH; i++) { float v = xr[i] * rs; v = v * nw[i]; yr[i] = v * SCALE_2P64; }
        }
        uint32_t args[OMEGA_DEC_RES_WORDS] = { 0 };
        put64(args, OMEGA_DEC_RES_X, (uintptr_t)x); put64(args, OMEGA_DEC_RES_A, (uintptr_t)a); put64(args, OMEGA_DEC_RES_NW, (uintptr_t)nw);
        put64(args, OMEGA_DEC_RES_Y, (uintptr_t)y); putf(args, OMEGA_DEC_RES_EPS, EPS); putf(args, OMEGA_DEC_RES_SCALE, SCALE_2P64);
        WsRange r[4] = { { (uintptr_t)x, (uintptr_t)x + sizeof x }, { (uintptr_t)a, (uintptr_t)a + sizeof a },
                         { (uintptr_t)nw, (uintptr_t)nw + sizeof nw }, { (uintptr_t)y, (uintptr_t)y + sizeof y } };
        if (sim_one(&kw[0], args, OMEGA_DEC_RES_WORDS, 1, 1, r, 4, "res wide")) return 1;
        int xbad = memcmp(x, xr, sizeof x) != 0, ybad = 0;
        if (modes[mi] & OMEGA_DEC_RES_NORM) for (unsigned i = 0; i < WH; i++) if (!close_rel(y[i], yr[i], 2e-5f, 1e-6f)) ybad++;
        if (!mut) { CHECK(!xbad, "sim res width %u mode %u: x bit-identical", WH, modes[mi]); CHECK(ybad == 0, "sim res width %u mode %u: y within 2e-5 (%d bad)", WH, modes[mi], ybad); }
        else CHECK(ybad > 0, "sim res width %u mode %u mutant caught", WH, modes[mi]);
        printf("sim res width %u mode %u mutant %d: x %s, y %d/%u outside tolerance\n", WH, modes[mi], mut, xbad ? "DIFFERS" : "identical", ybad, WH);
    }
    for (int mut = 0; mut < 2; mut++) {
        OmegaDecSpec s; memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_SWIGLU; s.mutant = mut; s.width = WI;
        if (kern_prog(&kw[1], &s)) { CHECK(0, "codegen swiglu width %u", WI); return 1; }
        uint32_t sd = 400;
        for (unsigned i = 0; i < 2 * WI; i++) gu[i] = frand(&sd, -4, 4);
        for (unsigned c = 0; c < WI; c++) { float v = gu[c], u = gu[WI + c]; actr[c] = v / (1.0f + expf(-v)) * u * SCALE_2P64; act[c] = -7.0f; }
        uint32_t args[OMEGA_DEC_SW_WORDS] = { 0 }; put64(args, OMEGA_DEC_SW_GU, (uintptr_t)gu); put64(args, OMEGA_DEC_SW_ACT, (uintptr_t)act); putf(args, OMEGA_DEC_SW_SCALE, SCALE_2P64);
        WsRange r[2] = { { (uintptr_t)gu, (uintptr_t)gu + sizeof gu }, { (uintptr_t)act, (uintptr_t)act + sizeof act } };
        if (sim_one(&kw[1], args, OMEGA_DEC_SW_WORDS, WI / OMEGA_DEC_THREADS, 1, r, 2, "swiglu wide")) return 1;
        int bad = 0; for (unsigned i = 0; i < WI; i++) if (!close_rel(act[i], actr[i], 5e-5f, 1e-6f)) bad++;
        if (!mut) CHECK(bad == 0, "sim swiglu width %u within 5e-5 (%d bad)", WI, bad); else CHECK(bad > 0, "sim swiglu width %u mutant caught", WI);
        printf("sim swiglu width %u mutant %d: %d/%u outside tolerance\n", WI, mut, bad, WI);
    }
    /* RES rows form: 3 rows of [5120], ADD|NORM, row r at r * 5120 (nw shared); the mutant drops the weight */
    {
        enum { ROWS = 3 };
        static float rx[ROWS * WH], rxr[ROWS * WH], ra[ROWS * WH], ry[ROWS * WH], ryr[ROWS * WH];
        for (int mut = 0; mut < 2; mut++) {
            OmegaDecSpec s; memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_RES; s.mode = OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_NORM | OMEGA_DEC_RES_ROWS; s.mutant = mut; s.width = WH;
            if (kern_prog(&kw[2], &s)) { CHECK(0, "codegen res rows"); return 1; }
            uint32_t sd = 500;
            for (unsigned i = 0; i < ROWS * WH; i++) { rx[i] = frand(&sd, -2, 2); ra[i] = frand(&sd, -1, 1); ry[i] = -7.0f; }
            for (unsigned i = 0; i < WH; i++) nw[i] = frand(&sd, 0.5f, 1.5f);
            for (unsigned r = 0; r < ROWS; r++) {
                float *xx = rxr + r * WH;
                for (unsigned i = 0; i < WH; i++) xx[i] = rx[r * WH + i] + ra[r * WH + i];
                double sq = 0; for (unsigned i = 0; i < WH; i++) sq += (double)xx[i] * xx[i];
                float rs = (float)(1.0 / sqrt(sq / WH + (double)EPS));
                for (unsigned i = 0; i < WH; i++) { float v = xx[i] * rs; v = v * nw[i]; ryr[r * WH + i] = v * SCALE_2P64; }
            }
            uint32_t args[OMEGA_DEC_RES_WORDS] = { 0 };
            put64(args, OMEGA_DEC_RES_X, (uintptr_t)rx); put64(args, OMEGA_DEC_RES_A, (uintptr_t)ra); put64(args, OMEGA_DEC_RES_NW, (uintptr_t)nw);
            put64(args, OMEGA_DEC_RES_Y, (uintptr_t)ry); putf(args, OMEGA_DEC_RES_EPS, EPS); putf(args, OMEGA_DEC_RES_SCALE, SCALE_2P64);
            WsRange r[4] = { { (uintptr_t)rx, (uintptr_t)rx + sizeof rx }, { (uintptr_t)ra, (uintptr_t)ra + sizeof ra },
                             { (uintptr_t)nw, (uintptr_t)nw + sizeof nw }, { (uintptr_t)ry, (uintptr_t)ry + sizeof ry } };
            if (sim_one(&kw[2], args, OMEGA_DEC_RES_WORDS, 1, ROWS, r, 4, "res rows")) return 1;
            int xbad = memcmp(rx, rxr, sizeof rx) != 0, ybad = 0;
            for (unsigned i = 0; i < ROWS * WH; i++) if (!close_rel(ry[i], ryr[i], 2e-5f, 1e-6f)) ybad++;
            if (!mut) { CHECK(!xbad, "sim res rows: x bit-identical"); CHECK(ybad == 0, "sim res rows: y within 2e-5 (%d bad)", ybad); }
            else CHECK(ybad > 0, "sim res rows mutant caught");
            printf("sim res rows %d x %u mutant %d: x %s, y %d/%u outside tolerance\n", ROWS, WH, mut, xbad ? "DIFFERS" : "identical", ybad, ROWS * WH);
        }
        OmegaDecSpec bad; memset(&bad, 0, sizeof bad); bad.kind = OMEGA_DEC_RES; bad.mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_ROWS;
        CHECK(omega_dec_prog_stats(&bad, NULL, NULL) != 0, "refusal res rows with combine");
        bad.mode = OMEGA_DEC_RES_ROWS; CHECK(omega_dec_prog_stats(&bad, NULL, NULL) != 0, "refusal res rows alone");
    }
    /* GATHER: 3 rows, input stride 14343, out width 14336 (the Qwen3.8 attention row), random source table */
    {
        enum { ROWS = 3, GW = 14336, GS = 14343 };
        static float gin[ROWS * GS], gout[ROWS * GW];
        static uint32_t tab[GW];
        for (int mut = 0; mut < 2; mut++) {
            OmegaDecSpec s; memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_GATHER; s.mutant = mut; s.width = GW; s.stride = GS;
            if (kern_prog(&kw[3], &s)) { CHECK(0, "codegen gather"); return 1; }
            uint32_t sd = 600;
            for (unsigned i = 0; i < ROWS * GS; i++) gin[i] = frand(&sd, -3, 3);
            for (unsigned i = 0; i < GW; i++) { sd = sd * 1664525u + 1013904223u; tab[i] = sd % (GS - 1); }
            for (unsigned i = 0; i < ROWS * GW; i++) gout[i] = -7.0f;
            uint32_t args[OMEGA_DEC_GA_WORDS] = { 0 };
            put64(args, OMEGA_DEC_GA_IN, (uintptr_t)gin); put64(args, OMEGA_DEC_GA_TAB, (uintptr_t)tab); put64(args, OMEGA_DEC_GA_OUT, (uintptr_t)gout);
            WsRange r[3] = { { (uintptr_t)gin, (uintptr_t)gin + sizeof gin }, { (uintptr_t)tab, (uintptr_t)tab + sizeof tab }, { (uintptr_t)gout, (uintptr_t)gout + sizeof gout } };
            if (sim_one(&kw[3], args, OMEGA_DEC_GA_WORDS, (GW + OMEGA_DEC_THREADS - 1) / OMEGA_DEC_THREADS, ROWS, r, 3, "gather")) return 1;
            int bad = 0;
            for (unsigned rr = 0; rr < ROWS; rr++) for (unsigned n = 0; n < GW; n++) if (memcmp(&gout[rr * GW + n], &gin[rr * GS + tab[n]], 4)) bad++;
            if (!mut) CHECK(bad == 0, "sim gather bit-exact (%d bad)", bad); else CHECK(bad > 0, "sim gather mutant caught");
            printf("sim gather %d x %u mutant %d: %d/%u wrong\n", ROWS, GW, mut, bad, ROWS * GW);
        }
        OmegaDecSpec bad; memset(&bad, 0, sizeof bad); bad.kind = OMEGA_DEC_GATHER; bad.width = 0; bad.stride = 1;
        CHECK(omega_dec_prog_stats(&bad, NULL, NULL) != 0, "refusal gather width 0");
    }
    /* GATE in place over 6144 + 100 values (a tail CTA), mutant drops the gate */
    {
        enum { GN = 6244 };
        static float a[GN], g[GN], ref[GN];
        for (int mut = 0; mut < 2; mut++) {
            OmegaDecSpec s; memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_GATE; s.mutant = mut;
            if (kern_prog(&kw[3], &s)) { CHECK(0, "codegen gate"); return 1; }
            uint32_t sd = 700;
            for (unsigned i = 0; i < GN; i++) { a[i] = frand(&sd, -3, 3) * SCALE_2P64; g[i] = frand(&sd, -8, 8); ref[i] = a[i] * (1.0f / (1.0f + expf(-g[i]))); }
            uint32_t args[OMEGA_DEC_GT_WORDS] = { 0 };
            put64(args, OMEGA_DEC_GT_A, (uintptr_t)a); put64(args, OMEGA_DEC_GT_G, (uintptr_t)g); put64(args, OMEGA_DEC_GT_OUT, (uintptr_t)a);
            args[(OMEGA_DEC_GT_N - 0x380) / 4] = GN;
            WsRange r[2] = { { (uintptr_t)a, (uintptr_t)a + sizeof a }, { (uintptr_t)g, (uintptr_t)g + sizeof g } };
            if (sim_one(&kw[3], args, OMEGA_DEC_GT_WORDS, (GN + OMEGA_DEC_THREADS - 1) / OMEGA_DEC_THREADS, 1, r, 2, "gate")) return 1;
            int bad = 0; for (unsigned i = 0; i < GN; i++) if (!close_rel(a[i], ref[i], 5e-5f, 1e-6f)) bad++;
            if (!mut) CHECK(bad == 0, "sim gate within 5e-5 (%d bad)", bad); else CHECK(bad > 0, "sim gate mutant caught");
            printf("sim gate %u mutant %d: %d/%u outside tolerance\n", GN, mut, bad, GN);
        }
    }
    /* SWIGLU over rows: the group index is the row (gu [3][2 * 17408] -> act [3][17408]) */
    {
        enum { ROWS = 3 };
        static float rgu[ROWS * 2 * WI], ract[ROWS * WI];
        OmegaDecSpec s; memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_SWIGLU; s.width = WI;
        if (kern_prog(&kw[1], &s)) { CHECK(0, "codegen swiglu rows"); return 1; }
        uint32_t sd = 800;
        for (unsigned i = 0; i < ROWS * 2 * WI; i++) rgu[i] = frand(&sd, -4, 4);
        uint32_t args[OMEGA_DEC_SW_WORDS] = { 0 }; put64(args, OMEGA_DEC_SW_GU, (uintptr_t)rgu); put64(args, OMEGA_DEC_SW_ACT, (uintptr_t)ract); putf(args, OMEGA_DEC_SW_SCALE, 1.0f);
        WsRange r[2] = { { (uintptr_t)rgu, (uintptr_t)rgu + sizeof rgu }, { (uintptr_t)ract, (uintptr_t)ract + sizeof ract } };
        if (sim_one(&kw[1], args, OMEGA_DEC_SW_WORDS, WI / OMEGA_DEC_THREADS, ROWS, r, 2, "swiglu rows")) return 1;
        int bad = 0;
        for (unsigned rr = 0; rr < ROWS; rr++) for (unsigned c = 0; c < WI; c++) {
            float v = rgu[rr * 2 * WI + c], u = rgu[rr * 2 * WI + WI + c], want = v / (1.0f + expf(-v)) * u;
            if (!close_rel(ract[rr * WI + c], want, 5e-5f, 1e-6f)) bad++;
        }
        CHECK(bad == 0, "sim swiglu rows within 5e-5 (%d bad)", bad);
        printf("sim swiglu rows %d x %u: %d outside tolerance\n", ROWS, WI, bad);
    }
    return g_failed ? 1 : 0;
}

/* ---- chip ---- */
static OmegaGpuLaunch axpy_launch(const Kern *k, uint32_t *args, uint32_t stride, uint32_t iters) {
    OmegaGpuLaunch L; memset(&L, 0, sizeof L);
    L.code_va = k->code.va; L.gpr_count = k->gpr; L.threads_x = OMEGA_DEC_THREADS; L.threads_y = 1; L.grid_x = stride / OMEGA_DEC_THREADS; L.grid_y = 1;
    L.num_elements = (size_t)stride * iters; L.args = args; L.n_args = OMEGA_DEC_AXPY_WORDS; L.timeout_ms = 20000;
    return L;
}
static int chip_fail(const char *what) { printf("FAIL: %s: %s\n", what, omega_gpu_session_last_error()); g_checks++; g_failed++; return 1; }

typedef struct { uint32_t L; uint64_t base_ns, pcas_ns, inline_ns, base_wall_ns, pcas_wall_ns, pcas_submits, inline_submits; int pcas_ok, inline_ok; uint64_t stamps_span_ns, stamps_min_ns, stamps_max_ns, stamps_chip_ns; } Row;

static int chip_run(const char *out_path, uint32_t reps) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot write %s\n", out_path); return 1; }
    if (specs_build()) { printf("FAIL: codegen\n"); return 1; }
    OmegaGpuCmdList *cl = NULL;
    if (omega_gpu_cmdlist_create(OMEGA_GPU_CMDLIST_MAX_LAUNCHES, &cl) != 0) return chip_fail("cmdlist create");
    omega_gpu_session_lock();
    Kern *ks[] = { &K_axpy, &K_axpy_big, &K_res_an, &K_res_cn, &K_res_m1, &K_res_m2, &K_res_cn_m1, &K_sw, &K_sw_m1, &K_am0, &K_am1, &K_am_m1 };
    for (unsigned i = 0; i < sizeof ks / sizeof ks[0]; i++) if (kern_upload(ks[i])) { omega_gpu_session_unlock(); return chip_fail("kernel upload"); }
    OmegaGpuMem bx, ba, bx0;
    const size_t big = (size_t)AXPY_BIG_STRIDE * AXPY_BIG_ITERS * 4u;
    if (omega_gpu_session_alloc(big, &bx) || omega_gpu_session_alloc(big, &ba) || omega_gpu_session_alloc(big, &bx0)) { omega_gpu_session_unlock(); return chip_fail("alloc"); }
    uint32_t s = 11; float *x0 = bx0.cpu, *a = ba.cpu, *x = bx.cpu;
    for (size_t i = 0; i < big / 4; i++) { x0[i] = frand(&s, -1, 1); a[i] = frand(&s, -1, 1); }
    omega_gpu_session_sync_to_gpu(&ba);
    uint32_t args[OMEGA_DEC_AXPY_WORDS] = { 0 }; put64(args, OMEGA_DEC_AXPY_X, bx.va); put64(args, OMEGA_DEC_AXPY_A, ba.va); args[(OMEGA_DEC_AXPY_ITERS - 0x380) / 4] = 1;
    OmegaGpuLaunch L = axpy_launch(&K_axpy, args, AXPY_STRIDE, 1);
    static float base[AXPY_STRIDE], xref[AXPY_STRIDE];
    const uint32_t Ls[] = { 64, 289, 1024 };
    Row rows[3]; memset(rows, 0, sizeof rows);
    if (out) fprintf(out, "{\"receipt\":\"OMEGA_GPU_CMDLIST_V1\",\"reps\":%u,\"spin_us\":%u,\"rows\":[", reps, omega_gpu_session_spin_us());
    for (unsigned li = 0; li < 3; li++) {
        const uint32_t N = Ls[li];
        Row *r = &rows[li]; r->L = N;
        memcpy(xref, x0, sizeof xref); for (uint32_t l = 0; l < N; l++) for (unsigned i = 0; i < AXPY_STRIDE; i++) xref[i] = xref[i] + a[i];
        /* baseline: one session launch at a time */
        uint64_t *tb = calloc(reps, 8), *tw = calloc(reps, 8), *tp = calloc(reps, 8), *tpw = calloc(reps, 8), *ti = calloc(reps, 8);
        int base_ok = 1;
        for (uint32_t rep = 0; rep < reps && base_ok; rep++) {
            memcpy(x, x0, sizeof base); omega_gpu_session_sync_to_gpu(&bx);
            uint64_t ns = 0; struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
            for (uint32_t l = 0; l < N; l++) if (omega_gpu_session_launch(&L, &ns, NULL) != 0) { base_ok = 0; chip_fail("baseline launch"); break; }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            omega_gpu_session_sync_to_cpu(&bx);
            if (rep == 0) memcpy(base, x, sizeof base);
            tb[rep] = ns; tw[rep] = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull + (uint64_t)(t1.tv_nsec - t0.tv_nsec);
        }
        if (!base_ok) break;
        CHECK(memcmp(base, xref, sizeof base) == 0, "L=%u baseline x == host reference (bit-identical)", N);
        r->base_ns = median_u64(tb, reps); r->base_wall_ns = median_u64(tw, reps);
        /* main: one list, PCAS */
        OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.mode = OMEGA_GPU_CMDLIST_PCAS; o.timeout_ms = 20000;
        OmegaGpuCmdListStats st; int ok = 1;
        for (uint32_t rep = 0; rep < reps && ok; rep++) {
            memcpy(x, x0, sizeof base); omega_gpu_session_sync_to_gpu(&bx);
            omega_gpu_cmdlist_reset(cl);
            for (uint32_t l = 0; l < N; l++) if (omega_gpu_cmdlist_add(cl, &L) < 0) { ok = 0; chip_fail("cmdlist add"); break; }
            if (!ok) break;
            struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
            int rc = omega_gpu_cmdlist_submit(cl, &o, &st);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            omega_gpu_session_sync_to_cpu(&bx);
            if (rc != 0) { ok = 0; printf("PCAS L=%u rep %u: submit rc %d (%s) sems_done %u first_missing %d\n", N, rep, rc, omega_gpu_session_last_error(), st.sems_done, st.first_missing); break; }
            if (memcmp(x, base, sizeof base) != 0) { ok = 0; printf("PCAS L=%u rep %u: x differs from the baseline\n", N, rep); break; }
            tp[rep] = st.elapsed_ns; tpw[rep] = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull + (uint64_t)(t1.tv_nsec - t0.tv_nsec);
            r->pcas_submits = st.submits;
        }
        r->pcas_ok = ok;
        CHECK(ok, "L=%u PCAS list: every rep bit-identical to the baseline, all semaphores DONE", N);
        if (ok) { r->pcas_ns = median_u64(tp, reps); r->pcas_wall_ns = median_u64(tpw, reps); }
        if (omega_gpu_cmdlist_is_latched()) { printf("cmdlist latched; stopping\n"); break; }
        /* instrument: GPU timestamps around every launch (stamps on), x still identical; stamps off -> accessor refuses */
        if (ok) {
            static uint64_t stamps[1025];
            memcpy(x, x0, sizeof base); omega_gpu_session_sync_to_gpu(&bx);
            omega_gpu_cmdlist_reset(cl);
            for (uint32_t l = 0; l < N; l++) (void)omega_gpu_cmdlist_add(cl, &L);
            OmegaGpuCmdListOpts os = o; os.stamps = 1;
            int rc = omega_gpu_cmdlist_submit(cl, &os, &st);
            omega_gpu_session_sync_to_cpu(&bx);
            int cnt = omega_gpu_cmdlist_stamps(cl, stamps, 1025);
            int mono = 1; uint64_t mn = UINT64_MAX, mx = 0;
            for (int i = 1; i < cnt; i++) { if (stamps[i] < stamps[i - 1]) mono = 0; uint64_t d = stamps[i] - stamps[i - 1]; if (d < mn) mn = d; if (d > mx) mx = d; }
            const uint64_t span = cnt > 1 ? stamps[cnt - 1] - stamps[0] : 0;
            CHECK(rc == 0 && memcmp(x, base, sizeof base) == 0 && cnt == (int)N + 1 && mono && span > 0 && span <= st.elapsed_ns,
                  "L=%u stamps on: rc %d, x identical, %d stamps (want %u), monotonic %d, span %.1f us <= chip %.1f us, per launch min %.2f max %.2f us",
                  N, rc, cnt, N + 1, mono, span / 1e3, st.elapsed_ns / 1e3, mn / 1e3, mx / 1e3);
            r->stamps_span_ns = span; r->stamps_min_ns = mn; r->stamps_max_ns = mx; r->stamps_chip_ns = st.elapsed_ns;
            memcpy(x, x0, sizeof base); omega_gpu_session_sync_to_gpu(&bx);
            omega_gpu_cmdlist_reset(cl);
            for (uint32_t l = 0; l < N; l++) (void)omega_gpu_cmdlist_add(cl, &L);
            rc = omega_gpu_cmdlist_submit(cl, &o, &st);
            cnt = omega_gpu_cmdlist_stamps(cl, stamps, 1025);
            CHECK(rc == 0 && cnt < 0, "L=%u stamps off: submit rc %d, stamps accessor refuses (%d)", N, rc, cnt);
        }
        /* INLINE mode (the session's own words per launch, several submissions) */
        o.mode = OMEGA_GPU_CMDLIST_INLINE; ok = 1;
        for (uint32_t rep = 0; rep < reps && ok; rep++) {
            memcpy(x, x0, sizeof base); omega_gpu_session_sync_to_gpu(&bx);
            omega_gpu_cmdlist_reset(cl);
            for (uint32_t l = 0; l < N; l++) if (omega_gpu_cmdlist_add(cl, &L) < 0) { ok = 0; chip_fail("cmdlist add"); break; }
            if (!ok) break;
            int rc = omega_gpu_cmdlist_submit(cl, &o, &st);
            omega_gpu_session_sync_to_cpu(&bx);
            if (rc != 0) { ok = 0; printf("INLINE L=%u rep %u: submit rc %d (%s) sems_done %u first_missing %d\n", N, rep, rc, omega_gpu_session_last_error(), st.sems_done, st.first_missing); break; }
            if (memcmp(x, base, sizeof base) != 0) { ok = 0; printf("INLINE L=%u rep %u: x differs from the baseline\n", N, rep); break; }
            ti[rep] = st.elapsed_ns; r->inline_submits = st.submits;
        }
        r->inline_ok = ok;
        CHECK(ok, "L=%u INLINE list: every rep bit-identical to the baseline, all semaphores DONE", N);
        if (ok) r->inline_ns = median_u64(ti, reps);
        if (omega_gpu_cmdlist_is_latched()) { printf("cmdlist latched; stopping\n"); break; }
        printf("L=%u: baseline %.1f us/launch (chip %.1f us, wall %.1f us per launch); PCAS %s %.2f us/launch (wall %.2f, %" PRIu64 " submits); INLINE %s %.2f us/launch (%" PRIu64 " submits)\n",
               N, r->base_ns / 1e3 / N, r->base_ns / 1e3 / N, r->base_wall_ns / 1e3 / N, r->pcas_ok ? "ok" : "FAIL", r->pcas_ns / 1e3 / N, r->pcas_wall_ns / 1e3 / N, r->pcas_submits,
               r->inline_ok ? "ok" : "FAIL", r->inline_ns / 1e3 / N, r->inline_submits);
        if (out) fprintf(out, "%s{\"L\":%u,\"baseline_ns\":%" PRIu64 ",\"baseline_wall_ns\":%" PRIu64 ",\"pcas_ok\":%d,\"pcas_ns\":%" PRIu64 ",\"pcas_wall_ns\":%" PRIu64 ",\"pcas_submits\":%" PRIu64 ",\"inline_ok\":%d,\"inline_ns\":%" PRIu64 ",\"inline_submits\":%" PRIu64 ",\"stamps_span_ns\":%" PRIu64 ",\"stamps_min_ns\":%" PRIu64 ",\"stamps_max_ns\":%" PRIu64 ",\"stamps_chip_ns\":%" PRIu64 "}",
                         li ? "," : "", N, r->base_ns, r->base_wall_ns, r->pcas_ok, r->pcas_ns, r->pcas_wall_ns, r->pcas_submits, r->inline_ok, r->inline_ns, r->inline_submits, r->stamps_span_ns, r->stamps_min_ns, r->stamps_max_ns, r->stamps_chip_ns);
        free(tb); free(tw); free(tp); free(tpw); free(ti);
    }
    if (out) fprintf(out, "],");
    /* negative control: drop launch 32 of 64 from the stream; the semaphore check must name it */
    if (!omega_gpu_cmdlist_is_latched()) {
        const uint32_t N = 64;
        memcpy(xref, x0, sizeof xref); for (uint32_t l = 0; l + 1 < N; l++) for (unsigned i = 0; i < AXPY_STRIDE; i++) xref[i] = xref[i] + a[i];
        memcpy(x, x0, sizeof base); omega_gpu_session_sync_to_gpu(&bx);
        omega_gpu_cmdlist_reset(cl);
        for (uint32_t l = 0; l < N; l++) omega_gpu_cmdlist_add(cl, &L);
        OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.mode = OMEGA_GPU_CMDLIST_PCAS; o.timeout_ms = 20000; o.skip_launch = 33;
        OmegaGpuCmdListStats st; int rc = omega_gpu_cmdlist_submit(cl, &o, &st);
        omega_gpu_session_sync_to_cpu(&bx);
        CHECK(rc != 0 && st.first_missing == 32 && st.sems_done == N - 1, "negative control: skipped launch 32 caught (rc %d, first_missing %d, sems_done %u)", rc, st.first_missing, st.sems_done);
        CHECK(memcmp(x, xref, sizeof base) == 0, "negative control: x == x0 + 63 a (the other 63 launches ran)");
        printf("negative control skip_launch=33: rc %d first_missing %d sems_done %u/%u x %s\n", rc, st.first_missing, st.sems_done, N, memcmp(x, xref, sizeof base) == 0 ? "x0+63a" : "UNEXPECTED");
        if (out) fprintf(out, "\"skip_control\":{\"rc\":%d,\"first_missing\":%d,\"sems_done\":%u},", rc, st.first_missing, st.sems_done);
    }
    /* informational: no WFI between heavy dependent launches (256 CTAs x 16 iterations each, 32 launches) */
    if (!omega_gpu_cmdlist_is_latched()) {
        const uint32_t N = 32; const size_t n = (size_t)AXPY_BIG_STRIDE * AXPY_BIG_ITERS;
        float *xr = malloc(n * 4); memcpy(xr, x0, n * 4);
        for (uint32_t l = 0; l < N; l++) for (size_t i = 0; i < n; i++) xr[i] = xr[i] + a[i];
        uint32_t ab[OMEGA_DEC_AXPY_WORDS] = { 0 }; put64(ab, OMEGA_DEC_AXPY_X, bx.va); put64(ab, OMEGA_DEC_AXPY_A, ba.va); ab[(OMEGA_DEC_AXPY_ITERS - 0x380) / 4] = AXPY_BIG_ITERS;
        OmegaGpuLaunch LB = axpy_launch(&K_axpy_big, ab, AXPY_BIG_STRIDE, AXPY_BIG_ITERS);
        size_t diff_wfi = 0, diff_none = 0; uint64_t ns_wfi = 0, ns_none = 0; int rc_wfi = -1, rc_none = -1;
        for (int variant = 0; variant < 2; variant++) {
            memcpy(x, x0, n * 4); omega_gpu_session_sync_to_gpu(&bx);
            omega_gpu_cmdlist_reset(cl);
            for (uint32_t l = 0; l < N; l++) omega_gpu_cmdlist_add(cl, &LB);
            OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.mode = OMEGA_GPU_CMDLIST_PCAS; o.timeout_ms = 20000;
            o.barrier = variant ? OMEGA_GPU_CMDLIST_BARRIER_NONE : OMEGA_GPU_CMDLIST_BARRIER_WFI;
            OmegaGpuCmdListStats st; int rc = omega_gpu_cmdlist_submit(cl, &o, &st);
            omega_gpu_session_sync_to_cpu(&bx);
            size_t d = 0; for (size_t i = 0; i < n; i++) if (x[i] != xr[i]) d++;
            if (variant) { diff_none = d; ns_none = st.elapsed_ns; rc_none = rc; } else { diff_wfi = d; ns_wfi = st.elapsed_ns; rc_wfi = rc; }
            if (omega_gpu_cmdlist_is_latched()) break;
        }
        CHECK(rc_wfi == 0 && diff_wfi == 0, "heavy chain with WFI: rc %d, %zu of %zu elements differ", rc_wfi, diff_wfi, n);
        printf("heavy chain (32 x 1M elements): WFI rc %d %zu differ %.1f us; BARRIER_NONE rc %d %zu differ %.1f us (informational: %s)\n",
               rc_wfi, diff_wfi, ns_wfi / 1e3, rc_none, diff_none, ns_none / 1e3,
               diff_none ? "grids overlapped without the barrier: the WFI is load-bearing" : "no difference seen; this does not prove ordering without a barrier");
        if (out) fprintf(out, "\"no_barrier_control\":{\"wfi_rc\":%d,\"wfi_diff\":%zu,\"wfi_ns\":%" PRIu64 ",\"none_rc\":%d,\"none_diff\":%zu,\"none_ns\":%" PRIu64 "},", rc_wfi, diff_wfi, ns_wfi, rc_none, diff_none, ns_none);
        free(xr);
    }
    /* glue kernels in one list: RES ADD|NORM, SWIGLU, RES COMBINE|NORM, ARGMAX x2; then mutants */
    if (!omega_gpu_cmdlist_is_latched()) {
        OmegaGpuMem m; if (omega_gpu_session_alloc(4u * (H + TK * H + TK + H + H + TK * 2 * INTER + TK * INTER + AM_PAD + 2 * OMEGA_DEC_ARGMAX_SPAN + 2 * OMEGA_DEC_ARGMAX_SPAN), &m)) { omega_gpu_session_unlock(); return chip_fail("alloc glue"); }
        float *p = m.cpu; uint64_t va = m.va;
        Glue g; g.x = p; g.a8 = p + H; g.w = g.a8 + TK * H; g.nw = g.w + TK; g.y = g.nw + H; g.gu = g.y + H; g.act = g.gu + TK * 2 * INTER; g.logits = g.act + TK * INTER;
        g.ov = g.logits + AM_PAD; g.oi = (uint32_t *)(g.ov + OMEGA_DEC_ARGMAX_SPAN); float *ov2 = (float *)(g.oi + OMEGA_DEC_ARGMAX_SPAN); uint32_t *oi2 = (uint32_t *)(ov2 + OMEGA_DEC_ARGMAX_SPAN);
#define VA(ptr) (va + (uint64_t)((const uint8_t *)(ptr) - (const uint8_t *)p))
        static float x_after_an[H], y_an[H], y_cn[H], act_ref[TK * INTER], x_after_cn[H];
        /* 0 real; 1 norm-without-weight in slot 2 (the norm whose output is compared: slot 0 norm output is overwritten by
         * slot 2, chip run r2 showed a slot-0 mutant cannot show); 2 combine skips expert 7 in slot 2; 3 argmax mutant */
        for (int mut = 0; mut < 4; mut++) {
            for (int tie = mut == 0 ? 0 : 1; tie < 2; tie++) { /* mutants run on the tie case: the max sits in a j == 7 slot */
                glue_fill(&g, 400 + tie, tie);
                memcpy(x_after_an, g.x, sizeof x_after_an); ref_res(OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_NORM, x_after_an, g.a8, g.w, g.nw, y_an, EPS, 1.0f);
                ref_swiglu(g.gu, act_ref, 1.0f);
                memcpy(x_after_cn, x_after_an, sizeof x_after_cn); ref_res(OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM, x_after_cn, g.a8, g.w, g.nw, y_cn, EPS, SCALE_2P64);
                omega_gpu_session_sync_to_gpu(&m);
                uint32_t ra[OMEGA_DEC_RES_WORDS] = { 0 }, rb[OMEGA_DEC_RES_WORDS] = { 0 }, sa[OMEGA_DEC_SW_WORDS] = { 0 }, a0[OMEGA_DEC_AM_WORDS] = { 0 }, a1[OMEGA_DEC_AM_WORDS] = { 0 };
                put64(ra, OMEGA_DEC_RES_X, VA(g.x)); put64(ra, OMEGA_DEC_RES_A, VA(g.a8)); put64(ra, OMEGA_DEC_RES_W, VA(g.w)); put64(ra, OMEGA_DEC_RES_NW, VA(g.nw)); put64(ra, OMEGA_DEC_RES_Y, VA(g.y)); putf(ra, OMEGA_DEC_RES_EPS, EPS); putf(ra, OMEGA_DEC_RES_SCALE, 1.0f);
                memcpy(rb, ra, sizeof rb); putf(rb, OMEGA_DEC_RES_SCALE, SCALE_2P64);
                put64(sa, OMEGA_DEC_SW_GU, VA(g.gu)); put64(sa, OMEGA_DEC_SW_ACT, VA(g.act)); putf(sa, OMEGA_DEC_SW_SCALE, 1.0f);
                put64(a0, OMEGA_DEC_AM_V, VA(g.logits)); put64(a0, OMEGA_DEC_AM_I, 0); put64(a0, OMEGA_DEC_AM_OV, VA(g.ov)); put64(a0, OMEGA_DEC_AM_OI, VA(g.oi));
                put64(a1, OMEGA_DEC_AM_V, VA(g.ov)); put64(a1, OMEGA_DEC_AM_I, VA(g.oi)); put64(a1, OMEGA_DEC_AM_OV, VA(ov2)); put64(a1, OMEGA_DEC_AM_OI, VA(oi2));
                OmegaGpuLaunch l[5]; memset(l, 0, sizeof l);
                for (int i = 0; i < 5; i++) { l[i].threads_x = OMEGA_DEC_THREADS; l[i].threads_y = 1; l[i].grid_y = 1; l[i].timeout_ms = 20000; }
                Kern *k0 = &K_res_an, *k2 = mut == 1 ? &K_res_cn_m1 : mut == 2 ? &K_res_m2 : &K_res_cn, *k3 = mut == 3 ? &K_am_m1 : &K_am0;
                l[0].code_va = k0->code.va; l[0].gpr_count = k0->gpr; l[0].grid_x = 1; l[0].num_elements = H; l[0].args = ra; l[0].n_args = OMEGA_DEC_RES_WORDS;
                l[1].code_va = K_sw.code.va; l[1].gpr_count = K_sw.gpr; l[1].grid_x = INTER / OMEGA_DEC_THREADS; l[1].grid_y = TK; l[1].num_elements = TK * INTER; l[1].args = sa; l[1].n_args = OMEGA_DEC_SW_WORDS;
                l[2].code_va = k2->code.va; l[2].gpr_count = k2->gpr; l[2].grid_x = 1; l[2].num_elements = H; l[2].args = rb; l[2].n_args = OMEGA_DEC_RES_WORDS;
                l[3].code_va = k3->code.va; l[3].gpr_count = k3->gpr; l[3].grid_x = AM_CTAS; l[3].num_elements = AM_PAD; l[3].args = a0; l[3].n_args = OMEGA_DEC_AM_WORDS;
                l[4].code_va = K_am1.code.va; l[4].gpr_count = K_am1.gpr; l[4].grid_x = 1; l[4].num_elements = OMEGA_DEC_ARGMAX_SPAN; l[4].args = a1; l[4].n_args = OMEGA_DEC_AM_WORDS;
                omega_gpu_cmdlist_reset(cl);
                int ok = 1; for (int i = 0; i < 5; i++) if (omega_gpu_cmdlist_add(cl, &l[i]) < 0) { ok = 0; chip_fail("glue add"); }
                if (!ok) break;
                OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.mode = OMEGA_GPU_CMDLIST_PCAS; o.timeout_ms = 20000;
                OmegaGpuCmdListStats st; int rc = omega_gpu_cmdlist_submit(cl, &o, &st);
                omega_gpu_session_sync_to_cpu(&m);
                int xbad = memcmp(g.x, x_after_cn, sizeof x_after_cn) != 0, ybad = 0, abad = 0;
                for (unsigned i = 0; i < H; i++) if (!close_rel(g.y[i], y_cn[i], 2e-5f, 1e-6f)) ybad++;
                for (unsigned i = 0; i < TK * INTER; i++) if (!close_rel(g.act[i], act_ref[i], 5e-5f, 1e-6f)) abad++;
                int ambad = oi2[0] != g.am_ref;
                if (mut == 0) {
                    CHECK(rc == 0, "glue list rc %d (%s)", rc, rc ? omega_gpu_session_last_error() : "");
                    CHECK(!xbad, "glue tie %d: residual x after ADD then COMBINE bit-identical to the host reference", tie);
                    CHECK(ybad == 0, "glue tie %d: final RMSNorm (scale 2^64) within 2e-5 (%d bad)", tie, ybad);
                    CHECK(abad == 0, "glue tie %d: SwiGLU within 5e-5 (%d bad)", tie, abad);
                    CHECK(!ambad, "glue tie %d: argmax %u == host %u (value %g)", tie, oi2[0], g.am_ref, ov2[0]);
                    printf("glue list tie %d: rc %d, 5 launches in %u submit(s) %.1f us chip; x %s, y %d bad, act %d bad, argmax %u (ref %u)\n", tie, rc, st.submits, st.elapsed_ns / 1e3, xbad ? "DIFFERS" : "identical", ybad, abad, oi2[0], g.am_ref);
                    if (out) fprintf(out, "\"glue_tie%d\":{\"rc\":%d,\"chip_ns\":%" PRIu64 ",\"x_identical\":%d,\"y_bad\":%d,\"act_bad\":%d,\"argmax\":%u,\"argmax_ref\":%u},", tie, rc, st.elapsed_ns, !xbad, ybad, abad, oi2[0], g.am_ref);
                } else {
                    int caught = mut == 3 ? ambad : (xbad || ybad);
                    CHECK(rc == 0 && caught, "chip mutant %d caught (rc %d, x %s, y %d bad, argmax %u ref %u)", mut, rc, xbad ? "differs" : "same", ybad, oi2[0], g.am_ref);
                    printf("chip mutant %d: rc %d, x %s, y %d bad, argmax %u (ref %u): %s\n", mut, rc, xbad ? "differs" : "same", ybad, oi2[0], g.am_ref, caught ? "caught" : "NOT CAUGHT");
                }
                if (omega_gpu_cmdlist_is_latched()) break;
            }
            if (omega_gpu_cmdlist_is_latched()) break;
        }
#undef VA
        omega_gpu_session_free(&m);
    }
    omega_gpu_session_free(&bx); omega_gpu_session_free(&ba); omega_gpu_session_free(&bx0);
    omega_gpu_session_unlock();
    omega_gpu_cmdlist_free(cl);
    if (out) { fprintf(out, "\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS"); fclose(out); }
    return g_failed ? 1 : 0;
}

int main(int argc, char **argv) {
    int host = 0, sim = 0; uint32_t reps = 20; const char *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--spin-us") == 0 && i + 1 < argc) omega_gpu_session_set_spin_us((uint32_t)atoi(argv[++i]));
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out = argv[++i];
        else { printf("usage: %s [--host-only|--sim] [--reps N] [--spin-us U] [--out receipt.json]\n", argv[0]); return 2; }
    }
    if (reps == 0) reps = 1;
    int rc = host ? host_only() : sim ? (sim_mode() | sim_wide()) : chip_run(out, reps);
    printf("%s: %d checks, %d failed: %s\n", host ? "host-only" : sim ? "sim" : "chip", g_checks, g_failed, rc ? "FAIL" : "PASS");
    return rc;
}
