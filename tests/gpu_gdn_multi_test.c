/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * gpu_gdn_multi_test: the multi-token Gated DeltaNet decode (L369, omega_gpu_gdn_api.h) must be the decode step T times,
 * bit for bit: every output row, the conv tail and the state after the last token.
 *   --sim         warp simulator over the whole grid: T decode launches against one multi launch (both fragment orders),
 *                 T 1, 2, 5; controls: a zero row stride and mutant 4 (tail not shifted) must differ. Then the bf16 state
 *                 (DRA-76): one bf16-state multi launch == the f32 multi launch from the widened state, the final state
 *                 rounded to nearest even (outputs and tails bit for bit, T 1, 2, 5, both orders); controls: mutant 1 with
 *                 the bf16 state changes the outputs, mutant 5 (OMEGA_GDN_MUT_BF16_TRUNC) stores rn & 0xfff0 from the
 *                 multi and the decode launch. Then the host checks below.
 *   --host-only   (DRA-76) codegen: the bf16-state multi program builds, fits and encodes with every placeholder patched;
 *                 mutant 5 accepted with the bf16 state by all five GDN programs and refused without it;
 *                 omega_gpu_gdn_state_env parses AIEN_OMEGA_GDN_STATE_BF16 / _MUTANT as alloc does. No device.
 *   --bf16        the chip, bf16 state through the API (DRA-76): gdns allocated under AIEN_OMEGA_GDN_STATE_BF16=1 (and
 *                 AIEN_OMEGA_GDN_STATE_MUTANT 1 / 5, set here) against an f32 gdn from the widened state, T 2, 37, 128:
 *                 the multi launch's outputs and tail bit for bit, its state == the rounded f32 state; mutant 1 differs,
 *                 mutant 5 stores the truncated state. Run it with both variables unset in the environment.
 *   (default)     the chip: T decode steps (omega_gpu_gdn_step) against one recorded multi launch from the same nonzero
 *                 state, T 1, 2, 37, 128, two input scales; the same controls must differ. f32 state only: under
 *                 AIEN_OMEGA_GDN_STATE_BF16=1 the decode rounds the state every token and the multi once, so they differ.
 * Rows as the 35B graph lays them out: token t's qkv | z in one row (stride (conv + vdim) * 4 bytes), ab rows of vh * 8
 * bytes, out rows of vdim * 4 bytes. Verdict line last: "gpu_gdn_multi_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_gpu_gdn_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_codegen_tc.h"
#include "bw_warp_sim_k.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GDN_TEST_RATIO
#define GDN_TEST_RATIO 2u
#endif
#define RT (GDN_TEST_RATIO ? GDN_TEST_RATIO : 3u)
#define KH OMEGA_GDN_KEY_HEADS
#define HD OMEGA_GDN_HEAD_DIM
#define VH (KH * RT)
#define VD (VH * HD)
#define CD (2u * KH * HD + VD)
#define NS (VH * HD * HD)
#define NT (3u * CD)
#define ROW ((CD + VD) * 4u)      /* qkv | z */
#define ABROW (VH * 8u)
#define OROW (VD * 4u)
#define P_SCAL (CD * OMEGA_GDN_TAPS * 4u)
#define P_NORM (P_SCAL + VH * 8u)
#define P_BYTES (P_NORM + HD * 4u)
#define TAIL_BYTES (VH * OMEGA_GDN_TAIL_HEAD_BYTES)
#define STATE_BYTES (VH * OMEGA_GDN_STATE_HEAD_BYTES)
static const float EPS_L2 = 1e-6f, EPS_N = 1e-6f;

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }

typedef struct {
    float *conv_w, a_log[VH], dtb[VH], nw[HD];
    float *tail0, *state0;  /* reference layouts: tail [3][conv], state [vh][128][128] */
    float *rows, *ab;       /* T rows of qkv | z (true values / in_scale), T rows of {a, b} per head */
} Case;
static void case_make(Case *c, uint32_t T, uint32_t seed, float in_scale) {
    uint32_t s = seed;
    c->conv_w = malloc((size_t)CD * 16); c->tail0 = malloc((size_t)NT * 4); c->state0 = malloc((size_t)NS * 4);
    c->rows = malloc((size_t)T * ROW); c->ab = malloc((size_t)T * ABROW);
    for (size_t i = 0; i < (size_t)CD * 4; i++) c->conv_w[i] = frand(&s, -0.6f, 0.6f);
    for (unsigned h = 0; h < VH; h++) { c->a_log[h] = frand(&s, -1.5f, 1.5f); c->dtb[h] = frand(&s, -3.0f, 1.0f); }
    for (unsigned i = 0; i < HD; i++) c->nw[i] = frand(&s, 0.4f, 1.6f);
    for (size_t i = 0; i < NT; i++) c->tail0[i] = frand(&s, -2.0f, 2.0f);
    for (size_t i = 0; i < NS; i++) c->state0[i] = frand(&s, -0.2f, 0.2f);
    for (uint32_t t = 0; t < T; t++) {
        float *r = c->rows + (size_t)t * (CD + VD);
        for (unsigned i = 0; i < CD; i++) r[i] = frand(&s, -2.5f, 2.5f) / in_scale;
        for (unsigned i = 0; i < VD; i++) r[CD + i] = frand(&s, -3.0f, 3.0f) / in_scale;
        float *a = c->ab + (size_t)t * VH * 2;
        for (unsigned i = 0; i < 2 * VH; i++) a[i] = frand(&s, -4.0f, 4.0f);
        if (t % 3 == 0) { a[0] = 25.0f; a[2] = 19.9f; a[4] = -30.0f; a[3] = 12.0f; a[5] = -12.0f; } /* softplus threshold, beta ~ 0 / 1 */
    }
}
static void case_free(Case *c) { free(c->conv_w); free(c->tail0); free(c->state0); free(c->rows); free(c->ab); }
static size_t first_diff(const void *a, const void *b, size_t words) {
    const uint32_t *x = a, *y = b;
    for (size_t i = 0; i < words; i++) if (x[i] != y[i]) return i;
    return (size_t)-1;
}

/* ---- simulator ---- */
typedef struct { uint32_t off; void *p; size_t bytes; } SimBuf;
static int sim_run(const BlackwellIRProgram *prog, int order, const SimBuf *b, int nb, const uint32_t *words, uint32_t nwords,
                   char *msg, size_t n) {
    static uint8_t smem[OMEGA_GDN_SHARED_BYTES];
    WsSim s; memset(&s, 0, sizeof s);
    s.p = prog; s.threads = OMEGA_GDN_THREADS; s.order = order;
    memset(smem, 0xcd, sizeof smem);
    s.smem = smem; s.smem_bytes = OMEGA_GDN_SHARED_BYTES;
    for (int i = 0; i < nb; i++) {
        const uint64_t a = (uintptr_t)b[i].p;
        s.cbank[b[i].off / 4] = (uint32_t)a; s.cbank[b[i].off / 4 + 1] = (uint32_t)(a >> 32);
        s.ranges[i].lo = a; s.ranges[i].hi = a + b[i].bytes;
    }
    s.nranges = nb;
    for (uint32_t i = 0; i < nwords; i++) s.cbank[OMEGA_GDN_ARG_EPS_L2 / 4 + i] = words[i];
    int err = ws_run(&s, RT, OMEGA_GDN_GRID_Y);
    if (err) snprintf(msg, n, "%.150s (pc %d)", s.msg, s.err_pc);
    return err;
}
/* bf16 state helpers: round to nearest even (the F2FP the bf16 store emits) and the exact widen */
static uint16_t t_bf16_rn(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float t_bf16_f(uint16_t b) { const uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
/* T decode launches (multi = 0) or one multi launch over the case; out [T][VD], tail and state device layouts. The state
 * starts at init (NULL: the case's state0), rounded to bf16 when bf16 (state then holds NS uint16 in the same order). */
static int sim_path(const Case *c, uint32_t T, int multi, int mutant, int bf16, const float *init, uint32_t xstride, float in_scale,
                    int order, float *out, uint8_t *tail, uint8_t *state, char *msg, size_t n) {
    static BlackwellIRProgram prog;
    OmegaGdnSpec spec = { .mutant = mutant, .ratio = GDN_TEST_RATIO, .state_bf16 = (uint32_t)bf16 };
    if ((multi ? omega_blackwell_codegen_gdn_multi_prog(&spec, &prog) : omega_blackwell_codegen_gdn_prog(&spec, &prog)) != 0) {
        snprintf(msg, n, "codegen refused"); return -1;
    }
    uint8_t *par = aligned_alloc(256, (P_BYTES + 255) & ~255u);
    omega_gdn_pack_params_r(GDN_TEST_RATIO, c->conv_w, c->a_log, c->dtb, c->nw, par);
    omega_gdn_pack_tail_r(GDN_TEST_RATIO, c->tail0, (float *)tail);
    const float *s0 = init ? init : c->state0;
    if (bf16) for (size_t i = 0; i < NS; i++) ((uint16_t *)state)[i] = t_bf16_rn(s0[i]);
    else memcpy(state, s0, STATE_BYTES);
    const float one = 1.0f;
    uint32_t w[10];
    memcpy(&w[0], &EPS_L2, 4); memcpy(&w[1], &EPS_N, 4); memcpy(&w[2], &in_scale, 4); memcpy(&w[3], &one, 4);
    int err = 0;
    for (uint32_t t = 0; t < (multi ? 1u : T) && !err; t++) {
        const uint8_t *row = (const uint8_t *)c->rows + (size_t)t * ROW;
        SimBuf b[7] = {
            { OMEGA_GDN_ARG_QKV, (void *)row, (size_t)(multi ? T : 1) * ROW }, { OMEGA_GDN_ARG_Z, (void *)(row + CD * 4u), (size_t)(multi ? T : 1) * ROW - CD * 4u },
            { OMEGA_GDN_ARG_AB, (uint8_t *)c->ab + (size_t)t * ABROW, (size_t)(multi ? T : 1) * ABROW }, { OMEGA_GDN_ARG_PAR, par, P_BYTES },
            { OMEGA_GDN_ARG_TAIL, tail, TAIL_BYTES }, { OMEGA_GDN_ARG_STATE, state, bf16 ? STATE_BYTES / 2u : STATE_BYTES },
            { OMEGA_GDN_ARG_OUT, (uint8_t *)out + (size_t)t * OROW, (size_t)(multi ? T : 1) * OROW },
        };
        w[4] = T; w[5] = xstride; w[6] = ABROW; w[7] = OROW; /* the multi words follow OUT_SCALE (0x3c4): 0x3c8.. */
        err = sim_run(&prog, order, b, 7, w, multi ? 8u : 4u, msg, n);
    }
    free(par);
    return err;
}
static void sim_battery(void) {
    const uint32_t Ts[] = { 1, 2, 5 };
    float *oa = malloc((size_t)5 * OROW), *ob = malloc((size_t)5 * OROW);
    uint8_t *ta = malloc(TAIL_BYTES), *tb = malloc(TAIL_BYTES), *sa = malloc(STATE_BYTES), *sb = malloc(STATE_BYTES);
    char msg[256] = "";
    for (unsigned i = 0; i < 3; i++) {
        const uint32_t T = Ts[i];
        Case c; case_make(&c, T, 77 + T, 1.0f);
        int e1 = sim_path(&c, T, 0, 0, 0, NULL, ROW, 1.0f, 0, oa, ta, sa, msg, sizeof msg);
        CHECK(e1 == 0, "sim decode x %u ran (%s)", T, msg);
        for (int order = 0; order < 2 && e1 == 0; order++) {
            int e2 = sim_path(&c, T, 1, 0, 0, NULL, ROW, 1.0f, order, ob, tb, sb, msg, sizeof msg);
            const size_t dO = first_diff(oa, ob, (size_t)T * VD), dT = first_diff(ta, tb, TAIL_BYTES / 4), dS = first_diff(sa, sb, STATE_BYTES / 4);
            printf("sim T %u order %d: multi %s | out %s tail %s state %s\n", T, order, e2 ? msg : "ran", dO == (size_t)-1 ? "identical" : "DIFFER",
                   dT == (size_t)-1 ? "identical" : "DIFFER", dS == (size_t)-1 ? "identical" : "DIFFER");
            CHECK(e2 == 0 && dO == (size_t)-1 && dT == (size_t)-1 && dS == (size_t)-1, "sim T %u order %d: multi == %u decode steps bit for bit", T, order, T);
        }
        if (T >= 2 && e1 == 0) {
            int e3 = sim_path(&c, T, 1, 0, 0, NULL, 0, 1.0f, 0, ob, tb, sb, msg, sizeof msg);
            CHECK(e3 == 0 && first_diff(oa, ob, (size_t)T * VD) != (size_t)-1, "sim T %u control: row stride 0 differs", T);
            int e4 = sim_path(&c, T, 1, 4, 0, NULL, ROW, 1.0f, 0, ob, tb, sb, msg, sizeof msg);
            CHECK(e4 == 0 && first_diff(oa, ob, (size_t)T * VD) != (size_t)-1, "sim T %u control: mutant 4 (tail not shifted) differs", T);
        }
        case_free(&c);
    }
    free(oa); free(ob); free(ta); free(tb); free(sa); free(sb);
}
/* DRA-76: the bf16 state. The multi launch widens the state once, runs the f32 recurrence over the T tokens and rounds once,
 * so it must be the f32 multi launch from the widened state, its final state rounded to nearest even. */
static size_t state_off(const float *f32, const uint16_t *b16, int trunc, size_t *moved) {
    size_t bad = 0, mv = 0;
    for (size_t k = 0; k < NS; k++) {
        const uint16_t r = t_bf16_rn(f32[k]), want = trunc ? (uint16_t)(r & 0xfff0u) : r;
        bad += b16[k] != want; mv += b16[k] != r;
    }
    if (moved) *moved = mv;
    return bad;
}
static void sim_bf16_battery(void) {
    const uint32_t Ts[] = { 1, 2, 5 };
    float *oa = malloc((size_t)5 * OROW), *ob = malloc((size_t)5 * OROW), *wide = malloc((size_t)NS * 4);
    uint8_t *ta = malloc(TAIL_BYTES), *tb = malloc(TAIL_BYTES), *sa = malloc(STATE_BYTES), *sb = malloc(STATE_BYTES);
    const float *fa = (const float *)sa; const uint16_t *hb = (const uint16_t *)sb;
    char msg[256] = "";
    for (unsigned i = 0; i < 3; i++) {
        const uint32_t T = Ts[i];
        Case c; case_make(&c, T, 177 + T, 1.0f);
        for (size_t k = 0; k < NS; k++) wide[k] = t_bf16_f(t_bf16_rn(c.state0[k]));
        const int e1 = sim_path(&c, T, 1, 0, 0, wide, ROW, 1.0f, 0, oa, ta, sa, msg, sizeof msg);
        CHECK(e1 == 0, "sim bf16 T %u: the f32 multi launch from the widened state ran (%s)", T, msg);
        if (e1) { case_free(&c); continue; }
        for (int order = 0; order < 2; order++) {
            const int e2 = sim_path(&c, T, 1, 0, 1, NULL, ROW, 1.0f, order, ob, tb, sb, msg, sizeof msg);
            const size_t dO = first_diff(oa, ob, (size_t)T * VD), dT = first_diff(ta, tb, TAIL_BYTES / 4), bad = e2 ? NS : state_off(fa, hb, 0, NULL);
            printf("sim bf16 T %u order %d: multi %s | out %s tail %s state %zu of %u off rn(f32)\n", T, order, e2 ? msg : "ran",
                   dO == (size_t)-1 ? "identical" : "DIFFER", dT == (size_t)-1 ? "identical" : "DIFFER", bad, (unsigned)NS);
            CHECK(e2 == 0 && dO == (size_t)-1 && dT == (size_t)-1 && bad == 0,
                  "sim bf16 T %u order %d: bf16 multi == f32 multi from the widened state, state rounded to nearest even", T, order);
        }
        size_t moved = 0;
        const int e5 = sim_path(&c, T, 1, OMEGA_GDN_MUT_BF16_TRUNC, 1, NULL, ROW, 1.0f, 0, ob, tb, sb, msg, sizeof msg);
        const size_t bad5 = e5 ? NS : state_off(fa, hb, 1, &moved);
        CHECK(e5 == 0 && first_diff(oa, ob, (size_t)T * VD) == (size_t)-1 && bad5 == 0 && moved > 0,
              "sim bf16 T %u control: multi mutant 5 keeps the outputs and stores rn & 0xfff0 (%zu of %u words moved)", T, moved, (unsigned)NS);
        const int e6 = sim_path(&c, T, 1, 1, 1, NULL, ROW, 1.0f, 0, ob, tb, sb, msg, sizeof msg);
        CHECK(e6 == 0 && first_diff(oa, ob, (size_t)T * VD) != (size_t)-1, "sim bf16 T %u control: multi mutant 1 (no decay) differs", T);
        /* the decode launch rounds the state every token: one launch is the multi launch with T 1, more drift */
        const int e7 = sim_path(&c, T, 0, 0, 1, NULL, ROW, 1.0f, 0, ob, tb, sb, msg, sizeof msg);
        float dmax = 0.0f;
        for (size_t k = 0; k < (size_t)T * VD && e7 == 0; k++) dmax = fmaxf(dmax, fabsf(oa[k] - ob[k]));
        printf("sim bf16 T %u: %u bf16 decode launches vs the bf16 multi launch: max |out diff| %g\n", T, T, (double)dmax);
        if (T == 1) {
            const size_t bad = e7 ? NS : state_off(fa, hb, 0, NULL);
            CHECK(e7 == 0 && first_diff(oa, ob, VD) == (size_t)-1 && first_diff(ta, tb, TAIL_BYTES / 4) == (size_t)-1 && bad == 0,
                  "sim bf16 T 1: the bf16 decode launch == the bf16 multi launch (out, tail, state)");
            const int e8 = sim_path(&c, 1, 0, OMEGA_GDN_MUT_BF16_TRUNC, 1, NULL, ROW, 1.0f, 0, ob, tb, sb, msg, sizeof msg);
            const size_t bad8 = e8 ? NS : state_off(fa, hb, 1, &moved);
            CHECK(e8 == 0 && first_diff(oa, ob, VD) == (size_t)-1 && bad8 == 0 && moved > 0,
                  "sim bf16 T 1 control: decode mutant 5 keeps the output and stores rn & 0xfff0 (%zu words moved)", moved);
        } else {
            CHECK(e7 == 0, "sim bf16 T %u: the bf16 decode launches ran (%s)", T, msg);
        }
        case_free(&c);
    }
    free(oa); free(ob); free(wide); free(ta); free(tb); free(sa); free(sb);
}

/* ---- DRA-76 host checks: codegen and the environment parse, no device ---- */
typedef int (*GdnGen)(const OmegaGdnSpec *, BlackwellIRProgram *);
static void host_checks(void) {
    static BlackwellIRProgram p, q;
    const GdnGen gen[5] = { omega_blackwell_codegen_gdn_prog, omega_blackwell_codegen_gdn_prep_prog, omega_blackwell_codegen_gdn_scan_prog,
                            omega_blackwell_codegen_gdn_post_prog, omega_blackwell_codegen_gdn_multi_prog };
    static const char *const name[5] = { "decode", "prep", "scan", "post", "multi" };
    for (unsigned k = 0; k < 5; k++) {
        const OmegaGdnSpec t5 = { .mutant = OMEGA_GDN_MUT_BF16_TRUNC, .ratio = GDN_TEST_RATIO, .state_bf16 = 1 };
        const OmegaGdnSpec f5 = { .mutant = OMEGA_GDN_MUT_BF16_TRUNC, .ratio = GDN_TEST_RATIO };
        const OmegaGdnSpec t6 = { .mutant = OMEGA_GDN_MUT_BF16_TRUNC + 1, .ratio = GDN_TEST_RATIO, .state_bf16 = 1 };
        const int a = gen[k](&t5, &p), b = gen[k](&f5, &p), d = gen[k](&t6, &p);
        CHECK(a == 0 && b != 0 && d != 0, "%s codegen: mutant 5 accepted with the bf16 state (%d), refused without it (%d), mutant 6 refused (%d)",
              name[k], a, b, d);
    }
    const OmegaGdnSpec m32 = { .ratio = GDN_TEST_RATIO }, m16 = { .ratio = GDN_TEST_RATIO, .state_bf16 = 1 };
    uint32_t i32 = 0, g32 = 0, i16 = 0, g16 = 0;
    CHECK(omega_gdn_multi_prog_stats(&m32, &i32, &g32) == 0 && omega_gdn_multi_prog_stats(&m16, &i16, &g16) == 0 && g16 > 0 &&
          OMEGA_GDN_THREADS * (g16 + 2u) <= 65536u,
          "bf16 multi program fits the register file (%u insns, %u registers; f32 %u insns, %u registers)", i16, g16, i32, g32);
    const int rq = omega_blackwell_codegen_gdn_multi_prog(&m32, &q), rc = omega_blackwell_codegen_gdn_multi_prog(&m16, &p);
    const size_t need = rc == 0 ? omega_bw_program_code_bytes(&p) : 0;
    uint8_t *code = need ? malloc(need) : NULL;
    OmegaBlackwellKernel kern;
    const int enc = code ? omega_blackwell_kernel_encode_into(&p, code, need, &kern) : -1;
    const int np = enc == 0 ? omega_tc_patch_code(&p, code, need) : -1;
    const uint32_t want = rc == 0 ? omega_gdn_bfly_count(&p) : 0, base = rq == 0 ? omega_gdn_bfly_count(&q) : 0;
    CHECK(rq == 0 && rc == 0 && enc == 0 && np >= 0 && (uint32_t)np == want && want == base + 32u,
          "bf16 multi encodes: %d of %u placeholders patched (f32 %u + 32 F2FP)", np, want, base);
    free(code);
    /* omega_gpu_gdn_state_env: what alloc takes from the environment */
    static const char *const var[2] = { "AIEN_OMEGA_GDN_STATE_BF16", "AIEN_OMEGA_GDN_STATE_MUTANT" };
    static const struct { const char *bf, *mu; int ok, bf16, mut; } ev[] = {
        { NULL, NULL, 1, 0, 0 }, { "1", NULL, 1, 1, 0 }, { "0", "", 1, 0, 0 }, { "10", NULL, 1, 0, 0 }, { "1", "0", 1, 1, 0 },
        { "1", "1", 1, 1, 1 }, { "1", "5", 1, 1, OMEGA_GDN_MUT_BF16_TRUNC }, { "1", "4", 0, 1, 0 }, { "1", "15", 0, 1, 0 },
        { NULL, "1", 0, 0, 0 }, { "true", "5", 0, 0, 0 },
    };
    char *saved[2];
    for (int j = 0; j < 2; j++) { const char *v = getenv(var[j]); saved[j] = v ? strdup(v) : NULL; }
    for (size_t k = 0; k < sizeof ev / sizeof ev[0]; k++) {
        const char *val[2] = { ev[k].bf, ev[k].mu };
        for (int j = 0; j < 2; j++) { if (val[j]) setenv(var[j], val[j], 1); else unsetenv(var[j]); }
        int bf16 = -1, mut = -1;
        const int rc_env = omega_gpu_gdn_state_env(&bf16, &mut);
        CHECK(ev[k].ok ? rc_env == OMEGA_GPU_GDN_OK && bf16 == ev[k].bf16 && mut == ev[k].mut : rc_env == OMEGA_GPU_GDN_BAD_ARGS && mut == 0,
              "state env BF16=%s MUTANT=%s: %s (rc %d, bf16 %d, mutant %d)", val[0] ? val[0] : "(unset)", val[1] ? val[1] : "(unset)",
              ev[k].ok ? "accepted" : "refused", rc_env, bf16, mut);
    }
    for (int j = 0; j < 2; j++) { if (saved[j]) setenv(var[j], saved[j], 1); else unsetenv(var[j]); free(saved[j]); }
}

/* ---- chip ---- */
/* the decode path: T omega_gpu_gdn_step calls; out [T][VD], tail (reference layout) and state after the last */
static int chip_decode(OmegaGpuGdn *g, const Case *c, uint32_t T, float in_scale, float *out, float *tail, float *state) {
    if (omega_gpu_gdn_set_state(g, 0, c->tail0, c->state0) != 0) return -1;
    for (uint32_t t = 0; t < T; t++) {
        const float *r = c->rows + (size_t)t * (CD + VD);
        if (omega_gpu_gdn_step(g, 0, r, r + CD, c->ab + (size_t)t * VH * 2, EPS_L2, EPS_N, in_scale, 1.0f, out + (size_t)t * VD, NULL) != 0) return -1;
    }
    return omega_gpu_gdn_get_state(g, 0, tail, state);
}
static int chip_multi(OmegaGpuGdn *g, const Case *c, uint32_t T, float in_scale, uint32_t xstride, float *out, float *tail, float *state) {
    if (omega_gpu_gdn_set_state(g, 0, c->tail0, c->state0) != 0 || omega_gpu_gdn_multi_prepare(g) != 0) return -1;
    OmegaGpuCmdList *cl = NULL;
    if (omega_gpu_cmdlist_create(4, &cl) != 0) return -1;
    omega_gpu_session_lock();
    OmegaGpuMem m; memset(&m, 0, sizeof m);
    const size_t s_rows = (size_t)T * ROW, s_ab = ((size_t)T * ABROW + 255) & ~(size_t)255, s_out = (size_t)T * OROW;
    int ok = omega_gpu_session_open() != NULL && omega_gpu_session_alloc(s_rows + s_ab + s_out, &m) == 0;
    if (ok) {
        uint8_t *p = m.cpu;
        memcpy(p, c->rows, s_rows); memcpy(p + s_rows, c->ab, (size_t)T * ABROW);
        float *o = (float *)(p + s_rows + s_ab);
        for (size_t i = 0; i < (size_t)T * VD; i++) o[i] = NAN;
        ok = omega_gpu_session_sync_to_gpu(&m) == 0;
    }
    if (ok) ok = omega_gpu_gdn_multi_record(cl, g, 0, T, m.va, m.va + CD * 4u, xstride, m.va + s_rows, ABROW, m.va + s_rows + s_ab, OROW,
                                            EPS_L2, EPS_N, in_scale, 1.0f) == 0;
    OmegaGpuCmdListOpts opts; memset(&opts, 0, sizeof opts);
    OmegaGpuCmdListStats st; memset(&st, 0, sizeof st);
    if (ok) ok = omega_gpu_cmdlist_submit(cl, &opts, &st) == 0 && st.sems_done == st.launches;
    if (ok) ok = omega_gpu_session_sync_to_cpu(&m) == 0;
    if (ok) memcpy(out, (uint8_t *)m.cpu + s_rows + s_ab, s_out);
    if (m.cpu) omega_gpu_session_free(&m);
    omega_gpu_session_unlock();
    omega_gpu_cmdlist_free(cl);
    if (!ok) return -1;
    return omega_gpu_gdn_get_state(g, 0, tail, state);
}
static void chip_battery(void) {
    OmegaGpuGdn *g = NULL;
    CHECK(omega_gpu_gdn_alloc_ratio(1, GDN_TEST_RATIO, &g) == 0, "alloc (%s)", omega_gpu_gdn_last_error());
    if (!g) return;
    const uint32_t Ts[] = { 1, 2, 37, 128 };
    const float scales[] = { 1.0f, 0.0009765625f };
    float *oa = malloc((size_t)128 * OROW), *ob = malloc((size_t)128 * OROW);
    float *ta = malloc((size_t)NT * 4), *tb = malloc((size_t)NT * 4), *sa = malloc(STATE_BYTES), *sb = malloc(STATE_BYTES);
    for (unsigned si = 0; si < 2; si++) for (unsigned i = 0; i < 4; i++) {
        const uint32_t T = Ts[i]; const float sc = scales[si];
        Case c; case_make(&c, T, 900 + 7 * T + si, sc);
        CHECK(omega_gpu_gdn_set_layer(g, 0, c.conv_w, c.a_log, c.dtb, c.nw) == 0, "set layer");
        const int ra = chip_decode(g, &c, T, sc, oa, ta, sa);
        const int rb = chip_multi(g, &c, T, sc, ROW, ob, tb, sb);
        const size_t dO = first_diff(oa, ob, (size_t)T * VD), dT = first_diff(ta, tb, NT), dS = first_diff(sa, sb, NS);
        printf("chip T %3u in_scale %g: decode %s multi %s | out %s tail %s state %s\n", T, (double)sc, ra ? "FAILED" : "ran", rb ? "FAILED" : "ran",
               dO == (size_t)-1 ? "identical" : "DIFFER", dT == (size_t)-1 ? "identical" : "DIFFER", dS == (size_t)-1 ? "identical" : "DIFFER");
        CHECK(ra == 0 && rb == 0, "chip T %u: both paths ran (%s)", T, omega_gpu_gdn_last_error());
        CHECK(ra == 0 && rb == 0 && dO == (size_t)-1 && dT == (size_t)-1 && dS == (size_t)-1, "chip T %u in_scale %g: multi == %u decode steps bit for bit", T, (double)sc, T);
        if (T >= 37 && si == 0) {
            const int r0 = chip_multi(g, &c, T, sc, 0, ob, tb, sb);
            CHECK(r0 == 0 && first_diff(oa, ob, (size_t)T * VD) != (size_t)-1, "chip T %u control: row stride 0 differs", T);
            omega_gpu_gdn_test_set_mutant(4);
            const int r4 = chip_multi(g, &c, T, sc, ROW, ob, tb, sb);
            omega_gpu_gdn_test_set_mutant(0);
            CHECK(r4 == 0 && first_diff(oa, ob, (size_t)T * VD) != (size_t)-1, "chip T %u control: mutant 4 (tail not shifted) differs", T);
        }
        case_free(&c);
    }
    free(oa); free(ob); free(ta); free(tb); free(sa); free(sb);
    omega_gpu_gdn_free(g);
}
/* DRA-76: the bf16 state on the chip, selected through the environment exactly as the 35B selects it */
static int env_alloc(const char *bf, const char *mu, OmegaGpuGdn **g) {
    if (bf) setenv("AIEN_OMEGA_GDN_STATE_BF16", bf, 1); else unsetenv("AIEN_OMEGA_GDN_STATE_BF16");
    if (mu) setenv("AIEN_OMEGA_GDN_STATE_MUTANT", mu, 1); else unsetenv("AIEN_OMEGA_GDN_STATE_MUTANT");
    const int rc = omega_gpu_gdn_alloc_ratio(1, GDN_TEST_RATIO, g);
    unsetenv("AIEN_OMEGA_GDN_STATE_BF16"); unsetenv("AIEN_OMEGA_GDN_STATE_MUTANT");
    return rc;
}
static size_t chip_state_off(const float *f32, const float *wid, int trunc, size_t *moved) {
    size_t bad = 0, mv = 0;
    for (size_t k = 0; k < NS; k++) {
        const uint16_t r = t_bf16_rn(f32[k]);
        const float want = t_bf16_f(trunc ? (uint16_t)(r & 0xfff0u) : r), rn = t_bf16_f(r);
        bad += memcmp(&wid[k], &want, 4) != 0; mv += memcmp(&wid[k], &rn, 4) != 0;
    }
    if (moved) *moved = mv;
    return bad;
}
static void chip_bf16_battery(void) {
    OmegaGpuGdn *gf = NULL, *gb = NULL, *g1 = NULL, *g5 = NULL, *gx = NULL;
    CHECK(env_alloc(NULL, NULL, &gf) == 0 && env_alloc("1", NULL, &gb) == 0 && env_alloc("1", "1", &g1) == 0 && env_alloc("1", "5", &g5) == 0,
          "alloc f32, bf16, bf16 mutant 1, bf16 mutant 5 (%s)", omega_gpu_gdn_last_error());
    CHECK(env_alloc(NULL, "1", &gx) != 0 && gx == NULL, "alloc refuses AIEN_OMEGA_GDN_STATE_MUTANT without the bf16 state");
    const uint32_t Ts[] = { 2, 37, 128 };
    float *oa = malloc((size_t)128 * OROW), *ob = malloc((size_t)128 * OROW), *wide = malloc((size_t)NS * 4);
    float *ta = malloc((size_t)NT * 4), *tb = malloc((size_t)NT * 4), *sa = malloc(STATE_BYTES), *sb = malloc(STATE_BYTES);
    for (unsigned i = 0; i < 3 && gf && gb && g1 && g5; i++) {
        const uint32_t T = Ts[i];
        Case c; case_make(&c, T, 1300 + T, 1.0f);
        for (size_t k = 0; k < NS; k++) wide[k] = t_bf16_f(t_bf16_rn(c.state0[k]));
        Case cw = c; cw.state0 = wide;
        OmegaGpuGdn *all[4] = { gf, gb, g1, g5 };
        for (int j = 0; j < 4; j++) CHECK(omega_gpu_gdn_set_layer(all[j], 0, c.conv_w, c.a_log, c.dtb, c.nw) == 0, "set layer");
        const int ra = chip_multi(gf, &cw, T, 1.0f, ROW, oa, ta, sa), rb = chip_multi(gb, &c, T, 1.0f, ROW, ob, tb, sb);
        const size_t dO = first_diff(oa, ob, (size_t)T * VD), dT = first_diff(ta, tb, NT), bad = ra || rb ? NS : chip_state_off(sa, sb, 0, NULL);
        printf("chip bf16 T %3u: f32 multi %s bf16 multi %s | out %s tail %s state %zu of %u off rn(f32)\n", T, ra ? "FAILED" : "ran", rb ? "FAILED" : "ran",
               dO == (size_t)-1 ? "identical" : "DIFFER", dT == (size_t)-1 ? "identical" : "DIFFER", bad, (unsigned)NS);
        CHECK(ra == 0 && rb == 0 && dO == (size_t)-1 && dT == (size_t)-1 && bad == 0,
              "chip bf16 T %u: bf16 multi == f32 multi from the widened state, state rounded to nearest even (%s)", T, omega_gpu_gdn_last_error());
        size_t moved = 0;
        const int r5 = chip_multi(g5, &c, T, 1.0f, ROW, ob, tb, sb);
        const size_t bad5 = r5 || ra ? NS : chip_state_off(sa, sb, 1, &moved);
        CHECK(r5 == 0 && first_diff(oa, ob, (size_t)T * VD) == (size_t)-1 && bad5 == 0 && moved > 0,
              "chip bf16 T %u control: env mutant 5 keeps the outputs and stores rn & 0xfff0 (%zu words moved)", T, moved);
        const int r1 = chip_multi(g1, &c, T, 1.0f, ROW, ob, tb, sb);
        CHECK(r1 == 0 && first_diff(oa, ob, (size_t)T * VD) != (size_t)-1, "chip bf16 T %u control: env mutant 1 (no decay) differs", T);
        const int rd = chip_decode(gb, &c, T, 1.0f, ob, tb, sb);
        float dmax = 0.0f;
        for (size_t k = 0; k < (size_t)T * VD && rd == 0; k++) dmax = fmaxf(dmax, fabsf(oa[k] - ob[k]));
        printf("chip bf16 T %3u: %u bf16 decode steps vs the bf16 multi launch: max |out diff| %g\n", T, T, (double)dmax);
        CHECK(rd == 0, "chip bf16 T %u: the bf16 decode steps ran", T);
        case_free(&c);
    }
    free(oa); free(ob); free(wide); free(ta); free(tb); free(sa); free(sb);
    if (gf) omega_gpu_gdn_free(gf);
    if (gb) omega_gpu_gdn_free(gb);
    if (g1) omega_gpu_gdn_free(g1);
    if (g5) omega_gpu_gdn_free(g5);
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "";
    const int sim = strcmp(mode, "--sim") == 0, host = strcmp(mode, "--host-only") == 0, bf = strcmp(mode, "--bf16") == 0;
    if (mode[0] && !sim && !host && !bf) { fprintf(stderr, "usage: gpu_gdn_multi_test [--sim | --host-only | --bf16]\n"); return 2; }
    uint32_t insns = 0, gprs = 0;
    OmegaGdnSpec spec = { .ratio = GDN_TEST_RATIO };
    CHECK(omega_gdn_multi_prog_stats(&spec, &insns, &gprs) == 0 && gprs > 0 && OMEGA_GDN_THREADS * (gprs + 2u) <= 65536u,
          "multi program builds and fits the register file (%u insns, %u registers)", insns, gprs);
    if (sim) { sim_battery(); sim_bf16_battery(); host_checks(); }
    else if (host) host_checks();
    else if (bf) chip_bf16_battery();
    else chip_battery();
    printf("gpu_gdn_multi_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
