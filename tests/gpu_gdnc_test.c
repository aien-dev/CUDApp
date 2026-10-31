/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * gpu_gdnc_test (DRA-84): the chunked tensor-core gated-delta-rule prefill scan (omega_blackwell_codegen_gdnc.h) against
 * its golden, the sequential multi-token kernel (omega_blackwell_codegen_gdn_multi_prog), both in the warp simulator.
 *   --sim              T 2, 37 (ragged sub-chunk), 256 (all 32 value heads), 2048 (heads 0..1), 8192 (head 0): every out
 *                      element (both sides rounded to bf16), the final f32 state and conv tail, and every snapshot against
 *                      the multi kernel's state and tail at the same token (the multi kernel run in segments).
 *                      Controls that must go red: mutant 9 (drop-decay), one perturbed input token, bad specs refused.
 *   --sim --mutant 9   the whole battery with the mutant: must exit 1 (the mutant is red).
 *   --sim-oracle       host check of the chip comparison path: gdnc in the simulator, T 256, all 32 value heads, against
 *                      the double oracle the chip arm uses (out, state, tail); the mutant 9 control must go red.
 *   --chip [--tokens T] [--layers N] [--reps R]   the GB10 (defaults T 8192, 32 value heads, N 30, R 3): one GDN layer of
 *                      state per launch, N launches in one command list standing in for the 30 GDN layers (the same
 *                      rows, parameters and initial state in every layer), GPU timestamps around every launch. Layer 0's
 *                      out rows, final state and conv tail against the double oracle (tests/gpu_gdn_test.c oracle_step,
 *                      TOL_OUT / TOL_STATE, tail bit-exact); layer N-1 bit-identical to layer 0; reps 2.. bit-identical
 *                      to rep 1; mutant 9 on layer 0 must go red. Timing line: "GDNC_G3 ..." (30-layer ms against the
 *                      60 ms G3 gate and the 38.3 ms bf16 budget; the slice judges it, the verdict line does not).
 *                      Without SPARK_WINDOW, or when the device does not open: "NOT_RUN" and exit 3, nothing launched.
 *   --chip --mutant 9  every launch is the mutant: must exit 1.
 * Tolerance: relative to the largest reference magnitude of the compared set (TOL_OUT, TOL_STATE; derivation in the
 * DRA-84 PR body, Wolfram). Verdict line last: "gpu_gdnc_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_blackwell_codegen_gdnc.h"
#include "omega_blackwell_codegen_gdn.h"
#include "omega_gpu_gdn_api.h"
#include "omega_gpu_gdnc_api.h"
#include "bw_warp_sim_tc.h"
#include "omega_gpu_session.h"
#include "omega_gpu_cmdlist.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RT 2u
#define KH OMEGA_GDN_KEY_HEADS
#define HD OMEGA_GDN_HEAD_DIM
#define VH (KH * RT)
#define VD (VH * HD)
#define CD (2u * KH * HD + VD)
#define NS (VH * HD * HD)
#define NT (3u * CD)
#define RW (CD + VD)               /* qkv | z elements per row */
#define ROWF (RW * 4u)             /* f32 row bytes (multi) */
#define ROWB (RW * 2u)             /* bf16 row bytes (gdnc) */
#define ABROW (VH * 8u)
#define OROWF (VD * 4u)
#define OROWB (VD * 2u)
#define P_SCAL (CD * OMEGA_GDN_TAPS * 4u)
#define P_NORM (P_SCAL + VH * 8u)
#define P_BYTES (P_NORM + HD * 4u)
#define TAIL_BYTES (VH * OMEGA_GDN_TAIL_HEAD_BYTES)
#define STATE_BYTES (VH * OMEGA_GDN_STATE_HEAD_BYTES)
#define SNAP_BYTES (STATE_BYTES + TAIL_BYTES)
/* Wolfram-derived bounds (PR body): out = bf16 rounding of both sides (2^-8 of the max) plus the bf16 HMMA operands
 * of the WY form, state = the bf16 operands over the 128-term dots; both relative to the max reference magnitude. */
#define TOL_OUT 0.01495 /* 2^-8 + 2^-9 Sqrt[2] Sqrt[16] (Wolfram: 0.0149548) */
#define TOL_STATE 0.01105 /* 2^-9 Sqrt[2] Sqrt[16] (Wolfram: 0.0110485) */
static const float EPS_L2 = 1e-6f, EPS_N = 1e-6f;

static int g_checks, g_failed, g_mutant;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }
static uint16_t bf16_rn(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)(u >> 16);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float bf16_f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

typedef struct {
    uint8_t *par; float *tail0, *state0;
    float *rowsf; uint16_t *rowsb; float *ab;
    float *conv_w, *tailr, a_log[VH], dtb[VH], nw[HD]; /* the reference layouts (the chip arm loads them through the gdn API) */
} Case;
static void case_make(Case *c, uint32_t T, uint32_t seed) {
    uint32_t s = seed;
    float *conv_w = malloc((size_t)CD * 16), *a_log = c->a_log, *dtb = c->dtb, *nw = c->nw;
    float *tail = malloc((size_t)NT * 4);
    for (size_t i = 0; i < (size_t)CD * 4; i++) conv_w[i] = frand(&s, -0.6f, 0.6f);
    for (unsigned h = 0; h < VH; h++) { a_log[h] = frand(&s, -1.5f, 1.0f); dtb[h] = frand(&s, -3.0f, 1.0f); }
    for (unsigned i = 0; i < HD; i++) nw[i] = frand(&s, 0.4f, 1.6f);
    for (size_t i = 0; i < NT; i++) tail[i] = frand(&s, -2.0f, 2.0f);
    c->par = aligned_alloc(256, (P_BYTES + 255) & ~255u);
    c->tail0 = aligned_alloc(256, (TAIL_BYTES + 255) & ~255u);
    c->state0 = aligned_alloc(256, STATE_BYTES);
    omega_gdn_pack_params_r(RT, conv_w, a_log, dtb, nw, c->par);
    omega_gdn_pack_tail_r(RT, tail, c->tail0);
    for (size_t i = 0; i < NS; i++) c->state0[i] = frand(&s, -0.2f, 0.2f);
    c->rowsf = malloc((size_t)T * ROWF); c->rowsb = malloc((size_t)T * ROWB); c->ab = malloc((size_t)T * ABROW);
    for (uint32_t t = 0; t < T; t++) {
        for (unsigned i = 0; i < RW; i++) {
            const float x = i < CD ? frand(&s, -2.5f, 2.5f) : frand(&s, -3.0f, 3.0f);
            const uint16_t b = bf16_rn(x);
            c->rowsb[(size_t)t * RW + i] = b; c->rowsf[(size_t)t * RW + i] = bf16_f(b);
        }
        float *a = c->ab + (size_t)t * VH * 2;
        for (unsigned i = 0; i < 2 * VH; i++) a[i] = frand(&s, -4.0f, 4.0f);
        if (t % 7 == 3) { a[0] = 25.0f; a[2] = 19.9f; a[3] = 12.0f; a[5] = -12.0f; } /* softplus threshold, beta ~ 1 / 0 */
    }
    c->conv_w = conv_w; c->tailr = tail;
}
static void case_free(Case *c) {
    free(c->par); free(c->tail0); free(c->state0); free(c->rowsf); free(c->rowsb); free(c->ab); free(c->conv_w); free(c->tailr);
}

typedef struct { uint32_t off; void *p; size_t bytes; } SimBuf;
static int sim_run(const BlackwellIRProgram *prog, uint32_t threads, uint32_t smem_bytes, const SimBuf *b, int nb,
                   uint32_t w0, const uint32_t *words, uint32_t nwords, uint32_t gx, uint32_t gy, char *msg, size_t n) {
    static uint8_t smem[OMEGA_GDNC_SHARED_BYTES];
    static WsSim s;
    memset(&s, 0, sizeof s);
    s.p = prog; s.threads = threads; s.smem = smem; s.smem_bytes = smem_bytes;
    for (int i = 0; i < nb; i++) {
        const uint64_t a = (uintptr_t)b[i].p;
        s.cbank[b[i].off / 4] = (uint32_t)a; s.cbank[b[i].off / 4 + 1] = (uint32_t)(a >> 32);
        s.ranges[i].lo = a; s.ranges[i].hi = a + b[i].bytes;
    }
    s.nranges = nb;
    for (uint32_t i = 0; i < nwords; i++) s.cbank[w0 / 4 + i] = words[i];
    int err = ws_run(&s, gx, gy);
    if (err) snprintf(msg, n, "%.150s (pc %d)", s.msg, s.err_pc);
    return err;
}

/* golden: the multi kernel over [t0, t1) from the given tail/state (updated in place); out f32 [T][VD] */
static int multi_seg(const Case *c, uint32_t t0, uint32_t t1, uint32_t gx, uint32_t gy, float *outf, float *tail, float *state,
                     char *msg, size_t n) {
    static BlackwellIRProgram prog; static int built;
    if (!built) { OmegaGdnSpec sp = { .ratio = RT }; if (omega_blackwell_codegen_gdn_multi_prog(&sp, &prog) != 0) { snprintf(msg, n, "multi codegen"); return -1; } built = 1; }
    const uint32_t T = t1 - t0;
    const uint8_t *row = (const uint8_t *)c->rowsf + (size_t)t0 * ROWF;
    SimBuf b[7] = {
        { OMEGA_GDN_ARG_QKV, (void *)row, (size_t)T * ROWF }, { OMEGA_GDN_ARG_Z, (void *)(row + CD * 4u), (size_t)T * ROWF - CD * 4u },
        { OMEGA_GDN_ARG_AB, (uint8_t *)c->ab + (size_t)t0 * ABROW, (size_t)T * ABROW }, { OMEGA_GDN_ARG_PAR, c->par, P_BYTES },
        { OMEGA_GDN_ARG_TAIL, tail, TAIL_BYTES }, { OMEGA_GDN_ARG_STATE, state, STATE_BYTES },
        { OMEGA_GDN_ARG_OUT, (uint8_t *)outf + (size_t)t0 * OROWF, (size_t)T * OROWF },
    };
    const float one = 1.0f;
    uint32_t w[8];
    memcpy(&w[0], &EPS_L2, 4); memcpy(&w[1], &EPS_N, 4); memcpy(&w[2], &one, 4); memcpy(&w[3], &one, 4);
    w[4] = T; w[5] = ROWF; w[6] = ABROW; w[7] = OROWF;
    return sim_run(&prog, OMEGA_GDN_THREADS, OMEGA_GDN_SHARED_BYTES, b, 7, OMEGA_GDN_ARG_EPS_L2, w, 8, gx, gy, msg, n);
}

static int gdnc_run(const Case *c, uint32_t T, int mutant, uint32_t gx, uint32_t gy, uint16_t *outb, float *tail, float *state,
                    uint8_t *snap, uint32_t every4, uint32_t count, char *msg, size_t n) {
    static BlackwellIRProgram prog; static int built = -1;
    if (built != mutant) {
        OmegaGdncSpec sp = { .mutant = mutant, .ratio = RT };
        if (omega_blackwell_codegen_gdnc_prog(&sp, &prog) != 0) { snprintf(msg, n, "gdnc codegen refused"); return -1; }
        built = mutant;
    }
    /* the constant bank comes from the API's plan (omega_gpu_gdnc_api.h), the host addresses standing in for the VAs */
    const OmegaGdncReq rq = {
        .ratio = RT,
        .io = { .tokens = T, .qkvz_va = (uintptr_t)c->rowsb, .xstride = ROWB, .ab_va = (uintptr_t)c->ab, .abstride = ABROW,
                .out_va = (uintptr_t)outb, .ostride = OROWB, .eps_l2 = EPS_L2, .eps_norm = EPS_N, .in_scale = 1.0f, .out_scale = 1.0f },
        .par_va = (uintptr_t)c->par, .tail_va = (uintptr_t)tail, .state_va = (uintptr_t)state,
        .snap_va = (uintptr_t)snap, .snap_every4 = every4, .snap_count = snap ? count : 0, .snap_stride = SNAP_BYTES,
    };
    OmegaGdncLaunch L;
    if (omega_gdnc_plan(&rq, &L) != 0) { snprintf(msg, n, "gdnc plan refused"); return -1; }
    if (L.threads != OMEGA_GDNC_THREADS || L.grid_x != RT || L.grid_y != 16u || L.shared_bytes != OMEGA_GDNC_SHARED_BYTES) {
        snprintf(msg, n, "gdnc plan geometry %ux(%u,%u) %u B", L.threads, L.grid_x, L.grid_y, L.shared_bytes); return -1;
    }
    SimBuf b[7] = {
        { OMEGA_GDNC_ARG_QKVZ, c->rowsb, (size_t)T * ROWB }, { OMEGA_GDNC_ARG_AB, c->ab, (size_t)T * ABROW },
        { OMEGA_GDNC_ARG_PAR, c->par, P_BYTES }, { OMEGA_GDNC_ARG_TAIL, tail, TAIL_BYTES },
        { OMEGA_GDNC_ARG_STATE, state, STATE_BYTES }, { OMEGA_GDNC_ARG_OUT, outb, (size_t)T * OROWB },
        { OMEGA_GDNC_ARG_SNAP, snap, snap ? (size_t)count * SNAP_BYTES : 0 },
    };
    return sim_run(&prog, L.threads, L.shared_bytes, b, snap ? 7 : 6, OMEGA_GDNC_ARG_QKVZ, L.args, L.n_args, gx, gy, msg, n);
}

/* max |a - b| / max |ref| over the heads [0, nh) of a [rows][VH][HD] set */
typedef struct { double err, ref; } Err;
static void err_add(Err *e, double a, double r) { const double d = fabs(a - r); if (d > e->err || d != d) e->err = d != d ? INFINITY : d; if (fabs(r) > e->ref) e->ref = fabs(r); }
static double err_rel(const Err *e) { return e->ref > 0 ? e->err / e->ref : e->err; }
static void cmp_state(Err *e, const float *a, const float *r, uint32_t nh) {
    for (size_t i = 0; i < (size_t)nh * HD * HD; i++) err_add(e, a[i], r[i]);
}
static int cmp_tail(const float *a, const float *r, uint32_t nh) { /* the tail is f32 data movement: bit exact */
    for (uint32_t h = 0; h < nh; h++) for (uint32_t i = 0; i < OMEGA_GDN_TAIL_HEAD_BYTES / 4; i++) {
        const size_t k = (size_t)h * (OMEGA_GDN_TAIL_HEAD_BYTES / 4) + i;
        if ((i & 3u) != 3u && memcmp(&a[k], &r[k], 4) != 0) { /* word 3 of each 16-byte slot is padding */
            if (getenv("GDNC_DBG")) fprintf(stderr, "tail diff head %u word %u: %g vs %g\n", h, i, (double)a[k], (double)r[k]);
            return 0;
        }
    }
    return 1;
}

typedef struct { uint32_t T, gx, gy, every4, count; } Run;
/* one comparison; returns the out relative error (or -1 on a sim failure) */
static double battery_run(const Run *R, uint32_t seed, int mutant, int perturb, int verbose) {
    const uint32_t T = R->T, nh = R->gx * R->gy; /* heads 0..nh-1 (gx = RT whenever gy > 1) */
    Case c; case_make(&c, T, seed);
    char msg[256] = "";
    float *outf = calloc((size_t)T, OROWF);
    uint16_t *outb = calloc((size_t)T, OROWB);
    float *tm = aligned_alloc(256, (TAIL_BYTES + 255) & ~255u), *sm = aligned_alloc(256, STATE_BYTES);
    float *tg = aligned_alloc(256, (TAIL_BYTES + 255) & ~255u), *sg = aligned_alloc(256, STATE_BYTES);
    uint8_t *snap = R->count ? aligned_alloc(256, (size_t)(R->count + 1) * SNAP_BYTES) : NULL;
    if (snap) memset(snap, 0xa5, (size_t)(R->count + 1) * SNAP_BYTES);
    memcpy(tm, c.tail0, TAIL_BYTES); memcpy(sm, c.state0, STATE_BYTES);
    memcpy(tg, c.tail0, TAIL_BYTES); memcpy(sg, c.state0, STATE_BYTES);
    if (perturb) for (unsigned i = 0; i < HD; i++) c.rowsb[(size_t)(T / 2) * RW + 2u * KH * HD + i] ^= 0x8000u; /* gdnc only: sign of head 0's v at token T/2 */
    const int eg = gdnc_run(&c, T, mutant, R->gx, R->gy, outb, tg, sg, snap, R->every4, R->count, msg, sizeof msg);
    if (perturb) for (unsigned i = 0; i < HD; i++) c.rowsb[(size_t)(T / 2) * RW + 2u * KH * HD + i] ^= 0x8000u;
    double rel = -1.0;
    if (eg != 0) { printf("  gdnc sim failed: %s\n", msg); goto out; }
    /* golden in segments: one per snapshot, then the rest */
    const uint32_t seg = R->count ? 16u * R->every4 : T;
    Err es_snap = { 0, 0 }; int snap_tail_ok = 1;
    uint32_t t0 = 0, k = 0;
    while (t0 < T) {
        const uint32_t t1 = t0 + seg < T ? t0 + seg : T;
        if (multi_seg(&c, t0, t1, R->gx, R->gy, outf, tm, sm, msg, sizeof msg) != 0) { printf("  multi sim failed: %s\n", msg); goto out; }
        if (R->count && k < R->count && t1 - t0 == seg) {
            Err e1 = { 0, 0 };
            cmp_state(&e1, (const float *)(snap + (size_t)k * SNAP_BYTES), sm, nh);
            if (err_rel(&e1) > err_rel(&es_snap)) es_snap = e1;
            if (es_snap.ref < e1.ref) es_snap.ref = e1.ref;
            snap_tail_ok &= cmp_tail((const float *)(snap + (size_t)k * SNAP_BYTES + STATE_BYTES), tm, nh);
            k++;
        }
        t0 = t1;
    }
    Err eo = { 0, 0 }, es = { 0, 0 };
    for (uint32_t t = 0; t < T; t++) for (uint32_t i = 0; i < nh * HD; i++)
        err_add(&eo, bf16_f(outb[(size_t)t * VD + i]), bf16_f(bf16_rn(outf[(size_t)t * VD + i])));
    cmp_state(&es, sg, sm, nh);
    const int tail_ok = cmp_tail(tg, tm, nh);
    int untouched = 1; /* the snapshot after the last allowed one is never written */
    if (snap) for (size_t i = 0; i < SNAP_BYTES; i++) if (snap[(size_t)R->count * SNAP_BYTES + i] != 0xa5) { untouched = 0; break; }
    rel = err_rel(&eo);
    if (verbose) {
        printf("sim T %u heads %u%s: out max err %.3e (rel %.3e, tol %.3g) | state rel %.3e (tol %.3g) | tail %s | snapshots %u rel %.3e tail %s, slot %u %s\n",
               T, nh, mutant ? " MUTANT" : "", eo.err, rel, TOL_OUT, err_rel(&es), TOL_STATE, tail_ok ? "bit-exact" : "DIFFER", k,
               err_rel(&es_snap), snap_tail_ok ? "bit-exact" : "DIFFER", R->count, untouched ? "untouched" : "WRITTEN");
        CHECK(rel <= TOL_OUT, "T %u: out within tolerance (%.3e > %.3g)", T, rel, TOL_OUT);
        CHECK(err_rel(&es) <= TOL_STATE, "T %u: state within tolerance (%.3e > %.3g)", T, err_rel(&es), TOL_STATE);
        CHECK(tail_ok, "T %u: conv tail bit-exact", T);
        if (R->count) CHECK(k == R->count && err_rel(&es_snap) <= TOL_STATE && snap_tail_ok && untouched,
                            "T %u: %u snapshots equal the sequential state at their boundaries, the next slot untouched", T, R->count);
        if (rel > TOL_OUT || err_rel(&es) > TOL_STATE) rel = INFINITY;
    } else if (err_rel(&es) > rel) rel = err_rel(&es) * (TOL_OUT / TOL_STATE); /* control: red on either set */
out:
    free(outf); free(outb); free(tm); free(sm); free(tg); free(sg); free(snap); case_free(&c);
    return rel;
}

/* ---- the double oracle: tests/gpu_gdn_test.c oracle_step at ratio 2, in_scale = out_scale = 1, over the bf16 rows ---- */
typedef struct { uint16_t *y; float *tail, *state; } Oracle; /* y bf16 [T][VD]; tail, state: reference layouts after T tokens */
static double sigm(double x) { return 1.0 / (1.0 + exp(-x)); }
static double softplus_ref(double x) { return x > 20.0 ? x : log1p(exp(x)); }
static void oracle_run(const Case *c, uint32_t T, Oracle *o) {
    double *tail = malloc((size_t)NT * 8), *St = malloc((size_t)NS * 8);
    static double mixed[CD], q[KH * HD], k[KH * HD];
    for (size_t i = 0; i < NT; i++) tail[i] = c->tailr[i];
    for (size_t i = 0; i < NS; i++) St[i] = c->state0[i];
    o->y = malloc((size_t)T * OROWB);
    for (uint32_t t = 0; t < T; t++) {
        const float *row = c->rowsf + (size_t)t * RW, *ab = c->ab + (size_t)t * VH * 2;
        for (unsigned ch = 0; ch < CD; ch++) {
            const double x = row[ch];
            const float *w = &c->conv_w[(size_t)ch * 4];
            const double acc = w[0] * tail[ch] + w[1] * tail[CD + ch] + w[2] * tail[2 * CD + ch] + w[3] * x;
            mixed[ch] = acc * sigm(acc);
            tail[ch] = tail[CD + ch]; tail[CD + ch] = tail[2 * CD + ch]; tail[2 * CD + ch] = x;
        }
        for (unsigned hk = 0; hk < KH; hk++) {
            double sq = 0, sk = 0;
            for (unsigned i = 0; i < HD; i++) { sq += mixed[hk * HD + i] * mixed[hk * HD + i]; sk += mixed[KH * HD + hk * HD + i] * mixed[KH * HD + hk * HD + i]; }
            const double iq = 1.0 / sqrt(sq + EPS_L2), ik = 1.0 / sqrt(sk + EPS_L2);
            for (unsigned i = 0; i < HD; i++) { q[hk * HD + i] = mixed[hk * HD + i] * iq / sqrt((double)HD); k[hk * HD + i] = mixed[KH * HD + hk * HD + i] * ik; }
        }
        for (unsigned h = 0; h < VH; h++) {
            const unsigned hk = h / RT;
            const double negA = -(double)(float)exp((double)c->a_log[h]); /* the API's float negA (double exp, rounded) */
            const double gl = (double)(float)negA * softplus_ref((double)ab[2 * h] + c->dtb[h]);
            const double decay = exp(gl), beta = sigm(ab[2 * h + 1]);
            double *S = St + (size_t)h * HD * HD, kv[HD], ov[HD], dl[HD];
            const double *kk = &k[hk * HD], *qq = &q[hk * HD], *v = &mixed[2 * KH * HD + h * HD];
            for (size_t i = 0; i < (size_t)HD * HD; i++) S[i] *= decay;
            for (unsigned j = 0; j < HD; j++) kv[j] = 0;
            for (unsigned r = 0; r < HD; r++) for (unsigned j = 0; j < HD; j++) kv[j] += S[r * HD + j] * kk[r];
            for (unsigned j = 0; j < HD; j++) dl[j] = (v[j] - kv[j]) * beta;
            for (unsigned r = 0; r < HD; r++) for (unsigned j = 0; j < HD; j++) S[r * HD + j] += kk[r] * dl[j];
            for (unsigned j = 0; j < HD; j++) ov[j] = 0;
            for (unsigned r = 0; r < HD; r++) for (unsigned j = 0; j < HD; j++) ov[j] += S[r * HD + j] * qq[r];
            double ms = 0; for (unsigned j = 0; j < HD; j++) ms += ov[j] * ov[j];
            const double inv = 1.0 / sqrt(ms / HD + EPS_N);
            for (unsigned j = 0; j < HD; j++) {
                const double zz = row[CD + h * HD + j];
                o->y[(size_t)t * VD + h * HD + j] = bf16_rn((float)(ov[j] * inv * c->nw[j] * (zz * sigm(zz))));
            }
        }
    }
    o->tail = malloc((size_t)NT * 4); o->state = malloc((size_t)NS * 4);
    for (size_t i = 0; i < NT; i++) o->tail[i] = (float)tail[i];
    for (size_t i = 0; i < NS; i++) o->state[i] = (float)St[i];
    free(tail); free(St);
}
static void oracle_free(Oracle *o) { free(o->y); free(o->tail); free(o->state); }
/* out (bf16, against the oracle rounded to bf16), state (relative), tail (reference layout, bit-exact) */
static void oracle_cmp(const Oracle *o, uint32_t T, const uint16_t *outb, const float *tail, const float *state,
                       double *ro, double *rs, int *tail_ok) {
    Err eo = { 0, 0 }, es = { 0, 0 };
    for (size_t i = 0; i < (size_t)T * VD; i++) err_add(&eo, bf16_f(outb[i]), bf16_f(o->y[i]));
    for (size_t i = 0; i < NS; i++) err_add(&es, state[i], o->state[i]);
    *ro = err_rel(&eo); *rs = err_rel(&es);
    *tail_ok = memcmp(tail, o->tail, (size_t)NT * 4) == 0;
}

/* --sim-oracle: the chip arm's comparison, exercised on the host with the simulated kernel */
static void sim_oracle(void) {
    const uint32_t T = 256;
    Case c; case_make(&c, T, 1256u);
    Oracle o; oracle_run(&c, T, &o);
    for (int m = 0; m < (g_mutant ? 1 : 2); m++) {
        const int mut = m ? OMEGA_GDNC_MUTANT_DROP_DECAY : g_mutant;
        uint16_t *outb = calloc(T, OROWB);
        float *tg = aligned_alloc(256, (TAIL_BYTES + 255) & ~255u), *sg = aligned_alloc(256, STATE_BYTES), *tr = malloc((size_t)NT * 4);
        memcpy(tg, c.tail0, TAIL_BYTES); memcpy(sg, c.state0, STATE_BYTES);
        char msg[256] = "";
        const int e = gdnc_run(&c, T, mut, RT, KH, outb, tg, sg, NULL, 1, 0, msg, sizeof msg);
        CHECK(e == 0 && omega_gdn_unpack_tail_r(RT, tg, tr) == 0, "sim-oracle%s: gdnc ran (%s)", mut ? " mutant" : "", msg);
        double ro = INFINITY, rs = INFINITY; int tok = 0;
        if (e == 0) oracle_cmp(&o, T, outb, tr, sg, &ro, &rs, &tok);
        printf("sim-oracle T %u heads %u%s: out rel %.3e (tol %.3g) | state rel %.3e (tol %.3g) | tail %s\n", T, VH,
               mut ? " MUTANT 9" : "", ro, TOL_OUT, rs, TOL_STATE, tok ? "bit-exact" : "DIFFER");
        if (m == 0) {
            CHECK(ro <= TOL_OUT, "sim-oracle: out within tolerance (%.3e)", ro);
            CHECK(rs <= TOL_STATE, "sim-oracle: state within tolerance (%.3e)", rs);
            CHECK(tok, "sim-oracle: conv tail bit-exact");
        } else {
            printf("control mutant 9 against the oracle -> %s\n", ro > TOL_OUT || rs > TOL_STATE ? "red" : "NOT RED");
            CHECK(ro > TOL_OUT || rs > TOL_STATE, "sim-oracle: mutant 9 goes red");
        }
        free(outb); free(tg); free(sg); free(tr);
    }
    oracle_free(&o); case_free(&c);
}

/* ---- chip ---- */
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp_u64(const void *a, const void *b) { const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
#define CHIP_MAX_LAYERS 63u
/* one command list: layer l's launch over the chunk, layer 0 into io0's out rows, the others into io1's; ts: n + 1 stamps */
static int chip_rep(OmegaGpuCmdList *cl, OmegaGpuGdn *g, const Case *c, uint32_t n, const OmegaGdncRows *io0,
                    const OmegaGdncRows *io1, uint64_t *ts) {
    for (uint32_t l = 0; l < n; l++) if (omega_gpu_gdn_set_state(g, l, c->tailr, c->state0) != 0) return -1;
    omega_gpu_session_lock();
    omega_gpu_cmdlist_reset(cl);
    int rc = 0;
    for (uint32_t l = 0; l < n && !rc; l++) rc = omega_gpu_gdnc_record(cl, g, l, l ? io1 : io0, NULL, NULL);
    OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.stamps = 1;
    OmegaGpuCmdListStats st; memset(&st, 0, sizeof st);
    if (!rc && (omega_gpu_cmdlist_submit(cl, &o, &st) != 0 || st.sems_done != st.launches || st.launches != n)) rc = -1;
    if (!rc && omega_gpu_cmdlist_stamps(cl, ts, n + 1) != (int)(n + 1)) rc = -1;
    omega_gpu_session_unlock();
    return rc;
}
static int chip_main(uint32_t T, uint32_t n, int reps) {
    if (!getenv("SPARK_WINDOW")) {
        printf("gpu_gdnc_test: NOT_RUN (--chip needs SPARK_WINDOW: an authorized GPU window, through chip/run-queue.sh; nothing opened)\n");
        return 3;
    }
    omega_gpu_session_lock();
    const int open = omega_gpu_session_open() != NULL;
    omega_gpu_session_unlock();
    if (!open) { printf("gpu_gdnc_test: NOT_RUN (--chip: no device: %s)\n", omega_gpu_session_last_error()); return 3; }
    printf("chip T %u, %u value heads, %u launches (GDN layers), %d reps%s\n", T, VH, n, reps, g_mutant ? ", every launch MUTANT 9" : "");
    Case c; case_make(&c, T, 1000u + T);
    const double t0 = now_ms();
    Oracle o; oracle_run(&c, T, &o);
    printf("host oracle (double, one layer, T %u): %.1f s\n", T, (now_ms() - t0) / 1e3);
    const size_t s_rows = (size_t)T * ROWB, s_ab = ((size_t)T * ABROW + 255) & ~(size_t)255, s_out = ((size_t)T * OROWB + 255) & ~(size_t)255;
    uint16_t *ref0 = malloc(s_out), *got = malloc(s_out);
    float *tl = malloc((size_t)NT * 4), *sl = malloc(STATE_BYTES), *tn = malloc((size_t)NT * 4), *sn = malloc(STATE_BYTES);
    float *t1 = malloc((size_t)NT * 4), *s1 = malloc(STATE_BYTES);
    uint64_t ts[CHIP_MAX_LAYERS + 1], lns[CHIP_MAX_LAYERS * 8], tot[8];
    uint32_t nl = 0;
    OmegaGpuGdn *g = NULL; OmegaGpuCmdList *cl = NULL; OmegaGpuMem m; memset(&m, 0, sizeof m);
    CHECK(omega_gpu_gdn_alloc_ratio(n, RT, &g) == 0, "alloc %u GDN layers (%s)", n, omega_gpu_gdn_last_error());
    int ok = g != NULL;
    for (uint32_t l = 0; ok && l < n; l++) ok = omega_gpu_gdn_set_layer(g, l, c.conv_w, c.a_log, c.dtb, c.nw) == 0;
    if (ok) { omega_gpu_gdnc_test_set_mutant(g_mutant); ok = omega_gpu_gdnc_prepare(g) == 0; }
    CHECK(ok, "layers set, gdnc prepared (%s)", omega_gpu_session_last_error());
    if (ok) {
        omega_gpu_session_lock();
        ok = omega_gpu_session_alloc(s_rows + s_ab + 2 * s_out, &m) == 0;
        if (ok) {
            uint8_t *p = m.cpu;
            memcpy(p, c.rowsb, s_rows); memcpy(p + s_rows, c.ab, (size_t)T * ABROW);
            memset(p + s_rows + s_ab, 0xff, 2 * s_out); /* bf16 NaN: an unwritten row never passes */
            ok = omega_gpu_session_sync_to_gpu(&m) == 0;
        }
        omega_gpu_session_unlock();
        CHECK(ok, "device rows (%zu B) uploaded (%s)", s_rows + s_ab + 2 * s_out, omega_gpu_session_last_error());
    }
    if (ok) { ok = omega_gpu_cmdlist_create(n, &cl) == 0; CHECK(ok, "command list (%s)", omega_gpu_session_last_error()); }
    OmegaGdncRows io0 = { .tokens = T, .qkvz_va = m.va, .xstride = ROWB, .ab_va = m.va + s_rows, .abstride = ABROW,
                          .out_va = m.va + s_rows + s_ab, .ostride = OROWB, .eps_l2 = EPS_L2, .eps_norm = EPS_N, .in_scale = 1.0f, .out_scale = 1.0f };
    OmegaGdncRows io1 = io0; io1.out_va = m.va + s_rows + s_ab + s_out;
    for (int r = 0; ok && r < reps; r++) {
        const int rc = chip_rep(cl, g, &c, n, &io0, &io1, ts);
        CHECK(rc == 0, "rep %d: %u launches ran, every semaphore done (%s)", r + 1, n, omega_gpu_session_last_error());
        if (rc) { ok = 0; break; }
        omega_gpu_session_lock(); const int sy = omega_gpu_session_sync_to_cpu(&m); omega_gpu_session_unlock();
        const uint8_t *p = (const uint8_t *)m.cpu + s_rows + s_ab;
        CHECK(sy == 0 && omega_gpu_gdn_get_state(g, 0, tl, sl) == 0 && omega_gpu_gdn_get_state(g, n - 1, tn, sn) == 0,
              "rep %d: rows and states read back (%s)", r + 1, omega_gpu_gdn_last_error());
        memcpy(got, p, s_out);
        const int same_last = memcmp(p, p + s_out, (size_t)T * OROWB) == 0 && memcmp(tl, tn, (size_t)NT * 4) == 0 && memcmp(sl, sn, STATE_BYTES) == 0;
        CHECK(n == 1 || same_last, "rep %d: layer %u bit-identical to layer 0 (out, tail, state)", r + 1, n - 1);
        if (r == 0) {
            double ro, rs; int tok;
            oracle_cmp(&o, T, got, tl, sl, &ro, &rs, &tok);
            printf("chip T %u heads %u layer 0%s: out rel %.3e (tol %.3g) | state rel %.3e (tol %.3g) | tail %s | layer %u %s\n", T, VH,
                   g_mutant ? " MUTANT" : "", ro, TOL_OUT, rs, TOL_STATE, tok ? "bit-exact" : "DIFFER", n - 1, same_last ? "== layer 0" : "DIFFERS");
            CHECK(ro <= TOL_OUT, "chip: out within tolerance (%.3e > %.3g)", ro, TOL_OUT);
            CHECK(rs <= TOL_STATE, "chip: state within tolerance (%.3e > %.3g)", rs, TOL_STATE);
            CHECK(tok, "chip: conv tail bit-exact");
            memcpy(ref0, got, s_out); memcpy(t1, tl, (size_t)NT * 4); memcpy(s1, sl, STATE_BYTES);
        } else {
            CHECK(memcmp(ref0, got, (size_t)T * OROWB) == 0 && memcmp(t1, tl, (size_t)NT * 4) == 0 && memcmp(s1, sl, STATE_BYTES) == 0,
                  "rep %d: bit-identical to rep 1", r + 1);
        }
        if (r < 8) tot[r] = ts[n] - ts[0];
        uint64_t mx = 0, mn = UINT64_MAX;
        for (uint32_t l = 0; l < n; l++) {
            const uint64_t d = ts[l + 1] - ts[l];
            if (nl < CHIP_MAX_LAYERS * 8) lns[nl++] = d;
            if (d > mx) mx = d;
            if (d < mn) mn = d;
        }
        printf("chip rep %d: %u launches %.3f ms (GPU timestamps), per launch min %.3f max %.3f ms\n", r + 1, n, (ts[n] - ts[0]) / 1e6, mn / 1e6, mx / 1e6);
    }
    if (ok && reps > 0) {
        const int rr = reps < 8 ? reps : 8;
        qsort(tot, rr, 8, cmp_u64); qsort(lns, nl, 8, cmp_u64);
        const double ms = tot[rr / 2] / 1e6, per = lns[nl / 2] / 1e6, ms30 = per * 30.0;
        printf("GDNC_G3 T %u heads %u launches %u reps %d: %u-launch median %.3f ms, per launch median %.4f ms, 30 layers %.3f ms, "
               "per token-layer %.2f ns; gate 60 ms: %s; bf16 budget 38.3 ms: %s%s\n", T, VH, n, reps, n, ms, per, ms30,
               per * 1e6 / T, ms30 <= 60.0 ? "ok" : "ABOVE", ms30 <= 38.3 ? "ok" : "ABOVE", g_mutant ? " (MUTANT: not a timing receipt)" : "");
    }
    if (ok && g_mutant == 0) { /* control: mutant 9 on layer 0 must go red against the same oracle */
        omega_gpu_gdnc_test_set_mutant(OMEGA_GDNC_MUTANT_DROP_DECAY);
        int rc = omega_gpu_gdnc_prepare(g);
        if (!rc) rc = chip_rep(cl, g, &c, 1, &io0, &io1, ts);
        omega_gpu_session_lock(); if (!rc) rc = omega_gpu_session_sync_to_cpu(&m); omega_gpu_session_unlock();
        if (!rc) rc = omega_gpu_gdn_get_state(g, 0, tl, sl);
        double ro = 0, rs = 0; int tok = 0;
        if (!rc) { memcpy(got, (const uint8_t *)m.cpu + s_rows + s_ab, s_out); oracle_cmp(&o, T, got, tl, sl, &ro, &rs, &tok); }
        printf("chip control mutant 9 (drop-decay) layer 0: %s out rel %.3e state rel %.3e -> %s\n", rc ? "run failed," : "", ro, rs,
               !rc && (ro > TOL_OUT || rs > TOL_STATE) ? "red" : "NOT RED");
        CHECK(!rc && (ro > TOL_OUT || rs > TOL_STATE), "chip: mutant 9 goes red");
        omega_gpu_gdnc_test_set_mutant(0);
    }
    if (cl) omega_gpu_cmdlist_free(cl);
    if (m.cpu) { omega_gpu_session_lock(); omega_gpu_session_free(&m); omega_gpu_session_unlock(); }
    if (g) omega_gpu_gdn_free(g);
    free(ref0); free(got); free(tl); free(sl); free(tn); free(sn); free(t1); free(s1);
    oracle_free(&o); case_free(&c);
    printf("gpu_gdnc_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}

int main(int argc, char **argv) {
    int sim = 0, simo = 0, chip = 0, reps = 3;
    uint32_t tokens = 8192, layers = 30;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--sim-oracle") == 0) simo = 1;
        else if (strcmp(argv[i], "--chip") == 0) chip = 1;
        else if (strcmp(argv[i], "--mutant") == 0 && i + 1 < argc) g_mutant = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = atoi(argv[++i]);
        else { printf("gpu_gdnc_test: INVALID argument %s\n", argv[i]); return 2; }
    }
    if (g_mutant != 0 && g_mutant != OMEGA_GDNC_MUTANT_DROP_DECAY) { printf("gpu_gdnc_test: INVALID --mutant %d (0 or 9)\n", g_mutant); return 2; }
    if (sim + simo + chip != 1) { printf("gpu_gdnc_test: pick one of --sim, --sim-oracle, --chip\n"); return 2; }
    if (chip) {
        if (tokens == 0 || tokens > 65536 || layers == 0 || layers > CHIP_MAX_LAYERS || reps < 1 || reps > 8) {
            printf("gpu_gdnc_test: INVALID --tokens/--layers/--reps (1..65536, 1..%u, 1..8)\n", CHIP_MAX_LAYERS); return 2;
        }
        return chip_main(tokens, layers, reps);
    }
    if (simo) {
        sim_oracle();
        printf("gpu_gdnc_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
        return g_failed ? 1 : 0;
    }
    OmegaGdncSpec spec = { .mutant = 0, .ratio = RT };
    uint32_t insns = 0, vregs = 0, gprs = 0;
    const int st = omega_gdnc_prog_stats(&spec, &insns, &vregs, &gprs);
    static BlackwellIRProgram prog;
    const int pr = omega_blackwell_codegen_gdnc_prog(&spec, &prog);
    printf("gdnc program: %u insns, %u vregs, %u registers, %u placeholders, %u B shared\n", insns, vregs, gprs,
           pr == 0 ? omega_gdnc_placeholder_count(&prog) : 0u, OMEGA_GDNC_SHARED_BYTES);
    CHECK(st == 0 && pr == 0 && insns <= 2048u && vregs <= 512u && gprs <= 246u && OMEGA_GDNC_THREADS * (gprs + 2u) <= 65536u,
          "gdnc builds within the IR caps and the register file");
    OmegaGdncSpec bad = { .mutant = 3, .ratio = RT }, bad2 = { .mutant = 0, .ratio = 9 };
    CHECK(omega_blackwell_codegen_gdnc_prog(&bad, &prog) != 0 && omega_blackwell_codegen_gdnc_prog(&bad2, &prog) != 0 &&
          omega_blackwell_codegen_gdnc_prog(NULL, &prog) != 0, "bad specs refused (mutant 3, ratio 9, NULL)");
    printf("bad specs: mutant 3, ratio 9, NULL refused\n");

    const Run runs[] = {
        { 2, RT, 2, 1, 0 }, { 37, RT, 1, 1, 2 }, { 256, RT, KH, 4, 3 }, { 2048, 2, 1, 32, 4 }, { 8192, 1, 1, 64, 8 },
    };
    for (unsigned i = 0; i < sizeof runs / sizeof runs[0]; i++) {
        const double r = battery_run(&runs[i], 1000u + runs[i].T, g_mutant, 0, 1);
        if (r < 0) CHECK(0, "T %u: both kernels ran", runs[i].T);
    }
    if (g_mutant == 0) {
        const Run ctl = { 256, RT, 1, 4, 0 };
        const double m9 = battery_run(&ctl, 77, OMEGA_GDNC_MUTANT_DROP_DECAY, 0, 0);
        printf("control mutant 9 (drop-decay) T 256: rel %.3e (tol %.3g) -> %s\n", m9, TOL_OUT, m9 > TOL_OUT ? "red" : "NOT RED");
        CHECK(m9 > TOL_OUT, "mutant 9 goes red");
        const double pe = battery_run(&ctl, 77, 0, 1, 0);
        printf("control perturbed (sign of head 0 v, token 128) (gdnc input only) T 256: rel %.3e (tol %.3g) -> %s\n", pe, TOL_OUT, pe > TOL_OUT ? "red" : "NOT RED");
        CHECK(pe > TOL_OUT, "a perturbed input token goes red");
    }
    printf("gpu_gdnc_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
