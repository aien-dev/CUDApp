/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * gpu_gdnc_test (DRA-84): the chunked tensor-core gated-delta-rule prefill scan (omega_blackwell_codegen_gdnc.h) against
 * its golden, the sequential multi-token kernel (omega_blackwell_codegen_gdn_multi_prog), both in the warp simulator.
 *   --sim              T 2, 37 (ragged sub-chunk), 256 (all 32 value heads), 2048 (heads 0..1), 8192 (head 0): every out
 *                      element (both sides rounded to bf16), the final f32 state and conv tail, and every snapshot against
 *                      the multi kernel's state and tail at the same token (the multi kernel run in segments).
 *                      Controls that must go red: mutant 9 (drop-decay), one perturbed input token, bad specs refused.
 *   --sim --mutant 9   the whole battery with the mutant: must exit 1 (the mutant is red).
 * Tolerance: relative to the largest reference magnitude of the compared set (TOL_OUT, TOL_STATE; derivation in the
 * DRA-84 PR body, Wolfram). Verdict line last: "gpu_gdnc_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_blackwell_codegen_gdnc.h"
#include "omega_blackwell_codegen_gdn.h"
#include "omega_gpu_gdn_api.h"
#include "bw_warp_sim_tc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
} Case;
static void case_make(Case *c, uint32_t T, uint32_t seed) {
    uint32_t s = seed;
    float *conv_w = malloc((size_t)CD * 16), a_log[VH], dtb[VH], nw[HD];
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
    free(conv_w); free(tail);
}
static void case_free(Case *c) { free(c->par); free(c->tail0); free(c->state0); free(c->rowsf); free(c->rowsb); free(c->ab); }

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
    SimBuf b[7] = {
        { OMEGA_GDNC_ARG_QKVZ, c->rowsb, (size_t)T * ROWB }, { OMEGA_GDNC_ARG_AB, c->ab, (size_t)T * ABROW },
        { OMEGA_GDNC_ARG_PAR, c->par, P_BYTES }, { OMEGA_GDNC_ARG_TAIL, tail, TAIL_BYTES },
        { OMEGA_GDNC_ARG_STATE, state, STATE_BYTES }, { OMEGA_GDNC_ARG_OUT, outb, (size_t)T * OROWB },
        { OMEGA_GDNC_ARG_SNAP, snap, snap ? (size_t)count * SNAP_BYTES : 0 },
    };
    const float one = 1.0f;
    uint32_t w[11];
    memcpy(&w[0], &EPS_L2, 4); memcpy(&w[1], &EPS_N, 4); memcpy(&w[2], &one, 4); memcpy(&w[3], &one, 4);
    w[4] = T; w[5] = ROWB; w[6] = ABROW; w[7] = OROWB; w[8] = every4; w[9] = snap ? count : 0; w[10] = SNAP_BYTES;
    return sim_run(&prog, OMEGA_GDNC_THREADS, OMEGA_GDNC_SHARED_BYTES, b, snap ? 7 : 6, OMEGA_GDNC_ARG_EPS_L2, w, 11, gx, gy, msg, n);
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

int main(int argc, char **argv) {
    int sim = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--mutant") == 0 && i + 1 < argc) g_mutant = atoi(argv[++i]);
    }
    if (!sim) { printf("gpu_gdnc_test: only --sim is implemented here; the chip arms are bench_gdnc (DRA-84 slices)\n"); return 2; }
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
