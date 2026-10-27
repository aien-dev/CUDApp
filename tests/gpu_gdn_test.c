/*
 * gpu_gdn_test: the one-launch Gated DeltaNet decode step of Qwen3.8-27B (omega_gpu_gdn_api.h), omega#542.
 *   --host-only   layout helpers, codegen budgets and refusals, the double oracle's delta-rule property
 *   --sim         warp-simulator parity over the whole grid (48 CTAs x 256 threads, both fragment orders bit-identical),
 *                 three chained steps against the double oracle, mutants 1-4 caught
 *   (default)     the chip: two layers (addressing), five chained steps each against the oracle (output, tail, state),
 *                 a rerun from the same state bit-identical, reset, mutants 1-4 caught, the record path (two steps in
 *                 one command list: the second reads the first's state), chip medians (--reps N, --out receipt.json)
 * The oracle is sc crates/aien-inference-abi/src/qwen38.rs `delta_net` in double, from the projections on.
 * Verdict line last: "gpu_gdn_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_gpu_gdn_api.h"
#include "omega_gpu_hostmath.h"
#include "omega_gpu_session.h"
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_codegen_tc.h"
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

/* GDN_TEST_RATIO: value heads per key head under test (OmegaGdnSpec.ratio; 0 = 3, Qwen3.8-27B; -DGDN_TEST_RATIO=2 builds the
 * Qwen3.6-35B-A3B shape, sc L306). Every size below follows it. */
#ifndef GDN_TEST_RATIO
#define GDN_TEST_RATIO 0u
#endif
#define RT (GDN_TEST_RATIO ? GDN_TEST_RATIO : 3u)
#define KH OMEGA_GDN_KEY_HEADS
#define HD OMEGA_GDN_HEAD_DIM
#define VH (KH * RT)
#define VD (VH * HD)
#define CD (2u * KH * HD + VD)
#define NS (VH * HD * HD)
#define NT (3u * CD)
#define T_PAR_SCAL (CD * OMEGA_GDN_TAPS * 4u)
#define T_PAR_NORM (T_PAR_SCAL + VH * 8u)
#define T_PAR_BYTES (T_PAR_NORM + HD * 4u)
#define T_TAIL_BYTES (VH * OMEGA_GDN_TAIL_HEAD_BYTES)
#define T_STATE_BYTES (VH * OMEGA_GDN_STATE_HEAD_BYTES)
static const float EPS_L2 = 1e-6f, EPS_N = 1e-6f;
/* tolerances, normalised by the largest reference magnitude of the vector (EX2 / RCP / RSQ / LG2 are approximate on
 * the chip, the oracle is double; the mutants miss by orders of magnitude more, see the receipts) */
#define TOL_OUT 2e-4
#define TOL_STATE 2e-5
#define TOL_TAIL 1e-6

typedef struct {
    float *conv_w, a_log[VH], dtb[VH], nw[HD];
    float *tail0, *state0; /* initial state, reference layouts */
} Layer;
typedef struct { float *qkv, *z, ab[2 * VH]; } Inp;

static void layer_make(Layer *L, uint32_t seed) {
    uint32_t s = seed;
    L->conv_w = malloc((size_t)CD * 4 * 4); L->tail0 = malloc((size_t)NT * 4); L->state0 = malloc((size_t)NS * 4);
    for (size_t i = 0; i < (size_t)CD * 4; i++) L->conv_w[i] = frand(&s, -0.6f, 0.6f);
    for (unsigned h = 0; h < VH; h++) { L->a_log[h] = frand(&s, -1.5f, 1.5f); L->dtb[h] = frand(&s, -3.0f, 1.0f); }
    for (unsigned i = 0; i < HD; i++) L->nw[i] = frand(&s, 0.4f, 1.6f);
    for (size_t i = 0; i < NT; i++) L->tail0[i] = frand(&s, -2.0f, 2.0f);
    for (size_t i = 0; i < NS; i++) L->state0[i] = frand(&s, -0.2f, 0.2f);
}
static void layer_free(Layer *L) { free(L->conv_w); free(L->tail0); free(L->state0); }
/* in_scale: the kernel multiplies qkv and z by it, so the stored rows are the true rows / in_scale */
static void inp_make(Inp *in, uint32_t seed, float in_scale) {
    uint32_t s = seed;
    in->qkv = malloc((size_t)CD * 4); in->z = malloc((size_t)VD * 4);
    for (unsigned i = 0; i < CD; i++) in->qkv[i] = frand(&s, -2.5f, 2.5f) / in_scale;
    for (unsigned i = 0; i < VD; i++) in->z[i] = frand(&s, -3.0f, 3.0f) / in_scale;
    for (unsigned h = 0; h < VH; h++) { in->ab[2 * h] = frand(&s, -4.0f, 4.0f); in->ab[2 * h + 1] = frand(&s, -4.0f, 4.0f); }
    /* a few heads at the softplus threshold and past it, and beta near 0 / 1 */
    if (!getenv("GDN_NOFORCE")) { in->ab[0] = 25.0f; in->ab[2] = 19.9f; in->ab[4] = -30.0f; in->ab[3] = 12.0f; in->ab[5] = -12.0f; }
}
static void inp_free(Inp *in) { free(in->qkv); free(in->z); }

/* ---- the double oracle ---- */
typedef struct { double *tail, *state; } DState;
static void dstate_init(DState *d, const Layer *L) {
    d->tail = malloc((size_t)NT * 8); d->state = malloc((size_t)NS * 8);
    for (size_t i = 0; i < NT; i++) d->tail[i] = L->tail0[i];
    for (size_t i = 0; i < NS; i++) d->state[i] = L->state0[i];
}
static void dstate_free(DState *d) { free(d->tail); free(d->state); }
static double sigm(double x) { return 1.0 / (1.0 + exp(-x)); }
static double softplus_ref(double x) { return x > 20.0 ? x : log1p(exp(x)); }
static void oracle_step(const Layer *L, DState *d, const Inp *in, float in_scale, float out_scale, double *y) {
    static double mixed[CD], q[KH * HD], k[KH * HD];
    for (unsigned ch = 0; ch < CD; ch++) {
        const double x = (double)(in->qkv[ch] * in_scale); /* the kernel's f32 product */
        const float *w = &L->conv_w[(size_t)ch * 4];
        double acc = w[0] * d->tail[ch] + w[1] * d->tail[CD + ch] + w[2] * d->tail[2 * CD + ch] + w[3] * x;
        mixed[ch] = acc * sigm(acc);
        d->tail[ch] = d->tail[CD + ch]; d->tail[CD + ch] = d->tail[2 * CD + ch]; d->tail[2 * CD + ch] = x;
    }
    for (unsigned hk = 0; hk < KH; hk++) {
        double sq = 0, sk = 0;
        for (unsigned i = 0; i < HD; i++) { sq += mixed[hk * HD + i] * mixed[hk * HD + i]; sk += mixed[KH * HD + hk * HD + i] * mixed[KH * HD + hk * HD + i]; }
        const double iq = 1.0 / sqrt(sq + EPS_L2), ik = 1.0 / sqrt(sk + EPS_L2);
        for (unsigned i = 0; i < HD; i++) { q[hk * HD + i] = mixed[hk * HD + i] * iq / sqrt((double)HD); k[hk * HD + i] = mixed[KH * HD + hk * HD + i] * ik; }
    }
    for (unsigned h = 0; h < VH; h++) {
        const unsigned hk = h / (VH / KH);
        const double negA = -(double)(float)exp((double)L->a_log[h]); /* the API's float negA (double exp, rounded) */
        const double g = (double)(float)negA * softplus_ref((double)in->ab[2 * h] + L->dtb[h]);
        const double decay = exp(g), beta = sigm(in->ab[2 * h + 1]);
        double *S = d->state + (size_t)h * HD * HD, kv[HD], o[HD], dl[HD];
        const double *kk = &k[hk * HD], *qq = &q[hk * HD], *v = &mixed[2 * KH * HD + h * HD];
        for (size_t i = 0; i < (size_t)HD * HD; i++) S[i] *= decay;
        for (unsigned c = 0; c < HD; c++) kv[c] = 0;
        for (unsigned r = 0; r < HD; r++) for (unsigned c = 0; c < HD; c++) kv[c] += S[r * HD + c] * kk[r];
        for (unsigned c = 0; c < HD; c++) dl[c] = (v[c] - kv[c]) * beta;
        for (unsigned r = 0; r < HD; r++) for (unsigned c = 0; c < HD; c++) S[r * HD + c] += kk[r] * dl[c];
        for (unsigned c = 0; c < HD; c++) o[c] = 0;
        for (unsigned r = 0; r < HD; r++) for (unsigned c = 0; c < HD; c++) o[c] += S[r * HD + c] * qq[r];
        double ms = 0; for (unsigned c = 0; c < HD; c++) ms += o[c] * o[c];
        const double inv = 1.0 / sqrt(ms / HD + EPS_N);
        for (unsigned c = 0; c < HD; c++) {
            const double zz = (double)(in->z[h * HD + c] * in_scale);
            y[h * HD + c] = o[c] * inv * L->nw[c] * (zz * sigm(zz)) * out_scale;
        }
    }
}
/* max |got - ref| / max |ref| */
static double nerr(const float *got, const double *ref, size_t n) {
    double e = 0, m = 1e-30;
    for (size_t i = 0; i < n; i++) {
        const double a = fabs((double)got[i] - ref[i]);
        if (!isfinite(a)) { if (getenv("GDN_DBG")) printf("  non-finite at %zu: got %g ref %g\n", i, (double)got[i], ref[i]); return INFINITY; } /* a NaN or inf on either side never passes */
        if (a > e) e = a;
        if (fabs(ref[i]) > m) m = fabs(ref[i]);
    }
    return e / m;
}
typedef struct { double out, tail, state; } Errs;
static int errs_ok(const Errs *e) { return e->out <= TOL_OUT && e->tail <= TOL_TAIL && e->state <= TOL_STATE; }

/* ---- host-only ---- */
static void host_only(void) {
    /* layout round trip, and the copy check */
    float *t = malloc((size_t)NT * 4), *t2 = malloc((size_t)NT * 4), *dev = malloc(T_TAIL_BYTES);
    uint32_t s = 5;
    for (size_t i = 0; i < NT; i++) t[i] = frand(&s, -1, 1);
    omega_gdn_pack_tail_r(GDN_TEST_RATIO, t, dev);
    CHECK(omega_gdn_unpack_tail_r(GDN_TEST_RATIO, dev, t2) == 0 && memcmp(t, t2, (size_t)NT * 4) == 0, "tail pack/unpack round trip");
    dev[(1u * OMEGA_GDN_SLOTS + 7u) * 4u + 1u] += 1.0f; /* head 1 (a second copy of key head 0), q slot 7 */
    CHECK(omega_gdn_unpack_tail_r(GDN_TEST_RATIO, dev, t2) == -1, "unpack refuses disagreeing copies");
    free(t); free(t2); free(dev);
    /* parameter block */
    Layer L; layer_make(&L, 9);
    uint8_t *blk = malloc(T_PAR_BYTES);
    CHECK(omega_gdn_pack_params_r(GDN_TEST_RATIO, L.conv_w, L.a_log, L.dtb, L.nw, blk) == 0, "pack params");
    const float *sc = (const float *)(blk + T_PAR_SCAL);
    CHECK(sc[2 * 5] == -(float)omega_hostmath_exp((double)L.a_log[5]) && sc[2 * 5 + 1] == L.dtb[5], "negA and dt_bias interleaved");
    for (unsigned h = 0; h < VH; h++) { /* the libm-free exp agrees with libm to within one float ulp */
        const float want = -(float)exp((double)L.a_log[h]), got = sc[2 * h];
        CHECK(fabsf(got - want) <= fabsf(want) * 0x1p-23f, "negA head %u: %.9g vs libm %.9g", h, (double)got, (double)want);
    }
    CHECK(memcmp(blk + T_PAR_NORM, L.nw, HD * 4) == 0, "norm weight at its offset");
    CHECK(omega_gdn_pack_params_r(GDN_TEST_RATIO, NULL, L.a_log, L.dtb, L.nw, blk) == -1, "pack params refuses NULL");
    free(blk);
    /* codegen budgets */
    for (int m = 0; m <= 4; m++) {
        OmegaGdnSpec sp = { .mutant = m, .ratio = GDN_TEST_RATIO }; uint32_t ni = 0, ng = 0;
        int rc = omega_gdn_prog_stats(&sp, &ni, &ng);
        CHECK(rc == 0 && ni <= BW_MAX_IR_INSNS && ng + 2 <= BW_PHYS_GPR_MAX, "mutant %d codegen: rc %d, %u insns, %u registers", m, rc, ni, ng);
        if (m == 0) printf("kernel: %u instructions, peak %u registers (%u declared, %u threads)\n", ni, ng, omega_bw_gpr_declared(ng), OMEGA_GDN_THREADS);
    }
    OmegaGdnSpec bad = { .mutant = 5, .ratio = GDN_TEST_RATIO };
    CHECK(omega_gdn_prog_stats(&bad, NULL, NULL) == -1, "mutant 5 refused");
    /* the state copy / zero programs (cached tail and state) */
    {
        static BlackwellIRProgram cp;
        OmegaGdnSpec c1 = { .copy = OMEGA_GDN_COPY }, c2 = { .copy = OMEGA_GDN_ZERO }, c3 = { .copy = 3 }, cm = { .mutant = 1, .copy = OMEGA_GDN_COPY };
        CHECK(omega_blackwell_codegen_gdn_copy_prog(&c1, &cp) == 0 && cp.regalloc.peak_gpr_usage + 2 <= BW_PHYS_GPR_MAX, "copy codegen");
        CHECK(omega_blackwell_codegen_gdn_copy_prog(&c2, &cp) == 0, "zero codegen");
        CHECK(omega_blackwell_codegen_gdn_copy_prog(&c3, &cp) == -1 && omega_blackwell_codegen_gdn_copy_prog(&cm, &cp) == -1, "copy codegen refuses kind 3 and a mutant");
        CHECK(omega_gdn_prog_stats(&c1, NULL, NULL) == -1, "decode codegen refuses a copy spec");
    }
    /* the oracle's delta rule: from S = 0 with beta ~ 1, the step writes v along k, so S^T k = v (|k| = 1) */
    {
        DState d; dstate_init(&d, &L);
        memset(d.state, 0, (size_t)NS * 8);
        Inp in; inp_make(&in, 77, 1.0f);
        for (unsigned h = 0; h < VH; h++) in.ab[2 * h + 1] = 40.0f;
        double *y = malloc((size_t)VD * 8);
        static double mixed_v[CD];
        /* the expected v and k: recompute the conv of this step from the oracle's tail before the step */
        for (unsigned ch = 0; ch < CD; ch++) {
            const float *w = &L.conv_w[(size_t)ch * 4];
            double acc = w[0] * d.tail[ch] + w[1] * d.tail[CD + ch] + w[2] * d.tail[2 * CD + ch] + w[3] * (double)in.qkv[ch];
            mixed_v[ch] = acc * sigm(acc);
        }
        oracle_step(&L, &d, &in, 1.0f, 1.0f, y);
        double worst = 0;
        for (unsigned h = 0; h < VH; h += 7) {
            const unsigned hk = h / RT;
            double nk = 0; for (unsigned i = 0; i < HD; i++) nk += mixed_v[KH * HD + hk * HD + i] * mixed_v[KH * HD + hk * HD + i];
            nk = sqrt(nk + EPS_L2);
            for (unsigned c = 0; c < HD; c++) {
                double kv = 0;
                for (unsigned r = 0; r < HD; r++) kv += d.state[(size_t)h * HD * HD + r * HD + c] * mixed_v[KH * HD + hk * HD + r] / nk;
                const double want = mixed_v[2 * KH * HD + h * HD + c];
                if (fabs(kv - want) > worst) worst = fabs(kv - want);
            }
        }
        CHECK(worst < 1e-6, "oracle delta rule: S^T k = v after one step from zero (worst %.3g)", worst);
        free(y); inp_free(&in); dstate_free(&d);
    }
    layer_free(&L);
    /* the prefill kernels encode, and every SHFL.BFLY placeholder is patched (the count kernel_for demands) */
    for (unsigned k = 0; k < 3; k++) {
        static BlackwellIRProgram p;
        OmegaGdnSpec spec = { .ratio = GDN_TEST_RATIO };
        int rc = k == 0 ? omega_blackwell_codegen_gdn_prep_prog(&spec, &p) : k == 1 ? omega_blackwell_codegen_gdn_scan_prog(&spec, &p)
                                                                         : omega_blackwell_codegen_gdn_post_prog(&spec, &p);
        size_t need = rc == 0 ? omega_bw_program_code_bytes(&p) : 0;
        uint8_t *code = need ? malloc(need) : NULL;
        OmegaBlackwellKernel kern;
        int enc = code ? omega_blackwell_kernel_encode_into(&p, code, need, &kern) : -1;
        int np = enc == 0 ? omega_tc_patch_code(&p, code, need) : -1;
        const uint32_t want = omega_gdn_bfly_count(&p);
        if (getenv("GDN_DUMP") && code && atoi(getenv("GDN_DUMP")) == (int)k) { FILE *f = fopen("gdn_pf.bin", "wb"); if (f) { fwrite(code, 1, need, f); fclose(f); } }
        CHECK(rc == 0 && enc == 0 && np >= 0 && (uint32_t)np == want && (k == 0 || want > 0),
              "prefill kernel %u encodes: %zu insns, %u registers, %d of %u butterflies patched", k, p.count, p.regalloc.peak_gpr_usage, np, want);
        free(code);
    }
}

/* ---- simulator ---- */
enum { SB_QKV, SB_Z, SB_AB, SB_PAR, SB_TAIL, SB_STATE, SB_OUT, SB_N };
static uint64_t g_sim_issues;
static int sim_run_steps(int mutant, uint32_t steps, float in_scale, float out_scale, Errs *worst, char *msg, size_t n) {
    static BlackwellIRProgram prog;
    OmegaGdnSpec spec = { .mutant = mutant, .ratio = GDN_TEST_RATIO };
    if (omega_blackwell_codegen_gdn_prog(&spec, &prog) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    Layer L; layer_make(&L, 21);
    DState d; dstate_init(&d, &L);
    size_t sz[SB_N] = { CD * 4u, VD * 4u, VH * 8u, T_PAR_BYTES, T_TAIL_BYTES, T_STATE_BYTES, VD * 4u };
    uint8_t *buf[SB_N];
    for (int i = 0; i < SB_N; i++) buf[i] = aligned_alloc(256, (sz[i] + 255) & ~(size_t)255);
    omega_gdn_pack_params_r(GDN_TEST_RATIO, L.conv_w, L.a_log, L.dtb, L.nw, buf[SB_PAR]);
    omega_gdn_pack_tail_r(GDN_TEST_RATIO, L.tail0, (float *)buf[SB_TAIL]);
    memcpy(buf[SB_STATE], L.state0, T_STATE_BYTES);
    uint8_t *tail_save = malloc(T_TAIL_BYTES), *state_save = malloc(T_STATE_BYTES), *out_copy = malloc(VD * 4u);
    uint8_t *tail_copy = malloc(T_TAIL_BYTES), *state_copy = malloc(T_STATE_BYTES);
    double *y = malloc((size_t)VD * 8);
    float *tref = malloc((size_t)NT * 4);
    int err = 0;
    memset(worst, 0, sizeof *worst);
    for (uint32_t st = 0; st < steps && !err; st++) {
        Inp in; inp_make(&in, 1000 + st, in_scale);
        memcpy(buf[SB_QKV], in.qkv, sz[SB_QKV]); memcpy(buf[SB_Z], in.z, sz[SB_Z]); memcpy(buf[SB_AB], in.ab, sz[SB_AB]);
        memcpy(tail_save, buf[SB_TAIL], T_TAIL_BYTES); memcpy(state_save, buf[SB_STATE], T_STATE_BYTES);
        for (int order = 0; order < 2 && !err; order++) {
            if (order == 1) { memcpy(buf[SB_TAIL], tail_save, T_TAIL_BYTES); memcpy(buf[SB_STATE], state_save, T_STATE_BYTES); }
            float *yo = (float *)buf[SB_OUT];
            for (unsigned i = 0; i < VD; i++) yo[i] = -999.0f;
            WsSim s; memset(&s, 0, sizeof s);
            s.p = &prog; s.threads = OMEGA_GDN_THREADS; s.order = order;
            static uint8_t smem[OMEGA_GDN_SHARED_BYTES]; s.smem = smem; s.smem_bytes = OMEGA_GDN_SHARED_BYTES;
            for (int i = 0; i < SB_N; i++) {
                const uint64_t a = (uintptr_t)buf[i];
                s.cbank[(0x380 + 8 * i) / 4] = (uint32_t)a; s.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(a >> 32);
                s.ranges[i].lo = (uintptr_t)buf[i]; s.ranges[i].hi = (uintptr_t)buf[i] + sz[i];
            }
            s.nranges = SB_N;
            memcpy(&s.cbank[OMEGA_GDN_ARG_EPS_L2 / 4], &EPS_L2, 4); memcpy(&s.cbank[OMEGA_GDN_ARG_EPS_NORM / 4], &EPS_N, 4);
            memcpy(&s.cbank[OMEGA_GDN_ARG_IN_SCALE / 4], &in_scale, 4); memcpy(&s.cbank[OMEGA_GDN_ARG_OUT_SCALE / 4], &out_scale, 4);
            err = ws_run(&s, RT, OMEGA_GDN_GRID_Y);
            if (err) { snprintf(msg, n, "step %u order %d: %s (pc %d)", st, order, s.msg, s.err_pc); break; }
            g_sim_issues += s.warp_issues;
            size_t unwritten = 0; for (unsigned i = 0; i < VD; i++) if (yo[i] == -999.0f) unwritten++;
            if (unwritten) { err = -1; snprintf(msg, n, "%zu output words never written", unwritten); break; }
            if (order == 0) { memcpy(out_copy, yo, VD * 4u); memcpy(tail_copy, buf[SB_TAIL], T_TAIL_BYTES); memcpy(state_copy, buf[SB_STATE], T_STATE_BYTES); }
            else if (memcmp(out_copy, yo, VD * 4u) || memcmp(tail_copy, buf[SB_TAIL], T_TAIL_BYTES) || memcmp(state_copy, buf[SB_STATE], T_STATE_BYTES)) {
                err = -1; snprintf(msg, n, "step %u: the fragment order changed the result", st);
            }
        }
        if (err) { inp_free(&in); break; }
        oracle_step(&L, &d, &in, in_scale, out_scale, y);
        double *ts = malloc((size_t)NT * 8);
        for (size_t i = 0; i < NT; i++) ts[i] = d.tail[i];
        Errs e;
        e.out = nerr((const float *)buf[SB_OUT], y, VD);
        if (omega_gdn_unpack_tail_r(GDN_TEST_RATIO, (const float *)buf[SB_TAIL], tref) != 0) e.tail = INFINITY;
        else e.tail = nerr(tref, ts, NT);
        e.state = nerr((const float *)buf[SB_STATE], d.state, NS);
        free(ts);
        if (e.out > worst->out) worst->out = e.out;
        if (e.tail > worst->tail) worst->tail = e.tail;
        if (e.state > worst->state) worst->state = e.state;
        inp_free(&in);
    }
    for (int i = 0; i < SB_N; i++) free(buf[i]);
    free(tail_save); free(state_save); free(out_copy); free(tail_copy); free(state_copy); free(y); free(tref);
    dstate_free(&d); layer_free(&L);
    return err;
}
static void sim_battery(void) {
    struct { int mutant; uint32_t steps; float in_scale, out_scale; const char *name; } pts[] = {
        { 0, 3, 1.0f, 1.0f, "3 steps" },
        { 0, 2, 0.0009765625f, 18446744073709551616.0f, "2 steps, in_scale 2^-10, out_scale 2^64" },
        { 1, 2, 1.0f, 1.0f, "mutant 1 (no decay)" }, { 2, 2, 1.0f, 1.0f, "mutant 2 (beta 1)" },
        { 3, 2, 1.0f, 1.0f, "mutant 3 (gate z, not silu z)" }, { 4, 2, 1.0f, 1.0f, "mutant 4 (tail not shifted)" },
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        char msg[300] = ""; Errs w = { 0, 0, 0 };
        double t0 = now_ms();
        int err = sim_run_steps(pts[i].mutant, pts[i].steps, pts[i].in_scale, pts[i].out_scale, &w, msg, sizeof msg);
        printf("sim %-44s err out %.3g tail %.3g state %.3g (%.0f ms)%s%s\n", pts[i].name, w.out, w.tail, w.state, now_ms() - t0, err ? " ERROR " : "", msg);
        CHECK(err == 0, "sim %s ran (%s)", pts[i].name, msg);
        if (pts[i].mutant == 0) CHECK(errs_ok(&w), "sim %s within tolerance", pts[i].name);
        else CHECK(!errs_ok(&w), "sim %s caught", pts[i].name);
    }
    printf("sim warp issues: %" PRIu64 "\n", g_sim_issues);
}

/* ---- L313 bf16 state on the simulator: the bf16 program equals the f32 program run from the widened state, with the
 * stored state rounded to nearest even: outputs and tails bit-identical, every stored bf16 equal to bf16_rn of the f32
 * program's stored state. Chained over steps (the f32 run restarts each step from the bf16 state). ---- */
static uint16_t t_bf16_rn(float f) { uint32_t u; memcpy(&u, &f, 4); if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u; return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static float t_bf16_f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static int sim_one(const BlackwellIRProgram *prog, uint8_t **buf, const size_t *sz, float in_scale, float out_scale, char *msg, size_t n) {
    WsSim s; memset(&s, 0, sizeof s);
    s.p = prog; s.threads = OMEGA_GDN_THREADS; s.order = 0;
    static uint8_t smem[OMEGA_GDN_SHARED_BYTES]; s.smem = smem; s.smem_bytes = OMEGA_GDN_SHARED_BYTES;
    for (int i = 0; i < SB_N; i++) {
        const uint64_t a = (uintptr_t)buf[i];
        s.cbank[(0x380 + 8 * i) / 4] = (uint32_t)a; s.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(a >> 32);
        s.ranges[i].lo = (uintptr_t)buf[i]; s.ranges[i].hi = (uintptr_t)buf[i] + sz[i];
    }
    s.nranges = SB_N;
    memcpy(&s.cbank[OMEGA_GDN_ARG_EPS_L2 / 4], &EPS_L2, 4); memcpy(&s.cbank[OMEGA_GDN_ARG_EPS_NORM / 4], &EPS_N, 4);
    memcpy(&s.cbank[OMEGA_GDN_ARG_IN_SCALE / 4], &in_scale, 4); memcpy(&s.cbank[OMEGA_GDN_ARG_OUT_SCALE / 4], &out_scale, 4);
    int err = ws_run(&s, RT, OMEGA_GDN_GRID_Y);
    if (err) snprintf(msg, n, "%s (pc %d)", s.msg, s.err_pc);
    g_sim_issues += s.warp_issues;
    return err;
}
static void sim_bf16_battery(void) {
    static BlackwellIRProgram p32, p16;
    OmegaGdnSpec s32 = { .ratio = GDN_TEST_RATIO }, s16 = { .ratio = GDN_TEST_RATIO, .state_bf16 = 1 };
    OmegaGdnShape sh16;
    CHECK(omega_gdn_shape(&s16, &sh16) == 0 && sh16.state_bytes == T_STATE_BYTES / 2u, "bf16 shape: half the state bytes");
    CHECK(omega_blackwell_codegen_gdn_prog(&s32, &p32) == 0 && omega_blackwell_codegen_gdn_prog(&s16, &p16) == 0, "bf16 codegen");
    uint32_t f2fp = 0;
    for (size_t i = 0; i < p16.count; i++) if (p16.insns[i].op == BW_IR_FMNMX_MAX && p16.insns[i].imm == OMEGA_TC_MARK_F2FP) f2fp++;
    CHECK(f2fp == 32u && omega_gdn_bfly_count(&p16) == omega_gdn_bfly_count(&p32) + 32u, "bf16: 32 F2FP placeholders counted (%u)", f2fp);
    Layer L; layer_make(&L, 21);
    size_t sz[SB_N] = { CD * 4u, VD * 4u, VH * 8u, T_PAR_BYTES, T_TAIL_BYTES, T_STATE_BYTES, VD * 4u };
    size_t sz16[SB_N]; memcpy(sz16, sz, sizeof sz); sz16[SB_STATE] = T_STATE_BYTES / 2u;
    uint8_t *a[SB_N], *b[SB_N];
    for (int i = 0; i < SB_N; i++) { a[i] = aligned_alloc(256, (sz[i] + 255) & ~(size_t)255); b[i] = aligned_alloc(256, (sz[i] + 255) & ~(size_t)255); }
    omega_gdn_pack_params_r(GDN_TEST_RATIO, L.conv_w, L.a_log, L.dtb, L.nw, b[SB_PAR]);
    omega_gdn_pack_tail_r(GDN_TEST_RATIO, L.tail0, (float *)b[SB_TAIL]);
    uint16_t *st16 = (uint16_t *)b[SB_STATE];
    for (size_t i = 0; i < NS; i++) st16[i] = t_bf16_rn(L.state0[i]);
    memcpy(a[SB_PAR], b[SB_PAR], T_PAR_BYTES);
    int err = 0; uint32_t bad_out = 0, bad_tail = 0, bad_state = 0, rounded = 0;
    char msg[300] = "";
    for (uint32_t st = 0; st < 3 && !err; st++) {
        Inp in; inp_make(&in, 1000 + st, 1.0f);
        for (int k = 0; k < 2; k++) {
            uint8_t **x = k ? b : a;
            memcpy(x[SB_QKV], in.qkv, sz[SB_QKV]); memcpy(x[SB_Z], in.z, sz[SB_Z]); memcpy(x[SB_AB], in.ab, sz[SB_AB]);
        }
        memcpy(a[SB_TAIL], b[SB_TAIL], T_TAIL_BYTES);
        for (size_t i = 0; i < NS; i++) ((float *)a[SB_STATE])[i] = t_bf16_f(st16[i]);
        err = sim_one(&p32, a, sz, 1.0f, 1.0f, msg, sizeof msg);
        if (!err) err = sim_one(&p16, b, sz16, 1.0f, 1.0f, msg, sizeof msg);
        if (!err) {
            for (size_t i = 0; i < VD; i++) bad_out += memcmp(&((float *)a[SB_OUT])[i], &((float *)b[SB_OUT])[i], 4) != 0;
            bad_tail += memcmp(a[SB_TAIL], b[SB_TAIL], T_TAIL_BYTES) != 0;
            for (size_t i = 0; i < NS; i++) {
                const float f = ((float *)a[SB_STATE])[i];
                bad_state += st16[i] != t_bf16_rn(f);
                rounded += t_bf16_f(st16[i]) != f;
            }
        }
        inp_free(&in);
    }
    printf("sim bf16 state: 3 steps, outputs differing %u, tails differing %u, stored states != bf16_rn(f32) %u, rounded %u of %u%s%s\n",
           bad_out, bad_tail, bad_state, rounded, 3u * NS, err ? " ERROR " : "", msg);
    CHECK(err == 0, "sim bf16 ran (%s)", msg);
    CHECK(bad_out == 0 && bad_tail == 0 && bad_state == 0, "sim bf16 == f32 program + round to nearest even on store");
    CHECK(rounded > NS, "sim bf16 really rounds (%u of %u values changed)", rounded, 3u * NS);
    for (int i = 0; i < SB_N; i++) { free(a[i]); free(b[i]); }
    layer_free(&L);
}

/* ---- prefill on the simulator (omega#543): prep, scan, post over T tokens against T oracle steps ---- */
typedef struct { uint32_t off; void *p; size_t bytes; } SimBuf;
typedef struct { uint32_t off; uint32_t word; } SimWord;
static int sim_launch(const BlackwellIRProgram *prog, uint32_t threads, uint32_t gx, uint32_t gy, uint32_t smem_bytes, int order,
                      const SimBuf *b, int nb, const SimWord *w, int nw, char *msg, size_t n) {
    static uint8_t smem[OMEGA_GDN_SCAN_SHARED_BYTES];
    WsSim s; memset(&s, 0, sizeof s);
    s.p = prog; s.threads = threads; s.order = order;
    memset(smem, 0xcd, sizeof smem); /* poison: a read before a write shows */
    s.smem = smem; s.smem_bytes = smem_bytes ? smem_bytes : 4u;
    for (int i = 0; i < nb; i++) {
        const uint64_t a = (uintptr_t)b[i].p;
        s.cbank[b[i].off / 4] = (uint32_t)a; s.cbank[b[i].off / 4 + 1] = (uint32_t)(a >> 32);
        s.ranges[i].lo = a; s.ranges[i].hi = a + b[i].bytes;
    }
    s.nranges = nb;
    for (int i = 0; i < nw; i++) s.cbank[w[i].off / 4] = w[i].word;
    int err = ws_run(&s, gx, gy);
    if (err) snprintf(msg, n, "%.150s (pc %d)", s.msg, s.err_pc);
    g_sim_issues += s.warp_issues;
    return err;
}
static uint32_t fw(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static int sim_prefill(int mutant, uint32_t T, float in_scale, float out_scale, Errs *worst, char *msg, size_t n) {
    static BlackwellIRProgram pp, ps, pq;
    OmegaGdnSpec spec = { .mutant = mutant, .ratio = GDN_TEST_RATIO };
    if (omega_blackwell_codegen_gdn_prep_prog(&spec, &pp) != 0 || omega_blackwell_codegen_gdn_scan_prog(&spec, &ps) != 0 ||
        omega_blackwell_codegen_gdn_post_prog(&spec, &pq) != 0) { snprintf(msg, n, "codegen refused"); return -1; }
    Layer L; layer_make(&L, 21);
    DState d; dstate_init(&d, &L);
    const uint32_t T8 = (T + OMEGA_GDN_SCAN_CHUNK - 1u) / OMEGA_GDN_SCAN_CHUNK * OMEGA_GDN_SCAN_CHUNK;
    const size_t s_qkv = (size_t)T * CD * 4u, s_z = (size_t)T * VD * 4u, s_ab = (size_t)T * VH * 8u;
    const size_t s_rec = (size_t)T8 * VH * OMEGA_GDN_REC_BYTES, s_o = (size_t)T * VD * 4u;
    float *qkv = aligned_alloc(256, s_qkv), *z = aligned_alloc(256, s_z), *ab = aligned_alloc(256, s_ab);
    uint8_t *par = aligned_alloc(256, T_PAR_BYTES), *tail = aligned_alloc(256, T_TAIL_BYTES);
    uint8_t *state = aligned_alloc(256, T_STATE_BYTES), *rec = aligned_alloc(256, s_rec);
    float *o = aligned_alloc(256, s_o), *out = aligned_alloc(256, s_o), *out0 = malloc(s_o);
    uint8_t *tail0 = malloc(T_TAIL_BYTES), *state0 = malloc(T_STATE_BYTES);
    double *y = malloc((size_t)T * VD * 8);
    float *tref = malloc((size_t)NT * 4);
    for (uint32_t t = 0; t < T; t++) {
        Inp in; inp_make(&in, 1000 + t, in_scale);
        memcpy(qkv + (size_t)t * CD, in.qkv, CD * 4u); memcpy(z + (size_t)t * VD, in.z, VD * 4u); memcpy(ab + (size_t)t * VH * 2u, in.ab, VH * 8u);
        oracle_step(&L, &d, &in, in_scale, out_scale, y + (size_t)t * VD);
        inp_free(&in);
    }
    omega_gdn_pack_params_r(GDN_TEST_RATIO, L.conv_w, L.a_log, L.dtb, L.nw, par);
    int err = 0;
    for (int order = 0; order < 2 && !err; order++) {
        omega_gdn_pack_tail_r(GDN_TEST_RATIO, L.tail0, (float *)tail);
        memcpy(state, L.state0, T_STATE_BYTES);
        memset(rec, 0xcd, s_rec);
        for (size_t i = 0; i < (size_t)T * VD; i++) { o[i] = -999.0f; out[i] = -999.0f; }
        const SimBuf bp[] = { { OMEGA_GDN_PREP_ARG_QKV, qkv, s_qkv }, { OMEGA_GDN_PREP_ARG_AB, ab, s_ab }, { OMEGA_GDN_PREP_ARG_PAR, par, T_PAR_BYTES },
                              { OMEGA_GDN_PREP_ARG_TAIL, tail, T_TAIL_BYTES }, { OMEGA_GDN_PREP_ARG_REC, rec, s_rec } };
        const SimWord wp[] = { { OMEGA_GDN_PREP_ARG_T, T }, { OMEGA_GDN_PREP_ARG_EPS_L2, fw(EPS_L2) }, { OMEGA_GDN_PREP_ARG_IN_SCALE, fw(in_scale) } };
        const SimBuf bs[] = { { OMEGA_GDN_SCAN_ARG_REC, rec, s_rec }, { OMEGA_GDN_SCAN_ARG_QKV, qkv, s_qkv }, { OMEGA_GDN_SCAN_ARG_TAIL, tail, T_TAIL_BYTES },
                              { OMEGA_GDN_SCAN_ARG_STATE, state, T_STATE_BYTES }, { OMEGA_GDN_SCAN_ARG_O, o, s_o } };
        const SimWord wsc[] = { { OMEGA_GDN_SCAN_ARG_T, T }, { OMEGA_GDN_SCAN_ARG_IN_SCALE, fw(in_scale) } };
        const SimBuf bq[] = { { OMEGA_GDN_POST_ARG_O, o, s_o }, { OMEGA_GDN_POST_ARG_Z, z, s_z }, { OMEGA_GDN_POST_ARG_PAR, par, T_PAR_BYTES },
                              { OMEGA_GDN_POST_ARG_OUT, out, s_o } };
        const SimWord wq[] = { { OMEGA_GDN_POST_ARG_EPS_NORM, fw(EPS_N) }, { OMEGA_GDN_POST_ARG_IN_SCALE, fw(in_scale) }, { OMEGA_GDN_POST_ARG_OUT_SCALE, fw(out_scale) } };
        char m2[200] = "";
        if (sim_launch(&pp, OMEGA_GDN_PREP_THREADS, (T + OMEGA_GDN_PREP_TOKENS - 1u) / OMEGA_GDN_PREP_TOKENS, KH, OMEGA_GDN_PREP_SHARED_BYTES, order, bp, 5, wp, 3, m2, sizeof m2)) { snprintf(msg, n, "prep: %s", m2); err = -1; break; }
        if (sim_launch(&ps, OMEGA_GDN_THREADS, RT, OMEGA_GDN_GRID_Y, OMEGA_GDN_SCAN_SHARED_BYTES, order, bs, 5, wsc, 2, m2, sizeof m2)) { snprintf(msg, n, "scan: %s", m2); err = -1; break; }
        if (sim_launch(&pq, OMEGA_GDN_THREADS, T, (VH / 8u), 0, order, bq, 4, wq, 3, m2, sizeof m2)) { snprintf(msg, n, "post: %s", m2); err = -1; break; }
        size_t unwritten = 0; for (size_t i = 0; i < (size_t)T * VD; i++) if (out[i] == -999.0f) unwritten++;
        if (unwritten) { snprintf(msg, n, "%zu output words never written", unwritten); err = -1; break; }
        if (order == 0) { memcpy(out0, out, s_o); memcpy(tail0, tail, T_TAIL_BYTES); memcpy(state0, state, T_STATE_BYTES); }
        else if (memcmp(out0, out, s_o) || memcmp(tail0, tail, T_TAIL_BYTES) || memcmp(state0, state, T_STATE_BYTES)) {
            snprintf(msg, n, "the warp order changed the result"); err = -1;
        }
    }
    if (!err) {
        double *ts = malloc((size_t)NT * 8);
        for (size_t i = 0; i < NT; i++) ts[i] = d.tail[i];
        worst->out = nerr(out, y, (size_t)T * VD);
        worst->tail = omega_gdn_unpack_tail_r(GDN_TEST_RATIO, (const float *)tail, tref) != 0 ? INFINITY : nerr(tref, ts, NT);
        worst->state = nerr((const float *)state, d.state, NS);
        free(ts);
    }
    free(qkv); free(z); free(ab); free(par); free(tail); free(state); free(rec); free(o); free(out); free(out0);
    free(tail0); free(state0); free(y); free(tref);
    dstate_free(&d); layer_free(&L);
    return err;
}
/* L313: the bf16 prefill scan equals the f32 scan run from the widened state, its final state rounded to nearest even
 * (the scan holds S in registers for the whole prompt: one widen on load, one round on store). */
static void sim_prefill_bf16(uint32_t T) {
    static BlackwellIRProgram pp, p32, p16;
    OmegaGdnSpec s32 = { .ratio = GDN_TEST_RATIO }, s16 = { .ratio = GDN_TEST_RATIO, .state_bf16 = 1 };
    CHECK(omega_blackwell_codegen_gdn_prep_prog(&s32, &pp) == 0 && omega_blackwell_codegen_gdn_scan_prog(&s32, &p32) == 0 &&
          omega_blackwell_codegen_gdn_scan_prog(&s16, &p16) == 0, "bf16 prefill codegen");
    Layer L; layer_make(&L, 21);
    const uint32_t T8 = (T + OMEGA_GDN_SCAN_CHUNK - 1u) / OMEGA_GDN_SCAN_CHUNK * OMEGA_GDN_SCAN_CHUNK;
    const size_t s_qkv = (size_t)T * CD * 4u, s_ab = (size_t)T * VH * 8u, s_rec = (size_t)T8 * VH * OMEGA_GDN_REC_BYTES, s_o = (size_t)T * VD * 4u;
    float *qkv = aligned_alloc(256, s_qkv), *ab = aligned_alloc(256, s_ab);
    uint8_t *par = aligned_alloc(256, T_PAR_BYTES), *tail = aligned_alloc(256, T_TAIL_BYTES), *tail2 = aligned_alloc(256, T_TAIL_BYTES);
    uint8_t *rec = aligned_alloc(256, s_rec), *s32b = aligned_alloc(256, T_STATE_BYTES), *s16b = aligned_alloc(256, T_STATE_BYTES / 2u);
    float *o32 = aligned_alloc(256, s_o), *o16 = aligned_alloc(256, s_o);
    for (uint32_t t = 0; t < T; t++) {
        Inp in; inp_make(&in, 1000 + t, 1.0f);
        memcpy(qkv + (size_t)t * CD, in.qkv, CD * 4u); memcpy(ab + (size_t)t * VH * 2u, in.ab, VH * 8u);
        inp_free(&in);
    }
    omega_gdn_pack_params_r(GDN_TEST_RATIO, L.conv_w, L.a_log, L.dtb, L.nw, par);
    omega_gdn_pack_tail_r(GDN_TEST_RATIO, L.tail0, (float *)tail);
    memcpy(tail2, tail, T_TAIL_BYTES);
    uint16_t *st16 = (uint16_t *)s16b; float *st32 = (float *)s32b;
    for (size_t i = 0; i < NS; i++) { st16[i] = t_bf16_rn(L.state0[i]); st32[i] = t_bf16_f(st16[i]); }
    memset(rec, 0xcd, s_rec);
    char m2[200] = ""; int err = 0;
    const SimBuf bp[] = { { OMEGA_GDN_PREP_ARG_QKV, qkv, s_qkv }, { OMEGA_GDN_PREP_ARG_AB, ab, s_ab }, { OMEGA_GDN_PREP_ARG_PAR, par, T_PAR_BYTES },
                          { OMEGA_GDN_PREP_ARG_TAIL, tail, T_TAIL_BYTES }, { OMEGA_GDN_PREP_ARG_REC, rec, s_rec } };
    const SimWord wp[] = { { OMEGA_GDN_PREP_ARG_T, T }, { OMEGA_GDN_PREP_ARG_EPS_L2, fw(EPS_L2) }, { OMEGA_GDN_PREP_ARG_IN_SCALE, fw(1.0f) } };
    const SimWord wsc[] = { { OMEGA_GDN_SCAN_ARG_T, T }, { OMEGA_GDN_SCAN_ARG_IN_SCALE, fw(1.0f) } };
    err = sim_launch(&pp, OMEGA_GDN_PREP_THREADS, (T + OMEGA_GDN_PREP_TOKENS - 1u) / OMEGA_GDN_PREP_TOKENS, KH, OMEGA_GDN_PREP_SHARED_BYTES, 0, bp, 5, wp, 3, m2, sizeof m2);
    for (int k = 0; k < 2 && !err; k++) {
        const SimBuf bs[] = { { OMEGA_GDN_SCAN_ARG_REC, rec, s_rec }, { OMEGA_GDN_SCAN_ARG_QKV, qkv, s_qkv }, { OMEGA_GDN_SCAN_ARG_TAIL, k ? tail2 : tail, T_TAIL_BYTES },
                              { OMEGA_GDN_SCAN_ARG_STATE, k ? s16b : s32b, k ? T_STATE_BYTES / 2u : T_STATE_BYTES }, { OMEGA_GDN_SCAN_ARG_O, k ? o16 : o32, s_o } };
        err = sim_launch(k ? &p16 : &p32, OMEGA_GDN_THREADS, RT, OMEGA_GDN_GRID_Y, OMEGA_GDN_SCAN_SHARED_BYTES, 0, bs, 5, wsc, 2, m2, sizeof m2);
    }
    uint32_t bad_state = 0;
    for (size_t i = 0; i < NS && !err; i++) bad_state += st16[i] != t_bf16_rn(st32[i]);
    const int same_o = !err && memcmp(o32, o16, s_o) == 0, same_tail = !err && memcmp(tail, tail2, T_TAIL_BYTES) == 0;
    printf("sim prefill bf16 T %u: scan output %s, tail %s, final states != bf16_rn(f32) %u%s%s\n", T, same_o ? "identical" : "DIFFERS",
           same_tail ? "identical" : "DIFFERS", bad_state, err ? " ERROR " : "", m2);
    CHECK(err == 0 && same_o && same_tail && bad_state == 0, "sim prefill bf16 T %u == f32 scan + round on store", T);
    free(qkv); free(ab); free(par); free(tail); free(tail2); free(rec); free(s32b); free(s16b); free(o32); free(o16);
    layer_free(&L);
}
static void sim_prefill_battery(void) {
    struct { int mutant; uint32_t T; float in_scale, out_scale; const char *name; } pts[] = {
        { 0, 37, 1.0f, 1.0f, "prefill T 37" }, { 0, 3, 1.0f, 1.0f, "prefill T 3" },
        { 0, 17, 0.0009765625f, 18446744073709551616.0f, "prefill T 17, in 2^-10, out 2^64" },
        { 1, 19, 1.0f, 1.0f, "prefill mutant 1 (no decay)" }, { 2, 19, 1.0f, 1.0f, "prefill mutant 2 (beta 1)" },
        { 3, 19, 1.0f, 1.0f, "prefill mutant 3 (gate z)" }, { 4, 19, 1.0f, 1.0f, "prefill mutant 4 (tail kept)" },
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        char msg[300] = ""; Errs w = { 0, 0, 0 };
        double t0 = now_ms();
        int err = sim_prefill(pts[i].mutant, pts[i].T, pts[i].in_scale, pts[i].out_scale, &w, msg, sizeof msg);
        printf("sim %-44s err out %.3g tail %.3g state %.3g (%.0f ms)%s%s\n", pts[i].name, w.out, w.tail, w.state, now_ms() - t0, err ? " ERROR " : "", msg);
        CHECK(err == 0, "sim %s ran (%s)", pts[i].name, msg);
        if (pts[i].mutant == 0) CHECK(errs_ok(&w), "sim %s within tolerance", pts[i].name);
        else CHECK(!errs_ok(&w), "sim %s caught", pts[i].name);
    }
}

/* ---- the state copy / zero on the simulator: 3 CTAs; a 2-CTA launch leaves the last 1024 words (negative control) ---- */
static void sim_copy_battery(void) {
    static BlackwellIRProgram cp;
    enum { W = 3 * OMEGA_GDN_COPY_WORDS_PER_CTA };
    static uint32_t src[W], dst[W];
    uint32_t x = 12345u;
    for (unsigned i = 0; i < W; i++) { x = x * 1664525u + 1013904223u; src[i] = x; }
    char msg[200] = "";
    for (int kind = OMEGA_GDN_COPY; kind <= OMEGA_GDN_ZERO; kind++) {
        OmegaGdnSpec sp = { .copy = kind };
        CHECK(omega_blackwell_codegen_gdn_copy_prog(&sp, &cp) == 0, "copy kind %d codegen", kind);
        for (uint32_t gx = 3; gx >= 2; gx--) {
            memset(dst, 0xcd, sizeof dst);
            SimBuf b[2] = { { OMEGA_GDN_COPY_ARG_SRC, src, sizeof src }, { OMEGA_GDN_COPY_ARG_DST, dst, sizeof dst } };
            int err = sim_launch(&cp, OMEGA_GDN_COPY_THREADS, gx, 1, 0, 0, b, 2, NULL, 0, msg, sizeof msg);
            unsigned bad = 0;
            for (unsigned i = 0; i < W; i++) if (dst[i] != (kind == OMEGA_GDN_COPY ? src[i] : 0u)) bad++;
            CHECK(err == 0, "sim copy kind %d grid %u ran (%s)", kind, gx, msg);
            if (gx == 3) CHECK(bad == 0, "sim copy kind %d: every word (%u wrong)", kind, bad);
            else CHECK(bad == OMEGA_GDN_COPY_WORDS_PER_CTA, "sim copy kind %d grid 2 caught: %u words untouched", kind, bad);
        }
    }
}

/* ---- chip ---- */
static int chip_steps(OmegaGpuGdn *g, uint32_t layer, const Layer *L, DState *d, uint32_t steps, uint32_t seed, float in_scale,
                      float out_scale, Errs *worst, float *last_out) {
    float *out = malloc((size_t)VD * 4), *tail = malloc((size_t)NT * 4), *state = malloc((size_t)NS * 4);
    double *y = malloc((size_t)VD * 8), *td = malloc((size_t)NT * 8);
    int rc = 0;
    memset(worst, 0, sizeof *worst);
    for (uint32_t st = 0; st < steps && rc == 0; st++) {
        Inp in; inp_make(&in, seed + st, in_scale);
        rc = omega_gpu_gdn_step(g, layer, in.qkv, in.z, in.ab, EPS_L2, EPS_N, in_scale, out_scale, out, NULL);
        if (rc == 0) rc = omega_gpu_gdn_get_state(g, layer, tail, state);
        if (rc == 0) {
            oracle_step(L, d, &in, in_scale, out_scale, y);
            for (size_t i = 0; i < NT; i++) td[i] = d->tail[i];
            Errs e = { nerr(out, y, VD), nerr(tail, td, NT), nerr(state, d->state, NS) };
            if (e.out > worst->out) worst->out = e.out;
            if (e.tail > worst->tail) worst->tail = e.tail;
            if (e.state > worst->state) worst->state = e.state;
        }
        inp_free(&in);
    }
    if (last_out) memcpy(last_out, out, (size_t)VD * 4);
    free(out); free(tail); free(state); free(y); free(td);
    return rc;
}
/* ---- prefill on the chip (omega#543) ---- */
typedef struct { float *qkv, *z, *ab; } PfIn;
static void pf_in_make(PfIn *p, uint32_t T, uint32_t seed, float in_scale) {
    p->qkv = malloc((size_t)T * CD * 4); p->z = malloc((size_t)T * VD * 4); p->ab = malloc((size_t)T * VH * 8);
    for (uint32_t t = 0; t < T; t++) {
        Inp in; inp_make(&in, seed + t, in_scale);
        memcpy(p->qkv + (size_t)t * CD, in.qkv, CD * 4u); memcpy(p->z + (size_t)t * VD, in.z, VD * 4u); memcpy(p->ab + (size_t)t * VH * 2u, in.ab, VH * 8u);
        inp_free(&in);
    }
}
static void pf_in_free(PfIn *p) { free(p->qkv); free(p->z); free(p->ab); }
/* oracle over the same T rows; y [T][VD] */
static void pf_oracle(const Layer *L, DState *d, uint32_t T, uint32_t seed, float in_scale, float out_scale, double *y) {
    for (uint32_t t = 0; t < T; t++) { Inp in; inp_make(&in, seed + t, in_scale); oracle_step(L, d, &in, in_scale, out_scale, y + (size_t)t * VD); inp_free(&in); }
}
/* prefill T tokens on `layer` from its current state, then `dec` decode steps; worst errors over every row, the tail and
 * the state after the prefill and after each step */
static int chip_prefill_chain(OmegaGpuGdn *g, uint32_t layer, const Layer *L, uint32_t T, uint32_t dec, uint32_t seed, float in_scale,
                              float out_scale, Errs *worst, float *out_copy) {
    DState d; dstate_init(&d, L);
    PfIn p; pf_in_make(&p, T, seed, in_scale);
    float *out = malloc((size_t)T * VD * 4), *tail = malloc((size_t)NT * 4), *state = malloc((size_t)NS * 4);
    double *y = malloc((size_t)T * VD * 8), *td = malloc((size_t)NT * 8);
    memset(worst, 0, sizeof *worst);
    int rc = omega_gpu_gdn_prefill(g, layer, T, p.qkv, p.z, p.ab, EPS_L2, EPS_N, in_scale, out_scale, out, NULL);
    if (rc == 0) rc = omega_gpu_gdn_get_state(g, layer, tail, state);
    if (rc == 0) {
        pf_oracle(L, &d, T, seed, in_scale, out_scale, y);
        for (size_t i = 0; i < NT; i++) td[i] = d.tail[i];
        worst->out = nerr(out, y, (size_t)T * VD); worst->tail = nerr(tail, td, NT); worst->state = nerr(state, d.state, NS);
        if (out_copy) memcpy(out_copy, out, (size_t)T * VD * 4);
    }
    if (rc == 0 && dec) {
        Errs e;
        rc = chip_steps(g, layer, L, &d, dec, seed + T, in_scale, out_scale, &e, NULL);
        if (e.out > worst->out) worst->out = e.out;
        if (e.tail > worst->tail) worst->tail = e.tail;
        if (e.state > worst->state) worst->state = e.state;
    }
    free(out); free(tail); free(state); free(y); free(td);
    pf_in_free(&p); dstate_free(&d);
    return rc;
}
static void chip_prefill(OmegaGpuGdn *g, Layer *L, FILE *out, int *first, uint32_t reps) {
#define PPOINT(name, e, extra) do { if (out) { fprintf(out, "%s{\"name\":\"%s\",\"err_out\":%.6g,\"err_tail\":%.6g,\"err_state\":%.6g%s}", *first ? "" : ",", name, (e).out, (e).tail, (e).state, extra); *first = 0; } } while (0)
    struct { uint32_t layer, T, dec, seed; float in_scale, out_scale; const char *name; } pp[] = {
        { 1, 37, 3, 2000, 1.0f, 1.0f, "prefill layer 1, T 37, then 3 decode steps" },
        { 0, 3, 1, 2100, 1.0f, 1.0f, "prefill layer 0, T 3, then 1 decode step" },
        { 0, 41, 0, 2200, 0.0009765625f, 18446744073709551616.0f, "prefill layer 0, T 41, in 2^-10 out 2^64" },
    };
    for (size_t i = 0; i < sizeof pp / sizeof pp[0]; i++) {
        CHECK(omega_gpu_gdn_set_state(g, pp[i].layer, L[pp[i].layer].tail0, L[pp[i].layer].state0) == 0, "set state for %s", pp[i].name);
        Errs e;
        int rc = chip_prefill_chain(g, pp[i].layer, &L[pp[i].layer], pp[i].T, pp[i].dec, pp[i].seed, pp[i].in_scale, pp[i].out_scale, &e, NULL);
        printf("chip %-48s rc=%s err out %.3g tail %.3g state %.3g %s\n", pp[i].name, omega_gpu_gdn_rc_name(rc), e.out, e.tail, e.state, rc ? omega_gpu_gdn_last_error() : "");
        CHECK(rc == 0 && errs_ok(&e), "chip %s", pp[i].name);
        PPOINT(pp[i].name, e, "");
    }
    /* rerun from the same state: bit-identical rows */
    {
        float *a = malloc((size_t)37 * VD * 4), *b = malloc((size_t)37 * VD * 4);
        Errs e1, e2;
        CHECK(omega_gpu_gdn_set_state(g, 1, L[1].tail0, L[1].state0) == 0, "set state for prefill rerun");
        int rc = chip_prefill_chain(g, 1, &L[1], 37, 0, 2000, 1.0f, 1.0f, &e1, a);
        CHECK(omega_gpu_gdn_set_state(g, 1, L[1].tail0, L[1].state0) == 0, "set state for prefill rerun 2");
        int rc2 = chip_prefill_chain(g, 1, &L[1], 37, 0, 2000, 1.0f, 1.0f, &e2, b);
        CHECK(rc == 0 && rc2 == 0 && memcmp(a, b, (size_t)37 * VD * 4) == 0, "prefill rerun from the same state is bit-identical");
        free(a); free(b);
    }
    /* mutants */
    for (int m = 1; m <= 4; m++) {
        omega_gpu_gdn_test_set_mutant(m);
        CHECK(omega_gpu_gdn_set_state(g, 1, L[1].tail0, L[1].state0) == 0, "set state prefill mutant %d", m);
        Errs e;
        int rc = chip_prefill_chain(g, 1, &L[1], 19, 0, 2300, 1.0f, 1.0f, &e, NULL);
        char nm[40]; snprintf(nm, sizeof nm, "prefill mutant %d", m);
        printf("chip %-48s rc=%s err out %.3g tail %.3g state %.3g\n", nm, omega_gpu_gdn_rc_name(rc), e.out, e.tail, e.state);
        CHECK(rc == 0 && !errs_ok(&e), "chip prefill mutant %d caught", m);
        PPOINT(nm, e, "");
    }
    omega_gpu_gdn_test_set_mutant(0);
    /* the record path: prefill 37 tokens and one decode step in one command list on layer 0 */
    {
        const uint32_t T = 37;
        CHECK(omega_gpu_gdn_set_state(g, 0, L[0].tail0, L[0].state0) == 0, "set state for prefill record");
        CHECK(omega_gpu_gdn_prefill_reserve(g, 64) == 0, "prefill reserve 64 tokens");
        PfIn p; pf_in_make(&p, T, 2400, 1.0f);
        Inp nx; inp_make(&nx, 2400 + T, 1.0f);
        DState d; dstate_init(&d, &L[0]);
        double *y = malloc((size_t)T * VD * 8), *y1 = malloc((size_t)VD * 8);
        pf_oracle(&L[0], &d, T, 2400, 1.0f, 1.0f, y);
        oracle_step(&L[0], &d, &nx, 1.0f, 1.0f, y1);
        /* warm all four kernels on layer 1 (a cache miss inside the record is refused) */
        float *tmp = malloc((size_t)T * VD * 4);
        CHECK(omega_gpu_gdn_prefill(g, 1, T, p.qkv, p.z, p.ab, EPS_L2, EPS_N, 1.0f, 1.0f, tmp, NULL) == 0, "prefill warm-up");
        CHECK(omega_gpu_gdn_step(g, 1, nx.qkv, nx.z, nx.ab, EPS_L2, EPS_N, 1.0f, 1.0f, tmp, NULL) == 0, "decode warm-up");
        OmegaGpuCmdList *cl = NULL; /* create and free take the session lock themselves (omega_gpu_cmdlist.h) */
        const int cl_ok = omega_gpu_cmdlist_create(8, &cl) == 0;
        omega_gpu_session_lock();
        OmegaGpuMem inb; memset(&inb, 0, sizeof inb);
        const size_t s_qkv = (size_t)T * CD * 4, s_z = (size_t)T * VD * 4, s_ab = (size_t)T * VH * 8;
        const size_t o_nx = (s_qkv + s_z + s_ab + 255) & ~(size_t)255;
        int ok = cl_ok && omega_gpu_session_open() != NULL && omega_gpu_session_alloc(o_nx + (size_t)CD * 4 + VD * 4 + VH * 8, &inb) == 0;
        if (ok) {
            uint8_t *c = inb.cpu;
            memcpy(c, p.qkv, s_qkv); memcpy(c + s_qkv, p.z, s_z); memcpy(c + s_qkv + s_z, p.ab, s_ab);
            memcpy(c + o_nx, nx.qkv, (size_t)CD * 4); memcpy(c + o_nx + CD * 4, nx.z, (size_t)VD * 4); memcpy(c + o_nx + CD * 4 + VD * 4, nx.ab, VH * 8);
            ok = omega_gpu_session_sync_to_gpu(&inb) == 0;
        }
        uint64_t pf_out = 0, dec_out = 0;
        if (ok) ok = omega_gpu_gdn_prefill_record(cl, g, 0, T, inb.va, inb.va + s_qkv, inb.va + s_qkv + s_z, EPS_L2, EPS_N, 1.0f, 1.0f, &pf_out) == 0;
        if (ok) ok = omega_gpu_gdn_record(cl, g, 0, inb.va + o_nx, inb.va + o_nx + CD * 4, inb.va + o_nx + CD * 4 + VD * 4, EPS_L2, EPS_N, 1.0f, 1.0f, &dec_out) == 0;
        OmegaGpuCmdListOpts opts; memset(&opts, 0, sizeof opts);
        OmegaGpuCmdListStats stc; memset(&stc, 0, sizeof stc);
        if (ok) ok = omega_gpu_cmdlist_submit(cl, &opts, &stc) == 0;
        if (inb.cpu) omega_gpu_session_free(&inb);
        omega_gpu_session_unlock();
        if (cl) omega_gpu_cmdlist_free(cl);
        CHECK(ok, "record prefill + decode step and submit (%s)", ok ? "" : omega_gpu_gdn_last_error());
        if (ok) {
            float *t = malloc((size_t)NT * 4), *s = malloc((size_t)NS * 4), *yo = malloc((size_t)T * VD * 4), *y1o = malloc((size_t)VD * 4);
            double *td = malloc((size_t)NT * 8);
            ok = omega_gpu_gdn_get_state(g, 0, t, s) == 0 && omega_gpu_gdn_prefill_read_out(g, T, yo) == 0 && omega_gpu_gdn_read_out(g, y1o) == 0;
            for (size_t i = 0; i < NT; i++) td[i] = d.tail[i];
            Errs e = { nerr(yo, y, (size_t)T * VD), nerr(t, td, NT), nerr(s, d.state, NS) };
            const double e1 = nerr(y1o, y1, VD);
            if (e1 > e.out) e.out = e1;
            printf("chip %-48s err out %.3g tail %.3g state %.3g\n", "record path, prefill 37 + 1 decode step", e.out, e.tail, e.state);
            CHECK(ok && errs_ok(&e), "record path: prefill then a decode step match the oracle");
            PPOINT("record path, prefill 37 + 1 decode step", e, "");
            free(t); free(s); free(yo); free(y1o); free(td);
        }
        free(tmp); free(y); free(y1); dstate_free(&d); pf_in_free(&p); inp_free(&nx);
        /* the reservation pins the scratch: a longer prefill is refused, not reallocated */
        PfIn q; pf_in_make(&q, 65, 2500, 1.0f);
        float *o65 = malloc((size_t)65 * VD * 4);
        CHECK(omega_gpu_gdn_prefill(g, 1, 65, q.qkv, q.z, q.ab, EPS_L2, EPS_N, 1.0f, 1.0f, o65, NULL) == OMEGA_GPU_GDN_TOO_LARGE, "prefill past the reservation is refused");
        free(o65); pf_in_free(&q);
    }
    /* timing per launch at three prompt lengths (a fresh, unreserved scratch per length) */
    const uint32_t Ts[3] = { 512, 2048, 8192 };
    for (unsigned k = 0; k < 3; k++) {
        const uint32_t T = Ts[k], nrep = reps < 5 ? reps : 5;
        CHECK(omega_gpu_gdn_prefill_reserve(g, T) == 0, "reserve %u tokens", T);
        PfIn p; pf_in_make(&p, T, 3000, 1.0f);
        float *o = malloc((size_t)T * VD * 4);
        uint64_t e[3][5], c[5];
        uint32_t done = 0;
        OmegaGpuGdnPfInfo info;
        for (uint32_t r = 0; r < nrep; r++) {
            if (omega_gpu_gdn_prefill(g, 0, T, p.qkv, p.z, p.ab, EPS_L2, EPS_N, 1.0f, 1.0f, o, &info) != 0) break;
            for (unsigned j = 0; j < 3; j++) e[j][done] = info.elapsed_ns[j];
            c[done] = info.call_ns; done++;
        }
        CHECK(done == nrep, "prefill T %u timing reps ran (%u of %u): %s", T, done, nrep, done == nrep ? "" : omega_gpu_gdn_last_error());
        if (done) {
            for (unsigned j = 0; j < 3; j++) qsort(e[j], done, 8, cmp_u64);
            qsort(c, done, 8, cmp_u64);
            const double us[3] = { e[0][done / 2] / 1e3, e[1][done / 2] / 1e3, e[2][done / 2] / 1e3 }, tot = us[0] + us[1] + us[2];
            printf("timing prefill T %u (medians of %u): prep %.1f us, scan %.1f us, post %.1f us, chip total %.1f us = %.2f us/token, call %.1f us; insns %u/%u/%u, registers %u/%u/%u\n",
                   T, done, us[0], us[1], us[2], tot, tot / T, c[done / 2] / 1e3, info.insns[0], info.insns[1], info.insns[2], info.gprs[0], info.gprs[1], info.gprs[2]);
            if (out) fprintf(out, "%s{\"name\":\"timing prefill\",\"tokens\":%u,\"reps\":%u,\"prep_us\":%.3f,\"scan_us\":%.3f,\"post_us\":%.3f,\"chip_us\":%.3f,\"us_per_token\":%.4f,\"call_us\":%.3f,\"insns\":[%u,%u,%u],\"gprs\":[%u,%u,%u]}",
                             *first ? "" : ",", T, done, us[0], us[1], us[2], tot, tot / T, c[done / 2] / 1e3, info.insns[0], info.insns[1], info.insns[2], info.gprs[0], info.gprs[1], info.gprs[2]);
            *first = 0;
        }
        free(o); pf_in_free(&p);
    }
#undef PPOINT
}
/* ---- L313 bf16 state on the chip: a bf16 GDN (AIEN_OMEGA_GDN_STATE_BF16=1 at alloc) against an f32 GDN started from the
 * same (bf16-representable) state, 4 decode steps then a 19-token prefill, re-synced after every call: outputs and tails
 * bit-identical, the bf16 state equal to bf16_rn of the f32 state. Then the chip time of one decode step of each, interleaved. ---- */
static int chip_bf16(const char *out_path, uint32_t reps) {
    OmegaGpuGdn *g16 = NULL, *g32 = NULL;
    setenv("AIEN_OMEGA_GDN_STATE_BF16", "1", 1);
    int rc = omega_gpu_gdn_alloc_ratio(1, GDN_TEST_RATIO, &g16);
    unsetenv("AIEN_OMEGA_GDN_STATE_BF16");
    if (rc == 0) rc = omega_gpu_gdn_alloc_ratio(1, GDN_TEST_RATIO, &g32);
    CHECK(rc == 0, "bf16 + f32 gdn alloc (%s)", rc ? omega_gpu_gdn_last_error() : "");
    if (rc) return 1;
    Layer L; layer_make(&L, 41);
    float *st = malloc((size_t)NS * 4), *s16 = malloc((size_t)NS * 4), *s32 = malloc((size_t)NS * 4);
    float *t16 = malloc((size_t)NT * 4), *t32 = malloc((size_t)NT * 4);
    for (size_t i = 0; i < NS; i++) st[i] = t_bf16_f(t_bf16_rn(L.state0[i]));
    for (int k = 0; k < 2; k++) {
        OmegaGpuGdn *g = k ? g32 : g16;
        CHECK(omega_gpu_gdn_set_layer(g, 0, L.conv_w, L.a_log, L.dtb, L.nw) == 0 && omega_gpu_gdn_set_state(g, 0, L.tail0, st) == 0, "bf16 setup %d", k);
    }
    const uint32_t T = 19;
    float *o16 = malloc((size_t)T * VD * 4), *o32 = malloc((size_t)T * VD * 4);
    uint32_t bad_out = 0, bad_tail = 0, bad_state = 0, rounded = 0, calls = 0;
    double us16 = 0, us32 = 0;
    for (uint32_t c = 0; c < 5 && rc == 0; c++) {
        const uint32_t n = c < 4 ? 1u : T;
        PfIn p; pf_in_make(&p, n, 900 + 37 * c, 1.0f);
        for (int k = 0; k < 2 && rc == 0; k++) {
            OmegaGpuGdn *g = k ? g32 : g16;
            rc = n == 1 ? omega_gpu_gdn_step(g, 0, p.qkv, p.z, p.ab, EPS_L2, EPS_N, 1.0f, 1.0f, k ? o32 : o16, NULL)
                        : omega_gpu_gdn_prefill(g, 0, n, p.qkv, p.z, p.ab, EPS_L2, EPS_N, 1.0f, 1.0f, k ? o32 : o16, NULL);
        }
        if (rc == 0) rc = omega_gpu_gdn_get_state(g16, 0, t16, s16);
        if (rc == 0) rc = omega_gpu_gdn_get_state(g32, 0, t32, s32);
        if (rc == 0) {
            calls++;
            for (size_t i = 0; i < (size_t)n * VD; i++) bad_out += memcmp(&o16[i], &o32[i], 4) != 0;
            bad_tail += memcmp(t16, t32, (size_t)NT * 4) != 0;
            for (size_t i = 0; i < NS; i++) { bad_state += t_bf16_rn(s32[i]) != t_bf16_rn(s16[i]) || t_bf16_f(t_bf16_rn(s16[i])) != s16[i]; rounded += s16[i] != s32[i]; }
            rc = omega_gpu_gdn_set_state(g32, 0, t16, s16); /* re-sync: the f32 GDN continues from the bf16 state */
        }
        pf_in_free(&p);
    }
    /* chip time of one decode step, bf16 vs f32 state, interleaved (the live server shares the chip): medians of reps */
    if (rc == 0) {
        PfIn p; pf_in_make(&p, 1, 950, 1.0f);
        uint64_t *k16 = malloc((size_t)reps * 8), *k32 = malloc((size_t)reps * 8), b16 = 0, b32 = 0;
        uint32_t done = 0;
        OmegaGpuGdnInfo i16, i32;
        for (uint32_t r = 0; r < reps; r++) {
            if (omega_gpu_gdn_step(g16, 0, p.qkv, p.z, p.ab, EPS_L2, EPS_N, 1.0f, 1.0f, o16, &i16) != 0) break;
            if (omega_gpu_gdn_step(g32, 0, p.qkv, p.z, p.ab, EPS_L2, EPS_N, 1.0f, 1.0f, o32, &i32) != 0) break;
            k16[done] = i16.elapsed_ns; k32[done] = i32.elapsed_ns; b16 = i16.state_bytes; b32 = i32.state_bytes; done++;
        }
        CHECK(done == reps, "bf16 timing reps ran (%u of %u)", done, reps);
        CHECK(done == 0 || 2 * b16 == b32, "bf16 step moves half the state bytes (%" PRIu64 " vs %" PRIu64 ")", b16, b32);
        if (done) {
            qsort(k16, done, 8, cmp_u64); qsort(k32, done, 8, cmp_u64);
            const double u16 = k16[done / 2] / 1e3, u32 = k32[done / 2] / 1e3;
            printf("timing bf16 vs f32 state (medians of %u, interleaved): bf16 chip %.1f us (state read + write %.2f MB -> %.1f GB/s), "
                   "f32 chip %.1f us (%.2f MB -> %.1f GB/s); 273 GB/s nominal; bf16 %u insns %u registers\n",
                   done, u16, b16 / 1e6, b16 / (u16 * 1e3), u32, b32 / 1e6, b32 / (u32 * 1e3), i16.insns, i16.gprs);
            us16 = u16; us32 = u32;
        }
        free(k16); free(k32); pf_in_free(&p);
    }
    printf("chip bf16 state: %u calls (4 decode steps, 1 prefill of %u), rc=%s, outputs differing %u, tails differing %u, "
           "states != bf16_rn(f32) %u, rounded %u of %u; decode step chip us (median) bf16 %.1f f32 %.1f %s\n",
           calls, T, omega_gpu_gdn_rc_name(rc), bad_out, bad_tail, bad_state, rounded, calls * NS, us16, us32, rc ? omega_gpu_gdn_last_error() : "");
    CHECK(rc == 0 && calls == 5, "chip bf16 ran");
    CHECK(bad_out == 0 && bad_tail == 0 && bad_state == 0, "chip bf16 == f32 + round to nearest even on store");
    CHECK(rounded > NS, "chip bf16 really rounds");
    if (out_path) {
        FILE *f = fopen(out_path, "w");
        if (f) {
            fprintf(f, "{\"receipt\":\"OMEGA_GPU_GDN_BF16_V1\",\"ratio\":%u,\"calls\":%u,\"prefill_tokens\":%u,\"outputs_differing\":%u,"
                       "\"tails_differing\":%u,\"states_not_rn\":%u,\"rounded\":%u,\"values\":%u,\"chip_us_bf16\":%.3f,\"chip_us_f32\":%.3f,\"verdict\":\"%s\"}\n",
                    (unsigned)RT, calls, T, bad_out, bad_tail, bad_state, rounded, calls * NS, us16, us32, g_failed ? "FAIL" : "PASS");
            fclose(f);
        }
    }
    free(st); free(s16); free(s32); free(t16); free(t32); free(o16); free(o32);
    layer_free(&L); omega_gpu_gdn_free(g16); omega_gpu_gdn_free(g32);
    return g_failed ? 1 : 0;
}
static int chip_run(const char *out_path, uint32_t reps) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot write %s\n", out_path); return 1; }
    if (out) fprintf(out, "{\"receipt\":\"OMEGA_GPU_GDN_V1\",\"reps\":%u,\"tol\":{\"out\":%g,\"tail\":%g,\"state\":%g},\"points\":[", reps, TOL_OUT, TOL_TAIL, TOL_STATE);
    OmegaGpuGdn *g = NULL;
    double t0 = now_ms();
    int rc = omega_gpu_gdn_alloc_ratio(2, GDN_TEST_RATIO, &g);
    printf("gdn alloc (2 layers): rc=%s %.1f ms (%s)\n", omega_gpu_gdn_rc_name(rc), now_ms() - t0, rc ? omega_gpu_gdn_last_error() : "");
    CHECK(rc == OMEGA_GPU_GDN_OK, "gdn alloc");
    if (rc) { if (out) fprintf(out, "],\"verdict\":\"FAIL\"}\n"); return 1; }
    Layer L[2]; layer_make(&L[0], 31); layer_make(&L[1], 32);
    for (uint32_t l = 0; l < 2; l++) {
        CHECK(omega_gpu_gdn_set_layer(g, l, L[l].conv_w, L[l].a_log, L[l].dtb, L[l].nw) == 0, "set layer %u", l);
        CHECK(omega_gpu_gdn_set_state(g, l, L[l].tail0, L[l].state0) == 0, "set state %u", l);
    }
    int first = 1;
#define POINT(name, e, extra) do { if (out) { fprintf(out, "%s{\"name\":\"%s\",\"err_out\":%.6g,\"err_tail\":%.6g,\"err_state\":%.6g%s}", first ? "" : ",", name, (e).out, (e).tail, (e).state, extra); first = 0; } } while (0)
    /* parity: layer 1 first (addressing), then layer 0 with in/out scales */
    struct { uint32_t layer, steps, seed; float in_scale, out_scale; const char *name; } pp[] = {
        { 1, 5, 500, 1.0f, 1.0f, "layer 1, 5 steps" },
        { 0, 5, 600, 0.0009765625f, 18446744073709551616.0f, "layer 0, 5 steps, in 2^-10 out 2^64" },
    };
    float *o1 = malloc((size_t)VD * 4), *o2 = malloc((size_t)VD * 4);
    for (size_t i = 0; i < 2; i++) {
        DState d; dstate_init(&d, &L[pp[i].layer]);
        Errs e;
        rc = chip_steps(g, pp[i].layer, &L[pp[i].layer], &d, pp[i].steps, pp[i].seed, pp[i].in_scale, pp[i].out_scale, &e, o1);
        printf("chip %-40s rc=%s err out %.3g tail %.3g state %.3g %s\n", pp[i].name, omega_gpu_gdn_rc_name(rc), e.out, e.tail, e.state, rc ? omega_gpu_gdn_last_error() : "");
        CHECK(rc == 0 && errs_ok(&e), "chip %s", pp[i].name);
        POINT(pp[i].name, e, "");
        dstate_free(&d);
    }
    /* rerun from the same state: bit-identical */
    {
        Inp in; inp_make(&in, 700, 1.0f);
        CHECK(omega_gpu_gdn_set_state(g, 1, L[1].tail0, L[1].state0) == 0, "set state for rerun");
        rc = omega_gpu_gdn_step(g, 1, in.qkv, in.z, in.ab, EPS_L2, EPS_N, 1.0f, 1.0f, o1, NULL);
        CHECK(omega_gpu_gdn_set_state(g, 1, L[1].tail0, L[1].state0) == 0, "set state for rerun 2");
        int rc2 = omega_gpu_gdn_step(g, 1, in.qkv, in.z, in.ab, EPS_L2, EPS_N, 1.0f, 1.0f, o2, NULL);
        CHECK(rc == 0 && rc2 == 0 && memcmp(o1, o2, (size_t)VD * 4) == 0, "rerun from the same state is bit-identical");
        inp_free(&in);
    }
    /* reset: state and tail zero */
    {
        float *t = malloc((size_t)NT * 4), *s = malloc((size_t)NS * 4);
        rc = omega_gpu_gdn_reset(g);
        int z = rc == 0 && omega_gpu_gdn_get_state(g, 0, t, s) == 0;
        for (size_t i = 0; z && i < NT; i++) if (t[i] != 0.0f) z = 0;
        for (size_t i = 0; z && i < NS; i++) if (s[i] != 0.0f) z = 0;
        CHECK(z, "reset zeroes the tail and the state");
        free(t); free(s);
    }
    /* mutants: each must miss */
    for (int m = 1; m <= 4; m++) {
        omega_gpu_gdn_test_set_mutant(m);
        CHECK(omega_gpu_gdn_set_state(g, 1, L[1].tail0, L[1].state0) == 0, "set state mutant %d", m);
        DState d; dstate_init(&d, &L[1]);
        Errs e;
        rc = chip_steps(g, 1, &L[1], &d, 2, 800, 1.0f, 1.0f, &e, NULL);
        char nm[32]; snprintf(nm, sizeof nm, "mutant %d", m);
        printf("chip %-40s rc=%s err out %.3g tail %.3g state %.3g\n", nm, omega_gpu_gdn_rc_name(rc), e.out, e.tail, e.state);
        CHECK(rc == 0 && !errs_ok(&e), "chip mutant %d caught", m);
        POINT(nm, e, "");
        dstate_free(&d);
    }
    omega_gpu_gdn_test_set_mutant(0);
    /* the record path: two chained steps in one command list on layer 0 */
    {
        CHECK(omega_gpu_gdn_set_state(g, 0, L[0].tail0, L[0].state0) == 0, "set state for record");
        Inp in[2]; inp_make(&in[0], 900, 1.0f); inp_make(&in[1], 901, 1.0f);
        DState d; dstate_init(&d, &L[0]);
        double *y = malloc((size_t)VD * 8);
        oracle_step(&L[0], &d, &in[0], 1.0f, 1.0f, y); oracle_step(&L[0], &d, &in[1], 1.0f, 1.0f, y);
        /* warm the kernel cache outside the record (a miss inside is refused) */
        float *tmp = malloc((size_t)VD * 4);
        CHECK(omega_gpu_gdn_step(g, 1, in[0].qkv, in[0].z, in[0].ab, EPS_L2, EPS_N, 1.0f, 1.0f, tmp, NULL) == 0, "warm-up step");
        OmegaGpuCmdList *cl = NULL; /* create and free take the session lock themselves (omega_gpu_cmdlist.h) */
        const int cl_ok = omega_gpu_cmdlist_create(4, &cl) == 0;
        omega_gpu_session_lock();
        OmegaGpuMem inb[2]; memset(inb, 0, sizeof inb);
        int ok = cl_ok && omega_gpu_session_open() != NULL;
        const size_t ib = (size_t)CD * 4 + (size_t)VD * 4 + VH * 8;
        for (int i = 0; ok && i < 2; i++) {
            ok = omega_gpu_session_alloc(ib, &inb[i]) == 0;
            if (ok) {
                memcpy(inb[i].cpu, in[i].qkv, (size_t)CD * 4); memcpy((uint8_t *)inb[i].cpu + CD * 4, in[i].z, (size_t)VD * 4);
                memcpy((uint8_t *)inb[i].cpu + CD * 4 + VD * 4, in[i].ab, VH * 8);
                ok = omega_gpu_session_sync_to_gpu(&inb[i]) == 0;
            }
        }
        uint64_t out_va = 0;
        for (int i = 0; ok && i < 2; i++)
            ok = omega_gpu_gdn_record(cl, g, 0, inb[i].va, inb[i].va + CD * 4, inb[i].va + CD * 4 + VD * 4, EPS_L2, EPS_N, 1.0f, 1.0f, &out_va) == 0;
        OmegaGpuCmdListOpts opts; memset(&opts, 0, sizeof opts);
        OmegaGpuCmdListStats stc; memset(&stc, 0, sizeof stc);
        if (ok) ok = omega_gpu_cmdlist_submit(cl, &opts, &stc) == 0;
        for (int i = 0; i < 2; i++) if (inb[i].cpu) omega_gpu_session_free(&inb[i]);
        omega_gpu_session_unlock();
        if (cl) omega_gpu_cmdlist_free(cl);
        CHECK(ok, "record two steps and submit (%s)", ok ? "" : omega_gpu_gdn_last_error());
        if (ok) {
            float *t = malloc((size_t)NT * 4), *s = malloc((size_t)NS * 4), *yo = malloc((size_t)VD * 4);
            double *td = malloc((size_t)NT * 8);
            ok = omega_gpu_gdn_get_state(g, 0, t, s) == 0 && omega_gpu_gdn_read_out(g, yo) == 0; /* the second step's output */
            for (size_t i = 0; i < NT; i++) td[i] = d.tail[i];
            Errs e = { nerr(yo, y, VD), nerr(t, td, NT), nerr(s, d.state, NS) };
            printf("chip %-40s err out %.3g tail %.3g state %.3g\n", "record path, 2 chained steps", e.out, e.tail, e.state);
            CHECK(ok && errs_ok(&e), "record path: two chained steps match the oracle");
            POINT("record path, 2 chained steps", e, "");
            free(t); free(s); free(yo); free(td);
        }
        free(tmp); free(y); dstate_free(&d); inp_free(&in[0]); inp_free(&in[1]);
    }
    chip_prefill(g, L, out, &first, reps);
    /* timing */
    {
        Inp in; inp_make(&in, 950, 1.0f);
        uint64_t *ec = malloc(reps * 8), *ek = malloc(reps * 8);
        uint32_t done = 0;
        OmegaGpuGdnInfo info;
        for (uint32_t r = 0; r < reps; r++) {
            if (omega_gpu_gdn_step(g, 0, in.qkv, in.z, in.ab, EPS_L2, EPS_N, 1.0f, 1.0f, o1, &info) != 0) break;
            ek[done] = info.elapsed_ns; ec[done] = info.call_ns; done++;
        }
        CHECK(done == reps, "timing reps ran (%u of %u)", done, reps);
        if (done) {
            qsort(ek, done, 8, cmp_u64); qsort(ec, done, 8, cmp_u64);
            const double kus = ek[done / 2] / 1e3, cus = ec[done / 2] / 1e3, gbs = (double)info.state_bytes / (ek[done / 2] * 1e-9) / 1e9;
            printf("timing (medians of %u): chip %.1f us (state read + write %.2f MB -> %.1f GB/s; 273 GB/s nominal), call %.1f us; %u insns, %u registers\n",
                   done, kus, info.state_bytes / 1e6, gbs, cus, info.insns, info.gprs);
            if (out) fprintf(out, "%s{\"name\":\"timing\",\"reps\":%u,\"chip_us_median\":%.3f,\"call_us_median\":%.3f,\"state_bytes\":%" PRIu64 ",\"state_gbs\":%.2f,\"nominal_gbs\":273,\"insns\":%u,\"gprs\":%u}",
                             first ? "" : ",", done, kus, cus, info.state_bytes, gbs, info.insns, info.gprs);
            first = 0;
        }
        free(ec); free(ek); inp_free(&in);
    }
#undef POINT
    free(o1); free(o2);
    layer_free(&L[0]); layer_free(&L[1]);
    omega_gpu_gdn_free(g);
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS"); fclose(out); }
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0); /* a stuck chip run still leaves every finished line in its log */
    int bf16 = 0;
    int host = 0, sim = 0; const char *out_path = NULL; uint32_t reps = 0, spin = 2000;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--bf16") == 0) bf16 = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--spin-us") == 0 && i + 1 < argc) spin = (uint32_t)atoi(argv[++i]);
        else { printf("unknown argument %s\n", argv[i]); return 2; }
    }
    if (host) host_only();
    if (sim) { sim_battery(); sim_bf16_battery(); sim_prefill_battery(); sim_prefill_bf16(3); sim_prefill_bf16(37); sim_copy_battery(); }
    if (!host && !sim) {
        (void)omega_gpu_session_set_spin_us(spin);
        if (bf16) chip_bf16(out_path, reps ? reps : 50); else chip_run(out_path, reps ? reps : 50);
        omega_gpu_session_close();
    }
    printf("gpu_gdn_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
