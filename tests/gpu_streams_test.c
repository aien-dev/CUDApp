/* gpu_streams_test.c -- row C4 of docs/OMEGA.md: two independent queues and an event between them, on the GB10.
 *
 *   --host-only  codegen + encode of the two kernels (and their mutants), refusals of omega_gpu_cmdlist_wait_for
 *   --sim        the kernels through the warp simulator (tests/bw_warp_sim.h): exact values, mutants caught
 *   (no flag)    the chip, under the GB10 lock, AIEN_OMEGA_GPU_CHANNELS = 2 (L136 extra queue):
 *     overlap    list A and list B each hold the same half-chip spin kernel (24 CTAs x 128 threads); solo A, solo B
 *                (session queue, one submit each) and the pair (A on queue 0, B on queue 1, omega_gpu_cmdlist_submit_pair),
 *                medians of REPS: PASS when pair < 0.75 (solo A + solo B), for the default barrier or first_barrier_none
 *                (both are printed). Red check: both spins in ONE list on one queue (WAIT_FOR_IDLE between them) must
 *                NOT come in under 0.75, or the measurement cannot tell overlap from order.
 *     event      A: a long spin writes X. B: omega_gpu_cmdlist_wait_for(A, 0), then copy Y = X + 1. X and Y start as a
 *                signalling-NaN poison; every Y must be the spin's value + 1. Negative control: the same pair without the
 *                wait must leave at least one Y wrong (B ran while A spun), else the event is "not demonstrated" (FAIL).
 * Every kernel comes from Omega's IR (omega_bw_em.h), encoded by our own sm_121 encoder; no CUDA. */
#include "omega_gpu_cmdlist.h"
#include "omega_gpu_session.h"
#include "bw_warp_sim.h" /* brings math.h, whose fadd() would hide the emitter's */
#define fadd em_fadd
#define ffma em_ffma
#include "omega_bw_em.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define THREADS 128u
#define CTAS 24u                 /* half of the GB10's 48 SMs */
#define NEL (THREADS * CTAS)
#define UNROLL 4u                /* FFMAs per loop turn: x = x * 1 + 1 */
#define MAX_ITERS ((1u << 22) - 1024u) /* UNROLL * iters + tag stays below 2^24: every value exact in f32 */
#define POISON 0x7fa00badu       /* a signalling NaN no FFMA / FADD writes */
#define REPS 5

/* argument words (constant bank 0x380) */
#define A_OUT 0x380   /* spin: out u64 | copy: in u64 */
#define A_ITERS 0x388 /* spin: loop turns | copy: out u64 (0x388..0x38f) */
#define A_TAG 0x38c   /* spin: f32 added at the end */
#define A_N 0x390     /* elements */
#define ARGS 5u

static int g_failed, g_checks;
static void check(int ok, const char *what) {
    g_checks++;
    if (!ok) g_failed++;
    printf("%s: %s\n", ok ? "ok" : "FAIL", what);
}
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* ------------------------------------------------------------------ the kernels */

typedef struct { int bssy, bra; } Region;
static Region skip_if_p0(Em *e) { Region r; r.bssy = bssy_fwd(e); r.bra = bra_if_p0_fwd(e); return r; }
static void region_end(Em *e, Region r) { int at = here(e); bsync(e); patch_bra(e, r.bra, at); patch_bra(e, r.bssy, at + 1); }

/* spin: out[gid] = UNROLL * iters + tag (mutant 1: one FFMA short per turn) */
static int gen_spin(Em *e, int mutant) {
    int udesc = UV64(e), pout = V64(e), a64 = V64(e), iters = V(e), tag = V(e), n = V(e), tid = V(e), cta = V(e), gid = V(e);
    int i = V(e), one = V(e), fone = V(e), x = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pout, A_OUT); ldc32(e, iters, A_ITERS); ldc32(e, tag, A_TAG); ldc32(e, n, A_N);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cta, BW_SR_CTAID_X);
    imadi(e, gid, cta, 0, THREADS, tid);
    movi(e, one, 1); movi(e, fone, 0x3f800000u); movi(e, x, 0); movi(e, i, 0);
    const int top = here(e);
    loop_begin(e);
    isetp_ge_u32(e, i, iters);
    const int done = bra_if_p0_fwd(e);
    for (unsigned u = 0; u < UNROLL - (mutant == 1 ? 1u : 0u); u++) ffma(e, x, x, fone, 0, fone);
    iadd(e, i, i, one);
    bra(e, top);
    loop_end(e);
    patch_bra(e, done, here(e));
    fadd(e, x, x, tag);
    isetp_ge_u32(e, gid, n);
    { Region r = skip_if_p0(e); imadw(e, a64, gid, 4u, pout); stg(e, a64, 0, x, udesc); region_end(e, r); }
    tail(e);
    return e->err;
}

/* copy: out[gid] = in[gid] + 1 (mutant 1: + 2) */
static int gen_copy(Em *e, int mutant) {
    int udesc = UV64(e), pin = V64(e), pout = V64(e), a64 = V64(e), n = V(e), tid = V(e), cta = V(e), gid = V(e), add = V(e), v = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pin, A_OUT); ldc64(e, pout, A_ITERS); ldc32(e, n, A_N);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cta, BW_SR_CTAID_X);
    imadi(e, gid, cta, 0, THREADS, tid);
    movi(e, add, mutant == 1 ? 0x40000000u : 0x3f800000u);
    isetp_ge_u32(e, gid, n);
    {
        Region r = skip_if_p0(e);
        imadw(e, a64, gid, 4u, pin); ldg32(e, v, a64, udesc, 1);
        fadd(e, v, v, add);
        imadw(e, a64, gid, 4u, pout); stg(e, a64, 0, v, udesc);
        region_end(e, r);
    }
    tail(e);
    return e->err;
}

static int codegen(int copy, int mutant, BlackwellIRProgram *prog) {
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    if ((copy ? gen_copy(&em, mutant) : gen_spin(&em, mutant)) != 0 || em.err || em.depth != 0) return -1;
    return omega_bw_regalloc_solve(prog);
}

static void spin_args(uint32_t a[ARGS], uint64_t out, uint32_t iters, float tag) {
    memset(a, 0, ARGS * 4u);
    a[0] = (uint32_t)out; a[1] = (uint32_t)(out >> 32); a[2] = iters; a[3] = f2u(tag); a[4] = NEL;
}
static void copy_args(uint32_t a[ARGS], uint64_t in, uint64_t out) {
    memset(a, 0, ARGS * 4u);
    a[0] = (uint32_t)in; a[1] = (uint32_t)(in >> 32); a[2] = (uint32_t)out; a[3] = (uint32_t)(out >> 32); a[4] = NEL;
}
static float spin_value(uint32_t iters, float tag) { return (float)(UNROLL * iters) + tag; }

/* ------------------------------------------------------------------ host + simulator */

static void host_only(void) {
    char w[200];
    for (int copy = 0; copy < 2; copy++)
        for (int mut = 0; mut < 2; mut++) {
            BlackwellIRProgram *p = calloc(1, sizeof *p);
            size_t need = 0;
            int ok = p && codegen(copy, mut, p) == 0 && (need = omega_bw_program_code_bytes(p)) > 0;
            void *code = ok ? calloc(1, need) : NULL;
            OmegaBlackwellKernel kk;
            ok = ok && code && omega_blackwell_kernel_encode_into(p, code, need, &kk) == 0 && kk.code_size == need;
            snprintf(w, sizeof w, "codegen + encode: %s kernel%s (%zu bytes, %u registers)", copy ? "copy" : "spin", mut ? " mutant 1" : "",
                     need, ok ? kk.gpr_count : 0u);
            check(ok, w);
            free(code); free(p);
        }
    check(omega_gpu_cmdlist_wait_for(NULL, NULL, 0) == -1, "wait_for refuses missing lists");
}

static int sim_run(BlackwellIRProgram *p, const uint32_t a[ARGS], WsRange *rg, int nrg, int order) {
    static WsSim s;
    memset(&s, 0, sizeof s);
    s.p = p; s.threads = THREADS; s.order = order;
    for (unsigned i = 0; i < ARGS; i++) s.cbank[0x380 / 4 + i] = a[i];
    memcpy(s.ranges, rg, (size_t)nrg * sizeof *rg); s.nranges = nrg;
    if (ws_run(&s, CTAS, 1) != 0) { printf("  simulator: %s at pc %d\n", s.msg, s.err_pc); return -1; }
    return 0;
}

static void sim_tier(void) {
    char w[200];
    static float X[NEL], Y[NEL];
    WsRange rg[2] = { { (uintptr_t)X, (uintptr_t)(X + NEL) }, { (uintptr_t)Y, (uintptr_t)(Y + NEL) } };
    for (int mut = 0; mut < 2; mut++) {
        BlackwellIRProgram *ps = calloc(1, sizeof *ps), *pc = calloc(1, sizeof *pc);
        if (!ps || !pc || codegen(0, mut, ps) || codegen(1, mut, pc)) { check(0, "simulator: kernels built"); free(ps); free(pc); return; }
        for (int order = 0; order < 2; order++) {
            const uint32_t iters = 37u + (uint32_t)order;
            const float tag = 5.0f + (float)order;
            uint32_t a[ARGS];
            for (unsigned i = 0; i < NEL; i++) { uint32_t pz = POISON; memcpy(&X[i], &pz, 4); memcpy(&Y[i], &pz, 4); }
            spin_args(a, (uintptr_t)X, iters, tag);
            int rc = sim_run(ps, a, rg, 2, order);
            copy_args(a, (uintptr_t)X, (uintptr_t)Y);
            rc = rc || sim_run(pc, a, rg, 2, order);
            unsigned wx = 0, wy = 0;
            for (unsigned i = 0; i < NEL; i++) {
                wx += f2u(X[i]) != f2u(spin_value(iters, tag));
                wy += f2u(Y[i]) != f2u(spin_value(iters, tag) + 1.0f);
            }
            if (mut == 0) snprintf(w, sizeof w, "simulator order %d: spin + copy exact on all %u elements (%u, %u wrong)", order, NEL, wx, wy);
            else snprintf(w, sizeof w, "simulator red check order %d: mutant 1 caught (%u, %u wrong)", order, wx, wy);
            check(mut == 0 ? rc == 0 && wx == 0 && wy == 0 : rc != 0 || (wx > 0 && wy > 0), w);
        }
        free(ps); free(pc);
    }
}

/* ------------------------------------------------------------------ the chip */

typedef struct { BlackwellIRProgram prog; OmegaGpuMem code; uint32_t gpr; } Kern;
static Kern K_spin, K_copy;

static int kern_upload(Kern *k, int copy) {
    if (codegen(copy, 0, &k->prog) != 0) return -1;
    size_t need = omega_bw_program_code_bytes(&k->prog);
    if (need == 0 || omega_gpu_session_alloc_code(need, &k->code) != 0) return -1;
    OmegaBlackwellKernel kk;
    if (omega_blackwell_kernel_encode_into(&k->prog, k->code.cpu, need, &kk) != 0 || omega_gpu_session_sync_to_gpu(&k->code) != 0) return -1;
    k->gpr = omega_bw_gpr_declared(kk.gpr_count);
    if (k->gpr < 64u) k->gpr = 64u;
    return 0;
}

static OmegaGpuMem g_buf; /* [OA | OB | X | Y], NEL floats each */
static float *buf_at(int i) { return (float *)g_buf.cpu + (size_t)i * NEL; }
static uint64_t va_at(int i) { return g_buf.va + (uint64_t)i * NEL * 4u; }
enum { B_OA = 0, B_OB = 1, B_X = 2, B_Y = 3 };

static void poison_all(void) {
    uint32_t *u = g_buf.cpu;
    for (size_t i = 0; i < 4u * NEL; i++) u[i] = POISON;
    (void)omega_gpu_session_sync_to_gpu(&g_buf);
}
static unsigned wrong(int b, float want) {
    (void)omega_gpu_session_sync_to_cpu(&g_buf);
    const float *p = buf_at(b);
    unsigned n = 0;
    for (unsigned i = 0; i < NEL; i++) n += f2u(p[i]) != f2u(want);
    return n;
}

static uint32_t g_args[8][ARGS]; /* the launches' argument words stay alive until submit */
static int add_spin(OmegaGpuCmdList *cl, int slot, int out_buf, uint32_t iters, float tag) {
    spin_args(g_args[slot], va_at(out_buf), iters, tag);
    OmegaGpuLaunch L = { .code_va = K_spin.code.va, .gpr_count = K_spin.gpr, .threads_x = THREADS, .threads_y = 1, .grid_x = CTAS, .grid_y = 1,
                         .num_elements = NEL, .args = g_args[slot], .n_args = ARGS };
    return omega_gpu_cmdlist_add(cl, &L) < 0 ? -1 : 0;
}
static int add_copy(OmegaGpuCmdList *cl, int slot) {
    copy_args(g_args[slot], va_at(B_X), va_at(B_Y));
    OmegaGpuLaunch L = { .code_va = K_copy.code.va, .gpr_count = K_copy.gpr, .threads_x = THREADS, .threads_y = 1, .grid_x = CTAS, .grid_y = 1,
                         .num_elements = NEL, .args = g_args[slot], .n_args = ARGS };
    return omega_gpu_cmdlist_add(cl, &L) < 0 ? -1 : 0;
}

static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static uint64_t median(uint64_t *v, int n) { qsort(v, (size_t)n, sizeof *v, cmp_u64); return v[n / 2]; }

static OmegaGpuCmdList *g_a, *g_b, *g_ab;

/* one solo submit of a one-spin list; values checked; elapsed ns or 0 on failure */
static uint64_t solo(OmegaGpuCmdList *cl, int out_buf, uint32_t iters, float tag, const OmegaGpuCmdListOpts *o) {
    OmegaGpuCmdListStats st;
    poison_all();
    if (omega_gpu_cmdlist_submit(cl, o, &st) != 0) { printf("  solo submit: %s\n", omega_gpu_session_last_error()); return 0; }
    unsigned bad = wrong(out_buf, spin_value(iters, tag));
    if (bad) { printf("  solo: %u wrong values\n", bad); return 0; }
    return st.elapsed_ns;
}

static int overlap(uint32_t iters, int fbn, double *ratio_out) {
    char w[300];
    OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o);
    o.first_barrier_none = fbn;
    uint64_t sa[REPS], sb[REPS], sp[REPS];
    for (int r = 0; r < REPS; r++) {
        sa[r] = solo(g_a, B_OA, iters, 1.0f, &o);
        sb[r] = solo(g_b, B_OB, iters, 2.0f, &o);
        OmegaGpuCmdListStats s1, s2;
        poison_all();
        int rc = omega_gpu_cmdlist_submit_pair(g_a, g_b, &o, &s1, &s2);
        unsigned bad = rc ? NEL : wrong(B_OA, spin_value(iters, 1.0f)) + wrong(B_OB, spin_value(iters, 2.0f));
        if (rc || bad) { printf("  pair: %s, %u wrong values\n", rc ? omega_gpu_session_last_error() : "ok", bad); sp[r] = 0; }
        else sp[r] = s1.elapsed_ns;
        if (!sa[r] || !sb[r] || !sp[r]) { check(0, "overlap: every submit ran and every value is right"); return -1; }
    }
    const uint64_t ma = median(sa, REPS), mb = median(sb, REPS), mp = median(sp, REPS);
    const double ratio = (double)mp / (double)(ma + mb);
    *ratio_out = ratio;
    snprintf(w, sizeof w, "overlap (%s): solo A %.3f ms, solo B %.3f ms, pair %.3f ms (medians of %d, values right): pair / (A + B) = %.3f",
             fbn ? "first_barrier_none" : "default barrier", ma / 1e6, mb / 1e6, mp / 1e6, REPS, ratio);
    printf("%s\n", w);
    return 0;
}

static int serial_control(uint32_t iters, double *ratio_out) {
    OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o);
    uint64_t sa[REPS], sab[REPS];
    for (int r = 0; r < REPS; r++) {
        sa[r] = solo(g_a, B_OA, iters, 1.0f, &o);
        OmegaGpuCmdListStats st;
        poison_all();
        int rc = omega_gpu_cmdlist_submit(g_ab, &o, &st);
        unsigned bad = rc ? NEL : wrong(B_OA, spin_value(iters, 1.0f)) + wrong(B_OB, spin_value(iters, 2.0f));
        if (!sa[r] || rc || bad) { check(0, "serial control: every submit ran and every value is right"); return -1; }
        sab[r] = st.elapsed_ns;
    }
    const uint64_t ma = median(sa, REPS), mab = median(sab, REPS);
    *ratio_out = (double)mab / (double)(2 * ma);
    printf("serial control: solo %.3f ms, both in one list on one queue %.3f ms: ratio %.3f\n", ma / 1e6, mab / 1e6, *ratio_out);
    return 0;
}

/* list A: spin into X; list B: [wait for A launch 0] copy X -> Y. Returns the count of wrong Y, or -1 on a submit failure. */
static long event_run(uint32_t iters, int with_wait, int fbn) {
    OmegaGpuCmdList *a = NULL, *b = NULL;
    long res = -1;
    if (omega_gpu_cmdlist_create(1, &a) || omega_gpu_cmdlist_create(1, &b)) { printf("  event: create: %s\n", omega_gpu_session_last_error()); goto out; }
    omega_gpu_session_lock();
    if (add_spin(a, 4, B_X, iters, 3.0f) || (with_wait && omega_gpu_cmdlist_wait_for(b, a, 0)) || add_copy(b, 5)) {
        omega_gpu_session_unlock(); printf("  event: add: %s\n", omega_gpu_session_last_error()); goto out;
    }
    OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o);
    o.first_barrier_none = fbn;
    OmegaGpuCmdListStats s1, s2;
    if (with_wait) { /* the refusals: a list with a wait never runs alone, nor waits on a second list */
        OmegaGpuCmdList *c = NULL;
        int alone = omega_gpu_cmdlist_submit(b, &o, &s2) == -1 && strstr(omega_gpu_session_last_error(), "only in a pair") != NULL;
        omega_gpu_session_unlock();
        int third = omega_gpu_cmdlist_create(1, &c) == 0;
        omega_gpu_session_lock();
        third = third && add_spin(c, 6, B_OA, 1, 0.0f) == 0 && omega_gpu_cmdlist_wait_for(b, c, 0) == -1;
        omega_gpu_session_unlock();
        omega_gpu_cmdlist_free(c);
        omega_gpu_session_lock();
        check(alone && third, "event refusals: a list with a wait is refused alone, and cannot wait on a second list");
    }
    poison_all();
    int rc = omega_gpu_cmdlist_submit_pair(a, b, &o, &s1, &s2);
    if (rc) printf("  event: pair: %s\n", omega_gpu_session_last_error());
    else {
        unsigned bx = wrong(B_X, spin_value(iters, 3.0f)), by = wrong(B_Y, spin_value(iters, 3.0f) + 1.0f);
        printf("  event (%s, %s): X %u wrong, Y %u wrong of %u, pair %.3f ms\n", with_wait ? "wait" : "NO wait", fbn ? "first_barrier_none" : "default barrier",
               bx, by, NEL, s1.elapsed_ns / 1e6);
        res = bx ? -1 : (long)by;
    }
    omega_gpu_session_unlock();
out:
    omega_gpu_cmdlist_free(a); omega_gpu_cmdlist_free(b);
    return res;
}

static void chip_tier(int values_only) {
    char w[300];
    setenv("AIEN_OMEGA_GPU_CHANNELS", "2", 1);
    if (omega_gpu_cmdlist_create(1, &g_a) || omega_gpu_cmdlist_create(1, &g_b) || omega_gpu_cmdlist_create(2, &g_ab)) {
        printf("NOT_RUN: command lists: %s\n", omega_gpu_session_last_error()); exit(2);
    }
    omega_gpu_session_lock();
    if (!omega_gpu_session_open() || omega_gpu_session_alloc(4u * NEL * 4u, &g_buf) || kern_upload(&K_spin, 0) || kern_upload(&K_copy, 1)) {
        printf("NOT_RUN: device, buffer or kernels: %s\n", omega_gpu_session_last_error()); omega_gpu_session_unlock(); exit(2);
    }
    OmegaGpuQueueHandle q1;
    check(omega_gpu_session_extra_queue(1, &q1) == 0, "the second queue opens (AIEN_OMEGA_GPU_CHANNELS = 2)");

    /* calibrate: one spin of about 60 ms */
    uint32_t iters = 1u << 16;
    OmegaGpuCmdListOpts o0; memset(&o0, 0, sizeof o0);
    omega_gpu_cmdlist_reset(g_a);
    if (add_spin(g_a, 0, B_OA, iters, 1.0f)) { check(0, "calibration list"); omega_gpu_session_unlock(); return; }
    uint64_t t = solo(g_a, B_OA, iters, 1.0f, &o0);
    if (!t) { check(0, "calibration spin ran with right values"); omega_gpu_session_unlock(); return; }
    double scaled = (double)iters * 60e6 / (double)t;
    iters = scaled > MAX_ITERS ? MAX_ITERS : (uint32_t)scaled;
    printf("calibration: 2^16 turns took %.3f ms; using %u turns\n", t / 1e6, iters);

    omega_gpu_cmdlist_reset(g_a); omega_gpu_cmdlist_reset(g_b); omega_gpu_cmdlist_reset(g_ab);
    if (add_spin(g_a, 0, B_OA, iters, 1.0f) || add_spin(g_b, 1, B_OB, iters, 2.0f) ||
        add_spin(g_ab, 2, B_OA, iters, 1.0f) || add_spin(g_ab, 3, B_OB, iters, 2.0f)) { check(0, "timing lists"); omega_gpu_session_unlock(); return; }

    if (!values_only) {
    double r_def = 9, r_fbn = 9, r_ser = 0;
    int ok_def = overlap(iters, 0, &r_def) == 0, ok_fbn = overlap(iters, 1, &r_fbn) == 0, ok_ser = serial_control(iters, &r_ser) == 0;
    snprintf(w, sizeof w, "C4 overlap: two queues of the same work finish in under 0.75 of the solo sum (default %.3f, first_barrier_none %.3f)", r_def, r_fbn);
    check((ok_def && r_def < 0.75) || (ok_fbn && r_fbn < 0.75), w);
    snprintf(w, sizeof w, "red check: the same two spins in order on one queue do NOT pass the 0.75 bar (ratio %.3f)", r_ser);
    check(ok_ser && r_ser >= 0.75, w);
    } else puts("C4 overlap: NOT_RUN (values-only; timing needs an exclusive window)");
    omega_gpu_session_unlock();

    /* the event: positive, then the control without the wait (both barrier modes) */
    long pos = event_run(iters, 1, 0);
    snprintf(w, sizeof w, "C4 event: B's copy waited for A's spin on the other queue; every Y right (%ld wrong)", pos);
    check(pos == 0, w);
    long neg0 = event_run(iters, 0, 0), neg1 = neg0 > 0 ? neg0 : event_run(iters, 0, 1);
    snprintf(w, sizeof w, "negative control: without the wait B ran early and Y is wrong (default barrier %ld wrong, first_barrier_none %ld wrong)", neg0, neg1);
    check(neg0 > 0 || neg1 > 0, w);
    long again = event_run(iters, 1, 1);
    snprintf(w, sizeof w, "C4 event with first_barrier_none: every Y right (%ld wrong)", again);
    check(again == 0, w);

    omega_gpu_session_lock();
    omega_gpu_session_free(&g_buf); omega_gpu_session_free(&K_spin.code); omega_gpu_session_free(&K_copy.code);
    omega_gpu_session_unlock();
    omega_gpu_cmdlist_free(g_a); omega_gpu_cmdlist_free(g_b); omega_gpu_cmdlist_free(g_ab);
}

int main(int argc, char **argv) {
    const char *tier = "chip";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) tier = "host";
        else if (!strcmp(argv[i], "--values-only")) tier = "values";
        else if (!strcmp(argv[i], "--sim")) tier = "sim";
        else { fprintf(stderr, "usage: %s [--host-only | --sim | --values-only]\n", argv[0]); return 2; }
    }
    if (!strcmp(tier, "host")) host_only();
    else if (!strcmp(tier, "sim")) { host_only(); sim_tier(); }
    else chip_tier(!strcmp(tier, "values"));
    if (g_failed) printf("C4 streams %s: FAIL (%d of %d)\n", tier, g_failed, g_checks);
    else printf("C4 streams %s: PASS (%d checks)\n", tier, g_checks);
    return g_failed ? 1 : 0;
}
