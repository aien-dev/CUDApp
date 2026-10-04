/* omega_gpu_elementwise_api.c: FB-1 cut 4 (2026-10-04). See the header.
 *
 * Layout of this file: (1) kernel codegen on top of the BlackwellIR, with a
 * small scheduler that writes every control word; (2) one launcher cloned from
 * the chip-proven tensor matmul path (omega_blackwell_execute_matmul_tensor);
 * (3) the public entry points, which split work into <= OMEGA_GPU_EW_MAX_CTAS
 * launches and run the host-side bookkeeping.
 *
 * Scheduling rules (chip-learned, src/omega_numeric.c omega_numeric_check_patch):
 *   - a variable-latency result (LDG, LDS, LDC, LDCU, S2R, SHFL, MUFU) arrives through
 *     a scoreboard; the producer stalls >= 2 and the consumer waits on that SB;
 *   - a fixed-latency result (FADD FMUL FFMA IMAD IADD3 MOV LOP3 SHF) has no
 *     scoreboard: the producer stalls >= 5 before its consumer (we use 6);
 *   - a predicate producer (ISETP) stalls 13 before @P0 use (the codegen default);
 *   - stores (STG, STS) set a read barrier so the next instruction cannot overwrite
 *     the source register early.
 * The emitter below serialises: every instruction waits on everything still
 * pending. Slow and safe; speed is a later cut.
 */
#include "omega_gpu_elementwise_api.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_submit.h"
#include "omega_gpu_session.h"
#include "sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ codegen */

#define CW(stall, wbar, rbar, wait) \
    (((uint32_t)(stall) << 9) | ((uint32_t)(wbar) << 14) | ((uint32_t)(rbar) << 17) | ((uint32_t)(wait) << 20))
#define CW_YIELD 0x2000u
#define SB_NONE 7u
#define SB_GPR 0u  /* variable-latency GPR results */
#define SB_RD 1u   /* store source-register read barrier */
#define SB_UGPR 2u /* uniform-register results (LDCU) */

enum { K_FIXED, K_PRED, K_VAR, K_UVAR, K_STORE, K_SYNC, K_EXIT, K_BRA };

#define EW_MAX_LOOPS 4
typedef struct {
    BlackwellIRProgram *p;
    uint32_t pending; /* scoreboards set and not yet waited on */
    int err;
    int loop_start[EW_MAX_LOOPS], loop_end[EW_MAX_LOOPS];
    int n_loops;
} Em;

static void emit(Em *e, BlackwellIRInsn in, int kind) {
    uint32_t wait = e->pending;
    switch (kind) {
    case K_FIXED: in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_PRED:  in.control = CW(13, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_VAR:   in.control = CW(4, SB_GPR, SB_NONE, wait) | CW_YIELD; e->pending = 1u << SB_GPR; break;
    case K_UVAR:  in.control = CW(4, SB_UGPR, SB_NONE, wait) | CW_YIELD; e->pending = 1u << SB_UGPR; break;
    case K_STORE: in.control = CW(4, SB_NONE, SB_RD, wait) | CW_YIELD; e->pending = 1u << SB_RD; break;
    case K_SYNC:  in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    /* a predicated EXIT or BRA that falls through must not be the only waiter */
    case K_EXIT:  in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    case K_BRA:   in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    default: e->err = -1; return;
    }
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1; /* append returns the index */
}

static int V(Em *e) { int v = omega_bw_ir_alloc_vreg(e->p); if (v < 0) e->err = -1; return v; }
static int V64(Em *e) { int v = omega_bw_ir_alloc_vreg64(e->p); if (v < 0) e->err = -1; return v; }
static int UV64(Em *e) { int v = omega_bw_ir_alloc_uvreg64(e->p); if (v < 0) e->err = -1; return v; }
static int here(const Em *e) { return (int)e->p->count; }

#define NOREG -1
static BlackwellIRInsn I0(BlackwellIROpcode op) {
    BlackwellIRInsn in; memset(&in, 0, sizeof in);
    in.op = op; in.dst_vreg = NOREG; in.src1_vreg = NOREG; in.src2_vreg = NOREG; in.src3_vreg = NOREG; in.ureg = NOREG;
    return in;
}
static void ldcu64(Em *e, int udst, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDCU64); i.dst_vreg = udst; i.imm = off; i.is_uniform = true; emit(e, i, K_UVAR); }
static void ldc64(Em *e, int dst64, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDC64); i.dst_vreg = dst64; i.imm = off; emit(e, i, K_VAR); }
static void ldc32(Em *e, int dst, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDC); i.dst_vreg = dst; i.imm = off; emit(e, i, K_VAR); }
static void s2r(Em *e, int dst, uint32_t sr) { BlackwellIRInsn i = I0(BW_IR_S2R); i.dst_vreg = dst; i.imm = sr; emit(e, i, K_VAR); }
static void movi(Em *e, int dst, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_MOV_IMM); i.dst_vreg = dst; i.imm = imm; emit(e, i, K_FIXED); }
static void movf(Em *e, int dst, float f) { uint32_t u; memcpy(&u, &f, 4); movi(e, dst, u); }
static void movrz(Em *e, int dst) { BlackwellIRInsn i = I0(BW_IR_MOV_RZ); i.dst_vreg = dst; emit(e, i, K_FIXED); }
/* dst = a * imm + c (c may be NOREG for RZ) */
static void imadi(Em *e, int dst, int a, uint32_t imm, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = dst; i.src1_vreg = a; i.imm = imm; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* dst = a * b + c (registers) */
static void imadr(Em *e, int dst, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = dst; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void iadd3(Em *e, int dst, int a, int b) { BlackwellIRInsn i = I0(BW_IR_IADD3); i.dst_vreg = dst; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void lop3xor(Em *e, int dst, int a, int b) { BlackwellIRInsn i = I0(BW_IR_LOP3_XOR); i.dst_vreg = dst; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
/* dst64 = idx * 4 + ptr64 (byte address of element idx) */
static void addr4(Em *e, int dst64, int idx, int ptr64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = dst64; i.src1_vreg = idx; i.imm = 4; i.src3_vreg = ptr64; emit(e, i, K_FIXED); }
static void ldg(Em *e, int dst, int addr64, int udesc) { BlackwellIRInsn i = I0(BW_IR_LDG_E); i.dst_vreg = dst; i.src1_vreg = addr64; i.ureg = udesc; emit(e, i, K_VAR); }
static void stg(Em *e, int addr64, int val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.src2_vreg = val; i.ureg = udesc; emit(e, i, K_STORE); }
static void fadd(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FADD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void fsub(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FSUB); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void fmul(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FMUL); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void ffma(Em *e, int d, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_FFMA); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void mufu(Em *e, BlackwellIROpcode op, int d, int a) { BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = a; emit(e, i, K_VAR); }
static void shfl_down(Em *e, int d, int a, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_SHFL_DOWN); i.dst_vreg = d; i.src1_vreg = a; i.imm = off; emit(e, i, K_VAR); }
static void sts32(Em *e, int addr, int val) { BlackwellIRInsn i = I0(BW_IR_STS32); i.src1_vreg = addr; i.src2_vreg = val; emit(e, i, K_STORE); }
static void lds32(Em *e, int d, int addr) { BlackwellIRInsn i = I0(BW_IR_LDS32); i.dst_vreg = d; i.src1_vreg = addr; emit(e, i, K_VAR); }
static void bar_sync(Em *e) { emit(e, I0(BW_IR_BAR_SYNC), K_SYNC); }
static void isetp_ge_u32(Em *e, int a, int b) { BlackwellIRInsn i = I0(BW_IR_ISETP_GE_U32); i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_PRED); }
static void exit_if_p0(Em *e) { BlackwellIRInsn i = I0(BW_IR_EXIT); i.predicate_p0 = true; emit(e, i, K_EXIT); }
static void bra_back_if_not_p0(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.predicate_not = true; i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
static void tail(Em *e) {
    emit(e, I0(BW_IR_EXIT), K_EXIT);
    BlackwellIRInsn self = I0(BW_IR_BRA); /* the self-branch every kernel here ends with */
    self.control = 0x000fc000u;
    if (omega_bw_ir_append(e->p, &self) < 0) e->err = -1;
}
static void loop_begin(Em *e) { if (e->n_loops < EW_MAX_LOOPS) e->loop_start[e->n_loops] = here(e); }
/* Closing a loop: every value defined before the loop and read inside it stays
 * live until the back-branch, otherwise the linear-scan allocator (which knows
 * nothing about back-edges) would hand its register to a body temporary whose
 * first definition comes after the value's last textual use (e.g. the scale y
 * in rmsnorm pass 2), and the next iteration would read garbage. */
static void loop_end(Em *e) {
    if (e->n_loops >= EW_MAX_LOOPS) { e->err = -1; return; }
    int start = e->loop_start[e->n_loops], end = here(e) - 1;
    e->loop_end[e->n_loops] = end;
    e->n_loops++;
    OmegaRegAlloc *ra = &e->p->regalloc;
    for (int v = 0; v < ra->num_vregs; v++) {
        OmegaLiveInterval *iv = &ra->intervals[v];
        if (iv->first_def >= 0 && iv->first_def < start && iv->last_use >= start && iv->last_use < end) iv->last_use = end;
    }
}

/* Kernel argument words (constant bank 0, same slots as the matmul path):
 * 0x380 a, 0x388 b, 0x390 c (64-bit each), then four 32-bit parameters. */
#define ARG_A 0x380u
#define ARG_B 0x388u
#define ARG_C 0x390u
#define ARG_P0 0x398u
#define ARG_P1 0x39cu
#define ARG_P2 0x3a0u
#define ARG_P3 0x3a4u
#define DESC_OFF 0x358u

typedef struct { int udesc, pa, pb, pc, tid, cta; } Pro;
static Pro prologue(Em *e) {
    Pro r;
    r.udesc = UV64(e); r.pa = V64(e); r.pb = V64(e); r.pc = V64(e); r.tid = V(e); r.cta = V(e);
    ldcu64(e, r.udesc, DESC_OFF);
    ldc64(e, r.pa, ARG_A); ldc64(e, r.pb, ARG_B); ldc64(e, r.pc, ARG_C);
    s2r(e, r.tid, BW_SR_TID_X); s2r(e, r.cta, BW_SR_CTAID_X);
    return r;
}

/* out[i] = 2^a[i] for i < n (n = P0). Threads 128, grid ceil(n/128). Mutant: 2^(a+1). */
static void gen_ex2(Em *e, int mutant) {
    Pro p = prologue(e);
    int n = V(e), i = V(e), off = V64(e), x = V(e), y = V(e);
    ldc32(e, n, ARG_P0);
    imadi(e, i, p.cta, OMEGA_GPU_EW_THREADS, p.tid);
    isetp_ge_u32(e, i, n);
    exit_if_p0(e);
    addr4(e, off, i, p.pa);
    ldg(e, x, off, p.udesc);
    if (mutant) { int one = V(e); movf(e, one, 1.0f); fadd(e, x, x, one); }
    mufu(e, BW_IR_MUFU_EX2, y, x);
    addr4(e, off, i, p.pc);
    stg(e, off, y, p.udesc);
    tail(e);
}

/* out[i] = a[i ^ 127] within each CTA of 128 (n % 128 == 0). STS own word, BAR.SYNC,
 * LDS the partner word. Mutant: LDS own word (no exchange). */
static void gen_xchg(Em *e, int mutant) {
    Pro p = prologue(e);
    int i = V(e), off = V64(e), x = V(e), saddr = V(e), paddr = V(e), mask = V(e), y = V(e);
    imadi(e, i, p.cta, OMEGA_GPU_EW_THREADS, p.tid);
    addr4(e, off, i, p.pa);
    ldg(e, x, off, p.udesc);
    imadi(e, saddr, p.tid, 4, NOREG);
    sts32(e, saddr, x);
    bar_sync(e);
    if (mutant) {
        lds32(e, y, saddr);
    } else {
        movi(e, mask, OMEGA_GPU_EW_THREADS - 1);
        lop3xor(e, paddr, p.tid, mask);
        imadi(e, paddr, paddr, 4, NOREG);
        lds32(e, y, paddr);
    }
    addr4(e, off, i, p.pc);
    stg(e, off, y, p.udesc);
    tail(e);
}

/* One CTA of 128 threads per row. P0 = dim/128 (loop count), P1 = bits(1/dim),
 * P2 = bits(eps), P3 = dim. Mutant: the warp tree drops the last SHFL step. */
static void gen_rmsnorm(Em *e, int mutant) {
    Pro p = prologue(e);
    int nk = V(e), invd = V(e), eps = V(e), dim = V(e);
    int rowbase = V(e), idx = V(e), col = V(e), k = V(e), acc = V(e), cT = V(e), c1 = V(e);
    int off = V64(e), x = V(e), t = V(e);
    ldc32(e, nk, ARG_P0); ldc32(e, invd, ARG_P1); ldc32(e, eps, ARG_P2); ldc32(e, dim, ARG_P3);
    imadr(e, rowbase, p.cta, dim, NOREG);
    iadd3(e, idx, rowbase, p.tid);
    movrz(e, acc); movrz(e, k);
    movi(e, cT, OMEGA_GPU_EW_THREADS); movi(e, c1, 1);
    loop_begin(e);
    int l1 = here(e);
    addr4(e, off, idx, p.pa);
    ldg(e, x, off, p.udesc);
    ffma(e, acc, x, x, acc);
    iadd3(e, idx, idx, cT);
    iadd3(e, k, k, c1);
    isetp_ge_u32(e, k, nk);
    bra_back_if_not_p0(e, l1);
    loop_end(e);
    static const uint32_t steps[5] = { 16, 8, 4, 2, 1 };
    for (int s = 0; s < (mutant ? 4 : 5); s++) { shfl_down(e, t, acc, steps[s]); fadd(e, acc, acc, t); }
    int saddr = V(e);
    imadi(e, saddr, p.tid, 4, NOREG);
    sts32(e, saddr, acc);
    bar_sync(e);
    int a0 = V(e), a1 = V(e), a2 = V(e), a3 = V(e), s0 = V(e), s1 = V(e), s2 = V(e), s3 = V(e), sum = V(e);
    movrz(e, a0); movi(e, a1, 128); movi(e, a2, 256); movi(e, a3, 384);
    lds32(e, s0, a0); lds32(e, s1, a1); lds32(e, s2, a2); lds32(e, s3, a3);
    fadd(e, sum, s0, s1); fadd(e, sum, sum, s2); fadd(e, sum, sum, s3);
    int v = V(e), y = V(e), u = V(e), cmh = V(e), c15 = V(e);
    fmul(e, v, sum, invd);
    fadd(e, v, v, eps);
    mufu(e, BW_IR_MUFU_RSQ, y, v);
    movf(e, cmh, -0.5f); movf(e, c15, 1.5f);
    fmul(e, u, v, y); fmul(e, u, u, y); ffma(e, u, u, cmh, c15); fmul(e, y, y, u);
    int offw = V64(e), offc = V64(e), w = V(e), o = V(e);
    iadd3(e, idx, rowbase, p.tid);
    imadi(e, col, p.tid, 1, NOREG); /* col = tid (the encoder maps a missing src2 to R0, not RZ, so no IADD3 here) */
    movrz(e, k);
    loop_begin(e);
    int l2 = here(e);
    addr4(e, off, idx, p.pa);
    ldg(e, x, off, p.udesc);
    addr4(e, offw, col, p.pb);
    ldg(e, w, offw, p.udesc);
    fmul(e, o, x, y);
    fmul(e, o, o, w);
    addr4(e, offc, idx, p.pc);
    stg(e, offc, o, p.udesc);
    iadd3(e, idx, idx, cT);
    iadd3(e, col, col, cT);
    iadd3(e, k, k, c1);
    isetp_ge_u32(e, k, nk);
    bra_back_if_not_p0(e, l2);
    loop_end(e);
    tail(e);
}

/* One CTA of head_dim/2 threads per head. P0 = head_dim, P1 = head_dim/2.
 * a = vector, b = table (cos in the first half of each head, sin in the second).
 * Mutant: out0 = q0*cos + q1*sin. */
static void gen_rope(Em *e, int mutant) {
    Pro p = prologue(e);
    int hd = V(e), half = V(e), i0 = V(e), i1 = V(e), off0 = V64(e), off1 = V64(e);
    int q0 = V(e), q1 = V(e), c = V(e), s = V(e), t0 = V(e), t1 = V(e), o0 = V(e), o1 = V(e);
    ldc32(e, hd, ARG_P0); ldc32(e, half, ARG_P1);
    imadr(e, i0, p.cta, hd, p.tid);
    iadd3(e, i1, i0, half);
    addr4(e, off0, i0, p.pa); ldg(e, q0, off0, p.udesc);
    addr4(e, off1, i1, p.pa); ldg(e, q1, off1, p.udesc);
    addr4(e, off0, i0, p.pb); ldg(e, c, off0, p.udesc);
    addr4(e, off1, i1, p.pb); ldg(e, s, off1, p.udesc);
    fmul(e, t0, q0, c); fmul(e, t1, q1, s);
    if (mutant) fadd(e, o0, t0, t1); else fsub(e, o0, t0, t1);
    fmul(e, t0, q0, s); fmul(e, t1, q1, c);
    fadd(e, o1, t0, t1);
    addr4(e, off0, i0, p.pc); stg(e, off0, o0, p.udesc);
    addr4(e, off1, i1, p.pc); stg(e, off1, o1, p.udesc);
    tail(e);
}

/* out[i] = g/(1+2^(g*-log2e)) * up for i < n (P0). Mutant: 2^(g*+log2e). */
static void gen_swiglu(Em *e, int mutant) {
    Pro p = prologue(e);
    int n = V(e), i = V(e), off = V64(e), g = V(e), u = V(e), k = V(e), t = V(e), ex = V(e), d = V(e), r = V(e), one = V(e), o = V(e);
    ldc32(e, n, ARG_P0);
    imadi(e, i, p.cta, OMEGA_GPU_EW_THREADS, p.tid);
    isetp_ge_u32(e, i, n);
    exit_if_p0(e);
    addr4(e, off, i, p.pa); ldg(e, g, off, p.udesc);
    addr4(e, off, i, p.pb); ldg(e, u, off, p.udesc);
    movf(e, k, mutant ? 1.4426950408889634f : -1.4426950408889634f);
    movf(e, one, 1.0f);
    fmul(e, t, g, k);
    mufu(e, BW_IR_MUFU_EX2, ex, t);
    fadd(e, d, ex, one);
    mufu(e, BW_IR_MUFU_RCP, r, d);
    fmul(e, o, g, r);
    fmul(e, o, o, u);
    addr4(e, off, i, p.pc);
    stg(e, off, o, p.udesc);
    tail(e);
}

/* A value first defined inside a loop body may not be read after the loop
 * before it is written again: the allocator could have let a later body
 * temporary share its register, and the last iteration's value would be lost. */
static int check_loop_invariant(const Em *e) {
    const OmegaRegAlloc *ra = &e->p->regalloc;
    for (int L = 0; L < e->n_loops; L++)
        for (int v = 0; v < ra->num_vregs; v++) {
            int fd = ra->intervals[v].first_def;
            if (fd < e->loop_start[L] || fd > e->loop_end[L]) continue;
            for (size_t i = (size_t)e->loop_end[L] + 1; i < e->p->count; i++) {
                const BlackwellIRInsn *in = &e->p->insns[i];
                if (in->src1_vreg == v || in->src2_vreg == v || in->src3_vreg == v) return -1;
                if (in->dst_vreg == v && !in->is_uniform) break;
            }
        }
    return 0;
}

#define EW_GPR_BUDGET 64u /* the QMD declares 64 registers, as the matmul launcher does */

static int build_kernel(OmegaGpuEwOp op, int mutant, OmegaBlackwellKernel *kernel) {
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    if (!prog) return OMEGA_GPU_EW_CODEGEN_FAIL;
    omega_bw_ir_init(prog);
    Em e; memset(&e, 0, sizeof e); e.p = prog;
    switch (op) {
    case OMEGA_GPU_EW_EX2: gen_ex2(&e, mutant); break;
    case OMEGA_GPU_EW_XCHG: gen_xchg(&e, mutant); break;
    case OMEGA_GPU_EW_RMSNORM: gen_rmsnorm(&e, mutant); break;
    case OMEGA_GPU_EW_ROPE: gen_rope(&e, mutant); break;
    case OMEGA_GPU_EW_SWIGLU: gen_swiglu(&e, mutant); break;
    default: e.err = -1; break;
    }
    int rc = OMEGA_GPU_EW_CODEGEN_FAIL;
    int ra_rc = (e.err == 0) ? omega_bw_regalloc_solve(prog) : -1;
    int loop_rc = (ra_rc == 0) ? check_loop_invariant(&e) : -1;
#ifdef OMEGA_EW_DEBUG
    fprintf(stderr, "ew codegen op=%d mutant=%d emit_err=%d insns=%zu vregs=%d regalloc=%d loops=%d peak_gpr=%u\n",
            (int)op, mutant, e.err, prog->count, prog->regalloc.num_vregs, ra_rc, loop_rc, prog->regalloc.peak_gpr_usage);
#endif
    if (e.err == 0 && ra_rc == 0 && loop_rc == 0 && prog->regalloc.peak_gpr_usage <= EW_GPR_BUDGET) {
        size_t max_bytes = (size_t)BW_MAX_IR_INSNS * 16;
        memset(kernel, 0, sizeof *kernel);
        kernel->code = malloc(max_bytes);
        size_t emitted = 0;
        if (kernel->code && omega_bw_encode_program(prog, kernel->code, max_bytes, &emitted) == 0) {
            kernel->code_size = emitted;
            kernel->insn_count = emitted / 16;
            kernel->gpr_count = prog->regalloc.peak_gpr_usage;
            kernel->uniform_gpr_count = prog->regalloc.peak_ugpr_usage;
            sha256_hash(kernel->code, kernel->code_size, kernel->code_digest);
            rc = OMEGA_GPU_EW_OK;
        } else {
#ifdef OMEGA_EW_DEBUG
            fprintf(stderr, "ew codegen op=%d: encode failed (an instruction form the encoder refuses)\n", (int)op);
#endif
            free(kernel->code); kernel->code = NULL;
        }
    }
    free(prog);
    return rc;
}

/* --------------------------------------------------------- kernel cache */

/* One kernel per (op, mutant); the device copy is uploaded once and stays
 * resident for the process (cut 4b: no per-call code page). */
typedef struct { int used; int op; int mutant; OmegaBlackwellKernel kernel; NvrmMem code; } Slot;
static int g_mutant_op; /* 0 = none */
static Slot g_cache[8];
static OmegaGpuScratch g_a, g_b, g_c; /* staging, grown to the high-water mark, reused per call */
static bool g_hooked;
#define LOCK() omega_gpu_session_lock()
#define UNLOCK() omega_gpu_session_unlock()

static void cache_clear_locked(void) {
    for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++)
        if (g_cache[i].used) { free(g_cache[i].kernel.code); omega_gpu_session_free(&g_cache[i].code); memset(&g_cache[i], 0, sizeof g_cache[i]); }
}

void omega_gpu_elementwise_cache_clear(void) {
    LOCK();
    cache_clear_locked();
    UNLOCK();
}

/* Called by the session (lock held, device still open) right before it closes. */
static void on_session_close(void) {
    cache_clear_locked();
    omega_gpu_session_scratch_free(&g_a); omega_gpu_session_scratch_free(&g_b); omega_gpu_session_scratch_free(&g_c);
}

void omega_gpu_elementwise_test_set_mutant(int op) {
    LOCK();
    g_mutant_op = op;
    cache_clear_locked();
    UNLOCK();
}

/* Caller holds the lock. Builds the kernel if needed; uploads the device copy when a
 * device is open (upload = false lets the host-only codegen path run without a chip). */
static int kernel_for(OmegaGpuEwOp op, bool upload, Slot **out, bool *hit) {
    int mutant = (g_mutant_op == (int)op);
    Slot *s = NULL;
    for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++)
        if (g_cache[i].used && g_cache[i].op == (int)op && g_cache[i].mutant == mutant) { s = &g_cache[i]; *hit = true; break; }
    if (!s) {
        OmegaBlackwellKernel k;
        int rc = build_kernel(op, mutant, &k);
        if (rc != OMEGA_GPU_EW_OK) return rc;
        size_t slot = 0;
        for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++) if (!g_cache[i].used) { slot = i; break; }
        s = &g_cache[slot];
        if (s->used) { free(s->kernel.code); omega_gpu_session_free(&s->code); } /* full: evict slot 0 */
        memset(s, 0, sizeof *s);
        s->used = 1; s->op = (int)op; s->mutant = mutant; s->kernel = k;
        *hit = false;
    }
    if (upload && !s->code.cpu) {
        if (omega_gpu_session_alloc(s->kernel.code_size, &s->code) != 0) return OMEGA_GPU_EW_CHIP_FAIL;
        memcpy(s->code.cpu, s->kernel.code, s->kernel.code_size);
        __asm__ volatile("dsb sy" ::: "memory");
    }
    *out = s;
    return OMEGA_GPU_EW_OK;
}

int omega_gpu_elementwise_codegen(OmegaGpuEwOp op, const uint32_t *dims, float eps, OmegaBlackwellKernel *kernel) {
    (void)dims; (void)eps; /* kernels read their shape from the argument words */
    if (!kernel) return OMEGA_GPU_EW_BAD_ARGS;
    LOCK();
    int mutant = (g_mutant_op == (int)op);
    UNLOCK();
    return build_kernel(op, mutant, kernel);
}

const char *omega_gpu_elementwise_last_error(void) { return omega_gpu_session_last_error(); }

/* ------------------------------------------------------------ launcher */

#define EW_POISON 0xffbadbadu /* a NaN pattern no kernel here produces on purpose */
#define EW_WAIT_MS 600000ull /* like main's load/store launchers; a 5 s wait failed under contention (omega#198) */

static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

/* Caller holds the lock. Opens the shared device on first use. */
static int dev_open_locked(void) {
    if (!g_hooked) { (void)omega_gpu_session_on_close(on_session_close); g_hooked = true; }
    return omega_gpu_session_open() ? OMEGA_GPU_EW_OK : OMEGA_GPU_EW_CHIP_FAIL;
}

/* One launch: a and b staged in, c out (c_bytes/4 words, poison-filled first), through
 * the shared session (omega_gpu_session_launch: shader-cache invalidate, constant bank,
 * QMD0/QMD1, first marker, L2_FLUSH_DIRTY, second marker), then readback and the
 * poison scan. b may be NULL when b_fill wrote the staging buffer already. */
static int run_abc(const Slot *ks, const void *a, size_t a_bytes, const void *b, size_t b_bytes,
                   void *c, size_t c_bytes, uint32_t threads_x, uint32_t grid_x,
                   const uint32_t params[4], OmegaGpuEwInfo *info) {
    if (omega_gpu_session_scratch(&g_a, a_bytes) || omega_gpu_session_scratch(&g_b, b_bytes) || omega_gpu_session_scratch(&g_c, c_bytes))
        return OMEGA_GPU_EW_CHIP_FAIL;
    memcpy(g_a.mem.cpu, a, a_bytes);
    if (b) memcpy(g_b.mem.cpu, b, b_bytes);
    size_t c_words = c_bytes / 4;
    uint32_t *c_dev = (uint32_t *)g_c.mem.cpu;
    for (size_t i = 0; i < c_words; i++) c_dev[i] = EW_POISON;
    __asm__ volatile("dsb sy" ::: "memory");

    uint32_t args[OMEGA_BW_CBANK_MATMUL_ARGS_WORDS];
    args[0] = (uint32_t)g_a.mem.va; args[1] = (uint32_t)(g_a.mem.va >> 32);
    args[2] = (uint32_t)g_b.mem.va; args[3] = (uint32_t)(g_b.mem.va >> 32);
    args[4] = (uint32_t)g_c.mem.va; args[5] = (uint32_t)(g_c.mem.va >> 32);
    args[6] = params[0]; args[7] = params[1]; args[8] = params[2]; args[9] = params[3];
    OmegaGpuLaunch L = {
        .code_va = ks->code.va, .gpr_count = EW_GPR_BUDGET, .threads_x = threads_x, .threads_y = 1,
        .grid_x = grid_x, .grid_y = 1, .num_elements = c_words,
        .args = args, .n_args = OMEGA_BW_CBANK_MATMUL_ARGS_WORDS, .timeout_ms = EW_WAIT_MS,
    };
    uint64_t elapsed = 0; uint32_t marker = 0;
    if (omega_gpu_session_launch(&L, &elapsed, &marker) != 0) return OMEGA_GPU_EW_CHIP_FAIL;

    memcpy(c, g_c.mem.cpu, c_bytes);
    uint32_t unwritten = 0;
    for (size_t i = 0; i < c_words; i++) if (((uint32_t *)c)[i] == EW_POISON) unwritten++;
    if (info) {
        info->chip_calls++;
        info->elapsed_ns += elapsed;
        info->completion_marker = marker;
        info->unwritten_words += unwritten;
        info->threads_per_cta = threads_x;
        info->ctas_last_launch = grid_x;
    }
    return unwritten ? OMEGA_GPU_EW_UNWRITTEN : OMEGA_GPU_EW_OK;
}

/* ------------------------------------------------------- public entries */

static void info_begin(OmegaGpuEwInfo *info, const OmegaBlackwellKernel *k, bool hit) {
    if (!info) return;
    memset(info, 0, sizeof *info);
    info->kernel_cache_hit = hit;
    info->gpr_count = k->gpr_count;
    info->insn_count = (uint32_t)k->insn_count;
    snprintf(info->target_chip, sizeof info->target_chip, "NVIDIA DGX Spark (Grace Blackwell GB10)");
    info->sm_architecture = 121;
}
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static void info_end(OmegaGpuEwInfo *info, uint64_t t0) { if (info) info->call_ns = now_ns() - t0; }

/* Common prologue of every chip entry: lock, open, kernel (resident), info. */
static int begin(OmegaGpuEwOp op, Slot **ks, OmegaGpuEwInfo *info) {
    LOCK();
    int rc = dev_open_locked();
    bool hit = false;
    if (rc == OMEGA_GPU_EW_OK) rc = kernel_for(op, true, ks, &hit);
    if (rc != OMEGA_GPU_EW_OK) { UNLOCK(); return rc; }
    info_begin(info, &(*ks)->kernel, hit);
    return OMEGA_GPU_EW_OK;
}

int omega_gpu_rmsnorm_f32(uint32_t rows, uint32_t dim, const float *x, const float *weight,
                          float eps, float *out, OmegaGpuEwInfo *info) {
    uint64_t t0 = now_ns();
    if (!x || !weight || !out || rows == 0 || dim == 0 || dim % OMEGA_GPU_EW_THREADS != 0) return OMEGA_GPU_EW_BAD_ARGS;
    if (dim > OMEGA_GPU_EW_MAX_DIM) return OMEGA_GPU_EW_TOO_LARGE;
    Slot *k = NULL;
    int rc = begin(OMEGA_GPU_EW_RMSNORM, &k, info);
    if (rc != OMEGA_GPU_EW_OK) return rc;
    uint32_t params[4] = { dim / OMEGA_GPU_EW_THREADS, f2u(1.0f / (float)dim), f2u(eps), dim };
    for (uint32_t r0 = 0; r0 < rows && rc == OMEGA_GPU_EW_OK; r0 += OMEGA_GPU_EW_MAX_CTAS) {
        uint32_t nr = rows - r0 < OMEGA_GPU_EW_MAX_CTAS ? rows - r0 : OMEGA_GPU_EW_MAX_CTAS;
        size_t bytes = (size_t)nr * dim * 4;
        rc = run_abc(k, x + (size_t)r0 * dim, bytes, weight, (size_t)dim * 4, out + (size_t)r0 * dim, bytes,
                     OMEGA_GPU_EW_THREADS, nr, params, info);
    }
    UNLOCK();
    info_end(info, t0);
    return rc;
}

int omega_gpu_rope_f32(uint32_t heads, uint32_t head_dim, const float *v,
                       const float *cos_half, const float *sin_half, float *out, OmegaGpuEwInfo *info) {
    uint64_t t0 = now_ns();
    if (!v || !cos_half || !sin_half || !out || heads == 0 || head_dim < 2 || (head_dim & 1)) return OMEGA_GPU_EW_BAD_ARGS;
    if (head_dim > 2048) return OMEGA_GPU_EW_TOO_LARGE;
    Slot *k = NULL;
    int rc = begin(OMEGA_GPU_EW_ROPE, &k, info);
    if (rc != OMEGA_GPU_EW_OK) return rc;
    uint32_t half = head_dim / 2;
    uint32_t params[4] = { head_dim, half, 0, 0 };
    for (uint32_t h0 = 0; h0 < heads && rc == OMEGA_GPU_EW_OK; h0 += OMEGA_GPU_EW_MAX_CTAS) {
        uint32_t nh = heads - h0 < OMEGA_GPU_EW_MAX_CTAS ? heads - h0 : OMEGA_GPU_EW_MAX_CTAS;
        size_t bytes = (size_t)nh * head_dim * 4;
        /* table: cos in the first half of each head, sin in the second, written straight into
         * the staging buffer (one copy per head of this launch) */
        if (omega_gpu_session_scratch(&g_b, bytes) != 0) { rc = OMEGA_GPU_EW_CHIP_FAIL; break; }
        float *table = g_b.mem.cpu;
        for (uint32_t h = 0; h < nh; h++) {
            memcpy(table + (size_t)h * head_dim, cos_half, half * sizeof *table);
            memcpy(table + (size_t)h * head_dim + half, sin_half, half * sizeof *table);
        }
        /* out may alias v: v is copied into the staging buffer before the result lands in out */
        rc = run_abc(k, v + (size_t)h0 * head_dim, bytes, NULL, bytes, out + (size_t)h0 * head_dim, bytes, half, nh, params, info);
    }
    UNLOCK();
    info_end(info, t0);
    return rc;
}

static int run_1d(OmegaGpuEwOp op, uint32_t n, const void *a, const void *b, void *c, OmegaGpuEwInfo *info) {
    uint64_t t0 = now_ns();
    Slot *k = NULL;
    int rc = begin(op, &k, info);
    if (rc != OMEGA_GPU_EW_OK) return rc;
    const uint32_t chunk = OMEGA_GPU_EW_MAX_CTAS * OMEGA_GPU_EW_THREADS;
    for (uint32_t i0 = 0; i0 < n && rc == OMEGA_GPU_EW_OK; i0 += chunk) {
        uint32_t ni = n - i0 < chunk ? n - i0 : chunk;
        uint32_t grid = (ni + OMEGA_GPU_EW_THREADS - 1) / OMEGA_GPU_EW_THREADS;
        uint32_t params[4] = { ni, 0, 0, 0 };
        size_t bytes = (size_t)ni * 4;
        rc = run_abc(k, (const uint8_t *)a + (size_t)i0 * 4, bytes, b ? (const uint8_t *)b + (size_t)i0 * 4 : a, bytes,
                     (uint8_t *)c + (size_t)i0 * 4, bytes, OMEGA_GPU_EW_THREADS, grid, params, info);
    }
    UNLOCK();
    info_end(info, t0);
    return rc;
}

int omega_gpu_swiglu_f32(uint32_t n, const float *gate, const float *up, float *out, OmegaGpuEwInfo *info) {
    if (!gate || !up || !out || n == 0) return OMEGA_GPU_EW_BAD_ARGS;
    return run_1d(OMEGA_GPU_EW_SWIGLU, n, gate, up, out, info);
}
int omega_gpu_ex2_f32(uint32_t n, const float *x, float *out, OmegaGpuEwInfo *info) {
    if (!x || !out || n == 0) return OMEGA_GPU_EW_BAD_ARGS;
    return run_1d(OMEGA_GPU_EW_EX2, n, x, NULL, out, info);
}
int omega_gpu_shared_xchg_u32(uint32_t n, const uint32_t *x, uint32_t *out, OmegaGpuEwInfo *info) {
    if (!x || !out || n == 0 || n % OMEGA_GPU_EW_THREADS != 0) return OMEGA_GPU_EW_BAD_ARGS;
    return run_1d(OMEGA_GPU_EW_XCHG, n, x, NULL, out, info);
}

const char *omega_gpu_elementwise_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_EW_OK: return "OK";
    case OMEGA_GPU_EW_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_EW_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_EW_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_EW_CHIP_FAIL: return "CHIP_FAIL";
    case OMEGA_GPU_EW_UNWRITTEN: return "UNWRITTEN";
    default: return "UNKNOWN";
    }
}
