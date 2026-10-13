/*
 * omega_blackwell_codegen_dec.c: decode glue kernels (see the header). The emitter helpers are the router's
 * (omega_blackwell_codegen_router.c): control words, read barrier on every load, loop live ranges, lane-0 regions.
 */
#include "omega_blackwell_codegen_dec.h"
#include <string.h>

#define CW(stall, wbar, rbar, wait) \
    (((uint32_t)(stall) << 9) | ((uint32_t)(wbar) << 14) | ((uint32_t)(rbar) << 17) | ((uint32_t)(wait) << 20))
#define CW_YIELD 0x2000u
#define SB_NONE 7u
#define SB_GPR 0u
#define SB_RD 1u
#define SB_UGPR 2u
#define NOREG (-1)
#define MAX_LOOPS 4

enum { K_FIXED, K_PRED, K_VAR, K_VAR_MORE, K_UVAR, K_STORE, K_SYNC, K_EXIT, K_BRA, K_BSSY, K_BSYNC };

typedef struct {
    BlackwellIRProgram *p;
    uint32_t pending;
    int err;
    int open_start[MAX_LOOPS];
    int depth;
} Em;

static void emit(Em *e, BlackwellIRInsn in, int kind) {
    uint32_t wait = e->pending;
    switch (kind) {
    case K_FIXED: in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_PRED:  in.control = CW(14, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_VAR:   in.control = CW(4, SB_GPR, SB_RD, wait) | CW_YIELD; e->pending = (1u << SB_GPR) | (1u << SB_RD); break;
    case K_VAR_MORE: in.control = CW(4, SB_GPR, SB_RD, (e->pending & (1u << SB_RD))) | CW_YIELD; e->pending |= (1u << SB_GPR) | (1u << SB_RD); break;
    case K_UVAR:  in.control = CW(4, SB_UGPR, SB_NONE, wait) | CW_YIELD; e->pending = 1u << SB_UGPR; break;
    case K_STORE: in.control = CW(4, SB_NONE, SB_RD, wait) | CW_YIELD; e->pending = 1u << SB_RD; break;
    case K_SYNC:  in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_EXIT:  in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    case K_BRA:   in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    case K_BSSY:  in.control = CW(1, SB_NONE, SB_NONE, wait) | CW_YIELD; e->pending = 0; break;
    case K_BSYNC: in.control = CW(5, SB_NONE, SB_NONE, wait) | CW_YIELD; e->pending = 0; break;
    default: e->err = -1; return;
    }
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}

static BlackwellIRInsn I0(BlackwellIROpcode op) {
    BlackwellIRInsn i; memset(&i, 0, sizeof i);
    i.op = op; i.dst_vreg = NOREG; i.src1_vreg = NOREG; i.src2_vreg = NOREG; i.src3_vreg = NOREG; i.ureg = NOREG;
    return i;
}
static int V(Em *e) { int v = omega_bw_ir_alloc_vreg(e->p); if (v < 0) e->err = -1; return v; }
static int V64(Em *e) { int v = omega_bw_ir_alloc_vreg64(e->p); if (v < 0) e->err = -1; return v; }
static int V128(Em *e) { int v = omega_bw_ir_alloc_vreg128(e->p); if (v < 0) e->err = -1; return v; }
static int UV64(Em *e) { int v = omega_bw_ir_alloc_uvreg64(e->p); if (v < 0) e->err = -1; return v; }
static int here(const Em *e) { return (int)e->p->count; }

typedef struct { int v; unsigned s; } Op;
static Op R(int v) { Op o = { v, 0 }; return o; }
static Op S(int v, unsigned s) { Op o = { v, s }; return o; }

static void ldcu64(Em *e, int ud, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDCU64); i.dst_vreg = ud; i.imm = off; i.is_uniform = true; emit(e, i, K_UVAR); }
static void ldc64(Em *e, int d, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDC64); i.dst_vreg = d; i.imm = off; emit(e, i, K_VAR); }
static void ldc32(Em *e, int d, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDC); i.dst_vreg = d; i.imm = off; emit(e, i, K_VAR); }
static void s2r(Em *e, int d, uint32_t sr) { BlackwellIRInsn i = I0(BW_IR_S2R); i.dst_vreg = d; i.imm = sr; emit(e, i, K_VAR); }
static void movi(Em *e, int d, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_MOV_IMM); i.dst_vreg = d; i.imm = imm; emit(e, i, K_FIXED); }
static void movrz(Em *e, Op d) { BlackwellIRInsn i = I0(BW_IR_MOV_RZ); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; emit(e, i, K_FIXED); }
static void imadi(Em *e, Op d, Op a, uint32_t imm, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void imadw(Em *e, int d64, int a, uint32_t imm, int c64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = d64; i.src1_vreg = a; i.imm = imm; i.src3_vreg = c64; emit(e, i, K_FIXED); }
static void mov(Em *e, Op d, Op a) { imadi(e, d, a, 1u, NOREG); }
static void shr(Em *e, int d, int a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = d; i.src1_vreg = a; i.imm = imm; emit(e, i, K_FIXED); }
static void andi(Em *e, Op d, Op a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; emit(e, i, K_FIXED); }
static void xorr(Em *e, Op d, Op a, Op b) { BlackwellIRInsn i = I0(BW_IR_LOP3_XOR); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.src2_vreg = b.v; i.src2_subreg = (uint8_t)b.s; emit(e, i, K_FIXED); }
static void lop3(Em *e, int d, int a, int b, int c, uint32_t lut) { BlackwellIRInsn i = I0(BW_IR_LOP3_LUT); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; i.imm = lut; emit(e, i, K_FIXED); }
static void ldg(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, int first) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE);
}
static void stg(Em *e, int addr64, Op val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.src2_vreg = val.v; i.src2_subreg = (uint8_t)val.s; i.ureg = udesc; emit(e, i, K_STORE); }
static void lds128(Em *e, int d128, int addr, uint32_t imm, int first) { BlackwellIRInsn i = I0(BW_IR_LDS128); i.dst_vreg = d128; i.src1_vreg = addr; i.imm = imm; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void sts32(Em *e, int addr, Op val) { BlackwellIRInsn i = I0(BW_IR_STS32); i.src1_vreg = addr; i.src2_vreg = val.v; i.src2_subreg = (uint8_t)val.s; emit(e, i, K_STORE); }
static void bar_sync(Em *e) { emit(e, I0(BW_IR_BAR_SYNC), K_SYNC); }
static void fop(Em *e, BlackwellIROpcode op, Op d, Op a, Op b, Op c) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s;
    i.src2_vreg = b.v; i.src2_subreg = (uint8_t)b.s; i.src3_vreg = c.v; i.src3_subreg = (uint8_t)c.s; emit(e, i, K_FIXED);
}
static void fadd(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FADD, d, a, b, R(NOREG)); }
static void fmul(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FMUL, d, a, b, R(NOREG)); }
static void ffma(Em *e, Op d, Op a, Op b, Op c) { fop(e, BW_IR_FFMA, d, a, b, c); }
static void fmaxx(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FMNMX_MAX, d, a, b, R(NOREG)); }
static void fminn(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FMNMX_MIN, d, a, b, R(NOREG)); }
static void mufu(Em *e, BlackwellIROpcode op, Op d, Op a) { BlackwellIRInsn i = I0(op); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; emit(e, i, K_VAR); }
static void shfl_down(Em *e, int d, int a, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_SHFL_DOWN); i.dst_vreg = d; i.src1_vreg = a; i.imm = off; emit(e, i, K_VAR); }
static void isetp_ge_u32(Em *e, int a, int b) { BlackwellIRInsn i = I0(BW_IR_ISETP_GE_U32); i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_PRED); }
static void bra(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
static int bra_if_p0_fwd(Em *e) { int at = here(e); BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.imm = 1; emit(e, i, K_BRA); return at; }
static void patch_bra(Em *e, int at, int target) { if (!e->err) e->p->insns[at].imm = (uint32_t)(target - at); }
static int bssy_fwd(Em *e) { int at = here(e); BlackwellIRInsn i = I0(BW_IR_BSSY); i.bar_reg = 0; i.imm = 2; emit(e, i, K_BSSY); return at; }
static void bsync(Em *e) { BlackwellIRInsn i = I0(BW_IR_BSYNC); i.bar_reg = 0; emit(e, i, K_BSYNC); }
static void exit_k(Em *e) { emit(e, I0(BW_IR_EXIT), K_EXIT); }
static void tail(Em *e) {
    exit_k(e);
    BlackwellIRInsn self = I0(BW_IR_BRA);
    self.control = 0x000fc000u;
    if (omega_bw_ir_append(e->p, &self) < 0) e->err = -1;
}
static void loop_begin(Em *e) { if (e->depth >= MAX_LOOPS) { e->err = -1; return; } e->open_start[e->depth++] = here(e); }
static void loop_end(Em *e) {
    if (e->depth <= 0) { e->err = -1; return; }
    int start = e->open_start[--e->depth], end = here(e) - 1;
    OmegaRegAlloc *ra = &e->p->regalloc;
    for (int v = 0; v < ra->num_vregs; v++) {
        OmegaLiveInterval *iv = &ra->intervals[v];
        if (iv->first_def >= 0 && iv->first_def < start && iv->last_use >= start && iv->last_use < end) iv->last_use = end;
    }
}
typedef struct { int bssy, bra; } Region;
static Region region_skip_if_p0(Em *e) { Region r; r.bssy = bssy_fwd(e); r.bra = bra_if_p0_fwd(e); return r; }
static void region_end(Em *e, Region r) { int at = here(e); bsync(e); patch_bra(e, r.bra, at); patch_bra(e, r.bssy, at + 1); }
static void ne_of(Em *e, int ne, Op a, Op b, int t, int t2) {
    xorr(e, R(t), a, b);
    imadi(e, R(t2), R(t), 0xffffffffu, NOREG);
    lop3(e, t, t, t2, NOREG, 0xfcu);
    shr(e, ne, t, 31);
}
static void warp_sum(Em *e, int v, int t) { for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, v, o); fadd(e, R(v), R(v), R(t)); } }
static void warp_max(Em *e, int v, int t) { for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, v, o); fmaxx(e, R(v), R(v), R(t)); } }
static void warp_min(Em *e, int v, int t) { for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, v, o); fminn(e, R(v), R(v), R(t)); } }
static void lane0_sts(Em *e, int lane, int one, int addr, Op v) {
    isetp_ge_u32(e, lane, one);
    Region r = region_skip_if_p0(e); sts32(e, addr, v); region_end(e, r);
}
/* the 8 per-warp partials at shared [base]: reduce with op (0 sum, 1 max, 2 min) into d (every thread) */
static void cta_reduce8(Em *e, int d, uint32_t base, int op, int a0) {
    int q0 = V128(e), q1 = V128(e);
    if (e->err) return;
    movi(e, a0, base);
    lds128(e, q0, a0, 0, 1); lds128(e, q1, a0, 16, 0);
    mov(e, R(d), S(q0, 0));
    for (unsigned i = 1; i < 8; i++) {
        Op s = i < 4 ? S(q0, i) : S(q1, i - 4);
        if (op == 0) fadd(e, R(d), R(d), s); else if (op == 1) fmaxx(e, R(d), R(d), s); else fminn(e, R(d), R(d), s);
    }
}

static const uint32_t C_INV2048 = 0x3A000000u; /* 1/2048 */
static const uint32_t C_NLOG2E = 0xBFB8AA3Bu;  /* -1.4426950f */
static const uint32_t C_ONE = 0x3F800000u;
static const uint32_t C_2P23 = 0x4B000000u;

static int gen_axpy(Em *e, const OmegaDecSpec *sp) {
    if (sp->stride == 0 || sp->stride % OMEGA_DEC_THREADS) return -1;
    int udesc = UV64(e), px = V64(e), pa = V64(e), iters = V(e), tid = V(e), cta = V(e), gid = V(e), it = V(e), one = V(e);
    int xa = V64(e), aa = V64(e), xv = V(e), av = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, px, OMEGA_DEC_AXPY_X); ldc64(e, pa, OMEGA_DEC_AXPY_A); ldc32(e, iters, OMEGA_DEC_AXPY_ITERS);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cta, BW_SR_CTAID_X);
    imadi(e, R(gid), R(cta), OMEGA_DEC_THREADS, tid);
    imadw(e, xa, gid, 4u, px); imadw(e, aa, gid, 4u, pa);
    movrz(e, R(it)); movi(e, one, 1);
    const int L = here(e);
    loop_begin(e);
    isetp_ge_u32(e, it, iters);
    const int done = bra_if_p0_fwd(e);
    ldg(e, BW_IR_LDG_E, xv, xa, 0, udesc, 1); ldg(e, BW_IR_LDG_E, av, aa, 0, udesc, 0);
    fadd(e, R(xv), R(xv), R(av));
    stg(e, xa, R(xv), udesc);
    imadw(e, xa, one, sp->stride * 4u, xa); imadw(e, aa, one, sp->stride * 4u, aa);
    imadi(e, R(it), R(one), 1u, it);
    bra(e, L);
    loop_end(e);
    patch_bra(e, done, here(e));
    tail(e);
    return e->err;
}

static int gen_res(Em *e, const OmegaDecSpec *sp) {
    const uint32_t M = sp->mode;
    if (M == 0 || M > 7 || ((M & OMEGA_DEC_RES_ADD) && (M & OMEGA_DEC_RES_COMBINE))) return -1;
    int udesc = UV64(e), px = V64(e), pa = V64(e), pw = V64(e), pnw = V64(e), py = V64(e), eps = V(e), scale = V(e);
    int tid = V(e), warp = V(e), lane = V(e), one = V(e), toff = V(e), t = V(e), t2 = V(e), a0 = V(e);
    int xa = V64(e), xq0 = V128(e), xq1 = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, px, OMEGA_DEC_RES_X); ldc64(e, pa, OMEGA_DEC_RES_A); ldc64(e, pw, OMEGA_DEC_RES_W); ldc64(e, pnw, OMEGA_DEC_RES_NW); ldc64(e, py, OMEGA_DEC_RES_Y);
    ldc32(e, eps, OMEGA_DEC_RES_EPS); ldc32(e, scale, OMEGA_DEC_RES_SCALE);
    s2r(e, tid, BW_SR_TID_X);
    movi(e, one, 1);
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);
    imadi(e, R(toff), R(tid), 32u, NOREG);
    imadw(e, xa, toff, 1u, px);
    ldg(e, BW_IR_LDG_E_128, xq0, xa, 0, udesc, 1); ldg(e, BW_IR_LDG_E_128, xq1, xa, 16, udesc, 0);
    Op x[8];
    for (unsigned i = 0; i < 8; i++) x[i] = i < 4 ? S(xq0, i) : S(xq1, i - 4);
    if (M & OMEGA_DEC_RES_ADD) {
        int aa = V64(e), aq0 = V128(e), aq1 = V128(e);
        if (e->err) return -1;
        imadw(e, aa, toff, 1u, pa);
        ldg(e, BW_IR_LDG_E_128, aq0, aa, 0, udesc, 1); ldg(e, BW_IR_LDG_E_128, aq1, aa, 16, udesc, 0);
        for (unsigned i = 0; i < 8; i++) fadd(e, x[i], x[i], i < 4 ? S(aq0, i) : S(aq1, i - 4));
    }
    if (M & OMEGA_DEC_RES_COMBINE) {
        int wq0 = V128(e), wq1 = V128(e), da = V64(e), dq0 = V128(e), dq1 = V128(e), m[8];
        for (unsigned i = 0; i < 8; i++) m[i] = V(e);
        if (e->err) return -1;
        ldg(e, BW_IR_LDG_E_128, wq0, pw, 0, udesc, 1); ldg(e, BW_IR_LDG_E_128, wq1, pw, 16, udesc, 0);
        imadw(e, da, toff, 1u, pa);
        const unsigned G = sp->mutant == 2 ? 7u : 8u;
        for (unsigned g = 0; g < G; g++) {
            ldg(e, BW_IR_LDG_E_128, dq0, da, g * OMEGA_DEC_HIDDEN * 4u, udesc, 1); ldg(e, BW_IR_LDG_E_128, dq1, da, g * OMEGA_DEC_HIDDEN * 4u + 16u, udesc, 0);
            Op w = g < 4 ? S(wq0, g) : S(wq1, g - 4);
            for (unsigned i = 0; i < 8; i++) {
                Op d = i < 4 ? S(dq0, i) : S(dq1, i - 4);
                if (g == 0) fmul(e, R(m[i]), w, d);
                else { fmul(e, R(t), w, d); fadd(e, R(m[i]), R(m[i]), R(t)); }
            }
        }
        for (unsigned i = 0; i < 8; i++) fadd(e, x[i], x[i], R(m[i]));
    }
    if (M & (OMEGA_DEC_RES_ADD | OMEGA_DEC_RES_COMBINE)) {
        int sa = V64(e);
        if (e->err) return -1;
        for (unsigned i = 0; i < 8; i++) { if (i == 0) imadw(e, sa, toff, 1u, px); else imadw(e, sa, one, 4u, sa); stg(e, sa, x[i], udesc); }
    }
    if (M & OMEGA_DEC_RES_NORM) {
        int sq = V(e), rs = V(e), na = V64(e), nq0 = V128(e), nq1 = V128(e), ya = V64(e), yv = V(e);
        if (e->err) return -1;
        imadw(e, na, toff, 1u, pnw);
        ldg(e, BW_IR_LDG_E_128, nq0, na, 0, udesc, 1); ldg(e, BW_IR_LDG_E_128, nq1, na, 16, udesc, 0);
        fmul(e, R(sq), x[0], x[0]);
        for (unsigned i = 1; i < 8; i++) ffma(e, R(sq), x[i], x[i], R(sq));
        warp_sum(e, sq, t);
        imadi(e, R(a0), R(warp), 4u, NOREG); imadi(e, R(a0), R(one), OMEGA_DEC_SM_RED, a0);
        lane0_sts(e, lane, one, a0, R(sq));
        bar_sync(e);
        cta_reduce8(e, sq, OMEGA_DEC_SM_RED, 0, a0);
        movi(e, t2, C_INV2048);
        ffma(e, R(rs), R(sq), R(t2), R(eps));
        mufu(e, BW_IR_MUFU_RSQ, R(rs), R(rs));
        imadw(e, ya, toff, 1u, py);
        for (unsigned i = 0; i < 8; i++) {
            fmul(e, R(yv), x[i], R(rs));
            if (sp->mutant != 1) fmul(e, R(yv), R(yv), i < 4 ? S(nq0, i) : S(nq1, i - 4));
            fmul(e, R(yv), R(yv), R(scale));
            if (i) imadw(e, ya, one, 4u, ya);
            stg(e, ya, R(yv), udesc);
        }
    }
    tail(e);
    return e->err;
}

static int gen_swiglu(Em *e, const OmegaDecSpec *sp) {
    int udesc = UV64(e), pgu = V64(e), pact = V64(e), scale = V(e), tid = V(e), cx = V(e), cy = V(e), c = V(e), gi = V(e), ai = V(e);
    int ga = V64(e), ua = V64(e), aa = V64(e), gv = V(e), uv = V(e), t = V(e), cst = V(e), ui = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pgu, OMEGA_DEC_SW_GU); ldc64(e, pact, OMEGA_DEC_SW_ACT); ldc32(e, scale, OMEGA_DEC_SW_SCALE);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    imadi(e, R(c), R(cx), OMEGA_DEC_THREADS, tid);
    imadi(e, R(gi), R(cy), 2u * OMEGA_DEC_INTER, c);
    imadi(e, R(ai), R(cy), OMEGA_DEC_INTER, c);
    /* the up half sits INTER floats after the gate: its own address (the 32-bit LDG.E encoding carries no immediate
     * offset, omega_blackwell_codegen.c BW_IR_LDG_E; the warp simulator caught the dropped offset, 2026-10-13) */
    movi(e, cst, sp->mutant == 1 ? OMEGA_DEC_INTER - 1u : OMEGA_DEC_INTER);
    imadi(e, R(ui), R(cst), 1u, gi);
    imadw(e, ga, gi, 4u, pgu);
    imadw(e, ua, ui, 4u, pgu);
    imadw(e, aa, ai, 4u, pact);
    ldg(e, BW_IR_LDG_E, gv, ga, 0, udesc, 1);
    ldg(e, BW_IR_LDG_E, uv, ua, 0, udesc, 0);
    movi(e, cst, C_NLOG2E);
    fmul(e, R(t), R(gv), R(cst));
    mufu(e, BW_IR_MUFU_EX2, R(t), R(t));
    movi(e, cst, C_ONE);
    fadd(e, R(t), R(t), R(cst));
    mufu(e, BW_IR_MUFU_RCP, R(t), R(t));
    fmul(e, R(t), R(gv), R(t));
    fmul(e, R(t), R(t), R(uv));
    fmul(e, R(t), R(t), R(scale));
    stg(e, aa, R(t), udesc);
    tail(e);
    return e->err;
}

static int gen_argmax(Em *e, const OmegaDecSpec *sp) {
    const int indexed = sp->mode == 1;
    int udesc = UV64(e), pv = V64(e), pi = V64(e), pov = V64(e), poi = V64(e);
    int tid = V(e), cta = V(e), warp = V(e), lane = V(e), one = V(e), gid = V(e), off = V(e), t = V(e), t2 = V(e), a0 = V(e);
    int va = V64(e), vq0 = V128(e), vq1 = V128(e), iq0 = 0, iq1 = 0, m = V(e), mx = V(e), ne = V(e), cand = V(e), best = V(e), c2p23 = V(e), idx = V(e);
    if (indexed) { iq0 = V128(e); iq1 = V128(e); }
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pv, OMEGA_DEC_AM_V); ldc64(e, pi, OMEGA_DEC_AM_I); ldc64(e, pov, OMEGA_DEC_AM_OV); ldc64(e, poi, OMEGA_DEC_AM_OI);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cta, BW_SR_CTAID_X);
    movi(e, one, 1); movi(e, c2p23, C_2P23);
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);
    imadi(e, R(gid), R(cta), OMEGA_DEC_THREADS, tid);
    imadi(e, R(off), R(gid), 32u, NOREG);
    imadw(e, va, off, 1u, pv);
    ldg(e, BW_IR_LDG_E_128, vq0, va, 0, udesc, 1); ldg(e, BW_IR_LDG_E_128, vq1, va, 16, udesc, 0);
    if (indexed) {
        imadw(e, va, off, 1u, pi);
        ldg(e, BW_IR_LDG_E_128, iq0, va, 0, udesc, 0); ldg(e, BW_IR_LDG_E_128, iq1, va, 16, udesc, 0);
    }
    Op v[8];
    for (unsigned j = 0; j < 8; j++) v[j] = j < 4 ? S(vq0, j) : S(vq1, j - 4);
    const unsigned J = sp->mutant == 1 ? 7u : 8u;
    mov(e, R(m), v[0]);
    for (unsigned j = 1; j < J; j++) fmaxx(e, R(m), R(m), v[j]);
    warp_max(e, m, t);
    imadi(e, R(a0), R(warp), 4u, NOREG); imadi(e, R(a0), R(one), OMEGA_DEC_SM_RED, a0);
    lane0_sts(e, lane, one, a0, R(m));
    bar_sync(e);
    cta_reduce8(e, mx, OMEGA_DEC_SM_RED, 1, a0);
    /* lowest index holding mx: candidate float bits 0x4B000000 | idx, lifted by 0x30000000 where v != mx */
    for (unsigned j = 0; j < J; j++) {
        if (indexed) mov(e, R(idx), j < 4 ? S(iq0, j) : S(iq1, j - 4));
        else imadi(e, R(idx), R(gid), 8u, NOREG), imadi(e, R(idx), R(one), j, idx);
        lop3(e, idx, idx, c2p23, NOREG, 0xfcu);
        ne_of(e, ne, v[j], R(mx), t, t2);
        imadi(e, R(cand), R(ne), 0x30000000u, idx);
        if (j == 0) mov(e, R(best), R(cand)); else fminn(e, R(best), R(best), R(cand));
    }
    warp_min(e, best, t);
    imadi(e, R(a0), R(warp), 4u, NOREG); imadi(e, R(a0), R(one), OMEGA_DEC_SM_RED2, a0);
    lane0_sts(e, lane, one, a0, R(best));
    bar_sync(e);
    cta_reduce8(e, best, OMEGA_DEC_SM_RED2, 2, a0);
    andi(e, R(best), R(best), 0x7fffffu);
    isetp_ge_u32(e, tid, one);
    {
        Region r = region_skip_if_p0(e);
        int oa = V64(e);
        if (e->err) return -1;
        imadw(e, oa, cta, 4u, pov); stg(e, oa, R(mx), udesc);
        imadw(e, oa, cta, 4u, poi); stg(e, oa, R(best), udesc);
        region_end(e, r);
    }
    tail(e);
    return e->err;
}

int omega_blackwell_codegen_dec_prog(const OmegaDecSpec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    int rc;
    switch (spec->kind) {
    case OMEGA_DEC_AXPY: rc = gen_axpy(&em, spec); break;
    case OMEGA_DEC_RES: rc = gen_res(&em, spec); break;
    case OMEGA_DEC_SWIGLU: rc = gen_swiglu(&em, spec); break;
    case OMEGA_DEC_ARGMAX: rc = gen_argmax(&em, spec); break;
    default: return -1;
    }
    if (rc != 0 || em.err || em.depth != 0) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_dec_prog_stats(const OmegaDecSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_dec_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}
