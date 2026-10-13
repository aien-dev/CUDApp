/*
 * omega_blackwell_codegen_router.c: the one-launch MoE router kernel (see the header). Emitter rules follow
 * omega_blackwell_codegen_attn.c / omega_blackwell_codegen_moe.c (control words, read barrier on every load,
 * loop live ranges).
 */
#include "omega_blackwell_codegen_router.h"
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

/* operand with subregister */
typedef struct { int v; unsigned s; } Op;
static Op R(int v) { Op o = { v, 0 }; return o; }
static Op S(int v, unsigned s) { Op o = { v, s }; return o; }

static void ldcu64(Em *e, int ud, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDCU64); i.dst_vreg = ud; i.imm = off; i.is_uniform = true; emit(e, i, K_UVAR); }
static void ldc64(Em *e, int d, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDC64); i.dst_vreg = d; i.imm = off; emit(e, i, K_VAR); }
static void ldc32(Em *e, int d, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDC); i.dst_vreg = d; i.imm = off; emit(e, i, K_VAR); }
static void s2r(Em *e, int d, uint32_t sr) { BlackwellIRInsn i = I0(BW_IR_S2R); i.dst_vreg = d; i.imm = sr; emit(e, i, K_VAR); }
static void movi(Em *e, int d, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_MOV_IMM); i.dst_vreg = d; i.imm = imm; emit(e, i, K_FIXED); }
static void movrz(Em *e, Op d) { BlackwellIRInsn i = I0(BW_IR_MOV_RZ); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; emit(e, i, K_FIXED); }
/* d = a * imm + c */
static void imadi(Em *e, Op d, Op a, uint32_t imm, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* d = a * b + c (registers) */
/* d64 = a * imm + c64 */
static void imadw(Em *e, int d64, int a, uint32_t imm, int c64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = d64; i.src1_vreg = a; i.imm = imm; i.src3_vreg = c64; emit(e, i, K_FIXED); }
static void mov(Em *e, Op d, Op a) { imadi(e, d, a, 1u, NOREG); }
static void shrs(Em *e, int d, Op a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = d; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; emit(e, i, K_FIXED); }
static void shr(Em *e, int d, int a, uint32_t imm) { shrs(e, d, R(a), imm); }
static void andi(Em *e, Op d, Op a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; emit(e, i, K_FIXED); }
static void xorr(Em *e, Op d, Op a, Op b) { BlackwellIRInsn i = I0(BW_IR_LOP3_XOR); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.src2_vreg = b.v; i.src2_subreg = (uint8_t)b.s; emit(e, i, K_FIXED); }
static void ldg_wide(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, int first) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE);
}
static void stg(Em *e, BlackwellIROpcode op, int addr64, Op val, int udesc) { BlackwellIRInsn i = I0(op); i.src1_vreg = addr64; i.src2_vreg = val.v; i.src2_subreg = (uint8_t)val.s; i.ureg = udesc; emit(e, i, K_STORE); }
static void lds128(Em *e, int d128, int addr, uint32_t imm, int first) { BlackwellIRInsn i = I0(BW_IR_LDS128); i.dst_vreg = d128; i.src1_vreg = addr; i.imm = imm; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void lds32(Em *e, Op d, int addr, int first) { BlackwellIRInsn i = I0(BW_IR_LDS32); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = addr; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void sts32(Em *e, int addr, Op val) { BlackwellIRInsn i = I0(BW_IR_STS32); i.src1_vreg = addr; i.src2_vreg = val.v; i.src2_subreg = (uint8_t)val.s; emit(e, i, K_STORE); }
static void bar_sync(Em *e) { emit(e, I0(BW_IR_BAR_SYNC), K_SYNC); }
static void fop(Em *e, BlackwellIROpcode op, Op d, Op a, Op b, Op c) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s;
    i.src2_vreg = b.v; i.src2_subreg = (uint8_t)b.s; i.src3_vreg = c.v; i.src3_subreg = (uint8_t)c.s; emit(e, i, K_FIXED);
}
static void fadd(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FADD, d, a, b, R(NOREG)); }
static void fsub(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FSUB, d, a, b, R(NOREG)); }
static void fmul(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FMUL, d, a, b, R(NOREG)); }
static void ffma(Em *e, Op d, Op a, Op b, Op c) { fop(e, BW_IR_FFMA, d, a, b, c); }
static void fmaxx(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FMNMX_MAX, d, a, b, R(NOREG)); }
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

/* A region only some lanes of a warp execute: BSSY ... @P0 BRA join ... BSYNC. P0 true = skip. */
typedef struct { int bssy, bra; } Region;
static Region region_skip_if_p0(Em *e) { Region r; r.bssy = bssy_fwd(e); r.bra = bra_if_p0_fwd(e); return r; }
static void region_end(Em *e, Region r) { int at = here(e); bsync(e); patch_bra(e, r.bra, at); patch_bra(e, r.bssy, at + 1); }

static void fminn(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FMNMX_MIN, d, a, b, R(NOREG)); }
/* d = LUT(a, b, c) with the LOP3 truth table (a = 0xf0, b = 0xcc, c = 0xaa); vreg -1 encodes RZ */
static void lop3(Em *e, int d, int a, int b, int c, uint32_t lut) { BlackwellIRInsn i = I0(BW_IR_LOP3_LUT); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; i.imm = lut; emit(e, i, K_FIXED); }
/* ne = (a != b) as 0/1: t = a ^ b; ne = (t | -t) >> 31 */
static void ne_of(Em *e, int ne, Op a, Op b, int t, int t2) {
    xorr(e, R(t), a, b);
    imadi(e, R(t2), R(t), 0xffffffffu, NOREG);
    lop3(e, t, t, t2, NOREG, 0xfcu);
    shr(e, ne, t, 31);
}
/* warp reductions: the total lands in lane 0 */
static void warp_sum(Em *e, int v, int t) { for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, v, o); fadd(e, R(v), R(v), R(t)); } }
static void warp_max(Em *e, int v, int t) { for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, v, o); fmaxx(e, R(v), R(v), R(t)); } }
static void warp_min(Em *e, int v, int t) { for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, v, o); fminn(e, R(v), R(v), R(t)); } }
/* lane 0 of every warp stores v to shared [addr] (the warps write identical words when the value is uniform) */
static void lane0_sts(Em *e, int lane, int one, int addr, Op v) {
    isetp_ge_u32(e, lane, one);
    Region r = region_skip_if_p0(e); sts32(e, addr, v); region_end(e, r);
}

static const uint32_t NEG_BIG = 0xF149F2CAu; /* -1e30f */
static const uint32_t C_2P23 = 0x4B000000u;  /* 8388608.0f; also the OR mask that turns a small integer n into the float 2^23 + n */
static const uint32_t C_LOG2E = 0x3FB8AA3Bu; /* 1.4426950f */
static const uint32_t C_INV2048 = 0x3A000000u; /* 1/2048 */

int omega_blackwell_codegen_router_prog(const OmegaRouterSpec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog) return -1;
    const uint32_t W = spec->warps, D = spec->depth;
    const int MUT = spec->mutant;
    if ((W != 8 && W != 16 && W != 32) || (D != 1 && D != 2 && D != 4 && D != 8) || MUT < 0 || MUT > 4) return -1;
    const uint32_t E = 64u / W; /* x values per thread: 8, 4 or 2 */
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;

    /* ---- arguments and identities ---- */
    int udesc = UV64(e);
    int px = V64(e), pgate = V64(e), pnorm = V64(e), pxout = V64(e);
    int eps = V(e), oscale = V(e);
    int tid = V(e), warp = V(e), lane = V(e), one = V(e), zero = V(e), c2p23 = V(e), cneg = V(e), ts = V(e), t1 = V(e), t2 = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, px, OMEGA_ROUTER_ARG_X); ldc64(e, pgate, OMEGA_ROUTER_ARG_GATE); ldc64(e, pnorm, OMEGA_ROUTER_ARG_NORM); ldc64(e, pxout, OMEGA_ROUTER_ARG_XOUT);
    ldc32(e, eps, OMEGA_ROUTER_ARG_EPS); ldc32(e, oscale, OMEGA_ROUTER_ARG_OUT_SCALE);
    s2r(e, tid, BW_SR_TID_X);
    movi(e, one, 1); movrz(e, R(zero)); movi(e, c2p23, C_2P23); movi(e, cneg, NEG_BIG);
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);

    /* ---- phase 1: RMSNorm ---- */
    int toff = V(e), xa = V64(e), wa = V64(e), sq = V(e), rs = V(e), a0 = V(e);
    int xq[2], wq[2];
    Op xe[8], we[8];
    imadi(e, R(toff), R(tid), E * 4u, NOREG);
    imadw(e, xa, toff, 1u, px); imadw(e, wa, toff, 1u, pnorm);
    if (E == 2) {
        xq[0] = V64(e); wq[0] = V64(e);
        ldg_wide(e, BW_IR_LDG_E_64, xq[0], xa, 0, udesc, 1); ldg_wide(e, BW_IR_LDG_E_64, wq[0], wa, 0, udesc, 0);
        for (unsigned i = 0; i < 2; i++) { xe[i] = S(xq[0], i); we[i] = S(wq[0], i); }
    } else {
        for (unsigned q = 0; q < E / 4; q++) { xq[q] = V128(e); wq[q] = V128(e); }
        for (unsigned q = 0; q < E / 4; q++) ldg_wide(e, BW_IR_LDG_E_128, xq[q], xa, q * 16u, udesc, q == 0);
        for (unsigned q = 0; q < E / 4; q++) ldg_wide(e, BW_IR_LDG_E_128, wq[q], wa, q * 16u, udesc, 0);
        for (unsigned i = 0; i < E; i++) { xe[i] = S(xq[i / 4], i % 4); we[i] = S(wq[i / 4], i % 4); }
    }
    if (e->err) return -1;
    fmul(e, R(sq), xe[0], xe[0]);
    for (unsigned i = 1; i < E; i++) ffma(e, R(sq), xe[i], xe[i], R(sq));
    warp_sum(e, sq, ts);
    imadi(e, R(a0), R(warp), 4u, NOREG); imadi(e, R(a0), R(one), OMEGA_ROUTER_SM_RED, a0);
    lane0_sts(e, lane, one, a0, R(sq));
    bar_sync(e);
    /* every thread sums the W partials (quads of 4) */
    {
        int rq0 = V128(e), rq1 = V128(e);
        movi(e, a0, OMEGA_ROUTER_SM_RED);
        for (uint32_t q = 0; q < W / 4; q += 2) {
            lds128(e, rq0, a0, q * 16u, 1);
            if (q + 1 < W / 4) lds128(e, rq1, a0, (q + 1) * 16u, 0);
            for (unsigned i = 0; i < 4; i++) { if (q == 0 && i == 0) mov(e, R(sq), S(rq0, 0)); else fadd(e, R(sq), R(sq), S(rq0, i)); }
            if (q + 1 < W / 4) for (unsigned i = 0; i < 4; i++) fadd(e, R(sq), R(sq), S(rq1, i));
        }
    }
    movi(e, ts, C_INV2048);
    ffma(e, R(rs), R(sq), R(ts), R(eps));
    mufu(e, BW_IR_MUFU_RSQ, R(rs), R(rs));
    /* xn = x * r * w -> shared [toff + 4i]; x' = xn * out_scale -> xout */
    {
        int xo = V64(e);
        imadw(e, xo, toff, 1u, pxout);
        for (unsigned i = 0; i < E; i++) {
            fmul(e, xe[i], xe[i], R(rs));
            if (MUT != 1) fmul(e, xe[i], xe[i], we[i]);
            imadi(e, R(a0), R(one), OMEGA_ROUTER_SM_XN + i * 4u, toff);
            sts32(e, a0, xe[i]);
            fmul(e, R(ts), xe[i], R(oscale));
            if (i) imadw(e, xo, one, 4u, xo);
            stg(e, BW_IR_STG_E, xo, R(ts), udesc);
        }
    }
    bar_sync(e);

    /* ---- phase 2: logits, warp w -> experts w, w + W, ... ---- */
    int c128 = V(e), ex = V(e), lane16 = V(e), lane32 = V(e), gl = V64(e), rowa = V64(e), la = V(e);
    int acc[4], lo = V(e), hi = V(e), xs0 = V128(e), xs1 = V128(e), ld[8];
    for (unsigned i = 0; i < 4; i++) acc[i] = V(e);
    for (uint32_t d = 0; d < D; d++) ld[d] = V128(e);
    if (e->err) return -1;
    movi(e, c128, OMEGA_ROUTER_EXPERTS);
    mov(e, R(ex), R(warp));
    imadi(e, R(lane16), R(lane), 16u, NOREG);
    imadi(e, R(lane32), R(lane), 32u, NOREG);
    imadw(e, gl, lane16, 1u, pgate);
    const int L_e = here(e);
    loop_begin(e);
    isetp_ge_u32(e, ex, c128);
    const int bra_edone = bra_if_p0_fwd(e);
    imadw(e, rowa, ex, OMEGA_ROUTER_ROW_BYTES, gl);
    for (unsigned i = 0; i < 4; i++) movrz(e, R(acc[i]));
    for (uint32_t b = 0; b < OMEGA_ROUTER_STEPS / D; b++) {
        for (uint32_t d = 0; d < D; d++) ldg_wide(e, BW_IR_LDG_E_128, ld[d], rowa, (b * D + d) * 512u, udesc, d == 0);
        for (uint32_t d = 0; d < D; d++) {
            const uint32_t s = b * D + d;
            lds128(e, xs0, lane32, OMEGA_ROUTER_SM_XN + s * 1024u, 1);
            lds128(e, xs1, lane32, OMEGA_ROUTER_SM_XN + s * 1024u + 16u, 0);
            for (unsigned w = 0; w < 4; w++) {
                imadi(e, R(lo), S(ld[d], w), 65536u, NOREG);
                andi(e, R(hi), S(ld[d], w), 0xffff0000u);
                Op x0 = w < 2 ? S(xs0, 2 * w) : S(xs1, 2 * (w - 2)), x1 = w < 2 ? S(xs0, 2 * w + 1) : S(xs1, 2 * (w - 2) + 1);
                ffma(e, R(acc[w]), R(lo), x0, R(acc[w]));
                ffma(e, R(acc[w]), R(hi), x1, R(acc[w]));
            }
        }
    }
    fadd(e, R(acc[0]), R(acc[0]), R(acc[1])); fadd(e, R(acc[2]), R(acc[2]), R(acc[3])); fadd(e, R(acc[0]), R(acc[0]), R(acc[2]));
    warp_sum(e, acc[0], ts);
    imadi(e, R(la), R(ex), 4u, NOREG); imadi(e, R(la), R(one), OMEGA_ROUTER_SM_LOGIT, la);
    lane0_sts(e, lane, one, la, R(acc[0]));
    imadi(e, R(ex), R(one), W, ex);
    bra(e, L_e);
    loop_end(e);
    patch_bra(e, bra_edone, here(e));
    bar_sync(e);

    /* ---- phase 3: top-8 (every warp, uniform barriers) ---- */
    int vq = V128(e), lane4 = V(e), ej[4], fj[4], lm = V(e), mr = V(e), lc = V(e), cj = V(e), ne = V(e), ci = V(e), cm = V(e);
    int bc = V(e), bc4 = V(e), r = V(e), c8 = V(e), sa = V(e);
    for (unsigned j = 0; j < 4; j++) { ej[j] = V(e); fj[j] = V(e); }
    if (e->err) return -1;
    lds128(e, vq, lane16, OMEGA_ROUTER_SM_LOGIT, 1);
    imadi(e, R(lane4), R(lane), 4u, NOREG);
    for (unsigned j = 0; j < 4; j++) {
        imadi(e, R(ej[j]), R(one), j, lane4);
        lop3(e, fj[j], ej[j], c2p23, NOREG, 0xfcu);         /* bits 0x4B000000 | ej = the float 2^23 + ej: monotonic in ej, never zero */
    }
    movi(e, bc, OMEGA_ROUTER_SM_BC); movi(e, bc4, OMEGA_ROUTER_SM_BC + 4u);
    movrz(e, R(r)); movi(e, c8, OMEGA_ROUTER_TOPK);
    const int L_r = here(e);
    loop_begin(e);
    isetp_ge_u32(e, r, c8);
    const int bra_rdone = bra_if_p0_fwd(e);
    /* the round's max */
    fmaxx(e, R(lm), S(vq, 0), S(vq, 1)); fmaxx(e, R(lm), R(lm), S(vq, 2)); fmaxx(e, R(lm), R(lm), S(vq, 3));
    warp_max(e, lm, ts);
    lane0_sts(e, lane, one, bc, R(lm));
    bar_sync(e);
    lds32(e, R(mr), bc, 1);
    /* lowest index holding it: candidate = 2^23 + ej when equal, else lifted by 0x30000000 (about 6e35, still positive) */
    for (unsigned j = 0; j < 4; j++) {
        ne_of(e, ne, S(vq, j), R(mr), t1, t2);
        imadi(e, R(cj), R(ne), 0x30000000u, fj[j]);
        if (j == 0) mov(e, R(lc), R(cj)); else fminn(e, R(lc), R(lc), R(cj));
    }
    warp_min(e, lc, ts);
    imadi(e, R(sa), R(r), 4u, NOREG); imadi(e, R(sa), R(one), OMEGA_ROUTER_SM_SELV, sa);
    isetp_ge_u32(e, lane, one);
    { Region rg = region_skip_if_p0(e); sts32(e, bc4, R(lc)); sts32(e, sa, R(mr)); region_end(e, rg); }
    bar_sync(e);
    lds32(e, R(ci), bc4, 1);
    andi(e, R(ci), R(ci), 0x7fu);                         /* the chosen expert id: the low bits of 2^23 + ej */
    imadi(e, R(sa), R(r), 4u, NOREG); imadi(e, R(sa), R(one), OMEGA_ROUTER_SM_SEL, sa);
    lane0_sts(e, lane, one, sa, R(ci));
    if (MUT == 2) xorr(e, R(cm), R(ci), R(one)); else mov(e, R(cm), R(ci));
    /* mask the winner: v = min(v, ne ? +1e30 : -1e30) */
    for (unsigned j = 0; j < 4; j++) {
        ne_of(e, ne, R(ej[j]), R(cm), t1, t2);
        imadi(e, R(cj), R(ne), 0x80000000u, cneg);
        fminn(e, S(vq, j), S(vq, j), R(cj));
    }
    imadi(e, R(r), R(one), 1u, r);
    bra(e, L_r);
    loop_end(e);
    patch_bra(e, bra_rdone, here(e));
    bar_sync(e);
    /* full 128-way softmax sum (den when norm_topk_prob is off): original logits from shared, l_0 = selv[0] */
    int full = V(e), m0 = V(e), cl2e = V(e);
    if (e->err) return -1;
    movi(e, cl2e, C_LOG2E);
    lds128(e, vq, lane16, OMEGA_ROUTER_SM_LOGIT, 1);
    movi(e, sa, OMEGA_ROUTER_SM_SELV);
    lds32(e, R(m0), sa, 0);
    for (unsigned j = 0; j < 4; j++) {
        fsub(e, R(ts), S(vq, j), R(m0)); fmul(e, R(ts), R(ts), R(cl2e)); mufu(e, BW_IR_MUFU_EX2, R(ts), R(ts));
        if (j == 0) mov(e, R(full), R(ts)); else fadd(e, R(full), R(full), R(ts));
    }
    warp_sum(e, full, ts);
    lane0_sts(e, lane, one, bc, R(full));
    bar_sync(e);
    lds32(e, R(full), bc, 1);

    /* ---- epilogue: warp 0 only ---- */
    isetp_ge_u32(e, warp, one);
    const int bra_tail = bra_if_p0_fwd(e);
    int pids = V64(e), pwts = V64(e), pdgu = V64(e), pddn = V64(e), ptgu = V64(e), ptdn = V64(e), ntk = V(e);
    int sv0 = V128(e), sv1 = V128(e), er[8], s8 = V(e), den = V(e), wa64 = V64(e);
    for (unsigned i = 0; i < 8; i++) er[i] = V(e);
    if (e->err) return -1;
    ldc64(e, pids, OMEGA_ROUTER_ARG_IDS); ldc64(e, pwts, OMEGA_ROUTER_ARG_WTS);
    ldc64(e, pdgu, OMEGA_ROUTER_ARG_DIR_GU); ldc64(e, pddn, OMEGA_ROUTER_ARG_DIR_DN);
    ldc64(e, ptgu, OMEGA_ROUTER_ARG_TAB_GU); ldc64(e, ptdn, OMEGA_ROUTER_ARG_TAB_DN);
    ldc32(e, ntk, OMEGA_ROUTER_ARG_NORM_TOPK);
    movi(e, sa, OMEGA_ROUTER_SM_SELV);
    lds128(e, sv0, sa, 0, 1); lds128(e, sv1, sa, 16, 0);
    for (unsigned i = 0; i < 8; i++) {
        Op v = i < 4 ? S(sv0, i) : S(sv1, i - 4);
        fsub(e, R(er[i]), v, S(sv0, 0)); fmul(e, R(er[i]), R(er[i]), R(cl2e)); mufu(e, BW_IR_MUFU_EX2, R(er[i]), R(er[i]));
    }
    fadd(e, R(s8), R(er[0]), R(er[1]));
    for (unsigned i = 2; i < 8; i++) fadd(e, R(s8), R(s8), R(er[i]));
    /* den = (s8 - full) * float(ntk) + full */
    lop3(e, t1, ntk, c2p23, NOREG, 0xfcu); fsub(e, R(t1), R(t1), R(c2p23));
    fsub(e, R(t2), R(s8), R(full));
    ffma(e, R(den), R(t2), R(t1), R(full));
    if (MUT == 4) movi(e, den, 0x3f800000u); else mufu(e, BW_IR_MUFU_RCP, R(den), R(den));
    for (unsigned i = 0; i < 8; i++) fmul(e, R(er[i]), R(er[i]), R(den));
    /* lane 0: the 8 weights */
    isetp_ge_u32(e, lane, one);
    {
        Region rg = region_skip_if_p0(e);
        for (unsigned i = 0; i < 8; i++) { imadw(e, wa64, one, i * 4u, pwts); stg(e, BW_IR_STG_E, wa64, R(er[i]), udesc); }
        region_end(e, rg);
    }
    /* lanes 0..7: id_g and the two directory entries of id_g into the two group tables */
    isetp_ge_u32(e, lane, c8);
    {
        Region rg = region_skip_if_p0(e);
        int si = V(e), da = V64(e), q0 = V64(e), q1 = V64(e), ta = V64(e), tb = V64(e);
        if (e->err) return -1;
        imadi(e, R(sa), R(lane), 4u, NOREG); imadi(e, R(sa), R(one), OMEGA_ROUTER_SM_SEL, sa);
        lds32(e, R(si), sa, 1);
        imadw(e, da, lane, 4u, pids);
        stg(e, BW_IR_STG_E, da, R(si), udesc);
        if (MUT == 3) xorr(e, R(si), R(si), R(one));
        for (unsigned which = 0; which < 2; which++) {
            imadw(e, da, si, 16u, which == 0 ? pdgu : pddn);
            ldg_wide(e, BW_IR_LDG_E_64, q0, da, 0, udesc, 1);
            ldg_wide(e, BW_IR_LDG_E_64, q1, da, 8, udesc, 0);
            imadw(e, ta, lane, 16u, which == 0 ? ptgu : ptdn);
            stg(e, BW_IR_STG_E, ta, S(q0, 0), udesc);
            imadw(e, tb, one, 4u, ta); stg(e, BW_IR_STG_E, tb, S(q0, 1), udesc);
            imadw(e, tb, one, 8u, ta); stg(e, BW_IR_STG_E, tb, S(q1, 0), udesc);
            imadw(e, tb, one, 12u, ta); stg(e, BW_IR_STG_E, tb, S(q1, 1), udesc);
        }
        region_end(e, rg);
    }
    patch_bra(e, bra_tail, here(e));
    tail(e);
    (void)zero;
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_router_prog_stats(const OmegaRouterSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_router_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}
