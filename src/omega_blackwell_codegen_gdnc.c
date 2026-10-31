/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * omega_blackwell_codegen_gdnc.c (DRA-84): the chunked tensor-core GDN prefill scan (see the header). The emitter
 * helpers are those of omega_blackwell_codegen_gdn.c (copied, not shared: PR #222 edits that file), plus HMMA, NOP,
 * LOP3.LUT, IADD3 and the register-register IMAD. HMMA scheduling follows omega_blackwell_codegen_tc.c (nvcc 13.0
 * oracle): independent HMMAs stall 8 when four or more chains interleave, stall 15 with two chains, and a non-HMMA
 * reader of an HMMA result waits stall 15 + NOP 14 (29 cycles).
 *
 * Control flow: warp-uniform forward branches (no BSSY) for the per-warp phases, single-level BSSY regions for the
 * lane-divergent stores; SHFL and HMMA only run with the whole warp converged.
 */
#include "omega_blackwell_codegen_gdnc.h"
#include "omega_blackwell_codegen_gdn.h"
#include "omega_blackwell_codegen_tc.h"
#include <string.h>

#define CW(stall, wbar, rbar, wait) \
    (((uint32_t)(stall) << 9) | ((uint32_t)(wbar) << 14) | ((uint32_t)(rbar) << 17) | ((uint32_t)(wait) << 20))
#define CW_YIELD 0x2000u
#define SB_NONE 7u
#define SB_GPR 0u
#define SB_RD 1u
#define SB_UGPR 2u
#define SB_GATE 4u    /* held: the z row of the sub-chunk */
#define NOREG (-1)

enum { K_FIXED, K_PRED, K_VAR, K_VAR_MORE, K_UVAR, K_STORE, K_SYNC, K_EXIT, K_BRA, K_BSSY, K_BSYNC };

#define MAX_LOOPS 4

typedef struct {
    BlackwellIRProgram *p;
    uint32_t pending, held;
    int err;
    int depth, open_start[MAX_LOOPS];
    uint32_t placeholders; /* SHFL.BFLY and F2FP words omega_tc_patch_code must patch */
    int one;               /* the register holding 1 (addi) */
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
/* d = a * imm + c */
static void imadi(Em *e, Op d, Op a, uint32_t imm, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* d = a * b + c (registers) */
static void imadr(Em *e, int d, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* d = a + imm */
static void addi(Em *e, int d, int a, uint32_t imm) { imadi(e, R(d), R(e->one), imm, a); }
/* d64 = a * imm + c64 */
static void imadw(Em *e, int d64, Op a, uint32_t imm, int c64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = d64; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; i.src3_vreg = c64; emit(e, i, K_FIXED); }
static void shr(Em *e, int d, int a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = d; i.src1_vreg = a; i.imm = imm; emit(e, i, K_FIXED); }
static void andi(Em *e, Op d, Op a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; emit(e, i, K_FIXED); }
static void lop3(Em *e, int d, int a, int b, int c, uint32_t lut) { BlackwellIRInsn i = I0(BW_IR_LOP3_LUT); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; i.imm = lut; emit(e, i, K_FIXED); }
static void iadd(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_IADD3); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void i2fp(Em *e, int d, int a) { BlackwellIRInsn i = I0(BW_IR_I2FP); i.dst_vreg = d; i.src1_vreg = a; emit(e, i, K_FIXED); }
static void ldg_wide(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, int first) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE);
}
static void ldg32(Em *e, Op d, int addr64, int udesc, int first) { BlackwellIRInsn i = I0(BW_IR_LDG_E); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = addr64; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE); }
/* A prefetch load: scoreboard `sb` is held, not waited for, until consume() */
static void ldg_pf(Em *e, int d, int addr64, uint32_t imm, int udesc, unsigned sb) {
    BlackwellIRInsn in = I0(BW_IR_LDG_E_128); in.dst_vreg = d; in.src1_vreg = addr64; in.imm = imm; in.ureg = udesc;
    in.control = CW(4, sb, SB_RD, (e->pending & (1u << SB_RD))) | CW_YIELD;
    e->pending |= 1u << SB_RD; e->held |= 1u << sb;
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}
static void consume(Em *e, unsigned sb) { e->pending |= e->held & (1u << sb); e->held &= ~(1u << sb); }
static void stg(Em *e, int addr64, uint32_t imm, Op val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.src2_vreg = val.v; i.src2_subreg = (uint8_t)val.s; i.imm = imm; i.ureg = udesc; emit(e, i, K_STORE); }
/* F2FP.BF16.F32.PACK_AB d = bf16_rn(hi) << 16 | bf16_rn(lo): a placeholder omega_tc_patch_code patches */
static void f2fp(Em *e, Op d, Op hi, Op lo) {
    BlackwellIRInsn i = I0(BW_IR_FMNMX_MAX); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = hi.v; i.src1_subreg = (uint8_t)hi.s;
    i.src2_vreg = lo.v; i.src2_subreg = (uint8_t)lo.s; i.imm = OMEGA_TC_MARK_F2FP; emit(e, i, K_FIXED); e->placeholders++;
}
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
static void fminn(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FMNMX_MIN, d, a, b, R(NOREG)); }
static void mufu(Em *e, BlackwellIROpcode op, Op d, Op a) { BlackwellIRInsn i = I0(op); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; emit(e, i, K_VAR); }
static void shfl_down(Em *e, int d, int a, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_SHFL_DOWN); i.dst_vreg = d; i.src1_vreg = a; i.imm = off; emit(e, i, K_VAR); }
static void isetp_ge_u32(Em *e, int a, int b) { BlackwellIRInsn i = I0(BW_IR_ISETP_GE_U32); i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_PRED); }
static void bra_if_p0(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
static int bra_if_p0_fwd(Em *e) { int at = here(e); BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.imm = 1; emit(e, i, K_BRA); return at; }
static void patch_bra(Em *e, int at, int target) { if (!e->err) e->p->insns[at].imm = (uint32_t)(target - at); }
static int bssy_fwd(Em *e) { int at = here(e); BlackwellIRInsn i = I0(BW_IR_BSSY); i.bar_reg = 0; i.imm = 2; emit(e, i, K_BSSY); return at; }
static void bsync(Em *e) { BlackwellIRInsn i = I0(BW_IR_BSYNC); i.bar_reg = 0; emit(e, i, K_BSYNC); }
static void tail(Em *e) {
    emit(e, I0(BW_IR_EXIT), K_EXIT);
    BlackwellIRInsn self = I0(BW_IR_BRA);
    self.control = 0x000fc000u;
    if (omega_bw_ir_append(e->p, &self) < 0) e->err = -1;
}
/* HMMA.16816.F32.BF16 d = a * b + c: a and c whole V128s, b two words from subregister b.s; c NOREG = zero */
static void hmma(Em *e, int d, int a, Op b, int c, uint32_t stall) {
    BlackwellIRInsn i = I0(BW_IR_HMMA_BF16); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b.v; i.src2_subreg = (uint8_t)b.s; i.src3_vreg = c;
    i.control = CW(stall, SB_NONE, SB_NONE, e->pending); e->pending = 0;
    if (omega_bw_ir_append(e->p, &i) < 0) e->err = -1;
}
static void nop(Em *e, uint32_t stall) {
    BlackwellIRInsn i = I0(BW_IR_NOP); i.control = CW(stall, SB_NONE, SB_NONE, e->pending); e->pending = 0;
    if (omega_bw_ir_append(e->p, &i) < 0) e->err = -1;
}

/* A region only some lanes of a warp execute: BSSY ... @P0 BRA join ... BSYNC. P0 true = skip. */
typedef struct { int bssy, bra; } Region;
static Region region_skip_if_p0(Em *e) { Region r; r.bssy = bssy_fwd(e); r.bra = bra_if_p0_fwd(e); return r; }
static void region_end(Em *e, Region r) { int at = here(e); bsync(e); patch_bra(e, r.bra, at); patch_bra(e, r.bssy, at + 1); }
/* Loops: a value defined before the loop and read inside it stays live until the back-branch (the linear-scan
 * allocator knows no back edges; omega_blackwell_codegen_pattn.c). */
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
static void mov(Em *e, Op d, Op a) { imadi(e, d, a, 1u, NOREG); }
static void shfl_bfly(Em *e, int d, int a, uint32_t lanes) { shfl_down(e, d, a, OMEGA_TC_MARK_BFLY | (lanes & 31u)); e->placeholders++; }
static void bfly_sum(Em *e, int v, int t, uint32_t lo, uint32_t hi) { for (uint32_t o = lo; o <= hi; o *= 2) { shfl_bfly(e, t, v, o); fadd(e, R(v), R(v), R(t)); } }
/* x = x * sigmoid(x) = x * RCP(1 + EX2(-x * log2 e)); t scratch */
static void silu(Em *e, Op x, int t, int nlog2e, int fone) {
    fmul(e, R(t), x, R(nlog2e)); mufu(e, BW_IR_MUFU_EX2, R(t), R(t)); fadd(e, R(t), R(t), R(fone));
    mufu(e, BW_IR_MUFU_RCP, R(t), R(t)); fmul(e, x, x, R(t));
}
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
/* d = min(t, tm1) for t, tm1 < 2^31 without a select: t - ((t - tm1) & (t >= tm1 ? ~0 : 0)) */
static void clamp_u(Em *e, int d, int t, int tm1) {
    int dd = V(e), ng = V(e);
    if (e->err) return;
    imadi(e, R(dd), R(tm1), 0xffffffffu, t);   /* t - tm1 */
    shr(e, ng, dd, 31);                        /* 1 when t < tm1 */
    addi(e, ng, ng, 0xffffffffu);              /* 0 when t < tm1, else ~0 */
    lop3(e, dd, dd, ng, NOREG, 0xc0u);         /* a & b */
    imadi(e, R(d), R(dd), 0xffffffffu, t);
}

/* Warp 7, lanes 0..15: ROFF[l] = min(tb + l, T - 1) * XSTRIDE (the qkvz row offsets of the next sub-chunk). */
static void roff_write(Em *e, int warp, int lane, int c6, int c16, int tb, int tm1, int xs) {
    isetp_ge_u32(e, c6, warp);
    int sk = bra_if_p0_fwd(e);
    {
        int t = V(e), tc = V(e), a = V(e);
        if (e->err) return;
        iadd(e, t, tb, lane);
        clamp_u(e, tc, t, tm1);
        imadr(e, tc, tc, xs, NOREG);
        imadi(e, R(a), R(lane), 4u, NOREG); addi(e, a, a, OMEGA_GDNC_SM_ROFF);
        isetp_ge_u32(e, lane, c16);
        Region r = region_skip_if_p0(e); sts32(e, a, R(tc)); region_end(e, r);
    }
    patch_bra(e, sk, here(e));
}

int omega_blackwell_codegen_gdnc_prog(const OmegaGdncSpec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog || (spec->mutant != 0 && spec->mutant != OMEGA_GDNC_MUTANT_DROP_DECAY)) return -1;
    OmegaGdnSpec gs; memset(&gs, 0, sizeof gs); gs.ratio = spec->ratio;
    OmegaGdnShape G;
    if (omega_gdn_shape(&gs, &G)) return -1;
    const uint32_t RR = G.r, VH = G.vh, CONV = G.conv;
    const int MUT = spec->mutant;
    const uint32_t C_LOG2E = fbits(1.44269504f), C_NLOG2E = fbits(-1.44269504f), C_LN2 = fbits(0.693147181f);
    const uint32_t C_ONE = fbits(1.0f), C_M1 = fbits(-1.0f), C_INV128 = fbits(1.0f / 128.0f);
    const float qs = 0.0883883476f; /* 1 / sqrt(128), as the multi kernel */
    const uint32_t SC = OMEGA_GDNC_SM_SC;
    omega_bw_ir_init(prog);
    if (omega_bw_ir_set_gpr_limit(prog, BW_PHYS_GPR_HW_LIMIT) != 0) return -1;
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;

    /* ---- identities, constants, pointers ---- */
    int udesc = UV64(e);
    int tid = V(e), cx = V(e), cy = V(e), h = V(e), warp = V(e), lane = V(e), one = V(e), g = V(e), tp = V(e), zr = V(e);
    int T = V(e), tm1 = V(e), t0 = V(e), xs = V(e), abs_ = V(e), os16 = V(e);
    int insc = V(e), osc = V(e), el2 = V(e), en = V(e), fone = V(e), cnl2e = V(e), cl2e = V(e);
    int c2 = V(e), c6 = V(e), c16 = V(e);
    int pq = V64(e), pab = V64(e), ppar = V64(e), ptail = V64(e), pst = V64(e), pout = V64(e), psnap = V64(e);
    int sq[16];
    for (unsigned n = 0; n < 16; n++) sq[n] = V128(e);
    if (e->err) return -1;
    e->one = one;
    ldcu64(e, udesc, 0x358);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    ldc32(e, T, OMEGA_GDNC_ARG_T); ldc32(e, xs, OMEGA_GDNC_ARG_XSTRIDE); ldc32(e, abs_, OMEGA_GDNC_ARG_ABSTRIDE);
    ldc32(e, os16, OMEGA_GDNC_ARG_OSTRIDE);
    ldc32(e, insc, OMEGA_GDNC_ARG_IN_SCALE); ldc32(e, osc, OMEGA_GDNC_ARG_OUT_SCALE);
    ldc32(e, el2, OMEGA_GDNC_ARG_EPS_L2); ldc32(e, en, OMEGA_GDNC_ARG_EPS_NORM);
    ldc64(e, pq, OMEGA_GDNC_ARG_QKVZ); ldc64(e, pab, OMEGA_GDNC_ARG_AB); ldc64(e, ppar, OMEGA_GDNC_ARG_PAR);
    ldc64(e, ptail, OMEGA_GDNC_ARG_TAIL); ldc64(e, pst, OMEGA_GDNC_ARG_STATE); ldc64(e, pout, OMEGA_GDNC_ARG_OUT);
    ldc64(e, psnap, OMEGA_GDNC_ARG_SNAP);
    movi(e, one, 1); movrz(e, R(zr));
    movi(e, fone, C_ONE); movi(e, cnl2e, C_NLOG2E); movi(e, cl2e, C_LOG2E);
    movi(e, c2, 2); movi(e, c6, 6); movi(e, c16, 16);
    imadi(e, R(h), R(cy), RR, cx);
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);
    shr(e, g, lane, 2);
    andi(e, R(tp), R(lane), 3u);
    addi(e, tm1, T, 0xffffffffu);

    /* ---- the state: S[k][v] f32, warp w holds v = 16w..16w+15 as the C fragments of S^T (v x k), k-tile n ---- */
    int loff = V(e), sa = V64(e);
    if (e->err) return -1;
    imadi(e, R(loff), R(h), OMEGA_GDN_STATE_HEAD_BYTES, NOREG);
    imadi(e, R(loff), R(tp), 1024u, loff);
    imadi(e, R(loff), R(warp), 64u, loff);
    imadi(e, R(loff), R(g), 4u, loff);
    imadw(e, sa, R(loff), 1u, pst);
    for (unsigned b = 0; b < 4; b++) {
        int ad[16];
        for (unsigned i = 0; i < 16; i++) ad[i] = V64(e);
        if (e->err) return -1;
        for (unsigned i = 0; i < 16; i++) {
            unsigned idx = b * 16u + i, n = idx >> 2, q = idx & 3u;
            imadw(e, ad[i], R(one), n * 4096u + (q & 1u) * 512u + (q >> 1) * 32u, sa);
        }
        for (unsigned i = 0; i < 16; i++) { unsigned idx = b * 16u + i; ldg32(e, S(sq[idx >> 2], idx & 3u), ad[i], udesc, i == 0); }
    }

    /* ---- conv threads: pp = tid (tid < 192) else tid - 64 (warps 6, 7 mirror 4, 5 and never run the conv) ---- */
    int xb = V64(e), ta = V64(e), wq0 = V128(e), wq1 = V128(e), tl0 = V128(e), tl1 = V128(e);
    int cs = V(e), tstr = V(e), cstr = V(e), tso = V(e);
    if (e->err) return -1;
    {
        int bit = V(e), pp = V(e), grp = V(e), cc = V(e), gb = V(e), dh = V(e), chan = V(e), tmp = V(e), slot = V(e), aw = V64(e);
        int csA = V(e), csV = V(e);
        if (e->err) return -1;
        addi(e, tmp, tid, 64u); shr(e, bit, tmp, 8);
        imadi(e, R(pp), R(bit), (uint32_t)-64, tid);
        shr(e, grp, pp, 6);
        andi(e, R(cc), R(pp), 63u); imadi(e, R(cc), R(cc), 2u, NOREG);
        shr(e, gb, grp, 1);
        imadi(e, R(dh), R(cy), 0xffffffffu, h);                  /* h - hk */
        /* q: hk*128 + cc; k: 2048 + hk*128 + cc; v: 4096 + h*128 + cc */
        imadi(e, R(chan), R(grp), 2048u, cc);
        imadi(e, R(chan), R(cy), 128u, chan);
        imadr(e, tmp, gb, dh, NOREG);
        imadi(e, R(chan), R(tmp), 128u, chan);
        imadw(e, xb, R(chan), 2u, pq);
        imadw(e, aw, R(chan), 16u, ppar);
        imadi(e, R(slot), R(grp), 128u, cc);
        imadw(e, ta, R(h), OMEGA_GDN_TAIL_HEAD_BYTES, ptail);
        imadw(e, ta, R(slot), 16u, ta);
        imadi(e, R(tso), R(h), OMEGA_GDN_TAIL_HEAD_BYTES, NOREG);
        imadi(e, R(tso), R(slot), 16u, tso);
        addi(e, tso, tso, VH * OMEGA_GDN_STATE_HEAD_BYTES);
        /* shared destination of token 0: q, k as [t][c] (t stride 512, c stride 4), v as [c][t] (t 4, c 64) */
        imadi(e, R(csA), R(grp), 8192u, NOREG); imadi(e, R(csA), R(cc), 4u, csA);
        imadi(e, R(csV), R(cc), 64u, NOREG); addi(e, csV, csV, OMEGA_GDNC_SM_XV);
        imadi(e, R(tmp), R(csA), 0xffffffffu, csV);
        imadr(e, cs, gb, tmp, csA);
        movi(e, tmp, 512u); imadi(e, R(tstr), R(gb), (uint32_t)-508, tmp);
        movi(e, tmp, 4u); imadi(e, R(cstr), R(gb), 60u, tmp);
        ldg_wide(e, BW_IR_LDG_E_128, wq0, aw, 0, udesc, 1);
        ldg_wide(e, BW_IR_LDG_E_128, wq1, aw, 16, udesc, 0);
        ldg_wide(e, BW_IR_LDG_E_128, tl0, ta, 0, udesc, 0);
        ldg_wide(e, BW_IR_LDG_E_128, tl1, ta, 16, udesc, 0);
    }

    /* ---- per-head scalars, norm thread bases, solve bases, output ---- */
    int scq = V64(e), nw0 = V128(e), nw1 = V128(e);
    int t16 = V(e), l16 = V(e), nrow = V(e), agam = V(e), aroff = V(e), nkht = V(e), nbp = V(e), nap = V(e), zb = V64(e), ao = V64(e);
    int srb = V(e), scb = V(e);
    if (e->err) return -1;
    {
        int asc = V64(e), an = V64(e), tmp = V(e), tmp2 = V(e);
        if (e->err) return -1;
        imadw(e, asc, R(h), 8u, ppar);
        shr(e, t16, tid, 4); andi(e, R(l16), R(tid), 15u);
        imadw(e, an, R(l16), 32u, ppar);
        ldg_wide(e, BW_IR_LDG_E_64, scq, asc, G.par_scal, udesc, 1);
        ldg_wide(e, BW_IR_LDG_E_128, nw0, an, G.par_norm, udesc, 0);
        ldg_wide(e, BW_IR_LDG_E_128, nw1, an, G.par_norm + 16u, udesc, 0);
        imadi(e, R(nrow), R(t16), 512u, NOREG); imadi(e, R(nrow), R(l16), 32u, nrow);
        imadi(e, R(agam), R(t16), 4u, NOREG); addi(e, agam, agam, SC + OMEGA_GDNC_SC_GAM);
        imadi(e, R(aroff), R(t16), 4u, NOREG); addi(e, aroff, aroff, OMEGA_GDNC_SM_ROFF);
        imadi(e, R(nkht), R(l16), 512u, NOREG); imadi(e, R(nkht), R(t16), 4u, nkht); addi(e, nkht, nkht, OMEGA_GDNC_SM_XK);
        /* B-perm: row t16 (256 B), word 4 l16 + m at m*64 + (l16>>1)*8 + (l16&1)*4 */
        shr(e, tmp, l16, 1); andi(e, R(tmp2), R(l16), 1u);
        imadi(e, R(nbp), R(t16), 256u, NOREG); imadi(e, R(nbp), R(tmp), 8u, nbp); imadi(e, R(nbp), R(tmp2), 4u, nbp);
        /* A-perm: word (t, w = 8j + 4r + t') at j*512 + (4(t&7) + t')*16 + ((t>>3) + 2r)*4; j = l16>>1, r = l16&1, t' = m */
        imadi(e, R(nap), R(tmp), 512u, NOREG); imadi(e, R(nap), R(tmp2), 8u, nap);
        andi(e, R(tmp), R(t16), 7u); imadi(e, R(nap), R(tmp), 64u, nap);
        shr(e, tmp, t16, 3); imadi(e, R(nap), R(tmp), 4u, nap);
        imadw(e, zb, R(h), 256u, pq); imadw(e, zb, R(l16), 16u, zb);
        imadr(e, tmp, t16, os16, NOREG);
        imadw(e, ao, R(tmp), 1u, pout); imadw(e, ao, R(h), 256u, ao); imadw(e, ao, R(l16), 16u, ao);
        imadi(e, R(os16), R(os16), 16u, NOREG);
        /* solve column: tid < 128 the v column tid (XV), else the k^ column tid - 128 (KHT); coefficient BETA or NBG */
        shr(e, tmp, tid, 7);
        imadi(e, R(srb), R(tid), 64u, NOREG); addi(e, srb, srb, OMEGA_GDNC_SM_XV);
        imadi(e, R(srb), R(tmp), (uint32_t)-(int32_t)OMEGA_GDNC_SM_XV, srb);
        imadi(e, R(scb), R(tmp), OMEGA_GDNC_SC_NBG - OMEGA_GDNC_SC_BETA, NOREG); addi(e, scb, scb, SC + OMEGA_GDNC_SC_BETA);
    }
    /* snapshots */
    int cd = V(e), ev4 = V(e), scnt = V(e), sstr = V(e), sidx = V(e);
    if (e->err) return -1;
    ldc32(e, ev4, OMEGA_GDNC_ARG_SNAP_EVERY4); ldc32(e, scnt, OMEGA_GDNC_ARG_SNAP_COUNT); ldc32(e, sstr, OMEGA_GDNC_ARG_SNAP_STRIDE);
    mov(e, R(cd), R(ev4)); movrz(e, R(sidx));

    movrz(e, R(t0));
    roff_write(e, warp, lane, c6, c16, t0, tm1, xs);
    bar_sync(e);

    const int L = here(e);
    loop_begin(e);
    {
        /* ================= P1: conv + SiLU (warps 0..5), scalars (warp 7) ================= */
        isetp_ge_u32(e, warp, c6);
        int skc = bra_if_p0_fwd(e);
        {
            int ri = V(e), a0 = V(e), it = V(e);
            if (e->err) return -1;
            movi(e, ri, OMEGA_GDNC_SM_ROFF); mov(e, R(a0), R(cs)); movrz(e, R(it));
            const int L2 = here(e);
            loop_begin(e);
            {
                int rq0 = V128(e), rq1 = V128(e), ad[8], wv[8], x0[8], x1[8], acc0 = V(e), acc1 = V(e), a1 = V(e), tt = V(e);
                for (unsigned i = 0; i < 8; i++) { ad[i] = V64(e); wv[i] = V(e); x0[i] = V(e); x1[i] = V(e); }
                if (e->err) return -1;
                lds128(e, rq0, ri, 0, 1); lds128(e, rq1, ri, 16, 0);
                for (unsigned i = 0; i < 8; i++) imadw(e, ad[i], S(i < 4 ? rq0 : rq1, i & 3u), 1u, xb);
                for (unsigned i = 0; i < 8; i++) ldg32(e, R(wv[i]), ad[i], udesc, i == 0);
                for (unsigned i = 0; i < 8; i++) {
                    imadi(e, R(x0[i]), R(wv[i]), 65536u, NOREG); andi(e, R(x1[i]), R(wv[i]), 0xffff0000u);
                    fmul(e, R(x0[i]), R(x0[i]), R(insc)); fmul(e, R(x1[i]), R(x1[i]), R(insc));
                }
                for (unsigned i = 0; i < 8; i++) {
                    Op w0[3], w1[3];
                    for (unsigned k = 0; k < 3; k++) {
                        unsigned p = i + k;
                        w0[k] = p < 3 ? S(tl0, p) : R(x0[p - 3]);
                        w1[k] = p < 3 ? S(tl1, p) : R(x1[p - 3]);
                    }
                    fmul(e, R(acc0), S(wq0, 0), w0[0]); ffma(e, R(acc0), S(wq0, 1), w0[1], R(acc0));
                    ffma(e, R(acc0), S(wq0, 2), w0[2], R(acc0)); ffma(e, R(acc0), S(wq0, 3), R(x0[i]), R(acc0));
                    fmul(e, R(acc1), S(wq1, 0), w1[0]); ffma(e, R(acc1), S(wq1, 1), w1[1], R(acc1));
                    ffma(e, R(acc1), S(wq1, 2), w1[2], R(acc1)); ffma(e, R(acc1), S(wq1, 3), R(x1[i]), R(acc1));
                    silu(e, R(acc0), tt, cnl2e, fone);
                    silu(e, R(acc1), tt, cnl2e, fone);
                    iadd(e, a1, a0, cstr);
                    sts32(e, a0, R(acc0)); sts32(e, a1, R(acc1));
                    iadd(e, a0, a0, tstr);
                }
                for (unsigned k = 0; k < 3; k++) { mov(e, S(tl0, k), R(x0[5 + k])); mov(e, S(tl1, k), R(x1[5 + k])); }
                addi(e, ri, ri, 32u);
                addi(e, it, it, 1u);
                isetp_ge_u32(e, one, it);
                bra_if_p0(e, L2);
            }
            loop_end(e);
        }
        patch_bra(e, skc, here(e));

        isetp_ge_u32(e, c6, warp);
        int sks = bra_if_p0_fwd(e);
        {
            int t = V(e), tc = V(e), a = V(e), b = V(e), vt = V(e), vf = V(e), off = V(e), aab = V64(e), abq = V64(e);
            int xsc = V(e), mn = V(e), sp = V(e), lg = V(e), beta = V(e), t1 = V(e), sf = V(e), E = V(e), tot = V(e);
            int gam = V(e), lam = V(e), gc = V(e), nb = V(e), nbg = V(e), fo = V(e), sl = V(e);
            if (e->err) return -1;
            iadd(e, t, t0, lane);
            clamp_u(e, tc, t, tm1);
            imadi(e, R(a), R(T), 0xffffffffu, t); shr(e, a, a, 31);         /* t < T */
            imadi(e, R(b), R(c16), 0xffffffffu, lane); shr(e, b, b, 31);    /* lane < 16 */
            lop3(e, vt, a, b, NOREG, 0xc0u);
            i2fp(e, vf, vt);
            imadi(e, R(off), R(h), 8u, NOREG);
            imadr(e, off, tc, abs_, off);
            imadw(e, aab, R(off), 1u, pab);
            ldg_wide(e, BW_IR_LDG_E_64, abq, aab, 0, udesc, 1);
            /* softplus and decay exactly as the multi kernel; lg = log2(decay) */
            fadd(e, R(xsc), S(abq, 0), S(scq, 1));
            movi(e, mn, C_M1); fmul(e, R(mn), R(xsc), R(mn)); fminn(e, R(mn), R(xsc), R(mn));
            fmul(e, R(mn), R(mn), R(cl2e)); mufu(e, BW_IR_MUFU_EX2, R(mn), R(mn)); fadd(e, R(mn), R(mn), R(fone));
            mufu(e, BW_IR_MUFU_LG2, R(mn), R(mn));
            movrz(e, R(sp)); fmaxx(e, R(sp), R(xsc), R(sp));
            movi(e, t1, C_LN2); ffma(e, R(sp), R(mn), R(t1), R(sp));
            if (MUT == OMEGA_GDNC_MUTANT_DROP_DECAY) movrz(e, R(lg));
            else { fmul(e, R(lg), S(scq, 0), R(sp)); fmul(e, R(lg), R(lg), R(cl2e)); }
            fmul(e, R(lg), R(lg), R(vf));
            fmul(e, R(beta), S(abq, 1), R(cnl2e)); mufu(e, BW_IR_MUFU_EX2, R(beta), R(beta));
            fadd(e, R(beta), R(beta), R(fone)); mufu(e, BW_IR_MUFU_RCP, R(beta), R(beta));
            fmul(e, R(beta), R(beta), R(vf));
            /* inclusive suffix sum over lanes 0..15 (lanes >= 16 carry 0); E = exclusive */
            mov(e, R(sf), R(lg));
            for (uint32_t o = 1; o <= 8; o *= 2) { shfl_down(e, t1, sf, o); fadd(e, R(sf), R(sf), R(t1)); }
            shfl_down(e, E, sf, 1);
            mov(e, R(tot), R(lg));
            bfly_sum(e, tot, t1, 1, 16);
            fsub(e, R(gam), R(tot), R(E)); mufu(e, BW_IR_MUFU_EX2, R(gam), R(gam));
            mufu(e, BW_IR_MUFU_EX2, R(lam), R(E));
            mufu(e, BW_IR_MUFU_EX2, R(gc), R(tot));
            movi(e, t1, C_M1); fmul(e, R(nb), R(beta), R(t1));
            fmul(e, R(nbg), R(nb), R(gam));
            movi(e, fo, C_ONE);
            imadi(e, R(sl), R(lane), 4u, NOREG); addi(e, sl, sl, SC);
            {
                const uint32_t offs[8] = { OMEGA_GDNC_SC_E, OMEGA_GDNC_SC_BETA, OMEGA_GDNC_SC_NBETA, OMEGA_GDNC_SC_GAM,
                                           OMEGA_GDNC_SC_LAM, OMEGA_GDNC_SC_NBG, OMEGA_GDNC_SC_GC, OMEGA_GDNC_SC_ONES };
                const int vals[8] = { E, beta, nb, gam, lam, nbg, gc, fo };
                int aa[8];
                for (unsigned i = 0; i < 8; i++) aa[i] = V(e);
                if (e->err) return -1;
                for (unsigned i = 0; i < 8; i++) addi(e, aa[i], sl, offs[i]);
                isetp_ge_u32(e, lane, c16);
                Region r = region_skip_if_p0(e);
                for (unsigned i = 0; i < 8; i++) sts32(e, aa[i], R(vals[i]));
                region_end(e, r);
            }
        }
        patch_bra(e, sks, here(e));
        bar_sync(e); /* B1 */

        /* ================= P2: l2 norms; operand layouts ================= */
        int zq = V128(e);
        if (e->err) return -1;
        {
            int q0 = V128(e), q1 = V128(e), k0 = V128(e), k1 = V128(e), ssq = V(e), ssk = V(e), t1 = V(e), gam = V(e), zo = V(e), az = V64(e);
            int qh[8], kh[8], gq[8];
            for (unsigned i = 0; i < 8; i++) { qh[i] = V(e); kh[i] = V(e); gq[i] = V(e); }
            if (e->err) return -1;
            lds128(e, q0, nrow, OMEGA_GDNC_SM_XQ, 1); lds128(e, q1, nrow, OMEGA_GDNC_SM_XQ + 16u, 0);
            lds128(e, k0, nrow, OMEGA_GDNC_SM_XK, 0); lds128(e, k1, nrow, OMEGA_GDNC_SM_XK + 16u, 0);
            lds32(e, R(gam), agam, 0); lds32(e, R(zo), aroff, 0);
            imadw(e, az, R(zo), 1u, zb);
            ldg_pf(e, zq, az, CONV * 2u, udesc, SB_GATE);
            fmul(e, R(ssq), S(q0, 0), S(q0, 0));
            for (unsigned i = 1; i < 8; i++) ffma(e, R(ssq), S(i < 4 ? q0 : q1, i & 3u), S(i < 4 ? q0 : q1, i & 3u), R(ssq));
            fmul(e, R(ssk), S(k0, 0), S(k0, 0));
            for (unsigned i = 1; i < 8; i++) ffma(e, R(ssk), S(i < 4 ? k0 : k1, i & 3u), S(i < 4 ? k0 : k1, i & 3u), R(ssk));
            bfly_sum(e, ssq, t1, 1, 8);
            bfly_sum(e, ssk, t1, 1, 8);
            fadd(e, R(ssq), R(ssq), R(el2)); mufu(e, BW_IR_MUFU_RSQ, R(ssq), R(ssq));
            fadd(e, R(ssk), R(ssk), R(el2)); mufu(e, BW_IR_MUFU_RSQ, R(ssk), R(ssk));
            movi(e, t1, fbits(qs));
            for (unsigned i = 0; i < 8; i++) {
                int qq = i < 4 ? q0 : q1, kk = i < 4 ? k0 : k1;
                fmul(e, R(qh[i]), S(qq, i & 3u), R(ssq)); fmul(e, R(qh[i]), R(qh[i]), R(t1));
                fmul(e, R(kh[i]), S(kk, i & 3u), R(ssk));
                fmul(e, R(gq[i]), R(qh[i]), R(gam));
            }
            bar_sync(e); /* B2: the raw q, k are read */
            {
                int a = V(e), wq[4], wk[4], wg[4];
                for (unsigned m = 0; m < 4; m++) { wq[m] = V(e); wk[m] = V(e); wg[m] = V(e); }
                if (e->err) return -1;
                for (unsigned i = 0; i < 8; i++) { addi(e, a, nkht, i * 64u); sts32(e, a, R(kh[i])); }
                for (unsigned m = 0; m < 4; m++) {
                    f2fp(e, R(wq[m]), R(qh[2 * m + 1]), R(qh[2 * m]));
                    f2fp(e, R(wk[m]), R(kh[2 * m + 1]), R(kh[2 * m]));
                    f2fp(e, R(wg[m]), R(gq[2 * m + 1]), R(gq[2 * m]));
                }
                for (unsigned m = 0; m < 4; m++) {
                    addi(e, a, nbp, OMEGA_GDNC_SM_QB + m * 64u); sts32(e, a, R(wg[m]));
                    addi(e, a, nbp, OMEGA_GDNC_SM_KB + m * 64u); sts32(e, a, R(wk[m]));
                    addi(e, a, nap, OMEGA_GDNC_SM_QT + m * 16u); sts32(e, a, R(wq[m]));
                    addi(e, a, nap, OMEGA_GDNC_SM_KA + m * 16u); sts32(e, a, R(wk[m]));
                }
            }
            bar_sync(e); /* B3 */
        }

        /* ================= P3: gram (warps 0, 1); next ROFF (warp 7) ================= */
        isetp_ge_u32(e, warp, c2);
        int skg = bra_if_p0_fwd(e);
        {
            int ga = V(e), gbb = V(e), A[8], B[2][4], C[2];
            for (unsigned j = 0; j < 8; j++) A[j] = V128(e);
            for (unsigned nt = 0; nt < 2; nt++) { C[nt] = V128(e); for (unsigned jj = 0; jj < 4; jj++) B[nt][jj] = V128(e); }
            if (e->err) return -1;
            imadi(e, R(ga), R(lane), 16u, NOREG); addi(e, ga, ga, OMEGA_GDNC_SM_KA);
            imadi(e, R(ga), R(warp), (uint32_t)-(int32_t)(OMEGA_GDNC_SM_KA - OMEGA_GDNC_SM_QT), ga);
            imadi(e, R(gbb), R(g), 256u, NOREG); imadi(e, R(gbb), R(tp), 64u, gbb); addi(e, gbb, gbb, OMEGA_GDNC_SM_KB);
            for (unsigned j = 0; j < 8; j++) lds128(e, A[j], ga, j * 512u, j == 0);
            for (unsigned nt = 0; nt < 2; nt++) for (unsigned jj = 0; jj < 4; jj++) lds128(e, B[nt][jj], gbb, nt * 2048u + jj * 16u, 0);
            for (unsigned j = 0; j < 8; j++)
                for (unsigned nt = 0; nt < 2; nt++) hmma(e, C[nt], A[j], S(B[nt][j >> 1], (j & 1u) * 2u), j == 0 ? NOREG : C[nt], 15);
            nop(e, 14);
            int Et[2], cf[2], Es[2][2], ad[8], sb = V(e), m = V(e), mf = V(e), d = V(e);
            for (unsigned i = 0; i < 2; i++) { Et[i] = V(e); cf[i] = V(e); Es[i][0] = V(e); Es[i][1] = V(e); }
            for (unsigned i = 0; i < 8; i++) ad[i] = V(e);
            if (e->err) return -1;
            imadi(e, R(ad[0]), R(g), 4u, NOREG); addi(e, ad[0], ad[0], SC + OMEGA_GDNC_SC_E);
            addi(e, ad[1], ad[0], 32u);
            imadi(e, R(ad[2]), R(warp), OMEGA_GDNC_SC_ONES - OMEGA_GDNC_SC_NBETA, ad[0]);
            addi(e, ad[2], ad[2], OMEGA_GDNC_SC_NBETA - OMEGA_GDNC_SC_E);
            addi(e, ad[3], ad[2], 32u);
            imadi(e, R(ad[4]), R(tp), 8u, NOREG); addi(e, ad[4], ad[4], SC + OMEGA_GDNC_SC_E);
            addi(e, ad[5], ad[4], 4u); addi(e, ad[6], ad[4], 32u); addi(e, ad[7], ad[4], 36u);
            lds32(e, R(Et[0]), ad[0], 1); lds32(e, R(Et[1]), ad[1], 0);
            lds32(e, R(cf[0]), ad[2], 0); lds32(e, R(cf[1]), ad[3], 0);
            lds32(e, R(Es[0][0]), ad[4], 0); lds32(e, R(Es[0][1]), ad[5], 0);
            lds32(e, R(Es[1][0]), ad[6], 0); lds32(e, R(Es[1][1]), ad[7], 0);
            /* sb = 2 tp - g - warp: element (nt, q) has s - t - warp = sb + 8 nt + (q&1) - 8 (q>>1) */
            imadi(e, R(sb), R(g), 0xffffffffu, NOREG); imadi(e, R(sb), R(tp), 2u, sb); imadi(e, R(sb), R(warp), 0xffffffffu, sb);
            for (unsigned nt = 0; nt < 2; nt++)
                for (unsigned q = 0; q < 4; q++) {
                    addi(e, m, sb, nt * 8u + (q & 1u) - 8u * (q >> 1));
                    shr(e, m, m, 31); i2fp(e, mf, m);
                    fsub(e, R(d), R(Es[nt][q & 1u]), R(Et[q >> 1])); fminn(e, R(d), R(d), R(zr));
                    mufu(e, BW_IR_MUFU_EX2, R(d), R(d));
                    fmul(e, S(C[nt], q), S(C[nt], q), R(d)); fmul(e, S(C[nt], q), S(C[nt], q), R(mf));
                    fmul(e, S(C[nt], q), S(C[nt], q), R(cf[q >> 1]));
                }
            isetp_ge_u32(e, warp, one);
            int skam = bra_if_p0_fwd(e);
            {
                int base = V(e), a = V(e);
                if (e->err) return -1;
                imadi(e, R(base), R(g), 64u, NOREG); imadi(e, R(base), R(tp), 8u, base); addi(e, base, base, OMEGA_GDNC_SM_AM);
                for (unsigned nt = 0; nt < 2; nt++)
                    for (unsigned q = 0; q < 4; q++) { addi(e, a, base, (q >> 1) * 512u + nt * 32u + (q & 1u) * 4u); sts32(e, a, S(C[nt], q)); }
            }
            patch_bra(e, skam, here(e));
            {
                int w1 = V(e);
                if (e->err) return -1;
                addi(e, w1, warp, 0xffffffffu);
                isetp_ge_u32(e, w1, one);
            }
            int skpb = bra_if_p0_fwd(e);
            {
                int base = V(e), a = V(e), pw = V128(e);
                if (e->err) return -1;
                f2fp(e, S(pw, 0), S(C[0], 1), S(C[0], 0)); f2fp(e, S(pw, 1), S(C[1], 1), S(C[1], 0));
                f2fp(e, S(pw, 2), S(C[0], 3), S(C[0], 2)); f2fp(e, S(pw, 3), S(C[1], 3), S(C[1], 2));
                imadi(e, R(base), R(lane), 16u, NOREG); addi(e, base, base, OMEGA_GDNC_SM_PB);
                for (unsigned q = 0; q < 4; q++) { addi(e, a, base, q * 4u); sts32(e, a, S(pw, q)); }
            }
            patch_bra(e, skpb, here(e));
        }
        patch_bra(e, skg, here(e));
        {
            int tn = V(e);
            if (e->err) return -1;
            addi(e, tn, t0, OMEGA_GDNC_SUB);
            roff_write(e, warp, lane, c6, c16, tn, tm1, xs);
        }
        bar_sync(e); /* B4 */

        /* ================= P4: forward substitution, column tid ================= */
        {
            int rh[4], cq[4], x[16], amq = V128(e);
            for (unsigned i = 0; i < 4; i++) { rh[i] = V128(e); cq[i] = V128(e); }
            for (unsigned t = 0; t < 16; t++) x[t] = V(e);
            if (e->err) return -1;
            for (unsigned i = 0; i < 4; i++) lds128(e, rh[i], srb, i * 16u, i == 0);
            for (unsigned i = 0; i < 4; i++) lds128(e, cq[i], scb, i * 16u, 0);
            for (unsigned t = 0; t < 16; t++) fmul(e, R(x[t]), S(cq[t >> 2], t & 3u), S(rh[t >> 2], t & 3u));
            for (unsigned t = 1; t < 16; t++)
                for (unsigned jq = 0; jq * 4u < t; jq++) {
                    lds128(e, amq, zr, OMEGA_GDNC_SM_AM + t * 64u + jq * 16u, 1);
                    for (unsigned s = jq * 4u; s < jq * 4u + 4u && s < t; s++) ffma(e, R(x[t]), S(amq, s & 3u), R(x[s]), R(x[t]));
                }
            bar_sync(e); /* B5: XV, KHT and AM are read */
            {
                int c4 = V(e);
                if (e->err) return -1;
                movi(e, c4, 4);
                isetp_ge_u32(e, warp, c4); /* tid >= 128: the k path */
            }
            int sku = bra_if_p0_fwd(e);
            {
                int base = V(e), tmp = V(e), a = V(e);
                if (e->err) return -1;
                /* U^T (v = tid, t) at XV + (v>>4)*1024 + (v&7)*64 + ((v>>3)&1)*8 + (t>>3)*512 + ((t&7)>>1)*16 + (t&1)*4 */
                shr(e, tmp, tid, 4); imadi(e, R(base), R(tmp), 1024u, NOREG);
                andi(e, R(tmp), R(tid), 7u); imadi(e, R(base), R(tmp), 64u, base);
                shr(e, tmp, tid, 3); andi(e, R(tmp), R(tmp), 1u); imadi(e, R(base), R(tmp), 8u, base);
                addi(e, base, base, OMEGA_GDNC_SM_XV);
                for (unsigned t = 0; t < 16; t++) { addi(e, a, base, (t >> 3) * 512u + ((t & 7u) >> 1) * 16u + (t & 1u) * 4u); sts32(e, a, R(x[t])); }
            }
            int sku_end = -1;
            {
                /* jump over the k path */
                BlackwellIRInsn i = I0(BW_IR_BRA); i.imm = 1; sku_end = here(e); emit(e, i, K_BRA);
            }
            patch_bra(e, sku, here(e));
            {
                int kk = V(e), tmp = V(e), wbase = V(e), kbase = V(e), pm = V(e), o = V(e), hi = V(e), lo = V(e), wd = V(e), a = V(e);
                int lq[4], kt[16];
                for (unsigned i = 0; i < 4; i++) lq[i] = V128(e);
                for (unsigned s = 0; s < 16; s++) kt[s] = V(e);
                if (e->err) return -1;
                addi(e, kk, tid, (uint32_t)-128);
                /* WB: row t (256 B), word w = kk>>1 at pos(w) = (w&3)*16 + (w>>3)*2 + ((w>>2)&1) */
                shr(e, tmp, kk, 1); andi(e, R(tmp), R(tmp), 3u); imadi(e, R(wbase), R(tmp), 64u, NOREG);
                shr(e, tmp, kk, 4); imadi(e, R(wbase), R(tmp), 8u, wbase);
                shr(e, tmp, kk, 3); andi(e, R(tmp), R(tmp), 1u); imadi(e, R(wbase), R(tmp), 4u, wbase);
                addi(e, wbase, wbase, OMEGA_GDNC_SM_WB);
                andi(e, R(pm), R(lane), 1u); imadi(e, R(pm), R(pm), 0xffffffffu, NOREG); /* odd lane: ~0 */
                for (unsigned t = 0; t < 16; t++) {
                    shfl_bfly(e, o, x[t], 1);
                    lop3(e, hi, x[t], o, pm, 0xe4u);   /* odd: own value is the high half */
                    lop3(e, lo, o, x[t], pm, 0xe4u);
                    f2fp(e, R(wd), R(hi), R(lo));
                    addi(e, a, wbase, t * 256u);
                    sts32(e, a, R(wd));
                }
                /* KT: channel kk, words m = (LAM_2m+1 k^_2m+1, LAM_2m k^_2m) at
                 * (nt>>1)*512 + g*64 + (nt&1)*8 + (m&3)*16 + (m>>2)*4, nt = kk>>3, g = kk&7 */
                for (unsigned i = 0; i < 4; i++) lds128(e, lq[i], zr, SC + OMEGA_GDNC_SC_LAM + i * 16u, i == 0);
                for (unsigned s = 0; s < 16; s++) fmul(e, R(kt[s]), S(lq[s >> 2], s & 3u), S(rh[s >> 2], s & 3u));
                shr(e, tmp, kk, 4); imadi(e, R(kbase), R(tmp), 512u, NOREG);
                andi(e, R(tmp), R(kk), 7u); imadi(e, R(kbase), R(tmp), 64u, kbase);
                shr(e, tmp, kk, 3); andi(e, R(tmp), R(tmp), 1u); imadi(e, R(kbase), R(tmp), 8u, kbase);
                addi(e, kbase, kbase, OMEGA_GDNC_SM_KT);
                for (unsigned mm = 0; mm < 8; mm++) {
                    f2fp(e, R(wd), R(kt[2 * mm + 1]), R(kt[2 * mm]));
                    addi(e, a, kbase, (mm & 3u) * 16u + (mm >> 2) * 4u);
                    sts32(e, a, R(wd));
                }
            }
            patch_bra(e, sku_end, here(e));
            bar_sync(e); /* B6 */
        }

        /* ================= P5: D^T, O^T, S^T on the tensor cores ================= */
        {
            int af[8], D0 = V128(e), D1 = V128(e), O0 = V128(e), O1 = V128(e), df = V128(e), pbq = V128(e);
            int p5u = V(e), p5w = V(e), p5q = V(e), tmp = V(e), gc = V(e), agc = V(e);
            for (unsigned j = 0; j < 8; j++) af[j] = V128(e);
            if (e->err) return -1;
            imadi(e, R(p5u), R(warp), 1024u, NOREG); imadi(e, R(p5u), R(lane), 16u, p5u); addi(e, p5u, p5u, OMEGA_GDNC_SM_XV);
            imadi(e, R(p5q), R(g), 256u, NOREG); imadi(e, R(p5q), R(tp), 64u, p5q);
            addi(e, p5w, p5q, OMEGA_GDNC_SM_WB);
            lds128(e, D0, p5u, 0, 1); lds128(e, D1, p5u, 512u, 0);
            for (unsigned j = 0; j < 8; j++) {
                f2fp(e, S(af[j], 0), S(sq[2 * j], 1), S(sq[2 * j], 0));
                f2fp(e, S(af[j], 1), S(sq[2 * j], 3), S(sq[2 * j], 2));
                f2fp(e, S(af[j], 2), S(sq[2 * j + 1], 1), S(sq[2 * j + 1], 0));
                f2fp(e, S(af[j], 3), S(sq[2 * j + 1], 3), S(sq[2 * j + 1], 2));
            }
            for (unsigned jj = 0; jj < 4; jj++) {
                int wb0 = V128(e), wb1 = V128(e), qb0 = V128(e), qb1 = V128(e);
                if (e->err) return -1;
                lds128(e, wb0, p5w, jj * 16u, 1); lds128(e, wb1, p5w, 2048u + jj * 16u, 0);
                lds128(e, qb0, p5q, jj * 16u, 0); lds128(e, qb1, p5q, 2048u + jj * 16u, 0);
                for (unsigned h2 = 0; h2 < 2; h2++) {
                    unsigned kk = 2u * jj + h2, sub = h2 * 2u;
                    hmma(e, D0, af[kk], S(wb0, sub), D0, 8);
                    hmma(e, D1, af[kk], S(wb1, sub), D1, 8);
                    hmma(e, O0, af[kk], S(qb0, sub), kk == 0 ? NOREG : O0, 8);
                    hmma(e, O1, af[kk], S(qb1, sub), kk == 0 ? NOREG : O1, 8);
                }
            }
            nop(e, 15); nop(e, 14);
            f2fp(e, S(df, 0), S(D0, 1), S(D0, 0)); f2fp(e, S(df, 1), S(D0, 3), S(D0, 2));
            f2fp(e, S(df, 2), S(D1, 1), S(D1, 0)); f2fp(e, S(df, 3), S(D1, 3), S(D1, 2));
            imadi(e, R(tmp), R(lane), 16u, NOREG); addi(e, tmp, tmp, OMEGA_GDNC_SM_PB);
            movi(e, agc, SC + OMEGA_GDNC_SC_GC);
            lds128(e, pbq, tmp, 0, 1);
            lds32(e, R(gc), agc, 0);
            hmma(e, O0, df, S(pbq, 0), O0, 15);
            hmma(e, O1, df, S(pbq, 2), O1, 15);
            for (unsigned n = 0; n < 16; n++) for (unsigned q = 0; q < 4; q++) fmul(e, S(sq[n], q), S(sq[n], q), R(gc));
            {
                int ktq[8], p5k = V(e);
                for (unsigned p = 0; p < 8; p++) ktq[p] = V128(e);
                if (e->err) return -1;
                imadi(e, R(p5k), R(lane), 16u, NOREG); addi(e, p5k, p5k, OMEGA_GDNC_SM_KT);
                for (unsigned p = 0; p < 8; p++) lds128(e, ktq[p], p5k, p * 512u, p == 0);
                for (unsigned n = 0; n < 16; n++) hmma(e, sq[n], df, S(ktq[n >> 1], (n & 1u) * 2u), sq[n], 8);
                nop(e, 15); nop(e, 14);
            }
            {
                int base = V(e), a = V(e);
                if (e->err) return -1;
                imadi(e, R(base), R(tp), 1024u, NOREG); imadi(e, R(base), R(warp), 64u, base); imadi(e, R(base), R(g), 4u, base);
                addi(e, base, base, OMEGA_GDNC_SM_XO);
                for (unsigned nt = 0; nt < 2; nt++)
                    for (unsigned q = 0; q < 4; q++) {
                        addi(e, a, base, nt * 4096u + (q & 1u) * 512u + (q >> 1) * 32u);
                        sts32(e, a, S(nt ? O1 : O0, q));
                    }
            }
            bar_sync(e); /* B7 */
        }

        /* ================= P6: gated RMSNorm, bf16 out ================= */
        {
            int o0 = V128(e), o1 = V128(e), ss = V(e), t1 = V(e), gz = V(e), pv[8], pw = V128(e), tg = V(e);
            for (unsigned i = 0; i < 8; i++) pv[i] = V(e);
            if (e->err) return -1;
            lds128(e, o0, nrow, OMEGA_GDNC_SM_XO, 1); lds128(e, o1, nrow, OMEGA_GDNC_SM_XO + 16u, 0);
            fmul(e, R(ss), S(o0, 0), S(o0, 0));
            for (unsigned i = 1; i < 8; i++) ffma(e, R(ss), S(i < 4 ? o0 : o1, i & 3u), S(i < 4 ? o0 : o1, i & 3u), R(ss));
            bfly_sum(e, ss, t1, 1, 8);
            movi(e, t1, C_INV128);
            ffma(e, R(ss), R(ss), R(t1), R(en));
            mufu(e, BW_IR_MUFU_RSQ, R(ss), R(ss));
            consume(e, SB_GATE);
            for (unsigned i = 0; i < 8; i++) {
                if (i % 2 == 0) imadi(e, R(gz), S(zq, i / 2), 65536u, NOREG);
                else andi(e, R(gz), S(zq, i / 2), 0xffff0000u);
                fmul(e, R(gz), R(gz), R(insc));
                silu(e, R(gz), t1, cnl2e, fone);
                fmul(e, R(pv[i]), S(i < 4 ? o0 : o1, i & 3u), R(ss));
                fmul(e, R(pv[i]), R(pv[i]), S(i < 4 ? nw0 : nw1, i & 3u));
                fmul(e, R(pv[i]), R(pv[i]), R(gz));
                fmul(e, R(pv[i]), R(pv[i]), R(osc));
            }
            for (unsigned m = 0; m < 4; m++) f2fp(e, S(pw, m), R(pv[2 * m + 1]), R(pv[2 * m]));
            iadd(e, tg, t0, t16);
            isetp_ge_u32(e, tg, T);
            Region r = region_skip_if_p0(e);
            for (unsigned m = 0; m < 4; m++) stg(e, ao, m * 4u, S(pw, m), udesc);
            region_end(e, r);
            imadw(e, ao, R(os16), 1u, ao);
        }

        /* ================= snapshot every SNAP_EVERY4 sub-chunks ================= */
        addi(e, cd, cd, 0xffffffffu);
        isetp_ge_u32(e, cd, one);
        int sk1 = bra_if_p0_fwd(e);
        mov(e, R(cd), R(ev4));
        isetp_ge_u32(e, sidx, scnt);
        int sk2 = bra_if_p0_fwd(e);
        {
            int sp = V64(e);
            if (e->err) return -1;
            imadw(e, sp, R(loff), 1u, psnap);
            for (unsigned n = 0; n < 16; n++)
                for (unsigned q = 0; q < 4; q++) stg(e, sp, n * 4096u + (q & 1u) * 512u + (q >> 1) * 32u, S(sq[n], q), udesc);
            isetp_ge_u32(e, warp, c6);
            int sk3 = bra_if_p0_fwd(e);
            imadw(e, sp, R(tso), 1u, psnap);
            for (unsigned k = 0; k < 3; k++) { stg(e, sp, k * 4u, S(tl0, k), udesc); stg(e, sp, 16u + k * 4u, S(tl1, k), udesc); }
            patch_bra(e, sk3, here(e));
            imadw(e, psnap, R(sstr), 1u, psnap);
            addi(e, sidx, sidx, 1u);
        }
        patch_bra(e, sk2, here(e));
        patch_bra(e, sk1, here(e));

        addi(e, t0, t0, OMEGA_GDNC_SUB);
        isetp_ge_u32(e, tm1, t0);
        bra_if_p0(e, L);
    }
    loop_end(e);

    /* ---- the state back, once ---- */
    for (unsigned n = 0; n < 16; n++)
        for (unsigned q = 0; q < 4; q++) stg(e, sa, n * 4096u + (q & 1u) * 512u + (q >> 1) * 32u, S(sq[n], q), udesc);
    /* ---- the conv tail: x of tokens T-3..T-1 (times in_scale), or the old tail where T < 3 (conv warps only) ---- */
    isetp_ge_u32(e, warp, c6);
    int skt = bra_if_p0_fwd(e);
    {
        int idx = V(e), ng = V(e), mneg = V(e), rc = V(e), off = V(e), fi = V(e), xa = V64(e), oa = V64(e), wv = V(e), xv = V(e), ov = V(e), fin = V(e);
        if (e->err) return -1;
        for (unsigned k = 0; k < 3; k++) {
            addi(e, idx, T, k - 3u);
            shr(e, ng, idx, 31);
            imadi(e, R(mneg), R(ng), 0xffffffffu, NOREG);           /* ~0 when idx < 0 */
            lop3(e, rc, idx, mneg, NOREG, 0x30u);                    /* idx & ~mneg */
            imadr(e, off, rc, xs, NOREG);
            imadw(e, xa, R(off), 1u, xb);
            addi(e, fi, T, k);
            lop3(e, fi, fi, mneg, NOREG, 0xc0u);                     /* T + k where idx < 0, else 0 */
            ldg32(e, R(wv), xa, udesc, 1);
            for (unsigned j = 0; j < 2; j++) {
                if (j == 0) imadw(e, oa, R(fi), 4u, ta);
                else imadw(e, oa, R(one), 16u, oa); /* the second channel's old tail starts 16 B later */
                ldg32(e, R(ov), oa, udesc, 1);
                if (j == 0) imadi(e, R(xv), R(wv), 65536u, NOREG);
                else andi(e, R(xv), R(wv), 0xffff0000u);
                fmul(e, R(xv), R(xv), R(insc));
                lop3(e, fin, ov, xv, mneg, 0xe4u);         /* old where idx < 0, else x */
                stg(e, ta, j * 16u + k * 4u, R(fin), udesc);
            }
        }
    }
    patch_bra(e, skt, here(e));
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_gdnc_prog_stats(const OmegaGdncSpec *spec, uint32_t *insns, uint32_t *vregs, uint32_t *gprs) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_gdnc_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (vregs) *vregs = (uint32_t)prog.regalloc.num_vregs;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}

uint32_t omega_gdnc_placeholder_count(const BlackwellIRProgram *prog) {
    uint32_t n = 0;
    for (size_t i = 0; i < prog->count; i++)
        if ((prog->insns[i].op == BW_IR_SHFL_DOWN && (prog->insns[i].imm & OMEGA_TC_MARK_BFLY))
            || (prog->insns[i].op == BW_IR_FMNMX_MAX && prog->insns[i].imm == OMEGA_TC_MARK_F2FP)) n++;
    return n;
}
