/*
 * omega_blackwell_codegen_pattn.c: causal prefill attention on tensor cores (see the header).
 * Emitter rules are those of omega_blackwell_codegen_attn.c / omega_blackwell_codegen_tc.c (control words, the read
 * barrier on every load, held scoreboards 3/4 for prefetch loads, loop live ranges, HMMA spacing: >= 4 independent
 * HMMAs at stall 8 between dependent ones, stall 15 + NOP 14 before a non-HMMA reader of an HMMA result).
 * The prologue's per-head RMSNorm + RoPE is the decode kernel's instruction sequence (omega_blackwell_codegen_attn.c,
 * prologue), so the appended K rows match the decode kernel bit for bit on the same chip.
 */
#include "omega_blackwell_codegen_pattn.h"
#include "omega_blackwell_codegen_tc.h"
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
    uint32_t pending, held;
    int err;
    int open_start[MAX_LOOPS];
    int depth;
    uint32_t placeholders;
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
static void movi(Em *e, Op d, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_MOV_IMM); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.imm = imm; emit(e, i, K_FIXED); }
static void movrz(Em *e, Op d) { BlackwellIRInsn i = I0(BW_IR_MOV_RZ); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; emit(e, i, K_FIXED); }
/* d = a * imm + c */
static void imadi(Em *e, Op d, Op a, uint32_t imm, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* d = a * b + c (registers) */
static void imadr(Em *e, int d, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* d64 = a * imm + c64 */
static void imadw(Em *e, int d64, int a, uint32_t imm, int c64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = d64; i.src1_vreg = a; i.imm = imm; i.src3_vreg = c64; emit(e, i, K_FIXED); }
static void mov(Em *e, Op d, Op a) { imadi(e, d, a, 1u, NOREG); }
static void shrs(Em *e, int d, Op a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = d; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; emit(e, i, K_FIXED); }
static void shr(Em *e, int d, int a, uint32_t imm) { shrs(e, d, R(a), imm); }
static void andi(Em *e, Op d, Op a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; emit(e, i, K_FIXED); }
static void xorr(Em *e, Op d, Op a, Op b) { BlackwellIRInsn i = I0(BW_IR_LOP3_XOR); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.src2_vreg = b.v; i.src2_subreg = (uint8_t)b.s; emit(e, i, K_FIXED); }
static void fop(Em *e, BlackwellIROpcode op, Op d, Op a, Op b, Op c) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s;
    i.src2_vreg = b.v; i.src2_subreg = (uint8_t)b.s; i.src3_vreg = c.v; i.src3_subreg = (uint8_t)c.s; emit(e, i, K_FIXED);
}
static void iadd(Em *e, int d, int a, int b) { fop(e, BW_IR_IADD3, R(d), R(a), R(b), R(NOREG)); }
static void iaddo(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_IADD3, d, a, b, R(NOREG)); }
static void fadd(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FADD, d, a, b, R(NOREG)); }
static void fsub(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FSUB, d, a, b, R(NOREG)); }
static void fmul(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FMUL, d, a, b, R(NOREG)); }
static void ffma(Em *e, Op d, Op a, Op b, Op c) { fop(e, BW_IR_FFMA, d, a, b, c); }
static void fmaxx(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FMNMX_MAX, d, a, b, R(NOREG)); }
static void mufu(Em *e, BlackwellIROpcode op, Op d, Op a, int first) { BlackwellIRInsn i = I0(op); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void ldg_wide(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, int first) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE);
}
static void stg(Em *e, int addr64, uint32_t imm, Op val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.imm = imm; i.src2_vreg = val.v; i.src2_subreg = (uint8_t)val.s; i.ureg = udesc; emit(e, i, K_STORE); }
static void membar(Em *e) { emit(e, I0(BW_IR_MEMBAR_ALL_SYS), K_FIXED); }
static void lds128(Em *e, int d128, int addr, uint32_t imm, int first) { BlackwellIRInsn i = I0(BW_IR_LDS128); i.dst_vreg = d128; i.src1_vreg = addr; i.imm = imm; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void lds32(Em *e, Op d, int addr, int first) { BlackwellIRInsn i = I0(BW_IR_LDS32); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = addr; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void sts32(Em *e, int addr, Op val) { BlackwellIRInsn i = I0(BW_IR_STS32); i.src1_vreg = addr; i.src2_vreg = val.v; i.src2_subreg = (uint8_t)val.s; emit(e, i, K_STORE); }
static void bar_sync(Em *e) { emit(e, I0(BW_IR_BAR_SYNC), K_SYNC); }
static void shfl_down(Em *e, int d, int a, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_SHFL_DOWN); i.dst_vreg = d; i.src1_vreg = a; i.imm = off; emit(e, i, K_VAR); }
/* SHFL.BFLY placeholder (omega_blackwell_codegen_tc.h), patched after encoding */
static void shfl_bfly(Em *e, int d, int a, uint32_t lanes) { shfl_down(e, d, a, OMEGA_TC_MARK_BFLY | (lanes & 31u)); e->placeholders++; }
/* F2FP.BF16.F32.PACK_AB placeholder: d = bf16_rn(hi) << 16 | bf16_rn(lo) */
static void f2fp(Em *e, Op d, Op hi, Op lo) {
    BlackwellIRInsn i = I0(BW_IR_FMNMX_MAX); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = hi.v; i.src1_subreg = (uint8_t)hi.s;
    i.src2_vreg = lo.v; i.src2_subreg = (uint8_t)lo.s; i.imm = OMEGA_TC_MARK_F2FP; emit(e, i, K_FIXED); e->placeholders++;
}
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

/* HMMA.16816.F32.BF16 d = a * b + c; b is the register pair at b + bsub (0 or 2); c NOREG = RZ */
static void hmma(Em *e, int d, int a, int b, unsigned bsub, int c, uint32_t stall) {
    BlackwellIRInsn i = I0(BW_IR_HMMA_BF16); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; i.src3_vreg = c;
    i.control = CW(stall, SB_NONE, SB_NONE, e->pending); e->pending = 0;
    if (omega_bw_ir_append(e->p, &i) < 0) e->err = -1;
}
static void nop(Em *e, uint32_t stall) {
    BlackwellIRInsn i = I0(BW_IR_NOP); i.control = CW(stall, SB_NONE, SB_NONE, e->pending); e->pending = 0;
    if (omega_bw_ir_append(e->p, &i) < 0) e->err = -1;
}
/* a prefetch LDS.128: scoreboard sb (3 or 4) is held, not waited for, until consume() */
static void lds_pf(Em *e, int d, int addr, uint32_t imm, unsigned sb) {
    BlackwellIRInsn in = I0(BW_IR_LDS128); in.dst_vreg = d; in.src1_vreg = addr; in.imm = imm;
    in.control = CW(4, sb, SB_RD, (e->pending & (1u << SB_RD))) | CW_YIELD;
    e->pending |= 1u << SB_RD; e->held |= 1u << sb;
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}
static void consume(Em *e, unsigned sb) { e->pending |= e->held & (1u << sb); e->held &= ~(1u << sb); }

/* bf16 round-to-nearest-even of f32 x into the low 16 bits of d (the decode kernel's sequence) */
static void bf16_of(Em *e, int d, Op x, int c7fff, int t) {
    shrs(e, t, x, 16);
    andi(e, R(t), R(t), 1u);
    iadd(e, t, t, c7fff);
    fop(e, BW_IR_IADD3, R(d), x, R(t), R(NOREG));
    shr(e, d, d, 16);
}
/* two bf16 of x[q0], x[q0+1] packed (element q0 low) */
static void bf16_pair(Em *e, Op d, int x4, unsigned q0, int c7fff, int h0, int h1, int tb) {
    bf16_of(e, h0, S(x4, q0), c7fff, tb); bf16_of(e, h1, S(x4, q0 + 1), c7fff, tb); imadi(e, d, R(h1), 65536u, h0);
}

/* shared memory map */
#define SM_K 0u            /* main loop: K block, [s 8][jp 2][lane 32][16 B] */
#define SM_V 8192u         /* main loop: V block, [kq 2][np 8][lane 32][16 B] */
#define SM_Q 0u            /* prologue: Q fragments of 2 tiles, [tile 2][s 8][lane 32][16 B] */
#define SM_X 8192u         /* prologue: per-warp f32 [128] partner staging */
#define SM_N 12288u        /* prologue: per-warp sum of squares */
static const uint32_t NEG_BIG = 0xF149F2CAu; /* -1e30f */

/* The decode kernel's RMSNorm + RoPE of the quad x4 (dims 4*lane .. +3) of one head, in place. sin4 carries the
 * sign already (lanes < 16 negative). Two CTA barriers (every warp runs the same number of items). */
typedef struct { int lane, one, eps, snw, sxl, sxp, sq, ts, rs, a1, pt4; } NR;
static void norm_rope(Em *e, const NR *n, int x4, int w4, int cos4, int sin4, int rope) {
    fmul(e, R(n->sq), S(x4, 0), S(x4, 0));
    for (unsigned i = 1; i < 4; i++) ffma(e, R(n->sq), S(x4, i), S(x4, i), R(n->sq));
    for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, n->ts, n->sq, o); fadd(e, R(n->sq), R(n->sq), R(n->ts)); }
    isetp_ge_u32(e, n->lane, n->one);
    { Region r = region_skip_if_p0(e); sts32(e, n->snw, R(n->sq)); region_end(e, r); }
    bar_sync(e);
    lds32(e, R(n->sq), n->snw, 1);
    movi(e, R(n->ts), 0x3c000000u);                       /* 1/128 */
    ffma(e, R(n->rs), R(n->sq), R(n->ts), R(n->eps));
    mufu(e, BW_IR_MUFU_RSQ, R(n->rs), R(n->rs), 1);
    for (unsigned i = 0; i < 4; i++) { fmul(e, S(x4, i), S(x4, i), R(n->rs)); fmul(e, S(x4, i), S(x4, i), S(w4, i)); }
    sts32(e, n->sxl, S(x4, 0));
    for (unsigned i = 1; i < 4; i++) { imadi(e, R(n->a1), R(n->one), 4u * i, n->sxl); sts32(e, n->a1, S(x4, i)); }
    bar_sync(e);
    lds128(e, n->pt4, n->sxp, 0, 1);
    if (rope) for (unsigned i = 0; i < 4; i++) { fmul(e, R(n->ts), S(x4, i), S(cos4, i)); ffma(e, S(x4, i), S(n->pt4, i), S(sin4, i), R(n->ts)); }
}

int omega_pattn_spec_ok(const OmegaPrefillAttnSpec *s) {
    return s && (s->warps == 4u || s->warps == 8u) && s->mutant >= 0 && s->mutant <= 4 && s->mode <= OMEGA_PATTN_MODE_ATTEND && (s->mirror == 0 || (s->mirror == 1 && s->mode == OMEGA_PATTN_MODE_APPEND));
}

int omega_blackwell_codegen_prefill_attn_prog(const OmegaPrefillAttnSpec *spec, BlackwellIRProgram *prog) {
    if (!omega_pattn_spec_ok(spec) || !prog) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    const uint32_t W = spec->warps, T = 32u * W, LW = W == 8u ? 3u : 2u;
    const uint32_t NKU = 512u / T, NVU = 256u / T, ROUNDS = W / 4u;
    const int MUT = spec->mutant;

    /* ---- arguments and identities ---- */
    int udesc = UV64(e);
    int pqkv = V64(e), pkc = V64(e), pvc = V64(e), prope = V64(e), pqn = V64(e), pkn = V64(e);
    int pos0 = V(e), mc = V(e), kvs = V(e), eps = V(e), qks = V(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), one = V(e), zero = V(e), c7fff = V(e), c16 = V(e);
    int cyk = V(e), nk = V(e), t1 = V(e), t2 = V(e), cxw = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pqkv, OMEGA_PATTN_ARG_QKV); ldc64(e, pkc, OMEGA_PATTN_ARG_KC); ldc64(e, pvc, OMEGA_PATTN_ARG_VC);
    ldc64(e, prope, OMEGA_PATTN_ARG_ROPE); ldc64(e, pqn, OMEGA_PATTN_ARG_QN); ldc64(e, pkn, OMEGA_PATTN_ARG_KN);
    ldc32(e, pos0, OMEGA_PATTN_ARG_POS0); ldc32(e, mc, OMEGA_PATTN_ARG_MC); ldc32(e, kvs, OMEGA_PATTN_ARG_KV_STRIDE);
    ldc32(e, eps, OMEGA_PATTN_ARG_EPS); ldc32(e, qks, OMEGA_PATTN_ARG_QK_SCALE);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, R(one), 1); movrz(e, R(zero)); movi(e, R(c7fff), 0x7fffu); movi(e, R(c16), 16);
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);
    imadr(e, cyk, cy, kvs, NOREG);                       /* byte offset of this KV head's plane */
    imadi(e, R(cxw), R(cx), W, NOREG);                   /* first chunk token of the CTA */
    /* nk = min(cxw + W, mc) = hi + over * (mc - hi), over = 1 when hi > mc */
    imadi(e, R(t2), R(one), W, cxw);                     /* hi */
    imadi(e, R(t1), R(t2), 0xffffffffu, mc);             /* mc - hi */
    { int ov = V(e); if (e->err) return -1;
      shr(e, ov, t1, 31); imadr(e, nk, ov, t1, t2); }

    /* ---- per-lane prologue constants ---- */
    NR n; memset(&n, 0, sizeof n);
    n.lane = lane; n.one = one; n.eps = eps;
    n.snw = V(e); n.sxl = V(e); n.sxp = V(e); n.sq = V(e); n.ts = V(e); n.rs = V(e); n.a1 = V(e); n.pt4 = V128(e);
    int sgn = V(e), ropel = V(e), ad64 = V64(e), rope64 = V64(e), cos4 = V128(e), sin4 = V128(e), tokc = V(e), item = V(e), cnt = V(e);
    int h0 = V(e), h1 = V(e), tb = V(e), w0 = V(e), w1 = V(e), w2 = V(e), w3 = V(e);
    if (e->err) return -1;
    imadi(e, R(n.snw), R(warp), 4u, NOREG); imadi(e, R(n.snw), R(one), SM_N, n.snw);
    imadi(e, R(t1), R(warp), 512u, NOREG); imadi(e, R(t1), R(one), SM_X, t1);
    imadi(e, R(n.sxl), R(lane), 16u, t1);
    { int pl = V(e); if (e->err) return -1;
      xorr(e, R(pl), R(lane), R(c16)); imadi(e, R(n.sxp), R(pl), 16u, t1); }
    imadi(e, R(sgn), R(one), 0xfffffff0u, lane); shr(e, sgn, sgn, 31); imadi(e, R(sgn), R(sgn), 0x80000000u, NOREG);
    andi(e, R(ropel), R(lane), 15u); imadi(e, R(ropel), R(ropel), 16u, NOREG);

    /* loads one token's rope row (position pos0 + tokc) with the sign on sin */
    #define ROPE_LOAD() do { iadd(e, t2, pos0, tokc); imadi(e, R(t2), R(t2), OMEGA_DATTN_ROPE_ROW_BYTES_PA, ropel); imadw(e, rope64, t2, 1u, prope); \
        ldg_wide(e, BW_IR_LDG_E_128, cos4, rope64, 0, udesc, 0); ldg_wide(e, BW_IR_LDG_E_128, sin4, rope64, 256, udesc, 0); } while (0)
    #define OMEGA_DATTN_ROPE_ROW_BYTES_PA 512u
    /* tokc = min(item + base, lim) for a token index t1 = item + base: tokc = t1 + over * (lim - t1) */
    #define CLAMP_TOK(tok, lim) do { int _d = V(e), _o = V(e); if (e->err) return -1; \
        imadi(e, R(_d), R(tok), 0xffffffffu, lim); shr(e, _o, _d, 31); imadr(e, tokc, _o, _d, tok); } while (0)

    /* ---- K/V append: chunk tokens 0 .. nk-1 (every CTA, identical bits); APPEND mode: only cxw .. nk-1 (one per warp);
     *      ATTEND mode: none (the APPEND launch wrote them) ---- */
    const bool APPEND = spec->mode == OMEGA_PATTN_MODE_APPEND;
    if (spec->mode != OMEGA_PATTN_MODE_ATTEND) {
        int kw4 = V128(e), k4 = V128(e), v4 = V128(e), nkm1 = V(e);
        if (e->err) return -1;
        imadw(e, ad64, lane, 16u, pkn);
        ldg_wide(e, BW_IR_LDG_E_128, kw4, ad64, 0, udesc, 1);
        imadi(e, R(nkm1), R(one), 0xffffffffu, nk);
        if (APPEND) { movi(e, R(cnt), 1u); iadd(e, item, warp, cxw); }
        else { imadi(e, R(cnt), R(one), W - 1u, nk); shr(e, cnt, cnt, LW);   /* ceil(nk / W) >= 1 */
               mov(e, R(item), R(warp)); }
        const int L_k = here(e);
        loop_begin(e);
        CLAMP_TOK(item, nkm1);
        imadi(e, R(t1), R(tokc), OMEGA_PATTN_QKV_WORDS * 4u, NOREG); imadi(e, R(t1), R(cy), 512u, t1); imadi(e, R(t1), R(lane), 16u, t1);
        imadi(e, R(t1), R(one), 32u * 128u * 4u, t1);
        imadw(e, ad64, t1, 1u, pqkv);
        ldg_wide(e, BW_IR_LDG_E_128, k4, ad64, 0, udesc, 1);
        ldg_wide(e, BW_IR_LDG_E_128, v4, ad64, 4u * 128u * 4u, udesc, 0);
        ROPE_LOAD();
        for (unsigned i = 0; i < 4; i++) xorr(e, S(sin4, i), S(sin4, i), R(sgn));
        norm_rope(e, &n, k4, kw4, cos4, sin4, MUT != 1);
        bf16_pair(e, R(w0), k4, 0, c7fff, h0, h1, tb); bf16_pair(e, R(w1), k4, 2, c7fff, h0, h1, tb);
        bf16_pair(e, R(w2), v4, 0, c7fff, h0, h1, tb); bf16_pair(e, R(w3), v4, 2, c7fff, h0, h1, tb);
        if (MUT != 3) {
            int ka = V64(e); if (e->err) return -1;
            isetp_ge_u32(e, item, nk);
            const int b = bra_if_p0_fwd(e);
            iadd(e, t1, pos0, item); imadi(e, R(t1), R(t1), 256u, cyk); imadi(e, R(t1), R(lane), 8u, t1);
            imadw(e, ka, t1, 1u, pkc); stg(e, ka, 0, R(w0), udesc); stg(e, ka, 4, R(w1), udesc);
            imadw(e, ka, t1, 1u, pvc); stg(e, ka, 0, R(w2), udesc); stg(e, ka, 4, R(w3), udesc);
            if (spec->mirror) {   /* the same words into the mirror planes */
                int pk2 = V64(e), pv2 = V64(e), kvs2 = V(e); if (e->err) return -1;
                ldc64(e, pk2, OMEGA_PATTN_ARG_KC2); ldc64(e, pv2, OMEGA_PATTN_ARG_VC2); ldc32(e, kvs2, OMEGA_PATTN_ARG_KV_STRIDE2);
                imadr(e, t2, cy, kvs2, NOREG);
                iadd(e, t1, pos0, item); imadi(e, R(t1), R(t1), 256u, t2); imadi(e, R(t1), R(lane), 8u, t1);
                imadw(e, ka, t1, 1u, pk2); stg(e, ka, 0, R(w0), udesc); stg(e, ka, 4, R(w1), udesc);
                imadw(e, ka, t1, 1u, pv2); stg(e, ka, 0, R(w2), udesc); stg(e, ka, 4, R(w3), udesc);
            }
            patch_bra(e, b, here(e));
        }
        imadi(e, R(item), R(one), W, item);
        imadi(e, R(cnt), R(one), 0xffffffffu, cnt);
        isetp_ge_u32(e, cnt, one);
        bra_if_p0(e, L_k);
        loop_end(e);
        membar(e);
        if (APPEND) {
            (void)h0;
            tail(e);
            if (e->err) return -1;
            return omega_bw_regalloc_solve(prog) != 0 ? -1 : 0;
        }
        bar_sync(e);
    }

    /* ---- Q fragments: rounds of 2 tiles (4 tokens) through shared memory ---- */
    int q[8];
    for (unsigned s = 0; s < 8; s++) q[s] = V128(e);
    {
        int qw4 = V128(e), x4 = V128(e), qlane = V(e), rr = V(e), cr = V(e), mcm1 = V(e), qbase = V(e), jt = V(e), hh = V(e);
        if (e->err) return -1;
        for (unsigned s = 0; s < 8; s++) for (unsigned k = 0; k < 4; k++) movrz(e, S(q[s], k)); /* defined before the rounds loop */
        imadw(e, ad64, lane, 16u, pqn);
        ldg_wide(e, BW_IR_LDG_E_128, qw4, ad64, 0, udesc, 1);
        imadi(e, R(mcm1), R(one), 0xffffffffu, mc);
        /* lane part of the fragment address: (lane>>2)*512 + (lane&1)*32 + ((lane>>1)&1)*8 */
        shr(e, t1, lane, 2); imadi(e, R(qlane), R(t1), 512u, NOREG);
        andi(e, R(t1), R(lane), 1u); imadi(e, R(qlane), R(t1), 32u, qlane);
        shr(e, t1, lane, 1); andi(e, R(t1), R(t1), 1u); imadi(e, R(qlane), R(t1), 8u, qlane);
        /* this warp's fragment base: tile (warp>>1)&1 of the round */
        shr(e, t1, warp, 1); andi(e, R(t1), R(t1), 1u); imadi(e, R(qbase), R(t1), 4096u, NOREG); imadi(e, R(qbase), R(lane), 16u, qbase);
        movrz(e, R(rr)); movi(e, R(cr), ROUNDS);
        const int L_r = here(e);
        loop_begin(e);
        mov(e, R(item), R(warp)); movi(e, R(cnt), 32u / W);
        const int L_i = here(e);
        loop_begin(e);
        shr(e, jt, item, 3); andi(e, R(hh), R(item), 7u);
        imadi(e, R(t1), R(rr), 4u, jt); iadd(e, t1, t1, cxw);          /* chunk token */
        { int tk = V(e); if (e->err) return -1; mov(e, R(tk), R(t1)); CLAMP_TOK(tk, mcm1); }
        imadi(e, R(t1), R(cy), 8u, hh); imadi(e, R(t1), R(t1), 512u, NOREG);
        imadi(e, R(t1), R(tokc), OMEGA_PATTN_QKV_WORDS * 4u, t1); imadi(e, R(t1), R(lane), 16u, t1);
        imadw(e, ad64, t1, 1u, pqkv);
        ldg_wide(e, BW_IR_LDG_E_128, x4, ad64, 0, udesc, 1);
        ROPE_LOAD();
        for (unsigned i = 0; i < 4; i++) xorr(e, S(sin4, i), S(sin4, i), R(sgn));
        norm_rope(e, &n, x4, qw4, cos4, sin4, MUT != 1);
        for (unsigned i = 0; i < 4; i++) fmul(e, S(x4, i), S(x4, i), R(qks));
        bf16_pair(e, R(w0), x4, 0, c7fff, h0, h1, tb); bf16_pair(e, R(w1), x4, 2, c7fff, h0, h1, tb);
        /* dst = qlane + (jt>>1)*4096 + h*64 + (jt&1)*4 */
        shr(e, t1, jt, 1); imadi(e, R(t1), R(t1), 4096u, qlane); imadi(e, R(t1), R(hh), 64u, t1);
        andi(e, R(t2), R(jt), 1u); imadi(e, R(t1), R(t2), 4u, t1);
        sts32(e, t1, R(w0)); imadi(e, R(t2), R(one), 16u, t1); sts32(e, t2, R(w1));
        imadi(e, R(item), R(one), W, item);
        imadi(e, R(cnt), R(one), 0xffffffffu, cnt);
        isetp_ge_u32(e, cnt, one);
        bra_if_p0(e, L_i);
        loop_end(e);
        bar_sync(e);
        /* the round's warps (warp>>2 == rr; every warp when W = 4) load their 8 fragments */
        int skip = -1;
        if (W == 8u) { shr(e, t1, warp, 2); xorr(e, R(t1), R(t1), R(rr)); isetp_ge_u32(e, t1, one); skip = bra_if_p0_fwd(e); }
        for (unsigned s = 0; s < 8; s++) lds128(e, q[s], qbase, s * 512u, s == 0);
        if (skip >= 0) { nop(e, 1); patch_bra(e, skip, here(e)); }
        bar_sync(e);
        imadi(e, R(rr), R(one), 1u, rr);
        imadi(e, R(cr), R(one), 0xffffffffu, cr);
        isetp_ge_u32(e, cr, one);
        bra_if_p0(e, L_r);
        loop_end(e);
    }
    #undef ROPE_LOAD
    #undef CLAMP_TOK

    /* ---- main loop state ---- */
    int o[8], sc[4], kf[2][2], pp[2], vf[4];
    for (unsigned i = 0; i < 8; i++) o[i] = V128(e);
    int mA = V(e), mB = V(e), lA = V(e), lB = V(e), dqa = V(e), gK = V64(e), gV = V64(e), sKb = V(e), sVb = V(e);
    int lane16 = V(e), vfb = V(e), negc = V(e), tokA = V(e);
    if (e->err) return -1;
    {
        int a = V(e), c = V(e), tq = V(e); if (e->err) return -1;
        shr(e, t1, warp, 1); imadi(e, R(tokA), R(t1), 2u, cxw);        /* the tile's first chunk token */
        iadd(e, dqa, pos0, tokA);                                    /* row-A position - kb (kb = 0) */
        /* blocks: kend = pos0 + nk - 1; nblk = kend / 32 + 1 */
        iadd(e, t1, pos0, nk); imadi(e, R(t1), R(one), 0xffffffffu, t1); shr(e, t1, t1, 5); imadi(e, R(cnt), R(one), 1u, t1);
        shr(e, a, tid, 4); andi(e, R(c), R(tid), 15u);
        /* staging pointers */
        imadw(e, gK, cyk, 1u, pkc); imadw(e, gK, a, 256u, gK); imadi(e, R(t1), R(c), 16u, NOREG); imadw(e, gK, t1, 1u, gK);
        imadw(e, gV, cyk, 1u, pvc); imadw(e, gV, a, 512u, gV); imadw(e, gV, t1, 1u, gV);
        /* sKb = (c>>1)*1024 + (a&7)*64 + ((a>>3)&1)*8 + (c&1)*4 */
        shr(e, t1, c, 1); imadi(e, R(sKb), R(t1), 1024u, NOREG);
        andi(e, R(t1), R(a), 7u); imadi(e, R(sKb), R(t1), 64u, sKb);
        shr(e, t1, a, 3); andi(e, R(t1), R(t1), 1u); imadi(e, R(sKb), R(t1), 8u, sKb);
        andi(e, R(t2), R(c), 1u); imadi(e, R(sKb), R(t2), 4u, sKb);
        /* sVb = SM_V + ((a>>3)*8 + (c>>1))*512 + (a&3)*16 + ((c&1)*2 + ((a>>2)&1))*4 */
        shr(e, t1, a, 3); shr(e, tq, c, 1); imadi(e, R(t1), R(t1), 8u, tq); imadi(e, R(sVb), R(t1), 512u, NOREG);
        andi(e, R(t1), R(a), 3u); imadi(e, R(sVb), R(t1), 16u, sVb);
        shr(e, t1, a, 2); andi(e, R(t1), R(t1), 1u); imadi(e, R(t1), R(t2), 2u, t1); imadi(e, R(sVb), R(t1), 4u, sVb);
        imadi(e, R(sVb), R(one), SM_V, sVb);
        imadi(e, R(lane16), R(lane), 16u, NOREG);
        andi(e, R(t1), R(warp), 1u); imadi(e, R(vfb), R(t1), 2048u, lane16); imadi(e, R(vfb), R(one), SM_V, vfb);
    }
    for (unsigned i = 0; i < 8; i++) for (unsigned k = 0; k < 4; k++) movrz(e, S(o[i], k));
    movi(e, R(negc), NEG_BIG); mov(e, R(mA), R(negc)); mov(e, R(mB), R(negc)); movrz(e, R(lA)); movrz(e, R(lB));

    int kq4[4], va4[2], vb4[2];
    for (unsigned i = 0; i < NKU; i++) kq4[i] = V128(e);
    for (unsigned i = 0; i < NVU; i++) { va4[i] = V128(e); vb4[i] = V128(e); }
    int ev = V(e), od = V(e), sa = V(e), mxA = V(e), mxB = V(e), alA = V(e), alB = V(e), sh = V(e);
    for (unsigned j = 0; j < 4; j++) sc[j] = V128(e);
    for (unsigned b = 0; b < 2; b++) { kf[b][0] = V128(e); kf[b][1] = V128(e); pp[b] = V128(e); }
    for (unsigned i = 0; i < 4; i++) vf[i] = V128(e);
    if (e->err) return -1;

    const int L_m = here(e);
    loop_begin(e);
    bar_sync(e);                                     /* every warp is done with the previous block (or the prologue) */
    /* ---- stage the block: K units (key a + i*2W, 16-byte chunk c), V units (key pair a + i*2W) ---- */
    for (unsigned i = 0; i < NKU; i++) ldg_wide(e, BW_IR_LDG_E_128, kq4[i], gK, i * 2u * W * 256u, udesc, i == 0);
    if (NKU * 4u + NVU * 8u > 16u) {                  /* W = 4: K first, then V (register budget) */
        for (unsigned i = 0; i < NKU; i++) for (unsigned k = 0; k < 4; k++) {
            uint32_t ci = W == 4u ? (i >> 1) * 512u + (i & 1u) * 8u : i * 512u;
            imadi(e, R(sa), R(one), ci + 16u * k, sKb); sts32(e, sa, S(kq4[i], k));
        }
    }
    for (unsigned i = 0; i < NVU; i++) {
        ldg_wide(e, BW_IR_LDG_E_128, va4[i], gV, i * 2u * W * 512u, udesc, NKU * 4u + NVU * 8u > 16u && i == 0);
        ldg_wide(e, BW_IR_LDG_E_128, vb4[i], gV, i * 2u * W * 512u + 256u, udesc, 0);
    }
    if (NKU * 4u + NVU * 8u <= 16u) {
        for (unsigned i = 0; i < NKU; i++) for (unsigned k = 0; k < 4; k++) {
            uint32_t ci = W == 4u ? (i >> 1) * 512u + (i & 1u) * 8u : i * 512u;
            imadi(e, R(sa), R(one), ci + 16u * k, sKb); sts32(e, sa, S(kq4[i], k));
        }
    }
    for (unsigned i = 0; i < NVU; i++) for (unsigned k = 0; k < 4; k++) {
        andi(e, R(t1), S(va4[i], k), 0xffffu); imadi(e, R(ev), S(vb4[i], k), 65536u, t1);
        shrs(e, t1, S(va4[i], k), 16); andi(e, R(t2), S(vb4[i], k), 0xffff0000u); iadd(e, od, t1, t2);
        imadi(e, R(sa), R(one), i * 4096u + (2u * k) * 64u, sVb); sts32(e, sa, R(ev));
        imadi(e, R(sa), R(one), i * 4096u + (2u * k + 1u) * 64u, sVb); sts32(e, sa, R(od));
    }
    bar_sync(e);

    /* ---- S = Q K^T: 8 k-steps x 4 n-tiles, K fragments double-buffered on scoreboards 3/4 ---- */
    lds_pf(e, kf[0][0], lane16, SM_K + 0u, 3); lds_pf(e, kf[0][1], lane16, SM_K + 512u, 3);
    consume(e, 3);
    for (unsigned s = 0; s < 8; s++) {
        const unsigned cur = s & 1u, nx = cur ^ 1u, sbn = 3u + nx;
        if (s < 7) { lds_pf(e, kf[nx][0], lane16, SM_K + (s + 1u) * 1024u, sbn); lds_pf(e, kf[nx][1], lane16, SM_K + (s + 1u) * 1024u + 512u, sbn); }
        hmma(e, sc[0], q[s], kf[cur][0], 0, s ? sc[0] : NOREG, 8);
        hmma(e, sc[1], q[s], kf[cur][0], 2, s ? sc[1] : NOREG, 8);
        hmma(e, sc[2], q[s], kf[cur][1], 0, s ? sc[2] : NOREG, 8);
        hmma(e, sc[3], q[s], kf[cur][1], 2, s ? sc[3] : NOREG, 8);
        if (s < 7) consume(e, sbn);
    }
    nop(e, 15); nop(e, 14);

    /* ---- causal mask: key kb + 8j + 2t + (e&1) > row position (row A for e < 2, A + 1 for e >= 2) ---- */
    if (MUT != 2) {
        imadi(e, R(t1), R(one), 0xffffffe1u, dqa); shr(e, t1, t1, 31);      /* 1 when kb + 31 > row-A position */
        isetp_ge_u32(e, zero, t1);
        const int b = bra_if_p0_fwd(e);
        int dql = V(e); if (e->err) return -1;
        shr(e, t2, lane16, 3); andi(e, R(t2), R(t2), 6u); imadi(e, R(dql), R(t2), 0xffffffffu, dqa); /* dqa - 2t */
        for (unsigned j = 0; j < 4; j++) for (unsigned k = 0; k < 4; k++) {
            int32_t cst = (k >= 2 ? 1 : 0) - 8 * (int32_t)j - (int32_t)(k & 1u);
            imadi(e, R(t1), R(one), (uint32_t)cst, dql); shr(e, t1, t1, 31); imadi(e, R(t1), R(t1), NEG_BIG, NOREG);
            fadd(e, S(sc[j], k), S(sc[j], k), R(t1));
        }
        patch_bra(e, b, here(e));
    }

    /* ---- online softmax (log2 units) ---- */
    fmaxx(e, R(mxA), S(sc[0], 0), S(sc[0], 1)); fmaxx(e, R(mxB), S(sc[0], 2), S(sc[0], 3));
    for (unsigned j = 1; j < 4; j++) {
        fmaxx(e, R(mxA), R(mxA), S(sc[j], 0)); fmaxx(e, R(mxA), R(mxA), S(sc[j], 1));
        fmaxx(e, R(mxB), R(mxB), S(sc[j], 2)); fmaxx(e, R(mxB), R(mxB), S(sc[j], 3));
    }
    shfl_bfly(e, sh, mxA, 1); fmaxx(e, R(mxA), R(mxA), R(sh)); shfl_bfly(e, sh, mxA, 2); fmaxx(e, R(mxA), R(mxA), R(sh));
    shfl_bfly(e, sh, mxB, 1); fmaxx(e, R(mxB), R(mxB), R(sh)); shfl_bfly(e, sh, mxB, 2); fmaxx(e, R(mxB), R(mxB), R(sh));
    fmaxx(e, R(mxA), R(mxA), R(mA)); fmaxx(e, R(mxB), R(mxB), R(mB));
    fsub(e, R(alA), R(mA), R(mxA)); fsub(e, R(alB), R(mB), R(mxB));
    mov(e, R(mA), R(mxA)); mov(e, R(mB), R(mxB));
    for (unsigned j = 0; j < 4; j++) for (unsigned k = 0; k < 4; k++) fsub(e, S(sc[j], k), S(sc[j], k), R(k < 2 ? mA : mB));
    mufu(e, BW_IR_MUFU_EX2, R(alA), R(alA), 1); mufu(e, BW_IR_MUFU_EX2, R(alB), R(alB), 0);
    for (unsigned j = 0; j < 4; j++) for (unsigned k = 0; k < 4; k++) mufu(e, BW_IR_MUFU_EX2, S(sc[j], k), S(sc[j], k), 0);
    fmul(e, R(lA), R(lA), R(alA)); fmul(e, R(lB), R(lB), R(alB));
    for (unsigned j = 0; j < 4; j++) for (unsigned k = 0; k < 4; k++) fadd(e, R(k < 2 ? lA : lB), R(k < 2 ? lA : lB), S(sc[j], k));
    if (MUT != 4) for (unsigned i = 0; i < 8; i++) for (unsigned k = 0; k < 4; k++) fmul(e, S(o[i], k), S(o[i], k), R(k < 2 ? alA : alB));
    for (unsigned b = 0; b < 2; b++) {
        f2fp(e, S(pp[b], 0), S(sc[2 * b], 1), S(sc[2 * b], 0)); f2fp(e, S(pp[b], 1), S(sc[2 * b], 3), S(sc[2 * b], 2));
        f2fp(e, S(pp[b], 2), S(sc[2 * b + 1], 1), S(sc[2 * b + 1], 0)); f2fp(e, S(pp[b], 3), S(sc[2 * b + 1], 3), S(sc[2 * b + 1], 2));
    }

    /* ---- O += P V: 2 k-steps (16 keys) x 8 n-tiles of this warp's 64 dims ---- */
    for (unsigned kq = 0; kq < 2; kq++) {
        for (unsigned np = 0; np < 4; np++) lds128(e, vf[np], vfb, kq * 4096u + np * 512u, np == 0);
        for (unsigned np = 0; np < 4; np++) { hmma(e, o[2 * np], pp[kq], vf[np], 0, o[2 * np], 8); hmma(e, o[2 * np + 1], pp[kq], vf[np], 2, o[2 * np + 1], 8); }
    }
    imadw(e, gK, one, 8192u, gK); imadw(e, gV, one, 8192u, gV);
    imadi(e, R(dqa), R(one), 0xffffffe0u, dqa);
    imadi(e, R(cnt), R(one), 0xffffffffu, cnt);
    isetp_ge_u32(e, cnt, one);
    bra_if_p0(e, L_m);
    loop_end(e);

    /* ---- epilogue ---- */
    nop(e, 15); nop(e, 14);
    {
        int pout = V64(e), osc = V(e), po = V64(e), g = V(e), tq = V(e), etid = V(e), ecx = V(e), ecy = V(e), emc = V(e), ewarp = V(e), elane = V(e), etok = V(e);
        if (e->err) return -1;
        ldc64(e, pout, OMEGA_PATTN_ARG_OUT); ldc32(e, osc, OMEGA_PATTN_ARG_OUT_SCALE); ldc32(e, emc, OMEGA_PATTN_ARG_MC);
        s2r(e, etid, BW_SR_TID_X); s2r(e, ecx, BW_SR_CTAID_X); s2r(e, ecy, BW_SR_CTAID_Y);
        shr(e, ewarp, etid, 5); andi(e, R(elane), R(etid), 31u);
        shr(e, t1, ewarp, 1); imadi(e, R(t1), R(t1), 2u, NOREG); imadi(e, R(etok), R(ecx), W, t1);
        shfl_bfly(e, sh, lA, 1); fadd(e, R(lA), R(lA), R(sh)); shfl_bfly(e, sh, lA, 2); fadd(e, R(lA), R(lA), R(sh));
        shfl_bfly(e, sh, lB, 1); fadd(e, R(lB), R(lB), R(sh)); shfl_bfly(e, sh, lB, 2); fadd(e, R(lB), R(lB), R(sh));
        mufu(e, BW_IR_MUFU_RCP, R(lA), R(lA), 1); mufu(e, BW_IR_MUFU_RCP, R(lB), R(lB), 0);
        fmul(e, R(lA), R(lA), R(osc)); fmul(e, R(lB), R(lB), R(osc));
        for (unsigned i = 0; i < 8; i++) for (unsigned k = 0; k < 4; k++) fmul(e, S(o[i], k), S(o[i], k), R(k < 2 ? lA : lB));
        /* out + ((tokA*32 + cy*8 + g)*128 + (warp&1)*64 + 2t)*4 */
        shr(e, g, elane, 2); andi(e, R(tq), R(elane), 3u);
        imadi(e, R(t1), R(etok), 32u, NOREG); imadi(e, R(t1), R(ecy), 8u, t1); iadd(e, t1, t1, g); imadi(e, R(t1), R(t1), 512u, NOREG);
        andi(e, R(t2), R(ewarp), 1u); imadi(e, R(t1), R(t2), 256u, t1); imadi(e, R(t1), R(tq), 8u, t1);
        imadw(e, po, t1, 1u, pout);
        for (unsigned row = 0; row < 2; row++) {
            imadi(e, R(t1), R(one), row, etok);
            isetp_ge_u32(e, t1, emc);
            const int b = bra_if_p0_fwd(e);
            for (unsigned i = 0; i < 8; i++) { stg(e, po, row * 16384u + i * 32u, S(o[i], 2 * row), udesc); stg(e, po, row * 16384u + i * 32u + 4u, S(o[i], 2 * row + 1), udesc); }
            patch_bra(e, b, here(e));
        }
    }
    (void)iaddo; (void)h0;
    tail(e);
    if (e->err) return -1;
    if (omega_bw_regalloc_solve(prog) != 0) return -1;
    return 0;
}

int omega_prefill_attn_prog_stats(const OmegaPrefillAttnSpec *spec, uint32_t *insns, uint32_t *gprs, uint32_t *placeholders) {
    static BlackwellIRProgram p;
    if (omega_blackwell_codegen_prefill_attn_prog(spec, &p) != 0) return -1;
    if (insns) *insns = (uint32_t)p.count;
    if (gprs) *gprs = p.regalloc.peak_gpr_usage;
    if (placeholders) {
        uint32_t n = 0;
        for (size_t i = 0; i < p.count; i++) {
            const BlackwellIRInsn *in = &p.insns[i];
            if ((in->op == BW_IR_SHFL_DOWN && (in->imm & OMEGA_TC_MARK_BFLY)) || (in->op == BW_IR_FMNMX_MAX && in->imm == OMEGA_TC_MARK_F2FP)) n++;
        }
        *placeholders = n;
    }
    return 0;
}
