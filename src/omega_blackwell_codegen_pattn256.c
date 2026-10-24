/*
 * omega_blackwell_codegen_pattn256.c: causal prefill attention for Qwen3.8-27B on tensor cores, the attend launch
 * (see the header). The emitter helpers are those of omega_blackwell_codegen_pattn.c, copied unchanged (control words,
 * the read barrier on every load, held scoreboards 3/4 for prefetch loads, loop live ranges, HMMA spacing: >= 4
 * independent HMMAs at stall 8 between dependent ones, stall 15 + NOP 14 before a non-HMMA reader of an HMMA result).
 */
#include "omega_blackwell_codegen_pattn256.h"
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
static void fop(Em *e, BlackwellIROpcode op, Op d, Op a, Op b, Op c) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s;
    i.src2_vreg = b.v; i.src2_subreg = (uint8_t)b.s; i.src3_vreg = c.v; i.src3_subreg = (uint8_t)c.s; emit(e, i, K_FIXED);
}
static void iadd(Em *e, int d, int a, int b) { fop(e, BW_IR_IADD3, R(d), R(a), R(b), R(NOREG)); }
static void fadd(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FADD, d, a, b, R(NOREG)); }
static void fsub(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FSUB, d, a, b, R(NOREG)); }
static void fmul(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FMUL, d, a, b, R(NOREG)); }
static void fmaxx(Em *e, Op d, Op a, Op b) { fop(e, BW_IR_FMNMX_MAX, d, a, b, R(NOREG)); }
static void mufu(Em *e, BlackwellIROpcode op, Op d, Op a, int first) { BlackwellIRInsn i = I0(op); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void ldg_wide(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, int first) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE);
}
static void stg(Em *e, int addr64, uint32_t imm, Op val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.imm = imm; i.src2_vreg = val.v; i.src2_subreg = (uint8_t)val.s; i.ureg = udesc; emit(e, i, K_STORE); }
static void lds128(Em *e, int d128, int addr, uint32_t imm, int first) { BlackwellIRInsn i = I0(BW_IR_LDS128); i.dst_vreg = d128; i.src1_vreg = addr; i.imm = imm; emit(e, i, first ? K_VAR : K_VAR_MORE); }
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


/* shared memory map */
#define SM_K 0u            /* K block, [s 16][lane 32][16 B]: lane (g, t) holds b0, b1 of key g and of key 8 + g */
#define SM_V 8192u         /* V block, [np 16][lane 32][16 B]: b0, b1 of dim tiles 2np and 2np + 1 */
static const uint32_t NEG_BIG = 0xF149F2CAu; /* -1e30f */

int omega_pattn256_spec_ok(const OmegaPrefillAttn256Spec *s) { return s && s->mutant >= 0 && s->mutant <= 2; }

int omega_blackwell_codegen_prefill_attn256_prog(const OmegaPrefillAttn256Spec *spec, BlackwellIRProgram *prog) {
    if (!omega_pattn256_spec_ok(spec) || !prog) return -1;
    omega_bw_ir_init(prog);
    if (omega_bw_ir_set_gpr_limit(prog, BW_PHYS_GPR_HW_LIMIT) != 0) return -1;
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    const int MUT = spec->mutant;

    /* ---- arguments and identities ---- */
    int udesc = UV64(e);
    int pqf = V64(e), pkc = V64(e), pvc = V64(e);
    int pos0 = V(e), mc = V(e), kvs = V(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), one = V(e), zero = V(e);
    int cyk = V(e), nk = V(e), t1 = V(e), t2 = V(e), cxw = V(e), cnt = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pqf, OMEGA_PATTN256_ARG_QF); ldc64(e, pkc, OMEGA_PATTN256_ARG_KC); ldc64(e, pvc, OMEGA_PATTN256_ARG_VC);
    ldc32(e, pos0, OMEGA_PATTN256_ARG_POS0); ldc32(e, mc, OMEGA_PATTN256_ARG_MC); ldc32(e, kvs, OMEGA_PATTN256_ARG_KV_STRIDE);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, R(one), 1); movrz(e, R(zero));
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);
    imadr(e, cyk, cy, kvs, NOREG);                       /* byte offset of this KV head's plane */
    imadi(e, R(cxw), R(cx), OMEGA_PATTN256_TOKENS, NOREG); /* first chunk token of the CTA */
    /* nk = min(cxw + 8, mc) = hi + over * (mc - hi), over = 1 when hi > mc */
    imadi(e, R(t2), R(one), OMEGA_PATTN256_TOKENS, cxw);   /* hi */
    imadi(e, R(t1), R(t2), 0xffffffffu, mc);             /* mc - hi */
    { int ov = V(e); if (e->err) return -1;
      shr(e, ov, t1, 31); imadr(e, nk, ov, t1, t2); }

    /* ---- Q: the tile's 16 A fragments straight from QF (16 LDG.128 from one address) ---- */
    int q[16];
    for (unsigned s = 0; s < 16; s++) q[s] = V128(e);
    int tokA = V(e), dqa = V(e);
    if (e->err) return -1;
    {
        int g = V(e), hg = V(e), tp = V(e), tpm = V(e), c5 = V(e), ad64 = V64(e); if (e->err) return -1;
        shr(e, g, lane, 2);
        movi(e, R(c5), 5u);
        imadi(e, R(t1), R(g), 0xffffffffu, c5);          /* 5 - g */
        shr(e, t2, t1, 31);                              /* 1 when g > 5 */
        imadr(e, hg, t2, t1, g);                         /* min(g, 5) */
        shr(e, t1, warp, 1);
        imadi(e, R(tokA), R(t1), 2u, cxw);               /* the tile's first chunk token */
        shr(e, tp, tokA, 1);
        /* token pair clamped to the last written one: tpm = (mc - 1) >> 1; tp = min(tp, tpm) */
        imadi(e, R(tpm), R(one), 0xffffffffu, mc); shr(e, tpm, tpm, 1);
        imadi(e, R(t1), R(tp), 0xffffffffu, tpm);        /* tpm - tp */
        shr(e, t2, t1, 31); imadr(e, tp, t2, t1, tp);
        imadi(e, R(t1), R(tp), 4u, cy); imadi(e, R(t1), R(t1), 6u, hg); imadi(e, R(t1), R(t1), 1024u, NOREG);
        andi(e, R(t2), R(lane), 3u); imadi(e, R(t1), R(t2), 16u, t1);
        imadw(e, ad64, t1, 1u, pqf);
        for (unsigned s = 0; s < 16; s++) ldg_wide(e, BW_IR_LDG_E_128, q[s], ad64, s * 64u, udesc, s == 0);
        iadd(e, dqa, pos0, tokA);                        /* row-A position - kb (kb = 0) */
    }

    /* ---- main loop state ---- */
    int o[16], sc[4], kf[2], pp, vf[4];
    for (unsigned i = 0; i < 16; i++) o[i] = V128(e);
    int mA = V(e), mB = V(e), lA = V(e), lB = V(e), gK = V64(e), gV = V64(e), sKb = V(e), sVb = V(e);
    int lane16 = V(e), vfb = V(e), negc = V(e);
    if (e->err) return -1;
    {
        int ka = V(e), c = V(e), tq = V(e); if (e->err) return -1;
        /* blocks: kend = pos0 + nk - 1; nblk = kend / 16 + 1 */
        iadd(e, t1, pos0, nk); imadi(e, R(t1), R(one), 0xffffffffu, t1); shr(e, t1, t1, 4); imadi(e, R(cnt), R(one), 1u, t1);
        shr(e, ka, tid, 5); andi(e, R(c), R(tid), 31u);
        /* staging pointers: K units (key ka + 8i, chunk c), V unit (keys 2ka, 2ka + 1, chunk c) */
        imadi(e, R(t1), R(c), 16u, NOREG);
        imadw(e, gK, cyk, 1u, pkc); imadw(e, gK, ka, 512u, gK); imadw(e, gK, t1, 1u, gK);
        imadw(e, gV, cyk, 1u, pvc); imadw(e, gV, ka, 1024u, gV); imadw(e, gV, t1, 1u, gV);
        /* sKb = (c>>1)*512 + ka*64 + (c&1)*4 (+ t'*16 for word t', + i*8 for key ka + 8i) */
        shr(e, t1, c, 1); imadi(e, R(sKb), R(t1), 512u, NOREG);
        imadi(e, R(sKb), R(ka), 64u, sKb);
        andi(e, R(t2), R(c), 1u); imadi(e, R(sKb), R(t2), 4u, sKb);
        /* sVb = SM_V + (c>>1)*512 + (ka&3)*16 + ((c&1)*2 + (ka>>2))*4 (+ j*64 for dim 8c + j) */
        shr(e, t1, c, 1); imadi(e, R(sVb), R(t1), 512u, NOREG);
        andi(e, R(t1), R(ka), 3u); imadi(e, R(sVb), R(t1), 16u, sVb);
        shr(e, tq, ka, 2); imadi(e, R(t1), R(t2), 2u, tq); imadi(e, R(sVb), R(t1), 4u, sVb);
        imadi(e, R(sVb), R(one), SM_V, sVb);
        imadi(e, R(lane16), R(lane), 16u, NOREG);
        andi(e, R(t1), R(warp), 1u); imadi(e, R(vfb), R(t1), 4096u, lane16); imadi(e, R(vfb), R(one), SM_V, vfb);
    }
    for (unsigned i = 0; i < 16; i++) for (unsigned k = 0; k < 4; k++) movrz(e, S(o[i], k));
    movi(e, R(negc), NEG_BIG); mov(e, R(mA), R(negc)); mov(e, R(mB), R(negc)); movrz(e, R(lA)); movrz(e, R(lB));

    int kq4[2], va4 = V128(e), vb4 = V128(e);
    for (unsigned i = 0; i < 2; i++) kq4[i] = V128(e);
    int ev = V(e), od = V(e), sa = V(e), mxA = V(e), mxB = V(e), alA = V(e), alB = V(e), sh = V(e);
    for (unsigned j = 0; j < 4; j++) sc[j] = V128(e);
    kf[0] = V128(e); kf[1] = V128(e); pp = V128(e);
    for (unsigned i = 0; i < 4; i++) vf[i] = V128(e);
    if (e->err) return -1;

    const int L_m = here(e);
    loop_begin(e);
    bar_sync(e);                                     /* every warp is done with the previous block */
    /* ---- stage the block ---- */
    ldg_wide(e, BW_IR_LDG_E_128, kq4[0], gK, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_128, kq4[1], gK, 8u * 512u, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, va4, gV, 0, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, vb4, gV, 512u, udesc, 0);
    for (unsigned i = 0; i < 2; i++) for (unsigned k = 0; k < 4; k++) { imadi(e, R(sa), R(one), i * 8u + 16u * k, sKb); sts32(e, sa, S(kq4[i], k)); }
    for (unsigned k = 0; k < 4; k++) {
        andi(e, R(t1), S(va4, k), 0xffffu); imadi(e, R(ev), S(vb4, k), 65536u, t1);
        shrs(e, t1, S(va4, k), 16); andi(e, R(t2), S(vb4, k), 0xffff0000u); iadd(e, od, t1, t2);
        imadi(e, R(sa), R(one), (2u * k) * 64u, sVb); sts32(e, sa, R(ev));
        imadi(e, R(sa), R(one), (2u * k + 1u) * 64u, sVb); sts32(e, sa, R(od));
    }
    bar_sync(e);

    /* ---- S = Q K^T: 16 k-steps x 2 key tiles, even / odd k-steps into separate accumulators, K fragments
     *      double-buffered on scoreboards 3/4 ---- */
    lds_pf(e, kf[0], lane16, SM_K, 3);
    consume(e, 3);
    for (unsigned s = 0; s < 16; s++) {
        const unsigned cur = s & 1u, nx = cur ^ 1u, sbn = 3u + nx, a = (s & 1u) * 2u;
        if (s < 15) lds_pf(e, kf[nx], lane16, SM_K + (s + 1u) * 512u, sbn);
        hmma(e, sc[a], q[s], kf[cur], 0, s >= 2 ? sc[a] : NOREG, 8);
        hmma(e, sc[a + 1], q[s], kf[cur], 2, s >= 2 ? sc[a + 1] : NOREG, 8);
        if (s < 15) consume(e, sbn);
    }
    nop(e, 15); nop(e, 14);
    for (unsigned k = 0; k < 4; k++) { fadd(e, S(sc[0], k), S(sc[0], k), S(sc[2], k)); fadd(e, S(sc[1], k), S(sc[1], k), S(sc[3], k)); }

    /* ---- causal mask: key kb + 8j + 2t + (k&1) > row position (row A for k < 2, A + 1 for k >= 2) ---- */
    if (MUT != 1) {
        imadi(e, R(t1), R(one), 0xfffffff1u, dqa); shr(e, t1, t1, 31);      /* 1 when kb + 15 > row-A position */
        isetp_ge_u32(e, zero, t1);
        const int b = bra_if_p0_fwd(e);
        int dql = V(e); if (e->err) return -1;
        shr(e, t2, lane16, 3); andi(e, R(t2), R(t2), 6u); imadi(e, R(dql), R(t2), 0xffffffffu, dqa); /* dqa - 2t */
        for (unsigned j = 0; j < 2; j++) for (unsigned k = 0; k < 4; k++) {
            int32_t cst = (k >= 2 ? 1 : 0) - 8 * (int32_t)j - (int32_t)(k & 1u);
            imadi(e, R(t1), R(one), (uint32_t)cst, dql); shr(e, t1, t1, 31); imadi(e, R(t1), R(t1), NEG_BIG, NOREG);
            fadd(e, S(sc[j], k), S(sc[j], k), R(t1));
        }
        patch_bra(e, b, here(e));
    }

    /* ---- online softmax (log2 units) ---- */
    fmaxx(e, R(mxA), S(sc[0], 0), S(sc[0], 1)); fmaxx(e, R(mxB), S(sc[0], 2), S(sc[0], 3));
    fmaxx(e, R(mxA), R(mxA), S(sc[1], 0)); fmaxx(e, R(mxA), R(mxA), S(sc[1], 1));
    fmaxx(e, R(mxB), R(mxB), S(sc[1], 2)); fmaxx(e, R(mxB), R(mxB), S(sc[1], 3));
    shfl_bfly(e, sh, mxA, 1); fmaxx(e, R(mxA), R(mxA), R(sh)); shfl_bfly(e, sh, mxA, 2); fmaxx(e, R(mxA), R(mxA), R(sh));
    shfl_bfly(e, sh, mxB, 1); fmaxx(e, R(mxB), R(mxB), R(sh)); shfl_bfly(e, sh, mxB, 2); fmaxx(e, R(mxB), R(mxB), R(sh));
    fmaxx(e, R(mxA), R(mxA), R(mA)); fmaxx(e, R(mxB), R(mxB), R(mB));
    fsub(e, R(alA), R(mA), R(mxA)); fsub(e, R(alB), R(mB), R(mxB));
    mov(e, R(mA), R(mxA)); mov(e, R(mB), R(mxB));
    for (unsigned j = 0; j < 2; j++) for (unsigned k = 0; k < 4; k++) fsub(e, S(sc[j], k), S(sc[j], k), R(k < 2 ? mA : mB));
    mufu(e, BW_IR_MUFU_EX2, R(alA), R(alA), 1); mufu(e, BW_IR_MUFU_EX2, R(alB), R(alB), 0);
    for (unsigned j = 0; j < 2; j++) for (unsigned k = 0; k < 4; k++) mufu(e, BW_IR_MUFU_EX2, S(sc[j], k), S(sc[j], k), 0);
    fmul(e, R(lA), R(lA), R(alA)); fmul(e, R(lB), R(lB), R(alB));
    for (unsigned j = 0; j < 2; j++) for (unsigned k = 0; k < 4; k++) fadd(e, R(k < 2 ? lA : lB), R(k < 2 ? lA : lB), S(sc[j], k));
    if (MUT != 2) for (unsigned i = 0; i < 16; i++) for (unsigned k = 0; k < 4; k++) fmul(e, S(o[i], k), S(o[i], k), R(k < 2 ? alA : alB));
    f2fp(e, S(pp, 0), S(sc[0], 1), S(sc[0], 0)); f2fp(e, S(pp, 1), S(sc[0], 3), S(sc[0], 2));
    f2fp(e, S(pp, 2), S(sc[1], 1), S(sc[1], 0)); f2fp(e, S(pp, 3), S(sc[1], 3), S(sc[1], 2));

    /* ---- O += P V: one k-step (16 keys) x 16 dim tiles of this warp's 128 dims, in two rounds of 4 fragments ---- */
    for (unsigned r = 0; r < 2; r++) {
        for (unsigned np = 0; np < 4; np++) lds128(e, vf[np], vfb, (r * 4u + np) * 512u, np == 0);
        for (unsigned np = 0; np < 4; np++) {
            const unsigned n = 2u * (4u * r + np);
            hmma(e, o[n], pp, vf[np], 0, o[n], 8); hmma(e, o[n + 1], pp, vf[np], 2, o[n + 1], 8);
        }
    }
    imadw(e, gK, one, 16u * 512u, gK); imadw(e, gV, one, 16u * 512u, gV);
    imadi(e, R(dqa), R(one), 0xfffffff0u, dqa);
    imadi(e, R(cnt), R(one), 0xffffffffu, cnt);
    isetp_ge_u32(e, cnt, one);
    bra_if_p0(e, L_m);
    loop_end(e);

    /* ---- epilogue: rows g < 6 (a lane-divergent region), tokens < mc (warp-uniform) ---- */
    nop(e, 15); nop(e, 14);
    {
        int pout = V64(e), osc = V(e), po = V64(e), g = V(e), tq = V(e), etid = V(e), ecx = V(e), ecy = V(e), emc = V(e), ewarp = V(e), elane = V(e), etok = V(e), c6 = V(e);
        if (e->err) return -1;
        ldc64(e, pout, OMEGA_PATTN256_ARG_OUT); ldc32(e, osc, OMEGA_PATTN256_ARG_OUT_SCALE); ldc32(e, emc, OMEGA_PATTN256_ARG_MC);
        s2r(e, etid, BW_SR_TID_X); s2r(e, ecx, BW_SR_CTAID_X); s2r(e, ecy, BW_SR_CTAID_Y);
        shr(e, ewarp, etid, 5); andi(e, R(elane), R(etid), 31u);
        shr(e, t1, ewarp, 1); imadi(e, R(t1), R(t1), 2u, NOREG); imadi(e, R(etok), R(ecx), OMEGA_PATTN256_TOKENS, t1);
        shfl_bfly(e, sh, lA, 1); fadd(e, R(lA), R(lA), R(sh)); shfl_bfly(e, sh, lA, 2); fadd(e, R(lA), R(lA), R(sh));
        shfl_bfly(e, sh, lB, 1); fadd(e, R(lB), R(lB), R(sh)); shfl_bfly(e, sh, lB, 2); fadd(e, R(lB), R(lB), R(sh));
        mufu(e, BW_IR_MUFU_RCP, R(lA), R(lA), 1); mufu(e, BW_IR_MUFU_RCP, R(lB), R(lB), 0);
        fmul(e, R(lA), R(lA), R(osc)); fmul(e, R(lB), R(lB), R(osc));
        for (unsigned i = 0; i < 16; i++) for (unsigned k = 0; k < 4; k++) fmul(e, S(o[i], k), S(o[i], k), R(k < 2 ? lA : lB));
        /* out + ((etok*24 + cy*6 + g)*256 + (warp&1)*128 + 2t)*4 */
        shr(e, g, elane, 2); andi(e, R(tq), R(elane), 3u);
        imadi(e, R(t1), R(etok), 24u, NOREG); imadi(e, R(t1), R(ecy), 6u, t1); iadd(e, t1, t1, g); imadi(e, R(t1), R(t1), 1024u, NOREG);
        andi(e, R(t2), R(ewarp), 1u); imadi(e, R(t1), R(t2), 512u, t1); imadi(e, R(t1), R(tq), 8u, t1);
        imadw(e, po, t1, 1u, pout);
        movi(e, R(c6), 6u);
        isetp_ge_u32(e, g, c6);
        Region rg = region_skip_if_p0(e);
        for (unsigned row = 0; row < 2; row++) {
            imadi(e, R(t1), R(one), row, etok);
            isetp_ge_u32(e, t1, emc);
            const int b = bra_if_p0_fwd(e);
            for (unsigned i = 0; i < 16; i++) {
                const uint32_t off = row * OMEGA_PATTN256_OUT_WORDS * 4u + i * 32u;
                stg(e, po, off, S(o[i], 2 * row), udesc); stg(e, po, off + 4u, S(o[i], 2 * row + 1), udesc);
            }
            patch_bra(e, b, here(e));
        }
        region_end(e, rg);
    }
    tail(e);
    if (e->err) return -1;
    if (omega_bw_regalloc_solve(prog) != 0) return -1;
    return 0;
}

int omega_prefill_attn256_prog_stats(const OmegaPrefillAttn256Spec *spec, uint32_t *insns, uint32_t *gprs, uint32_t *placeholders) {
    static BlackwellIRProgram p;
    if (omega_blackwell_codegen_prefill_attn256_prog(spec, &p) != 0) return -1;
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
