/*
 * omega_blackwell_codegen_attn256.c: the one-launch decode attention kernel for head_dim 256 (see the header). The
 * emitter helpers are the ones of omega_blackwell_codegen_attn.c (control words, read barrier on every load, loop
 * live ranges); the program follows that kernel with the 27B geometry and the dim-split V pass.
 */
#include "omega_blackwell_codegen_attn256.h"
#include "omega_blackwell_codegen_attn.h"
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
static void imadr(Em *e, int d, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* d64 = a * imm + c64 */
static void imadw(Em *e, int d64, int a, uint32_t imm, int c64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = d64; i.src1_vreg = a; i.imm = imm; i.src3_vreg = c64; emit(e, i, K_FIXED); }
static void mov(Em *e, Op d, Op a) { imadi(e, d, a, 1u, NOREG); }
static void shrs(Em *e, int d, Op a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = d; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; emit(e, i, K_FIXED); }
static void shr(Em *e, int d, int a, uint32_t imm) { shrs(e, d, R(a), imm); }
static void andi(Em *e, Op d, Op a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; emit(e, i, K_FIXED); }
static void xorr(Em *e, Op d, Op a, Op b) { BlackwellIRInsn i = I0(BW_IR_LOP3_XOR); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.src2_vreg = b.v; i.src2_subreg = (uint8_t)b.s; emit(e, i, K_FIXED); }
static void iadd(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_IADD3); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void ldg_wide(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, int first) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE);
}
static void ldg32(Em *e, BlackwellIROpcode op, int d, int addr64, int udesc, int first) { BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = addr64; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void stg(Em *e, BlackwellIROpcode op, int addr64, Op val, int udesc) { BlackwellIRInsn i = I0(op); i.src1_vreg = addr64; i.src2_vreg = val.v; i.src2_subreg = (uint8_t)val.s; i.ureg = udesc; emit(e, i, K_STORE); }
static void atom_add(Em *e, int d, int addr64, int val, int udesc) { BlackwellIRInsn i = I0(BW_IR_ATOMG_ADD_STRONG_SYS); i.dst_vreg = d; i.src1_vreg = addr64; i.src2_vreg = val; i.ureg = udesc; emit(e, i, K_VAR); }
static void membar(Em *e) { emit(e, I0(BW_IR_MEMBAR_ALL_SYS), K_FIXED); }
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

/* bf16 round-to-nearest-even of f32 x into the low 16 bits of d: r = x + 0x7fff + ((x >> 16) & 1); d = r >> 16 */
static void bf16_of(Em *e, int d, Op x, int c7fff, int t) {
    shrs(e, t, x, 16);
    andi(e, R(t), R(t), 1u);
    iadd(e, t, t, c7fff);
    fop(e, BW_IR_IADD3, R(d), x, R(t), R(NOREG)); /* IADD3 d = x + t */
    shr(e, d, d, 16);
}

static const uint32_t NEG_BIG = 0xF149F2CAu; /* -1e30f */

int omega_dattn256_shape(const OmegaDecodeAttn256Spec *spec, OmegaDattn256Shape *s) {
    const uint32_t qh = spec->q_heads ? spec->q_heads : OMEGA_DATTN256_Q_HEADS;
    const uint32_t kvh = spec->kv_heads ? spec->kv_heads : OMEGA_DATTN256_KV_HEADS;
    const uint32_t h = spec->q_per_cta ? spec->q_per_cta : OMEGA_DATTN256_Q_PER_KV;
    if (h < 1u || h > 6u || qh % h || kvh == 0 || (qh / h) % kvh) return -1;
    const uint32_t g = (qh / h) / kvh;
    if (g != 1u && g != 2u) return -1;
    s->qh = qh; s->kvh = kvh; s->h = h; s->vy = qh / h; s->g = g;
    s->qkv_words = (qh + 2u * kvh) * 256u; s->out_words = qh * 256u;
    return 0;
}
#define RB OMEGA_DATTN256_ROW_BYTES
#define PBYTES (OMEGA_DATTN256_PARTIAL_WORDS * 4u)

/* 8 consecutive f32 words (two quads) to shared at base + 0 .. 28 */
static void sts8(Em *e, int base, int one, int t, int qa, int qb) {
    for (unsigned i = 0; i < 8; i++) { imadi(e, R(t), R(one), i * 4u, base); sts32(e, t, i < 4 ? S(qa, i) : S(qb, i - 4)); }
}
/* sum of squares of 8 values, warp-reduced, broadcast through shared word `addr` (lane 0 writes), rs = RSQ(sum/256 + eps) */
static void rms_scale(Em *e, int qa, int qb, int sq, int ts, int rs, int addr, int lane, int one, int eps) {
    fmul(e, R(sq), S(qa, 0), S(qa, 0));
    for (unsigned i = 1; i < 4; i++) ffma(e, R(sq), S(qa, i), S(qa, i), R(sq));
    for (unsigned i = 0; i < 4; i++) ffma(e, R(sq), S(qb, i), S(qb, i), R(sq));
    for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, ts, sq, o); fadd(e, R(sq), R(sq), R(ts)); }
    isetp_ge_u32(e, lane, one);
    { Region r = region_skip_if_p0(e); sts32(e, addr, R(sq)); region_end(e, r); }
    bar_sync(e);
    lds32(e, R(sq), addr, 1);
    movi(e, ts, 0x3b800000u);                         /* 1/256 */
    ffma(e, R(rs), R(sq), R(ts), R(eps));
    mufu(e, BW_IR_MUFU_RSQ, R(rs), R(rs));
}

int omega_blackwell_codegen_decode_attn256_prog(const OmegaDecodeAttn256Spec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog || spec->depth < 1 || spec->depth > OMEGA_DATTN256_MAX_DEPTH || spec->mutant < 0 || spec->mutant > 5 || spec->prep > 1 || spec->multi > 1) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    const uint32_t D = spec->depth;
    OmegaDattn256Shape SH;
    if (omega_dattn256_shape(spec, &SH)) return -1;
    const uint32_t H = SH.h;                         /* q heads per CTA row */
    const int OLD = SH.qh == OMEGA_DATTN256_Q_HEADS && SH.kvh == OMEGA_DATTN256_KV_HEADS && H == OMEGA_DATTN256_Q_PER_KV;
    const int MUT = spec->mutant;

    /* ---- arguments and identities ---- */
    int udesc = UV64(e);
    int pqkv = V64(e), pkc = V64(e), pvc = V64(e), prope = V64(e), pqn = V64(e);
    int pos = V(e), per = V(e), splits = V(e), kvs = V(e), eps = V(e), qks = V(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), one = V(e), zero = V(e), c7fff = V(e), c16 = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pqkv, OMEGA_DATTN256_ARG_QKV); ldc64(e, pkc, OMEGA_DATTN256_ARG_KC); ldc64(e, pvc, OMEGA_DATTN256_ARG_VC);
    ldc64(e, prope, OMEGA_DATTN256_ARG_ROPE); ldc64(e, pqn, OMEGA_DATTN256_ARG_QN); /* k_norm is read at q_norm + 1024 bytes (ARG_KN is informational) */
    ldc32(e, pos, OMEGA_DATTN256_ARG_POS); ldc32(e, per, OMEGA_DATTN256_ARG_PER_SPLIT); ldc32(e, splits, OMEGA_DATTN256_ARG_SPLITS);
    ldc32(e, kvs, OMEGA_DATTN256_ARG_KV_STRIDE); ldc32(e, eps, OMEGA_DATTN256_ARG_EPS);
    ldc32(e, qks, OMEGA_DATTN256_ARG_QK_SCALE);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    if (spec->prep && !spec->multi) {                /* prefill prep (#545): CTA cx = token cx at position pos0 + cx, its qkv row */
        iadd(e, pos, pos, cx);
        { int rowoff = V(e); if (e->err) return -1; imadi(e, R(rowoff), R(cx), SH.qkv_words * 4u, NOREG); imadw(e, pqkv, rowoff, 1u, pqkv); }
    }
    movi(e, one, 1); movrz(e, R(zero)); movi(e, c7fff, 0x7fffu); movi(e, c16, 16);
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);
    if (spec->multi) {                               /* L369 multi-token: the token, its position and qkv row (ARG_XSTRIDE bytes per row) */
        int tok = V(e), xs = V(e); if (e->err) return -1;
        ldc32(e, xs, OMEGA_DATTN256_ARG_XSTRIDE);
        if (spec->prep) mov(e, R(tok), R(cx));       /* prep: CTA (token, row) */
        else {                                       /* attend: CTA (split, token * vy + row) */
            uint32_t lg = 0; while ((1u << lg) < SH.vy) lg++;
            if ((1u << lg) != SH.vy) return -1;
            shr(e, tok, cy, lg); andi(e, R(cy), R(cy), SH.vy - 1u);
        }
        iadd(e, pos, pos, tok);
        imadr(e, xs, tok, xs, NOREG); imadw(e, pqkv, xs, 1u, pqkv);
        if (!spec->prep) {                           /* this token's splits (per_split 256): CTAs past them exit */
            shr(e, splits, pos, 8); imadi(e, R(splits), R(splits), 1u, one);
            imadi(e, R(xs), R(one), 0xffffffffu, splits);
            isetp_ge_u32(e, xs, cx);
            int bk = bra_if_p0_fwd(e); exit_k(e); patch_bra(e, bk, here(e));
        }
    }

    /* range of this CTA: base = cx * per; end = min(base + per, pos + 1) */
    int base = V(e), bend = V(e), ctx = V(e), t1 = V(e), t2 = V(e), end = V(e), endm1 = V(e), negendm1 = V(e), splitsm1 = V(e), cyk = V(e);
    imadr(e, base, cx, per, NOREG);
    imadi(e, R(bend), R(base), 1u, per);
    imadi(e, R(ctx), R(pos), 1u, one);
    imadi(e, R(t1), R(ctx), 0xffffffffu, bend);      /* t1 = bend - ctx */
    imadi(e, R(t2), R(t1), 0xffffffffu, NOREG);      /* t2 = ctx - bend */
    shr(e, t2, t2, 31);                              /* 1 when bend > ctx */
    imadr(e, t2, t2, t1, NOREG);                     /* t2 = over * (bend - ctx) */
    imadi(e, R(end), R(t2), 0xffffffffu, bend);      /* end = bend - t2 */
    imadi(e, R(endm1), R(one), 0xffffffffu, end);
    imadi(e, R(negendm1), R(endm1), 0xffffffffu, NOREG);
    imadi(e, R(splitsm1), R(one), 0xffffffffu, splits);
    if (SH.g == 1u) imadr(e, cyk, cy, kvs, NOREG);   /* byte offset of this KV head's plane */
    else { shr(e, t1, cy, 1); imadr(e, cyk, t1, kvs, NOREG); }   /* G = 2: KV head cy / 2 */

    /* ---- prologue: one norm + rope block for all 8 warps. Warp h < 6 builds q'[h] (head cy*6+h, q_norm, * qk_scale);
     *      warp 6 builds k_new (k[cy], k_norm = q_norm + 1024 bytes, scale 1); warp 7 runs the block on v[cy] (the result is
     *      discarded) and reloads the raw v at the store. Warps 6, 7 stage in the s[][] region (unused until the chunk loop). ---- */
    int qa = V128(e), qb = V128(e), wa = V128(e), wb = V128(e), pa_ = V128(e), pb_ = V128(e);
    int cosa = V128(e), cosb = V128(e), sina = V128(e), sinb = V128(e);
    int sq = V(e), ts = V(e), rs = V(e), a0 = V(e), a1 = V(e), ad64 = V64(e), rope64 = V64(e), sgn = V(e), pl = V(e), f = V(e), voff = V(e), qksw = V(e);
    if (e->err) return -1;
    movi(e, t1, 5u); imadi(e, R(t1), R(warp), 0xffffffffu, t1); /* 5 - warp */
    shr(e, f, t1, 31);                               /* f = 1 for warps 6, 7 */
    /* byte offset of the vector in qkv: q head cy*6 + warp at (cy*6 + warp)*1024; k[cy] at 24576 + cy*1024 = warp*4096 + cy*1024
     * for warp 6; v[cy] at 28672 + cy*1024 = warp*4096 + cy*1024 for warp 7 */
    imadi(e, R(t1), R(cy), H, warp); imadi(e, R(t1), R(t1), 1024u, NOREG);
    if (OLD) { imadi(e, R(t2), R(warp), 4096u, NOREG); imadi(e, R(t2), R(cy), 1024u, t2); }
    else {                                           /* q_heads*1024 + (warp - 6)*kv_heads*1024 + (cy / G)*1024 */
        if (SH.g == 1u) mov(e, R(t2), R(cy)); else shr(e, t2, cy, 1);
        imadi(e, R(t2), R(t2), 1024u, NOREG);
        imadi(e, R(t2), R(warp), SH.kvh * 1024u, t2);
        imadi(e, R(t2), R(one), (SH.qh - 6u * SH.kvh) * 1024u, t2);
    }
    imadi(e, R(t2), R(t1), 0xffffffffu, t2);         /* off_kv - off_q */
    imadr(e, voff, f, t2, t1);                       /* f ? off_kv : off_q */
    imadi(e, R(voff), R(lane), 32u, voff);           /* + this lane's 8 dims */
    imadw(e, ad64, voff, 1u, pqkv);
    ldg_wide(e, BW_IR_LDG_E_128, qa, ad64, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_128, qb, ad64, 16, udesc, 0);
    imadi(e, R(t2), R(f), 1024u, NOREG); imadi(e, R(t2), R(lane), 32u, t2); /* q_norm, or k_norm 1024 bytes after it */
    imadw(e, ad64, t2, 1u, pqn);
    ldg_wide(e, BW_IR_LDG_E_128, wa, ad64, 0, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, wb, ad64, 16, udesc, 0);
    /* rope row for this lane: pair index = (lane & 15) * 8 .. +7 (dims 8i..8i+7 for lanes < 16, dims 128 + 8(i-16) above) */
    andi(e, R(t2), R(lane), 15u);
    imadi(e, R(t2), R(t2), 32u, NOREG);
    imadi(e, R(t2), R(pos), OMEGA_DATTN256_ROPE_ROW_BYTES, t2);
    imadw(e, rope64, t2, 1u, prope);
    ldg_wide(e, BW_IR_LDG_E_128, cosa, rope64, 0, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, cosb, rope64, 16, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, sina, rope64, 512, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, sinb, rope64, 528, udesc, 0);
    /* sign of the sin term: lanes < 16 take x[d] * cos - x[d+128] * sin, lanes >= 16 take x[d] * cos + x[d-128] * sin */
    imadi(e, R(sgn), R(one), 0xfffffff0u, lane);     /* lane - 16 */
    shr(e, sgn, sgn, 31);
    imadi(e, R(sgn), R(sgn), 0x80000000u, NOREG);
    for (unsigned i = 0; i < 4; i++) { xorr(e, S(sina, i), S(sina, i), R(sgn)); xorr(e, S(sinb, i), S(sinb, i), R(sgn)); }
    xorr(e, R(pl), R(lane), R(c16));                 /* partner lane */

    /* RMSNorm of the lane's 8 dims: sum of squares over the warp, broadcast through shared memory */
    imadi(e, R(a0), R(warp), 4u, NOREG); imadi(e, R(a0), R(one), OMEGA_DATTN256_SM_L, a0);
    rms_scale(e, qa, qb, sq, ts, rs, a0, lane, one, eps);
    for (unsigned i = 0; i < 4; i++) { fmul(e, S(qa, i), S(qa, i), R(rs)); fmul(e, S(qa, i), S(qa, i), S(wa, i)); fmul(e, S(qb, i), S(qb, i), R(rs)); fmul(e, S(qb, i), S(qb, i), S(wb, i)); }
    /* stage, read the partner half, rotate */
    imadi(e, R(a0), R(warp), 1024u, NOREG); imadi(e, R(a0), R(lane), 32u, a0); /* SM_Q + warp*1024 + lane*32 (SM_Q = 0) */
    sts8(e, a0, one, t1, qa, qb);
    bar_sync(e);
    imadi(e, R(t2), R(warp), 1024u, NOREG); imadi(e, R(t2), R(pl), 32u, t2);
    lds128(e, pa_, t2, 0, 1); lds128(e, pb_, t2, 16, 0);
    if (MUT != 1) for (unsigned i = 0; i < 4; i++) {
        fmul(e, R(ts), S(qa, i), S(cosa, i)); ffma(e, S(qa, i), S(pa_, i), S(sina, i), R(ts));
        fmul(e, R(ts), S(qb, i), S(cosb, i)); ffma(e, S(qb, i), S(pb_, i), S(sinb, i), R(ts));
    }
    /* qksw = qk_scale for the q warps, 1.0f for warps 6, 7 (integer select on the bit patterns) */
    movi(e, ts, 0x3f800000u); imadi(e, R(ts), R(qks), 0xffffffffu, ts); imadr(e, qksw, f, ts, qks);
    for (unsigned i = 0; i < 4; i++) { fmul(e, S(qa, i), S(qa, i), R(qksw)); fmul(e, S(qb, i), S(qb, i), R(qksw)); }
    bar_sync(e);                                      /* every partner read done before q' overwrites the staging */
    sts8(e, a0, one, t1, qa, qb);                     /* warps 6, 7 write scratch inside s[][] */
    if (spec->prep && !spec->multi) {
        /* q' as bf16 in the attend kernel's A-fragment layout (omega_blackwell_codegen_pattn256.h, OMEGA_PATTN256_QF_*):
         * word i of lane L (dims 8L+2i, +1) goes to ((((cx>>1)*4 + cy)*6 + warp)*16 + (L>>1))*64 + i*16 + ((L&1)*2 + (cx&1))*4.
         * Warps 6, 7 (k, v) skip it. */
        int qw = V128(e), pqf = V64(e), qa64 = V64(e), h0q = V(e), h1q = V(e), tbq = V(e), fq = f; if (e->err) return -1;
        if (H < 6u) {                                /* warps H .. 5 hold no head of this row: they skip q' too */
            fq = V(e); if (e->err) return -1;
            movi(e, t1, H - 1u); imadi(e, R(t1), R(warp), 0xffffffffu, t1); shr(e, fq, t1, 31);
        }
        isetp_ge_u32(e, fq, one);
        int bq = bra_if_p0_fwd(e);
        for (unsigned w = 0; w < 4; w++) {
            int src = w < 2 ? qa : qb; unsigned s0 = (2 * w) % 4;
            bf16_of(e, h0q, S(src, s0), c7fff, tbq); bf16_of(e, h1q, S(src, s0 + 1), c7fff, tbq); imadi(e, S(qw, w), R(h1q), 65536u, h0q);
        }
        ldc64(e, pqf, OMEGA_DATTN256_ARG_OUT);
        shr(e, t1, cx, 1); imadi(e, R(t1), R(t1), SH.vy, cy); imadi(e, R(t1), R(t1), H, warp);
        shr(e, t2, lane, 1); imadi(e, R(t1), R(t1), 16u, t2); imadi(e, R(t1), R(t1), 64u, NOREG);
        andi(e, R(t2), R(lane), 1u); imadi(e, R(t2), R(t2), 2u, NOREG);
        { int cb = V(e); if (e->err) return -1; andi(e, R(cb), R(cx), 1u); iadd(e, t2, t2, cb); }
        imadi(e, R(t1), R(t2), 4u, t1);
        for (unsigned w = 0; w < 4; w++) { imadi(e, R(t2), R(one), 16u * w, t1); imadw(e, qa64, t2, 1u, pqf); stg(e, BW_IR_STG_E, qa64, S(qw, w), udesc); }
        patch_bra(e, bq, here(e));
    }

    /* ---- the new token's K (warp 6: the rotated k) and V (warp 7: the raw v), bf16 into the cache by the CTA that owns `pos` ---- */
    if (MUT != 3 && !(spec->multi && !spec->prep)) { /* multi attend: the prep launch wrote every token's K/V */
        int sa64 = V64(e), sb64 = V64(e), key = V(e), h0 = V(e), h1 = V(e), tb = V(e);
        /* key = (splits - 1 - cx) + 1 - f: zero only for warps 6, 7 of the last split */
        imadi(e, R(t2), R(cx), 0xffffffffu, splitsm1);
        imadi(e, R(t2), R(one), 1u, t2);
        if (spec->prep) imadi(e, R(key), R(f), 0xffffffffu, one);   /* prep: every CTA owns its token: key = 1 - f */
        else imadi(e, R(key), R(f), 0xffffffffu, t2);
        isetp_ge_u32(e, key, one);
        int b1 = bra_if_p0_fwd(e);
        /* cache row address of position pos for this lane: 8 bf16 = 16 bytes per lane; K plane, V plane for warp 7 */
        imadi(e, R(t1), R(pos), RB, cyk); imadi(e, R(t1), R(lane), 16u, t1);
        imadw(e, sa64, t1, 1u, pkc);
        movi(e, t2, 7u); imadi(e, R(t2), R(warp), 0xffffffffu, t2); /* 7 - warp: zero for warp 7 */
        isetp_ge_u32(e, t2, one);
        { Region r = region_skip_if_p0(e);
          imadw(e, sb64, voff, 1u, pqkv);
          ldg_wide(e, BW_IR_LDG_E_128, qa, sb64, 0, udesc, 1);
          ldg_wide(e, BW_IR_LDG_E_128, qb, sb64, 16, udesc, 0);
          imadw(e, sa64, t1, 1u, pvc);
          region_end(e, r); }
        for (unsigned w = 0; w < 4; w++) {
            int src = w < 2 ? qa : qb; unsigned s0 = (2 * w) % 4;
            bf16_of(e, h0, S(src, s0), c7fff, tb); bf16_of(e, h1, S(src, s0 + 1), c7fff, tb); imadi(e, S(wa, w), R(h1), 65536u, h0);
        }
        for (unsigned w = 0; w < 4; w++) { imadw(e, sb64, one, 4u * w, sa64); stg(e, BW_IR_STG_E, sb64, S(wa, w), udesc); }
        patch_bra(e, b1, here(e));
    }
    membar(e);
    if (spec->prep) {                                /* prep ends here: the K/V rows and q' are written */
        tail(e);
        if (e->err) return -1;
        return omega_bw_regalloc_solve(prog);
    }
    bar_sync(e);

    /* ---- running state ---- */
    int acc[6], m[6], l[6], sc[6], dl[6], wd[8], mn = V(e);
    for (unsigned h = 0; h < H; h++) { acc[h] = V128(e); m[h] = V(e); l[h] = V(e); sc[h] = V(e); dl[h] = V(e); }
    for (unsigned q = 0; q < 8; q++) wd[q] = V(e);
    int kq[OMEGA_DATTN256_MAX_DEPTH], vq[OMEGA_DATTN256_MAX_DEPTH];
    for (uint32_t d = 0; d < D; d++) { kq[d] = V128(e); vq[d] = V64(e); }
    int qa4 = V128(e), qb4 = V128(e);
    int cb = V(e), tpos = V(e), over = V(e), krow = V64(e), kpf = V64(e), qoff = V(e), j = V(e), sa = V(e), smS = V(e);
    int n = V(e), vrow = V64(e), vpf = V64(e), pa = V(e), i = V(e), dsum = V(e), hoff = V(e), wp = V(e), negwp = V(e), ccnt = V(e);
    if (e->err) return -1;
    for (unsigned h = 0; h < H; h++) { for (unsigned q = 0; q < 4; q++) movrz(e, S(acc[h], q)); movi(e, m[h], NEG_BIG); movrz(e, R(l[h])); }
    movi(e, smS, OMEGA_DATTN256_SM_S);
    shr(e, wp, warp, 1);                              /* warp pair index 0..3 */
    imadi(e, R(negwp), R(wp), 0xffffffffu, NOREG);
    andi(e, R(hoff), R(warp), 1u);
    imadi(e, R(hoff), R(hoff), 256u, NOREG);          /* byte offset of this warp's dim half in the row */
    imadi(e, R(hoff), R(lane), 8u, hoff);             /* + 4 bf16 = 8 bytes per lane */
    if (MUT == 2) imadi(e, R(hoff), R(one), RB, hoff);
    movi(e, ccnt, 32);
    mov(e, R(cb), R(base));

    /* ---- chunk loop ---- */
    const int L_chunk = here(e);
    loop_begin(e);
    isetp_ge_u32(e, cb, end);
    const int bra_done = bra_if_p0_fwd(e);
    /* score pass: thread = position cb + tid, clamped to end - 1 (clamped lanes compute a score nobody reads) */
    imadi(e, R(tpos), R(cb), 1u, tid);
    imadi(e, R(over), R(tpos), 0xffffffffu, endm1);  /* endm1 - tpos */
    shr(e, over, over, 31);                           /* 1 when tpos > endm1 */
    imadi(e, R(t1), R(tpos), 1u, negendm1);           /* tpos - endm1 */
    imadr(e, t1, over, t1, NOREG);
    imadi(e, R(t1), R(t1), 0xffffffffu, tpos);        /* clamped position */
    imadi(e, R(t1), R(t1), RB, cyk);
    imadw(e, krow, t1, 1u, pkc);
    for (unsigned h = 0; h < H; h++) movrz(e, R(sc[h]));
    for (uint32_t d = 0; d < D; d++) ldg_wide(e, BW_IR_LDG_E_128, kq[d], krow, d * 16u, udesc, d == 0);
    imadw(e, kpf, one, D * 16u, krow);
    movrz(e, R(qoff)); movrz(e, R(j));
    const int L_q = here(e);
    loop_begin(e);
    isetp_ge_u32(e, j, ccnt);
    const int bra_qend = bra_if_p0_fwd(e);
    for (unsigned w = 0; w < 4; w++) { imadi(e, R(wd[2 * w]), S(kq[0], w), 65536u, NOREG); andi(e, R(wd[2 * w + 1]), S(kq[0], w), 0xffff0000u); }
    for (unsigned h = 0; h < H; h++) {
        lds128(e, qa4, qoff, h * 1024u, 1); lds128(e, qb4, qoff, h * 1024u + 16u, 0);
        for (unsigned q = 0; q < 4; q++) ffma(e, R(sc[h]), R(wd[q]), S(qa4, q), R(sc[h]));
        for (unsigned q = 0; q < 4; q++) ffma(e, R(sc[h]), R(wd[4 + q]), S(qb4, q), R(sc[h]));
    }
    for (uint32_t d = 0; d + 1 < D; d++) for (unsigned w = 0; w < 4; w++) mov(e, S(kq[d], w), S(kq[d + 1], w));
    ldg_wide(e, BW_IR_LDG_E_128, kq[D - 1], kpf, 0, udesc, 1);
    imadw(e, kpf, one, 16u, kpf);
    imadi(e, R(qoff), R(one), 32u, qoff);
    imadi(e, R(j), R(one), 1u, j);
    bra(e, L_q);
    loop_end(e);
    patch_bra(e, bra_qend, here(e));
    imadi(e, R(sa), R(tid), 32u, smS);
    for (unsigned h = 0; h < H; h++) { imadi(e, R(t1), R(one), h * 4u, sa); sts32(e, t1, R(sc[h])); }
    bar_sync(e);
    /* V pass: this warp's positions cb + wp + 4 i, i < n = max(0, ceil((min(end, cb + 256) - cb - wp) / 4)) */
    imadi(e, R(t1), R(one), OMEGA_DATTN256_CHUNK, cb);  /* cb + 256 */
    imadi(e, R(t2), R(t1), 0xffffffffu, end);           /* end - (cb + 256) */
    shr(e, over, t2, 31);                               /* 1 when the chunk is cut short by end */
    imadr(e, t2, over, t2, t1);                         /* cend = min(end, cb + 256) */
    imadi(e, R(t1), R(cb), 0xffffffffu, t2);            /* cend - cb */
    imadi(e, R(t1), R(t1), 1u, negwp);                  /* x = cend - cb - wp */
    shr(e, over, t1, 31);
    imadi(e, R(n), R(one), 3u, t1);
    shr(e, n, n, 2);
    imadr(e, t2, over, n, NOREG);
    imadi(e, R(n), R(t2), 0xffffffffu, n);
    imadi(e, R(t1), R(cb), 1u, wp);
    imadi(e, R(t1), R(t1), RB, hoff);
    imadi(e, R(t1), R(t1), 1u, cyk);
    imadw(e, vrow, t1, 1u, pvc);
    for (uint32_t d = 0; d < D; d++) ldg_wide(e, BW_IR_LDG_E_64, vq[d], vrow, d * 4u * RB, udesc, d == 0);
    imadw(e, vpf, one, D * 4u * RB, vrow);
    imadi(e, R(pa), R(wp), 32u, smS);
    movrz(e, R(i));
    const int L_v = here(e);
    loop_begin(e);
    isetp_ge_u32(e, i, n);
    const int bra_vend = bra_if_p0_fwd(e);
    imadi(e, R(wd[0]), S(vq[0], 0), 65536u, NOREG); andi(e, R(wd[1]), S(vq[0], 0), 0xffff0000u);
    imadi(e, R(wd[2]), S(vq[0], 1), 65536u, NOREG); andi(e, R(wd[3]), S(vq[0], 1), 0xffff0000u);
    lds128(e, qa4, pa, 0, 1); lds128(e, qb4, pa, 16, 0);
#define SH(h) ((h) < 4 ? S(qa4, (h)) : S(qb4, (h) - 4))
    for (unsigned h = 0; h < H; h++) { fmaxx(e, R(mn), R(m[h]), SH(h)); fsub(e, R(dl[h]), R(m[h]), R(mn)); mov(e, R(m[h]), R(mn)); } /* dl <= 0; exactly +0 when the max stayed */
    fadd(e, R(dsum), R(dl[0]), R(dl[1]));
    for (unsigned h = 2; h < H; h++) fadd(e, R(dsum), R(dsum), R(dl[h]));
    isetp_ge_u32(e, one, dsum);                       /* 1 >= bits(dsum): no head's max moved (dsum is +0); a negative dsum has the sign bit */
    const int bra_nores = bra_if_p0_fwd(e);
    for (unsigned h = 0; h < H; h++) mufu(e, BW_IR_MUFU_EX2, R(dl[h]), R(dl[h]));
    for (unsigned h = 0; h < H; h++) { for (unsigned q = 0; q < 4; q++) fmul(e, S(acc[h], q), S(acc[h], q), R(dl[h])); fmul(e, R(l[h]), R(l[h]), R(dl[h])); }
    patch_bra(e, bra_nores, here(e));
    for (unsigned h = 0; h < H; h++) { fsub(e, R(dl[h]), SH(h), R(m[h])); mufu(e, BW_IR_MUFU_EX2, R(dl[h]), R(dl[h])); }
#undef SH
    for (unsigned h = 0; h < H; h++) { for (unsigned q = 0; q < 4; q++) ffma(e, S(acc[h], q), R(dl[h]), R(wd[q]), S(acc[h], q)); fadd(e, R(l[h]), R(l[h]), R(dl[h])); }
    for (uint32_t d = 0; d + 1 < D; d++) for (unsigned w = 0; w < 2; w++) mov(e, S(vq[d], w), S(vq[d + 1], w));
    ldg_wide(e, BW_IR_LDG_E_64, vq[D - 1], vpf, 0, udesc, 1);
    imadw(e, vpf, one, 4u * RB, vpf);
    imadi(e, R(pa), R(one), 4u * 32u, pa);
    imadi(e, R(i), R(one), 1u, i);
    bra(e, L_v);
    loop_end(e);
    patch_bra(e, bra_vend, here(e));
    bar_sync(e);                                      /* the next score pass overwrites s[][] */
    imadi(e, R(cb), R(one), OMEGA_DATTN256_CHUNK, cb);
    bra(e, L_chunk);
    loop_end(e);
    patch_bra(e, bra_done, here(e));

    /* ---- epilogue: combine the warps, write the partial ---- */
    int ep[8], ea[8], eb[8];
    for (unsigned h = 0; h < 8; h++) { ep[h] = V(e); ea[h] = V(e); eb[h] = V(e); }
    int pout = V64(e), ppart = V64(e), pcnt = V64(e), oscale = V(e);
    if (e->err) return -1;
    ldc64(e, pout, OMEGA_DATTN256_ARG_OUT); ldc64(e, ppart, OMEGA_DATTN256_ARG_PART); ldc64(e, pcnt, OMEGA_DATTN256_ARG_CNT); ldc32(e, oscale, OMEGA_DATTN256_ARG_OUT_SCALE);
    if (spec->multi) {                               /* the token's output row, partials block and counters (ctaid.y re-read: no live register) */
        int tk = V(e), so = V(e); if (e->err) return -1;
        uint32_t lg = 0; while ((1u << lg) < SH.vy) lg++;
        s2r(e, tk, BW_SR_CTAID_Y); shr(e, tk, tk, lg);
        ldc32(e, so, OMEGA_DATTN256_ARG_OSTRIDE); imadr(e, so, tk, so, NOREG); imadw(e, pout, so, 1u, pout);
        ldc32(e, so, OMEGA_DATTN256_ARG_PSTRIDE); imadr(e, so, tk, so, NOREG); imadw(e, ppart, so, 1u, ppart);
        imadi(e, R(so), R(tk), SH.vy * 4u, NOREG); imadw(e, pcnt, so, 1u, pcnt);
    }
    /* per-warp maxes to shared: lane 0 of warp w writes m[h] at SM_M + w*32 + h*4 */
    imadi(e, R(a0), R(warp), 32u, NOREG); imadi(e, R(a0), R(one), OMEGA_DATTN256_SM_M, a0);
    isetp_ge_u32(e, lane, one);
    { Region r = region_skip_if_p0(e); for (unsigned h = 0; h < H; h++) { imadi(e, R(t1), R(one), h * 4u, a0); sts32(e, t1, R(m[h])); } region_end(e, r); }
    bar_sync(e);
    /* lanes 0..5 of every warp: M[lane] = max over the 8 warps; lanes of warp 0 write it at SM_M + 256 + lane*4 */
    movi(e, ccnt, H);
    isetp_ge_u32(e, lane, ccnt);
    {
        Region r = region_skip_if_p0(e);
        imadi(e, R(a1), R(lane), 4u, NOREG); imadi(e, R(a1), R(one), OMEGA_DATTN256_SM_M, a1);
        for (unsigned w = 0; w < 8; w++) { imadi(e, R(t1), R(one), w * 32u, a1); lds32(e, R(ep[w]), t1, w == 0); }
        for (unsigned w = 1; w < 8; w++) fmaxx(e, R(ep[0]), R(ep[0]), R(ep[w]));
        imadi(e, R(t1), R(one), 256u, a1);
        sts32(e, t1, R(ep[0]));
        region_end(e, r);
    }
    bar_sync(e);
    movi(e, t1, OMEGA_DATTN256_SM_M + 256u);
    lds128(e, qa4, t1, 0, 1); lds128(e, qb4, t1, 16, 0);
    for (unsigned h = 0; h < H; h++) { fsub(e, R(ea[h]), R(m[h]), h < 4 ? S(qa4, h) : S(qb4, h - 4)); mufu(e, BW_IR_MUFU_EX2, R(ea[h]), R(ea[h])); }
    for (unsigned h = 0; h < H; h++) { for (unsigned q = 0; q < 4; q++) fmul(e, S(acc[h], q), S(acc[h], q), R(ea[h])); fmul(e, R(l[h]), R(l[h]), R(ea[h])); }
    /* per-warp l to shared, then the partial's m and l by threads 0..5 (l summed over the even warps: the odd ones repeat them) */
    imadi(e, R(a0), R(warp), 32u, NOREG); imadi(e, R(a0), R(one), OMEGA_DATTN256_SM_L, a0);
    isetp_ge_u32(e, lane, one);
    { Region r = region_skip_if_p0(e); for (unsigned h = 0; h < H; h++) { imadi(e, R(t1), R(one), h * 4u, a0); sts32(e, t1, R(l[h])); } region_end(e, r); }
    bar_sync(e);
    int pp = V64(e), pa64 = V64(e), pb64 = V64(e);
    imadr(e, t2, cy, splits, cx);
    imadi(e, R(t2), R(t2), PBYTES, NOREG);
    imadw(e, pp, t2, 1u, ppart);
    isetp_ge_u32(e, tid, ccnt);                       /* tid >= 6 skips */
    {
        Region r = region_skip_if_p0(e);
        imadi(e, R(a1), R(tid), 4u, NOREG); imadi(e, R(a1), R(one), OMEGA_DATTN256_SM_L, a1);
        for (unsigned w = 0; w < 4; w++) { imadi(e, R(t1), R(one), w * 64u, a1); lds32(e, R(ep[w]), t1, w == 0); }
        for (unsigned w = 1; w < 4; w++) fadd(e, R(ep[0]), R(ep[0]), R(ep[w]));
        imadi(e, R(t1), R(tid), 4u, NOREG); imadi(e, R(t1), R(one), OMEGA_DATTN256_SM_M + 256u, t1);
        lds32(e, R(ep[1]), t1, 1);
        imadw(e, pa64, tid, 4u, pp);
        stg(e, BW_IR_STG_E, pa64, R(ep[1]), udesc);  /* m[h] */
        imadw(e, pb64, one, 4u * H, pa64);
        stg(e, BW_IR_STG_E, pb64, R(ep[0]), udesc);  /* l[h] */
        region_end(e, r);
    }
    /* acc: 6 rounds of one head through the s[][] region as r[w][256]: warp w writes its dim half, thread tid sums the
     * 4 warps that hold dim tid (warps (tid >> 7) + 2k); the store and load addresses computed once */
    imadi(e, R(a0), R(warp), 1024u, NOREG); imadi(e, R(a0), R(lane), 16u, a0);
    andi(e, R(t1), R(warp), 1u); imadi(e, R(a0), R(t1), 512u, a0); imadi(e, R(a0), R(one), OMEGA_DATTN256_SM_S, a0);
    shr(e, t1, tid, 7);
    if (MUT == 5) movrz(e, R(t1));
    imadi(e, R(a1), R(t1), 1024u, NOREG); imadi(e, R(a1), R(tid), 4u, a1); imadi(e, R(a1), R(one), OMEGA_DATTN256_SM_S, a1);
    for (unsigned q = 0; q < 4; q++) imadi(e, R(eb[q]), R(one), q * 4u, a0);
    for (unsigned w = 0; w < 4; w++) imadi(e, R(ea[w]), R(one), w * 2048u, a1);
    imadw(e, pa64, tid, 4u, pp);
    for (unsigned rd = 0; rd < H; rd++) {
        for (unsigned q = 0; q < 4; q++) sts32(e, eb[q], S(acc[rd], q));
        bar_sync(e);
        for (unsigned w = 0; w < 4; w++) lds32(e, R(ep[w]), ea[w], w == 0);
        for (unsigned w = 1; w < 4; w++) fadd(e, R(ep[0]), R(ep[0]), R(ep[w]));
        imadw(e, pb64, one, 8u * H + rd * 1024u, pa64);
        stg(e, BW_IR_STG_E, pb64, R(ep[0]), udesc);
        bar_sync(e);
    }
    /* publish, count, the last CTA combines */
    membar(e);
    bar_sync(e);
    movi(e, t1, OMEGA_DATTN256_SM_FLAG);
    isetp_ge_u32(e, tid, one);
    {
        Region r = region_skip_if_p0(e);
        imadw(e, pb64, cy, 4u, pcnt);
        atom_add(e, ep[0], pb64, one, udesc);
        sts32(e, t1, R(ep[0]));
        region_end(e, r);
    }
    bar_sync(e);
    lds32(e, R(ep[0]), t1, 1);
    isetp_ge_u32(e, ep[0], splitsm1);
    const int bra_last = bra_if_p0_fwd(e);
    exit_k(e);
    patch_bra(e, bra_last, here(e));
    /* combine: thread tid, outputs o = tid + 256 r (head r, dim tid), r = 0..5 */
    int ho = V(e), oidx = V(e), ma = V64(e), la = V64(e), aa = V64(e), num = V(e), den = V(e), mx = V(e), s_i = V(e), rr = V(e);
    if (e->err) return -1;
    movrz(e, R(ho));
    mov(e, R(oidx), R(tid));
    movrz(e, R(rr));
    movi(e, ccnt, H);
    imadr(e, t2, cy, splits, NOREG);
    imadi(e, R(t2), R(t2), PBYTES, NOREG);           /* byte offset of partials[cy][0] */
    const int L_r = here(e);
    loop_begin(e);
    isetp_ge_u32(e, rr, ccnt);
    const int bra_rend = bra_if_p0_fwd(e);
    imadi(e, R(t1), R(ho), 4u, t2); imadw(e, ma, t1, 1u, ppart);
    imadw(e, la, one, 4u * H, ma);
    imadi(e, R(t1), R(oidx), 4u, t2); imadw(e, aa, t1, 1u, ppart); imadw(e, aa, one, 8u * H, aa);
    movrz(e, R(num)); movrz(e, R(den)); movi(e, mx, NEG_BIG); movrz(e, R(s_i));
    const int L_s = here(e);
    loop_begin(e);
    isetp_ge_u32(e, s_i, splits);
    const int bra_send = bra_if_p0_fwd(e);
    ldg32(e, BW_IR_LDG_STRONG_SYS, ep[0], ma, udesc, 1);
    ldg32(e, BW_IR_LDG_STRONG_SYS, ep[1], la, udesc, 0);
    ldg32(e, BW_IR_LDG_STRONG_SYS, ep[2], aa, udesc, 0);
    if (MUT != 4) {
        fmaxx(e, R(ep[3]), R(mx), R(ep[0]));
        fsub(e, R(ep[4]), R(mx), R(ep[3])); mufu(e, BW_IR_MUFU_EX2, R(ep[4]), R(ep[4]));
        fsub(e, R(ep[5]), R(ep[0]), R(ep[3])); mufu(e, BW_IR_MUFU_EX2, R(ep[5]), R(ep[5]));
        mov(e, R(mx), R(ep[3]));
        fmul(e, R(num), R(num), R(ep[4])); fmul(e, R(den), R(den), R(ep[4]));
    } else {
        movi(e, ep[5], 0x3f800000u);
    }
    ffma(e, R(num), R(ep[5]), R(ep[2]), R(num));
    ffma(e, R(den), R(ep[5]), R(ep[1]), R(den));
    imadw(e, ma, one, PBYTES, ma);
    imadw(e, la, one, PBYTES, la);
    imadw(e, aa, one, PBYTES, aa);
    imadi(e, R(s_i), R(one), 1u, s_i);
    bra(e, L_s);
    loop_end(e);
    patch_bra(e, bra_send, here(e));
    mufu(e, BW_IR_MUFU_RCP, R(den), R(den));
    fmul(e, R(num), R(num), R(den));
    fmul(e, R(num), R(num), R(oscale));
    imadi(e, R(t1), R(cy), H * 256u, oidx);
    imadw(e, pb64, t1, 4u, pout);
    stg(e, BW_IR_STG_E, pb64, R(num), udesc);
    imadi(e, R(ho), R(one), 1u, ho);
    imadi(e, R(oidx), R(one), 256u, oidx);
    imadi(e, R(rr), R(one), 1u, rr);
    bra(e, L_r);
    loop_end(e);
    patch_bra(e, bra_rend, here(e));
    /* thread 0 resets the counter for the next launch (every other CTA of this head has already added) */
    isetp_ge_u32(e, tid, one);
    { Region r = region_skip_if_p0(e); imadw(e, pb64, cy, 4u, pcnt); stg(e, BW_IR_STG_STRONG_SYS, pb64, R(zero), udesc); region_end(e, r); }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_decode_attn256_prog_stats(const OmegaDecodeAttn256Spec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_decode_attn256_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}
