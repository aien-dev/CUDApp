/*
 * omega_blackwell_codegen_gdn.c: the one-launch Gated DeltaNet decode step (see the header). Emitter rules follow
 * omega_blackwell_codegen_router.c (control words, read barrier on every load) and the held-scoreboard prefetch of
 * omega_bw_em.h (the state loads are in flight through the conv, norm and scalar phases).
 */
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
#define SB_STATE 3u   /* held: the 16 state quads */
#define SB_GATE 4u    /* held: z and the norm weight */
#define NOREG (-1)

enum { K_FIXED, K_PRED, K_VAR, K_VAR_MORE, K_UVAR, K_STORE, K_SYNC, K_EXIT, K_BRA, K_BSSY, K_BSYNC };

#define MAX_LOOPS 2

typedef struct {
    BlackwellIRProgram *p;
    uint32_t pending, held;
    int err;
    int depth, open_start[MAX_LOOPS];
    uint32_t placeholders; /* SHFL.BFLY words omega_tc_patch_code must patch */
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
/* d64 = a * imm + c64 */
static void imadw(Em *e, int d64, int a, uint32_t imm, int c64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = d64; i.src1_vreg = a; i.imm = imm; i.src3_vreg = c64; emit(e, i, K_FIXED); }
static void shr(Em *e, int d, int a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = d; i.src1_vreg = a; i.imm = imm; emit(e, i, K_FIXED); }
static void andi(Em *e, Op d, Op a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = a.v; i.src1_subreg = (uint8_t)a.s; i.imm = imm; emit(e, i, K_FIXED); }
static void i2fp(Em *e, int d, int a) { BlackwellIRInsn i = I0(BW_IR_I2FP); i.dst_vreg = d; i.src1_vreg = a; emit(e, i, K_FIXED); }
static void ldg_wide(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, int first) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE);
}
static void ldg32(Em *e, int d, int addr64, int udesc, int first) { BlackwellIRInsn i = I0(BW_IR_LDG_E); i.dst_vreg = d; i.src1_vreg = addr64; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE); }
/* A prefetch load: scoreboard `sb` is held, not waited for, until consume() */
static void ldg_pf(Em *e, int d, int addr64, uint32_t imm, int udesc, unsigned sb) {
    BlackwellIRInsn in = I0(BW_IR_LDG_E_128); in.dst_vreg = d; in.src1_vreg = addr64; in.imm = imm; in.ureg = udesc;
    in.control = CW(4, sb, SB_RD, (e->pending & (1u << SB_RD))) | CW_YIELD;
    e->pending |= 1u << SB_RD; e->held |= 1u << sb;
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}
/* L313: the bf16 state prefetch, LDG.E.64 (four bf16 of one state row) under the held scoreboard `sb` */
static void ldg_pf64(Em *e, int d, int addr64, uint32_t imm, int udesc, unsigned sb) {
    BlackwellIRInsn in = I0(BW_IR_LDG_E_64); in.dst_vreg = d; in.src1_vreg = addr64; in.imm = imm; in.ureg = udesc;
    in.control = CW(4, sb, SB_RD, (e->pending & (1u << SB_RD))) | CW_YIELD;
    e->pending |= 1u << SB_RD; e->held |= 1u << sb;
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}
static void consume(Em *e, unsigned sb) { e->pending |= e->held & (1u << sb); e->held &= ~(1u << sb); }
/* STG.E [addr64 + imm], val (imm: signed 24-bit byte offset) */
static void stg(Em *e, int addr64, uint32_t imm, Op val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.src2_vreg = val.v; i.src2_subreg = (uint8_t)val.s; i.imm = imm; i.ureg = udesc; emit(e, i, K_STORE); }
/* L313 bf16 state. F2FP.BF16.F32.PACK_AB d = bf16_rn(hi) << 16 | bf16_rn(lo) (a placeholder omega_tc_patch_code patches,
 * omega_blackwell_codegen_tc.h; counted by omega_gdn_bfly_count). */
static void f2fp(Em *e, Op d, Op hi, Op lo) {
    BlackwellIRInsn i = I0(BW_IR_FMNMX_MAX); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = hi.v; i.src1_subreg = (uint8_t)hi.s;
    i.src2_vreg = lo.v; i.src2_subreg = (uint8_t)lo.s; i.imm = OMEGA_TC_MARK_F2FP; emit(e, i, K_FIXED); e->placeholders++;
}
/* Four f32 (q, a V128) from four bf16 (w, a V64; element 0 in the low half of word 0): the exact widen. */
static void bf16x4_widen(Em *e, int q, int w) {
    for (unsigned j = 0; j < 4; j++) {
        if (j % 2 == 0) imadi(e, S(q, j), S(w, j / 2), 65536u, NOREG);
        else andi(e, S(q, j), S(w, j / 2), 0xffff0000u);
    }
}
/* Four f32 (q) to four bf16 (w), round to nearest even. */
static void bf16x4_pack(Em *e, int w, int q) {
    f2fp(e, S(w, 0), S(q, 1), S(q, 0));
    f2fp(e, S(w, 1), S(q, 3), S(q, 2));
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
static int bra_fwd(Em *e) { int at = here(e); BlackwellIRInsn i = I0(BW_IR_BRA); i.imm = 1; emit(e, i, K_BRA); return at; }
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
/* SHFL.BFLY placeholder (omega_blackwell_codegen_tc.h), patched after encoding */
static void shfl_bfly(Em *e, int d, int a, uint32_t lanes) { shfl_down(e, d, a, OMEGA_TC_MARK_BFLY | (lanes & 31u)); e->placeholders++; }
/* butterfly sum over lane strides lo, 2 lo .. hi: every lane of a group ends with the same bits (a + b == b + a) */
static void bfly_sum(Em *e, int v, int t, uint32_t lo, uint32_t hi) { for (uint32_t o = lo; o <= hi; o *= 2) { shfl_bfly(e, t, v, o); fadd(e, R(v), R(v), R(t)); } }
/* warp sum: the total lands in lane 0 */
static void warp_sum(Em *e, int v, int t) { for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, v, o); fadd(e, R(v), R(v), R(t)); } }
/* lane 0 of every warp stores v to shared [addr] */
static void lane0_sts(Em *e, int lane, int one, int addr, Op v) {
    isetp_ge_u32(e, lane, one);
    Region r = region_skip_if_p0(e); sts32(e, addr, v); region_end(e, r);
}
/* x = x * sigmoid(x) = x * RCP(1 + EX2(-x * log2 e)); t scratch */
static void silu(Em *e, Op x, int t, int nlog2e, int fone) {
    fmul(e, R(t), x, R(nlog2e)); mufu(e, BW_IR_MUFU_EX2, R(t), R(t)); fadd(e, R(t), R(t), R(fone));
    mufu(e, BW_IR_MUFU_RCP, R(t), R(t)); fmul(e, x, x, R(t));
}
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
/* The 8-way cross-warp sum of acc[0..3] (columns 4L..4L+3 of each warp) through shared SM_P: pa = this thread's
 * partial slot, la = lane * 16. Every warp ends with the same totals, summed in warp order 0..7. */
static void cross_warp_sum(Em *e, int acc[4], int pa, int one, int la) {
    int pj = V(e), b0 = V128(e), b1 = V128(e);
    if (e->err) return;
    sts32(e, pa, R(acc[0]));
    for (unsigned j = 1; j < 4; j++) { imadi(e, R(pj), R(one), 4u * j, pa); sts32(e, pj, R(acc[j])); }
    bar_sync(e);
    for (unsigned w2 = 0; w2 < 8; w2 += 2) {
        lds128(e, b0, la, OMEGA_GDN_SM_P + w2 * OMEGA_GDN_HEAD_DIM * 4u, 1);
        lds128(e, b1, la, OMEGA_GDN_SM_P + (w2 + 1) * OMEGA_GDN_HEAD_DIM * 4u, 0);
        for (unsigned j = 0; j < 4; j++) {
            if (w2 == 0) fadd(e, R(acc[j]), S(b0, j), S(b1, j));
            else { fadd(e, R(acc[j]), R(acc[j]), S(b0, j)); fadd(e, R(acc[j]), R(acc[j]), S(b1, j)); }
        }
    }
}

int omega_gdn_shape(const OmegaGdnSpec *spec, OmegaGdnShape *s) {
    uint32_t r = spec->ratio ? spec->ratio : 3u;
    if (r < 1u || r > 3u) return -1;
    s->r = r;
    s->vh = OMEGA_GDN_KEY_HEADS * r;
    s->vdim = s->vh * OMEGA_GDN_HEAD_DIM;
    s->conv = 2u * OMEGA_GDN_KEY_HEADS * OMEGA_GDN_HEAD_DIM + s->vdim;
    s->par_scal = s->conv * OMEGA_GDN_TAPS * 4u;
    s->par_norm = s->par_scal + s->vh * 8u;
    s->par_bytes = s->par_norm + OMEGA_GDN_HEAD_DIM * 4u;
    s->tail_bytes = s->vh * OMEGA_GDN_TAIL_HEAD_BYTES;
    s->state_bytes = spec->state_bf16 ? s->vh * (OMEGA_GDN_STATE_HEAD_BYTES / 2u) : s->vh * OMEGA_GDN_STATE_HEAD_BYTES;
    s->post_grid_y = s->vh / 8u;
    return 0;
}

int omega_blackwell_codegen_gdn_prog(const OmegaGdnSpec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog || spec->copy || spec->mutant < 0 || spec->mutant > 4) return -1;
    OmegaGdnShape G;
    if (omega_gdn_shape(spec, &G)) return -1;
    const uint32_t RR = G.r, NC = 2u + G.r;
    (void)RR; (void)NC;
    const int MUT = spec->mutant;
    const uint32_t C_LOG2E = fbits(1.44269504f), C_NLOG2E = fbits(-1.44269504f), C_LN2 = fbits(0.693147181f);
    const uint32_t C_ONE = fbits(1.0f), C_M1 = fbits(-1.0f), C_INV128 = fbits(1.0f / 128.0f);
    const float qs = 0.0883883476f; /* 1 / sqrt(128) */
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;

    /* ---- identities; the state address ---- */
    int udesc = UV64(e);
    int tid = V(e), cx = V(e), cy = V(e), h = V(e), warp = V(e), lane = V(e), one = V(e);
    int ps = V64(e), sa = V64(e), toff = V(e), sq[16];
    for (unsigned r = 0; r < 16; r++) sq[r] = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, ps, OMEGA_GDN_ARG_STATE);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, one, 1);
    imadi(e, R(h), R(cy), RR, cx);
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);
    /* the state element size: 4 (f32) or 2 (bf16, L313); this thread holds rows 16 warp .. +15, columns 4 lane .. +3 */
    const unsigned sel = spec->state_bf16 ? 2u : 4u;
    imadi(e, R(toff), R(h), OMEGA_GDN_HEAD_DIM * OMEGA_GDN_HEAD_DIM * sel, NOREG);
    imadi(e, R(toff), R(warp), 16u * OMEGA_GDN_HEAD_DIM * sel, toff);
    imadi(e, R(toff), R(lane), 4u * sel, toff);
    imadw(e, sa, toff, 1u, ps);

    /* ---- phase A: conv + SiLU of channel A (q or k) and channel V ---- */
    int half = V(e), c = V(e), gi = V(e), slv = V(e);
    int pqkv = V64(e), ppar = V64(e), ptail = V64(e);
    int axA = V64(e), awA = V64(e), atA = V64(e), axV = V64(e), awV = V64(e), atV = V64(e);
    int xA = V(e), xV = V(e), wA = V128(e), wV = V128(e), tA = V128(e), tV = V128(e);
    int insc = V(e), accA = V(e), accV = V(e), t1 = V(e), cnl2e = V(e), fone = V(e);
    if (e->err) return -1;
    ldc64(e, pqkv, OMEGA_GDN_ARG_QKV); ldc64(e, ppar, OMEGA_GDN_ARG_PAR); ldc64(e, ptail, OMEGA_GDN_ARG_TAIL);
    ldc32(e, insc, OMEGA_GDN_ARG_IN_SCALE);
    shr(e, half, tid, 7);
    andi(e, R(c), R(tid), 127u);
    imadi(e, R(gi), R(cy), OMEGA_GDN_HEAD_DIM, tid);              /* q channel hk*128 + t; k channel 2048 + hk*128 + (t - 128) */
    imadi(e, R(gi), R(half), 2048u - 128u, gi);
    imadw(e, axA, gi, 4u, pqkv); imadw(e, awA, gi, 16u, ppar);
    imadi(e, R(gi), R(h), OMEGA_GDN_HEAD_DIM, c);                 /* v channel 4096 + h*128 + c */
    imadi(e, R(gi), R(one), 2u * OMEGA_GDN_KEY_HEADS * OMEGA_GDN_HEAD_DIM, gi);
    imadw(e, axV, gi, 4u, pqkv); imadw(e, awV, gi, 16u, ppar);
    imadw(e, atA, tid, 16u, ptail); imadw(e, atA, h, OMEGA_GDN_TAIL_HEAD_BYTES, atA);
    imadi(e, R(slv), R(one), 256u, c);
    imadw(e, atV, slv, 16u, ptail); imadw(e, atV, h, OMEGA_GDN_TAIL_HEAD_BYTES, atV);
    ldg32(e, xA, axA, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_128, wA, awA, 0, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, tA, atA, 0, udesc, 0);
    ldg32(e, xV, axV, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, wV, awV, 0, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, tV, atV, 0, udesc, 0);
    fmul(e, R(xA), R(xA), R(insc)); fmul(e, R(xV), R(xV), R(insc));
    fmul(e, R(accA), S(wA, 0), S(tA, 0)); ffma(e, R(accA), S(wA, 1), S(tA, 1), R(accA));
    ffma(e, R(accA), S(wA, 2), S(tA, 2), R(accA)); ffma(e, R(accA), S(wA, 3), R(xA), R(accA));
    fmul(e, R(accV), S(wV, 0), S(tV, 0)); ffma(e, R(accV), S(wV, 1), S(tV, 1), R(accV));
    ffma(e, R(accV), S(wV, 2), S(tV, 2), R(accV)); ffma(e, R(accV), S(wV, 3), R(xV), R(accV));
    movi(e, cnl2e, C_NLOG2E); movi(e, fone, C_ONE);
    silu(e, R(accA), t1, cnl2e, fone);
    silu(e, R(accV), t1, cnl2e, fone);
    /* the state prefetch (here, not earlier: the conv phase is the register peak) */
    int sw[16];
    if (spec->state_bf16) {
        for (unsigned r = 0; r < 16; r++) sw[r] = V64(e);
        if (e->err) return -1;
        for (unsigned r = 0; r < 16; r++) ldg_pf64(e, sw[r], sa, r * OMEGA_GDN_HEAD_DIM * 2u, udesc, SB_STATE);
    } else {
        for (unsigned r = 0; r < 16; r++) ldg_pf(e, sq[r], sa, r * OMEGA_GDN_HEAD_DIM * 4u, udesc, SB_STATE);
    }

    /* ---- phase B: l2 norms; q|k and v to shared; the tail shift ---- */
    int sqs = V(e), ts = V(e), ra = V(e);
    if (e->err) return -1;
    fmul(e, R(sqs), R(accA), R(accA));
    warp_sum(e, sqs, ts);
    imadi(e, R(ra), R(warp), 4u, NOREG); imadi(e, R(ra), R(one), OMEGA_GDN_SM_RED, ra);
    lane0_sts(e, lane, one, ra, R(sqs));
    imadi(e, R(ra), R(c), 4u, NOREG); imadi(e, R(ra), R(one), OMEGA_GDN_SM_V, ra);
    sts32(e, ra, R(accV));
    bar_sync(e);
    if (MUT != 4) {
        stg(e, atA, 0, S(tA, 1), udesc); stg(e, atA, 4, S(tA, 2), udesc); stg(e, atA, 8, R(xA), udesc);
        stg(e, atV, 0, S(tV, 1), udesc); stg(e, atV, 4, S(tV, 2), udesc); stg(e, atV, 8, R(xV), udesc);
    }
    {
        int rq = V128(e), el2 = V(e), hf = V(e), fq = V(e);
        if (e->err) return -1;
        ldc32(e, el2, OMEGA_GDN_ARG_EPS_L2);
        imadi(e, R(ra), R(half), 16u, NOREG);
        lds128(e, rq, ra, OMEGA_GDN_SM_RED, 1);
        fadd(e, R(sqs), S(rq, 0), S(rq, 1)); fadd(e, R(sqs), R(sqs), S(rq, 2)); fadd(e, R(sqs), R(sqs), S(rq, 3));
        fadd(e, R(sqs), R(sqs), R(el2));
        mufu(e, BW_IR_MUFU_RSQ, R(sqs), R(sqs));
        /* q gets 1/sqrt(128) more: factor = qs + half * (1 - qs) */
        i2fp(e, hf, half);
        movi(e, fq, fbits(qs)); movi(e, t1, fbits(1.0f - qs));
        ffma(e, R(fq), R(hf), R(t1), R(fq));
        fmul(e, R(accA), R(accA), R(sqs)); fmul(e, R(accA), R(accA), R(fq));
        imadi(e, R(ra), R(tid), 4u, NOREG);                       /* SM_Q = 0, SM_K = SM_Q + 512 */
        sts32(e, ra, R(accA));
        bar_sync(e); /* q and k are read across warps in phase D (a zero state hides a missing barrier here: 0 * poison = 0) */
    }

    /* ---- phase C: scalars ---- */
    int pab = V64(e), aab = V64(e), asc = V64(e), abq = V64(e), scq = V64(e);
    int xs = V(e), mn = V(e), sp = V(e), dec = V(e), beta = V(e), cl2e = V(e);
    if (e->err) return -1;
    ldc64(e, pab, OMEGA_GDN_ARG_AB);
    imadw(e, aab, h, 8u, pab); imadw(e, asc, h, 8u, ppar);
    ldg_wide(e, BW_IR_LDG_E_64, abq, aab, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_64, scq, asc, G.par_scal, udesc, 0);
    /* softplus(x) = max(x, 0) + ln2 * LG2(1 + EX2(-|x| * log2 e)), x = a + dt_bias */
    fadd(e, R(xs), S(abq, 0), S(scq, 1));
    movi(e, mn, C_M1); fmul(e, R(mn), R(xs), R(mn)); fminn(e, R(mn), R(xs), R(mn));
    movi(e, cl2e, C_LOG2E);
    fmul(e, R(mn), R(mn), R(cl2e)); mufu(e, BW_IR_MUFU_EX2, R(mn), R(mn)); fadd(e, R(mn), R(mn), R(fone));
    mufu(e, BW_IR_MUFU_LG2, R(mn), R(mn));
    movrz(e, R(sp)); fmaxx(e, R(sp), R(xs), R(sp));
    movi(e, t1, C_LN2); ffma(e, R(sp), R(mn), R(t1), R(sp));
    /* decay = EX2(negA * softplus * log2 e) */
    if (MUT == 1) movi(e, dec, C_ONE);
    else { fmul(e, R(dec), S(scq, 0), R(sp)); fmul(e, R(dec), R(dec), R(cl2e)); mufu(e, BW_IR_MUFU_EX2, R(dec), R(dec)); }
    /* beta = sigmoid(b) */
    if (MUT == 2) movi(e, beta, C_ONE);
    else {
        fmul(e, R(beta), S(abq, 1), R(cnl2e)); mufu(e, BW_IR_MUFU_EX2, R(beta), R(beta));
        fadd(e, R(beta), R(beta), R(fone)); mufu(e, BW_IR_MUFU_RCP, R(beta), R(beta));
    }

    /* ---- phase D: the state update (fresh registers per step: a reused name stays live across all its uses) ---- */
    int aw = V(e), pa = V(e), la = V(e), kq[4], kv[4], dl[4], vq = V128(e);
    for (unsigned i = 0; i < 4; i++) { kq[i] = V128(e); kv[i] = V(e); dl[i] = V(e); }
    if (e->err) return -1;
    imadi(e, R(aw), R(warp), 16u * 4u, NOREG);
    imadi(e, R(pa), R(warp), OMEGA_GDN_HEAD_DIM * 4u, NOREG); imadi(e, R(pa), R(lane), 16u, pa); imadi(e, R(pa), R(one), OMEGA_GDN_SM_P, pa);
    imadi(e, R(la), R(lane), 16u, NOREG);
    consume(e, SB_STATE);
    if (spec->state_bf16) for (unsigned r = 0; r < 16; r++) bf16x4_widen(e, sq[r], sw[r]);
    for (unsigned r = 0; r < 16; r++) for (unsigned j = 0; j < 4; j++) fmul(e, S(sq[r], j), S(sq[r], j), R(dec));
    /* kv partial over this warp's 16 rows: kv_j = sum_r S[r][j] k[r], then the 8-way cross-warp sum */
    for (unsigned i = 0; i < 4; i++) lds128(e, kq[i], aw, OMEGA_GDN_SM_K + i * 16u, i == 0);
    for (unsigned j = 0; j < 4; j++) fmul(e, R(kv[j]), S(sq[0], j), S(kq[0], 0));
    for (unsigned r = 1; r < 16; r++) for (unsigned j = 0; j < 4; j++) ffma(e, R(kv[j]), S(sq[r], j), S(kq[r / 4], r % 4), R(kv[j]));
    cross_warp_sum(e, kv, pa, one, la);
    /* delta = (v - kv) * beta; S += k (outer) delta; S back to memory */
    lds128(e, vq, la, OMEGA_GDN_SM_V, 1);
    for (unsigned j = 0; j < 4; j++) { fsub(e, R(dl[j]), S(vq, j), R(kv[j])); fmul(e, R(dl[j]), R(dl[j]), R(beta)); }
    {
        int k2[4];
        for (unsigned i = 0; i < 4; i++) k2[i] = V128(e);
        if (e->err) return -1;
        for (unsigned i = 0; i < 4; i++) lds128(e, k2[i], aw, OMEGA_GDN_SM_K + i * 16u, i == 0);
        for (unsigned r = 0; r < 16; r++) for (unsigned j = 0; j < 4; j++) ffma(e, S(sq[r], j), S(k2[r / 4], r % 4), R(dl[j]), S(sq[r], j));
    }
    if (spec->state_bf16) {
        int pk[16];
        for (unsigned r = 0; r < 16; r++) pk[r] = V64(e);
        if (e->err) return -1;
        for (unsigned r = 0; r < 16; r++) {
            bf16x4_pack(e, pk[r], sq[r]);
            for (unsigned w = 0; w < 2; w++) stg(e, sa, r * OMEGA_GDN_HEAD_DIM * 2u + w * 4u, S(pk[r], w), udesc);
        }
    } else {
        for (unsigned r = 0; r < 16; r++) for (unsigned j = 0; j < 4; j++) stg(e, sa, r * OMEGA_GDN_HEAD_DIM * 4u + j * 4u, S(sq[r], j), udesc);
    }
    /* the gate prefetch: z[h] and the norm weight, consumed in phase E */
    int pz = V64(e), ppar2 = V64(e), az = V64(e), an = V64(e), zq = V128(e), nq = V128(e);
    if (e->err) return -1;
    ldc64(e, pz, OMEGA_GDN_ARG_Z); ldc64(e, ppar2, OMEGA_GDN_ARG_PAR);
    imadw(e, az, lane, 16u, pz); imadw(e, az, h, OMEGA_GDN_HEAD_DIM * 4u, az);
    imadw(e, an, lane, 16u, ppar2);
    ldg_pf(e, zq, az, 0, udesc, SB_GATE);
    ldg_pf(e, nq, an, G.par_norm, udesc, SB_GATE);
    /* o partial: o_j = sum_r S[r][j] q[r], then the cross-warp sum */
    int p[4];
    {
        int q2[4];
        for (unsigned i = 0; i < 4; i++) { q2[i] = V128(e); p[i] = V(e); }
        if (e->err) return -1;
        for (unsigned i = 0; i < 4; i++) lds128(e, q2[i], aw, OMEGA_GDN_SM_Q + i * 16u, i == 0);
        for (unsigned j = 0; j < 4; j++) fmul(e, R(p[j]), S(sq[0], j), S(q2[0], 0));
        for (unsigned r = 1; r < 16; r++) for (unsigned j = 0; j < 4; j++) ffma(e, R(p[j]), S(sq[r], j), S(q2[r / 4], r % 4), R(p[j]));
    }
    bar_sync(e); /* every warp has read the kv partials */
    cross_warp_sum(e, p, pa, one, la);

    /* ---- phase E: gated RMSNorm, the output ---- */
    int ss = V(e), en = V(e), osc = V(e), pout = V64(e), ao = V64(e), g = V(e);
    if (e->err) return -1;
    fmul(e, R(ss), R(p[0]), R(p[0]));
    for (unsigned j = 1; j < 4; j++) ffma(e, R(ss), R(p[j]), R(p[j]), R(ss));
    warp_sum(e, ss, ts);
    imadi(e, R(ra), R(warp), 4u, NOREG); imadi(e, R(ra), R(one), OMEGA_GDN_SM_RED2, ra);
    lane0_sts(e, lane, one, ra, R(ss));
    bar_sync(e);
    ldc32(e, en, OMEGA_GDN_ARG_EPS_NORM); ldc32(e, osc, OMEGA_GDN_ARG_OUT_SCALE); ldc64(e, pout, OMEGA_GDN_ARG_OUT);
    lds32(e, R(ss), ra, 0);
    movi(e, t1, C_INV128);
    ffma(e, R(ss), R(ss), R(t1), R(en));
    mufu(e, BW_IR_MUFU_RSQ, R(ss), R(ss));
    imadw(e, ao, lane, 16u, pout); imadw(e, ao, h, OMEGA_GDN_HEAD_DIM * 4u, ao);
    consume(e, SB_GATE);
    for (unsigned j = 0; j < 4; j++) {
        fmul(e, R(g), S(zq, j), R(insc));
        if (MUT != 3) silu(e, R(g), t1, cnl2e, fone);
        fmul(e, R(p[j]), R(p[j]), R(ss)); fmul(e, R(p[j]), R(p[j]), S(nq, j));
        fmul(e, R(p[j]), R(p[j]), R(g)); fmul(e, R(p[j]), R(p[j]), R(osc));
    }
    for (unsigned j = 0; j < 4; j++) stg(e, ao, j * 4u, R(p[j]), udesc);
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_gdn_prog_stats(const OmegaGdnSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_gdn_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}

/* ======== state copy / zero (see the header) ======== */
int omega_blackwell_codegen_gdn_copy_prog(const OmegaGdnSpec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog || spec->mutant || (spec->copy != OMEGA_GDN_COPY && spec->copy != OMEGA_GDN_ZERO)) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    int udesc = UV64(e), tid = V(e), cx = V(e), w = V(e), pd = V64(e), ad = V64(e), v[4];
    for (unsigned j = 0; j < 4; j++) v[j] = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pd, OMEGA_GDN_COPY_ARG_DST);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X);
    imadi(e, R(w), R(cx), OMEGA_GDN_COPY_WORDS_PER_CTA, tid);
    imadw(e, ad, w, 4u, pd);
    if (spec->copy == OMEGA_GDN_COPY) {
        /* the 32-bit LDG.E carries no immediate: one address per word */
        int ps = V64(e), one = V(e), wj[4], as[4];
        for (unsigned j = 0; j < 4; j++) { wj[j] = V(e); as[j] = V64(e); }
        if (e->err) return -1;
        ldc64(e, ps, OMEGA_GDN_COPY_ARG_SRC);
        movi(e, one, 1);
        for (unsigned j = 0; j < 4; j++) { imadi(e, R(wj[j]), R(one), j * OMEGA_GDN_COPY_THREADS, w); imadw(e, as[j], wj[j], 4u, ps); }
        for (unsigned j = 0; j < 4; j++) ldg_wide(e, BW_IR_LDG_E, v[j], as[j], 0, udesc, j == 0);
    } else {
        for (unsigned j = 0; j < 4; j++) movrz(e, R(v[j]));
    }
    for (unsigned j = 0; j < 4; j++) stg(e, ad, j * OMEGA_GDN_COPY_THREADS * 4u, R(v[j]), udesc);
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

/* ======== prefill (omega#543): prep, scan, post (layouts in the header) ======== */

/* softplus, decay and beta of one value head (the decode phase C maths): abq = {a, b}, scq = {negA, dt_bias} */
static void gdn_scalars(Em *e, int MUT, int abq, int scq, int dec, int beta, int t1, int cnl2e, int fone) {
    int xs = V(e), mn = V(e), sp = V(e), cl2e = V(e);
    if (e->err) return;
    fadd(e, R(xs), S(abq, 0), S(scq, 1));
    movi(e, mn, fbits(-1.0f)); fmul(e, R(mn), R(xs), R(mn)); fminn(e, R(mn), R(xs), R(mn));
    movi(e, cl2e, fbits(1.44269504f));
    fmul(e, R(mn), R(mn), R(cl2e)); mufu(e, BW_IR_MUFU_EX2, R(mn), R(mn)); fadd(e, R(mn), R(mn), R(fone));
    mufu(e, BW_IR_MUFU_LG2, R(mn), R(mn));
    movrz(e, R(sp)); fmaxx(e, R(sp), R(xs), R(sp));
    movi(e, t1, fbits(0.693147181f)); ffma(e, R(sp), R(mn), R(t1), R(sp));
    if (MUT == 1) movi(e, dec, fbits(1.0f));
    else { fmul(e, R(dec), S(scq, 0), R(sp)); fmul(e, R(dec), R(dec), R(cl2e)); mufu(e, BW_IR_MUFU_EX2, R(dec), R(dec)); }
    if (MUT == 2) movi(e, beta, fbits(1.0f));
    else {
        fmul(e, R(beta), S(abq, 1), R(cnl2e)); mufu(e, BW_IR_MUFU_EX2, R(beta), R(beta));
        fadd(e, R(beta), R(beta), R(fone)); mufu(e, BW_IR_MUFU_RCP, R(beta), R(beta));
    }
}

int omega_blackwell_codegen_gdn_prep_prog(const OmegaGdnSpec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog || spec->copy || spec->mutant < 0 || spec->mutant > 4) return -1;
    OmegaGdnShape G;
    if (omega_gdn_shape(spec, &G)) return -1;
    const uint32_t RR = G.r, NC = 2u + G.r;
    (void)RR; (void)NC;
    const int MUT = spec->mutant;
    const float qs = 0.0883883476f; /* 1 / sqrt(128) */
    const uint32_t ROW = G.conv * 4u, REC_ROW = G.vh * OMEGA_GDN_REC_BYTES;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;

    int udesc = UV64(e);
    int c = V(e), cx = V(e), hk = V(e), warp = V(e), lane = V(e), one = V(e), T = V(e), t0 = V(e), n = V(e), t1 = V(e);
    int pqkv = V64(e), ppar = V64(e), ptail = V64(e), prec = V64(e), pab = V64(e), insc = V(e);
    int ch[5], tw[5], wq[5], ax[5];
    for (unsigned i = 0; i < NC; i++) { ch[i] = V(e); tw[i] = V128(e); wq[i] = V128(e); ax[i] = V64(e); }
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    s2r(e, c, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, hk, BW_SR_CTAID_Y);
    ldc64(e, pqkv, OMEGA_GDN_PREP_ARG_QKV); ldc64(e, ppar, OMEGA_GDN_PREP_ARG_PAR); ldc64(e, ptail, OMEGA_GDN_PREP_ARG_TAIL);
    ldc64(e, prec, OMEGA_GDN_PREP_ARG_REC); ldc64(e, pab, OMEGA_GDN_PREP_ARG_AB);
    ldc32(e, T, OMEGA_GDN_PREP_ARG_T); ldc32(e, insc, OMEGA_GDN_PREP_ARG_IN_SCALE);
    movi(e, one, 1);
    shr(e, warp, c, 5);
    andi(e, R(lane), R(c), 31u);
    imadi(e, R(t0), R(cx), OMEGA_GDN_PREP_TOKENS, NOREG);
    /* n = min(16, T - t0) >= 1 */
    imadi(e, R(n), R(t0), 0xffffffffu, T);
    movi(e, t1, OMEGA_GDN_PREP_TOKENS);
    isetp_ge_u32(e, t1, n);
    { int b = bra_if_p0_fwd(e); movi(e, n, OMEGA_GDN_PREP_TOKENS); patch_bra(e, b, here(e)); }
    /* channels: q hk*128 + c, k 2048 + hk*128 + c, v_i 4096 + (3 hk + i)*128 + c */
    imadi(e, R(ch[0]), R(hk), OMEGA_GDN_HEAD_DIM, c);
    imadi(e, R(ch[1]), R(one), 2048u, ch[0]);
    imadi(e, R(ch[2]), R(hk), RR * OMEGA_GDN_HEAD_DIM, c); imadi(e, R(ch[2]), R(one), 4096u, ch[2]);
    for (unsigned i = 3; i < NC; i++) imadi(e, R(ch[i]), R(one), (i - 2u) * OMEGA_GDN_HEAD_DIM, ch[2]);
    /* taps */
    {
        int aq = V64(e), av = V64(e);
        if (e->err) return -1;
        imadw(e, aq, ch[0], 16u, ppar); imadw(e, av, ch[2], 16u, ppar);
        ldg_wide(e, BW_IR_LDG_E_128, tw[0], aq, 0, udesc, 1);
        ldg_wide(e, BW_IR_LDG_E_128, tw[1], aq, 2048u * 16u, udesc, 0);
        for (unsigned i = 0; i < RR; i++) ldg_wide(e, BW_IR_LDG_E_128, tw[2 + i], av, i * OMEGA_GDN_HEAD_DIM * 16u, udesc, 0);
    }
    /* the window {x_{t0-3}, x_{t0-2}, x_{t0-1}}: CTA 0 from the tail, the others from the rows before t0 (uniform) */
    isetp_ge_u32(e, cx, one);
    const int b_rows = bra_if_p0_fwd(e);
    {
        int at = V64(e);
        if (e->err) return -1;
        imadi(e, R(t1), R(hk), RR * OMEGA_GDN_TAIL_HEAD_BYTES, NOREG);
        imadw(e, at, t1, 1u, ptail); imadw(e, at, c, 16u, at);
        ldg_wide(e, BW_IR_LDG_E_128, wq[0], at, 0, udesc, 1);
        ldg_wide(e, BW_IR_LDG_E_128, wq[1], at, OMEGA_GDN_HEAD_DIM * 16u, udesc, 0);
        for (unsigned i = 0; i < RR; i++) ldg_wide(e, BW_IR_LDG_E_128, wq[2 + i], at, i * OMEGA_GDN_TAIL_HEAD_BYTES + 256u * 16u, udesc, 0);
    }
    const int b_join = bra_fwd(e);
    patch_bra(e, b_rows, here(e));
    {
        int base = V64(e), a[5][3];
        for (unsigned i = 0; i < NC; i++) for (unsigned r = 0; r < 3; r++) a[i][r] = V64(e);
        if (e->err) return -1;
        imadi(e, R(t1), R(one), 0xfffffffdu, t0);                 /* t0 - 3 (t0 >= 16 here) */
        imadw(e, base, t1, ROW, pqkv);
        for (unsigned i = 0; i < NC; i++) {
            imadw(e, a[i][0], ch[i], 4u, base);
            imadw(e, a[i][1], one, ROW, a[i][0]); imadw(e, a[i][2], one, ROW, a[i][1]);
        }
        for (unsigned i = 0; i < NC; i++) for (unsigned r = 0; r < 3; r++) {
            BlackwellIRInsn in = I0(BW_IR_LDG_E); in.dst_vreg = wq[i]; in.dst_subreg = (uint8_t)r; in.src1_vreg = a[i][r]; in.ureg = udesc;
            emit(e, in, (i == 0 && r == 0) ? K_VAR : K_VAR_MORE);
        }
        for (unsigned i = 0; i < NC; i++) for (unsigned r = 0; r < 3; r++) fmul(e, S(wq[i], r), S(wq[i], r), R(insc));
    }
    patch_bra(e, b_join, here(e));

    /* per-thread addresses at token t0 */
    int idx = V(e), arec = V64(e), asc = V64(e), aab = V64(e), scq = V64(e);
    int cnl2e = V(e), fone = V(e), el2 = V(e), cqs = V(e), c3 = V(e), zero = V(e), raq = V(e), rak = V(e);
    if (e->err) return -1;
    {
        int base = V64(e);
        if (e->err) return -1;
        imadw(e, base, t0, ROW, pqkv);
        for (unsigned i = 0; i < NC; i++) imadw(e, ax[i], ch[i], 4u, base);
    }
    imadi(e, R(idx), R(t0), G.vh, NOREG); imadi(e, R(idx), R(hk), RR, idx);
    imadw(e, arec, idx, OMEGA_GDN_REC_BYTES, prec); imadw(e, arec, c, 4u, arec);
    imadi(e, R(t1), R(c), 1u, idx);  /* the scalar threads' head index t0*48 + 3hk + c */
    imadw(e, asc, t1, OMEGA_GDN_REC_BYTES, prec);
    imadw(e, aab, t1, 8u, pab);
    movi(e, c3, RR);
    isetp_ge_u32(e, c, c3);
    {
        Region r = region_skip_if_p0(e);
        int as = V64(e);
        if (e->err) return -1;
        imadi(e, R(t1), R(hk), RR, c);
        imadw(e, as, t1, 8u, ppar);
        ldg_wide(e, BW_IR_LDG_E_64, scq, as, G.par_scal, udesc, 1);
        region_end(e, r);
    }
    movi(e, cnl2e, fbits(-1.44269504f)); movi(e, fone, fbits(1.0f)); movi(e, cqs, fbits(qs));
    ldc32(e, el2, OMEGA_GDN_PREP_ARG_EPS_L2);
    movi(e, zero, 0u);
    imadi(e, R(raq), R(warp), 4u, NOREG); imadi(e, R(rak), R(one), 16u, raq);

    const int L = here(e);
    loop_begin(e);
    {
        int x[5], acc[5], sq = V(e), sk = V(e), ts = V(e), rq = V128(e), rk = V128(e), nq = V(e), nk = V(e);
        for (unsigned i = 0; i < NC; i++) { x[i] = V(e); acc[i] = V(e); }
        if (e->err) return -1;
        for (unsigned i = 0; i < NC; i++) ldg32(e, x[i], ax[i], udesc, i == 0);
        for (unsigned i = 0; i < NC; i++) {
            fmul(e, R(x[i]), R(x[i]), R(insc));
            fmul(e, R(acc[i]), S(tw[i], 0), S(wq[i], 0)); ffma(e, R(acc[i]), S(tw[i], 1), S(wq[i], 1), R(acc[i]));
            ffma(e, R(acc[i]), S(tw[i], 2), S(wq[i], 2), R(acc[i])); ffma(e, R(acc[i]), S(tw[i], 3), R(x[i]), R(acc[i]));
            silu(e, R(acc[i]), t1, cnl2e, fone);
            mov(e, S(wq[i], 0), S(wq[i], 1)); mov(e, S(wq[i], 1), S(wq[i], 2)); mov(e, S(wq[i], 2), R(x[i]));
        }
        /* l2 norms of q and k across the 4 warps (the decode order: warp sums, then partials 0 + 1 + 2 + 3) */
        fmul(e, R(sq), R(acc[0]), R(acc[0])); warp_sum(e, sq, ts); lane0_sts(e, lane, one, raq, R(sq));
        fmul(e, R(sk), R(acc[1]), R(acc[1])); warp_sum(e, sk, ts); lane0_sts(e, lane, one, rak, R(sk));
        bar_sync(e);
        lds128(e, rq, zero, 0, 1); lds128(e, rk, zero, 16, 0);
        fadd(e, R(nq), S(rq, 0), S(rq, 1)); fadd(e, R(nq), R(nq), S(rq, 2)); fadd(e, R(nq), R(nq), S(rq, 3)); fadd(e, R(nq), R(nq), R(el2));
        fadd(e, R(nk), S(rk, 0), S(rk, 1)); fadd(e, R(nk), R(nk), S(rk, 2)); fadd(e, R(nk), R(nk), S(rk, 3)); fadd(e, R(nk), R(nk), R(el2));
        mufu(e, BW_IR_MUFU_RSQ, R(nq), R(nq)); mufu(e, BW_IR_MUFU_RSQ, R(nk), R(nk));
        fmul(e, R(acc[0]), R(acc[0]), R(nq)); fmul(e, R(acc[0]), R(acc[0]), R(cqs));
        fmul(e, R(acc[1]), R(acc[1]), R(nk));
        for (unsigned i = 0; i < RR; i++) {
            stg(e, arec, i * OMEGA_GDN_REC_BYTES, R(acc[0]), udesc);
            stg(e, arec, i * OMEGA_GDN_REC_BYTES + OMEGA_GDN_REC_K, R(acc[1]), udesc);
            stg(e, arec, i * OMEGA_GDN_REC_BYTES + OMEGA_GDN_REC_V, R(acc[2 + i]), udesc);
        }
        /* threads 0..2: decay and beta of value head 3 hk + c */
        isetp_ge_u32(e, c, c3);
        {
            Region r = region_skip_if_p0(e);
            int abq = V64(e), dec = V(e), beta = V(e);
            if (e->err) return -1;
            ldg_wide(e, BW_IR_LDG_E_64, abq, aab, 0, udesc, 1);
            gdn_scalars(e, MUT, abq, scq, dec, beta, t1, cnl2e, fone);
            stg(e, asc, OMEGA_GDN_REC_SC, R(dec), udesc); stg(e, asc, OMEGA_GDN_REC_SC + 4u, R(beta), udesc);
            region_end(e, r);
        }
        bar_sync(e); /* every warp has read the partials before the next token writes them */
        for (unsigned i = 0; i < NC; i++) imadw(e, ax[i], one, ROW, ax[i]);
        imadw(e, arec, one, REC_ROW, arec); imadw(e, asc, one, REC_ROW, asc);
        imadw(e, aab, one, G.vh * 8u, aab);
        imadi(e, R(n), R(one), 0xffffffffu, n);
        isetp_ge_u32(e, n, one);
        bra_if_p0(e, L);
    }
    loop_end(e);
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_blackwell_codegen_gdn_scan_prog(const OmegaGdnSpec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog || spec->copy || spec->mutant < 0 || spec->mutant > 4) return -1;
    OmegaGdnShape G;
    if (omega_gdn_shape(spec, &G)) return -1;
    const uint32_t RR = G.r, NC = 2u + G.r;
    (void)RR; (void)NC;
    const int MUT = spec->mutant;
    const uint32_t ROW = G.conv * 4u, CHUNK_ROW = OMEGA_GDN_SCAN_CHUNK * G.vh * OMEGA_GDN_REC_BYTES;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;

    int udesc = UV64(e);
    int tid = V(e), cx = V(e), cy = V(e), h = V(e), warp = V(e), lane = V(e), one = V(e), cq = V(e), rg = V(e), c0 = V(e);
    int T = V(e), tm1 = V(e), t = V(e), t1 = V(e), sl = V(e), srec = V(e);
    int ps = V64(e), sa = V64(e), prec = V64(e), grec = V64(e), gsc = V64(e), po = V64(e), sq[16];
    for (unsigned r = 0; r < 16; r++) sq[r] = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    ldc64(e, ps, OMEGA_GDN_SCAN_ARG_STATE); ldc64(e, prec, OMEGA_GDN_SCAN_ARG_REC); ldc64(e, po, OMEGA_GDN_SCAN_ARG_O);
    ldc32(e, T, OMEGA_GDN_SCAN_ARG_T);
    movi(e, one, 1);
    imadi(e, R(h), R(cy), RR, cx);
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);
    andi(e, R(cq), R(lane), 3u);
    shr(e, rg, lane, 2);
    /* this thread's block of S: rows 16 rg .. +15, columns c0 = 16 warp + 4 cq .. +3 */
    imadi(e, R(c0), R(warp), 16u, NOREG); imadi(e, R(c0), R(cq), 4u, c0);
    const unsigned sel = spec->state_bf16 ? 2u : 4u; /* f32 or bf16 (L313) state element */
    imadi(e, R(t1), R(h), OMEGA_GDN_HEAD_DIM * OMEGA_GDN_HEAD_DIM * sel, NOREG);
    imadi(e, R(t1), R(rg), 16u * OMEGA_GDN_HEAD_DIM * sel, t1);
    imadi(e, R(t1), R(c0), sel, t1);
    imadw(e, sa, t1, 1u, ps);
    if (spec->state_bf16) {
        int sw[16];
        for (unsigned r = 0; r < 16; r++) sw[r] = V64(e);
        if (e->err) return -1;
        for (unsigned r = 0; r < 16; r++) ldg_wide(e, BW_IR_LDG_E_64, sw[r], sa, r * OMEGA_GDN_HEAD_DIM * 2u, udesc, r == 0);
        for (unsigned r = 0; r < 16; r++) bf16x4_widen(e, sq[r], sw[r]);
    } else {
        for (unsigned r = 0; r < 16; r++) ldg_wide(e, BW_IR_LDG_E_128, sq[r], sa, r * OMEGA_GDN_HEAD_DIM * 4u, udesc, r == 0);
    }
    imadi(e, R(tm1), R(one), 0xffffffffu, T);
    /* the record copy: warp w brings token tc + w into shared slot w */
    imadi(e, R(t1), R(warp), G.vh, h);
    imadw(e, gsc, t1, OMEGA_GDN_REC_BYTES, prec);
    imadw(e, grec, lane, 16u, gsc);
    imadi(e, R(srec), R(warp), OMEGA_GDN_REC_BYTES, NOREG);
    imadi(e, R(sl), R(lane), 16u, srec);
    /* o row of (token 0, head h), columns c0 .. c0 + 3 */
    imadi(e, R(t1), R(h), OMEGA_GDN_HEAD_DIM, c0);
    imadw(e, po, t1, 4u, po);
    movrz(e, R(t));

    const int L = here(e);
    loop_begin(e);
    {
        int tm = V(e), sb = V(e), ak = V(e), av = V(e), scq = V128(e), vq = V128(e), ts = V(e), kv[4], dl[4], p[4];
        for (unsigned j = 0; j < 4; j++) { kv[j] = V(e); dl[j] = V(e); p[j] = V(e); }
        if (e->err) return -1;
        andi(e, R(tm), R(t), OMEGA_GDN_SCAN_CHUNK - 1u);
        isetp_ge_u32(e, tm, one);
        const int b_copy = bra_if_p0_fwd(e);
        {
            int rr[4], a = V(e);
            for (unsigned k = 0; k < 4; k++) rr[k] = V128(e);
            if (e->err) return -1;
            bar_sync(e); /* every warp is done with the last chunk */
            for (unsigned k = 0; k < 3; k++) ldg_wide(e, BW_IR_LDG_E_128, rr[k], grec, k * 512u, udesc, k == 0);
            ldg_wide(e, BW_IR_LDG_E_128, rr[3], gsc, OMEGA_GDN_REC_SC, udesc, 0);
            for (unsigned k = 0; k < 3; k++) for (unsigned j = 0; j < 4; j++) {
                if (k == 0 && j == 0) sts32(e, sl, S(rr[0], 0));
                else { imadi(e, R(a), R(one), k * 512u + j * 4u, sl); sts32(e, a, S(rr[k], j)); }
            }
            for (unsigned j = 0; j < 4; j++) { imadi(e, R(a), R(one), OMEGA_GDN_REC_SC + j * 4u, srec); sts32(e, a, S(rr[3], j)); }
            bar_sync(e);
            imadw(e, grec, one, CHUNK_ROW, grec); imadw(e, gsc, one, CHUNK_ROW, gsc);
        }
        patch_bra(e, b_copy, here(e));
        imadi(e, R(sb), R(tm), OMEGA_GDN_REC_BYTES, NOREG);
        imadi(e, R(ak), R(rg), 16u * 4u, sb);
        imadi(e, R(av), R(c0), 4u, sb);
        lds128(e, scq, sb, OMEGA_GDN_REC_SC, 1);
        for (unsigned r = 0; r < 16; r++) for (unsigned j = 0; j < 4; j++) fmul(e, S(sq[r], j), S(sq[r], j), S(scq, 0));
        /* kv_j = sum over 128 rows of S[r][j] k[r]: 16 rows here, then lanes ^ 4, 8, 16 (the 8 row groups) */
        for (unsigned i = 0; i < 4; i++) {
            int kq = V128(e);
            if (e->err) return -1;
            lds128(e, kq, ak, OMEGA_GDN_REC_K + i * 16u, 1);
            for (unsigned r = 4 * i; r < 4 * i + 4; r++) for (unsigned j = 0; j < 4; j++) {
                if (r == 0) fmul(e, R(kv[j]), S(sq[0], j), S(kq, 0));
                else ffma(e, R(kv[j]), S(sq[r], j), S(kq, r % 4), R(kv[j]));
            }
        }
        for (unsigned j = 0; j < 4; j++) bfly_sum(e, kv[j], ts, 4u, 16u);
        lds128(e, vq, av, OMEGA_GDN_REC_V, 1);
        for (unsigned j = 0; j < 4; j++) { fsub(e, R(dl[j]), S(vq, j), R(kv[j])); fmul(e, R(dl[j]), R(dl[j]), S(scq, 1)); }
        for (unsigned i = 0; i < 4; i++) {
            int kq = V128(e);
            if (e->err) return -1;
            lds128(e, kq, ak, OMEGA_GDN_REC_K + i * 16u, 1);
            for (unsigned r = 4 * i; r < 4 * i + 4; r++) for (unsigned j = 0; j < 4; j++) ffma(e, S(sq[r], j), S(kq, r % 4), R(dl[j]), S(sq[r], j));
        }
        for (unsigned i = 0; i < 4; i++) {
            int qq = V128(e);
            if (e->err) return -1;
            lds128(e, qq, ak, i * 16u, 1);
            for (unsigned r = 4 * i; r < 4 * i + 4; r++) for (unsigned j = 0; j < 4; j++) {
                if (r == 0) fmul(e, R(p[j]), S(sq[0], j), S(qq, 0));
                else ffma(e, R(p[j]), S(sq[r], j), S(qq, r % 4), R(p[j]));
            }
        }
        for (unsigned j = 0; j < 4; j++) bfly_sum(e, p[j], ts, 4u, 16u);
        isetp_ge_u32(e, rg, one);
        {
            Region r = region_skip_if_p0(e);
            for (unsigned j = 0; j < 4; j++) stg(e, po, j * 4u, R(p[j]), udesc);
            region_end(e, r);
        }
        imadw(e, po, one, G.vdim * 4u, po);
        imadi(e, R(t), R(one), 1u, t);
        isetp_ge_u32(e, tm1, t);
        bra_if_p0(e, L);
    }
    loop_end(e);
    if (spec->state_bf16) {
        int pk[16];
        for (unsigned r = 0; r < 16; r++) pk[r] = V64(e);
        if (e->err) return -1;
        for (unsigned r = 0; r < 16; r++) {
            bf16x4_pack(e, pk[r], sq[r]);
            for (unsigned w = 0; w < 2; w++) stg(e, sa, r * OMEGA_GDN_HEAD_DIM * 2u + w * 4u, S(pk[r], w), udesc);
        }
    } else {
        for (unsigned r = 0; r < 16; r++) for (unsigned j = 0; j < 4; j++) stg(e, sa, r * OMEGA_GDN_HEAD_DIM * 4u + j * 4u, S(sq[r], j), udesc);
    }

    /* the new tail: qkv rows T-3 .. T-1 (times in_scale) of slot A (tid) and slot V (256 + (tid & 127)), as decode */
    if (MUT != 4) {
        int half = V(e), cc = V(e), gA = V(e), gV = V(e), insc = V(e), pqkv = V64(e), ptail = V64(e), base = V64(e);
        int atA = V64(e), atV = V64(e), aA[3], aV[3], xA[3], xV[3];
        for (unsigned r = 0; r < 3; r++) { aA[r] = V64(e); aV[r] = V64(e); xA[r] = V(e); xV[r] = V(e); }
        if (e->err) return -1;
        ldc64(e, pqkv, OMEGA_GDN_SCAN_ARG_QKV); ldc64(e, ptail, OMEGA_GDN_SCAN_ARG_TAIL); ldc32(e, insc, OMEGA_GDN_SCAN_ARG_IN_SCALE);
        shr(e, half, tid, 7);
        andi(e, R(cc), R(tid), 127u);
        imadi(e, R(gA), R(cy), OMEGA_GDN_HEAD_DIM, tid); imadi(e, R(gA), R(half), 2048u - 128u, gA);
        imadi(e, R(gV), R(h), OMEGA_GDN_HEAD_DIM, cc); imadi(e, R(gV), R(one), 2u * OMEGA_GDN_KEY_HEADS * OMEGA_GDN_HEAD_DIM, gV);
        imadi(e, R(t1), R(one), 0xfffffffdu, T);
        imadw(e, base, t1, ROW, pqkv);
        imadw(e, aA[0], gA, 4u, base); imadw(e, aV[0], gV, 4u, base);
        for (unsigned r = 1; r < 3; r++) { imadw(e, aA[r], one, ROW, aA[r - 1]); imadw(e, aV[r], one, ROW, aV[r - 1]); }
        for (unsigned r = 0; r < 3; r++) { ldg32(e, xA[r], aA[r], udesc, r == 0); ldg32(e, xV[r], aV[r], udesc, 0); }
        for (unsigned r = 0; r < 3; r++) { fmul(e, R(xA[r]), R(xA[r]), R(insc)); fmul(e, R(xV[r]), R(xV[r]), R(insc)); }
        imadw(e, atA, tid, 16u, ptail); imadw(e, atA, h, OMEGA_GDN_TAIL_HEAD_BYTES, atA);
        imadi(e, R(t1), R(one), 256u, cc);
        imadw(e, atV, t1, 16u, ptail); imadw(e, atV, h, OMEGA_GDN_TAIL_HEAD_BYTES, atV);
        for (unsigned r = 0; r < 3; r++) { stg(e, atA, 4u * r, R(xA[r]), udesc); stg(e, atV, 4u * r, R(xV[r]), udesc); }
    }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_blackwell_codegen_gdn_post_prog(const OmegaGdnSpec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog || spec->copy || spec->mutant < 0 || spec->mutant > 4) return -1;
    OmegaGdnShape G;
    if (omega_gdn_shape(spec, &G)) return -1;
    const uint32_t RR = G.r, NC = 2u + G.r;
    (void)RR; (void)NC;
    const int MUT = spec->mutant;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;

    int udesc = UV64(e);
    int tid = V(e), t = V(e), cy = V(e), warp = V(e), lane = V(e), idx = V(e), ss = V(e), ts = V(e), g = V(e), t1 = V(e);
    int en = V(e), osc = V(e), insc = V(e), cinv = V(e), cnl2e = V(e), fone = V(e);
    int po = V64(e), pz = V64(e), ppar = V64(e), pout = V64(e), ao = V64(e), az = V64(e), an = V64(e), ay = V64(e);
    int oq = V128(e), zq = V128(e), nq = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    s2r(e, tid, BW_SR_TID_X); s2r(e, t, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    ldc64(e, po, OMEGA_GDN_POST_ARG_O); ldc64(e, pz, OMEGA_GDN_POST_ARG_Z); ldc64(e, ppar, OMEGA_GDN_POST_ARG_PAR);
    ldc64(e, pout, OMEGA_GDN_POST_ARG_OUT);
    ldc32(e, en, OMEGA_GDN_POST_ARG_EPS_NORM); ldc32(e, insc, OMEGA_GDN_POST_ARG_IN_SCALE); ldc32(e, osc, OMEGA_GDN_POST_ARG_OUT_SCALE);
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);
    /* row (token t, head 8 cy + warp): o, z and out share the [T][48][128] layout */
    imadi(e, R(idx), R(cy), 8u, warp);
    imadi(e, R(idx), R(t), G.vh, idx);
    imadw(e, ao, idx, OMEGA_GDN_HEAD_DIM * 4u, po); imadw(e, ao, lane, 16u, ao);
    imadw(e, az, idx, OMEGA_GDN_HEAD_DIM * 4u, pz); imadw(e, az, lane, 16u, az);
    imadw(e, ay, idx, OMEGA_GDN_HEAD_DIM * 4u, pout); imadw(e, ay, lane, 16u, ay);
    imadw(e, an, lane, 16u, ppar);
    ldg_wide(e, BW_IR_LDG_E_128, oq, ao, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_128, zq, az, 0, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, nq, an, G.par_norm, udesc, 0);
    fmul(e, R(ss), S(oq, 0), S(oq, 0));
    for (unsigned j = 1; j < 4; j++) ffma(e, R(ss), S(oq, j), S(oq, j), R(ss));
    bfly_sum(e, ss, ts, 1u, 16u);
    movi(e, cinv, fbits(1.0f / 128.0f));
    ffma(e, R(ss), R(ss), R(cinv), R(en));
    mufu(e, BW_IR_MUFU_RSQ, R(ss), R(ss));
    movi(e, cnl2e, fbits(-1.44269504f)); movi(e, fone, fbits(1.0f));
    for (unsigned j = 0; j < 4; j++) {
        fmul(e, R(g), S(zq, j), R(insc));
        if (MUT != 3) silu(e, R(g), t1, cnl2e, fone);
        fmul(e, S(oq, j), S(oq, j), R(ss)); fmul(e, S(oq, j), S(oq, j), S(nq, j));
        fmul(e, S(oq, j), S(oq, j), R(g)); fmul(e, S(oq, j), S(oq, j), R(osc));
    }
    for (unsigned j = 0; j < 4; j++) stg(e, ay, j * 4u, S(oq, j), udesc);
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

/* number of placeholders in a program, SHFL.BFLY and (bf16 state, L313) F2FP: what omega_tc_patch_code must report */
uint32_t omega_gdn_bfly_count(const BlackwellIRProgram *prog) {
    uint32_t n = 0;
    for (size_t i = 0; i < prog->count; i++)
        if ((prog->insns[i].op == BW_IR_SHFL_DOWN && (prog->insns[i].imm & OMEGA_TC_MARK_BFLY))
            || (prog->insns[i].op == BW_IR_FMNMX_MAX && prog->insns[i].imm == OMEGA_TC_MARK_F2FP)) n++;
    return n;
}

int omega_gdn_pf_prog_stats(unsigned kind, const OmegaGdnSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog;
    int rc = kind == 0 ? omega_blackwell_codegen_gdn_prep_prog(spec, &prog)
           : kind == 1 ? omega_blackwell_codegen_gdn_scan_prog(spec, &prog)
           : kind == 2 ? omega_blackwell_codegen_gdn_post_prog(spec, &prog) : -1;
    if (rc != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}

/* ======== multi-token decode (L369, see the header): T decode steps of one head in one CTA, state and tails on chip ========
 * Every token runs the decode program's phases A..E with the same instructions in the same order on the same values; the
 * changes are where values live: S stays in the 64 registers between tokens (one load before the loop, one store after
 * it), the conv tails are shifted in registers (one load, one store), and the loop-invariant loads (conv taps, negA and
 * dt_bias, the norm weight) are issued once. */
int omega_blackwell_codegen_gdn_multi_prog(const OmegaGdnSpec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog || spec->copy || spec->mutant < 0 || spec->mutant > 4) return -1;
    OmegaGdnShape G;
    if (omega_gdn_shape(spec, &G)) return -1;
    const uint32_t RR = G.r;
    const int MUT = spec->mutant;
    const uint32_t C_LOG2E = fbits(1.44269504f), C_NLOG2E = fbits(-1.44269504f), C_LN2 = fbits(0.693147181f);
    const uint32_t C_ONE = fbits(1.0f), C_M1 = fbits(-1.0f), C_INV128 = fbits(1.0f / 128.0f);
    const float qs = 0.0883883476f; /* 1 / sqrt(128) */
    omega_bw_ir_init(prog);
    if (omega_bw_ir_set_gpr_limit(prog, BW_PHYS_GPR_HW_LIMIT) != 0) return -1;
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;

    /* ---- identities; the state address; the loop bounds and strides ---- */
    int udesc = UV64(e);
    int tid = V(e), cx = V(e), cy = V(e), h = V(e), warp = V(e), lane = V(e), one = V(e);
    int ps = V64(e), sa = V64(e), toff = V(e), sq[16];
    int T = V(e), tm1 = V(e), t = V(e), xs = V(e), abs_ = V(e), os = V(e);
    for (unsigned r = 0; r < 16; r++) sq[r] = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, ps, OMEGA_GDN_ARG_STATE);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, one, 1);
    imadi(e, R(h), R(cy), RR, cx);
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);
    imadi(e, R(toff), R(h), OMEGA_GDN_STATE_HEAD_BYTES, NOREG);
    imadi(e, R(toff), R(warp), 16u * OMEGA_GDN_HEAD_DIM * 4u, toff);
    imadi(e, R(toff), R(lane), 16u, toff);
    imadw(e, sa, toff, 1u, ps);
    ldc32(e, T, OMEGA_GDN_MULTI_ARG_T); ldc32(e, xs, OMEGA_GDN_MULTI_ARG_XSTRIDE);
    ldc32(e, abs_, OMEGA_GDN_MULTI_ARG_ABSTRIDE); ldc32(e, os, OMEGA_GDN_MULTI_ARG_OSTRIDE);
    imadi(e, R(tm1), R(one), 0xffffffffu, T);

    /* ---- the loop-invariant loads: taps, tails, scalars, norm weight; the state ---- */
    int half = V(e), c = V(e), gi = V(e), slv = V(e);
    int pqkv = V64(e), ppar = V64(e), ptail = V64(e);
    int axA = V64(e), awA = V64(e), atA = V64(e), axV = V64(e), awV = V64(e), atV = V64(e);
    int wA = V128(e), wV = V128(e), tA = V128(e), tV = V128(e);
    int insc = V(e), t1 = V(e), cnl2e = V(e), fone = V(e), cl2e = V(e);
    int pab = V64(e), aab = V64(e), asc = V64(e), scq = V64(e);
    int pz = V64(e), az = V64(e), an = V64(e), nq = V128(e), pout = V64(e), ao = V64(e);
    int el2 = V(e), en = V(e), osc = V(e);
    if (e->err) return -1;
    ldc64(e, pqkv, OMEGA_GDN_ARG_QKV); ldc64(e, ppar, OMEGA_GDN_ARG_PAR); ldc64(e, ptail, OMEGA_GDN_ARG_TAIL);
    ldc32(e, insc, OMEGA_GDN_ARG_IN_SCALE);
    shr(e, half, tid, 7);
    andi(e, R(c), R(tid), 127u);
    imadi(e, R(gi), R(cy), OMEGA_GDN_HEAD_DIM, tid);              /* q channel hk*128 + t; k channel 2048 + hk*128 + (t - 128) */
    imadi(e, R(gi), R(half), 2048u - 128u, gi);
    imadw(e, axA, gi, 4u, pqkv); imadw(e, awA, gi, 16u, ppar);
    imadi(e, R(gi), R(h), OMEGA_GDN_HEAD_DIM, c);                 /* v channel 4096 + h*128 + c */
    imadi(e, R(gi), R(one), 2u * OMEGA_GDN_KEY_HEADS * OMEGA_GDN_HEAD_DIM, gi);
    imadw(e, axV, gi, 4u, pqkv); imadw(e, awV, gi, 16u, ppar);
    imadw(e, atA, tid, 16u, ptail); imadw(e, atA, h, OMEGA_GDN_TAIL_HEAD_BYTES, atA);
    imadi(e, R(slv), R(one), 256u, c);
    imadw(e, atV, slv, 16u, ptail); imadw(e, atV, h, OMEGA_GDN_TAIL_HEAD_BYTES, atV);
    ldc64(e, pab, OMEGA_GDN_ARG_AB);
    imadw(e, aab, h, 8u, pab); imadw(e, asc, h, 8u, ppar);
    ldc64(e, pz, OMEGA_GDN_ARG_Z);
    imadw(e, az, lane, 16u, pz); imadw(e, az, h, OMEGA_GDN_HEAD_DIM * 4u, az);
    imadw(e, an, lane, 16u, ppar);
    ldc64(e, pout, OMEGA_GDN_ARG_OUT);
    imadw(e, ao, lane, 16u, pout); imadw(e, ao, h, OMEGA_GDN_HEAD_DIM * 4u, ao);
    ldc32(e, el2, OMEGA_GDN_ARG_EPS_L2); ldc32(e, en, OMEGA_GDN_ARG_EPS_NORM); ldc32(e, osc, OMEGA_GDN_ARG_OUT_SCALE);
    ldg_wide(e, BW_IR_LDG_E_128, wA, awA, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_128, tA, atA, 0, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, wV, awV, 0, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, tV, atV, 0, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_64, scq, asc, G.par_scal, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_128, nq, an, G.par_norm, udesc, 0);
    for (unsigned r = 0; r < 16; r++) ldg_pf(e, sq[r], sa, r * OMEGA_GDN_HEAD_DIM * 4u, udesc, SB_STATE);
    movi(e, cnl2e, C_NLOG2E); movi(e, fone, C_ONE); movi(e, cl2e, C_LOG2E);
    consume(e, SB_STATE);
    movrz(e, R(t)); /* waits for every load above: the loop is entered with nothing pending */

    const int L = here(e);
    loop_begin(e);
    {
        /* ---- phase A: conv + SiLU of channel A (q or k) and channel V, token t ---- */
        int xA = V(e), xV = V(e), accA = V(e), accV = V(e);
        if (e->err) return -1;
        ldg32(e, xA, axA, udesc, 1);
        ldg32(e, xV, axV, udesc, 0);
        fmul(e, R(xA), R(xA), R(insc)); fmul(e, R(xV), R(xV), R(insc));
        fmul(e, R(accA), S(wA, 0), S(tA, 0)); ffma(e, R(accA), S(wA, 1), S(tA, 1), R(accA));
        ffma(e, R(accA), S(wA, 2), S(tA, 2), R(accA)); ffma(e, R(accA), S(wA, 3), R(xA), R(accA));
        fmul(e, R(accV), S(wV, 0), S(tV, 0)); ffma(e, R(accV), S(wV, 1), S(tV, 1), R(accV));
        ffma(e, R(accV), S(wV, 2), S(tV, 2), R(accV)); ffma(e, R(accV), S(wV, 3), R(xV), R(accV));
        silu(e, R(accA), t1, cnl2e, fone);
        silu(e, R(accV), t1, cnl2e, fone);

        /* ---- phase B: l2 norms; q|k and v to shared; the tails shift in registers ---- */
        int sqs = V(e), ts = V(e), ra = V(e);
        if (e->err) return -1;
        fmul(e, R(sqs), R(accA), R(accA));
        warp_sum(e, sqs, ts);
        imadi(e, R(ra), R(warp), 4u, NOREG); imadi(e, R(ra), R(one), OMEGA_GDN_SM_RED, ra);
        lane0_sts(e, lane, one, ra, R(sqs));
        imadi(e, R(ra), R(c), 4u, NOREG); imadi(e, R(ra), R(one), OMEGA_GDN_SM_V, ra);
        sts32(e, ra, R(accV));
        bar_sync(e);
        if (MUT != 4) {
            mov(e, S(tA, 0), S(tA, 1)); mov(e, S(tA, 1), S(tA, 2)); mov(e, S(tA, 2), R(xA));
            mov(e, S(tV, 0), S(tV, 1)); mov(e, S(tV, 1), S(tV, 2)); mov(e, S(tV, 2), R(xV));
        }
        {
            int rq = V128(e), hf = V(e), fq = V(e);
            if (e->err) return -1;
            imadi(e, R(ra), R(half), 16u, NOREG);
            lds128(e, rq, ra, OMEGA_GDN_SM_RED, 1);
            fadd(e, R(sqs), S(rq, 0), S(rq, 1)); fadd(e, R(sqs), R(sqs), S(rq, 2)); fadd(e, R(sqs), R(sqs), S(rq, 3));
            fadd(e, R(sqs), R(sqs), R(el2));
            mufu(e, BW_IR_MUFU_RSQ, R(sqs), R(sqs));
            i2fp(e, hf, half);
            movi(e, fq, fbits(qs)); movi(e, t1, fbits(1.0f - qs));
            ffma(e, R(fq), R(hf), R(t1), R(fq));
            fmul(e, R(accA), R(accA), R(sqs)); fmul(e, R(accA), R(accA), R(fq));
            imadi(e, R(ra), R(tid), 4u, NOREG);
            sts32(e, ra, R(accA));
            bar_sync(e);
        }

        /* ---- phase C: scalars of token t ---- */
        int abq = V64(e), xsc = V(e), mn = V(e), sp = V(e), dec = V(e), beta = V(e);
        if (e->err) return -1;
        ldg_wide(e, BW_IR_LDG_E_64, abq, aab, 0, udesc, 1);
        fadd(e, R(xsc), S(abq, 0), S(scq, 1));
        movi(e, mn, C_M1); fmul(e, R(mn), R(xsc), R(mn)); fminn(e, R(mn), R(xsc), R(mn));
        fmul(e, R(mn), R(mn), R(cl2e)); mufu(e, BW_IR_MUFU_EX2, R(mn), R(mn)); fadd(e, R(mn), R(mn), R(fone));
        mufu(e, BW_IR_MUFU_LG2, R(mn), R(mn));
        movrz(e, R(sp)); fmaxx(e, R(sp), R(xsc), R(sp));
        movi(e, t1, C_LN2); ffma(e, R(sp), R(mn), R(t1), R(sp));
        if (MUT == 1) movi(e, dec, C_ONE);
        else { fmul(e, R(dec), S(scq, 0), R(sp)); fmul(e, R(dec), R(dec), R(cl2e)); mufu(e, BW_IR_MUFU_EX2, R(dec), R(dec)); }
        if (MUT == 2) movi(e, beta, C_ONE);
        else {
            fmul(e, R(beta), S(abq, 1), R(cnl2e)); mufu(e, BW_IR_MUFU_EX2, R(beta), R(beta));
            fadd(e, R(beta), R(beta), R(fone)); mufu(e, BW_IR_MUFU_RCP, R(beta), R(beta));
        }

        /* ---- phase D: the state update (S stays in registers) ---- */
        int aw = V(e), pa = V(e), la = V(e), kq[4], kv[4], dl[4], vq = V128(e), zq = V128(e);
        for (unsigned i = 0; i < 4; i++) { kq[i] = V128(e); kv[i] = V(e); dl[i] = V(e); }
        if (e->err) return -1;
        imadi(e, R(aw), R(warp), 16u * 4u, NOREG);
        imadi(e, R(pa), R(warp), OMEGA_GDN_HEAD_DIM * 4u, NOREG); imadi(e, R(pa), R(lane), 16u, pa); imadi(e, R(pa), R(one), OMEGA_GDN_SM_P, pa);
        imadi(e, R(la), R(lane), 16u, NOREG);
        for (unsigned r = 0; r < 16; r++) for (unsigned j = 0; j < 4; j++) fmul(e, S(sq[r], j), S(sq[r], j), R(dec));
        for (unsigned i = 0; i < 4; i++) lds128(e, kq[i], aw, OMEGA_GDN_SM_K + i * 16u, i == 0);
        for (unsigned j = 0; j < 4; j++) fmul(e, R(kv[j]), S(sq[0], j), S(kq[0], 0));
        for (unsigned r = 1; r < 16; r++) for (unsigned j = 0; j < 4; j++) ffma(e, R(kv[j]), S(sq[r], j), S(kq[r / 4], r % 4), R(kv[j]));
        cross_warp_sum(e, kv, pa, one, la);
        lds128(e, vq, la, OMEGA_GDN_SM_V, 1);
        for (unsigned j = 0; j < 4; j++) { fsub(e, R(dl[j]), S(vq, j), R(kv[j])); fmul(e, R(dl[j]), R(dl[j]), R(beta)); }
        {
            int k2[4];
            for (unsigned i = 0; i < 4; i++) k2[i] = V128(e);
            if (e->err) return -1;
            for (unsigned i = 0; i < 4; i++) lds128(e, k2[i], aw, OMEGA_GDN_SM_K + i * 16u, i == 0);
            for (unsigned r = 0; r < 16; r++) for (unsigned j = 0; j < 4; j++) ffma(e, S(sq[r], j), S(k2[r / 4], r % 4), R(dl[j]), S(sq[r], j));
        }
        ldg_pf(e, zq, az, 0, udesc, SB_GATE);
        int p[4];
        {
            int q2[4];
            for (unsigned i = 0; i < 4; i++) { q2[i] = V128(e); p[i] = V(e); }
            if (e->err) return -1;
            for (unsigned i = 0; i < 4; i++) lds128(e, q2[i], aw, OMEGA_GDN_SM_Q + i * 16u, i == 0);
            for (unsigned j = 0; j < 4; j++) fmul(e, R(p[j]), S(sq[0], j), S(q2[0], 0));
            for (unsigned r = 1; r < 16; r++) for (unsigned j = 0; j < 4; j++) ffma(e, R(p[j]), S(sq[r], j), S(q2[r / 4], r % 4), R(p[j]));
        }
        bar_sync(e);
        cross_warp_sum(e, p, pa, one, la);

        /* ---- phase E: gated RMSNorm, out row t ---- */
        int ss = V(e), g = V(e);
        if (e->err) return -1;
        fmul(e, R(ss), R(p[0]), R(p[0]));
        for (unsigned j = 1; j < 4; j++) ffma(e, R(ss), R(p[j]), R(p[j]), R(ss));
        warp_sum(e, ss, ts);
        imadi(e, R(ra), R(warp), 4u, NOREG); imadi(e, R(ra), R(one), OMEGA_GDN_SM_RED2, ra);
        lane0_sts(e, lane, one, ra, R(ss));
        bar_sync(e);
        lds32(e, R(ss), ra, 0);
        movi(e, t1, C_INV128);
        ffma(e, R(ss), R(ss), R(t1), R(en));
        mufu(e, BW_IR_MUFU_RSQ, R(ss), R(ss));
        consume(e, SB_GATE);
        for (unsigned j = 0; j < 4; j++) {
            fmul(e, R(g), S(zq, j), R(insc));
            if (MUT != 3) silu(e, R(g), t1, cnl2e, fone);
            fmul(e, R(p[j]), R(p[j]), R(ss)); fmul(e, R(p[j]), R(p[j]), S(nq, j));
            fmul(e, R(p[j]), R(p[j]), R(g)); fmul(e, R(p[j]), R(p[j]), R(osc));
        }
        for (unsigned j = 0; j < 4; j++) stg(e, ao, j * 4u, R(p[j]), udesc);
        /* every warp is done with this token's shared memory before the next token writes it */
        bar_sync(e);
        imadw(e, axA, xs, 1u, axA); imadw(e, axV, xs, 1u, axV); imadw(e, az, xs, 1u, az);
        imadw(e, aab, abs_, 1u, aab); imadw(e, ao, os, 1u, ao);
        imadi(e, R(t), R(one), 1u, t);
        isetp_ge_u32(e, tm1, t);
        bra_if_p0(e, L);
    }
    loop_end(e);

    /* ---- the state and the tails back to memory, once ---- */
    for (unsigned r = 0; r < 16; r++) for (unsigned j = 0; j < 4; j++) stg(e, sa, r * OMEGA_GDN_HEAD_DIM * 4u + j * 4u, S(sq[r], j), udesc);
    if (MUT != 4) {
        stg(e, atA, 0, S(tA, 0), udesc); stg(e, atA, 4, S(tA, 1), udesc); stg(e, atA, 8, S(tA, 2), udesc);
        stg(e, atV, 0, S(tV, 0), udesc); stg(e, atV, 4, S(tV, 1), udesc); stg(e, atV, 8, S(tV, 2), udesc);
    }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_gdn_multi_prog_stats(const OmegaGdnSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_gdn_multi_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}
