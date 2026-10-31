/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* omega_blackwell_codegen_w4.c: NVFP4 weight-only grouped GEMV/GEMM for decode (P-DRA-4 DRA-71). See the header for the
 * contract. The emitter helpers are copies of omega_blackwell_codegen_gemm.c's (same chip-learned scheduling rules). */
#include "omega_blackwell_codegen_w4.h"
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
} Em;

static void emit(Em *e, BlackwellIRInsn in, int kind) {
    uint32_t wait = e->pending;
    switch (kind) {
    case K_FIXED: in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_PRED:  in.control = CW(14, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    /* Loads read their address operands late (the read barrier SB_RD): a later instruction must not overwrite an
     * in-flight load's address register. Every load sets SB_RD and the next load of a batch waits on it (cheap:
     * the operand read, not the data). Found on the chip: the deep k=2048 kernel hung when the allocator reused a
     * dead scale address register as the next load's destination (evidence OWN30B-FP8-GEMV-c4006391). */
    case K_VAR:   in.control = CW(4, SB_GPR, SB_RD, wait) | CW_YIELD; e->pending = (1u << SB_GPR) | (1u << SB_RD); break;
    /* a further load of the same group: waits on nothing, joins the counting scoreboard */
    case K_VAR_MORE: in.control = CW(4, SB_GPR, SB_RD, (e->pending & (1u << SB_RD))) | CW_YIELD; e->pending |= (1u << SB_GPR) | (1u << SB_RD); break;
    case K_UVAR:  in.control = CW(4, SB_UGPR, SB_NONE, wait) | CW_YIELD; e->pending = 1u << SB_UGPR; break;
    case K_STORE: in.control = CW(4, SB_NONE, SB_RD, wait) | CW_YIELD; e->pending = 1u << SB_RD; break;
    case K_SYNC:  in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break; /* BAR.SYNC after the STS: the attention kernel's chip-proven pattern */
    case K_EXIT:  in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    case K_BRA:   in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    case K_BSSY:  in.control = CW(1, SB_NONE, SB_NONE, wait) | CW_YIELD; e->pending = 0; break;
    case K_BSYNC: in.control = CW(5, SB_NONE, SB_NONE, wait) | CW_YIELD; e->pending = 0; break;
    default: e->err = -1; return;
    }
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}

static BlackwellIRInsn I0(BlackwellIROpcode op) {
    BlackwellIRInsn in; memset(&in, 0, sizeof in);
    in.op = op; in.dst_vreg = NOREG; in.src1_vreg = NOREG; in.src2_vreg = NOREG; in.src3_vreg = NOREG; in.ureg = NOREG;
    return in;
}
static int V(Em *e) { int v = omega_bw_ir_alloc_vreg(e->p); if (v < 0) e->err = -1; return v; }
static int V64(Em *e) { int v = omega_bw_ir_alloc_vreg64(e->p); if (v < 0) e->err = -1; return v; }
static int V128(Em *e) { int v = omega_bw_ir_alloc_vreg128(e->p); if (v < 0) e->err = -1; return v; }
static int UV64(Em *e) { int v = omega_bw_ir_alloc_uvreg64(e->p); if (v < 0) e->err = -1; return v; }
static int here(const Em *e) { return (int)e->p->count; }

static void ldcu64(Em *e, int ud, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDCU64); i.dst_vreg = ud; i.imm = off; i.is_uniform = true; emit(e, i, K_UVAR); }
static void ldc64(Em *e, int d, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDC64); i.dst_vreg = d; i.imm = off; emit(e, i, K_VAR); }
static void ldc32(Em *e, int d, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDC); i.dst_vreg = d; i.imm = off; emit(e, i, K_VAR); }
static void s2r(Em *e, int d, uint32_t sr) { BlackwellIRInsn i = I0(BW_IR_S2R); i.dst_vreg = d; i.imm = sr; emit(e, i, K_VAR); }
static void movi(Em *e, int d, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_MOV_IMM); i.dst_vreg = d; i.imm = imm; emit(e, i, K_FIXED); }
static void movrz(Em *e, int d) { BlackwellIRInsn i = I0(BW_IR_MOV_RZ); i.dst_vreg = d; emit(e, i, K_FIXED); }
/* d = a * imm + c (c NOREG = RZ); a may be a sub-register of a quad */
static void imadi(Em *e, int d, int a, unsigned asub, uint32_t imm, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.imm = imm; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* d = a * b + c (registers) */
static void imadr(Em *e, int d, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* d64 = a * imm + c64 */
static void imadw(Em *e, int d64, int a, uint32_t imm, int c64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = d64; i.src1_vreg = a; i.imm = imm; i.src3_vreg = c64; emit(e, i, K_FIXED); }
static void shr(Em *e, int d, int a, unsigned asub, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.imm = imm; emit(e, i, K_FIXED); }
static void andi(Em *e, int d, int a, unsigned asub, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.imm = imm; emit(e, i, K_FIXED); }
static void ldg_wide(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, int first) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE);
}
static void ldg32(Em *e, int d, int addr64, int udesc, int first) { BlackwellIRInsn i = I0(BW_IR_LDG_E); i.dst_vreg = d; i.src1_vreg = addr64; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void stg(Em *e, int addr64, uint32_t imm, int val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.src2_vreg = val; i.imm = imm; i.ureg = udesc; emit(e, i, K_STORE); }
static void lds128(Em *e, int d128, int addr, uint32_t imm, int first) { BlackwellIRInsn i = I0(BW_IR_LDS128); i.dst_vreg = d128; i.src1_vreg = addr; i.imm = imm; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void sts32(Em *e, int addr, int val) { BlackwellIRInsn i = I0(BW_IR_STS32); i.src1_vreg = addr; i.src2_vreg = val; emit(e, i, K_STORE); }
static void bar_sync(Em *e) { emit(e, I0(BW_IR_BAR_SYNC), K_SYNC); }
static void fadd(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FADD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
/* d = a * b[bsub] + c */
static void ffma(Em *e, int d, int a, int b, unsigned bsub, int c) { BlackwellIRInsn i = I0(BW_IR_FFMA); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void shfl_down(Em *e, int d, int a, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_SHFL_DOWN); i.dst_vreg = d; i.src1_vreg = a; i.imm = off; emit(e, i, K_VAR); }
static void isetp_ge_u32(Em *e, int a, int b) { BlackwellIRInsn i = I0(BW_IR_ISETP_GE_U32); i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_PRED); }
static void bra_if_p0(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
static void bra(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
/* a forward branch whose target is patched when known: returns the instruction index */
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

/* Loops: a value defined before the loop and read inside it stays live until the back-branch
 * (the linear-scan allocator knows no back edges; same rule as the elementwise emitter). */
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


/* A weight/scale prefetch load: scoreboard `sb` (3 or 4) is held, not waited for, until consume() */
static void ldw_pf(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, unsigned sb) {
    BlackwellIRInsn in = I0(op); in.dst_vreg = d; in.src1_vreg = addr64; in.imm = imm; in.ureg = udesc;
    in.control = CW(4, sb, SB_RD, (e->pending & (1u << SB_RD))) | CW_YIELD;
    e->pending |= 1u << SB_RD; e->held |= 1u << sb;
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}
static void consume(Em *e, unsigned sb) { e->pending |= e->held & (1u << sb); e->held &= ~(1u << sb); }

static int log2u(uint32_t v) { int l = 0; while ((1u << l) < v) l++; return (1u << l) == v ? l : -1; }

static int spec_ok(const OmegaW4GemmSpec *s) {
    if (!s) return 0;
    if (s->lane_bytes != 8 && s->lane_bytes != 16) return 0;
    if (s->mt == 0 || (s->mt & (s->mt - 1)) || s->mt * 256u * s->lane_bytes > OMEGA_GEMM_MAX_SHARED) return 0;
    if (s->warps < 1 || s->warps > 32 || (s->warps & (s->warps - 1)) || (2u * s->mt * s->lane_bytes) % s->warps) return 0;
    if (s->k == 0 || s->k > 16384u || s->k % (64u * s->lane_bytes)) return 0;
    uint32_t ks = s->k / (64u * s->lane_bytes);
    if (ks % 2u && ks > 3u) return 0; /* odd step counts are fully unrolled: at most 3 */
    if (s->mutant < 0 || s->mutant > OMEGA_W4GEMM_MUTANTS) return 0;
    if (s->rowmap > 1u || (s->rowmap && s->mt > 4u)) return 0;
    if (s->rowmap && 2u * s->mt * s->lane_bytes / s->warps > 16u) return 0; /* NJ row bases in registers */
    return 1;
}
uint32_t omega_w4gemm_steps(const OmegaW4GemmSpec *spec) { return spec_ok(spec) ? spec->k / (64u * spec->lane_bytes) : 0; }
uint32_t omega_w4gemm_shared_bytes(const OmegaW4GemmSpec *spec) { return spec_ok(spec) ? spec->mt * 256u * spec->lane_bytes : 0u; }

void omega_w4gemm_permute_x(const float *x, uint32_t k, uint32_t lb, float *out) {
    const uint32_t el = 2u * lb, row = 32u * el, nq = el / 4u;
    for (uint32_t s0 = 0; s0 + row <= k; s0 += row)
        for (uint32_t L = 0; L < 32u; L++)
            for (uint32_t q = 0; q < nq; q++) {
                const float *src = x + s0 + L * el + q * 4u; float *dst = out + s0 + (q * 32u + L) * 4u;
                for (unsigned e = 0; e < 4; e++) dst[e] = src[e] * 18446744073709551616.0f;
            }
}

/* e4m3 byte i of word w -> f32 bits (value * 2^-120): ((b & 0x7F) << 20) | ((b & 0x80) << 24) */
static void decode_byte(Em *e, int f, int wq, unsigned word, unsigned i, int u, int a, int s) {
    const uint32_t M = 0x07F00000u;
    if (i < 3) imadi(e, u, wq, word, 1u << (20u - 8u * i), NOREG);
    else shr(e, u, wq, word, 4);
    andi(e, a, u, 0, M);
    andi(e, s, wq, word, 0x80u << (8u * i));
    if (i < 3) imadi(e, f, s, 0, 1u << (24u - 8u * i), a);
    else imadi(e, f, s, 0, 1u, a);
}
/* e2m1 nibble j of word w -> f32 bits (value * 2^-126): magnitude bits 4j..4j+2 to 22..24, sign bit 4j+3 to 31 */
static void decode_nib(Em *e, int f, int wq, unsigned word, unsigned j, int u, int a, int s) {
    if (j <= 5) imadi(e, u, wq, word, 1u << (22u - 4u * j), NOREG);
    else shr(e, u, wq, word, 4u * j - 22u);
    andi(e, a, u, 0, 0x01C00000u);
    andi(e, s, wq, word, 8u << (4u * j));
    imadi(e, f, s, 0, 1u << (28u - 4u * j), a);
}

int omega_blackwell_codegen_w4gemm_prog(const OmegaW4GemmSpec *spec, BlackwellIRProgram *prog) {
    if (!spec_ok(spec) || !prog) return -1;
    const uint32_t K = spec->k, LB = spec->lane_bytes, MT = spec->mt, W = spec->warps, EL = 2u * LB;
    const uint32_t KS = K / (32u * EL), BPL = LB / 8u, SADV = 4u * LB; /* 16-blocks per lane per step, scale bytes per step */
    const uint32_t U = KS % 2u == 0 ? 2u : KS;
    const uint32_t NJ = MT * 2u * LB / W, ROWW = 32u * EL, T = 32u * W;
    const int log2mt = log2u(MT), log2roww = log2u(ROWW), log2lpw = log2u(32u / LB); /* lanes sharing a scale word */
    const BlackwellIROpcode ldw = LB == 16 ? BW_IR_LDG_E_128 : BW_IR_LDG_E_64;
    const int M = spec->mutant;
    if (log2mt < 0 || log2roww < 0 || log2lpw < 0) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    Em *e = &em;

    /* ---- prologue ---- */
    int udesc = UV64(e), pseg = V64(e), px = V64(e), py = V64(e), nreg = V(e), stride = V(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), one = V(e), tent = V64(e);
    int wbase = V64(e), sbase = V64(e), rc = V64(e), gbase = V64(e), row0 = V(e), cnt = V(e), tpass = V(e), passes = V(e);
    int wlane = V(e), slane = V(e), xl = V(e), nm1 = V(e), ysn = V(e), mrow = V(e), off4 = V(e), sa0 = V(e), base = V(e), z = V(e);
    int msel = V(e), b0 = V(e), b1 = V(e), c120 = V(e), zr = V(e);
    const int RM = spec->rowmap ? 1 : 0;
    int pxm = RM ? V64(e) : NOREG, pym = RM ? V64(e) : NOREG;
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pseg, OMEGA_GEMM_ARG_SEG); ldc64(e, px, OMEGA_GEMM_ARG_X); ldc64(e, py, OMEGA_GEMM_ARG_Y);
    ldc32(e, nreg, OMEGA_GEMM_ARG_N); ldc32(e, stride, OMEGA_GEMM_ARG_STRIDE);
    if (RM) { ldc64(e, pxm, OMEGA_GEMM_ARG_XMAP); ldc64(e, pym, OMEGA_GEMM_ARG_YMAP); }
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, one, 1);
    shr(e, warp, tid, 0, 5);
    andi(e, lane, tid, 0, 31);
    imadw(e, tent, cy, OMEGA_W4GEMM_SEG_BYTES, pseg);
    ldg_wide(e, BW_IR_LDG_E_64, wbase, tent, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_64, sbase, tent, 8, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_64, rc, tent, 16, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_64, gbase, tent, 24, udesc, 0);
    imadi(e, row0, rc, 0, 1, NOREG); imadi(e, cnt, rc, 1, 1, NOREG);
    imadi(e, tpass, one, 0, MT - 1u, cnt); shr(e, passes, tpass, 0, (uint32_t)log2mt);
    int bra_empty = -1;
    if (M != 4) { /* an empty segment leaves before the loops (mutant 4 drops the guard) */
        imadi(e, z, one, 0, 0xffffffffu, cnt); shr(e, z, z, 0, 31);
        isetp_ge_u32(e, z, one);
        bra_empty = bra_if_p0_fwd(e);
    }
    imadi(e, wlane, lane, 0, LB, NOREG);
    shr(e, slane, lane, 0, (uint32_t)log2lpw); imadi(e, slane, slane, 0, 4, NOREG);
    /* msel: the lane's scale byte(s) to byte 3 (LB 8) or bytes 2, 3 (LB 16) */
    andi(e, b0, lane, 0, 1); imadi(e, b0, b0, 0, 0xffffffffu, one);               /* 1 - (L & 1) */
    if (LB == 16) imadi(e, msel, b0, 0, 65535u, one);                              /* 2^16 or 1 */
    else {
        imadi(e, msel, b0, 0, 255u, one);                                         /* 2^8 or 1 */
        shr(e, b1, lane, 0, 1); andi(e, b1, b1, 0, 1); imadi(e, b1, b1, 0, 0xffffffffu, one);
        imadi(e, b1, b1, 0, 65535u, one); imadr(e, msel, msel, b1, NOREG);        /* 2^(24 - 8 (L & 3)) */
    }
    movi(e, c120, 0x7B800000u); movrz(e, zr);
    imadi(e, xl, lane, 0, 16, NOREG);
    imadi(e, nm1, one, 0, 0xffffffffu, nreg);
    imadi(e, ysn, nreg, 0, 4, NOREG);
    shr(e, mrow, tid, 0, (uint32_t)log2roww);
    andi(e, off4, tid, 0, ROWW - 1u); imadi(e, off4, off4, 0, 4, NOREG);
    imadi(e, sa0, tid, 0, 4, NOREG);
    imadi(e, base, cx, 0, W, NOREG);

    int rg = V(e), over = V(e), t1 = V(e), d = V(e), tmp = V(e), rgc = V(e), lo = V(e), gr = V(e);
    int pw = V64(e), ps = V64(e), pyrg = V64(e), pg = V64(e), rowp = V(e), rem = V(e), pc = V(e);
    int xrp = V64(e), xs = V64(e), pwc = V64(e), psc = V64(e), cntk = V(e), pyr = V64(e), pya = V64(e), t = V(e), tq = V(e), fl = V(e), sv = V(e);
    int xb[16], yi[16], xso = NOREG, ma = NOREG;
    if (RM) { if (NJ > 16u) return -1; for (uint32_t j = 0; j < NJ; j++) xb[j] = V64(e); for (uint32_t m = 0; m < MT; m++) yi[m] = V(e); xso = V(e); ma = V64(e); }
    int acc[8], dot[8], xq[4], wq[2], sc[2], fr[8], u = V(e), a = V(e), sg = V(e), onef = V(e), sw = V(e), sraw = V(e), scf = V(e), yv = V(e);
    for (uint32_t m = 0; m < MT; m++) { acc[m] = V(e); dot[m] = V(e); }
    for (int q = 0; q < 4; q++) xq[q] = V128(e);
    for (int b = 0; b < 2; b++) { wq[b] = LB == 16 ? V128(e) : V64(e); sc[b] = V(e); }
    for (int i = 0; i < 8; i++) fr[i] = V(e);
    if (e->err) return -1;
    movi(e, onef, 0x3f800000u);

    /* ---- weight-row loop ---- */
    const int L_rg = here(e);
    loop_begin(e);
    isetp_ge_u32(e, base, nreg);
    const int bra_end = bra_if_p0_fwd(e);
    imadi(e, rg, base, 0, 1, warp);
    imadi(e, t1, rg, 0, 0xffffffffu, nm1); shr(e, over, t1, 0, 31);
    imadi(e, d, nm1, 0, 0xffffffffu, rg);
    imadr(e, tmp, over, d, NOREG); imadi(e, rgc, tmp, 0, 0xffffffffu, rg);
    imadw(e, pw, rgc, K / 2u, wbase); imadw(e, pw, wlane, 1, pw);
    imadw(e, ps, rgc, K / 16u, sbase); imadw(e, ps, slane, 1, ps);
    imadw(e, pg, rgc, 4, gbase); ldg32(e, gr, pg, udesc, 1);
    imadw(e, pyrg, rg, 4, py);
    imadi(e, lo, over, 0, 1, lane);
    imadi(e, rowp, row0, 0, 1, NOREG); imadi(e, rem, cnt, 0, 1, NOREG); imadi(e, pc, passes, 0, 1, NOREG);

    /* ---- pass loop ---- */
    const int L_pass = here(e);
    loop_begin(e);
    imadi(e, t, rowp, 0, 1, mrow);
    if (!RM) { imadw(e, xrp, t, K * 4u, px); imadw(e, xrp, off4, 1, xrp); imadw(e, xs, one, 0, xrp); }
    else {
        for (uint32_t j = 0; j < NJ; j++) { imadi(e, tq, one, 0, (j * T) / ROWW, t); imadw(e, ma, tq, 4u, pxm); ldg32(e, xb[j], ma, udesc, j == 0); }
        for (uint32_t j = 0; j < NJ; j++) {
            const uint32_t pos = j * T;
            imadw(e, xb[j], xb[j], K * 4u, px); imadw(e, xb[j], off4, 1, xb[j]); imadw(e, xb[j], one, (pos % ROWW) * 4u, xb[j]);
        }
        movi(e, xso, 0);
    }
    imadw(e, pwc, one, 0, pw); imadw(e, psc, one, 0, ps);
    for (uint32_t m = 0; m < MT; m++) movrz(e, acc[m]);
    ldw_pf(e, ldw, wq[0], pwc, 0, udesc, 3);
    ldw_pf(e, BW_IR_LDG_E, sc[0], psc, 0, udesc, 3);
    movi(e, cntk, KS / U);

    /* ---- K loop ---- */
    const int L_k = here(e);
    loop_begin(e);
    for (uint32_t us = 0; us < U; us++) {
        const int cur = (int)(us & 1u), nxt = 1 - cur;
        const unsigned sbc = cur ? 4u : 3u, sbn = nxt ? 4u : 3u;
        const int last = us + 1 == U, prefetch = !last || KS > U;
        int pwn = -1, psn = -1;
        if (prefetch) { pwn = V64(e); psn = V64(e); }
        int sa = V(e);
        if (e->err) return -1;
        if (prefetch && !last) {
            imadw(e, pwn, one, (us + 1) * 32u * LB, pwc); imadw(e, psn, one, (us + 1) * SADV, psc);
        } else if (prefetch) {
            int nf = V(e), inc = V(e), inc2 = V(e);
            if (e->err) return -1;
            imadi(e, nf, one, 0, 0xfffffffeu, cntk);
            shr(e, nf, nf, 0, 31);
            imadi(e, nf, nf, 0, 0xffffffffu, one);
            imadi(e, inc, nf, 0, U * 32u * LB, NOREG); imadi(e, inc2, nf, 0, U * SADV, NOREG);
            imadw(e, pwn, inc, 1, pwc); imadw(e, psn, inc2, 1, psc);
        }
        for (uint32_t j0 = 0; j0 < NJ; j0 += 4) {
            const uint32_t nj = NJ - j0 < 4 ? NJ - j0 : 4;
            int xv[4], xa[4];
            for (uint32_t q = 0; q < nj; q++) { xv[q] = V(e); xa[q] = V64(e); }
            if (e->err) return -1;
            for (uint32_t q = 0; q < nj; q++) { const uint32_t pos = (j0 + q) * T; if (RM) imadw(e, xa[q], xso, 1u, xb[j0 + q]); else imadw(e, xa[q], one, ((pos / ROWW) * K + pos % ROWW) * 4u, xs); }
            for (uint32_t q = 0; q < nj; q++) ldg32(e, xv[q], xa[q], udesc, q == 0);
            if (prefetch && j0 == 0) {
                ldw_pf(e, ldw, wq[nxt], pwn, 0, udesc, sbn);
                ldw_pf(e, BW_IR_LDG_E, sc[nxt], psn, 0, udesc, sbn);
            }
            for (uint32_t q = 0; q < nj; q++) { imadi(e, sa, one, 0, (j0 + q) * T * 4u, sa0); sts32(e, sa, xv[q]); }
        }
        bar_sync(e);
        consume(e, sbc);
        imadr(e, sw, sc[cur], msel, NOREG);
        for (uint32_t b = 0; b < BPL; b++) {
            for (uint32_t m = 0; m < MT; m++) movrz(e, dot[m]);
            for (uint32_t ww = 0; ww < 2; ww++) {
                const uint32_t word = 2u * b + ww;
                for (unsigned j = 0; j < 8; j++) decode_nib(e, fr[j], wq[cur], word, M == 5 ? j ^ 1u : j, u, a, sg);
                for (uint32_t m0 = 0; m0 < MT; m0 += 2) {
                    const uint32_t nb = MT - m0 < 2 ? MT - m0 : 2;
                    for (uint32_t q = 0; q < nb; q++) {
                        const uint32_t m = m0 + q, mm = M == 3 ? (m ^ 1u) % MT : m;
                        for (uint32_t h = 0; h < 2; h++) lds128(e, xq[2 * q + h], xl, mm * ROWW * 4u + (2u * word + h) * 512u, q == 0 && h == 0);
                    }
                    for (uint32_t q = 0; q < nb; q++) for (unsigned j = 0; j < 8; j++) ffma(e, dot[m0 + q], fr[j], xq[2 * q + j / 4u], j % 4u, dot[m0 + q]);
                }
            }
            /* the block's scale: byte 2 + b (LB 16) or 3 (LB 8) of sw; mutant 2 takes the neighbouring block's byte */
            const unsigned bi = LB == 16 ? 2u + (M == 2 ? b ^ 1u : b) : (M == 2 ? 2u : 3u);
            decode_byte(e, sraw, sw, 0, bi, u, a, sg);
            ffma(e, scf, sraw, c120, 0, zr);
            for (uint32_t m = 0; m < MT; m++) ffma(e, acc[m], dot[m], M == 1 ? onef : scf, 0, acc[m]);
        }
        bar_sync(e);
        if (RM) imadi(e, xso, one, 0, ROWW * 4u, xso); else imadw(e, xs, one, ROWW * 4u, xs);
    }
    imadw(e, pwc, one, U * 32u * LB, pwc);
    imadw(e, psc, one, U * SADV, psc);
    imadi(e, cntk, one, 0, 0xffffffffu, cntk);
    isetp_ge_u32(e, cntk, one);
    bra_if_p0(e, L_k);
    loop_end(e);
    consume(e, 3); consume(e, 4);

    /* ---- reduce, apply G', store ---- */
    for (uint32_t m = 0; m < MT; m++)
        for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, acc[m], o); fadd(e, acc[m], acc[m], t); }
    imadr(e, tq, rowp, ysn, NOREG); imadw(e, pyr, tq, 1, pyrg);
    if (RM) for (uint32_t m = 0; m < MT; m++) { imadi(e, tq, one, 0, m, rowp); imadw(e, ma, tq, 4u, pym); ldg32(e, yi[m], ma, udesc, m == 0); }
    for (uint32_t m = 0; m < MT; m++) {
        if (M == 6) imadi(e, yv, acc[m], 0, 1, NOREG); else ffma(e, yv, acc[m], gr, 0, zr);
        imadi(e, tq, one, 0, (uint32_t)(0u - 1u - m), rem); shr(e, fl, tq, 0, 31);
        imadi(e, sv, fl, 0, 1, lo);
        isetp_ge_u32(e, sv, one);
        const int at_bssy = bssy_fwd(e);
        const int bra_skip = bra_if_p0_fwd(e);
        if (RM) { imadr(e, tq, yi[m], ysn, NOREG); imadw(e, pya, tq, 1, pyrg); } else imadw(e, pya, ysn, m, pyr);
        stg(e, pya, 0, yv, udesc);
        const int at_bsync = here(e);
        bsync(e);
        patch_bra(e, bra_skip, at_bsync);
        patch_bra(e, at_bssy, at_bsync + 1);
    }
    imadi(e, rowp, one, 0, MT, rowp); imadi(e, rem, one, 0, (uint32_t)(0u - MT), rem);
    imadi(e, pc, one, 0, 0xffffffffu, pc);
    isetp_ge_u32(e, pc, one);
    bra_if_p0(e, L_pass);
    loop_end(e);

    imadi(e, base, stride, 0, 1, base);
    bra(e, L_rg);
    loop_end(e);
    const int L_end = here(e);
    patch_bra(e, bra_end, L_end);
    if (bra_empty >= 0) patch_bra(e, bra_empty, L_end);
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_w4gemm_prog_stats(const OmegaW4GemmSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_w4gemm_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}
