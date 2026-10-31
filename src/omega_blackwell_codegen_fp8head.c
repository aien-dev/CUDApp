/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* omega_blackwell_codegen_fp8head.c: FP8 lm_head GEMV over up to 8 x rows per launch (aien-dev/CUDApp#217).
 * See the header for the contract and the argument words.
 *
 * Scheduling and emitter helpers are omega_blackwell_codegen_moe.c's (copied, static): every instruction waits on
 * everything still pending; variable-latency results go through scoreboard 0, loads set the read barrier 1, uniform
 * loads scoreboard 2; fixed-latency producers stall 6; ISETP stalls 14 before a predicated branch. A K step issues its
 * weight, scale and x row 0 loads back to back (one round trip), decodes the weights once, then runs x row 0's FFMA
 * chains; every further x row is one more batch of 4 LDG.128 followed by its FFMA chains. One K step per loop
 * iteration: at rows 2, xb 8 a step is ~460 instructions, inside BW_MAX_IR_INSNS (1024) with the reduction tail.
 */
#include "omega_blackwell_codegen_fp8head.h"
#include "omega_blackwell_codegen_moe.h"
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

enum { K_FIXED, K_PRED, K_VAR, K_VAR_MORE, K_UVAR, K_STORE, K_EXIT, K_BRA, K_BSSY, K_BSYNC };

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
    /* loads read their address operands late (SB_RD): the next load of a batch waits on the read barrier (moe.c) */
    case K_VAR:   in.control = CW(4, SB_GPR, SB_RD, wait) | CW_YIELD; e->pending = (1u << SB_GPR) | (1u << SB_RD); break;
    case K_VAR_MORE: in.control = CW(4, SB_GPR, SB_RD, (e->pending & (1u << SB_RD))) | CW_YIELD; e->pending |= (1u << SB_GPR) | (1u << SB_RD); break;
    case K_UVAR:  in.control = CW(4, SB_UGPR, SB_NONE, wait) | CW_YIELD; e->pending = 1u << SB_UGPR; break;
    case K_STORE: in.control = CW(4, SB_NONE, SB_RD, wait) | CW_YIELD; e->pending = 1u << SB_RD; break;
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
/* d64 = a * imm + c64 (IMAD.WIDE.U32) */
static void imadw(Em *e, int d64, int a, uint32_t imm, int c64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = d64; i.src1_vreg = a; i.imm = imm; i.src3_vreg = c64; emit(e, i, K_FIXED); }
static void shr(Em *e, int d, int a, unsigned asub, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.imm = imm; emit(e, i, K_FIXED); }
static void andi(Em *e, int d, int a, unsigned asub, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.imm = imm; emit(e, i, K_FIXED); }
static void xorr(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_LOP3_XOR); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void ldg128(Em *e, int d, int addr64, uint32_t imm, int udesc, int first) {
    BlackwellIRInsn i = I0(BW_IR_LDG_E_128); i.dst_vreg = d; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE);
}
static void ldg32(Em *e, int d, int addr64, int udesc, int first) { BlackwellIRInsn i = I0(BW_IR_LDG_E); i.dst_vreg = d; i.src1_vreg = addr64; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void stg(Em *e, int addr64, uint32_t imm, int val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.src2_vreg = val; i.imm = imm; i.ureg = udesc; emit(e, i, K_STORE); }
static void fadd(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FADD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
/* d = a * b[bsub] + c */
static void ffma(Em *e, int d, int a, int b, unsigned bsub, int c) { BlackwellIRInsn i = I0(BW_IR_FFMA); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void shfl_down(Em *e, int d, int a, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_SHFL_DOWN); i.dst_vreg = d; i.src1_vreg = a; i.imm = off; emit(e, i, K_VAR); }
static void isetp_ge_u32(Em *e, int a, int b) { BlackwellIRInsn i = I0(BW_IR_ISETP_GE_U32); i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_PRED); }
static void bra_if_p0(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
static void bra(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
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
/* a value defined before the loop and read inside it stays live until the back-branch (moe.c) */
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
static int log2u(uint32_t v) { int l = 0; while ((1u << l) < v) l++; return (1u << l) == v ? l : -1; }

/* e4m3 byte i of word `word` -> f32 bits (value * 2^-120): ((b & 0x7F) << 20) | ((b & 0x80) << 24); moe.c decode_byte */
static void decode_byte(Em *e, int f, int wq, unsigned word, unsigned i, int u, int a, int s) {
    const uint32_t M = 0x07F00000u;
    if (i < 3) imadi(e, u, wq, word, 1u << (20u - 8u * i), NOREG);
    else shr(e, u, wq, word, 4);
    andi(e, a, u, 0, M);
    andi(e, s, wq, word, 0x80u << (8u * i));
    if (i < 3) imadi(e, f, s, 0, 1u << (24u - 8u * i), a);
    else imadi(e, f, s, 0, 1u, a);
}

int omega_fp8head_spec_ok(const OmegaFp8HeadSpec *s) {
    if (!s) return 0;
    const uint32_t step = 32u * OMEGA_FP8HEAD_LANE_BYTES;
    if (s->k == 0 || s->k > OMEGA_FP8_MAX_K || s->k % step) return 0;
    if (s->rows != 1 && s->rows != 2) return 0;
    if (s->warps < 1 || s->warps > 32) return 0;
    if (s->xb < 1 || s->xb > OMEGA_FP8HEAD_MAX_XB) return 0;
    if (s->mutant < 0 || s->mutant > 3) return 0;
    /* immediate offsets of a step's loads inside the 24-bit signed LDG field: x row b at b k 4 + 48, weight row r at r k */
    if ((uint64_t)(s->xb - 1u) * s->k * 4u + 48u >= (1u << 23)) return 0;
    if ((uint64_t)(s->rows - 1u) * s->k + OMEGA_FP8HEAD_LANE_BYTES >= (1u << 23)) return 0;
    return 1;
}

int omega_blackwell_codegen_fp8head_prog(const OmegaFp8HeadSpec *spec, BlackwellIRProgram *prog) {
    if (!omega_fp8head_spec_ok(spec) || !prog) return -1;
    const uint32_t LB = OMEGA_FP8HEAD_LANE_BYTES, K = spec->k, R = spec->rows, W = spec->warps, XB = spec->xb;
    const uint32_t KS = K / (32u * LB), KB = K / OMEGA_FP8_BLOCK;
    const uint32_t STEP_SCALE = (32u * LB / OMEGA_FP8_BLOCK) * 4u; /* scale bytes per K step (4 blocks) */
    const int log2lpb = log2u(OMEGA_FP8_BLOCK / LB), log2rpb = log2u(OMEGA_FP8_BLOCK / R);
    if (log2lpb < 0 || log2rpb < 0) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    Em *e = &em;

    /* ---- prologue ---- */
    int udesc = UV64(e), wbase = V64(e), sbase = V64(e), px = V64(e), py = V64(e);
    int rgn = V(e), ys = V(e), stride = V(e);
    int tid = V(e), cx = V(e), warp = V(e), lane = V(e), one = V(e);
    int xlane = V64(e), wlane = V(e), slane = V(e), rg = V(e);
    int pw = V64(e), ps = V64(e), pxc = V64(e), pyr = V64(e), pyb = V64(e), pya = V64(e), srg = V(e), cnt = V(e);
    int wq[2], fd[2][16], xq[4], acc[2][OMEGA_FP8HEAD_MAX_XB];
    for (uint32_t r = 0; r < R; r++) { wq[r] = V128(e); for (uint32_t j = 0; j < 16; j++) fd[r][j] = V(e); for (uint32_t b = 0; b < XB; b++) acc[r][b] = V(e); }
    for (uint32_t q = 0; q < 4; q++) xq[q] = V128(e);
    int sc = V(e), dot = V(e), u = V(e), a = V(e), sg = V(e), t = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, wbase, OMEGA_FP8HEAD_ARG_W); ldc64(e, sbase, OMEGA_FP8HEAD_ARG_S);
    ldc64(e, px, OMEGA_FP8HEAD_ARG_X); ldc64(e, py, OMEGA_FP8HEAD_ARG_Y);
    ldc32(e, rgn, OMEGA_FP8HEAD_ARG_RG); ldc32(e, ys, OMEGA_FP8HEAD_ARG_YSTRIDE); ldc32(e, stride, OMEGA_FP8HEAD_ARG_STRIDE);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X);
    movi(e, one, 1);
    shr(e, warp, tid, 0, 5);
    andi(e, lane, tid, 0, 31);
    imadw(e, xlane, lane, LB * 4u, px);             /* this lane's x' chunk of step 0, x row 0 */
    imadi(e, wlane, lane, 0, LB, NOREG);            /* byte offset of this lane's chunk in a row */
    shr(e, slane, lane, 0, (uint32_t)log2lpb); imadi(e, slane, slane, 0, 4, NOREG); /* scale word of this lane in a step */
    imadi(e, rg, cx, 0, W, warp);                   /* first row group of this warp */

    /* ---- row-group loop ---- */
    const int L_rg = here(e);
    loop_begin(e);
    isetp_ge_u32(e, rg, rgn);
    const int bra_end = bra_if_p0_fwd(e);
    imadw(e, pw, rg, R * K, wbase); imadw(e, pw, wlane, 1, pw);
    shr(e, srg, rg, 0, (uint32_t)log2rpb);
    if (spec->mutant == 2) xorr(e, srg, srg, one);  /* mutant 2: the neighbouring 128-row scale block */
    imadw(e, ps, srg, KB * 4u, sbase); imadw(e, ps, slane, 1, ps);
    imadw(e, pxc, one, 0, xlane);
    for (uint32_t r = 0; r < R; r++) for (uint32_t b = 0; b < XB; b++) movrz(e, acc[r][b]);
    movi(e, cnt, KS);

    /* ---- K loop: one 512-column step per iteration ---- */
    const int L_k = here(e);
    loop_begin(e);
    for (uint32_t b = 0; b < XB; b++) {
        const uint32_t xoff = spec->mutant == 3 ? 0u : b * K * 4u; /* mutant 3: every x row reads row 0 */
        int first = 1;
        if (b == 0) {
            for (uint32_t r = 0; r < R; r++) { ldg128(e, wq[r], pw, r * K, udesc, first); first = 0; }
            ldg32(e, sc, ps, udesc, 0);
        }
        for (uint32_t q = 0; q < 4; q++) { ldg128(e, xq[q], pxc, xoff + q * 16u, udesc, first); first = 0; }
        if (b == 0)
            for (uint32_t r = 0; r < R; r++)
                for (uint32_t word = 0; word < 4; word++)
                    for (unsigned i = 0; i < 4; i++) decode_byte(e, fd[r][word * 4u + i], wq[r], word, i, u, a, sg);
        for (uint32_t r = 0; r < R; r++) {
            movrz(e, dot);
            for (uint32_t word = 0; word < 4; word++)
                for (unsigned i = 0; i < 4; i++) ffma(e, dot, fd[r][word * 4u + i], xq[word], i, dot);
            ffma(e, acc[r][b], dot, sc, 0, acc[r][b]);
        }
    }
    imadw(e, pw, one, 32u * LB, pw);
    imadw(e, pxc, one, 32u * LB * 4u, pxc);
    imadw(e, ps, one, spec->mutant == 1 ? 0u : STEP_SCALE, ps); /* mutant 1: every step reads the step-0 k-blocks */
    imadi(e, cnt, one, 0, 0xffffffffu, cnt);
    isetp_ge_u32(e, cnt, one);
    bra_if_p0(e, L_k);
    loop_end(e);

    /* ---- warp reduction per (row, x row), lane 0 stores ---- */
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t b = 0; b < XB; b++)
            for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, acc[r][b], o); fadd(e, acc[r][b], acc[r][b], t); }
    imadw(e, pyr, rg, R * 4u, py);
    isetp_ge_u32(e, lane, one);
    const int at_bssy = bssy_fwd(e);
    const int bra_skip = bra_if_p0_fwd(e);
    for (uint32_t b = 0; b < XB; b++) {
        if (b == 0) imadw(e, pyb, one, 0, pyr); else imadw(e, pyb, ys, 1, pyb);
        for (uint32_t r = 0; r < R; r++) { imadw(e, pya, one, r * 4u, pyb); stg(e, pya, 0, acc[r][b], udesc); } /* STG.E imm 0 only */
    }
    const int at_bsync = here(e);
    bsync(e);
    patch_bra(e, bra_skip, at_bsync);
    patch_bra(e, at_bssy, at_bsync + 1);
    imadi(e, rg, stride, 0, 1, rg);
    bra(e, L_rg);
    loop_end(e);
    patch_bra(e, bra_end, here(e));
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog) != 0 ? -1 : 0;
}

int omega_fp8head_prog_stats(const OmegaFp8HeadSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog; /* 1024 insns: too big for a stack frame in a test loop (moe.c) */
    if (insns) *insns = 0;
    if (gprs) *gprs = 0;
    if (omega_blackwell_codegen_fp8head_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}
