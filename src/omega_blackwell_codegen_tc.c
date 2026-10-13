/* omega_blackwell_codegen_tc.c: tensor-core kernels for the OWN-30B one-pass prefill (T3). See the header.
 * The emitter helpers below are copied from omega_blackwell_codegen_gemm.c (same chip-learned scheduling rules: every load
 * sets the read barrier SB1 and the next load waits on it; prefetch loads hold scoreboards 3/4 until consumed).
 * HMMA scheduling is taken from nvcc 13.0.88 -arch=sm_121 output (oracle sealed in
 * ~/workspace/investigations/2026-10-13-t3-mma-oracle/oracle.{cu,sass}):
 *   independent HMMA.16816.F32.BF16 back to back: control 0x000ff000 / 0x020ff000 (stall 8, no barriers);
 *   a consumer of an HMMA result (dependent HMMA, store): HMMA stall 15 then NOP stall 13..14 (about 29 cycles), as
 *   omega_blackwell_verify_codegen_fixtures_fragmm records. */
#include "omega_blackwell_codegen_tc.h"
#include <string.h>

/* the full emitter set is shared by the probe and the GEMM generator; not every kernel uses every helper */
#pragma GCC diagnostic ignored "-Wunused-function"

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

/* HMMA d = a * b + c (a quad, b pair, c/d quad; c NOREG = RZ). stall 8: the next instruction must not read d (nvcc's
 * spacing for independent HMMAs); hmma_wait15 + nop14 before any reader of d. */
static void hmma(Em *e, int d, int a, int b, int c, uint32_t stall) {
    BlackwellIRInsn i = I0(BW_IR_HMMA_BF16); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c;
    i.control = CW(stall, SB_NONE, SB_NONE, e->pending); e->pending = 0;
    if (omega_bw_ir_append(e->p, &i) < 0) e->err = -1;
}
static void nop(Em *e, uint32_t stall) {
    BlackwellIRInsn i = I0(BW_IR_NOP); i.control = CW(stall, SB_NONE, SB_NONE, e->pending); e->pending = 0;
    if (omega_bw_ir_append(e->p, &i) < 0) e->err = -1;
}

int omega_blackwell_codegen_tc_probe_prog(int mutant, BlackwellIRProgram *prog) {
    if (!prog || mutant < 0 || mutant > 1) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    Em *e = &em;
    int udesc = UV64(e), pin = V64(e), pout = V64(e), tid = V(e), cx = V(e), pa = V64(e), po = V64(e);
    int a = V128(e), b = V64(e), c = V128(e), d = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pin, 0x380); ldc64(e, pout, 0x388);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X);
    int pcase = V64(e), pb = V64(e); if (e->err) return -1;
    imadw(e, pcase, cx, OMEGA_TC_PROBE_IN_BYTES, pin); imadw(e, pa, tid, 16, pcase); imadw(e, pb, tid, 8, pcase);
    imadw(e, po, cx, OMEGA_TC_PROBE_OUT_BYTES, pout); imadw(e, po, tid, 16, po);
    ldg_wide(e, BW_IR_LDG_E_128, a, pa, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_128, c, pa, 768, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_64, b, pb, 512, udesc, 0);
    /* wait for every load, then one HMMA, then 29 cycles before the stores read d */
    hmma(e, d, a, b, mutant == 1 ? NOREG : c, 15);
    nop(e, 14);
    for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = po; i.src2_vreg = d; i.src2_subreg = (uint8_t)q; i.imm = 4u * q; i.ureg = udesc; emit(e, i, K_STORE); }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}
