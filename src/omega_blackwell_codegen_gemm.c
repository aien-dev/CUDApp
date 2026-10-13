/* omega_blackwell_codegen_gemm.c: FP8 e4m3 block-scaled grouped GEMM for prefill (OWN-30B builder F item 26). See the
 * header for the contract. The emitter helpers are copies of omega_blackwell_codegen_moe.c's (same chip-learned
 * scheduling rules) plus one addition: weight and scale prefetch loads use scoreboards 3 (buffer A) and 4 (buffer B)
 * and are waited for only by the step that decodes them, so the next step's weights stream while this step computes. */
#include "omega_blackwell_codegen_gemm.h"
#ifndef OMEGA_GPU_NO_POSIX
#include <stdlib.h>
#endif
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

static int spec_ok(const OmegaFp8GemmSpec *s) {
    if (!s) return 0;
    if (s->lane_bytes != 8 && s->lane_bytes != 16) return 0;
    if (s->mt == 0 || (s->mt & (s->mt - 1)) || s->mt * 128u * s->lane_bytes > OMEGA_GEMM_MAX_SHARED) return 0;
    if (s->warps < 1 || s->warps > 32 || (s->warps & (s->warps - 1)) || (s->mt * s->lane_bytes) % s->warps) return 0; /* power-of-two warps: the staging words per thread are whole */
    if (s->k == 0 || s->k > 16384u || s->k % 128u || s->k % (32u * s->lane_bytes)) return 0;
    uint32_t ks = s->k / (32u * s->lane_bytes);
    if (ks % 2u && ks > 3u) return 0; /* odd step counts are fully unrolled: at most 3 */
    if (s->mutant < 0 || s->mutant > 3) return 0;
    return 1;
}
uint32_t omega_fp8gemm_steps(const OmegaFp8GemmSpec *spec) { return spec_ok(spec) ? spec->k / (32u * spec->lane_bytes) : 0; }
uint32_t omega_fp8gemm_shared_bytes(const OmegaFp8GemmSpec *spec) { return spec_ok(spec) ? spec->mt * 128u * spec->lane_bytes : 0u; }

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

int omega_blackwell_codegen_fp8gemm_prog(const OmegaFp8GemmSpec *spec, BlackwellIRProgram *prog) {
    if (!spec_ok(spec) || !prog) return -1;
    const uint32_t K = spec->k, LB = spec->lane_bytes, MT = spec->mt, W = spec->warps;
    const uint32_t KS = K / (32u * LB), BPS = 32u * LB / 128u, LPB = 128u / LB, KB = K / 128u, WORDS = LB / 4u;
    const uint32_t U = KS % 2u == 0 ? 2u : KS; /* K steps per loop iteration (the A/B register sets alternate) */
    const uint32_t NJ = MT * LB / W, ROWW = 32u * LB, T = 32u * W; /* staging words per thread, rows per staging j, words per row slab */
    const int log2lpb = log2u(LPB), log2mt = log2u(MT), log2roww = log2u(ROWW);
    const BlackwellIROpcode ldw = LB == 16 ? BW_IR_LDG_E_128 : BW_IR_LDG_E_64;
    if (log2lpb < 0 || log2mt < 0 || log2roww < 0) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    Em *e = &em;

    /* ---- prologue ---- */
    int udesc = UV64(e), pseg = V64(e), px = V64(e), py = V64(e), nreg = V(e), stride = V(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), one = V(e), tent = V64(e);
    int wbase = V64(e), sbase = V64(e), rc = V64(e), row0 = V(e), cnt = V(e), tpass = V(e), passes = V(e);
    int wlane = V(e), slane = V(e), xl = V(e), nm1 = V(e), ysn = V(e), mrow = V(e), off4 = V(e), sa0 = V(e), base = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pseg, OMEGA_GEMM_ARG_SEG); ldc64(e, px, OMEGA_GEMM_ARG_X); ldc64(e, py, OMEGA_GEMM_ARG_Y);
    ldc32(e, nreg, OMEGA_GEMM_ARG_N); ldc32(e, stride, OMEGA_GEMM_ARG_STRIDE);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, one, 1);
    shr(e, warp, tid, 0, 5);
    andi(e, lane, tid, 0, 31);
    imadw(e, tent, cy, OMEGA_GEMM_SEG_BYTES, pseg);
    ldg_wide(e, BW_IR_LDG_E_64, wbase, tent, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_64, sbase, tent, 8, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_64, rc, tent, 16, udesc, 0);
    imadi(e, row0, rc, 0, 1, NOREG); imadi(e, cnt, rc, 1, 1, NOREG);
    imadi(e, tpass, one, 0, MT - 1u, cnt); shr(e, passes, tpass, 0, (uint32_t)log2mt);
    imadi(e, wlane, lane, 0, LB, NOREG);
    shr(e, slane, lane, 0, (uint32_t)log2lpb); imadi(e, slane, slane, 0, 4, NOREG);
    imadi(e, xl, lane, 0, 16, NOREG);
    imadi(e, nm1, one, 0, 0xffffffffu, nreg);
    imadi(e, ysn, nreg, 0, 4, NOREG);
    shr(e, mrow, tid, 0, (uint32_t)log2roww);
    andi(e, off4, tid, 0, ROWW - 1u); imadi(e, off4, off4, 0, 4, NOREG);
    imadi(e, sa0, tid, 0, 4, NOREG);
    imadi(e, base, cx, 0, W, NOREG);

    /* loop-wide registers */
    int rg = V(e), over = V(e), t1 = V(e), d = V(e), tmp = V(e), rgc = V(e), lo = V(e);
    int pw = V64(e), ps = V64(e), pyrg = V64(e), rowp = V(e), rem = V(e), pc = V(e);
    int xrp = V64(e), xs = V64(e), pwc = V64(e), psc = V64(e), cntk = V(e), pyr = V64(e), pya = V64(e), t = V(e), tq = V(e), fl = V(e), sv = V(e);
    int acc[16], dot[16], xq[4], wq[2], sc[2], fr[4], u = V(e), a = V(e), sg = V(e), onef = V(e), dum = V(e);
    for (uint32_t m = 0; m < MT; m++) { acc[m] = V(e); dot[m] = V(e); }
    for (int q = 0; q < 4; q++) xq[q] = V128(e);
    for (int b = 0; b < 2; b++) { wq[b] = LB == 16 ? V128(e) : V64(e); sc[b] = V(e); }
    for (int i = 0; i < 4; i++) fr[i] = V(e);
    if (e->err) return -1;

    /* ---- weight-row loop (uniform per CTA) ---- */
    const int L_rg = here(e);
    loop_begin(e);
    isetp_ge_u32(e, base, nreg);
    const int bra_end = bra_if_p0_fwd(e);
    imadi(e, rg, base, 0, 1, warp);
    imadi(e, t1, rg, 0, 0xffffffffu, nm1); shr(e, over, t1, 0, 31);              /* over = (rg > n - 1) */
    imadi(e, d, nm1, 0, 0xffffffffu, rg);                                         /* rg - (n - 1) */
    imadr(e, tmp, over, d, NOREG); imadi(e, rgc, tmp, 0, 0xffffffffu, rg);        /* rg clamped to n - 1 */
    imadw(e, pw, rgc, K, wbase); imadw(e, pw, wlane, 1, pw);
    shr(e, tmp, rgc, 0, 7);
    imadw(e, ps, tmp, KB * 4u, sbase); imadw(e, ps, slane, 1, ps);
    imadw(e, pyrg, rg, 4, py);
    imadi(e, lo, over, 0, 1, lane);
    imadi(e, rowp, row0, 0, 1, NOREG); imadi(e, rem, cnt, 0, 1, NOREG); imadi(e, pc, passes, 0, 1, NOREG);

    /* ---- pass loop: MT rows of x per pass ---- */
    const int L_pass = here(e);
    loop_begin(e);
    imadi(e, t, rowp, 0, 1, mrow); imadw(e, xrp, t, K * 4u, px); imadw(e, xrp, off4, 1, xrp);
    imadw(e, xs, one, 0, xrp);
    imadw(e, pwc, one, 0, pw); imadw(e, psc, one, 0, ps);
    for (uint32_t m = 0; m < MT; m++) movrz(e, acc[m]);
    ldw_pf(e, ldw, wq[0], pwc, 0, udesc, 3);
    ldw_pf(e, BW_IR_LDG_E, sc[0], psc, 0, udesc, 3);
    movi(e, cntk, KS / U);

    /* ---- K loop: U steps per iteration ---- */
    const int L_k = here(e);
    loop_begin(e);
    for (uint32_t us = 0; us < U; us++) {
        const int cur = (int)(us & 1u), nxt = 1 - cur;
        const unsigned sbc = cur ? 4u : 3u, sbn = nxt ? 4u : 3u;
        /* addresses first (fixed-latency ops would otherwise wait for the loads below), then loads back to back; the
         * staging words go in rounds of 4 so the temporaries stay few */
        const int last = us + 1 == U, prefetch = !last || KS > U;
        int pwn = -1, psn = -1;
        if (prefetch) { pwn = V64(e); psn = V64(e); }
        int sa = V(e);
        if (e->err) return -1;
        if (prefetch && !last) {
            imadw(e, pwn, one, (us + 1) * 32u * LB, pwc); imadw(e, psn, one, spec->mutant == 2 ? 0u : (us + 1) * BPS * 4u, psc);
        } else if (prefetch) { /* the next iteration's step 0; the last iteration re-reads its own step (harmless, drained after the loop) */
            int nf = V(e), inc = V(e), inc2 = V(e);
            if (e->err) return -1;
            imadi(e, nf, one, 0, 0xfffffffeu, cntk);     /* cntk - 2 */
            shr(e, nf, nf, 0, 31);                       /* 1 iff cntk == 1 (the last iteration) */
            imadi(e, nf, nf, 0, 0xffffffffu, one);       /* 1 iff another iteration follows */
            imadi(e, inc, nf, 0, U * 32u * LB, NOREG); imadi(e, inc2, nf, 0, spec->mutant == 2 ? 0u : U * BPS * 4u, NOREG);
            imadw(e, pwn, inc, 1, pwc); imadw(e, psn, inc2, 1, psc);
        }
        for (uint32_t j0 = 0; j0 < NJ; j0 += 4) {
            const uint32_t nj = NJ - j0 < 4 ? NJ - j0 : 4;
            int xv[4], xa[4];
            for (uint32_t q = 0; q < nj; q++) { xv[q] = V(e); xa[q] = V64(e); }
            if (e->err) return -1;
            for (uint32_t q = 0; q < nj; q++) { const uint32_t pos = (j0 + q) * T; imadw(e, xa[q], one, ((pos / ROWW) * K + pos % ROWW) * 4u, xs); }
            for (uint32_t q = 0; q < nj; q++) ldg32(e, xv[q], xa[q], udesc, q == 0);
            if (prefetch && j0 == 0) {
                ldw_pf(e, ldw, wq[nxt], pwn, 0, udesc, sbn);
                ldw_pf(e, BW_IR_LDG_E, sc[nxt], psn, 0, udesc, sbn);
            }
            for (uint32_t q = 0; q < nj; q++) { imadi(e, sa, one, 0, (j0 + q) * T * 4u, sa0); sts32(e, sa, xv[q]); }
        }
        bar_sync(e);
        consume(e, sbc);
        /* compute */
        for (uint32_t m = 0; m < MT; m++) movrz(e, dot[m]);
        for (uint32_t word = 0; word < WORDS; word++) {
            for (unsigned i = 0; i < 4; i++) decode_byte(e, fr[i], wq[cur], word, i, u, a, sg);
            for (uint32_t m0 = 0; m0 < MT; m0 += 4) {
                const uint32_t nb = MT - m0 < 4 ? MT - m0 : 4;
                for (uint32_t q = 0; q < nb; q++) {
                    const uint32_t m = m0 + q, mm = spec->mutant == 3 ? (m ^ 1u) % MT : m;
                    lds128(e, xq[q], xl, mm * ROWW * 4u + word * 512u, q == 0);
                }
                for (uint32_t q = 0; q < nb; q++) for (unsigned i = 0; i < 4; i++) ffma(e, dot[m0 + q], fr[i], xq[q], i, dot[m0 + q]);
            }
        }
        if (spec->mutant == 1) { movi(e, onef, 0x3f800000u); ffma(e, dum, sc[cur], onef, 0, onef); } /* scales dropped, but the scale load stays consumed */
        for (uint32_t m = 0; m < MT; m++) {
            if (spec->mutant == 1) ffma(e, acc[m], dot[m], onef, 0, acc[m]);
            else ffma(e, acc[m], dot[m], sc[cur], 0, acc[m]);
        }
        bar_sync(e);
        imadw(e, xs, one, ROWW * 4u, xs);
    }
    imadw(e, pwc, one, U * 32u * LB, pwc);
    imadw(e, psc, one, spec->mutant == 2 ? 0u : U * BPS * 4u, psc);
    imadi(e, cntk, one, 0, 0xffffffffu, cntk);
    isetp_ge_u32(e, cntk, one);
    bra_if_p0(e, L_k);
    loop_end(e);
    consume(e, 3); consume(e, 4);

    /* ---- reduce and store, per row of the pass ---- */
    for (uint32_t m = 0; m < MT; m++)
        for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, acc[m], o); fadd(e, acc[m], acc[m], t); }
    imadr(e, tq, rowp, ysn, NOREG); imadw(e, pyr, tq, 1, pyrg);
    for (uint32_t m = 0; m < MT; m++) {
        imadi(e, tq, one, 0, (uint32_t)(0u - 1u - m), rem); shr(e, fl, tq, 0, 31); /* 1 iff m >= rem: past this segment's rows */
        imadi(e, sv, fl, 0, 1, lo);
        isetp_ge_u32(e, sv, one);
        const int at_bssy = bssy_fwd(e);
        const int bra_skip = bra_if_p0_fwd(e);
        imadw(e, pya, ysn, m, pyr); stg(e, pya, 0, acc[m], udesc);
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
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_fp8gemm_prog_stats(const OmegaFp8GemmSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_fp8gemm_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}
