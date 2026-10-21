/* omega_blackwell_codegen_bf16.c: bf16 streaming GEMV kernel (the lm_head), OWN-30B 2026-10-13. See the header.
 * The emitter helpers below are copies of omega_blackwell_codegen_moe.c's (same chip-learned scheduling rules: every
 * instruction waits on everything still pending; variable-latency results go through scoreboard 0, stores set the read
 * barrier 1, uniform loads scoreboard 2; fixed-latency producers stall 6; ISETP stalls 14 before a predicated branch;
 * loads of one iteration are issued back to back so a warp has all of them in flight at once). */
#include "omega_blackwell_codegen_bf16.h"
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
static void fmaxr(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FMNMX_MAX); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void xorr(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_LOP3_XOR); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
/* d = a * b[bsub] + c */
static void ffma(Em *e, int d, int a, int b, unsigned bsub, int c) { BlackwellIRInsn i = I0(BW_IR_FFMA); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void shfl_down(Em *e, int d, int a, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_SHFL_DOWN); i.dst_vreg = d; i.src1_vreg = a; i.imm = off; emit(e, i, K_VAR); }
static void isetp_ge_u32(Em *e, int a, int b) { BlackwellIRInsn i = I0(BW_IR_ISETP_GE_U32); i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_PRED); }
static void bra_if_p0(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
static void bra(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
/* a forward branch whose target is patched when known: returns the instruction index */
static int bra_if_p0_fwd(Em *e) { int at = here(e); BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.imm = 1; emit(e, i, K_BRA); return at; }
static int bra_ifnot_p0_fwd(Em *e) { int at = here(e); BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.predicate_not = true; i.imm = 1; emit(e, i, K_BRA); return at; }
static int bra_fwd(Em *e) { int at = here(e); BlackwellIRInsn i = I0(BW_IR_BRA); i.imm = 1; emit(e, i, K_BRA); return at; }
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


static int spec_ok(const OmegaBf16GemvSpec *s) {
    if (!s) return 0;
    if (s->rows != 1 && s->rows != 2 && s->rows != 4) return 0;
    if (s->warps < 1 || s->warps > 32) return 0;
    if (s->k == 0 || s->k > OMEGA_BF16_MAX_K || s->k % 256u || s->k % (32u * s->warps)) return 0;
    if (s->k > 512u * s->warps) return 0; /* at most 16 staging rounds */
    uint32_t ks = s->k / 256u;
    if (s->depth == 0 || s->depth > 16 || ks % s->depth || s->rows * s->depth > 16u) return 0;
    if (s->mask > 1) return 0;
    if (s->mask && (s->rows != 1 || s->ystride)) return 0; /* MASK form: one row per warp, one x row */
    if (s->mutant < 0 || s->mutant > (s->mask ? 4 : 3)) return 0;
    if (s->ystride > (1u << 28)) return 0; /* ystride * 4 fits the 32-bit IMAD.WIDE immediate */
    if ((uint64_t)(s->rows - 1) * s->k * 2u + 512u * s->depth >= (1u << 23)) return 0; /* 24-bit signed LDG immediate */
    return 1;
}
uint32_t omega_bf16_gemv_steps(const OmegaBf16GemvSpec *spec) { return spec_ok(spec) ? spec->k / 256u : 0; }
uint32_t omega_bf16_gemv_shared_bytes(const OmegaBf16GemvSpec *spec) { return spec_ok(spec) ? spec->k * 4u : 0u; }

int omega_blackwell_codegen_bf16_gemv_prog(const OmegaBf16GemvSpec *spec, BlackwellIRProgram *prog) {
    if (!spec_ok(spec) || !prog) return -1;
    const uint32_t K = spec->k, R = spec->rows, W = spec->warps, D = spec->depth, KS = K / 256u;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    Em *e = &em;

    /* ---- prologue ---- */
    int udesc = UV64(e), pwb = V64(e), px = V64(e), py = V64(e), rgn = V(e), stride = V(e);
    int tid = V(e), cx = V(e), warp = V(e), lane = V(e), one = V(e);
    int wlane = V(e), rg = V(e), xl0 = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pwb, OMEGA_BF16_ARG_W); ldc64(e, px, OMEGA_BF16_ARG_X); ldc64(e, py, OMEGA_BF16_ARG_Y);
    ldc32(e, rgn, OMEGA_BF16_ARG_RG); ldc32(e, stride, OMEGA_BF16_ARG_STRIDE);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X);
    if (spec->ystride) { /* ROWS form: this CTA row's x and y */
        int cy = V(e);
        if (e->err) return -1;
        s2r(e, cy, BW_SR_CTAID_Y);
        imadw(e, px, cy, K * 4u, px);
        imadw(e, py, cy, spec->ystride * 4u, py);
    }
    movi(e, one, 1);
    shr(e, warp, tid, 0, 5);
    andi(e, lane, tid, 0, 31);
    imadi(e, wlane, lane, 0, 16, NOREG);  /* byte offset of this lane's chunk in a row step */
    imadi(e, rg, cx, 0, W, warp);         /* first row group of this warp */
    imadi(e, xl0, lane, 0, 32, NOREG);    /* this lane's x chunk (8 floats) of step 0, shared byte offset */

    /* ---- stage x (k floats) in shared memory: one 32-bit word per thread per round ---- */
    {
        const uint32_t rounds = K / (32u * W);
        int pxt = V64(e), sa0 = V(e), sa = V(e), pxs[16], xst[16];
        for (uint32_t j = 0; j < rounds; j++) { pxs[j] = V64(e); xst[j] = V(e); }
        if (e->err) return -1;
        imadw(e, pxt, tid, 4, px);
        imadi(e, sa0, tid, 0, 4, NOREG);
        for (uint32_t j = 0; j < rounds; j++) imadw(e, pxs[j], one, j * 128u * W, pxt);
        for (uint32_t j = 0; j < rounds; j++) ldg32(e, xst[j], pxs[j], udesc, j == 0);
        for (uint32_t j = 0; j < rounds; j++) { imadi(e, sa, one, 0, j * 128u * W, sa0); sts32(e, sa, xst[j]); }
        bar_sync(e);
    }

    int pw = V64(e), pwr = V64(e), xl = V(e), cnt = V(e), pyr = V64(e), pya = V64(e), t = V(e), f = V(e);
    int acc[4], wq[4][16], xq[2];
    for (uint32_t r = 0; r < R; r++) { acc[r] = V(e); for (uint32_t s = 0; s < D; s++) wq[r][s] = V128(e); }
    xq[0] = V128(e); xq[1] = V128(e);
    if (e->err) return -1;
    /* ---- MASK form: tau once, normalized (-0 -> +0); -inf for the skipped rows ---- */
    int phi = NOREG, tn = NOREG, zero = NOREG, ninf = NOREG, phr = NOREG, hv = NOREG, mv = NOREG;
    if (spec->mask) {
        int ptau = V64(e);
        phi = V64(e); tn = V(e); zero = V(e); ninf = V(e); phr = V64(e); hv = V(e); mv = V(e);
        if (e->err) return -1;
        ldc64(e, phi, OMEGA_BF16_ARG_HI); ldc64(e, ptau, OMEGA_BF16_ARG_TAU);
        ldg32(e, tn, ptau, udesc, 1);
        movrz(e, zero);
        fadd(e, tn, tn, zero);
        movi(e, ninf, 0xff800000u);
    }

    /* ---- row-group loop ---- */
    const int L_rg = here(e);
    loop_begin(e);
    isetp_ge_u32(e, rg, rgn);
    const int bra_end = bra_if_p0_fwd(e);
    int bra_skip_row = -1;
    if (spec->mask) { /* warp-uniform: every lane loads the same hi[rg] */
        imadw(e, phr, rg, 4u, phi);
        ldg32(e, hv, phr, udesc, 1);
        fadd(e, hv, hv, zero);
        fmaxr(e, mv, hv, tn);
        xorr(e, mv, mv, hv);
        isetp_ge_u32(e, mv, one); /* P0: hi < tau (the max is not hi) */
        bra_skip_row = spec->mutant == 4 ? bra_ifnot_p0_fwd(e) : bra_if_p0_fwd(e);
    }
    imadw(e, pwr, rg, R * K * 2u, pwb); imadw(e, pw, wlane, 1, pwr);
    imadi(e, xl, xl0, 0, 1, NOREG);
    for (uint32_t r = 0; r < R; r++) movrz(e, acc[r]);
    movi(e, cnt, KS / D);

    /* ---- K loop: D steps per iteration, every load issued before any use ---- */
    const int L_k = here(e);
    loop_begin(e);
    {
        int first = 1;
        for (uint32_t s = 0; s < D; s++) for (uint32_t r = 0; r < R; r++) { ldg_wide(e, BW_IR_LDG_E_128, wq[r][s], pw, r * K * 2u + s * OMEGA_BF16_STEP_BYTES, udesc, first); first = 0; }
    }
    for (uint32_t s = 0; s < D; s++) {
        const uint32_t xoff = spec->mutant == 2 ? 0u : s * 1024u;
        lds128(e, xq[0], xl, xoff, 1); lds128(e, xq[1], xl, xoff + 16u, 0);
        for (uint32_t r = 0; r < R; r++) {
            const uint32_t words = spec->mutant == 3 ? 3u : 4u;
            for (uint32_t word = 0; word < words; word++) {
                const int q = xq[word >> 1]; const unsigned sub = (word & 1u) * 2u;
                /* low half = element 2 word, high half = element 2 word + 1 */
                imadi(e, f, wq[r][s], word, 65536u, NOREG);
                ffma(e, acc[r], f, q, spec->mutant == 1 ? sub + 1u : sub, acc[r]);
                andi(e, f, wq[r][s], word, 0xffff0000u);
                ffma(e, acc[r], f, q, spec->mutant == 1 ? sub : sub + 1u, acc[r]);
            }
        }
    }
    imadw(e, pw, one, D * OMEGA_BF16_STEP_BYTES, pw);
    imadi(e, xl, one, 0, spec->mutant == 2 ? 0u : D * 1024u, xl);
    imadi(e, cnt, one, 0, 0xffffffffu, cnt);
    isetp_ge_u32(e, cnt, one);
    bra_if_p0(e, L_k);
    loop_end(e);

    /* ---- warp reduction, lane 0 stores ---- */
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, acc[r], o); fadd(e, acc[r], acc[r], t); }
    imadw(e, pyr, rg, R * 4u, py);
    isetp_ge_u32(e, lane, one);
    const int at_bssy = bssy_fwd(e);
    const int bra_skip = bra_if_p0_fwd(e);
    for (uint32_t r = 0; r < R; r++) { imadw(e, pya, one, r * 4u, pyr); stg(e, pya, 0, acc[r], udesc); }
    const int at_bsync = here(e);
    bsync(e);
    patch_bra(e, bra_skip, at_bsync);
    patch_bra(e, at_bssy, at_bsync + 1);
    if (spec->mask) { /* skipped row: every lane stores the same -inf to y[rg] */
        const int bra_next = bra_fwd(e);
        patch_bra(e, bra_skip_row, here(e));
        imadw(e, pyr, rg, 4u, py);
        stg(e, pyr, 0, ninf, udesc);
        patch_bra(e, bra_next, here(e));
    }
    imadi(e, rg, stride, 0, 1, rg);
    bra(e, L_rg);
    loop_end(e);
    const int L_end = here(e);
    patch_bra(e, bra_end, L_end);
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_bf16_gemv_prog_stats(const OmegaBf16GemvSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_bf16_gemv_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}

#ifndef OMEGA_GPU_NO_POSIX
int omega_blackwell_codegen_bf16_gemv(const OmegaBf16GemvSpec *spec, OmegaBlackwellKernel *kernel) {
    if (!kernel) return -1;
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    if (!prog) return -1;
    int rc = omega_blackwell_codegen_bf16_gemv_prog(spec, prog);
    size_t need = rc == 0 ? omega_bw_program_code_bytes(prog) : 0;
    uint8_t *buf = need ? malloc(need) : NULL;
    if (!buf) { free(prog); return -1; }
    rc = omega_blackwell_kernel_encode_into(prog, buf, need, kernel);
    free(prog);
    if (rc != 0) { free(buf); return -1; }
    return 0;
}
#endif
