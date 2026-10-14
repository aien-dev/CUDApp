/* omega_blackwell_codegen_moe2.c: FP8 e4m3 block-scaled GEMV, variant 2 (row-pipelined). See the header.
 *
 * Scheduling: variant 1's chip-learned rules (every instruction waits on everything pending; variable-latency
 * results through scoreboard 0, every load sets the read barrier 1 and the next load of a batch waits on it,
 * uniform loads scoreboard 2; fixed-latency producers stall 6; ISETP stalls 14 before a predicated branch),
 * plus one new rule: the loads of the NEXT row group land through scoreboard 3 (SB_B), which no instruction
 * waits on except the register moves that take the set over (K_MOVB). The emitter's `pending` mask therefore
 * never carries SB_B: the arithmetic of the current row group runs with the next one's loads in flight.
 * Hazards covered: (1) the moves read set B before the next loads into B are issued (program order, fixed
 * latency); (2) the address registers of a batch are protected by SB_RD as in variant 1; (3) every load that
 * was issued is consumed by a move before EXIT (loads are issued iff the row group exists, and the loop runs
 * exactly over the existing row groups). */
#include "omega_blackwell_codegen_moe2.h"
#include <string.h>

#define CW(stall, wbar, rbar, wait) \
    (((uint32_t)(stall) << 9) | ((uint32_t)(wbar) << 14) | ((uint32_t)(rbar) << 17) | ((uint32_t)(wait) << 20))
#define CW_YIELD 0x2000u
#define SB_NONE 7u
#define SB_GPR 0u
#define SB_RD 1u
#define SB_UGPR 2u
#define SB_B 3u
#define NOREG (-1)
#define MAX_LOOPS 4
#define MAX_KS 16u
#define MAX_ROWS 8u

enum { K_FIXED, K_PRED, K_VAR, K_VAR_MORE, K_UVAR, K_STORE, K_SYNC, K_EXIT, K_BRA, K_BSSY, K_BSYNC, K_LDB, K_LDB_MORE, K_MOVB };

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
    /* next-row-group loads: data on SB_B (not auto-waited), address operands on SB_RD (waited by the next instruction) */
    case K_LDB:   in.control = CW(4, SB_B, SB_RD, wait) | CW_YIELD; e->pending = 1u << SB_RD; break;
    case K_LDB_MORE: in.control = CW(4, SB_B, SB_RD, (e->pending & (1u << SB_RD))) | CW_YIELD; e->pending = 1u << SB_RD; break;
    /* the first move of set B into set A: the one wait on SB_B */
    case K_MOVB:  in.control = CW(6, SB_NONE, SB_NONE, wait | (1u << SB_B)); e->pending = 0; break;
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
/* d[dsub] = s[ssub] (IMAD d = s * 1 + RZ, sub-register destination); kind K_MOVB for the first move of a set */
static void movsub(Em *e, int d, unsigned dsub, int s, unsigned ssub, int kind) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d; i.dst_subreg = (uint8_t)dsub; i.src1_vreg = s; i.src1_subreg = (uint8_t)ssub; i.imm = 1; emit(e, i, kind); }
static void shr(Em *e, int d, int a, unsigned asub, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.imm = imm; emit(e, i, K_FIXED); }
static void andi(Em *e, int d, int a, unsigned asub, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.imm = imm; emit(e, i, K_FIXED); }
static void xorr(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_LOP3_XOR); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void ldg_wide(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, int kind) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit(e, i, kind);
}
static void ldg32(Em *e, int d, int addr64, int udesc, int kind) { BlackwellIRInsn i = I0(BW_IR_LDG_E); i.dst_vreg = d; i.src1_vreg = addr64; i.ureg = udesc; emit(e, i, kind); }
static void stg(Em *e, int addr64, uint32_t imm, int val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.src2_vreg = val; i.imm = imm; i.ureg = udesc; emit(e, i, K_STORE); }
static void lds128(Em *e, int d128, int addr, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LDS128); i.dst_vreg = d128; i.src1_vreg = addr; i.imm = imm; emit(e, i, K_VAR); }
static void sts32(Em *e, int addr, int val) { BlackwellIRInsn i = I0(BW_IR_STS32); i.src1_vreg = addr; i.src2_vreg = val; emit(e, i, K_STORE); }
static void bar_sync(Em *e) { emit(e, I0(BW_IR_BAR_SYNC), K_SYNC); }
static void fadd(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FADD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
/* d = a * b[bsub] + c */
static void ffma(Em *e, int d, int a, int b, unsigned bsub, int c) { BlackwellIRInsn i = I0(BW_IR_FFMA); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void shfl_down(Em *e, int d, int a, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_SHFL_DOWN); i.dst_vreg = d; i.src1_vreg = a; i.imm = off; emit(e, i, K_VAR); }
static void isetp_ge_u32(Em *e, int a, int b) { BlackwellIRInsn i = I0(BW_IR_ISETP_GE_U32); i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_PRED); }
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

/* Loops: a value defined before the loop and read inside it stays live until the back-branch
 * (the linear-scan allocator knows no back edges; same rule as variant 1). */
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

int omega_fp8_gemv2_spec_ok(const OmegaFp8GemvSpec *s) {
    if (!s) return 0;
    if (s->lane_bytes != 8 && s->lane_bytes != 16) return 0;
    if (s->rows != 1 && s->rows != 2 && s->rows != 4 && s->rows != 8) return 0;
    if (s->warps < 1 || s->warps > 32) return 0;
    if (s->k == 0 || s->k > OMEGA_FP8_MAX_K || s->k % OMEGA_FP8_BLOCK || s->k % (32u * s->lane_bytes)) return 0;
    if (s->mutant < 0 || s->mutant > 3) return 0;
    const uint32_t ks = s->k / (32u * s->lane_bytes);
    if (ks > MAX_KS || s->rows * ks * s->lane_bytes > 128u) return 0; /* whole row group per register set */
    if (s->k > 512u * s->warps) return 0;                             /* at most 16 staging rounds */
    if ((uint64_t)(s->rows - 1) * s->k + s->lane_bytes >= (1u << 23)) return 0;
    return 1;
}

/* k * 4 for the staged row, plus one 16-byte dummy slot only when the last staging round is partial (clamped threads store there):
 * k = 4096 at 16 warps is exact and lands on the QMD cap of 16384 bytes (chip: 16400 was refused, receipt OWN30B-FP8-GEMV-V2 r1). */
uint32_t omega_fp8_gemv2_shared_bytes(const OmegaFp8GemvSpec *spec) {
    if (!omega_fp8_gemv2_spec_ok(spec)) return 0u;
    return spec->k * 4u + (spec->k % (32u * spec->warps) ? 16u : 0u);
}

/* e4m3 byte i of word w -> f32 bits (value * 2^-120): ((b & 0x7F) << 20) | ((b & 0x80) << 24). Variant 1's sequence. */
static void decode_byte(Em *e, int f, int wq, unsigned word, unsigned i, int u, int a, int s) {
    const uint32_t M = 0x07F00000u;
    if (i < 3) imadi(e, u, wq, word, 1u << (20u - 8u * i), NOREG);
    else shr(e, u, wq, word, 4);
    andi(e, a, u, 0, M);
    andi(e, s, wq, word, 0x80u << (8u * i));
    if (i < 3) imadi(e, f, s, 0, 1u << (24u - 8u * i), a);
    else imadi(e, f, s, 0, 1u, a);
}

typedef struct {
    int w[MAX_ROWS][MAX_KS]; /* weight bytes: row r, step s (quad or pair) */
    int sc[MAX_KS];          /* scale word per step */
} RegSet;

int omega_blackwell_codegen_fp8_gemv2_prog(const OmegaFp8GemvSpec *spec, BlackwellIRProgram *prog) {
    if (!omega_fp8_gemv2_spec_ok(spec) || !prog) return -1;
    const uint32_t K = spec->k, LB = spec->lane_bytes, R = spec->rows, W = spec->warps;
    const uint32_t KS = K / (32u * LB), BPS = 32u * LB / OMEGA_FP8_BLOCK, LPB = OMEGA_FP8_BLOCK / LB, KB = K / OMEGA_FP8_BLOCK;
    const int log2lpb = log2u(LPB), log2rows_per_block = log2u(OMEGA_FP8_BLOCK / R);
    const uint32_t WORDS = LB / 4u;
    const BlackwellIROpcode ldw = LB == 16 ? BW_IR_LDG_E_128 : BW_IR_LDG_E_64;
    if (log2lpb < 0 || log2rows_per_block < 0) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    Em *e = &em;

    /* ---- prologue (variant 1's) ---- */
    int udesc = UV64(e), ptab = V64(e), px = V64(e), py = V64(e);
    int rgn = V(e), xs = V(e), ys = V(e), stride = V(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), one = V(e), gidx = V(e);
    int tent = V64(e), wbase = V64(e), sbase = V64(e), off = V(e), xrow = V64(e), yrow = V64(e);
    int wlane = V(e), slane = V(e), rgl = V(e), rgc = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, ptab, OMEGA_FP8_ARG_TAB); ldc64(e, px, OMEGA_FP8_ARG_X); ldc64(e, py, OMEGA_FP8_ARG_Y);
    ldc32(e, rgn, OMEGA_FP8_ARG_RG); ldc32(e, xs, OMEGA_FP8_ARG_XSTRIDE); ldc32(e, ys, OMEGA_FP8_ARG_YSTRIDE); ldc32(e, stride, OMEGA_FP8_ARG_STRIDE);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, one, 1);
    shr(e, warp, tid, 0, 5);
    andi(e, lane, tid, 0, 31);
    if (spec->mutant == 3) xorr(e, gidx, cy, one); else imadi(e, gidx, cy, 0, 1, NOREG);
    imadw(e, tent, gidx, 16, ptab);
    ldg_wide(e, BW_IR_LDG_E_64, wbase, tent, 0, udesc, K_VAR);
    ldg_wide(e, BW_IR_LDG_E_64, sbase, tent, 8, udesc, K_VAR_MORE);
    imadr(e, off, cy, xs, NOREG); imadw(e, xrow, off, 1, px);
    imadr(e, off, cy, ys, NOREG); imadw(e, yrow, off, 1, py);
    imadi(e, wlane, lane, 0, LB, NOREG);            /* byte offset of this lane's chunk in a row */
    shr(e, slane, lane, 0, (uint32_t)log2lpb); imadi(e, slane, slane, 0, 4, NOREG); /* scale word of this lane in a step */
    imadi(e, rgl, cx, 0, W, warp);                  /* first row group of this warp: the load cursor */
    imadi(e, rgc, rgl, 0, 1, NOREG);                /* the compute cursor */

    /* ---- registers: two sets, x quad, accumulators, temporaries ---- */
    RegSet A, B;
    int acc[MAX_ROWS], dot[MAX_ROWS], xq = V128(e);
    for (uint32_t r = 0; r < R; r++) { acc[r] = V(e); dot[r] = V(e); for (uint32_t s = 0; s < KS; s++) { A.w[r][s] = LB == 16 ? V128(e) : V64(e); B.w[r][s] = LB == 16 ? V128(e) : V64(e); } }
    for (uint32_t s = 0; s < KS; s++) { A.sc[s] = V(e); B.sc[s] = V(e); }
    int pw = V64(e), ps = V64(e), psd = V64(e), pyr = V64(e), pya = V64(e), srg = V(e), xl0 = V(e);
    int u = V(e), a = V(e), sg = V(e), f = V(e), t = V(e);
    if (e->err) return -1;

    /* ---- stage this group's x' row in shared memory (k floats), one 32-bit word per thread per round (variant 1 deep) ---- */
    {
        const uint32_t rounds = (K + 32u * W - 1u) / (32u * W);
        int pxt = V64(e), sa0 = V(e), sa = V(e), sap = V(e), pxs[16], xst[16], kq = V(e), over = V(e), tq = V(e), qs = V(e), qg = V(e);
        for (uint32_t j = 0; j < rounds; j++) { pxs[j] = V64(e); xst[j] = V(e); }
        if (e->err) return -1;
        imadw(e, pxt, tid, 4, xrow);
        imadi(e, sa0, tid, 0, 4, NOREG);
        for (uint32_t j = 0; j < rounds; j++) imadw(e, pxs[j], one, j * 128u * W, pxt);
        if (K % (32u * W)) {
            const uint32_t j = rounds - 1u;
            imadi(e, kq, tid, 0, 1, NOREG); imadi(e, kq, one, 0, j * 32u * W, kq);          /* q */
            imadi(e, tq, kq, 0, 0xffffffffu, NOREG); imadi(e, tq, one, 0, K - 1u, tq);     /* K - 1 - q */
            shr(e, over, tq, 0, 31);                                                        /* 1 iff q >= K */
            imadi(e, tq, kq, 0, 1, NOREG); imadi(e, tq, one, 0, (uint32_t)(0u - K), tq);   /* q - K */
            imadr(e, qs, over, tq, NOREG); imadi(e, qs, qs, 0, 0xffffffffu, kq);           /* K when over, else q */
            imadi(e, qg, tq, 0, 1, NOREG); imadi(e, qg, one, 0, 1, qg);                   /* q - K + 1 */
            imadr(e, qg, over, qg, NOREG); imadi(e, qg, qg, 0, 0xffffffffu, kq);           /* K - 1 when over, else q */
            imadw(e, pxs[j], qg, 4, xrow);
            imadi(e, sap, qs, 0, 4, NOREG);
        }
        for (uint32_t j = 0; j < rounds; j++) ldg32(e, xst[j], pxs[j], udesc, j == 0 ? K_VAR : K_VAR_MORE);
        for (uint32_t j = 0; j < rounds; j++) {
            if (K % (32u * W) && j == rounds - 1u) { sts32(e, sap, xst[j]); continue; }
            imadi(e, sa, one, 0, j * 128u * W, sa0); sts32(e, sa, xst[j]);
        }
        bar_sync(e);
        imadi(e, xl0, lane, 0, LB * 4u, NOREG);     /* this lane's x' chunk of step 0, shared byte offset */
    }

    /* ---- the loads of one row group into set B (rows * KS weight loads, then KS scale words) ---- */
#define ISSUE_B() do { \
        imadw(e, pw, rgl, R * K, wbase); imadw(e, pw, wlane, 1, pw); \
        shr(e, srg, rgl, 0, (uint32_t)log2rows_per_block); \
        imadw(e, ps, srg, KB * 4u, sbase); imadw(e, ps, slane, 1, ps); \
        int kind = K_LDB; \
        for (uint32_t s_ = 0; s_ < KS; s_++) for (uint32_t r_ = 0; r_ < R; r_++) { ldg_wide(e, ldw, B.w[r_][s_], pw, r_ * K + s_ * 32u * LB, udesc, kind); kind = K_LDB_MORE; } \
        for (uint32_t s_ = 0; s_ < KS; s_++) { \
            if (s_ == 0) ldg32(e, B.sc[0], ps, udesc, kind); \
            else { imadw(e, psd, one, spec->mutant == 2 ? 0u : s_ * BPS * 4u, ps); ldg32(e, B.sc[s_], psd, udesc, K_LDB_MORE); } \
            kind = K_LDB_MORE; \
        } \
        imadi(e, rgl, stride, 0, 1, rgl); \
    } while (0)

    /* ---- prologue: nothing to do, or the first row group's loads ---- */
    isetp_ge_u32(e, rgl, rgn);
    const int bra_none = bra_if_p0_fwd(e);
    ISSUE_B();

    /* ---- row-group loop: take set B over, issue the next row group into B, compute set A, reduce, store ---- */
    const int L_rg = here(e);
    loop_begin(e);
    {
        int kind = K_MOVB;
        for (uint32_t s = 0; s < KS; s++) for (uint32_t r = 0; r < R; r++) for (uint32_t q = 0; q < WORDS; q++) { movsub(e, A.w[r][s], q, B.w[r][s], q, kind); kind = K_FIXED; }
        for (uint32_t s = 0; s < KS; s++) movsub(e, A.sc[s], 0, B.sc[s], 0, K_FIXED);
    }
    isetp_ge_u32(e, rgl, rgn);
    const int bra_skip_issue = bra_if_p0_fwd(e);
    ISSUE_B();
    const int L_compute = here(e);
    patch_bra(e, bra_skip_issue, L_compute);
    for (uint32_t r = 0; r < R; r++) movrz(e, acc[r]);
    for (uint32_t s = 0; s < KS; s++) {
        for (uint32_t r = 0; r < R; r++) movrz(e, dot[r]);
        for (uint32_t word = 0; word < WORDS; word++) {
            lds128(e, xq, xl0, s * 128u * LB + word * 16u);
            for (uint32_t r = 0; r < R; r++)
                for (unsigned i = 0; i < 4; i++) {
                    decode_byte(e, f, A.w[r][s], word, i, u, a, sg);
                    ffma(e, dot[r], f, xq, i, dot[r]);
                }
        }
        for (uint32_t r = 0; r < R; r++) {
            if (spec->mutant == 1) fadd(e, acc[r], acc[r], dot[r]);
            else ffma(e, acc[r], dot[r], A.sc[s], 0, acc[r]);
        }
    }
    /* warp reduction, lane 0 stores (variant 1's sequence) */
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, acc[r], o); fadd(e, acc[r], acc[r], t); }
    imadw(e, pyr, rgc, R * 4u, yrow);
    isetp_ge_u32(e, lane, one);
    const int at_bssy = bssy_fwd(e);
    const int bra_skip = bra_if_p0_fwd(e);
    for (uint32_t r = 0; r < R; r++) { imadw(e, pya, one, r * 4u, pyr); stg(e, pya, 0, acc[r], udesc); }
    const int at_bsync = here(e);
    bsync(e);
    patch_bra(e, bra_skip, at_bsync);
    patch_bra(e, at_bssy, at_bsync + 1);
    imadi(e, rgc, stride, 0, 1, rgc);
    isetp_ge_u32(e, rgc, rgn);
    const int bra_done = bra_if_p0_fwd(e);
    bra(e, L_rg);
    loop_end(e);
#undef ISSUE_B
    const int L_end = here(e);
    patch_bra(e, bra_none, L_end);
    patch_bra(e, bra_done, L_end);
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_fp8_gemv2_prog_stats(const OmegaFp8GemvSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog; /* 1024 insns: too big for a stack frame in a test loop */
    if (omega_blackwell_codegen_fp8_gemv2_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}

#ifndef OMEGA_GPU_NO_POSIX
#include <stdlib.h>
int omega_blackwell_codegen_fp8_gemv2(const OmegaFp8GemvSpec *spec, OmegaBlackwellKernel *kernel) {
    if (!kernel) return -1;
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    if (!prog) return -1;
    int rc = omega_blackwell_codegen_fp8_gemv2_prog(spec, prog);
    size_t need = rc == 0 ? omega_bw_program_code_bytes(prog) : 0;
    uint8_t *buf = need ? malloc(need) : NULL;
    if (!buf) { free(prog); return -1; }
    rc = omega_blackwell_kernel_encode_into(prog, buf, need, kernel);
    free(prog);
    if (rc != 0) { free(buf); return -1; }
    return 0; /* kernel->code points into buf; omega_blackwell_kernel_free releases it */
}
#endif
