/* omega_blackwell_codegen_tern.c: KT1 W2A8 ternary GEMV kernel, ternary engine stage T0 (2026-10-15). See the header.
 * The emitter helpers are copies of omega_blackwell_codegen_bf16.c's (same chip-learned scheduling rules: every
 * instruction waits on everything still pending; variable-latency results go through scoreboard 0, stores set the read
 * barrier 1, uniform loads scoreboard 2; ISETP stalls 14 before a predicated branch;
 * the loads of one iteration are issued back to back so a warp has all of them in flight at once). One change: the
 * fixed-latency stall is no longer 6 on every instruction; tern_schedule_stalls sets each one from the physical register
 * dependencies so every user still sits >= 6 cycles behind its producer (see the pass). */
#include "omega_blackwell_codegen_tern.h"
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
/* d = a[asub] * imm + c (c NOREG = RZ) */
static void imadi(Em *e, int d, int a, unsigned asub, uint32_t imm, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.imm = imm; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* d = a * b[bsub] + c (register form) */
static void imadr(Em *e, int d, int a, int b, unsigned bsub, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; i.src3_vreg = c; emit(e, i, K_FIXED); }
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
static void fmul(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FMUL); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void i2f(Em *e, int d, int a) { BlackwellIRInsn i = I0(BW_IR_I2FP); i.dst_vreg = d; i.src1_vreg = a; emit(e, i, K_FIXED); }
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
/* predicated region: lanes with P0 set skip to the join (bra_if_p0 inside BSSY .. BSYNC, the bf16 store pattern) */
typedef struct { int bssy, bra; } Region;
static Region region_begin(Em *e) { Region r; r.bssy = bssy_fwd(e); r.bra = bra_if_p0_fwd(e); return r; }
static void region_end(Em *e, Region r) { int at = here(e); bsync(e); patch_bra(e, r.bra, at); patch_bra(e, r.bssy, at + 1); }

uint32_t omega_tern_gemv_chunks(const OmegaTernGemvSpec *s) {
    if (!s || s->k == 0) return 0;
    return (s->k + OMEGA_TERN_CHUNK_MAX - 1u) / OMEGA_TERN_CHUNK_MAX;
}

int omega_tern_gemv_spec_ok(const OmegaTernGemvSpec *s) {
    if (!s) return 0;
    if (s->k == 0 || s->k > OMEGA_TERN_MAX_K || s->k % 256u) return 0;
    const uint32_t nch = omega_tern_gemv_chunks(s);
    if (s->k % (256u * nch)) return 0;
    if (s->layout == OMEGA_TERN_LAYOUT_ROWMAJOR) { if (s->rows != 1 && s->rows != 2 && s->rows != 4) return 0; }
    else if (s->layout == OMEGA_TERN_LAYOUT_BLOCK) { if (s->rows != 2 && s->rows != 4) return 0; }
    else return 0;
    if (s->warps < 1 || s->warps > 32) return 0;
    const uint32_t ksc = s->k / nch / 256u;
    if (s->depth == 0 || s->depth > 16 || ksc % s->depth || s->rows * s->depth > 16u) return 0;
    if (s->epi != OMEGA_TERN_EPI_I32 && s->epi != OMEGA_TERN_EPI_F32) return 0;
    if (s->epi == OMEGA_TERN_EPI_F32 && nch != 1) return 0;
    if (s->mutant < 0 || s->mutant > 3) return 0;
    return 1;
}
uint32_t omega_tern_gemv_steps(const OmegaTernGemvSpec *s) { return omega_tern_gemv_spec_ok(s) ? s->k / 256u : 0; }
uint32_t omega_tern_gemv_shared_bytes(const OmegaTernGemvSpec *s) {
    return omega_tern_gemv_spec_ok(s) ? (s->k / omega_tern_gemv_chunks(s)) * 2u : 0;
}

uint64_t omega_tern_word_offset(uint32_t layout, uint32_t R, uint32_t k, uint32_t r, uint32_t c) {
    const uint64_t kw = k / 16u;
    if (layout == OMEGA_TERN_LAYOUT_ROWMAJOR) return ((uint64_t)r * kw + c) * 4u;
    const uint64_t blk = r / (2u * R), j = r % (2u * R);
    return blk * 2u * R * kw * 4u + (uint64_t)c * 2u * R * 4u + j * 4u;
}


/* ---- dependency-driven stall counts (after register allocation) ----
 * The copied emitter gives every fixed-latency instruction stall 6, which keeps any user >= 6 cycles behind its
 * producer but also makes independent instructions wait 6 cycles each. The repository's chip-proven rule is the
 * distance, not the per-instruction count: a fixed-latency producer stalls >= 5 cycles before its consumer (we keep 6;
 * omega_gpu_elementwise_api.c:13). This pass walks the program in order on physical registers and gives each
 * instruction the smallest stall that keeps the next instruction >= OMEGA_TERN_FIXED_LAT cycles behind every
 * fixed-latency write it reads or overwrites. Before a control instruction (BRA, EXIT, BSSY, BSYNC, BAR) or a branch
 * target every fixed-latency write has completed, so straight-line timing never spans a jump. Instructions that are not
 * fixed latency keep at least their emitted stall (loads, ISETP 14 before a predicated branch, the scoreboard rules);
 * their wait masks are untouched. Hardware can only add cycles, which only lengthens every distance. */
static int tern_fixed(BlackwellIROpcode op) {
    switch (op) {
    case BW_IR_IMAD: case BW_IR_IMAD_WIDE: case BW_IR_SHF_R: case BW_IR_LOP3_AND: case BW_IR_MOV_IMM: case BW_IR_MOV_RZ:
    case BW_IR_FMUL: case BW_IR_I2FP: return 1;
    default: return 0;
    }
}
static int tern_control(BlackwellIROpcode op) {
    return op == BW_IR_BRA || op == BW_IR_EXIT || op == BW_IR_BSSY || op == BW_IR_BSYNC || op == BW_IR_BAR_SYNC;
}
/* physical registers an operand covers: [*lo, *lo + *cnt) */
static void tern_span(const BlackwellIRProgram *p, int v, int *lo, int *cnt) {
    *lo = -1; *cnt = 0;
    if (v < 0 || v >= p->regalloc.num_vregs) return;
    const int ph = p->regalloc.vreg_to_phys[v];
    if (ph < 0) return;
    *lo = ph; *cnt = (int)(p->regalloc.intervals[v].bundle_size ? p->regalloc.intervals[v].bundle_size : 1u);
}
static int tern_schedule_stalls(BlackwellIRProgram *p) {
    static uint8_t target[BW_MAX_IR_INSNS + 1];
    uint64_t ready[BW_MAX_VREGS + 8];
    const size_t n = p->count;
    memset(target, 0, sizeof target);
    memset(ready, 0, sizeof ready);
    for (size_t i = 0; i < n; i++) {
        const BlackwellIRInsn *in = &p->insns[i];
        if (in->op == BW_IR_BRA || in->op == BW_IR_BSSY) {
            const long t = (long)i + (long)(int32_t)in->imm;
            if (t < 0 || t > (long)n) return -1;
            target[t] = 1;
        }
    }
    uint64_t T = 0, all = 0;
    for (size_t i = 0; i < n; i++) {
        BlackwellIRInsn *in = &p->insns[i];
        const uint32_t s_emit = (in->control >> 9) & 0xfu;
        if (tern_fixed(in->op)) {
            int lo, c; tern_span(p, in->dst_vreg, &lo, &c);
            for (int k = 0; k < c; k++) { if (lo + k >= BW_MAX_VREGS + 8) return -1; ready[lo + k] = T + OMEGA_TERN_FIXED_LAT; }
            if (c && T + OMEGA_TERN_FIXED_LAT > all) all = T + OMEGA_TERN_FIXED_LAT;
        }
        uint64_t need = T + (tern_fixed(in->op) ? 1u : s_emit);
        if (i + 1 < n) {
            const BlackwellIRInsn *nx = &p->insns[i + 1];
            const int ops[4] = { nx->dst_vreg, nx->src1_vreg, nx->src2_vreg, nx->src3_vreg };
            for (int q = 0; q < 4; q++) {
                int lo, c; tern_span(p, ops[q], &lo, &c);
                for (int k = 0; k < c; k++) if (ready[lo + k] > need) need = ready[lo + k];
            }
            if ((tern_control(nx->op) || target[i + 1]) && all > need) need = all;
        }
        const uint64_t s = need - T;
        if (s > 15 || (s == 0 && (tern_fixed(in->op) || s_emit != 0))) return -1; /* stall 0 only where emitted (the final self-branch) */
        if (tern_fixed(in->op) || s > s_emit) in->control = (in->control & ~(0xfu << 9)) | ((uint32_t)s << 9);
        T = need;
    }
    return 0;
}
int omega_blackwell_codegen_tern_gemv_prog(const OmegaTernGemvSpec *spec, BlackwellIRProgram *prog) {
    if (!omega_tern_gemv_spec_ok(spec) || !prog) return -1;
    const uint32_t K = spec->k, R = spec->rows, W = spec->warps, D = spec->depth, TH = 32u * W;
    const uint32_t NCH = omega_tern_gemv_chunks(spec), KC = K / NCH, KSC = KC / 256u, KWC = KC / 16u;
    const int block = spec->layout == OMEGA_TERN_LAYOUT_BLOCK, f32 = spec->epi == OMEGA_TERN_EPI_F32;
    const uint32_t row_bytes = K / 4u;                         /* packed bytes per row (whole k) */
    const uint32_t step_bytes = block ? 128u * R : 64u;       /* bytes a lane's weight pointer advances per K step */
    const uint32_t chunk_bytes = block ? KC * R / 2u : KC / 4u; /* ... and per chunk */
    const uint32_t npairs = spec->mutant == 3 ? 7u : 8u;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    Em *e = &em;

    /* ---- prologue ---- */
    int udesc = UV64(e), pwb = V64(e), px = V64(e), py = V64(e), ureg = V(e), stride = V(e), negn = V(e);
    int tid = V(e), cx = V(e), warp = V(e), lane = V(e), h = V(e), l = V(e), one = V(e), t = V(e), nzl1 = V(e);
    int u0 = V(e), u = V(e), xl = V(e), hR = V(e), lo = V(e), pl = V64(e), k0 = V(e);
    int pws = NOREG, pas = NOREG, asv = NOREG, nb[3] = { NOREG, NOREG, NOREG };
    int chc = NOREG, mul = NOREG, prev = NOREG;
    if (f32) { pws = V64(e); pas = V64(e); asv = V(e); for (int i = 0; i < 3; i++) nb[i] = V(e); }
    if (NCH > 1) { chc = V(e); mul = V(e); prev = V(e); }
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pwb, OMEGA_TERN_ARG_W); ldc64(e, px, OMEGA_TERN_ARG_X); ldc64(e, py, OMEGA_TERN_ARG_Y);
    ldc32(e, ureg, OMEGA_TERN_ARG_U); ldc32(e, stride, OMEGA_TERN_ARG_STRIDE); ldc32(e, negn, OMEGA_TERN_ARG_NEGN);
    if (f32) {
        ldc64(e, pws, OMEGA_TERN_ARG_WS); ldc64(e, pas, OMEGA_TERN_ARG_AS);
        for (int i = 0; i < 3; i++) ldc32(e, nb[i], OMEGA_TERN_ARG_NEGB + 4u * (uint32_t)i);
    }
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X);
    movi(e, one, 1);
    shr(e, warp, tid, 0, 5);
    andi(e, lane, tid, 0, 31);
    shr(e, h, lane, 0, 4);
    andi(e, l, lane, 0, 15);
    imadi(e, t, l, 0, 0xffffffffu, NOREG);   /* -l */
    shr(e, t, t, 0, 31);                     /* 1 if l > 0 */
    imadi(e, nzl1, t, 0, 1, one);            /* (l > 0) + 1 */
    imadi(e, u0, cx, 0, W, warp);            /* first unit of this warp */
    imadi(e, xl, l, 0, 32, NOREG);           /* shared byte offset of word c = l */
    imadi(e, hR, h, 0, R, NOREG);
    if (block) { imadi(e, lo, l, 0, 8u * R, NOREG); imadi(e, lo, h, 0, 4u * R, lo); }
    else { imadi(e, lo, l, 0, 4u, NOREG); imadi(e, lo, h, 0, R * row_bytes, lo); }
    imadw(e, pl, lo, 1, pwb);                /* this lane's weight pointer in unit 0, chunk 0 */
    movi(e, k0, 0x80008000u);
    if (f32) ldg32(e, asv, pas, udesc, 1);
    if (NCH > 1) { movi(e, chc, NCH); movrz(e, mul); }

    /* ---- chunk loop (NCH > 1 only for k > OMEGA_TERN_CHUNK_MAX): x staged per chunk, y accumulated in place ---- */
    const int L_c = here(e);
    if (NCH > 1) { loop_begin(e); bar_sync(e); }

    /* ---- stage x: per word c of the chunk, X_j = x[16c+j] * 65536 + x[16c+j+8] (j < 8), 32 bytes ---- */
    {
        const uint32_t rounds = (KWC + TH - 1) / TH;
        int c = V(e), kwr = V(e), rc = V(e), pxa = V64(e), xq = V128(e), sb = V(e), va = V(e), vb = V(e), X = V(e), sad = V(e);
        if (e->err) return -1;
        movi(e, kwr, KWC);
        imadi(e, c, tid, 0, 1, NOREG);
        movi(e, rc, rounds);
        const int L_s = here(e);
        loop_begin(e);
        isetp_ge_u32(e, c, kwr);
        Region rg = region_begin(e);
        imadw(e, pxa, c, 16u, px);
        ldg_wide(e, BW_IR_LDG_E_128, xq, pxa, 0, udesc, 1);
        imadi(e, sb, c, 0, 32u, NOREG);
        for (uint32_t j = 0; j < 8; j++) {
            const unsigned sh = 8u * (j & 3u);
            int src[2] = { va, vb }; const unsigned sub[2] = { j >> 2, 2u + (j >> 2) };
            for (int q = 0; q < 2; q++) {
                if (sh) { shr(e, src[q], xq, sub[q], sh); andi(e, src[q], src[q], 0, 0xffu); }
                else andi(e, src[q], xq, sub[q], 0xffu);
                andi(e, t, src[q], 0, 0x80u);
                imadi(e, src[q], t, 0, 0xfffffffeu, src[q]); /* byte - 2 * (byte & 0x80): sign extension */
            }
            imadi(e, X, va, 0, 65536u, vb);
            if (spec->mutant == 1) imadi(e, X, X, 0, 0xffffffffu, NOREG); /* flipped planes: plus acts as minus */
            imadi(e, sad, one, 0, 4u * j, sb);
            sts32(e, sad, X);
        }
        region_end(e, rg);
        imadi(e, c, one, 0, TH, c);
        imadi(e, rc, one, 0, 0xffffffffu, rc);
        isetp_ge_u32(e, rc, one);
        bra_if_p0(e, L_s);
        loop_end(e);
        bar_sync(e);
    }

    /* ---- per-unit registers ---- */
    int rowbase = V(e), pw = V64(e), xlc = V(e), cnt = V(e);
    int T[4], wq[16], wr[4][16], wa[4][16], xs[8], Ut[4], P[4], M[4], ap[4], am[4];
    int row = V(e), okr = V(e), skip = V(e), yv = V(e), f = V(e), wsv = V(e), pya = V64(e), pwsa = V64(e), sum = V(e);
    for (uint32_t r = 0; r < R; r++) { T[r] = V(e); Ut[r] = V(e); P[r] = V(e); M[r] = V(e); ap[r] = V(e); am[r] = V(e); }
    if (block) { for (uint32_t d = 0; d < D; d++) wq[d] = R == 4 ? V128(e) : V64(e); }
    else for (uint32_t r = 0; r < R; r++) for (uint32_t d = 0; d < D; d++) { wr[r][d] = V(e); wa[r][d] = V64(e); }
    for (uint32_t i = 0; i < (D <= 4u ? 2u * D : 2u); i++) xs[i] = V128(e);
    if (e->err) return -1;

    /* ---- unit loop ---- */
    imadi(e, u, u0, 0, 1, NOREG);
    const int L_u = here(e);
    loop_begin(e);
    isetp_ge_u32(e, u, ureg);
    const int bra_end = bra_if_p0_fwd(e);
    imadi(e, rowbase, u, 0, 2u * R, hR);
    imadw(e, pw, u, 2u * R * row_bytes, pl);
    imadi(e, xlc, xl, 0, 1, NOREG);
    for (uint32_t r = 0; r < R; r++) movrz(e, T[r]);
    movi(e, cnt, KSC / D);

    /* ---- K loop: D steps per iteration, every weight load issued before any use ----
     * Rows are interleaved stage by stage (shift, two masks, two multiply-adds for every row before the next pair) so
     * independent work sits between each producer and its user; tern_schedule_stalls then sets every stall from the
     * real register dependencies. For D <= 4 the staged x of all D steps is read from shared memory together with the
     * weight loads (one wait per iteration); above that per step as before. */
    const int lds_up = D <= 4u;
    const int L_k = here(e);
    loop_begin(e);
    if (block) {
        for (uint32_t d = 0; d < D; d++) ldg_wide(e, R == 4 ? BW_IR_LDG_E_128 : BW_IR_LDG_E_64, wq[d], pw, d * step_bytes, udesc, d == 0);
    } else {
        for (uint32_t r = 0; r < R; r++) for (uint32_t d = 0; d < D; d++) imadw(e, wa[r][d], one, r * row_bytes + d * step_bytes, pw);
        int first = 1;
        for (uint32_t r = 0; r < R; r++) for (uint32_t d = 0; d < D; d++) { ldg32(e, wr[r][d], wa[r][d], udesc, first); first = 0; }
    }
    if (lds_up) for (uint32_t d = 0; d < D; d++) for (uint32_t i = 0; i < 2; i++) lds128(e, xs[2u * d + i], xlc, d * 512u + 16u * i, 0);
    for (uint32_t d = 0; d < D; d++) {
        const int *xd = lds_up ? &xs[2u * d] : xs;
        if (!lds_up) for (uint32_t i = 0; i < 2; i++) lds128(e, xs[i], xlc, d * 512u + 16u * i, i == 0);
        for (uint32_t j = 0; j < npairs; j++) {
            for (uint32_t r = 0; r < R && j; r++) shr(e, Ut[r], block ? wq[d] : wr[r][d], block ? r : 0u, 2u * j);
            for (uint32_t r = 0; r < R; r++) andi(e, P[r], j ? Ut[r] : (block ? wq[d] : wr[r][d]), j ? 0u : (block ? r : 0u), 0x00010001u);
            for (uint32_t r = 0; r < R; r++) andi(e, M[r], j ? Ut[r] : (block ? wq[d] : wr[r][d]), j ? 0u : (block ? r : 0u), 0x00020002u);
            for (uint32_t r = 0; r < R; r++) imadr(e, ap[r], P[r], xd[j >> 2], j & 3u, j == 0 ? k0 : ap[r]);   /* + P * X_j */
            for (uint32_t r = 0; r < R; r++) imadr(e, am[r], M[r], xd[j >> 2], j & 3u, j == 0 ? k0 : am[r]);   /* + M * X_j (minus bits one higher) */
        }
        for (uint32_t r = 0; r < R; r++) shr(e, P[r], ap[r], 0, 16);       /* 2^15 + plus dot of this word */
        for (uint32_t r = 0; r < R; r++) shr(e, M[r], am[r], 0, 17);       /* 2^14 + minus dot of this word */
        for (uint32_t r = 0; r < R; r++) imadi(e, T[r], P[r], 0, 1, T[r]);
        for (uint32_t r = 0; r < R; r++) imadi(e, T[r], M[r], 0, 0xffffffffu, T[r]);
    }
    imadw(e, pw, one, D * step_bytes, pw);
    imadi(e, xlc, one, 0, D * 512u, xlc);
    imadi(e, cnt, one, 0, 0xffffffffu, cnt);
    isetp_ge_u32(e, cnt, one);
    bra_if_p0(e, L_k);
    loop_end(e);

    /* ---- half-warp reduction (lanes 0 and 16 hold the sums), lane l == 0 stores rows < n ---- */
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t o = 8; o >= 1; o /= 2) { shfl_down(e, t, T[r], o); imadi(e, T[r], t, 0, 1, T[r]); }
    for (uint32_t r = 0; r < R; r++) {
        imadi(e, row, one, 0, r, rowbase);
        imadi(e, okr, row, 0, 1, negn);
        shr(e, okr, okr, 0, 31);                       /* 1 if row < n */
        imadi(e, skip, okr, 0, 0xffffffffu, nzl1);     /* (l > 0) + 1 - (row < n): 0 only for a store */
        isetp_ge_u32(e, skip, one);
        Region rg = region_begin(e);
        imadi(e, yv, one, 0, 0u - KSC * (1u << 18), T[r]); /* y = T - KSC * 2^18 (mod 2^32, exact) */
        imadw(e, pya, row, 4u, py);
        if (NCH > 1) { ldg32(e, prev, pya, udesc, 1); imadr(e, yv, prev, mul, 0, yv); } /* + earlier chunks */
        if (f32) {
            for (int i = 0; i < 3; i++) {
                imadi(e, t, row, 0, 1, nb[i]);
                shr(e, t, t, 0, 31);                   /* 1 if row < b_i */
                if (i == 0) imadi(e, sum, t, 0, 1, NOREG); else imadi(e, sum, t, 0, 1, sum);
            }
            imadw(e, pwsa, sum, 4u, pws);              /* ws_rev[boundaries above row] = w_scale[seg] */
            ldg32(e, wsv, pwsa, udesc, 1);
            i2f(e, f, yv);
            fmul(e, f, f, asv);
            fmul(e, f, f, wsv);
            stg(e, pya, 0, f, udesc);
        } else stg(e, pya, 0, yv, udesc);
        region_end(e, rg);
    }
    imadi(e, u, stride, 0, 1, u);
    bra(e, L_u);
    loop_end(e);
    patch_bra(e, bra_end, here(e));

    if (NCH > 1) {
        imadw(e, px, one, KC, px);
        imadw(e, pl, one, chunk_bytes, pl);
        imadi(e, mul, one, 0, 1, NOREG);
        imadi(e, chc, one, 0, 0xffffffffu, chc);
        isetp_ge_u32(e, chc, one);
        bra_if_p0(e, L_c);
        loop_end(e);
    }
    tail(e);
    if (e->err) return -1;
    if (omega_bw_regalloc_solve(prog) != 0) return -1;
    return tern_schedule_stalls(prog);
}

int omega_tern_gemv_prog_stats(const OmegaTernGemvSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_tern_gemv_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}
