/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * omega_blackwell_codegen_fattn256.c: flash-style causal prefill attention on tensor cores, one warp per 16-row tile,
 * no shared memory (see the header; Linear DRA-85). The emitter helpers are omega_blackwell_codegen_pattn256.c's, copied
 * unchanged (control words, the read barrier on every load, held scoreboards 3/4 for prefetch loads, loop live ranges,
 * HMMA spacing: >= 4 independent HMMAs at stall 8 between dependent ones, stall 15 + NOP 14 before a non-HMMA reader of
 * an HMMA result) minus the shared-memory ones, plus a held-scoreboard global prefetch (ldg_pf, lds_pf's twin) and an
 * LDG.64 into a quad's sub-register pair (ldg64s).
 */
#include "omega_blackwell_codegen_fattn256.h"
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
static void stg(Em *e, int addr64, uint32_t imm, Op val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.imm = imm; i.src2_vreg = val.v; i.src2_subreg = (uint8_t)val.s; i.ureg = udesc; emit(e, i, K_STORE); }
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
static void consume(Em *e, unsigned sb) { e->pending |= e->held & (1u << sb); e->held &= ~(1u << sb); }

/* a prefetch global load: scoreboard sb (3 or 4) is held, not waited for, until consume() */
static void ldg_pf(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, unsigned sb) {
    BlackwellIRInsn in = I0(op); in.dst_vreg = d; in.src1_vreg = addr64; in.imm = imm; in.ureg = udesc;
    in.control = CW(4, sb, SB_RD, (e->pending & (1u << SB_RD))) | CW_YIELD;
    e->pending |= 1u << SB_RD; e->held |= 1u << sb;
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}
/* LDG.64 into a sub-register pair (0 or 2) of a quad */
static void ldg64s(Em *e, Op d, int addr64, uint32_t imm, int udesc, int first) {
    BlackwellIRInsn i = I0(BW_IR_LDG_E_64); i.dst_vreg = d.v; i.dst_subreg = (uint8_t)d.s; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc;
    emit(e, i, first ? K_VAR : K_VAR_MORE);
}
/* variant 2: an LDG.64 into a sub-register pair whose value is never read (it pulls the line into L1); scoreboard 5 is
 * held until consume(5) */
static void ldg_touch(Em *e, Op d, int addr64, uint32_t imm, int udesc) {
    BlackwellIRInsn in = I0(BW_IR_LDG_E_64); in.dst_vreg = d.v; in.dst_subreg = (uint8_t)d.s; in.src1_vreg = addr64; in.imm = imm; in.ureg = udesc;
    in.control = CW(4, 5u, SB_RD, (e->pending & (1u << SB_RD))) | CW_YIELD;
    e->pending |= 1u << SB_RD; e->held |= 1u << 5;
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}

static const uint32_t NEG_BIG = 0xF149F2CAu; /* -1e30f */

int omega_fattn256_shape(const OmegaPrefillFattn256Spec *spec, OmegaFattn256Shape *s) {
    const uint32_t qh = spec->q_heads ? spec->q_heads : 16u, kvh = spec->kv_heads ? spec->kv_heads : 2u;
    if (kvh == 0 || qh % kvh || qh / kvh > 8u) return -1;
    s->qh = qh; s->kvh = kvh; s->group = qh / kvh; s->out_words = qh * 256u; s->qf_pair_bytes = qh * 1024u;
    return 0;
}

int omega_fattn256_spec_ok(const OmegaPrefillFattn256Spec *spec) {
    OmegaFattn256Shape s;
    return spec && spec->mutant >= 0 && spec->mutant <= 2 && spec->variant >= 0 && spec->variant <= 2 && omega_fattn256_shape(spec, &s) == 0;
}

int omega_blackwell_codegen_prefill_fattn256_prog(const OmegaPrefillFattn256Spec *spec, BlackwellIRProgram *prog) {
    if (!omega_fattn256_spec_ok(spec) || !prog) return -1;
    omega_bw_ir_init(prog);
    if (omega_bw_ir_set_gpr_limit(prog, BW_PHYS_GPR_HW_LIMIT) != 0) return -1;
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    const int MUT = spec->mutant, V2 = spec->variant == 2;
    OmegaFattn256Shape SH;
    if (omega_fattn256_shape(spec, &SH) != 0) return -1;

    /* ---- arguments and identities; warps past the chunk exit (warp-uniform, before any barrier or shuffle) ---- */
    int udesc = UV64(e);
    int pqf = V64(e), pkc = V64(e), pvc = V64(e);
    int pos0 = V(e), mc = V(e), kvs = V(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), one = V(e), zero = V(e);
    int t1 = V(e), t2 = V(e), tokA = V(e), cnt = V(e), dqa = V(e), tt2 = V(e);
    /* the loop-carried scalars are defined first so the linear-scan allocator packs them below the quads */
    int mA = V(e), mB = V(e), lA = V(e), lB = V(e), gK = V64(e), gV = V64(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pqf, OMEGA_FATTN256_ARG_QF); ldc64(e, pkc, OMEGA_FATTN256_ARG_KC); ldc64(e, pvc, OMEGA_FATTN256_ARG_VC);
    ldc32(e, pos0, OMEGA_FATTN256_ARG_POS0); ldc32(e, mc, OMEGA_FATTN256_ARG_MC); ldc32(e, kvs, OMEGA_FATTN256_ARG_KV_STRIDE);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, R(one), 1); movrz(e, R(zero));
    movrz(e, R(t1)); movrz(e, R(t2)); movrz(e, R(cnt)); movrz(e, R(dqa)); movrz(e, R(tt2));
    movi(e, R(mA), NEG_BIG); movi(e, R(mB), NEG_BIG); movrz(e, R(lA)); movrz(e, R(lB));
    imadw(e, gK, zero, 0u, pkc); imadw(e, gV, zero, 0u, pvc);
    shr(e, warp, tid, 5);
    andi(e, R(lane), R(tid), 31u);
    imadi(e, R(t1), R(warp), 2u, NOREG);
    imadi(e, R(tokA), R(cx), OMEGA_FATTN256_TOKENS, t1); /* the warp's row-A token */
    isetp_ge_u32(e, tokA, mc);
    const int b_exit = bra_if_p0_fwd(e);

    /* ---- Q: 16 A fragments, k-permuted (header), 32 LDG.64 into the fragment halves ---- */
    int q[16];
    for (unsigned s = 0; s < 16; s++) q[s] = V128(e);
    int g = V(e), tq = V(e);
    if (e->err) return -1;
    shr(e, g, lane, 2); andi(e, R(tq), R(lane), 3u);
    {
        int hg = V(e), c5 = V(e), ad64 = V64(e); if (e->err) return -1;
        movi(e, R(c5), SH.group - 1u);
        imadi(e, R(t1), R(g), 0xffffffffu, c5);          /* group-1 - g */
        shr(e, t2, t1, 31);                              /* 1 when g > group-1 */
        imadr(e, hg, t2, t1, g);                         /* min(g, group-1) */
        shr(e, t1, tokA, 1);                             /* token pair (tokA < mc, so its slot was written) */
        imadi(e, R(t1), R(t1), SH.kvh, cy); imadi(e, R(t1), R(t1), SH.group, hg); imadi(e, R(t1), R(t1), 1024u, NOREG);
        shr(e, t2, tq, 1); imadi(e, R(t1), R(t2), 64u, t1);
        andi(e, R(t2), R(tq), 1u); imadi(e, R(t1), R(t2), 8u, t1);
        imadw(e, ad64, t1, 1u, pqf);
        for (unsigned u = 0; u < 8; u++)
            for (unsigned j = 0; j < 4; j++)
                ldg64s(e, S(q[2u * u + (j >> 1)], (j & 1u) * 2u), ad64, u * 128u + j * 16u, udesc, u == 0 && j == 0);
    }
    iadd(e, dqa, pos0, tokA);                            /* row-A position - kb (kb = 0) */
    /* blocks: the warp's last key is row B's position dqa + 1; nblk = (dqa + 1) / 16 + 1 */
    imadi(e, R(t1), R(one), 1u, dqa); shr(e, t1, t1, 4); imadi(e, R(cnt), R(one), 1u, t1);
    imadi(e, R(tt2), R(tq), 2u, NOREG);

    /* ---- per-lane cache pointers: K rows kb + g (+ 8), 16 bytes at 16t; V rows kb + 2t (+1, +8, +9), 8 bytes at 64g ---- */
    int o[32], sc[4], kA[1], kB[1], pp, vb[2][4], bf[1][2];
    for (unsigned i = 0; i < 32; i++) o[i] = V128(e);
    int zq = V128(e);
    kA[0] = V128(e); kB[0] = V128(e);
    if (e->err) return -1;
    {
        int cyk = V(e); if (e->err) return -1;
        imadr(e, cyk, cy, kvs, NOREG);                   /* byte offset of this KV head's plane */
        imadw(e, gK, cyk, 1u, pkc); imadw(e, gK, g, 512u, gK); imadi(e, R(t1), R(tq), 16u, NOREG); imadw(e, gK, t1, 1u, gK);
        imadw(e, gV, cyk, 1u, pvc); imadw(e, gV, tq, 1024u, gV); imadi(e, R(t1), R(g), 64u, NOREG); imadw(e, gV, t1, 1u, gV);
    }
    /* O = 0 through the tensor core (0 x 0 + RZ, 32 instructions instead of 128 moves); zq dies here */
    for (unsigned k = 0; k < 4; k++) movrz(e, S(zq, k));
    for (unsigned i = 0; i < 32; i++) hmma(e, o[i], zq, zq, 0, NOREG, 8);

    int mxA = V(e), mxB = V(e), alA = V(e), alB = V(e), sh = V(e);
    for (unsigned j = 0; j < 4; j++) sc[j] = V128(e);
    pp = V128(e);
    for (unsigned i = 0; i < 2; i++) for (unsigned r = 0; r < 4; r++) vb[i][r] = V64(e);
    for (unsigned j = 0; j < 2; j++) bf[0][j] = V128(e); /* rebuilt every step; the step's 4 HMMAs at stall 8 cover the WAR */
    if (e->err) return -1;

    const int L_m = here(e);
    loop_begin(e);
    /* ---- S = Q K^T: 8 dim groups x (2 key tiles x 2 k-steps), even / odd k-steps into separate accumulators.
     *      One K quad pair per group (Q 64 + O 128 + S 16 registers leave no room for a second). The other warp of
     *      the sub-partition (8 warps on 4) covers an L1 hit with its 4 HMMAs; the first touch of a new 128-byte
     *      line goes to L2 and is not covered (MODELED, DRA-85 PLAN: 45-115 TFLOPS by how much of it shows) ---- */
    for (unsigned u = 0; u < 8; u++) {
        ldg_pf(e, BW_IR_LDG_E_128, kA[0], gK, u * 64u, udesc, 3);
        ldg_pf(e, BW_IR_LDG_E_128, kB[0], gK, 8u * 512u + u * 64u, udesc, 3);
        consume(e, 3);
        hmma(e, sc[0], q[2u * u], kA[0], 0, u ? sc[0] : NOREG, 8);
        hmma(e, sc[1], q[2u * u], kB[0], 0, u ? sc[1] : NOREG, 8);
        hmma(e, sc[2], q[2u * u + 1u], kA[0], 2, u ? sc[2] : NOREG, 8);
        hmma(e, sc[3], q[2u * u + 1u], kB[0], 2, u ? sc[3] : NOREG, 8);
    }
    nop(e, 15); nop(e, 14);
    for (unsigned k = 0; k < 4; k++) { fadd(e, S(sc[0], k), S(sc[0], k), S(sc[2], k)); fadd(e, S(sc[1], k), S(sc[1], k), S(sc[3], k)); }

    /* ---- variant 2: the next block's 128-byte K and V lines into L1 while this block's softmax and PV run, so its
     *      QK^T loads hit L1 instead of a first-touch L2 round trip. Each lane touches rows g, g + 8 of K at lines
     *      0..3 (8 LDG.64 from gK; lanes t share a line) and V rows 2t, 2t + 1, 2t + 8, 2t + 9 at line g >> 1 (4 from gV):
     *      all 64 + 64 lines. The data lands in sc[3], dead from here to the next QK^T (its value is never read);
     *      scoreboard 5 is held to the loop end. Skipped on the warp's last block (no next block: no rows past it are
     *      read). Not register-resident K: Q 64 + O 128 + S 16 leave no room for 64 more registers per lane. ---- */
    if (V2) {
        static const uint32_t TV[4] = { 0u, 512u, 4096u, 4608u };
        isetp_ge_u32(e, one, cnt);                       /* cnt <= 1: the last block */
        const int b = bra_if_p0_fwd(e);
        for (unsigned i = 0; i < 12; i++)
            ldg_touch(e, S(sc[3], (i & 1u) * 2u), i < 8 ? gK : gV, i < 8 ? 16u * 512u + (i >> 2) * 4096u + (i & 3u) * 128u : 16u * 512u + TV[i - 8u], udesc);
        patch_bra(e, b, here(e));
    }

    /* ---- causal mask: key kb + 8j + 2t + (k&1) > row position (row A for k < 2, A + 1 for k >= 2) ---- */
    if (MUT != 1) {
        imadi(e, R(t1), R(one), 0xfffffff1u, dqa); shr(e, t1, t1, 31);      /* 1 when kb + 15 > row-A position */
        isetp_ge_u32(e, zero, t1);
        const int b = bra_if_p0_fwd(e);
        imadi(e, R(t2), R(tt2), 0xffffffffu, dqa);                          /* dqa - 2t */
        for (unsigned j = 0; j < 2; j++) for (unsigned k = 0; k < 4; k++) {
            int32_t cst = (k >= 2 ? 1 : 0) - 8 * (int32_t)j - (int32_t)(k & 1u);
            imadi(e, R(t1), R(one), (uint32_t)cst, t2); shr(e, t1, t1, 31); imadi(e, R(t1), R(t1), NEG_BIG, NOREG);
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
    if (MUT != 2) for (unsigned i = 0; i < 32; i++) for (unsigned k = 0; k < 4; k++) fmul(e, S(o[i], k), S(o[i], k), R(k < 2 ? alA : alB));
    f2fp(e, S(pp, 0), S(sc[0], 1), S(sc[0], 0)); f2fp(e, S(pp, 1), S(sc[0], 3), S(sc[0], 2));
    f2fp(e, S(pp, 2), S(sc[1], 1), S(sc[1], 0)); f2fp(e, S(pp, 3), S(sc[1], 3), S(sc[1], 2));

    /* ---- O += P V: 8 half steps (h, half) x 4 n-tiles; the value rows double-buffered on scoreboards 3/4 ----
     * row r of a step: keys kb + 2t, kb + 2t + 1, kb + 8 + 2t, kb + 9 + 2t (byte offsets 0, 512, 4096, 4608 from gV) */
    static const uint32_t ROWOFF[4] = { 0u, 512u, 4096u, 4608u };
    for (unsigned r = 0; r < 4; r++) ldg_pf(e, BW_IR_LDG_E_64, vb[0][r], gV, ROWOFF[r], udesc, 3);
    consume(e, 3);
    for (unsigned st = 0; st < 8; st++) {
        const unsigned cur = st & 1u, nx = cur ^ 1u, sbn = 3u + nx, h = st >> 1, hq = st & 1u;
        if (st < 7) {
            const unsigned h2 = (st + 1u) >> 1, q2 = (st + 1u) & 1u;
            for (unsigned r = 0; r < 4; r++) ldg_pf(e, BW_IR_LDG_E_64, vb[nx][r], gV, ROWOFF[r] + 16u * h2 + 8u * q2, udesc, sbn);
        }
        for (unsigned jj = 0; jj < 2; jj++) {
            const int f = bf[0][jj];
            /* b0 / b1 of the even n-tile: low halves (dim 2w) of the key pairs; odd n-tile: the high halves */
            andi(e, R(t1), S(vb[cur][0], jj), 0xffffu); imadi(e, S(f, 0), S(vb[cur][1], jj), 65536u, t1);
            andi(e, R(t1), S(vb[cur][2], jj), 0xffffu); imadi(e, S(f, 1), S(vb[cur][3], jj), 65536u, t1);
            shrs(e, t1, S(vb[cur][0], jj), 16); andi(e, R(t2), S(vb[cur][1], jj), 0xffff0000u); fop(e, BW_IR_IADD3, S(f, 2), R(t1), R(t2), R(NOREG));
            shrs(e, t1, S(vb[cur][2], jj), 16); andi(e, R(t2), S(vb[cur][3], jj), 0xffff0000u); fop(e, BW_IR_IADD3, S(f, 3), R(t1), R(t2), R(NOREG));
        }
        for (unsigned jj = 0; jj < 2; jj++) {
            const unsigned n = 8u * h + 4u * hq + 2u * jj;
            hmma(e, o[n], pp, bf[0][jj], 0, o[n], 8); hmma(e, o[n + 1u], pp, bf[0][jj], 2, o[n + 1u], 8);
        }
        if (st < 7) consume(e, sbn);
    }
    if (V2) { consume(e, 5); imadi(e, S(sc[3], 0), S(sc[3], 0), 1u, NOREG); } /* the touches landed; keeps sc[3] live to here */
    imadw(e, gK, one, 16u * 512u, gK); imadw(e, gV, one, 16u * 512u, gV);
    imadi(e, R(dqa), R(one), 0xfffffff0u, dqa);
    imadi(e, R(cnt), R(one), 0xffffffffu, cnt);
    isetp_ge_u32(e, cnt, one);
    bra_if_p0(e, L_m);
    loop_end(e);

    /* ---- epilogue: rows g < group (lane-divergent only when group < 8), row B only when tokA + 1 < mc ---- */
    nop(e, 15); nop(e, 14);
    {
        int pout = V64(e), osc = V(e), ostr = V(e), po = V64(e), po2 = V64(e), eg = V(e), etq = V(e), etid = V(e), ecx = V(e), ecy = V(e), emc = V(e);
        int ewarp = V(e), elane = V(e), etok = V(e), cg = V(e);
        if (e->err) return -1;
        ldc64(e, pout, OMEGA_FATTN256_ARG_OUT); ldc32(e, osc, OMEGA_FATTN256_ARG_OUT_SCALE); ldc32(e, emc, OMEGA_FATTN256_ARG_MC);
        ldc32(e, ostr, OMEGA_FATTN256_ARG_OSTRIDE);
        s2r(e, etid, BW_SR_TID_X); s2r(e, ecx, BW_SR_CTAID_X); s2r(e, ecy, BW_SR_CTAID_Y);
        shr(e, ewarp, etid, 5); andi(e, R(elane), R(etid), 31u);
        imadi(e, R(t1), R(ewarp), 2u, NOREG); imadi(e, R(etok), R(ecx), OMEGA_FATTN256_TOKENS, t1);
        shfl_bfly(e, sh, lA, 1); fadd(e, R(lA), R(lA), R(sh)); shfl_bfly(e, sh, lA, 2); fadd(e, R(lA), R(lA), R(sh));
        shfl_bfly(e, sh, lB, 1); fadd(e, R(lB), R(lB), R(sh)); shfl_bfly(e, sh, lB, 2); fadd(e, R(lB), R(lB), R(sh));
        mufu(e, BW_IR_MUFU_RCP, R(lA), R(lA), 1); mufu(e, BW_IR_MUFU_RCP, R(lB), R(lB), 0);
        fmul(e, R(lA), R(lA), R(osc)); fmul(e, R(lB), R(lB), R(osc));
        for (unsigned i = 0; i < 32; i++) for (unsigned k = 0; k < 4; k++) fmul(e, S(o[i], k), S(o[i], k), R(k < 2 ? lA : lB));
        /* out + etok * ostride + ((cy * group + g) * 256 + 64t) * 4; row B one ostride further */
        shr(e, eg, elane, 2); andi(e, R(etq), R(elane), 3u);
        imadr(e, t1, etok, ostr, NOREG); imadw(e, po, t1, 1u, pout);
        imadi(e, R(t1), R(ecy), SH.group, eg); imadi(e, R(t1), R(t1), 1024u, NOREG); imadi(e, R(t1), R(etq), 256u, t1);
        imadw(e, po, t1, 1u, po);
        imadw(e, po2, ostr, 1u, po);
        movi(e, R(cg), SH.group);
        isetp_ge_u32(e, eg, cg);
        Region rg = region_skip_if_p0(e);
        for (unsigned row = 0; row < 2; row++) {
            int b = -1;
            if (row) { imadi(e, R(t1), R(one), 1u, etok); isetp_ge_u32(e, t1, emc); b = bra_if_p0_fwd(e); }
            for (unsigned i = 0; i < 32; i++) {
                stg(e, row ? po2 : po, i * 4u, S(o[i], 2u * row), udesc);
                stg(e, row ? po2 : po, (32u + i) * 4u, S(o[i], 2u * row + 1u), udesc);
            }
            if (row) patch_bra(e, b, here(e));
        }
        region_end(e, rg);
    }
    patch_bra(e, b_exit, here(e));
    tail(e);
    if (e->err) return -1;
    if (omega_bw_regalloc_solve(prog) != 0) return -1;
    return 0;
}

int omega_prefill_fattn256_prog_stats(const OmegaPrefillFattn256Spec *spec, uint32_t *insns, uint32_t *gprs, uint32_t *placeholders) {
    static BlackwellIRProgram p;
    if (omega_blackwell_codegen_prefill_fattn256_prog(spec, &p) != 0) return -1;
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
