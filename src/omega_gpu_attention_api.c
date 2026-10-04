/* omega_gpu_attention_api.c: FB-1 cut 5 (2026-10-04). See the header.
 *
 * Layout: (1) kernel codegen on the BlackwellIR with the serialising scheduler
 * of cut 4 (every instruction waits on everything still pending; slow and safe),
 * extended with forward branches and more loops; (2) the launcher, cloned from
 * cut 4's run_abc with a fourth buffer (block tables and context lengths), 16
 * argument words, and cut 1b's shader-cache invalidate; (3) the public entry
 * points, which stage the referenced KV blocks per launch and apply the oracle's
 * argument rules (dropped tokens past the table, negative table entries, zero
 * context) on the host.
 *
 * One CTA per (query head, sequence), 64 threads (= head_dim). The context is
 * walked in chunks of 64 tokens:
 *   phase A  thread t owns token base+t: block lookup, K dot q (f32 FFMA), s = dot * scale
 *            (scale folds 1/sqrt(d) and log2 e), s and the token's byte offset go to shared;
 *   reduce   chunk max (SHFL.DOWN tree per warp, lane partials to shared, BAR.SYNC, 2 LDS),
 *            m_new = max(m, chunk max), alpha = EX2(m - m_new), p = EX2(s - m_new) to shared,
 *            chunk sum the same way, l = l * alpha + chunk sum;
 *   phase B  thread d owns dimension d: acc = acc * alpha + sum over the chunk of p_t * V[t][d].
 * Then out[d] = acc * RCP(l). The first chunk sees m = -inf, so alpha = EX2(-inf) = 0 and
 * the initial l and acc drop out (PTX ISA ex2.approx: ex2(-Inf) = +0).
 *
 * Shared memory (1024 bytes, the QMD's declared size): [0,256) q, [256,512) s then p,
 * [512,768) token byte offsets, [768,1024) warp partials for the reductions.
 */
#include "omega_gpu_attention_api.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_submit.h"
#include "omega_gpu_session.h"
#include "sha256.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ codegen */

#define CW(stall, wbar, rbar, wait) \
    (((uint32_t)(stall) << 9) | ((uint32_t)(wbar) << 14) | ((uint32_t)(rbar) << 17) | ((uint32_t)(wait) << 20))
#define CW_YIELD 0x2000u
#define SB_NONE 7u
#define SB_GPR 0u
#define SB_RD 1u
#define SB_UGPR 2u

enum { K_FIXED, K_PRED, K_VAR, K_UVAR, K_STORE, K_SYNC, K_EXIT, K_BRA };

#define AT_MAX_LOOPS 8
typedef struct {
    BlackwellIRProgram *p;
    uint32_t pending;
    int err;
    int loop_start[AT_MAX_LOOPS], loop_end[AT_MAX_LOOPS];
    int n_loops, depth, stack[AT_MAX_LOOPS];
} Em;

static void emit(Em *e, BlackwellIRInsn in, int kind) {
    uint32_t wait = e->pending;
    switch (kind) {
    case K_FIXED: in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_PRED:  in.control = CW(13, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_VAR:   in.control = CW(4, SB_GPR, SB_NONE, wait) | CW_YIELD; e->pending = 1u << SB_GPR; break;
    case K_UVAR:  in.control = CW(4, SB_UGPR, SB_NONE, wait) | CW_YIELD; e->pending = 1u << SB_UGPR; break;
    case K_STORE: in.control = CW(4, SB_NONE, SB_RD, wait) | CW_YIELD; e->pending = 1u << SB_RD; break;
    case K_SYNC:  in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_EXIT:  in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    case K_BRA:   in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    default: e->err = -1; return;
    }
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}

static int V(Em *e) { int v = omega_bw_ir_alloc_vreg(e->p); if (v < 0) e->err = -1; return v; }
static int V64(Em *e) { int v = omega_bw_ir_alloc_vreg64(e->p); if (v < 0) e->err = -1; return v; }
static int UV64(Em *e) { int v = omega_bw_ir_alloc_uvreg64(e->p); if (v < 0) e->err = -1; return v; }
static int here(const Em *e) { return (int)e->p->count; }

#define NOREG -1
static BlackwellIRInsn I0(BlackwellIROpcode op) {
    BlackwellIRInsn in; memset(&in, 0, sizeof in);
    in.op = op; in.dst_vreg = NOREG; in.src1_vreg = NOREG; in.src2_vreg = NOREG; in.src3_vreg = NOREG; in.ureg = NOREG;
    return in;
}
static void ldcu64(Em *e, int udst, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDCU64); i.dst_vreg = udst; i.imm = off; i.is_uniform = true; emit(e, i, K_UVAR); }
static void ldc64(Em *e, int dst64, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDC64); i.dst_vreg = dst64; i.imm = off; emit(e, i, K_VAR); }
static void ldc32(Em *e, int dst, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_LDC); i.dst_vreg = dst; i.imm = off; emit(e, i, K_VAR); }
static void s2r(Em *e, int dst, uint32_t sr) { BlackwellIRInsn i = I0(BW_IR_S2R); i.dst_vreg = dst; i.imm = sr; emit(e, i, K_VAR); }
static void movi(Em *e, int dst, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_MOV_IMM); i.dst_vreg = dst; i.imm = imm; emit(e, i, K_FIXED); }
static void movf(Em *e, int dst, float f) { uint32_t u; memcpy(&u, &f, 4); movi(e, dst, u); }
static void movrz(Em *e, int dst) { BlackwellIRInsn i = I0(BW_IR_MOV_RZ); i.dst_vreg = dst; emit(e, i, K_FIXED); }
/* dst = a * imm + c (c may be NOREG for RZ) */
static void imadi(Em *e, int dst, int a, uint32_t imm, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = dst; i.src1_vreg = a; i.imm = imm; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* dst = a * b + c (registers; c may be NOREG) */
static void imadr(Em *e, int dst, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = dst; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void iadd3(Em *e, int dst, int a, int b) { BlackwellIRInsn i = I0(BW_IR_IADD3); i.dst_vreg = dst; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void lop3xor(Em *e, int dst, int a, int b) { BlackwellIRInsn i = I0(BW_IR_LOP3_XOR); i.dst_vreg = dst; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void lop3andi(Em *e, int dst, int a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = dst; i.src1_vreg = a; i.imm = imm; emit(e, i, K_FIXED); }
static void shri(Em *e, int dst, int a, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = dst; i.src1_vreg = a; i.imm = imm; emit(e, i, K_FIXED); }
/* dst64 = idx * imm + base64 (byte address) */
static void wide(Em *e, int dst64, int idx, uint32_t imm, int base64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = dst64; i.src1_vreg = idx; i.imm = imm; i.src3_vreg = base64; emit(e, i, K_FIXED); }
static void ldg(Em *e, int dst, int addr64, int udesc) { BlackwellIRInsn i = I0(BW_IR_LDG_E); i.dst_vreg = dst; i.src1_vreg = addr64; i.ureg = udesc; emit(e, i, K_VAR); }
static void ldg16(Em *e, int dst, int addr64, int udesc) { BlackwellIRInsn i = I0(BW_IR_LDG_E_U16); i.dst_vreg = dst; i.src1_vreg = addr64; i.ureg = udesc; emit(e, i, K_VAR); }
static void stg(Em *e, int addr64, int val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.src2_vreg = val; i.ureg = udesc; emit(e, i, K_STORE); }
static void e_fadd(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FADD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void e_fsub(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FSUB); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void e_fmul(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FMUL); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void e_ffma(Em *e, int d, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_FFMA); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void e_fmax(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FMNMX_MAX); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void mufu(Em *e, BlackwellIROpcode op, int d, int a) { BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = a; emit(e, i, K_VAR); }
static void shfl_down(Em *e, int d, int a, uint32_t off) { BlackwellIRInsn i = I0(BW_IR_SHFL_DOWN); i.dst_vreg = d; i.src1_vreg = a; i.imm = off; emit(e, i, K_VAR); }
static void sts32(Em *e, int addr, int val) { BlackwellIRInsn i = I0(BW_IR_STS32); i.src1_vreg = addr; i.src2_vreg = val; emit(e, i, K_STORE); }
static void lds32(Em *e, int d, int addr) { BlackwellIRInsn i = I0(BW_IR_LDS32); i.dst_vreg = d; i.src1_vreg = addr; emit(e, i, K_VAR); }
static void bar_sync(Em *e) { emit(e, I0(BW_IR_BAR_SYNC), K_SYNC); }
static void isetp_ge_u32(Em *e, int a, int b) { BlackwellIRInsn i = I0(BW_IR_ISETP_GE_U32); i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_PRED); }
static void bra_back_if_not_p0(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.predicate_not = true; i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
/* @P0 BRA forward; the target is patched in when known */
static int bra_fwd_if_p0(Em *e) { int at = here(e); BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.imm = 0; emit(e, i, K_BRA); return at; }
static void patch_fwd(Em *e, int at) { if (at >= 0 && (size_t)at < e->p->count) e->p->insns[at].imm = (uint32_t)(here(e) - at); else e->err = -1; }
static void tail(Em *e) {
    emit(e, I0(BW_IR_EXIT), K_EXIT);
    BlackwellIRInsn self = I0(BW_IR_BRA);
    self.control = 0x000fc000u;
    if (omega_bw_ir_append(e->p, &self) < 0) e->err = -1;
}
/* Loops nest; loop_end() extends every value defined before the loop and read inside
 * it to the back-branch (the linear-scan allocator knows nothing about back-edges). */
static void loop_begin(Em *e) {
    if (e->depth >= AT_MAX_LOOPS || e->n_loops >= AT_MAX_LOOPS) { e->err = -1; return; }
    e->stack[e->depth++] = here(e);
}
static void loop_end(Em *e) {
    if (e->depth == 0 || e->n_loops >= AT_MAX_LOOPS) { e->err = -1; return; }
    int start = e->stack[--e->depth], end = here(e) - 1;
    e->loop_start[e->n_loops] = start; e->loop_end[e->n_loops] = end; e->n_loops++;
    OmegaRegAlloc *ra = &e->p->regalloc;
    for (int v = 0; v < ra->num_vregs; v++) {
        OmegaLiveInterval *iv = &ra->intervals[v];
        if (iv->first_def >= 0 && iv->first_def < start && iv->last_use >= start && iv->last_use < end) iv->last_use = end;
    }
}
/* A value first defined inside a loop may not be read after it before being redefined. */
static int check_loop_invariant(const Em *e) {
    const OmegaRegAlloc *ra = &e->p->regalloc;
    for (int L = 0; L < e->n_loops; L++)
        for (int v = 0; v < ra->num_vregs; v++) {
            int fd = ra->intervals[v].first_def;
            if (fd < e->loop_start[L] || fd > e->loop_end[L]) continue;
            for (size_t i = (size_t)e->loop_end[L] + 1; i < e->p->count; i++) {
                const BlackwellIRInsn *in = &e->p->insns[i];
                if (in->src1_vreg == v || in->src2_vreg == v || in->src3_vreg == v) return -1;
                if (in->dst_vreg == v && !in->is_uniform) break;
            }
        }
    return 0;
}

/* Kernel argument words (constant bank 0 at 0x380; the bank is 1024 bytes, so 32 words fit). */
#define ARG_Q 0x380u
#define ARG_POOL 0x388u
#define ARG_OUT 0x390u
#define ARG_TAB 0x398u
#define ARG_MAXB 0x3a0u    /* words per block-table row */
#define ARG_SCALE 0x3a4u   /* bits of log2(e) / sqrt(head_dim) */
#define ARG_BSTRIDE 0x3a8u /* bytes between staged blocks */
#define ARG_TSTRIDE 0x3acu /* bytes between tokens */
#define ARG_KVP 0x3b0u     /* bytes from the K plane to the V plane */
#define ARG_HSTRIDE 0x3b4u /* bytes between kv heads */
#define ARG_NQH 0x3b8u     /* query heads (q and out row count per sequence) */
#define ARG_CTXOFF 0x3bcu  /* byte offset of the context-length array inside the table buffer */
#define ARG_WORDS 16u
#define DESC_OFF 0x358u

#define HD OMEGA_GPU_ATTN_HEAD_DIM
#define SH_Q 0u
#define SH_S 256u
#define SH_O 512u
#define SH_R 768u
#define F32_NEG_INF 0xff800000u

typedef struct { bool f32; uint32_t log2bs, log2gqa; int mutant; } KSpec;

static void gen_attention(Em *e, const KSpec *k) {
    const uint32_t elem = k->f32 ? 4u : 2u;
    const uint32_t kstep = k->f32 ? 8u : 4u; /* bytes per inner iteration: two dims */
    int udesc = UV64(e), A = V64(e), B = V64(e), C = V64(e), D = V64(e);
    int tid = V(e), h = V(e), seq = V(e);
    int p_maxb = V(e), scale = V(e), p_bstride = V(e), p_tstride = V(e), p_kvp = V(e), p_hstride = V(e), p_nqh = V(e), p_ctxoff = V(e);
    ldcu64(e, udesc, DESC_OFF);
    ldc64(e, A, ARG_Q); ldc64(e, B, ARG_POOL); ldc64(e, C, ARG_OUT); ldc64(e, D, ARG_TAB);
    ldc32(e, p_maxb, ARG_MAXB); ldc32(e, scale, ARG_SCALE); ldc32(e, p_bstride, ARG_BSTRIDE); ldc32(e, p_tstride, ARG_TSTRIDE);
    ldc32(e, p_kvp, ARG_KVP); ldc32(e, p_hstride, ARG_HSTRIDE); ldc32(e, p_nqh, ARG_NQH); ldc32(e, p_ctxoff, ARG_CTXOFF);
    s2r(e, tid, BW_SR_TID_X); s2r(e, h, BW_SR_CTAID_X); s2r(e, seq, BW_SR_CTAID_Y);

    int c1 = V(e), c4 = V(e), c32 = V(e), c64 = V(e), c256 = V(e), c512 = V(e), c768 = V(e), c896 = V(e), zero = V(e);
    movi(e, c1, 1); movi(e, c4, 4); movi(e, c32, 32); movi(e, c64, 64);
    movi(e, c256, SH_S); movi(e, c512, SH_O); movi(e, c768, SH_R); movi(e, c896, SH_R + 128); movrz(e, zero);

    /* q[tid] -> shared; out address kept for the end. Row = seq * num_q_heads + head for
     * both q and out (flattened head-major, the oracle's layout). The head-wiring mutants
     * swap the row of head h with h ^ 1 on one side only. */
    int tid4 = V(e), qrow = V(e), qoff = V(e), qaddr = V64(e), qv = V(e), oaddr = V64(e);
    int hq = h, ho = h;
    if (k->mutant == OMEGA_GPU_ATTN_MUTANT_Q_ROW) { hq = V(e); lop3xor(e, hq, h, c1); }
    if (k->mutant == OMEGA_GPU_ATTN_MUTANT_OUT_ROW) { ho = V(e); lop3xor(e, ho, h, c1); }
    imadi(e, tid4, tid, 4, NOREG);
    imadr(e, qrow, seq, p_nqh, hq);
    imadi(e, qoff, qrow, HD * 4, tid4);
    wide(e, qaddr, qoff, 1, A);
    ldg(e, qv, qaddr, udesc);
    sts32(e, tid4, qv); /* SH_Q + tid*4 */
    if (ho == hq) wide(e, oaddr, qoff, 1, C);
    else {
        int orow = V(e), ooff = V(e);
        imadr(e, orow, seq, p_nqh, ho);
        imadi(e, ooff, orow, HD * 4, tid4);
        wide(e, oaddr, ooff, 1, C);
    }

    /* kv head offset, this sequence's table row and context length */
    int kvh = V(e), kvoff = V(e), t1 = V(e), tab = V64(e), ca = V64(e), ctx = V(e), vb = V64(e);
    shri(e, kvh, h, k->log2gqa);
    if (k->mutant == OMEGA_GPU_ATTN_MUTANT_KV_HEAD) lop3xor(e, kvh, kvh, c1);
    imadr(e, kvoff, kvh, p_hstride, NOREG);
    imadr(e, t1, seq, p_maxb, NOREG);
    imadi(e, t1, t1, 4, NOREG);
    wide(e, tab, t1, 1, D);
    imadi(e, t1, seq, 4, p_ctxoff);
    wide(e, ca, t1, 1, D);
    ldg(e, ctx, ca, udesc);
    /* V column base for dimension tid: pool + kv plane + tid * elem. The kv head offset is
     * already inside the per-token offset phase A leaves in shared memory (found by the host
     * simulator: adding it here too read V from the wrong head for every kv head but 0). */
    wide(e, vb, tid, elem, B);
    wide(e, vb, p_kvp, 1, vb);

    int M = V(e), L = V(e), acc = V(e), base = V(e), sOff = V(e), oOff = V(e), rOff = V(e);
    movi(e, M, F32_NEG_INF);
    if (k->mutant == OMEGA_GPU_ATTN_MUTANT_NO_MAX) movrz(e, L); else movf(e, L, 1.0f); /* alpha = 0 on the first chunk clears it */
    movrz(e, acc); movrz(e, base);
    iadd3(e, sOff, tid4, c256); iadd3(e, oOff, tid4, c512); iadd3(e, rOff, tid4, c768);
    bar_sync(e); /* q visible to every thread */
    isetp_ge_u32(e, zero, ctx); /* ctx == 0: out = 0 */
    int bZ = bra_fwd_if_p0(e);

    /* ---------------- chunk loop ---------------- */
    int j = V(e), s = V(e), koff = V(e);
    int bi = V(e), sl = V(e), ta = V64(e), blk = V(e), ka = V64(e), dot = V(e), d = V(e);
    int x = V(e), lo = V(e), hi = V(e), qa = V(e), qb = V(e), q0 = V(e), q1 = V(e), ka2 = V64(e), j1 = V(e);
    int dneg = V(e), over = V(e), ot = V(e), ctxm1 = V(e), cm1 = V(e), cninf = V(e), nb = V(e), ns = V(e);
    movi(e, cm1, 0xFFFFFFFFu); iadd3(e, ctxm1, ctx, cm1); movi(e, cninf, F32_NEG_INF);
    loop_begin(e);
    int Lout = here(e);
    /* No branch here: a warp split by "j >= ctx" stays split (the IR has no reconvergence
     * instruction) and the SHFL reductions below then read inactive lanes, which is undefined
     * (wrong dims tid >= tail on the chip, 2026-10-04). over = (j >= ctx) as 0/1 from the sign of
     * ctx-1-j; out-of-range lanes read token `base` (always < ctx here) and their score
     * is forced to -inf after the dot product with s = -max(-s, nb), nb = -inf or +inf. */
    iadd3(e, j, base, tid);
    movi(e, s, F32_NEG_INF);
    movrz(e, koff);
    imadi(e, dneg, j, 0xFFFFFFFFu, ctxm1); /* ctx - 1 - j */
    shri(e, over, dneg, 31);
    imadr(e, ot, over, tid, NOREG);
    imadi(e, j, ot, 0xFFFFFFFFu, j);       /* j - over*tid */
    /* phase A: token j */
    shri(e, bi, j, k->log2bs);
    lop3andi(e, sl, j, (1u << k->log2bs) - 1u);
    wide(e, ta, bi, 4, tab);
    ldg(e, blk, ta, udesc);
    imadr(e, koff, blk, p_bstride, NOREG);
    if (k->mutant == OMEGA_GPU_ATTN_MUTANT_SLOT) {
        /* K read from slot (t + 1) % block_size, V from the right slot: a K/V mismatch.
         * (Shifting both only permutes matched pairs inside a full block, which softmax cannot see.) */
        int sl1 = V(e);
        iadd3(e, j1, j, c1); lop3andi(e, sl1, j1, (1u << k->log2bs) - 1u);
        imadr(e, sl1, sl1, p_tstride, koff);
        iadd3(e, sl1, sl1, kvoff);
        wide(e, ka, sl1, 1, B);
        imadr(e, koff, sl, p_tstride, koff);
        iadd3(e, koff, koff, kvoff);
    } else {
        imadr(e, koff, sl, p_tstride, koff);
        iadd3(e, koff, koff, kvoff);
        wide(e, ka, koff, 1, B);
    }
    movrz(e, dot); movrz(e, d);
    loop_begin(e);
    int Lin = here(e);
    if (k->f32) {
        ldg(e, lo, ka, udesc);
        wide(e, ka2, c1, 4, ka);
        ldg(e, hi, ka2, udesc);
    } else {
        ldg(e, x, ka, udesc);
        imadi(e, lo, x, 0x10000u, NOREG);   /* low bf16 -> f32: bits << 16 */
        lop3andi(e, hi, x, 0xffff0000u);    /* high bf16 -> f32: keep the top half */
    }
    imadi(e, qa, d, 8, NOREG);
    lds32(e, q0, qa);
    iadd3(e, qb, qa, c4);
    lds32(e, q1, qb);
    e_ffma(e, dot, q0, lo, dot);
    e_ffma(e, dot, q1, hi, dot);
    wide(e, ka, c1, kstep, ka);
    iadd3(e, d, d, c1);
    isetp_ge_u32(e, d, c32);
    bra_back_if_not_p0(e, Lin);
    loop_end(e);
    e_fmul(e, s, dot, scale);
    imadi(e, nb, over, 0x80000000u, cninf); /* over ? +inf : -inf (bits) */
    e_fsub(e, ns, zero, s); e_fmax(e, ns, ns, nb); e_fsub(e, s, zero, ns);
    sts32(e, sOff, s);
    sts32(e, oOff, koff);

    /* chunk max */
    int t = V(e), mw = V(e), ra = V(e), rb = V(e), mc = V(e), Mn = V(e), diff = V(e), alpha = V(e), parg = V(e), p = V(e), sw = V(e), lc = V(e);
    static const uint32_t steps[5] = { 16, 8, 4, 2, 1 };
    shfl_down(e, t, s, steps[0]); e_fmax(e, mw, s, t);
    for (int i = 1; i < 5; i++) { shfl_down(e, t, mw, steps[i]); e_fmax(e, mw, mw, t); }
    sts32(e, rOff, mw);
    bar_sync(e);
    lds32(e, ra, c768); lds32(e, rb, c896);
    e_fmax(e, mc, ra, rb);
    e_fmax(e, Mn, M, mc);
    if (k->mutant == OMEGA_GPU_ATTN_MUTANT_NO_MAX) {
        movf(e, alpha, 1.0f);
        mufu(e, BW_IR_MUFU_EX2, p, s);
        e_fmax(e, M, Mn, mc);
    } else {
        e_fsub(e, diff, M, Mn);
        mufu(e, BW_IR_MUFU_EX2, alpha, diff);
        e_fmax(e, M, Mn, mc); /* M = Mn */
        e_fsub(e, parg, s, Mn);
        mufu(e, BW_IR_MUFU_EX2, p, parg);
    }
    bar_sync(e); /* everyone has read the max partials before they are overwritten */
    sts32(e, sOff, p);
    shfl_down(e, t, p, steps[0]); e_fadd(e, sw, p, t);
    for (int i = 1; i < 5; i++) { shfl_down(e, t, sw, steps[i]); e_fadd(e, sw, sw, t); }
    sts32(e, rOff, sw);
    bar_sync(e);
    lds32(e, ra, c768); lds32(e, rb, c896);
    e_fadd(e, lc, ra, rb);
    e_ffma(e, L, L, alpha, lc);
    if (k->mutant != OMEGA_GPU_ATTN_MUTANT_NO_RESCALE) e_fmul(e, acc, acc, alpha);

    /* phase B: dimension tid over the chunk's tokens */
    int kk = V(e), tt = V(e), oa = V(e), ko = V(e), pa = V(e), pv = V(e), va = V64(e), v = V(e);
    movrz(e, kk);
    loop_begin(e);
    int LB = here(e);
    iadd3(e, tt, base, kk);
    isetp_ge_u32(e, tt, ctx);
    int bB = bra_fwd_if_p0(e);
    imadi(e, oa, kk, 4, c512);
    lds32(e, ko, oa);
    imadi(e, pa, kk, 4, c256);
    lds32(e, pv, pa);
    wide(e, va, ko, 1, vb);
    if (k->f32) ldg(e, v, va, udesc);
    else { ldg16(e, v, va, udesc); imadi(e, v, v, 0x10000u, NOREG); }
    e_ffma(e, acc, pv, v, acc);
    iadd3(e, kk, kk, c1);
    isetp_ge_u32(e, kk, c64);
    bra_back_if_not_p0(e, LB);
    loop_end(e);
    patch_fwd(e, bB);
    bar_sync(e); /* phase B reads of s/p and offsets finish before the next chunk writes them */
    iadd3(e, base, base, c64);
    isetp_ge_u32(e, base, ctx);
    bra_back_if_not_p0(e, Lout);
    loop_end(e);

    patch_fwd(e, bZ);
    int inv = V(e), o = V(e);
    mufu(e, BW_IR_MUFU_RCP, inv, L);
    e_fmul(e, o, acc, inv);
    stg(e, oaddr, o, udesc);
    tail(e);
}

#define AT_GPR_BUDGET 64u

static int build_kernel(const KSpec *k, OmegaBlackwellKernel *kernel, BlackwellIRProgram **keep) {
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    if (!prog) return OMEGA_GPU_ATTN_CODEGEN_FAIL;
    omega_bw_ir_init(prog);
    Em e; memset(&e, 0, sizeof e); e.p = prog;
    gen_attention(&e, k);
    if (e.depth != 0) e.err = -1;
    int rc = OMEGA_GPU_ATTN_CODEGEN_FAIL;
    int ra_rc = (e.err == 0) ? omega_bw_regalloc_solve(prog) : -1;
    int loop_rc = (ra_rc == 0) ? check_loop_invariant(&e) : -1;
#ifdef OMEGA_ATTN_DEBUG
    fprintf(stderr, "attn codegen f32=%d log2bs=%u log2gqa=%u mutant=%d emit_err=%d insns=%zu vregs=%d regalloc=%d loops=%d peak_gpr=%u\n",
            (int)k->f32, k->log2bs, k->log2gqa, k->mutant, e.err, prog->count, prog->regalloc.num_vregs, ra_rc, loop_rc, prog->regalloc.peak_gpr_usage);
#endif
    if (e.err == 0 && ra_rc == 0 && loop_rc == 0 && prog->regalloc.peak_gpr_usage <= AT_GPR_BUDGET) {
        size_t max_bytes = (size_t)BW_MAX_IR_INSNS * 16;
        memset(kernel, 0, sizeof *kernel);
        kernel->code = malloc(max_bytes);
        size_t emitted = 0;
        if (kernel->code && omega_bw_encode_program(prog, kernel->code, max_bytes, &emitted) == 0) {
            kernel->code_size = emitted;
            kernel->insn_count = emitted / 16;
            kernel->gpr_count = prog->regalloc.peak_gpr_usage;
            kernel->uniform_gpr_count = prog->regalloc.peak_ugpr_usage;
            sha256_hash(kernel->code, kernel->code_size, kernel->code_digest);
            rc = OMEGA_GPU_ATTN_OK;
        } else {
            free(kernel->code); kernel->code = NULL;
        }
    }
    if (keep && rc == OMEGA_GPU_ATTN_OK) *keep = prog; else free(prog);
    return rc;
}

/* --------------------------------------------------------- kernel cache */

/* One kernel per KSpec; the device copy is uploaded once and stays resident for
 * the process (cut 4b). The IR program is kept for the host simulator. */
static int g_mutant;
static bool g_dedupe = true; /* shared-prefix staging dedupe; false = legacy per-sequence staging (test hook) */
typedef struct { int used; KSpec k; OmegaBlackwellKernel kernel; BlackwellIRProgram *prog; NvrmMem code; } Slot;
static OmegaGpuAttnSim g_sim;
static Slot g_cache[8];
static OmegaGpuScratch g_q, g_pool, g_tab, g_out; /* staging, grown to the high-water mark, reused per call */
static bool g_hooked;
#define LOCK() omega_gpu_session_lock()
#define UNLOCK() omega_gpu_session_unlock()

void omega_gpu_attention_test_set_simulator(OmegaGpuAttnSim sim) { LOCK(); g_sim = sim; UNLOCK(); }
void omega_gpu_attention_test_set_dedupe(bool on) { LOCK(); g_dedupe = on; UNLOCK(); }

static void cache_clear_locked(void) {
    for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++)
        if (g_cache[i].used) { free(g_cache[i].kernel.code); free(g_cache[i].prog); omega_gpu_session_free(&g_cache[i].code); memset(&g_cache[i], 0, sizeof g_cache[i]); }
}

void omega_gpu_attention_cache_clear(void) {
    LOCK();
    cache_clear_locked();
    UNLOCK();
}

/* Called by the session (lock held, device still open) right before it closes. */
static void on_session_close(void) {
    cache_clear_locked();
    omega_gpu_session_scratch_free(&g_q); omega_gpu_session_scratch_free(&g_pool);
    omega_gpu_session_scratch_free(&g_tab); omega_gpu_session_scratch_free(&g_out);
}

void omega_gpu_attention_test_set_mutant(OmegaGpuAttnMutant m) {
    LOCK();
    g_mutant = (int)m;
    cache_clear_locked();
    UNLOCK();
}

/* Caller holds the lock. Builds the kernel if needed; uploads the device copy when
 * `upload` (the chip path; the simulator never needs one). */
static int kernel_for(bool f32, uint32_t log2bs, uint32_t log2gqa, bool upload, Slot **out, bool *hit) {
    KSpec k = { .f32 = f32, .log2bs = log2bs, .log2gqa = log2gqa, .mutant = g_mutant };
    Slot *s = NULL;
    for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++)
        if (g_cache[i].used && g_cache[i].k.f32 == k.f32 && g_cache[i].k.log2bs == k.log2bs && g_cache[i].k.log2gqa == k.log2gqa && g_cache[i].k.mutant == k.mutant) { /* field compare: memcmp would read struct padding */
            s = &g_cache[i]; *hit = true; break;
        }
    if (!s) {
        OmegaBlackwellKernel kern; BlackwellIRProgram *kp = NULL;
        int rc = build_kernel(&k, &kern, &kp);
        if (rc != OMEGA_GPU_ATTN_OK) return rc;
        size_t slot = 0;
        for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++) if (!g_cache[i].used) { slot = i; break; }
        s = &g_cache[slot];
        if (s->used) { free(s->kernel.code); free(s->prog); omega_gpu_session_free(&s->code); } /* full: evict slot 0 */
        memset(s, 0, sizeof *s);
        s->used = 1; s->k = k; s->kernel = kern; s->prog = kp;
        *hit = false;
    }
    if (upload && !s->code.cpu) {
        if (omega_gpu_session_alloc(s->kernel.code_size, &s->code) != 0) return OMEGA_GPU_ATTN_CHIP_FAIL;
        memcpy(s->code.cpu, s->kernel.code, s->kernel.code_size);
        __asm__ volatile("dsb sy" ::: "memory");
    }
    *out = s;
    return OMEGA_GPU_ATTN_OK;
}

int omega_gpu_attention_codegen(bool kv_f32, uint32_t log2_block_size, uint32_t log2_gqa, OmegaBlackwellKernel *kernel) {
    if (!kernel || log2_block_size > 16 || log2_gqa > 6) return OMEGA_GPU_ATTN_BAD_ARGS;
    LOCK();
    KSpec k = { .f32 = kv_f32, .log2bs = log2_block_size, .log2gqa = log2_gqa, .mutant = g_mutant };
    UNLOCK();
    return build_kernel(&k, kernel, NULL);
}

const char *omega_gpu_attention_last_error(void) { return omega_gpu_session_last_error(); }

/* ------------------------------------------------------------ launcher */

#define AT_POISON 0xffbadbadu
#define AT_WAIT_MS 600000ull

static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

typedef OmegaGpuAttnLaunch Launch;

/* Caller holds the lock. Chip path: open the shared device on first use. */
static int dev_open_locked(void) {
    if (!g_hooked) { (void)omega_gpu_session_on_close(on_session_close); g_hooked = true; }
    return omega_gpu_session_open() ? OMEGA_GPU_ATTN_OK : OMEGA_GPU_ATTN_CHIP_FAIL;
}

/* Where a call stages the KV slice it is about to hand the kernel: straight into the
 * device staging buffer on the chip path (one copy), a host buffer under the simulator.
 * Caller holds the lock; the result is valid until the next staging call. */
static void *stage_target(size_t bytes, void **host_tmp) {
    *host_tmp = NULL;
    if (g_sim) { *host_tmp = malloc(bytes ? bytes : 1); return *host_tmp; }
    if (omega_gpu_session_scratch(&g_pool, bytes) != 0) return NULL;
    return g_pool.mem.cpu;
}

/* One launch through the shared session (omega_gpu_session_launch: shader-cache
 * invalidate, constant bank, QMD0/QMD1, first marker, L2_FLUSH_DIRTY, second marker),
 * then readback and the poison scan. L->pool may already be the pool staging buffer. */
static int run_launch(const Slot *ks, const Launch *L, OmegaGpuAttnInfo *info) {
    if (omega_gpu_session_scratch(&g_q, L->q_bytes) || omega_gpu_session_scratch(&g_tab, L->tab_bytes) ||
        omega_gpu_session_scratch(&g_out, L->out_bytes) || (L->pool != g_pool.mem.cpu && omega_gpu_session_scratch(&g_pool, L->pool_bytes)))
        return OMEGA_GPU_ATTN_CHIP_FAIL;
    memcpy(g_q.mem.cpu, L->q, L->q_bytes);
    if (L->pool != g_pool.mem.cpu) memcpy(g_pool.mem.cpu, L->pool, L->pool_bytes);
    memcpy(g_tab.mem.cpu, L->tab, L->tab_bytes);
    size_t out_words = L->out_bytes / 4;
    uint32_t *out_dev = (uint32_t *)g_out.mem.cpu;
    for (size_t i = 0; i < out_words; i++) out_dev[i] = AT_POISON;
    __asm__ volatile("dsb sy" ::: "memory");

    uint32_t args[ARG_WORDS];
    args[0] = (uint32_t)g_q.mem.va; args[1] = (uint32_t)(g_q.mem.va >> 32);
    args[2] = (uint32_t)g_pool.mem.va; args[3] = (uint32_t)(g_pool.mem.va >> 32);
    args[4] = (uint32_t)g_out.mem.va; args[5] = (uint32_t)(g_out.mem.va >> 32);
    args[6] = (uint32_t)g_tab.mem.va; args[7] = (uint32_t)(g_tab.mem.va >> 32);
    memcpy(&args[8], L->params, sizeof L->params);
    OmegaGpuLaunch GL = {
        .code_va = ks->code.va, .gpr_count = AT_GPR_BUDGET, .threads_x = HD, .threads_y = 1,
        .grid_x = L->grid_x, .grid_y = L->grid_y, .num_elements = out_words,
        .args = args, .n_args = ARG_WORDS, .timeout_ms = AT_WAIT_MS,
    };
    uint64_t elapsed = 0; uint32_t marker = 0;
    if (omega_gpu_session_launch(&GL, &elapsed, &marker) != 0) return OMEGA_GPU_ATTN_CHIP_FAIL;

    memcpy(L->out, g_out.mem.cpu, L->out_bytes);
    uint32_t unwritten = 0;
    for (size_t i = 0; i < out_words; i++) if (((uint32_t *)L->out)[i] == AT_POISON) unwritten++;
    if (info) {
        info->chip_calls++;
        info->elapsed_ns += elapsed;
        info->completion_marker = marker;
        info->unwritten_words += unwritten;
        info->threads_per_cta = HD;
        info->ctas_last_launch = L->grid_x * L->grid_y;
    }
    return unwritten ? OMEGA_GPU_ATTN_UNWRITTEN : OMEGA_GPU_ATTN_OK;
}

/* Caller holds the lock. Accounts the launch's q / table / output traffic (the KV
 * staging bytes are accounted where the slices are copied). */
static int launch(const Slot *ks, const Launch *L, OmegaGpuAttnInfo *info) {
    if (info) { info->q_bytes += L->q_bytes; info->tab_bytes += L->tab_bytes; info->out_bytes += L->out_bytes; info->kv_source = OMEGA_GPU_ATTN_KV_STAGED; }
    if (g_sim) return g_sim(ks->prog, L, info);
    return run_launch(ks, L, info);
}

/* ------------------------------------------------------- public entries */

static void info_begin(OmegaGpuAttnInfo *info, const OmegaBlackwellKernel *k, bool hit) {
    if (!info) return;
    memset(info, 0, sizeof *info);
    info->kernel_cache_hit = hit;
    info->gpr_count = k->gpr_count;
    info->insn_count = (uint32_t)k->insn_count;
    snprintf(info->target_chip, sizeof info->target_chip, "NVIDIA DGX Spark (Grace Blackwell GB10)");
    info->sm_architecture = 121;
}
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static int log2_exact(uint32_t v, uint32_t *out) {
    if (v == 0 || (v & (v - 1))) return -1;
    uint32_t l = 0; while ((1u << l) != v) l++;
    *out = l; return 0;
}
static float score_scale(void) { return (float)(1.4426950408889634 / sqrt((double)HD)); }

/* Common head-shape checks; returns log2 of the gqa ratio. */
static int check_heads(uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t *log2gqa) {
    if (num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0) return OMEGA_GPU_ATTN_BAD_ARGS;
    if (head_dim != HD || num_q_heads > OMEGA_GPU_ATTN_MAX_Q_HEADS) return OMEGA_GPU_ATTN_TOO_LARGE;
    if (num_q_heads % num_kv_heads != 0 || log2_exact(num_q_heads / num_kv_heads, log2gqa) != 0) return OMEGA_GPU_ATTN_BAD_ARGS;
    return OMEGA_GPU_ATTN_OK;
}

/* Common prologue of every chip entry (lock held on success): device, kernel, info. */
static int begin(bool f32, uint32_t log2bs, uint32_t log2gqa, Slot **ks, OmegaGpuAttnInfo *info) {
    LOCK();
    int rc = g_sim ? OMEGA_GPU_ATTN_OK : dev_open_locked();
    bool hit = false;
    if (rc == OMEGA_GPU_ATTN_OK) rc = kernel_for(f32, log2bs, log2gqa, !g_sim, ks, &hit);
    if (rc != OMEGA_GPU_ATTN_OK) { UNLOCK(); return rc; }
    info_begin(info, &(*ks)->kernel, hit);
    return OMEGA_GPU_ATTN_OK;
}

int omega_gpu_gqa_attention_f32(const float *q, const float *k_cache, const float *v_cache,
                                uint32_t seq_len, uint32_t num_q_heads, uint32_t num_kv_heads,
                                uint32_t head_dim, float *out, OmegaGpuAttnInfo *info) {
    uint64_t t0 = now_ns();
    if (!q || !k_cache || !v_cache || !out) return OMEGA_GPU_ATTN_BAD_ARGS;
    uint32_t log2gqa;
    int rc = check_heads(num_q_heads, num_kv_heads, head_dim, &log2gqa);
    if (rc != OMEGA_GPU_ATTN_OK) return rc;
    if (seq_len > OMEGA_GPU_ATTN_MAX_GQA_CTX) return OMEGA_GPU_ATTN_TOO_LARGE;
    size_t n_out = (size_t)num_q_heads * HD;
    if (seq_len == 0) { memset(out, 0, n_out * 4); if (info) { memset(info, 0, sizeof *info); info->call_ns = now_ns() - t0; } return OMEGA_GPU_ATTN_OK; }
    Slot *ks = NULL;
    const uint32_t log2bs = 12; /* one block of 4096 slots */
    rc = begin(true, log2bs, log2gqa, &ks, info);
    if (rc != OMEGA_GPU_ATTN_OK) return rc;
    size_t plane = (size_t)seq_len * num_kv_heads * HD * 4;
    void *host_tmp = NULL;
    uint8_t *pool = stage_target(2 * plane, &host_tmp);
    if (!pool) { UNLOCK(); return OMEGA_GPU_ATTN_CHIP_FAIL; }
    memcpy(pool, k_cache, plane);
    memcpy(pool + plane, v_cache, plane);
    if (info) { info->kv_bytes_staged += 2 * plane; info->kv_bytes_naive += 2 * plane; info->kv_blocks_logical += 1; info->kv_blocks_unique += 1; } /* one contiguous pseudo-block */
    uint32_t tab[2] = { 0, seq_len }; /* table row {0}, then the context length at byte 4 */
    Launch L = {
        .q = q, .q_bytes = n_out * 4, .pool = pool, .pool_bytes = 2 * plane, .tab = tab, .tab_bytes = sizeof tab,
        .out = out, .out_bytes = n_out * 4, .grid_x = num_q_heads, .grid_y = 1,
        .params = { 1, f2u(score_scale()), (uint32_t)(2 * plane), num_kv_heads * HD * 4, (uint32_t)plane, HD * 4, num_q_heads, 4 }
    };
    rc = launch(ks, &L, info);
    free(host_tmp);
    UNLOCK();
    if (info) info->call_ns = now_ns() - t0;
    return rc;
}

/* Paged path shared by the single and batched entries. tables[s] has n_blocks[s] entries
 * (already filtered of negatives), ctx[s] the context length. */
static int paged_core(const float *q, const uint8_t *pool, const OmegaGpuKvLayout *ly,
                      const uint32_t *const *tables, const uint32_t *n_blocks, const uint32_t *ctx,
                      uint32_t num_seqs, uint32_t layer_idx, uint32_t num_q_heads, uint32_t num_kv_heads,
                      uint32_t head_dim, float *out, OmegaGpuAttnInfo *info) {
    uint64_t t0 = now_ns();
    uint32_t log2gqa, log2bs;
    int rc = check_heads(num_q_heads, num_kv_heads, head_dim, &log2gqa);
    if (rc != OMEGA_GPU_ATTN_OK) return rc;
    if (!ly || layer_idx >= ly->num_layers || log2_exact(ly->block_size, &log2bs) != 0) return OMEGA_GPU_ATTN_BAD_ARGS;
    /* the oracle addresses heads as kv_head * head_dim * 2 inside a token: the layout must agree */
    if (ly->head_stride_bytes != HD * 2 || ly->token_stride_bytes != (uint64_t)num_kv_heads * HD * 2 ||
        ly->kv_plane_stride_bytes != (uint64_t)ly->block_size * ly->token_stride_bytes ||
        ly->layer_stride_bytes != 2 * ly->kv_plane_stride_bytes || ly->block_stride_bytes < (uint64_t)ly->num_layers * ly->layer_stride_bytes)
        return OMEGA_GPU_ATTN_BAD_ARGS;
    size_t row = (size_t)num_q_heads * HD;
    /* effective context per sequence and block id range check */
    uint32_t *eff = calloc(num_seqs ? num_seqs : 1, sizeof *eff);
    if (!eff) return OMEGA_GPU_ATTN_CHIP_FAIL;
    for (uint32_t s = 0; s < num_seqs; s++) {
        uint64_t cap = (uint64_t)n_blocks[s] * ly->block_size;
        eff[s] = (uint32_t)(ctx[s] < cap ? ctx[s] : cap);
        for (uint32_t b = 0; b < n_blocks[s]; b++) {
            uint64_t id = tables[s][b];
            if (id >= ly->num_blocks || id * ly->block_stride_bytes + (uint64_t)(layer_idx + 1) * ly->layer_stride_bytes > ly->pool_bytes) { free(eff); return OMEGA_GPU_ATTN_BAD_ARGS; }
        }
    }
    Slot *ks = NULL;
    rc = begin(false, log2bs, log2gqa, &ks, info);
    if (rc != OMEGA_GPU_ATTN_OK) { free(eff); return rc; }
    uint32_t seqs_per_launch = OMEGA_GPU_ATTN_MAX_CTAS / num_q_heads;
    if (seqs_per_launch == 0) seqs_per_launch = 1;
    uint32_t maxb = 1;
    for (uint32_t s = 0; s < num_seqs; s++) if (n_blocks[s] > maxb) maxb = n_blocks[s];
    /* Dedupe map: physical block id -> staged slot of the current launch (-1 = not staged yet),
     * plus the list of ids touched by the launch (in slot order) so the reset is O(unique). */
    int32_t *slot_of = NULL; uint32_t *touched = NULL;
    if (g_dedupe) {
        size_t nb = ly->num_blocks ? ly->num_blocks : 1;
        slot_of = malloc(nb * sizeof *slot_of);
        touched = malloc(nb * sizeof *touched);
        if (!slot_of || !touched) { free(slot_of); free(touched); free(eff); UNLOCK(); return OMEGA_GPU_ATTN_CHIP_FAIL; }
        memset(slot_of, 0xff, nb * sizeof *slot_of); /* every entry -1 */
    }
    for (uint32_t s0 = 0; s0 < num_seqs && rc == OMEGA_GPU_ATTN_OK; s0 += seqs_per_launch) {
        uint32_t ns = num_seqs - s0 < seqs_per_launch ? num_seqs - s0 : seqs_per_launch;
        /* logical references: every sequence's blocks up to its (truncated) context */
        uint32_t total = 0;
        for (uint32_t s = 0; s < ns; s++) total += eff[s0 + s] ? (eff[s0 + s] + ly->block_size - 1) / ly->block_size : 0;
        uint32_t *tab = calloc((size_t)ns * maxb + ns, sizeof *tab);
        if (!tab) { rc = OMEGA_GPU_ATTN_CHIP_FAIL; break; }
        /* pass 1: assign staged slots. Dedupe keys on the PHYSICAL block id (not the table
         * position): a CoW prefix block shared by several branches gets one slot. Each
         * sequence's compact table keeps its own block order and truncated context. */
        uint32_t g = 0;
        for (uint32_t s = 0; s < ns; s++) {
            uint32_t need = eff[s0 + s] ? (eff[s0 + s] + ly->block_size - 1) / ly->block_size : 0;
            for (uint32_t b = 0; b < need; b++) {
                uint32_t id = tables[s0 + s][b];
                if (g_dedupe) {
                    if (slot_of[id] < 0) { slot_of[id] = (int32_t)g; touched[g] = id; g++; }
                    tab[(size_t)s * maxb + b] = (uint32_t)slot_of[id];
                } else {
                    tab[(size_t)s * maxb + b] = g++; /* legacy: one slot per reference */
                }
            }
            tab[(size_t)ns * maxb + s] = eff[s0 + s];
        }
        uint64_t staged_bytes = (uint64_t)(g ? g : 1) * ly->layer_stride_bytes;
        if (staged_bytes > 0xffffffffull) { free(tab); rc = OMEGA_GPU_ATTN_TOO_LARGE; break; }
        void *host_tmp = NULL;
        uint8_t *staged = stage_target((size_t)staged_bytes, &host_tmp);
        if (!staged) { free(tab); rc = OMEGA_GPU_ATTN_CHIP_FAIL; break; }
        /* pass 2: copy each staged slot's (physical block, layer) slice exactly once */
        if (g_dedupe) {
            for (uint32_t u = 0; u < g; u++) {
                const uint8_t *src = pool + (size_t)touched[u] * ly->block_stride_bytes + (size_t)layer_idx * ly->layer_stride_bytes;
                memcpy(staged + (size_t)u * ly->layer_stride_bytes, src, (size_t)ly->layer_stride_bytes);
                slot_of[touched[u]] = -1; /* reset for the next launch */
            }
        } else {
            uint32_t w = 0;
            for (uint32_t s = 0; s < ns; s++) {
                uint32_t need = eff[s0 + s] ? (eff[s0 + s] + ly->block_size - 1) / ly->block_size : 0;
                for (uint32_t b = 0; b < need; b++, w++) {
                    const uint8_t *src = pool + (size_t)tables[s0 + s][b] * ly->block_stride_bytes + (size_t)layer_idx * ly->layer_stride_bytes;
                    memcpy(staged + (size_t)w * ly->layer_stride_bytes, src, (size_t)ly->layer_stride_bytes);
                }
            }
        }
        if (info) {
            info->kv_blocks_logical += total; info->kv_blocks_unique += g;
            info->kv_bytes_staged += (uint64_t)g * ly->layer_stride_bytes;
            info->kv_bytes_naive += (uint64_t)total * ly->layer_stride_bytes;
        }
        Launch L = {
            .q = q + (size_t)s0 * row, .q_bytes = (size_t)ns * row * 4, .pool = staged, .pool_bytes = (size_t)staged_bytes,
            .tab = tab, .tab_bytes = ((size_t)ns * maxb + ns) * 4, .out = out + (size_t)s0 * row, .out_bytes = (size_t)ns * row * 4,
            .grid_x = num_q_heads, .grid_y = ns,
            .params = { maxb, f2u(score_scale()), (uint32_t)ly->layer_stride_bytes, (uint32_t)ly->token_stride_bytes,
                        (uint32_t)ly->kv_plane_stride_bytes, (uint32_t)ly->head_stride_bytes, num_q_heads, ns * maxb * 4 }
        };
        rc = launch(ks, &L, info);
        free(host_tmp); free(tab);
    }
    UNLOCK();
    free(eff); free(slot_of); free(touched);
    if (info) info->call_ns = now_ns() - t0;
    return rc;
}

int omega_gpu_paged_attention_bf16(const float *q, const void *pool, const OmegaGpuKvLayout *layout,
                                   const uint32_t *block_ids, uint32_t num_block_ids,
                                   uint32_t context_len, uint32_t layer_idx,
                                   uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                   float *out, OmegaGpuAttnInfo *info) {
    if (!q || !pool || !layout || !out || (num_block_ids && !block_ids)) return OMEGA_GPU_ATTN_BAD_ARGS;
    const uint32_t *tables[1] = { block_ids };
    uint32_t nb[1] = { num_block_ids }, cx[1] = { context_len };
    return paged_core(q, pool, layout, tables, nb, cx, 1, layer_idx, num_q_heads, num_kv_heads, head_dim, out, info);
}

int omega_gpu_paged_attention_batch_bf16(const float *q, const void *pool, const OmegaGpuKvLayout *layout,
                                         const int32_t *block_tables, const int32_t *context_lens,
                                         uint32_t max_blocks_per_seq, uint32_t num_seqs, uint32_t layer_idx,
                                         uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim,
                                         float *out, OmegaGpuAttnInfo *info) {
    if (!q || !pool || !layout || !out || !block_tables || !context_lens || num_seqs == 0) return OMEGA_GPU_ATTN_BAD_ARGS;
    /* the oracle's rules: ctx <= 0 -> 0; a table row past the array -> empty; negatives removed */
    uint32_t **tables = calloc(num_seqs, sizeof *tables);
    uint32_t *nb = calloc(num_seqs, sizeof *nb), *cx = calloc(num_seqs, sizeof *cx);
    if (!tables || !nb || !cx) { free(tables); free(nb); free(cx); return OMEGA_GPU_ATTN_CHIP_FAIL; }
    int rc = OMEGA_GPU_ATTN_OK;
    for (uint32_t s = 0; s < num_seqs; s++) {
        cx[s] = context_lens[s] > 0 ? (uint32_t)context_lens[s] : 0;
        tables[s] = calloc(max_blocks_per_seq ? max_blocks_per_seq : 1, sizeof **tables);
        if (!tables[s]) { rc = OMEGA_GPU_ATTN_CHIP_FAIL; break; }
        for (uint32_t b = 0; b < max_blocks_per_seq; b++) {
            int32_t v = block_tables[(size_t)s * max_blocks_per_seq + b];
            if (v >= 0) tables[s][nb[s]++] = (uint32_t)v;
        }
    }
    if (rc == OMEGA_GPU_ATTN_OK)
        rc = paged_core(q, pool, layout, (const uint32_t *const *)tables, nb, cx, num_seqs, layer_idx, num_q_heads, num_kv_heads, head_dim, out, info);
    for (uint32_t s = 0; s < num_seqs; s++) free(tables[s]);
    free(tables); free(nb); free(cx);
    return rc;
}

static const char *const k_mutant_names[OMEGA_GPU_ATTN_MUTANT_COUNT] = {
    "none", "NO_MAX", "KV_HEAD", "SLOT", "NO_RESCALE", "Q_ROW", "OUT_ROW"
};
_Static_assert(sizeof k_mutant_names / sizeof k_mutant_names[0] == OMEGA_GPU_ATTN_MUTANT_COUNT, "mutant name table");

const char *omega_gpu_attention_mutant_name(OmegaGpuAttnMutant m) {
    return (m >= 0 && m < OMEGA_GPU_ATTN_MUTANT_COUNT && k_mutant_names[m]) ? k_mutant_names[m] : "?";
}

const char *omega_gpu_attention_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_ATTN_OK: return "OK";
    case OMEGA_GPU_ATTN_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_ATTN_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_ATTN_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_ATTN_CHIP_FAIL: return "CHIP_FAIL";
    case OMEGA_GPU_ATTN_UNWRITTEN: return "UNWRITTEN";
    default: return "UNKNOWN";
    }
}
