/* omega_gpu_elementwise_api.c: FB-1 cut 4 (2026-10-04). See the header.
 *
 * Layout of this file: (1) kernel codegen on top of the BlackwellIR, with a
 * small scheduler that writes every control word; (2) one launcher cloned from
 * the chip-proven tensor matmul path (omega_blackwell_execute_matmul_tensor);
 * (3) the public entry points, which split work into <= OMEGA_GPU_EW_MAX_CTAS
 * launches and run the host-side bookkeeping.
 *
 * Scheduling rules (chip-learned, src/omega_numeric.c omega_numeric_check_patch):
 *   - a variable-latency result (LDG, LDS, LDC, LDCU, S2R, SHFL, MUFU) arrives through
 *     a scoreboard; the producer stalls >= 2 and the consumer waits on that SB;
 *   - a fixed-latency result (FADD FMUL FFMA IMAD IADD3 MOV LOP3 SHF) has no
 *     scoreboard: the producer stalls >= 5 before its consumer (we use 6);
 *   - a predicate producer (ISETP) stalls 14 before @P0 use (NAK SM120 padding);
 *   - stores (STG, STS) set a read barrier so the next instruction cannot overwrite
 *     the source register early.
 * The emitter below serialises: every instruction waits on everything still
 * pending. A final local pass tightens supported adjacent ALU pairs to 5 cycles.
 */
#include "omega_gpu_elementwise_api.h"
#include "omega_blackwell_codegen.h"
#include "omega_bw_reconv.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_submit.h"
#include "omega_gpu_session.h"
#include "sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ codegen */

#define CW(stall, wbar, rbar, wait) \
    (((uint32_t)(stall) << 9) | ((uint32_t)(wbar) << 14) | ((uint32_t)(rbar) << 17) | ((uint32_t)(wait) << 20))
#define CW_YIELD 0x2000u
#define SB_NONE 7u
#define SB_GPR 0u  /* variable-latency GPR results */
#define SB_RD 1u   /* store source-register read barrier */
#define SB_UGPR 2u /* uniform-register results (LDCU) */

enum { K_FIXED, K_PRED, K_VAR, K_UVAR, K_STORE, K_SYNC, K_EXIT, K_BRA, K_BSSY, K_BSYNC };

#define EW_MAX_LOOPS 4
typedef struct {
    BlackwellIRProgram *p;
    uint32_t pending; /* scoreboards set and not yet waited on */
    int err;
    int loop_start[EW_MAX_LOOPS], loop_end[EW_MAX_LOOPS];
    int n_loops;
    int open_start[EW_MAX_LOOPS]; /* starts of loops begun and not yet ended (nesting stack) */
    int depth;
    BwRegions rg;     /* structured reconvergence regions (omega #308) */
    int rg_mutant;    /* test only: 1 = drop the BSYNC at join, 2 = corrupt the BSSY target (host sim must catch both) */
    int exited;       /* a predicated EXIT was emitted: some lanes of a warp may be gone */
    int exit_sync_ok; /* test only: lets a kernel put SHFL/BAR after a predicated EXIT (the negative control R6) */
    int test_path;    /* building for omega_gpu_elementwise_codegen_ir (host tests), never for a launch */
} Em;

static void emit(Em *e, BlackwellIRInsn in, int kind) {
    uint32_t wait = e->pending;
    switch (kind) {
    case K_FIXED: in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_PRED:  in.control = CW(14, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_VAR:   in.control = CW(4, SB_GPR, SB_NONE, wait) | CW_YIELD; e->pending = 1u << SB_GPR; break;
    case K_UVAR:  in.control = CW(4, SB_UGPR, SB_NONE, wait) | CW_YIELD; e->pending = 1u << SB_UGPR; break;
    case K_STORE: in.control = CW(4, SB_NONE, SB_RD, wait) | CW_YIELD; e->pending = 1u << SB_RD; break;
    case K_SYNC:  in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    /* a predicated EXIT or BRA that falls through must not be the only waiter */
    case K_EXIT:  in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    case K_BRA:   in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    /* BSSY / BSYNC read no registers; the stall counts (1 and 5, both with yield) are the ones
     * nvcc 13.0.88 writes for sm_121 (docs/gpu-reconvergence-308.md). They wait on everything
     * pending so that, as for every branch here, nothing is in flight across a control edge
     * and the straight-line scoreboard walk in the tests stays valid. */
    case K_BSSY:  in.control = CW(1, SB_NONE, SB_NONE, wait) | CW_YIELD; e->pending = 0; break;
    case K_BSYNC: in.control = CW(5, SB_NONE, SB_NONE, wait) | CW_YIELD; e->pending = 0; break;
    default: e->err = -1; return;
    }
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1; /* append returns the index */
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
/* dst = a * b + c (registers) */
static void imadr(Em *e, int dst, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = dst; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void iadd3(Em *e, int dst, int a, int b) { BlackwellIRInsn i = I0(BW_IR_IADD3); i.dst_vreg = dst; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void lop3xor(Em *e, int dst, int a, int b) { BlackwellIRInsn i = I0(BW_IR_LOP3_XOR); i.dst_vreg = dst; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
/* dst64 = idx * 4 + ptr64 (byte address of element idx) */
static void addr4(Em *e, int dst64, int idx, int ptr64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = dst64; i.src1_vreg = idx; i.imm = 4; i.src3_vreg = ptr64; emit(e, i, K_FIXED); }
static void ldg(Em *e, int dst, int addr64, int udesc) { BlackwellIRInsn i = I0(BW_IR_LDG_E); i.dst_vreg = dst; i.src1_vreg = addr64; i.ureg = udesc; emit(e, i, K_VAR); }
/* Independent loads (LDG_E from global, or LDS32 from shared memory with ureg NOREG) issued
 * back to back (campaign C2a, 2026-10-05). The first load
 * waits on whatever is pending; the others wait on nothing, so all n are in flight together.
 * Every load sets SB_GPR (a counting scoreboard), so the next instruction waits once for all
 * of them instead of once per load. Fails the build if a load address is a destination of the
 * group or a destination repeats (the later loads would read an operand still in flight). */
static void ld_group(Em *e, BlackwellIROpcode op, int n, const int *dst, const int *addr, int ureg) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (addr[i] == dst[k] || (k != i && dst[i] == dst[k])) { e->err = -1; return; }
    for (int i = 0; i < n; i++) {
        if (op != BW_IR_LDG_E && op != BW_IR_LDS32) { e->err = -1; return; }
        BlackwellIRInsn in = I0(op); in.dst_vreg = dst[i]; in.src1_vreg = addr[i]; in.ureg = ureg;
        if (i == 0) { emit(e, in, K_VAR); continue; }
        in.control = CW(4, SB_GPR, SB_NONE, 0) | CW_YIELD; /* nonzero, so the encoder keeps it */
        if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
    }
}
static void stg(Em *e, int addr64, int val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.src2_vreg = val; i.ureg = udesc; emit(e, i, K_STORE); }
static void fadd(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FADD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void fsub(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FSUB); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void fmul(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FMUL); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void ffma(Em *e, int d, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_FFMA); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void mufu(Em *e, BlackwellIROpcode op, int d, int a) { BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = a; emit(e, i, K_VAR); }
/* Cross-lane and warp-ending ops are refused inside an open reconvergence region (omega #308):
 * a SHFL or BAR.SYNC reached by a split warp reads inactive lanes / counts a partial warp
 * (undefined; the 2026-10-04 attention bug), and a lane that EXITs inside a region can no
 * longer be joined. The build fails closed instead of emitting such code. */
/* Cross-lane instructions (SHFL, BAR.SYNC) need the whole warp present. The emitter refuses
 * them (fails the build) inside an open reconvergence region, where lanes may be on different
 * paths, and after a predicated EXIT, where lanes may be gone for good: a SHFL then reads
 * inactive lanes (undefined on the chip, the attention failure of 2026-10-04). A kernel that
 * must drop lanes before a cross-lane step clamps them instead and exits them afterwards. */
static int warp_split(Em *e) { if (e->rg.depth > 0 || (e->exited && !e->exit_sync_ok)) { e->err = -1; return 1; } return 0; }
static void shfl_down(Em *e, int d, int a, uint32_t off) { if (warp_split(e)) return; BlackwellIRInsn i = I0(BW_IR_SHFL_DOWN); i.dst_vreg = d; i.src1_vreg = a; i.imm = off; emit(e, i, K_VAR); }
static void sts32(Em *e, int addr, int val) { BlackwellIRInsn i = I0(BW_IR_STS32); i.src1_vreg = addr; i.src2_vreg = val; emit(e, i, K_STORE); }
static void lds32(Em *e, int d, int addr) { BlackwellIRInsn i = I0(BW_IR_LDS32); i.dst_vreg = d; i.src1_vreg = addr; emit(e, i, K_VAR); }
static void bar_sync(Em *e) { if (warp_split(e)) return; emit(e, I0(BW_IR_BAR_SYNC), K_SYNC); }
static void isetp_ge_u32(Em *e, int a, int b) { BlackwellIRInsn i = I0(BW_IR_ISETP_GE_U32); i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_PRED); }
/* An EXIT inside a region would leave the barrier waiting for a lane that never comes: refused. */
static void exit_if_p0(Em *e) { if (e->rg.depth > 0) { e->err = -1; return; } e->exited = 1; BlackwellIRInsn i = I0(BW_IR_EXIT); i.predicate_p0 = true; emit(e, i, K_EXIT); }
static void bra_back_if_not_p0(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.predicate_not = true; i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
static void tail(Em *e) {
    emit(e, I0(BW_IR_EXIT), K_EXIT);
    BlackwellIRInsn self = I0(BW_IR_BRA); /* the self-branch every kernel here ends with */
    self.control = 0x000fc000u;
    if (omega_bw_ir_append(e->p, &self) < 0) e->err = -1;
}
/* Loops may nest (prime race cut): begin pushes the start, end pops it, so the inner
 * loop is closed (and its liveness fixed) first. For the single-level loops above this
 * is exactly the old behaviour. */
static void loop_begin(Em *e) { if (e->depth >= EW_MAX_LOOPS) { e->err = -1; return; } e->open_start[e->depth++] = here(e); }
/* Closing a loop: every value defined before the loop and read inside it stays
 * live until the back-branch, otherwise the linear-scan allocator (which knows
 * nothing about back-edges) would hand its register to a body temporary whose
 * first definition comes after the value's last textual use (e.g. the scale y
 * in rmsnorm pass 2), and the next iteration would read garbage. */
static void loop_end(Em *e) {
    if (e->n_loops >= EW_MAX_LOOPS || e->depth <= 0) { e->err = -1; return; }
    int start = e->open_start[--e->depth], end = here(e) - 1;
    e->loop_start[e->n_loops] = start;
    e->loop_end[e->n_loops] = end;
    e->n_loops++;
    OmegaRegAlloc *ra = &e->p->regalloc;
    for (int v = 0; v < ra->num_vregs; v++) {
        OmegaLiveInterval *iv = &ra->intervals[v];
        if (iv->first_def >= 0 && iv->first_def < start && iv->last_use >= start && iv->last_use < end) iv->last_use = end;
    }
}

/* Kernel argument words (constant bank 0, same slots as the matmul path):
 * 0x380 a, 0x388 b, 0x390 c (64-bit each), then four 32-bit parameters. */
#define ARG_A 0x380u
#define ARG_B 0x388u
#define ARG_C 0x390u
#define ARG_P0 0x398u
#define ARG_P1 0x39cu
#define ARG_P2 0x3a0u
#define ARG_P3 0x3a4u
#define ARG_P4 0x3a8u
#define DESC_OFF 0x358u

typedef struct { int udesc, pa, pb, pc, tid, cta; } Pro;
static Pro prologue(Em *e) {
    Pro r;
    r.udesc = UV64(e); r.pa = V64(e); r.pb = V64(e); r.pc = V64(e); r.tid = V(e); r.cta = V(e);
    ldcu64(e, r.udesc, DESC_OFF);
    ldc64(e, r.pa, ARG_A); ldc64(e, r.pb, ARG_B); ldc64(e, r.pc, ARG_C);
    s2r(e, r.tid, BW_SR_TID_X); s2r(e, r.cta, BW_SR_CTAID_X);
    return r;
}

/* out[i] = 2^a[i] for i < n (n = P0). Threads 128, grid ceil(n/128). Mutant: 2^(a+1). */
static void gen_ex2(Em *e, int mutant) {
    Pro p = prologue(e);
    int n = V(e), i = V(e), off = V64(e), x = V(e), y = V(e);
    ldc32(e, n, ARG_P0);
    imadi(e, i, p.cta, OMEGA_GPU_EW_THREADS, p.tid);
    isetp_ge_u32(e, i, n);
    exit_if_p0(e);
    addr4(e, off, i, p.pa);
    ldg(e, x, off, p.udesc);
    if (mutant) { int one = V(e); movf(e, one, 1.0f); fadd(e, x, x, one); }
    mufu(e, BW_IR_MUFU_EX2, y, x);
    addr4(e, off, i, p.pc);
    stg(e, off, y, p.udesc);
    tail(e);
}

/* out[i] = a[i ^ 127] within each CTA of 128 (n % 128 == 0). STS own word, BAR.SYNC,
 * LDS the partner word. Mutant: LDS own word (no exchange). */
static void gen_xchg(Em *e, int mutant) {
    Pro p = prologue(e);
    int i = V(e), off = V64(e), x = V(e), saddr = V(e), paddr = V(e), mask = V(e), y = V(e);
    imadi(e, i, p.cta, OMEGA_GPU_EW_THREADS, p.tid);
    addr4(e, off, i, p.pa);
    ldg(e, x, off, p.udesc);
    imadi(e, saddr, p.tid, 4, NOREG);
    sts32(e, saddr, x);
    bar_sync(e);
    if (mutant) {
        lds32(e, y, saddr);
    } else {
        movi(e, mask, OMEGA_GPU_EW_THREADS - 1);
        lop3xor(e, paddr, p.tid, mask);
        imadi(e, paddr, paddr, 4, NOREG);
        lds32(e, y, paddr);
    }
    addr4(e, off, i, p.pc);
    stg(e, off, y, p.udesc);
    tail(e);
}

/* One CTA of 128 threads per row. P0 = dim/128 (loop count), P1 = bits(1/dim),
 * P2 = bits(eps), P3 = dim. Mutant: the warp tree drops the last SHFL step. */
static void gen_rmsnorm(Em *e, int mutant) {
    Pro p = prologue(e);
    int nk = V(e), invd = V(e), eps = V(e), dim = V(e);
    int rowbase = V(e), idx = V(e), col = V(e), k = V(e), acc = V(e), cT = V(e), c1 = V(e);
    int off = V64(e), x = V(e), t = V(e);
    ldc32(e, nk, ARG_P0); ldc32(e, invd, ARG_P1); ldc32(e, eps, ARG_P2); ldc32(e, dim, ARG_P3);
    imadr(e, rowbase, p.cta, dim, NOREG);
    iadd3(e, idx, rowbase, p.tid);
    movrz(e, acc); movrz(e, k);
    movi(e, cT, OMEGA_GPU_EW_THREADS); movi(e, c1, 1);
    loop_begin(e);
    int l1 = here(e);
    addr4(e, off, idx, p.pa);
    ldg(e, x, off, p.udesc);
    ffma(e, acc, x, x, acc);
    iadd3(e, idx, idx, cT);
    iadd3(e, k, k, c1);
    isetp_ge_u32(e, k, nk);
    bra_back_if_not_p0(e, l1);
    loop_end(e);
    static const uint32_t steps[5] = { 16, 8, 4, 2, 1 };
    for (int s = 0; s < (mutant ? 4 : 5); s++) { shfl_down(e, t, acc, steps[s]); fadd(e, acc, acc, t); }
    int saddr = V(e);
    imadi(e, saddr, p.tid, 4, NOREG);
    sts32(e, saddr, acc);
    bar_sync(e);
    int a0 = V(e), a1 = V(e), a2 = V(e), a3 = V(e), s0 = V(e), s1 = V(e), s2 = V(e), s3 = V(e), sum = V(e);
    movrz(e, a0); movi(e, a1, 128); movi(e, a2, 256); movi(e, a3, 384);
    lds32(e, s0, a0); lds32(e, s1, a1); lds32(e, s2, a2); lds32(e, s3, a3);
    fadd(e, sum, s0, s1); fadd(e, sum, sum, s2); fadd(e, sum, sum, s3);
    int v = V(e), y = V(e), u = V(e), cmh = V(e), c15 = V(e);
    fmul(e, v, sum, invd);
    fadd(e, v, v, eps);
    mufu(e, BW_IR_MUFU_RSQ, y, v);
    movf(e, cmh, -0.5f); movf(e, c15, 1.5f);
    fmul(e, u, v, y); fmul(e, u, u, y); ffma(e, u, u, cmh, c15); fmul(e, y, y, u);
    int offw = V64(e), offc = V64(e), w = V(e), o = V(e);
    iadd3(e, idx, rowbase, p.tid);
    imadi(e, col, p.tid, 1, NOREG); /* col = tid (the encoder maps a missing src2 to R0, not RZ, so no IADD3 here) */
    movrz(e, k);
    loop_begin(e);
    int l2 = here(e);
    addr4(e, off, idx, p.pa);
    ldg(e, x, off, p.udesc);
    addr4(e, offw, col, p.pb);
    ldg(e, w, offw, p.udesc);
    fmul(e, o, x, y);
    fmul(e, o, o, w);
    addr4(e, offc, idx, p.pc);
    stg(e, offc, o, p.udesc);
    iadd3(e, idx, idx, cT);
    iadd3(e, col, col, cT);
    iadd3(e, k, k, c1);
    isetp_ge_u32(e, k, nk);
    bra_back_if_not_p0(e, l2);
    loop_end(e);
    tail(e);
}

/* One CTA of head_dim/2 threads per head. P0 = head_dim, P1 = head_dim/2.
 * a = vector, b = table (cos in the first half of each head, sin in the second).
 * Mutant: out0 = q0*cos + q1*sin. */
static void gen_rope(Em *e, int mutant) {
    Pro p = prologue(e);
    int hd = V(e), half = V(e), i0 = V(e), i1 = V(e), off0 = V64(e), off1 = V64(e);
    int q0 = V(e), q1 = V(e), c = V(e), s = V(e), t0 = V(e), t1 = V(e), o0 = V(e), o1 = V(e);
    ldc32(e, hd, ARG_P0); ldc32(e, half, ARG_P1);
    imadr(e, i0, p.cta, hd, p.tid);
    iadd3(e, i1, i0, half);
    addr4(e, off0, i0, p.pa); ldg(e, q0, off0, p.udesc);
    addr4(e, off1, i1, p.pa); ldg(e, q1, off1, p.udesc);
    addr4(e, off0, i0, p.pb); ldg(e, c, off0, p.udesc);
    addr4(e, off1, i1, p.pb); ldg(e, s, off1, p.udesc);
    fmul(e, t0, q0, c); fmul(e, t1, q1, s);
    if (mutant) fadd(e, o0, t0, t1); else fsub(e, o0, t0, t1);
    fmul(e, t0, q0, s); fmul(e, t1, q1, c);
    fadd(e, o1, t0, t1);
    addr4(e, off0, i0, p.pc); stg(e, off0, o0, p.udesc);
    addr4(e, off1, i1, p.pc); stg(e, off1, o1, p.udesc);
    tail(e);
}

/* out[i] = g/(1+2^(g*-log2e)) * up for i < n (P0). Mutant: 2^(g*+log2e). */
static void gen_swiglu(Em *e, int mutant) {
    Pro p = prologue(e);
    int n = V(e), i = V(e), off = V64(e), g = V(e), u = V(e), k = V(e), t = V(e), ex = V(e), d = V(e), r = V(e), one = V(e), o = V(e);
    ldc32(e, n, ARG_P0);
    imadi(e, i, p.cta, OMEGA_GPU_EW_THREADS, p.tid);
    isetp_ge_u32(e, i, n);
    exit_if_p0(e);
    addr4(e, off, i, p.pa); ldg(e, g, off, p.udesc);
    addr4(e, off, i, p.pb); ldg(e, u, off, p.udesc);
    movf(e, k, mutant ? 1.4426950408889634f : -1.4426950408889634f);
    movf(e, one, 1.0f);
    fmul(e, t, g, k);
    mufu(e, BW_IR_MUFU_EX2, ex, t);
    fadd(e, d, ex, one);
    mufu(e, BW_IR_MUFU_RCP, r, d);
    fmul(e, o, g, r);
    fmul(e, o, o, u);
    addr4(e, off, i, p.pc);
    stg(e, off, o, p.udesc);
    tail(e);
}

/* ------------------------------------------- prime race sieve (2026-10-05) */
static void lop3lut(Em *e, int d, int a, int b, int c, uint32_t lut) { BlackwellIRInsn i = I0(BW_IR_LOP3_LUT); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; i.imm = lut & 0xffu; emit(e, i, K_FIXED); }
static void shl_r(Em *e, int d, int a, int n) { BlackwellIRInsn i = I0(BW_IR_SHF_L_U32); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = n; emit(e, i, K_FIXED); }
static void shr_i(Em *e, int d, int a, uint32_t n) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = d; i.src1_vreg = a; i.imm = n; emit(e, i, K_FIXED); }
static void imadhi(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_IMAD_HI_U32); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
/* dst64 = a * imm + c64 (IMAD.WIDE.U32 immediate form) */
static void widei(Em *e, int dst64, int a, uint32_t imm, int c64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = dst64; i.src1_vreg = a; i.imm = imm; i.src3_vreg = c64; emit(e, i, K_FIXED); }
/* @P0 BRA forward; returns the instruction index for bra_patch_here */
static int bra_fwd_if_p0(Em *e) { int at = here(e); BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.imm = 1; emit(e, i, K_BRA); return at; }
static void bra_patch_here(Em *e, int at) { if (e->err == 0 && at >= 0 && at < here(e)) e->p->insns[at].imm = (uint32_t)(here(e) - at); else e->err = -1; }
static void bra_always(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }

/* ---- structured reconvergence regions (omega #308; bookkeeping in omega_bw_reconv.h) ----
 * region_begin: BSSY Bd (d = nesting depth), fails closed when no barrier register is left.
 * region_exit:  a forward branch out of the innermost region, patched to its BSYNC at join
 *               (RX_ALWAYS, RX_IF_P0, RX_IF_NOT_P0).
 * region_join:  BSYNC Bd; patches the BSSY to the instruction after it and every exit to it.
 * Usage: divergent loop  = region_begin; L: isetp; region_exit(RX_IF_P0); body; bra_always(L); region_join.
 *        if/else diamond = region_begin; isetp; t = bra_fwd_if_p0; then; region_exit(RX_ALWAYS);
 *                          bra_patch_here(t); else; region_join. */
enum { RX_ALWAYS, RX_IF_P0, RX_IF_NOT_P0 };
static int region_begin(Em *e) {
    int bar = bw_regions_next_bar(&e->rg);
    if (bar < 0) { e->err = -1; return -1; }
    int at = here(e);
    BlackwellIRInsn i = I0(BW_IR_BSSY); i.bar_reg = (uint8_t)bar; i.imm = 2; emit(e, i, K_BSSY);
    if (bw_regions_begin(&e->rg, at) < 0) e->err = -1;
    return bar;
}
static void region_exit(Em *e, int cond) {
    int at = here(e);
    BlackwellIRInsn i = I0(BW_IR_BRA); i.imm = 1;
    if (cond != RX_ALWAYS) { i.predicate_p0 = true; i.predicate_not = (cond == RX_IF_NOT_P0); }
    emit(e, i, K_BRA);
    if (bw_regions_exit(&e->rg, at) < 0) e->err = -1;
}
static void region_join(Em *e) {
    if (e->rg.depth <= 0) { e->err = -1; return; }
    int d = e->rg.depth - 1, bsync = here(e);
    BlackwellIRInsn i = I0(e->rg_mutant == 1 ? BW_IR_NOP : BW_IR_BSYNC); i.bar_reg = (uint8_t)d; emit(e, i, K_BSYNC);
    if (e->rg_mutant == 1) {
        /* negative control (host sim only): the join is a NOP; patch as the real join would */
        BwRegions *r = &e->rg; r->depth--;
        e->p->insns[r->bssy_at[d]].imm = (uint32_t)(bsync + 1 - r->bssy_at[d]);
        for (int k = 0; k < r->n_exits[d]; k++) e->p->insns[r->exits[d][k]].imm = (uint32_t)(bsync - r->exits[d][k]);
        return;
    }
    if (bw_regions_join(&e->rg, e->p, bsync) != d) e->err = -1;
    if (e->rg_mutant == 2) e->p->insns[e->rg.bssy_at[d]].imm += 1; /* negative control: BSSY names the wrong join */
}

/* Word-parallel odd-only sieve (docs: handoff 2026-10-05 DESIGN "GPU kernel (native)").
 * Thread t owns 32-bit output words w = t, t + S, t + 2S, ... (S = P1 = threads launched;
 * the I42 envelope caps a launch at OMEGA_GPU_EW_MAX_CTAS CTAs, so a grid-stride loop
 * replaces one thread per word). Every word has exactly one writer; no atomics.
 * Bit j of word w is the odd number 2(32w + j) + 1.
 * Arguments: a (0x380) = prime table, entries {p, magic = ceil(2^32 / p), k0 = (p*p - 1)/2}
 * (12 bytes, ascending p, odd primes <= isqrt(limit)) followed by ONE sentinel entry with
 * k0 = 0xffffffff that ends every scan; c (0x390) = output words;
 * P0 = nwords, P1 = S, P2 = last word index, P3 = ~tailmask (bits past the last odd).
 * Range: limit < 2^32 (so base, k0 < 2^31 and the sign-bit tricks below hold).
 * Per word: base = 32w; for each prime while k0 < base + 32:
 *   d = base - k0; neg = d >> 31 (k0 > base); dpos = neg ? 0 : d; jadd = neg ? k0 - base : 0
 *   q = hi32(dpos * magic)  (= floor(dpos/p) or one more: magic is rounded up)
 *   r = dpos - q*p                            (q one too big -> r = (dpos mod p) - p, wrapped)
 *   j = p - r; j -= p * ((r - 1) >> 31)       (r == 0 -> 0; wrapped r -> p - (dpos mod p))
 * One fix covers both cases: with q one too big, dpos mod p >= p - dpos*e/2^32 > p/2 > 0
 * (e = magic*p - 2^32 < p, dpos < 2^31), so the true remainder is never 0 there and
 * p - r - p = p - (dpos mod p) is the right offset. A separate "r += p if wrapped" step was
 * removed after a negative control showed it changes no output (host sim, limit 1e7, where
 * the overshoot occurs 268 times; bench/prime_race/tests magic_model).
 *   j += jadd; while j < 32 { acc |= 1 << j; j += p }
 * then acc |= (w == 0) (the number 1), word = ~(acc | (w == last ? ~tailmask : 0)).
 * Branch-free except the loop branches. Mutant: drops the j fix (j = p - r unreduced), so words where
 * base - k0 is a multiple of p (or q overshot) mark the wrong bits.
 *
 * staged = 1 (OMEGA_GPU_EW_PRIME_SIEVE_SHARED, campaign C3): P4 = table words, sentinel included
 * (3 per entry). The CTA first copies the table into shared memory at offset 0 (thread t copies
 * words t, t + 128, ...), BAR_SYNC, then the prime loop reads shared memory (LDS) instead of
 * uncached global memory. Every thread reaches the barrier: the w >= nwords exit comes after
 * it. The launch must declare at least 4 * P4 bytes of shared memory.
 *
 * divergent (omega #308, regression R1): 0 = the warp-uniform mark loop above (campaign C5).
 * 1 = the original lane-dependent mark loop (omega 888a011: `while j < 32 { acc |= 1 << j; j += p }`)
 * wrapped in a reconvergence region. 2 = as 1, and the prime loop (whose break `k0 >= base + 32`
 * is also lane-dependent: every lane sieves a different word) wrapped in an outer region, so the
 * mark loop's region nests inside it (B0 outer, B1 inner). */
static void gen_prime_sieve(Em *e, int mutant, int staged, int divergent) {
    Pro pr = prologue(e);
    int nwords = V(e), stride = V(e), lastw = V(e), tailinv = V(e);
    int one = V(e), c32 = V(e), cm1 = V(e), w = V(e);
    ldc32(e, nwords, ARG_P0); ldc32(e, stride, ARG_P1); ldc32(e, lastw, ARG_P2); ldc32(e, tailinv, ARG_P3);
    movi(e, one, 1); movi(e, c32, 32); movi(e, cm1, 0xffffffffu);
    int c4 = -1, c8 = -1, c12 = -1;
    if (staged) {
        int twords = V(e), i = V(e), ga = V64(e), val = V(e), sa = V(e);
        c4 = V(e); c8 = V(e); c12 = V(e);
        ldc32(e, twords, ARG_P4);
        movi(e, c4, 4); movi(e, c8, 8); movi(e, c12, 12);
        imadi(e, i, pr.tid, 1, NOREG);
        /* The staging loop's trip count differs per thread (thread t copies words t, t+128, ...),
         * so its exit splits the warp. The divergent variants rejoin the warp before BAR.SYNC
         * (a barrier is not a reconvergence point: the fragments would stay split for the rest
         * of the kernel). The production kernel keeps the 2026-10-05 C3 form unchanged. */
        if (divergent) region_begin(e);
        loop_begin(e); /* staging loop */
        int ls = here(e);
        isetp_ge_u32(e, i, twords);
        int sdone = -1;
        if (divergent) region_exit(e, RX_IF_P0); else sdone = bra_fwd_if_p0(e);
        addr4(e, ga, i, pr.pa);
        ldg(e, val, ga, pr.udesc);
        imadi(e, sa, i, 4, NOREG);
        sts32(e, sa, val);
        imadi(e, i, one, OMEGA_GPU_EW_THREADS, i);
        bra_always(e, ls);
        loop_end(e);
        if (divergent) region_join(e); else bra_patch_here(e, sdone);
        bar_sync(e);
    }
    imadi(e, w, pr.cta, OMEGA_GPU_EW_THREADS, pr.tid);
    isetp_ge_u32(e, w, nwords);
    exit_if_p0(e);

    /* ptr, a64, a64m: global addresses, or (staged) 32-bit shared offsets */
    int base = V(e), basep32 = V(e), acc = V(e);
    int ptr = staged ? V(e) : V64(e), a64 = staged ? V(e) : V64(e), a64m = staged ? V(e) : V64(e);
    int p = V(e), magic = V(e), k0 = V(e), negp = V(e), d = V(e), neg = V(e), nd = V(e), dpos = V(e), jadd = V(e);
    int q = V(e), r = V(e), j = V(e), t = V(e), z = V(e), bit = V(e);
    int e1 = V(e), m = V(e), word = V(e), oaddr = V64(e);
    int tt = V(e), jj = V(e), sh = V(e), c31 = V(e), cm32 = V(e);
    movi(e, c31, 31); movi(e, cm32, 0xffffffe0u);

    loop_begin(e); /* word loop */
    int lw = here(e);
    imadi(e, base, w, 32, NOREG);
    iadd3(e, basep32, base, c32);
    movrz(e, acc);
    if (staged) movrz(e, ptr); else widei(e, ptr, one, 0, pr.pa);

    if (divergent >= 2) region_begin(e); /* outer region: the prime loop's break is lane-dependent */
    loop_begin(e); /* prime loop */
    int lp = here(e);
    /* one memory round trip per prime: all three entry words load together (C2a). The
     * sentinel entry is a whole entry, so loading p and magic before the k0 test is in bounds. */
    if (staged) { iadd3(e, a64, ptr, c8); iadd3(e, a64m, ptr, c4); }
    else { widei(e, a64, one, 8, ptr); widei(e, a64m, one, 4, ptr); }
    {
        const int dsts[3] = { k0, p, magic }, adrs[3] = { a64, ptr, a64m };
        if (staged) ld_group(e, BW_IR_LDS32, 3, dsts, adrs, NOREG);
        else ld_group(e, BW_IR_LDG_E, 3, dsts, adrs, pr.udesc);
    }
    isetp_ge_u32(e, k0, basep32);
    int brk = -1;
    if (divergent >= 2) region_exit(e, RX_IF_P0); else brk = bra_fwd_if_p0(e);
    imadi(e, negp, p, 0xffffffffu, NOREG);       /* -p */
    imadi(e, d, k0, 0xffffffffu, base);          /* base - k0 */
    shr_i(e, neg, d, 31);
    imadi(e, nd, d, 0xffffffffu, NOREG);         /* -d */
    imadr(e, dpos, neg, nd, d);                  /* d - neg*d */
    imadr(e, jadd, neg, nd, NOREG);              /* neg * (k0 - base) */
    imadhi(e, q, dpos, magic);
    imadr(e, r, q, negp, dpos);                  /* dpos - q*p */
    imadi(e, j, r, 0xffffffffu, p);              /* p - r */
    if (!mutant) {
        iadd3(e, t, r, cm1);
        shr_i(e, z, t, 31);                      /* r == 0, or r wrapped (q one too big) */
        imadr(e, j, z, negp, j);                 /* j - p */
    }
    iadd3(e, j, j, jadd);

    /* mark loop (campaign C5): the trip count depends only on p, so every active lane runs the
     * same ceil(32/p) steps and the warp never splits here (all active lanes are on the same
     * prime). Bits past this word are masked out instead of branched around. The old loop ran
     * "while j < 32", a lane-dependent branch with no reconvergence point. */
    if (divergent) {
        /* R1: the 888a011 mark loop, lane-dependent trip count, inside a region. Each lane
         * leaves at its own iteration (the exit jumps to the BSYNC) and the warp reunites there. */
        region_begin(e);
        loop_begin(e);
        int li = here(e);
        isetp_ge_u32(e, j, c32);
        region_exit(e, RX_IF_P0);
        shl_r(e, bit, one, j);
        lop3lut(e, acc, acc, bit, NOREG, 0xfc);  /* acc | bit */
        iadd3(e, j, j, p);
        bra_always(e, li);
        loop_end(e);
        region_join(e);
    } else {
        movrz(e, tt);
        loop_begin(e);
        int li = here(e);
        isetp_ge_u32(e, tt, c32);
        int done = bra_fwd_if_p0(e);
        iadd3(e, jj, j, tt);
        lop3lut(e, sh, jj, c31, NOREG, 0xc0);    /* jj & 31 */
        shl_r(e, bit, one, sh);
        iadd3(e, t, jj, cm32);
        shr_i(e, z, t, 31);                      /* jj < 32 (j <= p + 31, far below 2^31) */
        imadi(e, m, z, 0xffffffffu, NOREG);      /* all ones when jj < 32, else 0 */
        lop3lut(e, acc, acc, bit, m, 0xf8);      /* acc | (bit & m) */
        iadd3(e, tt, tt, p);
        bra_always(e, li);
        loop_end(e);
        bra_patch_here(e, done);
    }

    if (staged) iadd3(e, ptr, ptr, c12); else widei(e, ptr, one, 12, ptr);
    bra_always(e, lp);
    loop_end(e);
    if (divergent >= 2) region_join(e); else bra_patch_here(e, brk);

    iadd3(e, t, w, cm1);
    shr_i(e, z, t, 31);                          /* w == 0: the number 1 is not prime */
    lop3lut(e, acc, acc, z, NOREG, 0xfc);
    lop3xor(e, e1, w, lastw);
    iadd3(e, t, e1, cm1);
    shr_i(e, z, t, 31);                          /* w == last word */
    imadr(e, m, z, tailinv, NOREG);
    lop3lut(e, word, acc, m, NOREG, 0x03);       /* ~(acc | m) */
    addr4(e, oaddr, w, pr.pc);
    stg(e, oaddr, word, pr.udesc);
    /* IMAD, not IADD3: the IADD3 encoder ignores the scheduler control word (fixed
     * 0x010fca00), so right after a store it would not wait on the read barrier */
    imadr(e, w, stride, one, w);
    isetp_ge_u32(e, w, nwords);
    bra_back_if_not_p0(e, lw);
    loop_end(e);
    tail(e);
}

/* ---- reconvergence probes (omega #308, regressions R3..R6 and the resource bound) ----
 * All: a, b = u32 inputs, out = u32, P0 = n; 128 threads, grid ceil(n/128); launched by
 * omega_gpu_reconv_probe_u32. A lane past n clamps its index to n-1 so that it loads valid
 * data and takes part in the warp-wide SHFL; it exits only after the SHFL, before its store.
 * Each probe computes a per-lane r on divergent paths, joins, then
 *     out[i] = r + 4099 * s,  s = r of lane (lane+1) in the same warp (SHFL.DOWN 1; lane 31: own r)
 * so a lane that joined late, never, or with the wrong value shows in the neighbour's word too.
 * The host oracle is rc_oracle in tests/gpu_reconv_test.c. */
typedef struct { Pro pr; int n, i, idx, x, y, r, s, one, cm1; } Rp;
static Rp rp_begin(Em *e) {
    Rp k; memset(&k, 0, sizeof k);
    k.pr = prologue(e);
    k.n = V(e); k.i = V(e); k.idx = V(e); k.x = V(e); k.y = V(e); k.r = V(e); k.s = V(e); k.one = V(e); k.cm1 = V(e);
    int nm1 = V(e), d = V(e), neg = V(e), off = V64(e);
    ldc32(e, k.n, ARG_P0);
    movi(e, k.one, 1); movi(e, k.cm1, 0xffffffffu);
    imadi(e, k.i, k.pr.cta, OMEGA_GPU_EW_THREADS, k.pr.tid);
    iadd3(e, nm1, k.n, k.cm1);                   /* n - 1 */
    imadi(e, d, k.i, 0xffffffffu, nm1);          /* (n - 1) - i, top bit set when i > n - 1 */
    shr_i(e, neg, d, 31);
    imadr(e, k.idx, neg, d, k.i);                /* i, or n - 1 when past the end */
    addr4(e, off, k.idx, k.pr.pa); ldg(e, k.x, off, k.pr.udesc);
    addr4(e, off, k.idx, k.pr.pb); ldg(e, k.y, off, k.pr.udesc);
    return k;
}
static void rp_end(Em *e, const Rp *k) {
    int out = V(e), off = V64(e);
    shfl_down(e, k->s, k->r, 1);
    imadi(e, out, k->s, 4099, k->r);
    isetp_ge_u32(e, k->i, k->n);
    exit_if_p0(e);
    addr4(e, off, k->i, k->pr.pc);
    stg(e, off, out, k->pr.udesc);
    tail(e);
}
/* R3: if/else with sides of different length. x >= y: r = f(x, y) (5 steps); else r = x + 1.
 * Math mutant: the long side is one too big. */
static void gen_rc_diamond(Em *e, int mutant) {
    Rp k = rp_begin(e);
    int t = V(e), u = V(e);
    region_begin(e);
    isetp_ge_u32(e, k.x, k.y);
    int els = bra_fwd_if_p0(e);
    iadd3(e, k.r, k.x, k.one);                   /* short side */
    region_exit(e, RX_ALWAYS);
    bra_patch_here(e, els);
    imadi(e, t, k.x, 3, k.y);                    /* long side: t = 3x + y */
    lop3xor(e, u, t, k.x);                       /* u = t ^ x */
    imadi(e, k.r, u, 5, k.one);                  /* r = 5u + 1 */
    shr_i(e, t, k.r, 3);
    iadd3(e, k.r, k.r, t);                       /* r += r >> 3 */
    if (mutant) iadd3(e, k.r, k.r, k.one);
    region_join(e);
    rp_end(e, &k);
}
/* The lane-dependent loop shared by R4 and R5: v = x & 255, step = (y & 15) + 1,
 * while v < 256 { v += step; cnt += 1 }. Math mutant: the bound is 255. Emits the region. */
static void rc_count_loop(Em *e, const Rp *k, int mutant, int v, int cnt) {
    int st = V(e), bound = V(e), c255 = V(e), c15 = V(e);
    movi(e, c255, 255); movi(e, c15, 15); movi(e, bound, mutant ? 255 : 256);
    lop3lut(e, v, k->x, c255, NOREG, 0xc0);
    lop3lut(e, st, k->y, c15, NOREG, 0xc0);
    iadd3(e, st, st, k->one);
    movrz(e, cnt);
    region_begin(e);
    loop_begin(e);
    int L = here(e);
    isetp_ge_u32(e, v, bound);
    region_exit(e, RX_IF_P0);
    iadd3(e, v, v, st);
    iadd3(e, cnt, cnt, k->one);
    bra_always(e, L);
    loop_end(e);
    region_join(e);
}
/* R4: lane-dependent loop exit (break). r = 256 * cnt + v. */
static void gen_rc_loopbreak(Em *e, int mutant) {
    Rp k = rp_begin(e);
    int v = V(e), cnt = V(e);
    rc_count_loop(e, &k, mutant, v, cnt);
    imadi(e, k.r, cnt, 256, v);
    rp_end(e, &k);
}
/* R5: nested divergence. Odd x: r = 7 (skips the inner region). Even x: the R4 loop inside the
 * outer region, r = 256 * cnt + v + 100. */
static void gen_rc_nested(Em *e, int mutant) {
    Rp k = rp_begin(e);
    int v = V(e), cnt = V(e), low = V(e), c100 = V(e);
    movi(e, k.r, 7); movi(e, c100, 100);
    lop3lut(e, low, k.x, k.one, NOREG, 0xc0);    /* x & 1 */
    region_begin(e);
    isetp_ge_u32(e, low, k.one);
    region_exit(e, RX_IF_P0);
    rc_count_loop(e, &k, mutant, v, cnt);
    imadi(e, k.r, cnt, 256, v);
    iadd3(e, k.r, k.r, c100);
    region_join(e);
    rp_end(e, &k);
}
/* R6: lanes past the end, a divergent region, then a CTA-wide exchange and the warp SHFL.
 * xv = x | 1 computed on divergent paths (even lanes take the branch), every thread stores xv
 * to its shared slot, BAR.SYNC, reads its partner's (tid ^ 1), r = 3x + partner_xv.
 * Production form: lanes past n are clamped and exit after the SHFL (correct). early_exit
 * (mutant 1): lanes past n EXIT first, as a naive kernel would; the emitter refuses that
 * (BAR/SHFL after a predicated EXIT). Only the host-test build path (codegen_ir) may bypass the
 * refusal, so the simulator can show what the kernel would do (SHFL_EXITED); a launch never can. */
static void gen_rc_exit_bar(Em *e, int early_exit) {
    Rp k = rp_begin(e);
    int saddr = V(e), paddr = V(e), yv = V(e), xv = V(e), low = V(e);
    if (early_exit) {
        e->exit_sync_ok = early_exit == 1 && e->test_path;
        isetp_ge_u32(e, k.i, k.n);
        exit_if_p0(e);
    }
    imadi(e, xv, k.x, 1, NOREG);
    lop3lut(e, low, k.x, k.one, NOREG, 0xc0);    /* x & 1 */
    region_begin(e);
    isetp_ge_u32(e, low, k.one);
    region_exit(e, RX_IF_P0);                    /* odd lanes leave */
    iadd3(e, xv, xv, k.one);                     /* even lanes: xv = x + 1 */
    region_join(e);
    imadi(e, saddr, k.pr.tid, 4, NOREG);
    sts32(e, saddr, xv);
    bar_sync(e);
    lop3xor(e, paddr, k.pr.tid, k.one);
    imadi(e, paddr, paddr, 4, NOREG);
    lds32(e, yv, paddr);
    imadi(e, k.r, k.x, 3, yv);
    rp_end(e, &k);
}
/* Barrier-register bound: `levels` nested regions, each leaving lanes with x >= y out and adding 1
 * for the others: r = x + levels * (x < y). 16 levels must build (B0..B15); 17 must be refused. */
static void gen_rc_depth(Em *e, int levels) {
    Rp k = rp_begin(e);
    imadi(e, k.r, k.x, 1, NOREG);
    for (int d = 0; d < levels; d++) {
        region_begin(e);
        isetp_ge_u32(e, k.x, k.y);
        region_exit(e, RX_IF_P0);
        iadd3(e, k.r, k.r, k.one);
    }
    for (int d = 0; d < levels; d++) region_join(e);
    rp_end(e, &k);
}
/* Codegen refusals (each must come back CODEGEN_FAIL; a kernel with a split warp at a
 * cross-lane step must never reach the chip). */
static void gen_rc_refuse(Em *e, OmegaGpuEwOp which) {
    Rp k = rp_begin(e);
    switch (which) {
    case OMEGA_GPU_EW_RC_REFUSE_EXIT_IN_REGION: region_begin(e); isetp_ge_u32(e, k.i, k.n); exit_if_p0(e); region_join(e); break;
    case OMEGA_GPU_EW_RC_REFUSE_BAR_IN_REGION: region_begin(e); bar_sync(e); region_join(e); break;
    case OMEGA_GPU_EW_RC_REFUSE_SHFL_IN_REGION: region_begin(e); shfl_down(e, k.s, k.x, 1); region_join(e); break;
    case OMEGA_GPU_EW_RC_REFUSE_UNCLOSED: region_begin(e); isetp_ge_u32(e, k.x, k.y); region_exit(e, RX_IF_P0); break; /* no join */
    case OMEGA_GPU_EW_RC_REFUSE_JOIN_ONLY: region_join(e); break; /* BSYNC without BSSY */
    default: e->err = -1; break;
    }
    imadi(e, k.r, k.x, 1, NOREG);
    rp_end(e, &k);
}

/* A value first defined inside a loop body may not be read after the loop
 * before it is written again: the allocator could have let a later body
 * temporary share its register, and the last iteration's value would be lost. */
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

#define EW_GPR_BUDGET 64u /* the QMD declares 64 registers, as the matmul launcher does */

/* Mesa NAK sm120_instr_latencies.rs + sm100/reg_raw.csv: ALU/dual-ALU pairs
 * and FMA/FMA pairs need 4 + 1 cycles; mixed pairs need 5 + 1. CSV rows are
 * READERS, columns WRITERS (see lat_rs_gen.py), not the reverse. IMAD.WIDE
 * has operand-specific latencies and is deliberately excluded here.
 *
 * Retain at least 5 cycles after every fixed producer. Thus the next reader
 * uses the pair-specific bound and later readers are >= 9 cycles away (the
 * next instruction in a tightened pair is itself fixed). No instruction,
 * branch target, wait, barrier assignment or register allocation is changed.
 * IADD3's encoder still supplies its historical control word; do not pretend
 * to schedule that producer here. Its existing 5-cycle delay is unchanged.
 *
 * Predicate-to-guard uses NAK paw_latency -> raw(write, None): reader
 * RedirectedFp64, writer Dualalu = 13, plus SM120's 1 = 14, in emit above.
 * NVIDIA's public ISA docs are silent about SASS control-word semantics.
 */
static int fixed_pair_class(BlackwellIROpcode op) {
    switch (op) {
    case BW_IR_MOV_RZ: case BW_IR_MOV_IMM: case BW_IR_IADD3:
    case BW_IR_LOP3_XOR: case BW_IR_LOP3_LUT:
    case BW_IR_SHF_R: case BW_IR_SHF_L_U32: return 1;
    case BW_IR_IMAD: case BW_IR_IMAD_HI_U32:
    case BW_IR_FADD: case BW_IR_FSUB: case BW_IR_FMUL: case BW_IR_FFMA: return 2;
    default: return 0;
    }
}

static void schedule_fixed_pairs(BlackwellIRProgram *p) {
    for (size_t i = 0; i + 1 < p->count; i++) {
        BlackwellIRInsn *a = &p->insns[i];
        const BlackwellIRInsn *b = &p->insns[i + 1];
        int cls = fixed_pair_class(a->op);
        if (!cls || cls != fixed_pair_class(b->op) || a->op == BW_IR_IADD3 ||
            a->predicate_p0 || b->predicate_p0 || ((a->control >> 9) & 15u) != 6)
            continue;
        a->control = (a->control & ~(15u << 9)) | (5u << 9);
    }
}

static int build_kernel(OmegaGpuEwOp op, int mutant, OmegaBlackwellKernel *kernel, BlackwellIRProgram *prog_out, int test_path) {
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    if (!prog) return OMEGA_GPU_EW_CODEGEN_FAIL;
    omega_bw_ir_init(prog);
    Em e; memset(&e, 0, sizeof e); e.p = prog; e.test_path = test_path;
    /* mutant 1 = the op's math mutant (as before); 2 = the join (BSYNC) dropped; 3 = the BSSY
     * names the wrong join. 2 and 3 only change kernels that open a region. */
    e.rg_mutant = mutant == 2 ? 1 : mutant == 3 ? 2 : 0;
    int math = mutant == 1;
    switch (op) {
    case OMEGA_GPU_EW_EX2: gen_ex2(&e, math); break;
    case OMEGA_GPU_EW_XCHG: gen_xchg(&e, math); break;
    case OMEGA_GPU_EW_RMSNORM: gen_rmsnorm(&e, math); break;
    case OMEGA_GPU_EW_ROPE: gen_rope(&e, math); break;
    case OMEGA_GPU_EW_SWIGLU: gen_swiglu(&e, math); break;
    case OMEGA_GPU_EW_PRIME_SIEVE: gen_prime_sieve(&e, math, 0, 0); break;
    case OMEGA_GPU_EW_PRIME_SIEVE_SHARED: gen_prime_sieve(&e, math, 1, 0); break;
    case OMEGA_GPU_EW_PRIME_SIEVE_DIV1: gen_prime_sieve(&e, math, 0, 1); break;
    case OMEGA_GPU_EW_PRIME_SIEVE_SHARED_DIV1: gen_prime_sieve(&e, math, 1, 1); break;
    case OMEGA_GPU_EW_PRIME_SIEVE_DIV2: gen_prime_sieve(&e, math, 0, 2); break;
    case OMEGA_GPU_EW_PRIME_SIEVE_SHARED_DIV2: gen_prime_sieve(&e, math, 1, 2); break;
    case OMEGA_GPU_EW_RC_DIAMOND: gen_rc_diamond(&e, math); break;
    case OMEGA_GPU_EW_RC_LOOPBREAK: gen_rc_loopbreak(&e, math); break;
    case OMEGA_GPU_EW_RC_NESTED: gen_rc_nested(&e, math); break;
    case OMEGA_GPU_EW_RC_EXIT_BAR: gen_rc_exit_bar(&e, math); break;
    case OMEGA_GPU_EW_RC_REFUSE_EXIT_SHFL: gen_rc_exit_bar(&e, 2); break;
    case OMEGA_GPU_EW_RC_DEPTH16: gen_rc_depth(&e, BW_RECONV_MAX_BAR); break;
    case OMEGA_GPU_EW_RC_REFUSE_DEPTH17: gen_rc_depth(&e, BW_RECONV_MAX_BAR + 1); break;
    case OMEGA_GPU_EW_RC_REFUSE_EXIT_IN_REGION: case OMEGA_GPU_EW_RC_REFUSE_BAR_IN_REGION:
    case OMEGA_GPU_EW_RC_REFUSE_SHFL_IN_REGION: case OMEGA_GPU_EW_RC_REFUSE_UNCLOSED:
    case OMEGA_GPU_EW_RC_REFUSE_JOIN_ONLY: gen_rc_refuse(&e, op); break;
    default: e.err = -1; break;
    }
    if (e.rg.depth != 0) e.err = -1; /* a region left open has no join: refuse the build */
    schedule_fixed_pairs(prog);
    int rc = OMEGA_GPU_EW_CODEGEN_FAIL;
    int ra_rc = (e.err == 0) ? omega_bw_regalloc_solve(prog) : -1;
    int loop_rc = (ra_rc == 0) ? check_loop_invariant(&e) : -1;
#ifdef OMEGA_EW_DEBUG
    fprintf(stderr, "ew codegen op=%d mutant=%d emit_err=%d insns=%zu vregs=%d regalloc=%d loops=%d peak_gpr=%u\n",
            (int)op, mutant, e.err, prog->count, prog->regalloc.num_vregs, ra_rc, loop_rc, prog->regalloc.peak_gpr_usage);
#endif
    if (e.err == 0 && ra_rc == 0 && loop_rc == 0 && prog->regalloc.peak_gpr_usage <= EW_GPR_BUDGET) {
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
            if (prog_out) *prog_out = *prog;
            rc = OMEGA_GPU_EW_OK;
        } else {
#ifdef OMEGA_EW_DEBUG
            fprintf(stderr, "ew codegen op=%d: encode failed (an instruction form the encoder refuses)\n", (int)op);
#endif
            free(kernel->code); kernel->code = NULL;
        }
    }
    free(prog);
    return rc;
}

/* --------------------------------------------------------- kernel cache */

/* One kernel per (op, mutant); the device copy is uploaded once and stays
 * resident for the process (cut 4b: no per-call code page). */
typedef struct { int used; int op; int mutant; OmegaBlackwellKernel kernel; NvrmMem code; } Slot;
static int g_mutant_op; /* 0 = none */
static int g_mutant_level = 1; /* the mutant number built for g_mutant_op (1 math, 2 join dropped, 3 join corrupted) */
static Slot g_cache[8];
static OmegaGpuScratch g_a, g_b, g_c; /* staging, grown to the high-water mark, reused per call */
static bool g_hooked;
#define LOCK() omega_gpu_session_lock()
#define UNLOCK() omega_gpu_session_unlock()

static void cache_clear_locked(void) {
    for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++)
        if (g_cache[i].used) { free(g_cache[i].kernel.code); omega_gpu_session_free(&g_cache[i].code); memset(&g_cache[i], 0, sizeof g_cache[i]); }
}

void omega_gpu_elementwise_cache_clear(void) {
    LOCK();
    cache_clear_locked();
    UNLOCK();
}

/* Called by the session (lock held, device still open) right before it closes. */
static void on_session_close(void) {
    cache_clear_locked();
    omega_gpu_session_scratch_free(&g_a); omega_gpu_session_scratch_free(&g_b); omega_gpu_session_scratch_free(&g_c);
}

void omega_gpu_elementwise_test_set_mutant(int op) {
    LOCK();
    g_mutant_op = op; g_mutant_level = 1;
    cache_clear_locked();
    UNLOCK();
}
void omega_gpu_elementwise_test_set_mutant_level(int op, int level) {
    LOCK();
    g_mutant_op = op; g_mutant_level = level < 1 ? 1 : level > 3 ? 3 : level;
    cache_clear_locked();
    UNLOCK();
}

/* Caller holds the lock. Builds the kernel if needed; uploads the device copy when a
 * device is open (upload = false lets the host-only codegen path run without a chip). */
static int kernel_for(OmegaGpuEwOp op, bool upload, Slot **out, bool *hit) {
    int mutant = (g_mutant_op == (int)op) ? g_mutant_level : 0;
    Slot *s = NULL;
    for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++)
        if (g_cache[i].used && g_cache[i].op == (int)op && g_cache[i].mutant == mutant) { s = &g_cache[i]; *hit = true; break; }
    if (!s) {
        OmegaBlackwellKernel k;
        int rc = build_kernel(op, mutant, &k, NULL, 0);
        if (rc != OMEGA_GPU_EW_OK) return rc;
        size_t slot = 0;
        for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++) if (!g_cache[i].used) { slot = i; break; }
        s = &g_cache[slot];
        if (s->used) { free(s->kernel.code); omega_gpu_session_free(&s->code); } /* full: evict slot 0 */
        memset(s, 0, sizeof *s);
        s->used = 1; s->op = (int)op; s->mutant = mutant; s->kernel = k;
        *hit = false;
    }
    if (upload && !s->code.cpu) {
        if (omega_gpu_session_alloc_code(s->kernel.code_size, &s->code) != 0) return OMEGA_GPU_EW_CHIP_FAIL;
        memcpy(s->code.cpu, s->kernel.code, s->kernel.code_size);
        __asm__ volatile("dsb sy" ::: "memory");
    }
    *out = s;
    return OMEGA_GPU_EW_OK;
}

int omega_gpu_elementwise_codegen(OmegaGpuEwOp op, const uint32_t *dims, float eps, OmegaBlackwellKernel *kernel) {
    (void)dims; (void)eps; /* kernels read their shape from the argument words */
    if (!kernel) return OMEGA_GPU_EW_BAD_ARGS;
    LOCK();
    int mutant = (g_mutant_op == (int)op) ? g_mutant_level : 0;
    UNLOCK();
    return build_kernel(op, mutant, kernel, NULL, 0);
}

int omega_gpu_elementwise_codegen_ir(OmegaGpuEwOp op, int mutant, void *prog, OmegaBlackwellKernel *kernel) {
    if (!kernel) return OMEGA_GPU_EW_BAD_ARGS;
    if (mutant < 0 || mutant > 3) return OMEGA_GPU_EW_BAD_ARGS;
    return build_kernel(op, mutant, kernel, (BlackwellIRProgram *)prog, 1);
}

const char *omega_gpu_elementwise_last_error(void) { return omega_gpu_session_last_error(); }

/* ------------------------------------------------------------ launcher */

#define EW_POISON 0xffbadbadu /* a NaN pattern no kernel here produces on purpose */
#define EW_WAIT_MS 600000ull /* like main's load/store launchers; a 5 s wait failed under contention (omega#198) */

static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

/* Caller holds the lock. Opens the shared device on first use. */
static int dev_open_locked(void) {
    if (!g_hooked) { (void)omega_gpu_session_on_close(on_session_close); g_hooked = true; }
    return omega_gpu_session_open() ? OMEGA_GPU_EW_OK : OMEGA_GPU_EW_CHIP_FAIL;
}

/* One launch: a and b staged in, c out (c_bytes/4 words, poison-filled first), through
 * the shared session (omega_gpu_session_launch: shader-cache invalidate, constant bank,
 * QMD0/QMD1, first marker, L2_FLUSH_DIRTY, second marker), then readback and the
 * poison scan. b may be NULL when b_fill wrote the staging buffer already. */
static int run_abc(const Slot *ks, const void *a, size_t a_bytes, const void *b, size_t b_bytes,
                   void *c, size_t c_bytes, uint32_t threads_x, uint32_t grid_x,
                   const uint32_t params[4], OmegaGpuEwInfo *info) {
    if (omega_gpu_session_scratch(&g_a, a_bytes) || omega_gpu_session_scratch(&g_b, b_bytes) || omega_gpu_session_scratch(&g_c, c_bytes))
        return OMEGA_GPU_EW_CHIP_FAIL;
    memcpy(g_a.mem.cpu, a, a_bytes);
    if (b) memcpy(g_b.mem.cpu, b, b_bytes);
    size_t c_words = c_bytes / 4;
    uint32_t *c_dev = (uint32_t *)g_c.mem.cpu;
    for (size_t i = 0; i < c_words; i++) c_dev[i] = EW_POISON;
    __asm__ volatile("dsb sy" ::: "memory");

    uint32_t args[OMEGA_BW_CBANK_MATMUL_ARGS_WORDS];
    args[0] = (uint32_t)g_a.mem.va; args[1] = (uint32_t)(g_a.mem.va >> 32);
    args[2] = (uint32_t)g_b.mem.va; args[3] = (uint32_t)(g_b.mem.va >> 32);
    args[4] = (uint32_t)g_c.mem.va; args[5] = (uint32_t)(g_c.mem.va >> 32);
    args[6] = params[0]; args[7] = params[1]; args[8] = params[2]; args[9] = params[3];
    OmegaGpuLaunch L = {
        .code_va = ks->code.va, .gpr_count = EW_GPR_BUDGET, .threads_x = threads_x, .threads_y = 1,
        .grid_x = grid_x, .grid_y = 1, .num_elements = c_words,
        .args = args, .n_args = OMEGA_BW_CBANK_MATMUL_ARGS_WORDS, .timeout_ms = EW_WAIT_MS,
    };
    uint64_t elapsed = 0; uint32_t marker = 0;
    if (omega_gpu_session_launch(&L, &elapsed, &marker) != 0) return OMEGA_GPU_EW_CHIP_FAIL;

    memcpy(c, g_c.mem.cpu, c_bytes);
    uint32_t unwritten = 0;
    for (size_t i = 0; i < c_words; i++) if (((uint32_t *)c)[i] == EW_POISON) unwritten++;
    if (info) {
        info->chip_calls++;
        info->elapsed_ns += elapsed;
        info->completion_marker = marker;
        info->unwritten_words += unwritten;
        info->threads_per_cta = threads_x;
        info->ctas_last_launch = grid_x;
    }
    return unwritten ? OMEGA_GPU_EW_UNWRITTEN : OMEGA_GPU_EW_OK;
}

/* ------------------------------------------------------- public entries */

static void info_begin(OmegaGpuEwInfo *info, const OmegaBlackwellKernel *k, bool hit) {
    if (!info) return;
    memset(info, 0, sizeof *info);
    info->kernel_cache_hit = hit;
    info->gpr_count = k->gpr_count;
    info->insn_count = (uint32_t)k->insn_count;
    snprintf(info->target_chip, sizeof info->target_chip, "NVIDIA DGX Spark (Grace Blackwell GB10)");
    info->sm_architecture = 121;
}
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static void info_end(OmegaGpuEwInfo *info, uint64_t t0) { if (info) info->call_ns = now_ns() - t0; }

/* Common prologue of every chip entry: lock, open, kernel (resident), info. */
static int begin(OmegaGpuEwOp op, Slot **ks, OmegaGpuEwInfo *info) {
    LOCK();
    int rc = dev_open_locked();
    bool hit = false;
    if (rc == OMEGA_GPU_EW_OK) rc = kernel_for(op, true, ks, &hit);
    if (rc != OMEGA_GPU_EW_OK) { UNLOCK(); return rc; }
    info_begin(info, &(*ks)->kernel, hit);
    return OMEGA_GPU_EW_OK;
}

int omega_gpu_rmsnorm_f32(uint32_t rows, uint32_t dim, const float *x, const float *weight,
                          float eps, float *out, OmegaGpuEwInfo *info) {
    uint64_t t0 = now_ns();
    if (!x || !weight || !out || rows == 0 || dim == 0 || dim % OMEGA_GPU_EW_THREADS != 0) return OMEGA_GPU_EW_BAD_ARGS;
    if (dim > OMEGA_GPU_EW_MAX_DIM) return OMEGA_GPU_EW_TOO_LARGE;
    Slot *k = NULL;
    int rc = begin(OMEGA_GPU_EW_RMSNORM, &k, info);
    if (rc != OMEGA_GPU_EW_OK) return rc;
    uint32_t params[4] = { dim / OMEGA_GPU_EW_THREADS, f2u(1.0f / (float)dim), f2u(eps), dim };
    for (uint32_t r0 = 0; r0 < rows && rc == OMEGA_GPU_EW_OK; r0 += OMEGA_GPU_EW_MAX_CTAS) {
        uint32_t nr = rows - r0 < OMEGA_GPU_EW_MAX_CTAS ? rows - r0 : OMEGA_GPU_EW_MAX_CTAS;
        size_t bytes = (size_t)nr * dim * 4;
        rc = run_abc(k, x + (size_t)r0 * dim, bytes, weight, (size_t)dim * 4, out + (size_t)r0 * dim, bytes,
                     OMEGA_GPU_EW_THREADS, nr, params, info);
    }
    UNLOCK();
    info_end(info, t0);
    return rc;
}

int omega_gpu_rope_f32(uint32_t heads, uint32_t head_dim, const float *v,
                       const float *cos_half, const float *sin_half, float *out, OmegaGpuEwInfo *info) {
    uint64_t t0 = now_ns();
    if (!v || !cos_half || !sin_half || !out || heads == 0 || head_dim < 2 || (head_dim & 1)) return OMEGA_GPU_EW_BAD_ARGS;
    if (head_dim > 2048) return OMEGA_GPU_EW_TOO_LARGE;
    Slot *k = NULL;
    int rc = begin(OMEGA_GPU_EW_ROPE, &k, info);
    if (rc != OMEGA_GPU_EW_OK) return rc;
    uint32_t half = head_dim / 2;
    uint32_t params[4] = { head_dim, half, 0, 0 };
    for (uint32_t h0 = 0; h0 < heads && rc == OMEGA_GPU_EW_OK; h0 += OMEGA_GPU_EW_MAX_CTAS) {
        uint32_t nh = heads - h0 < OMEGA_GPU_EW_MAX_CTAS ? heads - h0 : OMEGA_GPU_EW_MAX_CTAS;
        size_t bytes = (size_t)nh * head_dim * 4;
        /* table: cos in the first half of each head, sin in the second, written straight into
         * the staging buffer (one copy per head of this launch) */
        if (omega_gpu_session_scratch(&g_b, bytes) != 0) { rc = OMEGA_GPU_EW_CHIP_FAIL; break; }
        float *table = g_b.mem.cpu;
        for (uint32_t h = 0; h < nh; h++) {
            memcpy(table + (size_t)h * head_dim, cos_half, half * sizeof *table);
            memcpy(table + (size_t)h * head_dim + half, sin_half, half * sizeof *table);
        }
        /* out may alias v: v is copied into the staging buffer before the result lands in out */
        rc = run_abc(k, v + (size_t)h0 * head_dim, bytes, NULL, bytes, out + (size_t)h0 * head_dim, bytes, half, nh, params, info);
    }
    UNLOCK();
    info_end(info, t0);
    return rc;
}

static int run_1d(OmegaGpuEwOp op, uint32_t n, const void *a, const void *b, void *c, OmegaGpuEwInfo *info) {
    uint64_t t0 = now_ns();
    Slot *k = NULL;
    int rc = begin(op, &k, info);
    if (rc != OMEGA_GPU_EW_OK) return rc;
    const uint32_t chunk = OMEGA_GPU_EW_MAX_CTAS * OMEGA_GPU_EW_THREADS;
    for (uint32_t i0 = 0; i0 < n && rc == OMEGA_GPU_EW_OK; i0 += chunk) {
        uint32_t ni = n - i0 < chunk ? n - i0 : chunk;
        uint32_t grid = (ni + OMEGA_GPU_EW_THREADS - 1) / OMEGA_GPU_EW_THREADS;
        uint32_t params[4] = { ni, 0, 0, 0 };
        size_t bytes = (size_t)ni * 4;
        rc = run_abc(k, (const uint8_t *)a + (size_t)i0 * 4, bytes, b ? (const uint8_t *)b + (size_t)i0 * 4 : a, bytes,
                     (uint8_t *)c + (size_t)i0 * 4, bytes, OMEGA_GPU_EW_THREADS, grid, params, info);
    }
    UNLOCK();
    info_end(info, t0);
    return rc;
}

int omega_gpu_swiglu_f32(uint32_t n, const float *gate, const float *up, float *out, OmegaGpuEwInfo *info) {
    if (!gate || !up || !out || n == 0) return OMEGA_GPU_EW_BAD_ARGS;
    return run_1d(OMEGA_GPU_EW_SWIGLU, n, gate, up, out, info);
}
int omega_gpu_ex2_f32(uint32_t n, const float *x, float *out, OmegaGpuEwInfo *info) {
    if (!x || !out || n == 0) return OMEGA_GPU_EW_BAD_ARGS;
    return run_1d(OMEGA_GPU_EW_EX2, n, x, NULL, out, info);
}
int omega_gpu_shared_xchg_u32(uint32_t n, const uint32_t *x, uint32_t *out, OmegaGpuEwInfo *info) {
    if (!x || !out || n == 0 || n % OMEGA_GPU_EW_THREADS != 0) return OMEGA_GPU_EW_BAD_ARGS;
    return run_1d(OMEGA_GPU_EW_XCHG, n, x, NULL, out, info);
}
int omega_gpu_reconv_probe_u32(OmegaGpuEwOp op, uint32_t n, const uint32_t *a, const uint32_t *b, uint32_t *out, OmegaGpuEwInfo *info) {
    if (!a || !b || !out || n == 0) return OMEGA_GPU_EW_BAD_ARGS;
    switch (op) {
    case OMEGA_GPU_EW_RC_DIAMOND: case OMEGA_GPU_EW_RC_LOOPBREAK: case OMEGA_GPU_EW_RC_NESTED:
    case OMEGA_GPU_EW_RC_EXIT_BAR: case OMEGA_GPU_EW_RC_DEPTH16: break;
    default: return OMEGA_GPU_EW_BAD_ARGS;
    }
    return run_1d(op, n, a, b, out, info);
}

const char *omega_gpu_elementwise_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_EW_OK: return "OK";
    case OMEGA_GPU_EW_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_EW_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_EW_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_EW_CHIP_FAIL: return "CHIP_FAIL";
    case OMEGA_GPU_EW_UNWRITTEN: return "UNWRITTEN";
    default: return "UNKNOWN";
    }
}
