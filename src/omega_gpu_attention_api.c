/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
 * One CTA per (query head, sequence), head_dim threads (64 or 128; the text below says 64, the
 * constants scale with KSpec.hd). The context is walked in chunks of head_dim tokens:
 *   phase A  thread t owns token base+t: block lookup, K dot q (f32 FFMA), s = dot * scale
 *            (scale folds 1/sqrt(d) and log2 e), s and the token's byte offset go to shared;
 *   reduce   chunk max (SHFL.DOWN tree per warp, lane partials to shared, BAR.SYNC, 2 LDS),
 *            m_new = max(m, chunk max), alpha = EX2(m - m_new), p = EX2(s - m_new) to shared,
 *            chunk sum the same way, l = l * alpha + chunk sum;
 *   phase B  thread d owns dimension d: acc = acc * alpha + sum over the chunk of p_t * V[t][d].
 * Then out[d] = acc * RCP(l). The first chunk sees m = -inf, so alpha = EX2(-inf) = 0 and
 * the initial l and acc drop out (PTX ISA ex2.approx: ex2(-Inf) = +0).
 *
 * Shared memory (16 * head_dim bytes; hd 64: 1024 bytes, the QMD default): [0,4hd) q, [4hd,8hd) s then p,
 * [8hd,12hd) token byte offsets, [12hd,16hd) warp partials for the reductions (hd 64: 0, 256, 512,
 * 768; hd 128: 0, 512, 1024, 1536; the partials of warp w are read at 12hd + 128w).
 */
#include "omega_gpu_attention_api.h"
#include "omega_blackwell_codegen.h"
#include "omega_bw_reconv.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_submit.h"
#include "omega_gpu_session.h"
#include "omega_gpu_fmt.h"
#include "sha256.h"
#ifdef OMEGA_ATTN_DEBUG
#include <stdio.h> /* the debug trace only; never in the freestanding build */
#endif
#include <string.h>

/* ------------------------------------------------------------------ codegen */

#define CW(stall, wbar, rbar, wait) \
    (((uint32_t)(stall) << 9) | ((uint32_t)(wbar) << 14) | ((uint32_t)(rbar) << 17) | ((uint32_t)(wait) << 20))
#define CW_YIELD 0x2000u
#define SB_NONE 7u
#define SB_GPR 0u
#define SB_RD 1u
#define SB_UGPR 2u
#define SB_LD2 4u       /* second scoreboard for batched loads (the matmul kernel's B-load word, omega #328) */
#define SB_LD3 3u       /* third: phase B keeps shared loads and the two halves of the V loads on separate scoreboards */
#define AT_BATCH_MAX 4  /* loads in flight per scoreboard: 4 complete on the chip, 8 never did (omega #328) */

enum { K_FIXED, K_PRED, K_VAR, K_UVAR, K_STORE, K_SYNC, K_EXIT, K_BRA, K_BSSY, K_BSYNC };

#define AT_MAX_LOOPS 8
typedef struct {
    BlackwellIRProgram *p;
    uint32_t pending;
    uint32_t batch;      /* scoreboards the open load batch is filling (no wait on them inside the batch) */
    int batch_n[8];      /* loads issued per scoreboard in the open batch */
    int err;
    int loop_start[AT_MAX_LOOPS], loop_end[AT_MAX_LOOPS];
    int n_loops, depth, stack[AT_MAX_LOOPS];
    BwRegions rg;  /* structured reconvergence regions (omega #308) */
    int rg_mutant; /* test only: 1 = drop the BSYNC at join (the 2026-10-04 bug shape; host sim must catch it) */
} Em;

static void emit(Em *e, BlackwellIRInsn in, int kind) {
    uint32_t wait = e->pending;
    e->batch = 0; memset(e->batch_n, 0, sizeof e->batch_n); /* an ordinary instruction waits on the whole batch */
    switch (kind) {
    case K_FIXED: in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_PRED:  in.control = CW(13, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_VAR:   in.control = CW(4, SB_GPR, SB_NONE, wait) | CW_YIELD; e->pending = 1u << SB_GPR; break;
    case K_UVAR:  in.control = CW(4, SB_UGPR, SB_NONE, wait) | CW_YIELD; e->pending = 1u << SB_UGPR; break;
    case K_STORE: in.control = CW(4, SB_NONE, SB_RD, wait) | CW_YIELD; e->pending = 1u << SB_RD; break;
    case K_SYNC:  in.control = CW(6, SB_NONE, SB_NONE, wait); e->pending = 0; break;
    case K_EXIT:  in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    case K_BRA:   in.control = CW(5, SB_NONE, SB_NONE, wait); e->pending = in.predicate_p0 ? wait : 0; break;
    /* stall counts as nvcc 13.0.88 writes them for sm_121 (see src/omega_gpu_elementwise_api.c) */
    case K_BSSY:  in.control = CW(1, SB_NONE, SB_NONE, wait) | CW_YIELD; e->pending = 0; break;
    case K_BSYNC: in.control = CW(5, SB_NONE, SB_NONE, wait) | CW_YIELD; e->pending = 0; break;
    default: e->err = -1; return;
    }
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}

static int V(Em *e) { int v = omega_bw_ir_alloc_vreg(e->p); if (v < 0) e->err = -1; return v; }
static int V64(Em *e) { int v = omega_bw_ir_alloc_vreg64(e->p); if (v < 0) e->err = -1; return v; }
static int V128(Em *e) { int v = omega_bw_ir_alloc_vreg128(e->p); if (v < 0) e->err = -1; return v; }
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
/* Batched loads (2026-10-12, long-context finding): every load above waits on the previous one, so
 * phase A paid one global latency per dim pair and phase B one per token (~6 GB/s at 1268 ctx). A
 * batched load sets `sb` and waits on every pending scoreboard except the ones the open batch is
 * filling (and SB_RD, its read barrier), so up to AT_BATCH_MAX loads per scoreboard are in flight.
 * The next ordinary instruction (emit) waits on all of them. SB_RD covers the address register: the
 * allocator may hand that register to a later load's data before this load has read it. */
static void emit_batched(Em *e, BlackwellIRInsn in, uint32_t sb) {
    if (sb >= 8 || e->batch_n[sb] >= AT_BATCH_MAX) { e->err = -1; return; }
    in.control = CW(4, sb, SB_RD, e->pending & ~e->batch) | CW_YIELD;
    e->batch |= (1u << sb) | (1u << SB_RD); e->batch_n[sb]++;
    e->pending |= (1u << sb) | (1u << SB_RD);
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}
/* A fixed-latency instruction inside an open load batch: it waits on the batch's read barrier (its
 * destination may be a load's address register, which the load may read late) but not on the loads'
 * data, so the address of the next load is computed while the earlier loads are in flight. Only for
 * instructions that read no result of the open batch. */
static void emit_fixed_in_batch(Em *e, BlackwellIRInsn in) {
    uint32_t wait = (e->pending & ~e->batch) | (e->batch ? (1u << SB_RD) : 0u);
    in.control = CW(6, SB_NONE, SB_NONE, wait);
    if (omega_bw_ir_append(e->p, &in) < 0) e->err = -1;
}
static void ldg128_b(Em *e, int dst128, int addr64, uint32_t imm, int udesc, uint32_t sb) { BlackwellIRInsn i = I0(BW_IR_LDG_E_128); i.dst_vreg = dst128; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit_batched(e, i, sb); }
static void ldg_b(Em *e, int dst, int addr64, int udesc, uint32_t sb) { BlackwellIRInsn i = I0(BW_IR_LDG_E); i.dst_vreg = dst; i.src1_vreg = addr64; i.ureg = udesc; emit_batched(e, i, sb); }
static void ldg16_b(Em *e, int dst, int addr64, int udesc, uint32_t sb) { BlackwellIRInsn i = I0(BW_IR_LDG_E_U16); i.dst_vreg = dst; i.src1_vreg = addr64; i.ureg = udesc; emit_batched(e, i, sb); }
static void lds128_b(Em *e, int dst128, int addr, uint32_t imm, uint32_t sb) { BlackwellIRInsn i = I0(BW_IR_LDS128); i.dst_vreg = dst128; i.src1_vreg = addr; i.imm = imm; emit_batched(e, i, sb); }
/* sub-register operands of a quad */
static void e_ffma_s(Em *e, int d, int a, int as, int b, int bs, int c) { BlackwellIRInsn i = I0(BW_IR_FFMA); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)as; i.src2_vreg = b; i.src2_subreg = (uint8_t)bs; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void imadi_s(Em *e, int dst, int a, int as, uint32_t imm, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = dst; i.src1_vreg = a; i.src1_subreg = (uint8_t)as; i.imm = imm; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void lop3andi_s(Em *e, int dst, int a, int as, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = dst; i.src1_vreg = a; i.src1_subreg = (uint8_t)as; i.imm = imm; emit(e, i, K_FIXED); }
static void wide_s(Em *e, int dst64, int idx, int is, uint32_t imm, int base64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = dst64; i.src1_vreg = idx; i.src1_subreg = (uint8_t)is; i.imm = imm; i.src3_vreg = base64; emit(e, i, K_FIXED); }
static void wide_s_b(Em *e, int dst64, int idx, int is, uint32_t imm, int base64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = dst64; i.src1_vreg = idx; i.src1_subreg = (uint8_t)is; i.imm = imm; i.src3_vreg = base64; emit_fixed_in_batch(e, i); }
static void stg(Em *e, int addr64, int val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.src2_vreg = val; i.ureg = udesc; emit(e, i, K_STORE); }
static void e_fadd(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FADD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void e_fsub(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FSUB); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void e_fmul(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FMUL); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void e_ffma(Em *e, int d, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_FFMA); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
static void e_fmax(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_FMNMX_MAX); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void mufu(Em *e, BlackwellIROpcode op, int d, int a) { BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = a; emit(e, i, K_VAR); }
/* SHFL and BAR.SYNC are refused inside an open reconvergence region (omega #308): on a split
 * warp they read inactive lanes / count a partial warp, which is what produced wrong dims on
 * the chip on 2026-10-04. The build fails closed instead. */
static int in_region(Em *e) { if (e->rg.depth > 0) { e->err = -1; return 1; } return 0; }
static void shfl_down(Em *e, int d, int a, uint32_t off) { if (in_region(e)) return; BlackwellIRInsn i = I0(BW_IR_SHFL_DOWN); i.dst_vreg = d; i.src1_vreg = a; i.imm = off; emit(e, i, K_VAR); }
static void sts32(Em *e, int addr, int val) { BlackwellIRInsn i = I0(BW_IR_STS32); i.src1_vreg = addr; i.src2_vreg = val; emit(e, i, K_STORE); }
static void lds32(Em *e, int d, int addr) { BlackwellIRInsn i = I0(BW_IR_LDS32); i.dst_vreg = d; i.src1_vreg = addr; emit(e, i, K_VAR); }
static void bar_sync(Em *e) { if (in_region(e)) return; emit(e, I0(BW_IR_BAR_SYNC), K_SYNC); }
static void isetp_ge_u32(Em *e, int a, int b) { BlackwellIRInsn i = I0(BW_IR_ISETP_GE_U32); i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_PRED); }
static void bra_back_if_not_p0(Em *e, int target) { BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.predicate_not = true; i.imm = (uint32_t)(target - here(e)); emit(e, i, K_BRA); }
/* @P0 BRA forward; the target is patched in when known */
static int bra_fwd_if_p0(Em *e) { int at = here(e); BlackwellIRInsn i = I0(BW_IR_BRA); i.predicate_p0 = true; i.imm = 0; emit(e, i, K_BRA); return at; }
static void patch_fwd(Em *e, int at) { if (at >= 0 && (size_t)at < e->p->count) e->p->insns[at].imm = (uint32_t)(here(e) - at); else e->err = -1; }

/* ---- structured reconvergence regions (omega #308); same API as the elementwise emitter ---- */
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
        BwRegions *r = &e->rg; r->depth--;
        e->p->insns[r->bssy_at[d]].imm = (uint32_t)(bsync + 1 - r->bssy_at[d]);
        for (int k = 0; k < r->n_exits[d]; k++) e->p->insns[r->exits[d][k]].imm = (uint32_t)(bsync - r->exits[d][k]);
        return;
    }
    if (bw_regions_join(&e->rg, e->p, bsync) != d) e->err = -1;
}
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

/* Head dimension is a kernel parameter (64 or 128, see KSpec.hd). Shared memory is four arrays of hd words:
 * q at SH_Q, s then p at hd*4, token byte offsets at hd*8, warp partials at hd*12 (hd 64: 0, 256, 512, 768,
 * the layout of every cut before head_dim 128). */
#define SH_Q 0u
#define F32_NEG_INF 0xff800000u

/* divergent (omega #308 regression R2): 0 = production branch-free chunk loop; 1 = the
 * lane-dependent "j >= ctx" branch restored inside a reconvergence region (BSSY/BSYNC) so
 * the SHFL reductions run on a joined warp; 2 = the same with the BSYNC dropped (the
 * 2026-10-04 bug shape, a negative control the host simulator must catch). */
typedef struct { bool f32; uint32_t log2bs, log2gqa; int mutant; int divergent; uint32_t hd; int split; } KSpec;
/* split (cut F, 2026-10-12): -1 = direct output (the gqa f32 and rows callers; byte for byte the kernel before).
 * 0..MAX_LOG2SPLIT = the paged path: 2^split CTAs share one (sequence, head). Each sequence derives its own split
 * count from its own context (so its result never depends on the other sequences in the launch), every CTA writes
 * an unnormalised partial (acc[hd], running max, running sum; an empty slice writes acc 0, max -inf) and the host
 * combines them, also when split == 0. */

/* cut K (2026-10-12, GQA group): at head_dim 128 one CTA serves the GC query heads of one kv head, GC =
 * min(2^log2gqa, 4), so every K and V byte of a launch is read once instead of once per query head. The KV
 * pool is GPU-uncached (host-written), so the four CTAs of a GQA group were four DRAM reads of the same
 * bytes: 21 MB per call at 1268 tokens, the memory ceiling (sc leg smoke-ctx-chunk: 152 us per call).
 * Grid x is the head group (num_q_heads / GC CTAs per sequence and split). Shared memory: q, scores and
 * warp partials per head, token offsets once. GC = 1 (every hd 64 kernel, hd 128 with gqa 1) emits exactly
 * the program before this cut (digest fixtures). */
static uint32_t heads_per_cta(uint32_t hd, uint32_t log2gqa) { return hd == 128u ? (1u << (log2gqa < 2u ? log2gqa : 2u)) : 1u; }
static uint32_t shared_bytes_for(uint32_t hd, uint32_t log2gqa) { return (3u * heads_per_cta(hd, log2gqa) + 1u) * hd * 4u; }
uint32_t omega_gpu_attention_heads_per_cta(uint32_t head_dim, uint32_t log2_gqa) { return heads_per_cta(head_dim, log2_gqa); }

static void gen_attention(Em *e, const KSpec *k) {
    const uint32_t elem = k->f32 ? 4u : 2u;
    const uint32_t HD = k->hd, GC = heads_per_cta(HD, k->log2gqa), log2gc = GC == 4u ? 2u : GC == 2u ? 1u : 0u;
    const uint32_t HB = HD * 4u; /* bytes of one per-head shared array */
    const uint32_t SH_S = GC * HB, SH_O = 2u * GC * HB, SH_R = SH_O + HB; /* warps per CTA: HD / 32 */
    int udesc = UV64(e), A = V64(e), B = V64(e), C = V64(e), D = V64(e);
    int tid = V(e), h = V(e), seq = V(e);
    int p_maxb = V(e), scale = V(e), p_bstride = V(e), p_tstride = V(e), p_kvp = V(e), p_hstride = V(e), p_nqh = V(e), p_ctxoff = V(e);
    ldcu64(e, udesc, DESC_OFF);
    ldc64(e, A, ARG_Q); ldc64(e, B, ARG_POOL); ldc64(e, C, ARG_OUT); ldc64(e, D, ARG_TAB);
    ldc32(e, p_maxb, ARG_MAXB); ldc32(e, scale, ARG_SCALE); ldc32(e, p_bstride, ARG_BSTRIDE); ldc32(e, p_tstride, ARG_TSTRIDE);
    ldc32(e, p_kvp, ARG_KVP); ldc32(e, p_hstride, ARG_HSTRIDE); ldc32(e, p_nqh, ARG_NQH); ldc32(e, p_ctxoff, ARG_CTXOFF);
    s2r(e, tid, BW_SR_TID_X); s2r(e, h, BW_SR_CTAID_X); s2r(e, seq, BW_SR_CTAID_Y);

    /* cut F, partial mode: grid y = sequence * S + split; the split index picks this CTA's slice of the context (below) */
    const bool part = k->split >= 0;
    const uint32_t LS = part ? (uint32_t)k->split : 0u, S = 1u << LS, log2hd = (HD == 128u) ? 7u : 6u;
    int split = -1, kb = -1, kend = -1;
    if (part) { split = V(e); if (LS) { lop3andi(e, split, seq, S - 1u); shri(e, seq, seq, LS); } else movrz(e, split); }

    int c1 = V(e), c4 = V(e), c32 = V(e), c64 = V(e), c256 = V(e), c512 = V(e), c768 = V(e), c896 = V(e), zero = V(e);
    /* c32 / c64 keep their hd 64 names: the inner (dim pair) loop trip count HD / 2 and the chunk size HD */
    movi(e, c1, 1); movi(e, c4, 4); movi(e, c32, HD / 2); movi(e, c64, HD);
    movi(e, c256, SH_S); movi(e, c512, SH_O); movi(e, c768, SH_R); movi(e, c896, SH_R + 128); movrz(e, zero);
    /* warp partials 2 and 3 (lane 0 of warps 2 and 3): only a 128-thread CTA has them. Created and emitted only
     * when HD == 128, so the hd 64 program (vreg numbering, instruction sequence) is exactly the one before. */
    int c1024 = -1, c1152 = -1, rc_ = -1, rd_ = -1;
    if (HD == 128) { c1024 = V(e); c1152 = V(e); rc_ = V(e); rd_ = V(e); movi(e, c1024, SH_R + 256); movi(e, c1152, SH_R + 384); }

    /* q[tid] -> shared; out address kept for the end. Row = seq * num_q_heads + head for
     * both q and out (flattened head-major, the oracle's layout). The head-wiring mutants
     * swap the row of head h with h ^ 1 on one side only. GC > 1: head g of the group is
     * h * GC + g; q goes to SH_Q + g * HB; the output rows are derived in the epilogue. */
    int tid4 = V(e), qrow = V(e), qoff = V(e), qaddr = V64(e), qv = V(e), oaddr = GC == 1 ? V64(e) : -1;
    int hq = h, ho = h;
    if (GC == 1 && k->mutant == OMEGA_GPU_ATTN_MUTANT_Q_ROW) { hq = V(e); lop3xor(e, hq, h, c1); }
    if (GC == 1 && k->mutant == OMEGA_GPU_ATTN_MUTANT_OUT_ROW) { ho = V(e); lop3xor(e, ho, h, c1); }
    imadi(e, tid4, tid, 4, NOREG);
    if (GC == 1) {
        imadr(e, qrow, seq, p_nqh, hq);
        imadi(e, qoff, qrow, HD * 4, tid4);
        wide(e, qaddr, qoff, 1, A);
        ldg(e, qv, qaddr, udesc);
        sts32(e, tid4, qv); /* SH_Q + tid*4 */
        if (!part) { /* direct output address; partial mode derives its row in the epilogue */
            if (ho == hq) wide(e, oaddr, qoff, 1, C);
            else {
                int orow = V(e), ooff = V(e);
                imadr(e, orow, seq, p_nqh, ho);
                imadi(e, ooff, orow, HD * 4, tid4);
                wide(e, oaddr, ooff, 1, C);
            }
        }
    } else {
        int h0 = V(e), hg = V(e), qsh = V(e);
        imadi(e, h0, h, GC, NOREG);
        for (uint32_t g = 0; g < GC; g++) {
            imadi(e, hg, c1, g, h0);
            if (k->mutant == OMEGA_GPU_ATTN_MUTANT_Q_ROW) lop3xor(e, hg, hg, c1);
            imadr(e, qrow, seq, p_nqh, hg);
            imadi(e, qoff, qrow, HD * 4, tid4);
            wide(e, qaddr, qoff, 1, A);
            ldg(e, qv, qaddr, udesc);
            imadi(e, qsh, c1, g * HB, tid4);
            sts32(e, qsh, qv);
        }
    }

    /* kv head offset, this sequence's table row and context length */
    int kvh = V(e), kvoff = V(e), t1 = V(e), tab = V64(e), ca = V64(e), ctx = V(e), vb = V64(e);
    shri(e, kvh, h, k->log2gqa - log2gc); /* GC heads per CTA: kv head = (h * GC) >> log2gqa */
    if (k->mutant == OMEGA_GPU_ATTN_MUTANT_KV_HEAD) lop3xor(e, kvh, kvh, c1);
    imadr(e, kvoff, kvh, p_hstride, NOREG);
    imadr(e, t1, seq, p_maxb, NOREG);
    imadi(e, t1, t1, 4, NOREG);
    wide(e, tab, t1, 1, D);
    imadi(e, t1, seq, 4, p_ctxoff);
    wide(e, ca, t1, 1, D);
    ldg(e, ctx, ca, udesc);
    if (part) {
        /* This sequence's own split count s_own = min(S, pow2floor(chunks)), chunks = ceil(ctx / HD), from its own
         * context alone (the host picks the launch S the same way over the batch); per = roundup(ceil(ctx / s_own), HD)
         * whole chunks per split; this CTA's slice is [kb, kend), kend = min(ctx, kb + per) through the sign bit of
         * ctx - (kb + per). Splits past s_own get kb >= kend: an empty partial. Branch-free: candidate 2^c replaces
         * per when chunks >= 2^c; the candidates nest, so the largest that fits wins. */
        int per = V(e), x = V(e), cand = V(e), dd = V(e), lt = V(e), cH1 = V(e), cK = V(e);
        kb = V(e); kend = V(e);
        movi(e, cH1, HD - 1u);
        iadd3(e, per, ctx, cH1); lop3andi(e, per, per, ~(HD - 1u)); /* roundup(ctx, HD): one split takes it all */
        shri(e, x, per, log2hd);                                     /* chunks (cut J: one whole chunk per split) */
        for (uint32_t c = 1; c <= LS; c++) {
            movi(e, cK, (1u << c) - 1u); iadd3(e, cand, ctx, cK); shri(e, cand, cand, c);
            iadd3(e, cand, cand, cH1); lop3andi(e, cand, cand, ~(HD - 1u)); /* roundup(ceil(ctx / 2^c), HD) */
            movi(e, cK, 0u - (1u << c)); iadd3(e, dd, x, cK); shri(e, lt, dd, 31);  /* lt = (x < 2^c) */
            imadi(e, dd, cand, 0xFFFFFFFFu, per);                                    /* per - cand */
            imadr(e, per, lt, dd, cand);                                             /* lt ? per : cand */
        }
        imadr(e, kb, split, per, NOREG);
        iadd3(e, kend, kb, per);
        imadi(e, dd, kend, 0xFFFFFFFFu, ctx);
        shri(e, lt, dd, 31);
        imadr(e, kend, lt, dd, kend);
    }
    const int lim = part ? kend : ctx; /* the end of this CTA's token range */
    /* V column base for dimension tid: pool + kv plane + tid * elem. The kv head offset is
     * already inside the per-token offset phase A leaves in shared memory (found by the host
     * simulator: adding it here too read V from the wrong head for every kv head but 0). */
    wide(e, vb, tid, elem, B);
    wide(e, vb, p_kvp, 1, vb);

    /* running max, sum and accumulator per head of the group (GC = 1: one of each, as before) */
    int M[4], L[4], acc[4];
    for (uint32_t g = 0; g < GC; g++) M[g] = V(e);
    for (uint32_t g = 0; g < GC; g++) L[g] = V(e);
    for (uint32_t g = 0; g < GC; g++) acc[g] = V(e);
    int base = V(e), sOff = V(e), oOff = V(e), rOff = V(e);
    for (uint32_t g = 0; g < GC; g++) movi(e, M[g], F32_NEG_INF);
    for (uint32_t g = 0; g < GC; g++) { if (k->mutant == OMEGA_GPU_ATTN_MUTANT_NO_MAX) movrz(e, L[g]); else movf(e, L[g], 1.0f); } /* alpha = 0 on the first chunk clears it */
    for (uint32_t g = 0; g < GC; g++) movrz(e, acc[g]);
    if (part) iadd3(e, base, kb, zero); else movrz(e, base);
    iadd3(e, sOff, tid4, c256); iadd3(e, oOff, tid4, c512); iadd3(e, rOff, tid4, c768);
    bar_sync(e); /* q visible to every thread */
    if (part) isetp_ge_u32(e, kb, kend); else isetp_ge_u32(e, zero, ctx); /* empty range: out = 0 (an empty partial) */
    int bZ = bra_fwd_if_p0(e);

    /* ---------------- chunk loop ---------------- */
    int j = V(e), s[4], koff;
    for (uint32_t g = 0; g < GC; g++) s[g] = V(e);
    koff = V(e);
    int bi = V(e), sl = V(e), ta = V64(e), blk = V(e), ka = V64(e), dot[4], d;
    for (uint32_t g = 0; g < GC; g++) dot[g] = V(e);
    d = V(e);
    int lo = V(e), hi = V(e), qa = V(e), j1 = V(e);
    int dneg = V(e), over = V(e), ot = V(e), ctxm1 = V(e), cm1 = V(e), cninf = V(e), nb = V(e), ns = V(e);
    int tsh = GC > 1 ? V(e) : -1; /* per-head shared address temporary (GC > 1 only) */
    movi(e, cm1, 0xFFFFFFFFu); iadd3(e, ctxm1, lim, cm1); movi(e, cninf, F32_NEG_INF);
    loop_begin(e);
    int Lout = here(e);
    /* No branch here: a warp split by "j >= ctx" without a BSSY/BSYNC region stays split, and
     * the SHFL reductions below then read inactive lanes, which is undefined (wrong dims
     * tid >= tail on the chip, 2026-10-04). The IR has BSSY/BSYNC regions since omega #310
     * (omega_bw_reconv.h); this kernel stays branchless and does not use them.
     * over = (j >= ctx) as 0/1 from the sign of ctx-1-j; out-of-range lanes read token `base`
     * (always < ctx here) and their score is forced to -inf after the dot product with s = -max(-s, nb), nb = -inf or +inf. */
    iadd3(e, j, base, tid);
    for (uint32_t g = 0; g < GC; g++) movi(e, s[g], F32_NEG_INF);
    movrz(e, koff);
    if (k->divergent) {
        /* omega #308 R2: the lane-dependent branch, inside a reconvergence region. Lanes with
         * j >= ctx skip phase A (s stays -inf, koff 0); the BSYNC at region_join rejoins them
         * before the SHFL reductions below. divergent == 2 drops that BSYNC (negative control). */
        region_begin(e);
        isetp_ge_u32(e, j, lim);
        region_exit(e, RX_IF_P0);
    } else {
        imadi(e, dneg, j, 0xFFFFFFFFu, ctxm1); /* ctx - 1 - j */
        shri(e, over, dneg, 31);
        imadr(e, ot, over, tid, NOREG);
        imadi(e, j, ot, 0xFFFFFFFFu, j);       /* j - over*tid */
    }
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
    /* phase A inner loop (batched, 2026-10-12): two 128-bit global loads of K per pass on SB_GPR (16 dims
     * bf16, 8 dims f32) with the first half of the matching q slice as 128-bit shared loads on SB_LD2, one
     * wait; then the second half of q, one more (short) wait. The serial loop before paid one global
     * latency per dim pair. Two halves keep the pass inside AT_GPR_BUDGET (the whole slice at once is
     * 65 to 76 registers). qa = d * 4 is the shared byte offset of q[d]. GC > 1: the same K pass feeds
     * every head of the group, each from its own q slice at SH_Q + g * HB (cut K). */
    const uint32_t dims = k->f32 ? 8u : 16u; /* dims per pass: 32 bytes of K either way */
    const uint32_t nq = dims / 4u;           /* q quads per pass, loaded in two halves */
    int kx[2], qq[4];
    kx[0] = V128(e); kx[1] = V128(e);
    for (uint32_t i = 0; i < nq; i++) qq[i] = V128(e);
    for (uint32_t g = 0; g < GC; g++) movrz(e, dot[g]);
    movrz(e, d);
    loop_begin(e);
    int Lin = here(e);
    imadi(e, qa, d, 4, NOREG);
    ldg128_b(e, kx[0], ka, 0u, udesc, SB_GPR);
    ldg128_b(e, kx[1], ka, 16u, udesc, SB_GPR);
    for (uint32_t g = 0; g < GC; g++) {
        for (int hh = 0; hh < 2; hh++) {
            for (uint32_t i = 0; i < nq / 2u; i++) { uint32_t qi = (uint32_t)hh * (nq / 2u) + i; lds128_b(e, qq[qi], qa, g * HB + qi * 16u, SB_LD2); }
            if (k->f32) {
                for (int s4 = 0; s4 < 4; s4++) e_ffma_s(e, dot[g], qq[hh], s4, kx[hh], s4, dot[g]); /* dim hh*4+s4 */
            } else {
                for (int s4 = 0; s4 < 4; s4++) {
                    int pi = hh * 4 + s4; /* bf16 pair: dims 2*pi (low half) and 2*pi+1 (high half) */
                    imadi_s(e, lo, kx[hh], s4, 0x10000u, NOREG);   /* low bf16 -> f32: bits << 16 */
                    lop3andi_s(e, hi, kx[hh], s4, 0xffff0000u);    /* high bf16 -> f32: keep the top half */
                    e_ffma_s(e, dot[g], qq[pi / 2], (pi % 2) * 2, lo, 0, dot[g]);
                    e_ffma_s(e, dot[g], qq[pi / 2], (pi % 2) * 2 + 1, hi, 0, dot[g]);
                }
            }
        }
    }
    wide(e, ka, c1, 32u, ka);
    imadi(e, d, c1, dims, d);
    isetp_ge_u32(e, d, c64);
    bra_back_if_not_p0(e, Lin);
    loop_end(e);
    for (uint32_t g = 0; g < GC; g++) e_fmul(e, s[g], dot[g], scale);
    if (k->divergent) {
        region_join(e); /* every lane back before the cross-lane reductions */
    } else {
        imadi(e, nb, over, 0x80000000u, cninf); /* over ? +inf : -inf (bits) */
        for (uint32_t g = 0; g < GC; g++) { e_fsub(e, ns, zero, s[g]); e_fmax(e, ns, ns, nb); e_fsub(e, s[g], zero, ns); }
    }
    for (uint32_t g = 0; g < GC; g++) {
        if (g == 0) sts32(e, sOff, s[0]);
        else { imadi(e, tsh, c1, g * HB, sOff); sts32(e, tsh, s[g]); }
    }
    sts32(e, oOff, koff);

    /* chunk max, per head */
    int t = V(e), mw = V(e), ra = V(e), rb = V(e), mc = V(e), Mn = V(e), diff = V(e), alpha[4], parg, p, sw, lc;
    alpha[0] = V(e); parg = V(e); p = V(e); sw = V(e); lc = V(e);
    for (uint32_t g = 1; g < GC; g++) alpha[g] = V(e);
    static const uint32_t steps[5] = { 16, 8, 4, 2, 1 };
    for (uint32_t g = 0; g < GC; g++) {
        shfl_down(e, t, s[g], steps[0]); e_fmax(e, mw, s[g], t);
        for (int i = 1; i < 5; i++) { shfl_down(e, t, mw, steps[i]); e_fmax(e, mw, mw, t); }
        if (g == 0) sts32(e, rOff, mw);
        else { imadi(e, tsh, c1, g * HB, rOff); sts32(e, tsh, mw); }
    }
    bar_sync(e);
    for (uint32_t g = 0; g < GC; g++) {
        const int pg = GC == 1 ? p : s[g]; /* GC > 1: p overwrites the score register of its head */
        if (g == 0) { lds32(e, ra, c768); lds32(e, rb, c896); }
        else { imadi(e, tsh, c1, g * HB, c768); lds32(e, ra, tsh); imadi(e, tsh, c1, g * HB, c896); lds32(e, rb, tsh); }
        e_fmax(e, mc, ra, rb);
        if (HD == 128) {
            if (g == 0) { lds32(e, rc_, c1024); lds32(e, rd_, c1152); }
            else { imadi(e, tsh, c1, g * HB, c1024); lds32(e, rc_, tsh); imadi(e, tsh, c1, g * HB, c1152); lds32(e, rd_, tsh); }
            e_fmax(e, mc, mc, rc_); e_fmax(e, mc, mc, rd_);
        }
        e_fmax(e, Mn, M[g], mc);
        if (k->mutant == OMEGA_GPU_ATTN_MUTANT_NO_MAX) {
            movf(e, alpha[g], 1.0f);
            mufu(e, BW_IR_MUFU_EX2, pg, s[g]);
            e_fmax(e, M[g], Mn, mc);
        } else {
            e_fsub(e, diff, M[g], Mn);
            mufu(e, BW_IR_MUFU_EX2, alpha[g], diff);
            e_fmax(e, M[g], Mn, mc); /* M = Mn */
            e_fsub(e, parg, s[g], Mn);
            mufu(e, BW_IR_MUFU_EX2, pg, parg);
        }
    }
    bar_sync(e); /* everyone has read the max partials before they are overwritten */
    for (uint32_t g = 0; g < GC; g++) {
        const int pg = GC == 1 ? p : s[g];
        if (g == 0) sts32(e, sOff, pg);
        else { imadi(e, tsh, c1, g * HB, sOff); sts32(e, tsh, pg); }
        shfl_down(e, t, pg, steps[0]); e_fadd(e, sw, pg, t);
        for (int i = 1; i < 5; i++) { shfl_down(e, t, sw, steps[i]); e_fadd(e, sw, sw, t); }
        if (g == 0) sts32(e, rOff, sw);
        else { imadi(e, tsh, c1, g * HB, rOff); sts32(e, tsh, sw); }
    }
    bar_sync(e);
    for (uint32_t g = 0; g < GC; g++) {
        if (g == 0) { lds32(e, ra, c768); lds32(e, rb, c896); }
        else { imadi(e, tsh, c1, g * HB, c768); lds32(e, ra, tsh); imadi(e, tsh, c1, g * HB, c896); lds32(e, rb, tsh); }
        e_fadd(e, lc, ra, rb);
        if (HD == 128) {
            if (g == 0) { lds32(e, rc_, c1024); lds32(e, rd_, c1152); }
            else { imadi(e, tsh, c1, g * HB, c1024); lds32(e, rc_, tsh); imadi(e, tsh, c1, g * HB, c1152); lds32(e, rd_, tsh); }
            e_fadd(e, lc, lc, rc_); e_fadd(e, lc, lc, rd_);
        }
        e_ffma(e, L[g], L[g], alpha[g], lc);
        if (k->mutant != OMEGA_GPU_ATTN_MUTANT_NO_RESCALE) e_fmul(e, acc[g], acc[g], alpha[g]);
    }

    /* phase B (batched, 2026-10-12): dimension tid over the chunk's tokens, AT_TOK tokens per pass. The
     * token offsets of a pass come in two 128-bit shared loads (SB_LD2, one wait), then each V address is
     * computed into one register pair while the earlier V loads are in flight (emit_fixed_in_batch: the
     * read barrier orders the overwrite after the load has read it) and its load issued on SB_GPR or
     * SB_LD3; the p values follow as two more shared loads on SB_LD2; one wait for all of it. A token past
     * ctx inside a pass has p = 0 and the offset of a real token (phase A), so it adds exactly 0.
     * GC > 1: one V load per token feeds every head's accumulator from that head's p slice (cut K). */
    enum { AT_TOK = 8 };
    int kk = V(e), tt = V(e), oa = V(e), pa = V(e), ko4[2], pv4[2], va = V64(e), v[AT_TOK];
    for (int i = 0; i < 2; i++) { ko4[i] = V128(e); pv4[i] = V128(e); }
    for (int i = 0; i < AT_TOK; i++) v[i] = V(e);
    movrz(e, kk);
    loop_begin(e);
    int LB = here(e);
    iadd3(e, tt, base, kk);
    isetp_ge_u32(e, tt, lim);
    int bB = bra_fwd_if_p0(e);
    imadi(e, oa, kk, 4, c512);
    imadi(e, pa, kk, 4, c256);
    for (int i = 0; i < 2; i++) lds128_b(e, ko4[i], oa, (uint32_t)i * 16u, SB_LD2);
    for (int i = 0; i < AT_TOK; i++) {
        if (i == 0) wide_s(e, va, ko4[0], 0, 1, vb); /* ordinary: waits for the offsets */
        else wide_s_b(e, va, ko4[i / 4], i % 4, 1, vb);
        uint32_t sb = (i < AT_TOK / 2) ? SB_GPR : SB_LD3;
        if (k->f32) ldg_b(e, v[i], va, udesc, sb); else ldg16_b(e, v[i], va, udesc, sb);
    }
    if (GC == 1) {
        for (int i = 0; i < 2; i++) lds128_b(e, pv4[i], pa, (uint32_t)i * 16u, SB_LD2);
        for (int i = 0; i < AT_TOK; i++) {
            if (!k->f32) imadi(e, v[i], v[i], 0x10000u, NOREG);
            e_ffma_s(e, acc[0], pv4[i / 4], i % 4, v[i], 0, acc[0]);
        }
    } else {
        if (!k->f32) for (int i = 0; i < AT_TOK; i++) imadi(e, v[i], v[i], 0x10000u, NOREG);
        for (uint32_t g = 0; g < GC; g++) {
            for (int i = 0; i < 2; i++) lds128_b(e, pv4[i], pa, g * HB + (uint32_t)i * 16u, SB_LD2);
            for (int i = 0; i < AT_TOK; i++) e_ffma_s(e, acc[g], pv4[i / 4], i % 4, v[i], 0, acc[g]);
        }
    }
    imadi(e, kk, c1, AT_TOK, kk);
    isetp_ge_u32(e, kk, c64);
    bra_back_if_not_p0(e, LB);
    loop_end(e);
    patch_fwd(e, bB);
    bar_sync(e); /* phase B reads of s/p and offsets finish before the next chunk writes them */
    iadd3(e, base, base, c64);
    isetp_ge_u32(e, base, lim);
    bra_back_if_not_p0(e, Lout);
    loop_end(e);

    patch_fwd(e, bZ);
    if (part && GC == 1) {
        /* partial row (seq * nqh + out head) * S + split: acc[HD] unnormalised, then M and L at words HD and HD + 1
         * (every lane stores the same CTA-wide M and L). Everything is re-derived here from the CTA ids and the
         * constant bank so no extra value lives across the chunk loop (the hd 128 kernel sits at the register budget). */
        int cy = V(e), sq = V(e), sp = V(e), hh = V(e), nq_ = V(e), t4 = V(e), prow = V(e), poff = V(e);
        int paddr = V64(e), maddr = V64(e), laddr = V64(e), Cp = V64(e);
        s2r(e, cy, BW_SR_CTAID_Y); s2r(e, hh, BW_SR_CTAID_X); ldc32(e, nq_, ARG_NQH); ldc64(e, Cp, ARG_OUT);
        if (k->mutant == OMEGA_GPU_ATTN_MUTANT_OUT_ROW) lop3xor(e, hh, hh, c1);
        if (LS) { lop3andi(e, sp, cy, S - 1u); shri(e, sq, cy, LS); } else { movrz(e, sp); iadd3(e, sq, cy, zero); }
        imadr(e, prow, sq, nq_, hh);
        imadi(e, prow, prow, S, sp);
        imadi(e, t4, tid, 4, NOREG);
        imadi(e, poff, prow, (HD + 2u) * 4u, t4);
        wide(e, paddr, poff, 1, Cp);
        stg(e, paddr, acc[0], udesc);
        imadi(e, poff, prow, (HD + 2u) * 4u, NOREG);
        wide(e, maddr, poff, 1, Cp);
        wide(e, maddr, c64, 4, maddr);
        stg(e, maddr, M[0], udesc);
        wide(e, laddr, c1, 4, maddr);
        stg(e, laddr, L[0], udesc);
    } else if (part) {
        /* GC heads: row (seq * nqh + h * GC + g) * S + split for head g (OUT_ROW: g ^ 1 on the output side) */
        int cy = V(e), sq = V(e), sp = V(e), h0 = V(e), hg = V(e), nq_ = V(e), t4 = V(e), prow = V(e), poff = V(e);
        int paddr = V64(e), maddr = V64(e), laddr = V64(e), Cp = V64(e);
        s2r(e, cy, BW_SR_CTAID_Y); s2r(e, h0, BW_SR_CTAID_X); ldc32(e, nq_, ARG_NQH); ldc64(e, Cp, ARG_OUT);
        imadi(e, h0, h0, GC, NOREG);
        if (LS) { lop3andi(e, sp, cy, S - 1u); shri(e, sq, cy, LS); } else { movrz(e, sp); iadd3(e, sq, cy, zero); }
        imadi(e, t4, tid, 4, NOREG);
        for (uint32_t g = 0; g < GC; g++) {
            imadi(e, hg, c1, g, h0);
            if (k->mutant == OMEGA_GPU_ATTN_MUTANT_OUT_ROW) lop3xor(e, hg, hg, c1);
            imadr(e, prow, sq, nq_, hg);
            imadi(e, prow, prow, S, sp);
            imadi(e, poff, prow, (HD + 2u) * 4u, t4);
            wide(e, paddr, poff, 1, Cp);
            stg(e, paddr, acc[g], udesc);
            imadi(e, poff, prow, (HD + 2u) * 4u, NOREG);
            wide(e, maddr, poff, 1, Cp);
            wide(e, maddr, c64, 4, maddr);
            stg(e, maddr, M[g], udesc);
            wide(e, laddr, c1, 4, maddr);
            stg(e, laddr, L[g], udesc);
        }
    } else if (GC == 1) {
        int inv = V(e), o = V(e);
        mufu(e, BW_IR_MUFU_RCP, inv, L[0]);
        e_fmul(e, o, acc[0], inv);
        stg(e, oaddr, o, udesc);
    } else {
        /* direct output, GC heads: row seq * nqh + h * GC + g (OUT_ROW: g ^ 1), re-derived from the CTA ids */
        int inv = V(e), o = V(e), cy = V(e), h0 = V(e), hg = V(e), nq_ = V(e), t4 = V(e), orow = V(e), ooff = V(e);
        int oa64 = V64(e), Cp = V64(e);
        s2r(e, cy, BW_SR_CTAID_Y); s2r(e, h0, BW_SR_CTAID_X); ldc32(e, nq_, ARG_NQH); ldc64(e, Cp, ARG_OUT);
        imadi(e, h0, h0, GC, NOREG);
        imadi(e, t4, tid, 4, NOREG);
        for (uint32_t g = 0; g < GC; g++) {
            imadi(e, hg, c1, g, h0);
            if (k->mutant == OMEGA_GPU_ATTN_MUTANT_OUT_ROW) lop3xor(e, hg, hg, c1);
            imadr(e, orow, cy, nq_, hg);
            imadi(e, ooff, orow, HD * 4, t4);
            wide(e, oa64, ooff, 1, Cp);
            mufu(e, BW_IR_MUFU_RCP, inv, L[g]);
            e_fmul(e, o, acc[g], inv);
            stg(e, oa64, o, udesc);
        }
    }
    tail(e);
}

#define AT_GPR_BUDGET 96u /* cut K: the GQA-group hd 128 kernel carries four heads of running state (the matmul blocked kernels launch at 108) */

static int build_kernel(const KSpec *k, OmegaBlackwellKernel *kernel, BlackwellIRProgram **keep) {
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) return OMEGA_GPU_ATTN_CODEGEN_FAIL;
    omega_bw_ir_init(prog);
    Em e; memset(&e, 0, sizeof e); e.p = prog;
    e.rg_mutant = (k->divergent == 2) ? 1 : 0;
    gen_attention(&e, k);
    if (e.depth != 0) e.err = -1;
    if (e.rg.depth != 0) e.err = -1; /* a region left open has no join: refuse the build */
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
        kernel->code = omega_gpu_host_alloc(max_bytes);
        size_t emitted = 0;
        if (kernel->code && omega_bw_encode_program(prog, kernel->code, max_bytes, &emitted) == 0) {
            kernel->code_size = emitted;
            kernel->insn_count = emitted / 16;
            kernel->gpr_count = prog->regalloc.peak_gpr_usage;
            kernel->uniform_gpr_count = prog->regalloc.peak_ugpr_usage;
            sha256_hash(kernel->code, kernel->code_size, kernel->code_digest);
            rc = OMEGA_GPU_ATTN_OK;
        } else {
            omega_gpu_host_free(kernel->code); kernel->code = NULL;
        }
    }
    if (keep && rc == OMEGA_GPU_ATTN_OK) *keep = prog; else omega_gpu_host_free(prog);
    return rc;
}

/* --------------------------------------------------------- kernel cache */

/* One kernel per KSpec; the device copy is uploaded once and stays resident for
 * the process (cut 4b). The IR program is kept for the host simulator. */
static int g_mutant;
static int g_divergent; /* omega #308 R2 test hook: see KSpec.divergent */
static bool g_dedupe = true; /* shared-prefix staging dedupe; false = legacy per-sequence staging (test hook) */
typedef struct { int used; KSpec k; OmegaBlackwellKernel kernel; BlackwellIRProgram *prog; OmegaGpuMem code; } Slot;
static OmegaGpuAttnSim g_sim;
static Slot g_cache[16]; /* split variants (cut F) multiply the shapes in flight */
static int g_split = -1; /* test hook: forced log2split (0..OMEGA_GPU_ATTN_MAX_LOG2SPLIT), -1 = chosen per launch from the context */
static OmegaGpuScratch g_q, g_pool, g_tab, g_out; /* staging, grown to the high-water mark, reused per call */
static OmegaGpuMem g_resident; /* the resident KV pool (omega_gpu_attention_pool_alloc); cpu NULL = none */
static bool g_hooked;
#define LOCK() omega_gpu_session_lock()
#define UNLOCK() omega_gpu_session_unlock()

void omega_gpu_attention_test_set_simulator(OmegaGpuAttnSim sim) { LOCK(); g_sim = sim; UNLOCK(); }
static uint32_t g_sim_shared_bytes; /* shared bytes the launcher would declare for the kernel the simulator is running */
uint32_t omega_gpu_attention_test_shared_bytes(void) { return g_sim_shared_bytes; }
void omega_gpu_attention_test_set_dedupe(bool on) { LOCK(); g_dedupe = on; UNLOCK(); }
void omega_gpu_attention_test_set_split(int log2split) { LOCK(); g_split = (log2split >= 0 && log2split <= (int)OMEGA_GPU_ATTN_MAX_LOG2SPLIT) ? log2split : -1; UNLOCK(); }

/* ---- resident KV pool (see the header) ---- */

static int dev_open_locked(void);

/* The kernel computes block_id * block_stride + slot * token_stride + head offset in 32 bits before
 * widening onto the 64-bit layer base: one layer slice (both planes, every token and head of the
 * block) past the block's start must stay inside 32 bits. */
static bool resident_fits(const OmegaGpuKvLayout *ly, uint32_t block_id) {
    return ly->block_stride_bytes <= 0xffffffffull &&
           (uint64_t)block_id * ly->block_stride_bytes + ly->layer_stride_bytes <= 0xffffffffull;
}
bool omega_gpu_attention_test_resident_fits(const OmegaGpuKvLayout *layout, uint32_t block_id) {
    return layout && resident_fits(layout, block_id);
}

int omega_gpu_attention_pool_alloc(size_t bytes, void **cpu) {
    if (!cpu || bytes == 0) return OMEGA_GPU_ATTN_BAD_ARGS;
    *cpu = NULL;
    LOCK();
    if (g_resident.cpu) { omega_gpu_session_set_error("attention pool: already allocated"); UNLOCK(); return OMEGA_GPU_ATTN_BAD_ARGS; }
    if (g_sim) {
        g_resident.cpu = omega_gpu_host_calloc(1, bytes);
        if (!g_resident.cpu) { UNLOCK(); return OMEGA_GPU_ATTN_CHIP_FAIL; }
        g_resident.va = (uintptr_t)g_resident.cpu; g_resident.size = bytes;
    } else {
        int rc = dev_open_locked();
        if (rc != OMEGA_GPU_ATTN_OK) { UNLOCK(); return rc; }
        if (omega_gpu_session_alloc(bytes, &g_resident) != 0) { memset(&g_resident, 0, sizeof g_resident); UNLOCK(); return OMEGA_GPU_ATTN_CHIP_FAIL; }
    }
    *cpu = g_resident.cpu;
    UNLOCK();
    return OMEGA_GPU_ATTN_OK;
}

void omega_gpu_attention_pool_free(void) {
    LOCK();
    if (g_resident.cpu) { if (g_sim) omega_gpu_host_free(g_resident.cpu); else omega_gpu_session_free(&g_resident); }
    memset(&g_resident, 0, sizeof g_resident);
    UNLOCK();
}

static void cache_clear_locked(void) {
    for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++)
        if (g_cache[i].used) { omega_gpu_host_free(g_cache[i].kernel.code); omega_gpu_host_free(g_cache[i].prog); omega_gpu_session_free(&g_cache[i].code); memset(&g_cache[i], 0, sizeof g_cache[i]); }
}

void omega_gpu_attention_cache_clear(void) {
    LOCK();
    cache_clear_locked();
    UNLOCK();
}

/* Called by the session (lock held, device still open) right before it closes. */
static void on_session_close(void) {
    cache_clear_locked();
    memset(&g_resident, 0, sizeof g_resident); /* the session's memory goes with it; the caller's mapping is dead */
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
static int kernel_for(bool f32, uint32_t log2bs, uint32_t log2gqa, uint32_t hd, int split, bool upload, Slot **out, bool *hit) {
    KSpec k = { .f32 = f32, .log2bs = log2bs, .log2gqa = log2gqa, .mutant = g_mutant, .divergent = g_divergent, .hd = hd, .split = split };
    Slot *s = NULL;
    for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++)
        if (g_cache[i].used && g_cache[i].k.f32 == k.f32 && g_cache[i].k.log2bs == k.log2bs && g_cache[i].k.log2gqa == k.log2gqa && g_cache[i].k.mutant == k.mutant && g_cache[i].k.divergent == k.divergent && g_cache[i].k.hd == k.hd && g_cache[i].k.split == k.split) { /* field compare: memcmp would read struct padding */
            s = &g_cache[i]; *hit = true; break;
        }
    if (!s) {
        OmegaBlackwellKernel kern; BlackwellIRProgram *kp = NULL;
        int rc = build_kernel(&k, &kern, &kp);
        if (rc != OMEGA_GPU_ATTN_OK) return rc;
        size_t slot = 0;
        for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++) if (!g_cache[i].used) { slot = i; break; }
        s = &g_cache[slot];
        if (s->used) { omega_gpu_host_free(s->kernel.code); omega_gpu_host_free(s->prog); omega_gpu_session_free(&s->code); } /* full: evict slot 0 */
        memset(s, 0, sizeof *s);
        s->used = 1; s->k = k; s->kernel = kern; s->prog = kp;
        *hit = false;
    }
    if (upload && !s->code.cpu) {
        if (omega_gpu_session_alloc_code(s->kernel.code_size, &s->code) != 0) return OMEGA_GPU_ATTN_CHIP_FAIL;
        memcpy(s->code.cpu, s->kernel.code, s->kernel.code_size);
        if (omega_gpu_session_sync_to_gpu(&s->code) != 0) { omega_gpu_session_free(&s->code); return OMEGA_GPU_ATTN_CHIP_FAIL; }
    }
    *out = s;
    return OMEGA_GPU_ATTN_OK;
}

int omega_gpu_attention_codegen_hd(bool kv_f32, uint32_t log2_block_size, uint32_t log2_gqa, uint32_t head_dim, OmegaBlackwellKernel *kernel) {
    if (!kernel || log2_block_size > 16 || log2_gqa > 6) return OMEGA_GPU_ATTN_BAD_ARGS;
    if (head_dim != 64u && head_dim != 128u) return OMEGA_GPU_ATTN_TOO_LARGE;
    LOCK();
    KSpec k = { .f32 = kv_f32, .log2bs = log2_block_size, .log2gqa = log2_gqa, .mutant = g_mutant, .divergent = g_divergent, .hd = head_dim, .split = -1 };
    UNLOCK();
    return build_kernel(&k, kernel, NULL);
}

int omega_gpu_attention_codegen(bool kv_f32, uint32_t log2_block_size, uint32_t log2_gqa, OmegaBlackwellKernel *kernel) {
    return omega_gpu_attention_codegen_hd(kv_f32, log2_block_size, log2_gqa, OMEGA_GPU_ATTN_HEAD_DIM, kernel);
}

const char *omega_gpu_attention_last_error(void) { return omega_gpu_session_last_error(); }

/* ------------------------------------------------------------ launcher */

#define AT_POISON 0xffbadbadu
#define AT_WAIT_MS 600000ull

static uint64_t now_ns(void) { return omega_gpu_session_now_ns(); } /* the platform clock (CLOCK_NS) */

typedef OmegaGpuAttnLaunch Launch;

/* Caller holds the lock. Chip path: open the shared device on first use. */
static int dev_open_locked(void) {
    if (!g_hooked) { (void)omega_gpu_session_on_close(on_session_close); g_hooked = true; }
    return omega_gpu_session_open() ? OMEGA_GPU_ATTN_OK : OMEGA_GPU_ATTN_CHIP_FAIL;
}

/* Where a call stages the KV slice it is about to hand the kernel: straight into the
 * device staging buffer on the chip path (one copy), a host buffer under the simulator.
 * Caller holds the lock; the result is valid until the next staging call. */
static void *stage_target(size_t bytes, void **host_tmp, int *fail_rc) {
    *host_tmp = NULL;
    *fail_rc = OMEGA_GPU_ATTN_CHIP_FAIL;
    if (g_sim) { *host_tmp = omega_gpu_host_alloc(bytes ? bytes : 1); return *host_tmp; }
    int r = omega_gpu_session_scratch(&g_pool, bytes);
    if (r != 0) { if (r == OMEGA_GPU_SESSION_SCRATCH_RESERVED) *fail_rc = OMEGA_GPU_ATTN_TOO_LARGE; return NULL; } /* reserved bound exceeded: loud, no allocation */
    return g_pool.mem.cpu;
}

/* One launch through the shared session (omega_gpu_session_launch: shader-cache
 * invalidate, constant bank, QMD0/QMD1, first marker, L2_FLUSH_DIRTY, second marker),
 * then readback and the poison scan. L->pool may already be the pool staging buffer. */
static int run_launch(const Slot *ks, const Launch *L, OmegaGpuAttnInfo *info) {
    int sr = omega_gpu_session_scratch(&g_q, L->q_bytes);
    if (!sr) sr = omega_gpu_session_scratch(&g_tab, L->tab_bytes);
    if (!sr) sr = omega_gpu_session_scratch(&g_out, L->out_bytes);
    if (!sr && !L->pool_va && L->pool != g_pool.mem.cpu) sr = omega_gpu_session_scratch(&g_pool, L->pool_bytes);
    if (sr) return sr == OMEGA_GPU_SESSION_SCRATCH_RESERVED ? OMEGA_GPU_ATTN_TOO_LARGE : OMEGA_GPU_ATTN_CHIP_FAIL;
    memcpy(g_q.mem.cpu, L->q, L->q_bytes);
    if (!L->pool_va && L->pool != g_pool.mem.cpu) memcpy(g_pool.mem.cpu, L->pool, L->pool_bytes);
    const uint64_t pool_va = L->pool_va ? L->pool_va : g_pool.mem.va; /* resident: the caller's KV, read in place */
    memcpy(g_tab.mem.cpu, L->tab, L->tab_bytes);
    size_t out_words = L->out_bytes / 4;
    uint32_t *out_dev = (uint32_t *)g_out.mem.cpu;
    for (size_t i = 0; i < out_words; i++) out_dev[i] = AT_POISON;
    /* the KV this launch reads: the staged pool, or the resident pool the caller wrote in place */
    if (omega_gpu_session_sync_to_gpu(&g_q.mem) != 0 || omega_gpu_session_sync_to_gpu(&g_tab.mem) != 0 ||
        omega_gpu_session_sync_to_gpu(&g_out.mem) != 0 || omega_gpu_session_sync_to_gpu(L->pool_va ? &g_resident : &g_pool.mem) != 0)
        return OMEGA_GPU_ATTN_CHIP_FAIL;

    uint32_t args[ARG_WORDS];
    args[0] = (uint32_t)g_q.mem.va; args[1] = (uint32_t)(g_q.mem.va >> 32);
    args[2] = (uint32_t)pool_va; args[3] = (uint32_t)(pool_va >> 32);
    args[4] = (uint32_t)g_out.mem.va; args[5] = (uint32_t)(g_out.mem.va >> 32);
    args[6] = (uint32_t)g_tab.mem.va; args[7] = (uint32_t)(g_tab.mem.va >> 32);
    memcpy(&args[8], L->params, sizeof L->params);
    OmegaGpuLaunch GL = {
        .code_va = ks->code.va, .gpr_count = AT_GPR_BUDGET, .threads_x = ks->k.hd, .threads_y = 1, .shared_bytes = shared_bytes_for(ks->k.hd, ks->k.log2gqa), /* (3 GC + 1) arrays of hd words: 1024 B for hd 64, 2048 B for hd 128 gqa 1, 6656 B for hd 128 gqa 4 */
        .grid_x = L->grid_x, .grid_y = L->grid_y, .num_elements = out_words,
        .args = args, .n_args = ARG_WORDS, .timeout_ms = AT_WAIT_MS,
    };
    uint64_t elapsed = 0; uint32_t marker = 0;
    if (omega_gpu_session_launch(&GL, &elapsed, &marker) != 0) return OMEGA_GPU_ATTN_CHIP_FAIL;
    if (omega_gpu_session_sync_to_cpu(&g_out.mem) != 0) return OMEGA_GPU_ATTN_CHIP_FAIL;

    memcpy(L->out, g_out.mem.cpu, L->out_bytes);
    uint32_t unwritten = 0;
    for (size_t i = 0; i < out_words; i++) if (((uint32_t *)L->out)[i] == AT_POISON) unwritten++;
    if (info) {
        info->chip_calls++;
        info->elapsed_ns += elapsed;
        info->completion_marker = marker;
        info->unwritten_words += unwritten;
        info->threads_per_cta = ks->k.hd;
        info->ctas_last_launch = L->grid_x * L->grid_y;
    }
    return unwritten ? OMEGA_GPU_ATTN_UNWRITTEN : OMEGA_GPU_ATTN_OK;
}

/* Caller holds the lock. Accounts the launch's q / table / output traffic (the KV
 * staging bytes are accounted where the slices are copied). */
static int launch(const Slot *ks, const Launch *L, OmegaGpuAttnInfo *info) {
    if (info) {
        info->q_bytes += L->q_bytes; info->tab_bytes += L->tab_bytes; info->out_bytes += L->out_bytes;
        info->kv_source = L->pool_va ? OMEGA_GPU_ATTN_KV_RESIDENT : OMEGA_GPU_ATTN_KV_STAGED;
    }
    if (g_sim) { g_sim_shared_bytes = shared_bytes_for(ks->k.hd, ks->k.log2gqa); return g_sim(ks->prog, L, info); }
    return run_launch(ks, L, info);
}

/* ---- opt-in serving reservation (see the header) ---- */

static int check_heads(uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t *log2gqa);
static int log2_exact(uint32_t v, uint32_t *out);

int omega_gpu_attention_reserve(uint32_t max_context, uint32_t max_seqs, uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t kv_block_size) {
    uint32_t log2gqa, log2bs;
    if (max_context == 0 || max_seqs == 0) { omega_gpu_session_set_error("attention reserve: zero bound"); return OMEGA_GPU_ATTN_BAD_ARGS; }
    if (kv_block_size && log2_exact(kv_block_size, &log2bs) != 0) { omega_gpu_session_set_error("attention reserve: block size not a power of two"); return OMEGA_GPU_ATTN_BAD_ARGS; }
    int rc = check_heads(num_q_heads, num_kv_heads, head_dim, &log2gqa);
    if (rc != OMEGA_GPU_ATTN_OK) { omega_gpu_session_set_error("attention reserve: bad head shape"); return rc; }
    /* KV staging, worst case over the two paths, per sequence: contiguous f32 holds both planes for max_context tokens
     * (2 * ctx * kvh * hd * 4); paged bf16 stages whole blocks of both planes (round_up(ctx, block) * kvh * hd * 4), which
     * is the larger one when max_context is under half a block. */
    size_t per_seq = 2u * max_context * (size_t)num_kv_heads * head_dim * 4u;
    if (kv_block_size) {
        size_t paged = ((size_t)max_context + kv_block_size - 1) / kv_block_size * kv_block_size * num_kv_heads * head_dim * 4u;
        if (paged > per_seq) per_seq = paged;
    }
    const size_t pool = per_seq * max_seqs;
    /* q rows for every sequence, and the output buffer sized for the paged path's partials of one launch
     * (cut J: up to 2^MAX_LOG2SPLIT partials of head_dim + 2 words per (sequence, head), at most
     * MAX_CTAS / num_q_heads sequences per launch; the rows reserve usually asks for more). */
    const size_t q_bytes = (size_t)max_seqs * num_q_heads * head_dim * 4u;
    size_t launch_seqs = OMEGA_GPU_ATTN_MAX_CTAS / num_q_heads;
    if (launch_seqs == 0) launch_seqs = 1;
    if (launch_seqs > max_seqs) launch_seqs = max_seqs;
    const size_t partials = launch_seqs * num_q_heads * ((size_t)1u << OMEGA_GPU_ATTN_MAX_LOG2SPLIT) * ((size_t)head_dim + 2u) * 4u;
    const size_t qo = q_bytes > partials ? q_bytes : partials;
    const size_t tab = ((size_t)max_seqs * max_context + max_seqs) * 4u;
    LOCK();
    rc = OMEGA_GPU_ATTN_OK;
    if (g_sim) rc = OMEGA_GPU_ATTN_OK; /* the simulator stages in host memory: nothing to reserve */
    else if (dev_open_locked() != OMEGA_GPU_ATTN_OK) rc = OMEGA_GPU_ATTN_CHIP_FAIL;
    else if (omega_gpu_session_scratch_reserve(&g_pool, pool) || omega_gpu_session_scratch_reserve(&g_q, q_bytes) ||
             omega_gpu_session_scratch_reserve(&g_out, qo) || omega_gpu_session_scratch_reserve(&g_tab, tab < 8 ? 8 : tab)) {
        /* roll back this API's own buffers: a failed reserve leaves nothing flagged (growth allowed as before) */
        omega_gpu_session_scratch_unreserve(&g_pool); omega_gpu_session_scratch_unreserve(&g_q);
        omega_gpu_session_scratch_unreserve(&g_out); omega_gpu_session_scratch_unreserve(&g_tab);
        rc = OMEGA_GPU_ATTN_CHIP_FAIL;
    }
    UNLOCK();
    return rc;
}

int omega_gpu_attention_reserve_rows(uint32_t max_rows, uint32_t num_q_heads, uint32_t head_dim) {
    if (max_rows == 0 || num_q_heads == 0 || num_q_heads > OMEGA_GPU_ATTN_MAX_Q_HEADS || (head_dim != 64u && head_dim != 128u)) {
        omega_gpu_session_set_error("attention reserve rows: bad bound");
        return OMEGA_GPU_ATTN_BAD_ARGS;
    }
    uint32_t per = OMEGA_GPU_ATTN_ROWS_MAX_CTAS / num_q_heads;
    if (per == 0) per = 1;
    if (per > max_rows) per = max_rows;
    const size_t qo = (size_t)per * num_q_heads * head_dim * 4u, tab = (size_t)per * 8u;
    LOCK();
    int rc = OMEGA_GPU_ATTN_OK;
    if (g_sim) rc = OMEGA_GPU_ATTN_OK;
    else if (dev_open_locked() != OMEGA_GPU_ATTN_OK) rc = OMEGA_GPU_ATTN_CHIP_FAIL;
    else if ((qo > g_q.cap && omega_gpu_session_scratch_reserve(&g_q, qo)) ||
             (qo > g_out.cap && omega_gpu_session_scratch_reserve(&g_out, qo)) ||
             (tab > g_tab.cap && omega_gpu_session_scratch_reserve(&g_tab, tab)))
        rc = OMEGA_GPU_ATTN_CHIP_FAIL; /* growth failed; whatever grew stays reserved at its larger size */
    UNLOCK();
    return rc;
}
void omega_gpu_attention_unreserve(void) {
    LOCK();
    omega_gpu_session_scratch_unreserve(&g_pool); omega_gpu_session_scratch_unreserve(&g_q);
    omega_gpu_session_scratch_unreserve(&g_out); omega_gpu_session_scratch_unreserve(&g_tab);
    UNLOCK();
}

/* ------------------------------------------------------- public entries */

static void info_begin(OmegaGpuAttnInfo *info, const OmegaBlackwellKernel *k, bool hit) {
    if (!info) return;
    memset(info, 0, sizeof *info);
    info->kernel_cache_hit = hit;
    info->gpr_count = k->gpr_count;
    info->insn_count = (uint32_t)k->insn_count;
    omega_gpu_fmt(info->target_chip, sizeof info->target_chip, "NVIDIA DGX Spark (Grace Blackwell GB10)");
    info->sm_architecture = 121;
}
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static int log2_exact(uint32_t v, uint32_t *out) {
    if (v == 0 || (v & (v - 1))) return -1;
    uint32_t l = 0; while ((1u << l) != v) l++;
    *out = l; return 0;
}
/* log2(e) / sqrt(head_dim) with the square roots as literals (8 and 11.313708498984761 = sqrt(128), both exactly what sqrt() returns):
 * the library must not need libm (omegatool links without -lm; a constant head_dim folded sqrt away, a runtime one does not). */
static float score_scale(uint32_t hd) { return (float)(1.4426950408889634 / (hd == 128u ? 11.313708498984761 : 8.0)); }

/* Common head-shape checks; returns log2 of the gqa ratio. */
static int check_heads(uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t *log2gqa) {
    if (num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0) return OMEGA_GPU_ATTN_BAD_ARGS;
    if ((head_dim != 64u && head_dim != 128u) || num_q_heads > OMEGA_GPU_ATTN_MAX_Q_HEADS) return OMEGA_GPU_ATTN_TOO_LARGE;
    if (num_q_heads % num_kv_heads != 0 || log2_exact(num_q_heads / num_kv_heads, log2gqa) != 0) return OMEGA_GPU_ATTN_BAD_ARGS;
    return OMEGA_GPU_ATTN_OK;
}

/* Common prologue of every chip entry (lock held on success): device, kernel, info. */
static int begin(bool f32, uint32_t log2bs, uint32_t log2gqa, uint32_t hd, int split, Slot **ks, OmegaGpuAttnInfo *info) {
    LOCK();
    int rc = g_sim ? OMEGA_GPU_ATTN_OK : dev_open_locked();
    bool hit = false;
    if (rc == OMEGA_GPU_ATTN_OK) rc = kernel_for(f32, log2bs, log2gqa, hd, split, !g_sim, ks, &hit);
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
    const uint32_t HD = head_dim;
    size_t n_out = (size_t)num_q_heads * HD;
    if (seq_len == 0) { memset(out, 0, n_out * 4); if (info) { memset(info, 0, sizeof *info); info->call_ns = now_ns() - t0; } return OMEGA_GPU_ATTN_OK; }
    Slot *ks = NULL;
    const uint32_t log2bs = 12; /* one block of 4096 slots */
    rc = begin(true, log2bs, log2gqa, HD, -1, &ks, info);
    if (rc != OMEGA_GPU_ATTN_OK) return rc;
    size_t plane = (size_t)seq_len * num_kv_heads * HD * 4;
    void *host_tmp = NULL;
    int stage_rc;
    uint8_t *pool = stage_target(2 * plane, &host_tmp, &stage_rc);
    if (!pool) { UNLOCK(); return stage_rc; }
    memcpy(pool, k_cache, plane);
    memcpy(pool + plane, v_cache, plane);
    if (info) { info->kv_bytes_staged += 2 * plane; info->kv_bytes_naive += 2 * plane; info->kv_blocks_logical += 1; info->kv_blocks_unique += 1; } /* one contiguous pseudo-block */
    uint32_t tab[2] = { 0, seq_len }; /* table row {0}, then the context length at byte 4 */
    Launch L = {
        .q = q, .q_bytes = n_out * 4, .pool = pool, .pool_bytes = 2 * plane, .tab = tab, .tab_bytes = sizeof tab,
        .out = out, .out_bytes = n_out * 4, .grid_x = num_q_heads / heads_per_cta(HD, log2gqa), .grid_y = 1,
        .params = { 1, f2u(score_scale(HD)), (uint32_t)(2 * plane), num_kv_heads * HD * 4, (uint32_t)plane, HD * 4, num_q_heads, 4 }
    };
    rc = launch(ks, &L, info);
    omega_gpu_host_free(host_tmp);
    UNLOCK();
    if (info) info->call_ns = now_ns() - t0;
    return rc;
}

/* Rows per launch of the rows path: the CTA bound, then (under a reservation) what the q buffer holds. Caller holds the lock. */
static uint32_t rows_per_launch(uint32_t num_rows, uint32_t num_q_heads, size_t row_bytes) {
    uint32_t per = OMEGA_GPU_ATTN_ROWS_MAX_CTAS / num_q_heads;
    if (per == 0) per = 1;
    if (per > num_rows) per = num_rows;
    if (!g_sim && g_q.reserved && g_q.cap >= row_bytes) {
        size_t fit = g_q.cap / row_bytes;
        if (fit < per) per = (uint32_t)fit;
    }
    return per;
}

int omega_gpu_gqa_attention_rows_f32(const float *q, const float *k_cache, const float *v_cache,
                                     uint32_t base_len, uint32_t num_rows, uint32_t num_q_heads,
                                     uint32_t num_kv_heads, uint32_t head_dim, float *out, OmegaGpuAttnInfo *info) {
    uint64_t t0 = now_ns();
    if (!q || !k_cache || !v_cache || !out || num_rows == 0) return OMEGA_GPU_ATTN_BAD_ARGS;
    uint32_t log2gqa;
    int rc = check_heads(num_q_heads, num_kv_heads, head_dim, &log2gqa);
    if (rc != OMEGA_GPU_ATTN_OK) return rc;
    if ((uint64_t)base_len + num_rows > OMEGA_GPU_ATTN_MAX_GQA_CTX) return OMEGA_GPU_ATTN_TOO_LARGE;
    const uint32_t HD = head_dim, total = base_len + num_rows;
    const size_t row = (size_t)num_q_heads * HD;
    Slot *ks = NULL;
    const uint32_t log2bs = 12; /* one block of 4096 slots, as the single-row call */
    rc = begin(true, log2bs, log2gqa, HD, -1, &ks, info);
    if (rc != OMEGA_GPU_ATTN_OK) return rc;
    /* the whole context (every row's keys are a prefix of it) staged once */
    size_t plane = (size_t)total * num_kv_heads * HD * 4;
    void *host_tmp = NULL;
    int stage_rc;
    uint8_t *pool = stage_target(2 * plane, &host_tmp, &stage_rc);
    if (!pool) { UNLOCK(); return stage_rc; }
    memcpy(pool, k_cache, plane);
    memcpy(pool + plane, v_cache, plane);
    if (info) { info->kv_bytes_staged += 2 * plane; info->kv_bytes_naive += (uint64_t)num_rows * 2 * plane; info->kv_blocks_logical += num_rows; info->kv_blocks_unique += 1; }
    const uint32_t per = rows_per_launch(num_rows, num_q_heads, row * 4);
    /* table of one launch: per block ids (all 0: the one staged block), then per context lengths */
    uint32_t *tab = omega_gpu_host_calloc(2 * (size_t)per, sizeof *tab);
    if (!tab) { omega_gpu_host_free(host_tmp); UNLOCK(); return OMEGA_GPU_ATTN_CHIP_FAIL; }
    for (uint32_t r0 = 0; r0 < num_rows && rc == OMEGA_GPU_ATTN_OK; r0 += per) {
        const uint32_t nr = num_rows - r0 < per ? num_rows - r0 : per;
        for (uint32_t i = 0; i < nr; i++) { tab[i] = 0; tab[nr + i] = base_len + r0 + i + 1; }
        Launch L = {
            .q = q + (size_t)r0 * row, .q_bytes = (size_t)nr * row * 4, .pool = pool, .pool_bytes = 2 * plane,
            .tab = tab, .tab_bytes = (size_t)2 * nr * 4, .out = out + (size_t)r0 * row, .out_bytes = (size_t)nr * row * 4,
            .grid_x = num_q_heads / heads_per_cta(HD, log2gqa), .grid_y = nr,
            .params = { 1, f2u(score_scale(HD)), (uint32_t)(2 * plane), num_kv_heads * HD * 4, (uint32_t)plane, HD * 4, num_q_heads, nr * 4 }
        };
        rc = launch(ks, &L, info);
    }
    omega_gpu_host_free(tab);
    omega_gpu_host_free(host_tmp);
    UNLOCK();
    if (info) info->call_ns = now_ns() - t0;
    return rc;
}

/* Paged path shared by the single and batched entries. tables[s] has n_blocks[s] entries
 * (already filtered of negatives), ctx[s] the context length. */
/* cut F: a sequence's own split count from its context alone (the kernel computes the same): log2 of the largest
 * power of two <= min(2^MAX_LOG2SPLIT, chunks), chunks = ceil(ctx / HD) (cut J: one whole chunk per split; cut F took two). */
static uint32_t own_log2split(uint32_t ctx, uint32_t HD) {
    uint32_t x = (ctx + HD - 1) / HD, l = 0;
    while (l < OMEGA_GPU_ATTN_MAX_LOG2SPLIT && (2u << l) <= x) l++;
    return l;
}

/* 2^x in float without the math library (the engine archive also links where there is none): round x to the
 * nearest integer n, 2^f on f in [-0.5, 0.5] by the Cephes exp2f polynomial, scale by 2^n through the exponent
 * bits. x <= -127 (so also -inf, an empty split) is 0. The combine only passes x <= 0. */
static float host_exp2(float x) {
    if (x <= -127.0f) return 0.0f;
    if (x > 127.0f) x = 127.0f;
    int n = (int)(x + (x < 0.0f ? -0.5f : 0.5f));
    const float f = x - (float)n;
    float px = 1.535336188319500e-4f;
    px = px * f + 1.339887440266574e-3f;
    px = px * f + 9.618437357674640e-3f;
    px = px * f + 5.550332471162809e-2f;
    px = px * f + 2.402264791363012e-1f;
    px = px * f + 6.931472028550421e-1f;
    px = 1.0f + f * px;
    if (n < -126) { px *= 0.5f; n = -126; }
    union { uint32_t u; float f; } sc;
    sc.u = (uint32_t)(n + 127) << 23;
    return px * sc.f;
}

/* cut F: fold the per-split partials of each (sequence, head) row into out. w_s = 2^(M_s - max M): the kernel's
 * running max and sum live in the log2 domain (scale carries log2 e, MUFU.EX2). A row whose every split is empty
 * (context 0) is 0. */
static void combine_splits(const float *part, float *out, size_t rows, uint32_t S, uint32_t HD) {
    const size_t pw = (size_t)HD + 2;
    for (size_t r = 0; r < rows; r++) {
        const float *p = part + r * S * pw;
        float *o = out + r * HD;
        float mx = -__builtin_inff();
        for (uint32_t s = 0; s < S; s++) if (p[s * pw + HD] > mx) mx = p[s * pw + HD];
        if (!(mx > -__builtin_inff())) { memset(o, 0, (size_t)HD * 4); continue; }
        float w[1u << OMEGA_GPU_ATTN_MAX_LOG2SPLIT], lsum = 0.0f;
        for (uint32_t s = 0; s < S; s++) { w[s] = host_exp2(p[s * pw + HD] - mx); lsum += w[s] * p[s * pw + HD + 1]; }
        const float inv = 1.0f / lsum;
        for (uint32_t d = 0; d < HD; d++) {
            float a = 0.0f;
            for (uint32_t s = 0; s < S; s++) a += w[s] * p[s * pw + d];
            o[d] = a * inv;
        }
    }
}

static int paged_core(const float *q, const uint8_t *pool, const OmegaGpuKvLayout *ly,
                      const uint32_t *const *tables, const uint32_t *n_blocks, const uint32_t *ctx,
                      uint32_t num_seqs, uint32_t layer_idx, uint32_t num_q_heads, uint32_t num_kv_heads,
                      uint32_t head_dim, float *out, OmegaGpuAttnInfo *info) {
    uint64_t t0 = now_ns();
    uint32_t log2gqa, log2bs;
    int rc = check_heads(num_q_heads, num_kv_heads, head_dim, &log2gqa);
    if (rc != OMEGA_GPU_ATTN_OK) return rc;
    const uint32_t HD = head_dim;
    if (!ly || layer_idx >= ly->num_layers || log2_exact(ly->block_size, &log2bs) != 0) return OMEGA_GPU_ATTN_BAD_ARGS;
    /* the oracle addresses heads as kv_head * head_dim * 2 inside a token: the layout must agree */
    if (ly->head_stride_bytes != HD * 2 || ly->token_stride_bytes != (uint64_t)num_kv_heads * HD * 2 ||
        ly->kv_plane_stride_bytes != (uint64_t)ly->block_size * ly->token_stride_bytes ||
        ly->layer_stride_bytes != 2 * ly->kv_plane_stride_bytes || ly->block_stride_bytes < (uint64_t)ly->num_layers * ly->layer_stride_bytes)
        return OMEGA_GPU_ATTN_BAD_ARGS;
    size_t row = (size_t)num_q_heads * HD;
    /* effective context per sequence and block id range check */
    uint32_t *eff = omega_gpu_host_calloc(num_seqs ? num_seqs : 1, sizeof *eff);
    if (!eff) return OMEGA_GPU_ATTN_CHIP_FAIL;
    for (uint32_t s = 0; s < num_seqs; s++) {
        uint64_t cap = (uint64_t)n_blocks[s] * ly->block_size;
        eff[s] = (uint32_t)(ctx[s] < cap ? ctx[s] : cap);
        for (uint32_t b = 0; b < n_blocks[s]; b++) {
            uint64_t id = tables[s][b];
            if (id >= ly->num_blocks || id * ly->block_stride_bytes + (uint64_t)(layer_idx + 1) * ly->layer_stride_bytes > ly->pool_bytes) { omega_gpu_host_free(eff); return OMEGA_GPU_ATTN_BAD_ARGS; }
        }
    }
    uint32_t seqs_per_launch = OMEGA_GPU_ATTN_MAX_CTAS / num_q_heads;
    if (seqs_per_launch == 0) seqs_per_launch = 1;
    /* cut F: split each (sequence, head) over 2^log2split CTAs along the context: at least one whole chunk per
     * split, from the longest context in the call, within the split launch bound (the test hook forces it). */
    uint32_t log2split = 0;
    if (g_split >= 0) log2split = (uint32_t)g_split;
    else for (uint32_t s = 0; s < num_seqs; s++) { uint32_t l = own_log2split(eff[s], HD); if (l > log2split) log2split = l; }
    while (log2split && ((uint64_t)num_q_heads * seqs_per_launch << log2split) > OMEGA_GPU_ATTN_SPLIT_MAX_CTAS) log2split--;
    const uint32_t S = 1u << log2split;
    float *part = omega_gpu_host_alloc((size_t)seqs_per_launch * num_q_heads * S * (HD + 2) * 4); /* the partials of one launch */
    if (!part) { omega_gpu_host_free(eff); return OMEGA_GPU_ATTN_CHIP_FAIL; }
    Slot *ks = NULL;
    rc = begin(false, log2bs, log2gqa, HD, (int)log2split, &ks, info);
    if (rc != OMEGA_GPU_ATTN_OK) { omega_gpu_host_free(eff); omega_gpu_host_free(part); return rc; }
    uint32_t maxb = 1;
    for (uint32_t s = 0; s < num_seqs; s++) if (n_blocks[s] > maxb) maxb = n_blocks[s];
    /* Resident pool (2026-10-12): the caller's pool IS omega's allocation. Read it in place when the
     * layout fits the allocation and every referenced block passes the 32-bit fit rule; else stage. */
    bool resident = g_resident.cpu && (const void *)pool == g_resident.cpu && ly->pool_bytes <= g_resident.size;
    for (uint32_t s = 0; resident && s < num_seqs; s++) {
        uint32_t need = eff[s] ? (eff[s] + ly->block_size - 1) / ly->block_size : 0;
        for (uint32_t b = 0; b < need; b++) if (!resident_fits(ly, tables[s][b])) { resident = false; break; }
    }
    /* Dedupe map: physical block id -> staged slot of the current launch (-1 = not staged yet),
     * plus the list of ids touched by the launch (in slot order) so the reset is O(unique). */
    int32_t *slot_of = NULL; uint32_t *touched = NULL;
    if (g_dedupe && !resident) {
        size_t nb = ly->num_blocks ? ly->num_blocks : 1;
        slot_of = omega_gpu_host_alloc(nb * sizeof *slot_of);
        touched = omega_gpu_host_alloc(nb * sizeof *touched);
        if (!slot_of || !touched) { omega_gpu_host_free(slot_of); omega_gpu_host_free(touched); omega_gpu_host_free(eff); UNLOCK(); return OMEGA_GPU_ATTN_CHIP_FAIL; }
        memset(slot_of, 0xff, nb * sizeof *slot_of); /* every entry -1 */
    }
    for (uint32_t s0 = 0; s0 < num_seqs && rc == OMEGA_GPU_ATTN_OK; s0 += seqs_per_launch) {
        uint32_t ns = num_seqs - s0 < seqs_per_launch ? num_seqs - s0 : seqs_per_launch;
        /* logical references: every sequence's blocks up to its (truncated) context */
        uint32_t total = 0;
        for (uint32_t s = 0; s < ns; s++) total += eff[s0 + s] ? (eff[s0 + s] + ly->block_size - 1) / ly->block_size : 0;
        uint32_t *tab = omega_gpu_host_calloc((size_t)ns * maxb + ns, sizeof *tab);
        if (!tab) { rc = OMEGA_GPU_ATTN_CHIP_FAIL; break; }
        if (resident) {
            /* the table carries the physical ids; the kernel's block stride is the pool's; the layer
             * base is the chip address of this layer's slice of block 0 */
            for (uint32_t s = 0; s < ns; s++) {
                uint32_t need = eff[s0 + s] ? (eff[s0 + s] + ly->block_size - 1) / ly->block_size : 0;
                for (uint32_t b = 0; b < need; b++) tab[(size_t)s * maxb + b] = tables[s0 + s][b];
                tab[(size_t)ns * maxb + s] = eff[s0 + s];
            }
            if (info) { info->kv_blocks_logical += total; info->kv_bytes_naive += (uint64_t)total * ly->layer_stride_bytes; }
            Launch L = {
                .q = q + (size_t)s0 * row, .q_bytes = (size_t)ns * row * 4,
                .pool = (const uint8_t *)pool + (size_t)layer_idx * ly->layer_stride_bytes, .pool_bytes = 0,
                .tab = tab, .tab_bytes = ((size_t)ns * maxb + ns) * 4,
                .out = part, .out_bytes = (size_t)ns * num_q_heads * S * (HD + 2) * 4,
                .grid_x = num_q_heads / heads_per_cta(HD, log2gqa), .grid_y = ns * S,
                .params = { maxb, f2u(score_scale(HD)), (uint32_t)ly->block_stride_bytes, (uint32_t)ly->token_stride_bytes,
                            (uint32_t)ly->kv_plane_stride_bytes, (uint32_t)ly->head_stride_bytes, num_q_heads, ns * maxb * 4 },
                .pool_va = g_resident.va + (uint64_t)layer_idx * ly->layer_stride_bytes,
            };
            rc = launch(ks, &L, info);
            if (rc == OMEGA_GPU_ATTN_OK) combine_splits(part, out + (size_t)s0 * row, (size_t)ns * num_q_heads, S, HD);
            omega_gpu_host_free(tab);
            continue;
        }
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
        if (staged_bytes > 0xffffffffull) { omega_gpu_host_free(tab); rc = OMEGA_GPU_ATTN_TOO_LARGE; break; }
        void *host_tmp = NULL;
        int stage_rc;
        uint8_t *staged = stage_target((size_t)staged_bytes, &host_tmp, &stage_rc);
        if (!staged) { omega_gpu_host_free(tab); rc = stage_rc; break; }
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
            .tab = tab, .tab_bytes = ((size_t)ns * maxb + ns) * 4,
            .out = part, .out_bytes = (size_t)ns * num_q_heads * S * (HD + 2) * 4,
            .grid_x = num_q_heads / heads_per_cta(HD, log2gqa), .grid_y = ns * S,
            .params = { maxb, f2u(score_scale(HD)), (uint32_t)ly->layer_stride_bytes, (uint32_t)ly->token_stride_bytes,
                        (uint32_t)ly->kv_plane_stride_bytes, (uint32_t)ly->head_stride_bytes, num_q_heads, ns * maxb * 4 }
        };
        rc = launch(ks, &L, info);
        if (rc == OMEGA_GPU_ATTN_OK) combine_splits(part, out + (size_t)s0 * row, (size_t)ns * num_q_heads, S, HD);
        omega_gpu_host_free(host_tmp); omega_gpu_host_free(tab);
    }
    UNLOCK();
    omega_gpu_host_free(eff); omega_gpu_host_free(slot_of); omega_gpu_host_free(touched); omega_gpu_host_free(part);
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
    uint32_t **tables = omega_gpu_host_calloc(num_seqs, sizeof *tables);
    uint32_t *nb = omega_gpu_host_calloc(num_seqs, sizeof *nb), *cx = omega_gpu_host_calloc(num_seqs, sizeof *cx);
    if (!tables || !nb || !cx) { omega_gpu_host_free(tables); omega_gpu_host_free(nb); omega_gpu_host_free(cx); return OMEGA_GPU_ATTN_CHIP_FAIL; }
    int rc = OMEGA_GPU_ATTN_OK;
    for (uint32_t s = 0; s < num_seqs; s++) {
        cx[s] = context_lens[s] > 0 ? (uint32_t)context_lens[s] : 0;
        tables[s] = omega_gpu_host_calloc(max_blocks_per_seq ? max_blocks_per_seq : 1, sizeof **tables);
        if (!tables[s]) { rc = OMEGA_GPU_ATTN_CHIP_FAIL; break; }
        for (uint32_t b = 0; b < max_blocks_per_seq; b++) {
            int32_t v = block_tables[(size_t)s * max_blocks_per_seq + b];
            if (v >= 0) tables[s][nb[s]++] = (uint32_t)v;
        }
    }
    if (rc == OMEGA_GPU_ATTN_OK)
        rc = paged_core(q, pool, layout, (const uint32_t *const *)tables, nb, cx, num_seqs, layer_idx, num_q_heads, num_kv_heads, head_dim, out, info);
    for (uint32_t s = 0; s < num_seqs; s++) omega_gpu_host_free(tables[s]);
    omega_gpu_host_free(tables); omega_gpu_host_free(nb); omega_gpu_host_free(cx);
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

/* omega #308 R2 test hook: 0 production, 1 lane-dependent branch inside a reconvergence
 * region, 2 the same with the BSYNC dropped (negative control). Clears the kernel cache. */
void omega_gpu_attention_test_set_divergent(int mode) {
    LOCK();
    g_divergent = (mode < 0 || mode > 2) ? 0 : mode;
    cache_clear_locked();
    UNLOCK();
}
