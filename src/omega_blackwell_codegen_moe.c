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

/* omega_blackwell_codegen_moe.c: FP8 e4m3 block-scaled GEMV kernel (OWN-30B Cut 6 / Cut 5).
 * See the header for the contract and the argument words.
 *
 * Scheduling: the same chip-learned rules as omega_gpu_elementwise_api.c (every instruction
 * waits on everything still pending; variable-latency results go through scoreboard 0, stores
 * set the read barrier 1, uniform loads scoreboard 2; fixed-latency producers stall 6; ISETP
 * stalls 14 before a predicated branch). Loads of one K step are issued back to back (all set
 * SB0, only the first waits), so a warp has every load of the step in flight at once.
 */
#include "omega_blackwell_codegen_moe.h"
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
/* d = a * b + c (registers) */
static void imadr(Em *e, int d, int a, int b, int c) { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = c; emit(e, i, K_FIXED); }
/* d64 = a * imm + c64 */
static void imadw(Em *e, int d64, int a, uint32_t imm, int c64) { BlackwellIRInsn i = I0(BW_IR_IMAD_WIDE); i.dst_vreg = d64; i.src1_vreg = a; i.imm = imm; i.src3_vreg = c64; emit(e, i, K_FIXED); }
static void shr(Em *e, int d, int a, unsigned asub, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_SHF_R); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.imm = imm; emit(e, i, K_FIXED); }
static void andi(Em *e, int d, int a, unsigned asub, uint32_t imm) { BlackwellIRInsn i = I0(BW_IR_LOP3_AND); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.imm = imm; emit(e, i, K_FIXED); }
static void xorr(Em *e, int d, int a, int b) { BlackwellIRInsn i = I0(BW_IR_LOP3_XOR); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED); }
static void ldg_wide(Em *e, BlackwellIROpcode op, int d, int addr64, uint32_t imm, int udesc, int first) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = addr64; i.imm = imm; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE);
}
static void ldg32(Em *e, int d, int addr64, int udesc, int first) { BlackwellIRInsn i = I0(BW_IR_LDG_E); i.dst_vreg = d; i.src1_vreg = addr64; i.ureg = udesc; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void stg(Em *e, int addr64, uint32_t imm, int val, int udesc) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = addr64; i.src2_vreg = val; i.imm = imm; i.ureg = udesc; emit(e, i, K_STORE); }
static void lds128(Em *e, int d128, int addr, uint32_t imm, int first) { BlackwellIRInsn i = I0(BW_IR_LDS128); i.dst_vreg = d128; i.src1_vreg = addr; i.imm = imm; emit(e, i, first ? K_VAR : K_VAR_MORE); }
static void lds32(Em *e, int d, int addr) { BlackwellIRInsn i = I0(BW_IR_LDS32); i.dst_vreg = d; i.src1_vreg = addr; emit(e, i, K_VAR); }
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

static int log2u(uint32_t v) { int l = 0; while ((1u << l) < v) l++; return (1u << l) == v ? l : -1; }

static int spec_ok(const OmegaFp8GemvSpec *s) {
    if (!s) return 0;
    if (s->lane_bytes != 8 && s->lane_bytes != 16) return 0; /* LDG.E (32-bit) carries no immediate offset: no 4-byte lanes */
    if (s->rows != 1 && s->rows != 2 && s->rows != 4 && s->rows != 8) return 0;
    if (s->warps < 1 || s->warps > 32) return 0; /* threads_x = 32 * warps <= 1024 */
    if (s->k == 0 || s->k > OMEGA_FP8_MAX_K || s->k % OMEGA_FP8_BLOCK || s->k % (32u * s->lane_bytes)) return 0;
    if (s->mutant < 0 || s->mutant > 5 || (s->mutant >= 4 && !s->coal)) return 0;
    if (s->sched > 1 || s->coal > 1 || s->wfirst > 1) return 0;
    if (s->coal && s->warps * s->rows > 32u) return 0; /* warp 0 stores the round: one row per lane */
    if (s->depth) { /* deep kernel: staging rounds (32-bit words, one per thread per round) must be exact and few */
        uint32_t ks = s->k / (32u * s->lane_bytes);
        if (s->depth > 16 || ks % s->depth || s->rows * s->depth * s->lane_bytes > 128u) return 0; /* <= 128 weight bytes per lane per iteration: instruction and register budgets */
        if (s->k > 512u * s->warps) return 0; /* at most 16 staging rounds; a partial last round is clamped */
    }
    /* row offsets of a step must fit the 24-bit signed LDG immediate */
    if ((uint64_t)(s->rows - 1) * s->k + s->lane_bytes >= (1u << 23)) return 0;
    return 1;
}

uint32_t omega_fp8_gemv_steps(const OmegaFp8GemvSpec *spec) { return spec_ok(spec) ? spec->k / (32u * spec->lane_bytes) : 0; }
/* deep: x' (k words) + one dummy slot for the clamped threads of a partial staging round; coal: then two output slot sets */
static uint32_t x_stage_bytes(const OmegaFp8GemvSpec *s) { return s->depth ? s->k * 4u + 16u : 0u; }
static uint32_t out_slot_bytes(const OmegaFp8GemvSpec *s) { return s->warps * s->rows * 4u; }
uint32_t omega_fp8_gemv_shared_bytes(const OmegaFp8GemvSpec *spec) {
    if (!spec_ok(spec)) return 0u;
    return x_stage_bytes(spec) + (spec->coal ? 2u * out_slot_bytes(spec) : 0u);
}

/* e4m3 byte i of word w -> f32 bits (value * 2^-120): ((b & 0x7F) << 20) | ((b & 0x80) << 24) */
static void decode_byte(Em *e, int f, int wq, unsigned word, unsigned i, int u, int a, int s) {
    const uint32_t M = 0x07F00000u;
    if (i < 3) imadi(e, u, wq, word, 1u << (20u - 8u * i), NOREG); /* b at bits 20..27 */
    else shr(e, u, wq, word, 4);
    andi(e, a, u, 0, M);
    andi(e, s, wq, word, 0x80u << (8u * i));
    if (i < 3) imadi(e, f, s, 0, 1u << (24u - 8u * i), a); /* sign to bit 31, plus the magnitude */
    else imadi(e, f, s, 0, 1u, a);
}

/* ---- sched 1 (CUT5): pipelined byte decode ----
 * Byte b of a K step (word b / 4, byte b % 4) goes through five instructions: E (shift the byte to bits 20..27),
 * S (isolate its sign), M (mask the magnitude), F (sign to bit 31 plus magnitude = the f32 bits) and C (the FFMA into
 * dot). Iteration c emits E(c), S(c), M(c - 2), F(c - 3), C(c - 5): every producer is at least 6 instructions ahead of
 * its user (E->M 12, S->F 12+, M->F 6, F->C 11), so the stall pass below can give most instructions stall 1 while the
 * FFMA chain (one per 5 instructions) runs at the 6-cycle fixed latency. t1 holds a byte from E to F (4 slots), t2 from
 * S to C (6 slots). The FFMA order (byte 0, 1, ... of word 0, then word 1 ...) and every operand are those of the
 * sched 0 loop, so y is bit-identical. */
#define PIPE_T1 4
#define PIPE_T2 6
static void decode_pipe(Em *e, int wqv, uint32_t words, const int *xq, int dot, const int *t1, const int *t2) {
    const uint32_t M = 0x07F00000u, NB = 4u * words;
    for (uint32_t c = 0; c < NB + 5u; c++) {
        if (c < NB) {
            const uint32_t word = c / 4u, i = c % 4u;
            const int u = t1[c % PIPE_T1], sg = t2[c % PIPE_T2];
            if (i < 3) imadi(e, u, wqv, word, 1u << (20u - 8u * i), NOREG); else shr(e, u, wqv, word, 4);
            andi(e, sg, wqv, word, 0x80u << (8u * i));
        }
        if (c >= 2 && c - 2u < NB) { const int u = t1[(c - 2u) % PIPE_T1]; andi(e, u, u, 0, M); }
        if (c >= 3 && c - 3u < NB) {
            const uint32_t b = c - 3u, i = b % 4u;
            const int sg = t2[b % PIPE_T2], u = t1[b % PIPE_T1];
            imadi(e, sg, sg, 0, i < 3 ? 1u << (24u - 8u * i) : 1u, u);
        }
        if (c >= 5 && c - 5u < NB) { const uint32_t b = c - 5u; ffma(e, dot, t2[b % PIPE_T2], xq[b / 4u], b % 4u, dot); }
    }
}

/* ---- sched 1: dependency-driven stall counts (after register allocation) ----
 * The emitter gives every fixed-latency instruction stall 6, which keeps any user >= 6 cycles behind its producer but
 * also makes independent instructions wait 6 cycles each. The repository's chip-proven rule is the distance, not the
 * per-instruction count: a fixed-latency producer stalls >= 5 cycles before its consumer (we keep 6;
 * omega_gpu_elementwise_api.c:13). Same pass as omega_blackwell_codegen_tern.c (chip-proven at TERN-T0 2300f164) with
 * this kernel's fixed-latency set, which adds FFMA, FADD and LOP3_XOR: walk the program in order on physical registers
 * and give each instruction the smallest stall that keeps the next instruction >= OMEGA_FP8_FIXED_LAT cycles behind
 * every fixed-latency write it reads or overwrites. Before a control instruction (BRA, EXIT, BSSY, BSYNC, BAR) or a
 * branch target every fixed-latency write has completed, so straight-line timing never spans a jump. Other instructions
 * keep at least their emitted stall (loads 4, ISETP 14 before a predicated branch); wait masks are untouched. */
static int fp8_fixed(BlackwellIROpcode op) {
    switch (op) {
    case BW_IR_IMAD: case BW_IR_IMAD_WIDE: case BW_IR_SHF_R: case BW_IR_LOP3_AND: case BW_IR_LOP3_XOR: case BW_IR_MOV_IMM:
    case BW_IR_MOV_RZ: case BW_IR_FFMA: case BW_IR_FADD: return 1;
    default: return 0;
    }
}
static int fp8_control(BlackwellIROpcode op) {
    return op == BW_IR_BRA || op == BW_IR_EXIT || op == BW_IR_BSSY || op == BW_IR_BSYNC || op == BW_IR_BAR_SYNC;
}
static void fp8_span(const BlackwellIRProgram *p, int v, int *lo, int *cnt) {
    *lo = -1; *cnt = 0;
    if (v < 0 || v >= p->regalloc.num_vregs) return;
    const int ph = p->regalloc.vreg_to_phys[v];
    if (ph < 0) return;
    *lo = ph; *cnt = (int)(p->regalloc.intervals[v].bundle_size ? p->regalloc.intervals[v].bundle_size : 1u);
}
static int fp8_schedule_stalls(BlackwellIRProgram *p) {
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
        const int fixed = fp8_fixed(in->op);
        if (fixed) {
            int lo, c; fp8_span(p, in->dst_vreg, &lo, &c);
            for (int k = 0; k < c; k++) { if (lo + k >= BW_MAX_VREGS + 8) return -1; ready[lo + k] = T + OMEGA_FP8_FIXED_LAT; }
            if (c && T + OMEGA_FP8_FIXED_LAT > all) all = T + OMEGA_FP8_FIXED_LAT;
        }
        uint64_t need = T + (fixed ? 1u : s_emit);
        if (i + 1 < n) {
            const BlackwellIRInsn *nx = &p->insns[i + 1];
            const int ops[4] = { nx->dst_vreg, nx->src1_vreg, nx->src2_vreg, nx->src3_vreg };
            for (int q = 0; q < 4; q++) {
                int lo, c; fp8_span(p, ops[q], &lo, &c);
                for (int k = 0; k < c; k++) if (lo + k < BW_MAX_VREGS + 8 && ready[lo + k] > need) need = ready[lo + k];
            }
            if ((fp8_control(nx->op) || target[i + 1]) && all > need) need = all;
        }
        const uint64_t s = need - T;
        if (s > 15 || (s == 0 && (fixed || s_emit != 0))) return -1; /* stall 0 only where emitted (the final self-branch) */
        if (fixed || s > s_emit) in->control = (in->control & ~(0xfu << 9)) | ((uint32_t)s << 9);
        T = need;
    }
    return 0;
}

int omega_blackwell_codegen_fp8_gemv_prog(const OmegaFp8GemvSpec *spec, BlackwellIRProgram *prog) {
    if (!spec_ok(spec) || !prog) return -1;
    const uint32_t K = spec->k, LB = spec->lane_bytes, R = spec->rows, W = spec->warps;
    const uint32_t KS = K / (32u * LB), BPS = 32u * LB / OMEGA_FP8_BLOCK, LPB = OMEGA_FP8_BLOCK / LB, KB = K / OMEGA_FP8_BLOCK;
    const int log2lpb = log2u(LPB), log2rows_per_block = log2u(OMEGA_FP8_BLOCK / R);
    const uint32_t WORDS = LB / 4u;
    const BlackwellIROpcode ldw = LB == 16 ? BW_IR_LDG_E_128 : BW_IR_LDG_E_64;
    if (log2lpb < 0 || log2rows_per_block < 0) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    Em *e = &em;

    /* ---- prologue ---- */
    int udesc = UV64(e), ptab = V64(e), px = V64(e), py = V64(e);
    int rgn = V(e), xs = V(e), ys = V(e), stride = V(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), one = V(e), gidx = V(e);
    int tent = V64(e), wbase = V64(e), sbase = V64(e), off = V(e), xrow = V64(e), yrow = V64(e), xlane = V64(e);
    int wlane = V(e), slane = V(e), rg = V(e);
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
    ldg_wide(e, BW_IR_LDG_E_64, wbase, tent, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_64, sbase, tent, 8, udesc, 0);
    imadr(e, off, cy, xs, NOREG); imadw(e, xrow, off, 1, px);
    imadr(e, off, cy, ys, NOREG); imadw(e, yrow, off, 1, py);
    imadw(e, xlane, lane, LB * 4u, xrow);           /* this lane's x' chunk of step 0 */
    imadi(e, wlane, lane, 0, LB, NOREG);            /* byte offset of this lane's chunk in a row */
    shr(e, slane, lane, 0, (uint32_t)log2lpb); imadi(e, slane, slane, 0, 4, NOREG); /* scale word of this lane in a step */
    imadi(e, rg, cx, 0, W, warp);                   /* first row group of this warp */

    const uint32_t D = spec->depth ? spec->depth : 1u; /* K steps per loop iteration */
    const int deep = spec->depth != 0;
    int pw = V64(e), ps = V64(e), pxc = V64(e), pyr = V64(e), pya = V64(e), srg = V(e), cnt = V(e);
    int acc[8], wq[8][16], xq[4], sc[16], psd[16], xl = -1, xl0 = -1;
    /* xq[word]: the 4 x floats of weight word `word` (one 16-byte quad each); wq[r][s]: row r, step s of the iteration */
    for (uint32_t r = 0; r < R; r++) { acc[r] = V(e); for (uint32_t s = 0; s < D; s++) wq[r][s] = LB == 16 ? V128(e) : V64(e); }
    for (uint32_t q = 0; q < WORDS; q++) xq[q] = V128(e);
    for (uint32_t s = 0; s < D; s++) { sc[s] = V(e); psd[s] = V64(e); }
    int dot = V(e), u = V(e), a = V(e), sg = V(e), f = V(e), t = V(e);
    int pt1[PIPE_T1], pt2[PIPE_T2]; /* sched 1 only: allocated after every sched 0 register, so sched 0 code is unchanged */
    if (spec->sched) { for (int q = 0; q < PIPE_T1; q++) pt1[q] = V(e); for (int q = 0; q < PIPE_T2; q++) pt2[q] = V(e); }
    /* coal 1 only, allocated last for the same reason: uc first row group of the CTA's round, rgx the clamped row group */
    const int coal = spec->coal != 0;
    int uc = NOREG, rgx = rg, um1 = NOREG, negU = NOREG, negn = NOREG, okw = NOREG, tq = NOREG, slb = NOREG, stog = NOREG;
    int so = NOREG, sa = NOREG, wl1 = NOREG, val = NOREG, row = NOREG, skip = NOREG;
    if (coal) {
        uc = V(e); rgx = V(e); um1 = V(e); negU = V(e); negn = V(e); okw = V(e); tq = V(e); slb = V(e); stog = V(e);
        so = V(e); sa = V(e); wl1 = V(e); val = V(e); row = V(e); skip = V(e);
    }
    if (e->err) return -1;

    if (deep) {
        /* ---- stage this group's x' row in shared memory (k floats), one 32-bit word per thread per round ---- */
        const uint32_t rounds = (K + 32u * W - 1u) / (32u * W);
        int pxt = V64(e), sa0 = V(e), sa = V(e), sap = V(e), pxs[16], xst[16], kq = V(e), over = V(e), tq = V(e), qs = V(e), qg = V(e);
        for (uint32_t j = 0; j < rounds; j++) { pxs[j] = V64(e); xst[j] = V(e); }
        xl = V(e); xl0 = V(e);
        if (e->err) return -1;
        imadw(e, pxt, tid, 4, xrow);
        imadi(e, sa0, tid, 0, 4, NOREG);
        for (uint32_t j = 0; j < rounds; j++) imadw(e, pxs[j], one, j * 128u * W, pxt);
        if (K % (32u * W)) {
            /* partial last round: word q = tid + j * 32 W may pass K. over = (q >= K) as 0/1 = (K - 1 - q) >> 31;
             * the load is clamped to x[K - 1] and the store to the dummy slot K (never read). */
            const uint32_t j = rounds - 1u;
            imadi(e, kq, tid, 0, 1, NOREG); imadi(e, kq, one, 0, j * 32u * W, kq);          /* q */
            imadi(e, tq, kq, 0, 0xffffffffu, NOREG); imadi(e, tq, one, 0, K - 1u, tq);     /* K - 1 - q */
            shr(e, over, tq, 0, 31);                                                        /* 1 iff q >= K */
            imadi(e, tq, kq, 0, 1, NOREG); imadi(e, tq, one, 0, (uint32_t)(0u - K), tq);   /* q - K */
            imadr(e, qs, over, tq, NOREG); imadi(e, qs, qs, 0, 0xffffffffu, kq);           /* q - over * (q - K): K when over */
            imadi(e, qg, tq, 0, 1, NOREG); imadi(e, qg, one, 0, 1, qg);                   /* q - K + 1 */
            imadr(e, qg, over, qg, NOREG); imadi(e, qg, qg, 0, 0xffffffffu, kq);           /* K - 1 when over */
            imadw(e, pxs[j], qg, 4, xrow);
            imadi(e, sap, qs, 0, 4, NOREG);
        }
        for (uint32_t j = 0; j < rounds; j++) ldg32(e, xst[j], pxs[j], udesc, j == 0);
        for (uint32_t j = 0; j < rounds; j++) {
            if (K % (32u * W) && j == rounds - 1u) { sts32(e, sap, xst[j]); continue; }
            imadi(e, sa, one, 0, j * 128u * W, sa0); sts32(e, sa, xst[j]);
        }
        bar_sync(e);
        imadi(e, xl0, lane, 0, LB * 4u, NOREG);     /* this lane's x' chunk of step 0, shared byte offset */
    }

    if (coal) {
        /* ---- coalesced output (variant A): per-CTA round variables. All values stay below 2^31 (RG, n <= 2^24),
         * so (a - b) >> 31 is the 0/1 test a < b. ---- */
        const uint32_t nrow = W * R;
        imadi(e, um1, one, 0, 0xffffffffu, rgn);              /* RG - 1 */
        imadi(e, negU, rgn, 0, 0xffffffffu, NOREG);           /* -RG */
        imadi(e, negn, rgn, 0, 0u - R, NOREG);                /* -n (n = RG * R) */
        movi(e, slb, x_stage_bytes(spec));                    /* slot set of this round (alternates) */
        /* stog = sum of the two set bases: slb = stog - slb alternates (XOR would need a base aligned to the set size) */
        movi(e, stog, 2u * x_stage_bytes(spec) + (spec->mutant == 4 ? 0u : out_slot_bytes(spec))); /* mutant 4: one slot set, the round race */
        imadi(e, so, warp, 0, R * 4u, NOREG);                 /* this warp's slot: warp * R words */
        imadi(e, tq, lane, 0, 0xffffffffu, NOREG); imadi(e, tq, one, 0, nrow - 1u, tq); shr(e, tq, tq, 0, 31); /* lane >= W R */
        imadi(e, wl1, warp, 0, 1, tq); imadi(e, wl1, one, 0, 1, wl1); /* warp + (lane >= W R) + 1 */
        imadi(e, uc, cx, 0, W, NOREG);                        /* first row group of this CTA's round (CTA-uniform) */
    }

    /* ---- row-group loop ---- */
    const int L_rg = here(e);
    loop_begin(e);
    isetp_ge_u32(e, coal ? uc : rg, rgn);                     /* coal: the whole CTA leaves together (barriers inside) */
    const int bra_end = bra_if_p0_fwd(e);
    if (coal) {
        /* a warp past the last row group recomputes row group RG - 1 (inside the tensor); its slot is never stored */
        imadi(e, tq, rg, 0, 1, negU);                         /* rg - RG */
        shr(e, okw, tq, 0, 31);                               /* 1 if rg < RG */
        imadi(e, tq, one, 0, 1, tq);                          /* rg - RG + 1 */
        if (spec->mutant == 5) imadi(e, rgx, rg, 0, 1, NOREG); /* mutant 5: no clamp, a warp past RG reads past the tensor */
        else imadr(e, rgx, okw, tq, um1);                     /* rg < RG ? rg : RG - 1 */
    }
    imadw(e, pw, rgx, R * K, wbase); imadw(e, pw, wlane, 1, pw);
    shr(e, srg, rgx, 0, (uint32_t)log2rows_per_block);
    imadw(e, ps, srg, KB * 4u, sbase); imadw(e, ps, slane, 1, ps);
    if (!deep) imadw(e, pxc, one, 0, xlane); else imadi(e, xl, xl0, 0, 1, NOREG);
    for (uint32_t r = 0; r < R; r++) movrz(e, acc[r]);
    movi(e, cnt, KS / D);

    /* ---- K loop: D steps per iteration, every load of the iteration issued before any use ---- */
    const int L_k = here(e);
    loop_begin(e);
    {
        int first = 1;
        const int xfirst = !deep && !spec->wfirst, xlast = !deep && spec->wfirst;
        if (xfirst) { for (uint32_t q = 0; q < WORDS; q++) { ldg_wide(e, BW_IR_LDG_E_128, xq[q], pxc, q * 16u, udesc, first); first = 0; } }
        for (uint32_t s = 0; s < D; s++) imadw(e, psd[s], one, spec->mutant == 2 ? 0u : s * BPS * 4u, ps);
        for (uint32_t s = 0; s < D; s++) for (uint32_t r = 0; r < R; r++) { ldg_wide(e, ldw, wq[r][s], pw, r * K + s * 32u * LB, udesc, first); first = 0; }
        for (uint32_t s = 0; s < D; s++) ldg32(e, sc[s], psd[s], udesc, 0);
        if (xlast) for (uint32_t q = 0; q < WORDS; q++) ldg_wide(e, BW_IR_LDG_E_128, xq[q], pxc, q * 16u, udesc, 0); /* variant B */
    }
    for (uint32_t s = 0; s < D; s++) {
        if (deep) { int first = 1; for (uint32_t q = 0; q < WORDS; q++) { lds128(e, xq[q], xl, s * 128u * LB + q * 16u, first); first = 0; } }
        for (uint32_t r = 0; r < R; r++) {
            movrz(e, dot);
            if (spec->sched) decode_pipe(e, wq[r][s], WORDS, xq, dot, pt1, pt2);
            else
            for (uint32_t word = 0; word < WORDS; word++)
                for (unsigned i = 0; i < 4; i++) {
                    decode_byte(e, f, wq[r][s], word, i, u, a, sg);
                    ffma(e, dot, f, xq[word], i, dot);
                }
            if (spec->mutant == 1) fadd(e, acc[r], acc[r], dot);
            else ffma(e, acc[r], dot, sc[s], 0, acc[r]);
        }
    }
    imadw(e, pw, one, D * 32u * LB, pw);
    if (!deep) imadw(e, pxc, one, 32u * LB * 4u, pxc);
    imadw(e, ps, one, spec->mutant == 2 ? 0u : D * BPS * 4u, ps);
    if (deep) imadi(e, xl, one, 0, D * 128u * LB, xl);
    imadi(e, cnt, one, 0, 0xffffffffu, cnt);
    isetp_ge_u32(e, cnt, one);
    bra_if_p0(e, L_k);
    loop_end(e);

    /* ---- warp reduction, lane 0 stores ---- */
    for (uint32_t r = 0; r < R; r++)
        for (uint32_t o = 16; o >= 1; o /= 2) { shfl_down(e, t, acc[r], o); fadd(e, acc[r], acc[r], t); }
    if (!coal) {
        imadw(e, pyr, rg, R * 4u, yrow);
        isetp_ge_u32(e, lane, one);
        const int at_bssy = bssy_fwd(e);
        const int bra_skip = bra_if_p0_fwd(e);
        for (uint32_t r = 0; r < R; r++) { imadw(e, pya, one, r * 4u, pyr); stg(e, pya, 0, acc[r], udesc); } /* STG.E with immediate 0 only (the chip-proven form) */
        const int at_bsync = here(e);
        bsync(e);
        patch_bra(e, bra_skip, at_bsync);
        patch_bra(e, at_bssy, at_bsync + 1);
    } else {
        /* lane 0 of each warp parks its R results in this round's slot set */
        isetp_ge_u32(e, lane, one);
        int at_bssy = bssy_fwd(e), bra_skip = bra_if_p0_fwd(e);
        imadi(e, sa, slb, 0, 1, so);                          /* slot set + warp * R words */
        for (uint32_t r = 0; r < R; r++) { if (r) imadi(e, sa, one, 0, 4u, sa); sts32(e, sa, acc[r]); }
        int at_bsync = here(e);
        bsync(e);
        patch_bra(e, bra_skip, at_bsync);
        patch_bra(e, at_bssy, at_bsync + 1);
        bar_sync(e);
        /* warp 0, lanes < W R: row = R uc + lane, stored when < n, one coalesced STG for the round */
        imadi(e, row, uc, 0, R, lane);
        imadi(e, okw, row, 0, 1, negn); shr(e, okw, okw, 0, 31); /* 1 if row < n */
        imadi(e, skip, okw, 0, 0xffffffffu, wl1);             /* warp + (lane >= W R) + 1 - (row < n): 0 only for a store */
        isetp_ge_u32(e, skip, one);
        at_bssy = bssy_fwd(e); bra_skip = bra_if_p0_fwd(e);
        imadi(e, sa, lane, 0, 4u, slb);
        lds32(e, val, sa);
        imadw(e, pya, row, 4u, yrow);
        stg(e, pya, 0, val, udesc);
        at_bsync = here(e);
        bsync(e);
        patch_bra(e, bra_skip, at_bsync);
        patch_bra(e, at_bssy, at_bsync + 1);
        imadi(e, slb, slb, 0, 0xffffffffu, stog);             /* next round parks in the other slot set: stog - slb */
        imadi(e, uc, stride, 0, 1, uc);
    }
    imadi(e, rg, stride, 0, 1, rg);
    bra(e, L_rg);
    loop_end(e);
    const int L_end = here(e);
    patch_bra(e, bra_end, L_end);
    tail(e);
    if (e->err) return -1;
    if (omega_bw_regalloc_solve(prog) != 0) return -1;
    return spec->sched ? fp8_schedule_stalls(prog) : 0;
}

int omega_fp8_gemv_prog_stats(const OmegaFp8GemvSpec *spec, uint32_t *insns, uint32_t *gprs) {
    static BlackwellIRProgram prog; /* 1024 insns: too big for a stack frame in a test loop */
    if (omega_blackwell_codegen_fp8_gemv_prog(spec, &prog) != 0) return -1;
    if (insns) *insns = (uint32_t)prog.count;
    if (gprs) *gprs = prog.regalloc.peak_gpr_usage;
    return 0;
}

#ifndef OMEGA_GPU_NO_POSIX
int omega_blackwell_codegen_fp8_gemv(const OmegaFp8GemvSpec *spec, OmegaBlackwellKernel *kernel) {
    if (!kernel) return -1;
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    if (!prog) return -1;
    int rc = omega_blackwell_codegen_fp8_gemv_prog(spec, prog);
    size_t need = rc == 0 ? omega_bw_program_code_bytes(prog) : 0;
    uint8_t *buf = need ? malloc(need) : NULL;
    if (!buf) { free(prog); return -1; }
    rc = omega_blackwell_kernel_encode_into(prog, buf, need, kernel);
    free(prog);
    if (rc != 0) { free(buf); return -1; }
    return 0; /* kernel->code points into buf; omega_blackwell_kernel_free releases it */
}
#endif
