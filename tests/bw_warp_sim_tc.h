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

/* bw_warp_sim.h: host SIMT simulator for the Blackwell IR (omega #308).
 *
 * Models what the earlier per-thread simulators could not: a warp of 32 lanes with an
 * active mask, divergence at predicated branches, structured reconvergence through the
 * barrier registers (BSSY arms Bn for the lanes executing it and names the join; BSYNC
 * parks arriving lanes until every armed lane that has not exited is there, then the
 * warp continues as one), nested regions, and lane participation in SHFL and BAR.SYNC.
 *
 * A warp is a set of fragments (pc, lane mask). A predicated branch whose lanes disagree
 * splits the fragment; the scheduler then runs one fragment at a time in a chosen order
 * (lowest pc first or highest pc first; tests run both), which is the freedom the chip's
 * scheduler also has. Fragments only merge at a BSYNC. Errors (all fatal, first one wins):
 *   WS_E_SHFL_SPLIT    a SHFL reached by a fragment that is not the whole live warp
 *                      (the chip reads inactive lanes: undefined; attention 2026-10-04)
 *   WS_E_SHFL_EXITED   a SHFL on a warp some of whose lanes already exited (undefined too)
 *   WS_E_BAR_SPLIT     (defined, never raised) a BAR.SYNC reached by a split warp is allowed:
 *                      since Volta the CTA barrier counts arriving threads, not warps (CUDA
 *                      programming guide, independent thread scheduling; INFERRED for sm_121,
 *                      not chip-verified by us). The simulator parks each fragment, releases
 *                      them together, counts the event in bar_split_arrivals, and leaves the
 *                      fragments split: a barrier is not a reconvergence point, so a SHFL after
 *                      it still fails with SHFL_SPLIT unless a BSYNC rejoined the warp.
 *   WS_E_BSYNC_UNARMED BSYNC on a barrier no BSSY armed
 *   WS_E_BSSY_REARM    BSSY on a barrier still armed (lanes may still be heading to its join)
 *   WS_E_JOIN_MISMATCH BSYNC at an address other than the one the BSSY named minus one
 *   WS_E_NONMEMBER     a lane reached BSYNC Bn without having executed its BSSY
 *   WS_E_STUCK         lanes parked at a BSYNC that can never complete (a join was dropped,
 *                      an exit jumped past it, or a target was corrupted)
 *   WS_E_NEVER_JOINED  the warp finished with a barrier still armed
 *   WS_E_SELF_BRANCH, WS_E_PC, WS_E_OP, WS_E_SHIFT, WS_E_MEM, WS_E_SMEM, WS_E_FRAGS
 * Every error names the pc. Memory: global addresses are host pointers (optionally range
 * checked); shared memory is a per-CTA buffer poisoned at CTA start so a read of a word
 * nobody wrote shows up in the oracle comparison.
 *
 * Per-lane ALU semantics are those of the two earlier simulators (tests/gpu_attention_test.c,
 * bench/prime_race/tests/gb10_sieve_host_test.c). EX2 and RCP are exact here; the chip's
 * are approximate. LDCU64 (memory descriptor) is not modelled. */
#ifndef BW_WARP_SIM_TC_H
#define BW_WARP_SIM_TC_H
/* bw_warp_sim_k.h (OWN-30B cut 8, 2026-10-13): tests/bw_warp_sim.h plus the system-scope opcodes the decode attention
 * kernel uses (ATOMG.ADD, LDG/STG.STRONG.SYS, MEMBAR): a derived copy so the 8B lane's file stays untouched.
 * The simulator runs CTAs one after another, so an atomic counter sees every earlier CTA's add. */
/* bw_warp_sim_tc.h (OWN-30B T3, 2026-10-13): bw_warp_sim_k.h plus (a) HMMA.16816.F32.BF16 as a warp-collective operation
 * computed with the GB10 arithmetic model that matched 3 x 262144 chip outputs bit for bit (tests/bw_hmma_model.h, evidence
 * OWN30B-T3-HMMA-MODEL), requiring the whole live warp (like SHFL), and (b) the STG.E immediate byte offset, which the
 * encoder emits and bw_warp_sim_k.h ignored. Timing (stall counts, scoreboards) is not modelled. */

#include "omega_blackwell_codegen.h"
#include "bw_hmma_model.h"
#include "omega_blackwell_codegen_tc.h"
#include "omega_blackwell_codegen_qmma.h"
#include "bw_qmma_model.h"
/* QMMA arithmetic used by the simulator (W-L1G): set by the test to the chip-measured model (gpu_qmma_probe_test) */
static QmmaModel ws_qmma_model = { 32, 26, 0, 2, 0 }; /* chip-exact 262144/262144, evidence OWN30B-L1G-QMMA-PFGEMM-c742aa13 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WS_WARP 32
#ifndef WS_MAX_THREADS
#define WS_MAX_THREADS 256
#endif
#define WS_MAX_WARPS (WS_MAX_THREADS / WS_WARP)
#define WS_MAX_FRAGS 64
#define WS_NBAR BW_RECONV_MAX_BAR
#define WS_MAX_RANGES 8

enum {
    WS_OK = 0, WS_E_SHFL_SPLIT, WS_E_SHFL_EXITED, WS_E_BAR_SPLIT, WS_E_BSYNC_UNARMED, WS_E_BSSY_REARM,
    WS_E_JOIN_MISMATCH, WS_E_NONMEMBER, WS_E_STUCK, WS_E_NEVER_JOINED, WS_E_SELF_BRANCH, WS_E_PC,
    WS_E_OP, WS_E_SHIFT, WS_E_MEM, WS_E_SMEM, WS_E_FRAGS, WS_E_BAR_TIMEOUT, WS_E_STEPS
};
static const char *ws_err_name(int e) {
    static const char *const n[] = { "OK", "SHFL_SPLIT", "SHFL_EXITED", "BAR_SPLIT", "BSYNC_UNARMED", "BSSY_REARM",
        "JOIN_MISMATCH", "NONMEMBER", "STUCK", "NEVER_JOINED", "SELF_BRANCH", "PC", "OP", "SHIFT", "MEM", "SMEM",
        "FRAGS", "BAR_TIMEOUT", "STEPS" };
    return (e >= 0 && e < (int)(sizeof n / sizeof n[0])) ? n[e] : "?";
}

typedef struct { uint32_t r[256]; uint8_t p0; } WsLane;
typedef struct { int pc; uint32_t mask; int parked; /* waiting at a BAR.SYNC */ } WsFrag;
typedef struct { int valid; int target; uint32_t members, arrived; } WsBar;
typedef struct {
    WsLane lane[WS_WARP];
    WsFrag frag[WS_MAX_FRAGS];
    int nfrag;
    WsBar bar[WS_NBAR];
    uint32_t exited;
} WsWarp;
typedef struct { uintptr_t lo, hi; } WsRange;

typedef struct {
    const BlackwellIRProgram *p;
    uint32_t cbank[256];      /* constant bank 0 words (byte offset / 4) */
    uint32_t threads;         /* per CTA, a multiple of 32, <= WS_MAX_THREADS */
    uint8_t *smem;            /* per-CTA shared memory buffer (poisoned at CTA start) */
    uint32_t smem_bytes;
    WsRange ranges[WS_MAX_RANGES]; int nranges; /* when > 0, every global access must fall in one */
    int order;                /* 0 = lowest pc first, 1 = highest pc first */
    int trace;                /* print every issue of warp 0 of CTA (0,0) */
    /* results */
    int err; int err_pc; char msg[200];
    uint64_t warp_issues;     /* fragment-instruction issues (the chip's cost unit for divergence) */
    uint64_t lane_issues;     /* lanes active over those issues */
    uint64_t splits;          /* branches that split a fragment */
    uint64_t bar_split_arrivals; /* BAR.SYNC reached by a fragment that is not the whole live warp */
    int max_frags, peak_bars;
} WsSim;

static void ws_fail(WsSim *s, int err, int pc, const char *what) {
    if (s->err) return;
    s->err = err; s->err_pc = pc;
    snprintf(s->msg, sizeof s->msg, "%s at pc %d: %s", ws_err_name(err), pc, what ? what : "");
}

static uint32_t ws_rd(const WsLane *l, int v, const BlackwellIRProgram *p) { if (v < 0) return 0; int ph = p->regalloc.vreg_to_phys[v]; return ph < 0 ? 0 : l->r[ph & 0xff]; }
static uint64_t ws_rd64(const WsLane *l, int v, const BlackwellIRProgram *p) { if (v < 0) return 0; int ph = p->regalloc.vreg_to_phys[v] & 0xff; return (uint64_t)l->r[ph] | ((uint64_t)l->r[(ph + 1) & 0xff] << 32); }
static void ws_wr(WsLane *l, int v, uint32_t x, const BlackwellIRProgram *p) { if (v < 0) return; int ph = p->regalloc.vreg_to_phys[v]; if (ph >= 0) l->r[ph & 0xff] = x; }
static void ws_wr64(WsLane *l, int v, uint64_t x, const BlackwellIRProgram *p) { if (v < 0) return; int ph = p->regalloc.vreg_to_phys[v] & 0xff; l->r[ph] = (uint32_t)x; l->r[(ph + 1) & 0xff] = (uint32_t)(x >> 32); }
/* sub-register operands (quad/pair lanes, BlackwellIRInsn.*_subreg) */
static uint32_t ws_rds(const WsLane *l, int v, unsigned sub, const BlackwellIRProgram *p) { if (v < 0) return 0; int ph = p->regalloc.vreg_to_phys[v]; return ph < 0 ? 0 : l->r[(ph + sub) & 0xff]; }
static void ws_wrs(WsLane *l, int v, unsigned sub, uint32_t x, const BlackwellIRProgram *p) { if (v < 0) return; int ph = p->regalloc.vreg_to_phys[v]; if (ph >= 0) l->r[(ph + sub) & 0xff] = x; }
static float ws_u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t ws_f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint32_t ws_bf16_rn(uint32_t u) { if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u; return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16; } /* f32 bits -> bf16 bits, round to nearest even */
/* MUFU (the SASS EX2 / RCP / RSQ unit) flushes a subnormal result to a sign-preserving zero. Chip-observed on GB10: the SwiGLU
 * reciprocal 1 / (2^t + 1) for t in (126, 128) came back as +-0 on all 101 of 101 differing elements (evidence
 * OWN30B-L1G-PFGEMM-SWIGLU-FTZ-df9bd9c4, gpu_pfgemm_test host red test). Doc-inferred, not separately chip-observed: the input flush
 * and the EX2 / RSQ cases follow the same rule, PTX ISA 9.7.3.13 (rcp) and 9.7.3.21 (ex2) "flushes subnormal inputs and results to
 * sign-preserving zero" for the .ftz form, which is what the bare MUFU we emit is (the non-ftz PTX variants are ptxas sequences around
 * MUFU; NVIDIA publishes no SASS semantics). Limit: the chip's MUFU is approximate (RCP 1 ulp, EX2 2 ulp), the model flushes a correctly
 * rounded IEEE result, so right at the subnormal boundary the two can disagree 0 versus nonzero. This simulator and the pfgemm reference
 * go through this helper; bw_warp_sim.h, bw_warp_sim_k.h, bw_warp_sim_pa.h and gpu_attention_test.c still model MUFU as IEEE (issue filed
 * with PR #461). omega_numeric.c has its own ftz() on the library side. */
static float ws_mufu_ftz(float f) { uint32_t u = ws_f2u(f); if ((u & 0x7f800000u) == 0u) u &= 0x80000000u; return ws_u2f(u); }
static uint32_t ws_lut3(uint32_t lut, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r = 0;
    for (int i = 0; i < 8; i++) if (lut >> i & 1) r |= ((i & 4) ? a : ~a) & ((i & 2) ? b : ~b) & ((i & 1) ? c : ~c);
    return r;
}
static int ws_mem_ok(WsSim *s, uint64_t a, unsigned bytes, int pc) {
    if (s->nranges == 0) return 1;
    for (int i = 0; i < s->nranges; i++) if (a >= s->ranges[i].lo && a + bytes <= s->ranges[i].hi) return 1;
    if (getenv("WS_DBG")) { fprintf(stderr, "bad addr %llx +%u pc %d\n", (unsigned long long)a, bytes, pc); for (int i = 0; i < s->nranges; i++) fprintf(stderr, "  range %d %llx..%llx\n", i, (unsigned long long)s->ranges[i].lo, (unsigned long long)s->ranges[i].hi); }
    ws_fail(s, WS_E_MEM, pc, "global access outside the declared buffers"); return 0;
}
static uint32_t *ws_smem(WsSim *s, uint32_t a, int pc) {
    if ((a & 3) || a + 4 > s->smem_bytes) { ws_fail(s, WS_E_SMEM, pc, "shared access past the declared size"); return NULL; }
    return (uint32_t *)(s->smem + a);
}
static int ws_popc(uint32_t m) { return __builtin_popcount(m); }

/* One lane, one non-control instruction. */
static void ws_lane_exec(WsSim *s, WsLane *l, const BlackwellIRInsn *in, int pc, uint32_t tid, uint32_t cx, uint32_t cy) {
    const BlackwellIRProgram *p = s->p;
    uint32_t a = ws_rds(l, in->src1_vreg, in->src1_subreg, p), b = ws_rds(l, in->src2_vreg, in->src2_subreg, p), c = ws_rds(l, in->src3_vreg, in->src3_subreg, p);
    switch (in->op) {
    case BW_IR_NOP: break;
    case BW_IR_S2R: ws_wrs(l, in->dst_vreg, in->dst_subreg, in->imm == BW_SR_TID_X ? tid : in->imm == BW_SR_CTAID_X ? cx : in->imm == BW_SR_CTAID_Y ? cy : 0, p); break;
    case BW_IR_LDC: ws_wrs(l, in->dst_vreg, in->dst_subreg, s->cbank[(in->imm / 4) & 0xff], p); break;
    case BW_IR_LDC64: ws_wr64(l, in->dst_vreg, (uint64_t)s->cbank[(in->imm / 4) & 0xff] | ((uint64_t)s->cbank[(in->imm / 4 + 1) & 0xff] << 32), p); break;
    case BW_IR_LDCU64: case BW_IR_LDCU: break; /* memory descriptor: not modelled */
    case BW_IR_MOV_IMM: ws_wrs(l, in->dst_vreg, in->dst_subreg, in->imm, p); break;
    case BW_IR_MOV_RZ: ws_wrs(l, in->dst_vreg, in->dst_subreg, 0, p); break;
    case BW_IR_IMAD: ws_wrs(l, in->dst_vreg, in->dst_subreg, in->src2_vreg >= 0 ? a * b + c : a * in->imm + c, p); break;
    case BW_IR_IMAD_WIDE: ws_wr64(l, in->dst_vreg, (uint64_t)a * in->imm + ws_rd64(l, in->src3_vreg, p), p); break;
    case BW_IR_IMAD_HI_U32: ws_wrs(l, in->dst_vreg, in->dst_subreg, (uint32_t)(((uint64_t)a * b) >> 32) + c, p); break;
    case BW_IR_IADD3: ws_wrs(l, in->dst_vreg, in->dst_subreg, a + b, p); break; /* the encoder's IADD3 has Rc = RZ */
    case BW_IR_SHF_R: ws_wrs(l, in->dst_vreg, in->dst_subreg, a >> (in->imm & 31), p); break;
    case BW_IR_SHF_L_U32: if (b >= 32) { ws_fail(s, WS_E_SHIFT, pc, "SHF.L.U32 amount >= 32 (not chip-verified)"); return; } ws_wrs(l, in->dst_vreg, in->dst_subreg, a << b, p); break;
    case BW_IR_LOP3_AND: ws_wrs(l, in->dst_vreg, in->dst_subreg, a & in->imm, p); break;
    case BW_IR_LOP3_XOR: ws_wrs(l, in->dst_vreg, in->dst_subreg, a ^ b, p); break;
    case BW_IR_LOP3_LUT: ws_wrs(l, in->dst_vreg, in->dst_subreg, ws_lut3(in->imm & 0xff, a, b, c), p); break;
    case BW_IR_ISETP_GE_U32: l->p0 = a >= b; break;
    case BW_IR_LDG_E: { uint64_t ad = ws_rd64(l, in->src1_vreg, p); if (!ws_mem_ok(s, ad, 4, pc)) return; uint32_t x; memcpy(&x, (const void *)(uintptr_t)ad, 4); ws_wrs(l, in->dst_vreg, in->dst_subreg, x, p); break; }
    case BW_IR_LDG_E_U16: { uint64_t ad = ws_rd64(l, in->src1_vreg, p); if (!ws_mem_ok(s, ad, 2, pc)) return; uint16_t x; memcpy(&x, (const void *)(uintptr_t)ad, 2); ws_wrs(l, in->dst_vreg, in->dst_subreg, x, p); break; }
    case BW_IR_STG_E: { uint64_t ad = ws_rd64(l, in->src1_vreg, p) + (uint64_t)(int64_t)(int32_t)in->imm; if (!ws_mem_ok(s, ad, 4, pc)) return; memcpy((void *)(uintptr_t)ad, &b, 4); break; }
    case BW_IR_LDG_E_64: case BW_IR_LDG_E_128: { unsigned n = in->op == BW_IR_LDG_E_64 ? 8u : 16u; uint64_t ad = ws_rd64(l, in->src1_vreg, p) + (uint64_t)(int64_t)(int32_t)in->imm; if (!ws_mem_ok(s, ad, n, pc)) { if (getenv("WS_DBG")) fprintf(stderr, "  tid %u cx %u cy %u\n", tid, cx, cy); return; } if (ad & (n - 1)) { ws_fail(s, WS_E_MEM, pc, "wide global load not naturally aligned"); return; } uint32_t x[4]; memcpy(x, (const void *)(uintptr_t)ad, n); for (unsigned i = 0; i < n / 4; i++) ws_wrs(l, in->dst_vreg, i, x[i], p); break; }
    case BW_IR_LDS128: { uint32_t ad = a + in->imm; if (ad & 15u) { ws_fail(s, WS_E_SMEM, pc, "LDS.128 address not 16-byte aligned"); return; } for (unsigned i = 0; i < 4; i++) { uint32_t *m = ws_smem(s, ad + 4u * i, pc); if (!m) return; ws_wrs(l, in->dst_vreg, i, *m, p); } break; }
    case BW_IR_LDS32: { uint32_t *m = ws_smem(s, a, pc); if (!m) return; ws_wrs(l, in->dst_vreg, in->dst_subreg, *m, p); break; }
    case BW_IR_STS32: { uint32_t *m = ws_smem(s, a, pc); if (!m) return; *m = b; break; }
    case BW_IR_FADD: ws_wrs(l, in->dst_vreg, in->dst_subreg, ws_f2u(ws_u2f(a) + ws_u2f(b)), p); break;
    case BW_IR_FSUB: ws_wrs(l, in->dst_vreg, in->dst_subreg, ws_f2u(ws_u2f(a) - ws_u2f(b)), p); break;
    case BW_IR_FMUL: ws_wrs(l, in->dst_vreg, in->dst_subreg, ws_f2u(ws_u2f(a) * ws_u2f(b)), p); break;
    case BW_IR_FFMA: ws_wrs(l, in->dst_vreg, in->dst_subreg, ws_f2u(fmaf(ws_u2f(a), ws_u2f(b), ws_u2f(c))), p); break;
    case BW_IR_FMNMX_MAX:
        if (in->imm == OMEGA_TC_MARK_F2FP) { ws_wrs(l, in->dst_vreg, in->dst_subreg, (ws_bf16_rn(a) << 16) | ws_bf16_rn(b), p); break; } /* F2FP.BF16.F32.PACK_AB placeholder */
        if (in->imm == OMEGA_QMMA_MARK_F2E4) { ws_wrs(l, in->dst_vreg, in->dst_subreg, (uint32_t)qm_e4m3_rne_sat(ws_u2f(a)) << 8 | qm_e4m3_rne_sat(ws_u2f(b)), p); break; } /* F2FP.SATFINITE.E4M3 placeholder (NaN not modelled) */
        ws_wrs(l, in->dst_vreg, in->dst_subreg, ws_f2u(fmaxf(ws_u2f(a), ws_u2f(b))), p); break;
    case BW_IR_FMNMX_MIN: ws_wrs(l, in->dst_vreg, in->dst_subreg, ws_f2u(fminf(ws_u2f(a), ws_u2f(b))), p); break;
    case BW_IR_MUFU_EX2: ws_wrs(l, in->dst_vreg, in->dst_subreg, ws_f2u(ws_mufu_ftz(exp2f(ws_mufu_ftz(ws_u2f(a))))), p); break;
    case BW_IR_MUFU_RCP: ws_wrs(l, in->dst_vreg, in->dst_subreg, ws_f2u(ws_mufu_ftz(1.0f / ws_mufu_ftz(ws_u2f(a)))), p); break;
    case BW_IR_MUFU_RSQ: ws_wrs(l, in->dst_vreg, in->dst_subreg, ws_f2u(ws_mufu_ftz(1.0f / sqrtf(ws_mufu_ftz(ws_u2f(a))))), p); break;
    case BW_IR_LDG_STRONG_SYS: { uint64_t ad = ws_rd64(l, in->src1_vreg, p); if (!ws_mem_ok(s, ad, 4, pc)) return; uint32_t x; memcpy(&x, (const void *)(uintptr_t)ad, 4); ws_wrs(l, in->dst_vreg, in->dst_subreg, x, p); break; }
    case BW_IR_STG_STRONG_SYS: { uint64_t ad = ws_rd64(l, in->src1_vreg, p); if (!ws_mem_ok(s, ad, 4, pc)) return; memcpy((void *)(uintptr_t)ad, &b, 4); break; }
    case BW_IR_ATOMG_ADD_STRONG_SYS: { uint64_t ad = ws_rd64(l, in->src1_vreg, p); if (!ws_mem_ok(s, ad, 4, pc)) return; uint32_t old; memcpy(&old, (const void *)(uintptr_t)ad, 4); uint32_t nw = old + b; memcpy((void *)(uintptr_t)ad, &nw, 4); ws_wrs(l, in->dst_vreg, in->dst_subreg, old, p); break; }
    case BW_IR_MEMBAR_ALL_SYS: break;
    default: ws_fail(s, WS_E_OP, pc, "opcode not modelled"); return;
    }
}

static int ws_pick(const WsSim *s, const WsWarp *w) {
    int best = -1;
    for (int i = 0; i < w->nfrag; i++) {
        if (w->frag[i].mask == 0 || w->frag[i].parked) continue;
        if (best < 0) { best = i; continue; }
        if (s->order == 0 ? w->frag[i].pc < w->frag[best].pc : w->frag[i].pc > w->frag[best].pc) best = i;
    }
    return best;
}
static void ws_drop_frag(WsWarp *w, int i) { w->frag[i] = w->frag[--w->nfrag]; }
static int ws_add_frag(WsSim *s, WsWarp *w, int pc, uint32_t mask) {
    if (mask == 0) return 0;
    if (w->nfrag >= WS_MAX_FRAGS) { ws_fail(s, WS_E_FRAGS, pc, "too many fragments"); return -1; }
    w->frag[w->nfrag].pc = pc; w->frag[w->nfrag].mask = mask; w->frag[w->nfrag].parked = 0; w->nfrag++;
    if (w->nfrag > s->max_frags) s->max_frags = w->nfrag;
    return 0;
}

/* Runs one warp until it has no runnable fragment (finished, parked at BAR.SYNC, or stuck).
 * Returns 1 when parked at BAR.SYNC, 0 when finished, -1 on error. */
static int ws_run_warp(WsSim *s, WsWarp *w, uint32_t warp, uint32_t cx, uint32_t cy, int trace) {
    const BlackwellIRProgram *p = s->p;
    const uint32_t live_all = 0xffffffffu;
    for (uint64_t guard = 0;; guard++) {
        if (s->err) return -1;
        if (guard > 2000000000ull) { ws_fail(s, WS_E_STEPS, -1, "step budget"); return -1; }
        int fi = ws_pick(s, w);
        if (fi < 0) {
            /* nothing runnable: parked at the CTA barrier, finished, or lanes waiting at a BSYNC
             * that can never complete */
            for (int i = 0; i < w->nfrag; i++) if (w->frag[i].parked) return 1;
            for (int b = 0; b < WS_NBAR; b++) if (w->bar[b].valid) {
                if (w->bar[b].arrived) { ws_fail(s, WS_E_STUCK, w->bar[b].target - 1, "lanes parked at BSYNC, the rest never arrive"); return -1; }
                ws_fail(s, WS_E_NEVER_JOINED, w->bar[b].target - 1, "warp finished with a barrier still armed"); return -1;
            }
            return 0;
        }
        WsFrag *f = &w->frag[fi];
        int pc = f->pc;
        if (pc < 0 || (size_t)pc >= p->count) { ws_fail(s, WS_E_PC, pc, "pc outside the program"); return -1; }
        const BlackwellIRInsn *in = &p->insns[pc];
        uint32_t mask = f->mask;
        s->warp_issues++; s->lane_issues += (uint64_t)ws_popc(mask);
        if (trace) printf("W%u pc=%d op=%d mask=%08x frags=%d\n", warp, pc, (int)in->op, mask, w->nfrag);
        uint32_t live = live_all & ~w->exited;
        uint32_t pmask = 0;
        for (int L = 0; L < WS_WARP; L++) if ((mask >> L & 1) && w->lane[L].p0) pmask |= 1u << L;
        uint32_t cond = in->predicate_p0 ? (in->predicate_not ? (mask & ~pmask) : (mask & pmask)) : mask;
        switch (in->op) {
        case BW_IR_EXIT: {
            w->exited |= cond;
            f->mask = mask & ~cond;
            if (f->mask == 0) ws_drop_frag(w, fi); else f->pc = pc + 1;
            break;
        }
        case BW_IR_BRA: {
            int32_t delta = (int32_t)in->imm;
            uint32_t taken = cond, fall = mask & ~cond;
            if (taken && delta == 0) { ws_fail(s, WS_E_SELF_BRANCH, pc, "branch to itself"); return -1; }
            if (taken && fall) s->splits++;
            if (taken == 0) { f->pc = pc + 1; break; }
            if (fall == 0) { f->pc = pc + delta; break; }
            f->mask = fall; f->pc = pc + 1;
            if (ws_add_frag(s, w, pc + delta, taken) < 0) return -1;
            break;
        }
        case BW_IR_BSSY: {
            WsBar *b = &w->bar[in->bar_reg & (WS_NBAR - 1)];
            if (in->bar_reg >= WS_NBAR) { ws_fail(s, WS_E_OP, pc, "barrier id"); return -1; }
            if (b->valid) { ws_fail(s, WS_E_BSSY_REARM, pc, "BSSY on an armed barrier"); return -1; }
            b->valid = 1; b->target = pc + (int32_t)in->imm; b->members = cond; b->arrived = 0;
            int armed = 0; for (int k = 0; k < WS_NBAR; k++) armed += w->bar[k].valid;
            if (armed > s->peak_bars) s->peak_bars = armed;
            f->pc = pc + 1;
            break;
        }
        case BW_IR_BSYNC: {
            if (in->bar_reg >= WS_NBAR) { ws_fail(s, WS_E_OP, pc, "barrier id"); return -1; }
            WsBar *b = &w->bar[in->bar_reg];
            if (!b->valid) { ws_fail(s, WS_E_BSYNC_UNARMED, pc, "BSYNC without a BSSY"); return -1; }
            if (cond & ~b->members) { ws_fail(s, WS_E_NONMEMBER, pc, "a lane at BSYNC did not execute the BSSY"); return -1; }
            b->arrived |= cond;
            uint32_t rest = mask & ~cond; /* a predicated-off lane walks on */
            if (rest) { f->mask = rest; f->pc = pc + 1; } else ws_drop_frag(w, fi);
            if (b->arrived == (b->members & ~w->exited)) {
                if (b->target != pc + 1) { ws_fail(s, WS_E_JOIN_MISMATCH, pc, "BSSY named a different join point"); return -1; }
                uint32_t joined = b->arrived; b->valid = 0; b->arrived = 0; b->members = 0;
                if (ws_add_frag(s, w, pc + 1, joined) < 0) return -1;
            }
            break;
        }
        case BW_IR_SHFL_DOWN: {
            if (w->exited) { ws_fail(s, WS_E_SHFL_EXITED, pc, "SHFL on a warp with exited lanes"); return -1; }
            if (mask != live) { ws_fail(s, WS_E_SHFL_SPLIT, pc, "SHFL on a split warp"); return -1; }
            uint32_t src[WS_WARP];
            for (int L = 0; L < WS_WARP; L++) src[L] = ws_rd(&w->lane[L], in->src1_vreg, p);
            for (int L = 0; L < WS_WARP; L++) {
                int from = (in->imm & OMEGA_TC_MARK_BFLY) ? (L ^ (int)(in->imm & 31)) : L + (int)(in->imm & 31); /* BFLY placeholder (omega_blackwell_codegen_tc.h) */
                ws_wr(&w->lane[L], in->dst_vreg, from < WS_WARP ? src[from] : src[L], p);
            }
            f->pc = pc + 1;
            break;
        }
        case BW_IR_HMMA_BF16: {
            if (w->exited) { ws_fail(s, WS_E_SHFL_EXITED, pc, "HMMA on a warp with exited lanes"); return -1; }
            if (mask != live) { ws_fail(s, WS_E_SHFL_SPLIT, pc, "HMMA on a split warp"); return -1; }
            if (in->imm == OMEGA_QMMA_MARK) { /* QMMA.16832.F32.E4M3.E4M3 (fragment layout: omega_blackwell_codegen_qmma.h) */
                static float QA[16][32], QB[32][8], QC[16][8];
                for (int L = 0; L < WS_WARP; L++) {
                    int gg = L / 4, tt = L % 4; const WsLane *ln = &w->lane[L];
                    for (unsigned q = 0; q < 4; q++) { uint32_t av = ws_rds(ln, in->src1_vreg, in->src1_subreg + q, p); int r = gg + (q & 1u ? 8 : 0), k0 = 4 * tt + (q & 2u ? 16 : 0);
                        for (int j = 0; j < 4; j++) QA[r][k0 + j] = qm_e4m3((uint8_t)(av >> (8 * j))); }
                    for (unsigned q = 0; q < 2; q++) { uint32_t bv = ws_rds(ln, in->src2_vreg, in->src2_subreg + q, p); int k0 = 4 * tt + 16 * (int)q;
                        for (int j = 0; j < 4; j++) QB[k0 + j][gg] = qm_e4m3((uint8_t)(bv >> (8 * j))); }
                    for (unsigned q = 0; q < 4; q++) QC[gg + (q >= 2 ? 8 : 0)][2 * tt + (int)(q & 1u)] = in->src3_vreg >= 0 ? ws_u2f(ws_rds(ln, in->src3_vreg, in->src3_subreg + q, p)) : 0.0f;
                }
                for (int L = 0; L < WS_WARP; L++) {
                    int gg = L / 4, tt = L % 4;
                    for (unsigned q = 0; q < 4; q++) { int rr = gg + (q >= 2 ? 8 : 0), nn = 2 * tt + (int)(q & 1u); float av[32], bv[32];
                        for (int k = 0; k < 32; k++) { av[k] = QA[rr][k]; bv[k] = QB[k][nn]; }
                        ws_wrs(&w->lane[L], in->dst_vreg, q, ws_f2u(qm_dot32(&ws_qmma_model, av, bv, QC[rr][nn])), p); }
                }
                f->pc = pc + 1;
                break;
            }
            static const HmmaModel gb10 = { 16, 2, 0, 2, 0 };
            float A[16][16], B[16][8], C[16][8];
            for (int L = 0; L < WS_WARP; L++) {
                int gg = L / 4, tt = L % 4;
                const WsLane *ln = &w->lane[L];
                uint32_t a4[4], b2[2], c4[4];
                for (unsigned q = 0; q < 4; q++) { a4[q] = ws_rds(ln, in->src1_vreg, in->src1_subreg + q, p); c4[q] = in->src3_vreg >= 0 ? ws_rds(ln, in->src3_vreg, in->src3_subreg + q, p) : 0u; }
                for (unsigned q = 0; q < 2; q++) b2[q] = ws_rds(ln, in->src2_vreg, in->src2_subreg + q, p); /* operands may start at a subregister (the encoder adds it) */
                const int ar[4] = { gg, gg + 8, gg, gg + 8 }, ac[4] = { 2 * tt, 2 * tt, 2 * tt + 8, 2 * tt + 8 };
                for (unsigned q = 0; q < 4; q++) { A[ar[q]][ac[q]] = ws_u2f(a4[q] << 16); A[ar[q]][ac[q] + 1] = ws_u2f(a4[q] & 0xffff0000u); }
                for (unsigned q = 0; q < 2; q++) { B[2 * tt + 8 * q][gg] = ws_u2f(b2[q] << 16); B[2 * tt + 8 * q + 1][gg] = ws_u2f(b2[q] & 0xffff0000u); }
                C[gg][2 * tt] = ws_u2f(c4[0]); C[gg][2 * tt + 1] = ws_u2f(c4[1]); C[gg + 8][2 * tt] = ws_u2f(c4[2]); C[gg + 8][2 * tt + 1] = ws_u2f(c4[3]);
            }
            for (int L = 0; L < WS_WARP; L++) {
                int gg = L / 4, tt = L % 4;
                for (unsigned q = 0; q < 4; q++) {
                    int rr = gg + (q >= 2 ? 8 : 0), nn = 2 * tt + (int)(q & 1u);
                    float av[16], bv[16]; for (int k = 0; k < 16; k++) { av[k] = A[rr][k]; bv[k] = B[k][nn]; }
                    ws_wrs(&w->lane[L], in->dst_vreg, q, ws_f2u(hm_dot16(&gb10, av, bv, C[rr][nn])), p);
                }
            }
            f->pc = pc + 1;
            break;
        }
        case BW_IR_BAR_SYNC: {
            if (mask != live) s->bar_split_arrivals++;
            f->parked = 1; /* released by ws_run when every live lane of the CTA has arrived */
            break;
        }
        default: {
            for (int L = 0; L < WS_WARP; L++) if (mask >> L & 1) {
                ws_lane_exec(s, &w->lane[L], in, pc, warp * WS_WARP + (uint32_t)L, cx, cy);
                if (s->err) return -1;
            }
            f->pc = pc + 1;
            break;
        }
        }
    }
}

/* Runs every CTA of the grid. Returns s->err (0 = clean). */
static int ws_run(WsSim *s, uint32_t grid_x, uint32_t grid_y) {
    s->err = 0; s->msg[0] = 0; s->warp_issues = s->lane_issues = s->splits = s->bar_split_arrivals = 0; s->max_frags = 0; s->peak_bars = 0;
    if (!s->p || s->threads == 0 || s->threads % WS_WARP || s->threads > WS_MAX_THREADS) { ws_fail(s, WS_E_OP, -1, "bad simulator configuration"); return s->err; }
    static WsWarp warps[WS_MAX_WARPS];
    uint32_t nw = s->threads / WS_WARP;
    for (uint32_t cy = 0; cy < grid_y && !s->err; cy++) for (uint32_t cx = 0; cx < grid_x && !s->err; cx++) {
        memset(warps, 0, sizeof(WsWarp) * nw);
        if (s->smem && s->smem_bytes) for (uint32_t i = 0; i + 4 <= s->smem_bytes; i += 4) { uint32_t poison = 0xfeedfaceu; memcpy(s->smem + i, &poison, 4); }
        for (uint32_t w = 0; w < nw; w++) { warps[w].nfrag = 1; warps[w].frag[0].pc = 0; warps[w].frag[0].mask = 0xffffffffu; warps[w].frag[0].parked = 0; }
        int trace = s->trace && cx == 0 && cy == 0;
        /* Barrier phases: every warp runs until all of its fragments are parked at BAR.SYNC or it
         * has finished (exited lanes do not hold the barrier up: INFERRED, see WS_E_BAR_SPLIT).
         * Then every parked fragment is released and the next phase starts. */
        for (int phase = 0; ; phase++) {
            int parked = 0;
            for (uint32_t w = 0; w < nw && !s->err; w++) {
                if (warps[w].nfrag == 0) continue;
                int r = ws_run_warp(s, &warps[w], w, cx, cy, trace && w == 0);
                if (r < 0) break;
                if (r == 1) parked++;
            }
            if (s->err || parked == 0) break;            /* error, or every warp finished */
            for (uint32_t w = 0; w < nw; w++) for (int i = 0; i < warps[w].nfrag; i++)
                if (warps[w].frag[i].parked) { warps[w].frag[i].parked = 0; warps[w].frag[i].pc++; }
            if (phase > 100000) { ws_fail(s, WS_E_BAR_TIMEOUT, -1, "barrier phases"); break; }
        }
    }
    return s->err;
}

#endif /* BW_WARP_SIM_TC_H */
