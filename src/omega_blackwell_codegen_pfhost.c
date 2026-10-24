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

/* omega_blackwell_codegen_pfhost.c: router logits and MoE combine for the whole-prompt prefill (W-L1G-D). See the header. */
#include "omega_blackwell_codegen_pfhost.h"
#include "omega_bw_em.h"

static void fmul_s(Em *e, int d, unsigned dsub, int a, unsigned asub, int b, unsigned bsub) {
    BlackwellIRInsn i = I0(BW_IR_FMUL); i.dst_vreg = d; i.dst_subreg = (uint8_t)dsub; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; emit(e, i, K_FIXED);
}
static void fadd_s(Em *e, int d, unsigned dsub, int a, unsigned asub, int b, unsigned bsub) {
    BlackwellIRInsn i = I0(BW_IR_FADD); i.dst_vreg = d; i.dst_subreg = (uint8_t)dsub; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; emit(e, i, K_FIXED);
}
/* d[dsub] = a[asub] * b[bsub] + d[dsub] (the mutant's fused form) */
static void ffma_s(Em *e, int d, unsigned dsub, int a, unsigned asub, int b, unsigned bsub) {
    BlackwellIRInsn i = I0(BW_IR_FFMA); i.dst_vreg = d; i.dst_subreg = (uint8_t)dsub; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub;
    i.src3_vreg = d; i.src3_subreg = (uint8_t)dsub; emit(e, i, K_FIXED);
}
/* acc[asub] += a[asub2] * b[bsub] as FMUL then FADD (bit-identical to the host's a * b then +), or FFMA for mutant 1 */
static void mul_add(Em *e, int fused, int acc, unsigned accsub, int a, unsigned asub, int b, unsigned bsub, int prod) {
    if (fused) { ffma_s(e, acc, accsub, a, asub, b, bsub); return; }
    fmul_s(e, prod, 0, a, asub, b, bsub);
    fadd_s(e, acc, accsub, acc, accsub, prod, 0);
}

/* ======================= router logits ======================= */
int omega_rl_spec_ok(const OmegaRlSpec *s) {
    if (!s) return 0;
    if (s->k < 128u || s->k > 16384u || s->k % 128u) return 0;
    if (s->e < 32u || s->e > 1024u || s->e % 32u) return 0;
    if (s->t < 1u || s->t > 16u) return 0;
    if (s->t % (s->e / 32u)) return 0;  /* tile rows per pass = warps */
    if ((uint64_t)(s->k / 4u + 1u) * s->e * 16u > 0x7fffffffu) return 0;
    return s->mutant >= 0 && s->mutant <= 2;
}
uint32_t omega_rl_shared_bytes(const OmegaRlSpec *s) { return omega_rl_spec_ok(s) ? s->t * 512u : 0u; }

int omega_blackwell_codegen_rl_prog(const OmegaRlSpec *spec, BlackwellIRProgram *prog) {
    if (!omega_rl_spec_ok(spec) || !prog) return -1;
    const uint32_t K = spec->k, E = spec->e, T = spec->t, W = E / 32u, PASSES = T / W, GB = E * 16u;
    const uint32_t CHUNKS = spec->mutant == 2 ? K / 128u - 1u : K / 128u;
    const int fused = spec->mutant == 1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    int udesc = UV64(e), px = V64(e), pw = V64(e), pl = V64(e), tid = V(e), cta = V(e), warp = V(e), lane = V(e), one = V(e);
    int row = V(e), ssa = V(e), tmp = V(e), cntc = V(e), cnti = V(e), lsa = V(e), prod = V(e);
    int rA = V128(e), rB = V128(e);
    int acc[16], vq[4], g[16];
    for (uint32_t t = 0; t < T; t++) acc[t] = V(e);
    for (uint32_t i = 0; i < 4; i++) vq[i] = V128(e);
    for (uint32_t p = 0; p < PASSES; p++) g[p] = V128(e);
    if (e->err) return -1;
    if (CHUNKS == 0) return -1;

    ldcu64(e, udesc, 0x358); ldc64(e, px, 0x380); ldc64(e, pw, 0x388); ldc64(e, pl, 0x390);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cta, BW_SR_CTAID_X);
    movi(e, one, 1);
    shr(e, warp, tid, 0, 5);
    andi(e, lane, tid, 0, 31u);
    imadi(e, row, cta, 0, T, warp);                 /* first tile row this thread stages */
    imadw(e, px, row, K * 4u, px); imadw(e, px, lane, 16u, px);
    imadi(e, ssa, warp, 0, 512u, NOREG); imadi(e, ssa, lane, 0, 16u, ssa);
    imadw(e, pw, tid, 16u, pw);
    imadi(e, tmp, cta, 0, T * E, tid); imadw(e, pl, tmp, 4u, pl);
    for (uint32_t t = 0; t < T; t++) movi(e, acc[t], 0x80000000u); /* -0.0: Rust's f32 sum starts there */
    ldw_pf(e, BW_IR_LDG_E_128, rA, pw, 0u, udesc, 3u);  /* group 0 */
    movi(e, cntc, CHUNKS);

    #define COMPUTE(rv, immoff) do { \
        for (uint32_t b0 = 0; b0 < T; b0 += 4u) { \
            const uint32_t nb = T - b0 < 4u ? T - b0 : 4u; \
            for (uint32_t i = 0; i < nb; i++) lds128(e, vq[i], lsa, (b0 + i) * 512u + (immoff), i == 0); \
            for (uint32_t i = 0; i < nb; i++) for (unsigned q = 0; q < 4; q++) mul_add(e, fused, acc[b0 + i], 0, rv, q, vq[i], q, prod); \
        } } while (0)

    const int L_c = here(e);
    loop_begin(e);
    bar_sync(e); /* every reader of the previous tile is done */
    for (uint32_t p = 0; p < PASSES; p++) ldg_wide(e, BW_IR_LDG_E_128, g[p], px, p * W * K * 4u, udesc, p == 0);
    for (uint32_t p = 0; p < PASSES; p++) for (unsigned q = 0; q < 4; q++) {
        imadi(e, tmp, one, 0, p * W * 512u + 4u * q, ssa);
        BlackwellIRInsn st = I0(BW_IR_STS32); st.src1_vreg = tmp; st.src2_vreg = g[p]; st.src2_subreg = (uint8_t)q; emit(e, st, K_STORE);
    }
    bar_sync(e);
    imadw(e, px, one, 512u, px);
    movi(e, lsa, 0);
    movi(e, cnti, 16u); /* 128 k per chunk, 8 per iteration */
    const int L_i = here(e);
    loop_begin(e);
    consume(e, 3u);
    ldw_pf(e, BW_IR_LDG_E_128, rB, pw, GB, udesc, 3u);
    COMPUTE(rA, 0u);
    consume(e, 3u);
    ldw_pf(e, BW_IR_LDG_E_128, rA, pw, 2u * GB, udesc, 3u);
    COMPUTE(rB, 16u);
    imadw(e, pw, one, 2u * GB, pw);
    imadi(e, lsa, one, 0, 32u, lsa);
    imadi(e, cnti, one, 0, 0xffffffffu, cnti);
    isetp_ge_u32(e, cnti, one);
    bra_if_p0(e, L_i);
    loop_end(e);
    imadi(e, cntc, one, 0, 0xffffffffu, cntc);
    isetp_ge_u32(e, cntc, one);
    bra_if_p0(e, L_c);
    loop_end(e);
    #undef COMPUTE
    consume(e, 3u); /* the last prefetch (the pad group) */
    for (uint32_t t = 0; t < T; t++) stg(e, pl, t * E * 4u, acc[t], udesc);
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

/* ======================= MoE weighted combine ======================= */
int omega_mc_spec_ok(const OmegaMcSpec *s) {
    if (!s) return 0;
    if (s->h < 128u || s->h > 4096u || s->h % 128u) return 0;
    if (s->top_k < 4u || s->top_k > 16u || s->top_k % 4u) return 0;
    return s->mutant >= 0 && s->mutant <= 2;
}

int omega_blackwell_codegen_mc_prog(const OmegaMcSpec *spec, BlackwellIRProgram *prog) {
    if (!omega_mc_spec_ok(spec) || !prog) return -1;
    const uint32_t H = spec->h, KS = spec->top_k, NQ = KS / 4u;
    const uint32_t USED = spec->mutant == 2 ? KS - 1u : KS;
    const int fused = spec->mutant == 1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    int udesc = UV64(e), pd = V64(e), pi = V64(e), pw = V64(e), po = V64(e), tid = V(e), cta = V(e), off = V(e), prod = V(e), accq = V128(e);
    int idx[4], wt[4], v[16], pa[16];
    for (uint32_t q = 0; q < NQ; q++) { idx[q] = V128(e); wt[q] = V128(e); }
    for (uint32_t s = 0; s < KS; s++) { v[s] = V128(e); pa[s] = V64(e); }
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358); ldc64(e, pd, 0x380); ldc64(e, pi, 0x388); ldc64(e, pw, 0x390); ldc64(e, po, 0x398);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cta, BW_SR_CTAID_X);
    imadw(e, pi, cta, KS * 4u, pi); imadw(e, pw, cta, KS * 4u, pw);
    for (uint32_t q = 0; q < NQ; q++) ldg_wide(e, BW_IR_LDG_E_128, idx[q], pi, 16u * q, udesc, q == 0);
    for (uint32_t q = 0; q < NQ; q++) ldg_wide(e, BW_IR_LDG_E_128, wt[q], pw, 16u * q, udesc, 0);
    for (uint32_t s = 0; s < KS; s++) { imadi(e, off, idx[s / 4u], s % 4u, H / 4u, tid); imadw(e, pa[s], off, 16u, pd); }
    for (uint32_t s = 0; s < KS; s++) ldg_wide(e, BW_IR_LDG_E_128, v[s], pa[s], 0u, udesc, s == 0);
    for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn mz = I0(BW_IR_MOV_RZ); mz.dst_vreg = accq; mz.dst_subreg = (uint8_t)q; emit(e, mz, K_FIXED); }
    for (uint32_t s = 0; s < USED; s++) for (unsigned q = 0; q < 4; q++) mul_add(e, fused, accq, q, wt[s / 4u], s % 4u, v[s], q, prod);
    imadi(e, off, cta, 0, H / 4u, tid); imadw(e, po, off, 16u, po);
    for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn st = I0(BW_IR_STG_E); st.src1_vreg = po; st.src2_vreg = accq; st.src2_subreg = (uint8_t)q; st.imm = 4u * q; st.ureg = udesc; emit(e, st, K_STORE); }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}
