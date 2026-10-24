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

/* omega_blackwell_codegen_qg.c: native FP8 prefill kernels (W-L1G). See the header. */
#include "omega_blackwell_codegen_qg.h"
#include "omega_bw_em.h"
#include "omega_bw_em_q.h"

static void fmul_r(Em *e, int d, unsigned dsub, int a, unsigned asub, int b, unsigned bsub) {
    BlackwellIRInsn i = I0(BW_IR_FMUL); i.dst_vreg = d; i.dst_subreg = (uint8_t)dsub; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; emit(e, i, K_FIXED);
}
static void fmax_r(Em *e, int d, int a, unsigned asub, int b, unsigned bsub) {
    BlackwellIRInsn i = I0(BW_IR_FMNMX_MAX); i.dst_vreg = d; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; emit(e, i, K_FIXED);
}

int omega_pq_spec_ok(const OmegaPqSpec *s) {
    if (!s || s->k < 128u || s->k > 32768u || s->k % 128u) return 0;
    const uint32_t kb = s->k / 128u, g = s->groups ? s->groups : kb;
    return g <= 32u && kb % g == 0 && s->in_type <= OMEGA_QG_IN_BF16 && s->mutant >= 0 && s->mutant <= 3;
}

int omega_blackwell_codegen_pq_prog(const OmegaPqSpec *spec, BlackwellIRProgram *prog) {
    if (!omega_pq_spec_ok(spec) || !prog) return -1;
    const uint32_t K = spec->k, KB = K / 128u;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    int udesc = UV64(e), px = V64(e), pq = V64(e), ps = V64(e), tid = V(e), row = V(e), warp = V(e);
    int v = V128(e), w = V64(e), a = V128(e), m = V(e), t = V(e), fl = V(e), E = V(e), M = V(e), c = V(e), k1 = V(e), kneg = V(e), k262 = V(e);
    int sb = V(e), ib = V(e), d0 = V(e), d1 = V(e), word = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358); ldc64(e, px, 0x380); ldc64(e, pq, 0x388); ldc64(e, ps, 0x390);
    s2r(e, tid, BW_SR_TID_X); s2r(e, row, BW_SR_CTAID_X);
    if (spec->groups) { /* row split over ctaid.y: this CTA's groups start at y G */
        int cy = V(e); if (e->err) return -1;
        s2r(e, cy, BW_SR_CTAID_Y); if (spec->mutant != 3) imadi(e, tid, cy, 0, 32u * spec->groups, tid);
    }
    shr(e, warp, tid, 0, 5);
    if (spec->in_type == OMEGA_QG_IN_F32) { imadw(e, px, row, K * 4u, px); imadw(e, px, tid, 16u, px); ldg_wide(e, BW_IR_LDG_E_128, v, px, 0, udesc, 1); }
    else { imadw(e, px, row, K * 2u, px); imadw(e, px, tid, 8u, px); ldg_wide(e, BW_IR_LDG_E_64, w, px, 0, udesc, 1); }
    imadw(e, pq, row, K, pq); imadw(e, pq, tid, 4u, pq);
    imadw(e, ps, row, KB * 4u, ps); imadw(e, ps, warp, 4u, ps);
    movi(e, fl, 0x0d800000u);   /* 2^-100 */
    movi(e, k1, 0x001fffffu);
    movi(e, kneg, 0xfc000000u); /* -8 << 23 */
    movi(e, k262, 0x83000000u); /* 262 << 23 */
    if (spec->in_type == OMEGA_QG_IN_BF16) { /* bf16 pairs -> f32: element 2j in the low half of word j */
        for (unsigned j = 0; j < 2; j++) {
            BlackwellIRInsn lo = I0(BW_IR_IMAD); lo.dst_vreg = v; lo.dst_subreg = (uint8_t)(2u * j); lo.src1_vreg = w; lo.src1_subreg = (uint8_t)j; lo.imm = 65536u; lo.src3_vreg = NOREG; emit(e, lo, K_FIXED);
            BlackwellIRInsn hi = I0(BW_IR_LOP3_AND); hi.dst_vreg = v; hi.dst_subreg = (uint8_t)(2u * j + 1u); hi.src1_vreg = w; hi.src1_subreg = (uint8_t)j; hi.imm = 0xffff0000u; emit(e, hi, K_FIXED);
        }
    }
    for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn an = I0(BW_IR_LOP3_AND); an.dst_vreg = a; an.dst_subreg = (uint8_t)q; an.src1_vreg = v; an.src1_subreg = (uint8_t)q; an.imm = 0x7fffffffu; emit(e, an, K_FIXED); }
    fmax_r(e, m, a, 0, a, 1); fmax_r(e, m, m, 0, a, 2); fmax_r(e, m, m, 0, a, 3); fmax_r(e, m, m, 0, fl, 0);
    if (spec->mutant != 1) for (uint32_t l = 16; l >= 1; l >>= 1) { shfl_bfly(e, t, m, l); fmax_r(e, m, m, 0, t, 0); }
    shr(e, E, m, 0, 23);
    andi(e, M, m, 0, 0x007fffffu);
    if (spec->mutant == 2) movi(e, c, 0);
    else { imadi(e, c, M, 0, 1u, k1); shr(e, c, c, 0, 23); }
    iadd(e, E, E, c);                                   /* E + c */
    imadi(e, sb, E, 0, 1u << 23, kneg);                 /* scale bits (E + c - 8) << 23 */
    imadi(e, ib, E, 0, 0xff800000u, k262);              /* inv bits (262 - E - c) << 23 */
    for (unsigned q = 0; q < 4; q++) fmul_r(e, v, q, v, q, ib, 0);
    f2e4(e, d0, v, 1, v, 0);
    f2e4(e, d1, v, 3, v, 2);
    imadi(e, word, d1, 0, 65536u, d0);
    stg(e, pq, 0, word, udesc);
    stg(e, ps, 0, sb, udesc);
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

/* ======================= grouped GEMM on QMMA (see the header) ======================= */
int omega_qg_spec_ok(const OmegaQgSpec *s) {
    if (!s) return 0;
    if (s->k < 128u || s->k > 32768u || s->k % 128u) return 0; /* the k loop counts at run time: no k-sized immediate */
    if (s->rt != 4) return 0; /* rt 8 needs more than 128 registers and 17408 shared bytes (QMD cap 16384) */
    if (s->warps != 8 && s->warps != 16) return 0;
    if (s->epi > OMEGA_QG_EPI_SWIGLU || s->direct > 1 || s->wait > 120u) return 0;
    if (s->epi == OMEGA_QG_EPI_SWIGLU && s->n % 256u) return 0; /* gate and up halves start on a 128-row scale block */
    if (s->n == 0 || s->n % (s->warps * 16u) || (uint64_t)(s->epi == OMEGA_QG_EPI_F32 ? 4u : 1u) * s->n * 16u * s->rt >= (1u << 23)) return 0; /* row stride (n f32, or n / 2 bf16 = n bytes) x 16 rt: the store offsets fit the signed 24-bit field */
    if ((16u * s->rt * 4u) % (32u * s->warps)) return 0; /* whole 16-byte staging slots per thread */
    if (s->mutant < 0 || s->mutant > 4) return 0;
    return 1;
}
uint32_t omega_qg_shared_bytes(const OmegaQgSpec *s) { return omega_qg_spec_ok(s) ? 2u * (16u * s->rt * 64u + 16u * s->rt * 4u) : 0u; }
uint32_t omega_qg_span(const OmegaQgSpec *s) { return omega_qg_spec_ok(s) ? s->warps * 16u : 0u; }

static void ffma_s(Em *e, int d, unsigned dsub, int a, unsigned asub, int b, unsigned bsub, int c, unsigned csub) {
    BlackwellIRInsn i = I0(BW_IR_FFMA); i.dst_vreg = d; i.dst_subreg = (uint8_t)dsub; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub;
    i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; i.src3_vreg = c; i.src3_subreg = (uint8_t)csub; emit(e, i, K_FIXED);
}
static void mufu_q(Em *e, BlackwellIROpcode op, int d, int a) { BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = a; emit(e, i, K_VAR); }

int omega_blackwell_codegen_qgemm_prog(const OmegaQgSpec *spec, BlackwellIRProgram *prog) {
    if (!omega_qg_spec_ok(spec) || !prog) return -1;
    const uint32_t K = spec->k, RT = spec->rt, W = spec->warps, T = 32u * W;
    const uint32_t BM = 16u * RT, KB = K / 128u, SPAN = W * 16u, NJ = BM * 4u / T, ABUF = BM * 64u, BUF = ABUF + BM * 4u;
    const uint32_t NS = spec->epi == OMEGA_QG_EPI_SWIGLU ? 2u : 1u; /* scale rows per warp: gate + up block, or one block */
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;

    int udesc = UV64(e), ptab = V64(e), pidx = V64(e), px = V64(e), pxs = V64(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), g = V(e), tq = V(e), one = V(e), tent = V64(e);
    int wbase = V64(e), sbase = V64(e), rc = V64(e), row0 = V(e), n0w = V(e), tmp = V(e), tmp2 = V(e), rb = V(e), la = V(e), lsa = V(e);
    int pw[2] = { V64(e), V64(e) }, ps[2] = { V64(e), NS > 1 ? V64(e) : NOREG };
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, ptab, 0x380); if (!spec->direct) ldc64(e, pidx, 0x388); ldc64(e, px, 0x390); ldc64(e, pxs, 0x3a0);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, one, 1);
    shr(e, warp, tid, 0, 5); andi(e, lane, tid, 0, 31); shr(e, g, lane, 0, 2); andi(e, tq, lane, 0, 3);
    imadw(e, tent, cy, OMEGA_QG_TILE_BYTES, ptab);
    ldg_wide(e, BW_IR_LDG_E_64, wbase, tent, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_64, sbase, tent, 8, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_64, rc, tent, 16, udesc, 0);
    imadi(e, row0, rc, 0, 1, NOREG);
    imadi(e, n0w, cx, 0, SPAN, NOREG); imadi(e, tmp, warp, 0, 16u, NOREG); iadd(e, n0w, n0w, tmp);
    /* weight rows of the GEMV tensor (original order): F32 rows n0w + g (+ 8 for column tile 1); SWIGLU gate rows n0w / 2 + g, up rows n / 2 later */
    if (NS > 1) shr(e, rb, n0w, 0, 1); else imadi(e, rb, n0w, 0, 1u, NOREG);
    iadd(e, tmp, rb, g); imadw(e, pw[0], tmp, K, wbase); imadi(e, tmp2, tq, 0, 16u, NOREG); imadw(e, pw[0], tmp2, 1, pw[0]);
    imadi(e, tmp, one, 0, NS > 1 ? spec->n / 2u : 8u, NOREG); imadw(e, pw[1], tmp, K, pw[0]);
    /* scales [n / 128][KB] f32 * 2^56 (omega_gpu_fp8_api.h): the block of row rb (and of rb + n / 2) */
    shr(e, tmp, rb, 0, 7); imadw(e, ps[0], tmp, KB * 4u, sbase);
    if (NS > 1) { imadi(e, tmp, one, 0, spec->n / 2u, rb); shr(e, tmp, tmp, 0, 7); imadw(e, ps[1], tmp, KB * 4u, sbase); }
    imadi(e, la, lane, 0, 16u, NOREG);
    imadi(e, lsa, g, 0, 16u, NOREG);

    /* staging: slot = tid + jj T -> row r = slot / 4, 16-byte segment t4 = tid % 4 of the 64-byte chunk */
    int gx[4], gs[4], sa[4], ssa[4];
    for (uint32_t jj = 0; jj < NJ; jj++) { gx[jj] = V64(e); gs[jj] = V64(e); sa[jj] = V(e); ssa[jj] = V(e); }
    int r = V(e), sidx = V64(e), srow = V(e), t4 = V(e), hi = V(e), gg = V(e), ti = V(e);
    if (e->err) return -1;
    andi(e, t4, tid, 0, 3);
    for (uint32_t jj = 0; jj < NJ; jj++) {
        shr(e, r, tid, 0, 2); imadi(e, r, one, 0, jj * T / 4u, r);
        iadd(e, tmp, row0, r);
        if (spec->direct) imadi(e, srow, tmp, 0, 1u, NOREG);
        else { imadw(e, sidx, tmp, 4u, pidx); ldg32(e, srow, sidx, udesc, 1); }
        imadw(e, gx[jj], srow, K, px); imadi(e, tmp2, t4, 0, 16u, NOREG); imadw(e, gx[jj], tmp2, 1, gx[jj]);
        imadw(e, gs[jj], srow, KB * 4u, pxs);
        /* A slot: i * 1024 + (gg * 4 + t4) * 16 + 4 hi, i = r / 16, gg = r % 8, hi = (r / 8) % 2 */
        shr(e, ti, r, 0, 4); andi(e, gg, r, 0, 7); shr(e, hi, r, 0, 3); andi(e, hi, hi, 0, 1);
        imadi(e, tmp, gg, 0, 4u, t4); imadi(e, tmp, tmp, 0, 16u, NOREG); imadi(e, tmp, ti, 0, 1024u, tmp); imadi(e, sa[jj], hi, 0, 4u, tmp);
        /* scale slot: ABUF + ((i / 2) * 8 + gg) * 16 + 4 * ((i % 2) * 2 + hi) */
        shr(e, tmp, ti, 0, 1); imadi(e, tmp, tmp, 0, 8u, gg); imadi(e, tmp, tmp, 0, 16u, NOREG);
        andi(e, tmp2, ti, 0, 1); imadi(e, tmp2, tmp2, 0, 2u, hi); imadi(e, ssa[jj], tmp2, 0, 4u, tmp);
    }

    int acc[16], wa[2], wb[2], tq2[2], aq = V128(e), sap = V128(e), sc = V128(e), stgr[4], scv[4], cntk = V(e), onef1 = V(e), c2m56 = V(e);
    int rr[2] = { V(e), NS > 1 ? V(e) : NOREG }, rrn[2] = { V(e), NS > 1 ? V(e) : NOREG };
    for (uint32_t x = 0; x < RT * 2u; x++) acc[x] = V128(e);
    for (uint32_t j = 0; j < 2; j++) { wa[j] = V128(e); wb[j] = V128(e); tq2[j] = V128(e); }
    for (uint32_t jj = 0; jj < NJ; jj++) { stgr[jj] = V128(e); scv[jj] = V(e); }
    if (e->err) return -1;
    movi(e, onef1, 0x3f800000u);
    movi(e, c2m56, 0x23800000u); /* 2^-56: undoes the GEMV tensor's scale fold exactly (sw * 2^-56 is the checkpoint bf16 scale) */

    #define ISSUE(wset, sb, xoff, woff, with_scale) do { \
        for (uint32_t jj = 0; jj < NJ; jj++) { ldw_pf(e, BW_IR_LDG_E_128, stgr[jj], gx[jj], (xoff), udesc, (sb)); ldw_pf(e, BW_IR_LDG_E, scv[jj], gs[jj], 0u, udesc, (sb)); } \
        for (uint32_t j = 0; j < 2; j++) ldw_pf(e, BW_IR_LDG_E_128, (wset)[j], pw[j], (woff), udesc, (sb)); \
        if (with_scale) for (uint32_t s = 0; s < NS; s++) ldw_pf(e, BW_IR_LDG_E, rrn[s], ps[s], 0u, udesc, (sb)); } while (0)
    #define STAGE(buf) do { for (uint32_t jj = 0; jj < NJ; jj++) { static const uint32_t off[4] = { 0, 8, 512, 520 }; \
        for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn st = I0(BW_IR_STS32); imadi(e, tmp, one, 0, (buf) + off[q], sa[jj]); \
            st.src1_vreg = tmp; st.src2_vreg = stgr[jj]; st.src2_subreg = (uint8_t)q; emit(e, st, K_STORE); } \
        { BlackwellIRInsn st = I0(BW_IR_STS32); imadi(e, tmp, one, 0, (buf) + ABUF, ssa[jj]); st.src1_vreg = tmp; st.src2_vreg = scv[jj]; emit(e, st, K_STORE); } } \
        bar_sync(e); } while (0)
    #define COMPUTE(buf, wset, odd) do { \
        for (uint32_t i = 0; i < RT; i++) { \
            const uint32_t ii = spec->mutant == 3 ? (i ^ 1u) : i; \
            if ((i & 1u) == 0) lds128(e, sap, lsa, (buf) + ABUF + (i / 2u) * 128u, 1); \
            lds128(e, aq, la, (buf) + (ii * 2u) * 512u, 1); \
            for (uint32_t j = 0; j < 2; j++) qmma_s(e, tq2[j], aq, (wset)[j], 0, NOREG, 2); \
            if (!(spec->mutant == 4 && (odd))) { lds128(e, aq, la, (buf) + (ii * 2u + 1u) * 512u, 1); \
                for (uint32_t j = 0; j < 2; j++) qmma_s(e, tq2[j], aq, (wset)[j], 2, tq2[j], 2); } \
            for (uint32_t h = 0; h < 2; h++) for (uint32_t j = 0; j < 2; j++) { \
                BlackwellIRInsn m = I0(BW_IR_FMUL); m.dst_vreg = sc; m.dst_subreg = (uint8_t)(2u * h + j); \
                m.src1_vreg = sap; m.src1_subreg = (uint8_t)(2u * (i & 1u) + h); m.src2_vreg = rr[NS > 1 ? j : 0]; \
                if (spec->mutant == 1) { m.src1_vreg = onef1; m.src1_subreg = 0; } /* drops sa: s = 1.0 * sw */ \
                emit(e, m, K_FIXED); } \
            { uint32_t wl = spec->wait; while (wl > 0) { uint32_t st = wl > 15u ? 15u : wl; nop(e, st); wl -= st; } } \
            for (uint32_t j = 0; j < 2; j++) for (unsigned q = 0; q < 4; q++) { const uint32_t x = i * 2u + j; \
                ffma_s(e, acc[x], q, tq2[j], q, sc, 2u * (q >> 1) + j, acc[x], q); } \
        } } while (0)

    for (uint32_t x = 0; x < RT * 2u; x++) for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn mz = I0(BW_IR_MOV_RZ); mz.dst_vreg = acc[x]; mz.dst_subreg = (uint8_t)q; emit(e, mz, K_FIXED); }
    ISSUE(wa, 3u, 0u, 0u, 1);
    consume(e, 3);
    for (uint32_t s = 0; s < NS; s++) fmul_r(e, rr[s], 0, rrn[s], 0, c2m56, 0);
    STAGE(0u);
    movi(e, cntk, KB);

    const int L_k = here(e);
    loop_begin(e);
    ISSUE(wb, 4u, 64u, 64u, 0);
    consume(e, 3);
    COMPUTE(0u, wa, 0);
    consume(e, 4);
    STAGE(BUF);
    {   int nf = V(e), inc = V(e); if (e->err) return -1;
        imadi(e, nf, one, 0, 0xfffffffeu, cntk); shr(e, nf, nf, 0, 31); imadi(e, nf, nf, 0, 0xffffffffu, one);
        for (uint32_t jj = 0; jj < NJ; jj++) { imadw(e, gx[jj], nf, 128u, gx[jj]); imadw(e, gs[jj], nf, 4u, gs[jj]); }
        for (uint32_t j = 0; j < 2; j++) imadw(e, pw[j], nf, 128u, pw[j]);
        imadi(e, inc, nf, 0, spec->mutant == 2 ? 0u : 4u, NOREG); for (uint32_t s = 0; s < NS; s++) imadw(e, ps[s], inc, 1, ps[s]);
    }
    ISSUE(wa, 3u, 0u, 0u, 1);
    COMPUTE(BUF, wb, 1);
    consume(e, 3);
    STAGE(0u);
    for (uint32_t s = 0; s < NS; s++) fmul_r(e, rr[s], 0, rrn[s], 0, c2m56, 0);
    imadi(e, cntk, one, 0, 0xffffffffu, cntk);
    isetp_ge_u32(e, cntk, one);
    bra_if_p0(e, L_k);
    loop_end(e);
    #undef ISSUE
    #undef STAGE
    #undef COMPUTE

    /* ---- epilogue (as omega_blackwell_codegen_tc2.c, natural SwiGLU order, no extra scale) ---- */
    int g2 = V(e), cnt2 = V(e), rrow = V(e), onee = V(e), pyr = V64(e);
    {
        int tid2 = V(e), cx2 = V(e), cy2 = V(e), lane2 = V(e), tq3 = V(e), tab2 = V64(e), ten2 = V64(e), rc2 = V64(e), py2 = V64(e), n2 = V(e), nw2 = V(e), r02 = V(e);
        if (e->err) return -1;
        ldc64(e, tab2, 0x380); ldc64(e, py2, 0x398); ldc32(e, n2, 0x3a8);
        s2r(e, tid2, BW_SR_TID_X); s2r(e, cx2, BW_SR_CTAID_X); s2r(e, cy2, BW_SR_CTAID_Y);
        imadw(e, ten2, cy2, OMEGA_QG_TILE_BYTES, tab2);
        ldg_wide(e, BW_IR_LDG_E_64, rc2, ten2, 16, udesc, 1);
        andi(e, lane2, tid2, 0, 31); shr(e, g2, lane2, 0, 2); andi(e, tq3, lane2, 0, 3);
        imadi(e, r02, rc2, 0, 1, NOREG); imadi(e, cnt2, rc2, 1, 1, NOREG); movi(e, onee, 1);
        shr(e, tmp, tid2, 0, 5); imadi(e, tmp, tmp, 0, 16u, NOREG); imadi(e, nw2, cx2, 0, SPAN, tmp);
        if (spec->epi == OMEGA_QG_EPI_F32) {
            iadd(e, tmp, r02, g2); imadr(e, tmp, tmp, n2, nw2); imadi(e, tmp2, tq3, 0, 2u, NOREG); iadd(e, tmp, tmp, tmp2); imadw(e, pyr, tmp, 4u, py2);
        } else {
            iadd(e, tmp, r02, g2); imadr(e, tmp, tmp, n2, NOREG); imadw(e, pyr, tmp, 1u, py2);
            shr(e, tmp, nw2, 0, 1); imadi(e, tmp2, tq3, 0, 2u, tmp); imadw(e, pyr, tmp2, 2u, pyr);
        }
    }
    int kf = NOREG, onef = NOREG, c125 = NOREG, tg = NOREG, tex = NOREG, o0 = NOREG, o1 = NOREG, pk = NOREG;
    if (spec->epi == OMEGA_QG_EPI_SWIGLU) {
        kf = V(e); onef = V(e); c125 = V(e); tg = V(e); tex = V(e); o0 = V(e); o1 = V(e); pk = V(e);
        if (e->err) return -1;
        movi(e, kf, 0xbfb8aa3bu); movi(e, onef, 0x3f800000u);
        movi(e, c125, 0x42fa0000u); /* 125.0f: caps t so 1 / (2^t + 1) stays a normal f32 (the chip flushes subnormal MUFU.RCP results: hardware, evidence OWN30B-L1G-SWIGLU-DIAG-ca29218e) */
    }
    if (e->err) return -1;
    const uint32_t ystride = spec->epi == OMEGA_QG_EPI_F32 ? spec->n * 4u : spec->n;
    int skips[16]; uint32_t nskip = 0;
    const int at_bssy = bssy_fwd(e);
    for (uint32_t i = 0; i < RT; i++) for (uint32_t h = 0; h < 2; h++) {
        const uint32_t roff = (16u * i + 8u * h) * ystride;
        imadi(e, rrow, onee, 0, 16u * i + 8u * h, g2);
        isetp_ge_u32(e, rrow, cnt2);
        skips[nskip++] = bra_if_p0_fwd(e);
        if (spec->epi == OMEGA_QG_EPI_F32) {
            for (uint32_t j = 0; j < 2; j++) for (unsigned q = 0; q < 2; q++) { BlackwellIRInsn st = I0(BW_IR_STG_E); st.src1_vreg = pyr; st.src2_vreg = acc[i * 2 + j]; st.src2_subreg = (uint8_t)(2u * h + q); st.imm = roff + j * 32u + 4u * q; st.ureg = udesc; emit(e, st, K_STORE); }
        } else {
            const int G = acc[i * 2], U = acc[i * 2 + 1];
            for (unsigned q = 0; q < 2; q++) {
                const int o = q ? o1 : o0; const unsigned sub = 2u * h + q;
                fmul_r(e, tg, 0, G, sub, kf, 0);
                { BlackwellIRInsn mn = I0(BW_IR_FMNMX_MIN); mn.dst_vreg = tg; mn.src1_vreg = tg; mn.src2_vreg = c125; emit(e, mn, K_FIXED); }
                mufu_q(e, BW_IR_MUFU_EX2, tex, tg);
                fadd(e, tex, tex, onef);
                mufu_q(e, BW_IR_MUFU_RCP, tex, tex);
                fmul_r(e, o, 0, G, sub, tex, 0);
                fmul_r(e, o, 0, o, 0, U, sub);
            }
            f2fp_pack(e, pk, o1, o0);
            stg(e, pyr, roff, pk, udesc);
        }
    }
    {   const int at_bsync = here(e);
        bsync(e);
        for (uint32_t q = 0; q < nskip; q++) patch_bra(e, skips[q], at_bsync);
        patch_bra(e, at_bssy, at_bsync + 1); }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}
