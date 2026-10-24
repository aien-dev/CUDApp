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

/* omega_blackwell_codegen_tc2.c: tensor-core grouped FP8 GEMM v2 (OWN-30B T3b). See the header. The K loop, staging and
 * decode are cut 1's (omega_blackwell_codegen_tc.c, chip gate OWN30B-T3-TCGEMM-6d2803ec) with the scale folded through the
 * accumulator and the v2 epilogues. */
#include "omega_blackwell_codegen_tc2.h"
#include "omega_bw_em.h"

int omega_tc2_spec_ok(const OmegaTc2Spec *s) {
    if (!s) return 0;
    if (s->k < 128u || s->k > 16384u || s->k % 128u) return 0;
    if (s->rt != 2 && s->rt != 4 && s->rt != 8) return 0;
    if (s->ct != 2) return 0; /* the ratio table pairs the warp's two column tiles (one LDG.64) */
    if (s->rt * s->ct < 4u) return 0;
    if (s->warps != 4 && s->warps != 8 && s->warps != 16) return 0;
    if (s->aqn != 1 && s->aqn != 2) return 0;
    if (s->epi > OMEGA_TC2_EPI_SWIGLU || (s->epi == OMEGA_TC2_EPI_SWIGLU && s->ct != 2)) return 0;
    if (s->direct > 1) return 0;
    if (s->n == 0 || s->n % (s->warps * 8u * s->ct) || (uint64_t)s->n * 4u * 16u * s->rt >= (1u << 23)) return 0;
    if ((16u * s->rt * 8u) % (32u * s->warps)) return 0; /* whole LDG.128 staging slots per thread */
    if (s->mutant < 0 || s->mutant > 4 || (s->mutant == 3 && s->rt < 2)) return 0;
    return 1;
}
uint32_t omega_tc2_shared_bytes(const OmegaTc2Spec *s) { return omega_tc2_spec_ok(s) ? 2u * 16u * s->rt * 128u : 0u; }
uint32_t omega_tc2_span(const OmegaTc2Spec *s) { return omega_tc2_spec_ok(s) ? s->warps * 8u * s->ct : 0u; }

/* one e4m3 word (bytes b0..b3) -> B register pair: lo = (bf(b0), bf(b2)), hi = (bf(b1), bf(b3)) (cut 1) */
static void decode_word(Em *e, int bpair, int wq, unsigned word, int m, int s, int t) {
    andi(e, m, wq, word, 0x007F007Fu); andi(e, s, wq, word, 0x00800080u);
    imadi(e, t, m, 0, 16u, NOREG);
    { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = bpair; i.dst_subreg = 0; i.src1_vreg = s; i.imm = 256u; i.src3_vreg = t; emit(e, i, K_FIXED); }
    andi(e, m, wq, word, 0x7F007F00u); andi(e, s, wq, word, 0x80008000u);
    shr(e, t, m, 0, 4);
    { BlackwellIRInsn i = I0(BW_IR_IADD3); i.dst_vreg = bpair; i.dst_subreg = 1; i.src1_vreg = t; i.src2_vreg = s; emit(e, i, K_FIXED); }
}
static void fmul_s(Em *e, int d, unsigned dsub, int a, unsigned asub, int b, unsigned bsub) {
    BlackwellIRInsn i = I0(BW_IR_FMUL); i.dst_vreg = d; i.dst_subreg = (uint8_t)dsub; i.src1_vreg = a; i.src1_subreg = (uint8_t)asub; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; emit(e, i, K_FIXED);
}
static void mufu(Em *e, BlackwellIROpcode op, int d, int a) { BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = a; emit(e, i, K_VAR); }

int omega_blackwell_codegen_tc2_prog(const OmegaTc2Spec *spec, BlackwellIRProgram *prog) {
    if (!omega_tc2_spec_ok(spec) || !prog) return -1;
    const uint32_t K = spec->k, RT = spec->rt, CT = spec->ct, W = spec->warps, T = 32u * W, AQN = spec->aqn;
    const uint32_t BM = 16u * RT, KB = K / 128u, SPAN = W * 8u * CT, NJ = BM * 8u / T, BUF = BM * 128u;
    const uint32_t RSTRIDE = (KB + 1u) * 8u; /* bytes per 16-row group pair of the ratio table */
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    Em *e = &em;

    /* ---- prologue: arguments, ids, tile entry ---- */
    int udesc = UV64(e), ptab = V64(e), pidx = V64(e), px = V64(e), py = V64(e), nreg = V(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), g = V(e), tq = V(e), one = V(e), tent = V64(e);
    int wbase = V64(e), rbase = V64(e), rc = V64(e), row0 = V(e);
    int n0w = V(e), tmp = V(e), tmp2 = V(e), pw = V64(e), pr = V64(e), la = V(e), pyr = V64(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, ptab, 0x380); if (!spec->direct) ldc64(e, pidx, 0x388); ldc64(e, px, 0x390); ldc64(e, py, 0x398);
    ldc32(e, nreg, 0x3a0);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, one, 1);
    shr(e, warp, tid, 0, 5); andi(e, lane, tid, 0, 31); shr(e, g, lane, 0, 2); andi(e, tq, lane, 0, 3);
    imadw(e, tent, cy, OMEGA_TC2_TILE_BYTES, ptab);
    ldg_wide(e, BW_IR_LDG_E_64, wbase, tent, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_64, rbase, tent, 8, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_64, rc, tent, 16, udesc, 0);
    imadi(e, row0, rc, 0, 1, NOREG);
    /* this warp's first weight row n0w; weight pointer row (n0w + g), byte 16 t; ratio row group n0w / 8 */
    imadi(e, n0w, cx, 0, SPAN, NOREG); imadi(e, tmp, warp, 0, 8u * CT, NOREG); iadd(e, n0w, n0w, tmp);
    iadd(e, tmp, n0w, g); imadw(e, pw, tmp, K, wbase); imadi(e, tmp2, tq, 0, 16u, NOREG); imadw(e, pw, tmp2, 1, pw);
    shr(e, tmp, n0w, 0, 4); imadw(e, pr, tmp, RSTRIDE, rbase);
    imadi(e, la, lane, 0, 16u, NOREG);

    /* ---- staging assignment: slot = tid + jj * T -> row r = slot / 8, words w0 = 4 * (tid % 8) of the 64-element chunk ---- */
    int gx[8], sa[8];
    for (uint32_t jj = 0; jj < NJ; jj++) { gx[jj] = V64(e); sa[jj] = V(e); }
    int r = V(e), sidx = V64(e), srow = V(e), w0b = V(e), ts = V(e), s0 = V(e);
    if (e->err) return -1;
    andi(e, tmp, tid, 0, 7);
    imadi(e, w0b, tmp, 0, 16u, NOREG);
    shr(e, ts, tmp, 0, 1);
    andi(e, s0, tid, 0, 1); imadi(e, s0, s0, 0, 2u, NOREG);
    for (uint32_t jj = 0; jj < NJ; jj++) {
        shr(e, r, tid, 0, 3); imadi(e, r, one, 0, jj * T / 8u, r);
        iadd(e, tmp, row0, r);
        if (spec->direct) imadi(e, srow, tmp, 0, 1u, NOREG);
        else { imadw(e, sidx, tmp, 4u, pidx); ldg32(e, srow, sidx, udesc, 1); }
        imadw(e, gx[jj], srow, K * 2u, px); imadw(e, gx[jj], w0b, 1, gx[jj]);
        shr(e, tmp, r, 0, 4); imadi(e, tmp, tmp, 0, 4u, s0); imadi(e, tmp, tmp, 0, 32u, ts);
        andi(e, tmp2, r, 0, 7); imadi(e, tmp2, tmp2, 0, 4u, tmp); imadi(e, sa[jj], tmp2, 0, 16u, NOREG);
        shr(e, tmp2, r, 0, 3); andi(e, tmp2, tmp2, 0, 1); imadi(e, sa[jj], tmp2, 0, 4u, sa[jj]);
    }

    /* ---- loop-wide registers ---- */
    int acc[16], wa[2], wb[2], bp[2], aq[2], stgr[8], rr = V64(e), cntk = V(e), dm = V(e), ds = V(e), dt = V(e);
    for (uint32_t x = 0; x < RT * CT; x++) acc[x] = V128(e);
    for (uint32_t j = 0; j < CT; j++) { wa[j] = V128(e); wb[j] = V128(e); bp[j] = V64(e); }
    for (uint32_t q = 0; q < AQN; q++) aq[q] = V128(e);
    for (uint32_t jj = 0; jj < NJ; jj++) stgr[jj] = V128(e);
    if (e->err) return -1;

    #define ISSUE(wset, sb, xoff, woff, with_ratio) do { \
        for (uint32_t jj = 0; jj < NJ; jj++) ldw_pf(e, BW_IR_LDG_E_128, stgr[jj], gx[jj], (xoff), udesc, (sb)); \
        for (uint32_t j = 0; j < CT; j++) ldw_pf(e, BW_IR_LDG_E_128, (wset)[j], pw, (woff) + j * 8u * K, udesc, (sb)); \
        if (with_ratio) ldw_pf(e, BW_IR_LDG_E_64, rr, pr, 0u, udesc, (sb)); } while (0)
    #define STAGE(buf) do { for (uint32_t jj = 0; jj < NJ; jj++) { static const uint32_t off[4] = { 0, 8, 512, 520 }; \
        for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn st = I0(BW_IR_STS32); imadi(e, tmp, one, 0, (buf) + off[q], sa[jj]); \
            st.src1_vreg = tmp; st.src2_vreg = stgr[jj]; st.src2_subreg = (uint8_t)q; emit(e, st, K_STORE); } } bar_sync(e); } while (0)
    #define COMPUTE(buf, wset, odd) do { \
        for (uint32_t s = 0; s < 4; s++) { \
            if (spec->mutant == 4 && (odd) && s == 3) break; \
            for (uint32_t j = 0; j < CT; j++) decode_word(e, bp[j], (wset)[j], s, dm, ds, dt); \
            for (uint32_t i0 = 0; i0 < RT; i0 += AQN) { \
                const uint32_t nb = RT - i0 < AQN ? RT - i0 : AQN; \
                for (uint32_t q = 0; q < nb; q++) { uint32_t i = i0 + q, ii = spec->mutant == 3 ? (i ^ 1u) : i; \
                    lds128(e, aq[q], la, (buf) + (ii * 4u + s) * 512u, q == 0); } \
                for (uint32_t q = 0; q < nb; q++) for (uint32_t j = 0; j < CT; j++) { uint32_t x = (i0 + q) * CT + j; \
                    hmma(e, acc[x], aq[q], bp[j], acc[x], 8); } \
            } \
        } } while (0)
    /* acc *= rr[j] (all four quads of every tile of column tile j); the HMMA results must have settled first */
    #define RESCALE() do { nop(e, 15); nop(e, 14); \
        for (uint32_t x = 0; x < RT * CT; x++) for (unsigned q = 0; q < 4; q++) fmul_s(e, acc[x], q, acc[x], q, rr, x % CT); } while (0)

    for (uint32_t x = 0; x < RT * CT; x++) for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn mz = I0(BW_IR_MOV_RZ); mz.dst_vreg = acc[x]; mz.dst_subreg = (uint8_t)q; emit(e, mz, K_FIXED); }
    ISSUE(wa, 3u, 0u, 0u, 1);
    consume(e, 3);
    STAGE(0u);
    movi(e, cntk, KB);

    /* ---- K loop: one 128-wide block (two chunks) per iteration ---- */
    const int L_k = here(e);
    loop_begin(e);
    ISSUE(wb, 4u, 128u, 64u, 0);                   /* chunk 2kb+1 */
    consume(e, 3);
    if (spec->mutant != 1) RESCALE();              /* acc = acc * R[kb] (R[0] = 1) */
    COMPUTE(0u, wa, 0);
    consume(e, 4);
    STAGE(BUF);
    {   /* chunk 2kb+2 (the next block's first chunk) and R[kb+1]; the last iteration re-reads its own (harmless) */
        int nf = V(e), inc = V(e); if (e->err) return -1;
        imadi(e, nf, one, 0, 0xfffffffeu, cntk); shr(e, nf, nf, 0, 31); imadi(e, nf, nf, 0, 0xffffffffu, one);
        for (uint32_t jj = 0; jj < NJ; jj++) imadw(e, gx[jj], nf, 256u, gx[jj]);
        imadw(e, pw, nf, 128u, pw);
        imadi(e, inc, nf, 0, spec->mutant == 2 ? 0u : 8u, NOREG); imadw(e, pr, inc, 1, pr);
    }
    ISSUE(wa, 3u, 0u, 0u, 1);
    COMPUTE(BUF, wb, 1);
    consume(e, 3);
    STAGE(0u);
    imadi(e, cntk, one, 0, 0xffffffffu, cntk);
    isetp_ge_u32(e, cntk, one);
    bra_if_p0(e, L_k);
    loop_end(e);
    #undef ISSUE
    #undef STAGE
    #undef COMPUTE

    /* ---- final scale R[KB] (pr points at R[KB-1]) ---- */
    ldg_wide(e, BW_IR_LDG_E_64, rr, pr, 8u, udesc, 1);
    RESCALE();
    #undef RESCALE
    int g2 = V(e), cnt2 = V(e), rrow = V(e), onee = V(e);
    /* ---- epilogue addressing, recomputed in fresh registers (nothing of it stays live through the K loop) ---- */
    {
        int tid2 = V(e), cx2 = V(e), cy2 = V(e), lane2 = V(e), tq2 = V(e), tab2 = V64(e), ten2 = V64(e), rc2 = V64(e), py2 = V64(e), n2 = V(e), nw2 = V(e), r02 = V(e);
        if (e->err) return -1;
        ldc64(e, tab2, 0x380); ldc64(e, py2, 0x398); ldc32(e, n2, 0x3a0);
        s2r(e, tid2, BW_SR_TID_X); s2r(e, cx2, BW_SR_CTAID_X); s2r(e, cy2, BW_SR_CTAID_Y);
        imadw(e, ten2, cy2, OMEGA_TC2_TILE_BYTES, tab2);
        ldg_wide(e, BW_IR_LDG_E_64, rc2, ten2, 16, udesc, 1);
        andi(e, lane2, tid2, 0, 31); shr(e, g2, lane2, 0, 2); andi(e, tq2, lane2, 0, 3);
        imadi(e, r02, rc2, 0, 1, NOREG); imadi(e, cnt2, rc2, 1, 1, NOREG); movi(e, onee, 1);
        shr(e, tmp, tid2, 0, 5); imadi(e, tmp, tmp, 0, 8u * CT, NOREG); imadi(e, nw2, cx2, 0, SPAN, tmp);
        if (spec->epi == OMEGA_TC2_EPI_F32) {

            /* y + ((row0 + g) * n + n0w + 2t) * 4 */
            iadd(e, tmp, r02, g2); imadr(e, tmp, tmp, n2, nw2); imadi(e, tmp2, tq2, 0, 2u, NOREG); iadd(e, tmp, tmp, tmp2); imadw(e, pyr, tmp, 4u, py2);
        } else {

            /* act + (row0 + g) * n + (n0w / 2 + 2t) * 2 */
            iadd(e, tmp, r02, g2); imadr(e, tmp, tmp, n2, NOREG); imadw(e, pyr, tmp, 1u, py2);
            shr(e, tmp, nw2, 0, 1); imadi(e, tmp2, tq2, 0, 2u, tmp); imadw(e, pyr, tmp2, 2u, pyr);
        }
    }

    /* ---- store: rows 16i + 8h + g of the tile only when < cnt (a partial tile must not touch the next expert's rows) ---- */
    int kf = NOREG, p64 = NOREG, onef = NOREG, tg = NOREG, tex = NOREG, o0 = NOREG, o1 = NOREG, pk = NOREG;
    if (spec->epi == OMEGA_TC2_EPI_SWIGLU) {
        kf = V(e); p64 = V(e); onef = V(e); tg = V(e); tex = V(e); o0 = V(e); o1 = V(e); pk = V(e);
        if (e->err) return -1;
        movi(e, kf, 0xbfb8aa3bu);   /* -log2(e) as f32 (-1.4426950409f) */
        movi(e, onef, 0x3f800000u); /* 1.0f */
        movi(e, p64, 0x5f800000u);  /* 2^64: the stored-activation scale of the next GEMM */
    }
    if (e->err) return -1;
    const uint32_t ystride = spec->epi == OMEGA_TC2_EPI_F32 ? spec->n * 4u : spec->n;
    /* rows grow with (i, h): a lane whose row reaches cnt jumps to the one BSYNC after the last store */
    int skips[2 * 8]; uint32_t nskip = 0;
    const int at_bssy = bssy_fwd(e);
    for (uint32_t i = 0; i < RT; i++) for (uint32_t h = 0; h < 2; h++) {
        const uint32_t roff = (16u * i + 8u * h) * ystride;
        imadi(e, rrow, onee, 0, 16u * i + 8u * h, g2);
        isetp_ge_u32(e, rrow, cnt2);
        skips[nskip++] = bra_if_p0_fwd(e);
        if (spec->epi == OMEGA_TC2_EPI_F32) {
            for (uint32_t j = 0; j < CT; j++) for (unsigned q = 0; q < 2; q++) { BlackwellIRInsn st = I0(BW_IR_STG_E); st.src1_vreg = pyr; st.src2_vreg = acc[i * CT + j]; st.src2_subreg = (uint8_t)(2u * h + q); st.imm = roff + j * 32u + 4u * q; st.ureg = udesc; emit(e, st, K_STORE); }
        } else {
            /* act = silu(g) * u * 2^64 (the next GEMM reads activations stored * 2^64) */
            const int G = acc[i * 2], U = acc[i * 2 + 1];
            for (unsigned q = 0; q < 2; q++) {
                const int o = q ? o1 : o0; const unsigned sub = 2u * h + q;
                fmul_s(e, tg, 0, G, sub, kf, 0);
                mufu(e, BW_IR_MUFU_EX2, tex, tg);
                fadd(e, tex, tex, onef);
                mufu(e, BW_IR_MUFU_RCP, tex, tex);
                fmul_s(e, o, 0, G, sub, tex, 0);
                fmul_s(e, o, 0, o, 0, U, sub);
                fmul_s(e, o, 0, o, 0, p64, 0);
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
