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
    return s && s->k >= 128u && s->k <= 4096u && s->k % 128u == 0 && s->in_type <= OMEGA_QG_IN_BF16 && s->mutant >= 0 && s->mutant <= 2;
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
