/* omega_blackwell_codegen_pf.c: device-resident prefill helper kernels (OWN-30B T3b). See the header. */
#include "omega_blackwell_codegen_pf.h"
#include "omega_bw_em.h"

int omega_blackwell_codegen_pf_copy_prog(BlackwellIRProgram *prog) {
    if (!prog) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    int udesc = UV64(e), pd = V64(e), ps = V64(e), units = V(e), tid = V(e), cta = V(e), i = V(e), v = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358); ldc64(e, pd, 0x380); ldc64(e, ps, 0x388); ldc32(e, units, 0x390);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cta, BW_SR_CTAID_X);
    imadi(e, i, cta, 0, OMEGA_PF_COPY_THREADS, tid);
    isetp_ge_u32(e, i, units);
    { BlackwellIRInsn x = I0(BW_IR_EXIT); x.predicate_p0 = true; emit(e, x, K_EXIT); }
    imadw(e, ps, i, 16u, ps); imadw(e, pd, i, 16u, pd);
    ldg_wide(e, BW_IR_LDG_E_128, v, ps, 0, udesc, 1);
    for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn st = I0(BW_IR_STG_E); st.src1_vreg = pd; st.src2_vreg = v; st.src2_subreg = (uint8_t)q; st.imm = 4u * q; st.ureg = udesc; emit(e, st, K_STORE); }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}
