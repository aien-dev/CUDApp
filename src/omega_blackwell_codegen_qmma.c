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

/* omega_blackwell_codegen_qmma.c: QMMA.16832.F32.E4M3.E4M3 placeholder, patch, probe and peak kernels (W-L1G). See the header. */
#include "omega_blackwell_codegen_qmma.h"
#include "omega_bw_em.h"
#include "omega_bw_em_q.h"
#include "omega_gpu_fmt.h"

static uint32_t rd32q(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static void wr32q(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }

int omega_qmma_is_mark(const BlackwellIRInsn *in) { return in && in->op == BW_IR_HMMA_BF16 && in->imm == OMEGA_QMMA_MARK; }

int omega_qmma_patch_code(const BlackwellIRProgram *prog, uint8_t *code, size_t len) {
    if (!prog || !code || len < prog->count * 16u) return -1;
    int n = 0;
    for (size_t i = 0; i < prog->count; i++) {
        if (prog->insns[i].op == BW_IR_FMNMX_MAX && prog->insns[i].imm == OMEGA_QMMA_MARK_F2E4) {
            uint8_t *w = code + i * 16u; uint32_t w0 = rd32q(w), w1 = rd32q(w + 4), w2 = rd32q(w + 8);
            if ((w0 & 0xffffu) != 0x7209u || (w1 & 0xffffff00u) != 0 || w2 != 0x07800000u) return -1;
            wr32q(w, (w0 & 0xffff0000u) | 0x723eu); wr32q(w + 8, 0x048070ffu); n++; continue;
        }
        if (!omega_qmma_is_mark(&prog->insns[i])) continue;
        uint8_t *w = code + i * 16u;
        uint32_t w0 = rd32q(w), w1 = rd32q(w + 4), w2 = rd32q(w + 8);
        if ((w0 & 0xffffu) != 0x723cu || (w1 & 0xffffff00u) != 0 || (w2 & 0xffffff00u) != 0x00041800u) return -1;
        wr32q(w, (w0 & 0xffff0000u) | 0x727au);
        wr32q(w + 8, (w2 & 0xffu) | 0x00002c00u);
        n++;
    }
    if (n > 0) { size_t b = omega_bw_program_code_bytes(prog); omega_kernel_dump(code, b && b <= len ? b : prog->count * 16u); } /* the patched body is a different kernel */
    return n;
}

int omega_qmma_patch_selftest(char *msg, size_t msgn) {
    static BlackwellIRProgram p; static uint8_t buf[64 * 16]; size_t len = 0;
    omega_bw_ir_init(&p);
    Em em; memset(&em, 0, sizeof em); em.p = &p; Em *e = &em;
    int udesc = UV64(e), py = V64(e), a = V128(e), b = V64(e), c = V128(e);
    if (e->err) { omega_gpu_fmt(msg, msgn, "vregs"); return -1; }
    ldcu64(e, udesc, 0x358); ldc64(e, py, 0x380);
    ldg_wide(e, BW_IR_LDG_E_128, a, py, 0, udesc, 1); ldg_wide(e, BW_IR_LDG_E_64, b, py, 16, udesc, 0); ldg_wide(e, BW_IR_LDG_E_128, c, py, 32, udesc, 0);
    qmma(e, c, a, b, c, 15); nop(e, 15);
    { int f = V(e); if (e->err) { omega_gpu_fmt(msg, msgn, "vregs"); return -1; } f2e4(e, f, c, 0, c, 1); stg(e, py, 64, f, udesc); }
    hmma(e, c, a, b, c, 15); nop(e, 15);
    for (unsigned q = 0; q < 4; q++) stg(e, py, 4u * q, c, udesc);
    tail(e);
    if (e->err || omega_bw_regalloc_solve(&p) != 0 || omega_bw_encode_program(&p, buf, sizeof buf, &len) != 0) { omega_gpu_fmt(msg, msgn, "build/encode failed"); return -1; }
    int np = omega_qmma_patch_code(&p, buf, len);
    if (np != 2) { omega_gpu_fmt(msg, msgn, "patched %d placeholders, expected 2", np); return -1; }
    int seenq = 0, seenh = 0, seenf = 0;
    for (size_t i = 0; i < p.count; i++) {
        if (p.insns[i].op == BW_IR_FMNMX_MAX) { const uint8_t *w = buf + i * 16u; uint32_t w0 = rd32q(w), w1 = rd32q(w + 4), w2 = rd32q(w + 8);
            if ((w0 & 0xffffu) != 0x723eu || (w1 & 0xffffff00u) != 0 || w2 != 0x048070ffu) { omega_gpu_fmt(msg, msgn, "F2FP.E4M3 word %zu: %08x %08x %08x", i, w0, w1, w2); return -1; } seenf++; continue; }
        if (p.insns[i].op != BW_IR_HMMA_BF16) continue;
        const uint8_t *w = buf + i * 16u; uint32_t w0 = rd32q(w), w1 = rd32q(w + 4), w2 = rd32q(w + 8);
        if (omega_qmma_is_mark(&p.insns[i])) {
            /* oracle 0x0414727a 0x0000000c 0x00002c14 with registers masked */
            if ((w0 & 0xffffu) != 0x727au || (w1 & 0xffffff00u) != 0 || (w2 & 0xffffff00u) != 0x00002c00u) { omega_gpu_fmt(msg, msgn, "QMMA word %zu: %08x %08x %08x", i, w0, w1, w2); return -1; }
            seenq++;
        } else {
            if ((w0 & 0xffffu) != 0x723cu || (w2 & 0xffffff00u) != 0x00041800u) { omega_gpu_fmt(msg, msgn, "HMMA word %zu changed: %08x %08x", i, w0, w2); return -1; }
            seenh++;
        }
    }
    if (seenq != 1 || seenh != 1 || seenf != 1) { omega_gpu_fmt(msg, msgn, "saw %d QMMA, %d HMMA", seenq, seenh); return -1; }
    omega_gpu_fmt(msg, msgn, "QMMA.16832.F32.E4M3.E4M3 and F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C match the oracle bits, HMMA untouched");
    return 0;
}

int omega_blackwell_codegen_qmma_probe_prog(int mutant, uint32_t wait_nops, BlackwellIRProgram *prog) {
    if (!prog || mutant < 0 || mutant > 1 || wait_nops > 16) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    int udesc = UV64(e), pin = V64(e), pout = V64(e), tid = V(e), cx = V(e), pa = V64(e), po = V64(e), pcase = V64(e), pb = V64(e);
    int a = V128(e), b = V64(e), c = V128(e), d = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pin, 0x380); ldc64(e, pout, 0x388);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X);
    imadw(e, pcase, cx, 1280u, pin); imadw(e, pa, tid, 16, pcase); imadw(e, pb, tid, 8, pcase);
    imadw(e, po, cx, 512u, pout); imadw(e, po, tid, 16, po);
    ldg_wide(e, BW_IR_LDG_E_128, a, pa, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_128, c, pa, 768, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_64, b, pb, 512, udesc, 0);
    qmma(e, d, a, b, mutant == 1 ? NOREG : c, 15);
    for (uint32_t i = 0; i < wait_nops; i++) nop(e, 15);
    for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = po; i.src2_vreg = d; i.src2_subreg = (uint8_t)q; i.imm = 4u * q; i.ureg = udesc; emit(e, i, K_STORE); }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_blackwell_codegen_qmma_peak_prog(uint32_t iters, uint32_t chains, uint32_t stall, uint32_t threads, BlackwellIRProgram *prog) {
    if (!prog || iters == 0 || chains == 0 || chains > 16 || stall == 0 || stall > 15 || threads == 0 || threads % 32 || threads > 1024) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    int udesc = UV64(e), py = V64(e), tid = V(e), cta = V(e), one = V(e), cnt = V(e), a = V128(e), b = V64(e), acc[16];
    for (uint32_t c = 0; c < chains; c++) acc[c] = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358); ldc64(e, py, 0x380); s2r(e, tid, BW_SR_TID_X); s2r(e, cta, BW_SR_CTAID_X);
    movi(e, one, 1);
    for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn m = I0(BW_IR_MOV_IMM); m.dst_vreg = a; m.dst_subreg = (uint8_t)q; m.imm = 0x38383838u; emit(e, m, K_FIXED); }
    for (unsigned q = 0; q < 2; q++) { BlackwellIRInsn m = I0(BW_IR_MOV_IMM); m.dst_vreg = b; m.dst_subreg = (uint8_t)q; m.imm = 0x38383838u; emit(e, m, K_FIXED); }
    for (uint32_t c = 0; c < chains; c++) for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn mz = I0(BW_IR_MOV_RZ); mz.dst_vreg = acc[c]; mz.dst_subreg = (uint8_t)q; emit(e, mz, K_FIXED); }
    movi(e, cnt, iters);
    const int L = here(e);
    loop_begin(e);
    for (uint32_t c = 0; c < chains; c++) qmma(e, acc[c], a, b, acc[c], stall);
    imadi(e, cnt, one, 0, 0xffffffffu, cnt);
    isetp_ge_u32(e, cnt, one);
    bra_if_p0(e, L);
    loop_end(e);
    nop(e, 15); nop(e, 15); nop(e, 15); nop(e, 15);
    {   int t = V(e); if (e->err) return -1;
        imadi(e, t, cta, 0, threads, tid); imadw(e, py, t, 16u * chains, py);
        for (uint32_t c = 0; c < chains; c++) for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn st = I0(BW_IR_STG_E); st.src1_vreg = py; st.src2_vreg = acc[c]; st.src2_subreg = (uint8_t)q; st.imm = c * 16u + 4u * q; st.ureg = udesc; emit(e, st, K_STORE); }
    }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_blackwell_codegen_f2e4_probe_prog(BlackwellIRProgram *prog) {
    if (!prog) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    int udesc = UV64(e), px = V64(e), py = V64(e), tid = V(e), cx = V(e), gi = V(e), x = V64(e), d = V(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358); ldc64(e, px, 0x380); ldc64(e, py, 0x388);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X);
    imadi(e, gi, cx, 0, 128u, tid);
    imadw(e, px, gi, 8u, px); imadw(e, py, gi, 4u, py);
    ldg_wide(e, BW_IR_LDG_E_64, x, px, 0, udesc, 1);
    f2e4(e, d, x, 0, x, 1);
    stg(e, py, 0, d, udesc);
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}
