/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Foreign ABI packing only. The complete SM86 encoding implementation is
 * om/encoder/ampere_enc_sm86.om, translated by the pinned omega-bridge. */
#include "omega_ampere_codegen.h"
#include "cudapp_native.h"

int omega_gpu_encode_program_target(const BlackwellIRProgram *prog, uint32_t target_sm,
                                    uint8_t *code_buf, size_t max_len, size_t *out_len)
{
    if (!out_len) return -1;
    *out_len = 0;
    if (!prog || !code_buf || prog->count > BW_MAX_IR_INSNS) return -1;
    if (target_sm != 86 && target_sm != 121) return OMEGA_GPU_ENCODE_UNSUPPORTED;
    const size_t needed = omega_bw_program_code_bytes(prog);
    if (needed > max_len) return -1;
    if (target_sm == 121) return omega_bw_encode_program(prog, code_buf, max_len, out_len);

    uint64_t ins[8192], phys[288];
    /* Fixed native output extent; staging permits short caller buffers and
     * leaves them unchanged on a refused instruction or native trap. */
    uint8_t code[16384];
    for (size_t i = 0; i < prog->count; ++i) {
        const BlackwellIRInsn *in = &prog->insns[i];
        uint64_t *p = ins + i * 8;
        p[0] = (uint64_t)in->op;
        p[1] = (uint64_t)(int64_t)in->dst_vreg;
        p[2] = (uint64_t)(int64_t)in->src1_vreg;
        p[3] = (uint64_t)(int64_t)in->src2_vreg;
        p[4] = (uint64_t)(int64_t)in->src3_vreg;
        p[5] = (uint64_t)(int64_t)in->ureg;
        p[6] = (uint64_t)in->imm | ((uint64_t)in->control << 32);
        p[7] = (uint64_t)in->is_uniform | ((uint64_t)in->predicate_p0 << 1)
             | ((uint64_t)in->predicate_not << 2) | ((uint64_t)in->bar_reg << 8)
             | ((uint64_t)in->dst_subreg << 16) | ((uint64_t)in->src1_subreg << 24)
             | ((uint64_t)in->src2_subreg << 32) | ((uint64_t)in->src3_subreg << 40);
    }
    for (size_t i = 0; i < BW_MAX_VREGS; ++i)
        phys[i] = (uint64_t)(int64_t)prog->regalloc.vreg_to_phys[i];
    for (size_t i = 0; i < BW_MAX_UVREGS; ++i)
        phys[256 + i] = (uint64_t)(int64_t)prog->regalloc.uvreg_to_phys[i];
    uint64_t bytes = 0;
    uint32_t status = omega_ampere_enc_sm86(&bytes, ins, 8192, phys, 288,
                                          prog->count, 1, code, 16384);
    if (status != OMEGA_OK || bytes != needed) return OMEGA_GPU_ENCODE_UNSUPPORTED;
    for (size_t i = 0; i < needed; ++i) code_buf[i] = code[i];
    *out_len = (size_t)bytes;
    return 0;
}
