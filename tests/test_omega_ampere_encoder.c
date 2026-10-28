/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#include "omega_ampere_codegen.h"
#include "omega_blackwell_codegen_tc.h"
#include "omega_blackwell_codegen_qmma.h"
#include <stdio.h>
#include <string.h>

static BlackwellIRProgram program;
static void init(void)
{
    omega_bw_ir_init(&program);
    for (unsigned i = 0; i < BW_MAX_VREGS; ++i) program.regalloc.vreg_to_phys[i] = (int)i;
    for (unsigned i = 0; i < BW_MAX_UVREGS; ++i) program.regalloc.uvreg_to_phys[i] = (int)i;
    program.regalloc.num_vregs = 256;
    program.regalloc.num_uvregs = 32;
    program.count = 1;
}
static BlackwellIRInsn fixture(unsigned op, unsigned variant)
{
    BlackwellIRInsn in = {0};
    in.op = (BlackwellIROpcode)op;
    in.dst_vreg = 4; in.src1_vreg = 8; in.src2_vreg = 12; in.src3_vreg = 16; in.ureg = 2;
    switch (op) {
    case BW_IR_S2R: in.imm = variant ? BW_SR_CTAID_Y : BW_SR_TID_X; break;
    case BW_IR_LDC: case BW_IR_LDC64: in.imm = 0x380 + variant * 8; break;
    case BW_IR_LDCU: case BW_IR_LDCU64: in.is_uniform = true; in.imm = 0x380 + variant * 8; break;
    case BW_IR_IMAD:
        if (variant == 1) { in.src2_vreg = -1; in.imm = 0x12345678; }
        if (variant == 2) { in.src2_vreg = -1; in.is_uniform = true; }
        break;
    case BW_IR_IMAD_WIDE: in.imm = 16; break;
    case BW_IR_ISETP_GE:
        in.src2_vreg = -1; in.imm = 17;
        if (variant) in.is_uniform = true;
        break;
    case BW_IR_STG_E: case BW_IR_LDG_E_64: case BW_IR_LDG_E_128: in.imm = variant ? (uint32_t)-16 : 16; break;
    case BW_IR_SHF_R: in.imm = variant ? 31 : 4; break;
    case BW_IR_LOP3_AND: in.imm = variant ? 0xffffffffu : 0xabcdef01; break;
    case BW_IR_SHFL_DOWN: in.imm = variant ? OMEGA_TC_MARK_BFLY | 16 : 1; break;
    case BW_IR_FMNMX_MAX: if (variant) in.imm = OMEGA_TC_MARK_F2FP; break;
    case BW_IR_LDS: case BW_IR_STS: case BW_IR_LDS128: in.imm = variant ? 64 : 16; break;
    case BW_IR_BRA: in.imm = variant ? (uint32_t)-2 : 2; in.predicate_p0 = variant != 0; in.predicate_not = variant == 2; break;
    case BW_IR_LOP3_LUT: in.imm = variant ? 0x96 : 0xe8; break;
    case BW_IR_BSSY: in.imm = 2; in.bar_reg = variant ? 15 : 0; break;
    case BW_IR_BSYNC: in.bar_reg = variant ? 15 : 0; break;
    default: break;
    }
    return in;
}
static int refused(uint32_t target)
{
    uint8_t code[16384]; size_t len = 99;
    memset(code, 0xa5, sizeof(code));
    if (omega_gpu_encode_program_target(&program, target, code, sizeof(code), &len) != OMEGA_GPU_ENCODE_UNSUPPORTED || len) return 1;
    for (size_t i = 0; i < sizeof(code); ++i) if (code[i] != 0xa5) return 1;
    return 0;
}
int main(void)
{
    uint8_t code[16384], legacy[16384];
    for (unsigned op = 0; op <= BW_IR_MUFU_LG2; ++op) {
        for (unsigned variant = 0; variant < 3; ++variant) {
            init(); program.insns[0] = fixture(op, variant);
            size_t len = 0, old_len = 0;
            /* The SM121 call triggers offline capture in the corpus build. */
            if (omega_bw_encode_program(&program, legacy, sizeof(legacy), &old_len) != 0 ||
                omega_gpu_encode_program_target(&program, 86, code, sizeof(code), &len) != 0 || len != old_len) {
                fprintf(stderr, "SM86 fixture refused op %u variant %u\n", op, variant); return 1;
            }
            if (omega_gpu_encode_program_target(&program, 121, code, sizeof(code), &len) != 0 ||
                len != old_len || memcmp(code, legacy, len)) return 1;
        }
    }
    /* Regression: is_uniform selects the IMAD source, not its GPR destination.
     * Nonidentity maps make the old uniform-destination mutation observable. */
    init(); program.insns[0] = fixture(BW_IR_IMAD, 2);
    program.regalloc.vreg_to_phys[4] = 12;
    program.regalloc.uvreg_to_phys[4] = 2;
    for (unsigned target = 0; target < 2; ++target) {
        size_t len = 0;
        if (omega_gpu_encode_program_target(&program, target ? 121 : 86, code, sizeof(code), &len) != 0 ||
            code[2] != 12) return 1;
    }
    /* The Blackwell descriptor preamble must not read past Ampere's CB0.
     * Lower it to a same-slot NOP, retaining every branch displacement. */
    {
        size_t len = 0, nop_len = 0;
        init(); program.insns[0] = fixture(BW_IR_NOP, 0);
        if (omega_gpu_encode_program_target(&program, 86, legacy, sizeof(legacy), &nop_len)) return 1;
        program.insns[0] = fixture(BW_IR_LDCU64, 0); program.insns[0].imm = 0x358;
        if (omega_gpu_encode_program_target(&program, 86, code, sizeof(code), &len) ||
            len != nop_len || memcmp(code, legacy, len)) return 1;
        if (omega_bw_encode_program(&program, legacy, sizeof(legacy), &len)) return 1;
    }
    init(); program.insns[0] = fixture(BW_IR_NOP, 0); if (refused(80)) return 1;
    program.count = 2; program.insns[1] = fixture(60, 0); if (refused(86)) return 1;
    init(); program.insns[0] = fixture(BW_IR_HMMA_BF16, 0); program.insns[0].imm = OMEGA_QMMA_MARK; if (refused(86)) return 1;
    { size_t len; if (omega_bw_encode_program(&program, legacy, sizeof(legacy), &len) != 0) return 1; }
    program.insns[0] = fixture(BW_IR_FMNMX_MAX, 0); program.insns[0].imm = OMEGA_QMMA_MARK_F2E4; if (refused(86)) return 1;
    { size_t len; if (omega_bw_encode_program(&program, legacy, sizeof(legacy), &len) != 0) return 1; }
    program.insns[0] = fixture(BW_IR_LDG_E, 0); program.insns[0].src1_vreg = 256; if (refused(86)) return 1;
    program.insns[0] = fixture(BW_IR_LDCU, 0); program.insns[0].dst_vreg = 32; if (refused(86)) return 1;
    program.insns[0] = fixture(BW_IR_MOV_IMM, 0); program.regalloc.vreg_to_phys[4] = -1;
    program.insns[0].dst_subreg = 1; if (refused(86)) return 1;
    init(); program.insns[0] = fixture(BW_IR_LDG_E_64, 0); program.insns[0].imm = 0x00800000; if (refused(86)) return 1;
    puts("Ampere encoder host fixtures: all IR opcodes, source modes, target preservation and refusals");
    return 0;
}
