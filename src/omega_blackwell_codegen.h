#ifndef OMEGA_BLACKWELL_CODEGEN_H
#define OMEGA_BLACKWELL_CODEGEN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "omega_blackwell_matmul.h"

#define BW_MAX_IR_INSNS         512
#define BW_MAX_VREGS            128
#define BW_MAX_UVREGS           32
#define BW_PHYS_GPR_START       2    /* Reserve R0, R1 */
#define BW_PHYS_GPR_MAX         128
#define BW_PHYS_UGPR_START      4    /* Reserve UR0-UR3 */
#define BW_PHYS_UGPR_MAX        32

/* Minimal Blackwell IR Opcodes for Stage 1 & Stage 2 */
typedef enum {
    BW_IR_NOP = 0,
    BW_IR_MOV_RZ,       /* MOV Rd, RZ (zero initialization) */
    BW_IR_MOV_IMM,      /* MOV Rd, imm32 */
    BW_IR_S2R,          /* S2R Rd, SR */
    BW_IR_LDC,          /* LDC Rd, c[bank][offset] */
    BW_IR_LDC64,        /* LDC.64 Rd:Rd+1, c[bank][offset] */
    BW_IR_LDCU,         /* LDCU URd, c[bank][offset] */
    BW_IR_LDCU64,       /* LDCU.64 URd:URd+1, c[bank][offset] */
    BW_IR_IMAD,         /* IMAD Rd, Ra, Rb/imm, Rc (multiply-add) */
    BW_IR_IMAD_WIDE,    /* IMAD.WIDE.U32 Rd:Rd+1, Ra, imm, Rc:Rc+1 */
    BW_IR_ISETP_GE,     /* ISETP.GE P0, PT, Ra, Rb/imm, PT */
    BW_IR_LDG_E,        /* LDG.E Rd, desc[URd][Ra.64] */
    BW_IR_STG_E,        /* STG.E desc[URd][Ra.64], Rb */
    BW_IR_IADD3,        /* IADD3 Rd, PT, PT, Ra, Rb, Rc */
    BW_IR_HMMA_F16,     /* HMMA.16816.F32 Rd, Ra, Rb, Rc (FP16 input, FP32 accumulator) */
    BW_IR_HMMA_BF16,    /* HMMA.16816.F32.BF16 Rd, Ra, Rb, Rc (BF16 input, FP32 accumulator) */
    BW_IR_SHF_R,        /* SHF.R.U32.HI Rd, RZ, imm, Ra */
    BW_IR_LOP3_AND,     /* LOP3.LUT Rd, Ra, imm, RZ, 0xc0, !PT (Bitwise AND) */
    BW_IR_LDG_E_U16,    /* LDG.E.U16 Rd, desc[URd][Ra.64] */
    BW_IR_EXIT,         /* EXIT */
    BW_IR_BRA           /* BRA target */
} BlackwellIROpcode;

/* Special Register Identifiers */
typedef enum {
    BW_SR_TID_X   = 0x21,
    BW_SR_TID_Y   = 0x22,
    BW_SR_CTAID_X = 0x25,
    BW_SR_CTAID_Y = 0x26
} BlackwellSR;

/* IR Instruction Node */
typedef struct {
    BlackwellIROpcode op;
    int dst_vreg;       /* Virtual destination register (-1 if none) */
    int src1_vreg;      /* Virtual source 1 register (-1 if none) */
    int src2_vreg;      /* Virtual source 2 register (-1 if none / immediate mode) */
    int src3_vreg;      /* Virtual source 3 register (-1 if none / RZ) */
    int ureg;           /* Uniform register (-1 if none) */
    uint32_t imm;       /* Immediate value / constant bank offset / SR code */
    uint32_t control;   /* Bundle control word */
    bool is_uniform;    /* True if targets/uses uniform registers */
    bool predicate_p0;  /* True if predicated on @P0 */
    uint8_t dst_subreg; /* Subregister offset (0..3) within register bundle */
    uint8_t src1_subreg;
    uint8_t src2_subreg;
    uint8_t src3_subreg;
} BlackwellIRInsn;

/* Live Interval for Bounded Linear Register Allocation */
typedef struct {
    int vreg;
    int first_def;
    int last_use;
    int phys_reg;
    uint32_t bundle_size; /* 1: scalar (32-bit), 2: pair (64-bit), 4: quad (128-bit) */
    bool is_pair;       /* True if 64-bit register pair (requires even-aligned R_2k) */
    bool is_quad;       /* True if 128-bit register quad (requires 4-aligned R_4k) */
    bool active;
} OmegaLiveInterval;

/* Register Allocator State */
typedef struct {
    int num_vregs;
    int num_uvregs;
    OmegaLiveInterval intervals[BW_MAX_VREGS];
    OmegaLiveInterval uintervals[BW_MAX_UVREGS];
    int vreg_to_phys[BW_MAX_VREGS];
    int uvreg_to_phys[BW_MAX_UVREGS];
    uint32_t peak_gpr_usage;
    uint32_t peak_ugpr_usage;
} OmegaRegAlloc;

/* IR Program */
typedef struct {
    BlackwellIRInsn insns[BW_MAX_IR_INSNS];
    size_t count;
    OmegaRegAlloc regalloc;
} BlackwellIRProgram;

/* Initializes IR program */
void omega_bw_ir_init(BlackwellIRProgram *prog);

/* Appends an instruction node to IR program */
int omega_bw_ir_append(BlackwellIRProgram *prog, const BlackwellIRInsn *insn);

/* Allocates a new 32-bit virtual general-purpose register */
int omega_bw_ir_alloc_vreg(BlackwellIRProgram *prog);

/* Allocates a new 64-bit aligned virtual register pair (for 64-bit memory addresses or HMMA B matrix) */
int omega_bw_ir_alloc_vreg64(BlackwellIRProgram *prog);

/* Allocates a new 128-bit aligned virtual register quad (for HMMA accumulator and A matrix) */
int omega_bw_ir_alloc_vreg128(BlackwellIRProgram *prog);

/* Allocates a new 32-bit virtual uniform register */
int omega_bw_ir_alloc_uvreg(BlackwellIRProgram *prog);

/* Allocates a new 64-bit aligned virtual uniform register pair */
int omega_bw_ir_alloc_uvreg64(BlackwellIRProgram *prog);

/* Solves bounded deterministic live-interval register allocation */
int omega_bw_regalloc_solve(BlackwellIRProgram *prog);

/* Encodes scheduled IR program into 128-bit sm_121 machine code */
int omega_bw_encode_program(const BlackwellIRProgram *prog, uint8_t *code_buf, size_t max_len, size_t *out_len);

/* Top-level dynamic code generator for integer matrix multiplication */
int omega_blackwell_codegen_matmul_i32(const OmegaMatMulSpec *spec, OmegaBlackwellKernel *kernel);

/* Dynamic sm_121 Tensor Core code generator (FP16 & BF16 via HMMA.16816) */
int omega_blackwell_codegen_matmul_tensor_prog(const OmegaMatMulSpec *spec, BlackwellIRProgram *prog);
int omega_blackwell_codegen_matmul_tensor(const OmegaMatMulSpec *spec, OmegaBlackwellKernel *kernel);

/* Unit test for instruction encoding bitfield fixtures (Gate 3) */
int omega_blackwell_verify_codegen_fixtures(void);

/* Unit test for bounded register allocation live intervals and bounds enforcement (Gate 4) */
int omega_blackwell_test_regalloc_bounds(void);

/* Stage-1 dynamic codegen variation demonstration between 16x16x16 and 32x16x64 (Gate 7) */
int omega_blackwell_test_codegen_variation(void);

#endif /* OMEGA_BLACKWELL_CODEGEN_H */
