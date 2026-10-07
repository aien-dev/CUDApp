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
    BW_IR_STG_EF,       /* STG.E.EF: evict the line so the other side can see it */
    BW_IR_IADD3,        /* IADD3 Rd, PT, PT, Ra, Rb, Rc */
    BW_IR_HMMA_F16,     /* HMMA.16816.F32 Rd, Ra, Rb, Rc (FP16 input, FP32 accumulator) */
    BW_IR_HMMA_BF16,    /* HMMA.16816.F32.BF16 Rd, Ra, Rb, Rc (BF16 input, FP32 accumulator) */
    BW_IR_SHF_R,        /* SHF.R.U32.HI Rd, RZ, imm, Ra */
    BW_IR_LOP3_AND,     /* LOP3.LUT Rd, Ra, imm, RZ, 0xc0, !PT (Bitwise AND) */
    BW_IR_LDG_E_U16,    /* LDG.E.U16 Rd, desc[URd][Ra.64] */
    /* System-scope ordering. Encodings checked against the sm_121 disassembler.
     * A host build of these opcodes is not a graphics-chip result. */
    BW_IR_LDG_STRONG_SYS,
    BW_IR_LDG_MMIO,     /* LDG.E.MMIO.GPU: do not keep a stale cached copy */
    BW_IR_STG_STRONG_SYS,
    BW_IR_MEMBAR_ALL_SYS,
    BW_IR_MEMBAR_SC_SYS,
    BW_IR_CCTL_IVALL,
    BW_IR_ATOMG_ADD_STRONG_SYS,
    BW_IR_ATOMG_EXCH_STRONG_SYS,
    BW_IR_ISETP_GE_U32, /* ISETP.GE.U32.AND P0, PT, Ra, Rb, PT */
    BW_IR_LOP3_XOR,     /* LOP3.LUT Rd, Ra, Rb, RZ, 0x3c, !PT */
    BW_IR_FADD,         /* FADD Rd, Ra, Rb */
    BW_IR_FSUB,         /* FSUB Rd, Ra, Rb (FADD Rd, Ra, -Rb) */
    BW_IR_FMUL,         /* FMUL Rd, Ra, Rb */
    BW_IR_FFMA,         /* FFMA Rd, Ra, Rb, Rc (Rd = Ra * Rb + Rc) */
    BW_IR_FSETP,        /* FSETP.cond.AND P0, PT, Ra, Rb, PT */
    BW_IR_FSEL,         /* FSEL Rd, Ra, Rb, P0 */
    BW_IR_FMNMX_MIN,    /* FMNMX Rd, Ra, Rb, PT (minimum) */
    BW_IR_FMNMX_MAX,    /* FMNMX Rd, Ra, Rb, !PT (maximum) */
    BW_IR_I2FP,         /* I2FP.F32.S32 Rd, Ra */
    BW_IR_F2I,          /* F2I.TRUNC.NTZ Rd, Ra */
    BW_IR_MUFU_RCP,     /* MUFU.RCP Rd, Ra */
    BW_IR_MUFU_RSQ,     /* MUFU.RSQ Rd, Ra */
    BW_IR_SHFL_DOWN,    /* SHFL.DOWN PT, Rd, Ra, offset, 0x1f */
    BW_IR_LDS,          /* LDS Rd, [Ra] */
    BW_IR_STS,          /* STS [Ra], Rb */
    BW_IR_EXIT,         /* EXIT */
    BW_IR_BRA,          /* BRA. imm = signed instruction delta; predicate_p0 / predicate_not select @P0 or @!P0 */
    /* FB-1 cut 4 (2026-10-04), additive. Encodings decoded with nvdisasm 13.0 -b SM121
     * (tools/divsqrt_nvdisasm_check.sh method); the BAR/LDS/STS words are the ones
     * src/omega_numeric.c:884-886 runs on the chip (LDS_STS op, 148/148 specs PASS). */
    BW_IR_MUFU_EX2,     /* MUFU.EX2 Rd, Ra (2^x, approximate: PTX ex2.approx.f32 bound 2^-22.5 rel) */
    BW_IR_BAR_SYNC,     /* BAR.SYNC.DEFER_BLOCKING 0x0 (CTA barrier 0; the QMD declares 1 barrier) */
    BW_IR_LDS32,        /* LDS Rd, [Ra+URZ]  32-bit shared load  (BW_IR_LDS encodes LDS.U8: byte) */
    BW_IR_STS32,        /* STS [Ra+URZ], Rb  32-bit shared store (BW_IR_STS encodes STS.U8: byte) */
    /* Prime race cut (2026-10-05), additive, general integer ops. Words derived from the
     * register forms of LOP3_XOR / SHF_R / IMAD and matched word for word against
     * nvcc 13.0.88 -arch=sm_121 -cubin + cuobjdump -sass (offline oracle only; see
     * omega_blackwell_verify_codegen_fixtures_intops). For these three ops an operand
     * vreg of -1 encodes RZ (not R0). */
    BW_IR_LOP3_LUT,     /* LOP3.LUT Rd, Ra, Rb, Rc, imm8, !PT: Rd = LUT(Ra, Rb, Rc). imm & 0xff is the
                         * truth table over Ra = 0xf0, Rb = 0xcc, Rc = 0xaa (a|b = 0xfc, ~(a|b) = 0x03) */
    BW_IR_SHF_L_U32,    /* SHF.L.U32 Rd, Ra, Rb, RZ: Rd = Ra << Rb, register amount (the form nvcc emits
                         * for PTX shl.b32). Amounts >= 32 are NOT chip-verified: callers keep Rb < 32 */
    BW_IR_IMAD_HI_U32,  /* IMAD.HI.U32 Rd, Ra, Rb, Rc: Rd = hi32(Ra * Rb) + Rc mod 2^32 (nvcc emits this
                         * word for __umulhi(a, b) + c) */
    /* Structured warp reconvergence (omega #308, 2026-10-05), additive. Words matched against
     * nvcc 13.0.88 -arch=sm_121 output (BSSY.RECONVERGENT / BSYNC.RECONVERGENT; the oracle
     * kernels are listed in docs/gpu-reconvergence-308.md) and the Mesa NAK encoder
     * (sm70_encode.rs OpBSSy 0x945 / OpBSync 0x941, barrier register at bits 16..19).
     * bar_reg selects B0..B15. BSSY arms the barrier for the lanes executing it and names
     * the join point; BSYNC waits for every armed lane that has not exited, then the warp
     * continues as one. imm of BSSY = signed instruction delta to the join point (the
     * instruction after the BSYNC), like BW_IR_BRA; the encoder refuses deltas < 1. */
    BW_IR_BSSY,
    BW_IR_BSYNC,
    /* Fragment-major matmul (omega #328, 2026-10-07), additive. Words matched against nvcc 13.0.88
     * -arch=sm_121 (omega_blackwell_verify_codegen_fixtures_fragmm). imm = signed byte offset,
     * -2^23 .. 2^23-1; dst is a register pair (64) or an aligned quad (128). */
    BW_IR_LDG_E_64,     /* LDG.E.64 Rd:Rd+1, desc[URd][Ra.64+imm] */
    BW_IR_LDG_E_128     /* LDG.E.128 Rd..Rd+3, desc[URd][Ra.64+imm] */
} BlackwellIROpcode;

/* Barrier (reconvergence) registers: the sm_121 BSSY/BSYNC words carry a 4-bit barrier id
 * (NAK sm70_encode.rs set_bar_dst(16..20)), so B0..B15 are encodable. How many the chip
 * provides is not documented to us; chip tests nest to BW_RECONV_CHIP_TESTED_DEPTH. */
#define BW_RECONV_MAX_BAR 16

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
    uint32_t imm;       /* Immediate value / constant bank offset / SR code.
                         * For BW_IR_BRA: signed instruction delta (target minus
                         * this instruction). 0 branches to itself. */
    uint32_t control;   /* Bundle control word */
    bool is_uniform;    /* True if targets/uses uniform registers */
    bool predicate_p0;  /* True if predicated on P0 */
    bool predicate_not; /* With predicate_p0, encode @!P0 */
    uint8_t dst_subreg; /* Subregister offset (0..3) within register bundle */
    uint8_t src1_subreg;
    uint8_t src2_subreg;
    uint8_t src3_subreg;
    uint8_t bar_reg;    /* BW_IR_BSSY / BW_IR_BSYNC: barrier register B0..B15 */
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

/* FB-1 cut 1b: looped Tensor Core matmul. One warp per CTA. CTA (x, y) owns row
 * tile y (16 rows) and column tiles x, x + grid_x, x + 2 grid_x, ... (8 columns
 * each); the whole K dimension is accumulated in the HMMA accumulator on the
 * chip (no host K slicing). Requires M % 16 == 0, N % 8 == 0, K % 16 == 0 and
 * 1 <= grid_x <= N / 8. The kernel does not depend on M (rows come from
 * CTAID.Y), so one kernel serves every row count of a (K, N, grid_x) triple.
 * Uses only instructions already present in the chip-proven single-tile kernel
 * plus the predicated backward BRA and ISETP.GE this IR already encodes.
 * mutant != 0 drops the last K step (test oracle: the wrong kernel must be
 * caught by the parity test; never set in production). */
int omega_blackwell_codegen_matmul_tensor_loop_prog(const OmegaMatMulSpec *spec, uint32_t grid_x,
                                                    int mutant, BlackwellIRProgram *prog);
int omega_blackwell_codegen_matmul_tensor_loop(const OmegaMatMulSpec *spec, uint32_t grid_x,
                                               int mutant, OmegaBlackwellKernel *kernel);

/* omega #328: fragment-major tensor matmul. Same grid, C layout (row-major f32) and argument
 * words as the looped kernel; A and B are in fragment order instead of row-major:
 *   A (mp x kp, from omega_matmul_frag_pack_a): block (mt, ks) of 256 bf16 at ((mt*KS + ks)*32
 *     + lane)*8, lane = group*4 + tig, holding A[r][c], A[r][c+1], A[r+8][c], A[r+8][c+1],
 *     A[r][c+8], A[r][c+9], A[r+8][c+8], A[r+8][c+9] with r = 16mt + group, c = 16ks + 2tig;
 *   B (kp x np, from omega_matmul_frag_pack_b): block (nt, ks) of 128 bf16 at ((nt*KS + ks)*32
 *     + lane)*4 holding B[k][j], B[k+1][j], B[k+8][j], B[k+9][j] with k = 16ks + 2tig,
 *     j = 8nt + group;  KS = kp / 16, entries outside the real matrix are zero.
 * One LDG.E.128 (A) and one LDG.E.64 (B) per K step; `unroll` (1, 2, 4 or 8, dividing KS)
 * steps of loads are issued before the HMMA chain. mutant != 0 drops the last pass (test
 * oracle only). Each tile occupies 16*kp (A) and 8*kp (B) elements, so a row-tile offset into
 * A is the same in bytes as for row-major A. */
int omega_blackwell_codegen_matmul_frag_prog(const OmegaMatMulSpec *spec, uint32_t grid_x, uint32_t unroll,
                                             int mutant, BlackwellIRProgram *prog);
int omega_blackwell_codegen_matmul_frag(const OmegaMatMulSpec *spec, uint32_t grid_x, uint32_t unroll,
                                        int mutant, OmegaBlackwellKernel *kernel);
/* The largest of 4, 2, 1 that divides kp / 16 (unroll 8 is generated for experiments only: it
 * did not complete on the GB10, see the definition). */
uint32_t omega_matmul_frag_unroll(uint32_t kp);
/* a: m x k row-major. dst: mp x kp elements, fully written (zero padding). */
void omega_matmul_frag_pack_a(const uint16_t *a, uint32_t m, uint32_t k, uint32_t mp, uint32_t kp, uint16_t *dst);
/* b: k x n row-major. dst: the column tiles [nt0, nt1), (nt1 - nt0) * kp * 8 elements, fully
 * written; it is the part of the packed tensor starting at element nt0 * kp * 8. */
void omega_matmul_frag_pack_b(const uint16_t *b, uint32_t k, uint32_t n, uint32_t kp,
                              uint32_t nt0, uint32_t nt1, uint16_t *dst);

/* Unit test for instruction encoding bitfield fixtures (Gate 3) */
int omega_blackwell_verify_codegen_fixtures(void);

/* FB-1 cut 4: fixtures for MUFU_EX2, BAR_SYNC, LDS32, STS32 (0 = pass) */
int omega_blackwell_verify_codegen_fixtures_fb1cut4(void);

/* Prime race cut: golden words for LOP3_LUT, SHF_L_U32, IMAD_HI_U32 (0 = pass) */
int omega_blackwell_verify_codegen_fixtures_intops(void);

/* omega #328: golden words for LDG.E.64 / LDG.E.128 with offsets and the timed NOP (0 = pass) */
int omega_blackwell_verify_codegen_fixtures_fragmm(void);

/* omega #308: golden words for BSSY / BSYNC (nvcc 13.0.88 sm_121 oracle), plus the
 * encoder's refusals (negative delta, barrier id > 15). 0 = pass. */
int omega_blackwell_verify_codegen_fixtures_reconv(void);

/* omega #308: encodes one instruction with a given physical register map of
 * identity (vreg n -> Rn) so offline tools can disassemble single words. 0 = ok. */
int omega_blackwell_encode_one(const BlackwellIRInsn *insn, uint32_t w[4]);

/* Unit test for bounded register allocation live intervals and bounds enforcement (Gate 4) */
int omega_blackwell_test_regalloc_bounds(void);

/* Stage-1 dynamic codegen variation demonstration between 16x16x16 and 32x16x64 (Gate 7) */
int omega_blackwell_test_codegen_variation(void);

#endif /* OMEGA_BLACKWELL_CODEGEN_H */
