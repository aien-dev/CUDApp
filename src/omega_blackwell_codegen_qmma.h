#ifndef OMEGA_BLACKWELL_CODEGEN_QMMA_H
#define OMEGA_BLACKWELL_CODEGEN_QMMA_H
/*
 * omega_blackwell_codegen_qmma: native FP8 tensor-core op QMMA.16832.F32.E4M3.E4M3 (OWN-30B L1 GEMM, W-L1G, 2026-10-14). New file.
 *
 * Encoding (the shared encoder lacks it): emitted as BW_IR_HMMA_BF16 with imm = OMEGA_QMMA_MARK and patched after encoding.
 * Oracle (nvcc 13.0.88 -arch=sm_121, investigations/2026-10-14-qmma-oracle/oracle.sass, C a real register):
 *   QMMA.16832.F32.E4M3.E4M3 R20, R4, R12, R20 = 0x0000000c0414727a / 0x004fe20000002c14
 *   HMMA.16816.F32.BF16      R8, R12, R6, R8   = 0x000000060c08723c / ...00041808 (encoder fixture)
 * Same register slots (Rd 16..23, Ra 24..31 of word 0, Rb word 1, Rc low byte of word 2); the patch sets the opcode low
 * bits 0x723c -> 0x727a and word 2 bits 0x00041800 -> 0x00002c00, keeping Rc.
 *
 * Fragment layout (INFERRED from the PTX ISA m16n8k32 8-bit figures, which Context7 does not return: DOCS SILENT there;
 * the chip probe checks it on exact integer cases). Lane L, g = L / 4, t = L % 4; byte j of a register is element j:
 *   A 16x32: reg0 = A[g][4t..4t+3]  reg1 = A[g+8][4t..4t+3]  reg2 = A[g][16+4t..]  reg3 = A[g+8][16+4t..]
 *   B 32x8:  reg0 = B[4t..4t+3][g]  reg1 = B[16+4t..16+4t+3][g]
 *   C/D f32: as HMMA.16816 (c0,c1 = C[g][2t, 2t+1], c2,c3 = C[g+8][2t, 2t+1]).
 *
 * Probe kernel: as omega_blackwell_codegen_tc_probe_prog (same 1280-byte case: A quads, B pairs, C quads; 512-byte D),
 * one QMMA per warp; mutant 1 drops C.
 * Peak kernel: as omega_blackwell_codegen_tc_peak_prog with QMMA (A = B = e4m3 1.0 = 0x38): every stored value 32 * iters.
 */
#include <stddef.h>
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif
#define OMEGA_QMMA_MARK 0x51303832u
/* F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C Rd, Ra, Rb, RZ: emitted as BW_IR_FMNMX_MAX with imm OMEGA_QMMA_MARK_F2E4; the patch sets
 * opcode 0x7209 -> 0x723e and word 2 0x07800000 -> 0x048070ff (oracle investigations/2026-10-14-qmma-oracle/cvt.sass:
 * F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R7, R7, R0, RZ = 0x000000000707723e / 0x004fca00048070ff). Result low 16 bits:
 * e4m3(Ra) << 8 | e4m3(Rb) (INFERRED from the chip-verified bf16 PACK_AB; the cvt probe checks), high 16 bits 0. */
#define OMEGA_QMMA_MARK_F2E4 0x46453433u
int omega_qmma_is_mark(const BlackwellIRInsn *in);
/* patch every QMMA placeholder in the encoded code (one 16-byte word per IR instruction); returns the number patched,
 * or -1 when a marked word does not hold the expected HMMA bits */
int omega_qmma_patch_code(const BlackwellIRProgram *prog, uint8_t *code, size_t len);
/* encoder self-test against the oracle words (register fields excluded); 0 = pass */
int omega_qmma_patch_selftest(char *msg, size_t msgn);
int omega_blackwell_codegen_qmma_probe_prog(int mutant, uint32_t wait_nops, BlackwellIRProgram *prog);
int omega_blackwell_codegen_qmma_peak_prog(uint32_t iters, uint32_t chains, uint32_t stall, uint32_t threads, BlackwellIRProgram *prog);
/* convert probe: thread i converts the f32 pair at arg0 + 8i (Ra = first, Rb = second) and stores the u32 result at arg1 + 4i */
int omega_blackwell_codegen_f2e4_probe_prog(BlackwellIRProgram *prog);
#ifdef __cplusplus
}
#endif
#endif
