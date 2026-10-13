#ifndef OMEGA_BLACKWELL_CODEGEN_TC_H
#define OMEGA_BLACKWELL_CODEGEN_TC_H
/*
 * omega_blackwell_codegen_tc: tensor-core (HMMA.16816.F32.BF16) kernels for the OWN-30B one-pass prefill (T3, 2026-10-13).
 * New file. Uses only opcodes the encoder already has and fixture-tests (omega_blackwell_codegen.h).
 *
 * HMMA.16816 fragment layout (lane L, g = L / 4, t = L % 4; register q holds element 2q in its low half, 2q+1 high):
 *   A 16x16 (row-major M x K): a0,a1 = A[g][2t, 2t+1]   a2,a3 = A[g+8][2t, 2t+1]   a4,a5 = A[g][2t+8, 2t+9]   a6,a7 = A[g+8][2t+8, 2t+9]
 *   B 16x8  (K x N):           b0,b1 = B[2t, 2t+1][g]   b2,b3 = B[2t+8, 2t+9][g]
 *   C/D 16x8 f32:              c0,c1 = C[g][2t, 2t+1]   c2,c3 = C[g+8][2t, 2t+1]
 * (PTX ISA mma.m16n8k16 fragment figures; the fragment-major matmul of omega_blackwell_codegen.h packs to the same order and
 * is chip-proven.)
 *
 * ---- probe kernel (characterisation of the chip's HMMA arithmetic) ----
 * One warp per CTA, ctaid.x = case. Per case, at in + case * 1280: A quads [32 lanes][4 u32] (512 B), B pairs [32][2] (256 B),
 * C quads [32][4] f32 (512 B). Output at out + case * 512: D quads [32][4] f32. Arguments: 0x380 in u64, 0x388 out u64.
 * D = A * B + C with one HMMA.16816.F32.BF16. mutant 1 (tests only): C is replaced by RZ (must be caught when C != 0).
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_TC_PROBE_IN_BYTES 1280u
#define OMEGA_TC_PROBE_OUT_BYTES 512u
int omega_blackwell_codegen_tc_probe_prog(int mutant, BlackwellIRProgram *prog);

#ifdef __cplusplus
}
#endif
#endif
