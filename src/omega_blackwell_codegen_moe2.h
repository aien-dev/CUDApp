#ifndef OMEGA_BLACKWELL_CODEGEN_MOE2_H
#define OMEGA_BLACKWELL_CODEGEN_MOE2_H
/*
 * omega_blackwell_codegen_moe2: FP8 (e4m3) block-scaled GEMV, variant 2 ("row-pipelined"), OWN-30B
 * W-L2c (2026-10-14). New file: omega_blackwell_codegen_moe.{c,h} (variant 1) are untouched; the API
 * (omega_gpu_fp8_api.c) selects the variant (omega_gpu_fp8_set_variant / AIEN_OMEGA_FP8_GEMV=2).
 *
 * Same contract as variant 1 (same OmegaFp8GemvSpec, argument words, grid, thread mapping, lane
 * mapping, group table, x' / S' folding, mutants) and the SAME ARITHMETIC ORDER per output row:
 *   per lane, per K step: dot = FFMA chain over its lane_bytes weights (word 0..W-1, byte 0..3);
 *   acc_r = fma(dot, S'_step, acc_r) in K order; then SHFL.DOWN 16, 8, 4, 2, 1 with FADD.
 * so y is bit-identical to variant 1 for every input (checked by the simulator battery and on the chip).
 *
 * What changes is only WHEN the loads are issued (bytes in flight per warp):
 *   - a warp's unit of work is a whole row group (rows x k bytes): every weight load and scale load of
 *     the row group is issued back to back (rows * KS LDG.E.64/128 + KS scale words per lane);
 *   - two register sets: while the warp decodes and multiplies row group i (set A), the loads of row
 *     group i + stride (set B) are already in flight on their own scoreboard (SB3). Set A is filled by
 *     register moves from set B after a single wait on SB3, so the arithmetic never waits on the loads
 *     of the next row group (variant 1 waits for every load of a step before using any of them, and has
 *     nothing in flight while it computes);
 *   - x' (the activation row, shared by every row of the group) is staged once per CTA in shared memory
 *     (variant 1's deep-kernel staging, BAR.SYNC) and read per word with LDS.128 (4 registers), so the
 *     global loads a lane keeps in flight are weights and scales only.
 * Register sets: 2 x (rows * KS * lane_bytes / 4 + KS) registers, plus ~30. The spec is refused unless
 * rows * KS * lane_bytes <= 128 (instruction and register budgets) and k <= 512 * warps (staging rounds).
 * The register-file rule (32 * warps * declared <= 65536) is checked by the API as for variant 1; the
 * intended shapes are 16 warps per CTA (two CTAs per SM) at the OWN-30B widths k = 768, 2048, 4096.
 *
 * spec->depth is ignored (the whole row group is always in flight); spec->mutant 1, 2, 3 as variant 1.
 */
#include <stdint.h>
#include "omega_blackwell_codegen_moe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 0 on success (prog solved and register-allocated), -1 on a refused spec or an allocator failure. */
int omega_blackwell_codegen_fp8_gemv2_prog(const OmegaFp8GemvSpec *spec, BlackwellIRProgram *prog);

/* 1 if the spec is one this variant builds (shape rules above), 0 otherwise. */
int omega_fp8_gemv2_spec_ok(const OmegaFp8GemvSpec *spec);

/* Shared memory bytes the launch must declare: k * 4 (the staged x' row) + 16 (a dummy slot, only when the last staging round is partial). 0 when refused. */
uint32_t omega_fp8_gemv2_shared_bytes(const OmegaFp8GemvSpec *spec);

/* Hosted build: allocate and encode (omega_blackwell_kernel_free releases). */
int omega_blackwell_codegen_fp8_gemv2(const OmegaFp8GemvSpec *spec, OmegaBlackwellKernel *kernel);

/* The IR instruction count and peak GPR count of the solved program (for budgets in tests). */
int omega_fp8_gemv2_prog_stats(const OmegaFp8GemvSpec *spec, uint32_t *insns, uint32_t *gprs);

#ifdef __cplusplus
}
#endif
#endif
