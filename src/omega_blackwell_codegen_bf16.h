#ifndef OMEGA_BLACKWELL_CODEGEN_BF16_H
#define OMEGA_BLACKWELL_CODEGEN_BF16_H
/*
 * omega_blackwell_codegen_bf16: bf16 streaming GEMV (the lm_head: vocab x hidden, bf16 weights, f32 x and y) for the
 * OWN-30B lane (2026-10-13). New file; it mirrors the structure of omega_blackwell_codegen_moe.c (the FP8 GEMV) with
 * no scales and no group table.
 *
 *   y[r] = sum_c W[r][c] * x[c],  W bf16 row-major [n][k], x f32 [k], y f32 [n], f32 accumulate.
 *
 * bf16 -> f32 is exact and free: element 2i of a 32-bit weight word is (word << 16), element 2i+1 is
 * (word & 0xFFFF0000). Thread mapping (as the FP8 kernel): a CTA has `warps` warps; warp w of CTA cx handles row
 * groups rg = cx * warps + w, rg + stride, ... < RG (RG = n / rows, a row group is `rows` consecutive rows). Per K
 * step a warp reads 512 contiguous bytes of each of its rows (lane L: one LDG.E.128 = 8 bf16 at byte 16 L); the CTA
 * stages the whole x row in shared memory once (k * 4 bytes, one BAR.SYNC) and every loop iteration issues the loads
 * of `depth` K steps of all its rows before using any of them. Per lane the 8 x depth x rows products of a row
 * accumulate in one FFMA chain (element order: step, word, low half, high half), then a SHFL.DOWN tree (16,8,4,2,1)
 * and lane 0 stores.
 *
 * Constant bank 0 argument words (cbank + 0x380):
 *   0x380 w   u64  weights base (row r at w + r * k * 2)
 *   0x388 x   u64  x row (f32, k)
 *   0x390 y   u64  y (f32, n)
 *   0x398 RG  u32  row groups (n / rows)
 *   0x39c stride u32 row groups per warp iteration (grid_x * warps)
 * Grid: grid_x CTAs, grid_y = 1. The kernel does not depend on n or the grid: one kernel per (k, rows, warps, depth,
 * mutant) serves every n. ROWS form (spec.ystride != 0, sc #642 prefill): grid_y token rows, row cy reads x + cy * k
 * floats and writes y + cy * ystride floats (the Qwen3.8-27B DeltaNet a|b projection of a prefill chunk).
 * XBATCH form (spec.xbatch 2..4, L312 stroke: the lm_head once for all heads): xbatch x rows (row m at x + m * k floats,
 * all staged in shared memory: k * 4 * xbatch bytes) share every weight read; each (weight row, x row) accumulates in the
 * SAME order as the one-row kernel, so every output row is bit-identical to the one-row GEMV of its own x; y row m
 * goes to y + m * ystride floats (ystride >= n, required). grid_y stays 1.
 *
 * mutant (tests only): 1 pairs the two halves of every word with the wrong x (low half takes the high half's x and
 * back), 2 never advances the x pointer (every step uses step 0's x: wrong whenever KS >= 2), 3 drops the last word
 * of each lane chunk. Each must be caught by the parity test.
 * 4 (XBATCH form only): every x row reads row 0's x (rows 1.. wrong). Must be caught by the xbatch parity test.
 * MASK form (spec.mask = 1, rows 1, one x row; HUM L314 exact two-pass head): before row r's K loop the warp loads
 * hi[r] (f32) and the threshold tau (one f32: the max of the screen's lower bounds, written by ARGMAX); a row with
 * hi[r] < tau stores -inf and skips its K loop, every other row runs the unchanged K loop and reduction, so its y[r]
 * is bit for bit the plain kernel's. No FSETP (never proven on the chip): h = hi + 0, t = tau + 0 (-0 becomes +0),
 * m = FMNMX.MAX(h, t), skip iff (m XOR h) != 0. hi and tau must not be NaN (the screen writes finite values).
 *
 * mutant (tests only): 1 pairs the two halves of every word with the wrong x (low half takes the high half's x and
 * back), 2 never advances the x pointer (every step uses step 0's x: wrong whenever KS >= 2), 3 drops the last word
 * of each lane chunk, 4 (MASK only) inverts the skip decision. Each must be caught by the parity test.
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_BF16_MAX_K 16384u
#define OMEGA_BF16_ARG_W 0x380u
#define OMEGA_BF16_ARG_X 0x388u
#define OMEGA_BF16_ARG_Y 0x390u
#define OMEGA_BF16_ARG_RG 0x398u
#define OMEGA_BF16_ARG_STRIDE 0x39cu
#define OMEGA_BF16_ARGS_WORDS 8u
#define OMEGA_BF16_ARG_HI 0x3a0u   /* MASK form: u64 hi (f32, n) */
#define OMEGA_BF16_ARG_TAU 0x3a8u  /* MASK form: u64 tau (one f32) */
#define OMEGA_BF16_MASK_ARGS_WORDS 12u
#define OMEGA_BF16_STEP_BYTES 512u /* weight bytes per row per K step (32 lanes x 16) */
#define OMEGA_BF16_MAX_XBATCH 4u    /* x rows per CTA in the XBATCH form (f32 rows staged in shared: 4 x 8 KiB at k 2048) */

typedef struct {
    uint32_t k;      /* multiple of 256 (one step) and of 32 * warps (exact staging rounds), <= OMEGA_BF16_MAX_K */
    uint32_t rows;   /* rows per warp: 1, 2 or 4 */
    uint32_t warps;  /* 1 .. 32 (the host simulator runs up to 8) */
    uint32_t depth;  /* K steps per loop iteration: divides k / 256; rows * depth <= 16 */
    int mutant;      /* 0 production; 1, 2, 3 as above */
    uint32_t ystride; /* 0: one x row (grid_y 1). Else the ROWS form: token row ctaid.y reads x + cy * k and writes
                       * y + cy * ystride (floats; ystride >= n). The default program is unchanged. */
    uint32_t xbatch; /* 0 or 1: one x row. 2..OMEGA_BF16_MAX_XBATCH: the XBATCH form (needs ystride >= n): x rows at
                      * x + m * k floats share every weight read, y row m at y + m * ystride. */
    uint32_t mask;    /* 0: every row. 1: the MASK form (rows 1, ystride 0). The default program is unchanged. */
} OmegaBf16GemvSpec;

int omega_blackwell_codegen_bf16_gemv_prog(const OmegaBf16GemvSpec *spec, BlackwellIRProgram *prog);
int omega_blackwell_codegen_bf16_gemv(const OmegaBf16GemvSpec *spec, OmegaBlackwellKernel *kernel);
/* K steps: k / 256 (0 when refused). */
uint32_t omega_bf16_gemv_steps(const OmegaBf16GemvSpec *spec);
/* Shared bytes the launch declares: k * 4. */
uint32_t omega_bf16_gemv_shared_bytes(const OmegaBf16GemvSpec *spec);
int omega_bf16_gemv_prog_stats(const OmegaBf16GemvSpec *spec, uint32_t *insns, uint32_t *gprs);

#ifdef __cplusplus
}
#endif
#endif
