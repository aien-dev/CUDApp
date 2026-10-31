/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_BLACKWELL_CODEGEN_FP8HEAD_H
#define OMEGA_BLACKWELL_CODEGEN_FP8HEAD_H
/*
 * omega_blackwell_codegen_fp8head: the FP8 (e4m3, 128 x 128 block-scaled) lm_head GEMV for up to
 * OMEGA_FP8HEAD_MAX_XB x rows in one launch (aien-dev/CUDApp#217, P-DRA-4 DRA-69). Each weight row is read once
 * per launch and dotted with every x row, so a batched decode stroke of N <= 8 conversations reads the
 * 0.509 GB FP8 head once instead of N times.
 *
 *   y[b][r] = sum_c W[r][c] * x[b][c] * S[r / 128][c / 128],   b < xb, r < n
 *
 * Arithmetic (bit-exact contract): per x row b and weight row r, this kernel performs the same f32 operations in
 * the same order as the single-x FP8 GEMV at lane_bytes 16 (omega_blackwell_codegen_moe.c, any rows / warps / depth /
 * sched / coal, which all share one arithmetic): lane L of the row's warp takes bytes [16 L, 16 L + 16) of each
 * 512-byte K step s; dot = 0; dot = fma(decode(w byte j), x'[512 s + 16 L + j], dot) for j = 0..15 (word-major);
 * acc = fma(dot, S'[block], acc); after the last step a SHFL.DOWN 16, 8, 4, 2, 1 FADD tree, lane 0 stores. So
 * y[b][r] is bit-identical to the single-x kernel's y[r] for x row b (tests/gpu_fp8_head_test.c proves it on the
 * warp simulator). The numeric scheme (decode = e4m3 * 2^-120, x' = x * 2^64, S' = S * 2^56) is moe.h's.
 *
 * Thread mapping: warp w of CTA cx handles row groups rg = cx * warps + w, rg + stride, ... < RG = n / rows; a row
 * group is `rows` consecutive weight rows (rows divides 128, so a group never crosses a scale block). Per K step
 * the warp loads its rows' 16-byte chunks and the scale word once, decodes them once (16 f32 per row per lane),
 * then for each x row b loads that row's 64 bytes of x' (4 LDG.128 through L1: x must be GPU-cached memory to be
 * cheap, see omega_gpu_fp8_head_api.h) and runs the FFMA chain of every row. x is not staged in shared memory:
 * 8 rows x 2048 f32 = 64 KiB exceed OMEGA_BW_QMD_MAX_SHARED_BYTES (32 KiB).
 *
 * Constant bank 0 argument words (cbank + 0x380; the layout is this kernel's own, not OMEGA_FP8_ARG_*):
 *   0x380 w       u64  e4m3 weights [n][k] (omega_gpu_fp8_tensor_entry)
 *   0x388 s       u64  f32 scales * 2^56 [n / 128][k / 128]
 *   0x390 x       u64  x' rows: row b at x + b * k * 4 (f32, already * 2^64; contiguous rows)
 *   0x398 y       u64  y rows: row b at y + b * y_stride
 *   0x3a0 RG      u32  row groups (n / rows)
 *   0x3a4 y_stride u32 bytes between y rows
 *   0x3a8 stride  u32  row groups per warp iteration (grid_x * warps)
 * Grid: grid_x CTAs, grid_y = 1, threads_x = 32 * warps, no shared memory. The kernel does not depend on n, RG,
 * y_stride or the grid: one kernel per (k, rows, warps, xb, mutant).
 *
 * mutant (tests only, never production): 1 never advances the scale pointer (every K step uses the step-0 k-block:
 * the wrong-scale-block negative control, wrong whenever k >= 256), 2 reads the neighbouring 128-row scale block
 * (row block ^ 1), 3 makes every x row read x row 0 (wrong whenever xb >= 2 and the rows differ).
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_FP8HEAD_MAX_XB 8u
#define OMEGA_FP8HEAD_LANE_BYTES 16u /* the single-x kernel's lane width whose arithmetic this kernel reproduces */
#define OMEGA_FP8HEAD_ARG_W 0x380u
#define OMEGA_FP8HEAD_ARG_S 0x388u
#define OMEGA_FP8HEAD_ARG_X 0x390u
#define OMEGA_FP8HEAD_ARG_Y 0x398u
#define OMEGA_FP8HEAD_ARG_RG 0x3a0u
#define OMEGA_FP8HEAD_ARG_YSTRIDE 0x3a4u
#define OMEGA_FP8HEAD_ARG_STRIDE 0x3a8u
#define OMEGA_FP8HEAD_ARGS_WORDS 11u

typedef struct {
    uint32_t k;     /* input columns: a multiple of 512 (32 lanes x 16 bytes), <= OMEGA_FP8_MAX_K */
    uint32_t rows;  /* consecutive weight rows per warp: 1 or 2 */
    uint32_t warps; /* warps per CTA, 1..32 */
    uint32_t xb;    /* x rows per launch, 1..OMEGA_FP8HEAD_MAX_XB */
    int mutant;     /* 0; tests only: 1..3 (see above) */
} OmegaFp8HeadSpec;

/* 1 when the generator accepts the spec (shape rules above, immediate offsets inside the 24-bit LDG field). */
int omega_fp8head_spec_ok(const OmegaFp8HeadSpec *spec);
/* Builds the register-allocated IR program into *prog. 0 ok, -1 refused or over BW_MAX_IR_INSNS / the registers. */
int omega_blackwell_codegen_fp8head_prog(const OmegaFp8HeadSpec *spec, BlackwellIRProgram *prog);
/* Instruction and peak-register counts (host, no device). 0 ok, -1 refused. */
int omega_fp8head_prog_stats(const OmegaFp8HeadSpec *spec, uint32_t *insns, uint32_t *gprs);

#ifdef __cplusplus
}
#endif
#endif
