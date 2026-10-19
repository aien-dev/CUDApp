#ifndef OMEGA_BLACKWELL_CODEGEN_MOE_H
#define OMEGA_BLACKWELL_CODEGEN_MOE_H
/*
 * omega_blackwell_codegen_moe: FP8 (e4m3) GEMV kernel family for the OWN-30B MoE lane
 * (Cut 6 + Cut 5, 2026-10-13). New file: nothing in omega_blackwell_codegen.{c,h} changes.
 *
 * One kernel: y[g][r] = sum_k W_g[r][k] * x[g or 0][k] * S_g[r/128][k/128], m = 1 (decode), for
 * G groups (experts) in one launch. Weights are e4m3 bytes in the checkpoint's row-major
 * [n][k] order (n output rows, k input columns), scales are the checkpoint's 128 x 128
 * `weight_scale_inv` grid, stored on the device as f32 (times 2^OMEGA_FP8_SCALE_EXP, see below).
 *
 * Numeric scheme (hand-checked bit arithmetic, proven exhaustively in tests/gpu_fp8_api_test.c):
 *   f32 bits ((b & 0x7F) << 20) | ((b & 0x80) << 24) equal e4m3(b) * 2^-120 exactly, for normals,
 *   subnormals and both zeros (bf16 field exponent 0 with the 3 mantissa bits shifted reproduces
 *   2^-6 * 0.M). The e4m3 NaN codes 0x7F / 0xFF decode wrong (a finite number): the upload refuses
 *   them. The 2^120 is folded on the host: x' = x * 2^OMEGA_FP8_X_EXP and S' = S * 2^OMEGA_FP8_SCALE_EXP
 *   (64 + 56 = 120), chosen so that every product w' * x' is a normal f32 for any practical x.
 *   The kernel accumulates in f32: per lane, a dot of lane_bytes weights (FFMA chain), then
 *   acc_r = fma(dot, S', acc_r) once per 128-block, then a warp reduction (SHFL.DOWN) per row.
 *
 * Thread mapping: a CTA has `warps` warps (threads_x = 32 * warps, threads_y = 1); warp w of CTA
 * (cx, cy) handles group cy and row groups rg = cx * warps + w, rg + stride, ... < RG (RG = n / rows,
 * stride = grid_x * warps, both launch arguments). A row group is `rows` consecutive rows, inside
 * one 128-row scale block (rows divides 128). Per K step a warp reads 32 * lane_bytes contiguous
 * bytes of each of its rows (lane L: bytes [L * lane_bytes, +lane_bytes)), the same lane_bytes
 * floats of x', and one scale word per lane (the block of its chunk; 128 / lane_bytes lanes share
 * it). Step count KS = k / (32 * lane_bytes) is baked into the kernel; k must be a multiple of
 * 32 * lane_bytes and of 128.
 *
 * Constant bank 0 argument words (cbank + 0x380, as the other APIs):
 *   0x380 tab   u64  group table: entry g at tab + 16 g = { w_va u64, s_va u64 }
 *   0x388 x     u64  x' rows (f32, k each; row g at x + g * x_stride)
 *   0x390 y     u64  y rows (f32, n each; row g at y + g * y_stride)
 *   0x398 RG    u32  row groups per group (n / rows)
 *   0x39c x_stride u32 bytes between x rows (0: one x row shared by every group)
 *   0x3a0 y_stride u32 bytes between y rows
 *   0x3a4 stride u32 row groups per warp iteration (grid_x * warps)
 * Grid: grid_x CTAs per group, grid_y = groups. The kernel does not depend on n, RG, the strides
 * or the grid, so one kernel per (k, lane_bytes, rows, warps, mutant) serves every n and G.
 *
 * Only lane 0 of a warp stores (R words per row group) inside a BSSY/BSYNC region; the warp is
 * whole again for the next row group's SHFLs.
 *
 * mutant (tests only, never production): 1 drops the block scale (acc += dot), 2 never advances the
 * scale pointer (every K step uses the step-0 block: wrong whenever KS >= 2), 3 reads the group
 * table entry of group cy ^ 1 (swapped bases). Each must be caught by the parity test.
 *
 * Instruction encodings: every IR op used here already exists in omega_blackwell_codegen.h; no new
 * encodings were added. ISETP_GE (immediate) and IADD3 carry fixed control words in the encoder,
 * so this generator uses ISETP_GE_U32 (register) and IMAD for adds instead.
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_FP8_X_EXP 64      /* x' = x * 2^64 (host, exact) */
#define OMEGA_FP8_SCALE_EXP 56  /* S' = S * 2^56 (host, exact for every finite bf16 scale below 2^72) */
#define OMEGA_FP8_BLOCK 128u    /* scale block (rows and columns) */
#define OMEGA_FP8_MAX_K 32768u /* the step kernel and the tensor-core kernel loop over k; 17408 is the Qwen3.8-27B down_proj (omega#541) */
#define OMEGA_FP8_MAX_GROUPS 64u /* group table entries per launch buffer (8 experts today) */

/* argument word offsets */
#define OMEGA_FP8_ARG_TAB 0x380u
#define OMEGA_FP8_ARG_X 0x388u
#define OMEGA_FP8_ARG_Y 0x390u
#define OMEGA_FP8_ARG_RG 0x398u
#define OMEGA_FP8_ARG_XSTRIDE 0x39cu
#define OMEGA_FP8_ARG_YSTRIDE 0x3a0u
#define OMEGA_FP8_ARG_STRIDE 0x3a4u
#define OMEGA_FP8_ARGS_WORDS 10u

typedef struct {
    uint32_t k;          /* input columns: multiple of 128 and of 32 * lane_bytes, <= OMEGA_FP8_MAX_K */
    uint32_t lane_bytes; /* 8 or 16 weight bytes per lane per step (LDG.E.64 / LDG.E.128); k must be a multiple of 256 */
    uint32_t rows;       /* rows per warp: 1, 2, 4 or 8 */
    uint32_t warps;      /* warps per CTA: 1 .. 32 (the host simulator runs up to 8) */
    uint32_t depth;      /* 0: one K step per loop iteration, x read from global memory each step.
                          * D >= 1: "deep" kernel: the CTA stages its group's x' row in shared memory once
                          * (k * 4 bytes, BAR.SYNC), then every loop iteration issues the weight and scale
                          * loads of D consecutive K steps before using any of them (D x more bytes in
                          * flight per warp); x comes from shared memory (LDS.128). Needs KS % D == 0,
                          * rows * D * lane_bytes <= 128, k % (32 * warps) == 0 and k <= 512 * warps (staging rounds). */
    int mutant;          /* 0 production; 1, 2, 3 as above; 4, 5 with coal = 1 (below) */
    uint32_t sched;      /* 0 (default, chip-proven): every fixed-latency instruction stalls 6, one byte decoded at a time.
                          * 1 (CUT5, measurement variant): the bytes of a step are decoded in a 3-deep software pipeline with
                          * their own temporaries, and a pass after register allocation (omega_fp8_schedule_stalls) gives each
                          * instruction the smallest stall that keeps every user >= OMEGA_FP8_FIXED_LAT cycles behind each
                          * fixed-latency write (the repo rule, omega_gpu_elementwise_api.c:13). The FFMA order is unchanged,
                          * so y is bit-identical to sched 0. Appended last: earlier fields keep their offsets. */
    uint32_t coal;       /* 0 (default): lane 0 of each warp stores its R row results (4-byte STG each) to y.
                          * 1 (CUT5 variant A, after omega#490 KT1): the CTA works in rounds (row groups uc .. uc + W - 1,
                          * uc uniform); lane 0 of each warp parks its R results in shared slots (two slot sets alternating by
                          * round, one BAR.SYNC per round) and warp 0 stores the round's W * R consecutive rows with one
                          * coalesced STG. A warp past the last row group recomputes row group RG - 1 and its slot is never
                          * stored. Needs warps * rows <= 32. Test mutants: 4 = one slot set (the round race), 5 = no clamp. */
    uint32_t wfirst;     /* 0 (default): each K step issues the x' loads, then the scale address IMADs, then the weight and
                          * scale loads. The IMADs wait on the x' loads' scoreboard (every fixed-latency op waits on the pending
                          * loads, emit()), so each step pays the x' round trip before any weight load goes out.
                          * 1 (CUT5 variant B): scale addresses first, then weight, scale and x' loads in one batch with no wait
                          * between them. Ignored by deep kernels (depth >= 1), which read x' from shared memory. */
} OmegaFp8GemvSpec;

#define OMEGA_FP8_FIXED_LAT 6u /* cycles a fixed-latency result needs before a user (the repo keeps 6; hardware minimum 5) */

/* 0 on success (prog solved and register-allocated), -1 on a refused spec or an allocator failure. */
int omega_blackwell_codegen_fp8_gemv_prog(const OmegaFp8GemvSpec *spec, BlackwellIRProgram *prog);

/* Hosted build: allocate and encode (omega_blackwell_kernel_free releases). */
int omega_blackwell_codegen_fp8_gemv(const OmegaFp8GemvSpec *spec, OmegaBlackwellKernel *kernel);

/* K steps the kernel runs: k / (32 * lane_bytes). 0 when the spec is refused. */
uint32_t omega_fp8_gemv_steps(const OmegaFp8GemvSpec *spec);

/* Shared memory bytes the launch must declare: k * 4 + 16 for a deep kernel, 0 otherwise, plus 2 * warps * rows * 4
 * (the two output slot sets) when coal = 1. */
uint32_t omega_fp8_gemv_shared_bytes(const OmegaFp8GemvSpec *spec);

/* The IR instruction count and peak GPR count of the solved program (for budgets in tests). */
int omega_fp8_gemv_prog_stats(const OmegaFp8GemvSpec *spec, uint32_t *insns, uint32_t *gprs);

#ifdef __cplusplus
}
#endif
#endif
