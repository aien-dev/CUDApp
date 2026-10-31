/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_BLACKWELL_CODEGEN_W4_H
#define OMEGA_BLACKWELL_CODEGEN_W4_H
/*
 * omega_blackwell_codegen_w4: NVFP4 weight-only (W4A16) grouped GEMV/GEMM for DECODE (P-DRA-4 DRA-71). New file beside
 * omega_blackwell_codegen_gemm.c (the FP8 grouped GEMM); same argument words, same row loop, pass loop and staging, a
 * different weight decode:
 *
 *   W[r][c] = e2m1(nibble c of row r) * e4m3(S[r][c / 16]) * G[r]
 *   y[row0 + i][r] = G'[r] * sum_c W'[r][c] * x'[row0 + i][c]
 *
 * Weights: k/2 bytes per row, byte c/2 holds column c in its low nibble when c is even (checked on the real
 * nvidia/Qwen3.6-35B-A3B-NVFP4 tensors by aien-sovereign-core-mv nvfp4.rs). Scales: k/16 raw e4m3 bytes per row.
 * G'[r] = weight_scale_2 * 2^62 per weight row (f32): a fused gate|up table carries gate's and up's globals.
 *
 * A lane owns LB weight bytes = EL = 2*LB columns per K step: KS = k / (32 * EL) steps. Each nibble is decoded in
 * registers to f32 bits (value * 2^-126; code 0.5 is an f32 denormal, so the FFMA must not flush: the FP8 kernels
 * already rely on that for e4m3 subnormals), dot = sum over 16 columns fma(f, x', dot) in column order, then once per
 * 16-block acc = fma(dot, s, acc) with s the exact e4m3 value, then the SHFL.DOWN tree 16..1, then y = fma(acc, G', 0).
 *
 * x' rows arrive PRE-PERMUTED by the host (omega_w4gemm_permute_x): column c = s*32*EL + L*EL + q*4 + e (step s,
 * lane L, quad q, element e) sits at index s*32*EL + (q*32 + L)*4 + e, scaled by 2^64. Shared memory holds one K step
 * of MT rows: MT * 32 * EL * 4 bytes <= 16384, so MT <= 4 at LB 16 and MT <= 8 at LB 8.
 * Scales are read as one u32 per lane per step (LB 16: a lane pair shares the word, two blocks per lane; LB 8: four
 * lanes share it, one block per lane); the lane's byte is moved to byte 3 (or bytes 2, 3) by a per-lane multiplier.
 *
 * Segment table (32 bytes per segment, ctaid.y = segment): { w_va u64, s_va u64, row0 u32, cnt u32, g_va u64 }.
 * Argument words: OMEGA_GEMM_ARG_* of omega_blackwell_codegen_gemm.h (seg, x', y, n, stride, XMAP, YMAP).
 *
 * mutant (tests only, each must be caught): 1 drops the block scale, 2 WRONG SCALE (the neighbouring block's byte),
 * 3 reads the neighbouring x row (m ^ 1), 4 drops the empty-segment guard, 5 swaps the nibble order, 6 drops G'.
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_codegen_gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_W4GEMM_SEG_BYTES 32u
#define OMEGA_W4GEMM_MUTANTS 6

typedef struct {
    uint32_t k;          /* multiple of 64 * lane_bytes, <= 16384 */
    uint32_t lane_bytes; /* 8 or 16 (16 when k % 1024 == 0) */
    uint32_t mt;         /* rows per pass: power of two, mt * 256 * lane_bytes <= 16384 */
    uint32_t warps;      /* power of two 1 .. 32, (2 * mt * lane_bytes) % warps == 0 */
    int mutant;          /* 0 .. OMEGA_W4GEMM_MUTANTS */
    uint32_t rowmap;     /* as OmegaFp8GemmSpec.rowmap; mt <= 4 */
} OmegaW4GemmSpec;

int omega_blackwell_codegen_w4gemm_prog(const OmegaW4GemmSpec *spec, BlackwellIRProgram *prog);
uint32_t omega_w4gemm_shared_bytes(const OmegaW4GemmSpec *spec); /* 0 when refused */
uint32_t omega_w4gemm_steps(const OmegaW4GemmSpec *spec);
int omega_w4gemm_prog_stats(const OmegaW4GemmSpec *spec, uint32_t *insns, uint32_t *gprs);
/* natural-order f32 row x (k) -> the kernel's permuted, 2^64-scaled row */
void omega_w4gemm_permute_x(const float *x, uint32_t k, uint32_t lane_bytes, float *out);

#ifdef __cplusplus
}
#endif
#endif
