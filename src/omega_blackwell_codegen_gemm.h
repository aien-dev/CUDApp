/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OMEGA_BLACKWELL_CODEGEN_GEMM_H
#define OMEGA_BLACKWELL_CODEGEN_GEMM_H
/*
 * omega_blackwell_codegen_gemm: FP8 e4m3 block-128-scaled grouped GEMM for PREFILL (OWN-30B, 2026-10-13, builder F
 * item 26). New file; it mirrors omega_blackwell_codegen_moe.c (the FP8 GEMV) so that its result is bit-identical to
 * that GEMV on the same row.
 *
 *   for every segment g (one expert, `cnt` rows of x'): y[row0 + i][r] = sum_c W_g[r][c] * x'[row0 + i][c] * S_g[r/128][c/128]
 *
 * Each distinct expert's weights are streamed once per pass of MT rows (one pass for the typical expert: 4 rows at 64
 * tokens), not once per token. A warp owns one weight row r and keeps MT f32 accumulators; per K step the lane decodes
 * its LB weight bytes once (the GEMV's exact bit trick) and FFMAs them into the MT rows' dot products, then
 * acc[m] = fma(dot[m], S', acc[m]) once per 128-block exactly as the GEMV, then the same SHFL.DOWN tree.
 *
 * x' rows arrive PRE-PERMUTED by the host (omega_gpu_fp8gemm_api): for a row, column c = s*32*LB + L*LB + w*4 + e
 * (K step s, lane L, word w, element e) sits at index s*32*LB + (w*32 + L)*4 + e, scaled by 2^64 as the GEMV. The CTA
 * stages one K step of MT rows (MT*32*LB*4 <= 16384 bytes: the QMD shared-memory limit of omega_blackwell_qmd.h) in
 * shared memory, so every lane reads its 4 columns of every row with one conflict-free LDS.128. Per step: stage,
 * BAR.SYNC, compute, BAR.SYNC. The weights and scales of step s+1 are loaded before step s computes (two register
 * sets, scoreboards 3 and 4 held until the step that uses them).
 *
 * Segment table (device, 32 bytes per segment, ctaid.y = segment): { w_va u64, s_va u64, row0 u32, cnt u32, pad[8] }.
 * Constant bank 0 argument words (cbank + 0x380):
 *   0x380 seg u64   segment table
 *   0x388 xp  u64   permuted x' rows, f32, k per row; MT padding rows must follow the last segment
 *   0x390 y   u64   y rows (f32, n each), row i = sorted row i
 *   0x398 n   u32   output rows per segment (weight rows)
 *   0x39c stride u32 weight rows per warp iteration (grid_x * warps)
 * Grid: grid_x CTAs per segment (each CTA: `warps` weight rows per iteration), grid_y = segments. All warps of a CTA
 * execute the same trip counts (rows past n are clamped to n-1 and not stored; rows past cnt in the last pass read
 * padding and are not stored), so the CTA barriers are uniform.
 *
 * mutant (tests only): 1 drops the block scale (acc += dot), 2 never advances the scale pointer (wrong when KS >= 2),
 * 3 reads the neighbouring row's x (m ^ 1) for every row. Each must be caught.
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_GEMM_ARG_SEG 0x380u
#define OMEGA_GEMM_ARG_X 0x388u
#define OMEGA_GEMM_ARG_Y 0x390u
#define OMEGA_GEMM_ARG_N 0x398u
#define OMEGA_GEMM_ARG_STRIDE 0x39cu
#define OMEGA_GEMM_ARGS_WORDS 8u
#define OMEGA_GEMM_SEG_BYTES 32u
#define OMEGA_GEMM_MAX_SHARED 16384u

typedef struct {
    uint32_t k;          /* multiple of 128 and of 32 * lane_bytes; <= 16384 */
    uint32_t lane_bytes; /* 8 or 16 (the GEMV's choice: 16 when k % 512 == 0, else 8) */
    uint32_t mt;         /* rows per pass: a power of two, mt * 128 * lane_bytes <= 16384 */
    uint32_t warps;      /* warps % lane_bytes == 0, (mt * lane_bytes) % warps == 0, 1 .. 32 (host simulator: build with -DWS_MAX_THREADS=512) */
    int mutant;
} OmegaFp8GemmSpec;

int omega_blackwell_codegen_fp8gemm_prog(const OmegaFp8GemmSpec *spec, BlackwellIRProgram *prog);
uint32_t omega_fp8gemm_shared_bytes(const OmegaFp8GemmSpec *spec); /* 0 when refused */
uint32_t omega_fp8gemm_steps(const OmegaFp8GemmSpec *spec);
int omega_fp8gemm_prog_stats(const OmegaFp8GemmSpec *spec, uint32_t *insns, uint32_t *gprs);

#ifdef __cplusplus
}
#endif
#endif
