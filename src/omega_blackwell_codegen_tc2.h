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

#ifndef OMEGA_BLACKWELL_CODEGEN_TC2_H
#define OMEGA_BLACKWELL_CODEGEN_TC2_H
/*
 * omega_blackwell_codegen_tc2: tensor-core grouped FP8 GEMM v2 for the device-resident prefill (OWN-30B T3b, 2026-10-13).
 * New file. Same fragment layout, weight decode, activation storage ((0,2,1,3)-permuted bf16 * 2^64) and A staging as
 * cut 1 (omega_blackwell_codegen_tc.h); differences:
 *  - big tiles: up to 16 warps x 16 columns x 128 rows (rt 8, ct 2: 64 accumulator registers per lane);
 *  - the block scale is folded through the accumulator chain: per 128-wide K block kb, acc = acc * R[kb], then the 8 HMMA
 *    steps accumulate straight into acc (no temporary); after the last block acc = acc * R[KB]. R is a per-expert f32
 *    table per 8-row group g of the PHYSICAL weight rows: R[g][0] = 1, R[g][kb] = S'[kb-1] / S'[kb] (RN), R[g][KB] =
 *    S'[KB-1], where S' = scale_inv * 2^56 of the 128-block holding the group's source rows. Stored as pairs
 *    [n/16][KB+1][2] (the warp's two column tiles side by side, one LDG.64 per block: plain LDG.E has no immediate offset);
 *  - rows >= cnt of a partial tile are not stored (one BSSY, forward jumps, one BSYNC);
 *  - epilogue F32: y[row][c] f32 rows of n; epilogue SWIGLU (ct = 2): column tile 0 of each warp is gate, tile 1 up (the
 *    upload interleaves gate and up rows in groups of 8), the lane computes a = g / (1 + ex2(g * -log2e)) * u * 2^64
 *    (MUFU.EX2, FADD, MUFU.RCP, FMUL, FMUL, FMUL x 2^64, in that order; MUFU flushes a subnormal result to +-0, so for
 *    g in (-88.72, -87.34), where 1 / (2^t + 1) is subnormal, the activation is +-0: PTX ISA 9.7.3.13 ftz, chip-observed in
 *    evidence OWN30B-L1G-PFGEMM-SWIGLU-FTZ-df9bd9c4) and stores bf16x2 (F2FP RN) at act column n0w/2 + 2t of the
 *    act row (n/2 bf16): with the upload's (0,2,1,3) output-row permutation the act row is directly the next GEMM's
 *    stored activation row;
 *  - direct rows (no row index): the source row of sorted row r is r itself (the down GEMM reading the act rows).
 *
 * Tile table entry (32 bytes, ctaid.y): { w_va u64 (expert's n x k e4m3), r_va u64 (expert's (n/8) x (KB+1) f32), row0 u32,
 * cnt u32 (1 .. 16*rt), pad u64 }. Rows row0 .. row0 + 16*rt - 1 of the row index must be readable (host pads).
 * Constant bank 0 (cbank + 0x380): tiles u64, rowidx u64 (ignored when direct), x u64, y u64, n u32.
 * Grid: grid_x = n / span (span = warps * 8 * ct), grid_y = tiles; CTA 32 * warps threads, shared 2 * 16 * rt * 128 bytes.
 * mutant (tests only): 1 skips the per-block ratio multiply, 2 never advances the ratio pointer, 3 row tile i reads tile
 * i^1, 4 drops the last 16-element step of every 128 block.
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif
enum { OMEGA_TC2_EPI_F32 = 0, OMEGA_TC2_EPI_SWIGLU = 1 };
typedef struct {
    uint32_t k;      /* multiple of 128, 128 .. 16384 */
    uint32_t rt;     /* row tiles of 16 per CTA: 2, 4 or 8 */
    uint32_t ct;     /* column tiles of 8 per warp: 2 (the paired ratio load) */
    uint32_t warps;  /* 4, 8 or 16 */
    uint32_t aqn;    /* A fragments in flight: 1 or 2 */
    uint32_t epi;    /* OMEGA_TC2_EPI_* */
    uint32_t direct; /* 1: no row index */
    uint32_t n;      /* output width (weight rows), a multiple of span; fixes the store offsets */
    int mutant;
} OmegaTc2Spec;
#define OMEGA_TC2_ARGS_WORDS 9u
#define OMEGA_TC2_TILE_BYTES 32u
int omega_tc2_spec_ok(const OmegaTc2Spec *s);
uint32_t omega_tc2_shared_bytes(const OmegaTc2Spec *s);
uint32_t omega_tc2_span(const OmegaTc2Spec *s);
int omega_blackwell_codegen_tc2_prog(const OmegaTc2Spec *s, BlackwellIRProgram *prog);
#ifdef __cplusplus
}
#endif
#endif
