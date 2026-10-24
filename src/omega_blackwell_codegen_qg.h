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

#ifndef OMEGA_BLACKWELL_CODEGEN_QG_H
#define OMEGA_BLACKWELL_CODEGEN_QG_H
/*
 * omega_blackwell_codegen_qg: native FP8 prefill kernels (OWN-30B L1 GEMM, W-L1G, 2026-10-14). New file.
 *
 * ---- activation quantiser (pq) ----
 * Per row r and 128-wide group b of k:  amax = max(max_j |x[r][128b + j]|, 2^-100)
 *   E = biased f32 exponent of amax, M = its 23 mantissa bits, c = (M + 0x1fffff) >> 23  (1 when amax / 2^(E-127) > 1.75)
 *   scale = 2^(E - 135 + c)  (the smallest power of two with amax / scale <= 448), inv = 2^(135 - E - c)
 *   q[r][128b + j] = e4m3_rne_satfinite(x * inv)   (FMUL by a power of two: exact for every result that can be nonzero)
 *   s[r][b] = scale (f32)
 * Power-of-two scales instead of amax / 448 (stated deviation from the usual dynamic scheme): the GEMM's scale products
 * stay exact and no reciprocal is needed; e4m3 precision is uniform over its normal range, so at most the subnormal tail
 * of a group loses one bit.
 * Layout: x f32 or bf16 rows of k (row stride k elements); q e4m3 rows of k bytes; s f32 [rows][k / 128].
 * One CTA per row (ctaid.x = row), 32 * (k / 128) threads: warp b = group b, lane l = elements 4l .. 4l+3 (k <= 4096).
 * groups G > 0 (any k <= 32768, G <= 32 dividing k / 128): a row spans k / (128 G) CTAs along ctaid.y, 32 G threads each;
 * CTA y covers groups yG .. yG+G-1 (thread index offset by 32 G y). groups 0 is the one-CTA form (programs unchanged).
 * Arguments (cbank 0x380): x u64, q u64, s u64.
 * mutant (tests only): 1 skips the butterfly (lane-local amax), 2 uses c = 0, 3 drops the ctaid.y offset (groups form).
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif
enum { OMEGA_QG_IN_F32 = 0, OMEGA_QG_IN_BF16 = 1 };
typedef struct { uint32_t k, in_type; int mutant; uint32_t groups; } OmegaPqSpec;
#define OMEGA_PQ_ARGS_WORDS 6u
int omega_pq_spec_ok(const OmegaPqSpec *s);
int omega_blackwell_codegen_pq_prog(const OmegaPqSpec *s, BlackwellIRProgram *prog);

/* ---- grouped GEMM on QMMA (qgemm) ----
 *   y[row][c] = sum_kb sum_{h in 0,1} fma-promoted( T(row, c, kb, h) * (sa[src(row)][kb] * sw[c][kb]) )
 * where T is two chained QMMA steps from zero over the 64-byte chunk h of block kb: T = Q(v1, C = Q(v0, C = 0)), and after each
 * chunk acc = fma(T, sa * sw, acc) in f32 (sa a power of two, so sa * sw is exact). Step v of chunk h: lane t's 16 bytes
 * 64h + 16t .. +15 of both the activation row and the weight row; bytes 8v .. 8v+3 are logical k 4t .. 4t+3 of the QMMA and
 * bytes 8v+4 .. 8v+7 logical k 16+4t .. 16+4t+3 (A and B alike, so no K permutation is stored anywhere).
 * Weights: read IN PLACE from an omega_gpu_fp8_api tensor (no second copy): e4m3 rows of k in checkpoint order, scales
 * [n / 128][KB] f32 * 2^56 per expert; the kernel uses sw = S * 2^-56 (exact: the checkpoint bf16 scale). Activations: e4m3 rows of k (xq) + f32 scales
 * [rows][KB] (xs), the quantiser's output. Output F32: y f32 rows of n (warp columns n0w .. n0w+15). SWIGLU (n = 2I, gate rows
 * 0 .. I-1 then up rows I .. 2I-1, as the checkpoint): column tile 0 of a warp is gate rows n0w / 2 .. +7, tile 1 the up rows I later; act[row][n0w / 2 + 2t + q] = bf16_rn(g * rcp(2^min(-g log2 e, 125) + 1) * u), no extra scale
 * (the cap keeps the reciprocal normal: for g < -86.6 the result is g * 2^-125 * u instead of a flushed or subnormal value).
 * Tile entry (32 bytes): { w_va u64, s_va u64, row0 u32, cnt u32 (1 .. 16 rt), pad u64 }; rows past cnt are not stored.
 * Constant bank 0x380: tiles, rowidx (ignored when direct), xq, y, xs (u64 each), n u32 at 0x3a8.
 * Grid: n / (warps * 16) x tiles; 32 * warps threads; shared 2 * (16 rt * 64 + 16 rt * 4) bytes.
 * mutant (tests only): 1 drops sa (uses sw only), 2 never advances the sw pointer, 3 row tile i reads tile i^1, 4 drops step 1
 * of every odd chunk. */
enum { OMEGA_QG_EPI_F32 = 0, OMEGA_QG_EPI_SWIGLU = 1 };
typedef struct {
    uint32_t k;      /* multiple of 128, 128 .. 32768 */
    uint32_t rt;     /* row tiles of 16: 4 (8 exceeds 128 registers and the 16384-byte QMD shared cap) */
    uint32_t warps;  /* 8 or 16 */
    uint32_t epi;    /* OMEGA_QG_EPI_* */
    uint32_t direct; /* 1: no row index */
    uint32_t n;      /* output width (weight rows), multiple of warps * 16 */
    uint32_t wait;   /* stall cycles between a row tile's last QMMA and the first FFMA reading its result */
    int mutant;
} OmegaQgSpec;
#define OMEGA_QG_ARGS_WORDS 11u
#define OMEGA_QG_TILE_BYTES 32u
int omega_qg_spec_ok(const OmegaQgSpec *s);
uint32_t omega_qg_shared_bytes(const OmegaQgSpec *s);
uint32_t omega_qg_span(const OmegaQgSpec *s);
int omega_blackwell_codegen_qgemm_prog(const OmegaQgSpec *s, BlackwellIRProgram *prog);
#ifdef __cplusplus
}
#endif
#endif
