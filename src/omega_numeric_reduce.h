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

#ifndef OMEGA_NUMERIC_REDUCE_H
#define OMEGA_NUMERIC_REDUCE_H

/*
 * E1 WP-D: general FP32 reductions with a frozen summation order
 * (docs/numeric/E1_REDUCTION_CONTRACT.md).
 *
 * The order depends only on n and the element index, never on scheduling,
 * thread count or CTA count:
 *
 *   level 0 values are x[0..n).
 *   One level turns len values into ceil(len/32) values: value j of the next
 *   level is the 32-lane warp tree (OMEGA_WARP_REDUCTION_DECLARED_ORDER,
 *   lane i combines with lane i+d for d = 16, 8, 4, 2, 1, lower lane on the
 *   left) over values [32j, 32j+32) of this level, positions >= len filled
 *   with the identity of the op.
 *   At least one level is applied; levels repeat until one value is left.
 *   The number of levels is the smallest L >= 1 with 32^L >= n.
 *
 * Every level is a pure tile-of-32 tree, so the order is exactly what the
 * chip-proven REDUCE_SUM warp kernel computes per tile, and what a future
 * 1024-thread CTA (32 warps: warp tree, then warp 0 tree over the 32 warp
 * results via shared memory) would compute for two levels.
 *
 * Build with -ffp-contract=off.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "omega_numeric.h"

#define OMEGA_REDUCE_DECLARED_ORDER \
    "RECURSIVE_TILE32_PAIRWISE_TREE_LANE_DELTA_16_8_4_2_1_PAD_IDENTITY_MIN_ONE_LEVEL"

typedef enum {
    OMEGA_RED_SUM = 0,  /* IEEE fadd, RNE, subnormals kept; identity -0.0       */
    OMEGA_RED_MAX,      /* FMNMX_MAX rule: NaN is missing data, -0 < +0;
                           identity: quiet NaN                                  */
    OMEGA_RED_MIN,      /* FMNMX_MIN rule, identity: quiet NaN                  */
    OMEGA_RED_MEAN,     /* omega_math_div(SUM, (float)n), 1 <= n <= 2^24        */
    OMEGA_RED_COUNT
} OmegaReduceOp;

/* Empty input (n = 0): SUM -> +0.0, MAX/MIN/MEAN -> quiet NaN 0x7fc00000. */
#define OMEGA_REDUCE_EMPTY_SUM_BITS   0x00000000U
#define OMEGA_REDUCE_EMPTY_OTHER_BITS 0x7fc00000U
#define OMEGA_REDUCE_SUM_PAD_BITS     0x80000000U  /* -0.0: x + -0 == x for all x */
#define OMEGA_REDUCE_MINMAX_PAD_BITS  0x7fc00000U  /* NaN: max(NaN, x) == x       */
#define OMEGA_REDUCE_MEAN_MAX_N       (1u << 24)   /* n exact in FP32             */

const char *omega_reduce_op_name(OmegaReduceOp op);
/* Padding value of one level (MEAN pads like SUM). */
float omega_reduce_identity(OmegaReduceOp op);
/* Levels applied for n elements: smallest L >= 1 with 32^L >= n (n = 0 -> 0). */
unsigned omega_reduce_levels(size_t n);

/*
 * Return codes are the OMEGA_NUMERIC_* codes of omega_numeric.h:
 * BAD_ARGS (unknown op, NULL buffer with n > 0), OPERANDS (MEAN with
 * n > 2^24, row_stride < n with rows > 1), FPENV (host FPCR not RNE/no FTZ).
 */

/* Semantic reference tier: written straight from the definition above (copy
 * each tile into 32 slots, pad, run the lane tree). */
int omega_reduce_reference(OmegaReduceOp op, const float *x, size_t n, float *out);

/* CPU realization tier: independent code (bottom-up in place, explicit
 * AArch64 FADD / FMAXNM / FMINNM / FDIV / UCVTF). */
int omega_reduce_cpu(OmegaReduceOp op, const float *x, size_t n, float *out);

/* Batched: out[r] = reduce(x[r*row_stride .. r*row_stride + n)) for r < rows,
 * each row with the same frozen order (reduce along the last axis). */
int omega_reduce_rows_reference(OmegaReduceOp op, const float *x, size_t rows, size_t n,
                                size_t row_stride, float *out);
int omega_reduce_rows_cpu(OmegaReduceOp op, const float *x, size_t rows, size_t n,
                          size_t row_stride, float *out);

/* NOT the contract. Left-to-right sum, used only by the negative test to
 * show that a different order gives different bits and is caught. */
float omega_reduce_sequential_sum_not_contract(const float *x, size_t n);

/*
 * GB10 realization (src/omega_numeric_reduce_gb10.c), all four ops. Each
 * level is padded with the op identity to a multiple of 32 on the host and run
 * through a warp kernel in chunks of at most OMEGA_NUMERIC_MAX_COUNT values;
 * the host only gathers lane 0 of each warp between launches (data movement,
 * no arithmetic).
 *   SUM       the chip-proven REDUCE_SUM patch words, own launch with a semaphore wait.
 *   MAX, MIN  this file's warp patch: five SHFL.DOWN (16..1) + FMNMX pairs.
 *   MEAN      SUM levels on the chip, then omega_math_div(SUM, u2f(n)) as the GB10
 *             DIV kernel (E1 row 7), one element. No host step.
 *
 * omega_reduce_gb10_check runs every pre-submission check (CHECK: markers)
 * without opening a device.
 */
#define OMEGA_REDUCE_GB10_MAX_N ((size_t)1 << 26)
int omega_reduce_gb10_check(OmegaReduceOp op, const float *x, size_t n, const float *out,
                            char *err, size_t err_len);
int omega_reduce_gb10(OmegaReduceOp op, const float *x, size_t n, float *out);
/* Launches the GB10 path made in the last omega_reduce_gb10 call. */
unsigned omega_reduce_gb10_last_launches(void);

/* MAX/MIN warp patch (12 instructions written at OMEGA_NUMERIC_PATCH_OFFSET
 * of the vecadd kernel). Returns the count or OMEGA_NUMERIC_ERR_NOT_ENCODED. */
int omega_reduce_gb10_minmax_patch(OmegaReduceOp op, OmegaNumericPatchInsn out[OMEGA_NUMERIC_PATCH_MAX]);
/* Structural check of a MAX/MIN patch against the QMD it runs in: the numeric
 * executor's generic rules plus deltas 16,8,4,2,1 in order, FMNMX form and
 * min/max predicate, operand order, waits, stall, store, exit. */
int omega_reduce_gb10_check_minmax_patch(OmegaReduceOp op, const OmegaNumericPatchInsn *p, int n,
                                         const uint32_t *qmd1, char *err, size_t err_len);
/* vecadd kernel with the MAX/MIN patch applied (what the executor submits). */
int omega_reduce_gb10_build_minmax_kernel(OmegaReduceOp op, uint8_t *code, size_t code_len, size_t *out_len);

#endif /* OMEGA_NUMERIC_REDUCE_H */
