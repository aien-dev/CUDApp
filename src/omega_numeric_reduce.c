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

/*
 * E1 WP-D: general reductions with a frozen order. Reference tier and CPU
 * realization tier. Host only; opens no device. See omega_numeric_reduce.h
 * and docs/numeric/E1_REDUCTION_CONTRACT.md.
 */
#include "omega_numeric_reduce.h"
#include "omega_numeric.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__aarch64__)
#error "E1 reduction CPU tier uses AArch64 FP instructions (fadd, fmaxnm, fminnm, fdiv, ucvtf)"
#endif

static const char *const RED_NAMES[OMEGA_RED_COUNT] = { "SUM", "MAX", "MIN", "MEAN" };

const char *omega_reduce_op_name(OmegaReduceOp op) {
    return ((unsigned)op < OMEGA_RED_COUNT) ? RED_NAMES[op] : NULL;
}

float omega_reduce_identity(OmegaReduceOp op) {
    if (op == OMEGA_RED_MAX || op == OMEGA_RED_MIN) return omega_bits_to_float(OMEGA_REDUCE_MINMAX_PAD_BITS);
    return omega_bits_to_float(OMEGA_REDUCE_SUM_PAD_BITS);
}

unsigned omega_reduce_levels(size_t n) {
    if (n == 0) return 0;
    unsigned l = 1;
    size_t cap = 32;
    while (cap < n) { cap = (cap > ((size_t)-1) / 32) ? (size_t)-1 : cap * 32; l++; }
    return l;
}

static float empty_result(OmegaReduceOp op) {
    return omega_bits_to_float(op == OMEGA_RED_SUM ? OMEGA_REDUCE_EMPTY_SUM_BITS : OMEGA_REDUCE_EMPTY_OTHER_BITS);
}

static int common_checks(OmegaReduceOp op, const float *x, size_t n, const float *out) {
    if ((unsigned)op >= OMEGA_RED_COUNT || !out || (n > 0 && !x)) return OMEGA_NUMERIC_ERR_BAD_ARGS;
    if (op == OMEGA_RED_MEAN && n > OMEGA_REDUCE_MEAN_MAX_N) return OMEGA_NUMERIC_ERR_OPERANDS;
    if (n > ((size_t)-1) / sizeof(float) - 64) return OMEGA_NUMERIC_ERR_OPERANDS; /* padding must not wrap */
    if (!omega_numeric_fpenv_ok()) return OMEGA_NUMERIC_ERR_FPENV;
    return OMEGA_NUMERIC_OK;
}

/* ---- Reference tier ------------------------------------------------------ */

/* The lane tree of omega_warp_reduce_sum with the combine of op. SUM calls
 * omega_warp_reduce_sum itself, so the tile order is the declared warp order. */
static float ref_tile(OmegaReduceOp op, const float tile[32]) {
    if (op == OMEGA_RED_SUM || op == OMEGA_RED_MEAN) return omega_warp_reduce_sum(tile);
    float val[32];
    memcpy(val, tile, sizeof(val));
    static const int deltas[5] = { 16, 8, 4, 2, 1 };
    for (int s = 0; s < 5; s++) {
        int d = deltas[s];
        for (int i = 0; i + d < 32; i++)
            val[i] = (op == OMEGA_RED_MAX) ? omega_ref_fmax(val[i], val[i + d]) : omega_ref_fmin(val[i], val[i + d]);
    }
    return val[0];
}

static int ref_sum_like(OmegaReduceOp op, const float *x, size_t n, float *out) {
    if (op == OMEGA_RED_MEAN) op = OMEGA_RED_SUM; /* MEAN reduces like SUM; the division is the caller's */
    if (n == 0) { *out = empty_result(op); return OMEGA_NUMERIC_OK; }
    const float pad = omega_reduce_identity(op);
    size_t len = n;
    const float *cur = x;
    float *owned = NULL;
    do {
        size_t next_len = (len + 31) / 32;
        float *next = malloc(next_len * sizeof(float));
        if (!next) { free(owned); return OMEGA_NUMERIC_ERR_BAD_ARGS; }
        for (size_t j = 0; j < next_len; j++) {
            float tile[32];
            for (size_t k = 0; k < 32; k++) {
                size_t idx = j * 32 + k;
                tile[k] = (idx < len) ? cur[idx] : pad;
            }
            next[j] = ref_tile(op, tile);
        }
        free(owned);
        owned = next;
        cur = next;
        len = next_len;
    } while (len > 1);
    *out = cur[0];
    free(owned);
    return OMEGA_NUMERIC_OK;
}

int omega_reduce_reference(OmegaReduceOp op, const float *x, size_t n, float *out) {
    int rc = common_checks(op, x, n, out);
    if (rc != OMEGA_NUMERIC_OK) return rc;
    if (op != OMEGA_RED_MEAN) return ref_sum_like(op, x, n, out);
    if (n == 0) { *out = empty_result(op); return OMEGA_NUMERIC_OK; }
    float s;
    rc = ref_sum_like(OMEGA_RED_SUM, x, n, &s);
    if (rc != OMEGA_NUMERIC_OK) return rc;
    *out = omega_math_div(s, omega_ref_u2f((uint32_t)n));
    return OMEGA_NUMERIC_OK;
}

/* ---- CPU realization tier ------------------------------------------------- */

static inline float cpu_combine(OmegaReduceOp op, float a, float b) {
    float r;
    if (op == OMEGA_RED_SUM) {
        __asm__ volatile("fadd %s0, %s1, %s2" : "=w"(r) : "w"(a), "w"(b));
        return r;
    }
    /* FMAXNM/FMINNM return NaN for a signaling NaN; Omega treats every NaN as
     * missing data, so quiet signaling NaNs first (as the FMNMX CPU tier does). */
    float qa = omega_isnan(a) ? omega_bits_to_float(omega_float_to_bits(a) | 0x00400000U) : a;
    float qb = omega_isnan(b) ? omega_bits_to_float(omega_float_to_bits(b) | 0x00400000U) : b;
    if (op == OMEGA_RED_MAX) __asm__ volatile("fmaxnm %s0, %s1, %s2" : "=w"(r) : "w"(qa), "w"(qb));
    else                     __asm__ volatile("fminnm %s0, %s1, %s2" : "=w"(r) : "w"(qa), "w"(qb));
    return r;
}

/* One buffer, compacted in place: tile j's result goes to buf[j] (j <= 32j). */
static int cpu_sum_like(OmegaReduceOp op, const float *x, size_t n, float *out) {
    if (op == OMEGA_RED_MEAN) op = OMEGA_RED_SUM; /* MEAN reduces like SUM; the division is the caller's */
    if (n == 0) { *out = empty_result(op); return OMEGA_NUMERIC_OK; }
    size_t cap = (n + 31) & ~(size_t)31;
    float *buf = malloc(cap * sizeof(float));
    if (!buf) return OMEGA_NUMERIC_ERR_BAD_ARGS;
    memcpy(buf, x, n * sizeof(float));
    const uint32_t pad_bits = (op == OMEGA_RED_SUM) ? OMEGA_REDUCE_SUM_PAD_BITS : OMEGA_REDUCE_MINMAX_PAD_BITS;
    size_t len = n;
    for (;;) {
        size_t padded = (len + 31) & ~(size_t)31;
        for (size_t i = len; i < padded; i++) buf[i] = omega_bits_to_float(pad_bits);
        size_t tiles = padded / 32;
        for (size_t j = 0; j < tiles; j++) {
            float *t = buf + j * 32;
            for (unsigned d = 16; d >= 1; d >>= 1)
                for (unsigned i = 0; i < d; i++) t[i] = cpu_combine(op, t[i], t[i + d]);
            buf[j] = t[0];
        }
        len = tiles;
        if (len == 1) break;
    }
    *out = buf[0];
    free(buf);
    return OMEGA_NUMERIC_OK;
}

int omega_reduce_cpu(OmegaReduceOp op, const float *x, size_t n, float *out) {
    int rc = common_checks(op, x, n, out);
    if (rc != OMEGA_NUMERIC_OK) return rc;
    if (op != OMEGA_RED_MEAN) return cpu_sum_like(op, x, n, out);
    if (n == 0) { *out = empty_result(op); return OMEGA_NUMERIC_OK; }
    float s, fn, r;
    rc = cpu_sum_like(OMEGA_RED_SUM, x, n, &s);
    if (rc != OMEGA_NUMERIC_OK) return rc;
    uint32_t n32 = (uint32_t)n;
    __asm__ volatile("ucvtf %s0, %w1" : "=w"(fn) : "r"(n32));
    __asm__ volatile("fdiv %s0, %s1, %s2" : "=w"(r) : "w"(s), "w"(fn));
    *out = r;
    return OMEGA_NUMERIC_OK;
}

/* ---- Batched (last axis) -------------------------------------------------- */

static int rows_common(OmegaReduceOp op, const float *x, size_t rows, size_t n, size_t row_stride,
                       const float *out) {
    if ((unsigned)op >= OMEGA_RED_COUNT || (rows > 0 && !out) || (rows > 0 && n > 0 && !x))
        return OMEGA_NUMERIC_ERR_BAD_ARGS;
    if (rows > 1 && row_stride < n) return OMEGA_NUMERIC_ERR_OPERANDS;
    if (rows > 0 && n > 0) {
        /* last element index (rows-1)*row_stride + n-1 and its byte offset must not wrap */
        size_t lim = SIZE_MAX / sizeof(float);
        if (rows > 1 && row_stride > (lim - n) / (rows - 1)) return OMEGA_NUMERIC_ERR_OPERANDS;
        if (n > lim) return OMEGA_NUMERIC_ERR_OPERANDS;
    }
    return OMEGA_NUMERIC_OK;
}

int omega_reduce_rows_reference(OmegaReduceOp op, const float *x, size_t rows, size_t n,
                                size_t row_stride, float *out) {
    int rc = rows_common(op, x, rows, n, row_stride, out);
    for (size_t r = 0; rc == OMEGA_NUMERIC_OK && r < rows; r++)
        rc = omega_reduce_reference(op, n ? x + r * row_stride : NULL, n, &out[r]);
    return rc;
}

int omega_reduce_rows_cpu(OmegaReduceOp op, const float *x, size_t rows, size_t n,
                          size_t row_stride, float *out) {
    int rc = rows_common(op, x, rows, n, row_stride, out);
    for (size_t r = 0; rc == OMEGA_NUMERIC_OK && r < rows; r++)
        rc = omega_reduce_cpu(op, n ? x + r * row_stride : NULL, n, &out[r]);
    return rc;
}

float omega_reduce_sequential_sum_not_contract(const float *x, size_t n) {
    float s = omega_bits_to_float(OMEGA_REDUCE_SUM_PAD_BITS);
    for (size_t i = 0; i < n; i++) s = s + x[i];
    return n ? s : 0.0f;
}
