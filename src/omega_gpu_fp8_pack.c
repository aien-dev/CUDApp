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

/* omega_gpu_fp8_pack.c: host side of the FP8 e4m3 GEMV (OWN-30B Cut 6). See omega_gpu_fp8_api.h.
 *
 * The weight bytes keep the checkpoint's row-major [n][k] order on the device (a warp reads
 * 32 * lane_bytes contiguous bytes of a row per step), so "packing" is: refuse the two e4m3 NaN
 * codes, convert the bf16 scale grid to f32 times 2^OMEGA_FP8_SCALE_EXP, and stage x as f32 times
 * 2^OMEGA_FP8_X_EXP. All three are exact (powers of two; refused on overflow). */
#include "omega_gpu_fp8_api.h"
#include "omega_blackwell_codegen_moe.h"
#include "omega_blackwell_matmul.h"
#include <math.h>
#include <string.h>

float omega_fp8_e4m3_ref(uint8_t b) {
    int exponent = (b >> 3) & 0x0f;
    float mantissa = (float)(b & 0x07);
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    if (exponent == 0x0f && (b & 0x07) == 0x07) return NAN; /* 0x7f / 0xff */
    if (exponent == 0) return sign * mantissa * ldexpf(1.0f, -9);
    return sign * (1.0f + mantissa / 8.0f) * ldexpf(1.0f, exponent - 7);
}

uint32_t omega_fp8_e4m3_kernel_bits(uint8_t b) {
    return ((uint32_t)(b & 0x7fu) << 20) | ((uint32_t)(b & 0x80u) << 24);
}

int omega_fp8_e4m3_is_nan(uint8_t b) { return (b & 0x7fu) == 0x7fu; }

size_t omega_fp8_pack_scan_nan(const uint8_t *w, size_t count, size_t *first) {
    size_t n = 0;
    for (size_t i = 0; i < count; i++)
        if (omega_fp8_e4m3_is_nan(w[i])) { if (n == 0 && first) *first = i; n++; }
    return n;
}

int omega_fp8_pack_scales(const uint16_t *bf16, size_t count, float *out, int mutant_no_fold) {
    const float fold = mutant_no_fold ? 1.0f : ldexpf(1.0f, OMEGA_FP8_SCALE_EXP);
    for (size_t i = 0; i < count; i++) {
        float s = omega_bf16_to_fp32(bf16[i]);
        if (!(fabsf(s) <= 3.0e38f)) return -1; /* NaN or inf scale */
        float f = s * fold;
        if (!(fabsf(f) <= 3.0e38f)) return -1; /* fold overflow */
        out[i] = f;
    }
    return 0;
}

void omega_fp8_pack_x_f32(const float *x, uint32_t k, float *out) {
    const float fold = ldexpf(1.0f, OMEGA_FP8_X_EXP);
    for (uint32_t i = 0; i < k; i++) out[i] = x[i] * fold;
}

void omega_fp8_pack_x_bf16(const uint16_t *x, uint32_t k, float *out) {
    const float fold = ldexpf(1.0f, OMEGA_FP8_X_EXP);
    for (uint32_t i = 0; i < k; i++) out[i] = omega_bf16_to_fp32(x[i]) * fold;
}

void omega_fp8_gemv_oracle(const uint8_t *w, const uint16_t *scale_bf16, uint32_t n, uint32_t k,
                           const float *x, double *y, double *yscale) {
    const uint32_t kb = k / OMEGA_FP8_BLOCK;
    for (uint32_t r = 0; r < n; r++) {
        double acc = 0.0, mag = 0.0;
        const uint8_t *row = w + (size_t)r * k;
        const uint16_t *srow = scale_bf16 + (size_t)(r / OMEGA_FP8_BLOCK) * kb;
        for (uint32_t b = 0; b < kb; b++) {
            double s = (double)omega_bf16_to_fp32(srow[b]);
            for (uint32_t c = b * OMEGA_FP8_BLOCK; c < (b + 1) * OMEGA_FP8_BLOCK; c++) {
                double p = (double)omega_fp8_e4m3_ref(row[c]) * (double)x[c] * s;
                acc += p; mag += fabs(p);
            }
        }
        y[r] = acc;
        if (yscale) yscale[r] = mag;
    }
}
