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

#ifndef OMEGA_GPU_TERN_API_H
#define OMEGA_GPU_TERN_API_H
/*
 * omega_gpu_tern_api: KT1, the native (no CUDA) W2A8 ternary GEMV on the shared session, ternary engine stage T0
 * (2026-10-15). New file; no existing API changes. Kernel: omega_blackwell_codegen_tern.h.
 *
 *   typedef struct OmegaGpuTernTensor OmegaGpuTernTensor;   // opaque, device resident
 *   int omega_gpu_tern_tensor_upload(uint32_t n, uint32_t k, const int8_t *w, const OmegaGpuTernSegments *seg,
 *                                    OmegaGpuTernTensor **out);
 *     w: n * k values in {-1, 0, +1}, row-major [n][k]. Any other value is refused (BAD_TRIT, nothing uploaded).
 *     Requires n >= 1, k % 256 == 0 (k % 512 above 8192), k <= 16384. Packed to omega's 2-bit trit code
 *     (src/algebra/oma_trit.h) in the layout set by omega_gpu_tern_set_layout (read at upload), zero rows padding n to a
 *     multiple of 8. Weights are GPU-cached by default (written once by the CPU, then only read by the GPU;
 *     omega_gpu_mem_cached.h). seg may be NULL (one segment, scale 1).
 *   seg: the per-matrix weight scales of a fused projection, one per OUTPUT SEGMENT with explicit row boundaries:
 *     count 1..4, end[i] the first row after segment i (strictly increasing, end[count - 1] == n), scale[i] finite.
 *     Example q|k|v: count 3, end {2560, 3200, 3840}. Row r uses scale[i] for the segment i that contains r.
 *   int omega_gpu_tern_gemv_i32(const OmegaGpuTernTensor *t, const int8_t *x, int32_t *y, OmegaGpuTernInfo *info);
 *     y[r] = sum_c W[r][c] * x[c], exact int32 (the contract of oma_rz_oracle). x: k int8, y: n int32. One launch.
 *   int omega_gpu_tern_gemv_f32(const OmegaGpuTernTensor *t, const int8_t *x, float a_scale, float *y, OmegaGpuTernInfo *info);
 *     y[r] = ((float)y_int[r] * a_scale) * scale[seg(r)], two round-to-nearest f32 multiplies in that order. a_scale finite.
 *     k <= 8192 only (one x chunk). Not the same rounding as the divide form (float)acc / s_a * w_scale; the T2 host
 *     reference defines which one the engine uses.
 * Return codes: 0 OK; negative OMEGA_GPU_TERN_* below. Every output word is poisoned before the launch and a result that
 * still holds poison is refused. An uncertain completion latches the process (the session's rule).
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "omega_blackwell_codegen_tern.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    OMEGA_GPU_TERN_OK = 0,
    OMEGA_GPU_TERN_BAD_ARGS = -1,
    OMEGA_GPU_TERN_TOO_LARGE = -2,
    OMEGA_GPU_TERN_CODEGEN_FAIL = -3,
    OMEGA_GPU_TERN_CHIP_FAIL = -4,
    OMEGA_GPU_TERN_BAD_TRIT = -5
};

#define OMEGA_GPU_TERN_POISON_I32 0x7f7f7f7f  /* |y| <= 128 * 16384 < 2^22: never a result */
#define OMEGA_GPU_TERN_POISON_F32_BITS 0x7fc0deadu

typedef struct {
    uint32_t count;
    uint32_t end[4];
    float scale[4];
} OmegaGpuTernSegments;

typedef struct {
    uint32_t n, k, n_padded, layout, rows_per_half, warps_per_cta, depth, grid_x, chunks, epi;
    uint64_t weight_bytes;  /* packed trit bytes the launch reads (n_padded * k / 4) */
    uint64_t elapsed_ns;    /* chip submit-to-second-marker */
    uint64_t call_ns;       /* whole API call */
    uint32_t completion_marker;
    bool kernel_cache_hit, weights_cached;
    uint32_t seg_bounds[3]; /* the row boundaries the launch used */
    char target_chip[64];
    uint32_t sm_architecture;
} OmegaGpuTernInfo;

typedef struct OmegaGpuTernTensor OmegaGpuTernTensor;

int omega_gpu_tern_tensor_upload(uint32_t n, uint32_t k, const int8_t *w, const OmegaGpuTernSegments *seg, OmegaGpuTernTensor **out);
void omega_gpu_tern_tensor_free(OmegaGpuTernTensor *t);
void omega_gpu_tern_tensor_shape(const OmegaGpuTernTensor *t, uint32_t *n, uint32_t *k);
uint64_t omega_gpu_tern_tensor_bytes(const OmegaGpuTernTensor *t); /* packed bytes on the device */
int omega_gpu_tern_gemv_i32(const OmegaGpuTernTensor *t, const int8_t *x, int32_t *y, OmegaGpuTernInfo *info);
int omega_gpu_tern_gemv_f32(const OmegaGpuTernTensor *t, const int8_t *x, float a_scale, float *y, OmegaGpuTernInfo *info);
/* Record path (omega_gpu_cmdlist.h): the same launch appended to cl. x_va: k int8 on the device (16-byte aligned);
 * y_va: n words; as_va: the f32 a_scale word for the f32 epilogue, 0 for int32 output. The caller holds
 * omega_gpu_session_lock(). Nothing staged, poisoned or read back. A kernel cache miss that would evict is refused
 * (TOO_LARGE): warm the shape up with one non-record call first. */
struct OmegaGpuCmdList;
int omega_gpu_tern_gemv_record(struct OmegaGpuCmdList *cl, const OmegaGpuTernTensor *t, uint64_t x_va, uint64_t y_va, uint64_t as_va);
const char *omega_gpu_tern_rc_name(int rc);
const char *omega_gpu_tern_last_error(void);
void omega_gpu_tern_cache_clear(void);

/* Trit packing and segment boundaries as the device sees them (tests and the host emulation use the same code). */
int omega_gpu_tern_pack(uint32_t n, uint32_t k, const int8_t *w, uint32_t layout, uint32_t R, uint32_t *out_words, uint32_t n_padded);
uint32_t omega_gpu_tern_padded_rows(uint32_t n);
int omega_gpu_tern_segment_bounds(uint32_t n, const OmegaGpuTernSegments *seg, int mutant, uint32_t bounds[3]);

/* Register-file check (32 * warps * declared registers <= 65536). */
int omega_gpu_tern_kernel_fits(const OmegaTernGemvSpec *spec, uint32_t *declared);
/* The kernel shape the API picks for (k, layout, R): rows, warps, depth, CTAs. */
void omega_gpu_tern_shape_for(uint32_t k, uint32_t layout, uint32_t R, uint32_t *rows, uint32_t *warps, uint32_t *depth, uint32_t *ctas);

/* ---- measurement knobs (tests; 0 restores the automatic choice) ---- */
void omega_gpu_tern_set_layout(uint32_t layout, uint32_t block_rows); /* read by later uploads; default row-major (rows 1, 160 CTAs: T0 sweep) */
void omega_gpu_tern_set_shape(uint32_t rows, uint32_t warps, uint32_t depth); /* clears the kernel cache */
void omega_gpu_tern_set_cta_budget(uint32_t ctas);
void omega_gpu_tern_set_weight_cache(int cached); /* 1 GPU-cached (default), 0 uncached; read by later uploads */
/* TEST ONLY: 1 flipped planes, 3 dropped pair (kernel mutants); 2 maps segments as r / (n / count) (the API). */
void omega_gpu_tern_test_set_mutant(int mutant);

#ifdef __cplusplus
}
#endif
#endif
