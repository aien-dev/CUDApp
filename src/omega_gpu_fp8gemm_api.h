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

#ifndef OMEGA_GPU_FP8GEMM_API_H
#define OMEGA_GPU_FP8GEMM_API_H
/*
 * omega_gpu_fp8gemm_api: native (no CUDA) FP8 e4m3 block-128-scaled GROUPED GEMM for prefill (OWN-30B, builder F
 * item 26). It streams each distinct expert's weights once per pass of up to MT rows for all the rows routed to it,
 * instead of once per row (omega_gpu_fp8_api's grouped GEMV). New file on the shared session; no existing API changes.
 *
 * ============ C signatures and semantics (for the Rust binding) ============
 *   Weights are the existing OmegaGpuFp8Tensor (omega_gpu_fp8_api.h): experts x n x k e4m3 + 128x128 bf16-derived
 *   scales, uploaded once and shared with the GEMV path. Dense projections (qkv n=5120 k=2048, o n=2048 k=4096) are
 *   tensors with experts = 1.
 *
 *   int omega_gpu_fp8gemm_f32(const OmegaGpuFp8Tensor *t, uint32_t M, const uint32_t *expert_ids,
 *                             const uint32_t *token_rows, const float *x, uint32_t x_rows, float *y,
 *                             OmegaGpuFp8GemmInfo *info);
 *   int omega_gpu_fp8gemm_bf16(... const uint16_t *x ...);   same with bf16 activations
 *     The row table: row i (0 <= i < M) computes expert expert_ids[i] on activation row x[token_rows[i]] (x: x_rows * k
 *     values). expert_ids must be non-decreasing (sorted by expert; equal ids form one segment, the weights of a segment
 *     are streamed once per pass of MT rows). Output y: M * n f32 in ROW-TABLE order: y[i][r] = sum_c W[expert_ids[i]][r][c]
 *     * x[token_rows[i]][c] * scale, f32 accumulate, the same summation order as omega_gpu_fp8_gemv_f32 (the result is
 *     bit-identical to calling the GEMV once per row). M <= 1024, at most 256 segments, k and n as the tensor.
 *     One launch. info may be NULL.
 *
 *   int omega_gpu_fp8gemm_prepare(uint32_t k);   build and pin the kernel for width k now (serving).
 *   const char *omega_gpu_fp8gemm_rc_name(int rc);  const char *omega_gpu_fp8gemm_last_error(void);
 *
 * Return codes: 0 OK; negative OMEGA_GPU_FP8GEMM_*. An uncertain completion latches the process (every later call of
 * every API returns CHIP_FAIL). Every output word is poisoned before the launch and the call refuses a result that still
 * holds poison. No host oracle runs inside the call.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "omega_gpu_fp8_api.h"
#include "omega_blackwell_codegen_gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    OMEGA_GPU_FP8GEMM_OK = 0,
    OMEGA_GPU_FP8GEMM_BAD_ARGS = -1,
    OMEGA_GPU_FP8GEMM_TOO_LARGE = -2,
    OMEGA_GPU_FP8GEMM_CODEGEN_FAIL = -3,
    OMEGA_GPU_FP8GEMM_CHIP_FAIL = -4
};
#define OMEGA_GPU_FP8GEMM_MAX_ROWS 1024u
#define OMEGA_GPU_FP8GEMM_MAX_SEGMENTS 256u

typedef struct {
    uint32_t rows, segments, n, k, mt, warps_per_cta, lane_bytes, grid_x, grid_y, passes_max;
    uint64_t weight_bytes;   /* e4m3 bytes of the distinct experts, one pass each */
    uint64_t streamed_bytes; /* bytes the launch reads from the weights (sum over segments of passes * n * k) */
    uint64_t elapsed_ns;     /* chip submit-to-second-marker */
    uint64_t call_ns;        /* whole API call (host permute, launch, copy back) */
    uint32_t completion_marker;
    bool kernel_cache_hit;
} OmegaGpuFp8GemmInfo;

int omega_gpu_fp8gemm_f32(const OmegaGpuFp8Tensor *t, uint32_t M, const uint32_t *expert_ids, const uint32_t *token_rows,
                          const float *x, uint32_t x_rows, float *y, OmegaGpuFp8GemmInfo *info);
int omega_gpu_fp8gemm_bf16(const OmegaGpuFp8Tensor *t, uint32_t M, const uint32_t *expert_ids, const uint32_t *token_rows,
                           const uint16_t *x, uint32_t x_rows, float *y, OmegaGpuFp8GemmInfo *info);
int omega_gpu_fp8gemm_prepare(uint32_t k);
const char *omega_gpu_fp8gemm_rc_name(int rc);
const char *omega_gpu_fp8gemm_last_error(void);
void omega_gpu_fp8gemm_cache_clear(void);

/* Register-file check (declared = peak + 1 rounded up to 8; 32 * warps * declared <= 65536). */
int omega_gpu_fp8gemm_kernel_fits(const OmegaFp8GemmSpec *spec, uint32_t *declared);
/* The shape the API picks for width k: rows per pass, warps per CTA, lane bytes. */
void omega_gpu_fp8gemm_shape_for(uint32_t k, uint32_t *mt, uint32_t *warps, uint32_t *lane_bytes);

/* ---- measurement knobs (each clears the kernel cache; 0 restores the automatic choice) ---- */
void omega_gpu_fp8gemm_set_shape(uint32_t mt, uint32_t warps);
void omega_gpu_fp8gemm_set_cta_budget(uint32_t ctas_per_segment);
/* TEST ONLY: deliberately wrong kernels 1 (scale dropped), 2 (scale pointer never advances), 3 (neighbour row's x). */
void omega_gpu_fp8gemm_test_set_mutant(int mutant);

/* ---- the stroke's record path (L312 v2: one weight read per layer for every conversation on the chip) ----
 * omega_gpu_fp8gemm_record appends ONE launch of the grouped GEMM to cl; every buffer is the caller's, in device memory:
 *   seg_va: segs entries of OMEGA_GEMM_SEG_BYTES {w_va, s_va, row0, cnt} (omega_gpu_fp8_tensor_entry, or a kernel on
 *           the chip writes them); a segment with cnt 0 is skipped by its CTAs (the group kernel leaves trailing
 *           segments empty when fewer experts are distinct);
 *   xp_va:  x' rows (k f32 each, permuted and * 2^64, omega_fp8gemm_permute_x) followed by mt rows of padding the
 *           kernel may read; y_va: rows * n f32;
 *   xmap_va, ymap_va: both 0 (x' row and y row of sorted row p are p) or both u32 tables of rows + mt entries:
 *           x' row XMAP[p] feeds sorted row p and its output lands in y row YMAP[p] (padding entries 0). With maps the
 *           kernel runs at mt <= 4 (rowmap at mt 8 does not fit the register file); mt 0 = the API's choice for segs.
 *   cta_budget 0 = the API's choice (omega_gpu_fp8gemm_cta_budget_for). rows: the y rows (num_elements), <= 1024.
 * The caller holds omega_gpu_session_lock(). Nothing is staged, poisoned or read back (the list's per-launch semaphore
 * and the caller's end-of-token checks replace the poison check). A kernel cache miss that would evict a resident kernel
 * is refused (TOO_LARGE): call omega_gpu_fp8gemm_prepare_shape for the shape first (segs, mt, rowmap as recorded). */
struct OmegaGpuCmdList;
int omega_gpu_fp8gemm_record(struct OmegaGpuCmdList *cl, const OmegaGpuFp8Tensor *t, uint32_t rows, uint32_t segs, uint64_t seg_va,
                             uint64_t xp_va, uint64_t y_va, uint64_t xmap_va, uint64_t ymap_va, uint32_t mt, uint32_t cta_budget,
                             OmegaGpuFp8GemmInfo *info);
int omega_gpu_fp8gemm_prepare_shape(uint32_t k, uint32_t segs, uint32_t mt, int rowmap);
uint32_t omega_gpu_fp8gemm_cta_budget_for(uint32_t segs);

/* host helper: the permuted, 2^64-scaled x' row the kernel reads (k values in, k out) */
void omega_fp8gemm_permute_x(const float *x, uint32_t k, uint32_t lane_bytes, float *out);

#ifdef __cplusplus
}
#endif
#endif
