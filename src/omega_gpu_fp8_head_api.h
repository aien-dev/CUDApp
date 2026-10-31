/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_GPU_FP8_HEAD_API_H
#define OMEGA_GPU_FP8_HEAD_API_H
/*
 * omega_gpu_fp8_head_api: the FP8 lm_head over up to OMEGA_FP8HEAD_MAX_XB x rows in ONE launch (aien-dev/CUDApp#217,
 * P-DRA-4 DRA-69). Kernel: omega_blackwell_codegen_fp8head.h. Weights are an omega_gpu_fp8_tensor (one expert, n x k
 * e4m3 + 128 x 128 scales, the upload / omega_gpu_fp8_quantize_bf16 format); the head is read once per launch however
 * many x rows it serves, so a batched decode stroke of N heads streams 0.509 GB instead of N x 0.509 GB.
 *
 * Bit contract: row b of the output is bit-identical to omega_gpu_fp8_gemv_f32 / the legacy single-x record at
 * lane_bytes 16 for x row b (same f32 operations in the same order; tests/gpu_fp8_head_test.c --sim proves it on the
 * warp simulator, --chip on the device).
 *
 * x: xb rows of k f32, ALREADY * 2^OMEGA_FP8_X_EXP (the q36 graph's head x'), contiguous (row b at x_va + b * k * 4).
 * Every row is re-read once per weight row group, so x should live in GPU-cached memory (the q36 graph copies its
 * hidden rows into the cached xq region first). y: row b at y_va + b * y_stride * 4 (y_stride in floats, >= n when
 * xb > 1); only [0, n) of each row is written.
 *
 * Shape: rows 2, warps 8 (rows 1 for xb 1), grid = 48 SMs x the CTAs that fit one SM's register file and warp slots,
 * capped at the row-group count. [INFERENCE: no chip timing yet]. AIEN_OMEGA_FP8_HEAD_SHAPE=rows:warps:ctas (read once,
 * each 0 = automatic) overrides it for chip A/B; omega_gpu_fp8_head_set_shape overrides the environment.
 *
 * Record rules are the other APIs': the caller holds omega_gpu_session_lock(); the kernel must already be resident
 * (omega_gpu_fp8_head_prepare, which pins it) or the record is refused (TOO_LARGE) rather than evicting a kernel.
 * Return codes: OMEGA_GPU_FP8_* (omega_gpu_fp8_api.h).
 */
#include <stdbool.h>
#include <stdint.h>
#include "omega_blackwell_codegen_fp8head.h"
#include "omega_gpu_fp8_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t n, k, xb;
    uint32_t rows, warps, grid_x, gpr_declared;
    uint64_t weight_bytes, scale_bytes; /* bytes of the head one launch reads (independent of xb) */
    uint64_t elapsed_ns, call_ns;
    uint32_t completion_marker;
    bool kernel_cache_hit;
} OmegaGpuFp8HeadInfo;

/* Spec and grid the API uses for (k, n, xb) at the current shape settings and test mutant. */
void omega_gpu_fp8_head_spec_for(uint32_t k, uint32_t n, uint32_t xb, OmegaFp8HeadSpec *spec, uint32_t *grid_x);
/* Build and pin the kernel for (k, xb) (n only sizes the grid, not the kernel). */
int omega_gpu_fp8_head_prepare(uint32_t k, uint32_t xb);
/* Record one launch into cl (caller holds the session lock). t: one expert. */
struct OmegaGpuCmdList;
int omega_gpu_fp8_head_record(struct OmegaGpuCmdList *cl, const OmegaGpuFp8Tensor *t, uint32_t xb, uint64_t x_va, uint64_t y_va,
                              uint32_t y_stride);
/* Chip test path: x = xb rows of k plain f32 (packed * 2^64 here), y = xb rows of n f32 (contiguous). Stages through
 * GPU-uncached scratch, poisons y and refuses a surviving poison word. */
int omega_gpu_fp8_head_gemv_f32(const OmegaGpuFp8Tensor *t, uint32_t xb, const float *x, float *y, OmegaGpuFp8HeadInfo *info);

void omega_gpu_fp8_head_set_shape(uint32_t rows, uint32_t warps, uint32_t ctas); /* 0 = automatic; clears the kernel cache */
void omega_gpu_fp8_head_cache_clear(void);
/* TEST ONLY: kernel mutant (codegen_fp8head.h: 1 wrong scale k-block, 2 wrong scale row block, 3 x row 0 everywhere);
 * clears the kernel cache. */
void omega_gpu_fp8_head_test_set_mutant(int mutant);
int omega_gpu_fp8_head_test_mutant(void);

#ifdef __cplusplus
}
#endif
#endif
