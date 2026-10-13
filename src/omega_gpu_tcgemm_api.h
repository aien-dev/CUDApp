#ifndef OMEGA_GPU_TCGEMM_API_H
#define OMEGA_GPU_TCGEMM_API_H
/*
 * omega_gpu_tcgemm_api (OWN-30B T3 cut 1): FP8 e4m3 block-128-scaled GROUPED GEMM on the tensor cores
 * (HMMA.16816.F32.BF16; e4m3 -> bf16 dequantised in registers, exact). New file on the shared session.
 *
 *   int omega_gpu_tcgemm_f32(const OmegaGpuFp8Tensor *t, uint32_t M, const uint32_t *expert_ids,
 *                            const uint32_t *token_rows, const float *x, uint32_t x_rows, float *y, OmegaGpuTcGemmInfo *info);
 *     Row i computes expert expert_ids[i] (non-decreasing) on x[token_rows[i]]; y is M x n f32 in row-table order.
 *     Arithmetic (bit-exact contract, tests/gpu_tcgemm_test.c emulate()): x rounded RNE to bf16 after * 2^64; per 128-wide
 *     k block, eight HMMA steps of 16 terms each (GB10 fused-sum model, tests/bw_hmma_model.h) from zero, then
 *     acc = fmaf(block, scale', acc). This differs from the FFMA path's order (not bit-identical to omega_gpu_fp8gemm).
 *     No row cap beyond device memory; every output word is poisoned and checked.
 *   int omega_gpu_tcgemm_prepare(uint32_t k);
 * Return codes: the OMEGA_GPU_TCGEMM_* values below; an uncertain completion latches the session (CHIP_FAIL).
 */
#include <stdbool.h>
#include <stdint.h>
#include "omega_gpu_fp8_api.h"
#include "omega_blackwell_codegen_tc.h"

#ifdef __cplusplus
extern "C" {
#endif
enum { OMEGA_GPU_TCGEMM_OK = 0, OMEGA_GPU_TCGEMM_BAD_ARGS = -1, OMEGA_GPU_TCGEMM_TOO_LARGE = -2, OMEGA_GPU_TCGEMM_CODEGEN_FAIL = -3, OMEGA_GPU_TCGEMM_CHIP_FAIL = -4 };

typedef struct {
    uint32_t rows, segments, tiles, n, k, rt, ct, warps, grid_x, grid_y, gpr_declared;
    uint64_t weight_bytes;   /* e4m3 bytes of the distinct experts */
    uint64_t streamed_bytes; /* weight bytes the launch reads (tiles * n * k) */
    uint64_t act_bytes;      /* activation bytes gathered (tiles * grid_x * 16 rt * k * 2) */
    uint64_t elapsed_ns;     /* chip submit-to-second-marker */
    uint64_t call_ns;
    uint32_t completion_marker;
    bool kernel_cache_hit;
} OmegaGpuTcGemmInfo;

int omega_gpu_tcgemm_f32(const OmegaGpuFp8Tensor *t, uint32_t M, const uint32_t *expert_ids, const uint32_t *token_rows,
                         const float *x, uint32_t x_rows, float *y, OmegaGpuTcGemmInfo *info);
int omega_gpu_tcgemm_prepare(uint32_t k);
const char *omega_gpu_tcgemm_rc_name(int rc);
const char *omega_gpu_tcgemm_last_error(void);
void omega_gpu_tcgemm_cache_clear(void);
/* the shape the API uses for width k and output width n */
void omega_gpu_tcgemm_spec_for(uint32_t k, uint32_t n, OmegaTcGemmSpec *spec);
/* measurement knob: force rt, ct, warps (0 restores); TEST ONLY mutants 1-4 (omega_blackwell_codegen_tc.h) */
void omega_gpu_tcgemm_set_shape(uint32_t rt, uint32_t ct, uint32_t warps);
void omega_gpu_tcgemm_test_set_mutant(int mutant);
#ifdef __cplusplus
}
#endif
#endif
