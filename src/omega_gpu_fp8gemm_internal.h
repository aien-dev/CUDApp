#ifndef OMEGA_GPU_FP8GEMM_INTERNAL_H
#define OMEGA_GPU_FP8GEMM_INTERNAL_H
/* Library-internal view of omega_gpu_fp8gemm_api.c for the host tests. Not part of the public ABI (no manifest entry). */
#include "omega_blackwell_codegen_gemm.h"
#include <stdint.h>
/* The program spec omega_gpu_fp8gemm_prepare_shape / _record would build for (k, segs, mt, rowmap) under the process's
 * environment and setters (AIEN_OMEGA_FP8GEMM_X_AHEAD, omega_gpu_fp8gemm_set_shape, ...). Host only: opens no device. */
void omega_gpu_fp8gemm_stroke_spec(uint32_t k, uint32_t segs, uint32_t mt, int rowmap, OmegaFp8GemmSpec *spec);
#endif
