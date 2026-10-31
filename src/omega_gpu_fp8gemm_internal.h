#ifndef OMEGA_GPU_FP8GEMM_INTERNAL_H
#define OMEGA_GPU_FP8GEMM_INTERNAL_H
/* Library-internal view of omega_gpu_fp8gemm_api.c for the host tests. Not part of the public ABI (no manifest entry). */
#include "omega_blackwell_codegen_gemm.h"
#include <stdint.h>
/* The program spec omega_gpu_fp8gemm_prepare_shape / _record would build for (k, segs, mt, rowmap) under the process's
 * environment and setters (AIEN_OMEGA_FP8GEMM_X_AHEAD, _ROW_AHEAD, _X_RESIDENT, _SHARED_ONE_PASS, omega_gpu_fp8gemm_set_shape, ...). Host only: opens no device. */
void omega_gpu_fp8gemm_stroke_spec(uint32_t k, uint32_t segs, uint32_t mt, int rowmap, OmegaFp8GemmSpec *spec);
/* The CTA budget per segment a call of width k with segs segments gets when it passes cta_budget 0 (the stroke), under
 * the process's environment and setters (AIEN_OMEGA_FP8GEMM_CTA_BUDGET, omega_gpu_fp8gemm_set_cta_budget; DRA-78). */
uint32_t omega_gpu_fp8gemm_stroke_cta_budget(uint32_t k, uint32_t segs);
#endif
