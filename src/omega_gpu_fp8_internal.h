#ifndef OMEGA_GPU_FP8_INTERNAL_H
#define OMEGA_GPU_FP8_INTERNAL_H
/* Library-internal helpers of omega_gpu_fp8_api.c for omega_gpu_fp8_slots.c (Expert Ribbon, omega#533). Not part of the
 * public ABI: the slot pool owns its tensor and is the only caller. */
#include "omega_gpu_fp8_api.h"
#include <stddef.h>
#include <stdint.h>
/* A zero-filled FP8 tensor of `experts` x [n][k] in GPU-UNCACHED memory, whatever omega_gpu_fp8_set_weight_cache says:
 * slots are rewritten by the CPU while the process runs, and nvos.h:1117-1118 says GPU-cached system memory "will not be
 * coherent with direct CPU mappings" (omega #457 measured 4096/4096 stale lines on a cached rewrite; uncached FRESH). */
int omega_gpu_fp8_tensor_alloc_uncached(uint32_t experts, uint32_t n, uint32_t k, OmegaGpuFp8Tensor **out);
/* CPU views of one expert: n*k weight bytes and (n/128)*(k/128) packed f32 scales. */
int omega_gpu_fp8_tensor_cpu_view(const OmegaGpuFp8Tensor *t, uint32_t expert, uint8_t **w, float **s);
/* CPU writes before the device reads them (platform SYNC on both buffers; Linux reference: dsb sy). 0 ok, -1. */
int omega_gpu_fp8_tensor_sync_to_gpu(const OmegaGpuFp8Tensor *t);
#endif
