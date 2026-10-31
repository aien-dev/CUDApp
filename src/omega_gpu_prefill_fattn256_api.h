/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_GPU_PREFILL_FATTN256_API_H
#define OMEGA_GPU_PREFILL_FATTN256_API_H
/*
 * omega_gpu_prefill_fattn256_api: flash-style causal prefill attention on the tensor cores for the head_dim 256 decode
 * cache (Linear DRA-85, P-DRA-4 Lane P; the attend kernel is omega_blackwell_codegen_fattn256.h). Built for the Qwen3.6
 * 35B full-attention layers (16 q heads / 2 kv heads); any cache shape with q_heads / kv_heads in 1..8 works.
 *
 *   int omega_gpu_prefill_fattn256(c, layer, pos0, m, qkv, xstride, eps, out_scale, out, ostride, info)
 *     qkv: host f32 rows xstride bytes apart (the first qkv_words of a row are the decode call's q|k|v; the rest, e.g.
 *     the q36 output gate, is ignored), tokens at positions pos0 .. pos0+m-1 (0 .. pos0-1 of `layer` already cached).
 *     Appends K and V for those positions with the decode step's own prep instructions (the cache rows equal m decode
 *     steps' rows) and writes out: host f32 rows ostride bytes apart, the first out_words of each the attention output
 *     (head h at h * 256, model dim order) times out_scale; the rest of a row is left untouched. Two launches: prep
 *     (grid m x q_heads / q_per_cta), attend (grid ceil(m / 16) x kv_heads, no shared memory).
 *     The output is NOT bit-identical to the decode steps (P is rounded to bf16 for the PV product, the sums run in
 *     another order); the tolerance and the golden-token check are the acceptance (DRA-85 G3).
 *   int omega_gpu_prefill_fattn256_dev(...): the same with device buffers (16-byte aligned session VAs, no poison check).
 *   int omega_gpu_prefill_fattn256_record(cl, ...): both launches appended to cl; the caller holds the session lock,
 *     reserved the q' scratch (reserve_for) and warmed both kernels (a kernel cache miss is refused while recording).
 *   int omega_gpu_prefill_fattn256_reserve_for(c, max_m, xstride): allocates the q' scratch (ceil(max_m / 2) * q_heads *
 *     1024 bytes) and builds both kernels for c's head shape and the q|k|v row stride xstride (0 = packed rows).
 *   void omega_gpu_prefill_fattn256_release(void).
 *   int omega_gpu_prefill_fattn256_set_variant(variant): the attend kernel variant for later calls (0 or 1: the base
 *     kernel, the default; 2: next-block L1 touch prefetch, omega_blackwell_codegen_fattn256.h); drops the kernel cache,
 *     so a record caller warms up again (reserve_for). BAD_ARGS outside 0..2.
 * Returns OMEGA_GPU_DATTN_OK or an OMEGA_GPU_DATTN_* code; the text is in omega_gpu_decode_attn_last_error().
 */
#include "omega_gpu_decode_attn256_api.h"
#include "omega_blackwell_codegen_fattn256.h"
#include <stddef.h>
#include <stdint.h>

#define OMEGA_GPU_FATTN256_MAX_M 32768u

typedef struct {
    uint32_t layer, pos0, m;
    uint32_t grid_x[2];         /* prep, attend */
    uint32_t insns[2], gprs[2];
    uint64_t elapsed_ns[2];     /* marker-to-marker chip time of each launch (host and dev paths) */
    uint64_t call_ns;
    uint32_t kernel_cache_hits; /* 0..2 */
} OmegaGpuPrefillFattn256Info;

int omega_gpu_prefill_fattn256(OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos0, uint32_t m, const float *qkv,
                               uint32_t xstride, float eps, float out_scale, float *out, uint32_t ostride, OmegaGpuPrefillFattn256Info *info);
int omega_gpu_prefill_fattn256_dev(OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos0, uint32_t m, uint64_t qkv_va,
                                   uint32_t xstride, float eps, float out_scale, uint64_t out_va, uint32_t ostride, OmegaGpuPrefillFattn256Info *info);
struct OmegaGpuCmdList;
int omega_gpu_prefill_fattn256_record(struct OmegaGpuCmdList *cl, OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos0, uint32_t m,
                                      uint64_t qkv_va, uint32_t xstride, float eps, float out_scale, uint64_t out_va, uint32_t ostride);
int omega_gpu_prefill_fattn256_reserve_for(const OmegaGpuDecodeAttn256Cache *c, uint32_t max_m, uint32_t xstride);
void omega_gpu_prefill_fattn256_release(void);
int omega_gpu_prefill_fattn256_set_variant(int variant);

/* ---- tests ---- */
void omega_gpu_prefill_fattn256_test_set_mutant(int prep_mutant, int attend_mutant); /* OmegaDecodeAttn256Spec / OmegaPrefillFattn256Spec .mutant */
/* host: build + encode + patch a kernel (0 prep, 1 attend) for shape q_heads / kv_heads / q_per_cta (0 = 16 / 2 / 4, the q36 cache) and
 * prep row stride xstride (0 = packed) into code (cap bytes); returns the code size or 0 */
size_t omega_gpu_prefill_fattn256_encode(unsigned kind, uint32_t q_heads, uint32_t kv_heads, uint32_t q_per_cta, uint32_t xstride,
                                         uint8_t *code, size_t cap, uint32_t *gprs, int *patched);

#endif
