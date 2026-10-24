#ifndef OMEGA_GPU_PREFILL_ATTN256_API_H
#define OMEGA_GPU_PREFILL_ATTN256_API_H
/*
 * omega_gpu_prefill_attn256_api: causal prefill attention for Qwen3.8-27B (omega-mv#545, 2026-10-18), on the decode
 * attention's cache (omega_gpu_decode_attn256_api.h); the kernels are described in omega_blackwell_codegen_pattn256.h.
 *
 *   int omega_gpu_prefill_attn256(c, layer, pos0, m, qkv, eps, out_scale, out, info)
 *     qkv: host f32 [m][8192], one row per token in the decode call's layout (q 24 x 256 permuted, k 4 x 256 permuted,
 *     v 4 x 256; the gate is not part of it), tokens at positions pos0 .. pos0+m-1 (positions 0 .. pos0-1 of `layer`
 *     already in the cache). Appends K and V for those positions with the decode step's instruction sequence (the
 *     rows equal m decode steps' rows on the same chip) and writes out: host f32 [m][6144] (token-major, head h at
 *     h * 256, model dim order) multiplied by out_scale. pos0 + m <= max_pos, 1 <= m <= OMEGA_GPU_PATTN256_MAX_M.
 *     Two launches: prep (grid m x 4), attend (grid ceil(m / 8) x 4). Returns OMEGA_GPU_DATTN_OK or an
 *     OMEGA_GPU_DATTN_* code; the text is in omega_gpu_decode_attn_last_error().
 *   int omega_gpu_prefill_attn256_dev(c, layer, pos0, m, qkv_va, eps, out_scale, out_va, info)
 *     The same with device buffers (qkv_va: f32 [m][8192], out_va: f32 [m][6144], 16-byte aligned GPU VAs in the
 *     session); no host copies and no poison check (the caller's download path is the check).
 *   int omega_gpu_prefill_attn256_record(cl, c, layer, pos0, m, qkv_va, eps, out_scale, out_va)
 *     Record path (omega_gpu_cmdlist.h): both launches appended to cl. The caller holds omega_gpu_session_lock(), has
 *     reserved the q' scratch for m tokens (omega_gpu_prefill_attn256_reserve) and warmed both kernels with one
 *     non-record call (a kernel cache miss is refused, never compiled while recording).
 *   int omega_gpu_prefill_attn256_reserve(max_m): allocates the q' scratch for max_m tokens now (ceil(max_m / 2) x
 *     24576 bytes) and builds both kernels (the warm-up the record path needs; sc #642); calls beyond it grow the
 *     scratch (host and dev paths) or are refused (record path: TOO_LARGE).
 *   void omega_gpu_prefill_attn256_release(void): frees the module's buffers (also released when the session closes).
 *
 * Every buffer is GPU-uncached host-visible session memory (nvos.h:1117 rule), launched through
 * omega_gpu_session_launch.
 */
#include "omega_gpu_decode_attn256_api.h"
#include "omega_blackwell_codegen_pattn256.h"
#include <stdint.h>

#define OMEGA_GPU_PATTN256_MAX_M 32768u

typedef struct {
    uint32_t layer, pos0, m;
    uint32_t grid_x[2];         /* prep, attend */
    uint32_t insns[2], gprs[2]; /* allocator peak registers */
    uint64_t elapsed_ns[2];     /* marker-to-marker chip time of each launch (host and dev paths) */
    uint64_t call_ns;
    uint32_t kernel_cache_hits; /* 0..2 */
} OmegaGpuPrefillAttn256Info;

int omega_gpu_prefill_attn256(OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos0, uint32_t m,
                              const float *qkv, float eps, float out_scale, float *out, OmegaGpuPrefillAttn256Info *info);
int omega_gpu_prefill_attn256_dev(OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos0, uint32_t m,
                                  uint64_t qkv_va, float eps, float out_scale, uint64_t out_va, OmegaGpuPrefillAttn256Info *info);
struct OmegaGpuCmdList;
int omega_gpu_prefill_attn256_record(struct OmegaGpuCmdList *cl, OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos0,
                                     uint32_t m, uint64_t qkv_va, float eps, float out_scale, uint64_t out_va);
int omega_gpu_prefill_attn256_reserve(uint32_t max_m);
void omega_gpu_prefill_attn256_release(void);

/* ---- tests ---- */
void omega_gpu_prefill_attn256_test_set_mutant(int prep_mutant, int attend_mutant); /* OmegaDecodeAttn256Spec / OmegaPrefillAttn256Spec .mutant */
/* host: build + encode + patch a kernel (0 prep, 1 attend) into code (cap bytes); returns the code size or 0 */
size_t omega_gpu_prefill_attn256_encode(unsigned kind, uint8_t *code, size_t cap, uint32_t *gprs, int *patched);

#endif
