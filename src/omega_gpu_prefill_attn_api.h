#ifndef OMEGA_GPU_PREFILL_ATTN_API_H
#define OMEGA_GPU_PREFILL_ATTN_API_H
/*
 * omega_gpu_prefill_attn_api: causal prefill attention on tensor cores for Qwen3-30B-A3B (OWN-30B L1 attention,
 * 2026-10-13). The API of ~/handoffs/own30b-attn-api.md; the kernel is described in omega_blackwell_codegen_pattn.h.
 *
 *   int omega_gpu_prefill_attn(c, layer, pos0, m, qkv, eps, out_scale, out, info)
 *     c: the decode attention's cache (shared KV cache; positions 0 .. pos0-1 of `layer` already in it).
 *     qkv: host f32 [m][5120] (q 32 x 128 | k 4 x 128 | v 4 x 128, raw), tokens at positions pos0 .. pos0+m-1.
 *     Appends K = bf16(RoPE(RMSNorm(k) * k_norm)) and V = bf16(v) for those positions (the decode kernel's operation
 *     order, so the rows equal m decode steps' rows on the same chip), and writes out: host f32 [m][32 * 128] (token-
 *     major, head h at h * 128) multiplied by out_scale. One launch per chunk of 64 tokens. pos0 + m <= max_pos.
 *     Returns OMEGA_GPU_DATTN_OK or an OMEGA_GPU_DATTN_* code; the text is in omega_gpu_decode_attn_last_error().
 *
 * Every buffer is GPU-uncached host-visible session memory (nvos.h:1117 rule), launched through
 * omega_gpu_session_launch. The chunk staging buffers (qkv 64 x 5120 f32, out 64 x 4096 f32) belong to this module
 * and are released with omega_gpu_prefill_attn_release() or when the session closes.
 */
#include "omega_gpu_decode_attn_api.h"
#include "omega_blackwell_codegen_pattn.h"
#include <stdint.h>

typedef struct {
    uint32_t layer, pos0, m, chunks;      /* chunks = launches made (ceil(m / 64)) */
    uint32_t grid_x, grid_y, threads, splits, per_split; /* of the first chunk; splits = 1 (no key split yet), per_split = keys of the first chunk's widest CTA */
    uint64_t elapsed_ns;                  /* chip time, summed over the launches */
    uint64_t call_ns;
    int kernel_cache_hit;
} OmegaGpuPrefillAttnInfo;

int omega_gpu_prefill_attn(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t pos0, uint32_t m,
                           const float *qkv, float eps, float out_scale, float *out,
                           OmegaGpuPrefillAttnInfo *info);
void omega_gpu_prefill_attn_release(void);

/* ---- measurement knobs (tests; each clears the kernel cache) ---- */
void omega_gpu_prefill_attn_set_warps(uint32_t warps);          /* 0 = default (OMEGA_GPU_PATTN_WARPS); 4 or 8 */
void omega_gpu_prefill_attn_test_set_mutant(int mutant);        /* TEST ONLY: OmegaPrefillAttnSpec.mutant */
#define OMEGA_GPU_PATTN_WARPS 8u

/* host: build + encode + patch the kernel for a spec into code (cap bytes); returns the code size or 0 (tests) */
size_t omega_gpu_prefill_attn_encode(const OmegaPrefillAttnSpec *spec, uint8_t *code, size_t cap, uint32_t *gprs, int *patched);

#endif
