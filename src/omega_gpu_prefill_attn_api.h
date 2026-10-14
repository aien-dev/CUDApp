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
 *     major, head h at h * 128) multiplied by out_scale. Staged in passes of up to 1024 tokens, each one APPEND + one ATTEND launch (cut 3; set_fused(1): 64-token fused chunks). pos0 + m <= max_pos.
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
    uint32_t layer, pos0, m, chunks;      /* chunks = launches made (2 per pass, or ceil(m / 64) fused) */
    uint32_t grid_x, grid_y, threads, splits, per_split; /* of the first launch; splits = 1 (no key split), per_split = keys seen by that launch's last CTA */
    uint64_t elapsed_ns;                  /* chip time, summed over the launches */
    uint64_t call_ns;
    int kernel_cache_hit;
} OmegaGpuPrefillAttnInfo;

int omega_gpu_prefill_attn(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t pos0, uint32_t m,
                           const float *qkv, float eps, float out_scale, float *out,
                           OmegaGpuPrefillAttnInfo *info);
void omega_gpu_prefill_attn_release(void);

/* Device-resident entry (cut 3), for the one-pass prefill pipeline: the same work with no host copies.
 *   qkv_va: f32 [m][5120] rows (20480-byte stride), out_va: f32 [m][4096] rows (16384-byte stride), both 16-byte aligned
 *   GPU VAs in the shared session (GPU-cached device buffers from omega_gpu_pf_alloc_dev, or session buffers).
 *   The kernels only read qkv and only write out; no poison check is possible here (the host never reads device
 *   buffers), the caller's download path is the check. Two launches (APPEND then ATTEND over all m tokens), or
 *   64-token fused chunks after omega_gpu_prefill_attn_set_fused(1). Same return codes as omega_gpu_prefill_attn. */
int omega_gpu_prefill_attn_dev(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t pos0, uint32_t m,
                               uint64_t qkv_va, float eps, float out_scale, uint64_t out_va,
                               OmegaGpuPrefillAttnInfo *info);

/* ---- measurement knobs (tests; each clears the kernel cache) ---- */
void omega_gpu_prefill_attn_set_warps(uint32_t warps);          /* 0 = default (OMEGA_GPU_PATTN_WARPS); 4 or 8 */
void omega_gpu_prefill_attn_test_set_mutant(int mutant);        /* TEST ONLY: OmegaPrefillAttnSpec.mutant */
#define OMEGA_GPU_PATTN_WARPS 8u
void omega_gpu_prefill_attn_set_fused(int fused);               /* 1: cut-1 path (64-token fused chunks); 0 (default): APPEND + ATTEND */
/* ATTEND reads a GPU-cached mirror of the layer's K/V (history copied in, new rows written by APPEND to both places).
 * Default on when built with omega_gpu_prefill.h (kv_mirror_available() == 1); a no-op otherwise. */
void omega_gpu_prefill_attn_set_kv_mirror(int on);
int omega_gpu_prefill_attn_kv_mirror_available(void);
/* launches of the last call (return value) and the chip time of each (first `cap` of at most 256 logged) */
uint32_t omega_gpu_prefill_attn_test_launch_ns(uint64_t *ns, uint32_t cap);
#define OMEGA_GPU_PATTN_HOST_PASS 1024u /* host entry: tokens per staged pass in split mode */

/* host: build + encode + patch the kernel for a spec into code (cap bytes); returns the code size or 0 (tests) */
size_t omega_gpu_prefill_attn_encode(const OmegaPrefillAttnSpec *spec, uint8_t *code, size_t cap, uint32_t *gprs, int *patched);

#endif
