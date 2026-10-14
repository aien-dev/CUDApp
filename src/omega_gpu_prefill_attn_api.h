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
/* Persistent mirror (cut 5): one per (cache, layer), kept between calls, in the decode cache's EXACT layout
 * ([K plane | V plane], plane = view.v_va - view.k_va, row (kv_head, pos) at plane + (kv_head * max_pos + pos) * 256, bf16)
 * but in GPU-cached device memory (omega_gpu_pf_alloc_dev), so the decode kernel can later be pointed at it unchanged.
 * Rows 0 .. valid-1 equal the cache. A call copies only cache rows valid .. pos0-1 (rows decode appended since) and
 * writes rows pos0 .. pos0+m-1 to both. CONTRACT for token identity (the mirror cannot see other writers):
 *  - anyone who rewrites cache rows below a mirror's `valid` other than through this API (a new sequence decoded into a
 *    used cache, omega_gpu_decode_attn_cache_write, ...) calls omega_gpu_prefill_attn_kv_mirror_invalidate(c, layer
 *    or UINT32_MAX for all, from_pos) first; appending at positions >= valid (normal decode) needs nothing;
 *  - before omega_gpu_decode_attn_cache_free(c), call omega_gpu_prefill_attn_kv_mirror_forget(c) (mirrors are keyed by
 *    the layer's K plane VA; a later cache at the same VA would otherwise inherit stale rows). */
#define OMEGA_GPU_PATTN_MAX_MIRRORS 128u
typedef struct { uint64_t k_va, v_va; uint32_t kv_stride, max_pos, valid; } OmegaGpuPrefillAttnMirrorView;
int omega_gpu_prefill_attn_kv_mirror_view(const OmegaGpuDecodeAttnCache *c, uint32_t layer, OmegaGpuPrefillAttnMirrorView *out);
void omega_gpu_prefill_attn_kv_mirror_invalidate(const OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t from_pos);
void omega_gpu_prefill_attn_kv_mirror_forget(const OmegaGpuDecodeAttnCache *c);
void omega_gpu_prefill_attn_test_set_mirror_per_call(int on);  /* TEST: forget mirrored rows before every call (cut 4 cost) */
int omega_gpu_prefill_attn_kv_mirror_available(void);
/* launches of the last call (return value) and the chip time of each (first `cap` of at most 256 logged) */
uint32_t omega_gpu_prefill_attn_test_launch_ns(uint64_t *ns, uint32_t cap);
#define OMEGA_GPU_PATTN_HOST_PASS 1024u /* host entry: tokens per staged pass in split mode */

/* host: build + encode + patch the kernel for a spec into code (cap bytes); returns the code size or 0 (tests) */
size_t omega_gpu_prefill_attn_encode(const OmegaPrefillAttnSpec *spec, uint8_t *code, size_t cap, uint32_t *gprs, int *patched);

#endif
