#ifndef OMEGA_GPU_DECODE_ATTN256_API_H
#define OMEGA_GPU_DECODE_ATTN256_API_H
/*
 * omega_gpu_decode_attn256_api: one-launch decode attention for Qwen3.8-27B (omega-mv#544, 2026-10-18): head_dim 256,
 * 24 query heads over 4 KV heads, partial RoPE by permutation. A derivative of omega_gpu_decode_attn_api.h (the 30B's
 * head_dim 128 kernel, untouched); the kernel is described in omega_blackwell_codegen_attn256.h. Return codes and the
 * bf16 helpers are shared with that API (OMEGA_GPU_DATTN_*, omega_dattn_bf16_rne).
 *
 *   int  omega_gpu_decode_attn256_cache_alloc(uint32_t layers, uint32_t max_pos, double rope_theta, uint32_t rotary_dim,
 *                                            OmegaGpuDecodeAttn256Cache **out);
 *        GPU-resident bf16 KV cache for `layers` layers of 4 KV heads x `max_pos` positions x 256 (one allocation per
 *        layer, 2 * 4 * max_pos * 512 bytes + slack), the RoPE table (f32 [max_pos][cos 128 | sin 128], built in double:
 *        pair j < rotary_dim / 2 rotates with inv_freq theta^(-2j / rotary_dim), the rest is identity), the per-layer
 *        q_norm / k_norm weights, the cross-split partials and counters. rotary_dim: even, 2..256 (64 for Qwen3.8-27B,
 *        256 = full RoPE). Opens the session. The Qwen3.8-27B cache at 131072 positions is 512 MiB per layer.
 *   void omega_gpu_decode_attn256_cache_free(OmegaGpuDecodeAttn256Cache *c);
 *   int  omega_gpu_decode_attn256_set_norms(c, layer, q_norm, k_norm): 256 f32 each, ALREADY folded to 1 + w (the model's
 *        zero-centred RMSNorm) and permuted with omega_dattn256_perm_vec.
 *   int  omega_gpu_decode_attn256(c, layer, pos, qkv, eps, out_scale, out, info);
 *        qkv: f32 [8192] = q 24 heads x 256 (permuted dims, gate NOT included), then k 4 x 256 (permuted), then v 4 x 256
 *        (NOT permuted); pos: the new token's position (< max_pos; positions 0..pos-1 already in the cache for that
 *        layer). Appends K/V at pos and writes out[24][256] * out_scale in the model's dim order (V and the output are
 *        never permuted). The gate (sigmoid of the second half of each head's q_proj rows) is applied by the caller.
 *        Returns OMEGA_GPU_DATTN_OK or an OMEGA_GPU_DATTN_* code; the chip failure text is in
 *        omega_gpu_decode_attn_last_error().
 *   Test helpers: cache_write moves `count` consecutive bf16 K and V rows (256 each; K rows in permuted order) of a layer
 *   and KV head into the cache in one sync; cache_read returns one row of each.
 *
 * The permutation (design decision 1 of #544; rotary_dim R = 2P, P pairs; slot i of the kernel's head holds the model's
 * dim omega_dattn256_perm_slot(i, R)):
 *     i < P:            model dim i                 (first half of the rotated dims)
 *     P <= i < 128:     model dim R + (i - P)       (unrotated dims, in order)
 *     128 <= i < 128+P: model dim P + (i - 128)     (second half of the rotated dims: the partner of slot i - 128)
 *     128+P <= i:       model dim i                 (the rest of the unrotated dims)
 * so the kernel's rotate-half pairs (i, i + 128), i < P, are exactly the model's pairs (j, j + P) and the table's
 * identity pairs i >= P leave the other dims alone. R = 256 gives the identity permutation. q, k and the two norm
 * weights are permuted; the K cache holds permuted rows; V and the output are not permuted.
 *
 * Launch shape: grid (splits, 4), 256 threads, splits = ceil((pos + 1) / per_split), per_split a multiple of 256.
 * Every buffer is GPU-uncached host-visible session memory (nvos.h:1117 rule), launched through
 * omega_gpu_session_launch. The counters must be zero before a launch; the last CTA of each KV head resets its
 * counter, and the API verifies the four words are zero after each call.
 */
#include "omega_blackwell_codegen_attn256.h"
#include "omega_gpu_decode_attn_api.h"
#include <stdint.h>
#include <stddef.h>

#define OMEGA_GPU_DATTN256_PER_SPLIT 256u
#define OMEGA_GPU_DATTN256_MAX_LAYERS 64u
#define OMEGA_GPU_DATTN256_MAX_POS 131072u
#define OMEGA_GPU_DATTN256_DEPTH 4u

typedef struct OmegaGpuDecodeAttn256Cache OmegaGpuDecodeAttn256Cache;

typedef struct {
    uint32_t layer, pos, splits, per_split, depth;
    uint32_t grid_x, grid_y, threads;
    uint64_t kv_bytes;          /* K + V bytes the launch streams: 2 * 4 * (pos + 1) * 512 */
    uint64_t elapsed_ns;        /* marker-to-marker chip time */
    uint64_t call_ns;           /* whole call */
    uint32_t completion_marker;
    int kernel_cache_hit;
} OmegaGpuDecodeAttn256Info;

int omega_gpu_decode_attn256_cache_alloc(uint32_t layers, uint32_t max_pos, double rope_theta, uint32_t rotary_dim, OmegaGpuDecodeAttn256Cache **out);
/* Q36 (sc L306): the same for another model shape (OmegaDecodeAttn256Spec.q_heads / kv_heads / q_per_cta; 0, 0, 0 = the 27B,
 * the plain alloc). Qwen3.6-35B-A3B: 16, 2, 4. qkv rows are (q_heads + 2 kv_heads) * 256 f32, out q_heads * 256; the cache has
 * kv_heads planes; partials and split counters one per CTA row (q_heads / q_per_cta <= 16). */
int omega_gpu_decode_attn256_cache_alloc_shape(uint32_t layers, uint32_t max_pos, double rope_theta, uint32_t rotary_dim, uint32_t q_heads,
                                               uint32_t kv_heads, uint32_t q_per_cta, OmegaGpuDecodeAttn256Cache **out);
void omega_gpu_decode_attn256_cache_free(OmegaGpuDecodeAttn256Cache *c);
int omega_gpu_decode_attn256_set_norms(OmegaGpuDecodeAttn256Cache *c, uint32_t layer, const float *q_norm, const float *k_norm);
int omega_gpu_decode_attn256(OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos, const float *qkv, float eps,
                             float out_scale, float *out, OmegaGpuDecodeAttn256Info *info);
/* Record path (decode graph; omega_gpu_cmdlist.h): the launch appended to cl instead of submitted. qkv_va: f32 [8192] on
 * the device; out_va: f32 [24 * 256]. The caller holds omega_gpu_session_lock(). A kernel cache miss that would evict
 * the resident kernel is refused (warm up with one non-record call first). */
struct OmegaGpuCmdList;
int omega_gpu_decode_attn256_record(struct OmegaGpuCmdList *cl, OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos, uint64_t qkv_va,
                                    float eps, float out_scale, uint64_t out_va);
/* Caller holds the lock, after the list ran: OK when the four split counters are zero, else CHIP_FAIL. */
int omega_gpu_decode_attn256_counters_check(OmegaGpuDecodeAttn256Cache *c);
/* L369 multi-token prompt attention: tokens prompt tokens of one conversation at positions pos0 .. pos0 + tokens - 1 (qkv rows
 * xstride bytes apart, outputs ostride bytes apart), every output bit-identical to `tokens` decode records in order. Records
 * one prep launch (every token's K/V rows, the decode step's own instructions) and the attend launches (grid (splits, n * vy),
 * groups of n tokens whose partials fit a shared 64 MiB buffer). per_split must be the default 256. multi_prepare builds the
 * two kernels and the buffer (call before recording, outside the record lock). TEST: set_multi_mutant(1) drops the prep launch. */
int omega_gpu_decode_attn256_multi_prepare(OmegaGpuDecodeAttn256Cache *c);
int omega_gpu_decode_attn256_multi_record(struct OmegaGpuCmdList *cl, OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos0, uint32_t tokens,
                                          uint64_t qkv_va, uint32_t xstride, float eps, float out_scale, uint64_t out_va, uint32_t ostride);
void omega_gpu_decode_attn256_test_set_multi_mutant(int mutant);
int omega_gpu_decode_attn256_cache_write(OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t kv_head, uint32_t pos, uint32_t count,
                                         const uint16_t *k_bf16, const uint16_t *v_bf16); /* count rows from pos, one sync */
int omega_gpu_decode_attn256_cache_read(OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t kv_head, uint32_t pos,
                                        uint16_t *k_bf16, uint16_t *v_bf16);
/* Read-only view of a layer of the cache for the prefill attention: device addresses only. */
typedef struct {
    uint32_t layers, max_pos, rotary_dim;
    double theta;
    uint64_t k_va, v_va;        /* K and V plane bases of `layer` (row (kv_head, pos) at + (kv_head * max_pos + pos) * 512) */
    uint64_t rope_va;           /* f32 [max_pos][cos 128 | sin 128] */
    uint64_t q_norm_va, k_norm_va; /* f32 [256] each for `layer` */
    uint64_t part_va, cnt_va;   /* the cross-split partials and counters (prefill may reuse them) */
    uint32_t q_heads, kv_heads, q_per_cta; /* the cache's head shape, resolved (24 / 4 / 6 for the 27B) */
    uint32_t qkv_words, out_words;         /* f32 per token row of the q|k|v input and of the output */
} OmegaGpuDecodeAttn256CacheView;
int omega_gpu_decode_attn256_cache_view(const OmegaGpuDecodeAttn256Cache *c, uint32_t layer, OmegaGpuDecodeAttn256CacheView *out);

/* ---- measurement knobs (tests; each clears the kernel cache) ---- */
void omega_gpu_decode_attn256_set_depth(uint32_t depth);          /* 0 = default (OMEGA_GPU_DATTN256_DEPTH) */
void omega_gpu_decode_attn256_set_per_split(uint32_t per_split);  /* 0 = default; multiple of 256 */
void omega_gpu_decode_attn256_test_set_mutant(int mutant);        /* TEST ONLY: OmegaDecodeAttn256Spec.mutant */

/* ---- host side ---- */
/* RoPE table row for `pos`: cos[128] then sin[128], f32 from double; pairs j < rotary_dim/2 rotate with
 * inv_freq[j] = theta^(-2j/rotary_dim), the rest are cos 1, sin 0 */
void omega_dattn256_rope_row(uint32_t pos, double theta, uint32_t rotary_dim, float *row256);
/* the model dim held by kernel slot i (see above); rotary_dim even, 2..256 */
static inline uint32_t omega_dattn256_perm_slot(uint32_t i, uint32_t rotary_dim) {
    const uint32_t P = rotary_dim / 2u;
    if (i < 128u) return i < P ? i : rotary_dim + (i - P);
    const uint32_t j = i - 128u;
    return j < P ? P + j : i;
}
/* dst[i] = src[perm_slot(i)] for 256 f32 (q, k, norm weights); src and dst must not overlap */
static inline void omega_dattn256_perm_vec(const float *src, float *dst, uint32_t rotary_dim) {
    for (uint32_t i = 0; i < 256u; i++) dst[i] = src[omega_dattn256_perm_slot(i, rotary_dim)];
}
static inline void omega_dattn256_perm_vec_bf16(const uint16_t *src, uint16_t *dst, uint32_t rotary_dim) {
    for (uint32_t i = 0; i < 256u; i++) dst[i] = src[omega_dattn256_perm_slot(i, rotary_dim)];
}
/* The double-precision reference of the whole layer step lives in tests/gpu_decode_attn256_test.c (libm). */

#endif
