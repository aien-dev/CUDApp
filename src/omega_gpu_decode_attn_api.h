#ifndef OMEGA_GPU_DECODE_ATTN_API_H
#define OMEGA_GPU_DECODE_ATTN_API_H
/*
 * omega_gpu_decode_attn_api: one-launch decode attention for Qwen3-30B-A3B (OWN-30B cut 8, 2026-10-13).
 * C signatures for the Rust side first; the kernel is described in omega_blackwell_codegen_attn.h.
 *
 *   int  omega_gpu_decode_attn_cache_alloc(uint32_t layers, uint32_t max_pos, double rope_theta, OmegaGpuDecodeAttnCache **out);
 *        GPU-resident bf16 KV cache for `layers` layers of 4 KV heads x `max_pos` positions x 128 (one allocation per
 *        layer, 2 * 4 * max_pos * 256 bytes + slack), the RoPE table (f32 cos/sin per position, built in double from
 *        rope_theta), the per-layer q_norm/k_norm weights, the cross-split partials and counters. Opens the session.
 *   void omega_gpu_decode_attn_cache_free(OmegaGpuDecodeAttnCache *c);
 *   int  omega_gpu_decode_attn_set_norms(OmegaGpuDecodeAttnCache *c, uint32_t layer, const float *q_norm, const float *k_norm);
 *        128 f32 each (the checkpoint's bf16 q_norm.weight / k_norm.weight widened by the caller).
 *   int  omega_gpu_decode_attn(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t pos, const float *qkv, float eps,
 *                              float out_scale, float *out, OmegaGpuDecodeAttnInfo *info);
 *        qkv: f32 [5120] as the fused q/k/v GEMV writes it (q 32 heads x 128, then k 4 x 128, then v 4 x 128), pos: the
 *        new token's position (< max_pos; the context is positions 0..pos, 0..pos-1 must already be in the cache for
 *        that layer). The kernel appends K/V at pos and writes out[32][128] * out_scale (2^64 makes it the x' row of
 *        the FP8 o_proj GEMV, omega_gpu_fp8_gemv_* with x_prescaled; 1.0 gives plain attention output).
 *        Returns OMEGA_GPU_DATTN_OK or a code below; the chip failure text is in omega_gpu_decode_attn_last_error().
 *   Test helpers: cache_write moves `count` consecutive bf16 K and V rows (128 each) of a layer and KV head into the
 *   cache in one sync; cache_read returns one row of each.
 *   The double-precision oracle the tests compare against (omega_decode_attn_reference) is defined in the test.
 *
 * Launch shape: grid (splits, 4), 256 threads, splits = ceil((pos + 1) / per_split), per_split a multiple of 256
 * (default OMEGA_GPU_DATTN_PER_SPLIT; omega_gpu_decode_attn_set_per_split overrides). Every buffer is GPU-uncached
 * host-visible session memory (nvos.h:1117 rule), launched through omega_gpu_session_launch like the FP8 GEMV.
 * The counters must be zero before a launch; the last CTA of each KV head resets its counter, and the API verifies
 * the four words are zero after each call (a non-zero word means a CTA never arrived: CHIP_FAIL).
 */
#include "omega_blackwell_codegen_attn.h"
#include <stdint.h>
#include <stddef.h>

enum {
    OMEGA_GPU_DATTN_OK = 0,
    OMEGA_GPU_DATTN_BAD_ARGS = -1,
    OMEGA_GPU_DATTN_TOO_LARGE = -2,
    OMEGA_GPU_DATTN_CODEGEN_FAIL = -3,
    OMEGA_GPU_DATTN_CHIP_FAIL = -4
};

#define OMEGA_GPU_DATTN_KV_HEADS 4u
#define OMEGA_GPU_DATTN_Q_HEADS 32u
#define OMEGA_GPU_DATTN_QKV_WORDS (32u * 128u + 2u * 4u * 128u)
#define OMEGA_GPU_DATTN_PER_SPLIT 256u
#define OMEGA_GPU_DATTN_MAX_LAYERS 64u
#define OMEGA_GPU_DATTN_MAX_POS 65536u

typedef struct OmegaGpuDecodeAttnCache OmegaGpuDecodeAttnCache;

typedef struct {
    uint32_t layer, pos, splits, per_split, depth;
    uint32_t grid_x, grid_y, threads;
    uint64_t kv_bytes;          /* K + V bytes the launch streams: 2 * 4 * (pos + 1) * 256 */
    uint64_t elapsed_ns;        /* marker-to-marker chip time */
    uint64_t call_ns;           /* whole call */
    uint32_t completion_marker;
    int kernel_cache_hit;
} OmegaGpuDecodeAttnInfo;

int omega_gpu_decode_attn_cache_alloc(uint32_t layers, uint32_t max_pos, double rope_theta, OmegaGpuDecodeAttnCache **out);
void omega_gpu_decode_attn_cache_free(OmegaGpuDecodeAttnCache *c);
int omega_gpu_decode_attn_set_norms(OmegaGpuDecodeAttnCache *c, uint32_t layer, const float *q_norm, const float *k_norm);
int omega_gpu_decode_attn(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t pos, const float *qkv, float eps,
                          float out_scale, float *out, OmegaGpuDecodeAttnInfo *info);
/* Record path (OWN-30B W-L2a, decode graph; omega_gpu_cmdlist.h): the launch of omega_gpu_decode_attn appended to cl
 * instead of submitted. qkv_va: f32 [5120] on the device (the fused q|k|v GEMV output); out_va: f32 [32 * 128] written
 * * out_scale when the list runs (2^64 = the o_proj FP8 x' row). pos and the split count are fixed at record time.
 * The caller holds omega_gpu_session_lock(). Nothing staged, poisoned or read back; the split counters rely on the
 * kernel's own reset (checked after the list by omega_gpu_decode_attn_counters_check). A kernel cache miss that would
 * evict the resident kernel is refused (warm up with one non-record call first). */
struct OmegaGpuCmdList;
int omega_gpu_decode_attn_record(struct OmegaGpuCmdList *cl, OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t pos, uint64_t qkv_va,
                                 float eps, float out_scale, uint64_t out_va);
/* Caller holds the lock, after the list ran: OK when the four split counters are zero, else CHIP_FAIL. */
int omega_gpu_decode_attn_counters_check(OmegaGpuDecodeAttnCache *c);
int omega_gpu_decode_attn_cache_write(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t kv_head, uint32_t pos, uint32_t count,
                                      const uint16_t *k_bf16, const uint16_t *v_bf16); /* count rows from pos, one sync */
int omega_gpu_decode_attn_cache_read(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t kv_head, uint32_t pos,
                                     uint16_t *k_bf16, uint16_t *v_bf16);
/* Read-only view of a layer of the cache for the prefill attention (omega_gpu_prefill_attn_api.h): device addresses only. */
typedef struct {
    uint32_t layers, max_pos;
    uint64_t k_va, v_va;        /* K and V plane bases of `layer` (row (kv_head, pos) at + (kv_head * max_pos + pos) * 256) */
    uint64_t rope_va;           /* f32 [max_pos][cos 64 | sin 64] */
    uint64_t q_norm_va, k_norm_va; /* f32 [128] each for `layer` */
} OmegaGpuDecodeAttnCacheView;
int omega_gpu_decode_attn_cache_view(const OmegaGpuDecodeAttnCache *c, uint32_t layer, OmegaGpuDecodeAttnCacheView *out);
const char *omega_gpu_decode_attn_last_error(void);
const char *omega_gpu_decode_attn_rc_name(int rc);

/* ---- measurement knobs (tests; each clears the kernel cache) ---- */
void omega_gpu_decode_attn_set_depth(uint32_t depth);          /* 0 = default (OMEGA_GPU_DATTN_DEPTH) */
void omega_gpu_decode_attn_set_per_split(uint32_t per_split);  /* 0 = default; multiple of 256 */
void omega_gpu_decode_attn_test_set_mutant(int mutant);        /* TEST ONLY: OmegaDecodeAttnSpec.mutant */
#define OMEGA_GPU_DATTN_DEPTH 4u

/* ---- host side ---- */
/* bf16 round-to-nearest-even of an f32, the rounding the kernel applies to the appended K/V */
uint16_t omega_dattn_bf16_rne(float f);
float omega_dattn_bf16_f32(uint16_t b);
/* RoPE table row for `pos`: cos[64] then sin[64], f32 from double, inv_freq[i] = theta^(-2i/128) */
void omega_dattn_rope_row(uint32_t pos, double theta, float *row128);
/* The double-precision reference of the whole layer step (omega_decode_attn_reference) lives in
 * tests/gpu_decode_attn_test.c: it needs libm and the library links without it. */

#endif
