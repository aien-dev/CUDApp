#ifndef OMEGA_GPU_GDN_API_H
#define OMEGA_GPU_GDN_API_H
/*
 * omega_gpu_gdn_api: the Gated DeltaNet decode step of Qwen3.8-27B in one launch per layer (omega#542, 2026-10-18).
 * The kernel is described in omega_blackwell_codegen_gdn.h; the maths is sc qwen38.rs `delta_net`.
 *
 *   int  omega_gpu_gdn_alloc(uint32_t layers, OmegaGpuGdn **out);
 *        Device-resident per-layer parameters, conv tails and recurrent states (zeroed), plus the launch buffer.
 *   int  omega_gpu_gdn_set_layer(OmegaGpuGdn *g, uint32_t layer, const float *conv_w, const float *a_log,
 *                                const float *dt_bias, const float *norm_w);
 *        conv_w: linear_attn.conv1d.weight [10240][1][4] widened to f32 ([10240][4]); a_log, dt_bias: [48];
 *        norm_w: linear_attn.norm.weight [128]. negA = -expf(a_log) is computed here (float).
 *   int  omega_gpu_gdn_set_state / omega_gpu_gdn_get_state(OmegaGpuGdn *g, uint32_t layer, float *tail, float *state);
 *        The reference layouts: tail [3][10240] (oldest first, qwen38.rs conv_tail), state [48][128 k][128 v].
 *        get_state refuses (CHIP_FAIL) when the three per-head copies of a key head's q/k tail disagree.
 *   int  omega_gpu_gdn_reset(OmegaGpuGdn *g);   every layer's tail and state to zero (a new sequence)
 *   int  omega_gpu_gdn_step(OmegaGpuGdn *g, uint32_t layer, const float *qkv, const float *z, const float *ab,
 *                           float eps_l2, float eps_norm, float in_scale, float out_scale, float *out,
 *                           OmegaGpuGdnInfo *info);
 *        qkv [10240] and z [6144] are the in_proj outputs (times 1 / in_scale: the kernel multiplies them by in_scale);
 *        ab [48][2] = {a[h], b[h]}: the in_proj_a and in_proj_b rows stacked interleaved (row 2h = a_h, 2h + 1 = b_h)
 *        give it from one bf16 GEMV. out [6144] = the out_proj input times out_scale. Updates the layer's tail and
 *        state on the device. eps_l2 is 1e-6 in the reference; eps_norm is rms_norm_eps.
 *   int  omega_gpu_gdn_record(cl, g, layer, qkv_va, z_va, ab_va, eps_l2, eps_norm, in_scale, out_scale, &out_va);
 *        The same launch appended to a command list (caller holds omega_gpu_session_lock()); inputs are device
 *        addresses, out lands in the launch buffer (out_va). A kernel cache miss that would evict is refused.
 *
 * Every buffer is GPU-uncached host-visible session memory (nvos.h:1117 rule), launched through
 * omega_gpu_session_launch like the router. The output is poisoned before the launch; a surviving poison word is
 * CHIP_FAIL.
 */
#include "omega_blackwell_codegen_gdn.h"
#include <stddef.h>
#include <stdint.h>

enum {
    OMEGA_GPU_GDN_OK = 0,
    OMEGA_GPU_GDN_BAD_ARGS = -1,
    OMEGA_GPU_GDN_TOO_LARGE = -2,
    OMEGA_GPU_GDN_CODEGEN_FAIL = -3,
    OMEGA_GPU_GDN_CHIP_FAIL = -4
};

#define OMEGA_GPU_GDN_MAX_LAYERS 64u
#define OMEGA_GPU_GDN_TAIL_FLOATS (3u * OMEGA_GDN_CONV_DIM)                                   /* reference tail */
#define OMEGA_GPU_GDN_STATE_FLOATS (OMEGA_GDN_VALUE_HEADS * OMEGA_GDN_HEAD_DIM * OMEGA_GDN_HEAD_DIM) /* 786432 */

typedef struct OmegaGpuGdn OmegaGpuGdn;

typedef struct {
    uint32_t layer, threads, grid_x, grid_y, insns, gprs;
    uint64_t state_bytes;       /* state bytes read and written: 2 * 48 * 128 * 128 * 4 */
    uint64_t elapsed_ns;        /* marker-to-marker chip time */
    uint64_t call_ns;           /* whole call */
    uint32_t completion_marker;
    int kernel_cache_hit;
} OmegaGpuGdnInfo;

int omega_gpu_gdn_alloc(uint32_t layers, OmegaGpuGdn **out);
void omega_gpu_gdn_free(OmegaGpuGdn *g);
int omega_gpu_gdn_set_layer(OmegaGpuGdn *g, uint32_t layer, const float *conv_w, const float *a_log, const float *dt_bias,
                            const float *norm_w);
int omega_gpu_gdn_reset(OmegaGpuGdn *g);
int omega_gpu_gdn_set_state(OmegaGpuGdn *g, uint32_t layer, const float *tail, const float *state);
int omega_gpu_gdn_get_state(OmegaGpuGdn *g, uint32_t layer, float *tail, float *state);
int omega_gpu_gdn_step(OmegaGpuGdn *g, uint32_t layer, const float *qkv, const float *z, const float *ab, float eps_l2,
                       float eps_norm, float in_scale, float out_scale, float *out, OmegaGpuGdnInfo *info);
struct OmegaGpuCmdList;
int omega_gpu_gdn_record(struct OmegaGpuCmdList *cl, OmegaGpuGdn *g, uint32_t layer, uint64_t qkv_va, uint64_t z_va,
                         uint64_t ab_va, float eps_l2, float eps_norm, float in_scale, float out_scale, uint64_t *out_va);
/* Test helper: the out row the last launch wrote to the launch buffer (after a record + submit, for example). */
int omega_gpu_gdn_read_out(OmegaGpuGdn *g, float *out);
const char *omega_gpu_gdn_last_error(void);
const char *omega_gpu_gdn_rc_name(int rc);

/* ---- prefill (omega#543, 2026-10-18): T >= 3 new tokens of one layer in three launches (prep, scan, post; see
 * omega_blackwell_codegen_gdn.h), continuing from the layer's tail and state exactly as T decode steps would, and leaving
 * them where the next prefill chunk or decode step picks them up. A long prompt is prefilled in chunks.
 *   int omega_gpu_gdn_prefill(g, layer, T, qkv [T][10240], z [T][6144], ab [T][48][2], eps_l2, eps_norm, in_scale,
 *                             out_scale, out [T][6144], info);   the decode step's row layouts, one row per token
 *   int omega_gpu_gdn_prefill_reserve(g, max_tokens);  sizes the prefill scratch now and pins it (the record path needs
 *        it: a recorded launch must never see its scratch move); a later prefill of more tokens is refused
 *   int omega_gpu_gdn_prefill_record(cl, g, layer, T, qkv_va, z_va, ab_va, eps_l2, eps_norm, in_scale, out_scale, &out_va);
 *        the three launches appended to a command list (caller holds the lock); out_va = the [T][6144] rows in the scratch
 * Scratch bytes for T tokens: T * (40960 + 3 * 24576 + 384) + ceil8(T) * 74496 (the per-token records dominate). */
#define OMEGA_GPU_GDN_PF_MAX_T 65536u

typedef struct {
    uint32_t layer, tokens;
    uint32_t insns[3], gprs[3];   /* prep, scan, post */
    uint64_t elapsed_ns[3];       /* marker-to-marker chip time per launch (host path) */
    uint64_t call_ns;
    uint32_t kernel_cache_hits;   /* 0..3 */
} OmegaGpuGdnPfInfo;

int omega_gpu_gdn_prefill(OmegaGpuGdn *g, uint32_t layer, uint32_t tokens, const float *qkv, const float *z, const float *ab,
                          float eps_l2, float eps_norm, float in_scale, float out_scale, float *out, OmegaGpuGdnPfInfo *info);
int omega_gpu_gdn_prefill_reserve(OmegaGpuGdn *g, uint32_t max_tokens);
int omega_gpu_gdn_prefill_record(struct OmegaGpuCmdList *cl, OmegaGpuGdn *g, uint32_t layer, uint32_t tokens, uint64_t qkv_va,
                                 uint64_t z_va, uint64_t ab_va, float eps_l2, float eps_norm, float in_scale, float out_scale,
                                 uint64_t *out_va);
/* Test helper: the out rows the last prefill wrote (after a record + submit). */
int omega_gpu_gdn_prefill_read_out(OmegaGpuGdn *g, uint32_t tokens, float *out);

/* Host layout helpers (no device; the test's simulator path uses them too). */
/* The parameter block (OMEGA_GDN_PAR_BYTES) from the checkpoint tensors (see set_layer). -1 on a NULL pointer. */
int omega_gdn_pack_params(const float *conv_w, const float *a_log, const float *dt_bias, const float *norm_w, uint8_t *block);
/* reference tail [3][10240] <-> device tail [48][384][4] (unpack: the q/k channels from head 3 * hk; -1 when the
 * three copies of a key head disagree bit for bit) */
void omega_gdn_pack_tail(const float *tail, float *dev);
int omega_gdn_unpack_tail(const float *dev, float *tail);

void omega_gpu_gdn_test_set_mutant(int mutant); /* TEST ONLY: OmegaGdnSpec.mutant; clears the kernel cache */

#endif
