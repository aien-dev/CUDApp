#ifndef OMEGA_GPU_Q38_GRAPH_H
#define OMEGA_GPU_Q38_GRAPH_H
/*
 * omega_gpu_q38_graph: one Qwen3.8-27B decode token as ONE command list (sc #642, 2026-10-18). The 30B's
 * omega_gpu_decode_graph (MoE, hidden 2048) is the template; this one is dense, hidden 5120, and mixes two layer kinds:
 * Gated DeltaNet (48 layers) and full attention with head_dim 256 (16 layers, every 4th).
 *
 *   (layer 0 only)  NORM      xq = RMSNorm(x) * in_norm[0] * 2^64                    (the FP8 GEMV x' row)
 *   GDN layer l:    fp8 mix   mix[16384] = W_qkvz[l] . xq   (in_proj_qkv 10240 | in_proj_z 6144)
 *                   bf16 ab   ab[96] = W_ab[l] . xq          (a/b rows interleaved, uploaded * 2^-64: plain ab out)
 *                   gdn       att[6144] = GDN step(mix, mix + 10240, ab) * 2^64      omega_gpu_gdn_record
 *   attention l:    fp8 mix   mix[14336] = W_qgkv[l] . xq   (q_proj 24 x [q 256 | gate 256], k_proj 1024, v_proj 1024)
 *                   GATHER    agg[14336] = permuted q | permuted k | v (8192, the attention kernel's row) | gate (6144)
 *                   attn      att[6144] = attention(agg, KV[l], pos) * 2^64          omega_gpu_decode_attn256_record
 *                   GATE      att *= sigmoid(agg + 8192)
 *   every layer:    fp8 o     o[5120] = W_o[l] . att
 *                   ADD|NORM  x += o; xr = RMSNorm(x) * post_norm[l] * 2^64
 *                   fp8 gu    gu[34816] = W_gu[l] . xr       (gate 17408 | up 17408)
 *                   SWIGLU    act[17408] = silu(gate) * up * 2^64
 *                   fp8 down  o[5120] = W_down[l] . act
 *                   ADD|NORM  x += o; xq = RMSNorm(x) * in_norm[l + 1] * 2^64, or (last layer) hidden = RMSNorm(x) * final_norm
 *   final:          bf16 lm_head  logits = W_lm . hidden;  ARGMAX x2 (122 CTAs over 248320, then 1)
 *
 *   Launches: GDN layer 9, attention layer 10, plus 1 + 3: 48 * 9 + 16 * 10 + 4 = 596 for the whole model.
 * Norm weights are passed already folded to (1 + w) (the model's zero-centred RMSNorm). The GDN and attention state live
 * in the caller's OmegaGpuGdn (one entry per GDN layer, in model order) and OmegaGpuDecodeAttn256Cache (one per attention
 * layer, in model order); the graph maps model layer l to its index of that kind.
 *
 * Memory: one omega_gpu_session_alloc buffer (GPU-uncached; nvos.h:1117 "For system memory this will not be coherent with
 * direct CPU mappings"): the host writes x before the submit and reads the token / logits after.
 * Threading and kernel caches: as omega_gpu_decode_graph.h (begin_token takes omega_gpu_session_lock(), end_token
 * releases it; the record APIs refuse a cache eviction, so warm each shape up with one non-record call first).
 *
 * ============ C signatures (for the Rust binding) ============
 *   int  omega_gpu_q38_graph_create(const OmegaGpuQ38GraphConfig *cfg, OmegaGpuQ38Graph **out);
 *   void omega_gpu_q38_graph_free(OmegaGpuQ38Graph *g);
 *   int  omega_gpu_q38_graph_set_layer(g, layer, mix, ab, o, gate_up, down, in_norm, post_norm);
 *   int  omega_gpu_q38_graph_set_final_norm(g, final_norm);
 *   int  omega_gpu_q38_graph_token(g, x_embed, pos, &token, &stats);
 *   int  omega_gpu_q38_graph_read(g, x, hidden, logits);
 *   int  omega_gpu_q38_graph_read_buf(g, which, out, n);
 * Return codes: the decode graph's OMEGA_GPU_DGRAPH_* (0 OK, -1 BAD_ARGS, -3 CODEGEN_FAIL, -4 CHIP_FAIL); the failing
 * stage is in omega_gpu_session_last_error().
 */
#include "omega_gpu_decode_graph.h"
#include "omega_gpu_decode_attn256_api.h"
#include "omega_gpu_gdn_api.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_GPU_Q38_HIDDEN 5120u
#define OMEGA_GPU_Q38_INTER 17408u
#define OMEGA_GPU_Q38_GDN_MIX 16384u     /* in_proj_qkv 10240 | in_proj_z 6144 */
#define OMEGA_GPU_Q38_GDN_QKV 10240u
#define OMEGA_GPU_Q38_AB 96u             /* 48 heads x {a, b} */
#define OMEGA_GPU_Q38_ATT_MIX 14336u     /* q_proj 12288 | k_proj 1024 | v_proj 1024 */
#define OMEGA_GPU_Q38_ATT_QKV 8192u      /* the attention kernel's row: q 24 x 256 | k 4 x 256 | v 4 x 256 */
#define OMEGA_GPU_Q38_MIXER_OUT 6144u    /* GDN 48 x 128 = attention 24 x 256 */
#define OMEGA_GPU_Q38_ROTARY 64u
#define OMEGA_GPU_Q38_MAX_LAYERS 64u

typedef struct {
    uint32_t layers;                    /* 1..64 (a model prefix in tests) */
    uint64_t attn_mask;                 /* bit l set: model layer l is full attention, else GDN */
    uint32_t vocab;                     /* lm_head rows (248320), <= 2048 * 2048 */
    float eps;                          /* rms_norm_eps (1e-6) */
    float eps_l2;                       /* GDN q/k l2norm eps (1e-6) */
    OmegaGpuGdn *gdn;                   /* >= (layers - attention layers) entries, parameters set */
    OmegaGpuDecodeAttn256Cache *attn;   /* >= attention layers entries, q/k norms set; rotary 64 */
    const OmegaGpuBf16Tensor *lm_head;  /* [vocab][5120] */
} OmegaGpuQ38GraphConfig;

typedef struct OmegaGpuQ38Graph OmegaGpuQ38Graph;

int omega_gpu_q38_graph_create(const OmegaGpuQ38GraphConfig *cfg, OmegaGpuQ38Graph **out);
void omega_gpu_q38_graph_free(OmegaGpuQ38Graph *g);
/* Shapes are checked. GDN layer: mix [1][16384][5120], ab bf16 [96][5120] (row 2h = in_proj_a[h], 2h + 1 = in_proj_b[h],
 * each value * 2^-64), o [1][5120][6144]. Attention layer: mix [1][14336][5120], ab NULL, o [1][5120][6144].
 * Both: gate_up [1][34816][5120], down [1][5120][17408], in_norm and post_norm f32 [5120] as (1 + w). The tensors stay
 * owned by the caller. */
int omega_gpu_q38_graph_set_layer(OmegaGpuQ38Graph *g, uint32_t layer, const OmegaGpuFp8Tensor *mix, const OmegaGpuBf16Tensor *ab,
                                  const OmegaGpuFp8Tensor *o, const OmegaGpuFp8Tensor *gate_up, const OmegaGpuFp8Tensor *down,
                                  const float *in_norm, const float *post_norm);
int omega_gpu_q38_graph_set_final_norm(OmegaGpuQ38Graph *g, const float *final_norm); /* f32 [5120] as (1 + w) */
/* Optional FP8 lm_head (qwen38/decode-speed): an FP8 tensor [vocab][5120] (experts 1, e.g. from omega_gpu_fp8_quantize_bf16)
 * replaces the bf16 lm_head in every later token (half the bytes); NULL goes back to bf16. The tensor must outlive its use.
 * While set, the device hidden buffer (OMEGA_GPU_Q38_BUF_HIDDEN) holds hidden x 2^64 (the FP8 GEMV input scale);
 * omega_gpu_q38_graph_read returns the unscaled hidden (exact power-of-two rescale). Logits are those of the quantised head:
 * callers own the drift study against the bf16 head. */
int omega_gpu_q38_graph_set_lm_head_fp8(OmegaGpuQ38Graph *g, const OmegaGpuFp8Tensor *head);
/* One whole token: x_embed (f32 [5120]) -> greedy token; pos: the token's position (attention layers append K/V there,
 * GDN layers advance their state). */
int omega_gpu_q38_graph_token(OmegaGpuQ38Graph *g, const float *x_embed, uint32_t pos, uint32_t *token, OmegaGpuDecodeGraphStats *st);
/* The same in steps (between begin and end only record calls on this thread). */
int omega_gpu_q38_graph_begin_token(OmegaGpuQ38Graph *g, const float *x_embed);
int omega_gpu_q38_graph_record_layer(OmegaGpuQ38Graph *g, uint32_t layer, uint32_t pos);
int omega_gpu_q38_graph_record_final(OmegaGpuQ38Graph *g);
int omega_gpu_q38_graph_end_token(OmegaGpuQ38Graph *g, uint32_t *token, OmegaGpuDecodeGraphStats *st);
/* Takes the lock; copies the device x, hidden and logits (vocab values) of the last token; any pointer may be NULL. */
int omega_gpu_q38_graph_read(OmegaGpuQ38Graph *g, float *x, float *hidden, float *logits);
/* Test and trace reads of one intermediate of the last token (f32, n values from the start of the buffer). */
enum { OMEGA_GPU_Q38_BUF_X, OMEGA_GPU_Q38_BUF_XQ, OMEGA_GPU_Q38_BUF_MIX, OMEGA_GPU_Q38_BUF_AB, OMEGA_GPU_Q38_BUF_AGG,
       OMEGA_GPU_Q38_BUF_ATT, OMEGA_GPU_Q38_BUF_O, OMEGA_GPU_Q38_BUF_GU, OMEGA_GPU_Q38_BUF_ACT, OMEGA_GPU_Q38_BUF_HIDDEN,
       OMEGA_GPU_Q38_BUF_COUNT };
int omega_gpu_q38_graph_read_buf(OmegaGpuQ38Graph *g, int which, float *out, uint32_t n);
/* The gather table (u32 [14336]): agg[n] = mix[tab[n]] (host side; the graph uploads it at create). */
void omega_q38_gather_table(uint32_t *tab);

/* ---- timeline (measurement; as the decode graph) ---- */
enum {
    OMEGA_GPU_Q38_K_NORM = 0, OMEGA_GPU_Q38_K_MIX, OMEGA_GPU_Q38_K_AB, OMEGA_GPU_Q38_K_GDN, OMEGA_GPU_Q38_K_GATHER, OMEGA_GPU_Q38_K_ATTN,
    OMEGA_GPU_Q38_K_GATE, OMEGA_GPU_Q38_K_O, OMEGA_GPU_Q38_K_RES, OMEGA_GPU_Q38_K_GATE_UP, OMEGA_GPU_Q38_K_SWIGLU, OMEGA_GPU_Q38_K_DOWN,
    OMEGA_GPU_Q38_K_LM_HEAD, OMEGA_GPU_Q38_K_ARGMAX, OMEGA_GPU_Q38_K_COUNT
};
void omega_gpu_q38_graph_set_stamps(OmegaGpuQ38Graph *g, int on);
int omega_gpu_q38_graph_stamps(const OmegaGpuQ38Graph *g, uint64_t *stamps, uint8_t *kinds, uint32_t cap);
const char *omega_gpu_q38_graph_kind_name(uint8_t kind);
/* 1: submit after every launch (the comparison path). */
void omega_gpu_q38_graph_set_flush_each(OmegaGpuQ38Graph *g, int on);
/* TEST ONLY: 1 SWIGLU up half off by one, 2 GATHER index + 1, 3 GATE dropped, 4 post-norm without its weight. 0 none. */
void omega_gpu_q38_graph_test_set_mutant(OmegaGpuQ38Graph *g, int mutant);

#ifdef __cplusplus
}
#endif
#endif
