#ifndef OMEGA_GPU_Q36_GRAPH_H
#define OMEGA_GPU_Q36_GRAPH_H
/*
 * omega_gpu_q36_graph: one Qwen3.6-35B-A3B decode token as ONE command list (sc L306, 2026-10-20). The layer mixers are the
 * Qwen3.8-27B graph's (omega_gpu_q38_graph.h) at the 35B's shapes; the MLP is the 30B decode graph's sparse block
 * (omega_gpu_decode_graph.h) with a ninth group, the shared expert:
 *
 *   (layer 0 only)  NORM      xq = RMSNorm(x) * in_norm[0] * 2^64                    (the FP8 GEMV x' row)
 *   GDN layer l:    fp8 mix   mix[12288] = W_qkvz[l] . xq   (in_proj_qkv 8192 | in_proj_z 4096)
 *                   bf16 ab   ab[64] = W_ab[l] . xq          (a/b rows interleaved, uploaded * 2^-64: plain ab out)
 *                   gdn       att[4096] = GDN step(mix, mix + 8192, ab) * 2^64       omega_gpu_gdn_record (ratio 2)
 *   attention l:    fp8 mix   mix[9216] = W_qgkv[l] . xq    (q_proj 16 x [q 256 | gate 256], k_proj 512, v_proj 512)
 *                   GATHER    agg[9216] = permuted q | permuted k | v (5120, the attention kernel's row) | gate (4096)
 *                   attn      att[4096] = attention(agg, KV[l], pos) * 2^64          omega_gpu_decode_attn256_record (16 / 2)
 *                   GATE      att *= sigmoid(agg + 5120)
 *   every layer:    fp8 o     o[2048] = W_o[l] . att
 *                   ADD       x += o
 *                   router    xr = RMSNorm(x) * post_norm[l] * 2^64, ids[9], wts[9], tables   omega_gpu_router_record
 *                             (slot 8 = the shared expert, weight sigmoid(shared_expert_gate . xn); omega_gpu_router_alloc_shape)
 *                   fp8 gu    gu[9][1024] = W_gu[ids] . xr   (gate 512 | up 512; tensor of experts + 1, the last the shared one)
 *                   SWIGLU    act[9][512] = silu(gate) * up * 2^64
 *                   fp8 down  dn[9][2048] = W_dn[ids] . act[g]
 *                   COMBINE|NORM  x += sum_g wts[g] * dn[g] (9 groups); xq = RMSNorm(x) * in_norm[l + 1] * 2^64, or (last
 *                             layer) hidden = RMSNorm(x) * final_norm
 *   final:          bf16 lm_head  logits = W_lm . hidden;  ARGMAX x2 (122 CTAs over 248320, then 1)
 *                   (or the exact two-pass head, omega_gpu_q36_graph_set_exact_head: + 5 launches)
 *
 *   Launches: GDN layer 10, attention layer 11, plus 1 + 3.
 * Norm weights are passed already folded to (1 + w) (the model's zero-centred RMSNorm); the post-attention norm lives in the
 * router (omega_gpu_router_set_layer), the router's directories in the caller's omega_gpu_router_set_directory_fp8.
 * Memory, threading and kernel caches: as omega_gpu_q38_graph.h.
 *
 * ============ C signatures (for the Rust binding) ============
 *   int  omega_gpu_q36_graph_create(const OmegaGpuQ36GraphConfig *cfg, OmegaGpuQ36Graph **out);
 *   void omega_gpu_q36_graph_free(OmegaGpuQ36Graph *g);
 *   int  omega_gpu_q36_graph_set_layer(g, layer, mix, ab, o, gate_up, down, in_norm);
 *   int  omega_gpu_q36_graph_set_final_norm(g, final_norm);
 *   int  omega_gpu_q36_graph_token(g, x_embed, pos, &token, &stats);
 *   int  omega_gpu_q36_graph_read(g, x, hidden, logits);
 *   int  omega_gpu_q36_graph_read_buf(g, which, out, n);
 * Return codes: the decode graph's OMEGA_GPU_DGRAPH_*.
 */
#include "omega_gpu_decode_graph.h"
#include "omega_gpu_decode_attn256_api.h"
#include "omega_gpu_gdn_api.h"
#include "omega_gpu_router_api.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_GPU_Q36_HIDDEN 2048u
#define OMEGA_GPU_Q36_EXPERT_INTER 512u  /* routed and shared expert width */
#define OMEGA_GPU_Q36_GROUPS 9u          /* 8 routed + the shared expert */
#define OMEGA_GPU_Q36_GDN_MIX 12288u     /* in_proj_qkv 8192 | in_proj_z 4096 */
#define OMEGA_GPU_Q36_GDN_QKV 8192u
#define OMEGA_GPU_Q36_AB 64u             /* 32 heads x {a, b} */
#define OMEGA_GPU_Q36_ATT_MIX 9216u      /* q_proj 8192 | k_proj 512 | v_proj 512 */
#define OMEGA_GPU_Q36_ATT_QKV 5120u      /* the attention kernel's row: q 16 x 256 | k 2 x 256 | v 2 x 256 */
#define OMEGA_GPU_Q36_MIXER_OUT 4096u    /* GDN 32 x 128 = attention 16 x 256 */
#define OMEGA_GPU_Q36_ROTARY 64u
#define OMEGA_GPU_Q36_MAX_LAYERS 64u
/* Heads per stroke (conversations through one command list): a 40-layer token records up to 11 * 40 + 4 = 444 launches,
 * and the list holds OMEGA_GPU_CMDLIST_MAX_LAUNCHES (4096), so eight (3552; the submit splits a list over the ring as it
 * needs). v1 reads the weights once per head per stroke; v2 batches the dense GEMVs at M = heads, not the list grows further. */
#define OMEGA_GPU_Q36_MAX_HEADS 8u

typedef struct {
    uint32_t layers;                    /* 1..64 (a model prefix in tests) */
    uint64_t attn_mask;                 /* bit l set: model layer l is full attention, else GDN */
    uint32_t vocab;                     /* lm_head rows (248320), <= 2048 * 2048 */
    float eps;                          /* rms_norm_eps (1e-6) */
    float eps_l2;                       /* GDN q/k l2norm eps (1e-6) */
    OmegaGpuGdn *gdn;                   /* omega_gpu_gdn_alloc_ratio(.., 2, ..), >= GDN layers entries, parameters set */
    OmegaGpuDecodeAttn256Cache *attn;   /* omega_gpu_decode_attn256_cache_alloc_shape(.., 16, 2, 4, ..), q/k norms set */
    OmegaGpuRouter *router;             /* omega_gpu_router_alloc_shape(layers, 256, 1, ..), every layer and directory set */
    const OmegaGpuBf16Tensor *lm_head;  /* [vocab][2048] */
} OmegaGpuQ36GraphConfig;

typedef struct OmegaGpuQ36Graph OmegaGpuQ36Graph;

int omega_gpu_q36_graph_create(const OmegaGpuQ36GraphConfig *cfg, OmegaGpuQ36Graph **out);
void omega_gpu_q36_graph_free(OmegaGpuQ36Graph *g);
/* Shapes are checked. GDN layer: mix [1][12288][2048], ab bf16 [64][2048] (row 2h = in_proj_a[h], 2h + 1 = in_proj_b[h],
 * each value * 2^-64), o [1][2048][4096]. Attention layer: mix [1][9216][2048], ab NULL, o [1][2048][4096].
 * Both: gate_up [257][1024][2048], down [257][2048][512] (expert 256 = the shared expert), in_norm f32 [2048] as (1 + w).
 * The tensors stay owned by the caller. */
int omega_gpu_q36_graph_set_layer(OmegaGpuQ36Graph *g, uint32_t layer, const OmegaGpuFp8Tensor *mix, const OmegaGpuBf16Tensor *ab,
                                  const OmegaGpuFp8Tensor *o, const OmegaGpuFp8Tensor *gate_up, const OmegaGpuFp8Tensor *down,
                                  const float *in_norm);
int omega_gpu_q36_graph_set_final_norm(OmegaGpuQ36Graph *g, const float *final_norm); /* f32 [2048] as (1 + w) */
/* Optional FP8 lm_head: as omega_gpu_q38_graph_set_lm_head_fp8 ([vocab][2048]). */
int omega_gpu_q36_graph_set_lm_head_fp8(OmegaGpuQ36Graph *g, const OmegaGpuFp8Tensor *head);
/* Strokes: N heads (conversations) through ONE command list, one submit. create_heads lays out N slots of x, hidden, logits
 * and token; head 0 uses cfg.gdn / cfg.attn, every head (0 included) can be pointed at other state objects by set_head_state (same shapes;
 * head swap = pointing a slot at another conversation's objects). The router, the weights and the per-layer scratch are
 * shared: launches run in order (barriers by default) and every head's layer launches stay contiguous. The one value that
 * crosses layers, the fused COMBINE|NORM's x' for the next layer (xq), is per head (2026-10-21 02:50Z: shared, every head
 * read the last head's x'; the stroke gate caught it). create() == create_heads(cfg, 1). */
int omega_gpu_q36_graph_create_heads(const OmegaGpuQ36GraphConfig *cfg, uint32_t heads, OmegaGpuQ36Graph **out);
int omega_gpu_q36_graph_set_head_state(OmegaGpuQ36Graph *g, uint32_t head, OmegaGpuGdn *gdn, OmegaGpuDecodeAttn256Cache *attn);
/* x_embed f32 [heads][2048], pos [heads] -> tokens [heads]; layer-major (every head's layer l before layer l + 1). */
int omega_gpu_q36_graph_stroke(OmegaGpuQ36Graph *g, const float *x_embed, const uint32_t *pos, uint32_t heads, uint32_t *tokens,
                               OmegaGpuDecodeGraphStats *st);
int omega_gpu_q36_graph_begin_stroke(OmegaGpuQ36Graph *g, const float *x_embed, uint32_t heads);
int omega_gpu_q36_graph_record_head_layer(OmegaGpuQ36Graph *g, uint32_t head, uint32_t layer, uint32_t pos);
int omega_gpu_q36_graph_record_head_final(OmegaGpuQ36Graph *g, uint32_t head);
int omega_gpu_q36_graph_end_stroke(OmegaGpuQ36Graph *g, uint32_t *tokens, OmegaGpuDecodeGraphStats *st);
/* As read(), for head h of the last stroke. */
int omega_gpu_q36_graph_read_head(OmegaGpuQ36Graph *g, uint32_t head, float *x, float *hidden, float *logits);
/* HUM L314 exact two-pass head (bf16 lm_head only; ignored while an FP8 lm_head is set). screen: the fp8b copy of lm_head
 * ([vocab][2048], one bf16 scale per 128 x 128 block); coef: f32 [vocab] Row bound coefficients, c[r] >= |w_r - w~_r| +
 * gamma(2100)(|w_r| + |w~_r|) (sc exact_head.rs). The final step then records
 *     XBOUND SCALE h8 = hidden * 2^64;  fp8 screen s = W~ . h8;  XBOUND lo / hi = s -/+ c|hidden|;  ARGMAX x2 tau = max lo;
 *     bf16 MASK logits = W_lm . hidden where hi >= tau (bit-identical to the plain head), -inf elsewhere;  ARGMAX x2
 * so the greedy token is the plain head's by construction, but the logits read back are -inf off the candidates: use it
 * only for greedy tokens (set_exact_head_on(g, 0) for sampled ones). Copies coef; screen must outlive g. NULL screen
 * removes it. On by default once set. In a stroke every head's final step
 * uses it in turn (one scratch, launches in order); on / off applies to the whole stroke. In batch mode (set_batch) the
 * final is the XBATCH plain head (lm_head once per group of heads) and the exact head is not used. */
int omega_gpu_q36_graph_set_exact_head(OmegaGpuQ36Graph *g, const OmegaGpuFp8Tensor *screen, const float *coef);
void omega_gpu_q36_graph_set_exact_head_on(OmegaGpuQ36Graph *g, int on); /* between tokens */
int omega_gpu_q36_graph_exact_head_active(const OmegaGpuQ36Graph *g);     /* set, on, and no FP8 lm_head */
/* One whole token: x_embed (f32 [2048]) -> greedy token at position pos. */
int omega_gpu_q36_graph_token(OmegaGpuQ36Graph *g, const float *x_embed, uint32_t pos, uint32_t *token, OmegaGpuDecodeGraphStats *st);
int omega_gpu_q36_graph_begin_token(OmegaGpuQ36Graph *g, const float *x_embed);
int omega_gpu_q36_graph_record_layer(OmegaGpuQ36Graph *g, uint32_t layer, uint32_t pos);
int omega_gpu_q36_graph_record_final(OmegaGpuQ36Graph *g);
int omega_gpu_q36_graph_end_token(OmegaGpuQ36Graph *g, uint32_t *token, OmegaGpuDecodeGraphStats *st);
/* Takes the lock; copies the device x, hidden and logits (vocab values) of the last token; any pointer may be NULL. */
int omega_gpu_q36_graph_read(OmegaGpuQ36Graph *g, float *x, float *hidden, float *logits);
enum { OMEGA_GPU_Q36_BUF_X, OMEGA_GPU_Q36_BUF_XQ, OMEGA_GPU_Q36_BUF_MIX, OMEGA_GPU_Q36_BUF_AB, OMEGA_GPU_Q36_BUF_AGG,
       OMEGA_GPU_Q36_BUF_ATT, OMEGA_GPU_Q36_BUF_O, OMEGA_GPU_Q36_BUF_GU, OMEGA_GPU_Q36_BUF_ACT, OMEGA_GPU_Q36_BUF_HIDDEN,
       OMEGA_GPU_Q36_BUF_DN, OMEGA_GPU_Q36_BUF_COUNT };
int omega_gpu_q36_graph_read_buf(OmegaGpuQ36Graph *g, int which, float *out, uint32_t n);
/* The gather table (u32 [9216]): agg[n] = mix[tab[n]]. */
void omega_q36_gather_table(uint32_t *tab);
/* the FP8 GEMM's x' lane permutation as a GATHER table (k a multiple of 512) */
void omega_q36_perm_table(uint32_t *tab, uint32_t k);

/* The batch (v2, L312 "the ignition"): with set_batch(g, 1), stroke() records each layer for every conversation at once:
 * the dense GEMVs become one grouped GEMM over the N rows (mix, o: one weight read per layer for all N), the expert blocks
 * one grouped GEMM over the N x 9 rows sorted by expert (GROUP kernel: every expert any conversation picked is read once),
 * the per-conversation state kernels (GDN, attention, router, COMBINE) stay per head. Values are those of the per-head
 * stroke (the gate: every head word-identical to its run alone). The final head (lever 1): the bf16 lm_head read once per
 * group of up to OMEGA_BF16_MAX_XBATCH heads (the XBATCH GEMV form), then the argmax per head; the FP8 head stays per head.
 * set_batch pins the GEMM and XBATCH kernels (session, once); AIEN_OMEGA_Q36_BATCH=1 at create does the same.
 * record_layer_heads is the per-layer step stroke() uses. */
int omega_gpu_q36_graph_set_batch(OmegaGpuQ36Graph *g, int on);
int omega_gpu_q36_graph_batch(const OmegaGpuQ36Graph *g);
int omega_gpu_q36_graph_record_layer_heads(OmegaGpuQ36Graph *g, uint32_t layer, const uint32_t *pos);

enum {
    OMEGA_GPU_Q36_K_NORM = 0, OMEGA_GPU_Q36_K_MIX, OMEGA_GPU_Q36_K_AB, OMEGA_GPU_Q36_K_GDN, OMEGA_GPU_Q36_K_GATHER, OMEGA_GPU_Q36_K_ATTN,
    OMEGA_GPU_Q36_K_GATE, OMEGA_GPU_Q36_K_O, OMEGA_GPU_Q36_K_RES, OMEGA_GPU_Q36_K_ROUTER, OMEGA_GPU_Q36_K_GATE_UP, OMEGA_GPU_Q36_K_SWIGLU,
    OMEGA_GPU_Q36_K_DOWN, OMEGA_GPU_Q36_K_LM_HEAD, OMEGA_GPU_Q36_K_ARGMAX, OMEGA_GPU_Q36_K_PERM, OMEGA_GPU_Q36_K_GROUP, OMEGA_GPU_Q36_K_COUNT
};
void omega_gpu_q36_graph_set_stamps(OmegaGpuQ36Graph *g, int on);
int omega_gpu_q36_graph_stamps(const OmegaGpuQ36Graph *g, uint64_t *stamps, uint8_t *kinds, uint32_t cap);
const char *omega_gpu_q36_graph_kind_name(uint8_t kind);
void omega_gpu_q36_graph_set_flush_each(OmegaGpuQ36Graph *g, int on);
/* TEST ONLY: 1 SWIGLU up half off by one, 2 GATHER index + 1, 3 GATE dropped, 4 COMBINE skips the shared expert,
 * 5 the exact head's MASK keeps the rows it should drop and drops the candidates (the greedy gate must catch it),
 * 6 the exact head's XBOUND drops the c|h| bound (too weak to change most tokens: the host gate saw one golden of five
 * diverge at its full scale x0). 0 none. */
void omega_gpu_q36_graph_test_set_mutant(OmegaGpuQ36Graph *g, int mutant);

#ifdef __cplusplus
}
#endif
#endif
