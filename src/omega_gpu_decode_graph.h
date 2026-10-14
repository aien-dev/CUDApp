#ifndef OMEGA_GPU_DECODE_GRAPH_H
#define OMEGA_GPU_DECODE_GRAPH_H
/*
 * omega_gpu_decode_graph: one Qwen3-30B-A3B decode token as ONE command list (OWN-30B W-L2a, 2026-10-13).
 *
 * Today's token (sc qwen3_moe_omega.rs step()) is 6 synchronous launches per layer + lm_head with the residual x
 * round-tripping through the host between them. Here x stays on the device and every step is recorded into one
 * omega_gpu_cmdlist and submitted once:
 *
 *   (layer 0 only) RES NORM      xq = RMSNorm(x) * input_norm[0] * 2^64        (the FP8 x' row)
 *   per layer l:   fp8 qkv       qkv = W_qkv[l] . xq                            omega_gpu_fp8_gemv_grouped_record
 *                  decode attn   attn = attention(qkv, KV[l], pos) * 2^64      omega_gpu_decode_attn_record
 *                  fp8 o         o = W_o[l] . attn
 *                  RES ADD       x += o
 *                  router        xr = RMSNorm(x) * post_norm[l] * 2^64, ids, wts, tables   omega_gpu_router_record
 *                  fp8 gate_up   gu[8][1536] = W_gu[ids] . xr                   (router's table)
 *                  SWIGLU        act[8][768] = silu(gate) * up * 2^64
 *                  fp8 down      dn[8][2048] = W_dn[ids] . act[g]               (router's table)
 *                  RES COMBINE|NORM  x += sum_g wts[g] * dn[g]; then
 *                                xq = RMSNorm(x) * input_norm[l+1] * 2^64   (not the last layer), or
 *                                hidden = RMSNorm(x) * final_norm               (the last layer)
 *   final:         bf16 lm_head  logits = W_lm . hidden                         omega_gpu_bf16_gemv_record
 *                  ARGMAX x2     token = lowest index of max(logits)            (75 CTAs, then 1, for 151936)
 *
 *   Launches: 9 per layer + 4 (48 layers: 436). The glue kernels are omega_blackwell_codegen_dec.h (chip parity PASS).
 *   Numerics vs the host-glue path: identical operations except the RMSNorm (f32 FFMA sum of squares + MUFU.RSQ vs the
 *   host's f64 sum and 1/sqrt) and SwiGLU (MUFU EX2/RCP vs expf): logits close, not bit-identical; the argmax is the
 *   host's rule (lowest index on ties).
 *
 * Memory: every buffer is omega_gpu_session_alloc (GPU-uncached; nvos.h:1117 "For system memory this will not be
 * coherent with direct CPU mappings"): the host writes x before the submit and reads the token / logits after.
 *
 * Threading: begin_token takes omega_gpu_session_lock() and end_token releases it. Between them only record calls
 * (this file's and the four APIs' *_record) may run on this thread; a non-record API call would self-deadlock.
 * The four APIs refuse a record that would evict a cached kernel: run one plain call per shape (or one token with the
 * host path) first, or simply record the first token (an empty cache slot is filled, never evicted).
 *
 * ============ C signatures (for the Rust binding) ============
 *   int  omega_gpu_decode_graph_create(const OmegaGpuDecodeGraphConfig *cfg, OmegaGpuDecodeGraph **out);
 *   void omega_gpu_decode_graph_free(OmegaGpuDecodeGraph *g);
 *   int  omega_gpu_decode_graph_set_layer(OmegaGpuDecodeGraph *g, uint32_t layer, const OmegaGpuFp8Tensor *qkv,
 *            const OmegaGpuFp8Tensor *o, const OmegaGpuFp8Tensor *gate_up, const OmegaGpuFp8Tensor *down, const float *input_norm);
 *   int  omega_gpu_decode_graph_set_final_norm(OmegaGpuDecodeGraph *g, const float *final_norm);
 *   int  omega_gpu_decode_graph_token(OmegaGpuDecodeGraph *g, const float *x_embed, uint32_t pos, uint32_t *token,
 *                                     OmegaGpuDecodeGraphStats *st);       begin + every layer + final + end
 *   int  omega_gpu_decode_graph_begin_token(OmegaGpuDecodeGraph *g, const float *x_embed);
 *   int  omega_gpu_decode_graph_record_layer(OmegaGpuDecodeGraph *g, uint32_t layer, uint32_t pos);
 *   int  omega_gpu_decode_graph_record_final(OmegaGpuDecodeGraph *g);
 *   int  omega_gpu_decode_graph_end_token(OmegaGpuDecodeGraph *g, uint32_t *token, OmegaGpuDecodeGraphStats *st);
 *   int  omega_gpu_decode_graph_record_res(OmegaGpuDecodeGraph *g, uint32_t mode, uint64_t x_va, uint64_t a_va, uint64_t w_va,
 *                                          uint64_t nw_va, uint64_t y_va, float eps, float scale);
 *   int  omega_gpu_decode_graph_record_swiglu(OmegaGpuDecodeGraph *g, uint64_t gu_va, uint64_t act_va, float scale);
 *   struct OmegaGpuCmdList *omega_gpu_decode_graph_list(OmegaGpuDecodeGraph *g);
 *   void omega_gpu_decode_graph_vas(const OmegaGpuDecodeGraph *g, OmegaGpuDecodeGraphVas *v);
 *   int  omega_gpu_decode_graph_read(OmegaGpuDecodeGraph *g, float *x, float *hidden, float *logits);
 * Return codes: 0 OK, negative OMEGA_GPU_DGRAPH_* (the failing stage in omega_gpu_session_last_error()).
 */
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_router_api.h"
#include "omega_gpu_decode_attn_api.h"
#include "omega_gpu_bf16_gemv_api.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    OMEGA_GPU_DGRAPH_OK = 0,
    OMEGA_GPU_DGRAPH_BAD_ARGS = -1,  /* NULL, bad layer, shape mismatch, layer not set, call out of order */
    OMEGA_GPU_DGRAPH_CODEGEN_FAIL = -3,
    OMEGA_GPU_DGRAPH_CHIP_FAIL = -4, /* device, alloc, a record refused, submit, semaphore, counter or output check */
};

#define OMEGA_GPU_DGRAPH_HIDDEN 2048u
#define OMEGA_GPU_DGRAPH_QKV 5120u    /* 32 * 128 + 2 * 4 * 128 */
#define OMEGA_GPU_DGRAPH_ATTN 4096u   /* 32 * 128 */
#define OMEGA_GPU_DGRAPH_TOPK 8u
#define OMEGA_GPU_DGRAPH_INTER 768u
#define OMEGA_GPU_DGRAPH_MAX_LAYERS 64u

typedef struct {
    uint32_t layers;                    /* 1..OMEGA_GPU_DGRAPH_MAX_LAYERS, <= the router's and the KV cache's */
    uint32_t vocab;                     /* lm_head rows (151936 for Qwen3-30B-A3B), <= 2048 * 2048 */
    float eps;                          /* rms_norm_eps */
    int norm_topk;                      /* config.json norm_topk_prob (1 for Qwen3-30B-A3B) */
    OmegaGpuRouter *router;             /* per-layer router weights, post-attention norms and expert directories set */
    OmegaGpuDecodeAttnCache *attn;      /* KV cache with the per-layer q/k norms set */
    const OmegaGpuBf16Tensor *lm_head;  /* [vocab][2048] */
} OmegaGpuDecodeGraphConfig;

typedef struct {
    uint32_t launches;     /* launches recorded for the token */
    uint32_t submits;      /* submissions (1 normally; = launches with flush_each) */
    uint64_t chip_ns;      /* sum of the list submissions' first-submit-to-last-marker time */
    uint64_t record_ns;    /* host: begin_token to the submit (recording) */
    uint64_t total_ns;     /* host: begin_token to end_token return */
    uint32_t sems_done;    /* launch semaphores found DONE */
} OmegaGpuDecodeGraphStats;

/* Device addresses of the graph's buffers (all f32 unless noted). */
typedef struct {
    uint64_t x;       /* [2048] residual, device resident across the token */
    uint64_t xq;      /* [2048] input-norm x' (* 2^64) */
    uint64_t qkv;     /* [5120] */
    uint64_t attn;    /* [4096] attention output * 2^64 */
    uint64_t o;       /* [2048] */
    uint64_t gu;      /* [8][1536] */
    uint64_t act;     /* [8][768] * 2^64 */
    uint64_t dn;      /* [8][2048] */
    uint64_t hidden;  /* [2048] final norm output */
    uint64_t logits;  /* [ctas * 2048], tail padded with -1e30 */
    uint64_t token;   /* u32 greedy token, then f32 its logit */
} OmegaGpuDecodeGraphVas;

typedef struct OmegaGpuDecodeGraph OmegaGpuDecodeGraph;

int omega_gpu_decode_graph_create(const OmegaGpuDecodeGraphConfig *cfg, OmegaGpuDecodeGraph **out);
void omega_gpu_decode_graph_free(OmegaGpuDecodeGraph *g);
/* Shapes are checked: qkv [1][5120][2048], o [1][2048][4096], gate_up [128][1536][2048], down [128][2048][768].
 * input_norm: the checkpoint's input_layernorm.weight widened to f32 [2048]. The tensors stay owned by the caller. */
int omega_gpu_decode_graph_set_layer(OmegaGpuDecodeGraph *g, uint32_t layer, const OmegaGpuFp8Tensor *qkv, const OmegaGpuFp8Tensor *o,
                                     const OmegaGpuFp8Tensor *gate_up, const OmegaGpuFp8Tensor *down, const float *input_norm);
int omega_gpu_decode_graph_set_final_norm(OmegaGpuDecodeGraph *g, const float *final_norm);

/* One whole token: x_embed (f32 [2048], the token's embedding row) -> greedy token. pos: the token's position. */
int omega_gpu_decode_graph_token(OmegaGpuDecodeGraph *g, const float *x_embed, uint32_t pos, uint32_t *token, OmegaGpuDecodeGraphStats *st);

/* The same in steps. begin_token takes the session lock, empties the list and (x_embed non-NULL) writes x. */
int omega_gpu_decode_graph_begin_token(OmegaGpuDecodeGraph *g, const float *x_embed);
int omega_gpu_decode_graph_record_layer(OmegaGpuDecodeGraph *g, uint32_t layer, uint32_t pos);
int omega_gpu_decode_graph_record_final(OmegaGpuDecodeGraph *g);
/* Submits, checks every launch semaphore, the attention split counters and the token output (poisoned before the
 * submit), reads the token (may be NULL), releases the lock. Always releases the lock, also after an earlier failed
 * record (then returns that failure without submitting). */
int omega_gpu_decode_graph_end_token(OmegaGpuDecodeGraph *g, uint32_t *token, OmegaGpuDecodeGraphStats *st);

/* Glue records (between begin and end): RES (omega_blackwell_codegen_dec.h modes ADD / COMBINE / NORM) and SWIGLU. */
int omega_gpu_decode_graph_record_res(OmegaGpuDecodeGraph *g, uint32_t mode, uint64_t x_va, uint64_t a_va, uint64_t w_va,
                                      uint64_t nw_va, uint64_t y_va, float eps, float scale);
int omega_gpu_decode_graph_record_swiglu(OmegaGpuDecodeGraph *g, uint64_t gu_va, uint64_t act_va, float scale);

struct OmegaGpuCmdList *omega_gpu_decode_graph_list(OmegaGpuDecodeGraph *g);
void omega_gpu_decode_graph_vas(const OmegaGpuDecodeGraph *g, OmegaGpuDecodeGraphVas *v);
/* Takes the lock; copies the device x, hidden and logits (vocab values) of the last token; any pointer may be NULL. */
int omega_gpu_decode_graph_read(OmegaGpuDecodeGraph *g, float *x, float *hidden, float *logits);

/* ---- timeline (measurement) ---- */
/* One kind per recorded launch, in list order. */
enum {
    OMEGA_GPU_DGRAPH_K_NORM = 0, OMEGA_GPU_DGRAPH_K_QKV, OMEGA_GPU_DGRAPH_K_ATTN, OMEGA_GPU_DGRAPH_K_O, OMEGA_GPU_DGRAPH_K_ADD,
    OMEGA_GPU_DGRAPH_K_ROUTER, OMEGA_GPU_DGRAPH_K_GATE_UP, OMEGA_GPU_DGRAPH_K_SWIGLU, OMEGA_GPU_DGRAPH_K_DOWN, OMEGA_GPU_DGRAPH_K_COMBINE,
    OMEGA_GPU_DGRAPH_K_LM_HEAD, OMEGA_GPU_DGRAPH_K_ARGMAX, OMEGA_GPU_DGRAPH_K_COUNT
};
/* 1: every token also records a GPU timestamp (ns) before the first launch and after each launch (after its WAIT_FOR_IDLE).
 * Call between tokens. Adds one report per launch to the list. 0 = off (default). */
void omega_gpu_decode_graph_set_stamps(OmegaGpuDecodeGraph *g, int on);
/* Barrier between the launches of a token: 0 WFI + invalidate all (default), 2 WFI only, 3 WFI + invalidate data, 4 NVK-style
 * flush_data (implicit WFI). Process-wide, read once (env AIEN_OMEGA_DGRAPH_BARRIER when never set). 1 (none) is refused. */
void omega_gpu_decode_graph_set_barrier(int b);
int omega_gpu_decode_graph_barrier(void);
/* The last token with stamps on: stamps[0 .. n] (n + 1 values), kinds[0 .. n). Returns n (launches) or BAD_ARGS
 * (stamps off, the token failed, buffers smaller than launches + 1). Launch i took stamps[i + 1] - stamps[i] ns. */
int omega_gpu_decode_graph_stamps(const OmegaGpuDecodeGraph *g, uint64_t *stamps, uint8_t *kinds, uint32_t cap);
const char *omega_gpu_decode_graph_kind_name(uint8_t kind);

/* ---- TEST ONLY ---- */
/* 1: submit after every launch (same kernels and arguments, one submission each: the batching comparison). */
void omega_gpu_decode_graph_set_flush_each(OmegaGpuDecodeGraph *g, int on);
/* 1 SWIGLU mutant (up half read at +767); 2 COMBINE skips expert 7; 3 wrong wiring: attention written at scale 1
 * instead of 2^64 (the o_proj then sees x' 2^64 too small). 0 none. */
void omega_gpu_decode_graph_test_set_mutant(OmegaGpuDecodeGraph *g, int mutant);
/* TEST ONLY: negative control for the barrier study: lets barrier 1 (none) through, which set_barrier refuses. The token is a
 * dependency chain, so the comparison against the per-launch path (B) must fail. */
void omega_gpu_decode_graph_test_set_barrier_none(void);
/* Weight prefetch beside the attention (W-L2d): 1 = each layer records an independent o_proj GEMV on scratch buffers right after
 * the attention launch (no barrier between them), pulling the o_proj weights into L2 while the attention runs; useful only
 * with GPU-cached weights (AIEN_OMEGA_FP8_WEIGHT_CACHE=1). Process-wide; env AIEN_OMEGA_DGRAPH_PREFETCH ("1") when never set.
 * Outputs unchanged by construction (the prefetch writes only its own scratch y). Default 0. */
void omega_gpu_decode_graph_set_prefetch(int on);
int omega_gpu_decode_graph_prefetch(void);
/* TEST ONLY red for the prefetch run: the real o_proj is also marked independent (unordered after the attention): must FAIL. */
void omega_gpu_decode_graph_test_set_prefetch_red(int on);

#ifdef __cplusplus
}
#endif
#endif
