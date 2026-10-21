#ifndef OMEGA_GPU_ROUTER_API_H
#define OMEGA_GPU_ROUTER_API_H
/*
 * omega_gpu_router_api: one-launch MoE router for Qwen3-30B-A3B (OWN-30B cut 8 item 2, 2026-10-13).
 * C signatures for the Rust side first; the kernel is described in omega_blackwell_codegen_router.h.
 *
 *   int  omega_gpu_router_alloc(uint32_t layers, OmegaGpuRouter **out);
 *        Device-resident per-layer router weights (bf16 [128][2048]), post-attention norm weights (f32 [2048]) and
 *        the two expert directories (u64 [128][2] {w_va, s_va} for the gate+up and the down FP8 tensors), plus the
 *        launch buffers: x in, x' out, ids, weights and the two group tables the grouped FP8 GEMV reads.
 *   int  omega_gpu_router_set_layer(OmegaGpuRouter *r, uint32_t layer, const uint16_t *gate_bf16, const float *norm_w);
 *        gate_bf16: the checkpoint's mlp.gate.weight [128][2048] as bf16 bits; norm_w: post_attention_layernorm.weight
 *        widened to f32 by the caller.
 *   int  omega_gpu_router_set_directory_fp8(OmegaGpuRouter *r, uint32_t layer, const OmegaGpuFp8Tensor *gate_up,
 *                                           const OmegaGpuFp8Tensor *down);
 *        Builds the layer's directories from two resident FP8 tensors with 128 experts (omega_gpu_fp8_tensor_entry).
 *   int  omega_gpu_router_set_directory(OmegaGpuRouter *r, uint32_t layer, const uint64_t *gu, const uint64_t *dn);
 *        Raw form of the above (tests): gu / dn are [128][2] {w_va, s_va}.
 *   int  omega_gpu_router(OmegaGpuRouter *r, uint32_t layer, const float *x, float eps, int norm_topk, float out_scale,
 *                         uint32_t *ids, float *weights, OmegaGpuRouterInfo *info);
 *        x: f32 [2048] residual row (host). One launch: RMSNorm(x) with the layer's norm weight and eps, logits,
 *        softmax, top-8, weights renormalised when norm_topk != 0 (config.json norm_topk_prob). Outputs: ids[8]
 *        (descending by logit) and weights[8] to the host; on the device x' = RMSNorm(x) * out_scale (2^64 gives the
 *        x' row of omega_gpu_fp8_gemv_grouped_dev) and the two group tables, entry g = directory[ids[g]], at the
 *        addresses omega_gpu_router_device_outputs returns. The grouped GEMV consumes them with no host round trip:
 *        omega_gpu_fp8_gemv_grouped_dev(gate_up, 8, tab_gu_va, xout_va, y, info).
 *   void omega_gpu_router_device_outputs(const OmegaGpuRouter *r, uint64_t *xout_va, uint64_t *tab_gu_va, uint64_t *tab_dn_va);
 *   int  omega_gpu_router_read_back(OmegaGpuRouter *r, float *xout, uint64_t *tab_gu, uint64_t *tab_dn);
 *        Test helper: syncs the device outputs of the last call to the host (any pointer may be NULL).
 *
 * Every buffer is GPU-uncached host-visible session memory (nvos.h:1117 rule), launched through
 * omega_gpu_session_launch like the FP8 GEMV and the decode attention. Output words are poisoned before the launch
 * and a surviving poison word is CHIP_FAIL.
 */
#include "omega_blackwell_codegen_router.h"
#include "omega_gpu_fp8_api.h"
#include <stdint.h>
#include <stddef.h>

enum {
    OMEGA_GPU_ROUTER_OK = 0,
    OMEGA_GPU_ROUTER_BAD_ARGS = -1,
    OMEGA_GPU_ROUTER_TOO_LARGE = -2,
    OMEGA_GPU_ROUTER_CODEGEN_FAIL = -3,
    OMEGA_GPU_ROUTER_CHIP_FAIL = -4
};

#define OMEGA_GPU_ROUTER_MAX_LAYERS 64u
#define OMEGA_GPU_ROUTER_WARPS 32u   /* default kernel shape (omega_gpu_router_set_shape overrides) */
#define OMEGA_GPU_ROUTER_DEPTH 4u

typedef struct OmegaGpuRouter OmegaGpuRouter;

typedef struct {
    uint32_t layer, warps, depth, threads;
    uint64_t weight_bytes;      /* bf16 router bytes the launch streams: 128 * 2048 * 2 */
    uint64_t elapsed_ns;        /* marker-to-marker chip time */
    uint64_t call_ns;           /* whole call */
    uint32_t completion_marker;
    int kernel_cache_hit;
} OmegaGpuRouterInfo;

int omega_gpu_router_alloc(uint32_t layers, OmegaGpuRouter **out);
/* Qwen3.6-35B-A3B shape (sc L306): experts 256 (0 = 128) and shared 1. Then set_layer's gate_bf16 is [experts + 1][2048]
 * (the last row = shared_expert_gate), the directories have experts + 1 entries (entry `experts` = the shared expert),
 * and ids / weights / both tables have 9 slots: slot 8 = {experts, sigmoid(shared gate . xn), directory[experts]}. */
int omega_gpu_router_alloc_shape(uint32_t layers, uint32_t experts, uint32_t shared, OmegaGpuRouter **out);
void omega_gpu_router_free(OmegaGpuRouter *r);
int omega_gpu_router_set_layer(OmegaGpuRouter *r, uint32_t layer, const uint16_t *gate_bf16, const float *norm_w);
int omega_gpu_router_set_directory(OmegaGpuRouter *r, uint32_t layer, const uint64_t *gu, const uint64_t *dn);
int omega_gpu_router_set_directory_fp8(OmegaGpuRouter *r, uint32_t layer, const OmegaGpuFp8Tensor *gate_up, const OmegaGpuFp8Tensor *down);
int omega_gpu_router(OmegaGpuRouter *r, uint32_t layer, const float *x, float eps, int norm_topk, float out_scale,
                     uint32_t *ids, float *weights, OmegaGpuRouterInfo *info);
void omega_gpu_router_device_outputs(const OmegaGpuRouter *r, uint64_t *xout_va, uint64_t *tab_gu_va, uint64_t *tab_dn_va);
/* Record path (OWN-30B W-L2a, decode graph; omega_gpu_cmdlist.h): the launch of omega_gpu_router with the residual row
 * read from x_va (f32 [2048] on the device) appended to cl instead of submitted. The caller holds
 * omega_gpu_session_lock(). Nothing is staged, poisoned or read back. The outputs land, when the list runs, at the
 * router's launch buffer (one per router: a later record of another layer overwrites them, in list order):
 * xout = RMSNorm(x) * out_scale [2048] f32, ids [8] u32, wts [8] f32 (the COMBINE weights), and the two group tables
 * (8 x {w_va, s_va}) for gate+up and down. A kernel cache miss that would evict the resident kernel is refused. */
typedef struct { uint64_t xout_va, ids_va, wts_va, tab_gu_va, tab_dn_va; } OmegaGpuRouterDevOut;
struct OmegaGpuCmdList;
int omega_gpu_router_record(struct OmegaGpuCmdList *cl, OmegaGpuRouter *r, uint32_t layer, uint64_t x_va, float eps, int norm_topk,
                            float out_scale, OmegaGpuRouterDevOut *out);
/* The stroke (L312 v2): the same launch with every output in the caller's device buffer at lb_va (256-byte aligned,
 * omega_gpu_router_layout_bytes(r) bytes, the launch buffer's layout: xout, ids, wts, tab_gu, tab_dn at the offsets
 * out reports). One buffer per conversation, so N routers of one layer can sit in one list. */
int omega_gpu_router_record_to(struct OmegaGpuCmdList *cl, OmegaGpuRouter *r, uint32_t layer, uint64_t x_va, float eps, int norm_topk,
                               float out_scale, uint64_t lb_va, OmegaGpuRouterDevOut *out);
uint32_t omega_gpu_router_layout_bytes(const OmegaGpuRouter *r);
int omega_gpu_router_read_back(OmegaGpuRouter *r, float *xout, uint64_t *tab_gu, uint64_t *tab_dn);
const char *omega_gpu_router_last_error(void);
const char *omega_gpu_router_rc_name(int rc);

/* ---- measurement knobs (tests) ---- */
void omega_gpu_router_set_shape(uint32_t warps, uint32_t depth); /* 0 = default; clears the kernel cache */
void omega_gpu_router_test_set_mutant(int mutant);              /* TEST ONLY: OmegaRouterSpec.mutant */

#endif
