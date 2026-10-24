/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OMEGA_BLACKWELL_CODEGEN_ROUTER_H
#define OMEGA_BLACKWELL_CODEGEN_ROUTER_H
/*
 * omega_blackwell_codegen_router: the MoE router of Qwen3-30B-A3B in ONE launch per layer (OWN-30B cut 8 item 2,
 * 2026-10-13). RMSNorm of the residual row, router logits (128 experts x 2048, bf16 weights), softmax, top-8,
 * renormalisation per config.json `norm_topk_prob` (true for this checkpoint: the 8 weights are the softmax of
 * the 8 selected logits; with the flag off they are the full 128-way softmax probabilities of the selected
 * experts), and the outputs the next launch needs without a host round trip:
 *   - x'[2048] f32 = RMSNorm(x) * out_scale (2^64 makes it the x' row of the grouped FP8 gate+up GEMV);
 *   - the grouped GEMV's group tables: entry g = directory[id_g] for the gate+up and the down tensors, each
 *     directory a device table of 128 x {w_va u64, s_va u64} built once per layer from the resident FP8 tensors;
 *   - ids[8] u32 and weights[8] f32 (descending by logit) for the host combine.
 *
 * Grid 1 x 1, `warps` warps (8, 16 or 32). ROWS form (spec.xstride and spec.lbstride set; L312 stroke lever 2, the router
 * once per layer for every conversation): grid 1 x rows, CTA cy reads the residual row at X + cy * xstride bytes and writes
 * x', ids, weights and both tables at their pointers + cy * lbstride bytes (the directories are shared); every row is
 * computed exactly as the plain kernel computes it alone. Per CTA:
 *   phase 1  thread t: E = 64 / warps consecutive x values and norm weights; sum of squares reduced per warp
 *            (SHFL.DOWN) and across warps through shared memory; r = RSQ(ss / 2048 + eps); xn = x * r * w staged
 *            in shared memory (f32 [2048]) and stored * out_scale to x'.
 *   phase 2  warp w computes the logits of experts w, w + warps, ...: lane L streams bytes [L*16, +16) of each
 *            256-column step of the expert's bf16 row (8 steps, `depth` of them in flight), widens the bf16 pairs
 *            by shift / mask, FFMAs against xn from shared memory (LDS.128), warp-reduces, lane 0 stores the
 *            logit to shared memory.
 *   phase 3  every warp redundantly (uniform barriers): lane L holds logits 4L..4L+3; 8 rounds of argmax: warp max
 *            by FMNMX.MAX, then the lowest index among the lanes holding that value (an integer "not equal" flag
 *            lifts the index candidates of the others above every real index, FMNMX.MIN picks the winner), the
 *            winner's logit is masked to -1e30. Selected logits and ids go to shared memory. Then the full
 *            128-way softmax sum (for norm_topk_prob = false) and the weights e_r = EX2((l_r - l_0) * log2 e) /
 *            den, den = sum of the 8 (flag on) or of all 128 (flag off).
 *   epilogue warp 0: lane 0 stores the 8 weights; lanes 0..7 store id_g and copy the two directory entries of
 *            id_g into the two group tables (4 x STG.E each).
 *
 * Only ops the attention and FP8 kernels already run on the chip are used (no FSETP / FSEL; the simulator does
 * not model them either). The host simulator (tests/bw_warp_sim_k.h) runs the 8-warp kernel.
 */
#include "omega_blackwell_codegen.h"
#include <stdint.h>

#define OMEGA_ROUTER_HIDDEN 2048u
#define OMEGA_ROUTER_EXPERTS 128u
#define OMEGA_ROUTER_TOPK 8u
#define OMEGA_ROUTER_STEPS 8u                 /* 256 bf16 columns per step */
#define OMEGA_ROUTER_ROW_BYTES (OMEGA_ROUTER_HIDDEN * 2u)
#define OMEGA_ROUTER_DIR_BYTES (OMEGA_ROUTER_EXPERTS * 16u)
#define OMEGA_ROUTER_TAB_BYTES (OMEGA_ROUTER_TOPK * 16u)

/* constant bank: addresses at 0x380 + 8n, then the scalars */
#define OMEGA_ROUTER_ARG_X 0x380u       /* f32 [2048] residual row */
#define OMEGA_ROUTER_ARG_GATE 0x388u    /* bf16 [128][2048] router weight */
#define OMEGA_ROUTER_ARG_NORM 0x390u    /* f32 [2048] post_attention_layernorm weight */
#define OMEGA_ROUTER_ARG_XOUT 0x398u    /* f32 [2048] x' = RMSNorm(x) * out_scale */
#define OMEGA_ROUTER_ARG_IDS 0x3a0u     /* u32 [8] */
#define OMEGA_ROUTER_ARG_WTS 0x3a8u     /* f32 [8] */
#define OMEGA_ROUTER_ARG_DIR_GU 0x3b0u  /* u64 [128][2] gate+up directory {w_va, s_va} */
#define OMEGA_ROUTER_ARG_DIR_DN 0x3b8u  /* u64 [128][2] down directory */
#define OMEGA_ROUTER_ARG_TAB_GU 0x3c0u  /* u64 [8][2] group table written for the gate+up launch */
#define OMEGA_ROUTER_ARG_TAB_DN 0x3c8u  /* u64 [8][2] group table written for the down launch */
#define OMEGA_ROUTER_ARG_EPS 0x3d0u     /* f32 rms eps */
#define OMEGA_ROUTER_ARG_OUT_SCALE 0x3d4u /* f32 */
#define OMEGA_ROUTER_ARG_NORM_TOPK 0x3d8u /* u32 0 or 1 (config.json norm_topk_prob) */
#define OMEGA_ROUTER_ARGS_WORDS 23u

/* shared memory map (bytes) */
#define OMEGA_ROUTER_SM_XN 0u           /* xn[2048] f32: 8192 */
#define OMEGA_ROUTER_SM_LOGIT 8192u     /* logits[128] f32: 512 */
#define OMEGA_ROUTER_SM_RED 8704u       /* per-warp partial sums [32]: 128 */
#define OMEGA_ROUTER_SM_BC 8832u        /* broadcast words: 16 */
#define OMEGA_ROUTER_SM_SEL 8848u       /* selected ids u32 [8]: 32 */
#define OMEGA_ROUTER_SM_SELV 8880u      /* selected logits f32 [8]: 32 */
#define OMEGA_ROUTER_SHARED_BYTES 8912u

typedef struct {
    uint32_t warps;  /* 8, 16 or 32 */
    uint32_t depth;  /* weight steps in flight per warp: 1, 2, 4 or 8 */
    int mutant;      /* TEST ONLY. 0 none; 1 RMSNorm without the weight; 2 the mask after each argmax round removes
                      * expert id ^ 1 (the winner can be picked again); 3 the directory entries copied are those of
                      * id ^ 1; 4 the weights are not normalised (den = 1); 5 (shared only) the shared expert's
                      * weight is 1 instead of sigmoid(gate); 6 (ROWS form only) every CTA reads row 0's x (rows 1.. wrong).
                      * Mutant 6 must be caught by the rows parity test. */
    uint32_t experts; /* routed experts: 0 or 128 (the 30B), 256 (Qwen3.6-35B-A3B) */
    uint32_t shared;  /* 0, or 1: the weight has one more row (the shared expert gate, row `experts`) that never
                       * enters the top-8 or the softmax; slot 8 of ids / weights / both tables is the shared expert:
                       * id = experts, weight = sigmoid(row . xn), directory[experts] (the directory has experts + 1
                       * entries). */
    uint32_t xstride;  /* 0: one row (grid_y 1). Else the ROWS form (with lbstride): bytes between consecutive x rows
                        * (>= 8192, a multiple of 16); CTA cy reads X + cy * xstride. */
    uint32_t lbstride; /* ROWS form: bytes between consecutive output buffers (a multiple of 16): CTA cy writes
                        * XOUT, IDS, WTS, TAB_GU and TAB_DN at + cy * lbstride. Both or neither of the strides are set. */
} OmegaRouterSpec;

/* Shared memory map of a spec (bytes); the OMEGA_ROUTER_SM_* constants are the 128 / no-shared case. */
typedef struct { uint32_t logit, red, bc, sel, selv, total; } OmegaRouterSmem;
int omega_router_smem(const OmegaRouterSpec *spec, OmegaRouterSmem *m); /* 0 ok, -1 refused shape */

/* Builds the register-allocated IR program into *prog. 0 ok, -1 refused. */
int omega_blackwell_codegen_router_prog(const OmegaRouterSpec *spec, BlackwellIRProgram *prog);
/* Instruction and peak-register counts (host, no device). */
int omega_router_prog_stats(const OmegaRouterSpec *spec, uint32_t *insns, uint32_t *gprs);

#endif
