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

#ifndef OMEGA_BLACKWELL_CODEGEN_PATTN_H
#define OMEGA_BLACKWELL_CODEGEN_PATTN_H
/*
 * omega_blackwell_codegen_pattn: causal prefill attention for Qwen3-30B-A3B on tensor cores (OWN-30B L1 attention,
 * 2026-10-13). One launch per chunk of up to 64 consecutive tokens of one layer. New file; uses the opcodes of
 * omega_blackwell_codegen.h plus the SHFL.BFLY and F2FP.BF16.F32.PACK_AB placeholders of omega_blackwell_codegen_tc.h
 * (patched after encoding with omega_tc_patch_code).
 *
 * Grid (ceil(mc / W), 4): CTA (cx, cy) serves KV head cy and the chunk tokens cx*W .. cx*W+W-1 (W = warps, 4 or 8).
 * Row tile = 16 query rows = 2 tokens x the 8 query heads of the group: row g (0..7) = token 2*tile, head cy*8+g;
 * row g+8 = token 2*tile+1, same head. Warp w: tile w/2, output dims 64*(w%2) .. +63 (two warps share a tile and
 * both compute its scores; each keeps half of O: 8 HMMA n-tiles = 32 f32 registers).
 *
 * Prologue (all warps, CTA barriers; uniform trip counts):
 *   K/V append: every CTA computes K = bf16(RoPE(RMSNorm(k) * k_norm)) and V = bf16(v) for the chunk tokens
 *     0 .. nk-1 (nk = min(cx*W + W, mc)) with the decode kernel's exact instruction sequence (quad per lane,
 *     SHFL.DOWN sum, lane 0's sum through shared memory, RSQ, partner half through shared memory) and stores them
 *     into the cache rows pos0 + t. CTAs write identical bits to the rows they share. MEMBAR.SYS + BAR.SYNC, then
 *     the CTA reads its own rows back from the cache.
 *   Q: q' = bf16(RoPE(RMSNorm(q) * q_norm) * qk_scale) per (token, head) item, staged in shared memory in HMMA
 *     A-fragment order (2 tiles per round), then each warp loads its 8 fragments (32 registers) for the whole run.
 * Main loop over key blocks of 32 (kb = 0, 32, .. <= kend = pos0 + nk - 1):
 *   stage K (fragment-major, B of S = Q K^T) and V (transposed pairs, B of O = P V) of the block in shared memory
 *   (16 KB), S = Q K^T (8 k-steps x 4 n-tiles, HMMA.16816.F32.BF16), causal mask (key > row position: s += -1e30,
 *   skipped when the whole block is at or below the tile's first token), online softmax in log2 units (EX2), O
 *   rescaled by alpha = EX2(m_old - m_new), P packed to bf16 (F2FP), O += P V (2 k-steps x 8 n-tiles).
 * Epilogue: l summed over the quad (SHFL.BFLY), out = O * RCP(l) * out_scale for tokens < mc.
 * Split modes (spec.mode, 2026-10-13 cut 3): an APPEND launch then an ATTEND launch over the same mc tokens do the
 * fused kernel's work with no redundant appends, so one pair of launches can cover a whole prompt (grid ceil(mc/W) x 4)
 * instead of 64-token chunks of 8 x 4 CTAs. The appended rows are the same instruction sequence (identical bits).
 *
 * The host simulator (tests/bw_warp_sim_pa.h) runs the kernel with the chip-exact HMMA model; EX2/RSQ/RCP are
 * exact there and approximate on the chip. A simulator pass is never a chip result.
 */
#include "omega_blackwell_codegen.h"
#include <stdint.h>

#define OMEGA_PATTN_CHUNK 64u              /* tokens per launch */
#define OMEGA_PATTN_KEYS 32u               /* keys per main-loop block */
#define OMEGA_PATTN_QKV_WORDS 5120u        /* f32 per token row of the fused q|k|v projection */
#define OMEGA_PATTN_OUT_WORDS 4096u        /* f32 per token row of the output (32 heads x 128) */
#define OMEGA_PATTN_SHARED_BYTES 16384u    /* the QMD cap (omega_blackwell_qmd.h) */

/* constant bank 0 at 0x380 */
#define OMEGA_PATTN_ARG_QKV 0x380u         /* f32 [64][5120]: row t = chunk token t (position pos0 + t) */
#define OMEGA_PATTN_ARG_KC 0x388u          /* bf16 K plane base [kv_head][max_pos][128] */
#define OMEGA_PATTN_ARG_VC 0x390u          /* bf16 V plane base */
#define OMEGA_PATTN_ARG_ROPE 0x398u        /* f32 [max_pos][cos 64 | sin 64] */
#define OMEGA_PATTN_ARG_QN 0x3a0u          /* f32 [128] q_norm */
#define OMEGA_PATTN_ARG_KN 0x3a8u          /* f32 [128] k_norm */
#define OMEGA_PATTN_ARG_OUT 0x3b0u         /* f32 [64][4096] */
#define OMEGA_PATTN_ARG_POS0 0x3b8u        /* u32 position of chunk token 0 */
#define OMEGA_PATTN_ARG_MC 0x3bcu          /* u32 tokens in this chunk, 1..64 */
#define OMEGA_PATTN_ARG_KV_STRIDE 0x3c0u   /* u32 bytes per KV head plane (max_pos * 256) */
#define OMEGA_PATTN_ARG_EPS 0x3c4u         /* f32 */
#define OMEGA_PATTN_ARG_OUT_SCALE 0x3c8u   /* f32 */
#define OMEGA_PATTN_ARG_QK_SCALE 0x3ccu    /* f32 (1/sqrt(128)) * log2(e) */
#define OMEGA_PATTN_ARG_KC2 0x3d0u         /* APPEND with spec.mirror: second K plane base (GPU-cached mirror) */
#define OMEGA_PATTN_ARG_VC2 0x3d8u         /* second V plane base */
#define OMEGA_PATTN_ARG_KV_STRIDE2 0x3e0u  /* u32 bytes per KV head plane of the mirror */
#define OMEGA_PATTN_ARGS_WORDS 25u

typedef struct {
    uint32_t warps;  /* 4 or 8 (tokens per CTA) */
    int mutant;      /* TEST ONLY. 0 none; 1 no RoPE on q and k; 2 no causal mask; 3 the new K/V are not written;
                      * 4 O is never rescaled by alpha. Each must be caught by the gates. */
    uint32_t mode;   /* OMEGA_PATTN_MODE_*: 0 fused (append + attend, chunks of <= 64 tokens);
                      * 1 append only: CTA (cx, cy) appends just its own tokens cx*W .. cx*W+W-1 (< mc), then exits;
                      * 2 attend only: no append (a mode-1 launch over the same tokens ran before), any mc >= 1. */
    uint32_t mirror; /* APPEND only: 1 also stores each new row into the mirror planes (ARG_KC2 / VC2 / KV_STRIDE2), e.g. a
                      * GPU-cached copy of the layer's K/V that the ATTEND launch then reads instead of the uncached cache */
} OmegaPrefillAttnSpec;
#define OMEGA_PATTN_MODE_FUSED 0u
#define OMEGA_PATTN_MODE_APPEND 1u
#define OMEGA_PATTN_MODE_ATTEND 2u

int omega_pattn_spec_ok(const OmegaPrefillAttnSpec *spec);
/* Builds the register-allocated IR program. 0 ok, -1 refused or over BW_MAX_IR_INSNS / BW_MAX_VREGS. */
int omega_blackwell_codegen_prefill_attn_prog(const OmegaPrefillAttnSpec *spec, BlackwellIRProgram *prog);
/* Instruction, peak-register and placeholder counts (host, no device). */
int omega_prefill_attn_prog_stats(const OmegaPrefillAttnSpec *spec, uint32_t *insns, uint32_t *gprs, uint32_t *placeholders);

#endif
