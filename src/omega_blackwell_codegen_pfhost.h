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

#ifndef OMEGA_BLACKWELL_CODEGEN_PFHOST_H
#define OMEGA_BLACKWELL_CODEGEN_PFHOST_H
/*
 * omega_blackwell_codegen_pfhost: the whole-prompt prefill's former host work on the chip (OWN-30B W-L1G-D, 2026-10-14). New file.
 * Both kernels reproduce a host loop of the Rust driver (sc qwen3_moe_omega_prefill.rs) BIT FOR BIT: every product is an FMUL
 * and every sum an FADD (never FFMA), in the host's summation order, so the results equal the host's f32 results exactly.
 *
 * ---- router logits (rl) ----
 *   logits[r][ex] = (((-0.0 + w[ex][0] * x[r][0]) + w[ex][1] * x[r][1]) + ...) + w[ex][k-1] * x[r][k-1]
 * i.e. Rust `router[ex].iter().zip(x[r]).map(|(a, b)| a * b).sum::<f32>()` (a sequential fold from -0.0).
 * Weights in the PACKED layout R4: wq[g][ex][q] = w[ex][4g + q] (g = 0 .. k/4 - 1), f32, then ONE extra group of e * 4 floats
 * (any value: the last prefetch reads it, the result never uses it). Total (k/4 + 1) * e * 16 bytes.
 * x: f32 rows of k (row stride k). Rows 0 .. ceil(rows / t) * t - 1 must be readable (the last CTA reads whole tiles).
 * logits: f32 rows of e; rows 0 .. ceil(rows / t) * t - 1 are written (rows >= rows hold garbage).
 * One CTA per t rows (ctaid.x), e threads: thread ex owns expert ex and t accumulators. Per 128-wide chunk of k the CTA stages
 * the t x 128 activation tile in shared memory (t * 512 bytes), then walks it four k at a time: one LDG.128 of wq (prefetched one
 * group ahead), one LDS.128 broadcast per row, 4 FMUL + 4 FADD per row.
 * Arguments (cbank 0x380): x u64, wq u64, logits u64.
 * mutant (tests only): 1 = FFMA (fused) instead of FMUL + FADD, 2 = the last 128-wide chunk of k is skipped.
 *
 * ---- MoE weighted combine (mc) ----
 *   out[r][c] = (((+0.0 + wt[r][0] * down[inv[r][0]][c]) + wt[r][1] * down[inv[r][1]][c]) + ...)   (slots 0 .. top_k - 1)
 * i.e. the driver's combine_rows accumulator (`acc = 0.0; acc += wts[gi] * d` in slot order); the host adds out to x.
 * down: f32 rows of h (the down GEMM's sorted rows); inv, wt: u32 / f32 [rows][top_k] (row stride top_k, 16-byte aligned rows);
 * out: f32 rows of h. One CTA per row, h / 4 threads, 4 columns each (LDG.128 per slot).
 * Arguments (cbank 0x380): down u64, inv u64, wt u64, out u64.
 * mutant (tests only): 1 = FFMA instead of FMUL + FADD, 2 = the last slot is skipped.
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct { uint32_t k, e, t; int mutant; } OmegaRlSpec;
#define OMEGA_RL_ARGS_WORDS 6u
int omega_rl_spec_ok(const OmegaRlSpec *s);
uint32_t omega_rl_shared_bytes(const OmegaRlSpec *s);
int omega_blackwell_codegen_rl_prog(const OmegaRlSpec *s, BlackwellIRProgram *prog);

typedef struct { uint32_t h, top_k; int mutant; } OmegaMcSpec;
#define OMEGA_MC_ARGS_WORDS 8u
int omega_mc_spec_ok(const OmegaMcSpec *s);
int omega_blackwell_codegen_mc_prog(const OmegaMcSpec *s, BlackwellIRProgram *prog);
#ifdef __cplusplus
}
#endif
#endif
