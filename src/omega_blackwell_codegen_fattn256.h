/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_BLACKWELL_CODEGEN_FATTN256_H
#define OMEGA_BLACKWELL_CODEGEN_FATTN256_H
/*
 * omega_blackwell_codegen_fattn256: flash-style causal prefill attention on tensor cores for the head_dim 256
 * full-attention layers, one warp per 16-row tile (Linear DRA-85, P-DRA-4 Lane P, 2026-11-01). Target shape:
 * Qwen3.6-35B-A3B (16 query heads over 2 KV heads, 8 per KV head: every tile row is a real head); the 27B shape
 * (24 / 4, 6 per KV head) also builds, with rows 6, 7 repeating head 5 and never stored, as in pattn256.
 *
 * The attend launch of omega_gpu_prefill_fattn256_api.h. It reads the same inputs as pattn256's attend
 * (omega_blackwell_codegen_pattn256.h): q' in the QF A-fragment layout written by the decode kernel's prep mode, and
 * the decode attention's K / V cache planes. What differs is the work split:
 *
 *   pattn256  grid (ceil(m / 8), kvh), 8 tokens per CTA, two warps per row tile (each owns 128 output dims and both
 *             compute the tile's full S), K and V staged through 16 KiB of shared memory with two CTA barriers per
 *             16-key block, every warp walking to the CTA's last key.
 *   fattn256  grid (ceil(m / 16), kvh), 16 tokens per CTA, ONE warp per row tile owning all 256 output dims (O = 32
 *             accumulator fragments = 128 registers, Q = 16 A fragments = 64 registers, both resident), no shared
 *             memory and no barriers: each lane loads its B fragments straight from the cache rows (LDG.128 / LDG.64
 *             through L1; the 8 warps of a CTA and the CTAs of a KV head read the same rows), and each warp stops at its
 *             own last key (causal work per warp, not per CTA). Warps whose first token is past m exit at once.
 *
 * Tile of warp w in CTA (cx, cy): rows g = 0..7 = token tokA = cx*16 + 2w, head cy*group + min(g, group-1); rows
 * g + 8 = token tokA + 1, same heads. Lane (g, t), t = lane & 3.
 *
 * S = Q K^T with a k-permutation: the 256 head dims are visited as 8 groups of 32 (u = 0..7); lane (g, t) loads 16
 * bytes = words 16u+4t .. 16u+4t+3 of key rows kb + g and kb + 8 + g (one LDG.128 each). k-step 2u takes words
 * 16u+4t (b0) and 16u+4t+1 (b1), k-step 2u+1 words +2 (b0) and +3 (b1). Q's A fragments are loaded once with the same
 * permutation from QF: k-step 2u register pair {a0, a1} (j even) or {a2, a3} (j odd) for word 16u+4t+j is the 8-byte
 * {tokA, tokA+1} pair at QF_OFF_SHAPE(tp, c, g, 2u + (t >> 1), j) + (t & 1) * 8 (one LDG.64 each, 32 per lane). The
 * dot product is a sum over dims, so the permutation (the same on both sides) does not change S.
 *
 * O = P V with a column permutation: n-tile i (0..31), fragment column n holds output dim 32n + i. For each half
 * step (h = 0..3, half q = 0, 1) lane (g, t) loads 8 bytes at byte 64g + 16h + 8q of value rows kb + 2t, kb + 2t + 1,
 * kb + 8 + 2t, kb + 9 + 2t (dims 32g + 8h + 4q .. +3) and packs the key pairs (low half = lower key) into the B
 * fragments of n-tiles 8h + 4q + {0, 1, 2, 3}. Accumulator element c0, c1 of n-tile i is (row g, dims 64t + i,
 * 64t + 32 + i); c2, c3 the same dims of row g + 8.
 *
 * Softmax, causal mask and P packing are pattn256's: scores already carry log2(e) / 16 (the prep's qk_scale), keys
 * kb + 8j + 2t + (k & 1) above the row position get -1e30 (the whole block is skipped when kb + 15 <= row A's
 * position), running max per row over the quad (SHFL.BFLY 1, 2), alpha = EX2(m_old - m_new), P to bf16 (F2FP),
 * l summed from the unrounded p. Epilogue: l over the quad, out = O * RCP(l) * out_scale.
 *
 * Keys past the warp's last key in its last block are read (rows the prompt has not reached, or the plane slack) and
 * masked, as in pattn256: the cache is zero at allocation and only ever holds finite K/V rows, so 0 * row stays 0.
 * The tile of an odd m's last pair reads tokA + 1's QF slot, which the prep never wrote: rows g + 8 of that tile may
 * be anything (NaN included); HMMA rows are independent and those rows are never stored.
 * Registers: the program opts into BW_PHYS_GPR_HW_LIMIT; 256 threads x the declared count must stay <= 65536.
 * The host simulator (tests/bw_warp_sim_pa.h) runs it; a simulator pass is never a chip result.
 */
#include "omega_blackwell_codegen.h"
#include <stdint.h>

#define OMEGA_FATTN256_WARPS 8u
#define OMEGA_FATTN256_THREADS 256u
#define OMEGA_FATTN256_TOKENS 16u           /* tokens per CTA: 8 warps x a token pair */
#define OMEGA_FATTN256_KEYS 16u             /* keys per main-loop block */
#define OMEGA_FATTN256_SHARED_BYTES 0u

/* constant bank 0 at 0x380 */
#define OMEGA_FATTN256_ARG_QF 0x380u        /* bf16 QF (OMEGA_PATTN256_QF_OFF_SHAPE layout), ceil(m / 2) token pairs */
#define OMEGA_FATTN256_ARG_KC 0x388u        /* bf16 K plane base of the layer [kv_head][max_pos][256] */
#define OMEGA_FATTN256_ARG_VC 0x390u        /* bf16 V plane base */
#define OMEGA_FATTN256_ARG_OUT 0x398u       /* f32 rows, head h at h * 256 (model dim order) */
#define OMEGA_FATTN256_ARG_POS0 0x3a0u      /* u32 position of chunk token 0 */
#define OMEGA_FATTN256_ARG_MC 0x3a4u        /* u32 tokens, >= 1 */
#define OMEGA_FATTN256_ARG_KV_STRIDE 0x3a8u /* u32 bytes per KV head plane (max_pos * 512) */
#define OMEGA_FATTN256_ARG_OUT_SCALE 0x3acu /* f32 */
#define OMEGA_FATTN256_ARG_OSTRIDE 0x3b0u   /* u32 bytes between output rows (>= q_heads * 1024) */
#define OMEGA_FATTN256_ARGS_WORDS 13u

typedef struct {
    int mutant;      /* TEST ONLY. 0 none; 1 no causal mask; 2 O is never rescaled by alpha. Each must be caught. */
    uint32_t q_heads, kv_heads; /* 0, 0 = Qwen3.6-35B-A3B (16, 2). q_heads / kv_heads must be 1..8 (grid y = kv_heads). */
} OmegaPrefillFattn256Spec;

typedef struct { uint32_t qh, kvh, group, out_words, qf_pair_bytes; } OmegaFattn256Shape;
int omega_fattn256_shape(const OmegaPrefillFattn256Spec *spec, OmegaFattn256Shape *s);
int omega_fattn256_spec_ok(const OmegaPrefillFattn256Spec *spec);
/* Builds the register-allocated IR program. 0 ok, -1 refused or over BW_MAX_IR_INSNS / BW_MAX_VREGS / the registers. */
int omega_blackwell_codegen_prefill_fattn256_prog(const OmegaPrefillFattn256Spec *spec, BlackwellIRProgram *prog);
/* Instruction, peak-register and placeholder counts (host, no device). */
int omega_prefill_fattn256_prog_stats(const OmegaPrefillFattn256Spec *spec, uint32_t *insns, uint32_t *gprs, uint32_t *placeholders);

#endif
