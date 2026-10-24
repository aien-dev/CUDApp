#ifndef OMEGA_BLACKWELL_CODEGEN_PATTN256_H
#define OMEGA_BLACKWELL_CODEGEN_PATTN256_H
/*
 * omega_blackwell_codegen_pattn256: causal prefill attention for Qwen3.8-27B on tensor cores (omega-mv#545,
 * 2026-10-18): head_dim 256, 24 query heads over 4 KV heads (6 per KV head), the decode attention's cache
 * (omega_gpu_decode_attn256_api.h). A derivative of omega_blackwell_codegen_pattn.h (the 30B's head_dim 128 kernel,
 * untouched). Two launches per call (omega_gpu_prefill_attn256_api.h):
 *
 *   prep    the decode kernel's prologue in prep mode (OmegaDecodeAttn256Spec.prep): per token, q' = RoPE(RMSNorm(q)
 *           * q_norm) * qk_scale as bf16 into the QF buffer below, and k_new / v_new into the cache rows with the
 *           decode step's own instruction sequence (so the rows a prompt leaves are the rows m decode steps leave).
 *   attend  this kernel: grid (ceil(m / 8), 4), 8 warps. CTA (cx, cy) serves KV head cy and the chunk tokens
 *           cx*8 .. cx*8+7. Row tile = 16 rows: row g (0..7) = token 2*tile, head cy*6 + min(g, 5); row g + 8 = the
 *           next token, same head (rows 6, 7 repeat head 5 and are never stored: a quarter of the tensor-core work of
 *           this first cut is spent on them). Warp w: tile w >> 1, output dims 128*(w & 1) .. +127. Both warps of a
 *           tile compute its scores (16 k-steps; Q's 16 A fragments = 64 registers, loaded once).
 *           Main loop over key blocks of 16 (kb = 0, 16, .. <= kend = pos0 + nk - 1, nk = min(cx*8 + 8, m)): stage
 *           K (fragment-major, B of S = Q K^T) and V (transposed pairs, B of O = P V) of the block in shared memory
 *           (8 KiB each), S = Q K^T (HMMA.16816.F32.BF16, four accumulators), causal mask (key > row position:
 *           s += -1e30, skipped when the whole block is at or below the tile's first token), online softmax in log2
 *           units (EX2), O rescaled by alpha = EX2(m_old - m_new), P packed to bf16 (F2FP), O += P V.
 *           Epilogue: l summed over the quad, out = O * RCP(l) * out_scale for rows g < 6 and tokens < m.
 *
 * Keys past kend in the last block are read from the cache (rows the prompt has not reached) and masked; the cache
 * is zero at allocation and only ever holds finite K/V rows, so 0 * row stays 0 in P V.
 * Registers: the program opts into BW_PHYS_GPR_HW_LIMIT (omega_bw_ir_set_gpr_limit); 256 threads x the declared
 * count must stay <= 65536 (the launcher checks). Shared memory: 16 KiB, the QMD cap.
 * The host simulator (tests/bw_warp_sim_pa.h) runs both launches; a simulator pass is never a chip result.
 */
#include "omega_blackwell_codegen.h"
#include <stdint.h>

#define OMEGA_PATTN256_WARPS 8u
#define OMEGA_PATTN256_THREADS 256u
#define OMEGA_PATTN256_TOKENS 8u            /* tokens per CTA */
#define OMEGA_PATTN256_KEYS 16u             /* keys per main-loop block */
#define OMEGA_PATTN256_SHARED_BYTES 16384u  /* K block 8 KiB + V block 8 KiB (the QMD cap) */
#define OMEGA_PATTN256_OUT_WORDS 6144u      /* f32 per token row of the output (24 heads x 256) */

/* QF: q' bf16 in A-fragment order. For token pair tp = tok >> 1, KV head c, head g of the group (0..5), k-step s
 * (dims 16s .. 16s+15) and quad lane t (0..3), 16 bytes at QF_OFF(tp, c, g, s, t) hold the four A registers
 * {a0, a1, a2, a3} = {(tok 2tp, dims 16s+2t, +1), (tok 2tp+1, same), (tok 2tp, dims 16s+8+2t, +1), (tok 2tp+1, same)},
 * each register = bf16 of the lower dim in the low half. */
#define OMEGA_PATTN256_QF_PAIR_BYTES 24576u /* one token pair, all 4 KV heads */
#define OMEGA_PATTN256_QF_OFF(tp, c, g, s, t) OMEGA_PATTN256_QF_OFF_SHAPE(4u, 6u, tp, c, g, s, t)
/* the same layout at kvh KV heads of grp query heads each */
#define OMEGA_PATTN256_QF_OFF_SHAPE(kvh, grp, tp, c, g, s, t) \
    ((((((uint64_t)(tp) * (kvh) + (c)) * (grp) + (g)) * 16u + (s)) * 4u + (t)) * 16u)

/* constant bank 0 at 0x380 */
#define OMEGA_PATTN256_ARG_QF 0x380u        /* bf16 QF, ceil(m / 2) token pairs */
#define OMEGA_PATTN256_ARG_KC 0x388u        /* bf16 K plane base of the layer [kv_head][max_pos][256] */
#define OMEGA_PATTN256_ARG_VC 0x390u        /* bf16 V plane base */
#define OMEGA_PATTN256_ARG_OUT 0x398u       /* f32 [m][6144] (token-major, head h at h * 256, model dim order) */
#define OMEGA_PATTN256_ARG_POS0 0x3a0u      /* u32 position of chunk token 0 */
#define OMEGA_PATTN256_ARG_MC 0x3a4u        /* u32 tokens, >= 1 */
#define OMEGA_PATTN256_ARG_KV_STRIDE 0x3a8u /* u32 bytes per KV head plane (max_pos * 512) */
#define OMEGA_PATTN256_ARG_OUT_SCALE 0x3acu /* f32 */
#define OMEGA_PATTN256_ARGS_WORDS 12u

typedef struct {
    int mutant;      /* TEST ONLY. 0 none; 1 no causal mask; 2 O is never rescaled by alpha. Each must be caught. */
    uint32_t q_heads, kv_heads; /* 0 = the 27B (24, 4). Grid y = kv_heads; q_heads / kv_heads (1..8) heads per CTA,
                                * the tile rows past the group repeat its last head and are never stored (Q36: 16, 2). */
} OmegaPrefillAttn256Spec;

/* Shape of one spec: the QF layout is QF_OFF with c < kv and g < group, a token pair = q * 1024 bytes. */
typedef struct { uint32_t qh, kvh, group, out_words, qf_pair_bytes; } OmegaPattn256Shape;
int omega_pattn256_shape(const OmegaPrefillAttn256Spec *spec, OmegaPattn256Shape *s);

int omega_pattn256_spec_ok(const OmegaPrefillAttn256Spec *spec);
/* Builds the register-allocated IR program. 0 ok, -1 refused or over BW_MAX_IR_INSNS / BW_MAX_VREGS / the registers. */
int omega_blackwell_codegen_prefill_attn256_prog(const OmegaPrefillAttn256Spec *spec, BlackwellIRProgram *prog);
/* Instruction, peak-register and placeholder counts (host, no device). */
int omega_prefill_attn256_prog_stats(const OmegaPrefillAttn256Spec *spec, uint32_t *insns, uint32_t *gprs, uint32_t *placeholders);

#endif
