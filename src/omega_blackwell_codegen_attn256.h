#ifndef OMEGA_BLACKWELL_CODEGEN_ATTN256_H
#define OMEGA_BLACKWELL_CODEGEN_ATTN256_H
/*
 * omega_blackwell_codegen_attn256: decode attention for Qwen3.8-27B in ONE launch per layer (omega-mv#544,
 * 2026-10-18). A derivative of omega_blackwell_codegen_attn.h (head_dim 128, 8 q per kv) for the 27B's full-attention
 * layers: 24 query heads, 4 KV heads (6 q per kv), head_dim 256, per-head RMSNorm on q and k (the caller folds the
 * model's zero-centred weights to 1 + w), rotary embedding from a host-built cos/sin table, the new token's K/V
 * appended to the GPU-resident bf16 cache inside the kernel, softmax over the whole context, output scaled by
 * `out_scale` ready for the gate (sigmoid(gate) * out happens outside) and o_proj.
 *
 * Partial RoPE by permutation (design decision 1 of #544): the kernel rotates the standard half pairs (i, i + 128) of
 * its 256-wide head with table row [cos 128 | sin 128]; the host builds the table so only the first P pairs rotate
 * (inv_freq_j = theta^(-2j / rotary_dim), rotary_dim = 2P) and the rest are identity (cos 1, sin 0). The engine
 * permutes the q and k dims (and q_norm / k_norm) at load with omega_dattn256_perm (omega_gpu_decode_attn256_api.h)
 * so the model's rotate-half within its first rotary_dim dims lands on those P pairs. Scores are invariant to a
 * shared q/k permutation; V and the output are not permuted. The K cache holds permuted rows.
 *
 * Grid: (splits, kv_heads). CTA (cx, cy) handles KV head cy and the positions [cx*per_split, min(+per_split, pos+1));
 * the host picks splits = ceil((pos+1)/per_split), so no CTA is empty. 8 warps = 256 threads per CTA.
 *
 * Per CTA:
 *   prologue  one norm + rope block for all 8 warps (8 f32 per lane): warp h < 6 builds q'[h] = RoPE(table[pos],
 *             RMSNorm(q[cy*6+h], q_norm, eps)) * qk_scale, staged in shared memory as q'[h][256]; warp 6 builds
 *             k_new = RoPE(RMSNorm(k[cy], k_norm)) the same way (k_norm lives 1024 bytes after q_norm); warp 7 runs
 *             the block on v[cy] (result discarded) and reloads the raw v. In the CTA whose range holds `pos` (the
 *             last split) warps 6 and 7 write k_new and v_new as bf16 into the cache at [cy][pos], then MEMBAR +
 *             BAR.SYNC, so its own stream reads them back.
 *   chunks    of 256 positions: score pass (thread = position: the 512-byte bf16 K row streamed in 32 quads with a
 *             `depth`-deep prefetch, 6 dot products against q' from shared memory, scores to shared s[t][8], 6 used);
 *             V pass (warp pair: warp w handles positions (w>>1) + 4i of the chunk and the dims 128*(w&1) .. +127;
 *             lane = 4 dims, LDG.64 per position; online softmax per warp: running max, alpha = EX2(m_old - m_new)
 *             applied only when the max moved (warp-uniform branch), p = EX2(s - m), acc += p * v, l += p. The two
 *             warps of a pair see the same positions, so their (m, l) are bit-identical).
 *   epilogue  warps combine (m over all 8, l over the even warps, acc over the 4 warps that hold each dim half)
 *             through shared memory and write the CTA partial {m[6], l[6], acc[6][256]} to partials[cy][cx]; then
 *             one ATOMG.ADD on counters[cy]; the last CTA of the KV head combines the partials with the usual max
 *             rescale and writes out[(cy*6+h)*256+d] * out_scale, and resets the counter.
 *
 * Numbers (f32 on the chip, double in the host reference, tolerances in tests/gpu_decode_attn256_test.c):
 *   scores in log2 units: q' already carries (1/sqrt(256)) * log2(e), so p = EX2(s - m) = exp((q.k)/sqrt(d) - M).
 *   bf16 K/V are widened by a 16-bit shift; the new K/V are rounded to bf16 round-to-nearest-even before storing.
 *
 * Shared memory stays under the 16 KiB QMD cap (14896 bytes), registers under the chip allocator's 126.
 * The host simulator (tests/bw_warp_sim_k.h) runs this kernel at 8 warps; a simulator pass is never a chip result.
 */
#include "omega_blackwell_codegen.h"
#include <stdint.h>

#define OMEGA_DATTN256_HEAD_DIM 256u
#define OMEGA_DATTN256_Q_PER_KV 6u
#define OMEGA_DATTN256_KV_HEADS 4u
#define OMEGA_DATTN256_Q_HEADS 24u
#define OMEGA_DATTN256_WARPS 8u
#define OMEGA_DATTN256_CHUNK 256u                /* positions per score pass = threads per CTA */
#define OMEGA_DATTN256_ROW_BYTES 512u            /* one bf16 K or V row */
#define OMEGA_DATTN256_ROPE_ROW_BYTES 1024u      /* cos[128] then sin[128], f32 */
#define OMEGA_DATTN256_PARTIAL_WORDS 1548u       /* m[6], l[6], acc[6][256] */
#define OMEGA_DATTN256_MAX_DEPTH 4u
/* bytes past the last row of a K or V plane the prefetch may touch (clamped reads of the last rows) */
#define OMEGA_DATTN256_PLANE_SLACK (OMEGA_DATTN256_MAX_DEPTH * 4u * OMEGA_DATTN256_ROW_BYTES + 64u)
#define OMEGA_DATTN256_QKV_WORDS (OMEGA_DATTN256_Q_HEADS * 256u + 2u * OMEGA_DATTN256_KV_HEADS * 256u) /* 8192 */
#define OMEGA_DATTN256_OUT_WORDS (OMEGA_DATTN256_Q_HEADS * 256u)                                       /* 6144 */

/* constant bank: every address at 0x380 + 8n, then the scalars */
#define OMEGA_DATTN256_ARG_QKV 0x380u      /* f32 [24*256 q | 4*256 k | 4*256 v] (the gate is NOT part of it) */
#define OMEGA_DATTN256_ARG_KC 0x388u       /* bf16 K cache base: [kv_head][max_pos][256], kv_stride bytes per head */
#define OMEGA_DATTN256_ARG_VC 0x390u       /* bf16 V cache base, same layout */
#define OMEGA_DATTN256_ARG_ROPE 0x398u     /* f32 [max_pos][cos 128 | sin 128] */
#define OMEGA_DATTN256_ARG_QN 0x3a0u       /* f32 [256] q_norm weight (1 + w, permuted) */
#define OMEGA_DATTN256_ARG_KN 0x3a8u       /* f32 [256] k_norm weight (1 + w, permuted); MUST be ARG_QN + 1024 bytes: the kernel addresses it so */
#define OMEGA_DATTN256_ARG_OUT 0x3b0u      /* f32 [24][256] */
#define OMEGA_DATTN256_ARG_PART 0x3b8u     /* f32 [kv_heads][splits][1548] */
#define OMEGA_DATTN256_ARG_CNT 0x3c0u      /* u32 [kv_heads], zero before the launch */
#define OMEGA_DATTN256_ARG_POS 0x3c8u      /* u32 position of the new token; context = pos + 1 */
#define OMEGA_DATTN256_ARG_PER_SPLIT 0x3ccu /* u32 positions per CTA (multiple of 256) */
#define OMEGA_DATTN256_ARG_SPLITS 0x3d0u   /* u32 = grid_x */
#define OMEGA_DATTN256_ARG_KV_STRIDE 0x3d4u /* u32 bytes per KV head plane */
#define OMEGA_DATTN256_ARG_EPS 0x3d8u      /* f32 rms eps */
#define OMEGA_DATTN256_ARG_OUT_SCALE 0x3dcu /* f32 multiplier on the output */
#define OMEGA_DATTN256_ARG_QK_SCALE 0x3e0u /* f32 (1/sqrt(256)) * log2(e) */
#define OMEGA_DATTN256_ARGS_WORDS 25u

/* shared memory map (bytes) */
#define OMEGA_DATTN256_SM_Q 0u             /* q'[6][256] f32: 6144 (warps 6, 7 stage k and v at 6144.. 8191: inside SM_S, unused then) */
#define OMEGA_DATTN256_SM_S 6144u          /* s[256][8] f32 (6 used): 8192; reused as the reduce buffer r[8 warps][256] */
#define OMEGA_DATTN256_SM_M 14336u         /* m[8 warps][8] then the head maxes M[8]: 256 + 32 */
#define OMEGA_DATTN256_SM_L 14624u         /* l[8 warps][8]: 256 */
#define OMEGA_DATTN256_SM_FLAG 14880u      /* the atomic's old value: 16 */
#define OMEGA_DATTN256_SHARED_BYTES 14896u

typedef struct {
    uint32_t depth;  /* prefetch depth of the K and V streams, 1..4 */
    int mutant;      /* TEST ONLY. 0 none; 1 no RoPE on q and k; 2 V read one position late; 3 the new K/V are not
                      * written to the cache; 4 the cross-split combine ignores the max (differs only when splits > 1);
                      * 5 the odd warps' dim half is dropped from the partial (dims 128..255 wrong). */
} OmegaDecodeAttn256Spec;

/* Builds the IR program (register-allocated) into *prog. 0 ok, -1 refused (depth 0 or > 4, mutant outside 0..5,
 * or the program outgrew BW_MAX_IR_INSNS / BW_MAX_VREGS). */
int omega_blackwell_codegen_decode_attn256_prog(const OmegaDecodeAttn256Spec *spec, BlackwellIRProgram *prog);
/* Instruction and peak-register counts of the program (host, no device). */
int omega_decode_attn256_prog_stats(const OmegaDecodeAttn256Spec *spec, uint32_t *insns, uint32_t *gprs);

#endif
