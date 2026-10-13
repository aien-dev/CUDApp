#ifndef OMEGA_BLACKWELL_CODEGEN_ATTN_H
#define OMEGA_BLACKWELL_CODEGEN_ATTN_H
/*
 * omega_blackwell_codegen_attn: decode attention for Qwen3-30B-A3B in ONE launch per layer (OWN-30B cut 8,
 * 2026-10-13). Grouped-query attention with 8 query heads per KV head, head_dim 128, per-head RMSNorm on q
 * and k, rotary embedding from a host-built cos/sin table, the new token's K/V appended to a GPU-resident
 * bf16 cache inside the kernel, softmax over the whole context, output ready for o_proj (scaled by
 * `out_scale`, so it can be the x' row of the FP8 GEMV without a host round trip).
 *
 * Grid: (splits, kv_heads). CTA (cx, cy) handles KV head cy and the positions [cx*per_split, min(+per_split,
 * pos+1)) of the context; the host picks splits = ceil((pos+1)/per_split), so no CTA is empty. 8 warps = 256
 * threads per CTA (fixed: warp h = query head h of the group in the prologue, lane = position in the score pass).
 *
 * Per CTA:
 *   prologue  warp h: q[cy*8+h] (f32 quad per lane) -> RMSNorm(q_norm, eps) -> RoPE(table[pos]) -> * qk_scale,
 *             staged in shared memory as q'[h][128]. Every warp also computes k_new = RoPE(RMSNorm(k[cy]))
 *             and v_new; the CTA whose range holds `pos` (the last split) writes them as bf16 into the cache at
 *             [cy][pos] (warp 0 K, warp 1 V), then MEMBAR + BAR.SYNC, so its own stream reads them back.
 *   chunks    of 256 positions: score pass (thread = position: the bf16 K row streamed in 16 quads with a
 *             `depth`-deep prefetch, 8 dot products against q' from shared memory, scores to shared s[t][8]);
 *             V pass (warp w: positions w, w+8, ...; lane = 4 dims; online softmax per warp: running max,
 *             alpha = EX2(m_old - m_new) applied only when the max moved (warp-uniform branch), p = EX2(s - m),
 *             acc += p * v, l += p).
 *   epilogue  warps combine (m, l, acc) through shared memory and write the CTA partial
 *             {m[8], l[8], acc[8][128]} to partials[cy][cx]; then one ATOMG.ADD on counters[cy]; the last CTA
 *             of the KV head combines the partials with the usual max rescale and writes out[(cy*8+h)*128+d]
 *             * out_scale, and resets the counter.
 *
 * Numbers (f32 on the chip, double in the host reference, tolerances in tests/gpu_decode_attn_test.c):
 *   scores in log2 units: q' already carries (1/sqrt(128)) * log2(e), so p = EX2(s - m) = exp((q.k)/sqrt(d) - M).
 *   bf16 K/V are widened by a 16-bit shift; the new K/V are rounded to bf16 round-to-nearest-even before storing
 *   (the reference rounds the same way, so the attention of the new token sees the stored value).
 *
 * The host simulator (tests/bw_warp_sim_k.h) runs this kernel at 8 warps; a simulator pass is never a chip result.
 */
#include "omega_blackwell_codegen.h"
#include <stdint.h>

#define OMEGA_DATTN_HEAD_DIM 128u
#define OMEGA_DATTN_Q_PER_KV 8u
#define OMEGA_DATTN_WARPS 8u
#define OMEGA_DATTN_CHUNK 256u                 /* positions per score pass = threads per CTA */
#define OMEGA_DATTN_ROW_BYTES 256u             /* one bf16 K or V row */
#define OMEGA_DATTN_ROPE_ROW_BYTES 512u        /* cos[64] then sin[64], f32 */
#define OMEGA_DATTN_PARTIAL_WORDS 1040u        /* m[8], l[8], acc[8][128] */
#define OMEGA_DATTN_MAX_DEPTH 4u
/* bytes past the last row of a K or V plane the prefetch may touch (clamped reads of the last rows) */
#define OMEGA_DATTN_PLANE_SLACK (OMEGA_DATTN_MAX_DEPTH * 8u * OMEGA_DATTN_ROW_BYTES + 64u)

/* constant bank: every address at 0x380 + 8n, then the scalars */
#define OMEGA_DATTN_ARG_QKV 0x380u      /* f32 [32*128 q | 4*128 k | 4*128 v] */
#define OMEGA_DATTN_ARG_KC 0x388u       /* bf16 K cache base: [kv_head][max_pos][128], kv_stride bytes per head */
#define OMEGA_DATTN_ARG_VC 0x390u       /* bf16 V cache base, same layout */
#define OMEGA_DATTN_ARG_ROPE 0x398u     /* f32 [max_pos][cos 64 | sin 64] */
#define OMEGA_DATTN_ARG_QN 0x3a0u       /* f32 [128] q_norm weight */
#define OMEGA_DATTN_ARG_KN 0x3a8u       /* f32 [128] k_norm weight */
#define OMEGA_DATTN_ARG_OUT 0x3b0u      /* f32 [32][128] */
#define OMEGA_DATTN_ARG_PART 0x3b8u     /* f32 [kv_heads][splits][1040] */
#define OMEGA_DATTN_ARG_CNT 0x3c0u      /* u32 [kv_heads], zero before the launch */
#define OMEGA_DATTN_ARG_POS 0x3c8u      /* u32 position of the new token; context = pos + 1 */
#define OMEGA_DATTN_ARG_PER_SPLIT 0x3ccu /* u32 positions per CTA (multiple of 256) */
#define OMEGA_DATTN_ARG_SPLITS 0x3d0u   /* u32 = grid_x */
#define OMEGA_DATTN_ARG_KV_STRIDE 0x3d4u /* u32 bytes per KV head plane */
#define OMEGA_DATTN_ARG_EPS 0x3d8u      /* f32 rms eps */
#define OMEGA_DATTN_ARG_OUT_SCALE 0x3dcu /* f32 multiplier on the output (2^64 for the FP8 GEMV x' row, 1 otherwise) */
#define OMEGA_DATTN_ARG_QK_SCALE 0x3e0u /* f32 (1/sqrt(128)) * log2(e) */
#define OMEGA_DATTN_ARGS_WORDS 25u

/* shared memory map (bytes) */
#define OMEGA_DATTN_SM_Q 0u            /* q'[8][128] f32: 4096 */
#define OMEGA_DATTN_SM_S 4096u         /* s[256][8] f32: 8192; reused as the reduce buffer r[8 warps][256] */
#define OMEGA_DATTN_SM_M 12288u        /* m[8 warps][8] then the head maxes M[8]: 256 + 32 */
#define OMEGA_DATTN_SM_L 12576u        /* l[8 warps][8]: 256 */
#define OMEGA_DATTN_SM_FLAG 12832u     /* the atomic's old value: 16 */
#define OMEGA_DATTN_SM_K 12864u        /* k_new[128] f32 staging: 512 */
#define OMEGA_DATTN_SHARED_BYTES 13376u

typedef struct {
    uint32_t depth;  /* prefetch depth of the K and V streams, 1..4 */
    int mutant;      /* TEST ONLY. 0 none; 1 no RoPE on q and k; 2 V read one position late; 3 the new K/V are not
                      * written to the cache; 4 the cross-split combine ignores the max (differs only when splits > 1). */
} OmegaDecodeAttnSpec;

/* Builds the IR program (register-allocated) into *prog. 0 ok, -1 refused (depth 0 or > 4, mutant outside 0..4,
 * or the program outgrew BW_MAX_IR_INSNS / BW_MAX_VREGS). */
int omega_blackwell_codegen_decode_attn_prog(const OmegaDecodeAttnSpec *spec, BlackwellIRProgram *prog);
/* Instruction and peak-register counts of the program (host, no device). */
int omega_decode_attn_prog_stats(const OmegaDecodeAttnSpec *spec, uint32_t *insns, uint32_t *gprs);

#endif
