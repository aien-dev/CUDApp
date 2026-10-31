/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_BLACKWELL_CODEGEN_GDNC_H
#define OMEGA_BLACKWELL_CODEGEN_GDNC_H
/*
 * omega_blackwell_codegen_gdnc.h (DRA-84): the chunked tensor-core gated-delta-rule prefill scan. One CTA (256 threads)
 * per value head, grid (ratio, 16), h = cy * ratio + cx, key head cy; the same head mapping, parameter block, conv tail
 * and f32 state as the sequential multi-token kernel (omega_blackwell_codegen_gdn_multi_prog), which is its golden.
 *
 * Per 16-token sub-chunk (WY form, design ~/handoffs/pdra4-hive/DRA-84-GDNC-DESIGN.md):
 *   P1 conv + SiLU of q|k|v (192 threads, two channels each) and the gate scalars (warp 7): lg_t = log2 decay_t,
 *      b_t = sigmoid(beta); E_s = sum_{i>s} lg_i, GAM_t = 2^(tot - E_t) = P_t, LAM_s = 2^E_s, GC = 2^tot.
 *   P2 l2 norms; q^, k^ in the HMMA operand layouts (bf16).
 *   P3 gram (HMMA): AM[t][s] = -b_t (P_t/P_s) k^_t.k^_s (s < t), M[t][s] = (P_t/P_s) q^_t.k^_s (s <= t).
 *   P4 forward substitution (f32): U = (I+A)^-1 diag(b) V, W = (I+A)^-1 (-diag(b P) K).
 *   P5 HMMA: D^T = U^T + S0^T W^T; O^T = S0^T (diag(P) Q)^T + D^T M^T; S^T = GC S0^T + D^T (diag(LAM) K).
 *   P6 gated RMSNorm, bf16 out rows.
 * Operands of the HMMAs are bf16 (S0, k^, q^, W, D, P-scaled k^), accumulation f32; the state stays f32 in registers.
 *
 * Input: bf16 rows of in_proj qkv | z (z at conv * 2 bytes), times IN_SCALE. Output: bf16 rows [vh][128], times OUT_SCALE.
 * Snapshots: every SNAP_EVERY4 sub-chunks (4 per 64-token chunk) the state and conv tail of the head are stored to
 * SNAP + k * SNAP_STRIDE ([vh] f32 state | [vh] tails, the layer layout), for at most SNAP_COUNT snapshots.
 */
#include "omega_blackwell_codegen.h"
#include <stdint.h>

#define OMEGA_GDNC_ARG_QKVZ 0x380u        /* bf16 [conv + vdim] in_proj qkv | z row of token 0 */
#define OMEGA_GDNC_ARG_AB 0x388u          /* f32 [vh][2] {a, b} row of token 0 */
#define OMEGA_GDNC_ARG_PAR 0x390u         /* the layer parameter block (omega_gdn_shape) */
#define OMEGA_GDNC_ARG_TAIL 0x398u        /* the layer conv tail, f32 */
#define OMEGA_GDNC_ARG_STATE 0x3a0u       /* the layer state, f32 [vh][128 k][128 v] */
#define OMEGA_GDNC_ARG_OUT 0x3a8u         /* bf16 [vh][128] out row of token 0 */
#define OMEGA_GDNC_ARG_SNAP 0x3b0u        /* snapshot 0 (unused when SNAP_COUNT = 0) */
#define OMEGA_GDNC_ARG_EPS_L2 0x3b8u
#define OMEGA_GDNC_ARG_EPS_NORM 0x3bcu
#define OMEGA_GDNC_ARG_IN_SCALE 0x3c0u
#define OMEGA_GDNC_ARG_OUT_SCALE 0x3c4u
#define OMEGA_GDNC_ARG_T 0x3c8u           /* tokens, >= 1 */
#define OMEGA_GDNC_ARG_XSTRIDE 0x3ccu     /* bytes between qkvz rows */
#define OMEGA_GDNC_ARG_ABSTRIDE 0x3d0u    /* bytes between ab rows */
#define OMEGA_GDNC_ARG_OSTRIDE 0x3d4u     /* bytes between out rows */
#define OMEGA_GDNC_ARG_SNAP_EVERY4 0x3d8u /* sub-chunks (16 tokens) between snapshots: 4 x the 64-token chunk count */
#define OMEGA_GDNC_ARG_SNAP_COUNT 0x3dcu
#define OMEGA_GDNC_ARG_SNAP_STRIDE 0x3e0u /* bytes between snapshots */
#define OMEGA_GDNC_ARGS_WORDS 25u

#define OMEGA_GDNC_THREADS 256u
#define OMEGA_GDNC_SUB 16u                /* tokens per sub-chunk */
#define OMEGA_GDNC_SNAP_CHUNK 64u         /* tokens per snapshot unit */

/* shared memory map (bytes) */
#define OMEGA_GDNC_SM_XQ 0u      /* raw q f32 [16][128]; then QB (B-perm of P q^, bf16) at 0 and QT (A-perm of q^) at 4096 */
#define OMEGA_GDNC_SM_QB 0u
#define OMEGA_GDNC_SM_QT 4096u
#define OMEGA_GDNC_SM_XK 8192u   /* raw k f32 [16][128]; then KHT (k^ f32 [128][16]); then WB (B-perm of W) | KT */
#define OMEGA_GDNC_SM_WB 8192u
#define OMEGA_GDNC_SM_KT 12288u  /* B-perm of diag(LAM) K over t, bf16 */
#define OMEGA_GDNC_SM_XV 16384u  /* v f32 [128][16] (transposed); then UT (U^T f32 in C-fragment order) */
#define OMEGA_GDNC_SM_KB 24576u  /* B-perm of k^ bf16 */
#define OMEGA_GDNC_SM_AM 28672u  /* f32 [16][16] -b_t P_t/P_s k^_t.k^_s, strictly lower */
#define OMEGA_GDNC_SM_PB 29696u  /* the M^T B fragment, bf16 */
#define OMEGA_GDNC_SM_SC 30208u  /* the per-token scalars, 8 slots of [16] f32 */
#define OMEGA_GDNC_SM_ROFF 30720u /* u32 [16] qkvz row offset of each token (clamped) */
#define OMEGA_GDNC_SM_KA 30784u  /* A-perm of k^ bf16 */
#define OMEGA_GDNC_SM_XO 34880u  /* O f32 [16][128] */
#define OMEGA_GDNC_SHARED_BYTES 43072u

#define OMEGA_GDNC_SC_E 0u
#define OMEGA_GDNC_SC_BETA 64u
#define OMEGA_GDNC_SC_NBETA 128u
#define OMEGA_GDNC_SC_GAM 192u
#define OMEGA_GDNC_SC_LAM 256u
#define OMEGA_GDNC_SC_NBG 320u
#define OMEGA_GDNC_SC_GC 384u
#define OMEGA_GDNC_SC_ONES 448u

#define OMEGA_GDNC_MUTANT_DROP_DECAY 9 /* AIEN_OMEGA_Q36_PF_MUTANT=9: the decay is dropped (lg = 0); must go red */

typedef struct {
    int mutant;     /* 0, or OMEGA_GDNC_MUTANT_DROP_DECAY (TEST ONLY) */
    uint32_t ratio; /* value heads per key head, as OmegaGdnSpec.ratio (0 = 3; 2 = Qwen3.6-35B-A3B) */
} OmegaGdncSpec;

/* 0 ok; -1 bad spec or an emitter/regalloc failure */
int omega_blackwell_codegen_gdnc_prog(const OmegaGdncSpec *spec, BlackwellIRProgram *prog);
/* instruction count, virtual registers, peak physical registers of the program */
int omega_gdnc_prog_stats(const OmegaGdncSpec *spec, uint32_t *insns, uint32_t *vregs, uint32_t *gprs);
/* the placeholder words (SHFL.BFLY, F2FP) omega_tc_patch_code must patch in the encoded program */
uint32_t omega_gdnc_placeholder_count(const BlackwellIRProgram *prog);

#endif
