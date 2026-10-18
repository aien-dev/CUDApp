#ifndef OMEGA_BLACKWELL_CODEGEN_GDN_H
#define OMEGA_BLACKWELL_CODEGEN_GDN_H
/*
 * omega_blackwell_codegen_gdn: one decode step of a Gated DeltaNet layer of Qwen3.8-27B in ONE launch (omega#542,
 * 2026-10-18). The maths is the host reference sc crates/aien-inference-abi/src/qwen38.rs `delta_net`, from the
 * projections on: the caller runs in_proj_qkv / in_proj_z (FP8 GEMVs) and the stacked a/b rows (bf16 GEMV) first and
 * out_proj (FP8 GEMV) after. Per value head h (48):
 *   conv     channels q[hk], k[hk] (128 each, hk = h / 3) and v[h] (128) of the 10240-wide qkv row: causal depthwise
 *            conv over {tail0, tail1, tail2, x} with the layer's 4 taps, then SiLU; the tail shifts by one.
 *   l2norm   q and k: x / sqrt(sum x^2 + eps_l2); q also / sqrt(128).
 *   scalars  beta = sigmoid(b[h]); g = negA[h] * softplus(a[h] + dt_bias[h]), negA = -exp(A_log); decay = exp(g).
 *   state    S[h] (128 k x 128 v, f32): S *= decay; kv = S^T k; delta = (v - kv) * beta; S += k (outer) delta;
 *            o = S^T q.
 *   output   y = o / sqrt(mean(o^2) + eps_norm) * norm_w * silu(z[h]) * out_scale (norm_w shared by the heads).
 *
 * Grid 3 x 16 (h = ctaid.y * 3 + ctaid.x, hk = ctaid.y), 256 threads, no host round trip, no atomics. Per CTA:
 *   prefetch thread (warp w, lane L) owns S[h] rows 16w .. 16w+15, columns 4L .. 4L+3: 16 LDG.128 issued first on a
 *            held scoreboard, consumed at the state phase (the conv, norm and scalar phases hide the latency).
 *   phase A  thread t convolves channel A (slot t: q channel t for t < 128, k channel t - 128 above) and channel V
 *            (slot 256 + (t & 127), computed by both halves alike); taps and tail are one LDG.128 each.
 *   phase B  sums of squares of q and k per 4-warp half (SHFL + shared), rsqrt, q|k to shared, v to shared; then the
 *            shifted tails are stored (after the barrier: the two halves read the same v tail slot).
 *   phase C  every thread: a and b (one LDG.64), negA and dt_bias (one LDG.64), softplus via EX2/LG2, decay, beta.
 *   phase D  the state update in registers; kv and o are 8-way cross-warp sums through shared memory; S is stored
 *            back with 64 STG.E per thread (immediate offsets).
 *   phase E  every warp holds all 128 o values: sum of squares by SHFL, broadcast through shared, gated norm, store.
 * Branch-free except the lane-0 shared stores (BSSY / BSYNC regions). Ops used: the router's set plus MUFU.LG2
 * (encoding checked with nvdisasm, omega_blackwell_codegen.h) and I2FP.
 */
#include "omega_blackwell_codegen.h"
#include <stdint.h>

#define OMEGA_GDN_KEY_HEADS 16u
#define OMEGA_GDN_VALUE_HEADS 48u
#define OMEGA_GDN_HEAD_DIM 128u                                   /* k and v head dim */
#define OMEGA_GDN_CONV_DIM 10240u                                 /* q 2048 | k 2048 | v 6144 */
#define OMEGA_GDN_V_DIM (OMEGA_GDN_VALUE_HEADS * OMEGA_GDN_HEAD_DIM) /* 6144 */
#define OMEGA_GDN_TAPS 4u
#define OMEGA_GDN_SLOTS 384u                                      /* conv channels one head reads: q | k | v */

/* per-layer parameter block (bytes): conv taps f32 [10240][4] | {negA, dt_bias} f32 [48][2] | norm_w f32 [128] */
#define OMEGA_GDN_PAR_CONV 0u
#define OMEGA_GDN_PAR_SCAL 163840u
#define OMEGA_GDN_PAR_NORM 164224u
#define OMEGA_GDN_PAR_BYTES 164736u
/* per-layer conv tail: f32 [48 heads][384 slots][4] = {x_{t-3}, x_{t-2}, x_{t-1}, unused}; each head keeps its own
 * copy of the q/k channels of its key head (the three heads of a key head hold equal copies), so no CTA reads a word
 * another CTA writes */
#define OMEGA_GDN_TAIL_HEAD_BYTES (OMEGA_GDN_SLOTS * 16u)        /* 6144 */
#define OMEGA_GDN_TAIL_BYTES (OMEGA_GDN_VALUE_HEADS * OMEGA_GDN_TAIL_HEAD_BYTES)
/* per-layer state: f32 [48][128 k][128 v], the reference layout */
#define OMEGA_GDN_STATE_HEAD_BYTES (OMEGA_GDN_HEAD_DIM * OMEGA_GDN_HEAD_DIM * 4u) /* 65536 */
#define OMEGA_GDN_STATE_BYTES (OMEGA_GDN_VALUE_HEADS * OMEGA_GDN_STATE_HEAD_BYTES)

/* constant bank: addresses at 0x380 + 8n, then the scalars */
#define OMEGA_GDN_ARG_QKV 0x380u        /* f32 [10240] in_proj_qkv row (times in_scale) */
#define OMEGA_GDN_ARG_Z 0x388u          /* f32 [6144] in_proj_z row (times in_scale) */
#define OMEGA_GDN_ARG_AB 0x390u         /* f32 [48][2] {a[h], b[h]}: the bf16 GEMV of in_proj_a / in_proj_b rows interleaved */
#define OMEGA_GDN_ARG_PAR 0x398u        /* the layer's parameter block */
#define OMEGA_GDN_ARG_TAIL 0x3a0u       /* the layer's conv tail */
#define OMEGA_GDN_ARG_STATE 0x3a8u      /* the layer's state */
#define OMEGA_GDN_ARG_OUT 0x3b0u        /* f32 [6144] y (the out_proj input) */
#define OMEGA_GDN_ARG_EPS_L2 0x3b8u     /* f32, 1e-6 in the reference */
#define OMEGA_GDN_ARG_EPS_NORM 0x3bcu   /* f32, rms_norm_eps */
#define OMEGA_GDN_ARG_IN_SCALE 0x3c0u   /* f32: qkv and z are multiplied by it on read */
#define OMEGA_GDN_ARG_OUT_SCALE 0x3c4u  /* f32: y is multiplied by it */
#define OMEGA_GDN_ARGS_WORDS 18u

/* shared memory map (bytes) */
#define OMEGA_GDN_SM_Q 0u               /* q[128] f32, normalised and / sqrt(128) */
#define OMEGA_GDN_SM_K 512u             /* k[128] f32 */
#define OMEGA_GDN_SM_V 1024u            /* v[128] f32 */
#define OMEGA_GDN_SM_RED 1536u          /* per-warp sums of squares of q|k [8] */
#define OMEGA_GDN_SM_P 1600u            /* per-warp partials [8][128] f32 (kv, then o) */
#define OMEGA_GDN_SM_RED2 5696u         /* per-warp sums of squares of o [8] */
#define OMEGA_GDN_SHARED_BYTES 5728u

#define OMEGA_GDN_THREADS 256u
#define OMEGA_GDN_GRID_X 3u
#define OMEGA_GDN_GRID_Y 16u

typedef struct {
    int mutant; /* TEST ONLY. 0 none; 1 no decay (decay = 1); 2 beta = 1; 3 the output gate is z, not silu(z);
                 * 4 the conv tail is not shifted (the next step convolves stale history). */
} OmegaGdnSpec;

/* Builds the register-allocated IR program into *prog. 0 ok, -1 refused. */
int omega_blackwell_codegen_gdn_prog(const OmegaGdnSpec *spec, BlackwellIRProgram *prog);
/* Instruction and peak-register counts (host, no device). */
int omega_gdn_prog_stats(const OmegaGdnSpec *spec, uint32_t *insns, uint32_t *gprs);

#endif
