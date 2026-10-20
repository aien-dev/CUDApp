#ifndef OMEGA_BLACKWELL_CODEGEN_DEC_H
#define OMEGA_BLACKWELL_CODEGEN_DEC_H
/*
 * omega_blackwell_codegen_dec: the small glue kernels that keep a Qwen3-30B decode token on the chip (OWN-30B W-L2,
 * 2026-10-13). Before them the host did the residual adds, the input RMSNorm, SwiGLU, the expert combine and the
 * argmax between launches; with them a whole token is one command list (omega_gpu_cmdlist.h).
 *
 *   AXPY    x[i] += a[i] in place over iters * stride elements (grid-stride, uniform trip count). Chip test only: a
 *           chain of these in one list catches overlapping grids (a lost increment).
 *   RES     one CTA of 256 threads over the residual row (2048 wide by default, spec.width), width / 256 values per
 *           thread. Mode bits:
 *             ADD      x += a                                         (a = o_proj output)
 *             COMBINE  x += sum_g w[g] * d[g]  (g = 0..7, as the host: m = w0*d0, then m = m + w_g*d_g, x = x + m)
 *             NORM     y = x * RSQ(mean(x^2) + eps) * nw * scale     (scale 2^64 = the FP8 GEMV x' row, 1 = plain)
 *           x is written back when ADD or COMBINE is set.
 *   SWIGLU  act[g][c] = silu(gu[g][c]) * gu[g][I + c] * scale, g = ctaid.y (8 experts, or 1 dense row), c = ctaid.x * 256
 *           + tid, I the intermediate size (768 by default, spec.width; grid I / 256 x groups).
 *           silu(v) = v * RCP(1 + EX2(-v * log2 e)).
 *   GATHER  out[r][n] = in[r * stride + tab[n]] for n < width (spec.width, spec.stride: floats), r = ctaid.y. Qwen3.8-27B
 *           attention: the fused q|gate|k|v GEMV row to the attention kernel's permuted q|k|v plus the gate (sc #642;
 *           the permutation crosses the FP8 weights' 128-row scale blocks, so it cannot be folded into the weights).
 *   GATE    out[n] = a[n] * sigmoid(g[n]) for n < count, sigmoid(v) = RCP(1 + EX2(-v * log2 e)): the attention output gate.
 *   ARGMAX  CTA c reads 2048 values v[c*2048 + 8*tid + j] (and indices, or the identity index), writes the max and the
 *           lowest index holding it to out_v[c], out_i[c]. Two launches give the greedy token of 151936 logits
 *           (75 CTAs, then 1 CTA over the 75 partials padded with -1e30). The lowest-index rule is the host's
 *           sample_argmax (strict >). Max by FMNMX, the index by the router's trick (float 2^23 + idx, lifted when the
 *           value differs from the max, FMNMX.MIN). No FSETP / FSEL.
 *
 * Constant bank: addresses at 0x380 + 8n, then scalars (offsets below). Every load carries the read barrier, as in the
 * other OWN-30B emitters.
 */
#include "omega_blackwell_codegen.h"
#include <stdint.h>

typedef enum { OMEGA_DEC_AXPY = 1, OMEGA_DEC_RES = 2, OMEGA_DEC_SWIGLU = 3, OMEGA_DEC_ARGMAX = 4, OMEGA_DEC_GATHER = 5, OMEGA_DEC_GATE = 6 } OmegaDecKind;

#define OMEGA_DEC_RES_ADD 1u
#define OMEGA_DEC_RES_COMBINE 2u
#define OMEGA_DEC_RES_NORM 4u
#define OMEGA_DEC_RES_ROWS 8u     /* row ctaid.y of a [rows][width] block: x, a and y move by width per row, nw does not */

#define OMEGA_DEC_HIDDEN 2048u
#define OMEGA_DEC_THREADS 256u
#define OMEGA_DEC_TOPK 8u
#define OMEGA_DEC_INTER 768u
#define OMEGA_DEC_ARGMAX_SPAN 2048u /* values per ARGMAX CTA */

/* AXPY args */
#define OMEGA_DEC_AXPY_X 0x380u
#define OMEGA_DEC_AXPY_A 0x388u
#define OMEGA_DEC_AXPY_ITERS 0x390u
#define OMEGA_DEC_AXPY_WORDS 5u
/* RES args */
#define OMEGA_DEC_RES_X 0x380u      /* f32 [2048] residual, read and (ADD/COMBINE) written */
#define OMEGA_DEC_RES_A 0x388u      /* ADD: f32 [2048]; COMBINE: f32 [8][2048] down outputs */
#define OMEGA_DEC_RES_W 0x390u      /* COMBINE: f32 [8] routing weights */
#define OMEGA_DEC_RES_NW 0x398u     /* NORM: f32 [2048] norm weight */
#define OMEGA_DEC_RES_Y 0x3a0u      /* NORM: f32 [2048] output */
#define OMEGA_DEC_RES_EPS 0x3a8u
#define OMEGA_DEC_RES_SCALE 0x3acu
#define OMEGA_DEC_RES_WORDS 12u
/* SWIGLU args */
#define OMEGA_DEC_SW_GU 0x380u      /* f32 [8][1536] gate|up */
#define OMEGA_DEC_SW_ACT 0x388u     /* f32 [8][768] */
#define OMEGA_DEC_SW_SCALE 0x390u
#define OMEGA_DEC_SW_WORDS 5u
/* ARGMAX args */
#define OMEGA_DEC_AM_V 0x380u       /* f32 values, CTA c reads [c*2048, +2048) */
#define OMEGA_DEC_AM_I 0x388u       /* u32 indices (indexed spec), same layout */
#define OMEGA_DEC_AM_OV 0x390u      /* f32 [ctas] */
#define OMEGA_DEC_AM_OI 0x398u      /* u32 [ctas] */
#define OMEGA_DEC_AM_WORDS 8u
/* GATHER args (out[r][n] = in[r * stride + tab[n]], n < width; grid ceil(width / 256) x rows) */
#define OMEGA_DEC_GA_IN 0x380u
#define OMEGA_DEC_GA_TAB 0x388u     /* u32 [width] source index within an input row */
#define OMEGA_DEC_GA_OUT 0x390u
#define OMEGA_DEC_GA_WORDS 6u
/* GATE args (out[n] = a[n] * sigmoid(g[n]), n < count; grid ceil(count / 256); out may be a) */
#define OMEGA_DEC_GT_A 0x380u
#define OMEGA_DEC_GT_G 0x388u
#define OMEGA_DEC_GT_OUT 0x390u
#define OMEGA_DEC_GT_N 0x398u       /* u32 count */
#define OMEGA_DEC_GT_WORDS 7u

#define OMEGA_DEC_SM_RED 0u         /* 8 per-warp partials */
#define OMEGA_DEC_SM_RED2 64u
#define OMEGA_DEC_SHARED_BYTES 128u

typedef struct {
    OmegaDecKind kind;
    uint32_t mode;      /* RES mode bits; ARGMAX: 1 = indexed (indices loaded), 0 = identity */
    uint32_t stride;    /* AXPY: elements per iteration (grid_x * 256). GATHER: the input row stride (floats) */
    int mutant;         /* TEST ONLY. RES: 1 = norm without the weight; 2 = COMBINE skips expert 7.
                           SWIGLU: 1 = up half read at +767. ARGMAX: 1 = last equal index instead of first.
                           GATHER: 1 = source index + 1. GATE: 1 = the gate dropped (out = a). 0 none */
    uint32_t width;     /* RES: the row width (0 = OMEGA_DEC_HIDDEN; a multiple of 1024 up to 8192, COMBINE only at 2048).
                           SWIGLU: the intermediate size (0 = OMEGA_DEC_INTER; a multiple of 256). Qwen3.8-27B: 5120 and
                           17408 (sc #642). The default widths emit the same program as before the field existed. */
    uint32_t groups;    /* RES COMBINE: 0 or 8 routed experts, 9 = 8 + the shared expert (Qwen3.6-35B-A3B; the weights and down rows
                           have 9 entries, slot 8 the shared expert). Mutant 3 (9 only): the shared slot is skipped. */
} OmegaDecSpec;

int omega_blackwell_codegen_dec_prog(const OmegaDecSpec *spec, BlackwellIRProgram *prog);
int omega_dec_prog_stats(const OmegaDecSpec *spec, uint32_t *insns, uint32_t *gprs);

#endif
