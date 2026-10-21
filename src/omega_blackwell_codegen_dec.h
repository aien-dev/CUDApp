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
 *   XBOUND  (HUM L314 exact two-pass head) every CTA computes t = sum h^2 + 2^-100 over the 2048 plain hidden values and
 *           hn = t * RSQ(t) * (1 + 2^-10) >= |h| (the widening covers the f32 sum, RSQ and the products); then row r
 *           (r = ctaid * 256 + tid < count) writes B = c[r] * hn + |s[r]| * 2^-20 + 2^-100, lo = s + (-B), hi = s + B, so
 *           lo <= s - c|h| and hi >= s + c|h| exactly despite rounding. No FSUB / FSETP (negation by the sign bit).
 *           Mutant 1 drops the c[r] * hn term (the bound is then far too small: the identity gate must catch it).
 *           Mode 1 (SCALE): lo[n] = s[n] * 2^64, n < count, grid ceil(count / 256): the FP8 screen's x' row made from the
 *           plain hidden, exactly h * 2^64, so the screen and the recheck see the same h.
 *   GROUP   (L312 v2, the stroke) one CTA turns the expert picks of N conversations into one grouped-GEMM segment table.
 *           Row t = h * S + j (head h, slot j; S = spec.groups slots per head, 9 for the 35B) carries expert id
 *           IDS[h * STRIDE + j * 4]. Rows are sorted by (id, t): sorted row p of row t is its rank. XMAP[p] = h (the x'
 *           row the gate|up GEMM reads), YMAP[p] = t (its y row, head order). Distinct ids in ascending order get
 *           segments s = 0, 1, ...: SEG_GU[s] = {TGU[h][j].w_va, .s_va, row0 = rank of its first row, cnt}, SEG_DN[s]
 *           the same from TDN; segments D..R-1 (D distinct of R = N * S rows) are written as zeros (cnt 0: the GEMM
 *           CTA leaves); map entries R..R+7 are 0 (the GEMM's padding rows). N * S <= 128, N = HEADS at run time.
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

typedef enum { OMEGA_DEC_AXPY = 1, OMEGA_DEC_RES = 2, OMEGA_DEC_SWIGLU = 3, OMEGA_DEC_ARGMAX = 4, OMEGA_DEC_GATHER = 5, OMEGA_DEC_GATE = 6, OMEGA_DEC_GROUP = 7, OMEGA_DEC_XBOUND = 8 } OmegaDecKind;

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
/* XBOUND args (HUM L314 exact head; lo[r] = s[r] - B[r], hi[r] = s[r] + B[r], r < count; grid ceil(count / 256)) */
#define OMEGA_DEC_XB_S 0x380u      /* f32 [count] screen logits */
#define OMEGA_DEC_XB_C 0x388u      /* f32 [count] Row bound coefficients */
#define OMEGA_DEC_XB_H 0x390u      /* f32 [2048] the plain hidden (the recheck x) */
#define OMEGA_DEC_XB_LO 0x398u
#define OMEGA_DEC_XB_HI 0x3a0u
#define OMEGA_DEC_XB_N 0x3a8u      /* u32 count */
#define OMEGA_DEC_XB_WORDS 11u
/* GROUP args (one CTA of 256 threads; shared: ids at 0, first-of-id flags at 512) */
#define OMEGA_DEC_GR_IDS 0x380u     /* u32 [S] of head 0; head h at + h * STRIDE */
#define OMEGA_DEC_GR_TGU 0x388u     /* {w_va, s_va} [S] of head 0 (gate|up directory entries), head h at + h * STRIDE */
#define OMEGA_DEC_GR_TDN 0x390u     /* the same for down */
#define OMEGA_DEC_GR_SGU 0x398u     /* out: OMEGA_GEMM_SEG_BYTES [R] segment table for gate|up */
#define OMEGA_DEC_GR_SDN 0x3a0u     /* out: [R] for down */
#define OMEGA_DEC_GR_XMAP 0x3a8u    /* out: u32 [R + 8] */
#define OMEGA_DEC_GR_YMAP 0x3b0u    /* out: u32 [R + 8] */
#define OMEGA_DEC_GR_STRIDE 0x3b8u  /* u32 bytes between one head's router buffer and the next */
#define OMEGA_DEC_GR_HEADS 0x3bcu   /* u32 N */
#define OMEGA_DEC_GR_WORDS 16u
#define OMEGA_DEC_GR_PAD 8u         /* map padding entries (the GEMM's largest rows-per-pass) */
#define OMEGA_DEC_GR_MAX_ROWS 128u
#define OMEGA_DEC_GR_SHARED_BYTES 1024u
#define OMEGA_DEC_GR_SEG_BYTES 32u

#define OMEGA_DEC_SM_RED 0u         /* 8 per-warp partials */
#define OMEGA_DEC_SM_RED2 64u
#define OMEGA_DEC_SHARED_BYTES 128u

typedef struct {
    OmegaDecKind kind;
    uint32_t mode;      /* RES mode bits; ARGMAX: 1 = indexed (indices loaded), 0 = identity */
    uint32_t stride;    /* AXPY: elements per iteration (grid_x * 256). GATHER: the input row stride (floats) */
    int mutant;         /* TEST ONLY. RES: 1 = norm without the weight; 2 = COMBINE skips expert 7.
                           SWIGLU: 1 = up half read at +767. ARGMAX: 1 = last equal index instead of first.
                           GATHER: 1 = source index + 1. GATE: 1 = the gate dropped (out = a).
                           GROUP: 1 = sort key without the row tie-break; 2 = segment cnt one short; 3 = the trailing
                           empty segments left unwritten. 0 none */
    uint32_t width;     /* RES: the row width (0 = OMEGA_DEC_HIDDEN; a multiple of 1024 up to 8192, COMBINE only at 2048).
                           SWIGLU: the intermediate size (0 = OMEGA_DEC_INTER; a multiple of 256). Qwen3.8-27B: 5120 and
                           17408 (sc #642). The default widths emit the same program as before the field existed. */
    uint32_t groups;    /* RES COMBINE: 0 or 8 routed experts, 9 = 8 + the shared expert (Qwen3.6-35B-A3B; the weights and down rows
                           have 9 entries, slot 8 the shared expert). Mutant 3 (9 only): the shared slot is skipped.
                           GROUP: slots per head S (0 = 8; 9 for the 35B), 1..16. */
} OmegaDecSpec;

int omega_blackwell_codegen_dec_prog(const OmegaDecSpec *spec, BlackwellIRProgram *prog);
int omega_dec_prog_stats(const OmegaDecSpec *spec, uint32_t *insns, uint32_t *gprs);

#endif
