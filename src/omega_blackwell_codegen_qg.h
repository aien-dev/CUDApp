#ifndef OMEGA_BLACKWELL_CODEGEN_QG_H
#define OMEGA_BLACKWELL_CODEGEN_QG_H
/*
 * omega_blackwell_codegen_qg: native FP8 prefill kernels (OWN-30B L1 GEMM, W-L1G, 2026-10-14). New file.
 *
 * ---- activation quantiser (pq) ----
 * Per row r and 128-wide group b of k:  amax = max(max_j |x[r][128b + j]|, 2^-100)
 *   E = biased f32 exponent of amax, M = its 23 mantissa bits, c = (M + 0x1fffff) >> 23  (1 when amax / 2^(E-127) > 1.75)
 *   scale = 2^(E - 135 + c)  (the smallest power of two with amax / scale <= 448), inv = 2^(135 - E - c)
 *   q[r][128b + j] = e4m3_rne_satfinite(x * inv)   (FMUL by a power of two: exact for every result that can be nonzero)
 *   s[r][b] = scale (f32)
 * Power-of-two scales instead of amax / 448 (stated deviation from the usual dynamic scheme): the GEMM's scale products
 * stay exact and no reciprocal is needed; e4m3 precision is uniform over its normal range, so at most the subnormal tail
 * of a group loses one bit.
 * Layout: x f32 or bf16 rows of k (row stride k elements); q e4m3 rows of k bytes; s f32 [rows][k / 128].
 * One CTA per row (ctaid.x = row), 32 * (k / 128) threads: warp b = group b, lane l = elements 4l .. 4l+3.
 * Arguments (cbank 0x380): x u64, q u64, s u64.
 * mutant (tests only): 1 skips the butterfly (lane-local amax), 2 uses c = 0.
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif
enum { OMEGA_QG_IN_F32 = 0, OMEGA_QG_IN_BF16 = 1 };
typedef struct { uint32_t k, in_type; int mutant; } OmegaPqSpec;
#define OMEGA_PQ_ARGS_WORDS 6u
int omega_pq_spec_ok(const OmegaPqSpec *s);
int omega_blackwell_codegen_pq_prog(const OmegaPqSpec *s, BlackwellIRProgram *prog);
#ifdef __cplusplus
}
#endif
#endif
