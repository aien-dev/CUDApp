/* bw_mufu_ftz.h: the GB10 MUFU unit's flush-to-zero rule, shared by every host model of it (omega #462).
 *
 * MUFU (the SASS EX2 / RCP / RSQ unit) flushes a subnormal result to a sign-preserving zero. Chip-observed on GB10: the
 * SwiGLU reciprocal 1 / (2^t + 1) for t in (126, 128) came back as +-0 on all 101 of 101 differing elements (evidence
 * OWN30B-L1G-PFGEMM-SWIGLU-FTZ-df9bd9c4, gpu_pfgemm_test host red test). Doc-inferred, not separately chip-observed: the
 * input flush and the EX2 / RSQ cases follow the same rule, PTX ISA 9.7.3.13 (rcp) and 9.7.3.21 (ex2) "flushes subnormal
 * inputs and results to sign-preserving zero" for the .ftz form, which is what the bare MUFU we emit is (the non-ftz PTX
 * variants are ptxas sequences around MUFU; NVIDIA publishes no SASS semantics). Limit: the chip's MUFU is approximate
 * (RCP 1 ulp, EX2 2 ulp), the model flushes a correctly rounded IEEE result, so right at the subnormal boundary the two
 * can disagree 0 versus nonzero.
 *
 * Users: tests/bw_warp_sim.h, bw_warp_sim_k.h, bw_warp_sim_pa.h, bw_warp_sim_tc.h (MUFU cases), the attention test's own
 * per-thread simulator (gpu_attention_test.c) and the pfgemm / qgemm SwiGLU host references. omega_numeric.c has its own
 * ftz() on the library side. Self-contained (no simulator state) so any of them can include it, in any order. */
#ifndef BW_MUFU_FTZ_H
#define BW_MUFU_FTZ_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/* A subnormal (exponent field 0, mantissa nonzero) becomes a zero of the same sign; zeros, normals, inf and NaN pass. */
static inline float ws_mufu_ftz(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0u) u &= 0x80000000u;
    memcpy(&f, &u, 4);
    return f;
}
/* The three MUFU functions the codegen emits: input flushed, IEEE f32 function, result flushed. */
static inline float ws_mufu_ex2(float x) { return ws_mufu_ftz(exp2f(ws_mufu_ftz(x))); }
static inline float ws_mufu_rcp(float x) { return ws_mufu_ftz(1.0f / ws_mufu_ftz(x)); }
static inline float ws_mufu_rsq(float x) { return ws_mufu_ftz(1.0f / sqrtf(ws_mufu_ftz(x))); }

#endif
