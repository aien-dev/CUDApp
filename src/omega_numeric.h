/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OMEGA_NUMERIC_H
#define OMEGA_NUMERIC_H

/*
 * OMEGA-NUMERIC-0 (M19R Gate 5): FP32 substrate.
 *
 * This file is host-pure: the semantic reference tier, the CPU realization
 * tier, the Omega math sequences, the op registry (which ops have a GB10
 * encoding and how their results are compared), the kernel patch words, and
 * the parity comparators. Nothing here opens a device. The GB10 executor is
 * src/omega_numeric_gb10.c.
 *
 * Build with -ffp-contract=off: the reference and CPU tiers must not depend
 * on whether the compiler fuses a*b+c. Fused operations are written as
 * explicit AArch64 instructions.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Frozen numerical reduction order specification:
 * Pairwise binary tree across 32 lanes (delta = 16, 8, 4, 2, 1).
 */
#define OMEGA_WARP_REDUCTION_DECLARED_ORDER "PAIRWISE_TREE_LANE_DELTA_16_8_4_2_1"

/*
 * Bitcast utility functions
 */
static inline uint32_t omega_float_to_bits(float f) {
    union { float f; uint32_t u; } u;
    u.f = f;
    return u.u;
}

static inline float omega_bits_to_float(uint32_t u) {
    union { float f; uint32_t u; } un;
    un.u = u;
    return un.f;
}

/*
 * Bit-level IEEE-754 classification and manipulation (Zero Libm)
 */
#define OMEGA_INF_POS   0x7f800000U
#define OMEGA_INF_NEG   0xff800000U
#define OMEGA_QNAN_BITS 0x7fc00000U

static inline float omega_fabs(float x) {
    return omega_bits_to_float(omega_float_to_bits(x) & 0x7fffffffU);
}

static inline bool omega_isnan(float x) {
    uint32_t u = omega_float_to_bits(x);
    return ((u & 0x7f800000U) == 0x7f800000U) && ((u & 0x007fffffU) != 0);
}

static inline bool omega_isinf(float x) {
    uint32_t u = omega_float_to_bits(x);
    return (u & 0x7fffffffU) == 0x7f800000U;
}

static inline bool omega_signbit(float x) {
    return (omega_float_to_bits(x) & 0x80000000U) != 0;
}

static inline bool omega_issubnormal(float x) {
    uint32_t u = omega_float_to_bits(x);
    return ((u & 0x7f800000U) == 0) && ((u & 0x007fffffU) != 0);
}

static inline bool omega_iszero(float x) {
    return (omega_float_to_bits(x) & 0x7fffffffU) == 0;
}

/*
 * Required host FP environment. The reference and CPU tiers compute with the
 * host FPU, so a caller that changed the rounding mode or turned on flush to
 * zero would make both tiers agree on the same wrong answer. These FPCR bits
 * must be clear: RMode [23:22] (00 = round to nearest even), FZ [24], DN [25],
 * FZ16 [19], AHP [26] (alternative half precision: F16 conversions must use
 * IEEE binary16), and the FEAT_AFP controls FIZ [0], AH [1], NEP [2].
 * omega_numeric_reference, omega_numeric_cpu_realize,
 * omega_numeric_reference_ftz and omega_numeric_seed_bound refuse with
 * OMEGA_NUMERIC_ERR_FPENV (and compute nothing) when any is set. They never
 * change FPCR themselves.
 */
#define OMEGA_NUMERIC_FPCR_REQUIRED_CLEAR \
    ((1ULL << 0) | (1ULL << 1) | (1ULL << 2) | (1ULL << 19) | (3ULL << 22) | (1ULL << 24) | (1ULL << 25) | \
     (1ULL << 26))
uint64_t omega_numeric_read_fpcr(void);
bool omega_numeric_fpenv_ok(void);

/*
 * Semantic Reference Tier:
 * Exact IEEE 754-2008 single-precision specification, round to nearest even.
 * Subnormals are preserved bit-exactly (Flush-To-Zero is strictly rejected).
 */
float omega_ref_fadd(float a, float b);
float omega_ref_fsub(float a, float b);
float omega_ref_fmul(float a, float b);
float omega_ref_ffma(float a, float b, float c);
float omega_ref_fmin(float a, float b);
float omega_ref_fmax(float a, float b);
bool  omega_ref_fsetp_ge(float a, float b);
float omega_ref_i2f(int32_t a);
int32_t omega_ref_f2i(float a);
/*
 * E1 scalar contract (docs/numeric/E1_SCALAR_CONTRACT.md). Explicit,
 * FPCR-independent reference definitions written with integer operations on
 * the IEEE bit patterns; no host conversion or compare instruction is used.
 *
 * Integer conversions: NaN -> 0; the value is rounded to an integer in the
 * named direction, then saturated to the target range (F2U: negatives -> 0).
 * RNI is round to nearest, ties to even.
 */
int32_t  omega_ref_f2i_floor(float a);
int32_t  omega_ref_f2i_ceil(float a);
int32_t  omega_ref_f2i_rni(float a);
uint32_t omega_ref_f2u(float a);            /* truncate                       */
float    omega_ref_u2f(uint32_t a);         /* RNE                            */
/* Narrowing: RNE, overflow to signed infinity, subnormal results kept, NaN
 * stays NaN (payload not semantic). Widening is exact. */
uint16_t omega_ref_f32_to_f16(float a);
uint16_t omega_ref_f32_to_bf16(float a);
float    omega_ref_f16_to_f32(uint16_t h);
float    omega_ref_bf16_to_f32(uint16_t h);
/*
 * Compare predicate of a compare-and-select op (FSETP_SEL and the
 * FSETP_<P>_SEL ops): 1 if the predicate holds for (a, b), 0 if not, -1 if
 * op is not a compare-and-select op. Ordered predicates are false when either
 * operand is NaN, unordered (U) ones true; -0 equals +0.
 */
int omega_ref_fsetp_pred(int op, float a, float b);
/* a * b + c rounded once (RNE, subnormals kept), integer arithmetic on the bit
 * patterns (no FMADD). NaN results are the canonical quiet NaN. */
float    omega_ref_ffma_int(float a, float b, float c);
/* True when the host CPU has FEAT_BF16 (BFCVT); the CPU realization of
 * F32_TO_BF16 then uses BFCVT, otherwise it refuses (OMEGA_NUMERIC_ERR_NOT_ENCODED). */
bool     omega_numeric_cpu_has_bf16(void);

/* Correctly rounded IEEE division and square root (the AArch64 FDIV/FSQRT
 * instructions; not libm). The semantic target for omega_math_div/sqrt. */
float omega_ieee_div(float x, float y);
float omega_ieee_sqrt(float x);

/*
 * Omega-defined sequences for division and square root: correctly rounded
 * (round to nearest, ties to even, subnormal results kept, overflow to
 * infinity), computed with integer operations only (restoring long division,
 * bitwise integer square root, one shared rounding step). No MUFU seed and no
 * FP arithmetic, so the result does not depend on FPCR. Compared bit for bit
 * against omega_ieee_div / omega_ieee_sqrt. Host sweep on 2026-09-30: SQRT
 * over all 2^32 inputs and DIV over 10^9 stratified random pairs, zero
 * mismatches.
 */
float omega_math_div(float x, float y);
float omega_math_sqrt(float x);

/*
 * Omega-defined polynomial approximations for Transcendentals:
 * Zero dependency on libm or CUDA math libraries.
 * Range reduction, coefficients, and evaluation order are frozen.
 */
float omega_math_exp(float x);
float omega_math_log(float x);

/*
 * Declared-order warp reduction:
 * Reduces 32 float inputs across a warp using the frozen pairwise-tree order.
 */
float omega_warp_reduce_sum(const float warp_inputs[32]);

/*
 * Equivalence comparator:
 * Verifies that two floats match bit-for-bit, except that all quiet/signaling
 * NaNs are considered equivalent in the NaN class.
 */
bool omega_numeric_bits_equal(float a, float b);

/* ---- Op registry --------------------------------------------------------- */

typedef enum {
    OMEGA_NOP_FADD = 0,
    OMEGA_NOP_FSUB,
    OMEGA_NOP_FMUL,
    OMEGA_NOP_FFMA,        /* a*b + c, c uniform per launch (constant bank) */
    OMEGA_NOP_FSETP_SEL,   /* FSETP.GE P0, a, b ; FSEL out = P0 ? a : b      */
    OMEGA_NOP_FSEL,        /* FSETP.GE P0, a, RZ ; FSEL out = P0 ? b : a     */
    OMEGA_NOP_FMNMX_MIN,
    OMEGA_NOP_FMNMX_MAX,
    OMEGA_NOP_I2FP,        /* input bits read as int32, output FP32          */
    OMEGA_NOP_F2I,         /* input FP32, output bits are int32              */
    OMEGA_NOP_MUFU_RCP,    /* seed only: never compared bit-exactly          */
    OMEGA_NOP_MUFU_RSQ,    /* seed only: never compared bit-exactly          */
    OMEGA_NOP_LDS_STS,     /* STS, BAR.SYNC, LDS: out[i] = a[i ^ 63] per CTA  */
    OMEGA_NOP_SHFL_DOWN,   /* SHFL.DOWN by 1, clamp 0x1f                     */
    OMEGA_NOP_DIV,
    OMEGA_NOP_SQRT,
    OMEGA_NOP_EXP,
    OMEGA_NOP_LOG,
    OMEGA_NOP_REDUCE_SUM,
    /* E1 scalar contract ops (docs/numeric/E1_SCALAR_CONTRACT.md): semantic
     * reference, CPU realization and GB10 encoding (E1 WP-C; GB10 parity is
     * established by the Gate 5 chip receipt only). Compare-and-select: out = P(a, b) ? a : b, bits
     * moved unchanged (FSETP.<P> P0, a, b ; FSEL out, a, b, P0).          */
    OMEGA_NOP_FSETP_LT_SEL,
    OMEGA_NOP_FSETP_LE_SEL,
    OMEGA_NOP_FSETP_GT_SEL,
    OMEGA_NOP_FSETP_EQ_SEL,
    OMEGA_NOP_FSETP_NE_SEL,  /* ordered: false if either is NaN             */
    OMEGA_NOP_FSETP_NUM_SEL, /* neither is NaN                              */
    OMEGA_NOP_FSETP_NAN_SEL, /* either is NaN                               */
    OMEGA_NOP_FSETP_LTU_SEL,
    OMEGA_NOP_FSETP_LEU_SEL,
    OMEGA_NOP_FSETP_GTU_SEL,
    OMEGA_NOP_FSETP_GEU_SEL,
    OMEGA_NOP_FSETP_EQU_SEL,
    OMEGA_NOP_FSETP_NEU_SEL, /* IEEE !=: true if either is NaN              */
    OMEGA_NOP_F2I_FLOOR,     /* input FP32, output bits are int32           */
    OMEGA_NOP_F2I_CEIL,
    OMEGA_NOP_F2I_RNI,
    OMEGA_NOP_F2U,           /* input FP32, output bits are uint32          */
    OMEGA_NOP_I2FP_U32,      /* input bits read as uint32, output FP32      */
    OMEGA_NOP_F32_TO_F16,    /* output: F16 bits in [15:0], [31:16] zero    */
    OMEGA_NOP_F32_TO_BF16,   /* output: BF16 bits in [15:0], [31:16] zero   */
    OMEGA_NOP_F16_TO_F32,    /* input: F16 bits in [15:0], [31:16] ignored  */
    OMEGA_NOP_BF16_TO_F32,   /* input: BF16 bits in [15:0], [31:16] ignored */
    OMEGA_NOP_FFMA_V,        /* a*b + c, c read per element                 */
    OMEGA_NOP_COUNT
} OmegaNumericOp;

typedef enum {
    OMEGA_CMP_BIT_EXACT = 0,  /* FP32 bits equal; any NaN equals any NaN     */
    OMEGA_CMP_INT_EXACT,      /* raw 32-bit words equal (integer results)    */
    OMEGA_CMP_SEED_BOUND,     /* MUFU: relative-error bound only, never bits */
    OMEGA_CMP_F16_BITS,       /* [31:16] zero; [15:0] F16 bits equal, any F16 NaN equals any F16 NaN */
    OMEGA_CMP_BF16_BITS       /* [31:16] zero; [15:0] BF16 bits equal, any BF16 NaN equals any BF16 NaN */
} OmegaNumericCompare;

/* The equality a parity comparison uses for one element under mode (not
 * defined for SEED_BOUND, which returns false). */
bool omega_numeric_compare_equal(OmegaNumericCompare mode, uint32_t expect, uint32_t got);

typedef struct {
    OmegaNumericOp      op;
    const char         *name;
    int                 arity;          /* inputs used: 1 (a), 2 (a,b), 3 (a,b,c) */
    bool                gb10_encoded;   /* a real kernel exists for this op       */
    OmegaNumericCompare compare;
    const char         *reference;      /* what the expected value is             */
    const char         *not_encoded_reason;
} OmegaNumericOpInfo;

size_t omega_numeric_op_count(void);
const OmegaNumericOpInfo *omega_numeric_op_at(size_t index);
/* Finds an op by name. "LDS" and "STS" are aliases of LDS_STS, "FSETP" of
 * FSETP_SEL. Returns NULL for unknown names. */
const OmegaNumericOpInfo *omega_numeric_op_find(const char *name);
const char *omega_numeric_compare_name(OmegaNumericCompare c);

/* Ops whose GB10 kernel is a whole program from
 * src/omega_numeric_divsqrt_gb10.c (DIV, SQRT) instead of a patch into the
 * calibrated vecadd: they have no patch words, omega_numeric_build_kernel
 * refuses them, and their structural checks (omega_ds_check_args, _kernel,
 * _digest, _qmd) run inside omega_ds_gb10_run before the device is opened. */
bool omega_numeric_op_whole_kernel(OmegaNumericOp op);

/* Return codes of omega_numeric_submit_check and omega_gb10_execute_simt_op */
#define OMEGA_NUMERIC_OK               0
#define OMEGA_NUMERIC_ERR_BAD_ARGS    -1   /* unknown op or missing buffers      */
#define OMEGA_NUMERIC_ERR_NOT_ENCODED -2   /* op known, no GB10 kernel exists    */
#define OMEGA_NUMERIC_ERR_OPERANDS    -3   /* operand shape the kernel can't do  */
#define OMEGA_NUMERIC_ERR_DEVICE      -4   /* device open/alloc/submit/wait      */
#define OMEGA_NUMERIC_ERR_FPENV       -5   /* host FPCR is not RNE with FZ clear  */

#define OMEGA_NUMERIC_MAX_COUNT 65536u

/*
 * Threads per CTA of every numeric launch (QMD CTA_THREAD_DIMENSION0 and the
 * grid width in src/omega_numeric_gb10.c). LDS_STS exchanges values inside
 * one CTA, so its count must be a whole number of CTAs: a partial CTA would
 * have exited threads that never reach BAR.SYNC.
 */
#define OMEGA_NUMERIC_CTA_THREADS 64u
/* Bytes of shared memory LDS_STS touches per CTA: one 32-bit word per thread. */
#define OMEGA_NUMERIC_LDS_STS_SHARED_BYTES (OMEGA_NUMERIC_CTA_THREADS * 4u)

/*
 * Variants of the shared-memory and warp ops that have no GB10 encoding.
 * omega_numeric_submit_check refuses each with OMEGA_NUMERIC_ERR_NOT_ENCODED
 * and names it, so a caller asking for LDS.U8 or SHFL_UP learns it is
 * unencoded instead of being handed the 32-bit or SHFL.DOWN kernel.
 */
size_t omega_numeric_refused_variant_count(void);
const char *omega_numeric_refused_variant_at(size_t index);

/*
 * Everything that must be true before any device is touched. Writes a
 * one-line reason into err (if non-NULL) on refusal.
 */
int omega_numeric_submit_check(const char *op_name,
                               const float *in_a, const float *in_b, const float *in_c,
                               const float *out_res, size_t count,
                               char *err, size_t err_len);

/*
 * Kernel patch: the instruction words written at byte offset
 * OMEGA_NUMERIC_PATCH_OFFSET of the vecadd kernel (replacing its IADD3, and
 * for multi-instruction ops also its STG/EXIT). Returns the number of 128-bit
 * instructions, or a negative OMEGA_NUMERIC_ERR_* code for ops without an
 * encoding.
 */
#define OMEGA_NUMERIC_PATCH_OFFSET 0x110u
#define OMEGA_NUMERIC_PATCH_MAX    15u  /* slots 0x110..0x1f0 of the 0x200-byte kernel */
typedef struct {
    uint32_t    w[4];
    const char *text;
    const char *provenance_key;   /* NULL for vecadd baseline STG / EXIT */
} OmegaNumericPatchInsn;

int omega_numeric_patch_words(OmegaNumericOp op,
                              OmegaNumericPatchInsn out[OMEGA_NUMERIC_PATCH_MAX]);

/* Encodes the vecadd baseline and applies the patch for op. */
int omega_numeric_build_kernel(OmegaNumericOp op, uint8_t *code, size_t code_len,
                               size_t *out_len);

/*
 * Structural check of a patch against the launch it will run in. Called by
 * omega_numeric_submit_check (on a QMD built for count) and again by the GB10
 * executor on the exact QMD it submits. It decodes the words, so it catches a
 * wrong patch even when the patch table itself was edited:
 *   every patch  length 1..PATCH_MAX; a variable-latency result (write
 *                barrier set) is waited on by the very next instruction;
 *                STS/LDS/BAR/SHFL words match their one encoded form in every
 *                non-register bit (no .U8/.64/.128, no offsets, no other
 *                barrier id, no SHFL mode other than DOWN);
 *   shared use   STS before BAR.SYNC 0 before LDS; the QMD declares at least
 *                one barrier; the address is SHF.L R0 (tid.x) by exactly 2,
 *                the LDS address is that value XOR (CTA-1)*4; the CTA width
 *                in the QMD is OMEGA_NUMERIC_CTA_THREADS (a power of two);
 *                (CTA-1)*4 + 4 bytes fit in the QMD shared-memory size;
 *   REDUCE_SUM   exactly five SHFL.DOWN, deltas 16, 8, 4, 2, 1 in that
 *                order, clamp 0x1f, each followed by an FADD (no negate, no
 *                modifiers) that waits on the shuffle, adds the shuffled value
 *                to the running sum, and the last FADD writes R9, which the
 *                STG stores.
 * qmd1_words is the 96-word compute QMD. Returns OMEGA_NUMERIC_OK or
 * OMEGA_NUMERIC_ERR_OPERANDS with a one-line reason in err.
 */
int omega_numeric_check_patch(OmegaNumericOp op, const OmegaNumericPatchInsn *patch, int n,
                              const uint32_t *qmd1_words, char *err, size_t err_len);

/* Launch shape for count elements: OMEGA_NUMERIC_CTA_THREADS threads per CTA
 * and enough CTAs to cover count. Shared by submit_check and the executor. */
void omega_numeric_launch_shape(size_t count, uint32_t *threads_per_block, uint32_t *grid_width);

/* ---- Tiers and comparison ------------------------------------------------- */

/* Semantic reference tier for every op. c may be NULL except for FFMA and
 * FFMA_V (arity 3). */
int omega_numeric_reference(OmegaNumericOp op, const float *a, const float *b,
                            const float *c, float *out, size_t count);

/* CPU realization tier: the AArch64 instruction (or the Omega sequence for
 * DIV/SQRT/EXP/LOG) that realizes each op on the host. */
int omega_numeric_cpu_realize(OmegaNumericOp op, const float *a, const float *b,
                              const float *c, float *out, size_t count);

/* Elements that carry a result for op (REDUCE_SUM: lane 0 of each warp). */
bool omega_numeric_element_checked(OmegaNumericOp op, size_t index);

typedef struct {
    size_t   checked;        /* elements compared                          */
    size_t   mismatches;
    long     first_index;    /* -1 when none                               */
    uint32_t first_expect;
    uint32_t first_got;
    size_t   subnormal_expected;  /* checked elements whose expected value is subnormal */
    size_t   out_of_bound;   /* SEED_BOUND only: results outside the bound */
    size_t   bound_skipped;  /* SEED_BOUND only: elements outside the bounded domain */
} OmegaParityTrace;

/*
 * Bit parity: refuses (returns OMEGA_NUMERIC_ERR_OPERANDS) for a SEED_BOUND
 * op. That refusal is the structural guarantee behind
 * MUFU_SEED_ONLY_NOT_COMPARED.
 */
int omega_numeric_parity(OmegaNumericOp op, const float *expect, const float *got,
                         size_t count, OmegaParityTrace *trace);

/* Seed bound for MUFU: compares against the true 1/x or 1/sqrt(x) (IEEE)
 * with relative error <= 2^-20 on inputs whose result is a finite normal.
 * Refuses non-SEED_BOUND ops. */
int omega_numeric_seed_bound(OmegaNumericOp op, const float *a, const float *got,
                             size_t count, OmegaParityTrace *trace);

/* Flush-to-zero model used by the negative test: subnormal inputs and outputs
 * replaced by signed zero. */
int omega_numeric_reference_ftz(OmegaNumericOp op, const float *a, const float *b,
                                const float *c, float *out, size_t count);

/* Host magic-constant seeds (NOT a model of hardware MUFU). Used only to show
 * that an unrefined seed fails bit parity. */
float omega_numeric_host_rcp_seed(float y);

/* ---- GB10 executor (src/omega_numeric_gb10.c) ----------------------------- */

/*
 * Execution on physical GB10 silicon: runs omega_numeric_submit_check first
 * (no device is opened when it refuses), then patches the calibrated vecadd
 * kernel and dispatches it through the GPFIFO channel. FFMA's c operand must
 * be uniform (it is carried in the constant bank).
 */
int omega_gb10_execute_simt_op(const char *op_name,
                              const float *in_a,
                              const float *in_b,
                              const float *in_c,
                              float *out_res,
                              size_t count);

#endif /* OMEGA_NUMERIC_H */
