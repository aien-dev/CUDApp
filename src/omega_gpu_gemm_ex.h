/* omega_gpu_gemm_ex.h -- the general matrix multiply (GemmGeneral, 2026-10-27): one entry with the shape contract of
 * cblas_?gemm (RowMajor) plus strided batches and fp8 scales, built on the C3 kernels of omega_gpu_gemm.h.
 *
 *   C_b = alpha * s * op(A_b) op(B_b) + beta * C_b,  b = 0 .. batch-1        (row-major, cblas RowMajor semantics)
 *
 * Shapes: any m, n, k >= 1 (up to OMEGA_GPU_GEMM_EX_MAX_DIM), any batch >= 1. op(A) is m x k: stored m rows of k (OP_N,
 * lda >= k) or k rows of m (OP_T, lda >= m). op(B) is k x n: stored k rows of n (OP_N, ldb >= n) or n rows of k (OP_T,
 * ldb >= k). C is m rows of n, ldc >= n. A leading dimension of 0 means dense. Batch b reads A + b stride_a, B + b stride_b,
 * C + b stride_c (elements); stride 0 for A or B reuses one matrix in every batch; C batches must not overlap each other and
 * C must not overlap A or B (BAD_ARGS). beta == 0: C is written without being read. alpha == 0: A and B are not read.
 *
 * Types: A and B each f32, bf16, f16, e4m3 or e5m2; C f32, bf16 or f16 (round to nearest even; fp8 output: UNSUPPORTED).
 * How the product runs (all accumulation is fp32 on the tensor cores, k parts summed on the host in double, epilogue
 * alpha * s * acc + beta * C in double, then one rounding to the C type):
 *   bf16 x bf16, f16 x f16, e4m3 x e4m3   native: HMMA.16816.F32.BF16, HMMA.16816.F32, QMMA.16832.F32.E4M3. With a single k
 *                                         part, alpha 1, beta 0, no scales and C f32 the result is bit for bit the C3
 *                                         kernel's (tests/bw_hmma_model.h, bw_qmma_model.h).
 *   pairs within {bf16, e4m3, e5m2}       both widened exactly to bf16, HMMA.BF16
 *   pairs within {f16, e4m3, e5m2}        both widened exactly to f16, HMMA (f16)
 *   anything with f32, and bf16 x f16     fp32-input GEMM via bf16x3: each value split exactly (truncation) into three bf16
 *                                         planes, the nine plane products concatenated along k on HMMA.BF16, smallest first.
 *                                         Not exact fp32: accuracy is stated against a double reference in
 *                                         tests/gpu_gemm_ex_test.c. Parts of a value below 2^-133 are dropped.
 *                                         compute OMEGA_GEMM_EX_F32_FAST_BF16: one bf16 plane (round to nearest even).
 *   compute OMEGA_GEMM_EX_F32_FFMA_REF    the reference comparator for any input types: every input widened to f32, one
 *                                         FFMA per k in increasing k from +0 (C3 type F32_FFMA); slow by design.
 * Non-finite inputs (inf, NaN; e4m3 0x7f/0xff): the outputs in their row of op(A) or column of op(B) are recomputed on the
 * host in double over the whole k, so inf and NaN propagate as IEEE arithmetic says.
 *
 * Scales: SCALE_TENSOR multiplies by scale_a[0] scale_b[0] (per batch at + b scale_stride_a / _b). SCALE_BLOCK: op(A) is cut
 * into block_m x block_k blocks, scale_a row-major [ceil(m/block_m)][ceil(k/block_k)]; op(B) into block_k x block_n blocks,
 * scale_b [ceil(k/block_k)][ceil(n/block_n)]; the k parts break at block_k. Scales must be finite. Any input type may be scaled.
 *
 * Migration-pending C (GpuStack PLAN rules): build switch CUDAPP_MIGRATION_PENDING_GEMM_EX (mk/gemm-ex.mk, default 1: new
 * capability, no served caller); named successor src/omega_gpu_gemm_ex.om. */
#ifndef OMEGA_GPU_GEMM_EX_H
#define OMEGA_GPU_GEMM_EX_H

#include <stdint.h>
#include "omega_gpu_gemm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_GPU_GEMM_EX_MAX_DIM 0x7fffffffull
#define OMEGA_GPU_GEMM_EX_MAX_BATCH 0x7fffffffull

typedef enum { OMEGA_GEMM_OP_N = 0, OMEGA_GEMM_OP_T = 1 } OmegaGemmOp;
typedef enum {
    OMEGA_GEMM_DT_F32 = 0, OMEGA_GEMM_DT_BF16 = 1, OMEGA_GEMM_DT_F16 = 2, OMEGA_GEMM_DT_E4M3 = 3, OMEGA_GEMM_DT_E5M2 = 4
} OmegaGemmDtype;
typedef enum { OMEGA_GEMM_EX_DEFAULT = 0, OMEGA_GEMM_EX_F32_FAST_BF16 = 1, OMEGA_GEMM_EX_F32_FFMA_REF = 2 } OmegaGemmExCompute;
typedef enum { OMEGA_GEMM_SCALE_NONE = 0, OMEGA_GEMM_SCALE_TENSOR = 1, OMEGA_GEMM_SCALE_BLOCK = 2 } OmegaGemmScale;
/* the route a call took (OmegaGemmExInfo.route) */
typedef enum {
    OMEGA_GEMM_ROUTE_NONE = 0, OMEGA_GEMM_ROUTE_NATIVE, OMEGA_GEMM_ROUTE_WIDEN, OMEGA_GEMM_ROUTE_BF16X3,
    OMEGA_GEMM_ROUTE_FAST_BF16, OMEGA_GEMM_ROUTE_FFMA
} OmegaGemmRoute;

typedef struct {
    uint32_t route;        /* OmegaGemmRoute of the product (NONE when alpha == 0) */
    uint32_t core_type;    /* OmegaGpuGemmType the kernels ran */
    uint64_t core_calls;   /* C3 calls made */
    uint64_t host_fixed;   /* outputs recomputed on the host for non-finite inputs */
} OmegaGemmExInfo;

typedef struct {
    OmegaGemmOp op_a, op_b;
    uint64_t m, n, k, batch;            /* batch 0 counts as 1 */
    OmegaGemmDtype a_type, b_type, c_type;
    const void *a; uint64_t lda, stride_a;
    const void *b; uint64_t ldb, stride_b;
    void *c; uint64_t ldc, stride_c;
    float alpha, beta;
    OmegaGemmExCompute compute;
    OmegaGemmScale scale;
    const float *scale_a, *scale_b;
    uint64_t scale_stride_a, scale_stride_b; /* per batch, elements (0: shared) */
    uint32_t block_m, block_n, block_k;      /* SCALE_BLOCK only */
    OmegaGemmExInfo *info;                   /* optional */
} OmegaGemmExArgs;

/* OMEGA_GPU_GEMM_OK, _BAD_ARGS (malformed), _TOO_LARGE (a dimension or batch past the maxima), _UNSUPPORTED (fp8 C),
 * _CODEGEN_FAIL, _CHIP_FAIL (from the kernels or a host allocation) */
int omega_gpu_gemm_ex(const OmegaGemmExArgs *args);

/* The block-scaled fp8 weight product the fp8 GEMV and grouped GEMM fall back to for shapes their tuned kernels refuse
 * (omega_gpu_fp8_api.h, omega_gpu_fp8gemm_api.h):
 *   y[r][j] = sum_c x[r][c] * e4m3(w[j][c]) * scale[j / block][c / block] * scale_mul,  r < m, j < n, c < k
 * w: n rows of k e4m3 bytes; scale: ceil(n/block) rows of ceil(k/block) f32; x: m rows of k, f32 (x32) or bf16 (x16), exactly
 * one non-NULL; y: m rows of n f32 at ldy (0: n). Any m, n, k, block >= 1. Each weight is dequantized to f32 with one rounding
 * (exact when the scale has <= 20 significant bits, as the fp8 tensors' bf16 scales do), then the product runs as
 * omega_gpu_gemm_ex f32 x f32 (the bf16x3 route, its stated accuracy). Every output element's sum is independent of m, so
 * a row computed in a batch of m equals the same row computed alone. Returns the omega_gpu_gemm_ex codes; UNSUPPORTED
 * when the build switch is off. */
int omega_gpu_gemm_ex_fp8_block_rows(const uint8_t *w, const float *scale, double scale_mul, uint32_t n, uint32_t k, uint32_t block,
                                     const float *x32, const uint16_t *x16, uint32_t m, float *y, uint64_t ldy, OmegaGemmExInfo *info);

/* The kernels the product runs on: default the chip (omega_gpu_gemm_ld). Tests install a host model or the warp simulator;
 * fn NULL restores the chip. C is dense M x N f32. */
typedef int (*OmegaGemmExCoreFn)(void *ctx, OmegaGpuGemmType type, unsigned layout, uint32_t M, uint32_t N, uint32_t K,
                                 const void *A, uint64_t lda, const void *B, uint64_t ldb, float *C);
void omega_gpu_gemm_ex_set_backend(OmegaGemmExCoreFn fn, void *ctx);
/* Red checks: 1 the last k element of every product is dropped, 2 op_b is read as OP_N, 3 the block scale of the next k
 * block is used, 4 beta is ignored, 5 omega_gpu_gemm_ex_fp8_block_rows dequantizes with the next k block's scale; 0 off. */
void omega_gpu_gemm_ex_test_set_mutant(int mutant);

#ifdef __cplusplus
}
#endif
#endif /* OMEGA_GPU_GEMM_EX_H */
