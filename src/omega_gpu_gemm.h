/* omega_gpu_gemm.h -- matrix multiply of any shape on the GB10 tensor cores (row C3 of docs/OMEGA.md).
 *
 *   C = A * B     A is M x K, B is K x N, C is M x N f32 (row-major), every M, N, K from 1 to OMEGA_GPU_GEMM_MAX_DIM
 *   BF16: A and B hold bf16 values, the product runs on HMMA.16816.F32.BF16
 *   F16:  A and B hold IEEE fp16 values, the product runs on HMMA.16816.F32 (GemmGeneral, 2026-10-27)
 *   E4M3: A and B hold FP8 e4m3 bytes, the product runs on QMMA.16832.F32.E4M3.E4M3
 *   F32_FFMA: A and B hold f32, one thread per C element runs IEEE FFMA in k order from +0 (bit-exact with a host fmaf
 *         loop); no tensor cores, no packs, dense row-major A and B only (layout 0). The independent comparator for the
 *         fp32-input bf16x3 path of omega_gpu_gemm_ex (GemmGeneral, 2026-10-27); slow by design.
 * Layouts: A row-major (M rows of K) or column-major (K rows of M); B row-major (K rows of N) or column-major (N rows of K).
 * Leading dimensions (omega_gpu_gemm_ld, _drive_ld): the distance in elements between consecutive stored rows, at least the
 * stored row length (0: dense); the span an operand covers must stay below 2^32 elements (the packs index in 32 bits).
 *
 * The arithmetic is fixed by the shape alone: every C[m][n] starts at +0 and takes one tensor-core step per 16 (BF16) or
 * 32 (E4M3) values of k, in increasing k, k past K counting as zero. The rounding of one step is the chip-measured model
 * (tests/bw_hmma_model.h { 16, 2, RZ, 2, 0 }, tests/bw_qmma_model.h { 32, 26, RZ, 2, 0 }), so a host model of the same
 * steps reproduces the chip bit for bit (tests/gpu_gemm_test.c).
 *
 * Kernels, all from Omega's own code generator (no CUDA):
 *   PACK_A, PACK_B  copy A and B, whatever their layout, into the tensor cores' fragment order (each lane's operand
 *                   registers are 16 contiguous bytes; a warp's fragment is 512 contiguous bytes), zero past the edges.
 *   MAIN            each warp computes a 16 rt x 8 ct block of C from the packed operands (LDG.128 straight into the
 *                   operand registers, two register sets so the next loads fly while the tensor cores work), a CTA holds
 *                   wm x wn warps; edge elements past M or N are not stored.
 * Packed operand layout (bytes): block (row group, k pair) at ((group * KS2 + pair) * F + f) * 512, lane L at + 16 L, where
 * a k pair is two tensor-core steps, KS2 = Kp / (2 * step), F = 2 rt (A: f = u * rt + i, step u of the pair, row tile i)
 * or ct (B: f = column tile j; registers 0, 1 step 0 and 2, 3 step 1). */
#ifndef OMEGA_GPU_GEMM_H
#define OMEGA_GPU_GEMM_H

#include <stddef.h>
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_GPU_GEMM_MAX_DIM 16384u

enum {
    OMEGA_GPU_GEMM_OK = 0,
    OMEGA_GPU_GEMM_BAD_ARGS = -1,     /* NULL pointer, a zero dimension, unknown type or layout bits, a bad configuration,
                                         a leading dimension below the stored row length or a span past 2^32 elements */
    OMEGA_GPU_GEMM_TOO_LARGE = -2,    /* a dimension past OMEGA_GPU_GEMM_MAX_DIM */
    OMEGA_GPU_GEMM_CODEGEN_FAIL = -3, /* our codegen refused a kernel */
    OMEGA_GPU_GEMM_CHIP_FAIL = -4,    /* device, allocation, launch, or an output word the chip never wrote */
    OMEGA_GPU_GEMM_UNSUPPORTED = -5   /* omega_gpu_gemm_ex: a well-formed request outside the domain (e.g. fp8 output) */
};

typedef enum { OMEGA_GPU_GEMM_BF16 = 0, OMEGA_GPU_GEMM_E4M3 = 1, OMEGA_GPU_GEMM_F16 = 2, OMEGA_GPU_GEMM_F32_FFMA = 3 } OmegaGpuGemmType;
#define OMEGA_GPU_GEMM_A_COL 1u /* A is column-major: K rows of M */
#define OMEGA_GPU_GEMM_B_COL 2u /* B is column-major: N rows of K */

typedef enum { OMEGA_GEMM_K_PACK_A = 0, OMEGA_GEMM_K_PACK_B, OMEGA_GEMM_K_MAIN, OMEGA_GEMM_K_COUNT } OmegaGemmKind;

/* rt row tiles of 16 and ct column tiles of 8 per warp (1, 2 or 4 each), wm x wn warps per CTA (1, 2 or 4 each),
 * with 4 (4 rt + 2 ct + rt ct) <= 100 (the registers of two operand sets and the accumulators; rt 2 ct 4 is the largest),
 * stall: issue cycles after each tensor-core op (1 .. 15).
 * mutant (red checks): 1 MAIN drops the second step of every k pair, 2 the packs let element 0 in past the edges (instead of zero),
 * 3 MAIN stores each column pair swapped. */
typedef struct { OmegaGpuGemmType type; uint32_t rt, ct, wm, wn, stall; int mutant; } OmegaGemmCfg;
void omega_gpu_gemm_default_cfg(OmegaGpuGemmType type, OmegaGemmCfg *cfg);
int omega_gpu_gemm_cfg_ok(const OmegaGemmCfg *cfg);

/* Argument words (constant bank 0x380). Packs: src u64, dst u64, rows, kdim, row stride, k stride (elements), KS2.
 * MAIN: packed A u64, packed B u64, C u64, M, N, KS2. */
#define OMEGA_GPU_GEMM_ARGS_WORDS 9u
int omega_gpu_gemm_codegen(const OmegaGemmCfg *cfg, OmegaGemmKind kind, BlackwellIRProgram *prog);

typedef struct {
    uint32_t Mp, Np, Kp, KS2;           /* padded sizes; KS2 k pairs (even) */
    uint64_t a_bytes, b_bytes;          /* the inputs as given */
    uint64_t ap_bytes, bp_bytes;        /* the packed operands, slack for the last prefetch included */
    uint32_t grid[OMEGA_GEMM_K_COUNT][2];
    uint32_t threads[OMEGA_GEMM_K_COUNT];
} OmegaGemmPlan;
int omega_gpu_gemm_plan(const OmegaGemmCfg *cfg, uint32_t M, uint32_t N, uint32_t K, OmegaGemmPlan *plan);

/* Runs the three kernels through a launch callback (the chip, or the warp simulator in the test). Device addresses
 * are plain numbers. */
typedef int (*OmegaGemmLaunchFn)(void *ctx, OmegaGemmKind kind, const uint32_t args[OMEGA_GPU_GEMM_ARGS_WORDS],
                                 uint32_t grid_x, uint32_t grid_y, uint32_t threads);
int omega_gpu_gemm_drive(const OmegaGemmCfg *cfg, unsigned layout, uint32_t M, uint32_t N, uint32_t K, uint64_t a_va,
                         uint64_t b_va, uint64_t ap_va, uint64_t bp_va, uint64_t c_va, OmegaGemmLaunchFn launch, void *ctx);
/* the same with leading dimensions lda, ldb in elements (0: dense) */
int omega_gpu_gemm_drive_ld(const OmegaGemmCfg *cfg, unsigned layout, uint32_t M, uint32_t N, uint32_t K, uint64_t a_va, uint64_t lda,
                            uint64_t b_va, uint64_t ldb, uint64_t ap_va, uint64_t bp_va, uint64_t c_va, OmegaGemmLaunchFn launch, void *ctx);
/* elements A and B span with those leading dimensions; OMEGA_GPU_GEMM_BAD_ARGS when a leading dimension is short or a span
 * passes 2^32 elements */
int omega_gpu_gemm_span(unsigned layout, uint32_t M, uint32_t N, uint32_t K, uint64_t lda, uint64_t ldb, uint64_t *a_elems, uint64_t *b_elems);

typedef struct {
    uint64_t pack_ns;   /* the two pack kernels, launch to completion */
    uint64_t main_ns;   /* the multiply kernel */
    uint64_t call_ns;   /* the whole call, uploads and readback included */
    uint32_t gpr[OMEGA_GEMM_K_COUNT];
} OmegaGpuGemmInfo;

/* On the chip, with the configuration set by omega_gpu_gemm_test_set_cfg (default: omega_gpu_gemm_default_cfg). */
int omega_gpu_gemm(OmegaGpuGemmType type, unsigned layout, uint32_t M, uint32_t N, uint32_t K, const void *A, const void *B,
                   float *C, OmegaGpuGemmInfo *info);
/* the same with leading dimensions (elements, 0: dense); C stays dense M x N */
int omega_gpu_gemm_ld(OmegaGpuGemmType type, unsigned layout, uint32_t M, uint32_t N, uint32_t K, const void *A, uint64_t lda,
                      const void *B, uint64_t ldb, float *C, OmegaGpuGemmInfo *info);
/* Tests and tuning: later calls of that type use cfg (NULL: back to the default); clears the kernel cache. */
int omega_gpu_gemm_test_set_cfg(OmegaGpuGemmType type, const OmegaGemmCfg *cfg);
const char *omega_gpu_gemm_rc_name(int rc);

#ifdef __cplusplus
}
#endif
#endif /* OMEGA_GPU_GEMM_H */
