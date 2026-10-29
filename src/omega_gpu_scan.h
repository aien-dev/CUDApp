/* omega_gpu_scan.h -- reductions and prefix sums of any length on the GB10 (row C3 of docs/OMEGA.md).
 *
 * The sum, the maximum and the minimum of n float values, and the inclusive prefix sum (the running total
 * y[i] = x[0] + ... + x[i]), for every n from 1 to 2^28. The order of the additions is frozen: it depends only on n
 * and the element index, never on the grid, the scheduling or the chip, so the chip result equals the host
 * reference below bit for bit.
 *
 * Reductions (the order of docs/numeric/E1_REDUCTION_CONTRACT.md, omega_numeric_reduce.h): one level turns len values
 * into ceil(len / 32) values; value j is the 32-lane tree over values [32j, 32j + 32) with lane i combining lane i + d
 * for d = 16, 8, 4, 2, 1 (lower lane on the left), positions past len filled with the identity (-0.0 for the sum, a
 * quiet NaN for the maximum and minimum, which those operations ignore). Levels repeat until one value is left; at
 * least one level runs.
 *
 * Prefix sum, same tile structure:
 *   level scan  inside each tile of 32 (positions past len filled with -0.0), for d = 1, 2, 4, 8, 16 in that order,
 *               every position i >= d takes v[i - d] + v[i] (both from before the step); the tile total T[j] is the
 *               value at position 31.
 *   carry       when there is more than one tile, the totals T are prefix-summed by the same definition (recursively)
 *               into S, and every element of tile j >= 1 becomes S[j - 1] + (its tile value).
 * Inputs are any float values. Results with no NaN are bit-exact; a NaN result is a NaN (the chip writes one canonical
 * NaN, the host may keep another payload).
 *
 * On the chip each level is one launch of a kernel from Omega's own codegen (no CUDA): the reduction tree and the
 * tile scan are warp shuffles (SHFL.DOWN; the scan holds each tile in reversed lane order so a shuffle down reads the
 * lower element), the carry is a separate pass. omega_gpu_scan_drive_* run the levels through a launch callback, so
 * the same driver runs on the chip (omega_gpu_reduce, omega_gpu_prefix_sum) and in the warp simulator
 * (tests/gpu_scan_test.c). */
#ifndef OMEGA_GPU_SCAN_H
#define OMEGA_GPU_SCAN_H

#include <stddef.h>
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_GPU_SCAN_MAX_N ((size_t)1 << 28)
#define OMEGA_GPU_SCAN_THREADS 256u /* 8 warps per CTA; each warp walks tiles gw, gw + warps, ... */
#define OMEGA_GPU_SCAN_MAX_CTAS 1024u

enum {
    OMEGA_GPU_SCAN_OK = 0,
    OMEGA_GPU_SCAN_BAD_ARGS = -1,     /* NULL pointer, n == 0, unknown op */
    OMEGA_GPU_SCAN_TOO_LARGE = -2,    /* n > OMEGA_GPU_SCAN_MAX_N */
    OMEGA_GPU_SCAN_CODEGEN_FAIL = -3, /* our codegen refused a kernel */
    OMEGA_GPU_SCAN_CHIP_FAIL = -4     /* device, allocation, launch, or an output word the chip never wrote */
};

typedef enum { OMEGA_GPU_RED_SUM = 0, OMEGA_GPU_RED_MAX = 1, OMEGA_GPU_RED_MIN = 2 } OmegaGpuRedOp;

/* The kernels. Argument words at constant bank 0x380 (OMEGA_GPU_SCAN_A_*). */
typedef enum {
    OMEGA_SCAN_K_RED_SUM = 0, OMEGA_SCAN_K_RED_MAX, OMEGA_SCAN_K_RED_MIN, /* one reduction level: in -> out[tile] */
    OMEGA_SCAN_K_TILE,                                                  /* one scan level: in -> out (may alias), tot[tile] */
    OMEGA_SCAN_K_ADD,                                                   /* carry: out[e] = S[tile - 1] + out[e], tiles >= 1 */
    OMEGA_SCAN_K_COUNT
} OmegaScanKind;
#define OMEGA_GPU_SCAN_A_IN 0x380u     /* u64: level input (K_ADD: the level being carried into) */
#define OMEGA_GPU_SCAN_A_OUT 0x388u    /* u64: level output (K_ADD: S, the scanned totals) */
#define OMEGA_GPU_SCAN_A_TOT 0x390u    /* u64: tile totals (K_TILE only) */
#define OMEGA_GPU_SCAN_A_LEN 0x398u    /* u32: values in this level */
#define OMEGA_GPU_SCAN_A_TILES 0x39cu  /* u32: ceil(len / 32) */
#define OMEGA_GPU_SCAN_A_WARPS 0x3a0u  /* u32: warps in the grid (the tile stride) */
#define OMEGA_GPU_SCAN_ARGS_WORDS 9u

/* Mutants for the red checks (each must change some result): 1 the reduction tree skips d = 1, 2 the tile scan skips
 * d = 16, 3 the carry adds S[tile] instead of S[tile - 1]. 0 = the real kernels. */
typedef struct { OmegaScanKind kind; int mutant; } OmegaScanSpec;

/* Builds the IR program of one kernel and allocates its registers; 0 or -1. */
int omega_gpu_scan_codegen(const OmegaScanSpec *spec, BlackwellIRProgram *prog);

/* Host references, the frozen order above. 0, or -1 for bad arguments or out of memory. */
int omega_gpu_reduce_reference(OmegaGpuRedOp op, const float *x, size_t n, float *out);
int omega_gpu_prefix_sum_reference(const float *x, size_t n, float *y);

/* The level driver. Device addresses are plain numbers (host pointers in the simulator). launch runs one kernel over
 * grid_x CTAs of OMEGA_GPU_SCAN_THREADS threads with the given argument words and returns 0 when it completed. */
typedef int (*OmegaScanLaunchFn)(void *ctx, OmegaScanKind kind, const uint32_t args[OMEGA_GPU_SCAN_ARGS_WORDS], uint32_t grid_x);
/* Scratch words a call needs beyond its input and output. */
size_t omega_gpu_scan_reduce_scratch_words(size_t n);
size_t omega_gpu_scan_prefix_scratch_words(size_t n);
/* Reduction: the result lands in the scratch word the function returns through *out_va. */
int omega_gpu_scan_drive_reduce(OmegaGpuRedOp op, size_t n, uint64_t in_va, uint64_t scratch_va, uint64_t *out_va,
                                OmegaScanLaunchFn launch, void *ctx, uint32_t *launches);
/* Prefix sum of in into out (out may equal in). */
int omega_gpu_scan_drive_prefix(size_t n, uint64_t in_va, uint64_t out_va, uint64_t scratch_va,
                                OmegaScanLaunchFn launch, void *ctx, uint32_t *launches);
/* CTAs for a level of len values. */
uint32_t omega_gpu_scan_grid(size_t len);

typedef struct {
    uint32_t launches;      /* kernels run */
    uint64_t elapsed_ns;    /* sum of launch-to-completion over the launches */
    uint64_t call_ns;       /* the whole call, uploads and readback included */
} OmegaGpuScanInfo;

/* On the chip. x is read, out / y written; nothing else is kept between calls except the kernels. */
int omega_gpu_reduce(OmegaGpuRedOp op, const float *x, size_t n, float *out, OmegaGpuScanInfo *info);
int omega_gpu_prefix_sum(const float *x, size_t n, float *y, OmegaGpuScanInfo *info);
/* Red checks: every later call builds its kernels with this mutant (clears the kernel cache). */
void omega_gpu_scan_test_set_mutant(int mutant);
const char *omega_gpu_scan_rc_name(int rc);

#ifdef __cplusplus
}
#endif
#endif /* OMEGA_GPU_SCAN_H */
