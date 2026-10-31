/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_GPU_PFQ_API_H
#define OMEGA_GPU_PFQ_API_H
/*
 * omega_gpu_pfq_api: the tiled mma.sync (QMMA) FP8 path for grouped expert GEMMs in prefill (DRA-82, CUDApp #220).
 * One expert GEMM is two recorded launches of the native FP8 kernels (omega_blackwell_codegen_qg.h):
 *   quant  f32 activation rows (row stride in elements, 0 = k) -> e4m3 rows of k + f32 power-of-two scales [rows][k / 128]
 *   gemm   grouped GEMM on QMMA over 64-row tiles, weights read IN PLACE from the FP8 GEMV tensor (no second copy)
 * Rows come in the segment form of the fp8 gemm record path (OmegaPfqTile is the q36 graph's 32-byte Seg32: w_va, s_va,
 * row0, cnt, pad; cnt any size). omega_pfq_tiles cuts each segment into tiles of at most OMEGA_PFQ_BM rows.
 * Tile row r: activation row rowidx[row0 + r] (rowidx_va 0: row0 + r), output y row ymap[row0 + r] (ymap_va 0: row0 + r),
 * f32 rows of n. The gemm reads rowidx for every row of a tile, cnt or not: a row map needs OMEGA_PFQ_MAP_PAD entries
 * past its last row, each naming a row the quantiser wrote (0 does). ymap is read only below cnt.
 * x64 1: the activations carry an extra 2^64 (the q36 graph's x' and act rows) and y is the true product. Exact: the
 * quantiser's scales are powers of two, so q(2^64 x) = q(x) and s(2^64 x) = 2^64 s(x) unless amax(x) < 2^-164.
 * Limit (x64): the fold sa * sw overflows when amax(x) * S >= 2^16 (S the checkpoint weight scale).
 * Arithmetic: the qg header's (QMMA chunks accumulated in f32, an exact power-of-two scale fold). NOT bit-identical to the
 * FP8 GEMV / fp8gemm FFMA path (activations quantised to e4m3, another summation order): a consumer is gated on output
 * tolerance, not bit-identity (q36: AIEN_OMEGA_Q36_PF_QMMA=1, off by default).
 * Kernel cache kinds (omega_gpu_prefill.h OmegaPfKey): OMEGA_PFQ_KIND_QUANT, OMEGA_PFQ_KIND_GEMM.
 * Shape: 4 row tiles of 16, 8 warps; QMMA read distance AIEN_OMEGA_PFQ_WAIT (0 .. 120, default 45).
 * AIEN_OMEGA_PFQ_MUTANT (tests and chip gates only; read once, omega_gpu_pfq_test_set_mutant overrides):
 *   1 the tiler starts tile p >= 1 of a segment one row early, 2 the gemm never advances the weight scale pointer (the
 *   wrong-scale-block mutant), 3 gemm row tile i reads tile i^1, 4 the quantiser uses c = 0.
 * Locking: the *_record_locked calls run with the session lock held (as omega_gpu_fp8gemm_record); prepare takes it.
 */
#include <stdint.h>
#include "omega_blackwell_codegen_qg.h"

#ifdef __cplusplus
extern "C" {
#endif
#define OMEGA_PFQ_BM 64u
#define OMEGA_PFQ_MAP_PAD 64u
#define OMEGA_PFQ_KIND_QUANT 0x60u
#define OMEGA_PFQ_KIND_GEMM 0x61u
typedef struct { uint64_t w_va, s_va; uint32_t row0, cnt; uint64_t pad; } OmegaPfqTile;

/* tiles needed for rows spread over segs segments: segs + rows / OMEGA_PFQ_BM (an upper bound) */
uint32_t omega_pfq_tile_cap(uint32_t rows, uint32_t segs);
/* cuts segs[0 .. nsegs) into tiles; returns the tile count, 0 when cap is too small or an argument is NULL
 * (a table of empty segments also returns 0: nothing to launch) */
uint32_t omega_pfq_tiles(const OmegaPfqTile *segs, uint32_t nsegs, OmegaPfqTile *tiles, uint32_t cap);

typedef struct { uint32_t k, rows, stride; uint64_t x_va, q_va, s_va; } OmegaPfqQuant;
typedef struct {
    uint32_t n, k, x64, ntiles;
    uint64_t tiles_va, rowidx_va, xq_va, xs_va, ymap_va, y_va;
} OmegaPfqGemm;
/* the launch a record call makes (host-checkable: tests run these programs, arguments and grids in the warp simulator) */
typedef struct {
    uint32_t threads, grid_x, grid_y, shared, nargs;
    uint32_t args[OMEGA_QG_ARGS_WORDS_YMAP];
} OmegaPfqLaunch;
/* OMEGA_GPU_PF_OK, or BAD_ARGS / TOO_LARGE (omega_gpu_prefill.h); rows 0 / ntiles 0 plan nothing (grid 0) */
int omega_pfq_quant_plan(const OmegaPfqQuant *q, OmegaPqSpec *spec, OmegaPfqLaunch *L);
int omega_pfq_gemm_plan(const OmegaPfqGemm *c, OmegaQgSpec *spec, OmegaPfqLaunch *L);

struct OmegaGpuCmdList;
int omega_gpu_pfq_quant_record_locked(struct OmegaGpuCmdList *cl, const OmegaPfqQuant *q);
int omega_gpu_pfq_gemm_record_locked(struct OmegaGpuCmdList *cl, const OmegaPfqGemm *c);
/* generates and uploads the kernels a later record of this shape uses (a record never has to generate code) */
int omega_gpu_pfq_quant_prepare(uint32_t k, uint32_t stride);
int omega_gpu_pfq_gemm_prepare(uint32_t n, uint32_t k, uint32_t x64, int indexed, int ymap);

int omega_gpu_pfq_mutant(void);
void omega_gpu_pfq_test_set_mutant(int m);
#ifdef __cplusplus
}
#endif
#endif
