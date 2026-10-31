/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_GPU_GDNC_API_H
#define OMEGA_GPU_GDNC_API_H
/*
 * omega_gpu_gdnc_api (DRA-84, 2026-10-31): the chunked tensor-core gated-delta-rule prefill scan
 * (omega_blackwell_codegen_gdnc.h) as one recorded launch per GDN layer over a chunk of T tokens, on the layer's resident
 * parameter block, conv tail and f32 state of an OmegaGpuGdn (omega_gpu_gdn_api.h). Tolerance-equal to the sequential
 * multi-token kernel (tests/gpu_gdnc_test.c), not bit-equal: the HMMA operands are bf16.
 *
 *   int omega_gpu_gdnc_prepare(g);   builds and caches the program for g's ratio (the record path refuses a cache miss
 *                                    that would evict); BAD_ARGS for a bf16 state (AIEN_OMEGA_GDN_STATE_BF16=1)
 *   int omega_gpu_gdnc_record(cl, g, layer, &io, snap, &written);   appended to cl; the caller holds the session lock.
 *        Token t reads the bf16 qkv | z row at io.qkvz_va + t * xstride (z at conv * 2 bytes), the f32 [vh][2] a|b row at
 *        io.ab_va + t * abstride (abstride 0: every token reads row 0) and writes its bf16 [vdim] row at
 *        io.out_va + t * ostride. The kernel multiplies the inputs by in_scale and the outputs by out_scale.
 *        snap (NULL = none, DRA-86): snapshot k (0 .. count-1) of GDN layer `layer` is the f32 state | conv tail (the layer
 *        layout, state_bytes + tail_bytes) after token (k + 1) * 64 * every_chunks of the chunk, at
 *        va + layer * layer_stride + k * snap_stride. count = omega_gdnc_snap_count(T, every_chunks, cap): only whole
 *        snapshot spans, so a ragged last sub-chunk never lands one at a wrong position. *written = that count.
 *   void omega_gpu_gdnc_test_set_mutant(int);   TEST ONLY: 0 or OMEGA_GDNC_MUTANT_DROP_DECAY (9); clears the cache
 *
 * Host planning (no device): omega_gdnc_plan fills the 25 argument words and the launch shape from a request; the record
 * path launches exactly that plan, so the warp-simulator test runs the same words.
 */
#include "omega_blackwell_codegen_gdnc.h"
#include "omega_gpu_gdn_api.h"
#include <stdint.h>

/* the per-chunk rows of one GDN layer */
typedef struct {
    uint32_t tokens;
    uint64_t qkvz_va; uint32_t xstride;   /* bf16 rows, bytes apart (16-byte aligned) */
    uint64_t ab_va; uint32_t abstride;    /* f32 rows, bytes apart (8-byte aligned; 0 = every token reads row 0) */
    uint64_t out_va; uint32_t ostride;    /* bf16 rows, bytes apart (16-byte aligned) */
    float eps_l2, eps_norm, in_scale, out_scale;
} OmegaGdncRows;

/* where the snapshots of every GDN layer go (DRA-86 prefix cache) */
typedef struct {
    uint64_t va;                  /* 0 = none */
    uint32_t every_chunks;        /* 64-token chunks between snapshots (>= 1) */
    uint32_t count;               /* the slots per layer (the cap) */
    uint64_t layer_stride_bytes;  /* GDN layer l's slots start at va + l * layer_stride_bytes */
    uint32_t snap_stride_bytes;   /* between slots; >= state_bytes + tail_bytes, a multiple of 16 */
} OmegaGdncSnapOut;

/* one fully resolved launch request (the layer's blocks included) */
typedef struct {
    uint32_t ratio;               /* OmegaGdnSpec.ratio */
    OmegaGdncRows io;
    uint64_t par_va, tail_va, state_va;
    uint64_t snap_va; uint32_t snap_every4, snap_count, snap_stride; /* snap_count 0 = none */
} OmegaGdncReq;

typedef struct {
    uint32_t threads, grid_x, grid_y, shared_bytes, n_args;
    uint32_t args[OMEGA_GDNC_ARGS_WORDS]; /* the constant bank from OMEGA_GDNC_ARG_QKVZ (0x380) on */
} OmegaGdncLaunch;

/* Host: the launch for r. OMEGA_GPU_GDN_OK, or BAD_ARGS (unsupported ratio, zero tokens or address, misaligned rows,
 * non-finite scalars, snapshots without an address or with a stride below the state + tail of the ratio). */
int omega_gdnc_plan(const OmegaGdncReq *r, OmegaGdncLaunch *L);
/* Host: the snapshots a T-token chunk writes at one every `every_chunks` 64-token chunks, at most cap: min(cap,
 * floor(T / (64 every_chunks))); 0 when every_chunks or cap is 0. */
uint32_t omega_gdnc_snap_count(uint32_t tokens, uint32_t every_chunks, uint32_t cap);
/* Host: r for GDN layer `layer` of a state view (omega_gpu_gdn_state_view) plus the rows and snapshots. BAD_ARGS for a
 * layer out of range, a bf16 state or an unusable snapshot layout. */
int omega_gdnc_req(const OmegaGpuGdnStateView *v, uint32_t layer, const OmegaGdncRows *io, const OmegaGdncSnapOut *snap,
                   OmegaGdncReq *r);

struct OmegaGpuCmdList;
int omega_gpu_gdnc_prepare(OmegaGpuGdn *g);
int omega_gpu_gdnc_record(struct OmegaGpuCmdList *cl, OmegaGpuGdn *g, uint32_t layer, const OmegaGdncRows *io,
                          const OmegaGdncSnapOut *snap, uint32_t *written);
/* launches recorded since process start (the default-off proofs read it) */
uint64_t omega_gpu_gdnc_records(void);
void omega_gpu_gdnc_test_set_mutant(int mutant);

#endif
