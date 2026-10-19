#ifndef OMEGA_GPU_FP8_API_H
#define OMEGA_GPU_FP8_API_H
/*
 * omega_gpu_fp8_api: native (no CUDA) FP8 e4m3 block-scaled GEMV for the OWN-30B lane
 * (Cut 6: m = 1 decode GEMV; Cut 5: grouped experts, one launch for the selected experts).
 * New file on the shared session (omega_gpu_session.h); no existing API file changes.
 *
 * ============ C signatures and semantics (for the Rust binding) ============
 *
 *   typedef struct OmegaGpuFp8Tensor OmegaGpuFp8Tensor;   // opaque, device resident
 *
 *   int omega_gpu_fp8_tensor_upload(uint32_t experts, uint32_t n, uint32_t k,
 *                                   const uint8_t *w, const uint16_t *scale_inv_bf16,
 *                                   OmegaGpuFp8Tensor **out);
 *     w:  experts * n * k bytes, e4m3, row-major [expert][n][k] (n output rows, k input columns):
 *         exactly the checkpoint's `weight` for one projection, experts concatenated (the sc
 *         capsule GATE_UP [128][1536][2048] and DOWN [128][2048][768] layouts; attention
 *         projections have experts = 1, e.g. q_proj [4096][2048]).
 *     scale_inv_bf16: experts * (n/128) * (k/128) bf16, [expert][n/128][k/128] (`weight_scale_inv`).
 *     Requires n % 128 == 0, k % 128 == 0, 128 <= k <= 32768 (omega#541), experts >= 1. Refuses any e4m3 NaN
 *     code (0x7F, 0xFF) in w with OMEGA_GPU_FP8_NAN_WEIGHT (nothing uploaded), a NaN/inf scale with
 *     BAD_ARGS. Weights are copied byte for byte (no reorder); scales are stored as f32 * 2^56.
 *     Opens the device on first use. Resident until omega_gpu_fp8_tensor_free.
 *
 *   int omega_gpu_fp8_gemv_f32 (const OmegaGpuFp8Tensor *t, uint32_t expert, const float    *x, float *y, OmegaGpuFp8Info *info);
 *   int omega_gpu_fp8_gemv_bf16(const OmegaGpuFp8Tensor *t, uint32_t expert, const uint16_t *x, float *y, OmegaGpuFp8Info *info);
 *     y[r] = sum_c W[expert][r][c] * x[c] * scale[expert][r/128][c/128], r < n, c < k. f32 accumulate.
 *     x: k values (f32, or bf16 bits), y: n f32. One launch. info may be NULL.
 *
 *   int omega_gpu_fp8_gemv_grouped_f32(const OmegaGpuFp8Tensor *t, uint32_t groups, const uint32_t *expert_ids,
 *                                      const float *x, uint32_t x_rows, float *y, OmegaGpuFp8Info *info);
 *     One launch for `groups` experts (1..OMEGA_GPU_FP8_MAX_GROUPS, each id < experts, repeats allowed):
 *     y[g][r] = sum_c W[expert_ids[g]][r][c] * x[g'][c] * scale[...], with g' = 0 when x_rows == 1
 *     (one activation row shared by every group: the gate+up projections of 8 experts) or g' = g when
 *     x_rows == groups (one row per group: the down projections). x: x_rows * k f32, y: groups * n f32.
 *     The m = 1 call above is this call with groups = 1.
 *
 *   int omega_gpu_fp8_prepare(uint32_t k);        build and pin the kernel for width k now (serving).
 *   void omega_gpu_fp8_tensor_free(OmegaGpuFp8Tensor *t);
 *   void omega_gpu_fp8_tensor_shape(const OmegaGpuFp8Tensor *t, uint32_t *experts, uint32_t *n, uint32_t *k);
 *   const char *omega_gpu_fp8_rc_name(int rc);     const char *omega_gpu_fp8_last_error(void);
 *
 * Return codes: 0 OK; negative OMEGA_GPU_FP8_* below. An uncertain completion latches the process
 * (every later call of every API returns CHIP_FAIL), as the other APIs do.
 *
 * Every output word is poisoned before the launch and the call refuses a result that still holds
 * poison (CHIP_FAIL "output poison survived"). No host oracle runs inside the call (speed); the test
 * holds the reference (omega_fp8_gemv_oracle below).
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "omega_blackwell_codegen_moe.h" /* OmegaFp8GemvSpec (depth query) */

#ifdef __cplusplus
extern "C" {
#endif

enum {
    OMEGA_GPU_FP8_OK = 0,
    OMEGA_GPU_FP8_BAD_ARGS = -1,     /* NULL, zero or misaligned dimension, bad expert id, out of host memory */
    OMEGA_GPU_FP8_TOO_LARGE = -2,    /* k past OMEGA_FP8_MAX_K, groups past the table, sealed cache miss */
    OMEGA_GPU_FP8_CODEGEN_FAIL = -3, /* our codegen refused the shape */
    OMEGA_GPU_FP8_CHIP_FAIL = -4,    /* device open, alloc, submit, marker or output-written check failed */
    OMEGA_GPU_FP8_NAN_WEIGHT = -5    /* an e4m3 NaN code in the weights (refused at upload) */
};

#define OMEGA_GPU_FP8_MAX_GROUPS 64u
#define OMEGA_GPU_FP8_MAX_CTAS 256u /* default CTA budget per group for a single projection; the elementwise and attention rows paths proved 1024 */
#define OMEGA_GPU_FP8_GROUPED_CTAS 6u /* CTAs per group for a grouped launch (32 warps each): measured best for 8 experts */

typedef struct {
    uint32_t experts, n, k, groups;
    uint32_t lane_bytes, rows_per_warp, warps_per_cta; /* kernel shape chosen for the call */
    uint32_t depth;             /* 0: step-by-step kernel; D: deep kernel, D K steps of loads in flight, x' in shared memory */
    uint32_t grid_x, grid_y, steps;
    uint64_t weight_bytes;      /* e4m3 bytes the launch reads (groups * n * k) */
    uint64_t scale_bytes;       /* f32 scale words it reads */
    uint64_t elapsed_ns;        /* chip submit-to-second-marker */
    uint64_t call_ns;           /* whole API call */
    uint32_t completion_marker;
    bool kernel_cache_hit;
    char target_chip[64];
    uint32_t sm_architecture;
} OmegaGpuFp8Info;

typedef struct OmegaGpuFp8Tensor OmegaGpuFp8Tensor;

int omega_gpu_fp8_tensor_upload(uint32_t experts, uint32_t n, uint32_t k, const uint8_t *w,
                                const uint16_t *scale_inv_bf16, OmegaGpuFp8Tensor **out);
void omega_gpu_fp8_tensor_free(OmegaGpuFp8Tensor *t);
void omega_gpu_fp8_tensor_shape(const OmegaGpuFp8Tensor *t, uint32_t *experts, uint32_t *n, uint32_t *k);

int omega_gpu_fp8_gemv_f32(const OmegaGpuFp8Tensor *t, uint32_t expert, const float *x, float *y, OmegaGpuFp8Info *info);
int omega_gpu_fp8_gemv_bf16(const OmegaGpuFp8Tensor *t, uint32_t expert, const uint16_t *x, float *y, OmegaGpuFp8Info *info);
int omega_gpu_fp8_gemv_grouped_f32(const OmegaGpuFp8Tensor *t, uint32_t groups, const uint32_t *expert_ids,
                                   const float *x, uint32_t x_rows, float *y, OmegaGpuFp8Info *info);
/* Device-resident inputs (OWN-30B router, 2026-10-13): the group table ({w_va, s_va} per group, `groups` entries) and
 * the single x' row (f32, k values, already * 2^64) are read from tab_va / x_va, which another launch (the router)
 * wrote; no host copy of either. y and the poison check as above. */
int omega_gpu_fp8_gemv_grouped_dev(const OmegaGpuFp8Tensor *t, uint32_t groups, uint64_t tab_va, uint64_t x_va, float *y, OmegaGpuFp8Info *info);
/* Record path (OWN-30B W-L2a, decode graph; omega_gpu_cmdlist.h): the same launch as grouped_dev, appended to cl instead
 * of submitted. tab_va: `groups` {w_va, s_va} entries (omega_gpu_fp8_tensor_entry, or the router's table); x_va: x_rows
 * rows of k f32 already * 2^64 (1 row shared, or one per group); y_va: groups * n f32, written when the list runs.
 * The caller holds omega_gpu_session_lock() (do not call another API's non-record entry point while holding it).
 * Nothing is staged, poisoned or read back: the list's per-launch semaphore check and the caller's end-of-token checks
 * replace the poison check. A kernel cache miss that would evict a resident kernel is refused (TOO_LARGE): warm the
 * shape up with one non-record call before recording. */
struct OmegaGpuCmdList;
int omega_gpu_fp8_gemv_grouped_record(struct OmegaGpuCmdList *cl, const OmegaGpuFp8Tensor *t, uint32_t groups, uint64_t tab_va,
                                      uint64_t x_va, uint32_t x_rows, uint64_t y_va);
/* Device addresses of one expert's weights and scales (for a directory the router copies from). */
int omega_gpu_fp8_tensor_entry(const OmegaGpuFp8Tensor *t, uint32_t expert, uint64_t *w_va, uint64_t *s_va);

/* Serving: build and pin the kernel for width k (the one the automatic shape choice uses for that k). */
int omega_gpu_fp8_prepare(uint32_t k);
/* on = 1: a call whose kernel is not cached is refused with TOO_LARGE instead of built. */
void omega_gpu_fp8_seal(int on);

const char *omega_gpu_fp8_rc_name(int rc);
const char *omega_gpu_fp8_last_error(void);
void omega_gpu_fp8_cache_clear(void);

/* ---- measurement knobs (tests; each clears the kernel cache) ---- */
/* Kernel shape override: lane_bytes 8/16, rows 1/2/4/8, warps 1..32; 0 = automatic. */
void omega_gpu_fp8_set_shape(uint32_t lane_bytes, uint32_t rows_per_warp, uint32_t warps_per_cta);
/* Kernel depth override: set = 0 automatic (deep for a single projection of at most 4 MiB whose row fits: rows * k <= 4096), set = 1 forces `depth`
 * (0 = step-by-step kernel). Each call clears the kernel cache. */
void omega_gpu_fp8_set_depth(int set, uint32_t depth);
/* The depth the automatic choice makes for a spec (k, lane_bytes, rows, warps filled in), n rows, `groups` experts:
 * deep only for a single projection of at most 4 MiB of weights (measured, see omega_gpu_fp8_api.c). */
uint32_t omega_gpu_fp8_depth_for(const OmegaFp8GemvSpec *spec, uint32_t n, uint32_t groups);
/* The deep default was measured (SPEED-1B, 2026-10-17) while the launch descriptor allowed 16 KiB of shared memory; the
 * 32 KiB cap (#559, Qwen3.8 bf16 GEMV) would newly admit the 30B o_proj (16464 bytes), which no run has measured. The
 * automatic choice keeps the measured bound; set_depth still reaches anything the descriptor allows. */
#define OMEGA_GPU_FP8_DEEP_MAX_SHARED_BYTES 16384u
/* Register-file check for a spec (chip-found rule: 32 * warps * declared registers <= 65536, declared = peak + 1 rounded
 * up to 8). OK, TOO_LARGE (last_error names the counts) or CODEGEN_FAIL. The automatic depth choice and every build apply it. */
int omega_gpu_fp8_kernel_fits(const OmegaFp8GemvSpec *spec, uint32_t *declared);
/* CTAs per group per launch (0 restores the automatic choice: MAX_CTAS single, GROUPED_CTAS grouped). */
void omega_gpu_fp8_set_cta_budget(uint32_t ctas);
/* 1: tensors uploaded from now on live in GPU-cached memory (read-only weights; omega_gpu_mem_cached.h). Default 0:
 * GPU-uncached like every session buffer. Tensors already uploaded keep their policy. Never set: the first upload reads env
 * AIEN_OMEGA_FP8_WEIGHT_CACHE ("1" = cached; W-L2d, so sc can switch it without a code change). */
void omega_gpu_fp8_set_weight_cache(int on);
/* GEMV variant (W-L2c, 2026-10-14): 1 = omega_blackwell_codegen_moe.c (the default, unchanged), 2 = omega_blackwell_codegen_moe2.c
 * (row-pipelined: the whole row group in flight per warp and the next row group's loads in flight while this one is multiplied;
 * same arithmetic order, y bit-identical). Read once from AIEN_OMEGA_FP8_GEMV ("2") at the first call unless set here first.
 * Each call clears the kernel cache. Variant 2 defaults: OMEGA_GPU_FP8_V2_WARPS warps per CTA, OMEGA_GPU_FP8_V2_GROUPED_CTAS CTAs
 * per group for grouped launches (the shape and budget knobs above still override). */
void omega_gpu_fp8_set_variant(int variant);
int omega_gpu_fp8_variant(void);
/* Kernel schedule (CUT5, measurement knob; clears the kernel cache): 0 (default) = the emitted stalls, 1 = the bytes of a step
 * decoded in a software pipeline plus the dependency-driven stall pass (OmegaFp8GemvSpec.sched; y bit-identical to 0). Variant 1
 * only (variant 2 ignores it). Never set: the first kernel build reads env AIEN_OMEGA_FP8_SCHED ("1" = 1). */
void omega_gpu_fp8_set_sched(int sched);
int omega_gpu_fp8_sched(void);
/* CUT5 variants (variant 1 only; each setter clears the kernel cache; never set: the env is read once at the first build).
 * A, coalesced output (OmegaFp8GemvSpec.coal; applied where warps * rows <= 32): ON by default since SPEED-A (2026-10-17,
 *    OWN-30B decode graph 24.49 -> 21.34 ms/token, tokens identical to golden); AIEN_OMEGA_FP8_COAL=0 or set_coal(0) restores
 *    the earlier kernel (one 4-byte STG per row, no shared slots).
 * B, weight loads before the x loads in each K step (OmegaFp8GemvSpec.wfirst): OFF by default; AIEN_OMEGA_FP8_WFIRST=1 turns it on. */
void omega_gpu_fp8_set_coal(int on);
int omega_gpu_fp8_coal(void);
void omega_gpu_fp8_set_wfirst(int on);
int omega_gpu_fp8_wfirst(void);
/* SPEED-1B (2026-10-17): the automatic deep kernel (x staged in shared memory, every load of the row in flight) for single
 * projections whenever it fits the register file and the launch descriptor's shared-memory limit; ON by default (qkv 5120 x 2048
 * cold 0.732 -> 0.853 of line). AIEN_OMEGA_FP8_DEEP=0 or set_deep(0) keeps the step-by-step kernel everywhere. */
void omega_gpu_fp8_set_deep(int on);
int omega_gpu_fp8_deep(void);
/* Host-only: the spec the automatic choice makes for a call of k columns, n rows, `groups` experts under the current knobs
 * (shape, depth, sched, coal, wfirst, mutant). No chip, no kernel build. */
void omega_gpu_fp8_spec_for(uint32_t k, uint32_t n, uint32_t groups, OmegaFp8GemvSpec *spec);
#define OMEGA_GPU_FP8_V2_WARPS 16u
#define OMEGA_GPU_FP8_V2_GROUPED_CTAS 12u
int omega_gpu_fp8_weight_cache(void);
/* Diagnosis: declare `pad` registers beyond the kernel's peak in the QMD (0 = exact peak). */
void omega_gpu_fp8_set_gpr_pad(uint32_t pad);
/* TEST ONLY: deliberately wrong kernels (1 scale dropped, 2 scale block never advances, 3 group table entry
 * of cy ^ 1). The parity test must catch each. Production never calls this. */
void omega_gpu_fp8_test_set_mutant(int mutant);
/* TEST ONLY: upload scales without the 2^56 fold (host-side mutant; the parity test must catch it). */
void omega_gpu_fp8_test_set_no_fold(int on);
/* The spec the API would pick for (k, n): for tests and the bandwidth table. */
void omega_gpu_fp8_shape_for(uint32_t k, uint32_t n, uint32_t *lane_bytes, uint32_t *rows_per_warp, uint32_t *warps_per_cta);

/* ---- host helpers (omega_gpu_fp8_pack.c), no device ---- */
/* Reference e4m3 decode (port of sc fp8_e4m3fn_to_f32: bias 7, subnormal m * 2^-9, NaN for 0x7f / 0xff). */
float omega_fp8_e4m3_ref(uint8_t b);
/* The kernel's conversion: f32 bits equal to e4m3(b) * 2^-120 (wrong for the NaN codes, which are refused). */
uint32_t omega_fp8_e4m3_kernel_bits(uint8_t b);
int omega_fp8_e4m3_is_nan(uint8_t b);
/* NaN codes in w; first = index of the first one. */
size_t omega_fp8_pack_scan_nan(const uint8_t *w, size_t count, size_t *first);
/* bf16 scales -> f32 * 2^56 (mutant_no_fold: * 1). -1 on a NaN/inf scale or fold overflow. */
int omega_fp8_pack_scales(const uint16_t *bf16, size_t count, float *out, int mutant_no_fold);
void omega_fp8_pack_x_f32(const float *x, uint32_t k, float *out);
void omega_fp8_pack_x_bf16(const uint16_t *x, uint32_t k, float *out);
/* Host oracle in double: y[r] = sum_c ref(w[r][c]) * x[c] * bf16(scale[r/128][c/128]); yscale[r] = sum |terms|
 * (the accumulation scale a relative tolerance is measured against). w: n x k, scales (n/128) x (k/128). */
void omega_fp8_gemv_oracle(const uint8_t *w, const uint16_t *scale_bf16, uint32_t n, uint32_t k,
                           const float *x, double *y, double *yscale);

#ifdef __cplusplus
}
#endif
#endif
