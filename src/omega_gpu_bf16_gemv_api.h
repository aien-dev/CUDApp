#ifndef OMEGA_GPU_BF16_GEMV_API_H
#define OMEGA_GPU_BF16_GEMV_API_H
/*
 * omega_gpu_bf16_gemv_api: native (no CUDA) bf16 streaming GEMV for the OWN-30B lane: the lm_head
 * (vocab 151936 x hidden 2048, bf16 weights, f32 logits). New file on the shared session; no existing API changes.
 *
 * ============ C signatures and semantics (for the Rust binding) ============
 *   typedef struct OmegaGpuBf16Tensor OmegaGpuBf16Tensor;   // opaque, device resident
 *
 *   int omega_gpu_bf16_tensor_upload(uint32_t n, uint32_t k, const uint16_t *w_bf16, OmegaGpuBf16Tensor **out);
 *     w: n * k bf16 bit patterns, row-major [n][k] (n output rows, k input columns): the checkpoint's lm_head.weight
 *     as is, copied byte for byte. Requires n >= 1, k % 256 == 0, 256 <= k <= 16384. Refuses a NaN or infinite bf16
 *     (exponent all ones) with OMEGA_GPU_BF16_NAN_WEIGHT (nothing uploaded). Opens the device on first use. Resident
 *     until omega_gpu_bf16_tensor_free.
 *
 *   int omega_gpu_bf16_gemv_f32 (const OmegaGpuBf16Tensor *t, const float    *x, float *y, OmegaGpuBf16Info *info);
 *   int omega_gpu_bf16_gemv_bf16(const OmegaGpuBf16Tensor *t, const uint16_t *x, float *y, OmegaGpuBf16Info *info);
 *     y[r] = sum_c W[r][c] * x[c], r < n, c < k, f32 accumulate (summation order: omega_blackwell_codegen_bf16.h; the
 *     result is deterministic and the test's host emulation reproduces it bit for bit). x: k values, y: n f32. One
 *     launch. info may be NULL.
 *
 *   int omega_gpu_bf16_prepare(uint32_t k);       build and pin the kernel for width k now (serving).
 *   void omega_gpu_bf16_tensor_free(OmegaGpuBf16Tensor *t);
 *   void omega_gpu_bf16_tensor_shape(const OmegaGpuBf16Tensor *t, uint32_t *n, uint32_t *k);
 *   const char *omega_gpu_bf16_rc_name(int rc);   const char *omega_gpu_bf16_last_error(void);
 *
 * Return codes: 0 OK; negative OMEGA_GPU_BF16_* below. An uncertain completion latches the process (every later call
 * of every API returns CHIP_FAIL), as the other APIs do. Every output word is poisoned before the launch and the
 * call refuses a result that still holds poison. No host oracle runs inside the call.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "omega_blackwell_codegen_bf16.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    OMEGA_GPU_BF16_OK = 0,
    OMEGA_GPU_BF16_BAD_ARGS = -1,
    OMEGA_GPU_BF16_TOO_LARGE = -2,
    OMEGA_GPU_BF16_CODEGEN_FAIL = -3,
    OMEGA_GPU_BF16_CHIP_FAIL = -4,
    OMEGA_GPU_BF16_NAN_WEIGHT = -5
};

typedef struct {
    uint32_t n, k, rows_per_warp, warps_per_cta, depth, grid_x, steps;
    uint64_t weight_bytes;  /* bf16 bytes the launch reads (n * k * 2) */
    uint64_t elapsed_ns;    /* chip submit-to-second-marker */
    uint64_t call_ns;       /* whole API call */
    uint32_t completion_marker;
    bool kernel_cache_hit;
    char target_chip[64];
    uint32_t sm_architecture;
} OmegaGpuBf16Info;

typedef struct OmegaGpuBf16Tensor OmegaGpuBf16Tensor;

int omega_gpu_bf16_tensor_upload(uint32_t n, uint32_t k, const uint16_t *w_bf16, OmegaGpuBf16Tensor **out);
void omega_gpu_bf16_tensor_free(OmegaGpuBf16Tensor *t);
void omega_gpu_bf16_tensor_shape(const OmegaGpuBf16Tensor *t, uint32_t *n, uint32_t *k);
int omega_gpu_bf16_gemv_f32(const OmegaGpuBf16Tensor *t, const float *x, float *y, OmegaGpuBf16Info *info);
int omega_gpu_bf16_gemv_bf16(const OmegaGpuBf16Tensor *t, const uint16_t *x, float *y, OmegaGpuBf16Info *info);
/* Record path (OWN-30B W-L2a, decode graph; omega_gpu_cmdlist.h): the same launch as gemv_f32 appended to cl instead
 * of submitted. x_va: k f32 on the device (plain, not prescaled); y_va: n f32 written when the list runs. The caller
 * holds omega_gpu_session_lock(). Nothing staged, poisoned or read back. A kernel cache miss that would evict a resident
 * kernel is refused (TOO_LARGE): warm the shape up with one non-record call (or omega_gpu_bf16_prepare) first. */
struct OmegaGpuCmdList;
int omega_gpu_bf16_gemv_record(struct OmegaGpuCmdList *cl, const OmegaGpuBf16Tensor *t, uint64_t x_va, uint64_t y_va);
/* ROWS form (sc #642 prefill): `rows` token rows in one launch (grid_y = rows <= 65535); row r reads x_va + r * k f32
 * and writes y_va + r * n f32. Record-only; the kernel (spec.ystride = n) is cached per n like any other shape. */
int omega_gpu_bf16_gemv_rows_record(struct OmegaGpuCmdList *cl, const OmegaGpuBf16Tensor *t, uint32_t rows, uint64_t x_va, uint64_t y_va);
/* MASK form (HUM L314 exact two-pass head; omega_blackwell_codegen_bf16.h): record-only, rows 1. Row r with
 * hi[r] < *tau (hi_va: n f32, tau_va: one f32, both device) gets -inf in y; every other row gets the plain kernel's
 * bits. Caller holds omega_gpu_session_lock(); the first call compiles into a free cache slot. */
int omega_gpu_bf16_gemv_mask_record(struct OmegaGpuCmdList *cl, const OmegaGpuBf16Tensor *t, uint64_t x_va, uint64_t y_va, uint64_t hi_va, uint64_t tau_va);
/* XBATCH form (the 35B stroke: one lm_head read for several conversations): `xbatch` (1..OMEGA_BF16_MAX_XBATCH) x rows
 * at x_va + m * k f32 share every weight read inside one CTA (every row staged in shared memory, so xbatch * k * 4 bytes
 * must fit the launch descriptor's shared limit); y row m at y_va + m * ystride f32 (ystride >= n; the gap stays
 * untouched). Row m equals the one-row result of its x bit for bit (the same FMA order). xbatch 1 is the plain launch.
 * Record-only; the kernel (spec.xbatch, spec.ystride) is cached per (k, xbatch, ystride): pin it with prepare_xbatch. */
int omega_gpu_bf16_gemv_xbatch_record(struct OmegaGpuCmdList *cl, const OmegaGpuBf16Tensor *t, uint32_t xbatch, uint64_t x_va, uint64_t y_va, uint32_t ystride);
int omega_gpu_bf16_prepare(uint32_t k);
int omega_gpu_bf16_prepare_mask(uint32_t k); /* build and pin the MASK kernel for width k (the exact head, serving) */
int omega_gpu_bf16_prepare_xbatch(uint32_t k, uint32_t xbatch, uint32_t ystride); /* build and pin the XBATCH kernel (xbatch <= 1: prepare(k)) */
const char *omega_gpu_bf16_rc_name(int rc);
const char *omega_gpu_bf16_last_error(void);
void omega_gpu_bf16_cache_clear(void);

/* Register-file check (chip-found rule: 32 * warps * declared registers <= 65536, declared = peak + 1 rounded up to 8). */
int omega_gpu_bf16_kernel_fits(const OmegaBf16GemvSpec *spec, uint32_t *declared);
/* The shape the API picks for width k (rows, warps, depth, CTAs): for tests and the bandwidth table. */
void omega_gpu_bf16_shape_for(uint32_t k, uint32_t *rows, uint32_t *warps, uint32_t *depth, uint32_t *ctas);

/* ---- measurement knobs (tests; each clears the kernel cache; 0 restores the automatic choice) ---- */
void omega_gpu_bf16_set_shape(uint32_t rows, uint32_t warps, uint32_t depth);
void omega_gpu_bf16_set_cta_budget(uint32_t ctas);
/* TEST ONLY: deliberately wrong kernels 1, 2, 3 (see the codegen header). Production never calls this. */
void omega_gpu_bf16_test_set_mutant(int mutant);
/* TEST ONLY: the MASK form alone under mutant m (4: rows with hi >= tau get -inf, the others the plain bits). 0 none. */
void omega_gpu_bf16_test_set_mask_mutant(int m);

#ifdef __cplusplus
}
#endif
#endif
