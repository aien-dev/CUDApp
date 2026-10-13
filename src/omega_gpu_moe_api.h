#ifndef OMEGA_GPU_MOE_API_H
#define OMEGA_GPU_MOE_API_H
/*
 * omega_gpu_moe_api: column windows over a resident bf16 weight tensor (OWN-30B cut 2, 2026-10-13).
 *
 * Why: a mixture-of-experts layer keeps the 128 experts' gate+up (k 2048, n 128 x 1536) and down
 * (k 768, n 128 x 2048) projections as ONE resident tensor each and, per token, multiplies the
 * activation row by the 8 selected experts' column windows. The whole-tensor entry
 * (omega_gpu_matmul_resident_*) computes every column; this file computes only [col0, col0 + cols)
 * of a resident tensor with the SAME chip-proven kernel: B is tile-major (column tile nt = kp x 8
 * elements contiguous at element nt * kp * 8, omega_blackwell_codegen.h, omega_matmul_frag_pack_b),
 * so a window that starts on a column tile is the same kernel with the B base advanced to that
 * tile and np = the window's padded width. No kernel change, new file only (8B-lane collision rule).
 *
 * ---- FFI contract (for the Rust wrapper; C ABI, every function is thread-safe through the
 *      process-wide session lock, every pointer is a host pointer) ----
 *
 *   typedef struct OmegaGpuMoeTensor OmegaGpuMoeTensor;            // opaque, device resident
 *
 *   int  omega_gpu_moe_tensor_upload_bf16(uint32_t k, uint32_t n, const uint16_t *b, OmegaGpuMoeTensor **out);
 *   int  omega_gpu_moe_tensor_upload_f32 (uint32_t k, uint32_t n, const float    *b, OmegaGpuMoeTensor **out);
 *        b: k x n row-major (f32 is rounded to bf16, round-to-nearest-even); packed tile-major on the
 *        host and kept in GPU-uncached memory until omega_gpu_moe_tensor_free. Limits: k <= 16384,
 *        n <= 262144 (OMEGA_BW_MATMUL_MAX_*). Returns OMEGA_GPU_MATMUL_OK (0) or an OMEGA_GPU_MATMUL_*
 *        code (omega_gpu_matmul_api.h); *out is NULL on failure.
 *   void omega_gpu_moe_tensor_free (OmegaGpuMoeTensor *t);          // NULL is a no-op
 *   void omega_gpu_moe_tensor_shape(const OmegaGpuMoeTensor *t, uint32_t *k, uint32_t *n);
 *
 *   int  omega_gpu_matmul_resident_window_f32 (uint32_t m, const float    *a, const OmegaGpuMoeTensor *t,
 *                                              uint32_t col0, uint32_t cols, float *c, OmegaGpuMatmulInfo *info);
 *   int  omega_gpu_matmul_resident_window_bf16(uint32_t m, const uint16_t *a, const OmegaGpuMoeTensor *t,
 *                                              uint32_t col0, uint32_t cols, float *c, OmegaGpuMatmulInfo *info);
 *        C[m x cols] = A[m x k] * B[:, col0 .. col0 + cols), f32 output row-major with row stride cols,
 *        i.e. element (r, j) of C equals element (r, col0 + j) of the whole-tensor product.
 *        a: m x k row-major (f32 rounded to bf16 on the way in, as the whole-tensor calls do).
 *        info may be NULL; when given it is filled like the whole-tensor call (n = cols, padded_n = the
 *        window's padded width, resident = true, elapsed_ns = chip time, call_ns = whole call).
 *        ALIGNMENT RULE (fail-closed, OMEGA_GPU_MATMUL_BAD_ARGS, omega_gpu_matmul_last_error() names it):
 *          col0 % 8 == 0  (OMEGA_GPU_MATMUL_TILE_N: a window starts on a column tile),
 *          cols >= 1 and col0 + cols <= n (never past the real columns),
 *          m >= 1 and m <= 8192.
 *        cols itself need not be a multiple of 8: the kernel computes whole tiles and only the real
 *        cols columns are copied out (the rest of the last tile is the tensor's next columns or its
 *        zero padding, computed and discarded).
 *   int  omega_gpu_matmul_resident_windows_f32(uint32_t m, const float *a, const OmegaGpuMoeTensor *t,
 *                                              uint32_t nwin, const uint32_t *col0, uint32_t cols, float *c,
 *                                              OmegaGpuMatmulInfo *info);
 *        nwin windows of the same width (the 8 experts of a token): A is packed and staged once, then one
 *        launch per window. c holds nwin consecutive row-major blocks of m x cols: window w's result is at
 *        c + w * m * cols. Every col0[w] obeys the alignment rule; nwin >= 1 and <= 64. info (if given)
 *        sums elapsed_ns over the windows and counts chip_calls.
 *
 *   int  omega_gpu_moe_reserve(uint32_t max_rows, uint32_t max_k, uint32_t max_cols, uint32_t kernel_slots);
 *   int  omega_gpu_moe_prepare(uint32_t m, uint32_t k, uint32_t cols);
 *   void omega_gpu_moe_seal(int on);
 *   void omega_gpu_moe_unreserve(void);
 *        Serving reservation, the same contract as omega_gpu_matmul_reserve / _prepare / _seal (no
 *        driver allocation on the call path once reserved, sealed = an unprepared kernel is refused
 *        with OMEGA_GPU_MATMUL_TOO_LARGE). This API has its own staging buffers and kernel cache, so a
 *        server reserves both APIs.
 *   void omega_gpu_moe_cache_clear(void);
 *   void omega_gpu_moe_set_cta_budget(uint32_t ctas);   // 0 restores the default; clears the kernel cache
 *   uint32_t omega_gpu_moe_cta_budget(void);
 *
 *   Return codes and OmegaGpuMatmulInfo are those of omega_gpu_matmul_api.h; the failure stage is in
 *   omega_gpu_matmul_last_error(). An uncertain completion latches the process exactly as the
 *   whole-tensor API does (omega_gpu_matmul_is_blocked()).
 */
#include "omega_gpu_matmul_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OmegaGpuMoeTensor OmegaGpuMoeTensor;

#define OMEGA_GPU_MOE_MAX_WINDOWS 64u

int omega_gpu_moe_tensor_upload_bf16(uint32_t k, uint32_t n, const uint16_t *b, OmegaGpuMoeTensor **out);
int omega_gpu_moe_tensor_upload_f32(uint32_t k, uint32_t n, const float *b, OmegaGpuMoeTensor **out);
void omega_gpu_moe_tensor_free(OmegaGpuMoeTensor *t);
void omega_gpu_moe_tensor_shape(const OmegaGpuMoeTensor *t, uint32_t *k, uint32_t *n);

int omega_gpu_matmul_resident_window_f32(uint32_t m, const float *a, const OmegaGpuMoeTensor *t,
                                         uint32_t col0, uint32_t cols, float *c, OmegaGpuMatmulInfo *info);
int omega_gpu_matmul_resident_window_bf16(uint32_t m, const uint16_t *a, const OmegaGpuMoeTensor *t,
                                          uint32_t col0, uint32_t cols, float *c, OmegaGpuMatmulInfo *info);
int omega_gpu_matmul_resident_windows_f32(uint32_t m, const float *a, const OmegaGpuMoeTensor *t,
                                          uint32_t nwin, const uint32_t *col0, uint32_t cols, float *c,
                                          OmegaGpuMatmulInfo *info);

int omega_gpu_moe_reserve(uint32_t max_rows, uint32_t max_k, uint32_t max_cols, uint32_t kernel_slots);
int omega_gpu_moe_prepare(uint32_t m, uint32_t k, uint32_t cols);
void omega_gpu_moe_seal(int on);
void omega_gpu_moe_unreserve(void);
void omega_gpu_moe_cache_clear(void);
void omega_gpu_moe_set_cta_budget(uint32_t ctas);
uint32_t omega_gpu_moe_cta_budget(void);

/* Pure host arithmetic, exported for tests and the Rust side: the byte offset of column tile
 * (col0 / 8) inside a packed tensor of padded depth kp, i.e. where a window's B starts. */
uint64_t omega_gpu_moe_window_byte_offset(uint32_t kp, uint32_t col0);

/* TEST ONLY. mutant: deliberately wrong kernels (last K pass dropped), the parity oracle must catch
 * them. skew: every window's B base is moved by this many column tiles (the "wrong window" control:
 * the window must then disagree with the whole-tensor columns). Both clear the kernel cache / take
 * effect on the next call; production code never calls them. */
void omega_gpu_moe_test_set_mutant(int on);
void omega_gpu_moe_test_set_window_skew(int32_t tiles);
/* Test hook: one-row calls so far that rewrote row 0 only of an already zero-padded activation pack. */
uint64_t omega_gpu_moe_test_a_row0_hits(void);

#ifdef __cplusplus
}
#endif
#endif
