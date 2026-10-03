#include "omega_gpu_matmul_api.h"
#include "omega_blackwell_matmul.h"
#include "omega_blackwell_submit.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* Small in-process kernel cache keyed by shape. Eight shapes cover a decode
 * loop (gemv per projection) comfortably; least-recently-inserted is evicted. */
#define CACHE_SLOTS 8

typedef struct {
    bool valid;
    uint32_t m, k, n;
    OmegaMatMulSpec spec;
    OmegaBlackwellKernel kernel;
} CacheSlot;

static CacheSlot g_cache[CACHE_SLOTS];
static unsigned g_cache_next;
static pthread_mutex_t g_cache_mu = PTHREAD_MUTEX_INITIALIZER;

const char *omega_gpu_matmul_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_MATMUL_OK: return "OK";
    case OMEGA_GPU_MATMUL_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_MATMUL_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_MATMUL_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_MATMUL_CHIP_FAIL: return "CHIP_FAIL";
    case OMEGA_GPU_MATMUL_PARITY_FAIL: return "PARITY_FAIL";
    default: return "UNKNOWN";
    }
}

void omega_gpu_matmul_cache_clear(void) {
    pthread_mutex_lock(&g_cache_mu);
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        if (g_cache[i].valid) omega_blackwell_kernel_free(&g_cache[i].kernel);
        memset(&g_cache[i], 0, sizeof g_cache[i]);
    }
    g_cache_next = 0;
    pthread_mutex_unlock(&g_cache_mu);
}

/* Returns a slot holding a generated kernel for the shape, or NULL with *rc set.
 * The slot stays valid until cache_clear or eviction; callers copy spec+kernel
 * pointers under the lock and release before the chip call (kernel code bytes
 * are immutable once generated, eviction only happens under the same lock, and
 * with 8 slots a caller's shape cannot be evicted by its own call). */
static int kernel_for_shape(uint32_t m, uint32_t k, uint32_t n,
                            OmegaMatMulSpec *spec_out, OmegaBlackwellKernel *kernel_out,
                            bool *hit) {
    pthread_mutex_lock(&g_cache_mu);
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        CacheSlot *s = &g_cache[i];
        if (s->valid && s->m == m && s->k == k && s->n == n) {
            *spec_out = s->spec;
            *kernel_out = s->kernel;
            *hit = true;
            pthread_mutex_unlock(&g_cache_mu);
            return OMEGA_GPU_MATMUL_OK;
        }
    }
    OmegaMatMulSpec spec;
    if (omega_matmul_spec_init(&spec, m, k, n, OMEGA_MATMUL_PRECISION_BF16) != 0) {
        pthread_mutex_unlock(&g_cache_mu);
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    }
    OmegaBlackwellKernel kernel;
    memset(&kernel, 0, sizeof kernel);
    if (omega_blackwell_codegen_matmul(&spec, &kernel) != 0 || !kernel.code || kernel.code_size == 0) {
        pthread_mutex_unlock(&g_cache_mu);
        return OMEGA_GPU_MATMUL_CODEGEN_FAIL;
    }
    CacheSlot *s = &g_cache[g_cache_next];
    g_cache_next = (g_cache_next + 1) % CACHE_SLOTS;
    if (s->valid) omega_blackwell_kernel_free(&s->kernel);
    s->valid = true;
    s->m = m; s->k = k; s->n = n;
    s->spec = spec;
    s->kernel = kernel;
    *spec_out = spec;
    *kernel_out = kernel;
    *hit = false;
    pthread_mutex_unlock(&g_cache_mu);
    return OMEGA_GPU_MATMUL_OK;
}

static uint32_t round_up(uint32_t v, uint32_t q) { return (v + q - 1) / q * q; }

int omega_gpu_matmul_bf16(uint32_t m, uint32_t k, uint32_t n,
                          const uint16_t *a, const uint16_t *b, float *c,
                          OmegaGpuMatmulInfo *info) {
    if (info) memset(info, 0, sizeof *info);
    if (!a || !b || !c || m == 0 || k == 0 || n == 0) return OMEGA_GPU_MATMUL_BAD_ARGS;
    if (m > OMEGA_BW_MATMUL_MAX_M || k > OMEGA_BW_MATMUL_MAX_K || n > OMEGA_BW_MATMUL_MAX_N)
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    const uint32_t mp = round_up(m, OMEGA_GPU_MATMUL_TILE_M);
    const uint32_t np = round_up(n, OMEGA_GPU_MATMUL_TILE_N);
    const uint32_t tk = OMEGA_GPU_MATMUL_TILE_K;
    const uint32_t slices = (k + tk - 1) / tk;
    /* CTA envelope: the kernel rasters 16x16 tiles (grid_x = np/16 rounded up,
     * grid_y = rows/16). Keep grid_x * grid_y <= OMEGA_GPU_MATMUL_MAX_CTAS by
     * limiting the rows handed to one launch. */
    const uint32_t grid_x = (np + 15) / 16;
    uint32_t rows_per_call = (OMEGA_GPU_MATMUL_MAX_CTAS / grid_x) * OMEGA_GPU_MATMUL_TILE_M;
    if (rows_per_call < OMEGA_GPU_MATMUL_TILE_M) rows_per_call = OMEGA_GPU_MATMUL_TILE_M; /* grid_x alone exceeds: 1024 cols = 64 CTAs, still inside */
    if (rows_per_call > mp) rows_per_call = mp;
    const uint32_t row_blocks = (mp + rows_per_call - 1) / rows_per_call;
    if (info) {
        info->m = m; info->k = k; info->n = n; info->padded_m = mp; info->padded_n = np;
        info->k_slices = slices; info->rows_per_call = rows_per_call;
    }

    uint16_t *ap = calloc((size_t)rows_per_call * tk, sizeof *ap);
    uint16_t *bp = calloc((size_t)tk * np, sizeof *bp);
    float *cp = malloc((size_t)rows_per_call * np * sizeof *cp);
    if (!ap || !bp || !cp) { free(ap); free(bp); free(cp); return OMEGA_GPU_MATMUL_BAD_ARGS; }
    for (size_t i = 0; i < (size_t)m * n; i++) c[i] = 0.0f;

    bool all_parity = true, any_miss = false;
    int rc = OMEGA_GPU_MATMUL_OK;
    for (uint32_t s = 0; s < slices && rc == OMEGA_GPU_MATMUL_OK; s++) {
        const uint32_t k0 = s * tk;
        const uint32_t kw = (k - k0 < tk) ? (k - k0) : tk;
        /* B slice: rows k0..k0+kw of columns 0..n, zero elsewhere (row-major 16 x np) */
        memset(bp, 0, (size_t)tk * np * sizeof *bp);
        for (uint32_t j = 0; j < kw; j++)
            memcpy(&bp[(size_t)j * np], &b[(size_t)(k0 + j) * n], n * sizeof *bp);
        for (uint32_t rb = 0; rb < row_blocks && rc == OMEGA_GPU_MATMUL_OK; rb++) {
            const uint32_t r0 = rb * rows_per_call;
            uint32_t rows = mp - r0 < rows_per_call ? mp - r0 : rows_per_call;
            rows = round_up(rows, OMEGA_GPU_MATMUL_TILE_M);
            const uint32_t rows_real = (m > r0) ? ((m - r0 < rows) ? (m - r0) : rows) : 0;
            OmegaMatMulSpec spec;
            OmegaBlackwellKernel kernel;
            bool hit = false;
            rc = kernel_for_shape(rows, tk, np, &spec, &kernel, &hit);
            if (rc != OMEGA_GPU_MATMUL_OK) break;
            if (info) { if (hit) info->kernel_cache_hit = true; else any_miss = true; }
            /* A slice: rows r0..r0+rows_real of columns k0..k0+kw, zero elsewhere (row-major rows x 16) */
            memset(ap, 0, (size_t)rows * tk * sizeof *ap);
            for (uint32_t i = 0; i < rows_real; i++)
                memcpy(&ap[(size_t)i * tk], &a[(size_t)(r0 + i) * k + k0], kw * sizeof *ap);

            OmegaBlackwellMatMulExecution exec;
            memset(&exec, 0, sizeof exec);
            float max_abs = 0.0f, max_rel = 0.0f;
            int r = omega_blackwell_execute_matmul_tensor(&spec, &kernel, ap, bp, cp, &exec, &max_abs, &max_rel);
            if (info) {
                info->chip_calls++;
                info->elapsed_ns += exec.elapsed_ns;
                if (max_abs > info->max_abs_err) info->max_abs_err = max_abs;
                if (max_rel > info->max_rel_err) info->max_rel_err = max_rel;
                info->mismatch_count += exec.mismatch_count;
                info->completion_marker = exec.completion_marker;
                info->sm_architecture = exec.sm_architecture;
                memcpy(info->target_chip, exec.target_chip, sizeof info->target_chip);
            }
            if (r != 0 || exec.completion_marker != OMEGA_BW_MARKER_COMPLETION_PAYLOAD) { rc = OMEGA_GPU_MATMUL_CHIP_FAIL; break; }
            if (!exec.parity_verified) all_parity = false;
            for (uint32_t i = 0; i < rows_real; i++)
                for (uint32_t j = 0; j < n; j++)
                    c[(size_t)(r0 + i) * n + j] += cp[(size_t)i * np + j];
        }
    }
    free(ap); free(bp); free(cp);
    if (info) {
        info->parity_verified = all_parity && rc == OMEGA_GPU_MATMUL_OK;
        if (any_miss) info->kernel_cache_hit = false; /* hit only when every launch found its kernel */
    }
    if (rc != OMEGA_GPU_MATMUL_OK) return rc;
    if (!all_parity) return OMEGA_GPU_MATMUL_PARITY_FAIL;
    return OMEGA_GPU_MATMUL_OK;
}

int omega_gpu_matmul_f32(uint32_t m, uint32_t k, uint32_t n,
                         const float *a, const float *b, float *c,
                         OmegaGpuMatmulInfo *info) {
    if (info) memset(info, 0, sizeof *info);
    if (!a || !b || !c || m == 0 || k == 0 || n == 0) return OMEGA_GPU_MATMUL_BAD_ARGS;
    if (m > OMEGA_BW_MATMUL_MAX_M || k > OMEGA_BW_MATMUL_MAX_K || n > OMEGA_BW_MATMUL_MAX_N)
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    size_t na = (size_t)m * k, nb = (size_t)k * n;
    uint16_t *ab = malloc(na * sizeof *ab);
    uint16_t *bb = malloc(nb * sizeof *bb);
    if (!ab || !bb) { free(ab); free(bb); return OMEGA_GPU_MATMUL_BAD_ARGS; }
    for (size_t i = 0; i < na; i++) ab[i] = omega_fp32_to_bf16(a[i]);
    for (size_t i = 0; i < nb; i++) bb[i] = omega_fp32_to_bf16(b[i]);
    int rc = omega_gpu_matmul_bf16(m, k, n, ab, bb, c, info);
    free(ab);
    free(bb);
    return rc;
}
