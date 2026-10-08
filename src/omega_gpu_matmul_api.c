/*
 * omega_gpu_matmul_api.c: FB-1 cut 1b, on the shared session since cut 4b.
 * See the header for the contract.
 *
 * Submission path: omega_gpu_session_launch (src/omega_gpu_session.c), the same
 * words this file emitted in cut 1b (chip PASS FB1-CUT1B-4345406): SETUP_WORDS,
 * shader-cache invalidate, constant bank upload, QMD0/QMD1 launch, first marker
 * release, then the C3 tail (L2_FLUSH_DIRTY and a second release on uncached
 * memory the host waits for before reading). Every buffer both sides touch is
 * GPU-uncached (nvos.h NVOS32_ATTR2_GPU_CACHEABLE: "For system memory this will
 * not be coherent with direct CPU mappings"). The device, channel, pushbuffer,
 * constant bank, marker page and QMD page now belong to the session and are
 * shared with the elementwise and attention APIs; this file keeps the resident
 * tensors, the kernel cache (host and device copies) and its staging buffers.
 */
#include "omega_gpu_matmul_api.h"
#include "omega_gpu_session.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_matmul.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_submit.h"
#include "nvrm.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define CACHE_SLOTS 8      /* default kernel cache slots; raised only by omega_gpu_matmul_reserve */
#define CACHE_SLOTS_MAX 128 /* storage for the opt-in larger cache (a code buffer is a few KB); Qwen3-4B serving needs 95 */
#define FRAG_PACK_TILES 64u /* column tiles per host packing chunk at upload */
#define MARKER_TIMEOUT_MS 20000ull /* a 1x2048x128256 call is ~0.1 s; 20 s is a stall, not a slow run */
#define POISON_F32 (-999.0f)
#define ORACLE_TOL 1e-5

struct OmegaGpuTensor {
    uint32_t k, n, kp, np;
    NvrmMem mem; /* kp x np bf16 in fragment order (omega_matmul_frag_pack_b) */
};

typedef struct {
    bool valid;
    bool pinned; /* built by omega_gpu_matmul_prepare: never evicted */
    uint32_t kp, np, grid_x;
    int mutant;
    OmegaMatMulSpec spec;
    OmegaBlackwellKernel kernel;
    NvrmMem code; /* resident copy of kernel.code */
} CacheSlot;

static struct {
    OmegaGpuScratch a_buf, c_buf; /* activation and result staging, grown to the high-water mark */
    uint32_t cta_budget;
    int oracle, mutant;
    uint32_t unroll; /* test override of the fragment kernel unroll; 0 = omega_matmul_frag_unroll */
    bool hooked;
    CacheSlot cache[CACHE_SLOTS_MAX];
    unsigned cache_next;
    unsigned slots; /* active kernel cache slots, <= CACHE_SLOTS_MAX */
    int sealed;     /* omega_gpu_matmul_seal: a kernel cache miss is refused, never built */
} g = { .cta_budget = OMEGA_GPU_MATMUL_MAX_CTAS, .oracle = 1, .slots = CACHE_SLOTS };

#define FAILAT(s) (omega_gpu_session_set_error(s), 1)
const char *omega_gpu_matmul_last_error(void) { return omega_gpu_session_last_error(); }

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static uint32_t round_up(uint32_t v, uint32_t q) { return (v + q - 1) / q * q; }

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

int omega_gpu_matmul_is_blocked(void) { return omega_gpu_session_is_blocked(); }

/* ---- device-side state (caller holds the session lock) ---- */

static void cache_clear_locked(void) {
    for (unsigned i = 0; i < CACHE_SLOTS_MAX; i++) {
        CacheSlot *s = &g.cache[i];
        if (!s->valid) continue;
        omega_blackwell_kernel_free(&s->kernel);
        omega_gpu_session_free(&s->code);
        memset(s, 0, sizeof *s);
    }
    g.cache_next = 0;
}

/* Called by the session (lock held, device still open) right before it closes. */
static void on_session_close(void) {
    cache_clear_locked();
    omega_gpu_session_scratch_free(&g.a_buf);
    omega_gpu_session_scratch_free(&g.c_buf);
    g.slots = CACHE_SLOTS; /* a serving reservation ends with the device */
    g.sealed = 0;
}

static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}

void omega_gpu_device_close(void) { omega_gpu_session_close(); }

void omega_gpu_matmul_cache_clear(void) {
    omega_gpu_session_lock();
    cache_clear_locked();
    omega_gpu_session_unlock();
}

void omega_gpu_matmul_set_cta_budget(uint32_t ctas) {
    omega_gpu_session_lock();
    g.cta_budget = ctas ? ctas : OMEGA_GPU_MATMUL_MAX_CTAS;
    cache_clear_locked();
    omega_gpu_session_unlock();
}

uint32_t omega_gpu_matmul_cta_budget(void) { return g.cta_budget; }

/* ---- opt-in serving reservation (see the header) ---- */

int omega_gpu_matmul_reserve(uint32_t max_rows, uint32_t max_k, uint32_t max_n, uint32_t max_n_one_row, uint32_t kernel_slots) {
    if (max_rows == 0 || max_k == 0 || max_n == 0) { omega_gpu_session_set_error("matmul reserve: zero bound"); return OMEGA_GPU_MATMUL_BAD_ARGS; }
    if (max_rows > OMEGA_BW_MATMUL_MAX_M || max_k > OMEGA_BW_MATMUL_MAX_K || max_n > OMEGA_BW_MATMUL_MAX_N || max_n_one_row > OMEGA_BW_MATMUL_MAX_N ||
        kernel_slots > CACHE_SLOTS_MAX) {
        omega_gpu_session_set_error("matmul reserve: bound past the matmul envelope");
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    }
    const size_t mp = round_up(max_rows, OMEGA_GPU_MATMUL_TILE_M), kp = round_up(max_k, OMEGA_GPU_MATMUL_TILE_K);
    size_t c_bytes = mp * round_up(max_n, OMEGA_GPU_MATMUL_TILE_N) * 4u;
    if (max_n_one_row) { /* a call of up to 16 rows (one row tile) may be wider than the batch width: lm_head */
        size_t one = (size_t)OMEGA_GPU_MATMUL_TILE_M * round_up(max_n_one_row, OMEGA_GPU_MATMUL_TILE_N) * 4u;
        if (one > c_bytes) c_bytes = one;
    }
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_MATMUL_OK;
    if (!dev_open_locked()) rc = OMEGA_GPU_MATMUL_CHIP_FAIL;
    else if (omega_gpu_session_scratch_reserve(&g.a_buf, mp * kp * 2u) != 0 || omega_gpu_session_scratch_reserve(&g.c_buf, c_bytes) != 0) {
        /* roll back this API's own buffers: a failed reserve leaves nothing flagged */
        omega_gpu_session_scratch_unreserve(&g.a_buf); omega_gpu_session_scratch_unreserve(&g.c_buf);
        rc = OMEGA_GPU_MATMUL_CHIP_FAIL;
    }
    else if (kernel_slots > g.slots) { /* grow only; the cache restarts empty */
        cache_clear_locked();
        g.slots = kernel_slots;
    }
    omega_gpu_session_unlock();
    return rc;
}

void omega_gpu_matmul_unreserve(void) {
    omega_gpu_session_lock();
    omega_gpu_session_scratch_unreserve(&g.a_buf);
    omega_gpu_session_scratch_unreserve(&g.c_buf);
    omega_gpu_session_unlock();
}

static int kernel_for(uint32_t kp, uint32_t np, uint32_t grid_x, CacheSlot **out, bool *hit);
static void launch_shape(uint32_t mp, uint32_t np, uint32_t *rows_per_launch, uint32_t *grid_x);

int omega_gpu_matmul_prepare(uint32_t m, uint32_t k, uint32_t n) {
    if (m == 0 || k == 0 || n == 0) { omega_gpu_session_set_error("matmul prepare: zero dimension"); return OMEGA_GPU_MATMUL_BAD_ARGS; }
    if (m > OMEGA_BW_MATMUL_MAX_M || k > OMEGA_BW_MATMUL_MAX_K || n > OMEGA_BW_MATMUL_MAX_N) {
        omega_gpu_session_set_error("matmul prepare: shape past the matmul envelope");
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    }
    const uint32_t mp = round_up(m, OMEGA_GPU_MATMUL_TILE_M), kp = round_up(k, OMEGA_GPU_MATMUL_TILE_K), np = round_up(n, OMEGA_GPU_MATMUL_TILE_N);
    uint32_t rows_per_launch, grid_x;
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_MATMUL_OK;
    if (!dev_open_locked()) rc = OMEGA_GPU_MATMUL_CHIP_FAIL;
    else {
        launch_shape(mp, np, &rows_per_launch, &grid_x);
        CacheSlot *s = NULL; bool hit = false;
        rc = kernel_for(kp, np, grid_x, &s, &hit);
        if (rc == OMEGA_GPU_MATMUL_OK) s->pinned = true; /* a prepared kernel is never evicted */
    }
    omega_gpu_session_unlock();
    return rc;
}

/* Anything that clears the kernel cache while sealed (cache_clear, set_cta_budget, the test knobs, a
 * reserve that grows the slot count) also drops the pinned kernels; the seal stays, so every later
 * call is refused by name ("was not prepared") until the server prepares again or releases. */
void omega_gpu_matmul_seal(int on) {
    omega_gpu_session_lock();
    g.sealed = on ? 1 : 0;
    omega_gpu_session_unlock();
}

void omega_gpu_matmul_set_oracle(int on) { g.oracle = on ? 1 : 0; }

void omega_gpu_matmul_test_set_mutant(int on) {
    omega_gpu_session_lock();
    g.mutant = on ? 1 : 0;
    cache_clear_locked();
    omega_gpu_session_unlock();
}

void omega_gpu_matmul_test_set_unroll(uint32_t unroll) {
    omega_gpu_session_lock();
    g.unroll = unroll;
    cache_clear_locked();
    omega_gpu_session_unlock();
}

/* ---- one launch shape (the kernel cache key bakes in grid_x) ---- */

/* rows per launch: all of them unless the row tiles alone exceed the CTA budget; grid_x fills the rest of the budget */
static void launch_shape(uint32_t mp, uint32_t np, uint32_t *rows_per_launch, uint32_t *grid_x) {
    const uint32_t budget = g.cta_budget ? g.cta_budget : OMEGA_GPU_MATMUL_MAX_CTAS, mt = mp / 16u, nt = np / 8u;
    *rows_per_launch = mt <= budget ? mp : budget * 16u;
    uint32_t gx = budget / (*rows_per_launch / 16u);
    if (gx == 0) gx = 1;
    if (gx > nt) gx = nt;
    *grid_x = gx;
}

/* ---- kernel cache (caller holds the lock, device open) ---- */

static int kernel_for(uint32_t kp, uint32_t np, uint32_t grid_x, CacheSlot **out, bool *hit) {
    for (unsigned i = 0; i < g.slots; i++) {
        CacheSlot *s = &g.cache[i];
        if (s->valid && s->kp == kp && s->np == np && s->grid_x == grid_x && s->mutant == g.mutant) {
            *out = s; *hit = true;
            return OMEGA_GPU_MATMUL_OK;
        }
    }
    if (g.sealed) {
        char e[160];
        snprintf(e, sizeof e, "serving reservation exceeded: matmul kernel kp=%u np=%u grid_x=%u was not prepared", kp, np, grid_x);
        omega_gpu_session_set_error(e);
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    }
    /* choose the slot before any build or driver call: round-robin, skipping prepared (pinned) kernels; all pinned = refused, nothing evicted */
    CacheSlot *s = NULL;
    for (unsigned i = 0; i < g.slots && !s; i++) {
        CacheSlot *c = &g.cache[(g.cache_next + i) % g.slots];
        if (!c->valid || !c->pinned) { s = c; g.cache_next = (unsigned)((c - g.cache) + 1) % g.slots; }
    }
    if (!s) {
        omega_gpu_session_set_error("matmul kernel cache: every slot holds a prepared kernel (raise kernel_slots)");
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    }
    OmegaMatMulSpec spec;
    /* the kernel ignores M; 16 keeps the spec inside the tile rule */
    if (omega_matmul_spec_init(&spec, 16, kp, np, OMEGA_MATMUL_PRECISION_BF16) != 0) return OMEGA_GPU_MATMUL_TOO_LARGE;
    OmegaBlackwellKernel kernel;
    memset(&kernel, 0, sizeof kernel);
    uint32_t unroll = g.unroll ? g.unroll : omega_matmul_frag_unroll(kp);
    /* the mutant drops the last pass, so it needs two: halve the unroll of a one-pass shape */
    if (g.mutant && !g.unroll) while (unroll > 1 && kp / 16u / unroll < 2) unroll /= 2;
    if (omega_blackwell_codegen_matmul_frag(&spec, grid_x, unroll, g.mutant, &kernel) != 0 ||
        !kernel.code || kernel.code_size == 0)
        return OMEGA_GPU_MATMUL_CODEGEN_FAIL;
    NvrmMem code;
    if (omega_gpu_session_alloc_code(kernel.code_size, &code) != 0) { omega_blackwell_kernel_free(&kernel); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    memcpy(code.cpu, kernel.code, kernel.code_size);
    if (s->valid) { omega_blackwell_kernel_free(&s->kernel); omega_gpu_session_free(&s->code); }
    s->valid = true; s->pinned = false; s->kp = kp; s->np = np; s->grid_x = grid_x; s->mutant = g.mutant;
    s->spec = spec; s->kernel = kernel; s->code = code;
    *out = s; *hit = false;
    return OMEGA_GPU_MATMUL_OK;
}

/* ---- one launch (caller holds the lock, device open, not blocked) ---- */

static int launch(const CacheSlot *ks, uint64_t a_va, uint64_t b_va, uint64_t c_va,
                  uint32_t grid_x, uint32_t grid_y, uint64_t *elapsed_ns, uint32_t *marker_out) {
    uint32_t cbank_args[OMEGA_BW_CBANK_MATMUL_ARGS_WORDS];
    omega_blackwell_build_cbank_args_matmul(cbank_args, a_va, b_va, c_va, grid_y * 16u, ks->kp, ks->np);
    OmegaGpuLaunch L = {
        .code_va = ks->code.va, .gpr_count = ks->kernel.gpr_count,
        .threads_x = 32, .threads_y = 1, .grid_x = grid_x, .grid_y = grid_y,
        .num_elements = (size_t)grid_x * grid_y * 128u,
        .args = cbank_args, .n_args = OMEGA_BW_CBANK_MATMUL_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
    };
    return omega_gpu_session_launch(&L, elapsed_ns, marker_out) == 0 ? OMEGA_GPU_MATMUL_OK : OMEGA_GPU_MATMUL_CHIP_FAIL;
}

/* ---- tensors ---- */

void omega_gpu_tensor_shape(const OmegaGpuTensor *t, uint32_t *k, uint32_t *n) {
    if (k) *k = t ? t->k : 0;
    if (n) *n = t ? t->n : 0;
}

int omega_gpu_tensor_upload_bf16(uint32_t k, uint32_t n, const uint16_t *b, OmegaGpuTensor **out) {
    if (out) *out = NULL;
    if (!b || !out || k == 0 || n == 0) return OMEGA_GPU_MATMUL_BAD_ARGS;
    if (k > OMEGA_BW_MATMUL_MAX_K || n > OMEGA_BW_MATMUL_MAX_N) return OMEGA_GPU_MATMUL_TOO_LARGE;
    OmegaGpuTensor *t = calloc(1, sizeof *t);
    if (!t) return OMEGA_GPU_MATMUL_BAD_ARGS;
    t->k = k; t->n = n; t->kp = round_up(k, OMEGA_GPU_MATMUL_TILE_K); t->np = round_up(n, OMEGA_GPU_MATMUL_TILE_N);
    /* fragment order (omega_matmul_frag_pack_b), packed on the host FRAG_PACK_TILES column
     * tiles at a time and copied in order into the uncached device buffer */
    const uint32_t nt = t->np / 8u;
    uint16_t *h = malloc((size_t)FRAG_PACK_TILES * t->kp * 8u * sizeof *h);
    if (!h) { free(t); return OMEGA_GPU_MATMUL_BAD_ARGS; }
    int rc = OMEGA_GPU_MATMUL_OK;
    omega_gpu_session_lock();
    if (!dev_open_locked() || omega_gpu_session_alloc((size_t)t->kp * t->np * 2u, &t->mem) != 0) rc = OMEGA_GPU_MATMUL_CHIP_FAIL;
    omega_gpu_session_unlock();
    if (rc != OMEGA_GPU_MATMUL_OK) { free(h); free(t); return rc; }
    uint16_t *d = t->mem.cpu;
    for (uint32_t nt0 = 0; nt0 < nt; nt0 += FRAG_PACK_TILES) {
        const uint32_t nt1 = nt - nt0 < FRAG_PACK_TILES ? nt : nt0 + FRAG_PACK_TILES;
        omega_matmul_frag_pack_b(b, k, n, t->kp, nt0, nt1, h);
        memcpy(&d[(size_t)nt0 * t->kp * 8u], h, (size_t)(nt1 - nt0) * t->kp * 8u * sizeof *h);
    }
    free(h);
    __asm__ volatile("dsb sy" ::: "memory");
    *out = t;
    return OMEGA_GPU_MATMUL_OK;
}

int omega_gpu_tensor_upload_f32(uint32_t k, uint32_t n, const float *b, OmegaGpuTensor **out) {
    if (out) *out = NULL;
    if (!b || !out || k == 0 || n == 0) return OMEGA_GPU_MATMUL_BAD_ARGS;
    if (k > OMEGA_BW_MATMUL_MAX_K || n > OMEGA_BW_MATMUL_MAX_N) return OMEGA_GPU_MATMUL_TOO_LARGE;
    size_t cnt = (size_t)k * n;
    uint16_t *h = malloc(cnt * sizeof *h);
    if (!h) return OMEGA_GPU_MATMUL_BAD_ARGS;
    for (size_t i = 0; i < cnt; i++) h[i] = omega_fp32_to_bf16(b[i]);
    int rc = omega_gpu_tensor_upload_bf16(k, n, h, out);
    free(h);
    return rc;
}

void omega_gpu_tensor_free(OmegaGpuTensor *t) {
    if (!t) return;
    omega_gpu_session_lock();
    omega_gpu_session_free(&t->mem);
    omega_gpu_session_unlock();
    free(t);
}

/* ---- the matmul ---- */

static int matmul_core(uint32_t m, const uint16_t *a, const OmegaGpuTensor *b, float *c, OmegaGpuMatmulInfo *info) {
    const uint32_t k = b->k, n = b->n, kp = b->kp, np = b->np;
    const uint32_t mp = round_up(m, OMEGA_GPU_MATMUL_TILE_M);
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_MATMUL_OK;
    if (!dev_open_locked()) { omega_gpu_session_unlock(); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    uint32_t rows_per_launch, grid_x;
    launch_shape(mp, np, &rows_per_launch, &grid_x);
    uint32_t grid_y_full = rows_per_launch / 16u;
    if (info) {
        info->m = m; info->k = k; info->n = n; info->padded_m = mp; info->padded_n = np; info->padded_k = kp;
        info->k_slices = 1; info->rows_per_call = rows_per_launch; info->grid_x = grid_x; info->grid_y = grid_y_full;
        info->resident = true; info->kernel_cache_hit = true;
        strncpy(info->target_chip, "NVIDIA DGX Spark (Grace Blackwell GB10)", sizeof info->target_chip - 1);
        info->sm_architecture = 121;
    }
    if (g.sealed && !(g.a_buf.reserved && g.c_buf.reserved)) {
        /* sealed means no driver allocation on the call path; without a scratch reservation the
         * buffers below could still grow, so refuse before touching them */
        omega_gpu_session_set_error("serving reservation exceeded: matmul sealed without a scratch reservation (call omega_gpu_reserve_serving first)");
        omega_gpu_session_unlock();
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    }
    int src = omega_gpu_session_scratch(&g.a_buf, (size_t)mp * kp * 2u);
    if (src == 0) src = omega_gpu_session_scratch(&g.c_buf, (size_t)mp * np * 4u);
    if (src != 0) {
        omega_gpu_session_unlock();
        /* a serving reservation refuses growth loudly (TOO_LARGE, error text names the sizes); anything else is a driver failure */
        return src == OMEGA_GPU_SESSION_SCRATCH_RESERVED ? OMEGA_GPU_MATMUL_TOO_LARGE : OMEGA_GPU_MATMUL_CHIP_FAIL;
    }
    /* activation rows in, fragment order (omega_matmul_frag_pack_a), zero padded, packed on the
     * host and copied in order into the uncached buffer; result rows poisoned */
    uint16_t *ah = malloc((size_t)mp * kp * sizeof *ah);
    if (!ah) { omega_gpu_session_unlock(); return OMEGA_GPU_MATMUL_BAD_ARGS; }
    omega_matmul_frag_pack_a(a, m, k, mp, kp, ah);
    memcpy(g.a_buf.mem.cpu, ah, (size_t)mp * kp * sizeof *ah);
    free(ah);
    float *cd = g.c_buf.mem.cpu;
    for (size_t i = 0; i < (size_t)m * np; i++) cd[i] = POISON_F32;
    __asm__ volatile("dsb sy" ::: "memory");

    for (uint32_t r0 = 0; r0 < mp && rc == OMEGA_GPU_MATMUL_OK; r0 += rows_per_launch) {
        uint32_t rows = mp - r0 < rows_per_launch ? mp - r0 : rows_per_launch;
        uint32_t grid_y = rows / 16u;
        CacheSlot *ks = NULL; bool hit = false;
        rc = kernel_for(kp, np, grid_x, &ks, &hit);
        if (rc != OMEGA_GPU_MATMUL_OK) break;
        if (info && !hit) info->kernel_cache_hit = false;
        rc = launch(ks, g.a_buf.mem.va + (uint64_t)r0 * kp * 2u, b->mem.va, g.c_buf.mem.va + (uint64_t)r0 * np * 4u,
                    grid_x, grid_y, info ? &info->elapsed_ns : NULL, info ? &info->completion_marker : NULL);
        if (info) info->chip_calls++;
    }
    if (rc == OMEGA_GPU_MATMUL_OK) {
        /* the chip must have written every real output word (poison check, as the engine does) */
        size_t unchanged = 0;
        for (uint32_t r = 0; r < m; r++) {
            const float *row = &cd[(size_t)r * np];
            for (uint32_t j = 0; j < n; j++) if (row[j] == POISON_F32) unchanged++;
            memcpy(&c[(size_t)r * n], row, (size_t)n * sizeof *c);
        }
        if (unchanged && FAILAT("output poison survived")) rc = OMEGA_GPU_MATMUL_CHIP_FAIL;
    }
    omega_gpu_session_unlock();
    return rc;
}

static int resident_entry(uint32_t m, const uint16_t *a, const OmegaGpuTensor *b, float *c, OmegaGpuMatmulInfo *info) {
    uint64_t t0 = now_ns();
    if (info) memset(info, 0, sizeof *info);
    if (!a || !b || !c || m == 0 || !b->mem.cpu) return OMEGA_GPU_MATMUL_BAD_ARGS;
    if (m > OMEGA_BW_MATMUL_MAX_M) return OMEGA_GPU_MATMUL_TOO_LARGE;
    int rc = matmul_core(m, a, b, c, info);
    if (info) info->call_ns = now_ns() - t0;
    return rc;
}

int omega_gpu_matmul_resident_bf16(uint32_t m, const uint16_t *a, const OmegaGpuTensor *b, float *c,
                                   OmegaGpuMatmulInfo *info) {
    return resident_entry(m, a, b, c, info);
}

int omega_gpu_matmul_resident_f32(uint32_t m, const float *a, const OmegaGpuTensor *b, float *c,
                                  OmegaGpuMatmulInfo *info) {
    uint64_t t0 = now_ns();
    if (info) memset(info, 0, sizeof *info);
    if (!a || !b || !c || m == 0 || !b->mem.cpu) return OMEGA_GPU_MATMUL_BAD_ARGS;
    if (m > OMEGA_BW_MATMUL_MAX_M) return OMEGA_GPU_MATMUL_TOO_LARGE;
    size_t cnt = (size_t)m * b->k;
    uint16_t *h = malloc(cnt * sizeof *h);
    if (!h) return OMEGA_GPU_MATMUL_BAD_ARGS;
    for (size_t i = 0; i < cnt; i++) h[i] = omega_fp32_to_bf16(a[i]);
    int rc = matmul_core(m, h, b, c, info);
    free(h);
    if (info) info->call_ns = now_ns() - t0;
    return rc;
}

/* Host oracle: double accumulation of the bf16 inputs; an element mismatches when
 * its error exceeds ORACLE_TOL of the accumulation scale (sum of |terms|). */
static void oracle(uint32_t m, uint32_t k, uint32_t n, const uint16_t *a, const uint16_t *b, const float *c, OmegaGpuMatmulInfo *info) {
    float max_abs = 0.0f, max_rel = 0.0f;
    size_t mism = 0;
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            double acc = 0.0, scale = 0.0;
            for (uint32_t t = 0; t < k; t++) {
                double p = (double)omega_bf16_to_fp32(a[(size_t)i * k + t]) * (double)omega_bf16_to_fp32(b[(size_t)t * n + j]);
                acc += p; scale += fabs(p);
            }
            double d = fabs(acc - (double)c[(size_t)i * n + j]);
            double rel = d / (scale > 1e-6 ? scale : 1e-6);
            if ((float)d > max_abs) max_abs = (float)d;
            if ((float)rel > max_rel) max_rel = (float)rel;
            if (rel > ORACLE_TOL) mism++;
        }
    }
    info->max_abs_err = max_abs; info->max_rel_err = max_rel; info->mismatch_count = mism;
    info->oracle_ran = true; info->parity_verified = (mism == 0);
}

int omega_gpu_matmul_bf16(uint32_t m, uint32_t k, uint32_t n,
                          const uint16_t *a, const uint16_t *b, float *c,
                          OmegaGpuMatmulInfo *info) {
    uint64_t t0 = now_ns();
    OmegaGpuMatmulInfo local;
    if (!info) info = &local;
    memset(info, 0, sizeof *info);
    if (!a || !b || !c || m == 0 || k == 0 || n == 0) return OMEGA_GPU_MATMUL_BAD_ARGS;
    if (m > OMEGA_BW_MATMUL_MAX_M || k > OMEGA_BW_MATMUL_MAX_K || n > OMEGA_BW_MATMUL_MAX_N)
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    OmegaGpuTensor *t = NULL;
    int rc = omega_gpu_tensor_upload_bf16(k, n, b, &t);
    if (rc != OMEGA_GPU_MATMUL_OK) return rc;
    rc = matmul_core(m, a, t, c, info);
    omega_gpu_tensor_free(t);
    info->resident = false;
    if (rc == OMEGA_GPU_MATMUL_OK && g.oracle) {
        oracle(m, k, n, a, b, c, info);
        if (!info->parity_verified) rc = OMEGA_GPU_MATMUL_PARITY_FAIL;
    }
    info->call_ns = now_ns() - t0;
    return rc;
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
