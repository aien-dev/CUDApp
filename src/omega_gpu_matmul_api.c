/*
 * omega_gpu_matmul_api.c: FB-1 cut 1b. See the header for the contract.
 *
 * Submission path: the same words the chip-proven single-tile path uses
 * (omega_blackwell_execute_matmul_tensor: SETUP_WORDS, constant bank upload,
 * QMD0/QMD1 launch, first marker release, then the C3 tail: L2_FLUSH_DIRTY and a
 * second release on uncached memory that the host waits for before reading).
 * Every buffer both sides touch is GPU-uncached (nvos.h NVOS32_ATTR2_GPU_CACHEABLE:
 * "For system memory this will not be coherent with direct CPU mappings").
 * Difference from cut 1: the device, channel, pushbuffer, constant bank, marker
 * page, QMD page and kernel code buffers persist for the process; the GPFIFO is
 * retired after every completed launch (nvrm_enqueue refuses to overrun
 * unretired entries, nvrm.c).
 */
#include "omega_gpu_matmul_api.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_matmul.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_submit.h"
#include "m16_native.h"
#include "nvrm.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CACHE_SLOTS 8
#define PAGE 0x1000ull
#define MARKER_TIMEOUT_MS 20000ull /* a 1x2048x128256 call is ~0.1 s; 20 s is a stall, not a slow run */
#define POISON_F32 (-999.0f)
#define ORACLE_TOL 1e-5

/* Same words as src/omega_blackwell_submit.c SETUP_WORDS (compute class setup). */
static const uint32_t SETUP_WORDS[18] = {
    0x20012061, 0x0000cec0, 0x20012092, 0x00000001, 0x200120a8, 0x0000000f, 0x2001255d, 0x00000003,
    0x2001255e, 0x20000000, 0x2001255f, 0x000fffff, 0x20012557, 0x00000003, 0x20012558, 0x22000000,
    0x20012559, 0x00000000,
};

struct OmegaGpuTensor {
    uint32_t k, n, kp, np;
    NvrmMem mem; /* kp x np bf16 */
};

typedef struct {
    bool valid;
    uint32_t kp, np, grid_x;
    int mutant;
    OmegaMatMulSpec spec;
    OmegaBlackwellKernel kernel;
    NvrmMem code; /* resident copy of kernel.code */
} CacheSlot;

static struct {
    bool open, blocked;
    M16NativeContext ctx;
    NvrmMem pb, cbank, marker, qmd; /* per-launch scratch, reused */
    NvrmMem a_buf, c_buf;           /* activation and result staging, grown on demand */
    size_t a_cap, c_cap;
    uint32_t cta_budget;
    int oracle, mutant;
    CacheSlot cache[CACHE_SLOTS];
    unsigned cache_next;
} g = { .cta_budget = OMEGA_GPU_MATMUL_MAX_CTAS, .oracle = 1 };

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static const char *g_err = "";
#define FAILAT(s) (g_err = (s))
const char *omega_gpu_matmul_last_error(void) { return g_err; }

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

static uint32_t round_up(uint32_t v, uint32_t q) { return (v + q - 1) / q * q; }
static size_t page_up(size_t b) { b = (b + PAGE - 1) & ~(PAGE - 1); return b < PAGE ? PAGE : b; }

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

int omega_gpu_matmul_is_blocked(void) { return g.blocked ? 1 : 0; }

/* ---- device lifetime (caller holds g_mu) ---- */

static int dev_alloc(size_t bytes, NvrmMem *out) {
    return nvrm_alloc_gpu_uncached(&g.ctx.rm, page_up(bytes), out) == 0 ? 0 : -1;
}

static void dev_free(NvrmMem *m) {
    if (m->cpu) (void)nvrm_free(&g.ctx.rm, m);
    memset(m, 0, sizeof *m);
}

static void cache_clear_locked(void) {
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        CacheSlot *s = &g.cache[i];
        if (!s->valid) continue;
        omega_blackwell_kernel_free(&s->kernel);
        if (g.open && !g.blocked) dev_free(&s->code);
        memset(s, 0, sizeof *s);
    }
    g.cache_next = 0;
}

static int dev_open_locked(void) {
    if (g.blocked) return -1;
    if (g.open) return 0;
    memset(&g.ctx, 0, sizeof g.ctx);
    if (m16_native_open(&g.ctx) != 0) { FAILAT("m16_native_open"); return -1; }
    if (m16_native_create_channel(&g.ctx) != 0) { FAILAT("m16_native_create_channel"); m16_native_close(&g.ctx); return -1; }
    if (dev_alloc(0x10000, &g.pb) || dev_alloc(PAGE, &g.cbank) || dev_alloc(PAGE, &g.marker) || dev_alloc(0x10000, &g.qmd)) {
        FAILAT("device scratch alloc");
        m16_native_close(&g.ctx);
        memset(&g.pb, 0, sizeof g.pb); memset(&g.cbank, 0, sizeof g.cbank);
        memset(&g.marker, 0, sizeof g.marker); memset(&g.qmd, 0, sizeof g.qmd);
        return -1;
    }
    g.ctx.pb_mem = g.pb; /* m16_native_enqueue_methods copies the method stream here */
    g.open = true;
    return 0;
}

void omega_gpu_device_close(void) {
    pthread_mutex_lock(&g_mu);
    if (g.open && !g.blocked) {
        cache_clear_locked();
        dev_free(&g.a_buf); dev_free(&g.c_buf); g.a_cap = g.c_cap = 0;
        dev_free(&g.qmd); dev_free(&g.marker); dev_free(&g.cbank); dev_free(&g.pb);
        m16_native_close(&g.ctx);
        g.open = false;
    }
    pthread_mutex_unlock(&g_mu);
}

void omega_gpu_matmul_cache_clear(void) {
    pthread_mutex_lock(&g_mu);
    cache_clear_locked();
    pthread_mutex_unlock(&g_mu);
}

void omega_gpu_matmul_set_cta_budget(uint32_t ctas) {
    pthread_mutex_lock(&g_mu);
    g.cta_budget = ctas ? ctas : OMEGA_GPU_MATMUL_MAX_CTAS;
    cache_clear_locked();
    pthread_mutex_unlock(&g_mu);
}

uint32_t omega_gpu_matmul_cta_budget(void) { return g.cta_budget; }

void omega_gpu_matmul_set_oracle(int on) { g.oracle = on ? 1 : 0; }

void omega_gpu_matmul_test_set_mutant(int on) {
    pthread_mutex_lock(&g_mu);
    g.mutant = on ? 1 : 0;
    cache_clear_locked();
    pthread_mutex_unlock(&g_mu);
}

/* Staging buffer at least `bytes` long (grown by reallocation; contents not kept). */
static int staging(NvrmMem *buf, size_t *cap, size_t bytes) {
    if (*cap >= bytes && buf->cpu) return 0;
    dev_free(buf);
    *cap = 0;
    if (dev_alloc(bytes, buf) != 0) return -1;
    *cap = buf->size;
    return 0;
}

/* ---- kernel cache (caller holds g_mu, device open) ---- */

static int kernel_for(uint32_t kp, uint32_t np, uint32_t grid_x, CacheSlot **out, bool *hit) {
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        CacheSlot *s = &g.cache[i];
        if (s->valid && s->kp == kp && s->np == np && s->grid_x == grid_x && s->mutant == g.mutant) {
            *out = s; *hit = true;
            return OMEGA_GPU_MATMUL_OK;
        }
    }
    OmegaMatMulSpec spec;
    /* the kernel ignores M; 16 keeps the spec inside the tile rule */
    if (omega_matmul_spec_init(&spec, 16, kp, np, OMEGA_MATMUL_PRECISION_BF16) != 0) return OMEGA_GPU_MATMUL_TOO_LARGE;
    OmegaBlackwellKernel kernel;
    memset(&kernel, 0, sizeof kernel);
    if (omega_blackwell_codegen_matmul_tensor_loop(&spec, grid_x, g.mutant, &kernel) != 0 || !kernel.code || kernel.code_size == 0)
        return OMEGA_GPU_MATMUL_CODEGEN_FAIL;
    NvrmMem code;
    if (dev_alloc(kernel.code_size, &code) != 0) { FAILAT("kernel code alloc"); omega_blackwell_kernel_free(&kernel); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    memcpy(code.cpu, kernel.code, kernel.code_size);
    CacheSlot *s = &g.cache[g.cache_next];
    g.cache_next = (g.cache_next + 1) % CACHE_SLOTS;
    if (s->valid) { omega_blackwell_kernel_free(&s->kernel); dev_free(&s->code); }
    s->valid = true; s->kp = kp; s->np = np; s->grid_x = grid_x; s->mutant = g.mutant;
    s->spec = spec; s->kernel = kernel; s->code = code;
    *out = s; *hit = false;
    return OMEGA_GPU_MATMUL_OK;
}

/* ---- one launch (caller holds g_mu, device open, not blocked) ---- */

static int launch(const CacheSlot *ks, uint64_t a_va, uint64_t b_va, uint64_t c_va,
                  uint32_t grid_x, uint32_t grid_y, uint64_t *elapsed_ns, uint32_t *marker_out) {
    const uint32_t threads_x = 32, threads_y = 1;
    uint32_t cbank_data[OMEGA_BW_CBANK_DRIVER_WORDS];
    uint32_t cbank_args[OMEGA_BW_CBANK_MATMUL_ARGS_WORDS];
    omega_blackwell_build_cbank_driver_2d(cbank_data, g.cbank.va, threads_x, threads_y, grid_x, grid_y);
    omega_blackwell_build_cbank_args_matmul(cbank_args, a_va, b_va, c_va, grid_y * 16u, ks->kp, ks->np);
    memcpy(g.cbank.cpu, cbank_data, sizeof cbank_data);
    memcpy((uint8_t *)g.cbank.cpu + 0x380, cbank_args, sizeof cbank_args);

    const uint64_t qmd0_va = g.qmd.va, qmd1_va = g.qmd.va + 0x1000, sem_va = g.qmd.va + 0x2000, scratch_va = g.qmd.va + 0x4000;
    uint32_t gpr = ks->kernel.gpr_count > 64u ? round_up(ks->kernel.gpr_count, 8u) : 64u;
    OmegaBlackwellQmdConfig cfg = {
        .code_va = ks->code.va, .cbank_va = g.cbank.va, .scratch_va = scratch_va, .sem_va = sem_va,
        .qmd0_va = qmd0_va, .qmd1_va = qmd1_va, .num_elements = (size_t)grid_x * grid_y * 128u,
        .threads_per_block = 32, .grid_width = grid_x * grid_y, .threads_x = threads_x, .threads_y = threads_y,
        .grid_x = grid_x, .grid_y = grid_y, .gpr_count = gpr,
    };
    uint32_t qmd0_words[OMEGA_BW_QMD_WORDS], qmd1_words[OMEGA_BW_QMD_WORDS];
    omega_blackwell_build_qmd0(qmd0_words, qmd0_va, qmd1_va);
    omega_blackwell_build_qmd1(qmd1_words, &cfg);
    if (omega_blackwell_verify_qmd_invariants(qmd1_words) != 0) { FAILAT("qmd invariants"); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    memcpy(g.qmd.cpu, qmd0_words, sizeof qmd0_words);
    memcpy((uint8_t *)g.qmd.cpu + 0x1000, qmd1_words, sizeof qmd1_words);

    volatile uint32_t *hsem = (volatile uint32_t *)((uint8_t *)g.qmd.cpu + 0x2000);
    volatile uint32_t *hmarker = (volatile uint32_t *)g.marker.cpu;
    volatile uint32_t *hmarker2 = (volatile uint32_t *)((uint8_t *)g.marker.cpu + 0x10);
    *hsem = 0; *hmarker = 0; *hmarker2 = 0;
    __asm__ volatile("dsb sy" ::: "memory");

    uint32_t pb[1024];
    size_t n = 0;
    memcpy(&pb[n], SETUP_WORDS, sizeof SETUP_WORDS); n += sizeof SETUP_WORDS / 4;
    /* Invalidate the SM instruction, constant and data caches before this launch.
     * Kernel code lives in GPU-uncached memory, but the SM instruction cache is
     * not coherent with host writes: when a cache slot is evicted and its code
     * address is reused for a different kernel, the chip ran stale instructions
     * (FB1-CUT1B-f629798 sweep: 8x1024x8, 192x16x192, 512x16x64 unwritten,
     * 128x16x512 hung, all after the eighth distinct kernel). Method and bits
     * from NVIDIA open-gpu-doc classes/compute/clcec0.h (BLACKWELL_COMPUTE_B):
     * NVCEC0_INVALIDATE_SHADER_CACHES 0x021c, INSTRUCTION 0:0, DATA 4:4,
     * CONSTANT 12:12 (the 0x021c form waits for idle first). Shared constants in
     * omega_blackwell_submit.h so every launcher emits the same words. */
    pb[n++] = nvrm_mthd(1, OMEGA_BW_MTHD_INVALIDATE_SHADER_CACHES, 1); pb[n++] = OMEGA_BW_INVALIDATE_SHADER_CACHES_ALL;
    /* constant bank 0: driver words then our arguments at 0x380 (inline upload, same as cut 1) */
    pb[n++] = nvrm_mthd(1, 0x0188, 2); pb[n++] = (uint32_t)(g.cbank.va >> 32); pb[n++] = (uint32_t)g.cbank.va;
    pb[n++] = nvrm_mthd(1, 0x0180, 2); pb[n++] = 0x00000380; pb[n++] = 0x00000001;
    pb[n++] = nvrm_mthd(1, 0x01b0, 1); pb[n++] = 0x00000041;
    pb[n++] = (224u << 16) | (1u << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[n], cbank_data, 224 * 4); n += 224;
    pb[n++] = nvrm_mthd(1, 0x0188, 2); pb[n++] = (uint32_t)((g.cbank.va + 0x380) >> 32); pb[n++] = (uint32_t)(g.cbank.va + 0x380);
    pb[n++] = nvrm_mthd(1, 0x0180, 2); pb[n++] = 0x00000028; pb[n++] = 0x00000001;
    pb[n++] = nvrm_mthd(1, 0x01b0, 1); pb[n++] = 0x00000041;
    pb[n++] = (10u << 16) | (1u << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[n], cbank_args, 10 * 4); n += 10;
    /* QMD0 (null grid, dependence 2) */
    pb[n++] = (98u << 16) | (1u << 13) | (0x0318 >> 2) | (2u << 28);
    pb[n++] = (1u << 30) | (uint32_t)((qmd0_va >> 40) & 0x1ff); pb[n++] = (uint32_t)(qmd0_va >> 8);
    memcpy(&pb[n], qmd0_words, 96 * 4); n += 96;
    /* intermediate semaphore init */
    pb[n++] = nvrm_mthd(1, 0x0188, 2); pb[n++] = (uint32_t)(sem_va >> 32); pb[n++] = (uint32_t)sem_va;
    pb[n++] = nvrm_mthd(1, 0x0180, 2); pb[n++] = 0x00000004; pb[n++] = 0x00000001;
    pb[n++] = nvrm_mthd(1, 0x01b0, 1); pb[n++] = 0x00000041;
    pb[n++] = (1u << 16) | (1u << 13) | (0x01b4 >> 2) | (6u << 28);
    pb[n++] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT;
    /* QMD1 (the kernel) */
    pb[n++] = (98u << 16) | (1u << 13) | (0x0318 >> 2) | (2u << 28);
    pb[n++] = (1u << 30) | (uint32_t)((qmd1_va >> 40) & 0x1ff); pb[n++] = (uint32_t)(qmd1_va >> 8);
    memcpy(&pb[n], qmd1_words, 96 * 4); n += 96;
    /* first marker */
    pb[n++] = nvrm_mthd(0, 0x005c, 5);
    pb[n++] = (uint32_t)g.marker.va; pb[n++] = (uint32_t)(g.marker.va >> 32);
    pb[n++] = OMEGA_BW_MARKER_COMPLETION_PAYLOAD; pb[n++] = 0; pb[n++] = 0x1 | (1u << 20);
    /* C3 tail: L2_FLUSH_DIRTY, then the second release on the uncached marker page */
    pb[n++] = nvrm_mthd(0, 0x0028, 4); pb[n++] = 0; pb[n++] = 0; pb[n++] = 0; pb[n++] = (0x10u << 27);
    pb[n++] = nvrm_mthd(0, 0x005c, 5);
    pb[n++] = (uint32_t)(g.marker.va + 0x10); pb[n++] = (uint32_t)((g.marker.va + 0x10) >> 32);
    pb[n++] = OMEGA_BW_MARKER2_PAYLOAD; pb[n++] = 0; pb[n++] = 0x1 | (1u << 20);

    uint64_t t0 = now_ns();
    if (m16_native_submit_methods(&g.ctx, pb, n) != 0) { FAILAT("submit_methods"); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    if (m16_native_wait_marker(hmarker, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, MARKER_TIMEOUT_MS) != 0 ||
        m16_native_wait_marker(hmarker2, OMEGA_BW_MARKER2_PAYLOAD, MARKER_TIMEOUT_MS) != 0) {
        g.blocked = true; (void)FAILAT("marker wait timed out (latched)"); /* uncertain completion: keep everything, refuse every later call */
        return OMEGA_GPU_MATMUL_CHIP_FAIL;
    }
    __asm__ volatile("dsb sy" ::: "memory");
    uint64_t t1 = now_ns();
    nvrm_retire(&g.ctx.rm, g.ctx.rm.put); /* both releases landed: the GPFIFO entry is consumed */
    if (*hsem != OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE) { FAILAT("intermediate semaphore not DONE"); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    if (elapsed_ns) *elapsed_ns += t1 - t0;
    if (marker_out) *marker_out = *hmarker;
    return OMEGA_GPU_MATMUL_OK;
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
    int rc = OMEGA_GPU_MATMUL_OK;
    pthread_mutex_lock(&g_mu);
    if (dev_open_locked() != 0 || (FAILAT("tensor alloc"), dev_alloc((size_t)t->kp * t->np * 2u, &t->mem) != 0)) rc = OMEGA_GPU_MATMUL_CHIP_FAIL;
    pthread_mutex_unlock(&g_mu);
    if (rc != OMEGA_GPU_MATMUL_OK) { free(t); return rc; }
    uint16_t *d = t->mem.cpu;
    if (t->np != n || t->kp != k) memset(d, 0, (size_t)t->kp * t->np * 2u);
    for (uint32_t r = 0; r < k; r++) memcpy(&d[(size_t)r * t->np], &b[(size_t)r * n], (size_t)n * 2u);
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
    pthread_mutex_lock(&g_mu);
    if (g.open && !g.blocked) dev_free(&t->mem);
    pthread_mutex_unlock(&g_mu);
    free(t);
}

/* ---- the matmul ---- */

static int matmul_core(uint32_t m, const uint16_t *a, const OmegaGpuTensor *b, float *c, OmegaGpuMatmulInfo *info) {
    const uint32_t k = b->k, n = b->n, kp = b->kp, np = b->np;
    const uint32_t mp = round_up(m, OMEGA_GPU_MATMUL_TILE_M), mt = mp / 16, nt = np / 8;
    pthread_mutex_lock(&g_mu);
    int rc = OMEGA_GPU_MATMUL_OK;
    if (g.blocked && FAILAT("process latched")) { pthread_mutex_unlock(&g_mu); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    if (dev_open_locked() != 0) { pthread_mutex_unlock(&g_mu); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    const uint32_t budget = g.cta_budget ? g.cta_budget : OMEGA_GPU_MATMUL_MAX_CTAS;
    /* rows per launch: all of them unless the row tiles alone exceed the budget */
    uint32_t rows_per_launch = mt <= budget ? mp : budget * 16u;
    uint32_t grid_y_full = rows_per_launch / 16u;
    uint32_t grid_x = budget / grid_y_full; if (grid_x == 0) grid_x = 1; if (grid_x > nt) grid_x = nt;
    if (info) {
        info->m = m; info->k = k; info->n = n; info->padded_m = mp; info->padded_n = np; info->padded_k = kp;
        info->k_slices = 1; info->rows_per_call = rows_per_launch; info->grid_x = grid_x; info->grid_y = grid_y_full;
        info->resident = true; info->kernel_cache_hit = true;
        strncpy(info->target_chip, "NVIDIA DGX Spark (Grace Blackwell GB10)", sizeof info->target_chip - 1);
        info->sm_architecture = 121;
    }
    if (staging(&g.a_buf, &g.a_cap, (size_t)mp * kp * 2u) != 0 || staging(&g.c_buf, &g.c_cap, (size_t)mp * np * 4u) != 0) {
        pthread_mutex_unlock(&g_mu); return OMEGA_GPU_MATMUL_CHIP_FAIL;
    }
    /* activation rows in, zero padded; result rows poisoned */
    uint16_t *ad = g.a_buf.cpu;
    if (mp != m || kp != k) memset(ad, 0, (size_t)mp * kp * 2u);
    for (uint32_t r = 0; r < m; r++) memcpy(&ad[(size_t)r * kp], &a[(size_t)r * k], (size_t)k * 2u);
    float *cd = g.c_buf.cpu;
    for (size_t i = 0; i < (size_t)m * np; i++) cd[i] = POISON_F32;
    __asm__ volatile("dsb sy" ::: "memory");

    for (uint32_t r0 = 0; r0 < mp && rc == OMEGA_GPU_MATMUL_OK; r0 += rows_per_launch) {
        uint32_t rows = mp - r0 < rows_per_launch ? mp - r0 : rows_per_launch;
        uint32_t grid_y = rows / 16u;
        CacheSlot *ks = NULL; bool hit = false;
        rc = kernel_for(kp, np, grid_x, &ks, &hit);
        if (rc != OMEGA_GPU_MATMUL_OK) break;
        if (info && !hit) info->kernel_cache_hit = false;
        rc = launch(ks, g.a_buf.va + (uint64_t)r0 * kp * 2u, b->mem.va, g.c_buf.va + (uint64_t)r0 * np * 4u,
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
        if (unchanged) { FAILAT("output poison survived"); rc = OMEGA_GPU_MATMUL_CHIP_FAIL; }
    }
    pthread_mutex_unlock(&g_mu);
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
