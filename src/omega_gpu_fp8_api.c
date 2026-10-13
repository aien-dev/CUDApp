/* omega_gpu_fp8_api.c: FP8 e4m3 GEMV entry points on the shared session (OWN-30B Cut 6 / Cut 5).
 * See the header. Submission path and buffer policy are the matmul API's: every buffer both sides
 * touch is GPU-uncached (nvos.h NVOS32_ATTR2_GPU_CACHEABLE: "For system memory this will not be
 * coherent with direct CPU mappings"), one launch = omega_gpu_session_launch (shader-cache
 * invalidate, constant bank, QMD0/QMD1, marker, L2_FLUSH_DIRTY, second marker on uncached memory). */
#include "omega_gpu_fp8_api.h"
#include "omega_blackwell_codegen_moe.h"
#include "omega_gpu_session.h"
#include "omega_gpu_fmt.h"
#include <string.h>

#define CACHE_SLOTS 16
#define MARKER_TIMEOUT_MS 20000ull
#define POISON_F32 (-999.0f)

struct OmegaGpuFp8Tensor {
    uint32_t experts, n, k;
    OmegaGpuMem w; /* experts * n * k e4m3 bytes, row-major */
    OmegaGpuMem s; /* experts * (n/128) * (k/128) f32 (scale * 2^56) */
};

typedef struct {
    bool valid, pinned;
    OmegaFp8GemvSpec spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
} Slot;

static struct {
    OmegaGpuScratch x_buf, y_buf;
    OmegaGpuMem tab; /* group table, OMEGA_GPU_FP8_MAX_GROUPS * 16 bytes */
    uint32_t cta_budget;
    uint32_t lane_bytes, rows, warps; /* overrides, 0 = automatic */
    int mutant, no_fold, sealed;
    bool hooked;
    Slot cache[CACHE_SLOTS];
    unsigned next;
} g = { .cta_budget = OMEGA_GPU_FP8_MAX_CTAS };

const char *omega_gpu_fp8_last_error(void) { return omega_gpu_session_last_error(); }
const char *omega_gpu_fp8_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_FP8_OK: return "OK";
    case OMEGA_GPU_FP8_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_FP8_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_FP8_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_FP8_CHIP_FAIL: return "CHIP_FAIL";
    case OMEGA_GPU_FP8_NAN_WEIGHT: return "NAN_WEIGHT";
    default: return "UNKNOWN";
    }
}

static void cache_clear_locked(void) {
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        if (!g.cache[i].valid) continue;
        omega_gpu_session_free(&g.cache[i].code);
        memset(&g.cache[i], 0, sizeof g.cache[i]);
    }
    g.next = 0;
}
static void on_session_close(void) {
    cache_clear_locked();
    omega_gpu_session_scratch_free(&g.x_buf);
    omega_gpu_session_scratch_free(&g.y_buf);
    omega_gpu_session_free(&g.tab);
    g.sealed = 0;
}
static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}
void omega_gpu_fp8_cache_clear(void) { omega_gpu_session_lock(); cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_fp8_set_shape(uint32_t lb, uint32_t rows, uint32_t warps) {
    omega_gpu_session_lock(); g.lane_bytes = lb; g.rows = rows; g.warps = warps; cache_clear_locked(); omega_gpu_session_unlock();
}
void omega_gpu_fp8_set_cta_budget(uint32_t ctas) { omega_gpu_session_lock(); g.cta_budget = ctas ? ctas : OMEGA_GPU_FP8_MAX_CTAS; cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_fp8_test_set_mutant(int m) { omega_gpu_session_lock(); g.mutant = m; cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_fp8_test_set_no_fold(int on) { g.no_fold = on ? 1 : 0; }
void omega_gpu_fp8_seal(int on) { omega_gpu_session_lock(); g.sealed = on ? 1 : 0; omega_gpu_session_unlock(); }

/* ---- shape choice ---- */

/* Largest lane width dividing k (16 for k % 512 == 0: one LDG.E.128 per row per step); rows per warp so
 * that at least ~512 warps have a row group (bandwidth needs loads in flight), else fewer rows per warp. */
void omega_gpu_fp8_shape_for(uint32_t k, uint32_t n, uint32_t *lane_bytes, uint32_t *rows, uint32_t *warps) {
    uint32_t lb = k % 512u == 0 ? 16u : 8u; /* k % 256 == 0 is required (codegen refuses otherwise) */
    uint32_t r = 8;
    while (r > 1 && n / r < 512u) r /= 2;
    if (lane_bytes) *lane_bytes = g.lane_bytes ? g.lane_bytes : lb;
    if (rows) *rows = g.rows ? g.rows : r;
    if (warps) *warps = g.warps ? g.warps : 8u;
}

/* ---- kernel cache (lock held, device open) ---- */

static int kernel_for(const OmegaFp8GemvSpec *spec, Slot **out, bool *hit) {
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        Slot *s = &g.cache[i];
        if (s->valid && memcmp(&s->spec, spec, sizeof *spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_FP8_OK; }
    }
    if (g.sealed) {
        char e[160];
        omega_gpu_fmt(e, sizeof e, "serving reservation exceeded: fp8 kernel k=%u lb=%u rows=%u warps=%u was not prepared",
                      spec->k, spec->lane_bytes, spec->rows, spec->warps);
        omega_gpu_session_set_error(e);
        return OMEGA_GPU_FP8_TOO_LARGE;
    }
    Slot *s = NULL;
    for (unsigned i = 0; i < CACHE_SLOTS && !s; i++) {
        Slot *c = &g.cache[(g.next + i) % CACHE_SLOTS];
        if (!c->valid || !c->pinned) { s = c; g.next = (unsigned)((c - g.cache) + 1) % CACHE_SLOTS; }
    }
    if (!s) { omega_gpu_session_set_error("fp8 kernel cache: every slot holds a prepared kernel"); return OMEGA_GPU_FP8_TOO_LARGE; }
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("fp8 codegen: host memory"); return OMEGA_GPU_FP8_CODEGEN_FAIL; }
    int cg = omega_blackwell_codegen_fp8_gemv_prog(spec, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("fp8 codegen refused the shape"); return OMEGA_GPU_FP8_CODEGEN_FAIL; }
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_FP8_CHIP_FAIL; }
    OmegaBlackwellKernel kernel;
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need) { omega_gpu_session_free(&code); return OMEGA_GPU_FP8_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_FP8_CHIP_FAIL; }
    if (s->valid) omega_gpu_session_free(&s->code);
    memset(s, 0, sizeof *s);
    s->valid = true; s->spec = *spec; s->kernel = kernel; s->code = code;
    *out = s; *hit = false;
    return OMEGA_GPU_FP8_OK;
}

static void spec_for(uint32_t k, uint32_t n, OmegaFp8GemvSpec *spec) {
    memset(spec, 0, sizeof *spec);
    spec->k = k;
    omega_gpu_fp8_shape_for(k, n, &spec->lane_bytes, &spec->rows, &spec->warps);
    spec->mutant = g.mutant;
}

int omega_gpu_fp8_prepare(uint32_t k) {
    OmegaFp8GemvSpec spec; spec_for(k, 1u << 20, &spec);
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_FP8_OK;
    if (!dev_open_locked()) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    else { Slot *s = NULL; bool hit = false; rc = kernel_for(&spec, &s, &hit); if (rc == OMEGA_GPU_FP8_OK) s->pinned = true; }
    omega_gpu_session_unlock();
    return rc;
}

/* ---- tensors ---- */

void omega_gpu_fp8_tensor_shape(const OmegaGpuFp8Tensor *t, uint32_t *experts, uint32_t *n, uint32_t *k) {
    if (experts) *experts = t ? t->experts : 0;
    if (n) *n = t ? t->n : 0;
    if (k) *k = t ? t->k : 0;
}

int omega_gpu_fp8_tensor_upload(uint32_t experts, uint32_t n, uint32_t k, const uint8_t *w, const uint16_t *scale, OmegaGpuFp8Tensor **out) {
    if (out) *out = NULL;
    if (!w || !scale || !out || experts == 0 || n == 0 || k == 0 || n % OMEGA_FP8_BLOCK || k % OMEGA_FP8_BLOCK) return OMEGA_GPU_FP8_BAD_ARGS;
    if (k > OMEGA_FP8_MAX_K) return OMEGA_GPU_FP8_TOO_LARGE;
    const size_t wbytes = (size_t)experts * n * k, scount = (size_t)experts * (n / OMEGA_FP8_BLOCK) * (k / OMEGA_FP8_BLOCK);
    size_t first = 0;
    if (omega_fp8_pack_scan_nan(w, wbytes, &first)) {
        char e[120]; omega_gpu_fmt(e, sizeof e, "fp8 upload: e4m3 NaN code at byte %zu", first);
        omega_gpu_session_set_error(e);
        return OMEGA_GPU_FP8_NAN_WEIGHT;
    }
    OmegaGpuFp8Tensor *t = omega_gpu_host_calloc(1, sizeof *t);
    if (!t) return OMEGA_GPU_FP8_BAD_ARGS;
    t->experts = experts; t->n = n; t->k = k;
    int rc = OMEGA_GPU_FP8_OK;
    omega_gpu_session_lock();
    if (!dev_open_locked() || omega_gpu_session_alloc(wbytes, &t->w) != 0 || omega_gpu_session_alloc(scount * 4u, &t->s) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    omega_gpu_session_unlock();
    if (rc == OMEGA_GPU_FP8_OK) {
        memcpy(t->w.cpu, w, wbytes);
        if (omega_fp8_pack_scales(scale, scount, t->s.cpu, g.no_fold) != 0) { omega_gpu_session_set_error("fp8 upload: NaN, inf or overflowing scale"); rc = OMEGA_GPU_FP8_BAD_ARGS; }
        else if (omega_gpu_session_sync_to_gpu(&t->w) != 0 || omega_gpu_session_sync_to_gpu(&t->s) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    }
    if (rc != OMEGA_GPU_FP8_OK) { omega_gpu_fp8_tensor_free(t); return rc; }
    *out = t;
    return OMEGA_GPU_FP8_OK;
}

void omega_gpu_fp8_tensor_free(OmegaGpuFp8Tensor *t) {
    if (!t) return;
    omega_gpu_session_lock();
    if (t->w.cpu) omega_gpu_session_free(&t->w);
    if (t->s.cpu) omega_gpu_session_free(&t->s);
    omega_gpu_session_unlock();
    omega_gpu_host_free(t);
}

/* ---- the GEMV (lock held inside) ---- */

static int gemv_core(const OmegaGpuFp8Tensor *t, uint32_t groups, const uint32_t *ids, const float *x32, const uint16_t *x16,
                     uint32_t x_rows, float *y, OmegaGpuFp8Info *info) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    if (!t || !t->w.cpu || !ids || (!x32 && !x16) || !y || groups == 0 || (x_rows != 1 && x_rows != groups)) return OMEGA_GPU_FP8_BAD_ARGS;
    if (groups > OMEGA_GPU_FP8_MAX_GROUPS) return OMEGA_GPU_FP8_TOO_LARGE;
    for (uint32_t gi = 0; gi < groups; gi++) if (ids[gi] >= t->experts) return OMEGA_GPU_FP8_BAD_ARGS;
    const uint32_t n = t->n, k = t->k;
    OmegaFp8GemvSpec spec; spec_for(k, n, &spec);
    if (n % spec.rows) return OMEGA_GPU_FP8_CODEGEN_FAIL;
    const uint32_t RG = n / spec.rows;
    uint32_t grid_x = (RG + spec.warps - 1) / spec.warps;
    if (grid_x > g.cta_budget) grid_x = g.cta_budget;
    if (grid_x == 0) grid_x = 1;

    omega_gpu_session_lock();
    int rc = OMEGA_GPU_FP8_OK;
    if (!dev_open_locked()) { omega_gpu_session_unlock(); return OMEGA_GPU_FP8_CHIP_FAIL; }
    if (!g.tab.cpu && omega_gpu_session_alloc((size_t)OMEGA_GPU_FP8_MAX_GROUPS * 16u, &g.tab) != 0) { omega_gpu_session_unlock(); return OMEGA_GPU_FP8_CHIP_FAIL; }
    int src = omega_gpu_session_scratch(&g.x_buf, (size_t)x_rows * k * 4u);
    if (src == 0) src = omega_gpu_session_scratch(&g.y_buf, (size_t)groups * n * 4u);
    if (src != 0) { omega_gpu_session_unlock(); return src == OMEGA_GPU_SESSION_SCRATCH_RESERVED ? OMEGA_GPU_FP8_TOO_LARGE : OMEGA_GPU_FP8_CHIP_FAIL; }
    Slot *ks = NULL; bool hit = false;
    rc = kernel_for(&spec, &ks, &hit);
    if (rc != OMEGA_GPU_FP8_OK) { omega_gpu_session_unlock(); return rc; }

    /* group table: {w_va, s_va} per group */
    uint64_t *tab = g.tab.cpu;
    const uint64_t wstride = (uint64_t)n * k, sstride = (uint64_t)(n / OMEGA_FP8_BLOCK) * (k / OMEGA_FP8_BLOCK) * 4u;
    for (uint32_t gi = 0; gi < groups; gi++) { tab[2 * gi] = t->w.va + ids[gi] * wstride; tab[2 * gi + 1] = t->s.va + ids[gi] * sstride; }
    /* x' rows, y poisoned */
    float *xd = g.x_buf.mem.cpu, *yd = g.y_buf.mem.cpu;
    for (uint32_t r = 0; r < x_rows; r++) {
        if (x32) omega_fp8_pack_x_f32(x32 + (size_t)r * k, k, xd + (size_t)r * k);
        else omega_fp8_pack_x_bf16(x16 + (size_t)r * k, k, xd + (size_t)r * k);
    }
    for (size_t i = 0; i < (size_t)groups * n; i++) yd[i] = POISON_F32;
    if (omega_gpu_session_sync_to_gpu(&g.tab) != 0 || omega_gpu_session_sync_to_gpu(&g.x_buf.mem) != 0 || omega_gpu_session_sync_to_gpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;

    uint32_t args[OMEGA_FP8_ARGS_WORDS];
    args[0] = (uint32_t)g.tab.va; args[1] = (uint32_t)(g.tab.va >> 32);
    args[2] = (uint32_t)g.x_buf.mem.va; args[3] = (uint32_t)(g.x_buf.mem.va >> 32);
    args[4] = (uint32_t)g.y_buf.mem.va; args[5] = (uint32_t)(g.y_buf.mem.va >> 32);
    args[6] = RG; args[7] = x_rows == 1 ? 0u : k * 4u; args[8] = n * 4u; args[9] = grid_x * spec.warps;
    if (info) {
        info->experts = t->experts; info->n = n; info->k = k; info->groups = groups;
        info->lane_bytes = spec.lane_bytes; info->rows_per_warp = spec.rows; info->warps_per_cta = spec.warps;
        info->grid_x = grid_x; info->grid_y = groups; info->steps = omega_fp8_gemv_steps(&spec);
        info->weight_bytes = (uint64_t)groups * wstride; info->scale_bytes = (uint64_t)groups * sstride;
        info->kernel_cache_hit = hit;
        (void)omega_gpu_fmt(info->target_chip, sizeof info->target_chip, "%s", "NVIDIA DGX Spark (Grace Blackwell GB10)");
        info->sm_architecture = 121;
    }
    if (rc == OMEGA_GPU_FP8_OK) {
        OmegaGpuLaunch L = {
            .code_va = ks->code.va, .gpr_count = ks->kernel.gpr_count,
            .threads_x = 32u * spec.warps, .threads_y = 1, .grid_x = grid_x, .grid_y = groups,
            .num_elements = (size_t)groups * n,
            .args = args, .n_args = OMEGA_FP8_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
        };
        if (omega_gpu_session_launch(&L, info ? &info->elapsed_ns : NULL, info ? &info->completion_marker : NULL) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_FP8_OK && omega_gpu_session_sync_to_cpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    if (rc == OMEGA_GPU_FP8_OK) {
        size_t unchanged = 0;
        for (size_t i = 0; i < (size_t)groups * n; i++) if (yd[i] == POISON_F32) unchanged++;
        if (unchanged) { omega_gpu_session_set_error("output poison survived"); rc = OMEGA_GPU_FP8_CHIP_FAIL; }
        else memcpy(y, yd, (size_t)groups * n * sizeof *y);
    }
    omega_gpu_session_unlock();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}

int omega_gpu_fp8_gemv_f32(const OmegaGpuFp8Tensor *t, uint32_t expert, const float *x, float *y, OmegaGpuFp8Info *info) {
    return gemv_core(t, 1, &expert, x, NULL, 1, y, info);
}
int omega_gpu_fp8_gemv_bf16(const OmegaGpuFp8Tensor *t, uint32_t expert, const uint16_t *x, float *y, OmegaGpuFp8Info *info) {
    return gemv_core(t, 1, &expert, NULL, x, 1, y, info);
}
int omega_gpu_fp8_gemv_grouped_f32(const OmegaGpuFp8Tensor *t, uint32_t groups, const uint32_t *ids, const float *x, uint32_t x_rows, float *y, OmegaGpuFp8Info *info) {
    return gemv_core(t, groups, ids, x, NULL, x_rows, y, info);
}
