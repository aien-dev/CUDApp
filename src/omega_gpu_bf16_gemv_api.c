/* omega_gpu_bf16_gemv_api.c: bf16 streaming GEMV (lm_head) on the shared session, OWN-30B 2026-10-13.
 * See the header. Submission path and buffer policy are the FP8 API's: every buffer both sides touch is
 * GPU-uncached (nvos.h NVOS32_ATTR2_GPU_CACHEABLE: "For system memory this will not be coherent with direct CPU
 * mappings"); one launch = omega_gpu_session_launch. */
#include "omega_gpu_bf16_gemv_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_fmt.h"
#include "omega_gpu_cmdlist.h"
#include <string.h>

#define CACHE_SLOTS 8
#define MARKER_TIMEOUT_MS 20000ull
#define POISON_F32 (-999.0f)
#define DEFAULT_CTAS 96u

struct OmegaGpuBf16Tensor {
    uint32_t n, k;
    OmegaGpuMem w;
};

typedef struct {
    bool valid, pinned;
    OmegaBf16GemvSpec spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
} Slot;

static struct {
    OmegaGpuScratch x_buf, y_buf;
    uint32_t rows, warps, depth, ctas; /* overrides, 0 = automatic */
    int mutant;
    bool hooked;
    Slot cache[CACHE_SLOTS];
    unsigned next;
} g;
static int g_rec; /* 1 while a record call runs (lock held): kernel_for must not evict */

const char *omega_gpu_bf16_last_error(void) { return omega_gpu_session_last_error(); }
const char *omega_gpu_bf16_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_BF16_OK: return "OK";
    case OMEGA_GPU_BF16_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_BF16_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_BF16_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_BF16_CHIP_FAIL: return "CHIP_FAIL";
    case OMEGA_GPU_BF16_NAN_WEIGHT: return "NAN_WEIGHT";
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
}
static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}
void omega_gpu_bf16_cache_clear(void) { omega_gpu_session_lock(); cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_bf16_set_shape(uint32_t rows, uint32_t warps, uint32_t depth) {
    omega_gpu_session_lock(); g.rows = rows; g.warps = warps; g.depth = depth; cache_clear_locked(); omega_gpu_session_unlock();
}
void omega_gpu_bf16_set_cta_budget(uint32_t ctas) { omega_gpu_session_lock(); g.ctas = ctas; omega_gpu_session_unlock(); }
void omega_gpu_bf16_test_set_mutant(int m) { omega_gpu_session_lock(); g.mutant = m; cache_clear_locked(); omega_gpu_session_unlock(); }

/* Declared registers: peak + 1 rounded up to 8; a CTA must fit the 65536-register file (chip-found rules, see
 * omega_gpu_fp8_api.c and ~/handoffs/own30b-K-report.md). */
static uint32_t gpr_declared(uint32_t peak) { return omega_bw_gpr_declared(peak); } /* peak + 2 rounded up to 8: the chip hung at peak + 1 (omega_blackwell_codegen.h) */
static bool register_file_fits(uint32_t warps, uint32_t peak) {
    uint32_t decl = gpr_declared(peak);
    if ((uint64_t)32u * warps * decl <= 65536u) return true;
    char e[160]; omega_gpu_fmt(e, sizeof e, "bf16 kernel: %u warps x %u declared registers exceed the 65536-register file", warps, decl);
    omega_gpu_session_set_error(e);
    return false;
}
int omega_gpu_bf16_kernel_fits(const OmegaBf16GemvSpec *spec, uint32_t *declared) {
    uint32_t peak = 0;
    if (!spec || omega_bf16_gemv_prog_stats(spec, NULL, &peak) != 0) { omega_gpu_session_set_error("bf16 codegen refused the shape"); return OMEGA_GPU_BF16_CODEGEN_FAIL; }
    if (declared) *declared = gpr_declared(peak);
    return register_file_fits(spec->warps, peak) ? OMEGA_GPU_BF16_OK : OMEGA_GPU_BF16_TOO_LARGE;
}

/* Automatic shape. DEFAULTS_BEGIN: set from the chip sweep (evidence OWN30B-BF16-*); see the report. */
void omega_gpu_bf16_shape_for(uint32_t k, uint32_t *rows, uint32_t *warps, uint32_t *depth, uint32_t *ctas) {
    uint32_t ks = k / 256u, w = 16u, r = 1u, d = ks;
    while (d > 16u || ks % d || r * d > 16u) d--;
    if (k % (32u * w) || k > 512u * w) w = 32u;
    if (rows) *rows = g.rows ? g.rows : r;
    if (warps) *warps = g.warps ? g.warps : w;
    if (depth) *depth = g.depth ? g.depth : d;
    if (ctas) *ctas = g.ctas ? g.ctas : DEFAULT_CTAS;
}
static void spec_for(uint32_t k, OmegaBf16GemvSpec *spec, uint32_t *ctas) {
    memset(spec, 0, sizeof *spec);
    spec->k = k;
    omega_gpu_bf16_shape_for(k, &spec->rows, &spec->warps, &spec->depth, ctas);
    spec->mutant = g.mutant;
}

static int kernel_for(const OmegaBf16GemvSpec *spec, Slot **out, bool *hit) {
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        Slot *s = &g.cache[i];
        if (s->valid && memcmp(&s->spec, spec, sizeof *spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_BF16_OK; }
    }
    Slot *s = NULL;
    for (unsigned i = 0; i < CACHE_SLOTS && !s; i++) {
        Slot *c = &g.cache[(g.next + i) % CACHE_SLOTS];
        if (!c->valid || !c->pinned) { s = c; g.next = (unsigned)((c - g.cache) + 1) % CACHE_SLOTS; }
    }
    if (!s) { omega_gpu_session_set_error("bf16 kernel cache: every slot holds a prepared kernel"); return OMEGA_GPU_BF16_TOO_LARGE; }
    if (g_rec && s->valid) { omega_gpu_session_set_error("bf16 record: kernel cache miss would evict a resident kernel (warm up before recording)"); return OMEGA_GPU_BF16_TOO_LARGE; }
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("bf16 codegen: host memory"); return OMEGA_GPU_BF16_CODEGEN_FAIL; }
    int cg = omega_blackwell_codegen_bf16_gemv_prog(spec, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("bf16 codegen refused the shape"); return OMEGA_GPU_BF16_CODEGEN_FAIL; }
    if (!register_file_fits(spec->warps, prog->regalloc.peak_gpr_usage)) { omega_gpu_host_free(prog); return OMEGA_GPU_BF16_TOO_LARGE; }
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_BF16_CHIP_FAIL; }
    OmegaBlackwellKernel kernel;
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need) { omega_gpu_session_free(&code); return OMEGA_GPU_BF16_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_BF16_CHIP_FAIL; }
    if (s->valid) omega_gpu_session_free(&s->code);
    memset(s, 0, sizeof *s);
    s->valid = true; s->spec = *spec; s->kernel = kernel; s->code = code;
    *out = s; *hit = false;
    return OMEGA_GPU_BF16_OK;
}

int omega_gpu_bf16_prepare(uint32_t k) {
    OmegaBf16GemvSpec spec; uint32_t ctas; spec_for(k, &spec, &ctas);
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_BF16_OK;
    if (!dev_open_locked()) rc = OMEGA_GPU_BF16_CHIP_FAIL;
    else { Slot *s = NULL; bool hit = false; rc = kernel_for(&spec, &s, &hit); if (rc == OMEGA_GPU_BF16_OK) s->pinned = true; }
    omega_gpu_session_unlock();
    return rc;
}

void omega_gpu_bf16_tensor_shape(const OmegaGpuBf16Tensor *t, uint32_t *n, uint32_t *k) {
    if (n) *n = t ? t->n : 0;
    if (k) *k = t ? t->k : 0;
}

int omega_gpu_bf16_tensor_upload(uint32_t n, uint32_t k, const uint16_t *w, OmegaGpuBf16Tensor **out) {
    if (out) *out = NULL;
    if (!w || !out || n == 0 || k == 0 || k % 256u) return OMEGA_GPU_BF16_BAD_ARGS;
    if (k > OMEGA_BF16_MAX_K) return OMEGA_GPU_BF16_TOO_LARGE;
    const size_t count = (size_t)n * k;
    for (size_t i = 0; i < count; i++) if ((w[i] & 0x7f80u) == 0x7f80u) {
        char e[120]; omega_gpu_fmt(e, sizeof e, "bf16 upload: NaN or infinity at element %zu", i);
        omega_gpu_session_set_error(e);
        return OMEGA_GPU_BF16_NAN_WEIGHT;
    }
    OmegaGpuBf16Tensor *t = omega_gpu_host_calloc(1, sizeof *t);
    if (!t) return OMEGA_GPU_BF16_BAD_ARGS;
    t->n = n; t->k = k;
    int rc = OMEGA_GPU_BF16_OK;
    omega_gpu_session_lock();
    if (!dev_open_locked() || omega_gpu_session_alloc(count * 2u, &t->w) != 0) rc = OMEGA_GPU_BF16_CHIP_FAIL;
    omega_gpu_session_unlock();
    if (rc == OMEGA_GPU_BF16_OK) {
        memcpy(t->w.cpu, w, count * 2u);
        if (omega_gpu_session_sync_to_gpu(&t->w) != 0) rc = OMEGA_GPU_BF16_CHIP_FAIL;
    }
    if (rc != OMEGA_GPU_BF16_OK) { omega_gpu_bf16_tensor_free(t); return rc; }
    *out = t;
    return OMEGA_GPU_BF16_OK;
}

void omega_gpu_bf16_tensor_free(OmegaGpuBf16Tensor *t) {
    if (!t) return;
    omega_gpu_session_lock();
    if (t->w.cpu) omega_gpu_session_free(&t->w);
    omega_gpu_session_unlock();
    omega_gpu_host_free(t);
}

/* rec non-NULL (record path, caller holds the session lock): x and y are the device buffers dev_x / dev_y, the launch is
 * appended to rec instead of submitted, nothing is staged, poisoned or read back. */
static int gemv_core(const OmegaGpuBf16Tensor *t, const float *x32, const uint16_t *x16, float *y, OmegaGpuBf16Info *info,
                     OmegaGpuCmdList *rec, uint64_t dev_x, uint64_t dev_y, uint32_t xrows) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    if (rec && (!dev_x || !dev_y)) return OMEGA_GPU_BF16_BAD_ARGS;
    if (!t || !t->w.cpu || (!rec && ((!x32 && !x16) || !y))) return OMEGA_GPU_BF16_BAD_ARGS;
    const uint32_t n = t->n, k = t->k;
    if (xrows && (!rec || xrows > 65535u)) return OMEGA_GPU_BF16_BAD_ARGS; /* the ROWS form is record-only */
    OmegaBf16GemvSpec spec; uint32_t ctas; spec_for(k, &spec, &ctas);
    if (xrows) spec.ystride = n;
    if (n % spec.rows) return OMEGA_GPU_BF16_CODEGEN_FAIL;
    const uint32_t RG = n / spec.rows;
    uint32_t grid_x = (RG + spec.warps - 1) / spec.warps;
    if (grid_x > ctas) grid_x = ctas;
    if (grid_x == 0) grid_x = 1;

    if (!rec) omega_gpu_session_lock();
#define UNLOCK() do { if (!rec) omega_gpu_session_unlock(); } while (0)
    int rc = OMEGA_GPU_BF16_OK;
    if (!dev_open_locked()) { UNLOCK(); return OMEGA_GPU_BF16_CHIP_FAIL; }
    if (!rec) {
        int src = omega_gpu_session_scratch(&g.x_buf, (size_t)k * 4u);
        if (src == 0) src = omega_gpu_session_scratch(&g.y_buf, (size_t)n * 4u);
        if (src != 0) { UNLOCK(); return src == OMEGA_GPU_SESSION_SCRATCH_RESERVED ? OMEGA_GPU_BF16_TOO_LARGE : OMEGA_GPU_BF16_CHIP_FAIL; }
    }
    Slot *ks = NULL; bool hit = false;
    g_rec = rec != NULL;
    rc = kernel_for(&spec, &ks, &hit);
    g_rec = 0;
    if (rc != OMEGA_GPU_BF16_OK) { UNLOCK(); return rc; }

    float *xd = g.x_buf.mem.cpu, *yd = g.y_buf.mem.cpu;
    if (!rec) {
        if (x32) memcpy(xd, x32, (size_t)k * 4u);
        else for (uint32_t i = 0; i < k; i++) { uint32_t b = (uint32_t)x16[i] << 16; memcpy(&xd[i], &b, 4); }
        for (size_t i = 0; i < n; i++) yd[i] = POISON_F32;
        if (omega_gpu_session_sync_to_gpu(&g.x_buf.mem) != 0 || omega_gpu_session_sync_to_gpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_BF16_CHIP_FAIL;
    }

    uint32_t args[OMEGA_BF16_ARGS_WORDS];
    const uint64_t x_va = rec ? dev_x : g.x_buf.mem.va, y_va = rec ? dev_y : g.y_buf.mem.va;
    args[0] = (uint32_t)t->w.va; args[1] = (uint32_t)(t->w.va >> 32);
    args[2] = (uint32_t)x_va; args[3] = (uint32_t)(x_va >> 32);
    args[4] = (uint32_t)y_va; args[5] = (uint32_t)(y_va >> 32);
    args[6] = RG; args[7] = grid_x * spec.warps;
    if (info) {
        info->n = n; info->k = k; info->rows_per_warp = spec.rows; info->warps_per_cta = spec.warps; info->depth = spec.depth;
        info->grid_x = grid_x; info->steps = omega_bf16_gemv_steps(&spec);
        info->weight_bytes = (uint64_t)n * k * 2u;
        info->kernel_cache_hit = hit;
        (void)omega_gpu_fmt(info->target_chip, sizeof info->target_chip, "%s", "NVIDIA DGX Spark (Grace Blackwell GB10)");
        info->sm_architecture = 121;
    }
    if (rc == OMEGA_GPU_BF16_OK) {
        OmegaGpuLaunch L = {
            .code_va = ks->code.va, .gpr_count = gpr_declared(ks->kernel.gpr_count),
            .threads_x = 32u * spec.warps, .threads_y = 1, .grid_x = grid_x, .grid_y = xrows ? xrows : 1u,
            .num_elements = (size_t)n * (xrows ? xrows : 1u),
            .args = args, .n_args = OMEGA_BF16_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
            .shared_bytes = omega_bf16_gemv_shared_bytes(&spec),
        };
        if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); rc = OMEGA_GPU_BF16_CHIP_FAIL; }
        else if (rec) { if (omega_gpu_cmdlist_add(rec, &L) < 0) rc = OMEGA_GPU_BF16_CHIP_FAIL; }
        else if (omega_gpu_session_launch(&L, info ? &info->elapsed_ns : NULL, info ? &info->completion_marker : NULL) != 0) rc = OMEGA_GPU_BF16_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_BF16_OK && !rec && omega_gpu_session_sync_to_cpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_BF16_CHIP_FAIL;
    if (rc == OMEGA_GPU_BF16_OK && !rec) {
        size_t unchanged = 0;
        for (size_t i = 0; i < n; i++) if (yd[i] == POISON_F32) unchanged++;
        if (unchanged) { omega_gpu_session_set_error("output poison survived"); rc = OMEGA_GPU_BF16_CHIP_FAIL; }
        else memcpy(y, yd, (size_t)n * sizeof *y);
    }
    UNLOCK();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}
#undef UNLOCK

int omega_gpu_bf16_gemv_f32(const OmegaGpuBf16Tensor *t, const float *x, float *y, OmegaGpuBf16Info *info) { return gemv_core(t, x, NULL, y, info, NULL, 0, 0, 0); }
int omega_gpu_bf16_gemv_bf16(const OmegaGpuBf16Tensor *t, const uint16_t *x, float *y, OmegaGpuBf16Info *info) { return gemv_core(t, NULL, x, y, info, NULL, 0, 0, 0); }

/* Record path (OWN-30B W-L2a decode graph): see the header. Caller holds omega_gpu_session_lock(). */
int omega_gpu_bf16_gemv_record(OmegaGpuCmdList *cl, const OmegaGpuBf16Tensor *t, uint64_t x_va, uint64_t y_va) {
    if (!cl) return OMEGA_GPU_BF16_BAD_ARGS;
    return gemv_core(t, NULL, NULL, NULL, NULL, cl, x_va, y_va, 0);
}

/* ROWS form record path (sc #642 prefill): see the header. Caller holds omega_gpu_session_lock(). */
int omega_gpu_bf16_gemv_rows_record(OmegaGpuCmdList *cl, const OmegaGpuBf16Tensor *t, uint32_t rows, uint64_t x_va, uint64_t y_va) {
    if (!cl || rows == 0) return OMEGA_GPU_BF16_BAD_ARGS;
    return gemv_core(t, NULL, NULL, NULL, NULL, cl, x_va, y_va, rows);
}
