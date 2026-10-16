/* omega_gpu_tern_api.c: KT1 W2A8 ternary GEMV on the shared session, ternary engine stage T0 (2026-10-15). See the header.
 * Buffer policy: weights GPU-cached by default (written once by the CPU before any GPU read, then only read by the GPU;
 * omega_gpu_mem_cached.h). Everything else both sides touch (x, a_scale, scales, y) is GPU-uncached (nvos.h
 * NVOS32_ATTR2_GPU_CACHEABLE: "For system memory this will not be coherent with direct CPU mappings"). Submission path
 * and kernel cache are the bf16 API's. */
#include "omega_gpu_tern_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_mem_cached.h"
#include "omega_gpu_fmt.h"
#include "omega_gpu_cmdlist.h"
#include <math.h>
#include <string.h>

#define CACHE_SLOTS 8
#define MARKER_TIMEOUT_MS 20000ull
#define DEFAULT_CTAS 160u /* T0 sweep: 160 beat 96 and 256 on the four BitNet shapes (row-major, rows 1) */
#define DEFAULT_WARPS 8u

struct OmegaGpuTernTensor {
    uint32_t n, k, n_padded, layout, R;
    bool cached;
    OmegaGpuTernSegments seg;
    OmegaGpuMem w, ws;
};

typedef struct {
    bool valid;
    OmegaTernGemvSpec spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
} Slot;

static struct {
    OmegaGpuScratch x_buf, y_buf, as_buf;
    uint32_t rows, warps, depth, ctas; /* overrides, 0 = automatic */
    uint32_t layout, block_rows;       /* upload layout; block_rows 0 = 4 */
    int uncached;                      /* 1 = uploads use uncached weights */
    int mutant;
    bool hooked;
    Slot cache[CACHE_SLOTS];
    unsigned next;
} g = { .layout = OMEGA_TERN_LAYOUT_ROWMAJOR };
static int g_rec; /* 1 while a record call runs (lock held): kernel_for must not evict */

const char *omega_gpu_tern_last_error(void) { return omega_gpu_session_last_error(); }
const char *omega_gpu_tern_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_TERN_OK: return "OK";
    case OMEGA_GPU_TERN_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_TERN_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_TERN_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_TERN_CHIP_FAIL: return "CHIP_FAIL";
    case OMEGA_GPU_TERN_BAD_TRIT: return "BAD_TRIT";
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
    omega_gpu_session_scratch_free(&g.as_buf);
}
static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}
void omega_gpu_tern_cache_clear(void) { omega_gpu_session_lock(); cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_tern_set_layout(uint32_t layout, uint32_t block_rows) { omega_gpu_session_lock(); g.layout = layout; g.block_rows = block_rows; omega_gpu_session_unlock(); }
void omega_gpu_tern_set_shape(uint32_t rows, uint32_t warps, uint32_t depth) {
    omega_gpu_session_lock(); g.rows = rows; g.warps = warps; g.depth = depth; cache_clear_locked(); omega_gpu_session_unlock();
}
void omega_gpu_tern_set_cta_budget(uint32_t ctas) { omega_gpu_session_lock(); g.ctas = ctas; omega_gpu_session_unlock(); }
void omega_gpu_tern_set_weight_cache(int cached) { omega_gpu_session_lock(); g.uncached = !cached; omega_gpu_session_unlock(); }
void omega_gpu_tern_test_set_mutant(int m) { omega_gpu_session_lock(); g.mutant = m; cache_clear_locked(); omega_gpu_session_unlock(); }

static uint32_t gpr_declared(uint32_t peak) { return omega_bw_gpr_declared(peak); }
static bool register_file_fits(uint32_t warps, uint32_t peak) {
    uint32_t decl = gpr_declared(peak);
    if ((uint64_t)32u * warps * decl <= 65536u) return true;
    char e[160]; omega_gpu_fmt(e, sizeof e, "tern kernel: %u warps x %u declared registers exceed the 65536-register file", warps, decl);
    omega_gpu_session_set_error(e);
    return false;
}
int omega_gpu_tern_kernel_fits(const OmegaTernGemvSpec *spec, uint32_t *declared) {
    uint32_t peak = 0;
    if (!spec || omega_tern_gemv_prog_stats(spec, NULL, &peak) != 0) { omega_gpu_session_set_error("tern codegen refused the shape"); return OMEGA_GPU_TERN_CODEGEN_FAIL; }
    if (declared) *declared = gpr_declared(peak);
    return register_file_fits(spec->warps, peak) ? OMEGA_GPU_TERN_OK : OMEGA_GPU_TERN_TOO_LARGE;
}

/* Automatic shape. Defaults from the T0 chip sweep (evidence TERN-T0-*). */
void omega_gpu_tern_shape_for(uint32_t k, uint32_t layout, uint32_t R, uint32_t *rows, uint32_t *warps, uint32_t *depth, uint32_t *ctas) {
    uint32_t nch = (k + OMEGA_TERN_CHUNK_MAX - 1u) / OMEGA_TERN_CHUNK_MAX;
    uint32_t ksc = nch ? k / nch / 256u : 0;
    uint32_t r = layout == OMEGA_TERN_LAYOUT_BLOCK ? R : (g.rows ? g.rows : 1u);
    uint32_t d = g.depth ? g.depth : 16u / r;
    while (d > 1u && (ksc % d || r * d > 16u)) d--;
    if (rows) *rows = r;
    if (warps) *warps = g.warps ? g.warps : DEFAULT_WARPS;
    if (depth) *depth = d;
    if (ctas) *ctas = g.ctas ? g.ctas : DEFAULT_CTAS;
}

static int kernel_for(const OmegaTernGemvSpec *spec, Slot **out, bool *hit) {
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        Slot *s = &g.cache[i];
        if (s->valid && memcmp(&s->spec, spec, sizeof *spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_TERN_OK; }
    }
    Slot *s = &g.cache[g.next];
    if (g_rec && s->valid) { omega_gpu_session_set_error("tern record: kernel cache miss would evict a resident kernel (warm up before recording)"); return OMEGA_GPU_TERN_TOO_LARGE; }
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("tern codegen: host memory"); return OMEGA_GPU_TERN_CODEGEN_FAIL; }
    int cg = omega_blackwell_codegen_tern_gemv_prog(spec, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("tern codegen refused the shape"); return OMEGA_GPU_TERN_CODEGEN_FAIL; }
    if (!register_file_fits(spec->warps, prog->regalloc.peak_gpr_usage)) { omega_gpu_host_free(prog); return OMEGA_GPU_TERN_TOO_LARGE; }
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_TERN_CHIP_FAIL; }
    OmegaBlackwellKernel kernel;
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need) { omega_gpu_session_free(&code); return OMEGA_GPU_TERN_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_TERN_CHIP_FAIL; }
    if (s->valid) omega_gpu_session_free(&s->code);
    memset(s, 0, sizeof *s);
    s->valid = true; s->spec = *spec; s->kernel = kernel; s->code = code;
    g.next = (g.next + 1u) % CACHE_SLOTS;
    *out = s; *hit = false;
    return OMEGA_GPU_TERN_OK;
}

uint32_t omega_gpu_tern_padded_rows(uint32_t n) { return (n + OMEGA_TERN_ROW_PAD - 1u) / OMEGA_TERN_ROW_PAD * OMEGA_TERN_ROW_PAD; }

int omega_gpu_tern_pack(uint32_t n, uint32_t k, const int8_t *w, uint32_t layout, uint32_t R, uint32_t *out, uint32_t n_padded) {
    if (!w || !out || n == 0 || k % 16u || n_padded < n || n_padded % (2u * (layout == OMEGA_TERN_LAYOUT_BLOCK ? R : 1u))) return OMEGA_GPU_TERN_BAD_ARGS;
    const uint32_t kw = k / 16u;
    memset(out, 0, (size_t)n_padded * kw * 4u);
    for (uint32_t r = 0; r < n; r++) {
        const int8_t *row = w + (size_t)r * k;
        for (uint32_t c = 0; c < kw; c++) {
            uint32_t word = 0;
            for (uint32_t i = 0; i < 16u; i++) {
                const int8_t v = row[16u * c + i];
                const uint32_t code = v == 1 ? 1u : v == -1 ? 2u : v == 0 ? 0u : 3u;
                if (code == 3u) return OMEGA_GPU_TERN_BAD_TRIT;
                word |= code << (2u * i);
            }
            out[omega_tern_word_offset(layout, R, k, r, c) / 4u] = word;
        }
    }
    return OMEGA_GPU_TERN_OK;
}

int omega_gpu_tern_segment_bounds(uint32_t n, const OmegaGpuTernSegments *seg, int mutant, uint32_t b[3]) {
    const uint32_t cnt = seg ? seg->count : 1u;
    if (cnt < 1u || cnt > 4u) return OMEGA_GPU_TERN_BAD_ARGS;
    for (uint32_t i = 0; i < 3; i++) {
        if (i + 1u >= cnt) b[i] = n;
        else if (mutant == 2) b[i] = (i + 1u) * (n / cnt); /* TEST ONLY: ws_idx = r / (n / count) */
        else b[i] = seg->end[i];
    }
    return OMEGA_GPU_TERN_OK;
}

static int seg_ok(uint32_t n, const OmegaGpuTernSegments *seg) {
    if (!seg) return 1;
    if (seg->count < 1u || seg->count > 4u || seg->end[seg->count - 1u] != n) return 0;
    for (uint32_t i = 0; i < seg->count; i++) {
        if (!isfinite(seg->scale[i])) return 0;
        if (seg->end[i] == 0 || (i && seg->end[i] <= seg->end[i - 1u])) return 0;
    }
    return 1;
}

void omega_gpu_tern_tensor_shape(const OmegaGpuTernTensor *t, uint32_t *n, uint32_t *k) {
    if (n) *n = t ? t->n : 0;
    if (k) *k = t ? t->k : 0;
}
uint64_t omega_gpu_tern_tensor_bytes(const OmegaGpuTernTensor *t) { return t ? (uint64_t)t->n_padded * t->k / 4u : 0; }

int omega_gpu_tern_tensor_upload(uint32_t n, uint32_t k, const int8_t *w, const OmegaGpuTernSegments *seg, OmegaGpuTernTensor **out) {
    if (out) *out = NULL;
    if (!w || !out || n == 0 || n > 0x40000000u || k == 0 || k % 256u || !seg_ok(n, seg)) return OMEGA_GPU_TERN_BAD_ARGS;
    if (k > OMEGA_TERN_MAX_K || (k > OMEGA_TERN_CHUNK_MAX && k % 512u)) return OMEGA_GPU_TERN_TOO_LARGE;
    omega_gpu_session_lock();
    const uint32_t layout = g.layout, R = g.block_rows ? g.block_rows : 4u; const int uncached = g.uncached;
    omega_gpu_session_unlock();
    if (layout > OMEGA_TERN_LAYOUT_BLOCK || (layout == OMEGA_TERN_LAYOUT_BLOCK && R != 2u && R != 4u)) return OMEGA_GPU_TERN_BAD_ARGS;
    const size_t count = (size_t)n * k;
    for (size_t i = 0; i < count; i++) if (w[i] < -1 || w[i] > 1) {
        char e[120]; omega_gpu_fmt(e, sizeof e, "tern upload: value %d at element %zu is not a trit", (int)w[i], i);
        omega_gpu_session_set_error(e);
        return OMEGA_GPU_TERN_BAD_TRIT;
    }
    OmegaGpuTernTensor *t = omega_gpu_host_calloc(1, sizeof *t);
    if (!t) return OMEGA_GPU_TERN_BAD_ARGS;
    t->n = n; t->k = k; t->n_padded = omega_gpu_tern_padded_rows(n); t->layout = layout; t->R = R; t->cached = !uncached;
    if (seg) t->seg = *seg; else { t->seg.count = 1; t->seg.end[0] = n; t->seg.scale[0] = 1.0f; }
    const size_t bytes = (size_t)t->n_padded * k / 4u;
    int rc = OMEGA_GPU_TERN_OK;
    omega_gpu_session_lock();
    if (!dev_open_locked() || (uncached ? omega_gpu_session_alloc(bytes, &t->w) : omega_gpu_session_alloc_cached(bytes, &t->w)) != 0
        || omega_gpu_session_alloc(16u, &t->ws) != 0) rc = OMEGA_GPU_TERN_CHIP_FAIL;
    omega_gpu_session_unlock();
    if (rc == OMEGA_GPU_TERN_OK) rc = omega_gpu_tern_pack(n, k, w, layout, R, t->w.cpu, t->n_padded);
    if (rc == OMEGA_GPU_TERN_OK) {
        float rev[4] = { 0, 0, 0, 0 };
        for (uint32_t i = 0; i < t->seg.count; i++) rev[3u - i] = t->seg.scale[i];
        memcpy(t->ws.cpu, rev, sizeof rev);
        if (omega_gpu_session_sync_to_gpu(&t->w) != 0 || omega_gpu_session_sync_to_gpu(&t->ws) != 0) rc = OMEGA_GPU_TERN_CHIP_FAIL;
    }
    if (rc != OMEGA_GPU_TERN_OK) { omega_gpu_tern_tensor_free(t); return rc; }
    *out = t;
    return OMEGA_GPU_TERN_OK;
}

void omega_gpu_tern_tensor_free(OmegaGpuTernTensor *t) {
    if (!t) return;
    omega_gpu_session_lock();
    if (t->w.cpu) omega_gpu_session_free(&t->w);
    if (t->ws.cpu) omega_gpu_session_free(&t->ws);
    omega_gpu_session_unlock();
    omega_gpu_host_free(t);
}

/* rec non-NULL (record path, caller holds the session lock): device buffers dev_x / dev_y / dev_as, the launch is appended
 * to rec, nothing staged, poisoned or read back. */
static int gemv_core(const OmegaGpuTernTensor *t, const int8_t *x, int f32, float a_scale, void *y, OmegaGpuTernInfo *info,
                     OmegaGpuCmdList *rec, uint64_t dev_x, uint64_t dev_y, uint64_t dev_as) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    if (rec && (!dev_x || !dev_y || (dev_x & 15u))) return OMEGA_GPU_TERN_BAD_ARGS;
    if (!t || !t->w.cpu || (!rec && (!x || !y))) return OMEGA_GPU_TERN_BAD_ARGS;
    if (f32 && !rec && !isfinite(a_scale)) return OMEGA_GPU_TERN_BAD_ARGS;
    const uint32_t n = t->n, k = t->k;

    if (!rec) omega_gpu_session_lock();
#define UNLOCK() do { if (!rec) omega_gpu_session_unlock(); } while (0)
    OmegaTernGemvSpec spec; uint32_t ctas;
    memset(&spec, 0, sizeof spec);
    spec.k = k; spec.layout = t->layout; spec.epi = f32 ? OMEGA_TERN_EPI_F32 : OMEGA_TERN_EPI_I32;
    omega_gpu_tern_shape_for(k, t->layout, t->R, &spec.rows, &spec.warps, &spec.depth, &ctas);
    spec.mutant = g.mutant == 2 ? 0 : g.mutant;
    uint32_t bounds[3];
    if (!omega_tern_gemv_spec_ok(&spec) || omega_gpu_tern_segment_bounds(n, &t->seg, g.mutant, bounds) != 0) {
        omega_gpu_session_set_error("tern: kernel shape refused for this tensor");
        UNLOCK(); return OMEGA_GPU_TERN_CODEGEN_FAIL;
    }
    const uint32_t U = t->n_padded / (2u * spec.rows);
    if (t->n_padded % (2u * spec.rows)) { UNLOCK(); return OMEGA_GPU_TERN_CODEGEN_FAIL; }
    uint32_t grid_x = (U + spec.warps - 1u) / spec.warps;
    if (grid_x > ctas) grid_x = ctas;
    if (grid_x == 0) grid_x = 1;

    int rc = OMEGA_GPU_TERN_OK;
    if (!dev_open_locked()) { UNLOCK(); return OMEGA_GPU_TERN_CHIP_FAIL; }
    if (!rec) {
        int src = omega_gpu_session_scratch(&g.x_buf, (size_t)k);
        if (src == 0) src = omega_gpu_session_scratch(&g.y_buf, (size_t)n * 4u);
        if (src == 0) src = omega_gpu_session_scratch(&g.as_buf, 16u);
        if (src != 0) { UNLOCK(); return src == OMEGA_GPU_SESSION_SCRATCH_RESERVED ? OMEGA_GPU_TERN_TOO_LARGE : OMEGA_GPU_TERN_CHIP_FAIL; }
        if (g.x_buf.mem.va & 15u) { omega_gpu_session_set_error("tern: x scratch not 16-byte aligned"); UNLOCK(); return OMEGA_GPU_TERN_CHIP_FAIL; }
    }
    Slot *ks = NULL; bool hit = false;
    g_rec = rec != NULL;
    rc = kernel_for(&spec, &ks, &hit);
    g_rec = 0;
    if (rc != OMEGA_GPU_TERN_OK) { UNLOCK(); return rc; }

    uint32_t *yd = g.y_buf.mem.cpu;
    if (!rec) {
        memcpy(g.x_buf.mem.cpu, x, k);
        memcpy(g.as_buf.mem.cpu, &a_scale, 4);
        const uint32_t poison = f32 ? OMEGA_GPU_TERN_POISON_F32_BITS : (uint32_t)OMEGA_GPU_TERN_POISON_I32;
        for (size_t i = 0; i < n; i++) yd[i] = poison;
        if (omega_gpu_session_sync_to_gpu(&g.x_buf.mem) != 0 || omega_gpu_session_sync_to_gpu(&g.y_buf.mem) != 0
            || omega_gpu_session_sync_to_gpu(&g.as_buf.mem) != 0) rc = OMEGA_GPU_TERN_CHIP_FAIL;
    }

    uint32_t args[OMEGA_TERN_ARGS_WORDS];
    memset(args, 0, sizeof args);
    const uint64_t x_va = rec ? dev_x : g.x_buf.mem.va, y_va = rec ? dev_y : g.y_buf.mem.va;
    const uint64_t as_va = rec ? dev_as : g.as_buf.mem.va, ws_va = t->ws.va;
    args[0] = (uint32_t)t->w.va; args[1] = (uint32_t)(t->w.va >> 32);
    args[2] = (uint32_t)x_va; args[3] = (uint32_t)(x_va >> 32);
    args[4] = (uint32_t)y_va; args[5] = (uint32_t)(y_va >> 32);
    args[6] = (uint32_t)ws_va; args[7] = (uint32_t)(ws_va >> 32);
    args[8] = (uint32_t)as_va; args[9] = (uint32_t)(as_va >> 32);
    args[10] = U; args[11] = grid_x * spec.warps;
    args[12] = 0u - n;
    for (int i = 0; i < 3; i++) args[13 + i] = 0u - bounds[i];
    if (info) {
        info->n = n; info->k = k; info->n_padded = t->n_padded; info->layout = t->layout; info->rows_per_half = spec.rows;
        info->warps_per_cta = spec.warps; info->depth = spec.depth; info->grid_x = grid_x;
        info->chunks = omega_tern_gemv_chunks(&spec); info->epi = spec.epi;
        info->weight_bytes = omega_gpu_tern_tensor_bytes(t);
        info->kernel_cache_hit = hit; info->weights_cached = t->cached;
        memcpy(info->seg_bounds, bounds, sizeof bounds);
        (void)omega_gpu_fmt(info->target_chip, sizeof info->target_chip, "%s", "NVIDIA DGX Spark (Grace Blackwell GB10)");
        info->sm_architecture = 121;
    }
    if (rc == OMEGA_GPU_TERN_OK) {
        OmegaGpuLaunch L = {
            .code_va = ks->code.va, .gpr_count = gpr_declared(ks->kernel.gpr_count),
            .threads_x = 32u * spec.warps, .threads_y = 1, .grid_x = grid_x, .grid_y = 1,
            .num_elements = n,
            .args = args, .n_args = OMEGA_TERN_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
            .shared_bytes = omega_tern_gemv_shared_bytes(&spec),
        };
        if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); rc = OMEGA_GPU_TERN_CHIP_FAIL; }
        else if (rec) { if (omega_gpu_cmdlist_add(rec, &L) < 0) rc = OMEGA_GPU_TERN_CHIP_FAIL; }
        else if (omega_gpu_session_launch(&L, info ? &info->elapsed_ns : NULL, info ? &info->completion_marker : NULL) != 0) rc = OMEGA_GPU_TERN_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_TERN_OK && !rec && omega_gpu_session_sync_to_cpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_TERN_CHIP_FAIL;
    if (rc == OMEGA_GPU_TERN_OK && !rec) {
        const uint32_t poison = f32 ? OMEGA_GPU_TERN_POISON_F32_BITS : (uint32_t)OMEGA_GPU_TERN_POISON_I32;
        size_t unchanged = 0;
        for (size_t i = 0; i < n; i++) if (yd[i] == poison) unchanged++;
        if (unchanged) {
            char e[120]; omega_gpu_fmt(e, sizeof e, "tern: output poison survived in %zu of %u rows", unchanged, n);
            omega_gpu_session_set_error(e); rc = OMEGA_GPU_TERN_CHIP_FAIL;
        } else memcpy(y, yd, (size_t)n * 4u);
    }
    UNLOCK();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}
#undef UNLOCK

int omega_gpu_tern_gemv_i32(const OmegaGpuTernTensor *t, const int8_t *x, int32_t *y, OmegaGpuTernInfo *info) {
    return gemv_core(t, x, 0, 1.0f, y, info, NULL, 0, 0, 0);
}
int omega_gpu_tern_gemv_f32(const OmegaGpuTernTensor *t, const int8_t *x, float a_scale, float *y, OmegaGpuTernInfo *info) {
    return gemv_core(t, x, 1, a_scale, y, info, NULL, 0, 0, 0);
}
int omega_gpu_tern_gemv_record(OmegaGpuCmdList *cl, const OmegaGpuTernTensor *t, uint64_t x_va, uint64_t y_va, uint64_t as_va) {
    if (!cl) return OMEGA_GPU_TERN_BAD_ARGS;
    return gemv_core(t, NULL, as_va != 0, 1.0f, NULL, NULL, cl, x_va, y_va, as_va);
}
