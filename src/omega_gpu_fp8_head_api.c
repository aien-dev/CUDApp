/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* omega_gpu_fp8_head_api.c: FP8 lm_head, all x rows in one launch (aien-dev/CUDApp#217). See the header.
 * Kernel cache, pinning, register-file check and submission follow omega_gpu_bf16_gemv_api.c. */
#include "omega_gpu_fp8_head_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_fmt.h"
#include "omega_gpu_cmdlist.h"
#include <stdlib.h>
#include <string.h>

#define CACHE_SLOTS 8
#define MARKER_TIMEOUT_MS 20000ull
#define POISON_F32 (-999.0f)
#define DEFAULT_ROWS 2u
#define DEFAULT_WARPS 8u
#define REG_FILE 65536u
#define SM_THREADS 1536u /* CC 12.x resident threads per SM (OMEGA_GPU_SM_COUNT comment, omega_gpu_fp8_api.h) */

typedef struct {
    bool valid, pinned;
    OmegaFp8HeadSpec spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
} Slot;

static struct {
    OmegaGpuScratch x_buf, y_buf;
    uint32_t rows, warps, ctas; /* overrides, 0 = automatic */
    int env_read;
    int mutant;
    bool hooked;
    Slot cache[CACHE_SLOTS];
    unsigned next;
} g;
static int g_rec; /* 1 while a record call runs (lock held): kernel_for must not evict */

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
void omega_gpu_fp8_head_cache_clear(void) { omega_gpu_session_lock(); cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_fp8_head_set_shape(uint32_t rows, uint32_t warps, uint32_t ctas) {
    omega_gpu_session_lock(); g.env_read = 1; g.rows = rows; g.warps = warps; g.ctas = ctas; cache_clear_locked(); omega_gpu_session_unlock();
}
void omega_gpu_fp8_head_test_set_mutant(int m) { omega_gpu_session_lock(); g.mutant = m; cache_clear_locked(); omega_gpu_session_unlock(); }
int omega_gpu_fp8_head_test_mutant(void) { return g.mutant; }

/* AIEN_OMEGA_FP8_HEAD_SHAPE=rows:warps:ctas, read once (chip A/B knob; malformed = ignored) */
static void env_shape(void) {
    if (g.env_read) return;
    g.env_read = 1;
    const char *v = getenv("AIEN_OMEGA_FP8_HEAD_SHAPE");
    if (!v) return;
    uint32_t f[3] = { 0, 0, 0 }; unsigned i = 0;
    for (const char *p = v; ; p++) {
        if (*p >= '0' && *p <= '9') { f[i] = f[i] * 10u + (uint32_t)(*p - '0'); if (f[i] > 100000u) return; }
        else if (*p == ':' && i < 2) i++;
        else if (*p == 0) break;
        else return;
    }
    if (i != 2) return;
    g.rows = f[0]; g.warps = f[1]; g.ctas = f[2];
}

static uint32_t gpr_declared(uint32_t peak) { return omega_bw_gpr_declared(peak); }
static bool register_file_fits(uint32_t warps, uint32_t peak) {
    uint32_t decl = gpr_declared(peak);
    if ((uint64_t)32u * warps * decl <= REG_FILE) return true;
    char e[160]; omega_gpu_fmt(e, sizeof e, "fp8 head kernel: %u warps x %u declared registers exceed the 65536-register file", warps, decl);
    omega_gpu_session_set_error(e);
    return false;
}

void omega_gpu_fp8_head_spec_for(uint32_t k, uint32_t n, uint32_t xb, OmegaFp8HeadSpec *spec, uint32_t *grid_x) {
    env_shape();
    memset(spec, 0, sizeof *spec);
    spec->k = k; spec->xb = xb; spec->mutant = g.mutant;
    spec->rows = g.rows ? g.rows : (xb <= 1u ? 1u : DEFAULT_ROWS);
    spec->warps = g.warps ? g.warps : DEFAULT_WARPS;
    if (!grid_x) return;
    uint32_t ctas = g.ctas;
    if (!ctas) { /* every SM full: CTAs per SM limited by the register file and the resident-thread count */
        uint32_t insns = 0, peak = 0, per_sm = 1;
        if (omega_fp8head_prog_stats(spec, &insns, &peak) == 0) {
            const uint32_t by_regs = REG_FILE / (32u * spec->warps * gpr_declared(peak)), by_threads = SM_THREADS / (32u * spec->warps);
            per_sm = by_regs < by_threads ? by_regs : by_threads;
            if (per_sm == 0) per_sm = 1;
        }
        ctas = OMEGA_GPU_SM_COUNT * per_sm;
    }
    const uint32_t RG = spec->rows ? n / spec->rows : 0;
    uint32_t gx = (RG + spec->warps - 1u) / spec->warps;
    if (gx > ctas) gx = ctas;
    *grid_x = gx ? gx : 1u;
}

static int kernel_for(const OmegaFp8HeadSpec *spec, Slot **out, bool *hit) {
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        Slot *s = &g.cache[i];
        if (s->valid && memcmp(&s->spec, spec, sizeof *spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_FP8_OK; }
    }
    Slot *s = NULL;
    for (unsigned i = 0; i < CACHE_SLOTS && !s; i++) {
        Slot *c = &g.cache[(g.next + i) % CACHE_SLOTS];
        if (!c->valid || !c->pinned) { s = c; g.next = (unsigned)((c - g.cache) + 1) % CACHE_SLOTS; }
    }
    if (!s) { omega_gpu_session_set_error("fp8 head kernel cache: every slot holds a prepared kernel"); return OMEGA_GPU_FP8_TOO_LARGE; }
    if (g_rec && s->valid) { omega_gpu_session_set_error("fp8 head record: kernel cache miss would evict a resident kernel (omega_gpu_fp8_head_prepare before recording)"); return OMEGA_GPU_FP8_TOO_LARGE; }
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("fp8 head codegen: host memory"); return OMEGA_GPU_FP8_CODEGEN_FAIL; }
    int cg = omega_blackwell_codegen_fp8head_prog(spec, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("fp8 head codegen refused the shape"); return OMEGA_GPU_FP8_CODEGEN_FAIL; }
    if (!register_file_fits(spec->warps, prog->regalloc.peak_gpr_usage)) { omega_gpu_host_free(prog); return OMEGA_GPU_FP8_TOO_LARGE; }
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

int omega_gpu_fp8_head_prepare(uint32_t k, uint32_t xb) {
    OmegaFp8HeadSpec spec;
    omega_gpu_session_lock();
    omega_gpu_fp8_head_spec_for(k, 0, xb, &spec, NULL);
    int rc = OMEGA_GPU_FP8_OK;
    if (!omega_fp8head_spec_ok(&spec)) { omega_gpu_session_set_error("fp8 head: shape refused (k multiple of 512, xb 1..8)"); rc = OMEGA_GPU_FP8_CODEGEN_FAIL; }
    else if (!dev_open_locked()) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    else { Slot *s = NULL; bool hit = false; rc = kernel_for(&spec, &s, &hit); if (rc == OMEGA_GPU_FP8_OK) s->pinned = true; }
    omega_gpu_session_unlock();
    return rc;
}

/* rec non-NULL: record path (caller holds the lock), x and y are device addresses; else the scratch test path. */
static int head_core(const OmegaGpuFp8Tensor *t, uint32_t xb, const float *x, float *y, OmegaGpuFp8HeadInfo *info,
                     OmegaGpuCmdList *rec, uint64_t dev_x, uint64_t dev_y, uint32_t y_stride) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    uint32_t experts = 0, n = 0, k = 0;
    omega_gpu_fp8_tensor_shape(t, &experts, &n, &k);
    if (!t || experts != 1u || n == 0 || xb == 0 || xb > OMEGA_FP8HEAD_MAX_XB) return OMEGA_GPU_FP8_BAD_ARGS;
    if (rec ? (!dev_x || !dev_y || (xb > 1u && y_stride < n) || (uint64_t)y_stride * 4u > 0xffffffffull) : (!x || !y)) return OMEGA_GPU_FP8_BAD_ARGS;
    if (!rec) y_stride = n;
    if (!rec) omega_gpu_session_lock();
#define UNLOCK() do { if (!rec) omega_gpu_session_unlock(); } while (0)
    OmegaFp8HeadSpec spec; uint32_t grid_x = 1;
    omega_gpu_fp8_head_spec_for(k, n, xb, &spec, &grid_x);
    if (!omega_fp8head_spec_ok(&spec) || n % spec.rows) { UNLOCK(); omega_gpu_session_set_error("fp8 head: shape refused"); return OMEGA_GPU_FP8_CODEGEN_FAIL; }
    const uint32_t RG = n / spec.rows;
    uint64_t w_va = 0, s_va = 0;
    if (omega_gpu_fp8_tensor_entry(t, 0, &w_va, &s_va) != OMEGA_GPU_FP8_OK) { UNLOCK(); return OMEGA_GPU_FP8_BAD_ARGS; }
    int rc = OMEGA_GPU_FP8_OK;
    if (!dev_open_locked()) { UNLOCK(); return OMEGA_GPU_FP8_CHIP_FAIL; }
    if (!rec) {
        int src = omega_gpu_session_scratch(&g.x_buf, (size_t)xb * k * 4u);
        if (src == 0) src = omega_gpu_session_scratch(&g.y_buf, (size_t)xb * n * 4u);
        if (src != 0) { UNLOCK(); return src == OMEGA_GPU_SESSION_SCRATCH_RESERVED ? OMEGA_GPU_FP8_TOO_LARGE : OMEGA_GPU_FP8_CHIP_FAIL; }
    }
    Slot *ks = NULL; bool hit = false;
    g_rec = rec != NULL;
    rc = kernel_for(&spec, &ks, &hit);
    g_rec = 0;
    if (rc != OMEGA_GPU_FP8_OK) { UNLOCK(); return rc; }

    float *xd = rec ? NULL : g.x_buf.mem.cpu, *yd = rec ? NULL : g.y_buf.mem.cpu;
    if (!rec) {
        for (uint32_t b = 0; b < xb; b++) omega_fp8_pack_x_f32(x + (size_t)b * k, k, xd + (size_t)b * k);
        for (size_t i = 0; i < (size_t)xb * n; i++) yd[i] = POISON_F32;
        if (omega_gpu_session_sync_to_gpu(&g.x_buf.mem) != 0 || omega_gpu_session_sync_to_gpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    }
    const uint64_t x_va = rec ? dev_x : g.x_buf.mem.va, y_va = rec ? dev_y : g.y_buf.mem.va;
    uint32_t args[OMEGA_FP8HEAD_ARGS_WORDS];
    args[0] = (uint32_t)w_va; args[1] = (uint32_t)(w_va >> 32);
    args[2] = (uint32_t)s_va; args[3] = (uint32_t)(s_va >> 32);
    args[4] = (uint32_t)x_va; args[5] = (uint32_t)(x_va >> 32);
    args[6] = (uint32_t)y_va; args[7] = (uint32_t)(y_va >> 32);
    args[8] = RG; args[9] = y_stride * 4u; args[10] = grid_x * spec.warps;
    if (info) {
        info->n = n; info->k = k; info->xb = xb; info->rows = spec.rows; info->warps = spec.warps; info->grid_x = grid_x;
        info->gpr_declared = gpr_declared(ks->kernel.gpr_count);
        info->weight_bytes = (uint64_t)n * k;
        info->scale_bytes = (uint64_t)(n / OMEGA_FP8_BLOCK) * (k / OMEGA_FP8_BLOCK) * 4u;
        info->kernel_cache_hit = hit;
    }
    if (rc == OMEGA_GPU_FP8_OK) {
        OmegaGpuLaunch L = {
            .code_va = ks->code.va, .gpr_count = gpr_declared(ks->kernel.gpr_count),
            .threads_x = 32u * spec.warps, .threads_y = 1, .grid_x = grid_x, .grid_y = 1,
            .num_elements = (size_t)n * xb,
            .args = args, .n_args = OMEGA_FP8HEAD_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
            .shared_bytes = 0,
        };
        if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); rc = OMEGA_GPU_FP8_CHIP_FAIL; }
        else if (rec) { if (omega_gpu_cmdlist_add(rec, &L) < 0) rc = OMEGA_GPU_FP8_CHIP_FAIL; }
        else if (omega_gpu_session_launch(&L, info ? &info->elapsed_ns : NULL, info ? &info->completion_marker : NULL) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_FP8_OK && !rec && omega_gpu_session_sync_to_cpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    if (rc == OMEGA_GPU_FP8_OK && !rec) {
        size_t unchanged = 0;
        for (size_t i = 0; i < (size_t)xb * n; i++) if (yd[i] == POISON_F32) unchanged++;
        if (unchanged) { omega_gpu_session_set_error("fp8 head: output poison survived"); rc = OMEGA_GPU_FP8_CHIP_FAIL; }
        else memcpy(y, yd, (size_t)xb * n * sizeof *y);
    }
    UNLOCK();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}
#undef UNLOCK

int omega_gpu_fp8_head_gemv_f32(const OmegaGpuFp8Tensor *t, uint32_t xb, const float *x, float *y, OmegaGpuFp8HeadInfo *info) {
    return head_core(t, xb, x, y, info, NULL, 0, 0, 0);
}
int omega_gpu_fp8_head_record(OmegaGpuCmdList *cl, const OmegaGpuFp8Tensor *t, uint32_t xb, uint64_t x_va, uint64_t y_va, uint32_t y_stride) {
    if (!cl) return OMEGA_GPU_FP8_BAD_ARGS;
    return head_core(t, xb, NULL, NULL, NULL, cl, x_va, y_va, y_stride);
}
