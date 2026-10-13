/* omega_gpu_tcgemm_api.c: tensor-core FP8 grouped GEMM on the shared session (OWN-30B T3 cut 1). See the header.
 * Submission path and buffer policy follow omega_gpu_fp8gemm_api.c (GPU-uncached buffers, one omega_gpu_session_launch). */
#include "omega_gpu_tcgemm_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_fmt.h"
#include <string.h>
#include <stdlib.h>

#define CACHE_SLOTS 8
#define MARKER_TIMEOUT_MS 20000ull
#define POISON_F32 (-999.0f)

typedef struct { bool valid, pinned; OmegaTcGemmSpec spec; OmegaBlackwellKernel kernel; OmegaGpuMem code; } Slot;
typedef struct { uint64_t w_va, s_va; uint32_t row0, cnt, pad[2]; } TileEntry;
_Static_assert(sizeof(TileEntry) == OMEGA_TCGEMM_TILE_BYTES, "tile entry layout");

static struct {
    OmegaGpuScratch x_buf, y_buf, tile_buf, ridx_buf;
    uint32_t rt, ct, warps;
    int mutant;
    bool hooked;
    Slot cache[CACHE_SLOTS];
    unsigned next;
} g;

const char *omega_gpu_tcgemm_last_error(void) { return omega_gpu_session_last_error(); }
const char *omega_gpu_tcgemm_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_TCGEMM_OK: return "OK";
    case OMEGA_GPU_TCGEMM_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_TCGEMM_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_TCGEMM_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_TCGEMM_CHIP_FAIL: return "CHIP_FAIL";
    default: return "UNKNOWN";
    }
}
static void cache_clear_locked(void) {
    for (unsigned i = 0; i < CACHE_SLOTS; i++) { if (!g.cache[i].valid) continue; omega_gpu_session_free(&g.cache[i].code); memset(&g.cache[i], 0, sizeof g.cache[i]); }
    g.next = 0;
}
static void on_session_close(void) {
    cache_clear_locked();
    omega_gpu_session_scratch_free(&g.x_buf); omega_gpu_session_scratch_free(&g.y_buf);
    omega_gpu_session_scratch_free(&g.tile_buf); omega_gpu_session_scratch_free(&g.ridx_buf);
}
static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}
void omega_gpu_tcgemm_cache_clear(void) { omega_gpu_session_lock(); cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_tcgemm_set_shape(uint32_t rt, uint32_t ct, uint32_t w) { omega_gpu_session_lock(); g.rt = rt; g.ct = ct; g.warps = w; cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_tcgemm_test_set_mutant(int m) { omega_gpu_session_lock(); g.mutant = m; cache_clear_locked(); omega_gpu_session_unlock(); }

void omega_gpu_tcgemm_spec_for(uint32_t k, uint32_t n, OmegaTcGemmSpec *spec) {
    memset(spec, 0, sizeof *spec);
    spec->k = k;
    /* 64 rows x 128 columns per CTA, 8 warps (peak 124 registers, declared 128); 64 columns when n is not a multiple of 128 */
    spec->rt = g.rt ? g.rt : 4u; spec->ct = g.ct ? g.ct : (n % 128u == 0 ? 2u : 1u); spec->warps = g.warps ? g.warps : 8u;
    spec->mutant = g.mutant;
}

static int kernel_for(const OmegaTcGemmSpec *spec, Slot **out, bool *hit) {
    for (unsigned i = 0; i < CACHE_SLOTS; i++) { Slot *s = &g.cache[i]; if (s->valid && memcmp(&s->spec, spec, sizeof *spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_TCGEMM_OK; } }
    Slot *s = NULL;
    for (unsigned i = 0; i < CACHE_SLOTS && !s; i++) { Slot *c = &g.cache[(g.next + i) % CACHE_SLOTS]; if (!c->valid || !c->pinned) { s = c; g.next = (unsigned)((c - g.cache) + 1) % CACHE_SLOTS; } }
    if (!s) { omega_gpu_session_set_error("tc gemm kernel cache: every slot holds a prepared kernel"); return OMEGA_GPU_TCGEMM_TOO_LARGE; }
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("tc gemm codegen: host memory"); return OMEGA_GPU_TCGEMM_CODEGEN_FAIL; }
    int cg = omega_blackwell_codegen_tcgemm_prog(spec, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("tc gemm codegen refused the shape"); return OMEGA_GPU_TCGEMM_CODEGEN_FAIL; }
    const uint32_t decl = omega_bw_gpr_declared(prog->regalloc.peak_gpr_usage);
    if (decl > 128u || (uint64_t)32u * spec->warps * decl > 65536u) {
        char e[160]; omega_gpu_fmt(e, sizeof e, "tc gemm kernel: %u warps x %u declared registers do not fit", spec->warps, decl);
        omega_gpu_session_set_error(e); omega_gpu_host_free(prog); return OMEGA_GPU_TCGEMM_TOO_LARGE;
    }
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_TCGEMM_CHIP_FAIL; }
    OmegaBlackwellKernel kernel;
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need) { omega_gpu_session_free(&code); return OMEGA_GPU_TCGEMM_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_TCGEMM_CHIP_FAIL; }
    if (s->valid) omega_gpu_session_free(&s->code);
    memset(s, 0, sizeof *s);
    s->valid = true; s->spec = *spec; s->kernel = kernel; s->code = code;
    *out = s; *hit = false;
    return OMEGA_GPU_TCGEMM_OK;
}

int omega_gpu_tcgemm_prepare(uint32_t k) {
    OmegaTcGemmSpec sp; omega_gpu_tcgemm_spec_for(k, 128u, &sp);
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_TCGEMM_OK; Slot *s = NULL; bool hit = false;
    if (!dev_open_locked()) rc = OMEGA_GPU_TCGEMM_CHIP_FAIL;
    else if ((rc = kernel_for(&sp, &s, &hit)) == OMEGA_GPU_TCGEMM_OK) s->pinned = true;
    omega_gpu_session_unlock();
    return rc;
}

int omega_gpu_tcgemm_f32(const OmegaGpuFp8Tensor *t, uint32_t M, const uint32_t *ids, const uint32_t *trow, const float *x, uint32_t x_rows, float *y, OmegaGpuTcGemmInfo *info) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    uint32_t experts = 0, n = 0, k = 0;
    if (!t) return OMEGA_GPU_TCGEMM_BAD_ARGS;
    omega_gpu_fp8_tensor_shape(t, &experts, &n, &k);
    if (!experts || !ids || !trow || !x || !y || M == 0 || x_rows == 0) return OMEGA_GPU_TCGEMM_BAD_ARGS;
    OmegaTcGemmSpec spec; omega_gpu_tcgemm_spec_for(k, n, &spec);
    if (!omega_tcgemm_spec_ok(&spec)) return OMEGA_GPU_TCGEMM_BAD_ARGS;
    const uint32_t BM = 16u * spec.rt, span = omega_tcgemm_span(&spec);
    if (n % span || n % 128u) return OMEGA_GPU_TCGEMM_BAD_ARGS;
    uint32_t segs = 0, ntiles = 0;
    for (uint32_t i = 0; i < M;) {
        if (ids[i] >= experts) return OMEGA_GPU_TCGEMM_BAD_ARGS;
        uint32_t j = i; while (j < M && ids[j] == ids[i]) { if (trow[j] >= x_rows) return OMEGA_GPU_TCGEMM_BAD_ARGS; j++; }
        if (j < M && ids[j] < ids[i]) return OMEGA_GPU_TCGEMM_BAD_ARGS;
        segs++; ntiles += (j - i + BM - 1) / BM; i = j;
    }
    if (ntiles > 65535u) return OMEGA_GPU_TCGEMM_TOO_LARGE;
    const uint32_t grid_x = n / span;

    omega_gpu_session_lock();
    int rc = OMEGA_GPU_TCGEMM_OK;
    if (!dev_open_locked()) { omega_gpu_session_unlock(); return OMEGA_GPU_TCGEMM_CHIP_FAIL; }
    int src = omega_gpu_session_scratch(&g.x_buf, (size_t)x_rows * k * 2u);
    if (src == 0) src = omega_gpu_session_scratch(&g.y_buf, (size_t)M * n * 4u);
    if (src == 0) src = omega_gpu_session_scratch(&g.tile_buf, (size_t)ntiles * OMEGA_TCGEMM_TILE_BYTES);
    if (src == 0) src = omega_gpu_session_scratch(&g.ridx_buf, ((size_t)M + BM) * 4u);
    if (src != 0) { omega_gpu_session_unlock(); return src == OMEGA_GPU_SESSION_SCRATCH_RESERVED ? OMEGA_GPU_TCGEMM_TOO_LARGE : OMEGA_GPU_TCGEMM_CHIP_FAIL; }
    Slot *ks = NULL; bool hit = false;
    rc = kernel_for(&spec, &ks, &hit);
    if (rc != OMEGA_GPU_TCGEMM_OK) { omega_gpu_session_unlock(); return rc; }

    TileEntry *tab = g.tile_buf.mem.cpu; uint32_t *ridx = g.ridx_buf.mem.cpu; uint32_t nt = 0;
    uint64_t distinct = 0;
    for (uint32_t i = 0; i < M;) {
        uint32_t j = i; while (j < M && ids[j] == ids[i]) j++;
        uint64_t w_va = 0, s_va = 0; (void)omega_gpu_fp8_tensor_entry(t, ids[i], &w_va, &s_va);
        for (uint32_t r = i; r < j; r += BM) { memset(&tab[nt], 0, sizeof tab[nt]); tab[nt].w_va = w_va; tab[nt].s_va = s_va; tab[nt].row0 = r; tab[nt].cnt = j - r < BM ? j - r : BM; nt++; }
        distinct += (uint64_t)n * k; i = j;
    }
    { uint32_t *tmp = malloc(((size_t)M + BM) * 4u); if (!tmp) { omega_gpu_session_unlock(); return OMEGA_GPU_TCGEMM_CHIP_FAIL; }
      for (uint32_t i = 0; i < M + BM; i++) tmp[i] = i < M ? trow[i] : 0u;
      memcpy(ridx, tmp, ((size_t)M + BM) * 4u); free(tmp); }
    {   /* permute in cached memory, one sequential copy per row into the uncached buffer */
        uint16_t *row = malloc((size_t)k * 2u); uint16_t *xd = g.x_buf.mem.cpu;
        if (!row) { omega_gpu_session_unlock(); return OMEGA_GPU_TCGEMM_CHIP_FAIL; }
        for (uint32_t r = 0; r < x_rows; r++) { omega_tcgemm_permute_x(x + (size_t)r * k, k, row); memcpy(xd + (size_t)r * k, row, (size_t)k * 2u); }
        free(row);
    }
    float *yd = g.y_buf.mem.cpu;
    for (size_t i = 0; i < (size_t)M * n; i++) yd[i] = POISON_F32;
    if (omega_gpu_session_sync_to_gpu(&g.tile_buf.mem) != 0 || omega_gpu_session_sync_to_gpu(&g.ridx_buf.mem) != 0 ||
        omega_gpu_session_sync_to_gpu(&g.x_buf.mem) != 0 || omega_gpu_session_sync_to_gpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_TCGEMM_CHIP_FAIL;

    uint32_t args[OMEGA_TCGEMM_ARGS_WORDS];
    const uint64_t va[4] = { g.tile_buf.mem.va, g.ridx_buf.mem.va, g.x_buf.mem.va, g.y_buf.mem.va };
    for (int i = 0; i < 4; i++) { args[2 * i] = (uint32_t)va[i]; args[2 * i + 1] = (uint32_t)(va[i] >> 32); }
    args[8] = n;
    const uint32_t decl = omega_bw_gpr_declared(ks->kernel.gpr_count);
    if (info) {
        info->rows = M; info->segments = segs; info->tiles = nt; info->n = n; info->k = k; info->rt = spec.rt; info->ct = spec.ct; info->warps = spec.warps;
        info->grid_x = grid_x; info->grid_y = nt; info->gpr_declared = decl;
        info->weight_bytes = distinct; info->streamed_bytes = (uint64_t)nt * n * k;
        info->act_bytes = (uint64_t)nt * grid_x * BM * k * 2u; info->kernel_cache_hit = hit;
    }
    if (rc == OMEGA_GPU_TCGEMM_OK) {
        OmegaGpuLaunch L = {
            .code_va = ks->code.va, .gpr_count = decl,
            .threads_x = 32u * spec.warps, .threads_y = 1, .grid_x = grid_x, .grid_y = nt,
            .num_elements = (size_t)M * n,
            .args = args, .n_args = OMEGA_TCGEMM_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
            .shared_bytes = omega_tcgemm_shared_bytes(&spec),
        };
        if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); rc = OMEGA_GPU_TCGEMM_CHIP_FAIL; }
        else if (omega_gpu_session_launch(&L, info ? &info->elapsed_ns : NULL, info ? &info->completion_marker : NULL) != 0) rc = OMEGA_GPU_TCGEMM_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_TCGEMM_OK && omega_gpu_session_sync_to_cpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_TCGEMM_CHIP_FAIL;
    if (rc == OMEGA_GPU_TCGEMM_OK) {
        size_t unchanged = 0;
        for (size_t i = 0; i < (size_t)M * n; i++) if (yd[i] == POISON_F32) unchanged++;
        if (unchanged) { omega_gpu_session_set_error("output poison survived"); rc = OMEGA_GPU_TCGEMM_CHIP_FAIL; }
        else memcpy(y, yd, (size_t)M * n * sizeof *y);
    }
    omega_gpu_session_unlock();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}
