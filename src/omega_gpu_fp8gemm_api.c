/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* omega_gpu_fp8gemm_api.c: FP8 grouped GEMM for prefill on the shared session (OWN-30B builder F item 26).
 * See the header. Submission path and buffer policy are the FP8 GEMV API's: every buffer both sides touch is
 * GPU-uncached (nvos.h NVOS32_ATTR2_GPU_CACHEABLE), one launch = omega_gpu_session_launch. */
#include "omega_gpu_fp8gemm_api.h"
#include "omega_gpu_fp8gemm_internal.h"
#include "omega_gpu_session.h"
#include "omega_gpu_cmdlist.h"
#include "omega_gpu_fmt.h"
#include "omega_gpu_native_runtime.h"
#include "omega_gpu_code_alloc.h"
#include <string.h>
#include <stdlib.h>

#define CACHE_SLOTS 8
#define MARKER_TIMEOUT_MS 20000ull
#define POISON_F32 (-999.0f)

typedef struct {
    bool valid, pinned;
    bool exact;
    uint8_t manifest[32];
    OmegaFp8GemmSpec spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
} Slot;

static struct {
    OmegaGpuScratch x_buf, y_buf, seg_buf;
    uint32_t mt, warps, ctas;
    int mutant;
    bool hooked;
    Slot cache[CACHE_SLOTS];
    unsigned next;
    OmegaGpuNativeCache native;
} g;

const char *omega_gpu_fp8gemm_last_error(void) { return omega_gpu_session_last_error(); }
const char *omega_gpu_fp8gemm_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_FP8GEMM_OK: return "OK";
    case OMEGA_GPU_FP8GEMM_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_FP8GEMM_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_FP8GEMM_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_FP8GEMM_CHIP_FAIL: return "CHIP_FAIL";
    default: return "UNKNOWN";
    }
}
static void cache_clear_locked(void) {
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        if (!g.cache[i].valid) continue;
        if (g.cache[i].pinned) {
            OmegaGpuNativeLease lease = omega_native_cache_lease(&g.native, i);
            if (omega_native_lease_op(&lease, 5, 0) == 0) g.cache[i].pinned = false;
        }
        if (omega_native_cache_victim(&g.native, i, 0) != 1) continue;
        omega_gpu_session_free(&g.cache[i].code);
        (void)omega_native_cache_victim(&g.native, i, 1);
        memset(&g.cache[i], 0, sizeof g.cache[i]);
    }
    g.next = 0;
}
static void on_session_close(void) {
    cache_clear_locked();
    omega_gpu_session_scratch_free(&g.x_buf);
    omega_gpu_session_scratch_free(&g.y_buf);
    omega_gpu_session_scratch_free(&g.seg_buf);
}
static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}
void omega_gpu_fp8gemm_cache_clear(void) { omega_gpu_session_lock(); cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_fp8gemm_set_shape(uint32_t mt, uint32_t warps) { omega_gpu_session_lock(); g.mt = mt; g.warps = warps; cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_fp8gemm_set_cta_budget(uint32_t c) { omega_gpu_session_lock(); g.ctas = c; omega_gpu_session_unlock(); }
void omega_gpu_fp8gemm_test_set_mutant(int m) { omega_gpu_session_lock(); g.mutant = m; cache_clear_locked(); omega_gpu_session_unlock(); }

static uint32_t gpr_declared(uint32_t peak) { return omega_bw_gpr_declared(peak); } /* peak + 2 rounded up to 8: the chip hung at peak + 1 (omega_blackwell_codegen.h) */
static bool register_file_fits(uint32_t warps, uint32_t peak) {
    uint32_t decl = gpr_declared(peak);
    if ((uint64_t)32u * warps * decl <= 65536u) return true;
    char e[160]; omega_gpu_fmt(e, sizeof e, "fp8 gemm kernel: %u warps x %u declared registers exceed the 65536-register file", warps, decl);
    omega_gpu_session_set_error(e);
    return false;
}
int omega_gpu_fp8gemm_kernel_fits(const OmegaFp8GemmSpec *spec, uint32_t *declared) {
    uint32_t peak = 0;
    if (!spec || omega_fp8gemm_prog_stats(spec, NULL, &peak) != 0) { omega_gpu_session_set_error("fp8 gemm codegen refused the shape"); return OMEGA_GPU_FP8GEMM_CODEGEN_FAIL; }
    if (declared) *declared = gpr_declared(peak);
    return register_file_fits(spec->warps, peak) ? OMEGA_GPU_FP8GEMM_OK : OMEGA_GPU_FP8GEMM_TOO_LARGE;
}

void omega_gpu_fp8gemm_shape_for(uint32_t k, uint32_t *mt, uint32_t *warps, uint32_t *lane_bytes) {
    uint32_t lb = k % 512u == 0 ? 16u : 8u;
    if (lane_bytes) *lane_bytes = lb;
    /* default 4 rows per pass: chip sweep 2026-10-13, gate|up 3.5 ms at 4 vs 6.0 ms at 8 (8 rows fill the 16 KB of shared memory and run at half the rate) */
    if (mt) *mt = g.mt ? g.mt : 4u;
    if (warps) *warps = g.warps ? g.warps : 16u;
}
/* L312 lever 3: x' staging words one K step ahead, OFF unless AIEN_OMEGA_FP8GEMM_X_AHEAD=1 (read once; P-DRA-4 DRA-67:
 * decode changes are env-gated, default off, until the stroke gate passes), and only where the NJ = mt * lane_bytes /
 * warps staging words fit the registers the program holds for them (NJ <= 16); call after mt is final */
static void x_ahead_for(OmegaFp8GemmSpec *spec) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("AIEN_OMEGA_FP8GEMM_X_AHEAD"); on = (e && e[0] == '1' && e[1] == 0) ? 1 : 0; }
    spec->x_ahead = on && spec->mt * spec->lane_bytes / spec->warps <= 16u ? 1u : 0u;
}
/* dense = a few distinct experts with many rows each (qkv, o): 8 rows per pass measured best there */
static void spec_for(uint32_t k, uint32_t segs, OmegaFp8GemmSpec *spec) {
    memset(spec, 0, sizeof *spec);
    spec->k = k;
    omega_gpu_fp8gemm_shape_for(k, &spec->mt, &spec->warps, &spec->lane_bytes);
    if (!g.mt && segs < 8u) spec->mt = 8u;
    spec->mutant = g.mutant;
    x_ahead_for(spec);
}

static int g_rec; /* 1 while a record call runs (lock held): kernel_for must not evict */
static int kernel_for(const OmegaFp8GemmSpec *spec, Slot **out, bool *hit) {
    int fresh = omega_native_cache_refresh(&g.native, CACHE_SLOTS);
    if (fresh < 0) return OMEGA_GPU_FP8GEMM_CODEGEN_FAIL;
    if (fresh) memset(g.cache, 0, sizeof g.cache); /* prior device handles are dead */
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        Slot *s = &g.cache[i];
        if (s->valid && !s->exact && memcmp(&s->spec, spec, sizeof *spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_FP8GEMM_OK; }
    }
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("fp8 gemm codegen: host memory"); return OMEGA_GPU_FP8GEMM_CODEGEN_FAIL; }
    int cg = omega_blackwell_codegen_fp8gemm_prog(spec, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0 || !register_file_fits(spec->warps, prog->regalloc.peak_gpr_usage)) {
        omega_gpu_host_free(prog); return OMEGA_GPU_FP8GEMM_CODEGEN_FAIL;
    }
    uint8_t key[32]; uint64_t request[16];
    int index = omega_native_shape_key(spec, sizeof *spec, key) == 0 ?
        omega_native_cache_reserve(&g.native, key, omega_gpu_code_alloc_bytes(need), request) : -1;
    if (index < 0) { omega_gpu_host_free(prog); return OMEGA_GPU_FP8GEMM_TOO_LARGE; }
    Slot *s = &g.cache[index];
    if (g_rec && s->valid) {
        (void)omega_native_cache_finish(&g.native, (unsigned)index, request, 0);
        omega_gpu_host_free(prog); return OMEGA_GPU_FP8GEMM_TOO_LARGE;
    }
    OmegaGpuMem code = {0}; OmegaBlackwellKernel kernel;
    if (omega_gpu_session_alloc_code(need, &code) != 0) cg = -1;
    else cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need ||
        omega_gpu_session_sync_to_gpu(&code) != 0 ||
        omega_native_cache_finish(&g.native, (unsigned)index, request, 1) != 0) {
        omega_gpu_session_free(&code);
        (void)omega_native_cache_finish(&g.native, (unsigned)index, request, 0);
        return OMEGA_GPU_FP8GEMM_CHIP_FAIL;
    }
    if (s->valid) omega_gpu_session_free(&s->code);
    memset(s, 0, sizeof *s);
    s->valid = true; s->spec = *spec; s->kernel = kernel; s->code = code;
    *out = s; *hit = false;
    return OMEGA_GPU_FP8GEMM_OK;
}

OmegaGpuNativeCache *omega_gpu_fp8gemm_native_cache(void) { return &g.native; }

int omega_gpu_fp8gemm_prepare_native(const OmegaGpuNativeArtifact *a, OmegaGpuNativeLease *lease) {
    uint8_t key[32]; uint64_t req[16], tok[16] = {0};
    if (!lease || omega_native_admitted(a, key) != 0) return -1;
    if (!dev_open_locked()) return -1;
    int fresh = omega_native_cache_refresh(&g.native, CACHE_SLOTS);
    if (fresh < 0) return -1;
    if (fresh) memset(g.cache, 0, sizeof g.cache);
    omega_native_context(req); req[2] = omega_gpu_code_alloc_bytes((size_t)a->sizes[1]);
    memcpy(req + 3, key, 32);
    int64_t found = omega_native_cache_call(&g.native, 12, req, tok);
    if (found < 0) return -1;
    int index = found ? (int)(found - 1) : omega_native_cache_reserve(&g.native, key, (size_t)req[2], req);
    if (index < 0) return -1;
    Slot *s = &g.cache[index];
    if (found) {
        if (!s->valid || !s->exact || memcmp(s->manifest, a->manifest, 32)) return -1;
    } else {
        OmegaGpuMem code = {0};
        if (omega_gpu_session_alloc_code((size_t)a->sizes[1], &code) != 0) goto abort;
        memcpy(code.cpu, a->code, (size_t)a->sizes[1]);
        if (omega_gpu_session_sync_to_gpu(&code) != 0 ||
            omega_native_cache_finish(&g.native, (unsigned)index, req, 1) != 0) {
            omega_gpu_session_free(&code); goto abort;
        }
        if (s->valid) omega_gpu_session_free(&s->code);
        memset(s, 0, sizeof *s);
        s->valid = s->exact = true; s->code = code;
        s->kernel.code = code.cpu; s->kernel.code_size = (size_t)a->sizes[1];
        memcpy(s->manifest, a->manifest, 32);
    }
    *lease = omega_native_cache_lease(&g.native, (unsigned)index);
    return omega_native_lease_op(lease, 5, 1);
abort:
    (void)omega_native_cache_finish(&g.native, (unsigned)index, req, 0);
    return -1;
}

int omega_gpu_fp8gemm_prepare(uint32_t k) {
    /* both variants: many experts (4 rows per pass) and dense (8 rows per pass) */
    OmegaFp8GemmSpec sp[2]; spec_for(k, 64u, &sp[0]); spec_for(k, 1u, &sp[1]);
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_FP8GEMM_OK;
    if (!dev_open_locked()) rc = OMEGA_GPU_FP8GEMM_CHIP_FAIL;
    else for (int v = 0; v < 2 && rc == OMEGA_GPU_FP8GEMM_OK; v++) {
        Slot *s = NULL; bool hit = false; rc = kernel_for(&sp[v], &s, &hit);
        if (rc == OMEGA_GPU_FP8GEMM_OK && !s->pinned) {
            OmegaGpuNativeLease lease = omega_native_cache_lease(&g.native, (unsigned)(s - g.cache));
            if (omega_native_lease_op(&lease, 5, 1) != 0) rc = OMEGA_GPU_FP8GEMM_TOO_LARGE;
            else s->pinned = true;
        }
    }
    omega_gpu_session_unlock();
    return rc;
}

/* ---- the stroke's record path (L312 v2): one launch appended to a command list, every buffer the caller's ---- */
static void spec_for_stroke(uint32_t k, uint32_t segs, uint32_t mt, int rowmap, OmegaFp8GemmSpec *spec) {
    spec_for(k, segs, spec);
    if (mt) spec->mt = mt;
    spec->rowmap = rowmap ? 1u : 0u;
    x_ahead_for(spec);
}
void omega_gpu_fp8gemm_stroke_spec(uint32_t k, uint32_t segs, uint32_t mt, int rowmap, OmegaFp8GemmSpec *spec) {
    spec_for_stroke(k, segs, mt, rowmap, spec);
}
int omega_gpu_fp8gemm_prepare_shape(uint32_t k, uint32_t segs, uint32_t mt, int rowmap) {
    OmegaFp8GemmSpec spec; spec_for_stroke(k, segs, mt, rowmap, &spec);
    if (omega_fp8gemm_shared_bytes(&spec) == 0) { omega_gpu_session_set_error("fp8 gemm prepare: shape refused (rowmap needs mt <= 4)"); return OMEGA_GPU_FP8GEMM_BAD_ARGS; }
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_FP8GEMM_OK;
    if (!dev_open_locked()) rc = OMEGA_GPU_FP8GEMM_CHIP_FAIL;
    else {
        Slot *s = NULL; bool hit = false; rc = kernel_for(&spec, &s, &hit);
        if (rc == OMEGA_GPU_FP8GEMM_OK && !s->pinned) {
            OmegaGpuNativeLease lease = omega_native_cache_lease(&g.native, (unsigned)(s - g.cache));
            if (omega_native_lease_op(&lease, 5, 1) != 0) rc = OMEGA_GPU_FP8GEMM_TOO_LARGE;
            else s->pinned = true;
        }
    }
    omega_gpu_session_unlock();
    return rc;
}
uint32_t omega_gpu_fp8gemm_cta_budget_for(uint32_t segs) { return g.ctas ? g.ctas : (segs >= 64u ? 2u : segs >= 8u ? 12u : 96u); }

int omega_gpu_fp8gemm_record(OmegaGpuCmdList *cl, const OmegaGpuFp8Tensor *t, uint32_t rows, uint32_t segs, uint64_t seg_va, uint64_t xp_va,
                             uint64_t y_va, uint64_t xmap_va, uint64_t ymap_va, uint32_t mt, uint32_t cta_budget, OmegaGpuFp8GemmInfo *info) {
    if (info) memset(info, 0, sizeof *info);
    uint32_t experts = 0, n = 0, k = 0;
    omega_gpu_fp8_tensor_shape(t, &experts, &n, &k);
    if (!cl || !t || !experts || !seg_va || !xp_va || !y_va || rows == 0 || segs == 0 || (!xmap_va) != (!ymap_va)) return OMEGA_GPU_FP8GEMM_BAD_ARGS;
    if (rows > OMEGA_GPU_FP8GEMM_MAX_ROWS || segs > OMEGA_GPU_FP8GEMM_MAX_SEGMENTS) return OMEGA_GPU_FP8GEMM_TOO_LARGE;
    OmegaFp8GemmSpec spec; spec_for_stroke(k, segs, mt, xmap_va != 0, &spec);
    if (omega_fp8gemm_shared_bytes(&spec) == 0) { omega_gpu_session_set_error("fp8 gemm record: shape refused (rowmap needs mt <= 4)"); return OMEGA_GPU_FP8GEMM_BAD_ARGS; }
    const uint32_t W = spec.warps;
    uint32_t grid_x = (n + W - 1) / W;
    const uint32_t budget = cta_budget ? cta_budget : omega_gpu_fp8gemm_cta_budget_for(segs);
    if (grid_x > budget) grid_x = budget;
    if (grid_x == 0) grid_x = 1;
    /* caller holds the session lock */
    if (!dev_open_locked()) return OMEGA_GPU_FP8GEMM_CHIP_FAIL;
    Slot *ks = NULL; bool hit = false;
    g_rec = 1;
    int rc = kernel_for(&spec, &ks, &hit);
    g_rec = 0;
    if (rc != OMEGA_GPU_FP8GEMM_OK) return rc;
    uint32_t args[OMEGA_GEMM_ARGS_WORDS_MAP]; memset(args, 0, sizeof args);
    args[0] = (uint32_t)seg_va; args[1] = (uint32_t)(seg_va >> 32);
    args[2] = (uint32_t)xp_va; args[3] = (uint32_t)(xp_va >> 32);
    args[4] = (uint32_t)y_va; args[5] = (uint32_t)(y_va >> 32);
    args[6] = n; args[7] = grid_x * W;
    args[8] = (uint32_t)xmap_va; args[9] = (uint32_t)(xmap_va >> 32);
    args[10] = (uint32_t)ymap_va; args[11] = (uint32_t)(ymap_va >> 32);
    if (info) {
        info->rows = rows; info->segments = segs; info->n = n; info->k = k; info->mt = spec.mt; info->warps_per_cta = W; info->lane_bytes = spec.lane_bytes;
        info->grid_x = grid_x; info->grid_y = segs; info->kernel_cache_hit = hit;
    }
    OmegaGpuLaunch L = {
        .code_va = ks->code.va, .gpr_count = gpr_declared(ks->kernel.gpr_count),
        .threads_x = 32u * W, .threads_y = 1, .grid_x = grid_x, .grid_y = segs,
        .num_elements = (size_t)rows * n,
        .args = args, .n_args = spec.rowmap ? OMEGA_GEMM_ARGS_WORDS_MAP : OMEGA_GEMM_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
        .shared_bytes = omega_fp8gemm_shared_bytes(&spec),
    };
    if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); return OMEGA_GPU_FP8GEMM_CHIP_FAIL; }
    OmegaGpuNativeLease lease = omega_native_cache_lease(&g.native, (unsigned)(ks - g.cache));
    if (omega_gpu_cmdlist_add_native(cl, &L, &lease) < 0) return OMEGA_GPU_FP8GEMM_CHIP_FAIL;
    return OMEGA_GPU_FP8GEMM_OK;
}

/* x' = x * 2^64 in the kernel's lane layout: column c = s*32*LB + L*LB + w*4 + e  ->  index s*32*LB + (w*32 + L)*4 + e */
void omega_fp8gemm_permute_x(const float *x, uint32_t k, uint32_t lb, float *out) {
    const uint32_t row = 32u * lb, nw = lb / 4u;
    for (uint32_t s0 = 0; s0 + row <= k; s0 += row)
        for (uint32_t L = 0; L < 32u; L++)
            for (uint32_t w = 0; w < nw; w++) {
                const float *src = x + s0 + L * lb + w * 4u; float *dst = out + s0 + (w * 32u + L) * 4u;
                dst[0] = src[0] * 18446744073709551616.0f; dst[1] = src[1] * 18446744073709551616.0f;
                dst[2] = src[2] * 18446744073709551616.0f; dst[3] = src[3] * 18446744073709551616.0f;
            }
}

typedef struct { uint64_t w_va, s_va; uint32_t row0, cnt; uint32_t pad[2]; } SegEntry;
_Static_assert(sizeof(SegEntry) == OMEGA_GEMM_SEG_BYTES, "segment entry layout");

/* General fallback (GemmGeneral, 2026-10-27): a call the kernel refuses (rows past OMEGA_GPU_FP8GEMM_MAX_ROWS, segments
 * past OMEGA_GPU_FP8GEMM_MAX_SEGMENTS, a k the codegen refuses) runs each segment through omega_gpu_fp8_general_rows:
 * the segment's x rows gathered by token_rows into a host buffer, y rows row0..row0+cnt written directly. */
static int general_gemm(const OmegaGpuFp8Tensor *t, uint32_t M, const uint32_t *ids, const uint32_t *trow, const float *x32,
                        const uint16_t *x16, float *y, uint32_t n, uint32_t k, uint32_t segs, OmegaGpuFp8GemmInfo *info, int refused, uint64_t t0) {
    int rc = OMEGA_GPU_FP8GEMM_OK;
    for (uint32_t i = 0; i < M && rc == OMEGA_GPU_FP8GEMM_OK;) {
        uint32_t j = i; while (j < M && ids[j] == ids[i]) j++;
        const uint32_t cnt = j - i;
        void *xs = omega_gpu_host_calloc((size_t)cnt * k, x32 ? 4u : 2u);
        if (!xs) return OMEGA_GPU_FP8GEMM_BAD_ARGS;
        for (uint32_t r = 0; r < cnt; r++) {
            if (x32) memcpy((float *)xs + (size_t)r * k, x32 + (size_t)trow[i + r] * k, (size_t)k * 4u);
            else memcpy((uint16_t *)xs + (size_t)r * k, x16 + (size_t)trow[i + r] * k, (size_t)k * 2u);
        }
        rc = omega_gpu_fp8_general_rows(t, ids[i], x32 ? xs : NULL, x32 ? NULL : xs, cnt, y + (size_t)i * n, refused);
        omega_gpu_host_free(xs);
        i = j;
    }
    if (info) {
        info->rows = M; info->segments = segs; info->n = n; info->k = k; info->general = rc == OMEGA_GPU_FP8GEMM_OK;
        info->call_ns = omega_gpu_session_now_ns() - t0;
    }
    return rc;
}

static int gemm_core(const OmegaGpuFp8Tensor *t, uint32_t M, const uint32_t *ids, const uint32_t *trow, const float *x32,
                     const uint16_t *x16, uint32_t x_rows, float *y, OmegaGpuFp8GemmInfo *info) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    uint32_t experts = 0, n = 0, k = 0;
    omega_gpu_fp8_tensor_shape(t, &experts, &n, &k);
    if (!t || !experts || !ids || !trow || (!x32 && !x16) || !y || M == 0 || x_rows == 0) return OMEGA_GPU_FP8GEMM_BAD_ARGS;
    /* a call with a test override of the kernel shape or a kernel mutant keeps the refusal: those calls test the kernel */
    const int general_ok = !g.mt && !g.warps && !g.ctas && !g.mutant;
    if (M > OMEGA_GPU_FP8GEMM_MAX_ROWS && !general_ok) return OMEGA_GPU_FP8GEMM_TOO_LARGE;
    uint32_t segs = 0;
    for (uint32_t i = 0; i < M; i++) {
        if (ids[i] >= experts || trow[i] >= x_rows || (i && ids[i] < ids[i - 1])) return OMEGA_GPU_FP8GEMM_BAD_ARGS;
        if (i == 0 || ids[i] != ids[i - 1]) segs++;
    }
    if (M > OMEGA_GPU_FP8GEMM_MAX_ROWS || segs > OMEGA_GPU_FP8GEMM_MAX_SEGMENTS)
        return general_ok ? general_gemm(t, M, ids, trow, x32, x16, y, n, k, segs, info, OMEGA_GPU_FP8GEMM_TOO_LARGE, t0) : OMEGA_GPU_FP8GEMM_TOO_LARGE;
    OmegaFp8GemmSpec spec; spec_for(k, segs, &spec);
    if (general_ok && omega_fp8gemm_shared_bytes(&spec) == 0) return general_gemm(t, M, ids, trow, x32, x16, y, n, k, segs, info, OMEGA_GPU_FP8GEMM_CODEGEN_FAIL, t0);
    const uint32_t MT = spec.mt, W = spec.warps;
    uint32_t grid_x = (n + W - 1) / W;
    uint32_t budget = g.ctas ? g.ctas : (segs >= 64u ? 2u : segs >= 8u ? 12u : 96u);
    if (grid_x > budget) grid_x = budget;
    if (grid_x == 0) grid_x = 1;

    omega_gpu_session_lock();
    int rc = OMEGA_GPU_FP8GEMM_OK;
    if (!dev_open_locked()) { omega_gpu_session_unlock(); return OMEGA_GPU_FP8GEMM_CHIP_FAIL; }
    int src = omega_gpu_session_scratch(&g.x_buf, ((size_t)M + MT) * k * 4u);
    if (src == 0) src = omega_gpu_session_scratch(&g.y_buf, (size_t)M * n * 4u);
    if (src == 0) src = omega_gpu_session_scratch(&g.seg_buf, (size_t)OMEGA_GPU_FP8GEMM_MAX_SEGMENTS * OMEGA_GEMM_SEG_BYTES);
    if (src != 0) { omega_gpu_session_unlock(); return src == OMEGA_GPU_SESSION_SCRATCH_RESERVED ? OMEGA_GPU_FP8GEMM_TOO_LARGE : OMEGA_GPU_FP8GEMM_CHIP_FAIL; }
    Slot *ks = NULL; bool hit = false;
    rc = kernel_for(&spec, &ks, &hit);
    if (rc != OMEGA_GPU_FP8GEMM_OK) { omega_gpu_session_unlock(); return rc; }

    /* segment table, permuted x' rows, poisoned y */
    SegEntry *tab = g.seg_buf.mem.cpu;
    uint64_t streamed = 0, distinct = 0; uint32_t sg = 0, passes_max = 0;
    for (uint32_t i = 0; i < M;) {
        uint32_t j = i; while (j < M && ids[j] == ids[i]) j++;
        memset(&tab[sg], 0, sizeof tab[sg]);
        (void)omega_gpu_fp8_tensor_entry(t, ids[i], &tab[sg].w_va, &tab[sg].s_va);
        tab[sg].row0 = i; tab[sg].cnt = j - i;
        uint32_t passes = (j - i + MT - 1) / MT; if (passes > passes_max) passes_max = passes;
        streamed += (uint64_t)passes * n * k; distinct += (uint64_t)n * k;
        sg++; i = j;
    }
    float *xd = g.x_buf.mem.cpu, *yd = g.y_buf.mem.cpu;
    {
        float tmp[OMEGA_FP8_MAX_K], tp[OMEGA_FP8_MAX_K]; /* permute in cached memory, then one sequential copy: element-wise writes into the uncached buffer cost 2.8 ms per 4 MB */
        /* a source row routed to several experts (top-k) is permuted once and copied after: the permuted rows are kept in cached memory */
        float *pc = ((size_t)x_rows * k * 4u <= (16u << 20)) ? malloc((size_t)x_rows * k * 4u) : NULL;
        unsigned char *have = pc ? calloc(x_rows, 1) : NULL;
        for (uint32_t i = 0; i < M; i++) {
            const uint32_t r = trow[i];
            if (have && have[r]) { memcpy(xd + (size_t)i * k, pc + (size_t)r * k, (size_t)k * 4u); continue; }
            if (x32) memcpy(tmp, x32 + (size_t)r * k, (size_t)k * 4u);
            else for (uint32_t c = 0; c < k; c++) { uint32_t b = (uint32_t)x16[(size_t)r * k + c] << 16; memcpy(&tmp[c], &b, 4); }
            float *dst = have ? pc + (size_t)r * k : tp;
            omega_fp8gemm_permute_x(tmp, k, spec.lane_bytes, dst);
            if (have) have[r] = 1;
            memcpy(xd + (size_t)i * k, dst, (size_t)k * 4u);
        }
        free(pc); free(have);
        memset(xd + (size_t)M * k, 0, (size_t)MT * k * 4u);
    }
    for (size_t i = 0; i < (size_t)M * n; i++) yd[i] = POISON_F32;
    if (omega_gpu_session_sync_to_gpu(&g.seg_buf.mem) != 0 || omega_gpu_session_sync_to_gpu(&g.x_buf.mem) != 0 || omega_gpu_session_sync_to_gpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_FP8GEMM_CHIP_FAIL;

    uint32_t args[OMEGA_GEMM_ARGS_WORDS];
    args[0] = (uint32_t)g.seg_buf.mem.va; args[1] = (uint32_t)(g.seg_buf.mem.va >> 32);
    args[2] = (uint32_t)g.x_buf.mem.va; args[3] = (uint32_t)(g.x_buf.mem.va >> 32);
    args[4] = (uint32_t)g.y_buf.mem.va; args[5] = (uint32_t)(g.y_buf.mem.va >> 32);
    args[6] = n; args[7] = grid_x * W;
    if (info) {
        info->rows = M; info->segments = sg; info->n = n; info->k = k; info->mt = MT; info->warps_per_cta = W; info->lane_bytes = spec.lane_bytes;
        info->grid_x = grid_x; info->grid_y = sg; info->passes_max = passes_max;
        info->weight_bytes = distinct; info->streamed_bytes = streamed; info->kernel_cache_hit = hit;
    }
    if (rc == OMEGA_GPU_FP8GEMM_OK) {
        OmegaGpuLaunch L = {
            .code_va = ks->code.va, .gpr_count = gpr_declared(ks->kernel.gpr_count),
            .threads_x = 32u * W, .threads_y = 1, .grid_x = grid_x, .grid_y = sg,
            .num_elements = (size_t)M * n,
            .args = args, .n_args = OMEGA_GEMM_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
            .shared_bytes = omega_fp8gemm_shared_bytes(&spec),
        };
        if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); rc = OMEGA_GPU_FP8GEMM_CHIP_FAIL; }
        else if (omega_gpu_session_launch(&L, info ? &info->elapsed_ns : NULL, info ? &info->completion_marker : NULL) != 0) rc = OMEGA_GPU_FP8GEMM_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_FP8GEMM_OK && omega_gpu_session_sync_to_cpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_FP8GEMM_CHIP_FAIL;
    if (rc == OMEGA_GPU_FP8GEMM_OK) {
        size_t unchanged = 0;
        for (size_t i = 0; i < (size_t)M * n; i++) if (yd[i] == POISON_F32) unchanged++;
        if (unchanged) { omega_gpu_session_set_error("output poison survived"); rc = OMEGA_GPU_FP8GEMM_CHIP_FAIL; }
        else memcpy(y, yd, (size_t)M * n * sizeof *y);
    }
    omega_gpu_session_unlock();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}

int omega_gpu_fp8gemm_f32(const OmegaGpuFp8Tensor *t, uint32_t M, const uint32_t *ids, const uint32_t *trow, const float *x, uint32_t x_rows, float *y, OmegaGpuFp8GemmInfo *info) {
    return gemm_core(t, M, ids, trow, x, NULL, x_rows, y, info);
}
int omega_gpu_fp8gemm_bf16(const OmegaGpuFp8Tensor *t, uint32_t M, const uint32_t *ids, const uint32_t *trow, const uint16_t *x, uint32_t x_rows, float *y, OmegaGpuFp8GemmInfo *info) {
    return gemm_core(t, M, ids, trow, NULL, x, x_rows, y, info);
}

int omega_gpu_fp8gemm_native_code_observe(const OmegaGpuNativeLease *lease, OmegaGpuMem *out) {
    if (!out) return -1;
    *out = (OmegaGpuMem){0};
    if (!lease || lease->cache != &g.native || omega_native_lease_op(lease, 16, 0) != 0)
        return -1;
    const Slot *s = &g.cache[lease->slot];
    if (!s->exact) return -1;
    *out = s->code;
    return 0;
}

int omega_gpu_fp8gemm_record_native(OmegaGpuCmdList *cl, const OmegaGpuFp8Tensor *t,
    const OmegaGpuNativeArtifact *a, const OmegaGpuNativeLease *lease,
    uint32_t rows, uint32_t segs, uint64_t table, uint64_t xp, uint64_t y,
    uint64_t xmap, uint64_t ymap) {
    OmegaGpuNativeTensorObservation owner;
    if (omega_gpu_fp8_native_observe(t, &owner) != 0) return -1;
    const uint32_t n = owner.n, k = owner.k;
    if (!cl || !t || !rows || !segs || !table || !xp || !y || (!!xmap != !!ymap) ||
        !lease || lease->cache != &g.native || omega_native_bound(a, lease) != 0) return -1;
    uint64_t call[12] = {2, rows, n, k, segs, xmap ? 12u : 8u, xmap ? 1u : 0u, 0,
        owner.allocation_epoch, omega_gpu_session_epoch(), owner.owner_generation, 0}, launch[32];
    if (omega_native_launch(a, call, launch) != 0) return -1;
    Slot *s = &g.cache[lease->slot];
    if (!s->exact || memcmp(s->manifest, a->manifest, 32)) return -1;
    uint32_t args[12] = {(uint32_t)table, (uint32_t)(table >> 32),
        (uint32_t)xp, (uint32_t)(xp >> 32), (uint32_t)y, (uint32_t)(y >> 32),
        n, (uint32_t)launch[29], (uint32_t)xmap, (uint32_t)(xmap >> 32),
        (uint32_t)ymap, (uint32_t)(ymap >> 32)};
    OmegaGpuLaunch L = {.code_va = s->code.va, .gpr_count = (uint32_t)launch[8],
        .threads_x = (uint32_t)launch[4], .threads_y = (uint32_t)launch[5],
        .grid_x = (uint32_t)launch[1], .grid_y = (uint32_t)launch[2],
        .num_elements = (size_t)rows * n, .args = args, .n_args = (uint32_t)call[5],
        .timeout_ms = MARKER_TIMEOUT_MS, .shared_bytes = (uint32_t)launch[7]};
    return omega_gpu_cmdlist_add_native(cl, &L, lease);
}
