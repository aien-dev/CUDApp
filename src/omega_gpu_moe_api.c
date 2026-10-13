/*
 * omega_gpu_moe_api.c: column windows over a resident bf16 tensor (OWN-30B cut 2). See the header.
 *
 * Same submission path, buffers, poison check and latch rules as omega_gpu_matmul_api.c (FB-1 cut 1b
 * on the shared session), restated here rather than shared because that file is owned by the 8B
 * decode-speed lane and its tensor type is opaque; this file only ADDS exports. Every buffer both
 * sides touch is GPU-uncached (nvos.h NVOS32_ATTR2_GPU_CACHEABLE: "For system memory this will not
 * be coherent with direct CPU mappings").
 *
 * The window rule: B is in fragment order (omega_matmul_frag_pack_b), column tile nt = kp x 8
 * elements contiguous at element nt * kp * 8. A window starting at column tile t0 of width wnp
 * (padded to 8) is therefore the chip-proven kernel for (kp, wnp) with b_va advanced by
 * t0 * kp * 8 * 2 bytes. Nothing else about the launch changes.
 */
#include "omega_gpu_moe_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_fmt.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_matmul.h"
#include "omega_blackwell_qmd.h"
#include <string.h>

#define MOE_CACHE_SLOTS 16      /* default kernel cache slots; raised only by omega_gpu_moe_reserve */
#define MOE_CACHE_SLOTS_MAX 128
#define FRAG_PACK_TILES 64u     /* column tiles per host packing chunk at upload */
#define MARKER_TIMEOUT_MS 20000ull
#define POISON_F32 (-999.0f)

struct OmegaGpuMoeTensor {
    uint32_t k, n, kp, np;
    OmegaGpuMem mem; /* kp x np bf16 in fragment order */
};

typedef struct {
    bool valid, pinned;
    uint32_t kp, np, grid_x, rt, ct;
    int mutant;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
} MoeCacheSlot;

static struct {
    OmegaGpuScratch a_buf, c_buf;
    bool a_row0_ready; uint32_t a_row0_kp; const void *a_row0_cpu;
    uint64_t a_row0_hits;
    uint32_t cta_budget;
    int mutant;
    int32_t skew; /* test only: column tiles added to every window's B base */
    bool hooked;
    MoeCacheSlot cache[MOE_CACHE_SLOTS_MAX];
    unsigned cache_next, slots;
    int sealed;
} g = { .cta_budget = OMEGA_GPU_MATMUL_MAX_CTAS, .slots = MOE_CACHE_SLOTS };

#define FAILAT(s) (omega_gpu_session_set_error(s), 1)
static uint64_t now_ns(void) { return omega_gpu_session_now_ns(); }
static uint32_t round_up(uint32_t v, uint32_t q) { return (v + q - 1) / q * q; }

uint64_t omega_gpu_moe_window_byte_offset(uint32_t kp, uint32_t col0) {
    return (uint64_t)(col0 / OMEGA_GPU_MATMUL_TILE_N) * kp * OMEGA_GPU_MATMUL_TILE_N * 2u;
}

/* ---- device-side state (caller holds the session lock) ---- */

static void cache_clear_locked(void) {
    for (unsigned i = 0; i < MOE_CACHE_SLOTS_MAX; i++) {
        MoeCacheSlot *s = &g.cache[i];
        if (!s->valid) continue;
        omega_gpu_session_free(&s->code);
        memset(s, 0, sizeof *s);
    }
    g.cache_next = 0;
}

static void on_session_close(void) {
    cache_clear_locked();
    g.a_row0_ready = false;
    omega_gpu_session_scratch_free(&g.a_buf);
    omega_gpu_session_scratch_free(&g.c_buf);
    g.slots = MOE_CACHE_SLOTS;
    g.sealed = 0;
}

static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}

void omega_gpu_moe_cache_clear(void) {
    omega_gpu_session_lock();
    cache_clear_locked();
    omega_gpu_session_unlock();
}

void omega_gpu_moe_set_cta_budget(uint32_t ctas) {
    omega_gpu_session_lock();
    g.cta_budget = ctas ? ctas : OMEGA_GPU_MATMUL_MAX_CTAS;
    cache_clear_locked();
    omega_gpu_session_unlock();
}

uint32_t omega_gpu_moe_cta_budget(void) { return g.cta_budget; }

void omega_gpu_moe_test_set_mutant(int on) {
    omega_gpu_session_lock();
    g.mutant = on ? 1 : 0;
    cache_clear_locked();
    omega_gpu_session_unlock();
}

void omega_gpu_moe_test_set_window_skew(int32_t tiles) {
    omega_gpu_session_lock();
    g.skew = tiles;
    omega_gpu_session_unlock();
}

uint64_t omega_gpu_moe_test_a_row0_hits(void) { return g.a_row0_hits; }

/* ---- launch shape and kernel cache (the matmul API's rules, restated) ---- */

static void launch_shape(uint32_t mp, uint32_t np, uint32_t *rows_per_launch, uint32_t *grid_x, uint32_t *rt, uint32_t *ct) {
    const uint32_t budget = g.cta_budget ? g.cta_budget : OMEGA_GPU_MATMUL_MAX_CTAS, mt = mp / 16u, nt = np / 8u;
    omega_matmul_frag_block_shape(mt, nt, rt, ct);
    const uint32_t rb = mt / *rt, nb = nt / *ct;
    *rows_per_launch = rb <= budget ? mp : budget * 16u * *rt;
    uint32_t gx = budget / (*rows_per_launch / 16u / *rt);
    if (gx == 0) gx = 1;
    if (gx > nb) gx = nb;
    *grid_x = gx;
}

static int kernel_for(uint32_t kp, uint32_t np, uint32_t grid_x, uint32_t rt, uint32_t ct, MoeCacheSlot **out, bool *hit) {
    for (unsigned i = 0; i < g.slots; i++) {
        MoeCacheSlot *s = &g.cache[i];
        if (s->valid && s->kp == kp && s->np == np && s->grid_x == grid_x && s->rt == rt && s->ct == ct && s->mutant == g.mutant) {
            *out = s; *hit = true;
            return OMEGA_GPU_MATMUL_OK;
        }
    }
    if (g.sealed) {
        char e[160];
        omega_gpu_fmt(e, sizeof e, "serving reservation exceeded: moe window kernel kp=%u np=%u grid_x=%u rt=%u ct=%u was not prepared", kp, np, grid_x, rt, ct);
        omega_gpu_session_set_error(e);
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    }
    MoeCacheSlot *s = NULL;
    for (unsigned i = 0; i < g.slots && !s; i++) {
        MoeCacheSlot *c = &g.cache[(g.cache_next + i) % g.slots];
        if (!c->valid || !c->pinned) { s = c; g.cache_next = (unsigned)((c - g.cache) + 1) % g.slots; }
    }
    if (!s) {
        omega_gpu_session_set_error("moe window kernel cache: every slot holds a prepared kernel (raise kernel_slots)");
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    }
    OmegaMatMulSpec spec;
    if (omega_matmul_spec_init(&spec, 16, kp, np, OMEGA_MATMUL_PRECISION_BF16) != 0) return OMEGA_GPU_MATMUL_TOO_LARGE;
    OmegaBlackwellKernel kernel;
    memset(&kernel, 0, sizeof kernel);
    const bool blocked = rt > 1 || ct > 1;
    uint32_t unroll = blocked ? omega_matmul_frag_block_unroll(kp, rt, ct) : omega_matmul_frag_unroll(kp);
    if (g.mutant) while (unroll > 1 && kp / 16u / unroll < 2) unroll /= 2;
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("moe window codegen: host memory"); return OMEGA_GPU_MATMUL_CODEGEN_FAIL; }
    int cg = blocked ? omega_blackwell_codegen_matmul_frag_block_prog(&spec, grid_x, unroll, rt, ct, g.mutant, prog)
                     : omega_blackwell_codegen_matmul_frag_prog(&spec, grid_x, unroll, g.mutant, prog);
    const size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); return OMEGA_GPU_MATMUL_CODEGEN_FAIL; }
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need) { omega_gpu_session_free(&code); return OMEGA_GPU_MATMUL_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    if (s->valid) omega_gpu_session_free(&s->code);
    s->valid = true; s->pinned = false; s->kp = kp; s->np = np; s->grid_x = grid_x; s->rt = rt; s->ct = ct; s->mutant = g.mutant;
    s->kernel = kernel; s->code = code;
    *out = s; *hit = false;
    return OMEGA_GPU_MATMUL_OK;
}

static int launch(const MoeCacheSlot *ks, uint64_t a_va, uint64_t b_va, uint64_t c_va,
                  uint32_t grid_x, uint32_t grid_y, uint64_t *elapsed_ns, uint32_t *marker_out) {
    uint32_t cbank_args[OMEGA_BW_CBANK_MATMUL_ARGS_WORDS];
    omega_blackwell_build_cbank_args_matmul(cbank_args, a_va, b_va, c_va, grid_y * 16u * ks->rt, ks->kp, ks->np);
    OmegaGpuLaunch L = {
        .code_va = ks->code.va, .gpr_count = omega_bw_gpr_declared(ks->kernel.gpr_count), /* peak + 2 rounded up to 8 (the session passed the raw peak before) */
        .threads_x = 32, .threads_y = 1, .grid_x = grid_x, .grid_y = grid_y,
        .num_elements = (size_t)grid_x * grid_y * 128u,
        .args = cbank_args, .n_args = OMEGA_BW_CBANK_MATMUL_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
    };
    if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    return omega_gpu_session_launch(&L, elapsed_ns, marker_out) == 0 ? OMEGA_GPU_MATMUL_OK : OMEGA_GPU_MATMUL_CHIP_FAIL;
}

/* ---- serving reservation ---- */

int omega_gpu_moe_reserve(uint32_t max_rows, uint32_t max_k, uint32_t max_cols, uint32_t kernel_slots) {
    if (max_rows == 0 || max_k == 0 || max_cols == 0) { omega_gpu_session_set_error("moe reserve: zero bound"); return OMEGA_GPU_MATMUL_BAD_ARGS; }
    if (max_rows > OMEGA_BW_MATMUL_MAX_M || max_k > OMEGA_BW_MATMUL_MAX_K || max_cols > OMEGA_BW_MATMUL_MAX_N || kernel_slots > MOE_CACHE_SLOTS_MAX) {
        omega_gpu_session_set_error("moe reserve: bound past the matmul envelope");
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    }
    const size_t mp = round_up(max_rows, OMEGA_GPU_MATMUL_TILE_M), kp = round_up(max_k, OMEGA_GPU_MATMUL_TILE_K);
    const size_t c_bytes = mp * round_up(max_cols, OMEGA_GPU_MATMUL_TILE_N) * 4u;
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_MATMUL_OK;
    if (!dev_open_locked()) rc = OMEGA_GPU_MATMUL_CHIP_FAIL;
    else if (omega_gpu_session_scratch_reserve(&g.a_buf, mp * kp * 2u) != 0 || omega_gpu_session_scratch_reserve(&g.c_buf, c_bytes) != 0) {
        omega_gpu_session_scratch_unreserve(&g.a_buf); omega_gpu_session_scratch_unreserve(&g.c_buf);
        rc = OMEGA_GPU_MATMUL_CHIP_FAIL;
    } else if (kernel_slots > g.slots) {
        cache_clear_locked();
        g.slots = kernel_slots;
    }
    g.a_row0_ready = false;
    omega_gpu_session_unlock();
    return rc;
}

void omega_gpu_moe_unreserve(void) {
    omega_gpu_session_lock();
    omega_gpu_session_scratch_unreserve(&g.a_buf);
    omega_gpu_session_scratch_unreserve(&g.c_buf);
    omega_gpu_session_unlock();
}

int omega_gpu_moe_prepare(uint32_t m, uint32_t k, uint32_t cols) {
    if (m == 0 || k == 0 || cols == 0) { omega_gpu_session_set_error("moe prepare: zero dimension"); return OMEGA_GPU_MATMUL_BAD_ARGS; }
    if (m > OMEGA_BW_MATMUL_MAX_M || k > OMEGA_BW_MATMUL_MAX_K || cols > OMEGA_BW_MATMUL_MAX_N) {
        omega_gpu_session_set_error("moe prepare: shape past the matmul envelope");
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    }
    const uint32_t mp = round_up(m, OMEGA_GPU_MATMUL_TILE_M), kp = round_up(k, OMEGA_GPU_MATMUL_TILE_K), np = round_up(cols, OMEGA_GPU_MATMUL_TILE_N);
    uint32_t rows_per_launch, grid_x, rt, ct;
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_MATMUL_OK;
    if (!dev_open_locked()) rc = OMEGA_GPU_MATMUL_CHIP_FAIL;
    else {
        launch_shape(mp, np, &rows_per_launch, &grid_x, &rt, &ct);
        MoeCacheSlot *s = NULL; bool hit = false;
        rc = kernel_for(kp, np, grid_x, rt, ct, &s, &hit);
        if (rc == OMEGA_GPU_MATMUL_OK) s->pinned = true;
    }
    omega_gpu_session_unlock();
    return rc;
}

void omega_gpu_moe_seal(int on) {
    omega_gpu_session_lock();
    g.sealed = on ? 1 : 0;
    omega_gpu_session_unlock();
}

/* ---- tensors ---- */

void omega_gpu_moe_tensor_shape(const OmegaGpuMoeTensor *t, uint32_t *k, uint32_t *n) {
    if (k) *k = t ? t->k : 0;
    if (n) *n = t ? t->n : 0;
}

int omega_gpu_moe_tensor_upload_bf16(uint32_t k, uint32_t n, const uint16_t *b, OmegaGpuMoeTensor **out) {
    if (out) *out = NULL;
    if (!b || !out || k == 0 || n == 0) return OMEGA_GPU_MATMUL_BAD_ARGS;
    if (k > OMEGA_BW_MATMUL_MAX_K || n > OMEGA_BW_MATMUL_MAX_N) return OMEGA_GPU_MATMUL_TOO_LARGE;
    OmegaGpuMoeTensor *t = omega_gpu_host_calloc(1, sizeof *t);
    if (!t) return OMEGA_GPU_MATMUL_BAD_ARGS;
    t->k = k; t->n = n; t->kp = round_up(k, OMEGA_GPU_MATMUL_TILE_K); t->np = round_up(n, OMEGA_GPU_MATMUL_TILE_N);
    const uint32_t nt = t->np / 8u;
    uint16_t *h = omega_gpu_host_alloc((size_t)FRAG_PACK_TILES * t->kp * 8u * sizeof *h);
    if (!h) { omega_gpu_host_free(t); return OMEGA_GPU_MATMUL_BAD_ARGS; }
    int rc = OMEGA_GPU_MATMUL_OK;
    omega_gpu_session_lock();
    if (!dev_open_locked() || omega_gpu_session_alloc((size_t)t->kp * t->np * 2u, &t->mem) != 0) rc = OMEGA_GPU_MATMUL_CHIP_FAIL;
    omega_gpu_session_unlock();
    if (rc != OMEGA_GPU_MATMUL_OK) { omega_gpu_host_free(h); omega_gpu_host_free(t); return rc; }
    uint16_t *d = t->mem.cpu;
    for (uint32_t nt0 = 0; nt0 < nt; nt0 += FRAG_PACK_TILES) {
        const uint32_t nt1 = nt - nt0 < FRAG_PACK_TILES ? nt : nt0 + FRAG_PACK_TILES;
        omega_matmul_frag_pack_b(b, k, n, t->kp, nt0, nt1, h);
        memcpy(&d[(size_t)nt0 * t->kp * 8u], h, (size_t)(nt1 - nt0) * t->kp * 8u * sizeof *h);
    }
    omega_gpu_host_free(h);
    if (omega_gpu_session_sync_to_gpu(&t->mem) != 0) { omega_gpu_moe_tensor_free(t); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    *out = t;
    return OMEGA_GPU_MATMUL_OK;
}

int omega_gpu_moe_tensor_upload_f32(uint32_t k, uint32_t n, const float *b, OmegaGpuMoeTensor **out) {
    if (out) *out = NULL;
    if (!b || !out || k == 0 || n == 0) return OMEGA_GPU_MATMUL_BAD_ARGS;
    if (k > OMEGA_BW_MATMUL_MAX_K || n > OMEGA_BW_MATMUL_MAX_N) return OMEGA_GPU_MATMUL_TOO_LARGE;
    size_t cnt = (size_t)k * n;
    uint16_t *h = omega_gpu_host_alloc(cnt * sizeof *h);
    if (!h) return OMEGA_GPU_MATMUL_BAD_ARGS;
    for (size_t i = 0; i < cnt; i++) h[i] = omega_fp32_to_bf16(b[i]);
    int rc = omega_gpu_moe_tensor_upload_bf16(k, n, h, out);
    omega_gpu_host_free(h);
    return rc;
}

void omega_gpu_moe_tensor_free(OmegaGpuMoeTensor *t) {
    if (!t) return;
    omega_gpu_session_lock();
    omega_gpu_session_free(&t->mem);
    omega_gpu_session_unlock();
    omega_gpu_host_free(t);
}

/* ---- the window matmul ---- */

/* The alignment rule, fail-closed, with the stage named. */
static int window_args_ok(const OmegaGpuMoeTensor *t, uint32_t col0, uint32_t cols) {
    char e[160];
    if (col0 % OMEGA_GPU_MATMUL_TILE_N != 0) {
        omega_gpu_fmt(e, sizeof e, "moe window: col0=%u is not a multiple of %u (a window starts on a column tile)", col0, OMEGA_GPU_MATMUL_TILE_N);
        omega_gpu_session_set_error(e);
        return 0;
    }
    if (cols == 0 || col0 > t->n || cols > t->n - col0) {
        omega_gpu_fmt(e, sizeof e, "moe window: [%u, %u + %u) is not inside the tensor's %u columns", col0, col0, cols, t->n);
        omega_gpu_session_set_error(e);
        return 0;
    }
    return 1;
}

/* a16 or a32 (exactly one non-NULL). nwin windows of `cols` columns at col0[w]; c receives nwin
 * blocks of m x cols. The activation pack and the kernel are shared by every window of the call. */
static int windows_core(uint32_t m, const uint16_t *a16, const float *a32, const OmegaGpuMoeTensor *t,
                        uint32_t nwin, const uint32_t *col0, uint32_t cols, float *c, OmegaGpuMatmulInfo *info) {
    const uint32_t k = t->k, kp = t->kp;
    const uint32_t mp = round_up(m, OMEGA_GPU_MATMUL_TILE_M), np = round_up(cols, OMEGA_GPU_MATMUL_TILE_N);
    for (uint32_t w = 0; w < nwin; w++) if (!window_args_ok(t, col0[w], cols)) return OMEGA_GPU_MATMUL_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_MATMUL_OK;
    if (!dev_open_locked()) { omega_gpu_session_unlock(); return OMEGA_GPU_MATMUL_CHIP_FAIL; }
    uint32_t rows_per_launch, grid_x, rt, ct;
    launch_shape(mp, np, &rows_per_launch, &grid_x, &rt, &ct);
    if (info) {
        info->m = m; info->k = k; info->n = cols; info->padded_m = mp; info->padded_n = np; info->padded_k = kp;
        info->k_slices = 1; info->rows_per_call = rows_per_launch; info->grid_x = grid_x; info->grid_y = rows_per_launch / 16u / rt;
        info->resident = true; info->kernel_cache_hit = true;
        (void)omega_gpu_fmt(info->target_chip, sizeof info->target_chip, "%s", "NVIDIA DGX Spark (Grace Blackwell GB10)");
        info->sm_architecture = 121;
    }
    if (g.sealed && !(g.a_buf.reserved && g.c_buf.reserved)) {
        omega_gpu_session_set_error("serving reservation exceeded: moe window sealed without a scratch reservation (call omega_gpu_moe_reserve first)");
        omega_gpu_session_unlock();
        return OMEGA_GPU_MATMUL_TOO_LARGE;
    }
    int src = omega_gpu_session_scratch(&g.a_buf, (size_t)mp * kp * 2u);
    if (src == 0) src = omega_gpu_session_scratch(&g.c_buf, (size_t)mp * np * 4u);
    if (src != 0) {
        omega_gpu_session_unlock();
        return src == OMEGA_GPU_SESSION_SCRATCH_RESERVED ? OMEGA_GPU_MATMUL_TOO_LARGE : OMEGA_GPU_MATMUL_CHIP_FAIL;
    }
    uint16_t *ad = g.a_buf.mem.cpu;
    const bool one_row = m == 1 && mp == 16u;
    if (one_row && g.a_row0_ready && g.a_row0_kp == kp && g.a_row0_cpu == ad) {
        if (a32) omega_matmul_frag_pack_a_row0_f32(a32, k, kp, ad);
        else omega_matmul_frag_pack_a_row0(a16, k, kp, ad);
        g.a_row0_hits++;
    } else {
        if (a32) omega_matmul_frag_pack_a_f32(a32, m, k, mp, kp, ad);
        else omega_matmul_frag_pack_a(a16, m, k, mp, kp, ad);
        g.a_row0_ready = one_row; g.a_row0_kp = kp; g.a_row0_cpu = ad;
    }
    if (omega_gpu_session_sync_to_gpu(&g.a_buf.mem) != 0) rc = OMEGA_GPU_MATMUL_CHIP_FAIL;
    float *cd = g.c_buf.mem.cpu;
    for (uint32_t w = 0; w < nwin && rc == OMEGA_GPU_MATMUL_OK; w++) {
        /* the window's B base: column tile col0 / 8 (plus the test skew, which must be caught) */
        const int64_t tile0 = (int64_t)(col0[w] / OMEGA_GPU_MATMUL_TILE_N) + g.skew;
        const uint64_t b_va = t->mem.va + (uint64_t)tile0 * kp * OMEGA_GPU_MATMUL_TILE_N * 2u;
        omega_matmul_result_sentinels(cd, m, cols, np, POISON_F32, 1);
        if (omega_gpu_session_sync_to_gpu(&g.c_buf.mem) != 0) { rc = OMEGA_GPU_MATMUL_CHIP_FAIL; break; }
        for (uint32_t r0 = 0; r0 < mp && rc == OMEGA_GPU_MATMUL_OK; r0 += rows_per_launch) {
            uint32_t rows = mp - r0 < rows_per_launch ? mp - r0 : rows_per_launch;
            uint32_t grid_y = rows / 16u / rt;
            MoeCacheSlot *ks = NULL; bool hit = false;
            rc = kernel_for(kp, np, grid_x, rt, ct, &ks, &hit);
            if (rc != OMEGA_GPU_MATMUL_OK) break;
            if (info && !hit) info->kernel_cache_hit = false;
            rc = launch(ks, g.a_buf.mem.va + (uint64_t)r0 * kp * 2u, b_va, g.c_buf.mem.va + (uint64_t)r0 * np * 4u,
                        grid_x, grid_y, info ? &info->elapsed_ns : NULL, info ? &info->completion_marker : NULL);
            if (info) info->chip_calls++;
        }
        if (rc != OMEGA_GPU_MATMUL_OK) break;
        if (omega_gpu_session_sync_to_cpu(&g.c_buf.mem) != 0) { rc = OMEGA_GPU_MATMUL_CHIP_FAIL; break; }
        size_t unchanged = omega_matmul_result_sentinels(cd, m, cols, np, POISON_F32, 0);
        if (unchanged && FAILAT("output poison survived")) { rc = OMEGA_GPU_MATMUL_CHIP_FAIL; break; }
        float *cw = c + (size_t)w * m * cols;
        for (uint32_t r = 0; r < m; r++) memcpy(&cw[(size_t)r * cols], &cd[(size_t)r * np], (size_t)cols * sizeof *c);
    }
    omega_gpu_session_unlock();
    return rc;
}

static int entry(uint32_t m, const uint16_t *a16, const float *a32, const OmegaGpuMoeTensor *t,
                 uint32_t nwin, const uint32_t *col0, uint32_t cols, float *c, OmegaGpuMatmulInfo *info) {
    uint64_t t0 = now_ns();
    if (info) memset(info, 0, sizeof *info);
    if ((!a16 && !a32) || !t || !c || !col0 || m == 0 || nwin == 0 || !t->mem.cpu) return OMEGA_GPU_MATMUL_BAD_ARGS;
    if (nwin > OMEGA_GPU_MOE_MAX_WINDOWS) { omega_gpu_session_set_error("moe window: too many windows in one call"); return OMEGA_GPU_MATMUL_BAD_ARGS; }
    if (m > OMEGA_BW_MATMUL_MAX_M) return OMEGA_GPU_MATMUL_TOO_LARGE;
    int rc = windows_core(m, a16, a32, t, nwin, col0, cols, c, info);
    if (info) info->call_ns = now_ns() - t0;
    return rc;
}

int omega_gpu_matmul_resident_window_f32(uint32_t m, const float *a, const OmegaGpuMoeTensor *t,
                                         uint32_t col0, uint32_t cols, float *c, OmegaGpuMatmulInfo *info) {
    return entry(m, NULL, a, t, 1, &col0, cols, c, info);
}

int omega_gpu_matmul_resident_window_bf16(uint32_t m, const uint16_t *a, const OmegaGpuMoeTensor *t,
                                          uint32_t col0, uint32_t cols, float *c, OmegaGpuMatmulInfo *info) {
    return entry(m, a, NULL, t, 1, &col0, cols, c, info);
}

int omega_gpu_matmul_resident_windows_f32(uint32_t m, const float *a, const OmegaGpuMoeTensor *t,
                                          uint32_t nwin, const uint32_t *col0, uint32_t cols, float *c,
                                          OmegaGpuMatmulInfo *info) {
    return entry(m, NULL, a, t, nwin, col0, cols, c, info);
}
