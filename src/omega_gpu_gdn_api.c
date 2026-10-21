/*
 * omega_gpu_gdn_api.c: resident Gated DeltaNet parameters, tails, states and the one-launch decode step (see the
 * header). Session use mirrors omega_gpu_router_api.c: buffers from omega_gpu_session_alloc (GPU-uncached, host-visible),
 * sync_to_gpu after host writes, sync_to_cpu before host reads, one omega_gpu_session_launch per call; the tails and states
 * are the exception (GPU-cached, reached by the host only through the copy / zero launches).
 */
#include "omega_gpu_gdn_api.h"
#include "omega_gpu_hostmath.h"
#include "omega_gpu_session.h"
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_codegen_tc.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "omega_gpu_mem_cached.h"

#define MARKER_TIMEOUT_MS 20000ull
#define POISON_F32 (-999.0f)

/* the launch buffer: one allocation, fixed offsets (bytes) */
#define LB_QKV 0u
#define LB_Z (LB_QKV + OMEGA_GDN_CONV_DIM * 4u)
#define LB_AB (LB_Z + OMEGA_GDN_V_DIM * 4u)
#define LB_OUT (LB_AB + OMEGA_GDN_VALUE_HEADS * 8u)
#define LB_BYTES (LB_OUT + OMEGA_GDN_V_DIM * 4u)

struct OmegaGpuGdn {
    uint32_t layers;
    uint32_t ratio;    /* OmegaGdnSpec.ratio (0 = 3, the 27B) */
    OmegaGdnShape sh;
    OmegaGpuMem par;   /* [layers][OMEGA_GDN_PAR_BYTES] */
    OmegaGpuMem tail;  /* [layers][OMEGA_GDN_TAIL_BYTES] */
    OmegaGpuMem state; /* [layers][OMEGA_GDN_STATE_BYTES] */
    OmegaGpuMem lb;    /* launch buffer */
    int cached;        /* tail and state GPU-cached: the host reaches them only through the chip (reset, stage) */
    int bf16;          /* L313: the state is stored bf16 (AIEN_OMEGA_GDN_STATE_BF16=1 at alloc); set/get_state keep the f32 layout */
    OmegaGpuMem stage; /* cached: uncached staging for set_state / get_state, [state of one layer][tail of one layer] */
    OmegaGpuScratch pf; /* prefill scratch (see pf_layout) */
};

typedef struct {
    bool valid;
    OmegaGdnSpec spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
    uint32_t insns;
} Slot;

/* the resident kernels: decode, prefill prep, scan, post; the state copy and zero */
enum { KK_DECODE, KK_PREP, KK_SCAN, KK_POST, KK_COPY, KK_ZERO, KK_MULTI, KK_N };
static const uint32_t kk_threads[KK_N] = { OMEGA_GDN_THREADS, OMEGA_GDN_PREP_THREADS, OMEGA_GDN_THREADS, OMEGA_GDN_THREADS,
                                           OMEGA_GDN_COPY_THREADS, OMEGA_GDN_COPY_THREADS, OMEGA_GDN_THREADS };
static struct {
    bool hooked;
    Slot slot[KK_N];
    int mutant;
} g_k;
static int g_rec; /* 1 while a record call runs (lock held): kernel_for must not evict */

static void slot_clear_locked(void) {
    for (unsigned k = 0; k < KK_N; k++) { if (g_k.slot[k].valid) omega_gpu_session_free(&g_k.slot[k].code); memset(&g_k.slot[k], 0, sizeof g_k.slot[k]); }
}
static void on_session_close(void) { memset(g_k.slot, 0, sizeof g_k.slot); }
static OmegaGpuSession *dev_open_locked(void) {
    if (!g_k.hooked) { (void)omega_gpu_session_on_close(on_session_close); g_k.hooked = true; }
    return omega_gpu_session_open();
}

void omega_gpu_gdn_test_set_mutant(int mutant) { omega_gpu_session_lock(); g_k.mutant = mutant; slot_clear_locked(); omega_gpu_session_unlock(); }
const char *omega_gpu_gdn_last_error(void) { return omega_gpu_session_last_error(); }
const char *omega_gpu_gdn_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_GDN_OK: return "OK";
    case OMEGA_GPU_GDN_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_GDN_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_GDN_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_GDN_CHIP_FAIL: return "CHIP_FAIL";
    default: return "?";
    }
}

/* ---- host layout ---- */
/* the qkv channel slot s of value head h reads */
static uint32_t slot_channel(const OmegaGdnShape *shp, uint32_t h, uint32_t s) {
    const uint32_t hk = h / (shp->vh / OMEGA_GDN_KEY_HEADS), D = OMEGA_GDN_HEAD_DIM;
    if (s < D) return hk * D + s;
    if (s < 2u * D) return OMEGA_GDN_KEY_HEADS * D + hk * D + (s - D);
    return 2u * OMEGA_GDN_KEY_HEADS * D + h * D + (s - 2u * D);
}

static int shape_of(uint32_t ratio, OmegaGdnShape *s) { OmegaGdnSpec sp; memset(&sp, 0, sizeof sp); sp.ratio = ratio; return omega_gdn_shape(&sp, s); }

int omega_gdn_pack_params(const float *conv_w, const float *a_log, const float *dt_bias, const float *norm_w, uint8_t *block) {
    return omega_gdn_pack_params_r(0, conv_w, a_log, dt_bias, norm_w, block);
}

int omega_gdn_pack_params_r(uint32_t ratio, const float *conv_w, const float *a_log, const float *dt_bias, const float *norm_w, uint8_t *block) {
    OmegaGdnShape S; const OmegaGdnShape *shp = &S;
    if (shape_of(ratio, &S) || !conv_w || !a_log || !dt_bias || !norm_w || !block) return -1;
    memcpy(block + OMEGA_GDN_PAR_CONV, conv_w, (size_t)shp->conv * OMEGA_GDN_TAPS * 4u);
    float *sc = (float *)(block + shp->par_scal);
    for (uint32_t h = 0; h < shp->vh; h++) { sc[2 * h] = -(float)omega_hostmath_exp((double)a_log[h]); /* libm-free: omegatool links without -lm */ sc[2 * h + 1] = dt_bias[h]; }
    memcpy(block + shp->par_norm, norm_w, OMEGA_GDN_HEAD_DIM * 4u);
    return 0;
}

void omega_gdn_pack_tail(const float *tail, float *dev) { (void)omega_gdn_pack_tail_r(0, tail, dev); }

int omega_gdn_pack_tail_r(uint32_t ratio, const float *tail, float *dev) {
    OmegaGdnShape S; const OmegaGdnShape *shp = &S;
    if (shape_of(ratio, &S)) return -1;
    for (uint32_t h = 0; h < shp->vh; h++)
        for (uint32_t s = 0; s < OMEGA_GDN_SLOTS; s++) {
            float *d = dev + ((size_t)h * OMEGA_GDN_SLOTS + s) * 4u;
            const uint32_t ch = slot_channel(shp, h, s);
            for (uint32_t j = 0; j < 3; j++) d[j] = tail[(size_t)j * shp->conv + ch];
            d[3] = 0.0f;
        }
    return 0;
}

int omega_gdn_unpack_tail(const float *dev, float *tail) { return omega_gdn_unpack_tail_r(0, dev, tail); }

int omega_gdn_unpack_tail_r(uint32_t ratio, const float *dev, float *tail) {
    OmegaGdnShape S; const OmegaGdnShape *shp = &S;
    if (shape_of(ratio, &S)) return -1;
    int rc = 0;
    for (uint32_t h = 0; h < shp->vh; h++)
        for (uint32_t s = 0; s < OMEGA_GDN_SLOTS; s++) {
            const float *d = dev + ((size_t)h * OMEGA_GDN_SLOTS + s) * 4u;
            const uint32_t ch = slot_channel(shp, h, s);
            const bool first = s >= 2u * OMEGA_GDN_HEAD_DIM || h % (shp->vh / OMEGA_GDN_KEY_HEADS) == 0;
            for (uint32_t j = 0; j < 3; j++) {
                float *t = &tail[(size_t)j * shp->conv + ch];
                if (first) *t = d[j];
                else if (memcmp(t, &d[j], 4) != 0) rc = -1;
            }
        }
    return rc;
}

/* ---- buffers ---- */
static int zero_ts_locked(OmegaGpuGdn *g);
static int move_locked(unsigned kind, uint64_t src, uint64_t dst, size_t bytes);
int omega_gpu_gdn_alloc(uint32_t layers, OmegaGpuGdn **out) { return omega_gpu_gdn_alloc_ratio(layers, 0, out); }

int omega_gpu_gdn_alloc_ratio(uint32_t layers, uint32_t ratio, OmegaGpuGdn **out) {
    OmegaGdnShape S;
    if (!out || layers == 0 || layers > OMEGA_GPU_GDN_MAX_LAYERS || shape_of(ratio, &S)) return OMEGA_GPU_GDN_BAD_ARGS;
    /* L313: AIEN_OMEGA_GDN_STATE_BF16=1 stores the state bf16 (half the bytes per conversation; compute stays f32) */
    const char *pb = getenv("AIEN_OMEGA_GDN_STATE_BF16");
    const int bf16 = pb && pb[0] == '1' && pb[1] == 0;
    if (bf16) { OmegaGdnSpec sp; memset(&sp, 0, sizeof sp); sp.ratio = ratio; sp.state_bf16 = 1; if (omega_gdn_shape(&sp, &S)) return OMEGA_GPU_GDN_BAD_ARGS; }
    *out = NULL;
    OmegaGpuGdn *g = omega_gpu_host_calloc(1, sizeof *g);
    if (!g) return OMEGA_GPU_GDN_BAD_ARGS;
    g->bf16 = bf16;
    g->layers = layers;
    g->ratio = ratio;
    g->sh = S;
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_GDN_OK;
    if (!dev_open_locked()) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    if (rc == OMEGA_GPU_GDN_OK && omega_gpu_session_alloc((size_t)layers * g->sh.par_bytes, &g->par) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    /* tail and state GPU-cached unless AIEN_OMEGA_GDN_STATE_CACHED=0 (measured: the decode step 129 -> 43 us per layer,
     * runs/probe-gdn): the step's 64 scalar state stores per thread merge in L2 instead of going to DRAM one by one */
    const char *pc = getenv("AIEN_OMEGA_GDN_STATE_CACHED");
    g->cached = !(pc && pc[0] == '0' && pc[1] == 0);
    int (*const alloc_ts)(size_t, OmegaGpuMem *) = g->cached ? omega_gpu_session_alloc_cached : omega_gpu_session_alloc;
    if (rc == OMEGA_GPU_GDN_OK && alloc_ts((size_t)layers * g->sh.tail_bytes, &g->tail) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    if (rc == OMEGA_GPU_GDN_OK && alloc_ts((size_t)layers * g->sh.state_bytes, &g->state) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    if (rc == OMEGA_GPU_GDN_OK && omega_gpu_session_alloc(LB_BYTES, &g->lb) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    if (rc == OMEGA_GPU_GDN_OK) {
        memset(g->par.cpu, 0, (size_t)layers * g->sh.par_bytes);
        memset(g->lb.cpu, 0, LB_BYTES);
        if (omega_gpu_session_sync_to_gpu(&g->par) != 0 || omega_gpu_session_sync_to_gpu(&g->lb) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    }
    /* tail and state start at zero (cached: zeroed by the chip, since a recycled page may still have lines in L2 that a
     * CPU write would not reach) */
    if (rc == OMEGA_GPU_GDN_OK) rc = zero_ts_locked(g);
    omega_gpu_session_unlock();
    if (rc != OMEGA_GPU_GDN_OK) { omega_gpu_gdn_free(g); return rc; }
    *out = g;
    return OMEGA_GPU_GDN_OK;
}

void omega_gpu_gdn_free(OmegaGpuGdn *g) {
    if (!g) return;
    omega_gpu_session_lock();
    if (g->par.cpu) omega_gpu_session_free(&g->par);
    if (g->tail.cpu) omega_gpu_session_free(&g->tail);
    if (g->state.cpu) omega_gpu_session_free(&g->state);
    if (g->lb.cpu) omega_gpu_session_free(&g->lb);
    if (g->stage.cpu) omega_gpu_session_free(&g->stage);
    omega_gpu_session_scratch_free(&g->pf);
    omega_gpu_session_unlock();
    omega_gpu_host_free(g);
}

int omega_gpu_gdn_set_layer(OmegaGpuGdn *g, uint32_t layer, const float *conv_w, const float *a_log, const float *dt_bias,
                            const float *norm_w) {
    if (!g || layer >= g->layers) return OMEGA_GPU_GDN_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = omega_gdn_pack_params_r(g->ratio, conv_w, a_log, dt_bias, norm_w, (uint8_t *)g->par.cpu + (size_t)layer * g->sh.par_bytes) == 0
                 ? OMEGA_GPU_GDN_OK : OMEGA_GPU_GDN_BAD_ARGS;
    if (rc == OMEGA_GPU_GDN_OK && omega_gpu_session_sync_to_gpu(&g->par) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    omega_gpu_session_unlock();
    return rc;
}

int omega_gpu_gdn_reset(OmegaGpuGdn *g) {
    if (!g) return OMEGA_GPU_GDN_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = dev_open_locked() ? zero_ts_locked(g) : OMEGA_GPU_GDN_CHIP_FAIL;
    omega_gpu_session_unlock();
    return rc;
}

/* L313: one layer's state between the f32 reference layout and the stored format (bf16: round to nearest even, the chip's
 * F2FP.BF16.F32 rule; widen exact). */
static void state_put(const OmegaGpuGdn *g, void *dst, const float *src) {
    if (!g->bf16) { memcpy(dst, src, g->sh.state_bytes); return; }
    uint16_t *d = dst;
    for (size_t i = 0; i < g->sh.state_bytes / 2u; i++) {
        uint32_t b; memcpy(&b, &src[i], 4);
        d[i] = (b & 0x7fffffffu) > 0x7f800000u ? (uint16_t)0x7fc0u : (uint16_t)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
    }
}
static void state_get(const OmegaGpuGdn *g, float *dst, const void *src) {
    if (!g->bf16) { memcpy(dst, src, g->sh.state_bytes); return; }
    const uint16_t *s = src;
    for (size_t i = 0; i < g->sh.state_bytes / 2u; i++) { uint32_t b = (uint32_t)s[i] << 16; memcpy(&dst[i], &b, 4); }
}

/* cached: the staging buffer [state][tail] of one layer (uncached), made on first use */
static int stage_locked(OmegaGpuGdn *g) {
    if (g->stage.cpu) return OMEGA_GPU_GDN_OK;
    return omega_gpu_session_alloc((size_t)g->sh.state_bytes + g->sh.tail_bytes, &g->stage) == 0 ? OMEGA_GPU_GDN_OK : OMEGA_GPU_GDN_CHIP_FAIL;
}

int omega_gpu_gdn_set_state(OmegaGpuGdn *g, uint32_t layer, const float *tail, const float *state) {
    if (!g || layer >= g->layers || !tail || !state) return OMEGA_GPU_GDN_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_GDN_OK;
    const size_t so = (size_t)layer * g->sh.state_bytes, to = (size_t)layer * g->sh.tail_bytes;
    if (!g->cached) {
        omega_gdn_pack_tail_r(g->ratio, tail, (float *)((uint8_t *)g->tail.cpu + to));
        state_put(g, (uint8_t *)g->state.cpu + so, state);
        rc = (omega_gpu_session_sync_to_gpu(&g->tail) == 0 && omega_gpu_session_sync_to_gpu(&g->state) == 0) ? OMEGA_GPU_GDN_OK : OMEGA_GPU_GDN_CHIP_FAIL;
    } else {
        rc = dev_open_locked() ? stage_locked(g) : OMEGA_GPU_GDN_CHIP_FAIL;
        if (rc == OMEGA_GPU_GDN_OK) {
            state_put(g, g->stage.cpu, state);
            omega_gdn_pack_tail_r(g->ratio, tail, (float *)((uint8_t *)g->stage.cpu + g->sh.state_bytes));
            if (omega_gpu_session_sync_to_gpu(&g->stage) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
        }
        if (rc == OMEGA_GPU_GDN_OK) rc = move_locked(KK_COPY, g->stage.va, g->state.va + so, g->sh.state_bytes);
        if (rc == OMEGA_GPU_GDN_OK) rc = move_locked(KK_COPY, g->stage.va + g->sh.state_bytes, g->tail.va + to, g->sh.tail_bytes);
    }
    omega_gpu_session_unlock();
    return rc;
}

int omega_gpu_gdn_get_state(OmegaGpuGdn *g, uint32_t layer, float *tail, float *state) {
    if (!g || layer >= g->layers || (!tail && !state)) return OMEGA_GPU_GDN_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_GDN_OK;
    const size_t so = (size_t)layer * g->sh.state_bytes, to = (size_t)layer * g->sh.tail_bytes;
    const uint8_t *sp, *tp;
    if (!g->cached) {
        rc = (omega_gpu_session_sync_to_cpu(&g->tail) == 0 && omega_gpu_session_sync_to_cpu(&g->state) == 0) ? OMEGA_GPU_GDN_OK : OMEGA_GPU_GDN_CHIP_FAIL;
        sp = (const uint8_t *)g->state.cpu + so; tp = (const uint8_t *)g->tail.cpu + to;
    } else {
        rc = dev_open_locked() ? stage_locked(g) : OMEGA_GPU_GDN_CHIP_FAIL;
        if (rc == OMEGA_GPU_GDN_OK) rc = move_locked(KK_COPY, g->state.va + so, g->stage.va, g->sh.state_bytes);
        if (rc == OMEGA_GPU_GDN_OK) rc = move_locked(KK_COPY, g->tail.va + to, g->stage.va + g->sh.state_bytes, g->sh.tail_bytes);
        if (rc == OMEGA_GPU_GDN_OK && omega_gpu_session_sync_to_cpu(&g->stage) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
        sp = g->stage.cpu; tp = sp + g->sh.state_bytes;
    }
    if (rc == OMEGA_GPU_GDN_OK && tail && omega_gdn_unpack_tail_r(g->ratio, (const float *)tp, tail) != 0) {
        omega_gpu_session_set_error("gdn: the per-head copies of a key head's conv tail disagree");
        rc = OMEGA_GPU_GDN_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_GDN_OK && state) state_get(g, state, sp);
    omega_gpu_session_unlock();
    return rc;
}

/* ---- kernel ---- */
static int kernel_build(unsigned kind, const OmegaGdnSpec *spec, BlackwellIRProgram *prog) {
    switch (kind) {
    case KK_DECODE: return omega_blackwell_codegen_gdn_prog(spec, prog);
    case KK_PREP: return omega_blackwell_codegen_gdn_prep_prog(spec, prog);
    case KK_SCAN: return omega_blackwell_codegen_gdn_scan_prog(spec, prog);
    case KK_POST: return omega_blackwell_codegen_gdn_post_prog(spec, prog);
    case KK_COPY: case KK_ZERO: return omega_blackwell_codegen_gdn_copy_prog(spec, prog);
    case KK_MULTI: return omega_blackwell_codegen_gdn_multi_prog(spec, prog);
    default: return -1;
    }
}
static int kernel_for(unsigned kind, const OmegaGdnSpec *spec, Slot **out, bool *hit) {
    Slot *s = &g_k.slot[kind];
    if (s->valid && memcmp(&s->spec, spec, sizeof *spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_GDN_OK; }
    if (g_rec && s->valid) { omega_gpu_session_set_error("gdn record: kernel cache miss would evict the resident kernel (warm up before recording)"); return OMEGA_GPU_GDN_TOO_LARGE; }
    if (s->valid) omega_gpu_session_free(&s->code);
    memset(s, 0, sizeof *s);
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("gdn codegen: host memory"); return OMEGA_GPU_GDN_CODEGEN_FAIL; }
    int cg = kernel_build(kind, spec, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("gdn codegen refused the spec"); return OMEGA_GPU_GDN_CODEGEN_FAIL; }
    if ((uint64_t)kk_threads[kind] * omega_bw_gpr_declared(prog->regalloc.peak_gpr_usage) > 65536u) { omega_gpu_host_free(prog); omega_gpu_session_set_error("gdn: register file"); return OMEGA_GPU_GDN_TOO_LARGE; }
    const uint32_t insns = (uint32_t)prog->count, expect = omega_gdn_bfly_count(prog);
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_GDN_CHIP_FAIL; }
    OmegaBlackwellKernel kernel;
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    /* SHFL.BFLY placeholders become real butterflies (omega_blackwell_codegen_tc.h); the count must match */
    const int np = cg == 0 ? omega_tc_patch_code(prog, code.cpu, need) : -1;
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need) { omega_gpu_session_free(&code); omega_gpu_session_set_error("gdn encode"); return OMEGA_GPU_GDN_CODEGEN_FAIL; }
    if (np < 0 || (uint32_t)np != expect) { omega_gpu_session_free(&code); omega_gpu_session_set_error("gdn: butterfly patch count mismatch"); return OMEGA_GPU_GDN_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_GDN_CHIP_FAIL; }
    memset(s, 0, sizeof *s);
    s->valid = true; s->spec = *spec; s->kernel = kernel; s->code = code; s->insns = insns;
    *out = s; *hit = false;
    return OMEGA_GPU_GDN_OK;
}


/* One copy (KK_COPY: dst = src) or zero (KK_ZERO: dst = 0, src unused) launch over `bytes` (a multiple of 4096), submitted
 * and waited for. Caller holds the session lock with the device open. */
static int move_locked(unsigned kind, uint64_t src, uint64_t dst, size_t bytes) {
    const size_t per = (size_t)OMEGA_GDN_COPY_WORDS_PER_CTA * 4u;
    if ((kind != KK_COPY && kind != KK_ZERO) || !dst || bytes == 0 || bytes % per || bytes / per > 0x7fffffffu) return OMEGA_GPU_GDN_BAD_ARGS;
    OmegaGdnSpec spec; memset(&spec, 0, sizeof spec);
    spec.copy = kind == KK_COPY ? OMEGA_GDN_COPY : OMEGA_GDN_ZERO;
    Slot *ks = NULL; bool hit = false;
    int rc = kernel_for(kind, &spec, &ks, &hit);
    if (rc != OMEGA_GPU_GDN_OK) return rc;
    uint32_t args[OMEGA_GDN_COPY_ARGS_WORDS] = { (uint32_t)src, (uint32_t)(src >> 32), (uint32_t)dst, (uint32_t)(dst >> 32) };
    OmegaGpuLaunch L = {
        .code_va = ks->code.va, .gpr_count = omega_bw_gpr_declared(ks->kernel.gpr_count),
        .threads_x = OMEGA_GDN_COPY_THREADS, .threads_y = 1, .grid_x = (uint32_t)(bytes / per), .grid_y = 1,
        .num_elements = bytes / 4u, .args = args, .n_args = OMEGA_GDN_COPY_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
    };
    if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); return OMEGA_GPU_GDN_CHIP_FAIL; }
    return omega_gpu_session_launch(&L, NULL, NULL) == 0 ? OMEGA_GPU_GDN_OK : OMEGA_GPU_GDN_CHIP_FAIL;
}

/* Every layer's tail and state to zero: cached on the chip, uncached by the host. Caller holds the lock, device open. */
static int zero_ts_locked(OmegaGpuGdn *g) {
    const size_t tb = (size_t)g->layers * g->sh.tail_bytes, sb = (size_t)g->layers * g->sh.state_bytes;
    if (!g->cached) {
        memset(g->tail.cpu, 0, tb); memset(g->state.cpu, 0, sb);
        return (omega_gpu_session_sync_to_gpu(&g->tail) == 0 && omega_gpu_session_sync_to_gpu(&g->state) == 0) ? OMEGA_GPU_GDN_OK : OMEGA_GPU_GDN_CHIP_FAIL;
    }
    int rc = move_locked(KK_ZERO, 0, g->tail.va, tb);
    return rc == OMEGA_GPU_GDN_OK ? move_locked(KK_ZERO, 0, g->state.va, sb) : rc;
}
/* rec non-NULL (record path, caller holds the session lock): the inputs are the device rows dev_*, the launch is
 * appended to rec instead of submitted, nothing is staged, poisoned or read back. */
static int gdn_core(OmegaGpuGdn *g, uint32_t layer, const float *qkv, const float *z, const float *ab, float eps_l2,
                    float eps_norm, float in_scale, float out_scale, float *out, OmegaGpuGdnInfo *info,
                    OmegaGpuCmdList *rec, uint64_t dev_qkv, uint64_t dev_z, uint64_t dev_ab) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    if (!g || layer >= g->layers || (rec ? (!dev_qkv || !dev_z || !dev_ab) : (!qkv || !z || !ab || !out))) return OMEGA_GPU_GDN_BAD_ARGS;
    if (!isfinite(eps_l2) || !isfinite(eps_norm) || !isfinite(in_scale) || !isfinite(out_scale)) return OMEGA_GPU_GDN_BAD_ARGS;
    OmegaGdnSpec spec; memset(&spec, 0, sizeof spec);
    spec.mutant = g_k.mutant;
    spec.ratio = g->ratio;
    spec.state_bf16 = (uint32_t)g->bf16;

    if (!rec) omega_gpu_session_lock();
#define UNLOCK() do { if (!rec) omega_gpu_session_unlock(); } while (0)
    int rc = OMEGA_GPU_GDN_OK;
    if (!dev_open_locked()) { UNLOCK(); return OMEGA_GPU_GDN_CHIP_FAIL; }
    Slot *ks = NULL; bool hit = false;
    g_rec = rec != NULL;
    rc = kernel_for(KK_DECODE, &spec, &ks, &hit);
    g_rec = 0;
    if (rc != OMEGA_GPU_GDN_OK) { UNLOCK(); return rc; }
    uint8_t *b = g->lb.cpu;
    float *yo = (float *)(b + LB_OUT);
    if (!rec) {
        memcpy(b + LB_QKV, qkv, g->sh.conv * 4u);
        memcpy(b + LB_Z, z, g->sh.vdim * 4u);
        memcpy(b + LB_AB, ab, g->sh.vh * 8u);
        for (size_t i = 0; i < g->sh.vdim; i++) yo[i] = POISON_F32;
        if (omega_gpu_session_sync_to_gpu(&g->lb) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    }

    uint32_t args[OMEGA_GDN_ARGS_WORDS];
#define A64(i, v) do { args[i] = (uint32_t)(v); args[(i) + 1] = (uint32_t)((v) >> 32); } while (0)
    A64(0, rec ? dev_qkv : g->lb.va + LB_QKV); A64(2, rec ? dev_z : g->lb.va + LB_Z); A64(4, rec ? dev_ab : g->lb.va + LB_AB);
    A64(6, g->par.va + (uint64_t)layer * g->sh.par_bytes); A64(8, g->tail.va + (uint64_t)layer * g->sh.tail_bytes);
    A64(10, g->state.va + (uint64_t)layer * g->sh.state_bytes); A64(12, g->lb.va + LB_OUT);
#undef A64
    memcpy(&args[14], &eps_l2, 4); memcpy(&args[15], &eps_norm, 4); memcpy(&args[16], &in_scale, 4); memcpy(&args[17], &out_scale, 4);
    if (info) {
        info->layer = layer; info->threads = OMEGA_GDN_THREADS; info->grid_x = g->sh.r; info->grid_y = OMEGA_GDN_GRID_Y;
        info->insns = ks->insns; info->gprs = ks->kernel.gpr_count;
        info->state_bytes = 2ull * g->sh.state_bytes;
        info->kernel_cache_hit = hit;
    }
    if (rc == OMEGA_GPU_GDN_OK) {
        OmegaGpuLaunch L = {
            .code_va = ks->code.va, .gpr_count = omega_bw_gpr_declared(ks->kernel.gpr_count),
            .threads_x = OMEGA_GDN_THREADS, .threads_y = 1, .grid_x = g->sh.r, .grid_y = OMEGA_GDN_GRID_Y,
            .num_elements = (size_t)g->sh.vdim,
            .args = args, .n_args = OMEGA_GDN_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
            .shared_bytes = OMEGA_GDN_SHARED_BYTES,
        };
        if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); rc = OMEGA_GPU_GDN_CHIP_FAIL; }
        else if (rec) { if (omega_gpu_cmdlist_add(rec, &L) < 0) rc = OMEGA_GPU_GDN_CHIP_FAIL; }
        else if (omega_gpu_session_launch(&L, info ? &info->elapsed_ns : NULL, info ? &info->completion_marker : NULL) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_GDN_OK && !rec && omega_gpu_session_sync_to_cpu(&g->lb) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    if (rc == OMEGA_GPU_GDN_OK && !rec) {
        size_t unchanged = 0;
        for (size_t i = 0; i < g->sh.vdim; i++) if (yo[i] == POISON_F32) unchanged++;
        if (unchanged) { omega_gpu_session_set_error("gdn: output poison survived"); rc = OMEGA_GPU_GDN_CHIP_FAIL; }
        else memcpy(out, yo, g->sh.vdim * 4u);
    }
    UNLOCK();
#undef UNLOCK
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}

int omega_gpu_gdn_step(OmegaGpuGdn *g, uint32_t layer, const float *qkv, const float *z, const float *ab, float eps_l2,
                       float eps_norm, float in_scale, float out_scale, float *out, OmegaGpuGdnInfo *info) {
    return gdn_core(g, layer, qkv, z, ab, eps_l2, eps_norm, in_scale, out_scale, out, info, NULL, 0, 0, 0);
}

/* Record path (decode graph): see the header. Caller holds omega_gpu_session_lock(). */
int omega_gpu_gdn_record(OmegaGpuCmdList *cl, OmegaGpuGdn *g, uint32_t layer, uint64_t qkv_va, uint64_t z_va, uint64_t ab_va,
                         float eps_l2, float eps_norm, float in_scale, float out_scale, uint64_t *out_va) {
    if (!cl) return OMEGA_GPU_GDN_BAD_ARGS;
    int rc = gdn_core(g, layer, NULL, NULL, NULL, eps_l2, eps_norm, in_scale, out_scale, NULL, NULL, cl, qkv_va, z_va, ab_va);
    if (rc == OMEGA_GPU_GDN_OK && out_va) *out_va = g->lb.va + LB_OUT;
    return rc;
}

/* ---- multi-token decode (L369): see the header ---- */
int omega_gpu_gdn_multi_prepare(OmegaGpuGdn *g) {
    if (!g) return OMEGA_GPU_GDN_BAD_ARGS;
    OmegaGdnSpec spec; memset(&spec, 0, sizeof spec);
    spec.mutant = g_k.mutant; spec.ratio = g->ratio;
    omega_gpu_session_lock();
    int rc = dev_open_locked() ? OMEGA_GPU_GDN_OK : OMEGA_GPU_GDN_CHIP_FAIL;
    Slot *ks = NULL; bool hit = false;
    if (rc == OMEGA_GPU_GDN_OK) rc = kernel_for(KK_MULTI, &spec, &ks, &hit);
    omega_gpu_session_unlock();
    return rc;
}

int omega_gpu_gdn_multi_record(OmegaGpuCmdList *cl, OmegaGpuGdn *g, uint32_t layer, uint32_t tokens, uint64_t qkv_va, uint64_t z_va,
                               uint32_t xstride, uint64_t ab_va, uint32_t abstride, uint64_t out_va, uint32_t ostride, float eps_l2,
                               float eps_norm, float in_scale, float out_scale) {
    if (!cl || !g || layer >= g->layers || tokens == 0 || !qkv_va || !z_va || !ab_va || !out_va) return OMEGA_GPU_GDN_BAD_ARGS;
    if (!isfinite(eps_l2) || !isfinite(eps_norm) || !isfinite(in_scale) || !isfinite(out_scale)) return OMEGA_GPU_GDN_BAD_ARGS;
    if (((qkv_va | z_va | out_va | xstride | ostride) & 15u) || ((ab_va | abstride) & 7u)) return OMEGA_GPU_GDN_BAD_ARGS;
    OmegaGdnSpec spec; memset(&spec, 0, sizeof spec);
    spec.mutant = g_k.mutant; spec.ratio = g->ratio;
    /* caller holds the session lock */
    if (!dev_open_locked()) return OMEGA_GPU_GDN_CHIP_FAIL;
    Slot *ks = NULL; bool hit = false;
    g_rec = 1;
    int rc = kernel_for(KK_MULTI, &spec, &ks, &hit);
    g_rec = 0;
    if (rc != OMEGA_GPU_GDN_OK) return rc;
    uint32_t args[OMEGA_GDN_MULTI_ARGS_WORDS];
#define A64(i, v) do { args[i] = (uint32_t)(v); args[(i) + 1] = (uint32_t)((v) >> 32); } while (0)
    A64(0, qkv_va); A64(2, z_va); A64(4, ab_va);
    A64(6, g->par.va + (uint64_t)layer * g->sh.par_bytes); A64(8, g->tail.va + (uint64_t)layer * g->sh.tail_bytes);
    A64(10, g->state.va + (uint64_t)layer * g->sh.state_bytes); A64(12, out_va);
#undef A64
    memcpy(&args[14], &eps_l2, 4); memcpy(&args[15], &eps_norm, 4); memcpy(&args[16], &in_scale, 4); memcpy(&args[17], &out_scale, 4);
    args[18] = tokens; args[19] = xstride; args[20] = abstride; args[21] = ostride;
    OmegaGpuLaunch L = {
        .code_va = ks->code.va, .gpr_count = omega_bw_gpr_declared(ks->kernel.gpr_count),
        .threads_x = OMEGA_GDN_THREADS, .threads_y = 1, .grid_x = g->sh.r, .grid_y = OMEGA_GDN_GRID_Y,
        .num_elements = (size_t)g->sh.vdim,
        .args = args, .n_args = OMEGA_GDN_MULTI_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
        .shared_bytes = OMEGA_GDN_SHARED_BYTES,
    };
    if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); return OMEGA_GPU_GDN_CHIP_FAIL; }
    return omega_gpu_cmdlist_add(cl, &L) < 0 ? OMEGA_GPU_GDN_CHIP_FAIL : OMEGA_GPU_GDN_OK;
}

int omega_gpu_gdn_read_out(OmegaGpuGdn *g, float *out) {
    if (!g || !out) return OMEGA_GPU_GDN_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = omega_gpu_session_sync_to_cpu(&g->lb) == 0 ? OMEGA_GPU_GDN_OK : OMEGA_GPU_GDN_CHIP_FAIL;
    if (rc == OMEGA_GPU_GDN_OK) memcpy(out, (const uint8_t *)g->lb.cpu + LB_OUT, g->sh.vdim * 4u);
    omega_gpu_session_unlock();
    return rc;
}

/* ---- prefill (omega#543) ---- */
/* scratch sections for T tokens (bytes, 256-aligned): qkv [T][10240] | z [T][6144] | ab [T][48][2] | out [T][6144] |
 * o [T][6144] | records [T rounded up to 8][48] */
enum { PF_QKV, PF_Z, PF_AB, PF_OUT, PF_O, PF_REC, PF_N };
static size_t pf_layout(const OmegaGdnShape *shp, uint32_t T, size_t off[PF_N]) {
    const uint64_t T8 = ((uint64_t)T + OMEGA_GDN_SCAN_CHUNK - 1u) / OMEGA_GDN_SCAN_CHUNK * OMEGA_GDN_SCAN_CHUNK;
    const uint64_t sz[PF_N] = { (uint64_t)T * shp->conv * 4u, (uint64_t)T * shp->vdim * 4u, (uint64_t)T * shp->vh * 8u,
                                (uint64_t)T * shp->vdim * 4u, (uint64_t)T * shp->vdim * 4u, T8 * shp->vh * OMEGA_GDN_REC_BYTES };
    size_t at = 0;
    for (unsigned i = 0; i < PF_N; i++) { off[i] = at; at += (size_t)((sz[i] + 255u) & ~(uint64_t)255u); }
    return at;
}

int omega_gpu_gdn_prefill_reserve(OmegaGpuGdn *g, uint32_t max_tokens) {
    if (!g || max_tokens < OMEGA_GDN_PF_MIN_T || max_tokens > OMEGA_GPU_GDN_PF_MAX_T) return OMEGA_GPU_GDN_BAD_ARGS;
    size_t off[PF_N];
    const size_t bytes = pf_layout(&g->sh, max_tokens, off);
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_GDN_OK;
    if (!dev_open_locked()) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    else {
        omega_gpu_session_scratch_unreserve(&g->pf);
        if (omega_gpu_session_scratch_reserve(&g->pf, bytes) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    }
    omega_gpu_session_unlock();
    return rc;
}

/* rec non-NULL (record path, caller holds the lock): the inputs are device rows, the scratch must already hold T
 * tokens (omega_gpu_gdn_prefill_reserve), the three launches are appended to rec. */
static int pf_core(OmegaGpuGdn *g, uint32_t layer, uint32_t T, const float *qkv, const float *z, const float *ab, float eps_l2,
                   float eps_norm, float in_scale, float out_scale, float *out, OmegaGpuGdnPfInfo *info,
                   OmegaGpuCmdList *rec, uint64_t dev_qkv, uint64_t dev_z, uint64_t dev_ab, uint64_t *out_va) {
    uint64_t c0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    if (!g || layer >= g->layers || T < OMEGA_GDN_PF_MIN_T || T > OMEGA_GPU_GDN_PF_MAX_T) return OMEGA_GPU_GDN_BAD_ARGS;
    if (rec ? (!dev_qkv || !dev_z || !dev_ab) : (!qkv || !z || !ab || !out)) return OMEGA_GPU_GDN_BAD_ARGS;
    if (!isfinite(eps_l2) || !isfinite(eps_norm) || !isfinite(in_scale) || !isfinite(out_scale)) return OMEGA_GPU_GDN_BAD_ARGS;
    OmegaGdnSpec spec; memset(&spec, 0, sizeof spec);
    spec.mutant = g_k.mutant;
    spec.ratio = g->ratio;
    spec.state_bf16 = (uint32_t)g->bf16;
    size_t off[PF_N];
    const size_t bytes = pf_layout(&g->sh, T, off);

    if (!rec) omega_gpu_session_lock();
#define UNLOCK() do { if (!rec) omega_gpu_session_unlock(); } while (0)
    if (!dev_open_locked()) { UNLOCK(); return OMEGA_GPU_GDN_CHIP_FAIL; }
    if (rec && (!g->pf.mem.cpu || g->pf.cap < bytes)) {
        omega_gpu_session_set_error("gdn prefill record: scratch below T tokens (call omega_gpu_gdn_prefill_reserve first)");
        UNLOCK(); return OMEGA_GPU_GDN_TOO_LARGE;
    }
    if (!rec && omega_gpu_session_scratch(&g->pf, bytes) != 0) { UNLOCK(); return OMEGA_GPU_GDN_TOO_LARGE; }
    Slot *ks[3] = { NULL, NULL, NULL };
    int rc = OMEGA_GPU_GDN_OK, hits = 0;
    g_rec = rec != NULL;
    for (unsigned k = 0; k < 3 && rc == OMEGA_GPU_GDN_OK; k++) { bool hit = false; rc = kernel_for(KK_PREP + k, &spec, &ks[k], &hit); hits += hit; }
    g_rec = 0;
    if (rc != OMEGA_GPU_GDN_OK) { UNLOCK(); return rc; }
    uint8_t *b = g->pf.mem.cpu;
    const uint64_t va = g->pf.mem.va;
    float *yo = (float *)(b + off[PF_OUT]);
    const size_t nout = (size_t)T * g->sh.vdim;
    if (!rec) {
        memcpy(b + off[PF_QKV], qkv, (size_t)T * g->sh.conv * 4u);
        memcpy(b + off[PF_Z], z, nout * 4u);
        memcpy(b + off[PF_AB], ab, (size_t)T * g->sh.vh * 8u);
        for (size_t i = 0; i < nout; i++) yo[i] = POISON_F32;
        if (omega_gpu_session_sync_to_gpu(&g->pf.mem) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    }
    const uint64_t a_qkv = rec ? dev_qkv : va + off[PF_QKV], a_z = rec ? dev_z : va + off[PF_Z], a_ab = rec ? dev_ab : va + off[PF_AB];
    const uint64_t a_par = g->par.va + (uint64_t)layer * g->sh.par_bytes, a_tail = g->tail.va + (uint64_t)layer * g->sh.tail_bytes;
    const uint64_t a_state = g->state.va + (uint64_t)layer * g->sh.state_bytes, a_rec = va + off[PF_REC], a_o = va + off[PF_O];
    const uint64_t a_out = va + off[PF_OUT];
    uint32_t ap[OMEGA_GDN_PREP_ARGS_WORDS], as[OMEGA_GDN_SCAN_ARGS_WORDS], aq[OMEGA_GDN_POST_ARGS_WORDS];
#define A64(arr, i, v) do { (arr)[i] = (uint32_t)(v); (arr)[(i) + 1] = (uint32_t)((v) >> 32); } while (0)
    A64(ap, 0, a_qkv); A64(ap, 2, a_ab); A64(ap, 4, a_par); A64(ap, 6, a_tail); A64(ap, 8, a_rec);
    ap[10] = T; memcpy(&ap[11], &eps_l2, 4); memcpy(&ap[12], &in_scale, 4);
    A64(as, 0, a_rec); A64(as, 2, a_qkv); A64(as, 4, a_tail); A64(as, 6, a_state); A64(as, 8, a_o);
    as[10] = T; memcpy(&as[11], &in_scale, 4);
    A64(aq, 0, a_o); A64(aq, 2, a_z); A64(aq, 4, a_par); A64(aq, 6, a_out);
    memcpy(&aq[8], &eps_norm, 4); memcpy(&aq[9], &in_scale, 4); memcpy(&aq[10], &out_scale, 4);
#undef A64
    const uint32_t thr[3] = { OMEGA_GDN_PREP_THREADS, OMEGA_GDN_THREADS, OMEGA_GDN_THREADS };
    const uint32_t gx[3] = { (T + OMEGA_GDN_PREP_TOKENS - 1u) / OMEGA_GDN_PREP_TOKENS, g->sh.r, T };
    const uint32_t gy[3] = { OMEGA_GDN_KEY_HEADS, OMEGA_GDN_GRID_Y, g->sh.post_grid_y };
    const uint32_t sh[3] = { OMEGA_GDN_PREP_SHARED_BYTES, OMEGA_GDN_SCAN_SHARED_BYTES, 0 };
    const uint32_t *args[3] = { ap, as, aq };
    const uint32_t nargs[3] = { OMEGA_GDN_PREP_ARGS_WORDS, OMEGA_GDN_SCAN_ARGS_WORDS, OMEGA_GDN_POST_ARGS_WORDS };
    if (info) {
        info->layer = layer; info->tokens = T; info->kernel_cache_hits = (uint32_t)hits;
        for (unsigned k = 0; k < 3; k++) { info->insns[k] = ks[k]->insns; info->gprs[k] = ks[k]->kernel.gpr_count; }
    }
    for (unsigned k = 0; k < 3 && rc == OMEGA_GPU_GDN_OK; k++) {
        OmegaGpuLaunch L = {
            .code_va = ks[k]->code.va, .gpr_count = omega_bw_gpr_declared(ks[k]->kernel.gpr_count),
            .threads_x = thr[k], .threads_y = 1, .grid_x = gx[k], .grid_y = gy[k],
            .num_elements = nout,
            .args = args[k], .n_args = nargs[k], .timeout_ms = MARKER_TIMEOUT_MS,
            .shared_bytes = sh[k],
        };
        if (!omega_bw_gpr_declared_ok(L.gpr_count, ks[k]->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); rc = OMEGA_GPU_GDN_CHIP_FAIL; }
        else if (rec) { if (omega_gpu_cmdlist_add(rec, &L) < 0) rc = OMEGA_GPU_GDN_CHIP_FAIL; }
        else if (omega_gpu_session_launch(&L, info ? &info->elapsed_ns[k] : NULL, NULL) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_GDN_OK && !rec && omega_gpu_session_sync_to_cpu(&g->pf.mem) != 0) rc = OMEGA_GPU_GDN_CHIP_FAIL;
    if (rc == OMEGA_GPU_GDN_OK && !rec) {
        size_t unchanged = 0;
        for (size_t i = 0; i < nout; i++) if (yo[i] == POISON_F32) unchanged++;
        if (unchanged) { omega_gpu_session_set_error("gdn prefill: output poison survived"); rc = OMEGA_GPU_GDN_CHIP_FAIL; }
        else memcpy(out, yo, nout * 4u);
    }
    if (rc == OMEGA_GPU_GDN_OK && out_va) *out_va = a_out;
    UNLOCK();
#undef UNLOCK
    if (info) info->call_ns = omega_gpu_session_now_ns() - c0;
    return rc;
}

int omega_gpu_gdn_prefill(OmegaGpuGdn *g, uint32_t layer, uint32_t tokens, const float *qkv, const float *z, const float *ab,
                          float eps_l2, float eps_norm, float in_scale, float out_scale, float *out, OmegaGpuGdnPfInfo *info) {
    return pf_core(g, layer, tokens, qkv, z, ab, eps_l2, eps_norm, in_scale, out_scale, out, info, NULL, 0, 0, 0, NULL);
}

int omega_gpu_gdn_prefill_record(OmegaGpuCmdList *cl, OmegaGpuGdn *g, uint32_t layer, uint32_t tokens, uint64_t qkv_va,
                                 uint64_t z_va, uint64_t ab_va, float eps_l2, float eps_norm, float in_scale, float out_scale,
                                 uint64_t *out_va) {
    if (!cl) return OMEGA_GPU_GDN_BAD_ARGS;
    return pf_core(g, layer, tokens, NULL, NULL, NULL, eps_l2, eps_norm, in_scale, out_scale, NULL, NULL, cl, qkv_va, z_va, ab_va, out_va);
}

int omega_gpu_gdn_prefill_read_out(OmegaGpuGdn *g, uint32_t tokens, float *out) {
    if (!g || !out || tokens < OMEGA_GDN_PF_MIN_T || tokens > OMEGA_GPU_GDN_PF_MAX_T) return OMEGA_GPU_GDN_BAD_ARGS;
    size_t off[PF_N];
    const size_t bytes = pf_layout(&g->sh, tokens, off);
    omega_gpu_session_lock();
    int rc = (!g->pf.mem.cpu || g->pf.cap < bytes) ? OMEGA_GPU_GDN_BAD_ARGS
           : omega_gpu_session_sync_to_cpu(&g->pf.mem) == 0 ? OMEGA_GPU_GDN_OK : OMEGA_GPU_GDN_CHIP_FAIL;
    if (rc == OMEGA_GPU_GDN_OK) memcpy(out, (const uint8_t *)g->pf.mem.cpu + off[PF_OUT], (size_t)tokens * g->sh.vdim * 4u);
    omega_gpu_session_unlock();
    return rc;
}
