/*
 * omega_gpu_prefill_attn_api.c: chunked launches of the tensor-core causal prefill attention (see the header).
 * Session use mirrors omega_gpu_decode_attn_api.c: buffers from omega_gpu_session_alloc (GPU-uncached, host-visible),
 * sync_to_gpu after host writes, sync_to_cpu before host reads, one omega_gpu_session_launch per chunk.
 * The kernel uses the SHFL.BFLY / F2FP placeholders of omega_blackwell_codegen_tc.h: the code is patched after encoding
 * and the patch count must equal the program's placeholder count.
 */
#include "omega_gpu_prefill_attn_api.h"
#include "omega_blackwell_codegen_tc.h"
#include "omega_gpu_session.h"
#include "omega_gpu_fmt.h"
#include <string.h>
#if defined(__has_include)
#if __has_include("omega_gpu_prefill.h")
#include "omega_gpu_prefill.h"            /* GPU-cached device buffers + copy kernel (integration builds) */
#include "omega_blackwell_codegen_pf.h"
#define PATTN_HAVE_PF 1
#endif
#endif
#ifndef PATTN_HAVE_PF
#define PATTN_HAVE_PF 0
#endif

#define MARKER_TIMEOUT_MS 20000ull
#define POISON_F32 (-999.0f)

typedef struct {
    bool valid;
    OmegaPrefillAttnSpec spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
} Slot;

#define MAX_LAUNCH_LOG 256u

static struct {
    bool hooked;
    Slot slot[3];                 /* one per OMEGA_PATTN_MODE_* */
    uint32_t warps;
    int mutant;
    int fused;                    /* 1: 64-token fused chunks (cut 1 path); 0 (default): APPEND + ATTEND launch pairs */
    bool have_bufs;
    uint32_t buf_rows;            /* host-entry staging capacity in tokens */
    OmegaGpuMem qkv, out;
    uint64_t launch_ns[MAX_LAUNCH_LOG];
    uint32_t n_launch;            /* launches of the last call (log keeps the first MAX_LAUNCH_LOG) */
    int mirror_off;               /* 1: ATTEND reads the uncached cache (cut 3); 0 (default with PATTN_HAVE_PF): GPU-cached K/V mirror */
    int mirror_per_call;          /* TEST: 1 forgets the mirrored rows before every call (cut 4 behaviour: history copied every time) */
    struct Mir { uint64_t key; uint32_t max_pos, valid; uint64_t plane; OmegaGpuMem mem; } mir[OMEGA_GPU_PATTN_MAX_MIRRORS];
    uint32_t n_mir;               /* persistent mirrors, one per (cache, layer), keyed by the layer's K plane VA */
} g;

static void slot_clear_locked(void) {
    for (unsigned i = 0; i < 3; i++) { if (g.slot[i].valid) omega_gpu_session_free(&g.slot[i].code); memset(&g.slot[i], 0, sizeof g.slot[i]); }
}
static void on_session_close(void) { memset(g.mir, 0, sizeof g.mir); g.n_mir = 0; memset(g.slot, 0, sizeof g.slot); g.have_bufs = false; g.buf_rows = 0; memset(&g.qkv, 0, sizeof g.qkv); memset(&g.out, 0, sizeof g.out); }
static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}

void omega_gpu_prefill_attn_set_warps(uint32_t warps) { omega_gpu_session_lock(); g.warps = warps; slot_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_prefill_attn_test_set_mutant(int mutant) { omega_gpu_session_lock(); g.mutant = mutant; slot_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_prefill_attn_set_fused(int fused) { omega_gpu_session_lock(); g.fused = fused ? 1 : 0; omega_gpu_session_unlock(); }
void omega_gpu_prefill_attn_set_kv_mirror(int on) { omega_gpu_session_lock(); g.mirror_off = on ? 0 : 1; omega_gpu_session_unlock(); }
int omega_gpu_prefill_attn_kv_mirror_available(void) { return PATTN_HAVE_PF; }
uint32_t omega_gpu_prefill_attn_test_launch_ns(uint64_t *ns, uint32_t cap) {
    omega_gpu_session_lock();
    uint32_t n = g.n_launch < MAX_LAUNCH_LOG ? g.n_launch : MAX_LAUNCH_LOG;
    for (uint32_t i = 0; i < n && i < cap; i++) ns[i] = g.launch_ns[i];
    uint32_t total = g.n_launch;
    omega_gpu_session_unlock();
    return total;
}
void omega_gpu_prefill_attn_release(void) {
    omega_gpu_session_lock();
    slot_clear_locked();
    if (g.have_bufs) { omega_gpu_session_free(&g.qkv); omega_gpu_session_free(&g.out); g.have_bufs = false; g.buf_rows = 0; }
    omega_gpu_session_unlock();
#if PATTN_HAVE_PF
    for (uint32_t i = 0; i < g.n_mir; i++) omega_gpu_pf_free(&g.mir[i].mem);
    memset(g.mir, 0, sizeof g.mir); g.n_mir = 0;
#endif
}

#if PATTN_HAVE_PF
/* Persistent mirror of one (cache, layer): the decode cache's exact layout ([K plane | V plane], plane = v_va - k_va,
 * row (kv_head, pos) at plane + (kv_head * max_pos + pos) * 256) in GPU-cached device memory. Rows 0 .. valid-1 equal
 * the cache. Zero-filled when allocated, so rows past the last key of a 32-key block are finite (their P is exactly 0,
 * and 0 * NaN would poison O). Lock NOT held (the pf calls take it). */
static struct Mir *mirror_get(const OmegaGpuDecodeAttnCacheView *v) {
    for (uint32_t i = 0; i < g.n_mir; i++)
        if (g.mir[i].key == v->k_va && g.mir[i].max_pos == v->max_pos) return &g.mir[i];
    if (g.n_mir >= OMEGA_GPU_PATTN_MAX_MIRRORS) { omega_gpu_session_set_error("prefill attn: too many KV mirrors"); return NULL; }
    struct Mir *m = &g.mir[g.n_mir];
    memset(m, 0, sizeof *m);
    const uint64_t plane = v->v_va - v->k_va, bytes = 2u * plane;
    if (v->v_va <= v->k_va || (plane & 15u)) { omega_gpu_session_set_error("prefill attn: cache planes"); return NULL; }
    if (omega_gpu_pf_alloc_dev(bytes, &m->mem) != OMEGA_GPU_PF_OK) { memset(m, 0, sizeof *m); return NULL; }
    const size_t step = (size_t)16u << 20;                 /* zero-fill in 16 MB uploads */
    void *z = omega_gpu_host_calloc(1, step);
    int ok = z != NULL;
    for (uint64_t off = 0; ok && off < bytes; off += step) {
        const size_t part = bytes - off < step ? (size_t)(bytes - off) : step;
        ok = omega_gpu_pf_upload(&m->mem, off, z, part) == OMEGA_GPU_PF_OK;
    }
    omega_gpu_host_free(z);
    if (!ok) { omega_gpu_pf_free(&m->mem); memset(m, 0, sizeof *m); omega_gpu_session_set_error("prefill attn: KV mirror zero-fill"); return NULL; }
    m->key = v->k_va; m->max_pos = v->max_pos; m->plane = plane; m->valid = 0;
    g.n_mir++;
    return m;
}
static struct Mir *mirror_find(const OmegaGpuDecodeAttnCacheView *v) {
    for (uint32_t i = 0; i < g.n_mir; i++) if (g.mir[i].key == v->k_va && g.mir[i].max_pos == v->max_pos) return &g.mir[i];
    return NULL;
}
/* lock held: the copy kernel of the prefill module (same key and generator as omega_gpu_pf_copy_va) */
static int gen_copy(const OmegaPfKey *key, void *prog) { (void)key; return omega_blackwell_codegen_pf_copy_prog(prog); }
static int copy_locked(uint64_t dst, uint64_t src, size_t bytes, uint64_t *ns) {
    if ((bytes & 15u) || (dst & 15u) || (src & 15u)) return OMEGA_GPU_PF_BAD_ARGS;
    OmegaPfKey key; memset(&key, 0, sizeof key); key.kind = 1u; /* KIND_COPY of omega_gpu_prefill.c */
    OmegaPfKernel k; int rc = omega_gpu_pf_open_locked();
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_kernel_locked(&key, gen_copy, &k);
    if (rc != OMEGA_GPU_PF_OK || bytes == 0) return rc;
    const uint32_t units = (uint32_t)(bytes / 16u), grid = (units + OMEGA_PF_COPY_THREADS - 1u) / OMEGA_PF_COPY_THREADS;
    uint32_t args[5] = { (uint32_t)dst, (uint32_t)(dst >> 32), (uint32_t)src, (uint32_t)(src >> 32), units };
    return omega_gpu_pf_launch_locked(&k, OMEGA_PF_COPY_THREADS, grid, 1, 0, args, 5, ns);
}
#endif

size_t omega_gpu_prefill_attn_encode(const OmegaPrefillAttnSpec *spec, uint8_t *code, size_t cap, uint32_t *gprs, int *patched) {
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) return 0;
    size_t need = 0;
    if (omega_blackwell_codegen_prefill_attn_prog(spec, prog) == 0) need = omega_bw_program_code_bytes(prog);
    OmegaBlackwellKernel k;
    if (need == 0 || need > cap || omega_blackwell_kernel_encode_into(prog, code, cap, &k) != 0) { omega_gpu_host_free(prog); return 0; }
    int np = omega_tc_patch_code(prog, code, need);
    if (gprs) *gprs = k.gpr_count;
    if (patched) *patched = np;
    omega_gpu_host_free(prog);
    return np < 0 ? 0 : need;
}

static int kernel_for(const OmegaPrefillAttnSpec *spec, Slot **out, bool *hit) {
    Slot *s = &g.slot[spec->mode];
    if (s->valid && memcmp(&s->spec, spec, sizeof *spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_DATTN_OK; }
    if (s->valid) omega_gpu_session_free(&s->code);
    memset(s, 0, sizeof *s);
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("prefill attn codegen: host memory"); return OMEGA_GPU_DATTN_CODEGEN_FAIL; }
    int cg = omega_blackwell_codegen_prefill_attn_prog(spec, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("prefill attn codegen refused the spec"); return OMEGA_GPU_DATTN_CODEGEN_FAIL; }
    if ((uint64_t)32u * spec->warps * omega_bw_gpr_declared(prog->regalloc.peak_gpr_usage) > 65536u) { omega_gpu_host_free(prog); omega_gpu_session_set_error("prefill attn: register file"); return OMEGA_GPU_DATTN_TOO_LARGE; }
    uint32_t expect = 0;
    for (size_t i = 0; i < prog->count; i++) {
        const BlackwellIRInsn *in = &prog->insns[i];
        if ((in->op == BW_IR_SHFL_DOWN && (in->imm & OMEGA_TC_MARK_BFLY)) || (in->op == BW_IR_FMNMX_MAX && in->imm == OMEGA_TC_MARK_F2FP)) expect++;
    }
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    OmegaBlackwellKernel kernel;
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    int np = cg == 0 ? omega_tc_patch_code(prog, code.cpu, need) : -1;
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need) { omega_gpu_session_free(&code); omega_gpu_session_set_error("prefill attn encode"); return OMEGA_GPU_DATTN_CODEGEN_FAIL; }
    if (np < 0 || (uint32_t)np != expect) { omega_gpu_session_free(&code); omega_gpu_session_set_error("prefill attn: placeholder patch count mismatch"); return OMEGA_GPU_DATTN_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    memset(s, 0, sizeof *s);
    s->valid = true; s->spec = *spec; s->kernel = kernel; s->code = code;
    *out = s; *hit = false;
    return OMEGA_GPU_DATTN_OK;
}

/* host-entry staging for `rows` tokens (grown on demand, contents not kept) */
static int bufs_locked(uint32_t rows) {
    if (g.have_bufs && g.buf_rows >= rows) return 0;
    if (g.have_bufs) { omega_gpu_session_free(&g.qkv); omega_gpu_session_free(&g.out); g.have_bufs = false; g.buf_rows = 0; }
    if (omega_gpu_session_alloc((size_t)rows * OMEGA_PATTN_QKV_WORDS * 4u, &g.qkv) != 0) return -1;
    if (omega_gpu_session_alloc((size_t)rows * OMEGA_PATTN_OUT_WORDS * 4u, &g.out) != 0) { omega_gpu_session_free(&g.qkv); return -1; }
    g.have_bufs = true; g.buf_rows = rows;
    return 0;
}

typedef struct { OmegaGpuDecodeAttnCacheView v; uint32_t ue, uo, uq; OmegaPrefillAttnSpec base; struct Mir *mir; } Run;

static int prepare(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t pos0, uint32_t m, float eps, float out_scale, Run *r) {
    memset(r, 0, sizeof *r);
    if (!c || m == 0 || omega_gpu_decode_attn_cache_view(c, layer, &r->v) != OMEGA_GPU_DATTN_OK) return OMEGA_GPU_DATTN_BAD_ARGS;
    if (pos0 > r->v.max_pos || m > r->v.max_pos - pos0) return OMEGA_GPU_DATTN_BAD_ARGS;
    r->base.warps = g.warps ? g.warps : OMEGA_GPU_PATTN_WARPS; r->base.mutant = g.mutant;
    if (!omega_pattn_spec_ok(&r->base)) return OMEGA_GPU_DATTN_BAD_ARGS;
    float qk_f = (float)(0.08838834764831845 * 1.4426950408889634); /* log2(e) / sqrt(128) */
    memcpy(&r->ue, &eps, 4); memcpy(&r->uo, &out_scale, 4); memcpy(&r->uq, &qk_f, 4);
    return OMEGA_GPU_DATTN_OK;
}

/* lock held, device open: one launch of `mode` over tokens pos0 .. pos0+mc-1 whose qkv / out rows start at the VAs */
/* K/V planes a launch reads (kc, vc, kvs) and, for APPEND with a mirror, also writes (kc2, vc2, kvs2) */
typedef struct { uint64_t kc, vc, kc2, vc2; uint32_t kvs, kvs2; } KvPlanes;
static int launch_locked(const Run *r, uint32_t mode, const KvPlanes *kv, uint64_t qkv_va, uint64_t out_va, uint32_t p0, uint32_t mc,
                         OmegaGpuPrefillAttnInfo *info) {
    OmegaPrefillAttnSpec spec = r->base; spec.mode = mode; spec.mirror = kv->kc2 ? 1u : 0u;
    Slot *ks = NULL; bool hit = false;
    int rc = kernel_for(&spec, &ks, &hit);
    if (rc != OMEGA_GPU_DATTN_OK) return rc;
    const uint32_t W = spec.warps;
    uint32_t args[OMEGA_PATTN_ARGS_WORDS];
    memset(args, 0, sizeof args);
#define A64(i, val) do { uint64_t _x = (val); args[i] = (uint32_t)_x; args[(i) + 1] = (uint32_t)(_x >> 32); } while (0)
    A64(0, qkv_va); A64(2, kv->kc); A64(4, kv->vc); A64(6, r->v.rope_va); A64(8, r->v.q_norm_va); A64(10, r->v.k_norm_va); A64(12, out_va); A64(20, kv->kc2); A64(22, kv->vc2);
#undef A64
    args[14] = p0; args[15] = mc; args[16] = kv->kvs; args[24] = kv->kvs2; args[17] = r->ue; args[18] = r->uo; args[19] = r->uq;
    OmegaGpuLaunch L = {
        .code_va = ks->code.va, .gpr_count = omega_bw_gpr_declared(ks->kernel.gpr_count),
        .threads_x = 32u * W, .threads_y = 1, .grid_x = (mc + W - 1u) / W, .grid_y = OMEGA_GPU_DATTN_KV_HEADS,
        .num_elements = (size_t)mc * OMEGA_PATTN_OUT_WORDS,
        .args = args, .n_args = OMEGA_PATTN_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
        .shared_bytes = OMEGA_PATTN_SHARED_BYTES,
    };
    if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    uint64_t el = 0; uint32_t marker = 0;
    if (omega_gpu_session_launch(&L, &el, &marker) != 0) return OMEGA_GPU_DATTN_CHIP_FAIL;
    if (g.n_launch < MAX_LAUNCH_LOG) g.launch_ns[g.n_launch] = el;
    g.n_launch++;
    if (info) {
        if (info->chunks == 0) { info->grid_x = L.grid_x; info->grid_y = L.grid_y; info->threads = L.threads_x; info->splits = 1; info->per_split = p0 + mc; info->kernel_cache_hit = hit; }
        else info->kernel_cache_hit &= hit;
        info->chunks++; info->elapsed_ns += el;
    }
    return OMEGA_GPU_DATTN_OK;
}

/* lock held, device open: all the launches for tokens pos0 .. pos0+m-1, rows at the VAs (device memory, row strides
 * 5120 / 4096 f32). Split (default): one APPEND then one ATTEND launch over all m tokens. Fused: 64-token chunks. */
static int run_locked(const Run *r, uint32_t pos0, uint32_t m, uint64_t qkv_va, uint64_t out_va, OmegaGpuPrefillAttnInfo *info) {
    int rc = OMEGA_GPU_DATTN_OK;
    const KvPlanes cache = { .kc = r->v.k_va, .vc = r->v.v_va, .kvs = r->v.max_pos * OMEGA_DATTN_ROW_BYTES };
    if (!g.fused) {
#if PATTN_HAVE_PF
        if (!g.mirror_off) {
            /* persistent GPU-cached mirror (same layout as the cache): rows valid .. pos0-1 that the cache gained since the
             * last call (decode steps) are copied in, rows >= pos0 are (re)written by APPEND to both places, ATTEND reads
             * only the mirror (L2 reuse across CTAs). Rows 0 .. min(valid, pos0)-1 are trusted: see the header contract. */
            struct Mir *mm = r->mir ? mirror_find(&r->v) : NULL;   /* re-found under the lock (forget may have moved it) */
            if (!mm) { omega_gpu_session_set_error("prefill attn: KV mirror not allocated"); return OMEGA_GPU_DATTN_CHIP_FAIL; }
            if (g.mirror_per_call) mm->valid = 0;
            const uint32_t t = mm->valid < pos0 ? mm->valid : pos0;
            mm->valid = 0;                                   /* until this call succeeds */
            const KvPlanes mir = { .kc = mm->mem.va, .vc = mm->mem.va + mm->plane, .kvs = cache.kvs };
            for (uint32_t h = 0; h < OMEGA_GPU_DATTN_KV_HEADS && t < pos0 && rc == OMEGA_GPU_DATTN_OK; h++) {
                uint64_t ns = 0;
                const uint64_t off = (uint64_t)h * cache.kvs + (uint64_t)t * OMEGA_DATTN_ROW_BYTES;
                const size_t bytes = (size_t)(pos0 - t) * OMEGA_DATTN_ROW_BYTES;
                int prc = copy_locked(mir.kc + off, cache.kc + off, bytes, &ns);
                if (prc == OMEGA_GPU_PF_OK) prc = copy_locked(mir.vc + off, cache.vc + off, bytes, &ns);
                if (prc != OMEGA_GPU_PF_OK) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
                if (g.n_launch < MAX_LAUNCH_LOG) g.launch_ns[g.n_launch] = ns;
                g.n_launch++;
                if (info) info->elapsed_ns += ns;
            }
            KvPlanes app = cache; app.kc2 = mir.kc; app.vc2 = mir.vc; app.kvs2 = mir.kvs;
            if (rc == OMEGA_GPU_DATTN_OK) rc = launch_locked(r, OMEGA_PATTN_MODE_APPEND, &app, qkv_va, out_va, pos0, m, info);
            if (rc == OMEGA_GPU_DATTN_OK) rc = launch_locked(r, OMEGA_PATTN_MODE_ATTEND, &mir, qkv_va, out_va, pos0, m, info);
            if (rc == OMEGA_GPU_DATTN_OK) mm->valid = pos0 + m;
            return rc;
        }
#endif
        rc = launch_locked(r, OMEGA_PATTN_MODE_APPEND, &cache, qkv_va, out_va, pos0, m, info);
        if (rc == OMEGA_GPU_DATTN_OK) rc = launch_locked(r, OMEGA_PATTN_MODE_ATTEND, &cache, qkv_va, out_va, pos0, m, info);
        return rc;
    }
    for (uint32_t done = 0; done < m && rc == OMEGA_GPU_DATTN_OK; done += OMEGA_PATTN_CHUNK) {
        const uint32_t mc = m - done < OMEGA_PATTN_CHUNK ? m - done : OMEGA_PATTN_CHUNK;
        rc = launch_locked(r, OMEGA_PATTN_MODE_FUSED, &cache, qkv_va + (uint64_t)done * OMEGA_PATTN_QKV_WORDS * 4u,
                           out_va + (uint64_t)done * OMEGA_PATTN_OUT_WORDS * 4u, pos0 + done, mc, info);
    }
    return rc;
}

static void info_head(OmegaGpuPrefillAttnInfo *info, uint32_t layer, uint32_t pos0, uint32_t m) {
    if (!info) return;
    memset(info, 0, sizeof *info);
    info->layer = layer; info->pos0 = pos0; info->m = m;
}

int omega_gpu_prefill_attn(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t pos0, uint32_t m, const float *qkv, float eps,
                           float out_scale, float *out, OmegaGpuPrefillAttnInfo *info) {
    uint64_t t0 = omega_gpu_session_now_ns();
    info_head(info, layer, pos0, m);
    Run r;
    if (!qkv || !out) return OMEGA_GPU_DATTN_BAD_ARGS;
    int rc = prepare(c, layer, pos0, m, eps, out_scale, &r);
    if (rc != OMEGA_GPU_DATTN_OK) return rc;
    const uint32_t pass = g.fused ? OMEGA_PATTN_CHUNK : OMEGA_GPU_PATTN_HOST_PASS;
#if PATTN_HAVE_PF
    if (!g.fused && !g.mirror_off && !(r.mir = mirror_get(&r.v))) return OMEGA_GPU_DATTN_CHIP_FAIL;
#endif

    omega_gpu_session_lock();
    if (!dev_open_locked()) { omega_gpu_session_unlock(); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    g.n_launch = 0;
    if (bufs_locked(m < pass ? m : pass) != 0) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
    for (uint32_t done = 0; done < m && rc == OMEGA_GPU_DATTN_OK; done += pass) {
        const uint32_t mp = m - done < pass ? m - done : pass;
        memcpy(g.qkv.cpu, qkv + (size_t)done * OMEGA_PATTN_QKV_WORDS, (size_t)mp * OMEGA_PATTN_QKV_WORDS * 4u);
        float *od = g.out.cpu;
        for (size_t i = 0; i < (size_t)mp * OMEGA_PATTN_OUT_WORDS; i++) od[i] = POISON_F32;
        if (omega_gpu_session_sync_to_gpu(&g.qkv) != 0 || omega_gpu_session_sync_to_gpu(&g.out) != 0) { rc = OMEGA_GPU_DATTN_CHIP_FAIL; break; }
        rc = run_locked(&r, pos0 + done, mp, g.qkv.va, g.out.va, info);
        if (rc != OMEGA_GPU_DATTN_OK) break;
        if (omega_gpu_session_sync_to_cpu(&g.out) != 0) { rc = OMEGA_GPU_DATTN_CHIP_FAIL; break; }
        size_t unchanged = 0;
        for (size_t i = 0; i < (size_t)mp * OMEGA_PATTN_OUT_WORDS; i++) if (od[i] == POISON_F32) unchanged++;
        if (unchanged) { omega_gpu_session_set_error("prefill attn: output poison survived"); rc = OMEGA_GPU_DATTN_CHIP_FAIL; break; }
        memcpy(out + (size_t)done * OMEGA_PATTN_OUT_WORDS, od, (size_t)mp * OMEGA_PATTN_OUT_WORDS * 4u);
    }
    omega_gpu_session_unlock();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}

int omega_gpu_prefill_attn_dev(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t pos0, uint32_t m, uint64_t qkv_va, float eps,
                               float out_scale, uint64_t out_va, OmegaGpuPrefillAttnInfo *info) {
    uint64_t t0 = omega_gpu_session_now_ns();
    info_head(info, layer, pos0, m);
    Run r;
    if (!qkv_va || !out_va || (qkv_va & 15u) || (out_va & 15u)) return OMEGA_GPU_DATTN_BAD_ARGS;
    int rc = prepare(c, layer, pos0, m, eps, out_scale, &r);
    if (rc != OMEGA_GPU_DATTN_OK) return rc;
#if PATTN_HAVE_PF
    if (!g.fused && !g.mirror_off && !(r.mir = mirror_get(&r.v))) return OMEGA_GPU_DATTN_CHIP_FAIL;
#endif
    omega_gpu_session_lock();
    if (!dev_open_locked()) { omega_gpu_session_unlock(); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    g.n_launch = 0;
    rc = run_locked(&r, pos0, m, qkv_va, out_va, info);
    omega_gpu_session_unlock();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}

/* ---- persistent K/V mirror: view and contract (see the header) ---- */
int omega_gpu_prefill_attn_kv_mirror_view(const OmegaGpuDecodeAttnCache *c, uint32_t layer, OmegaGpuPrefillAttnMirrorView *out) {
    OmegaGpuDecodeAttnCacheView v;
    if (!out) return OMEGA_GPU_DATTN_BAD_ARGS;
    memset(out, 0, sizeof *out);
    if (!c || omega_gpu_decode_attn_cache_view(c, layer, &v) != OMEGA_GPU_DATTN_OK) return OMEGA_GPU_DATTN_BAD_ARGS;
    omega_gpu_session_lock();
#if PATTN_HAVE_PF
    struct Mir *m = mirror_find(&v);
    if (m) { out->k_va = m->mem.va; out->v_va = m->mem.va + m->plane; out->kv_stride = v.max_pos * OMEGA_DATTN_ROW_BYTES; out->max_pos = v.max_pos; out->valid = m->valid; }
#endif
    omega_gpu_session_unlock();
    return out->k_va ? OMEGA_GPU_DATTN_OK : OMEGA_GPU_DATTN_BAD_ARGS;
}
static uint32_t layers_of(const OmegaGpuDecodeAttnCache *c) {
    OmegaGpuDecodeAttnCacheView v;
    return c && omega_gpu_decode_attn_cache_view(c, 0, &v) == OMEGA_GPU_DATTN_OK ? v.layers : 0;
}
void omega_gpu_prefill_attn_kv_mirror_invalidate(const OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t from_pos) {
    const uint32_t nl = layers_of(c);
    omega_gpu_session_lock();
#if PATTN_HAVE_PF
    for (uint32_t l = 0; l < nl; l++) {
        if (layer != UINT32_MAX && l != layer) continue;
        OmegaGpuDecodeAttnCacheView v;
        if (omega_gpu_decode_attn_cache_view(c, l, &v) != OMEGA_GPU_DATTN_OK) continue;
        struct Mir *m = mirror_find(&v);
        if (m && m->valid > from_pos) m->valid = from_pos;
    }
#else
    (void)layer; (void)from_pos; (void)nl;
#endif
    omega_gpu_session_unlock();
}
void omega_gpu_prefill_attn_kv_mirror_forget(const OmegaGpuDecodeAttnCache *c) {
#if PATTN_HAVE_PF
    const uint32_t nl = layers_of(c);
    for (uint32_t l = 0; l < nl; l++) {
        OmegaGpuDecodeAttnCacheView v;
        if (omega_gpu_decode_attn_cache_view(c, l, &v) != OMEGA_GPU_DATTN_OK) continue;
        omega_gpu_session_lock();
        struct Mir *m = mirror_find(&v);
        OmegaGpuMem mem; memset(&mem, 0, sizeof mem);
        if (m) { mem = m->mem; *m = g.mir[g.n_mir - 1]; memset(&g.mir[g.n_mir - 1], 0, sizeof g.mir[0]); g.n_mir--; }
        omega_gpu_session_unlock();
        if (mem.va) omega_gpu_pf_free(&mem);
    }
#else
    (void)c;
#endif
}
void omega_gpu_prefill_attn_test_set_mirror_per_call(int on) { omega_gpu_session_lock(); g.mirror_per_call = on ? 1 : 0; omega_gpu_session_unlock(); }
