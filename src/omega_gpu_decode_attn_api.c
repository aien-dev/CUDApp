/*
 * omega_gpu_decode_attn_api.c: resident KV cache, RoPE table, launch and host reference for the one-launch decode
 * attention (see the header). Session use mirrors omega_gpu_fp8_api.c: every buffer from omega_gpu_session_alloc
 * (GPU-uncached, host-visible), sync_to_gpu after host writes, sync_to_cpu before host reads, one
 * omega_gpu_session_launch per call.
 */
#include "omega_gpu_decode_attn_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_fmt.h"
#include <math.h>
#include <string.h>

#define MARKER_TIMEOUT_MS 20000ull
#define POISON_F32 (-999.0f)

struct OmegaGpuDecodeAttnCache {
    uint32_t layers, max_pos, max_splits;
    double theta;
    uint64_t plane_bytes;                     /* one K or V plane of a layer: 4 * max_pos * 256 + slack */
    OmegaGpuMem kv[OMEGA_GPU_DATTN_MAX_LAYERS]; /* [K plane | V plane] per layer */
    OmegaGpuMem rope, norms, part, cnt, qkv, out;
};

typedef struct {
    bool valid;
    OmegaDecodeAttnSpec spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
} Slot;

static struct {
    bool hooked;
    Slot slot;
    uint32_t depth, per_split;
    int mutant;
} g;

static void slot_clear_locked(void) { if (g.slot.valid) omega_gpu_session_free(&g.slot.code); memset(&g.slot, 0, sizeof g.slot); }
static void on_session_close(void) { memset(&g.slot, 0, sizeof g.slot); }
static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}
static uint32_t gpr_declared(uint32_t peak) { return ((peak + 1u + 7u) / 8u) * 8u; } /* chip rule, see omega_gpu_fp8_api.c */

void omega_gpu_decode_attn_set_depth(uint32_t depth) { omega_gpu_session_lock(); g.depth = depth; slot_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_decode_attn_set_per_split(uint32_t per_split) { omega_gpu_session_lock(); g.per_split = per_split; omega_gpu_session_unlock(); }
void omega_gpu_decode_attn_test_set_mutant(int mutant) { omega_gpu_session_lock(); g.mutant = mutant; slot_clear_locked(); omega_gpu_session_unlock(); }
const char *omega_gpu_decode_attn_last_error(void) { return omega_gpu_session_last_error(); }
const char *omega_gpu_decode_attn_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_DATTN_OK: return "OK";
    case OMEGA_GPU_DATTN_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_DATTN_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_DATTN_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_DATTN_CHIP_FAIL: return "CHIP_FAIL";
    default: return "?";
    }
}

/* ---- host math ---- */
uint16_t omega_dattn_bf16_rne(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t r = x + 0x7fffu + ((x >> 16) & 1u);
    return (uint16_t)(r >> 16);
}
float omega_dattn_bf16_f32(uint16_t b) { uint32_t x = (uint32_t)b << 16; float f; memcpy(&f, &x, 4); return f; }
void omega_dattn_rope_row(uint32_t pos, double theta, float *row128) {
    for (unsigned i = 0; i < 64; i++) {
        double inv = pow(theta, -(double)(2 * i) / 128.0), ang = (double)pos * inv;
        row128[i] = (float)cos(ang); row128[64 + i] = (float)sin(ang);
    }
}
/* RMSNorm + RoPE of one head (double), in place; cs: cos[64], sn: sin[64] */
static void norm_rope(double *x, const float *w, float eps, const double *cs, const double *sn, int rope) {
    double ss = 0; for (unsigned i = 0; i < 128; i++) ss += x[i] * x[i];
    double r = 1.0 / sqrt(ss / 128.0 + (double)eps);
    for (unsigned i = 0; i < 128; i++) x[i] = x[i] * r * (double)w[i];
    if (!rope) return;
    double y[128];
    for (unsigned i = 0; i < 64; i++) { y[i] = x[i] * cs[i] - x[i + 64] * sn[i]; y[i + 64] = x[i + 64] * cs[i] + x[i] * sn[i]; }
    memcpy(x, y, sizeof y);
}
void omega_decode_attn_reference(const float *qkv, const uint16_t *kc, const uint16_t *vc, uint32_t max_pos, uint32_t pos,
                                 const float *q_norm, const float *k_norm, float eps, double rope_theta, float out_scale,
                                 double *out, uint16_t *k_new, uint16_t *v_new) {
    double cs[64], sn[64];
    for (unsigned i = 0; i < 64; i++) { double ang = (double)pos * pow(rope_theta, -(double)(2 * i) / 128.0); cs[i] = cos(ang); sn[i] = sin(ang); }
    const double qscale = 1.0 / sqrt(128.0);
    for (unsigned gk = 0; gk < 4; gk++) {
        double k[128];
        for (unsigned i = 0; i < 128; i++) k[i] = qkv[4096 + gk * 128 + i];
        norm_rope(k, k_norm, eps, cs, sn, 1);
        for (unsigned i = 0; i < 128; i++) { k_new[gk * 128 + i] = omega_dattn_bf16_rne((float)k[i]); v_new[gk * 128 + i] = omega_dattn_bf16_rne(qkv[4096 + 512 + gk * 128 + i]); }
    }
    double *p = (double *)omega_gpu_host_calloc((size_t)pos + 1, sizeof(double));
    for (unsigned h = 0; h < 32; h++) {
        unsigned gk = h / 8;
        double q[128];
        for (unsigned i = 0; i < 128; i++) q[i] = qkv[h * 128 + i];
        norm_rope(q, q_norm, eps, cs, sn, 1);
        double mx = -1e300;
        for (uint32_t t = 0; t <= pos; t++) {
            const uint16_t *kr = t == pos ? k_new + gk * 128 : kc + ((size_t)gk * max_pos + t) * 128;
            double s = 0; for (unsigned i = 0; i < 128; i++) s += q[i] * (double)omega_dattn_bf16_f32(kr[i]);
            s *= qscale; p[t] = s; if (s > mx) mx = s;
        }
        double l = 0; for (uint32_t t = 0; t <= pos; t++) { p[t] = exp(p[t] - mx); l += p[t]; }
        double o[128]; memset(o, 0, sizeof o);
        for (uint32_t t = 0; t <= pos; t++) {
            const uint16_t *vr = t == pos ? v_new + gk * 128 : vc + ((size_t)gk * max_pos + t) * 128;
            for (unsigned i = 0; i < 128; i++) o[i] += p[t] * (double)omega_dattn_bf16_f32(vr[i]);
        }
        for (unsigned i = 0; i < 128; i++) out[h * 128 + i] = o[i] / l * (double)out_scale;
    }
    omega_gpu_host_free(p);
}

/* ---- cache ---- */
int omega_gpu_decode_attn_cache_alloc(uint32_t layers, uint32_t max_pos, double rope_theta, OmegaGpuDecodeAttnCache **out) {
    if (!out) return OMEGA_GPU_DATTN_BAD_ARGS;
    *out = NULL;
    if (layers == 0 || layers > OMEGA_GPU_DATTN_MAX_LAYERS || max_pos == 0 || max_pos > OMEGA_GPU_DATTN_MAX_POS || max_pos % OMEGA_DATTN_CHUNK || !(rope_theta > 0)) return OMEGA_GPU_DATTN_BAD_ARGS;
    OmegaGpuDecodeAttnCache *c = omega_gpu_host_calloc(1, sizeof *c);
    if (!c) return OMEGA_GPU_DATTN_CHIP_FAIL;
    c->layers = layers; c->max_pos = max_pos; c->theta = rope_theta;
    c->max_splits = max_pos / OMEGA_DATTN_CHUNK;
    c->plane_bytes = (uint64_t)OMEGA_GPU_DATTN_KV_HEADS * max_pos * OMEGA_DATTN_ROW_BYTES + OMEGA_DATTN_PLANE_SLACK;
    int rc = OMEGA_GPU_DATTN_OK;
    omega_gpu_session_lock();
    if (!dev_open_locked()) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
    for (uint32_t L = 0; L < layers && rc == OMEGA_GPU_DATTN_OK; L++) if (omega_gpu_session_alloc((size_t)(2 * c->plane_bytes), &c->kv[L]) != 0) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
    if (rc == OMEGA_GPU_DATTN_OK && (omega_gpu_session_alloc((size_t)max_pos * OMEGA_DATTN_ROPE_ROW_BYTES, &c->rope) != 0 ||
                                     omega_gpu_session_alloc((size_t)layers * 2 * 128 * 4, &c->norms) != 0 ||
                                     omega_gpu_session_alloc((size_t)OMEGA_GPU_DATTN_KV_HEADS * c->max_splits * OMEGA_DATTN_PARTIAL_WORDS * 4, &c->part) != 0 ||
                                     omega_gpu_session_alloc(64, &c->cnt) != 0 ||
                                     omega_gpu_session_alloc((size_t)OMEGA_GPU_DATTN_QKV_WORDS * 4, &c->qkv) != 0 ||
                                     omega_gpu_session_alloc((size_t)OMEGA_GPU_DATTN_Q_HEADS * 128 * 4, &c->out) != 0)) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
    if (rc == OMEGA_GPU_DATTN_OK) {
        float *r = c->rope.cpu;
        for (uint32_t p = 0; p < max_pos; p++) omega_dattn_rope_row(p, rope_theta, r + (size_t)p * 128);
        memset(c->cnt.cpu, 0, 64);
        memset(c->norms.cpu, 0, (size_t)layers * 2 * 128 * 4);
        for (uint32_t L = 0; L < layers; L++) memset(c->kv[L].cpu, 0, (size_t)(2 * c->plane_bytes));
        if (omega_gpu_session_sync_to_gpu(&c->rope) != 0 || omega_gpu_session_sync_to_gpu(&c->cnt) != 0 || omega_gpu_session_sync_to_gpu(&c->norms) != 0) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
        for (uint32_t L = 0; L < layers && rc == OMEGA_GPU_DATTN_OK; L++) if (omega_gpu_session_sync_to_gpu(&c->kv[L]) != 0) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
    }
    omega_gpu_session_unlock();
    if (rc != OMEGA_GPU_DATTN_OK) { omega_gpu_decode_attn_cache_free(c); return rc; }
    *out = c;
    return OMEGA_GPU_DATTN_OK;
}
void omega_gpu_decode_attn_cache_free(OmegaGpuDecodeAttnCache *c) {
    if (!c) return;
    omega_gpu_session_lock();
    for (uint32_t L = 0; L < c->layers; L++) if (c->kv[L].cpu) omega_gpu_session_free(&c->kv[L]);
    OmegaGpuMem *ms[] = { &c->rope, &c->norms, &c->part, &c->cnt, &c->qkv, &c->out };
    for (unsigned i = 0; i < 6; i++) if (ms[i]->cpu) omega_gpu_session_free(ms[i]);
    omega_gpu_session_unlock();
    omega_gpu_host_free(c);
}
int omega_gpu_decode_attn_set_norms(OmegaGpuDecodeAttnCache *c, uint32_t layer, const float *q_norm, const float *k_norm) {
    if (!c || layer >= c->layers || !q_norm || !k_norm) return OMEGA_GPU_DATTN_BAD_ARGS;
    omega_gpu_session_lock();
    float *n = (float *)c->norms.cpu + (size_t)layer * 256;
    memcpy(n, q_norm, 512); memcpy(n + 128, k_norm, 512);
    int rc = omega_gpu_session_sync_to_gpu(&c->norms) == 0 ? OMEGA_GPU_DATTN_OK : OMEGA_GPU_DATTN_CHIP_FAIL;
    omega_gpu_session_unlock();
    return rc;
}
static uint16_t *row_ptr(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t kv_head, uint32_t pos, int plane) {
    return (uint16_t *)((uint8_t *)c->kv[layer].cpu + (plane ? c->plane_bytes : 0) + ((size_t)kv_head * c->max_pos + pos) * OMEGA_DATTN_ROW_BYTES);
}
int omega_gpu_decode_attn_cache_write(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t kv_head, uint32_t pos, uint32_t count, const uint16_t *k, const uint16_t *v) {
    if (!c || layer >= c->layers || kv_head >= 4 || pos >= c->max_pos || count == 0 || count > c->max_pos - pos || !k || !v) return OMEGA_GPU_DATTN_BAD_ARGS;
    omega_gpu_session_lock();
    memcpy(row_ptr(c, layer, kv_head, pos, 0), k, (size_t)count * 256); memcpy(row_ptr(c, layer, kv_head, pos, 1), v, (size_t)count * 256);
    int rc = omega_gpu_session_sync_to_gpu(&c->kv[layer]) == 0 ? OMEGA_GPU_DATTN_OK : OMEGA_GPU_DATTN_CHIP_FAIL;
    omega_gpu_session_unlock();
    return rc;
}
int omega_gpu_decode_attn_cache_read(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t kv_head, uint32_t pos, uint16_t *k, uint16_t *v) {
    if (!c || layer >= c->layers || kv_head >= 4 || pos >= c->max_pos || !k || !v) return OMEGA_GPU_DATTN_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = omega_gpu_session_sync_to_cpu(&c->kv[layer]) == 0 ? OMEGA_GPU_DATTN_OK : OMEGA_GPU_DATTN_CHIP_FAIL;
    if (rc == OMEGA_GPU_DATTN_OK) { memcpy(k, row_ptr(c, layer, kv_head, pos, 0), 256); memcpy(v, row_ptr(c, layer, kv_head, pos, 1), 256); }
    omega_gpu_session_unlock();
    return rc;
}

/* ---- kernel ---- */
static int kernel_for(const OmegaDecodeAttnSpec *spec, Slot **out, bool *hit) {
    Slot *s = &g.slot;
    if (s->valid && memcmp(&s->spec, spec, sizeof *spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_DATTN_OK; }
    slot_clear_locked();
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("decode attn codegen: host memory"); return OMEGA_GPU_DATTN_CODEGEN_FAIL; }
    int cg = omega_blackwell_codegen_decode_attn_prog(spec, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("decode attn codegen refused the spec"); return OMEGA_GPU_DATTN_CODEGEN_FAIL; }
    if ((uint64_t)OMEGA_DATTN_CHUNK * gpr_declared(prog->regalloc.peak_gpr_usage) > 65536u) { omega_gpu_host_free(prog); omega_gpu_session_set_error("decode attn: register file"); return OMEGA_GPU_DATTN_TOO_LARGE; }
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    OmegaBlackwellKernel kernel;
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need) { omega_gpu_session_free(&code); omega_gpu_session_set_error("decode attn encode"); return OMEGA_GPU_DATTN_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    memset(s, 0, sizeof *s);
    s->valid = true; s->spec = *spec; s->kernel = kernel; s->code = code;
    *out = s; *hit = false;
    return OMEGA_GPU_DATTN_OK;
}

int omega_gpu_decode_attn(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t pos, const float *qkv, float eps, float out_scale, float *out, OmegaGpuDecodeAttnInfo *info) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    if (!c || layer >= c->layers || pos >= c->max_pos || !qkv || !out) return OMEGA_GPU_DATTN_BAD_ARGS;
    const uint32_t per = g.per_split ? g.per_split : OMEGA_GPU_DATTN_PER_SPLIT;
    if (per % OMEGA_DATTN_CHUNK) return OMEGA_GPU_DATTN_BAD_ARGS;
    const uint32_t splits = (pos + per) / per; /* ceil((pos + 1) / per) */
    if (splits > c->max_splits) return OMEGA_GPU_DATTN_TOO_LARGE;
    OmegaDecodeAttnSpec spec; memset(&spec, 0, sizeof spec);
    spec.depth = g.depth ? g.depth : OMEGA_GPU_DATTN_DEPTH; spec.mutant = g.mutant;

    omega_gpu_session_lock();
    int rc = OMEGA_GPU_DATTN_OK;
    if (!dev_open_locked()) { omega_gpu_session_unlock(); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    Slot *ks = NULL; bool hit = false;
    rc = kernel_for(&spec, &ks, &hit);
    if (rc != OMEGA_GPU_DATTN_OK) { omega_gpu_session_unlock(); return rc; }
    memcpy(c->qkv.cpu, qkv, (size_t)OMEGA_GPU_DATTN_QKV_WORDS * 4);
    float *od = c->out.cpu;
    for (size_t i = 0; i < 32 * 128; i++) od[i] = POISON_F32;
    memset(c->cnt.cpu, 0, 64);
    if (omega_gpu_session_sync_to_gpu(&c->qkv) != 0 || omega_gpu_session_sync_to_gpu(&c->out) != 0 || omega_gpu_session_sync_to_gpu(&c->cnt) != 0) rc = OMEGA_GPU_DATTN_CHIP_FAIL;

    uint32_t args[OMEGA_DATTN_ARGS_WORDS];
    const uint64_t kc = c->kv[layer].va, vc = kc + c->plane_bytes, nm = c->norms.va + (uint64_t)layer * 1024;
    float fe = eps, fo = out_scale, fq = (float)((1.0 / sqrt(128.0)) * 1.4426950408889634);
    uint32_t ue, uo, uq; memcpy(&ue, &fe, 4); memcpy(&uo, &fo, 4); memcpy(&uq, &fq, 4);
#define A64(i, v) do { args[i] = (uint32_t)(v); args[(i) + 1] = (uint32_t)((v) >> 32); } while (0)
    A64(0, c->qkv.va); A64(2, kc); A64(4, vc); A64(6, c->rope.va); A64(8, nm); A64(10, nm + 512); A64(12, c->out.va); A64(14, c->part.va); A64(16, c->cnt.va);
#undef A64
    args[18] = pos; args[19] = per; args[20] = splits; args[21] = (uint32_t)(c->max_pos * OMEGA_DATTN_ROW_BYTES); args[22] = ue; args[23] = uo; args[24] = uq;
    if (info) {
        info->layer = layer; info->pos = pos; info->splits = splits; info->per_split = per; info->depth = spec.depth;
        info->grid_x = splits; info->grid_y = OMEGA_GPU_DATTN_KV_HEADS; info->threads = OMEGA_DATTN_CHUNK;
        info->kv_bytes = 2ull * OMEGA_GPU_DATTN_KV_HEADS * (pos + 1ull) * OMEGA_DATTN_ROW_BYTES;
        info->kernel_cache_hit = hit;
    }
    if (rc == OMEGA_GPU_DATTN_OK) {
        OmegaGpuLaunch L = {
            .code_va = ks->code.va, .gpr_count = gpr_declared(ks->kernel.gpr_count),
            .threads_x = OMEGA_DATTN_CHUNK, .threads_y = 1, .grid_x = splits, .grid_y = OMEGA_GPU_DATTN_KV_HEADS,
            .num_elements = (size_t)32 * 128,
            .args = args, .n_args = OMEGA_DATTN_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
            .shared_bytes = OMEGA_DATTN_SHARED_BYTES,
        };
        if (omega_gpu_session_launch(&L, info ? &info->elapsed_ns : NULL, info ? &info->completion_marker : NULL) != 0) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_DATTN_OK && (omega_gpu_session_sync_to_cpu(&c->out) != 0 || omega_gpu_session_sync_to_cpu(&c->cnt) != 0)) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
    if (rc == OMEGA_GPU_DATTN_OK) {
        size_t unchanged = 0;
        for (size_t i = 0; i < 32 * 128; i++) if (od[i] == POISON_F32) unchanged++;
        const uint32_t *cn = c->cnt.cpu;
        if (unchanged) { omega_gpu_session_set_error("decode attn: output poison survived"); rc = OMEGA_GPU_DATTN_CHIP_FAIL; }
        else if (cn[0] | cn[1] | cn[2] | cn[3]) { omega_gpu_session_set_error("decode attn: a split counter did not return to zero"); rc = OMEGA_GPU_DATTN_CHIP_FAIL; }
        else memcpy(out, od, (size_t)32 * 128 * 4);
    }
    omega_gpu_session_unlock();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}
