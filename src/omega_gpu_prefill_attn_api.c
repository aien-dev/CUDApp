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

#define MARKER_TIMEOUT_MS 20000ull
#define POISON_F32 (-999.0f)

typedef struct {
    bool valid;
    OmegaPrefillAttnSpec spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
} Slot;

static struct {
    bool hooked;
    Slot slot;
    uint32_t warps;
    int mutant;
    bool have_bufs;
    OmegaGpuMem qkv, out;
} g;

static void slot_clear_locked(void) { if (g.slot.valid) omega_gpu_session_free(&g.slot.code); memset(&g.slot, 0, sizeof g.slot); }
static void on_session_close(void) { memset(&g.slot, 0, sizeof g.slot); g.have_bufs = false; memset(&g.qkv, 0, sizeof g.qkv); memset(&g.out, 0, sizeof g.out); }
static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}

void omega_gpu_prefill_attn_set_warps(uint32_t warps) { omega_gpu_session_lock(); g.warps = warps; slot_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_prefill_attn_test_set_mutant(int mutant) { omega_gpu_session_lock(); g.mutant = mutant; slot_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_prefill_attn_release(void) {
    omega_gpu_session_lock();
    slot_clear_locked();
    if (g.have_bufs) { omega_gpu_session_free(&g.qkv); omega_gpu_session_free(&g.out); g.have_bufs = false; }
    omega_gpu_session_unlock();
}

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
    Slot *s = &g.slot;
    if (s->valid && memcmp(&s->spec, spec, sizeof *spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_DATTN_OK; }
    slot_clear_locked();
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

static int bufs_locked(void) {
    if (g.have_bufs) return 0;
    if (omega_gpu_session_alloc((size_t)OMEGA_PATTN_CHUNK * OMEGA_PATTN_QKV_WORDS * 4u, &g.qkv) != 0) return -1;
    if (omega_gpu_session_alloc((size_t)OMEGA_PATTN_CHUNK * OMEGA_PATTN_OUT_WORDS * 4u, &g.out) != 0) { omega_gpu_session_free(&g.qkv); return -1; }
    memset(g.qkv.cpu, 0, (size_t)OMEGA_PATTN_CHUNK * OMEGA_PATTN_QKV_WORDS * 4u);
    g.have_bufs = true;
    return 0;
}

int omega_gpu_prefill_attn(OmegaGpuDecodeAttnCache *c, uint32_t layer, uint32_t pos0, uint32_t m, const float *qkv, float eps,
                           float out_scale, float *out, OmegaGpuPrefillAttnInfo *info) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    OmegaGpuDecodeAttnCacheView v;
    if (!c || !qkv || !out || m == 0 || omega_gpu_decode_attn_cache_view(c, layer, &v) != OMEGA_GPU_DATTN_OK) return OMEGA_GPU_DATTN_BAD_ARGS;
    if (pos0 > v.max_pos || m > v.max_pos - pos0) return OMEGA_GPU_DATTN_BAD_ARGS;
    OmegaPrefillAttnSpec spec; memset(&spec, 0, sizeof spec);
    spec.warps = g.warps ? g.warps : OMEGA_GPU_PATTN_WARPS; spec.mutant = g.mutant;
    if (!omega_pattn_spec_ok(&spec)) return OMEGA_GPU_DATTN_BAD_ARGS;
    const uint32_t W = spec.warps, chunks = (m + OMEGA_PATTN_CHUNK - 1u) / OMEGA_PATTN_CHUNK;

    omega_gpu_session_lock();
    int rc = OMEGA_GPU_DATTN_OK;
    if (!dev_open_locked()) { omega_gpu_session_unlock(); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    Slot *ks = NULL; bool hit = false;
    rc = kernel_for(&spec, &ks, &hit);
    if (rc == OMEGA_GPU_DATTN_OK && bufs_locked() != 0) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
    if (info) {
        uint32_t mc0 = m < OMEGA_PATTN_CHUNK ? m : OMEGA_PATTN_CHUNK;
        info->layer = layer; info->pos0 = pos0; info->m = m; info->chunks = chunks;
        info->grid_x = (mc0 + W - 1u) / W; info->grid_y = OMEGA_GPU_DATTN_KV_HEADS; info->threads = 32u * W;
        info->splits = 1; info->per_split = pos0 + mc0;
        info->kernel_cache_hit = hit;
    }
    float eps_f = eps, os_f = out_scale, qk_f = (float)(0.08838834764831845 * 1.4426950408889634); /* log2(e) / sqrt(128) */
    uint32_t ue, uo, uq; memcpy(&ue, &eps_f, 4); memcpy(&uo, &os_f, 4); memcpy(&uq, &qk_f, 4);
    for (uint32_t ch = 0; ch < chunks && rc == OMEGA_GPU_DATTN_OK; ch++) {
        const uint32_t p0 = pos0 + ch * OMEGA_PATTN_CHUNK, mc = m - ch * OMEGA_PATTN_CHUNK < OMEGA_PATTN_CHUNK ? m - ch * OMEGA_PATTN_CHUNK : OMEGA_PATTN_CHUNK;
        memcpy(g.qkv.cpu, qkv + (size_t)ch * OMEGA_PATTN_CHUNK * OMEGA_PATTN_QKV_WORDS, (size_t)mc * OMEGA_PATTN_QKV_WORDS * 4u);
        float *od = g.out.cpu;
        for (size_t i = 0; i < (size_t)mc * OMEGA_PATTN_OUT_WORDS; i++) od[i] = POISON_F32;
        if (omega_gpu_session_sync_to_gpu(&g.qkv) != 0 || omega_gpu_session_sync_to_gpu(&g.out) != 0) { rc = OMEGA_GPU_DATTN_CHIP_FAIL; break; }
        uint32_t args[OMEGA_PATTN_ARGS_WORDS];
        memset(args, 0, sizeof args);
#define A64(i, val) do { uint64_t _x = (val); args[i] = (uint32_t)_x; args[(i) + 1] = (uint32_t)(_x >> 32); } while (0)
        A64(0, g.qkv.va); A64(2, v.k_va); A64(4, v.v_va); A64(6, v.rope_va); A64(8, v.q_norm_va); A64(10, v.k_norm_va); A64(12, g.out.va);
#undef A64
        args[14] = p0; args[15] = mc; args[16] = v.max_pos * OMEGA_DATTN_ROW_BYTES; args[17] = ue; args[18] = uo; args[19] = uq;
        OmegaGpuLaunch L = {
            .code_va = ks->code.va, .gpr_count = omega_bw_gpr_declared(ks->kernel.gpr_count),
            .threads_x = 32u * W, .threads_y = 1, .grid_x = (mc + W - 1u) / W, .grid_y = OMEGA_GPU_DATTN_KV_HEADS,
            .num_elements = (size_t)mc * OMEGA_PATTN_OUT_WORDS,
            .args = args, .n_args = OMEGA_PATTN_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
            .shared_bytes = OMEGA_PATTN_SHARED_BYTES,
        };
        uint64_t el = 0; uint32_t marker = 0;
        if (!omega_bw_gpr_declared_ok(L.gpr_count, ks->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); rc = OMEGA_GPU_DATTN_CHIP_FAIL; break; }
        if (omega_gpu_session_launch(&L, &el, &marker) != 0) { rc = OMEGA_GPU_DATTN_CHIP_FAIL; break; }
        if (info) info->elapsed_ns += el;
        if (omega_gpu_session_sync_to_cpu(&g.out) != 0) { rc = OMEGA_GPU_DATTN_CHIP_FAIL; break; }
        size_t unchanged = 0;
        for (size_t i = 0; i < (size_t)mc * OMEGA_PATTN_OUT_WORDS; i++) if (od[i] == POISON_F32) unchanged++;
        if (unchanged) { omega_gpu_session_set_error("prefill attn: output poison survived"); rc = OMEGA_GPU_DATTN_CHIP_FAIL; break; }
        memcpy(out + (size_t)ch * OMEGA_PATTN_CHUNK * OMEGA_PATTN_OUT_WORDS, od, (size_t)mc * OMEGA_PATTN_OUT_WORDS * 4u);
    }
    omega_gpu_session_unlock();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}
