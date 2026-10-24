/*
 * omega_gpu_prefill_attn256_api.c: the two launches of the head_dim 256 prefill attention (omega-mv#545; see the
 * header). The prep kernel is the decode kernel's own program in prep mode (omega_blackwell_codegen_attn256.h), the
 * attend kernel is omega_blackwell_codegen_pattn256.h; it uses the SHFL.BFLY / F2FP placeholders of
 * omega_blackwell_codegen_tc.h, so its code is patched after encoding and the patch count must equal the program's
 * placeholder count. Session use mirrors omega_gpu_prefill_attn_api.c.
 */
#include "omega_gpu_prefill_attn256_api.h"
#include "omega_blackwell_codegen_tc.h"
#include "omega_gpu_session.h"
#include "omega_gpu_cmdlist.h"
#include <string.h>

#define MARKER_TIMEOUT_MS 60000ull
#define POISON_F32 (-999.0f)
#define QKVW OMEGA_DATTN256_QKV_WORDS
#define OUTW OMEGA_PATTN256_OUT_WORDS
enum { KK_PREP, KK_ATTEND, KK_N };

typedef struct {
    bool valid;
    union { OmegaDecodeAttn256Spec prep; OmegaPrefillAttn256Spec att; } spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
    uint32_t insns;
} Slot;

static struct {
    bool hooked;
    Slot slot[KK_N];
    int prep_mut, att_mut;
    OmegaGpuMem qf; uint32_t qf_rows; bool have_qf;           /* q' scratch for qf_rows tokens */
    OmegaGpuMem qkv, out; uint32_t buf_rows; bool have_bufs;  /* host-entry staging */
} g;

static void bufs_forget(void) { g.have_qf = false; g.qf_rows = 0; g.have_bufs = false; g.buf_rows = 0; }
static void on_session_close(void) { memset(g.slot, 0, sizeof g.slot); bufs_forget(); }
static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}
static void slots_clear_locked(void) {
    for (unsigned k = 0; k < KK_N; k++) { if (g.slot[k].valid) omega_gpu_session_free(&g.slot[k].code); memset(&g.slot[k], 0, sizeof g.slot[k]); }
}
void omega_gpu_prefill_attn256_test_set_mutant(int prep_mutant, int attend_mutant) {
    omega_gpu_session_lock(); g.prep_mut = prep_mutant; g.att_mut = attend_mutant; slots_clear_locked(); omega_gpu_session_unlock();
}
void omega_gpu_prefill_attn256_release(void) {
    omega_gpu_session_lock();
    if (g.have_qf) omega_gpu_session_free(&g.qf);
    if (g.have_bufs) { omega_gpu_session_free(&g.qkv); omega_gpu_session_free(&g.out); }
    bufs_forget();
    omega_gpu_session_unlock();
}

static void spec_of(unsigned kind, Slot *tmp) {
    memset(tmp, 0, sizeof *tmp);
    if (kind == KK_PREP) { tmp->spec.prep.depth = OMEGA_GPU_DATTN256_DEPTH; tmp->spec.prep.mutant = g.prep_mut; tmp->spec.prep.prep = 1; }
    else tmp->spec.att.mutant = g.att_mut;
}
static int prog_of(unsigned kind, const Slot *tmp, BlackwellIRProgram *prog) {
    return kind == KK_PREP ? omega_blackwell_codegen_decode_attn256_prog(&tmp->spec.prep, prog)
                           : omega_blackwell_codegen_prefill_attn256_prog(&tmp->spec.att, prog);
}
static uint32_t placeholders_of(const BlackwellIRProgram *prog) {
    uint32_t n = 0;
    for (size_t i = 0; i < prog->count; i++) {
        const BlackwellIRInsn *in = &prog->insns[i];
        if ((in->op == BW_IR_SHFL_DOWN && (in->imm & OMEGA_TC_MARK_BFLY)) || (in->op == BW_IR_FMNMX_MAX && in->imm == OMEGA_TC_MARK_F2FP)) n++;
    }
    return n;
}

size_t omega_gpu_prefill_attn256_encode(unsigned kind, uint8_t *code, size_t cap, uint32_t *gprs, int *patched) {
    if (kind >= KK_N) return 0;
    Slot tmp; spec_of(kind, &tmp);
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) return 0;
    size_t need = 0;
    if (prog_of(kind, &tmp, prog) == 0) need = omega_bw_program_code_bytes(prog);
    OmegaBlackwellKernel k;
    if (need == 0 || need > cap || omega_blackwell_kernel_encode_into(prog, code, cap, &k) != 0) { omega_gpu_host_free(prog); return 0; }
    int np = omega_tc_patch_code(prog, code, need);
    if (gprs) *gprs = k.gpr_count;
    if (patched) *patched = np;
    const int ok = np >= 0 && (uint32_t)np == placeholders_of(prog);
    omega_gpu_host_free(prog);
    return ok ? need : 0;
}

/* rec: a record call is running (lock held): a miss is refused instead of compiled */
static int kernel_for(unsigned kind, int rec, Slot **out, bool *hit) {
    Slot tmp; spec_of(kind, &tmp);
    Slot *s = &g.slot[kind];
    if (s->valid && memcmp(&s->spec, &tmp.spec, sizeof tmp.spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_DATTN_OK; }
    if (rec) { omega_gpu_session_set_error("prefill attn256 record: kernel cache miss (warm up before recording)"); return OMEGA_GPU_DATTN_TOO_LARGE; }
    if (s->valid) omega_gpu_session_free(&s->code);
    memset(s, 0, sizeof *s);
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("prefill attn256 codegen: host memory"); return OMEGA_GPU_DATTN_CODEGEN_FAIL; }
    int cg = prog_of(kind, &tmp, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("prefill attn256 codegen refused the spec"); return OMEGA_GPU_DATTN_CODEGEN_FAIL; }
    /* both launches run 256 threads per CTA: the declared registers must fit the 64K register file (CUDA Programming
     * Guide, Compute Capabilities: "maximum number of 32-bit registers per thread block is 64K") */
    if ((uint64_t)OMEGA_PATTN256_THREADS * omega_bw_gpr_declared(prog->regalloc.peak_gpr_usage) > 65536u) { omega_gpu_host_free(prog); omega_gpu_session_set_error("prefill attn256: register file"); return OMEGA_GPU_DATTN_TOO_LARGE; }
    const uint32_t expect = placeholders_of(prog), insns = (uint32_t)prog->count;
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    OmegaBlackwellKernel kernel;
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    int np = cg == 0 ? omega_tc_patch_code(prog, code.cpu, need) : -1;
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need) { omega_gpu_session_free(&code); omega_gpu_session_set_error("prefill attn256 encode"); return OMEGA_GPU_DATTN_CODEGEN_FAIL; }
    if (np < 0 || (uint32_t)np != expect) { omega_gpu_session_free(&code); omega_gpu_session_set_error("prefill attn256: placeholder patch count mismatch"); return OMEGA_GPU_DATTN_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    memset(s, 0, sizeof *s);
    s->valid = true; s->spec = tmp.spec; s->kernel = kernel; s->code = code; s->insns = insns;
    *out = s; *hit = false;
    return OMEGA_GPU_DATTN_OK;
}

static size_t qf_bytes(uint32_t rows) { return (size_t)((rows + 1u) / 2u) * OMEGA_PATTN256_QF_PAIR_BYTES; }
static int qf_locked(uint32_t rows) {
    if (g.have_qf && g.qf_rows >= rows) return 0;
    if (g.have_qf) { omega_gpu_session_free(&g.qf); g.have_qf = false; g.qf_rows = 0; }
    if (omega_gpu_session_alloc(qf_bytes(rows), &g.qf) != 0) return -1;
    g.have_qf = true; g.qf_rows = rows;
    return 0;
}
int omega_gpu_prefill_attn256_reserve(uint32_t max_m) {
    if (max_m == 0 || max_m > OMEGA_GPU_PATTN256_MAX_M) return OMEGA_GPU_DATTN_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = dev_open_locked() && qf_locked(max_m) == 0 ? OMEGA_GPU_DATTN_OK : OMEGA_GPU_DATTN_CHIP_FAIL;
    /* and builds both kernels, so a record call after reserve never misses the cache (sc #642 prefill graph) */
    for (unsigned k = 0; k < KK_N && rc == OMEGA_GPU_DATTN_OK; k++) { Slot *s = NULL; bool hit = false; rc = kernel_for(k, 0, &s, &hit); }
    omega_gpu_session_unlock();
    return rc;
}
static int bufs_locked(uint32_t rows) {
    if (g.have_bufs && g.buf_rows >= rows) return 0;
    if (g.have_bufs) { omega_gpu_session_free(&g.qkv); omega_gpu_session_free(&g.out); g.have_bufs = false; g.buf_rows = 0; }
    if (omega_gpu_session_alloc((size_t)rows * QKVW * 4u, &g.qkv) != 0) return -1;
    if (omega_gpu_session_alloc((size_t)rows * OUTW * 4u, &g.out) != 0) { omega_gpu_session_free(&g.qkv); return -1; }
    g.have_bufs = true; g.buf_rows = rows;
    return 0;
}

/* host != 0: qkv / out are host arrays staged through the module buffers (poison-checked); else qkv_va / out_va are
 * device buffers. rec non-NULL: record path (lock held, nothing allocated, compiled or submitted). */
static int core(OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos0, uint32_t m, const float *qkv, float *out,
                uint64_t qkv_va, uint64_t out_va, float eps, float out_scale, OmegaGpuPrefillAttn256Info *info, OmegaGpuCmdList *rec) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    const int host = qkv != NULL;
    OmegaGpuDecodeAttn256CacheView v;
    if (!c || m == 0 || m > OMEGA_GPU_PATTN256_MAX_M || omega_gpu_decode_attn256_cache_view(c, layer, &v) != OMEGA_GPU_DATTN_OK ||
        pos0 > v.max_pos || m > v.max_pos - pos0 || (host ? !out : (!qkv_va || !out_va || (qkv_va & 15u) || (out_va & 15u)))) return OMEGA_GPU_DATTN_BAD_ARGS;
    if (!rec) omega_gpu_session_lock();
#define UNLOCK() do { if (!rec) omega_gpu_session_unlock(); } while (0)
    if (!dev_open_locked()) { UNLOCK(); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    Slot *ks[KK_N] = { NULL, NULL }; bool hit[KK_N] = { false, false };
    int rc = OMEGA_GPU_DATTN_OK;
    for (unsigned k = 0; k < KK_N && rc == OMEGA_GPU_DATTN_OK; k++) rc = kernel_for(k, rec != NULL, &ks[k], &hit[k]);
    if (rc != OMEGA_GPU_DATTN_OK) { UNLOCK(); return rc; }
    if (rec) { if (!g.have_qf || g.qf_rows < m) { omega_gpu_session_set_error("prefill attn256 record: q' scratch not reserved for m tokens"); return OMEGA_GPU_DATTN_TOO_LARGE; } }
    else if (qf_locked(m) != 0) { UNLOCK(); return OMEGA_GPU_DATTN_CHIP_FAIL; }
    if (host) {
        if (bufs_locked(m) != 0) { UNLOCK(); return OMEGA_GPU_DATTN_CHIP_FAIL; }
        memcpy(g.qkv.cpu, qkv, (size_t)m * QKVW * 4u);
        float *od = g.out.cpu;
        for (size_t i = 0; i < (size_t)m * OUTW; i++) od[i] = POISON_F32;
        if (omega_gpu_session_sync_to_gpu(&g.qkv) != 0 || omega_gpu_session_sync_to_gpu(&g.out) != 0) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
        qkv_va = g.qkv.va; out_va = g.out.va;
    }
    const uint32_t kvs = v.max_pos * OMEGA_DATTN256_ROW_BYTES, gx_att = (m + OMEGA_PATTN256_TOKENS - 1u) / OMEGA_PATTN256_TOKENS;
    float fe = eps, f1 = 1.0f, fo = out_scale, fq = (float)(0.0625 * 1.4426950408889634); /* log2(e) / sqrt(256) */
    uint32_t ue, u1, uo, uq; memcpy(&ue, &fe, 4); memcpy(&u1, &f1, 4); memcpy(&uo, &fo, 4); memcpy(&uq, &fq, 4);
    uint32_t ap[OMEGA_DATTN256_ARGS_WORDS], aa[OMEGA_PATTN256_ARGS_WORDS];
#define A64(a, i, x) do { (a)[i] = (uint32_t)(x); (a)[(i) + 1] = (uint32_t)((uint64_t)(x) >> 32); } while (0)
    A64(ap, 0, qkv_va); A64(ap, 2, v.k_va); A64(ap, 4, v.v_va); A64(ap, 6, v.rope_va); A64(ap, 8, v.q_norm_va); A64(ap, 10, v.k_norm_va);
    A64(ap, 12, g.qf.va); A64(ap, 14, v.part_va); A64(ap, 16, v.cnt_va);
    ap[18] = pos0; ap[19] = OMEGA_DATTN256_CHUNK; ap[20] = 1u; ap[21] = kvs; ap[22] = ue; ap[23] = u1; ap[24] = uq;
    A64(aa, 0, g.qf.va); A64(aa, 2, v.k_va); A64(aa, 4, v.v_va); A64(aa, 6, out_va);
    aa[8] = pos0; aa[9] = m; aa[10] = kvs; aa[11] = uo;
#undef A64
    OmegaGpuLaunch L[KK_N] = {
        { .code_va = ks[KK_PREP]->code.va, .gpr_count = omega_bw_gpr_declared(ks[KK_PREP]->kernel.gpr_count),
          .threads_x = OMEGA_PATTN256_THREADS, .threads_y = 1, .grid_x = m, .grid_y = OMEGA_DATTN256_KV_HEADS,
          .num_elements = (size_t)m * QKVW, .args = ap, .n_args = OMEGA_DATTN256_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
          .shared_bytes = OMEGA_DATTN256_SHARED_BYTES },
        { .code_va = ks[KK_ATTEND]->code.va, .gpr_count = omega_bw_gpr_declared(ks[KK_ATTEND]->kernel.gpr_count),
          .threads_x = OMEGA_PATTN256_THREADS, .threads_y = 1, .grid_x = gx_att, .grid_y = OMEGA_DATTN256_KV_HEADS,
          .num_elements = (size_t)m * OUTW, .args = aa, .n_args = OMEGA_PATTN256_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
          .shared_bytes = OMEGA_PATTN256_SHARED_BYTES },
    };
    if (info) {
        info->layer = layer; info->pos0 = pos0; info->m = m; info->grid_x[0] = m; info->grid_x[1] = gx_att;
        for (unsigned k = 0; k < KK_N; k++) { info->insns[k] = ks[k]->insns; info->gprs[k] = ks[k]->kernel.gpr_count; info->kernel_cache_hits += hit[k]; }
    }
    for (unsigned k = 0; k < KK_N && rc == OMEGA_GPU_DATTN_OK; k++) {
        if (!omega_bw_gpr_declared_ok(L[k].gpr_count, ks[k]->kernel.gpr_count)) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); rc = OMEGA_GPU_DATTN_CHIP_FAIL; }
        else if (rec) { if (omega_gpu_cmdlist_add(rec, &L[k]) < 0) rc = OMEGA_GPU_DATTN_CHIP_FAIL; }
        else if (omega_gpu_session_launch(&L[k], info ? &info->elapsed_ns[k] : NULL, NULL) != 0) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_DATTN_OK && host) {
        if (omega_gpu_session_sync_to_cpu(&g.out) != 0) rc = OMEGA_GPU_DATTN_CHIP_FAIL;
        else {
            const float *od = g.out.cpu;
            size_t unchanged = 0;
            for (size_t i = 0; i < (size_t)m * OUTW; i++) if (od[i] == POISON_F32) unchanged++;
            if (unchanged) { omega_gpu_session_set_error("prefill attn256: output poison survived"); rc = OMEGA_GPU_DATTN_CHIP_FAIL; }
            else memcpy(out, od, (size_t)m * OUTW * 4u);
        }
    }
    UNLOCK();
#undef UNLOCK
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}

int omega_gpu_prefill_attn256(OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos0, uint32_t m,
                              const float *qkv, float eps, float out_scale, float *out, OmegaGpuPrefillAttn256Info *info) {
    if (!qkv) return OMEGA_GPU_DATTN_BAD_ARGS;
    return core(c, layer, pos0, m, qkv, out, 0, 0, eps, out_scale, info, NULL);
}
int omega_gpu_prefill_attn256_dev(OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos0, uint32_t m,
                                  uint64_t qkv_va, float eps, float out_scale, uint64_t out_va, OmegaGpuPrefillAttn256Info *info) {
    return core(c, layer, pos0, m, NULL, NULL, qkv_va, out_va, eps, out_scale, info, NULL);
}
int omega_gpu_prefill_attn256_record(OmegaGpuCmdList *cl, OmegaGpuDecodeAttn256Cache *c, uint32_t layer, uint32_t pos0,
                                     uint32_t m, uint64_t qkv_va, float eps, float out_scale, uint64_t out_va) {
    if (!cl) return OMEGA_GPU_DATTN_BAD_ARGS;
    return core(c, layer, pos0, m, NULL, NULL, qkv_va, out_va, eps, out_scale, NULL, cl);
}
