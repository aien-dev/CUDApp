/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * omega_gpu_gdnc_api.c (DRA-84): the chunked GDN prefill scan as a recorded launch (see the header). The kernel cache is
 * its own (the shared prefill cache omega_gpu_pf_kernel_locked caps declared registers at 128; gdnc peaks at 228) and
 * mirrors kernel_for in omega_gpu_gdn_api.c: one resident program per (mutant, ratio), built on prepare, a cache miss
 * that would evict refused while recording.
 */
#include "omega_gpu_gdnc_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_codegen_tc.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define MARKER_TIMEOUT_MS 20000ull

static struct {
    bool hooked, valid;
    OmegaGdncSpec spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
    int mutant;
    uint64_t records;
} g_c;
static int g_crec; /* 1 while a record call runs (lock held): a cache miss must not evict */

static void on_session_close(void) { g_c.valid = false; memset(&g_c.code, 0, sizeof g_c.code); }
static OmegaGpuSession *dev_open_locked(void) {
    if (!g_c.hooked) { (void)omega_gpu_session_on_close(on_session_close); g_c.hooked = true; }
    return omega_gpu_session_open();
}
static void cache_clear_locked(void) {
    if (g_c.valid) omega_gpu_session_free(&g_c.code);
    g_c.valid = false; memset(&g_c.code, 0, sizeof g_c.code);
}

void omega_gpu_gdnc_test_set_mutant(int mutant) {
    omega_gpu_session_lock(); g_c.mutant = mutant; cache_clear_locked(); omega_gpu_session_unlock();
}
uint64_t omega_gpu_gdnc_records(void) { return g_c.records; }

/* ---- host planning ---- */
uint32_t omega_gdnc_snap_count(uint32_t tokens, uint32_t every_chunks, uint32_t cap) {
    if (!every_chunks || !cap) return 0;
    const uint64_t span = (uint64_t)OMEGA_GDNC_SNAP_CHUNK * every_chunks;
    const uint64_t n = tokens / span;
    return n < cap ? (uint32_t)n : cap;
}

static int shape_of(uint32_t ratio, OmegaGdnShape *s) {
    OmegaGdnSpec sp; memset(&sp, 0, sizeof sp); sp.ratio = ratio;
    return omega_gdn_shape(&sp, s);
}

int omega_gdnc_plan(const OmegaGdncReq *r, OmegaGdncLaunch *L) {
    if (!r || !L) return OMEGA_GPU_GDN_BAD_ARGS;
    OmegaGdnShape sh;
    if (shape_of(r->ratio, &sh) != 0) return OMEGA_GPU_GDN_BAD_ARGS;
    const OmegaGdncRows *io = &r->io;
    if (io->tokens == 0 || !io->qkvz_va || !io->ab_va || !io->out_va || !r->par_va || !r->tail_va || !r->state_va) return OMEGA_GPU_GDN_BAD_ARGS;
    if (!isfinite(io->eps_l2) || !isfinite(io->eps_norm) || !isfinite(io->in_scale) || !isfinite(io->out_scale)) return OMEGA_GPU_GDN_BAD_ARGS;
    if (((io->qkvz_va | io->out_va | io->xstride | io->ostride) & 15u) || ((io->ab_va | io->abstride) & 7u)) return OMEGA_GPU_GDN_BAD_ARGS;
    /* a row must hold what the kernel reads / writes (bf16 qkv|z, f32 a|b, bf16 out), unless every token shares row 0 */
    if (io->tokens > 1 && (io->xstride < (sh.conv + sh.vdim) * 2u || io->ostride < sh.vdim * 2u
                           || (io->abstride && io->abstride < sh.vh * 8u))) return OMEGA_GPU_GDN_BAD_ARGS;
    if (((r->par_va | r->tail_va | r->state_va) & 15u)) return OMEGA_GPU_GDN_BAD_ARGS;
    if (r->snap_count) {
        if (!r->snap_va || (r->snap_va & 15u) || r->snap_every4 == 0 || (r->snap_stride & 15u)
            || r->snap_stride < sh.state_bytes + sh.tail_bytes) return OMEGA_GPU_GDN_BAD_ARGS;
    }
    memset(L, 0, sizeof *L);
    L->threads = OMEGA_GDNC_THREADS; L->grid_x = sh.r; L->grid_y = OMEGA_GDN_KEY_HEADS;
    L->shared_bytes = OMEGA_GDNC_SHARED_BYTES; L->n_args = OMEGA_GDNC_ARGS_WORDS;
    uint32_t *a = L->args;
#define W(off) a[((off) - OMEGA_GDNC_ARG_QKVZ) / 4u]
#define A64(off, v) do { W(off) = (uint32_t)(v); a[((off) - OMEGA_GDNC_ARG_QKVZ) / 4u + 1u] = (uint32_t)((uint64_t)(v) >> 32); } while (0)
    A64(OMEGA_GDNC_ARG_QKVZ, io->qkvz_va); A64(OMEGA_GDNC_ARG_AB, io->ab_va);
    A64(OMEGA_GDNC_ARG_PAR, r->par_va); A64(OMEGA_GDNC_ARG_TAIL, r->tail_va); A64(OMEGA_GDNC_ARG_STATE, r->state_va);
    A64(OMEGA_GDNC_ARG_OUT, io->out_va); A64(OMEGA_GDNC_ARG_SNAP, r->snap_count ? r->snap_va : 0u);
    memcpy(&W(OMEGA_GDNC_ARG_EPS_L2), &io->eps_l2, 4); memcpy(&W(OMEGA_GDNC_ARG_EPS_NORM), &io->eps_norm, 4);
    memcpy(&W(OMEGA_GDNC_ARG_IN_SCALE), &io->in_scale, 4); memcpy(&W(OMEGA_GDNC_ARG_OUT_SCALE), &io->out_scale, 4);
    W(OMEGA_GDNC_ARG_T) = io->tokens; W(OMEGA_GDNC_ARG_XSTRIDE) = io->xstride;
    W(OMEGA_GDNC_ARG_ABSTRIDE) = io->abstride; W(OMEGA_GDNC_ARG_OSTRIDE) = io->ostride;
    W(OMEGA_GDNC_ARG_SNAP_EVERY4) = r->snap_count ? r->snap_every4 : 0u;
    W(OMEGA_GDNC_ARG_SNAP_COUNT) = r->snap_count;
    W(OMEGA_GDNC_ARG_SNAP_STRIDE) = r->snap_count ? r->snap_stride : 0u;
#undef A64
#undef W
    return OMEGA_GPU_GDN_OK;
}

int omega_gdnc_req(const OmegaGpuGdnStateView *v, uint32_t layer, const OmegaGdncRows *io, const OmegaGdncSnapOut *snap,
                   OmegaGdncReq *r) {
    if (!v || !io || !r || layer >= v->layers || v->state_bf16) return OMEGA_GPU_GDN_BAD_ARGS;
    memset(r, 0, sizeof *r);
    r->ratio = v->ratio; r->io = *io;
    r->par_va = v->par_va + (uint64_t)layer * v->par_bytes;
    r->tail_va = v->tail_va + (uint64_t)layer * v->tail_bytes;
    r->state_va = v->state_va + (uint64_t)layer * v->state_bytes;
    if (snap && snap->va) {
        if (snap->every_chunks == 0 || snap->every_chunks > UINT32_MAX / 4u || (snap->layer_stride_bytes & 15u)) return OMEGA_GPU_GDN_BAD_ARGS;
        const uint32_t n = omega_gdnc_snap_count(io->tokens, snap->every_chunks, snap->count);
        if (v->layers > 1u && snap->layer_stride_bytes < (uint64_t)snap->count * snap->snap_stride_bytes)
            return OMEGA_GPU_GDN_BAD_ARGS; /* layer l's slots would overlap layer l + 1's */
        r->snap_count = n;
        r->snap_va = snap->va + (uint64_t)layer * snap->layer_stride_bytes;
        r->snap_every4 = snap->every_chunks * (OMEGA_GDNC_SNAP_CHUNK / OMEGA_GDNC_SUB);
        r->snap_stride = snap->snap_stride_bytes;
    }
    return OMEGA_GPU_GDN_OK;
}

/* ---- device ---- */
static int kernel_locked(const OmegaGdncSpec *spec) {
    if (g_c.valid && memcmp(&g_c.spec, spec, sizeof *spec) == 0) return OMEGA_GPU_GDN_OK;
    if (g_crec && g_c.valid) { omega_gpu_session_set_error("gdnc record: kernel cache miss would evict the resident kernel (prepare before recording)"); return OMEGA_GPU_GDN_TOO_LARGE; }
    cache_clear_locked();
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("gdnc codegen: host memory"); return OMEGA_GPU_GDN_CODEGEN_FAIL; }
    int cg = omega_blackwell_codegen_gdnc_prog(spec, prog);
    const size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("gdnc codegen refused the spec"); return OMEGA_GPU_GDN_CODEGEN_FAIL; }
    if ((uint64_t)OMEGA_GDNC_THREADS * omega_bw_gpr_declared(prog->regalloc.peak_gpr_usage) > 65536u) { omega_gpu_host_free(prog); omega_gpu_session_set_error("gdnc: register file"); return OMEGA_GPU_GDN_TOO_LARGE; }
    const uint32_t expect = omega_gdnc_placeholder_count(prog);
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_GDN_CHIP_FAIL; }
    OmegaBlackwellKernel kernel;
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    /* SHFL.BFLY and F2FP placeholders become the real words (omega_blackwell_codegen_tc.h); the count must match */
    const int np = cg == 0 ? omega_tc_patch_code(prog, code.cpu, need) : -1;
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need) { omega_gpu_session_free(&code); omega_gpu_session_set_error("gdnc encode"); return OMEGA_GPU_GDN_CODEGEN_FAIL; }
    if (np < 0 || (uint32_t)np != expect) { omega_gpu_session_free(&code); omega_gpu_session_set_error("gdnc: placeholder patch count mismatch"); return OMEGA_GPU_GDN_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_GDN_CHIP_FAIL; }
    g_c.valid = true; g_c.spec = *spec; g_c.kernel = kernel; g_c.code = code;
    return OMEGA_GPU_GDN_OK;
}

static int spec_of(const OmegaGpuGdn *g, OmegaGdncSpec *spec, OmegaGpuGdnStateView *v) {
    if (omega_gpu_gdn_state_view(g, v) != OMEGA_GPU_GDN_OK) return OMEGA_GPU_GDN_BAD_ARGS;
    if (v->state_bf16) { omega_gpu_session_set_error("gdnc: the chunked scan keeps an f32 state (AIEN_OMEGA_GDN_STATE_BF16=1 is refused)"); return OMEGA_GPU_GDN_BAD_ARGS; }
    memset(spec, 0, sizeof *spec);
    spec->mutant = g_c.mutant; spec->ratio = v->ratio;
    return OMEGA_GPU_GDN_OK;
}

int omega_gpu_gdnc_prepare(OmegaGpuGdn *g) {
    if (!g) return OMEGA_GPU_GDN_BAD_ARGS;
    OmegaGdncSpec spec; OmegaGpuGdnStateView v;
    omega_gpu_session_lock();
    int rc = spec_of(g, &spec, &v);
    if (rc == OMEGA_GPU_GDN_OK) rc = dev_open_locked() ? kernel_locked(&spec) : OMEGA_GPU_GDN_CHIP_FAIL;
    omega_gpu_session_unlock();
    return rc;
}

int omega_gpu_gdnc_record(OmegaGpuCmdList *cl, OmegaGpuGdn *g, uint32_t layer, const OmegaGdncRows *io,
                          const OmegaGdncSnapOut *snap, uint32_t *written) {
    if (written) *written = 0;
    if (!cl || !g || !io) return OMEGA_GPU_GDN_BAD_ARGS;
    OmegaGdncSpec spec; OmegaGpuGdnStateView v; OmegaGdncReq r; OmegaGdncLaunch P;
    int rc = spec_of(g, &spec, &v);
    if (rc == OMEGA_GPU_GDN_OK) rc = omega_gdnc_req(&v, layer, io, snap, &r);
    if (rc == OMEGA_GPU_GDN_OK) rc = omega_gdnc_plan(&r, &P);
    if (rc != OMEGA_GPU_GDN_OK) return rc;
    /* caller holds the session lock */
    if (!dev_open_locked()) return OMEGA_GPU_GDN_CHIP_FAIL;
    g_crec = 1;
    rc = kernel_locked(&spec);
    g_crec = 0;
    if (rc != OMEGA_GPU_GDN_OK) return rc;
    OmegaGpuLaunch L = {
        .code_va = g_c.code.va, .gpr_count = omega_bw_gpr_declared(g_c.kernel.gpr_count),
        .threads_x = P.threads, .threads_y = 1, .grid_x = P.grid_x, .grid_y = P.grid_y,
        .num_elements = (size_t)v.state_bytes / (OMEGA_GDN_HEAD_DIM * 4u), /* vdim: vh * 128 */
        .args = P.args, .n_args = P.n_args, .timeout_ms = MARKER_TIMEOUT_MS,
        .shared_bytes = P.shared_bytes,
    };
    if (!omega_bw_gpr_declared_ok(L.gpr_count, g_c.kernel.gpr_count)) { omega_gpu_session_set_error("gdnc: declared registers below peak + 2: refusing to launch"); return OMEGA_GPU_GDN_CHIP_FAIL; }
    if (omega_gpu_cmdlist_add(cl, &L) < 0) return OMEGA_GPU_GDN_CHIP_FAIL;
    g_c.records++;
    if (written) *written = r.snap_count;
    return OMEGA_GPU_GDN_OK;
}
