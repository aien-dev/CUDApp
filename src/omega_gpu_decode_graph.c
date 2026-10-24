/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * omega_gpu_decode_graph.c: one decode token as one command list (see the header). Records the four decode APIs'
 * *_record launches and the omega_blackwell_codegen_dec glue kernels into an omega_gpu_cmdlist. Buffers:
 * omega_gpu_session_alloc only (GPU-uncached, nvos.h:1117); the host touches x (before), token and logits (after).
 */
#include "omega_gpu_decode_graph.h"
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_codegen_dec.h"
#include "omega_gpu_session.h"
#include "omega_gpu_fmt.h"
#include <stdlib.h>
#include <string.h>

#define TIMEOUT_MS 20000ull
#define SCALE_2P64 18446744073709551616.0f
#define NEG_PAD (-1e30f)
#define TOKEN_POISON 0xffffffffu

/* glue kernels */
enum { K_NORM, K_ADD, K_CN, K_SW, K_AM0, K_AM1, K_SW_M1, K_CN_M2, K_COUNT };

typedef struct { OmegaGpuMem code; uint32_t gpr; } Kern;

/* the one buffer: byte offsets, every region 256-byte aligned (RES/SWIGLU load 16-byte vectors) */
typedef struct {
    size_t x, xq, qkv, attn, o, gu, act, dn, hidden, logits, am_v, am_i, token, norms, fnorm, tabs, pf_x, pf_y, bytes;
} Layout;

struct OmegaGpuDecodeGraph {
    OmegaGpuDecodeGraphConfig cfg;
    uint32_t am_ctas;                    /* stage-1 ARGMAX CTAs = ceil(vocab / 2048) */
    Layout lo;
    OmegaGpuMem buf;
    Kern k[K_COUNT];
    OmegaGpuCmdList *cl;
    struct { const OmegaGpuFp8Tensor *qkv, *o, *gu, *dn; int set; } layer[OMEGA_GPU_DGRAPH_MAX_LAYERS];
    int final_set;
    /* token state */
    int in_token, failed;
    int xq_layer;                        /* layer whose input-norm x' sits in xq, -1 none */
    int hidden_ready;
    uint64_t t_begin;
    OmegaGpuDecodeGraphStats acc;
    int flush_each, mutant;
    /* timeline (set_stamps): one kind per recorded launch and the GPU timestamps around every launch of the last token */
    int stamps;
    uint32_t n_kind, tagged_in_list, n_stamp;
    uint8_t kind[OMEGA_GPU_CMDLIST_MAX_LAUNCHES];
    uint64_t stamp[OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1];
};

static size_t al(size_t v) { return (v + 255u) & ~(size_t)255u; }
static uint64_t va_of(const OmegaGpuDecodeGraph *g, size_t off) { return g->buf.va + off; }
static void *cpu_of(const OmegaGpuDecodeGraph *g, size_t off) { return (uint8_t *)g->buf.cpu + off; }
static void put64(uint32_t *a, unsigned off, uint64_t v) { a[(off - 0x380) / 4] = (uint32_t)v; a[(off - 0x380) / 4 + 1] = (uint32_t)(v >> 32); }
static void putf(uint32_t *a, unsigned off, float f) { memcpy(&a[(off - 0x380) / 4], &f, 4); }
static int fail(OmegaGpuDecodeGraph *g, const char *what, int rc) {
    if (g) g->failed = 1;
    if (what) omega_gpu_session_set_error(what);
    return rc;
}

/* lock held */
static int kern_build(Kern *k, OmegaDecKind kind, uint32_t mode, int mutant) {
    OmegaDecSpec s; memset(&s, 0, sizeof s); s.kind = kind; s.mode = mode; s.mutant = mutant;
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) return OMEGA_GPU_DGRAPH_CODEGEN_FAIL;
    int rc = OMEGA_GPU_DGRAPH_OK;
    size_t need = 0;
    if (omega_blackwell_codegen_dec_prog(&s, prog) != 0 || (need = omega_bw_program_code_bytes(prog)) == 0) rc = OMEGA_GPU_DGRAPH_CODEGEN_FAIL;
    else if (omega_gpu_session_alloc_code(need, &k->code) != 0) rc = OMEGA_GPU_DGRAPH_CHIP_FAIL;
    else {
        OmegaBlackwellKernel kk;
        if (omega_blackwell_kernel_encode_into(prog, k->code.cpu, need, &kk) != 0 || kk.code_size != need) rc = OMEGA_GPU_DGRAPH_CODEGEN_FAIL;
        else if (omega_gpu_session_sync_to_gpu(&k->code) != 0) rc = OMEGA_GPU_DGRAPH_CHIP_FAIL;
        else { k->gpr = omega_bw_gpr_declared(kk.gpr_count); if (k->gpr < 64u) k->gpr = 64u; } /* as tests/gpu_cmdlist_test.c (chip PASS) */
        if (rc != OMEGA_GPU_DGRAPH_OK) { omega_gpu_session_free(&k->code); memset(k, 0, sizeof *k); }
    }
    omega_gpu_host_free(prog);
    return rc;
}

void omega_gpu_decode_graph_free(OmegaGpuDecodeGraph *g) {
    if (!g) return;
    if (g->cl) omega_gpu_cmdlist_free(g->cl);
    omega_gpu_session_lock();
    for (unsigned i = 0; i < K_COUNT; i++) if (g->k[i].code.cpu) omega_gpu_session_free(&g->k[i].code);
    if (g->buf.cpu) omega_gpu_session_free(&g->buf);
    omega_gpu_session_unlock();
    omega_gpu_host_free(g);
}

int omega_gpu_decode_graph_create(const OmegaGpuDecodeGraphConfig *cfg, OmegaGpuDecodeGraph **out) {
    if (out) *out = NULL;
    if (!cfg || !out || cfg->layers == 0 || cfg->layers > OMEGA_GPU_DGRAPH_MAX_LAYERS || !cfg->router || !cfg->attn || !cfg->lm_head ||
        cfg->vocab == 0 || cfg->vocab > OMEGA_DEC_ARGMAX_SPAN * OMEGA_DEC_ARGMAX_SPAN) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    uint32_t n = 0, k = 0;
    omega_gpu_bf16_tensor_shape(cfg->lm_head, &n, &k);
    if (n != cfg->vocab || k != OMEGA_GPU_DGRAPH_HIDDEN) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    OmegaGpuDecodeGraph *g = omega_gpu_host_calloc(1, sizeof *g);
    if (!g) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    g->cfg = *cfg;
    g->am_ctas = (cfg->vocab + OMEGA_DEC_ARGMAX_SPAN - 1) / OMEGA_DEC_ARGMAX_SPAN;
    g->xq_layer = -1;
    const size_t H = OMEGA_GPU_DGRAPH_HIDDEN, TK = OMEGA_GPU_DGRAPH_TOPK, I = OMEGA_GPU_DGRAPH_INTER;
    Layout *L = &g->lo; size_t o = 0;
#define REG(f, bytes) do { L->f = o; o = al(o + (bytes)); } while (0)
    REG(x, H * 4); REG(xq, H * 4); REG(qkv, OMEGA_GPU_DGRAPH_QKV * 4); REG(attn, OMEGA_GPU_DGRAPH_ATTN * 4); REG(o, H * 4);
    REG(gu, TK * 2 * I * 4); REG(act, TK * I * 4); REG(dn, TK * H * 4); REG(hidden, H * 4);
    REG(logits, (size_t)g->am_ctas * OMEGA_DEC_ARGMAX_SPAN * 4); REG(am_v, OMEGA_DEC_ARGMAX_SPAN * 4); REG(am_i, OMEGA_DEC_ARGMAX_SPAN * 4);
    REG(token, 16); REG(norms, (size_t)cfg->layers * H * 4); REG(fnorm, H * 4); REG(tabs, (size_t)cfg->layers * 2 * 16);
    REG(pf_x, OMEGA_GPU_DGRAPH_ATTN * 4); REG(pf_y, H * 4); /* prefetch launch (W-L2d): x read by nothing else, y read by nothing */
#undef REG
    L->bytes = o;
    int rc = OMEGA_GPU_DGRAPH_OK;
    if (omega_gpu_cmdlist_create(OMEGA_GPU_CMDLIST_MAX_LAUNCHES, &g->cl) != 0) { omega_gpu_decode_graph_free(g); return OMEGA_GPU_DGRAPH_CHIP_FAIL; }
    omega_gpu_session_lock();
    if (!omega_gpu_session_open() || omega_gpu_session_alloc(L->bytes, &g->buf) != 0) rc = OMEGA_GPU_DGRAPH_CHIP_FAIL;
    static const struct { OmegaDecKind kind; uint32_t mode; int mutant; } spec[K_COUNT] = {
        [K_NORM] = { OMEGA_DEC_RES, OMEGA_DEC_RES_NORM, 0 },
        [K_ADD] = { OMEGA_DEC_RES, OMEGA_DEC_RES_ADD, 0 },
        [K_CN] = { OMEGA_DEC_RES, OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM, 0 },
        [K_SW] = { OMEGA_DEC_SWIGLU, 0, 0 },
        [K_AM0] = { OMEGA_DEC_ARGMAX, 0, 0 },
        [K_AM1] = { OMEGA_DEC_ARGMAX, 1, 0 },
        [K_SW_M1] = { OMEGA_DEC_SWIGLU, 0, 1 },
        [K_CN_M2] = { OMEGA_DEC_RES, OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM, 2 },
    };
    for (unsigned i = 0; i < K_COUNT && rc == OMEGA_GPU_DGRAPH_OK; i++) rc = kern_build(&g->k[i], spec[i].kind, spec[i].mode, spec[i].mutant);
    if (rc == OMEGA_GPU_DGRAPH_OK) {
        memset(g->buf.cpu, 0, L->bytes);
        float *lg = cpu_of(g, L->logits), *amv = cpu_of(g, L->am_v);
        for (size_t i = cfg->vocab; i < (size_t)g->am_ctas * OMEGA_DEC_ARGMAX_SPAN; i++) lg[i] = NEG_PAD; /* lm_head writes [0, vocab) only */
        for (size_t i = 0; i < OMEGA_DEC_ARGMAX_SPAN; i++) amv[i] = NEG_PAD;                              /* stage 1 writes [0, ctas) only */
        if (omega_gpu_session_sync_to_gpu(&g->buf) != 0) rc = OMEGA_GPU_DGRAPH_CHIP_FAIL;
    }
    omega_gpu_session_unlock();
    if (rc != OMEGA_GPU_DGRAPH_OK) { omega_gpu_decode_graph_free(g); return rc; }
    *out = g;
    return OMEGA_GPU_DGRAPH_OK;
}

static int shape_is(const OmegaGpuFp8Tensor *t, uint32_t e, uint32_t n, uint32_t k) {
    uint32_t te = 0, tn = 0, tk = 0; omega_gpu_fp8_tensor_shape(t, &te, &tn, &tk);
    return t && te == e && tn == n && tk == k;
}

int omega_gpu_decode_graph_set_layer(OmegaGpuDecodeGraph *g, uint32_t layer, const OmegaGpuFp8Tensor *qkv, const OmegaGpuFp8Tensor *o,
                                     const OmegaGpuFp8Tensor *gate_up, const OmegaGpuFp8Tensor *down, const float *input_norm) {
    if (!g || layer >= g->cfg.layers || !input_norm || g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    const uint32_t H = OMEGA_GPU_DGRAPH_HIDDEN, I = OMEGA_GPU_DGRAPH_INTER;
    if (!shape_is(qkv, 1, OMEGA_GPU_DGRAPH_QKV, H) || !shape_is(o, 1, H, OMEGA_GPU_DGRAPH_ATTN) || !shape_is(gate_up, 128, 2 * I, H) || !shape_is(down, 128, H, I))
        return OMEGA_GPU_DGRAPH_BAD_ARGS;
    uint64_t t[4];
    if (omega_gpu_fp8_tensor_entry(qkv, 0, &t[0], &t[1]) != 0 || omega_gpu_fp8_tensor_entry(o, 0, &t[2], &t[3]) != 0) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    omega_gpu_session_lock();
    memcpy((uint8_t *)cpu_of(g, g->lo.tabs) + (size_t)layer * 32u, t, sizeof t);
    memcpy((float *)cpu_of(g, g->lo.norms) + (size_t)layer * H, input_norm, H * 4u);
    int rc = omega_gpu_session_sync_to_gpu(&g->buf) == 0 ? OMEGA_GPU_DGRAPH_OK : OMEGA_GPU_DGRAPH_CHIP_FAIL;
    omega_gpu_session_unlock();
    if (rc == OMEGA_GPU_DGRAPH_OK) { g->layer[layer].qkv = qkv; g->layer[layer].o = o; g->layer[layer].gu = gate_up; g->layer[layer].dn = down; g->layer[layer].set = 1; }
    return rc;
}

int omega_gpu_decode_graph_set_final_norm(OmegaGpuDecodeGraph *g, const float *final_norm) {
    if (!g || !final_norm || g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    omega_gpu_session_lock();
    memcpy(cpu_of(g, g->lo.fnorm), final_norm, OMEGA_GPU_DGRAPH_HIDDEN * 4u);
    int rc = omega_gpu_session_sync_to_gpu(&g->buf) == 0 ? OMEGA_GPU_DGRAPH_OK : OMEGA_GPU_DGRAPH_CHIP_FAIL;
    omega_gpu_session_unlock();
    if (rc == OMEGA_GPU_DGRAPH_OK) g->final_set = 1;
    return rc;
}

OmegaGpuCmdList *omega_gpu_decode_graph_list(OmegaGpuDecodeGraph *g) { return g ? g->cl : NULL; }
void omega_gpu_decode_graph_set_flush_each(OmegaGpuDecodeGraph *g, int on) { if (g) g->flush_each = on ? 1 : 0; }
void omega_gpu_decode_graph_test_set_mutant(OmegaGpuDecodeGraph *g, int mutant) { if (g) g->mutant = mutant; }

void omega_gpu_decode_graph_vas(const OmegaGpuDecodeGraph *g, OmegaGpuDecodeGraphVas *v) {
    if (!v) return;
    memset(v, 0, sizeof *v);
    if (!g) return;
    v->x = va_of(g, g->lo.x); v->xq = va_of(g, g->lo.xq); v->qkv = va_of(g, g->lo.qkv); v->attn = va_of(g, g->lo.attn); v->o = va_of(g, g->lo.o);
    v->gu = va_of(g, g->lo.gu); v->act = va_of(g, g->lo.act); v->dn = va_of(g, g->lo.dn); v->hidden = va_of(g, g->lo.hidden);
    v->logits = va_of(g, g->lo.logits); v->token = va_of(g, g->lo.token);
}

/* ---- the list ---- */

/* barrier between the launches of a token (W-L2c launch-cost study): omega_gpu_decode_graph_set_barrier, else env
 * AIEN_OMEGA_DGRAPH_BARRIER (0 wfi+inval_all = default, 2 wfi_only, 3 wfi+inval_data, 4 flush_data; 1 none is refused: the
 * token is a dependency chain). Read once. */
static int g_barrier = -1;
void omega_gpu_decode_graph_set_barrier(int b) { g_barrier = (b >= 0 && b <= 4 && b != 1) ? b : 0; }
int omega_gpu_decode_graph_barrier(void) {
    if (g_barrier < 0) { const char *e = getenv("AIEN_OMEGA_DGRAPH_BARRIER"); omega_gpu_decode_graph_set_barrier(e ? atoi(e) : 0); }
    return g_barrier;
}
void omega_gpu_decode_graph_test_set_barrier_none(void) { g_barrier = 1; }
/* weight prefetch (W-L2d): 1 = beside each layer's attention, an independent launch (no barrier before it) runs the o_proj GEMV
 * on a scratch x into a scratch y, so the o_proj weights stream from DRAM while the latency-bound attention runs and the real
 * o_proj then reads them from L2 (needs GPU-cached weights: an uncached mapping is never kept in L2, gpu_l2_reuse_test r2).
 * omega_gpu_decode_graph_set_prefetch, else env AIEN_OMEGA_DGRAPH_PREFETCH ("1"). Read once. Default 0. */
static int g_prefetch = -1, g_prefetch_red;
void omega_gpu_decode_graph_set_prefetch(int on) { g_prefetch = on ? 1 : 0; }
int omega_gpu_decode_graph_prefetch(void) {
    if (g_prefetch < 0) { const char *e = getenv("AIEN_OMEGA_DGRAPH_PREFETCH"); g_prefetch = (e && e[0] == '1' && e[1] == 0) ? 1 : 0; }
    return g_prefetch;
}
void omega_gpu_decode_graph_test_set_prefetch_red(int on) { g_prefetch_red = on ? 1 : 0; }
static OmegaGpuCmdListBarrier barrier_now(void) { return (OmegaGpuCmdListBarrier)omega_gpu_decode_graph_barrier(); }

static OmegaGpuCmdListOpts list_opts(const OmegaGpuDecodeGraph *g) {
    OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o);
    o.mode = OMEGA_GPU_CMDLIST_PCAS; o.barrier = barrier_now(); o.timeout_ms = TIMEOUT_MS; o.stamps = g->stamps;
    return o;
}

static int submit_now(OmegaGpuDecodeGraph *g) {
    if (omega_gpu_cmdlist_count(g->cl) == 0) return OMEGA_GPU_DGRAPH_OK;
    OmegaGpuCmdListOpts o = list_opts(g);
    OmegaGpuCmdListStats st; memset(&st, 0, sizeof st);
    int rc = omega_gpu_cmdlist_submit(g->cl, &o, &st);
    g->acc.launches += st.launches; g->acc.submits += st.submits; g->acc.chip_ns += st.elapsed_ns; g->acc.sems_done += st.sems_done;
    if (rc == 0 && g->stamps) { /* before reset: the accessor sizes by the list; later submits (flush_each) drop their start slot */
        uint64_t tmp[OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1];
        int cnt = omega_gpu_cmdlist_stamps(g->cl, tmp, OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1);
        if (cnt < 0) rc = -1;
        else { const int from = g->n_stamp == 0 ? 0 : 1; for (int i = from; i < cnt; i++) g->stamp[g->n_stamp++] = tmp[i]; }
    }
    g->tagged_in_list = 0;
    omega_gpu_cmdlist_reset(g->cl);
    if (rc != 0) return fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    if (st.sems_done != st.launches) {
        char e[128]; omega_gpu_fmt(e, sizeof e, "decode graph: %u of %u launch semaphores DONE (first missing %d)", st.sems_done, st.launches, st.first_missing);
        return fail(g, e, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    }
    return OMEGA_GPU_DGRAPH_OK;
}

/* kinds for the launches added since the last tag (the timeline) */
static void tag(OmegaGpuDecodeGraph *g, uint8_t kind) {
    const uint32_t n = omega_gpu_cmdlist_count(g->cl);
    for (; g->tagged_in_list < n && g->n_kind < OMEGA_GPU_CMDLIST_MAX_LAUNCHES; g->tagged_in_list++) g->kind[g->n_kind++] = kind;
}

/* after every recorded launch: flush_each submits it on its own (the comparison path) */
static int after_op(OmegaGpuDecodeGraph *g, uint8_t kind, int api_rc) {
    if (api_rc != 0) return fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    tag(g, kind);
    return g->flush_each ? submit_now(g) : OMEGA_GPU_DGRAPH_OK;
}

static int add_glue(OmegaGpuDecodeGraph *g, uint8_t kind, const Kern *k, uint32_t gx, uint32_t gy, size_t elems, const uint32_t *args, uint32_t n_args) {
    OmegaGpuLaunch L; memset(&L, 0, sizeof L);
    L.code_va = k->code.va; L.gpr_count = k->gpr; L.threads_x = OMEGA_DEC_THREADS; L.threads_y = 1; L.grid_x = gx; L.grid_y = gy;
    L.num_elements = elems; L.args = args; L.n_args = n_args; L.timeout_ms = TIMEOUT_MS;
    return after_op(g, kind, omega_gpu_cmdlist_add(g->cl, &L) < 0 ? -1 : 0);
}

static int res_kernel(const OmegaGpuDecodeGraph *g, uint32_t mode) {
    if (mode == OMEGA_DEC_RES_NORM) return K_NORM;
    if (mode == OMEGA_DEC_RES_ADD) return K_ADD;
    if (mode == (OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM)) return g->mutant == 2 ? K_CN_M2 : K_CN;
    return -1;
}

int omega_gpu_decode_graph_record_res(OmegaGpuDecodeGraph *g, uint32_t mode, uint64_t x_va, uint64_t a_va, uint64_t w_va,
                                      uint64_t nw_va, uint64_t y_va, float eps, float scale) {
    if (!g || !g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    int ki = res_kernel(g, mode);
    if (ki < 0) return fail(g, "decode graph: RES mode not built (NORM, ADD, COMBINE|NORM)", OMEGA_GPU_DGRAPH_BAD_ARGS);
    uint32_t a[OMEGA_DEC_RES_WORDS]; memset(a, 0, sizeof a);
    put64(a, OMEGA_DEC_RES_X, x_va); put64(a, OMEGA_DEC_RES_A, a_va); put64(a, OMEGA_DEC_RES_W, w_va); put64(a, OMEGA_DEC_RES_NW, nw_va);
    put64(a, OMEGA_DEC_RES_Y, y_va); putf(a, OMEGA_DEC_RES_EPS, eps); putf(a, OMEGA_DEC_RES_SCALE, scale);
    const uint8_t kind = ki == K_ADD ? OMEGA_GPU_DGRAPH_K_ADD : ki == K_NORM ? OMEGA_GPU_DGRAPH_K_NORM : OMEGA_GPU_DGRAPH_K_COMBINE;
    return add_glue(g, kind, &g->k[ki], 1, 1, OMEGA_GPU_DGRAPH_HIDDEN, a, OMEGA_DEC_RES_WORDS);
}

int omega_gpu_decode_graph_record_swiglu(OmegaGpuDecodeGraph *g, uint64_t gu_va, uint64_t act_va, float scale) {
    if (!g || !g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    uint32_t a[OMEGA_DEC_SW_WORDS]; memset(a, 0, sizeof a);
    put64(a, OMEGA_DEC_SW_GU, gu_va); put64(a, OMEGA_DEC_SW_ACT, act_va); putf(a, OMEGA_DEC_SW_SCALE, scale);
    return add_glue(g, OMEGA_GPU_DGRAPH_K_SWIGLU, &g->k[g->mutant == 1 ? K_SW_M1 : K_SW], OMEGA_DEC_INTER / OMEGA_DEC_THREADS, OMEGA_DEC_TOPK, (size_t)OMEGA_DEC_TOPK * OMEGA_DEC_INTER, a, OMEGA_DEC_SW_WORDS);
}

int omega_gpu_decode_graph_begin_token(OmegaGpuDecodeGraph *g, const float *x_embed) {
    if (!g || g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    omega_gpu_session_lock();
    g->t_begin = omega_gpu_session_now_ns();
    g->in_token = 1; g->failed = 0; g->xq_layer = -1; g->hidden_ready = 0;
    memset(&g->acc, 0, sizeof g->acc);
    g->n_kind = 0; g->tagged_in_list = 0; g->n_stamp = 0;
    omega_gpu_cmdlist_reset(g->cl);
    if (!omega_gpu_session_open()) return fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    uint32_t *tok = cpu_of(g, g->lo.token);
    tok[0] = TOKEN_POISON; tok[1] = TOKEN_POISON;
    if (x_embed) memcpy(cpu_of(g, g->lo.x), x_embed, OMEGA_GPU_DGRAPH_HIDDEN * 4u);
    if (omega_gpu_session_sync_to_gpu(&g->buf) != 0) return fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    return OMEGA_GPU_DGRAPH_OK;
}

int omega_gpu_decode_graph_record_layer(OmegaGpuDecodeGraph *g, uint32_t l, uint32_t pos) {
    if (!g || !g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (g->failed) return OMEGA_GPU_DGRAPH_CHIP_FAIL;
    if (l >= g->cfg.layers || !g->layer[l].set || g->hidden_ready || (l + 1 == g->cfg.layers && !g->final_set))
        return fail(g, "decode graph: layer not set, out of order, or final norm missing", OMEGA_GPU_DGRAPH_BAD_ARGS);
    OmegaGpuDecodeGraphVas v; omega_gpu_decode_graph_vas(g, &v);
    const float eps = g->cfg.eps;
    const uint64_t norm_l = va_of(g, g->lo.norms) + (uint64_t)l * OMEGA_GPU_DGRAPH_HIDDEN * 4u, tab = va_of(g, g->lo.tabs) + (uint64_t)l * 32u;
    int rc = OMEGA_GPU_DGRAPH_OK;
    if (g->xq_layer != (int)l) rc = omega_gpu_decode_graph_record_res(g, OMEGA_DEC_RES_NORM, v.x, 0, 0, norm_l, v.xq, eps, SCALE_2P64);
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = after_op(g, OMEGA_GPU_DGRAPH_K_QKV, omega_gpu_fp8_gemv_grouped_record(g->cl, g->layer[l].qkv, 1, tab, v.xq, 1, v.qkv));
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = after_op(g, OMEGA_GPU_DGRAPH_K_ATTN, omega_gpu_decode_attn_record(g->cl, g->cfg.attn, l, pos, v.qkv, eps, g->mutant == 3 ? 1.0f : SCALE_2P64, v.attn));
    if (rc == OMEGA_GPU_DGRAPH_OK && omega_gpu_decode_graph_prefetch()) { /* tagged attn: its time is the overlapped window */
        int prc = omega_gpu_fp8_gemv_grouped_record(g->cl, g->layer[l].o, 1, tab + 16u, va_of(g, g->lo.pf_x), 1, va_of(g, g->lo.pf_y));
        if (prc == 0 && omega_gpu_cmdlist_mark_independent(g->cl) != 0) prc = -1;
        rc = after_op(g, OMEGA_GPU_DGRAPH_K_ATTN, prc);
    }
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = after_op(g, OMEGA_GPU_DGRAPH_K_O, omega_gpu_fp8_gemv_grouped_record(g->cl, g->layer[l].o, 1, tab + 16u, v.attn, 1, v.o));
    if (rc == OMEGA_GPU_DGRAPH_OK && g_prefetch_red && !g->flush_each && omega_gpu_cmdlist_mark_independent(g->cl) != 0) rc = OMEGA_GPU_DGRAPH_CHIP_FAIL; /* TEST ONLY red: o_proj itself unordered after the attention */
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = omega_gpu_decode_graph_record_res(g, OMEGA_DEC_RES_ADD, v.x, v.o, 0, 0, 0, eps, 1.0f);
    OmegaGpuRouterDevOut r; memset(&r, 0, sizeof r);
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = after_op(g, OMEGA_GPU_DGRAPH_K_ROUTER, omega_gpu_router_record(g->cl, g->cfg.router, l, v.x, eps, g->cfg.norm_topk, SCALE_2P64, &r));
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = after_op(g, OMEGA_GPU_DGRAPH_K_GATE_UP, omega_gpu_fp8_gemv_grouped_record(g->cl, g->layer[l].gu, OMEGA_GPU_DGRAPH_TOPK, r.tab_gu_va, r.xout_va, 1, v.gu));
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = omega_gpu_decode_graph_record_swiglu(g, v.gu, v.act, SCALE_2P64);
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = after_op(g, OMEGA_GPU_DGRAPH_K_DOWN, omega_gpu_fp8_gemv_grouped_record(g->cl, g->layer[l].dn, OMEGA_GPU_DGRAPH_TOPK, r.tab_dn_va, v.act, OMEGA_GPU_DGRAPH_TOPK, v.dn));
    if (rc == OMEGA_GPU_DGRAPH_OK) {
        const int last = l + 1 == g->cfg.layers;
        const uint64_t nw = last ? va_of(g, g->lo.fnorm) : norm_l + OMEGA_GPU_DGRAPH_HIDDEN * 4u;
        rc = omega_gpu_decode_graph_record_res(g, OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM, v.x, v.dn, r.wts_va, nw, last ? v.hidden : v.xq, eps, last ? 1.0f : SCALE_2P64);
        if (rc == OMEGA_GPU_DGRAPH_OK) { if (last) g->hidden_ready = 1; else g->xq_layer = (int)l + 1; }
    }
    return rc == OMEGA_GPU_DGRAPH_OK ? rc : fail(g, NULL, rc);
}

int omega_gpu_decode_graph_record_final(OmegaGpuDecodeGraph *g) {
    if (!g || !g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (g->failed) return OMEGA_GPU_DGRAPH_CHIP_FAIL;
    if (!g->final_set) return fail(g, "decode graph: final norm not set", OMEGA_GPU_DGRAPH_BAD_ARGS);
    OmegaGpuDecodeGraphVas v; omega_gpu_decode_graph_vas(g, &v);
    int rc = OMEGA_GPU_DGRAPH_OK;
    if (!g->hidden_ready) rc = omega_gpu_decode_graph_record_res(g, OMEGA_DEC_RES_NORM, v.x, 0, 0, va_of(g, g->lo.fnorm), v.hidden, g->cfg.eps, 1.0f);
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = after_op(g, OMEGA_GPU_DGRAPH_K_LM_HEAD, omega_gpu_bf16_gemv_record(g->cl, g->cfg.lm_head, v.hidden, v.logits));
    uint32_t a0[OMEGA_DEC_AM_WORDS], a1[OMEGA_DEC_AM_WORDS]; memset(a0, 0, sizeof a0); memset(a1, 0, sizeof a1);
    const uint64_t amv = va_of(g, g->lo.am_v), ami = va_of(g, g->lo.am_i);
    put64(a0, OMEGA_DEC_AM_V, v.logits); put64(a0, OMEGA_DEC_AM_I, 0); put64(a0, OMEGA_DEC_AM_OV, amv); put64(a0, OMEGA_DEC_AM_OI, ami);
    put64(a1, OMEGA_DEC_AM_V, amv); put64(a1, OMEGA_DEC_AM_I, ami); put64(a1, OMEGA_DEC_AM_OV, v.token + 4u); put64(a1, OMEGA_DEC_AM_OI, v.token);
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = add_glue(g, OMEGA_GPU_DGRAPH_K_ARGMAX, &g->k[K_AM0], g->am_ctas, 1, (size_t)g->am_ctas * OMEGA_DEC_ARGMAX_SPAN, a0, OMEGA_DEC_AM_WORDS);
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = add_glue(g, OMEGA_GPU_DGRAPH_K_ARGMAX, &g->k[K_AM1], 1, 1, OMEGA_DEC_ARGMAX_SPAN, a1, OMEGA_DEC_AM_WORDS);
    return rc == OMEGA_GPU_DGRAPH_OK ? rc : fail(g, NULL, rc);
}

int omega_gpu_decode_graph_end_token(OmegaGpuDecodeGraph *g, uint32_t *token, OmegaGpuDecodeGraphStats *st) {
    if (!g || !g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    int rc = g->failed ? OMEGA_GPU_DGRAPH_CHIP_FAIL : OMEGA_GPU_DGRAPH_OK;
    const uint64_t t_rec = omega_gpu_session_now_ns();
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = submit_now(g);
    if (rc == OMEGA_GPU_DGRAPH_OK && omega_gpu_decode_attn_counters_check(g->cfg.attn) != 0) rc = fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    if (rc == OMEGA_GPU_DGRAPH_OK && omega_gpu_session_sync_to_cpu(&g->buf) != 0) rc = fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    if (rc == OMEGA_GPU_DGRAPH_OK) {
        const uint32_t t = ((const uint32_t *)cpu_of(g, g->lo.token))[0];
        if (t == TOKEN_POISON || t >= g->cfg.vocab) {
            char e[96]; omega_gpu_fmt(e, sizeof e, "decode graph: token output %u not written or out of range", t);
            rc = fail(g, e, OMEGA_GPU_DGRAPH_CHIP_FAIL);
        } else if (token) *token = t;
    }
    omega_gpu_cmdlist_reset(g->cl);
    g->in_token = 0;
    const uint64_t t_end = omega_gpu_session_now_ns();
    if (st) { *st = g->acc; st->record_ns = (g->flush_each ? t_end : t_rec) - g->t_begin; st->total_ns = t_end - g->t_begin; }
    omega_gpu_session_unlock();
    return rc;
}

int omega_gpu_decode_graph_token(OmegaGpuDecodeGraph *g, const float *x_embed, uint32_t pos, uint32_t *token, OmegaGpuDecodeGraphStats *st) {
    if (!g) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (omega_gpu_decode_graph_begin_token(g, x_embed) == OMEGA_GPU_DGRAPH_BAD_ARGS) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    for (uint32_t l = 0; l < g->cfg.layers && !g->failed; l++) (void)omega_gpu_decode_graph_record_layer(g, l, pos);
    if (!g->failed) (void)omega_gpu_decode_graph_record_final(g);
    return omega_gpu_decode_graph_end_token(g, token, st);
}

int omega_gpu_decode_graph_read(OmegaGpuDecodeGraph *g, float *x, float *hidden, float *logits) {
    if (!g || g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = omega_gpu_session_sync_to_cpu(&g->buf) == 0 ? OMEGA_GPU_DGRAPH_OK : OMEGA_GPU_DGRAPH_CHIP_FAIL;
    if (rc == OMEGA_GPU_DGRAPH_OK) {
        if (x) memcpy(x, cpu_of(g, g->lo.x), OMEGA_GPU_DGRAPH_HIDDEN * 4u);
        if (hidden) memcpy(hidden, cpu_of(g, g->lo.hidden), OMEGA_GPU_DGRAPH_HIDDEN * 4u);
        if (logits) memcpy(logits, cpu_of(g, g->lo.logits), (size_t)g->cfg.vocab * 4u);
    }
    omega_gpu_session_unlock();
    return rc;
}

/* ---- timeline ---- */
void omega_gpu_decode_graph_set_stamps(OmegaGpuDecodeGraph *g, int on) { if (g && !g->in_token) g->stamps = on ? 1 : 0; }

int omega_gpu_decode_graph_stamps(const OmegaGpuDecodeGraph *g, uint64_t *stamps, uint8_t *kinds, uint32_t cap) {
    if (!g || g->in_token || !stamps || !kinds) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (!g->stamps || g->n_stamp != g->n_kind + 1u) { omega_gpu_session_set_error("decode graph: no timeline for the last token (stamps off or it failed)"); return OMEGA_GPU_DGRAPH_BAD_ARGS; }
    if (cap < g->n_stamp) { omega_gpu_session_set_error("decode graph: timeline buffers too small"); return OMEGA_GPU_DGRAPH_BAD_ARGS; }
    memcpy(stamps, g->stamp, (size_t)g->n_stamp * sizeof stamps[0]);
    memcpy(kinds, g->kind, (size_t)g->n_kind);
    return (int)g->n_kind;
}

const char *omega_gpu_decode_graph_kind_name(uint8_t kind) {
    static const char *const names[OMEGA_GPU_DGRAPH_K_COUNT] = { "norm", "qkv", "attn", "o_proj", "res_add", "router", "gate_up", "swiglu", "down", "combine_norm", "lm_head", "argmax" };
    return kind < OMEGA_GPU_DGRAPH_K_COUNT ? names[kind] : "?";
}
