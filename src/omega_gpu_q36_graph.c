/*
 * omega_gpu_q36_graph.c: one Qwen3.6-35B-A3B decode token as one command list (see the header). Records the FP8 / bf16 GEMV,
 * GDN, hd256 attention and router APIs' *_record launches and the omega_blackwell_codegen_dec glue kernels at width 2048 into an
 * omega_gpu_cmdlist. Buffers: buf from omega_gpu_session_alloc (GPU-uncached, nvos.h:1117) holds every region the host
 * touches: x (before), token and logits (after), the tables; dbuf (GPU-cached, omega_gpu_mem_cached.h) holds the chip-only
 * intermediates (see Layout).
 */
#include "omega_gpu_q36_graph.h"
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_codegen_dec.h"
#include "omega_gpu_session.h"
#include "omega_gpu_fmt.h"
#include "omega_gpu_mem_cached.h"
#include <stdlib.h>
#include <string.h>

#define TIMEOUT_MS 20000ull
#define SCALE_2P64 18446744073709551616.0f
#define NEG_PAD (-1e30f)
#define TOKEN_POISON 0xffffffffu
#define TOKEN_STRIDE 256u /* bytes per head's token slot (two u32, 256-aligned) */
#define H OMEGA_GPU_Q36_HIDDEN
#define EI OMEGA_GPU_Q36_EXPERT_INTER
#define NG OMEGA_GPU_Q36_GROUPS

/* glue kernels */
enum { K_NORM, K_ADD, K_CN, K_SW, K_GATHER, K_GATE, K_AM0, K_AM1, K_SW_M1, K_GATHER_M1, K_GATE_M1, K_CN_M3, K_COPY, K_COUNT };
#define COPY_ROW 256u /* K_COPY: GATHER rows of 256 floats through the identity table */

typedef struct { OmegaGpuMem code; uint32_t gpr; } Kern;

/* byte offsets into two buffers, every region 256-byte aligned (RES loads 16-byte vectors). [0, split): buf, GPU-uncached,
 * every region the host writes or reads. [split, bytes): dbuf, the chip-only intermediates, GPU-cached unless
 * AIEN_OMEGA_Q36_BUF_CACHED=0 (measured 2026-10-19: every FP8 GEMV CTA reads the whole x' row, and uncached each of those
 * reads goes to DRAM; cached, down 2100 -> 1735 us and mix 1596 -> 1458 us over 4 layers, runs/probe-buf). read_buf copies
 * a dbuf region into stage with a chip launch (K_COPY) before the host reads it. */
typedef struct {
    size_t x, ab, hidden, logits, am_v, am_i, token, norms, fnorm, tabs, gtab, idtab, stage, split;
    size_t xq, mix, agg, att, o, gu, act, dn, bytes;
} Layout;

struct OmegaGpuQ36Graph {
    OmegaGpuQ36GraphConfig cfg;
    uint32_t am_ctas;                    /* stage-1 ARGMAX CTAs = ceil(vocab / 2048) */
    Layout lo;
    OmegaGpuMem buf, dbuf;               /* the host-visible regions; the chip-only regions (see Layout) */
    Kern k[K_COUNT];
    OmegaGpuCmdList *cl;
    struct { const OmegaGpuFp8Tensor *mix, *o, *gu, *dn; const OmegaGpuBf16Tensor *ab; uint32_t kind_index; int attn, set; } layer[OMEGA_GPU_Q36_MAX_LAYERS];
    int final_set;
    const OmegaGpuFp8Tensor *head8;      /* set_lm_head_fp8: FP8 lm_head (hidden is then written x 2^64, read() undoes it) */
    /* heads (strokes): slot h of x, hidden, logits and token; head 0's state objects are cfg.gdn / cfg.attn */
    uint32_t heads;                      /* slots laid out (create_heads); 1 = the plain graph */
    OmegaGpuGdn *gdn_h[OMEGA_GPU_Q36_MAX_HEADS];
    OmegaGpuDecodeAttn256Cache *attn_h[OMEGA_GPU_Q36_MAX_HEADS];
    /* token / stroke state */
    int in_token, failed;
    uint32_t stroke_heads;               /* heads in the open stroke */
    int xq_layer[OMEGA_GPU_Q36_MAX_HEADS];      /* per head: layer whose input-norm x' sits in the head's xq slot, -1 none
                                                 * (COMBINE|NORM of layer l writes layer l+1's x' there: the one value that
                                                 * crosses layers, so it is per head; 02:50Z, found by the stroke gate) */
    int hidden_ready[OMEGA_GPU_Q36_MAX_HEADS];
    uint64_t t_begin;
    OmegaGpuDecodeGraphStats acc;
    int flush_each, mutant;
    int stamps;
    uint32_t n_kind, tagged_in_list, n_stamp;
    uint8_t kind[OMEGA_GPU_CMDLIST_MAX_LAUNCHES];
    uint64_t stamp[OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1];
};

static size_t al(size_t v) { return (v + 255u) & ~(size_t)255u; }
static uint64_t va_of(const OmegaGpuQ36Graph *g, size_t off) { return off < g->lo.split ? g->buf.va + off : g->dbuf.va + (off - g->lo.split); }
static void *cpu_of(const OmegaGpuQ36Graph *g, size_t off) { return (uint8_t *)g->buf.cpu + off; } /* host regions only (off < split) */
static size_t x_of(const OmegaGpuQ36Graph *g, uint32_t h) { return g->lo.x + (size_t)h * H * 4u; }
static size_t hidden_of(const OmegaGpuQ36Graph *g, uint32_t h) { return g->lo.hidden + (size_t)h * H * 4u; }
static size_t xq_of(const OmegaGpuQ36Graph *g, uint32_t h) { return g->lo.xq + (size_t)h * H * 4u; }
static size_t logits_of(const OmegaGpuQ36Graph *g, uint32_t h) { return g->lo.logits + (size_t)h * g->am_ctas * OMEGA_DEC_ARGMAX_SPAN * 4u; }
static size_t token_of(const OmegaGpuQ36Graph *g, uint32_t h) { return g->lo.token + (size_t)h * TOKEN_STRIDE; }
static void put64(uint32_t *a, unsigned off, uint64_t v) { a[(off - 0x380) / 4] = (uint32_t)v; a[(off - 0x380) / 4 + 1] = (uint32_t)(v >> 32); }
static void putf(uint32_t *a, unsigned off, float f) { memcpy(&a[(off - 0x380) / 4], &f, 4); }
static int fail(OmegaGpuQ36Graph *g, const char *what, int rc) {
    if (g) g->failed = 1;
    if (what) omega_gpu_session_set_error(what);
    return rc;
}

void omega_q36_gather_table(uint32_t *tab) {
    const uint32_t R = OMEGA_GPU_Q36_ROTARY, QP = 16u * 512u, KP = QP + 2u * 256u;
    for (uint32_t n = 0; n < OMEGA_GPU_Q36_ATT_MIX; n++) {
        uint32_t s;
        if (n < 4096u) s = (n / 256u) * 512u + omega_dattn256_perm_slot(n % 256u, R);              /* q, permuted */
        else if (n < 4608u) s = QP + ((n - 4096u) / 256u) * 256u + omega_dattn256_perm_slot(n % 256u, R); /* k, permuted */
        else if (n < 5120u) s = KP + (n - 4608u);                                                    /* v */
        else s = ((n - 5120u) / 256u) * 512u + 256u + (n - 5120u) % 256u;                            /* gate */
        tab[n] = s;
    }
}

/* lock held */
static int kern_build(Kern *k, const OmegaDecSpec *s) {
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) return OMEGA_GPU_DGRAPH_CODEGEN_FAIL;
    int rc = OMEGA_GPU_DGRAPH_OK;
    size_t need = 0;
    if (omega_blackwell_codegen_dec_prog(s, prog) != 0 || (need = omega_bw_program_code_bytes(prog)) == 0) rc = OMEGA_GPU_DGRAPH_CODEGEN_FAIL;
    else if (omega_gpu_session_alloc_code(need, &k->code) != 0) rc = OMEGA_GPU_DGRAPH_CHIP_FAIL;
    else {
        OmegaBlackwellKernel kk;
        if (omega_blackwell_kernel_encode_into(prog, k->code.cpu, need, &kk) != 0 || kk.code_size != need) rc = OMEGA_GPU_DGRAPH_CODEGEN_FAIL;
        else if (omega_gpu_session_sync_to_gpu(&k->code) != 0) rc = OMEGA_GPU_DGRAPH_CHIP_FAIL;
        else { k->gpr = omega_bw_gpr_declared(kk.gpr_count); if (k->gpr < 64u) k->gpr = 64u; } /* as the decode graph */
        if (rc != OMEGA_GPU_DGRAPH_OK) { omega_gpu_session_free(&k->code); memset(k, 0, sizeof *k); }
    }
    omega_gpu_host_free(prog);
    return rc;
}

void omega_gpu_q36_graph_free(OmegaGpuQ36Graph *g) {
    if (!g) return;
    if (g->cl) omega_gpu_cmdlist_free(g->cl);
    omega_gpu_session_lock();
    for (unsigned i = 0; i < K_COUNT; i++) if (g->k[i].code.cpu) omega_gpu_session_free(&g->k[i].code);
    if (g->buf.cpu) omega_gpu_session_free(&g->buf);
    if (g->dbuf.cpu) omega_gpu_session_free(&g->dbuf);
    omega_gpu_session_unlock();
    omega_gpu_host_free(g);
}

int omega_gpu_q36_graph_create(const OmegaGpuQ36GraphConfig *cfg, OmegaGpuQ36Graph **out) { return omega_gpu_q36_graph_create_heads(cfg, 1, out); }

int omega_gpu_q36_graph_create_heads(const OmegaGpuQ36GraphConfig *cfg, uint32_t heads, OmegaGpuQ36Graph **out) {
    if (out) *out = NULL;
    if (heads == 0 || heads > OMEGA_GPU_Q36_MAX_HEADS) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (!cfg || !out || cfg->layers == 0 || cfg->layers > OMEGA_GPU_Q36_MAX_LAYERS || !cfg->gdn || !cfg->attn || !cfg->router || !cfg->lm_head ||
        cfg->vocab == 0 || cfg->vocab > OMEGA_DEC_ARGMAX_SPAN * OMEGA_DEC_ARGMAX_SPAN) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (cfg->layers < 64u && (cfg->attn_mask >> cfg->layers) != 0) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    uint32_t n = 0, k = 0;
    omega_gpu_bf16_tensor_shape(cfg->lm_head, &n, &k);
    if (n != cfg->vocab || k != H) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    OmegaGpuQ36Graph *g = omega_gpu_host_calloc(1, sizeof *g);
    if (!g) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    g->cfg = *cfg;
    g->am_ctas = (cfg->vocab + OMEGA_DEC_ARGMAX_SPAN - 1) / OMEGA_DEC_ARGMAX_SPAN;
    g->heads = heads; g->gdn_h[0] = cfg->gdn; g->attn_h[0] = cfg->attn;
    for (uint32_t h = 0; h < OMEGA_GPU_Q36_MAX_HEADS; h++) g->xq_layer[h] = -1;
    for (uint32_t l = 0, na = 0, ng = 0; l < cfg->layers; l++) {
        g->layer[l].attn = (int)((cfg->attn_mask >> l) & 1u);
        g->layer[l].kind_index = g->layer[l].attn ? na++ : ng++;
    }
    Layout *L = &g->lo; size_t o = 0;
#define REG(f, bytes) do { L->f = o; o = al(o + (bytes)); } while (0)
    REG(x, (size_t)H * 4 * heads); REG(ab, OMEGA_GPU_Q36_AB * 4); REG(hidden, (size_t)H * 4 * heads);
    REG(logits, (size_t)g->am_ctas * OMEGA_DEC_ARGMAX_SPAN * 4 * heads); REG(am_v, OMEGA_DEC_ARGMAX_SPAN * 4); REG(am_i, OMEGA_DEC_ARGMAX_SPAN * 4);
    REG(token, (size_t)TOKEN_STRIDE * heads); REG(norms, (size_t)cfg->layers * H * 4); REG(fnorm, H * 4); REG(tabs, (size_t)cfg->layers * 4 * 16 + 16); /* + the FP8 lm_head entry */
    REG(gtab, OMEGA_GPU_Q36_ATT_MIX * 4); REG(idtab, COPY_ROW * 4); REG(stage, (size_t)NG * H * 4); /* the largest read_buf region (dn) */
    L->split = o;
    REG(xq, (size_t)H * 4 * heads); REG(mix, OMEGA_GPU_Q36_GDN_MIX * 4); REG(agg, OMEGA_GPU_Q36_ATT_MIX * 4);
    REG(att, OMEGA_GPU_Q36_MIXER_OUT * 4); REG(o, H * 4); REG(gu, (size_t)NG * 2 * EI * 4); REG(act, (size_t)NG * EI * 4); REG(dn, (size_t)NG * H * 4);
#undef REG
    L->bytes = o;
    int rc = OMEGA_GPU_DGRAPH_OK;
    if (omega_gpu_cmdlist_create(OMEGA_GPU_CMDLIST_MAX_LAUNCHES, &g->cl) != 0) { omega_gpu_q36_graph_free(g); return OMEGA_GPU_DGRAPH_CHIP_FAIL; }
    omega_gpu_session_lock();
    const char *pbc = getenv("AIEN_OMEGA_Q36_BUF_CACHED");
    int (*const dalloc)(size_t, OmegaGpuMem *) = (pbc && pbc[0] == '0' && pbc[1] == 0) ? omega_gpu_session_alloc : omega_gpu_session_alloc_cached;
    if (!omega_gpu_session_open() || omega_gpu_session_alloc(L->split, &g->buf) != 0 || dalloc(L->bytes - L->split, &g->dbuf) != 0)
        rc = OMEGA_GPU_DGRAPH_CHIP_FAIL;
    static const OmegaDecSpec spec[K_COUNT] = {
        [K_NORM] = { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_NORM },
        [K_ADD] = { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_ADD },
        [K_CN] = { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM, .groups = NG },
        [K_SW] = { .kind = OMEGA_DEC_SWIGLU, .width = EI },
        [K_GATHER] = { .kind = OMEGA_DEC_GATHER, .width = OMEGA_GPU_Q36_ATT_MIX, .stride = OMEGA_GPU_Q36_ATT_MIX },
        [K_GATE] = { .kind = OMEGA_DEC_GATE },
        [K_AM0] = { .kind = OMEGA_DEC_ARGMAX, .mode = 0 },
        [K_AM1] = { .kind = OMEGA_DEC_ARGMAX, .mode = 1 },
        [K_SW_M1] = { .kind = OMEGA_DEC_SWIGLU, .width = EI, .mutant = 1 },
        [K_GATHER_M1] = { .kind = OMEGA_DEC_GATHER, .width = OMEGA_GPU_Q36_ATT_MIX, .stride = OMEGA_GPU_Q36_ATT_MIX, .mutant = 1 },
        [K_GATE_M1] = { .kind = OMEGA_DEC_GATE, .mutant = 1 },
        [K_CN_M3] = { .kind = OMEGA_DEC_RES, .mode = OMEGA_DEC_RES_COMBINE | OMEGA_DEC_RES_NORM, .groups = NG, .mutant = 3 },
        [K_COPY] = { .kind = OMEGA_DEC_GATHER, .width = COPY_ROW, .stride = COPY_ROW },
    };
    for (unsigned i = 0; i < K_COUNT && rc == OMEGA_GPU_DGRAPH_OK; i++) rc = kern_build(&g->k[i], &spec[i]);
    if (rc == OMEGA_GPU_DGRAPH_OK) {
        memset(g->buf.cpu, 0, L->split);
        memset(g->dbuf.cpu, 0, L->bytes - L->split); /* fresh; every region is written by the chip before a launch reads it */
        uint32_t *id = cpu_of(g, L->idtab);
        for (uint32_t i = 0; i < COPY_ROW; i++) id[i] = i;
        float *amv = cpu_of(g, L->am_v);
        for (uint32_t h = 0; h < heads; h++) {
            float *lg = cpu_of(g, logits_of(g, h));
            for (size_t i = cfg->vocab; i < (size_t)g->am_ctas * OMEGA_DEC_ARGMAX_SPAN; i++) lg[i] = NEG_PAD; /* lm_head writes [0, vocab) only */
        }
        for (size_t i = 0; i < OMEGA_DEC_ARGMAX_SPAN; i++) amv[i] = NEG_PAD;                              /* stage 1 writes [0, ctas) only */
        omega_q36_gather_table(cpu_of(g, L->gtab));
        if (omega_gpu_session_sync_to_gpu(&g->buf) != 0 || omega_gpu_session_sync_to_gpu(&g->dbuf) != 0) rc = OMEGA_GPU_DGRAPH_CHIP_FAIL;
    }
    omega_gpu_session_unlock();
    if (rc != OMEGA_GPU_DGRAPH_OK) { omega_gpu_q36_graph_free(g); return rc; }
    *out = g;
    return OMEGA_GPU_DGRAPH_OK;
}

static int shape_e(const OmegaGpuFp8Tensor *t, uint32_t e, uint32_t n, uint32_t k) {
    uint32_t te = 0, tn = 0, tk = 0; if (t) omega_gpu_fp8_tensor_shape(t, &te, &tn, &tk);
    return t && te == e && tn == n && tk == k;
}
static int shape_is(const OmegaGpuFp8Tensor *t, uint32_t n, uint32_t k) { return shape_e(t, 1, n, k); }
#define N_EXPERTS 257u /* routed experts + the shared one (the router's directory) */

int omega_gpu_q36_graph_set_head_state(OmegaGpuQ36Graph *g, uint32_t head, OmegaGpuGdn *gdn, OmegaGpuDecodeAttn256Cache *attn) {
    if (!g || head >= g->heads || !gdn || !attn || g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    g->gdn_h[head] = gdn; g->attn_h[head] = attn;
    return OMEGA_GPU_DGRAPH_OK;
}

int omega_gpu_q36_graph_set_layer(OmegaGpuQ36Graph *g, uint32_t layer, const OmegaGpuFp8Tensor *mix, const OmegaGpuBf16Tensor *ab,
                                  const OmegaGpuFp8Tensor *o, const OmegaGpuFp8Tensor *gate_up, const OmegaGpuFp8Tensor *down,
                                  const float *in_norm) {
    if (!g || layer >= g->cfg.layers || !in_norm || g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    const int attn = g->layer[layer].attn;
    if (!shape_is(mix, attn ? OMEGA_GPU_Q36_ATT_MIX : OMEGA_GPU_Q36_GDN_MIX, H) || !shape_is(o, H, OMEGA_GPU_Q36_MIXER_OUT) ||
        !shape_e(gate_up, N_EXPERTS, 2 * EI, H) || !shape_e(down, N_EXPERTS, H, EI)) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (attn ? ab != NULL : ab == NULL) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (!attn) { uint32_t n = 0, k = 0; omega_gpu_bf16_tensor_shape(ab, &n, &k); if (n != OMEGA_GPU_Q36_AB || k != H) return OMEGA_GPU_DGRAPH_BAD_ARGS; }
    uint64_t t[8] = { 0 }; /* mix, o; the experts' group tables come from the router */
    if (omega_gpu_fp8_tensor_entry(mix, 0, &t[0], &t[1]) != 0 || omega_gpu_fp8_tensor_entry(o, 0, &t[2], &t[3]) != 0) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    omega_gpu_session_lock();
    memcpy((uint8_t *)cpu_of(g, g->lo.tabs) + (size_t)layer * 64u, t, sizeof t);
    memcpy((float *)cpu_of(g, g->lo.norms) + (size_t)layer * H, in_norm, H * 4u);
    int rc = omega_gpu_session_sync_to_gpu(&g->buf) == 0 ? OMEGA_GPU_DGRAPH_OK : OMEGA_GPU_DGRAPH_CHIP_FAIL;
    omega_gpu_session_unlock();
    if (rc == OMEGA_GPU_DGRAPH_OK) {
        g->layer[layer].mix = mix; g->layer[layer].ab = ab; g->layer[layer].o = o; g->layer[layer].gu = gate_up; g->layer[layer].dn = down;
        g->layer[layer].set = 1;
    }
    return rc;
}

int omega_gpu_q36_graph_set_final_norm(OmegaGpuQ36Graph *g, const float *final_norm) {
    if (!g || !final_norm || g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    omega_gpu_session_lock();
    memcpy(cpu_of(g, g->lo.fnorm), final_norm, H * 4u);
    int rc = omega_gpu_session_sync_to_gpu(&g->buf) == 0 ? OMEGA_GPU_DGRAPH_OK : OMEGA_GPU_DGRAPH_CHIP_FAIL;
    omega_gpu_session_unlock();
    if (rc == OMEGA_GPU_DGRAPH_OK) g->final_set = 1;
    return rc;
}

int omega_gpu_q36_graph_set_lm_head_fp8(OmegaGpuQ36Graph *g, const OmegaGpuFp8Tensor *head) {
    if (!g || g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (!head) { g->head8 = NULL; return OMEGA_GPU_DGRAPH_OK; }
    uint64_t t[2];
    if (!shape_is(head, g->cfg.vocab, H) || omega_gpu_fp8_tensor_entry(head, 0, &t[0], &t[1]) != 0) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    omega_gpu_session_lock();
    memcpy((uint8_t *)cpu_of(g, g->lo.tabs) + (size_t)g->cfg.layers * 64u, t, sizeof t);
    int rc = omega_gpu_session_sync_to_gpu(&g->buf) == 0 ? OMEGA_GPU_DGRAPH_OK : OMEGA_GPU_DGRAPH_CHIP_FAIL;
    omega_gpu_session_unlock();
    if (rc == OMEGA_GPU_DGRAPH_OK) g->head8 = head;
    return rc;
}

void omega_gpu_q36_graph_set_flush_each(OmegaGpuQ36Graph *g, int on) { if (g) g->flush_each = on ? 1 : 0; }
void omega_gpu_q36_graph_test_set_mutant(OmegaGpuQ36Graph *g, int mutant) { if (g) g->mutant = mutant; }

/* ---- the list ---- */

static OmegaGpuCmdListOpts list_opts(const OmegaGpuQ36Graph *g) {
    OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o);
    o.mode = OMEGA_GPU_CMDLIST_PCAS; o.barrier = (OmegaGpuCmdListBarrier)omega_gpu_decode_graph_barrier(); o.timeout_ms = TIMEOUT_MS; o.stamps = g->stamps;
    return o;
}

static int submit_now(OmegaGpuQ36Graph *g) {
    if (omega_gpu_cmdlist_count(g->cl) == 0) return OMEGA_GPU_DGRAPH_OK;
    OmegaGpuCmdListOpts o = list_opts(g);
    OmegaGpuCmdListStats st; memset(&st, 0, sizeof st);
    int rc = omega_gpu_cmdlist_submit(g->cl, &o, &st);
    g->acc.launches += st.launches; g->acc.submits += st.submits; g->acc.chip_ns += st.elapsed_ns; g->acc.sems_done += st.sems_done;
    if (rc == 0 && g->stamps) {
        uint64_t tmp[OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1];
        int cnt = omega_gpu_cmdlist_stamps(g->cl, tmp, OMEGA_GPU_CMDLIST_MAX_LAUNCHES + 1);
        if (cnt < 0) rc = -1;
        else { const int from = g->n_stamp == 0 ? 0 : 1; for (int i = from; i < cnt; i++) g->stamp[g->n_stamp++] = tmp[i]; }
    }
    g->tagged_in_list = 0;
    omega_gpu_cmdlist_reset(g->cl);
    if (rc != 0) return fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    if (st.sems_done != st.launches) {
        char e[128]; omega_gpu_fmt(e, sizeof e, "q36 graph: %u of %u launch semaphores DONE (first missing %d)", st.sems_done, st.launches, st.first_missing);
        return fail(g, e, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    }
    return OMEGA_GPU_DGRAPH_OK;
}

static void tag(OmegaGpuQ36Graph *g, uint8_t kind) {
    const uint32_t n = omega_gpu_cmdlist_count(g->cl);
    for (; g->tagged_in_list < n && g->n_kind < OMEGA_GPU_CMDLIST_MAX_LAUNCHES; g->tagged_in_list++) g->kind[g->n_kind++] = kind;
}

static int after_op(OmegaGpuQ36Graph *g, uint8_t kind, int api_rc) {
    if (api_rc != 0) return fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    tag(g, kind);
    return g->flush_each ? submit_now(g) : OMEGA_GPU_DGRAPH_OK;
}

static int add_glue(OmegaGpuQ36Graph *g, uint8_t kind, int ki, uint32_t gx, uint32_t gy, size_t elems, const uint32_t *args, uint32_t n_args) {
    const Kern *k = &g->k[ki];
    OmegaGpuLaunch L; memset(&L, 0, sizeof L);
    L.code_va = k->code.va; L.gpr_count = k->gpr; L.threads_x = OMEGA_DEC_THREADS; L.threads_y = 1; L.grid_x = gx; L.grid_y = gy;
    L.num_elements = elems; L.args = args; L.n_args = n_args; L.timeout_ms = TIMEOUT_MS;
    return after_op(g, kind, omega_gpu_cmdlist_add(g->cl, &L) < 0 ? -1 : 0);
}

/* RES at width 2048: NORM (x -> y), ADD (x += a) or COMBINE|NORM (x += sum_g wts[g] * a[g]; y = norm(x)) */
static int rec_res_w(OmegaGpuQ36Graph *g, int ki, uint64_t x, uint64_t a, uint64_t wts, uint64_t nw, uint64_t y, float scale) {
    uint32_t w[OMEGA_DEC_RES_WORDS]; memset(w, 0, sizeof w);
    put64(w, OMEGA_DEC_RES_X, x); put64(w, OMEGA_DEC_RES_A, a); put64(w, OMEGA_DEC_RES_W, wts); put64(w, OMEGA_DEC_RES_NW, nw);
    put64(w, OMEGA_DEC_RES_Y, y); putf(w, OMEGA_DEC_RES_EPS, g->cfg.eps); putf(w, OMEGA_DEC_RES_SCALE, scale);
    return add_glue(g, ki == K_NORM ? OMEGA_GPU_Q36_K_NORM : OMEGA_GPU_Q36_K_RES, ki, 1, 1, H, w, OMEGA_DEC_RES_WORDS);
}
static int rec_res(OmegaGpuQ36Graph *g, int ki, uint64_t x, uint64_t a, uint64_t nw, uint64_t y, float scale) {
    return rec_res_w(g, ki, x, a, 0, nw, y, scale);
}

/* The a|b GEMV beside the mix GEMV (a speed lever of the 27B, values identical): OFF by default here until the plain graph
 * has passed its gates (PRIORITY CHANGE 2026-10-20); env AIEN_OMEGA_Q36_OVERLAP_AB=1 turns it on. */
static int g_overlap_ab = -1;
static int q36_overlap_ab(void) {
    if (g_overlap_ab < 0) { const char *e = getenv("AIEN_OMEGA_Q36_OVERLAP_AB"); g_overlap_ab = (e && e[0] == '1' && e[1] == 0) ? 1 : 0; }
    return g_overlap_ab;
}

static int rec_fp8(OmegaGpuQ36Graph *g, uint8_t kind, const OmegaGpuFp8Tensor *t, uint64_t tab, uint64_t x, uint64_t y) {
    return after_op(g, kind, omega_gpu_fp8_gemv_grouped_record(g->cl, t, 1, tab, x, 1, y));
}

int omega_gpu_q36_graph_begin_token(OmegaGpuQ36Graph *g, const float *x_embed) { return omega_gpu_q36_graph_begin_stroke(g, x_embed, 1); }

int omega_gpu_q36_graph_begin_stroke(OmegaGpuQ36Graph *g, const float *x_embed, uint32_t heads) {
    if (!g || g->in_token || heads == 0 || heads > g->heads) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    for (uint32_t h = 1; h < heads; h++) if (!g->gdn_h[h] || !g->attn_h[h]) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    omega_gpu_session_lock();
    g->t_begin = omega_gpu_session_now_ns();
    g->in_token = 1; g->failed = 0; g->stroke_heads = heads;
    for (uint32_t h = 0; h < heads; h++) { g->xq_layer[h] = -1; g->hidden_ready[h] = 0; }
    memset(&g->acc, 0, sizeof g->acc);
    g->n_kind = 0; g->tagged_in_list = 0; g->n_stamp = 0;
    omega_gpu_cmdlist_reset(g->cl);
    if (!omega_gpu_session_open()) return fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    for (uint32_t h = 0; h < heads; h++) {
        uint32_t *tok = cpu_of(g, token_of(g, h));
        tok[0] = TOKEN_POISON; tok[1] = TOKEN_POISON;
        if (x_embed) memcpy(cpu_of(g, x_of(g, h)), x_embed + (size_t)h * H, H * 4u);
    }
    if (omega_gpu_session_sync_to_gpu(&g->buf) != 0) return fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    return OMEGA_GPU_DGRAPH_OK;
}

int omega_gpu_q36_graph_record_layer(OmegaGpuQ36Graph *g, uint32_t l, uint32_t pos) { return omega_gpu_q36_graph_record_head_layer(g, 0, l, pos); }

int omega_gpu_q36_graph_record_head_layer(OmegaGpuQ36Graph *g, uint32_t h, uint32_t l, uint32_t pos) {
    if (!g || !g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (g->failed) return OMEGA_GPU_DGRAPH_CHIP_FAIL;
    if (h >= g->stroke_heads) return fail(g, "q36 graph: head not in the stroke", OMEGA_GPU_DGRAPH_BAD_ARGS);
    if (l >= g->cfg.layers || !g->layer[l].set || g->hidden_ready[h] || (l + 1 == g->cfg.layers && !g->final_set))
        return fail(g, "q36 graph: layer not set, out of order, or final norm missing", OMEGA_GPU_DGRAPH_BAD_ARGS);
    const uint64_t x = va_of(g, x_of(g, h)), xq = va_of(g, xq_of(g, h)), mix = va_of(g, g->lo.mix), agg = va_of(g, g->lo.agg);
    const uint64_t att = va_of(g, g->lo.att), o = va_of(g, g->lo.o), gu = va_of(g, g->lo.gu), act = va_of(g, g->lo.act), dn = va_of(g, g->lo.dn);
    const uint64_t in_norm = va_of(g, g->lo.norms) + (uint64_t)l * H * 4u, tab = va_of(g, g->lo.tabs) + (uint64_t)l * 64u;
    const uint32_t ki = g->layer[l].kind_index;
    int rc = OMEGA_GPU_DGRAPH_OK;
    if (g->xq_layer[h] != (int)l) rc = rec_res(g, K_NORM, x, 0, in_norm, xq, SCALE_2P64);
    /* GDN layers: the a|b GEMV (latency-bound, ~18 us) is recorded first and the mix GEMV marked independent of it: both only
     * read xq and write their own buffers, so the two grids overlap; the GDN step after them waits for both (normal barrier).
     * AIEN_OMEGA_Q36_OVERLAP_AB=0 or set_overlap_ab(0): the old order, mix then a|b, each behind a barrier. */
    const int overlap = !g->layer[l].attn && q36_overlap_ab();
    if (overlap && rc == OMEGA_GPU_DGRAPH_OK) rc = after_op(g, OMEGA_GPU_Q36_K_AB, omega_gpu_bf16_gemv_record(g->cl, g->layer[l].ab, xq, va_of(g, g->lo.ab)));
    if (rc == OMEGA_GPU_DGRAPH_OK) {
        int mrc = omega_gpu_fp8_gemv_grouped_record(g->cl, g->layer[l].mix, 1, tab, xq, 1, mix);
        if (mrc == 0 && overlap && !g->flush_each && omega_gpu_cmdlist_mark_independent(g->cl) != 0) mrc = -1;
        rc = after_op(g, OMEGA_GPU_Q36_K_MIX, mrc);
    }
    uint64_t mixer_out = att;
    if (!g->layer[l].attn) {
        const uint64_t ab = va_of(g, g->lo.ab);
        if (!overlap && rc == OMEGA_GPU_DGRAPH_OK) rc = after_op(g, OMEGA_GPU_Q36_K_AB, omega_gpu_bf16_gemv_record(g->cl, g->layer[l].ab, xq, ab));
        if (rc == OMEGA_GPU_DGRAPH_OK)
            rc = after_op(g, OMEGA_GPU_Q36_K_GDN, omega_gpu_gdn_record(g->cl, g->gdn_h[h], ki, mix, mix + OMEGA_GPU_Q36_GDN_QKV * 4u, ab,
                                                                     g->cfg.eps_l2, g->cfg.eps, 1.0f, SCALE_2P64, &mixer_out));
    } else {
        uint32_t w[OMEGA_DEC_GA_WORDS]; memset(w, 0, sizeof w);
        put64(w, OMEGA_DEC_GA_IN, mix); put64(w, OMEGA_DEC_GA_TAB, va_of(g, g->lo.gtab)); put64(w, OMEGA_DEC_GA_OUT, agg);
        if (rc == OMEGA_GPU_DGRAPH_OK)
            rc = add_glue(g, OMEGA_GPU_Q36_K_GATHER, g->mutant == 2 ? K_GATHER_M1 : K_GATHER, (OMEGA_GPU_Q36_ATT_MIX + OMEGA_DEC_THREADS - 1) / OMEGA_DEC_THREADS, 1,
                          OMEGA_GPU_Q36_ATT_MIX, w, OMEGA_DEC_GA_WORDS);
        if (rc == OMEGA_GPU_DGRAPH_OK)
            rc = after_op(g, OMEGA_GPU_Q36_K_ATTN, omega_gpu_decode_attn256_record(g->cl, g->attn_h[h], ki, pos, agg, g->cfg.eps, SCALE_2P64, att));
        uint32_t w2[OMEGA_DEC_GT_WORDS]; memset(w2, 0, sizeof w2);
        put64(w2, OMEGA_DEC_GT_A, att); put64(w2, OMEGA_DEC_GT_G, agg + OMEGA_GPU_Q36_ATT_QKV * 4u); put64(w2, OMEGA_DEC_GT_OUT, att);
        w2[(OMEGA_DEC_GT_N - 0x380) / 4] = OMEGA_GPU_Q36_MIXER_OUT;
        if (rc == OMEGA_GPU_DGRAPH_OK)
            rc = add_glue(g, OMEGA_GPU_Q36_K_GATE, g->mutant == 3 ? K_GATE_M1 : K_GATE, OMEGA_GPU_Q36_MIXER_OUT / OMEGA_DEC_THREADS, 1,
                          OMEGA_GPU_Q36_MIXER_OUT, w2, OMEGA_DEC_GT_WORDS);
    }
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = rec_fp8(g, OMEGA_GPU_Q36_K_O, g->layer[l].o, tab + 16u, mixer_out, o);
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = rec_res_w(g, K_ADD, x, o, 0, 0, 0, 1.0f);
    /* the sparse block: router (post-attention norm, top-8 + the shared slot), 9 expert groups, COMBINE|NORM */
    OmegaGpuRouterDevOut r; memset(&r, 0, sizeof r);
    if (rc == OMEGA_GPU_DGRAPH_OK)
        rc = after_op(g, OMEGA_GPU_Q36_K_ROUTER, omega_gpu_router_record(g->cl, g->cfg.router, l, x, g->cfg.eps, 1, SCALE_2P64, &r));
    if (rc == OMEGA_GPU_DGRAPH_OK)
        rc = after_op(g, OMEGA_GPU_Q36_K_GATE_UP, omega_gpu_fp8_gemv_grouped_record(g->cl, g->layer[l].gu, NG, r.tab_gu_va, r.xout_va, 1, gu));
    uint32_t w3[OMEGA_DEC_SW_WORDS]; memset(w3, 0, sizeof w3);
    put64(w3, OMEGA_DEC_SW_GU, gu); put64(w3, OMEGA_DEC_SW_ACT, act); putf(w3, OMEGA_DEC_SW_SCALE, SCALE_2P64);
    if (rc == OMEGA_GPU_DGRAPH_OK)
        rc = add_glue(g, OMEGA_GPU_Q36_K_SWIGLU, g->mutant == 1 ? K_SW_M1 : K_SW, EI / OMEGA_DEC_THREADS, NG, (size_t)NG * EI, w3, OMEGA_DEC_SW_WORDS);
    if (rc == OMEGA_GPU_DGRAPH_OK)
        rc = after_op(g, OMEGA_GPU_Q36_K_DOWN, omega_gpu_fp8_gemv_grouped_record(g->cl, g->layer[l].dn, NG, r.tab_dn_va, act, NG, dn));
    if (rc == OMEGA_GPU_DGRAPH_OK) {
        const int last = l + 1 == g->cfg.layers;
        const uint64_t nw = last ? va_of(g, g->lo.fnorm) : in_norm + H * 4u;
        rc = rec_res_w(g, g->mutant == 4 ? K_CN_M3 : K_CN, x, dn, r.wts_va, nw, last ? va_of(g, hidden_of(g, h)) : xq,
                       last && !g->head8 ? 1.0f : SCALE_2P64);
        if (rc == OMEGA_GPU_DGRAPH_OK) { if (last) g->hidden_ready[h] = 1; else g->xq_layer[h] = (int)l + 1; }
    }
    return rc == OMEGA_GPU_DGRAPH_OK ? rc : fail(g, NULL, rc);
}

int omega_gpu_q36_graph_record_final(OmegaGpuQ36Graph *g) { return omega_gpu_q36_graph_record_head_final(g, 0); }

int omega_gpu_q36_graph_record_head_final(OmegaGpuQ36Graph *g, uint32_t h) {
    if (!g || !g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (g->failed) return OMEGA_GPU_DGRAPH_CHIP_FAIL;
    if (h >= g->stroke_heads) return fail(g, "q36 graph: head not in the stroke", OMEGA_GPU_DGRAPH_BAD_ARGS);
    if (!g->final_set) return fail(g, "q36 graph: final norm not set", OMEGA_GPU_DGRAPH_BAD_ARGS);
    const uint64_t x = va_of(g, x_of(g, h)), hidden = va_of(g, hidden_of(g, h)), logits = va_of(g, logits_of(g, h)), token = va_of(g, token_of(g, h));
    int rc = OMEGA_GPU_DGRAPH_OK;
    if (!g->hidden_ready[h]) rc = rec_res(g, K_NORM, x, 0, va_of(g, g->lo.fnorm), hidden, g->head8 ? SCALE_2P64 : 1.0f);
    if (rc == OMEGA_GPU_DGRAPH_OK)
        rc = g->head8 ? rec_fp8(g, OMEGA_GPU_Q36_K_LM_HEAD, g->head8, va_of(g, g->lo.tabs) + (uint64_t)g->cfg.layers * 64u, hidden, logits)
                      : after_op(g, OMEGA_GPU_Q36_K_LM_HEAD, omega_gpu_bf16_gemv_record(g->cl, g->cfg.lm_head, hidden, logits));
    uint32_t a0[OMEGA_DEC_AM_WORDS], a1[OMEGA_DEC_AM_WORDS]; memset(a0, 0, sizeof a0); memset(a1, 0, sizeof a1);
    const uint64_t amv = va_of(g, g->lo.am_v), ami = va_of(g, g->lo.am_i);
    put64(a0, OMEGA_DEC_AM_V, logits); put64(a0, OMEGA_DEC_AM_I, 0); put64(a0, OMEGA_DEC_AM_OV, amv); put64(a0, OMEGA_DEC_AM_OI, ami);
    put64(a1, OMEGA_DEC_AM_V, amv); put64(a1, OMEGA_DEC_AM_I, ami); put64(a1, OMEGA_DEC_AM_OV, token + 4u); put64(a1, OMEGA_DEC_AM_OI, token);
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = add_glue(g, OMEGA_GPU_Q36_K_ARGMAX, K_AM0, g->am_ctas, 1, (size_t)g->am_ctas * OMEGA_DEC_ARGMAX_SPAN, a0, OMEGA_DEC_AM_WORDS);
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = add_glue(g, OMEGA_GPU_Q36_K_ARGMAX, K_AM1, 1, 1, OMEGA_DEC_ARGMAX_SPAN, a1, OMEGA_DEC_AM_WORDS);
    return rc == OMEGA_GPU_DGRAPH_OK ? rc : fail(g, NULL, rc);
}

int omega_gpu_q36_graph_end_token(OmegaGpuQ36Graph *g, uint32_t *token, OmegaGpuDecodeGraphStats *st) { return omega_gpu_q36_graph_end_stroke(g, token, st); }

int omega_gpu_q36_graph_end_stroke(OmegaGpuQ36Graph *g, uint32_t *tokens, OmegaGpuDecodeGraphStats *st) {
    if (!g || !g->in_token) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    int rc = g->failed ? OMEGA_GPU_DGRAPH_CHIP_FAIL : OMEGA_GPU_DGRAPH_OK;
    const uint64_t t_rec = omega_gpu_session_now_ns();
    if (rc == OMEGA_GPU_DGRAPH_OK) rc = submit_now(g);
    for (uint32_t h = 0; h < g->stroke_heads && rc == OMEGA_GPU_DGRAPH_OK; h++)
        if (omega_gpu_decode_attn256_counters_check(g->attn_h[h]) != 0) rc = fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    if (rc == OMEGA_GPU_DGRAPH_OK && omega_gpu_session_sync_to_cpu(&g->buf) != 0) rc = fail(g, NULL, OMEGA_GPU_DGRAPH_CHIP_FAIL);
    for (uint32_t h = 0; h < g->stroke_heads && rc == OMEGA_GPU_DGRAPH_OK; h++) {
        const uint32_t t = ((const uint32_t *)cpu_of(g, token_of(g, h)))[0];
        if (t == TOKEN_POISON || t >= g->cfg.vocab) {
            char e[96]; omega_gpu_fmt(e, sizeof e, "q36 graph: head %u token output %u not written or out of range", h, t);
            rc = fail(g, e, OMEGA_GPU_DGRAPH_CHIP_FAIL);
        } else if (tokens) tokens[h] = t;
    }
    omega_gpu_cmdlist_reset(g->cl);
    g->in_token = 0;
    const uint64_t t_end = omega_gpu_session_now_ns();
    if (st) { *st = g->acc; st->record_ns = (g->flush_each ? t_end : t_rec) - g->t_begin; st->total_ns = t_end - g->t_begin; }
    omega_gpu_session_unlock();
    return rc;
}

int omega_gpu_q36_graph_token(OmegaGpuQ36Graph *g, const float *x_embed, uint32_t pos, uint32_t *token, OmegaGpuDecodeGraphStats *st) {
    if (!g) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (omega_gpu_q36_graph_begin_token(g, x_embed) == OMEGA_GPU_DGRAPH_BAD_ARGS) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    for (uint32_t l = 0; l < g->cfg.layers && !g->failed; l++) (void)omega_gpu_q36_graph_record_layer(g, l, pos);
    if (!g->failed) (void)omega_gpu_q36_graph_record_final(g);
    return omega_gpu_q36_graph_end_token(g, token, st);
}

int omega_gpu_q36_graph_stroke(OmegaGpuQ36Graph *g, const float *x_embed, const uint32_t *pos, uint32_t heads, uint32_t *tokens,
                               OmegaGpuDecodeGraphStats *st) {
    if (!g || !pos) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (omega_gpu_q36_graph_begin_stroke(g, x_embed, heads) == OMEGA_GPU_DGRAPH_BAD_ARGS) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    /* layer-major: every head's layer l before layer l + 1, so one layer's expert rows are warm across the heads */
    for (uint32_t l = 0; l < g->cfg.layers && !g->failed; l++)
        for (uint32_t h = 0; h < heads && !g->failed; h++) (void)omega_gpu_q36_graph_record_head_layer(g, h, l, pos[h]);
    for (uint32_t h = 0; h < heads && !g->failed; h++) (void)omega_gpu_q36_graph_record_head_final(g, h);
    return omega_gpu_q36_graph_end_stroke(g, tokens, st);
}

int omega_gpu_q36_graph_read(OmegaGpuQ36Graph *g, float *x, float *hidden, float *logits) { return omega_gpu_q36_graph_read_head(g, 0, x, hidden, logits); }

int omega_gpu_q36_graph_read_head(OmegaGpuQ36Graph *g, uint32_t h, float *x, float *hidden, float *logits) {
    if (!g || g->in_token || h >= g->heads) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = omega_gpu_session_sync_to_cpu(&g->buf) == 0 ? OMEGA_GPU_DGRAPH_OK : OMEGA_GPU_DGRAPH_CHIP_FAIL;
    if (rc == OMEGA_GPU_DGRAPH_OK) {
        if (x) memcpy(x, cpu_of(g, x_of(g, h)), H * 4u);
        if (hidden) memcpy(hidden, cpu_of(g, hidden_of(g, h)), H * 4u);
        if (hidden && g->head8) for (unsigned i = 0; i < H; i++) hidden[i] *= 0x1p-64f; /* exact: a power of two */
        if (logits) memcpy(logits, cpu_of(g, logits_of(g, h)), (size_t)g->cfg.vocab * 4u);
    }
    omega_gpu_session_unlock();
    return rc;
}

int omega_gpu_q36_graph_read_buf(OmegaGpuQ36Graph *g, int which, float *out, uint32_t n) {
    if (!g || g->in_token || !out || which < 0 || which >= OMEGA_GPU_Q36_BUF_COUNT) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    const struct { size_t off; uint32_t cap; } b[OMEGA_GPU_Q36_BUF_COUNT] = {
        [OMEGA_GPU_Q36_BUF_X] = { g->lo.x, H }, [OMEGA_GPU_Q36_BUF_XQ] = { g->lo.xq, H }, [OMEGA_GPU_Q36_BUF_MIX] = { g->lo.mix, OMEGA_GPU_Q36_GDN_MIX },
        [OMEGA_GPU_Q36_BUF_AB] = { g->lo.ab, OMEGA_GPU_Q36_AB }, [OMEGA_GPU_Q36_BUF_AGG] = { g->lo.agg, OMEGA_GPU_Q36_ATT_MIX },
        [OMEGA_GPU_Q36_BUF_ATT] = { g->lo.att, OMEGA_GPU_Q36_MIXER_OUT }, [OMEGA_GPU_Q36_BUF_O] = { g->lo.o, H },
        [OMEGA_GPU_Q36_BUF_GU] = { g->lo.gu, NG * 2 * EI }, [OMEGA_GPU_Q36_BUF_ACT] = { g->lo.act, NG * EI }, [OMEGA_GPU_Q36_BUF_HIDDEN] = { g->lo.hidden, H },
        [OMEGA_GPU_Q36_BUF_DN] = { g->lo.dn, NG * H },
    };
    if (n > b[which].cap) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_DGRAPH_OK;
    size_t from = b[which].off;
    if (from >= g->lo.split && n) {
        /* a chip-only region: K_COPY moves ceil(n / 256) rows of 256 floats into stage (every cap is a multiple of 256) */
        uint32_t w[OMEGA_DEC_GA_WORDS]; memset(w, 0, sizeof w);
        put64(w, OMEGA_DEC_GA_IN, va_of(g, from)); put64(w, OMEGA_DEC_GA_TAB, va_of(g, g->lo.idtab)); put64(w, OMEGA_DEC_GA_OUT, va_of(g, g->lo.stage));
        const Kern *k = &g->k[K_COPY];
        OmegaGpuLaunch L; memset(&L, 0, sizeof L);
        L.code_va = k->code.va; L.gpr_count = k->gpr; L.threads_x = OMEGA_DEC_THREADS; L.threads_y = 1; L.grid_x = 1; L.grid_y = (n + COPY_ROW - 1) / COPY_ROW;
        L.num_elements = (size_t)L.grid_y * COPY_ROW; L.args = w; L.n_args = OMEGA_DEC_GA_WORDS; L.timeout_ms = TIMEOUT_MS;
        if (!omega_gpu_session_open() || omega_gpu_session_launch(&L, NULL, NULL) != 0) rc = OMEGA_GPU_DGRAPH_CHIP_FAIL;
        from = g->lo.stage;
    }
    if (rc == OMEGA_GPU_DGRAPH_OK && omega_gpu_session_sync_to_cpu(&g->buf) != 0) rc = OMEGA_GPU_DGRAPH_CHIP_FAIL;
    if (rc == OMEGA_GPU_DGRAPH_OK) memcpy(out, cpu_of(g, from), (size_t)n * 4u);
    omega_gpu_session_unlock();
    return rc;
}

/* ---- timeline ---- */
void omega_gpu_q36_graph_set_stamps(OmegaGpuQ36Graph *g, int on) { if (g && !g->in_token) g->stamps = on ? 1 : 0; }

int omega_gpu_q36_graph_stamps(const OmegaGpuQ36Graph *g, uint64_t *stamps, uint8_t *kinds, uint32_t cap) {
    if (!g || g->in_token || !stamps || !kinds) return OMEGA_GPU_DGRAPH_BAD_ARGS;
    if (!g->stamps || g->n_stamp != g->n_kind + 1u) { omega_gpu_session_set_error("q36 graph: no timeline for the last token (stamps off or it failed)"); return OMEGA_GPU_DGRAPH_BAD_ARGS; }
    if (cap < g->n_stamp) { omega_gpu_session_set_error("q36 graph: timeline buffers too small"); return OMEGA_GPU_DGRAPH_BAD_ARGS; }
    memcpy(stamps, g->stamp, (size_t)g->n_stamp * sizeof stamps[0]);
    memcpy(kinds, g->kind, (size_t)g->n_kind);
    return (int)g->n_kind;
}

const char *omega_gpu_q36_graph_kind_name(uint8_t kind) {
    static const char *const names[OMEGA_GPU_Q36_K_COUNT] = { "norm", "mix", "ab", "gdn", "gather", "attn", "gate", "o_proj", "res", "router",
                                                              "gate_up", "swiglu", "down", "lm_head", "argmax" };
    return kind < OMEGA_GPU_Q36_K_COUNT ? names[kind] : "?";
}
