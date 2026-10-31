/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* omega_gpu_pfq_api.c: tiled mma.sync (QMMA) FP8 path for prefill expert GEMMs (DRA-82). See the header. */
#include "omega_gpu_pfq_api.h"
#include "omega_gpu_prefill.h"
#include "omega_gpu_cmdlist.h"
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(OmegaPfqTile) == OMEGA_QG_TILE_BYTES, "tile entry = the q36 graph's segment entry");
_Static_assert(sizeof(OmegaQgSpec) <= sizeof(((OmegaPfKey *)0)->words), "gemm spec fits the kernel key");
_Static_assert(sizeof(OmegaPqSpec) <= sizeof(((OmegaPfKey *)0)->words), "quant spec fits the kernel key");

static int mutant = -1;
int omega_gpu_pfq_mutant(void) {
    if (mutant < 0) {
        const char *v = getenv("AIEN_OMEGA_PFQ_MUTANT");
        const long m = v && *v ? strtol(v, NULL, 10) : 0;
        mutant = m >= 1 && m <= 5 ? (int)m : 0;
    }
    return mutant;
}
void omega_gpu_pfq_test_set_mutant(int m) { mutant = m >= 1 && m <= 5 ? m : 0; }

static uint32_t pfq_wait(void) {
    static int w = -1;
    if (w < 0) {
        const char *v = getenv("AIEN_OMEGA_PFQ_WAIT");
        const long x = v && *v ? strtol(v, NULL, 10) : 45;
        w = x >= 0 && x <= 120 ? (int)x : 45; /* 45 cycles: the chip-safe QMMA read distance (evidence OWN30B-L1G-QMMA-PFGEMM-c742aa13) */
    }
    return (uint32_t)w;
}

uint32_t omega_pfq_tile_cap(uint32_t rows, uint32_t segs) { return segs + rows / OMEGA_PFQ_BM; }

uint32_t omega_pfq_tiles(const OmegaPfqTile *segs, uint32_t nsegs, OmegaPfqTile *tiles, uint32_t cap) {
    if (!segs || !tiles) return 0;
    const uint32_t skew = omega_gpu_pfq_mutant() == 1 ? 1u : 0u;
    uint32_t nt = 0;
    for (uint32_t i = 0; i < nsegs; i++) {
        const OmegaPfqTile *s = &segs[i];
        for (uint32_t r = 0; r < s->cnt; r += OMEGA_PFQ_BM) {
            if (nt >= cap) return 0;
            tiles[nt].w_va = s->w_va; tiles[nt].s_va = s->s_va;
            tiles[nt].row0 = s->row0 + r - (r ? skew : 0u);
            tiles[nt].cnt = s->cnt - r < OMEGA_PFQ_BM ? s->cnt - r : OMEGA_PFQ_BM;
            tiles[nt].pad = 0; nt++;
        }
    }
    return nt;
}

static void put64(uint32_t *a, uint64_t v) { a[0] = (uint32_t)v; a[1] = (uint32_t)(v >> 32); }

int omega_pfq_quant_plan(const OmegaPfqQuant *q, OmegaPqSpec *spec, OmegaPfqLaunch *L) {
    if (!q || !spec || !L || !q->x_va || !q->q_va || !q->s_va || q->in_type > OMEGA_QG_IN_BF16) return OMEGA_GPU_PF_BAD_ARGS;
    memset(spec, 0, sizeof *spec); memset(L, 0, sizeof *L);
    spec->k = q->k; spec->in_type = q->in_type; spec->stride = q->stride;
    spec->mutant = omega_gpu_pfq_mutant() == 4 ? 2 : 0;
    const uint32_t kb = q->k / 128u;
    if (q->k % 128u == 0 && kb > 32u) for (spec->groups = 32u; kb % spec->groups; spec->groups--) {} /* wide row: split over ctaid.y */
    if (!omega_pq_spec_ok(spec)) return OMEGA_GPU_PF_BAD_ARGS;
    if (q->rows > 0x7fffffffu) return OMEGA_GPU_PF_TOO_LARGE;
    L->threads = 32u * (spec->groups ? spec->groups : kb);
    L->grid_x = q->rows; L->grid_y = spec->groups ? kb / spec->groups : 1u;
    L->nargs = OMEGA_PQ_ARGS_WORDS;
    put64(&L->args[0], q->x_va); put64(&L->args[2], q->q_va); put64(&L->args[4], q->s_va);
    return OMEGA_GPU_PF_OK;
}

int omega_pfq_gemm_plan(const OmegaPfqGemm *c, OmegaQgSpec *spec, OmegaPfqLaunch *L) {
    if (!c || !spec || !L || !c->tiles_va || !c->xq_va || !c->xs_va || !c->y_va || c->x64 > 1u || (c->epi != OMEGA_QG_EPI_F32 && c->epi != OMEGA_QG_EPI_BF16))
        return OMEGA_GPU_PF_BAD_ARGS;
    memset(spec, 0, sizeof *spec); memset(L, 0, sizeof *L);
    const int m = omega_gpu_pfq_mutant();
    spec->k = c->k; spec->rt = 4u; spec->warps = 8u; spec->wait = pfq_wait(); spec->epi = c->epi;
    spec->direct = c->rowidx_va == 0; spec->n = c->n; spec->mutant = m == 2 || m == 3 || (m == 5 && c->epi == OMEGA_QG_EPI_BF16) ? m : 0;
    spec->ymap = c->ymap_va != 0; spec->x64 = c->x64;
    if (!omega_qg_spec_ok(spec)) return OMEGA_GPU_PF_BAD_ARGS;
    if (c->ntiles > 65535u) return OMEGA_GPU_PF_TOO_LARGE;
    L->threads = 32u * spec->warps; L->grid_x = c->n / omega_qg_span(spec); L->grid_y = c->ntiles;
    L->shared = omega_qg_shared_bytes(spec);
    const uint64_t va[5] = { c->tiles_va, c->rowidx_va, c->xq_va, c->y_va, c->xs_va };
    for (int i = 0; i < 5; i++) put64(&L->args[2 * i], va[i]);
    L->args[10] = c->n;
    if (spec->ymap) { put64(&L->args[12], c->ymap_va); L->nargs = OMEGA_QG_ARGS_WORDS_YMAP; }
    else L->nargs = OMEGA_QG_ARGS_WORDS;
    return OMEGA_GPU_PF_OK;
}

static int gen_quant(const OmegaPfKey *key, void *prog) { OmegaPqSpec s; memcpy(&s, key->words, sizeof s); return omega_blackwell_codegen_pq_prog(&s, prog); }
static int gen_gemm(const OmegaPfKey *key, void *prog) { OmegaQgSpec s; memcpy(&s, key->words, sizeof s); return omega_blackwell_codegen_qgemm_prog(&s, prog); }

/* cl NULL: generate and upload only (the prepare calls) */
static int run_locked(OmegaGpuCmdList *cl, uint32_t kind, const void *spec, size_t spec_bytes, OmegaPfGen gen, const OmegaPfqLaunch *L) {
    OmegaPfKey key; memset(&key, 0, sizeof key); key.kind = kind; memcpy(key.words, spec, spec_bytes);
    int rc = omega_gpu_pf_open_locked();
    OmegaPfKernel kern;
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_kernel_locked(&key, gen, &kern);
    if (rc == OMEGA_GPU_PF_OK && cl && L->grid_x && L->grid_y)
        rc = omega_gpu_pf_record_locked(cl, &kern, L->threads, L->grid_x, L->grid_y, L->shared, L->args, L->nargs);
    return rc;
}

int omega_gpu_pfq_quant_record_locked(OmegaGpuCmdList *cl, const OmegaPfqQuant *q) {
    if (!cl) return OMEGA_GPU_PF_BAD_ARGS;
    OmegaPqSpec s; OmegaPfqLaunch L;
    int rc = omega_pfq_quant_plan(q, &s, &L);
    if (rc != OMEGA_GPU_PF_OK || q->rows == 0) return rc;
    return run_locked(cl, OMEGA_PFQ_KIND_QUANT, &s, sizeof s, gen_quant, &L);
}

int omega_gpu_pfq_gemm_record_locked(OmegaGpuCmdList *cl, const OmegaPfqGemm *c) {
    if (!cl) return OMEGA_GPU_PF_BAD_ARGS;
    OmegaQgSpec s; OmegaPfqLaunch L;
    int rc = omega_pfq_gemm_plan(c, &s, &L);
    if (rc != OMEGA_GPU_PF_OK || c->ntiles == 0) return rc;
    return run_locked(cl, OMEGA_PFQ_KIND_GEMM, &s, sizeof s, gen_gemm, &L);
}

int omega_gpu_pfq_quant_prepare(uint32_t k, uint32_t stride, uint32_t in_type) {
    const OmegaPfqQuant q = { .k = k, .rows = 0, .stride = stride, .in_type = in_type, .x_va = 1, .q_va = 1, .s_va = 1 };
    OmegaPqSpec s; OmegaPfqLaunch L;
    int rc = omega_pfq_quant_plan(&q, &s, &L);
    if (rc != OMEGA_GPU_PF_OK) return rc;
    omega_gpu_session_lock();
    rc = run_locked(NULL, OMEGA_PFQ_KIND_QUANT, &s, sizeof s, gen_quant, &L);
    omega_gpu_session_unlock();
    return rc;
}

int omega_gpu_pfq_gemm_prepare(uint32_t n, uint32_t k, uint32_t x64, int indexed, int ymap, uint32_t epi) {
    const OmegaPfqGemm c = { .n = n, .k = k, .x64 = x64, .ntiles = 0, .tiles_va = 1, .rowidx_va = indexed ? 1u : 0u, .xq_va = 1,
                             .xs_va = 1, .ymap_va = ymap ? 1u : 0u, .y_va = 1, .epi = epi };
    OmegaQgSpec s; OmegaPfqLaunch L;
    int rc = omega_pfq_gemm_plan(&c, &s, &L);
    if (rc != OMEGA_GPU_PF_OK) return rc;
    omega_gpu_session_lock();
    rc = run_locked(NULL, OMEGA_PFQ_KIND_GEMM, &s, sizeof s, gen_gemm, &L);
    omega_gpu_session_unlock();
    return rc;
}
