/* omega_gpu_pfgemm.c: device-resident tensor-core FP8 grouped GEMM v2 (OWN-30B T3b). See omega_gpu_prefill.h. */
#include "omega_gpu_prefill.h"
#include "omega_blackwell_codegen_tc2.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_fmt.h"
#include <stdlib.h>
#include <string.h>

#define KIND_TC2 2u
static struct { uint32_t rt, warps, aqn; int mutant; } g;
void omega_gpu_pf_gemm_set_shape(uint32_t rt, uint32_t warps, uint32_t aqn) { g.rt = rt; g.warps = warps; g.aqn = aqn; }
void omega_gpu_pf_gemm_test_set_mutant(int m) { g.mutant = m; }
static uint32_t RT(void) { return g.rt ? g.rt : 8u; }
uint32_t omega_gpu_pf_gemm_bm(void) { return 16u * RT(); }

static uint32_t perm4(uint32_t q) { static const uint32_t o[4] = { 0, 2, 1, 3 }; return (q & ~3u) | o[q & 3u]; }

int omega_gpu_pf_weights_host_form(uint32_t E, uint32_t n, uint32_t k, const uint8_t *w, const uint16_t *sinv, uint32_t layout, uint8_t *pw, float *ratio) {
    if (!E || !n || !k || n % 128u || k % 128u || !w || !sinv || !pw || !ratio || layout > OMEGA_GPU_PF_SWIGLU) return OMEGA_GPU_PF_BAD_ARGS;
    if (layout == OMEGA_GPU_PF_SWIGLU && n % 256u) return OMEGA_GPU_PF_BAD_ARGS;
    const uint32_t KB = k / 128u, NB = n / 128u, I = n / 2u;
    float *sf = malloc((size_t)NB * KB * 4u);
    if (!sf) return OMEGA_GPU_PF_TOO_LARGE;
    int rc = OMEGA_GPU_PF_OK;
    for (uint32_t e = 0; e < E && rc == OMEGA_GPU_PF_OK; e++) {
        if (omega_fp8_pack_scales(sinv + (size_t)e * NB * KB, (size_t)NB * KB, sf, 0) != 0) { rc = OMEGA_GPU_PF_BAD_ARGS; break; }
        for (uint32_t p = 0; p < n; p++) {
            uint32_t src = p;
            if (layout == OMEGA_GPU_PF_SWIGLU) { uint32_t G = p / 16u, h = p % 16u, i = perm4(8u * G + h % 8u); src = h < 8u ? i : I + i; }
            memcpy(pw + ((size_t)e * n + p) * k, w + ((size_t)e * n + src) * k, k);
            if (p % 8u == 0) {
                /* pair layout: [e][p / 16][kb 0..KB][p / 8 % 2] (the kernel loads both column tiles of a warp with one LDG.64) */
                float *rt = ratio + ((size_t)e * (n / 16u) + p / 16u) * (KB + 1u) * 2u + (p / 8u) % 2u;
                const float *s = sf + (size_t)(src / 128u) * KB;
                rt[0] = 1.0f;
                for (uint32_t kb = 1; kb < KB; kb++) rt[2u * kb] = s[kb - 1] / s[kb];
                rt[2u * KB] = s[KB - 1];
            }
        }
    }
    free(sf);
    return rc;
}

int omega_gpu_pf_weights_upload(uint32_t E, uint32_t n, uint32_t k, const uint8_t *w, const uint16_t *sinv, uint32_t layout, OmegaPfWeights *out) {
    if (!out) return OMEGA_GPU_PF_BAD_ARGS;
    memset(out, 0, sizeof *out);
    const size_t wb = (size_t)E * n * k, rb = (size_t)E * (n / 8u) * (k / 128u + 1u) * 4u;
    uint8_t *pw = malloc(wb); float *ratio = malloc(rb + 16u);
    int rc = (!pw || !ratio) ? OMEGA_GPU_PF_TOO_LARGE : omega_gpu_pf_weights_host_form(E, n, k, w, sinv, layout, pw, ratio);
    const size_t rb16 = (rb + 15u) & ~(size_t)15u;
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_alloc_dev(wb, &out->w);
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_alloc_dev(rb16, &out->r);
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_upload(&out->w, 0, pw, wb & ~(size_t)15u);
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_upload(&out->r, 0, ratio, rb16);
    free(pw); free(ratio);
    if (rc != OMEGA_GPU_PF_OK) { omega_gpu_pf_free(&out->w); omega_gpu_pf_free(&out->r); memset(out, 0, sizeof *out); return rc; }
    out->experts = E; out->n = n; out->k = k; out->layout = layout;
    return OMEGA_GPU_PF_OK;
}
void omega_gpu_pf_weights_free(OmegaPfWeights *w) { if (!w) return; omega_gpu_pf_free(&w->w); omega_gpu_pf_free(&w->r); memset(w, 0, sizeof *w); }

typedef struct { uint64_t w_va, r_va; uint32_t row0, cnt; uint64_t pad; } Tile;
_Static_assert(sizeof(Tile) == OMEGA_TC2_TILE_BYTES, "tile entry");
uint32_t omega_gpu_pf_gemm_tiles(const OmegaPfWeights *w, const uint32_t *counts, void *tab, uint32_t max) {
    if (!w || !counts || !tab) return 0;
    const uint32_t bm = omega_gpu_pf_gemm_bm(), KB = w->k / 128u;
    Tile *t = tab; uint32_t nt = 0, row = 0;
    for (uint32_t e = 0; e < w->experts; e++) {
        for (uint32_t r = 0; r < counts[e]; r += bm) {
            if (nt >= max) return 0;
            t[nt].w_va = w->w.va + (uint64_t)e * w->n * w->k;
            t[nt].r_va = w->r.va + (uint64_t)e * (w->n / 8u) * (KB + 1u) * 4u;
            t[nt].row0 = row + r; t[nt].cnt = counts[e] - r < bm ? counts[e] - r : bm; t[nt].pad = 0; nt++;
        }
        row += counts[e];
    }
    return nt;
}

static int gen_tc2(const OmegaPfKey *key, void *prog) { OmegaTc2Spec s; memcpy(&s, key->words, sizeof s); return omega_blackwell_codegen_tc2_prog(&s, prog); }

int omega_gpu_pf_gemm(const OmegaPfWeights *w, uint64_t tiles_va, uint32_t ntiles, uint64_t rowidx_va, uint64_t x_va, uint64_t y_va, uint64_t *ns) {
    if (!w || !w->w.va || !tiles_va || !x_va || !y_va) return OMEGA_GPU_PF_BAD_ARGS;
    if (ntiles == 0) return OMEGA_GPU_PF_OK;
    if (ntiles > 65535u) return OMEGA_GPU_PF_TOO_LARGE;
    OmegaTc2Spec s; memset(&s, 0, sizeof s);
    s.k = w->k; s.rt = RT(); s.ct = 2; s.warps = g.warps ? g.warps : 16u; s.aqn = g.aqn ? g.aqn : 1u;
    s.epi = w->layout == OMEGA_GPU_PF_SWIGLU ? OMEGA_TC2_EPI_SWIGLU : OMEGA_TC2_EPI_F32; s.direct = rowidx_va == 0; s.n = w->n; s.mutant = g.mutant;
    if (!omega_tc2_spec_ok(&s)) return OMEGA_GPU_PF_BAD_ARGS;
    const uint32_t span = omega_tc2_span(&s);
    if (w->n % span) return OMEGA_GPU_PF_BAD_ARGS;
    OmegaPfKey key; memset(&key, 0, sizeof key); key.kind = KIND_TC2; memcpy(key.words, &s, sizeof s);
    _Static_assert(sizeof(OmegaTc2Spec) <= sizeof key.words, "spec fits the key");
    uint32_t args[OMEGA_TC2_ARGS_WORDS];
    const uint64_t va[4] = { tiles_va, rowidx_va, x_va, y_va };
    for (int i = 0; i < 4; i++) { args[2 * i] = (uint32_t)va[i]; args[2 * i + 1] = (uint32_t)(va[i] >> 32); }
    args[8] = w->n;
    omega_gpu_session_lock();
    int rc = omega_gpu_pf_open_locked();
    OmegaPfKernel k;
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_kernel_locked(&key, gen_tc2, &k);
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_launch_locked(&k, 32u * s.warps, w->n / span, ntiles, omega_tc2_shared_bytes(&s), args, OMEGA_TC2_ARGS_WORDS, ns);
    omega_gpu_session_unlock();
    return rc;
}

/* ---- native FP8 path (W-L1G): activation quantiser (omega_blackwell_codegen_qg.h) ---- */
#include "omega_blackwell_codegen_qg.h"
#define KIND_PQ 3u
static int pq_mutant;
void omega_gpu_pf_quant_test_set_mutant(int m) { pq_mutant = m; }
static int gen_pq(const OmegaPfKey *key, void *prog) { OmegaPqSpec s; memcpy(&s, key->words, sizeof s); return omega_blackwell_codegen_pq_prog(&s, prog); }
int omega_gpu_pf_quant(uint32_t k, uint32_t in_type, uint32_t rows, uint64_t x_va, uint64_t q_va, uint64_t s_va, uint64_t *ns) {
    OmegaPqSpec s; memset(&s, 0, sizeof s); s.k = k; s.in_type = in_type; s.mutant = pq_mutant;
    if (!omega_pq_spec_ok(&s) || !x_va || !q_va || !s_va) return OMEGA_GPU_PF_BAD_ARGS;
    if (rows == 0) return OMEGA_GPU_PF_OK;
    OmegaPfKey key; memset(&key, 0, sizeof key); key.kind = KIND_PQ; memcpy(key.words, &s, sizeof s);
    const uint32_t args[OMEGA_PQ_ARGS_WORDS] = { (uint32_t)x_va, (uint32_t)(x_va >> 32), (uint32_t)q_va, (uint32_t)(q_va >> 32), (uint32_t)s_va, (uint32_t)(s_va >> 32) };
    omega_gpu_session_lock();
    int rc = omega_gpu_pf_open_locked();
    OmegaPfKernel kern;
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_kernel_locked(&key, gen_pq, &kern);
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_launch_locked(&kern, 32u * (k / 128u), rows, 1, 0, args, OMEGA_PQ_ARGS_WORDS, ns);
    omega_gpu_session_unlock();
    return rc;
}
