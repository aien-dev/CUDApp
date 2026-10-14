/* gpu_pfhost_test.c (OWN-30B W-L1G-D, 2026-10-14): router logits and MoE combine for the whole-prompt prefill
 * (omega_blackwell_codegen_pfhost.h). Both kernels must equal the host loops of the Rust driver BIT FOR BIT.
 *   --host-only  reference self-checks (packing, the -0.0 start, mutant references differ)
 *   --sim        kernels in the warp simulator bit-identical to the reference, both scheduling orders; mutants caught
 *   --chip       (under flock /tmp/aien-gb10.lock) through the device API at --tokens rows: bit-identical, mutants caught, medians
 * Verdict line last: "GPU_PFHOST: PASS|FAIL". */
#include "omega_blackwell_codegen_pfhost.h"
#include "omega_gpu_pfhost.h"
#include "omega_gpu_prefill.h"
#include "bw_warp_sim_tc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static float rand_unit(void) { return ((float)(rnd() & 0xffffff) / 16777216.0f) * 2.0f - 1.0f; }
/* router weights as the checkpoint has them: bf16 values widened to f32 */
static float rand_bf16w(void) { return u2f(f2u(rand_unit() * 0.08f) & 0xffff0000u); }
/* activations: RMSNorm-like magnitudes with some outliers and exact zeros */
static float rand_act(void) { float v = rand_unit() * 2.0f; int r = rnd() % 64; if (r == 0) v *= 40.0f; else if (r == 1) v = 0.0f; return v; }

/* ---------------- references (compiled with -ffp-contract=off: a * b then +, as Rust) ---------------- */
static void rl_ref(const float *x, const float *w, uint32_t rows, uint32_t k, uint32_t e, float *lg, int mutant) {
    const uint32_t kk = mutant == 2 ? k - 128u : k;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t ex = 0; ex < e; ex++) {
            float acc = -0.0f;
            for (uint32_t j = 0; j < kk; j++) acc = mutant == 1 ? fmaf(w[(size_t)ex * k + j], x[(size_t)r * k + j], acc) : acc + w[(size_t)ex * k + j] * x[(size_t)r * k + j];
            lg[(size_t)r * e + ex] = acc;
        }
}
static void mc_ref(const float *down, const uint32_t *inv, const float *wt, uint32_t rows, uint32_t h, uint32_t K, float *out, int mutant) {
    const uint32_t used = mutant == 2 ? K - 1u : K;
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < h; c++) {
            float acc = 0.0f;
            for (uint32_t s = 0; s < used; s++) {
                const float d = down[(size_t)inv[(size_t)r * K + s] * h + c], ww = wt[(size_t)r * K + s];
                acc = mutant == 1 ? fmaf(ww, d, acc) : acc + ww * d;
            }
            out[(size_t)r * h + c] = acc;
        }
}
static size_t diff_bits(const float *a, const float *b, size_t n) { size_t bad = 0; for (size_t i = 0; i < n; i++) if (f2u(a[i]) != f2u(b[i])) bad++; return bad; }

typedef struct { uint32_t rows, k, e, t, rp; float *x, *w, *wq, *ref; } RlProb;
static void rl_make(RlProb *p, uint32_t rows, uint32_t k, uint32_t e, uint32_t t) {
    p->rows = rows; p->k = k; p->e = e; p->t = t; p->rp = (rows + t - 1u) / t * t;
    p->x = malloc((size_t)p->rp * k * 4); p->w = malloc((size_t)e * k * 4);
    p->wq = malloc(omega_gpu_pf_router_packed_bytes(e, k)); p->ref = malloc((size_t)rows * e * 4);
    for (size_t i = 0; i < (size_t)p->rp * k; i++) p->x[i] = i < (size_t)rows * k ? rand_act() : u2f(0x7fc00000u); /* pad rows NaN: must not leak */
    for (size_t i = 0; i < (size_t)e * k; i++) p->w[i] = rand_bf16w();
    if (rows > 2) memset(p->x + 2 * (size_t)k, 0, k * 4); /* an all-zero row: every logit -0.0 + 0 products */
    omega_gpu_pf_router_pack(e, k, p->w, p->wq);
    rl_ref(p->x, p->w, rows, k, e, p->ref, 0);
}
static void rl_free(RlProb *p) { free(p->x); free(p->w); free(p->wq); free(p->ref); }

typedef struct { uint32_t rows, h, K; float *down, *wt, *ref; uint32_t *inv; } McProb;
static void mc_make(McProb *p, uint32_t rows, uint32_t h, uint32_t K) {
    p->rows = rows; p->h = h; p->K = K;
    const size_t g = (size_t)rows * K;
    p->down = malloc(g * h * 4); p->wt = malloc(g * 4); p->inv = malloc(g * 4); p->ref = malloc((size_t)rows * h * 4);
    for (size_t i = 0; i < g * h; i++) p->down[i] = rand_act() * 0.5f;
    for (size_t i = 0; i < g; i++) { p->wt[i] = (float)(rnd() & 0xffff) / 65536.0f * 0.4f; p->inv[i] = (uint32_t)i; }
    for (size_t i = g; i > 1; i--) { size_t j = rnd() % i; uint32_t t = p->inv[i - 1]; p->inv[i - 1] = p->inv[j]; p->inv[j] = t; } /* a permutation, like the expert sort */
    mc_ref(p->down, p->inv, p->wt, rows, h, K, p->ref, 0);
}
static void mc_free(McProb *p) { free(p->down); free(p->wt); free(p->inv); free(p->ref); }

/* ---------------- host-only ---------------- */
static int host_checks(void) {
    int fail = 0;
    /* packing: wq[g][ex][q] = w[ex][4g + q], pad group zero */
    { const uint32_t e = 32, k = 256; float *w = malloc(e * k * 4), *wq = malloc(omega_gpu_pf_router_packed_bytes(e, k));
      for (uint32_t i = 0; i < e * k; i++) w[i] = (float)i;
      omega_gpu_pf_router_pack(e, k, w, wq);
      size_t bad = 0;
      for (uint32_t g = 0; g < k / 4; g++) for (uint32_t ex = 0; ex < e; ex++) for (uint32_t q = 0; q < 4; q++) if (wq[(g * e + ex) * 4 + q] != w[ex * k + 4 * g + q]) bad++;
      for (uint32_t i = 0; i < e * 4; i++) if (f2u(wq[(k / 4) * e * 4 + i]) != 0) bad++;
      printf("host pack R4: %zu wrong\n", bad); if (bad) fail++;
      free(w); free(wq); }
    /* the sum starts at -0.0 (Rust f32 Sum): an all-zero row gives -0.0 + (+0.0 products) = +0.0, a row of -0.0 products stays -0.0 */
    { float z = -0.0f; z = z + 0.0f * -1.0f; printf("host -0.0 start: -0 + (-0) = %s\n", signbit(z) ? "-0 (ok)" : "+0 (WRONG)"); if (!signbit(z)) fail++; }
    /* the mutant references really differ */
    { RlProb p; rl_make(&p, 5, 256, 32, 4); float *m = malloc(5 * 32 * 4);
      for (int mu = 1; mu <= 2; mu++) { rl_ref(p.x, p.w, 5, 256, 32, m, mu); size_t d = diff_bits(m, p.ref, 5 * 32); printf("host router mutant %d reference differs in %zu of 160\n", mu, d); if (!d) fail++; }
      free(m); rl_free(&p); }
    { McProb p; mc_make(&p, 3, 128, 8); float *m = malloc(3 * 128 * 4);
      for (int mu = 1; mu <= 2; mu++) { mc_ref(p.down, p.inv, p.wt, 3, 128, 8, m, mu); size_t d = diff_bits(m, p.ref, 3 * 128); printf("host combine mutant %d reference differs in %zu of 384\n", mu, d); if (!d) fail++; }
      free(m); mc_free(&p); }
    return fail;
}

/* ---------------- simulator ---------------- */
static int rl_sim_one(const RlProb *p, int mutant, int order, float *lg, char *msg, size_t msgn) {
    OmegaRlSpec sp = { p->k, p->e, p->t, mutant };
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_rl_prog(&sp, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    const size_t lgb = (size_t)p->rp * p->e * 4, wqb = omega_gpu_pf_router_packed_bytes(p->e, p->k);
    uint8_t *smem = malloc(omega_rl_shared_bytes(&sp));
    static WsSim ws; memset(&ws, 0, sizeof ws);
    ws.p = &prog; ws.threads = p->e; ws.smem = smem; ws.smem_bytes = omega_rl_shared_bytes(&sp); ws.order = order;
    uint64_t a[3] = { (uintptr_t)p->x, (uintptr_t)p->wq, (uintptr_t)lg };
    for (int i = 0; i < 3; i++) { ws.cbank[(0x380 + 8 * i) / 4] = (uint32_t)a[i]; ws.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(a[i] >> 32); }
    ws.ranges[0].lo = (uintptr_t)p->x; ws.ranges[0].hi = (uintptr_t)p->x + (size_t)p->rp * p->k * 4;
    ws.ranges[1].lo = (uintptr_t)p->wq; ws.ranges[1].hi = (uintptr_t)p->wq + wqb;
    ws.ranges[2].lo = (uintptr_t)lg; ws.ranges[2].hi = (uintptr_t)lg + lgb;
    ws.nranges = 3;
    memset(lg, 0xee, lgb);
    int err = ws_run(&ws, p->rp / p->t, 1);
    if (err) snprintf(msg, msgn, "simulator %.150s", ws.msg);
    free(smem);
    return err ? -1 : 0;
}
static int mc_sim_one(const McProb *p, int mutant, int order, float *out, char *msg, size_t msgn) {
    OmegaMcSpec sp = { p->h, p->K, mutant };
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_mc_prog(&sp, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    const size_t g = (size_t)p->rows * p->K;
    static WsSim ws; memset(&ws, 0, sizeof ws);
    ws.p = &prog; ws.threads = p->h / 4; ws.order = order;
    uint64_t a[4] = { (uintptr_t)p->down, (uintptr_t)p->inv, (uintptr_t)p->wt, (uintptr_t)out };
    for (int i = 0; i < 4; i++) { ws.cbank[(0x380 + 8 * i) / 4] = (uint32_t)a[i]; ws.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(a[i] >> 32); }
    ws.ranges[0].lo = (uintptr_t)p->down; ws.ranges[0].hi = (uintptr_t)p->down + g * p->h * 4;
    ws.ranges[1].lo = (uintptr_t)p->inv; ws.ranges[1].hi = (uintptr_t)p->inv + g * 4;
    ws.ranges[2].lo = (uintptr_t)p->wt; ws.ranges[2].hi = (uintptr_t)p->wt + g * 4;
    ws.ranges[3].lo = (uintptr_t)out; ws.ranges[3].hi = (uintptr_t)out + (size_t)p->rows * p->h * 4;
    ws.nranges = 4;
    memset(out, 0xee, (size_t)p->rows * p->h * 4);
    int err = ws_run(&ws, p->rows, 1);
    if (err) snprintf(msg, msgn, "simulator %.150s", ws.msg);
    return err ? -1 : 0;
}
static int sim_checks(void) {
    int fail = 0; char msg[200];
    static const struct { uint32_t rows, k, e, t; } rc[] = { { 37, 2048, 128, 16 }, { 9, 256, 32, 4 }, { 21, 384, 64, 8 }, { 16, 2048, 128, 4 } };
    for (size_t c = 0; c < sizeof rc / sizeof rc[0]; c++) {
        RlProb p; rl_make(&p, rc[c].rows, rc[c].k, rc[c].e, rc[c].t);
        float *lg = malloc((size_t)p.rp * p.e * 4);
        for (int order = 0; order < 2; order++) {
            int r = rl_sim_one(&p, 0, order, lg, msg, sizeof msg);
            size_t bad = r ? 0 : diff_bits(lg, p.ref, (size_t)p.rows * p.e);
            printf("sim router k %u e %u t %u rows %u order %d: %s, %zu of %zu differ\n", p.k, p.e, p.t, p.rows, order, r ? msg : "ran", bad, (size_t)p.rows * p.e);
            if (r || bad) fail++;
        }
        if (c == 0) for (int mu = 1; mu <= 2; mu++) {
            int r = rl_sim_one(&p, mu, 0, lg, msg, sizeof msg);
            size_t bad = r ? 0 : diff_bits(lg, p.ref, (size_t)p.rows * p.e);
            printf("sim router mutant %d: %s\n", mu, r ? msg : bad ? "caught" : "NOT CAUGHT");
            if (r || !bad) fail++;
        }
        free(lg); rl_free(&p);
    }
    static const struct { uint32_t rows, h, K; } mc[] = { { 7, 2048, 8 }, { 5, 128, 4 }, { 3, 384, 16 } };
    for (size_t c = 0; c < sizeof mc / sizeof mc[0]; c++) {
        McProb p; mc_make(&p, mc[c].rows, mc[c].h, mc[c].K);
        float *out = malloc((size_t)p.rows * p.h * 4);
        for (int order = 0; order < 2; order++) {
            int r = mc_sim_one(&p, 0, order, out, msg, sizeof msg);
            size_t bad = r ? 0 : diff_bits(out, p.ref, (size_t)p.rows * p.h);
            printf("sim combine h %u top_k %u rows %u order %d: %s, %zu of %zu differ\n", p.h, p.K, p.rows, order, r ? msg : "ran", bad, (size_t)p.rows * p.h);
            if (r || bad) fail++;
        }
        if (c == 0) for (int mu = 1; mu <= 2; mu++) {
            int r = mc_sim_one(&p, mu, 0, out, msg, sizeof msg);
            size_t bad = r ? 0 : diff_bits(out, p.ref, (size_t)p.rows * p.h);
            printf("sim combine mutant %d: %s\n", mu, r ? msg : bad ? "caught" : "NOT CAUGHT");
            if (r || !bad) fail++;
        }
        free(out); mc_free(&p);
    }
    return fail;
}

/* ---------------- chip ---------------- */
static int alloc_fill(int host, size_t bytes, const void *src, OmegaGpuMem *m) {
    bytes = (bytes + 15u) & ~(size_t)15u;
    int rc = host ? omega_gpu_pf_alloc_host(bytes, m) : omega_gpu_pf_alloc_dev(bytes, m);
    if (rc) return rc;
    if (!src) return 0;
    if (host) { memcpy(m->cpu, src, bytes); return 0; }
    return omega_gpu_pf_upload(m, 0, src, bytes);
}
static int rl_chip(uint32_t rows, int reps) {
    int fail = 0;
    const uint32_t k = 2048, e = 128, t = omega_gpu_pf_router_rows();
    RlProb p; rl_make(&p, rows, k, e, t);
    const size_t lgb = (size_t)p.rp * e * 4;
    OmegaGpuMem xd = { 0 }, wd = { 0 }, ld = { 0 };
    /* as the driver: x and logits in HOST (GPU-uncached) buffers, the packed weights GPU-cached */
    int rc = alloc_fill(1, (size_t)p.rp * k * 4, p.x, &xd);
    if (!rc) rc = alloc_fill(0, omega_gpu_pf_router_packed_bytes(e, k), p.wq, &wd);
    if (!rc) rc = alloc_fill(1, lgb, NULL, &ld);
    uint64_t ns[32] = { 0 }; if (reps > 32) reps = 32; if (reps < 1) reps = 1;
    float *lg = malloc(lgb);
    for (int r = 0; r < reps && !rc; r++) { memset(ld.cpu, 0xee, lgb); rc = omega_gpu_pf_router_logits(k, e, rows, xd.va, wd.va, ld.va, &ns[r]); }
    if (rc) { printf("ROUTER chip rows %u: %s (%s)\n", rows, omega_gpu_pf_rc_name(rc), omega_gpu_session_last_error()); fail++; }
    else {
        memcpy(lg, ld.cpu, lgb);
        size_t bad = diff_bits(lg, p.ref, (size_t)rows * e);
        qsort(ns, (size_t)reps, 8, cmp_u64);
        printf("ROUTER chip k %u e %u t %u rows %u: %zu of %zu differ from the host loop; median %.3f ms\n", k, e, t, rows, bad, (size_t)rows * e, ns[reps / 2] / 1e6);
        if (bad) fail++;
        for (int mu = 1; mu <= 2; mu++) {
            omega_gpu_pf_router_test_set_mutant(mu);
            memset(ld.cpu, 0xee, lgb);
            int mrc = omega_gpu_pf_router_logits(k, e, rows, xd.va, wd.va, ld.va, NULL);
            omega_gpu_pf_router_test_set_mutant(0);
            memcpy(lg, ld.cpu, lgb);
            size_t mb = mrc ? 0 : diff_bits(lg, p.ref, (size_t)rows * e);
            printf("ROUTER chip mutant %d: %s\n", mu, mrc ? "launch failed" : mb ? "caught" : "NOT CAUGHT");
            if (mrc || !mb) fail++;
        }
    }
    free(lg); omega_gpu_pf_free(&xd); omega_gpu_pf_free(&wd); omega_gpu_pf_free(&ld); rl_free(&p);
    return fail;
}
static int mc_chip(uint32_t rows, int reps) {
    int fail = 0;
    const uint32_t h = 2048, K = 8;
    McProb p; mc_make(&p, rows, h, K);
    const size_t g = (size_t)rows * K, ob = (size_t)rows * h * 4;
    OmegaGpuMem dd = { 0 }, id = { 0 }, wd = { 0 }, od = { 0 };
    /* as the driver: down rows GPU-cached (the down GEMM writes them), inv / weights / out in HOST buffers */
    int rc = alloc_fill(0, g * h * 4, p.down, &dd);
    if (!rc) rc = alloc_fill(1, g * 4, p.inv, &id);
    if (!rc) rc = alloc_fill(1, g * 4, p.wt, &wd);
    if (!rc) rc = alloc_fill(1, ob, NULL, &od);
    uint64_t ns[32] = { 0 }; if (reps > 32) reps = 32; if (reps < 1) reps = 1;
    float *out = malloc(ob);
    for (int r = 0; r < reps && !rc; r++) { memset(od.cpu, 0xee, ob); rc = omega_gpu_pf_moe_combine(h, K, rows, dd.va, id.va, wd.va, od.va, &ns[r]); }
    if (rc) { printf("COMBINE chip rows %u: %s (%s)\n", rows, omega_gpu_pf_rc_name(rc), omega_gpu_session_last_error()); fail++; }
    else {
        memcpy(out, od.cpu, ob);
        size_t bad = diff_bits(out, p.ref, (size_t)rows * h);
        qsort(ns, (size_t)reps, 8, cmp_u64);
        printf("COMBINE chip h %u top_k %u rows %u: %zu of %zu differ from the host loop; median %.3f ms\n", h, K, rows, bad, (size_t)rows * h, ns[reps / 2] / 1e6);
        if (bad) fail++;
        for (int mu = 1; mu <= 2; mu++) {
            omega_gpu_pf_combine_test_set_mutant(mu);
            memset(od.cpu, 0xee, ob);
            int mrc = omega_gpu_pf_moe_combine(h, K, rows, dd.va, id.va, wd.va, od.va, NULL);
            omega_gpu_pf_combine_test_set_mutant(0);
            memcpy(out, od.cpu, ob);
            size_t mb = mrc ? 0 : diff_bits(out, p.ref, (size_t)rows * h);
            printf("COMBINE chip mutant %d: %s\n", mu, mrc ? "launch failed" : mb ? "caught" : "NOT CAUGHT");
            if (mrc || !mb) fail++;
        }
    }
    free(out); omega_gpu_pf_free(&dd); omega_gpu_pf_free(&id); omega_gpu_pf_free(&wd); omega_gpu_pf_free(&od); mc_free(&p);
    return fail;
}

int main(int argc, char **argv) {
    int host = 1, sim = 1, chip = 0, reps = 7; uint32_t tokens = 2048;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) sim = 0;
        else if (!strcmp(argv[i], "--sim")) host = 0;
        else if (!strcmp(argv[i], "--chip")) { chip = 1; host = 0; sim = 0; }
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    }
    int fail = 0;
    if (host) fail += host_checks();
    if (sim) fail += sim_checks();
    if (chip) {
        fail += rl_chip(tokens, reps); fail += rl_chip(37, 3);
        fail += mc_chip(tokens, reps); fail += mc_chip(7, 3);
    }
    printf("GPU_PFHOST: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
