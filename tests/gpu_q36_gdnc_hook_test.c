/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * gpu_q36_gdnc_hook_test (DRA-84): the q36 prefill hook of the chunked GDN scan, host only (no device).
 *   (a) default off: every env unset (and every value but "1") routes the legacy GDN path; no gdnc launch was recorded.
 *   (b) the refused gate combinations (-1, prefill_alloc returns BAD_ARGS).
 *   (c) omega_gdnc_plan's 25 words at 0x380 .. 0x3e0 from a synthetic request built by the hook's own row plumbing
 *       (omega_gpu_q36_pf_gdnc_rows + omega_gdnc_req), against literal offsets and values; bad strides / NULLs refused.
 *   (d) the snapshot count min(cap, floor(T / (64 every))) over T {63, 64, 128, 8192} x every {1, 2} x cap {0, 3, 8}
 *       (table from Wolfram), and the slots of a chunk at pos0 (omega_gpu_q36_pf_gdn_snap_slots), off a boundary included.
 * Verdict line last: "gpu_q36_gdnc_hook_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_gpu_q36_graph.h"
#include "omega_gpu_gdnc_api.h"
#include "omega_blackwell_codegen_gdn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_n, g_f;
#define CHECK(c, ...) do { g_n++; if (!(c)) { g_f++; printf("FAIL %s:%d ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const char *const ENVS[] = { "AIEN_OMEGA_Q36_PF_GDN", "AIEN_OMEGA_Q36_PF_GDN_CHUNK", "AIEN_OMEGA_Q36_PF_QMMA",
                                    "AIEN_OMEGA_Q36_PF_MUTANT" };

/* the route prefill_alloc computes from the process environment (same reads, same order) */
static int env_route(void) {
    const char *qe = getenv("AIEN_OMEGA_Q36_PF_QMMA");
    const char *m = getenv("AIEN_OMEGA_Q36_PF_MUTANT");
    return omega_gpu_q36_pf_gdn_route(getenv("AIEN_OMEGA_Q36_PF_GDN"), getenv("AIEN_OMEGA_Q36_PF_GDN_CHUNK"), m ? atoi(m) : 0,
                                      qe && qe[0] == '1' && qe[1] == 0);
}

static void test_default_off(void) {
    for (size_t i = 0; i < sizeof ENVS / sizeof ENVS[0]; i++) unsetenv(ENVS[i]);
    CHECK(env_route() == OMEGA_GPU_Q36_PF_GDN_MULTI, "(a) env unset: route %d, want MULTI (legacy multi-token)", env_route());
    /* QMMA alone does not turn the chunked scan on */
    setenv("AIEN_OMEGA_Q36_PF_QMMA", "1", 1);
    CHECK(env_route() == OMEGA_GPU_Q36_PF_GDN_MULTI, "(a) QMMA=1 only: route %d, want MULTI", env_route());
    /* anything but exactly "1" is off */
    static const char *const notone[] = { "", "0", "11", "true", "yes", " 1", "1 " };
    for (size_t i = 0; i < sizeof notone / sizeof notone[0]; i++) {
        setenv("AIEN_OMEGA_Q36_PF_GDN_CHUNK", notone[i], 1);
        CHECK(env_route() == OMEGA_GPU_Q36_PF_GDN_MULTI, "(a) PF_GDN_CHUNK='%s': route %d, want MULTI", notone[i], env_route());
    }
    unsetenv("AIEN_OMEGA_Q36_PF_GDN_CHUNK");
    setenv("AIEN_OMEGA_Q36_PF_GDN", "0", 1);
    CHECK(env_route() == OMEGA_GPU_Q36_PF_GDN_STEP, "(a) PF_GDN=0: route %d, want STEP", env_route());
    for (size_t i = 0; i < sizeof ENVS / sizeof ENVS[0]; i++) unsetenv(ENVS[i]);
    /* the positive control: the same reads do reach CHUNK when asked */
    setenv("AIEN_OMEGA_Q36_PF_QMMA", "1", 1); setenv("AIEN_OMEGA_Q36_PF_GDN_CHUNK", "1", 1);
    CHECK(env_route() == OMEGA_GPU_Q36_PF_GDN_CHUNK, "(a) control QMMA=1 CHUNK=1: route %d, want CHUNK", env_route());
    for (size_t i = 0; i < sizeof ENVS / sizeof ENVS[0]; i++) unsetenv(ENVS[i]);
    CHECK(omega_gpu_gdnc_records() == 0, "(a) gdnc launches recorded: %llu, want 0", (unsigned long long)omega_gpu_gdnc_records());
}

static void test_refusals(void) {
    /* {gdn_env, chunk_env, mutant, qmma, want} */
    static const struct { const char *gdn, *chunk; int mutant, qmma, want; } c[] = {
        { NULL, "1", 0, 0, -1 },                                /* CHUNK without QMMA */
        { "0", "1", 0, 1, -1 },                                 /* CHUNK with GDN=0 */
        { NULL, "1", 1, 1, -1 },                                /* CHUNK with mutant 1 */
        { NULL, "1", 3, 1, -1 },                                /* CHUNK with mutant 3 */
        { "0", "1", 9, 1, -1 },                                 /* both refusals at once */
        { NULL, NULL, 9, 1, -1 },                               /* mutant 9 without CHUNK */
        { NULL, NULL, 9, 0, -1 },
        { "0", NULL, 9, 0, -1 },
        { NULL, "0", 9, 1, -1 },
        { NULL, "1", 0, 1, OMEGA_GPU_Q36_PF_GDN_CHUNK },        /* accepted */
        { "1", "1", 0, 1, OMEGA_GPU_Q36_PF_GDN_CHUNK },
        { NULL, "1", 5, 1, OMEGA_GPU_Q36_PF_GDN_CHUNK },        /* mutant 5 runs on the chunk route (abstride 0) */
        { NULL, "1", 9, 1, OMEGA_GPU_Q36_PF_GDN_CHUNK },        /* the drop-decay control */
        { NULL, NULL, 1, 0, OMEGA_GPU_Q36_PF_GDN_MULTI },       /* legacy mutants stay legal off the chunk route */
        { "0", NULL, 3, 0, OMEGA_GPU_Q36_PF_GDN_STEP },
    };
    for (size_t i = 0; i < sizeof c / sizeof c[0]; i++) {
        const int r = omega_gpu_q36_pf_gdn_route(c[i].gdn, c[i].chunk, c[i].mutant, c[i].qmma);
        CHECK(r == c[i].want, "(b) gdn=%s chunk=%s mutant=%d qmma=%d: route %d, want %d", c[i].gdn ? c[i].gdn : "unset",
              c[i].chunk ? c[i].chunk : "unset", c[i].mutant, c[i].qmma, r, c[i].want);
    }
}

#define WORD(L, off) ((L).args[((off) - 0x380u) / 4u])
static uint64_t w64(const OmegaGdncLaunch *L, uint32_t off) {
    return (uint64_t)L->args[(off - 0x380u) / 4u] | (uint64_t)L->args[(off - 0x380u) / 4u + 1u] << 32;
}
static float wf(const OmegaGdncLaunch *L, uint32_t off) { float f; memcpy(&f, &L->args[(off - 0x380u) / 4u], 4); return f; }

static void test_plan(void) {
    OmegaGdnSpec sp; memset(&sp, 0, sizeof sp); sp.ratio = 2;
    OmegaGdnShape sh;
    CHECK(omega_gdn_shape(&sp, &sh) == 0, "(c) ratio 2 shape");
    /* the hook's row layout must hold what the kernel reads: q36 mix row = conv | vdim, ab = vh {a, b}, out = vdim */
    CHECK(OMEGA_GPU_Q36_GDN_QKV == sh.conv && OMEGA_GPU_Q36_GDN_MIX == sh.conv + sh.vdim && OMEGA_GPU_Q36_AB == 2u * sh.vh
          && OMEGA_GPU_Q36_MIXER_OUT == sh.vdim, "(c) q36 row constants vs gdn shape (conv %u vdim %u vh %u)", sh.conv, sh.vdim, sh.vh);
    const OmegaGpuGdnStateView v = { .layers = 30, .ratio = 2, .cached = 1, .state_bf16 = 0,
                                     .par_va = 0x7f0000100000ull, .tail_va = 0x7f0000900000ull, .state_va = 0x7f0010000000ull,
                                     .par_bytes = sh.par_bytes, .tail_bytes = sh.tail_bytes, .state_bytes = sh.state_bytes };
    const uint32_t sstride = (sh.state_bytes + sh.tail_bytes + 15u) & ~15u;
    const OmegaGdncSnapOut snap = { .va = 0x7f0100000000ull, .every_chunks = 2, .count = 8,
                                    .layer_stride_bytes = (uint64_t)8u * sstride, .snap_stride_bytes = sstride };
    const uint64_t mix = 0x7f0200000000ull, ab = 0x7f0300000000ull, out = 0x7f0400000000ull;
    OmegaGdncRows io;
    omega_gpu_q36_pf_gdnc_rows(8192, mix, ab, out, 0, 1e-6f, 2e-6f, &io);
    const uint32_t layer = 3;
    OmegaGdncReq r; OmegaGdncLaunch L;
    CHECK(omega_gdnc_req(&v, layer, &io, &snap, &r) == OMEGA_GPU_GDN_OK, "(c) req");
    CHECK(omega_gdnc_plan(&r, &L) == OMEGA_GPU_GDN_OK, "(c) plan of the hook's rows refused");
    CHECK(L.threads == 256u && L.grid_x == 2u && L.grid_y == 16u && L.n_args == 25u && L.shared_bytes == 43072u,
          "(c) shape %u threads %u x %u, %u args, %u shared", L.threads, L.grid_x, L.grid_y, L.n_args, L.shared_bytes);
    CHECK(w64(&L, 0x380) == mix, "(c) 0x380 qkvz %llx", (unsigned long long)w64(&L, 0x380));
    CHECK(w64(&L, 0x388) == ab, "(c) 0x388 ab %llx", (unsigned long long)w64(&L, 0x388));
    CHECK(w64(&L, 0x390) == v.par_va + layer * (uint64_t)sh.par_bytes, "(c) 0x390 par");
    CHECK(w64(&L, 0x398) == v.tail_va + layer * (uint64_t)sh.tail_bytes, "(c) 0x398 tail");
    CHECK(w64(&L, 0x3a0) == v.state_va + layer * (uint64_t)sh.state_bytes, "(c) 0x3a0 state");
    CHECK(w64(&L, 0x3a8) == out, "(c) 0x3a8 out %llx", (unsigned long long)w64(&L, 0x3a8));
    CHECK(w64(&L, 0x3b0) == snap.va + layer * snap.layer_stride_bytes, "(c) 0x3b0 snap %llx", (unsigned long long)w64(&L, 0x3b0));
    CHECK(wf(&L, 0x3b8) == 1e-6f && wf(&L, 0x3bc) == 2e-6f, "(c) 0x3b8/0x3bc eps %g %g", wf(&L, 0x3b8), wf(&L, 0x3bc));
    CHECK(wf(&L, 0x3c0) == 1.0f && wf(&L, 0x3c4) == 18446744073709551616.0f, "(c) 0x3c0/0x3c4 scales %g %g", wf(&L, 0x3c0), wf(&L, 0x3c4));
    CHECK(WORD(L, 0x3c8) == 8192u, "(c) 0x3c8 T %u", WORD(L, 0x3c8));
    CHECK(WORD(L, 0x3cc) == 24576u, "(c) 0x3cc xstride %u, want 24576 (12288 bf16)", WORD(L, 0x3cc));
    CHECK(WORD(L, 0x3d0) == 256u, "(c) 0x3d0 abstride %u, want 256 (64 f32)", WORD(L, 0x3d0));
    CHECK(WORD(L, 0x3d4) == 8192u, "(c) 0x3d4 ostride %u, want 8192 (4096 bf16)", WORD(L, 0x3d4));
    CHECK(WORD(L, 0x3d8) == 8u, "(c) 0x3d8 snap_every4 %u, want 8 (every 2 x 4)", WORD(L, 0x3d8));
    CHECK(WORD(L, 0x3dc) == 8u, "(c) 0x3dc snap_count %u, want 8", WORD(L, 0x3dc));
    CHECK(WORD(L, 0x3e0) == sstride, "(c) 0x3e0 snap_stride %u, want %u", WORD(L, 0x3e0), sstride);
    /* mutant 5 on the hook: abstride 0 (every token reads row 0), accepted */
    OmegaGdncRows io5; omega_gpu_q36_pf_gdnc_rows(8192, mix, ab, out, 5, 1e-6f, 2e-6f, &io5);
    OmegaGdncReq r5; OmegaGdncLaunch L5;
    CHECK(omega_gdnc_req(&v, layer, &io5, NULL, &r5) == OMEGA_GPU_GDN_OK && omega_gdnc_plan(&r5, &L5) == OMEGA_GPU_GDN_OK
          && WORD(L5, 0x3d0) == 0u && w64(&L5, 0x3b0) == 0u && WORD(L5, 0x3dc) == 0u && WORD(L5, 0x3d8) == 0u && WORD(L5, 0x3e0) == 0u,
          "(c) mutant 5 rows / no snapshots: abstride %u snap %llx", WORD(L5, 0x3d0), (unsigned long long)w64(&L5, 0x3b0));

    /* refusals: bad strides, NULL / zero pointers */
    struct { const char *what; OmegaGdncReq q; } bad[24]; size_t nb = 0;
#define BAD(w, stmt) do { bad[nb].what = w; bad[nb].q = r; { OmegaGdncReq *q = &bad[nb].q; stmt; } nb++; } while (0)
    BAD("xstride misaligned", q->io.xstride = 24584u);
    BAD("xstride below a row", q->io.xstride = 24576u - 16u);
    BAD("ostride misaligned", q->io.ostride = 8200u);
    BAD("ostride below a row", q->io.ostride = 8192u - 16u);
    BAD("abstride misaligned", q->io.abstride = 260u);
    BAD("abstride below a row", q->io.abstride = 248u);
    BAD("xstride / ostride swapped", { q->io.xstride = 8192u; q->io.ostride = 24576u; });
    BAD("snap stride below state + tail", q->snap_stride = (sh.state_bytes + sh.tail_bytes - 16u) & ~15u);
    BAD("snap stride misaligned", q->snap_stride = sstride + 8u);
    BAD("snap every4 0", q->snap_every4 = 0);
    BAD("snap va 0 with a count", q->snap_va = 0);
    BAD("qkvz va 0", q->io.qkvz_va = 0);
    BAD("ab va 0", q->io.ab_va = 0);
    BAD("out va 0", q->io.out_va = 0);
    BAD("par va 0", q->par_va = 0);
    BAD("tail va 0", q->tail_va = 0);
    BAD("state va 0", q->state_va = 0);
    BAD("qkvz misaligned", q->io.qkvz_va += 8u);
    BAD("tokens 0", q->io.tokens = 0);
    BAD("ratio 7", q->ratio = 7);
    BAD("eps NaN", q->io.eps_l2 = NAN);
#undef BAD
    for (size_t i = 0; i < nb; i++) {
        OmegaGdncLaunch Lb;
        CHECK(omega_gdnc_plan(&bad[i].q, &Lb) == OMEGA_GPU_GDN_BAD_ARGS, "(c) plan accepted: %s", bad[i].what);
    }
    CHECK(omega_gdnc_plan(NULL, &L) == OMEGA_GPU_GDN_BAD_ARGS, "(c) plan NULL req");
    CHECK(omega_gdnc_plan(&r, NULL) == OMEGA_GPU_GDN_BAD_ARGS, "(c) plan NULL launch");
    OmegaGdncReq rq;
    CHECK(omega_gdnc_req(NULL, 0, &io, NULL, &rq) == OMEGA_GPU_GDN_BAD_ARGS, "(c) req NULL view");
    CHECK(omega_gdnc_req(&v, 0, NULL, NULL, &rq) == OMEGA_GPU_GDN_BAD_ARGS, "(c) req NULL rows");
    CHECK(omega_gdnc_req(&v, 0, &io, NULL, NULL) == OMEGA_GPU_GDN_BAD_ARGS, "(c) req NULL out");
    CHECK(omega_gdnc_req(&v, 30, &io, NULL, &rq) == OMEGA_GPU_GDN_BAD_ARGS, "(c) req layer 30 of 30");
    OmegaGpuGdnStateView vb = v; vb.state_bf16 = 1;
    CHECK(omega_gdnc_req(&vb, 0, &io, NULL, &rq) == OMEGA_GPU_GDN_BAD_ARGS, "(c) req bf16 state");
    OmegaGdncSnapOut sb = snap; sb.layer_stride_bytes = 7u * (uint64_t)sstride;
    CHECK(omega_gdnc_req(&v, 0, &io, &sb, &rq) == OMEGA_GPU_GDN_BAD_ARGS, "(c) req overlapping layer slots");
    sb = snap; sb.every_chunks = 0;
    CHECK(omega_gdnc_req(&v, 0, &io, &sb, &rq) == OMEGA_GPU_GDN_BAD_ARGS, "(c) req every_chunks 0");
    sb = snap; sb.layer_stride_bytes += 8u;
    CHECK(omega_gdnc_req(&v, 0, &io, &sb, &rq) == OMEGA_GPU_GDN_BAD_ARGS, "(c) req layer stride misaligned");
    /* one token shares row 0: no row-size floor (the decode-shaped case) */
    OmegaGdncReq r1 = r; r1.io.tokens = 1; r1.io.xstride = 0; r1.io.ostride = 0; r1.snap_count = 0;
    CHECK(omega_gdnc_plan(&r1, &L) == OMEGA_GPU_GDN_OK, "(c) 1-token request with stride 0 refused");
}

static void test_snaps(void) {
    /* Wolfram: Table[{t,e,c,If[c==0,0,Min[c,Floor[t/(64 e)]]]},{t,{63,64,128,8192}},{e,{1,2}},{c,{0,3,8}}] */
    static const uint32_t tab[24][4] = {
        {63, 1, 0, 0}, {63, 1, 3, 0}, {63, 1, 8, 0}, {63, 2, 0, 0}, {63, 2, 3, 0}, {63, 2, 8, 0},
        {64, 1, 0, 0}, {64, 1, 3, 1}, {64, 1, 8, 1}, {64, 2, 0, 0}, {64, 2, 3, 0}, {64, 2, 8, 0},
        {128, 1, 0, 0}, {128, 1, 3, 2}, {128, 1, 8, 2}, {128, 2, 0, 0}, {128, 2, 3, 1}, {128, 2, 8, 1},
        {8192, 1, 0, 0}, {8192, 1, 3, 3}, {8192, 1, 8, 8}, {8192, 2, 0, 0}, {8192, 2, 3, 3}, {8192, 2, 8, 8} };
    OmegaGdnSpec sp; memset(&sp, 0, sizeof sp); sp.ratio = 2;
    OmegaGdnShape sh; omega_gdn_shape(&sp, &sh);
    const OmegaGpuGdnStateView v = { .layers = 30, .ratio = 2, .par_va = 0x100000ull, .tail_va = 0x900000ull, .state_va = 0x10000000ull,
                                     .par_bytes = sh.par_bytes, .tail_bytes = sh.tail_bytes, .state_bytes = sh.state_bytes };
    const uint32_t sstride = (sh.state_bytes + sh.tail_bytes + 15u) & ~15u;
    for (int i = 0; i < 24; i++) {
        const uint32_t n = omega_gdnc_snap_count(tab[i][0], tab[i][1], tab[i][2]);
        CHECK(n == tab[i][3], "(d) snap_count(T %u, every %u, cap %u) = %u, want %u", tab[i][0], tab[i][1], tab[i][2], n, tab[i][3]);
        /* the same count reaches the arg block through the hook rows and req */
        OmegaGdncRows io; omega_gpu_q36_pf_gdnc_rows(tab[i][0], 0x200000000ull, 0x300000000ull, 0x400000000ull, 0, 1e-6f, 1e-6f, &io);
        const OmegaGdncSnapOut s = { .va = 0x500000000ull, .every_chunks = tab[i][1], .count = tab[i][2],
                                     .layer_stride_bytes = (uint64_t)(tab[i][2] ? tab[i][2] : 1u) * sstride, .snap_stride_bytes = sstride };
        OmegaGdncReq r; OmegaGdncLaunch L;
        const int ok = omega_gdnc_req(&v, 0, &io, &s, &r) == OMEGA_GPU_GDN_OK && omega_gdnc_plan(&r, &L) == OMEGA_GPU_GDN_OK;
        CHECK(ok && WORD(L, 0x3dc) == tab[i][3] && WORD(L, 0x3d8) == (tab[i][3] ? 4u * tab[i][1] : 0u)
              && w64(&L, 0x3b0) == (tab[i][3] ? s.va : 0u),
              "(d) T %u every %u cap %u: plan ok %d count %u every4 %u", tab[i][0], tab[i][1], tab[i][2], ok, ok ? WORD(L, 0x3dc) : 0u,
              ok ? WORD(L, 0x3d8) : 0u);
    }
    /* chunk slots at pos0. Wolfram: f[p,e,c,t] = none when Mod[p, 64 e] != 0 or Quotient[p, 64 e] >= c, else
     * {base, c - base, Min[c - base, Floor[t / (64 e)]]}: f[32,1,8,8192] none, f[64,2,8,8192] none, f[128,1,8,8192] {2,6,6},
     * f[256,2,3,128] {2,1,1}, f[1024,1,8,64] none, f[192,1,3,8192] none, f[0,1,8,63] {0,8,0} */
    static const struct { uint32_t pos0, every, cap, T; int some; uint32_t base, count, written; } c[] = {
        { 32, 1, 8, 8192, 0, 0, 0, 0 }, { 64, 2, 8, 8192, 0, 0, 0, 0 }, { 128, 1, 8, 8192, 1, 2, 6, 6 },
        { 256, 2, 3, 128, 1, 2, 1, 1 }, { 1024, 1, 8, 64, 0, 0, 0, 0 }, { 192, 1, 3, 8192, 0, 0, 0, 0 }, { 0, 1, 8, 63, 1, 0, 8, 0 } };
    for (size_t i = 0; i < sizeof c / sizeof c[0]; i++) {
        const OmegaGdncSnapOut all = { .va = 0x500000000ull, .every_chunks = c[i].every, .count = c[i].cap,
                                       .layer_stride_bytes = (uint64_t)c[i].cap * sstride, .snap_stride_bytes = sstride };
        OmegaGdncSnapOut s; memset(&s, 0xa5, sizeof s);
        const int some = omega_gpu_q36_pf_gdn_snap_slots(&all, c[i].pos0, &s);
        CHECK(some == c[i].some, "(d) pos0 %u every %u cap %u: slots %d, want %d", c[i].pos0, c[i].every, c[i].cap, some, c[i].some);
        if (!some || !c[i].some) continue;
        CHECK(s.va == all.va + (uint64_t)c[i].base * sstride && s.count == c[i].count && s.every_chunks == c[i].every
              && s.snap_stride_bytes == sstride && s.layer_stride_bytes == all.layer_stride_bytes,
              "(d) pos0 %u: va +%llu count %u, want base %u count %u", c[i].pos0, (unsigned long long)((s.va - all.va) / sstride), s.count,
              c[i].base, c[i].count);
        OmegaGdncRows io; omega_gpu_q36_pf_gdnc_rows(c[i].T, 0x200000000ull, 0x300000000ull, 0x400000000ull, 0, 1e-6f, 1e-6f, &io);
        OmegaGdncReq r;
        CHECK(omega_gdnc_req(&v, 5, &io, &s, &r) == OMEGA_GPU_GDN_OK && r.snap_count == c[i].written
              && (!c[i].written || r.snap_va == s.va + 5u * all.layer_stride_bytes),
              "(d) pos0 %u T %u: written %u, want %u", c[i].pos0, c[i].T, r.snap_count, c[i].written);
    }
    OmegaGdncSnapOut s, none = { 0 };
    none.every_chunks = 1; none.count = 8;
    CHECK(omega_gpu_q36_pf_gdn_snap_slots(&none, 0, &s) == 0, "(d) va 0 = none");
    CHECK(omega_gpu_q36_pf_gdn_snap_slots(NULL, 0, &s) == 0, "(d) NULL = none");
}

int main(void) {
    test_default_off();
    test_refusals();
    test_plan();
    test_snaps();
    printf("gpu_q36_gdnc_hook_test: %d checks, %d failed: %s\n", g_n, g_f, g_f ? "FAIL" : "PASS");
    return g_f ? 1 : 0;
}
