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
 * gpu_fp8gemm_test: the prefill FP8 grouped GEMM (omega_gpu_fp8gemm_api.h), OWN-30B builder F item 26.
 *   --host-only   emulation vs the FP8 double oracle, permutation, codegen budgets and refusals (no device)
 *   --sim         warp-simulator parity (build with -DWS_MAX_THREADS=512): bit-identical to the host emulation of the
 *                 GEMV's summation order in both fragment orders, multi-segment, multi-pass, row clamping, mutants 1-3 caught
 *   (default)     the chip. Gate|up 128 x 1536 x 2048 and down 128 x 2048 x 768 over 64 tokens x top-8 routed rows, dense
 *                 qkv 5120 x 2048 and o 2048 x 4096 over 64 rows: every output row bit-identical to omega_gpu_fp8_gemv_f32
 *                 on the same row, a few rows against the host emulation and the double oracle, mutants caught,
 *                 timing medians (host caches evicted) vs the per-token grouped-GEMV path.
 *                 knobs: --reps N --mt N --warps N --ctas N --no-evict --tokens N --spin-us N --shape gateup|down|qkv|o|all --out receipt.json
 * Verdict line last: "gpu_fp8gemm_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_gpu_fp8gemm_api.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_session.h"
#include "bw_warp_sim_k.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static uint16_t bf16_rne(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t r = x + 0x7fffu + ((x >> 16) & 1u); return (uint16_t)(r >> 16); }
static float bf16_f32(uint16_t b) { uint32_t x = (uint32_t)b << 16; float f; memcpy(&f, &x, 4); return f; }
static uint8_t rand_e4m3(uint32_t *s) { uint8_t b = (uint8_t)(lcg(s) >> 24); if (omega_fp8_e4m3_is_nan(b)) b = (uint8_t)(b - 1); return b; }
static float f32_of_bits(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

typedef struct { uint32_t experts, n, k; uint8_t *w; uint16_t *s; } Weights;
static void weights_make(Weights *W, uint32_t experts, uint32_t n, uint32_t k, uint32_t seed) {
    W->experts = experts; W->n = n; W->k = k;
    size_t wb = (size_t)experts * n * k, sc = (size_t)experts * ((n + 127) / 128) * (k / 128);
    W->w = malloc(wb); W->s = malloc(sc * 2);
    uint32_t st = seed;
    for (size_t i = 0; i < wb; i++) W->w[i] = rand_e4m3(&st);
    for (size_t i = 0; i < sc; i++) W->s[i] = bf16_rne(frand(&st, 0.001f, 0.1f) * ((lcg(&st) & 1) ? 1.0f : -1.0f));
}
static void weights_free(Weights *W) { free(W->w); free(W->s); }

/* The kernel's exact f32 result for one (expert row, activation row): per lane, per K step a dot chain over the lane's LB
 * bytes (word, byte), acc = fma(dot, S', acc), then the SHFL.DOWN tree 16..1. w: the row's k bytes, s: its k/128 scale words
 * (bf16 bits of one 128-row block), x: natural-order activation row. */
static float emu_row(const uint8_t *w, const uint16_t *s, const float *x, uint32_t k, uint32_t lb) {
    float xp[16384]; omega_fp8_pack_x_f32(x, k, xp);
    float a[32]; const uint32_t steps = k / (32u * lb);
    for (unsigned lane = 0; lane < 32; lane++) {
        float acc = 0.0f;
        for (uint32_t st = 0; st < steps; st++) {
            float dot = 0.0f; const uint32_t c0 = st * 32u * lb + lane * lb;
            for (uint32_t b = 0; b < lb; b++) dot = fmaf(f32_of_bits(omega_fp8_e4m3_kernel_bits(w[c0 + b])), xp[c0 + b], dot);
            float sp = bf16_f32(s[c0 / 128u]) * 72057594037927936.0f; /* * 2^56 */
            acc = fmaf(dot, sp, acc);
        }
        a[lane] = acc;
    }
    for (unsigned o = 16; o >= 1; o /= 2) { float b[32]; for (unsigned l = 0; l < 32; l++) b[l] = a[l + o < 32 ? l + o : l]; for (unsigned l = 0; l < 32; l++) a[l] += b[l]; }
    return a[0];
}
/* full expected output of a row table against Weights W (emulation) */
static void expect_rows(const Weights *W, uint32_t M, const uint32_t *ids, const uint32_t *trow, const float *x, float *y, uint32_t lb) {
    const uint32_t n = W->n, k = W->k;
    for (uint32_t i = 0; i < M; i++) for (uint32_t r = 0; r < n; r++)
        y[(size_t)i * n + r] = emu_row(W->w + ((size_t)ids[i] * n + r) * k, W->s + ((size_t)ids[i] * ((n + 127) / 128) + r / 128) * (k / 128), x + (size_t)trow[i] * k, k, lb);
}
static float *make_x(uint32_t rows, uint32_t k, uint32_t seed) {
    float *x = malloc((size_t)rows * k * 4); uint32_t s = seed;
    for (size_t i = 0; i < (size_t)rows * k; i++) x[i] = frand(&s, -2.0f, 2.0f);
    return x;
}
/* a routed row table: `tokens` tokens x top-k distinct experts, sorted by expert (ties by token) */
static uint32_t make_table(uint32_t tokens, uint32_t topk, uint32_t experts, uint32_t seed, uint32_t *ids, uint32_t *trow, uint32_t *picks) {
    uint32_t M = tokens * topk, st = seed;
    uint32_t *cnt = calloc(experts, 4), (*pick)[16] = malloc((size_t)tokens * sizeof *pick);
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t j = 0; j < topk; j++) {
        uint32_t e; int dup;
        do { e = (lcg(&st) >> 8) % experts; dup = 0; for (uint32_t q = 0; q < j; q++) if (pick[t][q] == e) dup = 1; } while (dup);
        pick[t][j] = e; cnt[e]++;
    }
    uint32_t *pos = malloc(experts * 4), c = 0;
    for (uint32_t e = 0; e < experts; e++) { pos[e] = c; c += cnt[e]; }
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t j = 0; j < topk; j++) { uint32_t e = pick[t][j], p = pos[e]++; ids[p] = e; trow[p] = t; }
    for (uint32_t t = 0; t < tokens; t++) for (uint32_t j = 0; j < topk; j++) picks[t * topk + j] = pick[t][j];
    free(cnt); free(pick); free(pos);
    return M;
}

static void host_only(void) {
    printf("host-only: emulation, permutation, budgets\n");
    { Weights W; weights_make(&W, 1, 128, 2048, 5); float *x = make_x(1, 2048, 6);
      double yo[128], ys[128]; omega_fp8_gemv_oracle(W.w, W.s, 128, 2048, x, yo, ys);
      double worst = 0; for (uint32_t r = 0; r < 128; r++) { float e = emu_row(W.w + (size_t)r * 2048, W.s, x, 2048, 16); (void)e; }
      for (uint32_t r = 0; r < 128; r++) { float e = emu_row(W.w + (size_t)r * 2048, W.s + (r / 128) * 16, x, 2048, 16); double d = fabs((double)e - yo[r]) / (ys[r] + 1e-30); if (d > worst) worst = d; }
      CHECK(worst < 1e-5, "emulation vs the FP8 double oracle: worst rel %.2e", worst);
      printf("emulation vs oracle: worst relative error %.2e of the term scale\n", worst);
      weights_free(&W); free(x); }
    { /* permutation is a bijection of each row, scaled by 2^64 */
      float x[2048], p[2048]; for (unsigned i = 0; i < 2048; i++) x[i] = (float)(i + 1);
      omega_fp8gemm_permute_x(x, 2048, 16, p);
      char *seen = calloc(2048, 1); int ok = 1;
      for (unsigned i = 0; i < 2048; i++) { double v = (double)p[i] / 18446744073709551616.0; if (v < 1 || v > 2048 || seen[(int)v - 1]++) ok = 0; }
      CHECK(ok, "permutation of a 2048-row is a bijection"); free(seen);
      float p8[768]; float x8[768]; for (unsigned i = 0; i < 768; i++) x8[i] = (float)(i + 1);
      omega_fp8gemm_permute_x(x8, 768, 8, p8); ok = 1; seen = calloc(768, 1);
      for (unsigned i = 0; i < 768; i++) { double v = (double)p8[i] / 18446744073709551616.0; if (v < 1 || v > 768 || seen[(int)v - 1]++) ok = 0; }
      CHECK(ok, "permutation of a 768-row (lane 8) is a bijection"); free(seen);
      /* column c = lane L, word w, byte e of step 0 lands at (w*32 + L)*4 + e: c = 5 * 16 + 6 = 86 -> L 5, j 6 -> w 1, e 2 */
      CHECK(p[(1 * 32 + 5) * 4 + 2] / 18446744073709551616.0f == x[86], "column 86 sits at lane 5 word 1 element 2"); }
    struct { uint32_t k, lb, mt, w; } shp[] = { { 2048, 16, 8, 16 }, { 4096, 16, 8, 16 }, { 1024, 16, 8, 16 }, { 768, 8, 8, 16 }, { 768, 8, 8, 8 }, { 2048, 8, 8, 16 }, { 2048, 16, 4, 16 }, { 2048, 16, 8, 32 }, { 2048, 16, 8, 8 }, { 2048, 16, 4, 8 }, { 4096, 16, 4, 8 } };
    for (unsigned i = 0; i < sizeof shp / sizeof shp[0]; i++) for (int m = 0; m <= 3; m++) {
        OmegaFp8GemmSpec s = { shp[i].k, shp[i].lb, shp[i].mt, shp[i].w, m, 0 }; uint32_t n = 0, gp = 0;
        int rc = omega_fp8gemm_prog_stats(&s, &n, &gp);
        uint32_t decl = ((gp + 1 + 7) / 8) * 8;
        CHECK(rc == 0 && n <= 1024, "codegen k %u lb %u mt %u warps %u mutant %d: rc %d insns %u gprs %u", s.k, s.lane_bytes, s.mt, s.warps, m, rc, n, gp);
        if (m == 0 && rc == 0) printf("codegen k %4u lane %2u mt %2u warps %2u: %u instructions, %u registers (declared %u, register file %u of 65536)%s, shared %u B\n", s.k, s.lane_bytes, s.mt, s.warps, n, gp, decl, 32u * s.warps * decl, 32u * s.warps * decl <= 65536u ? "" : " DOES NOT FIT", omega_fp8gemm_shared_bytes(&s));
    }
    { OmegaFp8GemmSpec s = { 2048, 16, 16, 16, 0, 0 }; CHECK(omega_fp8gemm_prog_stats(&s, NULL, NULL) != 0, "mt 16 at lane 16 refused (32 KB of shared memory)"); }
    { OmegaFp8GemmSpec s = { 2048, 16, 6, 16, 0, 0 }; CHECK(omega_fp8gemm_prog_stats(&s, NULL, NULL) != 0, "mt 6 refused"); }
    { OmegaFp8GemmSpec s = { 2048, 16, 8, 16, 5, 0 }; CHECK(omega_fp8gemm_prog_stats(&s, NULL, NULL) != 0, "mutant 5 refused"); }
    { OmegaFp8GemmSpec s = { 2048, 16, 8, 16, 0, 2 }; CHECK(omega_fp8gemm_prog_stats(&s, NULL, NULL) != 0, "rowmap 2 refused"); }
    { OmegaFp8GemmSpec s = { 2048, 16, 8, 16, 0, 1 }; CHECK(omega_fp8gemm_prog_stats(&s, NULL, NULL) != 0, "rowmap at mt 8 refused (registers)"); }
    { uint32_t ks[] = { 2048, 4096, 512, 768 }; for (unsigned i = 0; i < 4; i++) {
        OmegaFp8GemmSpec s = { ks[i], ks[i] % 512 ? 8u : 16u, 4, 16, 0, 1 }; uint32_t n = 0, gp = 0;
        int rc = omega_fp8gemm_prog_stats(&s, &n, &gp);
        uint32_t decl = ((gp + 1 + 7) / 8) * 8;
        CHECK(rc == 0 && n <= 1024 && 32u * s.warps * decl <= 65536u, "codegen k %u mt 4 rowmap: rc %d insns %u gprs %u", ks[i], rc, n, gp);
        printf("codegen k %4u lane %2u mt  4 warps 16 rowmap: %u instructions, %u registers (declared %u, register file %u of 65536)\n", s.k, s.lane_bytes, n, gp, decl, 32u * s.warps * decl);
    } }
    CHECK(OMEGA_GEMM_ARGS_WORDS <= OMEGA_GPU_SESSION_MAX_ARGS, "argument words fit the constant bank window");
    uint32_t ids[3] = { 2, 1, 1 }, tr[3] = { 0, 0, 0 }; float xr[2048] = { 0 }, yo[4];
    CHECK(omega_gpu_fp8gemm_f32(NULL, 3, ids, tr, xr, 1, yo, NULL) == OMEGA_GPU_FP8GEMM_BAD_ARGS, "a NULL tensor is refused");
}

/* ------------------------------------------------------------- simulator */
static uint64_t g_sim_issues;
typedef struct { uint32_t k, n, lb, mt, w, grid; uint32_t counts[4]; uint32_t nseg; } SimPt;
static int sim_point(const SimPt *p, int mutant, uint32_t seed, char *msg, size_t msgn, size_t *bad) {
    static BlackwellIRProgram prog;
    OmegaFp8GemmSpec spec = { p->k, p->lb, p->mt, p->w, mutant, 0 };
    if (omega_blackwell_codegen_fp8gemm_prog(&spec, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    uint32_t M = 0; for (uint32_t g = 0; g < p->nseg; g++) M += p->counts[g];
    Weights W; weights_make(&W, p->nseg, p->n, p->k, seed);
    uint32_t ids[64], trow[64]; { uint32_t i = 0; for (uint32_t g = 0; g < p->nseg; g++) for (uint32_t c = 0; c < p->counts[g]; c++, i++) { ids[i] = g; trow[i] = (i * 5 + 1) % 16; } }
    float *x = make_x(16, p->k, seed + 1);
    float *yexp = malloc((size_t)M * p->n * 4);
    expect_rows(&W, M, ids, trow, x, yexp, p->lb);
    size_t wbytes = (size_t)p->nseg * p->n * p->k, sc = (size_t)p->nseg * ((p->n + 127) / 128) * (p->k / 128);
    uint8_t *wd = aligned_alloc(16, (wbytes + 15) & ~15ull); memcpy(wd, W.w, wbytes);
    float *sd = aligned_alloc(16, ((sc * 4) + 15) & ~15ull); omega_fp8_pack_scales(W.s, sc, sd, 0);
    float *xp = aligned_alloc(16, ((size_t)(M + p->mt) * p->k * 4 + 15) & ~15ull); memset(xp, 0, (size_t)(M + p->mt) * p->k * 4);
    float *tmp = malloc((size_t)p->k * 4);
    for (uint32_t i = 0; i < M; i++) { memcpy(tmp, x + (size_t)trow[i] * p->k, (size_t)p->k * 4); omega_fp8gemm_permute_x(tmp, p->k, p->lb, xp + (size_t)i * p->k); }
    typedef struct { uint64_t w, s; uint32_t row0, cnt, pad[2]; } Seg;
    Seg *seg = aligned_alloc(16, 64 * sizeof(Seg)); memset(seg, 0, 64 * sizeof(Seg));
    { uint32_t r0 = 0; for (uint32_t g = 0; g < p->nseg; g++) { seg[g].w = (uintptr_t)(wd + (size_t)g * p->n * p->k); seg[g].s = (uintptr_t)(sd + (size_t)g * ((p->n + 127) / 128) * (p->k / 128)); seg[g].row0 = r0; seg[g].cnt = p->counts[g]; r0 += p->counts[g]; } }
    float *y = aligned_alloc(16, ((size_t)M * p->n * 4 + 15) & ~15ull);
    uint32_t shared = omega_fp8gemm_shared_bytes(&spec);
    uint8_t *smem = aligned_alloc(16, shared);
    float *ycopy = NULL; int err = 0;
    for (int order = 0; order < 2 && !err; order++) {
        WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = 32u * p->w; s.order = order; s.smem = smem; s.smem_bytes = shared;
        for (size_t i = 0; i < (size_t)M * p->n; i++) y[i] = -999.0f;
        uint64_t A[3] = { (uintptr_t)seg, (uintptr_t)xp, (uintptr_t)y };
        for (unsigned i = 0; i < 3; i++) { s.cbank[(OMEGA_GEMM_ARG_SEG + 8 * i) / 4] = (uint32_t)A[i]; s.cbank[(OMEGA_GEMM_ARG_SEG + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
        s.cbank[OMEGA_GEMM_ARG_N / 4] = p->n; s.cbank[OMEGA_GEMM_ARG_STRIDE / 4] = p->grid * p->w;
        s.nranges = 5;
        s.ranges[0].lo = (uintptr_t)seg; s.ranges[0].hi = (uintptr_t)seg + 64 * sizeof(Seg);
        s.ranges[1].lo = (uintptr_t)wd; s.ranges[1].hi = (uintptr_t)wd + wbytes;
        s.ranges[2].lo = (uintptr_t)sd; s.ranges[2].hi = (uintptr_t)sd + sc * 4;
        s.ranges[3].lo = (uintptr_t)xp; s.ranges[3].hi = (uintptr_t)xp + (size_t)(M + p->mt) * p->k * 4;
        s.ranges[4].lo = (uintptr_t)y; s.ranges[4].hi = (uintptr_t)y + (size_t)M * p->n * 4;
        err = ws_run(&s, p->grid, p->nseg);
        if (err) { snprintf(msg, msgn, "order %d: %s (pc %d)", order, s.msg, s.err_pc); break; }
        g_sim_issues += s.warp_issues;
        size_t unwritten = 0; for (size_t i = 0; i < (size_t)M * p->n; i++) if (y[i] == -999.0f) unwritten++;
        if (unwritten) { err = -1; snprintf(msg, msgn, "%zu output words never written", unwritten); break; }
        if (order == 0) { ycopy = malloc((size_t)M * p->n * 4); memcpy(ycopy, y, (size_t)M * p->n * 4); }
        else if (memcmp(ycopy, y, (size_t)M * p->n * 4) != 0) { err = -1; snprintf(msg, msgn, "the fragment order changed the result"); }
    }
    if (!err) { size_t b = 0; for (size_t i = 0; i < (size_t)M * p->n; i++) if (memcmp(&yexp[i], &y[i], 4) != 0) b++; *bad = b; snprintf(msg, msgn, "%zu of %zu outputs differ from the emulation%s", b, (size_t)M * p->n, b ? "" : " (bit-identical)"); }
    free(ycopy); free(wd); free(sd); free(xp); free(tmp); free(seg); free(y); free(smem); free(yexp); free(x); weights_free(&W);
    return err;
}
/* The stroke's forms of the kernel (L312, 2026-10-21): an empty segment (cnt 0) in the table, and rowmap (x' rows through XMAP,
 * y rows through YMAP). empty_at >= 0 inserts a cnt 0 entry at that table index (grid_y = nseg + 1). With rowmap the x' buffer
 * holds the 16 permuted SOURCE rows and XMAP[i] = trow[i]; YMAP is a random permutation of the M sorted rows, so the mapped
 * result must equal the emulation moved through YMAP and differ from the unmoved one. budget: the simulator's step budget. */
static int sim_point2(const SimPt *p, int mutant, uint32_t seed, int rowmap, int empty_at, uint64_t budget, char *msg, size_t msgn, size_t *bad, int *moved) {
    static BlackwellIRProgram prog;
    OmegaFp8GemmSpec spec = { p->k, p->lb, p->mt, p->w, mutant, (uint32_t)rowmap };
    if (omega_blackwell_codegen_fp8gemm_prog(&spec, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    uint32_t M = 0; for (uint32_t g = 0; g < p->nseg; g++) M += p->counts[g];
    Weights W; weights_make(&W, p->nseg, p->n, p->k, seed);
    uint32_t ids[64], trow[64]; { uint32_t i = 0; for (uint32_t g = 0; g < p->nseg; g++) for (uint32_t c = 0; c < p->counts[g]; c++, i++) { ids[i] = g; trow[i] = (i * 5 + 1) % 16; } }
    float *x = make_x(16, p->k, seed + 1);
    float *yexp = malloc((size_t)M * p->n * 4);
    expect_rows(&W, M, ids, trow, x, yexp, p->lb);
    size_t wbytes = (size_t)p->nseg * p->n * p->k, sc = (size_t)p->nseg * ((p->n + 127) / 128) * (p->k / 128);
    uint8_t *wd = aligned_alloc(16, (wbytes + 15) & ~15ull); memcpy(wd, W.w, wbytes);
    float *sd = aligned_alloc(16, ((sc * 4) + 15) & ~15ull); omega_fp8_pack_scales(W.s, sc, sd, 0);
    const uint32_t xrows = rowmap ? 16u : M + p->mt;
    float *xp = aligned_alloc(16, ((size_t)xrows * p->k * 4 + 15) & ~15ull); memset(xp, 0, (size_t)xrows * p->k * 4);
    float *tmp = malloc((size_t)p->k * 4);
    if (rowmap) for (uint32_t r = 0; r < 16; r++) { memcpy(tmp, x + (size_t)r * p->k, (size_t)p->k * 4); omega_fp8gemm_permute_x(tmp, p->k, p->lb, xp + (size_t)r * p->k); }
    else for (uint32_t i = 0; i < M; i++) { memcpy(tmp, x + (size_t)trow[i] * p->k, (size_t)p->k * 4); omega_fp8gemm_permute_x(tmp, p->k, p->lb, xp + (size_t)i * p->k); }
    uint32_t *xmap = aligned_alloc(16, 256 * 4), *ymap = aligned_alloc(16, 256 * 4);
    memset(xmap, 0, 256 * 4); memset(ymap, 0, 256 * 4);
    { uint32_t s = seed * 7u + 3u; for (uint32_t i = 0; i < M; i++) { xmap[i] = trow[i]; ymap[i] = i; }
      for (uint32_t i = M; i > 1; i--) { uint32_t j = lcg(&s) % i, t = ymap[i - 1]; ymap[i - 1] = ymap[j]; ymap[j] = t; } }
    float *yexp2 = malloc((size_t)M * p->n * 4);
    for (uint32_t i = 0; i < M; i++) memcpy(yexp2 + (size_t)(rowmap ? ymap[i] : i) * p->n, yexp + (size_t)i * p->n, (size_t)p->n * 4);
    typedef struct { uint64_t w, s; uint32_t row0, cnt, pad[2]; } Seg;
    Seg *seg = aligned_alloc(16, 64 * sizeof(Seg)); memset(seg, 0, 64 * sizeof(Seg));
    uint32_t segs = 0;
    { uint32_t r0 = 0;
      for (uint32_t g = 0; g < p->nseg; g++) {
          if ((int)segs == empty_at) { seg[segs].w = (uintptr_t)wd; seg[segs].s = (uintptr_t)sd; seg[segs].row0 = r0; seg[segs].cnt = 0; segs++; }
          seg[segs].w = (uintptr_t)(wd + (size_t)g * p->n * p->k); seg[segs].s = (uintptr_t)(sd + (size_t)g * ((p->n + 127) / 128) * (p->k / 128));
          seg[segs].row0 = r0; seg[segs].cnt = p->counts[g]; r0 += p->counts[g]; segs++;
      }
      if ((int)segs == empty_at) { seg[segs].w = (uintptr_t)wd; seg[segs].s = (uintptr_t)sd; seg[segs].row0 = r0; seg[segs].cnt = 0; segs++; } }
    float *y = aligned_alloc(16, ((size_t)M * p->n * 4 + 15) & ~15ull);
    uint32_t shared = omega_fp8gemm_shared_bytes(&spec);
    uint8_t *smem = aligned_alloc(16, shared);
    float *ycopy = NULL; int err = 0;
    for (int order = 0; order < 2 && !err; order++) {
        WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = 32u * p->w; s.order = order; s.smem = smem; s.smem_bytes = shared; s.step_budget = budget;
        for (size_t i = 0; i < (size_t)M * p->n; i++) y[i] = -999.0f;
        uint64_t A[3] = { (uintptr_t)seg, (uintptr_t)xp, (uintptr_t)y };
        for (unsigned i = 0; i < 3; i++) { s.cbank[(OMEGA_GEMM_ARG_SEG + 8 * i) / 4] = (uint32_t)A[i]; s.cbank[(OMEGA_GEMM_ARG_SEG + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
        s.cbank[OMEGA_GEMM_ARG_N / 4] = p->n; s.cbank[OMEGA_GEMM_ARG_STRIDE / 4] = p->grid * p->w;
        s.cbank[OMEGA_GEMM_ARG_XMAP / 4] = (uint32_t)(uintptr_t)xmap; s.cbank[OMEGA_GEMM_ARG_XMAP / 4 + 1] = (uint32_t)((uintptr_t)xmap >> 32);
        s.cbank[OMEGA_GEMM_ARG_YMAP / 4] = (uint32_t)(uintptr_t)ymap; s.cbank[OMEGA_GEMM_ARG_YMAP / 4 + 1] = (uint32_t)((uintptr_t)ymap >> 32);
        s.nranges = 7;
        s.ranges[0].lo = (uintptr_t)seg; s.ranges[0].hi = (uintptr_t)seg + 64 * sizeof(Seg);
        s.ranges[1].lo = (uintptr_t)wd; s.ranges[1].hi = (uintptr_t)wd + wbytes;
        s.ranges[2].lo = (uintptr_t)sd; s.ranges[2].hi = (uintptr_t)sd + sc * 4;
        s.ranges[3].lo = (uintptr_t)xp; s.ranges[3].hi = (uintptr_t)xp + (size_t)xrows * p->k * 4;
        s.ranges[4].lo = (uintptr_t)y; s.ranges[4].hi = (uintptr_t)y + (size_t)M * p->n * 4;
        s.ranges[5].lo = (uintptr_t)xmap; s.ranges[5].hi = (uintptr_t)xmap + (M + p->mt) * 4;
        s.ranges[6].lo = (uintptr_t)ymap; s.ranges[6].hi = (uintptr_t)ymap + (M + p->mt) * 4;
        err = ws_run(&s, p->grid, segs);
        if (err) { snprintf(msg, msgn, "order %d: %s (pc %d)", order, s.msg, s.err_pc); break; }
        g_sim_issues += s.warp_issues;
        size_t unwritten = 0; for (size_t i = 0; i < (size_t)M * p->n; i++) if (y[i] == -999.0f) unwritten++;
        if (unwritten) { err = -1; snprintf(msg, msgn, "%zu output words never written", unwritten); break; }
        if (order == 0) { ycopy = malloc((size_t)M * p->n * 4); memcpy(ycopy, y, (size_t)M * p->n * 4); }
        else if (memcmp(ycopy, y, (size_t)M * p->n * 4) != 0) { err = -1; snprintf(msg, msgn, "the fragment order changed the result"); }
    }
    if (!err) {
        size_t b = 0; for (size_t i = 0; i < (size_t)M * p->n; i++) if (memcmp(&yexp2[i], &y[i], 4) != 0) b++;
        *bad = b; if (moved) *moved = memcmp(yexp, y, (size_t)M * p->n * 4) != 0;
        snprintf(msg, msgn, "%zu of %zu outputs differ from the emulation%s%s", b, (size_t)M * p->n, b ? "" : " (bit-identical)", rowmap ? (memcmp(yexp, y, (size_t)M * p->n * 4) ? ", rows moved by YMAP" : ", YMAP moved nothing") : "");
    }
    free(ycopy); free(wd); free(sd); free(xp); free(tmp); free(seg); free(y); free(smem); free(yexp); free(yexp2); free(x); free(xmap); free(ymap); weights_free(&W);
    return err;
}

static void sim_battery(void) {
    SimPt pts[] = {
        { 2048, 40, 16, 8, 16, 2, { 1, 5, 9 }, 3 },           /* lane 16, KS 4 (2 iterations), 2 passes in segment 3, row clamping (n 40 < 2 x 32 x 1) */
        { 4096, 20, 16, 8, 16, 1, { 3, 8 }, 2 },              /* KS 8: 4 iterations */
        { 1024, 33, 16, 8, 16, 2, { 2, 1 }, 2 },              /* KS 2: one iteration, no tail prefetch */
        { 768, 130, 8, 8, 16, 2, { 4, 17 }, 2 },             /* lane 8, KS 3 (odd, unrolled), 2 row blocks of scales, 2 passes */
        { 768, 24, 8, 8, 8, 1, { 1, 8, 2 }, 3 },            /* lane 8 at 8 warps */
        { 2048, 16, 16, 4, 16, 1, { 6, 4 }, 2 },              /* mt 4 */
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        char msg[300]; size_t bad = 1;
        int err = sim_point(&pts[i], 0, 31 + (uint32_t)i, msg, sizeof msg, &bad);
        CHECK(!err && bad == 0, "sim k %u n %u lane %u mt %u warps %u grid %u: %s", pts[i].k, pts[i].n, pts[i].lb, pts[i].mt, pts[i].w, pts[i].grid, msg);
        printf("sim k %u n %u lane %u mt %u warps %u grid %u: %s %s\n", pts[i].k, pts[i].n, pts[i].lb, pts[i].mt, pts[i].w, pts[i].grid, err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
    }
    for (int m = 1; m <= 3; m++) {
        char msg[300]; size_t bad = 0;
        int err = sim_point(&pts[0], m, 77, msg, sizeof msg, &bad);
        CHECK(!err && bad != 0, "sim mutant %d must be caught: %s", m, msg);
        printf("sim mutant %d: %s %s\n", m, err ? "ERROR" : bad ? "caught" : "NOT CAUGHT", msg);
    }
    /* the stroke's forms (L312): empty segments and row maps */
    {
        char msg[300]; size_t bad = 1; int moved = 0;
        /* RED first: mutant 4 (the guard dropped) on an empty segment must run out of steps */
        int err = sim_point2(&pts[5], 4, 99, 0, 1, 2000000ull, msg, sizeof msg, &bad, NULL);
        /* the runaway pass loop is caught either way: the step budget, or a read past the x' rows (the simulator checks every address) */
        CHECK(err != 0 && (strstr(msg, "step budget") || strstr(msg, "outside the declared buffers")), "sim mutant 4 (no empty-segment guard) must be caught: %s", msg);
        printf("sim empty segment, mutant 4 (guard dropped): %s %s\n", err ? "caught" : "NOT CAUGHT", msg);
        /* GREEN: the guard; an empty segment first, in the middle, last */
        for (int at = 0; at <= 2; at++) {
            err = sim_point2(&pts[0], 0, 41 + (uint32_t)at, 0, at + (at == 2 ? 1 : 0), 0, msg, sizeof msg, &bad, NULL);
            CHECK(!err && bad == 0, "sim empty segment at %d: %s", at, msg);
            printf("sim k %u empty segment at index %d: %s %s\n", pts[0].k, at + (at == 2 ? 1 : 0), err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
        }
        /* rowmap: x' rows through XMAP (16 source rows), y rows through YMAP (a random permutation); with and without an empty segment */
        const SimPt rmpts[] = {
            { 2048, 40, 16, 4, 16, 2, { 1, 5, 9 }, 3 },   /* 3 passes in segment 3 (mt 4), row clamping */
            { 4096, 20, 16, 4, 16, 1, { 3, 8 }, 2 },      /* KS 8, + an empty segment */
            { 512, 40, 16, 4, 16, 2, { 1, 5, 9 }, 3 },    /* the down projection: KS 1 (unrolled) */
            { 768, 24, 8, 4, 8, 1, { 1, 8, 2 }, 3 },      /* lane 8 at 8 warps */
            { 2048, 16, 16, 4, 16, 1, { 6, 4 }, 2 },
        };
        for (size_t i = 0; i < sizeof rmpts / sizeof rmpts[0]; i++) {
            const SimPt *p = &rmpts[i];
            err = sim_point2(p, 0, 61 + (uint32_t)i, 1, i == 1 ? 1 : -1, 0, msg, sizeof msg, &bad, &moved);
            CHECK(!err && bad == 0 && moved, "sim rowmap k %u mt %u%s: %s", p->k, p->mt, i == 1 ? " + empty segment" : "", msg);
            printf("sim rowmap k %u n %u mt %u warps %u%s: %s %s\n", p->k, p->n, p->mt, p->w, i == 1 ? " + empty segment" : "", err ? "ERROR" : bad ? "MISMATCH" : "ok", msg);
        }
        for (int m = 1; m <= 3; m++) {
            err = sim_point2(&rmpts[0], m, 77, 1, -1, 0, msg, sizeof msg, &bad, NULL);
            CHECK(!err && bad != 0, "sim rowmap mutant %d must be caught: %s", m, msg);
            printf("sim rowmap mutant %d: %s %s\n", m, err ? "ERROR" : bad ? "caught" : "NOT CAUGHT", msg);
        }
    }
    printf("sim: %" PRIu64 " warp issues in total\n", g_sim_issues);
}

/* ------------------------------------------------------------- chip */
static int g_evict = 1;
static void evict_host_caches(void) {
    static uint8_t *buf; const size_t sz = 512u << 20; static uint8_t pass;
    if (!buf) buf = malloc(sz);
    if (!buf) return;
    memset(buf, ++pass, sz);
    volatile uint64_t sum = 0; for (size_t i = 0; i < sz; i += 64) sum += buf[i];
}
static double med(uint64_t *v, uint32_t n) { qsort(v, n, 8, cmp_u64); return (double)v[n / 2]; }

typedef struct { const char *name; uint32_t experts, n, k, tokens, topk; } Shape;
static FILE *g_out; static int g_first = 1;

static void run_shape(const Shape *S, uint32_t reps, uint32_t seed) {
    printf("== %s: %u experts, n %u, k %u, %u tokens x top-%u\n", S->name, S->experts, S->n, S->k, S->tokens, S->topk);
    Weights W; weights_make(&W, S->experts, S->n, S->k, seed);
    OmegaGpuFp8Tensor *t = NULL;
    double t0 = now_ms();
    int rc = omega_gpu_fp8_tensor_upload(S->experts, S->n, S->k, W.w, W.s, &t);
    printf("upload %.0f MB: rc=%s %.0f ms\n", (double)S->experts * S->n * S->k / 1e6, omega_gpu_fp8_rc_name(rc), now_ms() - t0);
    CHECK(rc == 0, "%s upload", S->name);
    if (rc) { weights_free(&W); return; }
    const uint32_t M = S->tokens * S->topk;
    uint32_t *ids = malloc(M * 4), *trow = malloc(M * 4);
    uint32_t *picks = malloc((size_t)M * 4);
    make_table(S->tokens, S->topk, S->experts, seed + 7, ids, trow, picks);
    uint32_t xrows = S->experts > 1 ? (S->k == 768 ? M : S->tokens) : S->tokens;
    if (S->experts > 1 && S->k == 768) for (uint32_t i = 0; i < M; i++) trow[i] = i; /* down: one activation row per routed row */
    float *x = make_x(xrows, S->k, seed + 3);
    float *y = malloc((size_t)M * S->n * 4), *yref = malloc((size_t)M * S->n * 4), *y2 = malloc((size_t)M * S->n * 4);
    OmegaGpuFp8GemmInfo info;
    rc = omega_gpu_fp8gemm_f32(t, M, ids, trow, x, xrows, y, &info);
    CHECK(rc == 0, "%s gemm: %s %s", S->name, omega_gpu_fp8gemm_rc_name(rc), rc ? omega_gpu_fp8gemm_last_error() : "");
    if (rc == 0) {
        /* reference: the existing GEMV, one call per row */
        double tr = now_ms(); size_t bad = 0, first = (size_t)-1;
        for (uint32_t i = 0; i < M; i++) {
            int r2 = omega_gpu_fp8_gemv_f32(t, ids[i], x + (size_t)trow[i] * S->k, yref + (size_t)i * S->n, NULL);
            if (r2) { printf("gemv reference failed at row %u: %s\n", i, omega_gpu_fp8_last_error()); bad = (size_t)-1; break; }
        }
        if (bad == 0) for (size_t i = 0; i < (size_t)M * S->n; i++) if (memcmp(&y[i], &yref[i], 4) != 0) { if (!bad) first = i; bad++; }
        CHECK(bad == 0, "%s: %zu of %zu outputs differ from the per-row GEMV (first %zd)", S->name, bad, (size_t)M * S->n, (ssize_t)first);
        printf("parity vs per-row GEMV (%u calls, %.0f ms): %zu of %zu outputs differ%s\n", M, now_ms() - tr, bad, (size_t)M * S->n, bad ? "" : " (bit-identical)");
        /* host emulation + oracle on a few rows */
        size_t eb = 0; double worst = 0; uint32_t lb = info.lane_bytes;
        for (uint32_t q = 0; q < 4; q++) {
            uint32_t i = (q * 977u + 3u) % M, r = (q * 331u + 17u) % S->n;
            float e = emu_row(W.w + ((size_t)ids[i] * S->n + r) * S->k, W.s + ((size_t)ids[i] * ((S->n + 127) / 128) + r / 128) * (S->k / 128), x + (size_t)trow[i] * S->k, S->k, lb);
            if (memcmp(&e, &y[(size_t)i * S->n + r], 4) != 0) eb++;
            double yo[128], ys[128]; /* oracle over the 128-row block holding r */
            uint32_t b0 = (r / 128) * 128;
            omega_fp8_gemv_oracle(W.w + ((size_t)ids[i] * S->n + b0) * S->k, W.s + ((size_t)ids[i] * ((S->n + 127) / 128) + r / 128) * (S->k / 128), 128, S->k, x + (size_t)trow[i] * S->k, yo, ys);
            double d = fabs((double)y[(size_t)i * S->n + r] - yo[r - b0]) / (ys[r - b0] + 1e-30); if (d > worst) worst = d;
        }
        CHECK(eb == 0 && worst < 1e-5, "%s: host emulation mismatches %zu, oracle worst %.2e", S->name, eb, worst);
        printf("4 sampled outputs: %zu differ from the host emulation, worst relative error vs the double oracle %.2e\n", eb, worst);
        rc = omega_gpu_fp8gemm_f32(t, M, ids, trow, x, xrows, y2, NULL);
        CHECK(rc == 0 && memcmp(y, y2, (size_t)M * S->n * 4) == 0, "%s: second call bit-identical", S->name);
        printf("second call: %s\n", rc == 0 && memcmp(y, y2, (size_t)M * S->n * 4) == 0 ? "bit-identical" : "DIFFERS");
        /* mutants */
        for (int m = 1; m <= 3; m++) {
            omega_gpu_fp8gemm_test_set_mutant(m);
            rc = omega_gpu_fp8gemm_f32(t, M, ids, trow, x, xrows, y2, NULL);
            size_t b = 0; if (rc == 0) for (size_t i = 0; i < (size_t)M * S->n; i++) if (memcmp(&y[i], &y2[i], 4) != 0) b++;
            if (rc != 0) printf("chip mutant %d: call returned %s: %s\n", m, omega_gpu_fp8gemm_rc_name(rc), omega_gpu_fp8gemm_last_error());
            int caught = rc == 0 && b != 0;
            if (m == 2 && info.k / (32u * info.lane_bytes) < 2) caught = 1; /* single-step k: mutant 2 has no effect by construction */
            CHECK(caught, "%s: chip mutant %d must be caught", S->name, m);
            printf("chip mutant %d: %s (%zu of %zu outputs differ)\n", m, caught ? "caught" : "NOT CAUGHT", b, (size_t)M * S->n);
            if (g_out) fprintf(g_out, "%s{\"point\":\"%s mutant %d\",\"caught\":%d}", g_first ? "" : ",", S->name, m, caught), g_first = 0;
        }
        omega_gpu_fp8gemm_test_set_mutant(0);
        /* timing: the new path vs the per-token path (grouped GEMV, 8 experts per token for the MoE shapes; one GEMV per row for dense) */
        uint64_t *ce = malloc(reps * 8), *cc = malloc(reps * 8); uint32_t done = 0;
        for (uint32_t i = 0; i < reps; i++) {
            if (g_evict) evict_host_caches();
            { int r3 = omega_gpu_fp8gemm_f32(t, M, ids, trow, x, xrows, y2, &info); if (r3 != 0) { printf("timing call %u: %s: %s\n", i, omega_gpu_fp8gemm_rc_name(r3), omega_gpu_fp8gemm_last_error()); break; } }
            ce[done] = info.elapsed_ns; cc[done] = info.call_ns; done++;
        }
        CHECK(done == reps, "%s timing loop completed %u of %u", S->name, done, reps);
        double mc = done ? med(ce, done) : 0, ma = done ? med(cc, done) : 0;
        printf("GEMM (medians of %u, evict %d): mt %u warps %u ctas %u x %u segments, %u rows, up to %u passes: chip %.1f us, call %.1f us; streams %.0f MB (distinct experts %.0f MB) = %.1f GB/s on streamed, %.1f GB/s on distinct\n",
               done, g_evict, info.mt, info.warps_per_cta, info.grid_x, info.segments, M, info.passes_max, mc / 1e3, ma / 1e3, info.streamed_bytes / 1e6, info.weight_bytes / 1e6, info.streamed_bytes / mc, info.weight_bytes / mc);
        /* per-token baseline: for the MoE shapes one grouped GEMV (topk experts) per token; the sorted table is rebuilt per token */
        uint64_t *bc = malloc(reps * 8), *bw = malloc(reps * 8); uint32_t bd = 0;
        for (uint32_t i = 0; i < reps; i++) {
            if (g_evict) evict_host_caches();
            uint64_t sum_e = 0, sum_c = 0; int ok = 1;
            for (uint32_t tk = 0; tk < S->tokens && ok; tk++) {
                OmegaGpuFp8Info fi; int r2;
                if (S->experts == 1) r2 = omega_gpu_fp8_gemv_f32(t, 0, x + (size_t)tk * S->k, yref, &fi);
                else if (S->k == 768) r2 = omega_gpu_fp8_gemv_grouped_f32(t, S->topk, picks + (size_t)tk * S->topk, x + (size_t)tk * S->topk * S->k, S->topk, yref, &fi);
                else r2 = omega_gpu_fp8_gemv_grouped_f32(t, S->topk, picks + (size_t)tk * S->topk, x + (size_t)tk * S->k, 1, yref, &fi);
                if (r2) ok = 0; else { sum_e += fi.elapsed_ns; sum_c += fi.call_ns; }
            }
            if (!ok) break;
            bc[bd] = sum_e; bw[bd] = sum_c; bd++;
        }
        double mb = bd ? med(bc, bd) : 0, mbw = bd ? med(bw, bd) : 0;
        if (bd) printf("per-token grouped GEMV baseline (%u tokens, medians of %u): chip %.1f us, call %.1f us => GEMM is %.2fx faster (chip), %.2fx (call)\n", S->tokens, bd, mb / 1e3, mbw / 1e3, mb / mc, mbw / ma);
        printf("per token: GEMM chip %.1f us/token, call %.1f us/token\n", mc / 1e3 / S->tokens, ma / 1e3 / S->tokens);
        if (g_out) fprintf(g_out, "%s{\"point\":\"%s timing\",\"rows\":%u,\"segments\":%u,\"mt\":%u,\"warps\":%u,\"ctas\":%u,\"passes_max\":%u,\"reps\":%u,\"chip_median_ns\":%.0f,\"call_median_ns\":%.0f,\"streamed_bytes\":%" PRIu64 ",\"distinct_bytes\":%" PRIu64 ",\"gbps_streamed\":%.2f,\"baseline_chip_median_ns\":%.0f,\"baseline_call_median_ns\":%.0f,\"parity_bad\":%zu}",
                           g_first ? "" : ",", S->name, M, info.segments, info.mt, info.warps_per_cta, info.grid_x, info.passes_max, done, mc, ma, info.streamed_bytes, info.weight_bytes, info.streamed_bytes / mc, mb, mbw, bad), g_first = 0;
        free(ce); free(cc); free(bc); free(bw);
    }
    omega_gpu_fp8_tensor_free(t);
    free(ids); free(trow); free(picks); free(x); free(y); free(yref); free(y2); weights_free(&W);
}

int main(int argc, char **argv) {
    int host = 0, sim = 0; const char *out_path = NULL, *shape = "all"; uint32_t reps = 0, mt = 0, warps = 0, ctas = 0, spin = 2000, tokens = 64;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--shape") == 0 && i + 1 < argc) shape = argv[++i];
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--mt") == 0 && i + 1 < argc) mt = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--warps") == 0 && i + 1 < argc) warps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctas") == 0 && i + 1 < argc) ctas = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--spin-us") == 0 && i + 1 < argc) spin = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-evict") == 0) g_evict = 0;
        else { printf("unknown argument %s\n", argv[i]); return 2; }
    }
    if (host) host_only();
    if (sim) sim_battery();
    if (!host && !sim) {
        if (mt || warps) omega_gpu_fp8gemm_set_shape(mt, warps);
        if (ctas) omega_gpu_fp8gemm_set_cta_budget(ctas);
        (void)omega_gpu_session_set_spin_us(spin);
        if (out_path) { g_out = fopen(out_path, "w"); if (g_out) fprintf(g_out, "{\"receipt\":\"OMEGA_GPU_FP8GEMM_V1\",\"reps\":%u,\"evict\":%d,\"tokens\":%u,\"spin_us\":%u,\"points\":[", reps ? reps : 1, g_evict, tokens, omega_gpu_session_spin_us()); }
        Shape shapes[] = { { "gateup", 128, 1536, 2048, tokens, 8 }, { "down", 128, 2048, 768, tokens, 8 }, { "qkv", 1, 5120, 2048, tokens, 1 }, { "o", 1, 2048, 4096, tokens, 1 }, { "d768", 1, 2048, 768, tokens, 1 } };
        for (unsigned i = 0; i < 5; i++) if ((!strcmp(shape, "all") && i < 4) || !strcmp(shape, shapes[i].name)) run_shape(&shapes[i], reps ? reps : 1, 1000 + i);
        if (g_out) { fprintf(g_out, "],\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS"); fclose(g_out); }
        omega_gpu_session_close();
    }
    printf("gpu_fp8gemm_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
