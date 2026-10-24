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

/* gpu_matmul_api_test: FB-1 cut 1 / 1b gate for omega_gpu_matmul_api.
 *
 *   ./gpu_matmul_api_test --host-only                 argument and limit refusals, codegen only, no chip
 *   ./gpu_matmul_api_test --out receipt.json          chip sweep over shapes, JSON receipt
 *   ./gpu_matmul_api_test --timing --out r.json       resident GEMV timing (median of 100 calls) + parity
 *   ./gpu_matmul_api_test --update --out r.json       tensor_update_* in place vs a fresh upload, and its timing (R3)
 *   ./gpu_matmul_api_test --mutant --out r.json       wrong kernel (last K step dropped) must FAIL parity;
 *                                                     exit 0 when every shape caught it ("mutant killed")
 *   ./gpu_matmul_api_test --cta-budget N ...          run with a different CTA budget (I42 experiment)
 *   ./gpu_matmul_api_test --spin-us N ...             marker-wait spin window for every launch (omega#328)
 *   ./gpu_matmul_api_test --unroll N ...              force the fragment kernel unroll (omega #328 experiment)
 *   ./gpu_matmul_api_test --stages --out r.json       omega #473 stage profile (resident, bf16 and f32 rows, median of 7)
 *
 * Shapes come from OMEGA_GPU_MATMUL_SHAPES ("m,k,n;m,k,n;..."), default below.
 * Pass rule per shape: rc OK, parity verified by the API oracle, end-to-end error < 1e-5 relative to
 * the accumulation scale (bf16 inputs against a double reference; only accumulation order and the
 * chip's f32 accumulator differ), second call hits the kernel cache and agrees bit for bit.
 */
#include "omega_gpu_matmul_api.h"
#include "omega_gpu_session.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_matmul.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* round a float to bf16 and back, as the API does on the way in */
static float bf16r(float f) { uint16_t h = omega_fp32_to_bf16(f); uint32_t u = (uint32_t)h << 16; float o; memcpy(&o, &u, 4); return o; }

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* cut-1 list unchanged, then the cut-1b Llama shapes (hidden 2048, ffn 8192, vocab 128256) */
static const char *DEFAULT_SHAPES =
    "16,16,16;32,32,32;64,64,64;128,128,128;256,256,256;512,512,512;"
    "1,1024,1024;1,256,1024;16,1024,256;8,1024,8;"
    "1024,16,128;192,16,192;512,16,64;256,16,256;128,16,512;"
    "1,2048,8192;1,8192,2048;1,2048,128256";
static const char *MUTANT_SHAPES = "16,32,16;1,2048,8192;64,64,64;1,1024,1024";
static const char *TIMING_SHAPES = "1,2048,8192;1,8192,2048";
/* omega #473 training shapes */
static const char *STAGE_SHAPES = "8192,768,2304;8192,768,768;8192,768,3072;8192,3072,768;1024,768,50304";

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static void fill(float *a, size_t n, uint32_t *seed) { for (size_t i = 0; i < n; i++) a[i] = ((float)(lcg(seed) % 2001) - 1000.0f) / 1000.0f; }

/* end-to-end: full-K product on the host from the same bf16-rounded inputs, in double; error relative
 * to the accumulation scale (sum of |terms|), not to a result that may cancel to near zero */
static double e2e_max_rel(uint32_t m, uint32_t k, uint32_t n, const float *a, const float *b, const float *c, size_t *bad) {
    double worst = 0.0; size_t nbad = 0;
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < n; j++) {
        double acc = 0.0, scale = 0.0;
        for (size_t t = 0; t < k; t++) { double p = (double)bf16r(a[i * k + t]) * (double)bf16r(b[t * n + j]); acc += p; scale += fabs(p); }
        double rel = fabs(acc - (double)c[i * n + j]) / (scale > 1e-6 ? scale : 1e-6);
        if (rel > worst) worst = rel;
        if (rel >= 1e-5) nbad++;
    }
    if (bad) *bad = nbad;
    return worst;
}

/* omega #328 fragment-major kernel, host only. The unpack below is written from the PTX ISA
 * table for mma.m16n8k16 (.bf16 A: a0,a1 row g cols 2t,2t+1; a2,a3 row g+8; a4,a5 row g cols
 * 2t+8,2t+9; a6,a7 row g+8. B: b0,b1 rows 2t,2t+1 col g; b2,b3 rows 2t+8,2t+9; g = lane>>2,
 * t = lane&3), independently of the packers: pack then unpack must give the matrix back. */
/* Result sentinels: one poison word per 16 x 8 fragment tile that holds a real output word, inside the
 * tile's real rows and columns; the scan counts exactly the survivors. */
static void sentinel_host_checks(void) {
    static const uint32_t shapes[][2] = { { 1u, 24576u }, { 1u, 4096u }, { 16u, 151936u }, { 128u, 24576u }, { 116u, 4096u }, { 3u, 20u }, { 17u, 9u } };
    const float poison = -999.0f;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        const uint32_t m = shapes[s][0], n = shapes[s][1], mp = (m + 15u) / 16u * 16u, np = (n + 7u) / 8u * 8u;
        const uint32_t mt = (m + 15u) / 16u, nt = (n + 7u) / 8u;
        float *cd = calloc((size_t)mp * np, sizeof *cd);
        const size_t set = omega_matmul_result_sentinels(cd, m, n, np, poison, 1);
        CHECK(set == (size_t)mt * nt, "sentinel count m=%u n=%u: %zu != %u", m, n, set, mt * nt);
        /* every fragment tile with a real word holds exactly one sentinel, inside its real rows and columns */
        size_t tiles_ok = 0, outside = 0;
        for (uint32_t ti = 0; ti < mt; ti++)
            for (uint32_t tj = 0; tj < nt; tj++) {
                size_t in_tile = 0;
                for (uint32_t r = ti * 16u; r < ti * 16u + 16u; r++)
                    for (uint32_t c = tj * 8u; c < tj * 8u + 8u; c++)
                        if (cd[(size_t)r * np + c] == poison) { if (r < m && c < n) in_tile++; else outside++; }
                if (in_tile == 1) tiles_ok++;
            }
        CHECK(tiles_ok == (size_t)mt * nt && outside == 0, "sentinel placement m=%u n=%u: %zu tiles ok, %zu outside", m, n, tiles_ok, outside);
        CHECK(omega_matmul_result_sentinels(cd, m, n, np, poison, 0) == set, "sentinel scan m=%u n=%u: all survive", m, n);
        /* clearing one tile's sentinel removes exactly one survivor; a fully written result has none */
        cd[(size_t)(m - 1u) * np + (n - 1u)] = 0.0f;
        CHECK(omega_matmul_result_sentinels(cd, m, n, np, poison, 0) == set - 1u, "sentinel scan m=%u n=%u: one cleared", m, n);
        for (size_t i = 0; i < (size_t)mp * np; i++) cd[i] = 1.5f;
        CHECK(omega_matmul_result_sentinels(cd, m, n, np, poison, 0) == 0, "sentinel scan m=%u n=%u: none survive", m, n);
        free(cd);
    }
}

static void frag_host_checks(void) {
    sentinel_host_checks();
    /* row-0 pack (one-row decode call): a full one-row pack of A2 equals a full pack of A1 followed by the
     * row-0 pack of A2, bf16 and f32 inputs, including a width with padding columns */
    {
        const uint32_t ks[] = { 4096u, 40u };
        for (size_t i = 0; i < 2; i++) {
            const uint32_t k = ks[i], kp = (k + 15u) / 16u * 16u;
            float *a1 = malloc(k * sizeof *a1), *a2 = malloc(k * sizeof *a2);
            uint16_t *b1 = malloc(k * 2), *b2 = malloc(k * 2), *full = malloc((size_t)16 * kp * 2), *inc = malloc((size_t)16 * kp * 2);
            uint32_t seed = 77u + k;
            fill(a1, k, &seed); fill(a2, k, &seed);
            for (uint32_t j = 0; j < k; j++) { b1[j] = omega_fp32_to_bf16(a1[j]); b2[j] = omega_fp32_to_bf16(a2[j]); }
            omega_matmul_frag_pack_a(b2, 1, k, 16, kp, full);
            omega_matmul_frag_pack_a(b1, 1, k, 16, kp, inc);
            omega_matmul_frag_pack_a_row0(b2, k, kp, inc);
            CHECK(memcmp(full, inc, (size_t)16 * kp * 2) == 0, "row-0 pack bf16 k=%u equals the full one-row pack", k);
            omega_matmul_frag_pack_a_f32(a1, 1, k, 16, kp, inc);
            omega_matmul_frag_pack_a_row0_f32(a2, k, kp, inc);
            CHECK(memcmp(full, inc, (size_t)16 * kp * 2) == 0, "row-0 pack f32 k=%u equals the full one-row pack", k);
            free(a1); free(a2); free(b1); free(b2); free(full); free(inc);
        }
    }
    CHECK(omega_blackwell_verify_codegen_fixtures_fragmm() == 0, "fragment matmul golden words (rc %d)", omega_blackwell_verify_codegen_fixtures_fragmm());
    struct { uint32_t m, k, n; } sh[] = { {1, 16, 8}, {1, 40, 13}, {17, 48, 24}, {33, 2560, 72}, {5, 160, 1000} };
    for (size_t s = 0; s < sizeof sh / sizeof sh[0]; s++) {
        const uint32_t m = sh[s].m, k = sh[s].k, n = sh[s].n;
        const uint32_t mp = (m + 15) / 16 * 16, kp = (k + 15) / 16 * 16, np = (n + 7) / 8 * 8, KS = kp / 16, NT = np / 8;
        uint16_t *a = malloc((size_t)m * k * 2), *b = malloc((size_t)k * n * 2);
        uint16_t *pa = malloc((size_t)mp * kp * 2), *pb = malloc((size_t)kp * np * 2);
        uint32_t seed = 7 + (uint32_t)s;
        for (size_t i = 0; i < (size_t)m * k; i++) a[i] = (uint16_t)(1 + lcg(&seed) % 60000);
        for (size_t i = 0; i < (size_t)k * n; i++) b[i] = (uint16_t)(1 + lcg(&seed) % 60000);
        omega_matmul_frag_pack_a(a, m, k, mp, kp, pa);
        /* B in two uneven chunks, as the upload does in FRAG_PACK_TILES steps */
        uint32_t split = NT / 2;
        omega_matmul_frag_pack_b(b, k, n, kp, 0, split, pb);
        omega_matmul_frag_pack_b(b, k, n, kp, split, NT, pb + (size_t)split * kp * 8);
        size_t bad_a = 0, bad_b = 0;
        for (uint32_t mt = 0; mt < mp / 16; mt++)
            for (uint32_t ks = 0; ks < KS; ks++)
                for (uint32_t lane = 0; lane < 32; lane++) {
                    const uint32_t gg = lane >> 2, t = lane & 3;
                    const uint32_t rows[8] = { gg, gg, gg + 8, gg + 8, gg, gg, gg + 8, gg + 8 };
                    const uint32_t cols[8] = { 2 * t, 2 * t + 1, 2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9, 2 * t + 8, 2 * t + 9 };
                    for (int e = 0; e < 8; e++) {
                        uint32_t r = mt * 16 + rows[e], c = ks * 16 + cols[e];
                        uint16_t want = (r < m && c < k) ? a[(size_t)r * k + c] : 0;
                        if (pa[(((size_t)mt * KS + ks) * 32 + lane) * 8 + e] != want) bad_a++;
                    }
                }
        for (uint32_t nt = 0; nt < NT; nt++)
            for (uint32_t ks = 0; ks < KS; ks++)
                for (uint32_t lane = 0; lane < 32; lane++) {
                    const uint32_t gg = lane >> 2, t = lane & 3;
                    const uint32_t rows[4] = { 2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9 };
                    for (int e = 0; e < 4; e++) {
                        uint32_t r = ks * 16 + rows[e], c = nt * 8 + gg;
                        uint16_t want = (r < k && c < n) ? b[(size_t)r * n + c] : 0;
                        if (pb[(((size_t)nt * KS + ks) * 32 + lane) * 4 + e] != want) bad_b++;
                    }
                }
        CHECK(bad_a == 0 && bad_b == 0, "fragment pack round trip %ux%ux%u (A bad %zu, B bad %zu)", m, k, n, bad_a, bad_b);
        /* the fused f32 packer equals convert-then-pack bit for bit (ties, denormals, inf, NaN included) */
        float *af = malloc((size_t)m * k * sizeof *af);
        uint16_t *pf = malloc((size_t)mp * kp * 2);
        for (size_t i = 0; i < (size_t)m * k; i++) {
            uint32_t bits = lcg(&seed) * 2654435761u;
            switch (i % 7) {
            case 0: bits = (bits & 0x807fffffu) | 0x3f800000u; break; /* in [1,2) with every mantissa bit live (rounding ties) */
            case 1: bits = 0x7f800000u | ((bits >> 9) & 0x7fffffu); break; /* inf (0 payload) or NaN payloads */
            case 2: bits &= 0x807fffffu; break; /* zero and denormals */
            default: break;
            }
            memcpy(&af[i], &bits, 4);
            a[i] = omega_fp32_to_bf16(af[i]);
        }
        omega_matmul_frag_pack_a(a, m, k, mp, kp, pa);
        omega_matmul_frag_pack_a_f32(af, m, k, mp, kp, pf);
        CHECK(memcmp(pa, pf, (size_t)mp * kp * 2) == 0, "fused f32 pack equals convert-then-pack %ux%u", m, k);
        free(af); free(pf);
        free(a); free(b); free(pa); free(pb);
    }
    /* the kernel generates for every padded Qwen3-4B and Llama-3.2-1B decode shape at the API's unroll */
    struct { uint32_t k, n, gx; } ok[] = { {16, 8, 1}, {2560, 4096, 256}, {2560, 1024, 128}, {4096, 2560, 256}, {2560, 9728, 256},
                                          {9728, 2560, 256}, {2560, 151936, 256}, {2048, 8192, 64}, {8192, 2048, 64},
                                          {2048, 128256, 64}, {16384, 262144, 64}, {48, 16, 2} };
    for (size_t i = 0; i < sizeof ok / sizeof ok[0]; i++) {
        OmegaMatMulSpec spec; OmegaBlackwellKernel kernel; memset(&kernel, 0, sizeof kernel);
        const uint32_t u = omega_matmul_frag_unroll(ok[i].k);
        int rc = omega_matmul_spec_init(&spec, 16, ok[i].k, ok[i].n, OMEGA_MATMUL_PRECISION_BF16) == 0 &&
                 omega_blackwell_codegen_matmul_frag(&spec, ok[i].gx, u, 0, &kernel) == 0 && kernel.code_size > 0;
        CHECK(rc, "fragment kernel K=%u N=%u gx=%u unroll=%u generated", ok[i].k, ok[i].n, ok[i].gx, u);
        CHECK(kernel.gpr_count <= 64, "fragment kernel K=%u N=%u unroll=%u fits 64 registers (got %u)", ok[i].k, ok[i].n, u, kernel.gpr_count);
        if (i == 1) printf("fragment kernel K=2560 N=4096 unroll %u: %zu instructions, %u registers\n", u, kernel.insn_count, kernel.gpr_count);
        OmegaBlackwellKernel mut; memset(&mut, 0, sizeof mut);
        if (omega_blackwell_codegen_matmul_frag(&spec, ok[i].gx, u, 1, &mut) == 0) {
            CHECK(memcmp(mut.code_digest, kernel.code_digest, 32) != 0, "fragment mutant differs K=%u N=%u", ok[i].k, ok[i].n);
            omega_blackwell_kernel_free(&mut);
        } else CHECK(ok[i].k / 16 / u < 2, "fragment mutant refused only with one pass (K=%u)", ok[i].k);
        omega_blackwell_kernel_free(&kernel);
    }
    CHECK(omega_matmul_frag_unroll(2560) == 4 && omega_matmul_frag_unroll(48) == 1 && omega_matmul_frag_unroll(64) == 4 &&
          omega_matmul_frag_unroll(32) == 2 && omega_matmul_frag_unroll(128) == 4, "unroll choice (capped at 4)");
    {   /* unroll 8 still generates (experiments) */
        OmegaMatMulSpec s8; OmegaBlackwellKernel k8; memset(&k8, 0, sizeof k8);
        CHECK(omega_matmul_spec_init(&s8, 16, 2560, 4096, OMEGA_MATMUL_PRECISION_BF16) == 0 &&
              omega_blackwell_codegen_matmul_frag(&s8, 256, 8, 0, &k8) == 0 && k8.code_size > 0, "unroll 8 generates");
        omega_blackwell_kernel_free(&k8);
    }
    OmegaMatMulSpec spec; OmegaBlackwellKernel kernel; memset(&kernel, 0, sizeof kernel);
    CHECK(omega_matmul_spec_init(&spec, 16, 48, 16, OMEGA_MATMUL_PRECISION_BF16) == 0 &&
          omega_blackwell_codegen_matmul_frag(&spec, 1, 2, 0, &kernel) != 0, "unroll not dividing K/16 refused");
    CHECK(omega_blackwell_codegen_matmul_frag(&spec, 1, 3, 0, &kernel) != 0, "unroll 3 refused");
    CHECK(omega_blackwell_codegen_matmul_frag(&spec, 3, 1, 0, &kernel) != 0, "fragment grid_x beyond the column tiles refused");
    /* 2026-10-12: the blocked kernel (rt row tiles x ct column tiles per warp) generates for the Qwen3-8B
     * prefill shapes at m = 128 (rt 4, ct 4, grid_x 512 at the daemon's 1024-CTA budget) and the small
     * parity shapes; it fits the allocator's 128 registers; its mutant differs; the envelope is enforced */
    struct { uint32_t k, n, gx, rt, ct; } bk[] = { {4096, 24576, 512, 4, 4}, {12288, 4096, 128, 4, 4}, {4096, 6144, 192, 4, 4},
                                                  {128, 128, 4, 4, 4}, {32, 32, 1, 2, 4}, {16, 8, 1, 2, 1}, {1024, 8, 1, 4, 1},
                                                  {64, 64, 2, 4, 2}, {48, 16, 1, 2, 2} };
    for (size_t i = 0; i < sizeof bk / sizeof bk[0]; i++) {
        OmegaMatMulSpec spec; OmegaBlackwellKernel kernel; memset(&kernel, 0, sizeof kernel);
        const uint32_t u = omega_matmul_frag_block_unroll(bk[i].k, bk[i].rt, bk[i].ct);
        int rc = omega_matmul_spec_init(&spec, 16, bk[i].k, bk[i].n, OMEGA_MATMUL_PRECISION_BF16) == 0 &&
                 omega_blackwell_codegen_matmul_frag_block(&spec, bk[i].gx, u, bk[i].rt, bk[i].ct, 0, &kernel) == 0 && kernel.code_size > 0;
        CHECK(rc, "blocked kernel K=%u N=%u gx=%u rt=%u ct=%u unroll=%u generated", bk[i].k, bk[i].n, bk[i].gx, bk[i].rt, bk[i].ct, u);
        CHECK(u * (bk[i].rt + bk[i].ct) <= 8, "blocked kernel K=%u rt=%u ct=%u issues at most 8 loads per pass (unroll %u)", bk[i].k, bk[i].rt, bk[i].ct, u);
        if (i < 3) printf("blocked kernel K=%u N=%u rt=%u ct=%u unroll %u: %zu instructions, %u registers\n", bk[i].k, bk[i].n, bk[i].rt, bk[i].ct, u, kernel.insn_count, kernel.gpr_count);
        OmegaBlackwellKernel mut; memset(&mut, 0, sizeof mut);
        if (omega_blackwell_codegen_matmul_frag_block(&spec, bk[i].gx, u, bk[i].rt, bk[i].ct, 1, &mut) == 0) {
            CHECK(memcmp(mut.code_digest, kernel.code_digest, 32) != 0, "blocked mutant differs K=%u N=%u", bk[i].k, bk[i].n);
            omega_blackwell_kernel_free(&mut);
        } else CHECK(bk[i].k / 16 / u < 2, "blocked mutant refused only with one pass (K=%u)", bk[i].k);
        omega_blackwell_kernel_free(&kernel);
    }
    {
        uint32_t rt, ct;
        omega_matmul_frag_block_shape(8, 3072, &rt, &ct); CHECK(rt == 4 && ct == 4, "block shape m=128 n=24576 is 4x4");
        omega_matmul_frag_block_shape(1, 3072, &rt, &ct); CHECK(rt == 1 && ct == 1, "block shape m=1 is the single-tile kernel");
        omega_matmul_frag_block_shape(7, 3072, &rt, &ct); CHECK(rt == 1 && ct == 1, "block shape m=112 (7 row tiles) is the single-tile kernel");
        omega_matmul_frag_block_shape(6, 6, &rt, &ct); CHECK(rt == 2 && ct == 2, "block shape 6 row tiles, 6 column tiles is 2x2");
        omega_matmul_frag_block_shape(4, 1, &rt, &ct); CHECK(rt == 4 && ct == 1, "block shape 4 row tiles, 1 column tile is 4x1");
        /* the C store offset of the last row tile is a signed 24-bit immediate: wide N steps rt down (lm_head: to 1) */
        omega_matmul_frag_block_shape(2, 18992, &rt, &ct); CHECK(rt == 1 && ct == 1, "block shape lm_head (N 151936) at 2 row tiles is the single-tile kernel: %ux%u", rt, ct);
        omega_matmul_frag_block_shape(8, 18992, &rt, &ct); CHECK(rt == 1 && ct == 1, "block shape lm_head (N 151936) at 8 row tiles is the single-tile kernel: %ux%u", rt, ct);
        omega_matmul_frag_block_shape(8, 8192, &rt, &ct); CHECK(rt == 2 && ct == 4, "block shape N 65536 at 8 row tiles steps rt 4 -> 2: %ux%u", rt, ct);
        CHECK(omega_matmul_frag_block_unroll(4096, 4, 4) == 1 && omega_matmul_frag_block_unroll(4096, 2, 2) == 2 &&
              omega_matmul_frag_block_unroll(48, 2, 2) == 1 && omega_matmul_frag_block_unroll(4096, 4, 2) == 1, "blocked unroll rule (loads per pass <= 8)");
        OmegaMatMulSpec s; OmegaBlackwellKernel kk; memset(&kk, 0, sizeof kk);
        CHECK(omega_matmul_spec_init(&s, 16, 4096, 6144, OMEGA_MATMUL_PRECISION_BF16) == 0, "spec for refusals");
        CHECK(omega_blackwell_codegen_matmul_frag_block(&s, 1, 1, 1, 1, 0, &kk) != 0, "blocked 1x1 refused (single-tile kernel)");
        CHECK(omega_blackwell_codegen_matmul_frag_block(&s, 1, 1, 3, 2, 0, &kk) != 0, "blocked rt 3 refused");
        CHECK(omega_blackwell_codegen_matmul_frag_block(&s, 1, 1, 8, 4, 0, &kk) != 0, "blocked 8x4 (32 accumulators) refused");
        CHECK(omega_blackwell_codegen_matmul_frag_block(&s, 1, 1, 8, 1, 0, &kk) != 0, "blocked 8x1 (9 loads per pass) refused");
        CHECK(omega_blackwell_codegen_matmul_frag_block(&s, 1, 2, 4, 4, 0, &kk) != 0, "blocked unroll 2 at 4x4 (16 loads per pass) refused");
        CHECK(omega_blackwell_codegen_matmul_frag_block(&s, 1, 4, 2, 2, 0, &kk) != 0, "blocked unroll 4 refused");
        CHECK(omega_blackwell_codegen_matmul_frag_block(&s, 193, 1, 4, 4, 0, &kk) != 0, "blocked grid_x beyond the column blocks refused");
        OmegaMatMulSpec s3; CHECK(omega_matmul_spec_init(&s3, 16, 64, 24, OMEGA_MATMUL_PRECISION_BF16) == 0, "spec n=24");
        CHECK(omega_blackwell_codegen_matmul_frag_block(&s3, 1, 1, 2, 4, 0, &kk) != 0, "blocked ct not dividing the column tiles refused");
        CHECK(omega_blackwell_codegen_matmul_frag_block(&s3, 1, 1, 2, 2, 0, &kk) != 0, "blocked ct 2 on 3 column tiles refused");
    }
}

static void host_only(void) {
    float a[16 * 16] = {0}, b[16 * 16] = {0}, c[16 * 16] = {0};
    OmegaGpuMatmulInfo info;
    CHECK(omega_gpu_matmul_f32(0, 16, 16, a, b, c, &info) == OMEGA_GPU_MATMUL_BAD_ARGS, "m=0 refused");
    CHECK(omega_gpu_matmul_f32(16, 16, 16, NULL, b, c, &info) == OMEGA_GPU_MATMUL_BAD_ARGS, "NULL a refused");
    CHECK(omega_gpu_matmul_f32(OMEGA_BW_MATMUL_MAX_M + 1, 16, 16, a, b, c, &info) == OMEGA_GPU_MATMUL_TOO_LARGE, "m over limit refused");
    CHECK(omega_gpu_matmul_bf16(16, OMEGA_BW_MATMUL_MAX_K + 1, 16, (uint16_t *)a, (uint16_t *)b, c, NULL) == OMEGA_GPU_MATMUL_TOO_LARGE, "k over limit refused (NULL info ok)");
    /* session spin window (omega#328): in-range value kept, out-of-range refused unchanged, 0 restores */
    const uint32_t spin0 = omega_gpu_session_spin_us();
    CHECK(omega_gpu_session_set_spin_us(2000) == 0 && omega_gpu_session_spin_us() == 2000, "spin window 2000 us set");
    CHECK(omega_gpu_session_set_spin_us(OMEGA_GPU_SESSION_MAX_SPIN_US + 1) == -1 && omega_gpu_session_spin_us() == 2000,
          "spin window over the maximum refused, setting unchanged");
    CHECK(omega_gpu_session_set_spin_us(0) == 0 && omega_gpu_session_spin_us() == 0, "spin window 0 restores the sleep-poll");
    omega_gpu_session_set_spin_us(spin0);
    CHECK(omega_gpu_matmul_bf16(16, 16, OMEGA_BW_MATMUL_MAX_N + 1, (uint16_t *)a, (uint16_t *)b, c, NULL) == OMEGA_GPU_MATMUL_TOO_LARGE, "n over limit refused");
    CHECK(OMEGA_BW_MATMUL_MAX_N >= 128256 && OMEGA_BW_MATMUL_MAX_K >= 8192 && OMEGA_BW_MATMUL_MAX_M >= 2048, "limits cover Llama shapes (vocab 128256, ffn 8192)");
    OmegaGpuTensor *t = NULL;
    CHECK(omega_gpu_tensor_upload_f32(0, 16, a, &t) == OMEGA_GPU_MATMUL_BAD_ARGS && t == NULL, "tensor k=0 refused");
    CHECK(omega_gpu_tensor_upload_f32(16, 16, NULL, &t) == OMEGA_GPU_MATMUL_BAD_ARGS, "tensor NULL refused");
    CHECK(omega_gpu_matmul_resident_f32(16, a, NULL, c, &info) == OMEGA_GPU_MATMUL_BAD_ARGS, "resident NULL tensor refused");
    CHECK(omega_gpu_tensor_update_f32(NULL, 16, 16, a) == OMEGA_GPU_MATMUL_BAD_ARGS, "update_f32 NULL tensor refused");
    CHECK(omega_gpu_tensor_update_bf16(NULL, 16, 16, (const uint16_t *)a) == OMEGA_GPU_MATMUL_BAD_ARGS, "update_bf16 NULL tensor refused");
    CHECK(strcmp(omega_gpu_matmul_rc_name(OMEGA_GPU_MATMUL_PARITY_FAIL), "PARITY_FAIL") == 0, "rc name");
    CHECK(omega_gpu_matmul_cta_budget() == OMEGA_GPU_MATMUL_MAX_CTAS, "default CTA budget");
    {
        OmegaGpuMatmulStages s0;
        memset(&s0, 0xff, sizeof s0);
        omega_gpu_matmul_last_stages(&s0);
        omega_gpu_matmul_last_stages(NULL);
        CHECK(s0.core_ns == 0 && s0.launch_ns == 0 && s0.pack_ns == 0, "stages all zero before the first matmul");
    }
    /* omega #473 R4: the default is the best budget of the sealed receipt evidence/R4-MATMUL-CTA-ffc3176a (every element of
     * every call at the five training shapes and two single-tile shapes, budgets 64..1024, all sealed; 1024 fastest) */
    CHECK(OMEGA_GPU_MATMUL_MAX_CTAS == 1024u, "default CTA budget is the R4 sealed budget 1024 (got %u)", (unsigned)OMEGA_GPU_MATMUL_MAX_CTAS);
    CHECK(omega_gpu_matmul_is_blocked() == 0, "not blocked at start");

    /* codegen only: the looped kernel must accept every padded Llama shape without touching the chip */
    struct { uint32_t k, n, gx; } ok[] = { {16, 8, 1}, {2048, 8192, 64}, {8192, 2048, 64}, {2048, 128256, 64}, {1024, 1024, 1}, {16384, 262144, 64} };
    for (size_t i = 0; i < sizeof ok / sizeof ok[0]; i++) {
        OmegaMatMulSpec spec; OmegaBlackwellKernel kernel; memset(&kernel, 0, sizeof kernel);
        int rc = omega_matmul_spec_init(&spec, 16, ok[i].k, ok[i].n, OMEGA_MATMUL_PRECISION_BF16) == 0 &&
                 omega_blackwell_codegen_matmul_tensor_loop(&spec, ok[i].gx, 0, &kernel) == 0 && kernel.code_size > 0;
        CHECK(rc, "loop kernel K=%u N=%u gx=%u generated", ok[i].k, ok[i].n, ok[i].gx);
        CHECK(kernel.gpr_count <= 64, "loop kernel K=%u N=%u fits 64 registers (got %u)", ok[i].k, ok[i].n, kernel.gpr_count);
        /* the mutant kernel is a different program */
        OmegaBlackwellKernel mut; memset(&mut, 0, sizeof mut);
        if (omega_blackwell_codegen_matmul_tensor_loop(&spec, ok[i].gx, 1, &mut) == 0) {
            CHECK(memcmp(mut.code_digest, kernel.code_digest, 32) != 0, "mutant kernel differs K=%u N=%u", ok[i].k, ok[i].n);
            omega_blackwell_kernel_free(&mut);
        } else CHECK(ok[i].k == 16, "mutant refused only for K=16 (K=%u)", ok[i].k);
        omega_blackwell_kernel_free(&kernel);
    }
    OmegaMatMulSpec spec; OmegaBlackwellKernel kernel; memset(&kernel, 0, sizeof kernel);
    CHECK(omega_matmul_spec_init(&spec, 16, 24, 16, OMEGA_MATMUL_PRECISION_BF16) == 0 && omega_blackwell_codegen_matmul_tensor_loop(&spec, 1, 0, &kernel) != 0, "K not multiple of 16 refused by codegen (the API pads)");
    CHECK(omega_matmul_spec_init(&spec, 16, 16, 16, OMEGA_MATMUL_PRECISION_BF16) == 0 && omega_blackwell_codegen_matmul_tensor_loop(&spec, 3, 0, &kernel) != 0, "grid_x beyond the column tiles refused");
    /* the single-tile kernel of cut 1 still generates (other gates use it) */
    memset(&kernel, 0, sizeof kernel);
    CHECK(omega_matmul_spec_init(&spec, 16, 16, 8, OMEGA_MATMUL_PRECISION_BF16) == 0 && omega_blackwell_codegen_matmul(&spec, &kernel) == 0 && kernel.code_size > 0, "16x16x8 single-tile kernel generated");
    omega_blackwell_kernel_free(&kernel);
    frag_host_checks();
    omega_gpu_matmul_cache_clear();
}

/* One-row calls reuse the zero padding rows of the activation buffer: a second one-row call of the same
 * width writes row 0 only (hook counts it) and still matches the oracle with a different A; a wider
 * call in between, or another width, forces a full pack again. */
static void one_row_repack_checks(void) {
    const uint32_t k = 4096, n = 256;
    float *a = malloc((size_t)32 * k * sizeof *a), *b = malloc((size_t)k * n * sizeof *b), *c = malloc((size_t)32 * n * sizeof *c);
    uint32_t seed = 0x10c0u;
    fill(a, (size_t)32 * k, &seed); fill(b, (size_t)k * n, &seed);
    OmegaGpuMatmulInfo info;
    const uint64_t h0 = omega_gpu_matmul_test_a_row0_hits();
    int rc1 = omega_gpu_matmul_f32(1, k, n, a, b, c, &info);
    CHECK(rc1 == OMEGA_GPU_MATMUL_OK && info.parity_verified, "one-row call 1 (full pack) rc=%s parity=%d", omega_gpu_matmul_rc_name(rc1), info.parity_verified);
    const uint64_t h1 = omega_gpu_matmul_test_a_row0_hits();
    int rc2 = omega_gpu_matmul_f32(1, k, n, a + k, b, c, &info);
    const uint64_t h2 = omega_gpu_matmul_test_a_row0_hits();
    CHECK(rc2 == OMEGA_GPU_MATMUL_OK && info.parity_verified, "one-row call 2 (row 0 only, new A) rc=%s parity=%d", omega_gpu_matmul_rc_name(rc2), info.parity_verified);
    CHECK(h1 == h0 && h2 == h1 + 1, "row-0 path taken on the second one-row call only (hits %" PRIu64 " %" PRIu64 " %" PRIu64 ")", h0, h1, h2);
    int rc3 = omega_gpu_matmul_f32(32, k, n, a, b, c, &info);
    CHECK(rc3 == OMEGA_GPU_MATMUL_OK && info.parity_verified, "32-row call between rc=%s parity=%d", omega_gpu_matmul_rc_name(rc3), info.parity_verified);
    int rc4 = omega_gpu_matmul_f32(1, k, n, a + 2 * (size_t)k, b, c, &info);
    const uint64_t h4 = omega_gpu_matmul_test_a_row0_hits();
    CHECK(rc4 == OMEGA_GPU_MATMUL_OK && info.parity_verified && h4 == h2, "one-row call after a wider call packs in full (hits %" PRIu64 ")", h4);
    int rc5 = omega_gpu_matmul_f32(1, k, n, a + 3 * (size_t)k, b, c, &info);
    const uint64_t h5 = omega_gpu_matmul_test_a_row0_hits();
    CHECK(rc5 == OMEGA_GPU_MATMUL_OK && info.parity_verified && h5 == h4 + 1, "one-row call after a one-row call writes row 0 only (hits %" PRIu64 ")", h5);
    int rc6 = omega_gpu_matmul_f32(1, 64, n, a, b, c, &info);
    const uint64_t h6 = omega_gpu_matmul_test_a_row0_hits();
    CHECK(rc6 == OMEGA_GPU_MATMUL_OK && info.parity_verified && h6 == h5, "one-row call of another width packs in full (hits %" PRIu64 ")", h6);
    free(a); free(b); free(c);
}

/* omega #473: the stage timer must explain the call. The six stages tile matmul_core in order, the chip
 * time sits inside the launch stage, and the core span sits inside the whole call. */
static uint64_t stage_sum(const OmegaGpuMatmulStages *s) {
    return s->setup_ns + s->pack_ns + s->upload_ns + s->launch_ns + s->sync_ns + s->copy_out_ns;
}
static void stage_checks(const char *tag, const OmegaGpuMatmulInfo *info) {
    OmegaGpuMatmulStages s;
    omega_gpu_matmul_last_stages(&s);
    CHECK(s.chip_ns == info->elapsed_ns && s.chip_ns > 0, "%s stages: chip_ns %" PRIu64 " equals elapsed_ns %" PRIu64, tag, s.chip_ns, info->elapsed_ns);
    CHECK(s.launch_ns >= s.chip_ns, "%s stages: launch_ns %" PRIu64 " holds chip_ns %" PRIu64, tag, s.launch_ns, s.chip_ns);
    CHECK(s.pack_ns > 0 && s.copy_out_ns > 0, "%s stages: pack %" PRIu64 " and copy-out %" PRIu64 " measured", tag, s.pack_ns, s.copy_out_ns);
    CHECK(stage_sum(&s) <= s.core_ns && s.core_ns <= info->call_ns && s.core_ns > 0,
          "%s stages: sum %" PRIu64 " <= core %" PRIu64 " <= call %" PRIu64, tag, stage_sum(&s), s.core_ns, info->call_ns);
}

/* omega #473 profile: resident calls at the given shapes, 3 warmups then 7 timed calls per input type
 * (bf16 rows, f32 rows); median of each stage. Correctness is the R4 receipt's job; here the two input
 * types must agree bit for bit and every call must pass stage_checks. */
static uint64_t med_u64(uint64_t *v, int n) {
    for (int i = 1; i < n; i++) for (int j = i; j > 0 && v[j - 1] > v[j]; j--) { uint64_t t = v[j]; v[j] = v[j - 1]; v[j - 1] = t; }
    return v[n / 2];
}
static int stages_profile(const char *shapes, const char *out_path) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_MATMUL_STAGES_V1\",\"cta_budget\":%u,\"warmups\":3,\"timed\":7,\"rows\":[", omega_gpu_matmul_cta_budget());
    char *copy = strdup(shapes), *save = NULL;
    int first = 1;
    for (char *tok = strtok_r(copy, ";", &save); tok; tok = strtok_r(NULL, ";", &save)) {
        unsigned m, k, n;
        if (sscanf(tok, "%u,%u,%u", &m, &k, &n) != 3) { printf("bad shape %s\n", tok); continue; }
        const size_t na = (size_t)m * k, nb = (size_t)k * n, nc = (size_t)m * n;
        float *a = malloc(na * 4), *b = malloc(nb * 4), *c16 = malloc(nc * 4), *c32 = malloc(nc * 4);
        uint16_t *ah = malloc(na * 2);
        if (!a || !b || !c16 || !c32 || !ah) { CHECK(0, "host memory for %s", tok); break; }
        uint32_t seed = 0x5eed0000u ^ (m << 20) ^ (k << 10) ^ n;
        fill(a, na, &seed); fill(b, nb, &seed);
        for (size_t i = 0; i < na; i++) ah[i] = omega_fp32_to_bf16(a[i]);
        OmegaGpuTensor *t = NULL;
        int urc = omega_gpu_tensor_upload_f32(k, n, b, &t);
        CHECK(urc == OMEGA_GPU_MATMUL_OK, "upload %s rc=%s", tok, omega_gpu_matmul_rc_name(urc));
        if (urc != OMEGA_GPU_MATMUL_OK) break;
        for (int in32 = 0; in32 < 2; in32++) {
            uint64_t v[9][7]; /* setup pack upload launch chip sync copy_out core call */
            int ok = 1;
            for (int it = 0; it < 10 && ok; it++) {
                OmegaGpuMatmulInfo info;
                int rc = in32 ? omega_gpu_matmul_resident_f32(m, a, t, c32, &info) : omega_gpu_matmul_resident_bf16(m, ah, t, c16, &info);
                if (rc != OMEGA_GPU_MATMUL_OK) { CHECK(0, "%s %s rc=%s (%s)", tok, in32 ? "f32" : "bf16", omega_gpu_matmul_rc_name(rc), omega_gpu_matmul_last_error()); ok = 0; break; }
                stage_checks(tok, &info);
                if (it < 3) continue;
                OmegaGpuMatmulStages s; omega_gpu_matmul_last_stages(&s);
                const uint64_t row[9] = { s.setup_ns, s.pack_ns, s.upload_ns, s.launch_ns, s.chip_ns, s.sync_ns, s.copy_out_ns, s.core_ns, info.call_ns };
                for (int q = 0; q < 9; q++) v[q][it - 3] = row[q];
            }
            if (!ok) continue;
            uint64_t md[9];
            for (int q = 0; q < 9; q++) md[q] = med_u64(v[q], 7);
            const double flop = 2.0 * m * k * n;
            printf("STAGES %s in=%s budget=%u call_ms=%.3f core_ms=%.3f setup=%.3f pack=%.3f upload=%.3f launch=%.3f chip=%.3f launch_minus_chip=%.3f sync=%.3f copy_out=%.3f "
                   "tflops_chip=%.3f tflops_call=%.3f chip_share=%.1f%%\n", tok, in32 ? "f32" : "bf16", omega_gpu_matmul_cta_budget(), md[8] / 1e6, md[7] / 1e6,
                   md[0] / 1e6, md[1] / 1e6, md[2] / 1e6, md[3] / 1e6, md[4] / 1e6, (md[3] > md[4] ? md[3] - md[4] : 0) / 1e6, md[5] / 1e6, md[6] / 1e6,
                   flop / (double)md[4] / 1e3, flop / (double)md[8] / 1e3, 100.0 * (double)md[4] / (double)md[8]);
            if (out) {
                fprintf(out, "%s{\"m\":%u,\"k\":%u,\"n\":%u,\"input\":\"%s\",\"median_ns\":{\"setup\":%" PRIu64 ",\"pack\":%" PRIu64 ",\"upload\":%" PRIu64 ",\"launch\":%" PRIu64
                        ",\"chip\":%" PRIu64 ",\"sync\":%" PRIu64 ",\"copy_out\":%" PRIu64 ",\"core\":%" PRIu64 ",\"call\":%" PRIu64 "},\"tflops_chip\":%.4f,\"tflops_call\":%.4f}",
                        first ? "" : ",", m, k, n, in32 ? "f32" : "bf16", md[0], md[1], md[2], md[3], md[4], md[5], md[6], md[7], md[8], flop / (double)md[4] / 1e3, flop / (double)md[8] / 1e3);
                first = 0;
            }
        }
        size_t diff = 0;
        for (size_t i = 0; i < nc; i++) diff += memcmp(&c16[i], &c32[i], 4) != 0;
        CHECK(diff == 0, "%s bf16-in and f32-in outputs agree bit for bit (%zu differ)", tok, diff);
        omega_gpu_tensor_free(t);
        free(a); free(b); free(c16); free(c32); free(ah);
        if (omega_gpu_matmul_is_blocked()) break;
    }
    free(copy);
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS"); fclose(out); }
    return 0;
}

static int chip_sweep(const char *shapes, const char *out_path, int mutant) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_MATMUL_API_SWEEP_V2\",\"mutant\":%s,\"cta_budget\":%u,\"shapes\":[", mutant ? "true" : "false", omega_gpu_matmul_cta_budget());
    char *copy = strdup(shapes);
    char *save = NULL;
    int first = 1;
    for (char *tok = strtok_r(copy, ";", &save); tok; tok = strtok_r(NULL, ";", &save)) {
        unsigned m, k, n;
        if (sscanf(tok, "%u,%u,%u", &m, &k, &n) != 3) { printf("bad shape %s\n", tok); continue; }
        size_t na = (size_t)m * k, nb = (size_t)k * n, nc = (size_t)m * n;
        float *a = malloc(na * sizeof *a), *b = malloc(nb * sizeof *b), *c = malloc(nc * sizeof *c), *c2 = malloc(nc * sizeof *c2);
        if (!a || !b || !c || !c2) { printf("out of host memory for %s\n", tok); free(a); free(b); free(c); free(c2); continue; }
        uint32_t seed = 0x5eed0000u ^ (m << 20) ^ (k << 10) ^ n;
        fill(a, na, &seed); fill(b, nb, &seed);
        for (size_t i = 0; i < nc; i++) c[i] = -12345.0f;
        OmegaGpuMatmulInfo info, info2;
        double t0 = now_ms();
        int rc = omega_gpu_matmul_f32(m, k, n, a, b, c, &info);
        double wall_ms = now_ms() - t0;
        if (!mutant && rc == OMEGA_GPU_MATMUL_OK) stage_checks(tok, &info);
        /* second call on the same shape must hit the kernel cache and agree */
        int rc2 = omega_gpu_matmul_f32(m, k, n, a, b, c2, &info2);
        size_t repeat_mismatch = 0;
        for (size_t i = 0; i < nc; i++) if (c[i] != c2[i]) repeat_mismatch++;
        size_t e2e_bad = 0;
        double e2e = (rc == OMEGA_GPU_MATMUL_OK || rc == OMEGA_GPU_MATMUL_PARITY_FAIL) ? e2e_max_rel(m, k, n, a, b, c, &e2e_bad) : 0.0;
        int pass = rc == OMEGA_GPU_MATMUL_OK && info.parity_verified && e2e < 1e-5
                   && rc2 == OMEGA_GPU_MATMUL_OK && info2.kernel_cache_hit && repeat_mismatch == 0;
        if (mutant) {
            /* the wrong kernel must be caught: PARITY_FAIL from the API oracle and a visible e2e error */
            int caught = rc == OMEGA_GPU_MATMUL_PARITY_FAIL && !info.parity_verified && e2e >= 1e-5;
            CHECK(caught, "mutant shape %ux%ux%u NOT caught: rc=%s parity=%d e2e_rel=%g", m, k, n, omega_gpu_matmul_rc_name(rc), info.parity_verified, e2e);
            pass = caught;
        } else {
            CHECK(pass, "shape %ux%ux%u rc=%s parity=%d max_rel=%g e2e_rel=%g rc2=%s hit=%d repeat_mismatch=%zu",
                  m, k, n, omega_gpu_matmul_rc_name(rc), info.parity_verified, info.max_rel_err, e2e,
                  omega_gpu_matmul_rc_name(rc2), info2.kernel_cache_hit, repeat_mismatch);
        }
        printf("%s %ux%ux%u rc=%s chip_ns=%" PRIu64 " call_ms=%.3f wall_ms=%.1f max_abs=%g max_rel=%g mismatches=%zu e2e_rel=%g e2e_bad=%zu grid=%ux%u calls=%u rows_per_call=%u cache_hit2=%d err=\"%s\"\n",
               pass ? (mutant ? "KILLED" : "PASS") : (mutant ? "SURVIVED" : "FAIL"), m, k, n, omega_gpu_matmul_rc_name(rc), info.elapsed_ns, info.call_ns / 1e6, wall_ms,
               info.max_abs_err, info.max_rel_err, info.mismatch_count, e2e, e2e_bad, info.grid_x, info.grid_y, info.chip_calls, info.rows_per_call, info2.kernel_cache_hit, rc == OMEGA_GPU_MATMUL_CHIP_FAIL ? omega_gpu_matmul_last_error() : "");
        if (out) {
            fprintf(out, "%s{\"m\":%u,\"k\":%u,\"n\":%u,\"rc\":\"%s\",\"pass\":%s,\"chip_elapsed_ns\":%" PRIu64
                    ",\"call_ms\":%.3f,\"wall_ms\":%.3f,\"max_abs_err\":%g,\"max_rel_err\":%g,\"mismatch_count\":%zu,"
                    "\"parity_verified\":%s,\"completion_marker\":%u,\"repeat_rc\":\"%s\",\"repeat_cache_hit\":%s,"
                    "\"repeat_mismatch\":%zu,\"e2e_max_rel\":%g,\"e2e_bad\":%zu,\"grid_x\":%u,\"grid_y\":%u,\"chip_calls\":%u,\"rows_per_call\":%u,\"padded_m\":%u,\"padded_k\":%u,\"padded_n\":%u,\"target_chip\":\"%s\",\"sm_architecture\":%u}",
                    first ? "" : ",", m, k, n, omega_gpu_matmul_rc_name(rc), pass ? "true" : "false",
                    info.elapsed_ns, info.call_ns / 1e6, wall_ms, info.max_abs_err, info.max_rel_err, info.mismatch_count,
                    info.parity_verified ? "true" : "false", info.completion_marker,
                    omega_gpu_matmul_rc_name(rc2), info2.kernel_cache_hit ? "true" : "false",
                    repeat_mismatch, e2e, e2e_bad, info.grid_x, info.grid_y, info.chip_calls, info.rows_per_call, info.padded_m, info.padded_k, info.padded_n, info.target_chip, info.sm_architecture);
            first = 0;
        }
        free(a); free(b); free(c); free(c2);
        if (omega_gpu_matmul_is_blocked()) { printf("device latched after an uncertain completion; stopping the sweep\n"); break; }
    }
    free(copy);
    if (out) {
        fprintf(out, "],\"blocked\":%s,\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", omega_gpu_matmul_is_blocked() ? "true" : "false", g_checks, g_failed,
                g_failed == 0 ? (mutant ? "OMEGA_GPU_MATMUL_MUTANT_KILLED" : "OMEGA_GPU_MATMUL_API_PASS") : (mutant ? "OMEGA_GPU_MATMUL_MUTANT_SURVIVED" : "OMEGA_GPU_MATMUL_API_FAIL"));
        fclose(out);
    }
    return 0;
}

static int cmp_double(const void *x, const void *y) { double a = *(const double *)x, b = *(const double *)y; return a < b ? -1 : a > b; }

/* Resident path: upload B once, then 100 calls; median wall time per call must be under 5 ms.
 * Parity: first and last result against the double reference (0 elements at or beyond 1e-5). */
static int timing(const char *shapes, const char *out_path) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_MATMUL_API_TIMING_V1\",\"cta_budget\":%u,\"spin_us\":%u,\"calls\":100,\"shapes\":[", omega_gpu_matmul_cta_budget(), omega_gpu_session_spin_us());
    char *copy = strdup(shapes); char *save = NULL; int first = 1;
    for (char *tok = strtok_r(copy, ";", &save); tok; tok = strtok_r(NULL, ";", &save)) {
        unsigned m, k, n;
        if (sscanf(tok, "%u,%u,%u", &m, &k, &n) != 3) { printf("bad shape %s\n", tok); continue; }
        size_t na = (size_t)m * k, nb = (size_t)k * n, nc = (size_t)m * n;
        float *a = malloc(na * sizeof *a), *b = malloc(nb * sizeof *b), *c = malloc(nc * sizeof *c), *c_first = malloc(nc * sizeof *c_first);
        uint16_t *ah = malloc(na * sizeof *ah);
        if (!a || !b || !c || !c_first || !ah) { printf("out of host memory\n"); return 2; }
        uint32_t seed = 0x7111e000u ^ (m << 20) ^ (k << 10) ^ n;
        fill(a, na, &seed); fill(b, nb, &seed);
        for (size_t i = 0; i < na; i++) ah[i] = omega_fp32_to_bf16(a[i]);
        OmegaGpuTensor *t = NULL;
        double tu0 = now_ms();
        int urc = omega_gpu_tensor_upload_f32(k, n, b, &t);
        double upload_ms = now_ms() - tu0;
        CHECK(urc == OMEGA_GPU_MATMUL_OK, "upload %ux%u rc=%s", k, n, omega_gpu_matmul_rc_name(urc));
        double samples[100]; int ok_calls = 0; OmegaGpuMatmulInfo info; memset(&info, 0, sizeof info);
        uint64_t chip_ns_sum = 0; size_t repeat_mismatch = 0;
        if (urc == OMEGA_GPU_MATMUL_OK) {
            for (int i = 0; i < 100; i++) {
                for (size_t j = 0; j < nc; j++) c[j] = -12345.0f;
                double t0 = now_ms();
                int rc = omega_gpu_matmul_resident_bf16(m, ah, t, c, &info);
                samples[i] = now_ms() - t0;
                /* OMEGA_GPU_MATMUL_TIMING_NO_ORACLE=1: the per-call host oracle (m x n x k double MACs) only on the first
                 * call; at m = 128 it is minutes per shape and the timing is the point (2026-10-12). */
                if (i == 0 && getenv("OMEGA_GPU_MATMUL_TIMING_NO_ORACLE")) omega_gpu_matmul_set_oracle(0);
                if (rc == OMEGA_GPU_MATMUL_OK) ok_calls++;
                chip_ns_sum += info.elapsed_ns;
                if (i == 0) memcpy(c_first, c, nc * sizeof *c);
                else for (size_t j = 0; j < nc; j++) if (c[j] != c_first[j]) repeat_mismatch++;
                if (omega_gpu_matmul_is_blocked()) break;
            }
        }
        qsort(samples, 100, sizeof samples[0], cmp_double);
        double median = samples[50], p90 = samples[90], mn = samples[0], mx = samples[99];
        size_t bad_first = 0, bad_last = 0;
        double e2e_first = e2e_max_rel(m, k, n, a, b, c_first, &bad_first);
        double e2e_last = e2e_max_rel(m, k, n, a, b, c, &bad_last);
        int pass = ok_calls == 100 && bad_first == 0 && bad_last == 0 && repeat_mismatch == 0 && (m > 1 || median < 5.0); /* the 5 ms gate is the GEMV (m = 1) gate */
        CHECK(pass, "timing %ux%ux%u ok_calls=%d median_ms=%.3f e2e_first=%g e2e_last=%g bad=%zu/%zu repeat_mismatch=%zu",
              m, k, n, ok_calls, median, e2e_first, e2e_last, bad_first, bad_last, repeat_mismatch);
        printf("%s timing %ux%ux%u upload_ms=%.1f ok_calls=%d median_ms=%.3f p90_ms=%.3f min_ms=%.3f max_ms=%.3f chip_ns_mean=%" PRIu64 " grid=%ux%u e2e_first=%g e2e_last=%g repeat_mismatch=%zu\n",
               pass ? "PASS" : "FAIL", m, k, n, upload_ms, ok_calls, median, p90, mn, mx, ok_calls ? chip_ns_sum / (uint64_t)ok_calls : 0, info.grid_x, info.grid_y, e2e_first, e2e_last, repeat_mismatch);
        if (out) {
            fprintf(out, "%s{\"m\":%u,\"k\":%u,\"n\":%u,\"pass\":%s,\"upload_ms\":%.3f,\"ok_calls\":%d,\"median_ms\":%.4f,\"p90_ms\":%.4f,\"min_ms\":%.4f,\"max_ms\":%.4f,"
                    "\"chip_ns_mean\":%" PRIu64 ",\"grid_x\":%u,\"grid_y\":%u,\"e2e_first\":%g,\"e2e_last\":%g,\"e2e_bad_first\":%zu,\"e2e_bad_last\":%zu,\"repeat_mismatch\":%zu}",
                    first ? "" : ",", m, k, n, pass ? "true" : "false", upload_ms, ok_calls, median, p90, mn, mx,
                    ok_calls ? chip_ns_sum / (uint64_t)ok_calls : 0, info.grid_x, info.grid_y, e2e_first, e2e_last, bad_first, bad_last, repeat_mismatch);
            first = 0;
        }
        omega_gpu_matmul_set_oracle(1);
        omega_gpu_tensor_free(t);
        free(a); free(b); free(c); free(c_first); free(ah);
        if (omega_gpu_matmul_is_blocked()) break;
    }
    free(copy);
    if (out) {
        fprintf(out, "],\"blocked\":%s,\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", omega_gpu_matmul_is_blocked() ? "true" : "false", g_checks, g_failed,
                g_failed == 0 ? "OMEGA_GPU_MATMUL_TIMING_PASS" : "OMEGA_GPU_MATMUL_TIMING_FAIL");
        fclose(out);
    }
    return 0;
}

/* --update (R3, omega #473): omega_gpu_tensor_update_* on the chip. Per shape (m rows of A, k x n weights):
 * upload B1, matmul -> c1; update_f32(B2), matmul -> c2; fresh upload of B2, matmul -> c2f. c2 must equal c2f
 * bit for bit and the double reference of A x B2 (0 elements at or beyond 1e-5); update_bf16(bf16(B1)) then
 * matmul must give c1 back bit for bit. Timing, median of 5 each: update_f32, update_bf16, and the
 * upload_f32 + free that an update replaces. */
static const char *UPDATE_SHAPES = "16,768,2304;16,768,50304";
static double median5(double *s) { qsort(s, 5, sizeof *s, cmp_double); return s[2]; }
static int update_mode(const char *shapes, const char *out_path) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_TENSOR_UPDATE_V1\",\"cta_budget\":%u,\"timed_calls\":5,\"shapes\":[", omega_gpu_matmul_cta_budget());
    char *copy = strdup(shapes); char *save = NULL; int first = 1;
    for (char *tok = strtok_r(copy, ";", &save); tok; tok = strtok_r(NULL, ";", &save)) {
        unsigned m, k, n;
        if (sscanf(tok, "%u,%u,%u", &m, &k, &n) != 3) { printf("bad shape %s\n", tok); continue; }
        size_t na = (size_t)m * k, nb = (size_t)k * n, nc = (size_t)m * n;
        float *a = malloc(na * sizeof *a), *b1 = malloc(nb * sizeof *b1), *b2 = malloc(nb * sizeof *b2);
        float *c1 = malloc(nc * sizeof *c1), *c2 = malloc(nc * sizeof *c2), *c2f = malloc(nc * sizeof *c2f), *c3 = malloc(nc * sizeof *c3);
        uint16_t *b1h = malloc(nb * sizeof *b1h);
        if (!a || !b1 || !b2 || !c1 || !c2 || !c2f || !c3 || !b1h) { printf("out of host memory\n"); return 2; }
        uint32_t seed = 0x0bda7e00u ^ (m << 20) ^ (k << 10) ^ n;
        fill(a, na, &seed); fill(b1, nb, &seed); fill(b2, nb, &seed);
        for (size_t i = 0; i < nb; i++) b1h[i] = omega_fp32_to_bf16(b1[i]);
        OmegaGpuTensor *t = NULL, *tf = NULL;
        int rc = omega_gpu_tensor_upload_f32(k, n, b1, &t);
        CHECK(rc == OMEGA_GPU_MATMUL_OK, "update %ux%ux%u: upload B1 rc=%s", m, k, n, omega_gpu_matmul_rc_name(rc));
        if (rc != OMEGA_GPU_MATMUL_OK) return 2;
        rc = omega_gpu_matmul_resident_f32(m, a, t, c1, NULL);
        CHECK(rc == OMEGA_GPU_MATMUL_OK, "matmul B1 rc=%s", omega_gpu_matmul_rc_name(rc));
        rc = omega_gpu_tensor_update_f32(t, k, n, b2);
        CHECK(rc == OMEGA_GPU_MATMUL_OK, "update_f32 rc=%s %s", omega_gpu_matmul_rc_name(rc), omega_gpu_matmul_last_error());
        rc = omega_gpu_matmul_resident_f32(m, a, t, c2, NULL);
        CHECK(rc == OMEGA_GPU_MATMUL_OK, "matmul after update rc=%s", omega_gpu_matmul_rc_name(rc));
        rc = omega_gpu_tensor_upload_f32(k, n, b2, &tf);
        CHECK(rc == OMEGA_GPU_MATMUL_OK, "fresh upload B2 rc=%s", omega_gpu_matmul_rc_name(rc));
        if (rc == OMEGA_GPU_MATMUL_OK) rc = omega_gpu_matmul_resident_f32(m, a, tf, c2f, NULL);
        CHECK(rc == OMEGA_GPU_MATMUL_OK, "matmul fresh B2 rc=%s", omega_gpu_matmul_rc_name(rc));
        omega_gpu_tensor_free(tf);
        size_t vs_fresh = 0, back = 0, bad = 0;
        for (size_t j = 0; j < nc; j++) vs_fresh += memcmp(&c2[j], &c2f[j], 4) != 0;
        double e2e = e2e_max_rel(m, k, n, a, b2, c2, &bad);
        rc = omega_gpu_tensor_update_bf16(t, k, n, b1h);
        CHECK(rc == OMEGA_GPU_MATMUL_OK, "update_bf16 rc=%s %s", omega_gpu_matmul_rc_name(rc), omega_gpu_matmul_last_error());
        rc = omega_gpu_matmul_resident_f32(m, a, t, c3, NULL);
        CHECK(rc == OMEGA_GPU_MATMUL_OK, "matmul after update back rc=%s", omega_gpu_matmul_rc_name(rc));
        for (size_t j = 0; j < nc; j++) back += memcmp(&c3[j], &c1[j], 4) != 0;
        CHECK(vs_fresh == 0, "update %ux%ux%u: %zu of %zu outputs differ from a fresh upload of the same values", m, k, n, vs_fresh, nc);
        CHECK(bad == 0, "update %ux%ux%u: %zu outputs off the double reference (worst %g)", m, k, n, bad, e2e);
        CHECK(back == 0, "update %ux%ux%u: %zu of %zu outputs differ after updating back to B1", m, k, n, back, nc);
        double s_f32[5], s_bf16[5], s_up[5];
        int timed_ok = 1;
        for (int i = 0; i < 5; i++) {
            double t0 = now_ms(); timed_ok &= omega_gpu_tensor_update_f32(t, k, n, b2) == OMEGA_GPU_MATMUL_OK; s_f32[i] = now_ms() - t0;
            t0 = now_ms(); timed_ok &= omega_gpu_tensor_update_bf16(t, k, n, b1h) == OMEGA_GPU_MATMUL_OK; s_bf16[i] = now_ms() - t0;
            t0 = now_ms(); timed_ok &= omega_gpu_tensor_upload_f32(k, n, b2, &tf) == OMEGA_GPU_MATMUL_OK; omega_gpu_tensor_free(tf); s_up[i] = now_ms() - t0;
        }
        CHECK(timed_ok, "update %ux%ux%u: a timed call failed", m, k, n);
        double mf = median5(s_f32), mb = median5(s_bf16), mu = median5(s_up);
        int pass = vs_fresh == 0 && bad == 0 && back == 0 && timed_ok;
        printf("%s update %ux%ux%u vs_fresh=%zu e2e=%g bad=%zu back=%zu update_f32_ms=%.2f update_bf16_ms=%.2f upload_f32_free_ms=%.2f\n",
               pass ? "PASS" : "FAIL", m, k, n, vs_fresh, e2e, bad, back, mf, mb, mu);
        if (out) {
            fprintf(out, "%s{\"m\":%u,\"k\":%u,\"n\":%u,\"pass\":%s,\"vs_fresh_diff\":%zu,\"e2e_max_rel\":%g,\"e2e_bad\":%zu,\"back_diff\":%zu,"
                    "\"update_f32_ms\":%.3f,\"update_bf16_ms\":%.3f,\"upload_f32_free_ms\":%.3f}",
                    first ? "" : ",", m, k, n, pass ? "true" : "false", vs_fresh, e2e, bad, back, mf, mb, mu);
            first = 0;
        }
        omega_gpu_tensor_free(t);
        free(a); free(b1); free(b2); free(c1); free(c2); free(c2f); free(c3); free(b1h);
        if (omega_gpu_matmul_is_blocked()) break;
    }
    free(copy);
    if (out) {
        fprintf(out, "],\"blocked\":%s,\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", omega_gpu_matmul_is_blocked() ? "true" : "false", g_checks, g_failed,
                g_failed == 0 ? "OMEGA_GPU_TENSOR_UPDATE_PASS" : "OMEGA_GPU_TENSOR_UPDATE_FAIL");
        fclose(out);
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *out_path = NULL;
    int host = 0, mutant = 0, do_timing = 0, do_update = 0, do_stages = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--mutant") == 0) mutant = 1;
        else if (strcmp(argv[i], "--timing") == 0) do_timing = 1;
        else if (strcmp(argv[i], "--update") == 0) do_update = 1;
        else if (strcmp(argv[i], "--stages") == 0) do_stages = 1;
        else if (strcmp(argv[i], "--cta-budget") == 0 && i + 1 < argc) omega_gpu_matmul_set_cta_budget((uint32_t)strtoul(argv[++i], NULL, 10));
        else if (strcmp(argv[i], "--spin-us") == 0 && i + 1 < argc) {
            if (omega_gpu_session_set_spin_us((uint32_t)strtoul(argv[++i], NULL, 10)) != 0) { fprintf(stderr, "--spin-us out of range\n"); return 2; }
        }
        else if (strcmp(argv[i], "--unroll") == 0 && i + 1 < argc) omega_gpu_matmul_test_set_unroll((uint32_t)strtoul(argv[++i], NULL, 10));
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    }
    const char *shapes = getenv("OMEGA_GPU_MATMUL_SHAPES");
    if (shapes && !*shapes) shapes = NULL;
    int rc = 0;
    if (host) host_only();
    else if (do_timing) rc = timing(shapes ? shapes : TIMING_SHAPES, out_path);
    else if (do_update) rc = update_mode(shapes ? shapes : UPDATE_SHAPES, out_path);
    else if (do_stages) rc = stages_profile(shapes ? shapes : STAGE_SHAPES, out_path);
    else if (mutant) { omega_gpu_matmul_test_set_mutant(1); rc = chip_sweep(shapes ? shapes : MUTANT_SHAPES, out_path, 1); omega_gpu_matmul_test_set_mutant(0); }
    else rc = chip_sweep(shapes ? shapes : DEFAULT_SHAPES, out_path, 0);
    if (!host && !do_timing && !do_update && !do_stages && !mutant && !omega_gpu_matmul_is_blocked()) one_row_repack_checks();
    if (!omega_gpu_matmul_is_blocked()) omega_gpu_device_close();
    if (rc) return 2;
    printf("%s: %d checks, %d failed%s\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed, mutant ? (g_failed ? " (mutant SURVIVED)" : " (mutant killed)") : "");
    return g_failed ? 1 : 0;
}
