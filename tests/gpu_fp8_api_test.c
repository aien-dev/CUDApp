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

/* gpu_fp8_api_test: OWN-30B Cut 6 / Cut 5 gate for omega_gpu_fp8_api (FP8 e4m3 block-scaled GEMV).
 *
 *   ./gpu_fp8_api_test --host-only          exhaustive 256-code e4m3 conversion vs the reference decode
 *                                           (subnormals, both zeros, NaN codes refused), pack refusals,
 *                                           codegen for every kernel shape (instruction and register budget),
 *                                           API argument refusals, listing by Omega's own disassembler
 *   ./gpu_fp8_api_test --sim                the kernel through tests/bw_warp_sim.h (no chip): one-hot conversion
 *                                           of every code through the kernel, parity battery vs the double oracle
 *                                           (both fragment orders bit-identical), grouped experts, and the four
 *                                           mutants (1 scale dropped, 2 scale block frozen, 3 group table swapped,
 *                                           host no-fold) which the same check must catch
 *   ./gpu_fp8_api_test --out receipt.json   chip gate: parity battery, mutants, grouped experts, then the
 *                                           bandwidth table (median of --reps calls) at the expert and attention shapes
 *   ./gpu_fp8_api_test --cold-single      SPEED-1B: the decode graph's two single projections (qkv 5120 x 2048, o_proj 2048 x 4096),
 *                                         16 copies rotated so L2 never holds the launch's weights; same report as --grouped-cold
 *   ./gpu_fp8_api_test --grouped-cold     CUT5: the decode graph's two expert launches (gate_up 8 of 128 x 1536 x 2048,
 *                                           down 8 of 128 x 2048 x 768) with L2 defeated by rotating 16 disjoint expert
 *                                           sets; parity on every timed launch; mutants 1..4; GB/s and fraction of 237.2
 *   knobs: --lane-bytes N --rows N --warps N --cta-budget N --reps N --spin-us N
 *   CUT5 variants: --coal 0|1 (A, coalesced output; default 1, AIEN_OMEGA_FP8_COAL=0 turns it off; --coal-default checks that) --wfirst 0|1 (B, weights before x); FP8_SIM_ALL_ORDERS=1 runs every
 *   simulated kernel in four warp orders (coal kernels always do)
 *          --sched 0|1 (CUT5 kernel schedule) --variant 1|2 --depth N --no-evict --only NAME
 *          --n N --k K: one custom point n x k instead of the battery (with mutants 1, 2, 4), e.g. 128256 x 2560
 *
 * Pass rule per point: rc OK, every output within TOL of the double oracle relative to the accumulation
 * scale (sum of |terms|). Host and simulator passes are never chip results.
 */
#include "omega_gpu_fp8_api.h"
#include "omega_blackwell_codegen_moe2.h"
#include "omega_blackwell_codegen_moe.h"
#include "omega_blackwell_matmul.h"
#include "omega_gpu_session.h"
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_qmd.h" /* OMEGA_BW_QMD_MAX_SHARED_BYTES (deep_default_check) */
#include "bw_warp_sim.h"
#include "omega_sm121_disasm.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define TOL 1e-5
static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }
static uint8_t rand_e4m3(uint32_t *s) { uint8_t b = (uint8_t)(lcg(s) >> 24); if (omega_fp8_e4m3_is_nan(b)) b = (uint8_t)(b - 1); return b; }
static uint16_t bf16_of(float f) { return omega_fp32_to_bf16(f); }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }

/* ------------------------------------------------------------------ data */
typedef struct {
    uint32_t experts, n, k;
    uint8_t *w; uint16_t *s; /* checkpoint layout */
} Weights;
static void weights_make(Weights *W, uint32_t experts, uint32_t n, uint32_t k, uint32_t seed) {
    W->experts = experts; W->n = n; W->k = k;
    size_t wb = (size_t)experts * n * k, sc = (size_t)experts * (n / 128) * (k / 128);
    W->w = aligned_alloc(64, (wb + 63) & ~(size_t)63); W->s = malloc(sc * 2);
    uint32_t st = seed;
    for (size_t i = 0; i < wb; i++) W->w[i] = rand_e4m3(&st);
    for (size_t i = 0; i < sc; i++) W->s[i] = bf16_of(frand(&st, 0.001f, 0.1f) * ((lcg(&st) & 1) ? 1.0f : -1.0f));
}
static void weights_free(Weights *W) { free(W->w); free(W->s); }

/* max relative error of y (groups x n) against the oracle over the selected experts */
static double parity(const Weights *W, uint32_t groups, const uint32_t *ids, const float *x, uint32_t x_rows, const float *y, size_t *bad) {
    double worst = 0.0; size_t nbad = 0;
    double *ref = malloc(W->n * sizeof *ref), *mag = malloc(W->n * sizeof *mag);
    for (uint32_t g = 0; g < groups; g++) {
        const uint8_t *w = W->w + (size_t)ids[g] * W->n * W->k;
        const uint16_t *s = W->s + (size_t)ids[g] * (W->n / 128) * (W->k / 128);
        omega_fp8_gemv_oracle(w, s, W->n, W->k, x + (size_t)(x_rows == 1 ? 0 : g) * W->k, ref, mag);
        for (uint32_t r = 0; r < W->n; r++) {
            double d = fabs(ref[r] - (double)y[(size_t)g * W->n + r]);
            double rel = d / (mag[r] > 1e-30 ? mag[r] : 1e-30);
            if (rel > worst) worst = rel;
            if (rel >= TOL || isnan(y[(size_t)g * W->n + r])) nbad++;
        }
    }
    free(ref); free(mag);
    if (bad) *bad = nbad;
    return worst;
}

/* ---------------------------------------------- stall checker (CUT5) */
/* Independent of the generator's pass (a different walk, as tests/gpu_tern_api_test.c does): for every fixed-latency
 * write, scan forward summing stalls; any instruction that reads or overwrites one of its registers, any control
 * instruction and any branch target reached before OMEGA_FP8_FIXED_LAT cycles is a violation. */
static int fp8_chk_fixed(BlackwellIROpcode op) {
    return op == BW_IR_IMAD || op == BW_IR_IMAD_WIDE || op == BW_IR_SHF_R || op == BW_IR_LOP3_AND || op == BW_IR_LOP3_XOR ||
           op == BW_IR_MOV_IMM || op == BW_IR_MOV_RZ || op == BW_IR_FFMA || op == BW_IR_FADD;
}
static int fp8_chk_overlap(const BlackwellIRProgram *p, int a, int b) {
    if (a < 0 || b < 0) return 0;
    const int pa = p->regalloc.vreg_to_phys[a], pb = p->regalloc.vreg_to_phys[b];
    if (pa < 0 || pb < 0) return 0;
    const int na = (int)(p->regalloc.intervals[a].bundle_size ? p->regalloc.intervals[a].bundle_size : 1u);
    const int nb = (int)(p->regalloc.intervals[b].bundle_size ? p->regalloc.intervals[b].bundle_size : 1u);
    return pa < pb + nb && pb < pa + na;
}
/* CUT5 A/B structure counters (independent of the generator): instructions of one opcode, and global-load batches. A batch
 * opens at a global load issued while no batch is open; any instruction whose wait mask holds the load scoreboard (bit 0 of
 * the wait field, SB_GPR in the generator) closes it. One batch = one memory round trip the warp waits for. */
static int fp8_count_op(const BlackwellIRProgram *p, BlackwellIROpcode op) { int n = 0; for (size_t i = 0; i < p->count; i++) n += p->insns[i].op == op; return n; }
static int fp8_load_batches(const BlackwellIRProgram *p) {
    int n = 0, open = 0;
    for (size_t i = 0; i < p->count; i++) {
        const BlackwellIRInsn *in = &p->insns[i];
        if ((in->control >> 20) & 1u) open = 0;
        const int gl = in->op == BW_IR_LDG_E || in->op == BW_IR_LDG_E_64 || in->op == BW_IR_LDG_E_128;
        if (gl && !open) { n++; open = 1; }
    }
    return n;
}
static long fp8_stall_violations(const BlackwellIRProgram *p, uint64_t *cycles) {
    static uint8_t tgt[BW_MAX_IR_INSNS + 1];
    memset(tgt, 0, sizeof tgt);
    long bad = 0; uint64_t cyc = 0;
    for (size_t i = 0; i < p->count; i++) {
        const BlackwellIRInsn *in = &p->insns[i];
        cyc += (in->control >> 9) & 0xfu;
        if (in->op == BW_IR_BRA || in->op == BW_IR_BSSY) { long t = (long)i + (long)(int32_t)in->imm; if (t >= 0 && t <= (long)p->count) tgt[t] = 1; }
    }
    for (size_t i = 0; i < p->count; i++) {
        const BlackwellIRInsn *in = &p->insns[i];
        if (!fp8_chk_fixed(in->op) || in->dst_vreg < 0) continue;
        uint32_t dist = 0;
        for (size_t j = i + 1; j < p->count; j++) {
            dist += (p->insns[j - 1].control >> 9) & 0xfu;
            if (dist >= OMEGA_FP8_FIXED_LAT) break;
            const BlackwellIRInsn *u = &p->insns[j];
            const BlackwellIROpcode o = u->op;
            if (o == BW_IR_BRA || o == BW_IR_EXIT || o == BW_IR_BSSY || o == BW_IR_BSYNC || o == BW_IR_BAR_SYNC || tgt[j]) { bad++; break; }
            if (fp8_chk_overlap(p, in->dst_vreg, u->dst_vreg) || fp8_chk_overlap(p, in->dst_vreg, u->src1_vreg) ||
                fp8_chk_overlap(p, in->dst_vreg, u->src2_vreg) || fp8_chk_overlap(p, in->dst_vreg, u->src3_vreg)) { bad++; break; }
        }
    }
    if (cycles) *cycles = cyc;
    return bad;
}

/* ------------------------------------------------------------- host-only */
static int sm121dis_check_v(int variant, const OmegaFp8GemvSpec *spec, const char *name);
static int sm121dis_check(const OmegaFp8GemvSpec *spec, const char *name) { return sm121dis_check_v(1, spec, name); }
static int sm121dis_check_v(int variant, const OmegaFp8GemvSpec *spec, const char *name) {
    OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
    if ((variant == 2 ? omega_blackwell_codegen_fp8_gemv2(spec, &k) : omega_blackwell_codegen_fp8_gemv(spec, &k)) != 0) return -1;
    char lst[96];
    snprintf(lst, sizeof lst, "/tmp/omega_fp8_%s_%d.sass", name, (int)getpid());
    FILE *f = fopen(lst, "w"); if (!f) { omega_blackwell_kernel_free(&k); return -1; }
    int rc = omega_sm121_listing(k.code, k.code_size, f); fclose(f);
    omega_blackwell_kernel_free(&k);
    static const char *const must[] = { "LDG.E.128", "FFMA", "SHFL.DOWN", "BSSY", "BSYNC", "STG.E", "LOP3.LUT", "IMAD" };
    FILE *l = fopen(lst, "r"); if (!l) return -1;
    char line[512]; int seen[8] = {0}; int bad = 0, insns = 0;
    while (fgets(line, sizeof line, l)) {
        if (!strstr(line, "/*") || !strstr(line, "*/")) continue; /* every address line */
        insns++;
        if (strstr(line, "?") || strstr(line, "error")) bad++;
        if (strstr(line, ".FTZ")) { printf("  %s: flush-to-zero instruction: %s", name, line); bad++; } /* e4m3 subnormals travel as f32 subnormals */
        for (int i = 0; i < 8; i++) if (strstr(line, must[i])) seen[i] = 1;
    }
    fclose(l);
    int miss = 0;
    for (int i = 0; i < 8; i++) if (!seen[i]) { printf("  %s: sm121dis listing lacks %s\n", name, must[i]); miss++; }
    printf("sm121dis %s: rc=%d insns=%d undecodable=%d missing=%d (%s)\n", name, rc, insns, bad, miss, lst);
    return (rc == 0 && bad == 0 && miss == 0) ? 0 : -1;
}

/* Variant A default (SPEED-A, 2026-10-17): coal is ON unless AIEN_OMEGA_FP8_COAL=0 (or set_coal(0)), and coal 0 gives back
 * the earlier kernel (one STG per row, no BAR.SYNC, no shared slots) at the decode graph's two grouped expert shapes and the
 * single-projection o_proj shape. make test-gpu-fp8 runs this once with the env unset and once with AIEN_OMEGA_FP8_COAL=0. */
static int g_coal_arg; /* --coal given: the env default is not what is being tested */
static void coal_default_check(void) {
    const char *e = getenv("AIEN_OMEGA_FP8_COAL");
    const int want = (e && e[0] == '0' && e[1] == 0) ? 0 : 1;
    CHECK(omega_gpu_fp8_coal() == want, "coal default %d with AIEN_OMEGA_FP8_COAL=%s (want %d)", omega_gpu_fp8_coal(), e ? e : "(unset)", want);
    static const struct { uint32_t k, n, groups; const char *name; } sh[] = {
        { 2048, 1536, 8, "gate_up 8 x 1536 x 2048" }, { 768, 2048, 8, "down 8 x 2048 x 768" }, { 4096, 2048, 1, "o_proj 2048 x 4096" } };
    for (int pass = 0; pass < 3; pass++) {
        const int on = pass == 0 ? want : pass == 1 ? 0 : 1;
        if (pass) omega_gpu_fp8_set_coal(on);
        for (size_t q = 0; q < sizeof sh / sizeof sh[0]; q++) {
            OmegaFp8GemvSpec sp; omega_gpu_fp8_spec_for(sh[q].k, sh[q].n, sh[q].groups, &sp);
            const int fits = sp.warps * sp.rows <= 32u;
            static BlackwellIRProgram p;
            const int rc = omega_blackwell_codegen_fp8_gemv_prog(&sp, &p);
            const int coal = on && fits, bar = rc == 0 ? fp8_count_op(&p, BW_IR_BAR_SYNC) : -1, stg = rc == 0 ? fp8_count_op(&p, BW_IR_STG_E) : -1;
            CHECK(rc == 0 && sp.coal == (uint32_t)coal && stg == 1 && (sp.depth ? 1 : bar == coal) && (omega_fp8_gemv_shared_bytes(&sp) != 0) == (coal || sp.depth != 0),
                  "%s coal %d (pass %d, warps %u rows %u depth %u): spec.coal %u, STG %d, BAR.SYNC %d, shared %u", sh[q].name, on, pass, sp.warps, sp.rows, sp.depth, sp.coal, stg, bar, omega_fp8_gemv_shared_bytes(&sp));
            if (pass == 0) printf("coal default %d: %s warps %u rows %u depth %u -> spec.coal %u, STG %d, BAR.SYNC %d\n", want, sh[q].name, sp.warps, sp.rows, sp.depth, sp.coal, stg, bar);
        }
    }
    omega_gpu_fp8_set_coal(want);
}

/* Deep-kernel default (SPEED-1B, 2026-10-17): the automatic spec for the decode graph's qkv (5120 x 2048, 10.5 MB) is the deep
 * kernel (depth KS = 4) unless AIEN_OMEGA_FP8_DEEP=0 (or set_deep(0)); o_proj (2048 x 4096) stays classic because its deep x stage
 * would exceed the 16 KiB launch-descriptor shared limit; grouped expert launches stay classic. make test-gpu-fp8 runs this once
 * with the env unset and once with AIEN_OMEGA_FP8_DEEP=0. */
static void deep_default_check(void) {
    const char *e = getenv("AIEN_OMEGA_FP8_DEEP");
    const int want = (e && e[0] == '0' && e[1] == 0) ? 0 : 1;
    CHECK(omega_gpu_fp8_deep() == want, "deep default %d with AIEN_OMEGA_FP8_DEEP=%s (want %d)", omega_gpu_fp8_deep(), e ? e : "(unset)", want);
    for (int pass = 0; pass < 3; pass++) {
        const int on = pass == 0 ? want : pass == 1 ? 0 : 1;
        if (pass) omega_gpu_fp8_set_deep(on);
        OmegaFp8GemvSpec q, o, gu; omega_gpu_fp8_spec_for(2048, 5120, 1, &q); omega_gpu_fp8_spec_for(4096, 2048, 1, &o); omega_gpu_fp8_spec_for(2048, 1536, 8, &gu);
        CHECK(q.depth == (on ? 4u : 0u) && o.depth == 0 && gu.depth == 0, "deep %d (pass %d): qkv depth %u (want %u), o_proj depth %u (want 0), gate_up depth %u (want 0)", on, pass, q.depth, on ? 4u : 0u, o.depth, gu.depth);
        OmegaFp8GemvSpec od = o; od.depth = 4096u / (32u * od.lane_bytes); od.mutant = 0;
        CHECK(omega_fp8_gemv_shared_bytes(&od) > OMEGA_GPU_FP8_DEEP_MAX_SHARED_BYTES, "o_proj deep x stage %u bytes must exceed the %u-byte measured deep bound (why it stays classic)", omega_fp8_gemv_shared_bytes(&od), OMEGA_GPU_FP8_DEEP_MAX_SHARED_BYTES);
        if (on) CHECK(omega_fp8_gemv_shared_bytes(&q) != 0 && omega_fp8_gemv_shared_bytes(&q) <= OMEGA_GPU_FP8_DEEP_MAX_SHARED_BYTES, "qkv deep shared %u bytes within the measured deep bound", omega_fp8_gemv_shared_bytes(&q));
        if (pass == 0) printf("deep default %d: qkv 5120x2048 depth %u (coal %u, shared %u), o_proj 2048x4096 depth %u, gate_up 8x1536x2048 depth %u\n", want, q.depth, q.coal, omega_fp8_gemv_shared_bytes(&q), o.depth, gu.depth);
    }
    /* The choice runs on every launch and every graph record (48 qkv records per token). SPEED-1B graph A/B: an unremembered
     * register check (full codegen per call) cost 853..1172 us of host record time per token. Bound: 4800 calls (100 tokens of
     * qkv records) under 20 ms in total (< ~4 us per call); the codegen path is ~17 us per call. */
    {
        omega_gpu_fp8_set_deep(1);
        OmegaFp8GemvSpec q; omega_gpu_fp8_spec_for(2048, 5120, 1, &q); /* warm */
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        uint32_t dsum = 0;
        for (int i = 0; i < 4800; i++) { omega_gpu_fp8_spec_for(2048, 5120, 1, &q); dsum += q.depth; }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double ms = (double)(t1.tv_sec - t0.tv_sec) * 1e3 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
        CHECK(dsum == 4800u * 4u && ms < 20.0, "4800 qkv spec_for calls (deep on): %.2f ms, depth sum %u (want < 20 ms, %u)", ms, dsum, 4800u * 4u);
        printf("deep default: 4800 qkv spec_for calls %.3f ms\n", ms);
    }
    omega_gpu_fp8_set_deep(want);
}

/* qwen38/decode-speed: the host quantiser (FP8 lm_head). Every e4m3 rounding against a brute-force nearest-code search
 * (ties to the even code, saturation at 448, never 0x7F), then a block round trip against the upload format's maths. */
static double e4m3_val(unsigned c) { unsigned e = (c >> 3) & 15u, m = c & 7u; double v = e ? (1.0 + m / 8.0) * pow(2.0, (int)e - 7) : m / 8.0 * pow(2.0, -6); return (c & 0x80u) ? -v : v; }
static unsigned e4m3_brute(float v) {
    double a = fabs((double)v); unsigned best = 0; double bd = 1e300;
    if (a >= 448.0) return (v < 0 ? 0x80u : 0u) | 0x7Eu;
    for (unsigned c = 0; c <= 0x7Eu; c++) { double d = fabs(e4m3_val(c) - a); if (d < bd || (d == bd && !(c & 1u))) { bd = d; best = c; } }
    return ((v < 0 || (v == 0 && signbit(v))) ? 0x80u : 0u) | best;
}
static void e4m3_quant_check(void) {
    int bad = 0, nan = 0; uint32_t s = 12345u;
    for (unsigned c = 0; c <= 0x7Eu; c++) { /* every code, the midpoints to both neighbours, and their negatives */
        float pts[3] = { (float)e4m3_val(c), c ? (float)((e4m3_val(c) + e4m3_val(c - 1)) / 2) : 0.0f, c < 0x7E ? (float)((e4m3_val(c) + e4m3_val(c + 1)) / 2) : 460.0f };
        for (int i = 0; i < 3; i++) for (int sg = 0; sg < 2; sg++) { float v = sg ? -pts[i] : pts[i]; unsigned q = omega_gpu_fp8_e4m3_rne(v); if (q != e4m3_brute(v)) bad++; if ((q & 0x7Fu) == 0x7Fu) nan++; }
    }
    for (int i = 0; i < 200000; i++) { /* random magnitudes over 2^-12 .. 2^10 */
        s = s * 1664525u + 1013904223u; float m = (float)(s >> 8) / 16777216.0f; s = s * 1664525u + 1013904223u;
        float v = (1.0f + m) * ldexpf(1.0f, (int)(s >> 27) - 12 - (int)((s >> 20) & 3u)); if (s & 1u) v = -v;
        unsigned q = omega_gpu_fp8_e4m3_rne(v); if (q != e4m3_brute(v)) bad++; if ((q & 0x7Fu) == 0x7Fu) nan++;
    }
    CHECK(bad == 0 && nan == 0, "e4m3 rounding: %d codes differ from the brute-force nearest-even, %d NaN codes", bad, nan);
    CHECK(omega_gpu_fp8_e4m3_rne(1e30f) == 0x7Eu && omega_gpu_fp8_e4m3_rne(-1e30f) == 0xFEu && omega_gpu_fp8_e4m3_rne(0.0f) == 0u, "e4m3 saturation and zero");
    /* block round trip: 256 x 384 bf16 with a zero block, w8 * scale within half an e4m3 step of w, scale >= amax / 448 */
    enum { N = 256, K = 384 }; static uint16_t w[N * K], sc[(N / 128) * (K / 128)]; static uint8_t w8[N * K];
    for (int i = 0; i < N * K; i++) { s = s * 1664525u + 1013904223u; float f = ((float)(s >> 8) / 16777216.0f - 0.5f) * ((i / K) < 128 ? 0.02f : 3.0f); uint32_t b; memcpy(&b, &f, 4); w[i] = (uint16_t)(b >> 16); }
    for (int r = 0; r < 128; r++) for (int c = 128; c < 256; c++) w[r * K + c] = 0;
    CHECK(omega_gpu_fp8_quantize_bf16(N, K, w, w8, sc) == OMEGA_GPU_FP8_OK, "quantize 256 x 384");
    int far = 0, nanq = 0; double worst = 0;
    for (int r = 0; r < N; r++) for (int c = 0; c < K; c++) {
        uint32_t b = (uint32_t)sc[(r / 128) * (K / 128) + c / 128] << 16; float scf; memcpy(&scf, &b, 4);
        uint32_t wb = (uint32_t)w[r * K + c] << 16; float wf; memcpy(&wf, &wb, 4);
        double back = e4m3_val(w8[r * K + c]) * scf, err = fabs(back - wf), rel = err / fabs((double)wf ? wf : 1);
        if ((w8[r * K + c] & 0x7Fu) == 0x7Fu) nanq++;
        if (w8[r * K + c] != e4m3_brute(wf / scf)) far++;
        if (wf != 0 && fabs(wf) > 0.1 * scf && rel > worst) worst = rel;
    }
    CHECK(far == 0 && nanq == 0 && worst <= 0.0625 + 1e-9, "quantize round trip: %d codes off the nearest, %d NaN, worst rel %.4f (<= 1/16)", far, nanq, worst);
    CHECK(sc[1] == 0x3F80u, "all-zero block gets scale 1 (0x%04x)", sc[1]);
    uint16_t inf[128 * 128] = { 0 }; inf[77] = 0x7F80u; uint8_t o8[128 * 128]; uint16_t os[1];
    CHECK(omega_gpu_fp8_quantize_bf16(128, 128, inf, o8, os) == OMEGA_GPU_FP8_BAD_ARGS, "quantize refuses an inf weight");
}

/* qwen38/decode-speed: the big-k rule's choices (chip parity of these specs: fp8par1 in the speed handoff) */
static void bigk_check(void) {
    static const struct { uint32_t n, k, warps, depth; } want[] = { { 34816, 5120, 16, 5 }, { 16384, 5120, 16, 5 }, { 14336, 5120, 16, 5 },
                                                                    { 5120, 6144, 16, 6 }, { 5120, 17408, 16, 0 }, { 2048, 4096, 8, 0 }, { 5120, 2048, 8, 4 } };
    for (int on = 0; on < 2; on++) {
        omega_gpu_fp8_set_bigk(on);
        for (size_t i = 0; i < sizeof want / sizeof want[0]; i++) {
            OmegaFp8GemvSpec sp; omega_gpu_fp8_spec_for(want[i].k, want[i].n, 1, &sp);
            const int big = on && want[i].k > 4096u;
            const uint32_t ww = big ? want[i].warps : 8u, wd = big ? want[i].depth : (want[i].k == 2048u ? 4u : 0u);
            CHECK(sp.warps == ww && sp.depth == wd && sp.coal == 1u, "bigk %d: %ux%u warps %u depth %u coal %u (want %u, %u, 1)", on, want[i].n, want[i].k, sp.warps, sp.depth, sp.coal, ww, wd);
        }
        OmegaFp8GemvSpec g8; omega_gpu_fp8_spec_for(5120, 34816, 8, &g8);
        CHECK(g8.warps == 32u && g8.depth == 0u, "bigk %d: grouped launches keep 32 warps, classic (warps %u depth %u)", on, g8.warps, g8.depth);
    }
    omega_gpu_fp8_set_bigk(1); /* the default */
}

static void host_only(void) {
    e4m3_quant_check();
    bigk_check();
    if (!g_coal_arg) coal_default_check();
    deep_default_check();
    /* 1. exhaustive conversion: kernel bits * 2^120 == reference, bit for bit, for every non-NaN code;
     *    sign of zero preserved; NaN codes are exactly 0x7f and 0xff and are refused by the scan */
    int exact = 0, nan_codes = 0, zero_sign_ok = 0, f32_subnormal = 0;
    for (int b = 0; b < 256; b++) {
        float ref = omega_fp8_e4m3_ref((uint8_t)b);
        uint32_t kb = omega_fp8_e4m3_kernel_bits((uint8_t)b);
        float kf; memcpy(&kf, &kb, 4);
        /* the kernel-side f32 is subnormal exactly for the e4m3 subnormal codes (exponent field 0, mantissa != 0):
         * the chip must not flush them (no .FTZ instruction in the listing, chip one-hot test) */
        if ((kb & 0x7f800000u) == 0 && (kb & 0x007fffffu) != 0) { f32_subnormal++; CHECK((b & 0x78) == 0 && (b & 7) != 0, "code 0x%02x maps to an f32 subnormal but is not an e4m3 subnormal", b); }
        else CHECK((b & 0x78) != 0 || (b & 7) == 0, "e4m3 subnormal code 0x%02x maps to a normal f32", b);
        float kv = ldexpf(kf, 120); /* exact: a power-of-two scaling (host, no flush to zero) */
        if (isnan(ref)) { nan_codes++; CHECK(omega_fp8_e4m3_is_nan((uint8_t)b), "code 0x%02x is NaN in the reference and must be flagged", b); continue; }
        CHECK(!omega_fp8_e4m3_is_nan((uint8_t)b), "code 0x%02x is finite in the reference and must not be flagged", b);
        uint32_t rb, vb; memcpy(&rb, &ref, 4); memcpy(&vb, &kv, 4);
        if (rb == vb) exact++;
        CHECK(rb == vb, "code 0x%02x: kernel conversion %a (bits %08x) != reference %a (bits %08x)", b, kv, vb, ref, rb);
        if (ref == 0.0f) { if (signbit(ref) == signbit(kv)) zero_sign_ok++; }
        /* the fold is exact as two steps too (x * 2^64 then * 2^56) */
        CHECK(ldexpf(ldexpf(kf, OMEGA_FP8_X_EXP), OMEGA_FP8_SCALE_EXP) == ref, "code 0x%02x: split fold not exact", b);
    }
    CHECK(exact == 254 && nan_codes == 2 && zero_sign_ok == 2, "conversion: %d exact, %d NaN codes, %d zero signs (want 254, 2, 2)", exact, nan_codes, zero_sign_ok);
    CHECK(f32_subnormal == 14, "f32-subnormal kernel values: %d (want 14: +-0x01..0x07)", f32_subnormal);
    printf("conversion: %d/256 exact, %d NaN codes refused, %d codes travel as f32 subnormals (2^-129 .. 7 * 2^-129): the chip must not flush them\n", exact, nan_codes, f32_subnormal);
    /* subnormal spot values from the e4m3 definition */
    CHECK(omega_fp8_e4m3_ref(0x01) == ldexpf(1.0f, -9) && omega_fp8_e4m3_ref(0x07) == 7.0f * ldexpf(1.0f, -9), "subnormal spot values");
    CHECK(omega_fp8_e4m3_ref(0x7e) == 448.0f && omega_fp8_e4m3_ref(0xfe) == -448.0f && omega_fp8_e4m3_ref(0x38) == 1.0f, "normal spot values");
    /* conversion mutant: skipping the fold is caught by the same comparison */
    { float kf; uint32_t kb = omega_fp8_e4m3_kernel_bits(0x38); memcpy(&kf, &kb, 4); CHECK(kf != 1.0f, "fold mutant: unfolded bits must not equal the value"); }

    /* 2. pack refusals */
    { uint8_t w[8] = { 0x38, 0x7f, 0x00, 0xff, 1, 2, 3, 4 }; size_t first = 99;
      CHECK(omega_fp8_pack_scan_nan(w, 8, &first) == 2 && first == 1, "NaN scan finds both codes, first at 1");
      CHECK(omega_fp8_pack_scan_nan(w, 1, &first) == 0, "NaN scan clean prefix"); }
    { uint16_t s[3] = { bf16_of(0.5f), 0x7fc0 /* NaN */, bf16_of(1.0f) }; float o[3];
      CHECK(omega_fp8_pack_scales(s, 1, o, 0) == 0 && o[0] == ldexpf(0.5f, OMEGA_FP8_SCALE_EXP), "scale fold exact");
      CHECK(omega_fp8_pack_scales(s, 3, o, 0) != 0, "NaN scale refused");
      s[1] = 0x7f80; CHECK(omega_fp8_pack_scales(s, 3, o, 0) != 0, "inf scale refused");
      s[1] = bf16_of(3.0e38f); CHECK(omega_fp8_pack_scales(s, 3, o, 0) != 0, "fold overflow refused");
      CHECK(omega_fp8_pack_scales(s, 1, o, 1) == 0 && o[0] == 0.5f, "no-fold mutant pack"); }
    { float x[4] = { 1.0f, -2.0f, 0.5f, 0.0f }, o[4]; omega_fp8_pack_x_f32(x, 4, o);
      CHECK(o[0] == ldexpf(1.0f, 64) && o[1] == ldexpf(-2.0f, 64) && o[3] == 0.0f, "x fold exact"); }

    /* 3. codegen: every shape builds inside the budgets; refusals */
    static const uint32_t ks[] = { 128, 256, 768, 2048, 4096, 16384, 17408, 32768 };
    static const uint32_t lbs[] = { 8, 16 }, rows[] = { 1, 2, 4, 8 };
    uint32_t worst_insns = 0, worst_gprs = 0; int built = 0;
    for (size_t a = 0; a < sizeof ks / sizeof ks[0]; a++) for (size_t b = 0; b < 2; b++) for (size_t c = 0; c < 4; c++) for (int m = 0; m <= 3; m++) for (int deep = 0; deep < 2; deep++) {
        OmegaFp8GemvSpec sp = { .k = ks[a], .lane_bytes = lbs[b], .rows = rows[c], .warps = 8, .mutant = m };
        uint32_t ni, ng;
        if (ks[a] % (32 * lbs[b])) { CHECK(omega_fp8_gemv_prog_stats(&sp, &ni, &ng) != 0, "k=%u lb=%u must be refused", ks[a], lbs[b]); continue; }
        if (deep) { sp.depth = ks[a] / (32 * lbs[b]); if (sp.depth * sp.rows * lbs[b] > 128 || ks[a] > 4096) { CHECK(omega_fp8_gemv_prog_stats(&sp, &ni, &ng) != 0, "deep k=%u lb=%u rows=%u must be refused", ks[a], lbs[b], rows[c]); continue; } }
        int rc = omega_fp8_gemv_prog_stats(&sp, &ni, &ng);
        CHECK(rc == 0, "codegen k=%u lb=%u rows=%u mutant=%d deep=%d", ks[a], lbs[b], rows[c], m, deep);
        if (rc == 0) { built++; if (ni > worst_insns) worst_insns = ni; if (ng > worst_gprs) worst_gprs = ng; }
    }
    printf("codegen: %d kernels built, largest %u IR instructions, peak %u GPRs\n", built, worst_insns, worst_gprs);
    CHECK(worst_insns <= BW_MAX_IR_INSNS && worst_gprs <= 128, "budgets");
    { OmegaFp8GemvSpec bad = { .k = 192, .lane_bytes = 8, .rows = 1, .warps = 1 }; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "k not a multiple of 128 refused");
      bad.k = 256; bad.lane_bytes = 4; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "lane_bytes 4 refused (LDG.E has no immediate offset)"); bad.lane_bytes = 8;
      bad.k = 2048; bad.rows = 3; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "rows 3 refused");
      bad.rows = 8; bad.warps = 33; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "warps 33 refused");
      bad.warps = 8; bad.lane_bytes = 32; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "lane_bytes 32 refused");
      bad.lane_bytes = 16; bad.k = OMEGA_FP8_MAX_K + 512u; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "k past the limit refused"); }
    /* Register file (chip-found 2026-10-13): 32 * warps * declared registers must be <= 65536; the API refuses the launch.
     * The deep k=2048 kernel at 32 warps (72 declared x 1024 threads = 73728) faulted on the chip with Xid 13. */
    { OmegaFp8GemvSpec deep = { .k = 2048, .lane_bytes = 16, .rows = 1, .warps = 32, .depth = 4 }; uint32_t gp = 0;
      CHECK(omega_fp8_gemv_prog_stats(&deep, NULL, &gp) == 0, "deep4 k2048 encodes");
      uint32_t decl = ((gp + 1u + 7u) / 8u) * 8u;
      CHECK(32u * 32u * decl > 65536u, "deep4 k2048 at 32 warps needs %u declared registers x 1024 threads: over the 65536 file", decl);
      uint32_t de = 0; int rf = omega_gpu_fp8_kernel_fits(&deep, &de);
      CHECK(rf == OMEGA_GPU_FP8_TOO_LARGE && de == decl, "API refuses the register-file overflow (rc %d, declared %u): %s", rf, de, omega_gpu_fp8_last_error());
      deep.warps = 16; CHECK(omega_gpu_fp8_kernel_fits(&deep, NULL) == OMEGA_GPU_FP8_OK, "the same kernel at 16 warps fits");
      deep.depth = 0; deep.warps = 32; CHECK(omega_gpu_fp8_kernel_fits(&deep, NULL) == OMEGA_GPU_FP8_OK, "the classic kernel at 32 warps fits");
      deep.depth = 4; CHECK(omega_gpu_fp8_depth_for(&deep, 2048, 1) == 0, "automatic depth falls back to classic when deep would overflow the register file"); }

    /* 4. API refusals that need no device */
    { OmegaGpuFp8Tensor *t = NULL; uint8_t w[128 * 128]; uint16_t s[1] = { bf16_of(1.0f) }; memset(w, 0x38, sizeof w);
      CHECK(omega_gpu_fp8_tensor_upload(1, 128, 128, NULL, s, &t) == OMEGA_GPU_FP8_BAD_ARGS, "NULL w refused");
      CHECK(omega_gpu_fp8_tensor_upload(1, 100, 128, w, s, &t) == OMEGA_GPU_FP8_BAD_ARGS, "n not a multiple of 128 refused");
      CHECK(omega_gpu_fp8_tensor_upload(1, 128, OMEGA_FP8_MAX_K + 256u, w, s, &t) == OMEGA_GPU_FP8_TOO_LARGE, "k past the limit refused");
      w[77] = 0xff; CHECK(omega_gpu_fp8_tensor_upload(1, 128, 128, w, s, &t) == OMEGA_GPU_FP8_NAN_WEIGHT && t == NULL, "NaN weight refused before any device work");
      CHECK(strstr(omega_gpu_fp8_last_error(), "byte 77") != NULL, "error names the byte (%s)", omega_gpu_fp8_last_error());
      float y[8]; CHECK(omega_gpu_fp8_gemv_f32(NULL, 0, y, y, NULL) == OMEGA_GPU_FP8_BAD_ARGS, "NULL tensor refused"); }

    /* 5. sm121dis listing of the production shapes */
    { OmegaFp8GemvSpec a = { .k = 2048, .lane_bytes = 16, .rows = 8, .warps = 8 }, b = { .k = 768, .lane_bytes = 8, .rows = 2, .warps = 8 };
      CHECK(sm121dis_check(&a, "k2048_lb16_r8") == 0, "sm121dis listing k=2048");
      OmegaFp8GemvSpec d = { .k = 2048, .lane_bytes = 16, .rows = 1, .warps = 8, .depth = 4 };
      CHECK(sm121dis_check(&d, "k2048_lb16_r1_deep4") == 0, "sm121dis listing deep k=2048");
      CHECK(sm121dis_check(&b, "k768_lb8_r2") == 0, "sm121dis listing k=768"); }

    /* 6. variant 2 (W-L2c, omega_blackwell_codegen_moe2.c): every OWN-30B width builds inside the budgets at 16 warps,
     *    the production shapes fit the register file, the whole-row-group rule refuses what it must */
    {
        static const uint32_t k2[] = { 768, 2048, 4096 }, r2[] = { 1, 2, 4 };
        uint32_t worst_i = 0, worst_g = 0; int built2 = 0;
        for (size_t a = 0; a < 3; a++) for (size_t c = 0; c < 3; c++) for (int m = 0; m <= 3; m++) {
            OmegaFp8GemvSpec sp = { .k = k2[a], .lane_bytes = k2[a] % 512u == 0 ? 16u : 8u, .rows = r2[c], .warps = 16, .mutant = m };
            uint32_t ks = sp.k / (32u * sp.lane_bytes), ni = 0, ng = 0;
            int rc = omega_fp8_gemv2_prog_stats(&sp, &ni, &ng);
            if (sp.rows * ks * sp.lane_bytes > 128u) { CHECK(rc != 0, "v2 k=%u rows=%u must be refused (row group past 128 bytes per lane)", sp.k, sp.rows); continue; }
            CHECK(rc == 0 && ni <= BW_MAX_IR_INSNS, "v2 codegen k=%u lb=%u rows=%u mutant=%d: rc %d, %u insns", sp.k, sp.lane_bytes, sp.rows, m, rc, ni);
            if (rc == 0) { built2++; if (ni > worst_i) worst_i = ni; if (ng > worst_g) worst_g = ng; if (m == 0) printf("v2 codegen k=%u lb=%u rows=%u: %u insns, peak %u GPRs, declared %u\n", sp.k, sp.lane_bytes, sp.rows, ni, ng, omega_bw_gpr_declared(ng)); }
        }
        printf("v2 codegen: %d kernels built, largest %u IR instructions, peak %u GPRs\n", built2, worst_i, worst_g);
        /* the production shapes at the variant's default (16 warps, 1 row) must fit the 65536-register file */
        omega_gpu_fp8_set_variant(2);
        static const uint32_t kp[] = { 768, 2048, 4096 };
        for (size_t a = 0; a < 3; a++) {
            OmegaFp8GemvSpec sp = { .k = kp[a], .lane_bytes = kp[a] % 512u == 0 ? 16u : 8u, .rows = 1, .warps = OMEGA_GPU_FP8_V2_WARPS };
            uint32_t decl = 0; int rf = omega_gpu_fp8_kernel_fits(&sp, &decl);
            CHECK(rf == OMEGA_GPU_FP8_OK && 32u * sp.warps * decl <= 65536u, "v2 k=%u at %u warps fits the register file (rc %d, declared %u)", sp.k, sp.warps, rf, decl);
            printf("v2 k=%u rows=1 warps=%u: declared %u registers, %u per SM at one CTA\n", sp.k, sp.warps, decl, 32u * sp.warps * decl);
        }
        omega_gpu_fp8_set_variant(1);
        { OmegaFp8GemvSpec bad = { .k = 2048, .lane_bytes = 16, .rows = 1, .warps = 2 }; CHECK(omega_fp8_gemv2_prog_stats(&bad, NULL, NULL) != 0, "v2 k=2048 at 2 warps refused (staging rounds)");
          bad.warps = 16; bad.k = 16384; CHECK(omega_fp8_gemv2_prog_stats(&bad, NULL, NULL) != 0, "v2 k=16384 refused (row past 128 bytes per lane)");
          bad.k = 2048; bad.lane_bytes = 8; bad.rows = 2; CHECK(omega_fp8_gemv2_prog_stats(&bad, NULL, NULL) == 0, "v2 k=2048 lb8 rows=2 (128 bytes per lane) builds"); }
        OmegaFp8GemvSpec v = { .k = 2048, .lane_bytes = 16, .rows = 1, .warps = 16 };
        CHECK(sm121dis_check_v(2, &v, "v2_k2048_lb16_r1_w16") == 0, "sm121dis listing v2 k=2048");
    }

    /* 7. sched 1 (CUT5): pipelined decode + dependency stall pass. Every battery shape builds; the two grouped expert
     *    shapes fit the register file at 32 warps; an independent walk finds no user closer than OMEGA_FP8_FIXED_LAT
     *    cycles to a fixed-latency write; shortening one binding stall must be caught (the checker can say no); the
     *    sched 0 programs pass the same checker (they stall 6 everywhere) */
    {
        int built1 = 0; long viol_all = 0;
        for (size_t a = 0; a < sizeof ks / sizeof ks[0]; a++) for (size_t b = 0; b < 2; b++) for (size_t c = 0; c < 4; c++) for (int m = 0; m <= 3; m++) for (int deep = 0; deep < 2; deep++) {
            OmegaFp8GemvSpec sp = { .k = ks[a], .lane_bytes = lbs[b], .rows = rows[c], .warps = 8, .mutant = m, .sched = 1 };
            if (ks[a] % (32 * lbs[b])) continue;
            if (deep) { sp.depth = ks[a] / (32 * lbs[b]); if (sp.depth * sp.rows * lbs[b] > 128 || ks[a] > 4096) continue; }
            static BlackwellIRProgram cp;
            int rc = omega_blackwell_codegen_fp8_gemv_prog(&sp, &cp);
            CHECK(rc == 0 && cp.count <= BW_MAX_IR_INSNS, "sched1 codegen k=%u lb=%u rows=%u mutant=%d deep=%d", sp.k, sp.lane_bytes, sp.rows, m, deep);
            if (rc != 0) continue;
            built1++;
            long v = fp8_stall_violations(&cp, NULL);
            viol_all += v;
            CHECK(v == 0, "sched1 stall distances k=%u lb=%u rows=%u mutant=%d deep=%d: %ld users closer than %u cycles", sp.k, sp.lane_bytes, sp.rows, m, deep, v, OMEGA_FP8_FIXED_LAT);
        }
        printf("sched1 codegen: %d kernels built, %ld stall-distance violations\n", built1, viol_all);
        static const struct { uint32_t k, lb; const char *name; } prod[] = { { 2048, 16, "gate_up k2048 lb16" }, { 768, 8, "down k768 lb8" } };
        for (size_t q = 0; q < 2; q++) {
            static BlackwellIRProgram p0, p1;
            OmegaFp8GemvSpec s0 = { .k = prod[q].k, .lane_bytes = prod[q].lb, .rows = 1, .warps = 32 }, s1 = s0; s1.sched = 1;
            uint32_t d1 = 0, g1 = 0, g0 = 0;
            const int ok = omega_blackwell_codegen_fp8_gemv_prog(&s0, &p0) == 0 && omega_blackwell_codegen_fp8_gemv_prog(&s1, &p1) == 0;
            CHECK(ok, "sched 0 and 1 build at %s 32 warps", prod[q].name);
            if (!ok) continue;
            CHECK(omega_fp8_gemv_prog_stats(&s0, NULL, &g0) == 0 && omega_fp8_gemv_prog_stats(&s1, NULL, &g1) == 0, "stats %s", prod[q].name);
            CHECK(omega_gpu_fp8_kernel_fits(&s1, &d1) == OMEGA_GPU_FP8_OK, "sched1 %s at 32 warps fits the register file (declared %u): %s", prod[q].name, d1, omega_gpu_fp8_last_error());
            uint64_t c0 = 0, c1 = 0;
            const long v0 = fp8_stall_violations(&p0, &c0), v1 = fp8_stall_violations(&p1, &c1);
            CHECK(v0 == 0 && v1 == 0, "%s stall distances: sched0 %ld, sched1 %ld violations", prod[q].name, v0, v1);
            /* negative control: the pass gives each instruction its smallest legal stall, so a fixed-latency stall above 1
             * is binding; shortening the first one to 1 must be caught by the independent walk */
            size_t at = p1.count;
            for (size_t i = 0; i < p1.count; i++) if (fp8_chk_fixed(p1.insns[i].op) && ((p1.insns[i].control >> 9) & 0xfu) > 1u) { at = i; break; }
            if (at < p1.count) p1.insns[at].control = (p1.insns[at].control & ~(0xfu << 9)) | (1u << 9);
            CHECK(at < p1.count && fp8_stall_violations(&p1, NULL) > 0, "%s: stall checker catches one shortened stall", prod[q].name);
            printf("sched1 %s w32: %zu insns (sched0 %zu), peak %u GPRs (sched0 %u), declared %u; stall cycles in program order %" PRIu64 " (sched0 %" PRIu64 "); shortened-stall control caught\n",
                   prod[q].name, p1.count, p0.count, g1, g0, d1, c1, c0);
        }
        { OmegaFp8GemvSpec bad = { .k = 2048, .lane_bytes = 16, .rows = 1, .warps = 8, .sched = 2 }; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "sched 2 refused"); }
    }

    /* 8. CUT5 variants A (coal: coalesced output through shared slots) and B (wfirst: weight loads before the x' loads).
     *    Every battery shape builds in every combination with sched 0 and 1 (sched 1 with no stall violation); the refusals
     *    hold; for the two grouped expert shapes at 32 warps: the register file fits, A has one STG and one BAR.SYNC per
     *    round and declares the two slot sets, and B removes exactly one load round trip per K step (fp8_load_batches:
     *    a global load that follows a wait on the load scoreboard opens a new batch). Default programs keep their batch
     *    count, so the counter says "no" when nothing changed (deep kernels, which ignore wfirst). */
    {
        int built = 0; long viol = 0;
        for (size_t a = 0; a < sizeof ks / sizeof ks[0]; a++) for (size_t b = 0; b < 2; b++) for (size_t c = 0; c < 4; c++)
        for (int m = 0; m <= 5; m++) for (int deep = 0; deep < 2; deep++) for (int cw = 1; cw < 4; cw++) for (int sch = 0; sch < 2; sch++) {
            OmegaFp8GemvSpec sp = { .k = ks[a], .lane_bytes = lbs[b], .rows = rows[c], .warps = 4, .mutant = m, .sched = (uint32_t)sch, .coal = (uint32_t)(cw & 1), .wfirst = (uint32_t)(cw >> 1) };
            if (ks[a] % (32 * lbs[b]) || (m >= 4 && !sp.coal)) continue;
            if (deep) { sp.depth = ks[a] / (32 * lbs[b]); if (sp.depth * sp.rows * lbs[b] > 128 || ks[a] > 2048) continue; }
            static BlackwellIRProgram cp;
            int rc = omega_blackwell_codegen_fp8_gemv_prog(&sp, &cp);
            CHECK(rc == 0 && cp.count <= BW_MAX_IR_INSNS, "A/B codegen k=%u lb=%u rows=%u mutant=%d deep=%d coal=%u wfirst=%u sched=%d", sp.k, sp.lane_bytes, sp.rows, m, deep, sp.coal, sp.wfirst, sch);
            if (rc != 0) continue;
            built++;
            if (sch) { long v = fp8_stall_violations(&cp, NULL); viol += v; CHECK(v == 0, "A/B sched1 stall distances k=%u lb=%u rows=%u coal=%u wfirst=%u: %ld", sp.k, sp.lane_bytes, sp.rows, sp.coal, sp.wfirst, v); }
        }
        printf("A/B codegen: %d kernels built (coal, wfirst, both x sched 0/1 x mutants), %ld stall-distance violations\n", built, viol);
        { OmegaFp8GemvSpec bad = { .k = 2048, .lane_bytes = 16, .rows = 2, .warps = 32, .coal = 1 }; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "coal refused at warps x rows = 64");
          bad.rows = 1; bad.coal = 2; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "coal 2 refused");
          bad.coal = 0; bad.wfirst = 2; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "wfirst 2 refused");
          bad.wfirst = 0; bad.mutant = 4; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "mutant 4 refused without coal"); }
        static const struct { uint32_t k, lb; const char *name; } prod[] = { { 2048, 16, "gate_up k2048 lb16" }, { 768, 8, "down k768 lb8" } };
        for (size_t q = 0; q < 2; q++) {
            static BlackwellIRProgram p0, pa, pb, pab;
            OmegaFp8GemvSpec s0 = { .k = prod[q].k, .lane_bytes = prod[q].lb, .rows = 1, .warps = 32 }, sa = s0, sb = s0, sab = s0;
            sa.coal = 1; sb.wfirst = 1; sab.coal = 1; sab.wfirst = 1;
            const int ok = omega_blackwell_codegen_fp8_gemv_prog(&s0, &p0) == 0 && omega_blackwell_codegen_fp8_gemv_prog(&sa, &pa) == 0 &&
                           omega_blackwell_codegen_fp8_gemv_prog(&sb, &pb) == 0 && omega_blackwell_codegen_fp8_gemv_prog(&sab, &pab) == 0;
            CHECK(ok, "A, B, A+B build at %s 32 warps", prod[q].name);
            if (!ok) continue;
            uint32_t da = 0, db = 0, dab = 0;
            CHECK(omega_gpu_fp8_kernel_fits(&sa, &da) == OMEGA_GPU_FP8_OK && omega_gpu_fp8_kernel_fits(&sb, &db) == OMEGA_GPU_FP8_OK && omega_gpu_fp8_kernel_fits(&sab, &dab) == OMEGA_GPU_FP8_OK,
                  "%s A/B/A+B fit the register file at 32 warps (declared %u %u %u)", prod[q].name, da, db, dab);
            const int stg0 = fp8_count_op(&p0, BW_IR_STG_E), stga = fp8_count_op(&pa, BW_IR_STG_E), bar0 = fp8_count_op(&p0, BW_IR_BAR_SYNC), bara = fp8_count_op(&pa, BW_IR_BAR_SYNC);
            CHECK(stg0 == 1 && stga == 1 && bar0 == 0 && bara == 1 && omega_fp8_gemv_shared_bytes(&sa) == 2u * 32u * 4u && omega_fp8_gemv_shared_bytes(&s0) == 0,
                  "%s A: STG %d (default %d), BAR.SYNC %d (default %d), shared %u bytes", prod[q].name, stga, stg0, bara, bar0, omega_fp8_gemv_shared_bytes(&sa));
            const int b0 = fp8_load_batches(&p0), bb = fp8_load_batches(&pb), ba = fp8_load_batches(&pa), bab = fp8_load_batches(&pab);
            CHECK(b0 - bb == 1 && ba - bab == 1, "%s B: load batches default %d -> wfirst %d, coal %d -> coal+wfirst %d (one round trip less per K step)", prod[q].name, b0, bb, ba, bab);
            OmegaFp8GemvSpec sd = s0, sdb; sd.warps = 8; sd.depth = omega_fp8_gemv_steps(&sd); sdb = sd; sdb.wfirst = 1;
            static BlackwellIRProgram pd, pdb;
            const int okd = omega_blackwell_codegen_fp8_gemv_prog(&sd, &pd) == 0 && omega_blackwell_codegen_fp8_gemv_prog(&sdb, &pdb) == 0;
            CHECK(okd && fp8_load_batches(&pd) == fp8_load_batches(&pdb) && pd.count == pdb.count, "%s deep: wfirst changes nothing (batches %d, %d)", prod[q].name, okd ? fp8_load_batches(&pd) : -1, okd ? fp8_load_batches(&pdb) : -1);
            printf("A/B %s w32: insns %zu / A %zu / B %zu / A+B %zu; declared regs A %u B %u A+B %u; load batches %d / %d / %d / %d\n",
                   prod[q].name, p0.count, pa.count, pb.count, pab.count, da, db, dab, b0, ba, bb, bab);
        }
    }
}

/* ------------------------------------------------------------- simulator */
static uint64_t g_sim_issues;
static int g_sim_variant = 1; /* which generator sim_gemv runs: 1 (omega_blackwell_codegen_moe.c) or 2 (moe2.c, W-L2c) */
/* Runs the kernel for (W, groups, ids, x, x_rows) on the warp simulator in both fragment orders. Returns 0 and y
 * on success, -1 on a simulator error or an order-dependent result. */
static int sim_gemv(const OmegaFp8GemvSpec *spec, const Weights *W, uint32_t groups, const uint32_t *ids, const float *x,
                    uint32_t x_rows, float *y, uint32_t cta_budget, int no_fold, char *msg, size_t msglen) {
    static BlackwellIRProgram prog;
    if ((g_sim_variant == 2 ? omega_blackwell_codegen_fp8_gemv2_prog(spec, &prog) : omega_blackwell_codegen_fp8_gemv_prog(spec, &prog)) != 0) { snprintf(msg, msglen, "codegen refused"); return -1; }
    const uint32_t n = W->n, k = W->k, RG = n / spec->rows;
    size_t sc = (size_t)W->experts * (n / 128) * (k / 128);
    float *sf = aligned_alloc(16, ((sc * 4) + 15) & ~(size_t)15);
    if (omega_fp8_pack_scales(W->s, sc, sf, no_fold) != 0) { free(sf); snprintf(msg, msglen, "scale pack refused"); return -1; }
    float *xf = aligned_alloc(16, ((size_t)x_rows * k * 4 + 15) & ~(size_t)15);
    for (uint32_t r = 0; r < x_rows; r++) omega_fp8_pack_x_f32(x + (size_t)r * k, k, xf + (size_t)r * k);
    uint64_t *tab = aligned_alloc(16, 16 * (size_t)groups);
    for (uint32_t g = 0; g < groups; g++) {
        tab[2 * g] = (uintptr_t)(W->w + (size_t)ids[g] * n * k);
        tab[2 * g + 1] = (uintptr_t)(sf + (size_t)ids[g] * (n / 128) * (k / 128));
    }
    float *yd = aligned_alloc(16, ((size_t)groups * n * 4 + 15) & ~(size_t)15);
    float *copy = malloc((size_t)groups * n * 4);
    uint32_t grid_x = (RG + spec->warps - 1) / spec->warps; if (grid_x > cta_budget) grid_x = cta_budget; if (!grid_x) grid_x = 1;
    int err = 0;
    const int runs = (spec->coal || getenv("FP8_SIM_ALL_ORDERS")) ? 4 : 2; /* coal: also descending and shuffled warp visit orders per barrier phase (slot races) */
    for (int run = 0; run < runs && !err; run++) {
        const int order = run == 1;
        WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = 32 * spec->warps; s.order = order; s.warp_sched = run == 2 ? 1u : run == 3 ? 0x9e3779b9u : 0u;
        static uint8_t smem[OMEGA_FP8_MAX_K * 4 + 4096]; s.smem = smem; s.smem_bytes = g_sim_variant == 2 ? omega_fp8_gemv2_shared_bytes(spec) : omega_fp8_gemv_shared_bytes(spec); /* the bytes the launch declares (0: any shared access fails) */
        uint64_t ta = (uintptr_t)tab, xa = (uintptr_t)xf, ya = (uintptr_t)yd;
        s.cbank[OMEGA_FP8_ARG_TAB / 4] = (uint32_t)ta; s.cbank[OMEGA_FP8_ARG_TAB / 4 + 1] = (uint32_t)(ta >> 32);
        s.cbank[OMEGA_FP8_ARG_X / 4] = (uint32_t)xa; s.cbank[OMEGA_FP8_ARG_X / 4 + 1] = (uint32_t)(xa >> 32);
        s.cbank[OMEGA_FP8_ARG_Y / 4] = (uint32_t)ya; s.cbank[OMEGA_FP8_ARG_Y / 4 + 1] = (uint32_t)(ya >> 32);
        s.cbank[OMEGA_FP8_ARG_RG / 4] = RG; s.cbank[OMEGA_FP8_ARG_XSTRIDE / 4] = x_rows == 1 ? 0 : k * 4;
        s.cbank[OMEGA_FP8_ARG_YSTRIDE / 4] = n * 4; s.cbank[OMEGA_FP8_ARG_STRIDE / 4] = grid_x * spec->warps;
        /* every global access must land in a declared buffer */
        s.nranges = 4;
        s.ranges[0].lo = (uintptr_t)W->w; s.ranges[0].hi = (uintptr_t)W->w + (size_t)W->experts * n * k;
        s.ranges[1].lo = (uintptr_t)sf; s.ranges[1].hi = (uintptr_t)sf + sc * 4;
        s.ranges[2].lo = (uintptr_t)xf; s.ranges[2].hi = (uintptr_t)xf + (size_t)x_rows * k * 4;
        s.ranges[3].lo = (uintptr_t)yd; s.ranges[3].hi = (uintptr_t)yd + (size_t)groups * n * 4;
        /* the table is read too */
        s.nranges = 5; s.ranges[4].lo = (uintptr_t)tab; s.ranges[4].hi = (uintptr_t)tab + 16 * (size_t)groups;
        for (size_t i = 0; i < (size_t)groups * n; i++) yd[i] = -999.0f;
        err = ws_run(&s, grid_x, groups);
        if (err) { snprintf(msg, msglen, "run %d (order %d, warp order %u): %s", run, order, s.warp_sched, s.msg); break; }
        g_sim_issues += s.warp_issues;
        size_t unwritten = 0; for (size_t i = 0; i < (size_t)groups * n; i++) if (yd[i] == -999.0f) unwritten++;
        if (unwritten) { err = -1; snprintf(msg, msglen, "%zu output words never written", unwritten); break; }
        if (run == 0) memcpy(copy, yd, (size_t)groups * n * 4);
        else if (memcmp(copy, yd, (size_t)groups * n * 4) != 0) { if (getenv("FP8_SIM_DIFF")) { int c = 0; for (size_t i = 0; i < (size_t)groups * n && c < 12; i++) if (memcmp(&copy[i], &yd[i], 4)) { fprintf(stderr, "  diff y[%zu] %g vs %g\n", i, copy[i], yd[i]); c++; } } err = -1; snprintf(msg, msglen, "run %d (order %d, warp order %u) changed the result", run, order, s.warp_sched); }
    }
    if (!err) memcpy(y, yd, (size_t)groups * n * 4);
    free(sf); free(xf); free(tab); free(yd); free(copy);
    return err ? -1 : 0;
}

typedef struct { uint32_t experts, n, k, groups, x_rows, lb, rows, warps; const char *name; } SimPoint;

static void sim_battery(uint32_t cta_budget) {
    char msg[240];
    /* 1. one-hot conversion through the kernel: 254 rows, row r holds code r at column 0 (NaN codes replaced by
     *    0x00), x = e_0, scale 1: y[r] must equal the reference exactly (zero rows compare as 0) */
    {
        Weights W; W.experts = 1; W.n = 256; W.k = 256;
        W.w = aligned_alloc(64, 256 * 256); memset(W.w, 0, 256 * 256);
        W.s = malloc(4 * 2); for (int b = 0; b < 4; b++) W.s[b] = bf16_of(1.0f);
        for (int r = 0; r < 256; r++) W.w[r * 256] = omega_fp8_e4m3_is_nan((uint8_t)r) ? 0 : (uint8_t)r;
        float x[256] = { 0 }; x[0] = 1.0f; float y[256]; uint32_t id = 0;
        OmegaFp8GemvSpec sp = { .k = 256, .lane_bytes = 8, .rows = 8, .warps = 4 };
        int rc = sim_gemv(&sp, &W, 1, &id, x, 1, y, cta_budget, 0, msg, sizeof msg);
        CHECK(rc == 0, "one-hot sim run: %s", msg);
        int exact = 0;
        if (rc == 0) for (int r = 0; r < 256; r++) {
            float ref = omega_fp8_e4m3_is_nan((uint8_t)r) ? 0.0f : omega_fp8_e4m3_ref((uint8_t)r);
            if (y[r] == ref) exact++; else { CHECK(0, "one-hot code 0x%02x: kernel %a, reference %a", r, y[r], ref); }
        }
        printf("sim one-hot conversion: %d/256 rows exact\n", exact);
        /* x on the other lanes/words: code at column c, x = e_c for c in {1, 17, 127} on one row each */
        for (int t = 0; t < 3; t++) {
            const int cols[3] = { 1, 17, 255 }; const uint8_t codes[3] = { 0xc0, 0x05, 0x7e };
            memset(W.w, 0, 256 * 256); W.w[3 * 256 + cols[t]] = codes[t];
            memset(x, 0, sizeof x); x[cols[t]] = -2.5f; for (int b = 0; b < 4; b++) W.s[b] = bf16_of(0.75f);
            rc = sim_gemv(&sp, &W, 1, &id, x, 1, y, cta_budget, 0, msg, sizeof msg);
            CHECK(rc == 0 && y[3] == omega_fp8_e4m3_ref(codes[t]) * -2.5f * 0.75f, "one-hot column %d code 0x%02x: %a vs %a (%s)", cols[t], codes[t], rc == 0 ? y[3] : 0.0f, omega_fp8_e4m3_ref(codes[t]) * -2.5f * 0.75f, msg);
        }
        weights_free(&W);
    }
    /* 2. parity battery */
    static const SimPoint pts[] = {
        { 1, 128, 256, 1, 1, 8, 1, 1, "k256 lb8 r1 w1" },
        { 1, 256, 256, 1, 1, 8, 2, 2, "k256 lb8 r2 w2" },
        { 1, 256, 768, 1, 1, 8, 4, 8, "k768 lb8 r4 w8 (down shape width)" },
        { 1, 256, 768, 1, 1, 8, 8, 8, "k768 lb8 r8" },
        { 1, 384, 2048, 1, 1, 16, 8, 8, "k2048 lb16 r8 w8 (gate shape width)" },
        { 1, 128, 2048, 1, 1, 16, 1, 8, "k2048 lb16 r1" },
        { 1, 256, 512, 1, 1, 16, 2, 3, "k512 lb16 r2 w3 (odd warps, tail row groups)" },
        { 4, 256, 2048, 4, 1, 16, 8, 8, "4 experts shared x (gate+up)" },
        { 6, 256, 768, 8, 8, 8, 4, 8, "8 groups per-group x, repeated ids (down)" },
        { 3, 128, 256, 2, 2, 8, 8, 4, "2 groups per-group x" },
        { 2, 256, 768, 2, 2, 8, 1, 7, "k768 w7: partial staging round (deep)" },
        { 1, 128, 17408, 1, 1, 16, 2, 8, "k17408 lb16 r2 w8 (Qwen3.8-27B down width, step kernel)" },
    };
    const size_t npts = sizeof pts / sizeof pts[0];
    double worst_all = 0.0;
    for (size_t i = 0; i < npts; i++) {
        const SimPoint *p = &pts[i];
        Weights W; weights_make(&W, p->experts, p->n, p->k, 1000 + (uint32_t)i);
        uint32_t ids[8]; uint32_t st = 77 + (uint32_t)i;
        for (uint32_t g = 0; g < p->groups; g++) ids[g] = lcg(&st) % p->experts;
        float *x = malloc((size_t)p->x_rows * p->k * 4); for (size_t j = 0; j < (size_t)p->x_rows * p->k; j++) x[j] = frand(&st, -1.0f, 1.0f);
        float *y = malloc((size_t)p->groups * p->n * 4);
        /* every point runs the step-by-step kernel (depth 0) and, when the row fits, the deep kernel (depth = KS) */
        for (int deep = 0; deep < 2; deep++) {
        OmegaFp8GemvSpec sp = { .k = p->k, .lane_bytes = p->lb, .rows = p->rows, .warps = p->warps };
        if (deep) { sp.depth = omega_fp8_gemv_steps(&sp); if (sp.depth * sp.rows * sp.lane_bytes > 128 || p->k > 512 * p->warps) continue; }
        char nm[96]; snprintf(nm, sizeof nm, "%s%s", p->name, deep ? " deep" : "");
        double t0 = now_ms();
        int rc = sim_gemv(&sp, &W, p->groups, ids, x, p->x_rows, y, cta_budget, 0, msg, sizeof msg);
        size_t bad = 0; double worst = rc == 0 ? parity(&W, p->groups, ids, x, p->x_rows, y, &bad) : 1.0;
        if (worst > worst_all) worst_all = worst;
        CHECK(rc == 0 && bad == 0, "sim %s: rc=%d bad=%zu worst_rel=%.3e %s", nm, rc, bad, worst, msg);
        printf("sim %-45s rc=%d worst_rel=%.3e bad=%zu (%.0f ms)\n", nm, rc, worst, bad, now_ms() - t0);
        /* mutants on the two grouped points and the first two single points: each must be caught */
        if (i == 1 || i == 4 || i == 7 || i == 8) {
            for (int m = 1; m <= 4; m++) {
                if (m == 3 && p->groups == 1) continue; /* group swap needs two groups */
                if (m == 2 && omega_fp8_gemv_steps(&sp) < 2) { printf("sim %s mutant 2: one K step, control not applicable\n", p->name); continue; }
                OmegaFp8GemvSpec ms = sp; ms.mutant = m == 4 ? 0 : m;
                rc = sim_gemv(&ms, &W, p->groups, ids, x, p->x_rows, y, cta_budget, m == 4, msg, sizeof msg);
                bad = 0; worst = rc == 0 ? parity(&W, p->groups, ids, x, p->x_rows, y, &bad) : 1.0;
                int caught = rc != 0 || bad > 0;
                if (m == 3 && rc == 0) { /* the ids must differ for the swap to show; make sure the control is meaningful */
                    int distinct = 0; for (uint32_t g = 0; g + 1 < p->groups; g += 2) if (ids[g] != ids[g ^ 1]) distinct = 1;
                    if (!distinct) { printf("sim %s mutant 3: ids pairwise equal, control skipped\n", p->name); continue; }
                }
                CHECK(caught, "sim %s mutant %d NOT caught (worst_rel=%.3e)", nm, m, worst);
                printf("sim %-45s mutant %d: %s (rc=%d bad=%zu worst_rel=%.3e)\n", nm, m, caught ? "caught" : "MISSED", rc, bad, worst);
            }
        }
        }
        free(x); free(y); weights_free(&W);
    }
    printf("sim battery: worst relative error %.3e over %zu points, %" PRIu64 " warp issues\n", worst_all, npts, g_sim_issues);

    /* 3. variant 2 (W-L2c): y bit-identical to variant 1 on every point (same arithmetic order), parity against the oracle,
     *    mutants caught; the identity check is shown to say "no" on a mutant */
    {
        static const SimPoint v2pts[] = {
            { 1, 256, 2048, 1, 1, 16, 1, 8, "v2 k2048 lb16 r1 w8 (gate_up/qkv width)" },
            { 1, 256, 2048, 1, 1, 16, 2, 8, "v2 k2048 lb16 r2 w8 (128 B per lane)" },
            { 1, 256, 4096, 1, 1, 16, 1, 8, "v2 k4096 lb16 r1 w8 (o_proj width)" },
            { 1, 256, 768, 1, 1, 8, 1, 8, "v2 k768 lb8 r1 w8 (down width)" },
            { 1, 256, 768, 1, 1, 8, 4, 8, "v2 k768 lb8 r4 w8" },
            { 2, 256, 768, 2, 2, 8, 1, 7, "v2 k768 w7: partial staging round" },
            { 4, 256, 2048, 4, 1, 16, 1, 8, "v2 4 experts shared x (gate_up)" },
            { 6, 256, 768, 8, 8, 8, 2, 8, "v2 8 groups per-group x, repeated ids (down)" },
            { 3, 128, 256, 2, 2, 8, 8, 4, "v2 2 groups per-group x, r8 k256" },
            { 1, 256, 512, 1, 1, 16, 2, 3, "v2 k512 w3 (odd warps, tail row groups)" },
        };
        const size_t n2 = sizeof v2pts / sizeof v2pts[0];
        int identical = 0;
        for (size_t i = 0; i < n2; i++) {
            const SimPoint *p = &v2pts[i];
            Weights W; weights_make(&W, p->experts, p->n, p->k, 5000 + (uint32_t)i);
            uint32_t ids[8]; uint32_t st = 177 + (uint32_t)i;
            for (uint32_t g = 0; g < p->groups; g++) ids[g] = lcg(&st) % p->experts;
            float *x = malloc((size_t)p->x_rows * p->k * 4); for (size_t j = 0; j < (size_t)p->x_rows * p->k; j++) x[j] = frand(&st, -1.0f, 1.0f);
            const size_t ny = (size_t)p->groups * p->n;
            float *y1 = malloc(ny * 4), *y2 = malloc(ny * 4);
            OmegaFp8GemvSpec sp = { .k = p->k, .lane_bytes = p->lb, .rows = p->rows, .warps = p->warps };
            g_sim_variant = 1; int rc1 = sim_gemv(&sp, &W, p->groups, ids, x, p->x_rows, y1, cta_budget, 0, msg, sizeof msg);
            CHECK(rc1 == 0, "v1 reference for %s: %s", p->name, msg);
            double t0 = now_ms();
            g_sim_variant = 2; int rc2 = sim_gemv(&sp, &W, p->groups, ids, x, p->x_rows, y2, cta_budget, 0, msg, sizeof msg);
            size_t bad = 0; double worst = rc2 == 0 ? parity(&W, p->groups, ids, x, p->x_rows, y2, &bad) : 1.0;
            int same = rc1 == 0 && rc2 == 0 && memcmp(y1, y2, ny * 4) == 0;
            CHECK(rc2 == 0 && bad == 0 && same, "sim %s: rc=%d bad=%zu worst_rel=%.3e identical_to_v1=%d %s", p->name, rc2, bad, worst, same, msg);
            if (same) identical++;
            printf("sim %-48s rc=%d worst_rel=%.3e bad=%zu identical_to_v1=%d (%.0f ms)\n", p->name, rc2, worst, bad, same, now_ms() - t0);
            if (i == 0 || i == 6 || i == 7) {
                for (int m = 1; m <= 3; m++) {
                    if (m == 3 && p->groups == 1) continue;
                    OmegaFp8GemvSpec ms = sp; ms.mutant = m;
                    int rcm = sim_gemv(&ms, &W, p->groups, ids, x, p->x_rows, y2, cta_budget, 0, msg, sizeof msg);
                    bad = 0; worst = rcm == 0 ? parity(&W, p->groups, ids, x, p->x_rows, y2, &bad) : 1.0;
                    int caught = rcm != 0 || bad > 0, differs = rcm != 0 || memcmp(y1, y2, ny * 4) != 0;
                    if (m == 3 && rcm == 0) { int distinct = 0; for (uint32_t g = 0; g + 1 < p->groups; g += 2) if (ids[g] != ids[g ^ 1]) distinct = 1; if (!distinct) { printf("sim %s mutant 3: ids pairwise equal, control skipped\n", p->name); continue; } }
                    CHECK(caught && differs, "sim %s mutant %d NOT caught (bad=%zu worst_rel=%.3e identity says %s)", p->name, m, bad, worst, differs ? "different" : "SAME");
                    printf("sim %-48s mutant %d: %s (rc=%d bad=%zu worst_rel=%.3e, identity check differs=%d)\n", p->name, m, caught ? "caught" : "MISSED", rcm, bad, worst, differs);
                }
            }
            g_sim_variant = 1;
            free(x); free(y1); free(y2); weights_free(&W);
        }
        printf("sim v2 battery: %d/%zu points bit-identical to variant 1\n", identical, n2);
    }

    /* 4. sched 1 (CUT5): y bit-identical to sched 0 on every battery point (step and deep kernels), parity against the
     *    oracle, mutants 1..3 caught on the grouped points; the identity check is shown to say "no" on a mutant */
    {
        int identical = 0, runs = 0;
        for (size_t i = 0; i < npts; i++) {
            const SimPoint *p = &pts[i];
            Weights W; weights_make(&W, p->experts, p->n, p->k, 1000 + (uint32_t)i);
            uint32_t ids[8]; uint32_t st = 77 + (uint32_t)i;
            for (uint32_t g = 0; g < p->groups; g++) ids[g] = lcg(&st) % p->experts;
            float *x = malloc((size_t)p->x_rows * p->k * 4); for (size_t j = 0; j < (size_t)p->x_rows * p->k; j++) x[j] = frand(&st, -1.0f, 1.0f);
            const size_t ny = (size_t)p->groups * p->n;
            float *y0 = malloc(ny * 4), *y1 = malloc(ny * 4);
            for (int deep = 0; deep < 2; deep++) {
                OmegaFp8GemvSpec sp = { .k = p->k, .lane_bytes = p->lb, .rows = p->rows, .warps = p->warps };
                if (deep) { sp.depth = omega_fp8_gemv_steps(&sp); if (sp.depth * sp.rows * sp.lane_bytes > 128 || p->k > 512 * p->warps) continue; }
                char nm[96]; snprintf(nm, sizeof nm, "sched1 %s%s", p->name, deep ? " deep" : "");
                int rc0 = sim_gemv(&sp, &W, p->groups, ids, x, p->x_rows, y0, cta_budget, 0, msg, sizeof msg);
                CHECK(rc0 == 0, "sched0 reference for %s: %s", nm, msg);
                OmegaFp8GemvSpec s1 = sp; s1.sched = 1;
                int rc1 = sim_gemv(&s1, &W, p->groups, ids, x, p->x_rows, y1, cta_budget, 0, msg, sizeof msg);
                size_t bad = 0; double worst = rc1 == 0 ? parity(&W, p->groups, ids, x, p->x_rows, y1, &bad) : 1.0;
                int same = rc0 == 0 && rc1 == 0 && memcmp(y0, y1, ny * 4) == 0;
                CHECK(rc1 == 0 && bad == 0 && same, "sim %s: rc=%d bad=%zu worst_rel=%.3e identical_to_sched0=%d %s", nm, rc1, bad, worst, same, msg);
                runs++; if (same) identical++;
                printf("sim %-52s rc=%d worst_rel=%.3e bad=%zu identical_to_sched0=%d\n", nm, rc1, worst, bad, same);
                if (i == 7 || i == 8) {
                    for (int m = 1; m <= 3; m++) {
                        if (m == 2 && omega_fp8_gemv_steps(&sp) < 2) continue;
                        OmegaFp8GemvSpec ms = s1; ms.mutant = m;
                        int rcm = sim_gemv(&ms, &W, p->groups, ids, x, p->x_rows, y1, cta_budget, 0, msg, sizeof msg);
                        bad = 0; worst = rcm == 0 ? parity(&W, p->groups, ids, x, p->x_rows, y1, &bad) : 1.0;
                        int caught = rcm != 0 || bad > 0, differs = rcm != 0 || memcmp(y0, y1, ny * 4) != 0;
                        CHECK(caught && differs, "sim %s mutant %d NOT caught (bad=%zu worst_rel=%.3e identity says %s)", nm, m, bad, worst, differs ? "different" : "SAME");
                        printf("sim %-52s mutant %d: %s (rc=%d bad=%zu worst_rel=%.3e, identity check differs=%d)\n", nm, m, caught ? "caught" : "MISSED", rcm, bad, worst, differs);
                    }
                }
            }
            free(x); free(y0); free(y1); weights_free(&W);
        }
        printf("sim sched1 battery: %d/%d runs bit-identical to sched 0\n", identical, runs);
    }

    /* 5. CUT5 variants A (coal) and B (wfirst), alone and together: y bit-identical to the default on every battery point
     *    where A applies (warps * rows <= 32), step and deep kernels, parity against the oracle; coal kernels run in four
     *    orders (fragment order both ways, warps descending and shuffled per barrier phase). Mutants: 4 (one slot set) must
     *    be caught by a warp order (points with two or more rounds per CTA), 5 (no clamp) on the points whose last round is
     *    partial (6, 10), 1..3 on the A+B kernel of point 8 (1, 2 also on point 9). */
    {
        int identical = 0, runs = 0;
        for (size_t i = 0; i < npts; i++) {
            const SimPoint *p = &pts[i];
            Weights W; weights_make(&W, p->experts, p->n, p->k, 1000 + (uint32_t)i);
            uint32_t ids[8]; uint32_t st = 77 + (uint32_t)i;
            for (uint32_t g = 0; g < p->groups; g++) ids[g] = lcg(&st) % p->experts;
            float *x = malloc((size_t)p->x_rows * p->k * 4); for (size_t j = 0; j < (size_t)p->x_rows * p->k; j++) x[j] = frand(&st, -1.0f, 1.0f);
            const size_t ny = (size_t)p->groups * p->n;
            float *y0 = malloc(ny * 4), *y1 = malloc(ny * 4);
            for (int deep = 0; deep < 2; deep++) {
                OmegaFp8GemvSpec sp = { .k = p->k, .lane_bytes = p->lb, .rows = p->rows, .warps = p->warps };
                if (deep) { sp.depth = omega_fp8_gemv_steps(&sp); if (sp.depth * sp.rows * sp.lane_bytes > 128 || p->k > 512 * p->warps) continue; }
                int rc0 = sim_gemv(&sp, &W, p->groups, ids, x, p->x_rows, y0, cta_budget, 0, msg, sizeof msg);
                CHECK(rc0 == 0, "default reference for A/B %s%s: %s", p->name, deep ? " deep" : "", msg);
                for (int cw = 1; cw < 4; cw++) {
                    OmegaFp8GemvSpec sv = sp; sv.coal = (uint32_t)(cw & 1); sv.wfirst = (uint32_t)(cw >> 1);
                    if (sv.coal && p->warps * p->rows > 32) continue;
                    char nm[96]; snprintf(nm, sizeof nm, "%s %s%s", cw == 1 ? "A" : cw == 2 ? "B" : "A+B", p->name, deep ? " deep" : "");
                    int rc1 = sim_gemv(&sv, &W, p->groups, ids, x, p->x_rows, y1, cta_budget, 0, msg, sizeof msg);
                    size_t bad = 0; double worst = rc1 == 0 ? parity(&W, p->groups, ids, x, p->x_rows, y1, &bad) : 1.0;
                    int same = rc0 == 0 && rc1 == 0 && memcmp(y0, y1, ny * 4) == 0;
                    CHECK(rc1 == 0 && bad == 0 && same, "sim %s: rc=%d bad=%zu worst_rel=%.3e identical_to_default=%d %s", nm, rc1, bad, worst, same, msg);
                    runs++; if (same) identical++;
                    printf("sim %-52s rc=%d worst_rel=%.3e bad=%zu identical_to_default=%d\n", nm, rc1, worst, bad, same);
                    int ms_list[5], nm_list = 0;
                    if (sv.coal && cw == 1 && (i == 6 || i == 10)) ms_list[nm_list++] = 5;
                    if (sv.coal && cw == 1 && (i == 1 || i == 6 || i == 8 || i == 10)) ms_list[nm_list++] = 4;
                    if (cw == 3 && (i == 8 || i == 9)) { ms_list[nm_list++] = 1; if (omega_fp8_gemv_steps(&sp) >= 2) ms_list[nm_list++] = 2; if (i == 8) ms_list[nm_list++] = 3; } /* point 9 draws the same expert for both groups: a table swap is invisible there */
                    for (int q = 0; q < nm_list; q++) {
                        OmegaFp8GemvSpec ms = sv; ms.mutant = ms_list[q];
                        int rcm = sim_gemv(&ms, &W, p->groups, ids, x, p->x_rows, y1, cta_budget, 0, msg, sizeof msg);
                        bad = 0; worst = rcm == 0 ? parity(&W, p->groups, ids, x, p->x_rows, y1, &bad) : 1.0;
                        int caught = rcm != 0 || bad > 0;
                        CHECK(caught, "sim %s mutant %d NOT caught (bad=%zu worst_rel=%.3e)", nm, ms.mutant, bad, worst);
                        printf("sim %-52s mutant %d: %s (rc=%d bad=%zu %s)\n", nm, ms.mutant, caught ? "caught" : "MISSED", rcm, bad, rcm ? msg : "");
                    }
                }
            }
            free(x); free(y0); free(y1); weights_free(&W);
        }
        printf("sim A/B battery: %d/%d runs bit-identical to the default\n", identical, runs);
    }
}

/* ------------------------------------------------------------------- chip */
typedef struct { uint32_t experts, n, k, groups, x_rows; const char *name; int parity_check; } ChipPoint;

static int g_evict = 1; /* stream 512 MB through the host caches between upload and timing (nvrm.c:433: freshly written
                          host memory is served from the CPU caches by snooping and reads faster) */
static void evict_host_caches(void) {
    static uint8_t *buf; const size_t sz = 512u << 20; static uint8_t pass;
    if (!buf) buf = malloc(sz);
    if (!buf) return;
    memset(buf, ++pass, sz);
    volatile uint64_t sum = 0; for (size_t i = 0; i < sz; i += 64) sum += buf[i];
}

static const char *g_only; static int g_depth_forced; /* run only the battery points whose name contains this (diagnosis) */
/* --n N --k K (ternary engine T0, 2026-10-15): run ONE custom point n x k (one expert, one group) instead of the battery,
 * with the parity check, timing and mutants 1, 2 and 4 (3 needs groups > 1); e.g. the BitNet word table 128256 x 2560. */
static uint32_t g_custom_n, g_custom_k; static char g_custom_name[48];
/* --grouped-cold (CUT5, 2026-10-16): the decode graph's two expert launches at their real shapes (omega_gpu_decode_graph.c
 * record_layer: gate_up [128][1536][2048] with one shared x row, down [128][2048][768] with one x row per group), 8 of 128
 * experts per launch. L2 is defeated by rotation: the 128 experts are split into 16 disjoint sets of 8 and launch r uses
 * set r % 16, so consecutive launches share no weight byte and a set returns only after 15 other sets (gate_up 403 MB,
 * down 201 MB in between; L2 is 24 MiB, evidence/TERN-T0-KT1-18185404-20261016T0151Z/README.md). Every timed launch is
 * checked against the double oracle (outside the chip time); a launch that fails parity fails the run. Mutants 1..4
 * must be caught at both shapes. GB/s = (weight + scale bytes of the 8 experts) / median chip time (launch to completion
 * marker: the launch floor is NOT subtracted). */
static int g_cold;
static int g_dump_stamps; /* --dump-stamps: one line per GPU-stamped launch (diagnosis of per-launch spread) */
/* GPU-clock timing for --grouped-cold (CUT5, after omega#507: a host clock around submit and completion is not kernel time).
 * The rotation above, recorded as one command list of LIST_N launches (launch i uses set i % 16, writes its own y slot),
 * WAIT_FOR_IDLE + shader-cache invalidate before every launch (the decode graph's default barrier), GPU timestamps
 * (PTIMER) before launch 0 and after every launch; launch time = stamp[i + 1] - stamp[i] (the barrier and launch floor are
 * inside it, not subtracted). Every launch's y is checked against the oracle after the list. lists x LIST_N launches. */
#define LIST_N 32u
static int cold_list_stamps(const ChipPoint *p, const Weights *W, const OmegaGpuFp8Tensor *t, const uint32_t *perm, const float *x,
                            uint32_t lists, uint64_t *el, uint32_t *n_el, size_t *bad_out, double *worst_out) {
    const size_t ny = (size_t)p->groups * p->n;
    const uint32_t STR = p->experts / 16u; /* experts per rotation set (8 for the 128-expert tables, 1 for --cold-single) */
    OmegaGpuCmdList *cl = NULL; OmegaGpuMem tab, xd, yd; int ok = 1;
    memset(&tab, 0, sizeof tab); memset(&xd, 0, sizeof xd); memset(&yd, 0, sizeof yd);
    if (omega_gpu_cmdlist_create(LIST_N, &cl) != 0) { printf("cmdlist create: %s\n", omega_gpu_session_last_error()); return -1; }
    omega_gpu_session_lock();
    int arc = omega_gpu_session_alloc((size_t)16u * p->groups * 16u, &tab) | omega_gpu_session_alloc((size_t)p->x_rows * p->k * 4u, &xd) | omega_gpu_session_alloc((size_t)LIST_N * ny * 4u, &yd);
    omega_gpu_session_unlock();
    if (arc != 0) { printf("list alloc: %s\n", omega_gpu_session_last_error()); ok = 0; }
    if (ok) {
        uint64_t *tw = tab.cpu;
        for (uint32_t s = 0; s < 16u && ok; s++) for (uint32_t g = 0; g < p->groups; g++)
            if (omega_gpu_fp8_tensor_entry(t, perm[STR * s + g], &tw[2u * (s * p->groups + g)], &tw[2u * (s * p->groups + g) + 1u]) != 0) ok = 0;
        for (uint32_t r = 0; r < p->x_rows; r++) omega_fp8_pack_x_f32(x + (size_t)r * p->k, p->k, (float *)xd.cpu + (size_t)r * p->k);
        if (!ok || omega_gpu_session_sync_to_gpu(&tab) != 0 || omega_gpu_session_sync_to_gpu(&xd) != 0) ok = 0;
    }
    size_t bad_all = 0; double worst_all = 0.0; uint32_t n = 0;
    for (uint32_t l = 0; l < lists && ok; l++) {
        float *y = yd.cpu; for (size_t i = 0; i < (size_t)LIST_N * ny; i++) y[i] = -999.0f;
        if (omega_gpu_session_sync_to_gpu(&yd) != 0) { ok = 0; break; }
        if (g_evict) evict_host_caches();
        omega_gpu_session_lock();
        omega_gpu_cmdlist_reset(cl);
        for (uint32_t i = 0; i < LIST_N && ok; i++)
            if (omega_gpu_fp8_gemv_grouped_record(cl, t, p->groups, tab.va + (uint64_t)(i % 16u) * p->groups * 16u, xd.va, p->x_rows, yd.va + (uint64_t)i * ny * 4u) != 0) { printf("record: %s\n", omega_gpu_fp8_last_error()); ok = 0; }
        OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.mode = OMEGA_GPU_CMDLIST_PCAS; o.barrier = OMEGA_GPU_CMDLIST_BARRIER_WFI; o.stamps = 1;
        OmegaGpuCmdListStats cs; memset(&cs, 0, sizeof cs);
        uint64_t sts[LIST_N + 1]; int cnt = -1;
        if (ok && omega_gpu_cmdlist_submit(cl, &o, &cs) != 0) { printf("submit: %s\n", omega_gpu_session_last_error()); ok = 0; }
        if (ok) cnt = omega_gpu_cmdlist_stamps(cl, sts, LIST_N + 1);
        omega_gpu_session_unlock();
        if (!ok || cnt != (int)LIST_N + 1 || cs.sems_done != LIST_N) { printf("list %u: stamps %d sems_done %u\n", l, cnt, cs.sems_done); ok = 0; break; }
        if (omega_gpu_session_sync_to_cpu(&yd) != 0) { ok = 0; break; }
        for (uint32_t i = 0; i < LIST_N; i++) {
            size_t b = 0; double w = parity(W, p->groups, perm + STR * (i % 16u), x, p->x_rows, y + (size_t)i * ny, &b);
            bad_all += b; if (w > worst_all) worst_all = w;
            el[n++] = sts[i + 1] - sts[i];
            if (g_dump_stamps) printf("stamp %-24s list=%u launch=%u set=%u elapsed_us=%.1f bad=%zu\n", p->name, l, i, i % 16u, (double)el[n - 1] / 1e3, b);
        }
    }
    omega_gpu_session_lock(); omega_gpu_session_free(&tab); omega_gpu_session_free(&xd); omega_gpu_session_free(&yd); omega_gpu_session_unlock();
    omega_gpu_cmdlist_free(cl);
    *n_el = n; *bad_out = bad_all; *worst_out = worst_all;
    return ok ? 0 : -1;
}
static void cold_points(FILE *out, uint32_t reps, int *first) {
    static const ChipPoint gpts[] = {
        { 128, 1536, 2048, 8, 1, "cold gate_up 8 of 128", 1 },
        { 128, 2048, 768, 8, 8, "cold down 8 of 128", 1 },
    };
    /* --cold-single (SPEED-1B, 2026-10-17): the decode graph's two single-projection launches (omega_gpu_decode_graph.c
     * record_layer: qkv [1][5120][2048], o_proj [1][2048][4096], groups 1). Same rotation as above with 16 copies of the
     * shape, one per set: consecutive launches share no weight byte and a copy returns after 15 others (qkv 157 MB,
     * o_proj 126 MB in between; L2 is 24 MiB). */
    static const ChipPoint spts[] = {
        { 16, 5120, 2048, 1, 1, "cold qkv 1 of 16", 1 },
        { 16, 2048, 4096, 1, 1, "cold o_proj 1 of 16", 1 },
    };
    const ChipPoint *cpts = g_cold == 2 ? spts : gpts;
    for (size_t i = 0; i < 2; i++) {
        const ChipPoint *p = &cpts[i];
        if (g_only && !strstr(p->name, g_only)) continue;
        const uint32_t STR = p->experts / 16u; /* experts per rotation set */
        Weights W; weights_make(&W, p->experts, p->n, p->k, 9100 + (uint32_t)i);
        uint32_t perm[128], st = 4242 + (uint32_t)i;
        for (uint32_t e = 0; e < p->experts; e++) perm[e] = e;
        for (uint32_t e = p->experts - 1u; e > 0; e--) { uint32_t j = lcg(&st) % (e + 1); uint32_t t = perm[e]; perm[e] = perm[j]; perm[j] = t; }
        float *x = malloc((size_t)p->x_rows * p->k * 4); for (size_t j = 0; j < (size_t)p->x_rows * p->k; j++) x[j] = frand(&st, -1.0f, 1.0f);
        float *y = malloc((size_t)p->groups * p->n * 4);
        OmegaGpuFp8Tensor *t = NULL;
        int rc = omega_gpu_fp8_tensor_upload(p->experts, p->n, p->k, W.w, W.s, &t);
        OmegaGpuFp8Info info; memset(&info, 0, sizeof info);
        if (rc == OMEGA_GPU_FP8_OK) rc = omega_gpu_fp8_gemv_grouped_f32(t, p->groups, perm, x, p->x_rows, y, &info); /* builds the kernel */
        size_t bad = 0; double worst = rc == OMEGA_GPU_FP8_OK ? parity(&W, p->groups, perm, x, p->x_rows, y, &bad) : 1.0;
        CHECK(rc == OMEGA_GPU_FP8_OK && bad == 0, "chip %s warm-up: rc=%s bad=%zu worst_rel=%.3e (%s)", p->name, omega_gpu_fp8_rc_name(rc), bad, worst, omega_gpu_fp8_last_error());
        uint64_t *el = calloc(reps, sizeof *el); uint32_t done = 0; size_t bad_all = 0; double worst_all = worst;
        if (g_evict) evict_host_caches();
        for (uint32_t r = 0; r < reps && rc == OMEGA_GPU_FP8_OK; r++) {
            const uint32_t *ids = perm + STR * (r % 16u);
            OmegaGpuFp8Info ti;
            if (omega_gpu_fp8_gemv_grouped_f32(t, p->groups, ids, x, p->x_rows, y, &ti) != OMEGA_GPU_FP8_OK) { rc = OMEGA_GPU_FP8_CHIP_FAIL; break; }
            size_t b = 0; double w = parity(&W, p->groups, ids, x, p->x_rows, y, &b);
            bad_all += b; if (w > worst_all) worst_all = w;
            el[done++] = ti.elapsed_ns;
        }
        CHECK(rc == OMEGA_GPU_FP8_OK && done == reps && bad_all == 0, "chip %s: %u/%u timed launches, %zu elements off the oracle, worst_rel=%.3e (%s)", p->name, done, reps, bad_all, worst_all, omega_gpu_fp8_last_error());
        uint64_t med = 0, lo = 0, hi = 0; double gbs = 0.0;
        if (done) { qsort(el, done, sizeof *el, cmp_u64); med = el[done / 2]; lo = el[0]; hi = el[done - 1]; gbs = med ? (double)(info.weight_bytes + info.scale_bytes) / (double)med : 0.0; }
        printf("chip %-24s variant=%d sched=%d rc=%-9s timed=%u bad=%zu worst_rel=%.3e grid=%ux%u lb=%u rows=%u warps=%u depth=%u bytes=%" PRIu64 " chip_med=%.1fus min=%.1fus max=%.1fus (%.1f GB/s, %.3f of 237.2)\n",
               p->name, omega_gpu_fp8_variant(), omega_gpu_fp8_sched(), omega_gpu_fp8_rc_name(rc), done, bad_all, worst_all, info.grid_x, info.grid_y, info.lane_bytes, info.rows_per_warp,
               info.warps_per_cta, info.depth, info.weight_bytes + info.scale_bytes, med / 1e3, lo / 1e3, hi / 1e3, gbs, gbs / 237.2);
        if (out) fprintf(out, "%s{\"name\":\"%s\",\"experts\":%u,\"n\":%u,\"k\":%u,\"groups\":%u,\"x_rows\":%u,\"rc\":\"%s\",\"timed\":%u,\"bad\":%zu,\"worst_rel\":%.3e,"
                         "\"variant\":%d,\"sched\":%d,\"grid_x\":%u,\"grid_y\":%u,\"lane_bytes\":%u,\"rows_per_warp\":%u,\"warps_per_cta\":%u,\"depth\":%u,\"bytes\":%" PRIu64 ",\"chip_median_ns\":%" PRIu64 ",\"chip_min_ns\":%" PRIu64 ",\"chip_max_ns\":%" PRIu64 ",\"gbps_chip\":%.2f,\"evict\":%d,\"rotation_sets\":16}",
                         *first ? "" : ",", p->name, p->experts, p->n, p->k, p->groups, p->x_rows, omega_gpu_fp8_rc_name(rc), done, bad_all, worst_all,
                         omega_gpu_fp8_variant(), omega_gpu_fp8_sched(), info.grid_x, info.grid_y, info.lane_bytes, info.rows_per_warp, info.warps_per_cta, info.depth, info.weight_bytes + info.scale_bytes, med, lo, hi, gbs, g_evict);
        *first = 0;
        { /* GPU-stamped timing of the same rotation (the figure CUT5 reports) */
            const uint32_t lists = reps < 2u * LIST_N ? 2u : (reps + LIST_N - 1u) / LIST_N;
            uint64_t *e2 = calloc((size_t)lists * LIST_N, sizeof *e2); uint32_t n2 = 0; size_t b2 = 0; double w2 = 0.0;
            int lrc = rc == OMEGA_GPU_FP8_OK ? cold_list_stamps(p, &W, t, perm, x, lists, e2, &n2, &b2, &w2) : -1;
            CHECK(lrc == 0 && n2 == lists * LIST_N && b2 == 0, "chip %s GPU-stamped list: %u/%u launches, %zu elements off the oracle, worst_rel=%.3e", p->name, n2, lists * LIST_N, b2, w2);
            uint64_t m2 = 0, lo2 = 0, hi2 = 0; double g2 = 0.0;
            if (n2) { qsort(e2, n2, sizeof *e2, cmp_u64); m2 = e2[n2 / 2]; lo2 = e2[0]; hi2 = e2[n2 - 1]; g2 = m2 ? (double)(info.weight_bytes + info.scale_bytes) / (double)m2 : 0.0; }
            printf("chip %-24s GPU-STAMPED coal=%d wfirst=%d launches=%u bad=%zu stamp_med=%.1fus min=%.1fus max=%.1fus (%.1f GB/s, %.3f of 237.2)\n",
                   p->name, omega_gpu_fp8_coal(), omega_gpu_fp8_wfirst(), n2, b2, m2 / 1e3, lo2 / 1e3, hi2 / 1e3, g2, g2 / 237.2);
            if (out) fprintf(out, ",{\"name\":\"%s gpu-stamped\",\"coal\":%d,\"wfirst\":%d,\"sched\":%d,\"launches\":%u,\"bad\":%zu,\"worst_rel\":%.3e,\"bytes\":%" PRIu64 ",\"stamp_median_ns\":%" PRIu64 ",\"stamp_min_ns\":%" PRIu64 ",\"stamp_max_ns\":%" PRIu64 ",\"gbps_stamp\":%.2f,\"barrier\":\"WFI\",\"floor_subtracted\":0}",
                             p->name, omega_gpu_fp8_coal(), omega_gpu_fp8_wfirst(), omega_gpu_fp8_sched(), n2, b2, w2, info.weight_bytes + info.scale_bytes, m2, lo2, hi2, g2);
            free(e2);
        }
        free(el);
        /* mutants: each must be caught at this shape and this kernel configuration */
        for (int m = 1; m <= 4 && rc == OMEGA_GPU_FP8_OK; m++) {
            if (m == 3 && p->groups == 1) continue; /* mutant 3 needs groups > 1 (as in the --n/--k path) */
            if (m == 4) omega_gpu_fp8_test_set_no_fold(1); else omega_gpu_fp8_test_set_mutant(m);
            OmegaGpuFp8Tensor *tm = t;
            if (m == 4) { tm = NULL; if (omega_gpu_fp8_tensor_upload(p->experts, p->n, p->k, W.w, W.s, &tm) != OMEGA_GPU_FP8_OK) tm = NULL; }
            int r = tm ? omega_gpu_fp8_gemv_grouped_f32(tm, p->groups, perm, x, p->x_rows, y, NULL) : OMEGA_GPU_FP8_CHIP_FAIL;
            size_t mb = 0; double mw = r == OMEGA_GPU_FP8_OK ? parity(&W, p->groups, perm, x, p->x_rows, y, &mb) : 1.0;
            int caught = r != OMEGA_GPU_FP8_OK || mb > 0;
            CHECK(caught, "chip %s mutant %d NOT caught", p->name, m);
            printf("chip %-24s mutant %d: %s (rc=%s bad=%zu worst_rel=%.3e)\n", p->name, m, caught ? "caught" : "MISSED", omega_gpu_fp8_rc_name(r), mb, mw);
            if (out) fprintf(out, ",{\"name\":\"%s mutant %d\",\"caught\":%d,\"rc\":\"%s\",\"bad\":%zu,\"worst_rel\":%.3e}", p->name, m, caught, omega_gpu_fp8_rc_name(r), mb, mw);
            if (m == 4) { omega_gpu_fp8_test_set_no_fold(0); if (tm) omega_gpu_fp8_tensor_free(tm); } else omega_gpu_fp8_test_set_mutant(0);
        }
        if (t) omega_gpu_fp8_tensor_free(t);
        free(x); free(y); weights_free(&W);
        if (omega_gpu_session_is_blocked()) { printf("session latched: stopping\n"); break; }
    }
}

static int chip_run(const char *out_path, uint32_t reps) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 1; }
    if (out) fprintf(out, "{\"test\":\"OMEGA_GPU_FP8_GEMV\",\"points\":[");
    static const ChipPoint pts[] = {
        { 1, 128, 256, 1, 1, "k256 n128", 1 },
        { 1, 256, 768, 1, 1, "k768 n256", 1 },
        { 1, 384, 2048, 1, 1, "k2048 n384", 1 },
        { 1, 768, 2048, 1, 1, "gate 768x2048", 1 },
        { 1, 1536, 2048, 1, 1, "gate+up 1536x2048", 1 },
        { 1, 2048, 768, 1, 1, "down 2048x768", 1 },
        { 1, 4096, 2048, 1, 1, "q_proj 4096x2048", 1 },
        { 1, 512, 2048, 1, 1, "k_proj 512x2048", 1 },
        { 1, 2048, 4096, 1, 1, "o_proj 2048x4096", 1 },
        { 1, 5120, 2048, 1, 1, "qkv fused 5120x2048", 1 },
        { 8, 1536, 2048, 8, 1, "8 experts gate+up shared x", 1 },
        { 8, 2048, 768, 8, 8, "8 experts down per-group x", 1 },
        { 16, 1536, 2048, 8, 1, "8 of 16 experts gate+up", 1 },
        { 1, 16384, 2048, 1, 1, "large 16384x2048", 1 },
        /* Qwen3.8-27B dense shapes (omega#541) and the new width cap */
        { 1, 17408, 5120, 1, 1, "q38 gate 17408x5120", 1 },
        { 1, 5120, 17408, 1, 1, "q38 down 5120x17408", 1 },
        { 1, 12288, 5120, 1, 1, "q38 q_proj 12288x5120", 1 },
        { 1, 1024, 5120, 1, 1, "q38 k_proj 1024x5120", 1 },
        { 1, 10240, 5120, 1, 1, "q38 in_proj_qkv 10240x5120", 1 },
        { 1, 6144, 5120, 1, 1, "q38 in_proj_z 6144x5120", 1 },
        { 1, 5120, 6144, 1, 1, "q38 out_proj 5120x6144", 1 },
        { 1, 256, 32768, 1, 1, "width cap 256x32768", 1 },
    };
    /* 0. one-hot conversion on the chip: row r holds code r at column 0 (NaN codes as 0x00), x = e_0, scale 1.
     *    y[r] must equal the reference bit for bit. The e4m3 subnormal codes 0x01..0x07 (and negatives) travel
     *    through the FFMA chain as f32 subnormals (e4m3 * 2^-120); a flush-to-zero FFMA would return 0 here. */
    int one_hot_exact = -1;
    {
        uint8_t *w = calloc(256, 256); uint16_t s[4]; float x[256] = { 0 }, y[256]; OmegaGpuFp8Tensor *t = NULL;
        for (int b = 0; b < 4; b++) s[b] = bf16_of(1.0f);
        for (int r = 0; r < 256; r++) w[r * 256] = omega_fp8_e4m3_is_nan((uint8_t)r) ? 0 : (uint8_t)r;
        x[0] = 1.0f;
        int rc = omega_gpu_fp8_tensor_upload(1, 256, 256, w, s, &t);
        if (rc == OMEGA_GPU_FP8_OK) rc = omega_gpu_fp8_gemv_f32(t, 0, x, y, NULL);
        one_hot_exact = 0;
        if (rc == OMEGA_GPU_FP8_OK) for (int r = 0; r < 256; r++) {
            float ref = omega_fp8_e4m3_is_nan((uint8_t)r) ? 0.0f : omega_fp8_e4m3_ref((uint8_t)r);
            if (y[r] == ref) one_hot_exact++; else CHECK(0, "chip one-hot code 0x%02x: kernel %a, reference %a", r, y[r], ref);
        }
        if (g_depth_forced && rc == OMEGA_GPU_FP8_CODEGEN_FAIL) printf("chip one-hot conversion: skipped (forced depth does not fit k=256)\n");
        else CHECK(rc == OMEGA_GPU_FP8_OK && one_hot_exact == 256, "chip one-hot conversion: rc=%s exact=%d/256", omega_gpu_fp8_rc_name(rc), one_hot_exact);
        printf("chip one-hot conversion: rc=%s %d/256 rows exact (subnormal codes 0x01..0x07 included: no flush to zero)\n", omega_gpu_fp8_rc_name(rc), one_hot_exact);
        if (out) fprintf(out, "{\"name\":\"one-hot conversion\",\"rc\":\"%s\",\"exact_rows\":%d,\"of\":256}", omega_gpu_fp8_rc_name(rc), one_hot_exact);
        if (t) omega_gpu_fp8_tensor_free(t);
        free(w);
    }
    static ChipPoint custom; const ChipPoint *P = pts; size_t npts = sizeof pts / sizeof pts[0];
    if (g_custom_n && !g_custom_k) { CHECK(0, "--n needs --k"); if (out) fclose(out); return 1; }
    if (g_custom_n) { snprintf(g_custom_name, sizeof g_custom_name, "custom %ux%u", g_custom_n, g_custom_k); custom = (ChipPoint){ 1, g_custom_n, g_custom_k, 1, 1, g_custom_name, 1 }; P = &custom; npts = 1; }
    int first = 1;
    first = out ? 0 : 1;
    if (g_cold) npts = 0; /* --grouped-cold: the two real expert shapes instead of the battery */
    for (size_t i = 0; i < npts; i++) {
        const ChipPoint *p = &P[i];
        if (g_only && !strstr(p->name, g_only)) continue;
        Weights W; weights_make(&W, p->experts, p->n, p->k, 5000 + (uint32_t)i);
        uint32_t ids[8]; uint32_t st = 99 + (uint32_t)i;
        for (uint32_t g = 0; g < p->groups; g++) ids[g] = (p->experts > p->groups ? lcg(&st) % p->experts : (p->groups - 1 - g) % p->experts);
        float *x = malloc((size_t)p->x_rows * p->k * 4); for (size_t j = 0; j < (size_t)p->x_rows * p->k; j++) x[j] = frand(&st, -1.0f, 1.0f);
        float *y = malloc((size_t)p->groups * p->n * 4), *y2 = malloc((size_t)p->groups * p->n * 4);
        OmegaGpuFp8Tensor *t = NULL;
        double t0 = now_ms();
        int rc = omega_gpu_fp8_tensor_upload(p->experts, p->n, p->k, W.w, W.s, &t);
        double up_ms = now_ms() - t0;
        OmegaGpuFp8Info info; memset(&info, 0, sizeof info);
        if (rc == OMEGA_GPU_FP8_OK) rc = omega_gpu_fp8_gemv_grouped_f32(t, p->groups, ids, x, p->x_rows, y, &info);
        size_t bad = 0; double worst = rc == OMEGA_GPU_FP8_OK ? parity(&W, p->groups, ids, x, p->x_rows, y, &bad) : 1.0;
        int ok = rc == OMEGA_GPU_FP8_OK && bad == 0;
        /* second call: cache hit, bit-identical */
        int same = 0; OmegaGpuFp8Info i2; memset(&i2, 0, sizeof i2);
        if (ok) { int r2 = omega_gpu_fp8_gemv_grouped_f32(t, p->groups, ids, x, p->x_rows, y2, &i2); same = r2 == OMEGA_GPU_FP8_OK && memcmp(y, y2, (size_t)p->groups * p->n * 4) == 0 && i2.kernel_cache_hit; }
        CHECK(ok && same, "chip %s: rc=%s bad=%zu worst_rel=%.3e second-call-identical=%d (%s)", p->name, omega_gpu_fp8_rc_name(rc), bad, worst, same, omega_gpu_fp8_last_error());
        /* timing: median of reps */
        uint64_t *el = calloc(reps, sizeof *el), *cl = calloc(reps, sizeof *cl);
        uint32_t done = 0;
        if (g_evict) evict_host_caches();
        for (uint32_t r = 0; r < reps && ok; r++) {
            OmegaGpuFp8Info ti; if (omega_gpu_fp8_gemv_grouped_f32(t, p->groups, ids, x, p->x_rows, y2, &ti) != OMEGA_GPU_FP8_OK) break;
            el[r] = ti.elapsed_ns; cl[r] = ti.call_ns; done++;
        }
        uint64_t med_el = 0, med_cl = 0; double gbs = 0.0, gbs_call = 0.0;
        if (done) { qsort(el, done, sizeof *el, cmp_u64); qsort(cl, done, sizeof *cl, cmp_u64); med_el = el[done / 2]; med_cl = cl[done / 2];
                    gbs = med_el ? (double)(info.weight_bytes + info.scale_bytes) / (double)med_el : 0.0; gbs_call = med_cl ? (double)(info.weight_bytes + info.scale_bytes) / (double)med_cl : 0.0; }
        printf("chip %-32s variant=%d rc=%-9s bad=%zu worst_rel=%.3e grid=%ux%u lb=%u rows=%u warps=%u depth=%u bytes=%" PRIu64 " upload=%.1fms chip_med=%.1fus (%.1f GB/s) call_med=%.1fus (%.1f GB/s) reps=%u\n",
               p->name, omega_gpu_fp8_variant(), omega_gpu_fp8_rc_name(rc), bad, worst, info.grid_x, info.grid_y, info.lane_bytes, info.rows_per_warp, info.warps_per_cta, info.depth,
               info.weight_bytes + info.scale_bytes, up_ms, med_el / 1e3, gbs, med_cl / 1e3, gbs_call, done);
        if (out) fprintf(out, "%s{\"name\":\"%s\",\"experts\":%u,\"n\":%u,\"k\":%u,\"groups\":%u,\"x_rows\":%u,\"rc\":\"%s\",\"bad\":%zu,\"worst_rel\":%.3e,\"second_call_identical\":%d,"
                         "\"variant\":%d,\"grid_x\":%u,\"grid_y\":%u,\"lane_bytes\":%u,\"rows_per_warp\":%u,\"warps_per_cta\":%u,\"depth\":%u,\"bytes\":%" PRIu64 ",\"upload_ms\":%.2f,\"chip_median_ns\":%" PRIu64 ",\"gbps_chip\":%.2f,\"call_median_ns\":%" PRIu64 ",\"gbps_call\":%.2f,\"reps\":%u,\"marker\":%u,\"evict\":%d}",
                         first ? "" : ",", p->name, p->experts, p->n, p->k, p->groups, p->x_rows, omega_gpu_fp8_rc_name(rc), bad, worst, same,
                         omega_gpu_fp8_variant(), info.grid_x, info.grid_y, info.lane_bytes, info.rows_per_warp, info.warps_per_cta, info.depth, info.weight_bytes + info.scale_bytes, up_ms, med_el, gbs, med_cl, gbs_call, done, info.completion_marker, g_evict);
        first = 0;
        free(el); free(cl);
        /* mutants on the first grouped point and the gate shape: each must be caught on the chip */
        if (ok && (g_custom_n || i == 4 || i == 10)) {
            for (int m = 1; m <= 4; m++) {
                if (m == 3 && p->groups == 1) continue;
                if (m == 4) omega_gpu_fp8_test_set_no_fold(1); else omega_gpu_fp8_test_set_mutant(m);
                OmegaGpuFp8Tensor *tm = t;
                if (m == 4) { tm = NULL; if (omega_gpu_fp8_tensor_upload(p->experts, p->n, p->k, W.w, W.s, &tm) != OMEGA_GPU_FP8_OK) tm = NULL; }
                int r = tm ? omega_gpu_fp8_gemv_grouped_f32(tm, p->groups, ids, x, p->x_rows, y2, NULL) : OMEGA_GPU_FP8_CHIP_FAIL;
                size_t mb = 0; double mw = r == OMEGA_GPU_FP8_OK ? parity(&W, p->groups, ids, x, p->x_rows, y2, &mb) : 1.0;
                int caught = r != OMEGA_GPU_FP8_OK || mb > 0;
                CHECK(caught, "chip %s mutant %d NOT caught", p->name, m);
                printf("chip %-32s mutant %d: %s (rc=%s bad=%zu worst_rel=%.3e)\n", p->name, m, caught ? "caught" : "MISSED", omega_gpu_fp8_rc_name(r), mb, mw);
                if (out) fprintf(out, ",{\"name\":\"%s mutant %d\",\"caught\":%d,\"rc\":\"%s\",\"bad\":%zu,\"worst_rel\":%.3e}", p->name, m, caught, omega_gpu_fp8_rc_name(r), mb, mw);
                if (m == 4) { omega_gpu_fp8_test_set_no_fold(0); if (tm) omega_gpu_fp8_tensor_free(tm); } else omega_gpu_fp8_test_set_mutant(0);
            }
        }
        omega_gpu_fp8_tensor_free(t);
        free(x); free(y); free(y2); weights_free(&W);
        if (omega_gpu_session_is_blocked()) { printf("session latched: stopping\n"); break; }
    }
    if (g_cold && !omega_gpu_session_is_blocked()) cold_points(out, reps, &first);
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"device_opens\":%u,\"verdict\":\"%s\"}\n", g_checks, g_failed, omega_gpu_session_open_count(), g_failed == 0 ? "OMEGA_GPU_FP8_GEMV_PASS" : "OMEGA_GPU_FP8_GEMV_FAIL"); fclose(out); }
    return 0;
}

/* L310: the grouped-launch CTA rule, pure, no chip. 8 groups keep the measured 6; 9 groups (the 35B) get 5 = one wave of 48. */
static void rule_checks(void) {
    static const struct { uint32_t groups, forced; int wave, want; } c[] = {
        {8, 0, 1, 6}, {9, 0, 1, 5}, {9, 0, 0, 6}, {9, 5, 0, 5}, {9, 5, 1, 5}, {9, 300, 1, 5}, {2, 0, 1, 6}, {48, 0, 1, 1}, {64, 0, 1, 1}, {0, 0, 1, 6} };
    for (size_t i = 0; i < sizeof c / sizeof c[0]; i++) {
        uint32_t got = omega_gpu_fp8_grouped_ctas_rule(c[i].groups, c[i].forced, c[i].wave);
        if (got != (uint32_t)c[i].want) { fprintf(stderr, "grouped CTA rule: groups %u forced %u wave %d: got %u want %d\n", c[i].groups, c[i].forced, c[i].wave, got, c[i].want); exit(1); }
    }
    printf("grouped CTA rule: %zu/%zu ok (8 groups -> 6, 9 groups -> 5)\n", sizeof c / sizeof c[0], sizeof c / sizeof c[0]);
}

int main(int argc, char **argv) {
    rule_checks();
    int host = 0, sim = 0; const char *out_path = NULL; uint32_t reps = 20, lb = 0, rows = 0, warps = 0, budget = 0, spin = 2000, depth = 0; int depth_set = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--lane-bytes") == 0 && i + 1 < argc) lb = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) rows = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--warps") == 0 && i + 1 < argc) warps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--cta-budget") == 0 && i + 1 < argc) budget = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) { depth_set = 1; depth = (uint32_t)atoi(argv[++i]); }
        else if (strcmp(argv[i], "--spin-us") == 0 && i + 1 < argc) spin = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-evict") == 0) g_evict = 0;
        else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) g_only = argv[++i];
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) g_custom_n = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) g_custom_k = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--gpr-pad") == 0 && i + 1 < argc) omega_gpu_fp8_set_gpr_pad((uint32_t)atoi(argv[++i]));
        else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) omega_gpu_fp8_set_variant(atoi(argv[++i])); /* 1 (default) or 2 (W-L2c row-pipelined) */
        else if (strcmp(argv[i], "--sched") == 0 && i + 1 < argc) omega_gpu_fp8_set_sched(atoi(argv[++i])); /* CUT5: 0 (default) or 1 (pipelined decode + stall pass) */
        else if (strcmp(argv[i], "--coal") == 0 && i + 1 < argc) { omega_gpu_fp8_set_coal(atoi(argv[++i])); g_coal_arg = 1; }     /* CUT5 variant A: coalesced output through shared slots */
        else if (strcmp(argv[i], "--wfirst") == 0 && i + 1 < argc) omega_gpu_fp8_set_wfirst(atoi(argv[++i])); /* CUT5 variant B: weight loads before the x loads */
        else if (strcmp(argv[i], "--coal-default") == 0) { coal_default_check(); printf("gpu_fp8_api_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS"); return g_failed ? 1 : 0; } /* variant A default vs env only */
        else if (strcmp(argv[i], "--deep-default") == 0) { deep_default_check(); printf("gpu_fp8_api_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS"); return g_failed ? 1 : 0; } /* deep default vs env only */
        else if (strcmp(argv[i], "--grouped-cold") == 0) g_cold = 1; /* CUT5: the two decode expert launches, 8 of 128, L2 defeated */
        else if (strcmp(argv[i], "--cold-single") == 0) g_cold = 2;  /* SPEED-1B: the two single projections qkv / o_proj, L2 defeated */
        else if (strcmp(argv[i], "--dump-stamps") == 0) g_dump_stamps = 1;
        else { printf("unknown argument %s\n", argv[i]); return 2; }
    }
    if (host) host_only();
    if (sim) sim_battery(budget ? budget : 4);
    if (!host && !sim) {
        if (lb || rows || warps) omega_gpu_fp8_set_shape(lb, rows, warps);
        if (budget) omega_gpu_fp8_set_cta_budget(budget);
        if (depth_set) { omega_gpu_fp8_set_depth(1, depth); g_depth_forced = 1; }
        (void)omega_gpu_session_set_spin_us(spin);
        chip_run(out_path, reps ? reps : 1);
        omega_gpu_session_close();
    }
    printf("gpu_fp8_api_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
