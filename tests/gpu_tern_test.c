/*
 * gpu_tern_test: KT1, the W2A8 ternary GEMV (omega_gpu_tern_api.h), ternary engine stage T0 (2026-10-15).
 *   --host-only   the word arithmetic (accp >> 16, accm >> 17) against the direct dot on random and extreme words,
 *                 trit packing round trip (both layouts), segment boundaries (explicit vs the r / (n / count) mapping),
 *                 codegen budgets at the four BitNet shapes, refusals (no device)
 *   --sim         warp-simulator parity: int32 output bit-equal to oma_rz_oracle over layouts, rows, depths, multi-CTA,
 *                 padded n, two x chunks (k > 8192); f32 epilogue with three different segment scales bit-equal to the
 *                 host formula; mutants 1 (flipped plane), 3 (dropped pair) and 2 (r / (n / count) scale map) caught
 *   (default)     the chip: 20,000 random rows plus the MA grid plus the four BitNet shapes bit-equal to oma_rz_oracle,
 *                 extremes, both layouts, second call identical, f32 q|k|v segment test, mutants 1 / 2 / 3 caught, then
 *                 cold bandwidth (L2 defeated by rotating weight copies totalling >= 96 MiB in one command list, GPU
 *                 timestamps per launch) and the in-list launch floor. --out receipt.json, --reps N, --no-bw, --bw-only,
 *                 --mutant M (runs the correctness battery with mutant M forced: the verdict must be FAIL),
 *                 --layout L --block-rows R --rows R --warps W --depth D --ctas C (shape knobs for sweeps), --weights cached|uncached, --hot (diagnostic)
 * Speeds are reported only as fractions of the physical limit: 273 GB/s (spec) and 237.2 GB/s (practical rate).
 * Verdict line last: "gpu_tern_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_gpu_tern_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_cmdlist.h"
#include "algebra/realize_common.h"
#include "bw_warp_sim_k.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SPEC_GBS 273.0
#define PRACTICAL_GBS 237.2

/* The layout the bandwidth runs use: the API default (row-major) unless --layout chose another. Restored after the
 * correctness loop, which sets both layouts in turn. */
static uint32_t g_lay = OMEGA_TERN_LAYOUT_ROWMAJOR, g_brows;
static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }

/* trits: zero with probability sparsity, else +1 / -1 evenly */
static int8_t *make_w(size_t count, double sparsity, uint32_t seed) {
    int8_t *w = malloc(count ? count : 1);
    const uint32_t zcut = (uint32_t)(sparsity * 4294967295.0);
    for (size_t i = 0; i < count; i++) { uint32_t r = lcg(&seed); w[i] = r < zcut ? 0 : (lcg(&seed) >> 31) ? 1 : -1; }
    return w;
}
static int8_t *make_x(uint32_t k, uint32_t seed) {
    int8_t *x = aligned_alloc(16, (k + 15u) & ~15u);
    for (uint32_t i = 0; i < k; i++) x[i] = (int8_t)(lcg(&seed) >> 24);
    return x;
}
static int32_t *oracle(const int8_t *w, uint32_t n, uint32_t k, const int8_t *x) {
    int32_t *y = malloc((size_t)n * 4);
    if (oma_rz_oracle(w, n, k, x, y) != OMA_RZ_OK) { free(y); return NULL; }
    return y;
}
/* f32 epilogue as the header specifies: ((float)y * a) * scale[seg], seg from explicit boundaries (this file is built with
 * -ffp-contract=off, so each product rounds to nearest f32 as FMUL does) */
static void expect_f32(const int32_t *y, uint32_t n, const uint32_t b[3], const float scale4[4], float a, float *out) {
    for (uint32_t r = 0; r < n; r++) {
        uint32_t seg = (r >= b[0]) + (r >= b[1]) + (r >= b[2]);
        float f = (float)y[r];
        f = f * a;
        f = f * scale4[seg];
        out[r] = f;
    }
}

/* ------------------------------------------------------------- host-only */
/* the kernel's arithmetic for one 16-trit word, mod 2^32, as emitted (mutant 1: staged X negated; 3: pair 7 dropped) */
static uint32_t emu_word(uint32_t W, const int8_t *x16, int mutant) {
    uint32_t accp = 0x80008000u, accm = 0x80008000u;
    for (uint32_t j = 0; j < (mutant == 3 ? 7u : 8u); j++) {
        uint32_t X = (uint32_t)(int32_t)x16[j] * 65536u + (uint32_t)(int32_t)x16[j + 8];
        if (mutant == 1) X = 0u - X;
        uint32_t U = W >> (2u * j);
        accp += (U & 0x00010001u) * X;
        accm += (U & 0x00020002u) * X;
    }
    return (accp >> 16) - (accm >> 17); /* 2^14 + dot */
}
static int32_t direct_word(uint32_t W, const int8_t *x16) {
    int32_t s = 0;
    for (uint32_t i = 0; i < 16; i++) { uint32_t c = (W >> (2u * i)) & 3u; s += (c == 1u ? 1 : c == 2u ? -1 : 0) * x16[i]; }
    return s;
}
static uint32_t rand_word(uint32_t *st, int mode) {
    uint32_t W = 0;
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t c = mode == 1 ? 1u : mode == 2 ? 2u : (lcg(st) >> 30) % 3u;
        W |= c << (2u * i);
    }
    return W;
}
/* Independent stall checker (a different walk from the generator's pass): for every fixed-latency write, scan forward
 * summing stalls; any instruction that touches one of its registers, any control instruction and any branch target
 * reached before OMEGA_TERN_FIXED_LAT cycles is a violation. */
static int chk_fixed(BlackwellIROpcode op) {
    return op == BW_IR_IMAD || op == BW_IR_IMAD_WIDE || op == BW_IR_SHF_R || op == BW_IR_LOP3_AND || op == BW_IR_MOV_IMM ||
           op == BW_IR_MOV_RZ || op == BW_IR_FMUL || op == BW_IR_I2FP;
}
static int chk_overlap(const BlackwellIRProgram *p, int a, int b) {
    if (a < 0 || b < 0) return 0;
    const int pa = p->regalloc.vreg_to_phys[a], pb = p->regalloc.vreg_to_phys[b];
    if (pa < 0 || pb < 0) return 0;
    const int na = (int)(p->regalloc.intervals[a].bundle_size ? p->regalloc.intervals[a].bundle_size : 1u);
    const int nb = (int)(p->regalloc.intervals[b].bundle_size ? p->regalloc.intervals[b].bundle_size : 1u);
    return pa < pb + nb && pb < pa + na;
}
static long stall_violations(const BlackwellIRProgram *p, uint64_t *cycles) {
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
        if (!chk_fixed(in->op) || in->dst_vreg < 0) continue;
        uint32_t dist = 0;
        for (size_t j = i + 1; j < p->count; j++) {
            dist += (p->insns[j - 1].control >> 9) & 0xfu;
            if (dist >= OMEGA_TERN_FIXED_LAT) break;
            const BlackwellIRInsn *u = &p->insns[j];
            const BlackwellIROpcode o = u->op;
            if (o == BW_IR_BRA || o == BW_IR_EXIT || o == BW_IR_BSSY || o == BW_IR_BSYNC || o == BW_IR_BAR_SYNC || tgt[j]) { bad++; break; }
            if (chk_overlap(p, in->dst_vreg, u->dst_vreg) || chk_overlap(p, in->dst_vreg, u->src1_vreg) ||
                chk_overlap(p, in->dst_vreg, u->src2_vreg) || chk_overlap(p, in->dst_vreg, u->src3_vreg)) { bad++; break; }
        }
    }
    if (cycles) *cycles = cyc;
    return bad;
}
static void host_battery(void) {
    uint32_t st = 0x7e57u; size_t bad = 0, bad1 = 0, bad3 = 0, n = 0;
    int8_t x16[16];
    const int8_t ext[4] = { -128, 127, -1, 0 };
    for (int mode = 0; mode < 3; mode++)
        for (int xm = 0; xm < 6; xm++)
            for (int rep = 0; rep < (mode == 0 && xm == 0 ? 400000 : 2000); rep++) {
                for (int i = 0; i < 16; i++) x16[i] = xm == 0 ? (int8_t)(lcg(&st) >> 24) : xm == 5 ? (int8_t)((i & 1) ? -128 : 127) : ext[xm - 1];
                uint32_t W = rand_word(&st, mode);
                int32_t want = direct_word(W, x16);
                n++;
                if (emu_word(W, x16, 0) != (1u << 14) + (uint32_t)want) bad++;
                if (emu_word(W, x16, 1) == (1u << 14) + (uint32_t)want && want != 0) bad1++;
                if (emu_word(W, x16, 3) != (1u << 14) + (uint32_t)want) bad3++;
            }
    CHECK(bad == 0, "word arithmetic: %zu of %zu words differ from the direct dot", bad, n);
    printf("host: word arithmetic exact on %zu words (random, all +1, all -1; x random, -128, 127, -1, 0, alternating)\n", n);
    CHECK(bad1 == 0 && bad3 > 0, "mutant arithmetic not distinguishable (flipped equal on %zu nonzero words, dropped pair differs on %zu)", bad1, bad3);

    /* packing round trip, both layouts, padded n */
    for (int lay = 0; lay < 2; lay++) for (uint32_t R = (lay ? 2u : 1u); R <= 4u; R *= 2u) {
        const uint32_t nn = 13, k = 512, np = omega_gpu_tern_padded_rows(nn);
        int8_t *w = make_w((size_t)nn * k, 0.3, 11u + R);
        uint32_t *pk = calloc((size_t)np * k / 16u, 4);
        int rc = omega_gpu_tern_pack(nn, k, w, (uint32_t)lay, R, pk, np);
        size_t diff = 0;
        for (uint32_t r = 0; r < np; r++) for (uint32_t c = 0; c < k / 16u; c++) {
            uint32_t word = pk[omega_tern_word_offset((uint32_t)lay, R, k, r, c) / 4u];
            for (uint32_t i = 0; i < 16; i++) {
                uint32_t code = (word >> (2u * i)) & 3u;
                int v = code == 1u ? 1 : code == 2u ? -1 : code == 0u ? 0 : 9;
                int want = r < nn ? w[(size_t)r * k + 16u * c + i] : 0;
                if (v != want) diff++;
            }
        }
        CHECK(rc == 0 && diff == 0, "pack layout %d R %u: rc %d, %zu trits differ", lay, R, rc, diff);
        free(w); free(pk);
    }
    { int8_t w2[512]; memset(w2, 0, sizeof w2); w2[300] = 2; uint32_t pk[8 * 512 / 16]; /* n_padded 8 rows of 32 words */
      CHECK(omega_gpu_tern_pack(1, 512, w2, 0, 1, pk, 8) == OMEGA_GPU_TERN_BAD_TRIT, "pack refuses a non-trit value"); }

    /* segments: explicit boundaries vs the r / (n / count) map (mutant 2) at q|k|v 3840 */
    OmegaGpuTernSegments s = { 3, { 2560, 3200, 3840, 0 }, { 0.5f, 2.0f, 3.0f, 0 } };
    uint32_t b[3], bm[3];
    CHECK(omega_gpu_tern_segment_bounds(3840, &s, 0, b) == 0 && b[0] == 2560 && b[1] == 3200 && b[2] == 3840, "explicit q|k|v boundaries");
    CHECK(omega_gpu_tern_segment_bounds(3840, &s, 2, bm) == 0 && bm[0] == 1280 && bm[1] == 2560, "mutant-2 boundaries are r / (n / 3)");
    OmegaGpuTernSegments one = { 1, { 2560 }, { 1.0f } };
    CHECK(omega_gpu_tern_segment_bounds(2560, &one, 0, b) == 0 && b[0] == 2560 && b[2] == 2560, "one segment: every boundary at n");

    /* codegen budgets at the BitNet shapes, both layouts */
    const uint32_t ks[] = { 2560, 6912, 16384 };
    for (unsigned i = 0; i < 3; i++) for (int lay = 0; lay < 2; lay++) for (int epi = 0; epi < 2; epi++) {
        if (epi && ks[i] > OMEGA_TERN_CHUNK_MAX) continue;
        OmegaTernGemvSpec sp = { ks[i], (uint32_t)lay, 4, 8, 0, (uint32_t)epi, 0 };
        omega_gpu_tern_shape_for(ks[i], (uint32_t)lay, 4, &sp.rows, &sp.warps, &sp.depth, NULL);
        uint32_t insns = 0, gprs = 0, decl = 0;
        int rc = omega_tern_gemv_prog_stats(&sp, &insns, &gprs);
        CHECK(rc == 0, "codegen k %u layout %d epi %d rows %u depth %u", sp.k, lay, epi, sp.rows, sp.depth);
        CHECK(rc != 0 || omega_gpu_tern_kernel_fits(&sp, &decl) == 0, "register file k %u layout %d", sp.k, lay);
        if (rc == 0) printf("host: codegen k %u layout %s epi %s rows %u depth %u warps %u: %u instructions, peak %u registers (declared %u), shared %u B\n",
                            sp.k, lay ? "block" : "rowmajor", epi ? "f32" : "i32", sp.rows, sp.depth, sp.warps, insns, gprs, decl, omega_tern_gemv_shared_bytes(&sp));
        {
            static BlackwellIRProgram cp; uint64_t cyc = 0;
            const int ok = rc == 0 && omega_blackwell_codegen_tern_gemv_prog(&sp, &cp) == 0;
            const long v = ok ? stall_violations(&cp, &cyc) : -1;
            CHECK(v == 0, "stall distances k %u layout %d epi %d: %ld users closer than %u cycles to a fixed-latency write", sp.k, lay, epi, v, OMEGA_TERN_FIXED_LAT);
            /* negative control: a fixed-latency stall above 1 is binding by construction, so shortening it must be caught */
            size_t at = cp.count;
            for (size_t q = 0; ok && q < cp.count; q++) if (chk_fixed(cp.insns[q].op) && ((cp.insns[q].control >> 9) & 0xfu) > 1u) { at = q; break; }
            if (ok && at < cp.count) cp.insns[at].control = (cp.insns[at].control & ~(0xfu << 9)) | (1u << 9);
            CHECK(ok && at < cp.count && stall_violations(&cp, NULL) > 0, "stall checker catches one shortened stall (k %u layout %d epi %d)", sp.k, lay, epi);
            if (ok) printf("host:   stall distances >= %u cycles hold; %" PRIu64 " stall cycles in program order; shortened-stall control caught\n", OMEGA_TERN_FIXED_LAT, cyc);
        }
    }
    /* refusals */
    OmegaTernGemvSpec r1 = { 200, 0, 1, 8, 1, 0, 0 }, r2 = { 8448, 0, 1, 8, 1, 0, 0 }, r3 = { 16896, 0, 1, 8, 1, 0, 0 },
                      r4 = { 2560, 0, 3, 8, 1, 0, 0 }, r5 = { 2560, 1, 1, 8, 1, 0, 0 }, r6 = { 2560, 0, 1, 8, 3, 0, 0 },
                      r7 = { 16384, 0, 1, 8, 1, 1, 0 }, r8 = { 2560, 0, 4, 8, 5, 0, 0 };
    CHECK(!omega_tern_gemv_spec_ok(&r1) && !omega_tern_gemv_spec_ok(&r2) && !omega_tern_gemv_spec_ok(&r3) && !omega_tern_gemv_spec_ok(&r4)
          && !omega_tern_gemv_spec_ok(&r5) && !omega_tern_gemv_spec_ok(&r6) && !omega_tern_gemv_spec_ok(&r7) && !omega_tern_gemv_spec_ok(&r8),
          "spec refusals: k 200, k 8448 (two chunks need k %% 512), k 16896, rows 3, block rows 1, depth 3 at k 2560, f32 with two chunks, rows*depth > 16");
    OmegaTernGemvSpec okk = { 16384, 0, 1, 8, 16, 0, 0 };
    CHECK(omega_tern_gemv_spec_ok(&okk) && omega_tern_gemv_chunks(&okk) == 2 && omega_tern_gemv_shared_bytes(&okk) == 16384, "k 16384: two chunks of 8192, 16 KiB shared");
    int8_t badw[512]; memset(badw, 0, sizeof badw); badw[7] = 2;
    OmegaGpuTernTensor *t = NULL;
    CHECK(omega_gpu_tern_tensor_upload(1, 256, badw, NULL, &t) == OMEGA_GPU_TERN_BAD_TRIT && !t, "a non-trit weight is refused before any device use");
    CHECK(omega_gpu_tern_tensor_upload(1, 200, badw, NULL, &t) == OMEGA_GPU_TERN_BAD_ARGS, "k 200 refused at upload");
    OmegaGpuTernSegments badseg = { 2, { 1, 3 }, { 1, 1 } };
    badw[7] = 0;
    CHECK(omega_gpu_tern_tensor_upload(2, 256, badw, &badseg, &t) == OMEGA_GPU_TERN_BAD_ARGS, "segments whose last end is not n are refused");
}

/* ------------------------------------------------------------- simulator */
static uint64_t g_sim_issues;
/* Runs spec on the simulator; y_out n words. bounds: the boundaries to pass (NULL: one segment). */
static int sim_run(const OmegaTernGemvSpec *spec, uint32_t n, const int8_t *w, const int8_t *x, const uint32_t *bounds, const float *scale4,
                   float a, uint32_t grid_x, int order, uint32_t *y_out, char *msg, size_t msgn) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_tern_gemv_prog(spec, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    if (stall_violations(&prog, NULL) != 0) { snprintf(msg, msgn, "stall distance violation"); return -1; }
    const uint32_t k = spec->k, np = omega_gpu_tern_padded_rows(n);
    const size_t wbytes = (size_t)np * k / 4u;
    uint32_t *pk = aligned_alloc(16, wbytes);
    if (omega_gpu_tern_pack(n, k, w, spec->layout, spec->rows, pk, np) != 0) { free(pk); snprintf(msg, msgn, "pack failed"); return -1; }
    int8_t *xd = aligned_alloc(16, (k + 15u) & ~15u); memcpy(xd, x, k);
    uint32_t *y = aligned_alloc(16, ((size_t)n * 4 + 15) & ~(size_t)15);
    float *ws = aligned_alloc(16, 16), *as = aligned_alloc(16, 16);
    float one4[4] = { 1, 1, 1, 1 };
    const float *sc = scale4 ? scale4 : one4;
    for (int i = 0; i < 4; i++) ws[i] = sc[3 - i];
    as[0] = a;
    const uint32_t sh = omega_tern_gemv_shared_bytes(spec);
    uint8_t *smem = aligned_alloc(16, sh);
    for (uint32_t i = 0; i < n; i++) y[i] = 0xdeadbeefu;
    WsSim s; memset(&s, 0, sizeof s);
    s.p = &prog; s.threads = 32u * spec->warps; s.order = order; s.smem = smem; s.smem_bytes = sh;
    uint64_t A[5] = { (uintptr_t)pk, (uintptr_t)xd, (uintptr_t)y, (uintptr_t)ws, (uintptr_t)as };
    for (unsigned i = 0; i < 5; i++) { s.cbank[(OMEGA_TERN_ARG_W + 8 * i) / 4] = (uint32_t)A[i]; s.cbank[(OMEGA_TERN_ARG_W + 8 * i) / 4 + 1] = (uint32_t)(A[i] >> 32); }
    s.cbank[OMEGA_TERN_ARG_U / 4] = np / (2u * spec->rows);
    s.cbank[OMEGA_TERN_ARG_STRIDE / 4] = grid_x * spec->warps;
    s.cbank[OMEGA_TERN_ARG_NEGN / 4] = 0u - n;
    for (int i = 0; i < 3; i++) s.cbank[OMEGA_TERN_ARG_NEGB / 4 + i] = 0u - (bounds ? bounds[i] : n);
    s.nranges = 5;
    s.ranges[0].lo = (uintptr_t)pk; s.ranges[0].hi = (uintptr_t)pk + wbytes;
    s.ranges[1].lo = (uintptr_t)xd; s.ranges[1].hi = (uintptr_t)xd + k;
    s.ranges[2].lo = (uintptr_t)y; s.ranges[2].hi = (uintptr_t)y + (size_t)n * 4;
    s.ranges[3].lo = (uintptr_t)ws; s.ranges[3].hi = (uintptr_t)ws + 16;
    s.ranges[4].lo = (uintptr_t)as; s.ranges[4].hi = (uintptr_t)as + 4;
    int err = ws_run(&s, grid_x, 1);
    if (err) snprintf(msg, msgn, "%s (pc %d)", s.msg, s.err_pc);
    else {
        g_sim_issues += s.warp_issues;
        size_t unwritten = 0; for (uint32_t i = 0; i < n; i++) if (y[i] == 0xdeadbeefu) unwritten++;
        if (unwritten) { err = -1; snprintf(msg, msgn, "%zu output words never written", unwritten); }
        memcpy(y_out, y, (size_t)n * 4);
    }
    free(pk); free(xd); free(y); free(ws); free(as); free(smem);
    return err;
}
/* one int32 point: returns rows differing from the oracle (or -1 on a sim error) */
static long sim_i32(uint32_t k, uint32_t n, uint32_t layout, uint32_t rows, uint32_t warps, uint32_t depth, uint32_t grid, int mutant, uint32_t seed, int order) {
    OmegaTernGemvSpec sp = { k, layout, rows, warps, depth, OMEGA_TERN_EPI_I32, mutant };
    int8_t *w = make_w((size_t)n * k, 0.3, seed), *x = make_x(k, seed + 7);
    int32_t *ref = oracle(w, n, k, x);
    uint32_t *y = malloc((size_t)n * 4);
    char msg[240];
    long bad = -1;
    if (sim_run(&sp, n, w, x, NULL, NULL, 1.0f, grid, order, y, msg, sizeof msg) != 0) printf("sim k %u n %u: %s\n", k, n, msg);
    else { bad = 0; for (uint32_t i = 0; i < n; i++) if ((int32_t)y[i] != ref[i]) bad++; }
    free(w); free(x); free(ref); free(y);
    return bad;
}
static void sim_battery(void) {
    struct { uint32_t k, n, layout, rows, warps, depth, grid; } pts[] = {
        { 256, 16, 0, 1, 2, 1, 1 }, { 512, 13, 0, 2, 2, 2, 2 }, { 1024, 40, 0, 4, 4, 4, 2 }, { 2560, 24, 0, 2, 2, 5, 1 },
        { 512, 16, 1, 2, 2, 2, 1 }, { 1024, 37, 1, 4, 2, 4, 3 }, { 2560, 32, 1, 4, 4, 2, 1 }, { 2560, 8, 1, 2, 1, 5, 1 },
        { 768, 64, 1, 4, 2, 1, 2 }, { 8704, 16, 0, 2, 1, 1, 1 }, { 8704, 16, 1, 2, 2, 1, 1 },
        { 2560, 20, 0, 1, 2, 10, 2 }, { 6912, 9, 0, 1, 2, 9, 1 }, { 16384, 8, 0, 1, 1, 16, 1 }, /* the API defaults (row-major, rows 1) */
    };
    for (unsigned i = 0; i < sizeof pts / sizeof pts[0]; i++)
        for (int order = 0; order < 2; order++) {
            long bad = sim_i32(pts[i].k, pts[i].n, pts[i].layout, pts[i].rows, pts[i].warps, pts[i].depth, pts[i].grid, 0, 100u + i, order);
            CHECK(bad == 0, "sim k %u n %u layout %u rows %u warps %u depth %u grid %u order %d: %ld rows differ from oma_rz_oracle",
                  pts[i].k, pts[i].n, pts[i].layout, pts[i].rows, pts[i].warps, pts[i].depth, pts[i].grid, order, bad);
        }
    printf("sim: %u int32 points x 2 orders bit-equal to oma_rz_oracle (%" PRIu64 " warp issues)\n", (unsigned)(sizeof pts / sizeof pts[0]), g_sim_issues);
    /* extremes: x all -128 / 127 with all-+1 and all--1 rows */
    {
        const uint32_t k = 1024, n = 8;
        int8_t *w = malloc((size_t)n * k), *x = make_x(k, 1);
        for (uint32_t r = 0; r < n; r++) for (uint32_t c = 0; c < k; c++) w[(size_t)r * k + c] = (int8_t)((r & 1) ? -1 : 1);
        for (int xv = 0; xv < 2; xv++) {
            for (uint32_t c = 0; c < k; c++) x[c] = (int8_t)(xv ? 127 : -128);
            int32_t *ref = oracle(w, n, k, x); uint32_t y[8]; char msg[240];
            OmegaTernGemvSpec sp = { k, 1, 4, 1, 4, 0, 0 };
            int e = sim_run(&sp, n, w, x, NULL, NULL, 1.0f, 1, 0, y, msg, sizeof msg);
            size_t bad = 0; for (uint32_t i = 0; i < n && !e; i++) if ((int32_t)y[i] != ref[i]) bad++;
            CHECK(!e && !bad, "sim extremes x=%d: %s, %zu rows differ (want %d / %d)", xv ? 127 : -128, e ? msg : "ok", bad, ref[0], ref[1]);
            free(ref);
        }
        free(w); free(x);
    }
    /* mutants 1 and 3 must be caught */
    for (int m = 1; m <= 3; m += 2) {
        long bad = sim_i32(1024, 16, 1, 2, 2, 2, 1, m, 900u + (uint32_t)m, 0);
        CHECK(bad > 0, "sim mutant %d not caught (%ld rows differ)", m, bad);
        printf("sim: mutant %d caught (%ld of 16 rows differ)\n", m, bad);
    }
    /* f32 epilogue: q|k|v-shaped segments with three different scales, small rows: 48 rows split 32 | 8 | 8 */
    {
        const uint32_t k = 512, n = 48;
        int8_t *w = make_w((size_t)n * k, 0.3, 77), *x = make_x(k, 78);
        int32_t *ref = oracle(w, n, k, x);
        OmegaGpuTernSegments seg = { 3, { 32, 40, 48 }, { 0.5f, 2.0f, 3.0f } };
        float sc4[4] = { 0.5f, 2.0f, 3.0f, 0 }, a = 0.0123f, want[48];
        uint32_t b[3], bm[3], y[48]; char msg[240];
        b[0] = 32; b[1] = 40; b[2] = 48; /* written out, not taken from the API under test */
        omega_gpu_tern_segment_bounds(n, &seg, 2, bm);
        expect_f32(ref, n, b, sc4, a, want);
        OmegaTernGemvSpec sp = { k, 1, 2, 2, 2, OMEGA_TERN_EPI_F32, 0 };
        for (int pass = 0; pass < 2; pass++) {
            int e = sim_run(&sp, n, w, x, pass ? bm : b, sc4, a, 2, 0, y, msg, sizeof msg);
            size_t bad = 0; for (uint32_t i = 0; i < n && !e; i++) if (memcmp(&y[i], &want[i], 4)) bad++;
            if (!pass) CHECK(!e && !bad, "sim f32 segments: %s, %zu of %u rows differ from ((float)y * a) * scale[seg]", e ? msg : "ok", bad, n);
            else { CHECK(!e && bad > 0, "sim mutant 2 (r / (n / count) scale map) not caught"); printf("sim: mutant 2 caught (%zu of %u rows take the wrong scale)\n", bad, n); }
        }
        free(w); free(x); free(ref);
    }
}

/* ------------------------------------------------------------- chip */
static char g_rcpt[65536]; static size_t g_rl;
#define RC(...) do { g_rl += (size_t)snprintf(g_rcpt + g_rl, sizeof g_rcpt - g_rl, __VA_ARGS__); } while (0)

/* upload + one i32 call + oracle compare; returns rows that differ (-1 on an API error) */
static long chip_point(uint32_t n, uint32_t k, double sp, uint32_t seed, int second, uint64_t *rows_checked) {
    int8_t *w = make_w((size_t)n * k, sp, seed), *x = make_x(k, seed + 3);
    int32_t *ref = oracle(w, n, k, x), *y = malloc((size_t)n * 4), *y2 = malloc((size_t)n * 4);
    OmegaGpuTernTensor *t = NULL; long bad = -1;
    int rc = omega_gpu_tern_tensor_upload(n, k, w, NULL, &t);
    if (rc == 0) rc = omega_gpu_tern_gemv_i32(t, x, y, NULL);
    if (rc != 0) printf("chip n %u k %u: %s (%s)\n", n, k, omega_gpu_tern_rc_name(rc), omega_gpu_tern_last_error());
    else {
        bad = 0; for (uint32_t i = 0; i < n; i++) if (y[i] != ref[i]) bad++;
        if (rows_checked) *rows_checked += n;
        if (second) { rc = omega_gpu_tern_gemv_i32(t, x, y2, NULL); if (rc != 0 || memcmp(y, y2, (size_t)n * 4)) bad = bad ? bad : 1, printf("chip n %u k %u: second call differs\n", n, k); }
    }
    omega_gpu_tern_tensor_free(t);
    free(w); free(x); free(ref); free(y); free(y2);
    return bad;
}

static void chip_correctness(void) {
    uint64_t rows = 0; long bad;
    /* 20,000 random rows: 10 tensors x 2000 rows, k and sparsity drawn per tensor */
    uint32_t st = 0x20000u; uint64_t r_rows = 0; long r_bad = 0;
    for (int i = 0; i < 10; i++) {
        uint32_t k = 512u * (1u + lcg(&st) % 32u); double sp = (double)(lcg(&st) % 1000u) / 1000.0;
        bad = chip_point(2000, k, sp, 5000u + (uint32_t)i, i == 0, &r_rows);
        CHECK(bad == 0, "random tensor %d (2000 x %u, sparsity %.3f): %ld rows differ from oma_rz_oracle", i, k, sp, bad);
        r_bad += bad < 0 ? 2000 : bad;
    }
    rows += r_rows;
    printf("chip: random rows %" PRIu64 ", rows differing from oma_rz_oracle %ld\n", r_rows, r_bad);
    RC("  \"random_rows\": {\"rows\": %" PRIu64 ", \"differing\": %ld},\n", r_rows, r_bad);
    /* MA grid: width {1024, 4096, 16384} x rows {1, 64, 4096} x sparsity {0, 0.3, 0.6, 0.9}, x uniform int8 */
    const uint32_t widths[3] = { 1024, 4096, 16384 }, mrows[3] = { 1, 64, 4096 }; const double sps[4] = { 0, 0.3, 0.6, 0.9 };
    long g_bad = 0; int pts = 0; uint64_t g_rows = 0;
    for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) for (int c = 0; c < 4; c++) {
        bad = chip_point(mrows[b], widths[a], sps[c], 7000u + (uint32_t)(a * 100 + b * 10 + c), 0, &g_rows);
        CHECK(bad == 0, "MA grid width %u rows %u sparsity %.1f: %ld rows differ", widths[a], mrows[b], sps[c], bad);
        g_bad += bad < 0 ? (long)mrows[b] : bad; pts++;
    }
    rows += g_rows;
    printf("chip: MA grid %d points (%" PRIu64 " rows), rows differing %ld\n", pts, g_rows, g_bad);
    RC("  \"ma_grid\": {\"points\": %d, \"rows\": %" PRIu64 ", \"differing\": %ld},\n", pts, g_rows, g_bad);
    /* BitNet shapes, both layouts */
    const uint32_t shp[4][2] = { { 3840, 2560 }, { 2560, 2560 }, { 13824, 2560 }, { 2560, 6912 } };
    long s_bad = 0;
    for (int lay = 0; lay < 2; lay++) {
        omega_gpu_tern_set_layout((uint32_t)lay, 4);
        for (int i = 0; i < 4; i++) {
            bad = chip_point(shp[i][0], shp[i][1], 1.0 / 3.0, 9000u + (uint32_t)i, 1, &rows);
            CHECK(bad == 0, "BitNet shape %ux%u layout %d: %ld rows differ", shp[i][0], shp[i][1], lay, bad);
            s_bad += bad < 0 ? (long)shp[i][0] : bad;
        }
    }
    omega_gpu_tern_set_layout(g_lay, g_brows);
    printf("chip: BitNet shapes x 2 layouts, rows differing %ld\n", s_bad);
    RC("  \"bitnet_shapes\": {\"layouts\": 2, \"differing\": %ld},\n", s_bad);
    /* extremes: all +1 / all -1 rows, x all -128 and all 127, k 16384 (|y| = 2^21) */
    {
        const uint32_t n = 16, k = 16384;
        int8_t *w = malloc((size_t)n * k), *x = make_x(k, 3);
        for (uint32_t r = 0; r < n; r++) for (uint32_t c = 0; c < k; c++) w[(size_t)r * k + c] = (int8_t)((r & 1) ? -1 : 1);
        OmegaGpuTernTensor *t = NULL; long eb = 0;
        int rc = omega_gpu_tern_tensor_upload(n, k, w, NULL, &t);
        for (int xv = 0; xv < 2 && rc == 0; xv++) {
            for (uint32_t c = 0; c < k; c++) x[c] = (int8_t)(xv ? 127 : -128);
            int32_t *ref = oracle(w, n, k, x), y[16];
            rc = omega_gpu_tern_gemv_i32(t, x, y, NULL);
            for (uint32_t i = 0; i < n && rc == 0; i++) if (y[i] != ref[i]) eb++;
            free(ref);
        }
        CHECK(rc == 0 && eb == 0, "extremes (|y| = 128 * 16384): %s, %ld rows differ", omega_gpu_tern_rc_name(rc), eb);
        omega_gpu_tern_tensor_free(t); free(w); free(x);
    }
    printf("chip: rows checked against oma_rz_oracle %" PRIu64 "\n", rows);
}

/* f32 q|k|v: three different scales at the real boundaries 2560 | 3200 | 3840; returns rows off the host formula */
static long chip_qkv(int mutant2, int *rc_out) {
    const uint32_t n = 3840, k = 2560;
    int8_t *w = make_w((size_t)n * k, 1.0 / 3.0, 4242), *x = make_x(k, 4243);
    int32_t *ref = oracle(w, n, k, x);
    OmegaGpuTernSegments seg = { 3, { 2560, 3200, 3840 }, { 0.75f, 1.5f, 2.25f } };
    float sc4[4] = { 0.75f, 1.5f, 2.25f, 0 }, *want = malloc((size_t)n * 4), *y = malloc((size_t)n * 4);
    int mx = 0; for (uint32_t i = 0; i < k; i++) mx = abs(x[i]) > mx ? abs(x[i]) : mx;
    const float a = (float)mx / 127.0f; /* 1 / s_a, s_a = 127 / max|x| */
    const uint32_t b[3] = { 2560, 3200, 3840 }; /* written out, not taken from the API under test */
    expect_f32(ref, n, b, sc4, a, want);
    if (mutant2) omega_gpu_tern_test_set_mutant(2);
    OmegaGpuTernTensor *t = NULL; long bad = -1;
    int rc = omega_gpu_tern_tensor_upload(n, k, w, &seg, &t);
    if (rc == 0) rc = omega_gpu_tern_gemv_f32(t, x, a, y, NULL);
    if (rc == 0) { bad = 0; for (uint32_t i = 0; i < n; i++) if (memcmp(&y[i], &want[i], 4)) bad++; }
    if (mutant2) omega_gpu_tern_test_set_mutant(0);
    *rc_out = rc;
    omega_gpu_tern_tensor_free(t); free(w); free(x); free(ref); free(want); free(y);
    return bad;
}

static void chip_mutants(void) {
    for (int m = 1; m <= 3; m += 2) {
        omega_gpu_tern_test_set_mutant(m);
        long bad = chip_point(512, 2560, 1.0 / 3.0, 31337u + (uint32_t)m, 0, NULL);
        omega_gpu_tern_test_set_mutant(0);
        CHECK(bad > 0, "chip mutant %d not caught (%ld rows differ)", m, bad);
        printf("chip: mutant %d caught (%ld of 512 rows differ from oma_rz_oracle)\n", m, bad);
        RC("  \"mutant_%d\": {\"rows\": 512, \"differing\": %ld, \"caught\": %s},\n", m, bad, bad > 0 ? "true" : "false");
    }
    int rc = 0;
    long bad = chip_qkv(0, &rc);
    CHECK(rc == 0 && bad == 0, "f32 q|k|v segments (scales 0.75 / 1.5 / 2.25): %s, %ld rows differ from ((float)y * a) * scale[seg]", omega_gpu_tern_rc_name(rc), bad);
    printf("chip: f32 q|k|v segments, rows differing from the host formula %ld\n", bad);
    RC("  \"f32_qkv\": {\"rows\": 3840, \"differing\": %ld},\n", bad);
    long bm = chip_qkv(1, &rc);
    CHECK(rc == 0 && bm > 0, "chip mutant 2 (r / (n / 3) scale map) not caught (%ld rows differ)", bm);
    printf("chip: mutant 2 caught (%ld of 3840 rows take the wrong scale)\n", bm);
    RC("  \"mutant_2\": {\"rows\": 3840, \"differing\": %ld, \"caught\": %s},\n", bm, bm > 0 ? "true" : "false");
}

/* cold per-launch time: `copies` uploads of the same n x k tensor, one command list reading each once, GPU stamps per
 * launch; median over reps lists. Returns median ns (0 on failure). */
/* --hot (diagnostic, never a gate): one tensor launched copies times, so after the first launch it is read from L2;
 * the gap between hot and cold separates instruction issue from memory. */
static int g_hot;
/* --shape NxK (repeatable, diagnostic, never a gate): bandwidth at these shapes instead of the four BitNet gate shapes */
static uint32_t g_shape[8][2]; static int g_shape_n;
static uint64_t cold_ns(uint32_t n, uint32_t k, uint32_t copies, uint32_t reps, uint64_t *bytes_out, uint64_t *min_out) {
    int8_t *w = make_w((size_t)n * k, 1.0 / 3.0, 555u + n + k), *x = make_x(k, 556);
    int32_t *ref = oracle(w, n, k, x), *y = malloc((size_t)n * 4);
    OmegaGpuTernTensor **ts = calloc(copies, sizeof *ts);
    uint64_t *d = calloc((size_t)copies * reps, 8), med = 0; size_t nd = 0;
    OmegaGpuCmdList *cl = NULL; OmegaGpuMem xd, yd; memset(&xd, 0, sizeof xd); memset(&yd, 0, sizeof yd);
    int ok = 1;
    const uint32_t nt = g_hot ? 1u : copies;
    /* Each launch writes its own output slice (n rounded up to 8 words), all poisoned before the list and each checked against
     * oma_rz_oracle after it, so a launch that did no work cannot hide inside the timing. */
    const size_t ys = ((size_t)n + 7u) & ~(size_t)7u;
    for (uint32_t i = 0; i < nt && ok; i++) if (omega_gpu_tern_tensor_upload(n, k, w, NULL, &ts[i]) != 0) ok = 0;
    if (ok && omega_gpu_tern_gemv_i32(ts[0], x, y, NULL) != 0) ok = 0; /* builds the kernel */
    if (ok && memcmp(y, ref, (size_t)n * 4)) { printf("cold %ux%u: reference call differs from oma_rz_oracle\n", n, k); ok = 0; }
    if (ok && omega_gpu_cmdlist_create(copies, &cl) != 0) ok = 0;
    omega_gpu_session_lock();
    if (ok && (omega_gpu_session_alloc((k + 15u) & ~15u, &xd) != 0 || omega_gpu_session_alloc(ys * 4u * copies, &yd) != 0)) ok = 0;
    omega_gpu_session_unlock();
    if (ok) { memcpy(xd.cpu, x, k); if (omega_gpu_session_sync_to_gpu(&xd) != 0) ok = 0; }
    if (bytes_out) *bytes_out = ok ? omega_gpu_tern_tensor_bytes(ts[0]) : 0;
    for (uint32_t r = 0; r < reps && ok; r++) {
        int32_t *yy = yd.cpu; for (size_t i = 0; i < ys * copies; i++) yy[i] = OMEGA_GPU_TERN_POISON_I32;
        if (omega_gpu_session_sync_to_gpu(&yd) != 0) { ok = 0; break; }
        omega_gpu_session_lock();
        omega_gpu_cmdlist_reset(cl);
        for (uint32_t i = 0; i < copies && ok; i++) if (omega_gpu_tern_gemv_record(cl, ts[g_hot ? 0 : i], xd.va, yd.va + (uint64_t)i * ys * 4u, 0) != 0) { printf("record: %s\n", omega_gpu_tern_last_error()); ok = 0; }
        OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.mode = OMEGA_GPU_CMDLIST_PCAS; o.barrier = OMEGA_GPU_CMDLIST_BARRIER_WFI; o.stamps = 1;
        OmegaGpuCmdListStats s; memset(&s, 0, sizeof s);
        uint64_t *sts = calloc(copies + 1u, 8); int cnt = -1;
        if (ok && omega_gpu_cmdlist_submit(cl, &o, &s) != 0) { printf("submit: %s\n", omega_gpu_session_last_error()); ok = 0; }
        if (ok) cnt = omega_gpu_cmdlist_stamps(cl, sts, copies + 1u);
        omega_gpu_session_unlock();
        if (ok && cnt == (int)copies + 1) for (uint32_t i = 0; i < copies; i++) d[nd++] = sts[i + 1] - sts[i]; else ok = 0;
        free(sts);
        if (ok && omega_gpu_session_sync_to_cpu(&yd) != 0) ok = 0;
        for (uint32_t i = 0; i < copies && ok; i++)
            if (memcmp((const int32_t *)yd.cpu + i * ys, ref, (size_t)n * 4)) { printf("cold %ux%u: launch %u of the list differs from oma_rz_oracle\n", n, k, i); ok = 0; }
    }
    if (ok && nd) { qsort(d, nd, 8, cmp_u64); med = d[nd / 2]; if (min_out) *min_out = d[0]; }
    CHECK(ok, "cold run %ux%u completed with results equal to oma_rz_oracle", n, k);
    omega_gpu_cmdlist_free(cl);
    omega_gpu_session_lock(); if (xd.cpu) omega_gpu_session_free(&xd); if (yd.cpu) omega_gpu_session_free(&yd); omega_gpu_session_unlock();
    for (uint32_t i = 0; i < nt; i++) omega_gpu_tern_tensor_free(ts[i]);
    free(ts); free(d); free(w); free(x); free(ref); free(y);
    return ok ? med : 0;
}

static void chip_bandwidth(uint32_t reps) {
    /* launch floor in the same list form: 32 launches of an 8 x 256 tensor */
    uint64_t fb = 0, fmin = 0;
    uint64_t floor_ns = cold_ns(8, 256, 32, reps, &fb, &fmin);
    printf("chip: in-list launch floor (8 x 256, %" PRIu64 " B) median %.2f us, min %.2f us\n", fb, floor_ns / 1e3, fmin / 1e3);
    RC("  \"launch_floor_in_list\": {\"median_us\": %.3f, \"min_us\": %.3f},\n", floor_ns / 1e3, fmin / 1e3);
    static const uint32_t gate_shp[4][2] = { { 3840, 2560 }, { 2560, 2560 }, { 13824, 2560 }, { 2560, 6912 } };
    const uint32_t (*shp)[2] = g_shape_n ? (const uint32_t (*)[2])g_shape : gate_shp;
    const int nshp = g_shape_n ? g_shape_n : 4;
    RC("  \"%s\": [\n", g_hot ? "hot_diagnostic" : "cold");
    for (int i = 0; i < nshp; i++) {
        const uint32_t n = shp[i][0], k = shp[i][1];
        const uint64_t bytes = (uint64_t)omega_gpu_tern_padded_rows(n) * k / 4u, need = 96ull << 20;
        uint32_t copies = (uint32_t)((need + bytes - 1) / bytes); if (copies < 4) copies = 4;
        uint64_t b = 0, mn = 0;
        uint64_t ns = cold_ns(n, k, copies, reps, &b, &mn);
        const double gbs = ns ? (double)b / (double)ns : 0, floor_gate = (n == 13824 ? 200.0 : 180.0);
        const double kern_ns = ns > floor_ns ? (double)(ns - floor_ns) : 0;
        printf("chip: %s %ux%u (%" PRIu64 " B x %u launches = %.1f MiB per list): %.2f us per launch -> %.1f GB/s = %.3f of 273 spec, %.3f of 237.2 practical; "
               "minus the launch floor %.2f us -> %.1f GB/s (%.3f of 237.2)\n",
               g_hot ? "hot (diagnostic, L2 resident)" : "cold", n, k, b, copies, (double)b * copies / 1048576.0, ns / 1e3, gbs, gbs / SPEC_GBS, gbs / PRACTICAL_GBS,
               kern_ns / 1e3, kern_ns > 0 ? (double)b / kern_ns : 0, kern_ns > 0 ? (double)b / kern_ns / PRACTICAL_GBS : 0);
        /* The gate is judged on the per-launch time minus the in-list launch floor. The floor is a real 8 x 256 launch (setup,
         * x staging, barrier, one small weight read), so this is a marginal rate, not an isolated kernel time. With the floor
         * counted, even the 273 GB/s spec cannot reach 180 GB/s at 2560 x 2560 (evidence README, calc ids there). */
        const double kgbs = kern_ns > 0 ? (double)b / kern_ns : 0;
        if (!g_hot && !g_shape_n) CHECK(kgbs >= floor_gate, "cold GB/s minus the launch floor at %ux%u: %.1f (%.1f with it), gate >= %.0f", n, k, kgbs, gbs, floor_gate);
        RC("    {\"n\": %u, \"k\": %u, \"bytes\": %" PRIu64 ", \"copies\": %u, \"median_ns\": %" PRIu64 ", \"min_ns\": %" PRIu64 ", \"gbs\": %.2f, "
           "\"of_spec_273\": %.4f, \"of_practical_237_2\": %.4f, \"minus_floor_ns\": %.0f, \"minus_floor_gbs\": %.2f, \"minus_floor_of_spec_273\": %.4f, \"minus_floor_of_practical_237_2\": %.4f, \"gate_gbs\": %.0f, \"gate_on\": \"minus_floor_gbs\", \"pass\": %s}%s\n",
           n, k, b, copies, ns, mn, gbs, gbs / SPEC_GBS, gbs / PRACTICAL_GBS, kern_ns, kgbs, kgbs / SPEC_GBS, kgbs / PRACTICAL_GBS, floor_gate, kgbs >= floor_gate ? "true" : "false", i < nshp - 1 ? "," : "");
    }
    RC("  ],\n");
}

int main(int argc, char **argv) {
    int host_only = 0, sim = 0, no_bw = 0, bw_only = 0, mutant = 0; uint32_t reps = 5;
    uint32_t layout = 99, brows = 0, rows = 0, warps = 0, depth = 0, ctas = 0;
    const char *wmode = NULL; /* --weights cached|uncached: the weight memory type (default: the API default) */
    const char *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) host_only = 1;
        else if (!strcmp(argv[i], "--sim")) sim = 1;
        else if (!strcmp(argv[i], "--no-bw")) no_bw = 1;
        else if (!strcmp(argv[i], "--bw-only")) bw_only = 1;
        else if (!strcmp(argv[i], "--hot")) { g_hot = 1; bw_only = 1; }
        else if (!strcmp(argv[i], "--shape") && i + 1 < argc && g_shape_n < 8) {
            unsigned a = 0, b = 0;
            if (sscanf(argv[++i], "%ux%u", &a, &b) != 2) { fprintf(stderr, "--shape wants NxK\n"); return 2; }
            g_shape[g_shape_n][0] = a; g_shape[g_shape_n][1] = b; g_shape_n++; bw_only = 1;
        }
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mutant") && i + 1 < argc) mutant = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layout") && i + 1 < argc) layout = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--block-rows") && i + 1 < argc) brows = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warps") && i + 1 < argc) warps = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depth") && i + 1 < argc) depth = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctas") && i + 1 < argc) ctas = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--weights") && i + 1 < argc) wmode = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else { fprintf(stderr, "usage: %s [--host-only | --sim] [--no-bw | --bw-only] [--reps N] [--mutant M] [--layout L] [--block-rows R] [--rows R] [--warps W] [--depth D] [--ctas C] [--weights cached|uncached] [--hot] [--out receipt.json]\n", argv[0]); return 2; }
    }
    if (reps < 1) reps = 1;
    if (reps > 64) reps = 64;
    if (host_only) host_battery();
    else if (sim) sim_battery();
    else {
        if (layout != 99) { g_lay = layout; g_brows = brows; omega_gpu_tern_set_layout(layout, brows); }
        if (rows || warps || depth) omega_gpu_tern_set_shape(rows, warps, depth);
        if (ctas) omega_gpu_tern_set_cta_budget(ctas);
        if (wmode) omega_gpu_tern_set_weight_cache(strcmp(wmode, "uncached") != 0);
        RC("{\n  \"test\": \"gpu_tern_test\",\n  \"stage\": \"TERN-T0 KT1\",\n  \"mutant_forced\": %d,\n", mutant);
        uint32_t sr = 0, sw = 0, sd = 0, sc = 0; omega_gpu_tern_shape_for(2560, g_lay, g_brows ? g_brows : 4, &sr, &sw, &sd, &sc);
        RC("  \"shape_k2560\": {\"layout\": %u, \"rows\": %u, \"warps\": %u, \"depth\": %u, \"ctas\": %u},\n", g_lay, sr, sw, sd, sc);
        if (mutant) {
            if (mutant == 2) { int rc = 0; long bad = chip_qkv(1, &rc); /* the f32 q|k|v test with the r / (n / 3) map forced */
                               CHECK(rc == 0 && bad == 0, "f32 q|k|v with mutant 2 forced: %s, %ld of 3840 rows differ from ((float)y * a) * scale[seg]", omega_gpu_tern_rc_name(rc), bad);
                               RC("  \"forced_mutant_2_differing\": %ld,\n", bad); }
            else { omega_gpu_tern_test_set_mutant(mutant); chip_correctness(); omega_gpu_tern_test_set_mutant(0); }
        } else {
            if (!bw_only) { chip_correctness(); chip_mutants(); }
            if (!no_bw) chip_bandwidth(reps);
        }
        RC("  \"checks\": %d, \"failed\": %d, \"verdict\": \"%s\"\n}\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
        if (out) { FILE *f = fopen(out, "w"); if (f) { fputs(g_rcpt, f); fclose(f); printf("receipt: %s\n", out); } else CHECK(0, "cannot write %s", out); }
    }
    printf("gpu_tern_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
