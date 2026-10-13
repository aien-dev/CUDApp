/* gpu_moe_window_test: OWN-30B cut 2 gate for omega_gpu_moe_api (column windows over a resident tensor).
 *
 *   ./gpu_moe_window_test --host-only            argument refusals (alignment rule fail-closed) and the packed
 *                                                window byte rule (a window's bytes in the packed tensor equal
 *                                                the pack of the sub-matrix; one tile of skew differs); no chip
 *   ./gpu_moe_window_test --out receipt.json     chip gate: for every case the window result equals the
 *                                                whole-tensor (omega_gpu_matmul_resident_*) columns bit for bit
 *                                                and the double oracle; repeat call is a cache hit and identical;
 *                                                the multi-window call equals the single windows; negative
 *                                                controls (wrong window by one tile, wrong kernel) are caught;
 *                                                timing of 8 expert windows per call (the Stage A decode shape)
 *   ./gpu_moe_window_test --cta-budget N --spin-us N --calls N --small   (knobs; --small skips the 128-expert tensors)
 */
#include "omega_gpu_moe_api.h"
#include "omega_gpu_session.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_matmul.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static void fill(float *a, size_t n, uint32_t *seed) { for (size_t i = 0; i < n; i++) a[i] = ((float)(lcg(seed) % 2001) - 1000.0f) / 1000.0f; }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static int cmp_double(const void *x, const void *y) { double a = *(const double *)x, b = *(const double *)y; return a < b ? -1 : a > b; }

/* double oracle of the window: error relative to the accumulation scale, count at or beyond 1e-5 */
static double window_oracle(uint32_t m, uint32_t k, uint32_t n, const uint16_t *a, const uint16_t *b, uint32_t col0, uint32_t cols, const float *c, size_t *bad) {
    double worst = 0.0; size_t nbad = 0;
    for (size_t i = 0; i < m; i++) for (size_t j = 0; j < cols; j++) {
        double acc = 0.0, scale = 0.0;
        for (size_t t = 0; t < k; t++) { double p = (double)bf2f(a[i * k + t]) * (double)bf2f(b[t * n + col0 + j]); acc += p; scale += fabs(p); }
        double rel = fabs(acc - (double)c[i * cols + j]) / (scale > 1e-6 ? scale : 1e-6);
        if (rel > worst) worst = rel;
        if (rel >= 1e-5) nbad++;
    }
    if (bad) *bad = nbad;
    return worst;
}

/* ------------------------------------------------------------------ host only */
static void host_only(void) {
    /* the packed window byte rule: for a tile-aligned col0, the bytes of the packed tensor from
     * omega_gpu_moe_window_byte_offset(kp, col0) are the pack of B[:, col0 .. col0 + wnp); moving one
     * tile (the wrong-window control) changes them */
    struct { uint32_t k, n, col0, cols; } sh[] = { { 48, 1000, 8, 13 }, { 48, 1000, 792, 200 }, { 2048, 6144, 1536, 1536 }, { 768, 8192, 6144, 2048 }, { 40, 24, 16, 8 } };
    for (size_t s = 0; s < sizeof sh / sizeof sh[0]; s++) {
        const uint32_t k = sh[s].k, n = sh[s].n, col0 = sh[s].col0, cols = sh[s].cols;
        const uint32_t kp = (k + 15) / 16 * 16, np = (n + 7) / 8 * 8, NT = np / 8, wnp = (cols + 7) / 8 * 8, WT = wnp / 8;
        uint16_t *b = malloc((size_t)k * n * 2), *full = malloc((size_t)kp * np * 2), *win = malloc((size_t)kp * wnp * 2), *sub = malloc((size_t)k * wnp * 2);
        uint32_t seed = 11 + (uint32_t)s;
        for (size_t i = 0; i < (size_t)k * n; i++) b[i] = (uint16_t)(1 + lcg(&seed) % 60000);
        omega_matmul_frag_pack_b(b, k, n, kp, 0, NT, full);
        /* the sub-matrix B[:, col0 .. col0 + real) where real = min(wnp, n - col0): past the real columns the
         * tensor's padding is zero, which the sub-matrix pack reproduces by having only `real` columns */
        const uint32_t real = n - col0 < wnp ? n - col0 : wnp;
        for (uint32_t r = 0; r < k; r++) for (uint32_t j = 0; j < real; j++) sub[(size_t)r * real + j] = b[(size_t)r * n + col0 + j];
        omega_matmul_frag_pack_b(sub, k, real, kp, 0, WT, win);
        const uint64_t off = omega_gpu_moe_window_byte_offset(kp, col0);
        CHECK(off == (uint64_t)(col0 / 8) * kp * 8 * 2, "offset arithmetic k=%u col0=%u", k, col0);
        CHECK(memcmp((const uint8_t *)full + off, win, (size_t)kp * wnp * 2) == 0, "window bytes k=%u n=%u col0=%u cols=%u equal the sub-matrix pack", k, n, col0, cols);
        /* wrong window: one tile of skew (either way) must differ */
        int differs = 1;
        if (col0 / 8 + 1 + WT <= NT) differs = memcmp((const uint8_t *)full + off + (size_t)kp * 8 * 2, win, (size_t)kp * wnp * 2) != 0;
        else differs = memcmp((const uint8_t *)full + off - (size_t)kp * 8 * 2, win, (size_t)kp * wnp * 2) != 0;
        CHECK(differs, "window bytes k=%u col0=%u: one tile of skew differs (wrong-window control)", k, col0);
        free(b); free(full); free(win); free(sub);
    }
    /* refusals: nothing below opens the device (the alignment rule is checked before the lock) */
    float a[64] = { 0 }, c[64];
    uint32_t col0s[2] = { 0, 8 };
    CHECK(omega_gpu_matmul_resident_window_f32(1, NULL, NULL, 0, 8, c, NULL) == OMEGA_GPU_MATMUL_BAD_ARGS, "NULL a / tensor refused");
    CHECK(omega_gpu_matmul_resident_window_f32(0, a, NULL, 0, 8, c, NULL) == OMEGA_GPU_MATMUL_BAD_ARGS, "m = 0 refused");
    CHECK(omega_gpu_matmul_resident_windows_f32(1, a, NULL, 0, col0s, 8, c, NULL) == OMEGA_GPU_MATMUL_BAD_ARGS, "nwin = 0 refused");
    CHECK(omega_gpu_moe_tensor_upload_bf16(0, 8, (const uint16_t *)a, NULL) == OMEGA_GPU_MATMUL_BAD_ARGS, "upload k = 0 / NULL out refused");
    OmegaGpuMoeTensor *t = NULL;
    CHECK(omega_gpu_moe_tensor_upload_bf16(16, OMEGA_BW_MATMUL_MAX_N + 8, (const uint16_t *)a, &t) == OMEGA_GPU_MATMUL_TOO_LARGE && t == NULL, "upload n past the envelope refused");
    CHECK(omega_gpu_moe_reserve(0, 16, 16, 0) == OMEGA_GPU_MATMUL_BAD_ARGS, "reserve zero bound refused");
    CHECK(omega_gpu_moe_prepare(1, 16, OMEGA_BW_MATMUL_MAX_N + 8) == OMEGA_GPU_MATMUL_TOO_LARGE, "prepare past the envelope refused");
    uint32_t kk, nn; omega_gpu_moe_tensor_shape(NULL, &kk, &nn);
    CHECK(kk == 0 && nn == 0, "shape of NULL is 0 x 0");
    omega_gpu_moe_tensor_free(NULL);
    printf("host-only: alignment refusals that need a tensor (col0 %% 8, window past n) run in the chip gate\n");
}

/* ------------------------------------------------------------------ chip */
typedef struct { uint32_t k, n, m, cols, nwin; const char *name; int big; } Case;

static int chip(const char *out_path, int calls, int small) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_MOE_WINDOW_V1\",\"cta_budget\":%u,\"spin_us\":%u,\"calls\":%d,\"cases\":[",
                     omega_gpu_moe_cta_budget(), omega_gpu_session_spin_us(), calls);
    const Case cases[] = {
        { 48, 1000, 1, 13, 3, "small_m1_unaligned_cols", 0 },
        { 48, 1000, 5, 200, 2, "small_m5", 0 },
        { 2048, 1536 * 16, 1, 1536, 8, "gate_up_16_experts", 0 },
        { 2048, 1536 * 128, 1, 1536, 8, "gate_up_128_experts", 1 },
        { 768, 2048 * 128, 1, 2048, 8, "down_128_experts", 1 },
        { 2048, 1536 * 16, 16, 1536, 8, "gate_up_16_experts_m16", 0 },
        /* one variable: the same 1536-wide window on a standalone one-expert tensor (one spare tile so the
         * skew control stays inside the allocation); compares against the window inside the 16/128-expert tensors */
        { 2048, 1536 + 8, 1, 1536, 1, "gate_up_1_expert_standalone", 0 },
        { 768, 2048 + 8, 1, 2048, 1, "down_1_expert_standalone", 0 },
    };
    int first = 1;
    for (size_t ci = 0; ci < sizeof cases / sizeof cases[0]; ci++) {
        const Case *cs = &cases[ci];
        if (small && cs->big) continue;
        const uint32_t k = cs->k, n = cs->n, m = cs->m, cols = cs->cols, nwin = cs->nwin;
        const uint32_t kp = (k + 15) / 16 * 16, wnp = (cols + 7) / 8 * 8;
        printf("== %s: k=%u n=%u m=%u cols=%u windows=%u\n", cs->name, k, n, m, cols, nwin);
        size_t na = (size_t)m * k, nb = (size_t)k * n, nc = (size_t)m * n, nw = (size_t)m * cols;
        float *a = malloc(na * 4), *whole = malloc(nc * 4), *cw = malloc(nw * 4), *cw2 = malloc(nw * 4), *multi = malloc(nw * 4 * nwin);
        uint16_t *ah = malloc(na * 2), *bh = malloc(nb * 2);
        if (!a || !whole || !cw || !cw2 || !multi || !ah || !bh) { printf("out of host memory\n"); return 2; }
        uint32_t seed = 0x30b0000u ^ (k << 8) ^ n ^ m;
        fill(a, na, &seed);
        for (size_t i = 0; i < na; i++) ah[i] = omega_fp32_to_bf16(a[i]);
        { float *bt = malloc((size_t)1 << 20); size_t done = 0; /* bf16 B in 256K-element chunks, no k x n f32 copy */
          while (done < nb) { size_t c = nb - done < (1u << 18) ? nb - done : (1u << 18); fill(bt, c, &seed); for (size_t i = 0; i < c; i++) bh[done + i] = omega_fp32_to_bf16(bt[i]); done += c; }
          free(bt); }
        /* the 8 windows: distinct tile-aligned expert slots, never the last tile (the skew control moves one tile up) */
        uint32_t col0[8]; const uint32_t slots = n / cols;
        for (uint32_t w = 0; w < nwin; w++) { uint32_t s; int dup; do { s = lcg(&seed) % (slots > 1 ? slots - 1 : 1); dup = 0; for (uint32_t v = 0; v < w; v++) if (col0[v] == s * cols) dup = 1; } while (dup); col0[w] = s * cols; }
        if (cs->cols == 13) { col0[0] = 8; col0[1] = 16; col0[2] = 976; } /* 976 + 13 = 989 < 1000, inside the real columns */
        if (cs->cols == 200) { col0[0] = 792; col0[1] = 0; }           /* 792 + 200 = 992 */

        OmegaGpuTensor *tc = NULL; OmegaGpuMoeTensor *tm = NULL;
        double t0 = now_ms();
        int rc1 = omega_gpu_tensor_upload_bf16(k, n, bh, &tc);
        double up_classic = now_ms() - t0; t0 = now_ms();
        int rc2 = omega_gpu_moe_tensor_upload_bf16(k, n, bh, &tm);
        double up_moe = now_ms() - t0;
        CHECK(rc1 == OMEGA_GPU_MATMUL_OK && rc2 == OMEGA_GPU_MATMUL_OK, "%s: uploads (classic %s, moe %s)", cs->name, omega_gpu_matmul_rc_name(rc1), omega_gpu_matmul_rc_name(rc2));
        printf("  upload classic %.0f ms, moe %.0f ms (%zu MB bf16)\n", up_classic, up_moe, nb * 2 / (1u << 20));
        if (rc1 != OMEGA_GPU_MATMUL_OK || rc2 != OMEGA_GPU_MATMUL_OK) break;

        /* alignment rule, fail-closed, with the tensor present */
        OmegaGpuMatmulInfo info;
        int rr = omega_gpu_matmul_resident_window_bf16(m, ah, tm, 4, 8, cw, &info);
        CHECK(rr == OMEGA_GPU_MATMUL_BAD_ARGS && strstr(omega_gpu_matmul_last_error(), "not a multiple of 8"), "%s: col0 = 4 refused (%s: %s)", cs->name, omega_gpu_matmul_rc_name(rr), omega_gpu_matmul_last_error());
        rr = omega_gpu_matmul_resident_window_bf16(m, ah, tm, (n - 8) / 8 * 8, 16, cw, &info);
        CHECK(rr == OMEGA_GPU_MATMUL_BAD_ARGS && strstr(omega_gpu_matmul_last_error(), "not inside"), "%s: window past n refused (%s: %s)", cs->name, omega_gpu_matmul_rc_name(rr), omega_gpu_matmul_last_error());
        rr = omega_gpu_matmul_resident_window_bf16(m, ah, tm, 0, 0, cw, &info);
        CHECK(rr == OMEGA_GPU_MATMUL_BAD_ARGS, "%s: cols = 0 refused (%s)", cs->name, omega_gpu_matmul_rc_name(rr));

        /* whole-tensor product through the existing API (the reference the window must match column for column) */
        OmegaGpuMatmulInfo wi;
        int wrc = omega_gpu_matmul_resident_bf16(m, ah, tc, whole, &wi);
        CHECK(wrc == OMEGA_GPU_MATMUL_OK, "%s: whole-tensor call rc=%s (%s)", cs->name, omega_gpu_matmul_rc_name(wrc), omega_gpu_matmul_last_error());
        if (wrc != OMEGA_GPU_MATMUL_OK) break;
        printf("  whole: %.3f ms call, chip %.3f ms, grid %ux%u\n", wi.call_ns / 1e6, wi.elapsed_ns / 1e6, wi.grid_x, wi.grid_y);

        size_t mism_total = 0, oracle_bad_total = 0; double oracle_worst = 0.0; int all_hit = 1, f32_same = 1; double win_chip_ms = 0.0;
        for (uint32_t w = 0; w < nwin; w++) {
            OmegaGpuMatmulInfo i1, i2, i3;
            int r1 = omega_gpu_matmul_resident_window_bf16(m, ah, tm, col0[w], cols, cw, &i1);
            CHECK(r1 == OMEGA_GPU_MATMUL_OK && i1.n == cols && i1.padded_n == wnp && i1.resident, "%s: window %u (col0 %u) rc=%s (%s)", cs->name, w, col0[w], omega_gpu_matmul_rc_name(r1), omega_gpu_matmul_last_error());
            if (r1 != OMEGA_GPU_MATMUL_OK) break;
            size_t mism = 0;
            for (uint32_t r = 0; r < m; r++) for (uint32_t j = 0; j < cols; j++) if (cw[(size_t)r * cols + j] != whole[(size_t)r * n + col0[w] + j]) mism++;
            mism_total += mism;
            size_t bad = 0; double worst = window_oracle(m, k, n, ah, bh, col0[w], cols, cw, &bad);
            oracle_bad_total += bad; if (worst > oracle_worst) oracle_worst = worst;
            /* repeat: cache hit, bit-identical; f32 activation path: identical too (same bf16 rounding) */
            int r2 = omega_gpu_matmul_resident_window_bf16(m, ah, tm, col0[w], cols, cw2, &i2);
            int r3 = omega_gpu_matmul_resident_window_f32(m, a, tm, col0[w], cols, multi, &i3);
            if (!(r2 == OMEGA_GPU_MATMUL_OK && i2.kernel_cache_hit && memcmp(cw, cw2, nw * 4) == 0)) all_hit = 0;
            if (!(r3 == OMEGA_GPU_MATMUL_OK && memcmp(cw, multi, nw * 4) == 0)) f32_same = 0;
            win_chip_ms += i1.elapsed_ns / 1e6;
            printf("  window %u col0=%u: mismatches vs whole %zu/%zu, oracle worst %.2e bad %zu, chip %.3f ms, grid %ux%u, hit=%d\n",
                   w, col0[w], mism, nw, worst, bad, i1.elapsed_ns / 1e6, i1.grid_x, i1.grid_y, (int)i2.kernel_cache_hit);
            if (omega_gpu_matmul_is_blocked()) break;
        }
        CHECK(mism_total == 0, "%s: every window equals the whole-tensor columns bit for bit (%zu mismatches)", cs->name, mism_total);
        CHECK(oracle_bad_total == 0, "%s: every window within 1e-5 of the double oracle (worst %g, %zu bad)", cs->name, oracle_worst, oracle_bad_total);
        CHECK(all_hit, "%s: repeated window calls hit the kernel cache and are bit-identical", cs->name);
        CHECK(f32_same, "%s: f32 activation path is bit-identical to the bf16 one", cs->name);
        if (omega_gpu_matmul_is_blocked()) break;

        /* the multi-window call (one activation pack, nwin launches) equals the single windows */
        OmegaGpuMatmulInfo mi;
        int mrc = omega_gpu_matmul_resident_windows_f32(m, a, tm, nwin, col0, cols, multi, &mi);
        size_t mm = 0;
        if (mrc == OMEGA_GPU_MATMUL_OK) for (uint32_t w = 0; w < nwin; w++) for (uint32_t r = 0; r < m; r++) for (uint32_t j = 0; j < cols; j++)
            if (multi[(size_t)w * nw + (size_t)r * cols + j] != whole[(size_t)r * n + col0[w] + j]) mm++;
        CHECK(mrc == OMEGA_GPU_MATMUL_OK && mi.chip_calls == nwin && mm == 0, "%s: multi-window call rc=%s launches=%u mismatches=%zu", cs->name, omega_gpu_matmul_rc_name(mrc), mi.chip_calls, mm);

        /* negative control 1: the wrong window (one tile up) must disagree with the whole-tensor columns */
        omega_gpu_moe_test_set_window_skew(1);
        int src = omega_gpu_matmul_resident_window_bf16(m, ah, tm, col0[0], cols, cw, &info);
        size_t skew_mism = 0;
        for (uint32_t r = 0; r < m; r++) for (uint32_t j = 0; j < cols; j++) if (cw[(size_t)r * cols + j] != whole[(size_t)r * n + col0[0] + j]) skew_mism++;
        omega_gpu_moe_test_set_window_skew(0);
        CHECK(src == OMEGA_GPU_MATMUL_OK && skew_mism > nw / 2, "%s: wrong-window control caught (rc=%s, %zu/%zu differ)", cs->name, omega_gpu_matmul_rc_name(src), skew_mism, nw);
        /* negative control 2: the wrong kernel (last K pass dropped) must fail the oracle */
        omega_gpu_moe_test_set_mutant(1);
        int mut = omega_gpu_matmul_resident_window_bf16(m, ah, tm, col0[0], cols, cw, &info);
        size_t mut_bad = 0; double mut_worst = mut == OMEGA_GPU_MATMUL_OK ? window_oracle(m, k, n, ah, bh, col0[0], cols, cw, &mut_bad) : 0.0;
        omega_gpu_moe_test_set_mutant(0);
        CHECK(mut == OMEGA_GPU_MATMUL_OK && mut_bad > 0, "%s: wrong-kernel control caught (rc=%s, worst %g, %zu bad)", cs->name, omega_gpu_matmul_rc_name(mut), mut_worst, mut_bad);
        if (omega_gpu_matmul_is_blocked()) break;

        /* timing: `calls` multi-window calls (the per-token expert pattern) and `calls` single-window calls */
        double *sm = malloc((size_t)calls * sizeof *sm), *ss = malloc((size_t)calls * sizeof *ss);
        uint64_t chip_multi = 0, chip_single = 0; int ok_m = 0, ok_s = 0; size_t rep_mism = 0;
        for (int i = 0; i < calls; i++) {
            double q0 = now_ms();
            int r = omega_gpu_matmul_resident_windows_f32(m, a, tm, nwin, col0, cols, multi, &mi);
            sm[i] = now_ms() - q0;
            if (r == OMEGA_GPU_MATMUL_OK) { ok_m++; chip_multi += mi.elapsed_ns; }
            if (omega_gpu_matmul_is_blocked()) break;
        }
        for (uint32_t w = 0; w < nwin; w++) for (uint32_t r = 0; r < m; r++) for (uint32_t j = 0; j < cols; j++)
            if (multi[(size_t)w * nw + (size_t)r * cols + j] != whole[(size_t)r * n + col0[w] + j]) rep_mism++;
        for (int i = 0; i < calls && !omega_gpu_matmul_is_blocked(); i++) {
            double q0 = now_ms();
            int r = omega_gpu_matmul_resident_window_f32(m, a, tm, col0[i % nwin], cols, cw, &info);
            ss[i] = now_ms() - q0;
            if (r == OMEGA_GPU_MATMUL_OK) { ok_s++; chip_single += info.elapsed_ns; }
        }
        qsort(sm, (size_t)calls, sizeof *sm, cmp_double); qsort(ss, (size_t)calls, sizeof *ss, cmp_double);
        const double med_m = sm[calls / 2], med_s = ss[calls / 2];
        const double bytes_win = (double)kp * wnp * 2.0;
        const double gbs_multi = bytes_win * nwin / (med_m * 1e-3) / 1e9, gbs_single = bytes_win / (med_s * 1e-3) / 1e9;
        CHECK(ok_m == calls && ok_s == calls && rep_mism == 0, "%s: timing calls ok %d/%d multi, %d/%d single, last multi result mismatches %zu", cs->name, ok_m, calls, ok_s, calls, rep_mism);
        printf("  timing: %u windows/call median %.3f ms (%.1f us/window, chip mean %.1f us/window, %.1f GB/s); single window median %.3f ms (%.1f GB/s, chip mean %.1f us); %.2f MB per window\n",
               nwin, med_m, med_m * 1e3 / nwin, ok_m ? chip_multi / 1e3 / ok_m / nwin : 0.0, gbs_multi, med_s, gbs_single, ok_s ? chip_single / 1e3 / ok_s : 0.0, bytes_win / 1e6);
        if (out) {
            fprintf(out, "%s{\"case\":\"%s\",\"k\":%u,\"n\":%u,\"m\":%u,\"cols\":%u,\"padded_cols\":%u,\"windows\":%u,\"col0\":[", first ? "" : ",", cs->name, k, n, m, cols, wnp, nwin);
            for (uint32_t w = 0; w < nwin; w++) fprintf(out, "%s%u", w ? "," : "", col0[w]);
            fprintf(out, "],\"upload_classic_ms\":%.1f,\"upload_moe_ms\":%.1f,\"whole_call_ms\":%.4f,\"whole_chip_ms\":%.4f,"
                    "\"window_mismatches_vs_whole\":%zu,\"oracle_worst_rel\":%g,\"oracle_bad\":%zu,\"repeat_cache_hit_identical\":%s,\"f32_path_identical\":%s,"
                    "\"multi_rc\":\"%s\",\"multi_mismatches\":%zu,\"skew_control_differ\":%zu,\"skew_control_elems\":%zu,\"mutant_control_bad\":%zu,\"mutant_control_worst\":%g,"
                    "\"calls\":%d,\"multi_median_ms\":%.4f,\"multi_p90_ms\":%.4f,\"multi_min_ms\":%.4f,\"multi_chip_us_per_window\":%.2f,\"multi_gbs\":%.1f,"
                    "\"single_median_ms\":%.4f,\"single_p90_ms\":%.4f,\"single_min_ms\":%.4f,\"single_chip_us\":%.2f,\"single_gbs\":%.1f,\"bytes_per_window\":%.0f,\"grid_x\":%u,\"grid_y\":%u}",
                    up_classic, up_moe, wi.call_ns / 1e6, wi.elapsed_ns / 1e6,
                    mism_total, oracle_worst, oracle_bad_total, all_hit ? "true" : "false", f32_same ? "true" : "false",
                    omega_gpu_matmul_rc_name(mrc), mm, skew_mism, nw, mut_bad, mut_worst,
                    calls, med_m, sm[calls * 9 / 10], sm[0], ok_m ? chip_multi / 1e3 / ok_m / nwin : 0.0, gbs_multi,
                    med_s, ss[calls * 9 / 10], ss[0], ok_s ? chip_single / 1e3 / ok_s : 0.0, gbs_single, bytes_win, info.grid_x, info.grid_y);
            first = 0;
        }
        free(sm); free(ss);
        omega_gpu_tensor_free(tc); omega_gpu_moe_tensor_free(tm);
        free(a); free(whole); free(cw); free(cw2); free(multi); free(ah); free(bh);
        if (omega_gpu_matmul_is_blocked()) break;
    }
    if (out) {
        fprintf(out, "],\"blocked\":%s,\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", omega_gpu_matmul_is_blocked() ? "true" : "false", g_checks, g_failed,
                g_failed == 0 ? "OMEGA_GPU_MOE_WINDOW_PASS" : "OMEGA_GPU_MOE_WINDOW_FAIL");
        fclose(out);
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *out_path = NULL; int host = 0, calls = 100, small = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--small") == 0) small = 1;
        else if (strcmp(argv[i], "--calls") == 0 && i + 1 < argc) calls = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cta-budget") == 0 && i + 1 < argc) { uint32_t b = (uint32_t)strtoul(argv[++i], NULL, 10); omega_gpu_moe_set_cta_budget(b); omega_gpu_matmul_set_cta_budget(b); }
        else if (strcmp(argv[i], "--spin-us") == 0 && i + 1 < argc) { if (omega_gpu_session_set_spin_us((uint32_t)strtoul(argv[++i], NULL, 10)) != 0) { fprintf(stderr, "--spin-us out of range\n"); return 2; } }
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    }
    if (calls < 2) calls = 2;
    int rc = 0;
    if (host) host_only();
    else rc = chip(out_path, calls, small);
    if (!omega_gpu_matmul_is_blocked()) omega_gpu_device_close();
    if (rc) return 2;
    printf("%s: %d checks, %d failed\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
