/* gpu_fp8_api_test: OWN-30B Cut 6 / Cut 5 gate for omega_gpu_fp8_api (FP8 e4m3 block-scaled GEMV).
 *
 *   ./gpu_fp8_api_test --host-only          exhaustive 256-code e4m3 conversion vs the reference decode
 *                                           (subnormals, both zeros, NaN codes refused), pack refusals,
 *                                           codegen for every kernel shape (instruction and register budget),
 *                                           API argument refusals, nvdisasm listing when /usr/local/cuda/bin/nvdisasm exists
 *   ./gpu_fp8_api_test --sim                the kernel through tests/bw_warp_sim.h (no chip): one-hot conversion
 *                                           of every code through the kernel, parity battery vs the double oracle
 *                                           (both fragment orders bit-identical), grouped experts, and the four
 *                                           mutants (1 scale dropped, 2 scale block frozen, 3 group table swapped,
 *                                           host no-fold) which the same check must catch
 *   ./gpu_fp8_api_test --out receipt.json   chip gate: parity battery, mutants, grouped experts, then the
 *                                           bandwidth table (median of --reps calls) at the expert and attention shapes
 *   knobs: --lane-bytes N --rows N --warps N --cta-budget N --reps N --spin-us N
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
#include "bw_warp_sim.h"
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

/* ------------------------------------------------------------- host-only */
static int nvdisasm_check_v(int variant, const OmegaFp8GemvSpec *spec, const char *name);
static int nvdisasm_check(const OmegaFp8GemvSpec *spec, const char *name) { return nvdisasm_check_v(1, spec, name); }
static int nvdisasm_check_v(int variant, const OmegaFp8GemvSpec *spec, const char *name) {
    const char *nv = "/usr/local/cuda/bin/nvdisasm";
    if (access(nv, X_OK) != 0) { printf("nvdisasm: not present, listing check skipped for %s\n", name); return 0; }
    OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
    if ((variant == 2 ? omega_blackwell_codegen_fp8_gemv2(spec, &k) : omega_blackwell_codegen_fp8_gemv(spec, &k)) != 0) return -1;
    char bin[96], lst[96], cmd[320];
    snprintf(bin, sizeof bin, "/tmp/omega_fp8_%s_%d.bin", name, (int)getpid());
    snprintf(lst, sizeof lst, "/tmp/omega_fp8_%s_%d.sass", name, (int)getpid());
    FILE *f = fopen(bin, "wb"); if (!f) { omega_blackwell_kernel_free(&k); return -1; }
    fwrite(k.code, 1, k.code_size, f); fclose(f);
    snprintf(cmd, sizeof cmd, "%s -b SM121 %s > %s 2>&1", nv, bin, lst);
    int rc = system(cmd);
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
    for (int i = 0; i < 8; i++) if (!seen[i]) { printf("  %s: nvdisasm listing lacks %s\n", name, must[i]); miss++; }
    printf("nvdisasm %s: rc=%d insns=%d undecodable=%d missing=%d (%s)\n", name, rc, insns, bad, miss, lst);
    unlink(bin);
    return (rc == 0 && bad == 0 && miss == 0) ? 0 : -1;
}

static void host_only(void) {
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
    static const uint32_t ks[] = { 128, 256, 768, 2048, 4096, 16384 };
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
      bad.lane_bytes = 16; bad.k = 32768; CHECK(omega_fp8_gemv_prog_stats(&bad, NULL, NULL) != 0, "k past the limit refused"); }
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
      CHECK(omega_gpu_fp8_tensor_upload(1, 128, 32768, w, s, &t) == OMEGA_GPU_FP8_TOO_LARGE, "k past the limit refused");
      w[77] = 0xff; CHECK(omega_gpu_fp8_tensor_upload(1, 128, 128, w, s, &t) == OMEGA_GPU_FP8_NAN_WEIGHT && t == NULL, "NaN weight refused before any device work");
      CHECK(strstr(omega_gpu_fp8_last_error(), "byte 77") != NULL, "error names the byte (%s)", omega_gpu_fp8_last_error());
      float y[8]; CHECK(omega_gpu_fp8_gemv_f32(NULL, 0, y, y, NULL) == OMEGA_GPU_FP8_BAD_ARGS, "NULL tensor refused"); }

    /* 5. nvdisasm listing of the production shapes */
    { OmegaFp8GemvSpec a = { .k = 2048, .lane_bytes = 16, .rows = 8, .warps = 8 }, b = { .k = 768, .lane_bytes = 8, .rows = 2, .warps = 8 };
      CHECK(nvdisasm_check(&a, "k2048_lb16_r8") == 0, "nvdisasm listing k=2048");
      OmegaFp8GemvSpec d = { .k = 2048, .lane_bytes = 16, .rows = 1, .warps = 8, .depth = 4 };
      CHECK(nvdisasm_check(&d, "k2048_lb16_r1_deep4") == 0, "nvdisasm listing deep k=2048");
      CHECK(nvdisasm_check(&b, "k768_lb8_r2") == 0, "nvdisasm listing k=768"); }

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
        CHECK(nvdisasm_check_v(2, &v, "v2_k2048_lb16_r1_w16") == 0, "nvdisasm listing v2 k=2048");
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
    for (int order = 0; order < 2 && !err; order++) {
        WsSim s; memset(&s, 0, sizeof s);
        s.p = &prog; s.threads = 32 * spec->warps; s.order = order;
        static uint8_t smem[OMEGA_FP8_MAX_K * 4]; s.smem = smem; s.smem_bytes = g_sim_variant == 2 ? omega_fp8_gemv2_shared_bytes(spec) : omega_fp8_gemv_shared_bytes(spec); /* the bytes the launch declares (0: any shared access fails) */
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
        if (err) { snprintf(msg, msglen, "order %d: %s", order, s.msg); break; }
        g_sim_issues += s.warp_issues;
        size_t unwritten = 0; for (size_t i = 0; i < (size_t)groups * n; i++) if (yd[i] == -999.0f) unwritten++;
        if (unwritten) { err = -1; snprintf(msg, msglen, "%zu output words never written", unwritten); break; }
        if (order == 0) memcpy(copy, yd, (size_t)groups * n * 4);
        else if (memcmp(copy, yd, (size_t)groups * n * 4) != 0) { err = -1; snprintf(msg, msglen, "the fragment order changed the result"); }
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
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"device_opens\":%u,\"verdict\":\"%s\"}\n", g_checks, g_failed, omega_gpu_session_open_count(), g_failed == 0 ? "OMEGA_GPU_FP8_GEMV_PASS" : "OMEGA_GPU_FP8_GEMV_FAIL"); fclose(out); }
    return 0;
}

int main(int argc, char **argv) {
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
