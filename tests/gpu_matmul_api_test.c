/* gpu_matmul_api_test: FB-1 cut 1 / 1b gate for omega_gpu_matmul_api.
 *
 *   ./gpu_matmul_api_test --host-only                 argument and limit refusals, codegen only, no chip
 *   ./gpu_matmul_api_test --out receipt.json          chip sweep over shapes, JSON receipt
 *   ./gpu_matmul_api_test --timing --out r.json       resident GEMV timing (median of 100 calls) + parity
 *   ./gpu_matmul_api_test --mutant --out r.json       wrong kernel (last K step dropped) must FAIL parity;
 *                                                     exit 0 when every shape caught it ("mutant killed")
 *   ./gpu_matmul_api_test --cta-budget N ...          run with a different CTA budget (I42 experiment)
 *
 * Shapes come from OMEGA_GPU_MATMUL_SHAPES ("m,k,n;m,k,n;..."), default below.
 * Pass rule per shape: rc OK, parity verified by the API oracle, end-to-end error < 1e-5 relative to
 * the accumulation scale (bf16 inputs against a double reference; only accumulation order and the
 * chip's f32 accumulator differ), second call hits the kernel cache and agrees bit for bit.
 */
#include "omega_gpu_matmul_api.h"
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

static void host_only(void) {
    float a[16 * 16] = {0}, b[16 * 16] = {0}, c[16 * 16] = {0};
    OmegaGpuMatmulInfo info;
    CHECK(omega_gpu_matmul_f32(0, 16, 16, a, b, c, &info) == OMEGA_GPU_MATMUL_BAD_ARGS, "m=0 refused");
    CHECK(omega_gpu_matmul_f32(16, 16, 16, NULL, b, c, &info) == OMEGA_GPU_MATMUL_BAD_ARGS, "NULL a refused");
    CHECK(omega_gpu_matmul_f32(OMEGA_BW_MATMUL_MAX_M + 1, 16, 16, a, b, c, &info) == OMEGA_GPU_MATMUL_TOO_LARGE, "m over limit refused");
    CHECK(omega_gpu_matmul_bf16(16, OMEGA_BW_MATMUL_MAX_K + 1, 16, (uint16_t *)a, (uint16_t *)b, c, NULL) == OMEGA_GPU_MATMUL_TOO_LARGE, "k over limit refused (NULL info ok)");
    CHECK(omega_gpu_matmul_bf16(16, 16, OMEGA_BW_MATMUL_MAX_N + 1, (uint16_t *)a, (uint16_t *)b, c, NULL) == OMEGA_GPU_MATMUL_TOO_LARGE, "n over limit refused");
    CHECK(OMEGA_BW_MATMUL_MAX_N >= 128256 && OMEGA_BW_MATMUL_MAX_K >= 8192 && OMEGA_BW_MATMUL_MAX_M >= 2048, "limits cover Llama shapes (vocab 128256, ffn 8192)");
    OmegaGpuTensor *t = NULL;
    CHECK(omega_gpu_tensor_upload_f32(0, 16, a, &t) == OMEGA_GPU_MATMUL_BAD_ARGS && t == NULL, "tensor k=0 refused");
    CHECK(omega_gpu_tensor_upload_f32(16, 16, NULL, &t) == OMEGA_GPU_MATMUL_BAD_ARGS, "tensor NULL refused");
    CHECK(omega_gpu_matmul_resident_f32(16, a, NULL, c, &info) == OMEGA_GPU_MATMUL_BAD_ARGS, "resident NULL tensor refused");
    CHECK(strcmp(omega_gpu_matmul_rc_name(OMEGA_GPU_MATMUL_PARITY_FAIL), "PARITY_FAIL") == 0, "rc name");
    CHECK(omega_gpu_matmul_cta_budget() == OMEGA_GPU_MATMUL_MAX_CTAS, "default CTA budget");
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
    omega_gpu_matmul_cache_clear();
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
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_MATMUL_API_TIMING_V1\",\"cta_budget\":%u,\"calls\":100,\"shapes\":[", omega_gpu_matmul_cta_budget());
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
        int pass = ok_calls == 100 && bad_first == 0 && bad_last == 0 && repeat_mismatch == 0 && median < 5.0;
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

int main(int argc, char **argv) {
    const char *out_path = NULL;
    int host = 0, mutant = 0, do_timing = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--mutant") == 0) mutant = 1;
        else if (strcmp(argv[i], "--timing") == 0) do_timing = 1;
        else if (strcmp(argv[i], "--cta-budget") == 0 && i + 1 < argc) omega_gpu_matmul_set_cta_budget((uint32_t)strtoul(argv[++i], NULL, 10));
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    }
    const char *shapes = getenv("OMEGA_GPU_MATMUL_SHAPES");
    if (shapes && !*shapes) shapes = NULL;
    int rc = 0;
    if (host) host_only();
    else if (do_timing) rc = timing(shapes ? shapes : TIMING_SHAPES, out_path);
    else if (mutant) { omega_gpu_matmul_test_set_mutant(1); rc = chip_sweep(shapes ? shapes : MUTANT_SHAPES, out_path, 1); omega_gpu_matmul_test_set_mutant(0); }
    else rc = chip_sweep(shapes ? shapes : DEFAULT_SHAPES, out_path, 0);
    if (!omega_gpu_matmul_is_blocked()) omega_gpu_device_close();
    if (rc) return 2;
    printf("%s: %d checks, %d failed%s\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed, mutant ? (g_failed ? " (mutant SURVIVED)" : " (mutant killed)") : "");
    return g_failed ? 1 : 0;
}
