/* gpu_matmul_api_test: FB-1 cut 1 gate for omega_gpu_matmul_api.
 *
 *   ./gpu_matmul_api_test --host-only            argument and limit refusals, codegen only, no chip
 *   ./gpu_matmul_api_test --out receipt.json     chip sweep over shapes, JSON receipt
 *
 * Shapes come from OMEGA_GPU_MATMUL_SHAPES ("m,k,n;m,k,n;..."), default below.
 * Pass rule per shape: rc OK, parity verified, end-to-end error < 1e-5 relative to the accumulation scale (bf16 inputs
 * against the host bf16 oracle, f32 accumulate; only summation order differs).
 */
#include "omega_gpu_matmul_api.h"
#include "omega_blackwell_matmul.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* round a float to bf16 and back, as the API does on the way in */
static float bf16r(float f) { uint16_t h = omega_fp32_to_bf16(f); uint32_t u = (uint32_t)h << 16; float o; memcpy(&o, &u, 4); return o; }

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const char *DEFAULT_SHAPES =
    "16,16,16;32,32,32;64,64,64;128,128,128;256,256,256;512,512,512;"
    "1,1024,1024;1,256,1024;16,1024,256;8,1024,8;"
    "1024,16,128;192,16,192;512,16,64;256,16,256;128,16,512";

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }

static void host_only(void) {
    float a[16 * 16] = {0}, b[16 * 16] = {0}, c[16 * 16] = {0};
    OmegaGpuMatmulInfo info;
    CHECK(omega_gpu_matmul_f32(0, 16, 16, a, b, c, &info) == OMEGA_GPU_MATMUL_BAD_ARGS, "m=0 refused");
    CHECK(omega_gpu_matmul_f32(16, 16, 16, NULL, b, c, &info) == OMEGA_GPU_MATMUL_BAD_ARGS, "NULL a refused");
    CHECK(omega_gpu_matmul_f32(OMEGA_BW_MATMUL_MAX_M + 1, 16, 16, a, b, c, &info) == OMEGA_GPU_MATMUL_TOO_LARGE, "m over limit refused");
    CHECK(omega_gpu_matmul_bf16(16, OMEGA_BW_MATMUL_MAX_K + 1, 16, (uint16_t *)a, (uint16_t *)b, c, NULL) == OMEGA_GPU_MATMUL_TOO_LARGE, "k over limit refused (NULL info ok)");
    CHECK(strcmp(omega_gpu_matmul_rc_name(OMEGA_GPU_MATMUL_PARITY_FAIL), "PARITY_FAIL") == 0, "rc name");
    /* codegen only: shapes must be accepted by our codegen without touching the chip */
    OmegaMatMulSpec spec; OmegaBlackwellKernel kernel; memset(&kernel, 0, sizeof kernel);
    /* documents the kernel rule the API pads around: M%16, N%8, K==16 */
    CHECK(omega_matmul_spec_init(&spec, 16, 16, 8, OMEGA_MATMUL_PRECISION_BF16) == 0 && omega_blackwell_codegen_matmul(&spec, &kernel) == 0 && kernel.code_size > 0, "16x16x8 tile kernel generated");
    omega_blackwell_kernel_free(&kernel); memset(&kernel, 0, sizeof kernel);
    CHECK(omega_matmul_spec_init(&spec, 1, 1024, 1024, OMEGA_MATMUL_PRECISION_BF16) == 0 && omega_blackwell_codegen_matmul(&spec, &kernel) != 0, "raw 1x1024x1024 refused by codegen (the API pads and slices it)");
    omega_gpu_matmul_cache_clear();
}

static int chip_sweep(const char *out_path) {
    const char *shapes = getenv("OMEGA_GPU_MATMUL_SHAPES");
    if (!shapes || !*shapes) shapes = DEFAULT_SHAPES;
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_MATMUL_API_SWEEP_V1\",\"shapes\":[");
    char *copy = strdup(shapes);
    char *save = NULL;
    int first = 1;
    for (char *tok = strtok_r(copy, ";", &save); tok; tok = strtok_r(NULL, ";", &save)) {
        unsigned m, k, n;
        if (sscanf(tok, "%u,%u,%u", &m, &k, &n) != 3) { printf("bad shape %s\n", tok); continue; }
        size_t na = (size_t)m * k, nb = (size_t)k * n, nc = (size_t)m * n;
        float *a = malloc(na * sizeof *a), *b = malloc(nb * sizeof *b), *c = malloc(nc * sizeof *c);
        uint32_t seed = 0x5eed0000u ^ (m << 20) ^ (k << 10) ^ n;
        for (size_t i = 0; i < na; i++) a[i] = ((float)(lcg(&seed) % 2001) - 1000.0f) / 1000.0f;
        for (size_t i = 0; i < nb; i++) b[i] = ((float)(lcg(&seed) % 2001) - 1000.0f) / 1000.0f;
        for (size_t i = 0; i < nc; i++) c[i] = -12345.0f;
        OmegaGpuMatmulInfo info;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int rc = omega_gpu_matmul_f32(m, k, n, a, b, c, &info);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        /* second call on the same shape must hit the kernel cache and agree */
        float *c2 = malloc(nc * sizeof *c2);
        OmegaGpuMatmulInfo info2;
        int rc2 = omega_gpu_matmul_f32(m, k, n, a, b, c2, &info2);
        size_t repeat_mismatch = 0;
        for (size_t i = 0; i < nc; i++) if (c[i] != c2[i]) repeat_mismatch++;
        /* end-to-end: full-K product on the host from the same bf16-rounded inputs, in double */
        double e2e_max_rel = 0.0;
        if (rc == OMEGA_GPU_MATMUL_OK) {
            for (size_t i = 0; i < m; i++) for (size_t j = 0; j < n; j++) {
                double acc = 0.0, scale = 0.0;
                for (size_t t = 0; t < k; t++) {
                    double p = (double)bf16r(a[i * k + t]) * (double)bf16r(b[t * n + j]);
                    acc += p; scale += fabs(p);
                }
                /* error relative to the accumulation scale (sum of |terms|), not to a
                 * result that may cancel to near zero: f32 accumulation over k terms */
                double d = fabs(acc - (double)c[i * n + j]);
                double rel = d / (scale > 1e-6 ? scale : 1e-6);
                if (rel > e2e_max_rel) e2e_max_rel = rel;
            }
        }
        double wall_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
        int pass = rc == OMEGA_GPU_MATMUL_OK && info.parity_verified && e2e_max_rel < 1e-5
                   && rc2 == OMEGA_GPU_MATMUL_OK && info2.kernel_cache_hit && repeat_mismatch == 0;
        CHECK(pass, "shape %ux%ux%u rc=%s parity=%d max_rel=%g e2e_rel=%g rc2=%s hit=%d repeat_mismatch=%zu",
              m, k, n, omega_gpu_matmul_rc_name(rc), info.parity_verified, info.max_rel_err, e2e_max_rel,
              omega_gpu_matmul_rc_name(rc2), info2.kernel_cache_hit, repeat_mismatch);
        printf("%s %ux%ux%u rc=%s chip_ns=%" PRIu64 " wall_ms=%.1f max_abs=%g max_rel=%g mismatches=%zu e2e_rel=%g slices=%u calls=%u rows_per_call=%u cache_hit2=%d\n",
               pass ? "PASS" : "FAIL", m, k, n, omega_gpu_matmul_rc_name(rc), info.elapsed_ns, wall_ms,
               info.max_abs_err, info.max_rel_err, info.mismatch_count, e2e_max_rel, info.k_slices, info.chip_calls, info.rows_per_call, info2.kernel_cache_hit);
        if (out) {
            fprintf(out, "%s{\"m\":%u,\"k\":%u,\"n\":%u,\"rc\":\"%s\",\"pass\":%s,\"chip_elapsed_ns\":%" PRIu64
                    ",\"wall_ms\":%.3f,\"max_abs_err\":%g,\"max_rel_err\":%g,\"mismatch_count\":%zu,"
                    "\"parity_verified\":%s,\"completion_marker\":%u,\"repeat_rc\":\"%s\",\"repeat_cache_hit\":%s,"
                    "\"repeat_mismatch\":%zu,\"e2e_max_rel\":%g,\"k_slices\":%u,\"chip_calls\":%u,\"rows_per_call\":%u,\"padded_m\":%u,\"padded_n\":%u,\"target_chip\":\"%s\",\"sm_architecture\":%u}",
                    first ? "" : ",", m, k, n, omega_gpu_matmul_rc_name(rc), pass ? "true" : "false",
                    info.elapsed_ns, wall_ms, info.max_abs_err, info.max_rel_err, info.mismatch_count,
                    info.parity_verified ? "true" : "false", info.completion_marker,
                    omega_gpu_matmul_rc_name(rc2), info2.kernel_cache_hit ? "true" : "false",
                    repeat_mismatch, e2e_max_rel, info.k_slices, info.chip_calls, info.rows_per_call, info.padded_m, info.padded_n, info.target_chip, info.sm_architecture);
            first = 0;
        }
        free(a); free(b); free(c); free(c2);
    }
    free(copy);
    if (out) {
        fprintf(out, "],\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed,
                g_failed == 0 ? "OMEGA_GPU_MATMUL_API_PASS" : "OMEGA_GPU_MATMUL_API_FAIL");
        fclose(out);
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *out_path = NULL;
    int host = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    }
    if (host) host_only();
    else if (chip_sweep(out_path) != 0) return 2;
    printf("%s: %d checks, %d failed\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
