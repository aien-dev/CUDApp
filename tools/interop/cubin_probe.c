/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* cubin_probe: load an exported sm_121 cubin through the CUDA driver and run it.
 *
 * This is an interop qualification tool, not part of libomega_gpu: it links
 * libcuda only to validate that the artifact produced by Omega's own encoder is
 * accepted by the vendor loader and computes the contract result. It refuses to
 * run without SPARK_WINDOW. It never resets the device or changes driver state.
 *
 *   cubin_probe [--load-only | --expect-reject] CUBIN MATMUL_FN M K N
 *
 * The image is loaded with cuModuleLoadDataEx and the driver's error and info
 * logs are printed on every load, so a rejection carries whatever reason the
 * loader gives. --load-only stops after cuModuleGetFunction. --expect-reject
 * is a negative control: it passes only when load or symbol lookup fails.
 * The output buffer is filled with a sentinel before launch, so a kernel that
 * writes nothing cannot pass against a zero reference.
 *
 * Exit: 0 PASS (or the expected rejection), 2 usage, 3 no window,
 *       4 CUDA/load failure, 5 numeric failure, 6 rejection expected but the
 *       image was accepted. */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omega_blackwell_matmul.h"

#define CK(x) do { CUresult r_ = (x); if (r_) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "FAIL %s -> %d %s\n", #x, (int)r_, s_); return 4; } } while (0)

#define LOG_BYTES 8192
#define SENTINEL (-12345.0f)

static uint8_t *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    uint8_t *b = malloc((size_t)sz);
    if (!b) { fclose(f); return NULL; }
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    fclose(f); *n = (size_t)sz; return b;
}

/* Load with the driver's error/info logs attached; prints one LOAD line. */
static CUresult load_logged(CUmodule *mod, const uint8_t *img) {
    static char err[LOG_BYTES], info[LOG_BYTES];
    err[0] = info[0] = 0;
    CUjit_option opt[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                           CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                           CU_JIT_LOG_VERBOSE };
    void *val[] = { err, (void *)(uintptr_t)LOG_BYTES, info, (void *)(uintptr_t)LOG_BYTES,
                    (void *)(uintptr_t)1 };
    CUresult r = cuModuleLoadDataEx(mod, img, (unsigned)(sizeof opt / sizeof opt[0]), opt, val);
    const char *s = "?";
    cuGetErrorString(r, &s);
    printf("LOAD cuModuleLoadDataEx -> %d %s\n", (int)r, s);
    printf("LOAD error_log=\"%s\"\nLOAD info_log=\"%s\"\n", err, info);
    return r;
}

int main(int argc, char **argv) {
    int load_only = 0, expect_reject = 0, a = 1;
    if (argc > 1 && strcmp(argv[1], "--load-only") == 0) { load_only = 1; a++; }
    else if (argc > 1 && strcmp(argv[1], "--expect-reject") == 0) { expect_reject = 1; a++; }
    if (argc - a != 5) {
        fprintf(stderr, "usage: cubin_probe [--load-only | --expect-reject] CUBIN MATMUL_FN M K N\n");
        return 2;
    }
    const char *path = argv[a], *fname = argv[a + 1];
    if (!getenv("SPARK_WINDOW")) {
        fprintf(stderr, "refused: SPARK_WINDOW is not set; no authorized GPU window\n");
        return 3;
    }
    const uint32_t M = (uint32_t)strtoul(argv[a + 2], NULL, 0);
    const uint32_t K = (uint32_t)strtoul(argv[a + 3], NULL, 0);
    const uint32_t N = (uint32_t)strtoul(argv[a + 4], NULL, 0);
    if (!M || !K || !N) return 2;
    size_t cub_n = 0;
    uint8_t *cub = slurp(path, &cub_n);
    if (!cub) { fprintf(stderr, "cannot read %s\n", path); return 2; }

    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CUctxCreateParams cparams;
    memset(&cparams, 0, sizeof cparams);
    CK(cuCtxCreate(&ctx, &cparams, 0, dev));
    CUmodule mod;
    CUresult lr = load_logged(&mod, cub);
    CUfunction fn;
    CUresult fr = CUDA_ERROR_NOT_FOUND;
    if (lr == CUDA_SUCCESS) {
        fr = cuModuleGetFunction(&fn, mod, fname);
        const char *s = "?";
        cuGetErrorString(fr, &s);
        printf("LOAD cuModuleGetFunction(%s) -> %d %s\n", fname, (int)fr, s);
    }
    if (expect_reject) {
        if (lr == CUDA_SUCCESS) cuModuleUnload(mod);
        cuCtxDestroy(ctx);
        free(cub);
        if (lr != CUDA_SUCCESS || fr != CUDA_SUCCESS) {
            printf("REJECT-PASS cubin=%s fn=%s window=%s\n", path, fname, getenv("SPARK_WINDOW"));
            return 0;
        }
        printf("REJECT-FAIL accepted cubin=%s fn=%s window=%s\n", path, fname, getenv("SPARK_WINDOW"));
        return 6;
    }
    if (lr != CUDA_SUCCESS || fr != CUDA_SUCCESS) return 4;
    if (load_only) {
        printf("LOAD-PASS cubin=%s fn=%s window=%s\n", path, fname, getenv("SPARK_WINDOW"));
        cuModuleUnload(mod); cuCtxDestroy(ctx); free(cub);
        return 0;
    }

    const size_t asz = (size_t)M * K * 2, bsz = (size_t)K * N * 2, csz = (size_t)M * N * 4;
    uint16_t *A = malloc(asz), *B = malloc(bsz);
    float *C = malloc(csz), *ref = malloc(csz);
    if (!A || !B || !C || !ref) return 4;
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = omega_fp32_to_bf16((float)((int)(i % 7) - 3) * 0.25f);
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = omega_fp32_to_bf16((float)((int)(i % 5) - 2) * 0.5f);
    omega_matmul_cpu_oracle_bf16(A, B, ref, M, K, N);

    CUdeviceptr dA, dB, dC;
    CK(cuMemAlloc(&dA, asz)); CK(cuMemAlloc(&dB, bsz)); CK(cuMemAlloc(&dC, csz));
    CK(cuMemcpyHtoD(dA, A, asz)); CK(cuMemcpyHtoD(dB, B, bsz));
    float sentinel = SENTINEL;
    uint32_t sbits;
    memcpy(&sbits, &sentinel, sizeof sbits);
    CK(cuMemsetD32(dC, sbits, (size_t)M * N));
    void *args[] = { &dA, &dB, &dC };
    CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(C, dC, csz));
    int bad = 0, untouched = 0;
    double worst = 0;
    for (uint32_t i = 0; i < M * N; i++) {
        if (C[i] == SENTINEL) untouched++;
        double d = (double)C[i] - (double)ref[i];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
        if (d > 1e-4) bad++;
    }
    printf("cubin=%s fn=%s M=%u K=%u N=%u worst_abs=%.9g bad=%d untouched=%d window=%s\n",
           path, fname, M, K, N, worst, bad, untouched, getenv("SPARK_WINDOW"));
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    cuModuleUnload(mod); cuCtxDestroy(ctx); free(A); free(B); free(C); free(ref); free(cub);
    return bad ? 5 : 0;
}
