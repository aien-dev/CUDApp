/* SPDX-License-Identifier: AGPL-3.0-only */
/* cubin_probe: load an exported sm_121 cubin through the CUDA driver and run it.
 *
 * This is an interop qualification tool, not part of libomega_gpu: it links
 * libcuda only to validate that the artifact produced by Omega's own encoder is
 * accepted by the vendor loader and computes the contract result. It refuses to
 * run without SPARK_WINDOW. It never resets the device or changes driver state.
 *
 *   cubin_probe CUBIN MATMUL_FN M K N
 *
 * Exit: 0 PASS, 2 usage, 3 no window, 4 CUDA/load failure, 5 numeric failure. */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omega_blackwell_matmul.h"

#define CK(x) do { CUresult r_ = (x); if (r_) { const char *s_; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "FAIL %s -> %d %s\n", #x, (int)r_, s_); return 4; } } while (0)

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

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: cubin_probe CUBIN MATMUL_FN M K N\n");
        return 2;
    }
    if (!getenv("SPARK_WINDOW")) {
        fprintf(stderr, "refused: SPARK_WINDOW is not set; no authorized GPU window\n");
        return 3;
    }
    const uint32_t M = (uint32_t)strtoul(argv[3], NULL, 0);
    const uint32_t K = (uint32_t)strtoul(argv[4], NULL, 0);
    const uint32_t N = (uint32_t)strtoul(argv[5], NULL, 0);
    if (!M || !K || !N) return 2;
    size_t cub_n = 0;
    uint8_t *cub = slurp(argv[1], &cub_n);
    if (!cub) { fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }

    CK(cuInit(0));
    CUdevice dev;
    CK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CUctxCreateParams cparams;
    memset(&cparams, 0, sizeof cparams);
    CK(cuCtxCreate(&ctx, &cparams, 0, dev));
    CUmodule mod;
    CK(cuModuleLoadData(&mod, cub));
    CUfunction fn;
    CK(cuModuleGetFunction(&fn, mod, argv[2]));

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
    void *args[] = { &dA, &dB, &dC };
    CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, NULL, args, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(C, dC, csz));
    int bad = 0;
    double worst = 0;
    for (uint32_t i = 0; i < M * N; i++) {
        double d = (double)C[i] - (double)ref[i];
        if (d < 0) d = -d;
        if (d > worst) worst = d;
        if (d > 1e-4) bad++;
    }
    printf("cubin=%s fn=%s M=%u K=%u N=%u worst_abs=%.9g bad=%d window=%s\n",
           argv[1], argv[2], M, K, N, worst, bad, getenv("SPARK_WINDOW"));
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    cuModuleUnload(mod); cuCtxDestroy(ctx); free(A); free(B); free(C); free(ref); free(cub);
    return bad ? 5 : 0;
}
