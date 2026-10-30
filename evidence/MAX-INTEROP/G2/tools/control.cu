// Known-good control: same name and parameter ABI as the Omega artifact
// omega_matmul_bf16(A bf16[M*K], B bf16[K*N], C f32[M*N]), M=16 K=16 N=8, 1 block x 32 threads.
#include <stdint.h>
__device__ static float bf(uint16_t h) { return __uint_as_float(((uint32_t)h) << 16); }
extern "C" __global__ void omega_matmul_bf16(const uint16_t *A, const uint16_t *B, float *C) {
    const int M = 16, K = 16, N = 8;
    for (int o = threadIdx.x; o < M * N; o += blockDim.x) {
        int i = o / N, j = o % N;
        float s = 0.0f;
        for (int p = 0; p < K; p++) s += bf(A[i * K + p]) * bf(B[p * N + j]);
        C[o] = s;
    }
}
