#include <cuda_bf16.h>
extern "C" __global__ void k(const float *a, unsigned *o, float *f) {
  unsigned i = threadIdx.x;
  float x = a[i], y = a[i + 32];
  float m = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
  float s = y + __shfl_xor_sync(0xffffffffu, y, 1);
  s = s + __shfl_xor_sync(0xffffffffu, s, 2);
  __nv_bfloat162 p = __floats2bfloat162_rn(m, s);
  o[i] = *(unsigned *)&p;
  __nv_bfloat162 q = __floats2bfloat162_rn(x * 3.0f, y * 5.0f);
  o[i + 32] = *(unsigned *)&q;
  f[i] = exp2f(m);
}
