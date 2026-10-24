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

#ifndef OMEGA_GPU_HOSTMATH_H
#define OMEGA_GPU_HOSTMATH_H
/*
 * omega_gpu_hostmath: libm-free double-precision host math for the production side of libomega_gpu.a
 * (OWN-30B, 2026-10-13). The library must link without -lm (omegatool and the Rust FFI crate link it bare),
 * so the few transcendental values the APIs build on the host (the RoPE cos/sin table, the inverse frequency
 * powers, the router oracle's exp) come from here. Everything is plain double arithmetic: range reduction plus
 * short Taylor/atanh series, checked against libm by tests/gpu_decode_attn_test.c --host-only (the test links -lm).
 *
 *   log(x):      x > 0 finite; NaN otherwise. Error a few ulp.
 *   exp(y):      0 below -745, +inf above 709.8. Error a few ulp.
 *   pow(b, e):   b > 0 finite; exp(e * log(b)). Relative error about |e * log b| * 2^-52.
 *   sincos(a):   valid for |a| < 2^20 * pi/2 (about 1.6e6; the RoPE table needs pos < 65536). Three-part
 *                Cody-Waite reduction with the fdlibm pi/2 pieces, series to r^16. Absolute error about 1e-16.
 *   pow2f(e):    2^e as f32, exact (built from the bit pattern) for -149 <= e <= 127; 0 below, +inf above.
 *                Replaces ldexpf(1.0f, e) (omega-mv#471).
 */
double omega_hostmath_log(double x);
double omega_hostmath_exp(double y);
double omega_hostmath_pow(double base, double e);
void omega_hostmath_sincos(double a, double *s, double *c);
float omega_hostmath_pow2f(int e);
#endif
