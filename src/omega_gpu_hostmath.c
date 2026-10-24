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

/* omega_gpu_hostmath: libm-free double-precision log / exp / pow / sincos (see the header). */
#include "omega_gpu_hostmath.h"
#include <stdint.h>
#include <string.h>

static double bits_d(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint64_t d_bits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

#define HM_LN2_HI 6.93147180369123816490e-01 /* fdlibm: ln2 split so that k * LN2_HI is exact for |k| < 2^20 */
#define HM_LN2_LO 1.90821492927058770002e-10
#define HM_INV_LN2 1.44269504088896338700e+00
#define HM_SQRT2 1.41421356237309504880e+00

/* 2^k as a double for -1074 <= k <= 1023 (two steps below the normal range) */
static double pow2i(int k) {
    if (k > 1023) return bits_d(0x7ff0000000000000ull);
    if (k < -1074) return 0.0;
    if (k >= -1022) return bits_d((uint64_t)(k + 1023) << 52);
    return bits_d((uint64_t)1 << 52) * bits_d((uint64_t)(k + 1022 + 1023) << 52); /* 2^-1022 * 2^(k+1022) */
}

double omega_hostmath_log(double x) {
    if (!(x > 0.0) || x > 1.7976931348623157e308) return bits_d(0x7ff8000000000000ull);
    int e = 0;
    if (x < 2.2250738585072014e-308) { x *= 18014398509481984.0; e = -54; } /* subnormal: scale by 2^54 */
    uint64_t u = d_bits(x);
    e += (int)((u >> 52) & 0x7ff) - 1023;
    double m = bits_d((u & 0x000fffffffffffffull) | 0x3ff0000000000000ull); /* [1, 2) */
    if (m > HM_SQRT2) { m *= 0.5; e += 1; }                                 /* [sqrt(1/2), sqrt(2)) */
    double t = (m - 1.0) / (m + 1.0), t2 = t * t;                           /* |t| < 0.1716 */
    double s = 0.0;
    for (int k = 25; k >= 1; k -= 2) s = s * t2 + 1.0 / (double)k;         /* sum t2^j / (2j+1), j = 0..12 */
    return (double)e * HM_LN2_HI + (2.0 * t * s + (double)e * HM_LN2_LO);
}

double omega_hostmath_exp(double y) {
    if (y != y) return y;
    if (y > 709.78) return bits_d(0x7ff0000000000000ull);
    if (y < -745.2) return 0.0;
    double fk = y * HM_INV_LN2;
    long long k = (long long)(fk + (fk >= 0.0 ? 0.5 : -0.5));
    double kd = (double)k;
    double r = (y - kd * HM_LN2_HI) - kd * HM_LN2_LO;                       /* |r| <= 0.3466 */
    double p = 1.0;
    for (int n = 17; n >= 1; n--) p = 1.0 + r * p / (double)n;             /* 1 + r(1 + r/2(1 + r/3(...))) */
    if (k < -1020) return p * pow2i(-1020) * pow2i((int)k + 1020);
    return p * pow2i((int)k);
}

double omega_hostmath_pow(double base, double e) {
    if (base == 1.0 || e == 0.0) return 1.0;
    return omega_hostmath_exp(e * omega_hostmath_log(base));
}

/* fdlibm e_rem_pio2 pieces of pi/2: the first three have 33 significant bits, so n * piece is exact for n < 2^20 */
#define HM_PIO2_1 1.57079632673412561417e+00
#define HM_PIO2_2 6.07710050630396597660e-11
#define HM_PIO2_3 2.02226624871116645580e-21
#define HM_PIO2_3T 8.47842766036889956997e-32
#define HM_2_OVER_PI 6.36619772367581382433e-01

void omega_hostmath_sincos(double a, double *s, double *c) {
    double fn = a * HM_2_OVER_PI;
    long long n = (long long)(fn + (fn >= 0.0 ? 0.5 : -0.5));
    double nd = (double)n;
    double r = a - nd * HM_PIO2_1;
    r -= nd * HM_PIO2_2;
    r -= nd * HM_PIO2_3;
    r -= nd * HM_PIO2_3T;                                                    /* |r| <= pi/4 + tiny */
    double w = r * r;
    double ps = 1.0;
    for (int k = 15; k >= 3; k -= 2) ps = 1.0 - w * ps / (double)(k * (k - 1)); /* sin r = r * ps, terms to r^15 */
    double pc = 1.0;
    for (int k = 16; k >= 2; k -= 2) pc = 1.0 - w * pc / (double)(k * (k - 1)); /* cos r = pc, terms to r^16 */
    double sr = r * ps, cr = pc;
    switch ((int)(n & 3)) {
    case 0: *s = sr; *c = cr; break;
    case 1: *s = cr; *c = -sr; break;
    case 2: *s = -sr; *c = -cr; break;
    default: *s = -cr; *c = sr; break;
    }
}
