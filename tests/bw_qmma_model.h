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

/* bw_qmma_model.h (OWN-30B L1 GEMM, W-L1G, 2026-10-14): parametrised C model of one QMMA.16832.F32.E4M3.E4M3 dot product
 *   d = c + sum_{k<32} a_k * b_k      (a_k, b_k e4m3 values given as f32; products exact in double)
 * Same family as tests/bw_hmma_model.h (chip-exact for HMMA: evidence bf9b0378) widened for FP8: the vendor documents no
 * rounding behaviour (DOCS SILENT), and FP8 tensor cores of earlier chips are reported to keep fewer accumulator bits than
 * f32 (UNVERIFIED for GB10), so `keep` spans 10 .. 32 significant bits below the group's largest exponent.
 *   group  G in {1, 2, 4, 8, 16, 32}: products consumed in 32/G groups, k order, each added to the accumulator in one step
 *   keep   bits kept below the group's largest exponent (terms truncated toward 0)
 *   rn     0: the fused sum converts to f32 by truncation, 1: round to nearest even
 *   nominal 1: a product's alignment exponent is exp(a) + exp(b) (frexp), 2: that minus 1
 *   csep   1: products aligned among themselves, the accumulator added exactly, one conversion */
#ifndef BW_QMMA_MODEL_H
#define BW_QMMA_MODEL_H
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct { int group, keep, rn, nominal, csep; } QmmaModel;

static float qm_to_f32(double v, int rn) {
    if (rn) return (float)v;
    float f = (float)v;
    if ((double)f != v && fabs((double)f) > fabs(v)) f = nextafterf(f, 0.0f);
    return f;
}
static double qm_trunc_to(double v, int qexp) { double q = ldexp(1.0, qexp); return trunc(v / q) * q; }
static float qm_dot32(const QmmaModel *m, const float *a, const float *b, float c) {
    if (m->group == 0) { float acc = c; for (int k = 0; k < 32; k++) acc = fmaf(a[k], b[k], acc); return acc; }
    float acc = c;
    for (int k0 = 0; k0 < 32; k0 += m->group) {
        double t[33]; int ne[33]; int n = 0;
        if (!m->csep) { t[n] = (double)acc; { int e; frexp(t[n], &e); ne[n] = e; } n++; }
        for (int k = k0; k < k0 + m->group; k++) {
            t[n] = (double)a[k] * (double)b[k];
            int e; frexp(t[n], &e);
            if (m->nominal && t[n] != 0.0) { int ea, eb; frexp(a[k], &ea); frexp(b[k], &eb); e = ea + eb - (m->nominal == 2); }
            ne[n] = e; n++;
        }
        int emax = -100000;
        for (int i = 0; i < n; i++) if (t[i] != 0.0 && ne[i] > emax) emax = ne[i];
        if (emax == -100000) { if (!m->csep) acc = 0.0f * acc; continue; }
        double s = 0.0;
        for (int i = 0; i < n; i++) s += qm_trunc_to(t[i], emax - m->keep);
        if (m->csep) s += (double)acc;
        acc = qm_to_f32(s, m->rn);
    }
    return acc;
}

/* e4m3 (OCP FP8, bias 7, no infinities, 0x7f / 0xff NaN) */
static float qm_e4m3(uint8_t c) {
    int s = c >> 7, e = (c >> 3) & 15, mm = c & 7;
    float v = e ? ldexpf(1.0f + (float)mm / 8.0f, e - 7) : ldexpf((float)mm / 8.0f, -6);
    return s ? -v : v;
}
/* f32 -> e4m3, round to nearest even, saturate to +-448 (PTX cvt.rn.satfinite; NaN handled by the caller) */
static uint8_t qm_e4m3_rne_sat(float f) {
    uint8_t s = signbit(f) ? 0x80 : 0; double a = fabs((double)f);
    if (isinf(f) || a > 448.0) return s | 0x7e;
    if (a < ldexp(1.0, -6)) return s | (uint8_t)rint(a * 512.0);
    int e; double m = frexp(a, &e); m *= 2.0; e -= 1; /* a = m * 2^e, m in [1, 2) */
    double mm = rint((m - 1.0) * 8.0);
    if (mm >= 8.0) { mm = 0.0; e++; }
    if (e + 7 > 15 || (((e + 7) << 3) | (int)mm) > 0x7e) return s | 0x7e;
    return s | (uint8_t)(((e + 7) << 3) | (int)mm);
}
#endif
