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

/* bw_hmma_model.h (OWN-30B T3, 2026-10-13): parametrised C model of one HMMA.16816.F32.BF16 dot product
 *   d = c + sum_{k<16} a_k * b_k      (a_k, b_k bf16 values given as f32; products are exact in double)
 * The vendor documents no rounding behaviour for mma (DOCS SILENT, Context7 PTX ISA). Published measurements of earlier
 * tensor cores (Fasi, Higham, Mikaitis, Pranesh 2021) describe: exact products, alignment of a group of terms to the
 * largest exponent with the shifted-out bits truncated, exact addition, and a truncating normalisation. This header
 * implements that family so the chip probe (tests/gpu_tc_probe_test.c) can find the member the GB10 matches bit for bit:
 *   group  G in {1, 2, 4, 8, 16}: the 16 products are consumed in 16/G groups, k order; each group is added to the
 *          running accumulator in one fused step
 *   extra  alignment keeps 24 + extra significant bits below the group's largest exponent (terms truncated toward 0)
 *   rn     0: the fused sum is converted to f32 by truncation (RZ), 1: round to nearest even
 * Zero, subnormal and huge inputs: products and sums in double; the probe keeps exponents in a range where double is exact.
 * HM_FMA (group 0) is the plain sequential fmaf chain, for comparison. */
#ifndef BW_HMMA_MODEL_H
#define BW_HMMA_MODEL_H
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct { int group, extra, rn, nominal, csep; } HmmaModel;
/* nominal 1: a product's alignment exponent is exp(a) + exp(b) + 1 (its significand taken as [1, 4)), not its true exponent.
 * csep 1: the group's products are aligned and summed among themselves first; the accumulator is then added exactly and the
 * total converted once (rn). */

static float hm_to_f32(double v, int rn) {
    if (rn) return (float)v; /* the C conversion is round to nearest even */
    float f = (float)v;
    if ((double)f != v && fabs((double)f) > fabs(v)) f = nextafterf(f, 0.0f); /* round toward zero */
    return f;
}
/* truncate v toward zero to a multiple of 2^qexp */
static double hm_trunc_to(double v, int qexp) {
    double q = ldexp(1.0, qexp);
    return trunc(v / q) * q;
}
static float hm_dot16(const HmmaModel *m, const float *a, const float *b, float c) {
    if (m->group == 0) { float acc = c; for (int k = 0; k < 16; k++) acc = fmaf(a[k], b[k], acc); return acc; }
    float acc = c;
    for (int k0 = 0; k0 < 16; k0 += m->group) {
        double t[17]; int ne[17]; int n = 0;
        if (!m->csep) { t[n] = (double)acc; { int e; frexp(t[n], &e); ne[n] = e; } n++; }
        for (int k = k0; k < k0 + m->group; k++) {
            t[n] = (double)a[k] * (double)b[k];
            int e; frexp(t[n], &e);
            if (m->nominal && t[n] != 0.0) { int ea, eb; frexp(a[k], &ea); frexp(b[k], &eb); e = ea + eb - (m->nominal == 2); } /* |a| < 2^ea, |b| < 2^eb; mode 2: exponent of the product of the leading bits, frexp convention */
            ne[n] = e; n++;
        }
        int emax = -100000;
        for (int i = 0; i < n; i++) if (t[i] != 0.0 && ne[i] > emax) emax = ne[i];
        if (emax == -100000) { if (!m->csep) acc = 0.0f * acc; continue; }
        double s = 0.0;
        for (int i = 0; i < n; i++) s += hm_trunc_to(t[i], emax - 24 - m->extra);
        if (m->csep) s += (double)acc;
        acc = hm_to_f32(s, m->rn);
    }
    return acc;
}
#endif
