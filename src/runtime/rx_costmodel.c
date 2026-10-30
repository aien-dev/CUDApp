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

/*
 * rx_costmodel.c -- Omega's empirical cost model. See rx_costmodel.h.
 *
 * Pure arithmetic over measurements the caller reports. No execution, no
 * authority, no generation store.
 */
#include "runtime/rx_costmodel.h"
#include "sha256.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PRIOR_A     1.0
#define PRIOR_B     0.01     /* prior noise variance ~0.01 log2^2 (about 7%) */
#define NOISE_FLOOR 0.0009   /* never claim better than ~2% sd per call */
#define FAIL_PRIOR  0.05

void rx_cm_default_policy(RxCmPolicy *p) {
    p->ei_min = 0.08;
    p->max_fail_p = 0.0;   /* any wrong result excludes the arm */
    p->fallback_arm = 0;
    p->explore_frac = 0.05;
    p->explore_allow_ps = 500000000ull;     /* 0.5 ms to get started */
}

void rx_cm_init(RxCostModel *m, uint32_t n_arms) {
    memset(m, 0, sizeof *m);
    m->n_arms = n_arms > RX_CM_ARMS ? RX_CM_ARMS : n_arms;
    m->lambda = 1e-3;
    m->sd_scale = 1.0;
    for (uint32_t c = 0; c < RX_CM_CELLS; c++)
        for (uint32_t a = 0; a < RX_CM_ARMS; a++)
            for (uint32_t i = 0, k = 0; i < RX_CM_D; i++)
                for (uint32_t j = i; j < RX_CM_D; j++, k++)
                    m->cell[c][a].tri[k] = i == j ? m->lambda : 0.0;
}

uint32_t rx_cm_cell(const RxCmFeatures *f) {
    uint32_t op = f->op < RX_CM_OPS ? f->op : RX_CM_OPS - 1;
    uint32_t core = f->core < RX_CM_CORES ? f->core : RX_CM_CORE_OTHER;
    uint32_t pr = f->pressure < RX_CM_PRESSURE ? f->pressure : RX_CM_PRESSURE - 1;
    return (op * RX_CM_CORES + core) * RX_CM_PRESSURE + pr;
}

void rx_cm_basis(const RxCmFeatures *f, double phi[RX_CM_D]) {
    double lm = log2((double)(f->M ? f->M : 1));
    double ln = log2((double)(f->N ? f->N : 1));
    double ov = 0.0;
    if (f->cache_bytes && f->state_bytes > f->cache_bytes)
        ov = log2((double)f->state_bytes / (double)f->cache_bytes);
    phi[0] = 1.0;
    phi[1] = lm;
    phi[2] = ln;
    phi[3] = lm * ln;
    phi[4] = ln * ln;
    phi[5] = f->N <= 3 ? 1.0 : 0.0;
    phi[6] = (f->N % 4u) != 0 ? 1.0 : 0.0;
    phi[7] = ov;
}

/* Cholesky of the D x D matrix held as an upper triangle. 0 on success. */
static int cholesky(const double *tri, double L[RX_CM_D][RX_CM_D]) {
    double A[RX_CM_D][RX_CM_D];
    for (uint32_t i = 0, k = 0; i < RX_CM_D; i++)
        for (uint32_t j = i; j < RX_CM_D; j++, k++) A[i][j] = A[j][i] = tri[k];
    memset(L, 0, sizeof(double) * RX_CM_D * RX_CM_D);
    for (uint32_t j = 0; j < RX_CM_D; j++) {
        double s = A[j][j];
        for (uint32_t k = 0; k < j; k++) s -= L[j][k] * L[j][k];
        if (!(s > 0.0)) return -1;
        L[j][j] = sqrt(s);
        for (uint32_t i = j + 1; i < RX_CM_D; i++) {
            double t = A[i][j];
            for (uint32_t k = 0; k < j; k++) t -= L[i][k] * L[j][k];
            L[i][j] = t / L[j][j];
        }
    }
    return 0;
}

static void forward(double L[RX_CM_D][RX_CM_D], const double *b, double *v) {
    for (uint32_t i = 0; i < RX_CM_D; i++) {
        double s = b[i];
        for (uint32_t k = 0; k < i; k++) s -= L[i][k] * v[k];
        v[i] = s / L[i][i];
    }
}

static void backward(double L[RX_CM_D][RX_CM_D], const double *v, double *x) {
    for (int i = (int)RX_CM_D - 1; i >= 0; i--) {
        double s = v[i];
        for (uint32_t k = (uint32_t)i + 1; k < RX_CM_D; k++) s -= L[k][i] * x[k];
        x[i] = s / L[i][i];
    }
}

/* Predictive t for one (cell, arm). Returns 0 if there is no data. */
static int fit_predict(const RxCmArmStats *s, const double phi[RX_CM_D], double *mean,
                       double *sd, double *dof) {
    if (s->n == 0) return 0;
    double L[RX_CM_D][RX_CM_D], v[RX_CM_D], mu[RX_CM_D], w[RX_CM_D];
    if (cholesky(s->tri, L) != 0) return 0;
    forward(L, s->xty, v);
    backward(L, v, mu);
    double quad = 0.0;          /* mu' Lambda mu = |L' mu|^2 = v . v */
    for (uint32_t i = 0; i < RX_CM_D; i++) quad += v[i] * v[i];
    double a = PRIOR_A + 0.5 * (double)s->n;
    double b = PRIOR_B + 0.5 * (s->yty - quad);
    if (b < PRIOR_B) b = PRIOR_B;
    double sigma2 = b / a;
    if (sigma2 < NOISE_FLOOR) sigma2 = NOISE_FLOOR;
    forward(L, phi, w);
    double lev = 0.0;
    for (uint32_t i = 0; i < RX_CM_D; i++) lev += w[i] * w[i];
    double m = 0.0;
    for (uint32_t i = 0; i < RX_CM_D; i++) m += mu[i] * phi[i];
    *mean = m;
    *sd = sqrt(sigma2 * (1.0 + lev));
    *dof = 2.0 * a;
    return 1;
}

static uint64_t failures_of(const RxCostModel *m, uint32_t arm) {
    uint64_t f = 0;
    for (uint32_t c = 0; c < RX_CM_CELLS; c++) f += m->cell[c][arm].failures;
    return f;
}

static double fail_p(const RxCostModel *m, uint32_t arm) {
    uint64_t t = 0, f = 0;
    for (uint32_t c = 0; c < RX_CM_CELLS; c++) {
        t += m->cell[c][arm].trials;
        f += m->cell[c][arm].failures;
    }
    return ((double)f + FAIL_PRIOR) / ((double)t + 1.0);
}

/* Pressure cells hold a correction over the same core class at pressure 0:
 * log2 cost under pressure = quiet prediction + a + b * (doublings past the
 * private cache). Two numbers per arm, readable as "pressure costs this much,
 * and this much more per doubling out of cache". */
static void residual_basis(const RxCmFeatures *f, double phi[RX_CM_D]) {
    double full[RX_CM_D];
    rx_cm_basis(f, full);
    memset(phi, 0, sizeof(double) * RX_CM_D);
    phi[0] = 1.0;
    phi[1] = full[7];
}

static int quiet_predict(const RxCostModel *m, const RxCmFeatures *f, uint32_t arm, double *mean,
                         double *sd, double *dof, uint64_t *n) {
    RxCmFeatures g = *f;
    g.pressure = 0;
    const RxCmArmStats *b = &m->cell[rx_cm_cell(&g)][arm];
    double phi[RX_CM_D];
    rx_cm_basis(f, phi);
    if (b->n < RX_CM_MIN_OBS || !fit_predict(b, phi, mean, sd, dof)) return 0;
    *n = b->n;
    return 1;
}

static uint32_t bucket(const RxCmFeatures *f) {
    return f->pressure < RX_CM_PRESSURE ? f->pressure : RX_CM_PRESSURE - 1;
}

void rx_cm_predict(const RxCostModel *m, const RxCmFeatures *f, uint32_t arm,
                   RxCmPrediction *out) {
    memset(out, 0, sizeof *out);
    if (arm >= m->n_arms) return;
    double mean = 0, sd = 0, dof = 0;
    uint64_t n = 0;
    out->fail_p = fail_p(m, arm);
    if (!quiet_predict(m, f, arm, &mean, &sd, &dof, &n)) return;
    if (bucket(f) == 0) {
        out->known = 1;
    } else {
        const RxCmArmStats *r = &m->cell[rx_cm_cell(f)][arm];
        double phi[RX_CM_D], rm = 0, rsd = 0, rdof = 0;
        residual_basis(f, phi);
        if (r->n >= RX_CM_RES_MIN_OBS && fit_predict(r, phi, &rm, &rsd, &rdof)) {
            mean += rm;
            sd = sqrt(sd * sd + rsd * rsd);
            if (rdof < dof) dof = rdof;
            n = r->n;
            out->known = 1;
        } else {
            /* Never seen under this pressure: the quiet prediction, with the
             * prior on how far an unseen condition may move cost. */
            sd = sqrt(sd * sd + RX_CM_UNSEEN_SHIFT * RX_CM_UNSEEN_SHIFT);
            out->borrowed = 1;
        }
    }
    out->n = (uint32_t)n;
    out->mean_log2_ps = mean;
    out->sd_log2 = sd * (m->sd_scale >= 1.0 ? m->sd_scale : 1.0);
    out->dof = dof;
    uint32_t core = f->core < RX_CM_CORES ? f->core : RX_CM_CORE_OTHER;
    double mw = m->power_mw[core][arm];
    if (mw > 0.0) out->energy_pj = exp2(mean) * mw * 1e-3;   /* ps * mW = 1e-3 pJ */
}

int rx_cm_observe(RxCostModel *m, const RxCmFeatures *f, uint32_t arm, const RxCmObservation *o) {
    if (!m || !f || !o || arm >= m->n_arms) return RX_CM_ERR_ARG;
    RxCmArmStats *s = &m->cell[rx_cm_cell(f)][arm];
    s->trials++;
    if (o->failed) {
        s->failures++;
        return RX_CM_OK;
    }
    if (o->ps_per_call == 0) return RX_CM_ERR_ARG;
    double phi[RX_CM_D];
    double y = log2((double)o->ps_per_call);
    if (bucket(f) == 0) {
        rx_cm_basis(f, phi);
    } else {
        double qm, qsd, qdof;
        uint64_t qn;
        if (!quiet_predict(m, f, arm, &qm, &qsd, &qdof, &qn)) return RX_CM_SKIPPED;
        residual_basis(f, phi);
        y -= qm;
    }
    for (uint32_t i = 0, k = 0; i < RX_CM_D; i++) {
        for (uint32_t j = i; j < RX_CM_D; j++, k++) s->tri[k] += phi[i] * phi[j];
        s->xty[i] += phi[i] * y;
    }
    s->yty += y * y;
    s->n++;
    return RX_CM_OK;
}

static double norm_cdf(double x) { return 0.5 * erfc(-x / sqrt(2.0)); }

/* Acklam's inverse normal. */
static double norm_inv(double p) {
    static const double a[] = { -3.969683028665376e+01, 2.209460984245205e+02,
                                -2.759285104469687e+02, 1.383577518672690e+02,
                                -3.066479806614716e+01, 2.506628277459239e+00 };
    static const double b[] = { -5.447609879822406e+01, 1.615858368580409e+02,
                                -1.556989798598866e+02, 6.680131188771972e+01,
                                -1.328068155288572e+01 };
    static const double c[] = { -7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                4.374664141464968e+00, 2.938163982698783e+00 };
    static const double d[] = { 7.784695709041462e-03, 3.224671290700398e-01,
                                2.445134137142996e+00, 3.754408661907416e+00 };
    if (p <= 0.0) return -INFINITY;
    if (p >= 1.0) return INFINITY;
    double q, r;
    if (p < 0.02425) {
        q = sqrt(-2 * log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p > 1 - 0.02425) {
        q = sqrt(-2 * log(1 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

double rx_cm_t_quantile(double q, double dof) {
    double z = norm_inv(0.5 + 0.5 * q);
    if (!(dof > 0.0)) return z;
    double z3 = z * z * z, z5 = z3 * z * z;
    return z + (z3 + z) / (4.0 * dof) + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * dof * dof);
}

static int explore_allowed(const RxCostModel *m, const RxCmPolicy *p) {
    double cap = p->explore_frac * (double)m->work_ps + (double)p->explore_allow_ps;
    return (double)m->explore_ps < cap;
}

/* E[max(0, X)] for X ~ N(d, s^2): the expected gain, in log2 units, of
 * switching to a competitor whose advantage over the chosen arm is d. */
static double expected_improvement(double d, double s) {
    if (!(s > 0.0)) return d > 0.0 ? d : 0.0;
    double u = d / s;
    return s * exp(-0.5 * u * u) / sqrt(2.0 * M_PI) + d * norm_cdf(u);
}

void rx_cm_decide(const RxCostModel *m, const RxCmPolicy *p, const RxCmFeatures *f,
                  int allow_measure, RxCmDecision *out) {
    memset(out, 0, sizeof *out);
    uint32_t fb = p->fallback_arm < m->n_arms ? p->fallback_arm : 0;
    uint32_t elig = (f->eligible | (1u << fb)) & ((1u << m->n_arms) - 1u);
    int have[RX_CM_ARMS] = { 0 };
    int n_elig = 0;
    for (uint32_t a = 0; a < m->n_arms; a++) {
        rx_cm_predict(m, f, a, &out->pred[a]);
        if (!(elig & (1u << a))) continue;
        if (a != fb && failures_of(m, a) > 0 && out->pred[a].fail_p > p->max_fail_p) {
            elig &= ~(1u << a);
            continue;
        }
        n_elig++;
        have[a] = out->pred[a].known || out->pred[a].borrowed;
    }
    out->action = RX_CM_FALLBACK;
    out->arm = fb;
    if (n_elig <= 1) {
        out->why = RX_CM_WHY_NO_ELIGIBLE;
        return;
    }

    /* Best by predicted latency among arms that meet the budgets; if none
     * meets them, best overall and say so. */
    int best = -1, best_fit = -1;
    for (uint32_t a = 0; a < m->n_arms; a++) {
        if (!have[a] || !(elig & (1u << a))) continue;
        const RxCmPrediction *pr = &out->pred[a];
        double hi = exp2(pr->mean_log2_ps + 1.2816 * pr->sd_log2);
        int fits = (!f->latency_budget_ps || hi <= (double)f->latency_budget_ps) &&
                   (!f->energy_budget_pj || pr->energy_pj == 0.0 ||
                    pr->energy_pj <= (double)f->energy_budget_pj);
        if (best < 0 || pr->mean_log2_ps < out->pred[best].mean_log2_ps) best = (int)a;
        if (fits && (best_fit < 0 || pr->mean_log2_ps < out->pred[best_fit].mean_log2_ps))
            best_fit = (int)a;
    }
    int over = 0;
    if (best_fit >= 0) best = best_fit;
    else if (best >= 0) over = 1;

    /* Competitor: the eligible arm with the largest expected improvement
     * over `best`. An arm with no prediction at all is always worth a look. */
    int comp = -1, comp_unknown = 0;
    double ei = 0.0, pc = 0.0;
    for (uint32_t a = 0; a < m->n_arms; a++) {
        if (!(elig & (1u << a)) || (int)a == best) continue;
        double e, pa;
        int unknown = !have[a] || best < 0;
        if (unknown) {
            e = INFINITY;
            pa = 0.5;
        } else {
            const RxCmPrediction *x = &out->pred[a], *b = &out->pred[best];
            double s = sqrt(x->sd_log2 * x->sd_log2 + b->sd_log2 * b->sd_log2);
            double d = b->mean_log2_ps - x->mean_log2_ps;
            e = expected_improvement(d, s);
            pa = norm_cdf(d / (s > 0 ? s : 1e-9));
        }
        if (comp < 0 || e > ei) {
            comp = (int)a;
            ei = e;
            pc = pa;
            comp_unknown = unknown;
        }
    }
    out->p_runner_up_better = pc;
    out->expected_improvement = isinf(ei) ? -1.0 : ei;

    uint32_t why = RX_CM_WHY_CONFIDENT;
    if (best < 0 || comp_unknown) why = RX_CM_WHY_UNKNOWN_ARM;
    else if (ei > p->ei_min) why = RX_CM_WHY_WORTH_MEASURING;

    if (why != RX_CM_WHY_CONFIDENT && allow_measure && explore_allowed(m, p) && comp >= 0) {
        out->action = RX_CM_MEASURE;
        out->measure[0] = best >= 0 ? (uint32_t)best : fb;
        out->measure[1] = (uint32_t)comp;
        if (out->measure[0] == out->measure[1]) out->measure[1] = fb;
        out->why = why;
        out->arm = out->measure[0];
        return;
    }
    if (best < 0) {
        out->why = allow_measure && !explore_allowed(m, p) ? RX_CM_WHY_BUDGET_SPENT
                                                           : RX_CM_WHY_UNKNOWN_ARM;
        return;   /* fallback */
    }
    out->action = RX_CM_SELECT;
    out->arm = (uint32_t)best;
    if (over) out->why = RX_CM_WHY_OVER_BUDGET;
    else if (why != RX_CM_WHY_CONFIDENT && allow_measure) out->why = RX_CM_WHY_BUDGET_SPENT;
    else out->why = RX_CM_WHY_CONFIDENT;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

double rx_cm_calibrate(RxCostModel *m, const RxCmFeatures *f, const uint32_t *arm,
                       const uint64_t *ps, uint32_t n, double q) {
    if (!m || !f || !arm || !ps || n == 0 || !(q > 0.0 && q < 1.0)) return 0.0;
    double *r = malloc(sizeof(double) * n);
    if (!r) return 0.0;
    double saved = m->sd_scale;
    m->sd_scale = 1.0;
    uint32_t k = 0;
    for (uint32_t i = 0; i < n; i++) {
        RxCmPrediction p;
        rx_cm_predict(m, &f[i], arm[i], &p);
        if (!(p.known || p.borrowed) || ps[i] == 0 || !(p.sd_log2 > 0.0)) continue;
        double z = fabs(log2((double)ps[i]) - p.mean_log2_ps) / p.sd_log2;
        r[k++] = z / rx_cm_t_quantile(q, p.dof);
    }
    double scale = 0.0;
    if (k >= 10) {
        qsort(r, k, sizeof(double), cmp_double);
        uint32_t idx = (uint32_t)ceil(q * (double)k) - 1u;
        scale = r[idx < k ? idx : k - 1];
        if (scale < 1.0) scale = 1.0;
        m->sd_scale = scale;
    } else {
        m->sd_scale = saved;
    }
    free(r);
    return scale;
}

void rx_cm_charge(RxCostModel *m, uint64_t work_ps, uint64_t explore_ps) {
    m->work_ps += work_ps;
    m->explore_ps += explore_ps;
}

/* ---- canonical bytes ---- */

#define MAGIC   0x4d43474fu   /* "OGCM" little-endian */
#define VERSION 1u

typedef struct {
    uint32_t magic, version, n_arms, d, cells, arms_cap, cores;
    uint32_t reserved;
    double lambda;
} Header;

static size_t body_size(void) {
    const RxCostModel *m = 0;
    return sizeof m->cell + sizeof m->power_mw + sizeof m->verify_ns + sizeof m->synth_ns +
           sizeof m->code_bytes + sizeof m->sd_scale;
}

size_t rx_cm_blob_size(void) { return sizeof(Header) + body_size() + 32u; }

static size_t pack(const RxCostModel *m, uint8_t *out) {
    Header h;
    memset(&h, 0, sizeof h);
    h.magic = MAGIC;
    h.version = VERSION;
    h.n_arms = m->n_arms;
    h.d = RX_CM_D;
    h.cells = RX_CM_CELLS;
    h.arms_cap = RX_CM_ARMS;
    h.cores = RX_CM_CORES;
    h.lambda = m->lambda;
    uint8_t *p = out;
    memcpy(p, &h, sizeof h);
    p += sizeof h;
    memcpy(p, m->cell, sizeof m->cell);
    p += sizeof m->cell;
    memcpy(p, m->power_mw, sizeof m->power_mw);
    p += sizeof m->power_mw;
    memcpy(p, m->verify_ns, sizeof m->verify_ns);
    p += sizeof m->verify_ns;
    memcpy(p, m->synth_ns, sizeof m->synth_ns);
    p += sizeof m->synth_ns;
    memcpy(p, m->code_bytes, sizeof m->code_bytes);
    p += sizeof m->code_bytes;
    memcpy(p, &m->sd_scale, sizeof m->sd_scale);
    p += sizeof m->sd_scale;
    return (size_t)(p - out);
}

int rx_cm_serialize(const RxCostModel *m, uint8_t *out, size_t cap, size_t *out_len) {
    if (!m || !out || cap < rx_cm_blob_size()) return RX_CM_ERR_ARG;
    size_t n = pack(m, out);
    sha256_hash(out, n, out + n);
    if (out_len) *out_len = n + 32u;
    return RX_CM_OK;
}

int rx_cm_deserialize(RxCostModel *m, const uint8_t *in, size_t len) {
    if (!m || !in || len != rx_cm_blob_size()) return RX_CM_ERR_FORMAT;
    uint8_t d[32];
    sha256_hash(in, len - 32u, d);
    if (memcmp(d, in + len - 32u, 32) != 0) return RX_CM_ERR_DIGEST;
    Header h;
    memcpy(&h, in, sizeof h);
    if (h.magic != MAGIC || h.version != VERSION || h.d != RX_CM_D || h.cells != RX_CM_CELLS ||
        h.arms_cap != RX_CM_ARMS || h.cores != RX_CM_CORES || h.n_arms == 0 ||
        h.n_arms > RX_CM_ARMS)
        return RX_CM_ERR_FORMAT;
    RxCostModel t;
    memset(&t, 0, sizeof t);
    t.n_arms = h.n_arms;
    t.lambda = h.lambda;
    const uint8_t *p = in + sizeof h;
    memcpy(t.cell, p, sizeof t.cell);
    p += sizeof t.cell;
    memcpy(t.power_mw, p, sizeof t.power_mw);
    p += sizeof t.power_mw;
    memcpy(t.verify_ns, p, sizeof t.verify_ns);
    p += sizeof t.verify_ns;
    memcpy(t.synth_ns, p, sizeof t.synth_ns);
    p += sizeof t.synth_ns;
    memcpy(t.code_bytes, p, sizeof t.code_bytes);
    p += sizeof t.code_bytes;
    memcpy(&t.sd_scale, p, sizeof t.sd_scale);
    if (!(t.sd_scale >= 1.0 && t.sd_scale < 1e6)) return RX_CM_ERR_FORMAT;
    *m = t;
    return RX_CM_OK;
}

void rx_cm_digest(const RxCostModel *m, uint8_t out[32]) {
    uint8_t buf[sizeof(Header) + sizeof(((RxCostModel *)0)->cell) + 4096];
    size_t n = pack(m, buf);
    sha256_hash(buf, n, out);
}
