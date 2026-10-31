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
 * rx_costmodel_unit -- the empirical cost model on synthetic measurements.
 *
 * Any host. Checks fit, calibration of the predictive intervals on held-out
 * points, the decision rules (select, measure when unsure, frozen, budget,
 * eligibility, failures, fallback, latency/energy budgets) and the canonical
 * blob (round trip, tamper, truncation, version, cell count). An unsupported
 * semantic operation is refused. `make test-costmodel` restores the old
 * matvec clamp in a mutant; that binary must fail
 * unsupported-op-not-aliased-to-matvec.
 */
#include "runtime/rx_costmodel.h"
#include "sha256.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_fail;

#define CHECK(cond, ...) do {                                            \
        g_checks++;                                                      \
        if (!(cond)) {                                                   \
            g_fail++;                                                    \
            fprintf(stderr, "  FAIL %s:%d ", __FILE__, __LINE__);        \
            fprintf(stderr, __VA_ARGS__);                                \
            fputc('\n', stderr);                                         \
        }                                                                \
    } while (0)

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t next(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}
static double unif(void) { return (double)(next() >> 11) / 9007199254740992.0; }
static double gauss(void) {
    double u = unif(), v = unif();
    if (u < 1e-300) u = 1e-300;
    return sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v);
}

/* Synthetic truth: log2 ps for three arms, shaped like the Spark probe
 * (a fixed cost, a per-element cost, a thin-matrix penalty for arm 2). */
static int g_exact;   /* 1: truth lies in the model's basis */
static double truth(uint32_t arm, uint32_t M, uint32_t N, uint32_t core) {
    double w = g_exact ? log2((double)M) + log2((double)N) : log2((double)M * N + 1.0);
    double base = 10.0 + 0.95 * w + (core ? 0.4 : 0.0);
    if (arm == 1) base += 0.15;
    if (arm == 2) base += (N <= 3 ? 0.8 : -0.7);
    return base;
}

static RxCmFeatures feat(uint32_t M, uint32_t N, uint32_t core, uint32_t pressure) {
    RxCmFeatures f;
    memset(&f, 0, sizeof f);
    f.op = RX_CM_OP_MATVEC;
    f.M = M;
    f.N = N;
    f.state_bytes = ((uint64_t)M * N + M + N) * 8u;
    f.cache_bytes = 2u << 20;
    f.core = core;
    f.pressure = pressure;
    f.eligible = 0x7;
    f.need_evidence = 1;
    return f;
}

static const uint32_t TRAIN_M[] = { 1, 4, 16, 64, 256 };
/* Odd and even sizes, so "N <= 3" and "N % 4 != 0" are not confounded. */
static const uint32_t TRAIN_N[] = { 1, 2, 3, 5, 8, 12, 33, 64, 100, 256, 1024 };
#define N_TRAIN_N (sizeof TRAIN_N / sizeof TRAIN_N[0])

static void train(RxCostModel *m, double noise, uint32_t core, uint32_t pressure, int reps) {
    for (int r = 0; r < reps; r++)
        for (unsigned i = 0; i < 5; i++)
            for (unsigned j = 0; j < N_TRAIN_N; j++)
                for (uint32_t a = 0; a < 3; a++) {
                    RxCmFeatures f = feat(TRAIN_M[i], TRAIN_N[j], core, pressure);
                    double y = truth(a, f.M, f.N, core) + noise * gauss();
                    RxCmObservation o = { (uint64_t)exp2(y), 0 };
                    rx_cm_observe(m, &f, a, &o);
                }
}

static void t_basis(void) {
    RxCmFeatures f = feat(8, 3, RX_CM_CORE_A725, 1);
    f.state_bytes = 8u << 20;
    double phi[RX_CM_D];
    rx_cm_basis(&f, phi);
    CHECK(phi[0] == 1.0 && phi[1] == 3.0 && fabs(phi[2] - log2(3.0)) < 1e-12, "basis logs");
    CHECK(phi[5] == 1.0 && phi[6] == 1.0 && fabs(phi[7] - 2.0) < 1e-12, "basis indicators");
    CHECK(rx_cm_cell(&f) == 1u * RX_CM_PRESSURE + 1u, "cell index");
    f.core = 99;
    f.pressure = 0;
    CHECK(rx_cm_cell(&f) == (RX_CM_OP_MATVEC * RX_CM_CORES + RX_CM_CORE_OTHER) * RX_CM_PRESSURE,
          "out-of-range core is the OTHER class");
    f.core = RX_CM_CORE_A725;
    f.pressure = 99;
    CHECK(rx_cm_cell(&f) == (RX_CM_OP_MATVEC * RX_CM_CORES + RX_CM_CORE_A725) * RX_CM_PRESSURE +
                                (RX_CM_PRESSURE - 1u),
          "out-of-range pressure saturates to the last bucket");
    f.op = RX_CM_OPS;
    CHECK(rx_cm_cell(&f) == RX_CM_CELL_NONE && RX_CM_CELL_NONE >= RX_CM_CELLS,
          "first unsupported op is not a cell index");
    f.op = UINT32_MAX;
    CHECK(rx_cm_cell(&f) == RX_CM_CELL_NONE, "UINT32_MAX op is not a cell index");
}

static void t_fit_and_calibration(int exact, double *cov80, double *cov95, double *rmse) {
    g_exact = exact;
    RxCostModel m;
    rx_cm_init(&m, 3);
    const double noise = 0.08;
    train(&m, noise, RX_CM_CORE_X925, 0, 2);
    uint32_t in80 = 0, in95 = 0, n = 0;
    double se = 0.0;
    double q80 = 0, q95 = 0;
    for (int k = 0; k < 4000; k++) {
        /* held out: shapes not on the training grid */
        uint32_t M = 1 + (uint32_t)(next() % 300);
        uint32_t N = 1 + (uint32_t)(next() % 1000);
        uint32_t a = (uint32_t)(next() % 3);
        RxCmFeatures f = feat(M, N, RX_CM_CORE_X925, 0);
        RxCmPrediction p;
        rx_cm_predict(&m, &f, a, &p);
        double y = truth(a, M, N, 0) + noise * gauss();
        double z = fabs(y - p.mean_log2_ps);
        q80 = rx_cm_t_quantile(0.80, p.dof) * p.sd_log2;
        q95 = rx_cm_t_quantile(0.95, p.dof) * p.sd_log2;
        in80 += z <= q80;
        in95 += z <= q95;
        se += (y - p.mean_log2_ps) * (y - p.mean_log2_ps);
        n++;
    }
    *cov80 = (double)in80 / n;
    *cov95 = (double)in95 / n;
    *rmse = sqrt(se / n);
    printf("    held-out synthetic (%s): rmse %.3f log2 (noise %.2f), coverage 80%%: %.3f, 95%%: %.3f\n",
           exact ? "truth in basis" : "truth outside basis", *rmse, noise, *cov80, *cov95);
    g_exact = 0;
    CHECK(*rmse < 0.2, "held-out error %.3f", *rmse);
    if (exact) {
        CHECK(*cov80 > 0.74 && *cov80 < 0.86, "80%% interval covers %.3f", *cov80);
        CHECK(*cov95 > 0.92 && *cov95 < 0.98, "95%% interval covers %.3f", *cov95);
    } else {   /* misspecified: may be wide, must not be overconfident */
        CHECK(*cov80 > 0.74, "80%% interval covers %.3f", *cov80);
        CHECK(*cov95 > 0.92, "95%% interval covers %.3f", *cov95);
    }
}

static void t_decide(void) {
    RxCmPolicy pol;
    rx_cm_default_policy(&pol);
    RxCostModel m;
    rx_cm_init(&m, 3);
    RxCmDecision d;
    RxCmFeatures f = feat(64, 256, RX_CM_CORE_X925, 0);

    rx_cm_decide(&m, &pol, &f, 1, &d);
    CHECK(d.action == RX_CM_MEASURE && d.why == RX_CM_WHY_UNKNOWN_ARM, "empty model measures");
    rx_cm_decide(&m, &pol, &f, 0, &d);
    CHECK(d.action == RX_CM_FALLBACK && d.arm == 0, "empty frozen model falls back");

    train(&m, 0.03, RX_CM_CORE_X925, 0, 2);
    rx_cm_decide(&m, &pol, &f, 1, &d);
    CHECK(d.action == RX_CM_SELECT && d.arm == 2 && d.why == RX_CM_WHY_CONFIDENT,
          "wide matrix: arm 2 selected confidently (action %d arm %u why %u)", d.action, d.arm,
          d.why);
    f = feat(256, 1, RX_CM_CORE_X925, 0);
    rx_cm_decide(&m, &pol, &f, 1, &d);
    CHECK(d.action == RX_CM_SELECT && d.arm == 0, "thin matrix: arm 0 selected (arm %u)", d.arm);

    /* two arms predicted equal: worth a run only if the model is unsure */
    RxCostModel c, noisy;
    rx_cm_init(&c, 2);
    rx_cm_init(&noisy, 2);
    RxCmFeatures g = feat(16, 16, RX_CM_CORE_X925, 0);
    g.eligible = 0x3;
    for (int r = 0; r < 30; r++) {
        RxCmObservation o0 = { (uint64_t)exp2(20.0 + 0.02 * gauss()), 0 };
        RxCmObservation o1 = { (uint64_t)exp2(20.01 + 0.02 * gauss()), 0 };
        rx_cm_observe(&c, &g, 0, &o0);
        rx_cm_observe(&c, &g, 1, &o1);
        RxCmObservation n0 = { (uint64_t)exp2(20.0 + 0.4 * gauss()), 0 };
        RxCmObservation n1 = { (uint64_t)exp2(20.0 + 0.4 * gauss()), 0 };
        rx_cm_observe(&noisy, &g, 0, &n0);
        rx_cm_observe(&noisy, &g, 1, &n1);
    }
    rx_cm_decide(&c, &pol, &g, 1, &d);
    CHECK(d.action == RX_CM_SELECT && d.why == RX_CM_WHY_CONFIDENT,
          "equal and tight: not worth a run (action %d, ei %.3f)", d.action, d.expected_improvement);
    rx_cm_decide(&noisy, &pol, &g, 1, &d);
    CHECK(d.action == RX_CM_MEASURE && d.why == RX_CM_WHY_WORTH_MEASURING,
          "equal and unsure: measured (action %d, ei %.3f)", d.action, d.expected_improvement);
    rx_cm_decide(&noisy, &pol, &g, 0, &d);
    CHECK(d.action == RX_CM_SELECT, "frozen: never measures");
    rx_cm_charge(&noisy, 0, pol.explore_allow_ps + 1);
    rx_cm_decide(&noisy, &pol, &g, 1, &d);
    CHECK(d.action == RX_CM_SELECT && d.why == RX_CM_WHY_BUDGET_SPENT,
          "spent budget: select instead of measure");
    rx_cm_charge(&noisy, (uint64_t)((double)(pol.explore_allow_ps + 1) / pol.explore_frac) * 2, 0);
    rx_cm_decide(&noisy, &pol, &g, 1, &d);
    CHECK(d.action == RX_CM_MEASURE, "budget refills as chosen work accrues");

    /* unseen pressure: borrowed and widened */
    f = feat(64, 256, RX_CM_CORE_X925, 2);
    RxCmPrediction p0, p2;
    RxCmFeatures f0 = feat(64, 256, RX_CM_CORE_X925, 0);
    rx_cm_predict(&m, &f0, 2, &p0);
    rx_cm_predict(&m, &f, 2, &p2);
    CHECK(p2.borrowed && !p0.borrowed && p2.mean_log2_ps == p0.mean_log2_ps &&
              fabs(p2.sd_log2 - sqrt(p0.sd_log2 * p0.sd_log2 + RX_CM_UNSEEN_SHIFT * RX_CM_UNSEEN_SHIFT)) < 1e-9,
          "unseen pressure: quiet prediction, widened by the unseen-shift prior");

    /* Pressure learned as a correction: arm 2 slows by 1.5 doublings, arm 0
     * by 0.2. A handful of observations flips the choice. */
    RxCostModel pm = m;
    RxCmFeatures pf[6];
    for (int i = 0; i < 6; i++) {
        pf[i] = feat(TRAIN_M[i % 5], TRAIN_N[(i * 3) % N_TRAIN_N], RX_CM_CORE_X925, 1);
        for (uint32_t a = 0; a < 3; a++) {
            double shift = a == 2 ? 1.5 : 0.2;
            RxCmObservation o = { (uint64_t)exp2(truth(a, pf[i].M, pf[i].N, 0) + shift + 0.03 * gauss()), 0 };
            CHECK(rx_cm_observe(&pm, &pf[i], a, &o) == RX_CM_OK, "pressure observation");
        }
    }
    RxCmFeatures q1 = feat(64, 256, RX_CM_CORE_X925, 1);
    RxCmPrediction pq;
    rx_cm_predict(&pm, &q1, 2, &pq);
    CHECK(pq.known && !pq.borrowed && fabs(pq.mean_log2_ps - (p0.mean_log2_ps + 1.5)) < 0.15,
          "pressure correction learned from 6 points (%.2f vs %.2f)", pq.mean_log2_ps,
          p0.mean_log2_ps + 1.5);
    rx_cm_decide(&pm, &pol, &q1, 0, &d);
    CHECK(d.arm == 0, "under pressure the choice flips to arm 0 (got %u)", d.arm);
    RxCostModel bare;
    rx_cm_init(&bare, 3);
    RxCmObservation ob = { 1000, 0 };
    CHECK(rx_cm_observe(&bare, &q1, 0, &ob) == RX_CM_SKIPPED, "no quiet model: pressure data skipped");
    rx_cm_decide(&m, &pol, &f, 0, &d);
    CHECK(d.action == RX_CM_SELECT && d.arm == 2, "frozen uses the borrowed prediction");
    RxCostModel mid;   /* close, moderately sure: not worth a run here, worth one when borrowed */
    rx_cm_init(&mid, 2);
    for (int r = 0; r < 40; r++) {
        RxCmObservation o0 = { (uint64_t)exp2(20.0 + 0.1 * gauss()), 0 };
        RxCmObservation o1 = { (uint64_t)exp2(20.0 + 0.1 * gauss()), 0 };
        rx_cm_observe(&mid, &g, 0, &o0);
        rx_cm_observe(&mid, &g, 1, &o1);
    }
    rx_cm_decide(&mid, &pol, &g, 1, &d);
    CHECK(d.action == RX_CM_SELECT, "close and moderately sure: select (ei %.3f)", d.expected_improvement);
    RxCmFeatures gp = g;
    gp.pressure = 1;
    rx_cm_decide(&mid, &pol, &gp, 1, &d);
    CHECK(d.action == RX_CM_MEASURE && d.why == RX_CM_WHY_WORTH_MEASURING,
          "same arms under unseen pressure: widened spread makes it worth a run (ei %.3f)",
          d.expected_improvement);

    /* unseen core class with no data anywhere: fallback or measure */
    f = feat(64, 256, RX_CM_CORE_A725, 0);
    rx_cm_decide(&m, &pol, &f, 0, &d);
    CHECK(d.action == RX_CM_FALLBACK && d.arm == 0, "no data for this core class: fallback");
}

static void t_eligibility_property(void) {
    RxCmPolicy pol;
    rx_cm_default_policy(&pol);
    RxCostModel m;
    rx_cm_init(&m, 4);
    /* arm 3 would be the fastest by far, but is never eligible */
    for (int r = 0; r < 3; r++)
        for (unsigned i = 0; i < 5; i++)
            for (unsigned j = 0; j < N_TRAIN_N; j++) {
                RxCmFeatures f = feat(TRAIN_M[i], TRAIN_N[j], RX_CM_CORE_X925, 0);
                for (uint32_t a = 0; a < 3; a++) {
                    RxCmObservation o = { (uint64_t)exp2(truth(a, f.M, f.N, 0) + 0.05 * gauss()), 0 };
                    rx_cm_observe(&m, &f, a, &o);
                }
                RxCmObservation o3 = { (uint64_t)exp2(truth(0, f.M, f.N, 0) - 3.0), 0 };
                rx_cm_observe(&m, &f, 3, &o3);
            }
    uint32_t touched = 0, decisions = 0;
    for (int k = 0; k < 20000; k++) {
        RxCmFeatures f = feat(1 + (uint32_t)(next() % 2000), 1 + (uint32_t)(next() % 4000),
                              (uint32_t)(next() % 3), (uint32_t)(next() % 3));
        f.eligible = 0x7;
        f.latency_budget_ps = (next() & 1) ? (uint64_t)exp2(10 + (double)(next() % 20)) : 0;
        RxCmDecision d;
        rx_cm_decide(&m, &pol, &f, (int)(next() & 1), &d);
        decisions++;
        if (d.arm == 3 || (d.action == RX_CM_MEASURE && (d.measure[0] == 3 || d.measure[1] == 3)))
            touched++;
    }
    CHECK(touched == 0, "an ineligible arm was chosen or measured %u times in %u decisions",
          touched, decisions);

    /* failures exclude an arm; the fallback is never excluded */
    RxCmFeatures f = feat(64, 256, RX_CM_CORE_X925, 0);
    RxCmDecision d;
    rx_cm_decide(&m, &pol, &f, 0, &d);
    uint32_t fast = d.arm;
    CHECK(fast == 2, "before failures, arm 2 wins");
    RxCmObservation bad = { 0, 1 };
    for (int i = 0; i < 3; i++) rx_cm_observe(&m, &f, 2, &bad);
    rx_cm_decide(&m, &pol, &f, 0, &d);
    CHECK(d.arm != 2 && d.pred[2].fail_p > pol.max_fail_p, "a failing arm is no longer selected");
    for (int i = 0; i < 400; i++) rx_cm_observe(&m, &f, 0, &bad);
    f.eligible = 0x1;
    rx_cm_decide(&m, &pol, &f, 1, &d);
    CHECK(d.action == RX_CM_FALLBACK && d.arm == 0 && d.why == RX_CM_WHY_NO_ELIGIBLE,
          "the fallback stays available");
}

static void t_budgets(void) {
    RxCmPolicy pol;
    rx_cm_default_policy(&pol);
    RxCostModel m;
    rx_cm_init(&m, 2);
    RxCmFeatures f = feat(32, 32, RX_CM_CORE_X925, 0);
    /* arm 0: 2^20 ps, tight. arm 1: 2^19.8 ps, noisy. */
    for (int r = 0; r < 40; r++) {
        RxCmObservation o0 = { (uint64_t)exp2(20.0 + 0.02 * gauss()), 0 };
        RxCmObservation o1 = { (uint64_t)exp2(19.8 + 0.6 * gauss()), 0 };
        rx_cm_observe(&m, &f, 0, &o0);
        rx_cm_observe(&m, &f, 1, &o1);
    }
    RxCmDecision d;
    f.eligible = 0x3;
    rx_cm_decide(&m, &pol, &f, 0, &d);
    uint32_t free_pick = d.arm;
    f.latency_budget_ps = (uint64_t)exp2(20.2);
    rx_cm_decide(&m, &pol, &f, 0, &d);
    CHECK(d.arm == 0 && d.why == RX_CM_WHY_CONFIDENT,
          "latency budget picks the arm whose 90%% bound fits (free pick %u, got %u)", free_pick,
          d.arm);
    f.latency_budget_ps = 1000;
    rx_cm_decide(&m, &pol, &f, 0, &d);
    CHECK(d.action == RX_CM_SELECT && d.why == RX_CM_WHY_OVER_BUDGET, "no arm fits: reported");
    f.latency_budget_ps = 0;
    m.power_mw[RX_CM_CORE_X925][0] = 1000.0;
    m.power_mw[RX_CM_CORE_X925][1] = 5000.0;
    RxCmPrediction p0, p1;
    rx_cm_predict(&m, &f, 0, &p0);
    rx_cm_predict(&m, &f, 1, &p1);
    f.energy_budget_pj = (uint64_t)((p0.energy_pj + p1.energy_pj) / 2.0);
    rx_cm_decide(&m, &pol, &f, 0, &d);
    CHECK(d.arm == 0, "energy budget excludes the hungrier arm");
    CHECK(fabs(p0.energy_pj - exp2(p0.mean_log2_ps) * 1.0) < 1e-6 * p0.energy_pj,
          "energy = predicted ps x measured mW");
}

static void t_calibrate(void) {
    /* Trained on quiet data; later the machine drifts (extra 0.15 log2). */
    RxCostModel m;
    rx_cm_init(&m, 3);
    g_exact = 1;
    train(&m, 0.03, RX_CM_CORE_X925, 0, 2);
    RxCmFeatures cf[600];
    uint32_t ca[600];
    uint64_t cps[600];
    for (int i = 0; i < 600; i++) {
        cf[i] = feat(1 + (uint32_t)(next() % 300), 1 + (uint32_t)(next() % 1000), RX_CM_CORE_X925, 0);
        ca[i] = (uint32_t)(next() % 3);
        cps[i] = (uint64_t)exp2(truth(ca[i], cf[i].M, cf[i].N, 0) + 0.03 * gauss() + 0.15 * gauss());
    }
    uint32_t before = 0, after = 0, n = 0;
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            double s = rx_cm_calibrate(&m, cf, ca, cps, 300, 0.80);
            printf("    calibration scale after drift: %.2f\n", s);
            CHECK(s > 1.5, "drift widens the spread (%.2f)", s);
        }
        for (int i = 300; i < 600; i++) {
            RxCmPrediction p;
            rx_cm_predict(&m, &cf[i], ca[i], &p);
            double z = fabs(log2((double)cps[i]) - p.mean_log2_ps);
            int in = z <= rx_cm_t_quantile(0.80, p.dof) * p.sd_log2;
            if (pass == 0) before += in;
            else after += in;
            n += pass == 0;
        }
    }
    g_exact = 0;
    printf("    80%% coverage on later, drifted data: %.3f before calibration, %.3f after\n",
           (double)before / n, (double)after / n);
    CHECK((double)before / n < 0.6, "uncalibrated model is overconfident under drift");
    CHECK((double)after / n > 0.72 && (double)after / n < 0.88, "calibrated coverage %.3f",
          (double)after / n);
    RxCostModel few;
    rx_cm_init(&few, 3);
    CHECK(rx_cm_calibrate(&few, cf, ca, cps, 300, 0.8) == 0.0 && few.sd_scale == 1.0,
          "no usable predictions: scale unchanged");
}

static void t_blob(void) {
    RxCostModel m, r;
    rx_cm_init(&m, 5);
    train(&m, 0.05, RX_CM_CORE_A725, 1, 1);
    m.power_mw[1][2] = 1234.5;
    m.verify_ns[2] = 77;
    m.synth_ns[2] = 88;
    m.code_bytes[2] = 99;
    m.explore_ps = 5;
    m.sd_scale = 1.7;
    size_t n = rx_cm_blob_size(), got = 0;
    printf("    model blob: %zu bytes\n", n);
    CHECK(n <= 65536u, "fits the generation store blob limit");
    uint8_t *a = malloc(n), *b = malloc(n);
    CHECK(rx_cm_serialize(&m, a, n, &got) == RX_CM_OK && got == n, "serialize");
    CHECK(rx_cm_deserialize(&r, a, n) == RX_CM_OK, "deserialize");
    CHECK(r.explore_ps == 0 && r.work_ps == 0, "runtime counters are not durable");
    CHECK(rx_cm_serialize(&r, b, n, &got) == RX_CM_OK && memcmp(a, b, n) == 0,
          "round trip is byte-identical");
    uint8_t d1[32], d2[32];
    rx_cm_digest(&m, d1);
    rx_cm_digest(&r, d2);
    CHECK(memcmp(d1, d2, 32) == 0, "digest survives the round trip");
    RxCmFeatures f = feat(100, 30, RX_CM_CORE_A725, 1);
    RxCmPrediction p, q;
    rx_cm_predict(&m, &f, 1, &p);
    rx_cm_predict(&r, &f, 1, &q);
    CHECK(p.mean_log2_ps == q.mean_log2_ps && p.sd_log2 == q.sd_log2, "same predictions");
    CHECK(r.sd_scale == 1.7, "calibration scale is durable");
    memcpy(b, a, n);
    b[100] ^= 1;
    CHECK(rx_cm_deserialize(&r, b, n) == RX_CM_ERR_DIGEST, "tampered body refused");
    CHECK(rx_cm_deserialize(&r, a, n - 1) == RX_CM_ERR_FORMAT, "truncated blob refused");
    memcpy(b, a, n);
    b[4] = 9;   /* version */
    sha256_hash(b, n - 32, b + n - 32);
    CHECK(rx_cm_deserialize(&r, b, n) == RX_CM_ERR_FORMAT, "unknown version refused");
    memcpy(b, a, n);
    /* Header.cells is the fifth uint32. A wider operation family changes it. */
    uint32_t cells = 0;
    memcpy(&cells, b + 16, sizeof cells);
    cells += 1u;
    memcpy(b + 16, &cells, sizeof cells);
    sha256_hash(b, n - 32, b + n - 32);
    CHECK(rx_cm_deserialize(&r, b, n) == RX_CM_ERR_FORMAT, "cell-count mismatch refused");
    free(a);
    free(b);
}

/* Issue 704: an unsupported semantic operation used to clamp onto matvec, so a
 * learned matvec prediction was reported for it and its observations trained
 * that cell. `make test-costmodel` rebuilds rx_costmodel.c with that clamp
 * restored; the named assertion unsupported-op-not-aliased-to-matvec must fail.
 * Decision modes for a supported operation are asserted here without the shared
 * rng. t_decide, t_budgets and t_eligibility cover the same modes on the
 * synthetic fit. */
static RxCmFeatures repro_features(uint32_t op) {
    RxCmFeatures f;
    memset(&f, 0, sizeof f);
    f.op = op;
    f.M = 64;
    f.N = 64;
    f.state_bytes = (64ull * 64ull + 64ull + 64ull) * 8ull;
    f.cache_bytes = 1048576ull;
    f.core = RX_CM_CORE_X925;
    f.eligible = 1;
    f.need_evidence = 1;
    return f;
}

static void t_refuse_unsupported_op(void) {
    RxCostModel m;
    rx_cm_init(&m, 1);
    RxCmFeatures good = repro_features(RX_CM_OP_MATVEC);
    RxCmObservation seen = { 1000000ull, 0 };
    for (int i = 0; i < 20; i++)
        CHECK(rx_cm_observe(&m, &good, 0, &seen) == RX_CM_OK, "matvec observation accepted");
    m.explore_ps = 42;
    m.work_ps = 99;

    RxCmPrediction learned;
    int lr = rx_cm_predict(&m, &good, 0, &learned);
    CHECK(lr == RX_CM_OK && learned.known && !learned.refused && learned.n == 20 &&
              learned.mean_log2_ps > 19.0 && learned.mean_log2_ps < 21.0,
          "valid matvec prediction is learned (rc %d known %d n %u mean %.6f)", lr, learned.known,
          learned.n, learned.mean_log2_ps);

    RxCmFeatures bad = good;
    bad.op = UINT32_MAX;
    RxCmPrediction aliased;
    int ur = rx_cm_predict(&m, &bad, 0, &aliased);
    RxCostModel snap = m;
    RxCmObservation failed = { 1000000ull, 1 };
    RxCmObservation zero_fail = { 0, 1 };
    int orc = rx_cm_observe(&m, &bad, 0, &seen);
    int frc = rx_cm_observe(&m, &bad, 0, &failed);
    int zrc = rx_cm_observe(&m, &bad, 0, &zero_fail);
    CHECK(ur == RX_CM_ERR_UNSUPPORTED && aliased.refused && !aliased.known && !aliased.borrowed &&
              aliased.mean_log2_ps == 0.0 && aliased.n == 0 && aliased.fail_p == 0.0 &&
              rx_cm_cell(&bad) == RX_CM_CELL_NONE &&
              orc == RX_CM_ERR_UNSUPPORTED && frc == RX_CM_ERR_UNSUPPORTED &&
              zrc == RX_CM_ERR_UNSUPPORTED && memcmp(&m, &snap, sizeof m) == 0 &&
              m.cell[0][0].n == 20 && m.cell[0][0].trials == 20 && m.cell[0][0].failures == 0,
          "unsupported-op-not-aliased-to-matvec");

    size_t blob_n = rx_cm_blob_size(), got = 0;
    uint8_t *blob0 = malloc(blob_n), *blob1 = malloc(blob_n);
    CHECK(blob0 && blob1 && rx_cm_serialize(&snap, blob0, blob_n, &got) == RX_CM_OK && got == blob_n,
          "serialize before rejected observations");
    CHECK(blob0 && blob1 && rx_cm_serialize(&m, blob1, blob_n, &got) == RX_CM_OK &&
              memcmp(blob0, blob1, blob_n) == 0,
          "frozen model serialization identical after rejected observations");
    free(blob0);
    free(blob1);

    bad.op = RX_CM_OPS;
    RxCmPrediction first;
    int fr = rx_cm_predict(&m, &bad, 0, &first);
    snap = m;
    orc = rx_cm_observe(&m, &bad, 0, &seen);
    frc = rx_cm_observe(&m, &bad, 0, &failed);
    CHECK(fr == RX_CM_ERR_UNSUPPORTED && first.refused && !first.known &&
              rx_cm_cell(&bad) == RX_CM_CELL_NONE && orc == RX_CM_ERR_UNSUPPORTED &&
              frc == RX_CM_ERR_UNSUPPORTED && memcmp(&m, &snap, sizeof m) == 0,
          "first unsupported id RX_CM_OPS is refused");

    /* A supported observation still trains matvec after the refusals. */
    CHECK(rx_cm_observe(&m, &good, 0, &seen) == RX_CM_OK && m.cell[0][0].n == 21,
          "supported matvec observation still trains");

    RxCmPolicy pol;
    rx_cm_default_policy(&pol);
    RxCmDecision d;
    bad.op = UINT32_MAX;
    rx_cm_decide(&m, &pol, &bad, 1, &d);
    CHECK(d.action == RX_CM_REFUSE && d.why == RX_CM_WHY_UNSUPPORTED_OP && d.arm == m.n_arms &&
              d.measure[0] == m.n_arms && d.measure[1] == m.n_arms && d.expected_improvement == 0.0 &&
              d.pred[0].refused && !d.pred[0].known && !d.pred[0].borrowed &&
              d.pred[0].mean_log2_ps == 0.0,
          "decide refuses UINT32_MAX without ranking");
    rx_cm_decide(&m, &pol, &bad, 0, &d);
    CHECK(d.action == RX_CM_REFUSE && d.why == RX_CM_WHY_UNSUPPORTED_OP,
          "frozen decide refuses UINT32_MAX");
    bad.op = RX_CM_OPS;
    rx_cm_decide(&m, &pol, &bad, 1, &d);
    CHECK(d.action == RX_CM_REFUSE && d.why == RX_CM_WHY_UNSUPPORTED_OP && d.pred[0].refused,
          "decide refuses RX_CM_OPS");

    /* Pressure on a real matvec is still a residual. Unsupported pressure is not. */
    RxCostModel quiet;
    rx_cm_init(&quiet, 1);
    RxCmFeatures qf = repro_features(RX_CM_OP_MATVEC);
    RxCmObservation exact = { (uint64_t)exp2(20.0), 0 };
    for (int i = 0; i < 16; i++) rx_cm_observe(&quiet, &qf, 0, &exact);
    RxCmFeatures unseen = qf;
    unseen.pressure = 2;
    RxCmPrediction pq, p0;
    rx_cm_predict(&quiet, &qf, 0, &p0);
    rx_cm_predict(&quiet, &unseen, 0, &pq);
    CHECK(pq.borrowed && !pq.refused && pq.mean_log2_ps == p0.mean_log2_ps,
          "unseen matvec pressure borrows the quiet prediction");
    RxCmFeatures pressed = qf;
    pressed.pressure = 1;
    RxCostModel bare;
    rx_cm_init(&bare, 1);
    CHECK(rx_cm_observe(&bare, &pressed, 0, &exact) == RX_CM_SKIPPED &&
              bare.cell[rx_cm_cell(&pressed)][0].trials == 1 &&
              bare.cell[rx_cm_cell(&pressed)][0].n == 0,
          "valid matvec pressure without a quiet model is skipped");
    pressed.op = RX_CM_OPS;
    RxCostModel bare_snap = bare;
    CHECK(rx_cm_observe(&bare, &pressed, 0, &exact) == RX_CM_ERR_UNSUPPORTED &&
              memcmp(&bare, &bare_snap, sizeof bare) == 0,
          "unsupported op under pressure is refused, not skipped into a bucket");
    pressed = qf;
    pressed.pressure = 1;
    RxCmObservation slow = { (uint64_t)exp2(21.0), 0 };
    for (int i = 0; i < 6; i++)
        CHECK(rx_cm_observe(&quiet, &pressed, 0, &slow) == RX_CM_OK, "pressure residual accepted");
    uint64_t residual_n = quiet.cell[rx_cm_cell(&pressed)][0].n;
    CHECK(residual_n == 6, "pressure residual has its own samples");
    rx_cm_predict(&quiet, &pressed, 0, &pq);
    CHECK(pq.known && !pq.borrowed && !pq.refused && fabs(pq.mean_log2_ps - (p0.mean_log2_ps + 1.0)) < 0.25,
          "pressure correction learned (%.3f vs %.3f)", pq.mean_log2_ps, p0.mean_log2_ps + 1.0);
    pressed.op = UINT32_MAX;
    snap = quiet;
    CHECK(rx_cm_observe(&quiet, &pressed, 0, &slow) == RX_CM_ERR_UNSUPPORTED &&
              memcmp(&quiet, &snap, sizeof quiet) == 0,
          "unsupported op does not train the pressure residual");

    /* OTHER core and a saturated pressure bucket stay real cells. */
    RxCostModel other_m;
    rx_cm_init(&other_m, 1);
    RxCmFeatures other = qf;
    other.core = 99;
    other.pressure = 0;
    uint32_t oc = rx_cm_cell(&other);
    CHECK(oc < RX_CM_CELLS && oc != rx_cm_cell(&qf), "OTHER core is a real cell");
    CHECK(rx_cm_observe(&other_m, &other, 0, &exact) == RX_CM_OK && other_m.cell[oc][0].n == 1 &&
              other_m.cell[0][0].n == 0,
          "OTHER core observation stays out of the matvec X925 cell");
    other.pressure = 99;
    uint32_t pc = rx_cm_cell(&other);
    CHECK(pc < RX_CM_CELLS && pc != oc && pc != RX_CM_CELL_NONE,
          "saturated pressure on OTHER is its own bucket");

    /* Each decision mode on a supported matvec, then refusal beside them. */
    RxCostModel empty;
    rx_cm_init(&empty, 2);
    RxCmFeatures g = feat(16, 16, RX_CM_CORE_X925, 0);
    g.eligible = 0x3;
    rx_cm_decide(&empty, &pol, &g, 1, &d);
    CHECK(d.action == RX_CM_MEASURE && d.why == RX_CM_WHY_UNKNOWN_ARM, "empty model measures");
    rx_cm_decide(&empty, &pol, &g, 0, &d);
    CHECK(d.action == RX_CM_FALLBACK && d.arm == 0 && d.why == RX_CM_WHY_UNKNOWN_ARM,
          "empty frozen model falls back");
    RxCostModel one;
    rx_cm_init(&one, 1);
    rx_cm_decide(&one, &pol, &g, 1, &d);
    CHECK(d.action == RX_CM_FALLBACK && d.why == RX_CM_WHY_NO_ELIGIBLE, "only the fallback is eligible");

    RxCostModel tight;
    rx_cm_init(&tight, 2);
    RxCmObservation hi = { (uint64_t)exp2(22.0), 0 };
    for (int i = 0; i < 40; i++) {
        rx_cm_observe(&tight, &g, 0, &exact);
        rx_cm_observe(&tight, &g, 1, &hi);
    }
    rx_cm_decide(&tight, &pol, &g, 1, &d);
    CHECK(d.action == RX_CM_SELECT && d.arm == 0 && d.why == RX_CM_WHY_CONFIDENT,
          "separated arms select confidently (action %d arm %u why %u)", d.action, d.arm, d.why);
    g.latency_budget_ps = 1000;
    rx_cm_decide(&tight, &pol, &g, 0, &d);
    CHECK(d.action == RX_CM_SELECT && d.why == RX_CM_WHY_OVER_BUDGET, "no arm fits the latency budget");
    g.latency_budget_ps = 0;

    RxCostModel equal;
    rx_cm_init(&equal, 2);
    for (int i = 0; i < 40; i++) {
        rx_cm_observe(&equal, &g, 0, &exact);
        rx_cm_observe(&equal, &g, 1, &exact);
    }
    rx_cm_decide(&equal, &pol, &g, 1, &d);
    CHECK(d.action == RX_CM_SELECT && d.why == RX_CM_WHY_CONFIDENT,
          "equal and tight: not worth a run (action %d ei %.3f)", d.action, d.expected_improvement);
    RxCmFeatures gp = g;
    gp.pressure = 1;
    rx_cm_decide(&equal, &pol, &gp, 1, &d);
    CHECK(d.action == RX_CM_MEASURE && d.why == RX_CM_WHY_WORTH_MEASURING,
          "unseen pressure is worth measuring (action %d why %u ei %.3f)", d.action, d.why,
          d.expected_improvement);
    rx_cm_charge(&equal, 0, pol.explore_allow_ps + 1);
    rx_cm_decide(&equal, &pol, &gp, 1, &d);
    CHECK(d.action == RX_CM_SELECT && d.why == RX_CM_WHY_BUDGET_SPENT,
          "spent exploration budget selects (action %d why %u)", d.action, d.why);
    rx_cm_charge(&equal, (uint64_t)((double)(pol.explore_allow_ps + 1) / pol.explore_frac) * 2, 0);
    rx_cm_decide(&equal, &pol, &gp, 1, &d);
    CHECK(d.action == RX_CM_MEASURE && d.why == RX_CM_WHY_WORTH_MEASURING,
          "budget refills as chosen work accrues");
    gp.op = RX_CM_OPS;
    rx_cm_decide(&equal, &pol, &gp, 1, &d);
    CHECK(d.action == RX_CM_REFUSE && d.why == RX_CM_WHY_UNSUPPORTED_OP && d.pred[0].refused &&
              d.pred[1].refused && !d.pred[0].known && !d.pred[1].borrowed,
          "unsupported op is not ranked in any decision mode");

    RxCmFeatures cal[16];
    uint32_t carm[16];
    uint64_t cps[16];
    for (int i = 0; i < 16; i++) {
        cal[i] = repro_features(i & 1 ? UINT32_MAX : RX_CM_OPS);
        carm[i] = 0;
        cps[i] = (uint64_t)exp2(30.0);
    }
    double scale_before = equal.sd_scale;
    double scale = rx_cm_calibrate(&equal, cal, carm, cps, 16, 0.80);
    CHECK(scale == 0.0 && equal.sd_scale == scale_before,
          "unsupported calibration points do not change the scale");
}

int main(int argc, char **argv) {
    int only_refuse = argc > 1 && strcmp(argv[1], "refuse") == 0;
    if (!only_refuse) {
        double c80, c95, rmse;
        t_basis();
        t_fit_and_calibration(1, &c80, &c95, &rmse);
        t_fit_and_calibration(0, &c80, &c95, &rmse);
        t_decide();
        t_eligibility_property();
        t_budgets();
        t_calibrate();
        t_blob();
    }
    t_refuse_unsupported_op();
    printf("checks %d failures %d\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
