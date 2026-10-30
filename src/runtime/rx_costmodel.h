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
 * rx_costmodel.h -- Omega's empirical cost model (OMEGA_EMPIRICAL_OPTIMIZER).
 *
 * Omega picks a realization for an operation from what it has measured, not
 * from a fixed rule. This is a calculator only:
 *
 *   features of a call  ->  predicted cost of every eligible realization,
 *                           with how sure the prediction is
 *                       ->  SELECT one, or MEASURE two when unsure
 *
 * It never runs code, never holds or reads authority, never writes a
 * generation. Whoever calls it executes the realizations and reports what
 * they cost. A learned model becomes durable only as the `model` blob of a
 * generation promoted through the R9 barrier by a subject that holds the
 * promotion right; the Makefile checks that rx_costmodel.o references none
 * of those operations.
 *
 * This is not authority learning. No feature is derived from a capability,
 * and a realization is eligible only if the caller says it was verified.
 *
 * Predictor: one Bayesian linear regression (normal-inverse-gamma, ridge
 * prior) per (cell, arm), on log2 picoseconds per call. A cell is
 * (operation class, core class, pressure bucket). Pressure cells hold a
 * two-number correction over the quiet cell of the same core class. The basis is small and
 * readable: 1, log2 M, log2 N, their product, (log2 N)^2, N <= 3,
 * N % 4 != 0, and how far the working set overflows the core's cache, in
 * doublings. Each (cell, arm) keeps only sufficient statistics, so the model
 * serializes to a fixed-size blob.
 */
#ifndef RX_COSTMODEL_H
#define RX_COSTMODEL_H

#include <stddef.h>
#include <stdint.h>

#define RX_CM_ARMS       8u
#define RX_CM_OPS        1u     /* matvec only today; widen with a second operation */
#define RX_CM_CORES      3u
#define RX_CM_PRESSURE   3u
#define RX_CM_CELLS      (RX_CM_OPS * RX_CM_CORES * RX_CM_PRESSURE)
#define RX_CM_D          8u        /* basis size */
#define RX_CM_TRI        (RX_CM_D * (RX_CM_D + 1u) / 2u)
#define RX_CM_MIN_OBS    12u       /* own-cell observations before a cell is used alone */
#define RX_CM_RES_MIN_OBS 4u      /* observations before a pressure correction is used */
#define RX_CM_UNSEEN_SHIFT 1.0    /* prior sd, log2: an unseen pressure may halve or double cost */

#define RX_CM_OK          0
#define RX_CM_ERR_ARG    -1
#define RX_CM_ERR_FORMAT -2
#define RX_CM_ERR_DIGEST -3
#define RX_CM_SKIPPED     1       /* observe: pressure data with no quiet model to correct */
enum { RX_CM_OP_MATVEC = 0 };
enum { RX_CM_CORE_X925 = 0, RX_CM_CORE_A725 = 1, RX_CM_CORE_OTHER = 2 };

/* What the caller knows about one call. Fields the matvec workload cannot
 * vary are carried and recorded, never used to fit (see spec). */
typedef struct {
    uint32_t op;                /* semantic operation class */
    uint32_t M, N;              /* input shape; branch count is 0 for matvec */
    uint64_t state_bytes;       /* working set: (M*N + N + M) * 8 */
    uint64_t cache_bytes;       /* last private cache of this core class */
    uint32_t core;              /* hardware state: core class */
    uint32_t pressure;          /* resource pressure bucket, 0 = none */
    uint32_t thermal_c;         /* thermal state, degrees C (recorded) */
    uint32_t current_arm;       /* realization in use now */
    uint32_t eligible;          /* bit per arm: verified and allowed here */
    uint32_t need_evidence;     /* 1: only arms with a passing verdict (always 1 here) */
    uint32_t cognitive;         /* cognitive requirement (recorded; 0 for matvec) */
    uint64_t latency_budget_ps; /* per call; 0 = none */
    uint64_t energy_budget_pj;  /* per call; 0 = none */
} RxCmFeatures;

/* One measurement the caller made. */
typedef struct {
    uint64_t ps_per_call;       /* latency */
    int failed;                 /* wrong result, crash or verification refusal */
} RxCmObservation;

typedef struct {
    double mean_log2_ps;        /* predicted log2 picoseconds per call */
    double sd_log2;             /* predictive standard deviation, log2 units */
    double dof;                 /* degrees of freedom of the predictive t */
    uint32_t n;                 /* observations behind it */
    int known;                  /* its own cell has at least RX_CM_MIN_OBS observations */
    int borrowed;               /* unseen pressure: the quiet prediction, widened */
    double energy_pj;           /* predicted energy per call (0 if no power figure) */
    double fail_p;              /* posterior mean failure probability */
} RxCmPrediction;

enum { RX_CM_SELECT = 1, RX_CM_MEASURE = 2, RX_CM_FALLBACK = 3 };

/* Why a MEASURE or FALLBACK was returned. */
enum {
    RX_CM_WHY_CONFIDENT = 0,
    RX_CM_WHY_UNKNOWN_ARM,       /* an eligible arm has too few observations here */
    RX_CM_WHY_WORTH_MEASURING,   /* expected improvement from measuring exceeds ei_min */
    RX_CM_WHY_BUDGET_SPENT,      /* would measure, but the exploration budget is spent */
    RX_CM_WHY_NO_ELIGIBLE,       /* nothing but the fallback may run */
    RX_CM_WHY_OVER_BUDGET        /* no arm is predicted to meet the latency/energy budget */
};

typedef struct {
    int action;
    uint32_t arm;               /* SELECT / FALLBACK: run this */
    uint32_t measure[2];        /* MEASURE: run both, keep the faster */
    uint32_t why;
    double p_runner_up_better;  /* P(competitor faster than the chosen arm) */
    double expected_improvement;  /* E[max(0, log2 cost chosen - log2 cost competitor)] */
    RxCmPrediction pred[RX_CM_ARMS];
} RxCmDecision;

typedef struct {
    /* Measure only when the expected improvement of the likeliest competitor
     * over the chosen arm exceeds this, in log2 units (0.08 is about 6%).
     * Two arms predicted equal and tight are not worth a run; an arm the
     * model knows little about, or two arms it cannot separate, are. */
    double ei_min;
    double max_fail_p;          /* an arm that has failed and is above this is not selected */
    uint32_t fallback_arm;      /* known-good: always verified, never removed */
    /* Exploration budget: extra time spent measuring may not exceed
     * explore_frac of the time spent on chosen work plus explore_allow_ps. */
    double explore_frac;
    uint64_t explore_allow_ps;
} RxCmPolicy;

typedef struct {
    double tri[RX_CM_TRI];      /* X'X + lambda I, upper triangle, row-major */
    double xty[RX_CM_D];
    double yty;
    uint64_t n;
    uint64_t trials, failures;  /* failure probability: Beta(1 + f, 1 + t - f) */
} RxCmArmStats;

typedef struct {
    uint32_t n_arms;
    double lambda;              /* ridge prior precision */
    RxCmArmStats cell[RX_CM_CELLS][RX_CM_ARMS];
    double power_mw[RX_CM_CORES][RX_CM_ARMS];   /* measured power while running, 0 = unknown */
    uint64_t verify_ns[RX_CM_ARMS];             /* verification cost */
    uint64_t synth_ns[RX_CM_ARMS];              /* synthesis cost; recompute = synth + verify */
    uint64_t code_bytes[RX_CM_ARMS];            /* memory held by the realization */
    /* Calibration: predictive spread is multiplied by this. Set from residuals
     * on validation work measured apart from training (rx_cm_calibrate). */
    double sd_scale;
    /* Runtime only; not serialized. */
    uint64_t explore_ps, work_ps;
} RxCostModel;

void rx_cm_default_policy(RxCmPolicy *p);
void rx_cm_init(RxCostModel *m, uint32_t n_arms);

/* Fold one measurement in. 0 on success; RX_CM_SKIPPED for a pressure
 * measurement when there is no quiet model yet to correct. */
int rx_cm_observe(RxCostModel *m, const RxCmFeatures *f, uint32_t arm, const RxCmObservation *o);

/* Predict one arm. Needs RX_CM_MIN_OBS quiet observations for the core class.
 * Under pressure, adds the learned correction once it has RX_CM_RES_MIN_OBS
 * observations; before that, widens by RX_CM_UNSEEN_SHIFT and marks it
 * borrowed. */
void rx_cm_predict(const RxCostModel *m, const RxCmFeatures *f, uint32_t arm, RxCmPrediction *out);

/* Decide. `allow_measure` 0 freezes the model's behaviour: it never asks to
 * measure and picks the best prediction. The fallback arm is chosen when no
 * other eligible arm has any prediction at all. */
void rx_cm_decide(const RxCostModel *m, const RxCmPolicy *p, const RxCmFeatures *f,
                  int allow_measure, RxCmDecision *out);

/* Split calibration. For n measurements taken apart from the training data,
 * set sd_scale so that the central q interval covers a fraction q of them
 * (never below 1). Returns the scale, or 0 if fewer than 10 were usable. */
double rx_cm_calibrate(RxCostModel *m, const RxCmFeatures *f, const uint32_t *arm,
                       const uint64_t *ps, uint32_t n, double q);

/* Account for time spent: chosen work, and extra time spent measuring. */
void rx_cm_charge(RxCostModel *m, uint64_t work_ps, uint64_t explore_ps);

/* Canonical bytes: header, the fixed-size body, SHA-256 of both. The size is
 * rx_cm_blob_size(). Runtime counters are not included. */
size_t rx_cm_blob_size(void);
int rx_cm_serialize(const RxCostModel *m, uint8_t *out, size_t cap, size_t *out_len);
int rx_cm_deserialize(RxCostModel *m, const uint8_t *in, size_t len);
void rx_cm_digest(const RxCostModel *m, uint8_t out[32]);

/* Basis for these features, exposed so tests can see exactly what is fit. */
void rx_cm_basis(const RxCmFeatures *f, double phi[RX_CM_D]);
uint32_t rx_cm_cell(const RxCmFeatures *f);

/* Two-sided central interval half-width multiplier for coverage q with
 * `dof` degrees of freedom (Student t, Cornish-Fisher approximation). */
double rx_cm_t_quantile(double q, double dof);

#endif /* RX_COSTMODEL_H */
