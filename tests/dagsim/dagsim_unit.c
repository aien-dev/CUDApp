/*
 * dagsim_unit -- analytical tests for the offline resource-DAG evaluator
 * (omega-mv#583 acceptance, first box). Any host, no GPU.
 *
 * Every case is a DAG whose answer is known by hand. The evaluator must
 * produce exactly that number (to 1 ns), the same number twice, and never a
 * makespan below its own bounds.
 */
#include "dagsim.h"
#include "runtime/rx_costmodel.h"

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
#define NEAR(a, b, tol) (fabs((a) - (b)) <= (tol))

static const char *HDR =
    "oev version=1\n"
    "machine id=t soc=t cores=4 mem_bytes=1000 prov=specified src=test\n"
    "build repo=omega-mv sha=0000000000000000000000000000000000000000\n"
    "calibration id=c1 taken=2026-10-19 valid_days=30 load1=1.0 src=test\n"
    "resource id=mem kind=capacity cap=1000 prov=specified src=test\n"
    "energy instrument=none\n";

static int eval_text(const char *body, ds_model *m, ds_result *r, const ds_opts *o) {
    char text[8192], err[256];
    snprintf(text, sizeof text, "%s%s", HDR, body);
    int rc = ds_parse_text(text, m, err, sizeof err);
    if (rc != DS_OK) { fprintf(stderr, "  parse: %s\n", err); r->status = rc; return rc; }
    ds_opts def; memset(&def, 0, sizeof def);
    def.asof_day = ds_day_from_date("2026-10-20");
    def.interval_q = 0.9;
    rc = ds_eval(m, o ? o : &def, r);
    /* invariants every OK result must satisfy */
    if (rc == DS_OK) {
        CHECK(r->makespan_ns + 1e-6 >= r->bound_ns, "makespan %.1f below bound %.1f", r->makespan_ns, r->bound_ns);
        CHECK(r->makespan_ns + 1e-6 >= r->bound_critical_path_ns, "below critical path bound");
        double s = 0; for (int i = 0; i < DS_ATTR_N; i++) s += r->attr_ns[i];
        CHECK(NEAR(s, r->makespan_ns, 1e-6), "attribution %.3f does not sum to makespan %.3f", s, r->makespan_ns);
        ds_result r2; ds_eval(m, o ? o : &def, &r2);
        CHECK(memcmp(&r2, r, sizeof r2) == 0, "not deterministic");
    }
    return rc;
}

static void test_serial_chain(void) {
    ds_model m; ds_result r;
    int rc = eval_text(
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=B kind=compute engine=cpu t_ns=20 prov=measured src=t\n"
        "op id=C kind=compute engine=cpu t_ns=30 prov=measured src=t\n"
        "dep A B\ndep B C\n", &m, &r, NULL);
    CHECK(rc == DS_OK, "serial chain status %s: %s", ds_status_name(r.status), r.reason);
    CHECK(NEAR(r.makespan_ns, 60, 1e-9), "serial makespan %.3f", r.makespan_ns);
    CHECK(NEAR(r.bound_critical_path_ns, 60, 1e-9), "critical path bound %.3f", r.bound_critical_path_ns);
    CHECK(r.n_critical == 3, "critical path length %d", r.n_critical);
    CHECK(NEAR(r.attr_ns[DS_ATTR_COMPUTE], 60, 1e-9), "compute attribution %.3f", r.attr_ns[DS_ATTR_COMPUTE]);
    CHECK(NEAR(r.attr_ns[DS_ATTR_RESIDUAL], 0, 1e-9), "residual %.3f", r.attr_ns[DS_ATTR_RESIDUAL]);
    CHECK(NEAR(r.start_ns[2], 30, 1e-9) && NEAR(r.end_ns[2], 60, 1e-9), "C window");
    CHECK(r.prov_mask == (1u << DS_PROV_MEASURED | 1u << DS_PROV_SPECIFIED), "prov mask %x", r.prov_mask);
    CHECK(r.n_ops == 3 && NEAR(r.total_busy_ns, 60, 1e-9), "conservation");
}

static void test_perfect_overlap(void) {
    ds_model m; ds_result r;
    int rc = eval_text(
        "resource id=cpu kind=compute slots=4 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=B kind=compute engine=cpu t_ns=20 prov=measured src=t\n"
        "op id=C kind=compute engine=cpu t_ns=30 prov=measured src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK, "overlap status %s", ds_status_name(r.status));
    CHECK(NEAR(r.makespan_ns, 30, 1e-9), "independent overlap makespan %.3f", r.makespan_ns);
    CHECK(NEAR(r.attr_ns[DS_ATTR_SERIALIZATION], 0, 1e-9), "no serialization when slots suffice");
    /* same DAG, one slot: work bound 60 binds and the wait is serialization */
    rc = eval_text(
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=B kind=compute engine=cpu t_ns=20 prov=measured src=t\n"
        "op id=C kind=compute engine=cpu t_ns=30 prov=measured src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK, "one slot status");
    CHECK(NEAR(r.makespan_ns, 60, 1e-9), "one-slot makespan %.3f", r.makespan_ns);
    CHECK(NEAR(r.bound_critical_path_ns, 30, 1e-9), "critical path stays 30");
    CHECK(NEAR(r.bound_ns, 60, 1e-9), "resource bound 60, got %.3f", r.bound_ns);
    CHECK(NEAR(r.attr_ns[DS_ATTR_SERIALIZATION], 30, 1e-9), "serialization %.3f", r.attr_ns[DS_ATTR_SERIALIZATION]);
}

static void test_shared_bandwidth(void) {
    ds_model m; ds_result r;
    /* two 1 GB copies on a 1 GB/s pipe, both slots free: each gets half, both end at 2 s */
    int rc = eval_text(
        "resource id=cpu kind=compute slots=2 prov=specified src=test\n"
        "resource id=pipe kind=bandwidth cap=1e9 prov=measured src=t\n"
        "op id=A kind=copy engine=cpu bytes=1e9 bw=pipe prov=measured src=t\n"
        "op id=B kind=copy engine=cpu bytes=1e9 bw=pipe prov=measured src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK, "bw status %s: %s", ds_status_name(r.status), r.reason);
    CHECK(NEAR(r.makespan_ns, 2e9, 1e-3), "two concurrent full-bandwidth claims must not sum: %.0f", r.makespan_ns);
    CHECK(NEAR(r.end_ns[0], 2e9, 1e-3) && NEAR(r.end_ns[1], 2e9, 1e-3), "both end at 2 s");
    CHECK(NEAR(r.op_contention_ns[0], 1e9, 1e-3), "A contention %.0f", r.op_contention_ns[0]);
    CHECK(NEAR(r.bound_critical_path_ns, 1e9, 1e-3), "cp bound 1 s");
    CHECK(NEAR(r.bound_ns, 2e9, 1e-3), "pipe work bound 2 s, got %.0f", r.bound_ns);
    CHECK(NEAR(r.attr_ns[DS_ATTR_BANDWIDTH] + r.attr_ns[DS_ATTR_CONTENTION], 2e9, 1e-3), "bw+contention = 2 s");
    CHECK(NEAR(r.attr_ns[DS_ATTR_CONTENTION], 1e9, 1e-3), "contention attributed 1 s, got %.0f", r.attr_ns[DS_ATTR_CONTENTION]);
    /* unequal: B is half the size; it finishes at 1 s, A at 1.5 s */
    rc = eval_text(
        "resource id=cpu kind=compute slots=2 prov=specified src=test\n"
        "resource id=pipe kind=bandwidth cap=1e9 prov=measured src=t\n"
        "op id=A kind=copy engine=cpu bytes=1e9 bw=pipe prov=measured src=t\n"
        "op id=B kind=copy engine=cpu bytes=5e8 bw=pipe prov=measured src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK, "bw2 status");
    CHECK(NEAR(r.end_ns[1], 1e9, 1e-3), "B ends at 1 s, got %.0f", r.end_ns[1]);
    CHECK(NEAR(r.end_ns[0], 1.5e9, 1e-3), "A ends at 1.5 s, got %.0f", r.end_ns[0]);
    /* compute and traffic inside one op overlap: floor is max(F/P, Q/B) */
    rc = eval_text(
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "resource id=pipe kind=bandwidth cap=1e9 prov=measured src=t\n"
        "op id=A kind=compute engine=cpu t_ns=5e8 bytes=1e9 bw=pipe prov=measured src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK && NEAR(r.makespan_ns, 1e9, 1e-3), "max(F/P,Q/B) floor, got %.0f", r.makespan_ns);
    CHECK(NEAR(r.op_floor_ns[0], 1e9, 1e-3), "op floor");
    /* a compute-bound op streaming alone has no contention, whatever its busy time */
    rc = eval_text(
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "resource id=pipe kind=bandwidth cap=1e9 prov=measured src=t\n"
        "op id=A kind=compute engine=cpu t_ns=4e9 bytes=1e9 bw=pipe prov=measured src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK && NEAR(r.makespan_ns, 4e9, 1e-3) && NEAR(r.op_contention_ns[0], 0, 1e-3),
          "compute-bound op alone: contention %.0f (want 0)", r.op_contention_ns[0]);
    CHECK(NEAR(r.attr_ns[DS_ATTR_COMPUTE], 4e9, 1e-3), "attributed as compute");
    /* serialized sharing: one slot, so no contention and makespan = sum */
    rc = eval_text(
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "resource id=pipe kind=bandwidth cap=1e9 prov=measured src=t\n"
        "op id=A kind=copy engine=cpu bytes=1e9 bw=pipe prov=measured src=t\n"
        "op id=B kind=copy engine=cpu bytes=1e9 bw=pipe prov=measured src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK && NEAR(r.makespan_ns, 2e9, 1e-3) && NEAR(r.attr_ns[DS_ATTR_CONTENTION], 0, 1e-3),
          "serialized copies: no contention, makespan %.0f contention %.0f", r.makespan_ns, r.attr_ns[DS_ATTR_CONTENTION]);
}

static void test_capacity_refusal(void) {
    ds_model m; ds_result r;
    int rc = eval_text(
        "resource id=cpu kind=compute slots=2 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "buffer id=x bytes=600 cap=mem first=A last=A\n"
        "buffer id=y bytes=600 cap=mem first=A last=A\n", &m, &r, NULL);
    CHECK(rc == DS_REFUSED_CAPACITY, "1200 live bytes in 1000 must refuse, got %s", ds_status_name(rc));
    CHECK(strstr(r.reason, "x") && strstr(r.reason, "y"), "reason names the buffers: %s", r.reason);
    /* same buffers, disjoint lifetimes: fits */
    rc = eval_text(
        "resource id=cpu kind=compute slots=2 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=B kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "dep A B\n"
        "buffer id=x bytes=600 cap=mem first=A last=A\n"
        "buffer id=y bytes=600 cap=mem first=B last=B\n", &m, &r, NULL);
    CHECK(rc == DS_OK, "disjoint lifetimes fit, got %s: %s", ds_status_name(rc), r.reason);
    CHECK(NEAR(r.peak_live_bytes[0], 600, 1e-9), "peak 600, got %.0f", r.peak_live_bytes[0]);
    /* a single buffer larger than the ledger refuses before any scheduling */
    rc = eval_text(
        "resource id=cpu kind=compute slots=2 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "buffer id=big bytes=2000 cap=mem\n", &m, &r, NULL);
    CHECK(rc == DS_REFUSED_CAPACITY, "oversized resident buffer refuses");
}

static void test_lifetimes(void) {
    ds_model m; ds_result r;
    int rc = eval_text(
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=B kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=C kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "dep A B\ndep B C\n"
        "buffer id=resident bytes=100 cap=mem\n"
        "buffer id=x bytes=300 cap=mem first=A last=B\n"
        "buffer id=y bytes=400 cap=mem first=B last=C\n", &m, &r, NULL);
    CHECK(rc == DS_OK, "lifetimes status %s: %s", ds_status_name(rc), r.reason);
    CHECK(NEAR(r.peak_live_bytes[0], 800, 1e-9), "peak = resident + x + y during B, got %.0f", r.peak_live_bytes[0]);
    CHECK(NEAR(r.peak_live_at_ns[0], 10, 1e-9), "peak reached when B starts, got %.1f", r.peak_live_at_ns[0]);
}

static void test_sync(void) {
    ds_model m; ds_result r;
    int rc = eval_text(
        "resource id=cpu kind=compute slots=2 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=B kind=compute engine=cpu t_ns=30 prov=measured src=t\n"
        "op id=C kind=compute engine=cpu t_ns=5 prov=measured src=t\n"
        "sync id=s ops=A,B\n"
        "dep s C\n", &m, &r, NULL);
    CHECK(rc == DS_OK, "sync status %s: %s", ds_status_name(rc), r.reason);
    int c = ds_find_op(&m, "C");
    CHECK(c >= 0 && NEAR(r.start_ns[c], 30, 1e-9), "C waits for the slower of A,B: start %.1f", c >= 0 ? r.start_ns[c] : -1);
    CHECK(NEAR(r.makespan_ns, 35, 1e-9), "sync makespan %.1f", r.makespan_ns);
    /* a barrier with a declared cost is attributed as sync */
    rc = eval_text(
        "resource id=cpu kind=compute slots=2 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=B kind=compute engine=cpu t_ns=30 prov=measured src=t\n"
        "op id=fence kind=barrier t_ns=7 prov=measured src=t\n"
        "op id=C kind=compute engine=cpu t_ns=5 prov=measured src=t\n"
        "dep A fence\ndep B fence\ndep fence C\n", &m, &r, NULL);
    CHECK(rc == DS_OK && NEAR(r.makespan_ns, 42, 1e-9), "costed barrier makespan %.1f", r.makespan_ns);
    CHECK(NEAR(r.attr_ns[DS_ATTR_SYNC], 7, 1e-9), "sync attribution %.1f", r.attr_ns[DS_ATTR_SYNC]);
}

static void test_launch(void) {
    ds_model m; ds_result r;
    /* two jobs, one launcher, one slot: launches serialize, 80 ms */
    int rc = eval_text(
        "resource id=gpu kind=compute slots=1 prov=specified src=test\n"
        "resource id=L kind=launch prov=measured src=t\n"
        "op id=A kind=compute engine=gpu t_ns=0 launch_ns=40e6 launcher=L prov=measured src=t\n"
        "op id=B kind=compute engine=gpu t_ns=0 launch_ns=40e6 launcher=L prov=measured src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK && NEAR(r.makespan_ns, 80e6, 1e-3), "serialized launches add: %.0f", r.makespan_ns);
    CHECK(NEAR(r.attr_ns[DS_ATTR_LAUNCH] + r.attr_ns[DS_ATTR_SERIALIZATION], 80e6, 1e-3), "launch+serialization");
    /* two engines, two launchers: launches overlap, 40 ms */
    rc = eval_text(
        "resource id=gpu kind=compute slots=1 prov=specified src=test\n"
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "resource id=L1 kind=launch prov=measured src=t\n"
        "resource id=L2 kind=launch prov=measured src=t\n"
        "op id=A kind=compute engine=gpu t_ns=0 launch_ns=40e6 launcher=L1 prov=measured src=t\n"
        "op id=B kind=compute engine=cpu t_ns=0 launch_ns=40e6 launcher=L2 prov=measured src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK && NEAR(r.makespan_ns, 40e6, 1e-3), "independent launches overlap: %.0f", r.makespan_ns);
}

static void test_calibration(void) {
    ds_model m; ds_result r; char text[4096], err[256];
    const char *body =
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n";
    /* stale: asof past taken + valid_days */
    ds_opts o; memset(&o, 0, sizeof o); o.interval_q = 0.9;
    o.asof_day = ds_day_from_date("2026-11-19");
    int rc = eval_text(body, &m, &r, &o);
    CHECK(rc == DS_REFUSED_STALE, "31 days after a 30-day calibration refuses, got %s", ds_status_name(rc));
    o.asof_day = ds_day_from_date("2026-11-18");
    rc = eval_text(body, &m, &r, &o);
    CHECK(rc == DS_OK, "last valid day is fine, got %s", ds_status_name(rc));
    /* missing: measured value without a calibration record */
    snprintf(text, sizeof text,
        "oev version=1\n"
        "machine id=t soc=t cores=4 mem_bytes=1000 prov=specified src=test\n"
        "build repo=omega-mv sha=0000000000000000000000000000000000000000\n"
        "energy instrument=none\n%s", body);
    rc = ds_parse_text(text, &m, err, sizeof err);
    CHECK(rc == DS_OK, "parse without calibration: %s", err);
    o.asof_day = ds_day_from_date("2026-10-20");
    rc = ds_eval(&m, &o, &r);
    CHECK(rc == DS_REFUSED_NO_CALIBRATION, "measured without calibration refuses, got %s", ds_status_name(rc));
    /* unknown provenance never becomes a number */
    rc = eval_text(
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=B kind=compute engine=cpu prov=unknown src=arch-244-not-run\n"
        "dep A B\n", &m, &r, NULL);
    CHECK(rc == DS_REFUSED_UNKNOWN, "unknown op refuses, got %s", ds_status_name(rc));
    CHECK(strstr(r.reason, "B") != NULL, "reason names the op: %s", r.reason);
    CHECK(r.makespan_ns == 0, "no makespan on refusal");
    /* synthetic input marks the result */
    rc = eval_text(
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=synthetic src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK && r.synthetic == 1, "synthetic flag");
}

static void test_intervals(void) {
    ds_model m; ds_result r; double lo, hi;
    char text[8192];
    snprintf(text, sizeof text, "%s%s", HDR,
        "resource id=cpu kind=compute slots=2 prov=specified src=test\n"
        "resource id=pipe kind=bandwidth cap=1e9 lo=0.5e9 hi=2e9 prov=measured src=t\n"
        "op id=A kind=compute engine=cpu t_ns=100 lo=80 hi=150 prov=measured src=t\n"
        "op id=B kind=copy engine=cpu bytes=1e9 bw=pipe prov=measured src=t\n"
        "dep A B\n");
    char err[256];
    CHECK(ds_parse_text(text, &m, err, sizeof err) == DS_OK, "parse: %s", err);
    ds_opts o; memset(&o, 0, sizeof o); o.asof_day = ds_day_from_date("2026-10-20"); o.interval_q = 0.9;
    int rc = ds_eval_interval(&m, &o, &r, &lo, &hi);
    CHECK(rc == DS_OK, "interval status");
    CHECK(NEAR(r.makespan_ns, 100 + 1e9, 1e-3), "nominal %.0f", r.makespan_ns);
    CHECK(NEAR(lo, 80 + 0.5e9, 1e-3), "optimistic %.0f", lo);
    CHECK(NEAR(hi, 150 + 2e9, 1e-3), "pessimistic %.0f", hi);
}

static void test_energy(void) {
    ds_model m; ds_result r; char text[4096], err[256];
    int rc = eval_text(
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=1e9 prov=measured src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK && !r.energy_measured && !r.energy_synthetic, "instrument=none means UNMEASURED");
    snprintf(text, sizeof text,
        "oev version=1\n"
        "machine id=t soc=t cores=4 mem_bytes=1000 prov=specified src=test\n"
        "build repo=omega-mv sha=0000000000000000000000000000000000000000\n"
        "calibration id=c1 taken=2026-10-19 valid_days=30 load1=1.0 src=test\n"
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=1e9 prov=measured src=t\n"
        "energy instrument=tdp-guess boundary=soc power_w=100 prov=synthetic\n");
    CHECK(ds_parse_text(text, &m, err, sizeof err) == DS_OK, "parse: %s", err);
    ds_opts o; memset(&o, 0, sizeof o); o.asof_day = ds_day_from_date("2026-10-20"); o.interval_q = 0.9;
    rc = ds_eval(&m, &o, &r);
    CHECK(rc == DS_OK && r.energy_synthetic == 1 && !r.energy_measured && NEAR(r.energy_j, 100.0, 1e-9) && r.synthetic,
          "synthetic power gives a labelled E=P*T (100 J), got %.3f syn=%d", r.energy_j, r.energy_synthetic);
    snprintf(text, sizeof text,
        "oev version=1\n"
        "machine id=t soc=t cores=4 mem_bytes=1000 prov=specified src=test\n"
        "build repo=omega-mv sha=0000000000000000000000000000000000000000\n"
        "calibration id=c1 taken=2026-10-19 valid_days=30 load1=1.0 src=test\n"
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=1e9 prov=measured src=t\n"
        "energy instrument=cyberpower-ups-nut boundary=wall joules=42.5 prov=measured src=receipt.json\n");
    CHECK(ds_parse_text(text, &m, err, sizeof err) == DS_OK, "parse: %s", err);
    rc = ds_eval(&m, &o, &r);
    CHECK(rc == DS_OK && r.energy_measured == 1 && NEAR(r.energy_j, 42.5, 1e-9) && !r.synthetic, "measured joules pass through");
    /* a measured joules figure with no instrument name is a parse error */
    snprintf(text, sizeof text,
        "oev version=1\n"
        "machine id=t soc=t cores=4 mem_bytes=1000 prov=specified src=test\n"
        "build repo=omega-mv sha=0000000000000000000000000000000000000000\n"
        "energy instrument=none joules=42.5 prov=measured src=x\n");
    CHECK(ds_parse_text(text, &m, err, sizeof err) == DS_PARSE_ERROR, "joules without an instrument is refused at parse");
}

static void test_parse_and_cycles(void) {
    ds_model m; ds_result r; char text[4096], err[256];
    snprintf(text, sizeof text, "%s%s", HDR, "resource id=cpu kind=compute slots=1 prov=specified src=test\nbogus x=1\n");
    int rc = ds_parse_text(text, &m, err, sizeof err);
    CHECK(rc == DS_PARSE_ERROR && strstr(err, "line 8"), "unknown keyword is an error with its line: %s", err);
    snprintf(text, sizeof text, "%s%s", HDR, "resource id=cpu kind=compute slots=1 prov=specified src=test\nop id=A kind=compute engine=cpu t_ns=1 prov=measured src=t typo=3\n");
    rc = ds_parse_text(text, &m, err, sizeof err);
    CHECK(rc == DS_PARSE_ERROR, "unknown field is an error: %s", err);
    snprintf(text, sizeof text, "%s%s", HDR, "resource id=cpu kind=compute slots=1 prov=specified src=test\nop id=A kind=compute engine=cpu t_ns=1 src=t\n");
    rc = ds_parse_text(text, &m, err, sizeof err);
    CHECK(rc == DS_PARSE_ERROR, "a number without prov is an error: %s", err);
    rc = eval_text(
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=1 prov=measured src=t\n"
        "op id=B kind=compute engine=cpu t_ns=1 prov=measured src=t\n"
        "dep A B\ndep B A\n", &m, &r, NULL);
    CHECK(rc == DS_REFUSED_CYCLE, "cycle refuses, got %s", ds_status_name(rc));
    /* the pinned fixture parses and evaluates */
    rc = ds_parse_file("tests/dagsim/fixtures/m1_spark_subset.oev", &m, err, sizeof err);
    CHECK(rc == DS_OK, "fixture parses: %s", err);
    if (rc == DS_OK) {
        ds_opts o; memset(&o, 0, sizeof o); o.asof_day = ds_day_from_date("2026-10-20"); o.interval_q = 0.9;
        rc = ds_eval(&m, &o, &r);
        CHECK(rc == DS_OK, "fixture evaluates: %s", r.reason);
        CHECK(NEAR(r.makespan_ns, 20000 + 40736700 + 111000000, 1e-3), "fixture serial makespan %.0f", r.makespan_ns);
        CHECK(!r.energy_measured && !r.energy_synthetic, "fixture energy UNMEASURED");
    }
}

/* rx_costmodel integration: a matvec op with a fitted blob gets its time from
 * rx_cm_predict; an unknown cell refuses. Synthetic observations, so the blob is
 * labelled fitted on synthetic data and the result is marked synthetic. */
static void test_costmodel_reuse(void) {
    RxCostModel cm; rx_cm_init(&cm, 2);
    uint64_t rng = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < 200; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        uint32_t M = 8 + (uint32_t)(rng % 120), N = 4 + (uint32_t)((rng >> 20) % 60);
        RxCmFeatures f; memset(&f, 0, sizeof f);
        f.op = RX_CM_OP_MATVEC; f.M = M; f.N = N; f.state_bytes = ((uint64_t)M * N + M + N) * 8u;
        f.cache_bytes = 1u << 20; f.core = RX_CM_CORE_X925; f.eligible = 3;
        double l2 = 10.0 + 0.95 * (log2((double)M) + log2((double)N));
        RxCmObservation o; o.failed = 0; o.ps_per_call = (uint64_t)pow(2.0, l2);
        rx_cm_observe(&cm, &f, 0, &o);
    }
    ds_model m; ds_result r; char text[4096], err[256];
    snprintf(text, sizeof text, "%s%s", HDR,
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=mv kind=matvec engine=cpu M=64 N=16 core=0 arm=0 prov=fitted src=rx_costmodel\n");
    CHECK(ds_parse_text(text, &m, err, sizeof err) == DS_OK, "parse: %s", err);
    ds_opts o; memset(&o, 0, sizeof o); o.asof_day = ds_day_from_date("2026-10-20"); o.interval_q = 0.9;
    int rc = ds_eval(&m, &o, &r);
    CHECK(rc == DS_REFUSED_UNKNOWN, "matvec with no model refuses, got %s", ds_status_name(rc));
    o.costmodel = &cm;
    rc = ds_eval(&m, &o, &r);
    CHECK(rc == DS_OK, "matvec with model: %s %s", ds_status_name(rc), r.reason);
    RxCmFeatures f; memset(&f, 0, sizeof f);
    f.op = RX_CM_OP_MATVEC; f.M = 64; f.N = 16; f.state_bytes = (64u * 16u + 64u + 16u) * 8u; f.cache_bytes = 1u << 20; f.core = 0;
    RxCmPrediction p; rx_cm_predict(&cm, &f, 0, &p);
    CHECK(p.known, "cell known");
    CHECK(NEAR(r.makespan_ns, pow(2.0, p.mean_log2_ps) / 1000.0, 1e-6), "time comes from rx_cm_predict: %.3f vs %.3f", r.makespan_ns, pow(2.0, p.mean_log2_ps) / 1000.0);
    CHECK(r.prov_mask & (1u << DS_PROV_FITTED), "fitted provenance recorded");
    /* arm 1 was never observed: unknown, refuse */
    snprintf(text, sizeof text, "%s%s", HDR,
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=mv kind=matvec engine=cpu M=64 N=16 core=0 arm=1 prov=fitted src=rx_costmodel\n");
    CHECK(ds_parse_text(text, &m, err, sizeof err) == DS_OK, "parse: %s", err);
    rc = ds_eval(&m, &o, &r);
    CHECK(rc == DS_REFUSED_UNKNOWN, "unobserved arm refuses, got %s", ds_status_name(rc));
}

static void test_counterfactual(void) {
    ds_model m; ds_result r;
    int rc = eval_text(
        "resource id=cpu kind=compute slots=2 prov=specified src=test\n"
        "resource id=pipe kind=bandwidth cap=1e9 prov=measured src=t\n"
        "op id=A kind=copy engine=cpu bytes=1e9 bw=pipe prov=measured src=t\n"
        "op id=B kind=copy engine=cpu bytes=1e9 bw=pipe prov=measured src=t\n", &m, &r, NULL);
    CHECK(rc == DS_OK, "cf status");
    ds_opts o; memset(&o, 0, sizeof o); o.asof_day = ds_day_from_date("2026-10-20"); o.interval_q = 0.9;
    double d = ds_counterfactual(&m, &o, ds_find_res(&m, "pipe"));
    CHECK(NEAR(d, 1e9, 1e-3), "doubling the pipe halves the bandwidth-bound DAG: %.0f", d);
    d = ds_counterfactual(&m, &o, ds_find_res(&m, "cpu"));
    CHECK(NEAR(d, 2e9, 1e-3), "doubling slots changes nothing when the pipe binds: %.0f", d);
}

static void test_minimum_work_diff(void) {
    ds_model base, tr; ds_diff d; char text[8192], err[256];
    const char *base_body =
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=B kind=compute engine=cpu t_ns=20 prov=measured src=t\n"
        "op id=C kind=compute engine=cpu t_ns=30 prov=measured src=t\n"
        "dep A B\ndep B C\n";
    snprintf(text, sizeof text, "%s%s", HDR, base_body);
    CHECK(ds_parse_text(text, &base, err, sizeof err) == DS_OK, "base: %s", err);
    ds_opts o; memset(&o, 0, sizeof o); o.asof_day = ds_day_from_date("2026-10-20"); o.interval_q = 0.9;
    /* no change: identical transformed DAG adds no nodes and removes none */
    CHECK(ds_parse_text(text, &tr, err, sizeof err) == DS_OK, "tr: %s", err);
    CHECK(ds_diff_models(&base, &tr, &o, &d) == DS_OK && d.n_removed == 0 && d.n_added == 0 && d.n_kept == 3, "no-change fixture: removed %d added %d kept %d", d.n_removed, d.n_added, d.n_kept);
    /* B reused from a lookup: valid, 20 ns of work moves to the reused ledger */
    snprintf(text, sizeof text, "%s%s", HDR,
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=B_lookup kind=compute engine=cpu t_ns=2 prov=measured src=t\n"
        "op id=C kind=compute engine=cpu t_ns=30 prov=measured src=t\n"
        "dep A B_lookup\ndep B_lookup C\n"
        "removed op=B reason=reused ref=B_lookup proof=rx_plan:digest\n");
    CHECK(ds_parse_text(text, &tr, err, sizeof err) == DS_OK, "tr2: %s", err);
    int rc = ds_diff_models(&base, &tr, &o, &d);
    CHECK(rc == DS_OK && NEAR(d.removed_ns[DS_RM_REUSED], 20, 1e-9) && d.n_added == 1, "reuse accounted: %s rc=%s", d.reason, ds_status_name(rc));
    /* eliminated with no proof while C still consumed it: required work cannot vanish by assertion */
    snprintf(text, sizeof text, "%s%s", HDR,
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=C kind=compute engine=cpu t_ns=30 prov=measured src=t\n"
        "dep A C\n"
        "removed op=B reason=eliminated\n");
    CHECK(ds_parse_text(text, &tr, err, sizeof err) == DS_OK, "tr3: %s", err);
    rc = ds_diff_models(&base, &tr, &o, &d);
    CHECK(rc == DS_REFUSED_REMOVED_REQUIRED && d.required_removed, "unproven elimination of consumed work refuses, got %s", ds_status_name(rc));
    /* unchanged_dep on A, but A changed: the consumer is invalidated */
    snprintf(text, sizeof text, "%s%s", HDR,
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=C kind=compute engine=cpu t_ns=30 prov=measured src=t\n"
        "dep A C\n"
        "removed op=B reason=unchanged_dep ref=A\n");
    CHECK(ds_parse_text(text, &tr, err, sizeof err) == DS_OK, "tr4: %s", err);
    rc = ds_diff_models(&base, &tr, &o, &d);
    CHECK(rc == DS_OK && NEAR(d.removed_ns[DS_RM_UNCHANGED_DEP], 20, 1e-9), "unchanged dependency allows skipping B");
    o.changed = "A";
    rc = ds_diff_models(&base, &tr, &o, &d);
    CHECK(rc == DS_REFUSED_REMOVED_REQUIRED, "changed A invalidates the skip of B, got %s", ds_status_name(rc));
    o.changed = NULL;
    /* shared work counted twice is detected */
    snprintf(text, sizeof text, "%s%s", HDR,
        "resource id=cpu kind=compute slots=1 prov=specified src=test\n"
        "op id=A kind=compute engine=cpu t_ns=10 prov=measured src=t\n"
        "op id=C kind=compute engine=cpu t_ns=30 prov=measured src=t\n"
        "dep A C\n"
        "removed op=B reason=shared ref=A\n"
        "removed op=B reason=shared ref=C\n");
    CHECK(ds_parse_text(text, &tr, err, sizeof err) == DS_OK, "tr5: %s", err);
    rc = ds_diff_models(&base, &tr, &o, &d);
    CHECK(rc != DS_OK && d.double_counted, "double-counted shared work detected");
}

int main(void) {
    test_serial_chain();
    test_perfect_overlap();
    test_shared_bandwidth();
    test_capacity_refusal();
    test_lifetimes();
    test_sync();
    test_launch();
    test_calibration();
    test_intervals();
    test_energy();
    test_parse_and_cycles();
    test_costmodel_reuse();
    test_counterfactual();
    test_minimum_work_diff();
    printf("dagsim_unit: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
