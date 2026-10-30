/*
 * dagsim.h -- offline resource-DAG cost evaluator (omega-mv#583).
 *
 * Reads an OEV1 file (spec/offline-evidence-v1.md) and composes the per-op
 * costs it carries, or that rx_costmodel predicts for matvec ops, into a
 * whole-DAG schedule under finite resources: compute slots, shared
 * bandwidth (processor sharing), serialized launches, one capacity ledger,
 * dependencies and barriers. It never executes anything. It refuses rather
 * than guess: unknown provenance, stale calibration, capacity overflow,
 * unsupported constructs and cycles are all refusals with a reason.
 *
 * Host only. No GPU, no authority, no generation store.
 */
#ifndef DAGSIM_H
#define DAGSIM_H

#include <stddef.h>
#include <stdint.h>

#define DS_MAX_OPS   256
#define DS_MAX_RES   32
#define DS_MAX_BUF   256
#define DS_MAX_DEPS  1024
#define DS_MAX_REMOVED 256
#define DS_ID        40
#define DS_SRC       128

enum ds_prov { DS_PROV_UNKNOWN = 0, DS_PROV_SYNTHETIC, DS_PROV_FITTED, DS_PROV_MEASURED, DS_PROV_SPECIFIED };
enum ds_res_kind { DS_RES_COMPUTE = 1, DS_RES_BANDWIDTH, DS_RES_CAPACITY, DS_RES_LAUNCH };
enum ds_op_kind { DS_OP_COMPUTE = 1, DS_OP_COPY, DS_OP_MATVEC, DS_OP_BARRIER, DS_OP_STORAGE, DS_OP_PERSIST };
enum ds_removed_reason { DS_RM_ELIMINATED = 1, DS_RM_REUSED, DS_RM_UNCHANGED_DEP, DS_RM_DEFERRED, DS_RM_MOVED, DS_RM_SHARED, DS_RM_LOOKUP };

enum ds_status {
    DS_OK = 0,
    DS_REFUSED_UNKNOWN,          /* an op or resource with prov=unknown feeds the result */
    DS_REFUSED_STALE,            /* asof is past calibration taken + valid_days */
    DS_REFUSED_NO_CALIBRATION,   /* measured/fitted values but no calibration record */
    DS_REFUSED_CAPACITY,         /* peak live bytes exceed a capacity resource */
    DS_REFUSED_CYCLE,
    DS_REFUSED_UNSUPPORTED,      /* construct the evaluator does not model */
    DS_REFUSED_REMOVED_REQUIRED, /* a remaining op depends on removed work without a valid derivation */
    DS_PARSE_ERROR
};

typedef struct {
    char id[DS_ID];
    int kind;
    int slots;                   /* compute */
    double cap, lo, hi;          /* bandwidth (B/s) or capacity (B) */
    int prov;
    char src[DS_SRC];
} ds_resource;

typedef struct {
    char id[DS_ID];
    int kind;
    int engine;                  /* resource index or -1 */
    double t_ns, lo, hi;         /* own busy time */
    int t_prov;
    int has_t;
    double bytes;
    int bw;                      /* bandwidth resource index or -1 */
    double launch_ns;
    int launcher;                /* launch resource index or -1 */
    uint32_t M, N, core, arm;    /* matvec */
    char src[DS_SRC];
} ds_op;

typedef struct { char id[DS_ID]; double bytes; int cap; int first, last; } ds_buffer;
typedef struct { int from, to; } ds_dep;
typedef struct { char op[DS_ID]; int reason; char ref[DS_ID]; char proof[DS_SRC]; } ds_removed;

typedef struct {
    int version;
    char machine_id[DS_ID], soc[DS_ID];
    int cores;
    double mem_bytes;
    char build_sha[48], physics[48], repo[DS_ID];
    int has_calibration;
    char cal_id[DS_ID];
    int cal_taken_day;           /* days since 1970-01-01 */
    int cal_valid_days;
    double cal_load1;
    char cal_src[DS_SRC], cal_model[DS_SRC];
    int n_res, n_ops, n_buf, n_deps, n_removed;
    ds_resource res[DS_MAX_RES];
    ds_op ops[DS_MAX_OPS];
    ds_buffer buf[DS_MAX_BUF];
    ds_dep deps[DS_MAX_DEPS];
    ds_removed removed[DS_MAX_REMOVED];
    /* energy */
    char instrument[DS_ID], boundary[DS_ID];
    int energy_prov;
    double joules, power_w;
    int has_energy;
} ds_model;

typedef struct {
    int asof_day;                /* days since epoch; 0 = today */
    int mode;                    /* 0 nominal, -1 pessimistic (lo bandwidth, hi time), +1 optimistic */
    const void *costmodel;       /* const RxCostModel* or NULL */
    double interval_q;           /* for rx_costmodel intervals, e.g. 0.9 */
    const char *changed;         /* diff only: comma list of baseline ops whose inputs changed */
} ds_opts;

enum { DS_ATTR_COMPUTE = 0, DS_ATTR_BANDWIDTH, DS_ATTR_CONTENTION, DS_ATTR_LAUNCH, DS_ATTR_SERIALIZATION, DS_ATTR_SYNC, DS_ATTR_RESIDUAL, DS_ATTR_N };

typedef struct {
    int status;
    char reason[256];
    double makespan_ns;
    double start_ns[DS_MAX_OPS], end_ns[DS_MAX_OPS], ready_ns[DS_MAX_OPS];
    double op_contention_ns[DS_MAX_OPS];   /* extra traffic time due to sharing */
    double op_floor_ns[DS_MAX_OPS];        /* launch + max(t, bytes/cap) */
    double bound_critical_path_ns;
    double bound_resource_ns[DS_MAX_RES];  /* per resource work bound */
    double bound_ns;                       /* max of all bounds */
    double peak_live_bytes[DS_MAX_RES];    /* per capacity resource */
    double peak_live_at_ns[DS_MAX_RES];
    double attr_ns[DS_ATTR_N];
    int critical_path[DS_MAX_OPS], n_critical;
    uint32_t prov_mask;                    /* bit per ds_prov that fed the result */
    int synthetic;
    /* conservation: what went in is what was scheduled */
    int n_ops;
    double total_bytes, total_busy_ns, total_launch_ns;
    /* energy */
    int energy_measured;                   /* 1: joules from a named instrument */
    int energy_synthetic;                  /* 1: E = P*T from a synthetic power figure */
    double energy_j;
} ds_result;

/* Parse. Returns DS_OK or DS_PARSE_ERROR with the line and reason in `err`. */
int ds_parse_file(const char *path, ds_model *m, char *err, size_t errcap);
int ds_parse_text(const char *text, ds_model *m, char *err, size_t errcap);

/* Evaluate deterministically. Fills every field of `r` or sets a refusal. */
int ds_eval(const ds_model *m, const ds_opts *o, ds_result *r);

/* Three-point interval: nominal, lo (optimistic) and hi (pessimistic) makespan.
 * Returns the nominal status; lo/hi are 0 when any point refused. */
int ds_eval_interval(const ds_model *m, const ds_opts *o, ds_result *nominal, double *lo_ns, double *hi_ns);

/* Counterfactual sensitivity: makespan with one resource doubled (bandwidth cap x2,
 * compute slots x2, launch cost 0). Returns the new makespan or a negative status. */
double ds_counterfactual(const ds_model *m, const ds_opts *o, int res_index);

/* Minimum-work accounting: compare a baseline and a transformed model by stable op
 * ids. Sums removed work by reason and detects double-counted shared work and removal
 * of required work. */
typedef struct {
    double removed_ns[8];        /* by ds_removed_reason */
    int n_removed, n_added, n_kept;
    int double_counted;          /* an op removed with reason=shared appears twice or refs a missing op */
    int required_removed;        /* a removed op without a valid derivation still has consumers */
    char reason[256];
} ds_diff;
int ds_diff_models(const ds_model *baseline, const ds_model *transformed, const ds_opts *o, ds_diff *d);

const char *ds_status_name(int s);
const char *ds_prov_name(int p);
int ds_day_from_date(const char *yyyy_mm_dd);
int ds_find_op(const ds_model *m, const char *id);
int ds_find_res(const ds_model *m, const char *id);

#endif /* DAGSIM_H */
