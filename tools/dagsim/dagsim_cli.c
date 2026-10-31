/*
 * dagsim_cli -- reproducible command line for the offline resource-DAG
 * evaluator (omega-mv#583). No GPU, no execution: it reads OEV1 files and
 * prints JSON. Every number it prints names the provenance it came from.
 *
 *   dagsim eval FILE [--asof YYYY-MM-DD] [--model BLOB]
 *   dagsim pareto FILE... [--asof YYYY-MM-DD] [--model BLOB]
 *   dagsim diff BASELINE TRANSFORMED [--changed a,b,...]
 */
#include "dagsim.h"
#include "runtime/rx_costmodel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void jstr(const char *s) {
    putchar('"');
    for (; *s; s++) {
        if (*s == '"' || *s == '\\') { putchar('\\'); putchar(*s); }
        else if ((unsigned char)*s < 0x20) printf("\\u%04x", *s);
        else putchar(*s);
    }
    putchar('"');
}
static void jnum(double v) { if (v == floor(v) && fabs(v) < 1e15) printf("%.0f", v); else printf("%.6g", v); }

static int load_model(const char *path, RxCostModel *cm) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    size_t cap = rx_cm_blob_size();
    uint8_t *buf = malloc(cap);
    if (!buf) { fclose(fp); return 0; }
    size_t got = fread(buf, 1, cap, fp);
    fclose(fp);
    int ok = rx_cm_deserialize(cm, buf, got) == RX_CM_OK;
    free(buf);
    return ok;
}

static const char *attr_name[DS_ATTR_N] = { "compute", "bandwidth", "contention", "launch", "serialization", "sync", "residual" };
static const char *res_kind_name(int k) {
    switch (k) { case DS_RES_COMPUTE: return "compute"; case DS_RES_BANDWIDTH: return "bandwidth"; case DS_RES_CAPACITY: return "capacity"; case DS_RES_LAUNCH: return "launch"; default: return "?"; }
}

static void emit_result_json(const char *path, const ds_model *m, const ds_opts *o, const ds_result *r, double lo, double hi) {
    printf("{\n  \"schema\": \"oev1\", \"tool\": \"dagsim\", \"file\": "); jstr(path);
    printf(",\n  \"machine\": "); jstr(m->machine_id);
    printf(", \"build_sha\": "); jstr(m->build_sha);
    printf(",\n  \"calibration\": {\"id\": "); jstr(m->has_calibration ? m->cal_id : "");
    printf(", \"taken_day\": %d, \"valid_days\": %d, \"load1\": ", m->cal_taken_day, m->cal_valid_days); jnum(m->cal_load1);
    printf(", \"src\": "); jstr(m->cal_src); printf("},\n  \"asof_day\": %d,\n", o->asof_day);
    printf("  \"status\": "); jstr(ds_status_name(r->status)); printf(", \"reason\": "); jstr(r->reason);
    printf(",\n  \"provenance\": ["); int first = 1;
    for (int p = 0; p <= DS_PROV_SPECIFIED; p++) if (r->prov_mask & (1u << p)) { if (!first) printf(", "); jstr(ds_prov_name(p)); first = 0; }
    printf("], \"synthetic\": %d, \"offline\": true, \"gpu_executed\": false", r->synthetic);
    if (r->status != DS_OK) { printf("\n}\n"); return; }
    printf(",\n  \"makespan_ns\": {\"nominal\": "); jnum(r->makespan_ns); printf(", \"lo\": "); jnum(lo); printf(", \"hi\": "); jnum(hi);
    printf(", \"interval_known\": %s}", (lo > 0 && hi > 0) ? "true" : "false");
    printf(",\n  \"bounds_ns\": {\"critical_path\": "); jnum(r->bound_critical_path_ns); printf(", \"max\": "); jnum(r->bound_ns); printf(", \"resources\": {");
    for (int i = 0; i < m->n_res; i++) { if (i) printf(", "); jstr(m->res[i].id); printf(": "); jnum(r->bound_resource_ns[i]); }
    printf("}, \"makespan_ge_bounds\": %s}", r->makespan_ns + 1e-6 >= r->bound_ns ? "true" : "false");
    printf(",\n  \"ops\": [");
    for (int i = 0; i < m->n_ops; i++) {
        printf("%s\n    {\"id\": ", i ? "," : ""); jstr(m->ops[i].id);
        printf(", \"start_ns\": "); jnum(r->start_ns[i]); printf(", \"end_ns\": "); jnum(r->end_ns[i]);
        printf(", \"floor_ns\": "); jnum(r->op_floor_ns[i]); printf(", \"contention_ns\": "); jnum(r->op_contention_ns[i]);
        printf(", \"prov\": "); jstr(ds_prov_name(m->ops[i].kind == DS_OP_MATVEC && !m->ops[i].has_t ? DS_PROV_FITTED : m->ops[i].t_prov));
        printf(", \"src\": "); jstr(m->ops[i].src); printf("}");
    }
    printf("\n  ],\n  \"critical_path\": [");
    for (int i = 0; i < r->n_critical; i++) { if (i) printf(", "); jstr(m->ops[r->critical_path[i]].id); }
    printf("],\n  \"peak_live_bytes\": {");
    first = 1;
    for (int i = 0; i < m->n_res; i++) if (m->res[i].kind == DS_RES_CAPACITY) {
        if (!first) printf(", ");
        first = 0;
        jstr(m->res[i].id); printf(": {\"bytes\": "); jnum(r->peak_live_bytes[i]); printf(", \"at_ns\": "); jnum(r->peak_live_at_ns[i]); printf(", \"cap\": "); jnum(m->res[i].cap); printf("}");
    }
    printf("},\n  \"attribution_ns\": {");
    for (int i = 0; i < DS_ATTR_N; i++) { if (i) printf(", "); jstr(attr_name[i]); printf(": "); jnum(r->attr_ns[i]); }
    printf("},\n  \"energy\": {\"status\": ");
    jstr(r->energy_measured ? "MEASURED" : r->energy_synthetic ? "SYNTHETIC_SENSITIVITY_ONLY" : "UNMEASURED");
    printf(", \"joules\": "); if (r->energy_measured || r->energy_synthetic) jnum(r->energy_j); else printf("null");
    printf(", \"instrument\": "); jstr(m->instrument); printf(", \"boundary\": "); jstr(m->boundary);
    printf(", \"ranking_valid\": %s}", r->energy_measured ? "true" : "false");
    printf(",\n  \"conservation\": {\"n_ops\": %d, \"total_bytes\": ", r->n_ops); jnum(r->total_bytes);
    printf(", \"total_busy_ns\": "); jnum(r->total_busy_ns); printf(", \"total_launch_ns\": "); jnum(r->total_launch_ns);
    printf(", \"instrument_overhead_ns\": null}");
    /* counterfactual sensitivities, ranked by how much they could move the makespan,
     * weighted by how uncertain the resource's own figure is */
    printf(",\n  \"sensitivities\": [");
    double score[DS_MAX_RES]; int idx[DS_MAX_RES]; double cf[DS_MAX_RES]; double width[DS_MAX_RES];
    for (int i = 0; i < m->n_res; i++) {
        cf[i] = ds_counterfactual(m, o, i);
        width[i] = m->res[i].kind == DS_RES_BANDWIDTH && m->res[i].cap > 0 ? (m->res[i].hi - m->res[i].lo) / m->res[i].cap : 0;
        if (m->res[i].prov == DS_PROV_SPECIFIED) width[i] = 1.0;    /* a datasheet number is not a measurement */
        score[i] = cf[i] >= 0 ? fabs(r->makespan_ns - cf[i]) * (width[i] > 0 ? width[i] : 0.05) : 0;
        idx[i] = i;
    }
    for (int a = 1; a < m->n_res; a++) {
        int k = idx[a]; int b = a - 1;
        while (b >= 0 && score[idx[b]] < score[k]) { idx[b + 1] = idx[b]; b--; }
        idx[b + 1] = k;
    }
    for (int a = 0; a < m->n_res; a++) {
        int i = idx[a];
        printf("%s\n    {\"resource\": ", a ? "," : ""); jstr(m->res[i].id);
        printf(", \"kind\": "); jstr(res_kind_name(m->res[i].kind));
        printf(", \"makespan_if_doubled_ns\": "); if (cf[i] >= 0) jnum(cf[i]); else printf("null");
        printf(", \"delta_ns\": "); if (cf[i] >= 0) jnum(r->makespan_ns - cf[i]); else printf("null");
        printf(", \"interval_width_frac\": "); jnum(width[i]);
        printf(", \"decision_score\": "); jnum(score[i]);
        printf(", \"next_experiment\": ");
        if (m->res[i].kind == DS_RES_BANDWIDTH) jstr(m->res[i].prov == DS_PROV_SPECIFIED ? "measure this bandwidth on this DAG's shapes (datasheet figure, never attained)" : "repeat the bandwidth receipt on a quiet box to tighten lo/hi");
        else if (m->res[i].kind == DS_RES_LAUNCH) jstr("measure launch with the device context held open (M1 block2 follow-up)");
        else if (m->res[i].kind == DS_RES_COMPUTE) jstr("measure the critical op at the next larger shape");
        else jstr("none: capacity is a ledger, not a rate");
        printf("}");
    }
    printf("\n  ],\n  \"unresolved_residual_ns\": "); jnum(r->attr_ns[DS_ATTR_RESIDUAL]);
    printf(",\n  \"note\": \"loaded-box calibration (load1 above); bounds are floors, not predictions; no performance claim from this simulation alone\"\n}\n");
}

static int cmd_eval(const char *path, const ds_opts *o) {
    ds_model *m = malloc(sizeof *m); char err[256];
    if (!m) return 2;
    if (ds_parse_file(path, m, err, sizeof err) != DS_OK) {
        printf("{\"schema\": \"oev1\", \"tool\": \"dagsim\", \"file\": "); jstr(path); printf(", \"status\": \"PARSE_ERROR\", \"reason\": "); jstr(err); printf("}\n");
        free(m); return 1;
    }
    ds_result *r = malloc(sizeof *r); double lo, hi;
    if (!r) { free(m); return 2; }
    int rc = ds_eval_interval(m, o, r, &lo, &hi);
    emit_result_json(path, m, o, r, lo, hi);
    free(r); free(m);
    return rc == DS_OK ? 0 : 1;
}

typedef struct { const char *path; int status; double t, lo, hi, peak, e; int e_meas, e_syn, front; char reason[256]; } prow;

static int cmd_pareto(int n, const char **paths, const ds_opts *o) {
    prow rows[64]; if (n > 64) n = 64;
    int all_energy = 1;
    for (int k = 0; k < n; k++) {
        prow *p = &rows[k]; memset(p, 0, sizeof *p); p->path = paths[k];
        ds_model *m = malloc(sizeof *m); ds_result *r = malloc(sizeof *r); char err[256];
        if (!m || !r) return 2;
        if (ds_parse_file(paths[k], m, err, sizeof err) != DS_OK) { p->status = DS_PARSE_ERROR; snprintf(p->reason, sizeof p->reason, "%s", err); free(m); free(r); all_energy = 0; continue; }
        p->status = ds_eval_interval(m, o, r, &p->lo, &p->hi);
        snprintf(p->reason, sizeof p->reason, "%s", r->reason);
        if (p->status == DS_OK) {
            p->t = r->makespan_ns;
            for (int i = 0; i < m->n_res; i++) if (m->res[i].kind == DS_RES_CAPACITY && r->peak_live_bytes[i] > p->peak) p->peak = r->peak_live_bytes[i];
            p->e = r->energy_j; p->e_meas = r->energy_measured; p->e_syn = r->energy_synthetic;
            if (!p->e_meas) all_energy = 0;
        } else all_energy = 0;
        free(m); free(r);
    }
    /* non-dominated on (time, peak bytes[, measured joules]) among OK rows */
    for (int a = 0; a < n; a++) {
        if (rows[a].status != DS_OK) continue;
        rows[a].front = 1;
        for (int b = 0; b < n; b++) {
            if (a == b || rows[b].status != DS_OK) continue;
            int le = rows[b].t <= rows[a].t && rows[b].peak <= rows[a].peak && (!all_energy || rows[b].e <= rows[a].e);
            int lt = rows[b].t < rows[a].t || rows[b].peak < rows[a].peak || (all_energy && rows[b].e < rows[a].e);
            if (le && lt) { rows[a].front = 0; break; }
        }
    }
    printf("# dagsim pareto: objectives = seconds, peak bytes%s. Energy %s.\n", all_energy ? ", joules" : "",
           all_energy ? "measured on every row and part of the dominance test" : "not measured on every row: shown, never ranked");
    printf("# %-36s %-24s %14s %14s %14s %12s %-14s %s\n", "file", "status", "seconds", "lo_s", "hi_s", "peak_bytes", "energy", "front");
    for (int k = 0; k < n; k++) {
        prow *p = &rows[k];
        char e[32];
        if (p->e_meas) snprintf(e, sizeof e, "%.3f J", p->e); else if (p->e_syn) snprintf(e, sizeof e, "syn %.3f J", p->e); else snprintf(e, sizeof e, "UNMEASURED");
        if (p->status == DS_OK)
            printf("  %-36s %-24s %14.6f %14.6f %14.6f %12.0f %-14s %s\n", p->path, "OK", p->t * 1e-9, p->lo * 1e-9, p->hi * 1e-9, p->peak, e, p->front ? "yes" : "no");
        else
            printf("  %-36s %-24s %14s %14s %14s %12s %-14s %s   # %s\n", p->path, ds_status_name(p->status), "-", "-", "-", "-", "-", "refused", p->reason);
    }
    return 0;
}

static int cmd_diff(const char *a, const char *b, const ds_opts *o) {
    ds_model *ma = malloc(sizeof *ma), *mb = malloc(sizeof *mb); char err[256]; ds_diff d;
    if (!ma || !mb) return 2;
    if (ds_parse_file(a, ma, err, sizeof err) != DS_OK) { fprintf(stderr, "%s: %s\n", a, err); return 1; }
    if (ds_parse_file(b, mb, err, sizeof err) != DS_OK) { fprintf(stderr, "%s: %s\n", b, err); return 1; }
    int rc = ds_diff_models(ma, mb, o, &d);
    static const char *rn[8] = { "", "eliminated", "reused", "unchanged_dep", "deferred", "moved", "shared", "lookup" };
    printf("{\"tool\": \"dagsim diff\", \"baseline\": "); jstr(a); printf(", \"transformed\": "); jstr(b);
    printf(", \"status\": "); jstr(ds_status_name(rc)); printf(", \"reason\": "); jstr(d.reason);
    printf(", \"kept\": %d, \"added\": %d, \"removed\": %d, \"double_counted\": %d, \"required_removed\": %d, \"removed_ns\": {", d.n_kept, d.n_added, d.n_removed, d.double_counted, d.required_removed);
    for (int i = 1; i < 8; i++) { if (i > 1) printf(", "); jstr(rn[i]); printf(": "); jnum(d.removed_ns[i]); }
    printf("}, \"changed\": "); jstr(o->changed ? o->changed : ""); printf("}\n");
    free(ma); free(mb);
    return rc == DS_OK ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: dagsim eval FILE | pareto FILE... | diff BASE TRANSFORMED  [--asof YYYY-MM-DD] [--model BLOB] [--changed a,b]\n"); return 2; }
    ds_opts o; memset(&o, 0, sizeof o); o.interval_q = 0.9;
    RxCostModel *cm = NULL;
    const char *files[64]; int nf = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--asof") && i + 1 < argc) { o.asof_day = ds_day_from_date(argv[++i]); if (o.asof_day < 0) { fprintf(stderr, "bad --asof\n"); return 2; } }
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) { cm = malloc(sizeof *cm); if (!cm || !load_model(argv[++i], cm)) { fprintf(stderr, "cannot load rx_costmodel blob %s\n", argv[i]); return 2; } o.costmodel = cm; }
        else if (!strcmp(argv[i], "--changed") && i + 1 < argc) o.changed = argv[++i];
        else if (nf < 64) files[nf++] = argv[i];
    }
    if (o.asof_day == 0) { fprintf(stderr, "dagsim: --asof YYYY-MM-DD is required so the run is reproducible (calibration validity is checked against it)\n"); return 2; }
    int rc;
    if (!strcmp(argv[1], "eval") && nf == 1) rc = cmd_eval(files[0], &o);
    else if (!strcmp(argv[1], "pareto") && nf >= 1) rc = cmd_pareto(nf, files, &o);
    else if (!strcmp(argv[1], "diff") && nf == 2) rc = cmd_diff(files[0], files[1], &o);
    else { fprintf(stderr, "usage: dagsim eval FILE | pareto FILE... | diff BASE TRANSFORMED\n"); rc = 2; }
    free(cm);
    return rc;
}
