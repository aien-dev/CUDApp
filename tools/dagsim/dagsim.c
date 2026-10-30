/*
 * dagsim.c -- offline resource-DAG cost evaluator (omega-mv#583).
 *
 * See dagsim.h and spec/offline-evidence-v1.md. The scheduler is a
 * deterministic fluid simulation: compute engines have slots, bandwidth
 * resources are shared equally among the ops streaming on them at any
 * instant, launches are serialized per launch resource, and an op finishes
 * when both its own busy time and its traffic are done (perfect overlap
 * inside one op, which is the issue's stated floor max(F/P, Q/B)). Ops are
 * dispatched in file order, so the same file always gives the same bytes.
 */
#include "dagsim.h"
#include "runtime/rx_costmodel.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------------------------------------------------------------- names */

const char *ds_status_name(int s) {
    switch (s) {
    case DS_OK: return "OK";
    case DS_REFUSED_UNKNOWN: return "REFUSED_UNKNOWN";
    case DS_REFUSED_STALE: return "REFUSED_STALE";
    case DS_REFUSED_NO_CALIBRATION: return "REFUSED_NO_CALIBRATION";
    case DS_REFUSED_CAPACITY: return "REFUSED_CAPACITY";
    case DS_REFUSED_CYCLE: return "REFUSED_CYCLE";
    case DS_REFUSED_UNSUPPORTED: return "REFUSED_UNSUPPORTED";
    case DS_REFUSED_REMOVED_REQUIRED: return "REFUSED_REMOVED_REQUIRED";
    case DS_PARSE_ERROR: return "PARSE_ERROR";
    default: return "?";
    }
}

const char *ds_prov_name(int p) {
    switch (p) {
    case DS_PROV_UNKNOWN: return "unknown";
    case DS_PROV_SYNTHETIC: return "synthetic";
    case DS_PROV_FITTED: return "fitted";
    case DS_PROV_MEASURED: return "measured";
    case DS_PROV_SPECIFIED: return "specified";
    default: return "?";
    }
}

static int prov_from_name(const char *s) {
    if (!strcmp(s, "unknown")) return DS_PROV_UNKNOWN;
    if (!strcmp(s, "synthetic")) return DS_PROV_SYNTHETIC;
    if (!strcmp(s, "fitted")) return DS_PROV_FITTED;
    if (!strcmp(s, "measured")) return DS_PROV_MEASURED;
    if (!strcmp(s, "specified")) return DS_PROV_SPECIFIED;
    return -1;
}

/* days since 1970-01-01 for a civil date (Howard Hinnant's days_from_civil) */
int ds_day_from_date(const char *s) {
    int y, m, d;
    if (!s || sscanf(s, "%d-%d-%d", &y, &m, &d) != 3) return -1;
    y -= m <= 2;
    int era = (y >= 0 ? y : y - 399) / 400;
    unsigned yoe = (unsigned)(y - era * 400);
    unsigned doy = (153u * (unsigned)(m + (m > 2 ? -3 : 9)) + 2u) / 5u + (unsigned)d - 1u;
    unsigned doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
    return era * 146097 + (int)doe - 719468;
}

int ds_find_op(const ds_model *m, const char *id) {
    for (int i = 0; i < m->n_ops; i++) if (!strcmp(m->ops[i].id, id)) return i;
    return -1;
}
int ds_find_res(const ds_model *m, const char *id) {
    for (int i = 0; i < m->n_res; i++) if (!strcmp(m->res[i].id, id)) return i;
    return -1;
}

/* ---------------------------------------------------------------- parse */
/* bounded string copy; ids and sources are truncated, never overflowed */
static void sset(char *dst, size_t cap, const char *src) {
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n); dst[n] = 0;
}


typedef struct { char key[32]; char val[DS_SRC]; int used; } kv;

static int split_kv(char *line, char *keyword, size_t kwcap, kv *kvs, int maxkv, char *positional[], int *npos) {
    int n = 0; *npos = 0; keyword[0] = 0;
    char *p = strchr(line, '#'); if (p) *p = 0;
    char *save = NULL;
    char *tok = strtok_r(line, " \t\r\n", &save);
    if (!tok) return -1;   /* empty */
    snprintf(keyword, kwcap, "%s", tok);
    while ((tok = strtok_r(NULL, " \t\r\n", &save))) {
        char *eq = strchr(tok, '=');
        if (!eq) { if (*npos < 8) positional[(*npos)++] = tok; continue; }
        if (n >= maxkv) return -2;
        *eq = 0;
        sset(kvs[n].key, sizeof kvs[n].key, tok);
        sset(kvs[n].val, sizeof kvs[n].val, eq + 1);
        kvs[n].used = 0;
        n++;
    }
    return n;
}

static const char *get(kv *kvs, int n, const char *key) {
    for (int i = 0; i < n; i++) if (!strcmp(kvs[i].key, key)) { kvs[i].used = 1; return kvs[i].val; }
    return NULL;
}

static int unused_key(kv *kvs, int n, char *out, size_t cap) {
    for (int i = 0; i < n; i++) if (!kvs[i].used) { snprintf(out, cap, "%s", kvs[i].key); return 1; }
    return 0;
}

static int parse_real(const char *s, double *out) {
    if (!s) return 0;
    char *end; double v = strtod(s, &end);
    if (end == s || *end) return 0;
    if (!(v == v) || v < 0) return 0;
    *out = v; return 1;
}

#define PERR(...) do { snprintf(err, errcap, "line %d: ", lineno); size_t _l = strlen(err); snprintf(err + _l, errcap - _l, __VA_ARGS__); return DS_PARSE_ERROR; } while (0)

static int parse_prov_src(kv *kvs, int n, int *prov, char *src, size_t srccap, int required, char *err, size_t errcap, int lineno) {
    const char *p = get(kvs, n, "prov");
    const char *s = get(kvs, n, "src");
    if (!p) { if (required) PERR("prov= is required wherever a number appears"); *prov = -1; return DS_OK; }
    int v = prov_from_name(p);
    if (v < 0) PERR("prov=%s is not one of specified|measured|fitted|synthetic|unknown", p);
    *prov = v;
    if ((v == DS_PROV_MEASURED || v == DS_PROV_FITTED) && !s) PERR("prov=%s needs src=", p);
    snprintf(src, srccap, "%s", s ? s : "");
    return DS_OK;
}

int ds_parse_text(const char *text, ds_model *m, char *err, size_t errcap) {
    memset(m, 0, sizeof *m);
    err[0] = 0;
    int lineno = 0;
    const char *cur = text;
    int seen_oev = 0, seen_machine = 0, seen_build = 0;
    /* two-pass is not needed: ops before deps/buffers is the convention; we
     * resolve references at use and require definition before use. */
    while (*cur) {
        const char *nl = strchr(cur, '\n');
        size_t len = nl ? (size_t)(nl - cur) : strlen(cur);
        char line[1024];
        lineno++;
        if (len >= sizeof line) PERR("line too long");
        memcpy(line, cur, len); line[len] = 0;
        cur = nl ? nl + 1 : cur + len;
        char keyword[32]; kv kvs[24]; char *pos[8]; int npos;
        int n = split_kv(line, keyword, sizeof keyword, kvs, 24, pos, &npos);
        if (n == -1) continue;
        if (n == -2) PERR("too many fields");
        if (!strcmp(keyword, "oev")) {
            const char *v = get(kvs, n, "version");
            if (!v || strcmp(v, "1")) PERR("oev version=1 required");
            m->version = 1; seen_oev = 1;
        } else if (!strcmp(keyword, "machine")) {
            const char *v;
            if (!(v = get(kvs, n, "id"))) PERR("machine needs id=");
            sset(m->machine_id, sizeof m->machine_id, v);
            if ((v = get(kvs, n, "soc"))) sset(m->soc, sizeof m->soc, v);
            if ((v = get(kvs, n, "cores"))) m->cores = atoi(v);
            if ((v = get(kvs, n, "mem_bytes")) && !parse_real(v, &m->mem_bytes)) PERR("bad mem_bytes");
            int prov; char src[DS_SRC];
            int rc = parse_prov_src(kvs, n, &prov, src, sizeof src, 1, err, errcap, lineno); if (rc) return rc;
            seen_machine = 1;
        } else if (!strcmp(keyword, "build")) {
            const char *v;
            if ((v = get(kvs, n, "repo"))) sset(m->repo, sizeof m->repo, v);
            if (!(v = get(kvs, n, "sha")) || strlen(v) != 40) PERR("build needs sha=<40 hex>");
            sset(m->build_sha, sizeof m->build_sha, v);
            if ((v = get(kvs, n, "physics"))) sset(m->physics, sizeof m->physics, v);
            seen_build = 1;
        } else if (!strcmp(keyword, "calibration")) {
            const char *v;
            if (m->has_calibration) PERR("one calibration record only");
            if (!(v = get(kvs, n, "id"))) PERR("calibration needs id=");
            sset(m->cal_id, sizeof m->cal_id, v);
            if (!(v = get(kvs, n, "taken")) || (m->cal_taken_day = ds_day_from_date(v)) < 0) PERR("calibration needs taken=YYYY-MM-DD");
            if (!(v = get(kvs, n, "valid_days"))) PERR("calibration needs valid_days=");
            m->cal_valid_days = atoi(v);
            if (!(v = get(kvs, n, "load1")) || !parse_real(v, &m->cal_load1)) PERR("calibration needs load1=");
            if (!(v = get(kvs, n, "src"))) PERR("calibration needs src=");
            sset(m->cal_src, sizeof m->cal_src, v);
            if ((v = get(kvs, n, "model"))) sset(m->cal_model, sizeof m->cal_model, v);
            m->has_calibration = 1;
        } else if (!strcmp(keyword, "resource")) {
            if (m->n_res >= DS_MAX_RES) PERR("too many resources (max %d)", DS_MAX_RES);
            ds_resource *r = &m->res[m->n_res];
            memset(r, 0, sizeof *r);
            const char *v;
            if (!(v = get(kvs, n, "id"))) PERR("resource needs id=");
            if (ds_find_res(m, v) >= 0) PERR("duplicate resource %s", v);
            sset(r->id, sizeof r->id, v);
            if (!(v = get(kvs, n, "kind"))) PERR("resource needs kind=");
            if (!strcmp(v, "compute")) {
                r->kind = DS_RES_COMPUTE;
                if (!(v = get(kvs, n, "slots")) || (r->slots = atoi(v)) < 1) PERR("compute resource needs slots>=1");
            } else if (!strcmp(v, "bandwidth")) {
                r->kind = DS_RES_BANDWIDTH;
                if (!parse_real(get(kvs, n, "cap"), &r->cap) || r->cap <= 0) PERR("bandwidth resource needs cap>0");
                r->lo = r->hi = r->cap;
                if ((v = get(kvs, n, "lo")) && !parse_real(v, &r->lo)) PERR("bad lo");
                if ((v = get(kvs, n, "hi")) && !parse_real(v, &r->hi)) PERR("bad hi");
                if (r->lo > r->cap || r->hi < r->cap) PERR("need lo <= cap <= hi");
            } else if (!strcmp(v, "capacity")) {
                r->kind = DS_RES_CAPACITY;
                if (!parse_real(get(kvs, n, "cap"), &r->cap) || r->cap <= 0) PERR("capacity resource needs cap>0");
            } else if (!strcmp(v, "launch")) {
                r->kind = DS_RES_LAUNCH;
            } else PERR("resource kind=%s unknown", v);
            int rc = parse_prov_src(kvs, n, &r->prov, r->src, sizeof r->src, 1, err, errcap, lineno); if (rc) return rc;
            m->n_res++;
        } else if (!strcmp(keyword, "op")) {
            if (m->n_ops >= DS_MAX_OPS) PERR("too many ops (max %d)", DS_MAX_OPS);
            ds_op *o = &m->ops[m->n_ops];
            memset(o, 0, sizeof *o);
            o->engine = o->bw = o->launcher = -1;
            const char *v;
            if (!(v = get(kvs, n, "id"))) PERR("op needs id=");
            if (ds_find_op(m, v) >= 0) PERR("duplicate op %s", v);
            sset(o->id, sizeof o->id, v);
            if (!(v = get(kvs, n, "kind"))) PERR("op needs kind=");
            if (!strcmp(v, "compute")) o->kind = DS_OP_COMPUTE;
            else if (!strcmp(v, "copy")) o->kind = DS_OP_COPY;
            else if (!strcmp(v, "matvec")) o->kind = DS_OP_MATVEC;
            else if (!strcmp(v, "barrier")) o->kind = DS_OP_BARRIER;
            else if (!strcmp(v, "storage")) o->kind = DS_OP_STORAGE;
            else if (!strcmp(v, "persist")) o->kind = DS_OP_PERSIST;
            else PERR("op kind=%s unknown", v);
            if ((v = get(kvs, n, "engine"))) {
                o->engine = ds_find_res(m, v);
                if (o->engine < 0 || m->res[o->engine].kind != DS_RES_COMPUTE) PERR("engine=%s is not a compute resource defined above", v);
            }
            if ((v = get(kvs, n, "t_ns"))) { if (!parse_real(v, &o->t_ns)) PERR("bad t_ns"); o->has_t = 1; o->lo = o->hi = o->t_ns; }
            if ((v = get(kvs, n, "lo"))) { if (!o->has_t || !parse_real(v, &o->lo) || o->lo > o->t_ns) PERR("lo needs t_ns and lo <= t_ns"); }
            if ((v = get(kvs, n, "hi"))) { if (!o->has_t || !parse_real(v, &o->hi) || o->hi < o->t_ns) PERR("hi needs t_ns and hi >= t_ns"); }
            if ((v = get(kvs, n, "bytes"))) {
                if (!parse_real(v, &o->bytes)) PERR("bad bytes");
                const char *b = get(kvs, n, "bw");
                if (!b || (o->bw = ds_find_res(m, b)) < 0 || m->res[o->bw].kind != DS_RES_BANDWIDTH) PERR("bytes= needs bw=<bandwidth resource defined above>");
            }
            if ((v = get(kvs, n, "launch_ns"))) {
                if (!parse_real(v, &o->launch_ns)) PERR("bad launch_ns");
                const char *l = get(kvs, n, "launcher");
                if (!l || (o->launcher = ds_find_res(m, l)) < 0 || m->res[o->launcher].kind != DS_RES_LAUNCH) PERR("launch_ns= needs launcher=<launch resource defined above>");
            }
            if ((v = get(kvs, n, "M"))) o->M = (uint32_t)strtoul(v, NULL, 10);
            if ((v = get(kvs, n, "N"))) o->N = (uint32_t)strtoul(v, NULL, 10);
            if ((v = get(kvs, n, "core"))) o->core = (uint32_t)strtoul(v, NULL, 10);
            if ((v = get(kvs, n, "arm"))) o->arm = (uint32_t)strtoul(v, NULL, 10);
            if (o->kind == DS_OP_MATVEC && !o->has_t && (o->M == 0 || o->N == 0)) PERR("matvec needs M= and N= (or t_ns=)");
            if (o->kind != DS_OP_BARRIER && o->engine < 0) PERR("op %s needs engine=", o->id);
            int required = o->has_t || o->bytes > 0 || o->launch_ns > 0 || o->kind == DS_OP_MATVEC;
            int rc = parse_prov_src(kvs, n, &o->t_prov, o->src, sizeof o->src, required, err, errcap, lineno); if (rc) return rc;
            if (o->t_prov < 0) o->t_prov = DS_PROV_SPECIFIED;   /* a costless barrier: ordering only */
            if (!o->has_t && o->kind != DS_OP_MATVEC && o->t_prov == DS_PROV_UNKNOWN) { /* fine: explicitly unknown */ }
            m->n_ops++;
        } else if (!strcmp(keyword, "dep")) {
            if (npos != 2 || n != 0) PERR("dep <from> <to>");
            if (m->n_deps >= DS_MAX_DEPS) PERR("too many deps");
            int a = ds_find_op(m, pos[0]), b = ds_find_op(m, pos[1]);
            if (a < 0 || b < 0) PERR("dep names an op not defined above: %s %s", pos[0], pos[1]);
            m->deps[m->n_deps].from = a; m->deps[m->n_deps].to = b; m->n_deps++;
        } else if (!strcmp(keyword, "buffer")) {
            if (m->n_buf >= DS_MAX_BUF) PERR("too many buffers");
            ds_buffer *b = &m->buf[m->n_buf];
            memset(b, 0, sizeof *b); b->first = b->last = -1;
            const char *v;
            if (!(v = get(kvs, n, "id"))) PERR("buffer needs id=");
            sset(b->id, sizeof b->id, v);
            if (!parse_real(get(kvs, n, "bytes"), &b->bytes)) PERR("buffer needs bytes=");
            if (!(v = get(kvs, n, "cap")) || (b->cap = ds_find_res(m, v)) < 0 || m->res[b->cap].kind != DS_RES_CAPACITY) PERR("buffer needs cap=<capacity resource>");
            if ((v = get(kvs, n, "first")) && (b->first = ds_find_op(m, v)) < 0) PERR("first=%s not an op", v);
            if ((v = get(kvs, n, "last")) && (b->last = ds_find_op(m, v)) < 0) PERR("last=%s not an op", v);
            m->n_buf++;
        } else if (!strcmp(keyword, "sync")) {
            /* a barrier op that depends on every listed op */
            if (m->n_ops >= DS_MAX_OPS) PERR("too many ops");
            const char *v, *list;
            if (!(v = get(kvs, n, "id"))) PERR("sync needs id=");
            if (ds_find_op(m, v) >= 0) PERR("duplicate op %s", v);
            if (!(list = get(kvs, n, "ops"))) PERR("sync needs ops=a,b,...");
            ds_op *o = &m->ops[m->n_ops];
            memset(o, 0, sizeof *o);
            o->engine = o->bw = o->launcher = -1; o->kind = DS_OP_BARRIER; o->t_prov = DS_PROV_SPECIFIED;
            sset(o->id, sizeof o->id, v);
            int self = m->n_ops++;
            char tmp[DS_SRC]; sset(tmp, sizeof tmp, list);
            char *save = NULL;
            for (char *t = strtok_r(tmp, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
                int a = ds_find_op(m, t);
                if (a < 0 || a == self) PERR("sync lists unknown op %s", t);
                if (m->n_deps >= DS_MAX_DEPS) PERR("too many deps");
                m->deps[m->n_deps].from = a; m->deps[m->n_deps].to = self; m->n_deps++;
            }
        } else if (!strcmp(keyword, "energy")) {
            const char *v;
            if (m->has_energy) PERR("one energy record only");
            if (!(v = get(kvs, n, "instrument"))) PERR("energy needs instrument=");
            sset(m->instrument, sizeof m->instrument, v);
            if ((v = get(kvs, n, "boundary"))) sset(m->boundary, sizeof m->boundary, v);
            const char *j = get(kvs, n, "joules"), *pw = get(kvs, n, "power_w");
            int prov = -1; char src[DS_SRC];
            int rc = parse_prov_src(kvs, n, &prov, src, sizeof src, j || pw ? 1 : 0, err, errcap, lineno); if (rc) return rc;
            m->energy_prov = prov;
            if (j) {
                if (!strcmp(m->instrument, "none") || !m->boundary[0]) PERR("joules= needs a named instrument and boundary=");
                if (prov != DS_PROV_MEASURED) PERR("joules= must be prov=measured with src=");
                if (!parse_real(j, &m->joules)) PERR("bad joules");
            }
            if (pw) {
                if (prov != DS_PROV_SYNTHETIC) PERR("power_w= is allowed only as prov=synthetic (a sensitivity figure); measured power must be integrated into joules= by the instrument");
                if (!parse_real(pw, &m->power_w)) PERR("bad power_w");
            }
            m->has_energy = 1;
        } else if (!strcmp(keyword, "removed")) {
            if (m->n_removed >= DS_MAX_REMOVED) PERR("too many removed records");
            ds_removed *r = &m->removed[m->n_removed];
            memset(r, 0, sizeof *r);
            const char *v;
            if (!(v = get(kvs, n, "op"))) PERR("removed needs op=");
            sset(r->op, sizeof r->op, v);
            if (ds_find_op(m, v) >= 0) PERR("removed op %s is still present in this DAG", v);
            if (!(v = get(kvs, n, "reason"))) PERR("removed needs reason=");
            if (!strcmp(v, "eliminated")) r->reason = DS_RM_ELIMINATED;
            else if (!strcmp(v, "reused")) r->reason = DS_RM_REUSED;
            else if (!strcmp(v, "unchanged_dep")) r->reason = DS_RM_UNCHANGED_DEP;
            else if (!strcmp(v, "deferred")) r->reason = DS_RM_DEFERRED;
            else if (!strcmp(v, "moved")) r->reason = DS_RM_MOVED;
            else if (!strcmp(v, "shared")) r->reason = DS_RM_SHARED;
            else if (!strcmp(v, "lookup")) r->reason = DS_RM_LOOKUP;
            else PERR("removed reason=%s unknown", v);
            if ((v = get(kvs, n, "ref"))) sset(r->ref, sizeof r->ref, v);
            if ((v = get(kvs, n, "proof"))) sset(r->proof, sizeof r->proof, v);
            m->n_removed++;
        } else PERR("unknown record '%s'", keyword);
        char bad[32];
        if (unused_key(kvs, n, bad, sizeof bad)) PERR("unknown field %s= on '%s'", bad, keyword);
    }
    lineno = 0;
    if (!seen_oev) PERR("missing 'oev version=1'");
    if (!seen_machine) PERR("missing machine record");
    if (!seen_build) PERR("missing build record");
    if (!m->has_energy) PERR("missing energy record (use instrument=none for UNMEASURED)");
    return DS_OK;
}

int ds_parse_file(const char *path, ds_model *m, char *err, size_t errcap) {
    FILE *fp = fopen(path, "r");
    if (!fp) { snprintf(err, errcap, "cannot open %s", path); return DS_PARSE_ERROR; }
    fseek(fp, 0, SEEK_END); long n = ftell(fp); fseek(fp, 0, SEEK_SET);
    if (n < 0 || n > (1 << 20)) { fclose(fp); snprintf(err, errcap, "file size out of range"); return DS_PARSE_ERROR; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(fp); snprintf(err, errcap, "out of memory"); return DS_PARSE_ERROR; }
    size_t got = fread(buf, 1, (size_t)n, fp); buf[got] = 0; fclose(fp);
    int rc = ds_parse_text(buf, m, err, errcap);
    free(buf);
    return rc;
}

/* ---------------------------------------------------------------- evaluate */

/* resolved per-op numbers for one evaluation mode */
typedef struct { double t, launch, bytes, cap; int prov; } ds_num;   /* cap in B/ns */

static double res_cap(const ds_resource *r, int mode) {
    if (mode < 0) return r->lo;
    if (mode > 0) return r->hi;
    return r->cap;
}

static int refuse(ds_result *r, int status, const char *fmt, const char *a, const char *b) {
    memset(r, 0, sizeof *r);
    r->status = status;
    snprintf(r->reason, sizeof r->reason, fmt, a ? a : "", b ? b : "");
    return status;
}

static int resolve(const ds_model *m, const ds_opts *o, ds_num *num, ds_result *r) {
    int any_cal = 0;
    for (int i = 0; i < m->n_res; i++) {
        if (m->res[i].prov == DS_PROV_UNKNOWN) return refuse(r, DS_REFUSED_UNKNOWN, "resource %s has prov=unknown; unknown stays unknown%s", m->res[i].id, "");
        if (m->res[i].prov == DS_PROV_MEASURED || m->res[i].prov == DS_PROV_FITTED) any_cal = 1;
    }
    for (int i = 0; i < m->n_ops; i++) {
        const ds_op *op = &m->ops[i];
        ds_num *x = &num[i];
        memset(x, 0, sizeof *x);
        x->prov = op->t_prov;
        x->launch = op->launch_ns;
        x->bytes = op->bytes;
        x->cap = op->bw >= 0 ? res_cap(&m->res[op->bw], o->mode) * 1e-9 : 0;   /* B/s -> B/ns */
        if (op->kind == DS_OP_MATVEC && !op->has_t) {
            const RxCostModel *cm = (const RxCostModel *)o->costmodel;
            if (!cm) return refuse(r, DS_REFUSED_UNKNOWN, "op %s is a matvec with no rx_costmodel blob supplied (calibration model=)%s", op->id, "");
            RxCmFeatures f; memset(&f, 0, sizeof f);
            f.op = RX_CM_OP_MATVEC; f.M = op->M; f.N = op->N;
            f.state_bytes = ((uint64_t)op->M * op->N + op->M + op->N) * 8u;
            f.cache_bytes = 1u << 20; f.core = op->core; f.eligible = 1u << op->arm; f.current_arm = op->arm;
            RxCmPrediction p; rx_cm_predict(cm, &f, op->arm, &p);
            if (!p.known || p.n == 0) return refuse(r, DS_REFUSED_UNKNOWN, "op %s: rx_costmodel has no known cell for this (op, core, pressure) and arm; measure it first%s", op->id, "");
            double q = o->interval_q > 0 && o->interval_q < 1 ? o->interval_q : 0.9;
            double k = rx_cm_t_quantile(q, p.dof);
            double l2 = p.mean_log2_ps + (o->mode < 0 ? k * p.sd_log2 : o->mode > 0 ? -k * p.sd_log2 : 0);
            x->t = pow(2.0, l2) / 1000.0;   /* ps -> ns */
            x->prov = DS_PROV_FITTED;
            any_cal = 1;
        } else {
            if (op->t_prov == DS_PROV_UNKNOWN) return refuse(r, DS_REFUSED_UNKNOWN, "op %s has prov=unknown (%s); unknown stays unknown", op->id, op->src[0] ? op->src : "no source");
            x->t = o->mode < 0 ? op->hi : o->mode > 0 ? op->lo : op->t_ns;
            if (op->t_prov == DS_PROV_MEASURED || op->t_prov == DS_PROV_FITTED) any_cal = 1;
        }
        if (op->engine < 0 && op->kind != DS_OP_BARRIER) return refuse(r, DS_REFUSED_UNSUPPORTED, "op %s has no engine%s", op->id, "");
        if (op->launcher >= 0 && op->engine < 0) return refuse(r, DS_REFUSED_UNSUPPORTED, "op %s: launch without an engine%s", op->id, "");
    }
    if (any_cal) {
        if (!m->has_calibration) return refuse(r, DS_REFUSED_NO_CALIBRATION, "measured or fitted values need a calibration record naming when and under what load they were taken%s%s", "", "");
        int asof = o->asof_day;
        if (asof <= 0) asof = (int)(time(NULL) / 86400);
        if (asof > m->cal_taken_day + m->cal_valid_days) {
            char d[64]; snprintf(d, sizeof d, "asof is %d day(s) past the validity envelope", asof - m->cal_taken_day - m->cal_valid_days);
            return refuse(r, DS_REFUSED_STALE, "calibration %s is stale: %s", m->cal_id, d);
        }
    }
    return DS_OK;
}

/* Kahn topological order; returns 0 on a cycle */
static int topo(const ds_model *m, int *order) {
    int indeg[DS_MAX_OPS] = {0};
    for (int i = 0; i < m->n_deps; i++) indeg[m->deps[i].to]++;
    int n = 0, head = 0;
    for (int i = 0; i < m->n_ops; i++) if (indeg[i] == 0) order[n++] = i;
    while (head < n) {
        int u = order[head++];
        for (int i = 0; i < m->n_deps; i++) if (m->deps[i].from == u && --indeg[m->deps[i].to] == 0) order[n++] = m->deps[i].to;
    }
    return n == m->n_ops;
}

enum { ST_WAIT = 0, ST_READY, ST_LAUNCH, ST_RUN, ST_DONE };
enum { BLK_NONE = 0, BLK_SLOT, BLK_LAUNCHER };

int ds_eval(const ds_model *m, const ds_opts *o, ds_result *r) {
    memset(r, 0, sizeof *r);
    if (m->n_ops == 0) return refuse(r, DS_REFUSED_UNSUPPORTED, "empty DAG%s%s", "", "");
    ds_num num[DS_MAX_OPS];
    int rc = resolve(m, o, num, r);
    if (rc) return rc;
    int order[DS_MAX_OPS];
    if (!topo(m, order)) return refuse(r, DS_REFUSED_CYCLE, "dependency cycle%s%s", "", "");
    /* a buffer that never fits */
    for (int i = 0; i < m->n_buf; i++)
        if (m->buf[i].bytes > m->res[m->buf[i].cap].cap)
            return refuse(r, DS_REFUSED_CAPACITY, "buffer %s alone exceeds capacity %s", m->buf[i].id, m->res[m->buf[i].cap].id);

    /* provenance mask */
    for (int i = 0; i < m->n_ops; i++) r->prov_mask |= 1u << num[i].prov;
    for (int i = 0; i < m->n_res; i++) {
        int used = 0;
        for (int j = 0; j < m->n_ops; j++) if (m->ops[j].engine == i || m->ops[j].bw == i || m->ops[j].launcher == i) used = 1;
        for (int j = 0; j < m->n_buf; j++) if (m->buf[j].cap == i) used = 1;
        if (used) r->prov_mask |= 1u << m->res[i].prov;
    }

    /* ---- fluid simulation ---- */
    int state[DS_MAX_OPS] = {0}, blocked[DS_MAX_OPS] = {0}, npred[DS_MAX_OPS] = {0}, done_pred[DS_MAX_OPS] = {0};
    double rem_t[DS_MAX_OPS], rem_b[DS_MAX_OPS], rem_l[DS_MAX_OPS], run_start[DS_MAX_OPS], bytes_end[DS_MAX_OPS];
    int free_slots[DS_MAX_RES], launcher_busy[DS_MAX_RES] = {0};
    for (int i = 0; i < m->n_res; i++) free_slots[i] = m->res[i].slots;
    for (int i = 0; i < m->n_deps; i++) npred[m->deps[i].to]++;
    for (int i = 0; i < m->n_ops; i++) {
        rem_t[i] = num[i].t; rem_b[i] = num[i].bytes; rem_l[i] = num[i].launch;
        run_start[i] = bytes_end[i] = -1;
        r->n_ops++;
        r->total_bytes += num[i].bytes; r->total_busy_ns += num[i].t; r->total_launch_ns += num[i].launch;
    }
    double t = 0;
    int remaining = m->n_ops, guard = 0;
    while (remaining > 0) {
        if (++guard > 8 * DS_MAX_OPS + 16) return refuse(r, DS_REFUSED_UNSUPPORTED, "scheduler did not converge%s%s", "", "");
        /* deps satisfied -> ready */
        for (int i = 0; i < m->n_ops; i++) if (state[i] == ST_WAIT && done_pred[i] == npred[i]) { state[i] = ST_READY; r->ready_ns[i] = t; }
        /* dispatch in file order */
        for (int i = 0; i < m->n_ops; i++) {
            if (state[i] != ST_READY) continue;
            const ds_op *op = &m->ops[i];
            if (op->engine >= 0 && free_slots[op->engine] <= 0) { blocked[i] = BLK_SLOT; continue; }
            if (op->launcher >= 0 && rem_l[i] > 0 && launcher_busy[op->launcher]) { blocked[i] = BLK_LAUNCHER; continue; }
            if (op->engine >= 0) free_slots[op->engine]--;
            r->start_ns[i] = t;
            if (op->launcher >= 0 && rem_l[i] > 0) { launcher_busy[op->launcher] = 1; state[i] = ST_LAUNCH; }
            else { state[i] = ST_RUN; run_start[i] = t; }
        }
        /* next event */
        int active[DS_MAX_RES] = {0};
        for (int i = 0; i < m->n_ops; i++) if (state[i] == ST_RUN && rem_b[i] > 0) active[m->ops[i].bw]++;
        double dt = -1; int any = 0;
        for (int i = 0; i < m->n_ops; i++) {
            if (state[i] == ST_LAUNCH) { any = 1; if (dt < 0 || rem_l[i] < dt) dt = rem_l[i]; }
            else if (state[i] == ST_RUN) {
                any = 1;
                /* next event for this op: the earlier of its two progress bars */
                double d = -1;
                if (rem_t[i] > 0) d = rem_t[i];
                if (rem_b[i] > 0) {
                    double share = num[i].cap / active[m->ops[i].bw];
                    double db = rem_b[i] / share;
                    if (d < 0 || db < d) d = db;
                }
                if (d < 0) d = 0;
                if (dt < 0 || d < dt) dt = d;
            }
        }
        if (!any) return refuse(r, DS_REFUSED_UNSUPPORTED, "no op can make progress (slots=0 or all blocked)%s%s", "", "");
        if (dt < 0) dt = 0;
        t += dt;
        const double eps = 1e-9;
        for (int i = 0; i < m->n_ops; i++) {
            if (state[i] == ST_LAUNCH) {
                rem_l[i] -= dt;
                if (rem_l[i] <= eps) { rem_l[i] = 0; launcher_busy[m->ops[i].launcher] = 0; state[i] = ST_RUN; run_start[i] = t; }
            } else if (state[i] == ST_RUN) {
                if (rem_t[i] > 0) { rem_t[i] -= dt; if (rem_t[i] <= eps) rem_t[i] = 0; }
                if (rem_b[i] > 0) {
                    double share = num[i].cap / active[m->ops[i].bw];
                    rem_b[i] -= share * dt;
                    if (rem_b[i] <= eps * (num[i].bytes > 1 ? num[i].bytes : 1)) { rem_b[i] = 0; bytes_end[i] = t; }
                }
                if (rem_t[i] <= 0 && rem_b[i] <= 0) {
                    state[i] = ST_DONE; r->end_ns[i] = t; remaining--;
                    if (m->ops[i].engine >= 0) free_slots[m->ops[i].engine]++;
                    for (int d = 0; d < m->n_deps; d++) if (m->deps[d].from == i) done_pred[m->deps[d].to]++;
                    if (num[i].bytes > 0) {
                        double ideal = num[i].bytes / num[i].cap;
                        double actual = (bytes_end[i] >= 0 ? bytes_end[i] : t) - run_start[i];
                        r->op_contention_ns[i] = actual - ideal > 0 ? actual - ideal : 0;
                    }
                }
            }
        }
        if (t > r->makespan_ns) r->makespan_ns = t;
    }

    /* ---- bounds ---- */
    double cp[DS_MAX_OPS];
    for (int k = 0; k < m->n_ops; k++) {
        int i = order[k];
        double floor_ = num[i].t;
        if (num[i].bytes > 0 && num[i].bytes / num[i].cap > floor_) floor_ = num[i].bytes / num[i].cap;
        floor_ += num[i].launch;
        r->op_floor_ns[i] = floor_;
        double best = 0;
        for (int d = 0; d < m->n_deps; d++) if (m->deps[d].to == i && cp[m->deps[d].from] > best) best = cp[m->deps[d].from];
        cp[i] = best + floor_;
        if (cp[i] > r->bound_critical_path_ns) r->bound_critical_path_ns = cp[i];
    }
    r->bound_ns = r->bound_critical_path_ns;
    for (int i = 0; i < m->n_res; i++) {
        double w = 0;
        for (int j = 0; j < m->n_ops; j++) {
            if (m->res[i].kind == DS_RES_BANDWIDTH && m->ops[j].bw == i) w += num[j].bytes / num[j].cap;
            if (m->res[i].kind == DS_RES_COMPUTE && m->ops[j].engine == i) w += (num[j].t + num[j].launch) / m->res[i].slots;
            if (m->res[i].kind == DS_RES_LAUNCH && m->ops[j].launcher == i) w += num[j].launch;
        }
        r->bound_resource_ns[i] = w;
        if (w > r->bound_ns) r->bound_ns = w;
    }

    /* ---- capacity ledger: frees before allocations at the same instant ---- */
    for (int c = 0; c < m->n_res; c++) {
        if (m->res[c].kind != DS_RES_CAPACITY) continue;
        double live = 0, peak = 0, peak_at = 0;
        /* event list */
        double et[2 * DS_MAX_BUF]; double ev[2 * DS_MAX_BUF]; int en = 0;
        for (int b = 0; b < m->n_buf; b++) {
            if (m->buf[b].cap != c) continue;
            double s = m->buf[b].first >= 0 ? r->start_ns[m->buf[b].first] : 0;
            double e = m->buf[b].last >= 0 ? r->end_ns[m->buf[b].last] : r->makespan_ns;
            et[en] = s; ev[en] = m->buf[b].bytes; en++;
            et[en] = e; ev[en] = -m->buf[b].bytes; en++;
        }
        /* insertion sort by time, frees (negative) first on ties; stable */
        for (int i = 1; i < en; i++) {
            double kt = et[i], kv_ = ev[i]; int j = i - 1;
            while (j >= 0 && (et[j] > kt || (et[j] == kt && ev[j] > 0 && kv_ < 0))) { et[j + 1] = et[j]; ev[j + 1] = ev[j]; j--; }
            et[j + 1] = kt; ev[j + 1] = kv_;
        }
        for (int i = 0; i < en; i++) { live += ev[i]; if (live > peak) { peak = live; peak_at = et[i]; } }
        r->peak_live_bytes[c] = peak; r->peak_live_at_ns[c] = peak_at;
        if (peak > m->res[c].cap + 1e-9) {
            char names[160] = ""; size_t L = 0;
            for (int b = 0; b < m->n_buf && L < sizeof names - 8; b++) {
                if (m->buf[b].cap != c) continue;
                double s = m->buf[b].first >= 0 ? r->start_ns[m->buf[b].first] : 0;
                double e = m->buf[b].last >= 0 ? r->end_ns[m->buf[b].last] : r->makespan_ns;
                if (s <= peak_at && e > peak_at) { snprintf(names + L, sizeof names - L, "%s%s", L ? "," : "", m->buf[b].id); L = strlen(names); }
            }
            char what[200]; snprintf(what, sizeof what, "%.0f B live at t=%.0f ns (%s) exceeds", peak, peak_at, names);
            int st = refuse(r, DS_REFUSED_CAPACITY, "%s %s", what, m->res[c].id);
            return st;
        }
    }

    /* ---- attribution along the schedule's critical chain ---- */
    int cur = 0;
    for (int i = 1; i < m->n_ops; i++) if (r->end_ns[i] > r->end_ns[cur]) cur = i;
    double accounted = 0;
    int chain[DS_MAX_OPS]; int nc = 0;
    while (cur >= 0 && nc < DS_MAX_OPS) {
        chain[nc++] = cur;
        const ds_op *op = &m->ops[cur];
        double runtime = r->end_ns[cur] - (run_start[cur] >= 0 ? run_start[cur] : r->start_ns[cur]);
        r->attr_ns[DS_ATTR_LAUNCH] += num[cur].launch;
        accounted += num[cur].launch;
        if (op->kind == DS_OP_BARRIER) r->attr_ns[DS_ATTR_SYNC] += runtime;
        else if (num[cur].bytes > 0) {
            double ideal = num[cur].bytes / num[cur].cap;
            double actual = ideal + r->op_contention_ns[cur];
            if (num[cur].t >= actual) r->attr_ns[DS_ATTR_COMPUTE] += runtime;
            else { r->attr_ns[DS_ATTR_BANDWIDTH] += ideal; r->attr_ns[DS_ATTR_CONTENTION] += r->op_contention_ns[cur]; if (runtime > actual) r->attr_ns[DS_ATTR_RESIDUAL] += runtime - actual; }
        } else r->attr_ns[DS_ATTR_COMPUTE] += runtime;
        accounted += runtime;
        /* predecessor: the dep that finished last */
        int pred = -1;
        for (int d = 0; d < m->n_deps; d++) if (m->deps[d].to == cur && (pred < 0 || r->end_ns[m->deps[d].from] > r->end_ns[pred])) pred = m->deps[d].from;
        double gap = r->start_ns[cur] - (pred >= 0 ? r->end_ns[pred] : 0);
        if (gap > 0) {
            if (blocked[cur] == BLK_SLOT) r->attr_ns[DS_ATTR_SERIALIZATION] += gap;
            else if (blocked[cur] == BLK_LAUNCHER) r->attr_ns[DS_ATTR_LAUNCH] += gap;
            else r->attr_ns[DS_ATTR_RESIDUAL] += gap;
            accounted += gap;
        }
        cur = pred;
    }
    for (int i = 0; i < nc; i++) r->critical_path[i] = chain[nc - 1 - i];
    r->n_critical = nc;
    double leftover = r->makespan_ns - accounted;
    if (fabs(leftover) > 1e-6) r->attr_ns[DS_ATTR_RESIDUAL] += leftover;

    /* ---- energy ---- */
    if (m->has_energy && m->joules > 0 && strcmp(m->instrument, "none")) { r->energy_measured = 1; r->energy_j = m->joules; }
    else if (m->has_energy && m->power_w > 0 && m->energy_prov == DS_PROV_SYNTHETIC) { r->energy_synthetic = 1; r->energy_j = m->power_w * r->makespan_ns * 1e-9; r->prov_mask |= 1u << DS_PROV_SYNTHETIC; }
    r->synthetic = (r->prov_mask & (1u << DS_PROV_SYNTHETIC)) ? 1 : 0;
    r->status = DS_OK;
    return DS_OK;
}

int ds_eval_interval(const ds_model *m, const ds_opts *o, ds_result *nominal, double *lo_ns, double *hi_ns) {
    ds_opts x = *o; x.mode = 0;
    int rc = ds_eval(m, &x, nominal);
    *lo_ns = *hi_ns = 0;
    if (rc) return rc;
    ds_result tmp;
    x.mode = +1; if (ds_eval(m, &x, &tmp) == DS_OK) *lo_ns = tmp.makespan_ns;
    x.mode = -1; if (ds_eval(m, &x, &tmp) == DS_OK) *hi_ns = tmp.makespan_ns;
    return rc;
}

double ds_counterfactual(const ds_model *m, const ds_opts *o, int res_index) {
    if (res_index < 0 || res_index >= m->n_res) return -DS_REFUSED_UNSUPPORTED;
    ds_model *c = malloc(sizeof *c);
    if (!c) return -DS_REFUSED_UNSUPPORTED;
    *c = *m;
    ds_resource *r = &c->res[res_index];
    switch (r->kind) {
    case DS_RES_BANDWIDTH: r->cap *= 2; r->lo *= 2; r->hi *= 2; break;
    case DS_RES_COMPUTE: r->slots *= 2; break;
    case DS_RES_LAUNCH: for (int i = 0; i < c->n_ops; i++) if (c->ops[i].launcher == res_index) c->ops[i].launch_ns = 0; break;
    case DS_RES_CAPACITY: r->cap *= 2; break;
    }
    ds_result res; ds_opts x = *o; x.mode = 0;
    int rc = ds_eval(c, &x, &res);
    free(c);
    return rc == DS_OK ? res.makespan_ns : -(double)rc;
}

/* ---------------------------------------------------------------- diff */

static int in_changed(const char *list, const char *id) {
    if (!list) return 0;
    const char *p = list; size_t L = strlen(id);
    while (*p) {
        const char *q = strchr(p, ','); size_t n = q ? (size_t)(q - p) : strlen(p);
        if (n == L && !strncmp(p, id, L)) return 1;
        if (!q) break;
        p = q + 1;
    }
    return 0;
}

int ds_diff_models(const ds_model *b, const ds_model *t, const ds_opts *o, ds_diff *d) {
    memset(d, 0, sizeof *d);
    for (int i = 0; i < t->n_ops; i++) if (ds_find_op(b, t->ops[i].id) < 0) d->n_added++;
    for (int i = 0; i < b->n_ops; i++) {
        int j = ds_find_op(t, b->ops[i].id);
        if (j >= 0) {
            d->n_kept++;
            if (b->ops[i].t_ns != t->ops[j].t_ns || b->ops[i].bytes != t->ops[j].bytes)
                snprintf(d->reason, sizeof d->reason, "op %s kept but its demand changed; trace conservation broken", b->ops[i].id);
            continue;
        }
        d->n_removed++;
        /* find its removed record(s) */
        int found = 0; const ds_removed *rec = NULL;
        for (int k = 0; k < t->n_removed; k++) if (!strcmp(t->removed[k].op, b->ops[i].id)) { found++; rec = &t->removed[k]; }
        if (found == 0) { d->required_removed = 1; snprintf(d->reason, sizeof d->reason, "op %s vanished without a removed record", b->ops[i].id); return DS_REFUSED_REMOVED_REQUIRED; }
        if (found > 1) { d->double_counted = 1; snprintf(d->reason, sizeof d->reason, "op %s removed %d times (double counted)", b->ops[i].id, found); return DS_REFUSED_REMOVED_REQUIRED; }
        /* consumers of the removed op in the baseline that still exist */
        int has_consumer = 0;
        for (int k = 0; k < b->n_deps; k++) if (b->deps[k].from == i && ds_find_op(t, b->ops[b->deps[k].to].id) >= 0) has_consumer = 1;
        int ok = 1;
        switch (rec->reason) {
        case DS_RM_ELIMINATED: if (has_consumer && !rec->proof[0]) ok = 0; break;
        case DS_RM_REUSED: case DS_RM_LOOKUP: case DS_RM_MOVED: if (!rec->ref[0] || ds_find_op(t, rec->ref) < 0) ok = 0; break;
        case DS_RM_UNCHANGED_DEP: if (!rec->ref[0] || ds_find_op(b, rec->ref) < 0 || in_changed(o ? o->changed : NULL, rec->ref)) ok = 0; break;
        case DS_RM_DEFERRED: if (has_consumer) ok = 0; break;
        case DS_RM_SHARED:
            if (!rec->ref[0] || ds_find_op(t, rec->ref) < 0) ok = 0;
            for (int k = 0; k < t->n_removed; k++) if (!strcmp(t->removed[k].op, rec->ref)) { d->double_counted = 1; ok = 0; }
            break;
        }
        if (!ok) {
            d->required_removed = 1;
            snprintf(d->reason, sizeof d->reason, "op %s: removal reason %d not supported by its evidence (ref=%s proof=%s changed=%s)",
                     b->ops[i].id, rec->reason, rec->ref, rec->proof, o && o->changed ? o->changed : "");
            return DS_REFUSED_REMOVED_REQUIRED;
        }
        double w = b->ops[i].t_ns + b->ops[i].launch_ns;
        if (b->ops[i].bytes > 0 && b->ops[i].bw >= 0) { double tb = b->ops[i].bytes / (b->res[b->ops[i].bw].cap * 1e-9); if (tb > b->ops[i].t_ns) w = tb + b->ops[i].launch_ns; }
        d->removed_ns[rec->reason] += w;
    }
    /* removed records that name ops never in the baseline are fabricated savings */
    for (int k = 0; k < t->n_removed; k++) if (ds_find_op(b, t->removed[k].op) < 0) {
        d->double_counted = 1;
        snprintf(d->reason, sizeof d->reason, "removed record for %s, which was never in the baseline", t->removed[k].op);
        return DS_REFUSED_REMOVED_REQUIRED;
    }
    return DS_OK;
}
