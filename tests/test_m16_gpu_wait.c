/* Host test for src/m16_gpu_wait.c: fake memory, fake clock, no GPU.
 * Every scenario runs against the real primitive (must behave) and against
 * mutants (a naive wall-clock loop, a fixed wait that accepts overshoot, a
 * sequence wait with an unsigned compare). Each mutant must be killed by the
 * scenarios named for it, otherwise this test fails. */
#include "m16_gpu_wait.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    uint64_t t_ns;
    uint64_t step_ns; /* clock advance per relax */
    uint64_t relaxes;
    uint64_t barriers;
    uint64_t at[32];
    volatile uint32_t *target[32];
    uint32_t value[32];
    int nsched;
    uint32_t invalid_value; /* valid() rejects this value when nonzero */
    int abort_at;           /* abort() true once relaxes >= abort_at when nonzero */
    uint64_t bar_at;        /* the bar_at-th barrier call lands bar_value in bar_target (a GPU write that becomes visible at the fence) */
    volatile uint32_t *bar_target;
    uint32_t bar_value;
} Fake;

static uint64_t fk_now(void *c) { return ((Fake *)c)->t_ns; }
static void fk_barrier(void *c) {
    Fake *f = c;
    f->barriers++;
    if (f->bar_at && f->barriers == f->bar_at) *f->bar_target = f->bar_value;
}
static void fk_relax(void *c) {
    Fake *f = c;
    f->relaxes++;
    f->t_ns += f->step_ns;
    for (int i = 0; i < f->nsched; i++)
        if (f->at[i] == f->relaxes) *f->target[i] = f->value[i];
}
static bool fk_valid(void *c, uint32_t v) { return !((Fake *)c)->invalid_value || v != ((Fake *)c)->invalid_value; }
static bool fk_abort(void *c) { Fake *f = c; return f->abort_at && f->relaxes >= (uint64_t)f->abort_at; }

typedef bool (*WaitFn)(volatile uint32_t *, uint32_t, m16_gpu_wait_report_t *, const m16_gpu_wait_cfg_t *);

/* Mutants. mode 0: naive wall clock (single deadline, unsigned <, no progress
 * tracking, no marker2, no barrier, no report). mode 1: exact for sequence too
 * is not needed; mode "fixed_ge": signed >= for a fixed wait. mode "seq_unsigned":
 * unsigned >= for a sequence wait. The last two keep everything else correct. */
static bool naive_wait(volatile uint32_t *word, uint32_t want, m16_gpu_wait_report_t *rpt,
                       const m16_gpu_wait_cfg_t *cfg) {
    uint64_t start = cfg->now_ns(cfg->hook_ctx);
    uint64_t lim = 1000000ull * cfg->progress_timeout_ms;
    if (rpt) memset(rpt, 0, sizeof(*rpt));
    while (*word < want) {
        if (cfg->now_ns(cfg->hook_ctx) - start >= lim) return false;
        cfg->relax(cfg->hook_ctx);
    }
    return true;
}
static bool mut_loop(int cmp, volatile uint32_t *w, uint32_t want, m16_gpu_wait_report_t *rpt,
                     const m16_gpu_wait_cfg_t *cfg, m16_gpu_wait_kind_t kind) {
    /* Correct kind-aware wait with one comparison replaced; delegate the rest
     * to the real primitive by wrapping the watched value would be circular, so
     * implement the minimal real loop here. */
    uint64_t start = cfg->now_ns(cfg->hook_ctx), last = start;
    uint64_t stall = 1000000ull * cfg->progress_timeout_ms;
    uint32_t prev = *w;
    if (rpt) { memset(rpt, 0, sizeof(*rpt)); rpt->wait_kind = kind; rpt->expected = want; }
    for (;;) {
        cfg->barrier(cfg->hook_ctx);
        uint32_t m = *w, m2 = cfg->marker2 ? *cfg->marker2 : 0;
        bool ok = cmp == 0 ? m == want : cmp == 1 ? (int32_t)(m - want) >= 0 : m >= want;
        if (rpt) { rpt->last_observed = m; rpt->last_marker2 = m2; }
        if (ok && (!cfg->marker2 || m2 == cfg->marker2_want)) {
            cfg->barrier(cfg->hook_ctx);
            if (rpt) rpt->result = M16_GPU_WAIT_PASS;
            return true;
        }
        uint64_t now = cfg->now_ns(cfg->hook_ctx);
        if (m != prev) { prev = m; last = now; if (rpt) rpt->progress_count++; }
        if (now - last >= stall) { if (rpt) rpt->result = M16_GPU_WAIT_STALLED; return false; }
        cfg->relax(cfg->hook_ctx);
    }
}
static bool mut_fixed_ge(volatile uint32_t *w, uint32_t v, m16_gpu_wait_report_t *r, const m16_gpu_wait_cfg_t *c) {
    return mut_loop(1, w, v, r, c, M16_GPU_WAIT_FIXED);
}
static bool mut_seq_unsigned(volatile uint32_t *w, uint32_t v, m16_gpu_wait_report_t *r, const m16_gpu_wait_cfg_t *c) {
    return mut_loop(2, w, v, r, c, M16_GPU_WAIT_SEQUENCE);
}

static int g_fail, g_checks, g_quiet;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_fail++; if (!g_quiet) { printf("  FAIL: " __VA_ARGS__); printf("\n"); } } } while (0)

static void setup(Fake *f, m16_gpu_wait_cfg_t *cfg) {
    memset(f, 0, sizeof(*f));
    f->step_ns = 1000000ull; /* 1 ms per poll */
    memset(cfg, 0, sizeof(*cfg));
    cfg->progress_timeout_ms = 100;
    cfg->total_timeout_ms = 100000;
    cfg->now_ns = fk_now;
    cfg->barrier = fk_barrier;
    cfg->relax = fk_relax;
    cfg->valid = fk_valid;
    cfg->abort = fk_abort;
    cfg->hook_ctx = f;
}
static void sched(Fake *f, uint64_t at, volatile uint32_t *t, uint32_t v) {
    f->at[f->nsched] = at; f->target[f->nsched] = t; f->value[f->nsched] = v; f->nsched++;
}

enum { S_NEVER, S_LATE, S_MARKER2, S_SEQ_WRAP, S_PROGRESS_RESETS, S_BARRIER, S_HARD_DEADLINE,
       S_FIXED_OVERSHOOT, S_INVALID, S_ABORT, S_CLASSES, S_COUNT };
static const char *names[S_COUNT] = {"never_written", "written_late", "marker2_gate", "seq_wrap32",
    "progress_resets_idle", "barrier_before_read", "hard_deadline_with_progress",
    "fixed_overshoot", "invalid_state", "abort", "failure_classes"};

/* Returns failed checks. fx and sq are the fixed and sequence implementations. */
static int scenario(const char *name, WaitFn fx, WaitFn sq, int which) {
    volatile uint32_t w = 0, w2 = 0;
    Fake f;
    m16_gpu_wait_cfg_t cfg;
    m16_gpu_wait_report_t r;
    int before = g_fail;
    bool ok;
    setup(&f, &cfg);

    switch (which) {
    case S_NEVER:
        ok = sq(&w, 5, &r, &cfg);
        CHECK(!ok, "%s: never-written marker must not pass", name);
        CHECK(r.result == M16_GPU_WAIT_STALLED, "%s: result STALLED (got %d)", name, r.result);
        CHECK(r.last_observed == 0 && r.expected == 5 && r.wait_kind == M16_GPU_WAIT_SEQUENCE,
              "%s: report carries kind/expected/last-seen 0", name);
        CHECK(r.elapsed_ns >= 100000000ull && r.progress_count == 0, "%s: elapsed %llu progress %llu", name,
              (unsigned long long)r.elapsed_ns, (unsigned long long)r.progress_count);
        break;
    case S_LATE:
        sched(&f, 60, &w, 7);
        ok = fx(&w, 7, &r, &cfg);
        CHECK(ok && r.result == M16_GPU_WAIT_PASS && r.last_observed == 7, "%s: late fixed marker seen", name);
        CHECK(r.progress_count == 1 && r.last_progress_ns > r.start_ns, "%s: progress recorded", name);
        break;
    case S_MARKER2:
        cfg.marker2 = &w2;
        cfg.marker2_want = 0x46464646u;
        sched(&f, 10, &w, 3);
        ok = fx(&w, 3, &r, &cfg);
        CHECK(!ok, "%s: marker without marker2 must not pass", name);
        CHECK(r.last_observed == 3 && r.last_marker2 == 0, "%s: report shows marker 3, marker2 0", name);
        w = 0; w2 = 0; setup(&f, &cfg);
        cfg.marker2 = &w2;
        cfg.marker2_want = 0x46464646u;
        sched(&f, 10, &w, 3);
        sched(&f, 20, &w2, 0x46464646u);
        ok = fx(&w, 3, &r, &cfg);
        CHECK(ok && r.last_marker2 == 0x46464646u, "%s: both markers pass", name);
        break;
    case S_SEQ_WRAP: {
        static const uint32_t tg[4] = {0xfffffffeu, 0xffffffffu, 0u, 1u};
        for (int i = 0; i < 4; i++) {
            w = tg[i] - 1u; /* one step behind, wrapping through 0 */
            setup(&f, &cfg);
            sched(&f, 10, &w, tg[i]);
            ok = sq(&w, tg[i], &r, &cfg);
            CHECK(ok, "%s: reaches target %08x from %08x", name, tg[i], tg[i] - 1u);
        }
        w = 3; setup(&f, &cfg);
        ok = sq(&w, 0xffffffffu, &r, &cfg);
        CHECK(ok, "%s: 3 is past 0xffffffff", name);
        w = 0; setup(&f, &cfg);
        ok = sq(&w, 0xfffffffeu, &r, &cfg);
        CHECK(ok, "%s: 0 is past 0xfffffffe", name);
        w = 0xfffffff0u; setup(&f, &cfg);
        ok = sq(&w, 0xffffffffu, &r, &cfg);
        CHECK(!ok, "%s: 0xfffffff0 is before 0xffffffff", name);
        w = 0xfffffffeu; setup(&f, &cfg);
        ok = sq(&w, 1, &r, &cfg);
        CHECK(!ok, "%s: 0xfffffffe is before 1", name);
        break;
    }
    case S_PROGRESS_RESETS:
        for (int i = 0; i < 4; i++) sched(&f, 60u * (i + 1), &w, (uint32_t)(i + 1));
        ok = sq(&w, 4, &r, &cfg); /* 240 ms total, 60 ms gaps, 100 ms stall bound */
        CHECK(ok, "%s: steady progress must not stall", name);
        CHECK(r.progress_count == 4, "%s: progress_count %llu", name, (unsigned long long)r.progress_count);
        break;
    case S_BARRIER:
        /* Order check: the value lands when the 3rd barrier runs. A barrier before
         * every read means the read right after it sees the value: polls == 3.
         * A barrier after the read (or none) cannot pass in 3 polls. The extra
         * barrier after PASS (acquire) makes barriers == polls + 1. */
        f.bar_at = 3; f.bar_target = &w; f.bar_value = 1;
        ok = sq(&w, 1, &r, &cfg);
        CHECK(ok && r.polls == 3 && f.barriers == 4, "%s: barrier-before-read order: polls %llu barriers %llu (want 3, 4)", name,
              (unsigned long long)r.polls, (unsigned long long)f.barriers);
        break;
    case S_HARD_DEADLINE:
        cfg.total_timeout_ms = 200;
        for (int i = 0; i < 30; i++) sched(&f, (uint64_t)(20 * (i + 1)), &w, (uint32_t)(i + 1));
        ok = sq(&w, 99, &r, &cfg); /* moves every 20 ms, never arrives */
        CHECK(!ok && r.result == M16_GPU_WAIT_TIMEOUT, "%s: hard deadline TIMEOUT (got %d)", name, r.result);
        CHECK(r.progress_count > 3, "%s: progress was occurring (%llu)", name, (unsigned long long)r.progress_count);
        break;
    case S_FIXED_OVERSHOOT:
        sched(&f, 10, &w, 5); /* jumps past the fixed value 3 */
        ok = fx(&w, 3, &r, &cfg);
        CHECK(!ok, "%s: fixed wait must not accept overshoot 5 for 3", name);
        CHECK(r.result == M16_GPU_WAIT_STALLED && r.last_observed == 5, "%s: stalled, saw 5", name);
        w = 0; setup(&f, &cfg);
        sched(&f, 10, &w, 5);
        ok = sq(&w, 3, &r, &cfg);
        CHECK(ok, "%s: sequence wait accepts 5 for target 3", name);
        break;
    case S_INVALID:
        f.invalid_value = 9;
        sched(&f, 10, &w, 9);
        ok = sq(&w, 20, &r, &cfg);
        CHECK(!ok && r.result == M16_GPU_WAIT_INVALID_STATE && r.last_observed == 9,
              "%s: invalid value reported (got %d)", name, r.result);
        break;
    case S_ABORT:
        f.abort_at = 7;
        ok = sq(&w, 20, &r, &cfg);
        CHECK(!ok && r.result == M16_GPU_WAIT_ABORTED, "%s: abort reported (got %d)", name, r.result);
        break;
    case S_CLASSES: {
        const char *cl;
        ok = fx(&w, 5, &r, &cfg);
        cl = m16_gpu_wait_failure_class(&r, 0);
        CHECK(!ok && !strcmp(cl, "no_progress"), "%s: never moved -> no_progress (got %s)", name, cl);
        setup(&f, &cfg); w = 0;
        cfg.total_timeout_ms = 200;
        for (int i = 0; i < 30; i++) sched(&f, (uint64_t)(20 * (i + 1)), &w, (uint32_t)(i + 1));
        ok = sq(&w, 99, &r, &cfg); /* counter climbing toward 99 */
        cl = m16_gpu_wait_failure_class(&r, 0);
        CHECK(!ok && r.result == M16_GPU_WAIT_TIMEOUT && !strcmp(cl, "late_progress"), "%s: counter moved, missed hard deadline -> late_progress (got %s)", name, cl);
        setup(&f, &cfg); w = 0;
        sched(&f, 10, &w, 5);
        ok = fx(&w, 3, &r, &cfg);
        cl = m16_gpu_wait_failure_class(&r, 0);
        CHECK(!ok && !strcmp(cl, "wrong_marker"), "%s: unexpected value -> wrong_marker (got %s)", name, cl);
        setup(&f, &cfg); w = 0; w2 = 0;
        cfg.marker2 = &w2; cfg.marker2_want = 0x46464646u;
        sched(&f, 10, &w, 3);
        ok = fx(&w, 3, &r, &cfg);
        cl = m16_gpu_wait_failure_class(&r, 0x46464646u);
        CHECK(!ok && !strcmp(cl, "late_progress"), "%s: marker ok, marker2 missing -> late_progress (got %s)", name, cl);
        setup(&f, &cfg); w = 0; w2 = 0;
        cfg.marker2 = &w2; cfg.marker2_want = 0x46464646u;
        sched(&f, 10, &w, 3);
        sched(&f, 12, &w2, 0x11u);
        ok = fx(&w, 3, &r, &cfg);
        cl = m16_gpu_wait_failure_class(&r, 0x46464646u);
        CHECK(!ok && !strcmp(cl, "wrong_marker"), "%s: marker2 wrong value -> wrong_marker (got %s)", name, cl);
        setup(&f, &cfg); w = 0; w2 = 0;
        cfg.total_timeout_ms = 200; cfg.progress_timeout_ms = 200; /* stall bound == hard bound, as in m16_concurrent */
        cfg.marker2 = &w2; cfg.marker2_want = 0x46464646u;
        sched(&f, 150, &w, 3);
        ok = fx(&w, 3, &r, &cfg);
        cl = m16_gpu_wait_failure_class(&r, 0x46464646u);
        CHECK(!ok && r.result == M16_GPU_WAIT_TIMEOUT && !strcmp(cl, "late_progress"), "%s: marker lands at 150 ms of 200, marker2 never -> TIMEOUT late_progress (got %s)", name, cl);
        setup(&f, &cfg); w = 0;
        cfg.total_timeout_ms = 200; cfg.progress_timeout_ms = 200;
        sched(&f, 150, &w, 5);
        ok = fx(&w, 3, &r, &cfg);
        cl = m16_gpu_wait_failure_class(&r, 0);
        CHECK(!ok && r.result == M16_GPU_WAIT_TIMEOUT && !strcmp(cl, "wrong_marker"), "%s: wrong value then hard TIMEOUT still wrong_marker (got %s)", name, cl);
        break;
    }
    }
    return g_fail - before;
}

typedef struct {
    const char *label;
    WaitFn fx, sq;
    unsigned must_kill; /* bitmask of scenarios that must kill this mutant */
} Mutant;

int main(void) {
    for (int s = 0; s < S_COUNT; s++) {
        int bad = scenario(names[s], m16_gpu_wait_fixed, m16_gpu_wait_sequence, s);
        printf("real  %-28s %s\n", names[s], bad ? "FAIL" : "ok");
    }
    int real_fail = g_fail;

    const Mutant muts[] = {
        {"naive_wall_clock", naive_wait, naive_wait, (1u << S_COUNT) - 1u},
        {"fixed_accepts_overshoot", mut_fixed_ge, m16_gpu_wait_sequence, 1u << S_FIXED_OVERSHOOT},
        {"seq_unsigned_compare", m16_gpu_wait_fixed, mut_seq_unsigned, 1u << S_SEQ_WRAP},
    };
    int saved = g_fail, kill_total = 0, kill_need = 0;
    g_quiet = 1;
    for (unsigned m = 0; m < sizeof(muts) / sizeof(muts[0]); m++) {
        unsigned killed = 0;
        for (int s = 0; s < S_COUNT; s++) {
            if (scenario(names[s], muts[m].fx, muts[m].sq, s) > 0) killed |= 1u << s;
        }
        g_fail = saved;
        g_quiet = 0;
        for (int s = 0; s < S_COUNT; s++) {
            if (!(muts[m].must_kill & (1u << s))) continue;
            kill_need++;
            g_checks++;
            if (killed & (1u << s)) kill_total++;
            else { g_fail++; printf("  FAIL: mutant %s survived scenario %s\n", muts[m].label, names[s]); }
        }
        printf("mutant %-26s killed mask %03x (required %03x)\n", muts[m].label, killed, muts[m].must_kill);
        saved = g_fail;
        g_quiet = 1;
    }
    g_quiet = 0;
    printf("m16_gpu_wait: checks=%d real_failures=%d required_kills=%d/%d\n", g_checks, real_fail, kill_total, kill_need);
    printf("%s\n", g_fail ? "M16_GPU_WAIT_HOST FAIL" : "M16_GPU_WAIT_HOST PASS");
    return g_fail ? 1 : 0;
}
