#include "m16_gpu_wait.h"

#include <string.h>
#include <time.h>

static uint64_t real_now_ns(void *ctx) {
    (void)ctx;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void real_barrier(void *ctx) {
    (void)ctx;
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/* Compile-flag mutants of the real code, used only by `make mutants-m16-gpu-wait`.
 * Each one must be killed by tests/test_m16_gpu_wait.c. Never defined in a real build. */
#ifdef M16_GPU_WAIT_MUT_NO_BARRIER
#define WAIT_BARRIER(fn, ctx) ((void)(fn), (void)(ctx))
#else
#define WAIT_BARRIER(fn, ctx) (fn)(ctx)
#endif

static void real_relax(void *ctx) {
    (void)ctx;
#if defined(__aarch64__)
    __asm__ volatile("yield");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

const char *m16_gpu_wait_result_name(m16_gpu_wait_result_t r) {
    switch (r) {
    case M16_GPU_WAIT_PASS: return "PASS";
    case M16_GPU_WAIT_TIMEOUT: return "TIMEOUT";
    case M16_GPU_WAIT_STALLED: return "STALLED";
    case M16_GPU_WAIT_INVALID_STATE: return "INVALID_STATE";
    case M16_GPU_WAIT_ABORTED: return "ABORTED";
    default: return "UNKNOWN";
    }
}

const char *m16_gpu_wait_kind_name(m16_gpu_wait_kind_t k) {
    return k == M16_GPU_WAIT_FIXED ? "fixed" : k == M16_GPU_WAIT_SEQUENCE ? "sequence" : "unknown";
}

static bool wait_core(m16_gpu_wait_kind_t kind, volatile uint32_t *word, uint32_t expected,
                      m16_gpu_wait_report_t *rpt, const m16_gpu_wait_cfg_t *cfg) {
    m16_gpu_wait_report_t local;
    m16_gpu_wait_cfg_t zero;
    if (!rpt) rpt = &local;
    memset(rpt, 0, sizeof(*rpt));
    rpt->wait_kind = kind;
    rpt->expected = expected;
    if (!cfg) {
        memset(&zero, 0, sizeof(zero));
        cfg = &zero;
    }

    uint64_t (*now_ns)(void *) = cfg->now_ns ? cfg->now_ns : real_now_ns;
    void (*barrier)(void *) = cfg->barrier ? cfg->barrier : real_barrier;
    void (*relax)(void *) = cfg->relax ? cfg->relax : real_relax;
    uint64_t total_ns = 1000000ull * (cfg->total_timeout_ms ? cfg->total_timeout_ms : M16_GPU_WAIT_DEFAULT_TOTAL_MS);
    uint64_t stall_ns = 1000000ull * (cfg->progress_timeout_ms ? cfg->progress_timeout_ms : M16_GPU_WAIT_DEFAULT_PROGRESS_MS);

    uint64_t start = now_ns(cfg->hook_ctx);
    uint64_t hard_deadline = start + total_ns;
    uint64_t stall_deadline = start + stall_ns;
    rpt->start_ns = start;
    rpt->last_progress_ns = start;
    uint64_t now = start;
    bool first = true;

    /* A null word is a caller bug; report it as an invalid state, not a hang. */
    if (!word) {
        rpt->result = M16_GPU_WAIT_INVALID_STATE;
        return false;
    }

    while (now < hard_deadline) {
        uint32_t m, m2 = 0, snap[2] = {0, 0};
        if (cfg->abort && cfg->abort(cfg->hook_ctx)) {
            rpt->result = M16_GPU_WAIT_ABORTED;
            rpt->elapsed_ns = now - start;
            return false;
        }
        WAIT_BARRIER(barrier, cfg->hook_ctx);
        m = *word;
        if (cfg->marker2) m2 = *cfg->marker2;
        if (cfg->snapshot) cfg->snapshot(cfg->snapshot_ctx, snap);
        rpt->polls++;
        now = now_ns(cfg->hook_ctx);

        bool changed = m != rpt->last_observed || m2 != rpt->last_marker2 ||
                       snap[0] != rpt->snap[0] || snap[1] != rpt->snap[1];
        rpt->last_observed = m;
        rpt->last_marker2 = m2;
        rpt->snap[0] = snap[0];
        rpt->snap[1] = snap[1];
        rpt->elapsed_ns = now - start;

        if (changed && !first) {
            rpt->progress_count++;
            rpt->last_progress_ns = now;
#ifndef M16_GPU_WAIT_MUT_STALL_NEVER_RESETS
            stall_deadline = now + stall_ns;
#endif
#ifdef M16_GPU_WAIT_MUT_PROGRESS_EXTENDS_HARD
            hard_deadline = now + total_ns;
#endif
        }
        bool ok = kind == M16_GPU_WAIT_FIXED ? (m == expected) : ((int32_t)(m - expected) >= 0);
        if (ok && (!cfg->marker2 || m2 == cfg->marker2_want)) {
            WAIT_BARRIER(barrier, cfg->hook_ctx); /* acquire: later CPU loads are ordered after the marker load */
            rpt->result = M16_GPU_WAIT_PASS;
            return true;
        }
        if (cfg->valid && !cfg->valid(cfg->hook_ctx, m)) {
            rpt->result = M16_GPU_WAIT_INVALID_STATE;
            return false;
        }
        first = false;
        if (now >= stall_deadline) {
            rpt->result = M16_GPU_WAIT_STALLED;
            return false;
        }
        relax(cfg->hook_ctx);
        now = now_ns(cfg->hook_ctx);
    }
    rpt->elapsed_ns = now - start;
    rpt->result = M16_GPU_WAIT_TIMEOUT;
    return false;
}

bool m16_gpu_wait_fixed(volatile uint32_t *word, uint32_t expected,
                          m16_gpu_wait_report_t *rpt, const m16_gpu_wait_cfg_t *cfg) {
    return wait_core(M16_GPU_WAIT_FIXED, word, expected, rpt, cfg);
}

bool m16_gpu_wait_sequence(volatile uint32_t *word, uint32_t target,
                             m16_gpu_wait_report_t *rpt, const m16_gpu_wait_cfg_t *cfg) {
    return wait_core(M16_GPU_WAIT_SEQUENCE, word, target, rpt, cfg);
}

const char *m16_gpu_wait_failure_class(const m16_gpu_wait_report_t *r, uint32_t m2_want) {
    if (r->result == M16_GPU_WAIT_INVALID_STATE) return "wrong_marker";
    if (r->wait_kind == M16_GPU_WAIT_FIXED && r->last_observed != 0 && r->last_observed != r->expected)
        return "wrong_marker";
    if (r->last_marker2 != 0 && r->last_marker2 != m2_want) return "wrong_marker";
    if (r->progress_count > 0 || r->last_observed == r->expected ||
        (m2_want != 0 && r->last_marker2 == m2_want))
        return "late_progress";
    return "no_progress";
}
