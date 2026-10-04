#include "omega_gpu_wait.h"

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

static void real_relax(void *ctx) {
    (void)ctx;
#if defined(__aarch64__)
    __asm__ volatile("yield");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

const char *omega_gpu_wait_result_name(omega_gpu_wait_result_t r) {
    switch (r) {
    case OMEGA_GPU_WAIT_PASS: return "PASS";
    case OMEGA_GPU_WAIT_TIMEOUT: return "TIMEOUT";
    case OMEGA_GPU_WAIT_STALLED: return "STALLED";
    case OMEGA_GPU_WAIT_INVALID_STATE: return "INVALID_STATE";
    case OMEGA_GPU_WAIT_ABORTED: return "ABORTED";
    default: return "UNKNOWN";
    }
}

const char *omega_gpu_wait_kind_name(omega_gpu_wait_kind_t k) {
    return k == OMEGA_GPU_WAIT_FIXED ? "fixed" : k == OMEGA_GPU_WAIT_SEQUENCE ? "sequence" : "unknown";
}

static bool wait_core(omega_gpu_wait_kind_t kind, volatile uint32_t *word, uint32_t expected,
                      omega_gpu_wait_report_t *rpt, const omega_gpu_wait_cfg_t *cfg) {
    omega_gpu_wait_report_t local;
    omega_gpu_wait_cfg_t zero;
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
    uint64_t total_ns = 1000000ull * (cfg->total_timeout_ms ? cfg->total_timeout_ms : OMEGA_GPU_WAIT_DEFAULT_TOTAL_MS);
    uint64_t stall_ns = 1000000ull * (cfg->progress_timeout_ms ? cfg->progress_timeout_ms : OMEGA_GPU_WAIT_DEFAULT_PROGRESS_MS);

    uint64_t start = now_ns(cfg->hook_ctx);
    uint64_t hard_deadline = start + total_ns;
    uint64_t stall_deadline = start + stall_ns;
    rpt->start_ns = start;
    rpt->last_progress_ns = start;
    uint64_t now = start;
    bool first = true;

    /* A null word is a caller bug; report it as an invalid state, not a hang. */
    if (!word) {
        rpt->result = OMEGA_GPU_WAIT_INVALID_STATE;
        return false;
    }

    while (now < hard_deadline) {
        uint32_t m, m2 = 0, snap[2] = {0, 0};
        if (cfg->abort && cfg->abort(cfg->hook_ctx)) {
            rpt->result = OMEGA_GPU_WAIT_ABORTED;
            rpt->elapsed_ns = now - start;
            return false;
        }
        barrier(cfg->hook_ctx);
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
            stall_deadline = now + stall_ns;
        }
        bool ok = kind == OMEGA_GPU_WAIT_FIXED ? (m == expected) : ((int32_t)(m - expected) >= 0);
        if (ok && (!cfg->marker2 || m2 == cfg->marker2_want)) {
            barrier(cfg->hook_ctx); /* acquire: later CPU loads are ordered after the marker load */
            rpt->result = OMEGA_GPU_WAIT_PASS;
            return true;
        }
        if (cfg->valid && !cfg->valid(cfg->hook_ctx, m)) {
            rpt->result = OMEGA_GPU_WAIT_INVALID_STATE;
            return false;
        }
        first = false;
        if (now >= stall_deadline) {
            rpt->result = OMEGA_GPU_WAIT_STALLED;
            return false;
        }
        relax(cfg->hook_ctx);
        now = now_ns(cfg->hook_ctx);
    }
    rpt->elapsed_ns = now - start;
    rpt->result = OMEGA_GPU_WAIT_TIMEOUT;
    return false;
}

bool omega_gpu_wait_fixed(volatile uint32_t *word, uint32_t expected,
                          omega_gpu_wait_report_t *rpt, const omega_gpu_wait_cfg_t *cfg) {
    return wait_core(OMEGA_GPU_WAIT_FIXED, word, expected, rpt, cfg);
}

bool omega_gpu_wait_sequence(volatile uint32_t *word, uint32_t target,
                             omega_gpu_wait_report_t *rpt, const omega_gpu_wait_cfg_t *cfg) {
    return wait_core(OMEGA_GPU_WAIT_SEQUENCE, word, target, rpt, cfg);
}
