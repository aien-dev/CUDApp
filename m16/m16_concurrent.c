/*
 * m16_concurrent.c -- M16 concurrent multi-channel qualification test.
 *
 * Sequence:
 *   1. Initialize two independent M16NativeContext instances (ctxA, ctxB).
 *   2. Allocate separate channels, GPFIFO rings, USERD, and tokens.
 *   3. Allocate independent GPU-uncached marker pages (nvrm_alloc_gpu_uncached,
 *      nvrm/nvrm.c:473): markerA, markerB. Each page also holds marker2 at +0x10.
 *   4. Submit independent work:
 *        Channel A -> 0x16000001, L2 flush, then 0x46464646 at +0x10
 *        Channel B -> 0x16000002, L2 flush, then 0x46464646 at +0x10
 *   5. Verify independent causality:
 *        Both withheld -> markerA=0, markerB=0
 *        Ring A -> markerA=0x16000001, markerB=0
 *        Ring B -> markerB=0x16000002
 *   6. Prove non-collision and dynamic resource independence.
 *
 * Waits use the canonical primitive m16/m16_gpu_wait.c: monotonic clock, hard
 * 2000 ms deadline that is never extended, 2000 ms stall deadline (equal, so the
 * old 2 s pass window is unchanged) that restarts only on progress, barrier before every read, PASS needs marker AND marker2.
 * Every failure prints one line tagged
 *   class=<submission_fail|no_progress|late_progress|wrong_marker|output_parity>
 * followed by the full wait report.
 *
 * Exit codes (unchanged unless marked NEW):
 *   1 init failure or enqueue failure (class submission_fail)
 *   2 premature execution on withheld doorbells (wrong_marker)
 *   3 markerA wait failed (no_progress / late_progress / wrong_marker)
 *   4 cross-talk: B executed on A's doorbell (wrong_marker)
 *   5 markerB wait failed (no_progress / late_progress / wrong_marker)
 *   6 root handle collision
 *   7 NEW: output_parity. Either a marker re-read right after its PASS disagrees
 *     with the value just accepted (nearly a tautology, it only catches a torn
 *     or unstable word), or, new tightening, the final check that both pages still
 *     hold their payload and 0x46464646 after B completed fails.
 *     This test has no output buffer; the markers are the output.
 *
 * The doorbell is a plain store (nvrm_ring, nvrm/nvrm.c:742, returns void), so a
 * doorbell write that fails cannot be detected; one that never reaches the chip
 * shows up as no_progress.
 */
#include "m16_native.h"
#include "m16_gpu_wait.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PAYLOAD_A 0x16000001u
#define PAYLOAD_B 0x16000002u
#define MARKER2_PAYLOAD 0x46464646u
#define MARKER2_OFF 0x10u
#define WAIT_TOTAL_MS 2000u  /* hard deadline: same 2 s window as the old wait */
#define WAIT_STALL_MS 2000u  /* >= hard bound so the pass window is not narrower than the old wait */

static void fence(void) { __asm__ volatile("dsb sy" ::: "memory"); }

/* Marker stream: bind class, WFI release of payload at marker_va, L2_FLUSH_DIRTY
 * mem-op (NVC96F_MEM_OP_A..D = 0x28..0x34, D bits 31:27 = 0x10), then a second WFI
 * release of 0x46464646 at marker_va + 0x10. Same block as omega's
 * src/omega_numeric_gb10.c (the C3 hardening block, "L2_FLUSH_DIRTY" comment
 * above the second 0x005c packet); nvrm_mthd is nvrm/nvrm.h:112. */
static int build_release_flushed(uint32_t *w, uint64_t marker_va, uint32_t payload, uint32_t cls) {
    int n = m16_native_build_release(w, marker_va, payload, cls);
    if (n < 0) return -1;
    w[n++] = nvrm_mthd(0, 0x0028, 4);
    w[n++] = 0;
    w[n++] = 0;
    w[n++] = 0;
    w[n++] = (0x10u << 27);
    uint64_t m2 = marker_va + MARKER2_OFF;
    w[n++] = nvrm_mthd(0, 0x005c, 5);
    w[n++] = (uint32_t)m2;
    w[n++] = (uint32_t)(m2 >> 32);
    w[n++] = MARKER2_PAYLOAD;
    w[n++] = 0;
    w[n++] = 0x1 | (1u << 20);
    return n;
}

static void print_report(const char *cls, const char *stage, int code, const m16_gpu_wait_report_t *r) {
    fprintf(stderr,
            "FAIL class=%s stage=%s exit=%d kind=%s result=%s expected=0x%08x last_marker=0x%08x "
            "last_marker2=0x%08x elapsed_ms=%llu last_progress_ms=%llu "
            "progress_count=%llu polls=%llu\n",
            cls, stage, code, m16_gpu_wait_kind_name(r->wait_kind), m16_gpu_wait_result_name(r->result),
            r->expected, r->last_observed, r->last_marker2, (unsigned long long)(r->elapsed_ns / 1000000ull),
            (unsigned long long)((r->last_progress_ns - r->start_ns) / 1000000ull),
            (unsigned long long)r->progress_count, (unsigned long long)r->polls);
}

static void print_simple(const char *cls, const char *stage, int code, const char *what) {
    fprintf(stderr, "FAIL class=%s stage=%s exit=%d %s\n", cls, stage, code, what);
}

/* Wait marker==want and marker2==MARKER2_PAYLOAD; on failure print the tagged
 * line and return the class's exit code, on PASS re-read both (parity). */
static int wait_marker(volatile uint32_t *marker, uint32_t want, const char *stage, int fail_exit) {
    volatile uint32_t *marker2 = (volatile uint32_t *)((volatile uint8_t *)marker + MARKER2_OFF);
    m16_gpu_wait_cfg_t cfg;
    m16_gpu_wait_report_t rpt;
    memset(&cfg, 0, sizeof cfg);
    cfg.total_timeout_ms = WAIT_TOTAL_MS;
    cfg.progress_timeout_ms = WAIT_STALL_MS;
    cfg.marker2 = marker2;
    cfg.marker2_want = MARKER2_PAYLOAD;
    if (!m16_gpu_wait_fixed(marker, want, &rpt, &cfg)) {
        print_report(m16_gpu_wait_failure_class(&rpt, MARKER2_PAYLOAD), stage, fail_exit, &rpt);
        return fail_exit;
    }
    fence();
    uint32_t m = *marker, m2 = *marker2;
    if (m != want || m2 != MARKER2_PAYLOAD) {
        char what[160];
        snprintf(what, sizeof what, "reread_marker=0x%08x want=0x%08x reread_marker2=0x%08x want2=0x%08x",
                 m, want, m2, MARKER2_PAYLOAD);
        print_simple("output_parity", stage, 7, what);
        return 7;
    }
    return 0;
}

int main(void) {
    printf("=== M16 CONCURRENT MULTI-CHANNEL QUALIFICATION ===\n");

    M16NativeContext ctxA, ctxB;

    printf("\n--- INITIALIZING CONTEXT A ---\n");
    if (m16_native_open(&ctxA) != 0 || m16_native_create_channel(&ctxA) != 0) {
        fprintf(stderr, "FATAL: Failed to initialize Context A: %s\n", ctxA.rm.err);
        return 1;
    }
    printf("[OBSERVED] Context A: token=0x%08x gpfifo=0x%08x fifo_va=0x%lx\n",
           ctxA.rm.token, ctxA.rm.gpfifo, (unsigned long)ctxA.rm.fifo.va);

    printf("\n--- INITIALIZING CONTEXT B ---\n");
    if (m16_native_open(&ctxB) != 0 || m16_native_create_channel(&ctxB) != 0) {
        fprintf(stderr, "FATAL: Failed to initialize Context B: %s\n", ctxB.rm.err);
        m16_native_close(&ctxA);
        return 1;
    }
    printf("[OBSERVED] Context B: token=0x%08x gpfifo=0x%08x fifo_va=0x%lx\n",
           ctxB.rm.token, ctxB.rm.gpfifo, (unsigned long)ctxB.rm.fifo.va);

    /* Allocate markers: GPU-uncached, so a GPU store reaches memory instead of
     * sitting in GPU L2 (was nvrm_alloc via m16_native_alloc_memory: cacheable). */
    NvrmMem memA, memB;
    if (nvrm_alloc_gpu_uncached(&ctxA.rm, 0x1000, &memA) != 0 ||
        nvrm_alloc_gpu_uncached(&ctxB.rm, 0x1000, &memB) != 0) {
        fprintf(stderr, "FATAL: Failed to allocate marker memories\n");
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 1;
    }
    uint64_t vaA = memA.va, vaB = memB.va;
    volatile uint32_t *markerA = (volatile uint32_t *)memA.cpu;
    volatile uint32_t *markerB = (volatile uint32_t *)memB.cpu;
    volatile uint32_t *marker2A = (volatile uint32_t *)((volatile uint8_t *)memA.cpu + MARKER2_OFF);
    volatile uint32_t *marker2B = (volatile uint32_t *)((volatile uint8_t *)memB.cpu + MARKER2_OFF);
    *markerA = 0;
    *markerB = 0;
    *marker2A = 0;
    *marker2B = 0;
    fence();

    printf("[OBSERVED] Marker A: VA=0x%lx CPU=%p (initial=0, GPU-uncached)\n", (unsigned long)vaA, memA.cpu);
    printf("[OBSERVED] Marker B: VA=0x%lx CPU=%p (initial=0, GPU-uncached)\n", (unsigned long)vaB, memB.cpu);

    /* Build methods */
    uint32_t mthdA[32], mthdB[32];
    int nA = build_release_flushed(mthdA, vaA, PAYLOAD_A, ctxA.rm.compute_class);
    int nB = build_release_flushed(mthdB, vaB, PAYLOAD_B, ctxB.rm.compute_class);

    /* Enqueue to both without ringing */
    printf("\n--- ENQUEUING WORK WITHOUT DOORBELLS ---\n");
    if (nA < 0 || nB < 0 ||
        m16_native_enqueue_methods(&ctxA, mthdA, (size_t)nA) != 0 ||
        m16_native_enqueue_methods(&ctxB, mthdB, (size_t)nB) != 0) {
        print_simple("submission_fail", "enqueue", 1, "enqueue of method stream failed");
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 1;
    }
    usleep(50000);
    fence();
    printf("[OBSERVED] Withheld doorbells: markerA=0x%08x markerB=0x%08x\n", *markerA, *markerB);
    if (*markerA != 0 || *markerB != 0 || *marker2A != 0 || *marker2B != 0) {
        char what[160];
        snprintf(what, sizeof what, "premature markerA=0x%08x markerB=0x%08x marker2A=0x%08x marker2B=0x%08x",
                 *markerA, *markerB, *marker2A, *marker2B);
        print_simple("wrong_marker", "withheld", 2, what);
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 2;
    }

    /* Ring Doorbell A only */
    printf("\n--- RINGING DOORBELL A ONLY ---\n");
    m16_native_ring_doorbell(&ctxA);
    int rc = wait_marker(markerA, PAYLOAD_A, "A", 3);
    if (rc != 0) {
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return rc;
    }
    printf("[OBSERVED] After Doorbell A: markerA=0x%08x markerB=0x%08x (expected A=0x16000001, B=0x00000000)\n",
           *markerA, *markerB);
    fence();
    if (*markerB != 0 || *marker2B != 0) {
        char what[120];
        snprintf(what, sizeof what, "cross-talk markerB=0x%08x marker2B=0x%08x with only doorbell A rung",
                 *markerB, *marker2B);
        print_simple("wrong_marker", "crosstalk", 4, what);
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 4;
    }

    /* Ring Doorbell B */
    printf("\n--- RINGING DOORBELL B ---\n");
    m16_native_ring_doorbell(&ctxB);
    rc = wait_marker(markerB, PAYLOAD_B, "B", 5);
    if (rc != 0) {
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return rc;
    }
    printf("[OBSERVED] After Doorbell B: markerA=0x%08x markerB=0x%08x (expected A=0x16000001, B=0x16000002)\n",
           *markerA, *markerB);
    /* output_parity, final: B's work must not have disturbed A's page. The old log
     * printed this as "expected" but never enforced it (NEW tightening). */
    fence();
    if (*markerA != PAYLOAD_A || *marker2A != MARKER2_PAYLOAD || *markerB != PAYLOAD_B || *marker2B != MARKER2_PAYLOAD) {
        char what[200];
        snprintf(what, sizeof what, "final markerA=0x%08x marker2A=0x%08x markerB=0x%08x marker2B=0x%08x",
                 *markerA, *marker2A, *markerB, *marker2B);
        print_simple("output_parity", "final", 7, what);
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 7;
    }

    /* Verify distinct resource handles */
    printf("\n--- VERIFYING RESOURCE INDEPENDENCE ---\n");
    printf("[OBSERVED] Handle check: Root A=0x%x B=0x%x, Channel A=0x%x B=0x%x, Token A=0x%x B=0x%x\n",
           ctxA.rm.root, ctxB.rm.root, ctxA.rm.gpfifo, ctxB.rm.gpfifo, ctxA.rm.token, ctxB.rm.token);

    if (ctxA.rm.root == ctxB.rm.root) {
        fprintf(stderr, "FATAL: Root handles collided between sessions!\n");
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 6;
    }

    printf("CONCURRENT_CHANNEL_PASS: 2 independent channels executed without collision.\n");

    nvrm_retire(&ctxA.rm, ctxA.rm.put);
    nvrm_retire(&ctxB.rm, ctxB.rm.put);
    m16_native_close(&ctxA);
    m16_native_close(&ctxB);
    return 0;
}
