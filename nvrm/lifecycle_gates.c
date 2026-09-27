/*
 * lifecycle_gates.c -- M19R NVRM lifecycle qualification gates.
 *
 * Validates nvrm_free() (VA reuse, live/idempotent/stale detection, CPU
 * unmap) and m16_native_wait_marker_ge() (monotonic serial-number wait)
 * against the real GB10 hardware and kernel driver. New file: does not
 * modify m16_requalify.c / m16_concurrent.c or any existing test.
 */
#include "nvrm.h"
#include "m16_native.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static int g_total = 0, g_passed = 0, g_failed = 0;

static void gate_result(const char *name, int ok) {
    g_total++;
    if (ok) { g_passed++; printf("[PASS] %s\n", name); }
    else    { g_failed++; printf("[FAIL] %s\n", name); }
}

static uint64_t now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000ull + (uint64_t)t.tv_nsec / 1000000ull;
}

/* ---- NVRM_FREE_CHURN ---------------------------------------------------
 * 20,000 alloc/touch/free cycles, sizes cycling 4 KiB..64 MiB, total bytes
 * churned must exceed 256 GiB (2x the machine's 128 GiB unified memory --
 * impossible without real nvrm_free() release), and va_next (the VA
 * high-water mark) must stay bounded, proving reuse rather than a bump leak.
 */
#define CHURN_CYCLES 20000

static int gate_free_churn(Nvrm *rm, uint64_t *out_bytes, uint64_t *out_high_water_delta) {
    static const uint64_t sizes[] = {
        4ull * 1024,               /* 4 KiB   */
        64ull * 1024,              /* 64 KiB  */
        1024ull * 1024,            /* 1 MiB   */
        4ull * 1024 * 1024,        /* 4 MiB   */
        16ull * 1024 * 1024,       /* 16 MiB  */
        32ull * 1024 * 1024,       /* 32 MiB  */
        64ull * 1024 * 1024,       /* 64 MiB  */
        64ull * 1024 * 1024,       /* 64 MiB  */
    };
    const int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    uint64_t va_start = rm->va_next;
    uint64_t va_high = rm->va_next;
    uint64_t total_bytes = 0;

    for (int i = 0; i < CHURN_CYCLES; i++) {
        uint64_t size = sizes[i % nsizes];
        NvrmMem m;
        if (nvrm_alloc(rm, size, &m) != 0) {
            fprintf(stderr, "  churn: nvrm_alloc failed at cycle %d (size 0x%lx): %s\n",
                    i, (unsigned long)size, rm->err);
            return -1;
        }
        if (rm->va_next > va_high) va_high = rm->va_next;

        volatile uint8_t *p = (volatile uint8_t *)m.cpu;
        uint8_t first = (uint8_t)(i & 0xff);
        uint8_t last  = (uint8_t)((i ^ 0xff) & 0xff);
        p[0] = first;
        p[m.size - 1] = last;
        if (p[0] != first || p[m.size - 1] != last) {
            fprintf(stderr, "  churn: readback mismatch at cycle %d\n", i);
            nvrm_free(rm, &m);
            return -1;
        }

        total_bytes += m.size;
        if (nvrm_free(rm, &m) != 0) {
            fprintf(stderr, "  churn: nvrm_free failed at cycle %d: %s\n", i, rm->err);
            return -1;
        }
        if ((i % 2000) == 0) {
            fprintf(stderr, "  churn: cycle %d/%d, %" PRIu64 " bytes so far\n", i, CHURN_CYCLES, total_bytes);
        }
    }

    *out_bytes = total_bytes;
    *out_high_water_delta = va_high - va_start;
    return 0;
}

/* ---- NVRM_FREE_UNMAPS ---------------------------------------------------
 * After nvrm_free(), the CPU range must no longer be mapped: msync() on it
 * must fail with ENOMEM.
 */
static int gate_free_unmaps(Nvrm *rm) {
    NvrmMem m;
    if (nvrm_alloc(rm, 0x4000, &m) != 0) {
        fprintf(stderr, "  unmaps: alloc failed: %s\n", rm->err);
        return -1;
    }
    void *va = m.cpu;
    size_t size = (size_t)m.size;

    if (nvrm_free(rm, &m) != 0) {
        fprintf(stderr, "  unmaps: free failed: %s\n", rm->err);
        return -1;
    }

    errno = 0;
    int rc = msync(va, size, MS_ASYNC);
    if (rc == 0) {
        fprintf(stderr, "  unmaps: msync unexpectedly succeeded on freed range %p\n", va);
        return -1;
    }
    if (errno != ENOMEM) {
        fprintf(stderr, "  unmaps: msync failed with errno %d (%s), expected ENOMEM\n", errno, strerror(errno));
        return -1;
    }
    return 0;
}

/* ---- NVRM_FREE_IDEMPOTENT_AND_STALE ------------------------------------
 * Freeing an already-zeroed NvrmMem (double free) is a silent no-op (0, no
 * syscalls). Freeing a saved stale copy (pre-free snapshot of a struct that
 * has since been freed through another handle) must fail with -1 and must
 * not touch the driver.
 */
static int gate_idempotent_and_stale(Nvrm *rm) {
    NvrmMem m;
    if (nvrm_alloc(rm, 0x2000, &m) != 0) {
        fprintf(stderr, "  idempotent: alloc failed: %s\n", rm->err);
        return -1;
    }
    NvrmMem stale = m; /* stale copy, snapshotted before the real free */

    if (nvrm_free(rm, &m) != 0) {
        fprintf(stderr, "  idempotent: first free failed: %s\n", rm->err);
        return -1;
    }
    if (m.handle != 0 || m.va != 0 || m.size != 0 || m.cpu != NULL) {
        fprintf(stderr, "  idempotent: nvrm_free did not zero the NvrmMem on success\n");
        return -1;
    }

    /* Double free of the now-zeroed struct: must return 0. */
    if (nvrm_free(rm, &m) != 0) {
        fprintf(stderr, "  idempotent: double free of zeroed struct returned nonzero\n");
        return -1;
    }

    /* Free of the stale (already-freed-elsewhere) copy: must return -1. */
    if (nvrm_free(rm, &stale) == 0) {
        fprintf(stderr, "  idempotent: freeing stale copy unexpectedly succeeded\n");
        return -1;
    }
    return 0;
}

/* ---- NVRM_ALLOC_AFTER_FREE_REUSES_VA ------------------------------------
 * Allocate, free, allocate the same size again: must get the same VA back.
 */
static int gate_alloc_after_free_reuses_va(Nvrm *rm) {
    NvrmMem a, b;
    if (nvrm_alloc(rm, 0x8000, &a) != 0) {
        fprintf(stderr, "  reuse: first alloc failed: %s\n", rm->err);
        return -1;
    }
    uint64_t va = a.va;
    if (nvrm_free(rm, &a) != 0) {
        fprintf(stderr, "  reuse: free failed: %s\n", rm->err);
        return -1;
    }
    if (nvrm_alloc(rm, 0x8000, &b) != 0) {
        fprintf(stderr, "  reuse: second alloc failed: %s\n", rm->err);
        return -1;
    }
    int ok = (b.va == va);
    if (!ok) {
        fprintf(stderr, "  reuse: expected va 0x%lx, got 0x%lx\n",
                (unsigned long)va, (unsigned long)b.va);
    }
    nvrm_free(rm, &b);
    return ok ? 0 : -1;
}

static int gate_injected_free_failure(int uvm_failure) {
    Nvrm rm;
    if (nvrm_open(&rm) != 0) return -1;
    NvrmMem m;
    if (nvrm_alloc(&rm, 0x4000, &m) != 0) { nvrm_close(&rm); return -1; }
    uint64_t va = m.va;
    uint32_t free_before = rm.free_count;
    if (uvm_failure) rm.inject_uvm_free_failure = 1;
    else rm.inject_rm_free_failure = 1;
    int refused = nvrm_free(&rm, &m) != 0 && rm.faulted &&
                  m.handle != 0 && rm.live_count == 1 &&
                  rm.free_count == free_before;
    NvrmMem other;
    refused = refused && nvrm_alloc(&rm, 0x4000, &other) != 0;
    int recovered_release = nvrm_free(&rm, &m) == 0 &&
                            rm.free_count > free_before && rm.free_list[rm.free_count - 1].va <= va;
    nvrm_close(&rm);
    return refused && recovered_release ? 0 : -1;
}

static int gate_corrupt_table_refusal(void) {
    Nvrm rm;
    if (nvrm_open(&rm) != 0) return -1;
    NvrmMem m;
    if (nvrm_alloc(&rm, 0x4000, &m) != 0) { nvrm_close(&rm); return -1; }
    void *cpu = rm.live[0].cpu;
    rm.live[0].cpu = NULL;
    int refused = nvrm_free(&rm, &m) != 0 && rm.faulted && rm.live_count == 1 && m.handle;
    rm.live[0].cpu = cpu;
    int released = nvrm_free(&rm, &m) == 0;
    nvrm_close(&rm);
    return refused && released ? 0 : -1;
}

static int gate_allocation_overflow(Nvrm *rm) {
    NvrmMem m = {0};
    uint32_t before = rm->live_count;
    uint64_t va_before = rm->va_next;
    return (nvrm_alloc(rm, UINT64_MAX, &m) != 0 &&
           nvrm_alloc(rm, 0, &m) != 0 &&
           rm->live_count == before && rm->va_next == va_before) ? 0 : -1;
}

/* ---- M16_WAIT_GE_SERIAL --------------------------------------------------
 * Pure CPU unit test, no hardware dependency: synthetic marker + serial
 * (RFC 1982 style) comparison against a target that has wrapped past 0.
 */
static int gate_wait_ge_serial(void) {
    uint32_t marker;

    /* target 0xFFFFFFF0 satisfied by marker 0x00000005 (wrapped past). */
    marker = 0x00000005u;
    if (m16_native_wait_marker_ge(&marker, 0xFFFFFFF0u, 50) != 0) {
        fprintf(stderr, "  wait_ge: wrapped-past case unexpectedly failed/timed out\n");
        return -1;
    }

    /* Not satisfied by 0xFFFFFFE0. Must time out, within timeout_ms + 50ms. */
    marker = 0xFFFFFFE0u;
    uint64_t t0 = now_ms();
    int rc = m16_native_wait_marker_ge(&marker, 0xFFFFFFF0u, 100);
    uint64_t elapsed = now_ms() - t0;
    if (rc == 0) {
        fprintf(stderr, "  wait_ge: not-yet-satisfied case unexpectedly returned success\n");
        return -1;
    }
    if (elapsed > 150) {
        fprintf(stderr, "  wait_ge: timeout took %" PRIu64 "ms, expected <= 150ms\n", elapsed);
        return -1;
    }
    return 0;
}

/* ---- GPU_SUBMIT_AFTER_CHURN ----------------------------------------------
 * After churn, open a channel and do one real submit + wait (same pattern
 * as m16/m16_native.c + m16/m16_requalify.c) to prove the device context
 * is still healthy after 20,000 alloc/free cycles.
 */
static int gate_gpu_submit_after_churn(Nvrm *rm) {
    if (nvrm_channel(rm) != 0) {
        fprintf(stderr, "  submit: nvrm_channel failed: %s\n", rm->err);
        return -1;
    }
    NvrmMem pb, marker_mem;
    if (nvrm_alloc(rm, 0x1000, &pb) != 0) {
        fprintf(stderr, "  submit: pushbuffer alloc failed: %s\n", rm->err);
        return -1;
    }
    if (nvrm_alloc(rm, 0x1000, &marker_mem) != 0) {
        fprintf(stderr, "  submit: marker alloc failed: %s\n", rm->err);
        return -1;
    }
    volatile uint32_t *marker = (volatile uint32_t *)marker_mem.cpu;
    *marker = 0;

    uint32_t methods[32];
    int nwords = m16_native_build_release(methods, marker_mem.va, 0x19c0ffee, rm->compute_class);
    if (nwords <= 0) {
        fprintf(stderr, "  submit: build_release failed\n");
        return -1;
    }
    memcpy(pb.cpu, methods, (size_t)nwords * sizeof(uint32_t));
    __asm__ volatile("dsb sy" ::: "memory");

    if (nvrm_submit(rm, &pb, 0, (uint32_t)nwords) != 0) {
        fprintf(stderr, "  submit: nvrm_submit failed: %s\n", rm->err);
        return -1;
    }
    if (m16_native_wait_marker_ge(marker, 0x19c0ffee, 2000) != 0) {
        fprintf(stderr, "  submit: timeout waiting for marker; actual=0x%08x\n", *marker);
        return -1;
    }
    if (*marker != 0x19c0ffee) {
        fprintf(stderr, "  submit: unexpected final marker value 0x%08x\n", *marker);
        return -1;
    }
    nvrm_retire(rm, rm->put);
    return 0;
}

int main(void) {
    printf("=== M19R NVRM LIFECYCLE GATES ===\n");

    gate_result("M16_WAIT_GE_SERIAL", gate_wait_ge_serial() == 0);

    Nvrm rm;
    if (nvrm_open(&rm) != 0) {
        fprintf(stderr, "FATAL: nvrm_open failed: %s\n", rm.err);
        gate_result("NVRM_FREE_CHURN", 0);
        gate_result("NVRM_FREE_UNMAPS", 0);
        gate_result("NVRM_FREE_IDEMPOTENT_AND_STALE", 0);
        gate_result("NVRM_ALLOC_AFTER_FREE_REUSES_VA", 0);
        gate_result("GPU_SUBMIT_AFTER_CHURN", 0);
        printf("TOTAL: %d PASSED: %d FAILED: %d\n", g_total, g_passed, g_failed);
        return 1;
    }

    uint64_t churn_bytes = 0, high_water_delta = 0;
    int churn_rc = gate_free_churn(&rm, &churn_bytes, &high_water_delta);
    uint64_t required_bytes = 256ull * 1024 * 1024 * 1024;      /* 256 GiB */
    uint64_t max_high_water = 2ull * 1024 * 1024 * 1024;        /* 2 GiB   */
    int churn_ok = (churn_rc == 0) && (churn_bytes > required_bytes) && (high_water_delta < max_high_water);
    printf("  churn: bytes=%" PRIu64 " (%.2f GiB) high_water_delta=%" PRIu64 " (%.2f MiB)\n",
           churn_bytes, (double)churn_bytes / (1024.0 * 1024.0 * 1024.0),
           high_water_delta, (double)high_water_delta / (1024.0 * 1024.0));
    gate_result("NVRM_FREE_CHURN", churn_ok);

    gate_result("NVRM_FREE_UNMAPS", gate_free_unmaps(&rm) == 0);
    gate_result("NVRM_FREE_IDEMPOTENT_AND_STALE", gate_idempotent_and_stale(&rm) == 0);
    gate_result("NVRM_ALLOC_AFTER_FREE_REUSES_VA", gate_alloc_after_free_reuses_va(&rm) == 0);
    gate_result("NVRM_ALLOCATION_OVERFLOW_REFUSED", gate_allocation_overflow(&rm) == 0);
    gate_result("GPU_SUBMIT_AFTER_CHURN", gate_gpu_submit_after_churn(&rm) == 0);

    nvrm_close(&rm);
    gate_result("UVM_FREE_FAILURE_QUARANTINES", gate_injected_free_failure(1) == 0);
    gate_result("RM_FREE_FAILURE_QUARANTINES", gate_injected_free_failure(0) == 0);
    gate_result("CORRUPT_TABLE_REFUSED", gate_corrupt_table_refusal() == 0);
    printf("TOTAL: %d PASSED: %d FAILED: %d\n", g_total, g_passed, g_failed);
    return (g_failed == 0) ? 0 : 1;
}
