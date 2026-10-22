/* SPDX-License-Identifier: MIT */
#ifndef OMEGA_GPU_ADMISSION_H
#define OMEGA_GPU_ADMISSION_H
#include "omega_gpu_platform.h"
/* Experimental Linux host API v1. No existing platform/session ABI changes.
 * Errors are negative errno values. All members of one group use identical
 * limits. Directory must exist, be private (0700), and never be replaced while
 * participants exist. Only trusted cooperating processes of one uid belong. */
#define OMEGA_ADMISSION_VERSION 1u
#define OMEGA_ADMISSION_SLOTS 8192u
#define OMEGA_ADMISSION_MAX_BYTES (1ull << 37)
typedef struct OmegaGpuAdmission OmegaGpuAdmission;
typedef struct {
    uint32_t version;
    const char *directory;
    uint64_t budget_bytes, keep_free_bytes;
    /* NULL means fresh /proc/meminfo every reserve. Non-NULL is a FIXTURE,
     * explicitly enabled, regular file, age <= fixture_max_age_sec (1..3600). */
    const char *fixture_meminfo;
    uint32_t fixture_max_age_sec;
} OmegaGpuAdmissionConfig;
typedef enum {
    OMEGA_ADMISSION_WEIGHT = 0, OMEGA_ADMISSION_INTERMEDIATE,
    OMEGA_ADMISSION_KV, OMEGA_ADMISSION_MARKER,
    OMEGA_ADMISSION_SHARED_WEIGHT, /* refused: no native import ownership proof */
    OMEGA_ADMISSION_USAGE_COUNT
} OmegaGpuAdmissionUsage;
typedef struct { uint64_t generation; uint32_t slot; } OmegaGpuReservation;
typedef struct {
    uint64_t reserved_bytes, acquired_bytes, committed_bytes;
    uint64_t by_usage[OMEGA_ADMISSION_USAGE_COUNT];
    uint64_t mem_available_bytes; /* fresh sample, includes external pressure */
    uint64_t process_rss_bytes; /* measured /proc/self/statm, includes explicit buffers */
    uint64_t overhead_bytes; /* unknown: cannot subtract GPU/shared bytes from RSS */
    uint32_t overhead_known; /* always 0 until disjoint #14 accounting is available */
} OmegaGpuAdmissionStats;
int omega_gpu_admission_open(const OmegaGpuAdmissionConfig *, OmegaGpuAdmission **);
/* Refuses close while this owner has any charge. Does not silently release buffers. */
int omega_gpu_admission_close(OmegaGpuAdmission *);
int omega_gpu_admission_reserve(OmegaGpuAdmission *, uint64_t, OmegaGpuAdmissionUsage, OmegaGpuReservation *);
int omega_gpu_admission_acquire(OmegaGpuAdmission *, OmegaGpuReservation);
int omega_gpu_admission_commit(OmegaGpuAdmission *, OmegaGpuReservation);
/* cancel: RESERVED only. abort: ACQUIRED only, caller proves allocation failed.
 * release: COMMITTED only, caller proves device work retired and free succeeded. */
int omega_gpu_admission_cancel(OmegaGpuAdmission *, OmegaGpuReservation);
int omega_gpu_admission_abort(OmegaGpuAdmission *, OmegaGpuReservation);
int omega_gpu_admission_release(OmegaGpuAdmission *, OmegaGpuReservation);
int omega_gpu_admission_stats(OmegaGpuAdmission *, OmegaGpuAdmissionStats *);
int omega_gpu_admission_seal(OmegaGpuAdmission *, int);
/* Explicit integration adapter. No env/default changes. Call instead of dma_alloc
 * and dma_free on an opted-in path. It calls the real platform only after admission.
 * Free requires the same platform/context and generation; underlying lifetime rules
 * (retire before free, timeout parking, device teardown) remain caller obligations.
 * Context operations serialized internally; do not fork with operations in flight. */
int omega_gpu_admission_dma_alloc(OmegaGpuAdmission *, const OmegaGpuPlatform *, const OmegaGpuPlatformCaps *,
    OmegaGpuDeviceHandle, const OmegaGpuBufferDesc *, OmegaGpuAdmissionUsage,
    OmegaGpuBufferHandle *, OmegaCpuView *, OmegaGpuVa *, OmegaGpuBusAddr *);
int omega_gpu_admission_dma_free(OmegaGpuAdmission *, const OmegaGpuPlatform *, OmegaGpuBufferHandle);
#endif
