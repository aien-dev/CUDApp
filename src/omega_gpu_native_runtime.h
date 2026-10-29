/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_GPU_NATIVE_RUNTIME_H
#define OMEGA_GPU_NATIVE_RUNTIME_H
/* Private caller glue. Policy and ownership decisions are generated from .om.
 * Device allocations remain in the existing FP8/GEMM/TC cache slots. Callers
 * hold the existing session lock; no artifact/qualification authority lives here. */
#include "cudapp_native.h"
#include "omega_gpu_session.h"
#include "omega_gpu_code_alloc.h"
#include "omega_native_artifact_snapshot.h"
#include <string.h>
#include <stdint.h>

/* Advances only after successful device teardown, including normal close. */
uint64_t omega_gpu_session_epoch(void);
typedef struct {
    uint64_t pool[528];
} OmegaGpuNativeCache;
typedef struct {
    OmegaGpuNativeCache *cache;
    uint64_t slot, revision, epoch;
} OmegaGpuNativeLease;

static inline int64_t omega_native_cache_call(OmegaGpuNativeCache *c, uint64_t op,
                                               const uint64_t req[16], const uint64_t tok[16]) {
    uint64_t bits = 0;
    uint32_t rc = omega_native_cache(&bits, c->pool, 528, req, 16, tok, 16, op);
    int64_t result;
    memcpy(&result, &bits, sizeof result);
    return rc == OMEGA_OK ? result : -1;
}
static inline void omega_native_context(uint64_t req[16]) {
    memset(req, 0, sizeof(uint64_t) * 16);
    req[0] = omega_gpu_session_epoch();
    req[1] = 121;
}
/* A changed epoch means the old device handles are already dead. The physical
 * cache owner must then discard stale host descriptors without dma_free. */
static inline int omega_native_cache_refresh(OmegaGpuNativeCache *c, unsigned slots) {
    uint64_t req[16], tok[16] = {0};
    omega_native_context(req);
    req[2] = UINT64_MAX; req[7] = slots;
    return (int)omega_native_cache_call(c, 17, req, tok);
}
static inline int omega_native_shape_key(const void *shape, size_t bytes, uint8_t key[32]) {
    uint8_t data[4096] = {0};
    uint64_t result = 0;
    if (!shape || bytes > sizeof data) return -1;
    memcpy(data, shape, bytes);
    return omega_native_key(&result, data, sizeof data, bytes, key, 32) == OMEGA_OK && result == 0 ? 0 : -1;
}
static inline int omega_native_cache_reserve(OmegaGpuNativeCache *c, const uint8_t key[32],
                                             size_t charged, uint64_t req[16]) {
    uint64_t tok[16] = {0};
    omega_native_context(req);
    req[2] = charged;
    memcpy(req + 3, key, 32);
    int64_t slot = omega_native_cache_call(c, 18, req, tok);
    req[7] = c->pool[12]; /* accepted authoritative reservation owner */
    return slot > 0 ? (int)(slot - 1) : -1;
}
static inline int omega_native_cache_finish(OmegaGpuNativeCache *c, unsigned slot,
                                            const uint64_t req[16], int commit) {
    uint64_t tok[16] = {0};
    tok[0] = slot; tok[2] = req[7];
    return (int)omega_native_cache_call(c, commit ? 2 : 3, req, tok);
}
static inline int omega_native_cache_victim(OmegaGpuNativeCache *c, unsigned slot, int erase) {
    uint64_t req[16], tok[16] = {0};
    omega_native_context(req); tok[0] = slot;
    return (int)omega_native_cache_call(c, erase ? 14 : 13, req, tok);
}
static inline OmegaGpuNativeLease omega_native_cache_lease(OmegaGpuNativeCache *c, unsigned slot) {
    uint64_t revision = 0, epoch = omega_gpu_session_epoch();
    uint32_t rc = omega_native_slotrev(&revision, c->pool, 528, slot, epoch);
    OmegaGpuNativeLease lease = { c, slot, rc == OMEGA_OK ? revision : 0, epoch };
    return lease;
}
static inline int64_t omega_native_live_call(const OmegaGpuNativeLease *lease, unsigned op,
                                            const uint64_t req[16], int on) {
    uint64_t tok[16] = {0}, bits = 0;
    int64_t result;
    if (!lease || !lease->cache) return -1;
    tok[0] = lease->slot; tok[1] = lease->revision;
    tok[3] = on ? 1 : 0; tok[4] = lease->epoch;
    uint32_t rc = omega_native_lease(&bits, lease->cache->pool, 528, req, 16, tok, 16, op);
    memcpy(&result, &bits, sizeof result);
    return rc == OMEGA_OK ? result : -1;
}
static inline int omega_native_lease_op(const OmegaGpuNativeLease *lease, unsigned op, int on) {
    uint64_t req[16];
    omega_native_context(req);
    return (int)omega_native_live_call(lease, op, req, on);
}
typedef struct {
    uint8_t canonical[4096], code[16384], launch[4096];
    uint64_t sizes[4];
    uint8_t realization[32], manifest[32];
    const OmegaNativeArtifactSnapshot *admission;
} OmegaGpuNativeArtifact;
/* The manifest ID remains the portable artifact authority. This object is an
 * immutable payload snapshot, not an admission or qualification assertion. */
static inline int omega_native_payload(const OmegaGpuNativeArtifact *a, uint8_t key[32]) {
    uint64_t result = 0;
    return a && omega_native_validate(&result, a->canonical, 4096, a->sizes, 4,
        a->code, 16384, a->launch, 4096, a->realization, 32, key, 32) == OMEGA_OK && result == 0 ? 0 : -1;
}
/* Narrow custody projection into generated predicates. Expected manifest and
 * launch remain selected immutable inputs; none is obtained from the snapshot.
 * Fixed-size padding is required by the bounded native ABI, not a second parser. */
static inline int omega_native_admitted(const OmegaGpuNativeArtifact *a, uint8_t key[32]) {
    const OmegaNativeArtifactSnapshot *s = a ? a->admission : NULL;
    uint8_t cubin[32768] = {0}, observed[2048] = {0}, text[16384] = {0};
    uint64_t sizes[8] = {0}, payload_sizes[4], result = 0;
    size_t name;
    if (!s || omega_native_payload(a, key) != 0) return -1;
    name = strnlen(s->kernel_function, sizeof s->kernel_function);
    if (s->sizes[OMEGA_SNAPSHOT_CUBIN] > sizeof cubin ||
        s->sizes[OMEGA_SNAPSHOT_TEXT] > sizeof text ||
        s->sizes[OMEGA_SNAPSHOT_QMD0] != 384 || s->sizes[OMEGA_SNAPSHOT_QMD1] != 384 ||
        s->sizes[OMEGA_SNAPSHOT_CBANK] > 128 || name >= sizeof s->kernel_function)
        return -1;
    memcpy(cubin, s->bytes[OMEGA_SNAPSHOT_CUBIN], s->sizes[OMEGA_SNAPSHOT_CUBIN]);
    memcpy(text, s->bytes[OMEGA_SNAPSHOT_TEXT], s->sizes[OMEGA_SNAPSHOT_TEXT]);
    memcpy(observed, s->bytes[OMEGA_SNAPSHOT_QMD0], 384);
    memcpy(observed + 384, s->bytes[OMEGA_SNAPSHOT_QMD1], 384);
    memcpy(observed + 768, s->bytes[OMEGA_SNAPSHOT_CBANK], s->sizes[OMEGA_SNAPSHOT_CBANK]);
    memcpy(observed + 896, s->kernel_function, name);
    memcpy(observed + 960, a->manifest, 32);
    memcpy(observed + 1024, s->id, 64);
    sizes[0] = s->sizes[OMEGA_SNAPSHOT_CUBIN];
    sizes[1] = s->sizes[OMEGA_SNAPSHOT_CBANK]; sizes[2] = name;
    if (omega_native_templates(&result, a->launch, 4096, cubin, 32768,
        observed, 2048, sizes, 8) != OMEGA_OK || result != 0) return -1;
    memcpy(payload_sizes, a->sizes, sizeof payload_sizes);
    payload_sizes[1] = s->sizes[OMEGA_SNAPSHOT_TEXT];
    return omega_native_validate(&result, a->canonical, 4096, payload_sizes, 4,
        text, 16384, a->launch, 4096, a->realization, 32, key, 32) == OMEGA_OK &&
        result == 0 ? 0 : -1;
}
static inline int omega_native_launch(const OmegaGpuNativeArtifact *a, const uint64_t call[12], uint64_t launch[32]) {
    uint64_t result = 0;
    return omega_native_binding(&result, a->canonical, 4096, a->launch, 4096,
        call, 12, launch, 32) == OMEGA_OK && result == 0 ? 0 : -1;
}
static inline int omega_native_bound(const OmegaGpuNativeArtifact *a, const OmegaGpuNativeLease *lease) {
    uint8_t key[32]; uint64_t req[16];
    if (!lease || !lease->cache || omega_native_payload(a, key) != 0) return -1;
    omega_native_context(req);
    req[2] = omega_gpu_code_alloc_bytes((size_t)a->sizes[1]);
    memcpy(req + 3, key, 32);
    return (int)omega_native_live_call(lease, 15, req, 0);
}
struct OmegaGpuFp8Tensor;
typedef struct {
    uint32_t experts, n, k;
    uint64_t allocation_epoch, owner_generation;
    OmegaGpuMem weights, scales;
} OmegaGpuNativeTensorObservation;
/* The existing session lock must be held. Copies owner descriptors only;
 * generation, membership and alias predicates remain generated native code. */
int omega_gpu_fp8_native_observe(const struct OmegaGpuFp8Tensor *,
                               OmegaGpuNativeTensorObservation *);
int omega_gpu_fp8_prepare_native(const OmegaGpuNativeArtifact *a, OmegaGpuNativeLease *lease);
int omega_gpu_fp8gemm_prepare_native(const OmegaGpuNativeArtifact *a, OmegaGpuNativeLease *lease);
struct OmegaGpuCmdList;
/* Appends a launch and atomically retains its exact resident code lease. */
int omega_gpu_cmdlist_add_native(struct OmegaGpuCmdList *cl, const OmegaGpuLaunch *launch,
                                const OmegaGpuNativeLease *lease);
OmegaGpuNativeCache *omega_gpu_fp8_native_cache(void);
OmegaGpuNativeCache *omega_gpu_fp8gemm_native_cache(void);
/* Existing session lock held; only a current generated-validated retained lease
 * can project the exact cache slot's typed code allocation. */
int omega_gpu_fp8gemm_native_code_observe(const OmegaGpuNativeLease *, OmegaGpuMem *);
int omega_gpu_fp8_record_native(struct OmegaGpuCmdList *, const struct OmegaGpuFp8Tensor *,
    const OmegaGpuNativeArtifact *, const OmegaGpuNativeLease *,
    uint32_t groups, uint64_t table, uint64_t xp, uint32_t xrows, uint64_t y);
int omega_gpu_fp8gemm_record_native(struct OmegaGpuCmdList *, const struct OmegaGpuFp8Tensor *,
    const OmegaGpuNativeArtifact *, const OmegaGpuNativeLease *,
    uint32_t rows, uint32_t segs, uint64_t table, uint64_t xp, uint64_t y,
    uint64_t xmap, uint64_t ymap);
#endif
