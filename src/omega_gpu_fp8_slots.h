#ifndef OMEGA_GPU_FP8_SLOTS_H
#define OMEGA_GPU_FP8_SLOTS_H
/*
 * omega_gpu_fp8_slots: versioned, replaceable FP8 expert slots (Expert Ribbon, aienix#4 / omega#533).
 *
 * A pool is S fixed slots of one FP8 projection shape [n][k] (weights + block-128 scales), backed by ONE
 * OmegaGpuFp8Tensor with experts = S, so every existing FP8 launch (grouped GEMV by slot index, the record path through
 * a group table of omega_gpu_fp8_slots_entry addresses) reads a slot exactly as it reads a resident expert. Same kernel,
 * same arithmetic: a slot holding expert E gives bit-identical y to a resident tensor holding E.
 *
 * Memory: GPU-UNCACHED always (nvos.h:1117-1118: GPU-cached system memory "will not be coherent with direct CPU
 * mappings"; omega #457 gpu_l2_stale_test: a CPU rewrite of a cached weight buffer read stale lines, uncached FRESH).
 * The AIEN_OMEGA_FP8_WEIGHT_CACHE / omega_gpu_fp8_set_weight_cache setting does not apply to pools.
 *
 * Protocol (state machine in omega_fp8_slot_table.h, host-tested):
 *   load(slot, key, w, scales) -> version
 *       refused BUSY while the slot is pinned or loading; the slot is hidden (LOADING) before the first byte is written,
 *       so no reader can pin the old or the half-written contents. NaN e4m3 codes refuse (NAN_WEIGHT), NaN/inf scales
 *       refuse (BAD_ARGS); a refused load leaves the slot EMPTY. Weights are copied byte for byte, scales packed as by
 *       omega_gpu_fp8_tensor_upload (f32 * 2^56); then the platform SYNC (CPU writes before device reads); only then is a
 *       new ring-wide version published. Versions never repeat, so (slot, version) names one load (no ABA).
 *       Does NOT take the session lock: a load may run while launches on other slots are in flight.
 *   pin(slot, version) / unpin(slot, version)
 *       a pin is the reader fence: hold it from before the launch is built until its completion marker is seen
 *       (synchronous APIs: until the call returns; a command list: until omega_gpu_cmdlist_submit returns).
 *       pin is refused STALE when the slot was reloaded, evicted or never loaded.
 *   entry(slot, version) -> {w_va, s_va}: only while pinned at that version (for a group table / router directory).
 *   evict(slot): BUSY while pinned.
 *   gemv_grouped_f32(slots, versions, ...): pins every slot, one grouped launch, unpins after the completion marker.
 *
 * Return codes: OMEGA_GPU_FP8_* (omega_gpu_fp8_api.h) plus BUSY / STALE below.
 */
#include "omega_gpu_fp8_api.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_GPU_FP8_SLOTS_BUSY (-6)
#define OMEGA_GPU_FP8_SLOTS_STALE (-7)

typedef struct OmegaGpuFp8Slots OmegaGpuFp8Slots;

typedef struct {
    uint32_t state;   /* 0 EMPTY, 1 LOADING, 2 READY */
    uint32_t pins;
    uint64_t key;     /* caller's expert identity of the READY contents */
    uint64_t version; /* 0 = never published */
} OmegaGpuFp8SlotInfo;

typedef struct {
    uint32_t slots, n, k;
    uint64_t device_bytes;   /* weights + packed scales of the whole pool */
    uint64_t loads, load_failures, busy_refusals, stale_refusals, evictions, pins;
    uint32_t pinned_now, pinned_high;
    uint64_t bytes_loaded;   /* weight + scale bytes copied by successful loads */
    uint64_t load_ns;        /* summed wall time of successful loads (scan + copy + pack + sync) */
} OmegaGpuFp8SlotsStats;

int omega_gpu_fp8_slots_alloc(uint32_t slots, uint32_t n, uint32_t k, OmegaGpuFp8Slots **out);
/* BUSY (nothing freed) while any slot is pinned or loading. */
int omega_gpu_fp8_slots_free(OmegaGpuFp8Slots *p);
/* The backing tensor (experts = slots): pass slot indices as expert ids to the FP8 launches. */
const OmegaGpuFp8Tensor *omega_gpu_fp8_slots_tensor(const OmegaGpuFp8Slots *p);
int omega_gpu_fp8_slots_load(OmegaGpuFp8Slots *p, uint32_t slot, uint64_t key, const uint8_t *w, const uint16_t *scale_inv_bf16,
                             uint64_t *version);
int omega_gpu_fp8_slots_pin(OmegaGpuFp8Slots *p, uint32_t slot, uint64_t version);
int omega_gpu_fp8_slots_unpin(OmegaGpuFp8Slots *p, uint32_t slot, uint64_t version);
int omega_gpu_fp8_slots_evict(OmegaGpuFp8Slots *p, uint32_t slot);
int omega_gpu_fp8_slots_entry(OmegaGpuFp8Slots *p, uint32_t slot, uint64_t version, uint64_t *w_va, uint64_t *s_va);
int omega_gpu_fp8_slots_query(OmegaGpuFp8Slots *p, uint32_t slot, OmegaGpuFp8SlotInfo *out);
void omega_gpu_fp8_slots_stats(OmegaGpuFp8Slots *p, OmegaGpuFp8SlotsStats *out);
/* Pins every (slot, version) (STALE: nothing launched, nothing left pinned), one grouped launch on the pool tensor
 * (omega_gpu_fp8_gemv_grouped_f32 semantics: x_rows 1 or groups), unpins after it returned. */
int omega_gpu_fp8_slots_gemv_grouped_f32(OmegaGpuFp8Slots *p, uint32_t groups, const uint32_t *slots, const uint64_t *versions,
                                         const float *x, uint32_t x_rows, float *y, OmegaGpuFp8Info *info);
const char *omega_gpu_fp8_slots_rc_name(int rc);
/* TEST ONLY: 1 = load ignores pins (negative control: a reader must then see replaced weights). */
void omega_gpu_fp8_slots_test_set_mutant(OmegaGpuFp8Slots *p, int mutant);

#ifdef __cplusplus
}
#endif
#endif
