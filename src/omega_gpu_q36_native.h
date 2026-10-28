/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_GPU_Q36_NATIVE_H
#define OMEGA_GPU_Q36_NATIVE_H
#include "omega_gpu_native_runtime.h"
#include "omega_gpu_cmdlist_native.h"
/* Borrowed router-owner facts. The caller holds the session lock and consumes
 * them before recording can replace either cached kernel. Exact encoded bytes
 * exclude the code allocation's prefetch tail. This conveys no admission. */
struct OmegaGpuRouter;
typedef struct {
    OmegaGpuMem code, gate, norm, directory, launch_buffer;
    uint64_t code_bytes;
    uint64_t owner_generation, code_generation;
    const uint64_t *gu_directory, *dn_directory;
    uint32_t entries;
} OmegaGpuQ36RouterObservation;
int omega_gpu_q36_router_observe(const struct OmegaGpuRouter *, uint32_t layer,
    uint64_t recorded_code_va, OmegaGpuQ36RouterObservation *);
/* Same mode-final callback in production and the existing typed HOST_ONLY
 * platform oracle. Borrowed exact owner descriptors only, not an admission
 * constructor. ordinal=UINT64_MAX matches every occurrence of that owner;
 * GU/DN use their independently retained successful-add ordinal instead. */
typedef struct {
    OmegaGpuMem code;
    uint64_t code_bytes, role, ordinal;
} OmegaGpuQ36MoeCode;
typedef struct {
    uint64_t context[16];
    OmegaGpuQ36MoeCode code[272]; /* six-role variants plus selected pair windows */
    uint32_t code_count;
    uint8_t nested[2048]; /* only actual groups10..21; cleared before projection */
} OmegaGpuQ36MoeProjection;
int omega_gpu_q36_moe_project(const OmegaGpuCmdList *,
    const OmegaGpuCmdListConsumed *, void *projection);
/* Independently retained source-owner identities, not an expected-wire view.
 * Admission owns this immutable snapshot. No production constructor exists.
 * Live code, launches, directories and allocations replace their groups. */
typedef struct {
    uint8_t hashes[2048];
} OmegaGpuQ36NativeProvenance;
/* Private graph-owned, immutable accepted-record snapshot. No public setter,
 * q-flag constructor, manifest-string activation, or new qualification store.
 * Production construction stays absent until the independent evaluator/judge
 * can supply an exact accepted receipt through the existing admission chain.
 * A NULL state means no qualified native candidate, not a failed incumbent. */
typedef struct {
    uint64_t query[64], rows[1024];
    uint8_t wanted[4096], records[65536];
    OmegaGpuNativeArtifact artifact[16];
    OmegaGpuNativeLease prepared[16];
    const uint8_t *expected_closure; /* same retained full MC wire for paired GU/DN */
    const OmegaGpuQ36NativeProvenance *provenance;
} OmegaGpuQ36NativeState;
/* Private lookup keys identify actual graph operations, never qualification.
 * Each operation/window owns its original immutable selector query and bytes.
 * Paired MoE roles retain the same independently frozen expected closure. */
typedef struct {
    uint64_t keys[128], count;
    OmegaGpuQ36NativeState *context; /* points inside the graph-owned immutable arena */
} OmegaGpuQ36NativeGraph;
static inline int omega_q36_native_context(OmegaGpuQ36NativeGraph *g,
    const uint64_t actual[4], OmegaGpuQ36NativeState **context) {
    uint64_t bits = 0;
    int64_t result;
    *context = NULL;
    if (!g) return 1;
    if (omega_native_context_lookup(&bits, g->keys, 128, g->count, actual, 4) != OMEGA_OK)
        return -1;
    memcpy(&result, &bits, sizeof result);
    if (result == 0) return 1;
    if (result < 0 || !g->context) return -1;
    /* The generated unique-match resolver bounds result to 1..count<=32. */
    *context = &g->context[(uint64_t)result - 1];
    return 0;
}
static inline int omega_q36_native_begin_selected(OmegaGpuQ36NativeState *s, OmegaGpuNativeCache *cache,
    const OmegaGpuNativeArtifact **artifact, OmegaGpuNativeLease *lease, const uint64_t **row) {
    uint64_t out[32] = {0}, result = 0;
    if (row) *row = NULL;
    if (!s) return 1;
    if (omega_native_graph(&result, s->query, 64, s->rows, 1024, cache->pool, 528,
        s->wanted, 4096, s->records, 65536, out, 32) != OMEGA_OK || result != 0) return -1;
    /* The generated resolver already bounds the immutable ordinal to <16. */
    *artifact = &s->artifact[out[1]];
    *lease = (OmegaGpuNativeLease){cache, out[17], out[18], out[19]};
    /* The returned lease pins the selected exact resident realization. Recording
     * takes its own reference; the temporary selection pin is always released. */
    if (omega_native_bound(*artifact, lease) != 0) {
        (void)omega_native_lease_op(lease, 9, 0);
        *lease = (OmegaGpuNativeLease){0};
        return -1;
    }
    if (row) *row = s->rows + out[1] * 64;
    return 0;
}
static inline int omega_q36_native_begin(OmegaGpuQ36NativeState *s, OmegaGpuNativeCache *cache,
    const OmegaGpuNativeArtifact **artifact, OmegaGpuNativeLease *lease) {
    return omega_q36_native_begin_selected(s, cache, artifact, lease, NULL);
}
static inline void omega_q36_native_end(const OmegaGpuNativeLease *lease) {
    (void)omega_native_lease_op(lease, 9, 0);
}
static inline void omega_q36_native_release(OmegaGpuQ36NativeState *s) {
    if (!s) return;
    for (unsigned i = 0; i < 16; ++i)
        if (s->prepared[i].cache) (void)omega_native_lease_op(&s->prepared[i], 5, 0);
}
static inline void omega_q36_native_graph_release(OmegaGpuQ36NativeGraph *g) {
    if (!g || !g->context) return;
    for (uint64_t i = 0; i < g->count && i < 32; ++i)
        omega_q36_native_release(&g->context[i]);
}
#endif
