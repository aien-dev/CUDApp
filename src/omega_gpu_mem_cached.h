#ifndef OMEGA_GPU_MEM_CACHED_H
#define OMEGA_GPU_MEM_CACHED_H
/* GPU-cached device-visible memory for read-only weights (OWN-30B L2 kernel-time cut).
 *
 * The session's own allocations are GPU-uncached (nvos.h NVOS32_ATTR2_GPU_CACHEABLE: "For system memory this will not
 * be coherent with direct CPU mappings"). Weights are written once by the CPU before their first GPU use
 * (omega_gpu_session_sync_to_gpu), are never host-polled and never rewritten, so the GPU may cache them. Same descriptor
 * as omega_gpu_prefill.c in omega-own30b-tc (W-L1G, T3b 4a84906b) minus CPU_READ: the platform allows GPU-cached data buffers
 * only without CPU_READ / HOST_POLLED. Invariant here: the CPU writes the buffer once (fresh allocation, never read by the GPU
 * yet, so no stale L2 line exists), omega_gpu_session_sync_to_gpu flushes the CPU side, then only the GPU reads it, ever.
 * The caller holds the session lock with the device
 * open (as for omega_gpu_session_alloc); the buffer is freed and synced through the session like any other. */
#include <stddef.h>
#include "omega_gpu_session.h"
#ifdef __cplusplus
extern "C" {
#endif
int omega_gpu_session_alloc_cached(size_t bytes, OmegaGpuMem *out);
#ifdef __cplusplus
}
#endif
#endif
