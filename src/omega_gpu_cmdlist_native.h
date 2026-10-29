/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_GPU_CMDLIST_NATIVE_H
#define OMEGA_GPU_CMDLIST_NATIVE_H
#include "omega_gpu_cmdlist.h"
/* Private owner projections, valid only under the existing session lock. They
 * expose actual typed allocations and mode-final bytes, never qualification.
 * The observer runs once after every entry is finalized and before any queue
 * commit, with the very options consumed by that submit. It must not mutate
 * the list, submit work, or retain these borrowed pointers. Refusal queues
 * nothing; the graph owner must reset its unsubmitted list without fallback. */
typedef struct {
    OmegaGpuMem banks, qmds, semaphores, scratch, stamps;
    uint64_t record_generation;
    uint32_t capacity, count;
} OmegaGpuCmdListOwner;
typedef struct {
    const uint8_t *bank, *qmd_slot;
    const uint8_t *intent_seal; /* owner-frozen successful native add, NOT admission/qualification */
    uint64_t record_generation, code_va, pending_wait;
    uint32_t ordinal, count, n_args, independent;
} OmegaGpuCmdListEntry;
typedef struct {
    OmegaGpuCmdListOpts options;
    uint64_t timeout_ms;
    uint32_t spin_us;
} OmegaGpuCmdListConsumed;
typedef int (*OmegaGpuCmdListObserver)(const OmegaGpuCmdList *,
                                     const OmegaGpuCmdListConsumed *, void *);
int omega_gpu_cmdlist_owner(const OmegaGpuCmdList *, OmegaGpuCmdListOwner *);
int omega_gpu_cmdlist_entry(const OmegaGpuCmdList *, uint32_t, OmegaGpuCmdListEntry *);
int omega_gpu_cmdlist_submit_observed(OmegaGpuCmdList *, const OmegaGpuCmdListOpts *,
                                     OmegaGpuCmdListStats *, OmegaGpuCmdListObserver, void *);
#endif
