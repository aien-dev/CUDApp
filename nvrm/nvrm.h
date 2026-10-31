/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * nvrm.h -- PHYSICS native accelerator substrate (NVRM client), used by M16 qualification,
 * FORGE realization and the NVRM gates in this tree. M17 (OMEGA_BLACKWELL_VECTOR) was
 * qualified in aien-spark/omega, not here (aien-architecture doctrine/ROADMAP.md).
 *
 * Drives the NVIDIA open kernel module (RM + UVM) directly through ioctls on
 * /dev/nvidiactl, /dev/nvidia0 and /dev/nvidia-uvm. No libcuda, no CUDA
 * runtime, no CUDA toolkit: every object (client, device, VA space, memory,
 * channel group, GPFIFO channel, compute object) is allocated here.
 */
#ifndef PHYSICS_NVRM_H
#define PHYSICS_NVRM_H

#include <stddef.h>
#include <stdint.h>

#define NVRM_MAX_MEM 16

/* Capacity of the free-VA-range table (nvrm_free / va_take reuse) and the
 * live-allocation table (nvrm_free validation). Separate from NVRM_MAX_MEM,
 * which no caller currently uses, so existing behavior is unaffected. */
#define NVRM_MAX_FREE 4096
#define NVRM_MAX_LIVE 4096

typedef struct {
    uint32_t handle;
    uint64_t va;      /* identical GPU and CPU virtual address */
    uint64_t size;
    void    *cpu;     /* CPU mapping (== (void *)va) */
} NvrmMem;

/* One coalesced free VA range available for reuse by va_take(). */
typedef struct {
    uint64_t va;
    uint64_t size;
} NvrmVaRange;

/* The compute object makes the RM allocate the channel's main graphics context
 * buffer as ONE physically contiguous 2 MiB system-memory block (order 9), asked
 * for once with __GFP_RETRY_MAYFAIL and no scattered fallback (open driver
 * kernel-open/nvidia/nv-vm.c:377-391, kernel_graphics_context.c:1129; kernel
 * tracepoint record in omega#523). On a long-running host that block is often
 * missing for a few hundred milliseconds, so nvrm_channel retries that one
 * RM_ALLOC, bounded, before NV_ERR_NO_MEMORY is final. Measured 2026-10-17 on
 * the Spark: every refusal cleared within 5 attempts at 100 ms (omega#523). */
#ifndef NVRM_CHANNEL_NO_MEMORY_RETRIES
#define NVRM_CHANNEL_NO_MEMORY_RETRIES 50
#endif
#ifndef NVRM_CHANNEL_NO_MEMORY_RETRY_MS
#define NVRM_CHANNEL_NO_MEMORY_RETRY_MS 100
#endif

/* One outstanding allocation, tracked so nvrm_free() can tell a real,
 * still-live NvrmMem apart from a stale/duplicate copy. */
typedef struct {
    uint32_t handle;
    uint64_t va;
    uint64_t size;
    void *cpu;
    uint8_t uvm_live;
    uint8_t dma_live;
    uint8_t rm_live;
    uint8_t cpu_live;
    uint8_t quarantined;
} NvrmLiveAlloc;

/* An extra command queue (GPFIFO channel) in the same channel group (TSG) and
 * sub-context (FERMI_CONTEXT_SHARE_A) as channel 0, created by nvrm_channel_n
 * before the group is scheduled: same VA space, so every buffer is visible to
 * both, and work on different channels is not ordered against each other
 * (CUDA guide, CUDA_DEVICE_MAX_CONNECTIONS: tasks on one work queue get a
 * false dependency; separate queues remove it). */
#define NVRM_MAX_EXTRA_CHANNELS 3
typedef struct {
    uint32_t gpfifo, compute_obj;
    uint32_t ctxshare;              /* Owned asynchronous subcontext, distinct from channel 0. */
    NvrmMem  fifo;                  /* GPFIFO ring, then USERD at userd_off */
    NvrmMem  notifier;              /* channel error notifier */
    uint32_t entries, userd_off;
    uint32_t token;                 /* work submit token */
    uint32_t put, retired;
    uint64_t channel_va, channel_va_size;
    uint8_t  registered;
} NvrmChan;

typedef struct {
    int fd_ctl, fd_dev, fd_uvm, fd_uvm2;
    uint32_t root, device, subdevice, virtmem, usermode, vaspace;
    uint32_t chgroup, ctxshare, gpfifo, compute_obj;
    uint32_t next_handle;
    uint8_t  gpu_uuid[16];
    uint32_t gpu_id, minor;
    uint32_t compute_class, gpfifo_class, usermode_class;
    uint32_t sm_version;            /* NV2080_CTRL_GR_INFO_INDEX_SM_VERSION */
    uint32_t target_sm;             /* exact execution target: 86 or 121 */
    uint8_t discrete;               /* local framebuffer, not GB10 coherent system memory */
    void    *usermode_cpu;          /* HOPPER_USERMODE_A MMIO mapping (RM-provided, BAR0-relative) */
    volatile uint32_t *doorbell;    /* usermode MMIO + NVC361_NOTIFY_CHANNEL_PENDING */
    NvrmMem  fifo;                  /* GPFIFO ring, then USERD at userd_off */
    NvrmMem  notifier;              /* channel error notifier */
    uint32_t entries;
    uint32_t userd_off;
    uint32_t token;                 /* work submit token */
    uint32_t put;                   /* entries enqueued (monotonic) */
    uint32_t retired;               /* entries the caller has seen complete */
    uint64_t va_next;
    uint64_t va_slot_base;
    NvrmVaRange   free_list[NVRM_MAX_FREE];  /* sorted ascending by va, coalesced */
    uint32_t      free_count;
    NvrmLiveAlloc live[NVRM_MAX_LIVE];
    uint32_t      live_count;
    uint64_t      channel_va;
    uint64_t      channel_va_size;
    uint8_t       channel_registered;
    uint8_t       gpu_registered;
    uint8_t       vas_registered;
    uint8_t       faulted;
    /* Fault injection applies once at the driver wrapper boundary. */
    uint8_t       inject_uvm_free_failure;
    uint8_t       inject_rm_free_failure;
    /* Refuse the next N compute-object RM_ALLOCs with NV_ERR_NO_MEMORY before
     * the ioctl, so the retry in nvrm_channel can be exercised deterministically. */
    uint8_t       inject_rm_alloc_no_memory;
    uint32_t      last_rm_status;             /* status of the last RM_ALLOC that reached the driver (or was injected) */
    uint32_t      channel_no_memory_retries;  /* NV_ERR_NO_MEMORY refusals the last nvrm_channel retried */
    uint32_t      channel_no_memory_retry_ms; /* delay between those retries; 0 selects NVRM_CHANNEL_NO_MEMORY_RETRY_MS */
    uint64_t      rm_alloc_accepted;
    uint64_t      rm_free_accepted;
    NvrmChan      extra[NVRM_MAX_EXTRA_CHANNELS]; /* channels 1.. (nvrm_channel_n) */
    uint32_t      n_extra;
    char     err[256];
} Nvrm;

int  nvrm_open(Nvrm *rm);
int  nvrm_alloc(Nvrm *rm, uint64_t size, NvrmMem *out);
/* Local framebuffer on discrete SM86, CPU BAR mapped by RM_MAP_MEMORY. The
 * CPU and GPU views share a VA but not a cache. No system-memory fallback. */
int nvrm_alloc_vram(Nvrm *rm, uint64_t size, NvrmMem *out);
/* Fresh driver framebuffer heap query. Never substitutes /proc/meminfo. */
int nvrm_vram_info(Nvrm *rm, uint64_t *free_bytes, uint64_t *total_bytes);
/* Bounds/live-identity checked CPU BAR transfers. Caller proves GPU idle before
 * readback or overwrite; the fence orders mapped PCIe writes before submission. */
int nvrm_copy_to(Nvrm *rm, const NvrmMem *m, uint64_t off, const void *src, size_t bytes);
int nvrm_copy_from(Nvrm *rm, const NvrmMem *m, uint64_t off, void *dst, size_t bytes);
/* Reverse of nvrm_alloc: drops the UVM external range, frees the RM memory
 * object, unmaps the CPU range, and returns the VA range for reuse by a
 * later nvrm_alloc. Idempotent on an already-zeroed NvrmMem (returns 0,
 * no syscalls). Returns -1 and sets rm->err, without touching the driver,
 * if *m does not match a currently-live allocation (e.g. a stale copy).
 * Zeroes *m on success. */
int  nvrm_free(Nvrm *rm, NvrmMem *m);
int  nvrm_channel(Nvrm *rm);
/* nvrm_channel plus `extra` (0..NVRM_MAX_EXTRA_CHANNELS) more GPFIFO channels in
 * the same group and address space, each with its own asynchronous subcontext,
 * all registered with UVM before the one GPFIFO_SCHEDULE (the
 * RM refuses to schedule a group holding an unschedulable channel,
 * kernel_channel_group_api.c:1095-1107). extra = 0 is nvrm_channel exactly. */
int  nvrm_channel_n(Nvrm *rm, uint32_t extra);
int  nvrm_channel_destroy(Nvrm *rm);
/* Channel ch (1..n_extra) of nvrm_channel_n: enqueue + doorbell, retire, entries
 * submitted so far. Same contract as nvrm_submit / nvrm_retire on channel 0. */
int  nvrm_chan_submit(Nvrm *rm, uint32_t ch, const NvrmMem *pb, uint32_t off_bytes, uint32_t nwords);
void nvrm_chan_retire(Nvrm *rm, uint32_t ch, uint32_t upto);
uint32_t nvrm_chan_put(const Nvrm *rm, uint32_t ch);
/* Write a GP entry and advance USERD GPPut; does NOT ring the doorbell. */
int  nvrm_enqueue(Nvrm *rm, const NvrmMem *pb, uint32_t off_bytes, uint32_t nwords);
/* Mark entries [.., upto) complete; the caller must have observed their completion. */
void nvrm_retire(Nvrm *rm, uint32_t upto);
/* Usermode doorbell store (NVC361_NOTIFY_CHANNEL_PENDING <- work submit token). */
void nvrm_ring(Nvrm *rm);
/* enqueue + ring */
int  nvrm_submit(Nvrm *rm, const NvrmMem *pb, uint32_t off_bytes, uint32_t nwords);
uint64_t nvrm_gp_entry(uint64_t va, uint32_t nwords);
volatile uint32_t *nvrm_userd_gpput(Nvrm *rm);
int  nvrm_close(Nvrm *rm);

/* Pushbuffer method header (non-incrementing=1, incrementing=2 / SEC_OP_INC_METHOD) */
static inline uint32_t nvrm_mthd(uint32_t subc, uint32_t mthd, uint32_t count) {
    return (2u << 28) | (count << 16) | (subc << 13) | (mthd >> 2);
}

/* As nvrm_alloc, but the graphics chip does not cache it in L2. For memory
 * that a resident chip program polls while the CPU writes it. */
int nvrm_alloc_gpu_uncached(Nvrm *rm, uint64_t size, NvrmMem *out);
#endif
