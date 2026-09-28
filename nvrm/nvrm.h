/*
 * nvrm.h -- PHYSICS native accelerator substrate (NVRM client) (shared by M16 qualification and M17).
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

typedef struct {
    int fd_ctl, fd_dev, fd_uvm, fd_uvm2;
    uint32_t root, device, subdevice, virtmem, usermode, vaspace;
    uint32_t chgroup, ctxshare, gpfifo, compute_obj;
    uint32_t next_handle;
    uint8_t  gpu_uuid[16];
    uint32_t gpu_id, minor;
    uint32_t compute_class, gpfifo_class, usermode_class;
    uint32_t sm_version;            /* NV2080_CTRL_GR_INFO_INDEX_SM_VERSION */
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
    uint64_t      rm_alloc_accepted;
    uint64_t      rm_free_accepted;
    char     err[256];
} Nvrm;

int  nvrm_open(Nvrm *rm);
int  nvrm_alloc(Nvrm *rm, uint64_t size, NvrmMem *out);
/* Reverse of nvrm_alloc: drops the UVM external range, frees the RM memory
 * object, unmaps the CPU range, and returns the VA range for reuse by a
 * later nvrm_alloc. Idempotent on an already-zeroed NvrmMem (returns 0,
 * no syscalls). Returns -1 and sets rm->err, without touching the driver,
 * if *m does not match a currently-live allocation (e.g. a stale copy).
 * Zeroes *m on success. */
int  nvrm_free(Nvrm *rm, NvrmMem *m);
int  nvrm_channel(Nvrm *rm);
int  nvrm_channel_destroy(Nvrm *rm);
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

#endif
/* As nvrm_alloc, but the graphics chip does not cache it in L2. For memory
 * that a resident chip program polls while the CPU writes it. */
int nvrm_alloc_gpu_uncached(Nvrm *rm, uint64_t size, NvrmMem *out);
