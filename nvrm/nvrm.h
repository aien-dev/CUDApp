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

typedef struct {
    uint32_t handle;
    uint64_t va;      /* identical GPU and CPU virtual address */
    uint64_t size;
    void    *cpu;     /* CPU mapping (== (void *)va) */
} NvrmMem;

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
    char     err[256];
} Nvrm;

int  nvrm_open(Nvrm *rm);
int  nvrm_alloc(Nvrm *rm, uint64_t size, NvrmMem *out);
int  nvrm_channel(Nvrm *rm);
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
void nvrm_close(Nvrm *rm);

/* Pushbuffer method header (non-incrementing=1, incrementing=2 / SEC_OP_INC_METHOD) */
static inline uint32_t nvrm_mthd(uint32_t subc, uint32_t mthd, uint32_t count) {
    return (2u << 28) | (count << 16) | (subc << 13) | (mthd >> 2);
}

#endif
