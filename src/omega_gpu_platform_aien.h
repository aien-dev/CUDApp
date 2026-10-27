/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* omega_gpu_platform_aien.h -- OmegaGpuPlatform (ABI 1.2) over AIEN's own GB10 driver, aien-gpu.ko (/dev/aien-gpu).
 *
 * The second implementation of the platform contract (src/omega_gpu_platform.h). The first,
 * src/omega_gpu_platform_linux.c, drives NVIDIA's open kernel module through nvrm/ and m16/; this one drives the
 * aien-gpu compute ABI 1 (third_party/aien-gpu-abi1/aien_gpu_uapi.h) and includes nothing from nvrm/, m16/ or
 * NVIDIA's headers. Everything above the contract (session, command list, libraries) runs unchanged on either.
 *
 * Same caps and the same refusals as the Linux reference: TIME, WAIT, LOCK, DMA, SYNC, COMPLETION (polling), DEVICE,
 * QUEUE, CLOCK_NS, HOST_MEM; no MMIO, no IRQ; one device (ordinal 0), one queue; GPU-cached buffers only for
 * device-private DATA buffers, with the omega #457 L2 invalidate prefix after a host sync into one. The bus address
 * is reported as 0 (the module owns the IOMMU mapping).
 *
 *   device_open  = open(/dev/aien-gpu) + AGPU_INFO; refused unless abi 1 and state AGPU_STATE_RM_READY
 *   dma_alloc    = reserve a CPU VA range in [64 GiB, 1 TiB) (mmap PROT_NONE, MAP_FIXED_NOREPLACE; a GPFIFO entry carries
 *                  40 address bits), AGPU_MEM_ALLOC at that VA (GPU VA == CPU VA), mmap the object
 *                  MAP_FIXED|MAP_SHARED over the reservation
 *   dma_free     = AGPU_MEM_FREE, munmap
 *   queue_open   = one GPU-uncached buffer holding the GPFIFO ring (1024 entries) and USERD right after it, one
 *                  error-notifier buffer, AGPU_CHAN_CREATE, mmap of the 64 KiB usermode (doorbell) page
 *   queue_submit = methods into the BUF_QUEUE ring, dsb, GP entry, dsb, USERD GPPut, dsb, token store at +0x90
 *   queue_retire = every submitted entry consumed (caller observed completion)
 *   poll_word    = (int32_t)(word - expected) >= 0, 50 us sleep-poll
 *
 * Evidence: IMPLEMENTED and host-tested on a simulated device (tests/test_omega_gpu_platform_aien.c). Hardware
 * NOT_RUN: it has never opened a real /dev/aien-gpu.
 *
 * Selection: a program installs it with omega_gpu_session_set_platform(omega_gpu_platform_aien(&caps), &caps)
 * before the first session open. The hosted session still installs the Linux reference by default.
 * CUDAPP_AIEN_GPU_DEV names another device node (default /dev/aien-gpu).
 *
 * Migration-pending C: the definition exists only when built with CUDAPP_MIGRATION_PENDING_AIEN_GPU=1 (OFF in
 * libomega_gpu.a; ON for the host test and its mutants). Successor native/aiengpu/context.om, aien-dev/CUDApp#128,
 * blocked on the bridge extension aien-dev/omega-bridge#29.
 */
#ifndef OMEGA_GPU_PLATFORM_AIEN_H
#define OMEGA_GPU_PLATFORM_AIEN_H

#include "omega_gpu_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_GPU_PLATFORM_AIEN_LOCKS 4u          /* lock ids 0..3, as the Linux reference */
#define OMEGA_GPU_AIEN_DEFAULT_DEV "/dev/aien-gpu"
#define OMEGA_GPU_AIEN_GP_ENTRIES 1024u           /* GPFIFO entries, as nvrm_channel */
#define OMEGA_GPU_AIEN_USERD_OFF (OMEGA_GPU_AIEN_GP_ENTRIES * 8u) /* USERD follows the ring in the same buffer */
#define OMEGA_GPU_AIEN_FIFO_BYTES 0x10000u        /* ring + USERD, GPU-uncached */
#define OMEGA_GPU_AIEN_NOTIFIER_BYTES 0x10000u    /* channel error notifier */
#define OMEGA_GPU_AIEN_USERD_GPPUT 0x8cu          /* Nvc96fControl.GPPut (clc96f.h: 008c-008f) */
#define OMEGA_GPU_AIEN_DOORBELL_BYTES 0x10000u    /* the usermode page */
#define OMEGA_GPU_AIEN_DOORBELL_WORD 0x90u        /* NVC361_NOTIFY_CHANNEL_PENDING */

/* The process-wide aien-gpu platform and its caps (written to *caps when caps is not NULL). */
const OmegaGpuPlatform *omega_gpu_platform_aien(OmegaGpuPlatformCaps *caps);

/* One GPFIFO entry: pushbuffer VA (dword aligned), LEVEL MAIN, LENGTH 30:10 in dwords, SYNC PROCEED. Bit for bit the
 * entry nvrm_gp_entry writes (tests/test_omega_gpu_platform_aien.c compares them). */
uint64_t omega_gpu_aien_gp_entry(uint64_t va, uint32_t nwords);

/* The system calls the platform makes, replaceable for host tests (a simulated device). NULL restores the real
 * open/close/ioctl/mmap/munmap. Install only while no device is open. */
typedef struct {
    int (*open_dev)(const char *path, int flags);
    int (*close_dev)(int fd);
    int (*ioctl_dev)(int fd, unsigned long request, void *arg);
    void *(*map)(void *addr, size_t len, int prot, int flags, int fd, uint64_t off);
    int (*unmap)(void *addr, size_t len);
} OmegaGpuAienSys;
int omega_gpu_platform_aien_set_sys(const OmegaGpuAienSys *sys); /* REFUSED_BUSY while a device is open */

#ifdef __cplusplus
}
#endif
#endif
