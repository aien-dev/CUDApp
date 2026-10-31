/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* omega_gpu_platform_linux.h -- the Linux REFERENCE implementation of OmegaGpuPlatform (ABI 1.1).
 *
 * REFERENCE ONLY (native v1 P07 cut C, session part; aienos#286). This is today's GB10 path under Linux
 * (physics nvrm + m16_native, the open NVIDIA kernel driver), wrapped in the platform contract so the
 * portable session (src/omega_gpu_session.c) runs the same launch words it always ran. It is NOT the
 * native platform and is not part of the freestanding closure: it includes Linux headers, takes pthread
 * mutexes, reads CLOCK_MONOTONIC and calls malloc. The AIENOS platform implements the same table over its
 * own device ownership (cut B); nothing in the session may depend on anything this file does beyond the
 * contract in src/omega_gpu_platform.h.
 *
 * What it provides: TIME, WAIT, LOCK, DMA, SYNC, COMPLETION (polling: m16_native_wait_marker, a 50 us
 * sleep-poll), DEVICE, QUEUE, CLOCK_NS, HOST_MEM. Not MMIO (the RM driver owns the registers) and not IRQ
 * (polling is never claimed as interrupt support). One device (ordinal 0), one queue (the m16 channel,
 * GPFIFO entries carrying the method words copied into the BUF_QUEUE ring), every buffer GPU-uncached
 * (nvrm_alloc_gpu_uncached; a GPU_CACHED request is refused with REFUSED_CACHE_POLICY because nothing here
 * would flush it). The bus address is reported as 0: the RM driver owns the IOMMU mapping and does not
 * expose it. Closing the device makes every buffer and queue handle of it stale (generation bump); the RM
 * client close releases the driver objects, as the session always did.
 *
 * Tested by tests/test_omega_gpu_platform_linux.c on the simulated driver (tests/fake_m16_native.c);
 * GB10 execution through it is the same driver calls as before this cut and is NOT re-run here.
 */
#ifndef OMEGA_GPU_PLATFORM_LINUX_H
#define OMEGA_GPU_PLATFORM_LINUX_H

#include "omega_gpu_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_GPU_PLATFORM_LINUX_LOCKS 4u /* lock ids 0..3 */

/* The process-wide Linux reference platform and its caps (written to *caps when caps is not NULL). */
const OmegaGpuPlatform *omega_gpu_platform_linux(OmegaGpuPlatformCaps *caps);

/* L136: channels (GPFIFO queues) the next device_open creates, from env AIEN_OMEGA_GPU_CHANNELS: 1 (unset, empty, "1" or
 * anything not 2..4) is today's single channel exactly; 2..4 adds that many minus one channels in the same channel group
 * (nvrm_channel_n). Queue i (0-based, in queue_open order) submits on channel i. Read at every device_open. */
uint32_t omega_gpu_platform_linux_channels_env(void);
/* Exact target of the open Linux device, or 0 while closed. */
uint32_t omega_gpu_platform_linux_target_sm(void);
int omega_gpu_platform_linux_vram_info(uint64_t *free_bytes, uint64_t *total_bytes);

/* TEST ONLY. The D1 memory reserve check reads MemAvailable from path instead of /proc/meminfo; NULL or "" restores
 * /proc/meminfo. A path that does not open, without following a final symlink and without blocking, to a regular file
 * falls back to /proc/meminfo. Returns 0, or -1 (nothing changed) when path is 4096 bytes or longer. */
int omega_gpu_platform_linux_test_set_meminfo(const char *path);

#ifdef __cplusplus
}
#endif
#endif
