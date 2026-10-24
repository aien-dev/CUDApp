/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

#ifdef __cplusplus
}
#endif
#endif
