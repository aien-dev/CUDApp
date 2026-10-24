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

#include "omega_gpu_mem_cached.h"
#include <string.h>
#include "omega_gpu_platform.h"
#include "omega_gpu_fmt.h"

int omega_gpu_session_alloc_cached(size_t bytes, OmegaGpuMem *out) {
    if (!out || bytes == 0) { omega_gpu_session_set_error("cached alloc: arguments"); return -1; }
    memset(out, 0, sizeof *out);
    OmegaGpuSession *s = omega_gpu_session_open();
    const OmegaGpuPlatform *p = omega_gpu_session_platform();
    if (!s || !p) return -1;
    const size_t want = (bytes + 0xfffu) & ~(size_t)0xfffu;
    OmegaGpuBufferDesc d = { .kind = OMEGA_GPU_HANDLE_BUF_DATA, .rights = OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_READ, /* no CPU_READ: the platform refuses a CPU-readable GPU-cached buffer */
                             .gpu_cache = OMEGA_GPU_CACHE_GPU_CACHED, .align = 0x1000u, .bytes = want };
    OmegaCpuView cpu; OmegaGpuVa va; OmegaGpuBusAddr bus;
    int prc = p->dma_alloc(p->ctx, s->dev, &d, &out->handle, &cpu, &va, &bus);
    if (prc != OMEGA_GPU_PLAT_OK || !cpu.ptr || cpu.len < want) {
        if (prc == OMEGA_GPU_PLAT_OK) (void)p->dma_free(p->ctx, out->handle);
        char e[96]; omega_gpu_fmt(e, sizeof e, "cached alloc of %zu bytes refused (%d)", want, prc);
        omega_gpu_session_set_error(e); memset(out, 0, sizeof *out); return -1;
    }
    out->kind = OMEGA_GPU_HANDLE_BUF_DATA; out->cpu = cpu.ptr; out->va = va.va; out->size = cpu.len;
    return 0;
}
