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

#ifndef OMEGA_BLACKWELL_CODEGEN_PF_H
#define OMEGA_BLACKWELL_CODEGEN_PF_H
/*
 * omega_blackwell_codegen_pf: small device-resident kernels for the OWN-30B one-pass prefill (T3b, 2026-10-13). New file.
 *
 * copy: dst[i] = src[i] for i < units (16-byte units). One thread per unit, grid_x = ceil(units / threads).
 *   Arguments (cbank + 0x380): dst u64, src u64, units u32. Used to fill GPU-cached (device-private) buffers from an
 *   uncached staging buffer and to read them back: every write to a cached buffer is a GPU write, so the GPU L2 can never
 *   hold a stale line of it (nvos.h:1116, omega_gpu_devmem.h).
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif
#define OMEGA_PF_COPY_THREADS 256u
int omega_blackwell_codegen_pf_copy_prog(BlackwellIRProgram *prog);
#ifdef __cplusplus
}
#endif
#endif
