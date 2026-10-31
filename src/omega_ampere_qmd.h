/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_AMPERE_QMD_H
#define OMEGA_AMPERE_QMD_H
#include <stdint.h>
/* NVIDIA open-gpu-doc classes/compute/clc7c0.h AMPERE_COMPUTE_B,
 * Vendor source pin: NVIDIA open-gpu-doc@9fdf5c4062007929d9f4e6cbad9c9771fe61b880.
 * clc6c0.h AMPERE_COMPUTE_A, host/clc56f.h AMPERE_CHANNEL_GPFIFO_A.
 * SM86 uses COMPUTE_B, not the GA100 COMPUTE_A class. These are not qualifications. */
#define OMEGA_AMPERE_COMPUTE_A 0xc6c0u
#define OMEGA_AMPERE_COMPUTE_B 0xc7c0u
#define OMEGA_AMPERE_GPFIFO_A 0xc56fu
#define OMEGA_AMPERE_QMD_WORDS 64u
/* Internal bridge entry shared by NVRM's class/reserve admission and QMD host glue.
 * om/encoder/amp_runtime.om is the implementation; no C fallback. */
int omega_ampere_policy(const uint64_t m[16], uint32_t q[64]);
int omega_ampere_build_qmd(uint32_t q[64], uint64_t code, uint64_t cbank,
                           uint64_t semaphore, uint32_t tx, uint32_t ty,
                           uint32_t gx, uint32_t gy, uint32_t registers,
                           uint32_t shared_bytes);
#endif
