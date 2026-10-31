/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#include "omega_ampere_qmd.h"
#include <stddef.h>
/* Declaration also emitted by pinned omega-bridge native.om. Until the main
 * integration regenerates cudapp_native.h, keep this exact ABI declaration. */
extern uint32_t omega_amp_runtime(uint64_t *, const void *, uint64_t, void *, uint64_t);
int omega_ampere_policy(const uint64_t m[16], uint32_t q[64]) {
    uint64_t out[64] = {0}, accepted = 0;
    if (!m || omega_amp_runtime(&accepted, m, 16, out, 64) != 0 || accepted != 1) return -1;
    if (q) for (size_t i = 0; i < 64; i++) q[i] = (uint32_t)out[i];
    return 0;
}
int omega_ampere_build_qmd(uint32_t q[64], uint64_t code, uint64_t cbank,
                           uint64_t semaphore, uint32_t tx, uint32_t ty,
                           uint32_t gx, uint32_t gy, uint32_t registers,
                           uint32_t shared_bytes) {
    const uint64_t m[16] = {3, code, cbank, semaphore, tx, ty, gx, gy, registers, shared_bytes};
    return q ? omega_ampere_policy(m, q) : -1;
}
