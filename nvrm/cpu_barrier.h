/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef NVRM_CPU_BARRIER_H
#define NVRM_CPU_BARRIER_H

/* CPU publication order for memory the device reads or the host reads back.
 *
 * AArch64, including the GB10 host, executes `dsb sy`. That system barrier
 * is the production ordering primitive. A portable host build executes a
 * sequentially consistent fence, which is a real instruction, so the same
 * sources compile on x86-64. Opening the Blackwell device stays AArch64-only.
 *
 * ADR 0043 missing feature: seeded Omega has no host CPU publication barrier
 * that is `dsb sy` on AArch64 and a real fence on every other host. This
 * header is that barrier in C. The non-AArch64 fence is migration-pending C.
 * The portable library compiles the callers with the existing
 * CUDAPP_MIGRATION_PENDING_C_ENCODER and CUDAPP_MIGRATION_PENDING_C_CUBIN
 * switches, which select the C encoder and the C cubin writer. Extension
 * proposal drafted, not yet filed on omega-mv:
 * evidence/HOST-X86-20261030/proposal-cpu-barrier.md
 */
static inline void nvrm_cpu_barrier(void)
{
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/* Spin hint between polls. AArch64 keeps `yield`. Other hosts keep the
 * compiler barrier already used by src/omega_gpu_wait.c. This is not the
 * publication barrier above.
 */
static inline void nvrm_cpu_relax(void)
{
#if defined(__aarch64__)
    __asm__ volatile("yield");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

#endif
