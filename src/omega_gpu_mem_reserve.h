/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* omega_gpu_mem_reserve.h -- developer bar D1 (docs/PAIN_POINTS.md, docs/issues/D1-memory.md), shared by the hosted
 * platforms (src/omega_gpu_platform_linux.c, src/omega_gpu_platform_aien.c).
 *
 * On the GB10 every buffer a platform hands out is the machine's own memory, which the CPU and GPU share, and the
 * machine has no swap by default. Running it out does not end in an error message: the whole box can freeze until a
 * power cycle. So an allocation that would leave less than a reserve free is refused before a handle slot or the
 * driver is touched, with a sentence. The reserve is CUDAPP_MEM_RESERVE_MIB (MiB, default 4096; 0 turns the check
 * off). Tests name another meminfo file with omega_gpu_mem_reserve_test_set_meminfo; CUDAPP_MEMINFO is ignored
 * (CodeQL cpp/path-injection #40). A test path that does not open, without following a final symlink and without
 * blocking, to a regular file falls back to /proc/meminfo. When that file cannot be read the check cannot know,
 * and allows the allocation.
 */
#ifndef OMEGA_GPU_MEM_RESERVE_H
#define OMEGA_GPU_MEM_RESERVE_H

#include <stddef.h>
#include <stdint.h>

#define CUDAPP_MEM_RESERVE_DEFAULT_MIB 4096ull

/* OMEGA_GPU_PLAT_OK, or OMEGA_GPU_PLAT_REFUSED_LIMIT with the sentence in diag (diag_len bytes). */
int omega_gpu_mem_reserve_check(uint64_t bytes, char *diag, size_t diag_len);

/* TEST ONLY. Read MemAvailable from path instead of /proc/meminfo; NULL or "" restores /proc/meminfo. A path that
 * does not open, without following a final symlink and without blocking, to a regular file falls back to
 * /proc/meminfo. Returns 0, or -1 (nothing changed) when path is 4096 bytes or longer. */
int omega_gpu_mem_reserve_test_set_meminfo(const char *path);

#endif
