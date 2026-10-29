/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* omega_gpu_mem_reserve.c -- developer bar D1: refuse an allocation that would leave less than the reserve free
 * (src/omega_gpu_mem_reserve.h). Hosted only: reads /proc/meminfo, or a regular file a test names by code. */
#include "omega_gpu_mem_reserve.h"

#include "omega_gpu_platform.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* CodeQL cpp/path-injection #40: the meminfo path is never taken from the environment. */
static char g_meminfo_test_path[4096];

int omega_gpu_mem_reserve_test_set_meminfo(const char *path) {
    size_t n = path ? strlen(path) : 0;
    if (n >= sizeof g_meminfo_test_path) return -1;
    memcpy(g_meminfo_test_path, path ? path : "", n);
    g_meminfo_test_path[n] = '\0';
    return 0;
}

static FILE *meminfo_open_regular(const char *path) {
    int fd = open(path, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) { close(fd); return NULL; }
    FILE *f = fdopen(fd, "r");
    if (!f) close(fd);
    return f;
}

static FILE *meminfo_open(void) {
    /* Read and discard: a leftover CUDAPP_MEMINFO must not choose the file. */
    const char *ignored = getenv("CUDAPP_MEMINFO");
    (void)ignored;
    if (g_meminfo_test_path[0]) {
        FILE *f = meminfo_open_regular(g_meminfo_test_path);
        if (f) return f;
    }
    return meminfo_open_regular("/proc/meminfo");
}

static void human_bytes(char *out, size_t n, uint64_t b) {
    if (b < (1ull << 30)) snprintf(out, n, "%.1f MiB", (double)b / (double)(1ull << 20));
    else snprintf(out, n, "%.1f GiB", (double)b / (double)(1ull << 30));
}

int omega_gpu_mem_reserve_check(uint64_t bytes, char *diag, size_t diag_len) {
    unsigned long long reserve_mib = CUDAPP_MEM_RESERVE_DEFAULT_MIB;
    const char *e = getenv("CUDAPP_MEM_RESERVE_MIB");
    if (e && *e) { char *end = NULL; unsigned long long v = strtoull(e, &end, 10); if (end && end != e && *end == 0) reserve_mib = v; }
    if (reserve_mib == 0) return OMEGA_GPU_PLAT_OK;
    FILE *f = meminfo_open();
    if (!f) return OMEGA_GPU_PLAT_OK;
    unsigned long long avail_kb = 0; int have = 0; char line[256];
    while (!have && fgets(line, sizeof line, f)) have = sscanf(line, "MemAvailable: %llu kB", &avail_kb) == 1;
    fclose(f);
    if (!have) return OMEGA_GPU_PLAT_OK;
    /* Saturate rather than wrap: 2^44 MiB shifted by 20 is 2^64, and 2^54 kB times 1024 is 2^64 (RtMem F9). */
    const uint64_t avail = avail_kb > UINT64_MAX / 1024ull ? UINT64_MAX : (uint64_t)avail_kb * 1024ull;
    const uint64_t reserve = reserve_mib > (UINT64_MAX >> 20) ? UINT64_MAX : (uint64_t)reserve_mib << 20;
    if (bytes <= avail && avail - bytes >= reserve) return OMEGA_GPU_PLAT_OK;
    char want[32], free_[32], keep[32];
    human_bytes(want, sizeof want, bytes); human_bytes(free_, sizeof free_, avail); human_bytes(keep, sizeof keep, reserve);
    snprintf(diag, diag_len,
             "dma_alloc: refused %s: %s of memory is free and CUDA++ keeps %s free (CUDAPP_MEM_RESERVE_MIB=%llu), because the CPU and "
             "GPU share it and running it out freezes the machine. Free buffers or use a smaller batch.",
             want, free_, keep, reserve_mib);
    return OMEGA_GPU_PLAT_REFUSED_LIMIT;
}
