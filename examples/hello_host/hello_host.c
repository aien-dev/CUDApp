/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
/* hello_host: the hello program on the CPU of the machine you have.
 *
 * c[i] = a[i] + b[i] for 1048576 floats, the same inputs as examples/hello_gpu.
 * The code is CUDA++'s. It does not call a vendor GPU stack (no CUDA, no ROCm, no oneAPI, no Metal).
 *
 * This is host execution. It is not the GPU writing machine code, and it is not a speed claim
 * against CUDA. Real GPU machine code remains the GB10 path (make hello-gpu) until another chip
 * has a receipt.
 *
 * Exit 0: PASS. Exit 1: a wrong answer, or --mutate (the negative control, which must not pass).
 * Exit 2: the host could not allocate.
 */
#include "cudapp_host_run.h"
#include "cudapp_target.h"

#include <stdio.h>
#include <string.h>
#include <sys/utsname.h>

static const char *kind_text(CudappTargetKind k) {
    if (k == CUDAPP_KIND_HOST) return "host";
    if (k == CUDAPP_KIND_GPU_QUALIFIED) return "gpu qualified";
    return "gpu named, not qualified";
}

static void print_sha(const uint8_t d[32]) {
    for (int i = 0; i < 32; i++) printf("%02x", d[i]);
}

int main(int argc, char **argv) {
    int mutate = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mutate") == 0) mutate = 1;
    }
    struct utsname u;
    if (uname(&u) == 0)
        printf("machine: %s %s %s\n", u.sysname, u.release, u.machine);
    const CudappTarget *host = cudapp_host_target();
    printf("running on: %s (%s, %s)\n", host->id, host->name, host->isa);
    printf("note: %s\n", host->note);
    if (host->receipt) printf("receipt path: %s\n", host->receipt);

    int n = 0;
    const CudappTarget *all = cudapp_targets(&n);
    printf("targets:\n");
    for (int i = 0; i < n; i++) {
        printf("  %s  %s  [%s]", all[i].id, all[i].name, kind_text(all[i].kind));
        if (all[i].receipt) printf("  receipt %s", all[i].receipt);
        printf("\n");
    }

    double seconds = 0;
    uint8_t sha[32];
    int rc = cudapp_host_vecadd(CUDAPP_HOST_N, mutate ? 0 : -1, &seconds, sha);
    if (rc < 0) {
        printf("FAIL: host memory\n");
        return 2;
    }
    uint64_t moved = (uint64_t)CUDAPP_HOST_N * sizeof(float) * 3u;
    printf("work: %llu bytes moved (2 reads, 1 write), %u floating-point adds, %.3f ms\n",
           (unsigned long long)moved, CUDAPP_HOST_N, seconds * 1e3);
    printf("output sha256 ");
    print_sha(sha);
    printf("\n");

    double gib_s = 0;
    double copy_s = cudapp_host_copy_probe(32u << 20, 8, &gib_s);
    if (copy_s > 0)
        printf("host copy probe: best of 8 copies of 32 MiB took %.3f ms (%.2f GiB/s). This probe is not a chip ceiling.\n",
               copy_s * 1e3, gib_s);

    if (mutate) {
        if (rc == 0) {
            printf("FAIL: mutant was not caught\n");
            return 1;
        }
        printf("MUTANT caught (one element left wrong). This run must not be called PASS.\n");
        return 1;
    }
    if (rc != 0) {
        printf("FAIL\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
