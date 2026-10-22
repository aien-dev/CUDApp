/* SPDX-License-Identifier: MIT */
/* Host checks for the target table and the host program.
 * A mutant that is not caught fails the test: a PASS that cannot be false checks nothing. */
#include "cudapp_host_run.h"
#include "cudapp_target.h"

#include <stdio.h>
#include <string.h>

static int g_fail;

static void expect(int cond, const char *what) {
    if (cond) {
        printf("  ok  %s\n", what);
        return;
    }
    printf("  FAIL %s\n", what);
    g_fail++;
}

int main(void) {
    const CudappTarget *gb10 = cudapp_gpu_target(CUDAPP_PCI_NVIDIA, CUDAPP_PCI_GB10);
    expect(gb10 && gb10->kind == CUDAPP_KIND_GPU_QUALIFIED && gb10->receipt, "GB10 is the qualified GPU");

    const CudappTarget *amd = cudapp_gpu_target(CUDAPP_PCI_AMD, 0x744c);
    expect(amd && strcmp(amd->id, "amd-gpu") == 0 && amd->kind == CUDAPP_KIND_GPU_NAMED && !amd->receipt,
           "AMD GPU is named and not qualified");

    const CudappTarget *intel = cudapp_gpu_target(CUDAPP_PCI_INTEL, 0xe20b);
    expect(intel && strcmp(intel->id, "intel-gpu") == 0 && intel->kind == CUDAPP_KIND_GPU_NAMED && !intel->receipt,
           "Intel GPU is named and not qualified");

    const CudappTarget *apple = cudapp_gpu_target(CUDAPP_PCI_APPLE, 0x0001);
    expect(apple && strcmp(apple->id, "apple-gpu") == 0 && !apple->receipt,
           "Apple GPU is named and not qualified");

    const CudappTarget *other = cudapp_gpu_target(CUDAPP_PCI_NVIDIA, 0x2b85);
    expect(other && strcmp(other->id, "nvidia-other") == 0 && !other->receipt,
           "another NVIDIA device is not the GB10");

    expect(cudapp_gpu_target(0x1234, 0x0001) == NULL, "an unknown vendor is not invented");

    const CudappTarget *host = cudapp_host_target();
    expect(host && host->kind == CUDAPP_KIND_HOST && host->receipt, "this binary has a host target");
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    expect(strcmp(host->id, "apple-m") == 0, "Apple M binary names apple-m");
#elif defined(__x86_64__)
    expect(strcmp(host->id, "amd64") == 0, "x86_64 binary names amd64");
#elif defined(__aarch64__)
    expect(strcmp(host->id, "aarch64-linux") == 0, "Linux ARM64 binary names aarch64-linux");
#endif

    uint8_t sha_a[32], sha_b[32];
    double sec = 0;
    int ok = cudapp_host_vecadd(4096, -1, &sec, sha_a);
    int again = cudapp_host_vecadd(4096, -1, &sec, sha_b);
    expect(ok == 0 && again == 0, "host vecadd matches");
    expect(memcmp(sha_a, sha_b, 32) == 0, "same inputs give the same output bits");
    int caught = cudapp_host_vecadd(4096, 3, &sec, sha_a);
    expect(caught == 1, "mutant (one wrong element) is caught");

    uint8_t hello[32];
    int hello_rc = cudapp_host_vecadd(CUDAPP_HOST_N, -1, &sec, hello);
    expect(hello_rc == 0 && sec >= 0, "hello length (1048576) matches");

    if (g_fail) {
        printf("FAIL %d\n", g_fail);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
