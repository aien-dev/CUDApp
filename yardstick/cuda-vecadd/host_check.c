/* SPDX-License-Identifier: MIT */
/* yardstick/cuda-vecadd/host_check.c: what both GPU programs must produce, computed on the CPU.
 *
 * Same inputs, same single-precision add, same 4194304 output bytes as examples/hello_gpu and as
 * vecadd.cu. Its output hash must equal hello_gpu's --sim hash, which is the hash the chip run
 * printed too. That makes the equivalence claim checkable without a chip: when the two GPU runs
 * happen, the only thing left to compare is time.
 *
 * usage: host_check [dump.bin]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define N_TOTAL (1u << 20)

int main(int argc, char **argv) {
    size_t bytes = (size_t)N_TOTAL * sizeof(float);
    float *a = (float *)malloc(bytes), *b = (float *)malloc(bytes);
    uint32_t *c = (uint32_t *)malloc(bytes);
    if (!a || !b || !c) { printf("FAIL: host memory\n"); return 1; }
    for (uint32_t i = 0; i < N_TOTAL; i++) { a[i] = (float)i * 0.25f; b[i] = 1.0f / (float)(1 + i % 7); }
    for (uint32_t i = 0; i < N_TOTAL; i++) { float s = a[i] + b[i]; memcpy(&c[i], &s, sizeof s); }
    printf("host reference: %u elements, c[i] = a[i] + b[i] on the CPU, same inputs as hello_gpu and vecadd_cuda\n", N_TOTAL);
    if (argc > 1) {
        FILE *f = fopen(argv[1], "wb");
        if (!f) { printf("FAIL: cannot write %s\n", argv[1]); return 1; }
        if (fwrite(c, 1, bytes, f) != bytes) { printf("FAIL: short write\n"); fclose(f); return 1; }
        fclose(f);
        printf("written to %s (%zu bytes)\n", argv[1], bytes);
    }
    return 0;
}
