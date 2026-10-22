/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L
#include "cudapp_host_run.h"

#include "sha256.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

int cudapp_host_vecadd(uint32_t n, int mutate_index, double *seconds, uint8_t out_sha[32]) {
    size_t bytes = (size_t)n * sizeof(float);
    float *a = aligned_alloc(64, bytes);
    float *b = aligned_alloc(64, bytes);
    float *c = aligned_alloc(64, bytes);
    if (!a || !b || !c) {
        free(a); free(b); free(c);
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        a[i] = (float)i * 0.25f;
        b[i] = 1.0f / (float)(1 + i % 7);
    }
    double t0 = now_s();
    for (uint32_t i = 0; i < n; i++) c[i] = a[i] + b[i];
    double t1 = now_s();
    if (mutate_index >= 0 && (uint32_t)mutate_index < n) c[mutate_index] = 0.0f;
    if (seconds) *seconds = t1 - t0;
    if (out_sha) sha256_hash((const uint8_t *)c, bytes, out_sha);

    uint32_t wrong = 0;
    for (uint32_t i = 0; i < n; i++) {
        float want = a[i] + b[i];
        if (memcmp(&want, &c[i], sizeof want) != 0) wrong++;
    }
    free(a); free(b); free(c);
    return wrong ? 1 : 0;
}

double cudapp_host_copy_probe(size_t bytes, int reps, double *gib_s) {
    if (reps < 1) reps = 1;
    char *src = aligned_alloc(64, bytes);
    char *dst = aligned_alloc(64, bytes);
    if (!src || !dst) {
        free(src); free(dst);
        if (gib_s) *gib_s = 0;
        return -1;
    }
    memset(src, 1, bytes);
    double best = 0;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
        memcpy(dst, src, bytes);
        double dt = now_s() - t0;
        if (r == 0 || dt < best) best = dt;
    }
    /* the compiler must not delete the copies */
    if (dst[0] == 2 && dst[bytes - 1] == 3) best = 0;
    free(src); free(dst);
    if (gib_s) *gib_s = best > 0 ? ((double)bytes / best) / (1024.0 * 1024.0 * 1024.0) : 0;
    return best;
}
