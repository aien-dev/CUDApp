/* omega_gpu_scan_ref.c -- the host references of omega_gpu_scan.h: the frozen order, in plain C on the CPU.
 * Separate from the chip code so a CPU-only program can check them against the E1 reduction reference
 * (tests/gpu_scan_ref_check.c). Additions only, so no fused multiply-add can change a bit. */
#include "omega_gpu_scan.h"

#include <stdlib.h>
#include <string.h>

#define PAD_SUM 0x80000000u    /* -0.0 */
#define PAD_MINMAX 0x7fc00000u /* quiet NaN */


static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static float combine(OmegaGpuRedOp op, float a, float b) {
    if (op == OMEGA_GPU_RED_SUM) return a + b;
    return op == OMEGA_GPU_RED_MAX ? __builtin_fmaxf(a, b) : __builtin_fminf(a, b); /* AArch64 FMAXNM / FMINNM: NaN ignored */
}

int omega_gpu_reduce_reference(OmegaGpuRedOp op, const float *x, size_t n, float *out) {
    if ((unsigned)op > OMEGA_GPU_RED_MIN || !x || !out || n == 0) return -1;
    const float pad = u2f(op == OMEGA_GPU_RED_SUM ? PAD_SUM : PAD_MINMAX);
    float *cur = NULL;
    size_t len = n;
    const float *in = x;
    do {
        size_t nt = (len + 31) / 32;
        float *next = malloc(nt * sizeof *next);
        if (!next) { free(cur); return -1; }
        for (size_t j = 0; j < nt; j++) {
            float v[32];
            for (size_t i = 0; i < 32; i++) v[i] = j * 32 + i < len ? in[j * 32 + i] : pad;
            for (size_t d = 16; d >= 1; d /= 2)
                for (size_t i = 0; i + d < 32; i++) v[i] = combine(op, v[i], v[i + d]); /* ascending: v[i + d] is still old */
            next[j] = v[0];
        }
        free(cur);
        cur = next; in = next; len = nt;
    } while (len > 1);
    *out = cur[0];
    free(cur);
    return 0;
}

/* one scan level of len values: out (may alias in) and the tile totals */
static void ref_scan_level(const float *in, float *out, float *tot, size_t len) {
    for (size_t j = 0; j * 32 < len; j++) {
        float v[32];
        for (size_t i = 0; i < 32; i++) v[i] = j * 32 + i < len ? in[j * 32 + i] : u2f(PAD_SUM);
        for (size_t d = 1; d <= 16; d *= 2)
            for (size_t i = 31; i >= d; i--) v[i] = v[i - d] + v[i]; /* descending: v[i - d] is still old */
        for (size_t i = 0; i < 32 && j * 32 + i < len; i++) out[j * 32 + i] = v[i];
        tot[j] = v[31];
    }
}

static int ref_prefix(const float *in, float *out, size_t len) {
    size_t nt = (len + 31) / 32;
    float *tot = malloc(nt * sizeof *tot);
    if (!tot) return -1;
    ref_scan_level(in, out, tot, len);
    int rc = 0;
    if (nt > 1) {
        rc = ref_prefix(tot, tot, nt);
        for (size_t e = 32; rc == 0 && e < len; e++) out[e] = tot[e / 32 - 1] + out[e];
    }
    free(tot);
    return rc;
}

int omega_gpu_prefix_sum_reference(const float *x, size_t n, float *y) {
    if (!x || !y || n == 0) return -1;
    return ref_prefix(x, y, n);
}
