/* SPDX-License-Identifier: MIT */
/* yardstick/cuda-vecadd: the CUDA half of the head-to-head measurement.
 *
 * What it is: the same job examples/hello_gpu does, on the same chip, in the same measurement window.
 * c[i] = a[i] + b[i] over 1048576 floats, the same inputs (a[i] = i * 0.25f, b[i] = 1.0f / (1 + i % 7)),
 * the same eight launches of 131072 elements, 128 threads per block and 1024 blocks per launch, and the
 * same check (every output compared against the CPU's own a + b, and every output hash printed).
 *
 * Why it is here: CUDA++ is a GPU stack of its own, and the honest way to say what that costs today is
 * to run CUDA's own stack on the same work and print both numbers side by side. Nothing in CUDA++ links
 * or calls this program. It is a yardstick, not a dependency, and the repository's Makefile does not
 * build it. Build and run it with yardstick/cuda-vecadd/run.sh inside a spark-window.
 *
 * Timing, mirrored from hello_gpu: hello_gpu times each launch on the GPU (the session returns the
 * elapsed GPU time for that launch) and sums the eight. The same is done here with a CUDA event pair
 * per launch plus a synchronise, so both sides wait for each launch to finish. Two more CUDA modes are
 * printed because CUDA can do them: eight launches queued on one stream and waited once (batched), and
 * an empty kernel per launch (what a launch costs with no work behind it).
 *
 * usage: vecadd_cuda [dump.bin]   the dump is the raw 4194304 output bytes, for sha256sum
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define N_TOTAL            (1u << 20)
#define THREADS            128u
#define MAX_CTAS           1024u
#define CHUNK              (THREADS * MAX_CTAS)   /* 131072 elements per launch, so 8 launches */
#define POISON             0xFFC0FFEEu
#define READ_CEILING_GBPS  195.1                  /* GB10 global reads, measured by CUDA++ probes */
#define PASSES             6                      /* warm passes after the cold one */

__global__ void vecadd(const float *a, const float *b, float *c, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void empty_kernel(void) { }

static double now_us(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec / 1e3;
}

static void die(const char *what, cudaError_t e) {
    printf("FAIL: %s: %s\n", what, cudaGetErrorString(e));
    exit(2);
}

/* one pass the way hello_gpu runs it: launch, wait for that launch, next. Returns the summed GPU time. */
static double pass_per_launch(const float *da, const float *db, float *dc, double *wall_us) {
    double sum_us = 0, t0 = now_us();
    for (unsigned base = 0; base < N_TOTAL; base += CHUNK) {
        unsigned n = N_TOTAL - base < CHUNK ? N_TOTAL - base : CHUNK;
        unsigned grid = (n + THREADS - 1) / THREADS;
        cudaEvent_t s, e;
        if (cudaEventCreate(&s) != cudaSuccess || cudaEventCreate(&e) != cudaSuccess)
            die("cudaEventCreate", cudaGetLastError());
        cudaEventRecord(s, 0);
        vecadd<<<grid, THREADS>>>(da + base, db + base, dc + base, n);
        cudaEventRecord(e, 0);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) die("vecadd launch", err);
        float ms = 0;
        cudaEventElapsedTime(&ms, s, e);
        sum_us += (double)ms * 1e3;
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }
    *wall_us = now_us() - t0;
    return sum_us;
}

/* eight launches queued on one stream and waited once: the least CUDA can spend on submission here. */
static double pass_batched(const float *da, const float *db, float *dc, double *wall_us) {
    double t0 = now_us();
    cudaEvent_t s, e;
    if (cudaEventCreate(&s) != cudaSuccess || cudaEventCreate(&e) != cudaSuccess)
        die("cudaEventCreate", cudaGetLastError());
    cudaEventRecord(s, 0);
    for (unsigned base = 0; base < N_TOTAL; base += CHUNK) {
        unsigned n = N_TOTAL - base < CHUNK ? N_TOTAL - base : CHUNK;
        vecadd<<<((n + THREADS - 1) / THREADS), THREADS>>>(da + base, db + base, dc + base, n);
    }
    cudaEventRecord(e, 0);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) die("vecadd batched", err);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    *wall_us = now_us() - t0;
    return (double)ms * 1e3;
}

/* eight empty launches, each waited for: launch cost with no work behind it */
static double pass_empty(void) {
    double sum_us = 0;
    for (int i = 0; i < 8; i++) {
        cudaEvent_t s, e;
        if (cudaEventCreate(&s) != cudaSuccess || cudaEventCreate(&e) != cudaSuccess)
            die("cudaEventCreate", cudaGetLastError());
        cudaEventRecord(s, 0);
        empty_kernel<<<MAX_CTAS, THREADS>>>();
        cudaEventRecord(e, 0);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) die("empty launch", err);
        float ms = 0;
        cudaEventElapsedTime(&ms, s, e);
        sum_us += (double)ms * 1e3;
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }
    return sum_us;
}

static void print_work(double measured_us) {
    double read = 8.0 * N_TOTAL, written = 4.0 * N_TOTAL;
    double floor_us = (read + written) / (READ_CEILING_GBPS * 1e3);
    printf("work: %llu bytes moved (%llu read, %llu written), %u floating-point operations\n",
           (unsigned long long)(read + written), (unsigned long long)read, (unsigned long long)written, N_TOTAL);
    printf("      at the GB10's measured %.1f GB/s read ceiling this needs at least %.1f us; CUDA took %.1f us (%.0f%% of the ceiling, %.1f GB/s)\n",
           READ_CEILING_GBPS, floor_us, measured_us, 100.0 * floor_us / measured_us, (read + written) / measured_us / 1e3);
}

static int check(const float *fa, const float *fb, const uint32_t *wc) {
    uint32_t wrong = 0, unwritten = 0, first_bad = 0;
    for (uint32_t i = 0; i < N_TOTAL; i++) {
        float want = fa[i] + fb[i];
        if (wc[i] == POISON) unwritten++;
        if (memcmp(&want, &wc[i], sizeof want) != 0) { if (!wrong++) first_bad = i; }
    }
    printf("checked %u of %u elements: %u wrong, %u never written\n", N_TOTAL, N_TOTAL, wrong, unwritten);
    if (wrong) {
        float want = fa[first_bad] + fb[first_bad], got;
        memcpy(&got, &wc[first_bad], sizeof got);
        printf("first mismatch: c[%u] = %g, expected %g\n", first_bad, got, want);
    }
    return wrong ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *dump = argc > 1 ? argv[1] : NULL;
    size_t bytes = (size_t)N_TOTAL * sizeof(float);
    float *ha = (float *)malloc(bytes), *hb = (float *)malloc(bytes);
    uint32_t *hc = (uint32_t *)malloc(bytes);
    float *da = NULL, *db = NULL, *dc = NULL;
    if (!ha || !hb || !hc) { printf("FAIL: host memory\n"); return 1; }
    for (uint32_t i = 0; i < N_TOTAL; i++) {
        ha[i] = (float)i * 0.25f;
        hb[i] = 1.0f / (float)(1 + i % 7);
        hc[i] = POISON;
    }

    int dev = 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) die("cudaGetDeviceProperties", cudaGetLastError());
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    printf("cuda yardstick: c = a + b over %u floats, built by nvcc, run on CUDA's own stack\n", N_TOTAL);
    printf("device: %s, compute capability %d.%d, %d multiprocessors, driver API %d.%d, runtime API %d.%d\n",
           p.name, p.major, p.minor, p.multiProcessorCount, drv / 1000, (drv % 1000) / 10, rt / 1000, (rt % 1000) / 10);
    printf("launches: %u of %u elements, %u threads per block, %u blocks per launch\n",
           N_TOTAL / CHUNK, CHUNK, THREADS, MAX_CTAS);

    if (cudaMalloc((void **)&da, bytes) != cudaSuccess ||
        cudaMalloc((void **)&db, bytes) != cudaSuccess ||
        cudaMalloc((void **)&dc, bytes) != cudaSuccess) die("cudaMalloc", cudaGetLastError());
    if (cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dc, hc, bytes, cudaMemcpyHostToDevice) != cudaSuccess) die("cudaMemcpy to device", cudaGetLastError());

    /* pass 1 is the cold one, matching hello_gpu's single run: no launch of this kernel has happened yet */
    double wall1 = 0, cold = pass_per_launch(da, db, dc, &wall1);
    printf("\ncold pass (what hello_gpu reports, one run from a fresh process)\n");
    printf("  launch by launch, each waited for: %.1f us GPU, %.1f us wall for the eight launches\n", cold, wall1);
    print_work(cold);

    /* warm passes */
    double wall = 0, sum = 0, best = 0, worst = 0;
    for (int i = 0; i < PASSES; i++) {
        double us = pass_per_launch(da, db, dc, &wall);
        sum += us;
        if (i == 0 || us < best) best = us;
        if (i == 0 || us > worst) worst = us;
    }
    printf("\nwarm passes (%d runs of the same eight launches)\n", PASSES);
    printf("  launch by launch, each waited for: best %.1f us, worst %.1f us, mean %.1f us GPU\n",
           best, worst, sum / PASSES);

    double bsum = 0, bbest = 0;
    for (int i = 0; i < 3; i++) {
        double us = pass_batched(da, db, dc, &wall);
        bsum += us;
        if (i == 0 || us < bbest) bbest = us;
    }
    printf("  eight launches queued on one stream, waited once: best %.1f us, mean %.1f us GPU\n", bbest, bsum / 3);
    print_work(bbest);

    double e = pass_empty();
    printf("\nlaunch cost: eight empty launches, each waited for: %.1f us total, %.1f us per launch\n", e, e / 8);

    if (cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) die("cudaMemcpy to host", cudaGetLastError());
    int rc = check(ha, hb, hc);
    if (dump) {
        FILE *f = fopen(dump, "wb");
        if (!f) { printf("FAIL: cannot write %s\n", dump); return 1; }
        if (fwrite(hc, 1, bytes, f) != bytes) { printf("FAIL: short write to %s\n", dump); fclose(f); return 1; }
        fclose(f);
        printf("output written to %s (%zu bytes)\n", dump, bytes);
    }
    printf("%s\n", rc ? "FAIL" : "PASS");
    cudaFree(da); cudaFree(db); cudaFree(dc);
    free(ha); free(hb); free(hc);
    return rc;
}
