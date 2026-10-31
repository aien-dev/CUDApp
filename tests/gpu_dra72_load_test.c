/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* gpu_dra72_load_test: a timed, logged GPU FP8 GEMV load for the DRA-72 interference matrix (G2/G3, P-DRA-4).
 *
 * The decode-like DRAM stream: T tensors of n x k e4m3 (default 32 x 16384 x 2048 = 1.07 GB, far past L2) are read by
 * command lists of M FP8 GEMV launches (omega_gpu_fp8_gemv_grouped_record, PCAS, WFI barrier), rotating through the
 * tensors so every submission streams fresh bytes, for --seconds of wall time. Each submission is one log line with its
 * CLOCK_MONOTONIC start and host-seen elapsed (omega_gpu_cmdlist stats), so the judge can align it with the cpubw
 * windows (tools/dra72/cpubw.c, same clock) and compute GPU GB/s alone vs under CPU load.
 *
 * Checks: every launch's y is bit-identical to that tensor's one-launch-at-a-time omega_gpu_fp8_gemv_f32 result
 * (poisoned before every submission); a latched session stops the run and fails it.
 * Chip only (no host mode): the caller holds the GB10 window. Usage errors exit 2 before any device call.
 *
 * Output (stdout, or --out FILE mode 0600), integers only after the tag:
 *   LOAD 1 n=.. k=.. tensors=.. per_submit=.. bytes_per_submit=.. t0_mono_ns=.. t0_real_ns=..
 *   READY                       (stdout too, flushed: uploads done, timed loop starts)
 *   L <seq> <start_mono_ns> <elapsed_ns> <bytes>
 *   END <ok 0|1> <identical 0|1> <submits>
 * Exit 0 = every submission ran and matched, 1 = a check failed, 2 = usage. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_cmdlist.h"
#include "omega_gpu_session.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "test_out_file.h"

#define MAX_T 64u
#define MAX_M 16u

static uint64_t clk(clockid_t c) { struct timespec t; clock_gettime(c, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }

int main(int argc, char **argv) {
    uint32_t seconds = 60, tensors = 32, m = 8, n = 16384, k = 2048; const char *out_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seconds") == 0 && i + 1 < argc) seconds = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--tensors") == 0 && i + 1 < argc) tensors = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--per-submit") == 0 && i + 1 < argc) m = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) n = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) k = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else { printf("usage: %s [--seconds S] [--tensors T<=%u] [--per-submit M<=%u] [--n ROWS] [--k K] [--out FILE]\n", argv[0], MAX_T, MAX_M); return 2; }
    }
    if (seconds == 0 || seconds > 900 || tensors == 0 || tensors > MAX_T || m == 0 || m > MAX_M || m > tensors || n == 0 || n % 128u || k == 0 || k % 512u) {
        printf("refused: seconds 1..900, tensors 1..%u, per-submit 1..min(%u, tensors), n multiple of 128, k multiple of 512\n", MAX_T, MAX_M);
        return 2;
    }
    FILE *out = out_path ? test_out_open(out_path) : stdout;
    if (!out) { printf("refused: cannot create %s\n", out_path); return 2; }

    omega_gpu_fp8_set_depth(1, 0); /* the step-by-step production kernel, as gpu_launch_cost_test */
    (void)omega_gpu_session_set_spin_us(2000);
    uint8_t *w = malloc((size_t)n * k); uint16_t *sc = malloc((size_t)(n / 128) * (k / 128) * 2); float *x = malloc((size_t)k * 4);
    float *ref = malloc((size_t)tensors * n * 4);
    if (!w || !sc || !x || !ref) { printf("FAIL: host alloc\n"); return 1; }
    OmegaGpuFp8Tensor *t[MAX_T] = { 0 };
    uint32_t st = 2026;
    for (uint32_t i = 0; i < k; i++) x[i] = (float)(lcg(&st) >> 8) / 16777216.0f * 2.0f - 1.0f;
    for (size_t i = 0; i < (size_t)(n / 128) * (k / 128); i++) sc[i] = 0x3F80; /* bf16 1.0 */
    for (uint32_t ti = 0; ti < tensors; ti++) { /* distinct bytes per tensor: no NaN codes */
        for (size_t i = 0; i < (size_t)n * k; i++) { uint8_t b = (uint8_t)(lcg(&st) >> 24); w[i] = (b & 0x7F) == 0x7F ? 0 : b; }
        int rc = omega_gpu_fp8_tensor_upload(1, n, k, w, sc, &t[ti]);
        if (rc == OMEGA_GPU_FP8_OK) rc = omega_gpu_fp8_gemv_f32(t[ti], 0, x, ref + (size_t)ti * n, NULL);
        if (rc != OMEGA_GPU_FP8_OK) { printf("FAIL: tensor %u: %s (%s)\n", ti, omega_gpu_fp8_rc_name(rc), omega_gpu_fp8_last_error()); return 1; }
    }
    free(w);
    const uint64_t bytes_one = (uint64_t)n * k + (uint64_t)(n / 128) * (k / 128) * 4u, bytes_sub = bytes_one * m;

    OmegaGpuCmdList *cl = NULL;
    if (omega_gpu_cmdlist_create(64, &cl) != 0) { printf("FAIL: cmdlist create: %s\n", omega_gpu_session_last_error()); return 1; }
    OmegaGpuMem tab, xd, yd;
    omega_gpu_session_lock();
    int arc = omega_gpu_session_alloc(16u * MAX_T, &tab) | omega_gpu_session_alloc((size_t)k * 4u, &xd) | omega_gpu_session_alloc((size_t)m * n * 4u, &yd);
    omega_gpu_session_unlock();
    if (arc != 0) { printf("FAIL: alloc: %s\n", omega_gpu_session_last_error()); return 1; }
    omega_fp8_pack_x_f32(x, k, xd.cpu);
    uint64_t *tabw = tab.cpu;
    for (uint32_t ti = 0; ti < tensors; ti++) if (omega_gpu_fp8_tensor_entry(t[ti], 0, &tabw[2 * ti], &tabw[2 * ti + 1]) != 0) { printf("FAIL: entry %u\n", ti); return 1; }
    if (omega_gpu_session_sync_to_gpu(&xd) != 0 || omega_gpu_session_sync_to_gpu(&tab) != 0) { printf("FAIL: sync\n"); return 1; }

    const uint64_t t0m = clk(CLOCK_MONOTONIC), t0r = clk(CLOCK_REALTIME);
    fprintf(out, "LOAD 1 n=%u k=%u tensors=%u per_submit=%u bytes_per_submit=%" PRIu64 " t0_mono_ns=%" PRIu64 " t0_real_ns=%" PRIu64 "\n",
            n, k, tensors, m, bytes_sub, t0m, t0r);
    printf("READY\n"); fflush(stdout); if (out != stdout) fflush(out);
    const uint64_t end = t0m + (uint64_t)seconds * 1000000000ull;
    int ok = 1, identical = 1; uint32_t seq = 0, next = 0;
    float *y = yd.cpu;
    while (ok && clk(CLOCK_MONOTONIC) < end) {
        for (uint32_t i = 0; i < m * n; i++) y[i] = -999.0f;
        if (omega_gpu_session_sync_to_gpu(&yd) != 0) { printf("FAIL: y sync\n"); ok = 0; break; }
        uint32_t idx[MAX_M];
        omega_gpu_session_lock();
        omega_gpu_cmdlist_reset(cl);
        for (uint32_t j = 0; j < m && ok; j++) {
            idx[j] = next; next = (next + 1) % tensors;
            if (omega_gpu_fp8_gemv_grouped_record(cl, t[idx[j]], 1, tab.va + 16u * idx[j], xd.va, 1, yd.va + (uint64_t)j * n * 4u) != 0) {
                printf("FAIL: record %u: %s\n", j, omega_gpu_fp8_last_error()); ok = 0;
            }
        }
        OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.mode = OMEGA_GPU_CMDLIST_PCAS; o.barrier = OMEGA_GPU_CMDLIST_BARRIER_WFI;
        OmegaGpuCmdListStats sst; memset(&sst, 0, sizeof sst);
        const uint64_t ts = clk(CLOCK_MONOTONIC);
        if (ok && omega_gpu_cmdlist_submit(cl, &o, &sst) != 0) { printf("FAIL: submit %u: %s\n", seq, omega_gpu_session_last_error()); ok = 0; }
        omega_gpu_session_unlock();
        if (!ok) break;
        if (omega_gpu_session_sync_to_cpu(&yd) != 0) { printf("FAIL: y sync back\n"); ok = 0; break; }
        for (uint32_t j = 0; j < m; j++) if (memcmp(y + (size_t)j * n, ref + (size_t)idx[j] * n, (size_t)n * 4) != 0) identical = 0;
        fprintf(out, "L %u %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", seq, ts, sst.elapsed_ns, bytes_sub);
        seq++;
        if (omega_gpu_session_is_blocked()) { printf("FAIL: session latched\n"); ok = 0; }
    }
    if (!identical) { printf("FAIL: y differs from the one-launch reference\n"); ok = 0; }
    fprintf(out, "END %d %d %u\n", ok, identical, seq);
    if (out != stdout) fclose(out);
    omega_gpu_session_lock(); omega_gpu_session_free(&tab); omega_gpu_session_free(&xd); omega_gpu_session_free(&yd); omega_gpu_session_unlock();
    omega_gpu_cmdlist_free(cl);
    for (uint32_t ti = 0; ti < tensors; ti++) omega_gpu_fp8_tensor_free(t[ti]);
    omega_gpu_session_close();
    free(ref); free(sc); free(x);
    printf("gpu_dra72_load_test: submits=%u identical=%d: %s\n", seq, identical, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
