/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* gpu_l2_reuse_test: does an FP8 GEMV that re-reads weights another launch just read run faster? (OWN-30B W-L2d, 2026-10-14)
 *
 * Question behind it: a decode token has ~120 us per layer where DRAM is idle (attention at pos 300 ~73 us latency-bound,
 * router ~25 us, glue ~18 us; W-L2 timeline). If the GB10 L2 keeps a GEMV's weights between two launches, an independent
 * prefetch launch overlapped with those latency-bound launches (no barrier between them: no data dependency) could pull the
 * NEXT GEMV's weights into L2, and that GEMV would then run from L2. This test measures only the precondition: the same
 * tensor read twice in one list, [GEMV(T) -> y][WAIT_FOR_IDLE][GEMV(T) -> y], GPU timestamps around each launch.
 * One variable per pass: the weight memory policy (GPU-uncached = every session buffer today, vs GPU-cached,
 * omega_gpu_fp8_set_weight_cache). Sizes n x 2048 for n = 1024 .. 16384 (2 .. 33.5 MB) bracket the unknown L2 size.
 * Barrier WFI_ONLY (no shader-cache invalidate; the chip study r1 showed the form does not change the time).
 * Every rep: y poisoned first, y bit-identical to the one-launch-at-a-time reference.
 * Negative control: the uncached pass (no reuse is possible through an uncached mapping, nvos.h:1117) must show warm ~= cold.
 * Chip only: caller holds the GB10 lock. Output "L2 cached=.. n=.. MB=.. cold_us=.. warm_us=.. ratio=..", --out receipt JSON. */
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_cmdlist.h"
#include "omega_gpu_session.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_failed;
#define EV_N 32768u
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } else { printf("ok: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static uint64_t median(uint64_t *v, uint32_t n) { qsort(v, n, sizeof *v, cmp_u64); return v[n / 2]; }

int main(int argc, char **argv) {
    uint32_t reps = 10, k = 2048; const char *out_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else { printf("usage: %s [--reps R] [--out receipt.json]\n", argv[0]); return 2; }
    }
    if (reps == 0 || reps > 64) reps = 10;
    static const uint32_t ns[] = { 1024, 2048, 4096, 8192, 12288, 16384 };
    const uint32_t nmax = 16384;
    omega_gpu_fp8_set_depth(1, 0);
    (void)omega_gpu_session_set_spin_us(2000);
    uint8_t *w = malloc((size_t)nmax * k); uint16_t *sc = malloc((size_t)(nmax / 128) * (k / 128) * 2); float *x = malloc((size_t)k * 4), *ref = malloc((size_t)nmax * 4);
    uint32_t st = 2026;
    for (size_t i = 0; i < (size_t)nmax * k; i++) { uint8_t b = (uint8_t)(lcg(&st) >> 24); w[i] = (b & 0x7F) == 0x7F ? 0 : b; }
    for (size_t i = 0; i < (size_t)(nmax / 128) * (k / 128); i++) sc[i] = 0x3F80;
    for (uint32_t i = 0; i < k; i++) x[i] = (float)(lcg(&st) >> 8) / 16777216.0f * 2.0f - 1.0f;

    OmegaGpuCmdList *cl = NULL;
    if (omega_gpu_cmdlist_create(8, &cl) != 0) { printf("cmdlist create: %s\n", omega_gpu_session_last_error()); return 1; }
    OmegaGpuMem tab, xd, yd, ye;
    omega_gpu_session_lock();
    int arc = omega_gpu_session_alloc(32u, &tab) | omega_gpu_session_alloc((size_t)k * 4u, &xd) | omega_gpu_session_alloc((size_t)nmax * 4u, &yd) | omega_gpu_session_alloc((size_t)EV_N * 4u, &ye);
    omega_gpu_session_unlock();
    if (arc != 0) { printf("alloc: %s\n", omega_gpu_session_last_error()); return 1; }
    omega_fp8_pack_x_f32(x, k, xd.cpu);
    if (omega_gpu_session_sync_to_gpu(&xd) != 0) return 1;

    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out) fprintf(out, "{\"test\":\"OMEGA_GPU_L2_REUSE\",\"k\":%u,\"reps\":%u,\"points\":[", k, reps);
    int first_pt = 1; double ratio_unc_max = 0, ratio_cached_min = 1e9;
    /* evictor: a GPU-cached EV_N x k tensor (67 MB) read first in every list, so "cold" is cold even if L2 keeps lines across submits */
    OmegaGpuFp8Tensor *ev = NULL;
    { uint8_t *we = malloc((size_t)EV_N * k); uint16_t *se = malloc((size_t)(EV_N / 128) * (k / 128) * 2);
      for (size_t i = 0; i < (size_t)EV_N * k; i++) { uint8_t b = (uint8_t)(lcg(&st) >> 24); we[i] = (b & 0x7F) == 0x7F ? 0 : b; }
      for (size_t i = 0; i < (size_t)(EV_N / 128) * (k / 128); i++) se[i] = 0x3F80;
      omega_gpu_fp8_set_weight_cache(1);
      int erc = omega_gpu_fp8_tensor_upload(1, EV_N, k, we, se, &ev);
      free(we); free(se);
      float *yev = malloc((size_t)EV_N * 4); if (erc == OMEGA_GPU_FP8_OK) erc = omega_gpu_fp8_gemv_f32(ev, 0, x, yev, NULL); free(yev); /* kernel built + warm */
      uint64_t *tw = tab.cpu;
      if (erc != OMEGA_GPU_FP8_OK || omega_gpu_fp8_tensor_entry(ev, 0, &tw[2], &tw[3]) != 0) { printf("evictor upload: %s\n", omega_gpu_fp8_last_error()); return 1; } }
    for (int pass = 0; pass < 3; pass++) { const int cached = pass > 0, evict = pass != 1; /* (uncached, evict) (cached, no evict) (cached, evict) */
        omega_gpu_fp8_set_weight_cache(cached);
        for (size_t ni = 0; ni < sizeof ns / sizeof ns[0]; ni++) {
            const uint32_t n = ns[ni]; const uint64_t bytes = (uint64_t)n * k + (uint64_t)(n / 128) * (k / 128) * 4u;
            OmegaGpuFp8Tensor *t = NULL;
            int rc = omega_gpu_fp8_tensor_upload(1, n, k, w, sc, &t);
            if (rc == OMEGA_GPU_FP8_OK) rc = omega_gpu_fp8_gemv_f32(t, 0, x, ref, NULL);
            if (rc != OMEGA_GPU_FP8_OK) { printf("upload/reference n=%u cached=%d: %s (%s)\n", n, cached, omega_gpu_fp8_rc_name(rc), omega_gpu_fp8_last_error()); g_failed++; omega_gpu_fp8_tensor_free(t); break; }
            uint64_t *tabw = tab.cpu;
            if (omega_gpu_fp8_tensor_entry(t, 0, &tabw[0], &tabw[1]) != 0 || omega_gpu_session_sync_to_gpu(&tab) != 0) { g_failed++; omega_gpu_fp8_tensor_free(t); break; }
            uint64_t colds[64], warms[64]; int identical = 1, ok = 1;
            for (uint32_t r = 0; r < reps && ok; r++) {
                float *y = yd.cpu; for (uint32_t i = 0; i < n; i++) y[i] = -999.0f;
                if (omega_gpu_session_sync_to_gpu(&yd) != 0) { ok = 0; break; }
                omega_gpu_session_lock();
                omega_gpu_cmdlist_reset(cl);
                if (evict && omega_gpu_fp8_gemv_grouped_record(cl, ev, 1, tab.va + 16u, xd.va, 1, ye.va) != 0) { printf("record evictor: %s\n", omega_gpu_fp8_last_error()); ok = 0; }
                for (int j = 0; j < 2 && ok; j++)
                    if (omega_gpu_fp8_gemv_grouped_record(cl, t, 1, tab.va, xd.va, 1, yd.va) != 0) { printf("record: %s\n", omega_gpu_fp8_last_error()); ok = 0; }
                OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.mode = OMEGA_GPU_CMDLIST_PCAS; o.barrier = OMEGA_GPU_CMDLIST_BARRIER_WFI_ONLY; o.stamps = 1;
                OmegaGpuCmdListStats s; memset(&s, 0, sizeof s);
                uint64_t sts[4]; int cnt = -1; const int want = evict ? 4 : 3;
                if (ok && omega_gpu_cmdlist_submit(cl, &o, &s) != 0) { printf("submit: %s\n", omega_gpu_session_last_error()); ok = 0; }
                if (ok) cnt = omega_gpu_cmdlist_stamps(cl, sts, 4);
                omega_gpu_session_unlock();
                if (!ok || cnt != want) { ok = 0; break; }
                colds[r] = sts[want - 2] - sts[want - 3]; warms[r] = sts[want - 1] - sts[want - 2];
                if (omega_gpu_session_sync_to_cpu(&yd) != 0) { ok = 0; break; }
                if (memcmp(y, ref, (size_t)n * 4) != 0) identical = 0;
            }
            CHECK(ok && identical, "cached=%d evict=%d n=%u: %u reps ran, y bit-identical to the reference on every rep", cached, evict, n, reps);
            if (ok) {
                const double c = median(colds, reps) / 1e3, wm = median(warms, reps) / 1e3, ratio = wm / c;
                printf("L2 cached=%d evict=%d n=%-5u MB=%.1f cold_us=%.1f warm_us=%.1f ratio=%.3f cold_GB/s=%.1f warm_GB/s=%.1f\n", cached, evict, n, bytes / 1e6, c, wm, ratio, bytes / (c * 1e3), bytes / (wm * 1e3));
                if (cached && evict) { if (ratio < ratio_cached_min) ratio_cached_min = ratio; } else if (!cached && ratio > ratio_unc_max) ratio_unc_max = ratio;
                if (out) { fprintf(out, "%s{\"cached\":%d,\"evict\":%d,\"n\":%u,\"bytes\":%" PRIu64 ",\"cold_ns\":%" PRIu64 ",\"warm_ns\":%" PRIu64 ",\"identical\":%d}", first_pt ? "" : ",", cached, evict, n, bytes, median(colds, reps), median(warms, reps), identical); first_pt = 0; }
            }
            omega_gpu_fp8_tensor_free(t);
            if (omega_gpu_session_is_blocked()) { printf("session latched: stopping\n"); g_failed++; break; }
        }
    }
    omega_gpu_fp8_set_weight_cache(0);
    printf("SUMMARY uncached max warm/cold %.3f, cached min warm/cold %.3f\n", ratio_unc_max, ratio_cached_min);
    if (out) { fprintf(out, "],\"uncached_ratio_max\":%.4f,\"cached_ratio_min\":%.4f,\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", ratio_unc_max, ratio_cached_min, g_checks, g_failed, g_failed ? "FAIL" : "PASS"); fclose(out); }
    printf("gpu_l2_reuse_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    omega_gpu_fp8_tensor_free(ev);
    omega_gpu_cmdlist_free(cl);
    free(w); free(sc); free(x); free(ref);
    return g_failed ? 1 : 0;
}
