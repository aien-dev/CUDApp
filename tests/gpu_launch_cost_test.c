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

/* gpu_launch_cost_test: what does one FP8 GEMV launch cost inside a command list beyond its bytes? (OWN-30B W-L2c, 2026-10-14)
 *
 * The same weight bytes (n x k e4m3, default 16384 x 2048 = 33.5 MB) are split into m = 1, 2, 4, 8, 16 launches (m tensors of
 * n/m rows, independent: distinct rows, distinct y), recorded into ONE command list and timed with the GPU timestamps
 * (omega_gpu_cmdlist stamps). T(m) - T(1) over (m - 1) is the per-launch fixed cost with the WAIT_FOR_IDLE barrier between
 * launches. Variant (one variable): the same lists with OMEGA_GPU_CMDLIST_BARRIER_NONE (legal here: the launches share no
 * data), which tells whether the barrier or the kernel's own ramp/tail is the cost. Every rep checks y bit-identical to the
 * session's one-launch-at-a-time result for every tensor (poisoned first). Negative control: a dropped launch (skip_launch)
 * must be named by submit.
 * Chip only (no host mode): caller holds the GB10 lock.  Output: lines "COST m=.. barrier=.. span_us=.. per_launch_us=.. GB/s=.."
 * and the fixed-cost lines; --out writes the receipt JSON. */
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_cmdlist.h"
#include "omega_gpu_session.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_failed;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } else { printf("ok: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static uint64_t median(uint64_t *v, uint32_t n) { qsort(v, n, sizeof *v, cmp_u64); return v[n / 2]; }

#define MAX_M 16
typedef struct { OmegaGpuFp8Tensor *t[MAX_M]; float *ref; uint32_t m, rows; } Split;

static int make_split(Split *s, uint32_t m, uint32_t n, uint32_t k, const uint8_t *w, const uint16_t *sc, const float *x) {
    memset(s, 0, sizeof *s);
    s->m = m; s->rows = n / m; s->ref = malloc((size_t)n * 4);
    for (uint32_t i = 0; i < m; i++) {
        int rc = omega_gpu_fp8_tensor_upload(1, s->rows, k, w + (size_t)i * s->rows * k, sc + (size_t)i * (s->rows / 128) * (k / 128), &s->t[i]);
        if (rc != OMEGA_GPU_FP8_OK) { printf("upload m=%u i=%u: %s (%s)\n", m, i, omega_gpu_fp8_rc_name(rc), omega_gpu_fp8_last_error()); return -1; }
        rc = omega_gpu_fp8_gemv_f32(s->t[i], 0, x, s->ref + (size_t)i * s->rows, NULL); /* warm-up (kernel cached) + the reference */
        if (rc != OMEGA_GPU_FP8_OK) { printf("reference m=%u i=%u: %s (%s)\n", m, i, omega_gpu_fp8_rc_name(rc), omega_gpu_fp8_last_error()); return -1; }
    }
    return 0;
}
static void free_split(Split *s) { for (uint32_t i = 0; i < s->m; i++) omega_gpu_fp8_tensor_free(s->t[i]); free(s->ref); }

/* one list of m launches, submitted reps times; medians of the stamp span and of the host-seen elapsed; y checked every rep */
static int time_split(Split *s, OmegaGpuCmdList *cl, OmegaGpuMem *tab, OmegaGpuMem *xd, OmegaGpuMem *yd, OmegaGpuCmdListBarrier bar,
                      uint32_t reps, uint64_t *span_med, uint64_t *elapsed_med, uint64_t *first_med, int *identical) {
    const uint32_t m = s->m, n = s->rows * m;
    uint64_t *spans = malloc(reps * 8), *els = malloc(reps * 8), *firsts = malloc(reps * 8);
    uint64_t *tabw = tab->cpu;
    for (uint32_t i = 0; i < m; i++) if (omega_gpu_fp8_tensor_entry(s->t[i], 0, &tabw[2 * i], &tabw[2 * i + 1]) != 0) return -1;
    if (omega_gpu_session_sync_to_gpu(tab) != 0) return -1;
    *identical = 1;
    int rc = 0;
    for (uint32_t r = 0; r < reps && rc == 0; r++) {
        float *y = yd->cpu; for (uint32_t i = 0; i < n; i++) y[i] = -999.0f;
        if (omega_gpu_session_sync_to_gpu(yd) != 0) { rc = -1; break; }
        omega_gpu_session_lock();
        omega_gpu_cmdlist_reset(cl);
        for (uint32_t i = 0; i < m && rc == 0; i++)
            if (omega_gpu_fp8_gemv_grouped_record(cl, s->t[i], 1, tab->va + 16u * i, xd->va, 1, yd->va + (uint64_t)i * s->rows * 4u) != 0) { printf("record m=%u i=%u: %s\n", m, i, omega_gpu_fp8_last_error()); rc = -1; }
        OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o);
        o.mode = OMEGA_GPU_CMDLIST_PCAS; o.barrier = bar; o.stamps = 1;
        OmegaGpuCmdListStats st; memset(&st, 0, sizeof st);
        if (rc == 0 && omega_gpu_cmdlist_submit(cl, &o, &st) != 0) { printf("submit m=%u barrier=%s: %s\n", m, omega_gpu_cmdlist_barrier_name(bar), omega_gpu_session_last_error()); rc = -1; }
        uint64_t stamps[MAX_M + 1]; const int none = bar == OMEGA_GPU_CMDLIST_BARRIER_NONE; int cnt = rc == 0 && !none ? omega_gpu_cmdlist_stamps(cl, stamps, MAX_M + 1) : -1; /* barrier none: only the first and last stamp exist, so the GPU span is not readable; elapsed (host wall, ~5 us above span) is the comparable figure */
        omega_gpu_session_unlock();
        if (rc == 0 && !none && cnt != (int)m + 1) { printf("stamps m=%u: %d\n", m, cnt); rc = -1; }
        if (rc != 0) break;
        els[r] = st.elapsed_ns; spans[r] = none ? 0 : stamps[m] - stamps[0]; firsts[r] = none ? 0 : stamps[1] - stamps[0];
        if (omega_gpu_session_sync_to_cpu(yd) != 0) { rc = -1; break; }
        if (memcmp(y, s->ref, (size_t)n * 4) != 0) *identical = 0;
    }
    if (rc == 0) { *span_med = median(spans, reps); *elapsed_med = median(els, reps); *first_med = median(firsts, reps); }
    free(spans); free(els); free(firsts);
    return rc;
}

int main(int argc, char **argv) {
    uint32_t reps = 20, n = 16384, k = 2048; const char *out_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) n = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) k = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else { printf("usage: %s [--reps N] [--n ROWS] [--k K] [--out receipt.json]\n", argv[0]); return 2; }
    }
    if (n % (128u * MAX_M) || k % 512u) { printf("n must be a multiple of %u and k of 512\n", 128u * MAX_M); return 2; }
    omega_gpu_fp8_set_depth(1, 0); /* one kernel (the step-by-step production kernel) for every split: the row count must not change the kernel */
    (void)omega_gpu_session_set_spin_us(2000);
    /* data: random e4m3 bytes without the NaN codes, scales 1.0, x in [-1, 1) */
    uint8_t *w = malloc((size_t)n * k); uint16_t *sc = malloc((size_t)(n / 128) * (k / 128) * 2); float *x = malloc((size_t)k * 4);
    uint32_t st = 2026;
    for (size_t i = 0; i < (size_t)n * k; i++) { uint8_t b = (uint8_t)(lcg(&st) >> 24); w[i] = (b & 0x7F) == 0x7F ? 0 : b; }
    for (size_t i = 0; i < (size_t)(n / 128) * (k / 128); i++) sc[i] = 0x3F80; /* bf16 1.0 */
    for (uint32_t i = 0; i < k; i++) x[i] = (float)(lcg(&st) >> 8) / 16777216.0f * 2.0f - 1.0f;
    const uint64_t bytes = (uint64_t)n * k + (uint64_t)(n / 128) * (k / 128) * 4u;

    OmegaGpuCmdList *cl = NULL;
    if (omega_gpu_cmdlist_create(64, &cl) != 0) { printf("cmdlist create: %s\n", omega_gpu_session_last_error()); return 1; }
    OmegaGpuMem tab, xd, yd;
    omega_gpu_session_lock();
    int arc = omega_gpu_session_alloc(16u * MAX_M, &tab) | omega_gpu_session_alloc((size_t)k * 4u, &xd) | omega_gpu_session_alloc((size_t)n * 4u, &yd);
    omega_gpu_session_unlock();
    if (arc != 0) { printf("alloc: %s\n", omega_gpu_session_last_error()); return 1; }
    omega_fp8_pack_x_f32(x, k, xd.cpu);
    if (omega_gpu_session_sync_to_gpu(&xd) != 0) return 1;

    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out) fprintf(out, "{\"test\":\"OMEGA_GPU_LAUNCH_COST\",\"n\":%u,\"k\":%u,\"bytes\":%" PRIu64 ",\"reps\":%u,\"points\":[", n, k, bytes, reps);
    static const uint32_t ms[] = { 1, 2, 4, 8, 16 };
    double t1[8] = { 0 }; int first_pt = 1;
    static const OmegaGpuCmdListBarrier bars[] = { OMEGA_GPU_CMDLIST_BARRIER_WFI, OMEGA_GPU_CMDLIST_BARRIER_WFI_ONLY, OMEGA_GPU_CMDLIST_BARRIER_WFI_DATA, OMEGA_GPU_CMDLIST_BARRIER_FLUSH_DATA, OMEGA_GPU_CMDLIST_BARRIER_NONE, OMEGA_GPU_CMDLIST_BARRIER_WFI };
    for (unsigned bi = 0; bi < sizeof bars / sizeof bars[0]; bi++) { const OmegaGpuCmdListBarrier bar = bars[bi]; const char *bn = omega_gpu_cmdlist_barrier_name(bar); /* WFI first and last: drift control */
        for (size_t mi = 0; mi < sizeof ms / sizeof ms[0]; mi++) {
            Split s;
            if (make_split(&s, ms[mi], n, k, w, sc, x) != 0) { printf("split m=%u failed\n", ms[mi]); g_failed++; free_split(&s); break; }
            uint64_t span = 0, el = 0, first = 0; int identical = 0;
            int rc = time_split(&s, cl, &tab, &xd, &yd, bar, reps, &span, &el, &first, &identical);
            CHECK(rc == 0 && identical, "m=%u barrier=%s: %u reps ran, y bit-identical to the one-launch-at-a-time reference on every rep", ms[mi], bn, reps);
            if (rc == 0) {
                double span_us = span / 1e3, el_us = el / 1e3, per = el_us / ms[mi], gbs = (double)bytes / (double)el; /* elapsed-based: comparable across barrier modes */
                if (ms[mi] == 1) t1[bi] = el_us;
                double fixed = ms[mi] > 1 ? (el_us - t1[bi]) / (ms[mi] - 1) : 0.0;
                printf("COST m=%-2u barrier=%-14s span_us=%.1f per_launch_us=%.1f first_launch_us=%.1f elapsed_us=%.1f GB/s=%.1f fixed_per_extra_launch_us=%.2f\n",
                       ms[mi], bn, span_us, per, first / 1e3, el / 1e3, gbs, fixed);
                if (out) { fprintf(out, "%s{\"m\":%u,\"barrier\":\"%s\",\"span_ns\":%" PRIu64 ",\"first_launch_ns\":%" PRIu64 ",\"elapsed_ns\":%" PRIu64 ",\"gbps\":%.2f,\"fixed_per_extra_launch_us\":%.2f,\"identical\":%d}",
                                  first_pt ? "" : ",", ms[mi], bn, span, first, el, gbs, fixed, identical); first_pt = 0; }
            }
            free_split(&s);
            if (omega_gpu_session_is_blocked()) { printf("session latched: stopping\n"); break; }
        }
    }
    /* negative control: a dropped launch is named */
    {
        Split s; int rc = make_split(&s, 2, n, k, w, sc, x);
        if (rc == 0) {
            uint64_t *tabw = tab.cpu; for (uint32_t i = 0; i < 2; i++) omega_gpu_fp8_tensor_entry(s.t[i], 0, &tabw[2 * i], &tabw[2 * i + 1]);
            omega_gpu_session_sync_to_gpu(&tab);
            omega_gpu_session_lock();
            omega_gpu_cmdlist_reset(cl);
            for (uint32_t i = 0; i < 2; i++) omega_gpu_fp8_gemv_grouped_record(cl, s.t[i], 1, tab.va + 16u * i, xd.va, 1, yd.va + (uint64_t)i * s.rows * 4u);
            OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.mode = OMEGA_GPU_CMDLIST_PCAS; o.barrier = OMEGA_GPU_CMDLIST_BARRIER_WFI; o.skip_launch = 2;
            OmegaGpuCmdListStats st2; memset(&st2, 0, sizeof st2);
            int src = omega_gpu_cmdlist_submit(cl, &o, &st2);
            omega_gpu_session_unlock();
            CHECK(src != 0 && st2.first_missing == 1, "negative control: dropped launch 1 named by submit (rc %d, first_missing %d)", src, st2.first_missing);
        } else g_failed++;
        free_split(&s);
    }
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS"); fclose(out); }
    omega_gpu_session_lock(); omega_gpu_session_free(&tab); omega_gpu_session_free(&xd); omega_gpu_session_free(&yd); omega_gpu_session_unlock();
    omega_gpu_cmdlist_free(cl);
    omega_gpu_session_close();
    printf("gpu_launch_cost_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
