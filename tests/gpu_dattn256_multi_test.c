/*
 * gpu_dattn256_multi_test (chip): the multi-token decode attention (L369, omega_gpu_decode_attn256_multi_record) must be
 * T decode steps (omega_gpu_decode_attn256 at positions pos0 .. pos0 + T - 1) bit for bit: every output row and every
 * K/V row it appends. Qwen3.6-35B-A3B shape (16 q heads, 2 KV heads, 4 q heads per CTA row, rotary 64), two caches with
 * the same random history, qkv rows 36864 bytes apart as the 35B graph lays them out (the attention row is the first
 * 20480 bytes of the q|k|v mix row). Cases cross split boundaries, start mid-context and reach 8k (several attend groups).
 * Controls that must differ: the prep launch dropped (test mutant 1) and a wrong row stride.
 * Verdict line last: "gpu_dattn256_multi_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_gpu_decode_attn256_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_cmdlist.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QH 16u
#define KVH 2u
#define QPC 4u
#define ROT 64u
#define QKVW ((QH + 2u * KVH) * 256u)   /* 5120 */
#define OUTW (QH * 256u)                /* 4096 */
#define XROW 36864u                     /* the graph's ATT_MIX row: 9216 f32 */
#define OROW (OUTW * 4u)
#define MAXP 8448u
static const float EPS = 1e-6f, OSCALE = 18446744073709551616.0f; /* 2^64, as the graph */

static int g_checks, g_failed;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_failed++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t rs = 12345u;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((float)(rs >> 8) / 16777216.0f) * 2.0f - 1.0f; }
static uint16_t bf(float x) { uint32_t u; memcpy(&u, &x, 4); return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }


static int make_cache(OmegaGpuDecodeAttn256Cache **c, const float *qn, const float *kn, const uint16_t *hk, const uint16_t *hv, uint32_t hist) {
    if (omega_gpu_decode_attn256_cache_alloc_shape(1, MAXP, 1e7, ROT, QH, KVH, QPC, c) != 0) return -1;
    if (omega_gpu_decode_attn256_set_norms(*c, 0, qn, kn) != 0) return -1;
    for (uint32_t h = 0; h < KVH && hist; h++)
        if (omega_gpu_decode_attn256_cache_write(*c, 0, h, 0, hist, hk + (size_t)h * MAXP * 256, hv + (size_t)h * MAXP * 256) != 0) return -1;
    return 0;
}

/* one multi record of T tokens from pos0; rows: T rows of XROW bytes */
static int run_multi(OmegaGpuDecodeAttn256Cache *c, uint32_t pos0, uint32_t T, const uint8_t *rows, uint32_t xstride, float *out) {
    if (omega_gpu_decode_attn256_multi_prepare(c) != 0) return -1;
    OmegaGpuCmdList *cl = NULL;
    if (omega_gpu_cmdlist_create(16, &cl) != 0) return -1;
    omega_gpu_session_lock();
    OmegaGpuMem m; memset(&m, 0, sizeof m);
    const size_t s_rows = (size_t)T * XROW, s_out = (size_t)T * OROW;
    int ok = omega_gpu_session_open() != NULL && omega_gpu_session_alloc(s_rows + s_out, &m) == 0;
    if (ok) {
        memcpy(m.cpu, rows, s_rows);
        float *o = (float *)((uint8_t *)m.cpu + s_rows);
        for (size_t i = 0; i < (size_t)T * OUTW; i++) o[i] = NAN;
        ok = omega_gpu_session_sync_to_gpu(&m) == 0;
    }
    if (ok) ok = omega_gpu_decode_attn256_multi_record(cl, c, 0, pos0, T, m.va, xstride, EPS, OSCALE, m.va + s_rows, OROW) == 0;
    OmegaGpuCmdListOpts opts; memset(&opts, 0, sizeof opts);
    OmegaGpuCmdListStats st; memset(&st, 0, sizeof st);
    if (ok) ok = omega_gpu_cmdlist_submit(cl, &opts, &st) == 0 && st.sems_done == st.launches;
    if (ok) ok = omega_gpu_session_sync_to_cpu(&m) == 0;
    if (ok) memcpy(out, (uint8_t *)m.cpu + s_rows, s_out);
    if (m.cpu) omega_gpu_session_free(&m);
    omega_gpu_session_unlock();
    omega_gpu_cmdlist_free(cl);
    if (!ok) return -1;
    return omega_gpu_decode_attn256_counters_check(c) == 0 ? 0 : -1;
}

static int one_case(uint32_t pos0, uint32_t T, int control) {
    float qn[256], kn[256];
    for (unsigned i = 0; i < 256; i++) { qn[i] = 1.0f + 0.2f * frand(); kn[i] = 1.0f + 0.2f * frand(); }
    uint16_t *hk = calloc((size_t)KVH * MAXP * 256, 2), *hv = calloc((size_t)KVH * MAXP * 256, 2);
    uint8_t *rows = calloc((size_t)T, XROW);
    float *oa = malloc((size_t)T * OROW), *ob = malloc((size_t)T * OROW);
    if (!hk || !hv || !rows || !oa || !ob) return -1;
    for (uint32_t h = 0; h < KVH; h++) for (uint32_t p = 0; p < pos0; p++) for (unsigned d = 0; d < 256; d++) {
        hk[((size_t)h * MAXP + p) * 256 + d] = bf(frand() * 3.0f); hv[((size_t)h * MAXP + p) * 256 + d] = bf(frand());
    }
    for (uint32_t t = 0; t < T; t++) { float *r = (float *)(rows + (size_t)t * XROW); for (unsigned i = 0; i < XROW / 4u; i++) r[i] = frand() * 2.0f; }
    OmegaGpuDecodeAttn256Cache *ca = NULL, *cb = NULL;
    int rc = make_cache(&ca, qn, kn, hk, hv, pos0) || make_cache(&cb, qn, kn, hk, hv, pos0) ? -1 : 0;
    for (uint32_t t = 0; t < T && rc == 0; t++)
        if (omega_gpu_decode_attn256(ca, 0, pos0 + t, (const float *)(rows + (size_t)t * XROW), EPS, OSCALE, oa + (size_t)t * OUTW, NULL) != 0) rc = -1;
    if (rc == 0) {
        if (control == 1) omega_gpu_decode_attn256_test_set_multi_mutant(1);
        rc = run_multi(cb, pos0, T, rows, control == 2 ? QKVW * 4u : XROW, ob);
        omega_gpu_decode_attn256_test_set_multi_mutant(0);
    }
    size_t odiff = 0, kdiff = 0;
    if (rc == 0) {
        odiff = 0; for (size_t i = 0; i < (size_t)T * OUTW; i++) { uint32_t a, b; memcpy(&a, &oa[i], 4); memcpy(&b, &ob[i], 4); if (a != b) odiff++; }
        uint16_t ka[256], va[256], kb[256], vb[256];
        for (uint32_t h = 0; h < KVH && rc == 0; h++) for (uint32_t t = 0; t < T; t++) {
            if (omega_gpu_decode_attn256_cache_read(ca, 0, h, pos0 + t, ka, va) != 0 || omega_gpu_decode_attn256_cache_read(cb, 0, h, pos0 + t, kb, vb) != 0) { rc = -1; break; }
            if (memcmp(ka, kb, sizeof ka) || memcmp(va, vb, sizeof va)) kdiff++;
        }
    }
    const char *what = control == 1 ? "control prep dropped" : control == 2 ? "control row stride 20480" : "real";
    if (rc != 0) { CHECK(0, "pos0 %u T %u %s: chip call failed (%s)", pos0, T, what, omega_gpu_decode_attn_last_error()); }
    else if (control) { CHECK(odiff > 0, "pos0 %u T %u %s: NOT caught", pos0, T, what); printf("chip pos0 %u T %u %s: %zu output words differ, %zu K/V rows differ -> %s\n", pos0, T, what, odiff, kdiff, odiff ? "caught" : "MISSED"); }
    else { CHECK(odiff == 0 && kdiff == 0, "pos0 %u T %u: %zu output words, %zu K/V rows differ", pos0, T, odiff, kdiff); printf("chip pos0 %u T %u: outputs %s, K/V rows %s\n", pos0, T, odiff ? "DIFFER" : "identical", kdiff ? "DIFFER" : "identical"); }
    omega_gpu_decode_attn256_cache_free(ca); omega_gpu_decode_attn256_cache_free(cb);
    free(hk); free(hv); free(rows); free(oa); free(ob);
    return rc;
}

int main(void) {
    static const uint32_t cases[][2] = { { 0, 1 }, { 0, 7 }, { 5, 128 }, { 250, 10 }, { 1000, 37 }, { 2040, 128 }, { 6000, 128 }, { 8191, 1 } };
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; i++) one_case(cases[i][0], cases[i][1], 0);
    one_case(300, 16, 1);
    one_case(300, 16, 2);
    printf("gpu_dattn256_multi_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
