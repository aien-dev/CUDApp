/*
 * gpu_lmhead_fp8_study: report-only study (OWN-30B, 2026-10-13). Does an FP8-quantised lm_head pick the same next token
 * as the bf16 lm_head? Loads the real lm_head.weight (bf16 [151936][2048]) and model.norm.weight from the checkpoint
 * dir, quantises the weight to e4m3 in 128x128 blocks (scale = amax / 448, rounded to bf16, the checkpoint's
 * weight_scale_inv convention; round-to-nearest-even into e4m3, saturating), then runs both GEMVs on the chip
 * (bf16 streaming kernel and the FP8 kernel) on N random hidden states and compares the argmax. Nothing is switched.
 *   usage: gpu_lmhead_fp8_study MODEL_DIR [--states N] [--seed S] [--out report.txt]
 */
#include "omega_gpu_bf16_gemv_api.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_session.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define V 151936u
#define H 2048u
#define DATA0 89616ull /* 8 + header length of shard 4 */
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float urand(uint32_t *s) { return (float)(lcg(s) >> 8) / 16777216.0f; }
static float gauss(uint32_t *s) { float a = 0; for (int i = 0; i < 12; i++) a += urand(s); return a - 6.0f; }
static float bf16_f32(uint16_t b) { uint32_t x = (uint32_t)b << 16; float f; memcpy(&f, &x, 4); return f; }
static uint16_t bf16_rne(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t r = x + 0x7fffu + ((x >> 16) & 1u); return (uint16_t)(r >> 16); }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }

/* positive e4m3 values by code 0..126 (127 is NaN): increasing */
static float pos[127];
static uint8_t enc_e4m3(float v) { /* nearest, ties to the even code, saturating at 448 */
    uint8_t sign = v < 0 ? 0x80 : 0; float a = fabsf(v);
    int lo = 0, hi = 126;
    if (a >= pos[126]) return sign | 126;
    while (hi - lo > 1) { int m = (lo + hi) / 2; if (pos[m] <= a) lo = m; else hi = m; }
    float dl = a - pos[lo], dh = pos[hi] - a;
    int c = dl < dh ? lo : dh < dl ? hi : ((lo & 1) ? hi : lo);
    return sign | (uint8_t)c;
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s MODEL_DIR [--states N] [--seed S] [--out file]\n", argv[0]); return 2; }
    uint32_t states = 1000, seed = 20261013; const char *out_path = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--states") && i + 1 < argc) states = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
    }
    for (int c = 0; c < 127; c++) { int e = c >> 3, m = c & 7; pos[c] = e == 0 ? (float)m * (1.0f / 512.0f) : ldexpf(1.0f + m / 8.0f, e - 7); }
    char path[600]; snprintf(path, sizeof path, "%s/model-00004-of-00004.safetensors", argv[1]);
    FILE *f = fopen(path, "rb"); if (!f) { printf("cannot open %s\n", path); return 1; }
    uint16_t *w = malloc((size_t)V * H * 2), nw[H];
    fseeko(f, (off_t)DATA0, SEEK_SET);
    if (fread(w, 2, (size_t)V * H, f) != (size_t)V * H) { printf("short read lm_head\n"); return 1; }
    fseeko(f, (off_t)(DATA0 + 622405248ull), SEEK_SET);
    if (fread(nw, 2, H, f) != H) { printf("short read norm\n"); return 1; }
    fclose(f);
    /* quantise */
    uint8_t *q = malloc((size_t)V * H); uint16_t *sc = malloc((size_t)(V / 128) * (H / 128) * 2);
    double sq_err = 0, sq_w = 0;
    for (uint32_t bi = 0; bi < V / 128; bi++) for (uint32_t bj = 0; bj < H / 128; bj++) {
        float amax = 0;
        for (uint32_t r = 0; r < 128; r++) for (uint32_t c = 0; c < 128; c++) { float a = fabsf(bf16_f32(w[(size_t)(bi * 128 + r) * H + bj * 128 + c])); if (a > amax) amax = a; }
        uint16_t sb = bf16_rne(amax > 0 ? amax / 448.0f : 1.0f); float s = bf16_f32(sb);
        sc[bi * (H / 128) + bj] = sb;
        for (uint32_t r = 0; r < 128; r++) for (uint32_t c = 0; c < 128; c++) {
            size_t idx = (size_t)(bi * 128 + r) * H + bj * 128 + c; float wv = bf16_f32(w[idx]);
            uint8_t e = enc_e4m3(wv / s); q[idx] = e;
            float d = (e & 0x80 ? -1.0f : 1.0f) * pos[e & 0x7f] * s; sq_err += (double)(d - wv) * (d - wv); sq_w += (double)wv * wv;
        }
    }
    printf("quantised lm_head: weight rms error %.4f of weight rms (block-scaled e4m3)\n", sqrt(sq_err / sq_w));
    OmegaGpuBf16Tensor *tb = NULL; OmegaGpuFp8Tensor *tf = NULL;
    int r1 = omega_gpu_bf16_tensor_upload(V, H, w, &tb), r2 = omega_gpu_fp8_tensor_upload(1, V, H, q, sc, &tf);
    printf("uploads: bf16 %s, fp8 %s\n", omega_gpu_bf16_rc_name(r1), omega_gpu_fp8_rc_name(r2));
    if (r1 || r2) return 1;
    (void)omega_gpu_session_set_spin_us(2000);
    float *x = malloc(H * 4), *yb = malloc(V * 4), *yf = malloc(V * 4);
    uint32_t s = seed, agree = 0, top5_in = 0, fails = 0;
    uint64_t *tb_ns = malloc(states * 8), *tf_ns = malloc(states * 8);
    double maxrel = 0;
    uint32_t bucket[4] = { 0, 0, 0, 0 }, bucket_dis[4] = { 0, 0, 0, 0 }; /* margin <0.05, <0.2, <1, >=1 (bf16 logit gap top1-top2) */
    for (uint32_t it = 0; it < states; it++) {
        for (uint32_t i = 0; i < H; i++) x[i] = gauss(&s) * bf16_f32(nw[i]);
        OmegaGpuBf16Info ib; OmegaGpuFp8Info jf;
        if (omega_gpu_bf16_gemv_f32(tb, x, yb, &ib) || omega_gpu_fp8_gemv_f32(tf, 0, x, yf, &jf)) { fails++; printf("state %u: chip call failed (%s)\n", it, omega_gpu_bf16_last_error()); break; }
        tb_ns[it] = ib.elapsed_ns; tf_ns[it] = jf.elapsed_ns;
        uint32_t ab = 0, af = 0; float m2 = -1e30f;
        for (uint32_t v = 1; v < V; v++) { if (yb[v] > yb[ab]) ab = v; if (yf[v] > yf[af]) af = v; }
        for (uint32_t v = 0; v < V; v++) if (v != ab && yb[v] > m2) m2 = yb[v];
        double margin = yb[ab] - m2;
        int ok = ab == af; agree += ok;
        /* is the bf16 argmax inside the fp8 top 5? */
        uint32_t better = 0; for (uint32_t v = 0; v < V; v++) if (yf[v] > yf[ab]) better++;
        top5_in += better < 5;
        int b = margin < 0.05 ? 0 : margin < 0.2 ? 1 : margin < 1.0 ? 2 : 3; bucket[b]++; if (!ok) { bucket_dis[b]++; }
        double num = 0, den = 0; for (uint32_t v = 0; v < V; v++) { double d = (double)yf[v] - yb[v]; num += d * d; den += (double)yb[v] * yb[v]; }
        double rel = sqrt(num / den); if (rel > maxrel) maxrel = rel;
        if (it == 0) printf("state 0: bf16 argmax %u (%.3f), fp8 argmax %u (%.3f), margin %.4f, logit rms rel diff %.4f\n", ab, yb[ab], af, yf[af], margin, rel);
    }
    uint32_t n = states - fails;
    qsort(tb_ns, n, 8, cmp_u64); qsort(tf_ns, n, 8, cmp_u64);
    char rep[2000];
    snprintf(rep, sizeof rep,
        "lm_head fp8 vs bf16 argmax study (real lm_head.weight, %u random hidden states = gaussian x model.norm.weight, seed %u)\n"
        "argmax agreement: %u of %u (%.2f%%); bf16 argmax within fp8 top-5: %u of %u\n"
        "logit rms relative difference, worst over states: %.4f\n"
        "agreement by bf16 top1-top2 logit gap: <0.05: %u/%u disagree, <0.2: %u/%u, <1: %u/%u, >=1: %u/%u\n"
        "chip medians: bf16 kernel %.1f us, fp8 kernel %.1f us\n",
        n, seed, agree, n, 100.0 * agree / (n ? n : 1), top5_in, n, maxrel,
        bucket_dis[0], bucket[0], bucket_dis[1], bucket[1], bucket_dis[2], bucket[2], bucket_dis[3], bucket[3],
        n ? tb_ns[n / 2] / 1e3 : 0.0, n ? tf_ns[n / 2] / 1e3 : 0.0);
    printf("%s", rep);
    if (out_path) { FILE *o = fopen(out_path, "w"); if (o) { fputs(rep, o); fclose(o); } }
    omega_gpu_bf16_tensor_free(tb); omega_gpu_fp8_tensor_free(tf);
    omega_gpu_session_close();
    return fails ? 1 : 0;
}
