/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* cpubw: CPU effective bandwidth on chosen cores, as a plain read stream and as a real FP8 expert GEMV (DRA-72, P-DRA-4).
 *
 * Question (DRA-72 measurement 1): how many GB/s of expert weights can the GB10's CPU cores consume while the 35B decodes on
 * the GPU? The decision rule needs >= ~70 GB/s under GPU load. A copy loop only bounds that; the "fp8gemv" mode runs the
 * work a CPU expert share would actually do, so its bytes/s is the number the rule is about.
 *
 * Modes
 *  - stream:  each thread sums its own buffer with 128-byte NEON loads (vector adds only): the read ceiling of a core.
 *  - fp8gemv: each thread owns a ring of Qwen3.6-35B-A3B experts in the checkpoint layout (gate_up 1024 x 2048 e4m3, down
 *             2048 x 512 e4m3, block-128 BF16 scales: 3,145,728 + 384 = 3,146,112 B per expert, DRA-63 frozen bytes) and
 *             runs the full expert per pass: gate_up GEMV, SiLU(gate) * up, down GEMV, accumulate y. GB10 has no FEAT_FP8,
 *             so weights are decoded in software: e4m3 byte b -> f16 bits ((b << 8) as int16 >> 1) & ~0x4000 equals the
 *             e4m3 value times 2^-8 for every finite code, subnormals included (the 2^8 is folded into the block scale);
 *             products accumulate in f32 with FMLAL (FEAT_FHM, /proc/cpuinfo asimdfhm). x is f16.
 *  - --check: one expert through the NEON path against a scalar f64 reference with the exact e4m3 decode; exits 0 when every
 *             output is within tolerance. --mutate drops the ~0x4000 mask (a wrong decode): the check must then fail.
 *
 * Threads are pinned one per listed core (pthread_setaffinity_np), touch their own memory after pinning, start on one
 * barrier and stop together after --seconds. Cores 10-14 are refused (P-DRA-4 host rule: never pin to them).
 * Output (stdout and --out), integers only so the Omega judge (tools/dra72/judge.om) parses it:
 *   CPUBW 1 mode=<m> threads=<n> unit_bytes=<u> window_ns=<w> t0_mono_ns=<t0> t0_real_ns=<t0 on CLOCK_REALTIME> mib=<m>
 *   CORE <core> <bytes> <ns>        per thread: completed bytes and the ns from the start barrier to its last completion
 *   WIN <i> <start_mono_ns> <bytes>  all threads' completed bytes per window (CLOCK_MONOTONIC, shared with gpu_dra72_load)
 *   TOTAL <bytes> <ns>
 * plus one human SUMMARY line. No GPU, no device file: runs anywhere on aarch64 with asimdfhm. */
#define _GNU_SOURCE
#include <arm_neon.h>
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define HID 2048u                     /* hidden size */
#define INTER 512u                    /* expert intermediate (OMEGA_GPU_Q36_EXPERT_INTER) */
#define GU_ROWS (2u * INTER)          /* gate rows then up rows */
#define W_GU ((size_t)GU_ROWS * HID)  /* 2,097,152 B */
#define W_D ((size_t)HID * INTER)     /* 1,048,576 B */
#define S_GU ((GU_ROWS / 128u) * (HID / 128u)) /* 8 x 16 BF16 */
#define S_D ((HID / 128u) * (INTER / 128u))    /* 16 x 4 BF16 */
#define UNIT_FP8 (W_GU + W_D + 2u * (S_GU + S_D)) /* 3,146,112 B */
#define UNIT_STREAM ((size_t)1 << 20)
#define MAX_THREADS 20
#define MAX_WINDOWS 65536

static uint64_t mono_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float bf16_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

/* one row of K e4m3 weights against x (f16), block-128 BF16 scales for this row group; f32 result */
/* 32 bytes per step into 8 independent f32 accumulators: FMLAL has a multi-cycle latency, so 4 chains left the X925
 * latency-bound (7.8 GB/s on core 5, host smoke); 8 chains keep the FP pipes fed. */
#define DECODE16(V, LO, HI, CLEAR)                                                                          \
    int16x8_t LO = vshrq_n_s16(vreinterpretq_s16_u8(vzip1q_u8(z, V)), 1);                                    \
    int16x8_t HI = vshrq_n_s16(vreinterpretq_s16_u8(vzip2q_u8(z, V)), 1);                                    \
    if (CLEAR) { LO = vbicq_s16(LO, m); HI = vbicq_s16(HI, m); }
#define DEFINE_ROW_DOT(NAME, CLEAR)                                                                         \
    static inline float NAME(const uint8_t *w, const float16_t *x, const uint16_t *sc, uint32_t K) {     \
        const uint8x16_t z = vdupq_n_u8(0);                                                                 \
        const int16x8_t m = vdupq_n_s16(0x4000);                                                            \
        float32x4_t tot = vdupq_n_f32(0.0f);                                                                \
        for (uint32_t b = 0; b < K; b += 128) {                                                             \
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0; \
            for (uint32_t j = b; j < b + 128; j += 32) {                                                    \
                const uint8x16_t v0 = vld1q_u8(w + j), v1 = vld1q_u8(w + j + 16);                           \
                DECODE16(v0, l0, h0, CLEAR)                                                                 \
                DECODE16(v1, l1, h1, CLEAR)                                                                 \
                const float16x8_t x0 = vld1q_f16(x + j), x1 = vld1q_f16(x + j + 8);                         \
                const float16x8_t x2 = vld1q_f16(x + j + 16), x3 = vld1q_f16(x + j + 24);                   \
                a0 = vfmlalq_low_f16(a0, vreinterpretq_f16_s16(l0), x0); a1 = vfmlalq_high_f16(a1, vreinterpretq_f16_s16(l0), x0); \
                a2 = vfmlalq_low_f16(a2, vreinterpretq_f16_s16(h0), x1); a3 = vfmlalq_high_f16(a3, vreinterpretq_f16_s16(h0), x1); \
                a4 = vfmlalq_low_f16(a4, vreinterpretq_f16_s16(l1), x2); a5 = vfmlalq_high_f16(a5, vreinterpretq_f16_s16(l1), x2); \
                a6 = vfmlalq_low_f16(a6, vreinterpretq_f16_s16(h1), x3); a7 = vfmlalq_high_f16(a7, vreinterpretq_f16_s16(h1), x3); \
            }                                                                                               \
            const float32x4_t s = vaddq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)), vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7))); \
            tot = vfmaq_n_f32(tot, s, bf16_f32(sc[b / 128]) * 256.0f);                                      \
        }                                                                                                   \
        return vaddvq_f32(tot);                                                                             \
    }
DEFINE_ROW_DOT(row_dot, 1)
DEFINE_ROW_DOT(row_dot_mutant, 0)

typedef struct { const uint8_t *gu, *d; const uint16_t *sgu, *sd; } Expert;
static Expert expert_at(const uint8_t *base) {
    Expert e; e.gu = base; e.d = base + W_GU;
    e.sgu = (const uint16_t *)(base + W_GU + W_D); e.sd = e.sgu + S_GU;
    return e;
}

/* full expert: g = Wgu x, h = silu(gate) * up (f16), y += Wd h */
static void expert_run(const Expert *e, const float16_t *x, float *g, float16_t *h, float *y, int mutant) {
    for (uint32_t r = 0; r < GU_ROWS; r++)
        g[r] = (mutant ? row_dot_mutant : row_dot)(e->gu + (size_t)r * HID, x, e->sgu + (r / 128u) * (HID / 128u), HID);
    for (uint32_t i = 0; i < INTER; i++) { const float a = g[i]; h[i] = (float16_t)(a / (1.0f + expf(-a)) * g[INTER + i]); }
    for (uint32_t r = 0; r < HID; r++)
        y[r] += (mutant ? row_dot_mutant : row_dot)(e->d + (size_t)r * INTER, h, e->sd + (r / 128u) * (INTER / 128u), INTER);
}

static void fill_expert(uint8_t *base, uint32_t *st) {
    for (size_t i = 0; i < W_GU + W_D; i++) { uint8_t b = (uint8_t)(lcg(st) >> 24); base[i] = (b & 0x7F) == 0x7F ? (uint8_t)(b ^ 1) : b; }
    uint16_t *s = (uint16_t *)(base + W_GU + W_D);
    for (uint32_t i = 0; i < S_GU + S_D; i++) { float f = ldexpf(1.0f + (float)(lcg(st) >> 9) / 8388608.0f, -9 + (int)(lcg(st) >> 30)); uint32_t u; memcpy(&u, &f, 4); s[i] = (uint16_t)(u >> 16); }
}
static void fill_x(float16_t *x, uint32_t n, uint32_t *st) { for (uint32_t i = 0; i < n; i++) x[i] = (float16_t)((float)(lcg(st) >> 8) / 16777216.0f * 2.0f - 1.0f); }

/* exact e4m3 (OCP FP8 E4M3, bias 7) */
static double e4m3(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const double v = e ? ldexp(1.0 + m / 8.0, e - 7) : ldexp(m / 8.0, -6);
    return (b & 0x80) ? -v : v;
}
/* reference row: value and sum of |terms| (the tolerance scale) */
static double ref_row(const uint8_t *w, const float16_t *x, const uint16_t *sc, uint32_t K, double *absum) {
    double t = 0.0, a = 0.0;
    for (uint32_t k = 0; k < K; k++) { const double p = e4m3(w[k]) * (double)x[k] * (double)bf16_f32(sc[k / 128]); t += p; a += fabs(p); }
    *absum = a; return t;
}

static int run_check(int mutant) {
    uint8_t *base = aligned_alloc(64, (UNIT_FP8 + 63) & ~(size_t)63);
    float16_t *x = aligned_alloc(64, HID * 2), *h = aligned_alloc(64, INTER * 2);
    float *g = aligned_alloc(64, GU_ROWS * 4), *y = aligned_alloc(64, HID * 4);
    if (!base || !x || !h || !g || !y) { printf("check: out of memory\n"); return 2; }
    uint32_t st = 72u;
    fill_expert(base, &st); fill_x(x, HID, &st);
    /* every finite e4m3 code once in the first gate row, so the decode is exercised on all 254 values (subnormals too) */
    for (uint32_t i = 0, c = 0; c < 256; c++) if ((c & 0x7F) != 0x7F) base[i++] = (uint8_t)c;
    memset(y, 0, HID * 4);
    const Expert e = expert_at(base);
    expert_run(&e, x, g, h, y, mutant);
    uint32_t bad = 0; double worst = 0.0;
    for (uint32_t r = 0; r < GU_ROWS; r++) {
        double a; const double ref = ref_row(e.gu + (size_t)r * HID, x, e.sgu + (r / 128u) * (HID / 128u), HID, &a);
        const double err = fabs((double)g[r] - ref), tol = 1e-5 * a + 1e-30, rel = err / tol;
        if (!(err <= tol)) bad++;
        if (rel > worst || rel != rel) worst = rel;
    }
    /* down: reference from the same f16 h the NEON path produced, so the check isolates the down GEMV */
    for (uint32_t r = 0; r < HID; r++) {
        double a; const double ref = ref_row(e.d + (size_t)r * INTER, h, e.sd + (r / 128u) * (INTER / 128u), INTER, &a);
        const double err = fabs((double)y[r] - ref), tol = 1e-5 * a + 1e-30, rel = err / tol;
        if (!(err <= tol)) bad++;
        if (rel > worst || rel != rel) worst = rel;
    }
    printf("CHECK %s rows=%u bad=%u worst_err_over_tol=%.3g tolerance=1e-5*sum|terms| %s\n", mutant ? "mutant" : "neon",
           GU_ROWS + HID, bad, worst, bad ? "FAIL" : "PASS");
    free(base); free(x); free(h); free(g); free(y);
    return bad ? 1 : 0;
}

typedef struct {
    int core, mode;
    size_t mib;
    pthread_barrier_t *bar;
    uint64_t t0, win_ns, bytes, last_ns, sink;
    int err;
} Worker;
static atomic_int g_stop;
static _Atomic uint64_t g_win[MAX_WINDOWS];
static uint64_t g_t0;

static void win_add(uint64_t t, uint64_t bytes, uint64_t win_ns) {
    const uint64_t i = (t - g_t0) / win_ns;
    if (i < MAX_WINDOWS) atomic_fetch_add_explicit(&g_win[i], bytes, memory_order_relaxed);
}

/* two barrier rounds: every thread is pinned with its memory touched, then main publishes g_t0, then all start */
static void start_gate(pthread_barrier_t *b) { pthread_barrier_wait(b); pthread_barrier_wait(b); }

static void *worker(void *arg) {
    Worker *w = arg;
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(w->core, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof set, &set) != 0) { w->err = 1; start_gate(w->bar); return NULL; }
    const size_t bytes = w->mib << 20;
    if (w->mode == 0) {
        const size_t nunits = bytes / UNIT_STREAM;
        uint8_t *buf = aligned_alloc(4096, nunits * UNIT_STREAM);
        if (!buf || nunits == 0) { w->err = 2; start_gate(w->bar); return NULL; }
        memset(buf, 0x5a, nunits * UNIT_STREAM); /* first touch on this core */
        start_gate(w->bar);
        const uint64_t t0 = g_t0; uint64x2_t acc0 = vdupq_n_u64(0), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        size_t u = 0;
        while (!atomic_load_explicit(&g_stop, memory_order_relaxed)) {
            const uint64_t *p = (const uint64_t *)(buf + u * UNIT_STREAM);
            for (size_t i = 0; i < UNIT_STREAM / 8; i += 16) {
                acc0 = vaddq_u64(acc0, vld1q_u64(p + i));      acc1 = vaddq_u64(acc1, vld1q_u64(p + i + 2));
                acc2 = vaddq_u64(acc2, vld1q_u64(p + i + 4));  acc3 = vaddq_u64(acc3, vld1q_u64(p + i + 6));
                acc0 = vaddq_u64(acc0, vld1q_u64(p + i + 8));  acc1 = vaddq_u64(acc1, vld1q_u64(p + i + 10));
                acc2 = vaddq_u64(acc2, vld1q_u64(p + i + 12)); acc3 = vaddq_u64(acc3, vld1q_u64(p + i + 14));
            }
            const uint64_t t = mono_ns();
            w->bytes += UNIT_STREAM; w->last_ns = t - t0; win_add(t, UNIT_STREAM, w->win_ns);
            if (++u == nunits) u = 0;
        }
        const uint64x2_t s = vaddq_u64(vaddq_u64(acc0, acc1), vaddq_u64(acc2, acc3));
        w->sink = vgetq_lane_u64(s, 0) + vgetq_lane_u64(s, 1);
        free(buf);
    } else {
        const size_t stride = (UNIT_FP8 + 4095) & ~(size_t)4095;
        size_t nexp = bytes / stride; if (nexp == 0) nexp = 1;
        uint8_t *buf = aligned_alloc(4096, nexp * stride);
        float16_t *x = aligned_alloc(64, HID * 2), *h = aligned_alloc(64, INTER * 2);
        float *g = aligned_alloc(64, GU_ROWS * 4), *y = aligned_alloc(64, HID * 4);
        if (!buf || !x || !h || !g || !y) { w->err = 2; start_gate(w->bar); return NULL; }
        uint32_t st = 1000u + (uint32_t)w->core;
        for (size_t i = 0; i < nexp; i++) fill_expert(buf + i * stride, &st); /* first touch on this core */
        fill_x(x, HID, &st); memset(y, 0, HID * 4);
        start_gate(w->bar);
        const uint64_t t0 = g_t0; size_t u = 0;
        while (!atomic_load_explicit(&g_stop, memory_order_relaxed)) {
            const Expert e = expert_at(buf + u * stride);
            expert_run(&e, x, g, h, y, 0);
            const uint64_t t = mono_ns();
            w->bytes += UNIT_FP8; w->last_ns = t - t0; win_add(t, UNIT_FP8, w->win_ns);
            if (++u == nexp) u = 0;
        }
        float s = 0.0f; for (uint32_t i = 0; i < HID; i++) s += y[i];
        memcpy(&w->sink, &s, 4);
        free(buf); free(x); free(h); free(g); free(y);
    }
    return NULL;
}

static int parse_cores(const char *s, int *cores) {
    int n = 0;
    while (*s) {
        char *end; long a = strtol(s, &end, 10), b = a;
        if (end == s) return -1;
        if (*end == '-') { s = end + 1; b = strtol(s, &end, 10); if (end == s) return -1; }
        if (a < 0 || b < a || b > 63) return -1;
        for (long c = a; c <= b; c++) { if (n == MAX_THREADS) return -1; for (int i = 0; i < n; i++) if (cores[i] == c) return -1; cores[n++] = (int)c; }
        s = end; if (*s == ',') s++; else if (*s) return -1;
    }
    return n;
}

int main(int argc, char **argv) {
    const char *mode_s = "fp8gemv", *cores_s = "5", *out_path = NULL;
    double seconds = 2.0; size_t mib = 256; uint64_t win_ms = 100; int check = 0, mutant = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode_s = argv[++i];
        else if (!strcmp(argv[i], "--cores") && i + 1 < argc) cores_s = argv[++i];
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
        else if (!strcmp(argv[i], "--mib") && i + 1 < argc) mib = (size_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--window-ms") && i + 1 < argc) win_ms = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--check")) check = 1;
        else if (!strcmp(argv[i], "--mutate")) mutant = 1;
        else {
            fprintf(stderr, "usage: %s [--mode stream|fp8gemv] [--cores 5,6|0-4,15-19] [--seconds S] [--mib PER_THREAD] [--window-ms W] [--out FILE]\n"
                            "       %s --check [--mutate]\n", argv[0], argv[0]);
            return 2;
        }
    }
    if (check) return run_check(mutant);
    const int mode = !strcmp(mode_s, "stream") ? 0 : !strcmp(mode_s, "fp8gemv") ? 1 : -1;
    int cores[MAX_THREADS]; const int n = parse_cores(cores_s, cores);
    if (mode < 0 || n <= 0 || !(seconds > 0.0 && seconds <= 3600.0) || mib == 0 || mib > 16384 || win_ms == 0) { fprintf(stderr, "cpubw: bad arguments\n"); return 2; }
    for (int i = 0; i < n; i++) if (cores[i] >= 10 && cores[i] <= 14) { fprintf(stderr, "REFUSED: core %d (cores 10-14 are reserved on this host)\n", cores[i]); return 2; }
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    for (int i = 0; i < n; i++) if (cores[i] >= ncpu) { fprintf(stderr, "cpubw: core %d not online (%ld cpus)\n", cores[i], ncpu); return 2; }
    const uint64_t win_ns = win_ms * 1000000ull;
    if ((uint64_t)(seconds * 1e9) / win_ns + 2 > MAX_WINDOWS) { fprintf(stderr, "cpubw: too many windows\n"); return 2; }

    pthread_barrier_t bar; pthread_barrier_init(&bar, NULL, (unsigned)n + 1u);
    Worker w[MAX_THREADS]; pthread_t th[MAX_THREADS];
    memset(w, 0, sizeof w);
    for (int i = 0; i < n; i++) { w[i].core = cores[i]; w[i].mode = mode; w[i].mib = mib; w[i].bar = &bar; w[i].win_ns = win_ns; }
    for (int i = 0; i < n; i++) if (pthread_create(&th[i], NULL, worker, &w[i]) != 0) { fprintf(stderr, "cpubw: pthread_create\n"); return 1; }
    pthread_barrier_wait(&bar); /* round 1: all threads pinned and their memory touched */
    struct timespec rt; clock_gettime(CLOCK_REALTIME, &rt);
    g_t0 = mono_ns();
    const uint64_t t0_real = (uint64_t)rt.tv_sec * 1000000000ull + (uint64_t)rt.tv_nsec; /* aligns WIN lines with shell $EPOCHREALTIME stamps */
    pthread_barrier_wait(&bar); /* round 2: go (the barrier orders the g_t0 store before every worker's read) */
    const uint64_t t_end = g_t0 + (uint64_t)(seconds * 1e9);
    struct timespec ts = { (time_t)(t_end / 1000000000ull), (long)(t_end % 1000000000ull) };
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, NULL) == EINTR) {}
    atomic_store(&g_stop, 1);
    const uint64_t t_stop = mono_ns();
    for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
    for (int i = 0; i < n; i++) if (w[i].err) { fprintf(stderr, "cpubw: worker on core %d failed (%s)\n", w[i].core, w[i].err == 1 ? "affinity" : "memory"); return 1; }

    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { fprintf(stderr, "cpubw: cannot write %s\n", out_path); return 1; }
    FILE *fs[2] = { stdout, out };
    const uint64_t unit = mode ? UNIT_FP8 : UNIT_STREAM;
    uint64_t total = 0, sink = 0;
    for (int i = 0; i < n; i++) { total += w[i].bytes; sink ^= w[i].sink; }
    const uint64_t ns = t_stop - g_t0, nwin = ns / win_ns;
    for (int f = 0; f < 2; f++) {
        if (!fs[f]) continue;
        fprintf(fs[f], "CPUBW 1 mode=%s threads=%d unit_bytes=%" PRIu64 " window_ns=%" PRIu64 " t0_mono_ns=%" PRIu64 " t0_real_ns=%" PRIu64 " mib=%zu\n", mode_s, n, unit, win_ns, g_t0, t0_real, mib);
        for (int i = 0; i < n; i++) fprintf(fs[f], "CORE %d %" PRIu64 " %" PRIu64 "\n", w[i].core, w[i].bytes, w[i].last_ns);
        for (uint64_t i = 0; i < nwin; i++) fprintf(fs[f], "WIN %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", i, g_t0 + i * win_ns, atomic_load(&g_win[i]));
        fprintf(fs[f], "TOTAL %" PRIu64 " %" PRIu64 "\n", total, ns);
    }
    printf("SUMMARY mode=%s cores=%s threads=%d %.2f GB/s over %.3f s (sink %" PRIx64 ")\n", mode_s, cores_s, n, (double)total / (double)ns, ns / 1e9, sink);
    if (out) fclose(out);
    return 0;
}
