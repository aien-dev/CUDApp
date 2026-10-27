/* gpu_scan_test.c -- row C3 of docs/OMEGA.md: reductions and prefix sums of any length, bit-exact against the
 * frozen-order host reference (src/omega_gpu_scan.h).
 *
 *   --host-only  no device: refusals, every kernel through our codegen and encoder, the references checked against
 *                exact integer sums and against the E1 reduction reference (omega_reduce_reference)
 *   --sim        the real kernels through the warp simulator (tests/bw_warp_sim.h), both scheduling orders, lengths
 *                that cross every tile, CTA and grid-stride edge; the three mutants must be caught
 *   (no flag)    the GB10: every n from 1 to 2048, then 2^k - 1, 2^k, 2^k + 1 up to 2^28, then random lengths;
 *                bit-exact (a NaN result must be a NaN); the three mutants must be caught.
 *                --max-log2 K stops the power-of-two ladder at 2^K (default 28).
 * Last line: "C3 scan <tier>: PASS (<checks> checks)" or "... FAIL (<failed> of <checks>)". */
#include "omega_gpu_scan.h"
#include "omega_gpu_session.h"
#include "omega_numeric_reduce.h"
#include "bw_warp_sim.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_failed;
static void check(int ok, const char *what) {
    g_checks++;
    if (!ok) { g_failed++; printf("FAIL: %s\n", what); }
}

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static int is_nan(uint32_t u) { return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu); }
static int same(float a, float b) { uint32_t x = f2u(a), y = f2u(b); return x == y || (is_nan(x) && is_nan(y)); }

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(g_rng >> 33); }

/* inputs: 0 small integers (every sum exact), 1 mixed magnitudes and signed zeros, 2 mixed plus NaN and infinities */
static void fill(float *x, size_t n, int kind, uint64_t seed) {
    g_rng = seed * 0x2545f4914f6cdd1dull + 1;
    for (size_t i = 0; i < n; i++) {
        if (kind == 0) { x[i] = (float)((int)(rnd() % 2001u) - 1000); continue; }
        float m = (float)((rnd() & 0x7fffffu) | 1u) * (rnd() & 1 ? -1.0f : 1.0f);
        int e = (int)(rnd() % 41u) - 20 - 23;
        x[i] = m * u2f((uint32_t)(127 + e) << 23);
        if (i % 97u == 5u) x[i] = 0.0f;
        if (i % 131u == 7u) x[i] = -0.0f;
        if (kind == 2 && i % 53u == 11u) x[i] = u2f(0x7fc00000u | (rnd() & 0xffffu));
        if (kind == 2 && i == n / 3) x[i] = u2f(0x7f800000u);
        if (kind == 2 && i == n / 2 && n > 4) x[i] = u2f(0xff800000u);
    }
}

static const char *op_name(OmegaGpuRedOp op) { return op == OMEGA_GPU_RED_SUM ? "sum" : op == OMEGA_GPU_RED_MAX ? "max" : "min"; }

/* ------------------------------------------------------------------ host only */

static int fake_fail(void *ctx, OmegaScanKind k, const uint32_t a[OMEGA_GPU_SCAN_ARGS_WORDS], uint32_t g) {
    (void)ctx; (void)k; (void)a; (void)g; return -1;
}

static void host_only(void) {
    char w[200];
    float x[4] = { 1, 2, 3, 4 }, o;
    check(omega_gpu_reduce(OMEGA_GPU_RED_SUM, x, 0, &o, NULL) == OMEGA_GPU_SCAN_BAD_ARGS, "reduce n = 0 refused");
    check(omega_gpu_reduce(OMEGA_GPU_RED_SUM, NULL, 4, &o, NULL) == OMEGA_GPU_SCAN_BAD_ARGS, "reduce NULL input refused");
    check(omega_gpu_reduce((OmegaGpuRedOp)7, x, 4, &o, NULL) == OMEGA_GPU_SCAN_BAD_ARGS, "reduce unknown op refused");
    check(omega_gpu_reduce(OMEGA_GPU_RED_SUM, x, OMEGA_GPU_SCAN_MAX_N + 1, &o, NULL) == OMEGA_GPU_SCAN_TOO_LARGE, "reduce 2^28 + 1 refused");
    check(omega_gpu_prefix_sum(x, 0, x, NULL) == OMEGA_GPU_SCAN_BAD_ARGS, "prefix n = 0 refused");
    check(omega_gpu_prefix_sum(x, OMEGA_GPU_SCAN_MAX_N + 1, x, NULL) == OMEGA_GPU_SCAN_TOO_LARGE, "prefix 2^28 + 1 refused");
    uint64_t r;
    check(omega_gpu_scan_drive_reduce(OMEGA_GPU_RED_SUM, 100, 0, 0, &r, fake_fail, NULL, NULL) == OMEGA_GPU_SCAN_CHIP_FAIL, "a failed launch is reported (reduce)");
    check(omega_gpu_scan_drive_prefix(100, 0, 0, 0, fake_fail, NULL, NULL) == OMEGA_GPU_SCAN_CHIP_FAIL, "a failed launch is reported (prefix)");

    for (int mut = 0; mut <= 3; mut++)
        for (int k = 0; k < OMEGA_SCAN_K_COUNT; k++) {
            BlackwellIRProgram *p = calloc(1, sizeof *p);
            OmegaScanSpec s = { (OmegaScanKind)k, mut };
            int rc = omega_gpu_scan_codegen(&s, p);
            size_t bytes = rc == 0 ? omega_bw_program_code_bytes(p) : 0;
            uint8_t *code = bytes ? malloc(bytes) : NULL;
            OmegaBlackwellKernel kk; memset(&kk, 0, sizeof kk);
            int enc = code ? omega_blackwell_kernel_encode_into(p, code, bytes, &kk) : -1;
            snprintf(w, sizeof w, "kernel %d mutant %d through codegen and encoder (%zu bytes, %u gprs)", k, mut, bytes, (unsigned)kk.gpr_count);
            check(rc == 0 && enc == 0 && kk.code_size == bytes, w);
            if (mut == 0 && rc == 0) printf("kernel %d: %u insns, %zu bytes, %u gprs\n", k, (unsigned)p->count, bytes, (unsigned)kk.gpr_count);
            free(code); free(p);
        }

    /* the references against exact sums: integers in [-1000, 1000], n <= 16384, every partial sum below 2^24 */
    static float a[16384], y[16384];
    const size_t ns[] = { 1, 2, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025, 1057, 16384 };
    for (size_t t = 0; t < sizeof ns / sizeof ns[0]; t++) {
        size_t n = ns[t];
        fill(a, n, 0, n);
        double s = 0, mx = -1e9, mn = 1e9;
        for (size_t i = 0; i < n; i++) { s += a[i]; if (a[i] > mx) mx = a[i]; if (a[i] < mn) mn = a[i]; }
        float rs, rmx, rmn;
        int ok = omega_gpu_reduce_reference(OMEGA_GPU_RED_SUM, a, n, &rs) == 0 && omega_gpu_reduce_reference(OMEGA_GPU_RED_MAX, a, n, &rmx) == 0 &&
                 omega_gpu_reduce_reference(OMEGA_GPU_RED_MIN, a, n, &rmn) == 0;
        snprintf(w, sizeof w, "reference sum/max/min exact at n = %zu", n);
        check(ok && rs == (float)s && rmx == (float)mx && rmn == (float)mn, w);
        ok = omega_gpu_prefix_sum_reference(a, n, y) == 0;
        double run = 0; size_t bad = 0;
        for (size_t i = 0; i < n; i++) { run += a[i]; bad += y[i] != (float)run; }
        snprintf(w, sizeof w, "reference prefix sum exact at n = %zu (%zu wrong)", n, bad);
        check(ok && bad == 0, w);
    }

    /* the reduction reference is the E1 order: equal bit for bit to omega_reduce_reference on any input */
    static float b[300000];
    const size_t ms[] = { 1, 2, 17, 32, 33, 100, 1000, 1024, 1025, 4097, 32768, 32769, 299999 };
    for (int kind = 1; kind <= 2; kind++)
        for (size_t t = 0; t < sizeof ms / sizeof ms[0]; t++)
            for (int op = 0; op < 3; op++) {
                size_t n = ms[t];
                fill(b, n, kind, n * 3 + (size_t)op);
                float mine, e1;
                int ok = omega_gpu_reduce_reference((OmegaGpuRedOp)op, b, n, &mine) == 0 && omega_reduce_reference((OmegaReduceOp)op, b, n, &e1) == 0;
                snprintf(w, sizeof w, "reference %s equals the E1 reference, inputs %d, n = %zu (%08x vs %08x)", op_name(op), kind, n, f2u(mine), f2u(e1));
                check(ok && same(mine, e1), w);
            }

    /* the frozen order is a real order: summing left to right gives a different answer on mixed inputs */
    fill(b, 4097, 1, 99);
    float tree, seq = -0.0f;
    for (size_t i = 0; i < 4097; i++) seq += b[i];
    omega_gpu_reduce_reference(OMEGA_GPU_RED_SUM, b, 4097, &tree);
    check(f2u(tree) != f2u(seq), "the frozen tree differs from a left-to-right sum (the order is pinned, not luck)");
}

/* ------------------------------------------------------------------ the simulator */

typedef struct {
    BlackwellIRProgram *prog[OMEGA_SCAN_K_COUNT];
    int order;
    WsRange ranges[3];
    char err[260];
} SimCtx;

static int sim_launch(void *ctx, OmegaScanKind k, const uint32_t a[OMEGA_GPU_SCAN_ARGS_WORDS], uint32_t grid_x) {
    SimCtx *c = ctx;
    static WsSim s;
    memset(&s, 0, sizeof s);
    s.p = c->prog[k]; s.threads = OMEGA_GPU_SCAN_THREADS; s.order = c->order;
    for (unsigned i = 0; i < OMEGA_GPU_SCAN_ARGS_WORDS; i++) s.cbank[0x380 / 4 + i] = a[i];
    memcpy(s.ranges, c->ranges, sizeof c->ranges); s.nranges = 3;
    if (ws_run(&s, grid_x, 1) != 0) { snprintf(c->err, sizeof c->err, "kernel %d: %s at pc %d", (int)k, s.msg, s.err_pc); return -1; }
    return 0;
}

static int sim_build(SimCtx *c, int mutant) {
    for (int k = 0; k < OMEGA_SCAN_K_COUNT; k++) {
        c->prog[k] = calloc(1, sizeof *c->prog[k]);
        OmegaScanSpec s = { (OmegaScanKind)k, mutant };
        if (!c->prog[k] || omega_gpu_scan_codegen(&s, c->prog[k]) != 0) return -1;
    }
    return 0;
}
static void sim_free(SimCtx *c) { for (int k = 0; k < OMEGA_SCAN_K_COUNT; k++) free(c->prog[k]); }

#define POISON 0x7fa00badu
static void poison(float *p, size_t n) { for (size_t i = 0; i < n; i++) memcpy(&p[i], &(uint32_t){ POISON }, 4); }

/* one reduction through the simulator; 0 when it ran, the result in *out */
static int sim_reduce(SimCtx *c, OmegaGpuRedOp op, const float *x, size_t n, float *out) {
    size_t sw = omega_gpu_scan_reduce_scratch_words(n);
    float *scr = malloc(sw * 4u);
    poison(scr, sw);
    c->ranges[0] = (WsRange){ (uintptr_t)x, (uintptr_t)(x + n) };
    c->ranges[1] = (WsRange){ (uintptr_t)scr, (uintptr_t)(scr + sw) };
    c->ranges[2] = c->ranges[1];
    uint64_t res = 0;
    int rc = omega_gpu_scan_drive_reduce(op, n, (uintptr_t)x, (uintptr_t)scr, &res, sim_launch, c, NULL);
    if (rc == 0) { memcpy(out, (const void *)(uintptr_t)res, 4); if (f2u(*out) == POISON) rc = -9; }
    free(scr);
    return rc;
}

static int sim_prefix(SimCtx *c, const float *x, size_t n, float *y) {
    size_t sw = omega_gpu_scan_prefix_scratch_words(n);
    float *scr = malloc(sw * 4u);
    poison(scr, sw); poison(y, n);
    c->ranges[0] = (WsRange){ (uintptr_t)x, (uintptr_t)(x + n) };
    c->ranges[1] = (WsRange){ (uintptr_t)y, (uintptr_t)(y + n) };
    c->ranges[2] = (WsRange){ (uintptr_t)scr, (uintptr_t)(scr + sw) };
    int rc = omega_gpu_scan_drive_prefix(n, (uintptr_t)x, (uintptr_t)y, (uintptr_t)scr, sim_launch, c, NULL);
    for (size_t i = 0; rc == 0 && i < n; i++) if (f2u(y[i]) == POISON) rc = -9;
    free(scr);
    return rc;
}

static size_t diff(const float *a, const float *b, size_t n, size_t *first) {
    size_t d = 0;
    for (size_t i = 0; i < n; i++) if (!same(a[i], b[i])) { if (!d) *first = i; d++; }
    return d;
}

static void sim_tier(void) {
    char w[600];
    static size_t ns[200]; size_t nn = 0;
    for (size_t n = 1; n <= 70; n++) ns[nn++] = n;
    const size_t more[] = { 255, 256, 257, 1000, 1023, 1024, 1025, 2047, 2048, 2049, 32767, 32768, 32769, 40000, 270000 };
    for (size_t i = 0; i < sizeof more / sizeof more[0]; i++) ns[nn++] = more[i];
    float *x = malloc(270000 * 4u), *y = malloc(270000 * 4u), *ref = malloc(270000 * 4u);
    SimCtx c; memset(&c, 0, sizeof c);
    check(sim_build(&c, 0) == 0, "simulator: all kernels built");
    for (int order = 0; order <= 1; order++) {
        c.order = order;
        size_t bad_red = 0, bad_pre = 0, runs = 0;
        for (size_t t = 0; t < nn; t++) {
            size_t n = ns[t];
            if (order == 1 && n > 40000) continue;
            int kind = (int)(t % 3);
            fill(x, n, kind, n * 7 + (size_t)order);
            for (int op = 0; op < 3; op++) {
                float got, want;
                omega_gpu_reduce_reference((OmegaGpuRedOp)op, x, n, &want);
                int rc = sim_reduce(&c, (OmegaGpuRedOp)op, x, n, &got);
                runs++;
                if (rc != 0 || !same(got, want)) {
                    bad_red++;
                    snprintf(w, sizeof w, "sim order %d %s n = %zu: rc %d got %08x want %08x %s", order, op_name(op), n, rc, f2u(got), f2u(want), c.err);
                    check(0, w);
                }
            }
            omega_gpu_prefix_sum_reference(x, n, ref);
            int rc = sim_prefix(&c, x, n, y);
            size_t first = 0, d = rc == 0 ? diff(y, ref, n, &first) : n;
            runs++;
            if (rc != 0 || d) {
                bad_pre++;
                snprintf(w, sizeof w, "sim order %d prefix n = %zu: rc %d, %zu wrong, first at %zu (%08x vs %08x) %s", order, n, rc, d, first,
                         f2u(y[first]), f2u(ref[first]), c.err);
                check(0, w);
            }
        }
        snprintf(w, sizeof w, "simulator order %d: %zu runs, reductions wrong %zu, prefix sums wrong %zu", order, runs, bad_red, bad_pre);
        check(bad_red == 0 && bad_pre == 0, w);
        printf("%s\n", w);
    }
    sim_free(&c);

    /* red checks: each mutant must change a result the real kernels get right */
    for (int mut = 1; mut <= 3; mut++) {
        SimCtx m; memset(&m, 0, sizeof m);
        int built = sim_build(&m, mut) == 0, caught = 0;
        const size_t n = 1000;
        fill(x, n, 1, 4242);
        if (built && mut == 1) {
            for (int op = 0; op < 3 && !caught; op++) {
                float got = 0, want; omega_gpu_reduce_reference((OmegaGpuRedOp)op, x, n, &want);
                caught = sim_reduce(&m, (OmegaGpuRedOp)op, x, n, &got) != 0 || !same(got, want);
            }
        } else if (built) {
            omega_gpu_prefix_sum_reference(x, n, ref);
            size_t first;
            caught = sim_prefix(&m, x, n, y) != 0 || diff(y, ref, n, &first) != 0;
        }
        snprintf(w, sizeof w, "simulator red check: mutant %d is caught", mut);
        check(built && caught, w);
        sim_free(&m);
    }
    free(x); free(y); free(ref);
}

/* ------------------------------------------------------------------ the chip */

static size_t mem_available(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256]; size_t kb = 0;
    while (f && fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    if (f) fclose(f);
    return kb * 1024u;
}

static float *g_x, *g_y, *g_ref;

static void chip_one(size_t n, int kind, int verbose) {
    char w[600];
    fill(g_x, n, kind, n * 11 + (size_t)kind);
    for (int op = 0; op < 3; op++) {
        float got = 0, want; OmegaGpuScanInfo info;
        omega_gpu_reduce_reference((OmegaGpuRedOp)op, g_x, n, &want);
        int rc = omega_gpu_reduce((OmegaGpuRedOp)op, g_x, n, &got, &info);
        int ok = rc == 0 && same(got, want);
        if (!ok || verbose) {
            snprintf(w, sizeof w, "chip %s n = %zu inputs %d: %s, got %08x want %08x, %u launches, kernels %.3f ms%s%s", op_name(op), n, kind,
                     omega_gpu_scan_rc_name(rc), f2u(got), f2u(want), info.launches, info.elapsed_ns / 1e6, rc ? ": " : "",
                     rc ? omega_gpu_session_last_error() : "");
            if (ok) { g_checks++; printf("ok: %s\n", w); } else check(0, w);
        } else g_checks++;
    }
    OmegaGpuScanInfo info;
    omega_gpu_prefix_sum_reference(g_x, n, g_ref);
    int rc = omega_gpu_prefix_sum(g_x, n, g_y, &info);
    size_t first = 0, d = rc == 0 ? diff(g_y, g_ref, n, &first) : n;
    if (rc || d || verbose) {
        snprintf(w, sizeof w, "chip prefix n = %zu inputs %d: %s, %zu wrong (first at %zu), %u launches, kernels %.3f ms%s%s", n, kind,
                 omega_gpu_scan_rc_name(rc), d, first, info.launches, info.elapsed_ns / 1e6, rc ? ": " : "", rc ? omega_gpu_session_last_error() : "");
        if (!rc && !d) { g_checks++; printf("ok: %s\n", w); } else check(0, w);
    } else g_checks++;
}

static void chip_tier(unsigned max_log2) {
    char w[200];
    size_t cap = (size_t)1 << max_log2;
    if (cap > OMEGA_GPU_SCAN_MAX_N) cap = OMEGA_GPU_SCAN_MAX_N;
    /* host: input, output, reference (3 x 4n); device buffer: in + out + scratch (~ 8n); plus 8 GiB for everyone else */
    size_t need = cap * 4u * 5u + ((size_t)8 << 30), have = mem_available();
    if (have < need) {
        printf("NOT_RUN: MemAvailable %zu MiB < %zu MiB needed for n up to %zu\n", have >> 20, need >> 20, cap);
        exit(2);
    }
    g_x = malloc(cap * 4u); g_y = malloc(cap * 4u); g_ref = malloc(cap * 4u);
    if (!g_x || !g_y || !g_ref) { printf("NOT_RUN: host allocation\n"); exit(2); }

    int before = g_failed;
    for (size_t n = 1; n <= 2048; n++) chip_one(n, (int)(n % 3), 0);
    printf("every n from 1 to 2048 (sum, max, min, prefix): %s\n", g_failed == before ? "PASS" : "FAIL");
    before = g_failed;
    for (unsigned k = 11; k <= max_log2 && k <= 28; k++) {
        size_t p = (size_t)1 << k;
        chip_one(p - 1, 1, 0);
        chip_one(p, 2, k >= 20);
        if (p + 1 <= cap) chip_one(p + 1, 1, 0);
    }
    printf("2^k - 1, 2^k, 2^k + 1 up to 2^%u: %s\n", max_log2, g_failed == before ? "PASS" : "FAIL");
    before = g_failed;
    g_rng = 0xc3c3c3c3ull;
    for (int i = 0; i < 64; i++) {
        size_t n = 2049 + (((size_t)rnd() << 16) ^ rnd()) % (cap > (1u << 24) ? (1u << 24) : cap - 2048);
        chip_one(n, i % 3, 0);
    }
    printf("64 random lengths: %s\n", g_failed == before ? "PASS" : "FAIL");

    for (int mut = 1; mut <= 3; mut++) {
        omega_gpu_scan_test_set_mutant(mut);
        const size_t n = 1000;
        fill(g_x, n, 1, 4242);
        int caught = 0;
        if (mut == 1) {
            for (int op = 0; op < 3 && !caught; op++) {
                float got = 0, want; omega_gpu_reduce_reference((OmegaGpuRedOp)op, g_x, n, &want);
                caught = omega_gpu_reduce((OmegaGpuRedOp)op, g_x, n, &got, NULL) != 0 || !same(got, want);
            }
        } else {
            size_t first;
            omega_gpu_prefix_sum_reference(g_x, n, g_ref);
            caught = omega_gpu_prefix_sum(g_x, n, g_y, NULL) != 0 || diff(g_y, g_ref, n, &first) != 0;
        }
        snprintf(w, sizeof w, "chip red check: mutant %d is caught", mut);
        check(caught, w);
    }
    omega_gpu_scan_test_set_mutant(0);
    fill(g_x, 1000, 1, 4242);
    omega_gpu_prefix_sum_reference(g_x, 1000, g_ref);
    size_t first;
    check(omega_gpu_prefix_sum(g_x, 1000, g_y, NULL) == 0 && diff(g_y, g_ref, 1000, &first) == 0, "the real kernels are right again after the red checks");
    omega_gpu_session_close(); /* the kernels go with the device (close hook); the next call builds them on a new one */
    check(omega_gpu_prefix_sum(g_x, 1000, g_y, NULL) == 0 && diff(g_y, g_ref, 1000, &first) == 0, "after the device closed and opened again the kernels are rebuilt and right");
    free(g_x); free(g_y); free(g_ref);
}

int main(int argc, char **argv) {
    const char *tier = "chip";
    unsigned max_log2 = 28;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) tier = "host";
        else if (!strcmp(argv[i], "--sim")) tier = "sim";
        else if (!strcmp(argv[i], "--max-log2") && i + 1 < argc) max_log2 = (unsigned)atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [--host-only | --sim] [--max-log2 K]\n", argv[0]); return 2; }
    }
    if (max_log2 < 11 || max_log2 > 28) { fprintf(stderr, "--max-log2 must be 11..28\n"); return 2; }
    if (!strcmp(tier, "host")) host_only();
    else if (!strcmp(tier, "sim")) { host_only(); sim_tier(); }
    else chip_tier(max_log2);
    if (g_failed) printf("C3 scan %s: FAIL (%d of %d)\n", tier, g_failed, g_checks);
    else printf("C3 scan %s: PASS (%d checks)\n", tier, g_checks);
    return g_failed ? 1 : 0;
}
