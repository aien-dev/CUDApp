/* gpu_tcgemm_test.c (OWN-30B T3, 2026-10-13): tensor-core grouped FP8 GEMM (omega_blackwell_codegen_tc.h).
 *   --host-only  program statistics and register budgets for the production shapes; the C emulation (exact HMMA model,
 *                tests/bw_hmma_model.h) against a double oracle; permutation and decode checks
 *   --sim        the generated kernel in the warp simulator (both fragment orders) bit-identical to the C emulation;
 *                mutants 1-4 caught (a simulator error counts as a FAIL, never as caught)
 *   --chip       (under flock /tmp/aien-gb10.lock) through omega_gpu_tcgemm_api: bit-identity to the emulation, second call
 *                identical, mutants caught on the chip, timing medians (TFLOPS, GB/s) at large M
 * Verdict line last: "GPU_TCGEMM: PASS|FAIL". */
#include "omega_blackwell_codegen_tc.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_session.h"
#include "bw_warp_sim_tc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static const HmmaModel GB10 = { 16, 2, 0, 2, 0 };
static uint64_t rng = 0x243f6a8885a308d3ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float frnd(void) { return ((float)(rnd() & 0xffffff) / 16777216.0f) * 2.0f - 1.0f; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }

/* a problem: E experts of n x k e4m3, M sorted rows (expert ids non-decreasing), X source rows */
typedef struct {
    uint32_t E, n, k, M, xr;
    uint8_t *w;          /* E * n * k */
    uint16_t *sbf;       /* E * (n/128) * (k/128) bf16 scale_inv */
    float *sf;           /* folded f32 * 2^56 */
    float *x;            /* xr * k */
    uint16_t *xp;        /* xr * k stored (permuted bf16 * 2^64) */
    uint32_t *eid, *src; /* M */
} Prob;

static uint8_t rand_e4m3(void) { for (;;) { uint8_t c = (uint8_t)rnd(); if ((c & 0x7f) != 0x7f) return c; } }
static uint16_t bf16_trunc(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }

static void prob_make(Prob *p, uint32_t E, uint32_t n, uint32_t k, uint32_t M, uint32_t xr, int skewed) {
    memset(p, 0, sizeof *p);
    p->E = E; p->n = n; p->k = k; p->M = M; p->xr = xr;
    size_t wn = (size_t)E * n * k, sn = (size_t)E * (n / 128) * (k / 128);
    p->w = malloc(wn); p->sbf = malloc(sn * 2); p->sf = malloc(sn * 4); p->x = malloc((size_t)xr * k * 4); p->xp = malloc((size_t)xr * k * 2);
    p->eid = malloc(M * 4); p->src = malloc(M * 4);
    for (size_t i = 0; i < wn; i++) p->w[i] = rand_e4m3();
    for (size_t i = 0; i < sn; i++) p->sbf[i] = bf16_trunc(ldexpf(1.0f + (float)(rnd() & 127) / 128.0f, -(int)(rnd() % 12) - 4));
    if (omega_fp8_pack_scales(p->sbf, sn, p->sf, 0) != 0) { printf("scale pack failed\n"); exit(2); }
    for (size_t i = 0; i < (size_t)xr * k; i++) p->x[i] = frnd() * (rnd() % 16 == 0 ? 8.0f : 1.0f);
    for (uint32_t r = 0; r < xr; r++) omega_tcgemm_permute_x(p->x + (size_t)r * k, k, p->xp + (size_t)r * k);
    /* expert ids: sorted; skewed puts many rows on a few experts (multi-tile segments) */
    for (uint32_t i = 0; i < M; i++) { uint32_t e = skewed ? (uint32_t)((uint64_t)i * i / ((uint64_t)M * M / E + 1)) : (uint32_t)((uint64_t)i * E / M); if (e >= E) e = E - 1; p->eid[i] = e; p->src[i] = rnd() % xr; }
}
static void prob_free(Prob *p) { free(p->w); free(p->sbf); free(p->sf); free(p->x); free(p->xp); free(p->eid); free(p->src); }

/* stored index of element p in a permuted row: group (0,2,1,3) */
static uint32_t stored_ix(uint32_t p) { static const uint32_t inv[4] = { 0, 2, 1, 3 }; return (p & ~3u) | inv[p & 3u]; }

/* C emulation: exactly the kernel's arithmetic (8 HMMA steps per 128 block from a zero temporary, then fma with S') */
static void emulate(const Prob *p, float *y) {
    const uint32_t n = p->n, k = p->k, KB = k / 128;
    for (uint32_t i = 0; i < p->M; i++) {
        const uint16_t *xs = p->xp + (size_t)p->src[i] * k;
        for (uint32_t c = 0; c < n; c++) {
            const uint8_t *wr = p->w + ((size_t)p->eid[i] * n + c) * k;
            const float *sr = p->sf + ((size_t)p->eid[i] * (n / 128) + c / 128) * KB;
            float acc = 0.0f;
            for (uint32_t kb = 0; kb < KB; kb++) {
                float tmp = 0.0f;
                for (uint32_t ch = 0; ch < 2; ch++) for (uint32_t s = 0; s < 4; s++) {
                    float a[16], b[16]; int m = 0;
                    for (uint32_t t = 0; t < 4; t++) for (uint32_t j = 0; j < 4; j++) {
                        uint32_t e = kb * 128 + ch * 64 + 16 * t + 4 * s + j;
                        a[m] = u2f((uint32_t)xs[stored_ix(e)] << 16); b[m] = u2f((uint32_t)omega_tcgemm_w_bf16(wr[e]) << 16); m++;
                    }
                    tmp = hm_dot16(&GB10, a, b, tmp);
                }
                acc = fmaf(tmp, sr[kb], acc);
            }
            y[(size_t)i * n + c] = acc;
        }
    }
}

/* tiles: split every expert segment into runs of at most bm rows; rowidx padded by bm entries */
typedef struct { uint64_t w_va, s_va; uint32_t row0, cnt, pad[2]; } Tile;
static uint32_t make_tiles(const Prob *p, uint32_t bm, Tile *t, const uint8_t *wbase, const float *sbase) {
    uint32_t nt = 0;
    for (uint32_t i = 0; i < p->M;) {
        uint32_t j = i; while (j < p->M && p->eid[j] == p->eid[i]) j++;
        for (uint32_t r = i; r < j; r += bm) {
            memset(&t[nt], 0, sizeof t[nt]);
            t[nt].w_va = (uint64_t)(uintptr_t)(wbase + (size_t)p->eid[i] * p->n * p->k);
            t[nt].s_va = (uint64_t)(uintptr_t)(sbase + (size_t)p->eid[i] * (p->n / 128) * (p->k / 128));
            t[nt].row0 = r; t[nt].cnt = j - r < bm ? j - r : bm; nt++;
        }
        i = j;
    }
    return nt;
}

static int host_only(void) {
    int fail = 0;
    /* decode: every non-NaN code, the bf16 bits are the value * 2^-120 exactly */
    for (unsigned c = 0; c < 256; c++) {
        if ((c & 0x7f) == 0x7f) continue;
        unsigned e = (c >> 3) & 15, m = c & 7; double v = e ? ldexp(1.0 + m / 8.0, (int)e - 7) : ldexp(m / 8.0, -6); if (c & 0x80) v = -v;
        if ((double)u2f((uint32_t)omega_tcgemm_w_bf16((uint8_t)c) << 16) != ldexp(v, -120)) { printf("decode code %02x wrong\n", c); fail++; }
    }
    /* production shapes: instructions and registers (declared = peak + 2 rounded to 8 <= 128; 32 * warps * declared <= 65536) */
    static const OmegaTcGemmSpec shapes[] = {
        { 2048, 4, 2, 8, 0 }, { 768, 4, 2, 8, 0 }, { 4096, 4, 2, 8, 0 }, { 2048, 4, 1, 8, 0 }, { 2048, 2, 2, 4, 0 }, { 2048, 4, 2, 4, 0 }, { 768, 4, 2, 4, 0 },
    };
    static BlackwellIRProgram prog;
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
        int rc = omega_blackwell_codegen_tcgemm_prog(&shapes[i], &prog);
        uint32_t peak = rc == 0 ? prog.regalloc.peak_gpr_usage : 0, decl = omega_bw_gpr_declared(peak);
        int ok = rc == 0 && decl <= 128u && 32u * shapes[i].warps * decl <= 65536u;
        printf("shape k %u rt %u ct %u warps %u: rc %d, %zu instructions, peak %u, declared %u, shared %u, span %u: %s\n", shapes[i].k, shapes[i].rt, shapes[i].ct, shapes[i].warps,
               rc, rc == 0 ? prog.count : 0, peak, decl, omega_tcgemm_shared_bytes(&shapes[i]), omega_tcgemm_span(&shapes[i]), ok ? "ok" : "REFUSED");
        if (i < 2 && !ok) fail++;
    }
    /* emulation vs double oracle (x as stored bf16, true e4m3 and bf16 scale): relative to the sum of |terms| */
    Prob p; prob_make(&p, 3, 256, 512, 40, 17, 0);
    float *y = malloc((size_t)p.M * p.n * 4); emulate(&p, y);
    double worst = 0;
    for (uint32_t i = 0; i < p.M; i++) for (uint32_t c = 0; c < p.n; c++) {
        double s = 0, mag = 0;
        for (uint32_t e = 0; e < p.k; e++) {
            double wv = (double)u2f((uint32_t)omega_tcgemm_w_bf16(p.w[((size_t)p.eid[i] * p.n + c) * p.k + e]) << 16) * ldexp(1.0, 120);
            double xv = (double)u2f((uint32_t)p.xp[(size_t)p.src[i] * p.k + stored_ix(e)] << 16) * ldexp(1.0, -64);
            double sv = (double)u2f((uint32_t)p.sbf[((size_t)p.eid[i] * (p.n / 128) + c / 128) * (p.k / 128) + e / 128] << 16);
            s += wv * xv * sv; mag += fabs(wv * xv * sv);
        }
        double rel = fabs(y[(size_t)i * p.n + c] - s) / (mag + 1e-30); if (rel > worst) worst = rel;
    }
    printf("emulation vs double oracle: worst |err| / sum|terms| = %.3g (%s, bound 1e-6)\n", worst, worst < 1e-6 ? "ok" : "FAIL");
    if (worst >= 1e-6) fail++;
    free(y); prob_free(&p);
    printf("host-only: %s\n", fail ? "FAIL" : "PASS");
    return fail;
}

/* run the kernel in the simulator on p; returns 0 and fills y, or -1 with msg */
static int sim_run(const Prob *p, const OmegaTcGemmSpec *sp, int order, float *y, char *msg, size_t msgn) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_tcgemm_prog(sp, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    const uint32_t bm = 16u * sp->rt, span = omega_tcgemm_span(sp);
    if (p->n % span) { snprintf(msg, msgn, "n %% span"); return -1; }
    Tile *tiles = calloc(p->M + 1, sizeof *tiles);
    uint32_t nt = make_tiles(p, bm, tiles, p->w, p->sf);
    uint32_t *ridx = malloc((p->M + bm) * 4); for (uint32_t i = 0; i < p->M + bm; i++) ridx[i] = i < p->M ? p->src[i] : 0;
    for (size_t i = 0; i < (size_t)p->M * p->n; i++) y[i] = -999.0f;
    uint8_t *smem = malloc(omega_tcgemm_shared_bytes(sp));
    static WsSim s; memset(&s, 0, sizeof s);
    s.p = &prog; s.threads = 32u * sp->warps; s.smem = smem; s.smem_bytes = omega_tcgemm_shared_bytes(sp); s.order = order;
    uint64_t a[5] = { (uintptr_t)tiles, (uintptr_t)ridx, (uintptr_t)p->xp, (uintptr_t)y, p->n };
    for (int i = 0; i < 4; i++) { s.cbank[(0x380 + 8 * i) / 4] = (uint32_t)a[i]; s.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(a[i] >> 32); }
    s.cbank[0x3a0 / 4] = p->n;
    s.ranges[0].lo = (uintptr_t)tiles; s.ranges[0].hi = (uintptr_t)(tiles + nt);
    s.ranges[1].lo = (uintptr_t)ridx; s.ranges[1].hi = (uintptr_t)(ridx + p->M + bm);
    s.ranges[2].lo = (uintptr_t)p->xp; s.ranges[2].hi = (uintptr_t)(p->xp + (size_t)p->xr * p->k);
    s.ranges[3].lo = (uintptr_t)y; s.ranges[3].hi = (uintptr_t)(y + (size_t)p->M * p->n);
    s.ranges[4].lo = (uintptr_t)p->w; s.ranges[4].hi = (uintptr_t)(p->w + (size_t)p->E * p->n * p->k);
    s.ranges[5].lo = (uintptr_t)p->sf; s.ranges[5].hi = (uintptr_t)(p->sf + (size_t)p->E * (p->n / 128) * (p->k / 128));
    s.nranges = 6;
    int err = ws_run(&s, p->n / span, nt);
    if (err) snprintf(msg, msgn, "simulator %s", s.msg);
    free(tiles); free(ridx); free(smem);
    return err ? -1 : 0;
}

static int sim_tests(void) {
    int fail = 0;
    static const struct { uint32_t E, n, k, M, xr; int skew; OmegaTcGemmSpec sp; } pts[] = {
        { 3, 256, 256, 40, 13, 0, { 256, 4, 2, 8, 0 } },
        { 2, 128, 384, 70, 9, 1, { 384, 4, 2, 8, 0 } },   /* three blocks, a multi-tile segment */
        { 4, 128, 128, 37, 11, 0, { 128, 2, 2, 4, 0 } },
        { 2, 256, 256, 20, 7, 0, { 256, 4, 1, 8, 0 } },
        { 2, 128, 256, 33, 5, 1, { 256, 1, 4, 4, 0 } },
        { 1, 512, 256, 64, 64, 0, { 256, 4, 2, 8, 0 } },  /* dense: one expert, four CTAs across n */
    };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        Prob p; prob_make(&p, pts[i].E, pts[i].n, pts[i].k, pts[i].M, pts[i].xr, pts[i].skew);
        float *ye = malloc((size_t)p.M * p.n * 4), *ys = malloc((size_t)p.M * p.n * 4), *y0 = malloc((size_t)p.M * p.n * 4);
        emulate(&p, ye);
        for (int order = 0; order < 2; order++) {
            char msg[240] = ""; int rc = sim_run(&p, &pts[i].sp, order, ys, msg, sizeof msg);
            size_t bad = 0, poison = 0;
            if (rc == 0) for (size_t q = 0; q < (size_t)p.M * p.n; q++) { if (ys[q] == -999.0f) poison++; if (memcmp(&ys[q], &ye[q], 4)) bad++; }
            if (order == 0 && rc == 0) memcpy(y0, ys, (size_t)p.M * p.n * 4);
            int same = order == 0 || (rc == 0 && memcmp(y0, ys, (size_t)p.M * p.n * 4) == 0);
            printf("sim E %u n %u k %u M %u rt %u ct %u warps %u order %d: %s %s%zu of %zu differ from the emulation, %zu unwritten%s\n", p.E, p.n, p.k, p.M, pts[i].sp.rt, pts[i].sp.ct, pts[i].sp.warps, order,
                   rc ? "ERROR" : (bad || poison || !same) ? "MISMATCH" : "ok", msg, bad, (size_t)p.M * p.n, poison, bad || poison ? "" : " (bit-identical)");
            if (rc || bad || poison || !same) fail++;
        }
        free(ye); free(ys); free(y0); prob_free(&p);
    }
    /* mutants: each must run cleanly in the simulator and differ from the emulation */
    for (int m = 1; m <= 4; m++) {
        Prob p; prob_make(&p, 2, 128, 384, 50, 9, 1);
        float *ye = malloc((size_t)p.M * p.n * 4), *ys = malloc((size_t)p.M * p.n * 4);
        emulate(&p, ye);
        OmegaTcGemmSpec sp = { 384, 4, 2, 8, m };
        char msg[240] = ""; int rc = sim_run(&p, &sp, 0, ys, msg, sizeof msg);
        size_t bad = 0; if (rc == 0) for (size_t q = 0; q < (size_t)p.M * p.n; q++) if (memcmp(&ys[q], &ye[q], 4)) bad++;
        printf("sim mutant %d: %s %s(%zu of %zu differ)\n", m, rc ? "ERROR (counts as FAIL)" : bad ? "caught" : "NOT CAUGHT", msg, bad, (size_t)p.M * p.n);
        if (rc || !bad) fail++;
        free(ye); free(ys); prob_free(&p);
    }
    printf("sim: %s\n", fail ? "FAIL" : "PASS");
    return fail;
}

int chip_tests(int argc, char **argv);

int main(int argc, char **argv) {
    if (argc > 2 && strcmp(argv[1], "--dump") == 0) { /* raw code words for nvdisasm -b SM121 (offline check only): --dump file [k rt ct warps mutant] */
        static BlackwellIRProgram dp; static uint8_t buf[OMEGA_BW_CODE_CAP_MAX]; size_t len = 0;
        OmegaTcGemmSpec sp = { argc > 3 ? (uint32_t)atoi(argv[3]) : 2048u, argc > 4 ? (uint32_t)atoi(argv[4]) : 4u, argc > 5 ? (uint32_t)atoi(argv[5]) : 2u, argc > 6 ? (uint32_t)atoi(argv[6]) : 8u, argc > 7 ? atoi(argv[7]) : 0 };
        if (omega_blackwell_codegen_tcgemm_prog(&sp, &dp) != 0 || omega_bw_encode_program(&dp, buf, sizeof buf, &len) != 0) return 1;
        FILE *f = fopen(argv[2], "wb"); if (!f) return 1; fwrite(buf, 1, len, f); fclose(f); printf("peak %u, %zu bytes\n", dp.regalloc.peak_gpr_usage, len); return 0;
    }
    int host = 0, sim = 0, chip = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) host = 1;
        else if (!strcmp(argv[i], "--sim")) sim = 1;
        else if (!strcmp(argv[i], "--chip")) chip = 1;
    }
    if (!host && !sim && !chip) host = sim = 1;
    int fail = 0;
    if (host) fail += host_only();
    if (sim && !fail) fail += sim_tests();
    if (chip && !fail) fail += chip_tests(argc, argv);
    printf("GPU_TCGEMM: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}

/* chip part: added with the API (omega_gpu_tcgemm_api.h) */
int chip_tests(int argc, char **argv) { (void)argc; (void)argv; (void)now_ms; printf("chip: not built yet\n"); return 1; }
