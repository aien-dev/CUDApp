/* gpu_gemm_test.c -- row C3 of docs/OMEGA.md: matrix multiply of any shape, BF16 and FP8 (e4m3), both layouts of A and B,
 * on the GB10 tensor cores, checked bit for bit against a host model of the same tensor-core steps (src/omega_gpu_gemm.h).
 *
 *   --host-only  no device: refusals, every kernel (both types, several shapes of tile, the red-check mutants) through our
 *                code generator and encoder, register counts
 *   --sim        the real kernels in the warp simulator (tensor-core ops computed with the chip-exact models), small shapes
 *                over every edge case, all four layouts, three tile configurations; the three mutants must be caught
 *   (no flag)    the GB10: at least 200 shapes with M, N, K from 1 to 8192, both types, all four layouts. Every output word
 *                must be written; shapes up to 2^27 multiply-adds are compared in full, larger ones on 4096 sampled
 *                elements plus the last row and column. Speed: 2 M N K / kernel time, as a fraction of the measured
 *                tensor-core ceiling (BF16 HMMA 115.36 TFLOPS, evidence OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z;
 *                FP8 QMMA 237.52 TFLOPS, L151 as quoted in evidence/L403-roof-20261022). The mutants must be caught.
 *                --quick: 24 shapes, for a first look.
 * Last line: "C3 gemm <tier>: PASS (<checks> checks)" or "... FAIL (<failed> of <checks>)". */
#include "omega_gpu_gemm.h"
#include "omega_gpu_session.h"
#include "bw_warp_sim_tc.h"

#include <inttypes.h>

static int g_checks, g_failed;
static void check(int ok, const char *what) {
    g_checks++;
    if (!ok) { g_failed++; printf("FAIL: %s\n", what); }
}

static const HmmaModel HM = { 16, 2, 0, 2, 0 };  /* chip-exact, evidence OWN30B-T3-HMMA-MODEL */
static const QmmaModel QM = { 32, 26, 0, 2, 0 }; /* chip-exact, evidence OWN30B-L1G-QMMA-PFGEMM-c742aa13 */
static const char *tname(OmegaGpuGemmType t) { return t == OMEGA_GPU_GEMM_BF16 ? "bf16" : "e4m3"; }
static const char *lname(unsigned l) { static const char *n[4] = { "A row B row", "A col B row", "A row B col", "A col B col" }; return n[l & 3u]; }

static uint64_t g_rng = 1;
static uint32_t rnd(void) { g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t)(g_rng >> 33); }

/* inputs: bf16 values 0 or +-(1 + m/128) 2^e, e in -8..8; e4m3 any code except the two NaNs */
static void fill(OmegaGpuGemmType t, void *p, uint64_t n, uint64_t seed) {
    g_rng = seed * 0x2545f4914f6cdd1dull + 7;
    if (t == OMEGA_GPU_GEMM_BF16) {
        uint16_t *h = p;
        for (uint64_t i = 0; i < n; i++) {
            uint32_t r = rnd();
            h[i] = r % 61u == 0 ? 0 : (uint16_t)(((r >> 8) & 1u) << 15 | (uint32_t)(127 + (int)((r >> 9) % 17u) - 8) << 7 | (r >> 16 & 0x7fu));
        }
    } else {
        uint8_t *b = p;
        for (uint64_t i = 0; i < n; i++) { uint8_t c = (uint8_t)rnd(); b[i] = (c & 0x7fu) == 0x7fu ? (uint8_t)(c ^ 1u) : c; }
    }
}

static float val(OmegaGpuGemmType t, const void *p, uint64_t i) {
    if (t == OMEGA_GPU_GEMM_BF16) { uint32_t u = (uint32_t)((const uint16_t *)p)[i] << 16; float f; memcpy(&f, &u, 4); return f; }
    return qm_e4m3(((const uint8_t *)p)[i]);
}

/* the host model: one tensor-core step per 16 (bf16) or 32 (e4m3) values of k, in increasing k, from +0 */
static float ref_elem(OmegaGpuGemmType t, unsigned lay, uint32_t M, uint32_t N, uint32_t K, const void *A, const void *B, uint32_t m, uint32_t n) {
    const uint32_t ks = t == OMEGA_GPU_GEMM_BF16 ? 16u : 32u;
    float acc = 0.0f;
    for (uint32_t k0 = 0; k0 < K; k0 += ks) {
        float av[32], bv[32];
        for (uint32_t k = 0; k < ks; k++) {
            uint32_t kk = k0 + k;
            if (kk >= K) { av[k] = bv[k] = 0.0f; continue; }
            av[k] = val(t, A, lay & OMEGA_GPU_GEMM_A_COL ? (uint64_t)kk * M + m : (uint64_t)m * K + kk);
            bv[k] = val(t, B, lay & OMEGA_GPU_GEMM_B_COL ? (uint64_t)n * K + kk : (uint64_t)kk * N + n);
        }
        acc = t == OMEGA_GPU_GEMM_BF16 ? hm_dot16(&HM, av, bv, acc) : qm_dot32(&QM, av, bv, acc);
    }
    return acc;
}

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* compare: every element when M N K <= 2^27, else 4096 samples + the last row and column; returns the mismatches */
static uint64_t compare(OmegaGpuGemmType t, unsigned lay, uint32_t M, uint32_t N, uint32_t K, const void *A, const void *B, const float *C,
                        uint64_t *checked, char *first, size_t firstn) {
    uint64_t bad = 0; *checked = 0; first[0] = 0;
    #define ONE(mm, nn) do { uint32_t m_ = (mm), n_ = (nn); float w_ = ref_elem(t, lay, M, N, K, A, B, m_, n_), g_ = C[(uint64_t)m_ * N + n_]; (*checked)++; \
        if (f2u(w_) != f2u(g_)) { if (!bad) snprintf(first, firstn, "C[%u][%u] = %08x, model %08x", m_, n_, f2u(g_), f2u(w_)); bad++; } } while (0)
    if ((uint64_t)M * N * K <= (1ull << 27)) {
        for (uint32_t m = 0; m < M; m++) for (uint32_t n = 0; n < N; n++) ONE(m, n);
    } else {
        g_rng = (uint64_t)M * 31 + N * 7 + K;
        for (int i = 0; i < 4096; i++) ONE(rnd() % M, rnd() % N);
        for (uint32_t n = 0; n < N; n += N > 512 ? N / 512 : 1) ONE(M - 1, n);
        for (uint32_t m = 0; m < M; m += M > 512 ? M / 512 : 1) ONE(m, N - 1);
        ONE(M - 1, N - 1);
    }
    #undef ONE
    return bad;
}

/* ------------------------------------------------------------------ host only */

static void host_only(void) {
    char w[200];
    uint16_t a[4] = { 0 }, b[4] = { 0 }; float c[4];
    check(omega_gpu_gemm(OMEGA_GPU_GEMM_BF16, 0, 0, 1, 1, a, b, c, NULL) == OMEGA_GPU_GEMM_BAD_ARGS, "M = 0 refused");
    check(omega_gpu_gemm(OMEGA_GPU_GEMM_BF16, 0, 1, 1, 0, a, b, c, NULL) == OMEGA_GPU_GEMM_BAD_ARGS, "K = 0 refused");
    check(omega_gpu_gemm(OMEGA_GPU_GEMM_BF16, 0, OMEGA_GPU_GEMM_MAX_DIM + 1, 1, 1, a, b, c, NULL) == OMEGA_GPU_GEMM_TOO_LARGE, "M past the maximum refused");
    check(omega_gpu_gemm(OMEGA_GPU_GEMM_BF16, 4, 1, 1, 1, a, b, c, NULL) == OMEGA_GPU_GEMM_BAD_ARGS, "unknown layout bits refused");
    check(omega_gpu_gemm((OmegaGpuGemmType)2, 0, 1, 1, 1, a, b, c, NULL) == OMEGA_GPU_GEMM_BAD_ARGS, "unknown type refused");
    check(omega_gpu_gemm(OMEGA_GPU_GEMM_E4M3, 0, 1, 1, 1, NULL, b, c, NULL) == OMEGA_GPU_GEMM_BAD_ARGS, "NULL A refused");
    OmegaGemmCfg bad; omega_gpu_gemm_default_cfg(OMEGA_GPU_GEMM_BF16, &bad); bad.rt = 3;
    check(!omega_gpu_gemm_cfg_ok(&bad), "a tile of 3 row tiles refused");

    /* the model itself is a multiply: within 2^-20 of the exact sum, relative to the sum of the magnitudes */
    for (int t = 0; t < 2; t++) for (unsigned lay = 0; lay < 4; lay++) {
        enum { MM = 23, NN = 19, KK = 300 };
        static uint16_t A[MM * KK], B[KK * NN];
        fill((OmegaGpuGemmType)t, A, MM * KK, 5 + lay); fill((OmegaGpuGemmType)t, B, KK * NN, 9 + lay);
        double worst = 0;
        for (uint32_t m = 0; m < MM; m++) for (uint32_t n = 0; n < NN; n++) {
            double ex = 0, mag = 0;
            for (uint32_t k = 0; k < KK; k++) {
                double p = (double)val((OmegaGpuGemmType)t, A, lay & OMEGA_GPU_GEMM_A_COL ? (uint64_t)k * MM + m : (uint64_t)m * KK + k) *
                           (double)val((OmegaGpuGemmType)t, B, lay & OMEGA_GPU_GEMM_B_COL ? (uint64_t)n * KK + k : (uint64_t)k * NN + n);
                ex += p; mag += fabs(p);
            }
            double d = fabs((double)ref_elem((OmegaGpuGemmType)t, lay, MM, NN, KK, A, B, m, n) - ex) / (mag > 0 ? mag : 1);
            if (d > worst) worst = d;
        }
        snprintf(w, sizeof w, "the %s model is a multiply (%s): worst error %.3g of the magnitude sum", tname((OmegaGpuGemmType)t), lname(lay), worst);
        check(worst < ldexp(1.0, -20), w);
        printf("%s\n", w);
    }

    const OmegaGemmCfg shapes[] = { { 0, 2, 4, 2, 2, 4, 0 }, { 0, 1, 1, 1, 1, 4, 0 }, { 0, 4, 1, 2, 2, 2, 0 }, { 0, 1, 4, 4, 1, 2, 0 }, { 0, 2, 2, 1, 4, 8, 0 } };
    OmegaGemmCfg big; omega_gpu_gemm_default_cfg(OMEGA_GPU_GEMM_BF16, &big); big.rt = 4; big.ct = 2;
    check(!omega_gpu_gemm_cfg_ok(&big), "a tile past the register budget (rt 4 ct 2) refused");
    for (int t = 0; t < 2; t++)
        for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++)
            for (int mut = 0; mut <= 3; mut++)
                for (int k = 0; k < OMEGA_GEMM_K_COUNT; k++) {
                    OmegaGemmCfg c = shapes[s]; c.type = (OmegaGpuGemmType)t; c.mutant = mut;
                    BlackwellIRProgram *p = calloc(1, sizeof *p);
                    int rc = omega_gpu_gemm_codegen(&c, (OmegaGemmKind)k, p);
                    size_t bytes = rc == 0 ? omega_bw_program_code_bytes(p) : 0;
                    uint8_t *code = bytes ? malloc(bytes) : NULL;
                    OmegaBlackwellKernel kk; memset(&kk, 0, sizeof kk);
                    int enc = code ? omega_blackwell_kernel_encode_into(p, code, bytes, &kk) : -1;
                    int np = enc == 0 ? omega_qmma_patch_code(p, code, bytes) : -1;
                    snprintf(w, sizeof w, "%s rt %u ct %u wm %u wn %u kernel %d mutant %d: codegen %d, encode %d, qmma patches %d, %u gprs", tname(c.type),
                             c.rt, c.ct, c.wm, c.wn, k, mut, rc, enc, np, (unsigned)kk.gpr_count);
                    check(rc == 0 && enc == 0 && kk.code_size == bytes && np >= 0 && (t == 1 || np == 0) && omega_bw_gpr_declared(kk.gpr_count) <= 255u, w);
                    if (mut == 0) printf("%s\n", w);
                    free(code); free(p);
                }
}

/* ------------------------------------------------------------------ the simulator */

typedef struct {
    BlackwellIRProgram *prog[OMEGA_GEMM_K_COUNT];
    WsRange ranges[5];
    int order;
    char err[300];
} SimCtx;

static int sim_launch(void *ctx, OmegaGemmKind k, const uint32_t a[OMEGA_GPU_GEMM_ARGS_WORDS], uint32_t gx, uint32_t gy, uint32_t threads) {
    SimCtx *c = ctx;
    static WsSim s;
    memset(&s, 0, sizeof s);
    s.p = c->prog[k]; s.threads = threads; s.order = c->order;
    for (unsigned i = 0; i < OMEGA_GPU_GEMM_ARGS_WORDS; i++) s.cbank[0x380 / 4 + i] = a[i];
    memcpy(s.ranges, c->ranges, sizeof c->ranges); s.nranges = 5;
    if (ws_run(&s, gx, gy) != 0) { snprintf(c->err, sizeof c->err, "kernel %d: %s at pc %d", (int)k, s.msg, s.err_pc); return -1; }
    return 0;
}

static int sim_build(SimCtx *c, const OmegaGemmCfg *cfg) {
    for (int k = 0; k < OMEGA_GEMM_K_COUNT; k++) {
        c->prog[k] = calloc(1, sizeof *c->prog[k]);
        if (!c->prog[k] || omega_gpu_gemm_codegen(cfg, (OmegaGemmKind)k, c->prog[k]) != 0) return -1;
    }
    return 0;
}
static void sim_free(SimCtx *c) { for (int k = 0; k < OMEGA_GEMM_K_COUNT; k++) { free(c->prog[k]); c->prog[k] = NULL; } }

/* one multiply through the simulator; 0 when it ran and every output word was written */
static int sim_gemm(SimCtx *c, const OmegaGemmCfg *cfg, unsigned lay, uint32_t M, uint32_t N, uint32_t K, const void *A, const void *B, float *C) {
    OmegaGemmPlan p;
    if (omega_gpu_gemm_plan(cfg, M, N, K, &p) != 0) return -2;
    uint8_t *ap = malloc(p.ap_bytes), *bp = malloc(p.bp_bytes);
    memset(ap, 0xa5, p.ap_bytes); memset(bp, 0x5a, p.bp_bytes);
    for (uint64_t i = 0; i < (uint64_t)M * N; i++) { uint32_t u = 0x7fa00badu; memcpy(&C[i], &u, 4); }
    c->ranges[0] = (WsRange){ (uintptr_t)A, (uintptr_t)A + p.a_bytes + 1 };  /* the e4m3 pack reads a u16 around the last byte */
    c->ranges[1] = (WsRange){ (uintptr_t)B, (uintptr_t)B + p.b_bytes + 1 };
    c->ranges[2] = (WsRange){ (uintptr_t)ap, (uintptr_t)ap + p.ap_bytes };
    c->ranges[3] = (WsRange){ (uintptr_t)bp, (uintptr_t)bp + p.bp_bytes };
    c->ranges[4] = (WsRange){ (uintptr_t)C, (uintptr_t)(C + (uint64_t)M * N) };
    int rc = omega_gpu_gemm_drive(cfg, lay, M, N, K, (uintptr_t)A, (uintptr_t)B, (uintptr_t)ap, (uintptr_t)bp, (uintptr_t)C, sim_launch, c);
    for (uint64_t i = 0; rc == 0 && i < (uint64_t)M * N; i++) if (f2u(C[i]) == 0x7fa00badu) rc = -9;
    free(ap); free(bp);
    return rc;
}

static void sim_tier(void) {
    char w[600], first[200];
    static const uint32_t sh[][3] = { { 1, 1, 1 }, { 1, 1, 17 }, { 16, 8, 16 }, { 17, 9, 33 }, { 64, 64, 64 }, { 65, 63, 100 },
                                      { 3, 130, 70 }, { 100, 3, 257 }, { 129, 65, 31 }, { 31, 200, 129 } };
    const size_t nsh = sizeof sh / sizeof sh[0];
    const OmegaGemmCfg cfgs[] = { { 0, 2, 4, 2, 2, 4, 0 }, { 0, 1, 1, 1, 1, 4, 0 }, { 0, 4, 1, 1, 4, 2, 0 } };
    void *A = malloc(300 * 300 * 2), *B = malloc(300 * 300 * 2);
    float *C = malloc(300 * 300 * 4);
    for (int t = 0; t < 2; t++)
        for (size_t ci = 0; ci < sizeof cfgs / sizeof cfgs[0]; ci++) {
            OmegaGemmCfg cfg = cfgs[ci]; cfg.type = (OmegaGpuGemmType)t;
            SimCtx c; memset(&c, 0, sizeof c);
            check(sim_build(&c, &cfg) == 0, "simulator: kernels built");
            uint64_t runs = 0, wrong = 0;
            for (size_t s = 0; s < nsh; s++)
                for (unsigned lay = 0; lay < 4; lay++) {
                    if (ci > 0 && lay != (unsigned)(s % 4)) continue; /* the other tile shapes: one layout per shape */
                    const uint32_t M = sh[s][0], N = sh[s][1], K = sh[s][2];
                    fill(cfg.type, A, (uint64_t)M * K, s * 4 + lay); fill(cfg.type, B, (uint64_t)K * N, s * 4 + lay + 1000);
                    c.order = (int)(s & 1u);
                    int rc = sim_gemm(&c, &cfg, lay, M, N, K, A, B, C);
                    uint64_t checked = 0, bad = rc == 0 ? compare(cfg.type, lay, M, N, K, A, B, C, &checked, first, sizeof first) : 1;
                    runs++;
                    if (rc != 0 || bad) {
                        wrong++;
                        snprintf(w, sizeof w, "sim %s rt %u ct %u wm %u wn %u, %ux%ux%u %s: rc %d, %" PRIu64 " wrong of %" PRIu64 " %s %s", tname(cfg.type), cfg.rt,
                                 cfg.ct, cfg.wm, cfg.wn, M, N, K, lname(lay), rc, bad, checked, first, c.err);
                        check(0, w);
                    }
                }
            snprintf(w, sizeof w, "simulator %s rt %u ct %u wm %u wn %u: %" PRIu64 " multiplies, %" PRIu64 " wrong", tname(cfg.type), cfg.rt, cfg.ct, cfg.wm, cfg.wn, runs, wrong);
            check(wrong == 0, w);
            printf("%s\n", w);
            sim_free(&c);
        }
    /* red checks: each mutant must change a result the real kernels get right */
    for (int t = 0; t < 2; t++)
        for (int mut = 1; mut <= 3; mut++) {
            OmegaGemmCfg cfg; omega_gpu_gemm_default_cfg((OmegaGpuGemmType)t, &cfg); cfg.mutant = mut;
            SimCtx c; memset(&c, 0, sizeof c);
            const uint32_t M = 65, N = 63, K = 100;
            fill(cfg.type, A, M * K, 77); fill(cfg.type, B, K * N, 78);
            int built = sim_build(&c, &cfg) == 0, rc = built ? sim_gemm(&c, &cfg, 0, M, N, K, A, B, C) : -1;
            uint64_t checked = 0, bad = rc == 0 ? compare(cfg.type, 0, M, N, K, A, B, C, &checked, first, sizeof first) : 1;
            snprintf(w, sizeof w, "simulator red check: %s mutant %d is caught (%" PRIu64 " wrong)", tname(cfg.type), mut, bad);
            check(built && bad > 0, w);
            printf("%s\n", w);
            sim_free(&c);
        }
    free(A); free(B); free(C);
}

/* ------------------------------------------------------------------ the chip */

static size_t mem_available(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256]; size_t kb = 0;
    while (f && fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
    if (f) fclose(f);
    return kb * 1024u;
}

static const double CEIL_TFLOPS[2] = { 115.36, 237.52 };

static void chip_one(OmegaGpuGemmType t, unsigned lay, uint32_t M, uint32_t N, uint32_t K, void *A, void *B, float *C) {
    char w[600], first[200];
    fill(t, A, (uint64_t)M * K, (uint64_t)M * 3 + N * 5 + K * 7 + lay);
    fill(t, B, (uint64_t)K * N, (uint64_t)M * 11 + N * 13 + K * 17 + lay);
    OmegaGpuGemmInfo info;
    int rc = omega_gpu_gemm(t, lay, M, N, K, A, B, C, &info);
    uint64_t checked = 0, bad = rc == 0 ? compare(t, lay, M, N, K, A, B, C, &checked, first, sizeof first) : 1;
    double flop = 2.0 * M * N * K, tf = info.main_ns ? flop / (double)info.main_ns / 1e3 : 0, tfp = info.main_ns ? flop / (double)(info.main_ns + info.pack_ns) / 1e3 : 0;
    snprintf(w, sizeof w, "%s %ux%ux%u %s: %s, %" PRIu64 " checked, %" PRIu64 " wrong%s%s; kernel %.3f ms %.2f TFLOPS = %.4f of ceiling, with packs %.2f TFLOPS = %.4f%s%s",
             tname(t), M, N, K, lname(lay), omega_gpu_gemm_rc_name(rc), checked, bad, bad && rc == 0 ? ", first " : "", bad && rc == 0 ? first : "",
             info.main_ns / 1e6, tf, tf / CEIL_TFLOPS[t], tfp, tfp / CEIL_TFLOPS[t], rc ? ": " : "", rc ? omega_gpu_session_last_error() : "");
    check(rc == 0 && bad == 0, w);
    if (rc == 0 && bad == 0) printf("ok: %s\n", w);
}

static void chip_tier(int quick) {
    char w[200];
    const size_t need = (size_t)8192 * 8192 * 2 * 2 + (size_t)8192 * 8192 * 4 + ((size_t)8 << 30);
    if (!quick && mem_available() < need * 2) { printf("NOT_RUN: MemAvailable %zu MiB < %zu MiB\n", mem_available() >> 20, need * 2 >> 20); exit(2); }
    const uint32_t big = quick ? 1024u : 8192u;
    void *A = malloc((size_t)big * big * 2), *B = malloc((size_t)big * big * 2);
    float *C = malloc((size_t)big * big * 4);
    if (!A || !B || !C) { printf("NOT_RUN: host allocation\n"); exit(2); }
    /* the shapes: per type and layout, the same list (26 each, 208 in all): edges, odd sizes, skinny, square up to 8192 */
    static const uint32_t sh[][3] = {
        { 1, 1, 1 }, { 1, 8192, 1 }, { 8192, 1, 1 }, { 1, 1, 8192 }, { 7, 13, 29 }, { 16, 8, 16 }, { 33, 65, 127 }, { 100, 100, 100 },
        { 128, 64, 64 }, { 255, 257, 511 }, { 1000, 999, 1001 }, { 1, 4096, 4096 }, { 4096, 1, 4096 }, { 4096, 4096, 1 },
        { 17, 8192, 3000 }, { 8192, 17, 3000 }, { 1024, 1024, 1024 }, { 2048, 2048, 2048 }, { 4096, 4096, 4096 }, { 3000, 5000, 700 },
        { 8192, 8192, 64 }, { 64, 8192, 8192 }, { 8192, 64, 8192 }, { 6000, 7000, 8000 }, { 8191, 8191, 8191 }, { 8192, 8192, 8192 } };
    const size_t nsh = quick ? 12 : sizeof sh / sizeof sh[0];
    int before = g_failed, shapes = 0;
    for (int t = 0; t < 2; t++)
        for (unsigned lay = 0; lay < 4; lay++)
            for (size_t s = 0; s < nsh; s++) {
                if (quick && lay != (unsigned)(s % 4)) continue;
                uint32_t M = sh[s][0] > big ? big : sh[s][0], N = sh[s][1] > big ? big : sh[s][1], K = sh[s][2] > big ? big : sh[s][2];
                chip_one((OmegaGpuGemmType)t, lay, M, N, K, A, B, C);
                shapes++;
            }
    printf("%d shapes, both types, all four layouts: %s\n", shapes, g_failed == before ? "PASS" : "FAIL");
    for (int t = 0; t < 2; t++)
        for (int mut = 1; mut <= 3; mut++) {
            OmegaGemmCfg cfg; omega_gpu_gemm_default_cfg((OmegaGpuGemmType)t, &cfg); cfg.mutant = mut;
            omega_gpu_gemm_test_set_cfg((OmegaGpuGemmType)t, &cfg);
            const uint32_t M = 65, N = 63, K = 100;
            fill((OmegaGpuGemmType)t, A, M * K, 77); fill((OmegaGpuGemmType)t, B, K * N, 78);
            int rc = omega_gpu_gemm((OmegaGpuGemmType)t, 0, M, N, K, A, B, C, NULL);
            char first[200]; uint64_t checked = 0, bad = rc == 0 ? compare((OmegaGpuGemmType)t, 0, M, N, K, A, B, C, &checked, first, sizeof first) : 1;
            snprintf(w, sizeof w, "chip red check: %s mutant %d is caught (%s, %" PRIu64 " wrong)", tname((OmegaGpuGemmType)t), mut, omega_gpu_gemm_rc_name(rc), bad);
            check(bad > 0, w);
            printf("%s\n", w);
            omega_gpu_gemm_test_set_cfg((OmegaGpuGemmType)t, NULL);
        }
    for (int t = 0; t < 2; t++) chip_one((OmegaGpuGemmType)t, 3, 65, 63, 100, A, B, C); /* right again after the red checks */
    omega_gpu_session_close(); /* the kernels go with the device (close hook); the next call builds them on a new one */
    printf("after the device closed and opened again:\n");
    for (int t = 0; t < 2; t++) chip_one((OmegaGpuGemmType)t, 0, 70, 33, 129, A, B, C);
    free(A); free(B); free(C);
}

int main(int argc, char **argv) {
    const char *tier = "chip";
    int quick = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) tier = "host";
        else if (!strcmp(argv[i], "--sim")) tier = "sim";
        else if (!strcmp(argv[i], "--quick")) quick = 1;
        else { fprintf(stderr, "usage: %s [--host-only | --sim | --quick]\n", argv[0]); return 2; }
    }
    if (!strcmp(tier, "host")) host_only();
    else if (!strcmp(tier, "sim")) { host_only(); sim_tier(); }
    else chip_tier(quick);
    if (g_failed) printf("C3 gemm %s: FAIL (%d of %d)\n", tier, g_failed, g_checks);
    else printf("C3 gemm %s: PASS (%d checks)\n", tier, g_checks);
    return g_failed ? 1 : 0;
}
