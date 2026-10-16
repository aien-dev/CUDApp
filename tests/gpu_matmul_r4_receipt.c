/* gpu_matmul_r4_receipt: the sealed CTA-budget receipt omega_gpu_matmul_api.h asks for before
 * OMEGA_GPU_MATMUL_MAX_CTAS moves (omega #473, R4).
 *
 *   ./gpu_matmul_r4_receipt --host-only            reference and comparator self-tests, no chip
 *   ./gpu_matmul_r4_receipt --out receipt.json     chip receipt (run under the quiet flag)
 *       [--budgets 64,128,256,512,1024] [--repeats 3] [--threads 12] [--shapes "m,k,n;..."]
 *
 * Per shape: B is uploaded once (resident, bf16). The reference is the full m x n product of the same
 * bf16 inputs in double on the host (threaded), with the accumulation scale (sum of |terms|) per element.
 * Per budget: the CTA budget is set (kernel cache cleared), then 2 x repeats resident calls alternate
 * -A, +A, -A, +A, ... (A negated by its sign bit, so the exact answer of each call is the negation of the
 * previous one). Every element of every call is checked:
 *   - rep 0 of each sign: every element within 1e-5 of the accumulation scale of the reference
 *     (the rule of gpu_matmul_api_test.c e2e_max_rel), and finite;
 *   - later repeats: bit-identical to rep 0 of the same sign;
 *   - I42 (cut 1: grids of 128 CTAs and above lost output with the single-tile kernel): an element still
 *     holding the previous call's bits (nonzero) is stale, i.e. not written by this call. Alternating the
 *     sign makes stale output visible; the budget sequence starts with -A because the previous budget
 *     ended with +A. The device-side sentinel check (OUTPUT_UNCHANGED) runs inside the API as well.
 * Negative controls: the comparator flags a one-element perturbation and a planted stale element; the
 * chip mutant kernel (last K step dropped) at the largest budget must be flagged by the full comparison.
 * Timing: chip = OmegaGpuMatmulInfo.elapsed_ns, call = call_ns; median over all 2 x repeats calls.
 * The best sealed budget is the one with every shape PASS and the least summed median chip time.
 */
#include "omega_gpu_matmul_api.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_matmul.h"
#include "sha256.h"
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOL 1e-5
#define MAX_BUDGETS 16
#define NAN_FILL 0x7fc0deadu

static const char *DEFAULT_SHAPES =
    /* the five training shapes (omega #473), then two I42 shapes on the single-tile kernel (rt = ct = 1):
     * decode 1 x 2048 x 8192 (grid_x up to the budget) and 7 row tiles (112 rows) */
    "8192,768,2304;8192,768,768;8192,768,3072;8192,3072,768;1024,768,50304;1,2048,8192;112,768,3072";
static const uint32_t DEFAULT_BUDGETS[] = { 64, 128, 256, 512, 1024 };

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static inline double absd(double x) { return x < 0.0 ? -x : x; }
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static int is_finite(float f) { return (fbits(f) & 0x7f800000u) != 0x7f800000u; }

/* inputs as gpu_matmul_api_test.c fill(): values in [-1, 1] at 1/1000 steps, rounded to bf16 */
static void fill_bf16(uint16_t *x, size_t n, uint32_t *seed) {
    for (size_t i = 0; i < n; i++) x[i] = omega_fp32_to_bf16(((float)(lcg(seed) % 2001) - 1000.0f) / 1000.0f);
}

/* ---- reference: acc[i*n+j] = sum_t a[i,t] b[t,j] in double, sc = max(sum |terms|, 1e-6) ---- */
typedef struct {
    uint32_t m, k, n, r0, r1;
    const double *ad, *btd; /* A m x k, B transposed n x k, both exact bf16 values in double */
    double *acc, *sc;
} RefJob;

static void *ref_rows(void *p) {
    RefJob *J = p;
    const uint32_t k = J->k, n = J->n;
    for (uint32_t i = J->r0; i < J->r1; i++) {
        const double *ar = J->ad + (size_t)i * k;
        for (uint32_t j = 0; j < n; j++) {
            const double *br = J->btd + (size_t)j * k;
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0, q0 = 0, q1 = 0, q2 = 0, q3 = 0;
            uint32_t t = 0;
            for (; t + 4 <= k; t += 4) {
                double p0 = ar[t] * br[t], p1 = ar[t + 1] * br[t + 1], p2 = ar[t + 2] * br[t + 2], p3 = ar[t + 3] * br[t + 3];
                s0 += p0; s1 += p1; s2 += p2; s3 += p3;
                q0 += absd(p0); q1 += absd(p1); q2 += absd(p2); q3 += absd(p3);
            }
            for (; t < k; t++) { double p0 = ar[t] * br[t]; s0 += p0; q0 += absd(p0); }
            const double q = (q0 + q1) + (q2 + q3);
            J->acc[(size_t)i * n + j] = (s0 + s1) + (s2 + s3);
            J->sc[(size_t)i * n + j] = q > 1e-6 ? q : 1e-6;
        }
    }
    return NULL;
}

static int reference(uint32_t m, uint32_t k, uint32_t n, const uint16_t *a, const uint16_t *b, unsigned threads, double *acc, double *sc) {
    double *ad = malloc((size_t)m * k * sizeof *ad), *btd = malloc((size_t)n * k * sizeof *btd);
    if (!ad || !btd) { free(ad); free(btd); return -1; }
    for (size_t i = 0; i < (size_t)m * k; i++) ad[i] = omega_bf16_to_fp32(a[i]);
    for (uint32_t t = 0; t < k; t++)
        for (uint32_t j = 0; j < n; j++) btd[(size_t)j * k + t] = omega_bf16_to_fp32(b[(size_t)t * n + j]);
    if (threads < 1) threads = 1;
    if (threads > 64) threads = 64;
    if (threads > m) threads = m;
    pthread_t th[64]; RefJob J[64];
    int rc = 0;
    for (unsigned w = 0; w < threads; w++) {
        J[w] = (RefJob){ m, k, n, (uint32_t)((uint64_t)m * w / threads), (uint32_t)((uint64_t)m * (w + 1) / threads), ad, btd, acc, sc };
        if (pthread_create(&th[w], NULL, ref_rows, &J[w]) != 0) { ref_rows(&J[w]); th[w] = 0; }
    }
    for (unsigned w = 0; w < threads; w++) if (th[w]) pthread_join(th[w], NULL);
    free(ad); free(btd);
    return rc;
}

/* ---- full comparison of one output against sign * reference ---- */
typedef struct { size_t mismatches, nonfinite; double max_rel; } Cmp;

static Cmp compare_full(size_t cnt, const float *c, const double *acc, const double *sc, double sign) {
    Cmp r = { 0, 0, 0.0 };
    for (size_t i = 0; i < cnt; i++) {
        if (!is_finite(c[i])) { r.nonfinite++; r.mismatches++; continue; }
        const double rel = absd((double)c[i] - sign * acc[i]) / sc[i];
        if (rel > r.max_rel) r.max_rel = rel;
        if (rel > TOL) r.mismatches++;
    }
    return r;
}

static size_t count_bit_diff(size_t cnt, const float *x, const float *y) {
    size_t d = 0;
    for (size_t i = 0; i < cnt; i++) d += fbits(x[i]) != fbits(y[i]);
    return d;
}

/* elements still holding the previous call's bits (the exact answer flipped sign, so only an element
 * whose answer is exactly zero may legitimately repeat; those are excluded) */
static size_t count_stale(size_t cnt, const float *cur, const float *prev) {
    size_t s = 0;
    for (size_t i = 0; i < cnt; i++) s += fbits(cur[i]) == fbits(prev[i]) && (fbits(cur[i]) & 0x7fffffffu) != 0;
    return s;
}

static size_t count_not_negation(size_t cnt, const float *pos, const float *neg) {
    size_t d = 0;
    for (size_t i = 0; i < cnt; i++) d += fbits(pos[i]) != (fbits(neg[i]) ^ 0x80000000u);
    return d;
}

/* sha256 of the output bytes (first 16 hex digits), so separate runs can be compared bit for bit */
static void digest16(const float *c, size_t cnt, char hex[17]) {
    uint8_t d[SHA256_DIGEST_SIZE];
    sha256_hash((const uint8_t *)c, cnt * sizeof *c, d);
    for (int i = 0; i < 8; i++) snprintf(hex + 2 * i, 3, "%02x", d[i]);
}

static int cmp_u64(const void *x, const void *y) { uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y; return a < b ? -1 : a > b; }
static uint64_t median_u64(uint64_t *v, size_t n) { qsort(v, n, sizeof *v, cmp_u64); return n ? v[n / 2] : 0; }

/* comparator self-test on data the caller has (exact reference rounded to f32 is the "chip" output) */
static void comparator_controls(uint32_t m, uint32_t n, const double *acc, const double *sc, const char *tag, FILE *out) {
    const size_t cnt = (size_t)m * n;
    float *c = malloc(cnt * sizeof *c), *prev = malloc(cnt * sizeof *prev);
    if (!c || !prev) { CHECK(0, "%s comparator controls: host memory", tag); free(c); free(prev); return; }
    for (size_t i = 0; i < cnt; i++) { c[i] = (float)acc[i]; prev[i] = -(float)acc[i]; }
    Cmp clean = compare_full(cnt, c, acc, sc, 1.0);
    const size_t at = cnt / 2 + 3;
    c[at] = (float)(acc[at] + 4.0 * TOL * sc[at]);
    Cmp pert = compare_full(cnt, c, acc, sc, 1.0);
    c[at] = (float)acc[at];
    const size_t at2 = cnt - 1;
    float keep = c[at2];
    if (acc[at2] == 0.0) { c[at2] = 1.0f; prev[at2] = 1.0f; } else c[at2] = prev[at2];
    const size_t stale = count_stale(cnt, c, prev);
    c[at2] = keep;
    c[0] = bitsf(NAN_FILL);
    Cmp nan = compare_full(cnt, c, acc, sc, 1.0);
    CHECK(clean.mismatches == 0, "%s control: exact reference must compare clean (%zu)", tag, clean.mismatches);
    CHECK(pert.mismatches == 1, "%s control: one perturbed element must be flagged (%zu)", tag, pert.mismatches);
    CHECK(stale == 1, "%s control: one planted stale element must be flagged (%zu)", tag, stale);
    CHECK(nan.mismatches == 1 && nan.nonfinite == 1, "%s control: one unwritten (NaN) element must be flagged (%zu)", tag, nan.mismatches);
    printf("%s comparator controls: clean=%zu perturbed=%zu stale=%zu unwritten=%zu\n", tag, clean.mismatches, pert.mismatches, stale, nan.mismatches);
    if (out) fprintf(out, "\"comparator_control\":{\"shape\":\"%s\",\"clean_mismatches\":%zu,\"perturbed_flagged\":%zu,\"stale_flagged\":%zu,\"unwritten_flagged\":%zu},",
                     tag, clean.mismatches, pert.mismatches, stale, nan.mismatches);
    free(c); free(prev);
}

static int host_only(unsigned threads) {
    /* threaded reference against a naive single-thread double loop, and the comparator controls */
    const uint32_t m = 37, k = 80, n = 52;
    uint16_t *a = malloc((size_t)m * k * 2), *b = malloc((size_t)k * n * 2);
    double *acc = malloc((size_t)m * n * 8), *sc = malloc((size_t)m * n * 8);
    uint32_t seed = 4730u;
    fill_bf16(a, (size_t)m * k, &seed); fill_bf16(b, (size_t)k * n, &seed);
    CHECK(reference(m, k, n, a, b, threads, acc, sc) == 0, "reference ran");
    double worst = 0.0;
    for (uint32_t i = 0; i < m; i++)
        for (uint32_t j = 0; j < n; j++) {
            double s = 0, q = 0;
            for (uint32_t t = 0; t < k; t++) { double p = (double)omega_bf16_to_fp32(a[i * k + t]) * omega_bf16_to_fp32(b[t * n + j]); s += p; q += absd(p); }
            double d = absd(s - acc[i * n + j]) / (q > 1e-6 ? q : 1e-6);
            if (d > worst) worst = d;
            if (absd(q - sc[i * n + j]) > 1e-12 * (q > 1.0 ? q : 1.0)) worst = 1.0;
        }
    CHECK(worst < 1e-12, "threaded reference equals the naive loop (worst rel %g)", worst);
    comparator_controls(m, n, acc, sc, "host 37x80x52", NULL);
    /* sign-bit negation of bf16 A negates the reference exactly */
    for (size_t i = 0; i < (size_t)m * k; i++) a[i] ^= 0x8000u;
    double *acc2 = malloc((size_t)m * n * 8), *sc2 = malloc((size_t)m * n * 8);
    reference(m, k, n, a, b, threads, acc2, sc2);
    size_t notneg = 0;
    for (size_t i = 0; i < (size_t)m * n; i++) notneg += acc2[i] != -acc[i] || sc2[i] != sc[i];
    CHECK(notneg == 0, "negated A gives the exact negated reference (%zu differ)", notneg);
    free(a); free(b); free(acc); free(sc); free(acc2); free(sc2);
    printf("%s: %d checks, %d failed\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
    return g_failed ? 1 : 0;
}

typedef struct { uint64_t chip_med_ns; int pass; } Cell;

int main(int argc, char **argv) {
    const char *out_path = NULL, *shapes = DEFAULT_SHAPES;
    uint32_t budgets[MAX_BUDGETS]; unsigned nb = 0;
    unsigned repeats = 3, threads = 12;
    int host = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--shapes") == 0 && i + 1 < argc) shapes = argv[++i];
        else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) repeats = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--budgets") == 0 && i + 1 < argc) {
            char *s = argv[++i];
            while (*s && nb < MAX_BUDGETS) { budgets[nb++] = (uint32_t)strtoul(s, &s, 10); if (*s == ',') s++; else break; }
        } else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (host) return host_only(threads);
    if (!nb) { nb = sizeof DEFAULT_BUDGETS / sizeof DEFAULT_BUDGETS[0]; memcpy(budgets, DEFAULT_BUDGETS, sizeof DEFAULT_BUDGETS); }
    if (repeats < 1) repeats = 1;
    if (repeats > 32) repeats = 32; /* 2 x repeats calls fit the per-budget timing arrays */
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) {
        fprintf(out, "{\"schema\":\"OMEGA_GPU_MATMUL_R4_RECEIPT_V1\",\"default_budget\":%u,\"repeats\":%u,\"tol_rel_scale\":%g,\"budgets\":[", OMEGA_GPU_MATMUL_MAX_CTAS, repeats, TOL);
        for (unsigned b = 0; b < nb; b++) fprintf(out, "%s%u", b ? "," : "", budgets[b]);
        fprintf(out, "],");
    }
    Cell cells[32][MAX_BUDGETS]; memset(cells, 0, sizeof cells);
    char shape_names[32][40]; unsigned ns = 0;
    int controls_done = 0, mutant_done = 0;
    if (out) fprintf(out, "\"runs\":[");
    int first_run = 1;
    char *copy = strdup(shapes), *save = NULL;
    for (char *tok = strtok_r(copy, ";", &save); tok && ns < 32 && !omega_gpu_matmul_is_blocked(); tok = strtok_r(NULL, ";", &save)) {
        unsigned m, k, n;
        if (sscanf(tok, "%u,%u,%u", &m, &k, &n) != 3) { CHECK(0, "bad shape %s", tok); continue; }
        snprintf(shape_names[ns], sizeof shape_names[ns], "%ux%ux%u", m, k, n);
        const size_t na = (size_t)m * k, nbb = (size_t)k * n, cnt = (size_t)m * n;
        uint16_t *a = malloc(na * 2), *an = malloc(na * 2), *b = malloc(nbb * 2);
        double *acc = malloc(cnt * 8), *sc = malloc(cnt * 8);
        float *pos0 = malloc(cnt * 4), *neg0 = malloc(cnt * 4), *tmp[2] = { malloc(cnt * 4), malloc(cnt * 4) }, *base_pos = malloc(cnt * 4);
        if (!a || !an || !b || !acc || !sc || !pos0 || !neg0 || !tmp[0] || !tmp[1] || !base_pos) { CHECK(0, "host memory for %s", tok); break; }
        uint32_t seed = 0x5eed0000u ^ (m << 20) ^ (k << 10) ^ n;
        fill_bf16(a, na, &seed); fill_bf16(b, nbb, &seed);
        for (size_t i = 0; i < na; i++) an[i] = a[i] ^ 0x8000u;
        uint32_t rt, ct; omega_matmul_frag_block_shape((m + 15u) / 16u, (n + 7u) / 8u, &rt, &ct);
        const double flop = 2.0 * m * k * n;
        if (reference(m, k, n, a, b, threads, acc, sc) != 0) { CHECK(0, "reference %s: host memory", tok); break; }
        if (!controls_done && m == 8192 && n == 768 && k == 768) {
            if (out) { fprintf(out, "%s{\"control\":true,", first_run ? "" : ","); first_run = 0; }
            comparator_controls(m, n, acc, sc, shape_names[ns], out);
            if (out) fprintf(out, "\"pass\":%s}", g_failed ? "false" : "true");
            controls_done = 1;
        }
        OmegaGpuTensor *t = NULL;
        int urc = omega_gpu_tensor_upload_bf16(k, n, b, &t);
        CHECK(urc == OMEGA_GPU_MATMUL_OK, "upload %s rc=%s (%s)", tok, omega_gpu_matmul_rc_name(urc), omega_gpu_matmul_last_error());
        if (urc != OMEGA_GPU_MATMUL_OK) break;
        float *prev = NULL;
        for (unsigned bi = 0; bi < nb; bi++) {
            omega_gpu_matmul_set_cta_budget(budgets[bi]);
            const unsigned calls = 2u * repeats;
            uint64_t chip[64], call[64]; int rcs_ok = 1, hit_after_first = 1;
            size_t rep_diff = 0, stale_total = 0, stale_max = 0;
            Cmp cp = { 0, 0, 0 }, cn = { 0, 0, 0 };
            OmegaGpuMatmulInfo info; memset(&info, 0, sizeof info);
            for (unsigned ci = 0; ci < calls && ci < 64; ci++) {
                const int neg = (ci % 2u) == 0, rep = (int)(ci / 2u);
                float *dst = rep == 0 ? (neg ? neg0 : pos0) : tmp[ci % 2u];
                for (size_t i = 0; i < cnt; i++) dst[i] = bitsf(NAN_FILL);
                int rc = omega_gpu_matmul_resident_bf16(m, neg ? an : a, t, dst, &info);
                chip[ci] = info.elapsed_ns; call[ci] = info.call_ns;
                if (rc != OMEGA_GPU_MATMUL_OK) {
                    rcs_ok = 0;
                    CHECK(0, "%s budget %u call %u rc=%s (%s)", tok, budgets[bi], ci, omega_gpu_matmul_rc_name(rc), omega_gpu_matmul_last_error());
                    break;
                }
                if (ci > 0 && !info.kernel_cache_hit) hit_after_first = 0;
                if (prev) { size_t s = count_stale(cnt, dst, prev); stale_total += s; if (s > stale_max) stale_max = s; }
                if (rep == 0) {
                    if (neg) cn = compare_full(cnt, dst, acc, sc, -1.0); else cp = compare_full(cnt, dst, acc, sc, 1.0);
                } else rep_diff += count_bit_diff(cnt, dst, neg ? neg0 : pos0);
                prev = dst;
            }
            const unsigned done = rcs_ok ? calls : 0;
            uint64_t chip_s[64], call_s[64];
            memcpy(chip_s, chip, sizeof chip); memcpy(call_s, call, sizeof call);
            const uint64_t chip_med = done ? median_u64(chip_s, done) : 0, call_med = done ? median_u64(call_s, done) : 0;
            const size_t neg_exact = rcs_ok ? count_not_negation(cnt, pos0, neg0) : cnt;
            size_t cross = 0;
            if (rcs_ok) { if (bi == 0) memcpy(base_pos, pos0, cnt * 4); else cross = count_bit_diff(cnt, pos0, base_pos); }
            char dig[17] = "-";
            if (rcs_ok) digest16(pos0, cnt, dig);
            const uint32_t ctas = info.grid_x * info.grid_y;
            const int pass = rcs_ok && cp.mismatches == 0 && cn.mismatches == 0 && rep_diff == 0 && stale_total == 0;
            CHECK(pass, "%s budget %u: rc_ok=%d pos_mism=%zu neg_mism=%zu repeat_diff=%zu stale=%zu", tok, budgets[bi], rcs_ok, cp.mismatches, cn.mismatches, rep_diff, stale_total);
            cells[ns][bi].pass = pass; cells[ns][bi].chip_med_ns = chip_med;
            const double tflops = chip_med ? flop / (double)chip_med / 1e3 : 0.0;
            printf("%s %s budget=%u grid=%ux%u ctas=%u launches=%u rt=%u ct=%u i42_regime=%d chip_med_ms=%.3f call_med_ms=%.3f tflops_chip=%.3f "
                   "elements=%zu pos_mism=%zu neg_mism=%zu max_rel=%.3g repeat_bitdiff=%zu stale=%zu neg_exact_diff=%zu vs_first_budget_bitdiff=%zu sha=%s cache_hit=%d\n",
                   pass ? "PASS" : "FAIL", tok, budgets[bi], info.grid_x, info.grid_y, ctas, info.chip_calls, rt, ct, ctas >= 128,
                   chip_med / 1e6, call_med / 1e6, tflops, cnt, cp.mismatches, cn.mismatches, cp.max_rel > cn.max_rel ? cp.max_rel : cn.max_rel,
                   rep_diff, stale_total, neg_exact, cross, dig, hit_after_first);
            if (out) {
                fprintf(out, "%s{\"m\":%u,\"k\":%u,\"n\":%u,\"budget\":%u,\"pass\":%s,\"grid_x\":%u,\"grid_y\":%u,\"ctas_per_launch\":%u,\"launches\":%u,"
                        "\"rt\":%u,\"ct\":%u,\"single_tile_kernel\":%s,\"i42_regime\":%s,\"calls\":%u,\"elements_per_call\":%zu,"
                        "\"pos_mismatches\":%zu,\"neg_mismatches\":%zu,\"nonfinite\":%zu,\"max_rel\":%.4g,\"repeat_bitdiff\":%zu,"
                        "\"stale_elements\":%zu,\"stale_max_per_call\":%zu,\"neg_exact_negation_diff\":%zu,\"vs_first_budget_bitdiff\":%zu,"
                        "\"pos_output_sha256_16\":\"%s\",\"cache_hit_after_first\":%s,\"chip_median_ns\":%" PRIu64 ",\"call_median_ns\":%" PRIu64 ",\"tflops_chip\":%.4f,\"chip_ns\":[",
                        first_run ? "" : ",", m, k, n, budgets[bi], pass ? "true" : "false", info.grid_x, info.grid_y, ctas, info.chip_calls,
                        rt, ct, rt == 1 && ct == 1 ? "true" : "false", ctas >= 128 ? "true" : "false", done, cnt,
                        cp.mismatches, cn.mismatches, cp.nonfinite + cn.nonfinite, cp.max_rel > cn.max_rel ? cp.max_rel : cn.max_rel, rep_diff,
                        stale_total, stale_max, neg_exact, cross, dig, hit_after_first ? "true" : "false", chip_med, call_med, tflops);
                for (unsigned ci = 0; ci < done; ci++) fprintf(out, "%s%" PRIu64, ci ? "," : "", chip[ci]);
                fprintf(out, "],\"call_ns\":[");
                for (unsigned ci = 0; ci < done; ci++) fprintf(out, "%s%" PRIu64, ci ? "," : "", call[ci]);
                fprintf(out, "]}");
                first_run = 0;
            }
            fflush(stdout);
            if (omega_gpu_matmul_is_blocked()) break;
        }
        /* negative control on the chip: the mutant kernel at the largest budget must be flagged */
        if (!mutant_done && m == 8192 && k == 768 && n == 768 && !omega_gpu_matmul_is_blocked()) {
            omega_gpu_matmul_set_cta_budget(budgets[nb - 1]);
            omega_gpu_matmul_test_set_mutant(1);
            OmegaGpuMatmulInfo info;
            int rc = omega_gpu_matmul_resident_bf16(m, a, t, pos0, &info);
            Cmp mc = rc == OMEGA_GPU_MATMUL_OK ? compare_full(cnt, pos0, acc, sc, 1.0) : (Cmp){ 0, 0, 0 };
            omega_gpu_matmul_test_set_mutant(0);
            const int caught = rc != OMEGA_GPU_MATMUL_OK || mc.mismatches > 0;
            CHECK(caught, "mutant at budget %u NOT caught by the full comparison", budgets[nb - 1]);
            printf("%s mutant control %s budget=%u rc=%s flagged=%zu of %zu\n", caught ? "KILLED" : "SURVIVED", tok, budgets[nb - 1], omega_gpu_matmul_rc_name(rc), mc.mismatches, cnt);
            if (out) { fprintf(out, ",{\"mutant_control\":true,\"m\":%u,\"k\":%u,\"n\":%u,\"budget\":%u,\"rc\":\"%s\",\"flagged\":%zu,\"elements\":%zu,\"killed\":%s}",
                               m, k, n, budgets[nb - 1], omega_gpu_matmul_rc_name(rc), mc.mismatches, cnt, caught ? "true" : "false"); }
            mutant_done = 1;
        }
        omega_gpu_tensor_free(t);
        free(a); free(an); free(b); free(acc); free(sc); free(pos0); free(neg0); free(tmp[0]); free(tmp[1]); free(base_pos);
        ns++;
    }
    free(copy);
    omega_gpu_matmul_set_cta_budget(0);
    CHECK(controls_done && mutant_done, "negative controls ran (comparator %d, mutant %d)", controls_done, mutant_done);
    /* best sealed budget: every shape PASS, least summed median chip time over every shape in the run (the default
     * list is the five training shapes plus 1x2048x8192 and 112x768x3072) */
    int best = -1; uint64_t best_sum = 0;
    if (out) fprintf(out, "],\"budget_summary\":[");
    for (unsigned bi = 0; bi < nb; bi++) {
        int all = ns > 0; uint64_t sum = 0;
        for (unsigned s = 0; s < ns; s++) { all &= cells[s][bi].pass; sum += cells[s][bi].chip_med_ns; }
        printf("budget %u: %s, summed median chip %.3f ms over %u shapes\n", budgets[bi], all ? "SEALED" : "NOT SEALED", sum / 1e6, ns);
        if (out) fprintf(out, "%s{\"budget\":%u,\"sealed\":%s,\"sum_chip_median_ns\":%" PRIu64 "}", bi ? "," : "", budgets[bi], all ? "true" : "false", sum);
        if (all && (best < 0 || sum < best_sum)) { best = (int)bi; best_sum = sum; }
    }
    const int ok = g_failed == 0 && best >= 0 && !omega_gpu_matmul_is_blocked();
    printf("best sealed budget: %d\n%s: %d checks, %d failed\n", best >= 0 ? (int)budgets[best] : -1, ok ? "PASS" : "FAIL", g_checks, g_failed);
    if (out) {
        fprintf(out, "],\"best_sealed_budget\":%d,\"blocked\":%s,\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", best >= 0 ? (int)budgets[best] : -1,
                omega_gpu_matmul_is_blocked() ? "true" : "false", g_checks, g_failed, ok ? "OMEGA_GPU_MATMUL_R4_SEALED" : "OMEGA_GPU_MATMUL_R4_FAIL");
        fclose(out);
    }
    return ok ? 0 : 1;
}
