/* gpu_attention_test: FB-1 cut 5 gate for omega_gpu_attention_api.
 *
 *   ./gpu_attention_test --host-only          refusals, codegen (both kernel families and every
 *                                             mutant), register budget, nvdisasm listing check
 *   ./gpu_attention_test --timing [--out r.json] cut 4b gate: 1 query x 32 heads x 2048 ctx, 100 calls
 *                                             through the persistent session, median under 3 ms
 *   ./gpu_attention_test --out receipt.json   chip gate: every shape against the host oracle, the
 *                                             mutants (deliberately wrong kernels) which the same
 *                                             check must catch, a negative control, timings
 *
 * Oracles re-state aien-sovereign-core crates/aien-inference-abi/src/backend.rs (main 8b5caed) in C:
 * ReferenceCpuBackend::gqa_attention (f64 scores, max, exp, sum, 1/max(sum,1e-12), f64 weighted sum),
 * TensorBackend::paged_attention (f64 online softmax token by token over the paged bf16 pool, tokens
 * past the block table dropped, zeros for empty context) and paged_attention_batch (ctx <= 0 -> 0,
 * table rows past the array -> empty, negative entries removed). Compiled with -ffp-contract=off.
 *
 * Tolerance (why): the chip accumulates the 64-term dot products, the per-chunk sums and the
 * rescaled running sums in f32 (worst case about n * 2^-24 relative over n = 2048 terms, 1.2e-4),
 * uses MUFU.EX2 (PTX ex2.approx: 2^-22.5 relative) for every weight and for the 32 chunk rescale
 * factors, and MUFU.RCP for the final normalisation. Every output is a convex combination of V
 * values of unit scale, so |error| <= 2e-4 * |want| + 2e-5 covers these with margin; the mutants
 * move results by 1e-2 or more (wrong head, wrong token) or produce NaN (overflow without the max).
 * The large-score case (scores up to ~300) carries the f32 rounding of the score itself into the
 * exponent (3e-5 absolute per score), so it uses 1e-3 relative + 1e-4 absolute.
 */
#include "omega_gpu_attention_api.h"
#include "omega_gpu_session.h"
#include "omega_blackwell_codegen.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int g_checks, g_failed;
static int g_sim_mode; /* --sim: the host simulator never opens the device */
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); uint32_t r = ((u >> 16) & 1u) + 0x7fffu; return (uint16_t)((u + r) >> 16); }
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

/* ----------------------------------------------------------------- oracles */
static void oracle_gqa(const float *q, const float *kc, const float *vc, size_t seq_len, size_t nqh, size_t nkv, size_t hd, float *out) {
    if (seq_len == 0) { memset(out, 0, nqh * hd * 4); return; }
    size_t ratio = nqh / nkv; double inv_sqrt_d = 1.0 / sqrt((double)hd);
    double *scores = malloc(seq_len * sizeof *scores);
    for (size_t h = 0; h < nqh; h++) {
        size_t kvh = h / ratio; const float *qh = q + h * hd;
        for (size_t t = 0; t < seq_len; t++) {
            const float *kt = kc + (t * nkv + kvh) * hd; double dot = 0.0;
            for (size_t d = 0; d < hd; d++) dot += (double)qh[d] * (double)kt[d];
            scores[t] = dot * inv_sqrt_d;
        }
        double mx = -INFINITY; for (size_t t = 0; t < seq_len; t++) if (scores[t] > mx) mx = scores[t];
        double sum = 0.0; for (size_t t = 0; t < seq_len; t++) { scores[t] = exp(scores[t] - mx); sum += scores[t]; }
        double inv = 1.0 / (sum > 1e-12 ? sum : 1e-12);
        for (size_t t = 0; t < seq_len; t++) scores[t] *= inv;
        for (size_t d = 0; d < hd; d++) {
            double s = 0.0;
            for (size_t t = 0; t < seq_len; t++) s += scores[t] * (double)vc[(t * nkv + kvh) * hd + d];
            out[h * hd + d] = (float)s;
        }
    }
    free(scores);
}
/* bf16 pool in the canonical layout; the oracle's read_kv_head: element_offset(block, layer, is_value, token) + kv_head*head_dim*2 */
static const uint16_t *kv_head(const uint8_t *pool, const OmegaGpuKvLayout *ly, size_t blk, size_t layer, int is_v, size_t tok, size_t kvh, size_t hd) {
    size_t off = blk * ly->block_stride_bytes + layer * ly->layer_stride_bytes + (is_v ? ly->kv_plane_stride_bytes : 0) + tok * ly->token_stride_bytes + kvh * hd * 2;
    return (const uint16_t *)(pool + off);
}
static void oracle_paged(const float *q, const uint8_t *pool, const OmegaGpuKvLayout *ly, const uint32_t *ids, size_t nids, size_t ctx, size_t layer,
                         size_t nqh, size_t nkv, size_t hd, float *out) {
    if (ctx == 0 || nids == 0) { memset(out, 0, nqh * hd * 4); return; }
    size_t ratio = nqh / nkv; double inv_sqrt_d = 1.0 / sqrt((double)hd); size_t bs = ly->block_size;
    double *acc = malloc(hd * sizeof *acc);
    for (size_t h = 0; h < nqh; h++) {
        size_t kvh = h / ratio; const float *qh = q + h * hd;
        double m = -INFINITY, l = 0.0; for (size_t d = 0; d < hd; d++) acc[d] = 0.0;
        for (size_t t = 0; t < ctx; t++) {
            size_t bi = t / bs; if (bi >= nids) break;
            const uint16_t *kh = kv_head(pool, ly, ids[bi], layer, 0, t % bs, kvh, hd);
            const uint16_t *vh = kv_head(pool, ly, ids[bi], layer, 1, t % bs, kvh, hd);
            double dot = 0.0; for (size_t d = 0; d < hd; d++) dot += (double)qh[d] * (double)bf2f(kh[d]);
            double score = dot * inv_sqrt_d, mn = m > score ? m : score, alpha = exp(m - mn), beta = exp(score - mn);
            l = l * alpha + beta;
            for (size_t d = 0; d < hd; d++) acc[d] = acc[d] * alpha + beta * (double)bf2f(vh[d]);
            m = mn;
        }
        double inv = l > 0.0 ? 1.0 / l : 0.0;
        for (size_t d = 0; d < hd; d++) out[h * hd + d] = (float)(acc[d] * inv);
    }
    free(acc);
}
static void oracle_batch(const float *q, const uint8_t *pool, const OmegaGpuKvLayout *ly, const int32_t *tables, const int32_t *ctxs, size_t maxb, size_t nseq,
                         size_t layer, size_t nqh, size_t nkv, size_t hd, float *out) {
    size_t stride = nqh * hd;
    uint32_t *ids = malloc((maxb ? maxb : 1) * sizeof *ids);
    for (size_t s = 0; s < nseq; s++) {
        size_t ctx = ctxs[s] > 0 ? (size_t)ctxs[s] : 0, n = 0;
        size_t total = nseq * maxb; /* the table array length as the trait sees it */
        if ((s + 1) * maxb <= total) for (size_t b = 0; b < maxb; b++) if (tables[s * maxb + b] >= 0) ids[n++] = (uint32_t)tables[s * maxb + b];
        oracle_paged(q + s * stride, pool, ly, ids, n, ctx, layer, nqh, nkv, hd, out + s * stride);
    }
    free(ids);
}

static size_t compare(const float *got, const float *want, size_t n, double rel, double abs_tol, double *worst) {
    size_t bad = 0; *worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i] - (double)want[i]);
        double lim = rel * fabs((double)want[i]) + abs_tol;
        double scaled = lim > 0 ? d / lim : (d > 0 ? INFINITY : 0.0);
        if (scaled > *worst) *worst = scaled;
        if (!(d <= lim)) bad++; /* NaN counts as bad */
    }
    return bad;
}

/* ------------------------------------------------------------- host only */
static int nvdisasm_check(bool f32, const char *name, const char *const *must, int n_must) {
    const char *nv = "/usr/local/cuda/bin/nvdisasm";
    if (access(nv, X_OK) != 0) { printf("nvdisasm: not present, listing check skipped for %s\n", name); return 0; }
    OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
    if (omega_gpu_attention_codegen(f32, f32 ? 12 : 4, 3, &k) != OMEGA_GPU_ATTN_OK) return -1;
    char bin[64], lst[64], cmd[256];
    snprintf(bin, sizeof bin, "/tmp/omega_attn_%s_%d.bin", name, (int)getpid());
    snprintf(lst, sizeof lst, "/tmp/omega_attn_%s_%d.sass", name, (int)getpid());
    FILE *f = fopen(bin, "wb"); if (!f) { free(k.code); return -1; }
    fwrite(k.code, 1, k.code_size, f); fclose(f);
    snprintf(cmd, sizeof cmd, "%s -b SM121 %s > %s 2>&1", nv, bin, lst);
    int rc = system(cmd);
    free(k.code);
    FILE *l = fopen(lst, "r"); if (!l) return -1;
    char line[512]; int seen[16] = {0}; int bad = 0, insns = 0;
    while (fgets(line, sizeof line, l)) {
        if (!strstr(line, "/*0")) continue;
        insns++;
        if (strstr(line, "?") || strstr(line, "error")) bad++;
        for (int i = 0; i < n_must; i++) if (strstr(line, must[i])) seen[i] = 1;
    }
    fclose(l);
    int miss = 0;
    for (int i = 0; i < n_must; i++) if (!seen[i]) { printf("  %s: nvdisasm listing lacks %s\n", name, must[i]); miss++; }
    printf("nvdisasm %s: rc=%d insns=%d undecodable=%d missing=%d (%s)\n", name, rc, insns, bad, miss, lst);
    unlink(bin);
    return (rc == 0 && bad == 0 && miss == 0) ? 0 : -1;
}

static OmegaGpuKvLayout layout_for(uint32_t num_blocks, uint32_t bs, uint32_t layers, uint32_t nkv, uint32_t hd) {
    OmegaGpuKvLayout ly;
    ly.head_stride_bytes = (uint64_t)hd * 2; ly.token_stride_bytes = ly.head_stride_bytes * nkv;
    ly.kv_plane_stride_bytes = ly.token_stride_bytes * bs; ly.layer_stride_bytes = 2 * ly.kv_plane_stride_bytes;
    ly.block_stride_bytes = ly.layer_stride_bytes * layers; ly.pool_bytes = ly.block_stride_bytes * num_blocks;
    ly.num_blocks = num_blocks; ly.num_layers = layers; ly.block_size = bs;
    return ly;
}

static void host_only(void) {
    float q[64 * 64], kv[64 * 64], out[64 * 64]; memset(q, 0, sizeof q); memset(kv, 0, sizeof kv);
    CHECK(omega_gpu_gqa_attention_f32(NULL, kv, kv, 1, 32, 4, 64, out, NULL) == OMEGA_GPU_ATTN_BAD_ARGS, "NULL q refused");
    CHECK(omega_gpu_gqa_attention_f32(q, kv, kv, 1, 32, 4, 128, out, NULL) == OMEGA_GPU_ATTN_TOO_LARGE, "head_dim 128 refused (TOO_LARGE)");
    CHECK(omega_gpu_gqa_attention_f32(q, kv, kv, 1, 6, 2, 64, out, NULL) == OMEGA_GPU_ATTN_BAD_ARGS, "gqa ratio 3 refused");
    CHECK(omega_gpu_gqa_attention_f32(q, kv, kv, 1, 32, 5, 64, out, NULL) == OMEGA_GPU_ATTN_BAD_ARGS, "32 q heads / 5 kv heads refused");
    CHECK(omega_gpu_gqa_attention_f32(q, kv, kv, OMEGA_GPU_ATTN_MAX_GQA_CTX + 1, 32, 4, 64, out, NULL) == OMEGA_GPU_ATTN_TOO_LARGE, "context past the envelope refused");
    for (int i = 0; i < 64; i++) out[i] = 1.0f;
    OmegaGpuAttnInfo info;
    CHECK(omega_gpu_gqa_attention_f32(q, kv, kv, 0, 1, 1, 64, out, &info) == OMEGA_GPU_ATTN_OK && out[0] == 0.0f && out[63] == 0.0f && info.chip_calls == 0, "seq_len 0 writes zeros without a launch");
    OmegaGpuKvLayout ly = layout_for(8, 16, 2, 4, 64);
    uint8_t *pool = calloc(ly.pool_bytes, 1);
    uint32_t ids[2] = { 1, 9 };
    CHECK(omega_gpu_paged_attention_bf16(q, pool, &ly, ids, 2, 20, 0, 32, 4, 64, out, NULL) == OMEGA_GPU_ATTN_BAD_ARGS, "block id past the pool refused");
    CHECK(omega_gpu_paged_attention_bf16(q, pool, &ly, ids, 1, 20, 2, 32, 4, 64, out, NULL) == OMEGA_GPU_ATTN_BAD_ARGS, "layer past the pool refused");
    OmegaGpuKvLayout bad = ly; bad.block_size = 12;
    CHECK(omega_gpu_paged_attention_bf16(q, pool, &bad, ids, 1, 10, 0, 32, 4, 64, out, NULL) == OMEGA_GPU_ATTN_BAD_ARGS, "block size 12 (not a power of two) refused");
    bad = ly; bad.head_stride_bytes = 256;
    CHECK(omega_gpu_paged_attention_bf16(q, pool, &bad, ids, 1, 10, 0, 32, 4, 64, out, NULL) == OMEGA_GPU_ATTN_BAD_ARGS, "layout whose head stride is not head_dim*2 refused");
    free(pool);
    CHECK(strcmp(omega_gpu_attention_rc_name(OMEGA_GPU_ATTN_UNWRITTEN), "UNWRITTEN") == 0, "rc name");

    static const char *const must_bf16[] = { "MUFU.EX2", "MUFU.RCP", "FMNMX", "BAR.SYNC", "SHFL.DOWN", "@P0 BRA", "@!P0 BRA", "LDG.E.U16", "LDG.E ", "LDS R", "STS [R", "STG.E", "LOP3.LUT", "SHF.R.U32.HI" };
    static const char *const must_f32[] = { "MUFU.EX2", "MUFU.RCP", "FMNMX", "BAR.SYNC", "SHFL.DOWN", "@P0 BRA", "@!P0 BRA", "LDG.E ", "LDS R", "STS [R", "STG.E" };
    for (int f32 = 0; f32 < 2; f32++) {
        const char *name = f32 ? "gqa_f32" : "paged_bf16";
        OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
        int rc = omega_gpu_attention_codegen(f32 != 0, f32 ? 12 : 4, 3, &k);
        CHECK(rc == OMEGA_GPU_ATTN_OK && k.code_size > 0 && k.gpr_count <= 64, "%s kernel generated (rc %s, %zu insns, %u gprs)", name, omega_gpu_attention_rc_name(rc), k.insn_count, k.gpr_count);
        printf("codegen %s: %zu insns, %u gprs, %u ugprs\n", name, k.insn_count, k.gpr_count, k.uniform_gpr_count);
        uint8_t digest[32]; memcpy(digest, k.code_digest, 32);
        free(k.code);
        for (int m = 1; m <= 4; m++) {
            omega_gpu_attention_test_set_mutant((OmegaGpuAttnMutant)m);
            memset(&k, 0, sizeof k);
            CHECK(omega_gpu_attention_codegen(f32 != 0, f32 ? 12 : 4, 3, &k) == OMEGA_GPU_ATTN_OK && memcmp(digest, k.code_digest, 32) != 0, "%s mutant %d kernel differs", name, m);
            free(k.code);
            omega_gpu_attention_test_set_mutant(OMEGA_GPU_ATTN_MUTANT_NONE);
        }
        /* a different shape key is a different kernel */
        memset(&k, 0, sizeof k);
        CHECK(omega_gpu_attention_codegen(f32 != 0, f32 ? 12 : 4, 2, &k) == OMEGA_GPU_ATTN_OK && memcmp(digest, k.code_digest, 32) != 0, "%s log2gqa 2 kernel differs from log2gqa 3", name);
        free(k.code);
        CHECK(nvdisasm_check(f32 != 0, name, f32 ? must_f32 : must_bf16, f32 ? 11 : 14) == 0, "%s nvdisasm listing decodes and shows the expected instructions", name);
    }
}

/* ------------------------------------------------------------- chip gate */
typedef struct { const char *name; size_t n; int rc; size_t bad; double worst; int pass; uint64_t chip_ns, call_ns; uint32_t calls, unwritten;
                 int mutants_run, mutants_caught; char mutant_note[160]; } Case;
static Case g_cases[32]; static int g_ncases;
static int g_mutant_caught[5], g_mutant_run[5];

static void record(Case c, FILE *out) {
    g_cases[g_ncases++] = c;
    CHECK(c.pass, "%s rc=%s violations=%zu worst=%g", c.name, omega_gpu_attention_rc_name(c.rc), c.bad, c.worst);
    CHECK(c.mutants_caught == c.mutants_run, "%s mutants caught %d of %d (%s)", c.name, c.mutants_caught, c.mutants_run, c.mutant_note);
    printf("%s %s n=%zu rc=%s violations=%zu worst_scaled_err=%g chip_ns=%" PRIu64 " call_ns=%" PRIu64 " calls=%u unwritten=%u | mutants %d/%d %s\n",
           c.pass && c.mutants_caught == c.mutants_run ? "PASS" : "FAIL", c.name, c.n, omega_gpu_attention_rc_name(c.rc), c.bad, c.worst, c.chip_ns, c.call_ns, c.calls, c.unwritten,
           c.mutants_caught, c.mutants_run, c.mutant_note);
    if (out) fprintf(out, "%s{\"case\":\"%s\",\"n\":%zu,\"rc\":\"%s\",\"violations\":%zu,\"worst_scaled_err\":%g,\"pass\":%s,\"chip_elapsed_ns\":%" PRIu64 ",\"call_ns\":%" PRIu64 ",\"chip_calls\":%u,\"unwritten_words\":%u,\"mutants_run\":%d,\"mutants_caught\":%d,\"mutant_note\":\"%s\"}",
                     g_ncases > 1 ? "," : "", c.name, c.n, omega_gpu_attention_rc_name(c.rc), c.bad, c.worst, c.pass ? "true" : "false", c.chip_ns, c.call_ns, c.calls, c.unwritten,
                     c.mutants_run, c.mutants_caught, c.mutant_note);
}

static const char *mutant_name(int m) { static const char *n[5] = { "none", "NO_MAX", "KV_HEAD", "SLOT", "NO_RESCALE" }; return n[m]; }

/* Runs the mutants listed in `ms` (terminated by 0) through `run` and counts the catches. */
typedef int (*RunFn)(void *ctx, float *got, OmegaGpuAttnInfo *info);
static void run_mutants(Case *c, const int *ms, RunFn run, void *ctx, float *got, const float *want, size_t n, double rel, double abs_tol) {
    size_t pos = 0;
    for (int i = 0; ms[i]; i++) {
        OmegaGpuAttnInfo info; double w;
        omega_gpu_attention_test_set_mutant((OmegaGpuAttnMutant)ms[i]);
        int rc = run(ctx, got, &info);
        size_t bad = compare(got, want, n, rel, abs_tol, &w);
        int caught = rc != OMEGA_GPU_ATTN_OK || bad > 0;
        omega_gpu_attention_test_set_mutant(OMEGA_GPU_ATTN_MUTANT_NONE);
        c->mutants_run++; c->mutants_caught += caught; g_mutant_run[ms[i]]++; g_mutant_caught[ms[i]] += caught;
        pos += (size_t)snprintf(c->mutant_note + pos, sizeof c->mutant_note - pos, "%s%s:%s(rc=%s,bad=%zu)", pos ? " " : "", mutant_name(ms[i]), caught ? "CAUGHT" : "MISSED", omega_gpu_attention_rc_name(rc), bad);
        if (pos >= sizeof c->mutant_note) break;
    }
}

typedef struct { const float *q, *kc, *vc; uint32_t seq, nqh, nkv; } GqaCtx;
static int run_gqa(void *p, float *got, OmegaGpuAttnInfo *info) { GqaCtx *c = p; return omega_gpu_gqa_attention_f32(c->q, c->kc, c->vc, c->seq, c->nqh, c->nkv, 64, got, info); }
typedef struct { const float *q; const uint8_t *pool; const OmegaGpuKvLayout *ly; const uint32_t *ids; uint32_t nids, ctx, layer, nqh, nkv; } PagedCtx;
static int run_paged(void *p, float *got, OmegaGpuAttnInfo *info) { PagedCtx *c = p; return omega_gpu_paged_attention_bf16(c->q, c->pool, c->ly, c->ids, c->nids, c->ctx, c->layer, c->nqh, c->nkv, 64, got, info); }
typedef struct { const float *q; const uint8_t *pool; const OmegaGpuKvLayout *ly; const int32_t *tables, *ctxs; uint32_t maxb, nseq, layer, nqh, nkv; } BatchCtx;
static int run_batch(void *p, float *got, OmegaGpuAttnInfo *info) { BatchCtx *c = p; return omega_gpu_paged_attention_batch_bf16(c->q, c->pool, c->ly, c->tables, c->ctxs, c->maxb, c->nseq, c->layer, c->nqh, c->nkv, 64, got, info); }

static int chip(const char *out_path) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_ATTENTION_V1\",\"tolerances\":{\"rel\":2e-4,\"abs\":2e-5,\"large_scores_rel\":1e-3,\"large_scores_abs\":1e-4},\"cases\":[");
    uint32_t seed = 0xfb1c0005u;
    const uint32_t hd = 64;
    const struct { uint32_t nqh, nkv; const char *model; } shapes[2] = { { 32, 4, "tinyllama" }, { 32, 8, "llama32_1b" } };
    const uint32_t ctxs[4] = { 1, 17, 256, 2048 };
    const int no_mutants[1] = { 0 };

    /* 1. contiguous f32 KV (gqa_attention) */
    for (int sh = 0; sh < 2; sh++) for (int ci = 0; ci < 4; ci++) {
        uint32_t nqh = shapes[sh].nqh, nkv = shapes[sh].nkv, seq = ctxs[ci];
        size_t n = (size_t)nqh * hd, nkvs = (size_t)seq * nkv * hd;
        float *q = malloc(n * 4), *kc = malloc(nkvs * 4), *vc = malloc(nkvs * 4), *got = malloc(n * 4), *want = malloc(n * 4), *again = malloc(n * 4);
        for (size_t i = 0; i < n; i++) q[i] = frand(&seed, -1.0f, 1.0f);
        for (size_t i = 0; i < nkvs; i++) { kc[i] = frand(&seed, -1.0f, 1.0f); vc[i] = frand(&seed, -1.0f, 1.0f); }
        oracle_gqa(q, kc, vc, seq, nqh, nkv, hd, want);
        char name[96]; snprintf(name, sizeof name, "gqa_f32_%s_%uq%ukv_ctx%u", shapes[sh].model, nqh, nkv, seq);
        Case c = { .name = strdup(name), .n = n };
        OmegaGpuAttnInfo info;
        GqaCtx gc = { q, kc, vc, seq, nqh, nkv };
        c.rc = run_gqa(&gc, got, &info);
        c.bad = compare(got, want, n, 2e-4, 2e-5, &c.worst);
        c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; c.chip_ns = info.elapsed_ns; c.call_ns = info.call_ns; c.calls = info.chip_calls; c.unwritten = info.unwritten_words;
        if (ci == 3) {
            OmegaGpuAttnInfo i2; int rc2 = run_gqa(&gc, again, &i2);
            CHECK(rc2 == OMEGA_GPU_ATTN_OK && i2.kernel_cache_hit && memcmp(got, again, n * 4) == 0, "%s repeat run (cache hit) is bit-identical", name);
            /* negative control: the comparison must reject a wrong expectation (kv heads swapped) */
            float *wrong = malloc(n * 4); double w2;
            for (uint32_t h = 0; h < nqh; h++) memcpy(wrong + (size_t)h * hd, want + (size_t)((h + nqh / nkv) % nqh) * hd, hd * 4);
            size_t neg = compare(got, wrong, n, 2e-4, 2e-5, &w2);
            CHECK(neg > n / 2, "%s negative control: wrong expectation flagged (%zu of %zu)", name, neg, n);
            free(wrong);
            static const int ms[3] = { OMEGA_GPU_ATTN_MUTANT_SLOT, OMEGA_GPU_ATTN_MUTANT_NO_RESCALE, 0 };
            run_mutants(&c, ms, run_gqa, &gc, got, want, n, 2e-4, 2e-5);
        } else if (ci == 1 && sh == 1) {
            static const int ms[2] = { OMEGA_GPU_ATTN_MUTANT_KV_HEAD, 0 };
            run_mutants(&c, ms, run_gqa, &gc, got, want, n, 2e-4, 2e-5);
        } else run_mutants(&c, no_mutants, run_gqa, &gc, got, want, n, 2e-4, 2e-5);
        record(c, out); free(q); free(kc); free(vc); free(got); free(want); free(again);
    }
    /* 2. large scores (q scaled by 200): the softmax without the running max overflows */
    {
        uint32_t nqh = 32, nkv = 4, seq = 256; size_t n = (size_t)nqh * hd, nkvs = (size_t)seq * nkv * hd;
        float *q = malloc(n * 4), *kc = malloc(nkvs * 4), *vc = malloc(nkvs * 4), *got = malloc(n * 4), *want = malloc(n * 4);
        for (size_t i = 0; i < n; i++) q[i] = frand(&seed, -200.0f, 200.0f); /* scores ~ +-200 natural: EX2 without the max subtraction overflows f32 (2^128) */
        for (size_t i = 0; i < nkvs; i++) { kc[i] = frand(&seed, -1.0f, 1.0f); vc[i] = frand(&seed, -1.0f, 1.0f); }
        oracle_gqa(q, kc, vc, seq, nqh, nkv, hd, want);
        Case c = { .name = "gqa_f32_large_scores_32q4kv_ctx256", .n = n };
        OmegaGpuAttnInfo info; GqaCtx gc = { q, kc, vc, seq, nqh, nkv };
        c.rc = run_gqa(&gc, got, &info);
        c.bad = compare(got, want, n, 1e-3, 1e-4, &c.worst);
        c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; c.chip_ns = info.elapsed_ns; c.call_ns = info.call_ns; c.calls = info.chip_calls; c.unwritten = info.unwritten_words;
        static const int ms[2] = { OMEGA_GPU_ATTN_MUTANT_NO_MAX, 0 };
        run_mutants(&c, ms, run_gqa, &gc, got, want, n, 1e-3, 1e-4);
        record(c, out); free(q); free(kc); free(vc); free(got); free(want);
    }
    /* 3. paged bf16 KV, block size 16, 2 layers (layer 1 under test), scattered block ids */
    for (int sh = 0; sh < 2; sh++) for (int ci = 0; ci < 4; ci++) {
        uint32_t nqh = shapes[sh].nqh, nkv = shapes[sh].nkv, ctx = ctxs[ci], bs = 16, layers = 2, layer = 1;
        uint32_t nblk = (ctx + bs - 1) / bs, pool_blocks = nblk + 7;
        OmegaGpuKvLayout ly = layout_for(pool_blocks, bs, layers, nkv, hd);
        uint8_t *pool = malloc(ly.pool_bytes); uint16_t *pw = (uint16_t *)pool;
        for (size_t i = 0; i < ly.pool_bytes / 2; i++) pw[i] = f2bf(frand(&seed, -1.0f, 1.0f));
        uint32_t *ids = malloc(nblk * 4); /* a scattered permutation of the pool blocks */
        for (uint32_t b = 0; b < nblk; b++) ids[b] = (b * 7 + 3) % pool_blocks;
        size_t n = (size_t)nqh * hd;
        float *q = malloc(n * 4), *got = malloc(n * 4), *want = malloc(n * 4);
        for (size_t i = 0; i < n; i++) q[i] = frand(&seed, -1.0f, 1.0f);
        oracle_paged(q, pool, &ly, ids, nblk, ctx, layer, nqh, nkv, hd, want);
        char name[96]; snprintf(name, sizeof name, "paged_bf16_%s_%uq%ukv_bs%u_ctx%u", shapes[sh].model, nqh, nkv, bs, ctx);
        Case c = { .name = strdup(name), .n = n };
        OmegaGpuAttnInfo info; PagedCtx pc = { q, pool, &ly, ids, nblk, ctx, layer, nqh, nkv };
        c.rc = run_paged(&pc, got, &info);
        c.bad = compare(got, want, n, 2e-4, 2e-5, &c.worst);
        c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; c.chip_ns = info.elapsed_ns; c.call_ns = info.call_ns; c.calls = info.chip_calls; c.unwritten = info.unwritten_words;
        if (ci == 3) {
            /* NO_MAX cannot overflow on unit-scale scores (softmax is shift invariant); it is caught by the large-score gqa case */
            static const int ms3[4] = { OMEGA_GPU_ATTN_MUTANT_KV_HEAD, OMEGA_GPU_ATTN_MUTANT_SLOT, OMEGA_GPU_ATTN_MUTANT_NO_RESCALE, 0 };
            run_mutants(&c, ms3, run_paged, &pc, got, want, n, 2e-4, 2e-5);
        } else if (ci == 1) {
            static const int ms[3] = { OMEGA_GPU_ATTN_MUTANT_KV_HEAD, OMEGA_GPU_ATTN_MUTANT_SLOT, 0 };
            run_mutants(&c, ms, run_paged, &pc, got, want, n, 2e-4, 2e-5);
        } else if (ci == 0) {
            /* one token: its softmax weight is 1 whatever K says, so a K-only slot shift is invisible; KV_HEAD still shows */
            static const int ms[2] = { OMEGA_GPU_ATTN_MUTANT_KV_HEAD, 0 };
            run_mutants(&c, ms, run_paged, &pc, got, want, n, 2e-4, 2e-5);
        } else run_mutants(&c, no_mutants, run_paged, &pc, got, want, n, 2e-4, 2e-5);
        record(c, out); free(pool); free(ids); free(q); free(got); free(want);
    }
    /* 4. batch: three sequences (short, empty, truncated by its table), a negative table entry, table rows of 16 */
    {
        uint32_t nqh = 32, nkv = 8, bs = 16, layers = 3, layer = 2, maxb = 16, nseq = 3, pool_blocks = 40;
        OmegaGpuKvLayout ly = layout_for(pool_blocks, bs, layers, nkv, hd);
        uint8_t *pool = malloc(ly.pool_bytes); uint16_t *pw = (uint16_t *)pool;
        for (size_t i = 0; i < ly.pool_bytes / 2; i++) pw[i] = f2bf(frand(&seed, -1.0f, 1.0f));
        int32_t *tables = malloc((size_t)nseq * maxb * 4);
        for (size_t i = 0; i < (size_t)nseq * maxb; i++) tables[i] = -1;
        /* seq 0: ctx 40 over blocks 5, 17 (-1 in the middle closes up), 29 */
        tables[0] = 5; tables[1] = -1; tables[2] = 17; tables[3] = 29;
        /* seq 1: ctx 0 with a table */
        tables[maxb + 0] = 1;
        /* seq 2: ctx 100 but only 4 blocks (64 tokens used) */
        tables[2 * maxb + 0] = 30; tables[2 * maxb + 1] = 2; tables[2 * maxb + 2] = 33; tables[2 * maxb + 3] = 11;
        int32_t ctxs2[3] = { 40, 0, 100 };
        size_t n = (size_t)nseq * nqh * hd;
        float *q = malloc(n * 4), *got = malloc(n * 4), *want = malloc(n * 4);
        for (size_t i = 0; i < n; i++) q[i] = frand(&seed, -1.0f, 1.0f);
        oracle_batch(q, pool, &ly, tables, ctxs2, maxb, nseq, layer, nqh, nkv, hd, want);
        Case c = { .name = "paged_bf16_batch_3seq_32q8kv_ctx40_0_100", .n = n };
        OmegaGpuAttnInfo info; BatchCtx bc = { q, pool, &ly, tables, ctxs2, maxb, nseq, layer, nqh, nkv };
        c.rc = run_batch(&bc, got, &info);
        c.bad = compare(got, want, n, 2e-4, 2e-5, &c.worst);
        c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; c.chip_ns = info.elapsed_ns; c.call_ns = info.call_ns; c.calls = info.chip_calls; c.unwritten = info.unwritten_words;
        int zeros = 1; for (size_t i = 0; i < (size_t)nqh * hd; i++) if (got[(size_t)nqh * hd + i] != 0.0f) zeros = 0;
        CHECK(zeros, "batch: the empty sequence is all zeros");
        static const int ms[3] = { OMEGA_GPU_ATTN_MUTANT_KV_HEAD, OMEGA_GPU_ATTN_MUTANT_SLOT, 0 };
        run_mutants(&c, ms, run_batch, &bc, got, want, n, 2e-4, 2e-5);
        record(c, out); free(pool); free(tables); free(q); free(got); free(want);
    }
    for (int m = 1; m <= 4; m++) {
        CHECK(g_mutant_run[m] > 0 && g_mutant_caught[m] == g_mutant_run[m], "mutant %s caught %d of %d runs", mutant_name(m), g_mutant_caught[m], g_mutant_run[m]);
        printf("mutant %s: caught %d of %d\n", mutant_name(m), g_mutant_caught[m], g_mutant_run[m]);
    }
    CHECK(omega_gpu_session_open_count() == (g_sim_mode ? 0u : 1u), "device opened %s for the battery (opens=%u)", g_sim_mode ? "never (simulator)" : "once", omega_gpu_session_open_count());
    printf("device_opens=%u\n", omega_gpu_session_open_count());
    if (out) {
        fprintf(out, "],\"mutants\":{");
        for (int m = 1; m <= 4; m++) fprintf(out, "%s\"%s\":{\"run\":%d,\"caught\":%d}", m > 1 ? "," : "", mutant_name(m), g_mutant_run[m], g_mutant_caught[m]);
        fprintf(out, "},\"device_opens\":%u,\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", omega_gpu_session_open_count(), g_checks, g_failed, g_failed == 0 ? "OMEGA_GPU_ATTENTION_PASS" : "OMEGA_GPU_ATTENTION_FAIL");
        fclose(out);
    }
    return 0;
}

/* ------------------------------------------------------------ host simulator
 * Executes the allocated IR program (physical registers, so the register allocator's
 * loop handling is under test too) for every CTA of a launch: 64 threads, 1 KB shared,
 * host pointers as addresses, SHFL and BAR as lockstep points (all threads of the CTA
 * must sit at the same instruction). EX2 and RCP are exact here; the chip's are approximate. */
typedef struct { uint32_t r[256]; int pc; int p0; int done; } SimThr;
static uint32_t sim_cbank[256];
static uint8_t sim_shared[1024];
#define SIM_T 64
static int g_sim_err;
static int g_trace, g_trace_launch; static uint32_t g_trace_cx;
static uint32_t rd(const SimThr *t, int v, const BlackwellIRProgram *p) { if (v < 0) return 0; int ph = p->regalloc.vreg_to_phys[v]; return ph < 0 ? 0 : t->r[ph & 0xff]; }
static uint64_t rd64(const SimThr *t, int v, const BlackwellIRProgram *p) { if (v < 0) return 0; int ph = p->regalloc.vreg_to_phys[v] & 0xff; return (uint64_t)t->r[ph] | ((uint64_t)t->r[ph + 1] << 32); }
static void wr(SimThr *t, int v, uint32_t x, const BlackwellIRProgram *p) { if (v < 0) return; int ph = p->regalloc.vreg_to_phys[v]; if (ph >= 0) t->r[ph & 0xff] = x; }
static void wr64(SimThr *t, int v, uint64_t x, const BlackwellIRProgram *p) { if (v < 0) return; int ph = p->regalloc.vreg_to_phys[v] & 0xff; t->r[ph] = (uint32_t)x; t->r[ph + 1] = (uint32_t)(x >> 32); }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u_(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
/* run one thread until it reaches a SHFL or BAR (not executed) or EXIT; returns 1 if parked at a sync op */
static int sim_step(SimThr *t, const BlackwellIRProgram *p, uint32_t tid, uint32_t cx, uint32_t cy) {
    for (int guard = 0; guard < 50000000; guard++) {
        if (t->done) return 0;
        if (t->pc < 0 || (size_t)t->pc >= p->count) { g_sim_err = 1; t->done = 1; return 0; }
        const BlackwellIRInsn *in = &p->insns[t->pc];
        uint32_t a = rd(t, in->src1_vreg, p), b = rd(t, in->src2_vreg, p), c = rd(t, in->src3_vreg, p);
        switch (in->op) {
        case BW_IR_SHFL_DOWN: case BW_IR_BAR_SYNC: return 1;
        case BW_IR_EXIT: if (!in->predicate_p0 || t->p0) { t->done = 1; return 0; } break;
        case BW_IR_BRA: {
            int take = !in->predicate_p0 || (in->predicate_not ? !t->p0 : t->p0);
            if (take) { if ((int32_t)in->imm == 0) { g_sim_err = 2; t->done = 1; return 0; } t->pc += (int32_t)in->imm; continue; }
            break; }
        case BW_IR_MOV_IMM: wr(t, in->dst_vreg, in->imm, p); break;
        case BW_IR_MOV_RZ: wr(t, in->dst_vreg, 0, p); break;
        case BW_IR_S2R: wr(t, in->dst_vreg, in->imm == BW_SR_TID_X ? tid : in->imm == BW_SR_CTAID_X ? cx : in->imm == BW_SR_CTAID_Y ? cy : 0, p); break;
        case BW_IR_LDC: wr(t, in->dst_vreg, sim_cbank[in->imm / 4], p); break;
        case BW_IR_LDC64: wr64(t, in->dst_vreg, (uint64_t)sim_cbank[in->imm / 4] | ((uint64_t)sim_cbank[in->imm / 4 + 1] << 32), p); break;
        case BW_IR_LDCU64: break; /* memory descriptor: not modelled */
        case BW_IR_IMAD: wr(t, in->dst_vreg, in->src2_vreg >= 0 ? a * b + c : a * in->imm + c, p); break;
        case BW_IR_IMAD_WIDE: wr64(t, in->dst_vreg, (uint64_t)a * in->imm + rd64(t, in->src3_vreg, p), p); break;
        case BW_IR_IADD3: wr(t, in->dst_vreg, a + b, p); break;
        case BW_IR_SHF_R: wr(t, in->dst_vreg, a >> (in->imm & 31), p); break;
        case BW_IR_LOP3_AND: wr(t, in->dst_vreg, a & in->imm, p); break;
        case BW_IR_LOP3_XOR: wr(t, in->dst_vreg, a ^ b, p); break;
        case BW_IR_ISETP_GE_U32: t->p0 = a >= b; break;
        case BW_IR_LDG_E: { uint32_t x; memcpy(&x, (const void *)(uintptr_t)rd64(t, in->src1_vreg, p), 4); wr(t, in->dst_vreg, x, p); break; }
        case BW_IR_LDG_E_U16: { uint16_t x; memcpy(&x, (const void *)(uintptr_t)rd64(t, in->src1_vreg, p), 2); wr(t, in->dst_vreg, x, p); break; }
        case BW_IR_STG_E: memcpy((void *)(uintptr_t)rd64(t, in->src1_vreg, p), &b, 4); break;
        case BW_IR_LDS32: { uint32_t x; if (a + 4 > sizeof sim_shared) { g_sim_err = 3; t->done = 1; return 0; } memcpy(&x, sim_shared + a, 4); wr(t, in->dst_vreg, x, p); break; }
        case BW_IR_STS32: if (a + 4 > sizeof sim_shared) { g_sim_err = 3; t->done = 1; return 0; } memcpy(sim_shared + a, &b, 4); break;
        case BW_IR_FADD: wr(t, in->dst_vreg, f2u_(u2f(a) + u2f(b)), p); break;
        case BW_IR_FSUB: wr(t, in->dst_vreg, f2u_(u2f(a) - u2f(b)), p); break;
        case BW_IR_FMUL: wr(t, in->dst_vreg, f2u_(u2f(a) * u2f(b)), p); break;
        case BW_IR_FFMA: wr(t, in->dst_vreg, f2u_(fmaf(u2f(a), u2f(b), u2f(c))), p); break;
        case BW_IR_FMNMX_MAX: wr(t, in->dst_vreg, f2u_(fmaxf(u2f(a), u2f(b))), p); break;
        case BW_IR_MUFU_EX2: wr(t, in->dst_vreg, f2u_(exp2f(u2f(a))), p); break;
        case BW_IR_MUFU_RCP: wr(t, in->dst_vreg, f2u_(1.0f / u2f(a)), p); break;
        default: g_sim_err = 100 + (int)in->op; t->done = 1; return 0;
        }
        if (g_trace && tid == 0 && cx == g_trace_cx && cy == 0 && g_trace_launch == 1) { uint32_t dv = rd(t, in->dst_vreg, p); printf("T pc=%d op=%d dst=v%d(r%d)=%08x f=%g p0=%d a=%08x b=%08x c=%08x\n", t->pc, (int)in->op, in->dst_vreg, in->dst_vreg >= 0 ? p->regalloc.vreg_to_phys[in->dst_vreg] : -1, dv, u2f(dv), t->p0, a, b, c); }
        t->pc++;
    }
    g_sim_err = 4; t->done = 1; return 0;
}
static int sim_launch(const void *progv, const OmegaGpuAttnLaunch *L, OmegaGpuAttnInfo *info) {
    const BlackwellIRProgram *p = progv;
    memset(sim_cbank, 0, sizeof sim_cbank);
    uint64_t qa = (uintptr_t)L->q, pa = (uintptr_t)L->pool, oa = (uintptr_t)L->out, ta = (uintptr_t)L->tab;
    sim_cbank[0x380 / 4] = (uint32_t)qa; sim_cbank[0x380 / 4 + 1] = (uint32_t)(qa >> 32);
    sim_cbank[0x388 / 4] = (uint32_t)pa; sim_cbank[0x388 / 4 + 1] = (uint32_t)(pa >> 32);
    sim_cbank[0x390 / 4] = (uint32_t)oa; sim_cbank[0x390 / 4 + 1] = (uint32_t)(oa >> 32);
    sim_cbank[0x398 / 4] = (uint32_t)ta; sim_cbank[0x398 / 4 + 1] = (uint32_t)(ta >> 32);
    memcpy(&sim_cbank[0x3a0 / 4], L->params, sizeof L->params);
    uint32_t *outw = L->out; for (size_t i = 0; i < L->out_bytes / 4; i++) outw[i] = 0xffbadbadu;
    g_sim_err = 0;
    static SimThr th[SIM_T];
    g_trace_launch++; if (getenv("ATTN_SIM_TRACE")) { g_trace = 1; g_trace_cx = (uint32_t)atoi(getenv("ATTN_SIM_TRACE")); }
    for (uint32_t cy = 0; cy < L->grid_y && !g_sim_err; cy++) for (uint32_t cx = 0; cx < L->grid_x && !g_sim_err; cx++) {
        memset(th, 0, sizeof th); memset(sim_shared, 0, sizeof sim_shared);
        for (;;) {
            int parked = 0, alive = 0;
            for (int i = 0; i < SIM_T; i++) { if (th[i].done) continue; alive++; parked += sim_step(&th[i], p, (uint32_t)i, cx, cy); }
            if (g_sim_err) break;
            if (alive == 0) break;
            if (parked == 0) continue; /* every live thread exited this round */
            if (parked != alive) { g_sim_err = 5; break; } /* a thread exited while others wait at a barrier */
            int pc = -1;
            for (int i = 0; i < SIM_T; i++) if (!th[i].done) { if (pc < 0) pc = th[i].pc; else if (th[i].pc != pc) g_sim_err = 6; }
            if (g_sim_err) break;
            const BlackwellIRInsn *in = &p->insns[pc];
            if (in->op == BW_IR_SHFL_DOWN) {
                uint32_t src[SIM_T];
                for (int i = 0; i < SIM_T; i++) src[i] = rd(&th[i], in->src1_vreg, p);
                for (int i = 0; i < SIM_T; i++) { int lane = i & 31, from = lane + (int)(in->imm & 31); wr(&th[i], in->dst_vreg, from < 32 ? src[(i & ~31) + from] : src[i], p); }
            }
            for (int i = 0; i < SIM_T; i++) th[i].pc++;
        }
    }
    uint32_t unwritten = 0; for (size_t i = 0; i < L->out_bytes / 4; i++) if (outw[i] == 0xffbadbadu) unwritten++;
    if (info) { info->chip_calls++; info->unwritten_words += unwritten; info->ctas_last_launch = L->grid_x * L->grid_y; info->threads_per_cta = SIM_T; }
    if (g_sim_err) { printf("simulator error %d\n", g_sim_err); return OMEGA_GPU_ATTN_CHIP_FAIL; }
    return unwritten ? OMEGA_GPU_ATTN_UNWRITTEN : OMEGA_GPU_ATTN_OK;
}


/* ---------------------------------------------------- timing (cut 4b) */
/* One query step of TinyLlama attention at context 2048 (32 q heads, 4 kv heads,
 * f32 KV) 100 times through the persistent session, every call checked against the
 * oracle; gate: median wall time per call under 3 ms, one device open per process. */
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static int timing(const char *out_path) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    enum { N = 100 };
    const uint32_t hd = 64, nqh = 32, nkv = 4, seq = 2048;
    const char *name = "gqa_f32_tinyllama_32q4kv_ctx2048";
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_ATTENTION_TIMING_V1\",\"gate_median_ms\":3.0,\"calls\":%d,\"cases\":[", N);
    uint32_t seed = 0xfb1c004bu;
    size_t n = (size_t)nqh * hd, nkvs = (size_t)seq * nkv * hd;
    float *q = malloc(n * 4), *kc = malloc(nkvs * 4), *vc = malloc(nkvs * 4), *got = malloc(n * 4), *want = malloc(n * 4);
    for (size_t i = 0; i < n; i++) q[i] = frand(&seed, -1.0f, 1.0f);
    for (size_t i = 0; i < nkvs; i++) { kc[i] = frand(&seed, -1.0f, 1.0f); vc[i] = frand(&seed, -1.0f, 1.0f); }
    oracle_gqa(q, kc, vc, seq, nqh, nkv, hd, want);
    double samples[N]; int ok_calls = 0; size_t bad_total = 0; double worst = 0; uint64_t chip_sum = 0; int rc_last = 0;
    OmegaGpuAttnInfo info;
    /* warm-up: the first call opens the device and generates the kernel (not timed) */
    int rc0 = omega_gpu_gqa_attention_f32(q, kc, vc, seq, nqh, nkv, hd, got, &info);
    double first_ms = info.call_ns / 1e6;
    CHECK(rc0 == OMEGA_GPU_ATTN_OK, "%s first call rc=%s err=\"%s\"", name, omega_gpu_attention_rc_name(rc0), omega_gpu_attention_last_error());
    for (int i = 0; i < N; i++) {
        double t0 = now_ms();
        int rc = omega_gpu_gqa_attention_f32(q, kc, vc, seq, nqh, nkv, hd, got, &info);
        samples[i] = now_ms() - t0;
        double w; size_t bad = compare(got, want, n, 2e-4, 2e-5, &w);
        if (w > worst) worst = w;
        bad_total += bad; chip_sum += info.elapsed_ns; rc_last = rc;
        if (rc == OMEGA_GPU_ATTN_OK && bad == 0) ok_calls++;
    }
    qsort(samples, N, sizeof samples[0], cmp_d);
    double median = samples[N / 2], p90 = samples[N * 9 / 10], mn = samples[0], mx = samples[N - 1];
    uint32_t opens = omega_gpu_session_open_count();
    int pass = ok_calls == N && bad_total == 0 && median < 3.0 && opens == 1;
    CHECK(pass, "timing %s ok_calls=%d/%d violations=%zu median_ms=%.3f opens=%u last_rc=%s err=\"%s\"", name, ok_calls, N, bad_total, median, opens, omega_gpu_attention_rc_name(rc_last), omega_gpu_attention_last_error());
    printf("%s timing %s first_call_ms=%.3f ok_calls=%d/%d violations=%zu worst_scaled_err=%g median_ms=%.3f p90_ms=%.3f min_ms=%.3f max_ms=%.3f chip_us_mean=%.1f device_opens=%u\n",
           pass ? "PASS" : "FAIL", name, first_ms, ok_calls, N, bad_total, worst, median, p90, mn, mx, chip_sum / 1e3 / N, opens);
    if (out) fprintf(out, "{\"case\":\"%s\",\"first_call_ms\":%.3f,\"ok_calls\":%d,\"violations\":%zu,\"worst_scaled_err\":%g,\"median_ms\":%.4f,\"p90_ms\":%.4f,\"min_ms\":%.4f,\"max_ms\":%.4f,\"chip_us_mean\":%.1f,\"device_opens\":%u,\"pass\":%s}",
                     name, first_ms, ok_calls, bad_total, worst, median, p90, mn, mx, chip_sum / 1e3 / N, opens, pass ? "true" : "false");
    free(q); free(kc); free(vc); free(got); free(want);
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed, g_failed == 0 ? "OMEGA_GPU_ATTENTION_TIMING_PASS" : "OMEGA_GPU_ATTENTION_TIMING_FAIL"); fclose(out); }
    return 0;
}

int main(int argc, char **argv) {
    const char *out_path = NULL; int host = 0, sim = 0, timing_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--timing") == 0) timing_mode = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    }
    if (sim) { g_sim_mode = 1; omega_gpu_attention_test_set_simulator(sim_launch); }
    if (host) host_only();
    else if (timing_mode) { if (timing(out_path) != 0) return 2; }
    else if (chip(out_path) != 0) return 2;
    printf("%s: %d checks, %d failed\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
