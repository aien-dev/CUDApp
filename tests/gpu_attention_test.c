/* gpu_attention_test: FB-1 cut 5 gate for omega_gpu_attention_api.
 *
 *   ./gpu_attention_test --host-only          refusals, codegen (both kernel families and every
 *                                             mutant), register budget, nvdisasm listing check
 *   ./gpu_attention_test --split-timing [--hd 128] [--spin-us N] [--out r.json]  paged bf16 32q/8kv at ATTN_SPLIT_CTX
 *                                             (default 1268): 100 calls per forced split 1,2,4,8 + auto, parity
 *   ./gpu_attention_test --timing [--out r.json] cut 4b gate: 1 query x 32 heads x 2048 ctx, 100 calls
 *                                             through the persistent session, median under 3 ms
 *   ./gpu_attention_test --sweep [--out s.json]  head-scaling sweep: paged bf16 at head_dim 64 over q heads
 *                                             1..64 x every kv-head divisor x ctx 256/2048, parity + timings
 *                                             (receipt OMEGA_GPU_ATTENTION_SWEEP_V1; --sim marks host timings)
 *   ./gpu_attention_test --out receipt.json   chip gate: every shape against the host oracle, the
 *                                             mutants (deliberately wrong kernels, all of
 *                                             1 .. OMEGA_GPU_ATTN_MUTANT_COUNT-1) which the same
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
#include "bw_warp_sim.h" /* omega #308: SIMT warp simulator for --divergent */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int g_checks, g_failed;
static int g_sim_mode; /* --sim: the host simulator never opens the device */
static int g_divergent; /* --divergent (omega #308 R2): the j >= ctx branch inside a BSSY/BSYNC region */
static uint32_t g_hd = 64; /* --hd 128: the whole battery at head_dim 128 (Qwen3-4B shapes); both simulators size the CTA from it */
/* Host-only register bound (omega#470). hd 64 kernels are the program from before cut K (#417 keeps them byte-identical,
 * see the hd64 digest pins) and stay under 64. hd 128 follows AT_GPR_BUDGET in src/omega_gpu_attention_api.c (96 since
 * cut K, chip battery 225/225): the codegen refuses any kernel above it and every launch declares exactly that many
 * registers, so a peak under it changes nothing on the chip. The pinned-digest check further down uses the same 96. */
#define ATTN_GPR_BOUND (g_hd == 64 ? 64u : 96u)
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
/* One instruction line of an nvdisasm listing: leading blanks, then the byte offset as a hex comment, "/" "*" hex "*" "/".
 * omega#470: the old test matched the text "/" "*" "0", which stops at offset 0x1000, so every listing looked cut off at
 * 256 instructions and any opcode first emitted past 4 KiB (the hd 128 kernels are 512 and 624 instructions) was missed. */
static bool is_sass_insn_line(const char *line) {
    while (*line == ' ' || *line == '\t') line++;
    if (line[0] != '/' || line[1] != '*') return false;
    const char *p = line + 2;
    int digits = 0;
    while ((*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'f') || (*p >= 'A' && *p <= 'F')) { p++; digits++; }
    return digits >= 4 && p[0] == '*' && p[1] == '/';
}

static int nvdisasm_check(bool f32, const char *name, const char *const *must, int n_must) {
    const char *nv = "/usr/local/cuda/bin/nvdisasm";
    if (access(nv, X_OK) != 0) { printf("nvdisasm: not present, listing check skipped for %s\n", name); return 0; }
    OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
    if (omega_gpu_attention_codegen_hd(f32, f32 ? 12 : 4, 3, g_hd, &k) != OMEGA_GPU_ATTN_OK) return -1;
    char bin[64], lst[64], cmd[256];
    snprintf(bin, sizeof bin, "/tmp/omega_attn_%s_%d.bin", name, (int)getpid());
    snprintf(lst, sizeof lst, "/tmp/omega_attn_%s_%d.sass", name, (int)getpid());
    FILE *f = fopen(bin, "wb"); if (!f) { free(k.code); return -1; }
    fwrite(k.code, 1, k.code_size, f); fclose(f);
    snprintf(cmd, sizeof cmd, "%s -b SM121 %s > %s 2>&1", nv, bin, lst);
    int rc = system(cmd);
    const int want_insns = (int)(k.code_size / 16); /* every 16-byte slot, padding included, is one listing line */
    free(k.code);
    FILE *l = fopen(lst, "r"); if (!l) return -1;
    char line[512]; int seen[16] = {0}; int bad = 0, insns = 0;
    while (fgets(line, sizeof line, l)) {
        if (!is_sass_insn_line(line)) continue;
        insns++;
        if (strstr(line, "?") || strstr(line, "error")) bad++;
        for (int i = 0; i < n_must; i++) if (strstr(line, must[i])) seen[i] = 1;
    }
    fclose(l);
    int miss = 0;
    for (int i = 0; i < n_must; i++) if (!seen[i]) { printf("  %s: nvdisasm listing lacks %s\n", name, must[i]); miss++; }
    printf("nvdisasm %s: rc=%d insns=%d of %d undecodable=%d missing=%d (%s)\n", name, rc, insns, want_insns, bad, miss, lst);
    unlink(bin);
    return (rc == 0 && insns == want_insns && bad == 0 && miss == 0) ? 0 : -1;
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
    CHECK(omega_gpu_gqa_attention_f32(q, kv, kv, 1, 32, 4, 96, out, NULL) == OMEGA_GPU_ATTN_TOO_LARGE, "head_dim 96 refused (TOO_LARGE)");
    CHECK(omega_gpu_gqa_attention_f32(q, kv, kv, 1, 32, 4, 256, out, NULL) == OMEGA_GPU_ATTN_TOO_LARGE, "head_dim 256 refused (TOO_LARGE)");
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

    /* with --divergent the listing must also show the region ops (last two entries) */
    static const char *const must_bf16[] = { "MUFU.EX2", "MUFU.RCP", "FMNMX", "BAR.SYNC", "SHFL.DOWN", "@P0 BRA", "@!P0 BRA", "LDG.E.U16", "LDG.E ", "LDG.E.128", "LDS.128", "LDS R", "STS [R", "STG.E", "LOP3.LUT", "SHF.R.U32.HI", "BSSY", "BSYNC" };
    static const char *const must_f32[] = { "MUFU.EX2", "MUFU.RCP", "FMNMX", "BAR.SYNC", "SHFL.DOWN", "@P0 BRA", "@!P0 BRA", "LDG.E ", "LDG.E.128", "LDS.128", "LDS R", "STS [R", "STG.E", "BSSY", "BSYNC" };
    for (int f32 = 0; f32 < 2; f32++) {
        const char *name = f32 ? "gqa_f32" : "paged_bf16";
        OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
        int rc = omega_gpu_attention_codegen_hd(f32 != 0, f32 ? 12 : 4, 3, g_hd, &k);
        CHECK(rc == OMEGA_GPU_ATTN_OK && k.code_size > 0 && k.gpr_count <= ATTN_GPR_BOUND, "%s kernel generated (rc %s, %zu insns, %u gprs)", name, omega_gpu_attention_rc_name(rc), k.insn_count, k.gpr_count);
        printf("codegen %s: %zu insns, %u gprs, %u ugprs\n", name, k.insn_count, k.gpr_count, k.uniform_gpr_count);
        /* every mutant (1 .. COUNT-1) is a distinct kernel: different from the baseline and from
         * every other mutant, named, and inside the register budget */
        uint8_t digests[OMEGA_GPU_ATTN_MUTANT_COUNT][32]; memcpy(digests[0], k.code_digest, 32);
        free(k.code);
        for (int m = 1; m < OMEGA_GPU_ATTN_MUTANT_COUNT; m++) {
            omega_gpu_attention_test_set_mutant((OmegaGpuAttnMutant)m);
            memset(&k, 0, sizeof k);
            int mrc = omega_gpu_attention_codegen_hd(f32 != 0, f32 ? 12 : 4, 3, g_hd, &k);
            CHECK(mrc == OMEGA_GPU_ATTN_OK && k.gpr_count <= ATTN_GPR_BOUND, "%s mutant %s generated (rc %s, %u gprs)", name, omega_gpu_attention_mutant_name((OmegaGpuAttnMutant)m), omega_gpu_attention_rc_name(mrc), k.gpr_count);
            CHECK(strcmp(omega_gpu_attention_mutant_name((OmegaGpuAttnMutant)m), "?") != 0, "mutant %d has a name", m);
            memcpy(digests[m], k.code_digest, 32);
            free(k.code);
            omega_gpu_attention_test_set_mutant(OMEGA_GPU_ATTN_MUTANT_NONE);
            for (int o = 0; o < m; o++)
                CHECK(memcmp(digests[o], digests[m], 32) != 0, "%s mutant %s kernel differs from %s", name, omega_gpu_attention_mutant_name((OmegaGpuAttnMutant)m), omega_gpu_attention_mutant_name((OmegaGpuAttnMutant)o));
        }
        CHECK(strcmp(omega_gpu_attention_mutant_name(OMEGA_GPU_ATTN_MUTANT_COUNT), "?") == 0, "COUNT has no name");
        /* a different shape key is a different kernel */
        memset(&k, 0, sizeof k);
        CHECK(omega_gpu_attention_codegen_hd(f32 != 0, f32 ? 12 : 4, 2, g_hd, &k) == OMEGA_GPU_ATTN_OK && memcmp(digests[0], k.code_digest, 32) != 0, "%s log2gqa 2 kernel differs from log2gqa 3", name);
        free(k.code);
        CHECK(nvdisasm_check(f32 != 0, name, f32 ? must_f32 : must_bf16, (f32 ? 11 : 14) + (g_divergent ? 2 : 0)) == 0, "%s nvdisasm listing decodes and shows the expected instructions", name);
    }
}

/* ------------------------------------------------------------- chip gate */
typedef struct { const char *name; size_t n; int rc; size_t bad; double worst; int pass; uint64_t chip_ns, call_ns; uint32_t calls, unwritten;
                 int mutants_run, mutants_caught; char mutant_note[400];
                 uint64_t kv_staged, kv_naive; uint32_t blk_logical, blk_unique; } Case;
static void case_info(Case *c, const OmegaGpuAttnInfo *i) {
    c->chip_ns = i->elapsed_ns; c->call_ns = i->call_ns; c->calls = i->chip_calls; c->unwritten = i->unwritten_words;
    c->kv_staged = i->kv_bytes_staged; c->kv_naive = i->kv_bytes_naive; c->blk_logical = i->kv_blocks_logical; c->blk_unique = i->kv_blocks_unique;
}
static Case g_cases[64]; static int g_ncases;
static int g_mutant_caught[OMEGA_GPU_ATTN_MUTANT_COUNT], g_mutant_run[OMEGA_GPU_ATTN_MUTANT_COUNT];

static void record(Case c, FILE *out) {
    g_cases[g_ncases++] = c;
    CHECK(c.pass, "%s rc=%s violations=%zu worst=%g", c.name, omega_gpu_attention_rc_name(c.rc), c.bad, c.worst);
    CHECK(c.mutants_caught == c.mutants_run, "%s mutants caught %d of %d (%s)", c.name, c.mutants_caught, c.mutants_run, c.mutant_note);
    printf("%s %s n=%zu rc=%s violations=%zu worst_scaled_err=%g chip_ns=%" PRIu64 " call_ns=%" PRIu64 " calls=%u unwritten=%u kv_staged=%" PRIu64 "/%" PRIu64 " blocks=%u/%u | mutants %d/%d %s\n",
           c.pass && c.mutants_caught == c.mutants_run ? "PASS" : "FAIL", c.name, c.n, omega_gpu_attention_rc_name(c.rc), c.bad, c.worst, c.chip_ns, c.call_ns, c.calls, c.unwritten,
           c.kv_staged, c.kv_naive, c.blk_unique, c.blk_logical, c.mutants_caught, c.mutants_run, c.mutant_note);
    if (out) fprintf(out, "%s{\"case\":\"%s\",\"n\":%zu,\"rc\":\"%s\",\"violations\":%zu,\"worst_scaled_err\":%g,\"pass\":%s,\"chip_elapsed_ns\":%" PRIu64 ",\"call_ns\":%" PRIu64 ",\"chip_calls\":%u,\"unwritten_words\":%u,"
                     "\"kv_bytes_staged\":%" PRIu64 ",\"kv_bytes_naive\":%" PRIu64 ",\"kv_blocks_unique\":%u,\"kv_blocks_logical\":%u,\"mutants_run\":%d,\"mutants_caught\":%d,\"mutant_note\":\"%s\"}",
                     g_ncases > 1 ? "," : "", c.name, c.n, omega_gpu_attention_rc_name(c.rc), c.bad, c.worst, c.pass ? "true" : "false", c.chip_ns, c.call_ns, c.calls, c.unwritten,
                     c.kv_staged, c.kv_naive, c.blk_unique, c.blk_logical, c.mutants_run, c.mutants_caught, c.mutant_note);
}

static const char *mutant_name(int m) { return omega_gpu_attention_mutant_name((OmegaGpuAttnMutant)m); }

/* Runs the mutants listed in `ms` (terminated by 0) through `run` and counts the catches. */
typedef int (*RunFn)(void *ctx, float *got, OmegaGpuAttnInfo *info);
/* The forced launch split the battery currently holds (the engine hook has no getter); -1 = automatic. */
static int g_forced_split = -1;
static void set_split(int log2split) { g_forced_split = log2split; omega_gpu_attention_test_set_split(log2split); }

static void run_mutants(Case *c, const int *ms, RunFn run, void *ctx, float *got, const float *want, size_t n, double rel, double abs_tol) {
    size_t pos = strlen(c->mutant_note); /* appends: a case may call this more than once */
    for (int i = 0; ms[i]; i++) {
        OmegaGpuAttnInfo info; double w;
        omega_gpu_attention_test_set_mutant((OmegaGpuAttnMutant)ms[i]);
        /* cut J (one chunk per split): a paged call at up to 16 chunks gives every CTA one chunk, so the in-kernel
         * rescale between chunks never runs and the NO_RESCALE mutant would be invisible. That mutant runs with the
         * whole context in one CTA per head (forced split 1), where the rescale happens on every chunk after the first. */
        const int one_cta = ms[i] == OMEGA_GPU_ATTN_MUTANT_NO_RESCALE;
        if (one_cta) omega_gpu_attention_test_set_split(0);
        int rc = run(ctx, got, &info);
        if (one_cta) omega_gpu_attention_test_set_split(g_forced_split);
        size_t bad = compare(got, want, n, rel, abs_tol, &w);
        int caught = rc != OMEGA_GPU_ATTN_OK || bad > 0;
        omega_gpu_attention_test_set_mutant(OMEGA_GPU_ATTN_MUTANT_NONE);
        c->mutants_run++; c->mutants_caught += caught; g_mutant_run[ms[i]]++; g_mutant_caught[ms[i]] += caught;
        pos += (size_t)snprintf(c->mutant_note + pos, sizeof c->mutant_note - pos, "%s%s:%s(rc=%s,bad=%zu)", pos ? " " : "", mutant_name(ms[i]), caught ? "CAUGHT" : "MISSED", omega_gpu_attention_rc_name(rc), bad);
        if (pos >= sizeof c->mutant_note) break;
    }
}

typedef struct { const float *q, *kc, *vc; uint32_t seq, nqh, nkv; } GqaCtx;
static int run_gqa(void *p, float *got, OmegaGpuAttnInfo *info) { GqaCtx *c = p; return omega_gpu_gqa_attention_f32(c->q, c->kc, c->vc, c->seq, c->nqh, c->nkv, g_hd, got, info); }
typedef struct { const float *q, *kc, *vc; uint32_t base, nr, nqh, nkv; } RowsCtx;
static int run_rows(void *p, float *got, OmegaGpuAttnInfo *info) { RowsCtx *c = p; return omega_gpu_gqa_attention_rows_f32(c->q, c->kc, c->vc, c->base, c->nr, c->nqh, c->nkv, g_hd, got, info); }
typedef struct { const float *q; const uint8_t *pool; const OmegaGpuKvLayout *ly; const uint32_t *ids; uint32_t nids, ctx, layer, nqh, nkv; } PagedCtx;
static int run_paged(void *p, float *got, OmegaGpuAttnInfo *info) { PagedCtx *c = p; return omega_gpu_paged_attention_bf16(c->q, c->pool, c->ly, c->ids, c->nids, c->ctx, c->layer, c->nqh, c->nkv, g_hd, got, info); }
typedef struct { const float *q; const uint8_t *pool; const OmegaGpuKvLayout *ly; const int32_t *tables, *ctxs; uint32_t maxb, nseq, layer, nqh, nkv; } BatchCtx;
static int run_batch(void *p, float *got, OmegaGpuAttnInfo *info) { BatchCtx *c = p; return omega_gpu_paged_attention_batch_bf16(c->q, c->pool, c->ly, c->tables, c->ctxs, c->maxb, c->nseq, c->layer, c->nqh, c->nkv, g_hd, got, info); }
/* single vs batched parity and repeat (L6-KV, CAND-0 coverage): each sequence of a batch, run
 * alone through the single-sequence entry point with the batch's own rules applied by hand
 * (negative table entries removed, ctx <= 0 -> 0), must equal its row of the batched output bit
 * for bit (same kernel, same per-CTA inputs; the batch only shares staging). Then the batch is
 * called again with the same inputs: kernel cache hit, bit-identical output, identical staging
 * counters. `got` must hold the batched output of `bc`. */
static void batch_single_repeat_checks(const char *name, const BatchCtx *bc, const float *got, const Case *c) {
    size_t row = (size_t)bc->nqh * g_hd, n = (size_t)bc->nseq * row;
    float *one = malloc(row * 4), *again = malloc(n * 4);
    uint32_t *ids = malloc((bc->maxb ? bc->maxb : 1) * 4);
    for (uint32_t s = 0; s < bc->nseq; s++) {
        uint32_t nb = 0;
        for (uint32_t b = 0; b < bc->maxb; b++) { int32_t v = bc->tables[(size_t)s * bc->maxb + b]; if (v >= 0) ids[nb++] = (uint32_t)v; }
        uint32_t ctx = bc->ctxs[s] > 0 ? (uint32_t)bc->ctxs[s] : 0;
        for (size_t i = 0; i < row; i++) one[i] = -7.0f; /* poison: an unwritten row cannot match */
        OmegaGpuAttnInfo si;
        int src = omega_gpu_paged_attention_bf16(bc->q + (size_t)s * row, bc->pool, bc->ly, ids, nb, ctx, bc->layer, bc->nqh, bc->nkv, g_hd, one, &si);
        CHECK(src == OMEGA_GPU_ATTN_OK && memcmp(one, got + (size_t)s * row, row * 4) == 0,
              "%s: sequence %u alone (single entry point, %u blocks, ctx %u) is bit-identical to its batched row (rc=%s)", name, s, nb, ctx, omega_gpu_attention_rc_name(src));
    }
    OmegaGpuAttnInfo ri;
    int rrc = omega_gpu_paged_attention_batch_bf16(bc->q, bc->pool, bc->ly, bc->tables, bc->ctxs, bc->maxb, bc->nseq, bc->layer, bc->nqh, bc->nkv, g_hd, again, &ri);
    CHECK(rrc == OMEGA_GPU_ATTN_OK && ri.kernel_cache_hit && memcmp(again, got, n * 4) == 0 && ri.kv_bytes_staged == c->kv_staged && ri.kv_bytes_naive == c->kv_naive
              && ri.kv_blocks_unique == c->blk_unique && ri.kv_blocks_logical == c->blk_logical,
          "%s: repeated batch call is a cache hit, bit-identical, same staging counters (rc=%s hit=%d staged %" PRIu64 "/%" PRIu64 ")",
          name, omega_gpu_attention_rc_name(rrc), (int)ri.kernel_cache_hit, ri.kv_bytes_staged, c->kv_staged);
    free(one); free(again); free(ids);
}


static int chip(const char *out_path) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_ATTENTION_V1\",\"tolerances\":{\"rel\":2e-4,\"abs\":2e-5,\"large_scores_rel\":1e-3,\"large_scores_abs\":1e-4},\"cases\":[");
    uint32_t seed = 0xfb1c0005u;
    const uint32_t hd = g_hd;
    typedef struct { uint32_t nqh, nkv; const char *model; } Shape;
    const Shape shapes64[2] = { { 32, 4, "tinyllama" }, { 32, 8, "llama32_1b" } };
    /* head_dim 128: Qwen3-4B (32 q heads, 8 kv heads, GQA 4:1) and a ratio-8 shape */
    const Shape shapes128[2] = { { 32, 8, "qwen3_4b" }, { 32, 4, "gqa8" } };
    const Shape *shapes = hd == 128 ? shapes128 : shapes64;
    /* 65, 129, 300 and 1000 end in a partial 64-token chunk after whole ones: the case that
     * caught the diverged-warp shuffle on the chip (2026-10-04); 1, 17, 128, 256, 2048 did not. */
    /* head_dim 128 chunks are 128 tokens: 127, 128, 129 straddle a chunk, 300 and 1000 end in a partial one */
    const uint32_t ctxs64[9] = { 1, 17, 256, 2048, 65, 128, 129, 300, 1000 }, ctxs128[9] = { 1, 17, 256, 2048, 127, 128, 129, 300, 1000 };
    const uint32_t *ctxs = hd == 128 ? ctxs128 : ctxs64;
    const int no_mutants[1] = { 0 };

    /* 1. contiguous f32 KV (gqa_attention) */
    for (int sh = 0; sh < 2; sh++) for (int ci = 0; ci < 9; ci++) {
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
        c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; case_info(&c, &info);
        if (ci == 3) {
            OmegaGpuAttnInfo i2; int rc2 = run_gqa(&gc, again, &i2);
            CHECK(rc2 == OMEGA_GPU_ATTN_OK && i2.kernel_cache_hit && memcmp(got, again, n * 4) == 0, "%s repeat run (cache hit) is bit-identical", name);
            /* negative control: the comparison must reject a wrong expectation (kv heads swapped) */
            float *wrong = malloc(n * 4); double w2;
            for (uint32_t h = 0; h < nqh; h++) memcpy(wrong + (size_t)h * hd, want + (size_t)((h + nqh / nkv) % nqh) * hd, hd * 4);
            size_t neg = compare(got, wrong, n, 2e-4, 2e-5, &w2);
            CHECK(neg > n / 2, "%s negative control: wrong expectation flagged (%zu of %zu)", name, neg, n);
            free(wrong);
            static const int ms[5] = { OMEGA_GPU_ATTN_MUTANT_SLOT, OMEGA_GPU_ATTN_MUTANT_NO_RESCALE, OMEGA_GPU_ATTN_MUTANT_Q_ROW, OMEGA_GPU_ATTN_MUTANT_OUT_ROW, 0 };
            run_mutants(&c, ms, run_gqa, &gc, got, want, n, 2e-4, 2e-5);
        } else if (ci == 1 && sh == 1) {
            static const int ms[2] = { OMEGA_GPU_ATTN_MUTANT_KV_HEAD, 0 };
            run_mutants(&c, ms, run_gqa, &gc, got, want, n, 2e-4, 2e-5);
        } else run_mutants(&c, no_mutants, run_gqa, &gc, got, want, n, 2e-4, 2e-5);
        record(c, out); free(q); free(kc); free(vc); free(got); free(want); free(again);
    }
    /* 1b. rows path (prompt read, 2026-10-12): base_len tokens already in the KV, then num_rows
     * query rows in one call; row i attends over base_len + i + 1 tokens. Expected per row from
     * oracle_gqa on that prefix. 40 rows at 32 q heads exceed the 1024-CTA launch bound: 2 launches. */
    for (int sh = 0; sh < 2; sh++) for (int ri = 0; ri < 3; ri++) {
        static const uint32_t bases[3] = { 0, 17, 300 }, nrows[3] = { 1, 5, 40 };
        uint32_t nqh = shapes[sh].nqh, nkv = shapes[sh].nkv, base = bases[ri], nr = nrows[ri], total = base + nr;
        size_t row = (size_t)nqh * hd, n = row * nr, nkvs = (size_t)total * nkv * hd;
        float *q = malloc(n * 4), *kc = malloc(nkvs * 4), *vc = malloc(nkvs * 4), *got = malloc(n * 4), *want = malloc(n * 4);
        for (size_t i = 0; i < n; i++) q[i] = frand(&seed, -1.0f, 1.0f);
        for (size_t i = 0; i < nkvs; i++) { kc[i] = frand(&seed, -1.0f, 1.0f); vc[i] = frand(&seed, -1.0f, 1.0f); }
        for (uint32_t i = 0; i < nr; i++) oracle_gqa(q + i * row, kc, vc, base + i + 1, nqh, nkv, hd, want + i * row);
        char name[96]; snprintf(name, sizeof name, "gqa_rows_f32_%s_%uq%ukv_base%u_rows%u", shapes[sh].model, nqh, nkv, base, nr);
        Case c = { .name = strdup(name), .n = n };
        OmegaGpuAttnInfo info;
        RowsCtx rx = { q, kc, vc, base, nr, nqh, nkv };
        c.rc = run_rows(&rx, got, &info);
        c.bad = compare(got, want, n, 2e-4, 2e-5, &c.worst);
        c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; case_info(&c, &info);
        if (nr == 1) {
            /* one row is the single-row call, bit for bit */
            float *one = malloc(row * 4); OmegaGpuAttnInfo i1;
            int rc1 = omega_gpu_gqa_attention_f32(q, kc, vc, total, nqh, nkv, hd, one, &i1);
            CHECK(rc1 == OMEGA_GPU_ATTN_OK && memcmp(got, one, row * 4) == 0, "%s equals omega_gpu_gqa_attention_f32 bit for bit", name);
            free(one);
        }
        if (nr == 40) {
            CHECK(info.chip_calls == 2, "%s 40 rows x %u heads go in 2 launches (got %u)", name, nqh, info.chip_calls);
            CHECK(info.kv_bytes_staged == 2 * nkvs * 4, "%s KV staged once (%" PRIu64 " bytes, expected %zu)", name, info.kv_bytes_staged, 2 * nkvs * 4);
            /* negative control: every row scored over the whole context (no causal cut) must be flagged */
            float *wrong = malloc(n * 4); double w2;
            for (uint32_t i = 0; i < nr; i++) oracle_gqa(q + i * row, kc, vc, total, nqh, nkv, hd, wrong + i * row);
            size_t neg = compare(got, wrong, n, 2e-4, 2e-5, &w2);
            CHECK(neg > n / 4, "%s negative control: non-causal expectation flagged (%zu of %zu)", name, neg, n);
            free(wrong);
            static const int ms[5] = { OMEGA_GPU_ATTN_MUTANT_SLOT, OMEGA_GPU_ATTN_MUTANT_NO_RESCALE, OMEGA_GPU_ATTN_MUTANT_Q_ROW, OMEGA_GPU_ATTN_MUTANT_OUT_ROW, 0 };
            run_mutants(&c, ms, run_rows, &rx, got, want, n, 2e-4, 2e-5);
        } else run_mutants(&c, no_mutants, run_rows, &rx, got, want, n, 2e-4, 2e-5);
        record(c, out); free(q); free(kc); free(vc); free(got); free(want);
    }
    {   /* rows path refusals */
        float d[1]; OmegaGpuAttnInfo i0;
        CHECK(omega_gpu_gqa_attention_rows_f32(d, d, d, 0, 0, 32, 4, hd, d, &i0) == OMEGA_GPU_ATTN_BAD_ARGS, "rows: zero rows refused");
        CHECK(omega_gpu_gqa_attention_rows_f32(d, d, d, OMEGA_GPU_ATTN_MAX_GQA_CTX, 1, 32, 4, hd, d, &i0) == OMEGA_GPU_ATTN_TOO_LARGE, "rows: base + rows past the context bound refused");
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
        c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; case_info(&c, &info);
        static const int ms[2] = { OMEGA_GPU_ATTN_MUTANT_NO_MAX, 0 };
        run_mutants(&c, ms, run_gqa, &gc, got, want, n, 1e-3, 1e-4);
        record(c, out); free(q); free(kc); free(vc); free(got); free(want);
    }
    /* 2b. MHA (16q/16kv, ratio 1, ctx 300 = 4 full chunks + a partial one): the head-wiring
     * mutants Q_ROW and OUT_ROW are observably different here (Q_ROW keeps kv head h, OUT_ROW
     * lands on kv head h^1), unlike GQA where both swap the same pair of outputs. */
    {
        uint32_t nqh = 16, nkv = 16, seq = 300; size_t n = (size_t)nqh * hd, nkvs = (size_t)seq * nkv * hd;
        float *q = malloc(n * 4), *kc = malloc(nkvs * 4), *vc = malloc(nkvs * 4), *got = malloc(n * 4), *want = malloc(n * 4), *got_q = malloc(n * 4);
        for (size_t i = 0; i < n; i++) q[i] = frand(&seed, -1.0f, 1.0f);
        for (size_t i = 0; i < nkvs; i++) { kc[i] = frand(&seed, -1.0f, 1.0f); vc[i] = frand(&seed, -1.0f, 1.0f); }
        oracle_gqa(q, kc, vc, seq, nqh, nkv, hd, want);
        Case c = { .name = "gqa_f32_mha_16q16kv_ctx300", .n = n };
        OmegaGpuAttnInfo info; GqaCtx gc = { q, kc, vc, seq, nqh, nkv };
        c.rc = run_gqa(&gc, got, &info);
        c.bad = compare(got, want, n, 2e-4, 2e-5, &c.worst);
        c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; case_info(&c, &info);
        static const int ms_q[2] = { OMEGA_GPU_ATTN_MUTANT_Q_ROW, 0 };
        static const int ms_o[3] = { OMEGA_GPU_ATTN_MUTANT_OUT_ROW, OMEGA_GPU_ATTN_MUTANT_KV_HEAD, 0 };
        run_mutants(&c, ms_q, run_gqa, &gc, got, want, n, 2e-4, 2e-5);
        memcpy(got_q, got, n * 4); /* Q_ROW output */
        run_mutants(&c, ms_o, run_gqa, &gc, got, want, n, 2e-4, 2e-5);
        /* `got` now holds the KV_HEAD output; rerun OUT_ROW alone to compare it with Q_ROW */
        omega_gpu_attention_test_set_mutant(OMEGA_GPU_ATTN_MUTANT_OUT_ROW);
        OmegaGpuAttnInfo i3; int rc3 = run_gqa(&gc, got, &i3);
        omega_gpu_attention_test_set_mutant(OMEGA_GPU_ATTN_MUTANT_NONE);
        double wq; size_t differ = compare(got, got_q, n, 2e-4, 2e-5, &wq);
        CHECK(rc3 == OMEGA_GPU_ATTN_OK && differ > n / 2, "mha: Q_ROW and OUT_ROW outputs are distinct wrong answers (%zu of %zu differ)", differ, n);
        record(c, out); free(q); free(kc); free(vc); free(got); free(want); free(got_q);
    }
    /* resident fit rule at the 32-bit edge (Qwen3-8B geometry: 36 layers, 8 kv heads x 128, block 16), no allocation */
    {
        OmegaGpuKvLayout big = layout_for(4096, 16, 36, 8, 128);
        uint32_t id_ok = (uint32_t)((0xffffffffull - big.layer_stride_bytes) / big.block_stride_bytes);
        CHECK(omega_gpu_attention_test_resident_fits(&big, id_ok) && !omega_gpu_attention_test_resident_fits(&big, id_ok + 1),
              "resident fit rule at the 4 GiB edge: block %u fits, %u does not (block stride %llu)", id_ok, id_ok + 1,
              (unsigned long long)big.block_stride_bytes);
    }
    /* 3. paged bf16 KV, block size 16, 2 layers (layer 1 under test), scattered block ids */
    for (int sh = 0; sh < 2; sh++) for (int ci = 0; ci < 9; ci++) {
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
        /* Resident pool (2026-10-12): the same bytes in omega's own allocation, read in place. Same kernel,
         * same data, only the addressing differs: bit-identical to the staged result just computed (which is
         * the negative control: a plain host pool reports STAGED), nothing staged, info says RESIDENT. */
        if (c.rc == OMEGA_GPU_ATTN_OK) {
            void *rp = NULL;
            int arc = omega_gpu_attention_pool_alloc(ly.pool_bytes, &rp);
            CHECK(arc == OMEGA_GPU_ATTN_OK && rp, "%s: resident pool alloc rc=%d (%s)", name, arc, omega_gpu_attention_last_error());
            if (arc == OMEGA_GPU_ATTN_OK) {
                memcpy(rp, pool, ly.pool_bytes);
                float *got_r = malloc(n * 4); OmegaGpuAttnInfo ri;
                PagedCtx rpc = { q, rp, &ly, ids, nblk, ctx, layer, nqh, nkv };
                int rrc = run_paged(&rpc, got_r, &ri);
                int same = memcmp(got_r, got, n * 4) == 0;
                CHECK(rrc == OMEGA_GPU_ATTN_OK && ri.kv_source == OMEGA_GPU_ATTN_KV_RESIDENT && ri.kv_bytes_staged == 0 &&
                      info.kv_source == OMEGA_GPU_ATTN_KV_STAGED && same,
                      "%s: resident pool rc=%d source=%u staged=%llu (control source=%u) bit-identical=%d", name, rrc, ri.kv_source,
                      (unsigned long long)ri.kv_bytes_staged, info.kv_source, same);
                free(got_r);
            }
            omega_gpu_attention_pool_free();
        }
        c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; case_info(&c, &info);
        if (ci == 3) {
            /* NO_MAX cannot overflow on unit-scale scores (softmax is shift invariant); it is caught by the large-score gqa case */
            static const int ms3[6] = { OMEGA_GPU_ATTN_MUTANT_KV_HEAD, OMEGA_GPU_ATTN_MUTANT_SLOT, OMEGA_GPU_ATTN_MUTANT_NO_RESCALE, OMEGA_GPU_ATTN_MUTANT_Q_ROW, OMEGA_GPU_ATTN_MUTANT_OUT_ROW, 0 };
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
    /* 3b. context split (cut F): the 32q/8kv shape at ctx 1000 under every forced split against the oracle, the
     * CTA count per launch grows with the split, the automatic choice at this context is 8 (hd 128: 8 chunks) or
     * 16 (hd 64: 16 chunks; cut J, one chunk per split), and a short context (ctx 100, one chunk) stays unsplit. A forced launch split at or above
     * the sequence's own count gives the same bits (the extra CTAs write empty partials that add exact zeros). */
    {
        uint32_t nqh = 32, nkv = 8, bs = 16, layers = 2, layer = 1;
        const uint32_t cx[2] = { 1000, 100 };
        for (int ci = 0; ci < 2; ci++) {
            uint32_t ctx = cx[ci], nblk = (ctx + bs - 1) / bs, pool_blocks = nblk + 3;
            OmegaGpuKvLayout ly = layout_for(pool_blocks, bs, layers, nkv, hd);
            uint8_t *pool = malloc(ly.pool_bytes); uint16_t *pw = (uint16_t *)pool;
            for (size_t i = 0; i < ly.pool_bytes / 2; i++) pw[i] = f2bf(frand(&seed, -1.0f, 1.0f));
            uint32_t *ids = malloc(nblk * 4);
            for (uint32_t b = 0; b < nblk; b++) ids[b] = (b * 5 + 1) % pool_blocks;
            size_t n = (size_t)nqh * hd;
            float *q = malloc(n * 4), *got = malloc(n * 4), *want = malloc(n * 4), *auto_got = malloc(n * 4);
            for (size_t i = 0; i < n; i++) q[i] = frand(&seed, -1.0f, 1.0f);
            oracle_paged(q, pool, &ly, ids, nblk, ctx, layer, nqh, nkv, hd, want);
            PagedCtx pc = { q, pool, &ly, ids, nblk, ctx, layer, nqh, nkv };
            OmegaGpuAttnInfo ai;
            int arc = run_paged(&pc, auto_got, &ai);
            uint32_t chunks = (ctx + hd - 1) / hd, want_log2 = 0;
            const uint32_t gcta = nqh / omega_gpu_attention_heads_per_cta(hd, 2); /* CTAs per split: cut K groups 4 heads per CTA at hd 128 */
            while (want_log2 < OMEGA_GPU_ATTN_MAX_LOG2SPLIT && (2u << want_log2) <= chunks) want_log2++;
            CHECK(arc == OMEGA_GPU_ATTN_OK && ai.ctas_last_launch == (gcta << want_log2),
                  "split auto ctx %u: rc=%s ctas=%u (expected %u)", ctx, omega_gpu_attention_rc_name(arc), ai.ctas_last_launch, gcta << want_log2);
            for (int ls = 0; ls <= (int)OMEGA_GPU_ATTN_MAX_LOG2SPLIT; ls++) {
                set_split(ls);
                OmegaGpuAttnInfo fi;
                char name[96]; snprintf(name, sizeof name, "paged_bf16_split%d_32q8kv_ctx%u", 1 << ls, ctx);
                Case c = { .name = strdup(name), .n = n };
                c.rc = run_paged(&pc, got, &fi);
                c.bad = compare(got, want, n, 2e-4, 2e-5, &c.worst);
                c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; case_info(&c, &fi);
                CHECK(fi.ctas_last_launch == (gcta << ls) && fi.out_bytes == (uint64_t)nqh * (1u << ls) * (hd + 2) * 4,
                      "%s: ctas=%u out_bytes=%llu", name, fi.ctas_last_launch, (unsigned long long)fi.out_bytes);
                if ((uint32_t)ls >= want_log2) CHECK(memcmp(got, auto_got, n * 4) == 0, "%s equals the automatic choice bit for bit (a larger launch split only adds empty partials)", name);
                if (ls == 1 && ci == 0) {
                    static const int ms[3] = { OMEGA_GPU_ATTN_MUTANT_KV_HEAD, OMEGA_GPU_ATTN_MUTANT_SLOT, 0 };
                    run_mutants(&c, ms, run_paged, &pc, got, want, n, 2e-4, 2e-5);
                } else run_mutants(&c, no_mutants, run_paged, &pc, got, want, n, 2e-4, 2e-5);
                record(c, out);
            }
            set_split(-1);
            if (ci == 1 && chunks == 1) CHECK(ai.ctas_last_launch == gcta, "ctx %u: one chunk stays unsplit (%u CTAs)", ctx, ai.ctas_last_launch);
            else if (ci == 1) CHECK(ai.ctas_last_launch == (gcta << want_log2), "ctx %u: %u chunks -> %u splits (%u CTAs)", ctx, chunks, 1u << want_log2, ai.ctas_last_launch);
            free(pool); free(ids); free(q); free(got); free(want); free(auto_got);
        }
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
        c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; case_info(&c, &info);
        int zeros = 1; for (size_t i = 0; i < (size_t)nqh * hd; i++) if (got[(size_t)nqh * hd + i] != 0.0f) zeros = 0;
        CHECK(zeros, "batch: the empty sequence is all zeros");
        batch_single_repeat_checks(c.name, &bc, got, &c);
        static const int ms[5] = { OMEGA_GPU_ATTN_MUTANT_KV_HEAD, OMEGA_GPU_ATTN_MUTANT_SLOT, OMEGA_GPU_ATTN_MUTANT_Q_ROW, OMEGA_GPU_ATTN_MUTANT_OUT_ROW, 0 };
        run_mutants(&c, ms, run_batch, &bc, got, want, n, 2e-4, 2e-5);
        /* all 7 referenced blocks are distinct here, so dedupe stages exactly the naive figure */
        CHECK(c.blk_logical == 7 && c.blk_unique == 7 && c.kv_staged == c.kv_naive && c.kv_staged == 7ull * ly.layer_stride_bytes, "batch: 7 distinct blocks staged once each (%u/%u, %" PRIu64 " bytes)", c.blk_unique, c.blk_logical, c.kv_staged);
        record(c, out); free(pool); free(tables); free(q); free(got); free(want);
    }
    /* 5. branch-heavy batch (attention hardening cut): four sequences share the same 10 physical
     * prefix blocks (160 tokens, scattered ids) and own private tails of 1..3 blocks ending in a
     * partial 64-token chunk. The shared prefix must be staged once per launch, keyed by the
     * physical block id. Two head shapes: 8q/2kv puts all four sequences in one launch
     * (unique = 10 + 9 tails), 32q/8kv splits them two per launch (unique = 2*10 + 9).
     * Negative control: the legacy per-sequence staging (test hook) copies the prefix four
     * times and the byte-reduction check must flag it. */
    for (int sh = 0; sh < 2; sh++) {
        uint32_t nqh = sh ? 32 : 8, nkv = sh ? 8 : 2, bs = 16, layers = 2, layer = 1, nseq = 4, prefix_blocks = 10, pool_blocks = 64;
        uint32_t expect_unique = sh ? 2 * prefix_blocks + 9 : prefix_blocks + 9;
        uint32_t maxb = prefix_blocks + 3;
        OmegaGpuKvLayout ly = layout_for(pool_blocks, bs, layers, nkv, hd);
        uint8_t *pool = malloc(ly.pool_bytes); uint16_t *pw = (uint16_t *)pool;
        for (size_t i = 0; i < ly.pool_bytes / 2; i++) pw[i] = f2bf(frand(&seed, -1.0f, 1.0f));
        int32_t *tables = malloc((size_t)nseq * maxb * 4);
        for (size_t i = 0; i < (size_t)nseq * maxb; i++) tables[i] = -1;
        const uint32_t tails[4] = { 1, 2, 3, 3 };                 /* private blocks per sequence: 9 in all */
        const int32_t ctxs5[4] = { 165, 180, 193, 207 };          /* 160 + 5, 20, 33, 47: every last chunk partial */
        for (uint32_t s = 0; s < nseq; s++) {
            for (uint32_t b = 0; b < prefix_blocks; b++) tables[s * maxb + b] = (int32_t)((b * 5 + 7) % pool_blocks); /* same scattered physical prefix */
            for (uint32_t b = 0; b < tails[s]; b++) tables[s * maxb + prefix_blocks + b] = (int32_t)(50 + s * 3 + b);   /* private, distinct */
        }
        size_t n = (size_t)nseq * nqh * hd;
        float *q = malloc(n * 4), *got = malloc(n * 4), *want = malloc(n * 4), *legacy = malloc(n * 4);
        for (size_t i = 0; i < n; i++) q[i] = frand(&seed, -1.0f, 1.0f);
        oracle_batch(q, pool, &ly, tables, ctxs5, maxb, nseq, layer, nqh, nkv, hd, want);
        char name[96]; snprintf(name, sizeof name, "paged_bf16_shared_prefix_4seq_%uq%ukv_10pre", nqh, nkv);
        Case c = { .name = strdup(name), .n = n };
        OmegaGpuAttnInfo info; BatchCtx bc = { q, pool, &ly, tables, ctxs5, maxb, nseq, layer, nqh, nkv };
        c.rc = run_batch(&bc, got, &info);
        c.bad = compare(got, want, n, 2e-4, 2e-5, &c.worst);
        c.pass = c.rc == OMEGA_GPU_ATTN_OK && c.bad == 0; case_info(&c, &info);
        uint32_t logical = 4 * prefix_blocks + 9;
        CHECK(c.blk_logical == logical && c.blk_unique == expect_unique, "%s: %u logical references, %u unique staged (expected %u/%u)", name, c.blk_logical, c.blk_unique, logical, expect_unique);
        CHECK(c.kv_naive == (uint64_t)logical * ly.layer_stride_bytes && c.kv_staged == (uint64_t)expect_unique * ly.layer_stride_bytes && c.kv_staged < c.kv_naive,
              "%s: staged %" PRIu64 " < naive %" PRIu64 " bytes (saved %" PRIu64 ")", name, c.kv_staged, c.kv_naive, c.kv_naive - c.kv_staged);
        /* cut F: the output traffic is the partials, launch split = the largest own split in the call (cut J: hd 64: ctx 193 and 207 are 4 chunks, 4 splits; hd 128: 2 chunks, 2 splits) */
        uint32_t ls_max = 0;
        for (uint32_t s = 0; s < nseq; s++) { uint32_t x = ((uint32_t)ctxs5[s] + hd - 1) / hd, l = 0; while (l < OMEGA_GPU_ATTN_MAX_LOG2SPLIT && (2u << l) <= x) l++; if (l > ls_max) ls_max = l; }
        const uint64_t out_expect = (uint64_t)nseq * nqh * (1u << ls_max) * (hd + 2) * 4;
        CHECK(info.kv_source == OMEGA_GPU_ATTN_KV_STAGED && info.q_bytes == n * 4 && info.out_bytes == out_expect && info.tab_bytes > 0, "%s: q/out/table traffic accounted (q=%" PRIu64 " out=%" PRIu64 " tab=%" PRIu64 ")", name, info.q_bytes, info.out_bytes, info.tab_bytes);
        /* negative control: legacy per-sequence staging duplicates the shared prefix; same answer, more bytes */
        omega_gpu_attention_test_set_dedupe(false);
        OmegaGpuAttnInfo li; int lrc = run_batch(&bc, legacy, &li);
        omega_gpu_attention_test_set_dedupe(true);
        double lw; size_t lbad = compare(legacy, want, n, 2e-4, 2e-5, &lw);
        CHECK(lrc == OMEGA_GPU_ATTN_OK && lbad == 0, "%s legacy staging: parity (rc=%s bad=%zu)", name, omega_gpu_attention_rc_name(lrc), lbad);
        CHECK(li.kv_blocks_unique == logical && li.kv_bytes_staged == li.kv_bytes_naive && li.kv_bytes_staged > c.kv_staged,
              "%s negative control: duplicated prefix staging is flagged (legacy staged %" PRIu64 " == naive %" PRIu64 ", dedupe %" PRIu64 ")", name, li.kv_bytes_staged, li.kv_bytes_naive, c.kv_staged);
        CHECK(memcmp(got, legacy, n * 4) == 0, "%s: dedupe and legacy staging are bit-identical", name);
        batch_single_repeat_checks(name, &bc, got, &c);
        static const int ms5[3] = { OMEGA_GPU_ATTN_MUTANT_KV_HEAD, OMEGA_GPU_ATTN_MUTANT_SLOT, 0 };
        run_mutants(&c, ms5, run_batch, &bc, got, want, n, 2e-4, 2e-5);
        record(c, out); free(pool); free(tables); free(q); free(got); free(want); free(legacy);
    }
    /* every mutant the API knows (1 .. COUNT-1) ran at least once and was caught every time;
     * a mutant added to the enum without a battery entry fails here */
    for (int m = 1; m < OMEGA_GPU_ATTN_MUTANT_COUNT; m++) {
        CHECK(g_mutant_run[m] > 0 && g_mutant_caught[m] == g_mutant_run[m], "mutant %s caught %d of %d runs", mutant_name(m), g_mutant_caught[m], g_mutant_run[m]);
        printf("mutant %s: caught %d of %d\n", mutant_name(m), g_mutant_caught[m], g_mutant_run[m]);
    }
    CHECK(omega_gpu_session_open_count() == (g_sim_mode ? 0u : 1u), "device opened %s for the battery (opens=%u)", g_sim_mode ? "never (simulator)" : "once", omega_gpu_session_open_count());
    printf("device_opens=%u\n", omega_gpu_session_open_count());
    if (out) {
        fprintf(out, "],\"mutant_count\":%d,\"mutants\":{", OMEGA_GPU_ATTN_MUTANT_COUNT - 1);
        for (int m = 1; m < OMEGA_GPU_ATTN_MUTANT_COUNT; m++) fprintf(out, "%s\"%s\":{\"run\":%d,\"caught\":%d}", m > 1 ? "," : "", mutant_name(m), g_mutant_run[m], g_mutant_caught[m]);
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
/* path: hash of every branch decision so far. The attention kernels use no reconvergence
 * region (the IR has BSSY/BSYNC since omega #310, omega_bw_reconv.h; the --divergent mode
 * below covers them), so once a warp splits it may stay split on the chip, and a
 * SHFL reading an inactive lane is undefined (CUDA guide, warp shuffle functions). Every
 * SHFL therefore requires one path across the warp, or the simulator stops with error 7. */
typedef struct { uint32_t r[256]; int pc; int p0; int done; uint64_t path; } SimThr;
static uint32_t sim_cbank[256];
static uint8_t sim_shared[8192];
#define SIM_TMAX 128
#define SIM_T ((int)g_hd) /* threads per CTA = head_dim */
static uint32_t g_smem_cap; /* negative control: pretend the QMD declared this many bytes (0 = the real 16 * head_dim) */
#define SIM_SMEM (g_smem_cap ? g_smem_cap : omega_gpu_attention_test_shared_bytes()) /* the shared memory the QMD declares for this kernel (launcher: shared_bytes_for) */
static int g_sim_err;
static int g_trace, g_trace_launch; static uint32_t g_trace_cx;
static uint32_t rd(const SimThr *t, int v, const BlackwellIRProgram *p) { if (v < 0) return 0; int ph = p->regalloc.vreg_to_phys[v]; return ph < 0 ? 0 : t->r[ph & 0xff]; }
static uint64_t rd64(const SimThr *t, int v, const BlackwellIRProgram *p) { if (v < 0) return 0; int ph = p->regalloc.vreg_to_phys[v] & 0xff; return (uint64_t)t->r[ph] | ((uint64_t)t->r[ph + 1] << 32); }
static void wr(SimThr *t, int v, uint32_t x, const BlackwellIRProgram *p) { if (v < 0) return; int ph = p->regalloc.vreg_to_phys[v]; if (ph >= 0) t->r[ph & 0xff] = x; }
static void wr64(SimThr *t, int v, uint64_t x, const BlackwellIRProgram *p) { if (v < 0) return; int ph = p->regalloc.vreg_to_phys[v] & 0xff; t->r[ph] = (uint32_t)x; t->r[ph + 1] = (uint32_t)(x >> 32); }
static uint32_t rds(const SimThr *t, int v, unsigned sub, const BlackwellIRProgram *p) { if (v < 0) return 0; int ph = p->regalloc.vreg_to_phys[v]; return ph < 0 ? 0 : t->r[(ph + sub) & 0xff]; }
static void wrs(SimThr *t, int v, unsigned sub, uint32_t x, const BlackwellIRProgram *p) { if (v < 0) return; int ph = p->regalloc.vreg_to_phys[v]; if (ph >= 0) t->r[(ph + sub) & 0xff] = x; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u_(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
/* run one thread until it reaches a SHFL or BAR (not executed) or EXIT; returns 1 if parked at a sync op */
static int sim_step(SimThr *t, const BlackwellIRProgram *p, uint32_t tid, uint32_t cx, uint32_t cy) {
    for (int guard = 0; guard < 50000000; guard++) {
        if (t->done) return 0;
        if (t->pc < 0 || (size_t)t->pc >= p->count) { g_sim_err = 1; t->done = 1; return 0; }
        const BlackwellIRInsn *in = &p->insns[t->pc];
        uint32_t a = rds(t, in->src1_vreg, in->src1_subreg, p), b = rds(t, in->src2_vreg, in->src2_subreg, p), c = rds(t, in->src3_vreg, in->src3_subreg, p);
        switch (in->op) {
        case BW_IR_SHFL_DOWN: case BW_IR_BAR_SYNC: return 1;
        case BW_IR_EXIT: if (!in->predicate_p0 || t->p0) { t->done = 1; return 0; } break;
        case BW_IR_BRA: {
            int take = !in->predicate_p0 || (in->predicate_not ? !t->p0 : t->p0);
            t->path = t->path * 1000003u + (((uint64_t)t->pc << 1) | (uint64_t)take);
            if (take) { if ((int32_t)in->imm == 0) { g_sim_err = 2; t->done = 1; return 0; } t->pc += (int32_t)in->imm; continue; }
            break; }
        case BW_IR_MOV_IMM: wrs(t, in->dst_vreg, in->dst_subreg, in->imm, p); break;
        case BW_IR_MOV_RZ: wrs(t, in->dst_vreg, in->dst_subreg, 0, p); break;
        case BW_IR_S2R: wrs(t, in->dst_vreg, in->dst_subreg, in->imm == BW_SR_TID_X ? tid : in->imm == BW_SR_CTAID_X ? cx : in->imm == BW_SR_CTAID_Y ? cy : 0, p); break;
        case BW_IR_LDC: wrs(t, in->dst_vreg, in->dst_subreg, sim_cbank[in->imm / 4], p); break;
        case BW_IR_LDC64: wr64(t, in->dst_vreg, (uint64_t)sim_cbank[in->imm / 4] | ((uint64_t)sim_cbank[in->imm / 4 + 1] << 32), p); break;
        case BW_IR_LDCU64: break; /* memory descriptor: not modelled */
        case BW_IR_IMAD: wrs(t, in->dst_vreg, in->dst_subreg, in->src2_vreg >= 0 ? a * b + c : a * in->imm + c, p); break;
        case BW_IR_IMAD_WIDE: wr64(t, in->dst_vreg, (uint64_t)a * in->imm + rd64(t, in->src3_vreg, p), p); break;
        case BW_IR_IADD3: wrs(t, in->dst_vreg, in->dst_subreg, a + b, p); break;
        case BW_IR_SHF_R: wrs(t, in->dst_vreg, in->dst_subreg, a >> (in->imm & 31), p); break;
        case BW_IR_LOP3_AND: wrs(t, in->dst_vreg, in->dst_subreg, a & in->imm, p); break;
        case BW_IR_LOP3_XOR: wrs(t, in->dst_vreg, in->dst_subreg, a ^ b, p); break;
        case BW_IR_ISETP_GE_U32: t->p0 = a >= b; break;
        case BW_IR_LDG_E: { uint32_t x; memcpy(&x, (const void *)(uintptr_t)rd64(t, in->src1_vreg, p), 4); wrs(t, in->dst_vreg, in->dst_subreg, x, p); break; }
        case BW_IR_LDG_E_U16: { uint16_t x; memcpy(&x, (const void *)(uintptr_t)rd64(t, in->src1_vreg, p), 2); wrs(t, in->dst_vreg, in->dst_subreg, x, p); break; }
        case BW_IR_STG_E: memcpy((void *)(uintptr_t)rd64(t, in->src1_vreg, p), &b, 4); break;
        case BW_IR_LDG_E_64: case BW_IR_LDG_E_128: { unsigned n = in->op == BW_IR_LDG_E_64 ? 8u : 16u; uint64_t ad = rd64(t, in->src1_vreg, p) + (uint64_t)(int64_t)(int32_t)in->imm; if (ad & (n - 1)) { g_sim_err = 5; t->done = 1; return 0; } uint32_t x[4]; memcpy(x, (const void *)(uintptr_t)ad, n); for (unsigned i = 0; i < n / 4; i++) wrs(t, in->dst_vreg, i, x[i], p); break; }
        case BW_IR_LDS128: { uint32_t ad = a + in->imm; if ((ad & 15u) || ad + 16 > SIM_SMEM) { g_sim_err = 3; t->done = 1; return 0; } for (unsigned i = 0; i < 4; i++) { uint32_t x; memcpy(&x, sim_shared + ad + 4u * i, 4); wrs(t, in->dst_vreg, i, x, p); } break; }
        case BW_IR_LDS32: { uint32_t x; if (a + 4 > SIM_SMEM) { g_sim_err = 3; t->done = 1; return 0; } memcpy(&x, sim_shared + a, 4); wrs(t, in->dst_vreg, in->dst_subreg, x, p); break; }
        case BW_IR_STS32: if (a + 4 > SIM_SMEM) { g_sim_err = 3; t->done = 1; return 0; } memcpy(sim_shared + a, &b, 4); break;
        case BW_IR_FADD: wrs(t, in->dst_vreg, in->dst_subreg, f2u_(u2f(a) + u2f(b)), p); break;
        case BW_IR_FSUB: wrs(t, in->dst_vreg, in->dst_subreg, f2u_(u2f(a) - u2f(b)), p); break;
        case BW_IR_FMUL: wrs(t, in->dst_vreg, in->dst_subreg, f2u_(u2f(a) * u2f(b)), p); break;
        case BW_IR_FFMA: wrs(t, in->dst_vreg, in->dst_subreg, f2u_(fmaf(u2f(a), u2f(b), u2f(c))), p); break;
        case BW_IR_FMNMX_MAX: wrs(t, in->dst_vreg, in->dst_subreg, f2u_(fmaxf(u2f(a), u2f(b))), p); break;
        case BW_IR_MUFU_EX2: wrs(t, in->dst_vreg, in->dst_subreg, f2u_(exp2f(u2f(a))), p); break;
        case BW_IR_MUFU_RCP: wrs(t, in->dst_vreg, in->dst_subreg, f2u_(1.0f / u2f(a)), p); break;
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
    static SimThr th[SIM_TMAX];
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
                for (int i = 0; i < SIM_T; i++) if (!th[i].done && !th[i & ~31].done && th[i].path != th[i & ~31].path) g_sim_err = 7;
                if (g_sim_err) break;
                uint32_t src[SIM_TMAX];
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
/* --split-timing: the paged bf16 decode shape (32 q / 8 kv heads at the current head_dim, one sequence, context
 * ATTN_SPLIT_CTX, default 1268: the campaign's document tasks) through the persistent session, 100 calls per
 * forced launch split 1, 2, 4, 8 and the automatic choice; median wall and mean chip time per call, every call
 * against the oracle. A measurement for the decode-speed work (2026-10-12): whether the kernel's time at this
 * context falls with more CTAs per (sequence, head) or stays flat (bandwidth bound: the four query heads of a
 * kv head each read its slice). */
static int split_timing(const char *out_path) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    enum { N = 100 };
    const uint32_t hd = g_hd, nqh = 32, nkv = 8, bs = 16, layers = 2, layer = 1;
    const char *ce = getenv("ATTN_SPLIT_CTX");
    const uint32_t ctx = ce && atoi(ce) > 0 ? (uint32_t)atoi(ce) : 1268u;
    const uint32_t nblk = (ctx + bs - 1) / bs, pool_blocks = nblk + 3;
    uint32_t seed = 0x5b17c0deu;
    OmegaGpuKvLayout ly = layout_for(pool_blocks, bs, layers, nkv, hd);
    uint8_t *pool = malloc(ly.pool_bytes); uint16_t *pw = (uint16_t *)pool;
    for (size_t i = 0; i < ly.pool_bytes / 2; i++) pw[i] = f2bf(frand(&seed, -1.0f, 1.0f));
    uint32_t *ids = malloc(nblk * 4);
    for (uint32_t b = 0; b < nblk; b++) ids[b] = (b * 5 + 1) % pool_blocks;
    const size_t n = (size_t)nqh * hd;
    float *q = malloc(n * 4), *got = malloc(n * 4), *want = malloc(n * 4);
    for (size_t i = 0; i < n; i++) q[i] = frand(&seed, -1.0f, 1.0f);
    oracle_paged(q, pool, &ly, ids, nblk, ctx, layer, nqh, nkv, hd, want);
    PagedCtx pc = { q, pool, &ly, ids, nblk, ctx, layer, nqh, nkv };
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_ATTENTION_SPLIT_TIMING_V1\",\"hd\":%u,\"ctx\":%u,\"calls\":%d,\"spin_us\":%u,\"cases\":[",
                     hd, ctx, N, omega_gpu_session_spin_us());
    int first = 1;
    for (int ls = -1; ls <= (int)OMEGA_GPU_ATTN_MAX_LOG2SPLIT; ls++) {
        set_split(ls);
        OmegaGpuAttnInfo info;
        int rc0 = run_paged(&pc, got, &info); /* warm-up: opens the device, generates this split's kernel */
        CHECK(rc0 == OMEGA_GPU_ATTN_OK, "split-timing split=%d first call rc=%s err=\"%s\"", ls, omega_gpu_attention_rc_name(rc0), omega_gpu_attention_last_error());
        double samples[N]; int ok_calls = 0; size_t bad_total = 0; double worst = 0; uint64_t chip_sum = 0;
        for (int i = 0; i < N; i++) {
            double t0 = now_ms();
            int rc = run_paged(&pc, got, &info);
            samples[i] = now_ms() - t0;
            double w; size_t bad = compare(got, want, n, 2e-4, 2e-5, &w);
            if (w > worst) worst = w;
            bad_total += bad; chip_sum += info.elapsed_ns;
            if (rc == OMEGA_GPU_ATTN_OK && bad == 0) ok_calls++;
        }
        qsort(samples, N, sizeof samples[0], cmp_d);
        const double median = samples[N / 2], p90 = samples[N * 9 / 10], mn = samples[0], mx = samples[N - 1];
        const int pass = ok_calls == N && bad_total == 0;
        CHECK(pass, "split-timing split=%d ctx=%u ok_calls=%d/%d violations=%zu", ls, ctx, ok_calls, N, bad_total);
        printf("%s split-timing split=%s ctas=%u ctx=%u ok_calls=%d/%d worst_scaled_err=%g median_ms=%.3f p90_ms=%.3f min_ms=%.3f max_ms=%.3f chip_us_mean=%.1f\n",
               pass ? "PASS" : "FAIL", ls < 0 ? "auto" : (const char *[]){ "1", "2", "4", "8", "16" }[ls], info.ctas_last_launch, ctx, ok_calls, N, worst,
               median, p90, mn, mx, chip_sum / 1e3 / N);
        if (out) fprintf(out, "%s{\"split\":%d,\"ctas\":%u,\"ok_calls\":%d,\"violations\":%zu,\"worst_scaled_err\":%g,\"median_ms\":%.4f,\"p90_ms\":%.4f,\"min_ms\":%.4f,\"max_ms\":%.4f,\"chip_us_mean\":%.1f,\"pass\":%s}",
                         first ? "" : ",", ls, info.ctas_last_launch, ok_calls, bad_total, worst, median, p90, mn, mx, chip_sum / 1e3 / N, pass ? "true" : "false");
        first = 0;
    }
    set_split(-1);
    free(pool); free(ids); free(q); free(got); free(want);
    if (out) { fprintf(out, "],\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", g_checks, g_failed, g_failed == 0 ? "OMEGA_GPU_ATTENTION_SPLIT_TIMING_PASS" : "OMEGA_GPU_ATTENTION_SPLIT_TIMING_FAIL"); fclose(out); }
    return 0;
}

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

/* ------------------------------------------------ head-scaling sweep (attention hardening cut)
 * `--sweep [--out sweep.json]`: the production paged bf16 path at head_dim 64 over query-head
 * counts 1, 2, 4, 8, 16, 24, 32, 64 and every divisor as the kv-head count, at context 256
 * (latency) and 2048 (bandwidth), block size 16, scattered block ids. Points whose GQA ratio
 * is not a power of two are mathematically legal (the generic contract allows them) but
 * outside the current kernel envelope (codegen selects the kv head with a shift): they are
 * called anyway, must be refused with BAD_ARGS, and are recorded as
 * "unsupported_by_kernel_envelope" (no chip time). Every supported point is checked against
 * the host oracle on its first call, then warmed and timed over N calls (median / p90 of the
 * whole call and of the chip kernel time). Gates: parity, no unwritten word, one device open,
 * every call OK; the timings are evidence, not a threshold. Only quantities the stack measures
 * are recorded; SM occupancy, bandwidth utilisation and cache hit rates are not observable
 * through this path and are reported as unavailable. Under --sim the timings are host-simulator
 * timings and the receipt says so. */
static int sweep(const char *out_path) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    const int N = g_sim_mode ? 1 : 20;
    const uint32_t hd = 64, bs = 16, layers = 1, layer = 0;
    static const uint32_t qheads[8] = { 1, 2, 4, 8, 16, 24, 32, 64 };
    static const uint32_t ctxs[2] = { 256, 2048 };
    uint32_t seed = 0xfb1c05eeu;
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_ATTENTION_SWEEP_V1\",\"path\":\"paged_bf16\",\"head_dim\":%u,\"block_size\":%u,\"timed_calls\":%d,\"host_simulator\":%s,"
                     "\"tolerances\":{\"rel\":2e-4,\"abs\":2e-5},\"hardware_counters\":\"unavailable (SM occupancy, bandwidth utilisation, cache hit rates are not observable through this path)\",\"points\":[",
                     hd, bs, N, g_sim_mode ? "true" : "false");
    int npoints = 0, supported = 0, unsupported = 0;
    double *samples = malloc((size_t)N * sizeof *samples), *ksamples = malloc((size_t)N * sizeof *ksamples);
    for (int qi = 0; qi < 8; qi++) for (uint32_t nkv = 1; nkv <= qheads[qi]; nkv++) {
        uint32_t nqh = qheads[qi];
        if (nqh % nkv) continue;
        uint32_t ratio = nqh / nkv; int pow2 = (ratio & (ratio - 1)) == 0;
        for (int ci = 0; ci < 2; ci++) {
            uint32_t ctx = ctxs[ci], nblk = (ctx + bs - 1) / bs, pool_blocks = nblk + 5;
            OmegaGpuKvLayout ly = layout_for(pool_blocks, bs, layers, nkv, hd);
            uint8_t *pool = malloc(ly.pool_bytes); uint16_t *pw = (uint16_t *)pool;
            for (size_t i = 0; i < ly.pool_bytes / 2; i++) pw[i] = f2bf(frand(&seed, -1.0f, 1.0f));
            uint32_t *ids = malloc(nblk * 4);
            for (uint32_t b = 0; b < nblk; b++) ids[b] = (b * 7 + 3) % pool_blocks;
            size_t n = (size_t)nqh * hd;
            float *q = malloc(n * 4), *got = malloc(n * 4), *want = malloc(n * 4);
            for (size_t i = 0; i < n; i++) q[i] = frand(&seed, -1.0f, 1.0f);
            OmegaGpuAttnInfo info; memset(&info, 0, sizeof info);
            int rc = omega_gpu_paged_attention_bf16(q, pool, &ly, ids, nblk, ctx, layer, nqh, nkv, hd, got, &info);
            const char *klass; int pass; size_t bad = 0; double worst = 0; int ok_calls = 0; uint32_t unwritten = 0, cache_hits = 0; double med = 0, p90 = 0, kmed = 0, kp90 = 0;
            if (!pow2) {
                klass = "unsupported_by_kernel_envelope";
                pass = rc == OMEGA_GPU_ATTN_BAD_ARGS; unsupported++;
                CHECK(pass, "sweep %uq/%ukv ctx %u: ratio %u (not a power of two) refused with BAD_ARGS (got %s)", nqh, nkv, ctx, ratio, omega_gpu_attention_rc_name(rc));
            } else {
                klass = "supported"; supported++;
                oracle_paged(q, pool, &ly, ids, nblk, ctx, layer, nqh, nkv, hd, want);
                bad = compare(got, want, n, 2e-4, 2e-5, &worst);
                unwritten = info.unwritten_words;
                uint64_t kv_staged = info.kv_bytes_staged, kv_naive = info.kv_bytes_naive, qb = info.q_bytes, tb = info.tab_bytes, ob = info.out_bytes;
                uint32_t launches = info.chip_calls, ctas = info.ctas_last_launch, blk_l = info.kv_blocks_logical, blk_u = info.kv_blocks_unique, kv_src = info.kv_source;
                for (int i = 0; i < N; i++) {
                    OmegaGpuAttnInfo ti;
                    int trc = omega_gpu_paged_attention_bf16(q, pool, &ly, ids, nblk, ctx, layer, nqh, nkv, hd, got, &ti);
                    samples[i] = ti.call_ns / 1e6; ksamples[i] = ti.elapsed_ns / 1e6;
                    double w; size_t b2 = compare(got, want, n, 2e-4, 2e-5, &w);
                    if (w > worst) worst = w;
                    bad += b2; unwritten += ti.unwritten_words; cache_hits += ti.kernel_cache_hit;
                    if (trc == OMEGA_GPU_ATTN_OK && b2 == 0) ok_calls++;
                }
                qsort(samples, (size_t)N, sizeof samples[0], cmp_d); qsort(ksamples, (size_t)N, sizeof ksamples[0], cmp_d);
                med = samples[N / 2]; p90 = samples[N * 9 / 10]; kmed = ksamples[N / 2]; kp90 = ksamples[N * 9 / 10];
                pass = rc == OMEGA_GPU_ATTN_OK && bad == 0 && ok_calls == N && unwritten == 0 && cache_hits == (uint32_t)N;
                CHECK(pass, "sweep %uq/%ukv ctx %u: rc=%s violations=%zu ok_calls=%d/%d unwritten=%u cache_hits=%u", nqh, nkv, ctx, omega_gpu_attention_rc_name(rc), bad, ok_calls, N, unwritten, cache_hits);
                printf("%s sweep %2uq/%2ukv ratio %2u ctx %4u: launches=%u ctas=%u call_ms med=%.3f p90=%.3f kernel_ms med=%.3f p90=%.3f kv_staged=%" PRIu64 " (naive %" PRIu64 ") blocks=%u/%u q=%" PRIu64 " tab=%" PRIu64 " out=%" PRIu64 " worst=%g\n",
                       pass ? "PASS" : "FAIL", nqh, nkv, ratio, ctx, launches, ctas, med, p90, kmed, kp90, kv_staged, kv_naive, blk_u, blk_l, qb, tb, ob, worst);
                if (out) fprintf(out, "%s{\"q_heads\":%u,\"kv_heads\":%u,\"gqa_ratio\":%u,\"head_dim\":%u,\"context\":%u,\"q_width\":%zu,\"out_width\":%zu,\"class\":\"%s\",\"rc\":\"%s\",\"pass\":%s,"
                                 "\"violations\":%zu,\"worst_scaled_err\":%g,\"unwritten_words\":%u,\"launches_per_call\":%u,\"ctas_last_launch\":%u,\"threads_per_cta\":%u,"
                                 "\"call_ms_median\":%.4f,\"call_ms_p90\":%.4f,\"kernel_ms_median\":%.4f,\"kernel_ms_p90\":%.4f,\"kernel_cache_hits\":%u,\"first_call_cache_hit\":%s,"
                                 "\"kv_bytes_staged\":%" PRIu64 ",\"kv_bytes_naive\":%" PRIu64 ",\"kv_blocks_logical\":%u,\"kv_blocks_unique\":%u,\"q_bytes\":%" PRIu64 ",\"tab_bytes\":%" PRIu64 ",\"out_bytes\":%" PRIu64 ",\"kv_source\":\"%s\"}",
                                 npoints ? "," : "", nqh, nkv, ratio, hd, ctx, n, n, klass, omega_gpu_attention_rc_name(rc), pass ? "true" : "false", bad, worst, unwritten, launches, ctas, info.threads_per_cta,
                                 med, p90, kmed, kp90, cache_hits, info.kernel_cache_hit ? "true" : "false", kv_staged, kv_naive, blk_l, blk_u, qb, tb, ob, kv_src == OMEGA_GPU_ATTN_KV_STAGED ? "staged" : "resident");
                npoints++;
            }
            if (!pow2) {
                printf("%s sweep %2uq/%2ukv ratio %2u ctx %4u: %s (rc=%s)\n", pass ? "PASS" : "FAIL", nqh, nkv, ratio, ctx, klass, omega_gpu_attention_rc_name(rc));
                if (out) fprintf(out, "%s{\"q_heads\":%u,\"kv_heads\":%u,\"gqa_ratio\":%u,\"head_dim\":%u,\"context\":%u,\"q_width\":%zu,\"out_width\":%zu,\"class\":\"%s\",\"rc\":\"%s\",\"pass\":%s}",
                                 npoints ? "," : "", nqh, nkv, ratio, hd, ctx, n, n, klass, omega_gpu_attention_rc_name(rc), pass ? "true" : "false");
                npoints++;
            }
            free(pool); free(ids); free(q); free(got); free(want);
        }
    }
    free(samples); free(ksamples);
    uint32_t opens = omega_gpu_session_open_count();
    CHECK(opens == (g_sim_mode ? 0u : 1u), "sweep: device opened %s (opens=%u)", g_sim_mode ? "never (simulator)" : "once", opens);
    printf("sweep: %d points (%d supported, %d unsupported by the kernel envelope), device_opens=%u\n", npoints, supported, unsupported, opens);
    if (out) { fprintf(out, "],\"points_total\":%d,\"points_supported\":%d,\"points_unsupported\":%d,\"device_opens\":%u,\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", npoints, supported, unsupported, opens, g_checks, g_failed, g_failed == 0 ? "OMEGA_GPU_ATTENTION_SWEEP_PASS" : "OMEGA_GPU_ATTENTION_SWEEP_FAIL"); fclose(out); }
    return 0;
}

/* ------------------------------------------------------------ warp simulator (omega #308)
 * The per-thread simulator above has no model of a warp (its SHFL rule is "one path across the
 * warp or error 7"). With --divergent the kernel carries a real lane-dependent branch inside a
 * BSSY/BSYNC region, and every launch runs through tests/bw_warp_sim.h instead: 32-lane
 * fragments, barrier registers, SHFL/BAR participation. Each launch runs in both fragment
 * orders and the two outputs must be bit-identical. */
static uint64_t g_ws_issues, g_ws_launches;
static int ws_attn_launch(const void *progv, const OmegaGpuAttnLaunch *L, OmegaGpuAttnInfo *info) {
    const BlackwellIRProgram *p = progv;
    uint32_t *outw = L->out; size_t nw = L->out_bytes / 4;
    uint32_t *copy = malloc(L->out_bytes);
    int err = 0; char msg[240] = "";
    for (int order = 0; order < 2 && !err && copy; order++) {
        WsSim s; memset(&s, 0, sizeof s);
        s.p = p; s.threads = SIM_T; s.smem = sim_shared; s.smem_bytes = SIM_SMEM; s.order = order;
        uint64_t qa = (uintptr_t)L->q, pa = (uintptr_t)L->pool, oa = (uintptr_t)L->out, ta = (uintptr_t)L->tab;
        s.cbank[0x380 / 4] = (uint32_t)qa; s.cbank[0x380 / 4 + 1] = (uint32_t)(qa >> 32);
        s.cbank[0x388 / 4] = (uint32_t)pa; s.cbank[0x388 / 4 + 1] = (uint32_t)(pa >> 32);
        s.cbank[0x390 / 4] = (uint32_t)oa; s.cbank[0x390 / 4 + 1] = (uint32_t)(oa >> 32);
        s.cbank[0x398 / 4] = (uint32_t)ta; s.cbank[0x398 / 4 + 1] = (uint32_t)(ta >> 32);
        memcpy(&s.cbank[0x3a0 / 4], L->params, sizeof L->params);
        for (size_t i = 0; i < nw; i++) outw[i] = 0xffbadbadu;
        err = ws_run(&s, L->grid_x, L->grid_y);
        if (err) { snprintf(msg, sizeof msg, "order %d: %s", order, s.msg); break; }
        g_ws_issues += s.warp_issues;
        if (order == 0) memcpy(copy, outw, L->out_bytes);
        else if (memcmp(copy, outw, L->out_bytes) != 0) { err = -1; snprintf(msg, sizeof msg, "the fragment order changed the result"); }
    }
    free(copy);
    g_ws_launches++;
    uint32_t unwritten = 0; for (size_t i = 0; i < nw; i++) if (outw[i] == 0xffbadbadu) unwritten++;
    if (info) { info->chip_calls++; info->unwritten_words += unwritten; info->ctas_last_launch = L->grid_x * L->grid_y; info->threads_per_cta = SIM_T; }
    if (err) { printf("warp simulator error: %s\n", msg); return OMEGA_GPU_ATTN_CHIP_FAIL; }
    return unwritten ? OMEGA_GPU_ATTN_UNWRITTEN : OMEGA_GPU_ATTN_OK;
}
/* Negative control for --divergent --sim: mode 2 builds the same kernel with the BSYNC dropped;
 * the warp simulator must reject it (the whole point of the simulator: the old one could not). */
static void divergent_negative_control(void) {
    enum { SEQ = 65, NQH = 8, NKV = 2 };
    const uint32_t HD = g_hd;
    float *q = malloc(NQH * HD * 4), *kv = malloc((size_t)SEQ * NKV * HD * 4), *out = malloc(NQH * HD * 4);
    uint32_t seed = 0x308u;
    for (uint32_t i = 0; i < NQH * HD; i++) q[i] = frand(&seed, -1.0f, 1.0f);
    for (size_t i = 0; i < (size_t)SEQ * NKV * HD; i++) kv[i] = frand(&seed, -1.0f, 1.0f);
    omega_gpu_attention_test_set_divergent(2);
    int rc = omega_gpu_gqa_attention_f32(q, kv, kv, SEQ, NQH, NKV, HD, out, NULL);
    omega_gpu_attention_test_set_divergent(1);
    CHECK(rc == OMEGA_GPU_ATTN_CHIP_FAIL, "divergent mode 2 (BSYNC dropped) rejected by the warp simulator (rc %s)", omega_gpu_attention_rc_name(rc));
    int rc1 = omega_gpu_gqa_attention_f32(q, kv, kv, SEQ, NQH, NKV, HD, out, NULL);
    CHECK(rc1 == OMEGA_GPU_ATTN_OK, "divergent mode 1 (regioned branch) runs clean on the same input (rc %s)", omega_gpu_attention_rc_name(rc1));
    free(q); free(kv); free(out);
}

/* ------------------------------------------------------------ head_dim 128 (CPU phase)
 * Checks that need the public API only, so the same code runs on the host simulators now and on
 * the chip in the coordinated window. Run with --hd 128 (--sim for the host simulators).
 * They target what head_dim 128 changed: four warps per CTA, so the chunk max and the chunk sum
 * each combine FOUR warp partials (lane 0 of warps 0..3), and 128-token chunks.
 *  (a) uniform scores (K = 0): every weight is 1, the output is the mean of V over the context;
 *      a sum that drops the partial of warp 2 or 3 gives about twice the mean;
 *  (b) one dominant token with a score above the f32 exponent range (EX2 overflows without the
 *      right running max): the token sits in each of the four warps of chunk 0 and in the later,
 *      partial chunks; a max that ignores warp 2 or 3 overflows to inf / NaN.
 * Negative controls: the comparison rejects the answer of a different token, and a shared-memory
 * declaration of 1024 bytes (the head_dim 64 size) is refused by the simulators. */
static void hd128_checks(void) {
    const uint32_t hd = 128, nqh = 8, nkv = 2, seq = 300;
    uint32_t seed = 0x128u;
    size_t n = (size_t)nqh * hd, nkvs = (size_t)seq * nkv * hd;
    float *q = malloc(n * 4), *kc = malloc(nkvs * 4), *vc = malloc(nkvs * 4), *got = malloc(n * 4), *want = malloc(n * 4);
    for (size_t i = 0; i < n; i++) q[i] = 1.0f;
    for (size_t i = 0; i < nkvs; i++) vc[i] = frand(&seed, -1.0f, 1.0f);
    OmegaGpuAttnInfo info;
    /* (a) uniform */
    memset(kc, 0, nkvs * 4);
    oracle_gqa(q, kc, vc, seq, nqh, nkv, hd, want);
    int rc = omega_gpu_gqa_attention_f32(q, kc, vc, seq, nqh, nkv, hd, got, &info);
    double w; size_t bad = compare(got, want, n, 2e-4, 2e-5, &w);
    CHECK(rc == OMEGA_GPU_ATTN_OK && bad == 0 && info.threads_per_cta == 128, "hd128 uniform scores: mean of V over 300 tokens (rc %s, %zu bad, worst %g, %u threads)", omega_gpu_attention_rc_name(rc), bad, w, info.threads_per_cta);
    float *half = malloc(n * 4); for (size_t i = 0; i < n; i++) half[i] = want[i] * 0.5f;
    size_t neg = compare(got, half, n, 2e-4, 2e-5, &w);
    CHECK(neg > n / 2, "hd128 uniform negative control: a result with the sum of two warps dropped (half the mean) is flagged (%zu of %zu)", neg, n);
    free(half);
    /* (b) one dominant token */
    static const uint32_t pos[] = { 5, 40, 70, 100, 127, 128 + 3, 256 + 43 };
    int all_ok = 1; size_t worst_bad = 0;
    for (size_t pi = 0; pi < sizeof pos / sizeof pos[0]; pi++) {
        memset(kc, 0, nkvs * 4);
        for (uint32_t kvh = 0; kvh < nkv; kvh++) for (uint32_t d = 0; d < hd; d++) kc[((size_t)pos[pi] * nkv + kvh) * hd + d] = 12.0f; /* dot 1536, scaled by log2e / sqrt(128): about 2^196 */
        oracle_gqa(q, kc, vc, seq, nqh, nkv, hd, want);
        rc = omega_gpu_gqa_attention_f32(q, kc, vc, seq, nqh, nkv, hd, got, &info);
        bad = compare(got, want, n, 2e-4, 2e-5, &w);
        if (rc != OMEGA_GPU_ATTN_OK || bad) { all_ok = 0; if (bad > worst_bad) worst_bad = bad; }
        printf("  hd128 dominant token %u (warp %u of chunk %u): rc %s, %zu bad, worst %g\n", pos[pi], (pos[pi] % 128) / 32, pos[pi] / 128, omega_gpu_attention_rc_name(rc), bad, w);
        if (pi == 0) {
            /* negative control: the answer of token 6 instead of token 5 must be flagged */
            float *other = malloc(n * 4);
            for (uint32_t h = 0; h < nqh; h++) for (uint32_t d = 0; d < hd; d++) other[h * hd + d] = vc[((size_t)6 * nkv + h / (nqh / nkv)) * hd + d];
            size_t negb = compare(got, other, n, 2e-4, 2e-5, &w);
            CHECK(negb > n / 2, "hd128 dominant-token negative control: the neighbouring token's V is flagged (%zu of %zu)", negb, n);
            free(other);
        }
    }
    CHECK(all_ok, "hd128 dominant token in every warp and chunk: exact max/sum over four warp partials (worst bad %zu)", worst_bad);
    /* a kernel that needs 2048 bytes of shared memory must not run in a 1024-byte declaration (simulators only: they bound shared accesses) */
    if (g_sim_mode) {
        g_smem_cap = 1024;
        rc = omega_gpu_gqa_attention_f32(q, kc, vc, seq, nqh, nkv, hd, got, NULL);
        g_smem_cap = 0;
        CHECK(rc == OMEGA_GPU_ATTN_CHIP_FAIL, "hd128 kernel refused under a 1024-byte shared-memory declaration (rc %s)", omega_gpu_attention_rc_name(rc));
        rc = omega_gpu_gqa_attention_f32(q, kc, vc, seq, nqh, nkv, hd, got, NULL);
        CHECK(rc == OMEGA_GPU_ATTN_OK, "hd128 kernel runs under the 2048-byte declaration (rc %s)", omega_gpu_attention_rc_name(rc));
    }
    free(q); free(kc); free(vc); free(got); free(want);
}

/* head_dim 64 byte-identity pin and head_dim 128 codegen facts (CPU only, no device, no simulator).
 * The digests are SHA-256 of the encoded kernel bytes as emitted by the batched-load program of 2026-10-12
 * (speed/attention-batched-loads; before it, main c0369e6 was pinned at 160 instructions); any change to the head_dim 64 program changes a digest and fails here. */
typedef struct { bool f32; uint32_t log2bs, log2gqa; size_t insns; const char *sha256; } Pin;
static const Pin k_pins64[] = {
    { false, 4, 3, 216, "44d27fae78ea21ee77fd7a0ebd2ebfcac52e9e45a81de84b035868092e13e424" }, 
    { false, 4, 2, 216, "3bc79d4a54fd0d82024b376dfdb718e8326431917251f13f5c44be6ba8eb6f0c" }, 
    { false, 4, 0, 216, "83eca9872e32c17321d2f8ec6a6ec92638ed72b136d8958cda5cdfc451c01ef8" }, 
    { true, 12, 3, 184, "b6a25d8739185bbd944bfcc1ec174abb24ce46529f06c56f8d5ac77bf6883ddd" }, 
    { true, 12, 2, 184, "46a2b82ebcb55237f4a7585679b42ac9558c7a434e8e80ccef6cc0bfaed8fcb9" }, 
    { true, 12, 0, 184, "970db9533c836eca7c86783875a2ad898cd69bc3070cae418cc92b3ce2dbbd9c" },
};
static void hex_digest(const uint8_t d[32], char out[65]) { for (int i = 0; i < 32; i++) snprintf(out + 2 * i, 3, "%02x", d[i]); }
static void hd_codegen_checks(void) {
    char hex[65];
    for (size_t i = 0; i < sizeof k_pins64 / sizeof k_pins64[0]; i++) {
        const Pin *p = &k_pins64[i];
        OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
        int rc = omega_gpu_attention_codegen_hd(p->f32, p->log2bs, p->log2gqa, 64, &k);
        hex_digest(k.code_digest, hex);
        CHECK(rc == OMEGA_GPU_ATTN_OK && k.insn_count == p->insns && strcmp(hex, p->sha256) == 0, "hd64 %s log2bs %u log2gqa %u: encoded kernel is byte-identical to the pinned program (%zu insns, sha256 %s)", p->f32 ? "f32 " : "bf16", p->log2bs, p->log2gqa, k.insn_count, hex);
        free(k.code);
        /* the 4-argument entry point is the same kernel */
        OmegaBlackwellKernel k4; memset(&k4, 0, sizeof k4);
        CHECK(omega_gpu_attention_codegen(p->f32, p->log2bs, p->log2gqa, &k4) == OMEGA_GPU_ATTN_OK && memcmp(k4.code_digest, k.code_digest, 32) == 0, "hd64 4-argument codegen entry is the same kernel");
        free(k4.code);
        /* head_dim 128: encodes, inside the register budget, a different (longer) program */
        OmegaBlackwellKernel k2; memset(&k2, 0, sizeof k2);
        int rc2 = omega_gpu_attention_codegen_hd(p->f32, p->log2bs, p->log2gqa, 128, &k2);
        CHECK(rc2 == OMEGA_GPU_ATTN_OK && k2.code_size > 0 && k2.gpr_count <= 96 /* AT_GPR_BUDGET; cut K: the four-head kernel peaks at 84 */ && k2.insn_count > p->insns && memcmp(k2.code_digest, k.code_digest, 32) != 0,
              "hd128 %s log2bs %u log2gqa %u: kernel encodes (%zu insns vs %zu, %u gprs)", p->f32 ? "f32 " : "bf16", p->log2bs, p->log2gqa, k2.insn_count, p->insns, k2.gpr_count);
        free(k2.code);
    }
    OmegaBlackwellKernel kx; memset(&kx, 0, sizeof kx);
    CHECK(omega_gpu_attention_codegen_hd(false, 4, 3, 96, &kx) == OMEGA_GPU_ATTN_TOO_LARGE, "codegen refuses head_dim 96");
    CHECK(omega_gpu_attention_codegen_hd(false, 4, 3, 256, &kx) == OMEGA_GPU_ATTN_TOO_LARGE, "codegen refuses head_dim 256");
    /* negative control for the pin itself: a one-bit change to a pinned digest must not match */
    OmegaBlackwellKernel kn; memset(&kn, 0, sizeof kn);
    (void)omega_gpu_attention_codegen_hd(k_pins64[0].f32, k_pins64[0].log2bs, k_pins64[0].log2gqa, 64, &kn);
    kn.code_digest[0] ^= 1u; hex_digest(kn.code_digest, hex);
    CHECK(strcmp(hex, k_pins64[0].sha256) != 0, "pin negative control: a flipped digest bit does not match the pin");
    free(kn.code);
}


int main(int argc, char **argv) {
    const char *out_path = NULL; int host = 0, sim = 0, timing_mode = 0, sweep_mode = 0, split_timing_mode = 0;
    uint32_t spin_us = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--divergent") == 0) g_divergent = 1;
        else if (strcmp(argv[i], "--timing") == 0) timing_mode = 1;
        else if (strcmp(argv[i], "--split-timing") == 0) split_timing_mode = 1;
        else if (strcmp(argv[i], "--sweep") == 0) sweep_mode = 1;
        else if (strcmp(argv[i], "--hd") == 0 && i + 1 < argc) g_hd = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--spin-us") == 0 && i + 1 < argc) spin_us = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    }
    if (g_hd != 64 && g_hd != 128) { printf("--hd must be 64 or 128\n"); return 2; }
    if (g_hd == 128 && (timing_mode || sweep_mode)) { printf("--hd 128 supports the battery, --split-timing and --host-only (timing and sweep are head_dim 64 only)\n"); return 2; }
    if (spin_us && omega_gpu_session_set_spin_us(spin_us) != 0) { printf("--spin-us %u refused\n", spin_us); return 2; }
    if (g_divergent) omega_gpu_attention_test_set_divergent(1);
    if (sim) { g_sim_mode = 1; omega_gpu_attention_test_set_simulator(g_divergent ? ws_attn_launch : sim_launch); }
    if (sim && g_divergent && !host) divergent_negative_control();
    if (host) { host_only(); hd_codegen_checks(); }
    else if (split_timing_mode) { if (split_timing(out_path) != 0) return 2; }
    else if (timing_mode) { if (timing(out_path) != 0) return 2; }
    else if (sweep_mode) { if (sweep(out_path) != 0) return 2; }
    else { if (chip(out_path) != 0) return 2; if (g_hd == 128) hd128_checks(); }
    printf("%s: %d checks, %d failed\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
