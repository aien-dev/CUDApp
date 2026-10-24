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

/* gpu_tc_probe_test.c (OWN-30B T3, 2026-10-13): characterise the chip's HMMA.16816.F32.BF16 arithmetic.
 *   --host-only   model self-checks (the model family reproduces exact sums on exactly representable cases)
 *   (default)     chip: CASES single-HMMA cases (one warp each) with operand classes chosen to separate the models
 *                 (wide exponent spread, cancellation, accumulator dominance, bf16 subnormals); prints, per model of
 *                 tests/bw_hmma_model.h, how many of the 128 * CASES outputs match the chip bit for bit; negative control:
 *                 mutant 1 (C dropped) must differ on the cases with C != 0.
 * Env TC_PROBE_SEED: a different random data set (confirmation runs). Verdict line: "TC_PROBE: model=... exact N/N" when one model matches every output, else "TC_PROBE: NO EXACT MODEL".
 * Chip runs only under flock /tmp/aien-gb10.lock. */
#include "omega_blackwell_codegen_tc.h"
#include "omega_gpu_session.h"
#include "bw_hmma_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CASES 2048u

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static uint16_t bf16_of(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); } /* truncation: the value is then exactly a bf16 */
static float f_of_bf16(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
/* a random bf16 with exponent in [lo, hi] */
static float rbf(int lo, int hi) {
    float m = 1.0f + (float)(rnd() & 127) / 128.0f;
    int e = lo + (int)(rnd() % (uint32_t)(hi - lo + 1));
    float v = ldexpf(m, e);
    if (rnd() & 1) v = -v;
    return f_of_bf16(bf16_of(v));
}

typedef struct { float A[16][16], B[16][8], C[16][8]; } Case;

static void gen_case(Case *c, unsigned idx) {
    unsigned cls = idx % 6;
    for (int r = 0; r < 16; r++) for (int k = 0; k < 16; k++) {
        float v;
        switch (cls) {
        case 0: v = rbf(-2, 2); break;                       /* narrow range */
        case 1: v = rbf(-20, 20); break;                     /* wide spread: alignment truncation shows */
        case 2: v = (k & 1) ? rbf(-1, 1) : rbf(-12, 0); break;
        case 3: v = rbf(-4, 4); break;                       /* with a big accumulator (below) */
        case 4: v = rbf(-130, -126); break;                  /* bf16 near / below the normal limit (subnormal products) */
        default: v = rbf(-8, 8); break;
        }
        if (cls == 4 && (rnd() & 3) == 0) { uint16_t h = (uint16_t)((rnd() & 0x7f) | (rnd() & 0x8000)); v = f_of_bf16(h); } /* bf16 subnormal */
        c->A[r][k] = v;
    }
    for (int k = 0; k < 16; k++) for (int n = 0; n < 8; n++) c->B[k][n] = cls == 4 ? rbf(-2, 2) : rbf(cls == 1 ? -20 : -3, cls == 1 ? 20 : 3);
    for (int r = 0; r < 16; r++) for (int n = 0; n < 8; n++) {
        float v = idx % 7 == 6 ? 0.0f : rbf(-6, 6);
        if (cls == 3) v = rbf(10, 14);
        if (cls == 2 && (rnd() & 1)) { /* cancellation: C = -(exact sum rounded) */
            double s = 0; for (int k = 0; k < 16; k++) s += (double)c->A[r][k] * c->B[k][n];
            v = (float)-s;
        }
        c->C[r][n] = v;
    }
}
static uint32_t pack2(float lo, float hi) { return (uint32_t)bf16_of(lo) | ((uint32_t)bf16_of(hi) << 16); }
static void to_frag(const Case *c, uint32_t *in) {
    uint32_t *A = in, *B = in + 128; float *C = (float *)(in + 192);
    for (int L = 0; L < 32; L++) {
        int g = L / 4, t = L % 4;
        A[L * 4 + 0] = pack2(c->A[g][2 * t], c->A[g][2 * t + 1]);
        A[L * 4 + 1] = pack2(c->A[g + 8][2 * t], c->A[g + 8][2 * t + 1]);
        A[L * 4 + 2] = pack2(c->A[g][2 * t + 8], c->A[g][2 * t + 9]);
        A[L * 4 + 3] = pack2(c->A[g + 8][2 * t + 8], c->A[g + 8][2 * t + 9]);
        B[L * 2 + 0] = pack2(c->B[2 * t][g], c->B[2 * t + 1][g]);
        B[L * 2 + 1] = pack2(c->B[2 * t + 8][g], c->B[2 * t + 9][g]);
        C[L * 4 + 0] = c->C[g][2 * t]; C[L * 4 + 1] = c->C[g][2 * t + 1];
        C[L * 4 + 2] = c->C[g + 8][2 * t]; C[L * 4 + 3] = c->C[g + 8][2 * t + 1];
    }
}
/* output word (lane L, q) -> (row, col) */
static void out_rc(int L, int q, int *r, int *n) { int g = L / 4, t = L % 4; *r = g + (q >= 2 ? 8 : 0); *n = 2 * t + (q & 1); }

static float model_out(const HmmaModel *m, const Case *c, int r, int n) {
    float a[16], b[16]; for (int k = 0; k < 16; k++) { a[k] = c->A[r][k]; b[k] = c->B[k][n]; }
    return hm_dot16(m, a, b, c->C[r][n]);
}

static int host_only(void) {
    int fail = 0;
    /* exactly representable sums must come out exact in every model */
    for (int i = 0; i < 1000; i++) {
        float a[16], b[16]; for (int k = 0; k < 16; k++) { a[k] = (float)((int)(rnd() % 17) - 8); b[k] = (float)((int)(rnd() % 9) - 4); }
        float c = (float)((int)(rnd() % 33) - 16); double ex = c; for (int k = 0; k < 16; k++) ex += (double)a[k] * b[k];
        int groups[] = { 0, 1, 2, 4, 8, 16 };
        for (int gi = 0; gi < 6; gi++) for (int rn = 0; rn < 2; rn++) { HmmaModel m = { groups[gi], 0, rn, 0, 0 }; if (hm_dot16(&m, a, b, c) != (float)ex) fail++; }
    }
    /* truncation is visible: 1 + 2^-30 (one product) in G=16, extra 0 truncates to 1, extra 6 keeps it only in double, RZ -> 1 */
    { float a[16] = { 1.0f, ldexpf(1.0f, -30) }, b[16] = { 1.0f, 1.0f }; HmmaModel m = { 16, 0, 0, 0, 0 }; if (hm_dot16(&m, a, b, 0.0f) != 1.0f) fail++; }
    /* the probe's case generator produces bf16 values and its fragment packing round-trips */
    { static Case c; gen_case(&c, 7); uint32_t in[320]; to_frag(&c, in); if ((in[0] & 0xffff) != bf16_of(c.A[0][0])) fail++; }
    printf("host-only model self-checks: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}

static int run_chip(int mutant, uint32_t *out_words, const uint32_t *in_words) {
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    if (!prog || omega_blackwell_codegen_tc_probe_prog(mutant, prog) != 0) { printf("codegen failed\n"); free(prog); return -1; }
    int rc = -1;
    omega_gpu_session_lock();
    OmegaGpuMem code = {0}, in = {0}, out = {0};
    size_t need = omega_bw_program_code_bytes(prog);
    OmegaBlackwellKernel k;
    if (!omega_gpu_session_open()) { printf("open failed: %s\n", omega_gpu_session_last_error()); goto done; }
    if (omega_gpu_session_alloc_code(need, &code) != 0 || omega_blackwell_kernel_encode_into(prog, code.cpu, need, &k) != 0 || omega_gpu_session_sync_to_gpu(&code) != 0) { printf("code upload failed\n"); goto done; }
    if (omega_gpu_session_alloc((size_t)CASES * OMEGA_TC_PROBE_IN_BYTES, &in) != 0 || omega_gpu_session_alloc((size_t)CASES * OMEGA_TC_PROBE_OUT_BYTES, &out) != 0) { printf("alloc failed\n"); goto done; }
    memcpy(in.cpu, in_words, (size_t)CASES * OMEGA_TC_PROBE_IN_BYTES);
    { float *o = out.cpu; for (size_t i = 0; i < (size_t)CASES * 128u; i++) o[i] = -999.0f; }
    if (omega_gpu_session_sync_to_gpu(&in) != 0 || omega_gpu_session_sync_to_gpu(&out) != 0) goto done;
    {
        uint32_t args[4] = { (uint32_t)in.va, (uint32_t)(in.va >> 32), (uint32_t)out.va, (uint32_t)(out.va >> 32) };
        uint32_t peak = prog->regalloc.peak_gpr_usage, decl = omega_bw_gpr_declared(peak);
        OmegaGpuLaunch L = { .code_va = code.va, .gpr_count = decl < 64 ? 64 : decl, .threads_x = 32, .threads_y = 1, .grid_x = CASES, .grid_y = 1,
                             .num_elements = (size_t)CASES * 128u, .args = args, .n_args = 4, .timeout_ms = 10000 };
        uint64_t ns = 0;
        if (omega_gpu_session_launch(&L, &ns, NULL) != 0) { printf("launch failed: %s\n", omega_gpu_session_last_error()); goto done; }
        if (omega_gpu_session_sync_to_cpu(&out) != 0) goto done;
        printf("chip launch mutant %d: peak %u declared %u, %llu ns\n", mutant, peak, L.gpr_count, (unsigned long long)ns);
    }
    memcpy(out_words, out.cpu, (size_t)CASES * OMEGA_TC_PROBE_OUT_BYTES);
    rc = 0;
done:
    if (code.cpu) omega_gpu_session_free(&code);
    if (in.cpu) omega_gpu_session_free(&in);
    if (out.cpu) omega_gpu_session_free(&out);
    omega_gpu_session_unlock();
    free(prog);
    return rc;
}

int main(int argc, char **argv) {
    if (argc > 2 && strcmp(argv[1], "--dump") == 0) { /* raw code words for nvdisasm -b SM121 (offline check only) */
        static BlackwellIRProgram p; static uint8_t buf[OMEGA_BW_CODE_CAP_MAX]; size_t len = 0;
        if (omega_blackwell_codegen_tc_probe_prog(argc > 3 ? atoi(argv[3]) : 0, &p) != 0 || omega_bw_encode_program(&p, buf, sizeof buf, &len) != 0) return 1;
        FILE *f = fopen(argv[2], "wb"); if (!f) return 1; fwrite(buf, 1, len, f); fclose(f); printf("peak %u, %zu bytes\n", p.regalloc.peak_gpr_usage, len); return 0;
    }
    int host = argc > 1 && strcmp(argv[1], "--host-only") == 0;
    if (getenv("TC_PROBE_SEED")) rng = strtoull(getenv("TC_PROBE_SEED"), NULL, 0) | 1u; /* fresh data for a confirmation run */
    if (host_only() != 0) return 1;
    if (host) return 0;
    static Case cases[CASES];
    uint32_t *in = calloc((size_t)CASES, OMEGA_TC_PROBE_IN_BYTES), *out = calloc((size_t)CASES, OMEGA_TC_PROBE_OUT_BYTES), *outm = calloc((size_t)CASES, OMEGA_TC_PROBE_OUT_BYTES);
    for (unsigned i = 0; i < CASES; i++) { gen_case(&cases[i], i); to_frag(&cases[i], in + (size_t)i * (OMEGA_TC_PROBE_IN_BYTES / 4)); }
    const char *replay = argc > 2 && strcmp(argv[1], "--replay") == 0 ? argv[2] : NULL, *save = argc > 2 && strcmp(argv[1], "--save") == 0 ? argv[2] : NULL;
    if (replay) { /* model search over saved chip words (offline; the words came from a sealed chip run) */
        FILE *rf = fopen(replay, "rb"); if (!rf) return 1;
        size_t got = fread(out, 1, (size_t)CASES * OMEGA_TC_PROBE_OUT_BYTES, rf) + fread(outm, 1, (size_t)CASES * OMEGA_TC_PROBE_OUT_BYTES, rf); fclose(rf);
        if (got != 2u * CASES * OMEGA_TC_PROBE_OUT_BYTES) { printf("short replay file\n"); return 1; }
        printf("replaying chip words from %s\n", replay);
    } else {
        if (run_chip(0, out, in) != 0) { printf("TC_PROBE: FAIL chip\n"); return 1; }
        if (run_chip(1, outm, in) != 0) { printf("TC_PROBE: FAIL chip (mutant)\n"); return 1; }
        if (save) { FILE *sf = fopen(save, "wb"); if (sf) { fwrite(out, 1, (size_t)CASES * OMEGA_TC_PROBE_OUT_BYTES, sf); fwrite(outm, 1, (size_t)CASES * OMEGA_TC_PROBE_OUT_BYTES, sf); fclose(sf); } }
    }
    /* poison and plausibility: relative error vs the exact double sum */
    size_t poison = 0, far = 0, total = (size_t)CASES * 128u;
    for (unsigned i = 0; i < CASES; i++) for (int L = 0; L < 32; L++) for (int q = 0; q < 4; q++) {
        float v; memcpy(&v, &out[(size_t)i * 128 + L * 4 + q], 4);
        if (v == -999.0f) poison++;
        int r, n; out_rc(L, q, &r, &n);
        double ex = cases[i].C[r][n], mag = fabs(ex); for (int k = 0; k < 16; k++) { double p = (double)cases[i].A[r][k] * cases[i].B[k][n]; ex += p; mag += fabs(p); }
        if (fabs(v - ex) > 1e-5 * mag + 1e-37) far++;
    }
    printf("plausibility: %zu poison, %zu of %zu outputs farther than 1e-5 of the term magnitude from the exact sum\n", poison, far, total);
    /* negative control: mutant 1 drops C; it must differ wherever C is nonzero and large enough to matter */
    size_t mdiff = 0, mcand = 0;
    /* counted only where C is clearly significant: |C| >= 2^-8 |D| */
    for (unsigned i = 0; i < CASES; i++) for (int L = 0; L < 32; L++) for (int q = 0; q < 4; q++) {
        int r, n; out_rc(L, q, &r, &n); float dv; memcpy(&dv, &out[(size_t)i * 128 + L * 4 + q], 4);
        if (fabsf(cases[i].C[r][n]) < ldexpf(fabsf(dv), -8) || cases[i].C[r][n] == 0.0f) continue;
        mcand++; if (out[(size_t)i * 128 + L * 4 + q] != outm[(size_t)i * 128 + L * 4 + q]) mdiff++;
    }
    printf("negative control (C dropped): %zu of %zu outputs with significant C differ (%s)\n", mdiff, mcand, mcand && mdiff == mcand ? "caught" : "NOT CAUGHT");
    /* model search */
    int groups[] = { 0, 1, 2, 4, 8, 16 }, best_i = -1; size_t best = 0; HmmaModel bestm = { 0, 0, 0, 0, 0 };
    static const char *clsname[6] = { "narrow", "wide", "mixed+cancel", "bigC", "subnormal", "medium" };
    for (int gi = 0; gi < 6; gi++) for (int ex = -1; ex <= 8; ex++) for (int rn = 0; rn < 2; rn++) for (int nom = 0; nom < 3; nom++) for (int cs = 0; cs < 2; cs++) {
        if (groups[gi] == 0 && (ex || rn || nom || cs)) continue;
        HmmaModel m = { groups[gi], ex, rn, nom, cs };
        size_t match = 0, cls_miss[6] = {0};
        for (unsigned i = 0; i < CASES; i++) for (int L = 0; L < 32; L++) for (int q = 0; q < 4; q++) {
            int r, n; out_rc(L, q, &r, &n);
            float mv = model_out(&m, &cases[i], r, n); uint32_t mu; memcpy(&mu, &mv, 4);
            if (mu == out[(size_t)i * 128 + L * 4 + q]) match++; else cls_miss[i % 6]++;
        }
        if (match * 100 >= total * 90 || match == total)
            printf("model group %2d extra %d %s nominal %d csep %d: %zu of %zu match  (misses by class: %s %zu, %s %zu, %s %zu, %s %zu, %s %zu, %s %zu)\n", m.group, m.extra, m.rn ? "RN" : "RZ", m.nominal, m.csep, match, total,
                   clsname[0], cls_miss[0], clsname[1], cls_miss[1], clsname[2], cls_miss[2], clsname[3], cls_miss[3], clsname[4], cls_miss[4], clsname[5], cls_miss[5]);
        if (match > best) { best = match; bestm = m; best_i = 1; }
    }
    if (best_i > 0) printf("best model: group %d extra %d %s: %zu of %zu\n", bestm.group, bestm.extra, bestm.rn ? "RN" : "RZ", best, total);
    int ok = poison == 0 && far == 0 && mcand && mdiff == mcand;
    if (ok && best == total) printf("TC_PROBE: model=group %d extra %d %s nominal %d csep %d exact %zu/%zu\n", bestm.group, bestm.extra, bestm.rn ? "RN" : "RZ", bestm.nominal, bestm.csep, best, total);
    else printf("TC_PROBE: NO EXACT MODEL (best %zu/%zu)%s\n", best, total, ok ? "" : " and plausibility/control FAILED");
    free(in); free(out); free(outm);
    return ok ? 0 : 1;
}
