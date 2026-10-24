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

/* gpu_qmma_probe_test.c (OWN-30B L1 GEMM, W-L1G, 2026-10-14): characterise the GB10's QMMA.16832.F32.E4M3.E4M3.
 *   --host-only  patch self-test against the nvcc oracle bits; model self-checks
 *   (default)    chip (caller holds flock /tmp/aien-gb10.lock):
 *     1 arithmetic: CASES single-QMMA cases; class "exact" (sparse power-of-two operands, every sum exact) checks the INFERRED
 *       fragment layout (any miss there = layout wrong); per model of tests/bw_qmma_model.h the bit-exact match count;
 *       negative control: mutant 1 (C dropped) must differ wherever C matters; a second run with fewer wait NOPs must match
 *     2 peak: QMMA chains x issue stall sweep; every stored value must be 32 * iters; TFLOPS = 8192 flop per QMMA / time
 * Env QMMA_PROBE_SEED: fresh data (confirmation). --save F / --replay F: chip words to / from a file (offline model search).
 * Verdict last: "QMMA_PROBE: model=... exact N/N" or "QMMA_PROBE: NO EXACT MODEL", then "GPU_QMMA_PROBE: PASS|FAIL". */
#include "omega_blackwell_codegen_qmma.h"
#include "omega_gpu_session.h"
#include "bw_qmma_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CASES 2048u
#define NCLS 7
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
/* random e4m3 code with unbiased exponent field in [lo, hi] (1..15; 0 = subnormal), never NaN */
static uint8_t re4(int lo, int hi) {
    for (;;) { uint8_t e = (uint8_t)(lo + (int)(rnd() % (uint32_t)(hi - lo + 1))), c = (uint8_t)((rnd() & 0x80) | e << 3 | (rnd() & 7)); if ((c & 0x7f) != 0x7f) return c; }
}
static float rf(int lo, int hi) { float m = 1.0f + (float)(rnd() & 127) / 128.0f; int e = lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); float v = ldexpf(m, e); return (rnd() & 1) ? -v : v; }

typedef struct { uint8_t A[16][32], B[32][8]; float C[16][8]; } Case;
static const char *clsname[NCLS] = { "narrow", "wide", "mixed+cancel", "bigC", "subnormal", "medium", "exact" };
static void gen_case(Case *c, unsigned idx) {
    unsigned cls = idx % NCLS;
    for (int r = 0; r < 16; r++) for (int k = 0; k < 32; k++) {
        uint8_t v;
        switch (cls) {
        case 0: v = re4(6, 8); break;
        case 1: v = re4(1, 15); break;
        case 2: v = (k & 1) ? re4(6, 8) : re4(1, 7); break;
        case 3: v = re4(5, 9); break;
        case 4: v = (rnd() & 3) ? re4(0, 0) : re4(1, 2); break;
        case 5: v = re4(3, 11); break;
        default: v = (rnd() % 6 == 0) ? (uint8_t)((rnd() & 0x80) | (uint8_t)((5 + rnd() % 5) << 3)) : 0; break; /* sparse +-2^(-2..2) */
        }
        c->A[r][k] = v;
    }
    for (int k = 0; k < 32; k++) for (int n = 0; n < 8; n++)
        c->B[k][n] = cls == 6 ? ((rnd() % 4 == 0) ? (uint8_t)((rnd() & 0x80) | (uint8_t)((5 + rnd() % 5) << 3)) : 0) : cls == 1 ? re4(1, 15) : re4(4, 10);
    for (int r = 0; r < 16; r++) for (int n = 0; n < 8; n++) {
        float v = idx % 11 == 10 ? 0.0f : rf(-6, 6);
        if (cls == 3) v = rf(12, 16);
        if (cls == 6) v = (float)((int)(rnd() % 9) - 4);
        if (cls == 2 && (rnd() & 1)) { double s = 0; for (int k = 0; k < 32; k++) s += (double)qm_e4m3(c->A[r][k]) * qm_e4m3(c->B[k][n]); v = (float)-s; }
        c->C[r][n] = v;
    }
}
static uint32_t pack4(uint8_t a, uint8_t b, uint8_t cc, uint8_t d) { return (uint32_t)a | (uint32_t)b << 8 | (uint32_t)cc << 16 | (uint32_t)d << 24; }
/* the INFERRED m16n8k32 8-bit fragment layout (header) */
static void to_frag(const Case *c, uint32_t *in) {
    uint32_t *A = in, *B = in + 128; float *C = (float *)(in + 192);
    for (int L = 0; L < 32; L++) {
        int g = L / 4, t = L % 4;
        for (int q = 0; q < 4; q++) { int r = g + (q & 1 ? 8 : 0), k0 = 4 * t + (q & 2 ? 16 : 0); A[L * 4 + q] = pack4(c->A[r][k0], c->A[r][k0 + 1], c->A[r][k0 + 2], c->A[r][k0 + 3]); }
        for (int q = 0; q < 2; q++) { int k0 = 4 * t + 16 * q; B[L * 2 + q] = pack4(c->B[k0][g], c->B[k0 + 1][g], c->B[k0 + 2][g], c->B[k0 + 3][g]); }
        C[L * 4 + 0] = c->C[g][2 * t]; C[L * 4 + 1] = c->C[g][2 * t + 1]; C[L * 4 + 2] = c->C[g + 8][2 * t]; C[L * 4 + 3] = c->C[g + 8][2 * t + 1];
    }
}
static void out_rc(int L, int q, int *r, int *n) { int g = L / 4, t = L % 4; *r = g + (q >= 2 ? 8 : 0); *n = 2 * t + (q & 1); }
static float model_out(const QmmaModel *m, const Case *c, int r, int n) {
    float a[32], b[32]; for (int k = 0; k < 32; k++) { a[k] = qm_e4m3(c->A[r][k]); b[k] = qm_e4m3(c->B[k][n]); }
    return qm_dot32(m, a, b, c->C[r][n]);
}
static double exact_out(const Case *c, int r, int n, double *mag) {
    double s = c->C[r][n]; *mag = fabs(s); for (int k = 0; k < 32; k++) { double p = (double)qm_e4m3(c->A[r][k]) * qm_e4m3(c->B[k][n]); s += p; *mag += fabs(p); } return s;
}

static int host_only(void) {
    int fail = 0; char msg[160];
    int rc = omega_qmma_patch_selftest(msg, sizeof msg);
    printf("QMMA patch self-test: %s (%s)\n", rc ? "FAIL" : "PASS", msg); if (rc) fail++;
    if (qm_e4m3(0x38) != 1.0f || qm_e4m3(0x7e) != 448.0f || qm_e4m3(0x01) != ldexpf(1.0f, -9) || qm_e4m3(0xb8) != -1.0f) { printf("e4m3 decode FAIL\n"); fail++; }
    for (int i = 0; i < 1000; i++) {
        float a[32], b[32]; for (int k = 0; k < 32; k++) { a[k] = (float)((int)(rnd() % 17) - 8); b[k] = (float)((int)(rnd() % 9) - 4); }
        float c = (float)((int)(rnd() % 33) - 16); double ex = c; for (int k = 0; k < 32; k++) ex += (double)a[k] * b[k];
        int groups[] = { 0, 1, 2, 4, 8, 16, 32 };
        for (int gi = 0; gi < 7; gi++) for (int rn = 0; rn < 2; rn++) { QmmaModel m = { groups[gi], 24, rn, 0, 0 }; if (qm_dot32(&m, a, b, c) != (float)ex) fail++; }
    }
    for (int c = 0; c < 256; c++) if ((c & 0x7f) != 0x7f && qm_e4m3_rne_sat(qm_e4m3((uint8_t)c)) != c) { printf("e4m3 round trip FAIL at %02x\n", c); fail++; }
    if (qm_e4m3_rne_sat(500.0f) != 0x7e || qm_e4m3_rne_sat(ldexpf(1.0f, -10)) != 0 || qm_e4m3_rne_sat(ldexpf(3.0f, -10)) != 2 || qm_e4m3_rne_sat(1.0625f) != 0x38) { printf("e4m3 rounding FAIL\n"); fail++; }
    { static Case c; gen_case(&c, 6); uint32_t in[320]; to_frag(&c, in); if ((in[0] & 0xff) != c.A[0][0] || ((in[1] >> 8) & 0xff) != c.A[8][1] || ((in[128 + 5 * 2 + 1] >> 24) & 0xff) != c.B[16 + 4 + 3][1]) fail++; }
    static BlackwellIRProgram p;
    if (omega_blackwell_codegen_qmma_probe_prog(0, 4, &p) != 0 || omega_blackwell_codegen_qmma_peak_prog(64, 16, 2, 128, &p) != 0) { printf("codegen refused\n"); fail++; }
    printf("host-only checks: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail;
}

static int load_code(BlackwellIRProgram *prog, OmegaGpuMem *code) {
    size_t need = omega_bw_program_code_bytes(prog); OmegaBlackwellKernel k;
    if (omega_gpu_session_alloc_code(need, code) != 0 || omega_blackwell_kernel_encode_into(prog, code->cpu, need, &k) != 0) return -1;
    if (omega_qmma_patch_code(prog, code->cpu, need) < 1) { printf("QMMA patch failed\n"); return -1; }
    return omega_gpu_session_sync_to_gpu(code);
}
static int run_probe(int mutant, uint32_t nops, uint32_t *out_words, const uint32_t *in_words) {
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    if (!prog || omega_blackwell_codegen_qmma_probe_prog(mutant, nops, prog) != 0) { printf("codegen failed\n"); free(prog); return -1; }
    int rc = -1;
    omega_gpu_session_lock();
    OmegaGpuMem code = { 0 }, in = { 0 }, out = { 0 };
    if (!omega_gpu_session_open()) { printf("open failed: %s\n", omega_gpu_session_last_error()); goto done; }
    if (load_code(prog, &code) != 0) { printf("code upload failed\n"); goto done; }
    if (omega_gpu_session_alloc((size_t)CASES * 1280u, &in) != 0 || omega_gpu_session_alloc((size_t)CASES * 512u, &out) != 0) { printf("alloc failed\n"); goto done; }
    memcpy(in.cpu, in_words, (size_t)CASES * 1280u);
    { float *o = out.cpu; for (size_t i = 0; i < (size_t)CASES * 128u; i++) o[i] = -999.0f; }
    if (omega_gpu_session_sync_to_gpu(&in) != 0 || omega_gpu_session_sync_to_gpu(&out) != 0) goto done;
    {
        uint32_t args[4] = { (uint32_t)in.va, (uint32_t)(in.va >> 32), (uint32_t)out.va, (uint32_t)(out.va >> 32) };
        uint32_t decl = omega_bw_gpr_declared(prog->regalloc.peak_gpr_usage);
        OmegaGpuLaunch L = { .code_va = code.va, .gpr_count = decl < 64 ? 64 : decl, .threads_x = 32, .threads_y = 1, .grid_x = CASES, .grid_y = 1,
                             .num_elements = (size_t)CASES * 128u, .args = args, .n_args = 4, .timeout_ms = 10000 };
        uint64_t ns = 0;
        if (omega_gpu_session_launch(&L, &ns, NULL) != 0) { printf("launch failed: %s\n", omega_gpu_session_last_error()); goto done; }
        if (omega_gpu_session_sync_to_cpu(&out) != 0) goto done;
        printf("chip probe mutant %d nops %u: declared %u, %llu ns\n", mutant, nops, L.gpr_count, (unsigned long long)ns);
    }
    memcpy(out_words, out.cpu, (size_t)CASES * 512u);
    rc = 0;
done:
    if (code.cpu) omega_gpu_session_free(&code);
    if (in.cpu) omega_gpu_session_free(&in);
    if (out.cpu) omega_gpu_session_free(&out);
    omega_gpu_session_unlock();
    free(prog);
    return rc;
}


/* convert probe: NP pairs through F2FP.SATFINITE.E4M3; byte order and rounding against qm_e4m3_rne_sat */
#define NP (1u << 20)
static float cvt_input(uint32_t i) {
    switch (i % 8) {
    case 0: return u2f((rnd() & 0x80000000u) | (uint32_t)(127 - 14 + (int)(rnd() % 25)) << 23 | (rnd() & 0x7fffff));      /* 2^-14 .. 2^10, full mantissa */
    case 1: { uint8_t c = (uint8_t)(rnd() & 0xff); if ((c & 0x7f) >= 0x7e) c = 0x10; float a = qm_e4m3(c), b = qm_e4m3((uint8_t)(c + 1)); return (a + b) * 0.5f; } /* exact ties */
    case 2: return u2f((rnd() & 0x80000000u) | (rnd() & 0x7fffff) | (uint32_t)(127 - 12 + (int)(rnd() % 4)) << 23);     /* subnormal e4m3 range */
    case 3: { static const uint32_t sp[] = { 0, 0x80000000u, 0x43e00000u, 0x43e80000u, 0x43e7ffffu, 0x7f800000u, 0xff800000u, 0x3a800000u, 0x3a000000u, 0x00000001u, 0x447a0000u, 0xc3e10000u }; return u2f(sp[rnd() % 12]); }
    default: return u2f((rnd() & 0x80000000u) | (uint32_t)(127 - 8 + (int)(rnd() % 17)) << 23 | (rnd() & 0x7fffff));
    }
}
static int cvt_probe(void) {
    static BlackwellIRProgram p; int fail = 0;
    if (omega_blackwell_codegen_f2e4_probe_prog(&p) != 0) { printf("cvt codegen refused\n"); return 1; }
    float *x = malloc((size_t)NP * 8); for (uint32_t i = 0; i < 2 * NP; i++) x[i] = cvt_input(i);
    omega_gpu_session_lock();
    OmegaGpuMem code = { 0 }, in = { 0 }, out = { 0 };
    if (!omega_gpu_session_open() || load_code(&p, &code) != 0 || omega_gpu_session_alloc((size_t)NP * 8, &in) != 0 || omega_gpu_session_alloc((size_t)NP * 4, &out) != 0) { printf("cvt setup failed: %s\n", omega_gpu_session_last_error()); fail = 1; goto done; }
    memcpy(in.cpu, x, (size_t)NP * 8); memset(out.cpu, 0xa5, (size_t)NP * 4);
    {   uint32_t args[4] = { (uint32_t)in.va, (uint32_t)(in.va >> 32), (uint32_t)out.va, (uint32_t)(out.va >> 32) };
        OmegaGpuLaunch L = { .code_va = code.va, .gpr_count = 64, .threads_x = 128, .threads_y = 1, .grid_x = NP / 128u, .grid_y = 1, .num_elements = NP, .args = args, .n_args = 4, .timeout_ms = 10000 };
        uint64_t ns = 0;
        if (omega_gpu_session_sync_to_gpu(&in) || omega_gpu_session_sync_to_gpu(&out) || omega_gpu_session_launch(&L, &ns, NULL) || omega_gpu_session_sync_to_cpu(&out)) { printf("cvt launch failed: %s\n", omega_gpu_session_last_error()); fail = 1; goto done; } }
    {   const uint32_t *o = out.cpu; size_t hi_ab = 0, hi_ba = 0, upper = 0, nan_n = 0, n = 0, tie_ab = 0, ties = 0; uint32_t first_bad = 0xffffffffu;
        for (uint32_t i = 0; i < NP; i++) {
            float a = x[2 * i], b = x[2 * i + 1];
            if (isnan(a) || isnan(b)) { nan_n++; continue; }
            n++; if (o[i] >> 16) upper++;
            uint32_t ab = (uint32_t)qm_e4m3_rne_sat(a) << 8 | qm_e4m3_rne_sat(b), ba = (uint32_t)qm_e4m3_rne_sat(b) << 8 | qm_e4m3_rne_sat(a);
            if ((o[i] & 0xffff) == ab) hi_ab++; else if (first_bad == 0xffffffffu) first_bad = i;
            if ((o[i] & 0xffff) == ba) hi_ba++;
            if (i % 4 == 1 || (2 * i + 1) % 8 == 1) { ties++; if ((o[i] & 0xffff) == ab) tie_ab++; }
        }
        printf("cvt probe: %zu pairs; Ra->high byte matches %zu, Ra->low byte matches %zu; upper 16 bits nonzero %zu; ties %zu of %zu exact\n", n, hi_ab, hi_ba, upper, tie_ab, ties);
        if (first_bad != 0xffffffffu) printf("  first miss pair %u: %a %a -> chip %04x want %04x\n", first_bad, x[2 * first_bad], x[2 * first_bad + 1], o[first_bad] & 0xffff, (uint32_t)qm_e4m3_rne_sat(x[2 * first_bad]) << 8 | qm_e4m3_rne_sat(x[2 * first_bad + 1]));
        printf("CVT_PROBE: %s\n", hi_ab == n && upper == 0 ? "e4m3(Ra) << 8 | e4m3(Rb), RNE satfinite, exact on every pair" : "MISMATCH");
        if (hi_ab != n || upper) fail = 1;
        (void)nan_n;
    }
done:
    if (code.cpu) omega_gpu_session_free(&code);
    if (in.cpu) omega_gpu_session_free(&in);
    if (out.cpu) omega_gpu_session_free(&out);
    omega_gpu_session_unlock();
    free(x);
    return fail;
}

static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static int peak(void) {
    static const struct { uint32_t stall, chains, threads, ctas; } pts[] = {
        { 8, 16, 128, 192 }, { 4, 16, 128, 192 }, { 2, 16, 128, 192 }, { 1, 16, 128, 192 }, { 2, 8, 128, 192 }, { 2, 4, 128, 192 },
        { 2, 16, 256, 192 }, { 2, 16, 256, 384 }, { 1, 16, 512, 96 }, { 4, 8, 256, 384 },
    };
    const uint32_t iters = 4096; int fail = 0;
    omega_gpu_session_lock();
    if (!omega_gpu_session_open()) { printf("open failed: %s\n", omega_gpu_session_last_error()); omega_gpu_session_unlock(); return 1; }
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        static BlackwellIRProgram p; OmegaGpuMem code = { 0 }, out = { 0 };
        if (omega_blackwell_codegen_qmma_peak_prog(iters, pts[i].chains, pts[i].stall, pts[i].threads, &p) != 0) { printf("codegen refused\n"); fail++; continue; }
        size_t ob = (size_t)pts[i].ctas * pts[i].threads * 16u * pts[i].chains;
        uint32_t decl = omega_bw_gpr_declared(p.regalloc.peak_gpr_usage);
        if (load_code(&p, &code) != 0 || omega_gpu_session_alloc(ob, &out) != 0) { printf("upload failed: %s\n", omega_gpu_session_last_error()); fail++; break; }
        uint64_t ns[5]; int ok = 1; size_t bad = 0;
        for (int r = 0; r < 5 && ok; r++) {
            ns[r] = 0;
            float *o = out.cpu; for (size_t q = 0; q < ob / 4; q++) o[q] = -1.0f;
            uint32_t args[2] = { (uint32_t)out.va, (uint32_t)(out.va >> 32) };
            OmegaGpuLaunch L = { .code_va = code.va, .gpr_count = decl, .threads_x = pts[i].threads, .threads_y = 1, .grid_x = pts[i].ctas, .grid_y = 1,
                                 .num_elements = ob / 4, .args = args, .n_args = 2, .timeout_ms = 20000 };
            if (omega_gpu_session_sync_to_gpu(&out) != 0 || omega_gpu_session_launch(&L, &ns[r], NULL) != 0 || omega_gpu_session_sync_to_cpu(&out) != 0) { printf("launch failed: %s\n", omega_gpu_session_last_error()); ok = 0; break; }
            for (size_t q = 0; q < ob / 4; q++) if (o[q] != 32.0f * iters) bad++;
        }
        if (!ok) { fail++; omega_gpu_session_free(&code); omega_gpu_session_free(&out); break; }
        qsort(ns, 5, 8, cmp_u64);
        double qm = (double)pts[i].ctas * (pts[i].threads / 32u) * iters * pts[i].chains;
        printf("QMMA peak stall %u chains %u threads %u ctas %u (declared %u): median %.1f us, %.2f TFLOPS, %zu wrong values\n", pts[i].stall, pts[i].chains, pts[i].threads, pts[i].ctas, decl,
               ns[2] / 1e3, qm * 8192.0 / ns[2] / 1e3, bad);
        if (bad) fail++;
        omega_gpu_session_free(&code); omega_gpu_session_free(&out);
    }
    omega_gpu_session_unlock();
    return fail;
}

int main(int argc, char **argv) {
    const char *replay = NULL, *save = NULL; int host = 0, do_peak = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) host = 1;
        else if (!strcmp(argv[i], "--no-peak")) do_peak = 0;
        else if (!strcmp(argv[i], "--replay") && i + 1 < argc) replay = argv[++i];
        else if (!strcmp(argv[i], "--save") && i + 1 < argc) save = argv[++i];
    }
    if (getenv("QMMA_PROBE_SEED")) rng = strtoull(getenv("QMMA_PROBE_SEED"), NULL, 0) | 1u;
    int fail = host_only();
    if (host || fail) { printf("GPU_QMMA_PROBE: %s\n", fail ? "FAIL" : "PASS"); return fail ? 1 : 0; }
    static Case cases[CASES];
    const size_t ow = (size_t)CASES * 128u;
    uint32_t *in = calloc((size_t)CASES, 1280u), *out = calloc(ow, 4), *outm = calloc(ow, 4), *out2 = calloc(ow, 4);
    for (unsigned i = 0; i < CASES; i++) { gen_case(&cases[i], i); to_frag(&cases[i], in + (size_t)i * 320u); }
    if (replay) {
        FILE *rf = fopen(replay, "rb"); if (!rf) return 1;
        size_t got = fread(out, 4, ow, rf) + fread(outm, 4, ow, rf) + fread(out2, 4, ow, rf); fclose(rf);
        if (got != 3 * ow) { printf("short replay file\n"); return 1; }
        printf("replaying chip words from %s\n", replay); do_peak = 0;
    } else {
        if (run_probe(0, 8, out, in) || run_probe(1, 8, outm, in) || run_probe(0, 2, out2, in)) { printf("GPU_QMMA_PROBE: FAIL chip\n"); return 1; }
        if (save) { FILE *sf = fopen(save, "wb"); if (sf) { fwrite(out, 4, ow, sf); fwrite(outm, 4, ow, sf); fwrite(out2, 4, ow, sf); fclose(sf); } }
    }
    size_t poison = 0, far = 0, exact_miss = 0, exact_n = 0, waitdiff = 0;
    for (unsigned i = 0; i < CASES; i++) for (int L = 0; L < 32; L++) for (int q = 0; q < 4; q++) {
        size_t w = (size_t)i * 128 + L * 4 + q; float v = u2f(out[w]);
        if (v == -999.0f) poison++;
        if (out[w] != out2[w]) waitdiff++;
        int r, n; out_rc(L, q, &r, &n); double mag, ex = exact_out(&cases[i], r, n, &mag);
        if (fabs(v - ex) > 1e-2 * mag + 1e-30) far++;
        if (i % NCLS == 6) { exact_n++; if ((double)v != ex) exact_miss++; }
    }
    printf("plausibility: %zu poison, %zu of %zu outputs farther than 1e-2 of the term magnitude from the exact sum\n", poison, far, ow);
    printf("layout check (class exact, every sum representable): %zu of %zu differ from the exact sum (%s)\n", exact_miss, exact_n, exact_miss ? "LAYOUT OR ARITHMETIC WRONG" : "layout confirmed");
    printf("wait check: 2-NOP run vs 8-NOP run: %zu of %zu words differ\n", waitdiff, ow);
    size_t mdiff = 0, mcand = 0;
    for (unsigned i = 0; i < CASES; i++) for (int L = 0; L < 32; L++) for (int q = 0; q < 4; q++) {
        size_t w = (size_t)i * 128 + L * 4 + q; int r, n; out_rc(L, q, &r, &n); float dv = u2f(out[w]);
        if (cases[i].C[r][n] == 0.0f || fabsf(cases[i].C[r][n]) < ldexpf(fabsf(dv), -8)) continue;
        mcand++; if (out[w] != outm[w]) mdiff++;
    }
    printf("negative control (C dropped): %zu of %zu outputs with significant C differ (%s)\n", mdiff, mcand, mcand && mdiff == mcand ? "caught" : "NOT CAUGHT");
    int groups[] = { 0, 1, 2, 4, 8, 16, 32 }; size_t best = 0; QmmaModel bm = { 0, 0, 0, 0, 0 };
    for (int gi = 0; gi < 7; gi++) for (int keep = 10; keep <= 32; keep++) for (int rn = 0; rn < 2; rn++) for (int nom = 0; nom < 3; nom++) for (int cs = 0; cs < 2; cs++) {
        if (groups[gi] == 0 && (keep != 10 || rn || nom || cs)) continue;
        QmmaModel m = { groups[gi], keep, rn, nom, cs };
        size_t match = 0, miss[NCLS] = { 0 };
        for (unsigned i = 0; i < CASES; i++) for (int L = 0; L < 32; L++) for (int q = 0; q < 4; q++) {
            int r, n; out_rc(L, q, &r, &n);
            if (f2u(model_out(&m, &cases[i], r, n)) == out[(size_t)i * 128 + L * 4 + q]) match++; else miss[i % NCLS]++;
        }
        if (match * 100 >= ow * 95)
            printf("model group %2d keep %d %s nominal %d csep %d: %zu of %zu (misses: %s %zu %s %zu %s %zu %s %zu %s %zu %s %zu %s %zu)\n", m.group, m.keep, m.rn ? "RN" : "RZ", m.nominal, m.csep, match, ow,
                   clsname[0], miss[0], clsname[1], miss[1], clsname[2], miss[2], clsname[3], miss[3], clsname[4], miss[4], clsname[5], miss[5], clsname[6], miss[6]);
        if (match > best) { best = match; bm = m; }
    }
    int ok = poison == 0 && far == 0 && exact_miss == 0 && waitdiff == 0 && mcand && mdiff == mcand;
    if (ok && best == ow) printf("QMMA_PROBE: model=group %d keep %d %s nominal %d csep %d exact %zu/%zu\n", bm.group, bm.keep, bm.rn ? "RN" : "RZ", bm.nominal, bm.csep, best, ow);
    else printf("QMMA_PROBE: NO EXACT MODEL (best group %d keep %d %s nominal %d csep %d: %zu/%zu)%s\n", bm.group, bm.keep, bm.rn ? "RN" : "RZ", bm.nominal, bm.csep, best, ow, ok ? "" : " and plausibility/layout/control FAILED");
    int pf = do_peak ? peak() : 0;
    int cf = replay ? 0 : cvt_probe();
    fail = !ok || best != ow || pf || cf;
    printf("GPU_QMMA_PROBE: %s\n", fail ? "FAIL" : "PASS");
    free(in); free(out); free(outm); free(out2);
    return fail;
}
