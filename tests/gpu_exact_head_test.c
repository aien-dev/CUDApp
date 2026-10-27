/*
 * gpu_exact_head_test: the exact two-pass lm_head (HUM lever L314) on the Qwen3.6-35B-A3B head.
 *   --sim   warp-simulator proofs of the new kernels: XBOUND (lo <= s - c|h|, hi >= s + c|h| on every row despite f32
 *           rounding, tight to 2^-8, rows past count untouched; mutant 1 caught). The MASK recheck form is proven in
 *           gpu_bf16_gemv_test --sim (candidates bit-identical to the plain kernel, mutant 4 caught).
 *   --chip <export-dir> <hidden.bin> [--limit N] [--mutant 1] [--out receipt.json]   per hidden record (the 35B host
 *           reference's final hidden states): the full bf16 head + ARGMAX x2 as one list, the two-pass head (FP8 block
 *           screen, XBOUND, ARGMAX x2 for tau, MASK recheck, ARGMAX x2) as another; token identical, every candidate
 *           logit bit-identical to the full head's, timing of both lists. --mutant 1 (XBOUND without the bound) must
 *           lose tokens. Inputs from sc q36_exact_head export (weights stay outside git).
 * Verdict line last: "gpu_exact_head_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_blackwell_codegen_dec.h"
#include "omega_gpu_bf16_gemv_api.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_cmdlist.h"
#include "omega_gpu_session.h"
#include "bw_warp_sim_k.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float frand(uint32_t *s, float lo, float hi) { return lo + (hi - lo) * ((float)(lcg(s) >> 8) / 16777216.0f); }

static void put64(uint32_t *cb, uint32_t off, const void *p) { uint64_t a = (uintptr_t)p; cb[off / 4] = (uint32_t)a; cb[off / 4 + 1] = (uint32_t)(a >> 32); }
/* launch argument words start at the constant bank's 0x380 */
static void put64v(uint32_t *args, uint32_t off, uint64_t a) { args[(off - 0x380u) / 4] = (uint32_t)a; args[(off - 0x380u) / 4 + 1] = (uint32_t)(a >> 32); }

/* XBOUND on n rows (grid ceil(n / 256), the last CTA partial). hscale spreads |h| over decades. Returns the number of
 * rows whose lo / hi break the bound (or are loose by more than 2^-8 relative); *untouched = rows past n left alone. */
static int sim_xbound(uint32_t n, int mutant, uint32_t seed, float hscale, size_t *bad, size_t *loose, int *untouched, char *msg, size_t msgn) {
    static BlackwellIRProgram prog;
    OmegaDecSpec spec; memset(&spec, 0, sizeof spec);
    spec.kind = OMEGA_DEC_XBOUND; spec.mutant = mutant;
    if (omega_blackwell_codegen_dec_prog(&spec, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    const uint32_t ctas = (n + 255u) / 256u, cap = ctas * 256u + 8u;
    float *s = aligned_alloc(16, cap * 4), *c = aligned_alloc(16, cap * 4), *h = aligned_alloc(16, 2048 * 4);
    float *lo = aligned_alloc(16, cap * 4), *hi = aligned_alloc(16, cap * 4);
    uint8_t *smem = aligned_alloc(16, OMEGA_DEC_SHARED_BYTES);
    uint32_t st = seed;
    for (uint32_t i = 0; i < 2048; i++) h[i] = frand(&st, -1.0f, 1.0f) * hscale * (i % 97 == 0 ? 40.0f : 1.0f);
    for (uint32_t i = 0; i < cap; i++) {
        s[i] = frand(&st, -30.0f, 30.0f) * (i % 7 == 0 ? 1e3f : 1.0f);
        c[i] = frand(&st, 0.0f, 0.02f);
        lo[i] = hi[i] = -999.0f;
    }
    s[0] = 0.0f; c[1] = 0.0f; s[2] = -0.0f;
    double hn = 0; for (uint32_t i = 0; i < 2048; i++) hn += (double)h[i] * h[i];
    hn = sqrt(hn);
    WsSim w; memset(&w, 0, sizeof w);
    w.p = &prog; w.threads = OMEGA_DEC_THREADS; w.order = (int)(seed & 1u); w.smem = smem; w.smem_bytes = OMEGA_DEC_SHARED_BYTES;
    put64(w.cbank, OMEGA_DEC_XB_S, s); put64(w.cbank, OMEGA_DEC_XB_C, c); put64(w.cbank, OMEGA_DEC_XB_H, h);
    put64(w.cbank, OMEGA_DEC_XB_LO, lo); put64(w.cbank, OMEGA_DEC_XB_HI, hi); w.cbank[OMEGA_DEC_XB_N / 4] = n;
    w.nranges = 5;
    w.ranges[0].lo = (uintptr_t)s; w.ranges[0].hi = (uintptr_t)(s + cap);
    w.ranges[1].lo = (uintptr_t)c; w.ranges[1].hi = (uintptr_t)(c + cap);
    w.ranges[2].lo = (uintptr_t)h; w.ranges[2].hi = (uintptr_t)(h + 2048);
    w.ranges[3].lo = (uintptr_t)lo; w.ranges[3].hi = (uintptr_t)(lo + cap);
    w.ranges[4].lo = (uintptr_t)hi; w.ranges[4].hi = (uintptr_t)(hi + cap);
    int err = ws_run(&w, ctas, 1);
    if (err) snprintf(msg, msgn, "%s (pc %d)", w.msg, w.err_pc);
    else {
        size_t b = 0, l = 0;
        for (uint32_t i = 0; i < n; i++) {
            const double B = (double)c[i] * hn, sv = s[i];
            if (!((double)lo[i] <= sv - B && (double)hi[i] >= sv + B)) b++;
            const double slack = B * 0x1p-8 + fabs(sv) * 0x1p-19 + 1e-29;
            if (sv - (double)lo[i] > B + slack || (double)hi[i] - sv > B + slack) l++;
        }
        int u = 1; for (uint32_t i = n; i < cap; i++) u &= lo[i] == -999.0f && hi[i] == -999.0f;
        *bad = b; *loose = l; *untouched = u;
        snprintf(msg, msgn, "%u rows: %zu break the bound, %zu looser than 2^-8, rows past count %s", n, b, l, u ? "untouched" : "WRITTEN");
    }
    free(s); free(c); free(h); free(lo); free(hi); free(smem);
    return err;
}

/* XBOUND mode 1 (SCALE) on n values: out == in * 2^64 bit for bit, values past n untouched */
static int sim_xscale(uint32_t n, uint32_t seed, size_t *bad, char *msg, size_t msgn) {
    static BlackwellIRProgram prog;
    OmegaDecSpec spec; memset(&spec, 0, sizeof spec);
    spec.kind = OMEGA_DEC_XBOUND; spec.mode = 1;
    if (omega_blackwell_codegen_dec_prog(&spec, &prog) != 0) { snprintf(msg, msgn, "codegen refused"); return -1; }
    const uint32_t ctas = (n + 255u) / 256u, cap = ctas * 256u + 8u;
    float *in = aligned_alloc(16, cap * 4), *out = aligned_alloc(16, cap * 4);
    uint32_t st = seed;
    for (uint32_t i = 0; i < cap; i++) { in[i] = frand(&st, -3.0f, 3.0f) * (i % 5 == 0 ? 1e-20f : 1.0f); out[i] = -999.0f; }
    in[0] = 0.0f; in[1] = -0.0f; in[2] = 1e30f;
    WsSim w; memset(&w, 0, sizeof w);
    w.p = &prog; w.threads = OMEGA_DEC_THREADS; w.order = (int)(seed & 1u);
    put64(w.cbank, OMEGA_DEC_XB_S, in); put64(w.cbank, OMEGA_DEC_XB_LO, out); w.cbank[OMEGA_DEC_XB_N / 4] = n;
    w.nranges = 2;
    w.ranges[0].lo = (uintptr_t)in; w.ranges[0].hi = (uintptr_t)(in + cap);
    w.ranges[1].lo = (uintptr_t)out; w.ranges[1].hi = (uintptr_t)(out + cap);
    int err = ws_run(&w, ctas, 1);
    if (err) snprintf(msg, msgn, "%s (pc %d)", w.msg, w.err_pc);
    else {
        size_t b = 0;
        for (uint32_t i = 0; i < n; i++) { const float want = in[i] * 0x1p64f; b += memcmp(&out[i], &want, 4) != 0; }
        for (uint32_t i = n; i < cap; i++) b += out[i] != -999.0f;
        *bad = b;
        snprintf(msg, msgn, "%u values: %zu differ from h * 2^64 or written past count", n, b);
    }
    free(in); free(out);
    return err;
}

static void sim_battery(void) {
    static const struct { uint32_t n, seed; float hscale; } pts[] = { { 700, 3, 1.0f }, { 256, 4, 1e-3f }, { 1000, 5, 30.0f }, { 37, 6, 1.0f } };
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        char msg[300]; size_t bad = 1, loose = 1; int u = 0;
        int err = sim_xbound(pts[i].n, 0, pts[i].seed, pts[i].hscale, &bad, &loose, &u, msg, sizeof msg);
        CHECK(!err && bad == 0 && loose == 0 && u, "sim XBOUND n %u: %s", pts[i].n, msg);
        printf("sim XBOUND n %u |h| x%g: %s %s\n", pts[i].n, pts[i].hscale, err ? "ERROR" : bad || loose || !u ? "MISMATCH" : "ok", msg);
    }
    char msg[300]; size_t bad = 0, loose = 0; int u = 0;
    int err = sim_xbound(700, 1, 3, 1.0f, &bad, &loose, &u, msg, sizeof msg);
    CHECK(!err && bad != 0, "sim XBOUND mutant 1 must be caught: %s", msg);
    printf("sim XBOUND mutant 1: %s %s\n", err ? "ERROR" : bad ? "caught" : "NOT CAUGHT", msg);
    OmegaDecSpec m2; memset(&m2, 0, sizeof m2); m2.kind = OMEGA_DEC_XBOUND; m2.mutant = 2;
    static BlackwellIRProgram p2;
    CHECK(omega_blackwell_codegen_dec_prog(&m2, &p2) != 0, "XBOUND refuses mutant 2");
    for (uint32_t n2 = 2048; n2 >= 37; n2 = n2 == 2048 ? 37 : 0) {
        size_t xb = 1; int xe = sim_xscale(n2, 7, &xb, msg, sizeof msg);
        CHECK(!xe && xb == 0, "sim XBOUND SCALE n %u: %s", n2, msg);
        printf("sim XBOUND SCALE n %u: %s %s\n", n2, xe ? "ERROR" : xb ? "MISMATCH" : "ok", msg);
    }
    m2.mutant = 1; m2.mode = 1;
    CHECK(omega_blackwell_codegen_dec_prog(&m2, &p2) != 0, "XBOUND SCALE refuses a mutant");
}

/* ------------------------------------------------------------- chip */
#define XH_N 248320u
#define XH_K 2048u
#define NEG_PAD (-1e30f)

static void *read_all(const char *dir, const char *name, size_t want) {
    char path[1024]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb"); if (!f) { printf("cannot open %s\n", path); return NULL; }
    void *p = malloc(want ? want : 1); size_t got = p ? fread(p, 1, want, f) : 0;
    int extra = fgetc(f) != EOF; fclose(f);
    if (got != want || extra) { printf("%s: expected exactly %zu bytes\n", path, want); free(p); return NULL; }
    return p;
}

typedef struct { OmegaGpuMem code; uint32_t gpr; } XKern;
static int xkern(XKern *k, OmegaDecKind kind, uint32_t mode, int mutant) {
    OmegaDecSpec s; memset(&s, 0, sizeof s); s.kind = kind; s.mode = mode; s.mutant = mutant;
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_dec_prog(&s, &prog) != 0) return -1;
    size_t need = omega_bw_program_code_bytes(&prog);
    if (!need || omega_gpu_session_alloc_code(need, &k->code) != 0) return -1;
    OmegaBlackwellKernel kk;
    if (omega_blackwell_kernel_encode_into(&prog, k->code.cpu, need, &kk) != 0 || kk.code_size != need) return -1;
    if (omega_gpu_session_sync_to_gpu(&k->code) != 0) return -1;
    k->gpr = omega_bw_gpr_declared(kk.gpr_count); if (k->gpr < 64u) k->gpr = 64u;
    return 0;
}
static int xadd(OmegaGpuCmdList *cl, const XKern *k, uint32_t gx, size_t elems, const uint32_t *args, uint32_t n_args) {
    OmegaGpuLaunch L; memset(&L, 0, sizeof L);
    L.code_va = k->code.va; L.gpr_count = k->gpr; L.threads_x = OMEGA_DEC_THREADS; L.threads_y = 1; L.grid_x = gx; L.grid_y = 1;
    L.num_elements = elems; L.args = args; L.n_args = n_args; L.timeout_ms = 20000;
    return omega_gpu_cmdlist_add(cl, &L) < 0 ? -1 : 0;
}
/* ARGMAX x2 over v (padded to ctas * 2048): value -> out_v, index -> out_i */
static int xargmax(OmegaGpuCmdList *cl, const XKern *a0k, const XKern *a1k, uint32_t ctas, uint64_t v, uint64_t amv, uint64_t ami, uint64_t out_v, uint64_t out_i) {
    uint32_t a0[OMEGA_DEC_AM_WORDS], a1[OMEGA_DEC_AM_WORDS]; memset(a0, 0, sizeof a0); memset(a1, 0, sizeof a1);
    put64v(a0, OMEGA_DEC_AM_V, v); put64v(a0, OMEGA_DEC_AM_OV, amv); put64v(a0, OMEGA_DEC_AM_OI, ami);
    put64v(a1, OMEGA_DEC_AM_V, amv); put64v(a1, OMEGA_DEC_AM_I, ami); put64v(a1, OMEGA_DEC_AM_OV, out_v); put64v(a1, OMEGA_DEC_AM_OI, out_i);
    if (xadd(cl, a0k, ctas, (size_t)ctas * OMEGA_DEC_ARGMAX_SPAN, a0, OMEGA_DEC_AM_WORDS)) return -1;
    return xadd(cl, a1k, 1, OMEGA_DEC_ARGMAX_SPAN, a1, OMEGA_DEC_AM_WORDS);
}
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static double pct(uint64_t *v, size_t n, double p) { qsort(v, n, sizeof *v, cmp_u64); return (double)v[(size_t)((double)(n - 1) * p + 0.5)]; }

/* Full head vs two-pass head on every hidden record: token identical, candidate logits bit-identical, timing. */
static void chip(const char *dir, const char *hidden_path, uint32_t limit, int mutant, const char *out_path) {
    const uint32_t n = XH_N, k = XH_K, ctas = (n + OMEGA_DEC_ARGMAX_SPAN - 1) / OMEGA_DEC_ARGMAX_SPAN, P = ctas * OMEGA_DEC_ARGMAX_SPAN;
    uint16_t *head = read_all(dir, "head.bf16", (size_t)n * k * 2);
    uint8_t *codes = read_all(dir, "fp8b.codes", (size_t)n * k);
    uint16_t *scales = read_all(dir, "fp8b.scales", (size_t)(n / 128) * (k / 128) * 2);
    float *coef = read_all(dir, "coef.f32", (size_t)n * 4);
    FILE *hf = fopen(hidden_path, "rb");
    CHECK(head && codes && scales && coef && hf, "chip inputs readable (export dir %s, hidden %s)", dir, hidden_path);
    if (!head || !codes || !scales || !coef || !hf) return;
    fseek(hf, 0, SEEK_END); const size_t rec_bytes = 16 + 4 * (size_t)k, nrec_all = (size_t)ftell(hf) / rec_bytes; fseek(hf, 0, SEEK_SET);
    const size_t nrec = limit && limit < nrec_all ? limit : nrec_all;
    uint8_t *recs = malloc(nrec * rec_bytes);
    CHECK(recs && fread(recs, rec_bytes, nrec, hf) == nrec, "read %zu hidden records", nrec);
    fclose(hf);
    OmegaGpuBf16Tensor *tb = NULL; OmegaGpuFp8Tensor *t8 = NULL;
    int rb = omega_gpu_bf16_tensor_upload(n, k, head, &tb), r8 = omega_gpu_fp8_tensor_upload(1, n, k, codes, scales, &t8);
    CHECK(rb == 0 && r8 == 0, "upload bf16 head (%d) and fp8b screen (%d): %s", rb, r8, omega_gpu_session_last_error());
    free(head); free(codes); free(scales);
    if (rb || r8) return;
    float *xw = calloc(k, 4), *yw = malloc((size_t)n * 4); xw[0] = 1.0f;
    CHECK(omega_gpu_bf16_gemv_f32(tb, xw, yw, NULL) == 0 && omega_gpu_fp8_gemv_f32(t8, 0, xw, yw, NULL) == 0, "warm-up launches");
    free(xw); free(yw);
    XKern kb, ka0, ka1;
    /* cmdlist_create takes the session lock itself: create the lists before holding it (it is not recursive) */
    OmegaGpuCmdList *clF = NULL, *clT = NULL;
    int cr = omega_gpu_cmdlist_create(16, &clF) | omega_gpu_cmdlist_create(16, &clT);
    omega_gpu_session_lock();
    int kr = xkern(&kb, OMEGA_DEC_XBOUND, 0, mutant) | xkern(&ka0, OMEGA_DEC_ARGMAX, 0, 0) | xkern(&ka1, OMEGA_DEC_ARGMAX, 1, 0);
    /* one buffer: h | h8 | s | coef | hi | lo (P) | y (P) | yf (P) | amv | ami | tau | tok | tokf | tab */
    const size_t o_h = 0, o_h8 = o_h + k * 4, o_s = o_h8 + k * 4, o_c = o_s + (size_t)n * 4, o_hi = o_c + (size_t)n * 4, o_lo = o_hi + (size_t)n * 4,
                 o_y = o_lo + (size_t)P * 4, o_yf = o_y + (size_t)P * 4, o_amv = o_yf + (size_t)P * 4, o_ami = o_amv + 8192, o_tau = o_ami + 8192,
                 o_tok = o_tau + 16, o_tokf = o_tok + 16, o_tab = o_tokf + 16, total = o_tab + 64;
    OmegaGpuMem buf; memset(&buf, 0, sizeof buf);
    int ar = omega_gpu_session_alloc(total, &buf);
    CHECK(kr == 0 && ar == 0 && cr == 0, "kernels, buffer (%zu bytes) and command lists", total);
    if (kr || ar || cr) { omega_gpu_session_unlock(); return; }
    uint8_t *B = buf.cpu; const uint64_t V = buf.va;
    memcpy(B + o_c, coef, (size_t)n * 4); free(coef);
    for (uint32_t i = n; i < P; i++) { ((float *)(B + o_lo))[i] = NEG_PAD; ((float *)(B + o_y))[i] = NEG_PAD; ((float *)(B + o_yf))[i] = NEG_PAD; }
    for (uint32_t i = 0; i < 2048; i++) ((float *)(B + o_amv))[i] = NEG_PAD;
    uint64_t wva = 0, sva = 0; omega_gpu_fp8_tensor_entry(t8, 0, &wva, &sva);
    memcpy(B + o_tab, &wva, 8); memcpy(B + o_tab + 8, &sva, 8);
    uint64_t *tf = malloc(nrec * 8), *tt = malloc(nrec * 8);
    size_t same = 0, host_same = 0, bits_bad = 0, missing = 0, csum = 0, cmax = 0, sub_fail = 0, *cs = malloc(nrec * sizeof *cs);
    OmegaGpuCmdListOpts opts; memset(&opts, 0, sizeof opts);
    for (size_t r = 0; r < nrec; r++) {
        const uint8_t *rec = recs + r * rec_bytes; uint32_t host_full; memcpy(&host_full, rec + 12, 4);
        float *h = (float *)(B + o_h), *h8 = (float *)(B + o_h8);
        memcpy(h, rec + 16, (size_t)k * 4);
        for (uint32_t i = 0; i < k; i++) h8[i] = h[i] * 0x1p64f; /* exact: a power of two */
        omega_gpu_cmdlist_reset(clF); omega_gpu_cmdlist_reset(clT);
        int e = omega_gpu_session_sync_to_gpu(&buf);
        e |= omega_gpu_bf16_gemv_record(clF, tb, V + o_h, V + o_yf);
        e |= xargmax(clF, &ka0, &ka1, ctas, V + o_yf, V + o_amv, V + o_ami, V + o_tokf + 4, V + o_tokf);
        e |= omega_gpu_fp8_gemv_grouped_record(clT, t8, 1, V + o_tab, V + o_h8, 1, V + o_s);
        uint32_t xb[OMEGA_DEC_XB_WORDS]; memset(xb, 0, sizeof xb);
        put64v(xb, OMEGA_DEC_XB_S, V + o_s); put64v(xb, OMEGA_DEC_XB_C, V + o_c); put64v(xb, OMEGA_DEC_XB_H, V + o_h);
        put64v(xb, OMEGA_DEC_XB_LO, V + o_lo); put64v(xb, OMEGA_DEC_XB_HI, V + o_hi); xb[(OMEGA_DEC_XB_N - 0x380u) / 4] = n;
        e |= xadd(clT, &kb, (n + OMEGA_DEC_THREADS - 1) / OMEGA_DEC_THREADS, n, xb, OMEGA_DEC_XB_WORDS);
        e |= xargmax(clT, &ka0, &ka1, ctas, V + o_lo, V + o_amv, V + o_ami, V + o_tau, V + o_tau + 4);
        e |= omega_gpu_bf16_gemv_mask_record(clT, tb, V + o_h, V + o_y, V + o_hi, V + o_tau);
        e |= xargmax(clT, &ka0, &ka1, ctas, V + o_y, V + o_amv, V + o_ami, V + o_tok + 4, V + o_tok);
        OmegaGpuCmdListStats sf, st; memset(&sf, 0, sizeof sf); memset(&st, 0, sizeof st);
        if (!e) e = omega_gpu_cmdlist_submit(clF, &opts, &sf);
        if (!e) e = omega_gpu_cmdlist_submit(clT, &opts, &st);
        if (!e) e = omega_gpu_session_sync_to_cpu(&buf);
        if (e) { sub_fail++; printf("record %zu: chip error: %s\n", r, omega_gpu_session_last_error()); break; }
        uint32_t tok, tokf; memcpy(&tok, B + o_tok, 4); memcpy(&tokf, B + o_tokf, 4);
        const float *y = (const float *)(B + o_y), *yf = (const float *)(B + o_yf);
        size_t c = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (y[i] == -INFINITY) continue;
            c++; bits_bad += memcmp(&y[i], &yf[i], 4) != 0;
        }
        missing += tokf < n && y[tokf] == -INFINITY;
        same += tok == tokf; host_same += tokf == host_full;
        cs[r] = c; csum += c; if (c > cmax) cmax = c;
        tf[r] = sf.elapsed_ns; tt[r] = st.elapsed_ns;
        if (tok != tokf && mutant == 0) printf("record %zu: two-pass %u != full %u (|C| %zu)\n", r, tok, tokf, c);
    }
    omega_gpu_session_unlock();
    const size_t done = nrec - sub_fail;
    if (mutant == 0) {
        CHECK(sub_fail == 0 && same == done, "two-pass token == full head token on %zu / %zu records", same, done);
        CHECK(bits_bad == 0 && missing == 0, "candidate logits bit-identical to the full head (%zu differ, %zu full argmax missing)", bits_bad, missing);
    } else CHECK(sub_fail == 0 && same < done, "mutant %d must lose a token: %zu / %zu identical", mutant, same, done);
    double cmean = done ? (double)csum / done : 0, c99 = done ? (double)pct(cs, done, 0.99) : 0;
    double f50 = pct(tf, done, 0.5) / 1e3, t50 = pct(tt, done, 0.5) / 1e3, f99 = pct(tf, done, 0.99) / 1e3, t99 = pct(tt, done, 0.99) / 1e3;
    printf("chip: %zu records, identical %zu, host-full agrees %zu, |C| mean %.1f p99 %.0f max %zu\n", done, same, host_same, cmean, c99, cmax);
    printf("chip: full head list p50 %.1f us p99 %.1f us; two-pass list p50 %.1f us p99 %.1f us\n", f50, f99, t50, t99);
    if (out_path) {
        FILE *o = fopen(out_path, "w");
        if (o) {
            fprintf(o, "{\n  \"test\": \"gpu_exact_head_test\", \"mutant\": %d, \"records\": %zu, \"identical\": %zu, \"host_full_agrees\": %zu,\n", mutant, done, same, host_same);
            fprintf(o, "  \"candidate_bits_differ\": %zu, \"full_argmax_missing\": %zu, \"C_mean\": %.3f, \"C_p99\": %.0f, \"C_max\": %zu,\n", bits_bad, missing, cmean, c99, cmax);
            fprintf(o, "  \"full_list_us\": { \"p50\": %.2f, \"p99\": %.2f }, \"two_pass_list_us\": { \"p50\": %.2f, \"p99\": %.2f },\n", f50, f99, t50, t99);
            fprintf(o, "  \"bytes\": { \"full_head\": %" PRIu64 ", \"fp8b_screen\": %" PRIu64 ", \"coef\": %u, \"row\": %u }\n}\n",
                    (uint64_t)n * k * 2, (uint64_t)n * k + (uint64_t)(n / 128) * (k / 128) * 2, n * 4, k * 2);
            fclose(o);
        }
    }
    free(recs); free(tf); free(tt); free(cs);
}

int main(int argc, char **argv) {
    int sim = 0, mutant = 0; uint32_t limit = 0; const char *dir = NULL, *hidden = NULL, *out = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--chip") == 0 && i + 2 < argc) { dir = argv[++i]; hidden = argv[++i]; }
        else if (strcmp(argv[i], "--limit") == 0 && i + 1 < argc) limit = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--mutant") == 0 && i + 1 < argc) mutant = atoi(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out = argv[++i];
        else { fprintf(stderr, "usage: gpu_exact_head_test --sim | --chip <export-dir> <hidden.bin> [--limit N] [--mutant 1] [--out receipt.json]\n"); return 2; }
    }
    if (sim) sim_battery();
    if (dir) chip(dir, hidden, limit, mutant, out);
    printf("gpu_exact_head_test: %u checks, %u failed: %s\n", g_checks, g_failed, g_failed || !g_checks ? "FAIL" : "PASS");
    return g_failed || !g_checks ? 1 : 0;
}
