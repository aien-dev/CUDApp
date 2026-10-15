/* gpu_pfgemm_test.c (OWN-30B T3b, 2026-10-13): tensor-core grouped FP8 GEMM v2, device resident (omega_blackwell_codegen_tc2.h,
 * omega_gpu_pfgemm.c).
 *   --host-only  register budgets of the production specs; the SWIGLU layout transform against a double oracle on the
 *                original (unpermuted) weights; the C emulation of the folded scale against a double oracle
 *   --sim        the kernel in the warp simulator (both fragment orders) bit-identical to the C emulation (F32 and SWIGLU,
 *                indexed and direct rows); mutants 1-4 caught (a simulator error is a FAIL, never "caught")
 *   --chip       (under flock /tmp/aien-gb10.lock) through the device API with GPU-cached buffers: sampled rows vs the
 *                emulation (F32 bit-identical; SWIGLU within 1 bf16 ulp: MUFU.EX2/RCP are approximate on the chip), second
 *                launch identical, mutants caught, medians of --reps at --tokens for every production shape
 * Verdict line last: "GPU_PFGEMM: PASS|FAIL". */
#include "omega_blackwell_codegen_tc2.h"
#include "omega_blackwell_codegen_tc.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_prefill.h"
#include "bw_warp_sim_tc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const HmmaModel GB10 = { 16, 2, 0, 2, 0 };
static uint64_t rng = 0x243f6a8885a308d3ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float frnd(void) { return ((float)(rnd() & 0xffffff) / 16777216.0f) * 2.0f - 1.0f; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint16_t bf16_rn(float f) { uint32_t u = f2u(f); if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u; return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
static uint32_t stored_ix(uint32_t p) { static const uint32_t inv[4] = { 0, 2, 1, 3 }; return (p & ~3u) | inv[p & 3u]; }

typedef struct {
    uint32_t E, n, k, M, xr, layout, direct;
    uint8_t *w, *pw;   /* original and physical weights */
    uint16_t *sbf;     /* scale_inv bf16 */
    float *ratio;      /* physical ratio table */
    float *x;          /* xr * k true values */
    uint16_t *xp;      /* (xr + pad) * k stored bf16 */
    uint32_t *eid, *src, *counts;
} Prob;

static uint8_t rand_e4m3(void) { for (;;) { uint8_t c = (uint8_t)rnd(); if ((c & 0x7f) != 0x7f) return c; } }
static void prob_make(Prob *p, uint32_t E, uint32_t n, uint32_t k, uint32_t M, uint32_t xr, uint32_t layout, uint32_t direct, uint32_t pad) {
    memset(p, 0, sizeof *p);
    p->E = E; p->n = n; p->k = k; p->M = M; p->xr = direct ? M : xr; p->layout = layout; p->direct = direct;
    const size_t wn = (size_t)E * n * k, sn = (size_t)E * (n / 128) * (k / 128), rn = (size_t)E * (n / 8) * (k / 128 + 1);
    p->w = malloc(wn); p->pw = malloc(wn); p->sbf = malloc(sn * 2); p->ratio = malloc(rn * 4);
    p->x = malloc((size_t)p->xr * k * 4); p->xp = calloc((size_t)(p->xr + pad) * k, 2);
    p->eid = malloc(M * 4); p->src = malloc(M * 4); p->counts = calloc(E, 4);
    for (size_t i = 0; i < wn; i++) p->w[i] = rand_e4m3();
    for (size_t i = 0; i < sn; i++) { float s = ldexpf(1.0f + (float)(rnd() & 127) / 128.0f, -(int)(rnd() % 12) - 4); p->sbf[i] = (uint16_t)(f2u(s) >> 16); }
    if (omega_gpu_pf_weights_host_form(E, n, k, p->w, p->sbf, layout, p->pw, p->ratio) != 0) { printf("host form refused\n"); exit(2); }
    for (size_t i = 0; i < (size_t)p->xr * k; i++) p->x[i] = frnd() * (rnd() % 16 == 0 ? 8.0f : 1.0f);
    for (uint32_t r = 0; r < p->xr; r++) omega_tcgemm_permute_x(p->x + (size_t)r * k, k, p->xp + (size_t)r * k);
    /* sorted expert ids with random segment lengths (some experts empty, some multi-tile) */
    uint32_t i = 0;
    for (uint32_t e = 0; e < E && i < M; e++) { uint32_t c = e + 1 == E ? M - i : (uint32_t)((uint64_t)(rnd() % 2000) * 2 * M / (2000ull * E)); if (c > M - i) c = M - i; for (uint32_t q = 0; q < c; q++) { p->eid[i + q] = e; } p->counts[e] = c; i += c; }
    for (uint32_t q = 0; q < M; q++) p->src[q] = direct ? q : rnd() % p->xr;
}
static void prob_free(Prob *p) { free(p->w); free(p->pw); free(p->sbf); free(p->ratio); free(p->x); free(p->xp); free(p->eid); free(p->src); free(p->counts); }

/* emulation of one physical output (row i, physical col c): the kernel's arithmetic */
static float emu_phys(const Prob *p, uint32_t i, uint32_t c) {
    const uint32_t n = p->n, k = p->k, KB = k / 128;
    const uint16_t *xs = p->xp + (size_t)p->src[i] * k;
    const uint8_t *wr = p->pw + ((size_t)p->eid[i] * n + c) * k;
    const float *R0 = p->ratio + ((size_t)p->eid[i] * (n / 16) + c / 16) * (KB + 1) * 2 + (c / 8) % 2;
#define R(q) R0[2 * (q)]
    float acc = 0.0f;
    for (uint32_t kb = 0; kb < KB; kb++) {
        acc = acc * R(kb);
        for (uint32_t ch = 0; ch < 2; ch++) for (uint32_t s = 0; s < 4; s++) {
            float a[16], b[16]; int m = 0;
            for (uint32_t t = 0; t < 4; t++) for (uint32_t j = 0; j < 4; j++) {
                uint32_t e = kb * 128 + ch * 64 + 16 * t + 4 * s + j;
                a[m] = u2f((uint32_t)xs[stored_ix(e)] << 16); b[m] = u2f((uint32_t)omega_tcgemm_w_bf16(wr[e]) << 16); m++;
            }
            acc = hm_dot16(&GB10, a, b, acc);
        }
    }
    return acc * R(KB);
#undef R
}
/* the kernel's epilogue arithmetic: MUFU.EX2, FADD, MUFU.RCP, FMUL x 3, F2FP. The two MUFU results pass through ws_mufu_ftz
 * (bw_warp_sim_tc.h): the chip's MUFU flushes a subnormal result to a sign-preserving zero, so for t = -g log2 e in (126, 128)
 * (g in (-88.72, -87.34)) the reciprocal is +-0 and the activation is +-0, not the 2^-56-ish product an IEEE 1/d would give. */
static uint16_t swiglu_bits(float g, float u) {
    float t = g * u2f(0xbfb8aa3bu), ex = ws_mufu_ftz(exp2f(ws_mufu_ftz(t))), d = ex + 1.0f, r = ws_mufu_ftz(1.0f / ws_mufu_ftz(d)), o = g * r;
    o = o * u; o = o * u2f(0x5f800000u);
    return bf16_rn(o);
}
/* emulated output row i: F32 -> n floats; SWIGLU -> n/2 bf16 (as u16 in the low half of each u32 slot of out) */
static void emu_row(const Prob *p, uint32_t i, float *yf, uint16_t *ya) {
    if (p->layout == OMEGA_GPU_PF_PLAIN) { for (uint32_t c = 0; c < p->n; c++) yf[c] = emu_phys(p, i, c); return; }
    for (uint32_t a = 0; a < p->n / 2; a++) { uint32_t G = a / 8, m = a % 8; ya[a] = swiglu_bits(emu_phys(p, i, 16 * G + m), emu_phys(p, i, 16 * G + 8 + m)); }
}

typedef struct { uint64_t w_va, r_va; uint32_t row0, cnt; uint64_t pad; } Tile;
static uint32_t make_tiles(const Prob *p, uint32_t bm, Tile *t) {
    uint32_t nt = 0, row = 0; const uint32_t KB = p->k / 128;
    for (uint32_t e = 0; e < p->E; e++) { for (uint32_t r = 0; r < p->counts[e]; r += bm) {
        t[nt].w_va = (uintptr_t)(p->pw + (size_t)e * p->n * p->k); t[nt].r_va = (uintptr_t)(p->ratio + (size_t)e * (p->n / 8) * (KB + 1));
        t[nt].row0 = row + r; t[nt].cnt = p->counts[e] - r < bm ? p->counts[e] - r : bm; t[nt].pad = 0; nt++; } row += p->counts[e]; }
    return nt;
}

static int host_only(void) {
    int fail = 0;
    static const OmegaTc2Spec specs[] = {
        { 2048, 8, 2, 16, 1, OMEGA_TC2_EPI_F32, 0, 5120, 0 }, { 2048, 8, 2, 16, 1, OMEGA_TC2_EPI_SWIGLU, 0, 1536, 0 }, { 768, 8, 2, 16, 1, OMEGA_TC2_EPI_F32, 1, 2048, 0 },
        { 4096, 8, 2, 16, 1, OMEGA_TC2_EPI_F32, 0, 2048, 0 }, { 2048, 8, 2, 16, 2, OMEGA_TC2_EPI_F32, 0, 5120, 0 }, { 2048, 8, 2, 16, 2, OMEGA_TC2_EPI_SWIGLU, 0, 1536, 0 },
        { 768, 8, 2, 16, 2, OMEGA_TC2_EPI_F32, 1, 2048, 0 }, { 2048, 4, 2, 8, 2, OMEGA_TC2_EPI_F32, 0, 2048, 0 },
    };
    static BlackwellIRProgram prog;
    for (size_t i = 0; i < sizeof specs / sizeof specs[0]; i++) {
        int rc = omega_blackwell_codegen_tc2_prog(&specs[i], &prog);
        uint32_t peak = rc == 0 ? prog.regalloc.peak_gpr_usage : 0, decl = omega_bw_gpr_declared(peak);
        int ok = rc == 0 && decl <= 128u && 32u * specs[i].warps * decl <= 65536u;
        printf("spec k %u rt %u ct %u warps %u aqn %u epi %u direct %u: %s, %zu instructions, peak %u, declared %u -> %s\n", specs[i].k, specs[i].rt, specs[i].ct, specs[i].warps, specs[i].aqn,
               specs[i].epi, specs[i].direct, rc ? "REFUSED" : "built", rc ? 0 : prog.count, peak, decl, ok ? "fits" : "DOES NOT FIT");
        if (!ok && specs[i].aqn == 1) fail++; /* aqn 2 is optional */
    }
    /* layout + emulation vs double oracle on the ORIGINAL weights: SWIGLU act[a] ~ silu(x.Wg[i]) * x.Wu[i] * 2^64, i = perm(a) */
    Prob p; prob_make(&p, 2, 256, 256, 24, 8, OMEGA_GPU_PF_SWIGLU, 0, 32);
    double worst = 0; const uint32_t KB = p.k / 128, NB = p.n / 128, I = p.n / 2;
    for (uint32_t i = 0; i < p.M; i++) {
        uint16_t act[128]; emu_row(&p, i, NULL, act);
        for (uint32_t a = 0; a < I; a++) {
            uint32_t li = stored_ix(a); double d[2];
            for (int h = 0; h < 2; h++) { uint32_t row = h ? I + li : li; double s = 0;
                for (uint32_t kk = 0; kk < p.k; kk++) {
                    double sc = (double)u2f((uint32_t)p.sbf[((size_t)p.eid[i] * NB + row / 128) * KB + kk / 128] << 16);
                    double wv = ldexp((double)u2f((uint32_t)omega_tcgemm_w_bf16(p.w[((size_t)p.eid[i] * p.n + row) * p.k + kk]) << 16), 120);
                    double xv = (double)u2f((uint32_t)p.xp[(size_t)p.src[i] * p.k + stored_ix(kk)] << 16) / 18446744073709551616.0;
                    s += sc * wv * xv; }
                d[h] = s; }
            double want = d[0] / (1.0 + exp(-d[0])) * d[1], got = (double)u2f((uint32_t)act[a] << 16) / 18446744073709551616.0;
            double err = fabs(got - want) / (fabs(want) + 1e-3 * 1e-0);
            if (err > worst) worst = err;
        }
    }
    printf("SWIGLU layout + emulation vs double oracle on the original weights: worst relative error %.3g (bf16 output: < 1e-2 expected)\n", worst);
    if (!(worst < 1e-2)) fail++;
    prob_free(&p);
    /* the reference against the chip's MUFU.RCP flush (GB10, evidence OWN30B-L1G-PFGEMM-SWIGLU-FTZ-df9bd9c4, 2026-10-15): at 3600
     * tokens, 48 sampled rows, all 101 of 101 elements that differed from the IEEE reference had t = -g log2 e in (126.03, 127.99)
     * and came back +-0 from the chip. g and u are the emulated f32 gate / up values (9 significant digits round-trip exactly),
     * act the chip's bf16; the first and last entries are the smallest and largest t observed. */
    static const struct { float g, u; uint16_t act; } chip[] = {
        { -87.3551483f, 274.485016f, 0x8000 }, { -87.4824142f, 122.849442f, 0x8000 }, { -87.8418655f, 45.4098854f, 0x8000 },
        { -88.1839066f, -28.427494f, 0x0000 }, { -88.3344727f, -3.59300494f, 0x0000 }, { -88.7186356f, 180.687149f, 0x8000 },
    };
    size_t ok = 0;
    for (size_t i = 0; i < sizeof chip / sizeof chip[0]; i++) ok += swiglu_bits(chip[i].g, chip[i].u) == chip[i].act;
    /* negative control: g -87 (t 125.5) has a normal reciprocal and must stay a nonzero finite negative activation (the flush must not
     * reach outside the window); g -89 (t 128.4) overflows ex2 to +inf and is -0 in both arithmetics (not a flush) */
    const uint16_t nc_lo = swiglu_bits(-87.0f, 100.0f), nc_hi = swiglu_bits(-89.0f, 100.0f);
    const int nc_ok = (nc_lo & 0x7fffu) != 0 && (nc_lo & 0x7f80u) != 0x7f80u && (nc_lo & 0x8000u) && nc_hi == 0x8000u;
    printf("SWIGLU reference vs chip-observed MUFU.RCP flush window: %zu/%zu match; control g -87 -> %04x (nonzero), g -89 -> %04x (-0): %s\n",
           ok, sizeof chip / sizeof chip[0], nc_lo, nc_hi, ok == sizeof chip / sizeof chip[0] && nc_ok ? "ok" : "FAIL");
    if (ok != sizeof chip / sizeof chip[0] || !nc_ok) fail++;
    return fail;
}

static int sim_run(const Prob *p, const OmegaTc2Spec *sp, int order, void *y, char *msg, size_t msgn) {
    static BlackwellIRProgram prog;
    if (omega_blackwell_codegen_tc2_prog(sp, &prog) != 0) { snprintf(msg, msgn, "codegen refused (spec_ok %d)", omega_tc2_spec_ok(sp)); return -1; }
    const uint32_t bm = 16u * sp->rt, span = omega_tc2_span(sp);
    Tile *tiles = calloc(p->M + 1, sizeof *tiles);
    uint32_t nt = make_tiles(p, bm, tiles);
    uint32_t *ridx = malloc((p->M + bm) * 4); for (uint32_t i = 0; i < p->M + bm; i++) ridx[i] = i < p->M ? p->src[i] : 0;
    const size_t yb = (size_t)(p->M + bm) * p->n * (p->layout == OMEGA_GPU_PF_SWIGLU ? 1 : 4);
    memset(y, 0xee, yb);
    uint8_t *smem = malloc(omega_tc2_shared_bytes(sp));
    static WsSim s; memset(&s, 0, sizeof s);
    s.p = &prog; s.threads = 32u * sp->warps; s.smem = smem; s.smem_bytes = omega_tc2_shared_bytes(sp); s.order = order;
    uint64_t a[4] = { (uintptr_t)tiles, sp->direct ? 0 : (uintptr_t)ridx, (uintptr_t)p->xp, (uintptr_t)y };
    for (int i = 0; i < 4; i++) { s.cbank[(0x380 + 8 * i) / 4] = (uint32_t)a[i]; s.cbank[(0x380 + 8 * i) / 4 + 1] = (uint32_t)(a[i] >> 32); }
    s.cbank[0x3a0 / 4] = p->n;
    const uint32_t KB = p->k / 128;
    s.ranges[0].lo = (uintptr_t)tiles; s.ranges[0].hi = (uintptr_t)(tiles + nt);
    s.ranges[1].lo = (uintptr_t)ridx; s.ranges[1].hi = (uintptr_t)(ridx + p->M + bm);
    s.ranges[2].lo = (uintptr_t)p->xp; s.ranges[2].hi = (uintptr_t)(p->xp + (size_t)(p->xr + (p->direct ? bm : 0)) * p->k);
    s.ranges[3].lo = (uintptr_t)y; s.ranges[3].hi = (uintptr_t)y + (size_t)p->M * p->n * (p->layout == OMEGA_GPU_PF_SWIGLU ? 1 : 4);
    s.ranges[4].lo = (uintptr_t)p->pw; s.ranges[4].hi = (uintptr_t)(p->pw + (size_t)p->E * p->n * p->k);
    s.ranges[5].lo = (uintptr_t)p->ratio; s.ranges[5].hi = (uintptr_t)(p->ratio + (size_t)p->E * (p->n / 8) * (KB + 1));
    s.nranges = 6;
    int err = ws_run(&s, p->n / span, nt);
    if (err) snprintf(msg, msgn, "simulator %s", s.msg);
    free(tiles); free(ridx); free(smem);
    return err ? -1 : 0;
}
static size_t compare(const Prob *p, const void *y, const uint32_t *rows, uint32_t nrows, int tol_ulp, size_t *ulp1) {
    size_t bad = 0; if (ulp1) *ulp1 = 0;
    float *yf = malloc(p->n * 4); uint16_t *ya = malloc(p->n);
    for (uint32_t q = 0; q < nrows; q++) {
        uint32_t i = rows[q]; emu_row(p, i, yf, ya);
        if (p->layout == OMEGA_GPU_PF_PLAIN) { const uint32_t *g = (const uint32_t *)y + (size_t)i * p->n; for (uint32_t c = 0; c < p->n; c++) if (g[c] != f2u(yf[c])) { if (getenv("PFDBG") && bad < 6) printf("  row %u col %u got %.9g want %.9g\n", i, c, u2f(g[c]), yf[c]); bad++; } }
        else { const uint16_t *g = (const uint16_t *)y + (size_t)i * (p->n / 2);
            for (uint32_t c = 0; c < p->n / 2; c++) if (g[c] != ya[c]) { int d = abs((int)g[c] - (int)ya[c]); if (d <= tol_ulp && ((g[c] ^ ya[c]) & 0x8000) == 0) { if (ulp1) (*ulp1)++; }
                else { if (getenv("PFDBG") && bad < 128) { const uint32_t G = c / 8, mm = c % 8; const float gg = emu_phys(p, i, 16 * G + mm), uu = emu_phys(p, i, 16 * G + 8 + mm);
                                                             printf("  row %u act %u got %04x want %04x g %.9g u %.9g t %.6f\n", i, c, g[c], ya[c], gg, uu, gg * u2f(0xbfb8aa3bu)); } bad++; } } }
    }
    free(yf); free(ya);
    return bad;
}

static int sim_tests(void) {
    int fail = 0; char msg[256];
    static const struct { uint32_t E, n, k, M, xr, layout, direct, rt, warps, aqn; } cs[] = {
        { 3, 256, 256, 70, 20, OMEGA_GPU_PF_PLAIN, 0, 8, 16, 1 }, { 3, 256, 384, 70, 20, OMEGA_GPU_PF_SWIGLU, 0, 8, 16, 1 },
        { 3, 256, 256, 150, 0, OMEGA_GPU_PF_PLAIN, 1, 8, 16, 1 }, { 2, 256, 256, 40, 9, OMEGA_GPU_PF_PLAIN, 0, 4, 8, 2 },
        { 2, 512, 128, 300, 30, OMEGA_GPU_PF_SWIGLU, 0, 8, 16, 2 },
        { 3, 512, 256, 150, 40, OMEGA_GPU_PF_SWIGLU, 0, 4, 16, 1 }, { 3, 256, 384, 130, 0, OMEGA_GPU_PF_PLAIN, 1, 4, 16, 1 }, /* the chip shape (QMD shared cap) */
    };
    for (size_t c = 0; c < sizeof cs / sizeof cs[0]; c++) {
        Prob p; prob_make(&p, cs[c].E, cs[c].n, cs[c].k, cs[c].M, cs[c].xr, cs[c].layout, cs[c].direct, 16 * cs[c].rt);
        OmegaTc2Spec sp = { cs[c].k, cs[c].rt, 2, cs[c].warps, cs[c].aqn, cs[c].layout == OMEGA_GPU_PF_SWIGLU ? OMEGA_TC2_EPI_SWIGLU : OMEGA_TC2_EPI_F32, cs[c].direct, cs[c].n, 0 };
        void *y = malloc((size_t)(p.M + 128) * p.n * 4);
        uint32_t *rows = malloc(p.M * 4); for (uint32_t i = 0; i < p.M; i++) rows[i] = i;
        for (int order = 0; order < 2; order++) {
            int rc = sim_run(&p, &sp, order, y, msg, sizeof msg);
            size_t bad = rc ? (size_t)-1 : compare(&p, y, rows, p.M, 0, NULL);
            printf("sim case %zu (E %u n %u k %u M %u %s%s rt %u warps %u aqn %u) order %d: %s, %zu differ\n", c, p.E, p.n, p.k, p.M, p.layout ? "SWIGLU" : "F32", p.direct ? " direct" : "", sp.rt, sp.warps, sp.aqn, order, rc ? msg : "ran", rc ? 0 : bad);
            if (rc || bad) fail++;
        }
        if (c == 0 || c == 1) for (int m = 1; m <= 4; m++) {
            OmegaTc2Spec mm = sp; mm.mutant = m;
            int rc = sim_run(&p, &mm, 0, y, msg, sizeof msg);
            size_t bad = rc ? 0 : compare(&p, y, rows, p.M, 0, NULL);
            printf("sim case %zu mutant %d: %s\n", c, m, rc ? msg : bad ? "caught" : "NOT CAUGHT");
            if (rc || !bad) fail++;
        }
        free(y); free(rows); prob_free(&p);
    }
    return fail;
}

static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }

static int chip_shape(const char *name, uint32_t tokens, int reps, int check_rows, int mutants) {
    uint32_t E, n, k, M, xr, layout, direct;
    if (!strcmp(name, "gateup")) { E = 128; n = 1536; k = 2048; M = tokens * 8; xr = tokens; layout = OMEGA_GPU_PF_SWIGLU; direct = 0; }
    else if (!strcmp(name, "down")) { E = 128; n = 2048; k = 768; M = tokens * 8; xr = 0; layout = OMEGA_GPU_PF_PLAIN; direct = 1; }
    else if (!strcmp(name, "qkv")) { E = 1; n = 5120; k = 2048; M = tokens; xr = tokens; layout = OMEGA_GPU_PF_PLAIN; direct = 0; }
    else if (!strcmp(name, "o")) { E = 1; n = 2048; k = 4096; M = tokens; xr = tokens; layout = OMEGA_GPU_PF_PLAIN; direct = 0; }
    else { printf("unknown shape %s\n", name); return 1; }
    const uint32_t bm = omega_gpu_pf_gemm_bm();
    Prob p; prob_make(&p, E, n, k, M, xr, layout, direct, bm);
    int fail = 0, rc;
    OmegaPfWeights wt; OmegaGpuMem xd = { 0 }, yd = { 0 }, tab = { 0 }, ridx = { 0 };
    rc = omega_gpu_pf_weights_upload(E, n, k, p.w, p.sbf, layout, &wt);
    const size_t xb = ((size_t)(p.xr + bm) * k * 2 + 15) & ~(size_t)15, yb = (size_t)(M + bm) * n * (layout == OMEGA_GPU_PF_SWIGLU ? 1 : 4);
    if (!rc) rc = omega_gpu_pf_alloc_dev(xb, &xd);
    if (!rc) rc = omega_gpu_pf_alloc_dev(yb, &yd);
    if (!rc) rc = omega_gpu_pf_upload(&xd, 0, p.xp, xb);
    const uint32_t maxt = M / bm + E + 1;
    if (!rc) rc = omega_gpu_pf_alloc_host((size_t)maxt * 32, &tab);
    if (!rc) rc = omega_gpu_pf_alloc_host((size_t)(M + bm) * 4, &ridx);
    if (rc) { printf("%s setup: %s (%s)\n", name, omega_gpu_pf_rc_name(rc), omega_gpu_session_last_error()); return 1; }
    uint32_t nt = omega_gpu_pf_gemm_tiles(&wt, p.counts, tab.cpu, maxt);
    { uint32_t *ri = ridx.cpu; for (uint32_t i = 0; i < M + bm; i++) ri[i] = i < M ? p.src[i] : 0; }
    omega_gpu_session_lock(); (void)omega_gpu_session_sync_to_gpu(&tab); (void)omega_gpu_session_sync_to_gpu(&ridx); omega_gpu_session_unlock();
    const uint64_t rva = direct ? 0 : ridx.va;
    uint8_t *y1 = malloc(yb), *y2 = malloc(yb);
    uint64_t ns[32] = { 0 };
    if (reps > 32) reps = 32;
    for (int r = 0; r < reps && !rc; r++) rc = omega_gpu_pf_gemm(&wt, tab.va, nt, rva, xd.va, yd.va, &ns[r]);
    if (!rc) rc = omega_gpu_pf_download(&yd, 0, y1, yb & ~(size_t)15);
    memset(y2, 0, yb);
    { uint8_t z[4096]; memset(z, 0, sizeof z); /* zero the output on the device, run once more: the second result must be identical */
      uint8_t *zb = calloc(1, yb); if (!rc) rc = omega_gpu_pf_upload(&yd, 0, zb, yb & ~(size_t)15); free(zb); (void)z; }
    uint64_t ns2 = 0;
    if (!rc) rc = omega_gpu_pf_gemm(&wt, tab.va, nt, rva, xd.va, yd.va, &ns2);
    if (!rc) rc = omega_gpu_pf_download(&yd, 0, y2, yb & ~(size_t)15);
    if (rc) { printf("%s run: %s (%s)\n", name, omega_gpu_pf_rc_name(rc), omega_gpu_session_last_error()); fail++; }
    else {
        const size_t outb = (size_t)M * n * (layout == OMEGA_GPU_PF_SWIGLU ? 1 : 4);
        printf("%s second launch identical: %s\n", name, memcmp(y1, y2, outb) ? "NO" : "yes");
        if (memcmp(y1, y2, outb)) fail++;
        uint32_t *rows = malloc(check_rows * 4); for (int q = 0; q < check_rows; q++) rows[q] = q == 0 ? 0 : q == 1 ? M - 1 : rnd() % M;
        size_t ulp1 = 0, bad = compare(&p, y1, rows, check_rows, layout == OMEGA_GPU_PF_SWIGLU ? 1 : 0, &ulp1);
        printf("%s bit-identity on %d sampled rows: %zu differ%s", name, check_rows, bad, layout == OMEGA_GPU_PF_SWIGLU ? "" : "\n");
        if (layout == OMEGA_GPU_PF_SWIGLU) printf(" (beyond 1 bf16 ulp; %zu within 1 ulp of %zu)\n", ulp1, (size_t)check_rows * n / 2);
        if (bad) fail++;
        qsort(ns, reps, 8, cmp_u64);
        const double fl = 2.0 * M * n * k, med = (double)ns[reps / 2];
        printf("TCGEMM2 %s tokens %u M %u n %u k %u tiles %u: median %.3f ms (min %.3f), %.1f TFLOPS, weights %.0f MB -> %.0f GB/s of distinct weights\n", name, tokens, M, n, k, nt,
               med / 1e6, ns[0] / 1e6, fl / med / 1e3, (double)E * n * k / 1e6, (double)E * n * k / med);
        if (mutants) for (int m = 1; m <= 4; m++) {
            omega_gpu_pf_gemm_test_set_mutant(m);
            int mrc = omega_gpu_pf_gemm(&wt, tab.va, nt, rva, xd.va, yd.va, NULL);
            if (!mrc) mrc = omega_gpu_pf_download(&yd, 0, y2, yb & ~(size_t)15);
            omega_gpu_pf_gemm_test_set_mutant(0);
            size_t mb = mrc ? 0 : compare(&p, y2, rows, check_rows > 8 ? 8 : check_rows, layout == OMEGA_GPU_PF_SWIGLU ? 1 : 0, NULL);
            printf("%s mutant %d: %s\n", name, m, mrc ? "launch failed" : mb ? "caught" : "NOT CAUGHT");
            if (mrc || !mb) fail++;
        }
        free(rows);
    }
    free(y1); free(y2);
    omega_gpu_pf_weights_free(&wt); omega_gpu_pf_free(&xd); omega_gpu_pf_free(&yd); omega_gpu_pf_free(&tab); omega_gpu_pf_free(&ridx);
    prob_free(&p);
    return fail;
}

int main(int argc, char **argv) {
    int host = 1, sim = 1, chip = 0, reps = 7, check = 48, mutants = 1; uint32_t tokens = 3600; const char *shape = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) sim = 0;
        else if (!strcmp(argv[i], "--sim")) host = 0;
        else if (!strcmp(argv[i], "--chip")) { chip = 1; host = 0; sim = 0; }
        else if (!strcmp(argv[i], "--shape") && i + 1 < argc) shape = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--check-rows") && i + 1 < argc) check = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-mutants")) mutants = 0;
        else if (!strcmp(argv[i], "--shape-knob") && i + 3 < argc) { uint32_t a = (uint32_t)atoi(argv[i + 1]), b = (uint32_t)atoi(argv[i + 2]), c = (uint32_t)atoi(argv[i + 3]); omega_gpu_pf_gemm_set_shape(a, b, c); i += 3; }
    }
    int fail = 0;
    if (host) fail += host_only();
    if (sim) fail += sim_tests();
    if (chip) {
        static const char *all[] = { "gateup", "down", "qkv", "o" };
        for (int s = 0; s < 4; s++) if (!shape || !strcmp(shape, all[s])) fail += chip_shape(all[s], tokens, reps, check, mutants);
    }
    printf("GPU_PFGEMM: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
