/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* gpu_provenance_enc_test: chip experiments that pin sm_121 encoding bits without any NVIDIA tool
 * (docs/PROVENANCE-ENCODINGS.md). No nvcc, ptxas, nvdisasm or cuobjdump output is read or compared here.
 *
 *   ./gpu_provenance_enc_test --host-only        the goldens against values stated in the public specs
 *                                                (IEEE 754 binary32/bfloat16, OCP OFP8 E4M3), the word builders,
 *                                                and the comparator's own negative control. No device.
 *   ./gpu_provenance_enc_test --list             the case names.
 *   ./gpu_provenance_enc_test --case NAME        one chip experiment (caller holds flock /tmp/aien-gb10.lock).
 *                                                One case per process: a word the chip refuses latches the
 *                                                session, and the next case gets a fresh process.
 *
 * Scaffold. The vecadd kernel (src/omega_blackwell_encoder.c) loads a[i] into R2 and b[i] into R5, adds them
 * into R9 at slot 17 and stores R9 to c[i]. Slot 17 is replaced by the word under test (dst R9, sources R2
 * and R5); the store at slot 18 additionally waits on scoreboard 0, which a variable-latency word under test
 * writes. Every scaffold word is NAK-derived (docs/PROVENANCE-ENCODINGS.md rows), so only the slot-17 word is
 * under test. Each case has a golden written here from the public spec and a negative control: the same
 * word with the one field under test changed must disagree with the golden (red), or the case proves nothing.
 *
 * Reconvergence case (bssy73): every reconvergence probe of src/omega_gpu_elementwise_api.c is generated with
 * bit 73 cleared on each BSSY/BSYNC (NAK never sets it) and run beside the production kernel (bit 73 set) on
 * the same inputs; the outputs must be identical. The production kernels are gated against the reference
 * oracle by tests/gpu_reconv_test.c. Negative control: the comparator must flag one corrupted word. */
#include "omega_blackwell_encoder.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_qmd.h"
#include "omega_gpu_elementwise_api.h"
#include "omega_gpu_session.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_ELEM 8192u
#define THREADS 128u
#define SLOT_OP 17u
#define SLOT_STG 18u

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint64_t g_rng = 0x2545f4914f6cdd1dull;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 16); }

/* ------------------------------------------------------------------ goldens from public specs */

/* IEEE 754 binary32 -> bfloat16 (the upper half of binary32), round to nearest, ties to even. NaN stays NaN
 * (quiet bit forced). */
static uint16_t bf16_rne(float f) {
    uint32_t u = f2u(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
    uint32_t lsb = (u >> 16) & 1u;
    return (uint16_t)((u + 0x7fffu + lsb) >> 16);
}

/* OCP 8-bit Floating Point Specification (OFP8) rev 1.0, E4M3: bias 7, no infinities, S.1111.111 is NaN,
 * max finite S.1111.110 = 448, min subnormal 2^-9. Value of a code: */
static double e4m3_val(uint8_t c) {
    int s = c >> 7, e = (c >> 3) & 15, m = c & 7;
    double v = e == 0 ? ldexp((double)m, -9) : ldexp(1.0 + m / 8.0, e - 7);
    return s ? -v : v;
}
/* Saturating ("satfinite") conversion, round to nearest, ties to the even code: magnitudes past 448
 * (including infinity) become 448, NaN becomes the NaN code 0x7f. */
static uint8_t e4m3_rne_satfinite(float f) {
    if (isnan(f)) return 0x7f;
    uint8_t sign = signbit(f) ? 0x80 : 0x00;
    double a = fabs((double)f);
    if (a >= 448.0) return (uint8_t)(sign | 0x7e);
    uint8_t best = 0; double bd = INFINITY;
    for (uint8_t c = 0; c <= 0x7e; c++) {
        double d = fabs(e4m3_val(c) - a);
        if (d < bd || (d == bd && (c & 1) == 0)) { bd = d; best = c; }
    }
    return (uint8_t)(sign | best);
}

/* ------------------------------------------------------------------ cases */

typedef enum { G_I2F_S32, G_F2FP_BF16, G_F2FP_E4M3, G_EX2, G_LG2 } Golden;
typedef struct {
    const char *name;
    const char *field;   /* what the case pins */
    const char *source;  /* where the word under test came from */
    uint32_t w0, w1, w2; /* slot-17 word, dst R9, src a = R2 at 24..31, src b = R5 at 32..39 */
    uint32_t nc_w2_xor;  /* negative control: w2 ^ this */
    int var_latency;     /* writes through scoreboard 0 */
    Golden golden;
} Case;

#define W0(op) (0x00097000u | (op))                 /* opcode 0..11, predicate PT 12..15, dst R9 16..23 */
#define W0A(op) (W0(op) | (2u << 24))               /* src0 R2 at 24..31 */
static const Case CASES[] = {
    { "i2fp_s32", "I2FP.F32.S32 opcode 0x245 with the NAK I2F modifier layout (74 signed, 75..77 dst size, 84..86 src size)",
      "CUDApp src/omega_blackwell_codegen.c BW_IR_I2FP", W0(0x245), 5, 0x00201400u, 1u << 10, 0, G_I2F_S32 },
    { "i2f_s32_nak", "I2F.F32.S32 from NAK OpI2F: opcode 0x106 form 1 (0x306), 74 signed, 75..77=2, 84..86=2",
      "Mesa NAK sm70_encode.rs OpI2F (02f13375)", W0(0x306), 5, 0x00201400u, 1u << 10, 1, G_I2F_S32 },
    { "f2fp_bf16", "F2FP.BF16.F32.PACK_AB: opcode 0x23e (NAK OpF2FP) with bit 76 selecting bf16",
      "CUDApp src/omega_blackwell_codegen_tc.c omega_tc_patch_code", W0A(0x23e), 5, 0x000010ffu, 1u << 12, 0, G_F2FP_BF16 },
    { "f2fp_e4m3", "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C: opcode 0x23e, bits 79, 87, 90",
      "CUDApp src/omega_blackwell_codegen_qmma.c omega_qmma_patch_code", W0A(0x23e), 5, 0x048070ffu, 1u << 26, 0, G_F2FP_E4M3 },
    { "mufu_ex2", "MUFU.EX2: NAK OpMuFu opcode 0x108 form 1 (0x308), sub-op 2 at 74..80",
      "Mesa NAK sm70_encode.rs OpMuFu (02f13375); CUDApp BW_IR_MUFU_EX2", W0(0x308), 5, 2u << 10, (2u ^ 3u) << 10, 1, G_EX2 },
    { "mufu_lg2", "MUFU.LG2: sub-op 3 at 74..80",
      "Mesa NAK sm70_encode.rs OpMuFu (02f13375); omega-mv BW_IR_MUFU_LG2", W0(0x308), 5, 3u << 10, (3u ^ 2u) << 10, 1, G_LG2 },
};
#define NCASES (sizeof CASES / sizeof CASES[0])

static void gen_inputs(Golden g, uint32_t *a, uint32_t *b) {
    for (uint32_t i = 0; i < N_ELEM; i++) {
        uint32_t r = rnd(), k = i % 8;
        switch (g) {
        case G_I2F_S32: { /* every magnitude class: small, around 2^24 (rounding starts), large, negatives, extremes */
            static const uint32_t sp[] = { 0u, 1u, 0xffffffffu, 0x80000000u, 0x7fffffffu, 16777217u, 16777219u, 0xfeffffffu };
            uint32_t v = k == 0 ? sp[(i / 8) % 8] : k < 3 ? (r & 0xffffu) - 0x8000u : k < 5 ? (0x01000000u | (r & 0xffffffu)) * ((r >> 31) ? 1u : 0xffffffffu) : rnd() ^ (r << 16);
            a[i] = 0; b[i] = v; break; }
        case G_F2FP_BF16: case G_F2FP_E4M3: {
            float x, y;
            if (k == 0) { static const uint32_t sp[] = { 0u, 0x80000000u, 0x3f808000u, 0x3f818000u, 0x43e00000u, 0x43e80000u, 0x7f800000u, 0xff800000u, 0x3a800000u, 0x3b000000u, 0x00000001u, 0x7f7fffffu };
                x = u2f(sp[(i / 8) % 12]); y = u2f(sp[(i / 96) % 12]); }
            else if (g == G_F2FP_E4M3 && k < 4) { /* exact ties between neighbouring e4m3 codes */
                uint8_t c = (uint8_t)(r & 0x7f); if (c >= 0x7e) c = 0x30; x = (float)((e4m3_val(c) + e4m3_val((uint8_t)(c + 1))) / 2.0);
                uint8_t d = (uint8_t)((r >> 8) & 0x7f); if (d >= 0x7e) d = 0x21; y = -(float)((e4m3_val(d) + e4m3_val((uint8_t)(d + 1))) / 2.0); }
            else if (g == G_F2FP_BF16 && k < 4) { /* exact bf16 ties: low half 0x8000 */
                x = u2f((r & 0xffff0000u) | 0x8000u); y = u2f(((r << 16) & 0xbfff0000u) | 0x8000u); }
            else { int span = g == G_F2FP_E4M3 ? 26 : 200, lo = g == G_F2FP_E4M3 ? -14 : -100;
                x = ldexpf(1.0f + (float)(rnd() & 0x7fffff) / 8388608.0f, lo + (int)(rnd() % (uint32_t)span)); if (r & 1) x = -x;
                y = ldexpf(1.0f + (float)(rnd() & 0x7fffff) / 8388608.0f, lo + (int)(rnd() % (uint32_t)span)); if (r & 2) y = -y; }
            a[i] = f2u(x); b[i] = f2u(y); break; }
        case G_EX2: a[i] = 0; b[i] = f2u(ldexpf((float)(int32_t)(r & 0xfffff) - 524288.0f, -15)); break; /* [-16, 16) */
        case G_LG2: a[i] = 0; b[i] = f2u(ldexpf(1.0f + (float)(r & 0x7fffff) / 8388608.0f, (int)(rnd() % 200) - 100)); break;
        }
    }
}

/* 1 when the chip's word agrees with the golden; *how (F2FP) records which operand landed in the upper half */
static int check_one(Golden g, uint32_t a, uint32_t b, uint32_t out, int *order_ab, int *order_ba) {
    switch (g) {
    case G_I2F_S32: return out == f2u((float)(int32_t)b);
    case G_F2FP_BF16: {
        uint32_t ab = (uint32_t)bf16_rne(u2f(a)) << 16 | bf16_rne(u2f(b)), ba = (uint32_t)bf16_rne(u2f(b)) << 16 | bf16_rne(u2f(a));
        *order_ab += out == ab; *order_ba += out == ba; return out == ab || out == ba; }
    case G_F2FP_E4M3: {
        uint32_t ab = (uint32_t)e4m3_rne_satfinite(u2f(a)) << 8 | e4m3_rne_satfinite(u2f(b)), ba = (uint32_t)e4m3_rne_satfinite(u2f(b)) << 8 | e4m3_rne_satfinite(u2f(a));
        *order_ab += out == ab; *order_ba += out == ba; return out == ab || out == ba; }
    case G_EX2: { double want = exp2((double)u2f(b)), got = (double)u2f(out); return fabs(got - want) <= want * ldexp(1.0, -21); }
    case G_LG2: { double want = log2((double)u2f(b)), got = (double)u2f(out); return fabs(got - want) <= ldexp(1.0, -21) * (fabs(want) > 1.0 ? fabs(want) : 1.0); }
    }
    return 0;
}

static void put32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static uint32_t get32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }

/* The scheduling word of the slot under test, from the NAK layout (sm70_encode.rs set_instr_deps): delay 105..108,
 * yield 109, write barrier 110..112, read barrier 113..115, wait mask 116..121 (w3 bits 9.., 13, 14.., 17.., 20..).
 * Waits on scoreboard 4 (the b load), stalls 15 cycles, writes scoreboard 0 when variable-latency. */
static uint32_t sched_word(int var) { return (0x10u << 20) | (7u << 17) | ((var ? 0u : 7u) << 14) | (15u << 9); }

static int build_scaffold(uint8_t *code, const Case *c, int nc) {
    size_t len = 0;
    if (omega_blackwell_encode_vecadd(code, OMEGA_BW_VECADD_CODE_SIZE, &len) != 0 || len != OMEGA_BW_VECADD_CODE_SIZE) return -1;
    uint8_t *op = code + SLOT_OP * 16u, *st = code + SLOT_STG * 16u;
    if ((get32(op) & 0xfffu) != 0x210u || (get32(st) & 0xfffu) != 0x986u) return -1; /* IADD3 / STG where expected */
    put32(op, c->w0); put32(op + 4, c->w1); put32(op + 8, nc ? c->w2 ^ c->nc_w2_xor : c->w2); put32(op + 12, sched_word(c->var_latency));
    put32(st + 12, get32(st + 12) | (1u << 20)); /* the store waits on scoreboard 0 as well */
    return 0;
}

/* ------------------------------------------------------------------ chip runs */

typedef struct { OmegaGpuMem code, a, b, c; } Bufs;
static void free_bufs(Bufs *m) {
    if (m->code.cpu) omega_gpu_session_free(&m->code);
    if (m->a.cpu) omega_gpu_session_free(&m->a);
    if (m->b.cpu) omega_gpu_session_free(&m->b);
    if (m->c.cpu) omega_gpu_session_free(&m->c);
}
/* one launch of code (len bytes) over n elements; out gets c[]. Returns 0, or -1 (stage printed). */
static int chip_run(const uint8_t *code, size_t len, uint32_t gpr, const uint32_t *a, const uint32_t *b, uint32_t n, uint32_t *out) {
    Bufs m; memset(&m, 0, sizeof m);
    size_t bytes = (size_t)n * 4u;
    int rc = -1;
    if (omega_gpu_session_alloc_code(len, &m.code) || omega_gpu_session_alloc(bytes, &m.a) || omega_gpu_session_alloc(bytes, &m.b) ||
        omega_gpu_session_alloc(bytes, &m.c)) { printf("alloc failed: %s\n", omega_gpu_session_last_error()); goto done; }
    memcpy(m.code.cpu, code, len); memcpy(m.a.cpu, a, bytes); memcpy(m.b.cpu, b, bytes); memset(m.c.cpu, 0xa5, bytes);
    if (omega_gpu_session_sync_to_gpu(&m.code) || omega_gpu_session_sync_to_gpu(&m.a) || omega_gpu_session_sync_to_gpu(&m.b) ||
        omega_gpu_session_sync_to_gpu(&m.c)) { printf("sync failed: %s\n", omega_gpu_session_last_error()); goto done; }
    {
        uint32_t args[OMEGA_BW_CBANK_MATMUL_ARGS_WORDS] = { (uint32_t)m.a.va, (uint32_t)(m.a.va >> 32), (uint32_t)m.b.va, (uint32_t)(m.b.va >> 32),
                                                          (uint32_t)m.c.va, (uint32_t)(m.c.va >> 32), n, 0, 0, 0 };
        OmegaGpuLaunch L = { .code_va = m.code.va, .gpr_count = gpr < 64 ? 64 : gpr, .threads_x = THREADS, .threads_y = 1,
                             .grid_x = (n + THREADS - 1) / THREADS, .grid_y = 1, .num_elements = n,
                             .args = args, .n_args = OMEGA_BW_CBANK_MATMUL_ARGS_WORDS, .timeout_ms = 3000 };
        uint64_t ns = 0;
        if (omega_gpu_session_launch(&L, &ns, NULL) != 0) { printf("launch failed: %s\n", omega_gpu_session_last_error()); goto done; }
        if (omega_gpu_session_sync_to_cpu(&m.c) != 0) { printf("sync back failed: %s\n", omega_gpu_session_last_error()); goto done; }
    }
    memcpy(out, m.c.cpu, bytes);
    rc = 0;
done:
    free_bufs(&m);
    return rc;
}

static int run_case(const Case *c, int nc) {
    static uint32_t a[N_ELEM], b[N_ELEM], out[N_ELEM];
    static uint8_t code[OMEGA_BW_VECADD_CODE_SIZE];
    gen_inputs(c->golden, a, b);
    if (build_scaffold(code, c, nc) != 0) { printf("scaffold build failed\n"); return 2; }
    const uint8_t *op = code + SLOT_OP * 16u;
    printf("case %s%s: word %08x %08x %08x %08x\n  pins: %s\n  word source: %s\n", c->name, nc ? " (negative control)" : "",
           get32(op), get32(op + 4), get32(op + 8), get32(op + 12), c->field, c->source);
    omega_gpu_session_lock();
    if (!omega_gpu_session_open()) { printf("GPU not available (%s)\nNOT_RUN\n", omega_gpu_session_last_error()); omega_gpu_session_unlock(); return 2; }
    int r = chip_run(code, sizeof code, 64, a, b, N_ELEM, out);
    omega_gpu_session_unlock();
    omega_gpu_session_close();
    if (r != 0) { printf("RESULT %s%s: CHIP_REFUSED (launch did not complete)\n", c->name, nc ? " nc" : ""); return nc ? 0 : 1; }
    uint32_t ok = 0, unwritten = 0, first = UINT32_MAX; int ab = 0, ba = 0;
    for (uint32_t i = 0; i < N_ELEM; i++) {
        if (out[i] == 0xa5a5a5a5u) unwritten++;
        if (check_one(c->golden, a[i], b[i], out[i], &ab, &ba)) ok++; else if (first == UINT32_MAX) first = i;
    }
    printf("  golden match %u/%u, unwritten %u", ok, N_ELEM, unwritten);
    if (c->golden == G_F2FP_BF16 || c->golden == G_F2FP_E4M3) printf(", packing: src0 high %d, src1 high %d", ab, ba);
    printf("\n");
    if (first != UINT32_MAX) printf("  first miss i=%u a=%08x b=%08x chip=%08x\n", first, a[first], b[first], out[first]);
    int pass = ok == N_ELEM && unwritten == 0;
    if (nc) {
        printf("RESULT %s nc: %s\n", c->name, pass ? "NOT_RED (the field under test changed nothing: the case pins nothing)" : "RED (as required)");
        return pass ? 1 : 0;
    }
    printf("RESULT %s: %s\n", c->name, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

/* bssy73: production reconvergence probes vs the same probes with bit 73 cleared on BSSY/BSYNC */
static int clear_bit73(uint8_t *code, size_t insns) {
    int n = 0;
    for (size_t i = 0; i < insns; i++) {
        uint8_t *w = code + i * 16u; uint32_t op = get32(w) & 0xfffu;
        if ((op == 0x945u || op == 0x941u) && (get32(w + 8) & 0x200u)) { put32(w + 8, get32(w + 8) & ~0x200u); n++; }
    }
    return n;
}
static int run_bssy73(int nc) {
    static const OmegaGpuEwOp ops[] = { OMEGA_GPU_EW_RC_DIAMOND, OMEGA_GPU_EW_RC_LOOPBREAK, OMEGA_GPU_EW_RC_NESTED, OMEGA_GPU_EW_RC_EXIT_BAR, OMEGA_GPU_EW_RC_DEPTH16 };
    static const char *names[] = { "rc_diamond", "rc_loopbreak", "rc_nested", "rc_exit_bar", "rc_depth16" };
    static uint32_t a[N_ELEM], b[N_ELEM], ref[N_ELEM], got[N_ELEM];
    for (uint32_t i = 0; i < N_ELEM; i++) { a[i] = rnd(); b[i] = rnd(); }
    printf("case bssy73%s: BSSY/BSYNC with bit 73 cleared (NAK form) vs set (production), %u elements per probe\n", nc ? " (negative control)" : "", N_ELEM);
    omega_gpu_session_lock();
    if (!omega_gpu_session_open()) { printf("GPU not available (%s)\nNOT_RUN\n", omega_gpu_session_last_error()); omega_gpu_session_unlock(); return 2; }
    int fail = 0, ran = 0;
    for (size_t p = 0; p < sizeof ops / sizeof ops[0] && !fail; p++) {
        OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
        if (omega_gpu_elementwise_codegen_ir(ops[p], 0, NULL, &k) != 0) { printf("  %s: codegen failed\n", names[p]); fail = 1; break; }
        uint8_t *off = malloc(k.code_size);
        if (!off) { free(k.code); fail = 1; break; }
        memcpy(off, k.code, k.code_size);
        int cleared = clear_bit73(off, k.insn_count);
        int r1 = chip_run(k.code, k.code_size, k.gpr_count, a, b, N_ELEM, ref);
        int r2 = r1 ? -1 : chip_run(off, k.code_size, k.gpr_count, a, b, N_ELEM, got);
        if (r1 || r2) { printf("  %s: launch failed (bit 73 %s)\n", names[p], r1 ? "set" : "cleared"); fail = 1; }
        else {
            if (nc) got[N_ELEM / 3] ^= 1u; /* the comparator must see one flipped bit */
            uint32_t diff = 0, poison = 0;
            for (uint32_t i = 0; i < N_ELEM; i++) { diff += ref[i] != got[i]; poison += ref[i] == 0xa5a5a5a5u; }
            printf("  %s: %zu insns, %d BSSY/BSYNC words cleared, outputs differ in %u/%u words, unwritten %u\n", names[p], k.insn_count, cleared, diff, N_ELEM, poison);
            if (cleared == 0 || poison) fail = 1;
            if (nc ? diff == 0 : diff != 0) fail = 1;
            ran++;
        }
        free(off); free(k.code);
    }
    omega_gpu_session_unlock();
    omega_gpu_session_close();
    if (nc) { printf("RESULT bssy73 nc: %s\n", fail ? "NOT_RED" : "RED (as required)"); return fail; }
    printf("RESULT bssy73: %s (%d probes)\n", fail ? "FAIL" : "PASS", ran);
    return fail;
}

/* ------------------------------------------------------------------ host-only */

static int host_only(void) {
    int fail = 0;
#define H(cond, msg) do { if (!(cond)) { printf("FAIL host: %s\n", msg); fail++; } } while (0)
    /* bfloat16: 1.0 = 0x3f80; 1 + 2^-8 is a tie to even (down), 1 + 3*2^-8 a tie up; NaN stays NaN */
    H(bf16_rne(1.0f) == 0x3f80, "bf16 1.0");
    H(bf16_rne(u2f(0x3f808000u)) == 0x3f80, "bf16 tie down to even");
    H(bf16_rne(u2f(0x3f818000u)) == 0x3f82, "bf16 tie up to even");
    H(bf16_rne(u2f(0x7f7fffffu)) == 0x7f80, "bf16 max float rounds to infinity");
    H((bf16_rne(NAN) & 0x7f80) == 0x7f80 && (bf16_rne(NAN) & 0x7f), "bf16 NaN");
    /* OCP OFP8 E4M3 table points: 0x38 = 1.0, 0x7e = 448, 0x01 = 2^-9, 0xb8 = -1.0, 0x08 = 2^-6 (min normal) */
    H(e4m3_val(0x38) == 1.0 && e4m3_val(0x7e) == 448.0 && e4m3_val(0x01) == ldexp(1.0, -9) && e4m3_val(0xb8) == -1.0 && e4m3_val(0x08) == ldexp(1.0, -6), "e4m3 decode");
    H(e4m3_rne_satfinite(1.0f) == 0x38 && e4m3_rne_satfinite(500.0f) == 0x7e && e4m3_rne_satfinite(INFINITY) == 0x7e && e4m3_rne_satfinite(-INFINITY) == 0xfe, "e4m3 satfinite");
    H(e4m3_rne_satfinite(NAN) == 0x7f, "e4m3 NaN");
    H(e4m3_rne_satfinite(1.0625f) == 0x38 && e4m3_rne_satfinite(1.1875f) == 0x3a, "e4m3 ties to even");
    H(e4m3_rne_satfinite(ldexpf(1.0f, -10)) == 0x00 && e4m3_rne_satfinite(ldexpf(3.0f, -10)) == 0x02, "e4m3 subnormal ties");
    for (int c = 0; c < 0x7f; c++) H(e4m3_rne_satfinite((float)e4m3_val((uint8_t)c)) == c, "e4m3 round trip");
    /* the I2F golden: (float)(int32) rounds to nearest even on this host */
    H(f2u((float)16777217) == 0x4b800000u && f2u((float)16777219) == 0x4b800002u && f2u((float)(int32_t)0x80000000u) == 0xcf000000u, "i2f RNE");
    /* scaffold: slot 17 IADD3, slot 18 STG; each case's word lands there; the store gains the scoreboard-0 wait */
    static uint8_t code[OMEGA_BW_VECADD_CODE_SIZE];
    for (size_t i = 0; i < NCASES; i++) {
        H(build_scaffold(code, &CASES[i], 0) == 0, "scaffold");
        H(get32(code + SLOT_OP * 16u) == CASES[i].w0 && get32(code + SLOT_OP * 16u + 8) == CASES[i].w2, "slot word");
        H((get32(code + SLOT_STG * 16u + 12) >> 20 & 0x3fu) == 1u, "store waits on scoreboard 0");
        H(build_scaffold(code, &CASES[i], 1) == 0 && get32(code + SLOT_OP * 16u + 8) == (CASES[i].w2 ^ CASES[i].nc_w2_xor) && CASES[i].nc_w2_xor, "negative control word differs");
    }
    /* comparator negative control: a golden-exact output passes, a one-ulp miss fails */
    { int x = 0, y = 0; H(check_one(G_I2F_S32, 0, 7, f2u(7.0f), &x, &y) && !check_one(G_I2F_S32, 0, 7, f2u(7.0f) + 1, &x, &y), "comparator i2f");
      H(check_one(G_F2FP_E4M3, f2u(1.0f), f2u(-1.0f), 0x38b8u, &x, &y) && !check_one(G_F2FP_E4M3, f2u(1.0f), f2u(-1.0f), 0x38b9u, &x, &y), "comparator e4m3"); }
    /* bssy73 patcher: production reconvergence kernels carry bit 73 on every BSSY/BSYNC, the patch clears all of them */
    { OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
      H(omega_gpu_elementwise_codegen_ir(OMEGA_GPU_EW_RC_NESTED, 0, NULL, &k) == 0, "rc_nested codegen");
      if (k.code) { int n = clear_bit73(k.code, k.insn_count); H(n >= 4, "bit 73 cleared on >= 4 BSSY/BSYNC words"); H(clear_bit73(k.code, k.insn_count) == 0, "none left"); free(k.code); } }
    printf("host-only: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
#undef H
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--host-only")) return host_only();
    if (argc == 2 && !strcmp(argv[1], "--list")) { for (size_t i = 0; i < NCASES; i++) printf("%s\n", CASES[i].name); printf("bssy73\n"); return 0; }
    if (argc >= 3 && !strcmp(argv[1], "--case")) {
        int nc = argc == 4 && !strcmp(argv[3], "--nc");
        if (!strcmp(argv[2], "bssy73")) return run_bssy73(nc);
        for (size_t i = 0; i < NCASES; i++) if (!strcmp(argv[2], CASES[i].name)) return run_case(&CASES[i], nc);
        printf("unknown case %s\n", argv[2]);
        return 2;
    }
    printf("usage: gpu_provenance_enc_test --host-only | --list | --case NAME [--nc]\n");
    return 2;
}
