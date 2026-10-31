/* bw_hazard_test.c: the static scoreboard hazard checker (src/omega_bw_hazard.c, omega-mv#554) run over every kernel
 * Omega generates, plus its own proofs. Host only, no device.
 *
 *   1. decoder vs encoder: every IR op encoded with known registers decodes to exactly those registers;
 *   2. red receipt: the pre-#553 head_dim 256 attend kernel (the bytes that faulted on the GB10) fails with WAR on
 *      R22/R23 at IR 88/89; the same kernel after the encoder fix passes; the fixed kernel with the one lost wait
 *      cleared fails again;
 *   3. negative controls: small encoded programs with one hazard each are caught, the same programs with the wait
 *      in place pass, and an undecodable word is "not checked", never clean;
 *   4. sweep: every generator over its accepted shapes (production kernels, mutant 0), encoded and patched as the
 *      launchers do; every finding is printed and must be on the triaged list below.
 *
 * --disasm: additionally cross-checks the decoder against Omega's own sm_121 disassembler (src/omega_sm121_disasm.c,
 * in process) on every distinct kernel: the base registers each instruction names must be the ones the decoder found.
 * The disassembler is checked word for word against nvdisasm by test-sm121dis; this test never calls nvdisasm (bar C6
 * of docs/OMEGA.md: every check that called nvdisasm calls ours). */
#include "omega_bw_hazard.h"
#include "omega_sm121_disasm.h"
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_codegen_moe.h"
#include "omega_blackwell_codegen_moe2.h"
#include "omega_blackwell_codegen_bf16.h"
#include "omega_blackwell_codegen_router.h"
#include "omega_blackwell_codegen_attn.h"
#include "omega_blackwell_codegen_gemm.h"
#include "omega_blackwell_codegen_tern.h"
#include "omega_blackwell_codegen_tc.h"
#include "omega_blackwell_codegen_tc2.h"
#include "omega_blackwell_codegen_qg.h"
#include "omega_blackwell_codegen_qmma.h"
#include "omega_blackwell_codegen_pf.h"
#include "omega_blackwell_codegen_pfhost.h"
#include "omega_blackwell_codegen_dec.h"
#include "omega_blackwell_codegen_pattn.h"
#include "omega_gpu_attention_api.h"
#include "omega_gpu_elementwise_api.h"
#include "omega_gpu_prefill_attn_api.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_checks, g_failed, g_disasm;
static const char *g_dump_dir; /* --dump DIR: write each distinct kernel with findings as DIR/<family>-<n>.bin (triage) */
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static BlackwellIRProgram g_prog;
static uint8_t g_code[OMEGA_BW_CODE_CAP_MAX + 4096];
static OmegaBwHazardReport g_rep;

#define CW(stall, wbar, rbar, wait) \
    (((uint32_t)(stall) << 9) | ((uint32_t)(wbar) << 14) | ((uint32_t)(rbar) << 17) | ((uint32_t)(wait) << 20))

/* ------------------------------------------------------------------------------------------- 1. decoder vs encoder */
typedef struct { int w[3]; int r[6]; } Ops; /* entries: reg | n << 8 | uniform << 12, 0 terminates */
#define G(reg, n) ((reg) | (n) << 8)
#define U(reg, n) ((reg) | (n) << 8 | 1 << 12)

static int has(const OmegaBwOperand *o, unsigned cnt, int want) {
    for (unsigned i = 0; i < cnt; i++)
        if (o[i].reg == (want & 0xff) && o[i].n == ((want >> 8) & 0xf) && o[i].uniform == ((want >> 12) & 1)) return 1;
    return 0;
}
static int count(const int *a, int max) { int n = 0; while (n < max && a[n]) n++; return n; }

static void test_decoder_vs_encoder(void) {
    /* registers: dst 8, src1 12, src2 16, src3 20, uniform 6 (encode_one maps vreg n to Rn / URn) */
    static const struct { BlackwellIROpcode op; int src2, src3, ureg; Ops e; } t[] = {
        { BW_IR_NOP, -1, -1, -1, { {0}, {0} } },
        { BW_IR_MOV_RZ, -1, -1, -1, { {G(8,1)}, {0} } },
        { BW_IR_MOV_IMM, -1, -1, -1, { {G(8,1)}, {0} } },
        { BW_IR_S2R, -1, -1, -1, { {G(8,1)}, {0} } },
        { BW_IR_LDC, -1, -1, -1, { {G(8,1)}, {0} } },
        { BW_IR_LDC64, -1, -1, -1, { {G(8,2)}, {0} } },
        { BW_IR_IMAD, 16, 20, -1, { {G(8,1)}, {G(12,1), G(16,1), G(20,1)} } },
        { BW_IR_IMAD, -1, 20, 6, { {G(8,1)}, {G(12,1), U(6,1), G(20,1)} } },
        { BW_IR_IMAD, -1, 20, -1, { {G(8,1)}, {G(12,1), G(20,1)} } },
        { BW_IR_IMAD_WIDE, -1, 20, -1, { {G(8,2)}, {G(12,1), G(20,2)} } },
        { BW_IR_ISETP_GE, -1, -1, -1, { {0}, {G(12,1)} } },
        { BW_IR_ISETP_GE, -1, -1, 6, { {0}, {G(12,1), U(6,1)} } },
        { BW_IR_LDG_E, -1, -1, 6, { {G(8,1)}, {G(12,2), U(6,1)} } },
        { BW_IR_STG_E, 16, -1, 6, { {0}, {G(12,2), G(16,1), U(6,1)} } },
        { BW_IR_STG_EF, 16, -1, 6, { {0}, {G(12,2), G(16,1), U(6,1)} } },
        { BW_IR_IADD3, 16, -1, -1, { {G(8,1)}, {G(12,1), G(16,1)} } },
        { BW_IR_HMMA_F16, 16, 20, -1, { {G(8,4)}, {G(12,4), G(16,2), G(20,4)} } },
        { BW_IR_HMMA_BF16, 16, 20, -1, { {G(8,4)}, {G(12,4), G(16,2), G(20,4)} } },
        { BW_IR_SHF_R, -1, -1, -1, { {G(8,1)}, {G(12,1)} } },
        { BW_IR_LOP3_AND, -1, -1, -1, { {G(8,1)}, {G(12,1)} } },
        { BW_IR_LDG_E_U16, -1, -1, 6, { {G(8,1)}, {G(12,2), U(6,1)} } },
        { BW_IR_LDG_STRONG_SYS, -1, -1, 6, { {G(8,1)}, {G(12,2), U(6,1)} } },
        { BW_IR_LDG_MMIO, -1, -1, 6, { {G(8,1)}, {G(12,2), U(6,1)} } },
        { BW_IR_STG_STRONG_SYS, 16, -1, 6, { {0}, {G(12,2), G(16,1), U(6,1)} } },
        { BW_IR_MEMBAR_ALL_SYS, -1, -1, -1, { {0}, {0} } },
        { BW_IR_MEMBAR_SC_SYS, -1, -1, -1, { {0}, {0} } },
        { BW_IR_CCTL_IVALL, -1, -1, -1, { {0}, {0} } },
        { BW_IR_ATOMG_ADD_STRONG_SYS, 16, -1, 6, { {G(8,1)}, {G(12,2), G(16,1), U(6,1)} } },
        { BW_IR_ATOMG_EXCH_STRONG_SYS, 16, -1, 6, { {G(8,1)}, {G(12,2), G(16,1), U(6,1)} } },
        { BW_IR_ISETP_GE_U32, 16, -1, -1, { {0}, {G(12,1), G(16,1)} } },
        { BW_IR_LOP3_XOR, 16, -1, -1, { {G(8,1)}, {G(12,1), G(16,1)} } },
        { BW_IR_FADD, 16, -1, -1, { {G(8,1)}, {G(12,1), G(16,1)} } },
        { BW_IR_FSUB, 16, -1, -1, { {G(8,1)}, {G(12,1), G(16,1)} } },
        { BW_IR_FMUL, 16, -1, -1, { {G(8,1)}, {G(12,1), G(16,1)} } },
        { BW_IR_FFMA, 16, 20, -1, { {G(8,1)}, {G(12,1), G(16,1), G(20,1)} } },
        { BW_IR_FSETP, 16, -1, -1, { {0}, {G(12,1), G(16,1)} } },
        { BW_IR_FSEL, 16, -1, -1, { {G(8,1)}, {G(12,1), G(16,1)} } },
        { BW_IR_FMNMX_MIN, 16, -1, -1, { {G(8,1)}, {G(12,1), G(16,1)} } },
        { BW_IR_FMNMX_MAX, 16, -1, -1, { {G(8,1)}, {G(12,1), G(16,1)} } },
        { BW_IR_I2FP, -1, -1, -1, { {G(8,1)}, {G(12,1)} } },
        { BW_IR_F2I, -1, -1, -1, { {G(8,1)}, {G(12,1)} } },
        { BW_IR_MUFU_RCP, -1, -1, -1, { {G(8,1)}, {G(12,1)} } },
        { BW_IR_MUFU_RSQ, -1, -1, -1, { {G(8,1)}, {G(12,1)} } },
        { BW_IR_SHFL_DOWN, -1, -1, -1, { {G(8,1)}, {G(12,1)} } },
        { BW_IR_LDS, -1, -1, -1, { {G(8,1)}, {G(12,1)} } },
        { BW_IR_STS, 16, -1, -1, { {0}, {G(12,1), G(16,1)} } },
        { BW_IR_EXIT, -1, -1, -1, { {0}, {0} } },
        { BW_IR_MUFU_EX2, -1, -1, -1, { {G(8,1)}, {G(12,1)} } },
        { BW_IR_BAR_SYNC, -1, -1, -1, { {0}, {0} } },
        { BW_IR_LDS32, -1, -1, -1, { {G(8,1)}, {G(12,1)} } },
        { BW_IR_STS32, 16, -1, -1, { {0}, {G(12,1), G(16,1)} } },
        { BW_IR_LOP3_LUT, 16, 20, -1, { {G(8,1)}, {G(12,1), G(16,1), G(20,1)} } },
        { BW_IR_SHF_L_U32, 16, 20, -1, { {G(8,1)}, {G(12,1), G(16,1), G(20,1)} } },
        { BW_IR_IMAD_HI_U32, 16, 20, -1, { {G(8,1)}, {G(12,1), G(16,1), G(20,1)} } },
        { BW_IR_LDG_E_64, -1, -1, 6, { {G(8,2)}, {G(12,2), U(6,1)} } },
        { BW_IR_LDG_E_128, -1, -1, 6, { {G(8,4)}, {G(12,2), U(6,1)} } },
        { BW_IR_LDS128, -1, -1, -1, { {G(8,4)}, {G(12,1)} } },
    };
    int bad = 0;
    for (unsigned i = 0; i < sizeof t / sizeof *t; i++) {
        BlackwellIRInsn in; memset(&in, 0, sizeof in);
        in.op = t[i].op; in.dst_vreg = 8; in.src1_vreg = 12; in.src2_vreg = t[i].src2; in.src3_vreg = t[i].src3; in.ureg = t[i].ureg;
        in.control = CW(3, 2, 5, 0x29) | 0x2000u;
        /* IMAD and ISETP_GE take the uniform-operand form only with is_uniform set */
        if ((t[i].op == BW_IR_IMAD || t[i].op == BW_IR_ISETP_GE) && t[i].ureg >= 0) in.is_uniform = true;
        uint32_t w[4]; uint8_t b[16]; OmegaBwInsn d;
        if (omega_blackwell_encode_one(&in, w) != 0) { CHECK(0, "encode op %d", (int)t[i].op); bad++; continue; }
        for (int k = 0; k < 4; k++) { b[4*k] = (uint8_t)w[k]; b[4*k+1] = (uint8_t)(w[k] >> 8); b[4*k+2] = (uint8_t)(w[k] >> 16); b[4*k+3] = (uint8_t)(w[k] >> 24); }
        if (omega_bw_decode(b, &d) != 0) { CHECK(0, "decode of op %d (word0 %08x) refused", (int)t[i].op, w[0]); bad++; continue; }
        int nw = count(t[i].e.w, 3), nr = count(t[i].e.r, 6), ok = d.nw == nw && d.nr == nr;
        for (int k = 0; k < nw && ok; k++) ok = has(d.wr, d.nw, t[i].e.w[k]);
        for (int k = 0; k < nr && ok; k++) ok = has(d.rd, d.nr, t[i].e.r[k]);
        if (!ok) { bad++; CHECK(0, "op %d (%s): decoded %u writes %u reads, expected %d / %d", (int)t[i].op, d.name ? d.name : "?", d.nw, d.nr, nw, nr); }
    }
    { /* uniform loads write the uniform file */
        BlackwellIROpcode u[2] = { BW_IR_LDCU, BW_IR_LDCU64 };
        for (int i = 0; i < 2; i++) {
            BlackwellIRInsn in; memset(&in, 0, sizeof in); in.op = u[i]; in.dst_vreg = 6; in.src1_vreg = in.src2_vreg = in.src3_vreg = in.ureg = -1; in.is_uniform = true;
            uint32_t w[4]; uint8_t b[16]; OmegaBwInsn d;
            CHECK(omega_blackwell_encode_one(&in, w) == 0, "encode LDCU");
            memcpy(b, w, 16);
            CHECK(omega_bw_decode(b, &d) == 0 && d.nw == 1 && d.wr[0].uniform && d.wr[0].reg == 6 && d.wr[0].n == (i ? 2 : 1) && d.nr == 0 && d.var,
                  "LDCU%s decodes to a uniform write of UR6", i ? ".64" : "");
        }
    }
    { /* the control word: every field at a distinct value (NOP keeps insn->control on every encoder version) */
        BlackwellIRInsn in; memset(&in, 0, sizeof in); in.op = BW_IR_NOP; in.dst_vreg = in.src1_vreg = in.src2_vreg = in.src3_vreg = in.ureg = -1;
        in.control = CW(3, 2, 5, 0x29) | 0x2000u;
        uint32_t w[4]; OmegaBwInsn d; CHECK(omega_blackwell_encode_one(&in, w) == 0, "encode NOP");
        CHECK(omega_bw_decode((const uint8_t *)w, &d) == 0 && d.stall == 3 && d.yield == 1 && d.wbar == 2 && d.rbar == 5 && d.wait == 0x29,
              "control fields stall %u yield %u wbar %u rbar %u wait 0x%x", d.stall, d.yield, d.wbar, d.rbar, d.wait);
    }
    { /* branch targets, both signs, both predicate senses */
        int32_t deltas[] = { -300, -64, -63, -5, -1, 0, 1, 2, 63, 64, 65, 1000 };
        for (unsigned i = 0; i < sizeof deltas / sizeof *deltas; i++)
            for (int p = 0; p < 3; p++) {
                BlackwellIRInsn in; memset(&in, 0, sizeof in); in.op = BW_IR_BRA; in.imm = (uint32_t)deltas[i];
                in.dst_vreg = in.src1_vreg = in.src2_vreg = in.src3_vreg = in.ureg = -1; in.predicate_p0 = p > 0; in.predicate_not = p == 2;
                uint32_t w[4]; OmegaBwInsn d;
                CHECK(omega_blackwell_encode_one(&in, w) == 0 && omega_bw_decode((const uint8_t *)w, &d) == 0 && d.flow == OMEGA_BW_FLOW_BRA &&
                      d.target == deltas[i] && d.guard == (p == 0 ? 7u : p == 1 ? 0u : 8u), "BRA delta %d pred %d decodes to %d guard %u", deltas[i], p, d.target, d.guard);
            }
    }
    printf("decoder vs encoder: %zu forms, %d mismatched\n", sizeof t / sizeof *t, bad);
}

/* -------------------------------------------------------------------------------------------- fixtures and helpers */
static size_t load(const char *dir, const char *name, uint8_t *buf, size_t cap) {
    char path[512]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    size_t n = fread(buf, 1, cap, f); fclose(f); return n;
}
static int finding(const OmegaBwHazardReport *r, uint32_t pc, uint8_t kind, uint8_t reg) {
    for (size_t i = 0; i < r->count && i < OMEGA_BW_HAZARD_MAX; i++)
        if (r->f[i].pc == pc && r->f[i].kind == kind && r->f[i].reg == reg && !r->f[i].uniform) return 1;
    return 0;
}
static void print_findings(const char *tag, const OmegaBwHazardReport *r) {
    for (size_t i = 0; i < r->count && i < OMEGA_BW_HAZARD_MAX; i++)
        printf("  %s: IR %u %s %sR%u (release mask 0x%x, last producer/reader IR %u)\n", tag, r->f[i].pc, omega_bw_hazard_kind(r->f[i].kind),
               r->f[i].uniform ? "U" : "", r->f[i].reg, r->f[i].release, r->f[i].src_pc);
}
static void set_wait(uint8_t *code, uint32_t pc, unsigned wait) {
    uint8_t *p = code + (size_t)pc * 16u + 12u; uint32_t w3 = (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    w3 = (w3 & ~(0x3fu << 20)) | (wait & 0x3fu) << 20;
    p[0] = (uint8_t)w3; p[1] = (uint8_t)(w3 >> 8); p[2] = (uint8_t)(w3 >> 16); p[3] = (uint8_t)(w3 >> 24);
}

/* ---------------------------------------------------------------------------------------------- 2. red receipt */
static void test_red_receipt(const char *dir) {
    static uint8_t a[1 << 16];
    size_t n = load(dir, "pattn256_attend_ad5d650b.bin", a, sizeof a);
    CHECK(n == 10112, "fixture pattn256_attend_ad5d650b.bin (%zu bytes)", n);
    if (n == 10112) {
        int rc = omega_bw_hazard_check(a, n, &g_rep);
        CHECK(rc == 0 && g_rep.count == 2 && finding(&g_rep, 88, OMEGA_BW_HZ_WAR, 22) && finding(&g_rep, 89, OMEGA_BW_HZ_WAR, 23) &&
              g_rep.f[0].release == 0x3 && g_rep.f[1].release == 0x3,
              "RED: pre-#553 attend must fail with WAR R22/R23 at IR 88/89 (rc %d, %zu findings)", rc, g_rep.count);
        printf("red receipt, pre-#553 attend (the kernel that faulted on the GB10): %zu findings\n", g_rep.count);
        print_findings("pre-#553 attend", &g_rep);
    }
    n = load(dir, "pattn256_attend_e3a26848.bin", a, sizeof a);
    CHECK(n == 10112, "fixture pattn256_attend_e3a26848.bin (%zu bytes)", n);
    if (n == 10112) {
        int rc = omega_bw_hazard_check(a, n, &g_rep);
        CHECK(rc == 0 && g_rep.count == 0, "GREEN: the fixed attend must pass (rc %d, %zu findings)", rc, g_rep.count);
        printf("fixed attend (e3a26848): rc %d, %zu findings, %zu of %zu instructions reachable\n", rc, g_rep.count, g_rep.reachable, g_rep.insns);
        /* the one wait the old encoder lost: the IADD3 at IR 55 waits on the Q loads' barriers 0x3 */
        OmegaBwInsn d; CHECK(omega_bw_decode(a + 55 * 16, &d) == 0 && d.op == 0x210 && d.wait == 0x3, "IR 55 is the IADD3 waiting 0x3");
        set_wait(a, 55, 0x10);
        rc = omega_bw_hazard_check(a, n, &g_rep);
        CHECK(rc == 0 && g_rep.count == 2 && finding(&g_rep, 88, OMEGA_BW_HZ_WAR, 22) && finding(&g_rep, 89, OMEGA_BW_HZ_WAR, 23),
              "MUTANT: the fixed attend with IR 55 waiting 0x10 must fail again (%zu findings)", g_rep.count);
    }
    n = load(dir, "pattn256_prep_ad5d650b.bin", a, sizeof a);
    CHECK(n == 5376, "fixture pattn256_prep_ad5d650b.bin");
    if (n == 5376) {
        int rc = omega_bw_hazard_check(a, n, &g_rep);
        CHECK(rc == 0 && g_rep.count == 1 && finding(&g_rep, 35, OMEGA_BW_HZ_RAW, 20) && g_rep.f[0].release == 0x1,
              "pre-#553 prep: RAW R20 at IR 35 (S2R at IR 14 on barrier 0; the IADD3 at IR 15 waited on 4 only), %zu findings", g_rep.count);
        print_findings("pre-#553 prep", &g_rep);
    }
    n = load(dir, "pattn256_prep_e3a26848.bin", a, sizeof a);
    CHECK(n == 5376 && omega_bw_hazard_check(a, n, &g_rep) == 0 && g_rep.count == 0, "fixed prep passes (%zu findings)", g_rep.count);
}

/* -------------------------------------------------------------------------------------------- 3. negative controls */
static size_t enc(const BlackwellIRInsn *ins, size_t n, uint8_t *out) {
    for (size_t i = 0; i < n; i++) { uint32_t w[4]; if (omega_blackwell_encode_one(&ins[i], w) != 0) return 0; memcpy(out + 16 * i, w, 16); }
    return 16 * n;
}
static BlackwellIRInsn mk(BlackwellIROpcode op, int d, int a, int b, int c, int u, uint32_t control) {
    BlackwellIRInsn in; memset(&in, 0, sizeof in);
    in.op = op; in.dst_vreg = d; in.src1_vreg = a; in.src2_vreg = b; in.src3_vreg = c; in.ureg = u; in.control = control; return in;
}
static void test_controls(void) {
    uint8_t c[16 * 16]; size_t n;
    const uint32_t FIX = CW(6, 7, 7, 0), LD = CW(4, 0, 1, 0) | 0x2000u, ST = CW(4, 7, 1, 0) | 0x2000u, ENDW = CW(5, 7, 7, 0);
    /* RAW: a load's result read with no wait / with the wait / with a wait on an unrelated barrier (the #545 pattern) */
    BlackwellIRInsn raw[] = { mk(BW_IR_LDG_E, 4, 2, -1, -1, 4, LD), mk(BW_IR_FADD, 6, 4, 4, -1, -1, FIX), mk(BW_IR_EXIT, -1, -1, -1, -1, -1, ENDW) };
    n = enc(raw, 3, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 1 && finding(&g_rep, 1, OMEGA_BW_HZ_RAW, 4), "control RAW caught");
    raw[1].control = CW(6, 7, 7, 0x1); n = enc(raw, 3, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 0, "control RAW with the wait passes");
    raw[1].control = CW(6, 7, 7, 0x10); n = enc(raw, 3, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 1 && finding(&g_rep, 1, OMEGA_BW_HZ_RAW, 4), "control RAW behind a wait on barrier 4 caught");
    raw[1].control = CW(6, 7, 7, 0x2); n = enc(raw, 3, c); /* waiting on the read barrier does not cover the result */
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 1 && finding(&g_rep, 1, OMEGA_BW_HZ_RAW, 4), "control RAW behind the read barrier only caught");
    /* WAR: a store's data register overwritten before the store's read barrier */
    BlackwellIRInsn war[] = { mk(BW_IR_STG_E, -1, 2, 4, -1, 4, ST), mk(BW_IR_MOV_RZ, 4, -1, -1, -1, -1, FIX), mk(BW_IR_EXIT, -1, -1, -1, -1, -1, ENDW) };
    n = enc(war, 3, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 1 && finding(&g_rep, 1, OMEGA_BW_HZ_WAR, 4), "control WAR (store data) caught");
    war[1].dst_vreg = 3; n = enc(war, 3, c); /* the store's address pair is R2:R3 */
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 1 && finding(&g_rep, 1, OMEGA_BW_HZ_WAR, 3), "control WAR (address high half) caught");
    war[1].control = CW(6, 7, 7, 0x2); n = enc(war, 3, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 0, "control WAR with the wait passes");
    war[1].dst_vreg = 5; war[1].control = FIX; n = enc(war, 3, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 0, "control: a register the store does not read passes");
    /* WAR released by the load's completion: waiting on the write barrier also frees the address */
    BlackwellIRInsn warc[] = { mk(BW_IR_LDG_E, 4, 2, -1, -1, 4, LD), mk(BW_IR_MOV_RZ, 2, -1, -1, -1, -1, CW(6, 7, 7, 0x1)), mk(BW_IR_EXIT, -1, -1, -1, -1, -1, ENDW) };
    n = enc(warc, 3, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 0, "control: address overwrite after the load completed passes");
    warc[1].control = FIX; n = enc(warc, 3, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 1 && finding(&g_rep, 1, OMEGA_BW_HZ_WAR, 2), "control WAR (load address) caught");
    /* WAW: a pending load's destination overwritten */
    BlackwellIRInsn waw[] = { mk(BW_IR_LDG_E, 4, 2, -1, -1, 4, LD), mk(BW_IR_MOV_RZ, 4, -1, -1, -1, -1, CW(6, 7, 7, 0x2)), mk(BW_IR_EXIT, -1, -1, -1, -1, -1, ENDW) };
    n = enc(waw, 3, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 1 && finding(&g_rep, 1, OMEGA_BW_HZ_WAW, 4), "control WAW caught");
    /* a variable-latency result with no write barrier at all: nothing can release it */
    BlackwellIRInsn nob[] = { mk(BW_IR_S2R, 4, -1, -1, -1, -1, CW(4, 7, 7, 0)), mk(BW_IR_FADD, 6, 4, 4, -1, -1, CW(6, 7, 7, 0x3f)), mk(BW_IR_EXIT, -1, -1, -1, -1, -1, ENDW) };
    nob[0].imm = BW_SR_TID_X; n = enc(nob, 3, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 1 && finding(&g_rep, 1, OMEGA_BW_HZ_RAW, 4) && g_rep.f[0].release == 0,
          "control: S2R with no write barrier caught even behind a wait on every barrier");
    /* a hazard carried round a loop: the read at the top sees the load issued at the bottom of the previous trip */
    BlackwellIRInsn loop[] = { mk(BW_IR_MOV_RZ, 4, -1, -1, -1, -1, FIX), mk(BW_IR_FADD, 6, 4, 4, -1, -1, FIX), mk(BW_IR_LDG_E, 4, 2, -1, -1, 4, LD),
                               mk(BW_IR_BRA, -1, -1, -1, -1, -1, CW(5, 7, 7, 0)), mk(BW_IR_EXIT, -1, -1, -1, -1, -1, ENDW) };
    loop[3].imm = (uint32_t)-2; loop[3].predicate_p0 = true;
    n = enc(loop, 5, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && finding(&g_rep, 1, OMEGA_BW_HZ_RAW, 4) && finding(&g_rep, 0, OMEGA_BW_HZ_WAW, 4) == 0,
          "control: RAW across the loop back edge caught (%zu findings)", g_rep.count);
    loop[3].control = CW(5, 7, 7, 0x1); n = enc(loop, 5, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 0, "control: the loop with the wait on the branch passes (%zu findings)", g_rep.count);
    /* the path the hazard is on is not the straight line: an unpredicated branch skips the wait */
    BlackwellIRInsn skip[] = { mk(BW_IR_LDG_E, 4, 2, -1, -1, 4, LD), mk(BW_IR_BRA, -1, -1, -1, -1, -1, CW(5, 7, 7, 0)), mk(BW_IR_NOP, -1, -1, -1, -1, -1, CW(1, 7, 7, 0x1)),
                               mk(BW_IR_FADD, 6, 4, 4, -1, -1, FIX), mk(BW_IR_EXIT, -1, -1, -1, -1, -1, ENDW) };
    skip[1].imm = 2; skip[1].predicate_p0 = true; n = enc(skip, 5, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 1 && finding(&g_rep, 3, OMEGA_BW_HZ_RAW, 4), "control: RAW on the branch-taken path caught");
    skip[1].predicate_p0 = false; n = enc(skip, 5, c);
    CHECK(n && omega_bw_hazard_check(c, n, &g_rep) == 0 && g_rep.count == 1 && g_rep.reachable == 4, "control: unreachable wait does not count (%zu reachable)", g_rep.reachable);
    /* undecodable word: not checked, never clean */
    n = enc(raw, 3, c); c[16] = 0x55; c[17] = 0x7f;
    CHECK(omega_bw_hazard_check(c, n, &g_rep) == -1 && g_rep.bad_pc == 1, "control: an unknown opcode is refused (bad_pc %lld)", (long long)g_rep.bad_pc);
    /* branch out of range */
    BlackwellIRInsn far[] = { mk(BW_IR_BRA, -1, -1, -1, -1, -1, CW(5, 7, 7, 0)) }; far[0].imm = 5; n = enc(far, 1, c);
    CHECK(omega_bw_hazard_check(c, n, &g_rep) == -1 && g_rep.bad_pc == 0, "control: a branch out of the buffer is refused");
    CHECK(omega_bw_hazard_check(c, 15, &g_rep) == -1 && omega_bw_hazard_check(NULL, 16, &g_rep) == -1, "control: bad length / NULL refused");
}

/* ------------------------------------------------------------------------------------------------------- 4. sweep */
typedef struct { const char *family; char variant[96]; uint64_t hash; size_t findings; } Seen;
#define MAX_SEEN 8192
static Seen g_seen[MAX_SEEN];
static size_t g_nseen, g_variants, g_unchecked, g_findings_total;

/* Triaged findings on main a8ffd2b9 (2026-10-19), each read in the bytes (instruction, barrier, every wait between).
 * Every finding the sweep prints must be listed here; a listed entry no longer seen prints a NOTE (fixed: remove it).
 * The rule behind A and B is the one ptxas follows (offline oracle, nvcc 13.0 -arch=sm_121: an LDG whose address
 * register is overwritten while in flight gets a read barrier and the overwriting IADD.64 waits on it; the first
 * reader of an LDCU.64 descriptor waits on the LDCU's write barrier) and the one the chip taught us twice (#545; the
 * fp8 GEMV hang in omega_bw_em.h, evidence OWN30B-FP8-GEMV-c4006391). None of A..C is observed failing on the chip:
 * they are contract violations that timing has hidden so far.
 *   A  the LDCU.64 descriptor UR4 (write barrier 1) is read by the first LDG with no wait on barrier 1 in between
 *      (matmul generators in omega_blackwell_codegen.c, default control words). Not caused by the #553 encoder bug.
 *   B  matmul_tensor: the LDG / STG address pair R16:R17 is rewritten by the next IMAD.WIDE before the op read it:
 *      the loads only have write barrier 2 and nothing waits on it; the STGs carry no barrier at all (release 0).
 *   C  prime sieve kernels: a second LDG / LDS writes its destination into the first load's address register while
 *      the first load (write barrier 0, no read barrier) is in flight; the pattern of the fp8 GEMV chip hang.
 *   D  (gone) RMSNORM: round the loop the IMAD.WIDE / FMUL rewrite registers the STG (read barrier 1) still read; the
 *      IADD3s scheduled to wait on barrier 1 carried the fixed word 3 (wait 0x10) of the #553 encoder bug. PR #553
 *      (main b11bd1f6) fixed the encoder and the sweep no longer sees them, so their three entries are removed. */
#define ANY_PC UINT32_MAX
typedef struct { const char *family; uint32_t pc; uint8_t kind; uint8_t reg; uint8_t uniform; char cls; int seen; } Known;
static Known g_known[] = {
    { "matmul_i32", ANY_PC, OMEGA_BW_HZ_RAW, 4, 1, 'A', 0 },
    { "matmul_tensor", ANY_PC, OMEGA_BW_HZ_RAW, 4, 1, 'A', 0 },
    { "matmul_tensor_loop", ANY_PC, OMEGA_BW_HZ_RAW, 4, 1, 'A', 0 },
    { "matmul_frag", ANY_PC, OMEGA_BW_HZ_RAW, 4, 1, 'A', 0 },
    { "matmul_frag_block", ANY_PC, OMEGA_BW_HZ_RAW, 4, 1, 'A', 0 },
    { "matmul_tensor", ANY_PC, OMEGA_BW_HZ_WAR, 16, 0, 'B', 0 },
    { "matmul_tensor", ANY_PC, OMEGA_BW_HZ_WAR, 17, 0, 'B', 0 },
    { "elementwise", 25, OMEGA_BW_HZ_WAR, 22, 0, 'C', 0 },   /* PRIME_SIEVE */
    { "elementwise", 26, OMEGA_BW_HZ_WAR, 23, 0, 'C', 0 },
    { "elementwise", 39, OMEGA_BW_HZ_WAR, 21, 0, 'C', 0 },   /* PRIME_SIEVE_SHARED */
    { "elementwise", 25, OMEGA_BW_HZ_WAR, 20, 0, 'C', 0 },   /* PRIME_SIEVE_DIV1 */
    { "elementwise", 26, OMEGA_BW_HZ_WAR, 21, 0, 'C', 0 },
    { "elementwise", 41, OMEGA_BW_HZ_WAR, 19, 0, 'C', 0 },   /* PRIME_SIEVE_SHARED_DIV1 */
    { "elementwise", 26, OMEGA_BW_HZ_WAR, 20, 0, 'C', 0 },   /* PRIME_SIEVE_DIV2 */
    { "elementwise", 27, OMEGA_BW_HZ_WAR, 21, 0, 'C', 0 },
    { "elementwise", 42, OMEGA_BW_HZ_WAR, 19, 0, 'C', 0 },   /* PRIME_SIEVE_SHARED_DIV2 */
    { NULL, 0, 0, 0, 0, 0, 0 }
};

static uint64_t fnv(const uint8_t *p, size_t n) { uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ull; } return h; }

static void disasm_cross_check(const char *family, const char *variant, const uint8_t *code, size_t len);

static void sweep_bytes(const char *family, const char *variant, const uint8_t *code, size_t len) {
    g_variants++;
    uint64_t h = fnv(code, len);
    for (size_t i = 0; i < g_nseen; i++) if (g_seen[i].hash == h) return; /* identical bytes already checked */
    int rc = omega_bw_hazard_check(code, len, &g_rep);
    if (g_nseen < MAX_SEEN) { Seen *s = &g_seen[g_nseen++]; s->family = family; snprintf(s->variant, sizeof s->variant, "%s", variant); s->hash = h; s->findings = g_rep.count; }
    if (rc != 0) {
        g_unchecked++;
        OmegaBwInsn d; long long pc = g_rep.bad_pc; uint32_t w0 = 0;
        if (pc >= 0) memcpy(&w0, code + pc * 16, 4);
        CHECK(0, "%s %s: not checked (instruction %lld, word 0 %08x, decode %d)", family, variant, pc, w0, pc >= 0 ? omega_bw_decode(code + pc * 16, &d) : -9);
        return;
    }
    for (size_t i = 0; i < g_rep.count && i < OMEGA_BW_HAZARD_MAX; i++) {
        const OmegaBwHazard *f = &g_rep.f[i]; int known = 0;
        for (size_t k = 0; g_known[k].family; k++)
            if (!strcmp(g_known[k].family, family) && (g_known[k].pc == ANY_PC || g_known[k].pc == f->pc) && g_known[k].kind == f->kind &&
                g_known[k].reg == f->reg && g_known[k].uniform == f->uniform) { g_known[k].seen = 1; known = g_known[k].cls; }
        printf("FINDING %s %s: IR %u %s %sR%u release 0x%x (last producer/reader IR %u) [%s %c]\n", family, variant, f->pc, omega_bw_hazard_kind(f->kind),
               f->uniform ? "U" : "", f->reg, f->release, f->src_pc, known ? "triaged" : "UNTRIAGED", known ? known : '-');
        CHECK(known, "%s %s: untriaged %s at IR %u", family, variant, omega_bw_hazard_kind(f->kind), f->pc);
    }
    g_findings_total += g_rep.count;
    if (g_dump_dir && g_rep.count) {
        char path[512]; snprintf(path, sizeof path, "%s/%s-%zu.bin", g_dump_dir, family, g_nseen - 1);
        /* 0644, not fopen's 0666 & ~umask: a dump is never world-writable */
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        FILE *f = fd >= 0 ? fdopen(fd, "wb") : NULL;
        if (fd >= 0 && !f) close(fd);
        if (f) { fwrite(code, 1, len, f); fclose(f); printf("DUMP %s = %s %s\n", path, family, variant); }
    }
    if (g_disasm) disasm_cross_check(family, variant, code, len);
}

/* encode + the launcher's patch steps */
enum { P_NONE = 0, P_TC = 1, P_QMMA = 2 };
static void sweep_prog(const char *family, const char *variant, int patch) {
    size_t len = 0;
    if (omega_bw_encode_program(&g_prog, g_code, sizeof g_code, &len) != 0) { CHECK(0, "%s %s: encode failed", family, variant); return; }
    if ((patch & P_TC) && omega_tc_patch_code(&g_prog, g_code, len) < 0) { CHECK(0, "%s %s: tc patch failed", family, variant); return; }
    if ((patch & P_QMMA) && omega_qmma_patch_code(&g_prog, g_code, len) < 0) { CHECK(0, "%s %s: qmma patch failed", family, variant); return; }
    sweep_bytes(family, variant, g_code, len);
}
#define BUILD(call) (memset(&g_prog, 0, sizeof g_prog), (call) == 0)

static void sweep(void) {
    char v[96];
    /* fp8 GEMV (MoE experts, attention projections): k values of the live models, every buildable lane / rows / warps / depth */
    { unsigned ks[] = { 256, 768, 2048, 4096, 8192 };
      for (unsigned a = 0; a < 5; a++) for (unsigned lb = 8; lb <= 16; lb += 8) for (unsigned rows = 1; rows <= 8; rows *= 2)
      for (unsigned w = 1; w <= 32; w *= 2) for (unsigned dp = 0; dp <= 8; dp += dp ? dp : 1) {
          OmegaFp8GemvSpec s = { .k = ks[a], .lane_bytes = lb, .rows = rows, .warps = w, .depth = dp };
          snprintf(v, sizeof v, "k%u lane%u rows%u warps%u depth%u", ks[a], lb, rows, w, dp);
          if (BUILD(omega_blackwell_codegen_fp8_gemv_prog(&s, &g_prog))) sweep_prog("fp8_gemv", v, P_NONE);
          if (BUILD(omega_blackwell_codegen_fp8_gemv2_prog(&s, &g_prog))) sweep_prog("fp8_gemv2", v, P_NONE);
      } }
    /* bf16 GEMV (lm_head) */
    { unsigned ks[] = { 256, 2048, 4096, 5120 };
      for (unsigned a = 0; a < 4; a++) for (unsigned rows = 1; rows <= 4; rows *= 2) for (unsigned w = 1; w <= 32; w *= 2) for (unsigned dp = 1; dp <= 16; dp *= 2) {
          OmegaBf16GemvSpec s = { .k = ks[a], .rows = rows, .warps = w, .depth = dp };
          snprintf(v, sizeof v, "k%u rows%u warps%u depth%u", ks[a], rows, w, dp);
          if (BUILD(omega_blackwell_codegen_bf16_gemv_prog(&s, &g_prog))) sweep_prog("bf16_gemv", v, P_NONE);
      } }
    /* fp8 grouped GEMM: the incumbent, rowmap (the stroke's expert form), x_ahead (L312 lever 3, scoreboard 5), row_ahead
     * (DRA-79, scoreboard 4 held across the row loop; one K step only: k 256 at lane 8, k 512 at lane 16) and x_resident
     * (DRA-80/81/89: x' staged once per pass, scoreboard 3 held across rows and passes) */
    { unsigned ks[] = { 512, 768, 2048, 4096, 256 };
      for (unsigned a = 0; a < 5; a++) for (unsigned lb = 8; lb <= 16; lb += 8) for (unsigned mt = 1; mt <= 16; mt *= 2) for (unsigned w = 1; w <= 32; w *= 2)
      for (unsigned rm = 0; rm <= 1; rm++) for (unsigned xa = 0; xa <= 1; xa++) for (unsigned ra = 0; ra <= 1; ra++) for (unsigned xr = 0; xr <= 1; xr++) {
          OmegaFp8GemmSpec s = { .k = ks[a], .lane_bytes = lb, .mt = mt, .warps = w, .rowmap = rm, .x_ahead = xa, .row_ahead = ra, .x_resident = xr };
          snprintf(v, sizeof v, "k%u lane%u mt%u warps%u%s%s%s%s", ks[a], lb, mt, w, rm ? " rowmap" : "", xa ? " x_ahead" : "", ra ? " row_ahead" : "", xr ? " x_resident" : "");
          if (BUILD(omega_blackwell_codegen_fp8gemm_prog(&s, &g_prog))) sweep_prog("fp8gemm", v, P_NONE);
      } }
    /* ternary GEMV */
    { unsigned ks[] = { 256, 2048, 4096, 8192, 16384 };
      for (unsigned a = 0; a < 5; a++) for (unsigned lay = 0; lay <= 1; lay++) for (unsigned rows = 1; rows <= 4; rows *= 2) for (unsigned w = 1; w <= 32; w *= 2)
      for (unsigned dp = 1; dp <= 16; dp *= 2) for (unsigned epi = 0; epi <= 1; epi++) {
          OmegaTernGemvSpec s = { ks[a], lay, rows, w, dp, epi, 0 };
          snprintf(v, sizeof v, "k%u layout%u rows%u warps%u depth%u epi%u", ks[a], lay, rows, w, dp, epi);
          if (BUILD(omega_blackwell_codegen_tern_gemv_prog(&s, &g_prog))) sweep_prog("tern_gemv", v, P_NONE);
      } }
    /* router, decode attention */
    for (unsigned w = 8; w <= 32; w *= 2) for (unsigned dp = 1; dp <= 8; dp *= 2) {
        OmegaRouterSpec s = { .warps = w, .depth = dp }; snprintf(v, sizeof v, "warps%u depth%u", w, dp);
        if (BUILD(omega_blackwell_codegen_router_prog(&s, &g_prog))) sweep_prog("router", v, P_NONE);
    }
    for (unsigned dp = 1; dp <= 4; dp++) {
        OmegaDecodeAttnSpec s = { dp, 0 }; snprintf(v, sizeof v, "depth%u", dp);
        if (BUILD(omega_blackwell_codegen_decode_attn_prog(&s, &g_prog))) sweep_prog("decode_attn", v, P_NONE);
    }
    /* decode glue kernels (command list / decode graph) */
    { struct { OmegaDecKind k; uint32_t mode, stride; } d[] = {
          { OMEGA_DEC_AXPY, 0, 256 }, { OMEGA_DEC_AXPY, 0, 512 }, { OMEGA_DEC_AXPY, 0, 2048 }, { OMEGA_DEC_AXPY, 0, 12288 },
          { OMEGA_DEC_SWIGLU, 0, 0 }, { OMEGA_DEC_ARGMAX, 0, 0 }, { OMEGA_DEC_ARGMAX, 1, 0 } };
      for (unsigned i = 0; i < sizeof d / sizeof *d; i++) {
          OmegaDecSpec s; memset(&s, 0, sizeof s); s.kind = d[i].k; s.mode = d[i].mode; s.stride = d[i].stride;
          snprintf(v, sizeof v, "kind%d mode%u stride%u", (int)d[i].k, d[i].mode, d[i].stride);
          if (BUILD(omega_blackwell_codegen_dec_prog(&s, &g_prog))) sweep_prog("dec", v, P_NONE);
      }
      for (uint32_t mode = 0; mode < 8; mode++) {
          OmegaDecSpec s; memset(&s, 0, sizeof s); s.kind = OMEGA_DEC_RES; s.mode = mode;
          snprintf(v, sizeof v, "RES mode%u", mode);
          if (BUILD(omega_blackwell_codegen_dec_prog(&s, &g_prog))) sweep_prog("dec", v, P_NONE);
      } }
    /* tensor-core GEMM and probes */
    { unsigned ks[] = { 128, 768, 2048, 4096, 16384 };
      for (unsigned a = 0; a < 5; a++) for (unsigned rt = 1; rt <= 4; rt *= 2) for (unsigned ct = 1; ct <= 4; ct *= 2) for (unsigned w = 1; w <= 8; w *= 2) {
          OmegaTcGemmSpec s = { ks[a], rt, ct, w, 0 }; snprintf(v, sizeof v, "k%u rt%u ct%u warps%u", ks[a], rt, ct, w);
          if (BUILD(omega_blackwell_codegen_tcgemm_prog(&s, &g_prog))) sweep_prog("tcgemm", v, P_NONE);
      } }
    if (BUILD(omega_blackwell_codegen_tc_probe_prog(0, &g_prog))) sweep_prog("tc_probe", "mutant0", P_TC);
    for (unsigned ch = 1; ch <= 16; ch *= 2) for (unsigned st = 1; st <= 4; st++) {
        snprintf(v, sizeof v, "iters5 chains%u stall%u threads64", ch, st);
        if (BUILD(omega_blackwell_codegen_tc_peak_prog(5, ch, st, 64, &g_prog))) sweep_prog("tc_peak", v, P_TC);
        snprintf(v, sizeof v, "iters64 chains%u stall%u threads128", ch, st);
        if (BUILD(omega_blackwell_codegen_qmma_peak_prog(64, ch, st, 128, &g_prog))) sweep_prog("qmma_peak", v, P_QMMA);
    }
    for (unsigned g = 1; g <= 8; g *= 2) for (unsigned t = 32; t <= 512; t *= 2) {
        snprintf(v, sizeof v, "iters16 groups%u threads%u", g, t);
        if (BUILD(omega_blackwell_codegen_tc_bw_prog(16, g, t, &g_prog))) sweep_prog("tc_bw", v, P_NONE);
    }
    for (unsigned nops = 0; nops <= 8; nops += 4) {
        snprintf(v, sizeof v, "nops%u", nops);
        if (BUILD(omega_blackwell_codegen_qmma_probe_prog(0, nops, &g_prog))) sweep_prog("qmma_probe", v, P_QMMA);
    }
    if (BUILD(omega_blackwell_codegen_f2e4_probe_prog(&g_prog))) sweep_prog("f2e4_probe", "-", P_QMMA);
    /* prefill module kernels (omega_gpu_prefill.c applies both patches to every one of them) */
    { unsigned ks[] = { 128, 768, 2048, 4096, 16384 };
      for (unsigned a = 0; a < 5; a++) for (unsigned rt = 2; rt <= 8; rt *= 2) for (unsigned w = 4; w <= 16; w *= 2) for (unsigned aq = 1; aq <= 2; aq++)
      for (unsigned epi = 0; epi <= 1; epi++) for (unsigned dir = 0; dir <= 1; dir++) {
          OmegaTc2Spec s = { ks[a], rt, 2, w, aq, epi, dir, 0, 0 };
          s.n = 4u * w * 8u * 2u; if (epi) s.n *= 2u;
          snprintf(v, sizeof v, "k%u rt%u warps%u aqn%u epi%u direct%u n%u", ks[a], rt, w, aq, epi, dir, s.n);
          if (BUILD(omega_blackwell_codegen_tc2_prog(&s, &g_prog))) sweep_prog("tc2", v, P_TC | P_QMMA);
      }
      for (unsigned a = 0; a < 5; a++) for (unsigned w = 8; w <= 16; w *= 2) for (unsigned epi = 0; epi <= 1; epi++) for (unsigned dir = 0; dir <= 1; dir++)
      for (unsigned wt = 0; wt <= 120; wt += 60) {
          OmegaQgSpec s = { ks[a], 4, w, epi, dir, 512, wt, 0 };
          snprintf(v, sizeof v, "k%u warps%u epi%u direct%u n512 wait%u", ks[a], w, epi, dir, wt);
          if (BUILD(omega_blackwell_codegen_qgemm_prog(&s, &g_prog))) sweep_prog("qgemm", v, P_TC | P_QMMA);
      }
      for (unsigned a = 0; a < 4; a++) for (unsigned it = 0; it <= 2; it++) {
          OmegaPqSpec s = { .k = ks[a], .in_type = it }; snprintf(v, sizeof v, "k%u in_type%u", ks[a], it);
          if (BUILD(omega_blackwell_codegen_pq_prog(&s, &g_prog))) sweep_prog("pq", v, P_TC | P_QMMA);
      } }
    { unsigned ks[] = { 768, 2048, 4096 }, es[] = { 8, 64, 128 }, ts[] = { 1, 16, 128, 1024 };
      for (unsigned a = 0; a < 3; a++) for (unsigned b = 0; b < 3; b++) for (unsigned c = 0; c < 4; c++) {
          OmegaRlSpec s = { ks[a], es[b], ts[c], 0 }; snprintf(v, sizeof v, "k%u e%u t%u", ks[a], es[b], ts[c]);
          if (BUILD(omega_blackwell_codegen_rl_prog(&s, &g_prog))) sweep_prog("rl", v, P_TC | P_QMMA);
      }
      unsigned hs[] = { 768, 2048, 4096 };
      for (unsigned a = 0; a < 3; a++) for (unsigned tk = 1; tk <= 8; tk *= 2) {
          OmegaMcSpec s = { hs[a], tk, 0 }; snprintf(v, sizeof v, "h%u top_k%u", hs[a], tk);
          if (BUILD(omega_blackwell_codegen_mc_prog(&s, &g_prog))) sweep_prog("mc", v, P_TC | P_QMMA);
      } }
    if (BUILD(omega_blackwell_codegen_pf_copy_prog(&g_prog))) sweep_prog("pf_copy", "-", P_TC | P_QMMA);
    /* prefill attention: the API's own host encode (build + encode + tc patch) */
    for (unsigned w = 4; w <= 8; w *= 2) for (unsigned md = 0; md <= 2; md++) for (unsigned mir = 0; mir <= 1; mir++) {
        OmegaPrefillAttnSpec s = { w, 0, md, mir };
        static uint8_t pa[OMEGA_BW_CODE_CAP_MAX + 4096]; uint32_t gp = 0; int np = 0;
        size_t len = omega_gpu_prefill_attn_encode(&s, pa, sizeof pa, &gp, &np);
        snprintf(v, sizeof v, "warps%u mode%u mirror%u", w, md, mir);
        if (len) sweep_bytes("prefill_attn", v, pa, len);
    }
    /* gqa / paged attention (FB-1 cut 5): the API's codegen */
    for (int f32 = 0; f32 <= 1; f32++) for (uint32_t lb = 0; lb <= 8; lb++) for (uint32_t lg = 0; lg <= 4; lg++) for (uint32_t hd = 64; hd <= 128; hd *= 2) {
        OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
        if (omega_gpu_attention_codegen_hd(f32 != 0, lb, lg, hd, &k) != 0 || !k.code) { free(k.code); continue; }
        snprintf(v, sizeof v, "kv_f32 %d log2_block %u log2_gqa %u hd %u", f32, lb, lg, hd);
        sweep_bytes("attention", v, k.code, k.code_size);
        free(k.code);
    }
    /* elementwise kernels and the reconvergence probes */
    for (int op = OMEGA_GPU_EW_EX2; op <= OMEGA_GPU_EW_RC_REFUSE_JOIN_ONLY; op++) {
        OmegaBlackwellKernel k; memset(&k, 0, sizeof k);
        if (omega_gpu_elementwise_codegen_ir((OmegaGpuEwOp)op, 0, &g_prog, &k) != 0 || !k.code) { free(k.code); continue; }
        snprintf(v, sizeof v, "op%d", op);
        sweep_bytes("elementwise", v, k.code, k.code_size);
        free(k.code);
    }
    /* matmul: integer, single-tile tensor, looped tensor, fragment, blocked fragment */
    { OmegaMatMulSpec s;
      uint32_t i32[][3] = { {16,16,16}, {32,16,64}, {16,32,16}, {64,32,64} };
      for (unsigned i = 0; i < 4; i++) if (omega_matmul_spec_init(&s, i32[i][0], i32[i][1], i32[i][2], OMEGA_MATMUL_PRECISION_INT32) == 0) {
          snprintf(v, sizeof v, "m%u k%u n%u", i32[i][0], i32[i][1], i32[i][2]);
          if (BUILD(omega_blackwell_codegen_matmul_i32_prog(&s, &g_prog))) sweep_prog("matmul_i32", v, P_NONE);
      }
      for (int pr = 0; pr < 2; pr++) {
          OmegaMatMulPrecision p = pr ? OMEGA_MATMUL_PRECISION_FP16 : OMEGA_MATMUL_PRECISION_BF16;
          if (omega_matmul_spec_init(&s, 16, 16, 8, p) == 0 && BUILD(omega_blackwell_codegen_matmul_tensor_prog(&s, &g_prog))) sweep_prog("matmul_tensor", pr ? "fp16" : "bf16", P_NONE);
          uint32_t kk[] = { 16, 64, 2048 }, nn[] = { 8, 64, 2048 };
          for (unsigned a = 0; a < 3; a++) for (unsigned b = 0; b < 3; b++) for (uint32_t gx = 1; gx <= 8; gx *= 2) {
              if (omega_matmul_spec_init(&s, 16, kk[a], nn[b], p) != 0) continue;
              snprintf(v, sizeof v, "%s k%u n%u grid_x%u", pr ? "fp16" : "bf16", kk[a], nn[b], gx);
              if (BUILD(omega_blackwell_codegen_matmul_tensor_loop_prog(&s, gx, 0, &g_prog))) sweep_prog("matmul_tensor_loop", v, P_NONE);
          }
      }
      unsigned kps[] = { 768, 2048, 4096 }, nps[] = { 8, 128, 2048 };
      for (unsigned a = 0; a < 3; a++) for (unsigned b = 0; b < 3; b++) {
          if (omega_matmul_spec_init(&s, 16, kps[a], nps[b], OMEGA_MATMUL_PRECISION_BF16) != 0) continue;
          for (uint32_t un = 1; un <= 4; un *= 2) {
              snprintf(v, sizeof v, "k%u n%u unroll%u", kps[a], nps[b], un);
              if (BUILD(omega_blackwell_codegen_matmul_frag_prog(&s, 48, un, 0, &g_prog))) sweep_prog("matmul_frag", v, P_NONE);
          }
          for (unsigned rt = 1; rt <= 8; rt *= 2) for (unsigned ct = 1; ct <= 4; ct *= 2) for (uint32_t un = 1; un <= 2; un++) {
              if (rt * ct < 2) continue;
              snprintf(v, sizeof v, "k%u n%u rt%u ct%u unroll%u", kps[a], nps[b], rt, ct, un);
              if (BUILD(omega_blackwell_codegen_matmul_frag_block_prog(&s, 1, un, rt, ct, 0, &g_prog))) sweep_prog("matmul_frag_block", v, P_NONE);
          }
      } }
    /* the command list's no-op kernel: EXIT; BRA self */
    { omega_bw_ir_init(&g_prog);
      BlackwellIRInsn ex = mk(BW_IR_EXIT, -1, -1, -1, -1, -1, 0), self = mk(BW_IR_BRA, -1, -1, -1, -1, -1, 0x000fc000u);
      if (omega_bw_ir_append(&g_prog, &ex) >= 0 && omega_bw_ir_append(&g_prog, &self) >= 0 && omega_bw_regalloc_solve(&g_prog) == 0) sweep_prog("cmdlist_nop", "-", P_NONE); }
}

/* ----------------------------------------------------------------------------------- disassembler cross-check (C6) */
static size_t g_dis_insns, g_dis_bad, g_dis_kernels;
static int decoded_has(const OmegaBwInsn *d, int reg, int uniform) {
    for (unsigned j = 0; j < d->nr; j++) if (d->rd[j].uniform == uniform && d->rd[j].reg == reg) return 1;
    for (unsigned j = 0; j < d->nw; j++) if (d->wr[j].uniform == uniform && d->wr[j].reg == reg) return 1;
    return 0;
}
static void disasm_cross_check(const char *family, const char *variant, const uint8_t *code, size_t len) {
    size_t seen = 0; g_dis_kernels++;
    for (size_t pc = 0; pc * 16u + 16u <= len; pc++) {
        const uint8_t *w = code + pc * 16u;
        OmegaBwInsn d; if (omega_bw_decode(w, &d) != 0) continue; /* unreachable padding words are not decoded either */
        seen++; g_dis_insns++;
        uint64_t lo = 0, hi = 0;
        for (int i = 7; i >= 0; i--) { lo = lo << 8 | w[i]; hi = hi << 8 | w[8 + i]; }
        char t[256];
        int rc = omega_sm121_disasm(lo, hi, pc * 16u, t, sizeof t);
        if (rc != 0) { g_dis_bad++; if (g_dis_bad <= 20) printf("DISASM UNKNOWN %s %s IR %zu: decoder %s, disassembler rc %d lo %016llx hi %016llx\n", family, variant, pc, d.name, rc, (unsigned long long)lo, (unsigned long long)hi); continue; }
        /* every register token the disassembler prints must be a base register the decoder has, and vice versa */
        int nvr[16], nvu[16], nn = 0, nu = 0;
        for (char *q = t; *q; q++) {
            int uni = 0; char *r = q;
            if (r[0] == 'U' && r[1] == 'R') { uni = 1; r++; } else if (q > t && (q[-1] == 'U' || q[-1] == 'S' || q[-1] == '_')) continue;
            if (r[0] != 'R' || r[1] < '0' || r[1] > '9') continue;
            if (q > t && ((q[-1] >= 'A' && q[-1] <= 'Z') || (q[-1] >= 'a' && q[-1] <= 'z') || q[-1] == '_')) continue;
            int reg = atoi(r + 1);
            if (uni) { if (nu < 16) nvu[nu++] = reg; } else { if (nn < 16) nvr[nn++] = reg; }
            q = r + 1;
        }
        int ok = 1;
        for (int i = 0; i < nn && ok; i++) ok = decoded_has(&d, nvr[i], 0);
        for (int i = 0; i < nu && ok; i++) ok = decoded_has(&d, nvu[i], 1);
        for (unsigned j = 0; j < d.nr + d.nw && ok; j++) {
            const OmegaBwOperand *o = j < d.nr ? &d.rd[j] : &d.wr[j - d.nr]; int f2 = 0;
            for (int i = 0; i < (o->uniform ? nu : nn); i++) if ((o->uniform ? nvu[i] : nvr[i]) == o->reg) f2 = 1;
            ok = f2;
        }
        if (!ok) { g_dis_bad++; if (g_dis_bad <= 20) printf("DISASM MISMATCH %s %s IR %zu: '%s' vs decoder %s (%u reads, %u writes)\n", family, variant, pc, t, d.name, d.nr, d.nw); }
    }
    CHECK(seen > 0, "%s %s: disassembled %zu instructions", family, variant, seen);
}

int main(int argc, char **argv) {
    const char *dir = "tests/fixtures/bw_hazard";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--disasm")) g_disasm = 1;
        else if (!strcmp(argv[i], "--fixtures") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) g_dump_dir = argv[++i];
    }
    test_decoder_vs_encoder();
    test_red_receipt(dir);
    test_controls();
    sweep();
    size_t clean = 0;
    printf("sweep per generator (distinct kernels, clean, findings):\n");
    for (size_t i = 0; i < g_nseen; i++) {
        clean += g_seen[i].findings == 0;
        int first = 1;
        for (size_t j = 0; j < i && first; j++) if (!strcmp(g_seen[j].family, g_seen[i].family)) first = 0;
        if (!first) continue;
        size_t fk = 0, fc = 0, fv = 0;
        for (size_t j = i; j < g_nseen; j++)
            if (!strcmp(g_seen[j].family, g_seen[i].family)) { fk++; fc += g_seen[j].findings == 0; fv += g_seen[j].findings; }
        printf("  %-20s %4zu distinct, %4zu clean, %4zu findings\n", g_seen[i].family, fk, fc, fv);
    }
    for (size_t k = 0; g_known[k].family; k++)
        if (!g_known[k].seen) printf("NOTE: triaged finding %s IR %d %s %sR%u (class %c) no longer seen\n", g_known[k].family, g_known[k].pc == ANY_PC ? -1 : (int)g_known[k].pc,
                                      omega_bw_hazard_kind(g_known[k].kind), g_known[k].uniform ? "U" : "", g_known[k].reg, g_known[k].cls);
    printf("sweep: %zu variants built, %zu distinct kernels, %zu clean, %zu with findings, %zu findings, %zu not checked\n",
           g_variants, g_nseen, clean, g_nseen - clean - g_unchecked, g_findings_total, g_unchecked);
    CHECK(g_nseen >= 50, "the sweep built only %zu distinct kernels", g_nseen);
    if (g_disasm) {
        printf("disassembler cross-check (sm121dis, in process): %zu kernels, %zu instructions, %zu mismatched\n", g_dis_kernels, g_dis_insns, g_dis_bad);
        CHECK(g_dis_bad == 0 && g_dis_insns > 0, "disassembler cross-check");
    }
    printf("bw_hazard_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
