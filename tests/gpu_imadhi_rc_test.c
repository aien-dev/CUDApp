/* omega #492: IMAD.HI.U32 with a register Rc through one host warp simulator (host-only, no chip).
 *
 * The chip reads the Rc of IMAD.HI.U32 as a 64-bit register pair, not as one 32-bit word.
 * Receipt (manufacturer's compiler, not a chip run): nvcc 13.0 for sm_121 emits, for
 *   hi32(a * b + c64)        IMAD.HI.U32 R7, R7, R6, R2      with R2:R3 = c64 (no separate add),
 *   hi32(a * b + (u64)c32)   UMOV UR5, URZ ; IMAD.HI.U32 R5, R7, R6, UR4   (upper half zeroed first),
 *   __umulhi(a, b) + c32     HFMA2 R4 = 0 ; R5 = c32 ; IMAD.HI.U32 R5, R8, R9, R4   (pair {lo 0, hi c32}),
 * so the low word is Rc and the high word Rc+1, and every base is even. Listing and source:
 * evidence/IMADHI-RC-ORACLE-nvcc13.0-20261016/. The first version of the trainer GEMM (PR #487)
 * packed the odd-column B fragment with IMAD.HI.U32 O, W0, 0x10000, tw (meaning (W0 >> 16) + tw),
 * passed the simulators (then hi32(Ra * Rb) + Rc) and gave wrong, non-repeatable odd columns on
 * the chip (the stale upper register of the pair); commit 5b10b824 replaced it with SHF.R + IADD3.
 * The simulators now model an even Rc as the pair and refuse an odd Rc (no evidence for it).
 * The pair model is the compiler's reading, not chip-probed.
 *
 * This file is compiled once per simulator header: -DSIM_HEADER='"bw_warp_sim_k.h"' and so on
 * (Makefile target test-gpu-imadhi-rc). The four headers share their static names, so one
 * translation unit holds one simulator.
 *
 * Checks (each runs a straight-line kernel on one warp and stores one word per lane):
 *   1. Rc = RZ: Rd = hi32(Ra * Rb)                                  (the one live user today)
 *   2. Rc = R6 (even): the encoder emits Rc byte 06 and Rd = hi32(Ra * Rb + R7:R6)
 *   3. Rc = R7 (odd): refused with WS_E_IMADHI_RC naming the pc
 *   4. PR #487 replay: the IMAD.HI.U32 odd-column pack must NOT match the emulation
 *      (W0 >> 16) + (W1 & 0xffff0000); it must give the pair value, for a stale upper register
 *      and for a zero one. The old 32-bit model matched the emulation here: the #487 false pass.
 *   5. The SHF.R + IADD3 pack of commit 5b10b824 matches the emulation (control).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "omega_blackwell_codegen.h"
#ifndef SIM_HEADER
#error "compile with -DSIM_HEADER='\"bw_warp_sim.h\"' (or _tc, _k, _pa)"
#endif
#include SIM_HEADER

enum { V_OUT = 0, V_A, V_B, V_C, V_D, V_W1, V_MASK, V_T, NV };
enum { CB_OUT = 0x380 };

static void ins(BlackwellIRProgram *p, BlackwellIROpcode op, int d, int a, int b, int c, uint32_t imm, uint8_t dsub) {
    BlackwellIRInsn *i = &p->insns[p->count++];
    memset(i, 0, sizeof *i);
    i->op = op; i->dst_vreg = d; i->src1_vreg = a; i->src2_vreg = b; i->src3_vreg = c; i->ureg = -1; i->imm = imm; i->dst_subreg = dsub;
}

/* physical registers: out pair R2:R3, a R4, b R5, c pair R6:R7 (base = c_phys), d R8, w1 R10, mask R11, t R12 */
static void regs(BlackwellIRProgram *p, int c_phys) {
    const int phys[NV] = { 2, 4, 5, c_phys, 8, 10, 11, 12 };
    p->regalloc.num_vregs = NV;
    for (int v = 0; v < NV; v++) p->regalloc.vreg_to_phys[v] = phys[v];
}

/* out[tid] = IMAD.HI.U32(a, b, rc ? {chi:clo} : RZ), Rc at physical c_phys */
static void build(BlackwellIRProgram *p, uint32_t a, uint32_t b, uint32_t clo, uint32_t chi, int rc, int c_phys) {
    memset(p, 0, sizeof *p);
    ins(p, BW_IR_LDC64, V_OUT, -1, -1, -1, CB_OUT, 0);
    ins(p, BW_IR_MOV_IMM, V_A, -1, -1, -1, a, 0);
    ins(p, BW_IR_MOV_IMM, V_B, -1, -1, -1, b, 0);
    ins(p, BW_IR_MOV_IMM, V_C, -1, -1, -1, clo, 0);
    ins(p, BW_IR_MOV_IMM, V_C, -1, -1, -1, chi, 1);
    ins(p, BW_IR_IMAD_HI_U32, V_D, V_A, V_B, rc ? V_C : -1, 0, 0);
    ins(p, BW_IR_STG_E, -1, V_OUT, V_D, -1, 0, 0);
    ins(p, BW_IR_EXIT, -1, -1, -1, -1, 0, 0);
    regs(p, c_phys);
}

/* The #487 odd-column B-fragment pack. tw = W1 & 0xffff0000 lands in R6; R7 holds whatever was there
 * (stale). imadhi = 1: O = IMAD.HI.U32(W0, 0x10000, tw), the first tgemm version.
 * imadhi = 0: O = SHF.R.U32.HI(W0, 16) + tw, the fix in 5b10b824. */
static void build_pack(BlackwellIRProgram *p, uint32_t w0, uint32_t w1, uint32_t stale, int imadhi) {
    memset(p, 0, sizeof *p);
    ins(p, BW_IR_LDC64, V_OUT, -1, -1, -1, CB_OUT, 0);
    ins(p, BW_IR_MOV_IMM, V_C, -1, -1, -1, stale, 1);          /* an earlier value left in R7 */
    ins(p, BW_IR_MOV_IMM, V_A, -1, -1, -1, w0, 0);
    ins(p, BW_IR_MOV_IMM, V_W1, -1, -1, -1, w1, 0);
    ins(p, BW_IR_MOV_IMM, V_MASK, -1, -1, -1, 0xffff0000u, 0);
    ins(p, BW_IR_LOP3_LUT, V_C, V_W1, V_MASK, -1, 0xc0u, 0);   /* tw = W1 & mask (a & b = 0xf0 & 0xcc) */
    if (imadhi) {
        ins(p, BW_IR_MOV_IMM, V_B, -1, -1, -1, 0x10000u, 0);
        ins(p, BW_IR_IMAD_HI_U32, V_D, V_A, V_B, V_C, 0, 0);
    } else {
        ins(p, BW_IR_SHF_R, V_T, V_A, -1, -1, 16u, 0);
        ins(p, BW_IR_IADD3, V_D, V_T, V_C, -1, 0, 0);
    }
    ins(p, BW_IR_STG_E, -1, V_OUT, V_D, -1, 0, 0);
    ins(p, BW_IR_EXIT, -1, -1, -1, -1, 0, 0);
    regs(p, 6);
}

static int run(const BlackwellIRProgram *p, uint32_t *out, char *msg, size_t msglen) {
    WsSim s; memset(&s, 0, sizeof s);
    static uint8_t smem[1024];
    s.p = p; s.threads = 32; s.smem = smem; s.smem_bytes = sizeof smem;
    uint64_t oa = (uintptr_t)out;
    s.cbank[CB_OUT / 4] = (uint32_t)oa; s.cbank[CB_OUT / 4 + 1] = (uint32_t)(oa >> 32);
    s.ranges[0] = (WsRange){ (uintptr_t)oa, (uintptr_t)oa + 4 }; s.nranges = 1;
    int err = ws_run(&s, 1, 1);
    snprintf(msg, msglen, "%s", s.msg);
    return err ? s.err : 0;
}

static uint32_t hi32_pair(uint32_t a, uint32_t b, uint32_t lo, uint32_t hi) {
    return (uint32_t)((((uint64_t)a * b) + (((uint64_t)hi << 32) | lo)) >> 32);
}

int main(void) {
    const uint32_t A = 0xdeadbeefu, B = 0x10000u, CLO = 0x1234u, CHI = 0x5678u; /* B = 2^16 is the tgemm pack */
    const uint32_t hi_ab = (uint32_t)(((uint64_t)A * B) >> 32);
    const uint32_t pair_model = hi32_pair(A, B, CLO, CHI);
    int fails = 0; char msg[200]; uint32_t out;
    BlackwellIRProgram *p = calloc(1, sizeof *p);
    printf("sim %s: hi32(a*b) 0x%08x, nvcc pair model 0x%08x (chip probe pending), old 32-bit model 0x%08x\n",
           SIM_HEADER, hi_ab, pair_model, hi_ab + CLO);

    /* 1. Rc = RZ: runs, Rd = hi32(a * b) */
    build(p, A, B, CLO, CHI, 0, 6); out = 0xffbadbadu;
    int err = run(p, &out, msg, sizeof msg);
    if (err || out != hi_ab) { fails++; printf("FAIL rc=RZ: err %d (%s) out 0x%08x want 0x%08x\n", err, msg, out, hi_ab); }
    else printf("PASS rc=RZ: Rd = hi32(a*b) = 0x%08x\n", out);

    /* 2. Rc = R6 (even): the word carries Rc = R6 and the simulator reads the pair R7:R6 */
    build(p, A, B, CLO, CHI, 1, 6);
    BlackwellIRInsn e = p->insns[5]; e.dst_vreg = 8; e.src1_vreg = 4; e.src2_vreg = 5; e.src3_vreg = 6; /* physical numbers for the encoder */
    uint32_t w[4] = { 0, 0, 0, 0 };
    if (omega_blackwell_encode_one(&e, w) != 0 || (w[0] & 0xffffu) != 0x7227u || (w[2] & 0xffu) != 6u) {
        fails++; printf("FAIL encode: IMAD.HI.U32 R8, R4, R5, R6 -> %08x %08x %08x %08x (want opcode 7227, Rc byte 06)\n", w[0], w[1], w[2], w[3]);
    } else printf("PASS encode: IMAD.HI.U32 R8, R4, R5, R6 carries Rc = R6 (the chip reads R7:R6)\n");
    out = 0xffbadbadu;
    err = run(p, &out, msg, sizeof msg);
    if (err || out != pair_model) {
        fails++; printf("FAIL rc=R6: err %d (%s) out 0x%08x want pair 0x%08x (old 32-bit model 0x%08x)\n", err, msg, out, pair_model, hi_ab + CLO);
    } else printf("PASS rc=R6: Rd = hi32(a*b + R7:R6) = 0x%08x\n", out);

    /* 3. Rc = R7 (odd): no evidence of what the chip reads, refused */
    build(p, A, B, CLO, CHI, 1, 7); out = 0xffbadbadu;
    err = run(p, &out, msg, sizeof msg);
    if (err != WS_E_IMADHI_RC) {
        fails++; printf("FAIL rc=R7: simulator did not refuse an odd Rc (err %d%s%s), out 0x%08x\n", err, err ? " " : "", err ? msg : "", out);
    } else if (!strstr(msg, "IMAD.HI.U32") || !strstr(msg, "at pc 5")) {
        fails++; printf("FAIL rc=R7: refused but the message does not name the instruction and pc: %s\n", msg);
    } else printf("PASS rc=R7: refused: %s\n", msg);

    /* 4 and 5. PR #487 replay: the odd-column pack, stale and zero upper register */
    static const struct { uint32_t w0, w1, stale; } pk[] = {
        { 0x3f80bf80u, 0x40004040u, 0x3f803f80u }, /* bf16 pairs, a live value left in R7 */
        { 0xdeadbeefu, 0x5678cafeu, 0x00000000u }, /* zero upper: still wrong, tw lands in the low word (carry here) */
        { 0x12340001u, 0xabcd0000u, 0x00000000u }, /* zero upper, no carry */
    };
    for (unsigned k = 0; k < sizeof pk / sizeof pk[0]; k++) {
        const uint32_t tw = pk[k].w1 & 0xffff0000u, emul = (pk[k].w0 >> 16) + tw;
        const uint32_t chip = hi32_pair(pk[k].w0, 0x10000u, tw, pk[k].stale);
        build_pack(p, pk[k].w0, pk[k].w1, pk[k].stale, 1); out = 0xffbadbadu;
        err = run(p, &out, msg, sizeof msg);
        if (err || out != chip || out == emul) {
            fails++;
            printf("FAIL pack %u IMAD.HI.U32: err %d (%s) out 0x%08x, want the pair value 0x%08x != emulation 0x%08x%s\n", k, err, msg, out, chip, emul,
                   (!err && out == emul) ? " (matches the emulation: the 32-bit Rc model of the #487 false pass)" : "");
        } else printf("PASS pack %u IMAD.HI.U32: odd column 0x%08x != emulation 0x%08x (stale R7 0x%08x), as on the chip in #487\n", k, out, emul, pk[k].stale);
        build_pack(p, pk[k].w0, pk[k].w1, pk[k].stale, 0); out = 0xffbadbadu;
        err = run(p, &out, msg, sizeof msg);
        if (err || out != emul) { fails++; printf("FAIL pack %u SHF.R+IADD3: err %d (%s) out 0x%08x want 0x%08x\n", k, err, msg, out, emul); }
        else printf("PASS pack %u SHF.R+IADD3: odd column 0x%08x = emulation\n", k, out);
    }

    free(p);
    printf("%s %s\n", fails ? "FAIL" : "PASS", SIM_HEADER);
    return fails ? 1 : 0;
}
