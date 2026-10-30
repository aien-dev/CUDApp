/* omega #492: IMAD.HI.U32 with a register Rc through one host warp simulator (host-only, no chip).
 *
 * The chip reads the Rc of IMAD.HI.U32 as a 64-bit register pair, not as one 32-bit word.
 * Receipt (manufacturer's compiler, not a chip run): nvcc 13.0 for sm_121 emits, for
 *   hi32(a * b + c64)        IMAD.HI.U32 R7, R7, R6, R2      with R2:R3 = c64 (no separate add),
 *   hi32(a * b + (u64)c32)   UMOV UR5, URZ ; IMAD.HI.U32 R5, R7, R6, UR4   (upper half zeroed first),
 *   __umulhi(a, b) + c32     Rc pair = {lo 0, hi c32}, so the add lands in the high word.
 * Listing and source: evidence/IMADHI-RC-ORACLE-nvcc13.0-20261016/. The first version of the
 * trainer GEMM (PR #487) used IMAD.HI.U32 with a register Rc, passed the simulators (32-bit Rc)
 * and gave wrong odd columns on the chip. A chip probe of the exact pair semantics is still
 * pending (lane E3), so the simulators refuse a non-RZ Rc instead of modelling it.
 *
 * This file is compiled once per simulator header: -DSIM_HEADER='"bw_warp_sim_k.h"' and so on
 * (Makefile target test-gpu-imadhi-rc). The four headers share their static names, so one
 * translation unit holds one simulator.
 *
 * Checks (each runs the one-instruction kernel on one warp and stores Rd):
 *   1. Rc = RZ: the simulator runs and Rd = hi32(Ra * Rb)        (control, the one live user today)
 *   2. Rc = register: the encoder emits a non-RZ Rc (the chip would read a pair) and the
 *      simulator refuses with WS_E_IMADHI_RC naming the pc. Before the fix it returned
 *      hi32(Ra * Rb) + Rc.lo, the 32-bit model the chip does not implement.
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

enum { V_OUT = 0, V_A, V_B, V_C, V_D, NV };
enum { CB_OUT = 0x380 };

static void ins(BlackwellIRProgram *p, BlackwellIROpcode op, int d, int a, int b, int c, uint32_t imm, uint8_t dsub) {
    BlackwellIRInsn *i = &p->insns[p->count++];
    memset(i, 0, sizeof *i);
    i->op = op; i->dst_vreg = d; i->src1_vreg = a; i->src2_vreg = b; i->src3_vreg = c; i->ureg = -1; i->imm = imm; i->dst_subreg = dsub;
}

/* out[tid] = IMAD.HI.U32(a, b, rc ? {chi:clo} : RZ) */
static void build(BlackwellIRProgram *p, uint32_t a, uint32_t b, uint32_t clo, uint32_t chi, int rc) {
    memset(p, 0, sizeof *p);
    ins(p, BW_IR_LDC64, V_OUT, -1, -1, -1, CB_OUT, 0);
    ins(p, BW_IR_MOV_IMM, V_A, -1, -1, -1, a, 0);
    ins(p, BW_IR_MOV_IMM, V_B, -1, -1, -1, b, 0);
    ins(p, BW_IR_MOV_IMM, V_C, -1, -1, -1, clo, 0);
    ins(p, BW_IR_MOV_IMM, V_C, -1, -1, -1, chi, 1);
    ins(p, BW_IR_IMAD_HI_U32, V_D, V_A, V_B, rc ? V_C : -1, 0, 0);
    ins(p, BW_IR_STG_E, -1, V_OUT, V_D, -1, 0, 0);
    ins(p, BW_IR_EXIT, -1, -1, -1, -1, 0, 0);
    /* physical registers: out pair R2:R3, a R4, b R5, c pair R6:R7 (even-aligned), d R8 */
    static const int phys[NV] = { 2, 4, 5, 6, 8 };
    p->regalloc.num_vregs = NV;
    for (int v = 0; v < NV; v++) p->regalloc.vreg_to_phys[v] = phys[v];
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

int main(void) {
    const uint32_t A = 0xdeadbeefu, B = 0x10000u, CLO = 0x1234u, CHI = 0x5678u; /* B = 2^16 is the tgemm pack */
    const uint32_t hi_ab = (uint32_t)(((uint64_t)A * B) >> 32);
    const uint32_t pair_model = (uint32_t)((((uint64_t)A * B) + (((uint64_t)CHI << 32) | CLO)) >> 32);
    int fails = 0; char msg[200]; uint32_t out;
    BlackwellIRProgram *p = calloc(1, sizeof *p);
    printf("sim %s: hi32(a*b) 0x%08x, nvcc pair model 0x%08x (chip probe pending), old 32-bit model 0x%08x\n",
           SIM_HEADER, hi_ab, pair_model, hi_ab + CLO);

    /* 1. Rc = RZ: runs, Rd = hi32(a * b) */
    build(p, A, B, CLO, CHI, 0); out = 0xffbadbadu;
    int err = run(p, &out, msg, sizeof msg);
    if (err || out != hi_ab) { fails++; printf("FAIL rc=RZ: err %d (%s) out 0x%08x want 0x%08x\n", err, msg, out, hi_ab); }
    else printf("PASS rc=RZ: Rd = hi32(a*b) = 0x%08x\n", out);

    /* 2. Rc = register pair: the word carries a non-RZ Rc, the simulator must refuse */
    build(p, A, B, CLO, CHI, 1);
    BlackwellIRInsn e = p->insns[5]; e.dst_vreg = 8; e.src1_vreg = 4; e.src2_vreg = 5; e.src3_vreg = 6; /* physical numbers for the encoder */
    uint32_t w[4] = { 0, 0, 0, 0 };
    if (omega_blackwell_encode_one(&e, w) != 0 || (w[0] & 0xffffu) != 0x7227u || (w[2] & 0xffu) != 6u) {
        fails++; printf("FAIL encode: IMAD.HI.U32 R8, R4, R5, R6 -> %08x %08x %08x %08x (want opcode 7227, Rc byte 06)\n", w[0], w[1], w[2], w[3]);
    } else printf("PASS encode: IMAD.HI.U32 R8, R4, R5, R6 carries Rc = R6 (the chip reads R6:R7)\n");
    out = 0xffbadbadu;
    err = run(p, &out, msg, sizeof msg);
    if (err != WS_E_IMADHI_RC) {
        fails++;
        printf("FAIL rc=R6: simulator did not refuse (err %d%s%s), out 0x%08x: a 32-bit Rc model the chip does not implement\n",
               err, err ? " " : "", err ? msg : "", out);
    } else if (!strstr(msg, "IMAD.HI.U32") || !strstr(msg, "at pc 5")) {
        fails++; printf("FAIL rc=R6: refused but the message does not name the instruction and pc: %s\n", msg);
    } else printf("PASS rc=R6: refused: %s\n", msg);

    free(p);
    printf("%s %s\n", fails ? "FAIL" : "PASS", SIM_HEADER);
    return fails ? 1 : 0;
}
