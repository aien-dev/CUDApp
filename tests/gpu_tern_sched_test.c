/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* gpu_tern_sched_test: the I2FP consumer rule of the ternary GEMV generator (Sass#4, 2026-10-27). Host only, no device.
 *
 * I2FP is decoupled on consumer Blackwell (chip sweep lat-20261027T025808Z: i2fp->fmul 18, i2f->fmul 19 cycles), but
 * the generator emitted it as a fixed-latency op with stall 6, so the f32 epilogue's FMUL could read a stale register.
 * The generator now puts I2FP on scoreboard 0 and refuses any program in which an I2FP reader neither waits on that
 * scoreboard nor sits >= OMEGA_TERN_I2FP_LAT cycles behind (omega_tern_i2fp_violations).
 *
 *  1. every production f32 spec builds, has an I2FP, and the I2FP sets a write scoreboard its reader waits on;
 *  2. negative control: mutant 4 (the pre-fix I2FP, stall 6, no scoreboard) is refused for every f32 spec;
 *  3. the checker on hand-built programs: I2FP -> FMUL at stall 6 fails, at 18 fails, at 19 passes, with the
 *     scoreboard wait passes, without it fails; an FMUL producer at stall 6 passes (other producers keep their rule).
 *
 * MIGRATION-PENDING C (GpuStack PLAN, Language rule): a host test of an existing C generator (omega_blackwell_codegen_tern.c).
 * It runs only while CUDAPP_MIGRATION_PENDING_TERN_SCHED is 1 (the default below); set it to 0 when the ternary kernel moves
 * to .om and this file retires (successor: the Omega tern generator's checker, omega tests/gpu_tern_api_test.c). */
#include "omega_blackwell_codegen_tern.h"
#include <stdio.h>
#ifndef CUDAPP_MIGRATION_PENDING_TERN_SCHED
#define CUDAPP_MIGRATION_PENDING_TERN_SCHED 1 /* migration-pending switch (see above) */
#endif
#include <string.h>

#define CW(stall, wbar, rbar, wait) \
    (((uint32_t)(stall) << 9) | ((uint32_t)(wbar) << 14) | ((uint32_t)(rbar) << 17) | ((uint32_t)(wait) << 20))

static int fails;
static void check(int ok, const char *what) { printf("  %-72s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) fails++; }

static BlackwellIRInsn insn(BlackwellIROpcode op, int d, int a, int b, uint32_t control) {
    BlackwellIRInsn i; memset(&i, 0, sizeof i);
    i.op = op; i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src3_vreg = -1; i.ureg = -1; i.control = control;
    return i;
}

/* MOV r0; producer r1 <- r0 (control pc); `pad` independent MOVs (stall 1 each) to r3; FMUL r2 <- r1 * r1 (wait mask
 * cw); STG-free tail EXIT. Returns the checker's count, or -2 when the program cannot be built. */
static long synth(BlackwellIROpcode prod, uint32_t pc, int pad, uint32_t cw) {
    static BlackwellIRProgram p;
    omega_bw_ir_init(&p);
    int r0 = omega_bw_ir_alloc_vreg(&p), r1 = omega_bw_ir_alloc_vreg(&p), r2 = omega_bw_ir_alloc_vreg(&p), r3 = omega_bw_ir_alloc_vreg(&p);
    if (r0 < 0 || r1 < 0 || r2 < 0 || r3 < 0) return -2;
    BlackwellIRInsn m = insn(BW_IR_MOV_IMM, r0, -1, -1, CW(6, 7, 7, 0)); m.imm = 3;
    if (omega_bw_ir_append(&p, &m) < 0) return -2;
    BlackwellIRInsn pr = insn(prod, r1, r0, prod == BW_IR_FMUL ? r0 : -1, pc);
    if (omega_bw_ir_append(&p, &pr) < 0) return -2;
    for (int k = 0; k < pad; k++) { BlackwellIRInsn q = insn(BW_IR_MOV_IMM, r3, -1, -1, CW(1, 7, 7, 0)); q.imm = (uint32_t)k; if (omega_bw_ir_append(&p, &q) < 0) return -2; }
    BlackwellIRInsn f = insn(BW_IR_FMUL, r2, r1, r1, cw);
    if (omega_bw_ir_append(&p, &f) < 0) return -2;
    BlackwellIRInsn g = insn(BW_IR_FMUL, r3, r2, r3, CW(6, 7, 7, 0));
    if (omega_bw_ir_append(&p, &g) < 0) return -2;
    BlackwellIRInsn x = insn(BW_IR_EXIT, -1, -1, -1, CW(5, 7, 7, 0));
    if (omega_bw_ir_append(&p, &x) < 0) return -2;
    if (omega_bw_regalloc_solve(&p) != 0) return -2;
    return omega_tern_i2fp_violations(&p);
}

int main(void) {
    if (!CUDAPP_MIGRATION_PENDING_TERN_SCHED) { printf("gpu_tern_sched_test: NOT_RUN (retired: CUDAPP_MIGRATION_PENDING_TERN_SCHED=0)\n"); return 77; }
    printf("gpu_tern_sched_test: I2FP consumer rule (OMEGA_TERN_I2FP_LAT %u)\n", OMEGA_TERN_I2FP_LAT);
    static BlackwellIRProgram prog;
    static const uint32_t ks[] = { 256, 2560, 8192 };
    static const uint32_t shapes[][3] = { { OMEGA_TERN_LAYOUT_ROWMAJOR, 1, 1 }, { OMEGA_TERN_LAYOUT_ROWMAJOR, 2, 1 },
                                          { OMEGA_TERN_LAYOUT_ROWMAJOR, 4, 1 }, { OMEGA_TERN_LAYOUT_BLOCK, 2, 1 },
                                          { OMEGA_TERN_LAYOUT_BLOCK, 4, 1 } };
    int built = 0, refused = 0, specs = 0, wired = 0;
    for (unsigned a = 0; a < sizeof ks / sizeof ks[0]; a++)
        for (unsigned b = 0; b < sizeof shapes / sizeof shapes[0]; b++)
            for (uint32_t w = 1; w <= 32; w *= 4) {
                OmegaTernGemvSpec s = { ks[a], shapes[b][0], shapes[b][1], w, shapes[b][2], OMEGA_TERN_EPI_F32, 0 };
                if (!omega_tern_gemv_spec_ok(&s)) continue;
                specs++;
                if (omega_blackwell_codegen_tern_gemv_prog(&s, &prog) == 0) {
                    built++;
                    int seen = 0, ok = 1;
                    for (size_t i = 0; i < prog.count; i++) {
                        if (prog.insns[i].op != BW_IR_I2FP) continue;
                        seen++;
                        const uint32_t wb = (prog.insns[i].control >> 14) & 7u;
                        const uint32_t nxw = i + 1 < prog.count ? (prog.insns[i + 1].control >> 20) & 0x3fu : 0;
                        if (wb == 7u || !((nxw >> wb) & 1u)) ok = 0;
                    }
                    if (seen && ok && omega_tern_i2fp_violations(&prog) == 0) wired++;
                }
                s.mutant = 4;
                if (omega_tern_gemv_spec_ok(&s) && omega_blackwell_codegen_tern_gemv_prog(&s, &prog) != 0) refused++;
            }
    printf("  f32 specs %d: built %d, I2FP on a scoreboard its reader waits on %d, mutant 4 refused %d\n", specs, built, wired, refused);
    check(specs > 0 && built == specs, "1. every production f32 spec builds");
    check(specs > 0 && wired == specs, "1. its I2FP writes a scoreboard and the next instruction waits on it");
    check(specs > 0 && refused == specs, "2. negative control: mutant 4 (I2FP fixed stall 6) refused for every spec");
    OmegaTernGemvSpec i32 = { 2560, OMEGA_TERN_LAYOUT_ROWMAJOR, 2, 4, 1, OMEGA_TERN_EPI_I32, 4 };
    check(omega_blackwell_codegen_tern_gemv_prog(&i32, &prog) == 0, "2. mutant 4 without an I2FP (i32 epilogue) still builds");

    check(synth(BW_IR_I2FP, CW(6, 7, 7, 0), 0, CW(6, 7, 7, 0)) == 1, "3. I2FP -> FMUL at stall 6, no scoreboard: violation");
    check(synth(BW_IR_I2FP, CW(15, 7, 7, 0), 3, CW(6, 7, 7, 0)) == 1, "3. I2FP -> FMUL at 18 cycles, no scoreboard: violation");
    check(synth(BW_IR_I2FP, CW(15, 7, 7, 0), 4, CW(6, 7, 7, 0)) == 0, "3. I2FP -> FMUL at 19 cycles, no scoreboard: pass");
    check(synth(BW_IR_I2FP, CW(4, 0, 1, 0), 0, CW(6, 7, 7, 1u)) == 0, "3. I2FP on scoreboard 0, FMUL waits on it at stall 4: pass");
    check(synth(BW_IR_I2FP, CW(4, 0, 1, 0), 0, CW(6, 7, 7, 2u)) == 1, "3. I2FP on scoreboard 0, FMUL waits on another: violation");
    check(synth(BW_IR_FMUL, CW(6, 7, 7, 0), 0, CW(6, 7, 7, 0)) == 0, "3. FMUL producer -> FMUL at stall 6: pass (not an I2FP)");
    if (fails) { printf("gpu_tern_sched_test: FAIL (%d)\n", fails); return 1; }
    printf("gpu_tern_sched_test: PASS\n");
    return 0;
}
