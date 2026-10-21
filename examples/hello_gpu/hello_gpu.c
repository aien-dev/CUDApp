/* hello_gpu: the smallest complete Omega program. It adds two arrays of one million floats on the GPU
 * (c[i] = a[i] + b[i]) and checks every answer on the CPU.
 *
 * What makes it "Omega": the GPU program (the kernel) is written below as a short list of machine-level
 * steps, handed to Omega's own code generator, and run through Omega's own launcher. Nothing here uses
 * CUDA, nvcc, PTX or any NVIDIA user-space library.
 *
 * The five steps, in the order they appear in main():
 *   1. Build the kernel      build_vecadd()   a list of IR instructions, then register allocation
 *   2. Open the GPU          omega_gpu_session_open()
 *   3. Upload kernel + data  omega_gpu_session_alloc_code(), omega_gpu_session_alloc()
 *   4. Launch                omega_gpu_session_launch()   (eight launches of 131072 elements each)
 *   5. Check                 every one of the 1048576 outputs against the CPU's own a + b
 *
 * Prints PASS or FAIL. Exit code: 0 PASS, 1 FAIL (wrong or missing results), 2 the GPU could not be used.
 *
 * Hardware: NVIDIA GB10 (DGX Spark, sm_121), Linux, the open NVIDIA kernel driver loaded.
 * Built by `make hello-gpu` (see README, "Run your first kernel").
 */
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_qmd.h"
#include "omega_gpu_session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_TOTAL     (1u << 20)          /* 1048576 floats per array */
#define THREADS     128u                /* threads per block (CTA) */
#define MAX_CTAS    1024u               /* blocks per launch the session is proven for (omega_gpu_elementwise_api.h) */
#define CHUNK       (THREADS * MAX_CTAS) /* elements per launch: 131072, so N_TOTAL is 8 launches */
#define POISON      0xFFC0FFEEu         /* a NaN pattern c starts out as: an element the GPU never wrote stays this */

/* ---- Step 1: the kernel, as IR instructions -------------------------------------------------------------
 * Each instruction carries a "control word" that tells the chip how long to wait before the next one.
 * Two rules are enough for this kernel (chip-learned, see the header of src/omega_gpu_elementwise_api.c):
 *   - a load or special-register read (LDC, LDCU, S2R, LDG) finishes later, on a scoreboard slot;
 *     the next instruction waits for that slot;
 *   - plain arithmetic (IMAD, FADD) needs about 6 cycles before its result is read.
 * To stay simple, every instruction here waits for everything still pending. */
#define NOREG (-1)
#define CTRL(stall, wbar, rbar, wait) \
    (((uint32_t)(stall) << 9) | ((uint32_t)(wbar) << 14) | ((uint32_t)(rbar) << 17) | ((uint32_t)(wait) << 20))
#define YIELD 0x2000u
enum { SB_LOAD = 0, SB_STORE = 1, SB_UNIFORM = 2, SB_NONE = 7 }; /* scoreboard slots */
enum { K_ALU, K_PRED, K_LOAD, K_ULOAD, K_STORE, K_EXIT, K_EXIT_IF_P0 }; /* what kind of instruction */

typedef struct { BlackwellIRProgram *p; int desc; uint32_t pending; int err; } Kb; /* desc: the memory descriptor register */

static void emit(Kb *k, BlackwellIROpcode op, int dst, int a, int b, int c, uint32_t imm, int kind) {
    BlackwellIRInsn in; memset(&in, 0, sizeof in);
    in.op = op; in.dst_vreg = dst; in.src1_vreg = a; in.src2_vreg = b; in.src3_vreg = c; in.ureg = (op == BW_IR_LDG_E || op == BW_IR_STG_E) ? k->desc : NOREG; in.imm = imm;
    uint32_t wait = k->pending;
    switch (kind) {
    case K_ALU:   in.control = CTRL(6, SB_NONE, SB_NONE, wait); k->pending = 0; break;
    case K_PRED:  in.control = CTRL(14, SB_NONE, SB_NONE, wait); k->pending = 0; break;
    case K_LOAD:  in.control = CTRL(4, SB_LOAD, SB_NONE, wait) | YIELD; k->pending = 1u << SB_LOAD; break;
    case K_ULOAD: in.control = CTRL(4, SB_UNIFORM, SB_NONE, wait) | YIELD; k->pending = 1u << SB_UNIFORM; in.is_uniform = true; break;
    case K_STORE: in.control = CTRL(4, SB_NONE, SB_STORE, wait) | YIELD; k->pending = 1u << SB_STORE; break;
    case K_EXIT:  in.control = CTRL(5, SB_NONE, SB_NONE, wait); k->pending = 0; break;
    case K_EXIT_IF_P0: in.control = CTRL(5, SB_NONE, SB_NONE, wait); in.predicate_p0 = true; break; /* may fall through: keep waiting */
    }
    if (omega_bw_ir_append(k->p, &in) < 0) k->err = -1;
}

/* The kernel the chip runs once per thread:
 *     i = block_index * 128 + thread_index
 *     if (i >= n) exit
 *     c[i] = a[i] + b[i]
 * a, b, c and n arrive in the launch's argument block (constant bank 0): a at 0x380, b at 0x388, c at 0x390
 * (64-bit addresses), n at 0x398. 0x358 holds the memory descriptor every load and store names. */
static int build_vecadd(BlackwellIRProgram *prog) {
    omega_bw_ir_init(prog);
    Kb k = { .p = prog, .desc = omega_bw_ir_alloc_uvreg64(prog) };
    int pa = omega_bw_ir_alloc_vreg64(prog), pb = omega_bw_ir_alloc_vreg64(prog), pc = omega_bw_ir_alloc_vreg64(prog);
    int tid = omega_bw_ir_alloc_vreg(prog), cta = omega_bw_ir_alloc_vreg(prog), n = omega_bw_ir_alloc_vreg(prog);
    int i = omega_bw_ir_alloc_vreg(prog), at = omega_bw_ir_alloc_vreg64(prog);
    int x = omega_bw_ir_alloc_vreg(prog), y = omega_bw_ir_alloc_vreg(prog), z = omega_bw_ir_alloc_vreg(prog);

    emit(&k, BW_IR_LDCU64, k.desc, NOREG, NOREG, NOREG, 0x358, K_ULOAD);
    emit(&k, BW_IR_LDC64, pa, NOREG, NOREG, NOREG, 0x380, K_LOAD);
    emit(&k, BW_IR_LDC64, pb, NOREG, NOREG, NOREG, 0x388, K_LOAD);
    emit(&k, BW_IR_LDC64, pc, NOREG, NOREG, NOREG, 0x390, K_LOAD);
    emit(&k, BW_IR_LDC, n, NOREG, NOREG, NOREG, 0x398, K_LOAD);
    emit(&k, BW_IR_S2R, tid, NOREG, NOREG, NOREG, BW_SR_TID_X, K_LOAD);
    emit(&k, BW_IR_S2R, cta, NOREG, NOREG, NOREG, BW_SR_CTAID_X, K_LOAD);
    emit(&k, BW_IR_IMAD, i, cta, NOREG, tid, THREADS, K_ALU);               /* i = cta * 128 + tid */
    emit(&k, BW_IR_ISETP_GE_U32, NOREG, i, n, NOREG, 0, K_PRED);            /* P0 = (i >= n) */
    emit(&k, BW_IR_EXIT, NOREG, NOREG, NOREG, NOREG, 0, K_EXIT_IF_P0);      /* @P0 EXIT */

    /* IMAD.WIDE builds a 64-bit byte address:  at = i * 4 + base pointer */
    emit(&k, BW_IR_IMAD_WIDE, at, i, NOREG, pa, 4, K_ALU);
    emit(&k, BW_IR_LDG_E, x, at, NOREG, NOREG, 0, K_LOAD);                  /* x = a[i] */
    emit(&k, BW_IR_IMAD_WIDE, at, i, NOREG, pb, 4, K_ALU);
    emit(&k, BW_IR_LDG_E, y, at, NOREG, NOREG, 0, K_LOAD);                  /* y = b[i] */
    emit(&k, BW_IR_FADD, z, x, y, NOREG, 0, K_ALU);                         /* z = x + y */
    emit(&k, BW_IR_IMAD_WIDE, at, i, NOREG, pc, 4, K_ALU);
    emit(&k, BW_IR_STG_E, NOREG, at, z, NOREG, 0, K_STORE);                 /* c[i] = z */

    /* every kernel ends with EXIT, then a branch-to-self the hardware never reaches */
    emit(&k, BW_IR_EXIT, NOREG, NOREG, NOREG, NOREG, 0, K_EXIT);
    emit(&k, BW_IR_BRA, NOREG, NOREG, NOREG, NOREG, 0, K_ALU);
    prog->insns[prog->count - 1].control = 0x000fc000u;

    if (k.err) return -1;
    return omega_bw_regalloc_solve(prog); /* picks the physical registers */
}

int main(void) {
    int rc = 1;
    OmegaGpuMem code = { 0 }, a = { 0 }, b = { 0 }, c = { 0 };
    static BlackwellIRProgram prog;

    printf("hello_gpu: c = a + b over %u floats on the GPU, built with Omega's own kernel builder\n", N_TOTAL);
    if (build_vecadd(&prog) != 0) { printf("FAIL: the kernel did not build\n"); return 1; }

    /* ---- Step 2: open the GPU. One process owns one device channel; the lock serialises threads. */
    omega_gpu_session_lock();
    if (!omega_gpu_session_open()) {
        printf("GPU not available (stage: %s)\nNOT_RUN\n", omega_gpu_session_last_error());
        omega_gpu_session_unlock();
        return 2;
    }

    /* ---- Step 3: upload. The code buffer is written by the CPU and executed by the GPU. */
    size_t bytes = (size_t)N_TOTAL * sizeof(float), need = omega_bw_program_code_bytes(&prog);
    OmegaBlackwellKernel kernel;
    if (omega_gpu_session_alloc_code(need, &code) != 0 || omega_blackwell_kernel_encode_into(&prog, code.cpu, need, &kernel) != 0 ||
        omega_gpu_session_alloc(bytes, &a) != 0 || omega_gpu_session_alloc(bytes, &b) != 0 || omega_gpu_session_alloc(bytes, &c) != 0) {
        printf("FAIL: could not set up buffers (stage: %s)\n", omega_gpu_session_last_error());
        goto done;
    }
    printf("kernel: %zu instructions, %u registers\n", kernel.insn_count, prog.regalloc.peak_gpr_usage);
    float *fa = a.cpu, *fb = b.cpu;
    uint32_t *wc = c.cpu;
    for (uint32_t i = 0; i < N_TOTAL; i++) { fa[i] = (float)i * 0.25f; fb[i] = 1.0f / (float)(1 + i % 7); wc[i] = POISON; }
    if (omega_gpu_session_sync_to_gpu(&code) || omega_gpu_session_sync_to_gpu(&a) || omega_gpu_session_sync_to_gpu(&b) ||
        omega_gpu_session_sync_to_gpu(&c)) { printf("FAIL: sync to GPU (stage: %s)\n", omega_gpu_session_last_error()); goto done; }

    /* ---- Step 4: launch. The argument words land in the kernel's constant bank: three addresses (low word,
     * high word) and n. Each launch is pointed at its own slice of the arrays. */
    uint32_t declared = omega_bw_gpr_declared(prog.regalloc.peak_gpr_usage);
    for (uint32_t base = 0; base < N_TOTAL; base += CHUNK) {
        uint64_t off = (uint64_t)base * sizeof(float);
        uint32_t n = N_TOTAL - base < CHUNK ? N_TOTAL - base : CHUNK;
        uint32_t args[OMEGA_BW_CBANK_MATMUL_ARGS_WORDS] = { (uint32_t)(a.va + off), (uint32_t)((a.va + off) >> 32),
            (uint32_t)(b.va + off), (uint32_t)((b.va + off) >> 32), (uint32_t)(c.va + off), (uint32_t)((c.va + off) >> 32), n };
        OmegaGpuLaunch L = { .code_va = code.va, .gpr_count = declared < 64 ? 64 : declared, .threads_x = THREADS, .threads_y = 1,
                             .grid_x = (n + THREADS - 1) / THREADS, .grid_y = 1, .num_elements = n,
                             .args = args, .n_args = OMEGA_BW_CBANK_MATMUL_ARGS_WORDS, .timeout_ms = 10000 };
        if (omega_gpu_session_launch(&L, NULL, NULL) != 0) { printf("FAIL: launch at element %u (stage: %s)\n", base, omega_gpu_session_last_error()); goto done; }
    }
    if (omega_gpu_session_sync_to_cpu(&c) != 0) { printf("FAIL: sync to CPU (stage: %s)\n", omega_gpu_session_last_error()); goto done; }

    /* ---- Step 5: check every element. Same IEEE single-precision add on both sides, so the bits must match. */
    uint32_t wrong = 0, unwritten = 0, first_bad = 0;
    for (uint32_t i = 0; i < N_TOTAL; i++) {
        float want = fa[i] + fb[i];
        if (wc[i] == POISON) unwritten++;
        if (memcmp(&want, &wc[i], sizeof want) != 0) { if (!wrong++) first_bad = i; }
    }
    printf("checked %u of %u elements: %u wrong, %u never written\n", N_TOTAL, N_TOTAL, wrong, unwritten);
    if (wrong) { float want = fa[first_bad] + fb[first_bad], got; memcpy(&got, &wc[first_bad], sizeof got);
                 printf("first mismatch: c[%u] = %g, expected %g\n", first_bad, got, want); }
    rc = wrong ? 1 : 0;
    printf("%s\n", rc ? "FAIL" : "PASS");

done:
    if (code.cpu) omega_gpu_session_free(&code);
    if (a.cpu) omega_gpu_session_free(&a);
    if (b.cpu) omega_gpu_session_free(&b);
    if (c.cpu) omega_gpu_session_free(&c);
    omega_gpu_session_unlock();
    omega_gpu_session_close();
    return rc;
}
