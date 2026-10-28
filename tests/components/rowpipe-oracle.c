/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Verification ONLY. Uses the existing b759afc3 rowpipe C generator, its allocator,
 * and independently retained C encoder. This is not a production implementation.
 * Build after all component edits, no GPU:
 * cc -std=c11 -Wall -Wextra -Werror -DOMEGA_GPU_NO_POSIX -ffunction-sections 
 *   -fdata-sections -Wl,--gc-sections -Isrc tests/components/rowpipe-oracle.c 
 *   src/omega_blackwell_codegen_moe2.c src/omega_blackwell_codegen.c -o ORACLE
 * Input: no arguments reads exactly eight LE64 spec words on stdin. Alternatively
 * ORACLE K LB ROWS WARPS accepts the production spec. Optional final argument
 * --sb-b / --sb-rd / --branch corrupts independent reference IR for negative cmp.
 * Native caller and oracle stdout must compare byte for byte, including widths,
 * allocation, every IR field, and every unpadded encoded instruction. Exit66
 * is refusal; run mutants1..3 and each reserved field nonzero against native
 * caller and require66 (caller also verifies all refusal outputs were cleared).
 * Cover K256 LB8 R1 W1, K768 LB8 R1 W5(partial staging),
 * K512 LB16 R8 W16, K4096 LB16 R1 W32, and R1/2/4/8 and LB8/16.
 * K4096 LB8 R1 W16 is a resource-refusal control: generator shape admission
 * precedes global allocation; native caller returns67 while this combined C
 * generator/allocator returns66. Neither may emit any output for that control.
 * Each --sb-b/--sb-rd/--branch output must FAIL cmp against the original caller.
 * No result from these host checks is device qualification.
 */
#include "omega_blackwell_codegen_moe2.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int word(uint64_t v) {
    for (unsigned j = 0; j < 8; ++j)
        if (putchar((int)((v >> (j * 8)) & 255)) == EOF) return 0;
    return 1;
}
static uint64_t regword(int v) { return v < 0 ? UINT64_MAX : (uint64_t)v; }
static int number(const char *s, uint64_t *v) {
    char *end;
    if (!s[0] || s[0] == '-') return 0;
    errno = 0;
    unsigned long long n = strtoull(s, &end, 10);
    if (errno || *end) return 0;
    *v = (uint64_t)n;
    return 1;
}
static int input(uint64_t spec[8]) {
    unsigned char b[64];
    if (fread(b, 1, sizeof b, stdin) != sizeof b || getchar() != EOF) return 0;
    for (unsigned j = 0; j < 8; ++j)
        for (unsigned k = 0; k < 8; ++k)
            spec[j] |= (uint64_t)b[j * 8 + k] << (k * 8);
    return !ferror(stdin);
}
static int mutate(BlackwellIRProgram *p, const char *kind) {
    if (!kind) return 1;
    for (size_t j = 0; j < p->count; ++j) {
        BlackwellIRInsn *i = &p->insns[j];
        if (!strcmp(kind, "--sb-b") && ((i->control >> 20) & (1u << 3))) {
            i->control ^= (1u << 3) << 20;
            return 1;
        }
        if (!strcmp(kind, "--sb-rd") && ((i->control >> 14) & 7) == 3) {
            i->control = (i->control & ~(7u << 17)) | (7u << 17);
            return 1;
        }
        if (!strcmp(kind, "--branch") && i->op == BW_IR_BRA && i->predicate_p0) {
            ++i->imm;
            return 1;
        }
    }
    return 0;
}
int main(int argc, char **argv) {
    uint64_t spec[8] = {0};
    const char *mutation = NULL;
    if (argc == 1 || argc == 2) {
        if (!input(spec)) return 65;
        if (argc == 2) mutation = argv[1];
    } else if (argc == 5 || argc == 6) {
        for (unsigned j = 0; j < 4; ++j)
            if (!number(argv[j + 1], &spec[j])) return 65;
        if (argc == 6) mutation = argv[5];
    } else return 65;
    for (unsigned j = 0; j < 4; ++j) if (spec[j] > UINT32_MAX) return 66;
    for (unsigned j = 4; j < 8; ++j) if (spec[j]) return 66;
    OmegaFp8GemvSpec s = {0};
    s.k = (uint32_t)spec[0]; s.lane_bytes = (uint32_t)spec[1];
    s.rows = (uint32_t)spec[2]; s.warps = (uint32_t)spec[3];
    static BlackwellIRProgram p;
    if (omega_blackwell_codegen_fp8_gemv2_prog(&s, &p) != 0) return 66;
    if (!mutate(&p, mutation)) return 70;
    unsigned char code[16384];
    size_t padded = 0;
    if (omega_bw_encode_program(&p, code, sizeof code, &padded) != 0) return 68;
    size_t used = p.count * 16;
    if (padded < used) return 68;
    if (!word(p.count) || !word((uint64_t)p.regalloc.num_vregs) ||
        !word((uint64_t)p.regalloc.num_uvregs) || !word(omega_fp8_gemv2_shared_bytes(&s)) ||
        !word(p.regalloc.peak_gpr_usage) || !word(p.regalloc.peak_ugpr_usage)) return 74;
    for (int v = 0; v < p.regalloc.num_vregs; ++v) {
        const OmegaLiveInterval *q = &p.regalloc.intervals[v];
        if (!word(q->bundle_size ? q->bundle_size : (q->is_pair ? 2 : (q->is_quad ? 4 : 1))) ||
            !word(regword(p.regalloc.vreg_to_phys[v]))) return 74;
    }
    for (int v = 0; v < p.regalloc.num_uvregs; ++v)
        if (!word(p.regalloc.uintervals[v].is_pair ? 2 : 1) ||
            !word(regword(p.regalloc.uvreg_to_phys[v]))) return 74;
    for (size_t j = 0; j < p.count; ++j) {
        const BlackwellIRInsn *i = &p.insns[j];
        uint64_t f = (uint64_t)i->is_uniform | ((uint64_t)i->predicate_p0 << 1) |
            ((uint64_t)i->predicate_not << 2) | ((uint64_t)i->bar_reg << 8) |
            ((uint64_t)i->dst_subreg << 16) | ((uint64_t)i->src1_subreg << 24) |
            ((uint64_t)i->src2_subreg << 32) | ((uint64_t)i->src3_subreg << 40);
        if (!word(i->op) || !word(regword(i->dst_vreg)) || !word(regword(i->src1_vreg)) ||
            !word(regword(i->src2_vreg)) || !word(regword(i->src3_vreg)) ||
            !word(regword(i->ureg)) || !word((uint64_t)i->imm | ((uint64_t)i->control << 32)) ||
            !word(f)) return 74;
    }
    if (fwrite(code, 1, used, stdout) != used || fflush(stdout) != 0) return 74;
    return 0;
}
