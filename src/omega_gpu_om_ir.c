/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* MIGRATION-PENDING C (successor: a seeded-Omega allocator beside om/encoder/bw_enc_sm121.om); see the header. */
#include "omega_gpu_om_ir.h"

#include <string.h>

#include "omega_native.h"

#define NONE64 UINT64_MAX

static int field(uint64_t w, int *out) {
    if (w == NONE64) { *out = -1; return 0; }
    if (w >= (uint64_t)BW_MAX_VREGS) return -1;
    *out = (int)w;
    return 0;
}

static int call_unit(uint64_t kind, uint64_t a, uint64_t b, uint64_t c, uint64_t *ins, uint64_t *meta,
                     uint64_t *result) {
    *result = 0;
    if (omega_mathlibs_ir(result, kind, a, b, c, ins, OMEGA_OM_IR_WORDS, meta, OMEGA_OM_IR_META) != OMEGA_OK)
        return OMEGA_OM_IR_BRIDGE;
    if (*result == 1) return OMEGA_OM_IR_REFUSED;
    if (*result != 0) return OMEGA_OM_IR_EMIT;
    return OMEGA_OM_IR_OK;
}

int omega_gpu_om_ir_data(uint64_t kind, uint64_t a, uint64_t b, uint64_t c, uint64_t *words) {
    static uint64_t meta[OMEGA_OM_IR_META];
    uint64_t result;
    if (!words) return OMEGA_OM_IR_REPLAY;
    memset(words, 0, OMEGA_OM_IR_WORDS * sizeof *words);
    return call_unit(kind, a, b, c, words, meta, &result);
}

/* The elementwise emitter's check (src/omega_gpu_elementwise_api.c check_loop_invariant): no value first defined
 * inside a loop may be read after the loop before it is redefined. */
static int loop_invariant_ok(const BlackwellIRProgram *p, const uint64_t *meta) {
    const OmegaRegAlloc *ra = &p->regalloc;
    for (uint64_t L = 0; L < meta[5]; L++) {
        int start = (int)meta[16 + 2 * L], end = (int)meta[17 + 2 * L];
        for (int v = 0; v < ra->num_vregs; v++) {
            int fd = ra->intervals[v].first_def;
            if (fd < start || fd > end) continue;
            for (size_t i = (size_t)end + 1; i < p->count; i++) {
                const BlackwellIRInsn *in = &p->insns[i];
                if (in->src1_vreg == v || in->src2_vreg == v || in->src3_vreg == v) return 0;
                if (in->dst_vreg == v && !in->is_uniform) break;
            }
        }
    }
    return 1;
}

/* A value live into a loop and read inside it is read again by every later iteration: hold it to the closing BRA
 * (the elementwise emitter's loop_end fixup). */
static void extend_loop(BlackwellIRProgram *p, int start, int end) {
    for (int v = 0; v < p->regalloc.num_vregs; v++) {
        OmegaLiveInterval *iv = &p->regalloc.intervals[v];
        if (iv->first_def >= 0 && iv->first_def < start && iv->last_use >= start && iv->last_use < end)
            iv->last_use = end;
    }
}

int omega_gpu_om_ir_build(uint64_t kind, uint64_t a, uint64_t b, uint64_t c, uint32_t threads,
                          BlackwellIRProgram *prog, uint8_t *code, size_t code_cap,
                          OmegaBlackwellKernel *kernel, uint32_t *declared, uint64_t *emit_err) {
    static uint64_t ins[OMEGA_OM_IR_WORDS], meta[OMEGA_OM_IR_META];
    uint64_t result;
    if (emit_err) *emit_err = 0;
    if (!prog || !code || !kernel || !declared || threads == 0) return OMEGA_OM_IR_REPLAY;
    memset(ins, 0, sizeof ins);
    int rc = call_unit(kind, a, b, c, ins, meta, &result);
    if (emit_err) *emit_err = meta[4];
    if (rc != OMEGA_OM_IR_OK) return rc;
    uint64_t n = meta[0];
    if (n == 0 || n > BW_MAX_IR_INSNS || meta[1] > BW_MAX_VREGS || meta[2] > BW_MAX_UVREGS || meta[5] > 24)
        return OMEGA_OM_IR_REPLAY;

    omega_bw_ir_init(prog);
    for (uint64_t v = 0; v < meta[1]; v++) {
        uint64_t k = meta[64 + v];
        int got = k == 2 ? omega_bw_ir_alloc_vreg64(prog) : k == 1 ? omega_bw_ir_alloc_vreg(prog) : -1;
        if (got != (int)v) return OMEGA_OM_IR_REPLAY;
    }
    for (uint64_t u = 0; u < meta[2]; u++) {
        uint64_t k = meta[320 + u];
        int got = k == 2 ? omega_bw_ir_alloc_uvreg64(prog) : omega_bw_ir_alloc_uvreg(prog);
        if (got != (int)u) return OMEGA_OM_IR_REPLAY;
    }

    uint64_t next_loop = 0;
    for (uint64_t i = 0; i < n; i++) {
        const uint64_t *r = &ins[i * 8];
        BlackwellIRInsn in;
        memset(&in, 0, sizeof in);
        if (r[0] > (uint64_t)BW_IR_MUFU_LG2) return OMEGA_OM_IR_REPLAY;
        in.op = (BlackwellIROpcode)r[0];
        if (field(r[1], &in.dst_vreg) || field(r[2], &in.src1_vreg) || field(r[3], &in.src2_vreg) ||
            field(r[4], &in.src3_vreg))
            return OMEGA_OM_IR_REPLAY;
        if (r[5] == NONE64) in.ureg = -1;
        else if (r[5] < (uint64_t)BW_MAX_UVREGS) in.ureg = (int)r[5];
        else return OMEGA_OM_IR_REPLAY;
        in.imm = (uint32_t)r[6];
        in.control = (uint32_t)(r[6] >> 32);
        in.is_uniform = (r[7] & 1u) != 0;
        in.predicate_p0 = (r[7] & 2u) != 0;
        in.predicate_not = (r[7] & 4u) != 0;
        in.dst_subreg = (uint8_t)(r[7] >> 16);
        in.src1_subreg = (uint8_t)(r[7] >> 24);
        in.src2_subreg = (uint8_t)(r[7] >> 32);
        in.src3_subreg = (uint8_t)(r[7] >> 40);
        if (omega_bw_ir_append(prog, &in) < 0) return OMEGA_OM_IR_REPLAY;
        while (next_loop < meta[5] && meta[17 + 2 * next_loop] == i) {
            extend_loop(prog, (int)meta[16 + 2 * next_loop], (int)i);
            next_loop++;
        }
    }
    if (next_loop != meta[5]) return OMEGA_OM_IR_REPLAY;

    if (omega_bw_regalloc_solve(prog) != 0) return OMEGA_OM_IR_REGALLOC;
    if (!loop_invariant_ok(prog, meta)) return OMEGA_OM_IR_LOOP;
    uint32_t peak = prog->regalloc.peak_gpr_usage;
    uint32_t decl = omega_bw_gpr_declared(peak);
    if (decl < 64u) decl = 64u;
    if (peak > BW_PHYS_GPR_MAX || !omega_bw_gpr_declared_ok(decl, peak) || (uint64_t)decl * threads > 65536u)
        return OMEGA_OM_IR_BUDGET;
    if (omega_blackwell_kernel_encode_into(prog, code, code_cap, kernel) != 0) return OMEGA_OM_IR_ENCODE;
    *declared = decl;
    return OMEGA_OM_IR_OK;
}

const char *omega_gpu_om_ir_rc_name(int rc) {
    switch (rc) {
    case OMEGA_OM_IR_OK: return "ok";
    case OMEGA_OM_IR_REFUSED: return "refused";
    case OMEGA_OM_IR_EMIT: return "emit";
    case OMEGA_OM_IR_BRIDGE: return "bridge";
    case OMEGA_OM_IR_REPLAY: return "replay";
    case OMEGA_OM_IR_REGALLOC: return "regalloc";
    case OMEGA_OM_IR_LOOP: return "loop";
    case OMEGA_OM_IR_BUDGET: return "budget";
    case OMEGA_OM_IR_ENCODE: return "encode";
    default: return "?";
    }
}
