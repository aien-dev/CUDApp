#include "omega_blackwell_codegen.h"
#include "omega_blackwell_submit.h"
#include "sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void omega_bw_ir_init(BlackwellIRProgram *prog) {
    if (!prog) return;
    memset(prog, 0, sizeof(*prog));
}

int omega_bw_ir_alloc_vreg(BlackwellIRProgram *prog) {
    if (!prog || prog->regalloc.num_vregs >= BW_MAX_VREGS) return -1;
    int v = prog->regalloc.num_vregs++;
    prog->regalloc.intervals[v].vreg = v;
    prog->regalloc.intervals[v].first_def = -1;
    prog->regalloc.intervals[v].last_use = -1;
    prog->regalloc.intervals[v].phys_reg = -1;
    prog->regalloc.intervals[v].is_pair = false;
    prog->regalloc.intervals[v].active = false;
    prog->regalloc.vreg_to_phys[v] = -1;
    return v;
}

int omega_bw_ir_alloc_vreg64(BlackwellIRProgram *prog) {
    if (!prog || prog->regalloc.num_vregs >= BW_MAX_VREGS) return -1;
    int v = prog->regalloc.num_vregs++;
    prog->regalloc.intervals[v].vreg = v;
    prog->regalloc.intervals[v].first_def = -1;
    prog->regalloc.intervals[v].last_use = -1;
    prog->regalloc.intervals[v].phys_reg = -1;
    prog->regalloc.intervals[v].is_pair = true;
    prog->regalloc.intervals[v].active = false;
    prog->regalloc.vreg_to_phys[v] = -1;
    return v;
}

int omega_bw_ir_alloc_uvreg(BlackwellIRProgram *prog) {
    if (!prog || prog->regalloc.num_uvregs >= BW_MAX_UVREGS) return -1;
    int uv = prog->regalloc.num_uvregs++;
    prog->regalloc.uintervals[uv].vreg = uv;
    prog->regalloc.uintervals[uv].first_def = -1;
    prog->regalloc.uintervals[uv].last_use = -1;
    prog->regalloc.uintervals[uv].phys_reg = -1;
    prog->regalloc.uintervals[uv].is_pair = false;
    prog->regalloc.uintervals[uv].active = false;
    prog->regalloc.uvreg_to_phys[uv] = -1;
    return uv;
}

int omega_bw_ir_alloc_uvreg64(BlackwellIRProgram *prog) {
    if (!prog || prog->regalloc.num_uvregs >= BW_MAX_UVREGS) return -1;
    int uv = prog->regalloc.num_uvregs++;
    prog->regalloc.uintervals[uv].vreg = uv;
    prog->regalloc.uintervals[uv].first_def = -1;
    prog->regalloc.uintervals[uv].last_use = -1;
    prog->regalloc.uintervals[uv].phys_reg = -1;
    prog->regalloc.uintervals[uv].is_pair = true;
    prog->regalloc.uintervals[uv].active = false;
    prog->regalloc.uvreg_to_phys[uv] = -1;
    return uv;
}

int omega_bw_ir_append(BlackwellIRProgram *prog, const BlackwellIRInsn *insn) {
    if (!prog || !insn || prog->count >= BW_MAX_IR_INSNS) return -1;
    int idx = (int)prog->count++;
    prog->insns[idx] = *insn;

    /* Update live-interval def/use tracking */
    if (insn->dst_vreg >= 0 && insn->dst_vreg < prog->regalloc.num_vregs) {
        if (!insn->is_uniform) {
            OmegaLiveInterval *iv = &prog->regalloc.intervals[insn->dst_vreg];
            if (iv->first_def < 0) iv->first_def = idx;
            if (idx > iv->last_use) iv->last_use = idx;
        }
    }
    if (insn->src1_vreg >= 0 && insn->src1_vreg < prog->regalloc.num_vregs) {
        OmegaLiveInterval *iv = &prog->regalloc.intervals[insn->src1_vreg];
        if (iv->first_def < 0) iv->first_def = idx;
        if (idx > iv->last_use) iv->last_use = idx;
    }
    if (insn->src2_vreg >= 0 && insn->src2_vreg < prog->regalloc.num_vregs) {
        OmegaLiveInterval *iv = &prog->regalloc.intervals[insn->src2_vreg];
        if (iv->first_def < 0) iv->first_def = idx;
        if (idx > iv->last_use) iv->last_use = idx;
    }
    if (insn->src3_vreg >= 0 && insn->src3_vreg < prog->regalloc.num_vregs) {
        OmegaLiveInterval *iv = &prog->regalloc.intervals[insn->src3_vreg];
        if (iv->first_def < 0) iv->first_def = idx;
        if (idx > iv->last_use) iv->last_use = idx;
    }

    /* Uniform register def/use tracking */
    if (insn->is_uniform && insn->dst_vreg >= 0 && insn->dst_vreg < prog->regalloc.num_uvregs) {
        OmegaLiveInterval *uiv = &prog->regalloc.uintervals[insn->dst_vreg];
        if (uiv->first_def < 0) uiv->first_def = idx;
        if (idx > uiv->last_use) uiv->last_use = idx;
    }
    if (insn->ureg >= 0 && insn->ureg < prog->regalloc.num_uvregs) {
        OmegaLiveInterval *uiv = &prog->regalloc.uintervals[insn->ureg];
        if (uiv->first_def < 0) uiv->first_def = idx;
        if (idx > uiv->last_use) uiv->last_use = idx;
    }

    return idx;
}

int omega_bw_regalloc_solve(BlackwellIRProgram *prog) {
    if (!prog) return -1;
    OmegaRegAlloc *ra = &prog->regalloc;

    /*
     * Bounded deterministic linear-scan register allocator.
     * Computes genuine live intervals across virtual registers and assigns
     * available physical GPRs (R2..R127) and UGPRs (UR4..UR31).
     * Enforces even-alignment and dual-register reservation for 64-bit pairs.
     */
    bool gpr_busy[BW_PHYS_GPR_MAX];
    memset(gpr_busy, 0, sizeof(gpr_busy));

    int max_phys_used = BW_PHYS_GPR_START;

    for (int v = 0; v < ra->num_vregs; v++) {
        OmegaLiveInterval *iv = &ra->intervals[v];
        if (iv->first_def < 0) {
            iv->phys_reg = BW_PHYS_GPR_START;
            ra->vreg_to_phys[v] = iv->phys_reg;
            continue;
        }

        /* Free physical registers whose virtual live intervals have ended */
        for (int p = 0; p < v; p++) {
            if (ra->intervals[p].active && ra->intervals[p].last_use < iv->first_def) {
                int freed = ra->intervals[p].phys_reg;
                if (freed >= 0 && freed < BW_PHYS_GPR_MAX) {
                    gpr_busy[freed] = false;
                    if (ra->intervals[p].is_pair && freed + 1 < BW_PHYS_GPR_MAX) {
                        gpr_busy[freed + 1] = false;
                    }
                }
                ra->intervals[p].active = false;
            }
        }

        /* Find lowest available physical register */
        int assigned = -1;
        if (iv->is_pair) {
            /* 64-bit pair requires even alignment and contiguous free registers */
            int start = (BW_PHYS_GPR_START + 1) & ~1;
            for (int p = start; p < BW_PHYS_GPR_MAX - 1; p += 2) {
                if (!gpr_busy[p] && !gpr_busy[p + 1]) {
                    assigned = p;
                    gpr_busy[p] = true;
                    gpr_busy[p + 1] = true;
                    break;
                }
            }
        } else {
            for (int p = BW_PHYS_GPR_START; p < BW_PHYS_GPR_MAX; p++) {
                if (!gpr_busy[p]) {
                    assigned = p;
                    gpr_busy[p] = true;
                    break;
                }
            }
        }

        if (assigned < 0) {
            /* Hardware register capacity exceeded: fail closed loudly */
            return -1;
        }

        iv->phys_reg = assigned;
        iv->active = true;
        ra->vreg_to_phys[v] = assigned;

        int top = iv->is_pair ? (assigned + 1) : assigned;
        if (top > max_phys_used) {
            max_phys_used = top;
        }
    }
    ra->peak_gpr_usage = (uint32_t)(max_phys_used + 1);

    /* Allocate uniform registers */
    int cur_u = BW_PHYS_UGPR_START;
    for (int uv = 0; uv < ra->num_uvregs; uv++) {
        OmegaLiveInterval *uiv = &ra->uintervals[uv];
        if (uiv->is_pair) {
            if (cur_u % 2 != 0) cur_u++;
            if (cur_u + 1 >= BW_PHYS_UGPR_MAX) return -1;
            uiv->phys_reg = cur_u;
            ra->uvreg_to_phys[uv] = cur_u;
            cur_u += 2;
        } else {
            if (cur_u >= BW_PHYS_UGPR_MAX) return -1;
            uiv->phys_reg = cur_u;
            ra->uvreg_to_phys[uv] = cur_u++;
        }
    }
    ra->peak_ugpr_usage = (uint32_t)cur_u;

    return 0;
}

static int encode_single_insn(const BlackwellIRInsn *insn, const OmegaRegAlloc *ra, uint32_t w[4]) {
    memset(w, 0, 16);

    int dst = (insn->dst_vreg >= 0) ? ra->vreg_to_phys[insn->dst_vreg] : 0;
    int src1 = (insn->src1_vreg >= 0) ? ra->vreg_to_phys[insn->src1_vreg] : 0;
    int src2 = (insn->src2_vreg >= 0) ? ra->vreg_to_phys[insn->src2_vreg] : 0;
    int src3 = (insn->src3_vreg >= 0) ? ra->vreg_to_phys[insn->src3_vreg] : 0xff; /* Default to RZ (0xff) */
    int ureg = (insn->ureg >= 0) ? ra->uvreg_to_phys[insn->ureg] : 0;

    if (insn->is_uniform && insn->dst_vreg >= 0) {
        dst = ra->uvreg_to_phys[insn->dst_vreg];
    }

    switch (insn->op) {
        case BW_IR_NOP:
            w[0] = 0x00007918;
            w[1] = 0x00000000;
            w[2] = 0x00000000;
            w[3] = 0x000fc000;
            break;

        case BW_IR_MOV_RZ:
            /* MOV Rd, RZ */
            w[0] = 0x7202 | ((uint32_t)(dst & 0xff) << 16);
            w[1] = 0x000000ff;
            w[2] = 0x00000f00;
            w[3] = insn->control ? insn->control : 0x000fe200;
            break;

        case BW_IR_MOV_IMM:
            /* MOV Rd, imm32 */
            w[0] = 0x7802 | ((uint32_t)(dst & 0xff) << 16);
            w[1] = insn->imm;
            w[2] = 0x00000f00;
            w[3] = insn->control ? insn->control : 0x000fe200;
            break;

        case BW_IR_S2R:
            /* S2R Rd, SR */
            w[0] = 0x7919 | ((uint32_t)(dst & 0xff) << 16);
            w[1] = 0x0;
            w[2] = (insn->imm & 0xff) << 8;
            w[3] = insn->control ? insn->control : 0x000e2200;
            break;

        case BW_IR_LDC:
            /* LDC Rd, c[0x0][imm] */
            w[0] = 0xff007b82 | ((uint32_t)(dst & 0xff) << 16);
            w[1] = (insn->imm << 6) & 0x00ffff00;
            w[2] = 0x00000800;
            w[3] = insn->control ? insn->control : 0x000fe200;
            break;

        case BW_IR_LDC64:
            /* LDC.64 Rd:Rd+1, c[0x0][imm] */
            w[0] = 0xff007b82 | ((uint32_t)(dst & 0xff) << 16);
            w[1] = (insn->imm << 6) & 0x00ffff00;
            w[2] = 0x00000a00;
            w[3] = insn->control ? insn->control : 0x000e2200;
            break;

        case BW_IR_LDCU:
            /* LDCU URd, c[0x0][imm] */
            w[0] = 0xff0077ac | ((uint32_t)(dst & 0xff) << 16);
            w[1] = (insn->imm << 5) & 0x00ffff00;
            w[2] = 0x08000800;
            w[3] = insn->control ? insn->control : 0x000e2200;
            break;

        case BW_IR_LDCU64:
            /* LDCU.64 URd:URd+1, c[0x0][imm] */
            w[0] = 0xff0077ac | ((uint32_t)(dst & 0xff) << 16);
            w[1] = (insn->imm << 5) & 0x00ffff00;
            w[2] = 0x08000a00;
            w[3] = insn->control ? insn->control : 0x000e6e00;
            break;

        case BW_IR_IMAD:
            if (insn->src2_vreg >= 0) {
                /* Register-register: IMAD Rd, Ra, Rb, Rc (0x7224) */
                w[0] = 0x7224 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
                w[1] = (uint32_t)(src2 & 0xff);
                w[2] = 0x078e0200 | (uint32_t)(src3 & 0xff);
                w[3] = insn->control ? insn->control : 0x001fca00;
            } else if (insn->is_uniform && insn->ureg >= 0) {
                /* Uniform register multiplier: IMAD Rd, Ra, URb, Rc (0x7c24) */
                w[0] = 0x7c24 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
                w[1] = (uint32_t)(ureg & 0xff);
                w[2] = 0x0f8e0200 | (uint32_t)(src3 & 0xff);
                w[3] = insn->control ? insn->control : 0x001fca00;
            } else {
                /* Immediate multiplier: IMAD Rd, Ra, imm, Rc (0x7824) */
                w[0] = 0x7824 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
                w[1] = insn->imm;
                w[2] = 0x078e0200 | (uint32_t)(src3 & 0xff);
                w[3] = insn->control ? insn->control : 0x001fca00;
            }
            break;

        case BW_IR_IMAD_WIDE:
            /* IMAD.WIDE.U32 Rd:Rd+1, Ra, imm, Rc:Rc+1 */
            w[0] = 0x7825 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = insn->imm;
            w[2] = 0x078e0000 | (uint32_t)(src3 & 0xff);
            w[3] = insn->control ? insn->control : 0x001fcc00;
            break;

        case BW_IR_ISETP_GE:
            if (insn->is_uniform && insn->ureg >= 0) {
                w[0] = 0x7c0c | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
                w[1] = (uint32_t)(ureg & 0xff);
                w[2] = 0x0bf06070;
                w[3] = 0x002fda00;
            } else {
                w[0] = 0x780c | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
                w[1] = insn->imm;
                w[2] = 0x03f06270;
                w[3] = 0x001fda00;
            }
            break;

        case BW_IR_LDG_E:
            /* LDG.E Rd, desc[URd][Ra.64] */
            w[0] = 0x7981 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(ureg & 0xff);
            w[2] = 0x0c1e1900;
            w[3] = insn->control ? insn->control : 0x000f2200;
            break;

        case BW_IR_STG_E:
            /* STG.E desc[URd][Ra.64], Rb */
            w[0] = 0x7986 | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x0c101900 | (uint32_t)(ureg & 0xff);
            w[3] = insn->control ? insn->control : 0x000fe200;
            break;

        case BW_IR_IADD3:
            /* IADD3 Rd, PT, PT, Ra, Rb, Rc */
            w[0] = 0x7210 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x07ffe0ff;
            w[3] = 0x010fca00;
            break;

        case BW_IR_EXIT:
            w[0] = insn->predicate_p0 ? 0x0000094d : 0x0000794d;
            w[1] = 0x00000000;
            w[2] = 0x03800000;
            w[3] = 0x000fea00;
            break;

        case BW_IR_BRA:
            w[0] = 0x00fc7947;
            w[1] = 0xfffffffc;
            w[2] = 0x0383ffff;
            w[3] = 0x000fc000;
            break;

        default:
            return -1;
    }
    return 0;
}

int omega_bw_encode_program(const BlackwellIRProgram *prog, uint8_t *code_buf, size_t max_len, size_t *out_len) {
    if (!prog || !code_buf || !out_len) return -1;

    size_t padded_insns = (prog->count + 7) & ~7ULL;
    if (padded_insns < 32) padded_insns = 32;

    size_t req_bytes = padded_insns * 16;
    if (max_len < req_bytes) return -1;

    uint8_t *p = code_buf;
    for (size_t i = 0; i < padded_insns; i++) {
        uint32_t w[4];
        if (i < prog->count) {
            if (encode_single_insn(&prog->insns[i], &prog->regalloc, w) != 0) {
                return -1;
            }
        } else {
            /* Pad with NOP */
            w[0] = 0x00007918;
            w[1] = 0x00000000;
            w[2] = 0x00000000;
            w[3] = 0x000fc000;
        }

        p[0] = (uint8_t)(w[0] & 0xff);
        p[1] = (uint8_t)((w[0] >> 8) & 0xff);
        p[2] = (uint8_t)((w[0] >> 16) & 0xff);
        p[3] = (uint8_t)((w[0] >> 24) & 0xff);

        p[4] = (uint8_t)(w[1] & 0xff);
        p[5] = (uint8_t)((w[1] >> 8) & 0xff);
        p[6] = (uint8_t)((w[1] >> 16) & 0xff);
        p[7] = (uint8_t)((w[1] >> 24) & 0xff);

        p[8] = (uint8_t)(w[2] & 0xff);
        p[9] = (uint8_t)((w[2] >> 8) & 0xff);
        p[10] = (uint8_t)((w[2] >> 16) & 0xff);
        p[11] = (uint8_t)((w[2] >> 24) & 0xff);

        p[12] = (uint8_t)(w[3] & 0xff);
        p[13] = (uint8_t)((w[3] >> 8) & 0xff);
        p[14] = (uint8_t)((w[3] >> 16) & 0xff);
        p[15] = (uint8_t)((w[3] >> 24) & 0xff);

        p += 16;
    }

    *out_len = req_bytes;
    return 0;
}

int omega_blackwell_codegen_matmul_i32(const OmegaMatMulSpec *spec, OmegaBlackwellKernel *kernel) {
    if (!spec || !kernel) return -1;
    if (spec->m == 0 || spec->k == 0 || spec->n == 0) return -1;

    /* Fail closed loudly if K exceeds bounded unroll limit */
    if (spec->k > 32) {
        fprintf(stderr, "ERROR: MatMul integer codegen exceeds bounded unroll limit (K=%u > 32)\n", spec->k);
        return -1;
    }

    BlackwellIRProgram prog;
    omega_bw_ir_init(&prog);

    /* 1. Allocate virtual registers */
    /* 64-bit base pointers */
    int v_ptr_a   = omega_bw_ir_alloc_vreg64(&prog);
    int v_ptr_b   = omega_bw_ir_alloc_vreg64(&prog);
    int v_ptr_c   = omega_bw_ir_alloc_vreg64(&prog);
    int uv_desc   = omega_bw_ir_alloc_uvreg64(&prog);

    /* 32-bit CTA / Thread indexing */
    int v_ctaid_x = omega_bw_ir_alloc_vreg(&prog);
    int v_ctaid_y = omega_bw_ir_alloc_vreg(&prog);
    int v_tid_x   = omega_bw_ir_alloc_vreg(&prog);
    int v_tid_y   = omega_bw_ir_alloc_vreg(&prog);

    /* 2D Coordinates & loop invariants */
    int v_col     = omega_bw_ir_alloc_vreg(&prog);
    int v_row     = omega_bw_ir_alloc_vreg(&prog);
    int v_row_k   = omega_bw_ir_alloc_vreg(&prog);
    int v_const_1 = omega_bw_ir_alloc_vreg(&prog);
    int v_acc     = omega_bw_ir_alloc_vreg(&prog);

    /* Working registers for memory address and operands */
    int v_idx_a   = omega_bw_ir_alloc_vreg(&prog);
    int v_off_a   = omega_bw_ir_alloc_vreg64(&prog);
    int v_val_a   = omega_bw_ir_alloc_vreg(&prog);

    int v_idx_b   = omega_bw_ir_alloc_vreg(&prog);
    int v_off_b   = omega_bw_ir_alloc_vreg64(&prog);
    int v_val_b   = omega_bw_ir_alloc_vreg(&prog);

    int v_idx_c   = omega_bw_ir_alloc_vreg(&prog);
    int v_off_c   = omega_bw_ir_alloc_vreg64(&prog);

    if (v_off_c < 0) return -1;

    /* 2. Load 64-bit global memory descriptor into uniform register pair */
    /* LDCU.64 uv_desc, c[0x0][0x358] */
    BlackwellIRInsn insn_desc = { .op = BW_IR_LDCU64, .dst_vreg = uv_desc, .imm = 0x358, .is_uniform = true };
    omega_bw_ir_append(&prog, &insn_desc);

    /* 3. Load 64-bit base pointers from Constant Bank 0 */
    /* LDC.64 v_ptr_a, c[0x0][0x380] */
    BlackwellIRInsn insn_ptra = { .op = BW_IR_LDC64, .dst_vreg = v_ptr_a, .imm = 0x380 };
    omega_bw_ir_append(&prog, &insn_ptra);

    /* LDC.64 v_ptr_b, c[0x0][0x388] */
    BlackwellIRInsn insn_ptrb = { .op = BW_IR_LDC64, .dst_vreg = v_ptr_b, .imm = 0x388 };
    omega_bw_ir_append(&prog, &insn_ptrb);

    /* LDC.64 v_ptr_c, c[0x0][0x390] */
    BlackwellIRInsn insn_ptrc = { .op = BW_IR_LDC64, .dst_vreg = v_ptr_c, .imm = 0x390 };
    omega_bw_ir_append(&prog, &insn_ptrc);

    /* 4. Query 2D CTA and Thread coordinates */
    /* S2R v_ctaid_x, SR_CTAID.X */
    BlackwellIRInsn insn_ctax = { .op = BW_IR_S2R, .dst_vreg = v_ctaid_x, .imm = BW_SR_CTAID_X };
    omega_bw_ir_append(&prog, &insn_ctax);

    /* S2R v_ctaid_y, SR_CTAID.Y */
    BlackwellIRInsn insn_ctay = { .op = BW_IR_S2R, .dst_vreg = v_ctaid_y, .imm = BW_SR_CTAID_Y };
    omega_bw_ir_append(&prog, &insn_ctay);

    /* S2R v_tid_x, SR_TID.X */
    BlackwellIRInsn insn_tidx = { .op = BW_IR_S2R, .dst_vreg = v_tid_x, .imm = BW_SR_TID_X };
    omega_bw_ir_append(&prog, &insn_tidx);

    /* S2R v_tid_y, SR_TID.Y */
    BlackwellIRInsn insn_tidy = { .op = BW_IR_S2R, .dst_vreg = v_tid_y, .imm = BW_SR_TID_Y };
    omega_bw_ir_append(&prog, &insn_tidy);

    /* 5. Compute 2D (row, col) coordinates */
    /* Standard 16x16 tile threads: col = ctaid_x * 16 + tid_x */
    BlackwellIRInsn insn_col = { .op = BW_IR_IMAD, .dst_vreg = v_col, .src1_vreg = v_ctaid_x, .src2_vreg = -1, .imm = 16, .src3_vreg = v_tid_x };
    omega_bw_ir_append(&prog, &insn_col);

    /* row = ctaid_y * 16 + tid_y */
    BlackwellIRInsn insn_row = { .op = BW_IR_IMAD, .dst_vreg = v_row, .src1_vreg = v_ctaid_y, .src2_vreg = -1, .imm = 16, .src3_vreg = v_tid_y };
    omega_bw_ir_append(&prog, &insn_row);

    /* 6. Precompute row base index: v_row_k = v_row * K + RZ */
    BlackwellIRInsn insn_row_k = { .op = BW_IR_IMAD, .dst_vreg = v_row_k, .src1_vreg = v_row, .src2_vreg = -1, .imm = spec->k, .src3_vreg = -1 };
    omega_bw_ir_append(&prog, &insn_row_k);

    /* 7. Initialize v_const_1 = 1 for immediate linear indexing */
    BlackwellIRInsn insn_c1 = { .op = BW_IR_MOV_IMM, .dst_vreg = v_const_1, .imm = 1 };
    omega_bw_ir_append(&prog, &insn_c1);

    /* 8. Initialize accumulator to zero: MOV v_acc, RZ */
    BlackwellIRInsn insn_init_acc = { .op = BW_IR_MOV_RZ, .dst_vreg = v_acc };
    omega_bw_ir_append(&prog, &insn_init_acc);

    /* 9. Synthesize inner accumulation loop across K iterations */
    for (uint32_t step = 0; step < spec->k; step++) {
        /* A[row, step]: idx_a = v_row_k + step = v_const_1 * step + v_row_k */
        BlackwellIRInsn insn_idxa = { .op = BW_IR_IMAD, .dst_vreg = v_idx_a, .src1_vreg = v_const_1, .src2_vreg = -1, .imm = step, .src3_vreg = v_row_k };
        omega_bw_ir_append(&prog, &insn_idxa);

        /* off_a = v_ptr_a + idx_a * 4 */
        BlackwellIRInsn insn_offa = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off_a, .src1_vreg = v_idx_a, .src2_vreg = -1, .imm = 4, .src3_vreg = v_ptr_a };
        omega_bw_ir_append(&prog, &insn_offa);

        /* Load A[row, step]: LDG.E v_val_a, desc[uv_desc][v_off_a.64] */
        BlackwellIRInsn insn_lda = { .op = BW_IR_LDG_E, .dst_vreg = v_val_a, .src1_vreg = v_off_a, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = 0x001ea800 };
        omega_bw_ir_append(&prog, &insn_lda);

        /* B[step, col]: idx_b = step * N + col = v_const_1 * (step * N) + v_col */
        uint32_t step_n = step * spec->n;
        BlackwellIRInsn insn_idxb = { .op = BW_IR_IMAD, .dst_vreg = v_idx_b, .src1_vreg = v_const_1, .src2_vreg = -1, .imm = step_n, .src3_vreg = v_col };
        omega_bw_ir_append(&prog, &insn_idxb);

        /* off_b = v_ptr_b + idx_b * 4 */
        BlackwellIRInsn insn_offb = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off_b, .src1_vreg = v_idx_b, .src2_vreg = -1, .imm = 4, .src3_vreg = v_ptr_b };
        omega_bw_ir_append(&prog, &insn_offb);

        /* Load B[step, col]: LDG.E v_val_b, desc[uv_desc][v_off_b.64] */
        BlackwellIRInsn insn_ldb = { .op = BW_IR_LDG_E, .dst_vreg = v_val_b, .src1_vreg = v_off_b, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = 0x002ea800 };
        omega_bw_ir_append(&prog, &insn_ldb);

        /* Multiply and accumulate: IMAD v_acc, v_val_a, v_val_b, v_acc */
        BlackwellIRInsn insn_acc = { .op = BW_IR_IMAD, .dst_vreg = v_acc, .src1_vreg = v_val_a, .src2_vreg = v_val_b, .src3_vreg = v_acc, .control = 0x004fca00 };
        omega_bw_ir_append(&prog, &insn_acc);
    }

    /* 10. Compute output index: idx_c = row * N + col */
    BlackwellIRInsn insn_idxc = { .op = BW_IR_IMAD, .dst_vreg = v_idx_c, .src1_vreg = v_row, .src2_vreg = -1, .imm = spec->n, .src3_vreg = v_col };
    omega_bw_ir_append(&prog, &insn_idxc);

    /* off_c = v_ptr_c + idx_c * 4 */
    BlackwellIRInsn insn_offc = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off_c, .src1_vreg = v_idx_c, .src2_vreg = -1, .imm = 4, .src3_vreg = v_ptr_c };
    omega_bw_ir_append(&prog, &insn_offc);

    /* Store C[row, col]: STG.E desc[uv_desc][v_off_c.64], v_acc */
    BlackwellIRInsn insn_stc = { .op = BW_IR_STG_E, .src1_vreg = v_off_c, .src2_vreg = v_acc, .src3_vreg = -1, .ureg = uv_desc };
    omega_bw_ir_append(&prog, &insn_stc);

    /* 11. Kernel termination */
    BlackwellIRInsn insn_exit = { .op = BW_IR_EXIT, .predicate_p0 = false };
    omega_bw_ir_append(&prog, &insn_exit);

    BlackwellIRInsn insn_bra = { .op = BW_IR_BRA };
    omega_bw_ir_append(&prog, &insn_bra);

    /* 12. Solve bounded deterministic register allocation */
    if (omega_bw_regalloc_solve(&prog) != 0) {
        return -1;
    }

    /* 13. Allocate code buffer and encode instructions */
    size_t padded_insns = (prog.count + 7) & ~7ULL;
    if (padded_insns < 32) padded_insns = 32;
    size_t alloc_size = padded_insns * 16;

    kernel->code = (uint8_t *)malloc(alloc_size);
    if (!kernel->code) return -1;

    size_t emitted_size = 0;
    if (omega_bw_encode_program(&prog, kernel->code, alloc_size, &emitted_size) != 0) {
        free(kernel->code);
        kernel->code = NULL;
        return -1;
    }

    kernel->code_size = emitted_size;
    kernel->insn_count = emitted_size / 16;
    kernel->gpr_count = prog.regalloc.peak_gpr_usage;
    kernel->uniform_gpr_count = prog.regalloc.peak_ugpr_usage;

    /* 14. Compute runtime SHA-256 digest of emitted machine code */
    sha256_hash(kernel->code, kernel->code_size, kernel->code_digest);
    return 0;
}

int omega_blackwell_verify_codegen_fixtures(void) {
    OmegaRegAlloc ra;
    memset(&ra, 0, sizeof(ra));
    ra.vreg_to_phys[0] = 2;
    ra.vreg_to_phys[1] = 4;
    ra.vreg_to_phys[2] = 6;
    ra.vreg_to_phys[3] = 7;
    ra.uvreg_to_phys[0] = 4;

    uint32_t w[4];

    /* 1. MOV R7, RZ */
    BlackwellIRInsn insn_mov_rz = { .op = BW_IR_MOV_RZ, .dst_vreg = 3 };
    if (encode_single_insn(&insn_mov_rz, &ra, w) != 0) return -1;
    if (w[0] != (0x7202 | (7 << 16)) || w[1] != 0xff || w[2] != 0x00000f00) return -2;

    /* 2. MOV R2, 1 */
    BlackwellIRInsn insn_mov_imm = { .op = BW_IR_MOV_IMM, .dst_vreg = 0, .imm = 1 };
    if (encode_single_insn(&insn_mov_imm, &ra, w) != 0) return -3;
    if (w[0] != (0x7802 | (2 << 16)) || w[1] != 1 || w[2] != 0x00000f00) return -4;

    /* 3. Register-Register IMAD: IMAD R7, R2, R6, R7 */
    BlackwellIRInsn insn_imad_rr = { .op = BW_IR_IMAD, .dst_vreg = 3, .src1_vreg = 0, .src2_vreg = 2, .src3_vreg = 3 };
    if (encode_single_insn(&insn_imad_rr, &ra, w) != 0) return -5;
    if (w[0] != (0x7224 | (7 << 16) | (2 << 24)) || w[1] != 6 || w[2] != (0x078e0200 | 7)) return -6;

    /* 4. Immediate IMAD: IMAD R6, R2, 16, RZ */
    BlackwellIRInsn insn_imad_imm = { .op = BW_IR_IMAD, .dst_vreg = 2, .src1_vreg = 0, .src2_vreg = -1, .imm = 16, .src3_vreg = -1 };
    if (encode_single_insn(&insn_imad_imm, &ra, w) != 0) return -7;
    if (w[0] != (0x7824 | (6 << 16) | (2 << 24)) || w[1] != 16 || w[2] != 0x078e02ff) return -8;

    /* 5. IMAD.WIDE: IMAD.WIDE R4, R2, 4, R4 */
    BlackwellIRInsn insn_imad_wide = { .op = BW_IR_IMAD_WIDE, .dst_vreg = 1, .src1_vreg = 0, .imm = 4, .src3_vreg = 1 };
    if (encode_single_insn(&insn_imad_wide, &ra, w) != 0) return -9;
    if (w[0] != (0x7825 | (4 << 16) | (2 << 24)) || w[1] != 4 || w[2] != (0x078e0000 | 4)) return -10;

    /* 6. LDC.64 R4, c[0x0][0x380] */
    BlackwellIRInsn insn_ldc64 = { .op = BW_IR_LDC64, .dst_vreg = 1, .imm = 0x380 };
    if (encode_single_insn(&insn_ldc64, &ra, w) != 0) return -11;
    if (w[0] != (0xff007b82 | (4 << 16)) || w[1] != ((0x380 << 6) & 0x00ffff00) || w[2] != 0x00000a00) return -12;

    /* 7. LDCU.64 UR4, c[0x0][0x358] */
    BlackwellIRInsn insn_ldcu64 = { .op = BW_IR_LDCU64, .dst_vreg = 0, .imm = 0x358, .is_uniform = true };
    if (encode_single_insn(&insn_ldcu64, &ra, w) != 0) return -13;
    if (w[0] != (0xff0077ac | (4 << 16)) || w[1] != ((0x358 << 5) & 0x00ffff00) || w[2] != 0x08000a00) return -14;

    /* 8. LDG.E R2, desc[UR4][R4.64] */
    BlackwellIRInsn insn_ldg = { .op = BW_IR_LDG_E, .dst_vreg = 0, .src1_vreg = 1, .ureg = 0 };
    if (encode_single_insn(&insn_ldg, &ra, w) != 0) return -15;
    if (w[0] != (0x7981 | (2 << 16) | (4 << 24)) || w[1] != 4 || w[2] != 0x0c1e1900) return -16;

    /* 9. STG.E desc[UR4][R4.64], R7 */
    BlackwellIRInsn insn_stg = { .op = BW_IR_STG_E, .src1_vreg = 1, .src2_vreg = 3, .ureg = 0 };
    if (encode_single_insn(&insn_stg, &ra, w) != 0) return -17;
    if (w[0] != (0x7986 | (4 << 24)) || w[1] != 7 || w[2] != (0x0c101900 | 4)) return -18;

    /* 10. S2R R2, SR_CTAID.X */
    BlackwellIRInsn insn_s2r = { .op = BW_IR_S2R, .dst_vreg = 0, .imm = BW_SR_CTAID_X };
    if (encode_single_insn(&insn_s2r, &ra, w) != 0) return -19;
    if (w[0] != (0x7919 | (2 << 16)) || w[2] != (BW_SR_CTAID_X << 8)) return -20;

    return 0;
}

int omega_blackwell_test_regalloc_bounds(void) {
    BlackwellIRProgram prog;
    omega_bw_ir_init(&prog);

    int v0 = omega_bw_ir_alloc_vreg64(&prog);
    int v1 = omega_bw_ir_alloc_vreg(&prog);
    int v2 = omega_bw_ir_alloc_vreg64(&prog);

    BlackwellIRInsn i0 = { .op = BW_IR_LDC64, .dst_vreg = v0, .imm = 0x380 };
    omega_bw_ir_append(&prog, &i0);
    BlackwellIRInsn i1 = { .op = BW_IR_MOV_IMM, .dst_vreg = v1, .imm = 10 };
    omega_bw_ir_append(&prog, &i1);
    BlackwellIRInsn i2 = { .op = BW_IR_LDC64, .dst_vreg = v2, .imm = 0x388 };
    omega_bw_ir_append(&prog, &i2);

    if (omega_bw_regalloc_solve(&prog) != 0) return -1;

    int phys_v0 = prog.regalloc.vreg_to_phys[v0];
    int phys_v1 = prog.regalloc.vreg_to_phys[v1];
    int phys_v2 = prog.regalloc.vreg_to_phys[v2];

    if (phys_v0 % 2 != 0) return -2;
    if (phys_v2 % 2 != 0) return -3;
    if (phys_v1 == phys_v0 || phys_v1 == phys_v0 + 1) return -4;
    if (phys_v2 == phys_v0 || phys_v2 == phys_v0 + 1) return -5;

    /* Test bounds refusal: intentionally attempt to allocate more registers than hardware capacity */
    BlackwellIRProgram overflow_prog;
    omega_bw_ir_init(&overflow_prog);
    for (int i = 0; i < BW_PHYS_GPR_MAX + 5; i++) {
        int v = omega_bw_ir_alloc_vreg(&overflow_prog);
        if (v < 0) break;
        BlackwellIRInsn insn = { .op = BW_IR_MOV_IMM, .dst_vreg = v, .imm = (uint32_t)i };
        omega_bw_ir_append(&overflow_prog, &insn);
    }
    for (size_t i = 0; i < overflow_prog.count; i++) {
        for (int v = 0; v < overflow_prog.regalloc.num_vregs; v++) {
            overflow_prog.regalloc.intervals[v].last_use = (int)overflow_prog.count - 1;
        }
    }
    if (omega_bw_regalloc_solve(&overflow_prog) == 0) {
        return -6;
    }

    return 0;
}

int omega_blackwell_test_codegen_variation(void) {
    OmegaMatMulSpec spec1, spec2;
    if (omega_matmul_spec_init(&spec1, 16, 16, 16, OMEGA_MATMUL_PRECISION_INT32) != 0) {
        printf("FAIL: spec1 init failed\n");
        return -1;
    }
    if (omega_matmul_spec_init(&spec2, 32, 16, 64, OMEGA_MATMUL_PRECISION_INT32) != 0) {
        printf("FAIL: spec2 init failed\n");
        return -1;
    }

    OmegaBlackwellKernel k1, k2;
    memset(&k1, 0, sizeof(k1));
    memset(&k2, 0, sizeof(k2));

    if (omega_blackwell_codegen_matmul(&spec1, &k1) != 0) {
        printf("FAIL: k1 codegen failed\n");
        return -1;
    }
    if (omega_blackwell_codegen_matmul(&spec2, &k2) != 0) {
        printf("FAIL: k2 codegen failed\n");
        omega_blackwell_kernel_free(&k1);
        return -1;
    }

    OmegaBlackwellRealizationIdentity id1, id2;
    omega_blackwell_bind_matmul_realization(&spec1, &k1, &id1);
    omega_blackwell_bind_matmul_realization(&spec2, &k2, &id2);

    printf("================================================================================\n");
    printf("    OMEGA BLACKWELL CODEGEN VARIATION DEMONSTRATION (GATE 7 STAGE-1)\n");
    printf("================================================================================\n");
    printf("Config 1: 16x16x16 INT32\n");
    printf("  Instructions: %zu (%zu bytes, %u GPRs, %u UGPRs)\n", k1.insn_count, k1.code_size, k1.gpr_count, k1.uniform_gpr_count);
    printf("  Spec ID:      ");
    for (int i = 0; i < 16; i++) printf("%02x", spec1.spec_id[i]);
    printf("...\n");
    printf("  Code Digest:  ");
    for (int i = 0; i < 16; i++) printf("%02x", k1.code_digest[i]);
    printf("...\n");
    printf("  Realization:  ");
    for (int i = 0; i < 16; i++) printf("%02x", id1.realization_id[i]);
    printf("...\n\n");

    printf("Config 2: 32x16x64 INT32\n");
    printf("  Instructions: %zu (%zu bytes, %u GPRs, %u UGPRs)\n", k2.insn_count, k2.code_size, k2.gpr_count, k2.uniform_gpr_count);
    printf("  Spec ID:      ");
    for (int i = 0; i < 16; i++) printf("%02x", spec2.spec_id[i]);
    printf("...\n");
    printf("  Code Digest:  ");
    for (int i = 0; i < 16; i++) printf("%02x", k2.code_digest[i]);
    printf("...\n");
    printf("  Realization:  ");
    for (int i = 0; i < 16; i++) printf("%02x", id2.realization_id[i]);
    printf("...\n");
    printf("================================================================================\n");

    /* Validation checks */
    if (memcmp(spec1.spec_id, spec2.spec_id, 32) == 0) {
        printf("FAIL: spec IDs match (expected distinct)\n");
        return -2;
    }
    if (memcmp(k1.code_digest, k2.code_digest, 32) == 0) {
        printf("FAIL: code digests match (expected distinct)\n");
        return -3;
    }
    if (memcmp(id1.realization_id, id2.realization_id, 32) == 0) {
        printf("FAIL: realization IDs match (expected distinct)\n");
        return -4;
    }

    printf("PASS: Dynamic codegen variation verified (distinct specifications, code digests, and realization IDs).\n");

    omega_blackwell_kernel_free(&k1);
    omega_blackwell_kernel_free(&k2);
    return 0;
}
