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
    prog->regalloc.intervals[v].bundle_size = 1;
    prog->regalloc.intervals[v].is_pair = false;
    prog->regalloc.intervals[v].is_quad = false;
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
    prog->regalloc.intervals[v].bundle_size = 2;
    prog->regalloc.intervals[v].is_pair = true;
    prog->regalloc.intervals[v].is_quad = false;
    prog->regalloc.intervals[v].active = false;
    prog->regalloc.vreg_to_phys[v] = -1;
    return v;
}

int omega_bw_ir_alloc_vreg128(BlackwellIRProgram *prog) {
    if (!prog || prog->regalloc.num_vregs >= BW_MAX_VREGS) return -1;
    int v = prog->regalloc.num_vregs++;
    prog->regalloc.intervals[v].vreg = v;
    prog->regalloc.intervals[v].first_def = -1;
    prog->regalloc.intervals[v].last_use = -1;
    prog->regalloc.intervals[v].phys_reg = -1;
    prog->regalloc.intervals[v].bundle_size = 4;
    prog->regalloc.intervals[v].is_pair = false;
    prog->regalloc.intervals[v].is_quad = true;
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

    /* Sort virtual register indices chronologically by first_def (Poletto & Sarkar Linear Scan) */
    int order[BW_MAX_VREGS];
    int count = 0;
    for (int v = 0; v < ra->num_vregs; v++) {
        if (ra->intervals[v].first_def >= 0) {
            order[count++] = v;
        } else {
            ra->intervals[v].phys_reg = BW_PHYS_GPR_START;
            ra->vreg_to_phys[v] = ra->intervals[v].phys_reg;
        }
    }
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (ra->intervals[order[i]].first_def > ra->intervals[order[j]].first_def) {
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }

    for (int idx = 0; idx < count; idx++) {
        int v = order[idx];
        OmegaLiveInterval *iv = &ra->intervals[v];

        /* Free physical registers whose virtual live intervals have ended */
        for (int p = 0; p < ra->num_vregs; p++) {
            if (ra->intervals[p].active && ra->intervals[p].last_use < iv->first_def) {
                int freed = ra->intervals[p].phys_reg;
                if (freed >= 0 && freed < BW_PHYS_GPR_MAX) {
                    uint32_t bsize = ra->intervals[p].bundle_size ? ra->intervals[p].bundle_size : (ra->intervals[p].is_pair ? 2 : (ra->intervals[p].is_quad ? 4 : 1));
                    for (uint32_t k = 0; k < bsize && (freed + (int)k) < BW_PHYS_GPR_MAX; k++) {
                        gpr_busy[freed + k] = false;
                    }
                }
                ra->intervals[p].active = false;
            }
        }

        /* Find lowest available physical register */
        int assigned = -1;
        uint32_t bsize = iv->bundle_size ? iv->bundle_size : (iv->is_pair ? 2 : (iv->is_quad ? 4 : 1));
        if (bsize == 4) {
            /* 128-bit quad requires 4-alignment and 4 contiguous free registers */
            int start = (BW_PHYS_GPR_START + 3) & ~3;
            for (int p = start; p <= BW_PHYS_GPR_MAX - 4; p += 4) {
                if (!gpr_busy[p] && !gpr_busy[p + 1] && !gpr_busy[p + 2] && !gpr_busy[p + 3]) {
                    assigned = p;
                    for (int k = 0; k < 4; k++) gpr_busy[p + k] = true;
                    break;
                }
            }
        } else if (bsize == 2) {
            /* 64-bit pair requires even alignment and contiguous free registers */
            int start = (BW_PHYS_GPR_START + 1) & ~1;
            for (int p = start; p <= BW_PHYS_GPR_MAX - 2; p += 2) {
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

        int top = assigned + (int)bsize - 1;
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

    int dst = (insn->dst_vreg >= 0) ? (ra->vreg_to_phys[insn->dst_vreg] + insn->dst_subreg) : 0;
    int src1 = (insn->src1_vreg >= 0) ? (ra->vreg_to_phys[insn->src1_vreg] + insn->src1_subreg) : 0;
    int src2 = (insn->src2_vreg >= 0) ? (ra->vreg_to_phys[insn->src2_vreg] + insn->src2_subreg) : 0;
    int src3 = (insn->src3_vreg >= 0) ? (ra->vreg_to_phys[insn->src3_vreg] + insn->src3_subreg) : 0xff; /* Default to RZ (0xff) */
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

        case BW_IR_STG_EF:
            /* STG.E.MMIO.GPU. Uncached store, meant to be visible without a fence. */
            w[0] = 0x7986 | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x0c111900 | (uint32_t)(ureg & 0xff);
            w[3] = insn->control ? insn->control : 0x000fe200;
            break;

        case BW_IR_IADD3:
            /* IADD3 Rd, PT, PT, Ra, Rb, Rc */
            w[0] = 0x7210 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x07ffe0ff;
            w[3] = 0x010fca00;
            break;

        case BW_IR_HMMA_F16:
        case BW_IR_HMMA_BF16: {
            /* HMMA.16816.F32[.BF16] Rd, Ra, Rb, Rc */
            w[0] = 0x723c | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            uint32_t prec_bit = (insn->op == BW_IR_HMMA_BF16) ? 0x00040000 : 0x00000000;
            w[2] = (uint32_t)(src3 & 0xff) | (0x18 << 8) | prec_bit;
            w[3] = insn->control ? insn->control : 0x000fe200;
            break;
        }

        case BW_IR_SHF_R:
            /* SHF.R.U32.HI Rd, RZ, imm, Ra */
            w[0] = 0x7819U | ((uint32_t)(dst & 0xff) << 16) | (0xffU << 24);
            w[1] = (uint32_t)insn->imm;
            w[2] = 0x00011600 | (uint32_t)(src1 & 0xff);
            w[3] = insn->control ? insn->control : 0x001fca00;
            break;

        case BW_IR_LOP3_AND:
            /* LOP3.LUT Rd, Ra, imm, RZ, 0xc0, !PT */
            w[0] = 0x7812U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)insn->imm;
            w[2] = 0x078ec0ff;
            w[3] = insn->control ? insn->control : 0x001fca00;
            break;

        case BW_IR_LDG_E_U16:
            /* LDG.E.U16 Rd, desc[URd][Ra.64] */
            w[0] = 0x7981U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(ureg & 0xff);
            w[2] = 0x0c1e1500;
            w[3] = insn->control ? insn->control : 0x000f2200;
            break;

        case BW_IR_LDG_STRONG_SYS:
            /* LDG.E.STRONG.SYS. Same shape as LDG.E; w[2] carries the
             * system-scope bits (0x0c1e1900 -> 0x0c1f5900). */
            w[0] = 0x7981 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(ureg & 0xff);
            w[2] = 0x0c1f5900;
            w[3] = insn->control ? insn->control : 0x000f2200;
            break;

        case BW_IR_LDG_MMIO:
            /* LDG.E.EF. Drop the cached line, then load. */
            w[0] = 0x7981 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(ureg & 0xff);
            w[2] = 0x0c0e1900;
            w[3] = insn->control ? insn->control : 0x000f2200;
            break;

        case BW_IR_STG_STRONG_SYS:
            w[0] = 0x7986 | ((uint32_t)(src1 & 0xff) << 24);
            if (insn->predicate_p0) {
                uint32_t pred = insn->predicate_not ? 0x8u : 0x0u;
                w[0] = (w[0] & ~0xF000u) | (pred << 12);
            }
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x0c115900 | (uint32_t)(ureg & 0xff);
            w[3] = insn->control ? insn->control : 0x000fe200;
            break;

        case BW_IR_MEMBAR_ALL_SYS:
            w[0] = 0x00007992;
            w[1] = 0x00000000;
            w[2] = 0x0000b000;
            w[3] = insn->control ? insn->control : 0x000fec00;
            break;

        case BW_IR_MEMBAR_SC_SYS:
            w[0] = 0x00007992;
            w[1] = 0x00000000;
            w[2] = 0x00003000;
            w[3] = insn->control ? insn->control : 0x000fec00;
            break;

        case BW_IR_CCTL_IVALL:
            w[0] = 0xff00798f;
            w[1] = 0x00000000;
            w[2] = 0x02000000;
            w[3] = insn->control ? insn->control : 0x000fe800;
            break;

        case BW_IR_ATOMG_ADD_STRONG_SYS:
            w[0] = 0x79a8 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff) | 0x80000000u;
            w[2] = 0x081f5100 | (uint32_t)(ureg & 0xff);
            w[3] = insn->control ? insn->control : 0x00321e00;
            break;

        case BW_IR_ATOMG_EXCH_STRONG_SYS:
            w[0] = 0x79a8 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff) | 0x80000000u;
            w[2] = 0x0c1f5100 | (uint32_t)(ureg & 0xff);
            w[3] = insn->control ? insn->control : 0x00321e00;
            break;

        case BW_IR_ISETP_GE_U32:
            /* ISETP.GE.U32.AND P0, PT, Ra, Rb, PT. Checked with nvdisasm -b SM121. */
            w[0] = 0x720c | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x03f06070;
            w[3] = insn->control ? insn->control : 0x001fda00;
            break;

        case BW_IR_LOP3_XOR:
            /* LOP3.LUT Rd, Ra, Rb, RZ, 0x3c, !PT. Checked with nvdisasm -b SM121. */
            w[0] = 0x7212 | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x078e3cff;
            w[3] = insn->control ? insn->control : 0x001fca00;
        case BW_IR_FADD:
            /* FADD Rd, Ra, Rb */
            w[0] = 0x7221U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x00000000;
            w[3] = insn->control ? insn->control : 0x004fc400;
            break;

        case BW_IR_FSUB:
            /* FSUB Rd, Ra, Rb (FADD Rd, Ra, -Rb) */
            w[0] = 0x7221U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff) | 0x80000000U;
            w[2] = 0x00000000;
            w[3] = insn->control ? insn->control : 0x004fc400;
            break;

        case BW_IR_FMUL:
            /* FMUL Rd, Ra, Rb */
            w[0] = 0x7220U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x00400000;
            w[3] = insn->control ? insn->control : 0x004fc400;
            break;

        case BW_IR_FFMA:
            /* FFMA Rd, Ra, Rb, Rc */
            w[0] = 0x7223U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = (uint32_t)(src3 & 0xff);
            w[3] = insn->control ? insn->control : 0x004fc400;
            break;

        case BW_IR_FSETP:
            /* FSETP.cond.AND P0, PT, Ra, Rb, PT */
            w[0] = 0x720bU | ((uint32_t)(dst & 0x7) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = insn->imm ? insn->imm : 0x03f06000; /* Default GE condition */
            w[3] = insn->control ? insn->control : 0x004fc400;
            break;

        case BW_IR_FSEL:
            /* FSEL Rd, Ra, Rb, P0 */
            w[0] = 0x7208U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = (uint32_t)(src3 & 0x7);
            w[3] = insn->control ? insn->control : 0x000fca00;
            break;

        case BW_IR_FMNMX_MIN:
            /* FMNMX Rd, Ra, Rb, PT (minimum) */
            w[0] = 0x7209U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x03800000;
            w[3] = insn->control ? insn->control : 0x004fc400;
            break;

        case BW_IR_FMNMX_MAX:
            /* FMNMX Rd, Ra, Rb, !PT (maximum) */
            w[0] = 0x7209U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x07800000;
            w[3] = insn->control ? insn->control : 0x004fc400;
            break;

        case BW_IR_I2FP:
            /* I2FP.F32.S32 Rd, Ra */
            w[0] = 0x7245U | ((uint32_t)(dst & 0xff) << 16);
            w[1] = (uint32_t)(src1 & 0xff);
            w[2] = 0x00201400;
            w[3] = insn->control ? insn->control : 0x004fe200;
            break;

        case BW_IR_F2I:
            /* F2I.TRUNC.NTZ Rd, Ra */
            w[0] = 0x7305U | ((uint32_t)(dst & 0xff) << 16);
            w[1] = (uint32_t)(src1 & 0xff);
            w[2] = 0x0020f100;
            w[3] = insn->control ? insn->control : 0x004e2200;
            break;

        case BW_IR_MUFU_RCP:
            /* MUFU.RCP Rd, Ra */
            w[0] = 0x7308U | ((uint32_t)(dst & 0xff) << 16);
            w[1] = (uint32_t)(src1 & 0xff);
            w[2] = 0x00001000;
            w[3] = insn->control ? insn->control : 0x000e2400;
            break;

        case BW_IR_MUFU_RSQ:
            /* MUFU.RSQ Rd, Ra */
            w[0] = 0x7308U | ((uint32_t)(dst & 0xff) << 16);
            w[1] = (uint32_t)(src1 & 0xff);
            w[2] = 0x00001400;
            w[3] = insn->control ? insn->control : 0x000e2400;
            break;

        case BW_IR_SHFL_DOWN:
            /* SHFL.DOWN PT, Rd, Ra, offset, 0x1f */
            w[0] = 0x7f89U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = 0x08001f00U | (((uint32_t)insn->imm & 0x1f) << 21);
            w[2] = 0x000e0000;
            w[3] = insn->control ? insn->control : 0x000e2400;
            break;

        case BW_IR_LDS:
            /* LDS Rd, [Ra] */
            w[0] = 0x7984U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = 0x00000000;
            w[2] = (uint32_t)(insn->imm & 0xffff);
            w[3] = insn->control ? insn->control : 0x000fe200;
            break;

        case BW_IR_STS:
            /* STS [Ra], Rb */
            w[0] = 0x7388U | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = (uint32_t)(insn->imm & 0xffff);
            w[3] = insn->control ? insn->control : 0x000fe200;
            break;

        case BW_IR_EXIT:
            w[0] = insn->predicate_p0 ? 0x0000094d : 0x0000794d;
            w[1] = 0x00000000;
            w[2] = 0x03800000;
            w[3] = 0x000fea00;
            break;

        case BW_IR_BRA: {
            /* sm_121 BRA. Distance is (target - this) - 1, split across the
             * low six bits at instruction bits 18..23 and the rest at bit 34.
             * Predicate bits 12..15: 7 = always, 0 = @P0, 8 = @!P0.
             * A zero delta with no predicate keeps the self-branch word the
             * existing kernels already emit after EXIT. */
            int32_t delta = (int32_t)insn->imm;
            int32_t rel = delta - 1;
            uint32_t pred_field = 0x7u;
            if (insn->predicate_p0)
                pred_field = insn->predicate_not ? 0x8u : 0x0u;
            w[0] = 0x947u | (pred_field << 12) | (((uint32_t)rel & 0x3fu) << 18);
            w[1] = (uint32_t)(rel >> 6) << 2;
            w[2] = (rel < 0) ? 0x0383ffffu : 0x03800000u;
            if (insn->control)
                w[3] = insn->control;
            else if (delta == 0 && !insn->predicate_p0)
                w[3] = 0x000fc000u;
            else
                w[3] = 0x000fea00u;
            break;
        }

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

int omega_blackwell_codegen_matmul_tensor_prog(const OmegaMatMulSpec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog) return -1;
    if (spec->m == 0 || spec->k == 0 || spec->n == 0) return -1;
    if (spec->precision != OMEGA_MATMUL_PRECISION_FP16 && spec->precision != OMEGA_MATMUL_PRECISION_BF16) return -1;
    if (spec->m % 16 != 0 || spec->n % 8 != 0 || spec->k != 16) {
        fprintf(stderr, "ERROR: MatMul tensor codegen requires M multiple of 16, N multiple of 8, K=16 (got %ux%ux%u)\n",
                spec->m, spec->k, spec->n);
        return -1;
    }

    omega_bw_ir_init(prog);

    /* 1. Allocate virtual registers */
    /* 64-bit base pointers */
    int v_ptr_a   = omega_bw_ir_alloc_vreg64(prog);
    int v_ptr_b   = omega_bw_ir_alloc_vreg64(prog);
    int v_ptr_c   = omega_bw_ir_alloc_vreg64(prog);
    int uv_desc   = omega_bw_ir_alloc_uvreg64(prog);

    /* 32-bit CTA / Thread indexing */
    int v_tid     = omega_bw_ir_alloc_vreg(prog);
    int v_ctaid_x = omega_bw_ir_alloc_vreg(prog);
    int v_ctaid_y = omega_bw_ir_alloc_vreg(prog);

    /* Tile and thread coordinates */
    int v_group_id   = omega_bw_ir_alloc_vreg(prog);
    int v_tid_in_grp = omega_bw_ir_alloc_vreg(prog);
    int v_row_base   = omega_bw_ir_alloc_vreg(prog);
    int v_col_base   = omega_bw_ir_alloc_vreg(prog);
    int v_r0         = omega_bw_ir_alloc_vreg(prog);
    int v_r1         = omega_bw_ir_alloc_vreg(prog);
    int v_col_b      = omega_bw_ir_alloc_vreg(prog);
    int v_tid_col2   = omega_bw_ir_alloc_vreg(prog);
    int v_const_1    = omega_bw_ir_alloc_vreg(prog);

    /* Bundles for HMMA */
    int v_ra = omega_bw_ir_alloc_vreg128(prog); /* Quad: Ra[0]..Ra[3] */
    int v_rb = omega_bw_ir_alloc_vreg64(prog);  /* Pair: Rb[0]..Rb[1] */
    int v_rd = omega_bw_ir_alloc_vreg128(prog); /* Quad: Rd[0]..Rd[3] */

    /* Scratch registers for indexing and loads */
    int v_idx    = omega_bw_ir_alloc_vreg(prog);
    int v_off    = omega_bw_ir_alloc_vreg64(prog);
    int v_b_tmp0 = omega_bw_ir_alloc_vreg(prog);
    int v_b_tmp1 = omega_bw_ir_alloc_vreg(prog);

    if (v_b_tmp1 < 0) return -1;

    /* 2. Load 64-bit global memory descriptor into uniform register pair */
    /* LDCU.64 uv_desc, c[0x0][0x358] */
    BlackwellIRInsn insn_desc = { .op = BW_IR_LDCU64, .dst_vreg = uv_desc, .imm = 0x358, .is_uniform = true };
    omega_bw_ir_append(prog, &insn_desc);

    /* 3. Load 64-bit base pointers from Constant Bank 0 */
    /* LDC.64 v_ptr_a, c[0x0][0x380] */
    BlackwellIRInsn insn_ptra = { .op = BW_IR_LDC64, .dst_vreg = v_ptr_a, .imm = 0x380 };
    omega_bw_ir_append(prog, &insn_ptra);

    /* LDC.64 v_ptr_b, c[0x0][0x388] */
    BlackwellIRInsn insn_ptrb = { .op = BW_IR_LDC64, .dst_vreg = v_ptr_b, .imm = 0x388 };
    omega_bw_ir_append(prog, &insn_ptrb);

    /* LDC.64 v_ptr_c, c[0x0][0x390] */
    BlackwellIRInsn insn_ptrc = { .op = BW_IR_LDC64, .dst_vreg = v_ptr_c, .imm = 0x390 };
    omega_bw_ir_append(prog, &insn_ptrc);

    /* 4. Query Special Registers */
    /* S2R v_tid, SR_TID.X */
    BlackwellIRInsn insn_tidx = { .op = BW_IR_S2R, .dst_vreg = v_tid, .imm = BW_SR_TID_X };
    omega_bw_ir_append(prog, &insn_tidx);

    /* S2R v_ctaid_x, SR_CTAID.X */
    BlackwellIRInsn insn_cta_x = { .op = BW_IR_S2R, .dst_vreg = v_ctaid_x, .imm = BW_SR_CTAID_X };
    omega_bw_ir_append(prog, &insn_cta_x);

    /* S2R v_ctaid_y, SR_CTAID.Y */
    BlackwellIRInsn insn_cta_y = { .op = BW_IR_S2R, .dst_vreg = v_ctaid_y, .imm = BW_SR_CTAID_Y };
    omega_bw_ir_append(prog, &insn_cta_y);

    /* 5. Set up constant 1 */
    BlackwellIRInsn insn_c1 = { .op = BW_IR_MOV_IMM, .dst_vreg = v_const_1, .imm = 1 };
    omega_bw_ir_append(prog, &insn_c1);

    /* 6. Extract warp-level tile and thread coordinates */
    /* group_id = tid >> 2 */
    BlackwellIRInsn insn_grp = { .op = BW_IR_SHF_R, .dst_vreg = v_group_id, .src1_vreg = v_tid, .imm = 2 };
    omega_bw_ir_append(prog, &insn_grp);

    /* tid_in_grp = tid & 3 */
    BlackwellIRInsn insn_tig = { .op = BW_IR_LOP3_AND, .dst_vreg = v_tid_in_grp, .src1_vreg = v_tid, .imm = 3 };
    omega_bw_ir_append(prog, &insn_tig);

    /* row_base = ctaid_y * 16 */
    BlackwellIRInsn insn_rbase = { .op = BW_IR_IMAD, .dst_vreg = v_row_base, .src1_vreg = v_ctaid_y, .src2_vreg = -1, .imm = 16, .src3_vreg = -1 };
    omega_bw_ir_append(prog, &insn_rbase);

    /* col_base = ctaid_x * 8 */
    BlackwellIRInsn insn_cbase = { .op = BW_IR_IMAD, .dst_vreg = v_col_base, .src1_vreg = v_ctaid_x, .src2_vreg = -1, .imm = 8, .src3_vreg = -1 };
    omega_bw_ir_append(prog, &insn_cbase);

    /* r0 = row_base + group_id */
    BlackwellIRInsn insn_r0 = { .op = BW_IR_IADD3, .dst_vreg = v_r0, .src1_vreg = v_row_base, .src2_vreg = v_group_id, .src3_vreg = -1 };
    omega_bw_ir_append(prog, &insn_r0);

    /* r1 = r0 + 8 */
    BlackwellIRInsn insn_r1 = { .op = BW_IR_IMAD, .dst_vreg = v_r1, .src1_vreg = v_const_1, .src2_vreg = -1, .imm = 8, .src3_vreg = v_r0 };
    omega_bw_ir_append(prog, &insn_r1);

    /* col_b = col_base + group_id */
    BlackwellIRInsn insn_colb = { .op = BW_IR_IADD3, .dst_vreg = v_col_b, .src1_vreg = v_col_base, .src2_vreg = v_group_id, .src3_vreg = -1 };
    omega_bw_ir_append(prog, &insn_colb);

    /* tid_col2 = tid_in_grp * 2 */
    BlackwellIRInsn insn_tcol2 = { .op = BW_IR_IMAD, .dst_vreg = v_tid_col2, .src1_vreg = v_tid_in_grp, .src2_vreg = -1, .imm = 2, .src3_vreg = -1 };
    omega_bw_ir_append(prog, &insn_tcol2);

    /* 7. Load A fragments (16x16 row-major, 2 bytes/element, row stride 32 bytes) */
    /* Pair 0: a0, a1 at r0 * 32 + tid_col2 * 2 */
    BlackwellIRInsn insn_a0_idx1 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r0, .src2_vreg = -1, .imm = 32, .src3_vreg = -1 };
    omega_bw_ir_append(prog, &insn_a0_idx1);
    BlackwellIRInsn insn_a0_idx2 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_tid_col2, .src2_vreg = -1, .imm = 2, .src3_vreg = v_idx };
    omega_bw_ir_append(prog, &insn_a0_idx2);
    BlackwellIRInsn insn_a0_off  = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 1, .src3_vreg = v_ptr_a };
    omega_bw_ir_append(prog, &insn_a0_off);
    BlackwellIRInsn insn_lda0    = { .op = BW_IR_LDG_E, .dst_vreg = v_ra, .dst_subreg = 0, .src1_vreg = v_off, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = 0x001ea800 };
    omega_bw_ir_append(prog, &insn_lda0);

    /* Pair 1: a2, a3 at r1 * 32 + tid_col2 * 2 */
    BlackwellIRInsn insn_a1_idx1 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r1, .src2_vreg = -1, .imm = 32, .src3_vreg = -1 };
    omega_bw_ir_append(prog, &insn_a1_idx1);
    BlackwellIRInsn insn_a1_idx2 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_tid_col2, .src2_vreg = -1, .imm = 2, .src3_vreg = v_idx };
    omega_bw_ir_append(prog, &insn_a1_idx2);
    BlackwellIRInsn insn_a1_off  = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 1, .src3_vreg = v_ptr_a };
    omega_bw_ir_append(prog, &insn_a1_off);
    BlackwellIRInsn insn_lda1    = { .op = BW_IR_LDG_E, .dst_vreg = v_ra, .dst_subreg = 1, .src1_vreg = v_off, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = 0x001ea800 };
    omega_bw_ir_append(prog, &insn_lda1);

    /* Pair 2: a4, a5 at r0 * 32 + tid_col2 * 2 + 16 */
    BlackwellIRInsn insn_a2_idx1 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r0, .src2_vreg = -1, .imm = 32, .src3_vreg = -1 };
    omega_bw_ir_append(prog, &insn_a2_idx1);
    BlackwellIRInsn insn_a2_idx2 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_tid_col2, .src2_vreg = -1, .imm = 2, .src3_vreg = v_idx };
    omega_bw_ir_append(prog, &insn_a2_idx2);
    BlackwellIRInsn insn_a2_idx3 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_const_1, .src2_vreg = -1, .imm = 16, .src3_vreg = v_idx };
    omega_bw_ir_append(prog, &insn_a2_idx3);
    BlackwellIRInsn insn_a2_off  = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 1, .src3_vreg = v_ptr_a };
    omega_bw_ir_append(prog, &insn_a2_off);
    BlackwellIRInsn insn_lda2    = { .op = BW_IR_LDG_E, .dst_vreg = v_ra, .dst_subreg = 2, .src1_vreg = v_off, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = 0x001ea800 };
    omega_bw_ir_append(prog, &insn_lda2);

    /* Pair 3: a6, a7 at r1 * 32 + tid_col2 * 2 + 16 */
    BlackwellIRInsn insn_a3_idx1 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r1, .src2_vreg = -1, .imm = 32, .src3_vreg = -1 };
    omega_bw_ir_append(prog, &insn_a3_idx1);
    BlackwellIRInsn insn_a3_idx2 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_tid_col2, .src2_vreg = -1, .imm = 2, .src3_vreg = v_idx };
    omega_bw_ir_append(prog, &insn_a3_idx2);
    BlackwellIRInsn insn_a3_idx3 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_const_1, .src2_vreg = -1, .imm = 16, .src3_vreg = v_idx };
    omega_bw_ir_append(prog, &insn_a3_idx3);
    BlackwellIRInsn insn_a3_off  = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 1, .src3_vreg = v_ptr_a };
    omega_bw_ir_append(prog, &insn_a3_off);
    BlackwellIRInsn insn_lda3    = { .op = BW_IR_LDG_E, .dst_vreg = v_ra, .dst_subreg = 3, .src1_vreg = v_off, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = 0x001ea800 };
    omega_bw_ir_append(prog, &insn_lda3);

    /* 8. Load and pack B fragments (16xN row-major, 2 bytes/element) */
    /* b0 at tid_col2 * N + col_b */
    BlackwellIRInsn insn_b0_idx = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_tid_col2, .src2_vreg = -1, .imm = spec->n, .src3_vreg = v_col_b };
    omega_bw_ir_append(prog, &insn_b0_idx);
    BlackwellIRInsn insn_b0_off = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 2, .src3_vreg = v_ptr_b };
    omega_bw_ir_append(prog, &insn_b0_off);
    BlackwellIRInsn insn_ldb0   = { .op = BW_IR_LDG_E_U16, .dst_vreg = v_b_tmp0, .src1_vreg = v_off, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = 0x002ea800 };
    omega_bw_ir_append(prog, &insn_ldb0);

    /* b1 at (tid_col2 + 1) * N + col_b = b0_idx + N */
    BlackwellIRInsn insn_b1_idx = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_const_1, .src2_vreg = -1, .imm = spec->n, .src3_vreg = v_idx };
    omega_bw_ir_append(prog, &insn_b1_idx);
    BlackwellIRInsn insn_b1_off = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 2, .src3_vreg = v_ptr_b };
    omega_bw_ir_append(prog, &insn_b1_off);
    BlackwellIRInsn insn_ldb1   = { .op = BW_IR_LDG_E_U16, .dst_vreg = v_b_tmp1, .src1_vreg = v_off, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = 0x002ea800 };
    omega_bw_ir_append(prog, &insn_ldb1);

    /* Pack Rb[0] = (b1 << 16) | b0 (wait on barrier 2) */
    BlackwellIRInsn insn_pack_b0 = { .op = BW_IR_IMAD, .dst_vreg = v_rb, .dst_subreg = 0, .src1_vreg = v_b_tmp1, .src2_vreg = -1, .imm = 0x10000, .src3_vreg = v_b_tmp0, .control = 0x004fca00 };
    omega_bw_ir_append(prog, &insn_pack_b0);

    /* b2 at (tid_col2 + 8) * N + col_b */
    BlackwellIRInsn insn_b2_idx1 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_tid_col2, .src2_vreg = -1, .imm = spec->n, .src3_vreg = v_col_b };
    omega_bw_ir_append(prog, &insn_b2_idx1);
    BlackwellIRInsn insn_b2_idx2 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_const_1, .src2_vreg = -1, .imm = 8 * spec->n, .src3_vreg = v_idx };
    omega_bw_ir_append(prog, &insn_b2_idx2);
    BlackwellIRInsn insn_b2_off  = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 2, .src3_vreg = v_ptr_b };
    omega_bw_ir_append(prog, &insn_b2_off);
    BlackwellIRInsn insn_ldb2    = { .op = BW_IR_LDG_E_U16, .dst_vreg = v_b_tmp0, .src1_vreg = v_off, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = 0x002ea800 };
    omega_bw_ir_append(prog, &insn_ldb2);

    /* b3 at (tid_col2 + 9) * N + col_b = b2_idx + N */
    BlackwellIRInsn insn_b3_idx = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_const_1, .src2_vreg = -1, .imm = spec->n, .src3_vreg = v_idx };
    omega_bw_ir_append(prog, &insn_b3_idx);
    BlackwellIRInsn insn_b3_off = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 2, .src3_vreg = v_ptr_b };
    omega_bw_ir_append(prog, &insn_b3_off);
    BlackwellIRInsn insn_ldb3   = { .op = BW_IR_LDG_E_U16, .dst_vreg = v_b_tmp1, .src1_vreg = v_off, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = 0x002ea800 };
    omega_bw_ir_append(prog, &insn_ldb3);

    /* Pack Rb[1] = (b3 << 16) | b2 (wait on barrier 2) */
    BlackwellIRInsn insn_pack_b1 = { .op = BW_IR_IMAD, .dst_vreg = v_rb, .dst_subreg = 1, .src1_vreg = v_b_tmp1, .src2_vreg = -1, .imm = 0x10000, .src3_vreg = v_b_tmp0, .control = 0x004fca00 };
    omega_bw_ir_append(prog, &insn_pack_b1);

    /* 9. Execute HMMA Tensor Core instruction (wait on barrier 1 for A loads) */
    BlackwellIROpcode mma_op = (spec->precision == OMEGA_MATMUL_PRECISION_BF16) ? BW_IR_HMMA_BF16 : BW_IR_HMMA_F16;
    BlackwellIRInsn insn_hmma = {
        .op = mma_op,
        .dst_vreg = v_rd,
        .dst_subreg = 0,
        .src1_vreg = v_ra,
        .src1_subreg = 0,
        .src2_vreg = v_rb,
        .src2_subreg = 0,
        .src3_vreg = -1, /* RZ accumulator input */
        .control = 0x002fca00 /* Wait on barrier 1 */
    };
    omega_bw_ir_append(prog, &insn_hmma);

    /* 10. Store C output fragments (16xN row-major, 4 bytes/float element) */
    /* d0: row r0, col col_base + tid_col2 + 0 */
    BlackwellIRInsn insn_d0_idx1 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r0, .src2_vreg = -1, .imm = spec->n, .src3_vreg = v_col_base };
    omega_bw_ir_append(prog, &insn_d0_idx1);
    BlackwellIRInsn insn_d0_idx2 = { .op = BW_IR_IADD3, .dst_vreg = v_idx, .src1_vreg = v_idx, .src2_vreg = v_tid_col2, .src3_vreg = -1 };
    omega_bw_ir_append(prog, &insn_d0_idx2);
    BlackwellIRInsn insn_d0_off  = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 4, .src3_vreg = v_ptr_c };
    omega_bw_ir_append(prog, &insn_d0_off);
    BlackwellIRInsn insn_std0    = { .op = BW_IR_STG_E, .src1_vreg = v_off, .src2_vreg = v_rd, .src2_subreg = 0, .ureg = uv_desc, .control = 0x000fe200 };
    omega_bw_ir_append(prog, &insn_std0);

    /* d1: row r0, col col_base + tid_col2 + 1 */
    BlackwellIRInsn insn_d1_idx  = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_const_1, .src2_vreg = -1, .imm = 1, .src3_vreg = v_idx };
    omega_bw_ir_append(prog, &insn_d1_idx);
    BlackwellIRInsn insn_d1_off  = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 4, .src3_vreg = v_ptr_c };
    omega_bw_ir_append(prog, &insn_d1_off);
    BlackwellIRInsn insn_std1    = { .op = BW_IR_STG_E, .src1_vreg = v_off, .src2_vreg = v_rd, .src2_subreg = 1, .ureg = uv_desc, .control = 0x000fe200 };
    omega_bw_ir_append(prog, &insn_std1);

    /* d2: row r1, col col_base + tid_col2 + 0 */
    BlackwellIRInsn insn_d2_idx1 = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r1, .src2_vreg = -1, .imm = spec->n, .src3_vreg = v_col_base };
    omega_bw_ir_append(prog, &insn_d2_idx1);
    BlackwellIRInsn insn_d2_idx2 = { .op = BW_IR_IADD3, .dst_vreg = v_idx, .src1_vreg = v_idx, .src2_vreg = v_tid_col2, .src3_vreg = -1 };
    omega_bw_ir_append(prog, &insn_d2_idx2);
    BlackwellIRInsn insn_d2_off  = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 4, .src3_vreg = v_ptr_c };
    omega_bw_ir_append(prog, &insn_d2_off);
    BlackwellIRInsn insn_std2    = { .op = BW_IR_STG_E, .src1_vreg = v_off, .src2_vreg = v_rd, .src2_subreg = 2, .ureg = uv_desc, .control = 0x000fe200 };
    omega_bw_ir_append(prog, &insn_std2);

    /* d3: row r1, col col_base + tid_col2 + 1 */
    BlackwellIRInsn insn_d3_idx  = { .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_const_1, .src2_vreg = -1, .imm = 1, .src3_vreg = v_idx };
    omega_bw_ir_append(prog, &insn_d3_idx);
    BlackwellIRInsn insn_d3_off  = { .op = BW_IR_IMAD_WIDE, .dst_vreg = v_off, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 4, .src3_vreg = v_ptr_c };
    omega_bw_ir_append(prog, &insn_d3_off);
    BlackwellIRInsn insn_std3    = { .op = BW_IR_STG_E, .src1_vreg = v_off, .src2_vreg = v_rd, .src2_subreg = 3, .ureg = uv_desc, .control = 0x000fe200 };
    omega_bw_ir_append(prog, &insn_std3);

    /* 11. Program Exit */
    BlackwellIRInsn insn_exit = { .op = BW_IR_EXIT };
    omega_bw_ir_append(prog, &insn_exit);
    BlackwellIRInsn insn_bra  = { .op = BW_IR_BRA };
    omega_bw_ir_append(prog, &insn_bra);

    /* 12. Run bounded register allocation */
    if (omega_bw_regalloc_solve(prog) != 0) {
        return -1;
    }
    return 0;
}

int omega_blackwell_codegen_matmul_tensor(const OmegaMatMulSpec *spec, OmegaBlackwellKernel *kernel) {
    if (!spec || !kernel) return -1;
    BlackwellIRProgram prog;
    if (omega_blackwell_codegen_matmul_tensor_prog(spec, &prog) != 0) {
        return -1;
    }

    size_t max_bytes = 4096;
    kernel->code = malloc(max_bytes);
    if (!kernel->code) return -1;

    size_t emitted_size = 0;
    if (omega_bw_encode_program(&prog, kernel->code, max_bytes, &emitted_size) != 0) {
        free(kernel->code);
        kernel->code = NULL;
        return -1;
    }

    kernel->code_size = emitted_size;
    kernel->insn_count = emitted_size / 16;
    kernel->gpr_count = prog.regalloc.peak_gpr_usage;
    kernel->uniform_gpr_count = prog.regalloc.peak_ugpr_usage;

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

    /* 11. HMMA.16816.F32 R4, R4, R2, RZ */
    BlackwellIRInsn insn_hmma_f16 = { .op = BW_IR_HMMA_F16, .dst_vreg = 1, .src1_vreg = 1, .src2_vreg = 0, .src3_vreg = -1 };
    if (encode_single_insn(&insn_hmma_f16, &ra, w) != 0) return -21;
    if (w[0] != 0x0404723c || w[1] != 0x00000002 || w[2] != 0x000018ff || w[3] != 0x000fe200) return -22;

    /* 12. HMMA.16816.F32.BF16 R4, R4, R2, RZ */
    BlackwellIRInsn insn_hmma_bf16 = { .op = BW_IR_HMMA_BF16, .dst_vreg = 1, .src1_vreg = 1, .src2_vreg = 0, .src3_vreg = -1 };
    uint32_t w_bf16[4];
    if (encode_single_insn(&insn_hmma_bf16, &ra, w_bf16) != 0) return -23;
    if (w_bf16[0] != 0x0404723c || w_bf16[1] != 0x00000002 || w_bf16[2] != 0x000418ff || w_bf16[3] != 0x000fe200) return -24;

    /* 13. Differential Invariant: w[2] bit 18 toggles strictly between FP16 and BF16 */
    if ((w[2] ^ w_bf16[2]) != 0x00040000) return -25;
    if (w[0] != w_bf16[0] || w[1] != w_bf16[1] || w[3] != w_bf16[3]) return -26;

    /* 14. SHF.R.U32.HI R4, RZ, 2, R2 */
    BlackwellIRInsn insn_shf = { .op = BW_IR_SHF_R, .dst_vreg = 1, .src1_vreg = 0, .imm = 2 };
    if (encode_single_insn(&insn_shf, &ra, w) != 0) return -27;
    if (w[0] != 0xff047819 || w[1] != 2 || w[2] != 0x00011602 || w[3] != 0x001fca00) return -28;

    /* 15. LOP3.LUT R4, R2, 3, RZ, 0xc0, !PT */
    BlackwellIRInsn insn_lop3 = { .op = BW_IR_LOP3_AND, .dst_vreg = 1, .src1_vreg = 0, .imm = 3 };
    if (encode_single_insn(&insn_lop3, &ra, w) != 0) return -29;
    if (w[0] != 0x02047812 || w[1] != 3 || w[2] != 0x078ec0ff || w[3] != 0x001fca00) return -30;

    /* 16. LDG.E.U16 R2, desc[UR4][R4.64] */
    BlackwellIRInsn insn_ldg16 = { .op = BW_IR_LDG_E_U16, .dst_vreg = 0, .src1_vreg = 1, .ureg = 0 };
    if (encode_single_insn(&insn_ldg16, &ra, w) != 0) return -31;
    if (w[0] != 0x04027981 || w[1] != 4 || w[2] != 0x0c1e1500 || w[3] != 0x000f2200) return -32;

    /* A zero-delta branch stays the self-branch the matmul kernels already emit. */
    BlackwellIRInsn insn_bra0 = { .op = BW_IR_BRA };
    if (encode_single_insn(&insn_bra0, &ra, w) != 0) return -33;
    if (w[0] != 0x00fc7947 || w[1] != 0xfffffffc || w[2] != 0x0383ffff || w[3] != 0x000fc000) return -34;

    BlackwellIRInsn insn_ge = { .op = BW_IR_ISETP_GE_U32, .src1_vreg = 1, .src2_vreg = 2 };
    if (encode_single_insn(&insn_ge, &ra, w) != 0) return -35;
    if (w[0] != (0x720c | (4 << 24)) || w[1] != 6 || w[2] != 0x03f06070) return -36;

    BlackwellIRInsn insn_xor = { .op = BW_IR_LOP3_XOR, .dst_vreg = 1, .src1_vreg = 0, .src2_vreg = 2 };
    if (encode_single_insn(&insn_xor, &ra, w) != 0) return -37;
    if (w[0] != (0x7212 | (4 << 16) | (2 << 24)) || w[1] != 6 || w[2] != 0x078e3cff) return -38;

    BlackwellIRInsn insn_ldg_ss = { .op = BW_IR_LDG_STRONG_SYS, .dst_vreg = 0, .src1_vreg = 1, .ureg = 0 };
    if (encode_single_insn(&insn_ldg_ss, &ra, w) != 0) return -39;
    if (w[0] != (0x7981 | (2 << 16) | (4 << 24)) || w[1] != 4 || w[2] != 0x0c1f5900) return -40;
    /* 17. FADD R7, R2, R4 */
    BlackwellIRInsn insn_fadd = { .op = BW_IR_FADD, .dst_vreg = 3, .src1_vreg = 0, .src2_vreg = 1 };
    if (encode_single_insn(&insn_fadd, &ra, w) != 0) return -133;
    if (w[0] != 0x02077221 || w[1] != 4 || w[2] != 0 || w[3] != 0x004fc400) return -134;

    /* 18. FSUB R7, R2, R4 (negated src2) */
    BlackwellIRInsn insn_fsub = { .op = BW_IR_FSUB, .dst_vreg = 3, .src1_vreg = 0, .src2_vreg = 1 };
    if (encode_single_insn(&insn_fsub, &ra, w) != 0) return -135;
    if (w[0] != 0x02077221 || w[1] != 0x80000004 || w[2] != 0 || w[3] != 0x004fc400) return -136;

    /* 19. FMUL R7, R2, R4 */
    BlackwellIRInsn insn_fmul = { .op = BW_IR_FMUL, .dst_vreg = 3, .src1_vreg = 0, .src2_vreg = 1 };
    if (encode_single_insn(&insn_fmul, &ra, w) != 0) return -137;
    if (w[0] != 0x02077220 || w[1] != 4 || w[2] != 0x00400000 || w[3] != 0x004fc400) return -138;

    /* 20. FFMA R7, R2, R4, R6 */
    BlackwellIRInsn insn_ffma = { .op = BW_IR_FFMA, .dst_vreg = 3, .src1_vreg = 0, .src2_vreg = 1, .src3_vreg = 2 };
    if (encode_single_insn(&insn_ffma, &ra, w) != 0) return -139;
    if (w[0] != 0x02077223 || w[1] != 4 || w[2] != 6 || w[3] != 0x004fc400) return -140;

    /* 21. FMNMX_MIN R7, R2, R4 */
    BlackwellIRInsn insn_min = { .op = BW_IR_FMNMX_MIN, .dst_vreg = 3, .src1_vreg = 0, .src2_vreg = 1 };
    if (encode_single_insn(&insn_min, &ra, w) != 0) return -141;
    if (w[0] != 0x02077209 || w[1] != 4 || w[2] != 0x03800000 || w[3] != 0x004fc400) return -142;

    /* 22. FMNMX_MAX R7, R2, R4 */
    BlackwellIRInsn insn_max = { .op = BW_IR_FMNMX_MAX, .dst_vreg = 3, .src1_vreg = 0, .src2_vreg = 1 };
    if (encode_single_insn(&insn_max, &ra, w) != 0) return -143;
    if (w[0] != 0x02077209 || w[1] != 4 || w[2] != 0x07800000 || w[3] != 0x004fc400) return -144;

    /* 23. I2FP R7, R2 */
    BlackwellIRInsn insn_i2f = { .op = BW_IR_I2FP, .dst_vreg = 3, .src1_vreg = 0 };
    if (encode_single_insn(&insn_i2f, &ra, w) != 0) return -145;
    if (w[0] != 0x00077245 || w[1] != 2 || w[2] != 0x00201400 || w[3] != 0x004fe200) return -146;

    /* 24. F2I R7, R2 */
    BlackwellIRInsn insn_f2i = { .op = BW_IR_F2I, .dst_vreg = 3, .src1_vreg = 0 };
    if (encode_single_insn(&insn_f2i, &ra, w) != 0) return -147;
    if (w[0] != 0x00077305 || w[1] != 2 || w[2] != 0x0020f100 || w[3] != 0x004e2200) return -148;

    /* 25. MUFU_RCP R7, R2 */
    BlackwellIRInsn insn_mufu_rcp = { .op = BW_IR_MUFU_RCP, .dst_vreg = 3, .src1_vreg = 0 };
    if (encode_single_insn(&insn_mufu_rcp, &ra, w) != 0) return -149;
    if (w[0] != 0x00077308 || w[1] != 2 || w[2] != 0x00001000 || w[3] != 0x000e2400) return -150;

    /* 26. MUFU_RSQ R7, R2 */
    BlackwellIRInsn insn_mufu_rsq = { .op = BW_IR_MUFU_RSQ, .dst_vreg = 3, .src1_vreg = 0 };
    if (encode_single_insn(&insn_mufu_rsq, &ra, w) != 0) return -151;
    if (w[0] != 0x00077308 || w[1] != 2 || w[2] != 0x00001400 || w[3] != 0x000e2400) return -152;

    /* 27. SHFL_DOWN R7, R2, 16 */
    BlackwellIRInsn insn_shfl = { .op = BW_IR_SHFL_DOWN, .dst_vreg = 3, .src1_vreg = 0, .imm = 16 };
    if (encode_single_insn(&insn_shfl, &ra, w) != 0) return -153;
    if (w[0] != 0x02077f89 || w[1] != 0x0a001f00 || w[2] != 0x000e0000 || w[3] != 0x000e2400) return -154;

    /* 28. LDS R7, [R2] */
    BlackwellIRInsn insn_lds = { .op = BW_IR_LDS, .dst_vreg = 3, .src1_vreg = 0, .imm = 0 };
    if (encode_single_insn(&insn_lds, &ra, w) != 0) return -155;
    if (w[0] != 0x02077984 || w[1] != 0 || w[2] != 0 || w[3] != 0x000fe200) return -156;

    /* 29. STS [R2], R4 */
    BlackwellIRInsn insn_sts = { .op = BW_IR_STS, .src1_vreg = 0, .src2_vreg = 1, .imm = 0 };
    if (encode_single_insn(&insn_sts, &ra, w) != 0) return -157;
    if (w[0] != 0x02007388 || w[1] != 4 || w[2] != 0 || w[3] != 0x000fe200) return -158;

    return 0;
}

int omega_blackwell_test_regalloc_bounds(void) {
    BlackwellIRProgram prog;
    omega_bw_ir_init(&prog);

    int v0 = omega_bw_ir_alloc_vreg64(&prog);
    int v1 = omega_bw_ir_alloc_vreg(&prog);
    int v2 = omega_bw_ir_alloc_vreg64(&prog);
    int v3 = omega_bw_ir_alloc_vreg128(&prog); /* Test 128-bit quad allocation */

    BlackwellIRInsn i0 = { .op = BW_IR_LDC64, .dst_vreg = v0, .imm = 0x380 };
    omega_bw_ir_append(&prog, &i0);
    BlackwellIRInsn i1 = { .op = BW_IR_MOV_IMM, .dst_vreg = v1, .imm = 10 };
    omega_bw_ir_append(&prog, &i1);
    BlackwellIRInsn i2 = { .op = BW_IR_LDC64, .dst_vreg = v2, .imm = 0x388 };
    omega_bw_ir_append(&prog, &i2);
    BlackwellIRInsn i3 = { .op = BW_IR_HMMA_F16, .dst_vreg = v3, .src1_vreg = v3, .src2_vreg = v0, .src3_vreg = -1 };
    omega_bw_ir_append(&prog, &i3);
    /* i4 uses v1, v2, v3 simultaneously so all intervals are concurrently active */
    int v_sink = omega_bw_ir_alloc_vreg(&prog);
    BlackwellIRInsn i4 = { .op = BW_IR_IADD3, .dst_vreg = v_sink, .src1_vreg = v1, .src2_vreg = v2, .src3_vreg = v3 };
    omega_bw_ir_append(&prog, &i4);

    if (omega_bw_regalloc_solve(&prog) != 0) return -1;

    int phys_v0 = prog.regalloc.vreg_to_phys[v0];
    int phys_v1 = prog.regalloc.vreg_to_phys[v1];
    int phys_v2 = prog.regalloc.vreg_to_phys[v2];
    int phys_v3 = prog.regalloc.vreg_to_phys[v3];

    if (phys_v0 % 2 != 0) return -2;
    if (phys_v2 % 2 != 0) return -3;
    if (phys_v3 % 4 != 0) return -4; /* 4-alignment for 128-bit quad */
    if (phys_v1 == phys_v0 || phys_v1 == phys_v0 + 1) return -5;
    if (phys_v2 == phys_v0 || phys_v2 == phys_v0 + 1) return -6;
    for (int k = 0; k < 4; k++) {
        if (phys_v3 + k == phys_v0 || phys_v3 + k == phys_v0 + 1) return -7;
        if (phys_v3 + k == phys_v2 || phys_v3 + k == phys_v2 + 1) return -8;
        if (phys_v3 + k == phys_v1) return -9;
    }

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
