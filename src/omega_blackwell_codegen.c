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
            w[3] = insn->control ? insn->control : 0x000fc000; /* control: a timed pause (fragment matmul) */
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

        case BW_IR_LDG_E_64:
        case BW_IR_LDG_E_128: {
            /* LDG.E.64 / LDG.E.128 Rd, desc[URd][Ra.64+imm]. imm = signed 24-bit byte offset at
             * bits 40..63 (w[1] bits 8..31); size field in w[2] bits 9..11 (32 = 4, 64 = 5, 128 = 6).
             * Words from nvcc 13.0.88 -arch=sm_121 (omega_blackwell_verify_codegen_fixtures_fragmm). */
            int32_t off = (int32_t)insn->imm;
            if (off < -(1 << 23) || off >= (1 << 23)) return -1;
            w[0] = 0x7981U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(ureg & 0xff) | (((uint32_t)off & 0xffffffU) << 8);
            w[2] = insn->op == BW_IR_LDG_E_64 ? 0x0c1e1b00U : 0x0c1e1d00U;
            w[3] = insn->control ? insn->control : 0x000f2200;
            break;
        }

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
            break;

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

        case BW_IR_BSSY: {
            /* BSSY.RECONVERGENT Bn, target (omega #308). nvcc 13.0.88 -arch=sm_121:
             *   [0070] BSSY.RECONVERGENT B0, 0x140 ; 0x000000c000007945 0x000fe20003800200
             * opcode 0x945 (NAK sm70_encode.rs:4165), barrier id at bits 16..19 (set_bar_dst
             * 16..20), guard predicate at bits 12..15 like BRA, relative byte offset
             * target - (this + 16) at bit 32 (NAK: field 34.. in 4-byte units), the cond
             * predicate PT at bits 87..89 (0x0380 << 16 in w[2]) and bit 73 (0x200 in w[2]),
             * which the disassembler prints as .RECONVERGENT; nvcc sets it on every BSSY/BSYNC
             * for sm_121 and Mesa NAK does not. We follow nvcc. The target is the instruction
             * after the matching BSYNC, so the delta is at least 2; smaller is refused. */
            int32_t delta = (int32_t)insn->imm;
            if (delta < 2 || insn->bar_reg >= BW_RECONV_MAX_BAR) return -1;
            uint32_t pred_field = 0x7u;
            if (insn->predicate_p0)
                pred_field = insn->predicate_not ? 0x8u : 0x0u;
            w[0] = 0x945u | (pred_field << 12) | ((uint32_t)insn->bar_reg << 16);
            w[1] = (uint32_t)(delta - 1) * 16u;
            w[2] = 0x03800200u;
            w[3] = insn->control ? insn->control : 0x000fe200u;
            break;
        }

        case BW_IR_BSYNC: {
            /* BSYNC.RECONVERGENT Bn: [0130] BSYNC.RECONVERGENT B0 ; 0x0000000000007941 0x000fea0003800200
             * opcode 0x941 (NAK sm70_encode.rs:4179), barrier id bits 16..19, cond PT, bit 73. */
            if (insn->bar_reg >= BW_RECONV_MAX_BAR) return -1;
            uint32_t pred_field = 0x7u;
            if (insn->predicate_p0)
                pred_field = insn->predicate_not ? 0x8u : 0x0u;
            w[0] = 0x941u | (pred_field << 12) | ((uint32_t)insn->bar_reg << 16);
            w[1] = 0;
            w[2] = 0x03800200u;
            w[3] = insn->control ? insn->control : 0x000fea00u;
            break;
        }

        case BW_IR_MUFU_EX2:
            /* MUFU.EX2 Rd, Ra. nvdisasm 13.0 -b SM121: 0x7308 with sub-op 0x0800 decodes
             * "MUFU.EX2 R2, R4" (RCP is 0x1000, RSQ 0x1400 above). */
            w[0] = 0x7308U | ((uint32_t)(dst & 0xff) << 16);
            w[1] = (uint32_t)(src1 & 0xff);
            w[2] = 0x00000800;
            w[3] = insn->control ? insn->control : 0x000e2400;
            break;

        case BW_IR_BAR_SYNC:
            /* BAR.SYNC.DEFER_BLOCKING 0x0: the exact words src/omega_numeric.c:885 runs on
             * the chip (nvdisasm: "BAR.SYNC.DEFER_BLOCKING 0x0"). No registers. */
            w[0] = 0x00007b1dU;
            w[1] = 0x00000000;
            w[2] = 0x00010000;
            w[3] = insn->control ? insn->control : 0x000fec00;
            break;

        case BW_IR_LDS32:
            /* LDS Rd, [Ra+URZ] (32-bit), the src/omega_numeric.c:886 form. */
            w[0] = 0x7984U | ((uint32_t)(dst & 0xff) << 16) | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = 0x000000ff;
            w[2] = 0x08000800;
            w[3] = insn->control ? insn->control : 0x000e2800;
            break;

        case BW_IR_STS32:
            /* STS [Ra+URZ], Rb (32-bit), the src/omega_numeric.c:884 form. */
            w[0] = 0x7988U | ((uint32_t)(src1 & 0xff) << 24);
            w[1] = (uint32_t)(src2 & 0xff);
            w[2] = 0x080008ffU;
            w[3] = insn->control ? insn->control : 0x000fe200;
            break;

        /* Prime race cut (2026-10-05). An absent operand is RZ for these three ops. */
        case BW_IR_LOP3_LUT: {
            /* LOP3.LUT Rd, Ra, Rb, Rc, imm8, !PT. LOP3_XOR above is this form with Rc = RZ, LUT 0x3c. */
            uint32_t a = (insn->src1_vreg >= 0) ? (uint32_t)(src1 & 0xff) : 0xffU;
            uint32_t b = (insn->src2_vreg >= 0) ? (uint32_t)(src2 & 0xff) : 0xffU;
            w[0] = 0x7212U | ((uint32_t)(dst & 0xff) << 16) | (a << 24);
            w[1] = b;
            w[2] = 0x078e0000U | ((insn->imm & 0xffU) << 8) | (uint32_t)(src3 & 0xff);
            w[3] = insn->control ? insn->control : 0x001fca00;
            break;
        }

        case BW_IR_SHF_L_U32: {
            /* SHF.L.U32 Rd, Ra, Rb, Rc: register-amount form (0x7219) of SHF_R's 0x7819, with the
             * .R (w[2] bit 12) and .HI (bit 16) flags clear. Rc is the funnel high word (RZ). */
            uint32_t a = (insn->src1_vreg >= 0) ? (uint32_t)(src1 & 0xff) : 0xffU;
            uint32_t b = (insn->src2_vreg >= 0) ? (uint32_t)(src2 & 0xff) : 0xffU;
            w[0] = 0x7219U | ((uint32_t)(dst & 0xff) << 16) | (a << 24);
            w[1] = b;
            w[2] = 0x00000600U | (uint32_t)(src3 & 0xff);
            w[3] = insn->control ? insn->control : 0x001fca00;
            break;
        }

        case BW_IR_IMAD_HI_U32: {
            /* IMAD.HI.U32 Rd, Ra, Rb, Rc: opcode 0x227 with the IMAD.WIDE.U32 modifier word. */
            uint32_t a = (insn->src1_vreg >= 0) ? (uint32_t)(src1 & 0xff) : 0xffU;
            uint32_t b = (insn->src2_vreg >= 0) ? (uint32_t)(src2 & 0xff) : 0xffU;
            w[0] = 0x7227U | ((uint32_t)(dst & 0xff) << 16) | (a << 24);
            w[1] = b;
            w[2] = 0x078e0000U | (uint32_t)(src3 & 0xff);
            w[3] = insn->control ? insn->control : 0x001fca00;
            break;
        }

        default:
            return -1;
    }
    return 0;
}

/* Prime race cut (2026-10-05): golden words for LOP3_LUT, SHF_L_U32 and IMAD_HI_U32.
 * Each expected (w0, w1, w2) is the low 96 bits of an instruction nvcc 13.0.88 emitted
 * for sm_121 (nvcc -arch=sm_121 -cubin, cuobjdump -sass; source and dump in
 * bench/prime_race/tests/encoding_oracle.md). Control words (w3) are scheduler output and
 * not compared. nvcc is an offline oracle only; nothing here links CUDA. 0 = all good. */
int omega_blackwell_verify_codegen_fixtures_intops(void) {
    OmegaRegAlloc ra;
    memset(&ra, 0, sizeof(ra));
    static const int phys[] = { 0, 2, 6, 7, 9, 11, 13 };
    for (int i = 0; i < (int)(sizeof phys / sizeof phys[0]); i++) ra.vreg_to_phys[i] = phys[i];
    enum { R0 = 0, R2, R6, R7, R9, R11, R13, RZ = -1 };
    static const struct {
        BlackwellIROpcode op; int d, a, b, c; uint32_t imm; uint32_t w0, w1, w2; const char *text;
    } fx[] = {
        { BW_IR_LOP3_LUT, R9, R7, R0, RZ, 0xfc, 0x07097212U, 0x00000000U, 0x078efcffU, "LOP3.LUT R9, R7, R0, RZ, 0xfc, !PT" },
        { BW_IR_LOP3_LUT, R11, R6, R7, R0, 0x78, 0x060b7212U, 0x00000007U, 0x078e7800U, "LOP3.LUT R11, R6, R7, R0, 0x78, !PT" },
        { BW_IR_LOP3_LUT, R13, R6, R7, R0, 0x10, 0x060d7212U, 0x00000007U, 0x078e1000U, "LOP3.LUT R13, R6, R7, R0, 0x10, !PT" },
        { BW_IR_LOP3_LUT, R7, RZ, R9, RZ, 0x33, 0xff077212U, 0x00000009U, 0x078e33ffU, "LOP3.LUT R7, RZ, R9, RZ, 0x33, !PT" },
        { BW_IR_SHF_L_U32, R7, R9, R2, RZ, 0, 0x09077219U, 0x00000002U, 0x000006ffU, "SHF.L.U32 R7, R9, R2, RZ" },
        { BW_IR_SHF_L_U32, R7, R0, R7, RZ, 0, 0x00077219U, 0x00000007U, 0x000006ffU, "SHF.L.U32 R7, R0, R7, RZ" },
        { BW_IR_IMAD_HI_U32, R9, R9, R0, R6, 0, 0x09097227U, 0x00000000U, 0x078e0006U, "IMAD.HI.U32 R9, R9, R0, R6" },
        { BW_IR_IMAD_HI_U32, R7, R0, R7, RZ, 0, 0x00077227U, 0x00000007U, 0x078e00ffU, "IMAD.HI.U32 R7, R0, R7, RZ" },
    };
    for (int i = 0; i < (int)(sizeof fx / sizeof fx[0]); i++) {
        BlackwellIRInsn in;
        memset(&in, 0, sizeof in);
        in.op = fx[i].op; in.dst_vreg = fx[i].d; in.src1_vreg = fx[i].a; in.src2_vreg = fx[i].b;
        in.src3_vreg = fx[i].c; in.ureg = -1; in.imm = fx[i].imm;
        uint32_t w[4];
        if (encode_single_insn(&in, &ra, w) != 0) return -(2 * i + 1);
        if (w[0] != fx[i].w0 || w[1] != fx[i].w1 || w[2] != fx[i].w2) return -(2 * i + 2);
    }
    return 0;
}

int omega_blackwell_encode_one(const BlackwellIRInsn *insn, uint32_t w[4]) {
    if (!insn || !w) return -1;
    OmegaRegAlloc ra;
    memset(&ra, 0, sizeof ra);
    for (int i = 0; i < BW_MAX_VREGS; i++) ra.vreg_to_phys[i] = i;
    for (int i = 0; i < BW_MAX_UVREGS; i++) ra.uvreg_to_phys[i] = i;
    return encode_single_insn(insn, &ra, w);
}

/* omega #328: golden words for the fragment-major matmul. Oracle: nvcc 13.0.88 -arch=sm_121
 * -cubin -O3, cuobjdump -sass, on a kernel with uint2 / uint4 loads at immediate offsets and a
 * dependent mma.m16n8k16 bf16 chain (source and dump sealed in
 * ~/workspace/investigations/2026-10-07-qwen3-decode-matmul/oracle/frag_plain.{cu,sass}):
 *   LDG.E.64 R2, desc[UR8][R30.64] ;            0x000000081e027981 0x000ea8000c1e1b00
 *   LDG.E.64 R28, desc[UR8][R30.64+0x100] ;     0x000100081e1c7981 0x000ee8000c1e1b00
 *   LDG.E.128 R8, desc[UR8][R32.64] ;           0x0000000820087981 0x000ea8000c1e1d00
 *   LDG.E.128 R20, desc[UR8][R32.64+0x1000] ;   0x0010000820147981 0x000e9e000c1e1d00
 *   HMMA.16816.F32.BF16 R8, R12, R6, R8 ;       0x000000060c08723c 0x010fe20000041808
 *   NOP ;                                       0x0000000000007918 0x000fdc0000000000
 * The dependent HMMA chain nvcc emits is HMMA (stall 15) then NOP (stall 14): 29 cycles from
 * one HMMA to the next that reads its accumulator. w3 is compared for the NOP only (it is the
 * whole point of that instruction). 0 = all good. */
int omega_blackwell_verify_codegen_fixtures_fragmm(void) {
    static const struct {
        BlackwellIROpcode op; int d, a, b, c, u; uint32_t imm, control; uint32_t w0, w1, w2, w3; const char *text;
    } fx[] = {
        { BW_IR_LDG_E_64, 2, 30, -1, -1, 8, 0, 0, 0x1e027981U, 0x00000008U, 0x0c1e1b00U, 0, "LDG.E.64 R2, desc[UR8][R30.64]" },
        { BW_IR_LDG_E_64, 28, 30, -1, -1, 8, 0x100, 0, 0x1e1c7981U, 0x00010008U, 0x0c1e1b00U, 0, "LDG.E.64 R28, desc[UR8][R30.64+0x100]" },
        { BW_IR_LDG_E_128, 8, 32, -1, -1, 8, 0, 0, 0x20087981U, 0x00000008U, 0x0c1e1d00U, 0, "LDG.E.128 R8, desc[UR8][R32.64]" },
        { BW_IR_LDG_E_128, 20, 32, -1, -1, 8, 0x1000, 0, 0x20147981U, 0x00100008U, 0x0c1e1d00U, 0, "LDG.E.128 R20, desc[UR8][R32.64+0x1000]" },
        { BW_IR_HMMA_BF16, 8, 12, 6, 8, -1, 0, 0, 0x0c08723cU, 0x00000006U, 0x00041808U, 0, "HMMA.16816.F32.BF16 R8, R12, R6, R8" },
        { BW_IR_NOP, -1, -1, -1, -1, -1, 0, 0x000fdc00U, 0x00007918U, 0, 0, 0x000fdc00U, "NOP (stall 14)" },
    };
    for (int i = 0; i < (int)(sizeof fx / sizeof fx[0]); i++) {
        BlackwellIRInsn in;
        memset(&in, 0, sizeof in);
        in.op = fx[i].op; in.dst_vreg = fx[i].d; in.src1_vreg = fx[i].a; in.src2_vreg = fx[i].b;
        in.src3_vreg = fx[i].c; in.ureg = fx[i].u; in.imm = fx[i].imm; in.control = fx[i].control;
        uint32_t w[4];
        if (omega_blackwell_encode_one(&in, w) != 0) return -(2 * i + 1);
        if (w[0] != fx[i].w0 || w[1] != fx[i].w1 || w[2] != fx[i].w2 || (fx[i].w3 && w[3] != fx[i].w3)) return -(2 * i + 2);
    }
    /* offsets outside the signed 24-bit field are refused, not wrapped */
    BlackwellIRInsn big;
    memset(&big, 0, sizeof big);
    big.op = BW_IR_LDG_E_64; big.dst_vreg = 2; big.src1_vreg = 30; big.src2_vreg = -1; big.src3_vreg = -1; big.ureg = 8;
    big.imm = 1u << 23;
    uint32_t w[4];
    if (omega_blackwell_encode_one(&big, w) == 0) return -100;
    big.imm = (uint32_t)-(1 << 23);
    if (omega_blackwell_encode_one(&big, w) != 0) return -101;
    return 0;
}

/* omega #308: BSSY / BSYNC golden words. Oracle: nvcc 13.0.88 -arch=sm_121 -cubin -O3 on the
 * kernels in docs/gpu-reconvergence-308.md, cuobjdump -sass (low 64-bit word, then high; the
 * top 23 bits of the high word are the control word and are not compared here):
 *   k_loop   [0070] BSSY.RECONVERGENT B0, 0x140 ; 0x000000c000007945 0x000fe20003800200  (delta 13)
 *   k_nested [0070] BSSY.RECONVERGENT B0, 0x300 ; 0x0000028000007945 0x000fe20003800200  (delta 41)
 *   k_break  [0070] BSSY.RECONVERGENT B0, 0x110 ; 0x0000009000007945 0x000fe20003800200  (delta 10)
 *   k_loop   [0130] BSYNC.RECONVERGENT B0 ;        0x0000000000007941 0x000fea0003800200
 * Barrier ids other than B0 and the predicated forms are decoded by tools/reconv_nvdisasm_check.sh
 * (nvdisasm -b SM121 text), not by this table. */
int omega_blackwell_verify_codegen_fixtures_reconv(void) {
    static const struct { BlackwellIROpcode op; int delta; int bar; uint32_t w0, w1, w2; } fx[] = {
        { BW_IR_BSSY, 13, 0, 0x00007945U, 0x000000c0U, 0x03800200U },
        { BW_IR_BSSY, 41, 0, 0x00007945U, 0x00000280U, 0x03800200U },
        { BW_IR_BSSY, 10, 0, 0x00007945U, 0x00000090U, 0x03800200U },
        { BW_IR_BSYNC, 0, 0, 0x00007941U, 0x00000000U, 0x03800200U },
    };
    for (int i = 0; i < (int)(sizeof fx / sizeof fx[0]); i++) {
        BlackwellIRInsn in;
        memset(&in, 0, sizeof in);
        in.op = fx[i].op; in.dst_vreg = -1; in.src1_vreg = -1; in.src2_vreg = -1; in.src3_vreg = -1; in.ureg = -1;
        in.imm = (uint32_t)fx[i].delta; in.bar_reg = (uint8_t)fx[i].bar;
        uint32_t w[4];
        if (omega_blackwell_encode_one(&in, w) != 0) return -(2 * i + 1);
        if (w[0] != fx[i].w0 || w[1] != fx[i].w1 || w[2] != fx[i].w2) return -(2 * i + 2);
        /* control word defaults mirror nvcc's: BSSY 0x000fe200, BSYNC 0x000fea00 */
        if (w[3] != (fx[i].op == BW_IR_BSSY ? 0x000fe200U : 0x000fea00U)) return -(2 * i + 2);
    }
    /* refusals: a BSSY must name a later instruction (delta >= 2), barrier ids stop at B15 */
    BlackwellIRInsn bad; uint32_t w[4];
    memset(&bad, 0, sizeof bad); bad.op = BW_IR_BSSY; bad.dst_vreg = bad.src1_vreg = bad.src2_vreg = bad.src3_vreg = bad.ureg = -1;
    bad.imm = 1; bad.bar_reg = 0;
    if (omega_blackwell_encode_one(&bad, w) == 0) return -101;
    bad.imm = (uint32_t)-3;
    if (omega_blackwell_encode_one(&bad, w) == 0) return -102;
    bad.imm = 5; bad.bar_reg = 16;
    if (omega_blackwell_encode_one(&bad, w) == 0) return -103;
    bad.op = BW_IR_BSYNC; bad.imm = 0; bad.bar_reg = 16;
    if (omega_blackwell_encode_one(&bad, w) == 0) return -104;
    bad.bar_reg = 15;
    if (omega_blackwell_encode_one(&bad, w) != 0 || w[0] != (0x00007941U | (15U << 16))) return -105;
    return 0;
}

/* FB-1 cut 4: word-level fixtures for the four additive ops (checked against the
 * nvdisasm decodes recorded in docs/numeric/FB1_CUT4_ELEMENTWISE.md). 0 = all good. */
int omega_blackwell_verify_codegen_fixtures_fb1cut4(void) {
    OmegaRegAlloc ra;
    memset(&ra, 0, sizeof(ra));
    ra.vreg_to_phys[0] = 2;  /* R2 */
    ra.vreg_to_phys[1] = 4;  /* R4 */
    ra.vreg_to_phys[2] = 9;  /* R9 */
    ra.vreg_to_phys[3] = 10; /* R10 */
    ra.vreg_to_phys[4] = 8;  /* R8 */
    uint32_t w[4];

    /* MUFU.EX2 R2, R4 */
    BlackwellIRInsn ex2 = { .op = BW_IR_MUFU_EX2, .dst_vreg = 0, .src1_vreg = 1 };
    if (encode_single_insn(&ex2, &ra, w) != 0) return -1;
    if (w[0] != 0x00027308U || w[1] != 4 || w[2] != 0x00000800U) return -2;
    /* BAR.SYNC.DEFER_BLOCKING 0x0 */
    BlackwellIRInsn bar = { .op = BW_IR_BAR_SYNC };
    if (encode_single_insn(&bar, &ra, w) != 0) return -3;
    if (w[0] != 0x00007b1dU || w[1] != 0 || w[2] != 0x00010000U || w[3] != 0x000fec00U) return -4;
    /* LDS R9, [R10+URZ] */
    BlackwellIRInsn lds = { .op = BW_IR_LDS32, .dst_vreg = 2, .src1_vreg = 3 };
    if (encode_single_insn(&lds, &ra, w) != 0) return -5;
    if (w[0] != 0x0a097984U || w[1] != 0xff || w[2] != 0x08000800U) return -6;
    /* STS [R8+URZ], R2 */
    BlackwellIRInsn sts = { .op = BW_IR_STS32, .src1_vreg = 4, .src2_vreg = 0 };
    if (encode_single_insn(&sts, &ra, w) != 0) return -7;
    if (w[0] != 0x08007988U || w[1] != 2 || w[2] != 0x080008ffU) return -8;
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

/* ---- FB-1 cut 1b: looped tensor matmul ------------------------------------------
 *
 * Control words (bundle bits 105..125, decoded from the chip-proven kernel and from
 * the vendor compiler's SASS for an equivalent HMMA loop on this chip, 2026-10-04):
 *   stall = bits 9..12 of w[3], yield = 13, write barrier = 14..16, read barrier =
 *   17..19, wait mask = 20..25. 7 in a barrier field means "none".
 *   LDG (A)      0x001ea800  stall 4, yield, sets SB2
 *   LDG.U16 (B)  0x002ea800  stall 4, yield, sets SB2
 *   pack IMAD    0x004fca00  stall 5, waits SB2 (every outstanding load has landed)
 *   HMMA         0x002fca00  stall 5 (fixed latency: the vendor compiler sets no
 *                            write barrier on HMMA either; consumers wait by stall count)
 *   STG          0x0007e200  stall 1, yield, read barrier SB3 (the store may read its
 *                            data/address registers late; the next writer waits SB3)
 *   wait SB3     mask bit 23 (0x00800000) on the first instruction that overwrites a
 *                            register a store reads: the C pointer advance.
 * Loop-carried registers are kept live to the end of their loop (keep_alive) because
 * the allocator is a straight-line linear scan and knows nothing about back edges.
 */
static void bw_keep_alive(BlackwellIRProgram *prog, int from_idx, int to_idx, int last_idx) {
    /* every vreg first defined in [from_idx, to_idx) stays live until last_idx */
    for (int v = 0; v < prog->regalloc.num_vregs; v++) {
        OmegaLiveInterval *iv = &prog->regalloc.intervals[v];
        if (iv->first_def >= from_idx && iv->first_def < to_idx && iv->last_use < last_idx)
            iv->last_use = last_idx;
    }
}

#define BW_APPEND(prog, ...) do { BlackwellIRInsn _i = { __VA_ARGS__ }; if (omega_bw_ir_append((prog), &_i) < 0) return -1; } while (0)

int omega_blackwell_codegen_matmul_tensor_loop_prog(const OmegaMatMulSpec *spec, uint32_t grid_x,
                                                    int mutant, BlackwellIRProgram *prog) {
    if (!spec || !prog) return -1;
    if (spec->m == 0 || spec->k == 0 || spec->n == 0) return -1;
    if (spec->precision != OMEGA_MATMUL_PRECISION_FP16 && spec->precision != OMEGA_MATMUL_PRECISION_BF16) return -1;
    if (spec->m % 16 != 0 || spec->n % 8 != 0 || spec->k % 16 != 0) {
        fprintf(stderr, "ERROR: looped MatMul tensor codegen requires M%%16, N%%8, K%%16 (got %ux%ux%u)\n",
                spec->m, spec->k, spec->n);
        return -1;
    }
    const uint32_t K = spec->k, N = spec->n, NT = N / 8, KSTEPS = K / 16;
    if (grid_x == 0 || grid_x > NT) return -1;
    /* every immediate below must fit 32 bits: 32 * N (B row-block stride in bytes) is the largest */
    if ((uint64_t)N * 32u > 0xffffffffull || (uint64_t)grid_x * 32u > 0xffffffffull) return -1;
    if (mutant && KSTEPS < 2) return -1;

    omega_bw_ir_init(prog);

    int v_ptr_a = omega_bw_ir_alloc_vreg64(prog), v_ptr_b = omega_bw_ir_alloc_vreg64(prog), v_ptr_c = omega_bw_ir_alloc_vreg64(prog);
    int uv_desc = omega_bw_ir_alloc_uvreg64(prog);
    int v_tid = omega_bw_ir_alloc_vreg(prog), v_ctaid_x = omega_bw_ir_alloc_vreg(prog), v_ctaid_y = omega_bw_ir_alloc_vreg(prog);
    int v_group = omega_bw_ir_alloc_vreg(prog), v_tig = omega_bw_ir_alloc_vreg(prog), v_tcol2 = omega_bw_ir_alloc_vreg(prog);
    int v_r0 = omega_bw_ir_alloc_vreg(prog), v_r1 = omega_bw_ir_alloc_vreg(prog), v_c1 = omega_bw_ir_alloc_vreg(prog);
    int v_idx = omega_bw_ir_alloc_vreg(prog), v_col = omega_bw_ir_alloc_vreg(prog), v_cnt = omega_bw_ir_alloc_vreg(prog);
    int v_pa0 = omega_bw_ir_alloc_vreg64(prog), v_pa1 = omega_bw_ir_alloc_vreg64(prog), v_pbcol = omega_bw_ir_alloc_vreg64(prog);
    int v_pc0 = omega_bw_ir_alloc_vreg64(prog), v_pc1 = omega_bw_ir_alloc_vreg64(prog);
    int v_pa0c = omega_bw_ir_alloc_vreg64(prog), v_pa1c = omega_bw_ir_alloc_vreg64(prog), v_pbc = omega_bw_ir_alloc_vreg64(prog);
    int v_ta2 = omega_bw_ir_alloc_vreg64(prog), v_ta3 = omega_bw_ir_alloc_vreg64(prog);
    int v_tb1 = omega_bw_ir_alloc_vreg64(prog), v_tb2 = omega_bw_ir_alloc_vreg64(prog), v_tb3 = omega_bw_ir_alloc_vreg64(prog);
    int v_tc1 = omega_bw_ir_alloc_vreg64(prog), v_tc3 = omega_bw_ir_alloc_vreg64(prog);
    int v_ra = omega_bw_ir_alloc_vreg128(prog), v_rb = omega_bw_ir_alloc_vreg64(prog), v_rd = omega_bw_ir_alloc_vreg128(prog);
    int v_b0 = omega_bw_ir_alloc_vreg(prog), v_b1 = omega_bw_ir_alloc_vreg(prog);
    if (v_b1 < 0 || uv_desc < 0) return -1;

    const uint32_t CTRL_LDA = 0x001ea800u, CTRL_LDB = 0x002ea800u, CTRL_PACK = 0x004fca00u;
    const uint32_t CTRL_HMMA = 0x002fca00u, CTRL_STG_RDBAR3 = 0x0007e200u, CTRL_IMADW_WAIT3 = 0x009fcc00u;
    const uint32_t CTRL_MOVRZ_WAIT3 = 0x008fe200u;

    /* prologue: descriptor, pointers, ids */
    BW_APPEND(prog, .op = BW_IR_LDCU64, .dst_vreg = uv_desc, .imm = 0x358, .is_uniform = true);
    BW_APPEND(prog, .op = BW_IR_LDC64, .dst_vreg = v_ptr_a, .imm = 0x380);
    BW_APPEND(prog, .op = BW_IR_LDC64, .dst_vreg = v_ptr_b, .imm = 0x388);
    BW_APPEND(prog, .op = BW_IR_LDC64, .dst_vreg = v_ptr_c, .imm = 0x390);
    BW_APPEND(prog, .op = BW_IR_S2R, .dst_vreg = v_tid, .imm = BW_SR_TID_X);
    BW_APPEND(prog, .op = BW_IR_S2R, .dst_vreg = v_ctaid_x, .imm = BW_SR_CTAID_X);
    BW_APPEND(prog, .op = BW_IR_S2R, .dst_vreg = v_ctaid_y, .imm = BW_SR_CTAID_Y);
    BW_APPEND(prog, .op = BW_IR_MOV_IMM, .dst_vreg = v_c1, .imm = 1);
    BW_APPEND(prog, .op = BW_IR_SHF_R, .dst_vreg = v_group, .src1_vreg = v_tid, .imm = 2);
    BW_APPEND(prog, .op = BW_IR_LOP3_AND, .dst_vreg = v_tig, .src1_vreg = v_tid, .imm = 3);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_tcol2, .src1_vreg = v_tig, .src2_vreg = -1, .imm = 2, .src3_vreg = -1);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_r0, .src1_vreg = v_ctaid_y, .src2_vreg = -1, .imm = 16, .src3_vreg = v_group);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_r1, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 8, .src3_vreg = v_r0);
    /* pa0 = A + (r0*K + tcol2)*2 ; pa1 likewise with r1 (A row-major, K bf16 per row) */
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r0, .src2_vreg = -1, .imm = K, .src3_vreg = v_tcol2);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pa0, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 2, .src3_vreg = v_ptr_a);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r1, .src2_vreg = -1, .imm = K, .src3_vreg = v_tcol2);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pa1, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 2, .src3_vreg = v_ptr_a);
    /* col = ctaid_x (first column tile of this CTA) */
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_col, .src1_vreg = v_ctaid_x, .src2_vreg = -1, .imm = 1, .src3_vreg = -1);
    /* pbcol = B + (tcol2*N + col*8 + group)*2 */
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_tcol2, .src2_vreg = -1, .imm = N, .src3_vreg = v_group);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_col, .src2_vreg = -1, .imm = 8, .src3_vreg = v_idx);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pbcol, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 2, .src3_vreg = v_ptr_b);
    /* pc0 = C + (r0*N + col*8 + tcol2)*4 ; pc1 with r1 */
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r0, .src2_vreg = -1, .imm = N, .src3_vreg = v_tcol2);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_col, .src2_vreg = -1, .imm = 8, .src3_vreg = v_idx);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pc0, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 4, .src3_vreg = v_ptr_c);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r1, .src2_vreg = -1, .imm = N, .src3_vreg = v_tcol2);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_col, .src2_vreg = -1, .imm = 8, .src3_vreg = v_idx);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pc1, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 4, .src3_vreg = v_ptr_c);

    /* ---- column-tile loop ---- */
    const int L_col = (int)prog->count;
    /* accumulator = 0 (the first write waits SB3: the previous tile's stores may still read it) */
    BW_APPEND(prog, .op = BW_IR_MOV_RZ, .dst_vreg = v_rd, .dst_subreg = 0, .control = CTRL_MOVRZ_WAIT3);
    BW_APPEND(prog, .op = BW_IR_MOV_RZ, .dst_vreg = v_rd, .dst_subreg = 1);
    BW_APPEND(prog, .op = BW_IR_MOV_RZ, .dst_vreg = v_rd, .dst_subreg = 2);
    BW_APPEND(prog, .op = BW_IR_MOV_RZ, .dst_vreg = v_rd, .dst_subreg = 3);
    /* K cursors start at the row/column bases (IMAD.WIDE with imm 0 is a 64-bit copy) */
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pa0c, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 0, .src3_vreg = v_pa0);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pa1c, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 0, .src3_vreg = v_pa1);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pbc, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 0, .src3_vreg = v_pbcol);
    BW_APPEND(prog, .op = BW_IR_MOV_IMM, .dst_vreg = v_cnt, .imm = mutant ? KSTEPS - 1 : KSTEPS);

    /* ---- K loop: one 16x16 A fragment and one 16x8 B fragment per step ---- */
    const int L_k = (int)prog->count;
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_ta2, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 16, .src3_vreg = v_pa0c);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_ta3, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 16, .src3_vreg = v_pa1c);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_tb1, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 2u * N, .src3_vreg = v_pbc);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_tb2, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 16u * N, .src3_vreg = v_pbc);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_tb3, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 18u * N, .src3_vreg = v_pbc);
    /* A: a0,a1 at (r0, kk+tcol2) ; a2,a3 at (r1, ...) ; a4,a5 at (r0, +8) ; a6,a7 at (r1, +8) */
    BW_APPEND(prog, .op = BW_IR_LDG_E, .dst_vreg = v_ra, .dst_subreg = 0, .src1_vreg = v_pa0c, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_LDA);
    BW_APPEND(prog, .op = BW_IR_LDG_E, .dst_vreg = v_ra, .dst_subreg = 1, .src1_vreg = v_pa1c, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_LDA);
    BW_APPEND(prog, .op = BW_IR_LDG_E, .dst_vreg = v_ra, .dst_subreg = 2, .src1_vreg = v_ta2, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_LDA);
    BW_APPEND(prog, .op = BW_IR_LDG_E, .dst_vreg = v_ra, .dst_subreg = 3, .src1_vreg = v_ta3, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_LDA);
    /* B: rows kk+tcol2, +1, +8, +9 at column col*8+group */
    BW_APPEND(prog, .op = BW_IR_LDG_E_U16, .dst_vreg = v_b0, .src1_vreg = v_pbc, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_LDB);
    BW_APPEND(prog, .op = BW_IR_LDG_E_U16, .dst_vreg = v_b1, .src1_vreg = v_tb1, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_LDB);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_rb, .dst_subreg = 0, .src1_vreg = v_b1, .src2_vreg = -1, .imm = 0x10000, .src3_vreg = v_b0, .control = CTRL_PACK);
    BW_APPEND(prog, .op = BW_IR_LDG_E_U16, .dst_vreg = v_b0, .src1_vreg = v_tb2, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_LDB);
    BW_APPEND(prog, .op = BW_IR_LDG_E_U16, .dst_vreg = v_b1, .src1_vreg = v_tb3, .src2_vreg = -1, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_LDB);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_rb, .dst_subreg = 1, .src1_vreg = v_b1, .src2_vreg = -1, .imm = 0x10000, .src3_vreg = v_b0, .control = CTRL_PACK);
    /* rd += ra * rb (accumulate in place) */
    BW_APPEND(prog, .op = (spec->precision == OMEGA_MATMUL_PRECISION_BF16) ? BW_IR_HMMA_BF16 : BW_IR_HMMA_F16,
              .dst_vreg = v_rd, .dst_subreg = 0, .src1_vreg = v_ra, .src1_subreg = 0, .src2_vreg = v_rb, .src2_subreg = 0,
              .src3_vreg = v_rd, .src3_subreg = 0, .control = CTRL_HMMA);
    /* advance K: A by 16 bf16 (32 bytes), B by 16 rows (32 N bytes); every load above has landed (packs waited SB2) */
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pa0c, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 32, .src3_vreg = v_pa0c);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pa1c, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 32, .src3_vreg = v_pa1c);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pbc, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 32u * N, .src3_vreg = v_pbc);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_cnt, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 0xffffffffu, .src3_vreg = v_cnt);
    BW_APPEND(prog, .op = BW_IR_ISETP_GE, .dst_vreg = -1, .src1_vreg = v_cnt, .src2_vreg = -1, .imm = 1, .src3_vreg = -1);
    {
        int here = (int)prog->count;
        BW_APPEND(prog, .op = BW_IR_BRA, .imm = (uint32_t)(int32_t)(L_k - here), .predicate_p0 = true);
    }
    const int L_k_end = (int)prog->count - 1;

    /* ---- store the 16x8 tile: d0 (r0, c), d1 (r0, c+1), d2 (r1, c), d3 (r1, c+1) ---- */
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_tc1, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 4, .src3_vreg = v_pc0);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_tc3, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 4, .src3_vreg = v_pc1);
    BW_APPEND(prog, .op = BW_IR_STG_E, .src1_vreg = v_pc0, .src2_vreg = v_rd, .src2_subreg = 0, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_STG_RDBAR3);
    BW_APPEND(prog, .op = BW_IR_STG_E, .src1_vreg = v_tc1, .src2_vreg = v_rd, .src2_subreg = 1, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_STG_RDBAR3);
    BW_APPEND(prog, .op = BW_IR_STG_E, .src1_vreg = v_pc1, .src2_vreg = v_rd, .src2_subreg = 2, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_STG_RDBAR3);
    BW_APPEND(prog, .op = BW_IR_STG_E, .src1_vreg = v_tc3, .src2_vreg = v_rd, .src2_subreg = 3, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_STG_RDBAR3);
    /* next column tile of this CTA: col += grid_x ; B by grid_x*8 bf16 ; C by grid_x*8 f32 (waits SB3 first) */
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_col, .src1_vreg = v_c1, .src2_vreg = -1, .imm = grid_x, .src3_vreg = v_col);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pc0, .src1_vreg = v_c1, .src2_vreg = -1, .imm = grid_x * 32u, .src3_vreg = v_pc0, .control = CTRL_IMADW_WAIT3);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pc1, .src1_vreg = v_c1, .src2_vreg = -1, .imm = grid_x * 32u, .src3_vreg = v_pc1);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pbcol, .src1_vreg = v_c1, .src2_vreg = -1, .imm = grid_x * 16u, .src3_vreg = v_pbcol);
    BW_APPEND(prog, .op = BW_IR_ISETP_GE, .dst_vreg = -1, .src1_vreg = v_col, .src2_vreg = -1, .imm = NT, .src3_vreg = -1);
    {
        int here = (int)prog->count;
        BW_APPEND(prog, .op = BW_IR_BRA, .imm = (uint32_t)(int32_t)(L_col - here), .predicate_p0 = true, .predicate_not = true);
    }
    const int L_col_end = (int)prog->count - 1;
    BW_APPEND(prog, .op = BW_IR_EXIT);
    BW_APPEND(prog, .op = BW_IR_BRA);

    /* loop-carried values live to the end of their loop */
    bw_keep_alive(prog, 0, L_col, L_col_end);
    bw_keep_alive(prog, L_col, L_k, L_k_end);
    /* K-loop temporaries too: a load's address pair must never be recycled by a
     * later instruction of the same iteration (the load may read it late). */
    bw_keep_alive(prog, L_k, L_k_end, L_k_end);
    /* the accumulator is read by the stores after the K loop */
    if (prog->regalloc.intervals[v_rd].last_use < L_col_end) prog->regalloc.intervals[v_rd].last_use = L_col_end;

    return omega_bw_regalloc_solve(prog);
}

int omega_blackwell_codegen_matmul_tensor_loop(const OmegaMatMulSpec *spec, uint32_t grid_x,
                                               int mutant, OmegaBlackwellKernel *kernel) {
    if (!spec || !kernel) return -1;
    BlackwellIRProgram prog;
    if (omega_blackwell_codegen_matmul_tensor_loop_prog(spec, grid_x, mutant, &prog) != 0) return -1;
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

/* ---- omega #328: fragment-major tensor matmul ----------------------------------------
 *
 * The looped kernel above reads B as four scattered 2-byte loads per K step and waits for
 * them before the next step (pack waits SB2), so each warp has one 256-byte step in flight;
 * at m = 1 on GB10 that streams the weights at about 28 GB/s (omega #328 measurement).
 * Here A and B arrive in the order HMMA.16816 consumes them (omega_matmul_frag_pack_a / _b):
 *   A block (mt, ks): 512 bytes, lane L at L*16: a0..a7, one LDG.E.128 per step;
 *   B block (nt, ks): 256 bytes, lane L at L*8:  b0..b3, one LDG.E.64 per step;
 * consecutive K steps of one tile are adjacent, so a warp streams its column tile. Each loop
 * pass issues `unroll` steps of loads at immediate offsets (A on SB2, B on SB4), then runs the HMMA
 * chain: every HMMA waits SB2 and SB4 (stall 15) and a NOP (stall 14) follows it, the 29 cycles nvcc
 * puts between dependent HMMA.16816 (omega_blackwell_verify_codegen_fixtures_fragmm).
 * The cursors advance only after the chain, when every load has landed (the HMMAs waited
 * SB2 and SB4), as in the looped kernel. Stores, the column-tile loop and the C layout (row-major
 * f32) are the looped kernel's. */
/* A loads set SB2 and B loads SB4 (nvcc spreads loads over several scoreboards too); at the
 * product unroll of 4 that is 4 loads per scoreboard per pass. The scoreboard depth is not
 * documented to us; splitting did not make unroll 8 complete (omega_matmul_frag_unroll). */
#define BW_FRAG_CTRL_LDG_A 0x000ea800u /* stall 4, yield, sets SB2 (nvcc word) */
#define BW_FRAG_CTRL_LDG_B 0x000f2800u /* stall 4, yield, sets SB4 (nvcc word) */
#define BW_FRAG_CTRL_HMMA  0x014fde00u /* stall 15, waits SB2 and SB4 (nvcc: dependent HMMA chain) */
#define BW_FRAG_CTRL_NOP  0x000fdc00u /* stall 14 (nvcc, between dependent HMMAs) */

static void bw_live_until(BlackwellIRProgram *prog, int vreg, int last_idx) {
    if (prog->regalloc.intervals[vreg].last_use < last_idx) prog->regalloc.intervals[vreg].last_use = last_idx;
}

int omega_blackwell_codegen_matmul_frag_prog(const OmegaMatMulSpec *spec, uint32_t grid_x, uint32_t unroll,
                                             int mutant, BlackwellIRProgram *prog) {
    if (!spec || !prog) return -1;
    if (spec->m == 0 || spec->k == 0 || spec->n == 0) return -1;
    if (spec->precision != OMEGA_MATMUL_PRECISION_FP16 && spec->precision != OMEGA_MATMUL_PRECISION_BF16) return -1;
    if (spec->m % 16 != 0 || spec->n % 8 != 0 || spec->k % 16 != 0) return -1;
    if (unroll != 1 && unroll != 2 && unroll != 4 && unroll != 8) return -1;
    const uint32_t K = spec->k, N = spec->n, NT = N / 8, KS = K / 16;
    if (KS % unroll != 0) return -1;
    const uint32_t ITERS = KS / unroll;
    if (grid_x == 0 || grid_x > NT) return -1;
    if (mutant && ITERS < 2) return -1;
    /* every index and immediate below must fit 32 bits (rows up to OMEGA_BW_MATMUL_MAX_M) */
    if ((uint64_t)NT * KS * 32u > 0xffffffffull || (uint64_t)(OMEGA_BW_MATMUL_MAX_M / 16u) * KS * 32u > 0xffffffffull ||
        (uint64_t)grid_x * KS * 256u > 0xffffffffull || (uint64_t)N * 32u > 0xffffffffull ||
        (uint64_t)OMEGA_BW_MATMUL_MAX_M * N > 0xffffffffull)
        return -1;

    omega_bw_ir_init(prog);
    int v_ptr_a = omega_bw_ir_alloc_vreg64(prog), v_ptr_b = omega_bw_ir_alloc_vreg64(prog), v_ptr_c = omega_bw_ir_alloc_vreg64(prog);
    int uv_desc = omega_bw_ir_alloc_uvreg64(prog);
    int v_tid = omega_bw_ir_alloc_vreg(prog), v_ctaid_x = omega_bw_ir_alloc_vreg(prog), v_ctaid_y = omega_bw_ir_alloc_vreg(prog);
    int v_group = omega_bw_ir_alloc_vreg(prog), v_tig = omega_bw_ir_alloc_vreg(prog), v_tcol2 = omega_bw_ir_alloc_vreg(prog);
    int v_r0 = omega_bw_ir_alloc_vreg(prog), v_r1 = omega_bw_ir_alloc_vreg(prog), v_c1 = omega_bw_ir_alloc_vreg(prog);
    int v_idx = omega_bw_ir_alloc_vreg(prog), v_col = omega_bw_ir_alloc_vreg(prog), v_cnt = omega_bw_ir_alloc_vreg(prog);
    int v_pa = omega_bw_ir_alloc_vreg64(prog), v_pbcol = omega_bw_ir_alloc_vreg64(prog);
    int v_pc0 = omega_bw_ir_alloc_vreg64(prog), v_pc1 = omega_bw_ir_alloc_vreg64(prog);
    int v_pac = omega_bw_ir_alloc_vreg64(prog), v_pbc = omega_bw_ir_alloc_vreg64(prog);
    int v_tc1 = omega_bw_ir_alloc_vreg64(prog), v_tc3 = omega_bw_ir_alloc_vreg64(prog);
    int v_rd = omega_bw_ir_alloc_vreg128(prog);
    int v_ra[8], v_rb[8];
    for (uint32_t u = 0; u < unroll; u++) { v_ra[u] = omega_bw_ir_alloc_vreg128(prog); v_rb[u] = omega_bw_ir_alloc_vreg64(prog); }
    if (v_rb[unroll - 1] < 0 || v_ra[unroll - 1] < 0 || uv_desc < 0) return -1;

    const uint32_t CTRL_STG_RDBAR3 = 0x0007e200u, CTRL_IMADW_WAIT3 = 0x009fcc00u, CTRL_MOVRZ_WAIT3 = 0x008fe200u;

    /* prologue: descriptor, pointers, ids (as the looped kernel) */
    BW_APPEND(prog, .op = BW_IR_LDCU64, .dst_vreg = uv_desc, .imm = 0x358, .is_uniform = true);
    BW_APPEND(prog, .op = BW_IR_LDC64, .dst_vreg = v_ptr_a, .imm = 0x380);
    BW_APPEND(prog, .op = BW_IR_LDC64, .dst_vreg = v_ptr_b, .imm = 0x388);
    BW_APPEND(prog, .op = BW_IR_LDC64, .dst_vreg = v_ptr_c, .imm = 0x390);
    BW_APPEND(prog, .op = BW_IR_S2R, .dst_vreg = v_tid, .imm = BW_SR_TID_X);
    BW_APPEND(prog, .op = BW_IR_S2R, .dst_vreg = v_ctaid_x, .imm = BW_SR_CTAID_X);
    BW_APPEND(prog, .op = BW_IR_S2R, .dst_vreg = v_ctaid_y, .imm = BW_SR_CTAID_Y);
    BW_APPEND(prog, .op = BW_IR_MOV_IMM, .dst_vreg = v_c1, .imm = 1);
    BW_APPEND(prog, .op = BW_IR_SHF_R, .dst_vreg = v_group, .src1_vreg = v_tid, .imm = 2);
    BW_APPEND(prog, .op = BW_IR_LOP3_AND, .dst_vreg = v_tig, .src1_vreg = v_tid, .imm = 3);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_tcol2, .src1_vreg = v_tig, .src2_vreg = -1, .imm = 2, .src3_vreg = -1);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_r0, .src1_vreg = v_ctaid_y, .src2_vreg = -1, .imm = 16, .src3_vreg = v_group);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_r1, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 8, .src3_vreg = v_r0);
    /* pa = A + (ctaid_y * KS * 32 + tid) * 16: this CTA's row tile, this lane's 16 bytes of step 0 */
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_ctaid_y, .src2_vreg = -1, .imm = KS * 32u, .src3_vreg = v_tid);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pa, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 16, .src3_vreg = v_ptr_a);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_col, .src1_vreg = v_ctaid_x, .src2_vreg = -1, .imm = 1, .src3_vreg = -1);
    /* pbcol = B + (col * KS * 32 + tid) * 8: column tile col, this lane's 8 bytes of step 0 */
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_col, .src2_vreg = -1, .imm = KS * 32u, .src3_vreg = v_tid);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pbcol, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 8, .src3_vreg = v_ptr_b);
    /* pc0 = C + (r0*N + col*8 + tcol2)*4 ; pc1 with r1 */
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r0, .src2_vreg = -1, .imm = N, .src3_vreg = v_tcol2);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_col, .src2_vreg = -1, .imm = 8, .src3_vreg = v_idx);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pc0, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 4, .src3_vreg = v_ptr_c);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_r1, .src2_vreg = -1, .imm = N, .src3_vreg = v_tcol2);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_idx, .src1_vreg = v_col, .src2_vreg = -1, .imm = 8, .src3_vreg = v_idx);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pc1, .src1_vreg = v_idx, .src2_vreg = -1, .imm = 4, .src3_vreg = v_ptr_c);

    /* ---- column-tile loop ---- */
    const int L_col = (int)prog->count;
    BW_APPEND(prog, .op = BW_IR_MOV_RZ, .dst_vreg = v_rd, .dst_subreg = 0, .control = CTRL_MOVRZ_WAIT3);
    BW_APPEND(prog, .op = BW_IR_MOV_RZ, .dst_vreg = v_rd, .dst_subreg = 1);
    BW_APPEND(prog, .op = BW_IR_MOV_RZ, .dst_vreg = v_rd, .dst_subreg = 2);
    BW_APPEND(prog, .op = BW_IR_MOV_RZ, .dst_vreg = v_rd, .dst_subreg = 3);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pac, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 0, .src3_vreg = v_pa);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pbc, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 0, .src3_vreg = v_pbcol);
    BW_APPEND(prog, .op = BW_IR_MOV_IMM, .dst_vreg = v_cnt, .imm = mutant ? ITERS - 1 : ITERS);

    /* ---- K loop: `unroll` steps of loads, then the HMMA chain ---- */
    const int L_k = (int)prog->count;
    for (uint32_t u = 0; u < unroll; u++) {
        BW_APPEND(prog, .op = BW_IR_LDG_E_128, .dst_vreg = v_ra[u], .src1_vreg = v_pac, .src2_vreg = -1, .src3_vreg = -1,
                  .ureg = uv_desc, .imm = u * 512u, .control = BW_FRAG_CTRL_LDG_A);
        BW_APPEND(prog, .op = BW_IR_LDG_E_64, .dst_vreg = v_rb[u], .src1_vreg = v_pbc, .src2_vreg = -1, .src3_vreg = -1,
                  .ureg = uv_desc, .imm = u * 256u, .control = BW_FRAG_CTRL_LDG_B);
    }
    const BlackwellIROpcode hmma = (spec->precision == OMEGA_MATMUL_PRECISION_BF16) ? BW_IR_HMMA_BF16 : BW_IR_HMMA_F16;
    for (uint32_t u = 0; u < unroll; u++) {
        BW_APPEND(prog, .op = hmma, .dst_vreg = v_rd, .src1_vreg = v_ra[u], .src2_vreg = v_rb[u], .src3_vreg = v_rd,
                  .control = BW_FRAG_CTRL_HMMA);
        BW_APPEND(prog, .op = BW_IR_NOP, .dst_vreg = -1, .src1_vreg = -1, .src2_vreg = -1, .src3_vreg = -1, .ureg = -1,
                  .control = BW_FRAG_CTRL_NOP);
    }
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pac, .src1_vreg = v_c1, .src2_vreg = -1, .imm = unroll * 512u, .src3_vreg = v_pac);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pbc, .src1_vreg = v_c1, .src2_vreg = -1, .imm = unroll * 256u, .src3_vreg = v_pbc);
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_cnt, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 0xffffffffu, .src3_vreg = v_cnt);
    BW_APPEND(prog, .op = BW_IR_ISETP_GE, .dst_vreg = -1, .src1_vreg = v_cnt, .src2_vreg = -1, .imm = 1, .src3_vreg = -1);
    {
        int here = (int)prog->count;
        BW_APPEND(prog, .op = BW_IR_BRA, .imm = (uint32_t)(int32_t)(L_k - here), .predicate_p0 = true);
    }
    const int L_k_end = (int)prog->count - 1;

    /* ---- store the 16x8 tile (as the looped kernel) ---- */
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_tc1, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 4, .src3_vreg = v_pc0);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_tc3, .src1_vreg = v_c1, .src2_vreg = -1, .imm = 4, .src3_vreg = v_pc1);
    BW_APPEND(prog, .op = BW_IR_STG_E, .src1_vreg = v_pc0, .src2_vreg = v_rd, .src2_subreg = 0, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_STG_RDBAR3);
    BW_APPEND(prog, .op = BW_IR_STG_E, .src1_vreg = v_tc1, .src2_vreg = v_rd, .src2_subreg = 1, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_STG_RDBAR3);
    BW_APPEND(prog, .op = BW_IR_STG_E, .src1_vreg = v_pc1, .src2_vreg = v_rd, .src2_subreg = 2, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_STG_RDBAR3);
    BW_APPEND(prog, .op = BW_IR_STG_E, .src1_vreg = v_tc3, .src2_vreg = v_rd, .src2_subreg = 3, .src3_vreg = -1, .ureg = uv_desc, .control = CTRL_STG_RDBAR3);
    /* next column tile: col += grid_x ; C by grid_x*8 f32 (waits SB3 first) ; B by grid_x tiles of KS*256 bytes */
    BW_APPEND(prog, .op = BW_IR_IMAD, .dst_vreg = v_col, .src1_vreg = v_c1, .src2_vreg = -1, .imm = grid_x, .src3_vreg = v_col);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pc0, .src1_vreg = v_c1, .src2_vreg = -1, .imm = grid_x * 32u, .src3_vreg = v_pc0, .control = CTRL_IMADW_WAIT3);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pc1, .src1_vreg = v_c1, .src2_vreg = -1, .imm = grid_x * 32u, .src3_vreg = v_pc1);
    BW_APPEND(prog, .op = BW_IR_IMAD_WIDE, .dst_vreg = v_pbcol, .src1_vreg = v_c1, .src2_vreg = -1, .imm = grid_x * KS * 256u, .src3_vreg = v_pbcol);
    BW_APPEND(prog, .op = BW_IR_ISETP_GE, .dst_vreg = -1, .src1_vreg = v_col, .src2_vreg = -1, .imm = NT, .src3_vreg = -1);
    {
        int here = (int)prog->count;
        BW_APPEND(prog, .op = BW_IR_BRA, .imm = (uint32_t)(int32_t)(L_col - here), .predicate_p0 = true, .predicate_not = true);
    }
    const int L_col_end = (int)prog->count - 1;
    BW_APPEND(prog, .op = BW_IR_EXIT);
    BW_APPEND(prog, .op = BW_IR_BRA);

    /* Loop-carried values live to the end of their loop (the allocator knows no back edges).
     * Only these are read in a later pass than the one that wrote them; prologue temporaries
     * are dead before the first loop and may be reused. K-loop temporaries (fragments) are
     * kept to the end of the pass, as in the looped kernel. */
    const int col_carried[] = { v_c1, v_col, v_pa, v_pbcol, v_pc0, v_pc1, v_rd };
    for (size_t i = 0; i < sizeof col_carried / sizeof col_carried[0]; i++) bw_live_until(prog, col_carried[i], L_col_end);
    const int k_carried[] = { v_pac, v_pbc, v_cnt };
    for (size_t i = 0; i < sizeof k_carried / sizeof k_carried[0]; i++) bw_live_until(prog, k_carried[i], L_k_end);
    bw_keep_alive(prog, L_k, L_k_end, L_k_end);
    return omega_bw_regalloc_solve(prog);
}

int omega_blackwell_codegen_matmul_frag(const OmegaMatMulSpec *spec, uint32_t grid_x, uint32_t unroll,
                                        int mutant, OmegaBlackwellKernel *kernel) {
    if (!spec || !kernel) return -1;
    BlackwellIRProgram prog;
    if (omega_blackwell_codegen_matmul_frag_prog(spec, grid_x, unroll, mutant, &prog) != 0) return -1;
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

/* Capped at 4: unroll 8 (72 registers, 16 loads per pass) never completed on the GB10, with all
 * loads on SB2 and again with A on SB2 and B on SB4 (2026-10-07 07:50Z and 07:54Z, omega #328),
 * while unroll 4 (48 registers) ran every Qwen3 decode shape correctly. Cause UNKNOWN; the next
 * experiment is the unroll-4 kernel launched with 72 declared registers (one variable). */
uint32_t omega_matmul_frag_unroll(uint32_t kp) {
    const uint32_t ks = kp / 16u;
    return ks % 4u == 0 ? 4u : ks % 2u == 0 ? 2u : 1u;
}

void omega_matmul_frag_pack_a(const uint16_t *a, uint32_t m, uint32_t k, uint32_t mp, uint32_t kp, uint16_t *dst) {
    const uint32_t KS = kp / 16u;
    memset(dst, 0, (size_t)mp * kp * sizeof *dst);
    for (uint32_t r = 0; r < m; r++) {
        const uint32_t mt = r / 16u, rr = r % 16u, group = rr % 8u, hi = rr / 8u;
        for (uint32_t c = 0; c < k; c++) {
            const uint32_t ks = c / 16u, cc = c % 16u, khi = cc / 8u, tig = (cc % 8u) / 2u, lo = cc % 2u;
            const size_t lane = group * 4u + tig;
            dst[(((size_t)mt * KS + ks) * 32u + lane) * 8u + khi * 4u + hi * 2u + lo] = a[(size_t)r * k + c];
        }
    }
}

void omega_matmul_frag_pack_b(const uint16_t *b, uint32_t k, uint32_t n, uint32_t kp,
                              uint32_t nt0, uint32_t nt1, uint16_t *dst) {
    const uint32_t KS = kp / 16u;
    memset(dst, 0, (size_t)(nt1 - nt0) * KS * 128u * sizeof *dst);
    for (uint32_t row = 0; row < k; row++) {
        const uint32_t ks = row / 16u, rr = row % 16u, khi = rr / 8u, tig = (rr % 8u) / 2u, lo = rr % 2u;
        const uint16_t *src = &b[(size_t)row * n];
        for (uint32_t nt = nt0; nt < nt1; nt++) {
            const uint32_t c0 = nt * 8u;
            for (uint32_t group = 0; group < 8u && c0 + group < n; group++) {
                const size_t lane = group * 4u + tig;
                dst[(((size_t)(nt - nt0) * KS + ks) * 32u + lane) * 4u + khi * 2u + lo] = src[c0 + group];
            }
        }
    }
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
