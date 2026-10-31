/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Verification only: existing C algorithm and independently retained C encoder. */
#include "omega_blackwell_codegen_moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void word(uint64_t v) { for (unsigned j=0;j<8;j++) putchar((int)((v>>(j*8))&255)); }
static uint64_t reg(int v) { return v < 0 ? UINT64_MAX : (uint64_t)v; }
int main(int argc,char **argv) {
    if(argc!=6) return 65;
    OmegaFp8GemvSpec s={0};
    s.k=(uint32_t)strtoul(argv[1],NULL,10);s.lane_bytes=(uint32_t)strtoul(argv[2],NULL,10);
    s.rows=(uint32_t)strtoul(argv[3],NULL,10);s.warps=(uint32_t)strtoul(argv[4],NULL,10);
    s.wfirst=(uint32_t)strtoul(argv[5],NULL,10);
    BlackwellIRProgram p;
    omega_bw_ir_init(&p);
    if(omega_blackwell_codegen_fp8_gemv_prog(&s,&p)!=0) return 66;
    uint8_t code[16384]; size_t nbytes=0;
    if(omega_bw_encode_program(&p,code,sizeof code,&nbytes)!=0) return 67;
    word(p.count);word(p.regalloc.num_vregs);word(p.regalloc.num_uvregs);word(0);
    word(p.regalloc.peak_gpr_usage);word(p.regalloc.peak_ugpr_usage);
    for(int v=0;v<p.regalloc.num_vregs;v++) {
        const OmegaLiveInterval *q=&p.regalloc.intervals[v];
        word(q->bundle_size?q->bundle_size:(q->is_pair?2:(q->is_quad?4:1)));
        word(reg(p.regalloc.vreg_to_phys[v]));
    }
    for(int v=0;v<p.regalloc.num_uvregs;v++) {
        word(p.regalloc.uintervals[v].is_pair?2:1);word(reg(p.regalloc.uvreg_to_phys[v]));
    }
    for(size_t k=0;k<p.count;k++) {
        const BlackwellIRInsn *i=&p.insns[k];
        word(i->op);word(reg(i->dst_vreg));word(reg(i->src1_vreg));word(reg(i->src2_vreg));
        word(reg(i->src3_vreg));word(reg(i->ureg));
        word((uint64_t)i->imm|((uint64_t)i->control<<32));
        word((uint64_t)i->is_uniform|((uint64_t)i->predicate_p0<<1)|((uint64_t)i->predicate_not<<2)|
             ((uint64_t)i->bar_reg<<8)|((uint64_t)i->dst_subreg<<16)|((uint64_t)i->src1_subreg<<24)|
             ((uint64_t)i->src2_subreg<<32)|((uint64_t)i->src3_subreg<<40));
    }
    if(fwrite(code,1,nbytes,stdout)!=nbytes) return 74;
    return 0;
}
