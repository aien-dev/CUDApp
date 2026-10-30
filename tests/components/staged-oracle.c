/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Verification only. The existing independent C generator/allocator/encoder
 * supplies every successful expected bit; this file has no GEMM algorithm.
 * Protocol and suggested cases are in staged-caller.om. No GPU is opened. */
#include "omega_blackwell_codegen_gemm.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

static void word(uint64_t v) {
    for (unsigned j=0;j<8;j++) putchar((int)((v>>(j*8))&255));
}
static uint64_t reg(int v) { return v<0 ? UINT64_MAX : (uint64_t)v; }
static int power2(uint64_t v) { return v && !(v&(v-1)); }
/* The native bounded contract is checked independently, before narrowing. */
static int supported(const uint64_t s[16]) {
    if (!s[0] || s[0]>16384 || (s[1]!=8 && s[1]!=16)) return 0;
    if (s[2]!=1 && s[2]!=2 && s[2]!=4 && s[2]!=8) return 0;
    if (!power2(s[3]) || s[3]>32 || (s[2]*s[1])%s[3]) return 0;
    if (s[2]*s[1]/s[3]>8 || s[2]*128*s[1]>16384) return 0;
    if (s[0]%(32*s[1])) return 0;
    const uint64_t ks=s[0]/(32*s[1]);
    if ((ks&1) && ks>3) return 0;
    if (s[4]>1 || (s[4] && s[2]>4)) return 0;
    for (unsigned j=5;j<16;j++) if (s[j]) return 0;
    return 1;
}
int main(int argc, char **argv) {
    uint64_t s[16]={0};
    if (argc==1) {
        unsigned char raw[128];
        if (fread(raw,1,sizeof raw,stdin)!=sizeof raw) return 65;
        for (unsigned k=0;k<16;k++)
            for (unsigned j=0;j<8;j++) s[k]|=(uint64_t)raw[k*8+j]<<(j*8);
    } else if (argc==6) {
        for (unsigned k=0;k<5;k++) {
            char *tail=NULL;
            if (argv[k+1][0]<'0' || argv[k+1][0]>'9') return 65;
            errno=0;
            s[k]=strtoull(argv[k+1],&tail,10);
            if (errno || !tail || *tail) return 65;
        }
    } else return 65;
    if (!supported(s)) { word(1); return ferror(stdout)?74:0; }
    OmegaFp8GemmSpec spec={0};
    spec.k=(uint32_t)s[0]; spec.lane_bytes=(uint32_t)s[1];
    spec.mt=(uint32_t)s[2]; spec.warps=(uint32_t)s[3];
    spec.rowmap=(uint32_t)s[4];
    BlackwellIRProgram p;
    memset(&p,0,sizeof p);
    if (omega_blackwell_codegen_fp8gemm_prog(&spec,&p)!=0) return 66;
    uint8_t code[16384]; size_t nbytes=0;
    if (omega_bw_encode_program(&p,code,sizeof code,&nbytes)!=0) return 68;
    word(0); word(p.count); word(p.regalloc.num_vregs);
    word(p.regalloc.num_uvregs); word(omega_fp8gemm_shared_bytes(&spec));
    word(p.regalloc.peak_gpr_usage); word(p.regalloc.peak_ugpr_usage);
    for (int v=0;v<p.regalloc.num_vregs;v++) {
        const OmegaLiveInterval *q=&p.regalloc.intervals[v];
        word(q->bundle_size?q->bundle_size:(q->is_pair?2:(q->is_quad?4:1)));
        word(reg(p.regalloc.vreg_to_phys[v]));
    }
    for (int v=0;v<p.regalloc.num_uvregs;v++) {
        word(p.regalloc.uintervals[v].is_pair?2:1);
        word(reg(p.regalloc.uvreg_to_phys[v]));
    }
    for (size_t k=0;k<p.count;k++) {
        const BlackwellIRInsn *i=&p.insns[k];
        word(i->op); word(reg(i->dst_vreg)); word(reg(i->src1_vreg));
        word(reg(i->src2_vreg)); word(reg(i->src3_vreg)); word(reg(i->ureg));
        word((uint64_t)i->imm|((uint64_t)i->control<<32));
        word((uint64_t)i->is_uniform|((uint64_t)i->predicate_p0<<1)|
             ((uint64_t)i->predicate_not<<2)|((uint64_t)i->bar_reg<<8)|
             ((uint64_t)i->dst_subreg<<16)|((uint64_t)i->src1_subreg<<24)|
             ((uint64_t)i->src2_subreg<<32)|((uint64_t)i->src3_subreg<<40));
    }
    if (fwrite(code,1,nbytes,stdout)!=nbytes || ferror(stdout)) return 74;
    return 0;
}
