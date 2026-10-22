/* SPDX-License-Identifier: MIT */
/* Adversarial ties in the production router program, executed by the existing
 * warp simulator. No near-margin exemption and no GPU access. */
#include "omega_gpu_router_api.h"
#include "omega_gpu_numerical_contract.h"
#include "bw_warp_sim_k.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
#define CHECK(x) do{checks++;if(!(x)){fprintf(stderr,"FAIL %u: %s\n",checks,#x);return 1;}}while(0)
static struct {
    float x[2048], norm[2048], xo[2048];
    uint32_t ids[9];float weights[9];
    uint64_t gu[514],dn[514],tgu[18],tdn[18];
} b;
int main(void) {
    static BlackwellIRProgram prog;static uint8_t smem[16384];static float baseline[2048];
    for(unsigned shape=0;shape<2;shape++) for(int scale=0;scale<2;scale++) for(int order=0;order<2;order++) for(int mutant=0;mutant<=3;mutant++) {
        if(mutant==1)continue;
        unsigned experts=shape?256:128, slots=shape?9:8;
        OmegaRouterSpec spec={.warps=8,.depth=4,.experts=experts,.shared=shape,.mutant=mutant};
        OmegaRouterSmem sm;CHECK(!omega_router_smem(&spec,&sm));CHECK(sm.total<=sizeof smem);
        CHECK(!omega_blackwell_codegen_router_prog(&spec,&prog));
        uint16_t *gate=calloc((size_t)(experts+shape)*2048,2);CHECK(gate);
        memset(&b,0,sizeof b);
        for(unsigned i=0;i<2048;i++)b.x[i]=b.norm[i]=1;
        for(unsigned i=0;i<(experts+shape)*2;i++){b.gu[i]=100+i;b.dn[i]=1000+i;}
        for(unsigned i=0;i<slots;i++)b.ids[i]=UINT32_MAX;
        WsSim s;memset(&s,0,sizeof s);s.p=&prog;s.threads=256;s.order=order;s.smem=smem;s.smem_bytes=sm.total;
        uint64_t addresses[]={(uintptr_t)b.x,(uintptr_t)gate,(uintptr_t)b.norm,(uintptr_t)b.xo,(uintptr_t)b.ids,
          (uintptr_t)b.weights,(uintptr_t)b.gu,(uintptr_t)b.dn,(uintptr_t)b.tgu,(uintptr_t)b.tdn};
        for(unsigned i=0;i<10;i++){s.cbank[(0x380+8*i)/4]=(uint32_t)addresses[i];s.cbank[(0x380+8*i)/4+1]=(uint32_t)(addresses[i]>>32);}
        float eps=1e-6f,sc=scale?0x1p64f:1;
        memcpy(&s.cbank[OMEGA_ROUTER_ARG_EPS/4],&eps,4);memcpy(&s.cbank[OMEGA_ROUTER_ARG_OUT_SCALE/4],&sc,4);
        s.cbank[OMEGA_ROUTER_ARG_NORM_TOPK/4]=1;s.nranges=2;
        s.ranges[0]=(WsRange){(uintptr_t)&b,(uintptr_t)&b+sizeof b};
        s.ranges[1]=(WsRange){(uintptr_t)gate,(uintptr_t)gate+(size_t)(experts+shape)*2048*2};
        int rc=ws_run(&s,1,1);free(gate);CHECK(!rc);
        int badids=0,badtables=0;
        for(unsigned i=0;i<slots;i++) {
            unsigned expected=i==8?experts:i;
            badids|=b.ids[i]!=expected;
            badtables|=b.tgu[2*i]!=b.gu[2*expected] || b.tdn[2*i]!=b.dn[2*expected];
        }
        if(mutant==2){CHECK(badids);continue;}
        CHECK(!badids); /* actual tied router IDs must be 0..7, plus shared */
        if(mutant==3){CHECK(badtables);continue;}
        CHECK(!badtables);
        for(unsigned i=0;i<slots;i++)CHECK(b.weights[i]==(i==8?0.5f:0.125f));
        if(!scale && !order)memcpy(baseline,b.xo,sizeof baseline);
        float expected[2048];for(unsigned i=0;i<2048;i++)expected[i]=baseline[i]*sc;
        CHECK(!omega_numeric_compare(expected,b.xo,2048,OMEGA_NUMERIC_REPEATABLE,0,0));
    }
    printf("numerical_router_tie_test: %u checks PASS; generated router ties/scales, masks/directories mutants\n",checks);return 0;
}
