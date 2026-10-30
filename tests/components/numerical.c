/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Verification only: substitute native-produced IR at the existing test's
 * generator seam. Reuse its unchanged simulator, data, packing and oracle. */
#ifdef COMPONENT_STAGED
#define omega_blackwell_codegen_fp8gemm_prog component_native_staged
#define main component_original_staged_main
#include "../gpu_fp8gemm_test.c"
#undef main
#undef omega_blackwell_codegen_fp8gemm_prog
#else
#define omega_blackwell_codegen_fp8_gemv_prog component_native_direct
#define omega_blackwell_codegen_fp8_gemv2_prog component_native_rowpipe
#define main component_original_direct_main
#include "../gpu_fp8_api_test.c"
#undef main
#undef omega_blackwell_codegen_fp8_gemv_prog
#undef omega_blackwell_codegen_fp8_gemv2_prog
#endif
#include "native-frame.h"
static uint32_t shape[5];
#ifdef COMPONENT_STAGED
int component_native_staged(const OmegaFp8GemmSpec *s,BlackwellIRProgram *p) {
    if(s->k!=shape[0]||s->lane_bytes!=shape[1]||s->mt!=shape[2]||s->warps!=shape[3]||s->rowmap!=shape[4]||s->mutant) return -1;
    *p=native_program;return 0;
}
#else
static int component_native_gemv(const OmegaFp8GemvSpec *s,BlackwellIRProgram *p) {
    if(s->k!=shape[0]||s->lane_bytes!=shape[1]||s->rows!=shape[2]||s->warps!=shape[3]||s->wfirst!=shape[4]||s->mutant||s->coal||s->depth||s->sched) return -1;
    *p=native_program;return 0;
}
int component_native_direct(const OmegaFp8GemvSpec *s,BlackwellIRProgram *p) {return component_native_gemv(s,p);}
int component_native_rowpipe(const OmegaFp8GemvSpec *s,BlackwellIRProgram *p) {return component_native_gemv(s,p);}
#endif
int main(int argc,char **argv) {
    /* frame K LB rows/MT warps wfirst/rowmap family */
    if(argc!=8) return 65;
    for(unsigned j=0;j<5;j++) shape[j]=(uint32_t)strtoul(argv[j+2],NULL,10);
    char msg[512];size_t bad=0;int rc;
#ifdef COMPONENT_STAGED
    if(!frame_load(argv[1],1)) return 66;
    SimPt point={shape[0],40,shape[1],shape[2],shape[3],2,{1,5,9},3};
    int moved=0;
    rc=sim_point2(&point,0,41,(int)shape[4],1,0,msg,sizeof msg,&bad,&moved);
    if(rc||bad||(shape[4]&&!moved)) {fprintf(stderr,"native staged numerical: rc%d bad%zu: %s\n",rc,bad,msg);return 1;}
    printf("native staged both fragment orders, empty segment, tail rows, multi-segment%s: %s\n",shape[4]?", YMAP movement":"",msg);
    if(!frame_wrong_result()) return 67;
    bad=0;rc=sim_point2(&point,0,41,(int)shape[4],1,0,msg,sizeof msg,&bad,&moved);
    if(rc||!bad) {fprintf(stderr,"FFMA->FADD wrong-result mutant not numerically caught: %s\n",msg);return 2;}
    printf("wrong-result mutant: %s\n",msg);
    native_program.insns[0].op=(BlackwellIROpcode)0x7fffffff;
    rc=sim_point2(&point,0,41,(int)shape[4],1,0,msg,sizeof msg,&bad,&moved);
#else
    if(!frame_load(argv[1],0)) return 66;
    g_sim_variant=atoi(argv[7]);
    if(g_sim_variant!=1&&g_sim_variant!=2) return 65;
    Weights weights;weights_make(&weights,3,128,shape[0],41);
    uint32_t ids[3]={2,0,1},seed=42;
    size_t words=(size_t)3*shape[0];float *x=malloc(words*4),*y=malloc(3*128*4);
    if(!x||!y) return 68;
    for(size_t j=0;j<words;j++) x[j]=frand(&seed,-2.0f,2.0f);
    OmegaFp8GemvSpec spec={0};spec.k=shape[0];spec.lane_bytes=shape[1];spec.rows=shape[2];spec.warps=shape[3];spec.wfirst=shape[4];
    rc=sim_gemv(&spec,&weights,3,ids,x,3,y,2,0,msg,sizeof msg);
    double worst=rc?1.0:parity(&weights,3,ids,x,3,y,&bad);
    if(rc||bad) {fprintf(stderr,"native gemv numerical: rc%d bad%zu worst%.9g: %s\n",rc,bad,worst,msg);return 1;}
    printf("native grouped gemv both fragment orders, distinct experts/X rows, double-oracle bad%zu worst%.9g\n",bad,worst);
    if(!frame_wrong_result()) return 67;
    bad=0;rc=sim_gemv(&spec,&weights,3,ids,x,3,y,2,0,msg,sizeof msg);
    if(!rc) (void)parity(&weights,3,ids,x,3,y,&bad);
    if(rc||!bad) {fprintf(stderr,"FFMA->FADD wrong-result mutant not numerically caught: %s\n",msg);return 2;}
    printf("wrong-result mutant: double-oracle bad%zu\n",bad);
    native_program.insns[0].op=(BlackwellIROpcode)0x7fffffff;
    rc=sim_gemv(&spec,&weights,3,ids,x,3,y,2,0,msg,sizeof msg);
    weights_free(&weights);free(x);free(y);
#endif
    if(!rc||!strstr(msg,"opcode not modelled")) {fprintf(stderr,"unknown-op refusal lost: rc%d %s\n",rc,msg);return 3;}
    printf("unknown-op mutant refused: %s\n",msg);
    return 0;
}
