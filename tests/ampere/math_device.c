/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Device-only verifier: full binary32 input sweep for the four emitted MUFU functions.
 * Independent host oracle: system long-double libm, not another CUDA++ kernel.
 * Bounds: NVIDIA PTX ISA 9.4 sections 9.7.3.13,16,20,21. This checks intended
 * non-FTZ PTX-level numerical contracts; raw MUFU equivalence is NOT yet qualified.
 * No GPU is opened without the explicit --device flag. Shards are never exhaustive PASS. */
#include "omega_ampere_codegen.h"
#include "omega_gpu_session.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BATCH 65536u
#define NOREG (-1)
#define FULL_DOMAIN UINT64_C(0x100000000)
#define CTRL(stall,wbar,rbar,wait) \
    (((uint32_t)(stall)<<9)|((uint32_t)(wbar)<<14)|((uint32_t)(rbar)<<17)|((uint32_t)(wait)<<20))
#define ALU CTRL(15,7,7,0)
#define LOAD_CTRL (CTRL(4,0,7,0)|0x2000u)
static uint32_t bits(float x) { uint32_t u; memcpy(&u,&x,sizeof u); return u; }
static float number(uint32_t u) { float x; memcpy(&x,&u,sizeof x); return x; }
static int append(BlackwellIRProgram *p, BlackwellIROpcode op, int d, int a, int b, int c, uint32_t imm, uint32_t control, int uniform, int pred) {
    BlackwellIRInsn i = {0};
    i.op=op; i.dst_vreg=d; i.src1_vreg=a; i.src2_vreg=b; i.src3_vreg=c; i.ureg=NOREG;
    i.imm=imm; i.control=control; i.is_uniform=uniform; i.predicate_p0=pred;
    return omega_bw_ir_append(p,&i)<0 ? -1 : 0;
}
static int kernel(BlackwellIRProgram *p, BlackwellIROpcode math) {
    omega_bw_ir_init(p);
    int src=omega_bw_ir_alloc_vreg64(p), dst=omega_bw_ir_alloc_vreg64(p);
    int tid=omega_bw_ir_alloc_vreg(p), cta=omega_bw_ir_alloc_vreg(p), n=omega_bw_ir_alloc_vreg(p);
    int i=omega_bw_ir_alloc_vreg(p), address=omega_bw_ir_alloc_vreg64(p);
    int x=omega_bw_ir_alloc_vreg(p), y=omega_bw_ir_alloc_vreg(p);
    /* Conservative scheduling: no dependent instruction is issued before its load barrier.
     * Each producer allocates barrier zero; the next instruction waits on barrier zero. */
    uint32_t pending=0;
#define EMIT(OP,D,A,B,C,IMM,BASE,PRED) do { if(append(p,OP,D,A,B,C,IMM,(BASE)|(pending<<20),0,PRED))return -1; pending=0; } while(0)
#define LOAD(OP,D,A,IMM) do { EMIT(OP,D,A,NOREG,NOREG,IMM,LOAD_CTRL,0); pending=1u; } while(0)
    LOAD(BW_IR_LDC64,src,NOREG,0x380);
    LOAD(BW_IR_LDC64,dst,NOREG,0x388);
    LOAD(BW_IR_LDC,n,NOREG,0x390);
    LOAD(BW_IR_S2R,tid,NOREG,BW_SR_TID_X);
    LOAD(BW_IR_S2R,cta,NOREG,BW_SR_CTAID_X);
    EMIT(BW_IR_IMAD,i,cta,NOREG,tid,256,ALU,0);
    EMIT(BW_IR_ISETP_GE_U32,NOREG,i,n,NOREG,0,ALU,0);
    EMIT(BW_IR_EXIT,NOREG,NOREG,NOREG,NOREG,0,ALU,1);
    EMIT(BW_IR_IMAD_WIDE,address,i,NOREG,src,4,ALU,0);
    LOAD(BW_IR_LDG_E,x,address,0);
    /* MUFU completes through a scoreboard, not a fixed arithmetic delay. */
    EMIT(math,y,x,NOREG,NOREG,0,LOAD_CTRL,0); pending=1u;
    EMIT(BW_IR_IMAD_WIDE,address,i,NOREG,dst,4,ALU,0);
    EMIT(BW_IR_STG_E,NOREG,address,y,NOREG,0,CTRL(4,7,1,0)|0x2000u,0); pending=2u;
    EMIT(BW_IR_EXIT,NOREG,NOREG,NOREG,NOREG,0,ALU,0);
    EMIT(BW_IR_BRA,NOREG,NOREG,NOREG,NOREG,0,0x000fc000u,0);
#undef EMIT
#undef LOAD
    return omega_bw_regalloc_solve(p);
}
static int agrees(unsigned op,uint32_t input,uint32_t output) {
    float x=number(input), got=number(output);
    long double ref;
    switch(op) {
    case 0: ref=1.0L/(long double)x; break;
    case 1: ref=1.0L/sqrtl((long double)x); break;
    case 2: ref=exp2l((long double)x); break;
    default: ref=log2l((long double)x); break;
    }
    if (isnan(ref)) return isnan(got);
    float rounded=(float)ref;
    if (isinf(rounded)||rounded==0.0f) return output==bits(rounded);
    if (!isfinite(got)) return 0;
    if (op==0||op==2) {
        uint32_t want=bits(rounded);
        if ((output>>31)!=(want>>31)) return 0;
        uint32_t distance=output>want?output-want:want-output;
        return distance <= (op==0?1u:2u);
    }
    long double error=fabsl((long double)got-ref);
    if(op==1) return error<=fabsl(ref)*exp2l(-22.9L);
    if(x>0.5f && x<2.0f) return error<=0x1p-22L;
    return error<=fabsl(ref)*0x1p-22L;
}
static int parse_u64(const char *s,uint64_t *v) {
    char *end; errno=0; *v=strtoull(s,&end,0);
    return errno||!s[0]||s[0]=='-'||*end ? -1:0;
}
int main(int argc,char **argv) {
    uint64_t start=0,count=FULL_DOMAIN; int device=0,inject=0,rc=1,emit=0,selftest=0;
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--device")) device=1;
        else if(!strcmp(argv[i],"--inject-output-error")) inject=1;
        else if(!strcmp(argv[i],"--emit-only")) emit=1;
        else if(!strcmp(argv[i],"--self-test")) selftest=1;
        else if(!strcmp(argv[i],"--start")&&i+1<argc) { if(parse_u64(argv[++i],&start))return 64; }
        else if(!strcmp(argv[i],"--count")&&i+1<argc) { if(parse_u64(argv[++i],&count))return 64; }
        else return 64;
    }
    if(start>=FULL_DOMAIN||!count||count>FULL_DOMAIN-start)return 64;
    static const BlackwellIROpcode ops[]={BW_IR_MUFU_RCP,BW_IR_MUFU_RSQ,BW_IR_MUFU_EX2,BW_IR_MUFU_LG2};
    static const char *names[]={"rcp","rsqrt","exp2","log2"};
    if (selftest) {
        for(unsigned op=0;op<4;op++) {
            uint32_t correct=op==3?0xff800000u:op==2?0x3f800000u:0x7f800000u;
            if(!agrees(op,0,correct)||agrees(op,0,correct^0x80000000u))return 1;
            if(!agrees(op,0x7fc00000u,0x7fc12345u)||agrees(op,0x7fc00000u,0x3f800000u))return 1;
        }
        puts("AMPERE MATH HOST ORACLE CONTROLS PASS; DEVICE NOT_RUN");return 0;
    }
    if (emit) {
        for(unsigned op=0;op<4;op++) {
            BlackwellIRProgram program; uint8_t code[16384];size_t written=0;
            if(kernel(&program,ops[op])||omega_bw_encode_program(&program,code,sizeof code,&written)||
               omega_gpu_encode_program_target(&program,86,code,sizeof code,&written))return 1;
        }
        puts("AMPERE MATH KERNEL EMISSION PASS; DEVICE NOT_RUN");return 0;
    }
    if(!device) { puts("AMPERE MATH DEVICE NOT_RUN (explicit --device required)");return 2; }
    if(inject&&(start!=0||count!=BATCH))return 64;
    OmegaGpuMem in={0},out={0},code={0};
    omega_gpu_session_lock();
    if(!omega_gpu_session_open()||omega_gpu_session_target_sm()!=86) {
        fprintf(stderr,"AMPERE MATH NOT_RUN: target/open: %s\n",omega_gpu_session_last_error());rc=2;goto done;
    }
    if(omega_gpu_session_alloc(BATCH*sizeof(uint32_t),&in)||omega_gpu_session_alloc(BATCH*sizeof(uint32_t),&out))goto done;
    for(unsigned op=0;op<sizeof ops/sizeof ops[0];op++) {
        BlackwellIRProgram program;
        if(kernel(&program,ops[op]))goto done;
        size_t capacity=omega_bw_program_code_bytes(&program),written=0;
        if(omega_gpu_session_alloc_code(capacity,&code))goto done;
        if(omega_gpu_encode_program_target(&program,86,code.cpu,capacity,&written)||written!=capacity||omega_gpu_session_sync_to_gpu(&code))goto done;
        for(uint64_t offset=0;offset<count;) {
            uint32_t n=count-offset>BATCH?BATCH:(uint32_t)(count-offset);
            for(uint32_t j=0;j<n;j++) ((uint32_t *)in.cpu)[j]=(uint32_t)(start+offset+j);
            memset(out.cpu,0xa5,BATCH*sizeof(uint32_t));
            if(omega_gpu_session_sync_to_gpu(&in)||omega_gpu_session_sync_to_gpu(&out))goto done;
            uint32_t args[32]={(uint32_t)in.va,(uint32_t)(in.va>>32),(uint32_t)out.va,(uint32_t)(out.va>>32),n};
            OmegaGpuLaunch launch={.code_va=code.va,.gpr_count=64,.threads_x=256,.threads_y=1,.grid_x=(n+255)/256,.grid_y=1,
                .num_elements=n,.args=args,.n_args=32,.timeout_ms=10000,.target_sm=86};
            if(omega_gpu_session_launch(&launch,NULL,NULL)||omega_gpu_session_sync_to_cpu(&out))goto done;
            if(inject&&offset==0)((uint32_t *)out.cpu)[0]^=0x80000000u;
            for(uint32_t j=0;j<n;j++)if(!agrees(op,((uint32_t *)in.cpu)[j],((uint32_t *)out.cpu)[j])) {
                fprintf(stderr,"AMPERE MATH FAIL op=%s input=%08"PRIx32" output=%08"PRIx32"\n",names[op],((uint32_t *)in.cpu)[j],((uint32_t *)out.cpu)[j]);goto done;
            }
            offset+=n;
        }
        printf("AMPERE MATH %s %s start=%"PRIu64" count=%"PRIu64"\n",names[op],count==FULL_DOMAIN?"EXHAUSTIVE PASS":"SHARD PASS",start,count);
        omega_gpu_session_free(&code); memset(&code,0,sizeof code);
    }
    rc=0;
done:
    if(code.cpu)omega_gpu_session_free(&code);
    if(out.cpu)omega_gpu_session_free(&out);
    if(in.cpu)omega_gpu_session_free(&in);
    if(rc==1)fprintf(stderr,"AMPERE MATH FAIL stage=%s\n",omega_gpu_session_last_error());
    omega_gpu_session_unlock();omega_gpu_session_close();return rc;
}
