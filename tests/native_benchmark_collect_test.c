/* SPDX-License-Identifier: MIT */
/* Host fixture callbacks replace only device entry points at link time.
 * The production collection wrapper and formatter run unchanged. */
#include "omega_native_benchmark.h"
#include <stdio.h>
#include <string.h>
static int stamps_on, calls, stat_calls, fail_token, fail_stamps;
void omega_gpu_decode_graph_set_stamps(OmegaGpuDecodeGraph *g,int on) {(void)g;stamps_on=on;}
void omega_gpu_session_alloc_stats(OmegaGpuAllocStats *s) {memset(s,0,sizeof *s);s->allocs=(unsigned)stat_calls++;}
int omega_gpu_decode_graph_token(OmegaGpuDecodeGraph *g,const float *x,uint32_t pos,uint32_t *token,OmegaGpuDecodeGraphStats *s) {
    (void)g;if(!x || pos!=7)return -1;calls++;
    if(fail_token)return -4;
    *token=1;*s=(OmegaGpuDecodeGraphStats){.launches=2,.submits=1,.record_ns=10,.total_ns=100,.chip_ns=80,.sems_done=2};return 0;
}
int omega_gpu_decode_graph_stamps(const OmegaGpuDecodeGraph *g,uint64_t *ts,uint8_t *k,uint32_t cap) {
    (void)g;if(fail_stamps || cap<3)return -1;
    ts[0]=100;ts[1]=120;ts[2]=150;k[0]=0;k[1]=1;return 2;
}
#define CHECK(x) do{checks++;if(!(x)){fprintf(stderr,"FAIL %u %s\n",checks,#x);return 1;}}while(0)
int main(void) {
    unsigned checks=0;uint32_t token=0,dummy=0;float x=1;FILE *f=tmpfile();CHECK(f);
    OmegaBenchmarkSample s={.session="fixture-a",.input_tokens=8,.seed=7,.stamps=1};
    CHECK(!omega_benchmark_graph_token(f,(OmegaGpuDecodeGraph *)&dummy,&x,7,&s,&token));
    CHECK(calls==1 && stat_calls==2 && stamps_on==1 && token==1);
    rewind(f);char line[2048];CHECK(fgets(line,sizeof line,f));CHECK(strstr(line,"\"gpu_ns\":50") && strstr(line,"\"allocations\":1"));
    fclose(f);f=tmpfile();CHECK(f);fail_token=1;
    CHECK(omega_benchmark_graph_token(f,(OmegaGpuDecodeGraph *)&dummy,&x,7,&s,&token)==-4);CHECK(ftell(f)==0);
    fail_token=0;fail_stamps=1;CHECK(omega_benchmark_graph_token(f,(OmegaGpuDecodeGraph *)&dummy,&x,7,&s,&token));CHECK(ftell(f)==0);
    fail_stamps=0;s.stamps=0;CHECK(!omega_benchmark_graph_token(f,(OmegaGpuDecodeGraph *)&dummy,&x,7,&s,&token));CHECK(stamps_on==0);
    fclose(f);printf("native_benchmark_collect_test: %u checks PASS (fixture device callbacks; hardware NOT_RUN)\n",checks);return 0;
}
