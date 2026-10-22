/* SPDX-License-Identifier: MIT */
#include "omega_native_benchmark.h"
#include <stdio.h>
#include <string.h>
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL %u %s\n",checks,#x);return 1;}}while(0)
int main(int argc,char **argv) {
    unsigned checks=0;
    OmegaBenchmarkSample s={.session="fixture-a",.ordinal=0,.input_tokens=8,.seed=7,.cold=0,.stamps=1};
    OmegaGpuDecodeGraphStats st={.launches=2,.submits=1,.chip_ns=80,.record_ns=10,.total_ns=100,.sems_done=2};
    OmegaGpuAllocStats before={0},after={.allocs=1,.alloc_bytes=4096};
    uint64_t stamps[]={100,120,150};uint8_t kinds[]={0,1};
    FILE *out=tmpfile();CHECK(out);
    CHECK(!omega_benchmark_write_graph(out,&s,7,1,&st,&before,&after,stamps,kinds,2));
    rewind(out);char line[4096];CHECK(fgets(line,sizeof line,out));
    CHECK(strstr(line,"\"gpu_ns\":50") && strstr(line,"\"allocation_bytes\":4096"));
    CHECK(!strstr(line,"barriers") && !strstr(line,"host_submit_ns"));
    stamps[1]=90;CHECK(omega_benchmark_write_graph(out,&s,7,1,&st,&before,&after,stamps,kinds,2));stamps[1]=120;
    stamps[2]=UINT64_MAX;CHECK(omega_benchmark_write_graph(out,&s,7,1,&st,&before,&after,stamps,kinds,2));stamps[2]=150;
    s.session="fixture\"injection";CHECK(omega_benchmark_write_graph(out,&s,7,1,&st,&before,&after,stamps,kinds,2));s.session="fixture-a";
    st.sems_done=1;CHECK(omega_benchmark_write_graph(out,&s,7,1,&st,&before,&after,stamps,kinds,2));st.sems_done=2;
    st.record_ns=101;CHECK(omega_benchmark_write_graph(out,&s,7,1,&st,&before,&after,stamps,kinds,2));st.record_ns=10;
    s.stamps=0;CHECK(omega_benchmark_write_graph(out,&s,7,1,&st,&before,&after,stamps,kinds,2));
    CHECK(!omega_benchmark_write_graph(out,&s,7,1,&st,&before,&after,NULL,NULL,0));
    CHECK(omega_benchmark_graph_token(out,NULL,NULL,0,&s,NULL));
    OmegaGpuCmdListStats cl={.launches=2,.submits=1,.words=42,.elapsed_ns=80,.host_ns=10,.sems_done=2,.first_missing=-1};
    CHECK(!omega_benchmark_write_cmdlist(out,&cl));cl.first_missing=1;CHECK(omega_benchmark_write_cmdlist(out,&cl));
    s.input_tokens=7;
    OmegaBenchmarkRequest req={.start_ns=100,.prefill_end_ns=120,.first_token_ns=130,.end_ns=200,.context=8,.output_tokens=2,.token=1};
    CHECK(!omega_benchmark_write_request(out,&s,&req));
    req.first_token_ns=110;CHECK(omega_benchmark_write_request(out,&s,&req));req.first_token_ns=130;
    req.end_ns=100;CHECK(omega_benchmark_write_request(out,&s,&req));req.end_ns=200;
    s.input_tokens=8;
    if(argc==2 && !strcmp(argv[1],"--fixture")) {
        s.stamps=1;CHECK(!omega_benchmark_write_graph(stdout,&s,7,1,&st,&before,&after,stamps,kinds,2));
    }
    fclose(out);fprintf(stderr,"native_benchmark_adapter_test: %u checks PASS (host fixtures; no graph invoked)\n",checks);return 0;
}
