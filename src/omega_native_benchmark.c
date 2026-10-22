/* SPDX-License-Identifier: MIT */
#include "omega_native_benchmark.h"
#include "omega_gpu_numerical_contract.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
static int sample_valid(const OmegaBenchmarkSample *s) {
    if (!s || !s->session || !s->input_tokens || (s->cold!=0 && s->cold!=1) || (s->stamps!=0 && s->stamps!=1)) return 0;
    size_t n=strlen(s->session);
    if (!n || n>96) return 0;
    for (size_t i=0;i<n;i++) {
        char c=s->session[i];
        if (!((c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='.' || c=='_' || c=='-')) return 0;
    }
    return 1;
}
int omega_benchmark_write_graph(FILE *out, const OmegaBenchmarkSample *s, uint32_t pos, uint32_t token,
    const OmegaGpuDecodeGraphStats *st, const OmegaGpuAllocStats *before, const OmegaGpuAllocStats *after,
    const uint64_t *ts, const uint8_t *kinds, uint32_t n) {
    if (!out || !sample_valid(s) || !st || !before || !after || pos==UINT32_MAX ||
        s->input_tokens>pos+1 || !st->launches || st->launches>OMEGA_GPU_CMDLIST_MAX_LAUNCHES || !st->submits || st->sems_done!=st->launches || !st->total_ns ||
        st->record_ns>st->total_ns || st->chip_ns>st->total_ns ||
        after->allocs<before->allocs || after->alloc_bytes<before->alloc_bytes ||
        (s->stamps ? (!ts || !kinds || n!=st->launches) : n!=0)) return -1;
    /* JSON numeric measurements must be exact safe integers for the host tool. */
    const uint64_t safe=9007199254740991ULL;
    if (st->total_ns>safe || after->allocs-before->allocs>safe || after->alloc_bytes-before->alloc_bytes>safe) return -1;
    for (uint32_t i=0;s->stamps && i<=n;i++) {
        if (ts[i]==UINT64_MAX || (i && ts[i]<=ts[i-1]) || (i<n && kinds[i]>=OMEGA_GPU_DGRAPH_K_COUNT)) return -1;
    }
    if (s->stamps && ts[n]-ts[0]>safe) return -1;
    fprintf(out,"{\"schema\":\"cudapp.graph-sample.v1\",\"contract\":\"%s\",\"session\":\"%s\","
        "\"ordinal\":%u,\"phase\":\"%s\",\"stamps\":%s,\"token\":%u,"
        "\"workload\":{\"batch\":1,\"context\":%u,\"input_tokens\":%u,\"output_tokens\":1,\"seed\":%u},"
        "\"measured\":{\"total_ns\":%"PRIu64",\"decode_ns\":%"PRIu64",\"host_record_ns\":%"PRIu64","
        "\"native_completion_ns\":%"PRIu64",\"launches\":%u,\"submits\":%u,"
        "\"allocations\":%"PRIu64",\"allocation_bytes\":%"PRIu64,
        omega_numeric_contract_id(),s->session,s->ordinal,s->cold?"cold":"warm",s->stamps?"true":"false",token,
        pos+1,s->input_tokens,s->seed,st->total_ns,st->total_ns,st->record_ns,st->chip_ns,st->launches,st->submits,
        after->allocs-before->allocs,after->alloc_bytes-before->alloc_bytes);
    if (s->stamps) fprintf(out,",\"gpu_ns\":%"PRIu64,ts[n]-ts[0]);
    fprintf(out,"},\"timestamps\":[");
    for (uint32_t i=0;s->stamps && i<=n;i++) fprintf(out,"%s\"%"PRIu64"\"",i?",":"",ts[i]);
    fprintf(out,"],\"kinds\":[");
    for (uint32_t i=0;i<n;i++) fprintf(out,"%s%u",i?",":"",kinds[i]);
    fprintf(out,"]}\n");
    return ferror(out)?-1:0;
}
int omega_benchmark_graph_token(FILE *out, OmegaGpuDecodeGraph *g, const float *embedding, uint32_t pos,
    const OmegaBenchmarkSample *s, uint32_t *token) {
    if (!out || !g || !embedding || !token || !sample_valid(s) || pos==UINT32_MAX) return -1;
    uint64_t *ts=NULL; uint8_t *kinds=NULL;
    const uint32_t cap=OMEGA_GPU_CMDLIST_MAX_LAUNCHES+1;
    if (s->stamps) {
        ts=malloc((size_t)cap*sizeof *ts); kinds=malloc(cap);
        if (!ts || !kinds) {free(ts);free(kinds);return -1;}
    }
    OmegaGpuAllocStats before,after; OmegaGpuDecodeGraphStats st;
    omega_gpu_decode_graph_set_stamps(g,s->stamps);
    omega_gpu_session_alloc_stats(&before);
    int rc=omega_gpu_decode_graph_token(g,embedding,pos,token,&st);
    omega_gpu_session_alloc_stats(&after);
    int n=0;
    if (!rc && s->stamps) {n=omega_gpu_decode_graph_stamps(g,ts,kinds,cap);if(n<0)rc=-1;}
    if (!rc) rc=omega_benchmark_write_graph(out,s,pos,*token,&st,&before,&after,ts,kinds,(uint32_t)n);
    free(ts); free(kinds); return rc;
}
int omega_benchmark_write_cmdlist(FILE *out, const OmegaGpuCmdListStats *s) {
    if (!out || !s || !s->launches || !s->submits || s->launches!=s->sems_done || s->first_missing!=-1 ||
        s->elapsed_ns>9007199254740991ULL || s->host_ns>9007199254740991ULL) return -1;
    fprintf(out,"{\"schema\":\"cudapp.cmdlist-sample.v1\",\"contract\":\"%s\","
        "\"launches\":%u,\"submits\":%u,\"method_words\":%u,\"host_method_build_ns\":%"PRIu64","
        "\"native_completion_ns\":%"PRIu64"}\n",omega_numeric_contract_id(),s->launches,s->submits,s->words,s->host_ns,s->elapsed_ns);
    return ferror(out)?-1:0;
}
int omega_benchmark_write_request(FILE *out, const OmegaBenchmarkSample *s, const OmegaBenchmarkRequest *r) {
    if (!out || !sample_valid(s) || s->stamps || !r || !r->context || !r->output_tokens ||
        r->output_tokens>r->context || (uint64_t)s->input_tokens+r->output_tokens-1>r->context || r->prefill_end_ns<r->start_ns ||
        r->first_token_ns<r->prefill_end_ns || r->end_ns<r->first_token_ns ||
        r->end_ns<=r->prefill_end_ns || r->end_ns-r->start_ns>9007199254740991ULL) return -1;
    fprintf(out,"{\"schema\":\"cudapp.graph-sample.v1\",\"contract\":\"%s\",\"session\":\"%s\","
        "\"ordinal\":%u,\"phase\":\"%s\",\"stamps\":false,\"token\":%u,"
        "\"workload\":{\"batch\":1,\"context\":%u,\"input_tokens\":%u,\"output_tokens\":%u,\"seed\":%u},"
        "\"measured\":{\"ttft_ns\":%"PRIu64",\"prefill_ns\":%"PRIu64",\"decode_ns\":%"PRIu64",\"total_ns\":%"PRIu64"},"
        "\"timestamps\":[],\"kinds\":[],\"request_clock\":[\"%"PRIu64"\",\"%"PRIu64"\",\"%"PRIu64"\",\"%"PRIu64"\"]}\n",
        omega_numeric_contract_id(),s->session,s->ordinal,s->cold?"cold":"warm",r->token,r->context,s->input_tokens,r->output_tokens,s->seed,
        r->first_token_ns-r->start_ns,r->prefill_end_ns-r->start_ns,r->end_ns-r->prefill_end_ns,r->end_ns-r->start_ns,
        r->start_ns,r->prefill_end_ns,r->first_token_ns,r->end_ns);
    return ferror(out)?-1:0;
}
