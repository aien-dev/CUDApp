/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Included by the existing session-platform fixture only. These are synthetic
 * typed host handles, never GPU execution or qualification evidence. Artifact
 * admission is real; independent C wire/hash construction is a test oracle. */
#include "omega_gpu_q36_native.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_fp8gemm_api.h"
#include "omega_gpu_cmdlist.h"
#include "omega_gpu_cmdlist_native.h"
#include "sha256.h"
#include "omega_blackwell_qmd.h"

static OmegaNativeArtifactSnapshot nr_snapshot;
static void nr_u32(uint8_t *p, uint32_t v) {
    for (unsigned i=0;i<4;++i) p[i]=(uint8_t)(v>>(8*i));
}
static void nr_admit(OmegaGpuNativeArtifact *a, uint64_t launch[32]) {
    char dir[]="/tmp/cudapp-native-runtime-XXXXXX", prefix[256], path[300], id[65], actual[65], err[256];
    uint8_t abi[136]={0}, qpair[768];
    uint32_t q0[96],q1[96],cb[10]={0,0,0,0,0,0,128,0,512,128};
    OmegaCubinParam params[7];
    OmegaBlackwellQmdConfig cfg={.threads_per_block=256,.grid_width=16,
        .threads_x=256,.threads_y=1,.grid_x=16,.grid_y=1,.gpr_count=64,.shared_bytes=1024};
    omega_native_artifact_snapshot_free(&nr_snapshot);
    EXPECT(mkdtemp(dir)!=NULL);
    snprintf(prefix,sizeof prefix,"%s/artifact",dir);
    EXPECT(omega_blackwell_build_qmd0(q0,0,0)==0);
    EXPECT(omega_blackwell_build_qmd1(q1,&cfg)==0);
    nr_u32(abi,14084u|(4u<<16)); nr_u32(abi+4,130);
    for(unsigned i=0;i<7;++i) {
        uint32_t off=i<3?8*i:24+4*(i-3);
        params[i]=(OmegaCubinParam){896+off,i<3?8:4,i<3};
        nr_u32(abi+8+16*i,5892u|(12u<<16));
        nr_u32(abi+16+16*i,i|(off<<16));
        nr_u32(abi+20+16*i,i<3?2225408:1175552);
    }
    nr_u32(abi+120,6403u|(40u<<16)); nr_u32(abi+124,2564u|(8u<<16));
    nr_u32(abi+128,3); nr_u32(abi+132,896u|(40u<<16));
    sha256_hash(abi,sizeof abi,(uint8_t *)(launch+15));
    sha256_hash((uint8_t *)cb,sizeof cb,(uint8_t *)(launch+19));
    memcpy(qpair,q0,384); memcpy(qpair+384,q1,384);
    sha256_hash(qpair,sizeof qpair,(uint8_t *)(launch+23));
    char hash[65]; memset(hash,'a',64); hash[64]=0;
    OmegaNativeArtifact artifact={.op="matmul",.shape="m=1,k=256,n=128",
        .bounds="m<=1,k<=256,n<=128",.contract_name="synthetic-host-runtime",
        .contract_id_hex=hash,.semantic_spec_id_hex=hash,.kernel_function="native_fixture",
        .precision="fp8",.accumulation="fp32",.tolerance="0",
        .text=a->code,.text_len=(size_t)a->sizes[1],.params=params,.nparams=7,
        .gpr_count=64,.grid_x=16,.grid_y=1,.block_x=256,.block_y=1,.block_z=1,
        .shared_bytes=1024,.cbank_offset=896,.cbank_args=cb,.cbank_words=10,
        .qmd0=q0,.qmd1=q1,.min_driver="580.173.02",.requires="sm_121",
        .source_commit="UNKNOWN",.tool_version="host-runtime-oracle/1",
        .qualification="device-load=NOT_RUN"};
    EXPECT(omega_native_artifact_export(&artifact,prefix,id,err,sizeof err)==0);
    for(unsigned i=0;i<32;++i) {
        unsigned value=0;
        EXPECT(sscanf(id+2*i,"%2x",&value)==1); a->manifest[i]=(uint8_t)value;
    }
    EXPECT(omega_native_artifact_snapshot(prefix,"580.173.02",actual,&nr_snapshot,err,sizeof err)==0);
    a->admission=&nr_snapshot;
    static const char *const suffix[]={".text.bin",".cubin",".qmd0.bin",".qmd1.bin",
        ".cbank_args.bin",".json",".manifest",".sha256"};
    for(unsigned i=0;i<sizeof suffix/sizeof suffix[0];++i) {
        snprintf(path,sizeof path,"%s%s",prefix,suffix[i]); EXPECT(unlink(path)==0);
    }
    EXPECT(rmdir(dir)==0);
}
static void nr_word(uint8_t *p, unsigned i, uint64_t v) {
    for (unsigned j = 0; j < 8; ++j) p[i * 8 + j] = (uint8_t)(v >> (j * 8));
}
static void nr_digest(uint8_t out[32], const void *p, size_t n) { sha256_hash(p, n, out); }
static void nr_fixture(OmegaGpuNativeArtifact *a, uint8_t weights[65536], uint8_t raw_scales[1024]) {
    uint64_t d[48] = {0}, status = 0;
    uint8_t wire[384] = {0}, src[65536] = {0}, scales[1024] = {0};
    uint64_t cfg[16] = {1,256,8,1,8}, ins[8192] = {0}, metadata[2048] = {0}, phys[288] = {0}, stats[8] = {0};
    d[0]=1; d[1]=2; d[2]=8; d[4]=1; d[5]=128; d[6]=128;
    d[11]=1; d[13]=1; d[14]=8; d[15]=1; d[17]=128; d[18]=256; d[19]=2;
    d[24]=3; d[25]=256; d[26]=4; d[27]=1; d[30]=32768; d[31]=4;
    d[33]=256; d[34]=8; d[35]=4; d[36]=32768; d[37]=8; d[40]=1; d[41]=256;
    memset(a, 0, sizeof *a);
    EXPECT(omega_native_quant(&status,0,d,48,wire,384)==OMEGA_OK && status==0);
    EXPECT(omega_native_repack(&status,d,48,src,65536,raw_scales,1024,weights,65536,scales,1024)==OMEGA_OK && status==0);
    EXPECT(memcmp(weights,src,32768)==0);
    EXPECT(omega_native_compose(&status,cfg,16,ins,8192,metadata,2048,phys,288,stats,8)==OMEGA_OK && status==0);
    EXPECT(stats[0]>0 && stats[1]<=64 && stats[2]<=32 && stats[3]<=1024);
    EXPECT(omega_bc_encode(&status,ins,8192,phys,288,stats[0],1,a->code,16384)==OMEGA_OK && status>0 && status<=16384);
    a->sizes[0]=1152; a->sizes[1]=status; a->sizes[2]=256;
    uint64_t words[38] = {1,1152,121,0,1,128,256,1,3,1,1,1,0,0,1,1,256,8,1,1,7,1024,64,6,1,256,16,1,1,7,1,2,0,4,5,6,7,8};
    for (unsigned i=0;i<38;++i) nr_word(a->canonical,i,words[i]);
    memcpy(a->canonical+768,wire,384);
    /* Digest each actual test input/output rather than a digest sentinel. The
     * first four identities are synthetic fixture provenance, not receipts. */
    nr_digest(a->canonical+512,cfg,sizeof cfg);
    nr_digest(a->canonical+544,ins,sizeof ins);
    nr_digest(a->canonical+576,phys,sizeof phys);
    nr_digest(a->canonical+608,d,sizeof d);
    nr_digest(a->canonical+640,metadata,sizeof metadata);
    nr_digest(a->canonical+672,a->code,(size_t)a->sizes[1]);
    nr_digest(a->canonical+736,stats,sizeof stats);
    uint64_t launch[32] = {1,16,1,1,256,1,1,1024,64,6,1,896,10,1,1};
    /* Expected template identities are computed independently before admission.
     * The retained actual snapshot must match; manifest text is not qualification. */
    nr_admit(a,launch);
    for(unsigned i=0;i<32;++i) nr_word(a->launch,i,launch[i]);
    nr_digest(a->canonical+704,a->launch,256);
    nr_digest(a->realization,a->canonical,1152);
    uint8_t got[32];
    EXPECT(omega_native_payload(a,got)==0 && memcmp(got,a->realization,32)==0);
    EXPECT(omega_native_admitted(a,got)==0);
}
static void nr_state(OmegaGpuQ36NativeState *s, const OmegaGpuNativeArtifact *a) {
    memset(s,0,sizeof *s); s->artifact[0]=*a;
    memcpy(s->wanted,a->canonical,4096); memcpy(s->records,a->canonical,4096);
    uint64_t *q=s->query,*r=s->rows;
    q[0]=1; q[1]=1; q[3]=omega_gpu_session_epoch(); q[4]=121; q[5]=1; q[6]=1;
    q[7]=1024; q[8]=256; q[9]=32768; q[10]=1024; q[13]=100;
    for(unsigned i=0;i<6;++i) nr_digest((uint8_t *)(q+14+i*4),&i,sizeof i);
    memcpy(q+30,a->realization,32);
    r[0]=1; r[1]=31; r[2]=63; r[3]=10; r[9]=10;
    r[12]=64; r[13]=1024; r[14]=256; r[16]=omega_gpu_code_alloc_bytes((size_t)a->sizes[1]);
    memcpy(r+17,q+3,4*sizeof(uint64_t));
    memcpy(r+21,a->realization,32); memcpy(r+25,a->canonical+704,32);
    nr_digest((uint8_t *)(r+29),"SYNTHETIC-NOT-QUALIFIED",sizeof("SYNTHETIC-NOT-QUALIFIED")-1);
    memcpy(r+33,q+14,24*sizeof(uint64_t)); memcpy(r+57,a->canonical+672,32);
}
static uint64_t nr_code_va(const OmegaGpuNativeArtifact *a) {
    for(uint32_t i=0;i<NB;++i)
        if(fk.mem[i] && fk.t.live[i] && fk.desc[i].kind==OMEGA_GPU_HANDLE_BUF_CODE &&
           fk.desc[i].bytes>=a->sizes[1] && memcmp(fk.mem[i],a->code,(size_t)a->sizes[1])==0)
            return 0x100000000ull+(uint64_t)i*0x1000000ull;
    return 0;
}
static unsigned nr_qmd_binds(uint64_t va) {
    unsigned n=0;
    for(uint32_t i=0;i<NB;++i) if(fk.mem[i] && fk.t.live[i] && fk.desc[i].kind==OMEGA_GPU_HANDLE_BUF_DATA) {
        const uint32_t *p=fk.mem[i];
        for(size_t j=0;j+64<=fk.desc[i].bytes/4;j+=64)
            if(omega_blackwell_verify_qmd_invariants(p+j)==0 && p[j+32]==(uint32_t)(va>>4) && (p[j+33]&0x1fffffu)==(uint32_t)(va>>36)) ++n;
    }
    return n;
}
/* Independent legacy C bank/QMD/SHA oracle. Its labels and typed handles are
 * synthetic; no GROUP semantics, executed kernel, or qualification is claimed. */
typedef struct {
    uint8_t expected[2][32];
    uint64_t code_bytes;
    const char *label;
    unsigned calls, submits, syncs;
    int refuse;
} NrObservedIntent;
static void nr_inputs(const OmegaGpuFp8Tensor *t, OmegaGpuMem mem[3], uint32_t args[10]) {
    OmegaGpuNativeTensorObservation owner;
    EXPECT(omega_gpu_fp8_native_observe(t,&owner)==0);
    EXPECT(omega_gpu_session_alloc(16,&mem[0])==0);
    EXPECT(omega_gpu_session_alloc(1024,&mem[1])==0);
    EXPECT(omega_gpu_session_alloc(512,&mem[2])==0);
    uint64_t directory[2]={owner.weights.va,owner.scales.va};
    memcpy(mem[0].cpu,directory,sizeof directory);
    uint32_t words[10]={(uint32_t)mem[0].va,(uint32_t)(mem[0].va>>32),
        (uint32_t)mem[1].va,(uint32_t)(mem[1].va>>32),
        (uint32_t)mem[2].va,(uint32_t)(mem[2].va>>32),128,0,512,128};
    memcpy(args,words,sizeof words);
}
static void nr_freeze_intent(NrObservedIntent *intent, const OmegaGpuCmdList *cl,
    const OmegaGpuNativeArtifact *a, uint64_t code_va, const uint32_t args[10],
    OmegaGpuCmdListMode mode, const char *label) {
    OmegaGpuCmdListOwner owner;
    EXPECT(omega_gpu_cmdlist_owner(cl,&owner)==0 && owner.count==0);
    memset(intent,0,sizeof *intent);
    intent->code_bytes=a->sizes[1]; intent->label=label;
    for(unsigned i=0;i<2;++i) {
        uint8_t bank[1024]={0},qmd[1024]={0},wire[384+1024+768]={0};
        uint32_t drv[OMEGA_BW_CBANK_DRIVER_WORDS],q0[96],q1[96];
        uint64_t bank_va=owner.banks.va+(uint64_t)i*1024;
        uint64_t qmd_va=owner.qmds.va+(uint64_t)i*1024;
        omega_blackwell_build_cbank_driver_2d(drv,bank_va,256,1,16,1);
        memcpy(bank,drv,sizeof drv); memcpy(bank+896,args,10*sizeof *args);
        OmegaBlackwellQmdConfig cfg={.code_va=code_va,.cbank_va=bank_va,
            .scratch_va=owner.scratch.va,.sem_va=owner.semaphores.va+(uint64_t)i*16,
            .qmd0_va=qmd_va+512,.qmd1_va=qmd_va,.num_elements=128,
            .threads_per_block=256,.grid_width=16,.threads_x=256,.threads_y=1,
            .grid_x=16,.grid_y=1,.gpr_count=64,.shared_bytes=1024};
        EXPECT(omega_blackwell_build_qmd1(q1,&cfg)==0);
        EXPECT(omega_blackwell_build_qmd0(q0,qmd_va+512,qmd_va)==0);
        q1[10]=mode==OMEGA_GPU_CMDLIST_INLINE?0x00190000u:0;
        q1[12]=mode==OMEGA_GPU_CMDLIST_INLINE?(uint32_t)((qmd_va+512)>>8):0;
        memcpy(qmd,q1,sizeof q1); memcpy(qmd+512,q0,sizeof q0);
        uint64_t meta[32]={1,159,1,0,10,a->sizes[1],i,owner.record_generation,0,0,code_va};
        uint64_t submit[16]={1,159,4,owner.record_generation,mode,0,1,1,0,0,0,2};
        for(unsigned j=0;j<32;++j) nr_word(wire,j,meta[j]);
        for(unsigned j=0;j<16;++j) nr_word(wire,32+j,submit[j]);
        size_t off=384;
        memcpy(wire+off,bank,896+10*sizeof *args); off+=896+10*sizeof *args;
        memcpy(wire+off,qmd,384); off+=384;
        memcpy(wire+off,qmd+512,384); off+=384;
        nr_digest(intent->expected[i],wire,off);
    }
}
static int nr_observe_actual(const OmegaGpuCmdList *cl, const OmegaGpuCmdListConsumed *consumed, void *data) {
    NrObservedIntent *intent=data; OmegaGpuCmdListOwner owner;
    ++intent->calls;
    EXPECT(intent->calls==1 && fk.submits==intent->submits && fk.syncs==intent->syncs);
    if(omega_gpu_cmdlist_owner(cl,&owner)!=0 || owner.count!=2) return -1;
    for(unsigned i=0;i<owner.count;++i) {
        OmegaGpuCmdListEntry entry;
        if(omega_gpu_cmdlist_entry(cl,i,&entry)!=0) return -1;
        uint64_t meta[32]={1,159,1,0,entry.n_args,intent->code_bytes,entry.ordinal,
            entry.record_generation,entry.independent,entry.pending_wait,entry.code_va};
        uint64_t submit[16]={1,159,4,owner.record_generation,consumed->options.mode,
            consumed->options.barrier,consumed->timeout_ms,consumed->spin_us,
            (uint64_t)consumed->options.first_barrier_none,(uint64_t)consumed->options.skip_launch,
            (uint64_t)consumed->options.stamps,owner.count};
        uint8_t got[32]={0}; uint64_t result=0;
        if(omega_native_mclaunch(&result,meta,32,entry.bank,1024,entry.qmd_slot,1024,
            submit,16,got,32)!=OMEGA_OK || result!=0 ||
            memcmp(got,intent->expected[i],32)!=0) return -1;
        const char *dir=getenv("OMEGA_NATIVE_OBSERVER_DIR");
        if(dir && *dir) {
            uint8_t wire[32+256+128+1024+1024]={0}; char path[1024];
            memcpy(wire,intent->expected[i],32);
            for(unsigned j=0;j<32;++j) nr_word(wire+32,j,meta[j]);
            for(unsigned j=0;j<16;++j) nr_word(wire+32+256,j,submit[j]);
            memcpy(wire+32+256+128,entry.bank,1024);
            memcpy(wire+32+256+128+1024,entry.qmd_slot,1024);
            int n=snprintf(path,sizeof path,"%s/%s-%u.wire",dir,intent->label,i);
            EXPECT(n>0 && (size_t)n<sizeof path);
            FILE *f=fopen(path,"wb"); EXPECT(f!=NULL);
            if(!f) return -1;
            EXPECT(fwrite(wire,1,sizeof wire,f)==sizeof wire);
            EXPECT(fclose(f)==0);
        }
    }
    return 0;
}
static void nr_submit_observed(OmegaGpuCmdList *cl, NrObservedIntent *intent, OmegaGpuCmdListMode mode) {
    OmegaGpuCmdListOpts opts={.mode=mode,.timeout_ms=1,.spin_us=1};
    OmegaGpuCmdListStats stats={0};
    intent->submits=fk.submits; intent->syncs=fk.syncs;
    EXPECT(omega_gpu_cmdlist_submit_observed(cl,&opts,&stats,nr_observe_actual,intent)==
        (intent->refuse?-1:0));
    EXPECT(intent->calls==1);
    if(intent->refuse) EXPECT(fk.submits==intent->submits && fk.syncs==intent->syncs);
    else EXPECT(stats.launches==2 && stats.sems_done==2 && fk.submits>intent->submits);
}
static void c_native_observer_modes(void) {
    fk_reset(); EXPECT(install()==0);
    OmegaGpuNativeArtifact a; uint8_t w[65536]={0},sc[1024]={0}; nr_fixture(&a,w,sc);
    OmegaGpuFp8Tensor *t=NULL; OmegaGpuCmdList *cl[3]={NULL,NULL,NULL};
    EXPECT(omega_gpu_fp8_tensor_upload(1,128,256,w,(const uint16_t *)sc,&t)==0);
    for(unsigned i=0;i<3;++i) EXPECT(omega_gpu_cmdlist_create(2,&cl[i])==0);
    omega_gpu_session_lock();
    OmegaGpuMem mem[3]={0}; uint32_t args[10]; nr_inputs(t,mem,args);
    OmegaGpuNativeLease lease={0}; EXPECT(omega_gpu_fp8_prepare_native(&a,&lease)==0);
    uint64_t code_va=nr_code_va(&a); EXPECT(code_va!=0);
    NrObservedIntent pcas,reuse,fresh,refused;
    nr_freeze_intent(&pcas,cl[0],&a,code_va,args,OMEGA_GPU_CMDLIST_PCAS,"fresh-pcas");
    nr_freeze_intent(&reuse,cl[0],&a,code_va,args,OMEGA_GPU_CMDLIST_INLINE,"reuse-inline");
    nr_freeze_intent(&fresh,cl[1],&a,code_va,args,OMEGA_GPU_CMDLIST_INLINE,"fresh-inline");
    nr_freeze_intent(&refused,cl[2],&a,code_va,args,OMEGA_GPU_CMDLIST_PCAS,"refused");
    refused.refuse=1;
    for(unsigned i=0;i<3;++i) for(unsigned j=0;j<2;++j)
        EXPECT(omega_gpu_fp8_record_native(cl[i],t,&a,&lease,1,mem[0].va,mem[1].va,1,mem[2].va)==(int)j);
    OmegaGpuCmdListOwner bad_owner;
    EXPECT(omega_gpu_cmdlist_owner(cl[2],&bad_owner)==0);
    ((uint8_t *)bad_owner.banks.cpu)[896]^=1; /* actual post-record argument corruption */
    nr_submit_observed(cl[0],&pcas,OMEGA_GPU_CMDLIST_PCAS);
    nr_submit_observed(cl[0],&reuse,OMEGA_GPU_CMDLIST_INLINE);
    nr_submit_observed(cl[1],&fresh,OMEGA_GPU_CMDLIST_INLINE);
    nr_submit_observed(cl[2],&refused,OMEGA_GPU_CMDLIST_PCAS);
    for(unsigned i=0;i<3;++i) omega_gpu_cmdlist_reset(cl[i]);
    EXPECT(omega_native_lease_op(&lease,5,0)==0); omega_gpu_fp8_cache_clear();
    for(unsigned i=0;i<3;++i) omega_gpu_session_free(&mem[i]);
    omega_gpu_session_unlock();
    for(unsigned i=0;i<3;++i) omega_gpu_cmdlist_free(cl[i]);
    omega_gpu_fp8_tensor_free(t); omega_gpu_session_close();
    omega_native_artifact_snapshot_free(&nr_snapshot);
    printf("HOST_ONLY actual typed-owner mode-final wires; role0=SYNTHETIC_LABEL; kernel/GPU/qualification=NOT_RUN\n");
}
static void c_native_chain(void) {
    fk_reset(); EXPECT(install()==0);
    OmegaGpuNativeArtifact a; uint8_t weights[65536]={0}, raw_scales[1024]={0};
    nr_fixture(&a,weights,raw_scales);
    printf("HOST_ONLY synthetic encoded_code_bytes=%llu encoded_code_sha256=", (unsigned long long)a.sizes[1]);
    for (unsigned i=0;i<32;++i) printf("%02x",a.canonical[672+i]);
    printf(" qualification=NOT_RUN\n");
    OmegaGpuFp8Tensor *t=NULL; OmegaGpuCmdList *cl=NULL;
    EXPECT(omega_gpu_fp8_tensor_upload(1,128,256,weights,(const uint16_t *)raw_scales,&t)==0);
    EXPECT(t!=NULL && omega_gpu_cmdlist_create(4,&cl)==0 && cl!=NULL);
    if(!t || !cl) return;
    omega_gpu_session_lock();
    OmegaGpuMem inputs[3]={0}; uint32_t input_args[10]; nr_inputs(t,inputs,input_args);
    OmegaGpuNativeLease prepared={0}; EXPECT(omega_gpu_fp8_prepare_native(&a,&prepared)==0);
    EXPECT(omega_native_bound(&a,&prepared)==0);
    uint64_t va=nr_code_va(&a); EXPECT(va!=0);
    OmegaGpuQ36NativeState *s=calloc(1,sizeof *s); EXPECT(s!=NULL);
    if(!s) { omega_gpu_session_unlock(); return; }
    nr_state(s,&a);
    const OmegaGpuNativeArtifact *chosen=NULL; OmegaGpuNativeLease selected={0};
    EXPECT(omega_q36_native_begin(NULL,prepared.cache,&chosen,&selected)==1);
    s->rows[1]=30;
    EXPECT(omega_q36_native_begin(s,prepared.cache,&chosen,&selected)<0);
    s->rows[1]=31; s->rows[21]^=1;
    EXPECT(omega_q36_native_begin(s,prepared.cache,&chosen,&selected)<0);
    s->rows[21]^=1;
    EXPECT(omega_q36_native_begin(s,prepared.cache,&chosen,&selected)==0);
    EXPECT(chosen==&s->artifact[0] && selected.slot==prepared.slot && selected.revision==prepared.revision);
    OmegaGpuNativeArtifact bad=a; bad.code[0]^=1;
    unsigned allocs=fk.allocs; OmegaGpuNativeLease refused={0};
    EXPECT(omega_gpu_fp8_prepare_native(&bad,&refused)!=0 && fk.allocs==allocs);
    bad=a; bad.admission=NULL;
    EXPECT(omega_gpu_fp8_prepare_native(&bad,&refused)!=0 && fk.allocs==allocs);
    bad=a; bad.manifest[0]^=1;
    EXPECT(omega_gpu_fp8_prepare_native(&bad,&refused)!=0 && fk.allocs==allocs);
    const unsigned mutated[]={OMEGA_SNAPSHOT_QMD1,OMEGA_SNAPSHOT_CBANK,
        OMEGA_SNAPSHOT_TEXT,OMEGA_SNAPSHOT_CUBIN};
    for(unsigned i=0;i<sizeof mutated/sizeof mutated[0];++i) {
        nr_snapshot.bytes[mutated[i]][0]^=1;
        EXPECT(omega_gpu_fp8_prepare_native(&a,&refused)!=0 && fk.allocs==allocs);
        nr_snapshot.bytes[mutated[i]][0]^=1;
    }
    OmegaGpuNativeLease wrong=selected; wrong.epoch++;
    EXPECT(omega_gpu_fp8_record_native(cl,t,chosen,&wrong,1,inputs[0].va,inputs[1].va,1,inputs[2].va)<0);
    EXPECT(omega_gpu_cmdlist_count(cl)==0);
    OmegaGpuNativeTensorObservation owner; EXPECT(omega_gpu_fp8_native_observe(t,&owner)==0);
    uint64_t actual_call[12] = {1,1,128,256,1,10,0,1,
        owner.allocation_epoch,omega_gpu_session_epoch(),owner.owner_generation,0}, actual_launch[32] = {0};
    EXPECT(omega_native_bound(chosen,&selected)==0);
    EXPECT(omega_native_launch(chosen,actual_call,actual_launch)==0);
    actual_call[3] = 128;
    EXPECT(omega_native_launch(chosen,actual_call,actual_launch)!=0);
    actual_call[3] = 256;
    actual_call[8]^=1; EXPECT(omega_native_launch(chosen,actual_call,actual_launch)!=0);
    for(unsigned i=0;i<32;++i) EXPECT(actual_launch[i]==0);
    actual_call[8]^=1; actual_call[10]=0;
    EXPECT(omega_native_launch(chosen,actual_call,actual_launch)!=0);
    actual_call[10]=owner.owner_generation; actual_call[11]=1;
    EXPECT(omega_native_launch(chosen,actual_call,actual_launch)!=0);
    actual_call[11]=0;
    EXPECT(omega_gpu_fp8_record_native(cl,t,chosen,&selected,1,inputs[0].va,inputs[1].va,1,inputs[2].va)==0);
    EXPECT(nr_qmd_binds(va)>0);
    omega_q36_native_end(&selected);
    size_t b=16+prepared.slot*16;
    EXPECT(prepared.cache->pool[b+3]==1 && prepared.cache->pool[b+4]==0 && prepared.cache->pool[b+5]==1);
    omega_gpu_fp8_cache_clear(); EXPECT(nr_code_va(&a)==va);
    OmegaGpuCmdListOpts opts={.timeout_ms=1}; OmegaGpuCmdListStats st={0};
    EXPECT(omega_gpu_cmdlist_submit(cl,&opts,&st)==0);
    EXPECT(st.launches==1 && st.sems_done==1 && fk.submits>0);
    EXPECT(prepared.cache->pool[b+3]==1 && prepared.cache->pool[b+4]==0);
    omega_gpu_cmdlist_reset(cl); EXPECT(omega_gpu_cmdlist_count(cl)==0);
    EXPECT(prepared.cache->pool[b+3]==0 && omega_native_lease_op(&prepared,5,0)==0);
    omega_gpu_fp8_cache_clear(); EXPECT(nr_code_va(&a)==0);
    for(unsigned i=0;i<3;++i) omega_gpu_session_free(&inputs[i]);
    free(s); omega_gpu_session_unlock();
    omega_gpu_cmdlist_free(cl); omega_gpu_fp8_tensor_free(t); omega_gpu_session_close();
    omega_native_artifact_snapshot_free(&nr_snapshot);
}
static void c_native_timeout_refs(void) {
    fk_reset(); EXPECT(install()==0);
    OmegaGpuNativeArtifact a; uint8_t w[65536]={0}, sc[1024]={0}; nr_fixture(&a,w,sc);
    OmegaGpuCmdList *cl=NULL; EXPECT(omega_gpu_cmdlist_create(1,&cl)==0);
    omega_gpu_session_lock(); OmegaGpuNativeLease lease={0};
    EXPECT(omega_gpu_fp8_prepare_native(&a,&lease)==0);
    uint64_t va=nr_code_va(&a), epoch=omega_gpu_session_epoch(); EXPECT(va!=0);
    OmegaGpuLaunch L=launch_desc(1); L.code_va=va;
    EXPECT(omega_gpu_cmdlist_add_native(cl,&L,&lease)==0);
    EXPECT(omega_native_lease_op(&lease,5,0)==0);
    fk.mode=MODE_HANG; OmegaGpuCmdListOpts o={.timeout_ms=1}; OmegaGpuCmdListStats st={0};
    EXPECT(omega_gpu_cmdlist_submit(cl,&o,&st)<0);
    size_t b=16+lease.slot*16;
    EXPECT(lease.cache->pool[b+3]==1 && lease.cache->pool[b+4]==1);
    omega_gpu_cmdlist_reset(cl); EXPECT(omega_gpu_cmdlist_count(cl)==1);
    omega_gpu_fp8_cache_clear(); EXPECT(nr_code_va(&a)==va && omega_gpu_session_epoch()==epoch);
    omega_gpu_session_unlock(); omega_gpu_cmdlist_free(cl);
    EXPECT(nr_code_va(&a)==va && lease.cache->pool[b+4]==1);
    EXPECT(omega_gpu_session_recover()==0 && omega_gpu_session_epoch()>epoch);
    fk.mode=MODE_COMPLETE; omega_gpu_session_lock();
    EXPECT(omega_native_bound(&a,&lease)!=0);
    OmegaGpuNativeLease fresh={0}; EXPECT(omega_gpu_fp8_prepare_native(&a,&fresh)==0);
    EXPECT(fresh.epoch!=lease.epoch && fresh.cache->pool[b+4]==0);
    EXPECT(omega_native_lease_op(&fresh,5,0)==0); omega_gpu_fp8_cache_clear();
    omega_gpu_session_unlock(); omega_gpu_session_close();
    omega_native_artifact_snapshot_free(&nr_snapshot);
}
static void c_native_close_epoch(void) {
    fk_reset(); EXPECT(install()==0);
    OmegaGpuNativeArtifact a; uint8_t w[65536]={0},sc[1024]={0}; nr_fixture(&a,w,sc);
    omega_gpu_session_lock(); OmegaGpuNativeLease old={0};
    EXPECT(omega_gpu_fp8_prepare_native(&a,&old)==0); uint64_t epoch=old.epoch;
    omega_gpu_session_unlock(); omega_gpu_session_close();
    EXPECT(omega_gpu_session_epoch()>epoch);
    omega_gpu_session_lock(); EXPECT(omega_native_bound(&a,&old)!=0);
    OmegaGpuNativeLease fresh={0}; EXPECT(omega_gpu_fp8_prepare_native(&a,&fresh)==0);
    EXPECT(fresh.epoch>epoch && omega_native_bound(&a,&fresh)==0);
    EXPECT(omega_native_lease_op(&fresh,5,0)==0); omega_gpu_fp8_cache_clear();
    omega_gpu_session_unlock(); omega_gpu_session_close();
    omega_native_artifact_snapshot_free(&nr_snapshot);
}
