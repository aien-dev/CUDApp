/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Native ABI caller and unchanged-incumbent oracle comparison. Test logic's
 * independent rank verifier lives in moe_verify.om; this is not production. */
#include "cudapp_native.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
uint32_t moe_incumbent(uint32_t, const uint32_t *, uint32_t *, uint32_t *, uint32_t *);
static uint64_t cfg[8], meta[8], req[16];
static uint64_t ids[1024], work[10368], seg[1028], maps[2320];
static uint32_t words[16], oi[1024], ox[1152], oy[1152], os[1028];
static uint64_t dirs[514];
static uint8_t mapwire[9280], segwire[8224], idwire[4096];
static uint64_t window_cfg[8], windows[1024], window_meta[4];
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while (0)
static uint64_t plan(void) {
    uint64_t r = 99;
    CHECK(omega_moe_plan(&r,cfg,8,ids,1024,work,10368,seg,1028,maps,2320,meta,8)==0);
    return r;
}
static uint64_t verify(void) {
    uint64_t r = 99;
    CHECK(omega_moe_verify(&r,cfg,8,ids,1024,work,10368,seg,1028,maps,2320,meta,8)==0);
    return r;
}
static void setup(uint32_t n, uint32_t slots) {
    uint64_t c[8] = {1,n,slots,256,256,1152,257,1160}; memcpy(cfg,c,sizeof c);
    memset(ids,0,sizeof ids); memset(work,0x5a,sizeof work); memset(seg,0x5a,sizeof seg);
    memset(maps,0x5a,sizeof maps); memset(meta,0x5a,sizeof meta);
}
static void dense(uint32_t n, unsigned mode) {
    setup(n,8);
    for (uint32_t t=0;t<n;t++) for (uint32_t s=0;s<8;s++) {
        uint32_t v=0;
        switch(mode) {
        case 0: v=7; break; /* all same, duplicate picks */
        case 1: v=(t*8+s)%256; break; /* all distinct where possible */
        case 2: v=s%3; break; /* duplicate within token */
        case 3: v=t%2 ? 255 : s; break; /* uneven occupancy */
        default: v=(t*17+s*13)%256; break; /* ragged expert segment sizes */
        }
        ids[t*8+s]=v;
    }
    CHECK(plan()==0); CHECK(verify()==0);
    uint64_t r=99;
    memset(mapwire,0x5a,sizeof mapwire); memset(segwire,0x5a,sizeof segwire);
    CHECK(omega_moe_maps32(&r,meta,8,maps,2320,mapwire,9280)==0 && r==0);
    for (uint32_t j=0;j<257;j++) { dirs[j*2]=(uint64_t)j*256+4096; dirs[j*2+1]=dirs[j*2]+128; }
    CHECK(omega_moe_segs32(&r,seg,1028,meta,8,dirs,514,segwire,8224)==0 && r==0);
    for (uint32_t j=0;j<n*8;j++) oi[j]=(uint32_t)ids[j];
    uint32_t ns=moe_incumbent(n,oi,ox,oy,os);
    CHECK(meta[1]==ns);
    for (uint32_t j=0;j<n*9;j++) { CHECK(maps[j]==ox[j]); CHECK(maps[1160+j]==oy[j]); }
    for (uint32_t j=0;j<ns*4;j++) CHECK(seg[j]==os[j]);
    CHECK(!memcmp(mapwire,ox,n*9*sizeof *ox));
    CHECK(!memcmp(mapwire+1160*4,oy,n*9*sizeof *oy));
    for (uint32_t z=0;z<ns;z++) {
        uint64_t addr[2], reserved; uint32_t range[2];
        memcpy(addr,segwire+z*32,16); memcpy(range,segwire+z*32+16,8); memcpy(&reserved,segwire+z*32+24,8);
        CHECK(addr[0]==dirs[os[z*4]*2] && addr[1]==dirs[os[z*4]*2+1]);
        CHECK(range[0]==os[z*4+1] && range[1]==os[z*4+2] && reserved==0);
    }
    /* Independent GROUP format oracle: rank (expert, original contribution row),
       including shared. This is a format check, NOT execution of GPU instructions. */
    if (n*9<=128) for (uint32_t t=0;t<n;t++) for (uint32_t s=0;s<9;s++) {
        uint32_t e=s==8?256:ids[t*8+s], rank=0;
        for (uint32_t u=0;u<n;u++) for (uint32_t v=0;v<9;v++) {
            uint32_t f=v==8?256:ids[u*8+v];
            if (f<e || (f==e && u*9+v<t*9+s)) rank++;
        }
        CHECK(maps[rank]==t); CHECK(maps[1160+rank]==t*9+s);
    }
    if (n) {
        uint32_t v=work[8]; work[8]=v+1; CHECK(verify()!=0); work[8]=v;
        uint32_t p=(uint32_t)meta[2]*9; v=work[p]; work[p]=0; CHECK(verify()!=0); work[p]=v;
        v=maps[1160]; maps[1160]=UINT32_MAX; CHECK(verify()!=0); maps[1160]=v;
        v=seg[2]; seg[2]=0; CHECK(verify()!=0); seg[2]=v;
        CHECK(verify()==0);
    }
}
static void refusal(uint64_t expected) {
    uint64_t w[10368],s[1028],p[2320],m[8];
    memcpy(w,work,sizeof w); memcpy(s,seg,sizeof s); memcpy(p,maps,sizeof p); memcpy(m,meta,sizeof m);
    CHECK(plan()==expected); CHECK(!memcmp(w,work,sizeof w)); CHECK(!memcmp(s,seg,sizeof s));
    CHECK(!memcmp(p,maps,sizeof p)); CHECK(!memcmp(m,meta,sizeof m));
}
static uint64_t group(void) {
    uint64_t r=99; CHECK(omega_moe_group(&r,req,16,(uint8_t *)words,64)==0); return r;
}
int main(void) {
    const uint32_t sizes[]={0,1,2,3,7,14,15,31,127,128};
    for (unsigned j=0;j<sizeof sizes/sizeof *sizes;j++) for (unsigned k=0;k<5;k++) dense(sizes[j],k);
    setup(3,8); for (unsigned j=0;j<24;j++) ids[j]=j%4 ? UINT32_MAX : j;
    CHECK(plan()==0); CHECK(verify()==0); /* absent routes retain original slots */
    setup(128,8); for (unsigned j=0;j<1024;j++) ids[j]=UINT32_MAX;
    CHECK(plan()==0); CHECK(verify()==0); CHECK(meta[0]==128 && meta[2]==0 && meta[1]==1);
    setup(5,0); CHECK(plan()==0); CHECK(verify()==0); CHECK(meta[0]==5 && meta[2]==0);
    setup(128,8); cfg[5]=1151; refusal(3); cfg[5]=1152; cfg[7]=1159; refusal(3);
    setup(1,8); cfg[6]=1; refusal(3);
    setup(1,8); ids[3]=256; refusal(2); ids[3]=UINT32_MAX-1; refusal(2);
    setup(1,8); cfg[0]=2; refusal(1); cfg[0]=1; cfg[1]=129; refusal(1);
    setup(1,8); cfg[2]=9; refusal(1); setup(1,8); cfg[4]=255; refusal(1);
    setup(1,8); cfg[5]=1153; refusal(1); setup(1,8); cfg[6]=258; refusal(1);
    setup(1,8); cfg[7]=1161; refusal(1); setup(1,8); cfg[3]=0; refusal(1);
    setup(1,8); cfg[3]=257; refusal(1);
    setup(1,8); uint64_t result=77;
    CHECK(omega_moe_plan(&result,cfg,8,ids,1024,work,10368,work,1028,maps,2320,meta,8)==OMEGA_BAD_ARG);
    CHECK(result==77); CHECK(work[0]==UINT64_C(0x5a5a5a5a5a5a5a5a));
    CHECK(omega_moe_plan(&result,cfg,8,ids,1023,work,10368,seg,1028,maps,2320,meta,8)==OMEGA_BAD_ARG);
    CHECK(result==77);
    for (unsigned j=0;j<1024;j++) { uint32_t v=j%257; memcpy(idwire+j*4,&v,4); }
    CHECK(omega_moe_ids32(&result,idwire,4096,1024,ids,1024)==0 && result==0);
    for (unsigned j=0;j<1024;j++) CHECK(ids[j]==j%257);
    CHECK(omega_moe_ids32(&result,idwire,4096,1025,ids,1024)==0 && result==1);
    setup(1,8); CHECK(plan()==0);
    memset(mapwire,90,sizeof mapwire); maps[0]=UINT64_MAX;
    CHECK(omega_moe_maps32(&result,meta,8,maps,2320,mapwire,9280)==0 && result==2);
    for (unsigned j=0;j<sizeof mapwire;j++) CHECK(mapwire[j]==90);
    CHECK(plan()==0); memset(segwire,90,sizeof segwire); dirs[0]=0;
    CHECK(omega_moe_segs32(&result,seg,1028,meta,8,dirs,514,segwire,8224)==0 && result==3);
    for (unsigned j=0;j<sizeof segwire;j++) CHECK(segwire[j]==90);
    dirs[0]=4096; seg[2]=UINT64_MAX;
    CHECK(omega_moe_segs32(&result,seg,1028,meta,8,dirs,514,segwire,8224)==0 && result==2);
    for (unsigned j=0;j<sizeof segwire;j++) CHECK(segwire[j]==90);
    for (unsigned n=0;n<=128;n++) {
        uint64_t wc[8]={1,n,2048,512,9,512,128,0}; memcpy(window_cfg,wc,sizeof wc);
        CHECK(omega_moe_windows(&result,window_cfg,8,windows,1024,window_meta,4)==0 && result==0);
        uint64_t first=0;
        for (unsigned z=0;z<window_meta[0];z++) {
            uint64_t *v=windows+z*8;
            CHECK(v[0]==first && v[1]>0 && v[2]==v[1]*9 && v[2]<=128);
            CHECK(v[3]==first*2048*4 && v[4]==first*9*512*2*4);
            CHECK(v[5]==first*9*512*4 && v[6]==first*9*2048*4 && v[7]==first*512);
            first+=v[1];
        }
        CHECK(first==n && window_meta[1]==n && window_meta[2]==n*9 && window_meta[3]==1);
    }
    uint64_t old_windows[1024],old_wmeta[4]; memcpy(old_windows,windows,sizeof windows); memcpy(old_wmeta,window_meta,sizeof window_meta);
    window_cfg[6]=1;
    CHECK(omega_moe_windows(&result,window_cfg,8,windows,1024,window_meta,4)==0 && result==2);
    CHECK(!memcmp(old_windows,windows,sizeof windows) && !memcmp(old_wmeta,window_meta,sizeof window_meta));
    window_cfg[6]=128; window_cfg[7]=1;
    CHECK(omega_moe_windows(&result,window_cfg,8,windows,1024,window_meta,4)==0 && result==1);
    CHECK(!memcmp(old_windows,windows,sizeof windows) && !memcmp(old_wmeta,window_meta,sizeof window_meta));
    uint64_t q[16]={4096,8192,12288,65536,131072,196608,262144,512,14,126,134,9,1,1,1,256};
    memcpy(req,q,sizeof q); CHECK(group()==0);
    for (unsigned i=0;i<7;i++) { CHECK(words[i*2]==req[i]); CHECK(words[i*2+1]==0); }
    CHECK(words[14]==512 && words[15]==14);
    uint32_t before[16]; memcpy(before,words,sizeof words);
#define BAD(field,value,rc) do { uint64_t save=req[field]; req[field]=value; CHECK(group()==rc); CHECK(!memcmp(before,words,sizeof words)); req[field]=save; } while(0)
    BAD(8,15,1); BAD(8,0,1); BAD(11,8,1); BAD(15,255,1);
    BAD(9,125,2); BAD(10,133,2); BAD(7,0,2); BAD(7,513,3); BAD(0,4097,3);
    BAD(12,0,4); BAD(13,2,4); BAD(14,0,4);
    BAD(3,4096,5); BAD(4,65536,5); BAD(5,65536,5); BAD(6,196608,5);
    BAD(0,UINT64_MAX-7,5); BAD(3,UINT64_MAX-7,5); BAD(7,128,2); BAD(8,UINT64_MAX,1);
    req[12]=0; req[13]=0; CHECK(group()==0); req[12]=1; req[13]=1;
    puts("MOE-INDEX HOST PASS: native ABI, incumbent pf_group, GROUP format, sparse rank, refusals, mutants");
    return 0;
}
