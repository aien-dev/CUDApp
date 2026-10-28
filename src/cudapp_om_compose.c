#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#endif
/* Emitted by Omega Bridge. GNU C11, POSIX. Scalars use canonical 64-bit bits. */
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
#include <setjmp.h>
static _Thread_local jmp_buf o_jmp;
static _Thread_local int o_busy;
static __attribute__((noreturn)) void o_trap(int c){if(o_busy)longjmp(o_jmp,c);_exit(100+c);}
static _Thread_local unsigned o_depth;
static void o_enter(void){if(++o_depth>128)o_trap(6);}
static uint64_t o_leave(uint64_t v){--o_depth;return v;}
static unsigned o_width(unsigned t){static const unsigned w[]={0,1,8,16,32,64,8,16,32,64};return w[t];}
static __int128 o_value(unsigned t,uint64_t v){return t>=6 && v>>63 ? (__int128)v-(((__int128)1)<<64):(__int128)v;}
static uint64_t o_norm(unsigned t,__int128 v,int code){unsigned w=o_width(t);if(t>=6){__int128 m=((__int128)1)<<(w-1);if(v< -m||v>=m)o_trap(code);}else {if(v<0||(__uint128_t)v>=(((__uint128_t)1)<<w))o_trap(code);}return (uint64_t)v;}
static uint64_t o_cast(unsigned t,unsigned s,uint64_t a){return o_norm(t,o_value(s,a),5);}
static uint64_t o_op(unsigned op,unsigned t,uint64_t a,uint64_t b){
 __int128 x=o_value(t,a),y=o_value(t,b),z=0;unsigned w=o_width(t);uint64_t mask=w==64?UINT64_MAX:(((uint64_t)1<<w)-1);
 if(op>=6&&op<=11){switch(op){case 6:return x==y;case 7:return x!=y;case 8:return x<y;case 9:return x<=y;case 10:return x>y;default:return x>=y;}}
 if(op==12)return a&b;if(op==13)return a|b;if(op==14)return a^b;
 if(op==15||op==16){if(b>=w)o_trap(7);if(op==15)return t>=6?o_norm(t,x*(((__int128)1)<<b),1):(a<<b)&mask;if(t<6)return a>>b;if(!b)return a;return (a>>b)|((a>>63)?(UINT64_MAX<<(64-b)):0);}
 if(op==17){if(t<6)return (0-a)&mask;return o_norm(t,-x,1);}if(op==18)return t==1?!a:(t>=6?~a:(~a)&mask);
 if(op==4||op==5){if(!b)o_trap(2);z=op==4?x/y:x%y;return t>=6?o_norm(t,z,1):(uint64_t)z;}
 if(t<6){if(op==1)return (a+b)&mask;if(op==2)return (a-b)&mask;return (a*b)&mask;}
 if(op==1)z=x+y;if(op==2)z=x-y;if(op==3)z=x*y;return o_norm(t,z,1);
}
static uint64_t o_get(const void *p,uint64_t n,uint64_t i,unsigned t){if(i>=n)o_trap(3);switch(t){case 2:return ((const uint8_t*)p)[i];case 3:return ((const uint16_t*)p)[i];case 4:return ((const uint32_t*)p)[i];case 5:return ((const uint64_t*)p)[i];case 6:return (uint64_t)((const int8_t*)p)[i];case 7:return (uint64_t)((const int16_t*)p)[i];case 8:return (uint64_t)((const int32_t*)p)[i];default:return (uint64_t)((const int64_t*)p)[i];}}
static void o_put(void *p,uint64_t n,uint64_t i,uint64_t v,unsigned t){if(i>=n)o_trap(3);o_norm(t,o_value(t,v),5);switch(t){case 2:((uint8_t*)p)[i]=(uint8_t)v;break;case 3:((uint16_t*)p)[i]=(uint16_t)v;break;case 4:((uint32_t*)p)[i]=(uint32_t)v;break;case 5:((uint64_t*)p)[i]=v;break;case 6:((int8_t*)p)[i]=(int8_t)o_value(t,v);break;case 7:((int16_t*)p)[i]=(int16_t)o_value(t,v);break;case 8:((int32_t*)p)[i]=(int32_t)o_value(t,v);break;default:((int64_t*)p)[i]=(int64_t)o_value(t,v);}}
static void o_fill(void *p,uint64_t n,uint64_t v,unsigned t){for(uint64_t i=0;i<n;i++)o_put(p,n,i,v,t);}
static uint64_t o_io(unsigned op,uint64_t fd,const void *p,uint64_t n,uint64_t off,uint64_t len){if(off>n||len>n-off)o_trap(3);int f=(int)o_value(8,fd);ssize_t r=op==1?read(f,(uint8_t*)p+off,(size_t)len):write(f,(const uint8_t*)p+off,(size_t)len);return r<0?(uint64_t)(-(int64_t)errno):(uint64_t)r;}
static int o_valid(unsigned t,uint64_t v){unsigned w=o_width(t);__int128 x=o_value(t,v);__int128 m=((__int128)1)<<(t>=6?w-1:w);return t>=6?x>=-m&&x<m:x>=0&&x<m;}
static int o_buf(const void *p,uint64_t n,uint64_t want,unsigned t){uintptr_t a=(uintptr_t)p;uint64_t z=t==2?1:8;return p&&n==want&&a%z==0&&n<=UINTPTR_MAX/z&&a<=UINTPTR_MAX-n*z;}
static int o_overlap(const void *a,uint64_t an,const void *b,uint64_t bn){uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;return x<y+bn&&y<x+an;}
#include <float.h>
#include <fenv.h>
#if FLT_RADIX != 2 || FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128 || FLT_EVAL_METHOD != 0
#error unsupported binary32 environment
#endif
#if defined(__FAST_MATH__)
#error fast math is not a supported Omega environment
#endif
static uint64_t o_f32(unsigned op,uint64_t a,uint64_t b){uint32_t ua=(uint32_t)a,ub=(uint32_t)b,ur;float x,y;__builtin_memcpy(&x,&ua,4);__builtin_memcpy(&y,&ub,4);if(op==1)return ua&UINT32_C(0x7fffffff);if(op==2){volatile float z=x*y;__builtin_memcpy(&ur,(const void*)&z,4);return ur;}return x<=y;}
static uint64_t f0(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f1(const void *v0, uint64_t v0_n, uint64_t v1, void *v2, uint64_t v2_n);
static uint64_t f2(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3);
static uint64_t f3(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f4(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2);
static uint64_t f5(const void *v0, uint64_t v0_n, uint64_t v1, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, void *v4, uint64_t v4_n, uint64_t v5);
static uint64_t f6(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f7(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3);
static uint64_t f8(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4);
static uint64_t f9(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f10(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2);
static uint64_t f11(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f12(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f13(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f14(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f15(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3);
static uint64_t f16(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3);
static uint64_t f17(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f18(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, const void *v3, uint64_t v3_n);
static uint64_t f19(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f20(const void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n);
static uint64_t f21(const void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n);
static uint64_t f22(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f23(const void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n);
static uint64_t f24(uint64_t v0);
static uint64_t f25(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f26(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f27(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f28(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f29(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n);
static uint64_t f30(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f31(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f32(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f33(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f34(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f35(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3);
static uint64_t f36(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f37(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f38(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f39(const void *v0, uint64_t v0_n);
static uint64_t f40(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f41(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n);
static uint64_t f42(const void *v0, uint64_t v0_n);
static uint64_t f43(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n);
static uint64_t f44(uint64_t v0, uint64_t v1);
static uint64_t f45(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, uint64_t v3);
static uint64_t f46(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n);
static uint64_t f47(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f48(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f49(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f50(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f51(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f52(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f53(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f54(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f55(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f56(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f57(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f58(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f59(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n);
static uint64_t f60(uint64_t v0);
static uint64_t f61(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n);
static uint64_t f62(const void *v0, uint64_t v0_n);
static uint64_t f63(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n);
static uint64_t f64(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n);
static uint64_t f65(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f66(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f67(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f68(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f69(void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f70(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f71(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, uint64_t v4);
static uint64_t f72(void *v0, uint64_t v0_n);
static uint64_t f73(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n);
static uint64_t f74(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f75(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f76(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, uint64_t v4);
static uint64_t f77(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f78(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3);
static uint64_t f79(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3);
static uint64_t f80(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3, uint64_t v4);
static uint64_t f81(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3);
static uint64_t f82(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3);
static uint64_t f83(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f84(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f85(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n);
static uint64_t f86(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, void *v4, uint64_t v4_n);
static uint64_t f87(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, void *v4, uint64_t v4_n);
static uint64_t f0(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=288ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=18446744073709551615ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=({uint64_t _a=v1;uint64_t _b=288ULL;o_op(1, 5, _a,_b);});uint64_t _v=v2;o_put(v0, v0_n, _i,_v,5);}
}
{uint64_t _i=({uint64_t _a=v1;uint64_t _b=576ULL;o_op(1, 5, _a,_b);});uint64_t _v=v2;o_put(v0, v0_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f1(const void *v0, uint64_t v0_n, uint64_t v1, void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=288ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=288ULL;o_op(1, 5, _a,_b);});uint64_t _v=18446744073709551615ULL;o_put(v2, v2_n, _i,_v,5);}
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=576ULL;o_op(1, 5, _a,_b);});uint64_t _v=18446744073709551615ULL;o_put(v2, v2_n, _i,_v,5);}
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=1152ULL;o_op(1, 5, _a,_b);});uint64_t _v=0ULL;o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v3=0;
uint64_t c1=0;
for(;o_op(8,5, v3, v1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v4=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});
uint64_t __attribute__((unused)) v5=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=7ULL;o_op(1, 5, _a,_b);}), 5);
{
uint64_t v6=0;
uint64_t k2=5ULL;
for(;o_op(8,5, v6, k2);v6=o_op(1,5, v6,1)){
uint64_t __attribute__((unused)) v7=o_get(v0, v0_n, ({uint64_t _a=({uint64_t _a=v4;uint64_t _b=v6;o_op(1, 5, _a,_b);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);
if(({uint64_t _a=v7;uint64_t _b=18446744073709551615ULL;o_op(7, 5, _a,_b);})){
uint64_t __attribute__((unused)) v8=0ULL;
if(({uint64_t _a=v6;uint64_t _b=4ULL;o_op(6, 5, _a,_b);})){
v8=256ULL;
}else{
if(({uint64_t _a=({uint64_t _a=v6;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=1ULL;o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
v8=256ULL;
}
}
if(({uint64_t _a=v7;uint64_t _b=({uint64_t _a=288ULL;uint64_t _b=v8;o_op(2, 5, _a,_b);});o_op(11, 5, _a,_b);})){
return o_leave(1ULL);
}
v7=({uint64_t _a=v7;uint64_t _b=v8;o_op(1, 5, _a,_b);});
if(({uint64_t _a=o_get(v2, v2_n, v7, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(1ULL);
}
({uint64_t p1=v7;uint64_t p2=v3;f0(v2, v2_n, p1, p2);});
}
}
}
}
}
{
uint64_t v3=0;
uint64_t c1=0;
for(;o_op(8,5, v3, v1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);}), 5);uint64_t _b=46ULL;o_op(6, 5, _a,_b);})){
uint64_t __attribute__((unused)) v4=({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=6ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=4294967295ULL;o_op(12, 5, _a,_b);});
if(({uint64_t _a=v4;uint64_t _b=2147483648ULL;o_op(11, 5, _a,_b);})){
uint64_t __attribute__((unused)) v5=({uint64_t _a=4294967296ULL;uint64_t _b=v4;o_op(2, 5, _a,_b);});
if(({uint64_t _a=v5;uint64_t _b=v3;o_op(10, 5, _a,_b);})){
return o_leave(1ULL);
}
uint64_t __attribute__((unused)) v6=({uint64_t _a=v3;uint64_t _b=v5;o_op(2, 5, _a,_b);});
{
uint64_t v7=0;
uint64_t k4=256ULL;
for(;o_op(8,5, v7, k4);v7=o_op(1,5, v7,1)){
uint64_t __attribute__((unused)) v8=o_get(v2, v2_n, ({uint64_t _a=v7;uint64_t _b=288ULL;o_op(1, 5, _a,_b);}), 5);
uint64_t __attribute__((unused)) v9=o_get(v2, v2_n, ({uint64_t _a=v7;uint64_t _b=576ULL;o_op(1, 5, _a,_b);}), 5);
if(({uint64_t _a=v8;uint64_t _b=18446744073709551615ULL;o_op(7, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=v8;uint64_t _b=v6;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=v9;uint64_t _b=v6;o_op(11, 5, _a,_b);});o_op(12, 1, _a,_b);})){
if(({uint64_t _a=v9;uint64_t _b=v3;o_op(8, 5, _a,_b);})){
{uint64_t _i=({uint64_t _a=v7;uint64_t _b=576ULL;o_op(1, 5, _a,_b);});uint64_t _v=v3;o_put(v2, v2_n, _i,_v,5);}
}
}
}
}
}
}
}
}
}
return o_leave(0ULL);
}
static uint64_t f2(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3){
o_enter();
{
uint64_t v4=0;
uint64_t k1=256ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=1152ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=576ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=v2;o_op(8, 5, _a,_b);})){
uint64_t __attribute__((unused)) v5=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=864ULL;o_op(1, 5, _a,_b);}), 5);
{
uint64_t v6=0;
uint64_t k4=o_get(v0, v0_n, v4, 5);
uint64_t c4=0;
for(;o_op(8,5, v6, k4);v6=o_op(1,5, v6,1)){
if(c4++>=16777216)o_trap(4);
if(({uint64_t _a=({uint64_t _a=v5;uint64_t _b=v6;o_op(1, 5, _a,_b);});uint64_t _b=v3;o_op(8, 5, _a,_b);})){
{uint64_t _i=({uint64_t _a=v5;uint64_t _b=v6;o_op(1, 5, _a,_b);});uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,2);}
}
}
}
{uint64_t _i=({uint64_t _a=v4;uint64_t _b=1152ULL;o_op(1, 5, _a,_b);});uint64_t _v=0ULL;o_put(v0, v0_n, _i,_v,5);}
}
}
}
}
return o_leave(0);
}
static uint64_t f3(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
{
uint64_t v3=0;
uint64_t c1=0;
for(;o_op(8,5, v3, v2);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
if(({uint64_t _a=v3;uint64_t _b=2ULL;o_op(11, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=v3;uint64_t _b=v1;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=v3;uint64_t _b=v1;o_op(1, 5, _a,_b);});uint64_t _b=v2;o_op(9, 5, _a,_b);})){
uint64_t __attribute__((unused)) v4=1ULL;
{
uint64_t v5=0;
uint64_t c5=0;
for(;o_op(8,5, v5, v1);v5=o_op(1,5, v5,1)){
if(c5++>=16777216)o_trap(4);
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v3;uint64_t _b=v5;o_op(1, 5, _a,_b);}), 2);uint64_t _b=0ULL;o_op(7, 2, _a,_b);})){
v4=0ULL;
}
}
}
if(v4){
return o_leave(v3);
}
}
}
}
}
}
return o_leave(18446744073709551615ULL);
}
static uint64_t f4(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2){
o_enter();
uint8_t v3[246]={0};
uint64_t v3_n=246;
o_fill(v3, v3_n, 0ULL, 2);
uint64_t __attribute__((unused)) v4=0ULL;
uint64_t __attribute__((unused)) v5=2ULL;
{
uint64_t v6=0;
uint64_t k1=256ULL;
for(;o_op(8,5, v6, k1);v6=o_op(1,5, v6,1)){
if(({uint64_t _a=o_get(v0, v0_n, v6, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v6;uint64_t _b=288ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=18446744073709551615ULL;o_op(7, 5, _a,_b);})){
{uint64_t _i=({uint64_t _a=v4;uint64_t _b=1440ULL;o_op(1, 5, _a,_b);});uint64_t _v=v6;o_put(v0, v0_n, _i,_v,5);}
v4=({uint64_t _a=v4;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
}else{
{uint64_t _i=v6;uint64_t _v=2ULL;o_put(v1, v1_n, _i,_v,5);}
}
}
}
}
{
uint64_t v6=0;
uint64_t c1=0;
for(;o_op(8,5, v6, v4);v6=o_op(1,5, v6,1)){
if(c1++>=16777216)o_trap(4);
{
uint64_t v7=0;
uint64_t c2=0;
for(;o_op(8,5, v7, v4);v7=o_op(1,5, v7,1)){
if(c2++>=16777216)o_trap(4);
if(({uint64_t _a=v7;uint64_t _b=v6;o_op(10, 5, _a,_b);})){
uint64_t __attribute__((unused)) v8=o_get(v0, v0_n, ({uint64_t _a=v6;uint64_t _b=1440ULL;o_op(1, 5, _a,_b);}), 5);
uint64_t __attribute__((unused)) v9=o_get(v0, v0_n, ({uint64_t _a=v7;uint64_t _b=1440ULL;o_op(1, 5, _a,_b);}), 5);
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v8;uint64_t _b=288ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=o_get(v0, v0_n, ({uint64_t _a=v9;uint64_t _b=288ULL;o_op(1, 5, _a,_b);}), 5);o_op(10, 5, _a,_b);})){
{uint64_t _i=({uint64_t _a=v6;uint64_t _b=1440ULL;o_op(1, 5, _a,_b);});uint64_t _v=v9;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=({uint64_t _a=v7;uint64_t _b=1440ULL;o_op(1, 5, _a,_b);});uint64_t _v=v8;o_put(v0, v0_n, _i,_v,5);}
}
}
}
}
}
}
{
uint64_t v6=0;
uint64_t c1=0;
for(;o_op(8,5, v6, v4);v6=o_op(1,5, v6,1)){
if(c1++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v7=o_get(v0, v0_n, ({uint64_t _a=v6;uint64_t _b=1440ULL;o_op(1, 5, _a,_b);}), 5);
({uint64_t p2=o_get(v0, v0_n, ({uint64_t _a=v7;uint64_t _b=288ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p3=v2;f2(v0, v0_n, v3, v3_n, p2, p3);});
uint64_t __attribute__((unused)) v8=o_get(v0, v0_n, v7, 5);
uint64_t __attribute__((unused)) v9=({uint64_t p1=v8;uint64_t p2=v2;f3(v3, v3_n, p1, p2);});
if(({uint64_t _a=v9;uint64_t _b=18446744073709551615ULL;o_op(6, 5, _a,_b);})){
return o_leave(0ULL);
}
{
uint64_t v10=0;
uint64_t c2=0;
for(;o_op(8,5, v10, v8);v10=o_op(1,5, v10,1)){
if(c2++>=16777216)o_trap(4);
{uint64_t _i=({uint64_t _a=v9;uint64_t _b=v10;o_op(1, 5, _a,_b);});uint64_t _v=1ULL;o_put(v3, v3_n, _i,_v,2);}
}
}
{uint64_t _i=({uint64_t _a=v7;uint64_t _b=864ULL;o_op(1, 5, _a,_b);});uint64_t _v=v9;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=({uint64_t _a=v7;uint64_t _b=1152ULL;o_op(1, 5, _a,_b);});uint64_t _v=1ULL;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=v7;uint64_t _v=v9;o_put(v1, v1_n, _i,_v,5);}
uint64_t __attribute__((unused)) v10=({uint64_t _a=({uint64_t _a=v9;uint64_t _b=v8;o_op(1, 5, _a,_b);});uint64_t _b=1ULL;o_op(2, 5, _a,_b);});
if(({uint64_t _a=v10;uint64_t _b=v5;o_op(10, 5, _a,_b);})){
v5=v10;
}
}
}
return o_leave(({uint64_t _a=v5;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}));
}
static uint64_t f5(const void *v0, uint64_t v0_n, uint64_t v1, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, void *v4, uint64_t v4_n, uint64_t v5){
o_enter();
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=1024ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=v5;uint64_t _b=3ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=v5;uint64_t _b=246ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v6=0;
uint64_t k1=288ULL;
for(;o_op(8,5, v6, k1);v6=o_op(1,5, v6,1)){
uint64_t __attribute__((unused)) v7=o_get(v2, v2_n, v6, 5);
if(({uint64_t _a=({uint64_t _a=v7;uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v7;uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v7;uint64_t _b=2ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=v7;uint64_t _b=4ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(12, 1, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=v6;uint64_t _b=256ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=v7;uint64_t _b=2ULL;o_op(10, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
{uint64_t _i=v6;uint64_t _v=18446744073709551615ULL;o_put(v3, v3_n, _i,_v,5);}
}
}
uint64_t __attribute__((unused)) v6=({uint64_t p1=v1;f1(v0, v0_n, p1, v2, v2_n);});
if(({uint64_t _a=v6;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v6);
}
uint64_t __attribute__((unused)) v7=({uint64_t p2=v5;f4(v2, v2_n, v3, v3_n, p2);});
if(({uint64_t _a=v7;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(2ULL);
}
uint64_t __attribute__((unused)) v8=4ULL;
{
uint64_t v9=0;
uint64_t k1=32ULL;
for(;o_op(8,5, v9, k1);v9=o_op(1,5, v9,1)){
uint64_t __attribute__((unused)) v10=({uint64_t _a=v9;uint64_t _b=256ULL;o_op(1, 5, _a,_b);});
uint64_t __attribute__((unused)) v11=o_get(v2, v2_n, v10, 5);
if(({uint64_t _a=v11;uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
if(({uint64_t _a=v11;uint64_t _b=2ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=v8;uint64_t _b=2ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
v8=({uint64_t _a=v8;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
}
}
if(({uint64_t _a=({uint64_t _a=v8;uint64_t _b=v11;o_op(1, 5, _a,_b);});uint64_t _b=32ULL;o_op(10, 5, _a,_b);})){
return o_leave(2ULL);
}
{uint64_t _i=v10;uint64_t _v=v8;o_put(v3, v3_n, _i,_v,5);}
v8=({uint64_t _a=v8;uint64_t _b=v11;o_op(1, 5, _a,_b);});
}
}
}
{uint64_t _i=0ULL;uint64_t _v=v1;o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=v7;o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=v8;o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=3ULL;uint64_t _v=0ULL;o_put(v4, v4_n, _i,_v,5);}
return o_leave(0ULL);
}
static uint64_t f6(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
{uint64_t _i=16ULL;uint64_t _v=v1;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=17ULL;uint64_t _v=v2;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=18ULL;uint64_t _v=v3;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=19ULL;uint64_t _v=v4;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=20ULL;uint64_t _v=v5;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=21ULL;uint64_t _v=18446744073709551615ULL;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=22ULL;uint64_t _v=0ULL;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=23ULL;uint64_t _v=0ULL;o_put(v0, v0_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f7(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3){
o_enter();
uint64_t __attribute__((unused)) v4=4ULL;
uint64_t __attribute__((unused)) v5=0ULL;
uint64_t __attribute__((unused)) v6=256ULL;
if(v3){
v4=5ULL;
v5=256ULL;
v6=32ULL;
}
uint64_t __attribute__((unused)) v7=o_get(v0, v0_n, v4, 5);
if(({uint64_t _a=v7;uint64_t _b=v6;o_op(11, 5, _a,_b);})){
{uint64_t _i=3ULL;uint64_t _v=1ULL;o_put(v0, v0_n, _i,_v,5);}
return o_leave(18446744073709551615ULL);
}
{uint64_t _i=v4;uint64_t _v=({uint64_t _a=v7;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=({uint64_t _a=v5;uint64_t _b=v7;o_op(1, 5, _a,_b);});uint64_t _v=v2;o_put(v1, v1_n, _i,_v,5);}
return o_leave(v7);
}
static uint64_t f8(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4){
o_enter();
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t _a=({uint64_t _a=v0;uint64_t _b=9ULL;o_op(15, 5, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=14ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v2;uint64_t _b=17ULL;o_op(15, 5, _a,_b);});uint64_t _b=({uint64_t _a=v3;uint64_t _b=20ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});o_op(13, 5, _a,_b);});
if(v4){
v5=({uint64_t _a=v5;uint64_t _b=8192ULL;o_op(13, 5, _a,_b);});
}
return o_leave(v5);
}
static uint64_t f9(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
{
uint64_t v2=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
{uint64_t _i=({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=v2;o_op(1, 5, _a,_b);});uint64_t _v=o_get(v1, v1_n, ({uint64_t _a=v2;uint64_t _b=16ULL;o_op(1, 5, _a,_b);}), 5);o_put(v0, v0_n, _i,_v,5);}
}
}
{uint64_t _i=0ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f10(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2){
o_enter();
if(({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1024ULL;o_op(11, 5, _a,_b);})){
{uint64_t _i=3ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
return o_leave(0);
}
uint64_t __attribute__((unused)) v3=o_get(v1, v1_n, 1ULL, 5);
uint64_t __attribute__((unused)) v4=6ULL;
uint64_t __attribute__((unused)) v5=7ULL;
uint64_t __attribute__((unused)) v6=7ULL;
uint64_t __attribute__((unused)) v7=0ULL;
uint64_t __attribute__((unused)) v8=0ULL;
if(({uint64_t _a=v2;uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
v4=14ULL;
}
if(({uint64_t _a=({uint64_t _a=v2;uint64_t _b=2ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v2;uint64_t _b=3ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
v4=4ULL;
v5=0ULL;
v6=1ULL;
v7=1ULL;
v8=3ULL;
if(({uint64_t _a=v2;uint64_t _b=3ULL;o_op(6, 5, _a,_b);})){
v3=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(12, 5, _a,_b);});
}
}
if(({uint64_t _a=v2;uint64_t _b=4ULL;o_op(6, 5, _a,_b);})){
v4=4ULL;
v5=2ULL;
v7=1ULL;
v8=4ULL;
}
if(({uint64_t _a=v2;uint64_t _b=5ULL;o_op(6, 5, _a,_b);})){
v4=4ULL;
v6=1ULL;
v7=1ULL;
v8=2ULL;
}
if(({uint64_t _a=({uint64_t _a=v2;uint64_t _b=7ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v2;uint64_t _b=8ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
v4=5ULL;
if(({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 23ULL, 5);uint64_t _b=2ULL;o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
v8=v3;
}
}
if(({uint64_t _a=v2;uint64_t _b=9ULL;o_op(6, 5, _a,_b);})){
v4=1ULL;
v7=1ULL;
}
if(({uint64_t _a=v2;uint64_t _b=10ULL;o_op(6, 5, _a,_b);})){
v4=5ULL;
v7=1ULL;
}
if(({uint64_t _a=({uint64_t _a=v2;uint64_t _b=11ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v2;uint64_t _b=12ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
v4=4ULL;
v5=3ULL;
v6=1ULL;
v7=1ULL;
v8=2ULL;
{uint64_t _i=2ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 2ULL, 5);uint64_t _b=8ULL;o_op(13, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=v2;uint64_t _b=12ULL;o_op(6, 5, _a,_b);})){
v3=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(12, 5, _a,_b);});
}
}
if(({uint64_t _a=v2;uint64_t _b=13ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 2ULL, 5);uint64_t _b=8ULL;o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=3ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
return o_leave(0);
}
v3=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(13, 5, _a,_b);});
{uint64_t _i=2ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 2ULL, 5);uint64_t _b=8ULL;o_op(14, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
}
if(({uint64_t _a=v2;uint64_t _b=13ULL;o_op(10, 5, _a,_b);})){
{uint64_t _i=3ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
return o_leave(0);
}
v4=({uint64_t p0=v4;uint64_t p1=v5;uint64_t p2=v6;uint64_t p3=v3;uint64_t p4=v7;f8(p0, p1, p2, p3, p4);});
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 22ULL, 5);uint64_t _b=4294967295ULL;o_op(12, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=32ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=v8;o_put(v1, v1_n, _i,_v,5);}
({f9(v0, v0_n, v1, v1_n);});
return o_leave(0);
}
static uint64_t f11(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
({uint64_t p1=v2;uint64_t p2=v3;uint64_t p3=v4;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=v5;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
return o_leave(0);
}
static uint64_t f12(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
({uint64_t p1=v2;uint64_t p2=v3;uint64_t p3=v4;uint64_t p4=18446744073709551615ULL;uint64_t p5=v5;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=o_get(v1, v1_n, 8ULL, 5);o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=23ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 9ULL, 5);uint64_t _b=24ULL;o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
return o_leave(0);
}
static uint64_t f13(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
({uint64_t p1=v2;uint64_t p2=v3;uint64_t p3=v4;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=21ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=22ULL;uint64_t _v=o_get(v1, v1_n, 8ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=v5;f10(v0, v0_n, v1, v1_n, p2);});
return o_leave(0);
}
static uint64_t f14(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
({uint64_t p1=33ULL;uint64_t p2=v2;uint64_t p3=v3;uint64_t p4=v4;uint64_t p5=v5;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=23ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 9ULL, 5);uint64_t _b=32ULL;o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
return o_leave(0);
}
static uint64_t f15(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3){
o_enter();
({uint64_t p1=28ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=v2;uint64_t p4=v3;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=1ULL;f10(v0, v0_n, v1, v1_n, p2);});
return o_leave(0);
}
static uint64_t f16(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3){
o_enter();
uint64_t __attribute__((unused)) v4=o_get(v1, v1_n, 0ULL, 5);
({uint64_t p1=46ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v2;uint64_t _b=v4;o_op(2, 5, _a,_b);});uint64_t _b=4294967295ULL;o_op(12, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
if(v3){
{uint64_t _i=23ULL;uint64_t _v=2ULL;o_put(v1, v1_n, _i,_v,5);}
}
({uint64_t p2=8ULL;f10(v0, v0_n, v1, v1_n, p2);});
return o_leave(v4);
}
static uint64_t f17(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
uint64_t __attribute__((unused)) v3=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=6ULL;o_op(1, 5, _a,_b);});
{uint64_t _i=v3;uint64_t _v=({uint64_t _a=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, v3, 5);uint64_t _b=32ULL;o_op(16, 5, _a,_b);});uint64_t _b=32ULL;o_op(15, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v2;uint64_t _b=v1;o_op(2, 5, _a,_b);});uint64_t _b=4294967295ULL;o_op(12, 5, _a,_b);});o_op(13, 5, _a,_b);});o_put(v0, v0_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f18(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, const void *v3, uint64_t v3_n){
o_enter();
{
uint64_t v4=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(({uint64_t _a=o_get(v3, v3_n, v4, 5);uint64_t _b=256ULL;o_op(11, 5, _a,_b);})){
return o_leave(1ULL);
}
}
}
if(({uint64_t _a=({uint64_t _a=o_get(v2, v2_n, o_get(v3, v3_n, 0ULL, 5), 5);uint64_t _b=4ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v2, v2_n, o_get(v3, v3_n, 1ULL, 5), 5);uint64_t _b=4ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v2, v2_n, o_get(v3, v3_n, 2ULL, 5), 5);uint64_t _b=2ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v2, v2_n, o_get(v3, v3_n, 3ULL, 5), 5);uint64_t _b=4ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1021ULL;o_op(10, 5, _a,_b);})){
return o_leave(1ULL);
}
({uint64_t p1=16ULL;uint64_t p2=o_get(v3, v3_n, 0ULL, 5);uint64_t p3=o_get(v3, v3_n, 1ULL, 5);uint64_t p4=o_get(v3, v3_n, 2ULL, 5);uint64_t p5=o_get(v3, v3_n, 3ULL, 5);f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=({uint64_t p0=8ULL;uint64_t p1=7ULL;uint64_t p2=7ULL;uint64_t p3=o_get(v1, v1_n, 1ULL, 5);uint64_t p4=0ULL;f8(p0, p1, p2, p3, p4);});uint64_t _b=32ULL;o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1024ULL;o_op(11, 5, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v4=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
{uint64_t _i=({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=v4;o_op(1, 5, _a,_b);});uint64_t _v=o_get(v1, v1_n, ({uint64_t _a=v4;uint64_t _b=16ULL;o_op(1, 5, _a,_b);}), 5);o_put(v0, v0_n, _i,_v,5);}
}
}
{uint64_t _i=0ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{
uint64_t v4=0;
uint64_t k1=2ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
({uint64_t p1=0ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
{uint64_t _i=({uint64_t _a=({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(2, 5, _a,_b);});uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=6ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t _a=({uint64_t p0=({uint64_t _a=15ULL;uint64_t _b=v4;o_op(2, 5, _a,_b);});uint64_t p1=7ULL;uint64_t p2=7ULL;uint64_t p3=0ULL;uint64_t p4=0ULL;f8(p0, p1, p2, p3, p4);});uint64_t _b=32ULL;o_op(15, 5, _a,_b);});o_put(v0, v0_n, _i,_v,5);}
}
}
return o_leave(0ULL);
}
static uint64_t f19(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
if(({uint64_t _a=v1;uint64_t _b=256ULL;o_op(11, 5, _a,_b);})){
return o_leave(0ULL);
}
return o_leave(({uint64_t _a=o_get(v0, v0_n, v1, 5);uint64_t _b=v2;o_op(6, 5, _a,_b);}));
}
static uint64_t f20(const void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=o_get(v0, v0_n, v1, 5);
uint64_t __attribute__((unused)) v4=o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=7ULL;o_op(1, 5, _a,_b);}), 5);
if(({uint64_t _a=v3;uint64_t _b=59ULL;o_op(10, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=48ULL;o_op(16, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v5=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v5, k1);v5=o_op(1,5, v5,1)){
uint64_t __attribute__((unused)) v6=o_get(v0, v0_n, ({uint64_t _a=({uint64_t _a=v1;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t _b=v5;o_op(1, 5, _a,_b);}), 5);
if(({uint64_t _a=v6;uint64_t _b=18446744073709551615ULL;o_op(7, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=v5;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=1ULL;o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
if(({uint64_t _a=v6;uint64_t _b=32ULL;o_op(11, 5, _a,_b);})){
return o_leave(2ULL);
}
v6=({uint64_t _a=v6;uint64_t _b=256ULL;o_op(1, 5, _a,_b);});
}else{
if(({uint64_t _a=v6;uint64_t _b=256ULL;o_op(11, 5, _a,_b);})){
return o_leave(2ULL);
}
}
uint64_t __attribute__((unused)) v7=o_get(v2, v2_n, v6, 5);
if(({uint64_t _a=v7;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(2ULL);
}
uint64_t __attribute__((unused)) v8=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=({uint64_t _a=16ULL;uint64_t _b=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});o_op(16, 5, _a,_b);});uint64_t _b=255ULL;o_op(12, 5, _a,_b);});
if(({uint64_t _a=v8;uint64_t _b=v7;o_op(11, 5, _a,_b);})){
return o_leave(2ULL);
}
}
}
}
uint64_t __attribute__((unused)) v5=o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=5ULL;o_op(1, 5, _a,_b);}), 5);
if(({uint64_t _a=v5;uint64_t _b=18446744073709551615ULL;o_op(7, 5, _a,_b);})){
if(({uint64_t _a=v5;uint64_t _b=32ULL;o_op(11, 5, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=o_get(v2, v2_n, ({uint64_t _a=v5;uint64_t _b=256ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(2ULL);
}
}
return o_leave(0ULL);
}
static uint64_t f21(const void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=o_get(v0, v0_n, v1, 5);
uint64_t __attribute__((unused)) v4=o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);
uint64_t __attribute__((unused)) v5=o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=2ULL;o_op(1, 5, _a,_b);}), 5);
uint64_t __attribute__((unused)) v6=o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=3ULL;o_op(1, 5, _a,_b);}), 5);
uint64_t __attribute__((unused)) v7=o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=4ULL;o_op(1, 5, _a,_b);}), 5);
if(({uint64_t _a=({uint64_t _a=v3;uint64_t _b=56ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v3;uint64_t _b=57ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
uint64_t __attribute__((unused)) v8=2ULL;
if(({uint64_t _a=v3;uint64_t _b=57ULL;o_op(6, 5, _a,_b);})){
v8=4ULL;
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=v4;uint64_t p2=v8;f19(v2, v2_n, p1, p2);});uint64_t _b=0ULL;o_op(6, 1, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=v5;uint64_t p2=2ULL;f19(v2, v2_n, p1, p2);});uint64_t _b=0ULL;o_op(6, 1, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=v4;uint64_t _b=v5;o_op(6, 5, _a,_b);})){
return o_leave(3ULL);
}
}
if(({uint64_t _a=v3;uint64_t _b=11ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=18446744073709551615ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=v5;uint64_t p2=2ULL;f19(v2, v2_n, p1, p2);});uint64_t _b=0ULL;o_op(6, 1, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
}
if(({uint64_t _a=v3;uint64_t _b=12ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=v5;uint64_t p2=2ULL;f19(v2, v2_n, p1, p2);});uint64_t _b=0ULL;o_op(6, 1, _a,_b);});uint64_t _b=({uint64_t _a=v6;uint64_t _b=18446744073709551615ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
}
if(({uint64_t _a=v3;uint64_t _b=16ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=v4;uint64_t p2=4ULL;f19(v2, v2_n, p1, p2);});uint64_t _b=0ULL;o_op(6, 1, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=v5;uint64_t p2=4ULL;f19(v2, v2_n, p1, p2);});uint64_t _b=0ULL;o_op(6, 1, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t p1=v6;uint64_t p2=2ULL;f19(v2, v2_n, p1, p2);});uint64_t _b=0ULL;o_op(6, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=v7;uint64_t _b=18446744073709551615ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=v7;uint64_t p2=4ULL;f19(v2, v2_n, p1, p2);});uint64_t _b=0ULL;o_op(6, 1, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(3ULL);
}
}
return o_leave(0ULL);
}
static uint64_t f22(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
uint64_t __attribute__((unused)) v3=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});
uint64_t __attribute__((unused)) v4=o_get(v0, v0_n, v3, 5);
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=46ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=54ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
uint64_t __attribute__((unused)) v5=({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v3;uint64_t _b=6ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=4294967295ULL;o_op(12, 5, _a,_b);});
uint64_t __attribute__((unused)) v6=({uint64_t _a=v1;uint64_t _b=v5;o_op(1, 5, _a,_b);});
if(({uint64_t _a=v5;uint64_t _b=2147483648ULL;o_op(11, 5, _a,_b);})){
v6=({uint64_t _a=v1;uint64_t _b=({uint64_t _a=4294967296ULL;uint64_t _b=v5;o_op(2, 5, _a,_b);});o_op(2, 5, _a,_b);});
}
if(({uint64_t _a=v6;uint64_t _b=v2;o_op(11, 5, _a,_b);})){
return o_leave(4ULL);
}
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=54ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v6;uint64_t _b=v1;o_op(9, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(4ULL);
}
}
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=11ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=56ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=57ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);});o_op(13, 1, _a,_b);})){
uint64_t __attribute__((unused)) v5=({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v3;uint64_t _b=6ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=32ULL;o_op(16, 5, _a,_b);});
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=17ULL;o_op(16, 5, _a,_b);});uint64_t _b=7ULL;o_op(12, 5, _a,_b);});uint64_t _b=1ULL;o_op(7, 5, _a,_b);})){
return o_leave(5ULL);
}
uint64_t __attribute__((unused)) v6=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=14ULL;o_op(16, 5, _a,_b);});uint64_t _b=7ULL;o_op(12, 5, _a,_b);});
if(({uint64_t _a=({uint64_t _a=v6;uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v6;uint64_t _b=3ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=v6;uint64_t _b=4ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(5ULL);
}
}
return o_leave(0ULL);
}
static uint64_t f23(const void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n){
o_enter();
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=1024ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v3=0;
uint64_t c1=0;
for(;o_op(8,5, v3, v1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v4=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});
uint64_t __attribute__((unused)) v5=({uint64_t p1=v4;f20(v0, v0_n, p1, v2, v2_n);});
if(({uint64_t _a=v5;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v5);
}
v5=({uint64_t p1=v4;f21(v0, v0_n, p1, v2, v2_n);});
if(({uint64_t _a=v5;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v5);
}
v5=({uint64_t p1=v3;uint64_t p2=v1;f22(v0, v0_n, p1, p2);});
if(({uint64_t _a=v5;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v5);
}
}
}
return o_leave(0ULL);
}
static uint64_t f24(uint64_t v0){
o_enter();
{
uint64_t v1=0;
uint64_t k1=32ULL;
for(;o_op(8,5, v1, k1);v1=o_op(1,5, v1,1)){
if(({uint64_t _a=({uint64_t _a=1ULL;uint64_t _b=v1;o_op(15, 5, _a,_b);});uint64_t _b=v0;o_op(6, 5, _a,_b);})){
return o_leave(v1);
}
}
}
return o_leave(18446744073709551615ULL);
}
static uint64_t f25(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
{uint64_t _i=8ULL;uint64_t _v=v5;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=v2;uint64_t p4=v3;uint64_t p5=v4;f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f26(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
{uint64_t _i=8ULL;uint64_t _v=v5;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=9ULL;uint64_t p3=v2;uint64_t p4=v3;uint64_t p5=v4;f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f27(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
({uint64_t p1=17ULL;uint64_t p2=v2;uint64_t p3=v3;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=v5;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=23ULL;uint64_t _v=({uint64_t _a=v4;uint64_t _b=24ULL;o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
return o_leave(0);
}
static uint64_t f28(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
({uint64_t p1=18ULL;uint64_t p2=v2;uint64_t p3=v3;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=v5;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=23ULL;uint64_t _v=({uint64_t _a=v4;uint64_t _b=24ULL;o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
return o_leave(0);
}
static uint64_t f29(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=({uint64_t p2=2ULL;uint64_t p3=1ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});
{
uint64_t v4=0;
uint64_t k1=31ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
uint64_t __attribute__((unused)) v5=1ULL;
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=3ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=14ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=16ULL;o_op(9, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(13, 1, _a,_b);})){
v5=2ULL;
}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=18ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=20ULL;o_op(9, 5, _a,_b);});o_op(12, 1, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=24ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=28ULL;o_op(9, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(13, 1, _a,_b);})){
v5=2ULL;
}
{uint64_t _i=v4;uint64_t _v=({uint64_t p2=v5;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v4=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=({uint64_t _a=32ULL;uint64_t _b=({uint64_t _a=v4;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
{uint64_t _i=({uint64_t _a=33ULL;uint64_t _b=({uint64_t _a=v4;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=({uint64_t _a=o_get(v1, v1_n, 33ULL, 5);uint64_t _b=4ULL;o_op(4, 5, _a,_b);});uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v4=0;
uint64_t k1=o_get(v1, v1_n, 38ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=({uint64_t _a=v4;uint64_t _b=48ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=4ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{uint64_t _i=52ULL;uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
{uint64_t _i=53ULL;uint64_t _v=({uint64_t p2=2ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
{
uint64_t v4=0;
uint64_t k1=6ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
{uint64_t _i=({uint64_t _a=v4;uint64_t _b=54ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
return o_leave(0);
}
static uint64_t f30(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
({uint64_t p1=7ULL;uint64_t p2=0ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=856ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=23ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=4ULL;f10(v0, v0_n, v1, v1_n, p2);});
{
uint64_t v3=0;
uint64_t k1=7ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
uint64_t __attribute__((unused)) v4=4ULL;
if(({uint64_t _a=v3;uint64_t _b=3ULL;o_op(8, 5, _a,_b);})){
v4=5ULL;
}
({uint64_t p1=v4;uint64_t p2=o_get(v2, v2_n, v3, 5);uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=896ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=v3;uint64_t _b=3ULL;o_op(11, 5, _a,_b);})){
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=920ULL;uint64_t _b=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=3ULL;o_op(2, 5, _a,_b);});uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
}
({uint64_t p2=2ULL;f10(v0, v0_n, v1, v1_n, p2);});
}
}
{
uint64_t v3=0;
uint64_t k1=3ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
({uint64_t p1=3ULL;uint64_t p2=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=7ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=33ULL;o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=v3;uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=22ULL;uint64_t _v=37ULL;o_put(v1, v1_n, _i,_v,5);}
}
if(({uint64_t _a=v3;uint64_t _b=2ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=22ULL;uint64_t _v=38ULL;o_put(v1, v1_n, _i,_v,5);}
}
({uint64_t p2=2ULL;f10(v0, v0_n, v1, v1_n, p2);});
}
}
({uint64_t p2=2ULL;uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 10ULL, 5);uint64_t p3=o_get(v2, v2_n, 7ULL, 5);uint64_t p4=0ULL;uint64_t p5=5ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 11ULL, 5);uint64_t p3=o_get(v2, v2_n, 7ULL, 5);uint64_t p4=0ULL;uint64_t p5=31ULL;f28(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 13ULL, 5);uint64_t p3=o_get(v2, v2_n, 9ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 14ULL, 5);uint64_t p3=o_get(v2, v2_n, 13ULL, 5);uint64_t p4=o_get(v2, v2_n, 0ULL, 5);uint64_t p5=16ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=56ULL;uint64_t p3=o_get(v2, v2_n, 15ULL, 5);uint64_t p4=o_get(v2, v2_n, 14ULL, 5);uint64_t p5=2ULL;f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=8ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=56ULL;uint64_t p3=o_get(v2, v2_n, 16ULL, 5);uint64_t p4=o_get(v2, v2_n, 14ULL, 5);uint64_t p5=3ULL;f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p1=8ULL;uint64_t p2=o_get(v2, v2_n, 17ULL, 5);uint64_t p3=o_get(v2, v2_n, 9ULL, 5);uint64_t p4=o_get(v2, v2_n, 4ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p2=o_get(v2, v2_n, 18ULL, 5);uint64_t p3=o_get(v2, v2_n, 17ULL, 5);uint64_t p4=o_get(v2, v2_n, 1ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p1=8ULL;uint64_t p2=o_get(v2, v2_n, 17ULL, 5);uint64_t p3=o_get(v2, v2_n, 9ULL, 5);uint64_t p4=o_get(v2, v2_n, 5ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p2=o_get(v2, v2_n, 19ULL, 5);uint64_t p3=o_get(v2, v2_n, 17ULL, 5);uint64_t p4=o_get(v2, v2_n, 2ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 20ULL, 5);uint64_t p3=o_get(v2, v2_n, 11ULL, 5);uint64_t p4=o_get(v2, v2_n, 18ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 33ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 21ULL, 5);uint64_t p3=o_get(v2, v2_n, 11ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=o_get(v1, v1_n, 33ULL, 5);f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 22ULL, 5);uint64_t p3=o_get(v2, v2_n, 11ULL, 5);uint64_t p4=0ULL;uint64_t p5=o_get(v1, v1_n, 39ULL, 5);f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 22ULL, 5);uint64_t p3=o_get(v2, v2_n, 22ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=4ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 23ULL, 5);uint64_t p3=o_get(v2, v2_n, 8ULL, 5);uint64_t p4=o_get(v2, v2_n, 10ULL, 5);uint64_t p5=o_get(v1, v1_n, 35ULL, 5);f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f31(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=o_get(v1, v1_n, 11ULL, 5);
uint64_t __attribute__((unused)) v4=o_get(v1, v1_n, 12ULL, 5);
uint64_t __attribute__((unused)) v5=o_get(v1, v1_n, 13ULL, 5);
if(({uint64_t _a=v5;uint64_t _b=3ULL;o_op(8, 5, _a,_b);})){
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=1ULL;uint64_t _b=({uint64_t _a=20ULL;uint64_t _b=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(2, 5, _a,_b);});o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=o_get(v2, v2_n, 55ULL, 5);uint64_t p4=v3;uint64_t p5=18446744073709551615ULL;f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}else{
({uint64_t p2=o_get(v2, v2_n, 55ULL, 5);uint64_t p3=v3;uint64_t p4=v4;uint64_t p5=4ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
({uint64_t p2=o_get(v2, v2_n, 56ULL, 5);uint64_t p3=o_get(v2, v2_n, 55ULL, 5);uint64_t p4=0ULL;uint64_t p5=({uint64_t _a=127ULL;uint64_t _b=20ULL;o_op(15, 5, _a,_b);});f28(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 57ULL, 5);uint64_t p3=v3;uint64_t p4=v4;uint64_t p5=({uint64_t _a=128ULL;uint64_t _b=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(15, 5, _a,_b);});f28(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=v5;uint64_t _b=3ULL;o_op(8, 5, _a,_b);})){
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=1ULL;uint64_t _b=({uint64_t _a=24ULL;uint64_t _b=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(2, 5, _a,_b);});o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=o_get(v2, v2_n, 58ULL, 5);uint64_t p4=o_get(v2, v2_n, 57ULL, 5);uint64_t p5=o_get(v2, v2_n, 56ULL, 5);f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f32(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 38ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=v3;uint64_t _b=16ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=57ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=48ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, 26ULL, 5);uint64_t p5=o_get(v1, v1_n, 10ULL, 5);f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=10ULL;uint64_t _v=3ULL;o_put(v1, v1_n, _i,_v,5);}
}
}
return o_leave(0);
}
static uint64_t f33(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{uint64_t _i=10ULL;uint64_t _v=2ULL;o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=o_get(v1, v1_n, 36ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
({f32(v0, v0_n, v1, v1_n, v2, v2_n);});
}
({uint64_t p2=o_get(v2, v2_n, 53ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=o_get(v2, v2_n, 25ULL, 5);uint64_t p5=0ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=v3;uint64_t _b=o_get(v1, v1_n, 32ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
uint64_t __attribute__((unused)) v4=56ULL;
if(({uint64_t _a=o_get(v1, v1_n, 33ULL, 5);uint64_t _b=16ULL;o_op(6, 5, _a,_b);})){
v4=57ULL;
}
uint64_t __attribute__((unused)) v5=o_get(v2, v2_n, ({uint64_t _a=33ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);
({uint64_t p2=v4;uint64_t p3=v5;uint64_t p4=o_get(v2, v2_n, 24ULL, 5);uint64_t p5=o_get(v1, v1_n, 10ULL, 5);f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=10ULL;uint64_t _v=3ULL;o_put(v1, v1_n, _i,_v,5);}
}
}
{uint64_t _i=8ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=11ULL;uint64_t p3=o_get(v2, v2_n, 52ULL, 5);uint64_t p4=o_get(v2, v2_n, 53ULL, 5);uint64_t p5=3ULL;f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
if(({uint64_t _a=o_get(v1, v1_n, 36ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
({f32(v0, v0_n, v1, v1_n, v2, v2_n);});
}
return o_leave(0);
}
static uint64_t f34(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=13ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
({f31(v0, v0_n, v1, v1_n, v2, v2_n);});
{uint64_t _i=9ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 54ULL, 5);uint64_t p3=o_get(v2, v2_n, 58ULL, 5);uint64_t p4=o_get(v2, v2_n, ({uint64_t _a=o_get(v1, v1_n, 12ULL, 5);uint64_t _b=48ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p5=o_get(v2, v2_n, 54ULL, 5);f14(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
return o_leave(0);
}
static uint64_t f35(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3){
o_enter();
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
uint64_t __attribute__((unused)) v4=o_get(v2, v2_n, ({uint64_t _a=32ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);
({uint64_t p2=v4;uint64_t p3=o_get(v2, v2_n, 54ULL, 5);uint64_t p4=o_get(v2, v2_n, 52ULL, 5);uint64_t p5=v4;f14(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f36(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p2=1ULL;uint64_t p3=o_get(v2, v2_n, 54ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=0ULL;f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=11ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=33ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
{
uint64_t v4=0;
uint64_t k2=o_get(v1, v1_n, 38ULL, 5);
uint64_t c2=0;
for(;o_op(8,5, v4, k2);v4=o_op(1,5, v4,1)){
if(c2++>=16777216)o_trap(4);
{uint64_t _i=12ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
({f34(v0, v0_n, v1, v1_n, v2, v2_n);});
}
}
({uint64_t p3=v3;f35(v0, v0_n, v1, v1_n, v2, v2_n, p3);});
}
}
return o_leave(0);
}
static uint64_t f37(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v4=o_get(v2, v2_n, ({uint64_t _a=32ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);
{
uint64_t v5=0;
uint64_t k2=5ULL;
for(;o_op(8,5, v5, k2);v5=o_op(1,5, v5,1)){
({uint64_t p1=42ULL;uint64_t p2=o_get(v2, v2_n, 59ULL, 5);uint64_t p3=v4;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=16ULL;uint64_t _b=v5;o_op(16, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=2ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p1=30ULL;uint64_t p2=v4;uint64_t p3=v4;uint64_t p4=o_get(v2, v2_n, 59ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
}
}
}
}
return o_leave(0);
}
static uint64_t f38(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
({f37(v0, v0_n, v1, v1_n, v2, v2_n);});
({uint64_t p2=o_get(v2, v2_n, 27ULL, 5);uint64_t p3=o_get(v2, v2_n, 23ULL, 5);uint64_t p4=o_get(v2, v2_n, 19ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 34ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 11ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);f15(v0, v0_n, v1, v1_n, p2, p3);});
{uint64_t _i=13ULL;uint64_t _v=o_get(v1, v1_n, 0ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p1=54ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=2ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=9ULL;f10(v0, v0_n, v1, v1_n, p2);});
{uint64_t _i=14ULL;uint64_t _v=({uint64_t p2=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t p3=1ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});o_put(v1, v1_n, _i,_v,5);}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p2=o_get(v2, v2_n, 28ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=o_get(v2, v2_n, 27ULL, 5);uint64_t p5=({uint64_t _a=v3;uint64_t _b=4ULL;o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p1=12ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=o_get(v2, v2_n, 28ULL, 5);uint64_t p4=o_get(v2, v2_n, ({uint64_t _a=32ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=21ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=5ULL;f10(v0, v0_n, v1, v1_n, p2);});
}
}
{uint64_t _i=15ULL;uint64_t _v=o_get(v1, v1_n, 0ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p1=55ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=10ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p1=o_get(v1, v1_n, 14ULL, 5);uint64_t p2=o_get(v1, v1_n, 15ULL, 5);f17(v0, v0_n, p1, p2);});
({uint64_t p1=o_get(v1, v1_n, 13ULL, 5);uint64_t p2=({uint64_t _a=o_get(v1, v1_n, 15ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});f17(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f39(const void *v0, uint64_t v0_n){
o_enter();
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=32768ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=8ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=16ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=2ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=4ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=8ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(12, 1, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=32ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=1ULL;o_op(10, 5, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v1=0;
uint64_t k1=3ULL;
for(;o_op(8,5, v1, k1);v1=o_op(1,5, v1,1)){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=5ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
}
}
return o_leave(0ULL);
}
static uint64_t f40(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
({f30(v0, v0_n, v1, v1_n, v2, v2_n);});
{uint64_t _i=41ULL;uint64_t _v=o_get(v1, v1_n, 0ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 23ULL, 5);uint64_t p3=o_get(v2, v2_n, 3ULL, 5);f15(v0, v0_n, v1, v1_n, p2, p3);});
{uint64_t _i=42ULL;uint64_t _v=({uint64_t p2=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t p3=1ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 24ULL, 5);uint64_t p3=o_get(v2, v2_n, 23ULL, 5);uint64_t p4=o_get(v2, v2_n, 15ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 34ULL, 5);uint64_t _b=o_get(v1, v1_n, 32ULL, 5);o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 24ULL, 5);uint64_t p3=o_get(v2, v2_n, 21ULL, 5);uint64_t p4=o_get(v2, v2_n, 24ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 29ULL, 5);uint64_t p3=o_get(v2, v2_n, 23ULL, 5);uint64_t p4=0ULL;uint64_t p5=o_get(v1, v1_n, 40ULL, 5);f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 25ULL, 5);uint64_t p3=o_get(v2, v2_n, 29ULL, 5);uint64_t p4=o_get(v2, v2_n, 16ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 32ULL, 5);uint64_t _b=32ULL;o_op(4, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 25ULL, 5);uint64_t p3=o_get(v2, v2_n, 22ULL, 5);uint64_t p4=o_get(v2, v2_n, 25ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 26ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=o_get(v2, v2_n, 20ULL, 5);uint64_t p5=0ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p2=1ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=32ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=0ULL;f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
({uint64_t p2=2ULL;uint64_t p3=o_get(v2, v2_n, 30ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=o_get(v1, v1_n, 37ULL, 5);f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=43ULL;uint64_t _v=o_get(v1, v1_n, 0ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({f33(v0, v0_n, v1, v1_n, v2, v2_n);});
({f36(v0, v0_n, v1, v1_n, v2, v2_n);});
({uint64_t p2=o_get(v2, v2_n, 24ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=o_get(v2, v2_n, 24ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 33ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 26ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=o_get(v2, v2_n, 26ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 33ULL, 5);uint64_t _b=128ULL;o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 25ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=o_get(v2, v2_n, 25ULL, 5);uint64_t p5=o_get(v1, v1_n, 33ULL, 5);f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 30ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=o_get(v2, v2_n, 30ULL, 5);uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 30ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);f15(v0, v0_n, v1, v1_n, p2, p3);});
uint64_t __attribute__((unused)) v3=({uint64_t p2=o_get(v1, v1_n, 43ULL, 5);uint64_t p3=1ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});
({f38(v0, v0_n, v1, v1_n, v2, v2_n);});
({uint64_t p2=o_get(v2, v2_n, 23ULL, 5);uint64_t p3=o_get(v2, v2_n, 6ULL, 5);uint64_t p4=o_get(v2, v2_n, 23ULL, 5);uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
v3=({uint64_t p2=o_get(v1, v1_n, 41ULL, 5);uint64_t p3=0ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});
({uint64_t p1=o_get(v1, v1_n, 42ULL, 5);uint64_t p2=o_get(v1, v1_n, 0ULL, 5);f17(v0, v0_n, p1, p2);});
({uint64_t p1=45ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=7ULL;f10(v0, v0_n, v1, v1_n, p2);});
v3=({uint64_t p2=o_get(v1, v1_n, 0ULL, 5);uint64_t p3=0ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});
{uint64_t _i=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=6ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t _a=1032192ULL;uint64_t _b=32ULL;o_op(15, 5, _a,_b);});o_put(v0, v0_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f41(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n){
o_enter();
if(({uint64_t _a=({f39(v0, v0_n);});uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(1ULL);
}
uint64_t v4[64]={0};
uint64_t v4_n=64;
o_fill(v4, v4_n, 0ULL, 5);
uint64_t v5[64]={0};
uint64_t v5_n=64;
o_fill(v5, v5_n, 0ULL, 5);
{
uint64_t v6=0;
uint64_t k1=2048ULL;
for(;o_op(8,5, v6, k1);v6=o_op(1,5, v6,1)){
{uint64_t _i=v6;uint64_t _v=0ULL;o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v6=0;
uint64_t k1=8192ULL;
for(;o_op(8,5, v6, k1);v6=o_op(1,5, v6,1)){
{uint64_t _i=v6;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
}
}
{
uint64_t v6=0;
uint64_t k1=5ULL;
for(;o_op(8,5, v6, k1);v6=o_op(1,5, v6,1)){
{uint64_t _i=({uint64_t _a=v6;uint64_t _b=32ULL;o_op(1, 5, _a,_b);});uint64_t _v=o_get(v0, v0_n, v6, 5);o_put(v4, v4_n, _i,_v,5);}
}
}
{uint64_t _i=37ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(4, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=38ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=4ULL;o_op(4, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=39ULL;uint64_t _v=({uint64_t p0=({uint64_t _a=128ULL;uint64_t _b=o_get(v0, v0_n, 1ULL, 5);o_op(4, 5, _a,_b);});f24(p0);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=40ULL;uint64_t _v=({uint64_t p0=({uint64_t _a=128ULL;uint64_t _b=o_get(v0, v0_n, 2ULL, 5);o_op(4, 5, _a,_b);});f24(p0);});o_put(v4, v4_n, _i,_v,5);}
({f29(v2, v2_n, v4, v4_n, v5, v5_n);});
({f40(v1, v1_n, v4, v4_n, v5, v5_n);});
if(({uint64_t _a=o_get(v4, v4_n, 3ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(2ULL);
}
{uint64_t _i=0ULL;uint64_t _v=o_get(v4, v4_n, 0ULL, 5);o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=o_get(v4, v4_n, 4ULL, 5);o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=o_get(v4, v4_n, 5ULL, 5);o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=3ULL;uint64_t _v=0ULL;o_put(v3, v3_n, _i,_v,5);}
return o_leave(0ULL);
}
static uint64_t f42(const void *v0, uint64_t v0_n){
o_enter();
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=32768ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=8ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=16ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=2ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=4ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=8ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(12, 1, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=32ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v1=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v1, k1);v1=o_op(1,5, v1,1)){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=4ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
}
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
uint64_t __attribute__((unused)) v1=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(4, 5, _a,_b);});
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=16ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=v1;o_op(3, 5, _a,_b);});uint64_t _b=o_get(v0, v0_n, 1ULL, 5);o_op(3, 5, _a,_b);});uint64_t _b=128ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=512ULL;o_op(3, 5, _a,_b);});o_op(10, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=1ULL;o_op(2, 5, _a,_b);});uint64_t _b=o_get(v0, v0_n, 0ULL, 5);o_op(3, 5, _a,_b);});uint64_t _b=o_get(v0, v0_n, 1ULL, 5);o_op(1, 5, _a,_b);});uint64_t _b=({uint64_t _a=1ULL;uint64_t _b=23ULL;o_op(15, 5, _a,_b);});o_op(11, 5, _a,_b);})){
return o_leave(1ULL);
}
return o_leave(0ULL);
}
static uint64_t f43(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=8192ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=v3;uint64_t _v=0ULL;o_put(v0, v0_n, _i,_v,5);}
}
}
{
uint64_t v3=0;
uint64_t k1=2048ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=v3;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
}
}
{
uint64_t v3=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=v3;uint64_t _v=0ULL;o_put(v2, v2_n, _i,_v,5);}
}
}
return o_leave(0);
}
static uint64_t f44(uint64_t v0, uint64_t v1){
o_enter();
return o_leave(({uint64_t _a=64ULL;uint64_t _b=({uint64_t _a=({uint64_t _a=({uint64_t _a=v0;uint64_t _b=16ULL;o_op(3, 5, _a,_b);});uint64_t _b=v1;o_op(1, 5, _a,_b);});uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}));
}
static uint64_t f45(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, uint64_t v3){
o_enter();
uint64_t __attribute__((unused)) v4=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});
{uint64_t _i=({uint64_t _a=32ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _v=v4;o_put(v2, v2_n, _i,_v,5);}
v4=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});
{uint64_t _i=({uint64_t _a=33ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _v=v4;o_put(v2, v2_n, _i,_v,5);}
{
uint64_t v5=0;
uint64_t k1=o_get(v1, v1_n, 36ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v5, k1);v5=o_op(1,5, v5,1)){
if(c1++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v6=({uint64_t p0=v3;uint64_t p1=v5;f44(p0, p1);});
v4=({uint64_t p2=o_get(v1, v1_n, 38ULL, 5);uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});
{uint64_t _i=v6;uint64_t _v=v4;o_put(v2, v2_n, _i,_v,5);}
v4=({uint64_t p2=o_get(v1, v1_n, 38ULL, 5);uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});
{uint64_t _i=({uint64_t _a=v6;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t _v=v4;o_put(v2, v2_n, _i,_v,5);}
}
}
return o_leave(0);
}
static uint64_t f46(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=({uint64_t p2=2ULL;uint64_t p3=1ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});
{
uint64_t v4=0;
uint64_t k1=24ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
uint64_t __attribute__((unused)) v5=1ULL;
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=3ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=14ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=16ULL;o_op(9, 5, _a,_b);});o_op(12, 1, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=18ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=19ULL;o_op(9, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(13, 1, _a,_b);});o_op(13, 1, _a,_b);})){
v5=2ULL;
}
{uint64_t _i=v4;uint64_t _v=({uint64_t p2=v5;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{uint64_t _i=24ULL;uint64_t _v=({uint64_t p2=4ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
{
uint64_t v4=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p3=v4;f45(v0, v0_n, v1, v1_n, v2, v2_n, p3);});
}
}
{
uint64_t v4=0;
uint64_t k1=o_get(v1, v1_n, 36ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=({uint64_t _a=320ULL;uint64_t _b=({uint64_t _a=v4;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
{uint64_t _i=({uint64_t _a=321ULL;uint64_t _b=({uint64_t _a=v4;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v4=0;
uint64_t k1=7ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
uint64_t __attribute__((unused)) v5=1ULL;
if(({uint64_t _a=v4;uint64_t _b=5ULL;o_op(8, 5, _a,_b);})){
v5=2ULL;
}
{uint64_t _i=({uint64_t _a=v4;uint64_t _b=25ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=v5;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v4=0;
uint64_t k1=5ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
{uint64_t _i=({uint64_t _a=v4;uint64_t _b=48ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v4=0;
uint64_t k1=9ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
uint64_t __attribute__((unused)) v5=1ULL;
if(({uint64_t _a=v4;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
v5=2ULL;
}
{uint64_t _i=({uint64_t _a=v4;uint64_t _b=352ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=v5;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v4=0;
uint64_t k1=o_get(v1, v1_n, 41ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=({uint64_t _a=384ULL;uint64_t _b=({uint64_t _a=v4;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=2ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
{uint64_t _i=({uint64_t _a=385ULL;uint64_t _b=({uint64_t _a=v4;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
return o_leave(0);
}
static uint64_t f47(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
({uint64_t p1=7ULL;uint64_t p2=0ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=856ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=23ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=4ULL;f10(v0, v0_n, v1, v1_n, p2);});
{
uint64_t v3=0;
uint64_t k1=7ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
uint64_t __attribute__((unused)) v4=4ULL;
if(({uint64_t _a=v3;uint64_t _b=3ULL;o_op(8, 5, _a,_b);})){
v4=5ULL;
}
({uint64_t p1=v4;uint64_t p2=o_get(v2, v2_n, v3, 5);uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=896ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=v3;uint64_t _b=3ULL;o_op(11, 5, _a,_b);})){
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=920ULL;uint64_t _b=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=3ULL;o_op(2, 5, _a,_b);});uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
}
({uint64_t p2=2ULL;f10(v0, v0_n, v1, v1_n, p2);});
}
}
{
uint64_t v3=0;
uint64_t k1=3ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
({uint64_t p1=3ULL;uint64_t p2=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=7ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=33ULL;o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=v3;uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=22ULL;uint64_t _v=37ULL;o_put(v1, v1_n, _i,_v,5);}
}
if(({uint64_t _a=v3;uint64_t _b=2ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=22ULL;uint64_t _v=38ULL;o_put(v1, v1_n, _i,_v,5);}
}
({uint64_t p2=2ULL;f10(v0, v0_n, v1, v1_n, p2);});
}
}
({uint64_t p2=2ULL;uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 10ULL, 5);uint64_t p3=o_get(v2, v2_n, 7ULL, 5);uint64_t p4=0ULL;uint64_t p5=5ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 11ULL, 5);uint64_t p3=o_get(v2, v2_n, 7ULL, 5);uint64_t p4=0ULL;uint64_t p5=31ULL;f28(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 13ULL, 5);uint64_t p3=o_get(v2, v2_n, 9ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 14ULL, 5);uint64_t p3=o_get(v2, v2_n, 13ULL, 5);uint64_t p4=o_get(v2, v2_n, 0ULL, 5);uint64_t p5=16ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=56ULL;uint64_t p3=o_get(v2, v2_n, 15ULL, 5);uint64_t p4=o_get(v2, v2_n, 14ULL, 5);uint64_t p5=2ULL;f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=8ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=56ULL;uint64_t p3=o_get(v2, v2_n, 16ULL, 5);uint64_t p4=o_get(v2, v2_n, 14ULL, 5);uint64_t p5=3ULL;f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p1=8ULL;uint64_t p2=o_get(v2, v2_n, 17ULL, 5);uint64_t p3=o_get(v2, v2_n, 9ULL, 5);uint64_t p4=o_get(v2, v2_n, 4ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p2=o_get(v2, v2_n, 18ULL, 5);uint64_t p3=o_get(v2, v2_n, 17ULL, 5);uint64_t p4=o_get(v2, v2_n, 1ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p1=8ULL;uint64_t p2=o_get(v2, v2_n, 17ULL, 5);uint64_t p3=o_get(v2, v2_n, 9ULL, 5);uint64_t p4=o_get(v2, v2_n, 5ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p2=o_get(v2, v2_n, 19ULL, 5);uint64_t p3=o_get(v2, v2_n, 17ULL, 5);uint64_t p4=o_get(v2, v2_n, 2ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 20ULL, 5);uint64_t p3=o_get(v2, v2_n, 11ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=o_get(v1, v1_n, 33ULL, 5);f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 21ULL, 5);uint64_t p3=o_get(v2, v2_n, 11ULL, 5);uint64_t p4=0ULL;uint64_t p5=o_get(v1, v1_n, 39ULL, 5);f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 21ULL, 5);uint64_t p3=o_get(v2, v2_n, 21ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=4ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 22ULL, 5);uint64_t p3=o_get(v2, v2_n, 8ULL, 5);uint64_t p4=o_get(v2, v2_n, 10ULL, 5);uint64_t p5=o_get(v1, v1_n, 35ULL, 5);f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 23ULL, 5);uint64_t p3=o_get(v2, v2_n, 22ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f48(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
({uint64_t p2=o_get(v2, v2_n, 356ULL, 5);uint64_t p3=o_get(v2, v2_n, 7ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 41ULL, 5);uint64_t _b=1ULL;o_op(2, 5, _a,_b);});uint64_t _b=32ULL;o_op(3, 5, _a,_b);});uint64_t _b=o_get(v1, v1_n, 35ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=o_get(v2, v2_n, 356ULL, 5);uint64_t p4=o_get(v2, v2_n, 12ULL, 5);uint64_t p5=o_get(v2, v2_n, 356ULL, 5);f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 358ULL, 5);uint64_t p3=o_get(v2, v2_n, 356ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 358ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=o_get(v2, v2_n, 358ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 32ULL, 5);uint64_t _b=1ULL;o_op(2, 5, _a,_b);});f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 357ULL, 5);uint64_t p3=o_get(v2, v2_n, 358ULL, 5);uint64_t p4=0ULL;uint64_t p5=31ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 358ULL, 5);uint64_t p3=o_get(v2, v2_n, 356ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=0ULL;uint64_t _b=o_get(v1, v1_n, 32ULL, 5);o_op(2, 5, _a,_b);});uint64_t _b=4294967295ULL;o_op(12, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=o_get(v2, v2_n, 358ULL, 5);uint64_t p4=o_get(v2, v2_n, 12ULL, 5);uint64_t p5=o_get(v2, v2_n, 358ULL, 5);f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p1=8ULL;uint64_t p2=o_get(v2, v2_n, 359ULL, 5);uint64_t p3=o_get(v2, v2_n, 357ULL, 5);uint64_t p4=o_get(v2, v2_n, 358ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p2=o_get(v2, v2_n, 359ULL, 5);uint64_t p3=o_get(v2, v2_n, 359ULL, 5);uint64_t p4=o_get(v2, v2_n, 356ULL, 5);uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 360ULL, 5);uint64_t p3=o_get(v2, v2_n, 358ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 360ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=o_get(v2, v2_n, 360ULL, 5);uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p1=8ULL;uint64_t p2=o_get(v2, v2_n, 360ULL, 5);uint64_t p3=o_get(v2, v2_n, 357ULL, 5);uint64_t p4=o_get(v2, v2_n, 360ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p2=o_get(v2, v2_n, 360ULL, 5);uint64_t p3=o_get(v2, v2_n, 360ULL, 5);uint64_t p4=o_get(v2, v2_n, 356ULL, 5);uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, ({uint64_t _a=384ULL;uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 41ULL, 5);uint64_t _b=1ULL;o_op(2, 5, _a,_b);});uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);uint64_t p3=o_get(v2, v2_n, 360ULL, 5);uint64_t p4=o_get(v2, v2_n, 18ULL, 5);uint64_t p5=4ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 355ULL, 5);uint64_t p3=o_get(v2, v2_n, 359ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=4ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f49(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{uint64_t _i=49ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 32ULL, 5);uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 35ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(5, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 352ULL, 5);uint64_t p3=o_get(v2, v2_n, 7ULL, 5);uint64_t p4=o_get(v2, v2_n, 18ULL, 5);uint64_t p5=4ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 353ULL, 5);uint64_t p3=o_get(v2, v2_n, 7ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=4ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 41ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=128ULL;o_op(3, 5, _a,_b);});uint64_t _b=o_get(v1, v1_n, 35ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=9ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=384ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, 12ULL, 5);uint64_t p5=o_get(v2, v2_n, 352ULL, 5);f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
if(({uint64_t _a=o_get(v1, v1_n, 49ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
({f48(v0, v0_n, v1, v1_n, v2, v2_n);});
}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 41ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=8ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
uint64_t __attribute__((unused)) v4=3ULL;
if(({uint64_t _a=v3;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
v4=2ULL;
}
{uint64_t _i=10ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=385ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=11ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=384ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=11ULL;uint64_t p3=o_get(v1, v1_n, 10ULL, 5);uint64_t p4=o_get(v1, v1_n, 11ULL, 5);uint64_t p5=v4;f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 41ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
if(({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 49ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=v3;uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 41ULL, 5);uint64_t _b=1ULL;o_op(2, 5, _a,_b);});o_op(6, 5, _a,_b);});o_op(12, 1, _a,_b);})){
({uint64_t p1=50ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=o_get(v2, v2_n, 355ULL, 5);uint64_t p4=o_get(v2, v2_n, ({uint64_t _a=385ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=5ULL;f10(v0, v0_n, v1, v1_n, p2);});
}else{
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=128ULL;o_op(3, 5, _a,_b);});uint64_t _b=o_get(v1, v1_n, 35ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=o_get(v2, v2_n, 354ULL, 5);uint64_t p4=o_get(v2, v2_n, 12ULL, 5);uint64_t p5=o_get(v2, v2_n, 353ULL, 5);f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p1=50ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=o_get(v2, v2_n, 354ULL, 5);uint64_t p4=o_get(v2, v2_n, ({uint64_t _a=385ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=5ULL;f10(v0, v0_n, v1, v1_n, p2);});
}
}
}
({uint64_t p1=48ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=6ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p2=o_get(v2, v2_n, 31ULL, 5);uint64_t p3=o_get(v2, v2_n, 11ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 33ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f50(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 51ULL, 5);uint64_t _b=o_get(v1, v1_n, 32ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=12ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 50ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=12ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 12ULL, 5);uint64_t _b=o_get(v1, v1_n, 33ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=o_get(v1, v1_n, 12ULL, 5);o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=12ULL;uint64_t _v=({uint64_t p0=o_get(v1, v1_n, 51ULL, 5);uint64_t p1=o_get(v1, v1_n, 50ULL, 5);f44(p0, p1);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=12ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=o_get(v1, v1_n, 12ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v1, v1_n, 11ULL, 5);uint64_t p3=o_get(v1, v1_n, 12ULL, 5);uint64_t p4=o_get(v2, v2_n, 25ULL, 5);uint64_t p5=o_get(v1, v1_n, 10ULL, 5);f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=10ULL;uint64_t _v=12ULL;o_put(v1, v1_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f51(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
({uint64_t p2=o_get(v2, v2_n, 25ULL, 5);uint64_t p3=o_get(v2, v2_n, 22ULL, 5);uint64_t p4=o_get(v2, v2_n, 15ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 34ULL, 5);uint64_t _b=o_get(v1, v1_n, 32ULL, 5);o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 25ULL, 5);uint64_t p3=o_get(v2, v2_n, 20ULL, 5);uint64_t p4=o_get(v2, v2_n, 25ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 30ULL, 5);uint64_t p3=o_get(v2, v2_n, 22ULL, 5);uint64_t p4=0ULL;uint64_t p5=o_get(v1, v1_n, 40ULL, 5);f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 26ULL, 5);uint64_t p3=o_get(v2, v2_n, 30ULL, 5);uint64_t p4=o_get(v2, v2_n, 16ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 32ULL, 5);uint64_t _b=32ULL;o_op(4, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 26ULL, 5);uint64_t p3=o_get(v2, v2_n, 21ULL, 5);uint64_t p4=o_get(v2, v2_n, 26ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=10ULL;uint64_t _v=11ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=11ULL;uint64_t _v=56ULL;o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=o_get(v1, v1_n, 33ULL, 5);uint64_t _b=16ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=11ULL;uint64_t _v=57ULL;o_put(v1, v1_n, _i,_v,5);}
}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 36ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{
uint64_t v4=0;
uint64_t k2=o_get(v1, v1_n, 34ULL, 5);
uint64_t c2=0;
for(;o_op(8,5, v4, k2);v4=o_op(1,5, v4,1)){
if(c2++>=16777216)o_trap(4);
{uint64_t _i=50ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=51ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
({f50(v0, v0_n, v1, v1_n, v2, v2_n);});
}
}
}
}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 36ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
if(({uint64_t _a=v3;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=8ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=11ULL;uint64_t p3=o_get(v2, v2_n, 321ULL, 5);uint64_t p4=o_get(v2, v2_n, 26ULL, 5);uint64_t p5=o_get(v1, v1_n, 10ULL, 5);f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}else{
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=o_get(v1, v1_n, 37ULL, 5);o_op(3, 5, _a,_b);});uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=9ULL;uint64_t p3=o_get(v2, v2_n, 27ULL, 5);uint64_t p4=o_get(v2, v2_n, 12ULL, 5);uint64_t p5=o_get(v2, v2_n, 26ULL, 5);f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=11ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=321ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, 27ULL, 5);uint64_t p5=12ULL;f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
{uint64_t _i=10ULL;uint64_t _v=12ULL;o_put(v1, v1_n, _i,_v,5);}
}
}
({uint64_t p2=o_get(v2, v2_n, 22ULL, 5);uint64_t p3=o_get(v2, v2_n, 6ULL, 5);uint64_t p4=o_get(v2, v2_n, 22ULL, 5);uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f52(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=({uint64_t p0=o_get(v1, v1_n, 51ULL, 5);uint64_t p1=o_get(v1, v1_n, 50ULL, 5);f44(p0, p1);});
{
uint64_t v4=0;
uint64_t k1=o_get(v1, v1_n, 38ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p1=8ULL;uint64_t p2=o_get(v2, v2_n, v3, 5);uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=23ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=16ULL;o_op(15, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=24ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v1, v1_n, 10ULL, 5);f10(v0, v0_n, v1, v1_n, p2);});
{uint64_t _i=10ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
}
}
return o_leave(0);
}
static uint64_t f53(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{uint64_t _i=10ULL;uint64_t _v=13ULL;o_put(v1, v1_n, _i,_v,5);}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 36ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{
uint64_t v4=0;
uint64_t k2=o_get(v1, v1_n, 34ULL, 5);
uint64_t c2=0;
for(;o_op(8,5, v4, k2);v4=o_op(1,5, v4,1)){
if(c2++>=16777216)o_trap(4);
{uint64_t _i=50ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=51ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
({f52(v0, v0_n, v1, v1_n, v2, v2_n);});
}
}
}
}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 36ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=10ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=320ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=11ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=321ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v1, v1_n, 10ULL, 5);uint64_t p3=o_get(v1, v1_n, 11ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
return o_leave(0);
}
static uint64_t f54(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
if(({uint64_t _a=v5;uint64_t _b=3ULL;o_op(8, 5, _a,_b);})){
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=1ULL;uint64_t _b=({uint64_t _a=20ULL;uint64_t _b=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(2, 5, _a,_b);});o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=o_get(v2, v2_n, 48ULL, 5);uint64_t p4=v3;uint64_t p5=18446744073709551615ULL;f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}else{
({uint64_t p2=o_get(v2, v2_n, 48ULL, 5);uint64_t p3=v3;uint64_t p4=v4;uint64_t p5=4ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
({uint64_t p2=o_get(v2, v2_n, 49ULL, 5);uint64_t p3=o_get(v2, v2_n, 48ULL, 5);uint64_t p4=0ULL;uint64_t p5=({uint64_t _a=127ULL;uint64_t _b=20ULL;o_op(15, 5, _a,_b);});f28(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 50ULL, 5);uint64_t p3=v3;uint64_t p4=v4;uint64_t p5=({uint64_t _a=128ULL;uint64_t _b=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(15, 5, _a,_b);});f28(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=v5;uint64_t _b=3ULL;o_op(8, 5, _a,_b);})){
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=1ULL;uint64_t _b=({uint64_t _a=24ULL;uint64_t _b=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(2, 5, _a,_b);});o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=o_get(v2, v2_n, 51ULL, 5);uint64_t p4=o_get(v2, v2_n, 50ULL, 5);uint64_t p5=o_get(v2, v2_n, 49ULL, 5);f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f55(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{uint64_t _i=53ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t p0=o_get(v1, v1_n, 51ULL, 5);uint64_t p1=o_get(v1, v1_n, 50ULL, 5);f44(p0, p1);}), 5);o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=54ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=33ULL;uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 51ULL, 5);uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
{
uint64_t v3=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
({uint64_t p3=o_get(v1, v1_n, 53ULL, 5);uint64_t p4=o_get(v1, v1_n, 52ULL, 5);uint64_t p5=v3;f54(v0, v0_n, v1, v1_n, v2, v2_n, p3, p4, p5);});
{uint64_t _i=9ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v1, v1_n, 54ULL, 5);uint64_t p3=o_get(v2, v2_n, 51ULL, 5);uint64_t p4=o_get(v2, v2_n, 24ULL, 5);uint64_t p5=o_get(v1, v1_n, 54ULL, 5);f14(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
return o_leave(0);
}
static uint64_t f56(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{uint64_t _i=57ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 50ULL, 5);uint64_t _b=128ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=57ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 57ULL, 5);uint64_t _b=o_get(v1, v1_n, 33ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=56ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=320ULL;uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 50ULL, 5);uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p2=1ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=33ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=0ULL;f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 38ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=52ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p1=58ULL;uint64_t p2=o_get(v2, v2_n, 24ULL, 5);uint64_t p3=o_get(v2, v2_n, 31ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=v3;uint64_t _b=16ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 22ULL, 5);uint64_t _b=o_get(v1, v1_n, 57ULL, 5);o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=2ULL;f10(v0, v0_n, v1, v1_n, p2);});
{
uint64_t v4=0;
uint64_t k2=o_get(v1, v1_n, 34ULL, 5);
uint64_t c2=0;
for(;o_op(8,5, v4, k2);v4=o_op(1,5, v4,1)){
if(c2++>=16777216)o_trap(4);
{uint64_t _i=51ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
({f55(v0, v0_n, v1, v1_n, v2, v2_n);});
}
}
}
}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=54ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=32ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=55ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=33ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v1, v1_n, 54ULL, 5);uint64_t p3=o_get(v1, v1_n, 55ULL, 5);uint64_t p4=o_get(v1, v1_n, 56ULL, 5);uint64_t p5=o_get(v1, v1_n, 54ULL, 5);f14(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
return o_leave(0);
}
static uint64_t f57(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p2=1ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=32ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=0ULL;f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 36ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=50ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
({f56(v0, v0_n, v1, v1_n, v2, v2_n);});
}
}
return o_leave(0);
}
static uint64_t f58(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=54ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=32ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
{
uint64_t v4=0;
uint64_t k2=5ULL;
for(;o_op(8,5, v4, k2);v4=o_op(1,5, v4,1)){
({uint64_t p1=42ULL;uint64_t p2=o_get(v2, v2_n, 52ULL, 5);uint64_t p3=o_get(v1, v1_n, 54ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=16ULL;uint64_t _b=v4;o_op(16, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=2ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p1=30ULL;uint64_t p2=o_get(v1, v1_n, 54ULL, 5);uint64_t p3=o_get(v1, v1_n, 54ULL, 5);uint64_t p4=o_get(v2, v2_n, 52ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
}
}
}
}
({uint64_t p2=o_get(v2, v2_n, 28ULL, 5);uint64_t p3=o_get(v2, v2_n, 23ULL, 5);uint64_t p4=o_get(v2, v2_n, 19ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 34ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 11ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);f15(v0, v0_n, v1, v1_n, p2, p3);});
{uint64_t _i=45ULL;uint64_t _v=o_get(v1, v1_n, 0ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p1=54ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=2ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=9ULL;f10(v0, v0_n, v1, v1_n, p2);});
{uint64_t _i=46ULL;uint64_t _v=({uint64_t p2=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t p3=1ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});o_put(v1, v1_n, _i,_v,5);}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p2=o_get(v2, v2_n, 29ULL, 5);uint64_t p3=o_get(v2, v2_n, 12ULL, 5);uint64_t p4=o_get(v2, v2_n, 28ULL, 5);uint64_t p5=({uint64_t _a=v3;uint64_t _b=4ULL;o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p1=12ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=o_get(v2, v2_n, 29ULL, 5);uint64_t p4=o_get(v2, v2_n, ({uint64_t _a=32ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}), 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=21ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=5ULL;f10(v0, v0_n, v1, v1_n, p2);});
}
}
{uint64_t _i=47ULL;uint64_t _v=o_get(v1, v1_n, 0ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p1=55ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=10ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p1=o_get(v1, v1_n, 46ULL, 5);uint64_t p2=o_get(v1, v1_n, 47ULL, 5);f17(v0, v0_n, p1, p2);});
({uint64_t p1=o_get(v1, v1_n, 45ULL, 5);uint64_t p2=({uint64_t _a=o_get(v1, v1_n, 47ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});f17(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f59(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n){
o_enter();
uint64_t v4[64]={0};
uint64_t v4_n=64;
o_fill(v4, v4_n, 0ULL, 5);
uint64_t v5[512]={0};
uint64_t v5_n=512;
o_fill(v5, v5_n, 0ULL, 5);
({f43(v1, v1_n, v2, v2_n, v3, v3_n);});
if(({uint64_t _a=({f42(v0, v0_n);});uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v6=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v6, k1);v6=o_op(1,5, v6,1)){
{uint64_t _i=({uint64_t _a=v6;uint64_t _b=32ULL;o_op(1, 5, _a,_b);});uint64_t _v=o_get(v0, v0_n, v6, 5);o_put(v4, v4_n, _i,_v,5);}
}
}
{uint64_t _i=36ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(4, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=37ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});uint64_t _b=128ULL;o_op(4, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=38ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=4ULL;o_op(4, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=39ULL;uint64_t _v=({uint64_t p0=({uint64_t _a=128ULL;uint64_t _b=o_get(v0, v0_n, 1ULL, 5);o_op(4, 5, _a,_b);});f24(p0);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=40ULL;uint64_t _v=({uint64_t p0=({uint64_t _a=128ULL;uint64_t _b=o_get(v0, v0_n, 2ULL, 5);o_op(4, 5, _a,_b);});f24(p0);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=10ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=41ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=o_get(v4, v4_n, 10ULL, 5);o_op(1, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=41ULL;uint64_t _v=({uint64_t _a=o_get(v4, v4_n, 41ULL, 5);uint64_t _b=1ULL;o_op(2, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=41ULL;uint64_t _v=({uint64_t _a=o_get(v4, v4_n, 41ULL, 5);uint64_t _b=o_get(v4, v4_n, 10ULL, 5);o_op(4, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
({f46(v2, v2_n, v4, v4_n, v5, v5_n);});
({f47(v1, v1_n, v4, v4_n, v5, v5_n);});
({f49(v1, v1_n, v4, v4_n, v5, v5_n);});
({uint64_t p2=o_get(v5, v5_n, 22ULL, 5);uint64_t p3=o_get(v5, v5_n, 3ULL, 5);f15(v1, v1_n, v4, v4_n, p2, p3);});
{uint64_t _i=42ULL;uint64_t _v=({uint64_t p2=({uint64_t _a=o_get(v4, v4_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t p3=1ULL;f16(v1, v1_n, v4, v4_n, p2, p3);});o_put(v4, v4_n, _i,_v,5);}
({f51(v1, v1_n, v4, v4_n, v5, v5_n);});
{uint64_t _i=43ULL;uint64_t _v=o_get(v4, v4_n, 0ULL, 5);o_put(v4, v4_n, _i,_v,5);}
({f53(v1, v1_n, v4, v4_n, v5, v5_n);});
({uint64_t p2=o_get(v5, v5_n, 22ULL, 5);uint64_t p3=o_get(v5, v5_n, 3ULL, 5);f15(v1, v1_n, v4, v4_n, p2, p3);});
{uint64_t _i=44ULL;uint64_t _v=({uint64_t p2=({uint64_t _a=o_get(v4, v4_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t p3=1ULL;f16(v1, v1_n, v4, v4_n, p2, p3);});o_put(v4, v4_n, _i,_v,5);}
({f51(v1, v1_n, v4, v4_n, v5, v5_n);});
({uint64_t p1=o_get(v4, v4_n, 44ULL, 5);uint64_t p2=o_get(v4, v4_n, 0ULL, 5);f17(v1, v1_n, p1, p2);});
({f57(v1, v1_n, v4, v4_n, v5, v5_n);});
({f58(v1, v1_n, v4, v4_n, v5, v5_n);});
({uint64_t p2=o_get(v5, v5_n, 23ULL, 5);uint64_t p3=o_get(v5, v5_n, 6ULL, 5);uint64_t p4=o_get(v5, v5_n, 23ULL, 5);uint64_t p5=1ULL;f25(v1, v1_n, v4, v4_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v5, v5_n, 23ULL, 5);uint64_t p3=o_get(v5, v5_n, 3ULL, 5);f15(v1, v1_n, v4, v4_n, p2, p3);});
{uint64_t _i=48ULL;uint64_t _v=({uint64_t p2=({uint64_t _a=o_get(v4, v4_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t p3=1ULL;f16(v1, v1_n, v4, v4_n, p2, p3);});o_put(v4, v4_n, _i,_v,5);}
uint64_t __attribute__((unused)) v6=({uint64_t p2=o_get(v4, v4_n, 43ULL, 5);uint64_t p3=0ULL;f16(v1, v1_n, v4, v4_n, p2, p3);});
({uint64_t p1=o_get(v4, v4_n, 42ULL, 5);uint64_t p2=o_get(v4, v4_n, 0ULL, 5);f17(v1, v1_n, p1, p2);});
({uint64_t p1=o_get(v4, v4_n, 48ULL, 5);uint64_t p2=o_get(v4, v4_n, 0ULL, 5);f17(v1, v1_n, p1, p2);});
({uint64_t p1=45ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v4, v4_n, p1, p2, p3, p4, p5);});
({uint64_t p2=7ULL;f10(v1, v1_n, v4, v4_n, p2);});
v6=({uint64_t p2=o_get(v4, v4_n, 0ULL, 5);uint64_t p3=0ULL;f16(v1, v1_n, v4, v4_n, p2, p3);});
{uint64_t _i=({uint64_t _a=({uint64_t _a=v6;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=6ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t _a=1032192ULL;uint64_t _b=32ULL;o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=o_get(v4, v4_n, 3ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
({f43(v1, v1_n, v2, v2_n, v3, v3_n);});
return o_leave(2ULL);
}
{uint64_t _i=0ULL;uint64_t _v=o_get(v4, v4_n, 0ULL, 5);o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=o_get(v4, v4_n, 4ULL, 5);o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=o_get(v4, v4_n, 5ULL, 5);o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=3ULL;uint64_t _v=({uint64_t _a=o_get(v4, v4_n, 32ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_put(v3, v3_n, _i,_v,5);}
if(({uint64_t _a=({uint64_t _a=o_get(v4, v4_n, 32ULL, 5);uint64_t _b=({uint64_t _a=o_get(v4, v4_n, 35ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
{uint64_t _i=3ULL;uint64_t _v=({uint64_t _a=o_get(v3, v3_n, 3ULL, 5);uint64_t _b=16ULL;o_op(1, 5, _a,_b);});o_put(v3, v3_n, _i,_v,5);}
}
return o_leave(0ULL);
}
static uint64_t f60(uint64_t v0){
o_enter();
if(({uint64_t _a=v0;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(2ULL);
}
return o_leave(3ULL);
}
static uint64_t f61(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=8192ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=v3;uint64_t _v=0ULL;o_put(v0, v0_n, _i,_v,5);}
}
}
{
uint64_t v3=0;
uint64_t k1=2048ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=v3;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
}
}
{
uint64_t v3=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=v3;uint64_t _v=0ULL;o_put(v2, v2_n, _i,_v,5);}
}
}
return o_leave(0);
}
static uint64_t f62(const void *v0, uint64_t v0_n){
o_enter();
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=16384ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=8ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=16ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=2ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=4ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=8ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(12, 1, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=32ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t p0=o_get(v0, v0_n, 3ULL, 5);f24(p0);});uint64_t _b=18446744073709551615ULL;o_op(6, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=o_get(v0, v0_n, 1ULL, 5);o_op(3, 5, _a,_b);});uint64_t _b=o_get(v0, v0_n, 3ULL, 5);o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=o_get(v0, v0_n, 1ULL, 5);o_op(3, 5, _a,_b);});uint64_t _b=o_get(v0, v0_n, 3ULL, 5);o_op(4, 5, _a,_b);});uint64_t _b=8ULL;o_op(10, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=128ULL;o_op(3, 5, _a,_b);});uint64_t _b=o_get(v0, v0_n, 1ULL, 5);o_op(3, 5, _a,_b);});uint64_t _b=16384ULL;o_op(10, 5, _a,_b);})){
return o_leave(0ULL);
}
uint64_t __attribute__((unused)) v1=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(4, 5, _a,_b);});
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=2ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=3ULL;o_op(10, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=1ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=4ULL;o_op(10, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
{
uint64_t v2=0;
uint64_t k1=11ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v2;uint64_t _b=5ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(0ULL);
}
}
}
return o_leave(1ULL);
}
static uint64_t f63(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=({uint64_t p2=2ULL;uint64_t p3=1ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});
{
uint64_t v4=0;
uint64_t k1=29ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
uint64_t __attribute__((unused)) v5=1ULL;
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=3ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=11ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=14ULL;o_op(9, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(13, 1, _a,_b);})){
v5=2ULL;
}
{uint64_t _i=v4;uint64_t _v=({uint64_t p2=v5;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
if(({uint64_t _a=o_get(v1, v1_n, 36ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=29ULL;uint64_t _v=({uint64_t p2=2ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
{uint64_t _i=30ULL;uint64_t _v=({uint64_t p2=2ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
return o_leave(0);
}
static uint64_t f64(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=24ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
uint64_t __attribute__((unused)) v4=({uint64_t _a=v3;uint64_t _b=31ULL;o_op(1, 5, _a,_b);});
uint64_t __attribute__((unused)) v5=1ULL;
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=38ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=40ULL;o_op(9, 5, _a,_b);});o_op(12, 1, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=44ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=47ULL;o_op(9, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(13, 1, _a,_b);})){
v5=2ULL;
}
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=49ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=50ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
v5=2ULL;
}
{uint64_t _i=v4;uint64_t _v=({uint64_t p2=v5;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
if(({uint64_t _a=o_get(v1, v1_n, 36ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
{
uint64_t v3=0;
uint64_t k2=o_get(v1, v1_n, 39ULL, 5);
uint64_t c2=0;
for(;o_op(8,5, v3, k2);v3=o_op(1,5, v3,1)){
if(c2++>=16777216)o_trap(4);
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=56ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=2ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v3=0;
uint64_t k2=o_get(v1, v1_n, 34ULL, 5);
uint64_t c2=0;
for(;o_op(8,5, v3, k2);v3=o_op(1,5, v3,1)){
if(c2++>=16777216)o_trap(4);
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=64ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{uint64_t _i=68ULL;uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
{uint64_t _i=69ULL;uint64_t _v=({uint64_t p2=2ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
{
uint64_t v3=0;
uint64_t k1=5ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=70ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=76ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=84ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v3=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=92ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=4ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v3=0;
uint64_t k1=2ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=96ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=({uint64_t _a=o_get(v1, v1_n, 33ULL, 5);uint64_t _b=4ULL;o_op(4, 5, _a,_b);});uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=98ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
{
uint64_t v3=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=100ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v0, v0_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
return o_leave(0);
}
static uint64_t f65(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
({uint64_t p1=v2;uint64_t p2=v3;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=v2;uint64_t _b=7ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=23ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
}
({uint64_t p2=v5;f10(v0, v0_n, v1, v1_n, p2);});
return o_leave(0);
}
static uint64_t f66(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
({uint64_t p2=7ULL;uint64_t p3=0ULL;uint64_t p4=856ULL;uint64_t p5=4ULL;f65(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{
uint64_t v3=0;
uint64_t k1=5ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
uint64_t __attribute__((unused)) v4=5ULL;
uint64_t __attribute__((unused)) v5=({uint64_t _a=896ULL;uint64_t _b=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
if(({uint64_t _a=v3;uint64_t _b=3ULL;o_op(11, 5, _a,_b);})){
v4=4ULL;
v5=({uint64_t _a=920ULL;uint64_t _b=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=3ULL;o_op(2, 5, _a,_b);});uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
}
({uint64_t p2=v4;uint64_t p3=o_get(v2, v2_n, v3, 5);uint64_t p4=v5;uint64_t p5=2ULL;f65(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
if(({uint64_t _a=o_get(v1, v1_n, 36ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
({uint64_t p2=5ULL;uint64_t p3=o_get(v2, v2_n, 29ULL, 5);uint64_t p4=928ULL;uint64_t p5=2ULL;f65(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=5ULL;uint64_t p3=o_get(v2, v2_n, 30ULL, 5);uint64_t p4=936ULL;uint64_t p5=2ULL;f65(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
{
uint64_t v3=0;
uint64_t k1=3ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
uint64_t __attribute__((unused)) v4=33ULL;
if(({uint64_t _a=v3;uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
v4=37ULL;
}
if(({uint64_t _a=v3;uint64_t _b=2ULL;o_op(6, 5, _a,_b);})){
v4=38ULL;
}
({uint64_t p2=3ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=5ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=v4;uint64_t p5=2ULL;f65(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
({uint64_t p2=2ULL;uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 8ULL, 5);uint64_t p3=o_get(v2, v2_n, 5ULL, 5);uint64_t p4=0ULL;uint64_t p5=5ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 9ULL, 5);uint64_t p3=o_get(v2, v2_n, 5ULL, 5);uint64_t p4=0ULL;uint64_t p5=31ULL;f28(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 11ULL, 5);uint64_t p3=o_get(v2, v2_n, 7ULL, 5);uint64_t p4=o_get(v2, v2_n, 0ULL, 5);uint64_t p5=32ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{
uint64_t v3=0;
uint64_t k1=3ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=56ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=12ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, 11ULL, 5);uint64_t p5=({uint64_t p0=v3;f60(p0);});f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
{uint64_t _i=8ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=o_get(v2, v2_n, 15ULL, 5);uint64_t p4=o_get(v2, v2_n, 14ULL, 5);uint64_t p5=18446744073709551615ULL;f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=9ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=o_get(v2, v2_n, 16ULL, 5);uint64_t p4=o_get(v2, v2_n, 14ULL, 5);uint64_t p5=18446744073709551615ULL;f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 17ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 16ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 34ULL, 5);uint64_t _b=1ULL;o_op(2, 5, _a,_b);});f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 18ULL, 5);uint64_t p3=o_get(v2, v2_n, 17ULL, 5);uint64_t p4=0ULL;uint64_t p5=o_get(v1, v1_n, 43ULL, 5);f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 28ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 16ULL, 5);uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 28ULL, 5);uint64_t p3=o_get(v2, v2_n, 28ULL, 5);uint64_t p4=0ULL;uint64_t p5=31ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 28ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);f15(v0, v0_n, v1, v1_n, p2, p3);});
{uint64_t _i=46ULL;uint64_t _v=({uint64_t p2=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t p3=1ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 19ULL, 5);uint64_t p3=o_get(v2, v2_n, 9ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=o_get(v1, v1_n, 33ULL, 5);f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 20ULL, 5);uint64_t p3=o_get(v2, v2_n, 9ULL, 5);uint64_t p4=0ULL;uint64_t p5=o_get(v1, v1_n, 42ULL, 5);f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 20ULL, 5);uint64_t p3=o_get(v2, v2_n, 20ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=4ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 21ULL, 5);uint64_t p3=o_get(v2, v2_n, 9ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=16ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 22ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 3ULL, 5);uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 23ULL, 5);uint64_t p3=o_get(v2, v2_n, 3ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=4ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 24ULL, 5);uint64_t p3=o_get(v2, v2_n, 5ULL, 5);uint64_t p4=0ULL;uint64_t p5=o_get(v1, v1_n, 44ULL, 5);f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 25ULL, 5);uint64_t p3=o_get(v2, v2_n, 5ULL, 5);uint64_t p4=0ULL;uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 40ULL, 5);uint64_t _b=1ULL;o_op(2, 5, _a,_b);});f28(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 25ULL, 5);uint64_t p3=o_get(v2, v2_n, 25ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=4ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 26ULL, 5);uint64_t p3=o_get(v2, v2_n, 5ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=4ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 27ULL, 5);uint64_t p3=o_get(v2, v2_n, 6ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=o_get(v1, v1_n, 35ULL, 5);f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f67(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{uint64_t _i=47ULL;uint64_t _v=o_get(v1, v1_n, 0ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 27ULL, 5);uint64_t p3=o_get(v2, v2_n, 3ULL, 5);f15(v0, v0_n, v1, v1_n, p2, p3);});
{uint64_t _i=48ULL;uint64_t _v=({uint64_t p2=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t p3=1ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 31ULL, 5);uint64_t p3=o_get(v2, v2_n, 27ULL, 5);uint64_t p4=o_get(v2, v2_n, 8ULL, 5);uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 33ULL, 5);uint64_t p3=o_get(v2, v2_n, 31ULL, 5);uint64_t p4=o_get(v2, v2_n, 22ULL, 5);uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 32ULL, 5);uint64_t p3=o_get(v2, v2_n, 33ULL, 5);uint64_t p4=0ULL;uint64_t p5=31ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 34ULL, 5);uint64_t p3=o_get(v2, v2_n, 22ULL, 5);uint64_t p4=o_get(v2, v2_n, 31ULL, 5);uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p1=8ULL;uint64_t p2=o_get(v2, v2_n, 35ULL, 5);uint64_t p3=o_get(v2, v2_n, 32ULL, 5);uint64_t p4=o_get(v2, v2_n, 34ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p2=o_get(v2, v2_n, 36ULL, 5);uint64_t p3=o_get(v2, v2_n, 35ULL, 5);uint64_t p4=o_get(v2, v2_n, 31ULL, 5);uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 38ULL, 5);uint64_t p3=o_get(v2, v2_n, 36ULL, 5);uint64_t p4=o_get(v2, v2_n, 12ULL, 5);uint64_t p5=o_get(v1, v1_n, 32ULL, 5);f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 38ULL, 5);uint64_t p3=o_get(v2, v2_n, 19ULL, 5);uint64_t p4=o_get(v2, v2_n, 38ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 35ULL, 5);uint64_t p3=o_get(v2, v2_n, 36ULL, 5);uint64_t p4=0ULL;uint64_t p5=7ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 39ULL, 5);uint64_t p3=o_get(v2, v2_n, 35ULL, 5);uint64_t p4=o_get(v2, v2_n, 13ULL, 5);uint64_t p5=({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 32ULL, 5);uint64_t _b=128ULL;o_op(4, 5, _a,_b);});uint64_t _b=4ULL;o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 39ULL, 5);uint64_t p3=o_get(v2, v2_n, 20ULL, 5);uint64_t p4=o_get(v2, v2_n, 39ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 40ULL, 5);uint64_t p3=o_get(v2, v2_n, 31ULL, 5);uint64_t p4=o_get(v2, v2_n, 2ULL, 5);uint64_t p5=4ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 37ULL, 5);uint64_t p3=o_get(v2, v2_n, 32ULL, 5);uint64_t p4=o_get(v2, v2_n, 9ULL, 5);uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 41ULL, 5);uint64_t p3=o_get(v2, v2_n, 15ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 42ULL, 5);uint64_t p3=o_get(v2, v2_n, 16ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 43ULL, 5);uint64_t p3=o_get(v2, v2_n, 18ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f68(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
({uint64_t p1=v2;uint64_t p2=v3;uint64_t p3=v4;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=21ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
uint64_t __attribute__((unused)) v6=({uint64_t p0=4ULL;uint64_t p1=v5;uint64_t p2=1ULL;uint64_t p3=({uint64_t _a=o_get(v1, v1_n, 1ULL, 5);uint64_t _b=2ULL;o_op(12, 5, _a,_b);});uint64_t p4=1ULL;f8(p0, p1, p2, p3, p4);});
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=v6;uint64_t _b=32ULL;o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 1ULL, 5);uint64_t _b=2ULL;o_op(13, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 2ULL, 5);uint64_t _b=({uint64_t _a=1ULL;uint64_t _b=v5;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1024ULL;o_op(11, 5, _a,_b);})){
{uint64_t _i=3ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
return o_leave(0);
}
{
uint64_t v7=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v7, k1);v7=o_op(1,5, v7,1)){
{uint64_t _i=({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=v7;o_op(1, 5, _a,_b);});uint64_t _v=o_get(v1, v1_n, ({uint64_t _a=v7;uint64_t _b=16ULL;o_op(1, 5, _a,_b);}), 5);o_put(v0, v0_n, _i,_v,5);}
}
}
{uint64_t _i=0ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f69(void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
uint64_t __attribute__((unused)) v2=({uint64_t _a=1ULL;uint64_t _b=v1;o_op(15, 5, _a,_b);});
{uint64_t _i=1ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=v2;o_op(12, 5, _a,_b);});o_op(13, 5, _a,_b);});o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=({uint64_t _a=v2;uint64_t _b=0;o_op(18, 5, _a,_b);});o_op(12, 5, _a,_b);});o_put(v0, v0_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f70(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{uint64_t _i=49ULL;uint64_t _v=o_get(v1, v1_n, 0ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 51ULL, 5);uint64_t p3=o_get(v2, v2_n, 41ULL, 5);uint64_t p4=o_get(v2, v2_n, 24ULL, 5);uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
if(({uint64_t _a=o_get(v1, v1_n, 36ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
({uint64_t p2=o_get(v2, v2_n, 44ULL, 5);uint64_t p3=o_get(v2, v2_n, 51ULL, 5);uint64_t p4=o_get(v2, v2_n, 1ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 32ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 44ULL, 5);uint64_t p3=o_get(v2, v2_n, 25ULL, 5);uint64_t p4=o_get(v2, v2_n, 44ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 45ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 44ULL, 5);uint64_t p5=0ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}else{
{
uint64_t v3=0;
uint64_t k2=o_get(v1, v1_n, 39ULL, 5);
uint64_t c2=0;
for(;o_op(8,5, v3, k2);v3=o_op(1,5, v3,1)){
if(c2++>=16777216)o_trap(4);
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=o_get(v1, v1_n, 41ULL, 5);o_op(3, 5, _a,_b);});uint64_t _b=o_get(v1, v1_n, 40ULL, 5);o_op(4, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 52ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 51ULL, 5);uint64_t p5=o_get(v1, v1_n, 8ULL, 5);f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 69ULL, 5);uint64_t p3=o_get(v2, v2_n, 52ULL, 5);uint64_t p4=o_get(v2, v2_n, 29ULL, 5);uint64_t p5=4ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=11ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=56ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, 69ULL, 5);uint64_t p5=({uint64_t p0=v3;f60(p0);});f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
{
uint64_t v3=0;
uint64_t k2=o_get(v1, v1_n, 39ULL, 5);
uint64_t c2=0;
for(;o_op(8,5, v3, k2);v3=o_op(1,5, v3,1)){
if(c2++>=16777216)o_trap(4);
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 32ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=56ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=56ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, 1ULL, 5);uint64_t p5=o_get(v1, v1_n, 8ULL, 5);f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=56ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p3=o_get(v2, v2_n, 25ULL, 5);uint64_t p4=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=56ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=v3;uint64_t _b=o_get(v1, v1_n, 41ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=o_get(v1, v1_n, 40ULL, 5);o_op(5, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=56ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=56ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p5=o_get(v1, v1_n, 8ULL, 5);f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
({uint64_t p2=2ULL;uint64_t p3=o_get(v2, v2_n, 68ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=0ULL;f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
({uint64_t p2=o_get(v2, v2_n, 46ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 38ULL, 5);uint64_t p5=0ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 47ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 39ULL, 5);uint64_t p5=0ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p2=1ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=76ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=0ULL;f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
({uint64_t p2=o_get(v1, v1_n, 45ULL, 5);uint64_t p3=o_get(v2, v2_n, 96ULL, 5);uint64_t p4=o_get(v2, v2_n, 46ULL, 5);uint64_t p5=3ULL;f68(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=11ULL;uint64_t p3=o_get(v2, v2_n, 98ULL, 5);uint64_t p4=o_get(v2, v2_n, 47ULL, 5);uint64_t p5=3ULL;f68(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=2ULL;uint64_t p3=o_get(v2, v2_n, 48ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 37ULL, 5);uint64_t _b=o_get(v1, v1_n, 38ULL, 5);o_op(4, 5, _a,_b);});f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=50ULL;uint64_t _v=o_get(v1, v1_n, 0ULL, 5);o_put(v1, v1_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f71(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, uint64_t v4){
o_enter();
{uint64_t _i=51ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t _b=o_get(v1, v1_n, 38ULL, 5);o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 37ULL, 5);uint64_t _b=o_get(v1, v1_n, 38ULL, 5);o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
{uint64_t _i=51ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
}
if(({uint64_t _a=o_get(v1, v1_n, 51ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=104ULL;uint64_t _v=({uint64_t p2=2ULL;uint64_t p3=0ULL;f7(v1, v1_n, v3, v3_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
{uint64_t _i=105ULL;uint64_t _v=({uint64_t p2=2ULL;uint64_t p3=0ULL;f7(v1, v1_n, v3, v3_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
{uint64_t _i=106ULL;uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v3, v3_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
if(({uint64_t _a=o_get(v1, v1_n, 51ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t _b=o_get(v1, v1_n, 38ULL, 5);o_op(7, 5, _a,_b);})){
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t _b=o_get(v1, v1_n, 40ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 104ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 46ULL, 5);uint64_t p5=o_get(v1, v1_n, 8ULL, 5);f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 40ULL, 5);uint64_t _b=128ULL;o_op(4, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t _b=o_get(v1, v1_n, 8ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 105ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 47ULL, 5);uint64_t p5=o_get(v1, v1_n, 8ULL, 5);f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}else{
{
uint64_t v5=0;
uint64_t k3=3ULL;
for(;o_op(8,5, v5, k3);v5=o_op(1,5, v5,1)){
{uint64_t _i=({uint64_t _a=v5;uint64_t _b=107ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v1, v1_n, v3, v3_n, p2, p3);});o_put(v2, v2_n, _i,_v,5);}
}
}
({uint64_t p2=o_get(v2, v2_n, 107ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 48ULL, 5);uint64_t p5=4294967294ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 107ULL, 5);uint64_t p3=o_get(v2, v2_n, 107ULL, 5);uint64_t p4=0ULL;uint64_t p5=31ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 107ULL, 5);uint64_t p3=o_get(v2, v2_n, 107ULL, 5);uint64_t p4=o_get(v2, v2_n, 10ULL, 5);uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 38ULL, 5);uint64_t _b=o_get(v1, v1_n, 40ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 108ULL, 5);uint64_t p3=o_get(v2, v2_n, 107ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=o_get(v1, v1_n, 8ULL, 5);f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 40ULL, 5);uint64_t _b=128ULL;o_op(4, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 38ULL, 5);uint64_t _b=o_get(v1, v1_n, 8ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 109ULL, 5);uint64_t p3=o_get(v2, v2_n, 107ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=o_get(v1, v1_n, 8ULL, 5);f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 104ULL, 5);uint64_t p3=o_get(v2, v2_n, 108ULL, 5);uint64_t p4=o_get(v2, v2_n, 46ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 105ULL, 5);uint64_t p3=o_get(v2, v2_n, 109ULL, 5);uint64_t p4=o_get(v2, v2_n, 47ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
return o_leave(0);
}
static uint64_t f72(void *v0, uint64_t v0_n){
o_enter();
{uint64_t _i=53ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 39ULL, 5);uint64_t _b=o_get(v0, v0_n, 52ULL, 5);o_op(2, 5, _a,_b);});o_put(v0, v0_n, _i,_v,5);}
if(({uint64_t _a=o_get(v0, v0_n, 53ULL, 5);uint64_t _b=4ULL;o_op(10, 5, _a,_b);})){
{uint64_t _i=53ULL;uint64_t _v=4ULL;o_put(v0, v0_n, _i,_v,5);}
}
return o_leave(0);
}
static uint64_t f73(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=o_get(v0, v0_n, 53ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=110ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=1ULL;uint64_t p3=0ULL;f7(v0, v0_n, v2, v2_n, p2, p3);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=114ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p2=2ULL;uint64_t p3=0ULL;f7(v0, v0_n, v2, v2_n, p2, p3);});o_put(v1, v1_n, _i,_v,5);}
}
}
return o_leave(0);
}
static uint64_t f74(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 53ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=57ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 52ULL, 5);uint64_t _b=v3;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=57ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 57ULL, 5);uint64_t _b=o_get(v1, v1_n, 41ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=o_get(v1, v1_n, 36ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=59ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 52ULL, 5);uint64_t _b=v3;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=59ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 59ULL, 5);uint64_t _b=56ULL;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=59ULL;uint64_t _v=o_get(v2, v2_n, o_get(v1, v1_n, 59ULL, 5), 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=114ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p3=o_get(v2, v2_n, 68ULL, 5);uint64_t p4=o_get(v1, v1_n, 59ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}else{
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 57ULL, 5);uint64_t _b=o_get(v1, v1_n, 40ULL, 5);o_op(4, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=o_get(v1, v1_n, 32ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=59ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 57ULL, 5);uint64_t _b=o_get(v1, v1_n, 40ULL, 5);o_op(5, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=o_get(v1, v1_n, 59ULL, 5);o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=114ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 45ULL, 5);uint64_t p5=o_get(v1, v1_n, 8ULL, 5);f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
}
return o_leave(0);
}
static uint64_t f75(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
({f74(v0, v0_n, v1, v1_n, v2, v2_n);});
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 53ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=8ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=11ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=110ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=114ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p5=({uint64_t p0=v3;f60(p0);});f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
if(({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 51ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 52ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});o_op(12, 1, _a,_b);})){
{uint64_t _i=59ULL;uint64_t _v=({uint64_t _a=1ULL;uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 58ULL, 5);uint64_t _b=1ULL;o_op(12, 5, _a,_b);});o_op(2, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v1, v1_n, 45ULL, 5);uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=o_get(v1, v1_n, 59ULL, 5);uint64_t _b=96ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, 104ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 59ULL, 5);uint64_t _b=3ULL;o_op(1, 5, _a,_b);});f68(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=11ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=o_get(v1, v1_n, 59ULL, 5);uint64_t _b=98ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, 105ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 59ULL, 5);uint64_t _b=3ULL;o_op(1, 5, _a,_b);});f68(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 53ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 52ULL, 5);uint64_t _b=v3;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=o_get(v1, v1_n, 41ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 106ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 26ULL, 5);uint64_t p5=o_get(v1, v1_n, 8ULL, 5);f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p1=50ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=o_get(v2, v2_n, 106ULL, 5);uint64_t p4=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=110ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=5ULL;f10(v0, v0_n, v1, v1_n, p2);});
}
}
return o_leave(0);
}
static uint64_t f76(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, uint64_t v4){
o_enter();
{uint64_t _i=58ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=52ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
{
uint64_t c1=0;
while(({uint64_t _a=o_get(v1, v1_n, 52ULL, 5);uint64_t _b=o_get(v1, v1_n, 39ULL, 5);o_op(8, 5, _a,_b);})){
if(c1++>=2)o_trap(4);
({f72(v1, v1_n);});
({f73(v1, v1_n, v2, v2_n, v3, v3_n);});
({f75(v0, v0_n, v1, v1_n, v2, v2_n);});
{uint64_t _i=52ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 52ULL, 5);uint64_t _b=4ULL;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
}
}
({uint64_t p1=48ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=6ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p1=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=1ULL;o_op(12, 5, _a,_b);});uint64_t _b=3ULL;o_op(1, 5, _a,_b);});f69(v1, v1_n, p1);});
return o_leave(0);
}
static uint64_t f77(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
if(({uint64_t _a=v5;uint64_t _b=3ULL;o_op(8, 5, _a,_b);})){
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=1ULL;uint64_t _b=({uint64_t _a=20ULL;uint64_t _b=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(2, 5, _a,_b);});o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=o_get(v2, v2_n, 70ULL, 5);uint64_t p4=v3;uint64_t p5=18446744073709551615ULL;f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}else{
({uint64_t p2=o_get(v2, v2_n, 70ULL, 5);uint64_t p3=v3;uint64_t p4=v4;uint64_t p5=4ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
({uint64_t p2=o_get(v2, v2_n, 71ULL, 5);uint64_t p3=o_get(v2, v2_n, 70ULL, 5);uint64_t p4=0ULL;uint64_t p5=({uint64_t _a=127ULL;uint64_t _b=20ULL;o_op(15, 5, _a,_b);});f28(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 72ULL, 5);uint64_t p3=v3;uint64_t p4=v4;uint64_t p5=({uint64_t _a=128ULL;uint64_t _b=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(15, 5, _a,_b);});f28(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=v5;uint64_t _b=3ULL;o_op(8, 5, _a,_b);})){
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=1ULL;uint64_t _b=({uint64_t _a=24ULL;uint64_t _b=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(2, 5, _a,_b);});o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=8ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v5;uint64_t _b=100ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, 72ULL, 5);uint64_t p5=o_get(v2, v2_n, 71ULL, 5);f12(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
return o_leave(0);
}
static uint64_t f78(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3){
o_enter();
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 60ULL, 5);uint64_t _b=v3;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=o_get(v1, v1_n, 40ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=62ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 61ULL, 5);uint64_t _b=512ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=o_get(v1, v1_n, 62ULL, 5);o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p1=58ULL;uint64_t p2=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=92ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p3=o_get(v2, v2_n, 21ULL, 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=o_get(v1, v1_n, 8ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=({uint64_t p0=v3;f60(p0);});f10(v0, v0_n, v1, v1_n, p2);});
return o_leave(0);
}
static uint64_t f79(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3){
o_enter();
{uint64_t _i=62ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 60ULL, 5);uint64_t _b=v3;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=62ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 62ULL, 5);uint64_t _b=84ULL;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=62ULL;uint64_t _v=o_get(v2, v2_n, o_get(v1, v1_n, 62ULL, 5), 5);o_put(v1, v1_n, _i,_v,5);}
{
uint64_t v4=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
{uint64_t _i=9ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v1, v1_n, 62ULL, 5);uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v4;uint64_t _b=100ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=92ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p5=o_get(v1, v1_n, 62ULL, 5);f14(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
return o_leave(0);
}
static uint64_t f80(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3, uint64_t v4){
o_enter();
{uint64_t _i=60ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=61ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
uint64_t __attribute__((unused)) v5=({uint64_t _a=o_get(v1, v1_n, 34ULL, 5);uint64_t _b=v3;o_op(2, 5, _a,_b);});
if(({uint64_t _a=v5;uint64_t _b=4ULL;o_op(10, 5, _a,_b);})){
v5=4ULL;
}
{
uint64_t v6=0;
uint64_t c1=0;
for(;o_op(8,5, v6, v5);v6=o_op(1,5, v6,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p3=v6;f78(v0, v0_n, v1, v1_n, v2, v2_n, p3);});
}
}
{
uint64_t v6=0;
uint64_t c1=0;
for(;o_op(8,5, v6, v5);v6=o_op(1,5, v6,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p3=v6;f79(v0, v0_n, v1, v1_n, v2, v2_n, p3);});
}
}
return o_leave(0);
}
static uint64_t f81(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3){
o_enter();
{uint64_t _i=59ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 58ULL, 5);uint64_t _b=1ULL;o_op(12, 5, _a,_b);});uint64_t _b=96ULL;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=59ULL;uint64_t _v=o_get(v2, v2_n, o_get(v1, v1_n, 59ULL, 5), 5);o_put(v1, v1_n, _i,_v,5);}
{
uint64_t v4=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
({uint64_t p3=o_get(v1, v1_n, 59ULL, 5);uint64_t p4=v3;uint64_t p5=v4;f77(v0, v0_n, v1, v1_n, v2, v2_n, p3, p4, p5);});
}
}
{
uint64_t v4=0;
uint64_t k1=({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 34ULL, 5);uint64_t _b=3ULL;o_op(1, 5, _a,_b);});uint64_t _b=4ULL;o_op(4, 5, _a,_b);});
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p3=({uint64_t _a=v4;uint64_t _b=4ULL;o_op(3, 5, _a,_b);});uint64_t p4=v3;f80(v0, v0_n, v1, v1_n, v2, v2_n, p3, p4);});
}
}
return o_leave(0);
}
static uint64_t f82(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, uint64_t v3){
o_enter();
{
uint64_t v4=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p2=1ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v4;uint64_t _b=84ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=0ULL;f11(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
{
uint64_t v4=0;
uint64_t k1=({uint64_t _a=o_get(v1, v1_n, 33ULL, 5);uint64_t _b=4ULL;o_op(4, 5, _a,_b);});
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p3=v4;f81(v0, v0_n, v1, v1_n, v2, v2_n, p3);});
}
}
{uint64_t _i=59ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=1ULL;o_op(12, 5, _a,_b);});uint64_t _b=98ULL;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=59ULL;uint64_t _v=o_get(v2, v2_n, o_get(v1, v1_n, 59ULL, 5), 5);o_put(v1, v1_n, _i,_v,5);}
{
uint64_t v4=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=60ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=v4;uint64_t _b=76ULL;o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=61ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=v4;uint64_t _b=84ULL;o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v1, v1_n, 60ULL, 5);uint64_t p3=o_get(v1, v1_n, 61ULL, 5);uint64_t p4=o_get(v1, v1_n, 59ULL, 5);uint64_t p5=o_get(v1, v1_n, 60ULL, 5);f14(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
({uint64_t p1=48ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=6ULL;f10(v0, v0_n, v1, v1_n, p2);});
if(({uint64_t _a=o_get(v1, v1_n, 36ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
({uint64_t p2=o_get(v2, v2_n, 68ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 68ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 40ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}else{
({uint64_t p2=o_get(v2, v2_n, 45ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 45ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 40ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
return o_leave(0);
}
static uint64_t f83(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
({uint64_t p2=o_get(v2, v2_n, 46ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 46ULL, 5);uint64_t p5=({uint64_t _a=o_get(v1, v1_n, 38ULL, 5);uint64_t _b=o_get(v1, v1_n, 40ULL, 5);o_op(3, 5, _a,_b);});f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 40ULL, 5);uint64_t _b=128ULL;o_op(4, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 38ULL, 5);uint64_t _b=o_get(v1, v1_n, 8ULL, 5);o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 8ULL, 5);uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=o_get(v2, v2_n, 47ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 47ULL, 5);uint64_t p5=o_get(v1, v1_n, 8ULL, 5);f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 48ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 48ULL, 5);uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 48ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);f15(v0, v0_n, v1, v1_n, p2, p3);});
uint64_t __attribute__((unused)) v3=({uint64_t p2=o_get(v1, v1_n, 50ULL, 5);uint64_t p3=1ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});
({uint64_t p1=3ULL;f69(v1, v1_n, p1);});
({uint64_t p1=4ULL;f69(v1, v1_n, p1);});
return o_leave(0);
}
static uint64_t f84(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{
uint64_t v4=0;
uint64_t k2=5ULL;
for(;o_op(8,5, v4, k2);v4=o_op(1,5, v4,1)){
({uint64_t p1=42ULL;uint64_t p2=o_get(v2, v2_n, 51ULL, 5);uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=76ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=16ULL;uint64_t _b=v4;o_op(16, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=2ULL;f10(v0, v0_n, v1, v1_n, p2);});
{uint64_t _i=59ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=76ULL;o_op(1, 5, _a,_b);}), 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p1=30ULL;uint64_t p2=o_get(v1, v1_n, 59ULL, 5);uint64_t p3=o_get(v1, v1_n, 59ULL, 5);uint64_t p4=o_get(v2, v2_n, 51ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
}
}
}
}
({uint64_t p1=8ULL;uint64_t p2=o_get(v2, v2_n, 52ULL, 5);uint64_t p3=o_get(v2, v2_n, 41ULL, 5);uint64_t p4=o_get(v2, v2_n, 23ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p2=o_get(v2, v2_n, 49ULL, 5);uint64_t p3=o_get(v2, v2_n, 52ULL, 5);uint64_t p4=o_get(v2, v2_n, 40ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
if(({uint64_t _a=o_get(v1, v1_n, 36ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
{
uint64_t v3=0;
uint64_t k2=o_get(v1, v1_n, 34ULL, 5);
uint64_t c2=0;
for(;o_op(8,5, v3, k2);v3=o_op(1,5, v3,1)){
if(c2++>=16777216)o_trap(4);
({uint64_t p2=o_get(v2, v2_n, 52ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 41ULL, 5);uint64_t p5=v3;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 69ULL, 5);uint64_t p3=o_get(v2, v2_n, 52ULL, 5);uint64_t p4=o_get(v2, v2_n, 30ULL, 5);uint64_t p5=4ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
{uint64_t _i=8ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=11ULL;uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=64ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, 69ULL, 5);uint64_t p5=({uint64_t p0=v3;f60(p0);});f13(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
}
}
{
uint64_t v3=0;
uint64_t k1=o_get(v1, v1_n, 34ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p2=o_get(v2, v2_n, 52ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 42ULL, 5);uint64_t p5=({uint64_t _a=4294967295ULL;uint64_t _b=v3;o_op(2, 5, _a,_b);});f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 53ULL, 5);uint64_t p3=o_get(v2, v2_n, 52ULL, 5);uint64_t p4=0ULL;uint64_t p5=31ULL;f27(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 54ULL, 5);uint64_t p3=o_get(v2, v2_n, 53ULL, 5);uint64_t p4=o_get(v2, v2_n, 37ULL, 5);uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 54ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);f15(v0, v0_n, v1, v1_n, p2, p3);});
{uint64_t _i=54ULL;uint64_t _v=o_get(v1, v1_n, 0ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p1=54ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=2ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=9ULL;f10(v0, v0_n, v1, v1_n, p2);});
{uint64_t _i=55ULL;uint64_t _v=({uint64_t p2=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t p3=1ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=o_get(v1, v1_n, 36ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=8ULL;uint64_t p2=o_get(v2, v2_n, 52ULL, 5);uint64_t p3=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=64ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p4=o_get(v2, v2_n, 23ULL, 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=0ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p2=o_get(v2, v2_n, 50ULL, 5);uint64_t p3=o_get(v2, v2_n, 52ULL, 5);uint64_t p4=o_get(v2, v2_n, 40ULL, 5);uint64_t p5=1ULL;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}else{
({uint64_t p2=o_get(v2, v2_n, 50ULL, 5);uint64_t p3=o_get(v2, v2_n, 23ULL, 5);uint64_t p4=o_get(v2, v2_n, 49ULL, 5);uint64_t p5=v3;f26(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
}
({uint64_t p1=12ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=o_get(v2, v2_n, 50ULL, 5);uint64_t p4=o_get(v2, v2_n, ({uint64_t _a=v3;uint64_t _b=76ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=21ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
({uint64_t p2=5ULL;f10(v0, v0_n, v1, v1_n, p2);});
{uint64_t _i=56ULL;uint64_t _v=o_get(v1, v1_n, 0ULL, 5);o_put(v1, v1_n, _i,_v,5);}
({uint64_t p1=55ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=10ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p1=o_get(v1, v1_n, 55ULL, 5);uint64_t p2=o_get(v1, v1_n, 56ULL, 5);f17(v0, v0_n, p1, p2);});
({uint64_t p1=o_get(v1, v1_n, 54ULL, 5);uint64_t p2=({uint64_t _a=o_get(v1, v1_n, 56ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});f17(v0, v0_n, p1, p2);});
}
}
({uint64_t p2=o_get(v2, v2_n, 41ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 41ULL, 5);uint64_t p5=o_get(v1, v1_n, 34ULL, 5);f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 42ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 42ULL, 5);uint64_t p5=({uint64_t _a=4294967296ULL;uint64_t _b=o_get(v1, v1_n, 34ULL, 5);o_op(2, 5, _a,_b);});f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 43ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);uint64_t p4=o_get(v2, v2_n, 43ULL, 5);uint64_t p5=4294967295ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
({uint64_t p2=o_get(v2, v2_n, 43ULL, 5);uint64_t p3=o_get(v2, v2_n, 10ULL, 5);f15(v0, v0_n, v1, v1_n, p2, p3);});
uint64_t __attribute__((unused)) v3=({uint64_t p2=o_get(v1, v1_n, 49ULL, 5);uint64_t p3=1ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});
({uint64_t p2=o_get(v2, v2_n, 27ULL, 5);uint64_t p3=o_get(v2, v2_n, 4ULL, 5);uint64_t p4=o_get(v2, v2_n, 27ULL, 5);uint64_t p5=1ULL;f25(v0, v0_n, v1, v1_n, p2, p3, p4, p5);});
v3=({uint64_t p2=o_get(v1, v1_n, 47ULL, 5);uint64_t p3=0ULL;f16(v0, v0_n, v1, v1_n, p2, p3);});
({uint64_t p1=o_get(v1, v1_n, 48ULL, 5);uint64_t p2=o_get(v1, v1_n, 0ULL, 5);f17(v0, v0_n, p1, p2);});
({uint64_t p1=o_get(v1, v1_n, 46ULL, 5);uint64_t p2=o_get(v1, v1_n, 0ULL, 5);f17(v0, v0_n, p1, p2);});
({uint64_t p1=45ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
({uint64_t p2=7ULL;f10(v0, v0_n, v1, v1_n, p2);});
({uint64_t p1=46ULL;uint64_t p2=18446744073709551615ULL;uint64_t p3=18446744073709551615ULL;uint64_t p4=18446744073709551615ULL;uint64_t p5=18446744073709551615ULL;f6(v1, v1_n, p1, p2, p3, p4, p5);});
{uint64_t _i=22ULL;uint64_t _v=({uint64_t _a=1032192ULL;uint64_t _b=32ULL;o_op(15, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
if(({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1024ULL;o_op(8, 5, _a,_b);})){
{
uint64_t v4=0;
uint64_t k2=8ULL;
for(;o_op(8,5, v4, k2);v4=o_op(1,5, v4,1)){
{uint64_t _i=({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=v4;o_op(1, 5, _a,_b);});uint64_t _v=o_get(v1, v1_n, ({uint64_t _a=v4;uint64_t _b=16ULL;o_op(1, 5, _a,_b);}), 5);o_put(v0, v0_n, _i,_v,5);}
}
}
{uint64_t _i=0ULL;uint64_t _v=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
}else{
{uint64_t _i=3ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
}
return o_leave(0);
}
static uint64_t f85(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n){
o_enter();
uint64_t v4[64]={0};
uint64_t v4_n=64;
o_fill(v4, v4_n, 0ULL, 5);
uint64_t v5[128]={0};
uint64_t v5_n=128;
o_fill(v5, v5_n, 18446744073709551615ULL, 5);
({f61(v1, v1_n, v2, v2_n, v3, v3_n);});
if(({uint64_t _a=({f62(v0, v0_n);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v6=0;
uint64_t k1=5ULL;
for(;o_op(8,5, v6, k1);v6=o_op(1,5, v6,1)){
{uint64_t _i=({uint64_t _a=v6;uint64_t _b=32ULL;o_op(1, 5, _a,_b);});uint64_t _v=o_get(v0, v0_n, v6, 5);o_put(v4, v4_n, _i,_v,5);}
}
}
{uint64_t _i=37ULL;uint64_t _v=({uint64_t _a=o_get(v4, v4_n, 32ULL, 5);uint64_t _b=({uint64_t _a=o_get(v4, v4_n, 33ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(4, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=38ULL;uint64_t _v=o_get(v4, v4_n, 37ULL, 5);o_put(v4, v4_n, _i,_v,5);}
if(({uint64_t _a=({uint64_t _a=o_get(v4, v4_n, 37ULL, 5);uint64_t _b=2ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=38ULL;uint64_t _v=2ULL;o_put(v4, v4_n, _i,_v,5);}
}
{uint64_t _i=39ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=o_get(v4, v4_n, 34ULL, 5);uint64_t _b=o_get(v4, v4_n, 33ULL, 5);o_op(3, 5, _a,_b);});uint64_t _b=o_get(v4, v4_n, 35ULL, 5);o_op(4, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=40ULL;uint64_t _v=({uint64_t _a=o_get(v4, v4_n, 33ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=41ULL;uint64_t _v=({uint64_t _a=o_get(v4, v4_n, 35ULL, 5);uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=42ULL;uint64_t _v=({uint64_t p0=({uint64_t _a=128ULL;uint64_t _b=o_get(v4, v4_n, 33ULL, 5);o_op(4, 5, _a,_b);});f24(p0);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=43ULL;uint64_t _v=({uint64_t p0=o_get(v4, v4_n, 34ULL, 5);f24(p0);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=44ULL;uint64_t _v=({uint64_t p0=o_get(v4, v4_n, 40ULL, 5);f24(p0);});o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=45ULL;uint64_t _v=56ULL;o_put(v4, v4_n, _i,_v,5);}
if(({uint64_t _a=o_get(v4, v4_n, 33ULL, 5);uint64_t _b=16ULL;o_op(6, 5, _a,_b);})){
{uint64_t _i=45ULL;uint64_t _v=57ULL;o_put(v4, v4_n, _i,_v,5);}
}
({f63(v2, v2_n, v4, v4_n, v5, v5_n);});
({f66(v1, v1_n, v4, v4_n, v5, v5_n);});
({f64(v2, v2_n, v4, v4_n, v5, v5_n);});
({f67(v1, v1_n, v4, v4_n, v5, v5_n);});
({f70(v1, v1_n, v4, v4_n, v5, v5_n);});
{
uint64_t v6=0;
uint64_t k1=o_get(v4, v4_n, 38ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v6, k1);v6=o_op(1,5, v6,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p4=v6;f71(v1, v1_n, v4, v4_n, v5, v5_n, v2, v2_n, p4);});
({uint64_t p4=v6;f76(v1, v1_n, v4, v4_n, v5, v5_n, v2, v2_n, p4);});
({uint64_t p3=v6;f82(v1, v1_n, v4, v4_n, v5, v5_n, p3);});
}
}
({f83(v1, v1_n, v4, v4_n, v5, v5_n);});
({f84(v1, v1_n, v4, v4_n, v5, v5_n);});
if(({uint64_t _a=o_get(v4, v4_n, 3ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
({f61(v1, v1_n, v2, v2_n, v3, v3_n);});
return o_leave(2ULL);
}
{uint64_t _i=0ULL;uint64_t _v=o_get(v4, v4_n, 0ULL, 5);o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=o_get(v4, v4_n, 4ULL, 5);o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=o_get(v4, v4_n, 5ULL, 5);o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=3ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=o_get(v4, v4_n, 34ULL, 5);uint64_t _b=128ULL;o_op(3, 5, _a,_b);});uint64_t _b=o_get(v4, v4_n, 33ULL, 5);o_op(3, 5, _a,_b);});o_put(v3, v3_n, _i,_v,5);}
return o_leave(0ULL);
}
static uint64_t f86(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, void *v4, uint64_t v4_n){
o_enter();
uint64_t v5[16]={0};
uint64_t v5_n=16;
o_fill(v5, v5_n, 0ULL, 5);
uint64_t v6[8]={0};
uint64_t v6_n=8;
o_fill(v6, v6_n, 0ULL, 5);
uint64_t v7[4]={0};
uint64_t v7_n=4;
o_fill(v7, v7_n, 0ULL, 5);
uint64_t v8[4]={0};
uint64_t v8_n=4;
o_fill(v8, v8_n, 0ULL, 5);
{
uint64_t v9=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v9, k1);v9=o_op(1,5, v9,1)){
{uint64_t _i=v9;uint64_t _v=0ULL;o_put(v4, v4_n, _i,_v,5);}
}
}
{
uint64_t v9=0;
uint64_t k1=10ULL;
for(;o_op(8,5, v9, k1);v9=o_op(1,5, v9,1)){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v9;uint64_t _b=6ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
}
}
uint64_t __attribute__((unused)) v9=o_get(v0, v0_n, 0ULL, 5);
if(({uint64_t _a=({uint64_t _a=v9;uint64_t _b=1ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=v9;uint64_t _b=3ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v10=0;
uint64_t k1=5ULL;
for(;o_op(8,5, v10, k1);v10=o_op(1,5, v10,1)){
{uint64_t _i=v10;uint64_t _v=o_get(v0, v0_n, ({uint64_t _a=v10;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);o_put(v5, v5_n, _i,_v,5);}
{uint64_t _i=v10;uint64_t _v=o_get(v0, v0_n, ({uint64_t _a=v10;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);o_put(v6, v6_n, _i,_v,5);}
}
}
uint64_t __attribute__((unused)) v10=1ULL;
if(({uint64_t _a=v9;uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
v10=({f41(v6, v6_n, v1, v1_n, v2, v2_n, v7, v7_n);});
}
if(({uint64_t _a=v9;uint64_t _b=2ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 5ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
v10=({f59(v6, v6_n, v1, v1_n, v2, v2_n, v7, v7_n);});
}
if(({uint64_t _a=v9;uint64_t _b=3ULL;o_op(6, 5, _a,_b);})){
v10=({f85(v5, v5_n, v1, v1_n, v2, v2_n, v7, v7_n);});
}
if(({uint64_t _a=v10;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v10);
}
v10=({uint64_t p1=o_get(v7, v7_n, 0ULL, 5);f23(v1, v1_n, p1, v2, v2_n);});
if(({uint64_t _a=v10;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(({uint64_t _a=v10;uint64_t _b=8ULL;o_op(1, 4, _a,_b);}));
}
v10=({uint64_t p1=o_get(v7, v7_n, 0ULL, 5);uint64_t p5=128ULL;f5(v1, v1_n, p1, v2, v2_n, v3, v3_n, v8, v8_n, p5);});
if(({uint64_t _a=v10;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(({uint64_t _a=v10;uint64_t _b=16ULL;o_op(1, 4, _a,_b);}));
}
{uint64_t _i=0ULL;uint64_t _v=o_get(v7, v7_n, 0ULL, 5);o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=o_get(v8, v8_n, 1ULL, 5);o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=o_get(v8, v8_n, 2ULL, 5);o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=3ULL;uint64_t _v=o_get(v7, v7_n, 3ULL, 5);o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=4ULL;uint64_t _v=o_get(v7, v7_n, 1ULL, 5);o_put(v4, v4_n, _i,_v,5);}
{uint64_t _i=5ULL;uint64_t _v=o_get(v7, v7_n, 2ULL, 5);o_put(v4, v4_n, _i,_v,5);}
return o_leave(0ULL);
}
static uint64_t f87(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, void *v4, uint64_t v4_n){
o_enter();
return o_leave(o_cast(5, 4, ({f86(v0, v0_n, v1, v1_n, v2, v2_n, v3, v3_n, v4, v4_n);})));
}
uint32_t omega_native_compose(uint64_t *result, const void *p0, uint64_t p0_n, void *p1, uint64_t p1_n, void *p2, uint64_t p2_n, void *p3, uint64_t p3_n, void *p4, uint64_t p4_n){
if(!o_buf(result,1,1,5))return 32;
if(!o_buf(p0, p0_n,16, 5))return 32;
if(o_overlap(result,8,p0, p0_n*8))return 32;
if(!o_buf(p1, p1_n,8192, 5))return 32;
if(o_overlap(result,8,p1, p1_n*8))return 32;
if(o_overlap(p1, p1_n*8, p0, p0_n*8))return 32;
if(!o_buf(p2, p2_n,2048, 5))return 32;
if(o_overlap(result,8,p2, p2_n*8))return 32;
if(o_overlap(p2, p2_n*8, p0, p0_n*8))return 32;
if(o_overlap(p2, p2_n*8, p1, p1_n*8))return 32;
if(!o_buf(p3, p3_n,288, 5))return 32;
if(o_overlap(result,8,p3, p3_n*8))return 32;
if(o_overlap(p3, p3_n*8, p0, p0_n*8))return 32;
if(o_overlap(p3, p3_n*8, p1, p1_n*8))return 32;
if(o_overlap(p3, p3_n*8, p2, p2_n*8))return 32;
if(!o_buf(p4, p4_n,8, 5))return 32;
if(o_overlap(result,8,p4, p4_n*8))return 32;
if(o_overlap(p4, p4_n*8, p0, p0_n*8))return 32;
if(o_overlap(p4, p4_n*8, p1, p1_n*8))return 32;
if(o_overlap(p4, p4_n*8, p2, p2_n*8))return 32;
if(o_overlap(p4, p4_n*8, p3, p3_n*8))return 32;
if(o_busy)return 33;
o_busy=1;o_depth=0;int code=setjmp(o_jmp);
if(!code)*result=f87(p0, p0_n, p1, p1_n, p2, p2_n, p3, p3_n, p4, p4_n);
o_busy=0;o_depth=0;return (uint32_t)code;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
