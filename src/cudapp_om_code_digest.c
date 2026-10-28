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
static uint64_t f0(const void *v0, uint64_t v0_n);
static uint64_t f1(const void *v0, uint64_t v0_n);
static uint64_t f2(uint64_t v0, uint64_t v1);
static uint64_t f3(uint64_t v0);
static uint64_t f4(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4);
static uint64_t f5(const void *v0, uint64_t v0_n);
static uint64_t f6(const void *v0, uint64_t v0_n);
static uint64_t f7(const void *v0, uint64_t v0_n);
static uint64_t f8(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f9(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f10(const void *v0, uint64_t v0_n);
static uint64_t f11(const void *v0, uint64_t v0_n);
static uint64_t f12(const void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f13(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f14(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f15(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n);
static uint64_t f16(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n);
static uint64_t f17(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, void *v2, uint64_t v2_n);
static uint64_t f18(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4);
static uint64_t f19(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, void *v2, uint64_t v2_n);
static uint64_t f20(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, void *v4, uint64_t v4_n);
static uint64_t f21(const void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f22(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f23(const void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n);
static uint64_t f24(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n);
static uint64_t f25(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n);
static uint64_t f26(const void *v0, uint64_t v0_n);
static uint64_t f27(const void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f28(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n);
static uint64_t f29(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n);
static uint64_t f30(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2);
static uint64_t f31(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f32(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f33(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n);
static uint64_t f34(uint64_t v0, uint64_t v1);
static uint64_t f35(void *v0, uint64_t v0_n);
static uint64_t f36(void *v0, uint64_t v0_n);
static uint64_t f37(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2);
static uint64_t f38(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2);
static uint64_t f39(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2);
static uint64_t f40(void *v0, uint64_t v0_n);
static uint64_t f41(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f42(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f43(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, uint64_t v4);
static uint64_t f44(const void *v0, uint64_t v0_n, uint64_t v1, void *v2, uint64_t v2_n);
static uint64_t f45(const void *v0, uint64_t v0_n, uint64_t v1, void *v2, uint64_t v2_n);
static uint64_t f46(const void *v0, uint64_t v0_n, uint64_t v1, void *v2, uint64_t v2_n);
static uint64_t f0(const void *v0, uint64_t v0_n){
o_enter();
if(({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
uint64_t __attribute__((unused)) v1=o_get(v0, v0_n, 1ULL, 5);
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=1ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=6ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 17ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 18ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 17ULL, 5);uint64_t _b=({uint64_t _a=18446744073709551615ULL;uint64_t _b=o_get(v0, v0_n, 18ULL, 5);o_op(4, 5, _a,_b);});o_op(10, 5, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 19ULL, 5);uint64_t _b=2ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 15ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 14ULL, 5);uint64_t _b=o_get(v0, v0_n, 2ULL, 5);o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 16ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v2=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v2;uint64_t _b=20ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
}
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 9ULL, 5);uint64_t _b=1ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 10ULL, 5);uint64_t _b=1ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=4ULL;o_op(10, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 7ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 8ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(7ULL);
}
}else{
if(({uint64_t _a=o_get(v0, v0_n, 7ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);})){
return o_leave(7ULL);
}
}
if(({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 5ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 6ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
}else{
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 5ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 6ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
}
if(({uint64_t _a=v1;uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=1ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=8ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=1ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=2ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 11ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 13ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 12ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 7ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);})){
return o_leave(7ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 8ULL, 5);uint64_t _b=({uint64_t _a=1ULL;uint64_t _b=o_get(v0, v0_n, 2ULL, 5);o_op(15, 5, _a,_b);});o_op(11, 5, _a,_b);})){
return o_leave(7ULL);
}
}
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 7ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(7ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 11ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 13ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 12ULL, 5);uint64_t _b=1ULL;o_op(10, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=5ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=4ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
}else{
if(({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=8ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
}
if(({uint64_t _a=v1;uint64_t _b=4ULL;o_op(11, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 12ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 5ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=v1;uint64_t _b=5ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=3ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 6ULL, 5);uint64_t _b=16ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
}else{
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=2ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 6ULL, 5);uint64_t _b=32ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
}
}
return o_leave(0ULL);
}
static uint64_t f1(const void *v0, uint64_t v0_n){
o_enter();
uint64_t __attribute__((unused)) v1=({f0(v0, v0_n);});
if(({uint64_t _a=v1;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v1);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 24ULL, 5);uint64_t _b=1ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 24ULL, 5);uint64_t _b=3ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 38ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 39ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 32ULL, 5);uint64_t _b=1ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 40ULL, 5);uint64_t _b=1ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 35ULL, 5);uint64_t _b=4ULL;o_op(10, 5, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v2=0;
uint64_t k1=6ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v2;uint64_t _b=42ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
}
}
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 27ULL, 5);
if(({uint64_t _a=({uint64_t _a=v2;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v2;uint64_t _b=4096ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(4ULL);
}
if(({uint64_t _a=({uint64_t _a=v2;uint64_t _b=({uint64_t _a=v2;uint64_t _b=1ULL;o_op(2, 5, _a,_b);});o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(4ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 30ULL, 5);uint64_t _b=65536ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 36ULL, 5);uint64_t _b=65536ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 31ULL, 5);uint64_t _b=1024ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 37ULL, 5);uint64_t _b=1024ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
return o_leave(0ULL);
}
static uint64_t f2(uint64_t v0, uint64_t v1){
o_enter();
if(({uint64_t _a=({uint64_t _a=v0;uint64_t _b=v1;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(({uint64_t _a=v0;uint64_t _b=v1;o_op(4, 5, _a,_b);}));
}
return o_leave(({uint64_t _a=({uint64_t _a=v0;uint64_t _b=v1;o_op(4, 5, _a,_b);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);}));
}
static uint64_t f3(uint64_t v0){
o_enter();
if(({uint64_t _a=v0;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=v0;uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=v0;uint64_t _b=4ULL;o_op(6, 5, _a,_b);})){
return o_leave(4ULL);
}
return o_leave(1ULL);
}
static uint64_t f4(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4){
o_enter();
if(({uint64_t _a=v3;uint64_t _b=v4;o_op(10, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=v1;uint64_t _b=v2;o_op(8, 5, _a,_b);})){
return o_leave(0ULL);
}
uint64_t __attribute__((unused)) v5=({uint64_t _a=v4;uint64_t _b=v3;o_op(2, 5, _a,_b);});
if(({uint64_t _a=v2;uint64_t _b=v5;o_op(10, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=v0;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=v1;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(0ULL);
}
return o_leave(({uint64_t _a=({uint64_t _a=v0;uint64_t _b=1ULL;o_op(2, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=v2;o_op(2, 5, _a,_b);});uint64_t _b=v1;o_op(4, 5, _a,_b);});o_op(9, 5, _a,_b);}));
}
static uint64_t f5(const void *v0, uint64_t v0_n){
o_enter();
uint64_t __attribute__((unused)) v1=o_get(v0, v0_n, 1ULL, 5);
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 24ULL, 5);
if(({uint64_t _a=v1;uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=v2;uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=3ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 35ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 40ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 32ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 41ULL, 5);uint64_t _b=o_get(v0, v0_n, 18ULL, 5);o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(4ULL);
}
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=2ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=v1;uint64_t _b=4ULL;o_op(8, 5, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 35ULL, 5);uint64_t _b=o_get(v0, v0_n, 4ULL, 5);o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 40ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 32ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 18ULL, 5);uint64_t _b=o_get(v0, v0_n, 6ULL, 5);o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(4ULL);
}
}
if(({uint64_t _a=o_get(v0, v0_n, 41ULL, 5);uint64_t _b=({uint64_t _a=({uint64_t p0=o_get(v0, v0_n, 18ULL, 5);uint64_t p1=o_get(v0, v0_n, 6ULL, 5);f2(p0, p1);});uint64_t _b=o_get(v0, v0_n, 6ULL, 5);o_op(3, 5, _a,_b);});o_op(7, 5, _a,_b);})){
return o_leave(4ULL);
}
return o_leave(0ULL);
}
if(({uint64_t _a=v1;uint64_t _b=2ULL;o_op(7, 5, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 35ULL, 5);uint64_t _b=4ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 5ULL, 5);uint64_t _b=128ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 6ULL, 5);uint64_t _b=128ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 40ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 12ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 17ULL, 5);uint64_t _b=128ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 18ULL, 5);uint64_t _b=128ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(4ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 32ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 41ULL, 5);uint64_t _b=o_get(v0, v0_n, 18ULL, 5);o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(4ULL);
}
return o_leave(0ULL);
}
static uint64_t f6(const void *v0, uint64_t v0_n){
o_enter();
uint64_t __attribute__((unused)) v1=o_get(v0, v0_n, 17ULL, 5);
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 18ULL, 5);
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=65536ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=v2;uint64_t _b=65536ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 41ULL, 5);uint64_t _b=65536ULL;o_op(10, 5, _a,_b);})){
return o_leave(3ULL);
}
uint64_t __attribute__((unused)) v3=({uint64_t _a=8ULL;uint64_t _b=o_get(v0, v0_n, 2ULL, 5);o_op(4, 5, _a,_b);});
uint64_t __attribute__((unused)) v4=({uint64_t p0=v2;uint64_t p1=v3;f2(p0, p1);});
if(({uint64_t _a=({uint64_t p0=v1;uint64_t p1=o_get(v0, v0_n, 25ULL, 5);uint64_t p2=v4;uint64_t p3=o_get(v0, v0_n, 28ULL, 5);uint64_t p4=o_get(v0, v0_n, 30ULL, 5);f4(p0, p1, p2, p3, p4);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(3ULL);
}
uint64_t __attribute__((unused)) v5=({uint64_t p0=o_get(v0, v0_n, 41ULL, 5);uint64_t p1=v3;f2(p0, p1);});
if(({uint64_t _a=({uint64_t p0=v1;uint64_t p1=o_get(v0, v0_n, 33ULL, 5);uint64_t p2=v5;uint64_t p3=0ULL;uint64_t p4=o_get(v0, v0_n, 36ULL, 5);f4(p0, p1, p2, p3, p4);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 33ULL, 5);uint64_t _b=o_get(v0, v0_n, 27ULL, 5);o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(4ULL);
}
if(({uint64_t _a=v1;uint64_t _b=({uint64_t _a=65536ULL;uint64_t _b=o_get(v0, v0_n, 33ULL, 5);o_op(4, 5, _a,_b);});o_op(10, 5, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 36ULL, 5);uint64_t _b=({uint64_t _a=v1;uint64_t _b=o_get(v0, v0_n, 33ULL, 5);o_op(3, 5, _a,_b);});o_op(7, 5, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 26ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 34ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(4ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 29ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 31ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(4ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 37ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(4ULL);
}
return o_leave(0ULL);
}
uint64_t __attribute__((unused)) v6=({uint64_t p0=v1;uint64_t p1=o_get(v0, v0_n, 5ULL, 5);f2(p0, p1);});
uint64_t __attribute__((unused)) v7=({uint64_t p0=v2;uint64_t p1=o_get(v0, v0_n, 6ULL, 5);f2(p0, p1);});
if(({uint64_t _a=({uint64_t p0=v6;uint64_t p1=o_get(v0, v0_n, 26ULL, 5);uint64_t p2=({uint64_t _a=v7;uint64_t _b=({uint64_t p0=o_get(v0, v0_n, 4ULL, 5);f3(p0);});o_op(3, 5, _a,_b);});uint64_t p3=o_get(v0, v0_n, 29ULL, 5);uint64_t p4=o_get(v0, v0_n, 31ULL, 5);f4(p0, p1, p2, p3, p4);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t p0=v6;uint64_t p1=o_get(v0, v0_n, 34ULL, 5);uint64_t p2=({uint64_t _a=v7;uint64_t _b=({uint64_t p0=o_get(v0, v0_n, 35ULL, 5);f3(p0);});o_op(3, 5, _a,_b);});uint64_t p3=0ULL;uint64_t p4=o_get(v0, v0_n, 37ULL, 5);f4(p0, p1, p2, p3, p4);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=v6;uint64_t _b=({uint64_t _a=1024ULL;uint64_t _b=o_get(v0, v0_n, 34ULL, 5);o_op(4, 5, _a,_b);});o_op(10, 5, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 37ULL, 5);uint64_t _b=({uint64_t _a=v6;uint64_t _b=o_get(v0, v0_n, 34ULL, 5);o_op(3, 5, _a,_b);});o_op(7, 5, _a,_b);})){
return o_leave(3ULL);
}
return o_leave(0ULL);
}
static uint64_t f7(const void *v0, uint64_t v0_n){
o_enter();
uint64_t __attribute__((unused)) v1=({f1(v0, v0_n);});
if(({uint64_t _a=v1;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v1);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 17ULL, 5);uint64_t _b=65536ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 18ULL, 5);uint64_t _b=65536ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
v1=({f5(v0, v0_n);});
if(({uint64_t _a=v1;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v1);
}
return o_leave(({f6(v0, v0_n);}));
}
static uint64_t f8(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
uint64_t __attribute__((unused)) v2=({f1(v0, v0_n);});
if(({uint64_t _a=v2;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v2);
}
{
uint64_t v3=0;
uint64_t k1=48ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
uint64_t __attribute__((unused)) v4=o_get(v0, v0_n, v3, 5);
{
uint64_t v5=0;
uint64_t k2=8ULL;
for(;o_op(8,5, v5, k2);v5=o_op(1,5, v5,1)){
{uint64_t _i=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=v5;o_op(1, 5, _a,_b);});uint64_t _v=o_cast(2, 5, ({uint64_t _a=({uint64_t _a=v4;uint64_t _b=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(16, 5, _a,_b);});uint64_t _b=255ULL;o_op(12, 5, _a,_b);}));o_put(v1, v1_n, _i,_v,2);}
}
}
}
}
return o_leave(0ULL);
}
static uint64_t f9(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
{
uint64_t v2=0;
uint64_t k1=48ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
uint64_t __attribute__((unused)) v3=0ULL;
{
uint64_t v4=0;
uint64_t k2=8ULL;
for(;o_op(8,5, v4, k2);v4=o_op(1,5, v4,1)){
v3=({uint64_t _a=v3;uint64_t _b=({uint64_t _a=o_cast(5, 2, o_get(v0, v0_n, ({uint64_t _a=({uint64_t _a=v2;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=v4;o_op(1, 5, _a,_b);}), 2));uint64_t _b=({uint64_t _a=v4;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
}
}
{uint64_t _i=v2;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
}
}
return o_leave(({f1(v1, v1_n);}));
}
static uint64_t f10(const void *v0, uint64_t v0_n){
o_enter();
uint64_t __attribute__((unused)) v1=14695981039346656037ULL;
{
uint64_t v2=0;
uint64_t k1=384ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
v1=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=o_cast(5, 2, o_get(v0, v0_n, v2, 2));o_op(14, 5, _a,_b);});uint64_t _b=1099511628211ULL;o_op(3, 5, _a,_b);});
}
}
return o_leave(v1);
}
static uint64_t f11(const void *v0, uint64_t v0_n){
o_enter();
uint64_t __attribute__((unused)) v1=14695981039346656037ULL;
{
uint64_t v2=0;
uint64_t k1=192ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
v1=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=o_cast(5, 2, o_get(v0, v0_n, v2, 2));o_op(14, 5, _a,_b);});uint64_t _b=1099511628211ULL;o_op(3, 5, _a,_b);});
}
}
return o_leave(v1);
}
static uint64_t f12(const void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
uint64_t __attribute__((unused)) v2=({uint64_t _a=o_cast(4, 2, o_get(v0, v0_n, v1, 2));uint64_t _b=({uint64_t _a=o_cast(4, 2, o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 2));uint64_t _b=8ULL;o_op(15, 4, _a,_b);});o_op(13, 4, _a,_b);});
uint64_t __attribute__((unused)) v3=({uint64_t _a=v2;uint64_t _b=16ULL;o_op(15, 4, _a,_b);});
return o_leave(({uint64_t _a=v3;uint64_t _b=({uint64_t _a=({uint64_t _a=127ULL;uint64_t _b=56ULL;o_op(1, 4, _a,_b);});uint64_t _b=23ULL;o_op(15, 4, _a,_b);});o_f32(2, _a,_b);}));
}
static uint64_t f13(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
uint64_t __attribute__((unused)) v3=({uint64_t _a=o_get(v0, v0_n, 29ULL, 5);uint64_t _b=({uint64_t _a=v1;uint64_t _b=o_get(v0, v0_n, 26ULL, 5);o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
return o_leave(({uint64_t _a=v3;uint64_t _b=({uint64_t _a=v2;uint64_t _b=({uint64_t p0=o_get(v0, v0_n, 4ULL, 5);f3(p0);});o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);}));
}
static uint64_t f14(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
uint64_t __attribute__((unused)) v3=o_cast(4, 2, o_get(v0, v0_n, v1, 2));
if(({uint64_t _a=v2;uint64_t _b=2ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=v3;uint64_t _b=255ULL;o_op(6, 4, _a,_b);})){
return o_leave(6ULL);
}
}
if(({uint64_t _a=v2;uint64_t _b=3ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=v3;uint64_t _b=126ULL;o_op(10, 4, _a,_b);})){
return o_leave(6ULL);
}
}
if(({uint64_t _a=v2;uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
v3=({uint64_t _a=v3;uint64_t _b=({uint64_t _a=o_cast(4, 2, o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 2));uint64_t _b=8ULL;o_op(15, 4, _a,_b);});o_op(13, 4, _a,_b);});
uint64_t __attribute__((unused)) v4=({uint64_t _a=v3;uint64_t _b=16ULL;o_op(15, 4, _a,_b);});
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=0;o_f32(1, _a,_b);});uint64_t _b=2137108966ULL;o_f32(3, _a,_b);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(6ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=({uint64_t p1=v1;f12(v0, v0_n, p1);});uint64_t _b=0;o_f32(1, _a,_b);});uint64_t _b=2137108966ULL;o_f32(3, _a,_b);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(6ULL);
}
}
return o_leave(0ULL);
}
static uint64_t f15(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n){
o_enter();
if(({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(0ULL);
}
uint64_t __attribute__((unused)) v2=({uint64_t p0=o_get(v0, v0_n, 17ULL, 5);uint64_t p1=o_get(v0, v0_n, 5ULL, 5);f2(p0, p1);});
uint64_t __attribute__((unused)) v3=({uint64_t p0=o_get(v0, v0_n, 18ULL, 5);uint64_t p1=o_get(v0, v0_n, 6ULL, 5);f2(p0, p1);});
{
uint64_t v4=0;
uint64_t k1=({uint64_t _a=v2;uint64_t _b=v3;o_op(3, 5, _a,_b);});
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v5=({uint64_t p1=({uint64_t _a=v4;uint64_t _b=v3;o_op(4, 5, _a,_b);});uint64_t p2=({uint64_t _a=v4;uint64_t _b=v3;o_op(5, 5, _a,_b);});f13(v0, v0_n, p1, p2);});
uint64_t __attribute__((unused)) v6=({uint64_t p1=v5;uint64_t p2=o_get(v0, v0_n, 4ULL, 5);f14(v1, v1_n, p1, p2);});
if(({uint64_t _a=v6;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v6);
}
}
}
return o_leave(0ULL);
}
static uint64_t f16(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n){
o_enter();
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 17ULL, 5);
uint64_t __attribute__((unused)) v3=o_get(v0, v0_n, 18ULL, 5);
{
uint64_t v4=0;
uint64_t c1=0;
for(;o_op(8,5, v4, v2);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v5=({uint64_t _a=o_get(v0, v0_n, 28ULL, 5);uint64_t _b=({uint64_t _a=v4;uint64_t _b=o_get(v0, v0_n, 25ULL, 5);o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
if(({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=4ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, ({uint64_t _a=v5;uint64_t _b=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(4, 5, _a,_b);});o_op(1, 5, _a,_b);}), 2);uint64_t _b=240ULL;o_op(12, 2, _a,_b);});uint64_t _b=0ULL;o_op(7, 2, _a,_b);})){
return o_leave(4ULL);
}
}
continue;
}
{
uint64_t v6=0;
uint64_t c2=0;
for(;o_op(8,5, v6, v3);v6=o_op(1,5, v6,1)){
if(c2++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v7=o_cast(4, 2, o_get(v1, v1_n, ({uint64_t _a=v5;uint64_t _b=v6;o_op(1, 5, _a,_b);}), 2));
if(({uint64_t _a=o_get(v0, v0_n, 12ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=3ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=v7;uint64_t _b=127ULL;o_op(12, 4, _a,_b);});uint64_t _b=124ULL;o_op(11, 4, _a,_b);})){
return o_leave(5ULL);
}
}else{
if(({uint64_t _a=({uint64_t _a=v7;uint64_t _b=127ULL;o_op(12, 4, _a,_b);});uint64_t _b=127ULL;o_op(6, 4, _a,_b);})){
return o_leave(5ULL);
}
}
}
}
}
}
}
return o_leave(0ULL);
}
static uint64_t f17(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=o_get(v0, v0_n, 36ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=v3;uint64_t _v=0ULL;o_put(v2, v2_n, _i,_v,2);}
}
}
uint64_t __attribute__((unused)) v3=({uint64_t p0=o_get(v0, v0_n, 18ULL, 5);uint64_t p1=({uint64_t _a=8ULL;uint64_t _b=o_get(v0, v0_n, 2ULL, 5);o_op(4, 5, _a,_b);});f2(p0, p1);});
{
uint64_t v4=0;
uint64_t k1=({uint64_t _a=o_get(v0, v0_n, 17ULL, 5);uint64_t _b=v3;o_op(3, 5, _a,_b);});
uint64_t c1=0;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v5=({uint64_t _a=o_get(v0, v0_n, 28ULL, 5);uint64_t _b=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=v3;o_op(4, 5, _a,_b);});uint64_t _b=o_get(v0, v0_n, 25ULL, 5);o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
v5=({uint64_t _a=v5;uint64_t _b=({uint64_t _a=v4;uint64_t _b=v3;o_op(5, 5, _a,_b);});o_op(1, 5, _a,_b);});
uint64_t __attribute__((unused)) v6=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=v3;o_op(4, 5, _a,_b);});uint64_t _b=o_get(v0, v0_n, 33ULL, 5);o_op(3, 5, _a,_b);});
v6=({uint64_t _a=v6;uint64_t _b=({uint64_t _a=v4;uint64_t _b=v3;o_op(5, 5, _a,_b);});o_op(1, 5, _a,_b);});
{uint64_t _i=v6;uint64_t _v=o_get(v1, v1_n, v5, 2);o_put(v2, v2_n, _i,_v,2);}
}
}
return o_leave(0);
}
static uint64_t f18(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, uint64_t v4){
o_enter();
if(v4){
uint64_t __attribute__((unused)) v5=({uint64_t p1=v2;f12(v0, v0_n, p1);});
{
uint64_t v6=0;
uint64_t k2=4ULL;
for(;o_op(8,5, v6, k2);v6=o_op(1,5, v6,1)){
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=v6;o_op(1, 5, _a,_b);});uint64_t _v=o_cast(2, 4, ({uint64_t _a=({uint64_t _a=v5;uint64_t _b=({uint64_t _a=v6;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(16, 4, _a,_b);});uint64_t _b=255ULL;o_op(12, 4, _a,_b);}));o_put(v1, v1_n, _i,_v,2);}
}
}
}else{
{uint64_t _i=v3;uint64_t _v=o_get(v0, v0_n, v2, 2);o_put(v1, v1_n, _i,_v,2);}
}
return o_leave(0);
}
static uint64_t f19(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=o_get(v0, v0_n, 37ULL, 5);
uint64_t c1=0;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=v3;uint64_t _v=0ULL;o_put(v2, v2_n, _i,_v,2);}
}
}
if(({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(0);
}
uint64_t __attribute__((unused)) v3=({uint64_t p0=o_get(v0, v0_n, 17ULL, 5);uint64_t p1=o_get(v0, v0_n, 5ULL, 5);f2(p0, p1);});
uint64_t __attribute__((unused)) v4=({uint64_t p0=o_get(v0, v0_n, 18ULL, 5);uint64_t p1=o_get(v0, v0_n, 6ULL, 5);f2(p0, p1);});
{
uint64_t v5=0;
uint64_t k1=({uint64_t _a=v3;uint64_t _b=v4;o_op(3, 5, _a,_b);});
uint64_t c1=0;
for(;o_op(8,5, v5, k1);v5=o_op(1,5, v5,1)){
if(c1++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v6=({uint64_t p1=({uint64_t _a=v5;uint64_t _b=v4;o_op(4, 5, _a,_b);});uint64_t p2=({uint64_t _a=v5;uint64_t _b=v4;o_op(5, 5, _a,_b);});f13(v0, v0_n, p1, p2);});
uint64_t __attribute__((unused)) v7=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=v4;o_op(4, 5, _a,_b);});uint64_t _b=o_get(v0, v0_n, 34ULL, 5);o_op(3, 5, _a,_b);});
v7=({uint64_t _a=v7;uint64_t _b=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=v4;o_op(5, 5, _a,_b);});uint64_t _b=({uint64_t p0=o_get(v0, v0_n, 35ULL, 5);f3(p0);});o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
({uint64_t p2=v6;uint64_t p3=v7;uint64_t p4=({uint64_t _a=o_get(v0, v0_n, 24ULL, 5);uint64_t _b=3ULL;o_op(6, 5, _a,_b);});f18(v1, v1_n, v2, v2_n, p2, p3, p4);});
}
}
return o_leave(0);
}
static uint64_t f20(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, void *v4, uint64_t v4_n){
o_enter();
uint64_t __attribute__((unused)) v5=({f7(v0, v0_n);});
if(({uint64_t _a=v5;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v5);
}
v5=({f16(v0, v0_n, v1, v1_n);});
if(({uint64_t _a=v5;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v5);
}
v5=({f15(v0, v0_n, v2, v2_n);});
if(({uint64_t _a=v5;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v5);
}
({f17(v0, v0_n, v1, v1_n, v3, v3_n);});
({f19(v0, v0_n, v2, v2_n, v4, v4_n);});
return o_leave(0ULL);
}
static uint64_t f21(const void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
uint64_t __attribute__((unused)) v2=0ULL;
{
uint64_t v3=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
v2=({uint64_t _a=v2;uint64_t _b=({uint64_t _a=o_cast(5, 2, o_get(v0, v0_n, ({uint64_t _a=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=v3;o_op(1, 5, _a,_b);}), 2));uint64_t _b=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
}
}
return o_leave(v2);
}
static uint64_t f22(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
{
uint64_t v3=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
{uint64_t _i=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});uint64_t _b=v3;o_op(1, 5, _a,_b);});uint64_t _v=o_cast(2, 5, ({uint64_t _a=({uint64_t _a=v2;uint64_t _b=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(16, 5, _a,_b);});uint64_t _b=255ULL;o_op(12, 5, _a,_b);}));o_put(v0, v0_n, _i,_v,2);}
}
}
return o_leave(0);
}
static uint64_t f23(const void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n){
o_enter();
if(({uint64_t _a=v1;uint64_t _b=8ULL;o_op(11, 5, _a,_b);})){
return o_leave(0ULL);
}
{
uint64_t v3=0;
uint64_t k1=32ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=({uint64_t _a=512ULL;uint64_t _b=({uint64_t _a=v1;uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _b=v3;o_op(1, 5, _a,_b);}), 2);uint64_t _b=o_get(v2, v2_n, v3, 2);o_op(7, 2, _a,_b);})){
return o_leave(0ULL);
}
}
}
return o_leave(1ULL);
}
static uint64_t f24(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n){
o_enter();
return o_leave(({uint64_t p1=5ULL;f23(v0, v0_n, p1, v1, v1_n);}));
}
static uint64_t f25(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n){
o_enter();
return o_leave(({uint64_t p1=6ULL;f23(v0, v0_n, p1, v1, v1_n);}));
}
static uint64_t f26(const void *v0, uint64_t v0_n){
o_enter();
uint64_t __attribute__((unused)) v1=({uint64_t p1=14ULL;f21(v0, v0_n, p1);});
uint64_t __attribute__((unused)) v2=({uint64_t p1=16ULL;f21(v0, v0_n, p1);});
uint64_t __attribute__((unused)) v3=({uint64_t p1=17ULL;f21(v0, v0_n, p1);});
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=2ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);});o_op(12, 1, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t p1=15ULL;f21(v0, v0_n, p1);});uint64_t _b=1ULL;o_op(7, 5, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=v2;uint64_t _b=256ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=v2;uint64_t _b=512ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=v3;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v3;uint64_t _b=32ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=v3;uint64_t _b=({uint64_t _a=v3;uint64_t _b=1ULL;o_op(2, 5, _a,_b);});o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(3ULL);
}
uint64_t __attribute__((unused)) v4=({uint64_t _a=v2;uint64_t _b=32ULL;o_op(4, 5, _a,_b);});
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=v4;o_op(3, 5, _a,_b);});uint64_t _b=v3;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=v4;o_op(3, 5, _a,_b);});uint64_t _b=v3;o_op(4, 5, _a,_b);});uint64_t _b=8ULL;o_op(10, 5, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=6ULL;f21(v0, v0_n, p1);});uint64_t _b=16384ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=({uint64_t p1=6ULL;f21(v0, v0_n, p1);});uint64_t _b=v2;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t p1=6ULL;f21(v0, v0_n, p1);});uint64_t _b=v2;o_op(4, 5, _a,_b);});
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=2ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=v5;uint64_t _b=3ULL;o_op(10, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(2ULL);
}
uint64_t __attribute__((unused)) v6=v5;
if(({uint64_t _a=({uint64_t _a=v5;uint64_t _b=2ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
v6=2ULL;
}
if(({uint64_t _a=({uint64_t p1=18ULL;f21(v0, v0_n, p1);});uint64_t _b=v6;o_op(7, 5, _a,_b);})){
return o_leave(2ULL);
}
uint64_t __attribute__((unused)) v7=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=128ULL;o_op(3, 5, _a,_b);});uint64_t _b=v4;o_op(3, 5, _a,_b);});
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=21ULL;f21(v0, v0_n, p1);});uint64_t _b=v7;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=({uint64_t p1=21ULL;f21(v0, v0_n, p1);});uint64_t _b=128ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=22ULL;f21(v0, v0_n, p1);});uint64_t _b=16ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=({uint64_t p1=22ULL;f21(v0, v0_n, p1);});uint64_t _b=8ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=23ULL;f21(v0, v0_n, p1);});uint64_t _b=6ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=24ULL;f21(v0, v0_n, p1);});uint64_t _b=1ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=26ULL;f21(v0, v0_n, p1);});uint64_t _b=4294967295ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=27ULL;f21(v0, v0_n, p1);});uint64_t _b=4294967295ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=28ULL;f21(v0, v0_n, p1);});uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=({uint64_t p1=26ULL;f21(v0, v0_n, p1);});uint64_t _b=v3;o_op(3, 5, _a,_b);});uint64_t _b=4294967295ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=29ULL;f21(v0, v0_n, p1);});uint64_t _b=7ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=32ULL;f21(v0, v0_n, p1);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
{
uint64_t v8=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v8, k1);v8=o_op(1,5, v8,1)){
if(({uint64_t _a=v8;uint64_t _b=2ULL;o_op(7, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t p1=({uint64_t _a=v8;uint64_t _b=30ULL;o_op(1, 5, _a,_b);});f21(v0, v0_n, p1);});uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(2ULL);
}
}
}
}
return o_leave(0ULL);
}
static uint64_t f27(const void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
uint64_t v2[48]={0};
uint64_t v2_n=48;
o_fill(v2, v2_n, 0ULL, 5);
uint8_t v3[384]={0};
uint64_t v3_n=384;
o_fill(v3, v3_n, 0ULL, 2);
if(({uint64_t _a=v1;uint64_t _b=1152ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=0ULL;f21(v0, v0_n, p1);});uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=1ULL;f21(v0, v0_n, p1);});uint64_t _b=1152ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t p1=2ULL;f21(v0, v0_n, p1);});uint64_t _b=121ULL;o_op(7, 5, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t p1=3ULL;f21(v0, v0_n, p1);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(2ULL);
}
{
uint64_t v4=0;
uint64_t k1=3ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
uint64_t __attribute__((unused)) v5=({uint64_t p1=({uint64_t _a=v4;uint64_t _b=4ULL;o_op(1, 5, _a,_b);});f21(v0, v0_n, p1);});
if(({uint64_t _a=({uint64_t _a=v5;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v5;uint64_t _b=16777216ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
}
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=7ULL;f21(v0, v0_n, p1);});uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=8ULL;f21(v0, v0_n, p1);});uint64_t _b=3ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=9ULL;f21(v0, v0_n, p1);});uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=10ULL;f21(v0, v0_n, p1);});uint64_t _b=1ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=11ULL;f21(v0, v0_n, p1);});uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=12ULL;f21(v0, v0_n, p1);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t p1=13ULL;f21(v0, v0_n, p1);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(2ULL);
}
{
uint64_t v4=0;
uint64_t k1=6ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(({uint64_t _a=({uint64_t p1=({uint64_t _a=v4;uint64_t _b=14ULL;o_op(1, 5, _a,_b);});f21(v0, v0_n, p1);});uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(3ULL);
}
}
}
if(({uint64_t _a=({uint64_t p1=17ULL;f21(v0, v0_n, p1);});uint64_t _b=32ULL;o_op(10, 5, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t p1=18ULL;f21(v0, v0_n, p1);});uint64_t _b=16ULL;o_op(10, 5, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t p1=19ULL;f21(v0, v0_n, p1);});uint64_t _b=3ULL;o_op(10, 5, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t p1=20ULL;f21(v0, v0_n, p1);});uint64_t _b=127ULL;o_op(10, 5, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t p1=21ULL;f21(v0, v0_n, p1);});uint64_t _b=16384ULL;o_op(10, 5, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=22ULL;f21(v0, v0_n, p1);});uint64_t _b=3ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=22ULL;f21(v0, v0_n, p1);});uint64_t _b=246ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=23ULL;f21(v0, v0_n, p1);});uint64_t _b=4ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=23ULL;f21(v0, v0_n, p1);});uint64_t _b=32ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t p1=24ULL;f21(v0, v0_n, p1);});uint64_t _b=16ULL;o_op(10, 5, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t p1=25ULL;f21(v0, v0_n, p1);});uint64_t _b=({uint64_t _a=({uint64_t p1=17ULL;f21(v0, v0_n, p1);});uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(7, 5, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t p1=25ULL;f21(v0, v0_n, p1);});uint64_t _b=({uint64_t p1=22ULL;f21(v0, v0_n, p1);});o_op(3, 5, _a,_b);});uint64_t _b=65536ULL;o_op(10, 5, _a,_b);})){
return o_leave(3ULL);
}
{
uint64_t v4=0;
uint64_t k1=3ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(({uint64_t _a=({uint64_t p1=({uint64_t _a=v4;uint64_t _b=26ULL;o_op(1, 5, _a,_b);});f21(v0, v0_n, p1);});uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(3ULL);
}
}
}
uint64_t __attribute__((unused)) v4=0ULL;
{
uint64_t v5=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v5, k1);v5=o_op(1,5, v5,1)){
if(({uint64_t _a=({uint64_t p1=({uint64_t _a=v5;uint64_t _b=30ULL;o_op(1, 5, _a,_b);});f21(v0, v0_n, p1);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
v4=({uint64_t _a=v4;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
}
}
}
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=({uint64_t p1=29ULL;f21(v0, v0_n, p1);});o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v5=0;
uint64_t k1=26ULL;
for(;o_op(8,5, v5, k1);v5=o_op(1,5, v5,1)){
if(({uint64_t _a=({uint64_t p1=({uint64_t _a=v5;uint64_t _b=38ULL;o_op(1, 5, _a,_b);});f21(v0, v0_n, p1);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
}
}
{
uint64_t v5=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v5, k1);v5=o_op(1,5, v5,1)){
uint64_t __attribute__((unused)) v6=0ULL;
{
uint64_t v7=0;
uint64_t k2=32ULL;
for(;o_op(8,5, v7, k2);v7=o_op(1,5, v7,1)){
v6=({uint64_t _a=v6;uint64_t _b=o_get(v0, v0_n, ({uint64_t _a=({uint64_t _a=512ULL;uint64_t _b=({uint64_t _a=v5;uint64_t _b=32ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _b=v7;o_op(1, 5, _a,_b);}), 2);o_op(13, 2, _a,_b);});
}
}
if(({uint64_t _a=v6;uint64_t _b=0ULL;o_op(6, 2, _a,_b);})){
return o_leave(1ULL);
}
}
}
{
uint64_t v5=0;
uint64_t k1=384ULL;
for(;o_op(8,5, v5, k1);v5=o_op(1,5, v5,1)){
{uint64_t _i=v5;uint64_t _v=o_get(v0, v0_n, ({uint64_t _a=v5;uint64_t _b=768ULL;o_op(1, 5, _a,_b);}), 2);o_put(v3, v3_n, _i,_v,2);}
}
}
uint64_t __attribute__((unused)) v5=({f9(v3, v3_n, v2, v2_n);});
if(({uint64_t _a=v5;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(4ULL);
}
v5=({f7(v2, v2_n);});
if(({uint64_t _a=v5;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(4ULL);
}
if(({uint64_t _a=({uint64_t p1=19ULL;f21(v0, v0_n, p1);});uint64_t _b=3ULL;o_op(6, 5, _a,_b);})){
v5=({f26(v0, v0_n);});
if(({uint64_t _a=v5;uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(v5);
}
}
if(({uint64_t _a=({uint64_t _a=o_get(v2, v2_n, 1ULL, 5);uint64_t _b=2ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v2, v2_n, 24ULL, 5);uint64_t _b=3ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v2, v2_n, 40ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v2, v2_n, 32ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
{
uint64_t v6=0;
uint64_t k1=2944ULL;
for(;o_op(8,5, v6, k1);v6=o_op(1,5, v6,1)){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v6;uint64_t _b=1152ULL;o_op(1, 5, _a,_b);}), 2);uint64_t _b=0ULL;o_op(7, 2, _a,_b);})){
return o_leave(1ULL);
}
}
}
return o_leave(0ULL);
}
static uint64_t f28(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n){
o_enter();
if(({uint64_t _a=({uint64_t p1=1152ULL;f27(v0, v0_n, p1);});uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t p1=1152ULL;f27(v1, v1_n, p1);});uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(0ULL);
}
{
uint64_t v2=0;
uint64_t k1=12ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
if(({uint64_t _a=({uint64_t p1=({uint64_t _a=v2;uint64_t _b=2ULL;o_op(1, 5, _a,_b);});f21(v0, v0_n, p1);});uint64_t _b=({uint64_t p1=({uint64_t _a=v2;uint64_t _b=2ULL;o_op(1, 5, _a,_b);});f21(v1, v1_n, p1);});o_op(7, 5, _a,_b);})){
return o_leave(0ULL);
}
}
}
{
uint64_t v2=0;
uint64_t k1=128ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v2;uint64_t _b=512ULL;o_op(1, 5, _a,_b);}), 2);uint64_t _b=o_get(v1, v1_n, ({uint64_t _a=v2;uint64_t _b=512ULL;o_op(1, 5, _a,_b);}), 2);o_op(7, 2, _a,_b);})){
return o_leave(0ULL);
}
}
}
{
uint64_t v2=0;
uint64_t k1=384ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
if(({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v2;uint64_t _b=768ULL;o_op(1, 5, _a,_b);}), 2);uint64_t _b=o_get(v1, v1_n, ({uint64_t _a=v2;uint64_t _b=768ULL;o_op(1, 5, _a,_b);}), 2);o_op(7, 2, _a,_b);})){
return o_leave(0ULL);
}
}
}
return o_leave(1ULL);
}
static uint64_t f29(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n){
o_enter();
if(({uint64_t _a=({uint64_t p1=1152ULL;f27(v0, v0_n, p1);});uint64_t _b=0ULL;o_op(7, 4, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 0ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 4ULL, 5);uint64_t _b=({uint64_t p1=25ULL;f21(v0, v0_n, p1);});o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 5ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 6ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v2=0;
uint64_t k1=3ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
if(({uint64_t _a=o_get(v1, v1_n, ({uint64_t _a=v2;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=({uint64_t p1=({uint64_t _a=v2;uint64_t _b=26ULL;o_op(1, 5, _a,_b);});f21(v0, v0_n, p1);});o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
}
}
{
uint64_t v2=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
if(({uint64_t _a=o_get(v1, v1_n, ({uint64_t _a=v2;uint64_t _b=7ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=({uint64_t p1=({uint64_t _a=v2;uint64_t _b=21ULL;o_op(1, 5, _a,_b);});f21(v0, v0_n, p1);});o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
}
}
if(({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 11ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 12ULL, 5);uint64_t _b=32ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 12ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 13ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v1, v1_n, 14ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t p1=19ULL;f21(v0, v0_n, p1);});uint64_t _b=3ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=o_get(v1, v1_n, 11ULL, 5);uint64_t _b=896ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 12ULL, 5);uint64_t _b=8ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v1, v1_n, 12ULL, 5);uint64_t _b=12ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v1, v1_n, 12ULL, 5);uint64_t _b=12ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=14ULL;f21(v0, v0_n, p1);});uint64_t _b=4ULL;o_op(10, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
}
{
uint64_t v2=0;
uint64_t k1=3ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
uint64_t __attribute__((unused)) v3=0ULL;
{
uint64_t v4=0;
uint64_t k2=4ULL;
for(;o_op(8,5, v4, k2);v4=o_op(1,5, v4,1)){
v3=({uint64_t _a=v3;uint64_t _b=o_get(v1, v1_n, ({uint64_t _a=({uint64_t _a=15ULL;uint64_t _b=({uint64_t _a=v2;uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _b=v4;o_op(1, 5, _a,_b);}), 5);o_op(13, 5, _a,_b);});
}
}
if(({uint64_t _a=v3;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(1ULL);
}
}
}
{
uint64_t v2=0;
uint64_t k1=5ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
if(({uint64_t _a=o_get(v1, v1_n, ({uint64_t _a=v2;uint64_t _b=27ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
}
}
return o_leave(0ULL);
}
static uint64_t f30(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2){
o_enter();
uint64_t v3[32]={0};
uint64_t v3_n=32;
o_fill(v3, v3_n, 0ULL, 5);
if(({uint64_t _a=v2;uint64_t _b=256ULL;o_op(7, 5, _a,_b);})){
return o_leave(1ULL);
}
{
uint64_t v4=0;
uint64_t k1=32ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
{uint64_t _i=v4;uint64_t _v=({uint64_t p1=v4;f21(v1, v1_n, p1);});o_put(v3, v3_n, _i,_v,5);}
}
}
{
uint64_t v4=0;
uint64_t k1=3840ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(({uint64_t _a=o_get(v1, v1_n, ({uint64_t _a=v4;uint64_t _b=256ULL;o_op(1, 5, _a,_b);}), 2);uint64_t _b=0ULL;o_op(7, 2, _a,_b);})){
return o_leave(1ULL);
}
}
}
return o_leave(({f29(v0, v0_n, v3, v3_n);}));
}
static uint64_t f31(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
{
uint64_t v2=0;
uint64_t k1=4096ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
{uint64_t _i=v2;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,2);}
}
}
{
uint64_t v2=0;
uint64_t k1=32ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
({uint64_t p1=v2;uint64_t p2=o_get(v0, v0_n, v2, 5);f22(v1, v1_n, p1, p2);});
}
}
return o_leave(0);
}
static uint64_t f32(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
{
uint64_t v2=0;
uint64_t k1=384ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
{uint64_t _i=v2;uint64_t _v=o_get(v0, v0_n, ({uint64_t _a=v2;uint64_t _b=768ULL;o_op(1, 5, _a,_b);}), 2);o_put(v1, v1_n, _i,_v,2);}
}
}
return o_leave(0);
}
static uint64_t f33(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n){
o_enter();
{
uint64_t v2=0;
uint64_t k1=384ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
{uint64_t _i=({uint64_t _a=v2;uint64_t _b=768ULL;o_op(1, 5, _a,_b);});uint64_t _v=o_get(v1, v1_n, v2, 2);o_put(v0, v0_n, _i,_v,2);}
}
}
return o_leave(0);
}
static uint64_t f34(uint64_t v0, uint64_t v1){
o_enter();
return o_leave(({uint64_t _a=({uint64_t _a=v0;uint64_t _b=v1;o_op(16, 4, _a,_b);});uint64_t _b=({uint64_t _a=v0;uint64_t _b=({uint64_t _a=32ULL;uint64_t _b=v1;o_op(2, 4, _a,_b);});o_op(15, 4, _a,_b);});o_op(13, 4, _a,_b);}));
}
static uint64_t f35(void *v0, uint64_t v0_n){
o_enter();
{uint64_t _i=0ULL;uint64_t _v=1116352408ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=1ULL;uint64_t _v=1899447441ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=2ULL;uint64_t _v=3049323471ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=3ULL;uint64_t _v=3921009573ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=4ULL;uint64_t _v=961987163ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=5ULL;uint64_t _v=1508970993ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=6ULL;uint64_t _v=2453635748ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=7ULL;uint64_t _v=2870763221ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=8ULL;uint64_t _v=3624381080ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=9ULL;uint64_t _v=310598401ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=10ULL;uint64_t _v=607225278ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=11ULL;uint64_t _v=1426881987ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=12ULL;uint64_t _v=1925078388ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=13ULL;uint64_t _v=2162078206ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=14ULL;uint64_t _v=2614888103ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=15ULL;uint64_t _v=3248222580ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=16ULL;uint64_t _v=3835390401ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=17ULL;uint64_t _v=4022224774ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=18ULL;uint64_t _v=264347078ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=19ULL;uint64_t _v=604807628ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=20ULL;uint64_t _v=770255983ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=21ULL;uint64_t _v=1249150122ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=22ULL;uint64_t _v=1555081692ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=23ULL;uint64_t _v=1996064986ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=24ULL;uint64_t _v=2554220882ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=25ULL;uint64_t _v=2821834349ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=26ULL;uint64_t _v=2952996808ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=27ULL;uint64_t _v=3210313671ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=28ULL;uint64_t _v=3336571891ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=29ULL;uint64_t _v=3584528711ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=30ULL;uint64_t _v=113926993ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=31ULL;uint64_t _v=338241895ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=32ULL;uint64_t _v=666307205ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=33ULL;uint64_t _v=773529912ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=34ULL;uint64_t _v=1294757372ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=35ULL;uint64_t _v=1396182291ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=36ULL;uint64_t _v=1695183700ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=37ULL;uint64_t _v=1986661051ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=38ULL;uint64_t _v=2177026350ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=39ULL;uint64_t _v=2456956037ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=40ULL;uint64_t _v=2730485921ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=41ULL;uint64_t _v=2820302411ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=42ULL;uint64_t _v=3259730800ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=43ULL;uint64_t _v=3345764771ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=44ULL;uint64_t _v=3516065817ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=45ULL;uint64_t _v=3600352804ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=46ULL;uint64_t _v=4094571909ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=47ULL;uint64_t _v=275423344ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=48ULL;uint64_t _v=430227734ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=49ULL;uint64_t _v=506948616ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=50ULL;uint64_t _v=659060556ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=51ULL;uint64_t _v=883997877ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=52ULL;uint64_t _v=958139571ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=53ULL;uint64_t _v=1322822218ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=54ULL;uint64_t _v=1537002063ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=55ULL;uint64_t _v=1747873779ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=56ULL;uint64_t _v=1955562222ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=57ULL;uint64_t _v=2024104815ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=58ULL;uint64_t _v=2227730452ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=59ULL;uint64_t _v=2361852424ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=60ULL;uint64_t _v=2428436474ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=61ULL;uint64_t _v=2756734187ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=62ULL;uint64_t _v=3204031479ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=63ULL;uint64_t _v=3329325298ULL;o_put(v0, v0_n, _i,_v,4);}
return o_leave(0);
}
static uint64_t f36(void *v0, uint64_t v0_n){
o_enter();
{uint64_t _i=0ULL;uint64_t _v=1779033703ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=1ULL;uint64_t _v=3144134277ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=2ULL;uint64_t _v=1013904242ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=3ULL;uint64_t _v=2773480762ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=4ULL;uint64_t _v=1359893119ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=5ULL;uint64_t _v=2600822924ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=6ULL;uint64_t _v=528734635ULL;o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=7ULL;uint64_t _v=1541459225ULL;o_put(v0, v0_n, _i,_v,4);}
return o_leave(0);
}
static uint64_t f37(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2){
o_enter();
uint64_t __attribute__((unused)) v3=0ULL;
uint64_t __attribute__((unused)) v4=0ULL;
{
uint64_t v5=0;
uint64_t k1=16ULL;
for(;o_op(8,5, v5, k1);v5=o_op(1,5, v5,1)){
v3=({uint64_t _a=v2;uint64_t _b=({uint64_t _a=v5;uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
v4=o_cast(4, 2, o_get(v1, v1_n, v3, 2));
v4=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=8ULL;o_op(15, 4, _a,_b);});uint64_t _b=o_cast(4, 2, o_get(v1, v1_n, ({uint64_t _a=v3;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 2));o_op(13, 4, _a,_b);});
v4=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=8ULL;o_op(15, 4, _a,_b);});uint64_t _b=o_cast(4, 2, o_get(v1, v1_n, ({uint64_t _a=v3;uint64_t _b=2ULL;o_op(1, 5, _a,_b);}), 2));o_op(13, 4, _a,_b);});
{uint64_t _i=v5;uint64_t _v=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=8ULL;o_op(15, 4, _a,_b);});uint64_t _b=o_cast(4, 2, o_get(v1, v1_n, ({uint64_t _a=v3;uint64_t _b=3ULL;o_op(1, 5, _a,_b);}), 2));o_op(13, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
}
}
({f40(v0, v0_n);});
return o_leave(0);
}
static uint64_t f38(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2){
o_enter();
uint64_t __attribute__((unused)) v3=0ULL;
uint64_t __attribute__((unused)) v4=0ULL;
{
uint64_t v5=0;
uint64_t k1=16ULL;
for(;o_op(8,5, v5, k1);v5=o_op(1,5, v5,1)){
v3=({uint64_t _a=v2;uint64_t _b=({uint64_t _a=v5;uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
v4=o_cast(4, 2, o_get(v1, v1_n, v3, 2));
v4=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=8ULL;o_op(15, 4, _a,_b);});uint64_t _b=o_cast(4, 2, o_get(v1, v1_n, ({uint64_t _a=v3;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 2));o_op(13, 4, _a,_b);});
v4=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=8ULL;o_op(15, 4, _a,_b);});uint64_t _b=o_cast(4, 2, o_get(v1, v1_n, ({uint64_t _a=v3;uint64_t _b=2ULL;o_op(1, 5, _a,_b);}), 2));o_op(13, 4, _a,_b);});
{uint64_t _i=v5;uint64_t _v=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=8ULL;o_op(15, 4, _a,_b);});uint64_t _b=o_cast(4, 2, o_get(v1, v1_n, ({uint64_t _a=v3;uint64_t _b=3ULL;o_op(1, 5, _a,_b);}), 2));o_op(13, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
}
}
({f40(v0, v0_n);});
return o_leave(0);
}
static uint64_t f39(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2){
o_enter();
uint64_t __attribute__((unused)) v3=0ULL;
uint64_t __attribute__((unused)) v4=0ULL;
{
uint64_t v5=0;
uint64_t k1=16ULL;
for(;o_op(8,5, v5, k1);v5=o_op(1,5, v5,1)){
v3=({uint64_t _a=v2;uint64_t _b=({uint64_t _a=v5;uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
v4=o_cast(4, 2, o_get(v1, v1_n, v3, 2));
v4=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=8ULL;o_op(15, 4, _a,_b);});uint64_t _b=o_cast(4, 2, o_get(v1, v1_n, ({uint64_t _a=v3;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 2));o_op(13, 4, _a,_b);});
v4=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=8ULL;o_op(15, 4, _a,_b);});uint64_t _b=o_cast(4, 2, o_get(v1, v1_n, ({uint64_t _a=v3;uint64_t _b=2ULL;o_op(1, 5, _a,_b);}), 2));o_op(13, 4, _a,_b);});
{uint64_t _i=v5;uint64_t _v=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=8ULL;o_op(15, 4, _a,_b);});uint64_t _b=o_cast(4, 2, o_get(v1, v1_n, ({uint64_t _a=v3;uint64_t _b=3ULL;o_op(1, 5, _a,_b);}), 2));o_op(13, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
}
}
({f40(v0, v0_n);});
return o_leave(0);
}
static uint64_t f40(void *v0, uint64_t v0_n){
o_enter();
uint64_t __attribute__((unused)) v1=0ULL;
uint64_t __attribute__((unused)) v2=0ULL;
{
uint64_t v3=0;
uint64_t k1=48ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
v1=o_get(v0, v0_n, ({uint64_t _a=v3;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 4);
v2=o_get(v0, v0_n, ({uint64_t _a=v3;uint64_t _b=14ULL;o_op(1, 5, _a,_b);}), 4);
v1=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, v3, 4);uint64_t _b=o_get(v0, v0_n, ({uint64_t _a=v3;uint64_t _b=9ULL;o_op(1, 5, _a,_b);}), 4);o_op(1, 4, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=({uint64_t p0=v1;uint64_t p1=7ULL;f34(p0, p1);});uint64_t _b=({uint64_t p0=v1;uint64_t p1=18ULL;f34(p0, p1);});o_op(14, 4, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=3ULL;o_op(16, 4, _a,_b);});o_op(14, 4, _a,_b);});o_op(1, 4, _a,_b);});
{uint64_t _i=({uint64_t _a=v3;uint64_t _b=16ULL;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t _a=v1;uint64_t _b=({uint64_t _a=({uint64_t _a=({uint64_t p0=v2;uint64_t p1=17ULL;f34(p0, p1);});uint64_t _b=({uint64_t p0=v2;uint64_t p1=19ULL;f34(p0, p1);});o_op(14, 4, _a,_b);});uint64_t _b=({uint64_t _a=v2;uint64_t _b=10ULL;o_op(16, 4, _a,_b);});o_op(14, 4, _a,_b);});o_op(1, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
}
}
return o_leave(0);
}
static uint64_t f41(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=o_get(v0, v0_n, 0ULL, 4);
uint64_t __attribute__((unused)) v4=o_get(v0, v0_n, 1ULL, 4);
uint64_t __attribute__((unused)) v5=o_get(v0, v0_n, 2ULL, 4);
uint64_t __attribute__((unused)) v6=o_get(v0, v0_n, 3ULL, 4);
uint64_t __attribute__((unused)) v7=o_get(v0, v0_n, 4ULL, 4);
uint64_t __attribute__((unused)) v8=o_get(v0, v0_n, 5ULL, 4);
uint64_t __attribute__((unused)) v9=o_get(v0, v0_n, 6ULL, 4);
uint64_t __attribute__((unused)) v10=o_get(v0, v0_n, 7ULL, 4);
uint64_t __attribute__((unused)) v11=0ULL;
{
uint64_t v12=0;
uint64_t k1=64ULL;
for(;o_op(8,5, v12, k1);v12=o_op(1,5, v12,1)){
v10=({uint64_t _a=v10;uint64_t _b=o_get(v2, v2_n, v12, 4);o_op(1, 4, _a,_b);});
v10=({uint64_t _a=v10;uint64_t _b=o_get(v1, v1_n, v12, 4);o_op(1, 4, _a,_b);});
v11=({uint64_t _a=({uint64_t p0=v7;uint64_t p1=6ULL;f34(p0, p1);});uint64_t _b=({uint64_t p0=v7;uint64_t p1=11ULL;f34(p0, p1);});o_op(14, 4, _a,_b);});
v11=({uint64_t _a=v11;uint64_t _b=({uint64_t p0=v7;uint64_t p1=25ULL;f34(p0, p1);});o_op(14, 4, _a,_b);});
v10=({uint64_t _a=v10;uint64_t _b=v11;o_op(1, 4, _a,_b);});
v10=({uint64_t _a=v10;uint64_t _b=({uint64_t _a=v9;uint64_t _b=({uint64_t _a=v7;uint64_t _b=({uint64_t _a=v8;uint64_t _b=v9;o_op(14, 4, _a,_b);});o_op(12, 4, _a,_b);});o_op(14, 4, _a,_b);});o_op(1, 4, _a,_b);});
v6=({uint64_t _a=v6;uint64_t _b=v10;o_op(1, 4, _a,_b);});
v11=({uint64_t _a=({uint64_t p0=v3;uint64_t p1=2ULL;f34(p0, p1);});uint64_t _b=({uint64_t p0=v3;uint64_t p1=13ULL;f34(p0, p1);});o_op(14, 4, _a,_b);});
v11=({uint64_t _a=v11;uint64_t _b=({uint64_t p0=v3;uint64_t p1=22ULL;f34(p0, p1);});o_op(14, 4, _a,_b);});
v10=({uint64_t _a=v10;uint64_t _b=v11;o_op(1, 4, _a,_b);});
v11=({uint64_t _a=v5;uint64_t _b=({uint64_t _a=v3;uint64_t _b=v4;o_op(13, 4, _a,_b);});o_op(12, 4, _a,_b);});
v10=({uint64_t _a=v10;uint64_t _b=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=v4;o_op(12, 4, _a,_b);});uint64_t _b=v11;o_op(13, 4, _a,_b);});o_op(1, 4, _a,_b);});
v11=v10;
v10=v9;
v9=v8;
v8=v7;
v7=v6;
v6=v5;
v5=v4;
v4=v3;
v3=v11;
}
}
{uint64_t _i=0ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 0ULL, 4);uint64_t _b=v3;o_op(1, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=1ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 1ULL, 4);uint64_t _b=v4;o_op(1, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=2ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 2ULL, 4);uint64_t _b=v5;o_op(1, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=3ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 3ULL, 4);uint64_t _b=v6;o_op(1, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=4ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 4ULL, 4);uint64_t _b=v7;o_op(1, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=5ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 5ULL, 4);uint64_t _b=v8;o_op(1, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=6ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 6ULL, 4);uint64_t _b=v9;o_op(1, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
{uint64_t _i=7ULL;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, 7ULL, 4);uint64_t _b=v10;o_op(1, 4, _a,_b);});o_put(v0, v0_n, _i,_v,4);}
return o_leave(0);
}
static uint64_t f42(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
{
uint64_t v2=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v2, k1);v2=o_op(1,5, v2,1)){
{
uint64_t v3=0;
uint64_t k2=4ULL;
for(;o_op(8,5, v3, k2);v3=o_op(1,5, v3,1)){
uint64_t __attribute__((unused)) v4=({uint64_t _a=({uint64_t _a=v2;uint64_t _b=4ULL;o_op(3, 5, _a,_b);});uint64_t _b=v3;o_op(1, 5, _a,_b);});
uint64_t __attribute__((unused)) v5=({uint64_t _a=o_get(v0, v0_n, v2, 4);uint64_t _b=({uint64_t _a=({uint64_t _a=3ULL;uint64_t _b=v3;o_op(2, 5, _a,_b);});uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(16, 4, _a,_b);});
{uint64_t _i=v4;uint64_t _v=o_cast(2, 4, ({uint64_t _a=v5;uint64_t _b=255ULL;o_op(12, 4, _a,_b);}));o_put(v1, v1_n, _i,_v,2);}
}
}
}
}
return o_leave(0);
}
static uint64_t f43(void *v0, uint64_t v0_n, void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, uint64_t v4){
o_enter();
uint64_t __attribute__((unused)) v5=({uint64_t _a=v4;uint64_t _b=64ULL;o_op(5, 5, _a,_b);});
{uint64_t _i=v5;uint64_t _v=128ULL;o_put(v3, v3_n, _i,_v,2);}
uint64_t __attribute__((unused)) v6=64ULL;
if(({uint64_t _a=v5;uint64_t _b=56ULL;o_op(11, 5, _a,_b);})){
v6=128ULL;
}
{
uint64_t v7=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v7, k1);v7=o_op(1,5, v7,1)){
uint64_t __attribute__((unused)) v8=({uint64_t _a=v6;uint64_t _b=({uint64_t _a=v7;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});o_op(2, 5, _a,_b);});
uint64_t __attribute__((unused)) v9=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=3ULL;o_op(15, 5, _a,_b);});uint64_t _b=({uint64_t _a=v7;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(16, 5, _a,_b);});
{uint64_t _i=v8;uint64_t _v=o_cast(2, 5, ({uint64_t _a=v9;uint64_t _b=255ULL;o_op(12, 5, _a,_b);}));o_put(v3, v3_n, _i,_v,2);}
}
}
({uint64_t p2=0ULL;f39(v1, v1_n, v3, v3_n, p2);});
({f41(v0, v0_n, v1, v1_n, v2, v2_n);});
if(({uint64_t _a=v6;uint64_t _b=128ULL;o_op(6, 5, _a,_b);})){
({uint64_t p2=64ULL;f39(v1, v1_n, v3, v3_n, p2);});
({f41(v0, v0_n, v1, v1_n, v2, v2_n);});
}
return o_leave(0);
}
static uint64_t f44(const void *v0, uint64_t v0_n, uint64_t v1, void *v2, uint64_t v2_n){
o_enter();
uint8_t v3[128]={0};
uint64_t v3_n=128;
o_fill(v3, v3_n, 0ULL, 2);
uint32_t v4[8]={0};
uint64_t v4_n=8;
o_fill(v4, v4_n, 0ULL, 4);
uint32_t v5[64]={0};
uint64_t v5_n=64;
o_fill(v5, v5_n, 0ULL, 4);
uint32_t v6[64]={0};
uint64_t v6_n=64;
o_fill(v6, v6_n, 0ULL, 4);
if(({uint64_t _a=v1;uint64_t _b=4096ULL;o_op(10, 5, _a,_b);})){
return o_leave(18446744073709551615ULL);
}
({f35(v6, v6_n);});
({f36(v4, v4_n);});
{
uint64_t v7=0;
uint64_t k1=({uint64_t _a=v1;uint64_t _b=64ULL;o_op(4, 5, _a,_b);});
uint64_t c1=0;
for(;o_op(8,5, v7, k1);v7=o_op(1,5, v7,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p2=({uint64_t _a=v7;uint64_t _b=64ULL;o_op(3, 5, _a,_b);});f37(v5, v5_n, v0, v0_n, p2);});
({f41(v4, v4_n, v5, v5_n, v6, v6_n);});
}
}
uint64_t __attribute__((unused)) v7=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=64ULL;o_op(4, 5, _a,_b);});uint64_t _b=64ULL;o_op(3, 5, _a,_b);});
{
uint64_t v8=0;
uint64_t k1=({uint64_t _a=v1;uint64_t _b=64ULL;o_op(5, 5, _a,_b);});
uint64_t c1=0;
for(;o_op(8,5, v8, k1);v8=o_op(1,5, v8,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=v8;uint64_t _v=o_get(v0, v0_n, ({uint64_t _a=v7;uint64_t _b=v8;o_op(1, 5, _a,_b);}), 2);o_put(v3, v3_n, _i,_v,2);}
}
}
({uint64_t p4=v1;f43(v4, v4_n, v5, v5_n, v6, v6_n, v3, v3_n, p4);});
({f42(v4, v4_n, v2, v2_n);});
return o_leave(0ULL);
}
static uint64_t f45(const void *v0, uint64_t v0_n, uint64_t v1, void *v2, uint64_t v2_n){
o_enter();
uint8_t v3[128]={0};
uint64_t v3_n=128;
o_fill(v3, v3_n, 0ULL, 2);
uint32_t v4[8]={0};
uint64_t v4_n=8;
o_fill(v4, v4_n, 0ULL, 4);
uint32_t v5[64]={0};
uint64_t v5_n=64;
o_fill(v5, v5_n, 0ULL, 4);
uint32_t v6[64]={0};
uint64_t v6_n=64;
o_fill(v6, v6_n, 0ULL, 4);
if(({uint64_t _a=v1;uint64_t _b=16384ULL;o_op(10, 5, _a,_b);})){
return o_leave(18446744073709551615ULL);
}
({f35(v6, v6_n);});
({f36(v4, v4_n);});
{
uint64_t v7=0;
uint64_t k1=({uint64_t _a=v1;uint64_t _b=64ULL;o_op(4, 5, _a,_b);});
uint64_t c1=0;
for(;o_op(8,5, v7, k1);v7=o_op(1,5, v7,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p2=({uint64_t _a=v7;uint64_t _b=64ULL;o_op(3, 5, _a,_b);});f38(v5, v5_n, v0, v0_n, p2);});
({f41(v4, v4_n, v5, v5_n, v6, v6_n);});
}
}
uint64_t __attribute__((unused)) v7=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=64ULL;o_op(4, 5, _a,_b);});uint64_t _b=64ULL;o_op(3, 5, _a,_b);});
{
uint64_t v8=0;
uint64_t k1=({uint64_t _a=v1;uint64_t _b=64ULL;o_op(5, 5, _a,_b);});
uint64_t c1=0;
for(;o_op(8,5, v8, k1);v8=o_op(1,5, v8,1)){
if(c1++>=16777216)o_trap(4);
{uint64_t _i=v8;uint64_t _v=o_get(v0, v0_n, ({uint64_t _a=v7;uint64_t _b=v8;o_op(1, 5, _a,_b);}), 2);o_put(v3, v3_n, _i,_v,2);}
}
}
({uint64_t p4=v1;f43(v4, v4_n, v5, v5_n, v6, v6_n, v3, v3_n, p4);});
({f42(v4, v4_n, v2, v2_n);});
return o_leave(0ULL);
}
static uint64_t f46(const void *v0, uint64_t v0_n, uint64_t v1, void *v2, uint64_t v2_n){
o_enter();
return o_leave(o_cast(9, 8, ({uint64_t p1=v1;f45(v0, v0_n, p1, v2, v2_n);})));
}
uint32_t omega_native_codekey(uint64_t *result, const void *p0, uint64_t p0_n, volatile uint64_t p1, void *p2, uint64_t p2_n){
if(!o_buf(result,1,1,5))return 32;
if(!o_buf(p0, p0_n,16384, 2))return 32;
if(o_overlap(result,8,p0, p0_n*1))return 32;
if(!o_valid(5, p1))return 32;
if(!o_buf(p2, p2_n,32, 2))return 32;
if(o_overlap(result,8,p2, p2_n*1))return 32;
if(o_overlap(p2, p2_n*1, p0, p0_n*1))return 32;
if(o_busy)return 33;
o_busy=1;o_depth=0;int code=setjmp(o_jmp);
if(!code)*result=f46(p0, p0_n, p1, p2, p2_n);
o_busy=0;o_depth=0;return (uint32_t)code;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
