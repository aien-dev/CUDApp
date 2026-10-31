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
static uint64_t f21(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, void *v4, uint64_t v4_n);
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
static uint64_t f21(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, void *v3, uint64_t v3_n, void *v4, uint64_t v4_n){
o_enter();
return o_leave(o_cast(5, 4, ({f20(v0, v0_n, v1, v1_n, v2, v2_n, v3, v3_n, v4, v4_n);})));
}
uint32_t omega_native_repack(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, void *p3, uint64_t p3_n, void *p4, uint64_t p4_n){
if(!o_buf(result,1,1,5))return 32;
if(!o_buf(p0, p0_n,48, 5))return 32;
if(o_overlap(result,8,p0, p0_n*8))return 32;
if(!o_buf(p1, p1_n,65536, 2))return 32;
if(o_overlap(result,8,p1, p1_n*1))return 32;
if(!o_buf(p2, p2_n,1024, 2))return 32;
if(o_overlap(result,8,p2, p2_n*1))return 32;
if(!o_buf(p3, p3_n,65536, 2))return 32;
if(o_overlap(result,8,p3, p3_n*1))return 32;
if(o_overlap(p3, p3_n*1, p0, p0_n*8))return 32;
if(o_overlap(p3, p3_n*1, p1, p1_n*1))return 32;
if(o_overlap(p3, p3_n*1, p2, p2_n*1))return 32;
if(!o_buf(p4, p4_n,1024, 2))return 32;
if(o_overlap(result,8,p4, p4_n*1))return 32;
if(o_overlap(p4, p4_n*1, p0, p0_n*8))return 32;
if(o_overlap(p4, p4_n*1, p1, p1_n*1))return 32;
if(o_overlap(p4, p4_n*1, p2, p2_n*1))return 32;
if(o_overlap(p4, p4_n*1, p3, p3_n*1))return 32;
if(o_busy)return 33;
o_busy=1;o_depth=0;int code=setjmp(o_jmp);
if(!code)*result=f21(p0, p0_n, p1, p1_n, p2, p2_n, p3, p3_n, p4, p4_n);
o_busy=0;o_depth=0;return (uint32_t)code;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
