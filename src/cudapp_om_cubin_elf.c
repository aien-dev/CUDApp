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
static uint64_t f0(const void *v0, uint64_t v0_n);
static uint64_t f1(const void *v0, uint64_t v0_n);
static uint64_t f2(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f3(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f4(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f5(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f6(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3);
static uint64_t f7(void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n, uint64_t v3);
static uint64_t f8(uint64_t v0, uint64_t v1);
static uint64_t f9(uint64_t v0);
static uint64_t f10(uint64_t v0);
static uint64_t f11(const void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f12(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f13(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3);
static uint64_t f14(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f15(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, void *v2, uint64_t v2_n);
static uint64_t f16(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f17(void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f18(const void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f19(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3);
static uint64_t f20(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f21(void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n, uint64_t v3);
static uint64_t f22(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f23(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f24(void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f25(void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n);
static uint64_t f26(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4);
static uint64_t f27(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f28(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4);
static uint64_t f29(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5);
static uint64_t f30(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3);
static uint64_t f31(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f32(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f33(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4);
static uint64_t f34(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3);
static uint64_t f35(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f36(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n);
static uint64_t f37(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, const void *v3, uint64_t v3_n, const void *v4, uint64_t v4_n);
static uint64_t f38(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, void *v3, uint64_t v3_n);
static uint64_t f0(const void *v0, uint64_t v0_n){
o_enter();
if(({uint64_t _a=o_get(v0, v0_n, 6ULL, 5);uint64_t _b=86ULL;o_op(6, 5, _a,_b);})){
return o_leave(352ULL);
}
return o_leave(896ULL);
}
static uint64_t f1(const void *v0, uint64_t v0_n){
o_enter();
if(({uint64_t _a=o_get(v0, v0_n, 6ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(121ULL);
}
return o_leave(o_get(v0, v0_n, 6ULL, 5));
}
static uint64_t f2(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
uint64_t __attribute__((unused)) v3=({uint64_t _a=v1;uint64_t _b=3ULL;o_op(16, 5, _a,_b);});
uint64_t __attribute__((unused)) v4=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=7ULL;o_op(12, 5, _a,_b);});uint64_t _b=3ULL;o_op(15, 5, _a,_b);});
uint64_t __attribute__((unused)) v5=({uint64_t _a=o_get(v0, v0_n, v3, 5);uint64_t _b=({uint64_t _a=({uint64_t _a=255ULL;uint64_t _b=v4;o_op(15, 5, _a,_b);});uint64_t _b=0;o_op(18, 5, _a,_b);});o_op(12, 5, _a,_b);});
{uint64_t _i=v3;uint64_t _v=({uint64_t _a=v5;uint64_t _b=({uint64_t _a=({uint64_t _a=v2;uint64_t _b=255ULL;o_op(12, 5, _a,_b);});uint64_t _b=v4;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});o_put(v0, v0_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f3(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
({uint64_t p1=v1;uint64_t p2=v2;f2(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t p2=({uint64_t _a=v2;uint64_t _b=8ULL;o_op(16, 5, _a,_b);});f2(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f4(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
({uint64_t p1=v1;uint64_t p2=v2;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=2ULL;o_op(1, 5, _a,_b);});uint64_t p2=({uint64_t _a=v2;uint64_t _b=16ULL;o_op(16, 5, _a,_b);});f3(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f5(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
({uint64_t p1=v1;uint64_t p2=v2;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(1, 5, _a,_b);});uint64_t p2=({uint64_t _a=v2;uint64_t _b=32ULL;o_op(16, 5, _a,_b);});f4(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f6(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3){
o_enter();
{
uint64_t v4=0;
uint64_t c1=0;
for(;o_op(8,5, v4, v3);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=v4;o_op(1, 5, _a,_b);});uint64_t p2=({uint64_t _a=v2;uint64_t _b=({uint64_t _a=v4;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(16, 5, _a,_b);});f2(v0, v0_n, p1, p2);});
}
}
return o_leave(({uint64_t _a=v1;uint64_t _b=v3;o_op(1, 5, _a,_b);}));
}
static uint64_t f7(void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n, uint64_t v3){
o_enter();
{
uint64_t v4=0;
uint64_t c1=0;
for(;o_op(8,5, v4, v3);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=v4;o_op(1, 5, _a,_b);});uint64_t p2=o_cast(5, 2, o_get(v2, v2_n, v4, 2));f2(v0, v0_n, p1, p2);});
}
}
return o_leave(({uint64_t _a=v1;uint64_t _b=v3;o_op(1, 5, _a,_b);}));
}
static uint64_t f8(uint64_t v0, uint64_t v1){
o_enter();
return o_leave(({uint64_t _a=({uint64_t _a=({uint64_t _a=v0;uint64_t _b=({uint64_t _a=v1;uint64_t _b=1ULL;o_op(2, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t _b=v1;o_op(4, 5, _a,_b);});uint64_t _b=v1;o_op(3, 5, _a,_b);}));
}
static uint64_t f9(uint64_t v0){
o_enter();
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=({uint64_t _a=v0;uint64_t _b=65ULL;o_op(11, 2, _a,_b);});uint64_t _b=({uint64_t _a=v0;uint64_t _b=90ULL;o_op(9, 2, _a,_b);});o_op(12, 1, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v0;uint64_t _b=97ULL;o_op(11, 2, _a,_b);});uint64_t _b=({uint64_t _a=v0;uint64_t _b=122ULL;o_op(9, 2, _a,_b);});o_op(12, 1, _a,_b);});o_op(13, 1, _a,_b);});uint64_t _b=({uint64_t _a=v0;uint64_t _b=95ULL;o_op(6, 2, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
return o_leave(0ULL);
}
static uint64_t f10(uint64_t v0){
o_enter();
if(({uint64_t _a=({uint64_t _a=({uint64_t p0=v0;f9(p0);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v0;uint64_t _b=48ULL;o_op(11, 2, _a,_b);});uint64_t _b=({uint64_t _a=v0;uint64_t _b=57ULL;o_op(9, 2, _a,_b);});o_op(12, 1, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
return o_leave(0ULL);
}
static uint64_t f11(const void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=64ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t p0=o_get(v0, v0_n, 0ULL, 2);f9(p0);});uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(0ULL);
}
{
uint64_t v2=0;
uint64_t c1=0;
for(;o_op(8,5, v2, v1);v2=o_op(1,5, v2,1)){
if(c1++>=16777216)o_trap(4);
if(({uint64_t _a=({uint64_t p0=o_get(v0, v0_n, v2, 2);f10(p0);});uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(0ULL);
}
}
}
return o_leave(1ULL);
}
static uint64_t f12(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
{uint64_t _i=0ULL;uint64_t _v=v1;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=v2;o_put(v0, v0_n, _i,_v,5);}
return o_leave(3ULL);
}
static uint64_t f13(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3){
o_enter();
if(({uint64_t _a=v0;uint64_t _b=v3;o_op(8, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=2ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=1ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=({uint64_t _a=v2;uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(3ULL);
}
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v0;uint64_t _b=7ULL;o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(4ULL);
}
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v0;uint64_t _b=3ULL;o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(4ULL);
}
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=2ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v0;uint64_t _b=1ULL;o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(4ULL);
}
return o_leave(0ULL);
}
static uint64_t f14(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
uint64_t __attribute__((unused)) v2=({f0(v0, v0_n);});
uint64_t __attribute__((unused)) v3=o_get(v0, v0_n, 2ULL, 5);
{
uint64_t v4=0;
uint64_t c1=0;
for(;o_op(8,5, v4, v3);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v5=({uint64_t _a=8ULL;uint64_t _b=({uint64_t _a=v4;uint64_t _b=3ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
uint64_t __attribute__((unused)) v6=({uint64_t p0=o_get(v0, v0_n, v5, 5);uint64_t p1=o_get(v0, v0_n, ({uint64_t _a=v5;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p2=o_get(v0, v0_n, ({uint64_t _a=v5;uint64_t _b=2ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p3=v2;f13(p0, p1, p2, p3);});
if(({uint64_t _a=v6;uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(({uint64_t p1=v6;uint64_t p2=v4;f12(v1, v1_n, p1, p2);}));
}
v2=({uint64_t _a=o_get(v0, v0_n, v5, 5);uint64_t _b=o_get(v0, v0_n, ({uint64_t _a=v5;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);o_op(1, 5, _a,_b);});
}
}
uint64_t __attribute__((unused)) v4=o_get(v0, v0_n, 3ULL, 5);
if(({uint64_t _a=v2;uint64_t _b=v4;o_op(10, 5, _a,_b);})){
{uint64_t _i=0ULL;uint64_t _v=5ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=v2;o_put(v1, v1_n, _i,_v,5);}
return o_leave(4ULL);
}
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=1024ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=({f0(v0, v0_n);});o_op(8, 5, _a,_b);});o_op(13, 1, _a,_b);})){
{uint64_t _i=0ULL;uint64_t _v=6ULL;o_put(v1, v1_n, _i,_v,5);}
return o_leave(4ULL);
}
return o_leave(0ULL);
}
static uint64_t f15(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, void *v2, uint64_t v2_n){
o_enter();
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=({f1(v0, v0_n);});uint64_t _b=86ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({f1(v0, v0_n);});uint64_t _b=121ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t p1=o_get(v0, v0_n, 0ULL, 5);f11(v1, v1_n, p1);});uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(1ULL);
}
uint64_t __attribute__((unused)) v3=o_get(v0, v0_n, 1ULL, 5);
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=15ULL;o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);});uint64_t _b=({uint64_t _a=v3;uint64_t _b=16777216ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(2ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=32ULL;o_op(10, 5, _a,_b);})){
return o_leave(({uint64_t p1=0ULL;uint64_t p2=0ULL;f12(v2, v2_n, p1, p2);}));
}
if(({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=255ULL;o_op(10, 5, _a,_b);})){
return o_leave(5ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 5ULL, 5);uint64_t _b=3ULL;o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(6ULL);
}
return o_leave(({f14(v0, v0_n, v2, v2_n);}));
}
static uint64_t f16(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 0ULL, 5);
uint64_t __attribute__((unused)) v3=64ULL;
{uint64_t _i=1ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
v3=({uint64_t _a=v3;uint64_t _b=({uint64_t _a=117ULL;uint64_t _b=({uint64_t _a=3ULL;uint64_t _b=v2;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});o_op(1, 5, _a,_b);});
{uint64_t _i=2ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
v3=({uint64_t _a=v3;uint64_t _b=({uint64_t _a=v2;uint64_t _b=2ULL;o_op(1, 5, _a,_b);});o_op(1, 5, _a,_b);});
v3=({uint64_t p0=v3;uint64_t p1=8ULL;f8(p0, p1);});
{uint64_t _i=3ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
v3=({uint64_t _a=v3;uint64_t _b=96ULL;o_op(1, 5, _a,_b);});
v3=({uint64_t p0=v3;uint64_t p1=128ULL;f8(p0, p1);});
{uint64_t _i=4ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
v3=({uint64_t _a=v3;uint64_t _b=o_get(v0, v0_n, 1ULL, 5);o_op(1, 5, _a,_b);});
v3=({uint64_t p0=v3;uint64_t p1=16ULL;f8(p0, p1);});
{uint64_t _i=5ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
v3=({uint64_t _a=v3;uint64_t _b=o_get(v0, v0_n, 3ULL, 5);o_op(1, 5, _a,_b);});
v3=({uint64_t p0=v3;uint64_t p1=4ULL;f8(p0, p1);});
{uint64_t _i=6ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
v3=({uint64_t _a=v3;uint64_t _b=36ULL;o_op(1, 5, _a,_b);});
v3=({uint64_t p0=v3;uint64_t p1=4ULL;f8(p0, p1);});
{uint64_t _i=7ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
v3=({uint64_t _a=v3;uint64_t _b=36ULL;o_op(1, 5, _a,_b);});
uint64_t __attribute__((unused)) v4=({uint64_t _a=48ULL;uint64_t _b=({uint64_t _a=16ULL;uint64_t _b=o_get(v0, v0_n, 2ULL, 5);o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
{uint64_t _i=14ULL;uint64_t _v=v4;o_put(v1, v1_n, _i,_v,5);}
v3=({uint64_t p0=v3;uint64_t p1=4ULL;f8(p0, p1);});
{uint64_t _i=8ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
v3=({uint64_t _a=v3;uint64_t _b=v4;o_op(1, 5, _a,_b);});
{uint64_t _i=9ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
v3=({uint64_t _a=v3;uint64_t _b=o_get(v0, v0_n, 5ULL, 5);o_op(1, 5, _a,_b);});
v3=({uint64_t p0=v3;uint64_t p1=4ULL;f8(p0, p1);});
{uint64_t _i=10ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
v3=({uint64_t _a=v3;uint64_t _b=32ULL;o_op(1, 5, _a,_b);});
v3=({uint64_t p0=v3;uint64_t p1=8ULL;f8(p0, p1);});
{uint64_t _i=11ULL;uint64_t _v=v3;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=12ULL;uint64_t _v=({uint64_t _a=v3;uint64_t _b=704ULL;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=13ULL;uint64_t _v=({uint64_t _a=v3;uint64_t _b=984ULL;o_op(1, 5, _a,_b);});o_put(v1, v1_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f17(void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
uint64_t __attribute__((unused)) v2=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=7ULL;o_op(1, 5, _a,_b);});uint64_t _b=3ULL;o_op(16, 5, _a,_b);});
uint64_t __attribute__((unused)) v3=({uint64_t _a=({uint64_t _a=v2;uint64_t _b=4095ULL;o_op(1, 5, _a,_b);});uint64_t _b=12ULL;o_op(16, 5, _a,_b);});
{
uint64_t v4=0;
uint64_t c1=0;
for(;o_op(8,5, v4, v3);v4=o_op(1,5, v4,1)){
if(c1++>=16777216)o_trap(4);
{
uint64_t v5=0;
uint64_t k2=4096ULL;
for(;o_op(8,5, v5, k2);v5=o_op(1,5, v5,1)){
uint64_t __attribute__((unused)) v6=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=12ULL;o_op(15, 5, _a,_b);});uint64_t _b=v5;o_op(1, 5, _a,_b);});
if(({uint64_t _a=v6;uint64_t _b=v2;o_op(8, 5, _a,_b);})){
{uint64_t _i=v6;uint64_t _v=0ULL;o_put(v0, v0_n, _i,_v,5);}
}
}
}
}
}
return o_leave(0);
}
static uint64_t f18(const void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
uint64_t __attribute__((unused)) v2=0ULL;
{
uint64_t v3=0;
uint64_t k1=8ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
v2=({uint64_t _a=v2;uint64_t _b=({uint64_t _a=o_cast(5, 2, o_get(v0, v0_n, ({uint64_t _a=v1;uint64_t _b=v3;o_op(1, 5, _a,_b);}), 2));uint64_t _b=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
}
}
return o_leave(v2);
}
static uint64_t f19(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3){
o_enter();
uint64_t __attribute__((unused)) v4=({uint64_t _a=v3;uint64_t _b=3ULL;o_op(16, 5, _a,_b);});
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=4095ULL;o_op(1, 5, _a,_b);});uint64_t _b=12ULL;o_op(16, 5, _a,_b);});
uint64_t __attribute__((unused)) v6=({uint64_t _a=v2;uint64_t _b=3ULL;o_op(16, 5, _a,_b);});
{
uint64_t v7=0;
uint64_t c1=0;
for(;o_op(8,5, v7, v5);v7=o_op(1,5, v7,1)){
if(c1++>=16777216)o_trap(4);
{
uint64_t v8=0;
uint64_t k2=4096ULL;
for(;o_op(8,5, v8, k2);v8=o_op(1,5, v8,1)){
uint64_t __attribute__((unused)) v9=({uint64_t _a=({uint64_t _a=v7;uint64_t _b=12ULL;o_op(15, 5, _a,_b);});uint64_t _b=v8;o_op(1, 5, _a,_b);});
if(({uint64_t _a=v9;uint64_t _b=v4;o_op(8, 5, _a,_b);})){
{uint64_t _i=({uint64_t _a=v6;uint64_t _b=v9;o_op(1, 5, _a,_b);});uint64_t _v=({uint64_t p1=({uint64_t _a=v9;uint64_t _b=3ULL;o_op(15, 5, _a,_b);});f18(v1, v1_n, p1);});o_put(v0, v0_n, _i,_v,5);}
}
}
}
}
}
return o_leave(0);
}
static uint64_t f20(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
({uint64_t p1=0ULL;uint64_t p2=4684026196722992511ULL;f5(v0, v0_n, p1, p2);});
if(({uint64_t _a=({f1(v2, v2_n);});uint64_t _b=86ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=7ULL;uint64_t p2=51ULL;f2(v0, v0_n, p1, p2);});
}
if(({uint64_t _a=({f1(v2, v2_n);});uint64_t _b=86ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=8ULL;uint64_t p2=7ULL;f2(v0, v0_n, p1, p2);});
}else{
({uint64_t p1=8ULL;uint64_t p2=8ULL;f2(v0, v0_n, p1, p2);});
}
({uint64_t p1=16ULL;uint64_t p2=2ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=18ULL;uint64_t p2=190ULL;f3(v0, v0_n, p1, p2);});
if(({uint64_t _a=({f1(v2, v2_n);});uint64_t _b=86ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=20ULL;uint64_t p2=130ULL;f4(v0, v0_n, p1, p2);});
}else{
({uint64_t p1=20ULL;uint64_t p2=1ULL;f4(v0, v0_n, p1, p2);});
}
({uint64_t p1=32ULL;uint64_t p2=o_get(v1, v1_n, 12ULL, 5);f5(v0, v0_n, p1, p2);});
({uint64_t p1=40ULL;uint64_t p2=o_get(v1, v1_n, 11ULL, 5);f5(v0, v0_n, p1, p2);});
if(({uint64_t _a=({f1(v2, v2_n);});uint64_t _b=86ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=48ULL;uint64_t p2=({uint64_t _a=({uint64_t _a=86ULL;uint64_t _b=16ULL;o_op(15, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=5ULL;uint64_t _b=8ULL;o_op(15, 5, _a,_b);});uint64_t _b=86ULL;o_op(13, 5, _a,_b);});o_op(13, 5, _a,_b);});f4(v0, v0_n, p1, p2);});
}else{
({uint64_t p1=48ULL;uint64_t p2=100694274ULL;f4(v0, v0_n, p1, p2);});
}
({uint64_t p1=52ULL;uint64_t p2=64ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=54ULL;uint64_t p2=56ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=56ULL;uint64_t p2=5ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=58ULL;uint64_t p2=64ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=60ULL;uint64_t p2=11ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=62ULL;uint64_t p2=1ULL;f3(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f21(void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n, uint64_t v3){
o_enter();
uint64_t __attribute__((unused)) v4=({uint64_t _a=v1;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
v4=({uint64_t p1=v4;uint64_t p2=7022363563455443758ULL;uint64_t p3=8ULL;f6(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=({uint64_t p1=v4;uint64_t p2=98ULL;uint64_t p3=1ULL;f6(v0, v0_n, p1, p2, p3);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
v4=({uint64_t _a=({uint64_t p1=v4;uint64_t p2=27691700481979182ULL;uint64_t p3=7ULL;f6(v0, v0_n, p1, p2, p3);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
v4=({uint64_t _a=({uint64_t p1=v4;uint64_t p2=27691700398420782ULL;uint64_t p3=7ULL;f6(v0, v0_n, p1, p2, p3);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
v4=({uint64_t p1=v4;uint64_t p2=51077770998830ULL;uint64_t p3=6ULL;f6(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=({uint64_t p1=v4;uint64_t p3=v3;f7(v0, v0_n, p1, v2, v2_n, p3);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
v4=({uint64_t p1=v4;uint64_t p2=8317708033571843630ULL;uint64_t p3=8ULL;f6(v0, v0_n, p1, p2, p3);});
v4=({uint64_t p1=v4;uint64_t p2=50785646698868ULL;uint64_t p3=6ULL;f6(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=({uint64_t p1=v4;uint64_t p3=v3;f7(v0, v0_n, p1, v2, v2_n, p3);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
v4=({uint64_t _a=({uint64_t p1=v4;uint64_t p2=8027224783864622638ULL;uint64_t p3=8ULL;f6(v0, v0_n, p1, p2, p3);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
v4=({uint64_t p1=v4;uint64_t p2=8101253776481349166ULL;uint64_t p3=8ULL;f6(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=({uint64_t p1=v4;uint64_t p2=29793ULL;uint64_t p3=2ULL;f6(v0, v0_n, p1, p2, p3);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
v4=({uint64_t p1=v4;uint64_t p2=8027224783864622638ULL;uint64_t p3=8ULL;f6(v0, v0_n, p1, p2, p3);});
v4=({uint64_t p1=v4;uint64_t p2=46ULL;uint64_t p3=1ULL;f6(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=({uint64_t p1=v4;uint64_t p3=v3;f7(v0, v0_n, p1, v2, v2_n, p3);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
v4=({uint64_t p1=v4;uint64_t p2=8241983636974759470ULL;uint64_t p3=8ULL;f6(v0, v0_n, p1, p2, p3);});
v4=({uint64_t p1=v4;uint64_t p2=8243121572516095077ULL;uint64_t p3=8ULL;f6(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=({uint64_t p1=v4;uint64_t p2=206936761718ULL;uint64_t p3=5ULL;f6(v0, v0_n, p1, p2, p3);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
v4=({uint64_t p1=v4;uint64_t p2=8533809357193702958ULL;uint64_t p3=8ULL;f6(v0, v0_n, p1, p2, p3);});
v4=({uint64_t p1=v4;uint64_t p2=31356346816619310ULL;uint64_t p3=7ULL;f6(v0, v0_n, p1, p2, p3);});
return o_leave(0);
}
static uint64_t f22(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=28ULL;o_op(1, 5, _a,_b);});uint64_t p2=3ULL;f2(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=30ULL;o_op(1, 5, _a,_b);});uint64_t p2=4ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=48ULL;o_op(1, 5, _a,_b);});uint64_t p2=1ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=52ULL;o_op(1, 5, _a,_b);});uint64_t p2=18ULL;f2(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=53ULL;o_op(1, 5, _a,_b);});uint64_t p2=16ULL;f2(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=54ULL;o_op(1, 5, _a,_b);});uint64_t p2=4ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=64ULL;o_op(1, 5, _a,_b);});uint64_t p2=v2;f5(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=76ULL;o_op(1, 5, _a,_b);});uint64_t p2=3ULL;f2(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=78ULL;o_op(1, 5, _a,_b);});uint64_t p2=5ULL;f3(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f23(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
({uint64_t p1=v1;uint64_t p2=12036ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=2ULL;o_op(1, 5, _a,_b);});uint64_t p2=8ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(1, 5, _a,_b);});uint64_t p2=2ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(1, 5, _a,_b);});uint64_t p2=v2;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=12ULL;o_op(1, 5, _a,_b);});uint64_t p2=4356ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=14ULL;o_op(1, 5, _a,_b);});uint64_t p2=8ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=16ULL;o_op(1, 5, _a,_b);});uint64_t p2=2ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=24ULL;o_op(1, 5, _a,_b);});uint64_t p2=4612ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=26ULL;o_op(1, 5, _a,_b);});uint64_t p2=8ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=28ULL;o_op(1, 5, _a,_b);});uint64_t p2=2ULL;f4(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f24(void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
({uint64_t p1=v1;uint64_t p2=2306ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(1, 5, _a,_b);});uint64_t p2=131586ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(1, 5, _a,_b);});uint64_t p2=328962ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=12ULL;o_op(1, 5, _a,_b);});uint64_t p2=16844547ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=16ULL;o_op(1, 5, _a,_b);});uint64_t p2=770ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=20ULL;o_op(1, 5, _a,_b);});uint64_t p2=67074ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=24ULL;o_op(1, 5, _a,_b);});uint64_t p2=527108ULL;f4(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f25(void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n){
o_enter();
({uint64_t p1=v1;uint64_t p2=14084ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=2ULL;o_op(1, 5, _a,_b);});uint64_t p2=4ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(1, 5, _a,_b);});uint64_t p2=130ULL;f4(v0, v0_n, p1, p2);});
uint64_t __attribute__((unused)) v3=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(1, 5, _a,_b);});
uint64_t __attribute__((unused)) v4=({f0(v2, v2_n);});
uint64_t __attribute__((unused)) v5=o_get(v2, v2_n, 2ULL, 5);
{
uint64_t v6=0;
uint64_t c1=0;
for(;o_op(8,5, v6, v5);v6=o_op(1,5, v6,1)){
if(c1++>=16777216)o_trap(4);
uint64_t __attribute__((unused)) v7=({uint64_t _a=8ULL;uint64_t _b=({uint64_t _a=v6;uint64_t _b=3ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
uint64_t __attribute__((unused)) v8=o_get(v2, v2_n, v7, 5);
({uint64_t p1=v3;uint64_t p2=5892ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(1, 5, _a,_b);});uint64_t p2=12ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(1, 5, _a,_b);});uint64_t p2=({uint64_t _a=v6;uint64_t _b=({uint64_t _a=({uint64_t _a=v8;uint64_t _b=({f0(v2, v2_n);});o_op(2, 5, _a,_b);});uint64_t _b=16ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});f4(v0, v0_n, p1, p2);});
uint64_t __attribute__((unused)) v9=1175552ULL;
if(({uint64_t _a=o_get(v2, v2_n, ({uint64_t _a=v7;uint64_t _b=2ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
v9=2225408ULL;
}
({uint64_t p1=({uint64_t _a=v3;uint64_t _b=12ULL;o_op(1, 5, _a,_b);});uint64_t p2=v9;f4(v0, v0_n, p1, p2);});
v3=({uint64_t _a=v3;uint64_t _b=16ULL;o_op(1, 5, _a,_b);});
if(({uint64_t _a=({uint64_t _a=v8;uint64_t _b=o_get(v2, v2_n, ({uint64_t _a=v7;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);o_op(1, 5, _a,_b);});uint64_t _b=v4;o_op(10, 5, _a,_b);})){
v4=({uint64_t _a=v8;uint64_t _b=o_get(v2, v2_n, ({uint64_t _a=v7;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);o_op(1, 5, _a,_b);});
}
}
}
return o_leave(v4);
}
static uint64_t f26(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4){
o_enter();
({uint64_t p1=v1;uint64_t p2=6403ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=2ULL;o_op(1, 5, _a,_b);});uint64_t p2=({uint64_t _a=v2;uint64_t _b=v4;o_op(2, 5, _a,_b);});f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(1, 5, _a,_b);});uint64_t p2=2564ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=6ULL;o_op(1, 5, _a,_b);});uint64_t p2=8ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(1, 5, _a,_b);});uint64_t p2=3ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=12ULL;o_op(1, 5, _a,_b);});uint64_t p2=({uint64_t _a=v4;uint64_t _b=({uint64_t _a=({uint64_t _a=v2;uint64_t _b=v4;o_op(2, 5, _a,_b);});uint64_t _b=16ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=16ULL;o_op(1, 5, _a,_b);});uint64_t p2=20483ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=20ULL;o_op(1, 5, _a,_b);});uint64_t p2=6915ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=22ULL;o_op(1, 5, _a,_b);});uint64_t p2=v3;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=24ULL;o_op(1, 5, _a,_b);});uint64_t p2=24323ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=26ULL;o_op(1, 5, _a,_b);});uint64_t p2=257ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=28ULL;o_op(1, 5, _a,_b);});uint64_t p2=18946ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=32ULL;o_op(1, 5, _a,_b);});uint64_t p2=13828ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=34ULL;o_op(1, 5, _a,_b);});uint64_t p2=4ULL;f3(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f27(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
({uint64_t p1=v1;uint64_t p2=12ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(1, 5, _a,_b);});uint64_t p2=8ULL;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(1, 5, _a,_b);});uint64_t p2=1000ULL;f4(v0, v0_n, p1, p2);});
uint64_t __attribute__((unused)) v3=({uint64_t p1=({uint64_t _a=v1;uint64_t _b=12ULL;o_op(1, 5, _a,_b);});uint64_t p2=4836937782729987662ULL;uint64_t p3=8ULL;f6(v0, v0_n, p1, p2, p3);});
v3=({uint64_t p1=v3;uint64_t p2=7369327ULL;uint64_t p3=3ULL;f6(v0, v0_n, p1, p2, p3);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=24ULL;o_op(1, 5, _a,_b);});uint64_t p2=2ULL;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=26ULL;o_op(1, 5, _a,_b);});uint64_t p2=v2;f3(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=28ULL;o_op(1, 5, _a,_b);});uint64_t p2=130ULL;f4(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f28(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4){
o_enter();
({uint64_t p1=v1;uint64_t p2=v2;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(1, 5, _a,_b);});uint64_t p2=v3;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(1, 5, _a,_b);});uint64_t p2=v4;f5(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f29(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4, uint64_t v5){
o_enter();
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=24ULL;o_op(1, 5, _a,_b);});uint64_t p2=v2;f5(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=32ULL;o_op(1, 5, _a,_b);});uint64_t p2=v3;f5(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=40ULL;o_op(1, 5, _a,_b);});uint64_t p2=v4;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=44ULL;o_op(1, 5, _a,_b);});uint64_t p2=v5;f4(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f30(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3){
o_enter();
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=48ULL;o_op(1, 5, _a,_b);});uint64_t p2=v2;f5(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=56ULL;o_op(1, 5, _a,_b);});uint64_t p2=v3;f5(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f31(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=o_get(v2, v2_n, 0ULL, 5);
uint64_t __attribute__((unused)) v4=({uint64_t _a=o_get(v1, v1_n, 11ULL, 5);uint64_t _b=64ULL;o_op(1, 5, _a,_b);});
({uint64_t p1=v4;uint64_t p2=1ULL;uint64_t p3=3ULL;uint64_t p4=0ULL;f28(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=v4;uint64_t p2=o_get(v1, v1_n, 1ULL, 5);uint64_t p3=({uint64_t _a=117ULL;uint64_t _b=({uint64_t _a=3ULL;uint64_t _b=v3;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t p4=0ULL;uint64_t p5=0ULL;f29(v0, v0_n, p1, p2, p3, p4, p5);});
({uint64_t p1=v4;uint64_t p2=1ULL;uint64_t p3=0ULL;f30(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=v4;uint64_t _b=64ULL;o_op(1, 5, _a,_b);});
({uint64_t p1=v4;uint64_t p2=11ULL;uint64_t p3=3ULL;uint64_t p4=0ULL;f28(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=v4;uint64_t p2=o_get(v1, v1_n, 2ULL, 5);uint64_t p3=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(1, 5, _a,_b);});uint64_t p4=0ULL;uint64_t p5=0ULL;f29(v0, v0_n, p1, p2, p3, p4, p5);});
({uint64_t p1=v4;uint64_t p2=1ULL;uint64_t p3=0ULL;f30(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=v4;uint64_t _b=64ULL;o_op(1, 5, _a,_b);});
({uint64_t p1=v4;uint64_t p2=19ULL;uint64_t p3=2ULL;uint64_t p4=0ULL;f28(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=v4;uint64_t p2=o_get(v1, v1_n, 3ULL, 5);uint64_t p3=96ULL;uint64_t p4=2ULL;uint64_t p5=2ULL;f29(v0, v0_n, p1, p2, p3, p4, p5);});
({uint64_t p1=v4;uint64_t p2=8ULL;uint64_t p3=24ULL;f30(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=v4;uint64_t _b=64ULL;o_op(1, 5, _a,_b);});
({uint64_t p1=v4;uint64_t p2=27ULL;uint64_t p3=1ULL;uint64_t p4=6ULL;f28(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=v4;uint64_t p2=o_get(v1, v1_n, 4ULL, 5);uint64_t p3=o_get(v2, v2_n, 1ULL, 5);uint64_t p4=3ULL;uint64_t p5=2ULL;f29(v0, v0_n, p1, p2, p3, p4, p5);});
({uint64_t p1=v4;uint64_t p2=128ULL;uint64_t p3=0ULL;f30(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=v4;uint64_t _b=64ULL;o_op(1, 5, _a,_b);});
({uint64_t p1=v4;uint64_t p2=({uint64_t _a=34ULL;uint64_t _b=v3;o_op(1, 5, _a,_b);});uint64_t p3=1ULL;uint64_t p4=66ULL;f28(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=v4;uint64_t p2=o_get(v1, v1_n, 5ULL, 5);uint64_t p3=o_get(v2, v2_n, 3ULL, 5);uint64_t p4=3ULL;uint64_t p5=4ULL;f29(v0, v0_n, p1, p2, p3, p4, p5);});
({uint64_t p1=v4;uint64_t p2=16ULL;uint64_t p3=0ULL;f30(v0, v0_n, p1, p2, p3);});
return o_leave(0);
}
static uint64_t f32(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=o_get(v2, v2_n, 0ULL, 5);
uint64_t __attribute__((unused)) v4=({uint64_t _a=o_get(v1, v1_n, 11ULL, 5);uint64_t _b=384ULL;o_op(1, 5, _a,_b);});
({uint64_t p1=v4;uint64_t p2=({uint64_t _a=49ULL;uint64_t _b=({uint64_t _a=2ULL;uint64_t _b=v3;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t p3=1879048192ULL;uint64_t p4=0ULL;f28(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=v4;uint64_t p2=o_get(v1, v1_n, 6ULL, 5);uint64_t p3=36ULL;uint64_t p4=3ULL;uint64_t p5=0ULL;f29(v0, v0_n, p1, p2, p3, p4, p5);});
({uint64_t p1=v4;uint64_t p2=4ULL;uint64_t p3=0ULL;f30(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=v4;uint64_t _b=64ULL;o_op(1, 5, _a,_b);});
({uint64_t p1=v4;uint64_t p2=({uint64_t _a=58ULL;uint64_t _b=({uint64_t _a=2ULL;uint64_t _b=v3;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t p3=1879048326ULL;uint64_t p4=0ULL;f28(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=v4;uint64_t p2=o_get(v1, v1_n, 7ULL, 5);uint64_t p3=36ULL;uint64_t p4=0ULL;uint64_t p5=0ULL;f29(v0, v0_n, p1, p2, p3, p4, p5);});
({uint64_t p1=v4;uint64_t p2=4ULL;uint64_t p3=0ULL;f30(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=v4;uint64_t _b=64ULL;o_op(1, 5, _a,_b);});
({uint64_t p1=v4;uint64_t p2=({uint64_t _a=69ULL;uint64_t _b=({uint64_t _a=2ULL;uint64_t _b=v3;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t p3=1879048192ULL;uint64_t p4=64ULL;f28(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=v4;uint64_t p2=o_get(v1, v1_n, 8ULL, 5);uint64_t p3=o_get(v1, v1_n, 14ULL, 5);uint64_t p4=3ULL;uint64_t p5=4ULL;f29(v0, v0_n, p1, p2, p3, p4, p5);});
({uint64_t p1=v4;uint64_t p2=4ULL;uint64_t p3=0ULL;f30(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=v4;uint64_t _b=64ULL;o_op(1, 5, _a,_b);});
({uint64_t p1=v4;uint64_t p2=({uint64_t _a=79ULL;uint64_t _b=({uint64_t _a=3ULL;uint64_t _b=v3;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t p3=8ULL;uint64_t p4=3ULL;f28(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=v4;uint64_t p2=o_get(v1, v1_n, 9ULL, 5);uint64_t p3=o_get(v2, v2_n, 5ULL, 5);uint64_t p4=0ULL;uint64_t p5=0ULL;f29(v0, v0_n, p1, p2, p3, p4, p5);});
({uint64_t p1=v4;uint64_t p2=1ULL;uint64_t p3=0ULL;f30(v0, v0_n, p1, p2, p3);});
v4=({uint64_t _a=v4;uint64_t _b=64ULL;o_op(1, 5, _a,_b);});
({uint64_t p1=v4;uint64_t p2=({uint64_t _a=101ULL;uint64_t _b=({uint64_t _a=3ULL;uint64_t _b=v3;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});uint64_t p3=7ULL;uint64_t p4=16777280ULL;f28(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=v4;uint64_t p2=o_get(v1, v1_n, 10ULL, 5);uint64_t p3=32ULL;uint64_t p4=0ULL;uint64_t p5=2ULL;f29(v0, v0_n, p1, p2, p3, p4, p5);});
({uint64_t p1=v4;uint64_t p2=4ULL;uint64_t p3=0ULL;f30(v0, v0_n, p1, p2, p3);});
return o_leave(0);
}
static uint64_t f33(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4){
o_enter();
({uint64_t p1=v1;uint64_t p2=v2;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=4ULL;o_op(1, 5, _a,_b);});uint64_t p2=v3;f4(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(1, 5, _a,_b);});uint64_t p2=v4;f5(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=48ULL;o_op(1, 5, _a,_b);});uint64_t p2=8ULL;f5(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f34(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3){
o_enter();
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=32ULL;o_op(1, 5, _a,_b);});uint64_t p2=v2;f5(v0, v0_n, p1, p2);});
({uint64_t p1=({uint64_t _a=v1;uint64_t _b=40ULL;o_op(1, 5, _a,_b);});uint64_t p2=v3;f5(v0, v0_n, p1, p2);});
return o_leave(0);
}
static uint64_t f35(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=o_get(v1, v1_n, 12ULL, 5);
({uint64_t p1=v3;uint64_t p2=6ULL;uint64_t p3=4ULL;uint64_t p4=v3;f33(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=v3;uint64_t p2=280ULL;uint64_t p3=280ULL;f34(v0, v0_n, p1, p2, p3);});
({uint64_t p1=({uint64_t _a=v3;uint64_t _b=56ULL;o_op(1, 5, _a,_b);});uint64_t p2=1ULL;uint64_t p3=4ULL;uint64_t p4=v3;f33(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=({uint64_t _a=v3;uint64_t _b=56ULL;o_op(1, 5, _a,_b);});uint64_t p2=280ULL;uint64_t p3=280ULL;f34(v0, v0_n, p1, p2, p3);});
({uint64_t p1=({uint64_t _a=v3;uint64_t _b=112ULL;o_op(1, 5, _a,_b);});uint64_t p2=1ULL;uint64_t p3=5ULL;uint64_t p4=o_get(v1, v1_n, 4ULL, 5);f33(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=({uint64_t _a=v3;uint64_t _b=112ULL;o_op(1, 5, _a,_b);});uint64_t p2=o_get(v2, v2_n, 1ULL, 5);uint64_t p3=o_get(v2, v2_n, 1ULL, 5);f34(v0, v0_n, p1, p2, p3);});
({uint64_t p1=({uint64_t _a=v3;uint64_t _b=168ULL;o_op(1, 5, _a,_b);});uint64_t p2=1ULL;uint64_t p3=6ULL;uint64_t p4=o_get(v1, v1_n, 9ULL, 5);f33(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=({uint64_t _a=v3;uint64_t _b=168ULL;o_op(1, 5, _a,_b);});uint64_t p2=0ULL;uint64_t p3=o_get(v2, v2_n, 5ULL, 5);f34(v0, v0_n, p1, p2, p3);});
({uint64_t p1=({uint64_t _a=v3;uint64_t _b=224ULL;o_op(1, 5, _a,_b);});uint64_t p2=1ULL;uint64_t p3=4ULL;uint64_t p4=o_get(v1, v1_n, 5ULL, 5);f33(v0, v0_n, p1, p2, p3, p4);});
({uint64_t p1=({uint64_t _a=v3;uint64_t _b=224ULL;o_op(1, 5, _a,_b);});uint64_t p2=o_get(v2, v2_n, 3ULL, 5);uint64_t p3=o_get(v2, v2_n, 3ULL, 5);f34(v0, v0_n, p1, p2, p3);});
return o_leave(0);
}
static uint64_t f36(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n){
o_enter();
uint64_t __attribute__((unused)) v3=o_get(v1, v1_n, 8ULL, 5);
uint64_t __attribute__((unused)) v4=({uint64_t p1=v3;f25(v0, v0_n, p1, v2, v2_n);});
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=8ULL;o_op(1, 5, _a,_b);});uint64_t _b=({uint64_t _a=16ULL;uint64_t _b=o_get(v2, v2_n, 2ULL, 5);o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
({uint64_t p1=v5;uint64_t p2=v4;uint64_t p3=o_get(v2, v2_n, 4ULL, 5);uint64_t p4=({f0(v2, v2_n);});f26(v0, v0_n, p1, p2, p3, p4);});
return o_leave(0);
}
static uint64_t f37(void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, const void *v3, uint64_t v3_n, const void *v4, uint64_t v4_n){
o_enter();
({f20(v0, v0_n, v1, v1_n, v2, v2_n);});
uint64_t __attribute__((unused)) v5=o_get(v2, v2_n, 0ULL, 5);
({uint64_t p1=o_get(v1, v1_n, 1ULL, 5);uint64_t p3=v5;f21(v0, v0_n, p1, v3, v3_n, p3);});
uint64_t __attribute__((unused)) v6=({uint64_t p1=({uint64_t _a=o_get(v1, v1_n, 2ULL, 5);uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t p3=v5;f7(v0, v0_n, p1, v3, v3_n, p3);});
({uint64_t p1=o_get(v1, v1_n, 3ULL, 5);uint64_t p2=o_get(v2, v2_n, 1ULL, 5);f22(v0, v0_n, p1, p2);});
({uint64_t p2=o_get(v1, v1_n, 4ULL, 5);uint64_t p3=o_get(v2, v2_n, 1ULL, 5);f19(v0, v0_n, v4, v4_n, p2, p3);});
({uint64_t p1=o_get(v1, v1_n, 6ULL, 5);uint64_t p2=o_get(v2, v2_n, 4ULL, 5);f23(v0, v0_n, p1, p2);});
({uint64_t p1=o_get(v1, v1_n, 7ULL, 5);f24(v0, v0_n, p1);});
({f36(v0, v0_n, v1, v1_n, v2, v2_n);});
({uint64_t p1=o_get(v1, v1_n, 10ULL, 5);uint64_t p2=({f1(v2, v2_n);});f27(v0, v0_n, p1, p2);});
({f31(v0, v0_n, v1, v1_n, v2, v2_n);});
({f32(v0, v0_n, v1, v1_n, v2, v2_n);});
({f35(v0, v0_n, v1, v1_n, v2, v2_n);});
return o_leave(0);
}
static uint64_t f38(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, const void *v2, uint64_t v2_n, void *v3, uint64_t v3_n){
o_enter();
uint64_t v4[16]={0};
uint64_t v4_n=16;
o_fill(v4, v4_n, 0ULL, 5);
uint64_t __attribute__((unused)) v5=({f15(v0, v0_n, v1, v1_n, v3, v3_n);});
if(({uint64_t _a=v5;uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(v5);
}
({f16(v0, v0_n, v4, v4_n);});
uint64_t __attribute__((unused)) v6=o_get(v4, v4_n, 13ULL, 5);
if(({uint64_t _a=v6;uint64_t _b=134217728ULL;o_op(10, 5, _a,_b);})){
{uint64_t _i=0ULL;uint64_t _v=7ULL;o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=v6;o_put(v3, v3_n, _i,_v,5);}
return o_leave(7ULL);
}
({uint64_t p1=v6;f17(v3, v3_n, p1);});
({f37(v3, v3_n, v4, v4_n, v0, v0_n, v1, v1_n, v2, v2_n);});
return o_leave(v6);
}
uint32_t omega_cubin_elf(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, void *p3, uint64_t p3_n){
if(!o_buf(result,1,1,5))return 32;
if(!o_buf(p0, p0_n,104, 5))return 32;
if(o_overlap(result,8,p0, p0_n*8))return 32;
if(!o_buf(p1, p1_n,64, 2))return 32;
if(o_overlap(result,8,p1, p1_n*1))return 32;
if(!o_buf(p2, p2_n,16777216, 2))return 32;
if(o_overlap(result,8,p2, p2_n*1))return 32;
if(!o_buf(p3, p3_n,16777216, 5))return 32;
if(o_overlap(result,8,p3, p3_n*8))return 32;
if(o_overlap(p3, p3_n*8, p0, p0_n*8))return 32;
if(o_overlap(p3, p3_n*8, p1, p1_n*1))return 32;
if(o_overlap(p3, p3_n*8, p2, p2_n*1))return 32;
if(o_busy)return 33;
o_busy=1;o_depth=0;int code=setjmp(o_jmp);
if(!code)*result=f38(p0, p0_n, p1, p1_n, p2, p2_n, p3, p3_n);
o_busy=0;o_depth=0;return (uint32_t)code;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
