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
static uint64_t f0(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4);
static uint64_t f1(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3);
static uint64_t f2(uint64_t v0);
static uint64_t f3(uint64_t v0);
static uint64_t f4(uint64_t v0, uint64_t v1);
static uint64_t f5(const void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n, void *v3, uint64_t v3_n);
static uint64_t f6(const void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f7(const void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f8(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2);
static uint64_t f9(const void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f10(const void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f11(const void *v0, uint64_t v0_n, uint64_t v1);
static uint64_t f12(const void *v0, uint64_t v0_n);
static uint64_t f13(uint64_t v0);
static uint64_t f14(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f15(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f16(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f17(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f18(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f19(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f20(void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n);
static uint64_t f21(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2);
static uint64_t f22(const void *v0, uint64_t v0_n);
static uint64_t f23(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, void *v4, uint64_t v4_n);
static uint64_t f0(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3, uint64_t v4){
o_enter();
{uint64_t _i=0ULL;uint64_t _v=v1;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=v2;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=v3;o_put(v0, v0_n, _i,_v,5);}
{uint64_t _i=3ULL;uint64_t _v=v4;o_put(v0, v0_n, _i,_v,5);}
return o_leave(0);
}
static uint64_t f1(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3){
o_enter();
if(({uint64_t _a=v1;uint64_t _b=9223372036854775808ULL;o_op(11, 5, _a,_b);})){
return o_leave(v3);
}
return o_leave(({uint64_t _a=o_get(v0, v0_n, v1, 5);uint64_t _b=v2;o_op(1, 5, _a,_b);}));
}
static uint64_t f2(uint64_t v0){
o_enter();
if(({uint64_t _a=v0;uint64_t _b=9223372036854775808ULL;o_op(11, 5, _a,_b);})){
return o_leave(0ULL);
}
return o_leave(1ULL);
}
static uint64_t f3(uint64_t v0){
o_enter();
return o_leave(({uint64_t _a=({uint64_t _a=v0;uint64_t _b=6ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v0;uint64_t _b=7ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);}));
}
static uint64_t f4(uint64_t v0, uint64_t v1){
o_enter();
if(({uint64_t _a=({uint64_t _a=v0;uint64_t _b=4ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=v0;uint64_t _b=7ULL;o_op(9, 5, _a,_b);});o_op(12, 1, _a,_b);})){
if(({uint64_t _a=v1;uint64_t _b=896ULL;o_op(11, 5, _a,_b);})){
return o_leave(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=896ULL;o_op(2, 5, _a,_b);});uint64_t _b=352ULL;o_op(1, 5, _a,_b);}));
}
}
return o_leave(v1);
}
static uint64_t f5(const void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n, void *v3, uint64_t v3_n){
o_enter();
uint64_t __attribute__((unused)) v4=({uint64_t _a=v1;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});
uint64_t __attribute__((unused)) v5=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=7ULL;o_op(1, 5, _a,_b);}), 5);
{uint64_t _i=0ULL;uint64_t _v=o_get(v0, v0_n, v4, 5);o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=({uint64_t p1=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p2=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=16ULL;o_op(16, 5, _a,_b);});uint64_t _b=255ULL;o_op(12, 5, _a,_b);});uint64_t p3=0ULL;f1(v2, v2_n, p1, p2, p3);});o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=({uint64_t p1=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=2ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p2=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=24ULL;o_op(16, 5, _a,_b);});uint64_t _b=255ULL;o_op(12, 5, _a,_b);});uint64_t p3=0ULL;f1(v2, v2_n, p1, p2, p3);});o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=3ULL;uint64_t _v=({uint64_t p1=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=3ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p2=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=32ULL;o_op(16, 5, _a,_b);});uint64_t _b=255ULL;o_op(12, 5, _a,_b);});uint64_t p3=0ULL;f1(v2, v2_n, p1, p2, p3);});o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=4ULL;uint64_t _v=({uint64_t p1=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=4ULL;o_op(1, 5, _a,_b);}), 5);uint64_t p2=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=40ULL;o_op(16, 5, _a,_b);});uint64_t _b=255ULL;o_op(12, 5, _a,_b);});uint64_t p3=255ULL;f1(v2, v2_n, p1, p2, p3);});o_put(v3, v3_n, _i,_v,5);}
uint64_t __attribute__((unused)) v6=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=5ULL;o_op(1, 5, _a,_b);}), 5);
{uint64_t _i=5ULL;uint64_t _v=0ULL;o_put(v3, v3_n, _i,_v,5);}
if(({uint64_t _a=v6;uint64_t _b=9223372036854775808ULL;o_op(8, 5, _a,_b);})){
{uint64_t _i=5ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=v6;uint64_t _b=256ULL;o_op(1, 5, _a,_b);}), 5);o_put(v3, v3_n, _i,_v,5);}
}
{uint64_t _i=14ULL;uint64_t _v=({uint64_t p0=v6;f2(p0);});o_put(v3, v3_n, _i,_v,5);}
uint64_t __attribute__((unused)) v7=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=6ULL;o_op(1, 5, _a,_b);}), 5);
{uint64_t _i=6ULL;uint64_t _v=({uint64_t p0=o_get(v3, v3_n, 0ULL, 5);uint64_t p1=({uint64_t _a=v7;uint64_t _b=4294967295ULL;o_op(12, 5, _a,_b);});f4(p0, p1);});o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=7ULL;uint64_t _v=({uint64_t _a=v7;uint64_t _b=32ULL;o_op(16, 5, _a,_b);});o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=8ULL;uint64_t _v=({uint64_t _a=v5;uint64_t _b=1ULL;o_op(12, 5, _a,_b);});o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=9ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=1ULL;o_op(16, 5, _a,_b);});uint64_t _b=1ULL;o_op(12, 5, _a,_b);});o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=10ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=2ULL;o_op(16, 5, _a,_b);});uint64_t _b=1ULL;o_op(12, 5, _a,_b);});o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=11ULL;uint64_t _v=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(16, 5, _a,_b);});uint64_t _b=255ULL;o_op(12, 5, _a,_b);});o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=12ULL;uint64_t _v=({uint64_t p0=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=2ULL;o_op(1, 5, _a,_b);}), 5);f2(p0);});o_put(v3, v3_n, _i,_v,5);}
{uint64_t _i=13ULL;uint64_t _v=({uint64_t p0=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=3ULL;o_op(1, 5, _a,_b);}), 5);f2(p0);});o_put(v3, v3_n, _i,_v,5);}
uint64_t __attribute__((unused)) v8=o_get(v0, v0_n, ({uint64_t _a=v4;uint64_t _b=1ULL;o_op(1, 5, _a,_b);}), 5);
if(({uint64_t _a=({uint64_t p0=o_get(v3, v3_n, 0ULL, 5);f3(p0);});uint64_t _b=({uint64_t _a=v8;uint64_t _b=9223372036854775808ULL;o_op(8, 5, _a,_b);});o_op(12, 1, _a,_b);})){
{uint64_t _i=1ULL;uint64_t _v=o_get(v2, v2_n, ({uint64_t _a=v8;uint64_t _b=256ULL;o_op(1, 5, _a,_b);}), 5);o_put(v3, v3_n, _i,_v,5);}
}
return o_leave(0);
}
static uint64_t f6(const void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 7ULL, 5);
if(({uint64_t _a=v2;uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(v2);
}
return o_leave(v1);
}
static uint64_t f7(const void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
return o_leave(({uint64_t _a=o_get(v0, v0_n, v1, 5);uint64_t _b=255ULL;o_op(12, 5, _a,_b);}));
}
static uint64_t f8(const void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2){
o_enter();
if(({uint64_t _a=o_get(v0, v0_n, v2, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(({uint64_t p1=v1;f7(v0, v0_n, p1);}));
}
return o_leave(255ULL);
}
static uint64_t f9(const void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
return o_leave(({uint64_t _a=v1;uint64_t _b=({uint64_t _a=({uint64_t p1=1ULL;f7(v0, v0_n, p1);});uint64_t _b=16ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);}));
}
static uint64_t f10(const void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
return o_leave(({uint64_t _a=v1;uint64_t _b=({uint64_t _a=({uint64_t p1=2ULL;f7(v0, v0_n, p1);});uint64_t _b=24ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);}));
}
static uint64_t f11(const void *v0, uint64_t v0_n, uint64_t v1){
o_enter();
return o_leave(({uint64_t _a=({uint64_t p1=v1;f9(v0, v0_n, p1);});uint64_t _b=({uint64_t _a=({uint64_t p1=2ULL;f7(v0, v0_n, p1);});uint64_t _b=24ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);}));
}
static uint64_t f12(const void *v0, uint64_t v0_n){
o_enter();
if(({uint64_t _a=o_get(v0, v0_n, 9ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(7ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 10ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
return o_leave(0ULL);
}
return o_leave(8ULL);
}
static uint64_t f13(uint64_t v0){
o_enter();
return o_leave(({uint64_t _a=({uint64_t _a=v0;uint64_t _b=8388608ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=v0;uint64_t _b=4286578688ULL;o_op(11, 5, _a,_b);});o_op(13, 1, _a,_b);}));
}
static uint64_t f14(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 0ULL, 5);
uint64_t __attribute__((unused)) v3=o_get(v0, v0_n, 6ULL, 5);
if(({uint64_t _a=v2;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=31000ULL;uint64_t p2=0ULL;uint64_t p3=0ULL;uint64_t p4=({uint64_t p1=1032192ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29186ULL;f9(v0, v0_n, p1);});uint64_t p2=255ULL;uint64_t p3=3840ULL;uint64_t p4=({uint64_t p1=1040896ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=2ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=30722ULL;f9(v0, v0_n, p1);});uint64_t p2=v3;uint64_t p3=3840ULL;uint64_t p4=({uint64_t p1=1040896ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=3ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31001ULL;f9(v0, v0_n, p1);});uint64_t p2=0ULL;uint64_t p3=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=255ULL;o_op(12, 5, _a,_b);});uint64_t _b=8ULL;o_op(15, 5, _a,_b);});uint64_t p4=({uint64_t p1=926208ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=4ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=4278221698ULL;f9(v0, v0_n, p1);});uint64_t p2=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=6ULL;o_op(15, 5, _a,_b);});uint64_t _b=16776960ULL;o_op(12, 5, _a,_b);});uint64_t p3=2048ULL;uint64_t p4=({uint64_t p1=1040896ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=5ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=4278221698ULL;f9(v0, v0_n, p1);});uint64_t p2=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=6ULL;o_op(15, 5, _a,_b);});uint64_t _b=16776960ULL;o_op(12, 5, _a,_b);});uint64_t p3=2560ULL;uint64_t p4=({uint64_t p1=926208ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=6ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31417ULL;f9(v0, v0_n, p1);});uint64_t p2=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=6ULL;o_op(15, 5, _a,_b);});uint64_t _b=16776960ULL;o_op(12, 5, _a,_b);});uint64_t p3=2048ULL;uint64_t p4=({uint64_t p1=926208ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=7ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31417ULL;f9(v0, v0_n, p1);});uint64_t p2=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=6ULL;o_op(15, 5, _a,_b);});uint64_t _b=16776960ULL;o_op(12, 5, _a,_b);});uint64_t p3=2560ULL;uint64_t p4=({uint64_t p1=945664ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=8ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 13ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29220ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=({uint64_t _a=126747136ULL;uint64_t _b=({uint64_t p1=4ULL;f7(v0, v0_n, p1);});o_op(13, 5, _a,_b);});uint64_t p4=({uint64_t p1=2083328ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 8ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 14ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
({uint64_t p1=({uint64_t p1=31780ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=5ULL;f7(v0, v0_n, p1);});uint64_t p3=({uint64_t _a=260964864ULL;uint64_t _b=({uint64_t p1=4ULL;f7(v0, v0_n, p1);});o_op(13, 5, _a,_b);});uint64_t p4=({uint64_t p1=2083328ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
({uint64_t p1=({uint64_t p1=30756ULL;f11(v0, v0_n, p1);});uint64_t p2=v3;uint64_t p3=({uint64_t _a=126747136ULL;uint64_t _b=({uint64_t p1=4ULL;f7(v0, v0_n, p1);});o_op(13, 5, _a,_b);});uint64_t p4=({uint64_t p1=2083328ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=9ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=30757ULL;f11(v0, v0_n, p1);});uint64_t p2=v3;uint64_t p3=({uint64_t _a=126746624ULL;uint64_t _b=({uint64_t p1=4ULL;f7(v0, v0_n, p1);});o_op(13, 5, _a,_b);});uint64_t p4=({uint64_t p1=2083840ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=10ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 8ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 14ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(12, 1, _a,_b);})){
({uint64_t p1=({uint64_t p1=31756ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=5ULL;f7(v0, v0_n, p1);});uint64_t p3=200302704ULL;uint64_t p4=({uint64_t p1=3136000ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
({uint64_t p1=({uint64_t p1=30732ULL;f11(v0, v0_n, p1);});uint64_t p2=v3;uint64_t p3=66085488ULL;uint64_t p4=({uint64_t p1=2087424ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=11ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31105ULL;f11(v0, v0_n, p1);});uint64_t p2=0ULL;uint64_t p3=203299072ULL;uint64_t p4=({uint64_t p1=991744ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=12ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t p0=v3;f13(p0);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(1ULL);
}
({uint64_t p1=({uint64_t p1=31110ULL;f10(v0, v0_n, p1);});uint64_t p2=({uint64_t _a=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t _b=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=16777215ULL;o_op(12, 5, _a,_b);});uint64_t _b=8ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});uint64_t p3=202381568ULL;uint64_t p4=({uint64_t p1=1040896ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=13ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31110ULL;f10(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=202447104ULL;uint64_t p4=({uint64_t p1=1040896ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=14ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29200ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=134209791ULL;uint64_t p4=({uint64_t p1=17811968ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
return o_leave(1ULL);
}
static uint64_t f15(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 0ULL, 5);
uint64_t __attribute__((unused)) v3=o_get(v0, v0_n, 6ULL, 5);
if(({uint64_t _a=({uint64_t _a=v2;uint64_t _b=15ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v2;uint64_t _b=16ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
uint64_t __attribute__((unused)) v4=0ULL;
if(({uint64_t _a=v2;uint64_t _b=16ULL;o_op(6, 5, _a,_b);})){
v4=262144ULL;
}
({uint64_t p1=({uint64_t p1=29244ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=({uint64_t _a=({uint64_t _a=({uint64_t p1=4ULL;f7(v0, v0_n, p1);});uint64_t _b=6144ULL;o_op(13, 5, _a,_b);});uint64_t _b=v4;o_op(13, 5, _a,_b);});uint64_t p4=({uint64_t p1=1040896ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=17ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t _a=({uint64_t p1=30745ULL;f9(v0, v0_n, p1);});uint64_t _b=4278190080ULL;o_op(13, 5, _a,_b);});uint64_t p2=v3;uint64_t p3=({uint64_t _a=71168ULL;uint64_t _b=({uint64_t p1=2ULL;f7(v0, v0_n, p1);});o_op(13, 5, _a,_b);});uint64_t p4=({uint64_t p1=2083328ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=18ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=30738ULL;f11(v0, v0_n, p1);});uint64_t p2=v3;uint64_t p3=126796031ULL;uint64_t p4=({uint64_t p1=2083328ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=19ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31105ULL;f11(v0, v0_n, p1);});uint64_t p2=0ULL;uint64_t p3=203298048ULL;uint64_t p4=({uint64_t p1=991744ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=20ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31105ULL;f11(v0, v0_n, p1);});uint64_t p2=0ULL;uint64_t p3=203380992ULL;uint64_t p4=({uint64_t p1=991744ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=21ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31105ULL;f11(v0, v0_n, p1);});uint64_t p2=0ULL;uint64_t p3=203364608ULL;uint64_t p4=({uint64_t p1=991744ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=22ULL;o_op(6, 5, _a,_b);})){
uint64_t __attribute__((unused)) v4=({uint64_t p1=31110ULL;f10(v0, v0_n, p1);});
if(({uint64_t _a=o_get(v0, v0_n, 9ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
v4=({uint64_t _a=({uint64_t p1=2438ULL;f10(v0, v0_n, p1);});uint64_t _b=({uint64_t _a=({f12(v0, v0_n);});uint64_t _b=12ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
}
({uint64_t p1=v4;uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=202463488ULL;uint64_t p4=({uint64_t p1=1040896ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=23ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=31122ULL;uint64_t p2=0ULL;uint64_t p3=45056ULL;uint64_t p4=({uint64_t p1=1043456ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=24ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=31122ULL;uint64_t p2=0ULL;uint64_t p3=12288ULL;uint64_t p4=({uint64_t p1=1043456ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=25ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=4278221199ULL;uint64_t p2=0ULL;uint64_t p3=33554432ULL;uint64_t p4=({uint64_t p1=1042432ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=26ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31144ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=136266111ULL;uint64_t p4=({uint64_t p1=3284480ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=27ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31144ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=203374975ULL;uint64_t p4=({uint64_t p1=3284480ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=28ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29196ULL;f10(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=66084976ULL;uint64_t p4=({uint64_t p1=2087424ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=29ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29202ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=126762239ULL;uint64_t p4=({uint64_t p1=2083328ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
return o_leave(1ULL);
}
static uint64_t f16(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 0ULL, 5);
uint64_t __attribute__((unused)) v3=o_get(v0, v0_n, 6ULL, 5);
if(({uint64_t _a=v2;uint64_t _b=30ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29217ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=0ULL;uint64_t p4=({uint64_t p1=5227520ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=31ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29217ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=0ULL;uint64_t p4=({uint64_t p1=5227520ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=32ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29216ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=4194304ULL;uint64_t p4=({uint64_t p1=5227520ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=33ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29219ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=({uint64_t p1=4ULL;f7(v0, v0_n, p1);});uint64_t p4=({uint64_t p1=5227520ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=34ULL;o_op(6, 5, _a,_b);})){
uint64_t __attribute__((unused)) v4=66084864ULL;
if(({uint64_t _a=v3;uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
v4=v3;
}
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t _a=29195ULL;uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=7ULL;o_op(12, 5, _a,_b);});uint64_t _b=16ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=2ULL;f7(v0, v0_n, p1);});uint64_t _b=24ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
({uint64_t p1=v5;uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=v4;uint64_t p4=({uint64_t p1=5227520ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=35ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29192ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=7ULL;o_op(12, 5, _a,_b);});uint64_t p4=({uint64_t p1=1034752ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=36ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29193ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=58720256ULL;uint64_t p4=({uint64_t p1=5227520ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=37ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=v3;uint64_t _b=1177681920ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29246ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=4351ULL;uint64_t p4=({uint64_t p1=5227520ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
({uint64_t p1=({uint64_t p1=29193ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=125829120ULL;uint64_t p4=({uint64_t p1=5227520ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=38ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29253ULL;f9(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=2ULL;f7(v0, v0_n, p1);});uint64_t p3=2102272ULL;uint64_t p4=({uint64_t p1=5235200ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=39ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29445ULL;f9(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=2ULL;f7(v0, v0_n, p1);});uint64_t p3=2158848ULL;uint64_t p4=({uint64_t p1=5120512ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=40ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29448ULL;f9(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=2ULL;f7(v0, v0_n, p1);});uint64_t p3=4096ULL;uint64_t p4=({uint64_t p1=926720ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=41ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29448ULL;f9(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=2ULL;f7(v0, v0_n, p1);});uint64_t p3=5120ULL;uint64_t p4=({uint64_t p1=926720ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=42ULL;o_op(6, 5, _a,_b);})){
uint64_t __attribute__((unused)) v4=({uint64_t _a=134225664ULL;uint64_t _b=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=31ULL;o_op(12, 5, _a,_b);});uint64_t _b=21ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
if(({uint64_t _a=({uint64_t _a=v3;uint64_t _b=2147483648ULL;o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
v4=({uint64_t _a=v4;uint64_t _b=({uint64_t _a=1ULL;uint64_t _b=26ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
}
({uint64_t p1=({uint64_t p1=32649ULL;f11(v0, v0_n, p1);});uint64_t p2=v4;uint64_t p3=917504ULL;uint64_t p4=({uint64_t p1=926720ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=43ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31108ULL;f11(v0, v0_n, p1);});uint64_t p2=0ULL;uint64_t p3=({uint64_t _a=v3;uint64_t _b=65535ULL;o_op(12, 5, _a,_b);});uint64_t p4=({uint64_t p1=1040896ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=44ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29576ULL;f10(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=({uint64_t _a=v3;uint64_t _b=65535ULL;o_op(12, 5, _a,_b);});uint64_t p4=({uint64_t p1=1040896ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=45ULL;o_op(6, 5, _a,_b);})){
uint64_t __attribute__((unused)) v4=31053ULL;
if(({uint64_t _a=o_get(v0, v0_n, 9ULL, 5);uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
v4=2381ULL;
}
({uint64_t p1=v4;uint64_t p2=0ULL;uint64_t p3=58720256ULL;uint64_t p4=({uint64_t p1=1042944ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
return o_leave(1ULL);
}
static uint64_t f17(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 6ULL, 5);
uint64_t __attribute__((unused)) v3=({uint64_t _a=({uint64_t _a=v2;uint64_t _b=1ULL;o_op(2, 5, _a,_b);});uint64_t _b=4294967295ULL;o_op(12, 5, _a,_b);});
uint64_t __attribute__((unused)) v4=({uint64_t _a=v3;uint64_t _b=6ULL;o_op(16, 5, _a,_b);});
uint64_t __attribute__((unused)) v5=58720256ULL;
if(({uint64_t _a=v3;uint64_t _b=2147483648ULL;o_op(11, 5, _a,_b);})){
v4=({uint64_t _a=v4;uint64_t _b=4227858432ULL;o_op(13, 5, _a,_b);});
v5=58982399ULL;
}
uint64_t __attribute__((unused)) v6=o_get(v0, v0_n, 7ULL, 5);
if(({uint64_t _a=v6;uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
v6=1042944ULL;
if(({uint64_t _a=({uint64_t _a=v2;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 9ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});o_op(12, 1, _a,_b);})){
v6=1032192ULL;
}
}
uint64_t __attribute__((unused)) v7=({uint64_t _a=({uint64_t _a=2375ULL;uint64_t _b=({uint64_t _a=({f12(v0, v0_n);});uint64_t _b=12ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=63ULL;o_op(12, 5, _a,_b);});uint64_t _b=18ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
({uint64_t p1=v7;uint64_t p2=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=2ULL;o_op(15, 5, _a,_b);});uint64_t _b=4294967295ULL;o_op(12, 5, _a,_b);});uint64_t p3=v5;uint64_t p4=v6;f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
static uint64_t f18(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 0ULL, 5);
uint64_t __attribute__((unused)) v3=o_get(v0, v0_n, 6ULL, 5);
uint64_t __attribute__((unused)) v4=o_get(v0, v0_n, 11ULL, 5);
if(({uint64_t _a=v2;uint64_t _b=46ULL;o_op(6, 5, _a,_b);})){
return o_leave(({f17(v0, v0_n, v1, v1_n);}));
}
if(({uint64_t _a=v2;uint64_t _b=47ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29448ULL;f9(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=2ULL;f7(v0, v0_n, p1);});uint64_t p3=2048ULL;uint64_t p4=({uint64_t p1=926720ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=48ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=31517ULL;uint64_t p2=0ULL;uint64_t p3=65536ULL;uint64_t p4=({uint64_t p1=1043456ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=49ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31108ULL;f11(v0, v0_n, p1);});uint64_t p2=255ULL;uint64_t p3=134219776ULL;uint64_t p4=({uint64_t p1=927744ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=50ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=31112ULL;f10(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=3ULL;f7(v0, v0_n, p1);});uint64_t p3=134220031ULL;uint64_t p4=({uint64_t p1=1040896ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=51ULL;o_op(6, 5, _a,_b);})){
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t _a=29202ULL;uint64_t _b=({uint64_t _a=({uint64_t p1=1ULL;f7(v0, v0_n, p1);});uint64_t _b=16ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=2ULL;uint64_t p2=12ULL;f8(v0, v0_n, p1, p2);});uint64_t _b=24ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
({uint64_t p1=v5;uint64_t p2=({uint64_t p1=3ULL;uint64_t p2=13ULL;f8(v0, v0_n, p1, p2);});uint64_t p3=({uint64_t _a=({uint64_t _a=126746624ULL;uint64_t _b=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=255ULL;o_op(12, 5, _a,_b);});uint64_t _b=8ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});uint64_t _b=({uint64_t p1=4ULL;f7(v0, v0_n, p1);});o_op(13, 5, _a,_b);});uint64_t p4=({uint64_t p1=2083328ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=52ULL;o_op(6, 5, _a,_b);})){
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t _a=29209ULL;uint64_t _b=({uint64_t _a=({uint64_t p1=1ULL;f7(v0, v0_n, p1);});uint64_t _b=16ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=2ULL;uint64_t p2=12ULL;f8(v0, v0_n, p1, p2);});uint64_t _b=24ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
({uint64_t p1=v5;uint64_t p2=({uint64_t p1=3ULL;uint64_t p2=13ULL;f8(v0, v0_n, p1, p2);});uint64_t p3=({uint64_t _a=1536ULL;uint64_t _b=({uint64_t p1=4ULL;f7(v0, v0_n, p1);});o_op(13, 5, _a,_b);});uint64_t p4=({uint64_t p1=2083328ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=53ULL;o_op(6, 5, _a,_b);})){
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t _a=29223ULL;uint64_t _b=({uint64_t _a=({uint64_t p1=1ULL;f7(v0, v0_n, p1);});uint64_t _b=16ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t p1=2ULL;uint64_t p2=12ULL;f8(v0, v0_n, p1, p2);});uint64_t _b=24ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
({uint64_t p1=v5;uint64_t p2=({uint64_t p1=3ULL;uint64_t p2=13ULL;f8(v0, v0_n, p1, p2);});uint64_t p3=({uint64_t _a=126746624ULL;uint64_t _b=({uint64_t p1=4ULL;f7(v0, v0_n, p1);});o_op(13, 5, _a,_b);});uint64_t p4=({uint64_t p1=2083328ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=54ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=2ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=v3;uint64_t _b=2147483648ULL;o_op(11, 5, _a,_b);});o_op(13, 1, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=16ULL;o_op(11, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(1ULL);
}
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t _a=2373ULL;uint64_t _b=({uint64_t _a=({f12(v0, v0_n);});uint64_t _b=12ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=16ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
({uint64_t p1=v5;uint64_t p2=({uint64_t _a=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=1ULL;o_op(2, 5, _a,_b);});uint64_t _b=16ULL;o_op(3, 5, _a,_b);});uint64_t _b=4294967295ULL;o_op(12, 5, _a,_b);});uint64_t p3=58720768ULL;uint64_t p4=({uint64_t p1=1040896ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=55ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=v4;uint64_t _b=16ULL;o_op(11, 5, _a,_b);})){
return o_leave(1ULL);
}
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t _a=2369ULL;uint64_t _b=({uint64_t _a=({f12(v0, v0_n);});uint64_t _b=12ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=16ULL;o_op(15, 5, _a,_b);});o_op(13, 5, _a,_b);});
({uint64_t p1=v5;uint64_t p2=0ULL;uint64_t p3=58720768ULL;uint64_t p4=({uint64_t p1=1042944ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=56ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t p0=v3;f13(p0);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(1ULL);
}
({uint64_t p1=({uint64_t p1=31105ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=16777215ULL;o_op(12, 5, _a,_b);});uint64_t _b=8ULL;o_op(15, 5, _a,_b);});uint64_t p3=203299584ULL;uint64_t p4=({uint64_t p1=991744ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=57ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t p0=v3;f13(p0);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(1ULL);
}
({uint64_t p1=({uint64_t p1=31105ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=16777215ULL;o_op(12, 5, _a,_b);});uint64_t _b=8ULL;o_op(15, 5, _a,_b);});uint64_t p3=203300096ULL;uint64_t p4=({uint64_t p1=991744ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=58ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=v3;uint64_t _b=8388608ULL;o_op(11, 5, _a,_b);})){
return o_leave(1ULL);
}
({uint64_t p1=({uint64_t p1=31108ULL;f11(v0, v0_n, p1);});uint64_t p2=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=16777215ULL;o_op(12, 5, _a,_b);});uint64_t _b=8ULL;o_op(15, 5, _a,_b);});uint64_t p3=3072ULL;uint64_t p4=({uint64_t p1=1042432ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
if(({uint64_t _a=v2;uint64_t _b=59ULL;o_op(6, 5, _a,_b);})){
({uint64_t p1=({uint64_t p1=29448ULL;f9(v0, v0_n, p1);});uint64_t p2=({uint64_t p1=2ULL;f7(v0, v0_n, p1);});uint64_t p3=3072ULL;uint64_t p4=({uint64_t p1=926720ULL;f6(v0, v0_n, p1);});f0(v1, v1_n, p1, p2, p3, p4);});
return o_leave(0ULL);
}
return o_leave(1ULL);
}
static uint64_t f19(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
uint64_t __attribute__((unused)) v2=o_get(v0, v0_n, 0ULL, 5);
if(({uint64_t _a=v2;uint64_t _b=15ULL;o_op(8, 5, _a,_b);})){
return o_leave(({f14(v0, v0_n, v1, v1_n);}));
}
if(({uint64_t _a=v2;uint64_t _b=30ULL;o_op(8, 5, _a,_b);})){
return o_leave(({f15(v0, v0_n, v1, v1_n);}));
}
if(({uint64_t _a=v2;uint64_t _b=46ULL;o_op(8, 5, _a,_b);})){
return o_leave(({f16(v0, v0_n, v1, v1_n);}));
}
return o_leave(({f18(v0, v0_n, v1, v1_n);}));
}
static uint64_t f20(void *v0, uint64_t v0_n, uint64_t v1, const void *v2, uint64_t v2_n){
o_enter();
{
uint64_t v3=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v3, k1);v3=o_op(1,5, v3,1)){
uint64_t __attribute__((unused)) v4=o_get(v2, v2_n, v3, 5);
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=16ULL;o_op(3, 5, _a,_b);});uint64_t _b=({uint64_t _a=v3;uint64_t _b=4ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});
{
uint64_t v6=0;
uint64_t k2=4ULL;
for(;o_op(8,5, v6, k2);v6=o_op(1,5, v6,1)){
{uint64_t _i=({uint64_t _a=v5;uint64_t _b=v6;o_op(1, 5, _a,_b);});uint64_t _v=o_cast(2, 5, ({uint64_t _a=({uint64_t _a=v4;uint64_t _b=({uint64_t _a=v6;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(16, 5, _a,_b);});uint64_t _b=255ULL;o_op(12, 5, _a,_b);}));o_put(v0, v0_n, _i,_v,2);}
}
}
}
}
return o_leave(0);
}
static uint64_t f21(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2){
o_enter();
uint64_t __attribute__((unused)) v3=({uint64_t _a=v2;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});
uint64_t __attribute__((unused)) v4=o_get(v0, v0_n, v3, 5);
uint64_t __attribute__((unused)) v5=o_get(v0, v0_n, ({uint64_t _a=v3;uint64_t _b=7ULL;o_op(1, 5, _a,_b);}), 5);
uint64_t __attribute__((unused)) v6=({uint64_t _a=o_get(v0, v0_n, ({uint64_t _a=v3;uint64_t _b=6ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=4294967295ULL;o_op(12, 5, _a,_b);});
if(({uint64_t _a=v4;uint64_t _b=59ULL;o_op(10, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=16ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v6;uint64_t _b=1362114610ULL;o_op(6, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=37ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v6;uint64_t _b=1178940467ULL;o_op(6, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=4ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=7ULL;o_op(9, 5, _a,_b);});o_op(12, 1, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=v6;uint64_t _b=65536ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v6;uint64_t _b=4ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=12ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=56ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=57ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
if(({uint64_t _a=({uint64_t p0=v6;f13(p0);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(0ULL);
}
}
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=58ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v6;uint64_t _b=8388608ULL;o_op(11, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=v4;uint64_t _b=54ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v6;uint64_t _b=2ULL;o_op(8, 5, _a,_b);});uint64_t _b=({uint64_t _a=v6;uint64_t _b=2147483648ULL;o_op(11, 5, _a,_b);});o_op(13, 1, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=54ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=55ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=8ULL;o_op(16, 5, _a,_b);});uint64_t _b=255ULL;o_op(12, 5, _a,_b);});uint64_t _b=16ULL;o_op(11, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(0ULL);
}
{
uint64_t v7=0;
uint64_t k1=4ULL;
for(;o_op(8,5, v7, k1);v7=o_op(1,5, v7,1)){
uint64_t __attribute__((unused)) v8=o_get(v0, v0_n, ({uint64_t _a=({uint64_t _a=v3;uint64_t _b=1ULL;o_op(1, 5, _a,_b);});uint64_t _b=v7;o_op(1, 5, _a,_b);}), 5);
if(({uint64_t _a=v8;uint64_t _b=9223372036854775808ULL;o_op(8, 5, _a,_b);})){
uint64_t __attribute__((unused)) v9=({uint64_t _a=({uint64_t _a=v5;uint64_t _b=({uint64_t _a=16ULL;uint64_t _b=({uint64_t _a=v7;uint64_t _b=8ULL;o_op(3, 5, _a,_b);});o_op(1, 5, _a,_b);});o_op(16, 5, _a,_b);});uint64_t _b=255ULL;o_op(12, 5, _a,_b);});
if(({uint64_t _a=({uint64_t _a=v7;uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v4;uint64_t _b=6ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=v4;uint64_t _b=7ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);});o_op(12, 1, _a,_b);})){
if(({uint64_t _a=v8;uint64_t _b=32ULL;o_op(11, 5, _a,_b);})){
return o_leave(0ULL);
}
uint64_t __attribute__((unused)) v10=o_get(v1, v1_n, ({uint64_t _a=v8;uint64_t _b=256ULL;o_op(1, 5, _a,_b);}), 5);
if(({uint64_t _a=({uint64_t _a=v10;uint64_t _b=64ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v10;uint64_t _b=v9;o_op(1, 5, _a,_b);});uint64_t _b=64ULL;o_op(11, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
}else{
if(({uint64_t _a=v8;uint64_t _b=256ULL;o_op(11, 5, _a,_b);})){
return o_leave(0ULL);
}
uint64_t __attribute__((unused)) v10=o_get(v1, v1_n, v8, 5);
if(({uint64_t _a=({uint64_t _a=v10;uint64_t _b=255ULL;o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=v10;uint64_t _b=v9;o_op(1, 5, _a,_b);});uint64_t _b=255ULL;o_op(11, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
}
}
}
}
uint64_t __attribute__((unused)) v7=o_get(v0, v0_n, ({uint64_t _a=v3;uint64_t _b=5ULL;o_op(1, 5, _a,_b);}), 5);
if(({uint64_t _a=v7;uint64_t _b=9223372036854775808ULL;o_op(8, 5, _a,_b);})){
if(({uint64_t _a=v7;uint64_t _b=32ULL;o_op(11, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=o_get(v1, v1_n, ({uint64_t _a=v7;uint64_t _b=256ULL;o_op(1, 5, _a,_b);}), 5);uint64_t _b=64ULL;o_op(11, 5, _a,_b);})){
return o_leave(0ULL);
}
}
return o_leave(1ULL);
}
static uint64_t f22(const void *v0, uint64_t v0_n){
o_enter();
return o_leave(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=7ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 6ULL, 5);uint64_t _b=856ULL;o_op(6, 5, _a,_b);});o_op(12, 1, _a,_b);}));
}
static uint64_t f23(const void *v0, uint64_t v0_n, const void *v1, uint64_t v1_n, uint64_t v2, uint64_t v3, void *v4, uint64_t v4_n){
o_enter();
uint64_t v5[16]={0};
uint64_t v5_n=16;
o_fill(v5, v5_n, 0ULL, 5);
uint64_t v6[4]={0};
uint64_t v6_n=4;
o_fill(v6, v6_n, 0ULL, 5);
if(({uint64_t _a=v2;uint64_t _b=1024ULL;o_op(10, 5, _a,_b);})){
return o_leave(0ULL);
}
{
uint64_t v7=0;
uint64_t c1=0;
for(;o_op(8,5, v7, v2);v7=o_op(1,5, v7,1)){
if(c1++>=16777216)o_trap(4);
if(({uint64_t _a=({uint64_t p2=v7;f21(v0, v0_n, v1, v1_n, p2);});uint64_t _b=0;o_op(18, 1, _a,_b);})){
return o_leave(0ULL);
}
}
}
uint64_t __attribute__((unused)) v7=v2;
if(({uint64_t _a=v3;uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
v7=({uint64_t _a=({uint64_t _a=({uint64_t _a=v2;uint64_t _b=7ULL;o_op(1, 5, _a,_b);});uint64_t _b=8ULL;o_op(4, 5, _a,_b);});uint64_t _b=8ULL;o_op(3, 5, _a,_b);});
if(({uint64_t _a=v7;uint64_t _b=32ULL;o_op(8, 5, _a,_b);})){
v7=32ULL;
}
}
{
uint64_t v8=0;
uint64_t c1=0;
for(;o_op(8,5, v8, v7);v8=o_op(1,5, v8,1)){
if(c1++>=16777216)o_trap(4);
if(({uint64_t _a=v8;uint64_t _b=v2;o_op(8, 5, _a,_b);})){
({uint64_t p1=v8;f5(v0, v0_n, p1, v1, v1_n, v5, v5_n);});
if(({f22(v5, v5_n);})){
({uint64_t p1=31000ULL;uint64_t p2=0ULL;uint64_t p3=0ULL;uint64_t p4=1032192ULL;f0(v6, v6_n, p1, p2, p3, p4);});
}else{
if(({uint64_t _a=({f19(v5, v5_n, v6, v6_n);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(0ULL);
}
}
}else{
({uint64_t p1=31000ULL;uint64_t p2=0ULL;uint64_t p3=0ULL;uint64_t p4=1032192ULL;f0(v6, v6_n, p1, p2, p3, p4);});
}
({uint64_t p1=v8;f20(v4, v4_n, p1, v6, v6_n);});
}
}
return o_leave(({uint64_t _a=v7;uint64_t _b=16ULL;o_op(3, 5, _a,_b);}));
}
uint32_t omega_ampere_enc_sm86(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, uint64_t p2, uint64_t p3, void *p4, uint64_t p4_n){
if(!o_buf(result,1,1,5))return 32;
if(!o_buf(p0, p0_n,8192, 5))return 32;
if(o_overlap(result,8,p0, p0_n*8))return 32;
if(!o_buf(p1, p1_n,288, 5))return 32;
if(o_overlap(result,8,p1, p1_n*8))return 32;
if(!o_valid(5, p2))return 32;
if(!o_valid(5, p3))return 32;
if(!o_buf(p4, p4_n,16384, 2))return 32;
if(o_overlap(result,8,p4, p4_n*1))return 32;
if(o_overlap(p4, p4_n*1, p0, p0_n*8))return 32;
if(o_overlap(p4, p4_n*1, p1, p1_n*8))return 32;
if(o_busy)return 33;
o_busy=1;o_depth=0;int code=setjmp(o_jmp);
if(!code)*result=f23(p0, p0_n, p1, p1_n, p2, p3, p4, p4_n);
o_busy=0;o_depth=0;return (uint32_t)code;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
