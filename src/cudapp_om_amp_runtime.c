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
static uint64_t f0(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3);
static uint64_t f1(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n);
static uint64_t f0(void *v0, uint64_t v0_n, uint64_t v1, uint64_t v2, uint64_t v3){
o_enter();
{
uint64_t v4=0;
uint64_t k1=64ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
if(({uint64_t _a=({uint64_t _a=v1;uint64_t _b=v4;o_op(1, 5, _a,_b);});uint64_t _b=v2;o_op(10, 5, _a,_b);})){
break;
}
uint64_t __attribute__((unused)) v5=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=v4;o_op(1, 5, _a,_b);});uint64_t _b=32ULL;o_op(4, 5, _a,_b);});
uint64_t __attribute__((unused)) v6=({uint64_t _a=({uint64_t _a=v1;uint64_t _b=v4;o_op(1, 5, _a,_b);});uint64_t _b=32ULL;o_op(5, 5, _a,_b);});
uint64_t __attribute__((unused)) v7=({uint64_t _a=({uint64_t _a=({uint64_t _a=v3;uint64_t _b=v4;o_op(16, 5, _a,_b);});uint64_t _b=1ULL;o_op(12, 5, _a,_b);});uint64_t _b=v6;o_op(15, 5, _a,_b);});
{uint64_t _i=v5;uint64_t _v=({uint64_t _a=o_get(v0, v0_n, v5, 5);uint64_t _b=v7;o_op(13, 5, _a,_b);});o_put(v0, v0_n, _i,_v,5);}
}
}
return o_leave(0);
}
static uint64_t f1(const void *v0, uint64_t v0_n, void *v1, uint64_t v1_n){
o_enter();
if(({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=1ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=o_get(v0, v0_n, 2ULL, 5);o_op(6, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 10ULL, 5);uint64_t _b=7ULL;o_op(7, 5, _a,_b);})){
return o_leave(0ULL);
}
{uint64_t _i=0ULL;uint64_t _v=86ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=1ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=o_get(v0, v0_n, 4ULL, 5);o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=3ULL;uint64_t _v=o_get(v0, v0_n, 5ULL, 5);o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=4ULL;uint64_t _v=o_get(v0, v0_n, 6ULL, 5);o_put(v1, v1_n, _i,_v,5);}
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=o_get(v0, v0_n, 3ULL, 5);o_op(6, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 11ULL, 5);uint64_t _b=7ULL;o_op(7, 5, _a,_b);})){
return o_leave(0ULL);
}
{uint64_t _i=0ULL;uint64_t _v=121ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=1ULL;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=2ULL;uint64_t _v=o_get(v0, v0_n, 7ULL, 5);o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=3ULL;uint64_t _v=o_get(v0, v0_n, 8ULL, 5);o_put(v1, v1_n, _i,_v,5);}
{uint64_t _i=4ULL;uint64_t _v=o_get(v0, v0_n, 9ULL, 5);o_put(v1, v1_n, _i,_v,5);}
return o_leave(1ULL);
}
return o_leave(0ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=2ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=o_get(v0, v0_n, 1ULL, 5);o_op(10, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=o_get(v0, v0_n, 2ULL, 5);o_op(2, 5, _a,_b);});uint64_t _b=o_get(v0, v0_n, 3ULL, 5);o_op(8, 5, _a,_b);})){
return o_leave(0ULL);
}
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=4ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);})){
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=o_get(v0, v0_n, 8ULL, 5);o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=5ULL;o_op(6, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=o_get(v0, v0_n, 9ULL, 5);o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=4ULL;o_op(6, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
return o_leave(0ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=1ULL;o_op(7, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=o_get(v0, v0_n, 5ULL, 5);o_op(6, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=o_get(v0, v0_n, 6ULL, 5);o_op(6, 5, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=o_get(v0, v0_n, 10ULL, 5);o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=o_get(v0, v0_n, 11ULL, 5);o_op(6, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=o_get(v0, v0_n, 7ULL, 5);o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=3ULL;o_op(6, 5, _a,_b);});o_op(12, 1, _a,_b);})){
return o_leave(1ULL);
}
return o_leave(0ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 0ULL, 5);uint64_t _b=3ULL;o_op(7, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=128ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=({uint64_t _a=1ULL;uint64_t _b=49ULL;o_op(15, 5, _a,_b);});o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=({uint64_t _a=1ULL;uint64_t _b=49ULL;o_op(15, 5, _a,_b);});o_op(11, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=16ULL;o_op(5, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=({uint64_t _a=1ULL;uint64_t _b=40ULL;o_op(15, 5, _a,_b);});o_op(11, 5, _a,_b);});uint64_t _b=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=15ULL;o_op(12, 5, _a,_b);});uint64_t _b=0ULL;o_op(7, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 5ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=1024ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 5ULL, 5);uint64_t _b=1024ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 4ULL, 5);uint64_t _b=o_get(v0, v0_n, 5ULL, 5);o_op(3, 5, _a,_b);});uint64_t _b=1024ULL;o_op(10, 5, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 6ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 7ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 6ULL, 5);uint64_t _b=2147483647ULL;o_op(10, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 7ULL, 5);uint64_t _b=65535ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 8ULL, 5);uint64_t _b=0ULL;o_op(6, 5, _a,_b);});uint64_t _b=({uint64_t _a=o_get(v0, v0_n, 8ULL, 5);uint64_t _b=255ULL;o_op(10, 5, _a,_b);});o_op(13, 1, _a,_b);})){
return o_leave(0ULL);
}
if(({uint64_t _a=o_get(v0, v0_n, 9ULL, 5);uint64_t _b=({uint64_t _a=99ULL;uint64_t _b=10ULL;o_op(15, 5, _a,_b);});o_op(10, 5, _a,_b);})){
return o_leave(0ULL);
}
uint64_t __attribute__((unused)) v2=({uint64_t _a=({uint64_t _a=({uint64_t _a=o_get(v0, v0_n, 9ULL, 5);uint64_t _b=255ULL;o_op(1, 5, _a,_b);});uint64_t _b=256ULL;o_op(4, 5, _a,_b);});uint64_t _b=256ULL;o_op(3, 5, _a,_b);});
uint64_t __attribute__((unused)) v3=9ULL;
if(({uint64_t _a=v2;uint64_t _b=({uint64_t _a=32ULL;uint64_t _b=10ULL;o_op(15, 5, _a,_b);});o_op(10, 5, _a,_b);})){
v3=({uint64_t _a=({uint64_t _a=64ULL;uint64_t _b=4ULL;o_op(4, 5, _a,_b);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
}
if(({uint64_t _a=v2;uint64_t _b=({uint64_t _a=64ULL;uint64_t _b=10ULL;o_op(15, 5, _a,_b);});o_op(10, 5, _a,_b);})){
v3=({uint64_t _a=({uint64_t _a=100ULL;uint64_t _b=4ULL;o_op(4, 5, _a,_b);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});
}
{
uint64_t v4=0;
uint64_t k1=64ULL;
for(;o_op(8,5, v4, k1);v4=o_op(1,5, v4,1)){
{uint64_t _i=v4;uint64_t _v=0ULL;o_put(v1, v1_n, _i,_v,5);}
}
}
({uint64_t p1=189ULL;uint64_t p2=191ULL;uint64_t p3=7ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=368ULL;uint64_t p2=369ULL;uint64_t p3=1ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=378ULL;uint64_t p2=378ULL;uint64_t p3=1ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=384ULL;uint64_t p2=415ULL;uint64_t p3=o_get(v0, v0_n, 6ULL, 5);f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=416ULL;uint64_t p2=431ULL;uint64_t p3=o_get(v0, v0_n, 7ULL, 5);f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=448ULL;uint64_t p2=463ULL;uint64_t p3=1ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=544ULL;uint64_t p2=561ULL;uint64_t p3=v2;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=562ULL;uint64_t p2=567ULL;uint64_t p3=v3;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=569ULL;uint64_t p2=574ULL;uint64_t p3=({uint64_t _a=({uint64_t _a=100ULL;uint64_t _b=4ULL;o_op(4, 5, _a,_b);});uint64_t _b=1ULL;o_op(1, 5, _a,_b);});f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=576ULL;uint64_t p2=579ULL;uint64_t p3=0ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=580ULL;uint64_t p2=583ULL;uint64_t p3=3ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=592ULL;uint64_t p2=607ULL;uint64_t p3=o_get(v0, v0_n, 4ULL, 5);f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=608ULL;uint64_t p2=623ULL;uint64_t p3=o_get(v0, v0_n, 5ULL, 5);f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=624ULL;uint64_t p2=639ULL;uint64_t p3=1ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=640ULL;uint64_t p2=640ULL;uint64_t p3=1ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=648ULL;uint64_t p2=656ULL;uint64_t p3=o_get(v0, v0_n, 8ULL, 5);f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=657ULL;uint64_t p2=662ULL;uint64_t p3=v3;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=763ULL;uint64_t p2=767ULL;uint64_t p3=1ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=768ULL;uint64_t p2=799ULL;uint64_t p3=o_get(v0, v0_n, 3ULL, 5);f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=800ULL;uint64_t p2=807ULL;uint64_t p3=({uint64_t _a=o_get(v0, v0_n, 3ULL, 5);uint64_t _b=32ULL;o_op(16, 5, _a,_b);});f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=819ULL;uint64_t p2=819ULL;uint64_t p3=1ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=823ULL;uint64_t p2=823ULL;uint64_t p3=1ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=830ULL;uint64_t p2=831ULL;uint64_t p3=1ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=832ULL;uint64_t p2=863ULL;uint64_t p3=1ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=1024ULL;uint64_t p2=1055ULL;uint64_t p3=o_get(v0, v0_n, 2ULL, 5);f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=1056ULL;uint64_t p2=1072ULL;uint64_t p3=({uint64_t _a=o_get(v0, v0_n, 2ULL, 5);uint64_t _b=32ULL;o_op(16, 5, _a,_b);});f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=1074ULL;uint64_t p2=1074ULL;uint64_t p3=1ULL;f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=1075ULL;uint64_t p2=1087ULL;uint64_t p3=({uint64_t _a=4096ULL;uint64_t _b=4ULL;o_op(16, 5, _a,_b);});f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=1536ULL;uint64_t p2=1567ULL;uint64_t p3=o_get(v0, v0_n, 1ULL, 5);f0(v1, v1_n, p1, p2, p3);});
({uint64_t p1=1568ULL;uint64_t p2=1584ULL;uint64_t p3=({uint64_t _a=o_get(v0, v0_n, 1ULL, 5);uint64_t _b=32ULL;o_op(16, 5, _a,_b);});f0(v1, v1_n, p1, p2, p3);});
return o_leave(1ULL);
}
uint32_t omega_amp_runtime(uint64_t *result, const void *p0, uint64_t p0_n, void *p1, uint64_t p1_n){
if(!o_buf(result,1,1,5))return 32;
if(!o_buf(p0, p0_n,16, 5))return 32;
if(o_overlap(result,8,p0, p0_n*8))return 32;
if(!o_buf(p1, p1_n,64, 5))return 32;
if(o_overlap(result,8,p1, p1_n*8))return 32;
if(o_overlap(p1, p1_n*8, p0, p0_n*8))return 32;
if(o_busy)return 33;
o_busy=1;o_depth=0;int code=setjmp(o_jmp);
if(!code)*result=f1(p0, p0_n, p1, p1_n);
o_busy=0;o_depth=0;return (uint32_t)code;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
