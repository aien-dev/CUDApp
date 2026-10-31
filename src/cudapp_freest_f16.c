/* Omega Bridge portable nontrapping scalar profile. No host runtime. */
#include <stdint.h>
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
static unsigned ombr_fs_width(unsigned t){static const unsigned w[]={0,1,8,16,32,64,8,16,32,64};return w[t];}
static int ombr_fs_valid(unsigned t,uint64_t v){unsigned w=ombr_fs_width(t);if(t<6)return w==64||v<((uint64_t)1<<w);if(w==64)return 1;uint64_t m=(uint64_t)1<<(w-1);return v<m||v>=UINT64_MAX-m+1;}
static uint64_t ombr_fs_op(unsigned op,unsigned t,uint64_t a,uint64_t b){unsigned w=ombr_fs_width(t);uint64_t mask=UINT64_MAX>>(64-w);if(op>=6&&op<=11){uint64_t x=a,y=b;if(t>=6){x^=(uint64_t)1<<63;y^=(uint64_t)1<<63;}switch(op){case 6:return x==y;case 7:return x!=y;case 8:return x<y;case 9:return x<=y;case 10:return x>y;default:return x>=y;}}switch(op){case 1:return (a+b)&mask;case 2:return (a-b)&mask;case 3:return (a*b)&mask;case 12:return a&b;case 13:return a|b;case 14:return a^b;case 15:return (a<<b)&mask;case 16:return a>>b;case 17:return (0-a)&mask;default:return t==1?!a:t>=6?~a:(~a)&mask;}}
static uint64_t f0(uint64_t v0, uint64_t v1);
static uint64_t f1(uint64_t v0);
static uint64_t f0(uint64_t v0, uint64_t v1){
if(ombr_fs_op(6, 5, v0, 112ULL)){
uint64_t v2=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v1, 14ULL), 1ULL);
uint64_t v3=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v1, 8191ULL), v2);
return ombr_fs_op(16, 5, v3, 14ULL);
}
if(ombr_fs_op(6, 5, v0, 111ULL)){
uint64_t v2=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v1, 15ULL), 1ULL);
uint64_t v3=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v1, 16383ULL), v2);
return ombr_fs_op(16, 5, v3, 15ULL);
}
if(ombr_fs_op(6, 5, v0, 110ULL)){
uint64_t v2=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v1, 16ULL), 1ULL);
uint64_t v3=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v1, 32767ULL), v2);
return ombr_fs_op(16, 5, v3, 16ULL);
}
if(ombr_fs_op(6, 5, v0, 109ULL)){
uint64_t v2=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v1, 17ULL), 1ULL);
uint64_t v3=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v1, 65535ULL), v2);
return ombr_fs_op(16, 5, v3, 17ULL);
}
if(ombr_fs_op(6, 5, v0, 108ULL)){
uint64_t v2=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v1, 18ULL), 1ULL);
uint64_t v3=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v1, 131071ULL), v2);
return ombr_fs_op(16, 5, v3, 18ULL);
}
if(ombr_fs_op(6, 5, v0, 107ULL)){
uint64_t v2=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v1, 19ULL), 1ULL);
uint64_t v3=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v1, 262143ULL), v2);
return ombr_fs_op(16, 5, v3, 19ULL);
}
if(ombr_fs_op(6, 5, v0, 106ULL)){
uint64_t v2=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v1, 20ULL), 1ULL);
uint64_t v3=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v1, 524287ULL), v2);
return ombr_fs_op(16, 5, v3, 20ULL);
}
if(ombr_fs_op(6, 5, v0, 105ULL)){
uint64_t v2=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v1, 21ULL), 1ULL);
uint64_t v3=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v1, 1048575ULL), v2);
return ombr_fs_op(16, 5, v3, 21ULL);
}
if(ombr_fs_op(6, 5, v0, 104ULL)){
uint64_t v2=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v1, 22ULL), 1ULL);
uint64_t v3=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v1, 2097151ULL), v2);
return ombr_fs_op(16, 5, v3, 22ULL);
}
if(ombr_fs_op(6, 5, v0, 103ULL)){
uint64_t v2=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v1, 23ULL), 1ULL);
uint64_t v3=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v1, 4194303ULL), v2);
return ombr_fs_op(16, 5, v3, 23ULL);
}
if(ombr_fs_op(6, 5, v0, 102ULL)){
uint64_t v2=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v1, 24ULL), 1ULL);
uint64_t v3=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v1, 8388607ULL), v2);
return ombr_fs_op(16, 5, v3, 24ULL);
}
return 0ULL;
}
static uint64_t f1(uint64_t v0){
uint64_t v1=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v0, 16ULL), 32768ULL);
uint64_t v2=ombr_fs_op(12, 5, v0, 2147483647ULL);
if(ombr_fs_op(11, 5, v2, 2139095040ULL)){
if(ombr_fs_op(10, 5, v2, 2139095040ULL)){
uint64_t v3=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v2, 13ULL), 1023ULL);
uint64_t v4=ombr_fs_op(13, 5, 32256ULL, v3);
return ombr_fs_op(13, 5, v1, v4);
}
return ombr_fs_op(13, 5, v1, 31744ULL);
}
if(ombr_fs_op(11, 5, v2, 1199566848ULL)){
return ombr_fs_op(13, 5, v1, 31744ULL);
}
if(ombr_fs_op(11, 5, v2, 947912704ULL)){
uint64_t v3=ombr_fs_op(12, 5, ombr_fs_op(16, 5, v2, 13ULL), 1ULL);
uint64_t v4=ombr_fs_op(1, 5, ombr_fs_op(1, 5, v2, 4095ULL), v3);
uint64_t v5=ombr_fs_op(2, 5, v4, 939524096ULL);
return ombr_fs_op(13, 5, v1, ombr_fs_op(16, 5, v5, 13ULL));
}
if(ombr_fs_op(10, 5, v2, 855638016ULL)){
uint64_t v3=ombr_fs_op(16, 5, v2, 23ULL);
uint64_t v4=ombr_fs_op(13, 5, 8388608ULL, ombr_fs_op(12, 5, v2, 8388607ULL));
uint64_t v5=f0(v3, v4);
return ombr_fs_op(13, 5, v1, v5);
}
return v1;
}
uint32_t omega_native_f16(uint64_t *result, uint64_t p0){
if(!result || (uintptr_t)result%_Alignof(uint64_t) || (uintptr_t)result>UINTPTR_MAX-sizeof(uint64_t))return 32;
if(!ombr_fs_valid(5, p0))return 32;
*result=f1(p0);
return 0;
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
