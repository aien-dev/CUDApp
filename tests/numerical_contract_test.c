/* SPDX-License-Identifier: MIT */
#include "omega_gpu_numerical_contract.h"
#include "omega_gpu_fp8_api.h"
#include "omega_blackwell_matmul.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
static unsigned checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL %u: %s\n",checks,#x); return 1; } } while (0)
int main(void) {
    CHECK(strcmp(omega_numeric_contract_id(), "cudapp.numeric.v1") == 0);
    float a[] = {0, -0.0f, 1, 1}, b[] = {0, -0.0f, 1, 1};
    uint32_t token = 99;
    CHECK(!omega_numeric_argmax(a, 4, &token) && token == 2);
    CHECK(!omega_numeric_compare(a,b,4,OMEGA_NUMERIC_BITWISE,0,0));
    b[1]=0; CHECK(omega_numeric_compare(a,b,4,OMEGA_NUMERIC_BITWISE,0,0));
    b[1]=-0.0f; b[2]=nextafterf(1,2);
    CHECK(omega_numeric_compare(a,b,4,OMEGA_NUMERIC_REPEATABLE,0,0));
    CHECK(!omega_numeric_compare(a,b,4,OMEGA_NUMERIC_EXPERIMENTAL,1e-6,0));
    CHECK(omega_numeric_compare(a,b,4,OMEGA_NUMERIC_EXPERIMENTAL,-1,0));
    CHECK(omega_numeric_compare(a,b,4,OMEGA_NUMERIC_EXPERIMENTAL,INFINITY,0));
    CHECK(omega_numeric_compare(a,b,4,999,0,0));
    b[0]=NAN; CHECK(omega_numeric_compare(a,b,4,OMEGA_NUMERIC_EXPERIMENTAL,1,1));
    CHECK(omega_numeric_argmax(b,4,&token));
    CHECK(omega_numeric_argmax(a,0,&token));
    CHECK(omega_numeric_compare(a,b,0,OMEGA_NUMERIC_BITWISE,0,0));
    float x[]={1, -2, FLT_TRUE_MIN}, out[3];
    CHECK(!omega_numeric_scaled_input(x,3));
    omega_fp8_pack_x_f32(x,3,out);
    CHECK(out[0]==0x1p64f && out[1]==-0x1p65f && out[2]==0x1p-85f);
    CHECK(omega_numeric_compare(x,out,3,OMEGA_NUMERIC_BITWISE,0,0)); /* wrong scale */
    float big=FLT_MAX; CHECK(omega_numeric_scaled_input(&big,1));
    big=INFINITY; CHECK(omega_numeric_scaled_input(&big,1));
    uint16_t scales[]={0x3f80}; float folded[1], wrong[1];
    CHECK(!omega_fp8_pack_scales(scales,1,folded,0));
    CHECK(!omega_fp8_pack_scales(scales,1,wrong,1));
    CHECK(omega_numeric_compare(folded,wrong,1,OMEGA_NUMERIC_BITWISE,0,0));
    scales[0]=0x7f80; CHECK(omega_fp8_pack_scales(scales,1,folded,0));
    CHECK(omega_gpu_fp8_e4m3_rne(1.0625f)==0x38); /* halfway, even */
    CHECK(omega_gpu_fp8_e4m3_rne(1.1875f)==0x3a); /* halfway, odd */
    CHECK(omega_gpu_fp8_e4m3_rne(0x1p-10f)==0);
    CHECK(omega_gpu_fp8_e4m3_rne(3*0x1p-10f)==2);
    CHECK(omega_gpu_fp8_e4m3_rne(-0.0f)==0x80);
    CHECK(omega_gpu_fp8_e4m3_rne(1000)==0x7e);
    CHECK(omega_gpu_fp8_e4m3_rne(1.1875f)!=0x39); /* truncation mutant */
    CHECK(isnan(omega_fp8_e4m3_ref(0x7f)) && isnan(omega_fp8_e4m3_ref(0xff)));
    for (unsigned i=0;i<256;i++) if (!omega_fp8_e4m3_is_nan(i))
        CHECK(omega_gpu_fp8_e4m3_rne(omega_fp8_e4m3_ref(i))==i);
    float sum[]={8}, omitted[]={7};
    CHECK(omega_numeric_compare(sum,omitted,1,OMEGA_NUMERIC_BITWISE,0,0));
    printf("numerical_contract_test: %u checks PASS; fixture controls only\n",checks);
    return 0;
}
