/* SPDX-License-Identifier: MIT */
#include "omega_gpu_numerical_contract.h"
#include <math.h>
#include <string.h>
const char *omega_numeric_contract_id(void) { return OMEGA_NUMERIC_CONTRACT_ID; }
int omega_numeric_compare(const float *a, const float *b, size_t n, int mode, double atol, double rtol) {
    if (!a || !b || !n || mode<0 || mode>2 || !isfinite(atol) || !isfinite(rtol) || atol<0 || rtol<0 ||
        (mode!=OMEGA_NUMERIC_EXPERIMENTAL && (atol!=0 || rtol!=0))) return -1;
    int mismatch=0;
    for (size_t i=0;i<n;i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) return -1;
        if (mode==OMEGA_NUMERIC_EXPERIMENTAL) {
            double budget=atol+rtol*fabs((double)a[i]);
            if (!isfinite(budget)) return -1;
            if (fabs((double)a[i]-b[i])>budget) mismatch=1;
        } else if (memcmp(a+i,b+i,sizeof(float))) mismatch=1;
    }
    return mismatch;
}
int omega_numeric_argmax(const float *a, size_t n, uint32_t *token) {
    if (!a || !token || !n || n>UINT32_MAX) return -1;
    uint32_t best=0;
    for (size_t i=0;i<n;i++) {
        if (!isfinite(a[i])) return -1;
        if (a[i]>a[best]) best=(uint32_t)i;
    }
    *token=best; return 0;
}
int omega_numeric_scaled_input(const float *a, size_t n) {
    if (!a || !n) return -1;
    for (size_t i=0;i<n;i++) if (!isfinite(a[i]) || !isfinite(a[i]*0x1p64f)) return -1;
    return 0;
}
