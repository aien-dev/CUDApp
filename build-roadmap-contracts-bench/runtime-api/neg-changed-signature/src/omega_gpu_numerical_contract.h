/* SPDX-License-Identifier: MIT */
#ifndef OMEGA_GPU_NUMERICAL_CONTRACT_H
#define OMEGA_GPU_NUMERICAL_CONTRACT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define OMEGA_NUMERIC_CONTRACT_ID "cudapp.numeric.v1"
enum { OMEGA_NUMERIC_BITWISE=0, OMEGA_NUMERIC_REPEATABLE=1, OMEGA_NUMERIC_EXPERIMENTAL=2 };
const char *omega_numeric_contract_id(void);
/* 0 match, -1 invalid/unsupported, 1 mismatch. Finite nonempty arrays only.
 * Experimental budget: |a-b| <= atol + rtol*|a|. Reference is a.
 * This function does not establish model quality or authorize default changes. */
int omega_numeric_compare(const float *a, const float *b, size_t n, int mode, double atol, double rtol);
/* Lowest index of maximum. Finite input required, signed zeros tie. */
int omega_numeric_argmax(const float *a, size_t n, uint32_t *token);
/* Admission for the existing f32 x*2^64 staging path. No mutation. */
int omega_numeric_scaled_input(const float *a, size_t n);
#ifdef __cplusplus
}
#endif
#endif
