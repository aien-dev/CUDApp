/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Host arithmetic and environment checks only. No GPU or native bridge. */
#include "omega_numeric.h"
#include "omega_numeric_reduce.h"
#include <stdio.h>
#if defined(__x86_64__)
#include <fenv.h>
#include <xmmintrin.h>
#endif
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)
int main(void) {
    float a[] = {1.0f, -0.0f, 0.0f, 2.0f};
    float b[] = {2.0f, 0.0f, -0.0f, 1.0f};
    float r[4], expected[4];
    CHECK(omega_numeric_fpenv_ok());
    const OmegaNumericOp ops[] = {OMEGA_NOP_FADD, OMEGA_NOP_FSUB, OMEGA_NOP_FMUL,
        OMEGA_NOP_FFMA, OMEGA_NOP_FMNMX_MIN, OMEGA_NOP_FMNMX_MAX,
        OMEGA_NOP_F2I, OMEGA_NOP_F32_TO_F16, OMEGA_NOP_F32_TO_BF16};
    for (size_t i = 0; i < sizeof ops / sizeof ops[0]; ++i) {
        CHECK(omega_numeric_reference(ops[i], a, b, b, expected, 4) == OMEGA_NUMERIC_OK);
        CHECK(omega_numeric_cpu_realize(ops[i], a, b, b, r, 4) == OMEGA_NUMERIC_OK);
        for (size_t j = 0; j < 4; ++j) CHECK(omega_numeric_bits_equal(r[j], expected[j]));
    }
    float reduced;
    CHECK(omega_reduce_cpu(OMEGA_RED_MAX, a, 4, &reduced) == OMEGA_NUMERIC_OK && reduced == 2.0f);
#if defined(__x86_64__)
    unsigned saved = _mm_getcsr();
    const unsigned bad[] = {1u << 15, 1u << 6, 1u << 13}; /* FTZ, DAZ, rounding */
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; ++i) {
        _mm_setcsr(saved | bad[i]);
        int guarded = !omega_numeric_fpenv_ok();
        int rc = omega_numeric_cpu_realize(OMEGA_NOP_FADD, a, b, NULL, r, 4);
        _mm_setcsr(saved);
        CHECK(guarded && rc == OMEGA_NUMERIC_ERR_FPENV);
    }
    int rounding = fegetround();
    CHECK(fesetround(FE_DOWNWARD) == 0);
    int guarded = !omega_numeric_fpenv_ok();
    CHECK(fesetround(rounding) == 0);
    CHECK(guarded && omega_numeric_fpenv_ok());
#endif
    puts("AMPERE NUMERIC HOST PASS; device NOT_RUN");
    return 0;
}
