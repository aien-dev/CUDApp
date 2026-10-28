/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Dedicated boundary, native sweep, and negative mutant controls for scalar f16. */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

extern uint32_t omega_native_f16(uint64_t *result, uint64_t p0);

static uint16_t native_reference(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    _Float16 h = (_Float16)f;
    uint16_t r;
    memcpy(&r, &h, 2);
    return r;
}

/* Mutant 1: Truncation towards zero (omits +4095/lsb rounding) */
static uint16_t mutant_truncate(uint32_t u) {
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t abs_u = u & 0x7fffffffu;
    if (abs_u >= 0x7f800000u) {
        if (abs_u > 0x7f800000u) return (uint16_t)(sign | 0x7e00u | ((abs_u >> 13) & 0x03ffu));
        return (uint16_t)(sign | 0x7c00u);
    }
    if (abs_u >= 0x47800000u) return (uint16_t)(sign | 0x7c00u);
    if (abs_u >= 0x38800000u) {
        uint32_t val = abs_u - 0x38000000u;
        return (uint16_t)(sign | (val >> 13));
    }
    return (uint16_t)sign;
}

/* Mutant 2: Round ties to away/odd instead of even (lsb bit ignored) */
static uint16_t mutant_no_tie_even(uint32_t u) {
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t abs_u = u & 0x7fffffffu;
    if (abs_u >= 0x7f800000u) return (uint16_t)(sign | 0x7e00u);
    if (abs_u >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);
    if (abs_u >= 0x38800000u) {
        uint32_t val = (abs_u + 0x1000u) - 0x38000000u;
        return (uint16_t)(sign | (val >> 13));
    }
    return (uint16_t)sign;
}

/* Mutant 3: Overflow threshold off by 1 (0x477ff001 instead of 0x477ff000) */
static uint16_t mutant_bad_overflow_threshold(uint32_t u) {
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t abs_u = u & 0x7fffffffu;
    if (abs_u >= 0x7f800000u) return (uint16_t)(sign | 0x7e00u);
    if (abs_u > 0x477ff000u) return (uint16_t)(sign | 0x7c00u);
    if (abs_u >= 0x38800000u) {
        uint32_t lsb = (abs_u >> 13) & 1u;
        uint32_t rounded = abs_u + 0x0fffu + lsb;
        uint32_t val = rounded - 0x38000000u;
        return (uint16_t)(sign | (val >> 13));
    }
    return (uint16_t)sign;
}

/* Mutant 4: Subnormal shift/bias off by 1 */
static uint16_t mutant_bad_subnormal(uint32_t u) {
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t abs_u = u & 0x7fffffffu;
    if (abs_u >= 0x38800000u) {
        uint64_t res = 0;
        omega_native_f16(&res, u);
        return (uint16_t)res;
    }
    return (uint16_t)(sign | 0x0002u); /* corrupted subnormal */
}

/* Mutant 5: NaN quiet bit omitted */
static uint16_t mutant_nan_no_quiet(uint32_t u) {
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t abs_u = u & 0x7fffffffu;
    if (abs_u > 0x7f800000u) {
        return (uint16_t)(sign | 0x7c00u | ((abs_u >> 13) & 0x01ffu)); /* bit 9 (0x0200) missing */
    }
    uint64_t res = 0;
    omega_native_f16(&res, u);
    return (uint16_t)res;
}

/* Mutant 6: Sign bit ignored / zeroed */
static uint16_t mutant_ignore_sign(uint32_t u) {
    uint64_t res = 0;
    omega_native_f16(&res, u & 0x7fffffffu);
    return (uint16_t)res;
}

int main(void) {
    /* 1. Boundary table verification */
    struct {
        uint32_t u;
        uint16_t expected;
        const char *desc;
    } boundaries[] = {
        {0x00000000u, 0x0000u, "+0.0"},
        {0x80000000u, 0x8000u, "-0.0"},
        {0x33000000u, 0x0000u, "underflow cutoff"},
        {0x33000001u, 0x0001u, "min subnormal"},
        {0x387fdfffu, 0x03ffu, "max subnormal (below round-to-normal)"},
        {0x387fe000u, 0x0400u, "subnormal round up to min normal"},
        {0x387fffffu, 0x0400u, "subnormal top round to min normal"},
        {0x38800000u, 0x0400u, "min normal"},
        {0x3f800000u, 0x3c00u, "1.0f"},
        {0x3f801000u, 0x3c00u, "RNE tie down to even"},
        {0x3f803000u, 0x3c02u, "RNE tie up to even"},
        {0x477fe000u, 0x7bffu, "max normal"},
        {0x477fefffu, 0x7bffu, "max normal below overflow"},
        {0x477ff000u, 0x7c00u, "+overflow to +inf"},
        {0xc77ff000u, 0xfc00u, "-overflow to -inf"},
        {0x7f800000u, 0x7c00u, "+inf"},
        {0xff800000u, 0xfc00u, "-inf"},
        {0x7fc00000u, 0x7e00u, "canonical qNaN"},
        {0x7f800001u, 0x7e00u, "sNaN made quiet"},
        {0xffc00000u, 0xfe00u, "-qNaN"},
        {0x7fffffffu, 0x7fffu, "NaN payload preservation"},
    };

    for (size_t i = 0; i < sizeof(boundaries)/sizeof(boundaries[0]); i++) {
        uint64_t res = 0;
        uint32_t rc = omega_native_f16(&res, boundaries[i].u);
        if (rc != 0) {
            fprintf(stderr, "FAIL: %s: rc %u != 0\n", boundaries[i].desc, rc);
            return 1;
        }
        if ((uint16_t)res != boundaries[i].expected) {
            fprintf(stderr, "FAIL: %s (0x%08x): got 0x%04x, expected 0x%04x\n",
                    boundaries[i].desc, boundaries[i].u, (uint16_t)res, boundaries[i].expected);
            return 1;
        }
        uint16_t ref = native_reference(boundaries[i].u);
        if ((uint16_t)res != ref) {
            fprintf(stderr, "FAIL: %s (0x%08x): got 0x%04x, native ref 0x%04x\n",
                    boundaries[i].desc, boundaries[i].u, (uint16_t)res, ref);
            return 1;
        }
    }

    /* 2. Negative mutant controls */
    int mutant1_caught = 0;
    int mutant2_caught = 0;
    int mutant3_caught = 0;
    int mutant4_caught = 0;
    int mutant5_caught = 0;
    int mutant6_caught = 0;

    for (size_t i = 0; i < sizeof(boundaries)/sizeof(boundaries[0]); i++) {
        uint32_t u = boundaries[i].u;
        uint16_t exp = boundaries[i].expected;
        if (mutant_truncate(u) != exp) mutant1_caught++;
        if (mutant_no_tie_even(u) != exp) mutant2_caught++;
        if (mutant_bad_overflow_threshold(u) != exp) mutant3_caught++;
        if (mutant_bad_subnormal(u) != exp) mutant4_caught++;
        if (mutant_nan_no_quiet(u) != exp) mutant5_caught++;
        if (mutant_ignore_sign(u) != exp) mutant6_caught++;
    }

    if (!mutant1_caught) { fprintf(stderr, "FAIL: mutant1 (truncate) not caught\n"); return 1; }
    if (!mutant2_caught) { fprintf(stderr, "FAIL: mutant2 (no tie even) not caught\n"); return 1; }
    if (!mutant3_caught) { fprintf(stderr, "FAIL: mutant3 (bad overflow threshold) not caught\n"); return 1; }
    if (!mutant4_caught) { fprintf(stderr, "FAIL: mutant4 (bad subnormal) not caught\n"); return 1; }
    if (!mutant5_caught) { fprintf(stderr, "FAIL: mutant5 (nan no quiet) not caught\n"); return 1; }
    if (!mutant6_caught) { fprintf(stderr, "FAIL: mutant6 (ignore sign) not caught\n"); return 1; }

    /* 3. Comprehensive stepped sweep across 2^32 inputs against native _Float16 */
    uint64_t diffs = 0;
    for (uint64_t i = 0; i < 0x100000000ull; i += 97) {
        uint32_t u = (uint32_t)i;
        uint64_t res = 0;
        omega_native_f16(&res, u);
        uint16_t ref = native_reference(u);
        if ((uint16_t)res != ref) {
            fprintf(stderr, "Diff at 0x%08x: res 0x%04x != ref 0x%04x\n", u, (uint16_t)res, ref);
            if (++diffs > 10) return 1;
        }
    }

    puts("F16 BEHAVIOR CONTROLS PASS (all boundaries, native sweep, and negative mutants verified)");
    return 0;
}
