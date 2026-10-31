/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Same revision and epoch vectors on the freestanding seeded units and on the
 * portable stand-in. With BINDING_PORTABLE, one seeded entry must refuse
 * without writing the result. */
#include "cudapp_native.h"

#include <stdint.h>
#include <stdio.h>

static int expect_u(const char *name, uint32_t st, uint64_t got, uint32_t want_st, uint64_t want)
{
    if (st == want_st && got == want)
        return 0;
    fprintf(stderr, "FAIL %s st=%u got=%llu want_st=%u want=%llu\n", name, st,
            (unsigned long long)got, want_st, (unsigned long long)want);
    return 1;
}

int main(void)
{
    int bad = 0;
    uint64_t r = 42;
    uint32_t st;
    st = omega_native_revision(&r, 0);
    bad |= expect_u("rev0", st, r, OMEGA_OK, 1);
    st = omega_native_revision(&r, 1);
    bad |= expect_u("rev1", st, r, OMEGA_OK, 2);
    st = omega_native_revision(&r, UINT64_MAX - 1);
    bad |= expect_u("revmax-1", st, r, OMEGA_OK, UINT64_MAX);
    st = omega_native_revision(&r, UINT64_MAX);
    bad |= expect_u("revmax", st, r, OMEGA_OK, 0);
    if (omega_native_revision(0, 0) != OMEGA_BAD_ARG) {
        fprintf(stderr, "FAIL revnull\n");
        bad = 1;
    }
    if (omega_native_revision((uint64_t *)(uintptr_t)1, 0) != OMEGA_BAD_ARG) {
        fprintf(stderr, "FAIL revmisaligned\n");
        bad = 1;
    }
    st = omega_native_epoch(&r, 0);
    bad |= expect_u("ep0", st, r, OMEGA_OK, 1);
    st = omega_native_epoch(&r, 3);
    bad |= expect_u("ep3", st, r, OMEGA_OK, 4);
    st = omega_native_epoch(&r, UINT64_MAX - 1);
    bad |= expect_u("epmax-1", st, r, OMEGA_OK, UINT64_MAX);
    st = omega_native_epoch(&r, UINT64_MAX);
    bad |= expect_u("epmax", st, r, OMEGA_OK, UINT64_MAX);
    if (omega_native_epoch(0, 0) != OMEGA_BAD_ARG) {
        fprintf(stderr, "FAIL epnull\n");
        bad = 1;
    }
#ifdef BINDING_PORTABLE
    r = 99;
    if (omega_bw_enc_sm121(&r, 0, 0, 0, 0, 0, 0, 0, 0) != OMEGA_BAD_RT || r != 99) {
        fprintf(stderr, "FAIL encoder refusal wrote a result or returned another status\n");
        bad = 1;
    }
#endif
    if (bad)
        return 1;
    puts("binding-portable: PASS");
    return 0;
}
