/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* out_sha.h -- print "<label> output sha256 <hex>" for a chip output buffer (D7, same inputs, same bits).
 * tools/d7_same_bits_check.sh compares these lines across separate runs of the same binary. */
#ifndef CUDAPP_TESTS_OUT_SHA_H
#define CUDAPP_TESTS_OUT_SHA_H
#include "sha256.h"
#include <stdio.h>

static inline void out_sha_print(const char *label, const sha256_ctx *done) {
    sha256_ctx c = *done; uint8_t d[SHA256_DIGEST_SIZE];
    sha256_final(&c, d);
    printf("%s output sha256 ", label);
    for (int i = 0; i < SHA256_DIGEST_SIZE; i++) printf("%02x", d[i]);
    printf("\n");
}
static inline void out_sha_buf(const char *label, const void *data, size_t len) {
    sha256_ctx c; sha256_init(&c); sha256_update(&c, (const uint8_t *)data, len); out_sha_print(label, &c);
}
#endif
