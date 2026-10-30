/* Link stub for the imported dagsim binary on hosts that are not AArch64.
 * src/omega_om_units.c is AArch64 machine code. rx_costmodel.o references
 * sha256_hash from serialize/digest only. The analytical dagsim tests do not
 * call it. On AArch64 the Makefile links src/omega_om_units.c instead, and
 * this file is not built. dagsim --model is NOT_RUN on this stub: a blob
 * digest would abort rather than return a made-up hash.
 */
#if !defined(__aarch64__)
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>

void sha256_hash(const uint8_t *d, size_t n, uint8_t out[32]) {
    (void)d; (void)n; (void)out;
    fprintf(stderr, "sha256_hash: host stub; dagsim --model is NOT_RUN off AArch64\n");
    abort();
}
#endif
