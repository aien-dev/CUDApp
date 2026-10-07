/* GPU code buffer sizing (src/omega_gpu_code_alloc.h): code + 2 KB instruction prefetch tail, page rounded.
 * Host only. The NO_TAIL mutant (the page-rounded size used before omega#323) must fail. */
#include "omega_gpu_code_alloc.h"
#include <stdint.h>
#include <stdio.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(void) {
    /* the two attention kernels of omega#323: 2560 B (head_dim 64) and 2688 B (head_dim 128) */
    CHECK(omega_gpu_code_alloc_bytes(2560) == 8192, "2560 B of code -> 8192 (got %zu)", omega_gpu_code_alloc_bytes(2560));
    CHECK(omega_gpu_code_alloc_bytes(2688) == 8192, "2688 B of code -> 8192 (got %zu)", omega_gpu_code_alloc_bytes(2688));
    CHECK(omega_gpu_code_alloc_bytes(2048) == 4096, "2048 B of code fits one page with the tail (got %zu)", omega_gpu_code_alloc_bytes(2048));
    CHECK(omega_gpu_code_alloc_bytes(2049) == 8192, "2049 B of code needs a second page (got %zu)", omega_gpu_code_alloc_bytes(2049));
    CHECK(omega_gpu_code_alloc_bytes(0) == 4096, "empty code -> one page (got %zu)", omega_gpu_code_alloc_bytes(0));
    CHECK(omega_gpu_code_alloc_bytes(0x4000) == 0x5000, "16 KB capacity -> 20 KB (got %zu)", omega_gpu_code_alloc_bytes(0x4000));
    CHECK(omega_gpu_code_alloc_bytes(SIZE_MAX) == 0, "overflow refused");
    CHECK(omega_gpu_code_alloc_bytes(SIZE_MAX - 2048) == 0, "overflow near the limit refused");
    /* every size: at least 2 KB mapped after the code, page multiple, no more than one extra page */
    for (size_t n = 1; n <= 64 * 1024; n += 16) {
        size_t b = omega_gpu_code_alloc_bytes(n);
        if (b < n + OMEGA_GPU_CODE_PREFETCH_TAIL || b % OMEGA_GPU_CODE_PAGE != 0 || b > n + OMEGA_GPU_CODE_PREFETCH_TAIL + OMEGA_GPU_CODE_PAGE) {
            CHECK(0, "%zu B of code -> %zu B (needs >= code + 2048, page multiple)", n, b);
            break;
        }
    }
    if (fails) { printf("OMEGA_GPU_CODE_ALLOC FAIL (%d)\n", fails); return 1; }
    printf("OMEGA_GPU_CODE_ALLOC PASS\n");
    return 0;
}
