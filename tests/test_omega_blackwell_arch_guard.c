/* SPDX-License-Identifier: MIT */
/* Runtime contract for the non-AArch64 production backend refusal. */
#include "omega_blackwell_engine.h"
#include "omega_blackwell_submit.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    const OmegaGpuBackend *b = omega_blackwell_engine_backend();
    OmegaGpuBackendDiag d;
    errno = 0;
    int rc = b->open_device(b->ctx);
    memset(&d, 0, sizeof d);
    b->diagnostics(b->ctx, &d);
    if (rc == 0 || errno != ENOTSUP || d.drv_rc == 0 ||
        strstr(d.drv_text, "requires AArch64 Linux") == NULL) {
        fprintf(stderr, "architecture guard failed: rc=%d errno=%d drv_rc=%d text=%s\n",
                rc, errno, d.drv_rc, d.drv_text);
        return 1;
    }
    puts("Blackwell backend refusal: ENOTSUP, requires AArch64 Linux");
    errno = 0;
    if (omega_blackwell_execute_matmul(NULL, NULL, NULL, NULL, NULL, NULL) == 0 || errno != ENOTSUP) {
        fputs("legacy Blackwell matmul path did not refuse unsupported architecture\n", stderr);
        return 1;
    }
    errno = 0;
    if (omega_blackwell_execute_matmul_tensor(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL) == 0 || errno != ENOTSUP) {
        fputs("tensor Blackwell matmul path did not refuse unsupported architecture\n", stderr);
        return 1;
    }
    puts("direct Blackwell matmul paths: ENOTSUP");
    return 0;
}
