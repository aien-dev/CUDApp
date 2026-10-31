/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Production backend on a non-AArch64 host: refuse before any driver call.
 * Compiled without OMEGA_BLACKWELL_HOST_SIMULATOR. The fake driver is linked
 * so a call that reached m16_native_open would be counted, not sent to a device.
 */
#include "fake_m16_native.h"
#include "omega_blackwell_engine.h"
#include "omega_blackwell_submit.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#if !defined(__aarch64__)
static int refuse_ok(const char *what, int rc)
{
    if (rc == 0 || errno != ENOTSUP) {
        fprintf(stderr, "%s: expected ENOTSUP, rc=%d errno=%d\n", what, rc, errno);
        return 1;
    }
    return 0;
}
#endif

int main(void)
{
#if defined(__aarch64__)
    puts("x86-64 production refusal: NOT_RUN (this binary is AArch64; production barriers stay dsb sy; device open is not attempted)");
    return 0;
#else
    const OmegaGpuBackend *b = omega_blackwell_engine_backend();
    OmegaGpuBackendDiag d;
    int failed = 0;

    fake_reset();
    errno = 0;
    int rc = b->open_device(b->ctx);
    memset(&d, 0, sizeof d);
    b->diagnostics(b->ctx, &d);
    if (rc == 0 || errno != ENOTSUP || d.drv_rc == 0 ||
        strstr(d.drv_text, "requires AArch64 Linux") == NULL) {
        fprintf(stderr, "architecture guard failed: rc=%d errno=%d drv_rc=%d text=%s\n",
                rc, errno, d.drv_rc, d.drv_text);
        failed = 1;
    }
    if (fk.open_calls != 0 || fk.alloc_calls != 0 || fk.submit_calls != 0) {
        fprintf(stderr, "architecture guard entered the driver: open=%d alloc=%d submit=%d\n",
                fk.open_calls, fk.alloc_calls, fk.submit_calls);
        failed = 1;
    }
    if (!failed) puts("Blackwell backend refusal: ENOTSUP, requires AArch64 Linux, driver not entered");

    errno = 0;
    failed |= refuse_ok("direct vector path",
                        omega_blackwell_execute_vector(NULL, NULL, NULL, NULL, NULL, NULL));
    errno = 0;
    failed |= refuse_ok("direct matmul path",
                        omega_blackwell_execute_matmul(NULL, NULL, NULL, NULL, NULL, NULL));
    errno = 0;
    failed |= refuse_ok("direct tensor matmul path",
                        omega_blackwell_execute_matmul_tensor(NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
    if (fk.open_calls != 0) {
        fprintf(stderr, "direct paths entered the driver: open=%d\n", fk.open_calls);
        failed = 1;
    }
    if (!failed) puts("direct Blackwell entry points: ENOTSUP");
    return failed ? 1 : 0;
#endif
}
