/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* test_omega_blackwell_matmul_bounds: the refusals at the top of omega_blackwell_execute_matmul and
 * omega_blackwell_execute_matmul_tensor (src/omega_blackwell_submit.c), on the simulated driver
 * (tests/fake_m16_native.c). Host program, opens no device.
 *
 * CudaAuditFable REPORT.md RtMem F3: both wrappers take a spec the caller may fill without omega_matmul_spec_init.
 * m = k = n = 2^31 made (size_t)m * k * 4 wrap to 0, so a 4 KiB buffer was allocated and then written as if it held
 * the whole matrix. Dimensions past OMEGA_BW_MATMUL_MAX_M/K/N are refused before the driver is opened. A code size
 * whose allocation size overflows is refused before the open too; it used to return with the device left open.
 * Every call runs in a child process: before the fix the oversize spec corrupts the heap.
 */
#include "fake_m16_native.h"
#include "omega_blackwell_matmul.h"
#include "omega_blackwell_submit.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int n_pass, n_fail;
static void check(const char *id, int ok) {
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", id);
    if (ok) n_pass++; else n_fail++;
}

enum { I32_OVERSIZE, TENSOR_OVERSIZE, I32_CODE_OVERFLOW, TENSOR_CODE_OVERFLOW, I32_M_ONE_PAST, TENSOR_N_ONE_PAST };

static int refused_before_open(int which) {
    static uint32_t in_a[16], in_b[16], out_c[16];
    static uint32_t code[4];
    OmegaBlackwellKernel kern;
    OmegaMatMulSpec spec;
    OmegaBlackwellMatMulExecution info;
    memset(&kern, 0, sizeof kern);
    memset(&spec, 0, sizeof spec);
    kern.code = (uint8_t *)code;
    kern.code_size = sizeof code;
    spec.m = spec.k = spec.n = 4;
    const int tensor = which == TENSOR_OVERSIZE || which == TENSOR_CODE_OVERFLOW || which == TENSOR_N_ONE_PAST;
    spec.precision = tensor ? OMEGA_MATMUL_PRECISION_FP16 : OMEGA_MATMUL_PRECISION_INT32;
    if (which == I32_OVERSIZE || which == TENSOR_OVERSIZE) spec.m = spec.k = spec.n = 1u << 31;
    if (which == I32_CODE_OVERFLOW || which == TENSOR_CODE_OVERFLOW) kern.code_size = SIZE_MAX;
    if (which == I32_M_ONE_PAST) spec.m = OMEGA_BW_MATMUL_MAX_M + 1u;
    if (which == TENSOR_N_ONE_PAST) spec.n = OMEGA_BW_MATMUL_MAX_N + 1u;
    fake_reset();
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        int rc = tensor ? omega_blackwell_execute_matmul_tensor(&spec, &kern, in_a, in_b, (float *)out_c, &info, NULL, NULL)
                        : omega_blackwell_execute_matmul(&spec, &kern, in_a, in_b, out_c, &info);
        if (rc == 0 || fk.open_calls != 0 || fk.alloc_calls != 0)
            printf("  case %d: rc %d, driver opens %d, allocs %d\n", which, rc, fk.open_calls, fk.alloc_calls);
        fflush(stdout);
        _exit(rc != 0 && fk.open_calls == 0 && fk.alloc_calls == 0 ? 0 : 1);
    }
    int st = 0;
    if (pid < 0 || waitpid(pid, &st, 0) != pid) return 0;
    if (WIFSIGNALED(st)) printf("  case %d: child killed by signal %d\n", which, WTERMSIG(st));
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}

int main(void) {
    check("matmul_i32_2p31_dims_refused_before_open", refused_before_open(I32_OVERSIZE));
    check("matmul_tensor_2p31_dims_refused_before_open", refused_before_open(TENSOR_OVERSIZE));
    check("matmul_i32_m_one_past_max_refused_before_open", refused_before_open(I32_M_ONE_PAST));
    check("matmul_tensor_n_one_past_max_refused_before_open", refused_before_open(TENSOR_N_ONE_PAST));
    check("matmul_i32_code_size_overflow_refused_before_open", refused_before_open(I32_CODE_OVERFLOW));
    check("matmul_tensor_code_size_overflow_refused_before_open", refused_before_open(TENSOR_CODE_OVERFLOW));
    printf("SUMMARY passed=%d failed=%d\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
