#include "omega_blackwell_matmul.h"
#include "sha256.h"
#include <stdlib.h>
#include <string.h>

int omega_matmul_spec_init(OmegaMatMulSpec *spec, uint32_t m, uint32_t k, uint32_t n, OmegaMatMulPrecision precision) {
    if (!spec || m == 0 || k == 0 || n == 0) return -1;
    if (m > OMEGA_BW_MATMUL_MAX_M || k > OMEGA_BW_MATMUL_MAX_K || n > OMEGA_BW_MATMUL_MAX_N) return -1;

    spec->m = m;
    spec->k = k;
    spec->n = n;
    spec->precision = precision;

    /* Canonical spec_id computed over (m, k, n, precision) */
    uint8_t spec_buf[16];
    spec_buf[0] = (uint8_t)(m & 0xff);
    spec_buf[1] = (uint8_t)((m >> 8) & 0xff);
    spec_buf[2] = (uint8_t)((m >> 16) & 0xff);
    spec_buf[3] = (uint8_t)((m >> 24) & 0xff);

    spec_buf[4] = (uint8_t)(k & 0xff);
    spec_buf[5] = (uint8_t)((k >> 8) & 0xff);
    spec_buf[6] = (uint8_t)((k >> 16) & 0xff);
    spec_buf[7] = (uint8_t)((k >> 24) & 0xff);

    spec_buf[8] = (uint8_t)(n & 0xff);
    spec_buf[9] = (uint8_t)((n >> 8) & 0xff);
    spec_buf[10] = (uint8_t)((n >> 16) & 0xff);
    spec_buf[11] = (uint8_t)((n >> 24) & 0xff);

    spec_buf[12] = (uint8_t)(precision & 0xff);
    spec_buf[13] = 0;
    spec_buf[14] = 0;
    spec_buf[15] = 0;

    sha256_hash(spec_buf, sizeof(spec_buf), spec->spec_id);
    return 0;
}

int omega_matmul_cpu_oracle_i32(const uint32_t *a, const uint32_t *b, uint32_t *c, uint32_t m, uint32_t k, uint32_t n) {
    if (!a || !b || !c || m == 0 || k == 0 || n == 0) return -1;

    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            uint32_t acc = 0;
            for (uint32_t p = 0; p < k; p++) {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    return 0;
}

void omega_blackwell_kernel_free(OmegaBlackwellKernel *kernel) {
    if (!kernel) return;
    if (kernel->code) {
        free(kernel->code);
        kernel->code = NULL;
    }
    kernel->code_size = 0;
    kernel->insn_count = 0;
}

int omega_blackwell_bind_matmul_realization(const OmegaMatMulSpec *spec, const OmegaBlackwellKernel *kernel, OmegaBlackwellRealizationIdentity *id) {
    if (!spec || !kernel || !id) return -1;

    memcpy(id->spec_id, spec->spec_id, 32);
    /* NVIDIA_DGX_SPARK_GB10_SM121 canonical machine descriptor digest */
    const char *machine_str = "NVIDIA_DGX_SPARK_GB10_SM121";
    sha256_hash((const uint8_t *)machine_str, strlen(machine_str), id->machine_id);
    memcpy(id->code_digest, kernel->code_digest, 32);
    id->sm_arch = 121;

    /* Realization ID = SHA-256(spec_id || machine_id || code_digest || sm_arch) */
    uint8_t bound[100];
    memcpy(bound + 0, id->spec_id, 32);
    memcpy(bound + 32, id->machine_id, 32);
    memcpy(bound + 64, id->code_digest, 32);
    bound[96] = (uint8_t)(id->sm_arch & 0xff);
    bound[97] = (uint8_t)((id->sm_arch >> 8) & 0xff);
    bound[98] = (uint8_t)((id->sm_arch >> 16) & 0xff);
    bound[99] = (uint8_t)((id->sm_arch >> 24) & 0xff);

    sha256_hash(bound, sizeof(bound), id->realization_id);
    return 0;
}

#include "omega_blackwell_codegen.h"

int omega_blackwell_codegen_matmul(const OmegaMatMulSpec *spec, OmegaBlackwellKernel *kernel) {
    if (!spec || !kernel) return -1;
    if (spec->precision == OMEGA_MATMUL_PRECISION_INT32) {
        return omega_blackwell_codegen_matmul_i32(spec, kernel);
    }
    return -1;
}
