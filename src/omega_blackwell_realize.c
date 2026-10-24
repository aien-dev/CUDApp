/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "omega_blackwell_realize.h"
#include "sha256.h"
#include <string.h>

int omega_blackwell_get_machine_id(SemanticId *out_machine_id) {
    if (!out_machine_id) return -1;
    sha256_ctx ctx;
    sha256_init(&ctx);
    const char *machine_str = "NVIDIA_DGX_SPARK_GB10_SM121";
    sha256_update(&ctx, (const uint8_t *)machine_str, strlen(machine_str));
    sha256_final(&ctx, out_machine_id->bytes);
    return 0;
}

int omega_blackwell_realization_compute_id(OmegaBlackwellRealization *real) {
    if (!real) return -1;
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, real->spec_id.bytes, OMEGA_ID_BYTES);
    sha256_update(&ctx, real->machine_id.bytes, OMEGA_ID_BYTES);
    sha256_update(&ctx, real->code_digest, sizeof(real->code_digest));
    sha256_update(&ctx, (const uint8_t *)&real->sm_architecture, sizeof(real->sm_architecture));
    sha256_final(&ctx, real->realization_digest);

    memcpy(real->realization_id.bytes, real->realization_digest, OMEGA_ID_BYTES);
    return 0;
}

int omega_blackwell_realize_vector(const OmegaVectorSpec *spec, OmegaBlackwellRealization *out_real) {
    if (!spec || !out_real) return -1;
    memset(out_real, 0, sizeof(*out_real));

    out_real->spec_id = spec->spec_id;
    if (omega_blackwell_get_machine_id(&out_real->machine_id) != 0) return -1;

    out_real->sm_architecture = OMEGA_BW_SM_ARCH_121;
    out_real->code_len = OMEGA_BW_VECADD_CODE_SIZE;

    /* Verify and digest generated Blackwell machine code */
    uint8_t code_buf[OMEGA_BW_VECADD_CODE_SIZE];
    size_t encoded_len = 0;
    if (omega_blackwell_encode_vecadd(code_buf, sizeof(code_buf), &encoded_len) != 0) return -1;
    if (omega_blackwell_compute_code_digest(code_buf, encoded_len, out_real->code_digest) != 0) return -1;

    /* Default launch geometry */
    out_real->qmd_cfg.num_elements = spec->element_count;
    out_real->qmd_cfg.threads_per_block = 64;
    out_real->qmd_cfg.grid_width = (spec->element_count + 63) / 64;
    if (out_real->qmd_cfg.grid_width == 0) out_real->qmd_cfg.grid_width = 1;

    return omega_blackwell_realization_compute_id(out_real);
}
