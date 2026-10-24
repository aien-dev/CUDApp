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

/* omega_gpu_serving.c: see omega_gpu_serving.h. Thin wrapper over the per-API reserve calls. */
#include "omega_gpu_serving.h"
#include "omega_gpu_attention_api.h"
#include "omega_gpu_matmul_api.h"

int omega_gpu_reserve_serving(const OmegaGpuServingBounds *b) {
    if (!b) return OMEGA_GPU_MATMUL_BAD_ARGS;
    int rc = omega_gpu_attention_reserve(b->max_context, b->max_seqs ? b->max_seqs : 1, b->num_q_heads, b->num_kv_heads, b->head_dim, b->kv_block_size);
    if (rc != 0) return rc;
    if (b->max_rows > 1) { /* the prompt read's rows path (omega_gpu_gqa_attention_rows_f32): q/out/table for a chunk of rows */
        rc = omega_gpu_attention_reserve_rows(b->max_rows, b->num_q_heads, b->head_dim);
        if (rc != 0) { omega_gpu_attention_unreserve(); return rc; }
    }
    rc = omega_gpu_matmul_reserve(b->max_rows, b->max_k, b->max_n, b->max_n_one_row ? b->max_n_one_row : b->max_n, b->kernel_slots);
    if (rc != 0) omega_gpu_attention_unreserve(); /* the matmul half failed: do not leave half a reservation active */
    return rc;
}

void omega_gpu_serving_seal(void) { omega_gpu_matmul_seal(1); }

void omega_gpu_serving_release(void) {
    omega_gpu_matmul_seal(0);
    omega_gpu_attention_unreserve();
    omega_gpu_matmul_unreserve();
}
