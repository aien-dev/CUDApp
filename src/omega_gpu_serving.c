/* omega_gpu_serving.c: see omega_gpu_serving.h. Thin wrapper over the per-API reserve calls. */
#include "omega_gpu_serving.h"
#include "omega_gpu_attention_api.h"
#include "omega_gpu_matmul_api.h"

int omega_gpu_reserve_serving(const OmegaGpuServingBounds *b) {
    if (!b) return OMEGA_GPU_MATMUL_BAD_ARGS;
    int rc = omega_gpu_attention_reserve(b->max_context, b->max_seqs ? b->max_seqs : 1, b->num_q_heads, b->num_kv_heads, b->head_dim, b->kv_block_size);
    if (rc != 0) return rc;
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
