/*
 * omega_gpu_serving.h: OPT-IN up-front reservation of the GB10 serving buffers (sovereign-core design
 * note docs/design/gb10-weight-ownership.md, sc#277).
 *
 * Without this call nothing changes: the attention staging pool, the matmul activation and result
 * buffers and the matmul kernel cache grow on demand, which on the GB10 means one driver allocation
 * per new context length and per row-count growth (omega PR #332 measured 458 after warm-up over 428
 * tokens). With it, a daemon calls omega_gpu_reserve_serving once after loading weights, and every
 * later call inside the declared bounds reuses the same buffers. A call outside the bounds is refused
 * with TOO_LARGE and a clear omega_gpu_*_last_error() text ("serving reservation exceeded: need N
 * bytes, reserved M"); it never asks the driver for more memory, so it cannot fault the session.
 *
 * Not covered (documented follow-ups): the elementwise API's scratch (it stops growing after the
 * first full-size call), the attention kernel cache (8 slots, one kernel per model shape), and the
 * first build of each distinct matmul shape unless the server prepares every shape it will run
 * (omega_gpu_matmul_prepare: one small code buffer per shape, pinned) and then seals
 * (omega_gpu_serving_seal): after that an unprepared shape is refused, never built.
 */
#ifndef OMEGA_GPU_SERVING_H
#define OMEGA_GPU_SERVING_H

#include <stdint.h>

typedef struct {
    uint32_t max_context;    /* longest sequence (tokens) one attention call may cover */
    uint32_t max_seqs;       /* sequences per attention call; 0 = 1 */
    uint32_t num_q_heads, num_kv_heads, head_dim; /* the model's attention shape */
    uint32_t kv_block_size;  /* paged KV block size in tokens (power of two); 0 = the paged bf16 path is not used */
    uint32_t max_rows;       /* most rows one matmul call may carry (prefill chunk) */
    uint32_t max_k;          /* widest weight input dimension */
    uint32_t max_n;          /* widest weight output dimension for calls of up to max_rows rows */
    uint32_t max_n_one_row;  /* widest output dimension for calls of at most 16 rows (lm_head); 0 = max_n */
    uint32_t kernel_slots;   /* matmul kernel cache slots; 0 = leave at 8; at most 128. Qwen3-4B, rows 1..256: 95 distinct shapes (grid_x varies with rows) */
} OmegaGpuServingBounds;

/* Opens the device if needed. 0 on success, else the first failing OMEGA_GPU_MATMUL_* / ATTN_* code
 * (omega_gpu_matmul_last_error() has the text). All-or-nothing: each API rolls back its own buffers when one of its
 * allocations fails, and this call rolls back the attention half when the matmul half fails, so a failed reserve
 * leaves nothing flagged. Notes: a refused paged/batched attention call may follow earlier launch chunks that already
 * wrote their outputs (TOO_LARGE = output incomplete); max_seqs is checked per launch chunk, not per call. */
int omega_gpu_reserve_serving(const OmegaGpuServingBounds *b);

/* After the server prepared its matmul shapes: refuse any unprepared one from now on (omega_gpu_matmul_seal). */
void omega_gpu_serving_seal(void);
/* Lift the reservation and the seal: buffers stay, growth is allowed again. (Closing the device also ends both.) */
void omega_gpu_serving_release(void);

#endif
