#ifndef OMEGA_GPU_PFHOST_H
#define OMEGA_GPU_PFHOST_H
/*
 * omega_gpu_pfhost (OWN-30B W-L1G-D, 2026-10-14): the whole-prompt prefill's router logits and MoE combine on the chip. New file.
 * Kernels and their exact semantics: omega_blackwell_codegen_pfhost.h (bit-identical to the Rust driver's host loops).
 * Buffers are omega_gpu_prefill.h buffers (DEVICE or HOST, by VA); every call takes the session lock and is synchronous.
 *
 *   uint32_t omega_gpu_pf_router_rows(void);
 *        Rows per CTA of the router-logits launch (t). x must have ceil(rows / t) * t readable rows; logits gets that many rows.
 *   size_t omega_gpu_pf_router_packed_bytes(uint32_t e, uint32_t k);
 *   void omega_gpu_pf_router_pack(uint32_t e, uint32_t k, const float *w, float *wq);
 *        w: f32 [e][k] (row ex = expert ex); wq: the packed layout R4 plus the pad group (zeros), packed_bytes bytes.
 *   int omega_gpu_pf_router_logits(uint32_t k, uint32_t e, uint32_t rows, uint64_t x_va, uint64_t wq_va, uint64_t logits_va, uint64_t *ns);
 *   int omega_gpu_pf_moe_combine(uint32_t h, uint32_t top_k, uint32_t rows, uint64_t down_va, uint64_t inv_va, uint64_t wt_va,
 *                                uint64_t out_va, uint64_t *ns);
 * Return codes OMEGA_GPU_PF_*; *ns (when given) gains the chip nanoseconds.
 */
#include <stddef.h>
#include <stdint.h>
#include "omega_gpu_prefill.h"

#ifdef __cplusplus
extern "C" {
#endif
uint32_t omega_gpu_pf_router_rows(void);
size_t omega_gpu_pf_router_packed_bytes(uint32_t e, uint32_t k);
void omega_gpu_pf_router_pack(uint32_t e, uint32_t k, const float *w, float *wq);
int omega_gpu_pf_router_logits(uint32_t k, uint32_t e, uint32_t rows, uint64_t x_va, uint64_t wq_va, uint64_t logits_va, uint64_t *ns);
int omega_gpu_pf_moe_combine(uint32_t h, uint32_t top_k, uint32_t rows, uint64_t down_va, uint64_t inv_va, uint64_t wt_va, uint64_t out_va, uint64_t *ns);
/* tests only: kernel mutants (0 restores) and the router's rows per CTA (0 = default 16) */
void omega_gpu_pf_router_test_set_mutant(int m);
void omega_gpu_pf_combine_test_set_mutant(int m);
void omega_gpu_pf_router_set_rows(uint32_t t);
#ifdef __cplusplus
}
#endif
#endif
