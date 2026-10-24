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

#ifndef OMEGA_GPU_PREFILL_H
#define OMEGA_GPU_PREFILL_H
/*
 * omega_gpu_prefill (OWN-30B T3b, 2026-10-13): the device-resident one-pass prefill API on the shared session. New file.
 * Every buffer the prefill pipeline passes between its kernels stays on the device; the host builds small tables only.
 *
 * Memory policy (nvos.h:1116-1118, read 2026-10-13: GPU caching "For system memory ... will not be coherent with direct CPU
 * mappings"; clc96f.h:72 L2_FLUSH_DIRTY):
 *  - DEVICE buffers (omega_gpu_pf_alloc_dev) are GPU-cached and device-private: the CPU never reads them and never writes
 *    them; they are filled only by GPU writes (omega_gpu_pf_upload copies through an uncached staging buffer with the copy
 *    kernel), so the GPU L2 never holds a stale line, and GPU producers / consumers share the L2 (the whole point: the
 *    uncached buffers get no cache reuse, evidence OWN30B-T3-TC-PEAK-BW).
 *  - HOST buffers (omega_gpu_pf_alloc_host) are the session's GPU-uncached buffers: tables and inputs the host writes,
 *    results the host reads.
 *  - Reading a device buffer back (tests, logits) goes through omega_gpu_pf_download: a GPU copy into uncached staging.
 * One session hook (on_close) frees every kernel and staging buffer of this module.
 *
 * All functions take the session lock themselves unless named _locked. Return codes OMEGA_GPU_PF_*.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "omega_gpu_session.h"

#ifdef __cplusplus
extern "C" {
#endif
enum { OMEGA_GPU_PF_OK = 0, OMEGA_GPU_PF_BAD_ARGS = -1, OMEGA_GPU_PF_TOO_LARGE = -2, OMEGA_GPU_PF_CODEGEN_FAIL = -3, OMEGA_GPU_PF_CHIP_FAIL = -4 };
const char *omega_gpu_pf_rc_name(int rc);

/* buffers */
int omega_gpu_pf_alloc_dev(size_t bytes, OmegaGpuMem *out);  /* GPU-cached, device-private */
int omega_gpu_pf_alloc_host(size_t bytes, OmegaGpuMem *out); /* GPU-uncached, host reads and writes */
void omega_gpu_pf_free(OmegaGpuMem *m);
/* host bytes -> device buffer at byte offset off (16-byte multiples), through the staging buffer and the copy kernel */
int omega_gpu_pf_upload(const OmegaGpuMem *dst, size_t off, const void *src, size_t bytes);
/* device buffer bytes at off -> host (16-byte multiples) */
int omega_gpu_pf_download(const OmegaGpuMem *src, size_t off, void *dst, size_t bytes);
/* device-to-device copy by VA (16-byte multiples) */
int omega_gpu_pf_copy_va(uint64_t dst_va, uint64_t src_va, size_t bytes);

/* accumulated chip time and launch count of this module since the last reset (prefill timing) */
void omega_gpu_pf_stats(uint64_t *chip_ns, uint64_t *launches);
void omega_gpu_pf_stats_reset(void);

/* ---- internal to the prefill module family (omega_gpu_prefill*.c) ---- */
typedef struct { uint32_t kind; uint32_t words[15]; } OmegaPfKey; /* kernel identity: kind + the generator's spec words */
typedef int (*OmegaPfGen)(const OmegaPfKey *key, void *prog);    /* builds the BlackwellIRProgram for key */
typedef struct { uint64_t code_va; uint32_t gpr_declared, gpr_peak; } OmegaPfKernel;
/* lock held, device open: the resident kernel for key (generated, encoded, patched and uploaded on first use) */
int omega_gpu_pf_kernel_locked(const OmegaPfKey *key, OmegaPfGen gen, OmegaPfKernel *out);
/* lock held: one launch, chip time added to the module stats (and to *ns when given) */
int omega_gpu_pf_launch_locked(const OmegaPfKernel *k, uint32_t threads, uint32_t grid_x, uint32_t grid_y, uint32_t shared_bytes,
                               const uint32_t *args, uint32_t n_args, uint64_t *ns);
int omega_gpu_pf_open_locked(void); /* opens the device, registers the hook: OMEGA_GPU_PF_OK or CHIP_FAIL */
/* lock held: the same launch appended to a command list (omega_gpu_cmdlist.h) instead of submitted (sc #642: a whole Qwen3.8
 * prefill chunk is one list). Prefill kernels are never evicted, so a recorded launch's code stays valid until the submit. */
struct OmegaGpuCmdList;
int omega_gpu_pf_record_locked(struct OmegaGpuCmdList *cl, const OmegaPfKernel *k, uint32_t threads, uint32_t grid_x, uint32_t grid_y,
                               uint32_t shared_bytes, const uint32_t *args, uint32_t n_args);

/* ---- GEMM v2 (omega_gpu_pfgemm.c, kernels omega_blackwell_codegen_tc2.h) ----
 * Weights live in device buffers in the kernel's PHYSICAL row order with the per-8-row ratio table (tc2 header).
 * Layout PLAIN: rows unchanged. Layout SWIGLU: w is [E][2I][k] with the I gate rows first then the I up rows (scales
 * [E][2I/128][k/128]); physical row p = 16G + h holds gate (h < 8) or up (h >= 8) of intermediate index
 * i = perm(8G + h % 8), perm = the (0,2,1,3) group permutation, so the SwiGLU epilogue writes act rows in stored order. */
#ifndef OMEGA_GPU_PF_LAYOUTS
#define OMEGA_GPU_PF_LAYOUTS
enum { OMEGA_GPU_PF_PLAIN = 0, OMEGA_GPU_PF_SWIGLU = 1, OMEGA_GPU_PF_SWIGLU_NAT = 2 };
#endif
typedef struct { OmegaGpuMem w, r; uint32_t experts, n, k, layout; } OmegaPfWeights;
/* w: experts * n * k e4m3 bytes; scale_inv: experts * (n/128) * (k/128) bf16 (checkpoint weight_scale_inv) */
int omega_gpu_pf_weights_upload(uint32_t experts, uint32_t n, uint32_t k, const uint8_t *w, const uint16_t *scale_inv, uint32_t layout, OmegaPfWeights *out);
void omega_gpu_pf_weights_free(OmegaPfWeights *w);
/* the host-side physical form (tests and emulation): pw experts * n * k, ratio experts * (n/8) * (k/128 + 1) f32 */
int omega_gpu_pf_weights_host_form(uint32_t experts, uint32_t n, uint32_t k, const uint8_t *w, const uint16_t *scale_inv, uint32_t layout, uint8_t *pw, float *ratio);
uint32_t omega_gpu_pf_gemm_bm(void); /* rows per tile */
/* tile table (32-byte entries) for rows sorted by expert: counts[e] rows of expert e; returns the tile count (<= max) or 0 */
uint32_t omega_gpu_pf_gemm_tiles(const OmegaPfWeights *w, const uint32_t *counts, void *tab, uint32_t max);
/* one launch. rowidx_va 0 = direct rows (row r reads x row r). x: bf16 stored rows of k (rows to row0+bm-1 readable);
 * y: f32 rows of n (PLAIN) or bf16 act rows of n/2 (SWIGLU). */
int omega_gpu_pf_gemm(const OmegaPfWeights *w, uint64_t tiles_va, uint32_t ntiles, uint64_t rowidx_va, uint64_t x_va, uint64_t y_va, uint64_t *ns);
/* measurement knobs (0 restores): rt, warps, aqn; test-only mutant */
void omega_gpu_pf_gemm_set_shape(uint32_t rt, uint32_t warps, uint32_t aqn);
void omega_gpu_pf_gemm_test_set_mutant(int m);
/* ---- native FP8 path (W-L1G) ----
 * activation quantiser: x (f32 or bf16 rows of k, in_type OMEGA_QG_IN_*) -> q (e4m3 rows of k) + s (f32 [rows][k/128] power-of-two
 * scales); semantics in omega_blackwell_codegen_qg.h */
int omega_gpu_pf_quant(uint32_t k, uint32_t in_type, uint32_t rows, uint64_t x_va, uint64_t q_va, uint64_t s_va, uint64_t *ns);
void omega_gpu_pf_quant_test_set_mutant(int m);
/* grouped GEMM on QMMA, weights read in place from an FP8 GEMV tensor (omega_gpu_fp8_api.h, checkpoint row order, no copy).
 * layout PLAIN: y f32 rows of n. layout SWIGLU_NAT: the tensor is [E][2I][k] gate rows then up rows; y = bf16 act rows of I = n/2,
 * act[i] = silu(gate_i) * up_i (capped exponent, omega_blackwell_codegen_qg.h). xq: e4m3 rows of k, xs: f32 [rows][k/128]
 * power-of-two scales (omega_gpu_pf_quant). Rows grouped by expert: counts[e] rows of expert e, tile table from qgemm_tiles. */
typedef struct OmegaGpuFp8Tensor OmegaGpuFp8Tensor;
uint32_t omega_gpu_pf_qgemm_bm(void);
uint32_t omega_gpu_pf_qgemm_tiles(const OmegaGpuFp8Tensor *t, const uint32_t *counts, void *tab, uint32_t max);
/* the same from per-expert base addresses of a GEMV-layout copy (e.g. a GPU-cached device buffer) */
uint32_t omega_gpu_pf_qgemm_tiles_va(uint32_t experts, const uint64_t *w_va, const uint64_t *s_va, const uint32_t *counts, void *tab, uint32_t max);
int omega_gpu_pf_qgemm(uint32_t n, uint32_t k, uint32_t layout, uint64_t tiles_va, uint32_t ntiles, uint64_t rowidx_va, uint64_t xq_va, uint64_t xs_va, uint64_t y_va, uint64_t *ns);
/* Record paths (sc #642): the same launches appended to cl; the caller holds omega_gpu_session_lock(). A first use compiles the
 * kernel there (prefill kernels are never evicted). */
struct OmegaGpuCmdList;
int omega_gpu_pf_quant_record(struct OmegaGpuCmdList *cl, uint32_t k, uint32_t in_type, uint32_t rows, uint64_t x_va, uint64_t q_va, uint64_t s_va);
int omega_gpu_pf_qgemm_record(struct OmegaGpuCmdList *cl, uint32_t n, uint32_t k, uint32_t layout, uint64_t tiles_va, uint32_t ntiles,
                              uint64_t rowidx_va, uint64_t xq_va, uint64_t xs_va, uint64_t y_va);
void omega_gpu_pf_qgemm_set_shape(uint32_t rt, uint32_t warps, uint32_t wait); /* knobs (all 0 restores rt 4, warps 8, wait 45) */
void omega_gpu_pf_qgemm_test_set_mutant(int m);
#ifdef __cplusplus
}
#endif
#endif
