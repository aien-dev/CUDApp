/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_BLACKWELL_QMD_H
#define OMEGA_BLACKWELL_QMD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define OMEGA_BW_QMD_WORDS              96
#define OMEGA_BW_QMD_BYTES              (OMEGA_BW_QMD_WORDS * 4) /* 384 bytes */
#define OMEGA_BW_CBANK_DRIVER_WORDS     224
#define OMEGA_BW_CBANK_ARGS_WORDS       7
#define OMEGA_BW_CBANK_MATMUL_ARGS_WORDS 10

/* Epistemic classifications for QMD fields */
typedef enum {
    EPISTEMIC_DOCUMENTED = 1,        /* NVIDIA open-gpu-kernel-modules / clcdc0qmd.h */
    EPISTEMIC_OBSERVED = 2,          /* Empirically verified on physical GB10 silicon */
    EPISTEMIC_REVERSE_ENGINEERED = 3 /* Derived from Blackwell instruction trace */
} OmegaEpistemicTier;

typedef struct {
    uint64_t code_va;
    uint64_t cbank_va;
    uint64_t scratch_va;
    uint64_t sem_va;
    uint64_t qmd0_va;
    uint64_t qmd1_va;
    uint32_t num_elements;
    uint32_t threads_per_block;
    uint32_t grid_width;
    uint32_t threads_x;
    uint32_t threads_y;
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t gpr_count;
    /* Shared memory per CTA in bytes (clcec0qmd.h SHARED_MEMORY_SIZE_SHIFTED7, 128-byte units).
     * 0 keeps the long-standing 1024. At most OMEGA_BW_QMD_MAX_SHARED_BYTES. */
    uint32_t shared_bytes;
} OmegaBlackwellQmdConfig;
/* The field is NVCEC0_QMDV05_00_SHARED_MEMORY_SIZE_SHIFTED7 = MW(1162:1152) (clcec0qmd.h in
 * NVIDIA's open-gpu-doc tree): 11 bits in units of 128 bytes, so it can name up to 2047 * 128 =
 * 262016 bytes, and the GB10 SM allows up to 227 KiB of shared memory per CTA.
 *
 * This cap is not the field: it is the TARGET_SM_CONFIG_SHARED_MEM_SIZE word 36 already declares
 * (value 9, encoded as size_kb / 4 + 1 by the rule in Mesa NVK's nak/qmd.rs
 * gv100_smem_size_to_hw, i.e. 32 KiB), with MAX at 100 KiB. Launching more than the target asks
 * the SM for a larger carveout than this QMD asks for, which is the case NVK answers by also
 * raising MIN_ (one workgroup) and TARGET_ (max workgroups) with gv100_pick_smem_size_kb. Do that
 * before raising this number (and keep the negative control in test-gpu-qmd-shared).
 *
 * It was 16384: the tensor-core FP8 grouped GEMM (omega_tc2_shared_bytes, 4096 * rt bytes) asks
 * 32 KiB at the production shape rt 8, so omega_gpu_pf_gemm() could not launch at all and
 * gpu_pfgemm_test --chip failed on every shape at "qmd build (shared_bytes)" (2026-10-25). */
#define OMEGA_BW_QMD_MAX_SHARED_BYTES 32768u

/* Build QMD 0: GRID_NULL null-barrier grid with dependence counter 2 */
int omega_blackwell_build_qmd0(uint32_t *qmd0_words, uint64_t qmd0_va, uint64_t qmd1_va);

/* Build QMD 1: GRID_CTA compute grid for Blackwell sm_121 vector add and matmul */
int omega_blackwell_build_qmd1(uint32_t *qmd1_words, const OmegaBlackwellQmdConfig *cfg);

/* Build driver constant bank 0 (224 words) with hardware descriptor addresses */
int omega_blackwell_build_cbank_driver(uint32_t *cbank_words, uint64_t cbank_va);

/* Build driver constant bank 0 (224 words) with 2D tile geometry */
int omega_blackwell_build_cbank_driver_2d(uint32_t *cbank_words, uint64_t cbank_va,
                                         uint32_t threads_x, uint32_t threads_y,
                                         uint32_t grid_x, uint32_t grid_y);

/* Build kernel argument constant bank buffer (7 words: pointers A, B, C, length N) */
int omega_blackwell_build_cbank_args(uint32_t *args_words, uint64_t a_va, uint64_t b_va,
                                    uint64_t c_va, uint32_t n);

/* Build matrix multiplication kernel argument constant bank buffer (10 words) */
int omega_blackwell_build_cbank_args_matmul(uint32_t *args_words, uint64_t a_va, uint64_t b_va,
                                           uint64_t c_va, uint32_t m, uint32_t k, uint32_t n);

/* Verify QMD structural invariants (SKEDCHECK05, version, alignment, etc.) */
int omega_blackwell_verify_qmd_invariants(const uint32_t *qmd1_words);

#endif /* OMEGA_BLACKWELL_QMD_H */
