#ifndef OMEGA_BLACKWELL_QMD_H
#define OMEGA_BLACKWELL_QMD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define OMEGA_BW_QMD_WORDS              96
#define OMEGA_BW_QMD_BYTES              (OMEGA_BW_QMD_WORDS * 4) /* 384 bytes */
#define OMEGA_BW_CBANK_DRIVER_WORDS     224
#define OMEGA_BW_CBANK_ARGS_WORDS       7

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
} OmegaBlackwellQmdConfig;

/* Build QMD 0: GRID_NULL null-barrier grid with dependence counter 2 */
int omega_blackwell_build_qmd0(uint32_t *qmd0_words, uint64_t qmd0_va, uint64_t qmd1_va);

/* Build QMD 1: GRID_CTA compute grid for Blackwell sm_121 vector add */
int omega_blackwell_build_qmd1(uint32_t *qmd1_words, const OmegaBlackwellQmdConfig *cfg);

/* Build driver constant bank 0 (224 words) with hardware descriptor addresses */
int omega_blackwell_build_cbank_driver(uint32_t *cbank_words, uint64_t cbank_va);

/* Build kernel argument constant bank buffer (7 words: pointers A, B, C, length N) */
int omega_blackwell_build_cbank_args(uint32_t *args_words, uint64_t a_va, uint64_t b_va,
                                    uint64_t c_va, uint32_t n);

/* Verify QMD structural invariants (SKEDCHECK05, version, alignment, etc.) */
int omega_blackwell_verify_qmd_invariants(const uint32_t *qmd1_words);

#endif /* OMEGA_BLACKWELL_QMD_H */
