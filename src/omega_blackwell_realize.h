#ifndef OMEGA_BLACKWELL_REALIZE_H
#define OMEGA_BLACKWELL_REALIZE_H

#include "omega_types.h"
#include "omega_vector.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_encoder.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define OMEGA_BW_SM_ARCH_121 121

typedef struct {
    SemanticId realization_id;
    SemanticId spec_id;
    SemanticId machine_id;
    uint32_t sm_architecture;
    size_t code_len;
    uint8_t code_digest[32];
    uint8_t realization_digest[32];
    OmegaBlackwellQmdConfig qmd_cfg;
} OmegaBlackwellRealization;

/* Machine PROFILE digest for the Blackwell GB10 SM target (same on every GB10).
 * Not a machine identity: see AienMachineId (src/runtime/aien_machine_id.h). */
int omega_blackwell_get_machine_id(SemanticId *out_machine_id);

/* Bind an OmegaVectorSpec to Blackwell sm_121 hardware realization */
int omega_blackwell_realize_vector(const OmegaVectorSpec *spec, OmegaBlackwellRealization *out_real);

/* Compute triple hash realization ID: SHA-256(spec_id || machine_id || code_digest) */
int omega_blackwell_realization_compute_id(OmegaBlackwellRealization *real);

#endif /* OMEGA_BLACKWELL_REALIZE_H */
