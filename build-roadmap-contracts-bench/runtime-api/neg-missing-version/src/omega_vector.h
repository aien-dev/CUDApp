#ifndef OMEGA_VECTOR_H
#define OMEGA_VECTOR_H

#include "omega_types.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Poison value used to verify buffers are truly written by device execution */
#define OMEGA_VECTOR_POISON_VALUE 0xdeadbeefU

typedef struct {
    SemanticId spec_id;
    char name[64];
    TypeTag element_type;
    uint16_t element_width;
    uint32_t element_count;
    OverflowPolicy overflow;
    uint8_t spec_digest[32];
} OmegaVectorSpec;

/* Initialize an OmegaVectorSpec for 32-bit unsigned addition modulo 2^32 */
int omega_vector_spec_init(OmegaVectorSpec *spec, const char *name, uint32_t n);

/* Machine-independent semantic oracle: C[i] = (A[i] + B[i]) mod 2^32 */
void omega_vector_oracle_u32(const uint32_t *a, const uint32_t *b, uint32_t *c, uint32_t n);

/* Compare device output with semantic oracle. Returns 0 on exact match, -1 on mismatch */
int omega_vector_verify_oracle(const uint32_t *a, const uint32_t *b, const uint32_t *actual_c,
                               uint32_t n, size_t *first_mismatch_idx);

/* Populate deterministic fixed inputs A and B, initialize C with poison pattern */
void omega_vector_generate_deterministic(uint32_t *a, uint32_t *b, uint32_t *c_poison, uint32_t n);

/* Construct canonical OmegaGraph representation of the vector addition specification */
int omega_vector_build_spec_graph(OmegaGraph *g, uint32_t n, SemanticId *out_spec_id);

#endif /* OMEGA_VECTOR_H */
