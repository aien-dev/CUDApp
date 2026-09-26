#include "omega_vector.h"
#include "omega_core.h"
#include "omega_canonical.h"
#include "sha256.h"
#include <string.h>

int omega_vector_spec_init(OmegaVectorSpec *spec, const char *name, uint32_t n) {
    if (!spec || !name || n == 0) return -1;
    memset(spec, 0, sizeof(*spec));
    strncpy(spec->name, name, sizeof(spec->name) - 1);
    spec->element_type = TYPE_UNSIGNED_INT;
    spec->element_width = 32;
    spec->element_count = n;
    spec->overflow = OVERFLOW_WRAP;

    /* Canonical spec digest: SHA-256("OMEGA_VECADD_U32" || N || overflow) */
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, (const uint8_t *)"OMEGA_VECADD_U32", 16);
    sha256_update(&ctx, (const uint8_t *)&spec->element_width, sizeof(spec->element_width));
    sha256_update(&ctx, (const uint8_t *)&spec->element_count, sizeof(spec->element_count));
    uint8_t ov = (uint8_t)spec->overflow;
    sha256_update(&ctx, &ov, sizeof(ov));
    sha256_final(&ctx, spec->spec_digest);

    memcpy(spec->spec_id.bytes, spec->spec_digest, OMEGA_ID_BYTES);
    return 0;
}

void omega_vector_oracle_u32(const uint32_t *a, const uint32_t *b, uint32_t *c, uint32_t n) {
    if (!a || !b || !c || n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        c[i] = (uint32_t)(a[i] + b[i]);
    }
}

int omega_vector_verify_oracle(const uint32_t *a, const uint32_t *b, const uint32_t *actual_c,
                               uint32_t n, size_t *first_mismatch_idx) {
    if (!a || !b || !actual_c || n == 0) return -1;
    for (size_t i = 0; i < n; i++) {
        uint32_t expected = (uint32_t)(a[i] + b[i]);
        if (actual_c[i] != expected) {
            if (first_mismatch_idx) *first_mismatch_idx = i;
            return -1;
        }
    }
    return 0;
}

void omega_vector_generate_deterministic(uint32_t *a, uint32_t *b, uint32_t *c_poison, uint32_t n) {
    if (!a || !b || !c_poison || n == 0) return;
    for (uint32_t i = 0; i < n; i++) {
        /* Deterministic vectors exercising small values, offsets, and high-bit modulo wrapping */
        if (i == 0) {
            a[i] = 0xFFFFFFFEU;
            b[i] = 3U; /* modulo 2^32 wrap check: 0xFFFFFFFE + 3 = 1 */
        } else if (i == 1) {
            a[i] = 0x80000000U;
            b[i] = 0x80000000U; /* modulo 2^32 wrap check: 2^31 + 2^31 = 0 */
        } else {
            a[i] = 1000U + i;
            b[i] = 2000U + i * 2;
        }
        c_poison[i] = OMEGA_VECTOR_POISON_VALUE;
    }
}

int omega_vector_build_spec_graph(OmegaGraph *g, uint32_t n, SemanticId *out_spec_id) {
    if (!g || n == 0 || !out_spec_id) return -1;

    OmegaObject *t_u32 = omega_build_type_uint(g, 32);
    if (!t_u32) return -1;

    OmegaObject *t_seq = omega_build_type_sequence(g, &t_u32->id, n);
    if (!t_seq) return -1;

    OmegaObject *op_add = omega_build_op_binary(g, OP_ADD, OVERFLOW_WRAP, &t_u32->id);
    if (!op_add) return -1;

    OmegaObject *spec_obj = omega_graph_add_object(g, KIND_OPERATION);
    if (!spec_obj) return -1;

    omega_object_add_attribute(spec_obj, "name", (const uint8_t *)"omega_blackwell_vector_add", 26);
    omega_object_add_relation(spec_obj, REL_DERIVED_FROM, &op_add->id);
    omega_object_add_relation(spec_obj, REL_DEPENDS_ON, &t_seq->id);

    uint8_t payload[8];
    memcpy(payload, &n, sizeof(n));
    uint32_t elem_sz = 4;
    memcpy(payload + 4, &elem_sz, sizeof(elem_sz));
    memcpy(spec_obj->payload, payload, sizeof(payload));
    spec_obj->payload_len = sizeof(payload);

    if (omega_compute_semantic_id(spec_obj) != 0) return -1;
    *out_spec_id = spec_obj->id;
    return 0;
}
