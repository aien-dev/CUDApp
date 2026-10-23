#ifndef OMEGA_CANONICAL_H
#define OMEGA_CANONICAL_H

#include "omega_types.h"

/* Object encoding version (canonical header byte +0x04) per kind:
 * KIND_EFFECT -> OMEGA_EFFECT_VERSION (0x02), every other kind -> OMEGA_VERSION
 * (0x01). A decoder accepts exactly that pair (spec/effect-cap64-migration.md). */
#define OMEGA_CANON_OK              0
#define OMEGA_CANON_ERR_VERSION    -1  /* unknown version, or a version not owned by this kind */
#define OMEGA_CANON_ERR_EFFECT_V1  -2  /* legacy version 0x01 effect: refused, never reinterpreted */
uint8_t omega_canonical_object_version(uint8_t kind);
int omega_canonical_check_header(uint8_t version, uint8_t kind);

int omega_canonical_encode(const OmegaObject *obj, uint8_t *out_buf, size_t max_len, size_t *out_len);
int omega_compute_semantic_id(OmegaObject *obj);
int omega_compare_semantic_id(const SemanticId *a, const SemanticId *b);
void omega_hex_semantic_id(const SemanticId *id, char out_hex[65]);
int omega_parse_hex_semantic_id(const char *hex, SemanticId *out_id);

#endif /* OMEGA_CANONICAL_H */
