#ifndef OMEGA_CANONICAL_H
#define OMEGA_CANONICAL_H

#include "omega_types.h"

int omega_canonical_encode(const OmegaObject *obj, uint8_t *out_buf, size_t max_len, size_t *out_len);
int omega_compute_semantic_id(OmegaObject *obj);
int omega_compare_semantic_id(const SemanticId *a, const SemanticId *b);
void omega_hex_semantic_id(const SemanticId *id, char out_hex[65]);
int omega_parse_hex_semantic_id(const char *hex, SemanticId *out_id);

#endif /* OMEGA_CANONICAL_H */
