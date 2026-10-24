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
