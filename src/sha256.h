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

/* The SHA-256 API (sha256_ctx, SHA256_DIGEST_SIZE, SHA256_BLOCK_SIZE, sha256_init/update/final/hash).
 * Everything is generated: om/sha256/sha256-om2.om (OM-2) exports the four functions, bound through the
 * omega-bridge native call ABI v2 as omega2_sha256_*; src/omega_api_decl.h declares sha256_ctx from the
 * unit's `layout shactx` and the four sha256_* functions, which src/omega_om_units.c defines as calls of
 * omega2_sha256_* (abort() on a nonzero status). Regenerate with om/units.sh. No hashing logic lives in C. */

#ifndef OMEGA_SHA256_H
#define OMEGA_SHA256_H
#include "omega_api_decl.h"
#endif /* OMEGA_SHA256_H */
