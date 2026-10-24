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

#ifndef OMEGA_RUNTIME_API_H
#define OMEGA_RUNTIME_API_H
/*
 * Omega Runtime API version (R1, omega-mv#594). Defines the version and nothing else; includes nothing.
 *
 * The API is the set of entries listed in docs/runtime/omega_runtime_api.manifest and described in
 * docs/runtime/API.md, which also states the compatibility rule:
 *   MINOR goes up when something is added; MAJOR goes up when a STABLE entry is removed, renamed or changes
 *   signature, layout, constant value or documented behaviour. MAJOR 0 is the first, written-down version.
 * tools/runtime_api_check.sh (make test-runtime-api) fails when this header, the manifest and API.md disagree.
 */
#define OMEGA_RUNTIME_API_VERSION_MAJOR 0
#define OMEGA_RUNTIME_API_VERSION_MINOR 1
/* One number for #if comparisons: (MAJOR << 16) | MINOR. */
#define OMEGA_RUNTIME_API_VERSION ((OMEGA_RUNTIME_API_VERSION_MAJOR << 16) | OMEGA_RUNTIME_API_VERSION_MINOR)
#endif
