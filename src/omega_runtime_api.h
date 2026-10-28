/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_RUNTIME_API_H
#define OMEGA_RUNTIME_API_H
/*
 * Omega Runtime API version (R1, omega-mv#594). Defines the version and nothing else; includes nothing.
 *
 * The API is the set of entries listed in docs/runtime/omega_runtime_api.manifest and described in
 * docs/runtime/API.md, which also states the compatibility rule (semantic versioning, MAJOR.MINOR.PATCH):
 *   MAJOR goes up when a STABLE entry is removed, renamed or changes signature, layout, constant value or
 *   documented behaviour; MINOR goes up when an entry is added or an EXPERIMENTAL entry changes or goes;
 *   PATCH goes up for a fix that changes no manifest entry. MAJOR 0 is the first, written-down version; while MAJOR
 *   is 0 a change that would need MAJOR needs a MINOR bump instead (semver 2.0.0 item 4; API.md section 2).
 * tools/runtime_api_check.sh (make test-runtime-api) fails when this header, the manifest and API.md disagree, and
 * tools/runtime_api_semver.sh fails when the manifest changed without the bump the rule asks for.
 */
#define OMEGA_RUNTIME_API_VERSION_MAJOR 0
#define OMEGA_RUNTIME_API_VERSION_MINOR 7
#define OMEGA_RUNTIME_API_VERSION_PATCH 0
/* One number for #if comparisons: (MAJOR << 16) | MINOR. PATCH is left out: it never changes the surface. */
#define OMEGA_RUNTIME_API_VERSION ((OMEGA_RUNTIME_API_VERSION_MAJOR << 16) | OMEGA_RUNTIME_API_VERSION_MINOR)
#endif
