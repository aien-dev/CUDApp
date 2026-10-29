/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_NATIVE_ARTIFACT_SNAPSHOT_H
#define OMEGA_NATIVE_ARTIFACT_SNAPSHOT_H
#include "omega_native_artifact.h"
/* Private I/O custody only. These are exactly the bytes checked by existing
 * admission, retained without reopening paths. Not qualification authority. */
enum {
    OMEGA_SNAPSHOT_MANIFEST, OMEGA_SNAPSHOT_TEXT, OMEGA_SNAPSHOT_CUBIN,
    OMEGA_SNAPSHOT_QMD0, OMEGA_SNAPSHOT_QMD1, OMEGA_SNAPSHOT_CBANK,
    OMEGA_SNAPSHOT_FILES
};
typedef struct {
    uint8_t *bytes[OMEGA_SNAPSHOT_FILES];
    size_t sizes[OMEGA_SNAPSHOT_FILES];
    char kernel_function[64], id[OMEGA_ARTIFACT_ID_MAX];
} OmegaNativeArtifactSnapshot;
/* out must be empty; on success owns buffers until snapshot_free. On failure
 * remains empty. No changed file can replace the retained checked bytes. */
int omega_native_artifact_snapshot(const char *prefix, const char *driver,
    char out_id[OMEGA_ARTIFACT_ID_MAX], OmegaNativeArtifactSnapshot *out,
    char *err, size_t err_cap);
void omega_native_artifact_snapshot_free(OmegaNativeArtifactSnapshot *snapshot);
#endif
