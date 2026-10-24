/* SPDX-License-Identifier: AGPL-3.0-only */
/*
 * omega_native_artifact: export one already-generated CUDApp/Omega GPU
 * realization as an offline, digest-bound handoff artifact.
 *
 * Identity is the canonical manifest bytes; no live GPU virtual address, no
 * open device handle and no privileged capability enters the artifact. The
 * QMD bytes are exported as a relocatable template (all device addresses zero)
 * so an admitting loader must fill them and bind the result in its own receipt.
 *
 * This module does not compile kernels, open a device, load a cubin or launch
 * work. It packages bytes the existing code generator produced and refuses a
 * malformed or unsupported descriptor.
 */
#ifndef OMEGA_NATIVE_ARTIFACT_H
#define OMEGA_NATIVE_ARTIFACT_H

#include <stddef.h>
#include <stdint.h>
#include "omega_blackwell_matmul.h"
#include "omega_cubin.h"

#define OMEGA_ARTIFACT_FORMAT "omega-native-artifact/1"
#define OMEGA_ARTIFACT_ID_MAX 65u
#define OMEGA_ARTIFACT_QMD_WORDS 96u

enum {
    OMEGA_ARTIFACT_OK = 0,
    OMEGA_ARTIFACT_E_ARG = 1,
    OMEGA_ARTIFACT_E_FORMAT = 2,
    OMEGA_ARTIFACT_E_CONTRACT = 3,
    OMEGA_ARTIFACT_E_ISA = 4,
    OMEGA_ARTIFACT_E_DIGEST = 5,
    OMEGA_ARTIFACT_E_IO = 6,
    OMEGA_ARTIFACT_E_RELOC = 7,
    OMEGA_ARTIFACT_E_DRIVER = 8,
    OMEGA_ARTIFACT_E_UNSUPPORTED = 9,
    OMEGA_ARTIFACT_E_MEM = 10
};

typedef struct {
    const char *op;                 /* "matmul" or "gqa-causal-attention" */
    const char *contract_name;      /* e.g. "matmul-bf16-v1" */
    const char *contract_id_hex;    /* 64 lowercase hex */
    const char *semantic_spec_id_hex; /* 64 lowercase hex */
    const char *kernel_function;    /* cubin entry symbol */
    const char *precision;          /* "bf16" / "f32-kv" */
    const char *accumulation;       /* "fp32" */
    const char *tolerance;          /* frozen numeric text, no spaces */
    const uint8_t *text; size_t text_len;
    const OmegaCubinParam *params; size_t nparams;
    uint32_t gpr_count;
    uint32_t grid_x, grid_y;
    uint32_t block_x, block_y, block_z;
    uint32_t shared_bytes;
    uint32_t cbank_offset;
    const uint32_t *cbank_args; size_t cbank_words;
    const uint32_t *qmd0;           /* 96 words, addresses zero */
    const uint32_t *qmd1;           /* 96 words, addresses zero */
    const char *min_driver;         /* e.g. "580.173.02" */
    const char *requires;           /* e.g. "sm_121,nvidia-cubin-abi-v8" */
    const char *source_commit;      /* 40 hex, or "UNKNOWN" */
    const char *tool_version;       /* e.g. "cudapp-artifact-export/1" */
    const char *qualification;      /* e.g. "offline-disassembly=PASS;device-load=NOT_RUN" */
} OmegaNativeArtifact;

const char *omega_native_artifact_rc_name(int rc);

/* Export <prefix>.{text.bin,cubin,qmd0.bin,qmd1.bin,cbank_args.bin,manifest,json,sha256}.
 * out_id receives the 64-hex artifact id. Refuses unsupported descriptors. */
int omega_native_artifact_export(const OmegaNativeArtifact *a, const char *out_prefix,
                                 char out_id[OMEGA_ARTIFACT_ID_MAX], char *err, size_t err_cap);

/* Recompute and check an exported artifact's identity and every file digest.
 * current_driver may be NULL to skip the minimum-driver check. */
int omega_native_artifact_admit(const char *out_prefix, const char *current_driver,
                                char out_id[OMEGA_ARTIFACT_ID_MAX], char *err, size_t err_cap);

#endif /* OMEGA_NATIVE_ARTIFACT_H */
