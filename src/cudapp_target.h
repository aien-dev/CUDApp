/* SPDX-License-Identifier: MIT */
/* cudapp_target.h: which machines CUDA++ names, and which of them have a receipt.
 *
 * A host target means this binary runs the program in CUDA++'s own CPU code.
 * A qualified GPU target means that chip ran the program and the receipt is in this repository.
 * A named GPU target is a vendor CUDA++ will write machine code for. It is not a claim that
 * the chip has run anything. Add a qualified row only with a receipt (same rule as omega_gpu_chip.h). */
#ifndef CUDAPP_TARGET_H
#define CUDAPP_TARGET_H

#include <stdint.h>

#define CUDAPP_PCI_NVIDIA 0x10deu
#define CUDAPP_PCI_AMD    0x1002u
#define CUDAPP_PCI_INTEL  0x8086u
#define CUDAPP_PCI_APPLE  0x106bu
#define CUDAPP_PCI_GB10   0x2e12u

typedef enum {
    CUDAPP_KIND_HOST = 1,
    CUDAPP_KIND_GPU_QUALIFIED = 2,
    CUDAPP_KIND_GPU_NAMED = 3
} CudappTargetKind;

typedef struct {
    const char *id;
    const char *name;
    const char *isa;
    CudappTargetKind kind;
    const char *receipt; /* path in this repository, or NULL when there is no receipt */
    const char *note;
} CudappTarget;

/* The CPU this binary was compiled for. Never NULL. */
const CudappTarget *cudapp_host_target(void);

/* The GPU target for a PCI vendor and device.
 * NULL when CUDA++ does not name that vendor.
 * A named row is not a qualified row: check kind and receipt. */
const CudappTarget *cudapp_gpu_target(uint16_t vendor, uint16_t device);

/* Every named target. *count is set to the length. */
const CudappTarget *cudapp_targets(int *count);

#endif
