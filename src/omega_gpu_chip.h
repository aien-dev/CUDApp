/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* omega_gpu_chip.h: which GPUs CUDA++ has machine code for, and what it says about the others.
 *
 * Developer bar D6 (docs/PAIN_POINTS.md): a GPU works on day one or is refused with a sentence
 * that says what is missing and what to do. Never a bare "no kernel image is available".
 *
 * A chip is named by its PCI vendor and device id, read from /sys/bus/pci/devices. That needs no
 * device open and no driver call. (The driver's SM-version field reads 0xa04 on the GB10, which does
 * not tell sm_120 from sm_121: evidence/m16-blackwell-native-path-requalification-receipt.json.)
 * A row is only added with a receipt: the chip ran examples/hello_gpu and its gates on a named commit. */
#ifndef OMEGA_GPU_CHIP_H
#define OMEGA_GPU_CHIP_H

#include <stddef.h>
#include <stdint.h>

#define OMEGA_GPU_PCI_VENDOR_NVIDIA 0x10deu

typedef struct {
    uint16_t vendor, device;
    const char *name;      /* "NVIDIA GB10" */
    const char *sm;        /* "sm_121" */
    const char *receipt;   /* path of the receipt that qualified it, in this repository */
} OmegaGpuChipRow;

typedef struct {
    char bdf[64];          /* PCI address, e.g. "000f:01:00.0" */
    uint16_t vendor, device;
    uint32_t class_code;   /* 0x03xxxx: display or 3D controller */
} OmegaGpuPciDev;

/* The qualified row for vendor:device, or NULL. */
const OmegaGpuChipRow *omega_gpu_chip_find(uint16_t vendor, uint16_t device);

/* Every NVIDIA display/3D-class PCI function under <root>/sys/bus/pci/devices (root "" = the real
 * system). Fills up to max entries; returns how many were found (may exceed max), or -1 when the
 * folder cannot be read. */
int omega_gpu_chip_scan(const char *root, OmegaGpuPciDev *out, int max);

/* The refusal text for a GPU with no qualified row: what it is, why CUDA++ refuses it now rather
 * than failing later, what is missing, and what to do. Returns the length written (snprintf rules). */
int omega_gpu_chip_refusal(const OmegaGpuPciDev *d, char *buf, size_t n);

#endif
