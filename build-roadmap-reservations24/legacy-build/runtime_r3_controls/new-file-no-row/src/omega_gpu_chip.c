/* SPDX-License-Identifier: MIT */
/* omega_gpu_chip.c: the qualified-chip table and the refusal sentence (developer bar D6).
 * See omega_gpu_chip.h. Hosted: reads sysfs with stdio and dirent. */
#include "omega_gpu_chip.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One row per chip that has run hello_gpu and the gates on a named commit. Add a row only with a receipt. */
static const OmegaGpuChipRow k_rows[] = {
    { OMEGA_GPU_PCI_VENDOR_NVIDIA, 0x2e12u, "NVIDIA GB10", "sm_121",
      "evidence/m16-blackwell-native-path-requalification-receipt.json" },
};

const OmegaGpuChipRow *omega_gpu_chip_find(uint16_t vendor, uint16_t device) {
    for (size_t i = 0; i < sizeof k_rows / sizeof k_rows[0]; i++)
        if (k_rows[i].vendor == vendor && k_rows[i].device == device) return &k_rows[i];
    return NULL;
}

static int read_hex(const char *path, unsigned long *out) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[64] = { 0 };
    int ok = fgets(line, sizeof line, f) != NULL;
    fclose(f);
    if (!ok) return -1;
    char *end = NULL;
    *out = strtoul(line, &end, 16);
    return end == line ? -1 : 0;
}

int omega_gpu_chip_scan(const char *root, OmegaGpuPciDev *out, int max) {
    char dir[4096];
    snprintf(dir, sizeof dir, "%s/sys/bus/pci/devices", root ? root : "");
    DIR *d = opendir(dir);
    if (!d) return -1;
    int found = 0;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;
        char p[4400];
        unsigned long vendor, device, cls;
        snprintf(p, sizeof p, "%s/%s/vendor", dir, e->d_name);
        if (read_hex(p, &vendor) != 0 || vendor != OMEGA_GPU_PCI_VENDOR_NVIDIA) continue;
        snprintf(p, sizeof p, "%s/%s/class", dir, e->d_name);
        if (read_hex(p, &cls) != 0 || (cls >> 16) != 0x03u) continue; /* display/3D only: skips the audio function */
        snprintf(p, sizeof p, "%s/%s/device", dir, e->d_name);
        if (read_hex(p, &device) != 0) continue;
        if (found < max && out) {
            OmegaGpuPciDev *g = &out[found];
            memset(g, 0, sizeof *g);
            snprintf(g->bdf, sizeof g->bdf, "%.63s", e->d_name);
            g->vendor = (uint16_t)vendor; g->device = (uint16_t)device; g->class_code = (uint32_t)cls;
        }
        found++;
    }
    closedir(d);
    return found;
}

int omega_gpu_chip_refusal(const OmegaGpuPciDev *d, char *buf, size_t n) {
    return snprintf(buf, n,
        "GPU %s: NVIDIA device 0x%04x is not one CUDA++ has qualified.\n"
        "Why: CUDA++ writes each chip's machine code itself and has run it only on the NVIDIA GB10 (sm_121),\n"
        "so it refuses this GPU now instead of failing later with \"no kernel image is available\".\n"
        "What is missing: a target row for this chip, a check that the encoder's instruction words are valid on it,\n"
        "its launch descriptor and class numbers, and the hello-gpu qualification run with its receipt.\n"
        "What to do: follow docs/runtime/PORTABILITY.md section 5 (\"What a new GPU needs\") and CONTRIBUTING.md (\"Port a chip\"),\n"
        "or report it with the output of `make doctor` attached.",
        d->bdf, d->device);
}
