/* SPDX-License-Identifier: MIT */
#include "cudapp_target.h"

#include <stddef.h>

/* Receipts are the runs of make hello-host on that ISA, not a GPU qualification.
 * The GB10 receipt is the existing chip qualification. AMD, Intel and Apple GPUs
 * have no chip receipt: the encoder in src/omega_blackwell_encoder.c is sm_121. */
static const CudappTarget k_targets[] = {
    { "apple-m", "Apple M", "arm64", CUDAPP_KIND_HOST,
      "evidence/targets/apple-m.txt",
      "host code on Apple M. Not the Apple GPU." },
    { "amd64", "AMD and Intel CPUs", "x86_64", CUDAPP_KIND_HOST,
      "evidence/targets/amd64.txt",
      "host code on the x86_64 ISA those CPUs share. Not an AMD or Intel GPU." },
    { "aarch64-linux", "Linux ARM64 CPU", "aarch64", CUDAPP_KIND_HOST,
      "evidence/targets/aarch64-linux.txt",
      "host code on a Linux ARM64 CPU, including the Spark's CPU. Not the GB10 GPU." },
    { "nvidia-gb10", "NVIDIA GB10", "sm_121", CUDAPP_KIND_GPU_QUALIFIED,
      "evidence/m16-blackwell-native-path-requalification-receipt.json",
      "the chip ran the program. This is the only qualified GPU." },
    { "amd-gpu", "AMD GPU", "not written", CUDAPP_KIND_GPU_NAMED,
      NULL,
      "named. No machine-code encoder and no chip receipt. The program runs on the CPU until one exists." },
    { "intel-gpu", "Intel GPU", "not written", CUDAPP_KIND_GPU_NAMED,
      NULL,
      "named. No machine-code encoder and no chip receipt. The program runs on the CPU until one exists." },
    { "apple-gpu", "Apple GPU", "not written", CUDAPP_KIND_GPU_NAMED,
      NULL,
      "named. No machine-code encoder and no chip receipt. Apple M above is the CPU, not this GPU." },
};

static const CudappTarget k_nvidia_other = {
    "nvidia-other", "NVIDIA GPU", "not this chip", CUDAPP_KIND_GPU_NAMED,
    NULL,
    "CUDA++ has machine code for the GB10 only. Another NVIDIA device is refused by make doctor, not run."
};

const CudappTarget *cudapp_targets(int *count) {
    if (count) *count = (int)(sizeof k_targets / sizeof k_targets[0]);
    return k_targets;
}

const CudappTarget *cudapp_host_target(void) {
#if defined(__x86_64__)
    return &k_targets[1];
#elif defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    return &k_targets[0];
#elif defined(__aarch64__) || defined(__arm64__)
    return &k_targets[2];
#else
    return &k_targets[2];
#endif
}

const CudappTarget *cudapp_gpu_target(uint16_t vendor, uint16_t device) {
    if (vendor == CUDAPP_PCI_NVIDIA && device == CUDAPP_PCI_GB10) return &k_targets[3];
    if (vendor == CUDAPP_PCI_NVIDIA) return &k_nvidia_other;
    if (vendor == CUDAPP_PCI_AMD) return &k_targets[4];
    if (vendor == CUDAPP_PCI_INTEL) return &k_targets[5];
    if (vendor == CUDAPP_PCI_APPLE) return &k_targets[6];
    return NULL;
}
