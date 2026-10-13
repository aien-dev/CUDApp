#ifndef OMEGA_GPU_PREFILL_H
#define OMEGA_GPU_PREFILL_H
/*
 * omega_gpu_prefill (OWN-30B T3b, 2026-10-13): the device-resident one-pass prefill API on the shared session. New file.
 * Every buffer the prefill pipeline passes between its kernels stays on the device; the host builds small tables only.
 *
 * Memory policy (nvos.h:1116-1118, read 2026-10-13: GPU caching "For system memory ... will not be coherent with direct CPU
 * mappings"; clc96f.h:72 L2_FLUSH_DIRTY):
 *  - DEVICE buffers (omega_gpu_pf_alloc_dev) are GPU-cached and device-private: the CPU never reads them and never writes
 *    them; they are filled only by GPU writes (omega_gpu_pf_upload copies through an uncached staging buffer with the copy
 *    kernel), so the GPU L2 never holds a stale line, and GPU producers / consumers share the L2 (the whole point: the
 *    uncached buffers get no cache reuse, evidence OWN30B-T3-TC-PEAK-BW).
 *  - HOST buffers (omega_gpu_pf_alloc_host) are the session's GPU-uncached buffers: tables and inputs the host writes,
 *    results the host reads.
 *  - Reading a device buffer back (tests, logits) goes through omega_gpu_pf_download: a GPU copy into uncached staging.
 * One session hook (on_close) frees every kernel and staging buffer of this module.
 *
 * All functions take the session lock themselves unless named _locked. Return codes OMEGA_GPU_PF_*.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "omega_gpu_session.h"

#ifdef __cplusplus
extern "C" {
#endif
enum { OMEGA_GPU_PF_OK = 0, OMEGA_GPU_PF_BAD_ARGS = -1, OMEGA_GPU_PF_TOO_LARGE = -2, OMEGA_GPU_PF_CODEGEN_FAIL = -3, OMEGA_GPU_PF_CHIP_FAIL = -4 };
const char *omega_gpu_pf_rc_name(int rc);

/* buffers */
int omega_gpu_pf_alloc_dev(size_t bytes, OmegaGpuMem *out);  /* GPU-cached, device-private */
int omega_gpu_pf_alloc_host(size_t bytes, OmegaGpuMem *out); /* GPU-uncached, host reads and writes */
void omega_gpu_pf_free(OmegaGpuMem *m);
/* host bytes -> device buffer at byte offset off (16-byte multiples), through the staging buffer and the copy kernel */
int omega_gpu_pf_upload(const OmegaGpuMem *dst, size_t off, const void *src, size_t bytes);
/* device buffer bytes at off -> host (16-byte multiples) */
int omega_gpu_pf_download(const OmegaGpuMem *src, size_t off, void *dst, size_t bytes);
/* device-to-device copy by VA (16-byte multiples) */
int omega_gpu_pf_copy_va(uint64_t dst_va, uint64_t src_va, size_t bytes);

/* accumulated chip time and launch count of this module since the last reset (prefill timing) */
void omega_gpu_pf_stats(uint64_t *chip_ns, uint64_t *launches);
void omega_gpu_pf_stats_reset(void);

/* ---- internal to the prefill module family (omega_gpu_prefill*.c) ---- */
typedef struct { uint32_t kind; uint32_t words[15]; } OmegaPfKey; /* kernel identity: kind + the generator's spec words */
typedef int (*OmegaPfGen)(const OmegaPfKey *key, void *prog);    /* builds the BlackwellIRProgram for key */
typedef struct { uint64_t code_va; uint32_t gpr_declared, gpr_peak; } OmegaPfKernel;
/* lock held, device open: the resident kernel for key (generated, encoded, patched and uploaded on first use) */
int omega_gpu_pf_kernel_locked(const OmegaPfKey *key, OmegaPfGen gen, OmegaPfKernel *out);
/* lock held: one launch, chip time added to the module stats (and to *ns when given) */
int omega_gpu_pf_launch_locked(const OmegaPfKernel *k, uint32_t threads, uint32_t grid_x, uint32_t grid_y, uint32_t shared_bytes,
                               const uint32_t *args, uint32_t n_args, uint64_t *ns);
int omega_gpu_pf_open_locked(void); /* opens the device, registers the hook: OMEGA_GPU_PF_OK or CHIP_FAIL */
#ifdef __cplusplus
}
#endif
#endif
