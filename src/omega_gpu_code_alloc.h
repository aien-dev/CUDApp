/* Size of a GPU code buffer. Header only: no device, no physics headers.
 *
 * The SM instruction cache prefetches past the last instruction of a kernel, so the code buffer must
 * stay mapped for 2 KB after the code. Source (NVIDIA guidance as recorded by Mesa NVK,
 * src/nouveau/vulkan/nvk_device.c, shader heap set-up): "the I-cache pre-fetches and NVIDIA has
 * informed us overallocating shaders BOs by 2K is sufficient". A page-rounded buffer alone leaves
 * whatever happens to be left in the last page (2560 B of code: 1536 B; 2688 B: 1408 B).
 *
 * omega#323 (2026-10-07): GB10 Xid 31, GPCCLIENT_GCC FAULT_PTE read on a page boundary while the
 * head_dim 128 attention kernel (2688 B) ran; that this tail is the cause is INFERRED until the
 * padded build turns the fault off on the chip. Every code upload sizes its buffer here so no site
 * can drop the tail (tests/test_omega_gpu_code_alloc.c also scans src/ for page-rounded code sizes).
 */
#ifndef OMEGA_GPU_CODE_ALLOC_H
#define OMEGA_GPU_CODE_ALLOC_H

#include <stddef.h>
#include <stdint.h>

#define OMEGA_GPU_CODE_PREFETCH_TAIL 2048u
#define OMEGA_GPU_CODE_PAGE 0x1000u

/* Bytes to allocate for code_size bytes of code: code + 2 KB, rounded up to a page, at least one page.
 * 0 when the sum would overflow (callers refuse the upload). */
static inline size_t omega_gpu_code_alloc_bytes(size_t code_size) {
#ifdef OMEGA_GPU_CODE_ALLOC_MUTANT_NO_TAIL
    const size_t tail = 0; /* mutant: the page-rounded size every site used before omega#323 */
#else
    const size_t tail = OMEGA_GPU_CODE_PREFETCH_TAIL;
#endif
    if (code_size > SIZE_MAX - tail - (OMEGA_GPU_CODE_PAGE - 1u)) return 0;
    size_t b = (code_size + tail + (OMEGA_GPU_CODE_PAGE - 1u)) & ~(size_t)(OMEGA_GPU_CODE_PAGE - 1u);
    return b < OMEGA_GPU_CODE_PAGE ? OMEGA_GPU_CODE_PAGE : b;
}

#endif
