/*
 * m16_native.h -- M16 libcuda-free native accelerator interface.
 *
 * Wraps the generic NVIDIA RM/UVM substrate (nvrm/) for independent
 * milestone qualification and testability.
 */
#ifndef M16_NATIVE_H
#define M16_NATIVE_H

#include "nvrm.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    Nvrm rm;
    NvrmMem pb_mem;
    NvrmMem marker_mem;
} M16NativeContext;

/* Open RM client, device, VA space, and query card state */
int  m16_native_open(M16NativeContext *ctx);

/* Allocate coherent unified memory mapped identically in CPU and GPU VA */
int  m16_native_alloc_memory(M16NativeContext *ctx, size_t size, void **cpu_addr, uint64_t *gpu_va);

/* Create channel group, context share, GPFIFO ring, and schedule channel */
int  m16_native_create_channel(M16NativeContext *ctx);

/* Build the standard M16 semaphore-release method stream */
int  m16_native_build_release(uint32_t *words, uint64_t marker_va, uint32_t payload, uint32_t compute_class);

/* Enqueue method stream to GPFIFO ring (without ringing doorbell) */
int  m16_native_enqueue_methods(M16NativeContext *ctx, const uint32_t *methods, size_t count);

/* Ring usermode doorbell with work-submit token */
void m16_native_ring_doorbell(M16NativeContext *ctx);

/* Enqueue and ring doorbell */
int  m16_native_submit_methods(M16NativeContext *ctx, const uint32_t *methods, size_t count);

/* Poll coherent marker until it equals expected value or timeout_ms expires */
int  m16_native_wait_marker(volatile uint32_t *marker, uint32_t expected, uint64_t timeout_ms);

/* Poll coherent marker until (int32_t)(*marker - target) >= 0 (monotonic,
 * serial-number compare per RFC 1982: correct across uint32_t wraparound)
 * or timeout_ms expires. Returns 0 once satisfied, -1 on timeout. */
int  m16_native_wait_marker_ge(volatile uint32_t *marker, uint32_t target, uint64_t timeout_ms);

/* Free all RM objects, mappings, and close file descriptors */
void m16_native_close(M16NativeContext *ctx);

#endif /* M16_NATIVE_H */
