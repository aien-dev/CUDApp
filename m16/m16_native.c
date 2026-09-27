/*
 * m16_native.c -- M16 libcuda-free native accelerator implementation.
 *
 * Implements clean, reusable submission wrappers around nvrm/ without
 * any closed NVIDIA userspace libraries or hardcoded addresses.
 */
#include "m16_native.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int m16_native_open(M16NativeContext *ctx) {
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));
    return nvrm_open(&ctx->rm);
}

int m16_native_alloc_memory(M16NativeContext *ctx, size_t size, void **cpu_addr, uint64_t *gpu_va) {
    if (!ctx || !cpu_addr || !gpu_va) return -1;
    NvrmMem mem;
    if (nvrm_alloc(&ctx->rm, size, &mem) != 0) return -1;
    *cpu_addr = mem.cpu;
    *gpu_va = mem.va;
    return 0;
}

int m16_native_create_channel(M16NativeContext *ctx) {
    if (!ctx) return -1;
    if (nvrm_channel(&ctx->rm) != 0) return -1;
    /* Preallocate pushbuffer memory (4 KiB) */
    if (nvrm_alloc(&ctx->rm, 0x1000, &ctx->pb_mem) != 0) return -1;
    return 0;
}

int m16_native_build_release(uint32_t *words, uint64_t marker_va, uint32_t payload, uint32_t compute_class) {
    if (!words) return -1;
    uint32_t n = 0;
    /* Bind compute class to subchannel 1 */
    words[n++] = nvrm_mthd(1, 0x0000, 1);
    words[n++] = compute_class;

    /* Semaphore release method packet on subchannel 0 */
    words[n++] = nvrm_mthd(0, 0x005c, 5);
    words[n++] = (uint32_t)marker_va;
    words[n++] = (uint32_t)(marker_va >> 32);
    words[n++] = payload;
    words[n++] = 0;
    words[n++] = 0x1 | (1u << 20); /* RELEASE | WFI */
    return (int)n;
}

int m16_native_enqueue_methods(M16NativeContext *ctx, const uint32_t *methods, size_t count) {
    if (!ctx || !methods || count == 0) return -1;
    if (count * sizeof(uint32_t) > ctx->pb_mem.size) return -1;
    memcpy(ctx->pb_mem.cpu, methods, count * sizeof(uint32_t));
    __asm__ volatile("dsb sy" ::: "memory");
    return nvrm_enqueue(&ctx->rm, &ctx->pb_mem, 0, (uint32_t)count);
}

void m16_native_ring_doorbell(M16NativeContext *ctx) {
    if (ctx) nvrm_ring(&ctx->rm);
}

int m16_native_submit_methods(M16NativeContext *ctx, const uint32_t *methods, size_t count) {
    if (m16_native_enqueue_methods(ctx, methods, count) != 0) return -1;
    m16_native_ring_doorbell(ctx);
    return 0;
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

int m16_native_wait_marker(volatile uint32_t *marker, uint32_t expected, uint64_t timeout_ms) {
    if (!marker) return -1;
    uint64_t end = now_ns() + timeout_ms * 1000000ull;
    while (now_ns() < end) {
        if (*marker == expected) return 0;
        usleep(50);
        __asm__ volatile("yield");
    }
    return (*marker == expected) ? 0 : -1;
}

int m16_native_wait_marker_ge(volatile uint32_t *marker, uint32_t target, uint64_t timeout_ms) {
    if (!marker) return -1;
    uint64_t end = now_ns() + timeout_ms * 1000000ull;
    while (now_ns() < end) {
        if ((int32_t)(*marker - target) >= 0) return 0;
        usleep(50);
        __asm__ volatile("yield");
    }
    return ((int32_t)(*marker - target) >= 0) ? 0 : -1;
}

void m16_native_close(M16NativeContext *ctx) {
    if (ctx) {
        nvrm_close(&ctx->rm);
    }
}
