/*
 * m16_concurrent.c -- M16 concurrent multi-channel qualification test.
 *
 * Sequence:
 *   1. Initialize two independent M16NativeContext instances (ctxA, ctxB).
 *   2. Allocate separate channels, GPFIFO rings, USERD, and tokens.
 *   3. Allocate independent coherent markers: markerA, markerB.
 *   4. Submit independent work:
 *        Channel A -> 0x16000001
 *        Channel B -> 0x16000002
 *   5. Verify independent causality:
 *        Both withheld -> markerA=0, markerB=0
 *        Ring A -> markerA=0x16000001, markerB=0
 *        Ring B -> markerB=0x16000002
 *   6. Prove non-collision and dynamic resource independence.
 */
#include "m16_native.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    printf("=== M16 CONCURRENT MULTI-CHANNEL QUALIFICATION ===\n");

    M16NativeContext ctxA, ctxB;

    printf("\n--- INITIALIZING CONTEXT A ---\n");
    if (m16_native_open(&ctxA) != 0 || m16_native_create_channel(&ctxA) != 0) {
        fprintf(stderr, "FATAL: Failed to initialize Context A: %s\n", ctxA.rm.err);
        return 1;
    }
    printf("[OBSERVED] Context A: token=0x%08x gpfifo=0x%08x fifo_va=0x%lx\n",
           ctxA.rm.token, ctxA.rm.gpfifo, (unsigned long)ctxA.rm.fifo.va);

    printf("\n--- INITIALIZING CONTEXT B ---\n");
    if (m16_native_open(&ctxB) != 0 || m16_native_create_channel(&ctxB) != 0) {
        fprintf(stderr, "FATAL: Failed to initialize Context B: %s\n", ctxB.rm.err);
        m16_native_close(&ctxA);
        return 1;
    }
    printf("[OBSERVED] Context B: token=0x%08x gpfifo=0x%08x fifo_va=0x%lx\n",
           ctxB.rm.token, ctxB.rm.gpfifo, (unsigned long)ctxB.rm.fifo.va);

    /* Allocate markers */
    void *cpuA = NULL, *cpuB = NULL;
    uint64_t vaA = 0, vaB = 0;
    if (m16_native_alloc_memory(&ctxA, 0x1000, &cpuA, &vaA) != 0 ||
        m16_native_alloc_memory(&ctxB, 0x1000, &cpuB, &vaB) != 0) {
        fprintf(stderr, "FATAL: Failed to allocate marker memories\n");
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 1;
    }
    volatile uint32_t *markerA = (volatile uint32_t *)cpuA;
    volatile uint32_t *markerB = (volatile uint32_t *)cpuB;
    *markerA = 0;
    *markerB = 0;

    printf("[OBSERVED] Marker A: VA=0x%lx CPU=%p (initial=0)\n", (unsigned long)vaA, cpuA);
    printf("[OBSERVED] Marker B: VA=0x%lx CPU=%p (initial=0)\n", (unsigned long)vaB, cpuB);

    /* Build methods */
    uint32_t mthdA[32], mthdB[32];
    int nA = m16_native_build_release(mthdA, vaA, 0x16000001, ctxA.rm.compute_class);
    int nB = m16_native_build_release(mthdB, vaB, 0x16000002, ctxB.rm.compute_class);

    /* Enqueue to both without ringing */
    printf("\n--- ENQUEUING WORK WITHOUT DOORBELLS ---\n");
    if (m16_native_enqueue_methods(&ctxA, mthdA, (size_t)nA) != 0 ||
        m16_native_enqueue_methods(&ctxB, mthdB, (size_t)nB) != 0) {
        fprintf(stderr, "FATAL: Failed to enqueue methods\n");
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 1;
    }
    usleep(50000);
    printf("[OBSERVED] Withheld doorbells: markerA=0x%08x markerB=0x%08x\n", *markerA, *markerB);
    if (*markerA != 0 || *markerB != 0) {
        fprintf(stderr, "FATAL: Premature execution on withheld doorbells!\n");
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 2;
    }

    /* Ring Doorbell A only */
    printf("\n--- RINGING DOORBELL A ONLY ---\n");
    m16_native_ring_doorbell(&ctxA);
    if (m16_native_wait_marker(markerA, 0x16000001, 2000) != 0) {
        fprintf(stderr, "FATAL: Timeout waiting for markerA!\n");
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 3;
    }
    printf("[OBSERVED] After Doorbell A: markerA=0x%08x markerB=0x%08x (expected A=0x16000001, B=0x00000000)\n",
           *markerA, *markerB);
    if (*markerB != 0) {
        fprintf(stderr, "FATAL: Cross-talk detected! Channel B executed when only Doorbell A was rung!\n");
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 4;
    }

    /* Ring Doorbell B */
    printf("\n--- RINGING DOORBELL B ---\n");
    m16_native_ring_doorbell(&ctxB);
    if (m16_native_wait_marker(markerB, 0x16000002, 2000) != 0) {
        fprintf(stderr, "FATAL: Timeout waiting for markerB!\n");
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 5;
    }
    printf("[OBSERVED] After Doorbell B: markerA=0x%08x markerB=0x%08x (expected A=0x16000001, B=0x16000002)\n",
           *markerA, *markerB);

    /* Verify distinct resource handles */
    printf("\n--- VERIFYING RESOURCE INDEPENDENCE ---\n");
    printf("[OBSERVED] Handle check: Root A=0x%x B=0x%x, Channel A=0x%x B=0x%x, Token A=0x%x B=0x%x\n",
           ctxA.rm.root, ctxB.rm.root, ctxA.rm.gpfifo, ctxB.rm.gpfifo, ctxA.rm.token, ctxB.rm.token);

    if (ctxA.rm.root == ctxB.rm.root) {
        fprintf(stderr, "FATAL: Root handles collided between sessions!\n");
        m16_native_close(&ctxA);
        m16_native_close(&ctxB);
        return 6;
    }

    printf("CONCURRENT_CHANNEL_PASS: 2 independent channels executed without collision.\n");

    nvrm_retire(&ctxA.rm, ctxA.rm.put);
    nvrm_retire(&ctxB.rm, ctxB.rm.put);
    m16_native_close(&ctxA);
    m16_native_close(&ctxB);
    return 0;
}
