/*
 * m16_requalify.c -- M16 libcuda-free causality qualification test.
 *
 * Sequence:
 *   1. Initialize raw RM client (zero libcuda).
 *   2. Create independent channel, USERD, and obtain work-submit token.
 *   3. Allocate coherent marker memory, initialize marker = 0.
 *   4. Construct M16 semaphore-release method stream (expected payload: 0x16c0ffee).
 *   5. Enqueue work to GPFIFO ring WITHOUT ringing doorbell.
 *   6. Negative control: wait 100ms, assert marker remains strictly 0.
 *   7. Positive stimulus: ring doorbell with session work-submit token.
 *   8. Assert marker changes to 0x16c0ffee.
 *   9. Inspect /proc/self/maps to prove libcuda.so was never loaded.
 *  10. Output machine-parsable receipts.
 */
#include "m16_native.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int check_proc_maps(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return -1;
    char line[512];
    int hit = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "libcuda.so") || strstr(line, "libcudart.so")) {
            hit = 1;
            break;
        }
    }
    fclose(f);
    return hit;
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    printf("=== M16 CORRECTIVE REQUALIFICATION: CONTROLLED CAUSALITY TRANSACTION ===\n");

    M16NativeContext ctx;
    if (m16_native_open(&ctx) != 0) {
        fprintf(stderr, "FATAL: m16_native_open failed: %s\n", ctx.rm.err);
        return 1;
    }
    printf("[OBSERVED] RM client opened: GPU minor=%u compute_class=0x%x sm_version=0x%x\n",
           ctx.rm.minor, ctx.rm.compute_class, ctx.rm.sm_version);

    if (m16_native_create_channel(&ctx) != 0) {
        fprintf(stderr, "FATAL: m16_native_create_channel failed: %s\n", ctx.rm.err);
        m16_native_close(&ctx);
        return 1;
    }
    printf("[OBSERVED] Channel initialized: token=0x%08x gpfifo=0x%08x fifo_va=0x%lx doorbell=%p\n",
           ctx.rm.token, ctx.rm.gpfifo, (unsigned long)ctx.rm.fifo.va, (void *)ctx.rm.doorbell);

    void *marker_cpu = NULL;
    uint64_t marker_va = 0;
    if (m16_native_alloc_memory(&ctx, 0x1000, &marker_cpu, &marker_va) != 0) {
        fprintf(stderr, "FATAL: alloc marker failed: %s\n", ctx.rm.err);
        m16_native_close(&ctx);
        return 1;
    }
    volatile uint32_t *marker = (volatile uint32_t *)marker_cpu;
    *marker = 0;
    printf("[OBSERVED] Coherent marker allocated at VA=0x%lx CPU=%p (initial=0x%08x)\n",
           (unsigned long)marker_va, marker_cpu, *marker);

    uint32_t methods[32];
    int nwords = m16_native_build_release(methods, marker_va, 0x16c0ffee, ctx.rm.compute_class);
    if (nwords <= 0) {
        fprintf(stderr, "FATAL: build release failed\n");
        m16_native_close(&ctx);
        return 1;
    }

    /* Step 1: Enqueue to GPFIFO without ringing doorbell */
    printf("\n--- STEP 1: NEGATIVE CONTROL (DOORBELL WITHHELD) ---\n");
    if (m16_native_enqueue_methods(&ctx, methods, (size_t)nwords) != 0) {
        fprintf(stderr, "FATAL: enqueue failed: %s\n", ctx.rm.err);
        m16_native_close(&ctx);
        return 1;
    }
    printf("[OBSERVED] Enqueued %d words to GPFIFO index %u. Withholding doorbell notification...\n",
           nwords, ctx.rm.put - 1);

    /* Wait 100ms and verify marker has NOT changed */
    usleep(100000);
    printf("[OBSERVED] Marker after 100ms without doorbell: 0x%08x (expected 0x00000000)\n", *marker);
    if (*marker != 0) {
        fprintf(stderr, "FATAL: Negative control failed: marker changed without doorbell!\n");
        m16_native_close(&ctx);
        return 2;
    }
    printf("NEGATIVE_CONTROL_PASS: Marker strictly remained 0x00000000 while doorbell was withheld.\n");

    /* Step 2: Positive stimulus (Ring doorbell) */
    printf("\n--- STEP 2: POSITIVE STIMULUS (DOORBELL RUNG) ---\n");
    printf("[OBSERVED] Storing work-submit token 0x%08x to doorbell MMIO at %p...\n",
           ctx.rm.token, (void *)ctx.rm.doorbell);
    m16_native_ring_doorbell(&ctx);

    if (m16_native_wait_marker(marker, 0x16c0ffee, 2000) != 0) {
        fprintf(stderr, "FATAL: Positive control failed: timeout waiting for marker 0x16c0ffee! Actual=0x%08x\n",
                *marker);
        m16_native_close(&ctx);
        return 3;
    }
    printf("[OBSERVED] Marker after doorbell rung: 0x%08x (expected 0x16c0ffee)\n", *marker);
    printf("POSITIVE_CONTROL_PASS: Marker successfully updated to 0x16c0ffee upon doorbell store.\n");

    /* Step 3: Process mappings audit */
    int cuda_mapped = check_proc_maps();
    printf("\n--- STEP 3: RUNTIME LIBCUDA AUDIT ---\n");
    printf("[OBSERVED] libcuda.so / libcudart.so present in /proc/self/maps: %s\n",
           cuda_mapped ? "YES (FAIL)" : "NO (PASS)");
    if (cuda_mapped) {
        fprintf(stderr, "FATAL: libcuda detected in process address space!\n");
        m16_native_close(&ctx);
        return 4;
    }
    printf("NO_LIBCUDA_PASS: Zero libcuda / libcudart mappings present in live process.\n");

    printf("\nCAUSALITY_VERIFICATION_PASS: Doorbell causality strictly confirmed.\n");
    printf("M16_REQUALIFICATION_PASS: Milestone 16 native execution verified.\n");

    nvrm_retire(&ctx.rm, ctx.rm.put);
    m16_native_close(&ctx);
    return 0;
}
