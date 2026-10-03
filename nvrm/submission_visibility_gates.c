/* Real-device regression for CPU-published queue entries and rewritten
 * command buffers. No CUDA runtime; run under the shared chip queue.
 * The unfixed allocator stalls on the second submission on GB10.
 */
#include "m16_native.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int run_sequence(int reuse) {
    M16NativeContext ctx;
    if (m16_native_open(&ctx) != 0) {
        fprintf(stderr, "visibility: open failed\n");
        return 0;
    }
    int ok = 0;
    NvrmMem completion;
    if (m16_native_create_channel(&ctx) != 0 ||
        nvrm_alloc_gpu_uncached(&ctx.rm, 4096, &completion) != 0)
        goto finish;
    volatile uint32_t *marker = completion.cpu;
    *marker = 0;
    __asm__ volatile("dsb sy" ::: "memory");
    uint32_t count = reuse ? ctx.rm.entries * 2 + 3 : 2;
    for (uint32_t payload = 1; payload <= count; ++payload) {
        uint32_t methods[32];
        int words = m16_native_build_release(methods, completion.va,
                                             payload, ctx.rm.compute_class);
        if (words <= 0) goto finish;
        if (reuse) {
            /* Previous use is retired before the next call overwrites slot 0. */
            if (ctx.rm.put != ctx.rm.retired ||
                m16_native_enqueue_methods(&ctx, methods, (size_t)words) != 0)
                goto finish;
        } else {
            uint32_t offset = (payload - 1) * 128;
            memcpy((unsigned char *)ctx.pb_mem.cpu + offset, methods,
                   (size_t)words * sizeof(uint32_t));
            __asm__ volatile("dsb sy" ::: "memory");
            if (nvrm_enqueue(&ctx.rm, &ctx.pb_mem, offset, (uint32_t)words) != 0)
                goto finish;
        }
        if (payload == 1) {
            /* Negative control: no notification, no completed work. */
            if (m16_native_wait_marker_ge(marker, payload, 5) == 0 || *marker != 0) {
                fprintf(stderr, "visibility: pre-notification control failed\n");
                goto finish;
            }
        }
        m16_native_ring_doorbell(&ctx);
        if (m16_native_wait_marker_ge(marker, payload, 5000) != 0 ||
            *marker != payload) {
            fprintf(stderr, "VISIBILITY_FAIL reuse=%d expected=%" PRIu32
                    " observed=%" PRIu32 " put=%" PRIu32 " retired=%" PRIu32 "\n",
                    reuse, payload, *marker, ctx.rm.put, ctx.rm.retired);
            goto finish;
        }
        nvrm_retire(&ctx.rm, ctx.rm.put);
    }
    printf("visibility: reuse=%d submissions=%" PRIu32
           " ring_capacity=%" PRIu32 " final_marker=%" PRIu32 "\n",
           reuse, count, ctx.rm.entries, *marker);
    ok = 1;
finish:
    /* Tear down the channel before freeing memory, including failure paths. */
    if (m16_native_close(&ctx) != 0) ok = 0;
    return ok;
}

int main(void) {
    int distinct = run_sequence(0);
    printf("[%s] SUBMISSION_VISIBILITY_DISTINCT_COMMANDS\n", distinct ? "PASS" : "FAIL");
    int reuse = run_sequence(1);
    printf("[%s] SUBMISSION_VISIBILITY_REUSE_AND_RING_WRAP\n", reuse ? "PASS" : "FAIL");
    return distinct && reuse ? 0 : 1;
}
