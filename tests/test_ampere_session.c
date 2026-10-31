/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Real session/QMD path with an in-memory platform oracle. No GPU or ioctl.
 * This proves ordering/serialization, not device cache coherence. */
#include "omega_gpu_session.h"
#include "omega_gpu_platform_linux.h"
#include "omega_ampere_qmd.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
static struct { void *cpu[16]; uint64_t va[16], bytes[16]; unsigned next, submits; uint64_t ns; } sim;
static uint64_t expected_code;
static uint32_t expected_arg;
static uint32_t expected_instruction;
static uint64_t now_ns(void *c) { (void)c; return sim.ns += 1000; }
static uint64_t now_ms(void *c) { return now_ns(c) / 1000000; }
static int wait_ms(void *c, uint64_t ms) { (void)c; sim.ns += ms * 1000000; return 0; }
static int lock(void *c, uint32_t id) { (void)c; (void)id; return 0; }
static const char *diag(void *c) { (void)c; return "simulated Ampere"; }
static int dev_open(void *c, uint32_t ord, OmegaGpuDeviceHandle *d) {
    (void)c; assert(ord == 0); *d = (OmegaGpuDeviceHandle){15, 1}; return 0;
}
static int dev_close(void *c, OmegaGpuDeviceHandle d) { (void)c; (void)d; return 0; }
static int alloc(void *c, OmegaGpuDeviceHandle d, const OmegaGpuBufferDesc *b, OmegaGpuBufferHandle *h,
                 OmegaCpuView *cpu, OmegaGpuVa *va, OmegaGpuBusAddr *bus) {
    (void)c; (void)d; unsigned i = ++sim.next; assert(i < 15);
    sim.cpu[i] = calloc(1, b->bytes); assert(sim.cpu[i]); sim.bytes[i] = b->bytes;
    sim.va[i] = 0x100000000ull + (uint64_t)i * 0x1000000ull;
    *h = (OmegaGpuBufferHandle){i, 1}; *cpu = (OmegaCpuView){sim.cpu[i], b->bytes}; va->va = sim.va[i];
    if (bus) bus->iova = 0;
    return 0;
}
static int dealloc(void *c, OmegaGpuBufferHandle h) {
    (void)c; assert(h.index < 16); free(sim.cpu[h.index]); sim.cpu[h.index] = NULL; return 0;
}
static int sync_mem(void *c, OmegaGpuBufferHandle h, uint64_t off, uint64_t len) {
    (void)c; assert(h.index < 16 && sim.cpu[h.index]);
    assert(off <= sim.bytes[h.index] && len <= sim.bytes[h.index] - off); return 0;
}
static void *resolve(uint64_t va) {
    for (unsigned i = 1; i <= sim.next; i++)
        if (sim.cpu[i] && va >= sim.va[i] && va - sim.va[i] < sim.bytes[i]) return (uint8_t *)sim.cpu[i] + (va - sim.va[i]);
    assert(!"unmapped simulated GPU address"); return NULL;
}
static int poll(void *c, OmegaGpuBufferHandle h, uint64_t off, uint32_t want, uint64_t ms, uint32_t *seen) {
    (void)c; (void)ms; *seen = *(uint32_t *)((uint8_t *)sim.cpu[h.index] + off); assert(*seen == want); return 0;
}
static int qopen(void *c, OmegaGpuDeviceHandle d, OmegaGpuBufferHandle ring, OmegaGpuQueueHandle *q) {
    (void)c; (void)d; (void)ring; *q = (OmegaGpuQueueHandle){14, 1}; return 0;
}
static int qclose(void *c, OmegaGpuQueueHandle q) { (void)c; (void)q; return 0; }
static int submit(void *c, OmegaGpuQueueHandle queue, const uint32_t *w, uint32_t n) {
    (void)c; (void)queue; unsigned invalidated = 0, launched = 0; uint64_t qva = 0;
    for (uint32_t i = 0; i < n;) {
        uint32_t header = w[i++], count = (header >> 16) & 0x1fffu;
        uint32_t method = (header & 0x1fffu) << 2, subc = (header >> 13) & 7u;
        assert(header >> 28 == 2 && count && count <= n - i);
        const uint64_t m[16] = {4, method, count, subc, w[i], OMEGA_AMPERE_COMPUTE_B,
            0x02b4, 0x02c0, 0x005c, 0x0028, 0x021c, (1u << 0) | (1u << 4) | (1u << 12)};
        assert(omega_ampere_policy(m, NULL) == 0);
        if (subc == 1 && method == 0x021c) invalidated++;
        if (subc == 1 && method == 0x02b4) qva = (uint64_t)w[i] << 8;
        if (subc == 1 && method == 0x02c0) {
            assert(invalidated == 1 && qva && !launched++);
            const uint32_t *q = resolve(qva);
            uint64_t code = q[48] | ((uint64_t)(q[49] & 0x1ffffu) << 32);
            uint64_t cb = q[32] | ((uint64_t)(q[33] & 0x1ffffu) << 32);
            assert(*(uint32_t *)resolve(code) == expected_instruction);
            assert(code == expected_code);
            assert(*(uint32_t *)((uint8_t *)resolve(cb) + 0x160) == expected_arg);
            uint64_t sem = q[24] | ((uint64_t)(q[25] & 0xffu) << 32);
            assert(((q[25] >> 23) & 1u) == 1); /* QMD3 RELEASE0_ENABLE */
            *(uint32_t *)resolve(sem) = q[26];
        }
        if (subc == 0 && method == 0x005c) {
            assert(launched == 1);
            *(uint32_t *)resolve(w[i] | ((uint64_t)w[i + 1] << 32)) = w[i + 2];
        }
        i += count;
    }
    assert(invalidated == 1 && launched == 1); sim.submits++; return 0;
}
static void *halloc(void *c, size_t n) { (void)c; return malloc(n); }
static void hfree(void *c, void *p) { (void)c; free(p); }
static const OmegaGpuPlatformCaps caps = {
    .abi_major = OMEGA_GPU_PLATFORM_ABI_MAJOR, .abi_minor = OMEGA_GPU_PLATFORM_ABI_MINOR,
    .services = OMEGA_GPU_SVC_REQUIRED_SESSION, .max_devices = 1, .max_buffers = 14, .max_queues = 1,
    .max_inflight = 1, .max_buffer_bytes = 1ull << 30, .max_total_bytes = 1ull << 32,
};
static const OmegaGpuPlatform platform = {
    .time_now_ms = now_ms, .wait_ms = wait_ms, .lock = lock, .unlock = lock,
    .dma_alloc = alloc, .dma_free = dealloc, .sync_cpu_to_gpu = sync_mem, .sync_gpu_to_cpu = sync_mem,
    .poll_word = poll, .device_open = dev_open, .device_close = dev_close, .diag_text = diag,
    .queue_open = qopen, .queue_close = qclose, .queue_submit = submit, .queue_retire = qclose,
    .time_now_ns = now_ns, .host_alloc = halloc, .host_free = hfree,
};
/* Model the Linux-selected device target without linking a real driver. */
const OmegaGpuPlatform *omega_gpu_platform_linux(OmegaGpuPlatformCaps *out) { if (out) *out = caps; return &platform; }
uint32_t omega_gpu_platform_linux_target_sm(void) { return 86; }
int main(void) {
    omega_gpu_session_lock(); assert(omega_gpu_session_open());
    assert(omega_gpu_session_target_sm() == 86);
    OmegaGpuMem code0, code1;
    assert(omega_gpu_session_alloc_code(16, &code0) == 0);
    assert(omega_gpu_session_alloc_code(16, &code1) == 0);
    *(uint32_t *)code0.cpu = 0x1234; *(uint32_t *)code1.cpu = 0x5678;
    assert(omega_gpu_session_sync_to_gpu(&code0) == 0 && omega_gpu_session_sync_to_gpu(&code1) == 0);
    uint32_t arg = 7;
    OmegaGpuLaunch launch = {.code_va = code0.va, .gpr_count = 64, .threads_x = 32, .grid_x = 1, .grid_y = 1,
        .num_elements = 32, .args = &arg, .n_args = 1, .timeout_ms = 1000, .target_sm = 86};
    expected_code = launch.code_va; expected_arg = arg; expected_instruction = *(uint32_t *)code0.cpu;
    assert(omega_gpu_session_launch(&launch, NULL, NULL) == 0);
    arg = 11; launch.code_va = code1.va; expected_code = launch.code_va; expected_arg = arg;
    expected_instruction = *(uint32_t *)code1.cpu;
    assert(omega_gpu_session_launch(&launch, NULL, NULL) == 0);
    /* Reuse the code VA with new content: invalidation remains mandatory. */
    *(uint32_t *)code1.cpu = 0x9abc; expected_instruction = *(uint32_t *)code1.cpu;
    assert(omega_gpu_session_sync_to_gpu(&code1) == 0);
    arg = 19; expected_arg = arg;
    assert(omega_gpu_session_launch(&launch, NULL, NULL) == 0);
    launch.target_sm = 0;
    assert(omega_gpu_session_launch(&launch, NULL, NULL) != 0 && sim.submits == 3);
    omega_gpu_session_free(&code0); omega_gpu_session_free(&code1);
    omega_gpu_session_unlock();
    omega_gpu_session_close();
    puts("Ampere session ordering/argument/code serialization simulation passed (HOST_ONLY; device NOT_RUN)");
    return 0;
}
