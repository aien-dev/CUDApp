/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* test_omega_gpu_platform_linux: the Linux REFERENCE platform (src/omega_gpu_platform_linux.c, reference only) on the
 * simulated driver (tests/fake_m16_native.c). Host program, hosted build, opens no device.
 *
 * What it checks: the reference claims exactly the session service set (no MMIO, no IRQ: polling is never claimed as
 * an interrupt) and passes the contract check; the hosted session installs it by default; the session makes the same
 * driver calls in the same order as before P07 cut C (channel, then the four session buffers through
 * nvrm_alloc_gpu_uncached, one submit with the C3 tail, both marker waits, close); the reference refuses a GPU-cached
 * buffer, a poll of a buffer that is not HOST_POLLED, freeing the live queue ring, an oversize submit and every handle
 * of a closed device; the driver's own failure text survives an open failure; a marker that never lands latches the
 * session, which then frees nothing and closes nothing. Each case runs in its own child process.
 */
#include "fake_m16_native.h"
#include "nvrm.h"
#include "omega_blackwell_submit.h"
#include "omega_gpu_cmdlist.h"
#include "omega_gpu_mem_cached.h"
#include "omega_gpu_platform_linux.h"
#include "omega_gpu_session.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include "test_out_file.h"

/* The three driver symbols the session path needs that the fake (written for the vector engine) lacks. */
static int g_open_fail;
static int g_discrete;
extern uint64_t fake_vram_free, fake_vram_total;
extern int fake_vram_query_fail, fake_vram_allocs;
int nvrm_open(Nvrm *rm) {
    if (g_open_fail) { snprintf(rm->err, sizeof rm->err, "fake RM: open refused"); return -1; }
    rm->target_sm = g_discrete ? 86 : 121; rm->discrete = (uint8_t)g_discrete;
    fk.device_open = 1; return 0;
}
int nvrm_close(Nvrm *rm) { (void)rm; return 0; }
static int g_retires;
void nvrm_retire(Nvrm *rm, uint32_t upto) { (void)rm; (void)upto; g_retires++; }

static int case_ok;
#define EXPECT(c) do { if (!(c)) { case_ok = 0; printf("  expect failed at line %d: %s\n", __LINE__, #c); } } while (0)

static const uint32_t ARGS[4] = {1, 2, 3, 4};
static OmegaGpuLaunch launch_desc(uint64_t timeout_ms) {
    OmegaGpuLaunch L = { .code_va = 0x200000000ull, .gpr_count = 64, .threads_x = 32, .grid_x = 1, .grid_y = 1,
                         .num_elements = 32, .args = ARGS, .n_args = 4, .timeout_ms = timeout_ms };
    return L;
}

static void c_caps(void) {
    OmegaGpuPlatformCaps caps;
    const OmegaGpuPlatform *p = omega_gpu_platform_linux(&caps);
    EXPECT(p != NULL);
    EXPECT(caps.services == OMEGA_GPU_SVC_REQUIRED_SESSION);
    EXPECT(!(caps.services & OMEGA_GPU_SVC_IRQ) && !(caps.services & OMEGA_GPU_SVC_MMIO));
    EXPECT(p->irq_wait == NULL && p->mmio_read32 == NULL && p->mmio_write32 == NULL);
    EXPECT(omega_gpu_platform_check_for(p, &caps, OMEGA_GPU_SVC_REQUIRED_SESSION) == OMEGA_GPU_PLAT_OK);
    EXPECT(caps.abi_major == 1 && caps.abi_minor == 2);
}

static void c_default_install(void) {
    fake_reset();
    EXPECT(omega_gpu_session_platform() == omega_gpu_platform_linux(NULL)); /* hosted: the reference by default */
    EXPECT(omega_gpu_session_completion_mode() == OMEGA_GPU_COMPLETION_POLLING);
}

static void c_same_driver_calls(void) {
    fake_reset();
    fk.kernel_mode = FK_KERNEL_SKIP;
    omega_gpu_session_lock();
    OmegaGpuSession *s = omega_gpu_session_open();
    EXPECT(s != NULL);
    if (!s) { printf("  %s\n", omega_gpu_session_last_error()); omega_gpu_session_unlock(); return; }
    EXPECT(fk.channel_calls == 1 && fk.uncached_calls == 4 && fk.alloc_calls == 4 && fk.live == 4);
    const int ch = fake_nth(FK_EV_CHANNEL, 1), a1 = fake_nth(FK_EV_ALLOC, 1);
    EXPECT(ch >= 0 && a1 > ch); /* channel first, then the session buffers */
    EXPECT(fk.ev[fake_nth(FK_EV_ALLOC, 1)].a == 0x10000 && fk.ev[fake_nth(FK_EV_ALLOC, 2)].a == 0x1000 &&
           fk.ev[fake_nth(FK_EV_ALLOC, 3)].a == 0x1000 && fk.ev[fake_nth(FK_EV_ALLOC, 4)].a == 0x10000);
    OmegaGpuLaunch L = launch_desc(1000);
    uint32_t mk = 0;
    EXPECT(omega_gpu_session_launch(&L, NULL, &mk) == 0);
    EXPECT(fk.submit_calls == 1 && fk.pb_mem_size == 0x10000);
    EXPECT(fake_count(FK_EV_PB_RELEASE) == 2 && fake_count(FK_EV_PB_FLUSH) == 1); /* the C3 tail */
    EXPECT(fk.marker1_va == s->marker.va && fk.marker2_va == s->marker.va + 0x10);
    EXPECT(fk.sem_va == s->qmd.va + 0x2000 && fk.sem_init_value == OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT);
    EXPECT(mk == OMEGA_BW_MARKER_COMPLETION_PAYLOAD && g_retires == 1);
    omega_gpu_session_unlock();
    omega_gpu_session_close();
    EXPECT(fk.free_calls == 4 && fk.close_calls == 1 && fk.live_at_close == 0 && fk.double_free == 0);
}

static void c_reference_refusals(void) {
    fake_reset();
    OmegaGpuPlatformCaps caps;
    const OmegaGpuPlatform *p = omega_gpu_platform_linux(&caps);
    OmegaGpuDeviceHandle d;
    EXPECT(p->device_open(p->ctx, 0, &d) == OMEGA_GPU_PLAT_OK);
    OmegaGpuDeviceHandle d2;
    EXPECT(p->device_open(p->ctx, 0, &d2) == OMEGA_GPU_PLAT_REFUSED_LIMIT); /* one device context */
    OmegaGpuBufferDesc cached = {.kind = OMEGA_GPU_HANDLE_BUF_DATA, .rights = OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_GPU_WRITE,
                                 .gpu_cache = OMEGA_GPU_CACHE_GPU_CACHED, .bytes = 4096};
    OmegaGpuBufferHandle h; OmegaCpuView cpu; OmegaGpuVa va; OmegaGpuBusAddr bus;
    const int allocs0 = fk.alloc_calls;
    EXPECT(p->dma_alloc(p->ctx, d, &cached, &h, &cpu, &va, &bus) == OMEGA_GPU_PLAT_REFUSED_CACHE_POLICY);
    cached.rights |= OMEGA_GPU_RIGHT_HOST_POLLED; /* polled and cached: the contract refuses it too */
    EXPECT(p->dma_alloc(p->ctx, d, &cached, &h, &cpu, &va, &bus) == OMEGA_GPU_PLAT_REFUSED_CACHE_POLICY);
    EXPECT(fk.alloc_calls == allocs0); /* refused before the driver */
    OmegaGpuBufferDesc data = {.kind = OMEGA_GPU_HANDLE_BUF_DATA, .rights = OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_GPU_WRITE,
                               .gpu_cache = OMEGA_GPU_CACHE_GPU_UNCACHED, .bytes = 4096};
    EXPECT(p->dma_alloc(p->ctx, d, &data, &h, &cpu, &va, &bus) == OMEGA_GPU_PLAT_OK);
    EXPECT(cpu.ptr != NULL && cpu.len >= 4096 && bus.iova == 0);
    uint32_t seen = 0;
    EXPECT(p->poll_word(p->ctx, h, 0, 1, 1, &seen) == OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS); /* not HOST_POLLED */
    OmegaGpuBufferDesc ring = {.kind = OMEGA_GPU_HANDLE_BUF_QUEUE, .rights = OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_READ,
                               .gpu_cache = OMEGA_GPU_CACHE_GPU_UNCACHED, .bytes = 4096};
    OmegaGpuBufferHandle rh; OmegaGpuQueueHandle q;
    EXPECT(p->dma_alloc(p->ctx, d, &ring, &rh, &cpu, &va, &bus) == OMEGA_GPU_PLAT_OK);
    EXPECT(p->queue_open(p->ctx, d, h, &q) == OMEGA_GPU_PLAT_REFUSED_BAD_KIND); /* a DATA buffer is not a ring */
    EXPECT(p->queue_open(p->ctx, d, rh, &q) == OMEGA_GPU_PLAT_OK);
    EXPECT(p->dma_free(p->ctx, rh) == OMEGA_GPU_PLAT_REFUSED_BUSY); /* the open queue still uses it */
    static uint32_t big[2048];
    EXPECT(p->queue_submit(p->ctx, q, big, 2048) == OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH); /* 8 KiB into a 4 KiB ring */
    EXPECT(fk.submit_calls == 0);
    const int frees0 = fk.free_calls;
    EXPECT(p->device_close(p->ctx, d) == OMEGA_GPU_PLAT_OK);
    /* every handle of the closed device is stale */
    EXPECT(p->dma_free(p->ctx, h) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(p->sync_cpu_to_gpu(p->ctx, h, 0, 4) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(p->queue_submit(p->ctx, q, big, 4) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(p->dma_alloc(p->ctx, d, &data, &h, &cpu, &va, &bus) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(fk.free_calls == frees0);
    EXPECT(strstr(p->diag_text(p->ctx), "STALE") != NULL);
}

static void c_open_failure_text(void) {
    fake_reset();
    fk.channel_fail = 1;
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() == NULL);
    EXPECT(strncmp(omega_gpu_session_last_error(), "m16_native_create_channel: ", 27) == 0);
    EXPECT(strstr(omega_gpu_session_last_error(), FK_TEXT) != NULL); /* the driver's own text survives */
    fk.channel_fail = 0;
    g_open_fail = 1;
    EXPECT(omega_gpu_session_open() == NULL);
    EXPECT(strcmp(omega_gpu_session_last_error(), "nvrm_open: fake RM: open refused") == 0);
    g_open_fail = 0;
    EXPECT(omega_gpu_session_open() != NULL && !omega_gpu_session_is_blocked());
    omega_gpu_session_unlock();
    omega_gpu_session_close();
}

static void c_marker_never_latches(void) {
    fake_reset();
    fk.kernel_mode = FK_KERNEL_SKIP;
    fk.marker_mode = FK_SYNC_NEVER;
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() != NULL);
    OmegaGpuLaunch L = launch_desc(5);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == -1);
    EXPECT(omega_gpu_session_is_blocked());
    EXPECT(fk.wait_calls == 5 && fk.clock_ms == 5); /* five 1 ms poll slices, then uncertain */
    EXPECT(g_retires == 0);
    omega_gpu_session_unlock();
    omega_gpu_session_close();
    EXPECT(fk.free_calls == 0 && fk.close_calls == 0 && fk.live == 4); /* parked: nothing released, nothing closed */
    OmegaGpuParkStats ps; omega_gpu_session_parked(&ps);
    EXPECT(ps.buffers == 4);
}

/* omega #457: the L2_SYSMEM_NCOH_INVALIDATE prefix, MEM_OP_A..D = 0, 0, 0, 0x11 << 27 (clc96f.h:30-73). The fake logs
 * every MEM_OP as FK_EV_PB_FLUSH with a = the operation word. */
#define L2_INVAL_OP (0x11u << 27)
static int count_l2_inval(void) {
    int n = 0;
    for (int i = 0; i < fk.nev; i++) n += fk.ev[i].kind == FK_EV_PB_FLUSH && fk.ev[i].a == L2_INVAL_OP;
    return n;
}

static void c_l2_inval_prefix_once(void) {
    fake_reset();
    fk.kernel_mode = FK_KERNEL_SKIP;
    OmegaGpuPlatformCaps caps;
    (void)omega_gpu_platform_linux(&caps);
    EXPECT(caps.submit_prefix_words == 5u); /* the cost the platform publishes is the prefix it writes */
    omega_gpu_session_lock();
    OmegaGpuSession *s = omega_gpu_session_open();
    EXPECT(s != NULL);
    if (!s) { omega_gpu_session_unlock(); return; }
    const OmegaGpuPlatform *p = omega_gpu_session_platform();
    OmegaGpuMem cached;
    EXPECT(omega_gpu_session_alloc_cached(4096, &cached) == 0);
    OmegaGpuLaunch L = launch_desc(1000);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == 0);
    EXPECT(count_l2_inval() == 0); /* no host write into a GPU-cached buffer yet: no prefix */
    const size_t plain = fk.pb_len;
    EXPECT(p->sync_cpu_to_gpu(p->ctx, s->marker.handle, 0, 4) == OMEGA_GPU_PLAT_OK); /* an uncached buffer: no prefix */
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == 0);
    EXPECT(count_l2_inval() == 0 && fk.pb_len == plain);
    EXPECT(p->sync_cpu_to_gpu(p->ctx, cached.handle, 0, 4096) == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == 0);
    EXPECT(count_l2_inval() == 1); /* exactly once, at the head of the next submission */
    EXPECT(fk.pb_len == plain + 5u);
    EXPECT(fk.pb[0] == ((2u << 28) | (4u << 16) | (0x0028u >> 2)) && fk.pb[1] == 0 && fk.pb[2] == 0 && fk.pb[3] == 0 &&
           fk.pb[4] == L2_INVAL_OP);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == 0);
    EXPECT(count_l2_inval() == 1 && fk.pb_len == plain); /* consumed: the launch after it has none */
    EXPECT(p->sync_gpu_to_cpu(p->ctx, cached.handle, 0, 4096) == OMEGA_GPU_PLAT_OK); /* GPU to CPU is no host write */
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == 0);
    EXPECT(count_l2_inval() == 1 && fk.pb_len == plain);
    EXPECT(fk.pb_parse_error == 0 && fk.ev_overflow == 0);
    omega_gpu_session_unlock();
}

/* Fable review of omega #459: a stamped standalone list of >= 1167 launches filled one submission to 16381 words; with
 * the 5-word prefix pending that is 16386 > 16384 ring words, refused BAD_LENGTH on every retry. The command list must
 * budget the prefix the platform publishes (caps.submit_prefix_words). */
/* 2048 launches (about 14 ring words each) fill the 16384-word ring so the list splits into exactly 2 submissions.
 * OMEGA_GPU_CMDLIST_MAX_LAUNCHES is 4096 since L312, which would split into 4, so the count is pinned here. */
#define NEAR_CAP_LAUNCHES 2048u
static void c_cmdlist_near_cap_with_l2_inval_pending(void) {
    fake_reset();
    fk.kernel_mode = FK_KERNEL_SKIP;
    fk.alloc_max = 4u << 20; /* the 2048-launch bank and QMD buffers are 2 MiB each */
    OmegaGpuCmdList *cl = NULL;
    EXPECT(omega_gpu_cmdlist_create(NEAR_CAP_LAUNCHES, &cl) == 0);
    if (!cl) { printf("  %s\n", omega_gpu_session_last_error()); return; }
    omega_gpu_session_lock();
    OmegaGpuLaunch L = launch_desc(1000);
    for (uint32_t i = 0; i < NEAR_CAP_LAUNCHES; i++) EXPECT(omega_gpu_cmdlist_add(cl, &L) == (int)i);
    const OmegaGpuPlatform *p = omega_gpu_session_platform();
    OmegaGpuMem cached;
    EXPECT(omega_gpu_session_alloc_cached(4096, &cached) == 0);
    EXPECT(p->sync_cpu_to_gpu(p->ctx, cached.handle, 0, 4096) == OMEGA_GPU_PLAT_OK); /* the prefix is now pending */
    const int submits0 = fk.submit_calls;
    OmegaGpuCmdListOpts o = { .mode = OMEGA_GPU_CMDLIST_PCAS, .barrier = OMEGA_GPU_CMDLIST_BARRIER_WFI, .timeout_ms = 1000, .stamps = 1 };
    OmegaGpuCmdListStats st;
    const int rc = omega_gpu_cmdlist_submit(cl, &o, &st);
    if (rc != 0) printf("  cmdlist submit: %s (platform: %s)\n", omega_gpu_session_last_error(), p->diag_text(p->ctx));
    EXPECT(rc == 0);
    EXPECT(st.sems_done == NEAR_CAP_LAUNCHES && st.first_missing == -1);
    EXPECT(st.submits == 2 && fk.submit_calls - submits0 == 2);
    EXPECT(count_l2_inval() == 1); /* the first submission only */
    for (int i = 0; i < fk.nev; i++)
        if (fk.ev[i].kind == FK_EV_SUBMIT) EXPECT(fk.ev[i].a * 4u <= fk.pb_mem_size); /* prefix included, inside the ring */
    EXPECT(fk.pb_parse_error == 0 && fk.ev_overflow == 0);
    EXPECT(!omega_gpu_session_is_blocked() && !omega_gpu_cmdlist_is_latched());
    omega_gpu_session_unlock();
}

/* Developer bar D1 (docs/PAIN_POINTS.md): on the GB10 every CUDA++ buffer is the machine's own memory, and running
 * it out freezes the whole box. The reference refuses an allocation that would leave less than the reserve free,
 * before the driver is called, with a sentence that names the bytes asked, the bytes free and the setting.
 * omega_gpu_platform_linux_test_set_meminfo points the platform at a fake /proc/meminfo; CUDAPP_MEM_RESERVE_MIB sets the
 * reserve (0 = off). */
static void write_meminfo(const char *path, unsigned long long avail_kb) {
    FILE *f = test_out_open(path);
    if (!f) { case_ok = 0; return; }
    fprintf(f, "MemTotal:       127535484 kB\nMemFree:          100000 kB\nMemAvailable:   %llu kB\n", avail_kb);
    fclose(f);
}
static void c_memory_reserve(void) {
    fake_reset();
    fk.alloc_max = 8u << 20; /* the fake driver may hand out up to 8 MiB, so only our check can refuse */
    char path[] = "/tmp/cudapp-meminfo-XXXXXX";
    int fd = mkstemp(path); if (fd >= 0) close(fd);
    write_meminfo(path, 4097ull * 1024ull); /* 4097 MiB free: 1 MiB above the default 4096 MiB reserve */
    EXPECT(omega_gpu_platform_linux_test_set_meminfo(path) == 0);
    unsetenv("CUDAPP_MEM_RESERVE_MIB");
    OmegaGpuPlatformCaps caps;
    const OmegaGpuPlatform *p = omega_gpu_platform_linux(&caps);
    OmegaGpuDeviceHandle d;
    EXPECT(p->device_open(p->ctx, 0, &d) == OMEGA_GPU_PLAT_OK);
    OmegaGpuBufferDesc small = {.kind = OMEGA_GPU_HANDLE_BUF_DATA, .rights = OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_GPU_WRITE,
                                .gpu_cache = OMEGA_GPU_CACHE_GPU_UNCACHED, .bytes = 4096};
    OmegaGpuBufferDesc big = small; big.bytes = 2u << 20;
    OmegaGpuBufferHandle h; OmegaCpuView cpu; OmegaGpuVa va; OmegaGpuBusAddr bus;
    EXPECT(p->dma_alloc(p->ctx, d, &small, &h, &cpu, &va, &bus) == OMEGA_GPU_PLAT_OK); /* leaves more than the reserve */
    const int allocs0 = fk.alloc_calls;
    EXPECT(p->dma_alloc(p->ctx, d, &big, &h, &cpu, &va, &bus) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    EXPECT(fk.alloc_calls == allocs0); /* refused before the driver */
    const char *t = p->diag_text(p->ctx);
    EXPECT(strstr(t, "dma_alloc: refused 2.0 MiB") != NULL);
    EXPECT(strstr(t, "4.0 GiB of memory is free") != NULL);
    EXPECT(strstr(t, "CUDAPP_MEM_RESERVE_MIB=4096") != NULL);
    EXPECT(strstr(t, "freezes the machine") != NULL);
    printf("  refusal: %s\n", t);
    setenv("CUDAPP_MEM_RESERVE_MIB", "0", 1); /* the reserve can be turned off */
    EXPECT(p->dma_alloc(p->ctx, d, &big, &h, &cpu, &va, &bus) == OMEGA_GPU_PLAT_OK);
    setenv("CUDAPP_MEM_RESERVE_MIB", "1", 1); /* and set: 1 MiB reserve, 2 MiB fits in 4097 MiB */
    EXPECT(p->dma_alloc(p->ctx, d, &big, &h, &cpu, &va, &bus) == OMEGA_GPU_PLAT_OK);
    unsetenv("CUDAPP_MEM_RESERVE_MIB");
    EXPECT(p->device_close(p->ctx, d) == OMEGA_GPU_PLAT_OK);
    /* through the session, the same sentence reaches omega_gpu_session_last_error() */
    EXPECT(omega_gpu_platform_linux_test_set_meminfo(path) == 0);
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() != NULL);
    OmegaGpuMem m;
    EXPECT(omega_gpu_session_alloc(2u << 20, &m) != 0);
    EXPECT(strstr(omega_gpu_session_last_error(), "dma_alloc: refused 2.0 MiB") != NULL);
    printf("  session: %s\n", omega_gpu_session_last_error());
    EXPECT(omega_gpu_session_alloc(4096, &m) == 0);
    omega_gpu_session_unlock();
    unlink(path);
}

/* Audit Q6 (CudaAuditFable REPORT.md): caps.max_inflight is 1, so the reference refuses a second submission while
 * the first is unretired, before the ring is overwritten or the doorbell rung; queue_retire then releases it. */
static void c_submit_refused_while_unretired(void) {
    fake_reset();
    fk.kernel_mode = FK_KERNEL_SKIP;
    OmegaGpuPlatformCaps caps;
    const OmegaGpuPlatform *p = omega_gpu_platform_linux(&caps);
    EXPECT(caps.max_inflight == 1u);
    omega_gpu_session_lock();
    OmegaGpuSession *s = omega_gpu_session_open();
    EXPECT(s != NULL);
    if (!s) { omega_gpu_session_unlock(); return; }
    static const uint32_t NOP[2] = { 0, 0 };
    const int submits0 = fk.submit_calls, retires0 = g_retires;
    EXPECT(p->queue_submit(p->ctx, s->queue, NOP, 2) == OMEGA_GPU_PLAT_OK);
    EXPECT(p->queue_submit(p->ctx, s->queue, NOP, 2) == OMEGA_GPU_PLAT_REFUSED_BUSY); /* first one not retired */
    EXPECT(fk.submit_calls - submits0 == 1); /* the refusal reached no driver call: no copy, no doorbell */
    EXPECT(strstr(p->diag_text(p->ctx), "queue_submit") != NULL);
    EXPECT(p->queue_retire(p->ctx, s->queue) == OMEGA_GPU_PLAT_OK);
    EXPECT(g_retires - retires0 == 1);
    EXPECT(p->queue_retire(p->ctx, s->queue) == OMEGA_GPU_PLAT_OK); /* nothing in flight: no driver retire */
    EXPECT(g_retires - retires0 == 1);
    EXPECT(p->queue_submit(p->ctx, s->queue, NOP, 2) == OMEGA_GPU_PLAT_OK);
    EXPECT(fk.submit_calls - submits0 == 2);
    EXPECT(p->queue_retire(p->ctx, s->queue) == OMEGA_GPU_PLAT_OK);
    OmegaGpuLaunch L = launch_desc(1000); /* the session's own launch path still runs submit, wait, retire */
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == 0);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == 0);
    omega_gpu_session_unlock();
}

/* Audit Q6 on an L136 extra queue: its ring is also rewritten from offset 0, and caps.max_inflight is per queue, so
 * a second submission on the extra queue while the first is unretired is refused REFUSED_BUSY before any copy or
 * doorbell; the two queues' in-flight counts are independent; queue_retire on the extra queue releases it. */
static void c_extra_queue_submit_refused_while_unretired(void) {
    fake_reset();
    fk.kernel_mode = FK_KERNEL_SKIP;
    fk.extra_channels = 1;
    setenv("AIEN_OMEGA_GPU_CHANNELS", "2", 1);
    OmegaGpuPlatformCaps caps;
    const OmegaGpuPlatform *p = omega_gpu_platform_linux(&caps);
    omega_gpu_session_lock();
    OmegaGpuSession *s = omega_gpu_session_open();
    EXPECT(s != NULL);
    OmegaGpuQueueHandle q1;
    const int xq = s ? omega_gpu_session_extra_queue(1, &q1) : -1;
    EXPECT(xq == 0);
    if (!s || xq != 0) { printf("  session: %s\n", omega_gpu_session_last_error()); omega_gpu_session_unlock(); return; }
    static const uint32_t NOP[2] = { 0, 0 };
    const int submits0 = fk.submit_calls;
    EXPECT(p->queue_submit(p->ctx, q1, NOP, 2) == OMEGA_GPU_PLAT_OK);
    EXPECT(fk.chan_submit_calls == 1);
    EXPECT(p->queue_submit(p->ctx, q1, NOP, 2) == OMEGA_GPU_PLAT_REFUSED_BUSY); /* first one not retired */
    EXPECT(fk.chan_submit_calls == 1); /* the refusal reached no driver call: no copy, no doorbell */
    EXPECT(strstr(p->diag_text(p->ctx), "queue_submit") != NULL);
    EXPECT(p->queue_submit(p->ctx, s->queue, NOP, 2) == OMEGA_GPU_PLAT_OK); /* queue 0 is independent */
    EXPECT(p->queue_submit(p->ctx, s->queue, NOP, 2) == OMEGA_GPU_PLAT_REFUSED_BUSY);
    EXPECT(fk.submit_calls - submits0 == 1);
    EXPECT(p->queue_retire(p->ctx, s->queue) == OMEGA_GPU_PLAT_OK);
    EXPECT(p->queue_submit(p->ctx, q1, NOP, 2) == OMEGA_GPU_PLAT_REFUSED_BUSY); /* retiring queue 0 does not free q1 */
    EXPECT(p->queue_retire(p->ctx, q1) == OMEGA_GPU_PLAT_OK);
    EXPECT(fk.chan_retire_calls == 1);
    EXPECT(p->queue_retire(p->ctx, q1) == OMEGA_GPU_PLAT_OK); /* nothing in flight: no driver retire */
    EXPECT(fk.chan_retire_calls == 1);
    EXPECT(p->queue_submit(p->ctx, q1, NOP, 2) == OMEGA_GPU_PLAT_OK);
    EXPECT(fk.chan_submit_calls == 2);
    EXPECT(p->queue_retire(p->ctx, q1) == OMEGA_GPU_PLAT_OK);
    omega_gpu_session_unlock();
}

/* Audit Q6 / top-10 item 4: a command list whose completion is uncertain must latch the process without queueing
 * anything behind the stuck list. Before the fix force_latch launched a 1 ms nop, which copied its methods over the
 * ring the stuck list may still be reading and rang a second doorbell. */
static void c_cmdlist_timeout_latches_without_launch(void) {
    fake_reset();
    fk.kernel_mode = FK_KERNEL_SKIP;
    OmegaGpuCmdList *cl = NULL;
    EXPECT(omega_gpu_cmdlist_create(4, &cl) == 0);
    if (!cl) { printf("  %s\n", omega_gpu_session_last_error()); return; }
    omega_gpu_session_lock();
    OmegaGpuLaunch L = launch_desc(1000);
    EXPECT(omega_gpu_cmdlist_add(cl, &L) == 0);
    fk.marker_mode = FK_SYNC_NEVER;
    const int submits0 = fk.submit_calls, retires0 = g_retires;
    OmegaGpuCmdListOpts o = { .mode = OMEGA_GPU_CMDLIST_PCAS, .barrier = OMEGA_GPU_CMDLIST_BARRIER_WFI, .timeout_ms = 5 };
    OmegaGpuCmdListStats st;
    EXPECT(omega_gpu_cmdlist_submit(cl, &o, &st) == -1);
    printf("  submits %d, error: %s\n", fk.submit_calls - submits0, omega_gpu_session_last_error());
    EXPECT(fk.submit_calls - submits0 == 1); /* the stuck list only: nothing queued behind it */
    EXPECT(g_retires == retires0);
    EXPECT(omega_gpu_cmdlist_is_latched());
    EXPECT(omega_gpu_session_is_blocked());
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == -1); /* every later launch is refused */
    EXPECT(fk.submit_calls - submits0 == 1);
    omega_gpu_session_unlock();
}

/* Audit F15: timeout_ms 0 made the session submit and then latch after a single poll. It is refused as an argument,
 * before anything is submitted, and the session stays usable. */
static void c_zero_timeout_refused(void) {
    fake_reset();
    fk.kernel_mode = FK_KERNEL_SKIP;
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() != NULL);
    const int submits0 = fk.submit_calls;
    OmegaGpuLaunch L = launch_desc(0);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == -1);
    EXPECT(fk.submit_calls == submits0);
    EXPECT(!omega_gpu_session_is_blocked());
    EXPECT(strcmp(omega_gpu_session_last_error(), "launch arguments") == 0);
    L = launch_desc(1000);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == 0);
    omega_gpu_session_unlock();
}

/* Audit F9: CUDAPP_MEM_RESERVE_MIB = 2^44 shifted left by 20 is 2^64, which wrapped to a 0 reserve and allowed
 * everything; a MemAvailable past 2^54 kB wrapped the same way and refused everything. Both saturate. */
static void c_reserve_overflow(void) {
    fake_reset();
    fk.alloc_max = 8u << 20;
    char path[] = "/tmp/cudapp-meminfo-XXXXXX";
    int fd = mkstemp(path); if (fd >= 0) close(fd);
    write_meminfo(path, 4097ull * 1024ull);
    EXPECT(omega_gpu_platform_linux_test_set_meminfo(path) == 0);
    setenv("CUDAPP_MEM_RESERVE_MIB", "17592186044416", 1); /* 2^44 MiB */
    const OmegaGpuPlatform *p = omega_gpu_platform_linux(NULL);
    OmegaGpuDeviceHandle d;
    EXPECT(p->device_open(p->ctx, 0, &d) == OMEGA_GPU_PLAT_OK);
    OmegaGpuBufferDesc small = {.kind = OMEGA_GPU_HANDLE_BUF_DATA, .rights = OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_GPU_WRITE,
                                .gpu_cache = OMEGA_GPU_CACHE_GPU_UNCACHED, .bytes = 4096};
    OmegaGpuBufferHandle h; OmegaCpuView cpu; OmegaGpuVa va; OmegaGpuBusAddr bus;
    const int allocs0 = fk.alloc_calls;
    EXPECT(p->dma_alloc(p->ctx, d, &small, &h, &cpu, &va, &bus) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    EXPECT(fk.alloc_calls == allocs0);
    printf("  refusal: %s\n", p->diag_text(p->ctx));
    unsetenv("CUDAPP_MEM_RESERVE_MIB");
    write_meminfo(path, 18014398509481984ull); /* 2^54 kB: 2^64 bytes, past uint64 */
    EXPECT(p->dma_alloc(p->ctx, d, &small, &h, &cpu, &va, &bus) == OMEGA_GPU_PLAT_OK);
    EXPECT(p->device_close(p->ctx, d) == OMEGA_GPU_PLAT_OK);
    unlink(path);
}

static void c_discrete_vram_reserve(void) {
    fake_reset(); g_discrete = 1;
    setenv("CUDAPP_MEM_RESERVE_MIB", "0", 1);
    setenv("CUDAPP_GPU_MIN_FREE_BYTES", "8192", 1);
    EXPECT(omega_gpu_platform_linux_test_set_meminfo("/nonexistent/host-memory-must-not-be-used") == 0);
    fake_vram_free = 4096; fake_vram_total = 8ull << 30;
    const OmegaGpuPlatform *p = omega_gpu_platform_linux(NULL);
    OmegaGpuDeviceHandle d;
    EXPECT(p->device_open(p->ctx, 0, &d) == 0);
    EXPECT(omega_gpu_platform_linux_target_sm() == 86);
    OmegaGpuBufferDesc desc = {.kind = OMEGA_GPU_HANDLE_BUF_DATA,
        .rights = OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_GPU_WRITE,
        .gpu_cache = OMEGA_GPU_CACHE_GPU_UNCACHED, .bytes = 4096};
    OmegaGpuBufferHandle h; OmegaCpuView cpu; OmegaGpuVa va;
    const int before = fk.alloc_calls;
    EXPECT(p->dma_alloc(p->ctx, d, &desc, &h, &cpu, &va, NULL) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    EXPECT(fk.alloc_calls == before && fake_vram_allocs == 0 && fake_vram_free == 4096);
    EXPECT(strstr(p->diag_text(p->ctx), "host RAM is not VRAM") != NULL);
    /* Exact byte override is read again, not cached at open. */
    setenv("CUDAPP_GPU_MIN_FREE_BYTES", "0", 1);
    EXPECT(p->dma_alloc(p->ctx, d, &desc, &h, &cpu, &va, NULL) == 0);
    EXPECT(fake_vram_allocs == 1);
    fk.free_fail_nth = fk.free_calls + 1;
    EXPECT(p->dma_free(p->ctx, h) == OMEGA_GPU_PLAT_REFUSED_DEVICE);
    fk.free_fail_nth = 0;
    EXPECT(p->dma_free(p->ctx, h) == 0); /* handle survived failed driver cleanup */
    fake_vram_query_fail = 1;
    setenv("CUDAPP_GPU_MIN_FREE_BYTES", "1", 1);
    const int allocated = fk.alloc_calls;
    EXPECT(p->dma_alloc(p->ctx, d, &desc, &h, &cpu, &va, NULL) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    EXPECT(fk.alloc_calls == allocated);
    fake_vram_query_fail = 0;
    setenv("CUDAPP_GPU_MIN_FREE_BYTES", "18446744073709551616", 1);
    EXPECT(p->dma_alloc(p->ctx, d, &desc, &h, &cpu, &va, NULL) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    EXPECT(fk.alloc_calls == allocated);
    EXPECT(p->device_close(p->ctx, d) == 0);
    EXPECT(omega_gpu_platform_linux_test_set_meminfo(NULL) == 0);
}

static void c_ampere_target_refusal(void) {
    fake_reset(); g_discrete = 1;
    setenv("CUDAPP_MEM_RESERVE_MIB", "0", 1);
    setenv("CUDAPP_GPU_MIN_FREE_BYTES", "0", 1);
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() != NULL);
    EXPECT(omega_gpu_session_target_sm() == 86);
    const int before = fk.submit_calls;
    OmegaGpuLaunch L = launch_desc(1000);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) != 0);
    L.target_sm = 121;
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) != 0);
    L.target_sm = 999;
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) != 0);
    EXPECT(fk.submit_calls == before && !omega_gpu_session_is_blocked());
    omega_gpu_session_unlock();
}

/* CodeQL cpp/path-injection #40: the meminfo file is set by code (omega_gpu_platform_linux_test_set_meminfo), never by
 * the environment, and is honoured only when it opens, without following a final symlink and without blocking, to a
 * regular file; anything else falls back to the real /proc/meminfo. The reserve is 2^30 MiB (1 PiB), which the real
 * /proc/meminfo cannot cover, so a fallback shows as REFUSED_LIMIT, while the fake file claims 2^52 kB (4 EiB) free and
 * shows as OK when honoured. The former CUDAPP_MEMINFO variable naming that same fake file must change nothing.
 * alarm() turns an open that blocks (the FIFO, before O_NONBLOCK) into a failed case instead of a hung run. */
static void c_meminfo_test_path(void) {
    fake_reset();
    fk.alloc_max = 8u << 20;
    alarm(10);
    char dir[] = "/tmp/cudapp-meminfo-dir-XXXXXX";
    if (!mkdtemp(dir)) { case_ok = 0; return; }
    char fake[64], lnk[64], fifo[64], sub[64], missing[64];
    snprintf(fake, sizeof fake, "%s/meminfo", dir);
    snprintf(lnk, sizeof lnk, "%s/link", dir);
    snprintf(fifo, sizeof fifo, "%s/fifo", dir);
    snprintf(sub, sizeof sub, "%s/sub", dir);
    snprintf(missing, sizeof missing, "%s/missing", dir);
    write_meminfo(fake, 1ull << 52);
    EXPECT(symlink(fake, lnk) == 0);
    EXPECT(mkfifo(fifo, 0600) == 0);
    EXPECT(mkdir(sub, 0700) == 0);
    setenv("CUDAPP_MEM_RESERVE_MIB", "1073741824", 1); /* 2^30 MiB */
    const OmegaGpuPlatform *p = omega_gpu_platform_linux(NULL);
    OmegaGpuDeviceHandle d;
    EXPECT(p->device_open(p->ctx, 0, &d) == OMEGA_GPU_PLAT_OK);
    OmegaGpuBufferDesc small = {.kind = OMEGA_GPU_HANDLE_BUF_DATA, .rights = OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_GPU_WRITE,
                                .gpu_cache = OMEGA_GPU_CACHE_GPU_UNCACHED, .bytes = 4096};
    OmegaGpuBufferHandle h; OmegaCpuView cpu; OmegaGpuVa va; OmegaGpuBusAddr bus;
    const struct { const char *path; int want; const char *what; } t[] = {
        {fake, OMEGA_GPU_PLAT_OK, "regular file (honoured)"},
        {lnk, OMEGA_GPU_PLAT_REFUSED_LIMIT, "symlink to that file"},
        {fifo, OMEGA_GPU_PLAT_REFUSED_LIMIT, "FIFO"},
        {sub, OMEGA_GPU_PLAT_REFUSED_LIMIT, "directory"},
        {missing, OMEGA_GPU_PLAT_REFUSED_LIMIT, "missing file"},
        {"", OMEGA_GPU_PLAT_REFUSED_LIMIT, "empty (restores /proc/meminfo)"},
        {fake, OMEGA_GPU_PLAT_OK, "regular file again (honoured)"},
        {NULL, OMEGA_GPU_PLAT_REFUSED_LIMIT, "NULL (restores /proc/meminfo)"},
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        EXPECT(omega_gpu_platform_linux_test_set_meminfo(t[i].path) == 0);
        const int rc = p->dma_alloc(p->ctx, d, &small, &h, &cpu, &va, &bus);
        EXPECT(rc == t[i].want);
        printf("  test meminfo %s: %s\n", t[i].what,
               rc == OMEGA_GPU_PLAT_OK ? "read" : rc == OMEGA_GPU_PLAT_REFUSED_LIMIT ? "/proc/meminfo read, refused" : "other");
    }
    /* the environment no longer chooses the file */
    setenv("CUDAPP_MEMINFO", fake, 1);
    int rc = p->dma_alloc(p->ctx, d, &small, &h, &cpu, &va, &bus);
    EXPECT(rc == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    printf("  env CUDAPP_MEMINFO=<fake>: %s\n", rc == OMEGA_GPU_PLAT_REFUSED_LIMIT ? "ignored, refused" : "HONOURED");
    unsetenv("CUDAPP_MEMINFO");
    /* a path too long for the buffer is refused and leaves the setting unchanged */
    char longp[4200];
    memset(longp, 'a', sizeof longp - 1);
    longp[0] = '/'; longp[sizeof longp - 1] = '\0';
    EXPECT(omega_gpu_platform_linux_test_set_meminfo(fake) == 0);
    EXPECT(omega_gpu_platform_linux_test_set_meminfo(longp) == -1);
    EXPECT(p->dma_alloc(p->ctx, d, &small, &h, &cpu, &va, &bus) == OMEGA_GPU_PLAT_OK);
    unsetenv("CUDAPP_MEM_RESERVE_MIB");
    EXPECT(p->device_close(p->ctx, d) == OMEGA_GPU_PLAT_OK);
    unlink(lnk); unlink(fifo); unlink(fake); rmdir(sub); rmdir(dir);
    alarm(0);
}

typedef struct { const char *id; void (*fn)(void); } Case;
static const Case CASES[] = {
    {"LINUX_REF_CAPS_NO_MMIO_NO_IRQ", c_caps},
    {"HOSTED_DEFAULT_IS_LINUX_REFERENCE", c_default_install},
    {"SAME_DRIVER_CALLS_AND_ORDER_AS_BEFORE", c_same_driver_calls},
    {"REFERENCE_REFUSALS_CACHED_POLLED_RING_STALE", c_reference_refusals},
    {"OPEN_FAILURE_KEEPS_DRIVER_TEXT", c_open_failure_text},
    {"MARKER_NEVER_LATCHES_AND_PARKS", c_marker_never_latches},
    {"L2_INVAL_PREFIX_ONCE_AFTER_CACHED_HOST_WRITE", c_l2_inval_prefix_once},
    {"CMDLIST_NEAR_CAP_SUBMITS_WITH_L2_INVAL_PENDING", c_cmdlist_near_cap_with_l2_inval_pending},
    {"D1_MEMORY_RESERVE_REFUSES_BEFORE_THE_DRIVER", c_memory_reserve},
    {"QUEUE_SUBMIT_REFUSED_WHILE_UNRETIRED", c_submit_refused_while_unretired},
    {"EXTRA_QUEUE_SUBMIT_REFUSED_WHILE_UNRETIRED", c_extra_queue_submit_refused_while_unretired},
    {"CMDLIST_TIMEOUT_LATCHES_WITHOUT_LAUNCH_BEHIND_IT", c_cmdlist_timeout_latches_without_launch},
    {"ZERO_TIMEOUT_REFUSED_BEFORE_SUBMIT", c_zero_timeout_refused},
    {"D1_RESERVE_AND_MEMINFO_OVERFLOW_SATURATE", c_reserve_overflow},
    {"AMPERE_D1_USES_VRAM_REFUSES_BEFORE_RM_ALLOC", c_discrete_vram_reserve},
    {"AMPERE_REJECTS_IMPLICIT_OR_MISMATCHED_LAUNCH_TARGET", c_ampere_target_refusal},
    {"D1_MEMINFO_TEST_PATH_REGULAR_ENV_IGNORED", c_meminfo_test_path},
};

int main(void) {
    int pass = 0, fail = 0;
    for (size_t i = 0; i < sizeof CASES / sizeof CASES[0]; i++) {
        fflush(stdout);
        pid_t pid = fork();
        if (pid == 0) { case_ok = 1; CASES[i].fn(); fflush(stdout); _exit(case_ok ? 0 : 1); }
        int st = 0;
        waitpid(pid, &st, 0);
        const int ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;
        printf("%s %s\n", ok ? "PASS" : "FAIL", CASES[i].id);
        if (ok) pass++; else fail++;
    }
    printf("test_omega_gpu_platform_linux (reference only, simulated driver): %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
