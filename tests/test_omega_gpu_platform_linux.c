/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
#include <sys/wait.h>
#include <unistd.h>

/* The three driver symbols the session path needs that the fake (written for the vector engine) lacks. */
static int g_open_fail;
int nvrm_open(Nvrm *rm) {
    if (g_open_fail) { snprintf(rm->err, sizeof rm->err, "fake RM: open refused"); return -1; }
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
