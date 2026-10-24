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

/* test_omega_gpu_session_platform: the portable GPU session (native v1 P07 cut C, session part; aienos#286) on a
 * FAKE OmegaGpuPlatform. Host program; links the FREESTANDING partial-link object (build/gpu_fs/
 * omega_gpu_freestanding.o: session, matmul and attention APIs, serving, wait, codegen, platform contract), so what
 * runs here is the code the native build would run. Opens no device, touches no GPU.
 *
 * The fake device "completes" a launch at submit time by writing the intermediate semaphore DONE and both marker
 * payloads, or (hang mode) writes nothing. Its clock advances only when asked (plus 1 us per read), so timeouts are
 * exact and instant. Each case runs in its own child process because an uncertain completion latches the process
 * for good (that is the rule under test).
 *
 * Cases: platform install refusals; freestanding build has no default platform; open/launch/close with typed
 * buffers, explicit rights and GPU-uncached policy (host-polled marker); polling only (irq never called, mode is
 * POLLING); cancel before submit (nothing submitted, not latched); cancel after submit (latched, parked); timeout
 * (latched, session buffers parked, later frees park, close/reopen/install refused, nothing released); stale handle
 * refusal after close and reopen; close/reopen (new device generation, opens 2); serving reservation refusal (no
 * platform allocation); matmul API code buffer is a CODE buffer encoded in place; wait helper refuses without a clock;
 * the bounded formatter matches snprintf and never writes past its cap.
 */
#include "omega_gpu_session.h"
#include "omega_gpu_matmul_api.h"
#include "omega_gpu_wait.h"
#include "omega_gpu_fmt.h"
#include "omega_blackwell_submit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int case_ok;
#define EXPECT(c) do { if (!(c)) { case_ok = 0; printf("  expect failed at line %d: %s\n", __LINE__, #c); } } while (0)

/* ---- the fake platform ---- */
#define NB OMEGA_GPU_HANDLE_SLOTS
enum { MODE_COMPLETE = 0, MODE_HANG = 1 };
static struct {
    OmegaGpuHandleTable t;
    void *mem[NB];
    OmegaGpuBufferDesc desc[NB];
    uint32_t dev_of[NB];      /* device generation a buffer belongs to */
    OmegaGpuDeviceHandle dev;
    OmegaGpuQueueHandle q;
    OmegaGpuBufferHandle ring;
    int mode;
    int cancel_on_poll;       /* poll_word calls omega_gpu_session_cancel() on its first call */
    uint64_t ns;
    unsigned allocs, frees, free_refusals, submits, polls, retires, syncs, irq_calls, dev_opens, dev_closes;
    unsigned locks_held;
    unsigned cached_requests; /* buffers asked GPU-cached (must stay 0) */
    unsigned allocs_unlocked; /* allocations without the session lock (must stay 0) */
    int fail_open;
} fk;

static uint64_t f_now_ns(void *c) { (void)c; fk.ns += 1000u; return fk.ns; }
static uint64_t f_now_ms(void *c) { return f_now_ns(c) / 1000000u; }
static int f_wait(void *c, uint64_t ms) { (void)c; fk.ns += ms * 1000000u; return 0; }
static int f_lock(void *c, uint32_t id) { (void)c; if (id != OMEGA_GPU_SESSION_LOCK_ID) return OMEGA_GPU_PLAT_REFUSED_ARGS; fk.locks_held++; return 0; }
static int f_unlock(void *c, uint32_t id) { (void)c; if (id != OMEGA_GPU_SESSION_LOCK_ID || !fk.locks_held) return OMEGA_GPU_PLAT_REFUSED_ARGS; fk.locks_held--; return 0; }
static const char *f_diag(void *c) { (void)c; return fk.fail_open ? "fake device: no GB10 here" : ""; }

static int f_dev_open(void *c, uint32_t ord, OmegaGpuDeviceHandle *out) {
    (void)c;
    if (fk.fail_open) return OMEGA_GPU_PLAT_REFUSED_DEVICE;
    if (ord != 0 || fk.dev.generation) return OMEGA_GPU_PLAT_REFUSED_LIMIT;
    int rc = omega_gpu_handles_open(&fk.t, OMEGA_GPU_HANDLE_DEVICE, 0, 1, 0, &out->index, &out->generation);
    if (rc == 0) { fk.dev = *out; fk.dev_opens++; }
    return rc;
}
static int f_dev_close(void *c, OmegaGpuDeviceHandle d) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    if (rc) return rc;
    for (uint32_t i = 0; i < NB; i++) /* every handle of the device goes stale */
        if (fk.t.live[i] && i != d.index) { (void)omega_gpu_handles_close(&fk.t, i, fk.t.generation[i], fk.t.kind[i]); free(fk.mem[i]); fk.mem[i] = NULL; }
    (void)omega_gpu_handles_close(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    memset(&fk.dev, 0, sizeof fk.dev); memset(&fk.q, 0, sizeof fk.q);
    fk.dev_closes++;
    return 0;
}

static OmegaGpuPlatformCaps g_caps = {
    .abi_major = OMEGA_GPU_PLATFORM_ABI_MAJOR, .abi_minor = OMEGA_GPU_PLATFORM_ABI_MINOR,
    .services = OMEGA_GPU_SVC_REQUIRED_SESSION, .max_devices = 1, .max_buffers = 1000, .max_queues = 1, .max_inflight = 1,
    .max_buffer_bytes = 1ull << 30, .max_total_bytes = 1ull << 32,
};

static int f_alloc(void *c, OmegaGpuDeviceHandle d, const OmegaGpuBufferDesc *desc, OmegaGpuBufferHandle *o,
                   OmegaCpuView *cpu, OmegaGpuVa *g, OmegaGpuBusAddr *b) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    if (!rc) rc = omega_gpu_platform_check_buffer(desc, &g_caps);
    if (rc) return rc;
    if (desc->gpu_cache != OMEGA_GPU_CACHE_GPU_UNCACHED) fk.cached_requests++;
    if (!fk.locks_held) fk.allocs_unlocked++;
    rc = omega_gpu_handles_open(&fk.t, desc->kind, desc->bytes, 1000, 1ull << 32, &o->index, &o->generation);
    if (rc) return rc;
    void *m = calloc(1, desc->bytes);
    fk.mem[o->index] = m; fk.desc[o->index] = *desc; fk.dev_of[o->index] = d.generation;
    cpu->ptr = m; cpu->len = desc->bytes; g->va = 0x100000000ull + (uint64_t)o->index * 0x1000000ull; b->iova = 0;
    fk.allocs++;
    return 0;
}
static int f_free(void *c, OmegaGpuBufferHandle h) {
    (void)c;
    if (h.index >= NB) { fk.free_refusals++; return OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE; }
    int rc = omega_gpu_handles_close(&fk.t, h.index, h.generation, fk.t.kind[h.index]);
    if (rc) { fk.free_refusals++; return rc; }
    free(fk.mem[h.index]); fk.mem[h.index] = NULL;
    fk.frees++;
    return 0;
}
static int buf_live(OmegaGpuBufferHandle h) {
    return h.index < NB && fk.t.live[h.index] && fk.t.generation[h.index] == h.generation;
}
static int f_sync(void *c, OmegaGpuBufferHandle h, uint64_t off, uint64_t len) {
    (void)c;
    if (!buf_live(h)) return OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE;
    if (len == 0 || off > fk.desc[h.index].bytes || len > fk.desc[h.index].bytes - off) return OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH;
    fk.syncs++;
    return 0;
}
static int f_poll(void *c, OmegaGpuBufferHandle h, uint64_t off, uint32_t want, uint64_t ms, uint32_t *seen) {
    (void)c;
    if (!buf_live(h)) return OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE;
    if (!(fk.desc[h.index].rights & OMEGA_GPU_RIGHT_HOST_POLLED)) return OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS;
    fk.polls++;
    if (fk.cancel_on_poll) { fk.cancel_on_poll = 0; omega_gpu_session_cancel(); }
    volatile uint32_t *w = (volatile uint32_t *)((uint8_t *)fk.mem[h.index] + off);
    *seen = *w;
    if ((int32_t)(*w - want) >= 0) return 0;
    fk.ns += ms * 1000000u;
    return OMEGA_GPU_POLL_TIMED_OUT;
}
static int f_irq(void *c, OmegaGpuDeviceHandle d, uint64_t t) { (void)c; (void)d; (void)t; fk.irq_calls++; return 0; }
static int f_qopen(void *c, OmegaGpuDeviceHandle d, OmegaGpuBufferHandle ring, OmegaGpuQueueHandle *o) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    if (!rc && (!buf_live(ring) || fk.t.kind[ring.index] != OMEGA_GPU_HANDLE_BUF_QUEUE)) rc = OMEGA_GPU_PLAT_REFUSED_BAD_KIND;
    if (!rc) rc = omega_gpu_handles_open(&fk.t, OMEGA_GPU_HANDLE_QUEUE, 0, 1, 0, &o->index, &o->generation);
    if (!rc) { fk.q = *o; fk.ring = ring; }
    return rc;
}
static int f_qclose(void *c, OmegaGpuQueueHandle q) {
    (void)c;
    int rc = omega_gpu_handles_close(&fk.t, q.index, q.generation, OMEGA_GPU_HANDLE_QUEUE);
    if (!rc) memset(&fk.q, 0, sizeof fk.q);
    return rc;
}
/* The fake device: the launch completes at submit (both markers and the semaphore), or never (hang). The marker and
 * semaphore addresses are read back from the method words: the QMD page is the buffer whose VA the QMD0 header
 * names, the marker page the one whose VA the first release names. */
static void *va_to_cpu(uint64_t va) {
    for (uint32_t i = 0; i < NB; i++) {
        if (!fk.mem[i] || !fk.t.live[i]) continue;
        uint64_t base = 0x100000000ull + (uint64_t)i * 0x1000000ull;
        if (va >= base && va < base + fk.desc[i].bytes) return (uint8_t *)fk.mem[i] + (va - base);
    }
    return NULL;
}
static int f_submit(void *c, OmegaGpuQueueHandle q, const uint32_t *w, uint32_t n) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, q.index, q.generation, OMEGA_GPU_HANDLE_QUEUE);
    if (rc) return rc;
    if ((uint64_t)n * 4u > fk.desc[fk.ring.index].bytes) return OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH;
    memcpy(fk.mem[fk.ring.index], w, (size_t)n * 4u);
    fk.submits++;
    if (fk.mode != MODE_COMPLETE) return 0;
    for (uint32_t i = 0; i + 5 < n; i++) {
        if (w[i] == ((2u << 28) | (5u << 16) | (0u << 13) | (0x005cu >> 2))) { /* release: va lo, va hi, payload */
            uint64_t va = (uint64_t)w[i + 1] | ((uint64_t)w[i + 2] << 32);
            uint32_t *m = va_to_cpu(va);
            if (m) *m = w[i + 3];
        }
        if (w[i] == ((2u << 28) | (2u << 16) | (1u << 13) | (0x0188u >> 2)) && w[i + 3] == ((2u << 28) | (2u << 16) | (1u << 13) | (0x0180u >> 2)) &&
            w[i + 4] == 4u) { /* the semaphore init upload: the kernel ran, the semaphore says DONE */
            uint32_t *s = va_to_cpu(((uint64_t)w[i + 1] << 32) | w[i + 2]);
            if (s) *s = OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE;
        }
    }
    return 0;
}
static int f_retire(void *c, OmegaGpuQueueHandle q) { (void)c; (void)q; fk.retires++; return 0; }
static void *f_halloc(void *c, size_t b) { (void)c; return malloc(b); }
static void f_hfree(void *c, void *p) { (void)c; free(p); }

static OmegaGpuPlatform g_plat = {
    .ctx = &fk, .time_now_ms = f_now_ms, .wait_ms = f_wait, .lock = f_lock, .unlock = f_unlock,
    .dma_alloc = f_alloc, .dma_free = f_free, .sync_cpu_to_gpu = f_sync, .sync_gpu_to_cpu = f_sync, .poll_word = f_poll,
    .device_open = f_dev_open, .device_close = f_dev_close, .diag_text = f_diag,
    .queue_open = f_qopen, .queue_close = f_qclose, .queue_submit = f_submit, .queue_retire = f_retire,
    .time_now_ns = f_now_ns, .host_alloc = f_halloc, .host_free = f_hfree,
};

static void fk_reset(void) { memset(&fk, 0, sizeof fk); omega_gpu_handles_init(&fk.t); }

static int install(void) { return omega_gpu_session_set_platform(&g_plat, &g_caps); }

static const uint32_t ARGS[4] = {1, 2, 3, 4};
static OmegaGpuLaunch launch_desc(uint64_t timeout_ms) {
    OmegaGpuLaunch L = { .code_va = 0x200000000ull, .gpr_count = 64, .threads_x = 32, .grid_x = 1, .grid_y = 1,
                         .num_elements = 32, .args = ARGS, .n_args = 4, .timeout_ms = timeout_ms };
    return L;
}

/* ---- cases ---- */

static void c_install_refusals(void) {
    fk_reset();
    OmegaGpuPlatform p = g_plat;
    OmegaGpuPlatformCaps caps = g_caps;
    caps.services &= ~OMEGA_GPU_SVC_CLOCK_NS; p.time_now_ns = NULL;
    EXPECT(omega_gpu_session_set_platform(&p, &caps) == OMEGA_GPU_PLAT_REFUSED_MISSING_SERVICE);
    p = g_plat; caps = g_caps;
    p.irq_wait = f_irq; /* a function for a service it does not claim: refused, never used */
    const int rc_irq = omega_gpu_session_set_platform(&p, &caps);
    EXPECT(rc_irq != OMEGA_GPU_PLAT_OK && omega_gpu_session_platform() == NULL);
    caps = g_caps; caps.abi_minor = 1; /* a 1.1 platform predates caps.submit_prefix_words: refused */
    EXPECT(omega_gpu_session_set_platform(&g_plat, &caps) == OMEGA_GPU_PLAT_REFUSED_ABI);
    EXPECT(omega_gpu_session_platform() == NULL); /* nothing was installed */
    EXPECT(install() == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_session_platform() == &g_plat);
    EXPECT(omega_gpu_session_completion_mode() == OMEGA_GPU_COMPLETION_POLLING);
}

static void c_no_default_platform(void) {
    fk_reset();
    /* the freestanding build has no Linux fallback: nothing installed means refused, not emulated */
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() == NULL);
    EXPECT(strstr(omega_gpu_session_last_error(), "no GPU platform installed") != NULL);
    omega_gpu_session_unlock();
    EXPECT(omega_gpu_host_alloc(16) == NULL);
    EXPECT(omega_gpu_session_now_ns() == 0);
}

static void c_open_launch_close(void) {
    fk_reset();
    EXPECT(install() == 0);
    omega_gpu_session_lock();
    EXPECT(fk.locks_held == 1);
    OmegaGpuSession *s = omega_gpu_session_open();
    EXPECT(s != NULL);
    if (!s) { omega_gpu_session_unlock(); return; }
    EXPECT(fk.allocs == 4 && fk.dev_opens == 1);
    /* typed buffers, explicit rights, all GPU-uncached; the polled marker is HOST_POLLED */
    EXPECT(fk.t.kind[s->pb.handle.index] == OMEGA_GPU_HANDLE_BUF_QUEUE);
    EXPECT(fk.desc[s->marker.handle.index].rights & OMEGA_GPU_RIGHT_HOST_POLLED);
    EXPECT(!(fk.desc[s->cbank.handle.index].rights & OMEGA_GPU_RIGHT_HOST_POLLED));
    EXPECT(fk.cached_requests == 0 && fk.allocs_unlocked == 0);
    EXPECT(s->pb.va != s->cbank.va && s->marker.size >= 0x1000 && s->qmd.size >= 0x10000);
    OmegaGpuLaunch L = launch_desc(100);
    uint64_t el = 0; uint32_t mk = 0;
    EXPECT(omega_gpu_session_launch(&L, &el, &mk) == 0);
    EXPECT(mk == OMEGA_BW_MARKER_COMPLETION_PAYLOAD && fk.submits == 1 && fk.retires == 1 && el > 0);
    EXPECT(fk.syncs >= 3);       /* explicit CPU->GPU and GPU->CPU visibility, never implicit */
    EXPECT(fk.irq_calls == 0);   /* polling only */
    EXPECT(!omega_gpu_session_is_blocked());
    /* a CODE buffer: CPU writes, GPU executes, never GPU-written, prefetch tail zeroed */
    OmegaGpuMem code;
    EXPECT(omega_gpu_session_alloc_code(100, &code) == 0);
    EXPECT(fk.t.kind[code.handle.index] == OMEGA_GPU_HANDLE_BUF_CODE);
    EXPECT((fk.desc[code.handle.index].rights & OMEGA_GPU_RIGHT_GPU_EXEC) && !(fk.desc[code.handle.index].rights & OMEGA_GPU_RIGHT_GPU_WRITE));
    omega_gpu_session_free(&code);
    EXPECT(code.cpu == NULL && fk.frees == 1);
    omega_gpu_session_unlock();
    EXPECT(fk.locks_held == 0);
    omega_gpu_session_close();
    EXPECT(fk.dev_closes == 1 && fk.frees == 5 && fk.t.nlive == 0);
    EXPECT(!omega_gpu_session_is_open());
}

static void c_cancel_before_submit(void) {
    fk_reset();
    EXPECT(install() == 0);
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() != NULL);
    omega_gpu_session_cancel();
    OmegaGpuLaunch L = launch_desc(100);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == OMEGA_GPU_SESSION_CANCELLED);
    EXPECT(fk.submits == 0);                 /* nothing went to the device */
    EXPECT(!omega_gpu_session_is_blocked()); /* so nothing is uncertain */
    OmegaGpuParkStats ps; omega_gpu_session_parked(&ps);
    EXPECT(ps.buffers == 0);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == 0); /* the request was consumed; the session still works */
    omega_gpu_session_unlock();
    omega_gpu_session_close();
    EXPECT(fk.dev_closes == 1);
}

static void check_latched(unsigned frees_before) {
    EXPECT(omega_gpu_session_is_blocked());
    OmegaGpuParkStats ps; omega_gpu_session_parked(&ps);
    EXPECT(ps.buffers == 4 && ps.untracked == 0 && ps.bytes >= 0x10000 + 0x1000 + 0x1000 + 0x10000);
    /* a buffer the caller frees now is parked too, never released: the device may still be using it */
    OmegaGpuMem m = {0};
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_alloc(64, &m) != 0); /* no new allocation after a latch */
    omega_gpu_session_unlock();
    EXPECT(fk.frees == frees_before);
    omega_gpu_session_close();
    EXPECT(fk.dev_closes == 0 && fk.frees == frees_before); /* close refused: nothing released */
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() == NULL && strstr(omega_gpu_session_last_error(), "latched") != NULL);
    OmegaGpuLaunch L = launch_desc(10);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == -1);
    EXPECT(strstr(omega_gpu_session_last_error(), "process latched") != NULL);
    omega_gpu_session_unlock();
    EXPECT(omega_gpu_session_set_platform(&g_plat, &g_caps) == OMEGA_GPU_PLAT_REFUSED_BUSY);
    EXPECT(fk.irq_calls == 0);
}

static void c_timeout_parks(void) {
    fk_reset();
    EXPECT(install() == 0);
    omega_gpu_session_lock();
    OmegaGpuSession *s = omega_gpu_session_open();
    EXPECT(s != NULL);
    OmegaGpuMem user;
    EXPECT(omega_gpu_session_alloc(4096, &user) == 0);
    fk.mode = MODE_HANG;
    OmegaGpuLaunch L = launch_desc(25);
    const uint64_t t0 = fk.ns;
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == -1);
    EXPECT(strstr(omega_gpu_session_last_error(), "timed out") != NULL);
    EXPECT(fk.polls >= 25 && fk.ns - t0 >= 25ull * 1000000u); /* polled in 1 ms slices for the whole timeout */
    EXPECT(fk.retires == 0); /* never retired: the entry may still be in flight */
    OmegaGpuBufferHandle uh = user.handle;
    omega_gpu_session_free(&user);
    EXPECT(user.cpu == NULL && omega_gpu_session_is_parked(uh));
    EXPECT(omega_gpu_session_is_parked(s->marker.handle) && omega_gpu_session_is_parked(s->pb.handle));
    omega_gpu_session_unlock();
    OmegaGpuParkStats ps; omega_gpu_session_parked(&ps);
    EXPECT(ps.buffers == 5);
    EXPECT(fk.frees == 0 && fk.t.nlive == 1 + 1 + 5); /* device, queue, 5 buffers: all still allocated */
    /* the generic latch checks expect the four session buffers; the user buffer made it five */
    EXPECT(omega_gpu_session_is_blocked());
    omega_gpu_session_close();
    EXPECT(fk.dev_closes == 0 && fk.frees == 0);
}

static void c_cancel_after_submit(void) {
    fk_reset();
    EXPECT(install() == 0);
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() != NULL);
    fk.mode = MODE_HANG;
    fk.cancel_on_poll = 1; /* the cancel arrives while the host is polling */
    OmegaGpuLaunch L = launch_desc(60000);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == OMEGA_GPU_SESSION_CANCELLED);
    EXPECT(fk.submits == 1 && fk.polls <= 2); /* seen within one poll slice, not after the 60 s timeout */
    EXPECT(strstr(omega_gpu_session_last_error(), "cancelled after submit") != NULL);
    omega_gpu_session_unlock();
    check_latched(0);
}

static void c_timeout_latch_rules(void) {
    fk_reset();
    EXPECT(install() == 0);
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() != NULL);
    fk.mode = MODE_HANG;
    OmegaGpuLaunch L = launch_desc(3);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == -1);
    omega_gpu_session_unlock();
    check_latched(0);
}

static void c_stale_after_reopen(void) {
    fk_reset();
    EXPECT(install() == 0);
    omega_gpu_session_lock();
    OmegaGpuSession *s = omega_gpu_session_open();
    EXPECT(s != NULL);
    const OmegaGpuDeviceHandle d1 = s->dev;
    OmegaGpuMem old;
    EXPECT(omega_gpu_session_alloc(8192, &old) == 0);
    omega_gpu_session_unlock();
    omega_gpu_session_close(); /* the caller kept `old` past the close */
    EXPECT(fk.dev_closes == 1);
    omega_gpu_session_lock();
    s = omega_gpu_session_open();
    EXPECT(s != NULL && omega_gpu_session_open_count() == 2);
    EXPECT(s->dev.index != d1.index || s->dev.generation != d1.generation); /* a new device generation */
    const unsigned frees = fk.frees;
    OmegaGpuAllocStats st0, st1;
    omega_gpu_session_alloc_stats(&st0);
    omega_gpu_session_free(&old); /* a handle of the closed device: refused, nothing released */
    omega_gpu_session_alloc_stats(&st1);
    EXPECT(st1.stale_refusals == st0.stale_refusals + 1 && st1.frees == st0.frees);
    EXPECT(fk.frees == frees && fk.free_refusals == 1);
    EXPECT(strstr(omega_gpu_session_last_error(), "STALE") != NULL || strstr(omega_gpu_session_last_error(), "stale") != NULL);
    EXPECT(old.cpu == NULL);
    OmegaGpuLaunch L = launch_desc(100);
    EXPECT(omega_gpu_session_launch(&L, NULL, NULL) == 0); /* the reopened session works */
    omega_gpu_session_unlock();
    omega_gpu_session_close();
    EXPECT(fk.dev_closes == 2 && fk.t.nlive == 0);
}

static void c_open_failure_text(void) {
    fk_reset();
    EXPECT(install() == 0);
    fk.fail_open = 1;
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() == NULL);
    EXPECT(strstr(omega_gpu_session_last_error(), "fake device: no GB10 here") != NULL); /* the driver text survives */
    EXPECT(!omega_gpu_session_is_blocked() && fk.allocs == 0);
    fk.fail_open = 0;
    EXPECT(omega_gpu_session_open() != NULL); /* a failed open is not a latch */
    omega_gpu_session_unlock();
    omega_gpu_session_close();
}

static void c_reservation_refusal(void) {
    fk_reset();
    EXPECT(install() == 0);
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() != NULL);
    OmegaGpuScratch sc; memset(&sc, 0, sizeof sc);
    EXPECT(omega_gpu_session_scratch_reserve(&sc, 10000) == 0);
    const unsigned allocs = fk.allocs;
    EXPECT(omega_gpu_session_scratch(&sc, 9000) == 0);
    EXPECT(omega_gpu_session_scratch(&sc, sc.cap + 1) == OMEGA_GPU_SESSION_SCRATCH_RESERVED);
    EXPECT(fk.allocs == allocs); /* refused loudly, nothing allocated */
    EXPECT(strstr(omega_gpu_session_last_error(), "serving reservation exceeded") != NULL);
    omega_gpu_session_scratch_free(&sc);
    omega_gpu_session_unlock();
    omega_gpu_session_close();
    EXPECT(fk.t.nlive == 0);
}

static void c_matmul_code_buffer(void) {
    fk_reset();
    EXPECT(install() == 0);
    uint16_t b[32 * 16];
    memset(b, 0, sizeof b);
    OmegaGpuTensor *t = NULL;
    EXPECT(omega_gpu_tensor_upload_bf16(32, 16, b, &t) == OMEGA_GPU_MATMUL_OK);
    uint16_t a[32]; memset(a, 0, sizeof a);
    float c[16];
    OmegaGpuMatmulInfo info;
    /* the fake device runs no kernel, so the result sentinels survive: CHIP_FAIL is the expected verdict here */
    int rc = omega_gpu_matmul_resident_bf16(1, a, t, c, &info);
    EXPECT(rc == OMEGA_GPU_MATMUL_CHIP_FAIL && fk.submits >= 1);
    unsigned code_bufs = 0;
    for (uint32_t i = 0; i < NB; i++)
        if (fk.t.live[i] && fk.t.kind[i] == OMEGA_GPU_HANDLE_BUF_CODE) {
            code_bufs++;
            const uint8_t *p = fk.mem[i];
            unsigned nz = 0;
            for (uint32_t j = 0; j < 64; j++) nz += p[j] != 0;
            EXPECT(nz > 0); /* the kernel was encoded in place into the device code buffer */
            EXPECT((fk.desc[i].rights & OMEGA_GPU_RIGHT_GPU_EXEC) && !(fk.desc[i].rights & OMEGA_GPU_RIGHT_GPU_WRITE));
        }
    EXPECT(code_bufs == 1);
    EXPECT(fk.cached_requests == 0 && fk.allocs_unlocked == 0 && fk.irq_calls == 0);
    omega_gpu_tensor_free(t);
    omega_gpu_session_close();
    EXPECT(fk.t.nlive == 0);
}

static void c_wait_needs_clock(void) {
    volatile uint32_t word = 0;
    omega_gpu_wait_report_t r;
    EXPECT(!omega_gpu_wait_fixed(&word, 1, &r, NULL));
    EXPECT(r.result == OMEGA_GPU_WAIT_INVALID_STATE); /* no clock in the freestanding build: refused, not unbounded */
}

static void c_fmt_matches_snprintf(void) {
    char a[96], b[96];
    const long long big = -9223372036854775807LL - 1;
    omega_gpu_fmt(a, sizeof a, "%s:%c:%d:%u:%x:%ld:%lu:%lld:%llu:%zu:%%", "stage", 'x', -42, 4000000000u, 0xbeefu,
                  -7L, 9UL, big, 18446744073709551615ULL, (size_t)12345);
    snprintf(b, sizeof b, "%s:%c:%d:%u:%x:%ld:%lu:%lld:%llu:%zu:%%", "stage", 'x', -42, 4000000000u, 0xbeefu,
             -7L, 9UL, big, 18446744073709551615ULL, (size_t)12345);
    EXPECT(strcmp(a, b) == 0);
    char s[8];
    memset(s, 'Z', sizeof s);
    omega_gpu_fmt(s, 6, "%s", "truncate me");
    EXPECT(strcmp(s, "trunc") == 0 && s[6] == 'Z' && s[7] == 'Z'); /* bounded: NUL inside cap, nothing past it */
    omega_gpu_fmt(s, 1, "%d", 123);
    EXPECT(s[0] == '\0');
}

typedef struct { const char *id; void (*fn)(void); } Case;
static const Case CASES[] = {
    {"INSTALL_REFUSALS_ABI_1_1", c_install_refusals},
    {"FREESTANDING_NO_DEFAULT_PLATFORM", c_no_default_platform},
    {"OPEN_LAUNCH_CLOSE_TYPED_UNCACHED_POLLING", c_open_launch_close},
    {"OPEN_FAILURE_KEEPS_DRIVER_TEXT", c_open_failure_text},
    {"CANCEL_BEFORE_SUBMIT_NOT_LATCHED", c_cancel_before_submit},
    {"CANCEL_AFTER_SUBMIT_LATCHES_AND_PARKS", c_cancel_after_submit},
    {"TIMEOUT_PARKS_NEVER_RELEASES", c_timeout_parks},
    {"TIMEOUT_LATCH_REFUSES_CLOSE_REOPEN_INSTALL", c_timeout_latch_rules},
    {"STALE_HANDLE_REFUSED_AFTER_CLOSE_REOPEN", c_stale_after_reopen},
    {"SERVING_RESERVATION_REFUSES_WITHOUT_ALLOC", c_reservation_refusal},
    {"MATMUL_CODE_BUFFER_ENCODED_IN_PLACE", c_matmul_code_buffer},
    {"WAIT_REFUSES_WITHOUT_CLOCK", c_wait_needs_clock},
    {"FMT_BOUNDED_MATCHES_SNPRINTF", c_fmt_matches_snprintf},
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
    printf("test_omega_gpu_session_platform: %d passed, %d failed\n", pass, fail);
    return fail ? 1 : 0;
}
