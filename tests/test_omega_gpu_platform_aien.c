/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* test_omega_gpu_platform_aien: the aien-gpu platform (src/omega_gpu_platform_aien.c) on a simulated /dev/aien-gpu.
 * Host program, opens no device: every case either installs the simulated system calls (OmegaGpuAienSys) or points
 * CUDAPP_AIEN_GPU_DEV at a path that does not exist.
 *
 * What it checks: the same caps as the Linux reference, bit for bit, and the session contract check; a missing device
 * node, a module that is not RM_READY and an ABI mismatch are named REFUSED_DEVICE refusals; every buffer is reserved,
 * allocated at that VA (GPU VA == CPU VA) and mapped MAP_FIXED|MAP_SHARED over it, GPU-uncached unless a device-private
 * DATA buffer asked for GPU_CACHED; the GPFIFO entry is nvrm_gp_entry's, USERD GPPut sits at Nvc96fControl.GPPut after
 * a 1024-entry ring, and the doorbell gets the work-submit token at +0x90; a simulated GPU that executes semaphore
 * releases from the ring completes a submission that poll_word then observes; refusals ring no doorbell; the ring
 * refuses its 1024th unretired entry; the omega #457 prefix follows a host sync into a GPU-cached buffer; the session
 * opens and closes on it; closing the device releases every object. Each case runs in its own child process.
 */
#include "omega_gpu_platform_aien.h"
#include "omega_gpu_platform_linux.h"
#include "omega_gpu_session.h"
#include "nvrm.h"
#include "aien_gpu_uapi.h"
#include "nvtypes.h"
#include "class/clc96f.h"

#include <errno.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

static int case_ok;
#define EXPECT(c) do { if (!(c)) { case_ok = 0; printf("  expect failed at line %d: %s\n", __LINE__, #c); } } while (0)

/* ---- simulated /dev/aien-gpu ---------------------------------------------------------------------------------- */

#define FAKE_FD 900
#define FAKE_PATH "/fake/aien-gpu"
#define FAKE_TOKEN 0x00010007u
#define FAKE_DOORBELL_OFF (1ull << 40)
#define FAKE_OBJS 64

typedef struct { uint64_t va, size; uint32_t flags; int live, mapped; } FakeObj;
static struct {
    uint32_t state, abi;
    int fail_chan, fail_free, opens, closes, chan_live, chan_creates, chan_destroys, frees, maps_fixed;
    FakeObj obj[FAKE_OBJS];
    uint32_t nobj;
    struct agpu_chan_create chan;
    volatile uint8_t *doorbell;
    int bad_fixed_map; /* an object mapped somewhere other than its VA, or not FIXED|SHARED */
} fk;

static int f_open(const char *path, int flags) {
    (void)flags;
    if (strcmp(path, FAKE_PATH) != 0) { errno = ENOENT; return -1; }
    fk.opens++;
    return FAKE_FD;
}
static int f_close(int fd) { if (fd == FAKE_FD) fk.closes++; return 0; }
static FakeObj *obj_of(uint32_t h) { return (h >= 1 && h <= fk.nobj && fk.obj[h - 1].live) ? &fk.obj[h - 1] : NULL; }
static int f_ioctl(int fd, unsigned long req, void *arg) {
    if (fd != FAKE_FD) { errno = EBADF; return -1; }
    if (req == AGPU_INFO) {
        struct agpu_info *i = arg; memset(i, 0, sizeof *i); i->abi = fk.abi; i->state = fk.state; return 0;
    }
    if (req == AGPU_MEM_ALLOC) {
        struct agpu_mem_alloc *a = arg;
        if (a->size == 0 || (a->size & 4095u) || (a->va & 4095u) || fk.nobj >= FAKE_OBJS) { a->st.verdict = AGPU_RM_E_ARG; return 0; }
        FakeObj *o = &fk.obj[fk.nobj++];
        *o = (FakeObj){ .va = a->va, .size = a->size, .flags = a->flags, .live = 1 };
        a->handle = fk.nobj; a->mmap_off = (2ull << 40) + ((uint64_t)a->handle << 36); a->st.verdict = AGPU_RM_OK;
        return 0;
    }
    if (req == AGPU_MEM_FREE) {
        struct agpu_mem_free *f = arg; FakeObj *o = obj_of(f->handle);
        if (fk.fail_free || !o) { f->st.verdict = AGPU_RM_E_ARG; f->st.code = 9; return 0; }
        o->live = 0; fk.frees++; f->st.verdict = AGPU_RM_OK; return 0;
    }
    if (req == AGPU_CHAN_CREATE) {
        struct agpu_chan_create *c = arg;
        FakeObj *ring = obj_of(c->ring_handle), *ud = obj_of(c->userd_handle), *er = obj_of(c->err_handle);
        if (fk.fail_chan) { c->st.verdict = AGPU_RM_E_RESULT; c->st.code = 0x1f; c->st.fn = 103; return 0; }
        if (!ring || !ud || !er || fk.chan_live || !(ring->flags & AGPU_MEM_GPU_UNCACHED) || !(ud->flags & AGPU_MEM_GPU_UNCACHED) ||
            c->entries < 8 || (c->entries & (c->entries - 1)) || c->ring_off + (uint64_t)c->entries * 8 > ring->size ||
            (c->userd_off & 511) || c->userd_off + 512 > ud->size) { c->st.verdict = AGPU_RM_E_ARG; return 0; }
        c->chid = 7; c->token = FAKE_TOKEN; c->doorbell_off = FAKE_DOORBELL_OFF; c->channel = 77; c->st.verdict = AGPU_RM_OK;
        fk.chan = *c; fk.chan_live = 1; fk.chan_creates++;
        return 0;
    }
    if (req == AGPU_CHAN_DESTROY) {
        struct agpu_chan_destroy *d = arg;
        if (!fk.chan_live || d->channel != 77) { d->st.verdict = AGPU_RM_E_ARG; return 0; }
        fk.chan_live = 0; fk.chan_destroys++; d->st.verdict = AGPU_RM_OK; return 0;
    }
    errno = ENOTTY;
    return -1;
}
static void *f_map(void *addr, size_t len, int prot, int flags, int fd, uint64_t off) {
    if (fd == -1) return mmap(addr, len, prot, flags, -1, 0);
    if (fd != FAKE_FD) { errno = EBADF; return MAP_FAILED; }
    if (off == FAKE_DOORBELL_OFF) {
        void *p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p != MAP_FAILED) fk.doorbell = p;
        return p;
    }
    uint32_t h = (uint32_t)((off - (2ull << 40)) >> 36);
    FakeObj *o = obj_of(h);
    if (!o || (uint64_t)(uintptr_t)addr != o->va || len != o->size || (flags & (MAP_FIXED | MAP_SHARED)) != (MAP_FIXED | MAP_SHARED)) {
        fk.bad_fixed_map = 1; errno = EINVAL; return MAP_FAILED;
    }
    fk.maps_fixed++; o->mapped = 1;
    return mmap(addr, len, prot, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
}
static int f_unmap(void *addr, size_t len) {
    if ((volatile uint8_t *)addr == fk.doorbell) fk.doorbell = NULL;
    return munmap(addr, len);
}
static const OmegaGpuAienSys FAKE_SYS = { f_open, f_close, f_ioctl, f_map, f_unmap };

static void fake_install(void) {
    memset(&fk, 0, sizeof fk);
    fk.state = AGPU_STATE_RM_READY; fk.abi = AGPU_ABI_VERSION;
    setenv("CUDAPP_AIEN_GPU_DEV", FAKE_PATH, 1);
    setenv("CUDAPP_MEM_RESERVE_MIB", "0", 1);
    (void)omega_gpu_platform_aien_set_sys(&FAKE_SYS);
}

static FakeObj *obj_at(uint64_t va) {
    for (uint32_t i = 0; i < fk.nobj; i++) if (fk.obj[i].live && fk.obj[i].va == va) return &fk.obj[i];
    return NULL;
}

/* ---- simulated GPU: on a doorbell write of the token it runs every GP entry up to USERD GPPut and executes the
 * semaphore releases (method 0x5c, subchannel 0, five words) it finds; every other method is skipped by count. ---- */
static volatile int g_gpu_stop;
static uint32_t g_gpu_get, g_gpu_entries_run, g_gpu_prefixes;
static uint64_t g_fifo_va;
static void run_methods(const uint32_t *m, uint32_t n) {
    for (uint32_t i = 0; i < n;) {
        const uint32_t h = m[i++], count = (h >> 16) & 0x1fffu, subc = (h >> 13) & 7u, mthd = (h & 0xfffu) << 2;
        if (h == ((2u << 28) | (4u << 16) | (0x0028u >> 2)) && i + 3 < n && m[i + 3] == (0x11u << 27)) g_gpu_prefixes++;
        if (mthd == 0x005c && subc == 0 && count == 5 && i + 5 <= n) {
            volatile uint32_t *dst = (volatile uint32_t *)(uintptr_t)(((uint64_t)m[i + 1] << 32) | m[i]);
            *dst = m[i + 2];
            __sync_synchronize();
        }
        i += count;
    }
}
static void *gpu_thread(void *arg) {
    (void)arg;
    while (!g_gpu_stop) {
        volatile uint8_t *db = fk.doorbell;
        if (db && *(volatile uint32_t *)(db + OMEGA_GPU_AIEN_DOORBELL_WORD) == FAKE_TOKEN) {
            *(volatile uint32_t *)(db + OMEGA_GPU_AIEN_DOORBELL_WORD) = 0;
            __sync_synchronize();
            volatile uint64_t *ring = (volatile uint64_t *)(uintptr_t)g_fifo_va;
            const uint32_t put = *(volatile uint32_t *)(uintptr_t)(g_fifo_va + OMEGA_GPU_AIEN_USERD_OFF + OMEGA_GPU_AIEN_USERD_GPPUT);
            while (g_gpu_get != put) {
                const uint64_t e = ring[g_gpu_get];
                run_methods((const uint32_t *)(uintptr_t)(e & ((1ull << 41) - 4u)), (uint32_t)((e >> 42) & 0x1fffffu));
                g_gpu_get = (g_gpu_get + 1u) % fk.chan.entries;
                g_gpu_entries_run++;
            }
        }
        usleep(20);
    }
    return NULL;
}

/* ---- helpers -------------------------------------------------------------------------------------------------- */

static const OmegaGpuPlatform *P(void) { return omega_gpu_platform_aien(NULL); }
static OmegaGpuBufferDesc desc(uint32_t kind, uint32_t rights, uint32_t cache, uint64_t bytes) {
    OmegaGpuBufferDesc d = { .kind = kind, .rights = rights, .gpu_cache = cache, .align = 0, .bytes = bytes };
    return d;
}
#define RW (OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_READ | OMEGA_GPU_RIGHT_GPU_WRITE)
#define POLLED (RW | OMEGA_GPU_RIGHT_HOST_POLLED)
#define RING_RIGHTS (OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_READ)

typedef struct { OmegaGpuDeviceHandle dev; OmegaGpuBufferHandle ring, marker; OmegaCpuView ring_cpu, marker_cpu;
                 OmegaGpuVa ring_va, marker_va; OmegaGpuQueueHandle q; } Rig;

static int rig_open(Rig *g) {
    const OmegaGpuPlatform *p = P();
    memset(g, 0, sizeof *g);
    OmegaGpuBufferDesc dr = desc(OMEGA_GPU_HANDLE_BUF_QUEUE, RING_RIGHTS, OMEGA_GPU_CACHE_GPU_UNCACHED, 0x10000);
    OmegaGpuBufferDesc dm = desc(OMEGA_GPU_HANDLE_BUF_DATA, POLLED, OMEGA_GPU_CACHE_GPU_UNCACHED, 4096);
    if (p->device_open(p->ctx, 0, &g->dev) != OMEGA_GPU_PLAT_OK) return -1;
    if (p->dma_alloc(p->ctx, g->dev, &dr, &g->ring, &g->ring_cpu, &g->ring_va, NULL) != OMEGA_GPU_PLAT_OK) return -1;
    if (p->dma_alloc(p->ctx, g->dev, &dm, &g->marker, &g->marker_cpu, &g->marker_va, NULL) != OMEGA_GPU_PLAT_OK) return -1;
    if (p->queue_open(p->ctx, g->dev, g->ring, &g->q) != OMEGA_GPU_PLAT_OK) return -1;
    g_fifo_va = fk.chan.ring_handle ? fk.obj[fk.chan.ring_handle - 1].va : 0;
    return 0;
}
static uint32_t release_words(uint32_t *w, uint64_t va, uint32_t payload) {
    uint32_t n = 0;
    w[n++] = nvrm_mthd(0, 0x005c, 5);
    w[n++] = (uint32_t)va; w[n++] = (uint32_t)(va >> 32); w[n++] = payload; w[n++] = 0; w[n++] = 0x1 | (1u << 20);
    return n;
}
static uint32_t doorbell_word(void) { return fk.doorbell ? *(volatile uint32_t *)(fk.doorbell + OMEGA_GPU_AIEN_DOORBELL_WORD) : 0xdeadu; }
static uint32_t gpput_word(void) { return *(volatile uint32_t *)(uintptr_t)(g_fifo_va + OMEGA_GPU_AIEN_USERD_OFF + OMEGA_GPU_AIEN_USERD_GPPUT); }

/* ---- cases ---------------------------------------------------------------------------------------------------- */

static void c_caps(void) {
    OmegaGpuPlatformCaps caps, lcaps;
    const OmegaGpuPlatform *p = omega_gpu_platform_aien(&caps);
    (void)omega_gpu_platform_linux(&lcaps);
    /* The same contract as the Linux reference except the queue count: the reference opens extra channels
     * (max_queues = 1 + NVRM_MAX_EXTRA_CHANNELS), aien-gpu ABI 1 drives one queue. */
    EXPECT(caps.max_queues == 1 && lcaps.max_queues >= 1);
    lcaps.max_queues = caps.max_queues;
    EXPECT(memcmp(&caps, &lcaps, sizeof caps) == 0);
    EXPECT(caps.services == OMEGA_GPU_SVC_REQUIRED_SESSION);
    EXPECT(p->irq_wait == NULL && p->mmio_read32 == NULL && p->mmio_write32 == NULL);
    EXPECT(omega_gpu_platform_check_for(p, &caps, OMEGA_GPU_SVC_REQUIRED_SESSION) == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_platform_negotiate(&caps, &lcaps) == OMEGA_GPU_PLAT_OK);
    OmegaGpuPlatformCaps want = caps; want.services |= OMEGA_GPU_SVC_IRQ;
    EXPECT(omega_gpu_platform_negotiate(&caps, &want) == OMEGA_GPU_PLAT_REFUSED_MISSING_SERVICE);
    want = caps; want.abi_minor = caps.abi_minor + 1;
    EXPECT(omega_gpu_platform_negotiate(&caps, &want) == OMEGA_GPU_PLAT_REFUSED_ABI);
}

static void c_absent_device(void) {
    /* The real system calls, a device node that does not exist: a named refusal, nothing opened. */
    setenv("CUDAPP_AIEN_GPU_DEV", "/nonexistent/aien-gpu", 1);
    EXPECT(omega_gpu_platform_aien_set_sys(NULL) == OMEGA_GPU_PLAT_OK);
    const OmegaGpuPlatform *p = P();
    OmegaGpuDeviceHandle d = {9, 9};
    EXPECT(p->device_open(p->ctx, 0, &d) == OMEGA_GPU_PLAT_REFUSED_DEVICE);
    EXPECT(d.index == 0 && d.generation == 0);
    const char *t = p->diag_text(p->ctx);
    printf("  diag: %s\n", t);
    EXPECT(strstr(t, "aien_gpu_open: open /nonexistent/aien-gpu: No such file or directory") == t);
    OmegaGpuBufferDesc dd = desc(OMEGA_GPU_HANDLE_BUF_DATA, RW, 0, 4096);
    OmegaGpuBufferHandle b; OmegaCpuView c; OmegaGpuVa v;
    EXPECT(p->dma_alloc(p->ctx, d, &dd, &b, &c, &v, NULL) == OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE);
    EXPECT(p->device_open(p->ctx, 1, &d) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    /* the session reports it too, and stays usable */
    EXPECT(omega_gpu_session_set_platform(omega_gpu_platform_aien(NULL), &(OmegaGpuPlatformCaps){0}) != OMEGA_GPU_PLAT_OK);
    OmegaGpuPlatformCaps caps; const OmegaGpuPlatform *ap = omega_gpu_platform_aien(&caps);
    EXPECT(omega_gpu_session_set_platform(ap, &caps) == OMEGA_GPU_PLAT_OK);
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() == NULL);
    omega_gpu_session_unlock();
}

static void c_not_ready(void) {
    fake_install();
    const OmegaGpuPlatform *p = P();
    OmegaGpuDeviceHandle d;
    fk.state = AGPU_STATE_GSP_UP;
    EXPECT(p->device_open(p->ctx, 0, &d) == OMEGA_GPU_PLAT_REFUSED_DEVICE);
    printf("  diag: %s\n", p->diag_text(p->ctx));
    EXPECT(strstr(p->diag_text(p->ctx), "state GSP_UP (3), not RM_READY") != NULL);
    EXPECT(fk.closes == fk.opens);
    fk.state = AGPU_STATE_RM_READY; fk.abi = 2;
    EXPECT(p->device_open(p->ctx, 0, &d) == OMEGA_GPU_PLAT_REFUSED_DEVICE);
    EXPECT(strstr(p->diag_text(p->ctx), "speaks ABI 2") != NULL);
    fk.abi = AGPU_ABI_VERSION;
    EXPECT(p->device_open(p->ctx, 0, &d) == OMEGA_GPU_PLAT_OK);
    EXPECT(p->diag_text(p->ctx)[0] == '\0');
    OmegaGpuDeviceHandle d2;
    EXPECT(p->device_open(p->ctx, 0, &d2) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    EXPECT(omega_gpu_platform_aien_set_sys(NULL) == OMEGA_GPU_PLAT_REFUSED_BUSY);
    EXPECT(p->device_close(p->ctx, d) == OMEGA_GPU_PLAT_OK);
    EXPECT(fk.closes == fk.opens);
}

static void c_layout_matches_nvrm(void) {
    static const uint64_t vas[] = { 0x1000, 0x7f12345000ull, 0xfffffffff000ull, 0x200000004ull, 0x123 };
    static const uint32_t ns[] = { 1, 6, 255, 0x1fffff, 0x200000 };
    for (unsigned i = 0; i < sizeof vas / sizeof vas[0]; i++)
        for (unsigned j = 0; j < sizeof ns / sizeof ns[0]; j++)
            EXPECT(omega_gpu_aien_gp_entry(vas[i], ns[j]) == nvrm_gp_entry(vas[i], ns[j]));
    EXPECT(OMEGA_GPU_AIEN_USERD_GPPUT == offsetof(Nvc96fControl, GPPut));
    EXPECT(OMEGA_GPU_AIEN_USERD_OFF == 1024u * 8u); /* nvrm_channel: entries 1024, USERD after the ring */
    EXPECT(OMEGA_GPU_AIEN_USERD_OFF + sizeof(Nvc96fControl) <= OMEGA_GPU_AIEN_FIFO_BYTES);
}

static void c_alloc(void) {
    fake_install();
    const OmegaGpuPlatform *p = P();
    OmegaGpuDeviceHandle d;
    EXPECT(p->device_open(p->ctx, 0, &d) == OMEGA_GPU_PLAT_OK);
    OmegaGpuBufferDesc dd = desc(OMEGA_GPU_HANDLE_BUF_DATA, POLLED, OMEGA_GPU_CACHE_GPU_UNCACHED, 5000);
    dd.align = 1u << 16;
    OmegaGpuBufferHandle b; OmegaCpuView c; OmegaGpuVa v; OmegaGpuBusAddr bus = {1};
    EXPECT(p->dma_alloc(p->ctx, d, &dd, &b, &c, &v, &bus) == OMEGA_GPU_PLAT_OK);
    EXPECT((uint64_t)(uintptr_t)c.ptr == v.va && c.len == 8192 && (v.va & 0xffffu) == 0 && bus.iova == 0);
    EXPECT(v.va >= (64ull << 30) && v.va + c.len <= (1ull << 40)); /* a GP entry carries 40 address bits */
    FakeObj *o = obj_at(v.va);
    EXPECT(o && o->mapped && o->size == 8192 && o->flags == AGPU_MEM_GPU_UNCACHED);
    EXPECT(!fk.bad_fixed_map);
    memset(c.ptr, 0xa5, c.len); /* the mapping is real memory */
    /* GPU-cached: only device-private DATA */
    OmegaGpuBufferDesc dc = desc(OMEGA_GPU_HANDLE_BUF_DATA, OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_READ, OMEGA_GPU_CACHE_GPU_CACHED, 4096);
    OmegaGpuBufferHandle bc; OmegaCpuView cc; OmegaGpuVa vc;
    EXPECT(p->dma_alloc(p->ctx, d, &dc, &bc, &cc, &vc, NULL) == OMEGA_GPU_PLAT_OK);
    EXPECT(obj_at(vc.va) && obj_at(vc.va)->flags == 0u);
    dc.rights |= OMEGA_GPU_RIGHT_CPU_READ;
    EXPECT(p->dma_alloc(p->ctx, d, &dc, &bc, &cc, &vc, NULL) == OMEGA_GPU_PLAT_REFUSED_CACHE_POLICY);
    OmegaGpuBufferDesc dk = desc(OMEGA_GPU_HANDLE_BUF_CODE, OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_EXEC, OMEGA_GPU_CACHE_GPU_CACHED, 4096);
    EXPECT(p->dma_alloc(p->ctx, d, &dk, &bc, &cc, &vc, NULL) == OMEGA_GPU_PLAT_REFUSED_CACHE_POLICY);
    /* free: MEM_FREE, unmapped, stale */
    const int frees = fk.frees;
    fk.fail_free = 1;
    EXPECT(p->dma_free(p->ctx, b) == OMEGA_GPU_PLAT_REFUSED_DEVICE);
    EXPECT(strstr(p->diag_text(p->ctx), "AGPU_MEM_FREE: AGPU_RM_E_ARG code 9") != NULL);
    EXPECT(p->sync_cpu_to_gpu(p->ctx, b, 0, 4) == OMEGA_GPU_PLAT_OK); /* a refused free released nothing */
    fk.fail_free = 0;
    EXPECT(p->dma_free(p->ctx, b) == OMEGA_GPU_PLAT_OK);
    EXPECT(fk.frees == frees + 1 && obj_at(v.va) == NULL);
    EXPECT(p->dma_free(p->ctx, b) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    uint32_t seen;
    EXPECT(p->poll_word(p->ctx, b, 0, 1, 0, &seen) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(p->device_close(p->ctx, d) == OMEGA_GPU_PLAT_OK);
}

static void c_submit_poll(void) {
    fake_install();
    const OmegaGpuPlatform *p = P();
    Rig g;
    EXPECT(rig_open(&g) == 0);
    EXPECT(fk.chan_creates == 1 && fk.chan.entries == 1024 && fk.chan.ring_off == 0 && fk.chan.userd_off == 8192 &&
           fk.chan.ring_handle == fk.chan.userd_handle && fk.chan.err_handle != fk.chan.ring_handle && fk.doorbell != NULL);
    pthread_t th; g_gpu_stop = 0;
    pthread_create(&th, NULL, gpu_thread, NULL);
    uint32_t w[16], seen = 0;
    volatile uint32_t *marker = g.marker_cpu.ptr;
    *marker = 0;
    for (uint32_t k = 1; k <= 3; k++) {
        const uint32_t n = release_words(w, g.marker_va.va, 0x16c0ffe0u + k);
        EXPECT(p->queue_submit(p->ctx, g.q, w, n) == OMEGA_GPU_PLAT_OK);
        EXPECT(p->poll_word(p->ctx, g.marker, 0, 0x16c0ffe0u + k, 2000, &seen) == OMEGA_GPU_PLAT_OK);
        EXPECT(seen == 0x16c0ffe0u + k);
        EXPECT(gpput_word() == k);
        const uint64_t e = ((volatile uint64_t *)(uintptr_t)g_fifo_va)[k - 1];
        EXPECT(e == nvrm_gp_entry(g.ring_va.va, n));
        EXPECT(p->queue_retire(p->ctx, g.q) == OMEGA_GPU_PLAT_OK);
    }
    EXPECT(g_gpu_entries_run == 3);
    /* timeout: the word never reaches the value */
    EXPECT(p->poll_word(p->ctx, g.marker, 0, 0x7fffffffu, 5, &seen) == OMEGA_GPU_POLL_TIMED_OUT);
    EXPECT(p->poll_word(p->ctx, g.ring, 0, 1, 0, &seen) == OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS);
    EXPECT(p->poll_word(p->ctx, g.marker, 2, 1, 0, &seen) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    EXPECT(p->poll_word(p->ctx, g.marker, 4096, 1, 0, &seen) == OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH);
    g_gpu_stop = 1;
    pthread_join(th, NULL);
    /* refusals ring no doorbell and change no ring state */
    *(volatile uint32_t *)(fk.doorbell + OMEGA_GPU_AIEN_DOORBELL_WORD) = 0;
    EXPECT(p->queue_submit(p->ctx, g.q, w, 0x10000 / 4 + 1) == OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH);
    EXPECT(p->queue_submit(p->ctx, g.q, NULL, 4) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    EXPECT(doorbell_word() == 0 && gpput_word() == 3);
    /* the ring refuses its 1024th unretired entry, and a retire frees it */
    uint32_t ok = 0;
    for (uint32_t k = 0; k < 1023; k++) ok += p->queue_submit(p->ctx, g.q, w, 6) == OMEGA_GPU_PLAT_OK;
    EXPECT(ok == 1023 && doorbell_word() == FAKE_TOKEN);
    EXPECT(gpput_word() == (3u + 1023u) % 1024u);
    *(volatile uint32_t *)(fk.doorbell + OMEGA_GPU_AIEN_DOORBELL_WORD) = 0;
    EXPECT(p->queue_submit(p->ctx, g.q, w, 6) == OMEGA_GPU_PLAT_REFUSED_DEVICE);
    EXPECT(strstr(p->diag_text(p->ctx), "GPFIFO full: 1023 entries unretired") != NULL);
    EXPECT(doorbell_word() == 0);
    EXPECT(p->queue_retire(p->ctx, g.q) == OMEGA_GPU_PLAT_OK);
    EXPECT(p->queue_submit(p->ctx, g.q, w, 6) == OMEGA_GPU_PLAT_OK && doorbell_word() == FAKE_TOKEN);
    /* the queue's ring cannot be freed while the queue is open; a second queue is refused */
    EXPECT(p->dma_free(p->ctx, g.ring) == OMEGA_GPU_PLAT_REFUSED_BUSY);
    OmegaGpuQueueHandle q2;
    EXPECT(p->queue_open(p->ctx, g.dev, g.ring, &q2) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    EXPECT(p->queue_open(p->ctx, g.dev, g.marker, &q2) == OMEGA_GPU_PLAT_REFUSED_BAD_KIND);
    EXPECT(p->queue_close(p->ctx, g.q) == OMEGA_GPU_PLAT_OK);
    EXPECT(fk.chan_destroys == 1 && fk.doorbell == NULL);
    EXPECT(p->queue_submit(p->ctx, g.q, w, 6) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(p->dma_free(p->ctx, g.ring) == OMEGA_GPU_PLAT_OK);
    EXPECT(p->device_close(p->ctx, g.dev) == OMEGA_GPU_PLAT_OK);
}

static void c_chan_fail(void) {
    fake_install();
    const OmegaGpuPlatform *p = P();
    Rig g;
    fk.fail_chan = 1;
    EXPECT(rig_open(&g) == -1);
    printf("  diag: %s\n", p->diag_text(p->ctx));
    EXPECT(strstr(p->diag_text(p->ctx), "aien_gpu_chan_create: AGPU_CHAN_CREATE: AGPU_RM_E_RESULT code 31 fn 103") != NULL);
    EXPECT(g.q.generation == 0);
    uint32_t live = 0;
    for (uint32_t i = 0; i < fk.nobj; i++) live += fk.obj[i].live;
    EXPECT(live == 2); /* the ring and the marker; the fifo and notifier went back */
    fk.fail_chan = 0;
    EXPECT(p->queue_open(p->ctx, g.dev, g.ring, &g.q) == OMEGA_GPU_PLAT_OK);
    EXPECT(p->device_close(p->ctx, g.dev) == OMEGA_GPU_PLAT_OK);
}

static void c_l2_prefix(void) {
    fake_install();
    const OmegaGpuPlatform *p = P();
    Rig g;
    EXPECT(rig_open(&g) == 0);
    OmegaGpuBufferDesc dc = desc(OMEGA_GPU_HANDLE_BUF_DATA, OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_READ, OMEGA_GPU_CACHE_GPU_CACHED, 4096);
    OmegaGpuBufferHandle bc; OmegaCpuView cc; OmegaGpuVa vc;
    EXPECT(p->dma_alloc(p->ctx, g.dev, &dc, &bc, &cc, &vc, NULL) == OMEGA_GPU_PLAT_OK);
    uint32_t w[8]; const uint32_t n = release_words(w, g.marker_va.va, 5);
    const uint32_t *pb = g.ring_cpu.ptr;
    EXPECT(p->sync_cpu_to_gpu(p->ctx, g.marker, 0, 4) == OMEGA_GPU_PLAT_OK); /* uncached: no prefix */
    EXPECT(p->queue_submit(p->ctx, g.q, w, n) == OMEGA_GPU_PLAT_OK && pb[0] == w[0]);
    EXPECT(p->sync_cpu_to_gpu(p->ctx, bc, 0, 4096) == OMEGA_GPU_PLAT_OK);
    EXPECT(p->queue_submit(p->ctx, g.q, w, n) == OMEGA_GPU_PLAT_OK);
    EXPECT(pb[4] == (0x11u << 27) && pb[5] == w[0]);
    EXPECT(((volatile uint64_t *)(uintptr_t)g_fifo_va)[1] == nvrm_gp_entry(g.ring_va.va, n + 5));
    EXPECT(p->queue_submit(p->ctx, g.q, w, n) == OMEGA_GPU_PLAT_OK && pb[0] == w[0]); /* once only */
    EXPECT(p->device_close(p->ctx, g.dev) == OMEGA_GPU_PLAT_OK);
}

static void c_session(void) {
    fake_install();
    OmegaGpuPlatformCaps caps;
    const OmegaGpuPlatform *p = omega_gpu_platform_aien(&caps);
    EXPECT(omega_gpu_session_set_platform(p, &caps) == OMEGA_GPU_PLAT_OK);
    omega_gpu_session_lock();
    EXPECT(omega_gpu_session_open() != NULL);
    EXPECT(fk.chan_creates == 1 && fk.maps_fixed >= 6 && !fk.bad_fixed_map);
    OmegaGpuMem m;
    EXPECT(omega_gpu_session_alloc(10000, &m) == 0);
    EXPECT(m.cpu && (uint64_t)(uintptr_t)m.cpu == m.va && obj_at(m.va) && obj_at(m.va)->flags == AGPU_MEM_GPU_UNCACHED);
    omega_gpu_session_free(&m);
    omega_gpu_session_unlock();
    omega_gpu_session_close(); /* takes the lock itself (docs/runtime/API.md) */
    uint32_t live = 0;
    for (uint32_t i = 0; i < fk.nobj; i++) live += fk.obj[i].live;
    EXPECT(live == 0 && fk.chan_live == 0 && fk.closes == fk.opens);
}

static void c_close_releases(void) {
    fake_install();
    const OmegaGpuPlatform *p = P();
    Rig g;
    EXPECT(rig_open(&g) == 0);
    EXPECT(p->device_close(p->ctx, g.dev) == OMEGA_GPU_PLAT_OK);
    uint32_t live = 0;
    for (uint32_t i = 0; i < fk.nobj; i++) live += fk.obj[i].live;
    EXPECT(live == 0 && fk.chan_destroys == 1 && fk.closes == 1 && fk.doorbell == NULL);
    uint32_t seen;
    EXPECT(p->poll_word(p->ctx, g.marker, 0, 0, 0, &seen) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(p->queue_retire(p->ctx, g.q) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(p->device_close(p->ctx, g.dev) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(rig_open(&g) == 0); /* reopens cleanly */
    EXPECT(p->device_close(p->ctx, g.dev) == OMEGA_GPU_PLAT_OK);
}

typedef struct { const char *name; void (*fn)(void); } Case;
static const Case CASES[] = {
    {"AIEN_CAPS_EQUAL_LINUX_REF", c_caps},
    {"AIEN_ABSENT_DEVICE_REFUSED", c_absent_device},
    {"AIEN_NOT_RM_READY_REFUSED", c_not_ready},
    {"AIEN_GP_ENTRY_USERD_MATCH_NVRM", c_layout_matches_nvrm},
    {"AIEN_ALLOC_VA_IDENTITY_CACHE_FREE", c_alloc},
    {"AIEN_SUBMIT_DOORBELL_POLL_RING", c_submit_poll},
    {"AIEN_CHAN_CREATE_FAIL_UNWINDS", c_chan_fail},
    {"AIEN_L2_INVAL_PREFIX", c_l2_prefix},
    {"AIEN_SESSION_OPEN_CLOSE", c_session},
    {"AIEN_DEVICE_CLOSE_RELEASES", c_close_releases},
};

int main(void) {
    int fails = 0;
    const unsigned n = sizeof CASES / sizeof CASES[0];
    const char *only = getenv("AIEN_TEST_CASE");
    for (unsigned i = 0; i < n; i++) {
        if (only && strcmp(only, CASES[i].name) != 0) continue;
        fflush(stdout);
        pid_t pid = fork();
        if (pid == 0) { alarm(30); case_ok = 1; CASES[i].fn(); fflush(stdout); _exit(case_ok ? 0 : 1); }
        int st = 0;
        waitpid(pid, &st, 0);
        const int ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;
        printf("%s %s\n", ok ? "PASS" : "FAIL", CASES[i].name);
        fails += !ok;
    }
    printf("test_omega_gpu_platform_aien: %u cases, %d failed\n", n, fails);
    return fails ? 1 : 0;
}
