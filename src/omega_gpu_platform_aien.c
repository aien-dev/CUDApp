/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* omega_gpu_platform_aien.c -- OmegaGpuPlatform 1.2 over aien-gpu.ko (/dev/aien-gpu, compute ABI 1). See the header for
 * the call map, the evidence level (IMPLEMENTED, hardware NOT_RUN) and how it is selected. The contract, caps and
 * refusals are those of src/omega_gpu_platform_linux.c; the submission path is nvrm_enqueue + nvrm_ring
 * (nvrm/nvrm.c) written against memory and a doorbell page this platform got from aien-gpu.ko instead of RM.
 */
#include "omega_gpu_platform_aien.h"

#include "omega_gpu_mem_reserve.h"
#include "aien_gpu_uapi.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define AIEN_BUFFERS (OMEGA_GPU_HANDLE_SLOTS - 2u) /* one slot each for the device and the queue */
#define AIEN_PAGE 4096ull
/* Every buffer lives in [64 GiB, 1 TiB): a GPFIFO entry carries 40 address bits (GET 31:2, GET_HI 39:32;
 * nvrm_gp_entry puts LEVEL at bit 41 and LENGTH at 42), and the kernel's own choice on arm64 is near 2^48. The
 * window is below AGPU_MEM_ALLOC's 2^48 limit and above GSP's [4 GiB, 4.5 GiB). */
#define AIEN_VA_BASE (64ull << 30)
#define AIEN_VA_TOP (1ull << 40)
#define AIEN_GP_MAX_WORDS 0x1fffffu /* LENGTH field, 21 bits */

typedef struct {
    void *cpu;      /* == GPU VA */
    uint64_t size;  /* bytes mapped, page rounded */
    uint32_t handle; /* AGPU_MEM_ALLOC handle */
} AienMem;

typedef struct {
    pthread_mutex_t mu; /* table, device and diagnostics */
    pthread_mutex_t locks[OMEGA_GPU_PLATFORM_AIEN_LOCKS];
    int ready;
    OmegaGpuHandleTable t;
    AienMem mem[OMEGA_GPU_HANDLE_SLOTS];
    uint32_t rights[OMEGA_GPU_HANDLE_SLOTS];
    uint8_t gpu_cached[OMEGA_GPU_HANDLE_SLOTS];
    int l2_inval_pending; /* omega #457, as the Linux reference */
    const OmegaGpuAienSys *sys;
    int fd;
    OmegaGpuDeviceHandle dev; /* generation 0 = closed */
    OmegaGpuQueueHandle queue;
    uint64_t va_next;        /* VA window cursor (mem_map) */
    uint32_t ring_index;     /* slot of the caller's BUF_QUEUE buffer (the pushbuffer) */
    AienMem fifo, notifier;  /* GPFIFO ring + USERD, error notifier */
    uint32_t channel, token;
    volatile uint8_t *doorbell;
    uint32_t put, retired;   /* entries enqueued / seen complete, monotonic (nvrm.h) */
    char err[256];           /* driver text of the last REFUSED_DEVICE */
    char diag[320];
} AienRef;

static int sys_open(const char *path, int flags) { return open(path, flags); }
static int sys_close(int fd) { return close(fd); }
static int sys_ioctl(int fd, unsigned long req, void *arg) { return ioctl(fd, req, arg); }
static void *sys_map(void *addr, size_t len, int prot, int flags, int fd, uint64_t off) {
    return mmap(addr, len, prot, flags, fd, (off_t)off);
}
static int sys_unmap(void *addr, size_t len) { return munmap(addr, len); }
static const OmegaGpuAienSys g_real_sys = { sys_open, sys_close, sys_ioctl, sys_map, sys_unmap };

static AienRef g_ref = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .locks = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER},
    .fd = -1,
};

/* omega #457: the same L2_SYSMEM_NCOH_INVALIDATE MEM_OP prefix as the Linux reference. */
static const uint32_t L2_INVAL_PREFIX[] = { (2u << 28) | (4u << 16) | (0x0028u >> 2), 0, 0, 0, 0x11u << 27 };
#define L2_INVAL_PREFIX_WORDS ((uint32_t)(sizeof L2_INVAL_PREFIX / sizeof L2_INVAL_PREFIX[0]))

static const OmegaGpuPlatformCaps g_caps = {
    .submit_prefix_words = L2_INVAL_PREFIX_WORDS,
    .abi_major = OMEGA_GPU_PLATFORM_ABI_MAJOR,
    .abi_minor = OMEGA_GPU_PLATFORM_ABI_MINOR,
    .services = OMEGA_GPU_SVC_REQUIRED_SESSION, /* no MMIO (the module owns the registers), no IRQ (polling only) */
    .max_devices = 1,
    .max_buffers = AIEN_BUFFERS,
    .max_queues = 1,
    .max_inflight = 1,
    .max_buffer_bytes = 1ull << 37,
    .max_total_bytes = 1ull << 37,
};

uint64_t omega_gpu_aien_gp_entry(uint64_t va, uint32_t nwords) {
    return (va & ~3ull) | ((uint64_t)(nwords & AIEN_GP_MAX_WORDS) << 42) | (1ull << 41);
}

static void ref_lock(AienRef *r) {
    pthread_mutex_lock(&r->mu);
    if (!r->ready) { omega_gpu_handles_init(&r->t); if (!r->sys) r->sys = &g_real_sys; r->ready = 1; }
}
static void ref_unlock(AienRef *r) { pthread_mutex_unlock(&r->mu); }

int omega_gpu_platform_aien_set_sys(const OmegaGpuAienSys *sys) {
    AienRef *r = &g_ref;
    ref_lock(r);
    int rc = r->dev.generation != 0u ? OMEGA_GPU_PLAT_REFUSED_BUSY : OMEGA_GPU_PLAT_OK;
    if (rc == OMEGA_GPU_PLAT_OK) r->sys = sys ? sys : &g_real_sys;
    ref_unlock(r);
    return rc;
}

/* Records why the call failed; REFUSED_DEVICE carries the driver text (r->err). Called with r->mu held. */
static int refuse(AienRef *r, int rc, const char *stage) {
    if (rc == OMEGA_GPU_PLAT_REFUSED_DEVICE)
        snprintf(r->diag, sizeof r->diag, "%s: %s", stage, r->err[0] ? r->err : "(no driver text)");
    else
        snprintf(r->diag, sizeof r->diag, "%s: %s", stage, omega_gpu_platform_refusal_name(rc));
    return rc;
}

static const char *state_name(uint32_t s) {
    switch (s) {
    case AGPU_STATE_BOUND: return "BOUND";
    case AGPU_STATE_BOOTING: return "BOOTING";
    case AGPU_STATE_GSP_UP: return "GSP_UP";
    case AGPU_STATE_RM_READY: return "RM_READY";
    case AGPU_STATE_FAILED: return "FAILED";
    default: return "UNKNOWN";
    }
}

static const char *verdict_name(uint32_t v) {
    static const char *const n[] = { "0", "OK", "E_STATE", "E_ARG", "E_QUEUE", "E_TIMEOUT", "E_RESULT", "E_STATUS",
                                     "E_VERSION", "E_STATIC", "E_REPLY", "E_NOMEM", "E_UNSUPPORTED" };
    return v < sizeof n / sizeof n[0] ? n[v] : "UNKNOWN";
}

/* One ioctl that carries an agpu_rm_status: 0, or -1 with r->err naming the call and why. */
static int rm_call(AienRef *r, unsigned long req, void *arg, const struct agpu_rm_status *st, const char *what) {
    if (r->sys->ioctl_dev(r->fd, req, arg) != 0) {
        snprintf(r->err, sizeof r->err, "%s: %s", what, strerror(errno));
        return -1;
    }
    if (st->verdict != AGPU_RM_OK) {
        snprintf(r->err, sizeof r->err, "%s: AGPU_RM_%s code %u fn %u step %u", what, verdict_name(st->verdict),
                 st->code, st->fn, st->step);
        return -1;
    }
    return 0;
}

/* AGPU_MEM_ALLOC at a CPU VA this process reserved, then the object mapped over the reservation. */
static int mem_map(AienRef *r, uint64_t bytes, uint64_t align, uint32_t flags, AienMem *m) {
    memset(m, 0, sizeof *m);
    if (align < AIEN_PAGE) align = AIEN_PAGE;
    if (bytes == 0 || bytes > (1ull << 37)) { snprintf(r->err, sizeof r->err, "mem: %llu bytes", (unsigned long long)bytes); return -1; }
    const uint64_t size = (bytes + AIEN_PAGE - 1u) & ~(AIEN_PAGE - 1u);
    /* Reserve [va, va + size) in the window with MAP_FIXED_NOREPLACE, first fit from a moving cursor: a range taken by
     * anything else in the process (EEXIST) is stepped over; the search wraps once. */
    uint64_t va = 0;
    const uint64_t start = (r->va_next < AIEN_VA_BASE || r->va_next >= AIEN_VA_TOP) ? AIEN_VA_BASE : r->va_next;
    uint64_t at = (start + align - 1u) & ~(align - 1u);
    for (int wrapped = 0;;) {
        if (at + size > AIEN_VA_TOP) {
            if (wrapped) break;
            wrapped = 1; at = AIEN_VA_BASE;
            continue;
        }
        if (wrapped && at >= start) break;
        void *p = r->sys->map((void *)(uintptr_t)at, (size_t)size, PROT_NONE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == (void *)(uintptr_t)at) { va = at; break; }
        if (p != MAP_FAILED) (void)r->sys->unmap(p, (size_t)size); /* a kernel without NOREPLACE moved it */
        else if (errno != EEXIST) break;
        at = (at + size + align - 1u) & ~(align - 1u);
    }
    if (!va) {
        snprintf(r->err, sizeof r->err, "no free %llu-byte range in VA window [0x%llx, 0x%llx): %s", (unsigned long long)size,
                 (unsigned long long)AIEN_VA_BASE, (unsigned long long)AIEN_VA_TOP, strerror(errno));
        return -1;
    }
    r->va_next = va + size;
    struct agpu_mem_alloc a;
    memset(&a, 0, sizeof a);
    a.size = size; a.va = va; a.flags = flags;
    if (rm_call(r, AGPU_MEM_ALLOC, &a, &a.st, "AGPU_MEM_ALLOC") != 0) { (void)r->sys->unmap((void *)(uintptr_t)va, (size_t)size); return -1; }
    void *cpu = r->sys->map((void *)(uintptr_t)va, (size_t)size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, r->fd, a.mmap_off);
    if (cpu != (void *)(uintptr_t)va) {
        snprintf(r->err, sizeof r->err, "mmap AGPU_MEM_ALLOC handle %u at 0x%llx: %s", a.handle, (unsigned long long)va,
                 cpu == MAP_FAILED ? strerror(errno) : "placed elsewhere");
        if (cpu != MAP_FAILED) (void)r->sys->unmap(cpu, (size_t)size);
        struct agpu_mem_free f = { .handle = a.handle };
        (void)r->sys->ioctl_dev(r->fd, AGPU_MEM_FREE, &f);
        (void)r->sys->unmap((void *)(uintptr_t)va, (size_t)size);
        return -1;
    }
    m->cpu = cpu; m->size = size; m->handle = a.handle;
    return 0;
}

/* AGPU_MEM_FREE (the GPU mapping goes first), then the CPU mapping. On a refused free nothing is unmapped unless
 * force (device close: the descriptor closes next and takes the object with it). */
static int mem_unmap(AienRef *r, AienMem *m, int force) {
    if (!m->cpu) return 0;
    struct agpu_mem_free f;
    memset(&f, 0, sizeof f);
    f.handle = m->handle;
    const int rc = rm_call(r, AGPU_MEM_FREE, &f, &f.st, "AGPU_MEM_FREE");
    if (rc != 0 && !force) return -1;
    (void)r->sys->unmap(m->cpu, (size_t)m->size);
    memset(m, 0, sizeof *m);
    return rc;
}

static int dev_ok(AienRef *r, OmegaGpuDeviceHandle d) {
    return omega_gpu_handles_check(&r->t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
}

static int buf_ok(AienRef *r, OmegaGpuBufferHandle b, uint32_t *kind) {
    if (b.index >= OMEGA_GPU_HANDLE_SLOTS || b.generation == 0u) return OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE;
    if (!r->t.live[b.index] || r->t.generation[b.index] != b.generation) return OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE;
    const uint32_t k = r->t.kind[b.index];
    if (k != OMEGA_GPU_HANDLE_BUF_DATA && k != OMEGA_GPU_HANDLE_BUF_CODE && k != OMEGA_GPU_HANDLE_BUF_QUEUE)
        return OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE;
    if (kind) *kind = k;
    return OMEGA_GPU_PLAT_OK;
}

static int range_ok(AienRef *r, OmegaGpuBufferHandle b, uint64_t off, uint64_t len) {
    const uint64_t size = r->mem[b.index].size;
    return (len == 0u || off > size || len > size - off) ? OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH : OMEGA_GPU_PLAT_OK;
}

static uint64_t ref_now_ns(void *ctx) {
    (void)ctx;
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}
static uint64_t ref_now_ms(void *ctx) { return ref_now_ns(ctx) / 1000000ull; }

static int ref_wait_ms(void *ctx, uint64_t ms) {
    (void)ctx;
    struct timespec t = {.tv_sec = (time_t)(ms / 1000u), .tv_nsec = (long)(ms % 1000u) * 1000000L};
    while (nanosleep(&t, &t) != 0) {}
    return OMEGA_GPU_PLAT_OK;
}

static int ref_lock_id(void *ctx, uint32_t id) {
    AienRef *r = ctx;
    if (id >= OMEGA_GPU_PLATFORM_AIEN_LOCKS) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    return pthread_mutex_lock(&r->locks[id]) == 0 ? OMEGA_GPU_PLAT_OK : OMEGA_GPU_PLAT_REFUSED_BUSY;
}
static int ref_unlock_id(void *ctx, uint32_t id) {
    AienRef *r = ctx;
    if (id >= OMEGA_GPU_PLATFORM_AIEN_LOCKS) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    return pthread_mutex_unlock(&r->locks[id]) == 0 ? OMEGA_GPU_PLAT_OK : OMEGA_GPU_PLAT_REFUSED_ARGS;
}

static int ref_device_open(void *ctx, uint32_t ordinal, OmegaGpuDeviceHandle *out) {
    AienRef *r = ctx;
    int rc;
    if (!out) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    memset(out, 0, sizeof *out);
    ref_lock(r);
    if (ordinal != 0u) { rc = refuse(r, OMEGA_GPU_PLAT_REFUSED_ARGS, "device_open"); goto out; }
    if (r->dev.generation != 0u) { rc = refuse(r, OMEGA_GPU_PLAT_REFUSED_LIMIT, "device_open (already open)"); goto out; }
    const char *path = getenv("CUDAPP_AIEN_GPU_DEV");
    if (!path || !*path) path = OMEGA_GPU_AIEN_DEFAULT_DEV;
    r->err[0] = '\0';
    r->fd = r->sys->open_dev(path, O_RDWR | O_CLOEXEC);
    if (r->fd < 0) {
        snprintf(r->err, sizeof r->err, "open %s: %s (aien-gpu.ko not loaded?)", path, strerror(errno));
        r->fd = -1;
        rc = refuse(r, OMEGA_GPU_PLAT_REFUSED_DEVICE, "aien_gpu_open");
        goto out;
    }
    struct agpu_info info;
    memset(&info, 0, sizeof info);
    if (r->sys->ioctl_dev(r->fd, AGPU_INFO, &info) != 0)
        snprintf(r->err, sizeof r->err, "AGPU_INFO on %s: %s", path, strerror(errno));
    else if (info.abi != AGPU_ABI_VERSION)
        snprintf(r->err, sizeof r->err, "%s speaks ABI %u; this platform speaks ABI %u", path, info.abi, AGPU_ABI_VERSION);
    else if (info.state != AGPU_STATE_RM_READY)
        snprintf(r->err, sizeof r->err, "%s is in state %s (%u), not RM_READY: AGPU_BOOT and AGPU_RM_INIT must finish first",
                 path, state_name(info.state), info.state);
    if (r->err[0]) {
        (void)r->sys->close_dev(r->fd);
        r->fd = -1;
        rc = refuse(r, OMEGA_GPU_PLAT_REFUSED_DEVICE, "aien_gpu_info");
        goto out;
    }
    rc = omega_gpu_handles_open(&r->t, OMEGA_GPU_HANDLE_DEVICE, 0, 1, 0, &out->index, &out->generation);
    if (rc != OMEGA_GPU_PLAT_OK) { (void)r->sys->close_dev(r->fd); r->fd = -1; rc = refuse(r, rc, "device_open"); goto out; }
    r->dev = *out;
    r->diag[0] = '\0';
out:
    ref_unlock(r);
    return rc;
}

/* Channel and its memory; called with r->mu held. -1 (r->err set) when AGPU_CHAN_DESTROY refused and !force. */
static int channel_release(AienRef *r, int force) {
    struct agpu_chan_destroy d;
    memset(&d, 0, sizeof d);
    d.channel = r->channel;
    if (rm_call(r, AGPU_CHAN_DESTROY, &d, &d.st, "AGPU_CHAN_DESTROY") != 0 && !force) return -1;
    if (r->doorbell) (void)r->sys->unmap((void *)(uintptr_t)r->doorbell, OMEGA_GPU_AIEN_DOORBELL_BYTES);
    (void)mem_unmap(r, &r->fifo, 1);
    (void)mem_unmap(r, &r->notifier, 1);
    r->doorbell = NULL;
    r->channel = r->token = 0;
    r->put = r->retired = 0;
    memset(&r->queue, 0, sizeof r->queue);
    r->ring_index = 0;
    return 0;
}

static int ref_device_close(void *ctx, OmegaGpuDeviceHandle dev) {
    AienRef *r = ctx;
    ref_lock(r);
    int rc = dev_ok(r, dev);
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "device_close"); ref_unlock(r); return rc; }
    /* Every handle of the device goes stale. Unlike the RM client, a mapping of an aien-gpu object keeps it alive past
     * close(fd), so every live buffer is released here. */
    if (r->queue.generation != 0u) (void)channel_release(r, 1);
    for (uint32_t i = 0; i < OMEGA_GPU_HANDLE_SLOTS; i++) {
        if (!r->t.live[i] || i == dev.index) continue;
        if (r->t.kind[i] != OMEGA_GPU_HANDLE_QUEUE) (void)mem_unmap(r, &r->mem[i], 1);
        (void)omega_gpu_handles_close(&r->t, i, r->t.generation[i], r->t.kind[i]);
        memset(&r->mem[i], 0, sizeof r->mem[i]);
        r->rights[i] = 0;
        r->gpu_cached[i] = 0;
    }
    (void)omega_gpu_handles_close(&r->t, dev.index, dev.generation, OMEGA_GPU_HANDLE_DEVICE);
    memset(&r->dev, 0, sizeof r->dev);
    r->l2_inval_pending = 0;
    (void)r->sys->close_dev(r->fd);
    r->fd = -1;
    ref_unlock(r);
    return OMEGA_GPU_PLAT_OK;
}

static const char *ref_diag_text(void *ctx) { return ((AienRef *)ctx)->diag; }

static int ref_dma_alloc(void *ctx, OmegaGpuDeviceHandle dev, const OmegaGpuBufferDesc *desc, OmegaGpuBufferHandle *out,
                         OmegaCpuView *cpu, OmegaGpuVa *gpu, OmegaGpuBusAddr *bus) {
    AienRef *r = ctx;
    if (!desc || !out || !cpu || !gpu) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    memset(out, 0, sizeof *out); memset(cpu, 0, sizeof *cpu); memset(gpu, 0, sizeof *gpu);
    if (bus) memset(bus, 0, sizeof *bus);
    ref_lock(r);
    int rc = dev_ok(r, dev);
    if (rc == OMEGA_GPU_PLAT_OK) rc = omega_gpu_platform_check_buffer(desc, &g_caps);
    /* The Linux reference's rule: GPU-cached only for device-private DATA buffers (no CPU_READ, no HOST_POLLED). */
    const int cached = desc->gpu_cache == OMEGA_GPU_CACHE_GPU_CACHED;
    if (rc == OMEGA_GPU_PLAT_OK && cached &&
        (desc->kind != OMEGA_GPU_HANDLE_BUF_DATA || (desc->rights & (OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_HOST_POLLED))))
        rc = OMEGA_GPU_PLAT_REFUSED_CACHE_POLICY;
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "dma_alloc"); ref_unlock(r); return rc; }
    rc = omega_gpu_mem_reserve_check(desc->bytes, r->diag, sizeof r->diag); /* D1, before any slot or driver call */
    if (rc != OMEGA_GPU_PLAT_OK) { ref_unlock(r); return rc; }
    uint32_t idx = 0, gen = 0;
    rc = omega_gpu_handles_open(&r->t, desc->kind, desc->bytes, AIEN_BUFFERS, g_caps.max_total_bytes, &idx, &gen);
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "dma_alloc"); ref_unlock(r); return rc; }
    AienMem m;
    if (mem_map(r, desc->bytes, desc->align, cached ? 0u : AGPU_MEM_GPU_UNCACHED, &m) != 0) {
        (void)omega_gpu_handles_close(&r->t, idx, gen, desc->kind);
        rc = refuse(r, OMEGA_GPU_PLAT_REFUSED_DEVICE, "aien_gpu_mem_alloc");
        ref_unlock(r);
        return rc;
    }
    r->mem[idx] = m;
    r->rights[idx] = desc->rights;
    r->gpu_cached[idx] = (uint8_t)cached;
    out->index = idx; out->generation = gen;
    cpu->ptr = m.cpu; cpu->len = (size_t)m.size;
    gpu->va = (uint64_t)(uintptr_t)m.cpu; /* GPU VA == CPU VA */
    ref_unlock(r);
    return OMEGA_GPU_PLAT_OK;
}

static int ref_dma_free(void *ctx, OmegaGpuBufferHandle b) {
    AienRef *r = ctx;
    uint32_t kind = 0;
    ref_lock(r);
    int rc = buf_ok(r, b, &kind);
    if (rc == OMEGA_GPU_PLAT_OK && r->queue.generation != 0u && r->ring_index == b.index)
        rc = OMEGA_GPU_PLAT_REFUSED_BUSY; /* the open queue still uses it as its ring */
    if (rc == OMEGA_GPU_PLAT_OK && mem_unmap(r, &r->mem[b.index], 0) != 0) rc = OMEGA_GPU_PLAT_REFUSED_DEVICE;
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "dma_free"); ref_unlock(r); return rc; }
    (void)omega_gpu_handles_close(&r->t, b.index, b.generation, kind);
    r->rights[b.index] = 0;
    r->gpu_cached[b.index] = 0;
    ref_unlock(r);
    return OMEGA_GPU_PLAT_OK;
}

static void barrier(void) {
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __sync_synchronize();
#endif
}

static int ref_sync(void *ctx, OmegaGpuBufferHandle b, uint64_t off, uint64_t len, const char *stage, int to_gpu) {
    AienRef *r = ctx;
    ref_lock(r);
    int rc = buf_ok(r, b, NULL);
    if (rc == OMEGA_GPU_PLAT_OK) rc = range_ok(r, b, off, len);
    if (rc != OMEGA_GPU_PLAT_OK) rc = refuse(r, rc, stage);
    else if (to_gpu && r->gpu_cached[b.index]) r->l2_inval_pending = 1; /* omega #457 */
    ref_unlock(r);
    if (rc == OMEGA_GPU_PLAT_OK) barrier();
    return rc;
}
static int ref_sync_to_gpu(void *ctx, OmegaGpuBufferHandle b, uint64_t off, uint64_t len) { return ref_sync(ctx, b, off, len, "sync_cpu_to_gpu", 1); }
static int ref_sync_to_cpu(void *ctx, OmegaGpuBufferHandle b, uint64_t off, uint64_t len) { return ref_sync(ctx, b, off, len, "sync_gpu_to_cpu", 0); }

/* The m16_native_wait_marker_ge rule: serial-number compare, 50 us sleep-poll, one last read at the deadline. */
static int wait_ge(volatile uint32_t *w, uint32_t target, uint64_t timeout_ms) {
    uint64_t duration, end;
    if (__builtin_mul_overflow(timeout_ms, 1000000ull, &duration) || __builtin_add_overflow(ref_now_ns(NULL), duration, &end))
        end = UINT64_MAX;
    while (ref_now_ns(NULL) < end) {
        barrier();
        if ((int32_t)(*w - target) >= 0) return 0;
        usleep(50);
    }
    barrier();
    return (int32_t)(*w - target) >= 0 ? 0 : -1;
}

static int ref_poll_word(void *ctx, OmegaGpuBufferHandle b, uint64_t off, uint32_t expected, uint64_t timeout_ms, uint32_t *observed) {
    AienRef *r = ctx;
    ref_lock(r);
    int rc = buf_ok(r, b, NULL);
    if (rc == OMEGA_GPU_PLAT_OK && !(r->rights[b.index] & OMEGA_GPU_RIGHT_HOST_POLLED)) rc = OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS;
    if (rc == OMEGA_GPU_PLAT_OK && (off & 3u)) rc = OMEGA_GPU_PLAT_REFUSED_ARGS;
    if (rc == OMEGA_GPU_PLAT_OK) rc = range_ok(r, b, off, 4u);
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "poll_word"); ref_unlock(r); return rc; }
    volatile uint32_t *w = (volatile uint32_t *)((uint8_t *)r->mem[b.index].cpu + off);
    ref_unlock(r); /* never hold the table lock while polling */
    rc = wait_ge(w, expected, timeout_ms) == 0 ? OMEGA_GPU_PLAT_OK : OMEGA_GPU_POLL_TIMED_OUT;
    if (observed) *observed = *w;
    return rc;
}

static int queue_ok(AienRef *r, OmegaGpuQueueHandle q) {
    return omega_gpu_handles_check(&r->t, q.index, q.generation, OMEGA_GPU_HANDLE_QUEUE);
}

/* GPFIFO ring + USERD in one GPU-uncached buffer (the CPU publishes both; nvrm_channel), the error notifier,
 * AGPU_CHAN_CREATE, the doorbell page. Called with r->mu held; -1 with r->err set and nothing left behind. */
static int channel_create(AienRef *r) {
    if (mem_map(r, OMEGA_GPU_AIEN_FIFO_BYTES, 0, AGPU_MEM_GPU_UNCACHED, &r->fifo) != 0) return -1;
    if (mem_map(r, OMEGA_GPU_AIEN_NOTIFIER_BYTES, 0, AGPU_MEM_GPU_UNCACHED, &r->notifier) != 0) {
        (void)mem_unmap(r, &r->fifo, 1);
        return -1;
    }
    struct agpu_chan_create c;
    memset(&c, 0, sizeof c);
    c.ring_handle = r->fifo.handle;
    c.entries = OMEGA_GPU_AIEN_GP_ENTRIES;
    c.ring_off = 0;
    c.userd_handle = r->fifo.handle;
    c.err_handle = r->notifier.handle;
    c.userd_off = OMEGA_GPU_AIEN_USERD_OFF;
    if (rm_call(r, AGPU_CHAN_CREATE, &c, &c.st, "AGPU_CHAN_CREATE") != 0) {
        (void)mem_unmap(r, &r->fifo, 1); (void)mem_unmap(r, &r->notifier, 1);
        return -1;
    }
    r->channel = c.channel;
    r->token = c.token;
    void *db = r->sys->map(NULL, OMEGA_GPU_AIEN_DOORBELL_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, r->fd, c.doorbell_off);
    if (db == MAP_FAILED) {
        char keep[sizeof r->err];
        snprintf(keep, sizeof keep, "mmap doorbell page (offset 0x%llx): %s", (unsigned long long)c.doorbell_off, strerror(errno));
        (void)channel_release(r, 1);
        memcpy(r->err, keep, sizeof keep);
        return -1;
    }
    r->doorbell = db;
    r->put = r->retired = 0;
    return 0;
}

static int ref_queue_open(void *ctx, OmegaGpuDeviceHandle dev, OmegaGpuBufferHandle ring, OmegaGpuQueueHandle *out) {
    AienRef *r = ctx;
    uint32_t kind = 0;
    if (!out) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    memset(out, 0, sizeof *out);
    ref_lock(r);
    int rc = dev_ok(r, dev);
    if (rc == OMEGA_GPU_PLAT_OK) rc = buf_ok(r, ring, &kind);
    if (rc == OMEGA_GPU_PLAT_OK && kind != OMEGA_GPU_HANDLE_BUF_QUEUE) rc = OMEGA_GPU_PLAT_REFUSED_BAD_KIND;
    if (rc == OMEGA_GPU_PLAT_OK && r->queue.generation != 0u) rc = OMEGA_GPU_PLAT_REFUSED_LIMIT;
    if (rc == OMEGA_GPU_PLAT_OK) rc = omega_gpu_handles_open(&r->t, OMEGA_GPU_HANDLE_QUEUE, 0, 1, 0, &out->index, &out->generation);
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "queue_open"); ref_unlock(r); return rc; }
    if (channel_create(r) != 0) {
        (void)omega_gpu_handles_close(&r->t, out->index, out->generation, OMEGA_GPU_HANDLE_QUEUE);
        memset(out, 0, sizeof *out);
        rc = refuse(r, OMEGA_GPU_PLAT_REFUSED_DEVICE, "aien_gpu_chan_create");
        ref_unlock(r);
        return rc;
    }
    r->ring_index = ring.index;
    r->queue = *out;
    ref_unlock(r);
    return OMEGA_GPU_PLAT_OK;
}

static int ref_queue_close(void *ctx, OmegaGpuQueueHandle q) {
    AienRef *r = ctx;
    ref_lock(r);
    int rc = queue_ok(r, q);
    if (rc == OMEGA_GPU_PLAT_OK && channel_release(r, 0) != 0) rc = OMEGA_GPU_PLAT_REFUSED_DEVICE;
    if (rc == OMEGA_GPU_PLAT_OK) rc = omega_gpu_handles_close(&r->t, q.index, q.generation, OMEGA_GPU_HANDLE_QUEUE);
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "queue_close"); ref_unlock(r); return rc; }
    ref_unlock(r);
    return OMEGA_GPU_PLAT_OK;
}

/* Every check comes before the first write, so a refused submit rang no doorbell and changed no ring state. */
static int ref_queue_submit(void *ctx, OmegaGpuQueueHandle q, const uint32_t *words, uint32_t nwords) {
    AienRef *r = ctx;
    ref_lock(r);
    int rc = queue_ok(r, q);
    if (rc == OMEGA_GPU_PLAT_OK && (!words || nwords == 0u)) rc = OMEGA_GPU_PLAT_REFUSED_ARGS;
    const uint32_t pre = r->l2_inval_pending ? L2_INVAL_PREFIX_WORDS : 0u;
    const AienMem *pb = rc == OMEGA_GPU_PLAT_OK ? &r->mem[r->ring_index] : NULL;
    const uint64_t total = (uint64_t)nwords + pre;
    if (rc == OMEGA_GPU_PLAT_OK && (total * 4u > pb->size || total > AIEN_GP_MAX_WORDS)) rc = OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH;
    if (rc == OMEGA_GPU_PLAT_OK && r->put - r->retired >= OMEGA_GPU_AIEN_GP_ENTRIES - 1u) {
        snprintf(r->err, sizeof r->err, "GPFIFO full: %u entries unretired", r->put - r->retired);
        rc = OMEGA_GPU_PLAT_REFUSED_DEVICE;
    }
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "submit_methods"); ref_unlock(r); return rc; }
    uint32_t *dst = pb->cpu;
    if (pre) memcpy(dst, L2_INVAL_PREFIX, sizeof L2_INVAL_PREFIX);
    memcpy(dst + pre, words, (size_t)nwords * 4u);
    barrier();
    volatile uint64_t *ring = (volatile uint64_t *)r->fifo.cpu;
    ring[r->put % OMEGA_GPU_AIEN_GP_ENTRIES] = omega_gpu_aien_gp_entry((uint64_t)(uintptr_t)pb->cpu, (uint32_t)total);
    r->put++;
    barrier();
    *(volatile uint32_t *)((uint8_t *)r->fifo.cpu + OMEGA_GPU_AIEN_USERD_OFF + OMEGA_GPU_AIEN_USERD_GPPUT) =
        r->put % OMEGA_GPU_AIEN_GP_ENTRIES;
    barrier();
    *(volatile uint32_t *)(r->doorbell + OMEGA_GPU_AIEN_DOORBELL_WORD) = r->token;
    if (pre) r->l2_inval_pending = 0;
    ref_unlock(r);
    return OMEGA_GPU_PLAT_OK;
}

static int ref_queue_retire(void *ctx, OmegaGpuQueueHandle q) {
    AienRef *r = ctx;
    ref_lock(r);
    int rc = queue_ok(r, q);
    if (rc == OMEGA_GPU_PLAT_OK) r->retired = r->put; /* the caller saw the work complete */
    else rc = refuse(r, rc, "queue_retire");
    ref_unlock(r);
    return rc;
}

static void *ref_host_alloc(void *ctx, size_t bytes) { (void)ctx; return bytes ? malloc(bytes) : NULL; }
static void ref_host_free(void *ctx, void *p) { (void)ctx; free(p); }

static const OmegaGpuPlatform g_platform = {
    .ctx = &g_ref,
    .time_now_ms = ref_now_ms,
    .wait_ms = ref_wait_ms,
    .lock = ref_lock_id,
    .unlock = ref_unlock_id,
    .mmio_read32 = NULL,  /* not provided: the module owns the registers */
    .mmio_write32 = NULL,
    .dma_alloc = ref_dma_alloc,
    .dma_free = ref_dma_free,
    .sync_cpu_to_gpu = ref_sync_to_gpu,
    .sync_gpu_to_cpu = ref_sync_to_cpu,
    .poll_word = ref_poll_word,
    .irq_wait = NULL,     /* polling is not interrupt support */
    .device_open = ref_device_open,
    .device_close = ref_device_close,
    .diag_text = ref_diag_text,
    .queue_open = ref_queue_open,
    .queue_close = ref_queue_close,
    .queue_submit = ref_queue_submit,
    .queue_retire = ref_queue_retire,
    .time_now_ns = ref_now_ns,
    .host_alloc = ref_host_alloc,
    .host_free = ref_host_free,
};

const OmegaGpuPlatform *omega_gpu_platform_aien(OmegaGpuPlatformCaps *caps) {
    if (caps) *caps = g_caps;
    return &g_platform;
}
