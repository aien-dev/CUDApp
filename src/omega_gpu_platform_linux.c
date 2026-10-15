/* omega_gpu_platform_linux.c -- Linux REFERENCE implementation of OmegaGpuPlatform 1.1. REFERENCE ONLY: see the
 * header. Every driver call here is the one src/omega_gpu_session.c made directly before P07 cut C, in the same
 * order, so the launch path on GB10 under Linux is unchanged:
 *   device_open  = nvrm_open + m16_native_create_channel   (session open, before the four session buffers)
 *   dma_alloc    = nvrm_alloc_gpu_uncached                  (every session / API buffer)
 *   dma_free     = nvrm_free
 *   queue_open   = ctx.pb_mem = the ring buffer             (m16_native_enqueue_methods copies methods there)
 *   queue_submit = m16_native_submit_methods
 *   queue_retire = nvrm_retire(&rm, rm.put)                 (after both markers landed)
 *   poll_word    = m16_native_wait_marker (>= serial compare, 50 us sleep-poll)
 *   sync_*       = dsb sy                                   (the barrier the session issued itself)
 *   device_close = m16_native_close (nvrm_close)
 */
#include "omega_gpu_platform_linux.h"

#include "m16_native.h"
#include "nvrm.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LINUX_REF_BUFFERS (OMEGA_GPU_HANDLE_SLOTS - 2u) /* one slot each for the device and the queue */

typedef struct {
    pthread_mutex_t mu; /* table, device and diagnostics */
    pthread_mutex_t locks[OMEGA_GPU_PLATFORM_LINUX_LOCKS];
    int ready;
    OmegaGpuHandleTable t;
    NvrmMem mem[OMEGA_GPU_HANDLE_SLOTS];
    uint32_t rights[OMEGA_GPU_HANDLE_SLOTS];
    uint8_t gpu_cached[OMEGA_GPU_HANDLE_SLOTS]; /* allocated GPU-cached (nvrm_alloc), omega #457 */
    int l2_inval_pending; /* a host sync into a GPU-cached buffer since the last submission, omega #457 */
    M16NativeContext m16;
    OmegaGpuDeviceHandle dev; /* generation 0 = closed */
    OmegaGpuQueueHandle queue;
    char diag[320];
} LinuxRef;

static LinuxRef g_ref = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .locks = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER, PTHREAD_MUTEX_INITIALIZER},
};

/* omega #457: MEM_OP_A..D = 0, 0, 0, OPERATION L2_SYSMEM_NCOH_INVALIDATE 0x11 in D bits 31:27 (clc96f.h:30-73), the
 * same method form as the C3 tail's L2_FLUSH_DIRTY. ref_queue_submit puts it ahead of the next submission after a host
 * sync into a GPU-cached buffer; g_caps.submit_prefix_words is its size, so callers budget exactly what is written. */
static const uint32_t L2_INVAL_PREFIX[] = { (2u << 28) | (4u << 16) | (0x0028u >> 2), 0, 0, 0, 0x11u << 27 };
#define L2_INVAL_PREFIX_WORDS ((uint32_t)(sizeof L2_INVAL_PREFIX / sizeof L2_INVAL_PREFIX[0]))

static const OmegaGpuPlatformCaps g_caps = {
    .submit_prefix_words = L2_INVAL_PREFIX_WORDS,
    .abi_major = OMEGA_GPU_PLATFORM_ABI_MAJOR,
    .abi_minor = OMEGA_GPU_PLATFORM_ABI_MINOR,
    .services = OMEGA_GPU_SVC_REQUIRED_SESSION, /* no MMIO (RM owns the registers), no IRQ (polling only) */
    .max_devices = 1,
    .max_buffers = LINUX_REF_BUFFERS,
    .max_queues = 1,
    .max_inflight = 1,
    .max_buffer_bytes = 1ull << 37,  /* 128 GiB: the whole unified pool */
    .max_total_bytes = 1ull << 37,
};

static void ref_lock(LinuxRef *r) {
    pthread_mutex_lock(&r->mu);
    if (!r->ready) { omega_gpu_handles_init(&r->t); r->ready = 1; }
}
static void ref_unlock(LinuxRef *r) { pthread_mutex_unlock(&r->mu); }

/* Records why the call failed; REFUSED_DEVICE carries the driver text (Nvrm.err). Called with r->mu held. */
static int refuse(LinuxRef *r, int rc, const char *stage) {
    if (rc == OMEGA_GPU_PLAT_REFUSED_DEVICE)
        snprintf(r->diag, sizeof r->diag, "%s: %s", stage, r->m16.rm.err[0] ? r->m16.rm.err : "(no driver text)");
    else
        snprintf(r->diag, sizeof r->diag, "%s: %s", stage, omega_gpu_platform_refusal_name(rc));
    return rc;
}

static int dev_ok(LinuxRef *r, OmegaGpuDeviceHandle d) {
    return omega_gpu_handles_check(&r->t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
}

/* A live buffer handle of any buffer kind; *kind gets it. */
static int buf_ok(LinuxRef *r, OmegaGpuBufferHandle b, uint32_t *kind) {
    if (b.index >= OMEGA_GPU_HANDLE_SLOTS || b.generation == 0u) return OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE;
    if (!r->t.live[b.index] || r->t.generation[b.index] != b.generation) return OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE;
    const uint32_t k = r->t.kind[b.index];
    if (k != OMEGA_GPU_HANDLE_BUF_DATA && k != OMEGA_GPU_HANDLE_BUF_CODE && k != OMEGA_GPU_HANDLE_BUF_QUEUE)
        return OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE;
    if (kind) *kind = k;
    return OMEGA_GPU_PLAT_OK;
}

static int range_ok(LinuxRef *r, OmegaGpuBufferHandle b, uint64_t off, uint64_t len) {
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
    LinuxRef *r = ctx;
    if (id >= OMEGA_GPU_PLATFORM_LINUX_LOCKS) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    return pthread_mutex_lock(&r->locks[id]) == 0 ? OMEGA_GPU_PLAT_OK : OMEGA_GPU_PLAT_REFUSED_BUSY;
}
static int ref_unlock_id(void *ctx, uint32_t id) {
    LinuxRef *r = ctx;
    if (id >= OMEGA_GPU_PLATFORM_LINUX_LOCKS) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    return pthread_mutex_unlock(&r->locks[id]) == 0 ? OMEGA_GPU_PLAT_OK : OMEGA_GPU_PLAT_REFUSED_ARGS;
}

static int ref_device_open(void *ctx, uint32_t ordinal, OmegaGpuDeviceHandle *out) {
    LinuxRef *r = ctx;
    int rc;
    if (!out) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    memset(out, 0, sizeof *out);
    ref_lock(r);
    if (ordinal != 0u) { rc = refuse(r, OMEGA_GPU_PLAT_REFUSED_ARGS, "device_open"); goto out; }
    if (r->dev.generation != 0u) { rc = refuse(r, OMEGA_GPU_PLAT_REFUSED_LIMIT, "device_open (already open)"); goto out; }
    memset(&r->m16, 0, sizeof r->m16);
    /* nvrm_open directly (m16_native_open is memset + nvrm_open + nvrm_close on failure) so the driver's own
     * failure text survives into the stage name, as the session did before cut C. */
    if (nvrm_open(&r->m16.rm) != 0) {
        rc = refuse(r, OMEGA_GPU_PLAT_REFUSED_DEVICE, "nvrm_open");
        (void)nvrm_close(&r->m16.rm);
        goto out;
    }
    if (m16_native_create_channel(&r->m16) != 0) {
        rc = refuse(r, OMEGA_GPU_PLAT_REFUSED_DEVICE, "m16_native_create_channel");
        (void)m16_native_close(&r->m16);
        goto out;
    }
    rc = omega_gpu_handles_open(&r->t, OMEGA_GPU_HANDLE_DEVICE, 0, 1, 0, &out->index, &out->generation);
    if (rc != OMEGA_GPU_PLAT_OK) { (void)m16_native_close(&r->m16); rc = refuse(r, rc, "device_open"); goto out; }
    r->dev = *out;
    r->diag[0] = '\0';
out:
    ref_unlock(r);
    return rc;
}

static int ref_device_close(void *ctx, OmegaGpuDeviceHandle dev) {
    LinuxRef *r = ctx;
    ref_lock(r);
    int rc = dev_ok(r, dev);
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "device_close"); ref_unlock(r); return rc; }
    /* Every handle of the device goes stale. The RM client close below releases the driver objects; buffers the
     * caller did not free are not unmapped here, exactly as before cut C. */
    for (uint32_t i = 0; i < OMEGA_GPU_HANDLE_SLOTS; i++) {
        if (!r->t.live[i] || i == dev.index) continue;
        (void)omega_gpu_handles_close(&r->t, i, r->t.generation[i], r->t.kind[i]);
        memset(&r->mem[i], 0, sizeof r->mem[i]);
        r->rights[i] = 0;
        r->gpu_cached[i] = 0;
    }
    (void)omega_gpu_handles_close(&r->t, dev.index, dev.generation, OMEGA_GPU_HANDLE_DEVICE);
    memset(&r->dev, 0, sizeof r->dev);
    r->l2_inval_pending = 0; /* omega #457: nothing pending survives the device */
    memset(&r->queue, 0, sizeof r->queue);
    (void)m16_native_close(&r->m16);
    ref_unlock(r);
    return OMEGA_GPU_PLAT_OK;
}

static const char *ref_diag_text(void *ctx) { return ((LinuxRef *)ctx)->diag; }

static int ref_dma_alloc(void *ctx, OmegaGpuDeviceHandle dev, const OmegaGpuBufferDesc *desc, OmegaGpuBufferHandle *out,
                         OmegaCpuView *cpu, OmegaGpuVa *gpu, OmegaGpuBusAddr *bus) {
    LinuxRef *r = ctx;
    if (!desc || !out || !cpu || !gpu) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    memset(out, 0, sizeof *out); memset(cpu, 0, sizeof *cpu); memset(gpu, 0, sizeof *gpu);
    if (bus) memset(bus, 0, sizeof *bus);
    ref_lock(r);
    int rc = dev_ok(r, dev);
    if (rc == OMEGA_GPU_PLAT_OK) rc = omega_gpu_platform_check_buffer(desc, &g_caps);
    /* GPU-uncached by default. GPU-cached only for DEVICE-PRIVATE data buffers (OWN-30B T3b, 2026-10-13): no CPU_READ, no
     * HOST_POLLED, so the CPU never reads a line the GPU may still hold in L2 (nvos.h:1116 "For system memory this will not be
     * coherent with direct CPU mappings"). omega_gpu_mem_cached.c (OWN-30B L2) writes such a buffer from the CPU exactly once,
     * before any GPU read, then syncs; omega_gpu_devmem.h (T3b) fills them with GPU writes only. Either way no stale L2 line exists. */
    const int cached = desc->gpu_cache == OMEGA_GPU_CACHE_GPU_CACHED;
    if (rc == OMEGA_GPU_PLAT_OK && cached &&
        (desc->kind != OMEGA_GPU_HANDLE_BUF_DATA || (desc->rights & (OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_HOST_POLLED))))
        rc = OMEGA_GPU_PLAT_REFUSED_CACHE_POLICY;
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "dma_alloc"); ref_unlock(r); return rc; }
    uint32_t idx = 0, gen = 0;
    rc = omega_gpu_handles_open(&r->t, desc->kind, desc->bytes, LINUX_REF_BUFFERS, g_caps.max_total_bytes, &idx, &gen);
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "dma_alloc"); ref_unlock(r); return rc; }
    NvrmMem m;
    if ((cached ? nvrm_alloc(&r->m16.rm, desc->bytes, &m) : nvrm_alloc_gpu_uncached(&r->m16.rm, desc->bytes, &m)) != 0) {
        (void)omega_gpu_handles_close(&r->t, idx, gen, desc->kind); /* this slot never got memory: its generation moves on */
        rc = refuse(r, OMEGA_GPU_PLAT_REFUSED_DEVICE, cached ? "nvrm_alloc" : "nvrm_alloc_gpu_uncached");
        ref_unlock(r);
        return rc;
    }
    r->mem[idx] = m;
    r->rights[idx] = desc->rights;
    r->gpu_cached[idx] = (uint8_t)cached;
    out->index = idx; out->generation = gen;
    cpu->ptr = m.cpu; cpu->len = (size_t)m.size;
    gpu->va = m.va;
    ref_unlock(r);
    return OMEGA_GPU_PLAT_OK;
}

static int ref_dma_free(void *ctx, OmegaGpuBufferHandle b) {
    LinuxRef *r = ctx;
    uint32_t kind = 0;
    ref_lock(r);
    int rc = buf_ok(r, b, &kind);
    if (rc == OMEGA_GPU_PLAT_OK && r->queue.generation != 0u && r->m16.pb_mem.handle == r->mem[b.index].handle &&
        r->m16.pb_mem.cpu == r->mem[b.index].cpu)
        rc = OMEGA_GPU_PLAT_REFUSED_BUSY; /* the open queue still uses it as its ring */
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "dma_free"); ref_unlock(r); return rc; }
    (void)nvrm_free(&r->m16.rm, &r->mem[b.index]);
    (void)omega_gpu_handles_close(&r->t, b.index, b.generation, kind);
    memset(&r->mem[b.index], 0, sizeof r->mem[b.index]);
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
    LinuxRef *r = ctx;
    ref_lock(r);
    int rc = buf_ok(r, b, NULL);
    if (rc == OMEGA_GPU_PLAT_OK) rc = range_ok(r, b, off, len);
    if (rc != OMEGA_GPU_PLAT_OK) rc = refuse(r, rc, stage);
    /* omega #457 (chip, 2026-10-15): after the host rewrites a GPU-cached buffer the GPU already read, every later launch
     * read the old bytes from L2 (nvos.h:1117 "For system memory this will not be coherent with direct CPU mappings").
     * The next submission therefore starts with L2_SYSMEM_NCOH_INVALIDATE (ref_queue_submit). */
    else if (to_gpu && r->gpu_cached[b.index]) r->l2_inval_pending = 1;
    ref_unlock(r);
    if (rc == OMEGA_GPU_PLAT_OK) barrier(); /* uncached system memory: a full barrier is the whole sync */
    return rc;
}
static int ref_sync_to_gpu(void *ctx, OmegaGpuBufferHandle b, uint64_t off, uint64_t len) { return ref_sync(ctx, b, off, len, "sync_cpu_to_gpu", 1); }
static int ref_sync_to_cpu(void *ctx, OmegaGpuBufferHandle b, uint64_t off, uint64_t len) { return ref_sync(ctx, b, off, len, "sync_gpu_to_cpu", 0); }

static int ref_poll_word(void *ctx, OmegaGpuBufferHandle b, uint64_t off, uint32_t expected, uint64_t timeout_ms, uint32_t *observed) {
    LinuxRef *r = ctx;
    ref_lock(r);
    int rc = buf_ok(r, b, NULL);
    if (rc == OMEGA_GPU_PLAT_OK && !(r->rights[b.index] & OMEGA_GPU_RIGHT_HOST_POLLED)) rc = OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS;
    if (rc == OMEGA_GPU_PLAT_OK && (off & 3u)) rc = OMEGA_GPU_PLAT_REFUSED_ARGS;
    if (rc == OMEGA_GPU_PLAT_OK) rc = range_ok(r, b, off, 4u);
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "poll_word"); ref_unlock(r); return rc; }
    volatile uint32_t *w = (volatile uint32_t *)((uint8_t *)r->mem[b.index].cpu + off);
    ref_unlock(r); /* never hold the table lock while polling */
    rc = m16_native_wait_marker(w, expected, timeout_ms) == 0 ? OMEGA_GPU_PLAT_OK : OMEGA_GPU_POLL_TIMED_OUT;
    if (observed) *observed = *w;
    return rc;
}

static int queue_ok(LinuxRef *r, OmegaGpuQueueHandle q) {
    return omega_gpu_handles_check(&r->t, q.index, q.generation, OMEGA_GPU_HANDLE_QUEUE);
}

static int ref_queue_open(void *ctx, OmegaGpuDeviceHandle dev, OmegaGpuBufferHandle ring, OmegaGpuQueueHandle *out) {
    LinuxRef *r = ctx;
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
    r->m16.pb_mem = r->mem[ring.index]; /* m16_native_enqueue_methods copies the method stream here */
    r->queue = *out;
    ref_unlock(r);
    return OMEGA_GPU_PLAT_OK;
}

static int ref_queue_close(void *ctx, OmegaGpuQueueHandle q) {
    LinuxRef *r = ctx;
    ref_lock(r);
    int rc = queue_ok(r, q);
    if (rc == OMEGA_GPU_PLAT_OK) rc = omega_gpu_handles_close(&r->t, q.index, q.generation, OMEGA_GPU_HANDLE_QUEUE);
    if (rc != OMEGA_GPU_PLAT_OK) { rc = refuse(r, rc, "queue_close"); ref_unlock(r); return rc; }
    memset(&r->queue, 0, sizeof r->queue);
    memset(&r->m16.pb_mem, 0, sizeof r->m16.pb_mem);
    ref_unlock(r);
    return OMEGA_GPU_PLAT_OK;
}

static int ref_queue_submit(void *ctx, OmegaGpuQueueHandle q, const uint32_t *words, uint32_t nwords) {
    LinuxRef *r = ctx;
    ref_lock(r);
    int rc = queue_ok(r, q);
    if (rc == OMEGA_GPU_PLAT_OK && (!words || nwords == 0u)) rc = OMEGA_GPU_PLAT_REFUSED_ARGS;
    /* omega #457: a host sync into a GPU-cached buffer is pending, so this submission starts with L2_INVAL_PREFIX.
     * No dirty line is dropped: every earlier submission ended with L2_FLUSH_DIRTY and the host waited for it.
     * DOCS SILENT on ordering between this MEM_OP and the launch that follows (no WFI is inserted); the chip gate
     * tools/run_gpu_l2_stale_chip.sh (20/20 FRESH) is the only proof. The prefix counts against the ring like any
     * word; callers budget it from g_caps.submit_prefix_words (omega_gpu_cmdlist.c). */
    const uint32_t pre = r->l2_inval_pending ? L2_INVAL_PREFIX_WORDS : 0u;
    if (rc == OMEGA_GPU_PLAT_OK && ((uint64_t)nwords + pre) * 4u > r->m16.pb_mem.size) rc = OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH;
    uint32_t *buf = NULL; const char *stage = "submit_methods";
    if (rc == OMEGA_GPU_PLAT_OK && pre) {
        buf = malloc(((size_t)nwords + pre) * 4u);
        if (!buf) { rc = OMEGA_GPU_PLAT_REFUSED_LIMIT; stage = "submit_methods (host copy for the #457 L2 invalidate prefix)"; }
        else {
            memcpy(buf, L2_INVAL_PREFIX, sizeof L2_INVAL_PREFIX);
            memcpy(buf + pre, words, (size_t)nwords * 4u);
        }
    }
    if (rc == OMEGA_GPU_PLAT_OK && m16_native_submit_methods(&r->m16, buf ? buf : words, nwords + pre) != 0) rc = OMEGA_GPU_PLAT_REFUSED_DEVICE;
    if (rc == OMEGA_GPU_PLAT_OK && pre) r->l2_inval_pending = 0;
    free(buf);
    if (rc != OMEGA_GPU_PLAT_OK) rc = refuse(r, rc, stage);
    ref_unlock(r);
    return rc;
}

static int ref_queue_retire(void *ctx, OmegaGpuQueueHandle q) {
    LinuxRef *r = ctx;
    ref_lock(r);
    int rc = queue_ok(r, q);
    if (rc == OMEGA_GPU_PLAT_OK) nvrm_retire(&r->m16.rm, r->m16.rm.put); /* both releases landed: the entry is consumed */
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
    .mmio_read32 = NULL,  /* not provided: the RM driver owns the registers */
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

const OmegaGpuPlatform *omega_gpu_platform_linux(OmegaGpuPlatformCaps *caps) {
    if (caps) *caps = g_caps;
    return &g_platform;
}
