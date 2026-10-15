/*
 * omega_gpu_session.c: FB-1 cut 4b, made portable in native v1 P07 cut C (session part). See the header.
 *
 * The launch words are the ones omega_gpu_matmul_api.c (cut 1b, chip PASS FB1-CUT1B-4345406) and
 * omega_gpu_elementwise_api.c (cut 4, FB1-CUT4-0d0241a) both emitted; cut C did not change one of them. What
 * changed is where the machine comes from: every device, buffer, queue, clock, lock and host-memory call goes
 * through the installed OmegaGpuPlatform (ABI 1.1), so this file builds freestanding (OMEGA_GPU_NO_POSIX) with no
 * Linux header and no libc call. The hosted build installs the Linux reference (src/omega_gpu_platform_linux.c)
 * on first use; it makes the same driver calls in the same order as this file did before.
 *
 * No read-modify-write atomics: the freestanding target must not pull libgcc outline atomics, so shared words are
 * plain atomic loads and stores, and counters are only written with the session lock held.
 */
#include "omega_gpu_session.h"
#include "omega_gpu_code_alloc.h"
#include "omega_gpu_fmt.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_submit.h" /* launch constants only (macros); the vector executor itself stays Linux-private */
#include <string.h>

#ifndef OMEGA_GPU_NO_POSIX
#include "omega_gpu_platform_linux.h"
#include <pthread.h>
#endif

#define PAGE 0x1000ull

/* Push-buffer method header (the nvrm_mthd encoding, physics nvrm.h): increasing, count words, subchannel. */
static inline uint32_t bw_mthd(uint32_t subc, uint32_t mthd, uint32_t count) {
    return (2u << 28) | (count << 16) | (subc << 13) | (mthd >> 2);
}

/* Same words as src/omega_blackwell_submit.c SETUP_WORDS (compute class setup). */
static const uint32_t SETUP_WORDS[18] = {
    0x20012061, 0x0000cec0, 0x20012092, 0x00000001, 0x200120a8, 0x0000000f, 0x2001255d, 0x00000003,
    0x2001255e, 0x20000000, 0x2001255f, 0x000fffff, 0x20012557, 0x00000003, 0x20012558, 0x22000000,
    0x20012559, 0x00000000,
};

static const OmegaGpuPlatform *g_p; /* installed platform (atomic load / store) */
static OmegaGpuPlatformCaps g_caps;
static OmegaGpuSession g_s;
static bool g_open;
static uint32_t g_blocked;  /* latched (atomic) */
static uint32_t g_locked;   /* session lock held (atomic) */
static uint32_t g_cancel;   /* cancel requested (atomic) */
static uint32_t g_spin_us;  /* omega_gpu_session_set_spin_us; 0 = poll_word only */
static char g_err[320] = "";
static void (*g_hooks[8])(void);
static unsigned g_nhooks;
static OmegaGpuAllocStats g_stats; /* written with the lock held, read with atomic loads */
static OmegaGpuBufferHandle g_parked[OMEGA_GPU_SESSION_PARK_MAX];
static OmegaGpuParkStats g_park;

#define LOAD(x) __atomic_load_n(&(x), __ATOMIC_ACQUIRE)
#define STORE(x, v) __atomic_store_n(&(x), (v), __ATOMIC_RELEASE)
#define BUMP(x, by) STORE((x), LOAD(x) + (by)) /* lock held: one writer */

/* ---- platform ---- */

static int install(const OmegaGpuPlatform *p, const OmegaGpuPlatformCaps *caps) {
    if (!p || !caps) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    OmegaGpuPlatformCaps req = {
        .abi_major = OMEGA_GPU_PLATFORM_ABI_MAJOR, .abi_minor = OMEGA_GPU_PLATFORM_ABI_MINOR,
        .services = OMEGA_GPU_SVC_REQUIRED_SESSION,
        .max_devices = 1, .max_buffers = 8, .max_queues = 1, .max_inflight = 1,
        .max_buffer_bytes = 0x10000, .max_total_bytes = 0x10000 + 3 * 0x10000,
    };
    int rc = omega_gpu_platform_negotiate(caps, &req);
    if (rc == OMEGA_GPU_PLAT_OK) rc = omega_gpu_platform_check_for(p, caps, OMEGA_GPU_SVC_REQUIRED_SESSION);
    if (rc != OMEGA_GPU_PLAT_OK) return rc;
    g_caps = *caps;
    STORE(g_p, p);
    return OMEGA_GPU_PLAT_OK;
}

int omega_gpu_session_set_platform(const OmegaGpuPlatform *p, const OmegaGpuPlatformCaps *caps) {
    if (g_open || LOAD(g_locked) || LOAD(g_blocked)) return OMEGA_GPU_PLAT_REFUSED_BUSY;
    if (!p) { STORE(g_p, (const OmegaGpuPlatform *)0); return OMEGA_GPU_PLAT_OK; }
    return install(p, caps);
}

#ifndef OMEGA_GPU_NO_POSIX
static pthread_once_t g_default_once = PTHREAD_ONCE_INIT;
static void install_default(void) {
    if (LOAD(g_p)) return; /* one was installed explicitly */
    OmegaGpuPlatformCaps c;
    const OmegaGpuPlatform *p = omega_gpu_platform_linux(&c);
    (void)install(p, &c);
}
#endif

static const OmegaGpuPlatform *platform(void) {
    const OmegaGpuPlatform *p = LOAD(g_p);
#ifndef OMEGA_GPU_NO_POSIX
    if (!p) { pthread_once(&g_default_once, install_default); p = LOAD(g_p); }
#endif
    return p;
}

const OmegaGpuPlatform *omega_gpu_session_platform(void) { return platform(); }
uint32_t omega_gpu_session_submit_prefix_words(void) { return platform() ? g_caps.submit_prefix_words : 0u; } /* g_caps is written before g_p is published */
OmegaGpuCompletionMode omega_gpu_session_completion_mode(void) { return OMEGA_GPU_COMPLETION_POLLING; }

uint64_t omega_gpu_session_now_ns(void) {
    const OmegaGpuPlatform *p = platform();
    return p ? p->time_now_ns(p->ctx) : 0u;
}

void *omega_gpu_host_alloc(size_t bytes) {
    const OmegaGpuPlatform *p = platform();
    return p ? p->host_alloc(p->ctx, bytes ? bytes : 1) : (void *)0; /* 0 bytes: a unique 1-byte block, as malloc(0) gives */
}
void *omega_gpu_host_calloc(size_t n, size_t size) {
    size_t bytes;
    if (__builtin_mul_overflow(n, size, &bytes)) return (void *)0;
    void *q = omega_gpu_host_alloc(bytes);
    if (q) memset(q, 0, bytes);
    return q;
}
void omega_gpu_host_free(void *q) {
    const OmegaGpuPlatform *p = platform();
    if (p && q) p->host_free(p->ctx, q);
}

/* ---- errors ---- */

const char *omega_gpu_session_last_error(void) { return g_err; }
void omega_gpu_session_set_error(const char *stage) { omega_gpu_fmt(g_err, sizeof g_err, "%s", stage ? stage : ""); }

/* A platform refusal: the driver text when the device refused (the Linux reference writes "<driver call>: <Nvrm.err>",
 * the stage names the session ever reported), else the stage and the refusal name. */
static void fail_plat(const char *stage, int rc) {
    const OmegaGpuPlatform *p = LOAD(g_p);
    const char *diag = p ? p->diag_text(p->ctx) : "";
    if (rc == OMEGA_GPU_PLAT_REFUSED_DEVICE && diag && diag[0])
        omega_gpu_fmt(g_err, sizeof g_err, "%s", diag);
    else
        omega_gpu_fmt(g_err, sizeof g_err, "%s: %s", stage, omega_gpu_platform_refusal_name(rc));
}

/* ---- lock, state ---- */

void omega_gpu_session_lock(void) {
    const OmegaGpuPlatform *p = platform();
    if (!p) return;
    (void)p->lock(p->ctx, OMEGA_GPU_SESSION_LOCK_ID);
    STORE(g_locked, 1u);
}
void omega_gpu_session_unlock(void) {
    const OmegaGpuPlatform *p = LOAD(g_p);
    if (!p) return;
    STORE(g_locked, 0u);
    (void)p->unlock(p->ctx, OMEGA_GPU_SESSION_LOCK_ID);
}
bool omega_gpu_session_is_open(void) { return g_open; }
int omega_gpu_session_is_blocked(void) { return LOAD(g_blocked) ? 1 : 0; }
int omega_gpu_session_set_spin_us(uint32_t us) {
    if (us > OMEGA_GPU_SESSION_MAX_SPIN_US) return -1;
    STORE(g_spin_us, us);
    return 0;
}
uint32_t omega_gpu_session_spin_us(void) { return LOAD(g_spin_us); }
void omega_gpu_session_cancel(void) { STORE(g_cancel, 1u); }

/* Consumes a pending cancel request. */
static bool take_cancel(void) {
    if (!LOAD(g_cancel)) return false;
    STORE(g_cancel, 0u);
    return true;
}

void omega_gpu_session_alloc_stats(OmegaGpuAllocStats *out) {
    if (!out) return;
    out->allocs = LOAD(g_stats.allocs);
    out->alloc_bytes = LOAD(g_stats.alloc_bytes);
    out->alloc_failures = LOAD(g_stats.alloc_failures);
    out->frees = LOAD(g_stats.frees);
    out->stale_refusals = LOAD(g_stats.stale_refusals);
}

/* ---- parking ---- */

static void park(const OmegaGpuMem *m) {
    uint32_t n = LOAD(g_park.buffers) - LOAD(g_park.untracked);
    for (uint32_t i = 0; i < n; i++)
        if (g_parked[i].index == m->handle.index && g_parked[i].generation == m->handle.generation) return;
    if (n < OMEGA_GPU_SESSION_PARK_MAX) g_parked[n] = m->handle;
    else BUMP(g_park.untracked, 1u);
    BUMP(g_park.bytes, m->size);
    BUMP(g_park.buffers, 1u);
}

void omega_gpu_session_parked(OmegaGpuParkStats *out) {
    if (!out) return;
    out->buffers = LOAD(g_park.buffers);
    out->untracked = LOAD(g_park.untracked);
    out->bytes = LOAD(g_park.bytes);
}

int omega_gpu_session_is_parked(OmegaGpuBufferHandle h) {
    uint32_t n = LOAD(g_park.buffers) - LOAD(g_park.untracked);
    if (h.generation == 0u) return 0;
    for (uint32_t i = 0; i < n && i < OMEGA_GPU_SESSION_PARK_MAX; i++)
        if (g_parked[i].index == h.index && g_parked[i].generation == h.generation) return 1;
    return 0;
}

/* Uncertain completion: the device may still touch any session buffer. Keep everything, refuse everything. */
static void latch(const char *why) {
    STORE(g_blocked, 1u);
    if (g_s.pb.cpu) park(&g_s.pb);
    if (g_s.cbank.cpu) park(&g_s.cbank);
    if (g_s.marker.cpu) park(&g_s.marker);
    if (g_s.qmd.cpu) park(&g_s.qmd);
    omega_gpu_session_set_error(why);
}

/* ---- buffers ---- */

static size_t page_up(size_t b) { b = (b + PAGE - 1) & ~(PAGE - 1); return b < PAGE ? PAGE : b; }

static int mem_alloc(uint32_t kind, uint32_t rights, size_t bytes, OmegaGpuMem *out) {
    const OmegaGpuPlatform *p = LOAD(g_p);
    memset(out, 0, sizeof *out);
    if (!g_open || LOAD(g_blocked) || !p) { omega_gpu_session_set_error("alloc without open device"); return -1; }
    const size_t want = page_up(bytes);
    OmegaGpuBufferDesc d = {.kind = kind, .rights = rights, .gpu_cache = OMEGA_GPU_CACHE_GPU_UNCACHED, .align = (uint32_t)PAGE, .bytes = want};
    OmegaCpuView cpu; OmegaGpuVa va; OmegaGpuBusAddr bus; /* the bus address never leaves this function */
    int rc = p->dma_alloc(p->ctx, g_s.dev, &d, &out->handle, &cpu, &va, &bus);
    if (rc != OMEGA_GPU_PLAT_OK || !cpu.ptr || cpu.len < want) {
        if (rc == OMEGA_GPU_PLAT_OK) { (void)p->dma_free(p->ctx, out->handle); rc = OMEGA_GPU_PLAT_REFUSED_LIMIT; }
        BUMP(g_stats.alloc_failures, 1u);
        fail_plat("dma_alloc", rc);
        memset(out, 0, sizeof *out);
        return -1;
    }
    out->kind = kind; out->cpu = cpu.ptr; out->va = va.va; out->size = cpu.len;
    BUMP(g_stats.allocs, 1u);
    BUMP(g_stats.alloc_bytes, (uint64_t)want);
    return 0;
}

#define RIGHTS_DATA (OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_READ | OMEGA_GPU_RIGHT_GPU_WRITE)
#define RIGHTS_CODE (OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_READ | OMEGA_GPU_RIGHT_GPU_EXEC)
#define RIGHTS_RING (OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_READ)
#define RIGHTS_MARKER (OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_WRITE | OMEGA_GPU_RIGHT_HOST_POLLED)
#define RIGHTS_CBANK (OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_READ)

int omega_gpu_session_alloc(size_t bytes, OmegaGpuMem *out) {
    return mem_alloc(OMEGA_GPU_HANDLE_BUF_DATA, RIGHTS_DATA, bytes, out);
}

int omega_gpu_session_alloc_code(size_t code_size, OmegaGpuMem *out) {
    size_t bytes = omega_gpu_code_alloc_bytes(code_size);
    if (bytes == 0) { omega_gpu_session_set_error("code size overflow"); memset(out, 0, sizeof *out); return -1; }
    if (mem_alloc(OMEGA_GPU_HANDLE_BUF_CODE, RIGHTS_CODE, bytes, out) != 0) return -1;
    memset((uint8_t *)out->cpu + code_size, 0, bytes - code_size);
    return 0;
}

void omega_gpu_session_free(OmegaGpuMem *m) {
    if (!m) return;
    if (m->cpu) {
        const OmegaGpuPlatform *p = LOAD(g_p);
        if (LOAD(g_blocked)) {
            park(m); /* the device may still use it */
        } else if (!g_open || !p) {
            BUMP(g_stats.stale_refusals, 1u); /* its device is closed: the handle is stale, nothing to release */
        } else {
            int rc = p->dma_free(p->ctx, m->handle);
            if (rc == OMEGA_GPU_PLAT_OK) BUMP(g_stats.frees, 1u);
            else { BUMP(g_stats.stale_refusals, 1u); fail_plat("dma_free", rc); }
        }
    }
    memset(m, 0, sizeof *m);
}

static int sync_mem(const OmegaGpuMem *m, int to_gpu) {
    const OmegaGpuPlatform *p = LOAD(g_p);
    if (!m || !m->cpu) return 0;
    if (!p || !g_open) { omega_gpu_session_set_error("sync without open device"); return -1; }
    int rc = to_gpu ? p->sync_cpu_to_gpu(p->ctx, m->handle, 0, m->size) : p->sync_gpu_to_cpu(p->ctx, m->handle, 0, m->size);
    if (rc != OMEGA_GPU_PLAT_OK) { fail_plat(to_gpu ? "sync_cpu_to_gpu" : "sync_gpu_to_cpu", rc); return -1; }
    return 0;
}
int omega_gpu_session_sync_to_gpu(const OmegaGpuMem *m) { return sync_mem(m, 1); }
int omega_gpu_session_sync_to_cpu(const OmegaGpuMem *m) { return sync_mem(m, 0); }

int omega_gpu_session_scratch(OmegaGpuScratch *s, size_t bytes) {
    if (s->reserved && (bytes > s->cap || !s->mem.cpu)) {
        omega_gpu_fmt(g_err, sizeof g_err, "serving reservation exceeded: need %zu bytes, reserved %zu", bytes, s->cap);
        return OMEGA_GPU_SESSION_SCRATCH_RESERVED;
    }
    if (bytes > s->high_water) s->high_water = bytes;
    if (s->cap >= bytes && s->mem.cpu) return 0;
    omega_gpu_session_free(&s->mem);
    s->cap = 0;
    if (omega_gpu_session_alloc(s->high_water, &s->mem) != 0) return -1;
    s->cap = (size_t)s->mem.size;
    return 0;
}

void omega_gpu_session_scratch_free(OmegaGpuScratch *s) {
    omega_gpu_session_free(&s->mem);
    s->cap = 0;
    s->reserved = false;
}

int omega_gpu_session_scratch_reserve(OmegaGpuScratch *s, size_t bytes) {
    s->reserved = false; /* grow normally first */
    if (omega_gpu_session_scratch(s, bytes) != 0) return -1;
    s->reserved = true;
    return 0;
}

void omega_gpu_session_scratch_unreserve(OmegaGpuScratch *s) { s->reserved = false; }

static void free_session_buffers(void) {
    omega_gpu_session_free(&g_s.qmd); omega_gpu_session_free(&g_s.marker);
    omega_gpu_session_free(&g_s.cbank); omega_gpu_session_free(&g_s.pb);
}

/* ---- open, close ---- */

OmegaGpuSession *omega_gpu_session_open(void) {
    if (LOAD(g_blocked)) { omega_gpu_session_set_error("process latched"); return NULL; }
    if (g_open) return &g_s;
    const OmegaGpuPlatform *p = platform();
    if (!p) { omega_gpu_session_set_error("no GPU platform installed (omega_gpu_session_set_platform)"); return NULL; }
    uint64_t t0 = p->time_now_ns(p->ctx);
    const uint32_t opens = g_s.opens;
    memset(&g_s, 0, sizeof g_s);
    g_s.opens = opens;
    int rc = p->device_open(p->ctx, 0, &g_s.dev);
    if (rc != OMEGA_GPU_PLAT_OK) { fail_plat("device_open", rc); memset(&g_s.dev, 0, sizeof g_s.dev); return NULL; }
    g_open = true; /* so alloc / free work during the rest of the open */
    /* same four buffers, same sizes, same order as before cut C; every one GPU-uncached */
    if (mem_alloc(OMEGA_GPU_HANDLE_BUF_QUEUE, RIGHTS_RING, 0x10000, &g_s.pb) ||
        mem_alloc(OMEGA_GPU_HANDLE_BUF_DATA, RIGHTS_CBANK, PAGE, &g_s.cbank) ||
        mem_alloc(OMEGA_GPU_HANDLE_BUF_DATA, RIGHTS_MARKER, PAGE, &g_s.marker) ||
        mem_alloc(OMEGA_GPU_HANDLE_BUF_DATA, RIGHTS_DATA, 0x10000, &g_s.qmd) ||
        (rc = p->queue_open(p->ctx, g_s.dev, g_s.pb.handle, &g_s.queue)) != OMEGA_GPU_PLAT_OK) {
        char keep[sizeof g_err];
        if (rc != OMEGA_GPU_PLAT_OK) fail_plat("queue_open", rc);
        memcpy(keep, g_err, sizeof keep);
        free_session_buffers();
        g_open = false;
        (void)p->device_close(p->ctx, g_s.dev);
        memset(&g_s.dev, 0, sizeof g_s.dev);
        omega_gpu_fmt(g_err, sizeof g_err, "session scratch alloc (%s)", keep);
        return NULL;
    }
    g_s.opens++;
    g_s.last_open_ns = p->time_now_ns(p->ctx) - t0;
    return &g_s;
}

int omega_gpu_session_on_close(void (*hook)(void)) { /* caller holds the lock (the APIs register from their open path) */
    if (!hook) return -1;
    int rc = 0;
    bool seen = false;
    for (unsigned i = 0; i < g_nhooks; i++) if (g_hooks[i] == hook) seen = true;
    if (!seen) { if (g_nhooks < sizeof g_hooks / sizeof g_hooks[0]) g_hooks[g_nhooks++] = hook; else rc = -1; }
    return rc;
}

void omega_gpu_session_close(void) {
    omega_gpu_session_lock();
    const OmegaGpuPlatform *p = LOAD(g_p);
    if (g_open && !LOAD(g_blocked) && p) {
        for (unsigned i = 0; i < g_nhooks; i++) g_hooks[i]();
        if (!LOAD(g_blocked)) { /* a hook's launch may have latched: then nothing below runs */
            (void)p->queue_close(p->ctx, g_s.queue);
            memset(&g_s.queue, 0, sizeof g_s.queue);
            free_session_buffers();
            (void)p->device_close(p->ctx, g_s.dev); /* every handle of this device is stale from here */
            memset(&g_s.dev, 0, sizeof g_s.dev);
            g_open = false;
        }
    }
    omega_gpu_session_unlock();
}

/* ---- launch ---- */

static uint32_t round_up(uint32_t v, uint32_t q) { return (v + q - 1) / q * q; }

static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

#define WAIT_CANCELLED 1
#define WAIT_TIMED_OUT 2
#define WAIT_REFUSED 3

/* Marker wait by POLLING. With spin_us, the host first re-reads the uncached marker word back to back (with the
 * ARM yield hint) so a launch that finishes inside the window is seen within about one read. Then the platform's
 * poll_word in slices of OMEGA_GPU_SESSION_POLL_SLICE_MS (Linux reference: the m16 50 us sleep-poll) until
 * timeout_ms, checking for a cancel request between slices. */
static int session_wait(const OmegaGpuPlatform *p, uint64_t off, uint32_t want, uint64_t timeout_ms, uint32_t spin_us) {
    volatile uint32_t *m = (volatile uint32_t *)((uint8_t *)g_s.marker.cpu + off);
    if (spin_us) {
        const uint64_t end = p->time_now_ns(p->ctx) + (uint64_t)spin_us * 1000u;
        do {
            if ((int32_t)(*m - want) >= 0) return 0;
            if (LOAD(g_cancel)) return WAIT_CANCELLED;
            cpu_relax();
        } while (p->time_now_ns(p->ctx) < end);
    }
    uint64_t waited = 0;
    for (;;) {
        if (LOAD(g_cancel)) return WAIT_CANCELLED;
        const uint64_t left = timeout_ms - waited;
        const uint64_t slice = left < OMEGA_GPU_SESSION_POLL_SLICE_MS ? left : OMEGA_GPU_SESSION_POLL_SLICE_MS;
        uint32_t seen = 0;
        int rc = p->poll_word(p->ctx, g_s.marker.handle, off, want, slice, &seen);
        if (rc == OMEGA_GPU_PLAT_OK) return 0;
        if (rc != OMEGA_GPU_POLL_TIMED_OUT) return WAIT_REFUSED;
        waited += slice;
        if (waited >= timeout_ms) return WAIT_TIMED_OUT;
    }
}

int omega_gpu_session_launch(const OmegaGpuLaunch *L, uint64_t *elapsed_ns, uint32_t *marker_out) {
    const OmegaGpuPlatform *p = LOAD(g_p);
    if (!g_open || LOAD(g_blocked) || !p) {
        omega_gpu_session_set_error(LOAD(g_blocked) ? "process latched" : "launch without open device"); return -1;
    }
    if (!L || !L->args || L->n_args == 0 || L->n_args > OMEGA_GPU_SESSION_MAX_ARGS || L->threads_x == 0 || L->grid_x == 0 || L->grid_y == 0 ||
        L->spin_us > OMEGA_GPU_SESSION_MAX_SPIN_US) {
        omega_gpu_session_set_error("launch arguments"); return -1;
    }
    if (take_cancel()) { omega_gpu_session_set_error("launch cancelled before submit"); return OMEGA_GPU_SESSION_CANCELLED; }
    const uint32_t threads_y = L->threads_y ? L->threads_y : 1;
    uint32_t cbank_data[OMEGA_BW_CBANK_DRIVER_WORDS];
    omega_blackwell_build_cbank_driver_2d(cbank_data, g_s.cbank.va, L->threads_x, threads_y, L->grid_x, L->grid_y);
    memcpy(g_s.cbank.cpu, cbank_data, sizeof cbank_data);
    memcpy((uint8_t *)g_s.cbank.cpu + 0x380, L->args, (size_t)L->n_args * 4u);

    const uint64_t qmd0_va = g_s.qmd.va, qmd1_va = g_s.qmd.va + 0x1000, sem_va = g_s.qmd.va + 0x2000, scratch_va = g_s.qmd.va + 0x4000;
    uint32_t gpr = L->gpr_count > 64u ? round_up(L->gpr_count, 8u) : 64u;
    OmegaBlackwellQmdConfig cfg = {
        .code_va = L->code_va, .cbank_va = g_s.cbank.va, .scratch_va = scratch_va, .sem_va = sem_va,
        .qmd0_va = qmd0_va, .qmd1_va = qmd1_va, .num_elements = L->num_elements,
        .threads_per_block = L->threads_x, .grid_width = L->grid_x * L->grid_y, .threads_x = L->threads_x, .threads_y = threads_y,
        .grid_x = L->grid_x, .grid_y = L->grid_y, .gpr_count = gpr,
        .shared_bytes = L->shared_bytes,
    };
    uint32_t qmd0_words[OMEGA_BW_QMD_WORDS], qmd1_words[OMEGA_BW_QMD_WORDS];
    omega_blackwell_build_qmd0(qmd0_words, qmd0_va, qmd1_va);
    if (omega_blackwell_build_qmd1(qmd1_words, &cfg) != 0) { omega_gpu_session_set_error("qmd build (shared_bytes)"); return -1; }
    if (omega_blackwell_verify_qmd_invariants(qmd1_words) != 0) { omega_gpu_session_set_error("qmd invariants"); return -1; }
    memcpy(g_s.qmd.cpu, qmd0_words, sizeof qmd0_words);
    memcpy((uint8_t *)g_s.qmd.cpu + 0x1000, qmd1_words, sizeof qmd1_words);

    volatile uint32_t *hsem = (volatile uint32_t *)((uint8_t *)g_s.qmd.cpu + 0x2000);
    volatile uint32_t *hmarker = (volatile uint32_t *)g_s.marker.cpu;
    volatile uint32_t *hmarker2 = (volatile uint32_t *)((uint8_t *)g_s.marker.cpu + 0x10);
    *hsem = 0; *hmarker = 0; *hmarker2 = 0;
    /* explicit visibility of everything the CPU just wrote (Linux reference: dsb sy, the barrier issued here before) */
    int rc = p->sync_cpu_to_gpu(p->ctx, g_s.cbank.handle, 0, PAGE);
    if (rc == OMEGA_GPU_PLAT_OK) rc = p->sync_cpu_to_gpu(p->ctx, g_s.qmd.handle, 0, 0x3000);
    if (rc == OMEGA_GPU_PLAT_OK) rc = p->sync_cpu_to_gpu(p->ctx, g_s.marker.handle, 0, 0x20);
    if (rc != OMEGA_GPU_PLAT_OK) { fail_plat("sync_cpu_to_gpu", rc); return -1; }

    uint32_t pb[1024];
    size_t n = 0;
    memcpy(&pb[n], SETUP_WORDS, sizeof SETUP_WORDS); n += sizeof SETUP_WORDS / 4;
    /* Invalidate the SM instruction, constant and data caches before this launch
     * (cut 1b finding: a reused code address served stale instructions; clcec0.h
     * NVCEC0_INVALIDATE_SHADER_CACHES 0x021c, bits INSTRUCTION 0, DATA 4, CONSTANT 12). */
    pb[n++] = bw_mthd(1, OMEGA_BW_MTHD_INVALIDATE_SHADER_CACHES, 1); pb[n++] = OMEGA_BW_INVALIDATE_SHADER_CACHES_ALL;
    /* constant bank 0: driver words, then our arguments at 0x380 (inline upload) */
    pb[n++] = bw_mthd(1, 0x0188, 2); pb[n++] = (uint32_t)(g_s.cbank.va >> 32); pb[n++] = (uint32_t)g_s.cbank.va;
    pb[n++] = bw_mthd(1, 0x0180, 2); pb[n++] = 0x00000380; pb[n++] = 0x00000001;
    pb[n++] = bw_mthd(1, 0x01b0, 1); pb[n++] = 0x00000041;
    pb[n++] = (224u << 16) | (1u << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[n], cbank_data, 224 * 4); n += 224;
    pb[n++] = bw_mthd(1, 0x0188, 2); pb[n++] = (uint32_t)((g_s.cbank.va + 0x380) >> 32); pb[n++] = (uint32_t)(g_s.cbank.va + 0x380);
    pb[n++] = bw_mthd(1, 0x0180, 2); pb[n++] = L->n_args * 4u; pb[n++] = 0x00000001;
    pb[n++] = bw_mthd(1, 0x01b0, 1); pb[n++] = 0x00000041;
    pb[n++] = (L->n_args << 16) | (1u << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[n], L->args, (size_t)L->n_args * 4u); n += L->n_args;
    /* QMD0 (null grid, dependence 2) */
    pb[n++] = (98u << 16) | (1u << 13) | (0x0318 >> 2) | (2u << 28);
    pb[n++] = (1u << 30) | (uint32_t)((qmd0_va >> 40) & 0x1ff); pb[n++] = (uint32_t)(qmd0_va >> 8);
    memcpy(&pb[n], qmd0_words, 96 * 4); n += 96;
    /* intermediate semaphore init */
    pb[n++] = bw_mthd(1, 0x0188, 2); pb[n++] = (uint32_t)(sem_va >> 32); pb[n++] = (uint32_t)sem_va;
    pb[n++] = bw_mthd(1, 0x0180, 2); pb[n++] = 0x00000004; pb[n++] = 0x00000001;
    pb[n++] = bw_mthd(1, 0x01b0, 1); pb[n++] = 0x00000041;
    pb[n++] = (1u << 16) | (1u << 13) | (0x01b4 >> 2) | (6u << 28);
    pb[n++] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT;
    /* QMD1 (the kernel) */
    pb[n++] = (98u << 16) | (1u << 13) | (0x0318 >> 2) | (2u << 28);
    pb[n++] = (1u << 30) | (uint32_t)((qmd1_va >> 40) & 0x1ff); pb[n++] = (uint32_t)(qmd1_va >> 8);
    memcpy(&pb[n], qmd1_words, 96 * 4); n += 96;
    /* first marker */
    pb[n++] = bw_mthd(0, 0x005c, 5);
    pb[n++] = (uint32_t)g_s.marker.va; pb[n++] = (uint32_t)(g_s.marker.va >> 32);
    pb[n++] = OMEGA_BW_MARKER_COMPLETION_PAYLOAD; pb[n++] = 0; pb[n++] = 0x1 | (1u << 20);
    /* C3 tail: L2_FLUSH_DIRTY, then the second release on the uncached marker page */
    pb[n++] = bw_mthd(0, 0x0028, 4); pb[n++] = 0; pb[n++] = 0; pb[n++] = 0; pb[n++] = (0x10u << 27);
    pb[n++] = bw_mthd(0, 0x005c, 5);
    pb[n++] = (uint32_t)(g_s.marker.va + 0x10); pb[n++] = (uint32_t)((g_s.marker.va + 0x10) >> 32);
    pb[n++] = OMEGA_BW_MARKER2_PAYLOAD; pb[n++] = 0; pb[n++] = 0x1 | (1u << 20);

    const uint32_t spin_us = L->spin_us ? L->spin_us : omega_gpu_session_spin_us();
    uint64_t t0 = p->time_now_ns(p->ctx);
    rc = p->queue_submit(p->ctx, g_s.queue, pb, (uint32_t)n);
    if (rc != OMEGA_GPU_PLAT_OK) {
        fail_plat("submit_methods", rc); /* contract: a refused submit rang no doorbell, nothing is in flight */
        return -1;
    }
    int w = session_wait(p, 0, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, L->timeout_ms, spin_us);
    if (w == 0) w = session_wait(p, 0x10, OMEGA_BW_MARKER2_PAYLOAD, L->timeout_ms, spin_us);
    if (w != 0) {
        /* uncertain completion: keep everything (parked), refuse every later call */
        if (w == WAIT_CANCELLED) {
            (void)take_cancel();
            latch("launch cancelled after submit (completion uncertain, latched)");
            return OMEGA_GPU_SESSION_CANCELLED;
        }
        latch(w == WAIT_TIMED_OUT ? "marker wait timed out (latched)" : "marker poll refused (latched)");
        return -1;
    }
    (void)p->sync_gpu_to_cpu(p->ctx, g_s.marker.handle, 0, 0x20);
    (void)p->sync_gpu_to_cpu(p->ctx, g_s.qmd.handle, 0x2000, 4);
    uint64_t t1 = p->time_now_ns(p->ctx);
    (void)p->queue_retire(p->ctx, g_s.queue); /* both releases landed: the GPFIFO entry is consumed */
    if (*hsem != OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE) { omega_gpu_session_set_error("intermediate semaphore not DONE"); return -1; }
    if (elapsed_ns) *elapsed_ns += t1 - t0;
    if (marker_out) *marker_out = *hmarker;
    (void)hmarker2;
    return 0;
}

uint32_t omega_gpu_session_open_count(void) { return g_s.opens; }
