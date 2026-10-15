/* Contract tests for src/omega_gpu_platform.h (native v1 P07 cut A). Host
 * program, links the FREESTANDING object of src/omega_gpu_platform.c. Opens
 * no device. Every rule in the header comment has a case here, and every
 * refusal name is reached at least once. */
#include "omega_gpu_platform.h"

#include <stdio.h>
#include <string.h>

static int n_pass, n_fail, case_ok;
#define EXPECT(c) do { if (!(c)) { case_ok = 0; printf("  expect failed at line %d: %s\n", __LINE__, #c); } } while (0)
static void case_begin(void) { case_ok = 1; }
static void case_end(const char *id)
{
    if (case_ok) n_pass++; else n_fail++;
    printf("%s %s\n", case_ok ? "PASS" : "FAIL", id);
}

/* ---- a complete fake platform (no device behind it) ---------------------- */
static uint64_t f_now(void *c) { (void)c; return 1u; }
static int f_wait(void *c, uint64_t ms) { (void)c; (void)ms; return 0; }
static int f_lock(void *c, uint32_t id) { (void)c; (void)id; return 0; }
static int f_rd(void *c, OmegaGpuDeviceHandle d, uint32_t r, uint64_t o, uint32_t *v) { (void)c; (void)d; (void)r; (void)o; *v = 0; return 0; }
static int f_wr(void *c, OmegaGpuDeviceHandle d, uint32_t r, uint64_t o, uint32_t v) { (void)c; (void)d; (void)r; (void)o; (void)v; return 0; }
static int f_alloc(void *c, OmegaGpuDeviceHandle d, const OmegaGpuBufferDesc *desc, OmegaGpuBufferHandle *o,
                   OmegaCpuView *cpu, OmegaGpuVa *g, OmegaGpuBusAddr *b)
{ (void)c; (void)d; (void)desc; (void)o; (void)cpu; (void)g; (void)b; return 0; }
static int f_free(void *c, OmegaGpuBufferHandle b) { (void)c; (void)b; return 0; }
static int f_sync(void *c, OmegaGpuBufferHandle b, uint64_t o, uint64_t l) { (void)c; (void)b; (void)o; (void)l; return 0; }
static int f_word(void *c, OmegaGpuBufferHandle b, uint64_t o, uint32_t e, uint64_t t, uint32_t *v) { (void)c; (void)b; (void)o; (void)e; (void)t; *v = 0; return 0; }
static int f_irq(void *c, OmegaGpuDeviceHandle d, uint64_t t) { (void)c; (void)d; (void)t; return 0; }

static void full_platform(OmegaGpuPlatform *p)
{
    memset(p, 0, sizeof *p);
    p->time_now_ms = f_now; p->wait_ms = f_wait; p->lock = f_lock; p->unlock = f_lock;
    p->mmio_read32 = f_rd; p->mmio_write32 = f_wr; p->dma_alloc = f_alloc; p->dma_free = f_free;
    p->sync_cpu_to_gpu = f_sync; p->sync_gpu_to_cpu = f_sync; p->poll_word = f_word;
}

static void caps_v1(OmegaGpuPlatformCaps *c)
{
    memset(c, 0, sizeof *c);
    c->abi_major = OMEGA_GPU_PLATFORM_ABI_MAJOR; c->abi_minor = OMEGA_GPU_PLATFORM_ABI_MINOR;
    c->services = OMEGA_GPU_SVC_REQUIRED_V1;
    c->max_devices = 1; c->max_buffers = 8; c->max_queues = 2; c->max_inflight = 1;
    c->max_buffer_bytes = 1u << 20; c->max_total_bytes = 4u << 20;
}

static void case_names(void)
{
    int i, j;
    case_begin();
    for (i = 0; i < (int)OMEGA_GPU_PLAT_REFUSAL_COUNT; i++) {
        EXPECT(strcmp(omega_gpu_platform_refusal_name(i), "UNKNOWN") != 0);
        for (j = 0; j < i; j++) EXPECT(strcmp(omega_gpu_platform_refusal_name(i), omega_gpu_platform_refusal_name(j)) != 0);
    }
    EXPECT(strcmp(omega_gpu_platform_refusal_name(-1), "UNKNOWN") == 0);
    EXPECT(strcmp(omega_gpu_platform_refusal_name(OMEGA_GPU_PLAT_REFUSAL_COUNT), "UNKNOWN") == 0);
    EXPECT(strcmp(omega_gpu_platform_refusal_name(0), "OK") == 0);
    case_end("REFUSAL_NAMES_DISTINCT");
}

static void case_negotiate(void)
{
    OmegaGpuPlatformCaps p, r;
    case_begin();
    caps_v1(&p); caps_v1(&r);
    EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_platform_negotiate(NULL, &r) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    EXPECT(omega_gpu_platform_negotiate(&p, NULL) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    /* a newer provided minor is fine; an older one or another major is refused */
    p.abi_minor = r.abi_minor + 1u;
    EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_OK);
    caps_v1(&p); r.abi_minor = p.abi_minor + 1u;
    EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_REFUSED_ABI);
    caps_v1(&r); p.abi_major = r.abi_major + 1u;
    EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_REFUSED_ABI);
    /* a required service the platform lacks is refused, never emulated */
    caps_v1(&p); r.services |= OMEGA_GPU_SVC_IRQ;
    EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_REFUSED_MISSING_SERVICE);
    p.services |= OMEGA_GPU_SVC_IRQ;
    EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_OK);
    /* unknown service bits are malformed */
    p.services |= 1u << 12; /* the first bit past ABI 1.1 */
    EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    /* every limit */
    caps_v1(&p); caps_v1(&r); r.max_devices = 2;      EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    caps_v1(&r); r.max_buffers = 9;                   EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    caps_v1(&r); r.max_queues = 3;                    EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    caps_v1(&r); r.max_inflight = 2;                  EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    caps_v1(&r); r.max_buffer_bytes = (1u << 20) + 1u; EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    caps_v1(&r); r.max_total_bytes = (4u << 20) + 1u;  EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    /* asking for less is fine */
    caps_v1(&r); r.max_buffers = 1; r.max_total_bytes = 1;
    EXPECT(omega_gpu_platform_negotiate(&p, &r) == OMEGA_GPU_PLAT_OK);
    case_end("NEGOTIATE_ABI_SERVICES_LIMITS");
}

static void case_check_table(void)
{
    OmegaGpuPlatform p;
    OmegaGpuPlatformCaps c;
    case_begin();
    full_platform(&p); caps_v1(&c);
    EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_platform_check(NULL, &c) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    EXPECT(omega_gpu_platform_check(&p, NULL) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    /* the v1 required set is mandatory */
    c.services = OMEGA_GPU_SVC_REQUIRED_V1 & ~OMEGA_GPU_SVC_SYNC;
    EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_REFUSED_MISSING_SERVICE);
    /* a claimed service with a missing function, one per service */
    caps_v1(&c);
    full_platform(&p); p.time_now_ms = NULL;     EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    full_platform(&p); p.wait_ms = NULL;         EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    full_platform(&p); p.unlock = NULL;          EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    full_platform(&p); p.mmio_write32 = NULL;    EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    full_platform(&p); p.dma_free = NULL;        EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    full_platform(&p); p.sync_gpu_to_cpu = NULL; EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    full_platform(&p); p.poll_word = NULL;       EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    /* IRQ: claimed needs irq_wait; polling-only platforms do not claim it and must not carry one */
    full_platform(&p); c.services |= OMEGA_GPU_SVC_IRQ;
    EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    p.irq_wait = f_irq;
    EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_OK);
    caps_v1(&c);
    EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_REFUSED_ARGS); /* undeclared function = hidden path */
    case_end("SERVICE_TABLE_MATCHES_CLAIMS_NO_HIDDEN_FALLBACK");
}


/* ABI 1.1: the services the persistent GPU session requires (OMEGA_GPU_SVC_REQUIRED_SESSION). */
static int f_dopen(void *c, uint32_t o, OmegaGpuDeviceHandle *d) { (void)c; (void)o; d->index = 0; d->generation = 1; return 0; }
static int f_dclose(void *c, OmegaGpuDeviceHandle d) { (void)c; (void)d; return 0; }
static const char *f_diag(void *c) { (void)c; return ""; }
static int f_qopen(void *c, OmegaGpuDeviceHandle d, OmegaGpuBufferHandle r, OmegaGpuQueueHandle *q) { (void)c; (void)d; (void)r; q->index = 0; q->generation = 1; return 0; }
static int f_qclose(void *c, OmegaGpuQueueHandle q) { (void)c; (void)q; return 0; }
static int f_qsubmit(void *c, OmegaGpuQueueHandle q, const uint32_t *w, uint32_t n) { (void)c; (void)q; (void)w; (void)n; return 0; }
static int f_qretire(void *c, OmegaGpuQueueHandle q) { (void)c; (void)q; return 0; }
static uint64_t f_ns(void *c) { (void)c; return 1u; }
static void *f_halloc(void *c, size_t b) { (void)c; (void)b; return NULL; }
static void f_hfree(void *c, void *p) { (void)c; (void)p; }
static void session_platform(OmegaGpuPlatform *p, OmegaGpuPlatformCaps *c)
{
    full_platform(p);
    p->mmio_read32 = NULL; p->mmio_write32 = NULL; /* the session needs no MMIO: a platform without it is fine */
    p->device_open = f_dopen; p->device_close = f_dclose; p->diag_text = f_diag;
    p->queue_open = f_qopen; p->queue_close = f_qclose; p->queue_submit = f_qsubmit; p->queue_retire = f_qretire;
    p->time_now_ns = f_ns; p->host_alloc = f_halloc; p->host_free = f_hfree;
    caps_v1(c);
    c->services = OMEGA_GPU_SVC_REQUIRED_SESSION;
}

static void case_session_services(void)
{
    OmegaGpuPlatform p;
    OmegaGpuPlatformCaps c;
    case_begin();
    EXPECT(OMEGA_GPU_PLATFORM_ABI_MINOR == 2u); /* 1.2: caps.submit_prefix_words (#459) */
    EXPECT(!(OMEGA_GPU_SVC_REQUIRED_SESSION & (OMEGA_GPU_SVC_IRQ | OMEGA_GPU_SVC_MMIO)));
    session_platform(&p, &c);
    EXPECT(omega_gpu_platform_check_for(&p, &c, OMEGA_GPU_SVC_REQUIRED_SESSION) == OMEGA_GPU_PLAT_OK);
    /* the v1 check still demands MMIO, so a session-only platform is not a v1 device-owner platform */
    EXPECT(omega_gpu_platform_check(&p, &c) == OMEGA_GPU_PLAT_REFUSED_MISSING_SERVICE);
    EXPECT(omega_gpu_platform_check_for(&p, &c, OMEGA_GPU_SVC_ALL + 1u) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    /* each new service: a missing function is refused */
    session_platform(&p, &c); p.device_close = NULL;  EXPECT(omega_gpu_platform_check_for(&p, &c, OMEGA_GPU_SVC_REQUIRED_SESSION) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    session_platform(&p, &c); p.diag_text = NULL;     EXPECT(omega_gpu_platform_check_for(&p, &c, OMEGA_GPU_SVC_REQUIRED_SESSION) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    session_platform(&p, &c); p.queue_retire = NULL;  EXPECT(omega_gpu_platform_check_for(&p, &c, OMEGA_GPU_SVC_REQUIRED_SESSION) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    session_platform(&p, &c); p.time_now_ns = NULL;   EXPECT(omega_gpu_platform_check_for(&p, &c, OMEGA_GPU_SVC_REQUIRED_SESSION) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    session_platform(&p, &c); p.host_free = NULL;     EXPECT(omega_gpu_platform_check_for(&p, &c, OMEGA_GPU_SVC_REQUIRED_SESSION) == OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE);
    /* a required service not claimed is missing; a function of an unclaimed service is a hidden path */
    session_platform(&p, &c); c.services &= ~OMEGA_GPU_SVC_QUEUE;
    EXPECT(omega_gpu_platform_check_for(&p, &c, OMEGA_GPU_SVC_REQUIRED_SESSION) == OMEGA_GPU_PLAT_REFUSED_MISSING_SERVICE);
    EXPECT(omega_gpu_platform_check_for(&p, &c, c.services) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    /* polling never claims IRQ: a session platform with an irq_wait it does not claim is refused */
    session_platform(&p, &c); p.irq_wait = f_irq;
    EXPECT(omega_gpu_platform_check_for(&p, &c, OMEGA_GPU_SVC_REQUIRED_SESSION) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    EXPECT(omega_gpu_platform_negotiate(&c, &c) == OMEGA_GPU_PLAT_OK);
    EXPECT(OMEGA_GPU_POLL_TIMED_OUT >= (int)OMEGA_GPU_PLAT_REFUSAL_COUNT);
    case_end("SESSION_SERVICES_ABI_1_1_POLLING_NOT_IRQ");
}
static void case_buffers(void)
{
    OmegaGpuBufferDesc d;
    OmegaGpuPlatformCaps c;
    case_begin();
    caps_v1(&c);
    memset(&d, 0, sizeof d);
    d.kind = OMEGA_GPU_HANDLE_BUF_DATA; d.rights = OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_READ; d.bytes = 4096;
    EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_platform_check_buffer(NULL, &c) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    EXPECT(omega_gpu_platform_check_buffer(&d, NULL) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    d.kind = OMEGA_GPU_HANDLE_QUEUE; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_BAD_KIND);
    d.kind = OMEGA_GPU_HANDLE_BUF_DATA;
    d.bytes = 0; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH);
    d.bytes = (1u << 20) + 1u; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    d.bytes = 4096;
    d.align = 48; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    d.align = 64; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_OK);
    d.gpu_cache = 7; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    d.gpu_cache = OMEGA_GPU_CACHE_GPU_CACHED;
    d.rights = 0; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS);
    d.rights = 1u << 7; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS);
    /* exec only on code */
    d.rights = OMEGA_GPU_RIGHT_GPU_EXEC | OMEGA_GPU_RIGHT_CPU_WRITE;
    EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS);
    d.kind = OMEGA_GPU_HANDLE_BUF_CODE;
    EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_OK);
    d.rights |= OMEGA_GPU_RIGHT_GPU_WRITE; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS);
    d.rights = OMEGA_GPU_RIGHT_CPU_WRITE; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS); /* code without exec */
    d.rights = OMEGA_GPU_RIGHT_GPU_EXEC; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS); /* nobody can fill it */
    /* queue: CPU writes, device reads */
    d.kind = OMEGA_GPU_HANDLE_BUF_QUEUE; d.rights = OMEGA_GPU_RIGHT_CPU_WRITE;
    EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS);
    d.rights |= OMEGA_GPU_RIGHT_GPU_READ; EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_OK);
    /* host-polled: CPU reads, GPU writes, GPU-uncached (the E1 rule) */
    d.kind = OMEGA_GPU_HANDLE_BUF_DATA;
    d.rights = OMEGA_GPU_RIGHT_HOST_POLLED | OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_GPU_WRITE;
    d.gpu_cache = OMEGA_GPU_CACHE_GPU_CACHED;
    EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_CACHE_POLICY);
    d.gpu_cache = OMEGA_GPU_CACHE_GPU_UNCACHED;
    EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_OK);
    d.rights = OMEGA_GPU_RIGHT_HOST_POLLED | OMEGA_GPU_RIGHT_CPU_READ;
    EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS);
    d.kind = OMEGA_GPU_HANDLE_BUF_CODE; d.rights = OMEGA_GPU_RIGHT_HOST_POLLED | OMEGA_GPU_RIGHT_CPU_WRITE | OMEGA_GPU_RIGHT_GPU_EXEC | OMEGA_GPU_RIGHT_CPU_READ;
    EXPECT(omega_gpu_platform_check_buffer(&d, &c) == OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS);
    case_end("BUFFER_DESC_KIND_LENGTH_RIGHTS_CACHE");
}

static void case_handles(void)
{
    static OmegaGpuHandleTable t;
    uint32_t i0, g0, i1, g1, i2, g2, i, g;
    uint32_t n;
    case_begin();
    omega_gpu_handles_init(&t);
    EXPECT(t.nlive == 0 && t.total_bytes == 0);
    EXPECT(omega_gpu_handles_open(NULL, OMEGA_GPU_HANDLE_DEVICE, 0, 1, 0, &i0, &g0) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    EXPECT(omega_gpu_handles_open(&t, 0, 0, 1, 0, &i0, &g0) == OMEGA_GPU_PLAT_REFUSED_BAD_KIND);
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_KIND_COUNT, 0, 1, 0, &i0, &g0) == OMEGA_GPU_PLAT_REFUSED_BAD_KIND);
    /* one device allowed */
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_DEVICE, 0, 1, 0, &i0, &g0) == OMEGA_GPU_PLAT_OK);
    EXPECT(g0 != 0u);
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_DEVICE, 0, 1, 0, &i1, &g1) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    /* buffers account bytes against the total */
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_BUF_DATA, 3000, 8, 4096, &i1, &g1) == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_BUF_DATA, 2000, 8, 4096, &i2, &g2) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_BUF_DATA, 5000, 8, 4096, &i2, &g2) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_BUF_CODE, 1096, 8, 4096, &i2, &g2) == OMEGA_GPU_PLAT_OK);
    EXPECT(t.nlive == 3 && t.total_bytes == 4096);
    /* checks: kind, index, generation */
    EXPECT(omega_gpu_handles_check(&t, i1, g1, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_handles_check(&t, i1, g1, OMEGA_GPU_HANDLE_BUF_CODE) == OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE);
    EXPECT(omega_gpu_handles_check(&t, OMEGA_GPU_HANDLE_SLOTS, 1, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE);
    EXPECT(omega_gpu_handles_check(&t, i1, 0, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE);
    EXPECT(omega_gpu_handles_check(&t, i1, g1 + 1u, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(omega_gpu_handles_check(&t, 40, 1, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE); /* never opened */
    EXPECT(omega_gpu_handles_check(NULL, i1, g1, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_REFUSED_ARGS);
    /* close: frees bytes, the old handle goes stale, double close refused, slot reuse gets a new generation */
    EXPECT(omega_gpu_handles_close(&t, i1, g1, OMEGA_GPU_HANDLE_BUF_CODE) == OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE);
    EXPECT(omega_gpu_handles_close(&t, i1, g1, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_OK);
    EXPECT(t.nlive == 2 && t.total_bytes == 1096);
    EXPECT(omega_gpu_handles_check(&t, i1, g1, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(omega_gpu_handles_close(&t, i1, g1, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_BUF_DATA, 3000, 8, 4096, &i, &g) == OMEGA_GPU_PLAT_OK);
    EXPECT(i == i1 && g == g1 + 1u);
    EXPECT(omega_gpu_handles_check(&t, i1, g1, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE);
    EXPECT(omega_gpu_handles_check(&t, i, g, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_OK);
    /* per-kind slot limit counts only live handles of that kind */
    for (n = 0; n < 2; n++) EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_QUEUE, 0, 2, 0, &i2, &g2) == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_QUEUE, 0, 2, 0, &i2, &g2) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    EXPECT(omega_gpu_handles_close(&t, i2, g2, OMEGA_GPU_HANDLE_QUEUE) == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_QUEUE, 0, 2, 0, &i2, &g2) == OMEGA_GPU_PLAT_OK);
    /* table capacity */
    omega_gpu_handles_init(&t);
    for (n = 0; n < OMEGA_GPU_HANDLE_SLOTS; n++)
        EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_BUF_DATA, 1, OMEGA_GPU_HANDLE_SLOTS + 1u, OMEGA_GPU_HANDLE_SLOTS + 1u, &i, &g) == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_BUF_DATA, 1, OMEGA_GPU_HANDLE_SLOTS + 1u, OMEGA_GPU_HANDLE_SLOTS + 1u, &i, &g) == OMEGA_GPU_PLAT_REFUSED_LIMIT);
    /* a wrapped generation retires the slot instead of reissuing generation 0 */
    omega_gpu_handles_init(&t);
    t.generation[0] = 0xFFFFFFFFu;
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_BUF_DATA, 1, 8, 8, &i, &g) == OMEGA_GPU_PLAT_OK);
    EXPECT(i == 0 && g == 0xFFFFFFFFu);
    EXPECT(omega_gpu_handles_close(&t, 0, g, OMEGA_GPU_HANDLE_BUF_DATA) == OMEGA_GPU_PLAT_OK);
    EXPECT(omega_gpu_handles_open(&t, OMEGA_GPU_HANDLE_BUF_DATA, 1, 8, 8, &i, &g) == OMEGA_GPU_PLAT_OK);
    EXPECT(i == 1 && g == 1u);
    case_end("HANDLES_TYPED_GENERATIONED_LIMITED");
}

static void case_sizes(void)
{
    /* No Linux fd, no raw pointer and no physical address can be smuggled
     * through a handle: it is two 32-bit fields, and the address types are
     * distinct structs that do not convert into each other. */
    case_begin();
    EXPECT(sizeof(OmegaGpuDeviceHandle) == 8u);
    EXPECT(sizeof(OmegaGpuBufferHandle) == 8u);
    EXPECT(sizeof(OmegaGpuQueueHandle) == 8u);
    EXPECT(sizeof(OmegaGpuVa) == 8u && sizeof(OmegaGpuBusAddr) == 8u);
    EXPECT(sizeof(OmegaCpuView) == sizeof(void *) + sizeof(size_t));
    EXPECT((OMEGA_GPU_SVC_REQUIRED_V1 | OMEGA_GPU_SVC_IRQ) == 0xFFu); /* the v1 bits; 1.1 adds 8..11 */
    case_end("HANDLE_AND_ADDRESS_SHAPES");
}

int main(void)
{
    case_names();
    case_negotiate();
    case_check_table();
    case_session_services();
    case_buffers();
    case_handles();
    case_sizes();
    printf("SUMMARY passed=%d failed=%d\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
