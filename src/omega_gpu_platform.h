/* omega_gpu_platform.h -- native platform contract for the Omega GPU engine.
 *
 * Native v1 P07 cut A (aienos#286 cut A, aien-architecture#190). This header
 * names what a GPU backend may ask of the platform it runs on, in a form that
 * works the same under Linux (today's nvrm/m16 backend) and under AIENOS
 * (the native backend of later cuts). It is pure C with no Linux header and
 * no libc call; it builds in the freestanding link target
 * (`make gpu-engine-freestanding`).
 *
 * Rules (from aienos#286 cut A):
 *  - Resources are typed handles (index + generation), never a Linux fd, a
 *    raw pointer or an unchecked physical address. A handle is valid only
 *    while its slot generation matches, so a closed handle is refused.
 *  - Every byte range carries an explicit length and explicit rights.
 *  - CPU views, GPU virtual addresses and bus (SMMU input) addresses are
 *    three distinct types. There is no type for a raw physical address on
 *    purpose: the native backend resolves DMA through the SMMU (cut B).
 *  - Buffers, code and queues have separate kinds and lifetimes.
 *  - A backend negotiates an ABI version and a service set up front; what
 *    the platform does not provide is refused, never emulated. There is no
 *    silent fallback to Linux, to the CPU, or to a remote device.
 *  - Polling a word is not interrupt support: OMEGA_GPU_SVC_IRQ is claimed
 *    only by a platform whose irq_wait blocks on a real device interrupt.
 *  - Anything the CPU polls for a device write (completion markers,
 *    semaphores) is OMEGA_GPU_RIGHT_HOST_POLLED and must be GPU-uncached
 *    (nvidia-open 580.173.02 nvos.h: GPU-cacheable system memory "will not
 *    be coherent with direct CPU mappings"; the 2026-10-02 E1 stall).
 *
 * What this header does NOT claim: no device has been opened through it, no
 * GB10 register has been touched, and the Linux backend still owns every
 * driver responsibility (docs/numeric/OMEGA_GPU_NATIVE_BOUNDARY.md). Hardware
 * evidence for device ownership is NOT_RUN until cut B runs in an attended
 * window.
 */
#ifndef OMEGA_GPU_PLATFORM_H
#define OMEGA_GPU_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

#define OMEGA_GPU_PLATFORM_ABI_MAJOR 1u
/* minor: additive change (a service bit or a caps field appended); negotiate refuses a provided minor older than the
 * required one. 1.1 (P07 cut C session part): DEVICE, QUEUE, CLOCK_NS, HOST_MEM; COMPLETION is poll_word.
 * 1.2 (omega #457/#459): caps.submit_prefix_words. */
#define OMEGA_GPU_PLATFORM_ABI_MINOR 2u

/* ---- typed handles ------------------------------------------------------ */

/* Generation 0 is never issued, so a zeroed handle is "none". */
typedef struct { uint32_t index; uint32_t generation; } OmegaGpuDeviceHandle;
typedef struct { uint32_t index; uint32_t generation; } OmegaGpuBufferHandle;
typedef struct { uint32_t index; uint32_t generation; } OmegaGpuQueueHandle;

/* Handle kinds (also the buffer kinds below share the BUF_* space). */
typedef enum {
    OMEGA_GPU_HANDLE_DEVICE = 1,
    OMEGA_GPU_HANDLE_BUF_DATA = 2,  /* inputs, outputs, scratch */
    OMEGA_GPU_HANDLE_BUF_CODE = 3,  /* kernel code: CPU writes once, GPU executes, GPU never writes */
    OMEGA_GPU_HANDLE_BUF_QUEUE = 4, /* pushbuffer / submission ring */
    OMEGA_GPU_HANDLE_QUEUE = 5,     /* a channel or submission queue */
    OMEGA_GPU_HANDLE_KIND_COUNT
} OmegaGpuHandleKind;

/* ---- address kinds ------------------------------------------------------ */

typedef struct { void *ptr; size_t len; } OmegaCpuView;   /* CPU mapping with its length */
typedef struct { uint64_t va; } OmegaGpuVa;                /* GPU virtual address (device page tables) */
typedef struct { uint64_t iova; } OmegaGpuBusAddr;         /* what the device emits on the fabric (SMMU input) */

/* ---- rights and buffer descriptors -------------------------------------- */

enum {
    OMEGA_GPU_RIGHT_CPU_READ = 1u << 0,
    OMEGA_GPU_RIGHT_CPU_WRITE = 1u << 1,
    OMEGA_GPU_RIGHT_GPU_READ = 1u << 2,
    OMEGA_GPU_RIGHT_GPU_WRITE = 1u << 3,
    OMEGA_GPU_RIGHT_GPU_EXEC = 1u << 4,    /* code buffers only */
    OMEGA_GPU_RIGHT_HOST_POLLED = 1u << 5, /* CPU polls a device write here: must be GPU-uncached */
    OMEGA_GPU_RIGHT_ALL = (1u << 6) - 1u
};

typedef enum { OMEGA_GPU_CACHE_GPU_UNCACHED = 0, OMEGA_GPU_CACHE_GPU_CACHED = 1 } OmegaGpuCachePolicy;

typedef struct {
    uint32_t kind;       /* OMEGA_GPU_HANDLE_BUF_DATA, _BUF_CODE or _BUF_QUEUE */
    uint32_t rights;     /* OMEGA_GPU_RIGHT_* mask, never 0 */
    uint32_t gpu_cache;  /* OmegaGpuCachePolicy */
    uint32_t align;      /* bytes, a power of two, 0 means the platform minimum */
    uint64_t bytes;      /* explicit length, never 0 */
} OmegaGpuBufferDesc;

/* ---- services and capabilities ------------------------------------------ */

enum {
    OMEGA_GPU_SVC_TIME = 1u << 0,       /* monotonic millisecond clock */
    OMEGA_GPU_SVC_WAIT = 1u << 1,       /* bounded sleep or yield */
    OMEGA_GPU_SVC_LOCK = 1u << 2,       /* mutual exclusion for queues and tables */
    OMEGA_GPU_SVC_MMIO = 1u << 3,       /* bounded register access by device handle and region */
    OMEGA_GPU_SVC_DMA = 1u << 4,        /* device-visible memory with CPU view, GPU VA and bus address */
    OMEGA_GPU_SVC_SYNC = 1u << 5,       /* explicit CPU<->GPU visibility (cache maintenance, flush) */
    OMEGA_GPU_SVC_COMPLETION = 1u << 6, /* POLL an uncached word (poll_word); polling is never an interrupt */
    OMEGA_GPU_SVC_IRQ = 1u << 7,        /* real device interrupt wait; polling never counts */
    OMEGA_GPU_SVC_REQUIRED_V1 = OMEGA_GPU_SVC_TIME | OMEGA_GPU_SVC_WAIT | OMEGA_GPU_SVC_LOCK |
                                OMEGA_GPU_SVC_MMIO | OMEGA_GPU_SVC_DMA | OMEGA_GPU_SVC_SYNC |
                                OMEGA_GPU_SVC_COMPLETION,
    /* ABI 1.1: what the persistent GPU session (src/omega_gpu_session.c) needs on top of v1 */
    OMEGA_GPU_SVC_DEVICE = 1u << 8,     /* open / close one device context, driver text of the last failure */
    OMEGA_GPU_SVC_QUEUE = 1u << 9,      /* submission queue over a BUF_QUEUE ring: open, submit, retire, close */
    OMEGA_GPU_SVC_CLOCK_NS = 1u << 10,  /* monotonic nanosecond clock (launch timing) */
    OMEGA_GPU_SVC_HOST_MEM = 1u << 11,  /* CPU-only memory for tables and staging (never device-visible) */
    OMEGA_GPU_SVC_REQUIRED_SESSION = OMEGA_GPU_SVC_TIME | OMEGA_GPU_SVC_WAIT | OMEGA_GPU_SVC_LOCK |
                                     OMEGA_GPU_SVC_DMA | OMEGA_GPU_SVC_SYNC | OMEGA_GPU_SVC_COMPLETION |
                                     OMEGA_GPU_SVC_DEVICE | OMEGA_GPU_SVC_QUEUE | OMEGA_GPU_SVC_CLOCK_NS |
                                     OMEGA_GPU_SVC_HOST_MEM,
    OMEGA_GPU_SVC_ALL = (1u << 12) - 1u
};

typedef struct {
    uint32_t abi_major;
    uint32_t abi_minor;
    uint32_t services;          /* OMEGA_GPU_SVC_* provided (or required) */
    uint32_t max_devices;
    uint32_t max_buffers;       /* live buffer handles */
    uint32_t max_queues;
    uint32_t max_inflight;      /* jobs in flight per queue (concurrency limit) */
    uint64_t max_buffer_bytes;  /* one buffer */
    uint64_t max_total_bytes;   /* all live buffers */
    uint32_t submit_prefix_words; /* ABI 1.2: most words queue_submit may put ahead of the caller's words in the ring (the Linux
                                   * reference: the omega #457 L2 invalidate); a caller filling the ring budgets
                                   * ring words minus this. 0 = never. Not negotiated. */
} OmegaGpuPlatformCaps;

/* ---- refusals ----------------------------------------------------------- */

typedef enum {
    OMEGA_GPU_PLAT_OK = 0,
    OMEGA_GPU_PLAT_REFUSED_ARGS,            /* NULL or malformed argument */
    OMEGA_GPU_PLAT_REFUSED_ABI,             /* major differs, or provided minor older than required */
    OMEGA_GPU_PLAT_REFUSED_MISSING_SERVICE, /* a required service is not provided */
    OMEGA_GPU_PLAT_REFUSED_NULL_SERVICE,    /* a claimed service has no function */
    OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE,      /* index out of range or kind mismatch */
    OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE,    /* generation mismatch: closed or never opened */
    OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH,      /* zero length */
    OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS,      /* empty, unknown or contradictory rights */
    OMEGA_GPU_PLAT_REFUSED_BAD_KIND,        /* unknown kind */
    OMEGA_GPU_PLAT_REFUSED_CACHE_POLICY,    /* host-polled buffer asked to be GPU-cached */
    OMEGA_GPU_PLAT_REFUSED_LIMIT,           /* a capability limit would be exceeded */
    OMEGA_GPU_PLAT_REFUSED_BUSY,            /* concurrency limit reached */
    OMEGA_GPU_PLAT_REFUSED_DEVICE,          /* 1.1: the device or its driver refused (diag_text says why) */
    OMEGA_GPU_PLAT_REFUSAL_COUNT
} OmegaGpuPlatformRefusal;

const char *omega_gpu_platform_refusal_name(int refusal); /* static strings, never NULL */

/* poll_word result when the word did not reach the value in time (not a refusal: nothing was wrong with the call). */
#define OMEGA_GPU_POLL_TIMED_OUT 100

/* ---- the platform service table ----------------------------------------- */

typedef struct {
    void *ctx;
    /* TIME, WAIT, LOCK */
    uint64_t (*time_now_ms)(void *ctx);
    int (*wait_ms)(void *ctx, uint64_t ms);
    int (*lock)(void *ctx, uint32_t lock_id);
    int (*unlock)(void *ctx, uint32_t lock_id);
    /* MMIO: bounded by (device, region); the platform refuses an offset past the region. */
    int (*mmio_read32)(void *ctx, OmegaGpuDeviceHandle dev, uint32_t region, uint64_t offset, uint32_t *out);
    int (*mmio_write32)(void *ctx, OmegaGpuDeviceHandle dev, uint32_t region, uint64_t offset, uint32_t value);
    /* DMA: one allocation yields a handle plus the three views; free by handle only. */
    int (*dma_alloc)(void *ctx, OmegaGpuDeviceHandle dev, const OmegaGpuBufferDesc *desc,
                     OmegaGpuBufferHandle *out, OmegaCpuView *cpu, OmegaGpuVa *gpu, OmegaGpuBusAddr *bus);
    int (*dma_free)(void *ctx, OmegaGpuBufferHandle buf);
    /* SYNC: explicit visibility over a byte range of one buffer. */
    int (*sync_cpu_to_gpu)(void *ctx, OmegaGpuBufferHandle buf, uint64_t offset, uint64_t len);
    int (*sync_gpu_to_cpu)(void *ctx, OmegaGpuBufferHandle buf, uint64_t offset, uint64_t len);
    /* COMPLETION, by POLLING (never an interrupt): read one GPU-uncached 32-bit word of a HOST_POLLED buffer until
     * it has reached expected in serial-number order, (int32_t)(word - expected) >= 0 (the m16_native_wait_marker
     * rule), or timeout_ms passes. 0 reached, OMEGA_GPU_POLL_TIMED_OUT not reached, else a refusal. *observed is
     * the last value read. A timeout says nothing about the device: the work may still be running. */
    int (*poll_word)(void *ctx, OmegaGpuBufferHandle buf, uint64_t offset, uint32_t expected,
                     uint64_t timeout_ms, uint32_t *observed);
    /* IRQ: NULL unless OMEGA_GPU_SVC_IRQ is claimed. */
    int (*irq_wait)(void *ctx, OmegaGpuDeviceHandle dev, uint64_t timeout_ms);
    /* ---- ABI 1.1 ---- */
    /* DEVICE: open the device context with this ordinal (0 = the one GB10), close it. Closing makes every buffer and
     * queue handle of the device stale. diag_text: driver text of the most recent failure, "" if none, never NULL. */
    int (*device_open)(void *ctx, uint32_t ordinal, OmegaGpuDeviceHandle *out);
    int (*device_close)(void *ctx, OmegaGpuDeviceHandle dev);
    const char *(*diag_text)(void *ctx);
    /* QUEUE: a submission queue whose ring is a BUF_QUEUE buffer of the same device. queue_submit copies nwords method
     * words into the ring and rings the doorbell (one GPFIFO entry); queue_retire marks every submitted entry consumed
     * and is called only after the caller saw that work complete. A refused queue_submit rang no doorbell: nothing of it is in
     * flight (m16_native_submit_methods fails before nvrm_ring). */
    int (*queue_open)(void *ctx, OmegaGpuDeviceHandle dev, OmegaGpuBufferHandle ring, OmegaGpuQueueHandle *out);
    int (*queue_close)(void *ctx, OmegaGpuQueueHandle q);
    int (*queue_submit)(void *ctx, OmegaGpuQueueHandle q, const uint32_t *words, uint32_t nwords);
    int (*queue_retire)(void *ctx, OmegaGpuQueueHandle q);
    /* CLOCK_NS */
    uint64_t (*time_now_ns)(void *ctx);
    /* HOST_MEM: NULL on refusal; host_free(NULL) is a no-op. */
    void *(*host_alloc)(void *ctx, size_t bytes);
    void (*host_free)(void *ctx, void *p);
} OmegaGpuPlatform;

/* ---- contract checks (pure, freestanding) -------------------------------- */

/* Version and service negotiation. required.services must be a subset of
 * provided.services, required limits must not exceed provided limits. */
int omega_gpu_platform_negotiate(const OmegaGpuPlatformCaps *provided, const OmegaGpuPlatformCaps *required);

/* Every service in caps->services has a function in p; every function
 * outside caps->services is NULL (no undeclared service, no fallback); the
 * v1 required set is present. */
int omega_gpu_platform_check(const OmegaGpuPlatform *p, const OmegaGpuPlatformCaps *caps);
/* The same, requiring the service set `required` (for example OMEGA_GPU_SVC_REQUIRED_SESSION) instead of the v1 set.
 * omega_gpu_platform_check(p, caps) is omega_gpu_platform_check_for(p, caps, OMEGA_GPU_SVC_REQUIRED_V1). */
int omega_gpu_platform_check_for(const OmegaGpuPlatform *p, const OmegaGpuPlatformCaps *caps, uint32_t required);

/* Buffer descriptor rules: kind known, length non-zero and within caps,
 * rights non-empty and consistent with the kind, host-polled means
 * GPU-uncached, align a power of two. */
int omega_gpu_platform_check_buffer(const OmegaGpuBufferDesc *d, const OmegaGpuPlatformCaps *caps);

/* ---- handle table (index + generation) ----------------------------------- */

#define OMEGA_GPU_HANDLE_SLOTS 1024u /* 1.1: a serving session holds one buffer per weight tensor (Qwen3-8B: ~260) */

typedef struct {
    uint32_t generation[OMEGA_GPU_HANDLE_SLOTS]; /* next generation to issue for the slot; starts at 1 */
    uint32_t kind[OMEGA_GPU_HANDLE_SLOTS];
    uint64_t bytes[OMEGA_GPU_HANDLE_SLOTS];
    uint8_t live[OMEGA_GPU_HANDLE_SLOTS];
    uint32_t nlive;
    uint64_t total_bytes;
} OmegaGpuHandleTable;

void omega_gpu_handles_init(OmegaGpuHandleTable *t);
/* Opens a slot of the given kind. limit_slots caps live handles of that kind.
 * bytes is accounted against limit_bytes across all slots; devices and queues
 * pass bytes 0 and are not byte-accounted. */
int omega_gpu_handles_open(OmegaGpuHandleTable *t, uint32_t kind, uint64_t bytes, uint32_t limit_slots,
                           uint64_t limit_bytes, uint32_t *index, uint32_t *generation);
/* OK only for a live slot whose generation and kind match. */
int omega_gpu_handles_check(const OmegaGpuHandleTable *t, uint32_t index, uint32_t generation, uint32_t kind);
/* Closes a live, matching slot and bumps its generation so the old handle is stale. */
int omega_gpu_handles_close(OmegaGpuHandleTable *t, uint32_t index, uint32_t generation, uint32_t kind);

#endif /* OMEGA_GPU_PLATFORM_H */
