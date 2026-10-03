/*
 * Simulated GB10 driver. See tests/fake_m16_native.h for what it models.
 * NOT_RUN: written in cut A3b2, never compiled or executed by its author.
 *
 * Linked INSTEAD of physics/m16/m16_native.c and physics/nvrm/nvrm.c. The
 * physics headers are still included, so the types (Nvrm, NvrmMem,
 * M16NativeContext) and the prototypes are the real ones.
 */
#include "fake_m16_native.h"
#include "m16_native.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FK_LIVE_MAX 64

FakeDriver fk;

typedef struct {
    uint32_t handle;
    void *ptr;
} LiveAlloc;

static LiveAlloc g_live[FK_LIVE_MAX];
static uint32_t g_next_handle;
static void *g_chan_pb; /* the channel's own 4 KiB pushbuffer, not counted in fk.live */

/* ---- test-side helpers ---------------------------------------------------- */

void fake_release_all(void)
{
    size_t i;
    for (i = 0; i < FK_LIVE_MAX; i++) {
        if (g_live[i].handle != 0) {
            free(g_live[i].ptr);
            g_live[i].handle = 0;
            g_live[i].ptr = NULL;
        }
    }
    fk.live = 0;
    free(g_chan_pb);
    g_chan_pb = NULL;
}

void fake_reset(void)
{
    fake_release_all();
    memset(&fk, 0, sizeof fk);
    g_next_handle = 0;
}

uint64_t fake_now_ms(void)
{
    return fk.clock_ms;
}

int fake_count(int kind)
{
    int i;
    int n = 0;
    for (i = 0; i < fk.nev; i++)
        if (fk.ev[i].kind == kind) n++;
    return n;
}

int fake_nth(int kind, int nth)
{
    int i;
    int n = 0;
    for (i = 0; i < fk.nev; i++) {
        if (fk.ev[i].kind != kind) continue;
        n++;
        if (n == nth) return i;
    }
    return -1;
}

/* ---- internal ------------------------------------------------------------- */

static void logev(int kind, uint64_t a, uint64_t b, uint64_t c)
{
    if (fk.nev >= FK_EV_MAX) {
        fk.ev_overflow = 1;
        return;
    }
    fk.ev[fk.nev].kind = kind;
    fk.ev[fk.nev].a = a;
    fk.ev[fk.nev].b = b;
    fk.ev[fk.nev].c = c;
    fk.nev++;
}

/* A failed driver call: errno and the driver's own text, like the real nvrm does. */
static int inject(Nvrm *rm, int rc)
{
    errno = EIO;
    if (rm) (void)snprintf(rm->err, sizeof rm->err, "%s", FK_TEXT);
    return rc;
}

static void live_add(uint32_t handle, void *ptr)
{
    size_t i;
    for (i = 0; i < FK_LIVE_MAX; i++) {
        if (g_live[i].handle == 0) {
            g_live[i].handle = handle;
            g_live[i].ptr = ptr;
            fk.live++;
            return;
        }
    }
    free(ptr); /* the table is full: more live buffers than any test needs */
}

/* Returns 1 if the handle was live and is now released, else 0. */
static int live_release(uint32_t handle)
{
    size_t i;
    for (i = 0; i < FK_LIVE_MAX; i++) {
        if (g_live[i].handle == handle) {
            free(g_live[i].ptr);
            g_live[i].handle = 0;
            g_live[i].ptr = NULL;
            fk.live--;
            return 1;
        }
    }
    return 0;
}

static volatile uint32_t *word_at(uint64_t va)
{
    return (volatile uint32_t *)(uintptr_t)va;
}

/* ---- the simulated GPU: pushbuffer parser and kernel ---------------------- */

/* The vector kernel: QMD1 words give the program address (32, 33), the thread
 * count (34 and 39), the constant bank (42, 43) and the completion semaphore
 * (15, 16, 17), as written by src/omega_blackwell_qmd.c. The arguments are read
 * from constant bank + 0x380: a, b, c addresses (lo, hi) and the element count. */
static void run_kernel(const uint32_t *q)
{
    uint64_t prog = ((uint64_t)(q[33] & 0x1fffffu) << 36) | ((uint64_t)q[32] << 4);
    uint64_t cb = ((uint64_t)(q[43] & 0x7ffffu) << 38) | ((uint64_t)q[42] << 6);
    uint64_t threads = (uint64_t)(q[34] & 0xffffu) * (uint64_t)q[39];
    const uint32_t *args = (const uint32_t *)(uintptr_t)(cb + 0x380u);
    uint32_t *a = (uint32_t *)(uintptr_t)(((uint64_t)args[1] << 32) | args[0]);
    uint32_t *b = (uint32_t *)(uintptr_t)(((uint64_t)args[3] << 32) | args[2]);
    uint32_t *c = (uint32_t *)(uintptr_t)(((uint64_t)args[5] << 32) | args[4]);
    uint32_t n = args[6];
    uint64_t sem = ((uint64_t)q[16] << 32) | q[15];
    uint64_t count = n < threads ? n : threads;
    uint32_t i;
    int ok = 1;

    fk.kernel_ran++;
    fk.kernel_threads = threads;
    fk.arg_a = (uint64_t)(uintptr_t)a;
    fk.arg_b = (uint64_t)(uintptr_t)b;
    fk.arg_c = (uint64_t)(uintptr_t)c;
    fk.prog_va_seen = prog;
    fk.cbank_va_seen = cb;
    fk.kernel_n = n;
    if (fk.prog_expect != NULL && fk.prog_len != 0)
        ok = memcmp((const void *)(uintptr_t)prog, fk.prog_expect, fk.prog_len) == 0;
    fk.prog_matched = ok;
    logev(FK_EV_KERNEL, threads, n, 0);

    /* What the device output holds just before the kernel runs, and whether the
     * poison words collide with the correct sums. */
    fk.snap_n = n < FK_SNAP_WORDS ? n : FK_SNAP_WORDS;
    for (i = 0; i < fk.snap_n; i++) {
        fk.out_before[i] = c[i];
        if (c[i] == (uint32_t)(a[i] + b[i])) fk.poison_collisions++;
    }

    if (ok && fk.kernel_mode != FK_KERNEL_NO_WRITE) {
        if (fk.kernel_mode == FK_KERNEL_HALF) count /= 2u;
        for (i = 0; i < count; i++) {
            c[i] = (uint32_t)(a[i] + b[i]);
            if (fk.kernel_mode == FK_KERNEL_WRONG) c[i] += 1u;
        }
    }

    /* Completion semaphore (RELEASE_SEMAPHORE0 of the QMD): value q[17]. */
    if (fk.sem_mode != FK_SYNC_NEVER)
        *word_at(sem) = q[17] + (fk.sem_mode == FK_SYNC_OVERSHOOT ? 1u : 0u);
}

/* Walks the submitted pushbuffer. exec == 0 only collects addresses (so the
 * state at the doorbell can be captured first); exec == 1 runs it. Every packet
 * is a header (type 2 or 6 in bits 31:28, count in 28:16, subchannel in 15:13,
 * method in 11:0 times 4) followed by count data words. */
static void walk(const uint32_t *w, size_t n, int exec)
{
    size_t i = 0;
    uint64_t dst = 0;
    uint32_t line_bytes = 0;
    uint32_t lines = 0;
    int nrel = 0;
    int flushed = 0;

    while (i < n) {
        uint32_t h = w[i++];
        uint32_t type = h >> 28;
        uint32_t cnt = (h >> 16) & 0x1fffu;
        uint32_t subc = (h >> 13) & 7u;
        uint32_t mthd = (h & 0xfffu) << 2;
        const uint32_t *d;

        if ((type != 2u && type != 6u) || i + cnt > n) {
            fk.pb_parse_error = 1;
            return;
        }
        d = &w[i];
        i += cnt;

        if (subc == 1u && mthd == 0x0188u && cnt == 2u) {
            dst = ((uint64_t)d[0] << 32) | d[1];
        } else if (subc == 1u && mthd == 0x0180u && cnt == 2u) {
            line_bytes = d[0];
            lines = d[1];
        } else if (subc == 1u && mthd == 0x01b4u) {
            if (dst == 0 || (uint64_t)cnt * 4u != (uint64_t)line_bytes * lines) {
                fk.pb_parse_error = 1;
                continue;
            }
            if (!exec) {
                if (cnt == 1u) {
                    fk.sem_va = dst;
                    fk.sem_init_value = d[0];
                }
            } else {
                memcpy((void *)(uintptr_t)dst, d, (size_t)cnt * 4u);
                if (cnt == 224u) logev(FK_EV_PB_CBANK, dst, cnt, 0);
                else if (cnt == 7u) logev(FK_EV_PB_ARGS, dst, cnt, 0);
                else if (cnt == 1u) logev(FK_EV_PB_SEM_INIT, dst, d[0], 0);
            }
        } else if (subc == 1u && mthd == 0x0318u && cnt == 98u) {
            uint64_t qaddr = ((uint64_t)(d[0] & 0x1ffu) << 40) | ((uint64_t)d[1] << 8);
            const uint32_t *q = d + 2;
            int launch = ((d[0] >> 30) & 1u) != 0;
            if (exec) {
                if (qaddr == 0) {
                    fk.pb_parse_error = 1;
                    continue;
                }
                memcpy((void *)(uintptr_t)qaddr, q, 96u * 4u);
                logev(FK_EV_PB_QMD, qaddr, (launch && (q[9] & 1u) != 0) ? 1u : 0u, 0);
                if (launch && (q[9] & 1u) != 0) run_kernel(q);
            }
        } else if (subc == 0u && mthd == 0x005cu && cnt == 5u) {
            uint64_t addr = ((uint64_t)d[1] << 32) | d[0];
            uint32_t payload = d[2];
            uint32_t flags = d[4];
            nrel++;
            if (!exec) {
                if (nrel == 1) fk.marker1_va = addr;
                else if (nrel == 2) fk.marker2_va = addr;
            } else {
                int mode = nrel == 1 ? fk.marker_mode : fk.marker2_mode;
                int writes = (flags & 1u) != 0;
                logev(FK_EV_PB_RELEASE, addr, payload, flags);
                if (nrel <= 2) fk.release_flags[nrel - 1] = flags;
                /* Modelling choice: the second release only lands if an L2 flush
                 * memory operation came between the two releases. */
                if (nrel == 2 && !flushed) writes = 0;
                if (mode == FK_SYNC_NEVER) writes = 0;
                if (writes && addr != 0)
                    *word_at(addr) = payload + (mode == FK_SYNC_OVERSHOOT ? 1u : 0u);
                flushed = 0;
            }
        } else if (subc == 0u && mthd == 0x0028u && cnt == 4u) {
            if (exec) {
                logev(FK_EV_PB_FLUSH, d[3], 0, 0);
                if (d[3] == (0x10u << 27)) flushed = 1;
            }
        }
        /* every other packet is a setup method: ignored */
    }
}

/* ---- the seven faked symbols ---------------------------------------------- */

int m16_native_open(M16NativeContext *ctx)
{
    if (!ctx) return -1;
    memset(ctx, 0, sizeof *ctx);
    fk.open_calls++;
    logev(FK_EV_OPEN, 0, 0, 0);
    if (fk.open_fail) return inject(&ctx->rm, -1);
    fk.device_open = 1;
    return 0;
}

int m16_native_create_channel(M16NativeContext *ctx)
{
    void *p = NULL;
    if (!ctx) return -1;
    fk.channel_calls++;
    logev(FK_EV_CHANNEL, 0, 0, 0);
    if (fk.channel_fail) return inject(&ctx->rm, -1);
    if (posix_memalign(&p, 4096, 0x1000) != 0) return -1;
    memset(p, 0, 0x1000);
    ctx->pb_mem.handle = ++g_next_handle;
    ctx->pb_mem.size = 0x1000;
    ctx->pb_mem.cpu = p;
    ctx->pb_mem.va = (uint64_t)(uintptr_t)p;
    g_chan_pb = p;
    fk.channel_up = 1;
    return 0;
}

int nvrm_alloc(Nvrm *rm, uint64_t size, NvrmMem *out)
{
    void *p = NULL;
    int mode = -1;
    fk.alloc_calls++;
    logev(FK_EV_ALLOC, size, 0, 0);
    if (!out || size == 0 || size > 0x100000u) return -1;
    if (fk.alloc_fail_nth != 0 && fk.alloc_fail_nth == fk.alloc_calls) {
        mode = fk.alloc_mode;
        if (mode == FK_ALLOC_FAIL) return inject(rm, -1);
    }
    if (posix_memalign(&p, 4096, (size_t)size) != 0) return -1;
    memset(p, 0x5a, (size_t)size); /* dirty memory */
    out->handle = ++g_next_handle;
    out->size = size;
    out->cpu = p;
    out->va = (uint64_t)(uintptr_t)p;
    live_add(out->handle, p);
    if (fk.alloc_calls >= 1 && fk.alloc_calls <= 16) fk.alloc_va[fk.alloc_calls - 1] = out->va;
    if (mode == FK_ALLOC_NO_CPU) out->cpu = NULL;
    else if (mode == FK_ALLOC_SHORT) out->size = size / 2u;
    else if (mode == FK_ALLOC_MISALIGNED) out->va += 0x10u;
    return 0;
}

/* Same as nvrm_alloc (the fake has no GPU cache); only counts the uncached request. */
int nvrm_alloc_gpu_uncached(Nvrm *rm, uint64_t size, NvrmMem *out)
{
    fk.uncached_calls++;
    return nvrm_alloc(rm, size, out);
}

int nvrm_free(Nvrm *rm, NvrmMem *m)
{
    if (!m || m->handle == 0) return 0; /* a zeroed NvrmMem is a no-op (nvrm.h) */
    fk.free_calls++;
    logev(FK_EV_FREE, m->handle, 0, 0);
    if (fk.free_fail_nth != 0 && fk.free_fail_nth == fk.free_calls) return inject(rm, -1);
    if (!live_release(m->handle)) {
        fk.double_free++;
        errno = EINVAL;
        return -1;
    }
    memset(m, 0, sizeof *m);
    errno = EBADF; /* a real free runs munmap and ioctl calls, which may change errno */
    return 0;
}

int m16_native_submit_methods(M16NativeContext *ctx, const uint32_t *methods, size_t count)
{
    if (!ctx || !methods || count == 0) return -1;
    fk.submit_calls++;
    logev(FK_EV_SUBMIT, count, 0, 0);
    if (!fk.channel_up || fk.submit_fail) return inject(&ctx->rm, -1);
    /* Like m16_native_enqueue_methods: the words must fit the ring buffer. */
    if (count > ctx->pb_mem.size / sizeof(uint32_t) || count > FK_PB_MAX) return inject(&ctx->rm, -1);
    fk.pb_mem_size = ctx->pb_mem.size;
    memcpy(fk.pb, methods, count * sizeof(uint32_t));
    fk.pb_len = count;

    walk(fk.pb, count, 0);
    if (fk.marker1_va != 0 && fk.sem_va != 0) {
        fk.pre_marker1 = *word_at(fk.marker1_va);
        fk.pre_marker2 = fk.marker2_va != 0 ? *word_at(fk.marker2_va) : 0u;
        fk.pre_sem = *word_at(fk.sem_va);
        fk.pre_valid = 1;
    }
    walk(fk.pb, count, 1);
    return 0;
}

static int which_word(volatile uint32_t *p)
{
    uint64_t a = (uint64_t)(uintptr_t)p;
    if (p == NULL) return FK_W_UNKNOWN;
    if (fk.marker1_va != 0 && a == fk.marker1_va) return FK_W_MARKER1;
    if (fk.marker2_va != 0 && a == fk.marker2_va) return FK_W_MARKER2;
    if (fk.sem_va != 0 && a == fk.sem_va) return FK_W_SEM;
    return FK_W_UNKNOWN;
}

/* The real wait is a ">=" serial compare in a loop with a deadline. The fake
 * checks once and, if it is not satisfied, "waits" the whole timeout by
 * advancing the fake clock. */
int m16_native_wait_marker(volatile uint32_t *marker, uint32_t expected, uint64_t timeout_ms)
{
    int w = which_word(marker);
    fk.wait_calls++;
    fk.wait_n[w]++;
    fk.wait_ms[w] = timeout_ms;
    fk.wait_expected[w] = expected;
    logev(FK_EV_WAIT, (uint64_t)w, expected, timeout_ms);
    if (!marker) return -1;
    if ((int32_t)(*marker - expected) >= 0) return 0;
    fk.clock_ms += timeout_ms;
    return -1;
}

int m16_native_close(M16NativeContext *ctx)
{
    int rc = 0;
    if (!ctx) return -1;
    fk.close_calls++;
    fk.live_at_close = fk.live;
    logev(FK_EV_CLOSE, (uint64_t)fk.live, 0, 0);
    /* Like nvrm_close: whatever is still allocated is released here. */
    fake_release_all();
    fk.device_open = 0;
    if (fk.close_fail) rc = inject(&ctx->rm, -1);
    return rc;
}
