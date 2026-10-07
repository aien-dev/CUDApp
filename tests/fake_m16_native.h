/*
 * Simulated GB10 driver for the host test of the Blackwell GPU engine backend
 * (tests/test_omega_blackwell_engine.c). NOT_RUN: written in cut A3b2 and never
 * compiled or executed by its author.
 *
 * It replaces seven symbols of the real driver stack, so the REAL backend
 * (src/omega_blackwell_engine.c) and the REAL vector wrapper
 * (src/omega_blackwell_submit.c) run unchanged on a machine with no GPU:
 *   m16_native_open, m16_native_create_channel, m16_native_submit_methods,
 *   m16_native_wait_marker, m16_native_close, nvrm_alloc, nvrm_free.
 * The signatures come from physics/m16/m16_native.h and physics/nvrm/nvrm.h at
 * the physics pin; the fake file includes those headers, so a mismatch is a
 * compile error.
 *
 * What it models (every point is a MODEL, UNVERIFIED on the chip, confidence
 * about 70 percent that it matches the real driver):
 *   - "device" memory is host memory, GPU address equals CPU address, dirty
 *     (0x5a fill) so a missing zeroing step shows up;
 *   - the submitted pushbuffer is parsed and executed in order: DMA uploads,
 *     the two inline QMDs, the semaphore initial value, the release packets and
 *     the L2 flush memory operation (method 0x28, operation 0x10 << 27);
 *   - the vector kernel is a simple loop (c[i] = a[i] + b[i] for the threads the
 *     QMD asks for) and only runs if the program bytes in device memory are the
 *     bytes the test expects;
 *   - the second marker release is only written if an L2 flush operation
 *     appears between the two releases (modelling choice, see the doc);
 *   - m16_native_wait_marker is the real ">=" serial compare with no loop; a
 *     timeout advances the fake clock by the timeout so waited_ms is exact;
 *   - nvrm_free of a zeroed NvrmMem is a no-op that counts nothing, a real free
 *     zeroes the NvrmMem and clobbers errno (like a munmap or ioctl might).
 */
#ifndef FAKE_M16_NATIVE_H
#define FAKE_M16_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#define FK_TEXT "fake driver says no"
#define FK_EV_MAX 512
#define FK_SNAP_WORDS 256
#define FK_PB_MAX 2048

/* Event kinds in the log (fk.ev). */
enum {
    FK_EV_OPEN = 1,
    FK_EV_CHANNEL,
    FK_EV_ALLOC,        /* a = bytes asked */
    FK_EV_FREE,         /* a = handle */
    FK_EV_SUBMIT,       /* a = word count */
    FK_EV_WAIT,         /* a = FK_W_*, b = expected value, c = timeout ms */
    FK_EV_CLOSE,        /* a = live allocations at the moment of close */
    FK_EV_PB_CBANK,     /* a = destination, b = words (224) */
    FK_EV_PB_ARGS,      /* a = destination, b = words (7) */
    FK_EV_PB_SEM_INIT,  /* a = destination, b = value */
    FK_EV_PB_QMD,       /* a = QMD address, b = 1 if it is the kernel QMD */
    FK_EV_PB_RELEASE,   /* a = address, b = payload, c = flags word */
    FK_EV_PB_FLUSH,     /* a = operation word (last word of the packet) */
    FK_EV_KERNEL        /* a = threads, b = element count */
};

/* Which coherent word a wait looked at. */
enum { FK_W_UNKNOWN = 0, FK_W_MARKER1, FK_W_MARKER2, FK_W_SEM, FK_W_COUNT };

enum { FK_ALLOC_FAIL = 0, FK_ALLOC_NO_CPU, FK_ALLOC_SHORT, FK_ALLOC_MISALIGNED };
enum { FK_SYNC_NORMAL = 0, FK_SYNC_NEVER, FK_SYNC_OVERSHOOT };
/* FK_KERNEL_SKIP: run no kernel at all and only release the completion semaphore (for tests that count
 * driver calls with kernels the fake does not model, such as the matmul and attention launches). */
enum { FK_KERNEL_OK = 0, FK_KERNEL_NO_WRITE, FK_KERNEL_HALF, FK_KERNEL_WRONG, FK_KERNEL_SKIP };

#define FK_SIZE_LOG 8192
#define FK_LIVE_MAX 128 /* live driver buffers the fake can track at once */

typedef struct {
    int kind;
    uint64_t a;
    uint64_t b;
    uint64_t c;
} FakeEvent;

typedef struct {
    /* ---- faults and behaviour, set by the test after fake_reset ---- */
    int open_fail;
    int channel_fail;
    int submit_fail;
    int close_fail;
    uint64_t alloc_max;   /* largest nvrm_alloc the fake grants, 0 = the default 1 MiB */
    int alloc_fail_nth;   /* 1-based nvrm_alloc call that misbehaves, 0 = none */
    int alloc_mode;       /* FK_ALLOC_* for that call */
    int free_fail_nth;    /* 1-based real nvrm_free call that fails, 0 = none */
    int marker_mode;      /* FK_SYNC_*: first marker release */
    int marker2_mode;     /* FK_SYNC_*: second marker release */
    int sem_mode;         /* FK_SYNC_*: kernel completion semaphore */
    int kernel_mode;      /* FK_KERNEL_* */
    const void *prog_expect; /* program bytes the device memory must hold, NULL = do not check */
    size_t prog_len;

    /* ---- observations ---- */
    uint64_t clock_ms;    /* the fake millisecond clock (fake_now_ms) */
    int open_calls;
    int channel_calls;
    int alloc_calls;
    int uncached_calls;   /* allocations made through nvrm_alloc_gpu_uncached (also counted in alloc_calls) */
    int free_calls;       /* real frees only (a zeroed NvrmMem is not counted) */
    int close_calls;
    int submit_calls;
    int wait_calls;
    int double_free;      /* frees of a handle that was not live */
    int live;             /* live backend allocations right now (not the channel's own 4 KiB buffer) */
    int live_at_close;    /* live allocations when m16_native_close was called */
    int device_open;
    int channel_up;
    uint64_t pb_mem_size; /* ctx->pb_mem.size when the pushbuffer was submitted */
    int pb_parse_error;   /* the pushbuffer had a packet the fake could not parse */
    uint32_t pb[FK_PB_MAX];
    size_t pb_len;

    uint64_t marker1_va;  /* first release address, from the pushbuffer */
    uint64_t marker2_va;  /* second release address, 0 with NO_C3 */
    uint64_t sem_va;      /* destination of the semaphore initial value DMA */
    uint32_t sem_init_value;
    uint32_t pre_marker1; /* values the GPU would see at the doorbell */
    uint32_t pre_marker2;
    uint32_t pre_sem;
    int pre_valid;        /* the three pre_* values were captured */
    int kernel_ran;
    int prog_matched;
    uint64_t kernel_threads;
    uint64_t kernel_n;
    uint32_t out_before[FK_SNAP_WORDS]; /* device output words just before the kernel ran */
    size_t snap_n;
    int poison_collisions; /* snapshot words equal to the correct sum (a bad poison word) */
    uint32_t release_flags[2];
    uint64_t alloc_va[16]; /* GPU address of the nth nvrm_alloc (1-based nth stored at [nth - 1]) */
    uint64_t arg_a;        /* kernel argument addresses the device saw in the constant bank */
    uint64_t arg_b;
    uint64_t arg_c;
    uint64_t prog_va_seen; /* program address from QMD1 */
    uint64_t cbank_va_seen;/* constant bank address from QMD1 */

    int wait_n[FK_W_COUNT];
    uint64_t wait_ms[FK_W_COUNT];
    uint32_t wait_expected[FK_W_COUNT];

    uint64_t alloc_size_log[FK_SIZE_LOG]; /* size asked by the nth granted nvrm_alloc (0-based), first FK_SIZE_LOG only */
    int alloc_size_n;
    uint64_t free_size_log[FK_SIZE_LOG];  /* size of the nth real nvrm_free */
    int free_size_n;

    FakeEvent ev[FK_EV_MAX];
    int nev;
    int ev_overflow;
} FakeDriver;

extern FakeDriver fk;

/* Frees every host allocation the fake still holds (the test's own teardown, not a driver call). */
void fake_release_all(void);
/* fake_release_all, then zero the whole state. Call at the start of every case. */
void fake_reset(void);
/* Millisecond clock for omega_gpu_engine_test_set_clock. */
uint64_t fake_now_ms(void);
/* Number of events of this kind in the log. */
int fake_count(int kind);
/* Index of the nth (1-based) event of this kind, or -1. */
int fake_nth(int kind, int nth);

#endif
