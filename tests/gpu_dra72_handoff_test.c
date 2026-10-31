/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* gpu_dra72_handoff_test: CPU -> GPU -> CPU hand-off latency through the GPU host semaphore (DRA-72 G4, P-DRA-4).
 *
 * Question: if a CPU core computes part of a decode step, what does one hand-off back to the GPU queue cost?
 * The CUDA answer would be cuStreamWaitValue32 / cuStreamWriteValue32. This driver has no CUDA: the equivalent is the
 * host-class semaphore the command list already uses for wait_for (omega_gpu_cmdlist.c, clc96f.h SEM_ADDR_LO 0x5c ..
 * SEM_EXECUTE 0x6c; the omega-mv#580 ORDER_SEM words, chip-proven):
 *   ACQUIRE  SEM_EXECUTE = 0 (OPERATION_ACQUIRE, 32-bit payload): the queue stops until word A == v
 *   RELEASE  SEM_EXECUTE = 1 (OPERATION_RELEASE; variant wfi adds RELEASE_WFI bit 20 and a compute WAIT_FOR_IDLE)
 * One submission carries `chunk` iterations {ACQUIRE A == v, RELEASE B = v}. The CPU (core 5 in the chip slice) writes
 * A = v, makes it visible (sync variant), and spins until B == v. One sample = that round trip in CLOCK_MONOTONIC ns.
 * Max in flight is one iteration: the queue cannot run ahead of the CPU, so every sample is a full CPU->GPU->CPU trip.
 *
 * Checks: B never runs ahead of A (B == v - 1 before A = v); negative control: A = 0xdeadbeef (not equal to v) must
 * leave B unchanged for 2 ms, which proves the queue really waits on A; each chunk ends with the session's two-marker
 * tail and is retired only after the second marker is seen. A spin past --timeout-ms latches the session
 * (omega_gpu_session_latch_uncertain) and fails: the queue may still be parked on A.
 *
 * --sim runs the same words through an in-process interpreter thread (no device is opened). It exercises the encoding,
 * the protocol, the negative control and the log format on the host; its latencies are thread hand-offs, not GPU
 * numbers, and are labelled mode=sim.
 *
 * Output (stdout, or --out FILE mode 0600), one record per line, integers only after the tag:
 *   HANDOFF 1 variant=plain|wfi sync=platform|dsb mode=chip|sim iters=N chunk=C warm=W
 *   NEG <ok 0|1> <b_before> <b_after>
 *   W <ns>            first W samples of each chunk (doorbell/fetch warm-up, excluded by the judge)
 *   S <ns>            measured samples
 *   END <ok 0|1> <samples> <chunks>
 * Exit 0 = protocol held for every iteration, 1 = a check failed, 2 = usage. Timing is judged by tools/dra72/judge.om. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "omega_blackwell_submit.h"
#include "omega_gpu_session.h"
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "test_out_file.h"

#define RING_WORDS (0x10000u / 4u) /* the session ring: 64 KB */
#define OFF_A 0x00u
#define OFF_B 0x40u
#define OFF_M1 0x100u
#define OFF_M2 0x110u
#define BUF_BYTES 0x1000u
#define SIM_VA 0x0000004000000000ull

static inline uint32_t bw_mthd(uint32_t subc, uint32_t mthd, uint32_t count) {
    return (2u << 28) | (count << 16) | (subc << 13) | (mthd >> 2);
}
/* Same words as omega_gpu_cmdlist.c / omega_gpu_session.c SETUP_WORDS (compute class setup). */
static const uint32_t SETUP_WORDS[18] = {
    0x20012061, 0x0000cec0, 0x20012092, 0x00000001, 0x200120a8, 0x0000000f, 0x2001255d, 0x00000003,
    0x2001255e, 0x20000000, 0x2001255f, 0x000fffff, 0x20012557, 0x00000003, 0x20012558, 0x22000000,
    0x20012559, 0x00000000,
};
#define MTHD_WAIT_FOR_IDLE 0x0110u
#define SEM_OP_ACQUIRE 0x0u
#define SEM_OP_RELEASE 0x1u
#define SEM_RELEASE_WFI (1u << 20)

static uint64_t now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static inline void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#endif
}
static inline void dsb_sy(void) {
#if defined(__aarch64__)
    __asm__ volatile("dsb sy" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static size_t emit_sem(uint32_t *pb, size_t n, uint64_t va, uint32_t payload, uint32_t exec) {
    pb[n++] = bw_mthd(0, 0x005c, 5);
    pb[n++] = (uint32_t)va; pb[n++] = (uint32_t)(va >> 32); pb[n++] = payload; pb[n++] = 0; pb[n++] = exec;
    return n;
}
/* the omega_gpu_cmdlist.c emit_tail words: marker release, L2 flush mem-op, second marker release */
static size_t emit_tail(uint32_t *pb, size_t n, uint64_t marker_va) {
    n = emit_sem(pb, n, marker_va, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, SEM_OP_RELEASE | SEM_RELEASE_WFI);
    pb[n++] = bw_mthd(0, 0x0028, 4); pb[n++] = 0; pb[n++] = 0; pb[n++] = 0; pb[n++] = (0x10u << 27);
    n = emit_sem(pb, n, marker_va + 0x10, OMEGA_BW_MARKER2_PAYLOAD, SEM_OP_RELEASE | SEM_RELEASE_WFI);
    return n;
}
#define TAIL_WORDS 17u

/* ---- device: the real queue, or the host interpreter ---- */
typedef struct {
    int sim;
    uint8_t *cpu;
    uint64_t va;
    const OmegaGpuPlatform *p;
    OmegaGpuSession *s;
    OmegaGpuMem mem;
    /* sim */
    pthread_t th;
    const uint32_t *words; uint32_t nwords;
    int pending, quit, sim_fault;
} Dev;

static uint32_t *word_at(Dev *d, uint64_t va) {
    if (va < d->va || va + 4 > d->va + BUF_BYTES) return NULL;
    return (uint32_t *)(d->cpu + (va - d->va));
}
/* Interprets incrementing-method headers; host semaphore groups (0x5c..0x6c) act on the buffer, everything else is
 * consumed. ACQUIRE spins (yielding) until the word equals the payload; RELEASE stores it. */
static void *sim_main(void *arg) {
    Dev *d = arg;
    for (;;) {
        while (!__atomic_load_n(&d->pending, __ATOMIC_ACQUIRE)) { if (__atomic_load_n(&d->quit, __ATOMIC_ACQUIRE)) return NULL; sched_yield(); }
        const uint32_t *w = d->words; uint32_t n = d->nwords, i = 0;
        uint32_t sem[5] = { 0 };
        while (i < n && !d->sim_fault) {
            const uint32_t h = w[i++];
            if ((h >> 28) != 2u) { d->sim_fault = 1; break; }
            const uint32_t cnt = (h >> 16) & 0x1fffu, subc = (h >> 13) & 7u, m0 = (h & 0x1fffu) << 2;
            for (uint32_t j = 0; j < cnt && i < n; j++, i++) {
                const uint32_t m = m0 + 4u * j;
                if (subc != 0 || m < 0x5c || m > 0x6c) continue;
                sem[(m - 0x5c) / 4] = w[i];
                if (m != 0x6c) continue;
                uint32_t *t = word_at(d, (uint64_t)sem[0] | ((uint64_t)sem[1] << 32));
                if (!t) { d->sim_fault = 2; break; }
                const uint32_t op = sem[4] & 0x1fu;
                if (op == SEM_OP_ACQUIRE) {
                    const uint64_t end = now_ns() + 5000000000ull;
                    while (__atomic_load_n(t, __ATOMIC_ACQUIRE) != sem[2]) { if (now_ns() > end || __atomic_load_n(&d->quit, __ATOMIC_ACQUIRE)) { d->sim_fault = 3; break; } sched_yield(); }
                } else if (op == SEM_OP_RELEASE) __atomic_store_n(t, sem[2], __ATOMIC_RELEASE);
                else d->sim_fault = 4;
            }
        }
        __atomic_store_n(&d->pending, 0, __ATOMIC_RELEASE);
    }
}

static int dev_open(Dev *d, int sim) {
    memset(d, 0, sizeof *d);
    d->sim = sim;
    if (sim) {
        d->cpu = aligned_alloc(4096, BUF_BYTES); if (!d->cpu) return -1;
        memset(d->cpu, 0, BUF_BYTES); d->va = SIM_VA;
        return pthread_create(&d->th, NULL, sim_main, d) == 0 ? 0 : -1;
    }
    omega_gpu_session_lock();
    d->s = omega_gpu_session_open();
    d->p = omega_gpu_session_platform();
    if (!d->s || !d->p || omega_gpu_session_alloc(BUF_BYTES, &d->mem) != 0) {
        printf("open: %s\n", omega_gpu_session_last_error()); omega_gpu_session_unlock(); return -1;
    }
    d->cpu = d->mem.cpu; d->va = d->mem.va;
    memset(d->cpu, 0, BUF_BYTES);
    return omega_gpu_session_sync_to_gpu(&d->mem);
}
static int dev_submit(Dev *d, const uint32_t *w, uint32_t n) {
    if (d->sim) { d->words = w; d->nwords = n; __atomic_store_n(&d->pending, 1, __ATOMIC_RELEASE); return 0; }
    return d->p->queue_submit(d->p->ctx, d->s->queue, w, n) == OMEGA_GPU_PLAT_OK ? 0 : -1;
}
static void dev_retire(Dev *d) {
    if (d->sim) { while (__atomic_load_n(&d->pending, __ATOMIC_ACQUIRE)) sched_yield(); return; }
    (void)d->p->queue_retire(d->p->ctx, d->s->queue);
}
static void dev_close(Dev *d, int clean) {
    if (d->sim) { __atomic_store_n(&d->quit, 1, __ATOMIC_RELEASE); pthread_join(d->th, NULL); free(d->cpu); return; }
    if (clean) omega_gpu_session_free(&d->mem); /* an unretired queue may still write it: leave it to the latched session */
    omega_gpu_session_unlock();
    omega_gpu_session_close();
}

static int g_dsb;
static Dev g_dev;
static void publish(Dev *d, uint32_t off, uint32_t v) {
    __atomic_store_n((uint32_t *)(d->cpu + off), v, __ATOMIC_RELEASE);
    if (d->sim) return;
    if (g_dsb) dsb_sy();
    else (void)d->p->sync_cpu_to_gpu(d->p->ctx, d->mem.handle, off, 4);
}
/* spin until the word reaches want (serial order); 0 reached, -1 timed out */
static int spin_until(Dev *d, uint32_t off, uint32_t want, uint64_t timeout_ns) {
    volatile uint32_t *m = (volatile uint32_t *)(d->cpu + off);
    const uint64_t end = now_ns() + timeout_ns;
    for (uint32_t k = 0;; k++) {
        if ((int32_t)(*m - want) >= 0) return 0;
        if (d->sim) sched_yield(); else cpu_relax();
        if ((k & 255u) == 0 && now_ns() > end) return -1;
    }
}

int main(int argc, char **argv) {
    uint32_t iters = 100000, chunk = 1000, warm = 8, timeout_ms = 1000; int wfi = 0, sim = 0, mutate = 0; const char *out_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) chunk = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--warm") == 0 && i + 1 < argc) warm = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--timeout-ms") == 0 && i + 1 < argc) timeout_ms = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) { ++i; if (strcmp(argv[i], "wfi") == 0) wfi = 1; else if (strcmp(argv[i], "plain") != 0) goto usage; }
        else if (strcmp(argv[i], "--sync") == 0 && i + 1 < argc) { ++i; if (strcmp(argv[i], "dsb") == 0) g_dsb = 1; else if (strcmp(argv[i], "platform") != 0) goto usage; }
        else if (strcmp(argv[i], "--sim") == 0) sim = 1;
        else if (strcmp(argv[i], "--mutate") == 0) mutate = 1; /* negative control: RELEASE before ACQUIRE; must FAIL */
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
        else goto usage;
    }
    {
        const uint32_t per = wfi ? 14u : 12u;
        const uint32_t room = (RING_WORDS - 18u - TAIL_WORDS - 64u) / per; /* 64: room for the platform prefix (checked below) */
        if (iters == 0 || chunk == 0 || chunk > room || warm >= chunk || timeout_ms == 0) {
            printf("refused: iters %u chunk %u (1..%u) warm %u (< chunk) timeout_ms %u\n", iters, chunk, room, warm, timeout_ms); return 2;
        }
        FILE *out = out_path ? test_out_open(out_path) : stdout;
        if (!out) { printf("refused: cannot create %s\n", out_path); return 2; }
        uint32_t *pb = malloc(RING_WORDS * 4u);
        if (!pb || dev_open(&g_dev, sim) != 0) { printf("FAIL: device open\n"); return 1; }
        Dev *d = &g_dev;
        if (!sim && omega_gpu_session_submit_prefix_words() > 64u) { printf("FAIL: platform prefix %u words > 64\n", omega_gpu_session_submit_prefix_words()); dev_close(d, 1); return 1; }
        fprintf(out, "HANDOFF 1 variant=%s sync=%s mode=%s%s iters=%u chunk=%u warm=%u\n", wfi ? "wfi" : "plain", g_dsb ? "dsb" : "platform", sim ? "sim" : "chip", mutate ? "-mutant" : "", iters, chunk, warm);
        int ok = 1, latched = 0; uint32_t v = 0, samples = 0, chunks = 0;
        const uint64_t tmo = (uint64_t)timeout_ms * 1000000ull;
        for (uint32_t done = 0; done < iters && ok; chunks++) {
            const uint32_t c = iters - done < chunk ? iters - done : chunk;
            size_t n = 0;
            memcpy(pb, SETUP_WORDS, sizeof SETUP_WORDS); n += 18;
            for (uint32_t i = 1; i <= c; i++) {
                if (mutate) n = emit_sem(pb, n, d->va + OFF_B, v + i, SEM_OP_RELEASE);
                n = emit_sem(pb, n, d->va + OFF_A, v + i, SEM_OP_ACQUIRE);
                if (wfi) { pb[n++] = bw_mthd(1, MTHD_WAIT_FOR_IDLE, 1); pb[n++] = 0; }
                if (!mutate) n = emit_sem(pb, n, d->va + OFF_B, v + i, wfi ? SEM_OP_RELEASE | SEM_RELEASE_WFI : SEM_OP_RELEASE);
            }
            n = emit_tail(pb, n, d->va + OFF_M1);
            publish(d, OFF_M1, 0); publish(d, OFF_M2, 0);
            if (dev_submit(d, pb, (uint32_t)n) != 0) { printf("FAIL: queue_submit refused (%s)\n", sim ? "sim" : omega_gpu_session_last_error()); ok = 0; break; }
            if (chunks == 0) { /* negative control: a wrong A must not release B */
                const uint32_t b0 = __atomic_load_n((uint32_t *)(d->cpu + OFF_B), __ATOMIC_ACQUIRE);
                publish(d, OFF_A, 0xdeadbeefu);
                const uint64_t t = now_ns(); while (now_ns() - t < 2000000ull) cpu_relax();
                const uint32_t b1 = __atomic_load_n((uint32_t *)(d->cpu + OFF_B), __ATOMIC_ACQUIRE);
                const int neg = b0 == v && b1 == v;
                fprintf(out, "NEG %d %u %u\n", neg, b0, b1);
                if (!neg) { printf("FAIL: negative control: B moved %u -> %u under a wrong A\n", b0, b1); ok = 0; }
            }
            for (uint32_t i = 1; i <= c && ok; i++) {
                const uint32_t want = v + i;
                if (__atomic_load_n((uint32_t *)(d->cpu + OFF_B), __ATOMIC_ACQUIRE) != want - 1) { printf("FAIL: B ran ahead at v=%u\n", want); ok = 0; break; }
                const uint64_t t0 = now_ns();
                publish(d, OFF_A, want);
                if (spin_until(d, OFF_B, want, tmo) != 0) { printf("FAIL: B did not reach %u in %u ms\n", want, timeout_ms); ok = 0; latched = 1; break; }
                const uint64_t t1 = now_ns();
                fprintf(out, "%s %" PRIu64 "\n", i <= warm ? "W" : "S", t1 - t0);
                if (i > warm) samples++;
            }
            if (ok && spin_until(d, OFF_M2, OMEGA_BW_MARKER2_PAYLOAD, tmo) != 0) { printf("FAIL: tail marker not seen\n"); ok = 0; latched = 1; }
            /* any failure after the submit leaves the queue parked on an ACQUIRE (or worse): never retire it; latch */
            if (!ok || latched) { latched = 1; if (!sim) omega_gpu_session_latch_uncertain("dra72 handoff: semaphore protocol failed; queue state uncertain"); break; }
            dev_retire(d);
            v += c; done += c;
        }
        if (sim && g_dev.sim_fault) { printf("FAIL: sim interpreter fault %d\n", g_dev.sim_fault); ok = 0; }
        fprintf(out, "END %d %u %u\n", ok, samples, chunks);
        if (out != stdout) fclose(out);
        dev_close(d, !latched);
        free(pb);
        printf("gpu_dra72_handoff_test: mode=%s samples=%u chunks=%u: %s\n", sim ? "sim" : "chip", samples, chunks, ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
usage:
    printf("usage: %s [--iters N] [--chunk C] [--warm W] [--variant plain|wfi] [--sync platform|dsb] [--timeout-ms MS] [--sim] [--mutate] [--out FILE]\n", argv[0]);
    return 2;
}
