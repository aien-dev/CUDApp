/* gpu_fp8_slots_test: Expert Ribbon P1 (aienix#4 / omega#533), versioned replaceable FP8 expert slots.
 *
 *   --host-only   the slot state machine (omega_fp8_slot_table.c) without a device: publish only after a finished load,
 *                 stale handles, double load, load/evict while pinned, ABA across reuse, failed loads, a threaded race
 *                 (readers pin/unpin while a writer reloads; a reader must never observe its slot rewritten), mutant.
 *   --chip        the pool on the GB10 (caller holds the GB10 lock):
 *                 1. parity: E random experts resident (one tensor) vs the same experts loaded into a smaller pool,
 *                    grouped launch, y bit-identical, gate|up (1536x2048) and down (2048x768) shapes;
 *                 2. churn: --rounds reloads cycling every slot through every expert, parity after each reload;
 *                 3. refusals on the chip pool: load while pinned BUSY, pin of a replaced version STALE, entry unpinned;
 *                 4. reader fence through a recorded command list: the list's group table points at a pinned slot; a
 *                    load into that slot before submit must be refused and the list must read the pinned expert;
 *                    NEGATIVE CONTROL (--mutant): the pool ignores pins, the load goes through and the list reads the
 *                    replaced weights, which the parity check must catch (exit 1 = caught);
 *                 5. memory: pool device bytes vs the session's driver allocation counter, pinned high-water.
 * Exit 0 PASS, 1 FAIL (or, with --mutant, CAUGHT), 2 usage. Verdict line last. */
#include "omega_fp8_slot_table.h"
#include "omega_gpu_fp8_slots.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_session.h"
#include "omega_gpu_cmdlist.h"
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failed, g_checks;
#define CHECK(c, ...) do { g_checks++; if (c) { printf("ok   "); } else { printf("FAIL "); g_failed++; } printf(__VA_ARGS__); printf("\n"); } while (0)

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }

/* ---------------- host-only: the state machine ---------------- */

typedef struct { OmegaFp8SlotTable *t; uint64_t *shadow_version; volatile int stop; uint64_t observed_rewrites, pins_ok; } Race;

static void *reader(void *arg) {
    Race *r = arg;
    while (!r->stop) {
        OmegaFp8SlotRec rec;
        if (omega_fp8_slot_query(r->t, 0, &rec) != 0) continue;
        if (rec.state != OMEGA_FP8_SLOT_READY) continue;
        if (omega_fp8_slot_pin(r->t, 0, rec.version) != 0) continue;
        r->pins_ok++;
        /* while pinned, the shadow (the slot bytes) must keep holding the content of the pinned key */
        for (int i = 0; i < 64; i++) if (__atomic_load_n(r->shadow_version, __ATOMIC_ACQUIRE) != rec.key) r->observed_rewrites++;
        (void)omega_fp8_slot_unpin(r->t, 0, rec.version);
    }
    return NULL;
}

static int host_tests(void) {
    OmegaFp8SlotTable t;
    CHECK(omega_fp8_slot_table_init(&t, 4) == 0, "init 4 slots");
    uint64_t v1 = 0, v2 = 0, v3 = 0;
    CHECK(omega_fp8_slot_pin(&t, 0, 1) == OMEGA_FP8_SLOT_STALE, "pin of a never-loaded slot is STALE");
    CHECK(omega_fp8_slot_begin_load(&t, 0) == 0, "begin load slot 0");
    CHECK(omega_fp8_slot_begin_load(&t, 0) == OMEGA_FP8_SLOT_BUSY, "double load refused BUSY");
    CHECK(omega_fp8_slot_pin(&t, 0, 1) == OMEGA_FP8_SLOT_STALE, "a LOADING slot cannot be pinned (not visible before publish)");
    CHECK(omega_fp8_slot_finish_load(&t, 0, 100, 1, &v1) == 0 && v1 == 1, "publish -> version 1");
    CHECK(omega_fp8_slot_finish_load(&t, 0, 100, 1, &v2) == OMEGA_FP8_SLOT_BAD_ARGS && v2 == 0, "finish without begin refused");
    CHECK(omega_fp8_slot_pin(&t, 0, v1) == 0, "pin v1");
    CHECK(omega_fp8_slot_begin_load(&t, 0) == OMEGA_FP8_SLOT_BUSY, "reload while pinned refused BUSY");
    CHECK(omega_fp8_slot_evict(&t, 0) == OMEGA_FP8_SLOT_BUSY, "evict while pinned refused BUSY");
    CHECK(omega_fp8_slot_unpin(&t, 0, v1 + 7) == OMEGA_FP8_SLOT_BAD_ARGS, "unpin with a wrong version refused");
    CHECK(omega_fp8_slot_unpin(&t, 0, v1) == 0, "unpin v1");
    CHECK(omega_fp8_slot_unpin(&t, 0, v1) == OMEGA_FP8_SLOT_BAD_ARGS, "unpin without a pin refused");
    CHECK(omega_fp8_slot_begin_load(&t, 0) == 0 && omega_fp8_slot_finish_load(&t, 0, 200, 1, &v2) == 0 && v2 > v1, "reload -> newer version");
    CHECK(omega_fp8_slot_pin(&t, 0, v1) == OMEGA_FP8_SLOT_STALE, "stale handle (old version) refused after reload");
    /* ABA: slot 0 holds key 200; load key 100 again: the version must differ from v1 */
    CHECK(omega_fp8_slot_begin_load(&t, 0) == 0 && omega_fp8_slot_finish_load(&t, 0, 100, 1, &v3) == 0 && v3 != v1 && v3 > v2,
          "same key reloaded gets a fresh version (no ABA): v1=%" PRIu64 " v3=%" PRIu64, v1, v3);
    CHECK(omega_fp8_slot_begin_load(&t, 1) == 0 && omega_fp8_slot_finish_load(&t, 1, 5, 0, &v2) == 0 && v2 == 0, "failed load publishes nothing");
    OmegaFp8SlotRec rec;
    CHECK(omega_fp8_slot_query(&t, 1, &rec) == 0 && rec.state == OMEGA_FP8_SLOT_EMPTY, "failed load leaves the slot EMPTY");
    CHECK(omega_fp8_slot_evict(&t, 0) == 0 && omega_fp8_slot_pin(&t, 0, v3) == OMEGA_FP8_SLOT_STALE, "evicted slot cannot be pinned");
    CHECK(omega_fp8_slot_begin_load(&t, 9) == OMEGA_FP8_SLOT_BAD_ARGS, "out-of-range slot refused");
    /* repeated evict/reinsert: versions strictly increase */
    uint64_t last = v3; int mono = 1;
    for (int i = 0; i < 1000; i++) {
        uint64_t v = 0;
        if (omega_fp8_slot_begin_load(&t, (uint32_t)(i % 4)) != 0 || omega_fp8_slot_finish_load(&t, (uint32_t)(i % 4), (uint64_t)i, 1, &v) != 0 || v <= last) mono = 0;
        last = v;
        if (i % 3 == 0 && omega_fp8_slot_evict(&t, (uint32_t)(i % 4)) != 0) mono = 0;
    }
    CHECK(mono, "1000 evict/reinsert cycles: versions strictly increase");
    CHECK(omega_fp8_slot_table_idle(&t), "idle when nothing is pinned or loading");
    omega_fp8_slot_table_fini(&t);

    /* threaded race: 4 readers pin slot 0 at its current version; one writer reloads it as fast as it can. The shadow
     * stands for the slot's bytes: the writer bumps it between begin_load and finish_load (while the slot is hidden). */
    for (int mutant = 0; mutant <= 1; mutant++) {
        OmegaFp8SlotTable rt; omega_fp8_slot_table_init(&rt, 1);
        omega_fp8_slot_table_set_mutant(&rt, mutant);
        uint64_t shadow = 0, v = 0;
        omega_fp8_slot_begin_load(&rt, 0); omega_fp8_slot_finish_load(&rt, 0, 0, 1, &v);
        __atomic_store_n(&shadow, 0, __ATOMIC_RELEASE);
        Race r = { .t = &rt, .shadow_version = &shadow };
        pthread_t th[4];
        for (int i = 0; i < 4; i++) pthread_create(&th[i], NULL, reader, &r);
        uint64_t reloads = 0, busy = 0;
        for (int i = 0; i < 200000; i++) {
            if (omega_fp8_slot_begin_load(&rt, 0) != 0) { busy++; continue; }
            __atomic_store_n(&shadow, 0xDEADull << 32, __ATOMIC_RELEASE); /* "bytes" half-written */
            __atomic_store_n(&shadow, (uint64_t)i + 1, __ATOMIC_RELEASE); /* "bytes" complete: the content of key i + 1 */
            uint64_t nv = 0; omega_fp8_slot_finish_load(&rt, 0, (uint64_t)i + 1, 1, &nv);
            reloads++;
        }
        r.stop = 1;
        for (int i = 0; i < 4; i++) pthread_join(th[i], NULL);
        if (!mutant)
            CHECK(r.observed_rewrites == 0 && reloads > 0 && r.pins_ok > 0,
                  "race: %" PRIu64 " reloads, %" PRIu64 " busy refusals, %" PRIu64 " reader pins, rewrites seen under a pin = %" PRIu64,
                  reloads, busy, r.pins_ok, r.observed_rewrites);
        else
            printf("info negative control (mutant ignores pins): %" PRIu64 " reloads, %" PRIu64 " reader pins, rewrites seen under a pin = %" PRIu64 " (%s)\n",
                   reloads, r.pins_ok, r.observed_rewrites, r.observed_rewrites ? "CAUGHT" : "not observed this run");
        if (mutant) CHECK(r.observed_rewrites > 0, "mutant: readers observe rewrites under a pin (the race check has teeth)");
        omega_fp8_slot_table_fini(&rt);
    }
    return 0;
}

/* ---------------- chip ---------------- */

static void fill_expert(uint32_t seed, uint32_t n, uint32_t k, uint8_t *w, uint16_t *s) {
    uint32_t st = seed * 2654435761u + 7u;
    for (size_t i = 0; i < (size_t)n * k; i++) { uint8_t b = (uint8_t)(lcg(&st) >> 24); w[i] = (b & 0x7F) == 0x7F ? 0x3C : b; }
    for (size_t i = 0; i < (size_t)(n / 128) * (k / 128); i++) s[i] = (uint16_t)(0x3B00u + (lcg(&st) >> 24) % 0x200u); /* bf16 ~2^-9..2^-7 */
}

static int chip_shape(uint32_t n, uint32_t k, uint32_t E, uint32_t S, uint32_t rounds, int mutant) {
    printf("== shape n=%u k=%u, %u resident experts, pool of %u slots, %u churn rounds%s\n", n, k, E, S, rounds, mutant ? " (MUTANT)" : "");
    const size_t wb = (size_t)n * k, sc = (size_t)(n / 128) * (k / 128);
    uint8_t *w = malloc(wb * E); uint16_t *s = malloc(sc * 2 * E);
    float *x = malloc((size_t)k * 4 * 8), *yr = malloc((size_t)n * 4 * 8), *yp = malloc((size_t)n * 4 * 8);
    if (!w || !s || !x || !yr || !yp) return 1;
    for (uint32_t e = 0; e < E; e++) fill_expert(e + n, n, k, w + wb * e, s + sc * e);
    uint32_t st = 99;
    for (uint32_t i = 0; i < k * 8; i++) x[i] = (float)(lcg(&st) >> 8) / 16777216.0f * 2.0f - 1.0f;
    OmegaGpuFp8Tensor *res = NULL;
    int rc = omega_gpu_fp8_tensor_upload(E, n, k, w, s, &res);
    if (rc != 0) { printf("FAIL resident upload: %s (%s)\n", omega_gpu_fp8_rc_name(rc), omega_gpu_fp8_last_error()); return 1; }
    OmegaGpuAllocStats a0; omega_gpu_session_alloc_stats(&a0);
    OmegaGpuFp8Slots *p = NULL;
    rc = omega_gpu_fp8_slots_alloc(S, n, k, &p);
    OmegaGpuAllocStats a1; omega_gpu_session_alloc_stats(&a1);
    if (rc != 0) { printf("FAIL pool alloc: %s (%s)\n", omega_gpu_fp8_rc_name(rc), omega_gpu_fp8_last_error()); return 1; }
    OmegaGpuFp8SlotsStats ps; omega_gpu_fp8_slots_stats(p, &ps);
    CHECK(a1.alloc_bytes - a0.alloc_bytes >= ps.device_bytes && a1.alloc_bytes - a0.alloc_bytes <= ps.device_bytes + 2u * 65536u,
          "pool driver bytes %" PRIu64 " for %" PRIu64 " payload bytes (page rounding only)", a1.alloc_bytes - a0.alloc_bytes, ps.device_bytes);
    uint64_t *ver = calloc(S, 8); uint32_t *held = calloc(S, 4);
    for (uint32_t sl = 0; sl < S; sl++) {
        held[sl] = sl % E;
        rc = omega_gpu_fp8_slots_load(p, sl, held[sl], w + wb * held[sl], s + sc * held[sl], &ver[sl]);
        if (rc != 0) { printf("FAIL load slot %u: %s\n", sl, omega_gpu_fp8_slots_rc_name(rc)); return 1; }
    }
    /* 1 + 2: parity after every reload; groups = min(S, 8) slots per launch, shared x row (gate|up) or a row per group (down) */
    const uint32_t G = S < 8 ? S : 8;
    uint32_t mism = 0, launches = 0;
    for (uint32_t r = 0; r <= rounds && !mutant; r++) {
        if (r > 0) { /* reload one slot with the next expert in the cycle */
            uint32_t sl = (r - 1) % S; held[sl] = (held[sl] + 1 + (r % 3)) % E;
            rc = omega_gpu_fp8_slots_load(p, sl, held[sl], w + wb * held[sl], s + sc * held[sl], &ver[sl]);
            if (rc != 0) { printf("FAIL reload r%u: %s\n", r, omega_gpu_fp8_slots_rc_name(rc)); mism++; break; }
        }
        uint32_t sls[8], ids[8]; uint64_t vs[8];
        for (uint32_t g = 0; g < G; g++) { sls[g] = (r + g) % S; vs[g] = ver[sls[g]]; ids[g] = held[sls[g]]; }
        const uint32_t xr = (r & 1) ? G : 1;
        int a = omega_gpu_fp8_gemv_grouped_f32(res, G, ids, x, xr, yr, NULL);
        int b = omega_gpu_fp8_slots_gemv_grouped_f32(p, G, sls, vs, x, xr, yp, NULL);
        launches += 2;
        if (a != 0 || b != 0) { printf("FAIL launch r%u: resident %s pool %s (%s)\n", r, omega_gpu_fp8_rc_name(a), omega_gpu_fp8_slots_rc_name(b), omega_gpu_fp8_last_error()); mism++; break; }
        if (memcmp(yr, yp, (size_t)G * n * 4) != 0) mism++;
    }
    if (!mutant) CHECK(mism == 0, "parity: %u launches, pool y bit-identical to the resident tensor after every reload", launches);

    /* 3: refusals on the chip pool */
    if (!mutant) {
        CHECK(omega_gpu_fp8_slots_pin(p, 0, ver[0]) == 0, "pin slot 0");
        uint64_t vx = 0;
        CHECK(omega_gpu_fp8_slots_load(p, 0, 7, w, s, &vx) == OMEGA_GPU_FP8_SLOTS_BUSY && vx == 0, "load into a pinned slot refused BUSY");
        CHECK(omega_gpu_fp8_slots_evict(p, 0) == OMEGA_GPU_FP8_SLOTS_BUSY, "evict of a pinned slot refused BUSY");
        omega_gpu_fp8_slots_unpin(p, 0, ver[0]);
        uint64_t wva, sva;
        CHECK(omega_gpu_fp8_slots_entry(p, 0, ver[0], &wva, &sva) == OMEGA_GPU_FP8_BAD_ARGS, "entry of an unpinned slot refused");
        uint64_t old = ver[0];
        CHECK(omega_gpu_fp8_slots_load(p, 0, held[0], w + wb * held[0], s + sc * held[0], &ver[0]) == 0 && ver[0] > old, "reload slot 0 -> newer version");
        uint32_t sl0 = 0;
        CHECK(omega_gpu_fp8_slots_gemv_grouped_f32(p, 1, &sl0, &old, x, 1, yp, NULL) == OMEGA_GPU_FP8_SLOTS_STALE, "launch with the replaced version refused STALE (nothing launched)");
        uint8_t *bad = malloc(wb); memcpy(bad, w, wb); bad[wb / 2] = 0x7F;
        uint64_t vb = 1;
        CHECK(omega_gpu_fp8_slots_load(p, 1, 9, bad, s, &vb) == OMEGA_GPU_FP8_NAN_WEIGHT && vb == 0, "NaN weight refused");
        OmegaGpuFp8SlotInfo inf; omega_gpu_fp8_slots_query(p, 1, &inf);
        CHECK(inf.state == 0 && omega_gpu_fp8_slots_pin(p, 1, ver[1]) == OMEGA_GPU_FP8_SLOTS_STALE, "refused load leaves the slot EMPTY; its old version cannot be pinned");
        free(bad);
        rc = omega_gpu_fp8_slots_load(p, 1, held[1], w + wb * held[1], s + sc * held[1], &ver[1]);
        if (rc != 0) g_failed++;
    }

    /* 4: reader fence through a recorded command list */
    {
        if (mutant) omega_gpu_fp8_slots_test_set_mutant(p, 1);
        const uint32_t A = held[0], B = (held[0] + 1) % E;
        OmegaGpuCmdList *cl = NULL; OmegaGpuMem tab, xd, yd;
        if (omega_gpu_cmdlist_create(4, &cl) != 0) { printf("FAIL cmdlist create: %s\n", omega_gpu_session_last_error()); return 1; }
        omega_gpu_session_lock();
        int arc = omega_gpu_session_alloc(16u, &tab) | omega_gpu_session_alloc((size_t)k * 4u, &xd) | omega_gpu_session_alloc((size_t)n * 4u, &yd);
        omega_gpu_session_unlock();
        if (arc) { printf("FAIL alloc: %s\n", omega_gpu_session_last_error()); return 1; }
        omega_fp8_pack_x_f32(x, k, xd.cpu);
        float *y = yd.cpu; for (uint32_t i = 0; i < n; i++) y[i] = -999.0f;
        /* reference for A and B on the resident tensor */
        float *ya = malloc((size_t)n * 4), *yb = malloc((size_t)n * 4);
        int r1 = omega_gpu_fp8_gemv_grouped_f32(res, 1, &A, x, 1, ya, NULL), r2 = omega_gpu_fp8_gemv_grouped_f32(res, 1, &B, x, 1, yb, NULL);
        CHECK(r1 == 0 && r2 == 0 && memcmp(ya, yb, (size_t)n * 4) != 0, "references: experts %u and %u give different y", A, B);
        /* warm the kernel (grouped 1, x_rows 1) through the pool before recording */
        uint32_t sl0 = 0; float *yw = malloc((size_t)n * 4);
        (void)omega_gpu_fp8_slots_gemv_grouped_f32(p, 1, &sl0, &ver[0], x, 1, yw, NULL);
        int ok = omega_gpu_fp8_slots_pin(p, 0, ver[0]) == 0;
        uint64_t *tw = tab.cpu;
        ok = ok && omega_gpu_fp8_slots_entry(p, 0, ver[0], &tw[0], &tw[1]) == 0;
        ok = ok && omega_gpu_session_sync_to_gpu(&tab) == 0 && omega_gpu_session_sync_to_gpu(&xd) == 0 && omega_gpu_session_sync_to_gpu(&yd) == 0;
        omega_gpu_session_lock();
        omega_gpu_cmdlist_reset(cl);
        ok = ok && omega_gpu_fp8_gemv_grouped_record(cl, omega_gpu_fp8_slots_tensor(p), 1, tab.va, xd.va, 1, yd.va) == 0;
        omega_gpu_session_unlock();
        /* the replacement attempt: between record and submit (the list may read the slot from here until submit returns) */
        uint64_t vnew = 0;
        int lrc = omega_gpu_fp8_slots_load(p, 0, B, w + wb * B, s + sc * B, &vnew);
        if (!mutant) CHECK(lrc == OMEGA_GPU_FP8_SLOTS_BUSY, "load into the slot a recorded list reads: refused BUSY");
        else printf("info mutant: load into the pinned slot returned %s\n", omega_gpu_fp8_slots_rc_name(lrc));
        OmegaGpuCmdListOpts o; memset(&o, 0, sizeof o); o.mode = OMEGA_GPU_CMDLIST_PCAS;
        OmegaGpuCmdListStats cs; memset(&cs, 0, sizeof cs);
        omega_gpu_session_lock();
        ok = ok && omega_gpu_cmdlist_submit(cl, &o, &cs) == 0;
        omega_gpu_session_unlock();
        ok = ok && omega_gpu_session_sync_to_cpu(&yd) == 0;
        (void)omega_gpu_fp8_slots_unpin(p, 0, ver[0]);
        const int same_a = ok && memcmp(y, ya, (size_t)n * 4) == 0, same_b = ok && memcmp(y, yb, (size_t)n * 4) == 0;
        if (!mutant) CHECK(ok && same_a, "the list read the pinned expert %u bit-exactly (submit ok=%d)", A, ok);
        else {
            CHECK(ok && same_a, "NEGATIVE CONTROL: list output equals pinned expert %u (expected to FAIL under the mutant; read B=%d)", A, same_b);
        }
        free(ya); free(yb); free(yw);
        omega_gpu_session_lock(); omega_gpu_session_free(&tab); omega_gpu_session_free(&xd); omega_gpu_session_free(&yd); omega_gpu_session_unlock();
        omega_gpu_cmdlist_free(cl);
        if (mutant) omega_gpu_fp8_slots_test_set_mutant(p, 0);
    }

    omega_gpu_fp8_slots_stats(p, &ps);
    printf("stats slots=%u device_bytes=%" PRIu64 " loads=%" PRIu64 " load_failures=%" PRIu64 " busy=%" PRIu64 " stale=%" PRIu64
           " pins=%" PRIu64 " pinned_now=%u pinned_high=%u bytes_loaded=%" PRIu64 " load_ns=%" PRIu64 " load_GBps=%.3f\n",
           ps.slots, ps.device_bytes, ps.loads, ps.load_failures, ps.busy_refusals, ps.stale_refusals, ps.pins, ps.pinned_now,
           ps.pinned_high, ps.bytes_loaded, ps.load_ns, ps.load_ns ? (double)ps.bytes_loaded / (double)ps.load_ns : 0.0);
    CHECK(ps.pinned_now == 0 && ps.pinned_high <= G, "no pin leaked (pinned_now 0, high-water %u <= %u)", ps.pinned_high, G);
    OmegaGpuAllocStats a2; omega_gpu_session_alloc_stats(&a2);
    CHECK(omega_gpu_fp8_slots_free(p) == 0, "pool freed");
    OmegaGpuAllocStats a3; omega_gpu_session_alloc_stats(&a3);
    CHECK(a3.frees >= a2.frees + 2, "pool returned its 2 driver buffers (frees %" PRIu64 " -> %" PRIu64 ")", a2.frees, a3.frees);
    omega_gpu_fp8_tensor_free(res);
    free(w); free(s); free(x); free(yr); free(yp); free(ver); free(held);
    return 0;
}

int main(int argc, char **argv) {
    int host = 0, chip = 0, mutant = 0; uint32_t rounds = 64;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--host-only")) host = 1;
        else if (!strcmp(argv[i], "--chip")) chip = 1;
        else if (!strcmp(argv[i], "--mutant")) mutant = 1;
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = (uint32_t)atoi(argv[++i]);
        else { printf("usage: %s --host-only | --chip [--rounds R] [--mutant]\n", argv[0]); return 2; }
    }
    if (!host && !chip) { printf("usage: %s --host-only | --chip [--rounds R] [--mutant]\n", argv[0]); return 2; }
    if (host) host_tests();
    if (chip) {
        (void)omega_gpu_session_set_spin_us(2000);
        chip_shape(1536, 2048, 12, 5, rounds, mutant); /* gate|up */
        if (!mutant) chip_shape(2048, 768, 12, 5, rounds, 0); /* down */
        if (omega_gpu_session_is_blocked()) { printf("FAIL session latched\n"); g_failed++; }
    }
    if (mutant) {
        printf("GPU_FP8_SLOTS_NEGATIVE_CONTROL: %s (%d of %d checks failed)\n", g_failed ? "CAUGHT" : "NOT CAUGHT", g_failed, g_checks);
        return g_failed ? 1 : 0;
    }
    printf("GPU_FP8_SLOTS: %s (%d checks, %d failed)\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
