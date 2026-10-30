#ifndef OMEGA_FP8_SLOT_TABLE_H
#define OMEGA_FP8_SLOT_TABLE_H
/*
 * omega_fp8_slot_table: the host-only state machine behind omega_gpu_fp8_slots (Expert Ribbon, omega#533).
 * No device code: every rule about who may read or rewrite a slot lives here so it can be tested without a chip.
 *
 * One record per slot: state, pins, key (caller's expert identity), version.
 *   EMPTY   --begin_load-->  LOADING  --finish_load(ok)--> READY (version = next ring-wide version, never reused)
 *                            LOADING  --finish_load(fail)-> EMPTY (nothing published)
 *   READY   --begin_load (0 pins)--> LOADING (the old version dies at that moment: pin(old) is STALE from then on)
 *   READY   --evict (0 pins)--> EMPTY
 * pin(slot, version) succeeds only on READY with exactly that version; begin_load and evict refuse (BUSY) while
 * pins > 0. A pin is held for as long as any launch or recorded command list may read the slot; the caller drops it
 * after the launch's completion marker (synchronous APIs: when the call returns; command lists: after submit).
 * Versions are a single ring-wide counter starting at 1, so a (slot, version) pair names exactly one load (no ABA).
 * Thread-safe (one mutex per table).
 */
#include <pthread.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum {
    OMEGA_FP8_SLOT_OK = 0,
    OMEGA_FP8_SLOT_BAD_ARGS = -1, /* slot out of range, unpin without a pin, NULL */
    OMEGA_FP8_SLOT_BUSY = -6,     /* pinned or already loading: nothing changed */
    OMEGA_FP8_SLOT_STALE = -7     /* the slot is not READY at that version (evicted, reloaded or never loaded) */
};

typedef enum { OMEGA_FP8_SLOT_EMPTY = 0, OMEGA_FP8_SLOT_LOADING = 1, OMEGA_FP8_SLOT_READY = 2 } OmegaFp8SlotState;

typedef struct {
    uint32_t state; /* OmegaFp8SlotState */
    uint32_t pins;
    uint64_t key;     /* caller's identity of the loaded expert (meaningful in READY) */
    uint64_t version; /* version of the READY contents; 0 = never published */
} OmegaFp8SlotRec;

typedef struct {
    uint64_t loads;          /* finish_load(ok) */
    uint64_t load_failures;  /* finish_load(fail) */
    uint64_t busy_refusals;  /* begin_load / evict refused because pinned or loading */
    uint64_t stale_refusals; /* pin refused */
    uint64_t evictions;
    uint64_t pins;           /* successful pins */
    uint32_t pinned_now;     /* sum of pins over slots */
    uint32_t pinned_high;    /* high-water of pinned_now */
} OmegaFp8SlotStats;

typedef struct {
    pthread_mutex_t mu;
    uint32_t slots;
    OmegaFp8SlotRec *rec;
    uint64_t next_version;
    int mutant; /* TEST ONLY: 1 = begin_load ignores pins (the negative control must catch a rewritten in-use slot) */
    OmegaFp8SlotStats stats;
} OmegaFp8SlotTable;

int omega_fp8_slot_table_init(OmegaFp8SlotTable *t, uint32_t slots);
void omega_fp8_slot_table_fini(OmegaFp8SlotTable *t);
int omega_fp8_slot_begin_load(OmegaFp8SlotTable *t, uint32_t slot);
/* ok != 0 publishes (READY, *version = new version); ok == 0 leaves the slot EMPTY. Must follow begin_load. */
int omega_fp8_slot_finish_load(OmegaFp8SlotTable *t, uint32_t slot, uint64_t key, int ok, uint64_t *version);
int omega_fp8_slot_pin(OmegaFp8SlotTable *t, uint32_t slot, uint64_t version);
int omega_fp8_slot_unpin(OmegaFp8SlotTable *t, uint32_t slot, uint64_t version);
int omega_fp8_slot_evict(OmegaFp8SlotTable *t, uint32_t slot);
int omega_fp8_slot_query(OmegaFp8SlotTable *t, uint32_t slot, OmegaFp8SlotRec *out);
/* 1 when no slot is pinned or loading (safe to free the memory behind the table). */
int omega_fp8_slot_table_idle(OmegaFp8SlotTable *t);
void omega_fp8_slot_table_stats(OmegaFp8SlotTable *t, OmegaFp8SlotStats *out);
void omega_fp8_slot_table_set_mutant(OmegaFp8SlotTable *t, int mutant);

#ifdef __cplusplus
}
#endif
#endif
