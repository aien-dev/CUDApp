/* omega_fp8_slot_table.c: host-only slot state machine (see the header). */
#include "omega_fp8_slot_table.h"
#include <stdlib.h>
#include <string.h>

int omega_fp8_slot_table_init(OmegaFp8SlotTable *t, uint32_t slots) {
    if (!t || slots == 0) return OMEGA_FP8_SLOT_BAD_ARGS;
    memset(t, 0, sizeof *t);
    t->rec = calloc(slots, sizeof *t->rec);
    if (!t->rec) return OMEGA_FP8_SLOT_BAD_ARGS;
    if (pthread_mutex_init(&t->mu, NULL) != 0) { free(t->rec); t->rec = NULL; return OMEGA_FP8_SLOT_BAD_ARGS; }
    t->slots = slots;
    t->next_version = 1;
    return OMEGA_FP8_SLOT_OK;
}

void omega_fp8_slot_table_fini(OmegaFp8SlotTable *t) {
    if (!t || !t->rec) return;
    pthread_mutex_destroy(&t->mu);
    free(t->rec);
    t->rec = NULL;
    t->slots = 0;
}

#define LOCKED(t, body) do { pthread_mutex_lock(&(t)->mu); body; pthread_mutex_unlock(&(t)->mu); } while (0)

int omega_fp8_slot_begin_load(OmegaFp8SlotTable *t, uint32_t slot) {
    if (!t || !t->rec || slot >= t->slots) return OMEGA_FP8_SLOT_BAD_ARGS;
    int rc = OMEGA_FP8_SLOT_OK;
    LOCKED(t, {
        OmegaFp8SlotRec *r = &t->rec[slot];
        if (r->state == OMEGA_FP8_SLOT_LOADING || (r->pins && !t->mutant)) { rc = OMEGA_FP8_SLOT_BUSY; t->stats.busy_refusals++; }
        else r->state = OMEGA_FP8_SLOT_LOADING; /* hides the old version: pin(old) is STALE from here */
    });
    return rc;
}

int omega_fp8_slot_finish_load(OmegaFp8SlotTable *t, uint32_t slot, uint64_t key, int ok, uint64_t *version) {
    if (version) *version = 0;
    if (!t || !t->rec || slot >= t->slots) return OMEGA_FP8_SLOT_BAD_ARGS;
    int rc = OMEGA_FP8_SLOT_OK;
    LOCKED(t, {
        OmegaFp8SlotRec *r = &t->rec[slot];
        if (r->state != OMEGA_FP8_SLOT_LOADING) rc = OMEGA_FP8_SLOT_BAD_ARGS;
        else if (ok) {
            r->state = OMEGA_FP8_SLOT_READY; r->key = key; r->version = t->next_version++;
            if (version) *version = r->version;
            t->stats.loads++;
        } else { r->state = OMEGA_FP8_SLOT_EMPTY; r->key = 0; t->stats.load_failures++; }
    });
    return rc;
}

int omega_fp8_slot_pin(OmegaFp8SlotTable *t, uint32_t slot, uint64_t version) {
    if (!t || !t->rec || slot >= t->slots) return OMEGA_FP8_SLOT_BAD_ARGS;
    int rc = OMEGA_FP8_SLOT_OK;
    LOCKED(t, {
        OmegaFp8SlotRec *r = &t->rec[slot];
        if (r->state != OMEGA_FP8_SLOT_READY || r->version != version || version == 0) { rc = OMEGA_FP8_SLOT_STALE; t->stats.stale_refusals++; }
        else {
            r->pins++; t->stats.pins++; t->stats.pinned_now++;
            if (t->stats.pinned_now > t->stats.pinned_high) t->stats.pinned_high = t->stats.pinned_now;
        }
    });
    return rc;
}

int omega_fp8_slot_unpin(OmegaFp8SlotTable *t, uint32_t slot, uint64_t version) {
    if (!t || !t->rec || slot >= t->slots) return OMEGA_FP8_SLOT_BAD_ARGS;
    int rc = OMEGA_FP8_SLOT_OK;
    LOCKED(t, {
        OmegaFp8SlotRec *r = &t->rec[slot];
        /* Under the mutant a pinned slot may have been reloaded: the pin is still dropped (version not checked). */
        if (r->pins == 0 || (!t->mutant && r->version != version)) rc = OMEGA_FP8_SLOT_BAD_ARGS;
        else { r->pins--; t->stats.pinned_now--; }
    });
    return rc;
}

int omega_fp8_slot_evict(OmegaFp8SlotTable *t, uint32_t slot) {
    if (!t || !t->rec || slot >= t->slots) return OMEGA_FP8_SLOT_BAD_ARGS;
    int rc = OMEGA_FP8_SLOT_OK;
    LOCKED(t, {
        OmegaFp8SlotRec *r = &t->rec[slot];
        if (r->pins || r->state == OMEGA_FP8_SLOT_LOADING) { rc = OMEGA_FP8_SLOT_BUSY; t->stats.busy_refusals++; }
        else { if (r->state == OMEGA_FP8_SLOT_READY) t->stats.evictions++; r->state = OMEGA_FP8_SLOT_EMPTY; r->key = 0; }
    });
    return rc;
}

int omega_fp8_slot_query(OmegaFp8SlotTable *t, uint32_t slot, OmegaFp8SlotRec *out) {
    if (!t || !t->rec || slot >= t->slots || !out) return OMEGA_FP8_SLOT_BAD_ARGS;
    LOCKED(t, { *out = t->rec[slot]; });
    return OMEGA_FP8_SLOT_OK;
}

int omega_fp8_slot_table_idle(OmegaFp8SlotTable *t) {
    if (!t || !t->rec) return 1;
    int idle = 1;
    LOCKED(t, {
        for (uint32_t i = 0; i < t->slots; i++)
            if (t->rec[i].pins || t->rec[i].state == OMEGA_FP8_SLOT_LOADING) { idle = 0; break; }
    });
    return idle;
}

void omega_fp8_slot_table_stats(OmegaFp8SlotTable *t, OmegaFp8SlotStats *out) {
    if (!out) return;
    memset(out, 0, sizeof *out);
    if (!t || !t->rec) return;
    LOCKED(t, { *out = t->stats; });
}

void omega_fp8_slot_table_set_mutant(OmegaFp8SlotTable *t, int mutant) {
    if (!t || !t->rec) return;
    LOCKED(t, { t->mutant = mutant; });
}
