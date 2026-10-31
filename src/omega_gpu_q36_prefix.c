/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Device-resident prefix snapshots of the hybrid Qwen3.6-35B (DRA-86); the layout and refusals are in the header. */
#include "omega_gpu_q36_prefix.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int g_mutant;

void omega_q36_prefix_test_set_mutant(int mutant) { g_mutant = mutant; }

int omega_q36_prefix_budget_from_env(uint64_t *bytes) {
    if (!bytes) return OMEGA_Q36_PREFIX_BAD_ARGS;
    const char *v = getenv(OMEGA_Q36_PREFIX_POOL_ENV);
    double gb = OMEGA_Q36_PREFIX_DEFAULT_GB;
    if (v && *v) {
        char *end = NULL;
        gb = strtod(v, &end);
        if (!end || *end || !isfinite(gb) || gb < 0.0 || gb > 1048576.0) return OMEGA_Q36_PREFIX_BAD_ARGS;
    }
    *bytes = (uint64_t)(gb * 1073741824.0);
    return OMEGA_Q36_PREFIX_OK;
}

void omega_q36_prefix_pool_init(OmegaQ36PrefixPool *p, uint64_t budget) {
    if (!p) return;
    memset(p, 0, sizeof *p);
    p->budget = budget;
}

int omega_q36_prefix_live_from_views(OmegaQ36PrefixLive *live, const OmegaGpuDecodeAttn256CacheView *attn, uint32_t attn_layers,
                                     const OmegaGpuGdnStateView *gdn) {
    if (!live || !attn || !gdn || !attn_layers || attn_layers > OMEGA_Q36_PREFIX_MAX_ATTN) return OMEGA_Q36_PREFIX_BAD_ARGS;
    memset(live, 0, sizeof *live);
    live->attn_layers = attn_layers;
    live->kv_heads = attn[0].kv_heads;
    live->max_pos = attn[0].max_pos;
    for (uint32_t l = 0; l < attn_layers; l++) {
        if (attn[l].kv_heads != live->kv_heads || attn[l].max_pos != live->max_pos) return OMEGA_Q36_PREFIX_SHAPE_MISMATCH;
        if (!attn[l].k_va || !attn[l].v_va) return OMEGA_Q36_PREFIX_BAD_ARGS;
        live->k_va[l] = attn[l].k_va;
        live->v_va[l] = attn[l].v_va;
    }
    live->gdn_layers = gdn->layers;
    live->tail_bytes = gdn->tail_bytes;
    live->state_bytes = gdn->state_bytes;
    live->tail_va = gdn->tail_va;
    live->state_va = gdn->state_va;
    return OMEGA_Q36_PREFIX_OK;
}

static size_t gdn_bytes(const OmegaQ36PrefixLive *live, uint32_t per_layer) { return (size_t)live->gdn_layers * per_layer; }

size_t omega_q36_prefix_blob_bytes(const OmegaQ36PrefixLive *live, uint32_t pos) {
    if (!live || !pos || pos % OMEGA_Q36_PREFIX_POS_ALIGN || pos > live->max_pos || !live->kv_heads) return 0;
    if (!live->attn_layers || live->attn_layers > OMEGA_Q36_PREFIX_MAX_ATTN) return 0;
    size_t st = gdn_bytes(live, live->state_bytes), tl = gdn_bytes(live, live->tail_bytes);
    if (st % OMEGA_Q36_PREFIX_COPY_ALIGN || tl % OMEGA_Q36_PREFIX_COPY_ALIGN) return 0;
    return st + tl + (size_t)live->attn_layers * 2u * live->kv_heads * pos * OMEGA_Q36_PREFIX_ROW_BYTES;
}

/* Every copy of the layout, live <-> blob; to_blob picks the direction. rows: positions copied per head. */
static int copy_all(const OmegaQ36PrefixOps *ops, const OmegaQ36PrefixLive *live, uint64_t blob, uint32_t pos, uint32_t rows,
                    int to_blob) {
#define MOVE(live_va, blob_va, n)                                                                                    \
    do {                                                                                                             \
        if ((n) && ops->copy(ops->ctx, to_blob ? (live_va) : (blob_va), to_blob ? (blob_va) : (live_va), (n)))      \
            return OMEGA_Q36_PREFIX_COPY_FAILED;                                                                     \
    } while (0)
    size_t st = gdn_bytes(live, live->state_bytes), tl = gdn_bytes(live, live->tail_bytes);
    MOVE(live->state_va, blob, st);
    MOVE(live->tail_va, blob + st, tl);
    uint64_t at = blob + st + tl;
    size_t head = (size_t)pos * OMEGA_Q36_PREFIX_ROW_BYTES;
    size_t moved = (size_t)rows * OMEGA_Q36_PREFIX_ROW_BYTES;
    for (uint32_t l = 0; l < live->attn_layers; l++)
        for (uint32_t plane = 0; plane < 2; plane++) {
            uint64_t base = plane ? live->v_va[l] : live->k_va[l];
            for (uint32_t h = 0; h < live->kv_heads; h++, at += head)
                MOVE(base + (uint64_t)h * live->max_pos * OMEGA_Q36_PREFIX_ROW_BYTES, at, moved);
        }
#undef MOVE
    return OMEGA_Q36_PREFIX_OK;
}

static int same_shape(const OmegaQ36PrefixSnap *s, const OmegaQ36PrefixLive *live) {
    return s->attn_layers == live->attn_layers && s->kv_heads == live->kv_heads && s->gdn_layers == live->gdn_layers &&
           s->tail_bytes == live->tail_bytes && s->state_bytes == live->state_bytes;
}

int omega_q36_prefix_save(OmegaQ36PrefixPool *p, const OmegaQ36PrefixOps *ops, const OmegaQ36PrefixLive *live,
                          const uint8_t tag[OMEGA_Q36_PREFIX_TAG_BYTES], const uint8_t digest[OMEGA_Q36_PREFIX_TAG_BYTES],
                          uint32_t pos, OmegaQ36PrefixSnap *out) {
    if (!p || !ops || !ops->alloc || !ops->free || !ops->copy || !live || !tag || !digest || !out || out->blob_va)
        return OMEGA_Q36_PREFIX_BAD_ARGS;
    size_t bytes = omega_q36_prefix_blob_bytes(live, pos);
    if (!bytes) return OMEGA_Q36_PREFIX_BAD_ARGS;
    if (bytes > p->budget || p->resident > p->budget - bytes) return OMEGA_Q36_PREFIX_OVER_BUDGET;
    uint64_t va = 0;
    if (ops->alloc(ops->ctx, bytes, &va) || !va) return OMEGA_Q36_PREFIX_NO_MEMORY;
    int rc = copy_all(ops, live, va, pos, pos, 1);
    if (rc) {
        ops->free(ops->ctx, va, bytes);
        return rc;
    }
    memcpy(out->model_tag, tag, OMEGA_Q36_PREFIX_TAG_BYTES);
    memcpy(out->token_digest, digest, OMEGA_Q36_PREFIX_TAG_BYTES);
    out->pos = pos;
    out->attn_layers = live->attn_layers;
    out->kv_heads = live->kv_heads;
    out->gdn_layers = live->gdn_layers;
    out->tail_bytes = live->tail_bytes;
    out->state_bytes = live->state_bytes;
    out->blob_va = va;
    out->blob_bytes = bytes;
    out->last_use = ++p->tick;
    p->resident += bytes;
    p->snaps++;
    p->saved++;
    return OMEGA_Q36_PREFIX_OK;
}

static int check(const OmegaQ36PrefixSnap *s, const OmegaQ36PrefixLive *live, const uint8_t *tag, const uint8_t *digest,
                 uint32_t pos) {
    if (memcmp(s->model_tag, tag, OMEGA_Q36_PREFIX_TAG_BYTES)) return OMEGA_Q36_PREFIX_STALE_MODEL;
    if (memcmp(s->token_digest, digest, OMEGA_Q36_PREFIX_TAG_BYTES)) return OMEGA_Q36_PREFIX_TOKEN_MISMATCH;
    if (!same_shape(s, live)) return OMEGA_Q36_PREFIX_SHAPE_MISMATCH;
    if (pos != s->pos || pos > live->max_pos || s->blob_bytes != omega_q36_prefix_blob_bytes(live, pos))
        return OMEGA_Q36_PREFIX_LENGTH_MISMATCH;
    return OMEGA_Q36_PREFIX_OK;
}

int omega_q36_prefix_restore(OmegaQ36PrefixPool *p, const OmegaQ36PrefixOps *ops, OmegaQ36PrefixSnap *s,
                             const OmegaQ36PrefixLive *live, const uint8_t tag[OMEGA_Q36_PREFIX_TAG_BYTES],
                             const uint8_t digest[OMEGA_Q36_PREFIX_TAG_BYTES], uint32_t pos) {
    if (!p || !ops || !ops->copy || !s || !s->blob_va || !live || !tag || !digest) return OMEGA_Q36_PREFIX_BAD_ARGS;
    int rc = g_mutant == 2 ? OMEGA_Q36_PREFIX_OK : check(s, live, tag, digest, pos);
    if (rc) {
        p->refused++;
        return rc;
    }
    if (g_mutant == 2) pos = s->pos < live->max_pos ? s->pos : live->max_pos;
    uint32_t rows = g_mutant == 1 ? pos - OMEGA_Q36_PREFIX_POS_ALIGN : pos;
    rc = copy_all(ops, live, s->blob_va, s->pos, rows, 0);
    if (rc) return rc;
    s->last_use = ++p->tick;
    p->restored++;
    return OMEGA_Q36_PREFIX_OK;
}

void omega_q36_prefix_drop(OmegaQ36PrefixPool *p, const OmegaQ36PrefixOps *ops, OmegaQ36PrefixSnap *s) {
    if (!p || !ops || !ops->free || !s || !s->blob_va) return;
    ops->free(ops->ctx, s->blob_va, s->blob_bytes);
    p->resident -= s->blob_bytes;
    p->snaps--;
    memset(s, 0, sizeof *s);
}

int omega_q36_prefix_make_room(OmegaQ36PrefixPool *p, const OmegaQ36PrefixOps *ops, OmegaQ36PrefixSnap *snaps, uint32_t n,
                               size_t need) {
    if (!p || !ops || !ops->free || (n && !snaps)) return OMEGA_Q36_PREFIX_BAD_ARGS;
    if (need > p->budget) return OMEGA_Q36_PREFIX_OVER_BUDGET;
    uint64_t listed = 0;
    for (uint32_t i = 0; i < n; i++)
        if (snaps[i].blob_va) listed += snaps[i].blob_bytes;
    uint64_t room = p->budget - need;
    if (p->resident > room && p->resident - room > listed) return OMEGA_Q36_PREFIX_OVER_BUDGET;
    int dropped = 0;
    while (p->resident > room) {
        OmegaQ36PrefixSnap *old = NULL;
        for (uint32_t i = 0; i < n; i++)
            if (snaps[i].blob_va && (!old || snaps[i].last_use < old->last_use)) old = &snaps[i];
        if (!old) return OMEGA_Q36_PREFIX_OVER_BUDGET; /* unreachable: listed covers the gap */
        omega_q36_prefix_drop(p, ops, old);
        p->evicted++;
        dropped++;
    }
    return dropped;
}

const char *omega_q36_prefix_rc_name(int rc) {
    switch (rc) {
    case OMEGA_Q36_PREFIX_OK: return "OK";
    case OMEGA_Q36_PREFIX_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_Q36_PREFIX_STALE_MODEL: return "STALE_MODEL";
    case OMEGA_Q36_PREFIX_TOKEN_MISMATCH: return "TOKEN_MISMATCH";
    case OMEGA_Q36_PREFIX_LENGTH_MISMATCH: return "LENGTH_MISMATCH";
    case OMEGA_Q36_PREFIX_SHAPE_MISMATCH: return "SHAPE_MISMATCH";
    case OMEGA_Q36_PREFIX_OVER_BUDGET: return "OVER_BUDGET";
    case OMEGA_Q36_PREFIX_NO_MEMORY: return "NO_MEMORY";
    case OMEGA_Q36_PREFIX_COPY_FAILED: return "COPY_FAILED";
    default: return rc > 0 ? "DROPPED" : "UNKNOWN";
    }
}
