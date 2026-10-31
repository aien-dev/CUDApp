/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_GPU_Q36_PREFIX_H
#define OMEGA_GPU_Q36_PREFIX_H
/*
 * omega_gpu_q36_prefix: device-resident prefix snapshots of the hybrid Qwen3.6-35B (DRA-86, 2026-10-31).
 *
 * A snapshot is the whole per-conversation state after `pos` prompt tokens, copied VA -> VA into one blob:
 *   [gdn_layers][state_bytes]                      the DeltaNet recurrent states (omega_gpu_gdn_state_view state_va)
 *   [gdn_layers][tail_bytes]                       the conv tails (tail_va)
 *   [attn_layers][K, V][kv_heads][pos][512 B]      the attention K/V rows 0 .. pos - 1 of every head
 * The live K/V row (layer, head, p) is at k_va/v_va[layer] + (head * max_pos + p) * 512
 * (omega_gpu_decode_attn256_cache_view), so each (layer, plane, head) is one contiguous copy of pos * 512 bytes.
 *
 * The module owns no memory and opens no device: the caller passes OmegaQ36PrefixOps (alloc / free / copy). On the
 * chip, copy is omega_gpu_gdn_dev_copy (one KK_COPY launch; every length a multiple of 4096 bytes, which is why pos
 * must be a multiple of OMEGA_Q36_PREFIX_POS_ALIGN = 8 and every GDN block a multiple of 4096); the host test passes
 * an arena.
 *
 * Restore refuses before any copy, so a refused restore leaves the live state untouched (the caller reads cold):
 *   STALE_MODEL      the snapshot's model tag (caller-made: lib sha + weights + numeric env) is not the live one
 *   TOKEN_MISMATCH   the token digest (caller-made over the pos prompt tokens) is not the prompt's
 *   LENGTH_MISMATCH  pos is not the snapshot's (off by a turn), pos > max_pos, or the blob size is not the layout's
 *   SHAPE_MISMATCH   layers, heads, tail or state bytes differ from the live state
 * Save refuses OVER_BUDGET when resident + blob > budget; omega_q36_prefix_make_room evicts least-recently-used
 * snapshots first. The budget defaults to 16 GiB (AIEN_Q36_PREFIX_POOL_GB, the sovereign door's knob).
 *
 * Nothing calls this module on a serving path yet: the sovereign door is switched by AIEN_Q36_PREFIX=1 (default off)
 * and restores through host copies until omega.lock moves to an omega-mv commit carrying this file.
 */
#include <stddef.h>
#include <stdint.h>

#include "omega_gpu_decode_attn256_api.h"
#include "omega_gpu_gdn_api.h"

#define OMEGA_Q36_PREFIX_TAG_BYTES 32u
#define OMEGA_Q36_PREFIX_POS_ALIGN 8u
#define OMEGA_Q36_PREFIX_ROW_BYTES 512u    /* one bf16 K or V row, head_dim 256 */
#define OMEGA_Q36_PREFIX_COPY_ALIGN 4096u  /* omega_gpu_gdn_dev_copy */
#define OMEGA_Q36_PREFIX_MAX_ATTN 64u
#define OMEGA_Q36_PREFIX_DEFAULT_GB 16.0
#define OMEGA_Q36_PREFIX_POOL_ENV "AIEN_Q36_PREFIX_POOL_GB"

enum {
    OMEGA_Q36_PREFIX_OK = 0,
    OMEGA_Q36_PREFIX_BAD_ARGS = -1,
    OMEGA_Q36_PREFIX_STALE_MODEL = -2,
    OMEGA_Q36_PREFIX_TOKEN_MISMATCH = -3,
    OMEGA_Q36_PREFIX_LENGTH_MISMATCH = -4,
    OMEGA_Q36_PREFIX_SHAPE_MISMATCH = -5,
    OMEGA_Q36_PREFIX_OVER_BUDGET = -6,
    OMEGA_Q36_PREFIX_NO_MEMORY = -7,
    OMEGA_Q36_PREFIX_COPY_FAILED = -8,
};

/* The live per-conversation state: device addresses only. */
typedef struct {
    uint32_t attn_layers, kv_heads, max_pos;
    uint64_t k_va[OMEGA_Q36_PREFIX_MAX_ATTN], v_va[OMEGA_Q36_PREFIX_MAX_ATTN];
    uint32_t gdn_layers, tail_bytes, state_bytes;
    uint64_t tail_va, state_va;
} OmegaQ36PrefixLive;

typedef struct {
    int (*alloc)(void *ctx, size_t bytes, uint64_t *va); /* 0 = ok */
    void (*free)(void *ctx, uint64_t va, size_t bytes);
    int (*copy)(void *ctx, uint64_t src_va, uint64_t dst_va, size_t bytes); /* 0 = ok */
    void *ctx;
} OmegaQ36PrefixOps;

typedef struct {
    uint8_t model_tag[OMEGA_Q36_PREFIX_TAG_BYTES];
    uint8_t token_digest[OMEGA_Q36_PREFIX_TAG_BYTES];
    uint32_t pos;
    uint32_t attn_layers, kv_heads, gdn_layers, tail_bytes, state_bytes;
    uint64_t blob_va; /* 0 = empty slot */
    size_t blob_bytes;
    uint64_t last_use; /* pool tick of the last save or restore */
} OmegaQ36PrefixSnap;

typedef struct {
    uint64_t budget, resident; /* bytes */
    uint32_t snaps;
    uint64_t tick, saved, restored, refused, evicted;
} OmegaQ36PrefixPool;

/* AIEN_Q36_PREFIX_POOL_GB (unset or empty: 16) as bytes; a negative, non-finite or unparsable value: BAD_ARGS. */
int omega_q36_prefix_budget_from_env(uint64_t *bytes);
void omega_q36_prefix_pool_init(OmegaQ36PrefixPool *p, uint64_t budget);
/* The live state from the attention cache views (one per attention layer) and the GDN state view. */
int omega_q36_prefix_live_from_views(OmegaQ36PrefixLive *live, const OmegaGpuDecodeAttn256CacheView *attn, uint32_t attn_layers,
                                     const OmegaGpuGdnStateView *gdn);
/* Blob bytes for `pos` tokens of this live shape; 0 when pos or a block breaks the copy alignment. */
size_t omega_q36_prefix_blob_bytes(const OmegaQ36PrefixLive *live, uint32_t pos);
int omega_q36_prefix_save(OmegaQ36PrefixPool *p, const OmegaQ36PrefixOps *ops, const OmegaQ36PrefixLive *live,
                          const uint8_t tag[OMEGA_Q36_PREFIX_TAG_BYTES], const uint8_t digest[OMEGA_Q36_PREFIX_TAG_BYTES],
                          uint32_t pos, OmegaQ36PrefixSnap *out);
int omega_q36_prefix_restore(OmegaQ36PrefixPool *p, const OmegaQ36PrefixOps *ops, OmegaQ36PrefixSnap *s,
                             const OmegaQ36PrefixLive *live, const uint8_t tag[OMEGA_Q36_PREFIX_TAG_BYTES],
                             const uint8_t digest[OMEGA_Q36_PREFIX_TAG_BYTES], uint32_t pos);
void omega_q36_prefix_drop(OmegaQ36PrefixPool *p, const OmegaQ36PrefixOps *ops, OmegaQ36PrefixSnap *s);
/* Drop least-recently-used snapshots of snaps[0..n) until need more bytes fit; the count dropped, or OVER_BUDGET
 * (nothing dropped) when need alone exceeds the budget or the listed snapshots cannot free enough. */
int omega_q36_prefix_make_room(OmegaQ36PrefixPool *p, const OmegaQ36PrefixOps *ops, OmegaQ36PrefixSnap *snaps, uint32_t n,
                               size_t need);
const char *omega_q36_prefix_rc_name(int rc);
/* TEST ONLY: 1 = restore one 8-position block short at the boundary, 2 = restore skips the refusal checks. */
void omega_q36_prefix_test_set_mutant(int mutant);

#endif
