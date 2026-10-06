#ifndef OMEGA_BW_RECONV_H
#define OMEGA_BW_RECONV_H
/* Structured warp reconvergence regions for the Blackwell IR emitters (omega #308).
 *
 * A region is the span between a BSSY (arm barrier Bn, name the join point) and the
 * matching BSYNC (wait for every armed lane, continue as one warp). Inside a region the
 * lanes of a warp may take different paths: a predicated forward branch, a loop whose
 * trip count differs per lane, a break. Every path that leaves the region jumps to the
 * BSYNC itself (that is where nvcc 13.0.88 sends loop exits on sm_121); the BSSY names the
 * instruction after the BSYNC. Regions nest; the innermost open region owns Bn with
 * n = depth, so nesting depth is bounded by the barrier register count and begin fails
 * closed (returns -1) when that bound is reached.
 *
 * This header holds the bookkeeping only (indices, patching, bounds). Each emitter wraps
 * it with its own control-word scheme: see region_begin / region_exit / region_join in
 * src/omega_gpu_elementwise_api.c and src/omega_gpu_attention_api.c.
 *
 * Reference: Mesa NAK nak_nir_lower_cf.c (push_scope / pop_scope: bar_set_nv before a
 * divergent if or loop, bar_sync_nv at its merge block, breaks jump to the sync) and the
 * nvcc oracle listing in docs/gpu-reconvergence-308.md. */

#include "omega_blackwell_codegen.h"

#define BW_RECONV_MAX_DEPTH BW_RECONV_MAX_BAR
#define BW_RECONV_MAX_EXITS 8 /* forward exits (breaks) recorded per region */

typedef struct {
    int depth;                      /* open regions */
    int peak;                       /* deepest nesting seen (reported in receipts) */
    int opened;                     /* regions begun (for receipts) */
    int bssy_at[BW_RECONV_MAX_DEPTH];
    int exits[BW_RECONV_MAX_DEPTH][BW_RECONV_MAX_EXITS];
    int n_exits[BW_RECONV_MAX_DEPTH];
} BwRegions;

/* Called after the emitter appended the BSSY at index `bssy_index`. Returns the barrier
 * register to use, or -1 when the nesting bound is reached (the caller must fail the
 * build: no code with a missing barrier may be emitted). The caller allocates the
 * barrier BEFORE emitting: use bw_regions_next_bar to pick it, then bw_regions_begin. */
static inline int bw_regions_next_bar(const BwRegions *r) {
    return r->depth < BW_RECONV_MAX_DEPTH ? r->depth : -1;
}
static inline int bw_regions_begin(BwRegions *r, int bssy_index) {
    if (r->depth >= BW_RECONV_MAX_DEPTH || bssy_index < 0) return -1;
    r->bssy_at[r->depth] = bssy_index;
    r->n_exits[r->depth] = 0;
    r->depth++;
    r->opened++;
    if (r->depth > r->peak) r->peak = r->depth;
    return r->depth - 1;
}

/* Records a forward branch (already appended at `bra_index`) that leaves the innermost
 * open region; its target is patched to the BSYNC by bw_regions_join. -1 when no region
 * is open or the per-region exit table is full. */
static inline int bw_regions_exit(BwRegions *r, int bra_index) {
    if (r->depth <= 0 || bra_index < 0) return -1;
    int d = r->depth - 1;
    if (r->n_exits[d] >= BW_RECONV_MAX_EXITS) return -1;
    r->exits[d][r->n_exits[d]++] = bra_index;
    return 0;
}

/* Called after the emitter appended the BSYNC at index `bsync_index` for the innermost
 * open region. Patches the BSSY to name the instruction after the BSYNC and every
 * recorded exit to jump to the BSYNC. Returns the barrier register closed, or -1 when
 * no region is open or an index is inconsistent (BSSY must precede the BSYNC). */
static inline int bw_regions_join(BwRegions *r, BlackwellIRProgram *p, int bsync_index) {
    if (r->depth <= 0 || !p || bsync_index < 0 || (size_t)bsync_index >= p->count) return -1;
    int d = --r->depth;
    int at = r->bssy_at[d];
    if (at < 0 || at >= bsync_index || p->insns[at].op != BW_IR_BSSY || p->insns[bsync_index].op != BW_IR_BSYNC) return -1;
    if (p->insns[at].bar_reg != (uint8_t)d || p->insns[bsync_index].bar_reg != (uint8_t)d) return -1;
    p->insns[at].imm = (uint32_t)(bsync_index + 1 - at);
    for (int i = 0; i < r->n_exits[d]; i++) {
        int b = r->exits[d][i];
        if (b <= at || b >= bsync_index || p->insns[b].op != BW_IR_BRA) return -1;
        p->insns[b].imm = (uint32_t)(bsync_index - b);
    }
    return d;
}

#endif /* OMEGA_BW_RECONV_H */
