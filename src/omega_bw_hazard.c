/* omega_bw_hazard.c: static scoreboard hazard checker over encoded sm_121 kernel bytes (omega-mv#554). See the header.
 *
 * Every form decoded below is a form src/omega_blackwell_codegen.c (encode_single_insn) or a post-encode patch
 * (omega_tc_patch_code, omega_qmma_patch_code) writes. The operand positions are the encoder's: Rd = word 0 bits 16..23,
 * Ra = word 0 bits 24..31, Rb = word 1 bits 0..7, Rc = word 2 bits 0..7. tests/bw_hazard_test.c checks the decoder
 * against the encoder for every IR op and against nvdisasm on every kernel. */
#include "omega_bw_hazard.h"

#include <stdlib.h>
#include <string.h>

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }

/* registers of a memory access of size field (word 2 bits 9..11): 0..4 one register, 5 a pair (.64), 6 a quad (.128) */
static int size_regs(uint32_t w2) {
    unsigned s = (w2 >> 9) & 7u;
    return s <= 4 ? 1 : s == 5 ? 2 : s == 6 ? 4 : -1;
}

static void R(OmegaBwInsn *d, unsigned reg, int n, int uniform) {
    if (reg == OMEGA_BW_RZ || n <= 0 || d->nr >= 5) return;
    d->rd[d->nr].reg = (uint8_t)reg; d->rd[d->nr].n = (uint8_t)n; d->rd[d->nr].uniform = (uint8_t)uniform; d->nr++;
}
static void W(OmegaBwInsn *d, unsigned reg, int n, int uniform) {
    if (reg == OMEGA_BW_RZ || n <= 0 || d->nw >= 1) return;
    d->wr[0].reg = (uint8_t)reg; d->wr[0].n = (uint8_t)n; d->wr[0].uniform = (uint8_t)uniform; d->nw = 1;
}

int omega_bw_decode(const uint8_t b[16], OmegaBwInsn *d) {
    if (!b || !d) return -1;
    uint32_t w0 = rd32(b), w1 = rd32(b + 4), w2 = rd32(b + 8), w3 = rd32(b + 12);
    memset(d, 0, sizeof *d);
    d->op = (uint16_t)(w0 & 0xfffu);
    d->guard = (uint8_t)((w0 >> 12) & 0xfu);
    d->stall = (uint8_t)((w3 >> 9) & 0xfu);
    d->yield = (uint8_t)((w3 >> 13) & 1u);
    d->wbar = (uint8_t)((w3 >> 14) & 7u);
    d->rbar = (uint8_t)((w3 >> 17) & 7u);
    d->wait = (uint8_t)((w3 >> 20) & 0x3fu);
    unsigned D = (w0 >> 16) & 0xffu, A = w0 >> 24, B = w1 & 0xffu, C = w2 & 0xffu;
    int n;
    switch (d->op) {
    case 0x918: d->name = "NOP"; break;
    case 0xb1d: d->name = "BAR"; break;
    case 0x992: d->name = "MEMBAR"; break;
    case 0x98f: d->name = "CCTL"; R(d, A, 1, 0); d->late = 1; break;
    case 0x94d: d->name = "EXIT"; d->flow = OMEGA_BW_FLOW_EXIT; break;
    case 0x947: {
        d->name = "BRA"; d->flow = OMEGA_BW_FLOW_BRA;
        int32_t rel = (int32_t)((uint32_t)((int32_t)w1 >> 2) << 6) | (int32_t)((w0 >> 18) & 0x3fu);
        d->target = rel + 1;
        break;
    }
    case 0x945: d->name = "BSSY"; break;
    case 0x941: d->name = "BSYNC"; break;
    case 0x202: d->name = "MOV"; W(d, D, 1, 0); R(d, B, 1, 0); break;
    case 0x802: d->name = "MOV"; W(d, D, 1, 0); break;
    case 0x919: d->name = "S2R"; W(d, D, 1, 0); d->var = 1; break;
    case 0xb82:
        d->name = "LDC"; if ((n = size_regs(w2)) < 0) return -1;
        W(d, D, n, 0); R(d, A, 1, 0); d->var = 1; d->late = 1; break;
    case 0x7ac:
        d->name = "LDCU"; if ((n = size_regs(w2)) < 0) return -1;
        W(d, D, n, 1); R(d, A, 1, 1); d->var = 1; d->late = 1; break;
    case 0x224: d->name = "IMAD"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 0); R(d, C, 1, 0); break;
    case 0x824: d->name = "IMAD"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, C, 1, 0); break;
    case 0xc24: d->name = "IMAD"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 1); R(d, C, 1, 0); break;
    case 0x825: d->name = "IMAD.WIDE"; W(d, D, 2, 0); R(d, A, 1, 0); R(d, C, 2, 0); break;
    case 0x227: d->name = "IMAD.HI"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 0); R(d, C, 1, 0); break;
    case 0x20c: d->name = "ISETP"; R(d, A, 1, 0); R(d, B, 1, 0); break;
    case 0x80c: d->name = "ISETP"; R(d, A, 1, 0); break;
    case 0xc0c: d->name = "ISETP"; R(d, A, 1, 0); R(d, B, 1, 1); break;
    case 0x210: d->name = "IADD3"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 0); R(d, C, 1, 0); break;
    case 0x23c: d->name = "HMMA"; W(d, D, 4, 0); R(d, A, 4, 0); R(d, B, 2, 0); R(d, C, 4, 0); break;
    case 0x27a: d->name = "QMMA"; W(d, D, 4, 0); R(d, A, 4, 0); R(d, B, 2, 0); R(d, C, 4, 0); break;
    case 0x819: d->name = "SHF"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, C, 1, 0); break;
    case 0x219: d->name = "SHF"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 0); R(d, C, 1, 0); break;
    case 0x812: d->name = "LOP3"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, C, 1, 0); break;
    case 0x212: d->name = "LOP3"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 0); R(d, C, 1, 0); break;
    case 0x221: d->name = "FADD"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 0); break;
    case 0x220: d->name = "FMUL"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 0); break;
    case 0x223: d->name = "FFMA"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 0); R(d, C, 1, 0); break;
    case 0x20b: d->name = "FSETP"; R(d, A, 1, 0); R(d, B, 1, 0); break;
    case 0x208: d->name = "FSEL"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 0); break;
    case 0x209: d->name = "FMNMX"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 0); break;
    case 0x23e: d->name = "F2FP"; W(d, D, 1, 0); R(d, A, 1, 0); R(d, B, 1, 0); break;
    case 0x245: d->name = "I2FP"; W(d, D, 1, 0); R(d, B, 1, 0); break;
    case 0x305: d->name = "F2I"; W(d, D, 1, 0); R(d, B, 1, 0); d->var = 1; break;
    case 0x308: d->name = "MUFU"; W(d, D, 1, 0); R(d, B, 1, 0); d->var = 1; break;
    case 0xf89: d->name = "SHFL"; W(d, D, 1, 0); R(d, A, 1, 0); d->var = 1; break;
    case 0x981: /* LDG Rd, desc[URb][Ra.64+imm]: word 2 bit 8 = 64-bit address, bit 27 = descriptor */
        d->name = "LDG"; if ((n = size_regs(w2)) < 0) return -1;
        W(d, D, n, 0); R(d, A, (w2 & 0x100u) ? 2 : 1, 0); if (w2 & 0x08000000u) R(d, B, 1, 1);
        d->var = 1; d->late = 1; break;
    case 0x986: /* STG desc[URc][Ra.64+imm], Rb */
        d->name = "STG"; if ((n = size_regs(w2)) < 0) return -1;
        R(d, A, (w2 & 0x100u) ? 2 : 1, 0); R(d, B, n, 0); if (w2 & 0x08000000u) R(d, C, 1, 1);
        d->late = 1; break;
    case 0x9a8: /* ATOMG desc[URc][Ra.64], Rb -> Rd */
        d->name = "ATOMG"; W(d, D, 1, 0); R(d, A, (w2 & 0x100u) ? 2 : 1, 0); R(d, B, 1, 0);
        if (w2 & 0x08000000u) R(d, C, 1, 1);
        d->var = 1; d->late = 1; break;
    case 0x984: /* LDS Rd, [Ra+URb+imm]: word 2 bit 27 = uniform offset */
        d->name = "LDS"; if ((n = size_regs(w2)) < 0) return -1;
        W(d, D, n, 0); R(d, A, 1, 0); if (w2 & 0x08000000u) R(d, B, 1, 1);
        d->var = 1; d->late = 1; break;
    case 0x388: case 0x988: /* STS [Ra+URc], Rb */
        d->name = "STS"; if ((n = size_regs(w2)) < 0) return -1;
        R(d, A, 1, 0); R(d, B, n, 0); if (w2 & 0x08000000u) R(d, C, 1, 1);
        d->late = 1; break;
    default:
        return -1;
    }
    return 0;
}

const char *omega_bw_hazard_kind(uint8_t kind) {
    return kind == OMEGA_BW_HZ_RAW ? "RAW" : kind == OMEGA_BW_HZ_WAR ? "WAR" : kind == OMEGA_BW_HZ_WAW ? "WAW" : "?";
}

/* ---- the walk ---------------------------------------------------------------------------------------------------
 * State per register: two 64-bit sets of "release masks". An outstanding op is entered as the bit (1 << m), m the
 * 6-bit set of barriers whose wait releases it (m = 0: the op declared none, nothing releases it). A wait mask W
 * releases every entry m with m & W != 0, so it clears kill(W). Union over paths is OR. */
#define NREG 512 /* 256 general + 256 uniform */

typedef struct {
    uint64_t w[NREG];   /* pending writes (RAW / WAW) */
    uint64_t r[NREG];   /* pending late reads (WAR) */
    uint32_t wpc[NREG]; /* last producer / queued reader seen (a hint for the report) */
    uint32_t rpc[NREG];
} State;

static uint64_t kill_mask(unsigned wait) {
    uint64_t k = 0;
    for (unsigned m = 1; m < 64; m++)
        if (m & wait) k |= 1ull << m;
    return k;
}

static void note(OmegaBwHazardReport *r, uint32_t pc, uint32_t src, uint8_t kind, unsigned idx, uint64_t pending) {
    unsigned rel = 0;
    for (unsigned m = 0; m < 64; m++)
        if (pending >> m & 1u) { rel = m; break; } /* the smallest mask: 0 when an op declared no barrier */
    for (size_t i = 0; i < r->count && i < OMEGA_BW_HAZARD_MAX; i++)
        if (r->f[i].pc == pc && r->f[i].kind == kind && r->f[i].reg == (uint8_t)idx && r->f[i].uniform == (idx >= 256))
            return;
    if (r->count < OMEGA_BW_HAZARD_MAX) {
        OmegaBwHazard *h = &r->f[r->count];
        h->pc = pc; h->src_pc = src; h->kind = kind; h->reg = (uint8_t)idx; h->uniform = (uint8_t)(idx >= 256); h->release = (uint8_t)rel;
    }
    r->count++;
}

/* one instruction: IN state s -> OUT state s (in place). rep != NULL records the findings. */
static void step(const OmegaBwInsn *d, uint32_t pc, State *s, OmegaBwHazardReport *rep) {
    if (d->wait) {
        uint64_t k = ~kill_mask(d->wait);
        for (unsigned i = 0; i < NREG; i++) { s->w[i] &= k; s->r[i] &= k; }
    }
    /* reads: RAW against pending writes; the first reader is reported and the entry dropped (no cascade) */
    for (unsigned j = 0; j < d->nr; j++)
        for (unsigned q = 0; q < d->rd[j].n; q++) {
            unsigned i = (d->rd[j].uniform ? 256u : 0u) + ((d->rd[j].reg + q) & 0xffu);
            if ((i & 0xffu) == OMEGA_BW_RZ) continue;
            if (s->w[i]) { if (rep) note(rep, pc, s->wpc[i], OMEGA_BW_HZ_RAW, i, s->w[i]); s->w[i] = 0; }
        }
    /* writes: WAR against queued late readers, WAW against pending writes */
    for (unsigned j = 0; j < d->nw; j++)
        for (unsigned q = 0; q < d->wr[j].n; q++) {
            unsigned i = (d->wr[j].uniform ? 256u : 0u) + ((d->wr[j].reg + q) & 0xffu);
            if ((i & 0xffu) == OMEGA_BW_RZ) continue;
            if (s->r[i]) { if (rep) note(rep, pc, s->rpc[i], OMEGA_BW_HZ_WAR, i, s->r[i]); s->r[i] = 0; }
            if (s->w[i]) { if (rep) note(rep, pc, s->wpc[i], OMEGA_BW_HZ_WAW, i, s->w[i]); s->w[i] = 0; }
            int has_w = d->wbar != OMEGA_BW_NOBAR;
            if (has_w || d->var) { s->w[i] |= 1ull << (has_w ? (1u << d->wbar) : 0u); s->wpc[i] = pc; }
        }
    /* late source reads: held until a wait on the read barrier or on the write barrier (completion) */
    if (d->late || d->rbar != OMEGA_BW_NOBAR) {
        unsigned m = (d->rbar != OMEGA_BW_NOBAR ? 1u << d->rbar : 0u) | (d->wbar != OMEGA_BW_NOBAR ? 1u << d->wbar : 0u);
        for (unsigned j = 0; j < d->nr; j++)
            for (unsigned q = 0; q < d->rd[j].n; q++) {
                unsigned i = (d->rd[j].uniform ? 256u : 0u) + ((d->rd[j].reg + q) & 0xffu);
                if ((i & 0xffu) == OMEGA_BW_RZ) continue;
                s->r[i] |= 1ull << m; s->rpc[i] = pc;
            }
    }
}

/* OR src into dst; 1 if dst changed */
static int join(State *dst, const State *src) {
    int ch = 0;
    for (unsigned i = 0; i < NREG; i++) {
        uint64_t w = dst->w[i] | src->w[i], r = dst->r[i] | src->r[i];
        if (w != dst->w[i]) { dst->w[i] = w; dst->wpc[i] = src->wpc[i]; ch = 1; }
        if (r != dst->r[i]) { dst->r[i] = r; dst->rpc[i] = src->rpc[i]; ch = 1; }
    }
    return ch;
}

int omega_bw_hazard_check(const uint8_t *code, size_t len, OmegaBwHazardReport *r) {
    if (!r) return -1;
    memset(r, 0, sizeof *r);
    r->bad_pc = -1;
    if (!code || len == 0 || len % 16u) return -1;
    size_t n = len / 16u;
    r->insns = n;
    OmegaBwInsn *ins = calloc(n, sizeof *ins);
    State *in = calloc(n, sizeof *in);
    uint8_t *seen = calloc(n, 1), *queued = calloc(n, 1);
    uint32_t *work = calloc(n, sizeof *work);
    State *cur = malloc(sizeof *cur);
    int rc = -1;
    if (!ins || !in || !seen || !queued || !work || !cur) goto out;

    /* decode what is reachable from 0 (padding after the final EXIT is never decoded) */
    {
        size_t top = 0;
        work[top++] = 0; queued[0] = 1;
        while (top) {
            uint32_t pc = work[--top];
            if (omega_bw_decode(code + (size_t)pc * 16u, &ins[pc]) != 0) { r->bad_pc = pc; goto out; }
            seen[pc] = 1;
            const OmegaBwInsn *d = &ins[pc];
            int64_t succ[2]; int ns = 0;
            if (d->flow == OMEGA_BW_FLOW_NEXT || d->guard != 7u) succ[ns++] = (int64_t)pc + 1;
            if (d->flow == OMEGA_BW_FLOW_BRA) succ[ns++] = (int64_t)pc + d->target;
            for (int k = 0; k < ns; k++) {
                if (succ[k] < 0 || succ[k] >= (int64_t)n) { r->bad_pc = pc; goto out; }
                if (!queued[succ[k]]) { queued[succ[k]] = 1; work[top++] = (uint32_t)succ[k]; }
            }
        }
    }
    for (size_t i = 0; i < n; i++) r->reachable += seen[i];

    /* fixed point: IN states only grow (OR), the lattice is finite. Every reachable instruction is stepped at least
     * once (seen is reused as "not yet stepped"), later only when its IN state grew. */
    memset(queued, 0, n);
    {
        size_t top = 0;
        work[top++] = 0; queued[0] = 1;
        while (top) {
            uint32_t pc = work[--top];
            queued[pc] = 0; seen[pc] = 2;
            *cur = in[pc];
            step(&ins[pc], pc, cur, NULL);
            const OmegaBwInsn *d = &ins[pc];
            int64_t succ[2]; int ns = 0;
            if (d->flow == OMEGA_BW_FLOW_NEXT || d->guard != 7u) succ[ns++] = (int64_t)pc + 1;
            if (d->flow == OMEGA_BW_FLOW_BRA) succ[ns++] = (int64_t)pc + d->target;
            for (int k = 0; k < ns; k++) {
                uint32_t t = (uint32_t)succ[k];
                if ((join(&in[t], cur) || seen[t] == 1) && !queued[t]) { queued[t] = 1; work[top++] = t; }
            }
        }
    }
    /* record: one pass over the reachable instructions with their fixed-point IN states */
    for (size_t pc = 0; pc < n; pc++) {
        if (!seen[pc]) continue;
        *cur = in[pc];
        step(&ins[pc], (uint32_t)pc, cur, r);
    }
    rc = 0;
out:
    free(ins); free(in); free(seen); free(queued); free(work); free(cur);
    return rc;
}
