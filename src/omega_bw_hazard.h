/* omega_bw_hazard.h: static scoreboard hazard checker over encoded sm_121 kernel bytes (omega-mv#554).
 *
 * The warp simulators run the IR and model no scoreboards, so a dropped wait mask passes every host test and only
 * shows on the chip (omega-mv#545: the encoder wrote a fixed word 3 for IADD3, the wait on the Q loads' read barrier
 * was lost and a MOV zeroed the address of 16 queued LDG.E.128; Xid 13). This checker reads the bytes the chip runs,
 * after every encode and patch step, and walks the control flow with the scoreboard rules:
 *
 *   RAW  a reader of a register written by a variable-latency op must have waited on that op's write barrier;
 *   WAR  a writer of a register still being read by a queued load or store must have waited on its read barrier
 *        (or on its write barrier: an op that has completed has read its sources);
 *   WAW  a writer of a register a variable-latency op has not yet written must have waited on that op's write
 *        barrier (the late result would land over the new value).
 *
 * Control word (word 3 of each 16-byte instruction, the same layout as CW() in omega_bw_em.h): stall bits 9..12,
 * yield 13, write barrier 14..16, read barrier 17..19 (7 = none), wait mask 20..25. A wait on barrier n releases every
 * register any outstanding op on n holds (the barrier is a counter that reaches zero). The analysis is a may-analysis
 * over all paths (branches, predicated EXIT/BRA, loops) to a fixed point; a predicated instruction may or may not
 * write, and its wait mask is taken as applied (UNKNOWN whether the chip applies the wait of a predicated-off
 * instruction; if it does not, this checker can miss a hazard behind such an instruction).
 *
 * Limits, said out loud: fixed-latency hazards (stall counts too short) are not checked; predicate registers are not
 * tracked; HMMA/QMMA are tracked only through the barriers their bytes declare. NVIDIA documents none of the control
 * field; the layout is our own decode, cross-checked against nvdisasm on every kernel (tests/bw_hazard_test.c). */
#ifndef OMEGA_BW_HAZARD_H
#define OMEGA_BW_HAZARD_H

#include <stddef.h>
#include <stdint.h>

#define OMEGA_BW_RZ 0xffu            /* register 255 is RZ, uniform 255 is URZ: never tracked */
#define OMEGA_BW_NOBAR 7u

enum { OMEGA_BW_FLOW_NEXT = 0, OMEGA_BW_FLOW_EXIT, OMEGA_BW_FLOW_BRA };

typedef struct {
    uint8_t reg;     /* first register */
    uint8_t n;       /* 1, 2 or 4 consecutive registers */
    uint8_t uniform; /* 1 = uniform register file */
} OmegaBwOperand;

typedef struct {
    uint16_t op;                         /* word 0 bits 0..11 */
    uint8_t guard;                       /* word 0 bits 12..15: 7 = always (PT) */
    uint8_t stall, yield, wbar, rbar, wait;
    uint8_t var;                         /* variable-latency writer: its result needs a write barrier */
    uint8_t late;                        /* reads its sources late (loads, stores, atomics, constant loads) */
    uint8_t nr, nw;
    OmegaBwOperand rd[5], wr[1];
    uint8_t flow;                        /* OMEGA_BW_FLOW_* */
    int32_t target;                      /* FLOW_BRA: target instruction index relative to this one */
    const char *name;
} OmegaBwInsn;

/* Decodes one 16-byte instruction. 0 ok, -1 an opcode or form this decoder does not know (the kernel is then
 * reported as not checked, never as clean). */
int omega_bw_decode(const uint8_t b[16], OmegaBwInsn *d);

enum { OMEGA_BW_HZ_RAW = 1, OMEGA_BW_HZ_WAR = 2, OMEGA_BW_HZ_WAW = 3 };

typedef struct {
    uint32_t pc;       /* instruction index of the offending reader / writer */
    uint32_t src_pc;   /* the last producer / queued reader of the register seen on some path (a hint) */
    uint8_t kind;      /* OMEGA_BW_HZ_* */
    uint8_t reg;
    uint8_t uniform;
    uint8_t release;   /* barrier mask whose wait would have released it; 0 = the op declared no barrier at all */
} OmegaBwHazard;

#define OMEGA_BW_HAZARD_MAX 256

typedef struct {
    size_t insns;          /* instructions in the buffer */
    size_t reachable;      /* reachable from instruction 0 */
    size_t count;          /* findings (the first OMEGA_BW_HAZARD_MAX are kept) */
    OmegaBwHazard f[OMEGA_BW_HAZARD_MAX];
    int64_t bad_pc;        /* -1, or the first instruction that could not be decoded or branches out of range */
} OmegaBwHazardReport;

/* Checks code[0..len). 0 = decoded and walked (r->count findings, possibly 0); -1 = not checked (bad_pc set, or a
 * NULL / misaligned argument, or out of memory). Host only; allocates its work state. */
int omega_bw_hazard_check(const uint8_t *code, size_t len, OmegaBwHazardReport *r);

const char *omega_bw_hazard_kind(uint8_t kind);

#endif /* OMEGA_BW_HAZARD_H */
