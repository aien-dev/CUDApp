/*
 * bw_encode_control_test: every opcode the encoder accepts must carry the control word the emitter scheduled
 * (BlackwellIRInsn.control: stall, yield, write / read scoreboards, wait mask) into the instruction's top word
 * (omega-mv#545, 2026-10-18). The encoder's golden-word checks compare words 0..2 only, so an opcode that wrote a
 * fixed word 3 kept its own wait mask and dropped the emitter's: IADD3 always waited on scoreboard 4 instead of the
 * Q loads' scoreboards, the next MOVs overwrote the loads' address registers while loads were still queued, and the
 * prefill attention faulted (Xid 13, Invalid Address Space) on the chip.
 *
 * Host only, no device. Exit 0 when every accepted opcode honours the control word.
 */
#include "omega_blackwell_codegen.h"
#include <stdio.h>
#include <string.h>

#define CTL_FIELD 0x03fffe00u /* stall 9..12, yield 13, wbar 14..16, rbar 17..19, wait 20..25 */

static const uint32_t sentinels[] = {
    (3u << 9) | (5u << 14) | (4u << 17) | (0x2au << 20),            /* stall 3, wbar 5, rbar 4, wait 0x2a */
    (12u << 9) | 0x2000u | (2u << 14) | (7u << 17) | (0x15u << 20), /* stall 12, yield, wbar 2, rbar none, wait 0x15 */
};

int main(void) {
    int checked = 0, refused = 0, bad = 0;
    for (int op = 0; op <= (int)BW_IR_MUFU_LG2; op++) {
        for (int form = 0; form < 4; form++) {
            const int pred = form & 1, uni = form >> 1;
            int accepted_any = 0;
            for (unsigned s = 0; s < sizeof sentinels / sizeof sentinels[0]; s++) {
                BlackwellIRInsn in;
                memset(&in, 0, sizeof in);
                in.op = (BlackwellIROpcode)op;
                in.dst_vreg = 4; in.src1_vreg = 6; in.src2_vreg = 8; in.src3_vreg = 10; in.ureg = 4;
                in.imm = 2; in.predicate_p0 = pred != 0; in.is_uniform = uni != 0;
                in.control = sentinels[s];
                uint32_t w[4] = { 0, 0, 0, 0 };
                if (omega_blackwell_encode_one(&in, w) != 0) continue;
                accepted_any = 1;
                checked++;
                if ((w[3] & CTL_FIELD) != (sentinels[s] & CTL_FIELD)) {
                    bad++;
                    printf("FAIL: op %d%s%s: control 0x%08x encoded as 0x%08x\n", op, pred ? " (@P0)" : "", uni ? " (uniform)" : "", sentinels[s], w[3]);
                }
            }
            if (!accepted_any && form == 0) refused++;
        }
    }
    printf("bw_encode_control_test: %d encodings checked, %d opcodes refused (no operands), %d dropped the control word: %s\n",
           checked, refused, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
