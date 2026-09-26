#include "omega_blackwell_encoder.h"
#include "sha256.h"
#include <stdio.h>
#include <string.h>

/*
 * Bit-for-bit verified sm_121 instruction sequence for unsigned 32-bit vector addition.
 * Emits 32 instructions (512 bytes), aligned to 128 bytes.
 */
static const BlackwellInstruction VECADD_INSTRUCTIONS[OMEGA_BW_VECADD_INSN_COUNT] = {
    /* 0000 */ { { 0xff017b82, 0x0000df00, 0x00000800, 0x000fe200 }, "LDC R1, c[0x0][0x37c]" },
    /* 0010 */ { { 0x00097919, 0x00000000, 0x00002500, 0x000e2200 }, "S2R R9, SR_CTAID.X" },
    /* 0020 */ { { 0xff0477ac, 0x00006c00, 0x08000800, 0x000e2200 }, "LDCU UR4, c[0x0][0x360]" },
    /* 0030 */ { { 0x00007919, 0x00000000, 0x00002100, 0x000e2200 }, "S2R R0, SR_TID.X" },
    /* 0040 */ { { 0xff0577ac, 0x00007300, 0x08000800, 0x000e6200 }, "LDCU UR5, c[0x0][0x398]" },
    /* 0050 */ { { 0x09097c24, 0x00000004, 0x0f8e0200, 0x001fca00 }, "IMAD R9, R9, UR4, R0" },
    /* 0060 */ { { 0x09007c0c, 0x00000005, 0x0bf06070, 0x002fda00 }, "ISETP.GE.U32.AND P0, PT, R9, UR5, PT" },
    /* 0070 */ { { 0x0000094d, 0x00000000, 0x03800000, 0x000fea00 }, "@P0 EXIT" },
    /* 0080 */ { { 0xff027b82, 0x0000e000, 0x00000a00, 0x000e2200 }, "LDC.64 R2, c[0x0][0x380]" },
    /* 0090 */ { { 0xff0477ac, 0x00006b00, 0x08000a00, 0x000e6e00 }, "LDCU.64 UR4, c[0x0][0x358]" },
    /* 00a0 */ { { 0xff047b82, 0x0000e200, 0x00000a00, 0x000eb000 }, "LDC.64 R4, c[0x0][0x388]" },
    /* 00b0 */ { { 0xff067b82, 0x0000e400, 0x00000a00, 0x000ee200 }, "LDC.64 R6, c[0x0][0x390]" },
    /* 00c0 */ { { 0x09027825, 0x00000004, 0x078e0002, 0x001fcc00 }, "IMAD.WIDE.U32 R2, R9, 0x4, R2" },
    /* 00d0 */ { { 0x02027981, 0x00000004, 0x0c1e1900, 0x002f2200 }, "LDG.E R2, desc[UR4][R2.64]" },
    /* 00e0 */ { { 0x09047825, 0x00000004, 0x078e0004, 0x004fcc00 }, "IMAD.WIDE.U32 R4, R9, 0x4, R4" },
    /* 00f0 */ { { 0x04057981, 0x00000004, 0x0c1e1900, 0x000f2200 }, "LDG.E R5, desc[UR4][R4.64]" },
    /* 0100 */ { { 0x09067825, 0x00000004, 0x078e0006, 0x008fe200 }, "IMAD.WIDE.U32 R6, R9, 0x4, R6" },
    /* 0110 */ { { 0x02097210, 0x00000005, 0x07ffe0ff, 0x010fca00 }, "IADD3 R9, PT, PT, R2, R5, RZ" },
    /* 0120 */ { { 0x06007986, 0x00000009, 0x0c101904, 0x000fe200 }, "STG.E desc[UR4][R6.64], R9" },
    /* 0130 */ { { 0x0000794d, 0x00000000, 0x03800000, 0x000fea00 }, "EXIT" },
    /* 0140 */ { { 0x00fc7947, 0xfffffffc, 0x0383ffff, 0x000fc000 }, "BRA (.L_x_0)" },
    /* 0150 */ { { 0x00007918, 0x00000000, 0x00000000, 0x000fc000 }, "NOP" },
    /* 0160 */ { { 0x00007918, 0x00000000, 0x00000000, 0x000fc000 }, "NOP" },
    /* 0170 */ { { 0x00007918, 0x00000000, 0x00000000, 0x000fc000 }, "NOP" },
    /* 0180 */ { { 0x00007918, 0x00000000, 0x00000000, 0x000fc000 }, "NOP" },
    /* 0190 */ { { 0x00007918, 0x00000000, 0x00000000, 0x000fc000 }, "NOP" },
    /* 01a0 */ { { 0x00007918, 0x00000000, 0x00000000, 0x000fc000 }, "NOP" },
    /* 01b0 */ { { 0x00007918, 0x00000000, 0x00000000, 0x000fc000 }, "NOP" },
    /* 01c0 */ { { 0x00007918, 0x00000000, 0x00000000, 0x000fc000 }, "NOP" },
    /* 01d0 */ { { 0x00007918, 0x00000000, 0x00000000, 0x000fc000 }, "NOP" },
    /* 01e0 */ { { 0x00007918, 0x00000000, 0x00000000, 0x000fc000 }, "NOP" },
    /* 01f0 */ { { 0x00007918, 0x00000000, 0x00000000, 0x000fc000 }, "NOP" },
};

int omega_blackwell_encode_vecadd(uint8_t *code_buf, size_t max_len, size_t *out_len) {
    if (!code_buf || max_len < OMEGA_BW_VECADD_CODE_SIZE) return -1;

    uint8_t *p = code_buf;
    for (size_t i = 0; i < OMEGA_BW_VECADD_INSN_COUNT; i++) {
        for (int w = 0; w < 4; w++) {
            uint32_t val = VECADD_INSTRUCTIONS[i].w[w];
            p[0] = (uint8_t)(val & 0xFF);
            p[1] = (uint8_t)((val >> 8) & 0xFF);
            p[2] = (uint8_t)((val >> 16) & 0xFF);
            p[3] = (uint8_t)((val >> 24) & 0xFF);
            p += 4;
        }
    }

    if (out_len) *out_len = OMEGA_BW_VECADD_CODE_SIZE;
    return 0;
}

int omega_blackwell_compute_code_digest(const uint8_t *code_buf, size_t len, uint8_t *out_digest) {
    if (!code_buf || len == 0 || !out_digest) return -1;
    sha256_ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, code_buf, len);
    sha256_final(&ctx, out_digest);
    return 0;
}

int omega_blackwell_verify_encoder_fixtures(void) {
    uint8_t code_buf[OMEGA_BW_VECADD_CODE_SIZE];
    size_t out_len = 0;
    if (omega_blackwell_encode_vecadd(code_buf, sizeof(code_buf), &out_len) != 0) {
        return -1;
    }
    if (out_len != OMEGA_BW_VECADD_CODE_SIZE) {
        return -1;
    }

    uint8_t digest[32];
    if (omega_blackwell_compute_code_digest(code_buf, out_len, digest) != 0) {
        return -1;
    }

    char hex[65];
    for (int i = 0; i < 32; i++) {
        sprintf(&hex[i * 2], "%02x", digest[i]);
    }
    hex[64] = 0;

    if (strcmp(hex, OMEGA_BW_VECADD_EXPECTED_SHA256) != 0) {
        return -1;
    }

    /* Verify critical instruction patterns: */
    /* 1. S2R R9, SR_CTAID.X */
    if (VECADD_INSTRUCTIONS[1].w[0] != 0x00097919) return -1;
    /* 2. S2R R0, SR_TID.X */
    if (VECADD_INSTRUCTIONS[3].w[0] != 0x00007919) return -1;
    /* 3. IADD3 R9, PT, PT, R2, R5, RZ */
    if (VECADD_INSTRUCTIONS[17].w[0] != 0x02097210) return -1;
    /* 4. STG.E desc[UR4][R6.64], R9 */
    if (VECADD_INSTRUCTIONS[18].w[0] != 0x06007986) return -1;

    return 0;
}
