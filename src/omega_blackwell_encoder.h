#ifndef OMEGA_BLACKWELL_ENCODER_H
#define OMEGA_BLACKWELL_ENCODER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define OMEGA_BW_INSTRUCTION_BYTES      16
#define OMEGA_BW_VECADD_INSN_COUNT      32
#define OMEGA_BW_VECADD_CODE_SIZE       (OMEGA_BW_VECADD_INSN_COUNT * OMEGA_BW_INSTRUCTION_BYTES) /* 512 bytes */
#define OMEGA_BW_VECADD_EXPECTED_SHA256 "39f3dfdfc529a75a8274a27900acf4e7382b7f280a211e0412c51609f44b62a1"

typedef struct {
    uint32_t w[4];
    const char *mnemonic;
} BlackwellInstruction;

/* Minimal sovereign encoder: emits 512-byte Blackwell sm_121 vector add machine code */
int omega_blackwell_encode_vecadd(uint8_t *code_buf, size_t max_len, size_t *out_len);

/* Computes SHA-256 digest of emitted machine code */
int omega_blackwell_compute_code_digest(const uint8_t *code_buf, size_t len, uint8_t *out_digest);

/* Verifies encoder instruction fixtures against reference bit-for-bit disassembled words */
int omega_blackwell_verify_encoder_fixtures(void);

#endif /* OMEGA_BLACKWELL_ENCODER_H */
