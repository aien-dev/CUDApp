/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
#ifndef OMEGA_AMPERE_CODEGEN_H
#define OMEGA_AMPERE_CODEGEN_H
#include "omega_blackwell_codegen.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Explicit target; never guesses from the host or a GPU. 121 preserves the existing
 * encoder. 86 lowers solved IR to Ampere, including parameter-bank offsets and
 * BF16 PACK / butterfly placeholders. Native FP8 QMMA and E4M3 PACK are refused.
 * 0 success, -1 invalid arguments/capacity, -2 unsupported target/IR/native trap.
 * Refusal writes no code bytes and sets *out_len=0. No driver access occurs. */
#define OMEGA_GPU_ENCODE_UNSUPPORTED (-2)
int omega_gpu_encode_program_target(const BlackwellIRProgram *prog, uint32_t target_sm,
                                    uint8_t *code_buf, size_t max_len, size_t *out_len);
#ifdef __cplusplus
}
#endif
#endif
