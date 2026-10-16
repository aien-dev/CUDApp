#ifndef OMEGA_BLACKWELL_CODEGEN_TERN_H
#define OMEGA_BLACKWELL_CODEGEN_TERN_H
/*
 * omega_blackwell_codegen_tern: KT1, the W2A8 ternary GEMV (m = 1) for the BitNet b1.58 lane (ternary engine design
 * stage T0, 2026-10-15). New file; same emitter rules as omega_blackwell_codegen_bf16.c (every instruction waits on
 * everything still pending, loads of one iteration issued back to back).
 *
 *   y[r] = sum_c W[r][c] * x[c],  W in {-1, 0, +1} (omega's 2-bit trit code), x int8, y int32, EXACT
 *   (the contract of oma_rz_oracle, src/algebra/realize_common.h). Optional f32 epilogue (below).
 *
 * Trit code (src/algebra/oma_trit.h): bit0 = plus plane, bit1 = minus plane, 0b11 invalid (the uploader refuses it).
 * A 32-bit word holds 16 trits of one row, columns 16c .. 16c+15, trit i at bits 2i, 2i+1.
 *
 * Packed layouts (the uploader writes whichever the tensor was created with; n is padded with zero rows to a multiple
 * of 8, kw = k / 16 words per row):
 *   OMEGA_TERN_LAYOUT_ROWMAJOR (0): word c of row r at byte (r * kw + c) * 4.
 *   OMEGA_TERN_LAYOUT_BLOCK    (1): rows in blocks of 2R (R = spec.rows, 2 or 4); in block b, word c of block row j
 *                                   (j = h * R + i, h = 0/1) at byte b * 2R * kw * 4 + c * 2R * 4 + j * 4. One
 *                                   LDG.64 / LDG.128 gives a lane word c of R rows; a warp's step is 2R * 64 contiguous bytes.
 *
 * Thread mapping: a warp is two half-warps of 16 lanes (h = lane >> 4, l = lane & 15). Unit u = 2R consecutive rows;
 * half h owns rows u * 2R + h * R + i (i < R). Per K step s (KS = k / 256 steps) lane l reads word c = 16 s + l of each of
 * its rows. Warp w of CTA cx handles units cx * warps + w, + stride, ... < U (U = padded n / 2R).
 *
 * Arithmetic (exact, integer only; IMAD low 32 bits): the CTA stages x in shared memory as, per word c and pair
 * j < 8, X_j = x[16c+j] * 65536 + x[16c+j+8] (signed): 32 bytes per c, 2 bytes per column. For a weight word W and
 * pair j: U = W >> 2j, P = U & 0x00010001 (plus bits of trits j and j+8 at bits 0 and 16), M = U & 0x00020002 (minus
 * bits at 1 and 17). accp += P * X_j, accm += M * X_j, both from K0 = 2^31 + 2^15. Mod 2^32 the high half of accp
 * collects HP = sum (p_j x_j + p_{j+8} x_{j+8}) and its low half |LP| <= 1024 never borrows, so accp >> 16 (logical) is
 * exactly 2^15 + HP; likewise accm >> 17 is exactly 2^14 + HM. Each lane adds (2^15 + HP) - (2^14 + HM) = 2^14 + dot
 * per word to an int32 total T; a SHFL.DOWN tree (8, 4, 2, 1) sums its half; y = T - KSC * 2^18 (mod 2^32, exact
 * because |y| <= 128 * k < 2^31).
 *
 * Shared memory: the launch descriptor allows at most OMEGA_BW_QMD_MAX_SHARED_BYTES (16 KiB) per CTA, so x is staged
 * in chunks of at most OMEGA_TERN_CHUNK_MAX = 8192 columns (k <= 8192: one chunk; 8192 < k <= 16384: two equal chunks,
 * k % 512 == 0). With two chunks each lane l == 0 adds the earlier chunk's int32 y it stored itself (same thread, same
 * address, program order); the f32 epilogue is refused there.
 *
 * Epilogue:
 *   OMEGA_TERN_EPI_I32 (0): y int32.
 *   OMEGA_TERN_EPI_F32 (1): y f32 = ((float)y_int * a_scale) * w_scale[seg(r)], two round-to-nearest FMULs in that order.
 *     a_scale: one f32 device word (the activation dequantisation step, 1 / s_a; the producer writes it).
 *     seg(r) = number of boundaries b0 <= b1 <= b2 that are <= r (the API passes -b_i). w_scale: 4 f32 device words
 *     stored REVERSED (ws_rev[i] = w_scale[3 - i]) so the address is ws_rev + 4 * (count of boundaries above r).
 *     Per output segment, explicit boundaries (q|k|v: 2560, 3200, 3840): never ws_idx = r / (n / segments).
 *
 * Constant bank 0 argument words (cbank + 0x380):
 *   0x380 w    u64  packed weights      0x388 x  u64  int8 x row (16-byte aligned, k bytes)
 *   0x390 y    u64  y (n words)         0x398 ws u64  w_scale reversed (f32 [4], EPI_F32)
 *   0x3a0 as   u64  a_scale (f32 word, EPI_F32)
 *   0x3a8 U    u32  units               0x3ac stride u32 units per warp iteration (grid_x * warps)
 *   0x3b0 negn u32  -n (store rows < n) 0x3b4, 0x3b8, 0x3bc: -b0, -b1, -b2
 *
 * mutant (tests only): 1 flips the planes (the staged X_j are negated: plus acts as minus, y becomes -y), 2 is host-side only
 * (the API maps segments as ws_idx = r / (n / segments)), 3 drops pair j = 7 (trits 7 and 15 of every word).
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_TERN_MAX_K 16384u
#define OMEGA_TERN_CHUNK_MAX 8192u
/* Cycles every user stays behind a fixed-latency producer (chip-proven rule: >= 5; omega_gpu_elementwise_api.c:13).
 * The generator sets each stall from the register dependencies after allocation (tern_schedule_stalls). */
#define OMEGA_TERN_FIXED_LAT 6u
#define OMEGA_TERN_LAYOUT_ROWMAJOR 0
#define OMEGA_TERN_LAYOUT_BLOCK 1
#define OMEGA_TERN_EPI_I32 0
#define OMEGA_TERN_EPI_F32 1
#define OMEGA_TERN_ARG_W 0x380u
#define OMEGA_TERN_ARG_X 0x388u
#define OMEGA_TERN_ARG_Y 0x390u
#define OMEGA_TERN_ARG_WS 0x398u
#define OMEGA_TERN_ARG_AS 0x3a0u
#define OMEGA_TERN_ARG_U 0x3a8u
#define OMEGA_TERN_ARG_STRIDE 0x3acu
#define OMEGA_TERN_ARG_NEGN 0x3b0u
#define OMEGA_TERN_ARG_NEGB 0x3b4u
#define OMEGA_TERN_ARGS_WORDS 16u
#define OMEGA_TERN_ROW_PAD 8u

typedef struct {
    uint32_t k;      /* multiple of 256 (512 above 8192), <= OMEGA_TERN_MAX_K */
    uint32_t layout; /* OMEGA_TERN_LAYOUT_* */
    uint32_t rows;   /* rows per half-warp R: row-major 1, 2, 4; block 2, 4 */
    uint32_t warps;  /* 1 .. 32 */
    uint32_t depth;  /* K steps per loop iteration: divides chunk / 256; rows * depth <= 16 */
    uint32_t epi;    /* OMEGA_TERN_EPI_* */
    int mutant;      /* 0 production; 1, 3 kernel mutants (2 is host-side, kernel identical to 0) */
} OmegaTernGemvSpec;

int omega_blackwell_codegen_tern_gemv_prog(const OmegaTernGemvSpec *spec, BlackwellIRProgram *prog);
int omega_tern_gemv_spec_ok(const OmegaTernGemvSpec *spec);
uint32_t omega_tern_gemv_steps(const OmegaTernGemvSpec *spec);        /* k / 256 (0 when refused) */
uint32_t omega_tern_gemv_shared_bytes(const OmegaTernGemvSpec *spec); /* 2 * chunk width */
uint32_t omega_tern_gemv_chunks(const OmegaTernGemvSpec *spec);       /* x chunks: 1 for k <= 8192, else 2 */
int omega_tern_gemv_prog_stats(const OmegaTernGemvSpec *spec, uint32_t *insns, uint32_t *gprs);

/* Byte offset of word c of padded row r in a packed tensor (layout, R = rows for the block layout). */
uint64_t omega_tern_word_offset(uint32_t layout, uint32_t R, uint32_t k, uint32_t r, uint32_t c);

#ifdef __cplusplus
}
#endif
#endif
