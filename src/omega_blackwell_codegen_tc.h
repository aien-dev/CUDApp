#ifndef OMEGA_BLACKWELL_CODEGEN_TC_H
#define OMEGA_BLACKWELL_CODEGEN_TC_H
/*
 * omega_blackwell_codegen_tc: tensor-core (HMMA.16816.F32.BF16) kernels for the OWN-30B one-pass prefill (T3, 2026-10-13).
 * New file. Uses only opcodes the encoder already has and fixture-tests (omega_blackwell_codegen.h).
 *
 * HMMA.16816 fragment layout (lane L, g = L / 4, t = L % 4; register q holds element 2q in its low half, 2q+1 high):
 *   A 16x16 (row-major M x K): a0,a1 = A[g][2t, 2t+1]   a2,a3 = A[g+8][2t, 2t+1]   a4,a5 = A[g][2t+8, 2t+9]   a6,a7 = A[g+8][2t+8, 2t+9]
 *   B 16x8  (K x N):           b0,b1 = B[2t, 2t+1][g]   b2,b3 = B[2t+8, 2t+9][g]
 *   C/D 16x8 f32:              c0,c1 = C[g][2t, 2t+1]   c2,c3 = C[g+8][2t, 2t+1]
 * (PTX ISA mma.m16n8k16 fragment figures; the fragment-major matmul of omega_blackwell_codegen.h packs to the same order and
 * is chip-proven.)
 *
 * ---- probe kernel (characterisation of the chip's HMMA arithmetic) ----
 * One warp per CTA, ctaid.x = case. Per case, at in + case * 1280: A quads [32 lanes][4 u32] (512 B), B pairs [32][2] (256 B),
 * C quads [32][4] f32 (512 B). Output at out + case * 512: D quads [32][4] f32. Arguments: 0x380 in u64, 0x388 out u64.
 * D = A * B + C with one HMMA.16816.F32.BF16. mutant 1 (tests only): C is replaced by RZ (must be caught when C != 0).
 */
#include <stdint.h>
#include "omega_blackwell_codegen.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_TC_PROBE_IN_BYTES 1280u
#define OMEGA_TC_PROBE_OUT_BYTES 512u
int omega_blackwell_codegen_tc_probe_prog(int mutant, BlackwellIRProgram *prog);

/* ---- grouped FP8 GEMM on tensor cores (cut 1: e4m3 weights decoded to bf16 in registers, HMMA.16816.F32.BF16) ----
 *   y[row0 + r][c] = sum_kb S'[c/128][kb] * ( sum_{k in kb} w(W[c][k]) * x(X[rowidx[row0 + r]][k]) )
 * w(e) = e4m3 value * 2^-120 as bf16 bits ((e & 0x7F) << 4 | (e & 0x80) << 8: exact, subnormal codes become bf16 subnormals,
 * which the HMMA honours: evidence OWN30B-T3-HMMA-MODEL), x = bf16(x * 2^64), S' = scale * 2^56 (the FP8 GEMV's fold, so
 * the true value is recovered). Per 128-wide K block: 8 HMMA steps into a zeroed temporary, then acc = fma(tmp, S', acc).
 *
 * Physical K order. Within every aligned group of 4 elements (k = 4q .. 4q+3) the activations are STORED as
 * [x(4q), x(4q+2), x(4q+1), x(4q+3)] (bf16, row-major otherwise): omega_tcgemm_permute_x. Weights stay in checkpoint order.
 * HMMA step (chunk c of 64 elements, s = 0..3) consumes the 16 elements c*64 + 16t + 4s + j (t = 0..3 lane%4, j = 0..3);
 * lane t's weight bytes j = 0,2 form B register 0 and j = 1,3 register 1, so one e4m3 word decodes into both B registers
 * with 8 integer ops, and the matching activation words arrive in order. The fused 16-term HMMA sum does not depend on the
 * order of its terms (tests/bw_hmma_model.h), so the result depends only on the step grouping above.
 *
 * Tile table (device, ctaid.y = tile, 32 bytes each, the FP8 GEMM's segment entry): { w_va u64, s_va u64, row0 u32, cnt u32,
 * pad[8] } with 1 <= cnt <= 16*rt; rows row0 .. row0+cnt-1 of the sorted row space belong to one expert.
 * rowidx (device u32): sorted row -> source row of X; entries row0 .. row0 + 16*rt - 1 must be readable for every tile (the
 * host pads with valid rows; rows past cnt are computed and not stored).
 * Constant bank 0 (cbank + 0x380): tiles u64, rowidx u64, x u64 (bf16 rows of k), y u64 (f32 rows of n), n u32.
 * Grid: grid_x = n / (warps * 8 * ct) (exact), grid_y = tiles. CTA: 32 * warps threads; shared 2 * 16 * rt * 128 bytes.
 * mutant (tests only): 1 drops the block scale, 2 never advances the scale pointer, 3 row tile i reads tile i^1 (rt >= 2),
 * 4 drops the last 64-element chunk of every 128 block. Each must be caught. */
typedef struct {
    uint32_t k;     /* multiple of 128, 128 .. 16384 */
    uint32_t rt;    /* row tiles (16 rows) per CTA: 1, 2 or 4 */
    uint32_t ct;    /* column tiles (8 weight rows) per warp: 1, 2 or 4; rt * ct >= 4 */
    uint32_t warps; /* 1, 2, 4 or 8; (16 * rt * 8) % (32 * warps) == 0 */
    int mutant;
} OmegaTcGemmSpec;
#define OMEGA_TCGEMM_ARG_TILES 0x380u
#define OMEGA_TCGEMM_ARG_ROWIDX 0x388u
#define OMEGA_TCGEMM_ARG_X 0x390u
#define OMEGA_TCGEMM_ARG_Y 0x398u
#define OMEGA_TCGEMM_ARG_N 0x3a0u
#define OMEGA_TCGEMM_ARGS_WORDS 9u
#define OMEGA_TCGEMM_TILE_BYTES 32u
int omega_tcgemm_spec_ok(const OmegaTcGemmSpec *spec);
uint32_t omega_tcgemm_shared_bytes(const OmegaTcGemmSpec *spec);
uint32_t omega_tcgemm_span(const OmegaTcGemmSpec *spec); /* weight rows per CTA */
int omega_blackwell_codegen_tcgemm_prog(const OmegaTcGemmSpec *spec, BlackwellIRProgram *prog);
/* x (k values) -> the stored bf16 row: RNE bf16 of x * 2^64 in the (0,2,1,3) group order */
void omega_tcgemm_permute_x(const float *x, uint32_t k, uint16_t *out);
/* e4m3 code -> the bf16 bits the kernel feeds the HMMA (value * 2^-120) */
uint16_t omega_tcgemm_w_bf16(uint8_t code);

/* tensor-core peak microbenchmark: each thread runs `iters` x `chains` HMMA.16816.F32.BF16 (A = B = 1.0, C chained per
 * accumulator, `stall` cycles between issues), then stores its chains x 4 accumulators at out (arg 0x380) + (ctaid * threads +
 * tid) * 16 * chains. Every stored value must be 16 * iters (a hazard on a chained accumulator shows as a wrong value). */
int omega_blackwell_codegen_tc_peak_prog(uint32_t iters, uint32_t chains, uint32_t stall, uint32_t threads, BlackwellIRProgram *prog);

/* read-bandwidth probe: CTA c streams region (c % groups) of iters * threads * 128 bytes (8 LDG.128 per thread per iteration,
 * XOR-folded) from arg 0x380 and stores one word per thread at arg 0x388 + (c * threads + tid) * 4. groups = grid: every
 * CTA distinct (DRAM); groups = 1: every CTA the same bytes (shows any cache reuse of these buffers). */
int omega_blackwell_codegen_tc_bw_prog(uint32_t iters, uint32_t groups, uint32_t threads, BlackwellIRProgram *prog);

/* ---- opcodes the shared encoder lacks, emitted as same-footprint placeholders and patched after encoding ----
 * SHFL.BFLY PT, Rd, Ra, lane, 0x1f: a BW_IR_SHFL_DOWN whose imm carries OMEGA_TC_MARK_BFLY; the patch sets bit 26 of word 1
 *   (nvcc 13.0.88 sm_121 oracle, investigations/2026-10-13-t3-mma-oracle/attn/fa_ops.sass: SHFL.BFLY PT, R5, R0, 0x1, 0x1f =
 *   0x0c201f0000057f89 / 0x004e2800000e0000; the shared encoder's SHFL.DOWN word 1 is 0x08001f00 | lane << 21).
 * F2FP.BF16.F32.PACK_AB Rd, Ra, Rb (Rd = bf16_rn(Ra) << 16 | bf16_rn(Rb)): a BW_IR_FMNMX_MAX whose imm is OMEGA_TC_MARK_F2FP;
 *   the patch rewrites the opcode to 0x723e and word 2 to 0x000010ff (oracle: F2FP.BF16.F32.PACK_AB R9, R3, R0 =
 *   0x000000000309723e / 0x000fe200000010ff; registers stay at Rd 16..23, Ra 24..31, Rb 32..39 as FMNMX). F2FP needs no
 *   scoreboard in the oracle (fixed latency), as FMNMX. */
#define OMEGA_TC_MARK_BFLY 0x80000000u
#define OMEGA_TC_MARK_F2FP 0x46320000u
/* patch the encoded code (len bytes, one 16-byte word per IR instruction); returns the number patched or -1 on a word that
 * does not hold the expected placeholder bits */
int omega_tc_patch_code(const BlackwellIRProgram *prog, uint8_t *code, size_t len);
/* encoder self-test: placeholders encoded and patched against the oracle words (register fields excluded); 0 = pass */
int omega_tc_patch_selftest(char *msg, size_t msgn);

#ifdef __cplusplus
}
#endif
#endif
