/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* test_omega_codegen_freestanding.c -- native v1 P07 cut A.2 (aienos#286, aien-architecture#190).
 *
 * Built twice by mk/omega-gpu-engine.mk (make test-codegen-freestanding):
 *  - hosted: linked against the hosted codegen/matmul objects; every fixture goes through the existing
 *    allocating generators (malloc'd code buffer, omega_blackwell_kernel_free);
 *  - freestanding (-DOMEGA_GPU_NO_POSIX): linked against the -ffreestanding objects of
 *    `make gpu-engine-freestanding`; every fixture goes through the *_prog generator and
 *    omega_blackwell_kernel_encode_into with a caller-supplied buffer, no allocator anywhere.
 * `--dump` prints every fixture's instruction stream as hex; the make target requires the two dumps to be
 * byte-identical (cmp). Without --dump the freestanding build also checks the bounded buffer contract
 * (every short buffer refused before a byte is written, nothing written past the reported size), the
 * golden-word fixture verifiers against the freestanding object, and the integer bf16 rounding path
 * against the hosted __bf16 cast on all 2^32 inputs.
 * Host only: opens no device. */
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_matmul.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum { FX_I32, FX_TENSOR, FX_LOOP, FX_FRAG, FX_BLOCK } FxKind;
typedef struct {
    const char *name;
    FxKind kind;
    uint32_t m, k, n;
    OmegaMatMulPrecision prec;
    uint32_t grid_x, unroll, rt, ct; /* unroll 0 = the API's rule (omega_matmul_frag_unroll / _block_unroll) */
    int mutant;
} Fixture;

/* The shapes the existing tests and the matmul API generate: the integer and K=16 tensor kernels of the
 * gates, the looped kernel, the decode fragment kernel at Qwen3 shapes and the blocked prefill kernel. */
static const Fixture FX[] = {
    {"i32_16x16x16", FX_I32, 16, 16, 16, OMEGA_MATMUL_PRECISION_INT32, 0, 0, 0, 0, 0},
    {"i32_32x16x64", FX_I32, 32, 16, 64, OMEGA_MATMUL_PRECISION_INT32, 0, 0, 0, 0, 0},
    {"i32_16x32x8", FX_I32, 16, 32, 8, OMEGA_MATMUL_PRECISION_INT32, 0, 0, 0, 0, 0},
    {"tensor_f16_16x16x8", FX_TENSOR, 16, 16, 8, OMEGA_MATMUL_PRECISION_FP16, 0, 0, 0, 0, 0},
    {"tensor_bf16_32x16x16", FX_TENSOR, 32, 16, 16, OMEGA_MATMUL_PRECISION_BF16, 0, 0, 0, 0, 0},
    {"loop_bf16_16x64x32_g2", FX_LOOP, 16, 64, 32, OMEGA_MATMUL_PRECISION_BF16, 2, 0, 0, 0, 0},
    {"loop_f16_32x128x64_g4", FX_LOOP, 32, 128, 64, OMEGA_MATMUL_PRECISION_FP16, 4, 0, 0, 0, 0},
    {"frag_k128_n64_g1", FX_FRAG, 16, 128, 64, OMEGA_MATMUL_PRECISION_BF16, 1, 0, 0, 0, 0},
    {"frag_k1024_n3072_g64", FX_FRAG, 16, 1024, 3072, OMEGA_MATMUL_PRECISION_BF16, 64, 0, 0, 0, 0},
    {"frag_k4096_n4096_g48", FX_FRAG, 16, 4096, 4096, OMEGA_MATMUL_PRECISION_BF16, 48, 0, 0, 0, 0},
    {"frag_k4096_n4096_g48_mutant", FX_FRAG, 16, 4096, 4096, OMEGA_MATMUL_PRECISION_BF16, 48, 0, 0, 0, 1},
    {"block_k256_n128_r4c2", FX_BLOCK, 16, 256, 128, OMEGA_MATMUL_PRECISION_BF16, 1, 0, 4, 2, 0},
    {"block_k1024_n3072_r2c2", FX_BLOCK, 16, 1024, 3072, OMEGA_MATMUL_PRECISION_BF16, 8, 0, 2, 2, 0},
    {"block_k4096_n4096_r4c4", FX_BLOCK, 16, 4096, 4096, OMEGA_MATMUL_PRECISION_BF16, 16, 0, 4, 4, 0},
};
#define NFX (sizeof FX / sizeof FX[0])

static uint32_t fx_unroll(const Fixture *f) {
    if (f->unroll) return f->unroll;
    if (f->kind == FX_BLOCK) return omega_matmul_frag_block_unroll(f->k, f->rt, f->ct);
    uint32_t u = omega_matmul_frag_unroll(f->k);
    if (f->mutant) while (u > 1 && f->k / 16u / u < 2) u /= 2; /* the API's mutant rule */
    return u;
}

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL "); printf(__VA_ARGS__); printf("\n"); } } while (0)

#ifdef OMEGA_GPU_NO_POSIX
static BlackwellIRProgram g_prog; /* one program at a time; static keeps the stack small */
static uint8_t g_code[OMEGA_BW_CODE_CAP_MAX + 64];

static int fx_prog(const Fixture *f, BlackwellIRProgram *prog) {
    OmegaMatMulSpec spec;
    if (omega_matmul_spec_init(&spec, f->m, f->k, f->n, f->prec) != 0) return -1;
    switch (f->kind) {
    case FX_I32: return omega_blackwell_codegen_matmul_i32_prog(&spec, prog);
    case FX_TENSOR: return omega_blackwell_codegen_matmul_tensor_prog(&spec, prog);
    case FX_LOOP: return omega_blackwell_codegen_matmul_tensor_loop_prog(&spec, f->grid_x, f->mutant, prog);
    case FX_FRAG: return omega_blackwell_codegen_matmul_frag_prog(&spec, f->grid_x, fx_unroll(f), f->mutant, prog);
    case FX_BLOCK:
        return omega_blackwell_codegen_matmul_frag_block_prog(&spec, f->grid_x, fx_unroll(f), f->rt, f->ct, f->mutant, prog);
    }
    return -1;
}

/* The cap each hosted generator always had: the i32 buffer is exactly the program, frag_block the IR maximum,
 * the others 4096 bytes. The freestanding build passes the same cap so a program the hosted build refuses is
 * refused here too. */
static size_t fx_cap(const Fixture *f, const BlackwellIRProgram *prog) {
    if (f->kind == FX_I32) return omega_bw_program_code_bytes(prog);
    if (f->kind == FX_BLOCK) return OMEGA_BW_CODE_CAP_MAX;
    return OMEGA_BW_CODE_CAP_SMALL;
}

static int fx_kernel(const Fixture *f, OmegaBlackwellKernel *k) {
    if (fx_prog(f, &g_prog) != 0) { memset(k, 0, sizeof *k); return -1; }
    return omega_blackwell_kernel_encode_into(&g_prog, g_code, fx_cap(f, &g_prog), k);
}
static void fx_release(OmegaBlackwellKernel *k) { memset(k, 0, sizeof *k); } /* caller-owned buffer: nothing to free */
#else
static int fx_kernel(const Fixture *f, OmegaBlackwellKernel *k) {
    OmegaMatMulSpec spec;
    memset(k, 0, sizeof *k);
    if (omega_matmul_spec_init(&spec, f->m, f->k, f->n, f->prec) != 0) return -1;
    switch (f->kind) {
    case FX_I32: return omega_blackwell_codegen_matmul_i32(&spec, k);
    case FX_TENSOR: return omega_blackwell_codegen_matmul_tensor(&spec, k);
    case FX_LOOP: return omega_blackwell_codegen_matmul_tensor_loop(&spec, f->grid_x, f->mutant, k);
    case FX_FRAG: return omega_blackwell_codegen_matmul_frag(&spec, f->grid_x, fx_unroll(f), f->mutant, k);
    case FX_BLOCK: return omega_blackwell_codegen_matmul_frag_block(&spec, f->grid_x, fx_unroll(f), f->rt, f->ct, f->mutant, k);
    }
    return -1;
}
static void fx_release(OmegaBlackwellKernel *k) { omega_blackwell_kernel_free(k); }
#endif

static int dump(void) {
    for (size_t i = 0; i < NFX; i++) {
        OmegaBlackwellKernel k;
        int rc = fx_kernel(&FX[i], &k);
        printf("fixture %s rc %d bytes %zu insns %zu gpr %u ugpr %u\n", FX[i].name, rc, k.code_size, k.insn_count,
               k.gpr_count, k.uniform_gpr_count);
        if (rc != 0 || !k.code || k.code_size == 0) { fails++; printf("FAIL fixture %s did not generate\n", FX[i].name); continue; }
        printf("digest ");
        for (int j = 0; j < 32; j++) printf("%02x", k.code_digest[j]);
        printf("\n");
        for (size_t j = 0; j < k.code_size; j++) printf("%02x%s", k.code[j], (j % 32u == 31u) ? "\n" : "");
        if (k.code_size % 32u) printf("\n");
        fx_release(&k);
    }
    return fails ? 1 : 0;
}

#ifdef OMEGA_GPU_NO_POSIX
static int all_bytes(const uint8_t *p, size_t n, uint8_t v) {
    for (size_t i = 0; i < n; i++) if (p[i] != v) return 0;
    return 1;
}

/* The bounded-buffer contract, on every fixture's program. */
static void buffer_contract(void) {
    for (size_t i = 0; i < NFX; i++) {
        const Fixture *f = &FX[i];
        OmegaBlackwellKernel k;
        if (fx_prog(f, &g_prog) != 0) { CHECK(0, "%s: program", f->name); continue; }
        const size_t need = omega_bw_program_code_bytes(&g_prog);
        CHECK(need >= 32u * 16u && need % 128u == 0, "%s: code bytes %zu (padded to 8 instructions, at least 32)", f->name, need);
        CHECK(need <= OMEGA_BW_CODE_CAP_MAX, "%s: code bytes %zu above the IR maximum", f->name, need);

        /* one byte short: refused, not a byte written, kernel zeroed */
        memset(g_code, 0xa5, sizeof g_code);
        memset(&k, 0x5a, sizeof k);
        int rc = omega_blackwell_kernel_encode_into(&g_prog, g_code, need - 1u, &k);
        CHECK(rc == OMEGA_BW_CODE_BUFFER_TOO_SMALL, "%s: need-1 rc %d", f->name, rc);
        CHECK(all_bytes(g_code, sizeof g_code, 0xa5), "%s: refused encode wrote into the buffer", f->name);
        CHECK(k.code == NULL && k.code_size == 0 && k.insn_count == 0, "%s: refused encode left a kernel", f->name);

        /* zero capacity and NULL buffer refused */
        CHECK(omega_blackwell_kernel_encode_into(&g_prog, g_code, 0, &k) == OMEGA_BW_CODE_BUFFER_TOO_SMALL, "%s: cap 0", f->name);
        CHECK(omega_blackwell_kernel_encode_into(&g_prog, NULL, need, &k) == -1 && k.code == NULL, "%s: NULL buffer", f->name);
        CHECK(omega_blackwell_kernel_encode_into(NULL, g_code, need, &k) == -1, "%s: NULL program", f->name);
        CHECK(omega_blackwell_kernel_encode_into(&g_prog, g_code, need, NULL) == -1, "%s: NULL kernel", f->name);

        /* exactly enough, and a larger buffer: same bytes, nothing written past the reported size */
        memset(g_code, 0xa5, sizeof g_code);
        rc = omega_blackwell_kernel_encode_into(&g_prog, g_code, need + 64u, &k);
        CHECK(rc == 0 && k.code == g_code && k.code_size == need && k.insn_count == need / 16u,
              "%s: roomy encode rc %d size %zu", f->name, rc, k.code_size);
        CHECK(all_bytes(g_code + need, 64u, 0xa5), "%s: encode wrote past code_size", f->name);
        static uint8_t first[OMEGA_BW_CODE_CAP_MAX];
        memcpy(first, g_code, need);
        uint8_t d1[32];
        memcpy(d1, k.code_digest, 32);
        memset(g_code, 0, sizeof g_code);
        rc = omega_blackwell_kernel_encode_into(&g_prog, g_code, need, &k);
        CHECK(rc == 0 && memcmp(first, g_code, need) == 0 && memcmp(d1, k.code_digest, 32) == 0,
              "%s: exact-size encode differs (rc %d)", f->name, rc);
    }
}

static void golden_fixtures(void) {
    CHECK(omega_blackwell_verify_codegen_fixtures() == 0, "golden words (rc %d)", omega_blackwell_verify_codegen_fixtures());
    CHECK(omega_blackwell_verify_codegen_fixtures_fb1cut4() == 0, "fb1cut4 golden words");
    CHECK(omega_blackwell_verify_codegen_fixtures_intops() == 0, "intops golden words");
    CHECK(omega_blackwell_verify_codegen_fixtures_fragmm() == 0, "fragmm golden words");
    CHECK(omega_blackwell_verify_codegen_fixtures_reconv() == 0, "reconv golden words");
    CHECK(omega_blackwell_test_regalloc_bounds() == 0, "regalloc bounds");
}

/* The freestanding omega_fp32_to_bf16 (integer rounding) against the hosted __bf16 cast, every input. */
static void bf16_sweep(void) {
    uint64_t bad = 0;
    uint32_t first_bad = 0;
    for (uint64_t x = 0; x <= 0xffffffffull; x++) {
        uint32_t u = (uint32_t)x;
        float f;
        memcpy(&f, &u, 4);
        __bf16 b = (__bf16)f;
        uint16_t want;
        memcpy(&want, &b, 2);
        if (omega_fp32_to_bf16(f) != want) { if (!bad) first_bad = u; bad++; }
    }
    CHECK(bad == 0, "bf16: %llu of 2^32 inputs differ from the __bf16 cast (first 0x%08x)", (unsigned long long)bad, first_bad);
    if (!bad) printf("bf16 sweep: all 2^32 inputs equal the hosted __bf16 cast\n");
}
#endif

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) return dump();
#ifdef OMEGA_GPU_NO_POSIX
    buffer_contract();
    golden_fixtures();
    if (!(argc > 1 && strcmp(argv[1], "--no-sweep") == 0)) bf16_sweep();
    printf("%s: freestanding codegen contract (%zu fixtures, %d failing checks)\n", fails ? "FAIL" : "PASS", (size_t)NFX, fails);
    return fails ? 1 : 0;
#else
    fprintf(stderr, "hosted build: only --dump\n");
    return 2;
#endif
}
