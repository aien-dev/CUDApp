/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * q36_codegen_hash: sha256 of the encoded program of every kernel shape the 27B / 30B engines build (host only, no
 * device). Run before and after a shape parameterisation: the old shapes' lines must be byte-identical (Q36, sc L306).
 * Extra lines for the new shapes are printed after "# new" (they have no "before").
 */
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_codegen_gdn.h"
#include "omega_blackwell_codegen_attn256.h"
#include "omega_blackwell_codegen_pattn256.h"
#include "omega_blackwell_codegen_router.h"
#include "omega_blackwell_codegen_dec.h"
#include "sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static BlackwellIRProgram *prog;
static uint8_t code[1u << 22];

static void emit(const char *name, int rc) {
    size_t len = 0;
    if (rc == 0) rc = omega_bw_encode_program(prog, code, sizeof code, &len);
    if (rc != 0) { printf("%-40s refused rc=%d\n", name, rc); return; }
    uint8_t d[32];
    sha256_ctx c;
    sha256_init(&c); sha256_update(&c, code, len); sha256_final(&c, d);
    printf("%-40s %7zu ", name, len);
    for (int i = 0; i < 32; i++) printf("%02x", d[i]);
    printf("\n");
}

int main(void) {
    prog = calloc(1, sizeof *prog);
    char n[64];
    for (int m = 0; m <= 4; m++) {
        OmegaGdnSpec s = {.mutant = m};
        snprintf(n, sizeof n, "gdn.decode.m%d", m); omega_bw_ir_init(prog); emit(n, omega_blackwell_codegen_gdn_prog(&s, prog));
        snprintf(n, sizeof n, "gdn.prep.m%d", m); omega_bw_ir_init(prog); emit(n, omega_blackwell_codegen_gdn_prep_prog(&s, prog));
        snprintf(n, sizeof n, "gdn.scan.m%d", m); omega_bw_ir_init(prog); emit(n, omega_blackwell_codegen_gdn_scan_prog(&s, prog));
        snprintf(n, sizeof n, "gdn.post.m%d", m); omega_bw_ir_init(prog); emit(n, omega_blackwell_codegen_gdn_post_prog(&s, prog));
        OmegaGdnSpec sh = {.mutant = m, .heads = 1}; /* L312 lever 4: the heads form */
        snprintf(n, sizeof n, "gdn.heads.m%d", m); omega_bw_ir_init(prog); emit(n, omega_blackwell_codegen_gdn_prog(&sh, prog));
    }
    for (int c = 1; c <= 2; c++) {
        OmegaGdnSpec s = {.copy = c};
        snprintf(n, sizeof n, "gdn.copy%d", c); omega_bw_ir_init(prog); emit(n, omega_blackwell_codegen_gdn_copy_prog(&s, prog));
    }
    for (uint32_t d = 1; d <= 4; d++)
        for (int m = 0; m <= 5; m++)
            for (uint32_t p = 0; p <= 1; p++) {
                OmegaDecodeAttn256Spec s = {.depth = d, .mutant = m, .prep = p};
                snprintf(n, sizeof n, "attn256.d%u.m%d.p%u", d, m, p); omega_bw_ir_init(prog);
                emit(n, omega_blackwell_codegen_decode_attn256_prog(&s, prog));
            }
    for (uint32_t d = 1; d <= 4; d++) /* L312 lever 5: the heads form (27B shape), and the 35B shape plain and heads */
        for (int m = 0; m <= 5; m++) {
            OmegaDecodeAttn256Spec s = {.depth = d, .mutant = m, .heads = 1};
            snprintf(n, sizeof n, "attn256.d%u.m%d.heads", d, m); omega_bw_ir_init(prog);
            emit(n, omega_blackwell_codegen_decode_attn256_prog(&s, prog));
            OmegaDecodeAttn256Spec q = {.depth = d, .mutant = m, .q_heads = 16, .kv_heads = 2, .q_per_cta = 4};
            snprintf(n, sizeof n, "attn256q36.d%u.m%d", d, m); omega_bw_ir_init(prog);
            emit(n, omega_blackwell_codegen_decode_attn256_prog(&q, prog));
            q.heads = 1;
            snprintf(n, sizeof n, "attn256q36.d%u.m%d.heads", d, m); omega_bw_ir_init(prog);
            emit(n, omega_blackwell_codegen_decode_attn256_prog(&q, prog));
        }
    for (int m = 0; m <= 4; m++) {
        OmegaPrefillAttn256Spec s = {.mutant = m};
        snprintf(n, sizeof n, "pattn256.m%d", m); omega_bw_ir_init(prog); emit(n, omega_blackwell_codegen_prefill_attn256_prog(&s, prog));
    }
    uint32_t warps[3] = {8, 16, 32};
    for (int w = 0; w < 3; w++)
        for (uint32_t d = 1; d <= 4; d++)
            for (int m = 0; m <= 4; m++) {
                OmegaRouterSpec s = {.warps = warps[w], .depth = d, .mutant = m};
                snprintf(n, sizeof n, "router.w%u.d%u.m%d", warps[w], d, m); omega_bw_ir_init(prog);
                emit(n, omega_blackwell_codegen_router_prog(&s, prog));
            }
    for (int w = 0; w < 3; w++) /* the ROWS form (L312 lever 2) at the stroke's strides, plain and mutant 6 */
        for (int m = 0; m <= 6; m += 6) {
            OmegaRouterSpec s = {.warps = warps[w], .depth = 4, .mutant = m, .experts = 256, .shared = 1, .xstride = 8192, .lbstride = 16896};
            snprintf(n, sizeof n, "router256s.rows.w%u.d4.m%d", warps[w], m); omega_bw_ir_init(prog);
            emit(n, omega_blackwell_codegen_router_prog(&s, prog));
        }
    /* the decode glue kernels (RES / SWIGLU / ARGMAX / GATHER / GATE / AXPY) at the widths the 30B and 27B graphs use */
    const uint32_t widths[3] = {0, 5120, 17408}, strides[3] = {0, 256, 65536};
    for (int k = 1; k <= 6; k++)
        for (uint32_t mode = 0; mode <= 15; mode++)
            for (int m = 0; m <= 2; m++)
                for (int wi = 0; wi < 3; wi++)
                    for (int si = 0; si < 3; si++) {
                        if ((k != OMEGA_DEC_RES && k != OMEGA_DEC_ARGMAX && mode) || (k != OMEGA_DEC_AXPY && k != OMEGA_DEC_GATHER && si)) continue;
                        OmegaDecSpec s = {.kind = (OmegaDecKind)k, .mode = mode, .stride = strides[si], .mutant = m, .width = widths[wi]};
                        snprintf(n, sizeof n, "dec.k%d.mode%u.m%d.w%u.s%u", k, mode, m, widths[wi], strides[si]); omega_bw_ir_init(prog);
                        emit(n, omega_blackwell_codegen_dec_prog(&s, prog));
                    }
    /* GROUP (the stroke v2 sort of N heads x S picks into expert segments): S = 8 (0) and 9 (the 35B), mutants 0..3 */
    for (uint32_t gr = 0; gr <= 9; gr += 9)
        for (int m = 0; m <= 3; m++) {
            OmegaDecSpec s = {.kind = OMEGA_DEC_GROUP, .groups = gr, .mutant = m};
            snprintf(n, sizeof n, "dec.k7.g%u.m%d", gr, m); omega_bw_ir_init(prog);
            emit(n, omega_blackwell_codegen_dec_prog(&s, prog));
        }
#ifdef Q36_NEW_SHAPES
    printf("# new\n");
    Q36_NEW_SHAPES
#endif
    free(prog);
    return 0;
}
