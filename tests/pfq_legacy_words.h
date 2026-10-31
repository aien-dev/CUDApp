/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* pfq_legacy_words.h: digest of the encoded words of every qg / pq spec that existed before DRA-82 (the new spec fields
 * pq.stride, qg.ymap, qg.x64 left 0). tests/gpu_pfq_test.c compares it with PFQ_LEGACY_DIGEST, the value this same code
 * printed when built against CUDApp main 4bbccf0 (evidence/DRA-82-pfq/). Uses only the pre-DRA-82 fields. */
#ifndef PFQ_LEGACY_WORDS_H
#define PFQ_LEGACY_WORDS_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_codegen_qg.h"
#include "omega_blackwell_codegen_tc.h"
#include "omega_blackwell_codegen_qmma.h"

/* FNV-1a 64 over (program index, code length, code bytes) of every accepted spec, encoded and patched as
 * omega_gpu_pf_kernel_locked does. Returns the number of programs, or -1 when one failed to encode. */
static uint64_t pfq_fnv(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; } return h;
}
static int pfq_encode_hash(BlackwellIRProgram *prog, uint8_t *code, size_t cap, uint64_t *h, uint32_t idx) {
    size_t need = omega_bw_program_code_bytes(prog);
    OmegaBlackwellKernel kern;
    if (need == 0 || need > cap || omega_blackwell_kernel_encode_into(prog, code, need, &kern) != 0) return -1;
    if (omega_tc_patch_code(prog, code, need) < 0 || omega_qmma_patch_code(prog, code, need) < 0) return -1;
    uint64_t len = need;
    *h = pfq_fnv(*h, &idx, sizeof idx); *h = pfq_fnv(*h, &len, sizeof len); *h = pfq_fnv(*h, code, need);
    return 0;
}
static int pfq_legacy_digest(uint64_t *out) {
    *out = 0;
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    size_t cap = 1u << 20; uint8_t *code = malloc(cap);
    uint64_t h = 0xcbf29ce484222325ull; uint32_t idx = 0; int bad = 0;
    if (!prog || !code) { free(prog); free(code); return -1; }
    static const uint32_t ks[] = { 128, 512, 2048, 4096 };
    for (unsigned a = 0; a < 4; a++) for (uint32_t w = 8; w <= 16; w *= 2) for (uint32_t epi = 0; epi <= 1; epi++)
    for (uint32_t dir = 0; dir <= 1; dir++) for (uint32_t wt = 0; wt <= 45; wt += 45) for (int m = 0; m <= 4; m++) {
        OmegaQgSpec s; memset(&s, 0, sizeof s);
        s.k = ks[a]; s.rt = 4; s.warps = w; s.epi = epi; s.direct = dir; s.n = epi ? 1024u : 512u; s.wait = wt; s.mutant = m;
        if (!omega_qg_spec_ok(&s)) continue;
        memset(prog, 0, sizeof *prog);
        if (omega_blackwell_codegen_qgemm_prog(&s, prog) != 0 || pfq_encode_hash(prog, code, cap, &h, idx) != 0) bad = 1;
        idx++;
    }
    static const uint32_t pk[] = { 128, 512, 2048, 4096, 8192 };
    for (unsigned a = 0; a < 5; a++) for (uint32_t it = 0; it <= 1; it++) for (int m = 0; m <= 3; m++) for (uint32_t gr = 0; gr <= 2; gr++) {
        OmegaPqSpec s; memset(&s, 0, sizeof s);
        s.k = pk[a]; s.in_type = it; s.mutant = m; s.groups = gr;
        if (!omega_pq_spec_ok(&s) || (gr == 0 && pk[a] > 4096u)) continue;
        memset(prog, 0, sizeof *prog);
        if (omega_blackwell_codegen_pq_prog(&s, prog) != 0 || pfq_encode_hash(prog, code, cap, &h, idx) != 0) bad = 1;
        idx++;
    }
    free(prog); free(code);
    *out = h;
    return bad ? -1 : (int)idx;
}
#endif
