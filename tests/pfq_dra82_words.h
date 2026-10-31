/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* pfq_dra82_words.h: digest of the encoded words of every qg / pq spec reachable through omega_gpu_pfq_api before DRA-84
 * (qg epi F32 / SWIGLU with the DRA-82 fields ymap, x64; pq f32 / bf16 input with stride), mutants 0 only. DRA-84 added
 * qg epi BF16 and qg mutant 5; neither appears here, so tests/gpu_pfq_test.c compares this digest with PFQ_DRA82_DIGEST,
 * the value printed by the same code built against the DRA-84 base (CUDApp #228 721826b merged with #238 61506fa).
 * Needs pfq_legacy_words.h (pfq_encode_hash). */
#ifndef PFQ_DRA82_WORDS_H
#define PFQ_DRA82_WORDS_H
#include "pfq_legacy_words.h"

static int pfq_dra82_digest(uint64_t *out) {
    *out = 0;
    BlackwellIRProgram *prog = calloc(1, sizeof *prog);
    size_t cap = 1u << 20; uint8_t *code = malloc(cap);
    uint64_t h = 0xcbf29ce484222325ull; uint32_t idx = 0; int bad = 0;
    if (!prog || !code) { free(prog); free(code); return -1; }
    /* the q36 prefill shapes (omega_gpu_q36_graph.c prepare list) and the pfq test shapes */
    static const struct { uint32_t n, k; } nk[] = { { 12288, 2048 }, { 9216, 2048 }, { 2048, 4096 }, { 1024, 2048 }, { 2048, 512 },
                                                     { 256, 384 }, { 256, 128 }, { 256, 2048 }, { 256, 4096 } };
    for (unsigned a = 0; a < sizeof nk / sizeof nk[0]; a++) for (uint32_t epi = 0; epi <= 1; epi++) for (uint32_t dir = 0; dir <= 1; dir++)
    for (uint32_t ym = 0; ym <= 1; ym++) for (uint32_t x = 0; x <= 1; x++) {
        OmegaQgSpec s; memset(&s, 0, sizeof s);
        s.k = nk[a].k; s.rt = 4; s.warps = 8; s.epi = epi; s.direct = dir; s.n = nk[a].n; s.wait = 45; s.ymap = ym; s.x64 = x;
        if (!omega_qg_spec_ok(&s)) continue;
        memset(prog, 0, sizeof *prog);
        if (omega_blackwell_codegen_qgemm_prog(&s, prog) != 0 || pfq_encode_hash(prog, code, cap, &h, idx) != 0) bad = 1;
        idx++;
    }
    static const struct { uint32_t k, stride; } ks[] = { { 2048, 0 }, { 2048, 2048 + 68 }, { 2048, 3080 }, { 4096, 0 }, { 512, 0 }, { 5120, 5248 },
                                                         { 8192, 0 } };
    for (unsigned a = 0; a < sizeof ks / sizeof ks[0]; a++) for (uint32_t it = 0; it <= 1; it++) {
        OmegaPqSpec s; memset(&s, 0, sizeof s);
        s.k = ks[a].k; s.in_type = it; s.stride = ks[a].stride;
        const uint32_t kb = s.k / 128u;
        if (kb > 32u) for (s.groups = 32u; kb % s.groups; s.groups--) {}
        if (!omega_pq_spec_ok(&s)) continue;
        memset(prog, 0, sizeof *prog);
        if (omega_blackwell_codegen_pq_prog(&s, prog) != 0 || pfq_encode_hash(prog, code, cap, &h, idx) != 0) bad = 1;
        idx++;
    }
    free(prog); free(code);
    *out = h;
    return bad ? -1 : (int)idx;
}
#endif
