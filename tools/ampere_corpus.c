/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Test-only IR capture. Linked into the existing host-only corpus producers;
 * no driver API is called here. A refusal is legal only for an explicit FP8 marker. */
#include "omega_ampere_codegen.h"
#include "omega_blackwell_codegen_tc.h"
#include "omega_blackwell_codegen_qmma.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void save(const char *dir, unsigned serial, const char *suffix, const void *data, size_t size)
{
    char path[4096];
    int n = snprintf(path, sizeof(path), "%s/%ld-%u.%s", dir, (long)getpid(), serial, suffix);
    if (n < 0 || (size_t)n >= sizeof(path)) abort();
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); abort(); }
    if (fwrite(data, 1, size, f) != size || fclose(f) != 0) abort();
}

void omega_ampere_corpus_capture(const BlackwellIRProgram *prog, const uint8_t *sm121, size_t len)
{
    const char *dir = getenv("OMEGA_AMPERE_CORPUS_DIR");
    if (!dir || !*dir) return;
    static unsigned serial;
    const unsigned id = serial++;
    uint8_t code[16384], reference[16384];
    uint32_t metadata[BW_MAX_IR_INSNS][2];
    int fp8 = 0;
    for (size_t i = 0; i < prog->count; ++i) {
        const BlackwellIRInsn *in = &prog->insns[i];
        metadata[i][0] = (uint32_t)in->op;
        metadata[i][1] = in->imm;
        if ((in->op == BW_IR_HMMA_BF16 && in->imm == OMEGA_QMMA_MARK) ||
            (in->op == BW_IR_FMNMX_MAX && in->imm == OMEGA_QMMA_MARK_F2E4)) fp8 = 1;
    }
    memset(code, 0xa5, sizeof(code));
    size_t bytes = 0;
    int rc = omega_gpu_encode_program_target(prog, 86, code, sizeof(code), &bytes);
    if (fp8) {
        if (rc != OMEGA_GPU_ENCODE_UNSUPPORTED || bytes != 0) abort();
        for (size_t i = 0; i < sizeof(code); ++i) if (code[i] != 0xa5) abort();
        save(dir, id, "refused", metadata, prog->count * sizeof(metadata[0]));
        return;
    }
    if (rc != 0 || bytes != len) {
        fprintf(stderr, "SM86 unexpectedly refused an existing kernel (%zu instructions, rc %d)\n", prog->count, rc);
        abort();
    }
    memcpy(reference, sm121, len);
    /* The corpus hook runs before family post-encoding patches. Reproduce the
     * published reference patches, not the Ampere implementation under test. */
    if (omega_tc_patch_code(prog, reference, len) < 0) abort();
    save(dir, id, "sm121.bin", reference, len);
    save(dir, id, "sm86.bin", code, bytes);
    save(dir, id, "ir", metadata, prog->count * sizeof(metadata[0]));
}
