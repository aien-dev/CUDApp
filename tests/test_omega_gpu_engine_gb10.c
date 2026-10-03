/*
 * Phase B only: needs the GB10, queen's go, quiet flag via chip_run; never run in Phase A.
 *
 * Chip test of the Omega GPU Engine (docs/numeric/OMEGA_GPU_ENGINE.md), cut A3b2.
 * Phase A only COMPILES it (make test-gpu-engine-gb10-compile). It was written
 * against the real API (src/omega_gpu_engine.h, src/omega_blackwell_submit.h) and
 * never compiled or executed by its author. It is a fresh rewrite: the first-cut
 * test by GPT-6 Astra (commit 21392ee on fix/omega-gpu-engine-first-cut) used an
 * older API, so it served as the outline only.
 *
 * What it does with --chip on the GB10, through tools/chip_run.sh and
 * tools/manifests/gpu_engine.chiprun:
 *   1. Runs omega_blackwell_execute_vector (engine core + Blackwell backend + the
 *      real driver) for element counts 1, 63, 64, 65, 4096 and 65537, ten times
 *      each. One extra pattern per size uses a[i] = 0xdeadbeef, b[i] = 0 (a sum
 *      that equals the old constant poison word, which must not fool the engine).
 *   2. Negative control: the vector program with its one store instruction
 *      replaced by the instruction after it (an EXIT), so no lane writes. The
 *      engine must report OUTPUT_UNCHANGED for all 65 words, after the run
 *      completed and the output became visible, with nothing retained and no
 *      cleanup failure. The offsets are the fixture locations in
 *      src/omega_blackwell_encoder.c (0x120 store, 0x130 exit).
 * Verdict lines (the only lines chip_run reads): "GPU_ENGINE: PASS" or
 * "GPU_ENGINE: FAIL"; without --chip it prints "GPU_ENGINE: NOT_RUN ..." and
 * exits 2. An uncertain completion blocks the process by design: this test then
 * parks instead of exiting, because exiting would close the device under a GPU
 * that may still be running.
 */
#include "omega_gpu_engine.h"
#include "omega_blackwell_engine.h"
#include "omega_blackwell_submit.h"
#include "omega_blackwell_encoder.h"
#include "omega_blackwell_realize.h"
#include "omega_vector.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define REPEATS 10u

/* An uncertain completion leaves the context and the device memory alive on
 * purpose. Never exit in that state. */
static void park_if_blocked(void)
{
    if (!omega_gpu_engine_is_blocked()) return;
    fprintf(stderr, "GPU_ENGINE_BLOCKED uncertain completion, parking (do not close the device)\n");
    puts("GPU_ENGINE: FAIL");
    fflush(stdout);
    for (;;) pause();
}

static int run_size(uint32_t n, int deadbeef)
{
    OmegaVectorSpec spec;
    OmegaBlackwellRealization real;
    OmegaBlackwellVectorExecution info;
    uint32_t *a = (uint32_t *)malloc((size_t)n * 4u);
    uint32_t *b = (uint32_t *)malloc((size_t)n * 4u);
    uint32_t *c = (uint32_t *)malloc((size_t)n * 4u);
    uint32_t i;
    unsigned rep;
    int ok = (a && b && c);

    memset(&spec, 0, sizeof spec);
    memset(&real, 0, sizeof real);
    real.sm_architecture = OMEGA_BW_SM_ARCH_121;
    if (ok && omega_vector_spec_init(&spec, "engine_chip", n) != 0) ok = 0;
    for (i = 0; ok && i < n; i++) {
        a[i] = deadbeef ? 0xdeadbeefu : (uint32_t)(i * 2654435761u + 1u);
        b[i] = deadbeef ? 0u : (uint32_t)(i * 40503u + 7u);
    }
    for (rep = 0; ok && rep < REPEATS; rep++) {
        int rc;
        memset(c, 0, (size_t)n * 4u);
        memset(&info, 0, sizeof info);
        rc = omega_blackwell_execute_vector(&spec, &real, a, b, c, &info);
        park_if_blocked();
        if (rc != 0 || !info.parity_verified) {
            fprintf(stderr, "vector run failed n=%u deadbeef=%d repeat=%u rc=%d\n", n, deadbeef, rep, rc);
            ok = 0;
        }
    }
    free(a);
    free(b);
    free(c);
    return ok;
}

static int no_store_control(void)
{
    uint8_t code[OMEGA_BW_VECADD_CODE_SIZE];
    size_t code_bytes = 0;
    uint32_t a[65], b[65], out[65], poison[65];
    OmegaGpuJob job;
    OmegaGpuResult res;
    uint32_t i;
    int rc;

    if (omega_blackwell_verify_encoder_fixtures() != 0) return 0;
    if (omega_blackwell_encode_vecadd(code, sizeof code, &code_bytes) != 0 || code_bytes < 0x140u) return 0;
    /* Replace only the store with the exit that follows it. */
    memcpy(code + 0x120, code + 0x130, OMEGA_BW_INSTRUCTION_BYTES);
    for (i = 0; i < 65u; i++) {
        a[i] = i + 1u;
        b[i] = 3u * i;
        poison[i] = ~(a[i] + b[i]);
    }
    memset(out, 0, sizeof out);
    memset(&job, 0, sizeof job);
    job.program = code;
    job.program_len = code_bytes;
    job.layout = OMEGA_GPU_LAYOUT_VECTOR_1D;
    job.element_count = 65u;
    job.input_a = a;
    job.input_a_len = sizeof a;
    job.input_b = b;
    job.input_b_len = sizeof b;
    job.output = out;
    job.output_len = sizeof out;
    job.poison = poison;
    job.poison_count = 65u;
    job.timeouts.marker_ms = 5000;
    job.timeouts.release_semaphore_ms = 5000;
    job.timeouts.marker2_ms = 5000;
    job.timeouts.visibility_ms = 5000;
    job.flags = 0;

    omega_gpu_engine_set_backend(omega_blackwell_engine_backend());
    rc = omega_gpu_execute(&job, &res);
    park_if_blocked();
    if (rc != OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED || res.last_state != OMEGA_GPU_ENGINE_STATE_OUTPUT_VISIBLE ||
        res.n_outputs != 1u || res.output_unchanged_words[0] != 65u || res.cleanup_failed != 0 ||
        res.retained != 0) {
        fprintf(stderr, "no-store control failed rc=%d state=%d unchanged=%llu cleanup_failed=%d retained=%d\n",
                rc, res.last_state, (unsigned long long)res.output_unchanged_words[0],
                res.cleanup_failed, res.retained);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    static const uint32_t sizes[] = {1u, 63u, 64u, 65u, 4096u, 65537u};
    size_t s;
    unsigned runs = 0;
    int failed = 0;

    if (argc < 2 || strcmp(argv[1], "--chip") != 0) {
        puts("GPU_ENGINE: NOT_RUN (pass --chip; needs the GB10, via tools/chip_run.sh)");
        return 2;
    }
    for (s = 0; !failed && s < sizeof sizes / sizeof sizes[0]; s++) {
        if (!run_size(sizes[s], 0) || !run_size(sizes[s], 1)) failed = 1;
        else runs += 2u * REPEATS;
    }
    if (!failed && !no_store_control()) failed = 1;
    if (failed) {
        puts("GPU_ENGINE: FAIL");
        return 1;
    }
    printf("GPU_ENGINE_CASES vector_runs=%u no_store_control=detected\n", runs);
    puts("GPU_ENGINE: PASS");
    return 0;
}
