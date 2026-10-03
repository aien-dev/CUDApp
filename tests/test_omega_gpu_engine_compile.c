/* Omega GPU Engine compile check: the header and the A3a core compile, every enum name
 * is distinct, argument and backend-table refusals return their defined results, the
 * uncertain-completion block functions behave as specified, and a no-op backend reaches
 * SUCCESS. */
#include "omega_gpu_engine.h"
#include <stdio.h>
#include <string.h>

static int bad;
#define CHECK(c) do { if (!(c)) { printf("[FAIL] line %d: %s\n", __LINE__, #c); bad++; } } while (0)

static int f_ctx(void *c) { (void)c; return 0; }
static int f_alloc(void *c, int r, size_t n) { (void)c; (void)r; (void)n; return 0; }
static int f_in(void *c, int r, const void *s, size_t n) { (void)c; (void)r; (void)s; (void)n; return 0; }
static int f_poison(void *c, const uint32_t *w, size_t n) { (void)c; (void)w; (void)n; return 0; }
static int f_build(void *c, const OmegaGpuJob *j) { (void)c; (void)j; return 0; }
static int f_wait(void *c, uint64_t ms) { (void)c; (void)ms; return 0; }
static int f_out(void *c, void *d, size_t n) { (void)c; (void)d; (void)n; return 0; }
static void f_diag(void *c, OmegaGpuBackendDiag *d) { (void)c; (void)d; }
static int f_free(void *c, int r) { (void)c; (void)r; return 0; }

static int all_distinct(const char *(*name)(int), int count)
{
    for (int i = 0; i < count; i++) {
        CHECK(name(i) != NULL && strcmp(name(i), "UNKNOWN") != 0);
        for (int j = i + 1; j < count; j++) CHECK(strcmp(name(i), name(j)) != 0);
    }
    return strcmp(name(-1), "UNKNOWN") == 0 && strcmp(name(count), "UNKNOWN") == 0;
}

/* Expects the stub's full defined result for a job refused at `step` with `failure`. */
static void expect_result(const OmegaGpuResult *r, int failure, int step)
{
    CHECK(r->failure == failure);
    CHECK(r->failed_step == step);
    CHECK(r->last_state == OMEGA_GPU_ENGINE_STATE_INITIAL);
    CHECK(r->wait == OMEGA_GPU_ENGINE_WAIT_NONE);
    CHECK(r->wait_timeout_ms == 0 && r->waited_ms == 0);
    CHECK(r->drv_rc == 0 && r->err_no == 0 && r->drv_text[0] == '\0');
    CHECK(r->sync_valid == 0 && r->marker == 0 && r->marker2 == 0 && r->semaphore[0] == 0);
    CHECK(r->retained == 0 && r->cleanup_failed == 0);
    CHECK(r->n_outputs == 0 && r->output_unchanged_words[0] == 0);
}

int main(void)
{
    CHECK(all_distinct(omega_gpu_engine_failure_name, OMEGA_GPU_ENGINE_FAILURE_COUNT));
    CHECK(all_distinct(omega_gpu_engine_state_name, OMEGA_GPU_ENGINE_STATE_COUNT));
    CHECK(all_distinct(omega_gpu_engine_wait_name, OMEGA_GPU_ENGINE_WAIT_COUNT));
    CHECK(all_distinct(omega_gpu_engine_step_name, OMEGA_GPU_ENGINE_STEP_COUNT));
    CHECK(OMEGA_GPU_ENGINE_OK == 0 && strcmp(omega_gpu_engine_failure_name(0), "OK") == 0);
    CHECK(strcmp(omega_gpu_engine_failure_name(OMEGA_GPU_ENGINE_COMPLETION_WAIT), "COMPLETION_WAIT") == 0);
    CHECK(strcmp(omega_gpu_engine_failure_name(OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED),
                 "UNCERTAIN_COMPLETION_BLOCKED") == 0);
    CHECK(strcmp(omega_gpu_engine_state_name(OMEGA_GPU_ENGINE_STATE_INITIAL), "INITIAL") == 0);
    CHECK(strcmp(omega_gpu_engine_state_name(OMEGA_GPU_ENGINE_STATE_SUCCESS), "SUCCESS") == 0);
    CHECK(OMEGA_GPU_ENGINE_STATE_INITIAL == 0 && OMEGA_GPU_ENGINE_STATE_COUNT == 7);
    CHECK(strcmp(omega_gpu_engine_wait_name(OMEGA_GPU_ENGINE_WAIT_RELEASE_SEMAPHORE), "RELEASE_SEMAPHORE") == 0);
    CHECK(strcmp(omega_gpu_engine_step_name(OMEGA_GPU_ENGINE_STEP_NOT_IMPLEMENTED), "NOT_IMPLEMENTED") == 0);

    unsigned char code[4] = {0};
    uint32_t ina[4] = {0}, inb[4] = {0}, outw[4] = {0}, poison[4] = {1, 2, 3, 4};
    OmegaGpuJob job;
    memset(&job, 0, sizeof job);
    job.program = code; job.program_len = sizeof code;
    job.layout = OMEGA_GPU_LAYOUT_VECTOR_1D; job.element_count = 4;
    job.input_a = ina; job.input_a_len = sizeof ina;
    job.input_b = inb; job.input_b_len = sizeof inb;
    job.output = outw; job.output_len = sizeof outw;
    job.poison = poison; job.poison_count = 4;
    job.timeouts.marker_ms = 1; job.timeouts.release_semaphore_ms = 1;
    job.timeouts.marker2_ms = 1; job.timeouts.visibility_ms = 1;

    OmegaGpuResult r;

    /* Not blocked at start; no backend set: the defined stub result. */
    omega_gpu_engine_test_reset_block();
    CHECK(omega_gpu_engine_is_blocked() == 0);
    memset(&r, 0x7f, sizeof r);
    CHECK(omega_gpu_execute(&job, &r) == OMEGA_GPU_ENGINE_INTERNAL_INVARIANT);
    expect_result(&r, OMEGA_GPU_ENGINE_INTERNAL_INVARIANT, OMEGA_GPU_ENGINE_STEP_NOT_IMPLEMENTED);

    /* A missing result pointer is refused and nothing is written. */
    CHECK(omega_gpu_execute(&job, NULL) == OMEGA_GPU_ENGINE_INVALID_ARGS);

    /* Counterexamples: each malformed job is INVALID_ARGS at VALIDATE, never NOT_IMPLEMENTED. */
    memset(&r, 0x7f, sizeof r);
    CHECK(omega_gpu_execute(NULL, &r) == OMEGA_GPU_ENGINE_INVALID_ARGS);
    expect_result(&r, OMEGA_GPU_ENGINE_INVALID_ARGS, OMEGA_GPU_ENGINE_STEP_VALIDATE);

    OmegaGpuJob bad_job;
#define EXPECT_INVALID(edit) do { bad_job = job; edit; memset(&r, 0x7f, sizeof r); \
        CHECK(omega_gpu_execute(&bad_job, &r) == OMEGA_GPU_ENGINE_INVALID_ARGS); \
        expect_result(&r, OMEGA_GPU_ENGINE_INVALID_ARGS, OMEGA_GPU_ENGINE_STEP_VALIDATE); } while (0)
    EXPECT_INVALID(bad_job.program = NULL);
    EXPECT_INVALID(bad_job.program_len = 0);
    EXPECT_INVALID(bad_job.layout = OMEGA_GPU_LAYOUT_COUNT);
    EXPECT_INVALID(bad_job.element_count = 0);
    EXPECT_INVALID(bad_job.input_a = NULL);
    EXPECT_INVALID(bad_job.input_a_len = sizeof ina - 4);
    EXPECT_INVALID(bad_job.input_b = NULL);
    EXPECT_INVALID(bad_job.input_b_len = sizeof inb + 4);
    EXPECT_INVALID(bad_job.output = NULL);
    EXPECT_INVALID(bad_job.output_len = sizeof outw - 4);
    EXPECT_INVALID(bad_job.output = (unsigned char *)outw + 1); /* not 4-byte aligned */
    EXPECT_INVALID(bad_job.poison = NULL);
    EXPECT_INVALID(bad_job.poison_count = 3);
    EXPECT_INVALID(bad_job.flags = 0x2u); /* unknown flag */
    EXPECT_INVALID(bad_job.timeouts.marker_ms = 0);
    EXPECT_INVALID(bad_job.timeouts.release_semaphore_ms = 0);
    EXPECT_INVALID(bad_job.timeouts.visibility_ms = 0);
    EXPECT_INVALID(bad_job.timeouts.marker2_ms = 0); /* C3 on, so marker2 needs a timeout */

    /* With NO_C3 a zero marker2 timeout is allowed. */
    bad_job = job; bad_job.flags = OMEGA_GPU_ENGINE_FLAG_NO_C3; bad_job.timeouts.marker2_ms = 0;
    memset(&r, 0x7f, sizeof r);
    CHECK(omega_gpu_execute(&bad_job, &r) == OMEGA_GPU_ENGINE_INTERNAL_INVARIANT);
    expect_result(&r, OMEGA_GPU_ENGINE_INTERNAL_INVARIANT, OMEGA_GPU_ENGINE_STEP_NOT_IMPLEMENTED);

    /* Backend table: incomplete is refused, complete runs the core to SUCCESS, NULL clears. */
    OmegaGpuBackend be = {
        .ctx = NULL, .open_device = f_ctx, .create_channel = f_ctx, .alloc = f_alloc,
        .copy_in = f_in, .fill_poison = f_poison, .build = f_build, .submit = f_ctx,
        .wait_marker = f_wait, .wait_marker2 = f_wait, .wait_release_semaphore = f_wait,
        .barrier = f_wait, .copy_out = f_out, .diagnostics = f_diag, .free_buf = f_free,
        .close = f_ctx,
    };
    omega_gpu_engine_set_backend(&be);
    /* The A3a core now runs the whole state machine: the all-success no-op backend
     * (copy-out leaves the output zeros, which differ from the poison words 1..4)
     * reaches SUCCESS. The full state machine is tested in tests/test_omega_gpu_engine.c. */
    omega_gpu_engine_test_set_clock(NULL); /* NULL keeps the default clock */
    memset(&r, 0x7f, sizeof r);
    CHECK(omega_gpu_execute(&job, &r) == OMEGA_GPU_ENGINE_OK);
    CHECK(r.failure == OMEGA_GPU_ENGINE_OK && r.failed_step == OMEGA_GPU_ENGINE_STEP_NONE);
    CHECK(r.last_state == OMEGA_GPU_ENGINE_STATE_SUCCESS && r.wait == OMEGA_GPU_ENGINE_WAIT_NONE);
    CHECK(r.retained == 0 && r.cleanup_failed == 0);
    CHECK(r.n_outputs == 1u && r.output_unchanged_words[0] == 0);
    be.close = NULL;
    memset(&r, 0x7f, sizeof r);
    CHECK(omega_gpu_execute(&job, &r) == OMEGA_GPU_ENGINE_INTERNAL_INVARIANT);
    expect_result(&r, OMEGA_GPU_ENGINE_INTERNAL_INVARIANT, OMEGA_GPU_ENGINE_STEP_BACKEND_CHECK);
    omega_gpu_engine_set_backend(NULL);
    memset(&r, 0x7f, sizeof r);
    CHECK(omega_gpu_execute(&job, &r) == OMEGA_GPU_ENGINE_INTERNAL_INVARIANT);
    expect_result(&r, OMEGA_GPU_ENGINE_INTERNAL_INVARIANT, OMEGA_GPU_ENGINE_STEP_NOT_IMPLEMENTED);

    /* Block: forced block refuses every job, even a malformed one, until reset. */
    omega_gpu_engine_test_force_block();
    CHECK(omega_gpu_engine_is_blocked() == 1);
    memset(&r, 0x7f, sizeof r);
    CHECK(omega_gpu_execute(&job, &r) == OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED);
    expect_result(&r, OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED, OMEGA_GPU_ENGINE_STEP_BLOCKED);
    memset(&r, 0x7f, sizeof r);
    CHECK(omega_gpu_execute(NULL, &r) == OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED);
    expect_result(&r, OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED, OMEGA_GPU_ENGINE_STEP_BLOCKED);
    CHECK(omega_gpu_engine_is_blocked() == 1); /* a refusal does not clear the block */
    omega_gpu_engine_test_reset_block();
    CHECK(omega_gpu_engine_is_blocked() == 0);
    memset(&r, 0x7f, sizeof r);
    CHECK(omega_gpu_execute(&job, &r) == OMEGA_GPU_ENGINE_INTERNAL_INVARIANT);
    expect_result(&r, OMEGA_GPU_ENGINE_INTERNAL_INVARIANT, OMEGA_GPU_ENGINE_STEP_NOT_IMPLEMENTED);

    printf("SUMMARY failed=%d\n", bad);
    return bad ? 1 : 0;
}
