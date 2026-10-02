/*
 * Omega GPU Engine, A2b compile stub. See src/omega_gpu_engine.h and
 * docs/numeric/OMEGA_GPU_ENGINE.md. No launch logic yet: omega_gpu_execute
 * checks the block, validates the job, then reports INTERNAL_INVARIANT at step
 * NOT_IMPLEMENTED. The real engine core (A3) replaces omega_gpu_execute and
 * sets the block on uncertain completion; the name functions, the backend
 * pointer and the block query below are final.
 */
#include "omega_gpu_engine.h"

#include <stdatomic.h>
#include <string.h>

static const char *const FAILURE_NAMES[OMEGA_GPU_ENGINE_FAILURE_COUNT] = {
    "OK",
    "DEVICE_OPEN",
    "CHANNEL_CREATE",
    "ALLOC",
    "PREPARE",
    "SUBMIT",
    "COMPLETION_WAIT",
    "RELEASE_WAIT",
    "VISIBILITY_WAIT",
    "OUTPUT_UNCHANGED",
    "INVALID_ARGS",
    "INTERNAL_INVARIANT",
    "CLEANUP",
    "UNCERTAIN_COMPLETION_BLOCKED",
};

static const char *const STATE_NAMES[OMEGA_GPU_ENGINE_STATE_COUNT] = {
    "INITIAL",
    "PREPARED",
    "SUBMITTED",
    "GPU_COMPLETE",
    "OUTPUT_VISIBLE",
    "OUTPUT_PRODUCED",
    "SUCCESS",
};

static const char *const WAIT_NAMES[OMEGA_GPU_ENGINE_WAIT_COUNT] = {
    "NONE",
    "MARKER",
    "RELEASE_SEMAPHORE",
    "MARKER2",
};

static const char *const STEP_NAMES[OMEGA_GPU_ENGINE_STEP_COUNT] = {
    "NONE",
    "VALIDATE",
    "BACKEND_CHECK",
    "BLOCKED",
    "NOT_IMPLEMENTED",
    "OPEN_DEVICE",
    "CREATE_CHANNEL",
    "ALLOC",
    "COPY_IN",
    "FILL_POISON",
    "BUILD",
    "SUBMIT",
    "WAIT_MARKER",
    "WAIT_MARKER2",
    "WAIT_RELEASE_SEMAPHORE",
    "BARRIER",
    "COPY_OUT",
    "SCAN",
    "FREE",
    "CLOSE",
};

static _Atomic(const OmegaGpuBackend *) g_backend;
static atomic_int g_blocked;

const char *omega_gpu_engine_failure_name(int failure)
{
    if (failure < 0 || failure >= (int)OMEGA_GPU_ENGINE_FAILURE_COUNT) return "UNKNOWN";
    return FAILURE_NAMES[failure];
}

const char *omega_gpu_engine_state_name(int state)
{
    if (state < 0 || state >= (int)OMEGA_GPU_ENGINE_STATE_COUNT) return "UNKNOWN";
    return STATE_NAMES[state];
}

const char *omega_gpu_engine_wait_name(int wait)
{
    if (wait < 0 || wait >= (int)OMEGA_GPU_ENGINE_WAIT_COUNT) return "UNKNOWN";
    return WAIT_NAMES[wait];
}

const char *omega_gpu_engine_step_name(int step)
{
    if (step < 0 || step >= (int)OMEGA_GPU_ENGINE_STEP_COUNT) return "UNKNOWN";
    return STEP_NAMES[step];
}

void omega_gpu_engine_set_backend(const OmegaGpuBackend *backend)
{
    atomic_store(&g_backend, backend);
}

int omega_gpu_engine_is_blocked(void)
{
    return atomic_load(&g_blocked) ? 1 : 0;
}

void omega_gpu_engine_test_reset_block(void)
{
    atomic_store(&g_blocked, 0);
}

void omega_gpu_engine_test_force_block(void)
{
    atomic_store(&g_blocked, 1);
}

static int backend_complete(const OmegaGpuBackend *b)
{
    return b->open_device && b->create_channel && b->alloc && b->copy_in && b->fill_poison &&
           b->build && b->submit && b->wait_marker && b->wait_marker2 &&
           b->wait_release_semaphore && b->barrier && b->copy_out && b->diagnostics &&
           b->free_buf && b->close;
}

/* Returns 1 when the job is well formed for the VECTOR_1D layout. */
static int job_valid(const OmegaGpuJob *job)
{
    uint64_t bytes;
    if (!job->program || job->program_len == 0) return 0;
    if (job->layout != OMEGA_GPU_LAYOUT_VECTOR_1D) return 0;
    if (job->element_count == 0) return 0;
    bytes = (uint64_t)job->element_count * 4u;
    if (!job->input_a || (uint64_t)job->input_a_len != bytes) return 0;
    if (!job->input_b || (uint64_t)job->input_b_len != bytes) return 0;
    if (!job->output || (uint64_t)job->output_len != bytes) return 0;
    if (((uintptr_t)job->output & 3u) != 0) return 0;
    if (!job->poison || (uint64_t)job->poison_count != (uint64_t)job->element_count) return 0;
    if ((job->flags & ~(uint32_t)OMEGA_GPU_ENGINE_FLAG_NO_C3) != 0) return 0;
    if (job->timeouts.marker_ms == 0) return 0;
    if (job->timeouts.release_semaphore_ms == 0) return 0;
    if (job->timeouts.visibility_ms == 0) return 0;
    if ((job->flags & OMEGA_GPU_ENGINE_FLAG_NO_C3) == 0 && job->timeouts.marker2_ms == 0) return 0;
    return 1;
}

static int stop(OmegaGpuResult *result, int failure, int step)
{
    result->failure = failure;
    result->failed_step = step;
    return failure;
}

int omega_gpu_execute(const OmegaGpuJob *job, OmegaGpuResult *result)
{
    const OmegaGpuBackend *b;
    if (!result) return OMEGA_GPU_ENGINE_INVALID_ARGS;
    memset(result, 0, sizeof *result);
    result->failure = OMEGA_GPU_ENGINE_OK;
    result->last_state = OMEGA_GPU_ENGINE_STATE_INITIAL;
    result->failed_step = OMEGA_GPU_ENGINE_STEP_NONE;
    result->wait = OMEGA_GPU_ENGINE_WAIT_NONE;
    if (omega_gpu_engine_is_blocked())
        return stop(result, OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED,
                    OMEGA_GPU_ENGINE_STEP_BLOCKED);
    if (!job || !job_valid(job))
        return stop(result, OMEGA_GPU_ENGINE_INVALID_ARGS, OMEGA_GPU_ENGINE_STEP_VALIDATE);
    b = atomic_load(&g_backend);
    if (b && !backend_complete(b))
        return stop(result, OMEGA_GPU_ENGINE_INTERNAL_INVARIANT,
                    OMEGA_GPU_ENGINE_STEP_BACKEND_CHECK);
    return stop(result, OMEGA_GPU_ENGINE_INTERNAL_INVARIANT, OMEGA_GPU_ENGINE_STEP_NOT_IMPLEMENTED);
}
