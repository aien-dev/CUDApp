/*
 * Omega GPU Engine: run one already-built GPU kernel job and return SUCCESS or
 * exactly one named failure.
 *
 * Purpose: today several launcher files repeat the same open / alloc / copy-in /
 * pushbuffer / submit / completion-wait / readback / close sequence with three
 * different error styles. This module owns that sequence once. Design:
 * docs/numeric/OMEGA_GPU_ENGINE.md (the first-cut plan is Drake's, 2026-10-02).
 *
 * Public entry point:
 *     int omega_gpu_execute(const OmegaGpuJob *job, OmegaGpuResult *result);
 * It returns OMEGA_GPU_ENGINE_OK (0) on success, otherwise the failure code,
 * which is also stored in result->failure.
 *
 * States: INITIAL -> PREPARED -> SUBMITTED -> GPU_COMPLETE -> OUTPUT_VISIBLE ->
 * OUTPUT_PRODUCED -> SUCCESS. The result records the last completed state and
 * the precise failed step. (The earlier draft of this header called the final
 * state COMMITTED and the first one NONE.)
 *
 * Poison: the wrapper supplies one poison word per output word. The engine
 * writes those words to the device output before submit and, after readback,
 * checks only whether any word is still equal to its poison word. The engine
 * never chooses poison and never judges whether the math is right.
 *
 * Uncertain completion: if the engine cannot tell whether the GPU finished (a
 * completion wait timed out, or the device returned an unknown state), it keeps
 * the device context and every allocation (no free, no close), records that in
 * the result, and blocks every later omega_gpu_execute call in this process,
 * which then returns UNCERTAIN_COMPLETION_BLOCKED.
 *
 * STATUS: A3a engine core. omega_gpu_execute drives the backend table below
 * through the whole state machine; only a fake backend exists (host tests).
 * With no backend set it still returns INTERNAL_INVARIANT at step
 * NOT_IMPLEMENTED. The real Blackwell backend is a later cut (A3b). This cut is
 * not chip-qualified (host NOT_RUN, chip NOT_RUN).
 */
#ifndef OMEGA_GPU_ENGINE_H
#define OMEGA_GPU_ENGINE_H

#include <stddef.h>
#include <stdint.h>

/* One failure name per run. OK is 0, so the return value reads as a boolean. */
typedef enum {
    OMEGA_GPU_ENGINE_OK = 0,
    OMEGA_GPU_ENGINE_DEVICE_OPEN,        /* device could not be opened */
    OMEGA_GPU_ENGINE_CHANNEL_CREATE,     /* channel could not be created */
    OMEGA_GPU_ENGINE_ALLOC,              /* a device buffer could not be allocated */
    OMEGA_GPU_ENGINE_PREPARE,            /* copy-in, poison fill or descriptor build failed */
    OMEGA_GPU_ENGINE_SUBMIT,             /* submitting the work failed */
    OMEGA_GPU_ENGINE_COMPLETION_WAIT,    /* marker or marker2 wait failed or timed out */
    OMEGA_GPU_ENGINE_RELEASE_WAIT,       /* release semaphore wait failed or timed out */
    OMEGA_GPU_ENGINE_VISIBILITY_WAIT,    /* barrier or readback failed or timed out */
    OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED,   /* after readback, output words still equal their poison */
    OMEGA_GPU_ENGINE_INVALID_ARGS,       /* the job or the result pointer is malformed */
    OMEGA_GPU_ENGINE_INTERNAL_INVARIANT, /* the engine or its backend broke its own contract */
    OMEGA_GPU_ENGINE_CLEANUP,            /* free or close failed after an otherwise good run */
    OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED, /* an earlier job left completion uncertain */
    OMEGA_GPU_ENGINE_FAILURE_COUNT       /* not a failure: number of names above */
} OmegaGpuEngineFailure;

/* How far a run got. Success is only SUCCESS. */
typedef enum {
    OMEGA_GPU_ENGINE_STATE_INITIAL = 0, /* nothing completed yet (device may be unopened) */
    OMEGA_GPU_ENGINE_STATE_PREPARED,
    OMEGA_GPU_ENGINE_STATE_SUBMITTED,
    OMEGA_GPU_ENGINE_STATE_GPU_COMPLETE,
    OMEGA_GPU_ENGINE_STATE_OUTPUT_VISIBLE,
    OMEGA_GPU_ENGINE_STATE_OUTPUT_PRODUCED,
    OMEGA_GPU_ENGINE_STATE_SUCCESS,
    OMEGA_GPU_ENGINE_STATE_COUNT /* not a state: number of names above */
} OmegaGpuEngineState;

/* Which completion wait failed (meaningful for COMPLETION_WAIT and RELEASE_WAIT). */
typedef enum {
    OMEGA_GPU_ENGINE_WAIT_NONE = 0,
    OMEGA_GPU_ENGINE_WAIT_MARKER,
    OMEGA_GPU_ENGINE_WAIT_RELEASE_SEMAPHORE,
    OMEGA_GPU_ENGINE_WAIT_MARKER2,
    OMEGA_GPU_ENGINE_WAIT_COUNT /* not a wait: number of names above */
} OmegaGpuEngineWait;

/* The precise step that failed (the step the run was in when it stopped). */
typedef enum {
    OMEGA_GPU_ENGINE_STEP_NONE = 0, /* no step failed (success) */
    OMEGA_GPU_ENGINE_STEP_VALIDATE,
    OMEGA_GPU_ENGINE_STEP_BACKEND_CHECK,
    OMEGA_GPU_ENGINE_STEP_BLOCKED,
    OMEGA_GPU_ENGINE_STEP_NOT_IMPLEMENTED,
    OMEGA_GPU_ENGINE_STEP_OPEN_DEVICE,
    OMEGA_GPU_ENGINE_STEP_CREATE_CHANNEL,
    OMEGA_GPU_ENGINE_STEP_ALLOC,
    OMEGA_GPU_ENGINE_STEP_COPY_IN,
    OMEGA_GPU_ENGINE_STEP_FILL_POISON,
    OMEGA_GPU_ENGINE_STEP_BUILD,
    OMEGA_GPU_ENGINE_STEP_SUBMIT,
    OMEGA_GPU_ENGINE_STEP_WAIT_MARKER,
    OMEGA_GPU_ENGINE_STEP_WAIT_MARKER2,
    OMEGA_GPU_ENGINE_STEP_WAIT_RELEASE_SEMAPHORE,
    OMEGA_GPU_ENGINE_STEP_BARRIER,
    OMEGA_GPU_ENGINE_STEP_COPY_OUT,
    OMEGA_GPU_ENGINE_STEP_SCAN,
    OMEGA_GPU_ENGINE_STEP_FREE,
    OMEGA_GPU_ENGINE_STEP_CLOSE,
    OMEGA_GPU_ENGINE_STEP_COUNT /* not a step: number of names above */
} OmegaGpuEngineStep;

/* Argument layouts the engine accepts. Later layouts are separate qualified cuts. */
typedef enum {
    OMEGA_GPU_LAYOUT_VECTOR_1D = 0, /* two inputs, one output, element_count 32-bit words each */
    OMEGA_GPU_LAYOUT_COUNT          /* not a layout: number of layouts above */
} OmegaGpuLayout;

/* Job flags. A zero flags word means every protection ON. */
#define OMEGA_GPU_ENGINE_FLAG_NO_C3 0x1u /* A/B control arm only: skip L2 flush and marker2 */

#define OMEGA_GPU_ENGINE_JOB_OUTPUTS 1u        /* VECTOR_1D has exactly one output buffer */
#define OMEGA_GPU_ENGINE_SEMAPHORE_WORDS 1u    /* the vector path uses one semaphore word */
#define OMEGA_GPU_ENGINE_DRV_TEXT_BYTES 128u

/* Per-step timeouts in milliseconds. Never hard-coded by the engine. Each wait
 * that is enabled needs a value greater than 0 (marker2_ms is ignored with NO_C3). */
typedef struct {
    uint64_t marker_ms;
    uint64_t release_semaphore_ms;
    uint64_t marker2_ms;
    uint64_t visibility_ms;
} OmegaGpuTimeouts;

typedef struct {
    const void *program;      /* encoded program bytes, already built by the caller */
    size_t program_len;       /* bytes, > 0 */
    int layout;               /* OmegaGpuLayout; only VECTOR_1D today */
    uint32_t element_count;   /* > 0; each buffer below holds element_count 32-bit words */
    const void *input_a;      /* element_count * 4 bytes */
    size_t input_a_len;
    const void *input_b;      /* element_count * 4 bytes */
    size_t input_b_len;
    void *output;             /* host destination, 4-byte aligned, element_count * 4 bytes */
    size_t output_len;
    const uint32_t *poison;   /* wrapper-supplied, one word per output word */
    size_t poison_count;      /* must equal element_count */
    OmegaGpuTimeouts timeouts;
    uint32_t flags;           /* OMEGA_GPU_ENGINE_FLAG_* */
} OmegaGpuJob;

typedef struct {
    int failure;              /* OmegaGpuEngineFailure, OK on success */
    int last_state;           /* last COMPLETED OmegaGpuEngineState */
    int failed_step;          /* OmegaGpuEngineStep, NONE on success */
    int wait;                 /* OmegaGpuEngineWait, NONE unless a completion or release wait failed */
    uint64_t wait_timeout_ms; /* the configured limit of that wait */
    uint64_t waited_ms;       /* how long the engine actually waited before giving up */
    int drv_rc;               /* driver return code of the failing call, 0 if none */
    int err_no;               /* errno captured right after the failing call, 0 if none */
    char drv_text[OMEGA_GPU_ENGINE_DRV_TEXT_BYTES]; /* driver message, NUL terminated, "" if none */
    uint32_t sync_valid;      /* OMEGA_GPU_SYNC_* mask: which observed values below are meaningful */
    uint32_t marker;          /* observed first marker word */
    uint32_t marker2;         /* observed second marker word */
    uint32_t semaphore[OMEGA_GPU_ENGINE_SEMAPHORE_WORDS]; /* observed semaphore words */
    int retained;             /* 1 when context and allocations were kept (uncertain completion) */
    int cleanup_failed;       /* 1 when free or close failed (first failure is never overwritten) */
    size_t n_outputs;         /* number of valid entries in output_unchanged_words */
    uint64_t output_unchanged_words[OMEGA_GPU_ENGINE_JOB_OUTPUTS]; /* words still equal to poison, per output */
} OmegaGpuResult;

#define OMEGA_GPU_SYNC_MARKER 0x1u
#define OMEGA_GPU_SYNC_MARKER2 0x2u
#define OMEGA_GPU_SYNC_SEMAPHORE 0x4u

/*
 * Backend seam. A fake backend implements these to drive every transition in
 * host tests; the Blackwell GB10 backend (src/omega_blackwell_engine.c, later
 * cut) wraps the m16_native_* calls the launchers use today. All callbacks
 * must be non-NULL. Every callback returns 0 on success. The three wait
 * callbacks and the barrier may also return:
 */
#define OMEGA_GPU_BACKEND_TIMEOUT 1       /* the wait ran out of time */
#define OMEGA_GPU_BACKEND_UNKNOWN_STATE 2 /* the device reported a state the backend cannot classify */
/* Any other nonzero value is a driver fault; the backend reports its raw driver code via diagnostics. */

typedef enum {
    OMEGA_GPU_BUF_SCRATCH = 0, /* backend-internal: pushbuffer, marker page, QMD */
    OMEGA_GPU_BUF_PROGRAM,
    OMEGA_GPU_BUF_INPUT_A,
    OMEGA_GPU_BUF_INPUT_B,
    OMEGA_GPU_BUF_OUTPUT,
    OMEGA_GPU_BUF_ROLE_COUNT /* not a role: number of roles above */
} OmegaGpuBufRole;

typedef struct {
    int drv_rc;
    char drv_text[OMEGA_GPU_ENGINE_DRV_TEXT_BYTES];
    uint32_t sync_valid; /* OMEGA_GPU_SYNC_* */
    uint32_t marker;
    uint32_t marker2;
    uint32_t semaphore[OMEGA_GPU_ENGINE_SEMAPHORE_WORDS];
} OmegaGpuBackendDiag;

typedef struct {
    void *ctx; /* passed to every callback; owned by the backend */
    int (*open_device)(void *ctx);
    int (*create_channel)(void *ctx);
    int (*alloc)(void *ctx, int role, size_t bytes); /* SCRATCH gets bytes == 0: the backend sizes it */
    int (*copy_in)(void *ctx, int role, const void *src, size_t bytes);
    int (*fill_poison)(void *ctx, const uint32_t *words, size_t count); /* device-side output pre-fill */
    int (*build)(void *ctx, const OmegaGpuJob *job);                    /* descriptors, QMD, pushbuffer */
    int (*submit)(void *ctx);
    int (*wait_marker)(void *ctx, uint64_t timeout_ms);
    int (*wait_marker2)(void *ctx, uint64_t timeout_ms);
    int (*wait_release_semaphore)(void *ctx, uint64_t timeout_ms);
    int (*barrier)(void *ctx, uint64_t timeout_ms);                     /* visibility before readback */
    int (*copy_out)(void *ctx, void *dst, size_t bytes);
    void (*diagnostics)(void *ctx, OmegaGpuBackendDiag *diag);          /* driver rc, text, observed words */
    int (*free_buf)(void *ctx, int role);
    int (*close)(void *ctx);
} OmegaGpuBackend;

/* Runs one job on the process-wide default backend. See the header comment. */
int omega_gpu_execute(const OmegaGpuJob *job, OmegaGpuResult *result);

/* Sets the process-wide default backend (NULL clears it). The pointer must stay
 * valid while it is set. With none set, omega_gpu_execute returns the defined
 * stub result: INTERNAL_INVARIANT at step NOT_IMPLEMENTED. Real code sets it;
 * host tests set a fake. */
void omega_gpu_engine_set_backend(const OmegaGpuBackend *backend);

/* 1 once an uncertain completion has blocked this process, else 0. */
int omega_gpu_engine_is_blocked(void);

/* TEST ONLY. Clears the block. Production code must never call this: a blocked
 * process stays blocked because the GPU may still be running. */
void omega_gpu_engine_test_reset_block(void);

/* TEST ONLY. Forces the blocked state so tests can exercise the blocked
 * result without a real uncertain completion. */
void omega_gpu_engine_test_force_block(void);

/* TEST ONLY. Replaces the millisecond clock the engine uses to measure
 * waited_ms (default: CLOCK_MONOTONIC). NULL restores the default. Not thread
 * safe: set it before any job runs. Lets host tests assert exact waited_ms. */
void omega_gpu_engine_test_set_clock(uint64_t (*now_ms)(void));

/* Static name strings, never NULL. Out-of-range values give "UNKNOWN". */
const char *omega_gpu_engine_failure_name(int failure);
const char *omega_gpu_engine_state_name(int state);
const char *omega_gpu_engine_wait_name(int wait);
const char *omega_gpu_engine_step_name(int step);

#endif
