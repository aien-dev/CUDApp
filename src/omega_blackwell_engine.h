/*
 * Omega GPU Engine, Blackwell GB10 backend (cut A3b1).
 *
 * This module is ONLY the device side of the engine. The public entry point is
 * omega_gpu_execute in src/omega_gpu_engine.c (the A3a core); this file supplies
 * the backend function table that the core drives (OmegaGpuBackend, declared in
 * src/omega_gpu_engine.h). It does not define any job, result or execute type.
 *
 * Origin: the device sequence (open, channel, allocations, descriptor and QMD
 * build, pushbuffer, submit, waits, copy-out, close) was seeded from the first
 * cut by GPT-6 Astra (branch fix/omega-gpu-engine-first-cut, commit 21392ee,
 * src/omega_blackwell_engine.c) and from omega_blackwell_execute_vector as it
 * stood at 676f16f. See docs/numeric/OMEGA_GPU_ENGINE.md, section Origin.
 *
 * Usage:
 *     omega_gpu_engine_set_backend(omega_blackwell_engine_backend());
 *     rc = omega_gpu_execute(&job, &result);
 * Only the VECTOR_1D layout (two inputs, one output, 64 threads per block) is
 * supported. One run at a time: the backend keeps one process-wide context and
 * a second open while a run is active is refused as DEVICE_OPEN.
 *
 * Observed values on success: the core only copies driver diagnostics on a
 * failure, and the core frees and closes the device (which unmaps the marker
 * and semaphore memory) before it returns. So the backend snapshots what it saw
 * inside its own wait callbacks, and omega_blackwell_engine_run_info returns
 * that snapshot after a run. The vector wrapper in omega_blackwell_submit.c
 * uses it to fill completion_marker, intermediate_semaphore and the timestamps.
 *
 * STATUS: DRAFT. Host NOT_RUN, chip NOT_RUN. Nothing here has been compiled or
 * executed by its author.
 */
#ifndef OMEGA_BLACKWELL_ENGINE_H
#define OMEGA_BLACKWELL_ENGINE_H

#include "omega_gpu_engine.h"

#include <stdint.h>

/* What the backend observed during the most recent run (valid after the run). */
typedef struct {
    uint64_t launch_ns;      /* CLOCK_MONOTONIC, just before the pushbuffer submit; 0 if never submitted */
    uint64_t marker_done_ns; /* CLOCK_MONOTONIC, just after the first marker wait succeeded; 0 if it did not */
    uint32_t marker;         /* first marker word as observed by the wait (0x44444444 on success) */
    uint32_t marker2;        /* second marker word as observed (0x46464646 on success, 0 with NO_C3) */
    uint32_t semaphore;      /* QMD release semaphore word as observed (6 on success) */
} OmegaBlackwellEngineRunInfo;

/* The process-wide Blackwell backend table. Never NULL. The ctx inside it is
 * the backend's static context; callers must not touch it. */
const OmegaGpuBackend *omega_blackwell_engine_backend(void);

/* Copies the snapshot of the most recent run into *out. Reset at every open. */
void omega_blackwell_engine_run_info(OmegaBlackwellEngineRunInfo *out);

#endif
