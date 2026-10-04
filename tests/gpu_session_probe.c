/* gpu_session_probe: FB-1 cut 4b flake investigation (first-call device-open failure).
 *
 * One fresh process = one sample: open the shared device through the API's own
 * path (omega_gpu_ex2_f32 on 128 values is the smallest launch there is), report
 * the stage that failed (with the driver's own text) or the open time, exit 0/1.
 *
 *   ./gpu_session_probe [--label L] [--calls N] [--hold-ms M]
 *     --calls N    launches after the first (default 1); keeps the device busy for an overlap test
 *     --hold-ms M  sleep M ms with the device open before exit (overlap window for a second probe)
 *
 * Output (one line, Astra's SNAP shape; every unknown says so):
 *   PROBE label=L pid=P rc=OK|CHIP_FAIL|... stage="..." first_call_ms=F open_ms=O calls_ok=K/N opens=1
 * Never run outside the heavy queue (tools/probe_gpu_open.sh).
 */
#include "omega_gpu_elementwise_api.h"
#include "omega_gpu_session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

int main(int argc, char **argv) {
    const char *label = "probe"; int calls = 1, hold_ms = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--label") == 0 && i + 1 < argc) label = argv[++i];
        else if (strcmp(argv[i], "--calls") == 0 && i + 1 < argc) calls = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hold-ms") == 0 && i + 1 < argc) hold_ms = atoi(argv[++i]);
        else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 2; }
    }
    float x[128], y[128];
    for (int i = 0; i < 128; i++) { x[i] = (float)i / 16.0f - 4.0f; y[i] = 0.0f; }
    OmegaGpuEwInfo info; memset(&info, 0, sizeof info);
    double t0 = now_ms();
    int rc = omega_gpu_ex2_f32(128, x, y, &info);
    double first_ms = now_ms() - t0;
    int ok = 0;
    if (rc == OMEGA_GPU_EW_OK) {
        /* negative control on the result itself: 2^0 must be 1 (x[64] = 0) */
        if (y[64] < 0.999f || y[64] > 1.001f) { rc = OMEGA_GPU_EW_UNWRITTEN; omega_gpu_session_set_error("probe: 2^0 != 1"); }
        else ok = 1;
    }
    int more_ok = 0;
    for (int i = 1; i < calls && ok; i++) if (omega_gpu_ex2_f32(128, x, y, &info) == OMEGA_GPU_EW_OK) more_ok++;
    if (hold_ms > 0) usleep((useconds_t)hold_ms * 1000u);
    const char *stage = omega_gpu_elementwise_last_error();
    printf("PROBE label=%s pid=%d rc=%s stage=\"%s\" first_call_ms=%.1f chip_us=%.1f calls_ok=%d/%d opens=%u%s\n",
           label, (int)getpid(), omega_gpu_elementwise_rc_name(rc), stage[0] ? stage : "-", first_ms,
           info.elapsed_ns / 1e3, ok + more_ok, calls, omega_gpu_session_open_count(),
           rc == OMEGA_GPU_EW_OK && !stage[0] ? "" : (rc == OMEGA_GPU_EW_OK ? " note=stage_text_from_earlier_failure" : ""));
    fflush(stdout);
    omega_gpu_session_close(); /* the explicit close is part of what the next process sees */
    return ok ? 0 : 1;
}
