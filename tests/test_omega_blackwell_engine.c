/*
 * Host test of the REAL Blackwell backend (src/omega_blackwell_engine.c), the
 * REAL engine core (src/omega_gpu_engine.c) and the REAL vector wrapper
 * omega_blackwell_execute_vector (src/omega_blackwell_submit.c), driven through
 * a SIMULATED driver (tests/fake_m16_native.c). Cut A3b2. NOT_RUN: this file has
 * never been compiled or executed by its author (Phase A rule).
 *
 * No GPU, no device node, no physics C sources: the seven driver symbols the
 * backend calls are replaced by the fake. Every check prints "[PASS] id" or
 * "[FAIL] id"; the last line is "SUMMARY passed=N failed=M"; the exit status is
 * nonzero on any failure. docs/numeric/OMEGA_GPU_ENGINE.md section 14.1 lists
 * every id and the mutants that need it.
 *
 * Techniques:
 *   - A tracing copy of the real backend table logs every callback (so the call
 *     order the core and the backend produce together can be asserted) and
 *     forwards to the original. The real table is never edited.
 *   - The fake parses and executes the pushbuffer, so the descriptors, the
 *     release packets and the L2 flush are checked as the device would see them.
 *   - A retained run (uncertain completion) leaves the backend busy by design.
 *     The test asserts that, then drains the backend through its own table
 *     (free every role, close), exactly what a process exit would do.
 *
 * Failure names that no driver fault can reach through the real backend, and so
 * are covered by the fake-backend test (tests/test_omega_gpu_engine.c) only:
 *   PREPARE (copy-in and build refuse only on arguments the core already
 *            validated, and alloc_checked catches bad buffers first as ALLOC),
 *   VISIBILITY_WAIT (the barrier always returns 0, copy_out refuses only on bad
 *            arguments),
 *   INTERNAL_INVARIANT (the table is complete).
 * The CPU barrier (dsb sy) has no effect a host can observe; it is chip only.
 */
#include "omega_gpu_engine.h"
#include "omega_blackwell_engine.h"
#include "omega_blackwell_submit.h"
#include "omega_blackwell_encoder.h"
#include "omega_vector.h"
#include "fake_m16_native.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 100u /* not a multiple of 64: a grid rounded down leaves words unwritten */
#define TMO_MARKER 111u
#define TMO_MARKER2 222u
#define TMO_RELEASE 333u
#define TMO_VISIBILITY 444u

#define MARKER1_PAYLOAD 0x44444444u
#define MARKER2_PAYLOAD 0x46464646u
#define SEM_INIT 5u
#define SEM_DONE 6u
#define RELEASE_FLAGS 0x00100001u /* RELEASE | WFI */
#define FLUSH_OP (0x10u << 27)    /* L2_FLUSH_DIRTY */

static int n_pass;
static int n_fail;

static void check(const char *id, int ok)
{
    if (ok) {
        n_pass++;
        printf("[PASS] %s\n", id);
    } else {
        n_fail++;
        printf("[FAIL] %s\n", id);
    }
}

/* ---- test data -------------------------------------------------------------- */

static uint32_t ina[N];
static uint32_t inb[N];
static uint32_t expect_out[N];
static uint32_t poison[N];
static uint32_t outbuf[N];
static uint8_t prog[512];
static uint8_t prog_other[512];
static uint8_t vec_code[0x1000];
static size_t vec_len;
static uint8_t bigbytes[0x2000];
static uint32_t bigwords[2048];

static void init_data(void)
{
    uint32_t i;
    for (i = 0; i < N; i++) {
        uint32_t s;
        ina[i] = i * 0x01000193u + 0x01234567u;
        inb[i] = i * 0x9e3779b1u + 7u;
        s = ina[i] + inb[i];
        /* a wrong-by-one result must never look like the poison word */
        if (s == 0x7fffffffu || s == 0xffffffffu) inb[i] += 0x10u;
    }
    /* the old fixed poison value as a legitimate sum */
    ina[7] = 0xdeadbeefu - 5u;
    inb[7] = 5u;
    for (i = 0; i < N; i++) {
        expect_out[i] = ina[i] + inb[i];
        poison[i] = ~expect_out[i];
    }
    for (i = 0; i < sizeof prog; i++) {
        prog[i] = (uint8_t)(i * 7u + 3u);
        prog_other[i] = (uint8_t)(i * 11u + 5u);
    }
}

static int output_correct(const uint32_t *o)
{
    uint32_t i;
    for (i = 0; i < N; i++)
        if (o[i] != expect_out[i]) return 0;
    return 1;
}

/* ---- tracing copy of the real backend table ---------------------------------- */

enum {
    T_OPEN = 1, T_CHANNEL, T_ALLOC, T_COPY_IN, T_POISON, T_BUILD, T_SUBMIT, T_WAIT1,
    T_WAIT2, T_WAITSEM, T_BARRIER, T_COPY_OUT, T_FREE, T_CLOSE
};
#define TLOG_MAX 128

static const OmegaGpuBackend *g_real;
static OmegaGpuBackend g_wrap;
static int g_tlog[TLOG_MAX];
static int g_targ[TLOG_MAX];
static int g_tn;

static void tlog(int kind, int arg)
{
    if (g_tn < TLOG_MAX) {
        g_tlog[g_tn] = kind;
        g_targ[g_tn] = arg;
        g_tn++;
    }
}

static int w_open(void *ctx) { tlog(T_OPEN, 0); return g_real->open_device(ctx); }
static int w_channel(void *ctx) { tlog(T_CHANNEL, 0); return g_real->create_channel(ctx); }
static int w_alloc(void *ctx, int role, size_t bytes) { tlog(T_ALLOC, role); return g_real->alloc(ctx, role, bytes); }
static int w_copy_in(void *ctx, int role, const void *src, size_t bytes) { tlog(T_COPY_IN, role); return g_real->copy_in(ctx, role, src, bytes); }
static int w_poison(void *ctx, const uint32_t *words, size_t count) { tlog(T_POISON, 0); return g_real->fill_poison(ctx, words, count); }
static int w_build(void *ctx, const OmegaGpuJob *job) { tlog(T_BUILD, 0); return g_real->build(ctx, job); }
static int w_submit(void *ctx) { tlog(T_SUBMIT, 0); return g_real->submit(ctx); }
static int w_wait1(void *ctx, uint64_t ms) { tlog(T_WAIT1, 0); return g_real->wait_marker(ctx, ms); }
static int w_wait2(void *ctx, uint64_t ms) { tlog(T_WAIT2, 0); return g_real->wait_marker2(ctx, ms); }
static int w_waitsem(void *ctx, uint64_t ms) { tlog(T_WAITSEM, 0); return g_real->wait_release_semaphore(ctx, ms); }
static int w_barrier(void *ctx, uint64_t ms) { tlog(T_BARRIER, 0); return g_real->barrier(ctx, ms); }
static int w_copy_out(void *ctx, void *dst, size_t bytes) { tlog(T_COPY_OUT, 0); return g_real->copy_out(ctx, dst, bytes); }
static void w_diag(void *ctx, OmegaGpuBackendDiag *d) { g_real->diagnostics(ctx, d); }
static int w_free(void *ctx, int role) { tlog(T_FREE, role); return g_real->free_buf(ctx, role); }
static int w_close(void *ctx) { tlog(T_CLOSE, 0); return g_real->close(ctx); }

static void wrap_init(void)
{
    g_real = omega_blackwell_engine_backend();
    g_wrap = *g_real;
    g_wrap.open_device = w_open;
    g_wrap.create_channel = w_channel;
    g_wrap.alloc = w_alloc;
    g_wrap.copy_in = w_copy_in;
    g_wrap.fill_poison = w_poison;
    g_wrap.build = w_build;
    g_wrap.submit = w_submit;
    g_wrap.wait_marker = w_wait1;
    g_wrap.wait_marker2 = w_wait2;
    g_wrap.wait_release_semaphore = w_waitsem;
    g_wrap.barrier = w_barrier;
    g_wrap.copy_out = w_copy_out;
    g_wrap.diagnostics = w_diag;
    g_wrap.free_buf = w_free;
    g_wrap.close = w_close;
}

static int tcount(int kind)
{
    int i;
    int n = 0;
    for (i = 0; i < g_tn; i++)
        if (g_tlog[i] == kind) n++;
    return n;
}

/* ---- case plumbing ---------------------------------------------------------- */

static OmegaGpuJob job;
static OmegaGpuResult res;

static void make_job(OmegaGpuJob *j)
{
    memset(j, 0, sizeof *j);
    j->program = prog;
    j->program_len = sizeof prog;
    j->layout = OMEGA_GPU_LAYOUT_VECTOR_1D;
    j->element_count = N;
    j->input_a = ina;
    j->input_a_len = N * 4u;
    j->input_b = inb;
    j->input_b_len = N * 4u;
    j->output = outbuf;
    j->output_len = N * 4u;
    j->poison = poison;
    j->poison_count = N;
    j->timeouts.marker_ms = TMO_MARKER;
    j->timeouts.marker2_ms = TMO_MARKER2;
    j->timeouts.release_semaphore_ms = TMO_RELEASE;
    j->timeouts.visibility_ms = TMO_VISIBILITY;
    j->flags = 0;
}

static void begin_case(void)
{
    fake_reset();
    omega_gpu_engine_test_reset_block();
    omega_gpu_engine_set_backend(&g_wrap);
    g_tn = 0;
    memset(outbuf, 0, sizeof outbuf);
    fk.prog_expect = prog;
    fk.prog_len = sizeof prog;
    make_job(&job);
}

static int run(void)
{
    return omega_gpu_execute(&job, &res);
}

static int res_is(int failure, int step, int state)
{
    return res.failure == failure && res.failed_step == step && res.last_state == state;
}

/* Everything the backend allocated was returned, the device was closed once, and
 * the number of real driver frees is as expected. */
static int balanced(int frees)
{
    return fk.live_at_close == 0 && fk.double_free == 0 && fk.close_calls == 1 &&
           fk.free_calls == frees && fk.device_open == 0;
}

static int has(const char *hay, const char *needle)
{
    return strstr(hay, needle) != NULL;
}

/* Drains a retained backend through its own table, the way process exit would:
 * free every role (all of them are held), then close. Returns 0 if all went well. */
static int drain(void)
{
    int bad = 0;
    int role;
    for (role = (int)OMEGA_GPU_BUF_ROLE_COUNT - 1; role >= 0; role--)
        if (g_real->free_buf(g_real->ctx, role) != 0) bad = 1;
    if (g_real->close(g_real->ctx) != 0) bad = 1;
    return bad;
}

static void idf(char *buf, size_t cap, const char *tag, const char *what)
{
    (void)snprintf(buf, cap, "sim_%s_%s", tag, what);
}

/* ---- cases ------------------------------------------------------------------ */

static void case_success(void)
{
    static const int k_order[24] = {
        T_OPEN, T_CHANNEL, T_ALLOC, T_ALLOC, T_ALLOC, T_ALLOC, T_ALLOC, T_COPY_IN, T_COPY_IN,
        T_COPY_IN, T_POISON, T_BUILD, T_SUBMIT, T_WAIT1, T_WAIT2, T_WAITSEM, T_BARRIER,
        T_COPY_OUT, T_FREE, T_FREE, T_FREE, T_FREE, T_FREE, T_CLOSE
    };
    static const int k_args[24] = {
        0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 2, 1, 0, 0
    };
    static const uint64_t k_sizes[8] = {
        0x10000u, 0x1000u, 0x1000u, 0x10000u, 0x1000u, 0x1000u, 0x1000u, 0x1000u
    };
    OmegaBlackwellEngineRunInfo info;
    int i;
    int ok;
    int i_cb, i_ar, i_q0, i_si, i_q1, i_k, i_r1, i_fl, i_r2;

    begin_case();
    check("sim_success_returns_ok", run() == OMEGA_GPU_ENGINE_OK);
    check("sim_success_state", res.last_state == OMEGA_GPU_ENGINE_STATE_SUCCESS &&
                               res.failed_step == OMEGA_GPU_ENGINE_STEP_NONE);
    check("sim_success_result_flags_clear",
          res.retained == 0 && res.cleanup_failed == 0 && res.wait == OMEGA_GPU_ENGINE_WAIT_NONE);
    check("sim_success_not_blocked", omega_gpu_engine_is_blocked() == 0);
    check("sim_success_output_values", output_correct(outbuf));
    check("sim_success_unchanged_zero", res.n_outputs == 1u && res.output_unchanged_words[0] == 0u);

    ok = g_tn == 24;
    for (i = 0; ok && i < 24; i++)
        if (g_tlog[i] != k_order[i] || g_targ[i] != k_args[i]) ok = 0;
    check("sim_success_call_order", ok);

    check("sim_success_driver_counts",
          fk.open_calls == 1 && fk.channel_calls == 1 && fk.alloc_calls == 8 &&
          fk.submit_calls == 1 && fk.close_calls == 1 && fk.wait_calls == 3);
    check("sim_success_all_allocs_gpu_uncached", fk.uncached_calls == fk.alloc_calls && fk.uncached_calls == 8);
    ok = fake_count(FK_EV_ALLOC) == 8;
    for (i = 0; ok && i < 8; i++)
        if (fk.ev[fake_nth(FK_EV_ALLOC, i + 1)].a != k_sizes[i]) ok = 0;
    check("sim_success_alloc_sizes", ok);
    check("sim_success_balanced", balanced(8));
    check("sim_success_large_pushbuffer_ring", fk.pb_mem_size == 0x10000u);
    check("sim_success_pushbuffer_parses", fk.pb_parse_error == 0 && fk.ev_overflow == 0);

    /* The state the GPU sees at the doorbell: the sync words were zeroed first. */
    check("sim_success_doorbell_state_captured", fk.pre_valid == 1);
    check("sim_success_marker1_zeroed_before_doorbell", fk.pre_marker1 == 0u);
    check("sim_success_marker2_zeroed_before_doorbell", fk.pre_marker2 == 0u);
    check("sim_success_semaphore_zeroed_before_doorbell", fk.pre_sem == 0u);
    check("sim_success_semaphore_init_value", fk.sem_init_value == SEM_INIT);

    /* The pushbuffer, as the device executes it. */
    i_cb = fake_nth(FK_EV_PB_CBANK, 1);
    i_ar = fake_nth(FK_EV_PB_ARGS, 1);
    i_q0 = fake_nth(FK_EV_PB_QMD, 1);
    i_si = fake_nth(FK_EV_PB_SEM_INIT, 1);
    i_q1 = fake_nth(FK_EV_PB_QMD, 2);
    i_k = fake_nth(FK_EV_KERNEL, 1);
    i_r1 = fake_nth(FK_EV_PB_RELEASE, 1);
    i_fl = fake_nth(FK_EV_PB_FLUSH, 1);
    i_r2 = fake_nth(FK_EV_PB_RELEASE, 2);
    check("sim_success_pushbuffer_order",
          i_cb >= 0 && i_cb < i_ar && i_ar < i_q0 && i_q0 < i_si && i_si < i_q1 && i_q1 < i_k &&
          i_k < i_r1 && i_r1 < i_fl && i_fl < i_r2);
    check("sim_success_one_flush_two_releases",
          fake_count(FK_EV_PB_FLUSH) == 1 && fake_count(FK_EV_PB_RELEASE) == 2);
    if (i_cb >= 0 && i_ar >= 0 && i_q0 >= 0 && i_q1 >= 0 && i_fl >= 0 && i_r1 >= 0 && i_r2 >= 0) {
        check("sim_success_cbank_upload_words_and_place",
              fk.ev[i_cb].b == 224u && fk.ev[i_cb].a == fk.alloc_va[1]);
        check("sim_success_args_upload_words_and_place",
              fk.ev[i_ar].b == 7u && fk.ev[i_ar].a == fk.ev[i_cb].a + 0x380u);
        check("sim_success_qmd0_is_not_the_kernel_and_place",
              fk.ev[i_q0].b == 0u && fk.ev[i_q0].a == fk.alloc_va[3]);
        check("sim_success_qmd1_is_the_kernel_and_place",
              fk.ev[i_q1].b == 1u && fk.ev[i_q1].a == fk.alloc_va[3] + 0x1000u);
        check("sim_success_flush_operation", fk.ev[i_fl].a == (uint64_t)FLUSH_OP);
        check("sim_success_release1_address_payload_flags",
              fk.ev[i_r1].a == fk.marker1_va && fk.ev[i_r1].b == MARKER1_PAYLOAD &&
              fk.ev[i_r1].c == RELEASE_FLAGS);
        check("sim_success_release2_is_marker_page_plus_0x10",
              fk.ev[i_r2].a == fk.ev[i_r1].a + 0x10u);
        check("sim_success_release2_payload_and_flags",
              fk.ev[i_r2].b == MARKER2_PAYLOAD && fk.ev[i_r2].c == RELEASE_FLAGS);
    } else {
        check("sim_success_cbank_upload_words_and_place", 0);
        check("sim_success_args_upload_words_and_place", 0);
        check("sim_success_qmd0_is_not_the_kernel_and_place", 0);
        check("sim_success_qmd1_is_the_kernel_and_place", 0);
        check("sim_success_flush_operation", 0);
        check("sim_success_release1_address_payload_flags", 0);
        check("sim_success_release2_is_marker_page_plus_0x10", 0);
        check("sim_success_release2_payload_and_flags", 0);
    }

    /* The kernel launch the device saw. */
    check("sim_success_kernel_ran_once_with_right_program", fk.kernel_ran == 1 && fk.prog_matched == 1);
    check("sim_success_kernel_threads_and_count", fk.kernel_threads == 128u && fk.kernel_n == N);
    check("sim_success_kernel_arguments_are_a_b_output",
          fk.arg_a == fk.alloc_va[5] && fk.arg_b == fk.alloc_va[6] && fk.arg_c == fk.alloc_va[7]);
    check("sim_success_kernel_program_and_cbank_addresses",
          fk.prog_va_seen == fk.alloc_va[4] && fk.cbank_va_seen == fk.alloc_va[1]);
    check("sim_success_marker_and_semaphore_places",
          fk.marker1_va == fk.alloc_va[2] && fk.sem_va == fk.alloc_va[3] + 0x2000u);
    ok = fk.snap_n == N;
    for (i = 0; ok && i < (int)N; i++)
        if (fk.out_before[i] != poison[i]) ok = 0;
    check("sim_success_poison_reached_device_output", ok);
    check("sim_success_poison_never_equals_a_sum", fk.poison_collisions == 0);

    /* The waits, by word, expected value and timeout. */
    check("sim_success_wait_counts",
          fk.wait_n[FK_W_MARKER1] == 1 && fk.wait_n[FK_W_MARKER2] == 1 &&
          fk.wait_n[FK_W_SEM] == 1 && fk.wait_n[FK_W_UNKNOWN] == 0);
    check("sim_success_wait_expected_values",
          fk.wait_expected[FK_W_MARKER1] == MARKER1_PAYLOAD &&
          fk.wait_expected[FK_W_MARKER2] == MARKER2_PAYLOAD &&
          fk.wait_expected[FK_W_SEM] == SEM_DONE);
    check("sim_success_wait_timeouts_come_from_the_job",
          fk.wait_ms[FK_W_MARKER1] == TMO_MARKER && fk.wait_ms[FK_W_MARKER2] == TMO_MARKER2 &&
          fk.wait_ms[FK_W_SEM] == TMO_RELEASE);
    i = fake_nth(FK_EV_WAIT, 1);
    ok = i >= 0 && fk.ev[i].a == (uint64_t)FK_W_MARKER1;
    i = fake_nth(FK_EV_WAIT, 2);
    ok = ok && i >= 0 && fk.ev[i].a == (uint64_t)FK_W_MARKER2;
    i = fake_nth(FK_EV_WAIT, 3);
    ok = ok && i >= 0 && fk.ev[i].a == (uint64_t)FK_W_SEM;
    check("sim_success_wait_order_marker_marker2_semaphore", ok);

    /* What the backend observed, for the vector wrapper. */
    memset(&info, 0, sizeof info);
    omega_blackwell_engine_run_info(&info);
    check("sim_success_run_info_words",
          info.marker == MARKER1_PAYLOAD && info.marker2 == MARKER2_PAYLOAD && info.semaphore == SEM_DONE);
    check("sim_success_run_info_launch_time_set", info.launch_ns != 0u);
    check("sim_success_run_info_done_time_after_launch",
          info.marker_done_ns != 0u && info.marker_done_ns >= info.launch_ns);
}

static void case_no_c3(void)
{
    OmegaBlackwellEngineRunInfo info;
    begin_case();
    job.flags = OMEGA_GPU_ENGINE_FLAG_NO_C3;
    check("sim_noc3_returns_ok", run() == OMEGA_GPU_ENGINE_OK);
    check("sim_noc3_output_values", output_correct(outbuf));
    check("sim_noc3_no_flush_in_pushbuffer", fake_count(FK_EV_PB_FLUSH) == 0);
    check("sim_noc3_one_release_only", fake_count(FK_EV_PB_RELEASE) == 1 && fk.marker2_va == 0u);
    check("sim_noc3_marker2_never_waited",
          fk.wait_n[FK_W_MARKER2] == 0 && tcount(T_WAIT2) == 0 && fk.wait_calls == 2);
    check("sim_noc3_marker1_and_semaphore_waited",
          fk.wait_n[FK_W_MARKER1] == 1 && fk.wait_n[FK_W_SEM] == 1);
    memset(&info, 0, sizeof info);
    omega_blackwell_engine_run_info(&info);
    check("sim_noc3_run_info_marker2_not_reported", info.marker2 == 0u && info.marker == MARKER1_PAYLOAD);
    check("sim_noc3_balanced", balanced(8));
}

static void case_back_to_back(void)
{
    int rc1;
    int rc2;
    int ok;
    begin_case();
    rc1 = run();
    ok = output_correct(outbuf);
    memset(outbuf, 0, sizeof outbuf);
    rc2 = run();
    check("sim_twice_first_ok", rc1 == OMEGA_GPU_ENGINE_OK && ok);
    check("sim_twice_second_ok_busy_flag_released", rc2 == OMEGA_GPU_ENGINE_OK && output_correct(outbuf));
    check("sim_twice_driver_opened_and_closed_twice", fk.open_calls == 2 && fk.close_calls == 2);
}

static void case_open_fail(void)
{
    OmegaBlackwellEngineRunInfo info;
    int rc;
    begin_case();
    (void)run(); /* leave a dirty run snapshot behind */
    begin_case();
    fk.open_fail = 1;
    rc = run();
    check("sim_open_fail_name_step_state",
          rc == OMEGA_GPU_ENGINE_DEVICE_OPEN &&
          res_is(OMEGA_GPU_ENGINE_DEVICE_OPEN, OMEGA_GPU_ENGINE_STEP_OPEN_DEVICE,
                 OMEGA_GPU_ENGINE_STATE_INITIAL));
    check("sim_open_fail_errno_and_driver_code", res.err_no == EIO && res.drv_rc == -1);
    check("sim_open_fail_driver_text_reported", strcmp(res.drv_text, FK_TEXT) == 0);
    check("sim_open_fail_nothing_else_touched",
          fk.channel_calls == 0 && fk.alloc_calls == 0 && fk.close_calls == 0 && fk.free_calls == 0);
    check("sim_open_fail_not_retained_not_blocked",
          res.retained == 0 && omega_gpu_engine_is_blocked() == 0);
    memset(&info, 0, sizeof info);
    omega_blackwell_engine_run_info(&info);
    check("sim_open_fail_run_info_reset_at_open",
          info.marker == 0u && info.semaphore == 0u && info.launch_ns == 0u && info.marker_done_ns == 0u);
    begin_case();
    check("sim_open_fail_then_next_run_ok", run() == OMEGA_GPU_ENGINE_OK);
}

static void case_channel_fail(void)
{
    int rc;
    begin_case();
    fk.channel_fail = 1;
    rc = run();
    check("sim_channel_fail_name_step_state",
          rc == OMEGA_GPU_ENGINE_CHANNEL_CREATE &&
          res_is(OMEGA_GPU_ENGINE_CHANNEL_CREATE, OMEGA_GPU_ENGINE_STEP_CREATE_CHANNEL,
                 OMEGA_GPU_ENGINE_STATE_INITIAL));
    check("sim_channel_fail_errno_text_code",
          res.err_no == EIO && res.drv_rc == -1 && strcmp(res.drv_text, FK_TEXT) == 0);
    check("sim_channel_fail_closed_nothing_allocated", fk.alloc_calls == 0 && balanced(0));
    begin_case();
    check("sim_channel_fail_then_next_run_ok", run() == OMEGA_GPU_ENGINE_OK);
}

static void case_alloc_faults(void)
{
    int nth;
    char id[64];
    for (nth = 1; nth <= 8; nth++) {
        int rc;
        begin_case();
        fk.alloc_fail_nth = nth;
        fk.alloc_mode = FK_ALLOC_FAIL;
        rc = run();
        (void)snprintf(id, sizeof id, "sim_alloc%d_name_step_state", nth);
        check(id, rc == OMEGA_GPU_ENGINE_ALLOC &&
                  res_is(OMEGA_GPU_ENGINE_ALLOC, OMEGA_GPU_ENGINE_STEP_ALLOC,
                         OMEGA_GPU_ENGINE_STATE_INITIAL));
        (void)snprintf(id, sizeof id, "sim_alloc%d_errno_code_text", nth);
        check(id, res.err_no == EIO && res.drv_rc == -1 && strcmp(res.drv_text, FK_TEXT) == 0);
        (void)snprintf(id, sizeof id, "sim_alloc%d_stops_at_the_failure", nth);
        check(id, fk.alloc_calls == nth);
        /* nothing is leaked: every part that was allocated is freed (nth - 1 of them) */
        (void)snprintf(id, sizeof id, "sim_alloc%d_balanced", nth);
        check(id, balanced(nth - 1));
    }
}

static void case_alloc_own_checks(void)
{
    static const struct { int mode; int nth; int frees; const char *tag; } t[] = {
        { FK_ALLOC_SHORT, 1, 1, "short_pb" },
        { FK_ALLOC_NO_CPU, 3, 3, "nocpu_marker" },
        { FK_ALLOC_MISALIGNED, 4, 4, "misaligned_qmd" },
        { FK_ALLOC_SHORT, 6, 6, "short_input_a" },
        { FK_ALLOC_NO_CPU, 6, 6, "nocpu_input_a" },
        { FK_ALLOC_MISALIGNED, 6, 6, "misaligned_input_a" },
        { FK_ALLOC_NO_CPU, 8, 8, "nocpu_output" }
    };
    size_t k;
    char id[64];
    for (k = 0; k < sizeof t / sizeof t[0]; k++) {
        int rc;
        begin_case();
        fk.alloc_fail_nth = t[k].nth;
        fk.alloc_mode = t[k].mode;
        rc = run();
        idf(id, sizeof id, t[k].tag, "refused_as_alloc");
        check(id, rc == OMEGA_GPU_ENGINE_ALLOC &&
                  res_is(OMEGA_GPU_ENGINE_ALLOC, OMEGA_GPU_ENGINE_STEP_ALLOC,
                         OMEGA_GPU_ENGINE_STATE_INITIAL));
        idf(id, sizeof id, t[k].tag, "own_check_text");
        check(id, has(res.drv_text, "own checks"));
        idf(id, sizeof id, t[k].tag, "balanced");
        check(id, balanced(t[k].frees));
    }
}

static void case_submit_fail(void)
{
    int rc;
    begin_case();
    fk.submit_fail = 1;
    rc = run();
    check("sim_submit_fail_name_step_state",
          rc == OMEGA_GPU_ENGINE_SUBMIT &&
          res_is(OMEGA_GPU_ENGINE_SUBMIT, OMEGA_GPU_ENGINE_STEP_SUBMIT, OMEGA_GPU_ENGINE_STATE_PREPARED));
    check("sim_submit_fail_errno_code_text",
          res.err_no == EIO && res.drv_rc == -1 && strcmp(res.drv_text, FK_TEXT) == 0);
    check("sim_submit_fail_is_not_uncertain",
          res.retained == 0 && omega_gpu_engine_is_blocked() == 0 && fk.kernel_ran == 0);
    check("sim_submit_fail_cleaned_up", balanced(8));
}

/* One uncertain completion: the engine keeps everything and blocks the process.
 * Asserts that, asserts the next job is refused without touching the driver, asserts
 * a second open of the backend is refused, drains, and proves the backend works again. */
static void uncertain_case(const char *tag, int mk, int mk2, int sem, uint32_t flags,
                           int failure, int step, int wait, uint64_t limit, uint64_t waited,
                           uint32_t want_marker, uint32_t want_marker2, uint32_t want_sem,
                           uint32_t want_sync, const char *want_text, int want_drv_rc)
{
    char id[96];
    int rc;
    int bad;
    int open_before;
    OmegaBlackwellEngineRunInfo info;

    begin_case();
    fk.marker_mode = mk;
    fk.marker2_mode = mk2;
    fk.sem_mode = sem;
    job.flags = flags;
    rc = run();

    idf(id, sizeof id, tag, "name_step_wait");
    check(id, rc == failure && res.failure == failure && res.failed_step == step && res.wait == wait);
    idf(id, sizeof id, tag, "last_state_submitted");
    check(id, res.last_state == OMEGA_GPU_ENGINE_STATE_SUBMITTED);
    idf(id, sizeof id, tag, "limit_and_waited_ms");
    check(id, res.wait_timeout_ms == limit && res.waited_ms == waited);
    idf(id, sizeof id, tag, "observed_words");
    check(id, res.marker == want_marker && res.marker2 == want_marker2 &&
              res.semaphore[0] == want_sem);
    idf(id, sizeof id, tag, "sync_valid_mask");
    check(id, res.sync_valid == want_sync);
    if (want_text != NULL) {
        idf(id, sizeof id, tag, "driver_text");
        check(id, has(res.drv_text, want_text));
    }
    idf(id, sizeof id, tag, "retained_and_blocked");
    check(id, res.retained == 1 && omega_gpu_engine_is_blocked() == 1);
    idf(id, sizeof id, tag, "nothing_freed_nothing_closed");
    check(id, fk.free_calls == 0 && fk.close_calls == 0 && fk.live == 8 && fk.device_open == 1 &&
              tcount(T_FREE) == 0 && tcount(T_CLOSE) == 0);

    idf(id, sizeof id, tag, "driver_code");
    check(id, res.drv_rc == want_drv_rc);
    memset(&info, 0, sizeof info);
    omega_blackwell_engine_run_info(&info);
    idf(id, sizeof id, tag, "run_info_marker");
    check(id, info.marker == want_marker && info.launch_ns != 0u);
    idf(id, sizeof id, tag, "run_info_done_time_only_if_marker_succeeded");
    check(id, (step == OMEGA_GPU_ENGINE_STEP_WAIT_MARKER) == (info.marker_done_ns == 0u));
    idf(id, sizeof id, tag, "run_info_marker2_when_it_was_the_failed_wait");
    check(id, step != OMEGA_GPU_ENGINE_STEP_WAIT_MARKER2 || info.marker2 == want_marker2);
    idf(id, sizeof id, tag, "run_info_semaphore_when_it_was_the_failed_wait");
    check(id, step != OMEGA_GPU_ENGINE_STEP_WAIT_RELEASE_SEMAPHORE || info.semaphore == want_sem);
    idf(id, sizeof id, tag, "no_marker2_wait_with_noc3");
    check(id, (flags & OMEGA_GPU_ENGINE_FLAG_NO_C3) == 0u || (fk.wait_n[FK_W_MARKER2] == 0 && tcount(T_WAIT2) == 0));
    open_before = fk.open_calls;
    rc = run();
    idf(id, sizeof id, tag, "next_job_blocked");
    check(id, rc == OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED &&
              res.failed_step == OMEGA_GPU_ENGINE_STEP_BLOCKED && fk.open_calls == open_before);
    idf(id, sizeof id, tag, "second_open_refused_by_backend");
    check(id, g_real->open_device(g_real->ctx) != 0 && fk.open_calls == open_before);
    idf(id, sizeof id, tag, "still_retained_after_refusals");
    check(id, fk.live == 8 && fk.device_open == 1 && fk.close_calls == 0);

    bad = drain();
    idf(id, sizeof id, tag, "drained_through_the_table");
    check(id, bad == 0 && fk.close_calls == 1 && fk.live_at_close == 0 && fk.double_free == 0 &&
              fk.free_calls == 8);
    omega_gpu_engine_test_reset_block();
    make_job(&job);
    /* The simulated driver still has the hang or overshoot of this case armed: restore a healthy chip. */
    fk.marker_mode = FK_SYNC_NORMAL;
    fk.marker2_mode = FK_SYNC_NORMAL;
    fk.sem_mode = FK_SYNC_NORMAL;
    idf(id, sizeof id, tag, "backend_works_again_after_drain");
    check(id, run() == OMEGA_GPU_ENGINE_OK && output_correct(outbuf));
}

static void case_uncertain(void)
{
    const uint32_t all = OMEGA_GPU_SYNC_MARKER | OMEGA_GPU_SYNC_MARKER2 | OMEGA_GPU_SYNC_SEMAPHORE;

    uncertain_case("unc_marker_hang", FK_SYNC_NEVER, FK_SYNC_NORMAL, FK_SYNC_NORMAL, 0u,
                   OMEGA_GPU_ENGINE_COMPLETION_WAIT, OMEGA_GPU_ENGINE_STEP_WAIT_MARKER,
                   OMEGA_GPU_ENGINE_WAIT_MARKER, TMO_MARKER, TMO_MARKER,
                   0u, MARKER2_PAYLOAD, SEM_DONE, all, NULL, -1);
    uncertain_case("unc_marker2_hang", FK_SYNC_NORMAL, FK_SYNC_NEVER, FK_SYNC_NORMAL, 0u,
                   OMEGA_GPU_ENGINE_COMPLETION_WAIT, OMEGA_GPU_ENGINE_STEP_WAIT_MARKER2,
                   OMEGA_GPU_ENGINE_WAIT_MARKER2, TMO_MARKER2, TMO_MARKER2,
                   MARKER1_PAYLOAD, 0u, SEM_DONE, all, NULL, -1);
    uncertain_case("unc_semaphore_hang", FK_SYNC_NORMAL, FK_SYNC_NORMAL, FK_SYNC_NEVER, 0u,
                   OMEGA_GPU_ENGINE_RELEASE_WAIT, OMEGA_GPU_ENGINE_STEP_WAIT_RELEASE_SEMAPHORE,
                   OMEGA_GPU_ENGINE_WAIT_RELEASE_SEMAPHORE, TMO_RELEASE, TMO_RELEASE,
                   MARKER1_PAYLOAD, MARKER2_PAYLOAD, SEM_INIT, all, NULL, -1);
    uncertain_case("unc_marker_overshoot", FK_SYNC_OVERSHOOT, FK_SYNC_NORMAL, FK_SYNC_NORMAL, 0u,
                   OMEGA_GPU_ENGINE_COMPLETION_WAIT, OMEGA_GPU_ENGINE_STEP_WAIT_MARKER,
                   OMEGA_GPU_ENGINE_WAIT_MARKER, TMO_MARKER, 0u,
                   MARKER1_PAYLOAD + 1u, MARKER2_PAYLOAD, SEM_DONE, all, "observed 0x44444445", 0);
    uncertain_case("unc_marker2_overshoot", FK_SYNC_NORMAL, FK_SYNC_OVERSHOOT, FK_SYNC_NORMAL, 0u,
                   OMEGA_GPU_ENGINE_COMPLETION_WAIT, OMEGA_GPU_ENGINE_STEP_WAIT_MARKER2,
                   OMEGA_GPU_ENGINE_WAIT_MARKER2, TMO_MARKER2, 0u,
                   MARKER1_PAYLOAD, MARKER2_PAYLOAD + 1u, SEM_DONE, all, "observed 0x46464647", 0);
    uncertain_case("unc_semaphore_overshoot", FK_SYNC_NORMAL, FK_SYNC_NORMAL, FK_SYNC_OVERSHOOT, 0u,
                   OMEGA_GPU_ENGINE_RELEASE_WAIT, OMEGA_GPU_ENGINE_STEP_WAIT_RELEASE_SEMAPHORE,
                   OMEGA_GPU_ENGINE_WAIT_RELEASE_SEMAPHORE, TMO_RELEASE, 0u,
                   MARKER1_PAYLOAD, MARKER2_PAYLOAD, SEM_DONE + 1u, all, "observed 0x00000007", 0);
    /* NO_C3: the second marker does not exist, so it is never reported either. */
    uncertain_case("unc_noc3_semaphore_hang", FK_SYNC_NORMAL, FK_SYNC_NORMAL, FK_SYNC_NEVER,
                   OMEGA_GPU_ENGINE_FLAG_NO_C3,
                   OMEGA_GPU_ENGINE_RELEASE_WAIT, OMEGA_GPU_ENGINE_STEP_WAIT_RELEASE_SEMAPHORE,
                   OMEGA_GPU_ENGINE_WAIT_RELEASE_SEMAPHORE, TMO_RELEASE, TMO_RELEASE,
                   MARKER1_PAYLOAD, 0u, SEM_INIT, OMEGA_GPU_SYNC_MARKER | OMEGA_GPU_SYNC_SEMAPHORE, NULL, -1);
}

static void case_cleanup_faults(void)
{
    int rc;

    /* a free of an ordinary buffer fails: every other free still runs */
    begin_case();
    fk.free_fail_nth = 1;
    rc = run();
    check("sim_cleanup_free_role_name_step_state",
          rc == OMEGA_GPU_ENGINE_CLEANUP &&
          res_is(OMEGA_GPU_ENGINE_CLEANUP, OMEGA_GPU_ENGINE_STEP_FREE,
                 OMEGA_GPU_ENGINE_STATE_OUTPUT_PRODUCED));
    check("sim_cleanup_free_role_flag_errno_text",
          res.cleanup_failed == 1 && res.err_no == EIO && strcmp(res.drv_text, FK_TEXT) == 0);
    check("sim_cleanup_free_role_every_free_attempted_and_closed",
          fk.free_calls == 8 && fk.close_calls == 1 && fk.live_at_close == 1);
    check("sim_cleanup_free_role_words_still_mapped_at_capture",
          res.sync_valid == (OMEGA_GPU_SYNC_MARKER | OMEGA_GPU_SYNC_MARKER2 | OMEGA_GPU_SYNC_SEMAPHORE) &&
          res.marker == MARKER1_PAYLOAD && res.marker2 == MARKER2_PAYLOAD && res.semaphore[0] == SEM_DONE);
    check("sim_cleanup_free_role_not_uncertain", res.retained == 0 && omega_gpu_engine_is_blocked() == 0);

    /* a free of one SCRATCH part fails (the marker page, sixth real free): the
     * other scratch parts are still freed and the sync words are unmapped first */
    begin_case();
    fk.free_fail_nth = 6;
    rc = run();
    check("sim_cleanup_free_scratch_name_step_state",
          rc == OMEGA_GPU_ENGINE_CLEANUP &&
          res_is(OMEGA_GPU_ENGINE_CLEANUP, OMEGA_GPU_ENGINE_STEP_FREE,
                 OMEGA_GPU_ENGINE_STATE_OUTPUT_PRODUCED));
    check("sim_cleanup_free_scratch_code_and_text",
          res.cleanup_failed == 1 && res.drv_rc == -1 && strcmp(res.drv_text, FK_TEXT) == 0);
    check("sim_cleanup_free_scratch_all_four_parts_attempted",
          fk.free_calls == 8 && fk.close_calls == 1 && fk.live_at_close == 1);
    check("sim_cleanup_free_scratch_sync_words_unmapped_before_free", res.sync_valid == 0u);

    /* close fails */
    begin_case();
    fk.close_fail = 1;
    rc = run();
    check("sim_cleanup_close_name_step_state",
          rc == OMEGA_GPU_ENGINE_CLEANUP &&
          res_is(OMEGA_GPU_ENGINE_CLEANUP, OMEGA_GPU_ENGINE_STEP_CLOSE,
                 OMEGA_GPU_ENGINE_STATE_OUTPUT_PRODUCED));
    check("sim_cleanup_close_flag_errno_text",
          res.cleanup_failed == 1 && res.err_no == EIO && strcmp(res.drv_text, FK_TEXT) == 0);
    check("sim_cleanup_close_all_freed_first", fk.free_calls == 8 && fk.live_at_close == 0);
    begin_case();
    check("sim_cleanup_close_failure_still_releases_busy_flag", run() == OMEGA_GPU_ENGINE_OK);
}

static void case_unchanged(void)
{
    int rc;

    begin_case();
    fk.kernel_mode = FK_KERNEL_NO_WRITE;
    rc = run();
    check("sim_unchanged_nowrite_name_step_state",
          rc == OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED &&
          res_is(OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED, OMEGA_GPU_ENGINE_STEP_SCAN,
                 OMEGA_GPU_ENGINE_STATE_OUTPUT_VISIBLE));
    check("sim_unchanged_nowrite_counts_every_word",
          res.n_outputs == 1u && res.output_unchanged_words[0] == N);
    check("sim_unchanged_nowrite_completion_was_seen",
          fk.kernel_ran == 1 && fk.wait_n[FK_W_MARKER1] == 1 && fk.wait_n[FK_W_SEM] == 1);
    check("sim_unchanged_nowrite_cleaned_up_not_retained",
          balanced(8) && res.retained == 0 && omega_gpu_engine_is_blocked() == 0);

    begin_case();
    fk.kernel_mode = FK_KERNEL_HALF;
    rc = run();
    check("sim_unchanged_half_counts_the_unwritten_half",
          rc == OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED && res.output_unchanged_words[0] == N / 2u);

    /* the device runs different program bytes than the job supplied: it writes nothing */
    begin_case();
    fk.prog_expect = prog_other;
    rc = run();
    check("sim_unchanged_wrong_program_bytes_detected",
          rc == OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED && fk.kernel_ran == 1 && fk.prog_matched == 0 &&
          res.output_unchanged_words[0] == N);
}

static void case_table_guards(void)
{
    const OmegaGpuBackend *be = omega_blackwell_engine_backend();
    void *ctx = be->ctx;
    OmegaGpuJob j;
    int role;
    int bad = 0;

    begin_case();
    memset(&j, 0, sizeof j);
    j.layout = OMEGA_GPU_LAYOUT_VECTOR_1D;
    j.element_count = N;

    check("sim_guard_open_ok", be->open_device(ctx) == 0);
    check("sim_guard_channel_ok", be->create_channel(ctx) == 0);
    check("sim_guard_alloc_zero_bytes_refused", be->alloc(ctx, (int)OMEGA_GPU_BUF_OUTPUT, 0u) != 0);
    check("sim_guard_alloc_unknown_role_refused", be->alloc(ctx, 99, 4096u) != 0);
    check("sim_guard_alloc_rounds_up_to_whole_pages",
          be->alloc(ctx, (int)OMEGA_GPU_BUF_OUTPUT, 5000u) == 0 && fake_count(FK_EV_ALLOC) == 1 &&
          fk.ev[fake_nth(FK_EV_ALLOC, 1)].a == 0x2000u);
    check("sim_guard_free_after_rounding_ok", be->free_buf(ctx, (int)OMEGA_GPU_BUF_OUTPUT) == 0);
    check("sim_guard_wait_before_build_refused_without_driver_call",
          be->wait_marker(ctx, 5u) != 0 && be->wait_marker2(ctx, 5u) != 0 &&
          be->wait_release_semaphore(ctx, 5u) != 0 && fk.wait_calls == 0);
    check("sim_guard_submit_before_build_refused", be->submit(ctx) != 0);
    check("sim_guard_build_without_buffers_refused", be->build(ctx, &j) != 0);

    check("sim_guard_alloc_scratch_ok", be->alloc(ctx, (int)OMEGA_GPU_BUF_SCRATCH, 0u) == 0);
    check("sim_guard_alloc_program_ok", be->alloc(ctx, (int)OMEGA_GPU_BUF_PROGRAM, sizeof prog) == 0);
    check("sim_guard_alloc_a_ok", be->alloc(ctx, (int)OMEGA_GPU_BUF_INPUT_A, N * 4u) == 0);
    check("sim_guard_alloc_b_ok", be->alloc(ctx, (int)OMEGA_GPU_BUF_INPUT_B, N * 4u) == 0);
    check("sim_guard_alloc_output_ok", be->alloc(ctx, (int)OMEGA_GPU_BUF_OUTPUT, N * 4u) == 0);
    check("sim_guard_copy_in_negative_role_refused", be->copy_in(ctx, -1, prog, sizeof prog) != 0);
    check("sim_guard_free_negative_role_refused", be->free_buf(ctx, -1) != 0);
    check("sim_guard_build_null_job_refused", be->build(ctx, NULL) != 0);
    j.layout = 7;
    check("sim_guard_build_unknown_layout_refused", be->build(ctx, &j) != 0);
    j.layout = (int)OMEGA_GPU_LAYOUT_VECTOR_1D;
    j.element_count = 0u;
    check("sim_guard_build_zero_count_refused", be->build(ctx, &j) != 0);
    j.element_count = 2000u;
    check("sim_guard_build_count_beyond_the_buffers_refused", be->build(ctx, &j) != 0);
    j.element_count = N;

    check("sim_guard_copy_in_too_long_refused",
          be->copy_in(ctx, (int)OMEGA_GPU_BUF_PROGRAM, bigbytes, 0x1000u + 8u) != 0);
    check("sim_guard_copy_in_unknown_role_refused", be->copy_in(ctx, 99, prog, sizeof prog) != 0);
    check("sim_guard_copy_in_null_source_refused",
          be->copy_in(ctx, (int)OMEGA_GPU_BUF_PROGRAM, NULL, sizeof prog) != 0);
    check("sim_guard_fill_poison_too_long_refused",
          be->fill_poison(ctx, bigwords, 0x1000u / 4u + 1u) != 0);
    check("sim_guard_fill_poison_null_refused", be->fill_poison(ctx, NULL, N) != 0);
    check("sim_guard_copy_out_too_long_refused",
          be->copy_out(ctx, bigbytes, 0x1000u + 8u) != 0);
    check("sim_guard_copy_out_null_refused", be->copy_out(ctx, NULL, N * 4u) != 0);
    check("sim_guard_free_unknown_role_refused", be->free_buf(ctx, 99) != 0);

    check("sim_guard_valid_copy_in_ok", be->copy_in(ctx, (int)OMEGA_GPU_BUF_PROGRAM, prog, sizeof prog) == 0);
    check("sim_guard_valid_fill_poison_ok", be->fill_poison(ctx, poison, N) == 0);
    memset(bigbytes, 0, sizeof bigbytes);
    check("sim_guard_valid_copy_out_returns_the_poison",
          be->copy_out(ctx, bigbytes, N * 4u) == 0 && memcmp(bigbytes, poison, N * 4u) == 0);

    for (role = (int)OMEGA_GPU_BUF_ROLE_COUNT - 1; role >= 0; role--)
        if (be->free_buf(ctx, role) != 0) bad = 1;
    if (be->close(ctx) != 0) bad = 1;
    check("sim_guard_drain_ok", bad == 0);
    check("sim_guard_nothing_leaked", fk.live_at_close == 0 && fk.double_free == 0 && fk.close_calls == 1);
}

/* A close without the SCRATCH free (a backend that is closed with its words still
 * mapped) must stop reading the sync words: the memory is gone. */
static void case_close_unmaps_sync_words(void)
{
    const OmegaGpuBackend *be = omega_blackwell_engine_backend();
    void *ctx = be->ctx;
    OmegaGpuBackendDiag d;
    OmegaGpuJob j;
    int ok;

    begin_case();
    make_job(&j);
    ok = be->open_device(ctx) == 0 && be->create_channel(ctx) == 0 &&
         be->alloc(ctx, (int)OMEGA_GPU_BUF_SCRATCH, 0u) == 0 &&
         be->alloc(ctx, (int)OMEGA_GPU_BUF_PROGRAM, sizeof prog) == 0 &&
         be->alloc(ctx, (int)OMEGA_GPU_BUF_INPUT_A, N * 4u) == 0 &&
         be->alloc(ctx, (int)OMEGA_GPU_BUF_INPUT_B, N * 4u) == 0 &&
         be->alloc(ctx, (int)OMEGA_GPU_BUF_OUTPUT, N * 4u) == 0 &&
         be->build(ctx, &j) == 0;
    check("sim_close_setup_through_the_table", ok);
    memset(&d, 0, sizeof d);
    be->diagnostics(ctx, &d);
    check("sim_close_words_mapped_after_build",
          d.sync_valid == (OMEGA_GPU_SYNC_MARKER | OMEGA_GPU_SYNC_MARKER2 | OMEGA_GPU_SYNC_SEMAPHORE));
    check("sim_close_returns_zero", be->close(ctx) == 0);
    memset(&d, 0, sizeof d);
    be->diagnostics(ctx, &d);
    check("sim_close_unmaps_every_sync_word", d.sync_valid == 0u);
    check("sim_close_released_everything", fk.close_calls == 1 && fk.live == 0);
}

/* Like begin_case, for the vector wrapper: the device must see the encoder's real
 * program bytes, and no backend is installed, so the wrapper has to install its own. */
static void begin_vec(void)
{
    begin_case();
    omega_gpu_engine_set_backend(NULL);
    fk.prog_expect = vec_code;
    fk.prog_len = vec_len;
}

static void case_vector(void)
{
    OmegaVectorSpec spec;
    OmegaBlackwellRealization real;
    OmegaBlackwellVectorExecution info;
    uint32_t c[N];
    uint32_t i;
    int rc;
    int ok;
    uint64_t want_elapsed;

    memset(&real, 0, sizeof real);
    real.sm_architecture = OMEGA_BW_SM_ARCH_121;
    check("sim_vec_spec_init", omega_vector_spec_init(&spec, "engine_sim", N) == 0);

    /* success */
    begin_vec();
    memset(c, 0, sizeof c);
    memset(&info, 0xff, sizeof info);
    rc = omega_blackwell_execute_vector(&spec, &real, ina, inb, c, &info);
    check("sim_vec_success_returns_zero", rc == 0);
    check("sim_vec_success_output_values", output_correct(c));
    check("sim_vec_success_parity_verified", info.parity_verified == true);
    check("sim_vec_success_completion_marker_and_semaphore",
          info.completion_marker == MARKER1_PAYLOAD && info.intermediate_semaphore == SEM_DONE);
    check("sim_vec_success_shape_fields",
          info.element_count == N && info.sm_architecture == (uint32_t)OMEGA_BW_SM_ARCH_121 &&
          info.target_chip[0] != '\0');
    want_elapsed = info.completion_timestamp_ns > info.launch_timestamp_ns ?
                   info.completion_timestamp_ns - info.launch_timestamp_ns : 0u;
    check("sim_vec_success_timestamps",
          info.launch_timestamp_ns != 0u && info.completion_timestamp_ns != 0u &&
          info.elapsed_ns == want_elapsed);
    check("sim_vec_success_timeouts_are_5000_each",
          fk.wait_ms[FK_W_MARKER1] == 5000u && fk.wait_ms[FK_W_MARKER2] == 5000u &&
          fk.wait_ms[FK_W_SEM] == 5000u);
    check("sim_vec_success_all_protections_on",
          fake_count(FK_EV_PB_FLUSH) == 1 && fake_count(FK_EV_PB_RELEASE) == 2 &&
          fk.wait_n[FK_W_MARKER2] == 1);
    check("sim_vec_success_encoded_program_reached_the_device",
          fk.kernel_ran == 1 && fk.prog_matched == 1);
    ok = fk.snap_n == N;
    for (i = 0; ok && i < N; i++)
        if (fk.out_before[i] != (uint32_t)~expect_out[i]) ok = 0;
    check("sim_vec_success_poison_is_the_complement_of_the_sum", ok);
    check("sim_vec_success_poison_never_collides_even_for_0xdeadbeef", fk.poison_collisions == 0);
    check("sim_vec_success_balanced", balanced(8));

    /* wrong math on the device: the engine passes it (it only checks poison), the wrapper must not */
    begin_vec();
    fk.kernel_mode = FK_KERNEL_WRONG;
    memset(c, 0, sizeof c);
    memset(&info, 0, sizeof info);
    rc = omega_blackwell_execute_vector(&spec, &real, ina, inb, c, &info);
    check("sim_vec_wrong_math_returns_nonzero", rc != 0);
    check("sim_vec_wrong_math_parity_not_verified", info.parity_verified == false);

    /* a cleanup fault after a good run is still a failure */
    begin_vec();
    fk.free_fail_nth = 1;
    memset(c, 0, sizeof c);
    rc = omega_blackwell_execute_vector(&spec, &real, ina, inb, c, &info);
    check("sim_vec_cleanup_fault_returns_nonzero", rc != 0);

    /* the device wrote nothing */
    begin_vec();
    fk.kernel_mode = FK_KERNEL_NO_WRITE;
    memset(c, 0, sizeof c);
    rc = omega_blackwell_execute_vector(&spec, &real, ina, inb, c, &info);
    check("sim_vec_unwritten_output_returns_nonzero", rc != 0);

    /* a hung first marker: failure, process blocked, backend retained */
    begin_vec();
    fk.marker_mode = FK_SYNC_NEVER;
    memset(c, 0, sizeof c);
    rc = omega_blackwell_execute_vector(&spec, &real, ina, inb, c, &info);
    check("sim_vec_hang_returns_nonzero", rc != 0);
    check("sim_vec_hang_blocks_the_process_and_retains",
          omega_gpu_engine_is_blocked() == 1 && fk.free_calls == 0 && fk.close_calls == 0);
    check("sim_vec_hang_drains", drain() == 0 && fk.live_at_close == 0);
    omega_gpu_engine_test_reset_block();

    /* refusals before any driver call */
    begin_vec();
    check("sim_vec_null_spec_refused", omega_blackwell_execute_vector(NULL, &real, ina, inb, c, &info) != 0);
    check("sim_vec_null_output_refused", omega_blackwell_execute_vector(&spec, &real, ina, inb, NULL, &info) != 0);
    check("sim_vec_null_input_refused", omega_blackwell_execute_vector(&spec, &real, NULL, inb, c, &info) != 0);
    spec.element_count = 0u;
    check("sim_vec_zero_count_refused", omega_blackwell_execute_vector(&spec, &real, ina, inb, c, &info) != 0);
    check("sim_vec_refusals_never_touch_the_driver", fk.open_calls == 0 && fk.alloc_calls == 0);
    spec.element_count = N;

    /* the wrapper can be used again afterwards (busy flag, block, tracing table) */
    begin_vec();
    memset(c, 0, sizeof c);
    rc = omega_blackwell_execute_vector(&spec, &real, ina, inb, c, &info);
    check("sim_vec_works_again", rc == 0 && output_correct(c));
}

int main(void)
{
    init_data();
    wrap_init();
    omega_gpu_engine_test_set_clock(fake_now_ms);
    check("sim_setup_encoder_ok",
          omega_blackwell_encode_vecadd(vec_code, sizeof vec_code, &vec_len) == 0 && vec_len > 0u);

    case_success();
    case_no_c3();
    case_back_to_back();
    case_open_fail();
    case_channel_fail();
    case_alloc_faults();
    case_alloc_own_checks();
    case_submit_fail();
    case_uncertain();
    case_cleanup_faults();
    case_unchanged();
    case_table_guards();
    case_close_unmaps_sync_words();
    case_vector();

    omega_gpu_engine_set_backend(NULL);
    omega_gpu_engine_test_reset_block();
    omega_gpu_engine_test_set_clock(NULL);
    fake_reset();
    printf("SUMMARY passed=%d failed=%d\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
