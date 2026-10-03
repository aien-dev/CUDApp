/*
 * Host test of the Omega GPU Engine core (A3a) against a FAKE backend.
 * No device, no physics headers. Every case prints "[PASS] id" or "[FAIL] id";
 * the last line is "SUMMARY passed=N failed=M"; exit status is nonzero on any
 * failure. docs/numeric/OMEGA_GPU_ENGINE.md section 13 lists every case id.
 *
 * The fake records every backend call in order, can fail the n-th call of any
 * kind with a chosen return code (and advance a fake clock), simulates device
 * memory (poison fill, a kernel that writes a chosen word range, copy-in and
 * copy-out) and flags protocol violations (free of a buffer that is not
 * allocated, copy-in to an unallocated buffer, and so on).
 */
#include "omega_gpu_engine.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#define N 16u
#define INJ_ERRNO EIO
#define DIAG_RC (-1234)
#define DIAG_TEXT "fake driver text"
#define LOGMAX 64
#define TMO_MARKER 111u
#define TMO_MARKER2 222u
#define TMO_RELEASE 333u
#define TMO_VISIBILITY 444u

enum {
    C_OPEN = 0, C_CHANNEL, C_ALLOC, C_COPY_IN, C_POISON, C_BUILD, C_SUBMIT, C_MARKER,
    C_MARKER2, C_RELEASE, C_BARRIER, C_COPY_OUT, C_FREE, C_CLOSE, C_DIAG, C_COUNT
};

typedef struct {
    int call; /* which kind of call to fail */
    int nth;  /* 1-based occurrence of that kind */
    int rc;   /* value the call returns */
    int advance; /* fake milliseconds the call takes */
} Inj;

typedef struct {
    int log[LOGMAX];            /* every call except diagnostics, in order */
    int nlog;
    int count[C_COUNT];
    int state_at[C_COUNT];      /* result->last_state at the first call of each kind, -1 if never */
    uint64_t tmo[C_COUNT];      /* timeout last passed to a timed call */
    Inj inj[2];
    int ninj;
    int violations;
    int dev_open;
    int channel;
    int live[OMEGA_GPU_BUF_ROLE_COUNT];
    int alloc_roles[8];
    size_t alloc_bytes[8];
    int nalloc;
    int copy_roles[8];
    size_t copy_bytes[8];
    int ncopy;
    int free_roles[8];
    int nfree;
    uint32_t dev_prog[16];
    uint32_t dev_a[N];
    uint32_t dev_b[N];
    uint32_t dev_out[N];
    uint32_t seen_at_submit[N];
    int submit_seen;
    const OmegaGpuJob *build_job;
    size_t klo;                 /* the fake kernel writes output words [klo, khi) */
    size_t khi;
} Fake;

static Fake F;
static OmegaGpuResult *g_res;   /* result being filled, to snapshot last_state at each call */
static uint64_t g_time;         /* fake clock, milliseconds */
static OmegaGpuBackend BE;

static const unsigned char code[8] = {1, 2, 3, 4, 5, 6, 7, 8};
static uint32_t ina[N];
static uint32_t inb[N];
static uint32_t poison[N];
static uint32_t host_out[N + 4u]; /* four spare words so a stray write is visible, not fatal */

static int case_ok;
static int n_pass;
static int n_fail;

#define EXPECT(c) do { if (!(c)) { case_ok = 0; printf("  expect failed at line %d: %s\n", __LINE__, #c); } } while (0)

static void case_begin(void) { case_ok = 1; }

static void case_end(const char *id)
{
    if (case_ok) { n_pass++; printf("[PASS] %s\n", id); }
    else { n_fail++; printf("[FAIL] %s\n", id); }
}

static uint64_t fake_now(void) { return g_time; }

static uint32_t exp_out(size_t i) { return ina[i] + inb[i] + 0x100u; }

/* ---- the fake backend ---------------------------------------------------- */

static int hit(int call)
{
    int i;
    int n;
    if (F.nlog < LOGMAX) F.log[F.nlog] = call;
    F.nlog++;
    n = ++F.count[call];
    if (F.state_at[call] < 0 && g_res) F.state_at[call] = g_res->last_state;
    for (i = 0; i < F.ninj; i++) {
        if (F.inj[i].call == call && F.inj[i].nth == n) {
            g_time += (uint64_t)F.inj[i].advance;
            errno = INJ_ERRNO;
            return F.inj[i].rc;
        }
    }
    return 0;
}

static int f_open(void *c)
{
    int rc;
    (void)c;
    rc = hit(C_OPEN);
    if (rc == 0) F.dev_open = 1;
    return rc;
}

static int f_channel(void *c)
{
    int rc;
    (void)c;
    rc = hit(C_CHANNEL);
    if (rc != 0) return rc;
    if (!F.dev_open) F.violations++;
    F.channel = 1;
    return 0;
}

static int role_ok(int role)
{
    return role >= 0 && role < (int)OMEGA_GPU_BUF_ROLE_COUNT;
}

static int f_alloc(void *c, int role, size_t bytes)
{
    int rc;
    (void)c;
    rc = hit(C_ALLOC);
    if (rc != 0) return rc;
    if (!role_ok(role) || F.live[role] || !F.channel) { F.violations++; return 0; }
    F.live[role] = 1;
    if (F.nalloc < 8) { F.alloc_roles[F.nalloc] = role; F.alloc_bytes[F.nalloc] = bytes; }
    F.nalloc++;
    return 0;
}

static int f_copy_in(void *c, int role, const void *src, size_t bytes)
{
    int rc;
    (void)c;
    rc = hit(C_COPY_IN);
    if (rc != 0) return rc;
    if (!role_ok(role) || !F.live[role]) { F.violations++; return 0; }
    if (F.ncopy < 8) { F.copy_roles[F.ncopy] = role; F.copy_bytes[F.ncopy] = bytes; }
    F.ncopy++;
    switch (role) {
    case OMEGA_GPU_BUF_PROGRAM:
        if (bytes > sizeof F.dev_prog) { F.violations++; break; }
        memcpy(F.dev_prog, src, bytes);
        break;
    case OMEGA_GPU_BUF_INPUT_A:
        if (bytes > sizeof F.dev_a) { F.violations++; break; }
        memcpy(F.dev_a, src, bytes);
        break;
    case OMEGA_GPU_BUF_INPUT_B:
        if (bytes > sizeof F.dev_b) { F.violations++; break; }
        memcpy(F.dev_b, src, bytes);
        break;
    default:
        F.violations++;
        break;
    }
    return 0;
}

static int f_poison(void *c, const uint32_t *words, size_t count)
{
    int rc;
    (void)c;
    rc = hit(C_POISON);
    if (rc != 0) return rc;
    if (!F.live[OMEGA_GPU_BUF_OUTPUT] || count != N) { F.violations++; return 0; }
    memcpy(F.dev_out, words, count * sizeof(uint32_t));
    return 0;
}

static int f_build(void *c, const OmegaGpuJob *job)
{
    int rc;
    (void)c;
    rc = hit(C_BUILD);
    if (rc != 0) return rc;
    F.build_job = job;
    return 0;
}

static int f_submit(void *c)
{
    int rc;
    size_t i;
    (void)c;
    rc = hit(C_SUBMIT);
    if (rc != 0) return rc;
    memcpy(F.seen_at_submit, F.dev_out, sizeof F.dev_out);
    F.submit_seen = 1;
    for (i = F.klo; i < F.khi && i < N; i++) F.dev_out[i] = F.dev_a[i] + F.dev_b[i] + 0x100u;
    return 0;
}

static int f_wait_marker(void *c, uint64_t ms)
{
    (void)c;
    F.tmo[C_MARKER] = ms;
    return hit(C_MARKER);
}

static int f_wait_marker2(void *c, uint64_t ms)
{
    (void)c;
    F.tmo[C_MARKER2] = ms;
    return hit(C_MARKER2);
}

static int f_wait_release(void *c, uint64_t ms)
{
    (void)c;
    F.tmo[C_RELEASE] = ms;
    return hit(C_RELEASE);
}

static int f_barrier(void *c, uint64_t ms)
{
    (void)c;
    F.tmo[C_BARRIER] = ms;
    return hit(C_BARRIER);
}

static int f_copy_out(void *c, void *dst, size_t bytes)
{
    int rc;
    (void)c;
    rc = hit(C_COPY_OUT);
    if (rc != 0) return rc;
    if (bytes != sizeof F.dev_out) { F.violations++; return 0; }
    memcpy(dst, F.dev_out, bytes);
    return 0;
}

static void f_diag(void *c, OmegaGpuBackendDiag *d)
{
    (void)c;
    F.count[C_DIAG]++;
    d->drv_rc = DIAG_RC;
    memcpy(d->drv_text, DIAG_TEXT, sizeof DIAG_TEXT);
    d->sync_valid = OMEGA_GPU_SYNC_MARKER | OMEGA_GPU_SYNC_MARKER2 | OMEGA_GPU_SYNC_SEMAPHORE;
    d->marker = 0x11111111u;
    d->marker2 = 0x22222222u;
    d->semaphore[0] = 0x33333333u;
    errno = 0; /* a diagnostics call that clobbers errno must not matter */
}

static int f_free(void *c, int role)
{
    int rc;
    (void)c;
    rc = hit(C_FREE);
    if (F.nfree < 8) F.free_roles[F.nfree] = role;
    F.nfree++;
    if (rc != 0) return rc;
    if (!role_ok(role) || !F.live[role]) { F.violations++; return 0; }
    F.live[role] = 0;
    return 0;
}

static int f_close(void *c)
{
    int rc;
    (void)c;
    rc = hit(C_CLOSE);
    if (rc != 0) return rc;
    if (!F.dev_open) F.violations++;
    F.dev_open = 0;
    F.channel = 0;
    return 0;
}

static void backend_init(OmegaGpuBackend *b)
{
    memset(b, 0, sizeof *b);
    b->ctx = NULL;
    b->open_device = f_open;
    b->create_channel = f_channel;
    b->alloc = f_alloc;
    b->copy_in = f_copy_in;
    b->fill_poison = f_poison;
    b->build = f_build;
    b->submit = f_submit;
    b->wait_marker = f_wait_marker;
    b->wait_marker2 = f_wait_marker2;
    b->wait_release_semaphore = f_wait_release;
    b->barrier = f_barrier;
    b->copy_out = f_copy_out;
    b->diagnostics = f_diag;
    b->free_buf = f_free;
    b->close = f_close;
}

/* ---- case helpers ------------------------------------------------------- */

static void reset_all(void)
{
    size_t i;
    int k;
    memset(&F, 0, sizeof F);
    for (k = 0; k < C_COUNT; k++) F.state_at[k] = -1;
    for (i = 0; i < N; i++) F.dev_out[i] = 0xA5A5A5A5u; /* garbage, never a poison word */
    F.klo = 0;
    F.khi = N;
    g_time = 1000u;
    memset(host_out, 0xCC, sizeof host_out);
    g_res = NULL;
    backend_init(&BE);
    omega_gpu_engine_test_reset_block();
    omega_gpu_engine_set_backend(&BE);
}

static void make_job(OmegaGpuJob *j)
{
    size_t i;
    for (i = 0; i < N; i++) {
        ina[i] = (uint32_t)(i + 1u);
        inb[i] = (uint32_t)(2u * i + 3u);
        poison[i] = 0xDEAD0000u + (uint32_t)(i * 7u); /* distinct per word, never an expected output */
    }
    memset(j, 0, sizeof *j);
    j->program = code;
    j->program_len = sizeof code;
    j->layout = OMEGA_GPU_LAYOUT_VECTOR_1D;
    j->element_count = N;
    j->input_a = ina;
    j->input_a_len = sizeof ina;
    j->input_b = inb;
    j->input_b_len = sizeof inb;
    j->output = host_out;
    j->output_len = N * sizeof(uint32_t);
    j->poison = poison;
    j->poison_count = N;
    j->timeouts.marker_ms = TMO_MARKER;
    j->timeouts.marker2_ms = TMO_MARKER2;
    j->timeouts.release_semaphore_ms = TMO_RELEASE;
    j->timeouts.visibility_ms = TMO_VISIBILITY;
    j->flags = 0;
}

static void inject(int call, int nth, int rc, int advance)
{
    F.inj[F.ninj].call = call;
    F.inj[F.ninj].nth = nth;
    F.inj[F.ninj].rc = rc;
    F.inj[F.ninj].advance = advance;
    F.ninj++;
}

static int run(const OmegaGpuJob *job, OmegaGpuResult *r)
{
    int rc;
    memset(r, 0x7f, sizeof *r); /* stale garbage the engine must fully overwrite */
    g_res = r;
    rc = omega_gpu_execute(job, r);
    g_res = NULL;
    return rc;
}

static void expect_log(const int *want, int n)
{
    int i;
    EXPECT(F.nlog == n);
    for (i = 0; i < n && i < F.nlog && i < LOGMAX; i++) EXPECT(F.log[i] == want[i]);
}

static void expect_failure(int rc, const OmegaGpuResult *r, int failure, int last, int step, int wait)
{
    EXPECT(rc == failure);
    EXPECT(r->failure == failure);
    EXPECT(r->last_state == last);
    EXPECT(r->failed_step == step);
    EXPECT(r->wait == wait);
}

static void expect_passthrough(const OmegaGpuResult *r, int err)
{
    EXPECT(r->drv_rc == DIAG_RC);
    EXPECT(r->err_no == err);
    EXPECT(strcmp(r->drv_text, DIAG_TEXT) == 0);
    EXPECT(r->sync_valid == (OMEGA_GPU_SYNC_MARKER | OMEGA_GPU_SYNC_MARKER2 | OMEGA_GPU_SYNC_SEMAPHORE));
    EXPECT(r->marker == 0x11111111u);
    EXPECT(r->marker2 == 0x22222222u);
    EXPECT(r->semaphore[0] == 0x33333333u);
    EXPECT(F.count[C_DIAG] == 1);
}

/* Device closed, every buffer freed, nothing broke the fake's protocol, no block. */
static void expect_released(const OmegaGpuResult *r)
{
    int role;
    EXPECT(F.dev_open == 0);
    for (role = 0; role < (int)OMEGA_GPU_BUF_ROLE_COUNT; role++) EXPECT(F.live[role] == 0);
    EXPECT(F.violations == 0);
    EXPECT(omega_gpu_engine_is_blocked() == 0);
    EXPECT(r->retained == 0);
}

/* Uncertain completion: nothing freed or closed, everything retained, process blocked. */
static void expect_retained(const OmegaGpuResult *r)
{
    int role;
    EXPECT(r->retained == 1);
    EXPECT(r->cleanup_failed == 0);
    EXPECT(F.count[C_FREE] == 0);
    EXPECT(F.count[C_CLOSE] == 0);
    EXPECT(F.dev_open == 1);
    for (role = 0; role < (int)OMEGA_GPU_BUF_ROLE_COUNT; role++) EXPECT(F.live[role] == 1);
    EXPECT(F.violations == 0);
    EXPECT(omega_gpu_engine_is_blocked() == 1);
}

static void expect_no_host_output(void)
{
    size_t i;
    for (i = 0; i < N; i++) EXPECT(host_out[i] == 0xCCCCCCCCu);
}

#define PRE_SUBMIT C_OPEN, C_CHANNEL, C_ALLOC, C_ALLOC, C_ALLOC, C_ALLOC, C_ALLOC, \
                   C_COPY_IN, C_COPY_IN, C_COPY_IN, C_POISON, C_BUILD, C_SUBMIT
#define FREE5 C_FREE, C_FREE, C_FREE, C_FREE, C_FREE
#define LOGLEN(a) ((int)(sizeof (a) / sizeof (a)[0]))

/* ---- cases -------------------------------------------------------------- */

static void case_success(void)
{
    OmegaGpuJob job;
    OmegaGpuResult r;
    size_t i;
    int rc;
    static const int want[] = { PRE_SUBMIT, C_MARKER, C_MARKER2, C_RELEASE, C_BARRIER, C_COPY_OUT, FREE5, C_CLOSE };

    reset_all();
    make_job(&job);
    rc = run(&job, &r);

    case_begin();
    EXPECT(rc == OMEGA_GPU_ENGINE_OK);
    EXPECT(r.failure == OMEGA_GPU_ENGINE_OK);
    EXPECT(r.last_state == OMEGA_GPU_ENGINE_STATE_SUCCESS);
    EXPECT(r.failed_step == OMEGA_GPU_ENGINE_STEP_NONE);
    EXPECT(r.wait == OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(r.wait_timeout_ms == 0 && r.waited_ms == 0);
    EXPECT(r.drv_rc == 0 && r.err_no == 0 && r.drv_text[0] == '\0');
    EXPECT(r.sync_valid == 0 && r.marker == 0 && r.marker2 == 0 && r.semaphore[0] == 0);
    EXPECT(r.retained == 0 && r.cleanup_failed == 0);
    EXPECT(r.n_outputs == 1u);
    EXPECT(r.output_unchanged_words[0] == 0);
    for (i = 0; i < N; i++) EXPECT(host_out[i] == exp_out(i));
    for (i = N; i < N + 4u; i++) EXPECT(host_out[i] == 0xCCCCCCCCu);
    EXPECT(F.count[C_DIAG] == 0);
    expect_released(&r);
    case_end("SUCCESS_FULL_RUN");

    case_begin();
    expect_log(want, LOGLEN(want));
    case_end("CALL_ORDER_FULL_RUN");

    case_begin();
    EXPECT(F.state_at[C_OPEN] == OMEGA_GPU_ENGINE_STATE_INITIAL);
    EXPECT(F.state_at[C_POISON] == OMEGA_GPU_ENGINE_STATE_INITIAL);
    EXPECT(F.state_at[C_BUILD] == OMEGA_GPU_ENGINE_STATE_INITIAL);
    EXPECT(F.state_at[C_SUBMIT] == OMEGA_GPU_ENGINE_STATE_PREPARED);
    EXPECT(F.state_at[C_MARKER] == OMEGA_GPU_ENGINE_STATE_SUBMITTED);
    EXPECT(F.state_at[C_MARKER2] == OMEGA_GPU_ENGINE_STATE_SUBMITTED);
    EXPECT(F.state_at[C_RELEASE] == OMEGA_GPU_ENGINE_STATE_SUBMITTED);
    EXPECT(F.state_at[C_BARRIER] == OMEGA_GPU_ENGINE_STATE_GPU_COMPLETE);
    EXPECT(F.state_at[C_COPY_OUT] == OMEGA_GPU_ENGINE_STATE_GPU_COMPLETE);
    EXPECT(F.state_at[C_FREE] == OMEGA_GPU_ENGINE_STATE_OUTPUT_PRODUCED);
    EXPECT(F.state_at[C_CLOSE] == OMEGA_GPU_ENGINE_STATE_OUTPUT_PRODUCED);
    case_end("STATE_ADVANCES_STEP_BY_STEP");

    case_begin();
    EXPECT(F.nalloc == 5);
    for (i = 0; i < 5u; i++) EXPECT(F.alloc_roles[i] == (int)i);
    EXPECT(F.alloc_bytes[OMEGA_GPU_BUF_SCRATCH] == 0u);
    EXPECT(F.alloc_bytes[OMEGA_GPU_BUF_PROGRAM] == sizeof code);
    EXPECT(F.alloc_bytes[OMEGA_GPU_BUF_INPUT_A] == sizeof ina);
    EXPECT(F.alloc_bytes[OMEGA_GPU_BUF_INPUT_B] == sizeof inb);
    EXPECT(F.alloc_bytes[OMEGA_GPU_BUF_OUTPUT] == N * sizeof(uint32_t));
    EXPECT(F.ncopy == 3);
    EXPECT(F.copy_roles[0] == OMEGA_GPU_BUF_PROGRAM && F.copy_bytes[0] == sizeof code);
    EXPECT(F.copy_roles[1] == OMEGA_GPU_BUF_INPUT_A && F.copy_bytes[1] == sizeof ina);
    EXPECT(F.copy_roles[2] == OMEGA_GPU_BUF_INPUT_B && F.copy_bytes[2] == sizeof inb);
    EXPECT(memcmp(F.dev_prog, code, sizeof code) == 0);
    EXPECT(F.build_job == &job);
    EXPECT(F.nfree == 5);
    EXPECT(F.free_roles[0] == OMEGA_GPU_BUF_OUTPUT);
    EXPECT(F.free_roles[1] == OMEGA_GPU_BUF_INPUT_B);
    EXPECT(F.free_roles[2] == OMEGA_GPU_BUF_INPUT_A);
    EXPECT(F.free_roles[3] == OMEGA_GPU_BUF_PROGRAM);
    EXPECT(F.free_roles[4] == OMEGA_GPU_BUF_SCRATCH);
    case_end("ROLES_BYTES_AND_REVERSE_FREE_ORDER");

    case_begin();
    EXPECT(F.tmo[C_MARKER] == TMO_MARKER);
    EXPECT(F.tmo[C_MARKER2] == TMO_MARKER2);
    EXPECT(F.tmo[C_RELEASE] == TMO_RELEASE);
    EXPECT(F.tmo[C_BARRIER] == TMO_VISIBILITY);
    case_end("PER_STEP_TIMEOUTS_FROM_JOB");

    case_begin();
    EXPECT(F.submit_seen == 1);
    EXPECT(memcmp(F.seen_at_submit, poison, sizeof poison) == 0);
    case_end("POISON_WRITTEN_BEFORE_SUBMIT");
}

static void case_no_c3(void)
{
    OmegaGpuJob job;
    OmegaGpuResult r;
    int rc;
    static const int want[] = { PRE_SUBMIT, C_MARKER, C_RELEASE, C_BARRIER, C_COPY_OUT, FREE5, C_CLOSE };

    reset_all();
    make_job(&job);
    job.flags = OMEGA_GPU_ENGINE_FLAG_NO_C3;
    job.timeouts.marker2_ms = 0; /* ignored with NO_C3 */
    rc = run(&job, &r);
    case_begin();
    EXPECT(rc == OMEGA_GPU_ENGINE_OK);
    EXPECT(r.last_state == OMEGA_GPU_ENGINE_STATE_SUCCESS);
    EXPECT(F.count[C_MARKER2] == 0);
    expect_log(want, LOGLEN(want));
    expect_released(&r);
    case_end("NO_C3_SKIPS_MARKER2_ONLY");
}

static void case_open_channel_alloc_prepare(void)
{
    OmegaGpuJob job;
    OmegaGpuResult r;
    int rc;

    {
        static const int want[] = { C_OPEN };
        reset_all(); make_job(&job);
        inject(C_OPEN, 1, -3, 0);
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_DEVICE_OPEN, OMEGA_GPU_ENGINE_STATE_INITIAL,
                       OMEGA_GPU_ENGINE_STEP_OPEN_DEVICE, OMEGA_GPU_ENGINE_WAIT_NONE);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        expect_released(&r);
        expect_no_host_output();
        case_end("FAIL_DEVICE_OPEN");
    }
    {
        static const int want[] = { C_OPEN, C_CHANNEL, C_CLOSE };
        reset_all(); make_job(&job);
        inject(C_CHANNEL, 1, -3, 0);
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_CHANNEL_CREATE, OMEGA_GPU_ENGINE_STATE_INITIAL,
                       OMEGA_GPU_ENGINE_STEP_CREATE_CHANNEL, OMEGA_GPU_ENGINE_WAIT_NONE);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        expect_released(&r);
        expect_no_host_output();
        case_end("FAIL_CHANNEL_CREATE");
    }
    {
        /* the third allocation (input A) fails: scratch and program are freed, newest first */
        static const int want[] = { C_OPEN, C_CHANNEL, C_ALLOC, C_ALLOC, C_ALLOC, C_FREE, C_FREE, C_CLOSE };
        reset_all(); make_job(&job);
        inject(C_ALLOC, 3, -3, 0);
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_ALLOC, OMEGA_GPU_ENGINE_STATE_INITIAL,
                       OMEGA_GPU_ENGINE_STEP_ALLOC, OMEGA_GPU_ENGINE_WAIT_NONE);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        EXPECT(F.nfree == 2);
        EXPECT(F.free_roles[0] == OMEGA_GPU_BUF_PROGRAM && F.free_roles[1] == OMEGA_GPU_BUF_SCRATCH);
        expect_released(&r);
        expect_no_host_output();
        case_end("FAIL_ALLOC");
    }
    {
        /* the second copy-in (input A) fails */
        static const int want[] = { C_OPEN, C_CHANNEL, C_ALLOC, C_ALLOC, C_ALLOC, C_ALLOC, C_ALLOC,
                                   C_COPY_IN, C_COPY_IN, FREE5, C_CLOSE };
        reset_all(); make_job(&job);
        inject(C_COPY_IN, 2, -3, 0);
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_PREPARE, OMEGA_GPU_ENGINE_STATE_INITIAL,
                       OMEGA_GPU_ENGINE_STEP_COPY_IN, OMEGA_GPU_ENGINE_WAIT_NONE);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        expect_released(&r);
        case_end("FAIL_PREPARE_COPY_IN");
    }
    {
        static const int want[] = { C_OPEN, C_CHANNEL, C_ALLOC, C_ALLOC, C_ALLOC, C_ALLOC, C_ALLOC,
                                   C_COPY_IN, C_COPY_IN, C_COPY_IN, C_POISON, FREE5, C_CLOSE };
        reset_all(); make_job(&job);
        inject(C_POISON, 1, -3, 0);
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_PREPARE, OMEGA_GPU_ENGINE_STATE_INITIAL,
                       OMEGA_GPU_ENGINE_STEP_FILL_POISON, OMEGA_GPU_ENGINE_WAIT_NONE);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        expect_released(&r);
        case_end("FAIL_PREPARE_FILL_POISON");
    }
    {
        static const int want[] = { C_OPEN, C_CHANNEL, C_ALLOC, C_ALLOC, C_ALLOC, C_ALLOC, C_ALLOC,
                                   C_COPY_IN, C_COPY_IN, C_COPY_IN, C_POISON, C_BUILD, FREE5, C_CLOSE };
        reset_all(); make_job(&job);
        inject(C_BUILD, 1, -3, 0);
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_PREPARE, OMEGA_GPU_ENGINE_STATE_INITIAL,
                       OMEGA_GPU_ENGINE_STEP_BUILD, OMEGA_GPU_ENGINE_WAIT_NONE);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        expect_released(&r);
        case_end("FAIL_PREPARE_BUILD");
    }
}

static void case_submit_and_visibility(void)
{
    OmegaGpuJob job;
    OmegaGpuResult r;
    int rc;

    {
        static const int want[] = { PRE_SUBMIT, FREE5, C_CLOSE };
        reset_all(); make_job(&job);
        inject(C_SUBMIT, 1, -3, 0);
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_SUBMIT, OMEGA_GPU_ENGINE_STATE_PREPARED,
                       OMEGA_GPU_ENGINE_STEP_SUBMIT, OMEGA_GPU_ENGINE_WAIT_NONE);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        EXPECT(F.submit_seen == 0);
        expect_released(&r); /* SUBMIT is a known failure: cleaned up, not blocked */
        expect_no_host_output();
        case_end("FAIL_SUBMIT_IS_NOT_UNCERTAIN");
    }
    {
        static const int want[] = { PRE_SUBMIT, C_MARKER, C_MARKER2, C_RELEASE, C_BARRIER, FREE5, C_CLOSE };
        reset_all(); make_job(&job);
        inject(C_BARRIER, 1, OMEGA_GPU_BACKEND_TIMEOUT, 0);
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_VISIBILITY_WAIT, OMEGA_GPU_ENGINE_STATE_GPU_COMPLETE,
                       OMEGA_GPU_ENGINE_STEP_BARRIER, OMEGA_GPU_ENGINE_WAIT_NONE);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        expect_released(&r); /* completion is known: cleaned up, not blocked */
        expect_no_host_output();
        case_end("FAIL_VISIBILITY_BARRIER");
    }
    {
        static const int want[] = { PRE_SUBMIT, C_MARKER, C_MARKER2, C_RELEASE, C_BARRIER, C_COPY_OUT, FREE5, C_CLOSE };
        reset_all(); make_job(&job);
        inject(C_COPY_OUT, 1, -3, 0);
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_VISIBILITY_WAIT, OMEGA_GPU_ENGINE_STATE_GPU_COMPLETE,
                       OMEGA_GPU_ENGINE_STEP_COPY_OUT, OMEGA_GPU_ENGINE_WAIT_NONE);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        expect_released(&r);
        case_end("FAIL_VISIBILITY_COPY_OUT");
    }
}

static void case_uncertain(void)
{
    OmegaGpuJob job;
    OmegaGpuResult r;
    OmegaGpuResult r2;
    int rc;
    int nlog_before;
    int k;

    {
        static const int want[] = { PRE_SUBMIT, C_MARKER };
        reset_all(); make_job(&job);
        inject(C_MARKER, 1, OMEGA_GPU_BACKEND_TIMEOUT, 4321);
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_COMPLETION_WAIT, OMEGA_GPU_ENGINE_STATE_SUBMITTED,
                       OMEGA_GPU_ENGINE_STEP_WAIT_MARKER, OMEGA_GPU_ENGINE_WAIT_MARKER);
        EXPECT(r.wait_timeout_ms == TMO_MARKER);
        EXPECT(r.waited_ms == 4321u);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        expect_retained(&r);
        expect_no_host_output();
        case_end("UNCERTAIN_MARKER_TIMEOUT");

        /* the next job in this process is refused with no backend call */
        nlog_before = F.nlog;
        k = F.count[C_DIAG];
        rc = run(&job, &r2);
        case_begin();
        expect_failure(rc, &r2, OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED,
                       OMEGA_GPU_ENGINE_STATE_INITIAL, OMEGA_GPU_ENGINE_STEP_BLOCKED,
                       OMEGA_GPU_ENGINE_WAIT_NONE);
        EXPECT(F.nlog == nlog_before);
        EXPECT(F.count[C_DIAG] == k);
        EXPECT(r2.retained == 0);
        EXPECT(omega_gpu_engine_is_blocked() == 1);
        case_end("UNCERTAIN_BLOCKS_NEXT_JOB_NO_BACKEND_CALL");

        /* test reset clears it; a fresh fake then runs a good job */
        omega_gpu_engine_test_reset_block();
        case_begin();
        EXPECT(omega_gpu_engine_is_blocked() == 0);
        reset_all();
        rc = run(&job, &r2);
        EXPECT(rc == OMEGA_GPU_ENGINE_OK);
        EXPECT(r2.last_state == OMEGA_GPU_ENGINE_STATE_SUCCESS);
        case_end("UNCERTAIN_TEST_RESET_CLEARS_BLOCK");
    }
    {
        static const int want[] = { PRE_SUBMIT, C_MARKER, C_MARKER2 };
        reset_all(); make_job(&job);
        inject(C_MARKER2, 1, OMEGA_GPU_BACKEND_UNKNOWN_STATE, 2500);
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_COMPLETION_WAIT, OMEGA_GPU_ENGINE_STATE_SUBMITTED,
                       OMEGA_GPU_ENGINE_STEP_WAIT_MARKER2, OMEGA_GPU_ENGINE_WAIT_MARKER2);
        EXPECT(r.wait_timeout_ms == TMO_MARKER2);
        EXPECT(r.waited_ms == 2500u);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        expect_retained(&r);
        case_end("UNCERTAIN_MARKER2_UNKNOWN_STATE");
        omega_gpu_engine_test_reset_block();
    }
    {
        static const int want[] = { PRE_SUBMIT, C_MARKER, C_MARKER2, C_RELEASE };
        reset_all(); make_job(&job);
        inject(C_RELEASE, 1, -5, 777); /* a driver fault, not a timeout */
        rc = run(&job, &r);
        case_begin();
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_RELEASE_WAIT, OMEGA_GPU_ENGINE_STATE_SUBMITTED,
                       OMEGA_GPU_ENGINE_STEP_WAIT_RELEASE_SEMAPHORE,
                       OMEGA_GPU_ENGINE_WAIT_RELEASE_SEMAPHORE);
        EXPECT(r.wait_timeout_ms == TMO_RELEASE);
        EXPECT(r.waited_ms == 777u);
        expect_passthrough(&r, INJ_ERRNO);
        expect_log(want, LOGLEN(want));
        expect_retained(&r);
        case_end("UNCERTAIN_RELEASE_DRIVER_FAULT");
        omega_gpu_engine_test_reset_block();
    }
}

static void case_unchanged(void)
{
    OmegaGpuJob job;
    OmegaGpuResult r;
    size_t i;
    int rc;

    /* the fake kernel writes nothing: all 16 words are still poison */
    reset_all(); make_job(&job);
    F.klo = 0; F.khi = 0;
    rc = run(&job, &r);
    case_begin();
    expect_failure(rc, &r, OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED, OMEGA_GPU_ENGINE_STATE_OUTPUT_VISIBLE,
                   OMEGA_GPU_ENGINE_STEP_SCAN, OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(r.n_outputs == 1u);
    EXPECT(r.output_unchanged_words[0] == 16u);
    EXPECT(r.err_no == 0);
    EXPECT(r.drv_rc == DIAG_RC); /* observed words are kept on every failure */
    EXPECT(F.count[C_DIAG] == 1);
    for (i = 0; i < N; i++) EXPECT(host_out[i] == poison[i]);
    expect_released(&r);
    case_end("UNCHANGED_NOTHING_WRITTEN");

    /* only words 4..9 are written: 10 words are still poison */
    reset_all(); make_job(&job);
    F.klo = 4; F.khi = 10;
    rc = run(&job, &r);
    case_begin();
    expect_failure(rc, &r, OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED, OMEGA_GPU_ENGINE_STATE_OUTPUT_VISIBLE,
                   OMEGA_GPU_ENGINE_STEP_SCAN, OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(r.output_unchanged_words[0] == 10u);
    for (i = 0; i < N; i++) EXPECT(host_out[i] == ((i >= 4u && i < 10u) ? exp_out(i) : poison[i]));
    expect_released(&r);
    case_end("UNCHANGED_PARTIAL_EXACT_COUNT");

    /* all but the last word written: exactly 1 */
    reset_all(); make_job(&job);
    F.klo = 0; F.khi = N - 1u;
    rc = run(&job, &r);
    case_begin();
    expect_failure(rc, &r, OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED, OMEGA_GPU_ENGINE_STATE_OUTPUT_VISIBLE,
                   OMEGA_GPU_ENGINE_STEP_SCAN, OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(r.output_unchanged_words[0] == 1u);
    EXPECT(host_out[N - 1u] == poison[N - 1u]);
    expect_released(&r);
    case_end("UNCHANGED_LAST_WORD_ONLY");

    /* only the first word is unwritten: exactly 1 */
    reset_all(); make_job(&job);
    F.klo = 1; F.khi = N;
    rc = run(&job, &r);
    case_begin();
    expect_failure(rc, &r, OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED, OMEGA_GPU_ENGINE_STATE_OUTPUT_VISIBLE,
                   OMEGA_GPU_ENGINE_STEP_SCAN, OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(r.output_unchanged_words[0] == 1u);
    EXPECT(host_out[0] == poison[0]);
    expect_released(&r);
    case_end("UNCHANGED_FIRST_WORD_ONLY");

    /* the fake kernel writes everything: SUCCESS and a zero count */
    reset_all(); make_job(&job);
    F.klo = 0; F.khi = N;
    rc = run(&job, &r);
    case_begin();
    EXPECT(rc == OMEGA_GPU_ENGINE_OK);
    EXPECT(r.last_state == OMEGA_GPU_ENGINE_STATE_SUCCESS);
    EXPECT(r.n_outputs == 1u && r.output_unchanged_words[0] == 0u);
    case_end("UNCHANGED_ALL_WRITTEN_IS_SUCCESS");

    /* a silent kernel is only caught because the engine wrote the poison to the
     * device output before submit (the fake device starts with other garbage). */
    reset_all(); make_job(&job);
    F.klo = 0; F.khi = 0;
    rc = run(&job, &r);
    case_begin();
    EXPECT(memcmp(F.seen_at_submit, poison, sizeof poison) == 0);
    EXPECT(rc == OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED);
    case_end("UNCHANGED_NEEDS_POISON_FILL");
}

static void case_cleanup_faults(void)
{
    OmegaGpuJob job;
    OmegaGpuResult r;
    size_t i;
    int rc;

    /* free fails on the 2nd free after an otherwise good run: the rest still run */
    reset_all(); make_job(&job);
    inject(C_FREE, 2, -9, 0);
    rc = run(&job, &r);
    case_begin();
    expect_failure(rc, &r, OMEGA_GPU_ENGINE_CLEANUP, OMEGA_GPU_ENGINE_STATE_OUTPUT_PRODUCED,
                   OMEGA_GPU_ENGINE_STEP_FREE, OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(r.cleanup_failed == 1);
    EXPECT(r.retained == 0);
    expect_passthrough(&r, INJ_ERRNO);
    EXPECT(F.count[C_FREE] == 5);
    EXPECT(F.count[C_CLOSE] == 1);
    for (i = 0; i < N; i++) EXPECT(host_out[i] == exp_out(i));
    EXPECT(omega_gpu_engine_is_blocked() == 0);
    case_end("CLEANUP_FREE_FAILS_AFTER_GOOD_RUN");

    reset_all(); make_job(&job);
    inject(C_CLOSE, 1, -9, 0);
    rc = run(&job, &r);
    case_begin();
    expect_failure(rc, &r, OMEGA_GPU_ENGINE_CLEANUP, OMEGA_GPU_ENGINE_STATE_OUTPUT_PRODUCED,
                   OMEGA_GPU_ENGINE_STEP_CLOSE, OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(r.cleanup_failed == 1);
    expect_passthrough(&r, INJ_ERRNO);
    EXPECT(F.count[C_FREE] == 5);
    EXPECT(omega_gpu_engine_is_blocked() == 0);
    case_end("CLEANUP_CLOSE_FAILS_AFTER_GOOD_RUN");

    /* a free fault and then a close fault: the first (FREE) is kept */
    reset_all(); make_job(&job);
    inject(C_FREE, 1, -9, 0);
    inject(C_CLOSE, 1, -9, 0);
    rc = run(&job, &r);
    case_begin();
    expect_failure(rc, &r, OMEGA_GPU_ENGINE_CLEANUP, OMEGA_GPU_ENGINE_STATE_OUTPUT_PRODUCED,
                   OMEGA_GPU_ENGINE_STEP_FREE, OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(r.cleanup_failed == 1);
    EXPECT(F.count[C_DIAG] == 1);
    case_end("CLEANUP_FIRST_CLEANUP_FAULT_KEPT");

    /* an earlier failure (submit) then a free fault: the first failure is kept */
    reset_all(); make_job(&job);
    inject(C_SUBMIT, 1, -3, 0);
    inject(C_FREE, 1, -9, 0);
    rc = run(&job, &r);
    case_begin();
    expect_failure(rc, &r, OMEGA_GPU_ENGINE_SUBMIT, OMEGA_GPU_ENGINE_STATE_PREPARED,
                   OMEGA_GPU_ENGINE_STEP_SUBMIT, OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(r.cleanup_failed == 1);
    EXPECT(F.count[C_FREE] == 5);
    EXPECT(F.count[C_CLOSE] == 1);
    EXPECT(F.count[C_DIAG] == 1);
    expect_passthrough(&r, INJ_ERRNO);
    case_end("CLEANUP_FAULT_NEVER_OVERWRITES_FIRST_FAILURE");

    /* unchanged output plus a close fault: still OUTPUT_UNCHANGED */
    reset_all(); make_job(&job);
    F.klo = 0; F.khi = 0;
    inject(C_CLOSE, 1, -9, 0);
    rc = run(&job, &r);
    case_begin();
    expect_failure(rc, &r, OMEGA_GPU_ENGINE_OUTPUT_UNCHANGED, OMEGA_GPU_ENGINE_STATE_OUTPUT_VISIBLE,
                   OMEGA_GPU_ENGINE_STEP_SCAN, OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(r.cleanup_failed == 1);
    EXPECT(r.output_unchanged_words[0] == 16u);
    case_end("CLEANUP_FAULT_AFTER_UNCHANGED_KEEPS_UNCHANGED");
}

#define EXPECT_INVALID(edit) do { \
        OmegaGpuJob bj; \
        OmegaGpuResult br; \
        int brc; \
        reset_all(); make_job(&bj); \
        edit; \
        brc = run(&bj, &br); \
        expect_failure(brc, &br, OMEGA_GPU_ENGINE_INVALID_ARGS, OMEGA_GPU_ENGINE_STATE_INITIAL, \
                       OMEGA_GPU_ENGINE_STEP_VALIDATE, OMEGA_GPU_ENGINE_WAIT_NONE); \
        EXPECT(F.nlog == 0 && F.count[C_DIAG] == 0); \
        EXPECT(br.drv_rc == 0 && br.err_no == 0 && br.retained == 0 && br.n_outputs == 0u); \
        EXPECT(omega_gpu_engine_is_blocked() == 0); \
    } while (0)

static void case_invalid_args(void)
{
    OmegaGpuJob job;
    OmegaGpuResult r;
    int rc;

    case_begin();
    EXPECT_INVALID(bj.program = NULL);
    EXPECT_INVALID(bj.program_len = 0);
    case_end("INVALID_ARGS_PROGRAM");

    case_begin();
    EXPECT_INVALID(bj.layout = OMEGA_GPU_LAYOUT_COUNT);
    case_end("INVALID_ARGS_LAYOUT");

    case_begin();
    /* zero elements with every length zeroed too: only the zero-count check rejects this */
    EXPECT_INVALID(bj.element_count = 0; bj.input_a_len = 0; bj.input_b_len = 0; bj.output_len = 0; bj.poison_count = 0);
    EXPECT_INVALID(bj.element_count = 0);
    case_end("INVALID_ARGS_ZERO_ELEMENTS");

    case_begin();
    EXPECT_INVALID(bj.input_a = NULL);
    EXPECT_INVALID(bj.input_a_len = sizeof ina - 4u);
    EXPECT_INVALID(bj.input_b = NULL);
    EXPECT_INVALID(bj.input_b_len = sizeof inb + 4u);
    EXPECT_INVALID(bj.output = NULL);
    EXPECT_INVALID(bj.output_len = N * sizeof(uint32_t) - 4u);
    case_end("INVALID_ARGS_BUFFERS");

    case_begin();
    EXPECT_INVALID(bj.output = (unsigned char *)host_out + 1);
    case_end("INVALID_ARGS_OUTPUT_ALIGNMENT");

    case_begin();
    EXPECT_INVALID(bj.poison = NULL);
    EXPECT_INVALID(bj.poison_count = N - 1u);
    EXPECT_INVALID(bj.poison_count = N + 1u);
    case_end("INVALID_ARGS_POISON");

    case_begin();
    EXPECT_INVALID(bj.flags = 0x2u);
    case_end("INVALID_ARGS_UNKNOWN_FLAG");

    case_begin();
    EXPECT_INVALID(bj.timeouts.marker_ms = 0);
    case_end("INVALID_ARGS_ZERO_MARKER_TIMEOUT");
    case_begin();
    EXPECT_INVALID(bj.timeouts.release_semaphore_ms = 0);
    case_end("INVALID_ARGS_ZERO_RELEASE_TIMEOUT");
    case_begin();
    EXPECT_INVALID(bj.timeouts.visibility_ms = 0);
    case_end("INVALID_ARGS_ZERO_VISIBILITY_TIMEOUT");
    case_begin();
    EXPECT_INVALID(bj.timeouts.marker2_ms = 0); /* C3 on: marker2 needs a timeout */
    case_end("INVALID_ARGS_ZERO_MARKER2_TIMEOUT_WITH_C3");

    case_begin();
    reset_all();
    memset(&r, 0x7f, sizeof r);
    rc = omega_gpu_execute(NULL, &r);
    EXPECT(rc == OMEGA_GPU_ENGINE_INVALID_ARGS);
    EXPECT(r.failure == OMEGA_GPU_ENGINE_INVALID_ARGS && r.failed_step == OMEGA_GPU_ENGINE_STEP_VALIDATE);
    EXPECT(r.last_state == OMEGA_GPU_ENGINE_STATE_INITIAL);
    EXPECT(F.nlog == 0);
    case_end("INVALID_ARGS_NULL_JOB");

    case_begin();
    reset_all(); make_job(&job);
    rc = omega_gpu_execute(&job, NULL);
    EXPECT(rc == OMEGA_GPU_ENGINE_INVALID_ARGS);
    EXPECT(F.nlog == 0);
    case_end("INVALID_ARGS_NULL_RESULT");
}

static void case_block_and_backend(void)
{
    OmegaGpuJob job;
    OmegaGpuResult r;
    static OmegaGpuBackend be2; /* static: it stays the default backend until main clears it */
    int rc;

    /* forced block refuses even a malformed or NULL job, with no backend call */
    reset_all(); make_job(&job);
    omega_gpu_engine_test_force_block();
    rc = run(&job, &r);
    case_begin();
    expect_failure(rc, &r, OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED,
                   OMEGA_GPU_ENGINE_STATE_INITIAL, OMEGA_GPU_ENGINE_STEP_BLOCKED,
                   OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(F.nlog == 0);
    job.element_count = 0;
    rc = run(&job, &r);
    EXPECT(rc == OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED);
    memset(&r, 0x7f, sizeof r);
    rc = omega_gpu_execute(NULL, &r);
    EXPECT(rc == OMEGA_GPU_ENGINE_UNCERTAIN_COMPLETION_BLOCKED);
    EXPECT(F.nlog == 0);
    EXPECT(omega_gpu_engine_is_blocked() == 1);
    case_end("BLOCKED_REFUSES_BEFORE_ANY_CHECK_OR_CALL");

    omega_gpu_engine_test_reset_block();
    case_begin();
    EXPECT(omega_gpu_engine_is_blocked() == 0);
    make_job(&job);
    rc = run(&job, &r);
    EXPECT(rc == OMEGA_GPU_ENGINE_OK);
    case_end("BLOCK_RESET_ALLOWS_JOBS_AGAIN");

    /* no backend set: the defined stub result */
    reset_all(); make_job(&job);
    omega_gpu_engine_set_backend(NULL);
    rc = run(&job, &r);
    case_begin();
    expect_failure(rc, &r, OMEGA_GPU_ENGINE_INTERNAL_INVARIANT, OMEGA_GPU_ENGINE_STATE_INITIAL,
                   OMEGA_GPU_ENGINE_STEP_NOT_IMPLEMENTED, OMEGA_GPU_ENGINE_WAIT_NONE);
    EXPECT(F.nlog == 0);
    case_end("NO_BACKEND_NOT_IMPLEMENTED");

    /* an incomplete backend table is refused before any call: each callback in turn */
    case_begin();
#define EXPECT_INCOMPLETE(field) do { \
        reset_all(); make_job(&job); \
        be2 = BE; be2.field = NULL; \
        omega_gpu_engine_set_backend(&be2); \
        rc = run(&job, &r); \
        expect_failure(rc, &r, OMEGA_GPU_ENGINE_INTERNAL_INVARIANT, OMEGA_GPU_ENGINE_STATE_INITIAL, \
                       OMEGA_GPU_ENGINE_STEP_BACKEND_CHECK, OMEGA_GPU_ENGINE_WAIT_NONE); \
        EXPECT(F.nlog == 0); \
    } while (0)
    EXPECT_INCOMPLETE(open_device);
    EXPECT_INCOMPLETE(create_channel);
    EXPECT_INCOMPLETE(alloc);
    EXPECT_INCOMPLETE(copy_in);
    EXPECT_INCOMPLETE(fill_poison);
    EXPECT_INCOMPLETE(build);
    EXPECT_INCOMPLETE(submit);
    EXPECT_INCOMPLETE(wait_marker);
    EXPECT_INCOMPLETE(wait_marker2);
    EXPECT_INCOMPLETE(wait_release_semaphore);
    EXPECT_INCOMPLETE(barrier);
    EXPECT_INCOMPLETE(copy_out);
    EXPECT_INCOMPLETE(diagnostics);
    EXPECT_INCOMPLETE(free_buf);
    EXPECT_INCOMPLETE(close);
    case_end("BACKEND_TABLE_INCOMPLETE_REFUSED");
}

int main(void)
{
    omega_gpu_engine_test_set_clock(fake_now);

    case_success();
    case_no_c3();
    case_open_channel_alloc_prepare();
    case_submit_and_visibility();
    case_uncertain();
    case_unchanged();
    case_cleanup_faults();
    case_invalid_args();
    case_block_and_backend();

    omega_gpu_engine_set_backend(NULL);
    omega_gpu_engine_test_reset_block();
    omega_gpu_engine_test_set_clock(NULL);
    printf("SUMMARY passed=%d failed=%d\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
