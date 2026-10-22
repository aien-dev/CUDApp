/* SPDX-License-Identifier: MIT */
#include "omega_gpu_admission.h"
#include "omega_gpu_platform_linux.h"
#include "fake_m16_native.h"
#include "nvrm.h"
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* Same missing fake-driver symbols as the existing Linux platform host test. */
int nvrm_open(Nvrm *rm) { (void)rm; fk.device_open = 1; return 0; }
int nvrm_close(Nvrm *rm) { (void)rm; return 0; }
void nvrm_retire(Nvrm *rm, uint32_t upto) { (void)rm; (void)upto; }
static int interrupt_write_fd = -1;
ssize_t __real_pwrite(int, const void *, size_t, off_t);
ssize_t __wrap_pwrite(int fd, const void *data, size_t bytes, off_t offset) {
    if (interrupt_write_fd >= 0 && bytes > 4096) {
        ssize_t n = __real_pwrite(fd, data, bytes / 2, offset);
        char ready = 'r';
        if (n < 0 || write(interrupt_write_fd, &ready, 1) != 1) _exit(3);
        raise(SIGSTOP); /* parent kills only this fixture child */
        _exit(4);
    }
    return __real_pwrite(fd, data, bytes, offset);
}

static unsigned checks;
#define CHECK(e) do { checks++; if (!(e)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #e); exit(1); } } while (0)
static char root[256], mempath[300];
#include "admission_state_fixture.h"
static OmegaGpuAdmissionConfig config;
static void fixture(const char *text) {
    FILE *f = fopen(mempath, "w"); CHECK(f != NULL); CHECK(fputs(text, f) >= 0); CHECK(fclose(f) == 0);
}
static int op_open(OmegaGpuAdmission **a) {
    int rc;
    for (int i = 0; i < 2000; i++) {
        rc = omega_gpu_admission_open(&config, a);
        if (rc != -EBUSY) return rc;
        struct timespec t = {0, 1000000}; nanosleep(&t, NULL);
    }
    return -ETIMEDOUT;
}
static int reserve(OmegaGpuAdmission *a, uint64_t n, OmegaGpuReservation *r) {
    int rc;
    for (int i = 0; i < 2000; i++) {
        rc = omega_gpu_admission_reserve(a, n, OMEGA_ADMISSION_KV, r);
        if (rc != -EBUSY) return rc;
        struct timespec t = {0, 1000000}; nanosleep(&t, NULL);
    }
    return -ETIMEDOUT;
}
static void io(int fd, void *p, size_t n, int writing) {
    CHECK((writing ? write(fd, p, n) : read(fd, p, n)) == (ssize_t)n);
}
static void wait_ok(pid_t pid) {
    int status; CHECK(waitpid(pid, &status, 0) == pid); CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
static void transitions(void) {
    OmegaGpuAdmission *a; CHECK(!op_open(&a));
    OmegaGpuReservation r, stale, denied;
    CHECK(!reserve(a, 8192, &r)); stale = r;
    CHECK(reserve(a, 1, &denied) == -ENOSPC && !denied.generation);
    CHECK(omega_gpu_admission_close(a) == -EBUSY);
    CHECK(omega_gpu_admission_commit(a, r) == -EPERM);
    CHECK(omega_gpu_admission_release(a, r) == -EPERM);
    CHECK(!omega_gpu_admission_acquire(a, r));
    CHECK(omega_gpu_admission_acquire(a, r) == -EPERM);
    CHECK(omega_gpu_admission_cancel(a, r) == -EPERM);
    CHECK(!omega_gpu_admission_abort(a, r));
    CHECK(omega_gpu_admission_acquire(a, stale) == -ESTALE);
    CHECK(!reserve(a, 8192, &r)); CHECK(r.generation > stale.generation);
    CHECK(omega_gpu_admission_cancel(a, stale) == -ESTALE);
    CHECK(!omega_gpu_admission_acquire(a, r)); CHECK(!omega_gpu_admission_commit(a, r));
    CHECK(omega_gpu_admission_abort(a, r) == -EPERM);
    OmegaGpuAdmissionStats s;
    CHECK(!omega_gpu_admission_stats(a, &s));
    CHECK(s.committed_bytes == 8192 && !s.reserved_bytes && !s.acquired_bytes && s.by_usage[OMEGA_ADMISSION_KV] == 8192);
    CHECK(s.process_rss_bytes > 0 && !s.overhead_known && !s.overhead_bytes && !s.by_usage[OMEGA_ADMISSION_SHARED_WEIGHT]);
    CHECK(!omega_gpu_admission_release(a, r));
    CHECK(omega_gpu_admission_release(a, r) == -ESTALE);
    CHECK(reserve(a, UINT64_MAX, &r) == -EINVAL);
    CHECK(reserve(a, 0, &r) == -EINVAL);
    CHECK(omega_gpu_admission_reserve(a, 1, OMEGA_ADMISSION_SHARED_WEIGHT, &r) == -EINVAL);
    CHECK(omega_gpu_admission_reserve(a, 1, (OmegaGpuAdmissionUsage)-1, &r) == -EINVAL);
    CHECK(!reserve(a, 1, &r)); CHECK(!omega_gpu_admission_seal(a, 1));
    CHECK(omega_gpu_admission_acquire(a, r) == -EPERM);
    CHECK(reserve(a, 1, &denied) == -EPERM);
    CHECK(!omega_gpu_admission_cancel(a, r)); CHECK(!omega_gpu_admission_seal(a, 0));
    CHECK(!omega_gpu_admission_close(a));
    puts("PASS transitions, stale generations, accounting and seal");
}
/* Children stay alive until both results are observed, so no winner can free
 * its charge before the loser tests the shared budget. They independently open
 * the actual admission implementation after a synchronized start. */
static void contenders(void) {
    int start[2], results[2], finish[2]; CHECK(!pipe(start)); CHECK(!pipe(results)); CHECK(!pipe(finish));
    pid_t children[2];
    for (int i = 0; i < 2; i++) {
        children[i] = fork(); CHECK(children[i] >= 0);
        if (!children[i]) {
            close(start[1]); close(results[0]); close(finish[1]);
            char go; io(start[0], &go, 1, 0);
            OmegaGpuAdmission *a; CHECK(!op_open(&a));
            OmegaGpuReservation r; int rc = reserve(a, 8192, &r);
            CHECK(rc == 0 || rc == -ENOSPC);
            io(results[1], &rc, sizeof rc, 1); io(finish[0], &go, 1, 0);
            /* Exit without cancellation: parent also verifies crash recovery. */
            _exit(0);
        }
    }
    close(start[0]); close(results[1]); close(finish[0]);
    char go[2] = {'g','g'}; io(start[1], go, 2, 1);
    int r[2]; io(results[0], &r[0], sizeof(int), 0); io(results[0], &r[1], sizeof(int), 0);
    CHECK((r[0] == 0) + (r[1] == 0) == 1);
    io(finish[1], go, 2, 1); wait_ok(children[0]); wait_ok(children[1]);
    close(start[1]); close(results[0]); close(finish[1]);
    OmegaGpuAdmission *a; OmegaGpuReservation t;
    CHECK(!op_open(&a)); CHECK(!reserve(a, 8192, &t)); CHECK(!omega_gpu_admission_cancel(a, t)); CHECK(!omega_gpu_admission_close(a));
    puts("PASS synchronized real-process contenders and abandoned reservation cleanup");
}
static void death_and_ownership(void) {
    OmegaGpuAdmission *parent; CHECK(!op_open(&parent));
    OmegaGpuReservation parent_token; CHECK(!reserve(parent, 1, &parent_token));
    int reports[2], finish[2]; CHECK(!pipe(reports)); CHECK(!pipe(finish));
    pid_t p = fork(); CHECK(p >= 0);
    if (!p) {
        close(reports[0]); close(finish[1]);
        OmegaGpuReservation ignored;
        CHECK(omega_gpu_admission_reserve(parent, 1, OMEGA_ADMISSION_KV, &ignored) == -EPERM);
        OmegaGpuAdmission *a; CHECK(!op_open(&a));
        CHECK(omega_gpu_admission_cancel(a, parent_token) == -ESTALE);
        CHECK(!reserve(a, 4096, &ignored)); CHECK(!omega_gpu_admission_acquire(a, ignored)); CHECK(!omega_gpu_admission_commit(a, ignored));
        io(reports[1], &ignored, sizeof ignored, 1);
        char go; io(finish[0], &go, 1, 0); _exit(0);
    }
    close(reports[1]); close(finish[0]);
    OmegaGpuReservation child_token; io(reports[0], &child_token, sizeof child_token, 0);
    CHECK(omega_gpu_admission_release(parent, child_token) == -ESTALE);
    OmegaGpuReservation denied; CHECK(reserve(parent, 8192, &denied) == -ENOSPC);
    CHECK(kill(p, SIGKILL) == 0); int status; CHECK(waitpid(p, &status, 0) == p); CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
    CHECK(!omega_gpu_admission_cancel(parent, parent_token));
    CHECK(!reserve(parent, 8192, &denied)); CHECK(omega_gpu_admission_release(parent, child_token) == -ESTALE);
    CHECK(!omega_gpu_admission_cancel(parent, denied)); CHECK(!omega_gpu_admission_close(parent));
    close(reports[0]); close(finish[1]);
    puts("PASS SIGKILL committed-owner recovery, foreign tokens and inherited-context refusal");
}
static void reused_identity(void) {
    OmegaGpuAdmission *a; CHECK(!op_open(&a));
    for (int previous_boot = 0; previous_boot <= 1; previous_boot++) {
        OmegaGpuReservation old, next;
        CHECK(!reserve(a, 8192, &old));
        forge_owner(old, previous_boot);
        CHECK(omega_gpu_admission_acquire(a, old) == -ESTALE);
        CHECK(!reserve(a, 8192, &next)); CHECK(next.generation > old.generation);
        CHECK(!omega_gpu_admission_cancel(a, next));
    }
    CHECK(!omega_gpu_admission_close(a));
    puts("PASS live-PID/different-start and previous-boot ownership FIXTURES");
}
static void interrupted_transaction(void) {
    OmegaGpuAdmission *a; CHECK(!op_open(&a));
    OmegaGpuReservation original; CHECK(!reserve(a, 1, &original));
    int report[2]; CHECK(!pipe(report)); pid_t child = fork(); CHECK(child >= 0);
    if (!child) {
        close(report[0]); OmegaGpuAdmission *b; CHECK(!op_open(&b));
        interrupt_write_fd = report[1]; OmegaGpuReservation t;
        (void)reserve(b, 4096, &t); _exit(5);
    }
    close(report[1]); char ready; io(report[0], &ready, 1, 0); CHECK(ready == 'r');
    CHECK(!kill(child, SIGKILL)); int status; CHECK(waitpid(child, &status, 0) == child && WIFSIGNALED(status));
    close(report[0]);
    OmegaGpuAdmissionStats s; CHECK(!omega_gpu_admission_stats(a, &s));
    CHECK(s.reserved_bytes == 1); /* previous complete transaction survived */
    OmegaGpuReservation rest; CHECK(!reserve(a, 8191, &rest));
    CHECK(!omega_gpu_admission_cancel(a, rest)); CHECK(!omega_gpu_admission_cancel(a, original));
    CHECK(!omega_gpu_admission_close(a));
    puts("PASS SIGKILL midway through actual persistent write, previous transaction preserved");
}
static void bad_memory(void) {
    OmegaGpuAdmission *a; CHECK(!op_open(&a)); OmegaGpuReservation r;
    const char *bad[] = {"MemFree: 100000 kB\n", "MemAvailable: -1 kB\n", "MemAvailable: 18446744073709551615 kB\n",
        "MemAvailable: 32 MB\n", "MemAvailable: 32 kB trailing\n", "MemAvailable: 32 kB\nMemAvailable: 32 kB\n"};
    for (unsigned i = 0; i < sizeof bad / sizeof bad[0]; i++) { fixture(bad[i]); CHECK(reserve(a, 1, &r) < 0 && !r.generation); }
    fixture("MemAvailable: 8 kB\n"); CHECK(reserve(a, 8192, &r) == -ENOSPC);
    fixture("MemAvailable: 12 kB\n"); CHECK(!reserve(a, 8192, &r)); CHECK(!omega_gpu_admission_cancel(a, r));
    struct timespec times[2] = {{1,0},{1,0}}; CHECK(!utimensat(AT_FDCWD, mempath, times, 0)); CHECK(reserve(a, 1, &r) == -ESTALE);
    CHECK(!unlink(mempath)); CHECK(reserve(a, 1, &r) == -ENOENT);
    CHECK(!symlink("/proc/meminfo", mempath)); CHECK(reserve(a, 1, &r) < 0); CHECK(!unlink(mempath));
    fixture("MemAvailable: 32 kB\n"); CHECK(!omega_gpu_admission_close(a));
    puts("PASS missing, stale, malformed, overflow and symlink memory fixtures");
}
static void settings_and_paths(void) {
    OmegaGpuAdmission *a; OmegaGpuAdmissionConfig c = config;
    c.version++; CHECK(omega_gpu_admission_open(&c, &a) == -EINVAL && a == NULL);
    c = config; c.budget_bytes = UINT64_MAX; CHECK(omega_gpu_admission_open(&c, &a) == -EINVAL);
    c = config; c.budget_bytes = 0; CHECK(omega_gpu_admission_open(&c, &a) == -EINVAL);
    c = config; c.keep_free_bytes = UINT64_MAX; CHECK(omega_gpu_admission_open(&c, &a) == -EINVAL);
    c = config; c.fixture_max_age_sec = 0; CHECK(omega_gpu_admission_open(&c, &a) == -EINVAL);
    c = config; c.budget_bytes = 4096; CHECK(omega_gpu_admission_open(&c, &a) == -EIO);
    char link[300]; snprintf(link, sizeof link, "%s-alias", root); CHECK(!symlink(root, link));
    c = config; c.directory = link; CHECK(omega_gpu_admission_open(&c, &a) < 0); CHECK(!unlink(link));
    CHECK(!chmod(root, 0777)); CHECK(op_open(&a) == -EPERM); CHECK(!chmod(root, 0700));
    puts("PASS invalid configuration, incompatible group and unsafe directory refusal");
}
static int refused_free(void *ctx, OmegaGpuBufferHandle b) { (void)ctx; (void)b; return OMEGA_GPU_PLAT_REFUSED_BUSY; }
static void adapter(void) {
    fake_reset();
    /* The old guard remains selectable and unchanged, independently of the new
     * adapter. Disabling it here proves the new path refuses before nvrm. */
    setenv("CUDAPP_MEM_RESERVE_MIB", "0", 1);
    OmegaGpuPlatformCaps caps; const OmegaGpuPlatform *p = omega_gpu_platform_linux(&caps);
    OmegaGpuDeviceHandle dev; CHECK(!p->device_open(p->ctx, 0, &dev));
    OmegaGpuAdmission *a; CHECK(!op_open(&a));
    OmegaGpuBufferDesc d = {.kind=OMEGA_GPU_HANDLE_BUF_DATA, .rights=OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_GPU_WRITE,
        .gpu_cache=OMEGA_GPU_CACHE_GPU_UNCACHED, .bytes=4097};
    OmegaGpuBufferHandle b, denied; OmegaCpuView cpu; OmegaGpuVa gpu;
    CHECK(!omega_gpu_admission_dma_alloc(a, p, &caps, dev, &d, OMEGA_ADMISSION_KV, &b, &cpu, &gpu, NULL));
    int calls = fk.alloc_calls; CHECK(calls == 1);
    CHECK(omega_gpu_admission_dma_alloc(a, p, &caps, dev, &d, OMEGA_ADMISSION_KV, &denied, &cpu, &gpu, NULL) == -ENOSPC);
    CHECK(fk.alloc_calls == calls && !denied.generation && !cpu.ptr && !gpu.va);
    OmegaGpuAdmissionStats s; CHECK(!omega_gpu_admission_stats(a, &s)); CHECK(s.committed_bytes == 8192);
    OmegaGpuBufferHandle wrong = b; wrong.generation++;
    CHECK(omega_gpu_admission_dma_free(a, p, wrong) == -ESTALE && fk.free_calls == 0);
    CHECK(!omega_gpu_admission_dma_free(a, p, b)); CHECK(omega_gpu_admission_dma_free(a, p, b) == -ESTALE);
    CHECK(!omega_gpu_admission_seal(a, 1));
    CHECK(omega_gpu_admission_dma_alloc(a, p, &caps, dev, &d, OMEGA_ADMISSION_KV, &b, &cpu, &gpu, NULL) == -EPERM);
    CHECK(fk.alloc_calls == calls); CHECK(!omega_gpu_admission_seal(a, 0));
    d.bytes = 4096; d.rights |= OMEGA_GPU_RIGHT_HOST_POLLED; d.gpu_cache = OMEGA_GPU_CACHE_GPU_CACHED;
    CHECK(omega_gpu_admission_dma_alloc(a, p, &caps, dev, &d, OMEGA_ADMISSION_MARKER, &b, &cpu, &gpu, NULL) != 0);
    d.gpu_cache = OMEGA_GPU_CACHE_GPU_UNCACHED;
    CHECK(omega_gpu_admission_dma_alloc(a, p, &caps, dev, &d, OMEGA_ADMISSION_KV, &b, &cpu, &gpu, NULL) != 0);
    CHECK(fk.alloc_calls == calls);
    CHECK(!omega_gpu_admission_dma_alloc(a, p, &caps, dev, &d, OMEGA_ADMISSION_MARKER, &b, &cpu, &gpu, NULL));
    CHECK(!omega_gpu_admission_stats(a, &s)); CHECK(s.by_usage[OMEGA_ADMISSION_MARKER] == 4096);
    *(uint32_t *)cpu.ptr = 7; uint32_t observed;
    CHECK(!p->poll_word(p->ctx, b, 0, 7, 1, &observed) && observed == 7);
    CHECK(!omega_gpu_admission_dma_free(a, p, b));
    d.rights = OMEGA_GPU_RIGHT_GPU_READ | OMEGA_GPU_RIGHT_CPU_WRITE; d.gpu_cache = OMEGA_GPU_CACHE_GPU_CACHED;
    OmegaGpuPlatform refusing = *p; refusing.dma_free = refused_free;
    CHECK(!omega_gpu_admission_dma_alloc(a, &refusing, &caps, dev, &d, OMEGA_ADMISSION_WEIGHT, &b, &cpu, &gpu, NULL));
    int frees = fk.free_calls;
    CHECK(omega_gpu_admission_dma_free(a, &refusing, b) == OMEGA_GPU_PLAT_REFUSED_BUSY);
    CHECK(fk.free_calls == frees); CHECK(!omega_gpu_admission_stats(a, &s)); CHECK(s.committed_bytes == 4096);
    CHECK(omega_gpu_admission_dma_free(a, p, b) == -ESTALE); /* wrong delegate identity */
    refusing.dma_free = p->dma_free; CHECK(!omega_gpu_admission_dma_free(a, &refusing, b));
    int allocs = fk.alloc_calls;
    CHECK(!unlink(mempath));
    CHECK(omega_gpu_admission_dma_alloc(a, p, &caps, dev, &d, OMEGA_ADMISSION_WEIGHT, &b, &cpu, &gpu, NULL) == -ENOENT);
    CHECK(fk.alloc_calls == allocs && !b.generation);
    fixture("MemAvailable: 32 kB\n");
    d.rights = OMEGA_GPU_RIGHT_GPU_READ | OMEGA_GPU_RIGHT_CPU_WRITE; d.gpu_cache = OMEGA_GPU_CACHE_GPU_CACHED;
    CHECK(!omega_gpu_admission_dma_alloc(a, p, &caps, dev, &d, OMEGA_ADMISSION_WEIGHT, &b, &cpu, &gpu, NULL));
    CHECK(!omega_gpu_admission_dma_free(a, p, b));
    /* Driver refusal must cancel the acquired charge, with no published handle. */
    fk.alloc_fail_nth = fk.alloc_calls + 1; fk.alloc_mode = FK_ALLOC_FAIL;
    CHECK(omega_gpu_admission_dma_alloc(a, p, &caps, dev, &d, OMEGA_ADMISSION_WEIGHT, &b, &cpu, &gpu, NULL) != 0);
    CHECK(!b.generation && !cpu.ptr); CHECK(!omega_gpu_admission_stats(a, &s)); CHECK(!s.committed_bytes && !s.acquired_bytes);
    CHECK(!omega_gpu_admission_close(a)); CHECK(!p->device_close(p->ctx, dev)); fake_release_all();
    puts("PASS actual Linux allocation boundary on simulated nvrm, cache policy, seal and rollback");
}
static int legacy_gap(void) {
    fixture("MemAvailable: 12 kB\n"); setenv("CUDAPP_MEMINFO", mempath, 1); setenv("CUDAPP_MEM_RESERVE_MIB", "0", 1);
    /* Use a nonzero 1MiB legacy reserve and 3MiB fixture for 2MiB requests. */
    fixture("MemAvailable: 3072 kB\n"); setenv("CUDAPP_MEM_RESERVE_MIB", "1", 1);
    int start[2], reports[2], hold[2]; CHECK(!pipe(start)); CHECK(!pipe(reports)); CHECK(!pipe(hold)); pid_t pids[2];
    for (int i=0; i<2; i++) {
        pids[i]=fork(); CHECK(pids[i]>=0);
        if (!pids[i]) {
            close(start[1]); close(reports[0]); close(hold[1]);
            char ch; io(start[0], &ch, 1, 0); fake_reset(); fk.alloc_max=2u<<20;
            const OmegaGpuPlatform *p=omega_gpu_platform_linux(NULL); OmegaGpuDeviceHandle dev;
            CHECK(!p->device_open(p->ctx,0,&dev));
            OmegaGpuBufferDesc d={.kind=OMEGA_GPU_HANDLE_BUF_DATA,.rights=OMEGA_GPU_RIGHT_CPU_READ|OMEGA_GPU_RIGHT_GPU_WRITE,.bytes=2u<<20};
            OmegaGpuBufferHandle b; OmegaCpuView cpu; OmegaGpuVa gpu;
            int rc=p->dma_alloc(p->ctx,dev,&d,&b,&cpu,&gpu,NULL);
            io(reports[1],&rc,sizeof rc,1); io(hold[0],&ch,1,0); _exit(0);
        }
    }
    close(start[0]); close(reports[1]); close(hold[0]);
    char go[2]={'g','g'}; io(start[1],go,2,1); int rc[2]; io(reports[0],&rc[0],sizeof(int),0); io(reports[0],&rc[1],sizeof(int),0);
    io(hold[1],go,2,1); wait_ok(pids[0]); wait_ok(pids[1]);
    int winners=(rc[0]==0)+(rc[1]==0);
    printf("legacy synchronized contenders admitted=%d expected=1\n",winners);
    if (winners != 1) { fputs("FAIL actual legacy guard permits double spending\n",stderr); return 1; }
    return 0;
}
static void hostile_state(void) {
    char state[300], lock[300]; snprintf(state,sizeof state,"%s/state",root); snprintf(lock,sizeof lock,"%s/lock",root);
    OmegaGpuAdmission *a; CHECK(!op_open(&a));
    CHECK(!unlink(lock)); CHECK(!symlink(state,lock));
    OmegaGpuReservation r; CHECK(reserve(a,1,&r)==-EIO); /* replaced lock refuses existing context */
    /* Leave the context quarantined, process exit closes fds. */
    OmegaGpuAdmission *b; CHECK(op_open(&b)<0);
    CHECK(!unlink(lock));
    CHECK(!link(state,lock)); CHECK(op_open(&b)==-EPERM); CHECK(!unlink(lock));
    int fd=open(state,O_WRONLY); CHECK(fd>=0); unsigned char bad=0; CHECK(pwrite(fd,&bad,1,0)==1); close(fd);
    CHECK(op_open(&b)==-EIO);
    CHECK(!unlink(state)); CHECK(!unlink(lock));
    puts("PASS lock replacement, symlink, hardlink and corrupt-state fail-closed controls");
}
int main(int argc, char **argv) {
    alarm(40);
    char temp[]="/tmp/cudapp-admission24-XXXXXX"; CHECK(mkdtemp(temp)!=NULL); strcpy(root,temp);
    snprintf(mempath,sizeof mempath,"%s/meminfo",root);
    config=(OmegaGpuAdmissionConfig){OMEGA_ADMISSION_VERSION,root,8192,4096,mempath,60};
    fixture("MemAvailable: 32 kB\n");
    if (argc==2 && !strcmp(argv[1],"--legacy-gap")) return legacy_gap();
    if (argc==2 && !strcmp(argv[1],"--contenders")) { contenders(); return 0; }
    if (argc==2 && !strcmp(argv[1],"--ownership")) { death_and_ownership(); reused_identity(); return 0; }
    if (argc==2 && !strcmp(argv[1],"--adapter")) { adapter(); return 0; }
    transitions(); contenders(); death_and_ownership(); reused_identity(); interrupted_transaction(); bad_memory(); settings_and_paths(); adapter(); hostile_state();
    CHECK(!unlink(mempath)); CHECK(!rmdir(root));
    printf("PASS issue24 host checks=%u (fixture inputs; no GPU or performance claim)\n",checks); return 0;
}
