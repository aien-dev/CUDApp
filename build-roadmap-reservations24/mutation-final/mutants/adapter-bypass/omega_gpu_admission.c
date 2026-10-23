/* SPDX-License-Identifier: MIT */
#include "omega_gpu_admission.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { RESERVED = 1, ACQUIRED, COMMITTED };
typedef struct {
    uint64_t generation, bytes, start;
    int64_t pid;
    uint32_t state, usage;
    char boot[40];
} Record;
typedef struct {
    uint64_t magic, serial, budget, keep, checksum;
    Record records[OMEGA_ADMISSION_SLOTS];
} Database;
typedef struct {
    OmegaGpuReservation token;
    OmegaGpuBufferHandle buffer;
    const OmegaGpuPlatform *platform;
    void *platform_ctx;
} Binding;
struct OmegaGpuAdmission {
    pthread_mutex_t mu;
    int dir, lock, state, sealed;
    pid_t pid;
    uint64_t start, budget, keep;
    char boot[40], fixture[PATH_MAX];
    uint32_t age;
    Database *db;
    Binding bindings[OMEGA_ADMISSION_SLOTS];
};
#define MAGIC 0x43323441444d0001ull
static int neg(void) { return errno ? -errno : -EIO; }
/* No symlink in any directory component. Ancestors may be shared; final
 * directory is private to this uid. Replacement by that trusted uid is outside
 * the cooperation contract. Never unlink lock/state to reset a live group. */
static int directory(const char *path) {
    if (!path || path[0] != '/' || strlen(path) >= PATH_MAX) return -EINVAL;
    char copy[PATH_MAX]; strcpy(copy, path);
    int fd = open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return neg();
    char *save, *part = strtok_r(copy, "/", &save);
    while (part) {
        if (!strcmp(part, ".") || !strcmp(part, "..")) { close(fd); return -EINVAL; }
        int next = openat(fd, part, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        close(fd);
        if (next < 0) return neg();
        fd = next; part = strtok_r(NULL, "/", &save);
    }
    struct stat s;
    if (fstat(fd, &s) || s.st_uid != geteuid() || (s.st_mode & 0777) != 0700) { close(fd); return -EPERM; }
    return fd;
}
static int secure_file(int dir, const char *name) {
    int fd = openat(dir, name, O_RDWR | O_CREAT | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, 0600);
    if (fd < 0) return neg();
    struct stat s;
    if (fstat(fd, &s) || !S_ISREG(s.st_mode) || s.st_uid != geteuid() || s.st_nlink != 1 ||
        (s.st_mode & 0777) != 0600) { close(fd); return -EPERM; }
    return fd;
}
static int identity(pid_t pid, uint64_t *start, int *dead) {
    char path[64], line[4096];
    snprintf(path, sizeof path, "/proc/%ld/stat", (long)pid);
    FILE *f = fopen(path, "re");
    if (!f) { if (errno == ENOENT) { *dead = 1; return 0; } return neg(); }
    if (!fgets(line, sizeof line, f)) { fclose(f); return -EIO; }
    fclose(f);
    char *p = strrchr(line, ')');
    if (!p || p[1] != ' ' || !p[2]) return -EIO;
    *dead = p[2] == 'Z' || p[2] == 'X';
    char *save, *word = strtok_r(p + 2, " ", &save);
    for (int field = 3; field < 22 && word; field++) word = strtok_r(NULL, " ", &save);
    if (!word) return -EIO;
    errno = 0; char *end; unsigned long long v = strtoull(word, &end, 10);
    if (errno || end == word || (*end && *end != '\n')) return -EIO;
    *start = v; return 0;
}
static uint64_t digest(Database *db) {
    uint64_t saved = db->checksum, h = 14695981039346656037ull;
    db->checksum = 0;
    const unsigned char *p = (const unsigned char *)db;
    for (size_t i = 0; i < sizeof *db; i++) { h ^= p[i]; h *= 1099511628211ull; }
    db->checksum = saved;
    return h;
}
static int save_db(OmegaGpuAdmission *a) {
    a->db->checksum = digest(a->db);
    /* A stable separate lock protects atomic state replacement. A process killed
     * during a write leaves the old complete state and a disposable next file. */
    int fd = secure_file(a->dir, "next");
    if (fd < 0) return fd;
    if (ftruncate(fd, 0) || pwrite(fd, a->db, sizeof *a->db, 0) != (ssize_t)sizeof *a->db || fdatasync(fd)) {
        close(fd); return -EIO;
    }
    if (renameat(a->dir, "next", a->dir, "state")) { close(fd); return -EIO; }
    close(a->state); a->state = fd;
    if (fsync(a->dir)) return -EIO;
    return 0;
}
static int enter(OmegaGpuAdmission *a) {
    if (!a || a->pid != getpid()) return -EPERM;
    pthread_mutex_lock(&a->mu);
    /* Different opens have independent flock descriptions; mu serializes threads
     * sharing one description. Nonblocking keeps refusal bounded. */
    if (flock(a->lock, LOCK_EX | LOCK_NB)) { pthread_mutex_unlock(&a->mu); return -EBUSY; }
    struct stat s, linked;
    if (fstat(a->lock, &s) || fstatat(a->dir, "lock", &linked, AT_SYMLINK_NOFOLLOW) ||
        linked.st_ino != s.st_ino || linked.st_dev != s.st_dev || s.st_nlink != 1) goto bad;
    int fd = openat(a->dir, "state", O_RDWR | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) goto bad;
    close(a->state); a->state = fd;
    if (fstat(fd, &s) || !S_ISREG(s.st_mode) || s.st_uid != geteuid() || s.st_nlink != 1 ||
        (s.st_mode & 0777) != 0600 || s.st_size != (off_t)sizeof *a->db) goto bad;
    if (pread(a->state, a->db, sizeof *a->db, 0) != (ssize_t)sizeof *a->db ||
        a->db->magic != MAGIC || a->db->budget != a->budget || a->db->keep != a->keep ||
        a->db->checksum != digest(a->db)) goto bad;
    for (unsigned i = 0; i < OMEGA_ADMISSION_SLOTS; i++) {
        Record *r = &a->db->records[i];
        if (!r->state) continue;
        if (r->state > COMMITTED || r->usage >= OMEGA_ADMISSION_SHARED_WEIGHT ||
            !r->bytes || r->bytes > a->budget || !r->generation || r->generation > a->db->serial ||
            r->pid <= 0 || r->pid > INT_MAX || !memchr(r->boot, 0, sizeof r->boot)) goto bad;
        int dead = 0; uint64_t start = 0;
        if (strcmp(r->boot, a->boot)) dead = 1;
        else {
            if (identity((pid_t)r->pid, &start, &dead)) goto bad;
            if (start != r->start) dead = 1;
        }
        if (dead) memset(r, 0, sizeof *r);
    }
    return 0;
bad:
    flock(a->lock, LOCK_UN); pthread_mutex_unlock(&a->mu); return -EIO;
}
static int leave(OmegaGpuAdmission *a, int rc, int write) {
    if (write) { int e = save_db(a); if (e) rc = e; }
    flock(a->lock, LOCK_UN); pthread_mutex_unlock(&a->mu); return rc;
}
static int available(OmegaGpuAdmission *a, uint64_t *out) {
    int fd = open(a->fixture[0] ? a->fixture : "/proc/meminfo", O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) return neg();
    struct stat s;
    if (fstat(fd, &s) || !S_ISREG(s.st_mode) || s.st_size > 65536) { close(fd); return -EIO; }
    if (a->fixture[0]) {
        time_t now = time(NULL);
        if (now < s.st_mtime || now - s.st_mtime > a->age) { close(fd); return -ESTALE; }
    }
    FILE *f = fdopen(fd, "r");
    if (!f) { close(fd); return -EIO; }
    char line[256]; int have = 0, rc = 0; unsigned lines = 0;
    while (fgets(line, sizeof line, f)) {
        if (++lines > 1024 || (!strchr(line, '\n') && !feof(f))) { rc = -EIO; break; }
        if (strncmp(line, "MemAvailable:", 13)) continue;
        if (have) { rc = -EINVAL; break; }
        char *p = line + 13;
        while (*p == ' ' || *p == '\t') p++;
        if (*p < '0' || *p > '9') { rc = -EINVAL; break; }
        errno = 0; char *end; unsigned long long kb = strtoull(p, &end, 10);
        while (*end == ' ' || *end == '\t') end++;
        if (errno || strcmp(end, "kB\n") || kb > UINT64_MAX / 1024) { rc = -EINVAL; break; }
        *out = kb * 1024; have = 1;
    }
    if (ferror(f)) rc = -EIO;
    fclose(f); return rc ? rc : have ? 0 : -ENODATA;
}
int omega_gpu_admission_open(const OmegaGpuAdmissionConfig *c, OmegaGpuAdmission **out) {
    if (!out) return -EINVAL;
    *out = NULL;
    if (!c || c->version != OMEGA_ADMISSION_VERSION || !c->budget_bytes ||
        c->budget_bytes > OMEGA_ADMISSION_MAX_BYTES || c->keep_free_bytes > OMEGA_ADMISSION_MAX_BYTES ||
        (c->fixture_meminfo && (!*c->fixture_meminfo || strlen(c->fixture_meminfo) >= PATH_MAX ||
        !c->fixture_max_age_sec || c->fixture_max_age_sec > 3600)) ||
        (!c->fixture_meminfo && c->fixture_max_age_sec)) return -EINVAL;
    OmegaGpuAdmission *a = calloc(1, sizeof *a);
    if (!a) return -ENOMEM;
    a->dir = a->lock = a->state = -1;
    a->db = calloc(1, sizeof *a->db);
    int rc = -ENOMEM;
    if (!a->db) goto fail;
    a->pid = getpid(); int dead = 0;
    if ((rc = identity(a->pid, &a->start, &dead))) goto fail;
    FILE *f = fopen("/proc/sys/kernel/random/boot_id", "re");
    if (!f) { rc = neg(); goto fail; }
    if (!fgets(a->boot, sizeof a->boot, f)) { fclose(f); rc = -EIO; goto fail; }
    fclose(f);
    if (strlen(a->boot) != 37 || a->boot[36] != '\n') { rc = -EIO; goto fail; }
    a->budget = c->budget_bytes; a->keep = c->keep_free_bytes;
    if (c->fixture_meminfo) strcpy(a->fixture, c->fixture_meminfo);
    a->age = c->fixture_max_age_sec;
    if ((a->dir = directory(c->directory)) < 0) { rc = a->dir; goto fail; }
    if ((a->lock = secure_file(a->dir, "lock")) < 0) { rc = a->lock; goto fail; }
    if (flock(a->lock, LOCK_EX | LOCK_NB)) { rc = -EBUSY; goto fail; }
    if ((a->state = secure_file(a->dir, "state")) < 0) { rc = a->state; goto fail; }
    struct stat s;
    if (fstat(a->state, &s)) { rc = -EIO; goto fail; }
    if (!s.st_size) {
        a->db->magic = MAGIC; a->db->budget = a->budget; a->db->keep = a->keep;
        if ((rc = save_db(a))) goto fail;
    } else if (s.st_size != (off_t)sizeof *a->db) { rc = -EIO; goto fail; }
    flock(a->lock, LOCK_UN);
    if (pthread_mutex_init(&a->mu, NULL)) { rc = -EIO; goto fail; }
    rc = enter(a);
    if (rc) { pthread_mutex_destroy(&a->mu); goto fail; }
    rc = leave(a, 0, 1);
    if (rc) { pthread_mutex_destroy(&a->mu); goto fail; }
    *out = a; return 0;
fail:
    if (a->state >= 0) close(a->state);
    if (a->lock >= 0) close(a->lock);
    if (a->dir >= 0) close(a->dir);
    free(a->db); free(a); return rc;
}
static int owned(OmegaGpuAdmission *a, Record *r) {
    return r->pid == a->pid && r->start == a->start && !strcmp(r->boot, a->boot);
}
int omega_gpu_admission_close(OmegaGpuAdmission *a) {
    int rc = enter(a); if (rc) return rc;
    for (unsigned i = 0; i < OMEGA_ADMISSION_SLOTS; i++)
        if (a->db->records[i].state && owned(a, &a->db->records[i])) return leave(a, -EBUSY, 0);
    rc = leave(a, 0, 1);
    if (rc) return rc;
    close(a->state); close(a->lock); close(a->dir);
    pthread_mutex_destroy(&a->mu); free(a->db); free(a); return 0;
}
/* Caller holds mu+flock. Requests are charged conservatively against fresh
 * MemAvailable minus ALL active charges, including already-resident buffers.
 * This can double subtract resident bytes for safety, never for a memory report. */
static int reserve_locked(OmegaGpuAdmission *a, uint64_t bytes, OmegaGpuAdmissionUsage usage, OmegaGpuReservation *out) {
    if (a->sealed) return -EPERM;
    if (!out || !bytes || bytes > a->budget || usage < 0 || usage >= OMEGA_ADMISSION_SHARED_WEIGHT) return -EINVAL;
    uint64_t total = 0, avail = 0; unsigned slot = OMEGA_ADMISSION_SLOTS;
    for (unsigned i = 0; i < OMEGA_ADMISSION_SLOTS; i++) {
        Record *r = &a->db->records[i];
        if (!r->state) { if (slot == OMEGA_ADMISSION_SLOTS) slot = i; continue; }
        if (r->bytes > a->budget - total) return -EIO;
        total += r->bytes;
    }
    int rc = available(a, &avail); if (rc) return rc;
    if (slot == OMEGA_ADMISSION_SLOTS || a->db->serial == UINT64_MAX ||
        bytes > a->budget - total || total > avail || bytes > avail - total ||
        a->keep > avail - total - bytes) return -ENOSPC;
    Record *r = &a->db->records[slot];
    memset(r, 0, sizeof *r);
    r->generation = ++a->db->serial; r->bytes = bytes; r->start = a->start;
    r->pid = a->pid; r->state = RESERVED; r->usage = usage; strcpy(r->boot, a->boot);
    *out = (OmegaGpuReservation){r->generation, slot}; return 0;
}
int omega_gpu_admission_reserve(OmegaGpuAdmission *a, uint64_t bytes, OmegaGpuAdmissionUsage usage, OmegaGpuReservation *out) {
    if (out) memset(out, 0, sizeof *out);
    int rc = enter(a); if (rc) return rc;
    rc = reserve_locked(a, bytes, usage, out);
    rc = leave(a, rc, rc == 0);
    if (rc && out) memset(out, 0, sizeof *out);
    return rc;
}
static int transition(OmegaGpuAdmission *a, OmegaGpuReservation t, unsigned from, unsigned to) {
    int rc = enter(a); if (rc) return rc;
    if (t.slot >= OMEGA_ADMISSION_SLOTS || !t.generation) return leave(a, -ESTALE, 0);
    Record *r = &a->db->records[t.slot];
    if (!r->state || r->generation != t.generation || !owned(a, r)) return leave(a, -ESTALE, 0);
    if (r->state != from || (to == ACQUIRED && a->sealed)) return leave(a, -EPERM, 0);
    if (a->bindings[t.slot].token.generation) return leave(a, -EBUSY, 0); /* adapter-owned */
    r->state = to;
    if (!to) memset(r, 0, sizeof *r);
    return leave(a, 0, 1);
}
int omega_gpu_admission_acquire(OmegaGpuAdmission *a, OmegaGpuReservation t) { return transition(a, t, RESERVED, ACQUIRED); }
int omega_gpu_admission_commit(OmegaGpuAdmission *a, OmegaGpuReservation t) { return transition(a, t, ACQUIRED, COMMITTED); }
int omega_gpu_admission_cancel(OmegaGpuAdmission *a, OmegaGpuReservation t) { return transition(a, t, RESERVED, 0); }
int omega_gpu_admission_abort(OmegaGpuAdmission *a, OmegaGpuReservation t) { return transition(a, t, ACQUIRED, 0); }
int omega_gpu_admission_release(OmegaGpuAdmission *a, OmegaGpuReservation t) { return transition(a, t, COMMITTED, 0); }
int omega_gpu_admission_seal(OmegaGpuAdmission *a, int on) {
    int rc = enter(a); if (rc) return rc;
    if (on != 0 && on != 1) return leave(a, -EINVAL, 0);
    a->sealed = on; return leave(a, 0, 0);
}
int omega_gpu_admission_stats(OmegaGpuAdmission *a, OmegaGpuAdmissionStats *out) {
    if (!out) return -EINVAL;
    memset(out, 0, sizeof *out);
    int rc = enter(a); if (rc) return rc;
    rc = available(a, &out->mem_available_bytes);
    if (rc) return leave(a, rc, 0);
    for (unsigned i = 0; i < OMEGA_ADMISSION_SLOTS; i++) {
        Record *r = &a->db->records[i]; if (!r->state) continue;
        uint64_t *sum = r->state == RESERVED ? &out->reserved_bytes : r->state == ACQUIRED ? &out->acquired_bytes : &out->committed_bytes;
        if (r->bytes > UINT64_MAX - *sum || r->bytes > UINT64_MAX - out->by_usage[r->usage]) return leave(a, -EOVERFLOW, 0);
        *sum += r->bytes; out->by_usage[r->usage] += r->bytes;
    }
    FILE *f = fopen("/proc/self/statm", "re");
    unsigned long long pages, rss; long page = sysconf(_SC_PAGESIZE);
    if (!f) return leave(a, -EIO, 0);
    int ok = fscanf(f, "%llu %llu", &pages, &rss); fclose(f);
    if (ok != 2 || page <= 0 || rss > UINT64_MAX / (uint64_t)page) return leave(a, -EIO, 0);
    out->process_rss_bytes = rss * (uint64_t)page;
    return leave(a, 0, 1);
}
int omega_gpu_admission_dma_alloc(OmegaGpuAdmission *a, const OmegaGpuPlatform *p, const OmegaGpuPlatformCaps *caps,
    OmegaGpuDeviceHandle dev, const OmegaGpuBufferDesc *d, OmegaGpuAdmissionUsage usage,
    OmegaGpuBufferHandle *out, OmegaCpuView *cpu, OmegaGpuVa *gpu, OmegaGpuBusAddr *bus) {
    if (out) memset(out, 0, sizeof *out);
    if (cpu) memset(cpu, 0, sizeof *cpu);
    if (gpu) memset(gpu, 0, sizeof *gpu);
    if (bus) memset(bus, 0, sizeof *bus);
    if (!p || !p->dma_alloc || !p->dma_free || !caps || !d || !out || !cpu || !gpu) return -EINVAL;
    int rc = omega_gpu_platform_check_buffer(d, caps);
    if (rc) return rc;
    if ((usage == OMEGA_ADMISSION_MARKER && (!(d->rights & OMEGA_GPU_RIGHT_HOST_POLLED) || d->gpu_cache != OMEGA_GPU_CACHE_GPU_UNCACHED)) ||
        ((d->rights & OMEGA_GPU_RIGHT_HOST_POLLED) && usage != OMEGA_ADMISSION_MARKER) ||
        (d->gpu_cache == OMEGA_GPU_CACHE_GPU_CACHED && (d->kind != OMEGA_GPU_HANDLE_BUF_DATA ||
        (d->rights & (OMEGA_GPU_RIGHT_CPU_READ | OMEGA_GPU_RIGHT_HOST_POLLED))))) return -EINVAL;
    /* Native nvrm alloc rounds to 4KiB. Charge actual rounded capacity. */
    if (d->bytes > UINT64_MAX - 4095) return -EOVERFLOW;
    uint64_t bytes = (d->bytes + 4095) & ~4095ull;
    rc = enter(a); if (rc) return rc;
    OmegaGpuReservation t;
    rc = reserve_locked(a, bytes, usage, &t);
    if (rc) { (void)p->dma_alloc(p->ctx, dev, d, out, cpu, gpu, bus); return leave(a, rc, 0); }
    a->db->records[t.slot].state = ACQUIRED;
    /* Persist before delegate: a crash must retain or detect this charge. */
    rc = save_db(a);
    if (rc) return leave(a, rc, 0);
    rc = p->dma_alloc(p->ctx, dev, d, out, cpu, gpu, bus);
    if (rc) {
        memset(&a->db->records[t.slot], 0, sizeof(Record));
        memset(out, 0, sizeof *out); memset(cpu, 0, sizeof *cpu); memset(gpu, 0, sizeof *gpu);
        if (bus) memset(bus, 0, sizeof *bus);
        return leave(a, rc, 1);
    }
    a->bindings[t.slot] = (Binding){t, *out, p, p->ctx};
    a->db->records[t.slot].state = COMMITTED;
    /* On storage failure return error but preserve the live binding/charge and
     * handle, allowing caller to free; never silently uncharge a live buffer. */
    return leave(a, 0, 1);
}
int omega_gpu_admission_dma_free(OmegaGpuAdmission *a, const OmegaGpuPlatform *p, OmegaGpuBufferHandle b) {
    if (!p || !p->dma_free) return -EINVAL;
    int rc = enter(a); if (rc) return rc;
    for (unsigned i = 0; i < OMEGA_ADMISSION_SLOTS; i++) {
        Binding *v = &a->bindings[i];
        if (!v->token.generation || v->platform != p || v->platform_ctx != p->ctx ||
            v->buffer.index != b.index || v->buffer.generation != b.generation) continue;
        Record *r = &a->db->records[i];
        if (r->generation != v->token.generation || r->state != COMMITTED || !owned(a, r)) return leave(a, -ESTALE, 0);
        rc = p->dma_free(p->ctx, b);
        if (rc) return leave(a, rc, 0);
        memset(r, 0, sizeof *r); memset(v, 0, sizeof *v);
        return leave(a, 0, 1);
    }
    return leave(a, -ESTALE, 0);
}
