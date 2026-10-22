/* SPDX-License-Identifier: MIT */
#include "omega_blackwell_codegen.h"
#include "sha256.h"
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int checks, failures;
#define CHECK(c, name) do { checks++; if (!(c)) { failures++; printf("FAIL %s\n", name); } } while (0)

int main(void) {
    static const unsigned char code[] = { 1, 2, 3, 4 };
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(code, sizeof code, digest);
    char leaf[21];
    for (size_t i = 0; i < 8; i++) sprintf(leaf + 2 * i, "%02x", digest[i]);
    memcpy(leaf + 16, ".bin", 5);
    char dir[] = "/tmp/cudapp-dump-XXXXXX";
    if (!mkdtemp(dir)) return 2;
    char path[256];
    snprintf(path, sizeof path, "%s/%s", dir, leaf);
    mode_t old_umask = umask(0);
    setenv("OMEGA_KERNEL_DUMP_DIR", dir, 1);
    omega_kernel_dump(code, sizeof code);
    struct stat st;
    CHECK(stat(path, &st) == 0, "capture exists");
    CHECK(stat(path, &st) == 0 && (st.st_mode & 0777) == 0600, "capture mode is 0600 with umask 0");
    FILE *f = fopen(path, "rb");
    unsigned char got[8] = {0};
    size_t n = f ? fread(got, 1, sizeof got, f) : 0;
    if (f) fclose(f);
    CHECK(n == sizeof code && memcmp(got, code, sizeof code) == 0, "capture bytes exact");
    int fd = open(path, O_WRONLY | O_TRUNC | O_CLOEXEC);
    f = fd >= 0 ? fdopen(fd, "wb") : NULL;
    if (!f) return 2;
    fputs("keep", f); fclose(f);
    omega_kernel_dump(code, sizeof code);
    f = fopen(path, "rb");
    n = f ? fread(got, 1, sizeof got, f) : 0;
    if (f) fclose(f);
    CHECK(n == 4 && memcmp(got, "keep", 4) == 0, "existing file preserved");
    unlink(path);
    char target[256]; snprintf(target, sizeof target, "%s/target", dir);
    fd = open(target, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    f = fd >= 0 ? fdopen(fd, "wb") : NULL; if (!f) return 2;
    fputs("safe", f); fclose(f);
    CHECK(symlink(target, path) == 0, "symlink fixture created");
    omega_kernel_dump(code, sizeof code);
    f = fopen(target, "rb"); n = f ? fread(got, 1, sizeof got, f) : 0;
    if (f) fclose(f);
    CHECK(n == 4 && memcmp(got, "safe", 4) == 0, "symlink target preserved");
    unlink(path); unlink(target);
    CHECK(chmod(dir, 0777) == 0, "unsafe directory fixture");
    omega_kernel_dump(code, sizeof code);
    CHECK(access(path, F_OK) != 0, "shared directory refused");
    unlink(path); chmod(dir, 0700);
    unsetenv("OMEGA_KERNEL_DUMP_DIR");
    omega_kernel_dump(code, sizeof code);
    CHECK(access(path, F_OK) != 0, "unset capture disabled");
    setenv("OMEGA_KERNEL_DUMP_DIR", dir, 1);
    omega_kernel_dump(NULL, sizeof code); omega_kernel_dump(code, 0);
    CHECK(access(path, F_OK) != 0, "empty inputs ignored");
    char long_dir[5000]; memset(long_dir, 'x', sizeof long_dir - 1); long_dir[sizeof long_dir - 1] = 0;
    setenv("OMEGA_KERNEL_DUMP_DIR", long_dir, 1);
    omega_kernel_dump(code, sizeof code);
    CHECK(access(path, F_OK) != 0, "overlong path refused");
    char outside[] = "/dev/shm/cudapp-dump-XXXXXX";
    if (!mkdtemp(outside)) return 2;
    char escaped[256]; snprintf(escaped, sizeof escaped, "%s/%s", outside, leaf);
    setenv("OMEGA_KERNEL_DUMP_DIR", outside, 1);
    omega_kernel_dump(code, sizeof code);
    CHECK(access(escaped, F_OK) != 0, "escape outside capture roots refused");
    unlink(escaped); rmdir(outside);
    char local[] = "build-l478-local-XXXXXX";
    if (!mkdtemp(local)) return 2;
    setenv("OMEGA_KERNEL_DUMP_DIR", local, 1);
    omega_kernel_dump(code, sizeof code);
    char local_path[256]; snprintf(local_path, sizeof local_path, "%s/%s", local, leaf);
    CHECK(stat(local_path, &st) == 0 && (st.st_mode & 0777) == 0600, "private checkout directory supported");
    unlink(local_path); rmdir(local);
    unsetenv("OMEGA_KERNEL_DUMP_DIR");
    umask(old_umask); rmdir(dir);
    printf("kernel dump: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
