/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* test_out_file.h: create a test output file (receipt JSON, kernel dump, saved probe data, fake fixture) with mode 0600.
 *
 * fopen(path, "w") asks for 0666 and leaves the rest to the umask, so under umask 0 every receipt is world-writable
 * and anyone on the host can rewrite a result before it is filed (CodeQL cpp/world-writable-file-creation). The user
 * who runs a test is the user who reads its output: the chip scripts (tools/c5/chip.sh and friends) write and file the
 * receipts as themselves, and no runner reads them as another user. Otherwise the same as fopen "w": create or
 * truncate, the path taken as given. */
#ifndef TEST_OUT_FILE_H
#define TEST_OUT_FILE_H
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

static inline FILE *test_out_open(const char *path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) return NULL;
    FILE *f = fdopen(fd, "w");
    if (!f) close(fd);
    return f;
}
#endif
