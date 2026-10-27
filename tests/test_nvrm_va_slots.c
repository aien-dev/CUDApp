/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* test_nvrm_va_slots: the driver client's process-wide GPU VA slots and its close path, on the host. Opens no
 * device: nvrm_open is never called; the Nvrm structs are built by hand and only va_take, acquire_va_slot and
 * nvrm_close run (nvrm_close with no RM objects does no ioctl).
 *
 * VA_SLOT_NO_OVERLAP (CudaAuditFable REPORT.md RtMem F5): every Nvrm gets a 4 GiB slot from a shared counter, and
 * the bump allocator used to run past the slot end into the next client's slot. A client may now grow past 4 GiB
 * only by claiming the address space after its slot from the counter, which works when nothing was handed out
 * after it; otherwise the allocation is refused.
 * CLOSE_TWICE_LEAVES_REUSED_FDS (RtMem F12): nvrm_close closed the four fds and kept their numbers, so a second
 * close closed whatever file had since reused the number.
 */
#include "nvrm.c"

#include <stdlib.h>

static int n_pass, n_fail;
static void check(const char *id, int ok) {
    printf("[%s] %s\n", ok ? "PASS" : "FAIL", id);
    if (ok) n_pass++; else n_fail++;
}

static void slot_rm(Nvrm *rm, uint64_t slot) {
    memset(rm, 0, sizeof *rm);
    rm->va_next = rm->va_slot_base = slot;
}

static void case_va_slots(void) {
    const uint64_t four = 0x100000000ull;
    const uint64_t a = acquire_va_slot(), b = acquire_va_slot();
    check("va_two_slots_adjacent", a != 0 && b == a + four);
    Nvrm ra, rb;
    slot_rm(&ra, a);
    slot_rm(&rb, b);
    check("va_whole_slot_fits", va_take(&ra, four, 0x1000) == a);
    const uint64_t past = va_take(&ra, 0x1000, 0x1000);
    printf("  slot A full, next take: 0x%llx (slot B starts at 0x%llx)\n", (unsigned long long)past,
           (unsigned long long)b);
    check("va_slot_full_refused_not_into_next_slot", past == 0);
    const uint64_t big = va_take(&rb, 2 * four, 0x1000);
    check("va_last_slot_grows_past_4gib", big == b);
    const uint64_t c = acquire_va_slot();
    printf("  after B took 8 GiB, next slot: 0x%llx (B ends at 0x%llx)\n", (unsigned long long)c,
           (unsigned long long)(b + 2 * four));
    check("va_next_slot_after_grown_slot", c >= b + 2 * four);
    check("va_grown_slot_cannot_grow_into_later_slot", va_take(&rb, 0x1000, 0x1000) == 0);
    Nvrm rc;
    slot_rm(&rc, c);
    check("va_later_slot_still_usable", va_take(&rc, 0x1000, 0x1000) == c);
}

static int fd_open(int fd) { return fcntl(fd, F_GETFD) != -1; }

static void case_close_twice(void) {
    int p[2];
    if (pipe(p) != 0) { check("close_pipe", 0); return; }
    Nvrm rm;
    memset(&rm, 0, sizeof rm);
    rm.fd_ctl = dup(p[0]);
    rm.fd_uvm = dup(p[0]);
    rm.fd_uvm2 = dup(p[0]);
    rm.fd_dev = dup(p[0]);
    const int ctl = rm.fd_ctl;
    check("close_first", nvrm_close(&rm) == 0 && !fd_open(ctl));
    check("close_fds_cleared", rm.fd_ctl <= 0 && rm.fd_uvm <= 0 && rm.fd_uvm2 <= 0 && rm.fd_dev <= 0);
    const int victim = dup(p[1]); /* lowest free number: one the first close released */
    printf("  reused fd %d (fd_ctl was %d)\n", victim, ctl);
    (void)nvrm_close(&rm);
    check("close_twice_leaves_reused_fd_open", victim >= 0 && fd_open(victim));
    close(victim);
    close(p[0]);
    close(p[1]);
}

int main(void) {
    case_va_slots();
    case_close_twice();
    printf("SUMMARY passed=%d failed=%d\n", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
