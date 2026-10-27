Run 1 of tools/c5/chip.sh at 64f0834, GB10, driver 580.173.02, 2026-10-27 02:52:14Z-02:55:10Z, one hold of /tmp/aien-gb10.lock.
Result: 8 of 9 checks as expected; 06-kill FAILED.

06-kill: SIGKILL of the process group 500 ms into the never-ending launch; the matrix polled 2 s for the group to vanish and
it had not (`[matrix] process group still alive`). The pid was gone afterwards; the CUDA tenant ran 60 s with 0 wrong values,
the fresh process after the kill passed (both 0 wrong), and 08-health passed. Kernel log: Xid 109 CTX SWITCH TIMEOUT for the
killed pid at 21:54:02 local. In 03-hang of the same hold, freeing a hung channel took 4634.1 ms and 2261.9 ms
(`recover: the device closed ... in`), so the 2 s bound was shorter than the driver's own teardown. Fixed after this run:
the matrix waits up to 20 s and prints the time the group took to go.

dmesg.txt / xid.txt here are INCOMPLETE: chip.sh took a line-count baseline, and the kernel ring buffer is full and rotates,
so lines were lost and one repeated. dmesg-window-recaptured.txt is the same hold window re-read by timestamp after the run
(3 lines: NV_ERR_TIMEOUT on NVA06F_CTRL_CMD_STOP_CHANNEL at 21:52:23, Xid 109 at 21:52:31 during 03-hang, Xid 109 at
21:54:02 during 06-kill). Fixed after this run: chip.sh filters by timestamp.
