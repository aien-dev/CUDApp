# Experimental cooperative memory admission v1

Issue24 adds `omega_gpu_admission.h` and an explicit native DMA adapter. It is
opt-in. Existing `omega_gpu_platform_linux()` and its #8 environment guard are
unchanged. No platform, session, buffer or serving ABI is widened. The production
archive includes the API through the existing source wildcard. Callers select
this API instead of direct `dma_alloc`/`dma_free`; existing callers retain the
legacy baseline. Session integration and default enabling await coordinated
integration review and hardware evidence.

## Limits and ownership

One group is a persistent, local-filesystem directory, owned by the effective
uid, mode 0700. Create it before opening the API. All cooperating participants
use that exact directory and the same fixed byte budget and free-memory floor.
Do not rename, replace, unlink, truncate or reset a group while participants
exist. No symlink directory component is followed. Lock and state files must be
regular, owner-only 0600, single-link files owned by that uid. This coordinates
trusted processes of one uid; it is not a security boundary against that uid or
root. Nonparticipants, other groups, the kernel, channel creation and host
allocations can consume memory outside it. There is no system-wide OOM guarantee.

API v1 supports Linux, a local filesystem with flock, pread/pwrite and fdatasync,
8192 outstanding reservations per group, and at most 128 GiB per budget or floor.
Limits are binary bytes. Values outside the limits fail. There is no environment
parser or implicit fallback in this API. Use a new versioned group for a different
budget. Shared-weight admission is refused until an actual native import and
cross-process lifetime contract is integrated. Private weights are fully charged.

Each reservation has a persistent nonwrapping generation and a slot. Its owner
is boot UUID + PID + /proc/PID/stat start time. A token from another process,
inherited context after fork, stale generation or invalid transition refuses.
Do not fork while operations are in flight or inherit live GPU objects. Each
operation takes a thread mutex and a nonblocking group flock; contention returns
-EBUSY for bounded caller retry. Locks are released by process death. Unknown
/proc identity fails closed; absent, zombie, different-boot or reused-PID owners
are reaped under the same group lock. A fresh memory sample still gates reuse:
reaping metadata does not assert immediate reclamation of physical pages or
completion of kernel teardown. Existing #6 retire, timeout parking and device
close rules remain mandatory.

## Transactions

- reserve: validate settings and fresh memory, sum every live charge, then issue
  RESERVED. No allocator call. Refusal publishes a zero token and no new charge.
- acquire: RESERVED -> ACQUIRED, once, by the owner before allocating.
- commit: ACQUIRED -> COMMITTED; the charge remains until free.
- cancel: RESERVED -> empty. It cannot uncharge an in-progress allocation.
- abort: ACQUIRED -> empty, only after allocation failed with no live buffer.
- release: COMMITTED -> empty, only after work retired and free succeeded.

Close refuses outstanding charges for the process. Seal forbids reserve,
acquire and adapter allocations, including hidden growth. Cancellation, abort,
commit and release remain possible. Existing serving reserve/seal APIs are
independent and unchanged; this API does not retroactively tag their buffers.

The DMA adapter validates the existing descriptor contract, rounds charges to
nvrm's 4 KiB allocation size, persists ACQUIRED before calling the platform,
commits on success and aborts on allocator refusal. Successful buffers bind the
reservation to platform/context plus buffer index/generation. Free validates that
binding, calls the original platform free first, then uncharges. A refused free
keeps the binding and charge. Use the adapter consistently for each opted-in
buffer. Raw transitions cannot release adapter-bound tokens through that context.
The low-level reserve API relies on its trusted caller to supply the stated
allocation/free proofs; it cannot prevent deliberate direct nvrm calls.

Persistent writes carry a whole-state checksum. Interrupted/torn writes or
storage faults quarantine the group and refuse further admission, rather than
guess at lost allocations. Recovery from such corruption requires an operator
proving all participants and driver objects are gone before creating a new group.
A storage failure after a successful delegate allocation can return an error with
a live output handle; that handle must be quarantined and disposed using proven
device teardown. A returned error is never permission to drop live ownership.
This exceptional fault path is separate from ordinary transactional refusal.

## Memory and accounting policy

Every reserve reads /proc/meminfo anew while holding the group lock. Missing,
malformed, duplicate, overflowed or unreadable MemAvailable refuses. There is no
reuse of an old sample. Explicit fixture files are test inputs: they must be
regular, nonsymlink files whose modification time is neither future nor older
than the configured 1..3600 seconds. They are never performance evidence.

Admission requires `request + all active charges <= budget` and
`request + all active charges + keep_free <= MemAvailable`, using subtraction
checks to prevent overflow. Already resident buffers can therefore be subtracted
twice in the safety decision. This deliberate conservatism must never be reported
as measured memory consumption. Unrelated pressure remains a distinct fresh
MemAvailable observation and can change immediately after the check.

Stats expose reserved/acquired/committed bytes, explicit categories (private
weights, intermediate, KV, marker), and measured process RSS. Shared weights
remain zero because imports are refused. RSS includes buffers and is never added
to explicit bytes. Process overhead is UNKNOWN (`overhead_known=0`); deriving it
by subtracting GPU allocations from RSS would fabricate disjoint accounting.
No #14 native overhead API exists in this CUDApp base. #14/#17 must supply a
measured, disjoint accounting source before reporting actual process overhead.

Cached data retains Linux's existing restriction to device-private data without
CPU_READ or HOST_POLLED. Markers must explicitly use the marker category,
HOST_POLLED and GPU_UNCACHED. The adapter does not alter sync, C3 flush, polling,
channel, or completion behavior. A host fake driver checks policy and call order;
it does not qualify GPU visibility.

## Gates

`make OUT_DIR=build-roadmap-reservations24 test-roadmap-memory` runs real-process
contention, cancellation, SIGKILL cleanup, ownership/generation checks, malformed
memory/security refusals and the actual Linux platform linked to the simulated
driver. No GPU device is opened. The legacy double-spend demonstration is an
expected failing test: `test_omega_gpu_admission --legacy-gap`.

Hardware allocation, existing serving readiness, cached-weight visibility,
process-death kernel teardown, whole-model correctness, shared import lifetime,
performance, and actual disjoint #14 overhead are NOT_RUN. No default enablement,
optimization, issue closure or deployment claim follows from the host gates.
