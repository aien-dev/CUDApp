# Issue24 host receipt

Status: DONE_HOST / AWAITING_HARDWARE. Experimental, explicit selection only.
No default enablement, deployment, issue closure or optimization claim.

Executable source: `264c7752bceb38e18b4f58125a07ad1719b1ad6b`.
Final gate source: `1aea93c6fda0d6012087b6559a47448af79085d3`.
Base dependency: `349913f17c083996680c72b32f767b225005155d` (canonical carries).
Public review baseline: `232332422bfdc31c8918085b50f13c1e18ab702e`.
The gate commit adds only two portability-table rows; `unchanged-inputs.txt`
proves executable/build/test inputs unchanged. This receipt is an evidence-only
follow-up, so its own commit does not claim to be its executable source.

## Results

- Own target: **228 parent checks PASS**, plus assertions in real child processes.
  `green-pinned.log` exits 0 on the final gate source. The rebuilt binary is
  byte-identical to the earlier tested binary (`binary-*-rebuild.sha256`).
- **8/8 behavioral mutants rejected by assertions**, not crashes/timeouts:
  budget, foreign owner, PID start time, boot identity, seal, missing memory,
  stale generation and allocation before refusal. `mutants-final.log` and
  per-mutant raw logs/patches are retained. Earlier harness alarm failure is
  retained and explained in `commands.txt`; it is not counted as a killed mutant.
- The actual unchanged legacy Linux guard admits **both** synchronized child
  allocations against one fixture budget. `red-legacy-double-spend.log` exits 1
  as expected. The new group admits exactly one contender while both remain alive.
- Full existing host suite: **33 targets PASS** across the original log and its
  continuation, final exit 0. The first run was intentionally stopped before
  recursive scratch cleanup; the first three completed targets were not repeated.
  Both raw logs and exact continuation command are retained. No cases or thresholds
  were removed. The cleanup shim restricts deletion to owner output/test scratch
  and uses find instead of recursive rm under the workspace.
- Additional canonical host/simulator gates: **4 targets PASS**: C4 streams
  9 checks, C5 isolation 5 checks, C3 GEMM 154 checks, Q38 graph 4 checks.
- Actual Linux DMA adapter linked to simulated nvrm: transactional refusal occurs
  before allocator calls; rollback, 4KiB charges, free refusal, cached-data and
  uncached-marker policy pass. `adapter-syscalls.log` traces the final adapter test;
  **no NVIDIA device open or ioctl** occurred. This proves host execution only.
- Portability registration RED and subsequent full-suite GREEN are retained.
  Production archive symbols prove the new native API is included in the library.

## What was implemented

A versioned, bounded native reservation API with process owner identity
(boot UUID/PID/start time), persistent nonwrapping tokens, a stable kernel lock,
atomic state replacement, conservative fresh memory admission and strict malformed
input/security refusals. Real-process tests include normal abandonment, SIGKILL
of a committed owner, inherited-context refusal, stale ownership fixtures, and
SIGKILL halfway through a real state write. The previous complete transaction
survives. An explicit DMA adapter exercises the actual Linux allocation boundary
without widening any existing ABI or changing the legacy default.

Stats distinguish group reserved/acquired/committed bytes and private buffer
categories from fresh external MemAvailable and measured inclusive local process
RSS. The API refuses shared weights until durable imported-object ownership is
proved. RSS is not summed with explicit buffers or reported as disjoint overhead.
See `docs/runtime/COOPERATIVE_ADMISSION.md` for exact semantics and supported limits.

## Limits and NOT_RUN

Unrelated/kernel/channel/host allocations remain outside the cooperative budget;
this is not a system-wide OOM guarantee. Admission conservatively subtracts all
active charges from fresh available memory, including resident buffers. Those
safety subtractions are not measured consumption. Corrupt state/storage faults
refuse; exceptional storage failure after a successful allocation retains live
ownership and may require proven device teardown.

Actual disjoint process overhead through #14 is **UNKNOWN / NOT_RUN**. No native
source for it exists in this base; `overhead_known=0`. Integration must supply a
measured disjoint source before claiming issue24's overhead acceptance.

Existing Linux `ref_dma_free` ignores `nvrm_free` errors. The adapter correctly
preserves charges when a platform reports free refusal; canonical integration
must fix and qualify the underlying error propagation before real failure-path
qualification. This stream leaves that existing owned file unchanged and records
the concern in the L475 interface handoff.

GB10 bounded allocation, process-death kernel cleanup, cached-weight/marker
visibility, actual disjoint overhead, shared import lifetime, serving readiness,
whole-model correctness, performance and x86_64 runtime: **NOT_RUN**. No model
request, server/model start/restart, GPU/chip/driver/module work or child agent
was used. The passive driver version is recorded in `pins.txt`, not a qualification.

Issue25 follow-on packet: `issue25-capability.md`, with pinned read-only API
inspection in `issue25-inspection.log`. Existing shared-object APIs import RM
allocations, not arbitrary file pages. No issue25 implementation was attempted.

## Reproduction

Exact commands and exit codes: `commands.txt` and raw logs. Source/compiler/tool/
configuration pins: `pins.txt`, `source-tool-sha256.txt`. Final native binary,
archive and mutant hashes: `binary-sha256.txt`. Host/simulator/hardware matrix:
`matrix.tsv`. All memory test inputs here are labeled fixtures; no model
checkpoint/config/prompt was accessed. Independent fixture arithmetic is in
`independent-arithmetic.log`; no paper-derived tolerance is used.
