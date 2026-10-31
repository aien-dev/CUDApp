# Chip-helper host-only repair: 2026-10-27

## Identity and scope

- Baseline: `9841bb3a5dfbde1c7bf7741fd2fceb916962c8cb` (merged #113).
- Tested local source commit: `742c453c4a62f1c47e97a1d3a802e278218d8cc8`.
- Tested source tree: `71ccf551d494dd70b744feecc66c3214850067fc`.
- Machine: Linux 6.18.55, x86_64; Python 3.12.14; GCC 14.2.0.
- Only shell orchestration, its host tests, and CI wiring changed. No C or
  Omega runtime implementation, instruction encoding, or hardware result changed.
- Existing C4 chip receipts remain immutable. This is not GPU qualification.

## Reproduction and results

Every executable child in the regression suite is a temporary dummy script.
Each lock is an isolated temporary file. Contention uses real Linux `flock`;
a test-only PATH shim makes the production wait nonblocking, without changing
production locking or opening the GB10 lock. No GPU device was opened.

1. Copy the two baseline helper blobs with `git show
   9841bb3a5dfbde1c7bf7741fd2fceb916962c8cb:tools/<helper>` into a temporary
   `tools/` directory, then run `CUDAPP_CHIP_HELPER_ROOT=<temporary-root>
   python3 tests/test_chip_helpers.py -v`.
   **Expected FAIL**, exit 1: 9 tests, 7 failing assertions/subtests
   (`baseline.log`). These expose zero/empty counts, missing locking,
   contention admission, and suppressed early/late child failures.
2. At the tested source commit, `python3 tests/test_chip_helpers.py -v`:
   **PASS**, exit 0, 9 tests (`repaired.log`). Positive controls keep the
   default 10 runs, requested count, complete 11-command success path,
   channel environment, failure rejection, and missing-verdict rejection.
3. `sh -n tools/c4_streams_chip.sh tools/l136_channel_gates.sh`:
   **PASS**, exit 0. `git diff --check`: **PASS**, exit 0.
4. Python AST parse and PyYAML parse of the changed test/workflow:
   **PASS**, exit 0.
5. `make test`: **FAIL**, exit 2, before running the full suite
   (`make-test.log`). The x86 assembler rejects existing ARM `dsb sy`
   instructions at `src/omega_blackwell_engine.c:558` and `:393` (diagnostic: number
   of operands mismatch for `ds`). This file's blob is unchanged from baseline:
   `bc053f133613d3dd22930f9eb461a426f397e3d3`.
   Native ARM64 exact-head CI is still required; no C workaround or gate
   bypass was used.
6. Separate `make test-native-artifact`, `make zero-cuda-gate`,
   `build/hello_gpu --sim`, and `make doctor`: **NOT RUN locally**;
   full build is blocked as above. Existing PR CI covers its normal host
   and simulator tiers without altering their gates.
7. GB10 tests: **NOT RUN**. No authorized hardware window was requested.

Logs are captured output, with the local checkout prefix replaced by
`<checkout>` in Python tracebacks and trailing whitespace removed from the
baseline log. These formatting changes do not alter verdicts.

## Independent review and ownership

A separate reviewer reported no findings and independently ran all 9 tests,
shell syntax and whitespace checks. Additional dummy checks confirmed a
missing first executable propagates 127 without launching a child or writing
`done`, and an existing lock retains its contents and is held for all children.

Reviewed combined patch SHA-256:
`7360405df7ed6c08fd405a393eebb68ea5a70119773f312af783678da6ca0a79`.
The hash input was the tracked binary diff against baseline followed by
`git diff --binary --no-index -- /dev/null tests/test_chip_helpers.py` before
staging the new test. No source edits followed review.

All 15 open PR file lists were rechecked; none touched these four source
paths. Coordination with C4Streams / TrainCarry2 is recorded on
[#113](https://github.com/aien-dev/CUDApp/pull/113#issuecomment-6436534499).
The .om successor in #115 and C5 work in #119 retain their existing owners.

## Publication audit

`bash tools/extract/audit-selftest.sh`: **PASS**, 15 planted / 15 found.
Delta audit: 8 blobs, 0 binaries, 0 secret hits; 2 private-pattern hits, both
the reserved fixture address described below.
Only new/changed publication files are audited for this patch; historical
receipts are unchanged. The fixture identity `test@example.invalid` is a
reserved dummy address, not personal data. Publication uses exact blob/tree
comparison; any API-vs-local commit identity mapping is stated in the PR.
