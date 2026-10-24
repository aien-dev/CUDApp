# PR 67 teardown retry candidate

Refs CUDApp [#65](https://github.com/aien-dev/CUDApp/issues/65) and
[#67](https://github.com/aien-dev/CUDApp/pull/67), including review
[5637134696](https://github.com/aien-dev/CUDApp/pull/67#pullrequestreview-5637134696).

This is an **unqualified source candidate**, based on PR 67 head
`5e037c7e3d0226abadaa10af9d44bc6339109b3f`. The candidate source and test revision
is `ff5b4f9f1b13cffff2669e8207fe38cb5831366f`. Native compilation, native test
execution, GPU execution, model inference and deployment are **NOT_RUN**.

## Repair and review

An unsuccessful `mapdrop` leaves its mapping handle owned. The old `nvstop`
cleared the BAR handle despite refusal and marked the context closed even when
channel or buffer mappings remained. A later `nvstop` could consequently report
success without retrying cleanup.

The repair records a distinct phase after descriptor closure, retains failed
mapping handles and retries only CPU mapping cleanup in that phase. It preserves
the sticky fault and retires each descriptor after one close attempt, including
close errors. No RM/UVM call or descriptor close is repeated after phase 1.

An independent source review traced normal cleanup, each mapping refusal,
driver teardown refusal and close-error paths against the compiler's mapping
semantics. It found no additional blocker in the runtime change. This review
is source analysis, not a signed supported-host qualification or native test
pass. Its suggested live-driver phase refusal check was added to the fixture.

The compiler source contract was checked at Omega
`d7390d88d07dbf48cfbb0583070599c350f814d3`, `seed/omc/omc.om`: unsuccessful
`munmap` returns without clearing the live mapping slot. Invalid or stale
handles trap and are not used as a substitute for the syscall refusal case.

## Verification recorded here

| Check | Result | Scope |
| --- | --- | --- |
| Shell syntax | PASS | `sh -n tests/speculation/run-teardown-host.sh` |
| Baseline and repaired source-unit composition | PASS | Extracted the actual composer from the tracked shell script; did not compile either unit |
| OS-effect isolation guard | PASS | Both composed units accepted; an appended unbound `devctl` rejected in each |
| Whitespace check | PASS | `git diff --check origin/experiment/27b-adaptive-sizing HEAD` |
| Native host test | NOT_RUN, exit 77 | Workspace is x86_64; no runnable pinned AArch64 compiler or emulator |
| Required legacy host suite | FAIL, exit 2 | `make -j2 OUT_DIR=build/teardown-review test`; existing unconditional AArch64 assembly at `src/omega_blackwell_engine.c:393,558` is rejected on x86_64 |
| Already-started legacy mutation target | PASS, 49/49 killed | Separate legacy host target, not qualification of the native repair |
| GPU/model/deployment | NOT_RUN | No device access or runtime activation |

`full-host.log` retains the complete failing host-suite output. The full host
suite ran against the same runtime change; later commits only adjusted the
test orchestration, added a fixture assertion and updated documentation.
`source-composition.json` records the exact units' hashes and guard exits.

## Required supported-host step

Follow [TEARDOWN-HOST.md](../../docs/speculation/TEARDOWN-HOST.md) on Linux
AArch64 using the pinned compiler. The original source must compile and fail
by the intended assertion; the repaired source must compile and pass. A
compiler error, trap, timeout or unsupported-host skip is not a pass. Preserve
the produced source units, ELF files, compiler logs, stdout/stderr and hashes.

The current candidate must remain draft/unqualified until that step and all
existing review and signed integration gates are satisfied. Older hardware
receipts cannot qualify this changed source, and the synthetic fixture does
not qualify 27B inference or production activation.
