# L641: G15 increment; M1–M5 request remains OPEN

**Qualified functional source:** `014f0a11f1cd98031206f179b181ad74be93bd33`.
This receipt records the implemented M3 increment, not completion through M5.
M1/M2 retain the shared G13/G14 bounded virtual subset. M4/M5 deliberately
refuse; their evidence gaps and next discriminating experiment are in
[`native/gpulab/APPLE_GENERATIONS.md`](../../native/gpulab/APPLE_GENERATIONS.md).

## Executed result

A fresh `git archive` of the functional pin was exported to a new directory.
The source-manifest check, full native suite, and comparison of fresh-built
binaries with the working candidate each exited 0. Original command result:
`FRESH_REPLAY_PASS source=014f0a11f1cd98031206f179b181ad74be93bd33`.

| Gate | Actual result |
|---|---|
| G15 / M3 numerical, encoding and refusal assertions | 140293 checks, 0 failures |
| K1 single-bit mutation classification | 559 refused, 37 changed answers detected, 12 equivalent on the probes |
| Complete K64 M3 shader | 3100 bytes match independently assembled frozen fixture |
| Same-length corrupted external fixture | refused, exit3, zero stdout |
| M1/M2/M3 explicit family paths | pass, every supported depth, three rows |
| M4/M5 family refusal | exit65, zero emitted stdout; no older-ISA fallback |
| Existing AMD / AGX / cross-target gates | 8224/0, 8218/0, 262930/0 |
| Existing AMD full-shader LLVM comparator | PASS |
| Standalone request/emitter/wire path | PASS for legacy AMD/AGX and explicit M1/M2/M3 |
| Physical GPU execution | NOT_RUN |
| Speed and full-model inference | NOT_MEASURED / NOT_IMPLEMENTED |
| Independent source review / merge | PENDING desk; author review only |

`fresh-replay.log` is the exact newly executed suite output.
`fresh-source-check.log` checks the pinned source/fixture closure.
`native-binaries.sha256` names the six new or changed path executables checked
against the fresh rebuild; `rebuilt-binary-check.log` records all six matching.
`binary-types.log` records the inspected static AArch64 Linux executables.
Historical receipts inherited by the archive were not relabeled as new runs.

## Reproduce

Use the pinned seed-native compiler SHA256
`191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.

```sh
sha256sum -c evidence/NATIVE-APPLE-L641/source-files.sha256
quiet-build make -f native/gpulab/Makefile test \
  OMEGA_NATIVE_COMPILER=$HOME/handoffs/quantum-development-L600/toolchain/build/bootstrap/omc
sha256sum -c evidence/NATIVE-APPLE-L641/native-binaries.sha256
```

All first-party executable work is Omega. Shell/Make are process/build glue.
LLVM remains an external AMD oracle. Unmodified upstream Python was invoked on
the Mac solely as an independent M3 assembler to create the pinned fixture;
it is neither linked nor required by the native build/test/execution path.
See APPLE_GENERATIONS.md for upstream source closure, actual invocation and
BSD-3-Clause attribution. Wolfram evaluator input/results are in WOLFRAM.md.

## Failures preserved and corrected

- Initial test-source compilation failed error12 on a deeply nested scalar
  reference expression. Splitting its loads into locals resolved register
  pressure. This was not counted as behavioral RED.
- The compilable M3 refusal baseline then failed assertion701, exit1; raw output
  is `m3-red.log`.
- The first implementation compile failed error13 at the nine-character local
  `remaining`; `m3-first-compile.log` is preserved. Renaming to `remain` met the
  seed identifier limit. `m3-second.log` is the subsequent native success.
- The family gate ran against the old dispatcher plus a refusal executor and
  returned1 before implementation; the final gate and wire refusals are in the
  full and fresh logs. This initial family status was captured by the terminal,
  not saved as a separate raw log.
- Final whitespace validation caught an extra blank line in notices; it was
  removed, without changing executable inputs.

The local Mac inspection returned Apple M2 and no `metal` or `metal-tt` through
xcrun. Published external M4/M5 reports were read, not treated as our own chip
measurements. No device driver, firmware, serving process or production caller
was changed. All raw emitted GPU bytes remain non-executable lab data.
