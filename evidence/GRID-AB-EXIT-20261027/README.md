# Grid A/B exit status repair

Base: `5b91c6a66595a0d876831107d6a7c8a4872f7d54`.

The caller discarded `grid_ab`'s return value. Invalid lists returned 2
before any CHECK, but main printed PASS and exited 0. Main now closes the
session and preserves that nonzero status, without printing PASS.

`sh tests/grid_ab_exit_test.sh ./build/gpu_fp8_api_test` checks empty,
malformed, single-arm and malformed-later-arm lists. These inputs return
before command-list creation. This regression runs in `make test` through
`test-gpu-fp8`. It never intentionally opens a device.

Local x86_64 validation:

- An isolated probe compiled the unchanged main dispatcher and grid parser
  from the source, with nondelegating stubs for dependencies. Baseline
  empty-list output was `0 checks, 0 failed: PASS`, exit 0. The regression
  failed as expected. With the patched dispatcher all invalid-list cases
  passed. This is dispatcher coverage, not execution of the full binary.
- `sh -n tests/grid_ab_exit_test.sh` and `git diff --check`: PASS.
- `make test`: exit 2. The unchanged AArch64 instructions in
  `src/omega_blackwell_engine.c:393,558` fail on this x86_64 assembler
  (`number of operands mismatch for ds`). Full local suite NOT_RUN past
  that compile failure. ARM64 CI is required for the real binary.
- GPU execution, benchmark reruns, deployment and qualification: NOT_RUN.

No kernel, runtime API or historical measurement changed. The separate
documentation correction distinguishes seed omc IR from OSC IRv7 as
specified in omega-mv issue 883; it does not upgrade either capability.
