# QMD standalone clean-output build repair

## Scope and identity

This is a host-only build-rule follow-up to CUDApp #178.
The production QMD builder, its shared-memory cap, existing C test assertions,
public APIs and historical receipts are unchanged.

Base: `b348277e4383ab1f37f45bb702ea7d4fb4363782`.
The tested replacement Makefile has Git blob
`609a004ab095dec51a7d5ef1d319d07a30425976` and SHA-256
`105e81766f32584be8d8730d3b7e54a0132bdad1828248d64519736383ef7203`.

Local validation used a source-bounded fixture, not a full repository checkout:
the exact Makefile, all mk/ include files, and these unchanged source blobs
were fetched from the base and their Git blob identities checked.

- `src/omega_blackwell_qmd.c`: `dd13ec5a2c0d712627240afa8b1151e45643cd06`
- `src/omega_blackwell_qmd.h`: `1d56f1a9198ee0e7f8d18f2712207b7a6c37b21a`
- `tests/test_omega_bw_qmd_shared.c`: `ea877dbe92010691daed3ea4ffa2dd31e9700ffb`

Machine: Linux x86_64. Measured tool versions: GNU Make 4.4.1,
GCC 14.2.0, Python 3.12.14.

## Defect and repair

The standalone test target did not depend on its output directory.
With an absent OUT_DIR, compilation failed while opening the dependency file.
Other host targets could create the directory first and hide this defect.

The repair adds the existing OUT_DIR rule as an order-only prerequisite.
The retained `test-gpu-qmd-shared-clean-build` regression creates a temporary
parent, leaves its child output directory absent, and invokes the ordinary
host-only QMD test target with that child as OUT_DIR. It is in HOST_TESTS.
The recipe propagates failures and removes its own temporary directory.

## Measured host outcomes

Commands and captured output are in `host-checks.log`.

- Same regression with only the directory prerequisite removed in a separate
  copied fixture: expected exit 2, missing output-directory error.
- `make test-gpu-qmd-shared-clean-build`: exit 0,
  `qmd shared: 112 checks, 0 failed: PASS`.
- `make -j4 test-gpu-qmd-shared-clean-build`: the same PASS, no jobserver warning.
- Direct `make OUT_DIR=<absent-directory> test-gpu-qmd-shared`: the same PASS.
- `CC=false`: expected exit 2. The temporary directory was removed.
- A missing TMPDIR: expected exit 2 before compilation.
- `make -n test-gpu-qmd-shared-clean-build`: exit 0, commands printed,
  no QMD binary executed. The temporary directory was removed.
- `git diff --check`: exit 0.
- A separate reviewer independently reproduced the fresh-output, parallel,
  removed-prerequisite, compiler-failure, cleanup and dry-run controls.
  No remaining finding in the final Makefile blob above.

The negative control must use a separate fixture's default Makefile: changing
only a `make -f` parent would leave the recursive make reading the original file.

The initial review found that directly executing the binary outside recursive
make broke dry-run behavior. The final recipe invokes the ordinary test target,
so recursive make carries the dry-run flag through both build and execution.

## Verification boundary

Full `make test`, complete library build, native-artifact and zero-CUDA gates
were NOT_RUN locally in this source-bounded fixture. The draft PR's exact-head
ARM64 CI is the full-repository gate; its result is reported separately.
No simulator result or physical GPU result is claimed by this receipt.
No device, driver, live service, installation or deployment was touched.
No existing receipt was edited.

The repository publication scanner's self-test passed: measured output
`SELFTEST PASS: planted 17 ... found 17; binary listed`.
The changed-file publication scan is recorded in the pull request.
