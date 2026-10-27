## Purpose and scope

Explain the user-visible change and the exact source paths involved. Link the tracking issue; distinguish implemented behavior from planned behavior.

## Evidence on the exact commit

Record `git rev-parse HEAD`, OS/architecture, toolchain, full commands and exit codes, and new receipt path. Mark each item **PASS**, **FAIL**, or **NOT RUN** (with reason); do not check a box for an unexecuted gate.

- [ ] `make test` (host-only)
- [ ] `make test-native-artifact` (host-only)
- [ ] `make zero-cuda-gate` (ELF audit and negative control)
- [ ] `build/hello_gpu --sim` (simulator only)
- [ ] `make doctor` (diagnostic; NOT READY on unsupported machines is expected)
- [ ] GB10 hardware tests, **only if an authorized GPU window is available** (otherwise NOT RUN)

## Correctness, compatibility and provenance

- [ ] Negative controls and affected contracts are documented.
- [ ] API/ABI and Omega consumer effects are identified; source and runtime status claims are not overstated.
- [ ] Third-party material, licensing, and public-history safety were checked for additions.
- [ ] No secrets, personal machine paths, private addresses, other processes' logs, vendor binaries, or unauthorized generated artifacts were included.
- [ ] Existing receipts remain immutable; new results have their own evidence folder.

## Known limitations

List unsupported hardware, missing functionality, pre-existing failures, and tests not performed. Documentation-only changes should explicitly state that no builds or chip tests were run when that is the case.
