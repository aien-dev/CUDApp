# Canonical motor candidate, L457

**HOST QUALIFIED, chip qualification pending.** This receipt proves a real serving binary was built from pinned canonical GPU sources, with both native ABI checks passing and both archive digests locked. It does not prove model-output parity, concurrency, recovery, performance or an independent driver. Nothing was deployed to 18091.

The exact source chain and artifact hashes are in `receipt.json`. The revision in this receipt is CUDApp 349913f1 / Omega 1ad2dc1a / sc aaab111c; the earlier revision and the revisions before it keep their own logs on this branch. Cargo identity logs are unedited build-script output. `abi13-sc-build.log` ends in `RC=0` for the source chain in this receipt; `sc-native-final-build.log` retains the earlier successful artifact. `abi13-zero-cuda.log` records that the binary imports only ordinary Linux runtime libraries, not NVIDIA user-space libraries. The binary is test-only (`aien-test-switches`) and is not a production-swap artifact.

The source map removes duplicate GPU ownership from Omega and binds it to a full CUDApp commit. Twelve behavioral checks cover successful binding and refusals for a missing checkout, wrong pin, disabled check, modified source, modified build metadata, shadow copies, unmapped GPU sources, unsafe paths external source symlinks, ignored untracked source overlays, and malformed pins with embedded whitespace.

## Failures retained, not waived

- `cudapp-host.log`: initial carry did not yet include the multi-channel driver dependency. This attempt included in-progress build integration, so it is not an exact-commit acceptance receipt.
- `cudapp-host-driver-green.log`: the driver compiles, but old experimental-capacity metadata and missing portability rows correctly fail.
- `cudapp-host-final.log`: old GPR test initializers correctly fail with the extended motor structs and `-Werror`.
- `cudapp-host-complete.log`: the command-list fixtures also need explicit zero defaults for the appended groups field. All thirteen cases were preserved in the subsequent fix.
- `sc-native-build.log`: the old Rust CTA contract and missing pinned AienOS capability source correctly refused the build. The ABI/FFI/tests were updated to CUDApp's existing capacity, and the actual pinned AienOS repository was supplied.

No warning flag, check, test case, or refusal was removed. The final standalone host suite passed all33 targets and the supplemental C3/C4/C5/q38 host/simulator gates at exact source04607cf7. See cudapp-host-qualified.log. The ignored-source and malformed-pin repros failed before the linker fixes; all12 behavioral binding checks now pass at Omega7052d780. The serving build is being requalified after those build-script-only changes. Host-only ABI checks and portability negative controls already pass; their logs are retained.

## Platform contract move (0.2 / 1.3)

The handle table and launch capacity the motor carries are not additive changes: the platform contract names `OMEGA_GPU_HANDLE_SLOTS` and sizes `OmegaGpuHandleTable`, so the platform minor moved 1.2 to 1.3 with runtime minor 0.1 to 0.2, as `docs/runtime/API.md` requires. The two tests that hard-coded the old number now assert the new value exactly (one binds the header constant); neither was weakened. `device_close` returned `int` before this candidate too, so no callback signature changed.

## Remaining acceptance gates

Linked native host checks passed (41 tests, zero failures). The chip-only parity case was not run and remains a required hardware gate. Independent review and the shared serving train on the dedicated test port 18157 are still required. Any hang/reset/module testing stays out of this value-only request. Live serving swaps remain coordinator-owned and require qualification. Issue CUDApp#5 stays open.
