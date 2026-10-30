# Current-contract docs and stack legend repair

Host-only receipt, 2026-10-27. No GPU device was opened.

- Base: `bba6da8045ba5baaf0a81f7152af44d52e24f563`.
- Tested source: `8b022055ad01bc148ff4f8fa6c7443995e5b5510`.
- Machine: Linux x86_64, kernel `6.18.55`.
- Tools: Python `3.12.14`, Inkscape `1.4 (e7c3feb100, 2024-10-09)`, GCC `14.2.0`.

## Scope

Move only the open-source and firmware legend items so the longer first-party
licence label does not overlap them. Keep all legend text and the figure size.
Add a focused Inkscape geometry check and a deliberately overlapping negative
control. The normal build and runtime do not acquire an Inkscape dependency.

Correct the current decode-tool requirements and residual present-tense claims
about the removed trace listing. Preserve #84/#93 history and #98's existing
corrections. Qualify the L401 kernel-module scope as pending a separate kernel
decision. No runtime, swap, driver, test-runner, licence or architecture changes.

## Results

| Command or check | Result | Evidence |
|---|---|---|
| `python3 tools/figures/test_stack_legend.py -v`, before the fix | exit 1, rendered items 3 and 4 overlap; negative control passes | `legend-red.log` |
| Same command on tested source | exit 0, both tests pass | `focused.log` |
| `python3 tools/figures/make_figures.py --check` | exit 0, `figures current: 10` | `focused.log` |
| Inkscape before/after crops and queried bounds | overlap before; clear labels after | `legend-geometry.txt` and independent review |
| Python AST, shell syntax, Markdown parse, whitespace | pass | `docs-checks.log` |
| Current runner/tool references | source-checked, not executed | `docs-checks.log` |
| `make test` | exit 2, build failure before suite completion | `make-test.log` |
| `make -j2` | exit 2, build failure | `make-build.log` |

The red check ran on `061493a39608b6e4eed261cb52c6d024b374f1b4` with the new test.
The generator is byte-identical on the later base `bba6da8` (SHA-256
`7375e608e7edcf494916e00b676072c3cf4e9b49f35f762f0b34ad46a4897564`). The independent
review also replayed that base generator and reproduced the same failure.
`legend-red.log` replaces the private checkout prefix with a repository-relative
path; no result text is changed. Source-commit runs reuse earlier build outputs.

### Build and full-suite failures

The x86_64 assembler rejects unchanged AArch64 barriers:

- `src/omega_blackwell_engine.c:393` and `:558`, `ds` operand mismatch.
- The parallel build also reaches `src/omega_blackwell_submit.c:279`, `:372`,
  `:532` and `:613`, with the same operand mismatch.

Neither build nor the complete host suite passed. No barrier, gate or compiler
setting was changed. A native qualified-host build and host-suite run remain
required. The trace-emitter runner and all GPU/chip checks are NOT_RUN here.

## Review and publication limits

An independent read-only reviewer reproduced the baseline failure, reran both
focused tests and the figure check, inspected the before/after pixels and verified
that only legend x coordinates changed in the generated SVG. It checked the
prose against the runner and current tree. Final verdict: no remaining review
findings. See `REVIEW.md` for the exact reviewed content hashes.

The repository's audit scanner self-test passed with all planted controls found.
A scan of the source patch and new test reported no secret or private-pattern
hits. Commands and output are in `audit.log`. This is a patch-level check, not a
new audit of all repository history.
