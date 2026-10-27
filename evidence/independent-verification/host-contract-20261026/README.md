# Host-only documentation and manifest checks

Measured 2026-10-26 on Linux x86_64, GCC `14.2.0`, Python `3.12.14`, external
`jsonschema` package `4.26.0`. Base: `c73afb25cce8d22e039287a8ab2e4cd4c7aecf67`.
The changed document/schema hashes are listed in `artifacts.sha256`.
This is not native AArch64, GPU, independent-operator or release qualification.
No device was opened. No benchmark or claim of physical proof was produced.

## Contract checks

An external JSON Schema referee checked draft 2020-12 schema validity and the
empty template. Before the schema/template were added, the check stopped with
`AssertionError: schema and NOT_RUN template must exist`, exit 1. The final
shape/refusal checks exited 0; raw summary: [schema-checks.log](schema-checks.log).
Synthetic in-memory data was used only to test validation, never saved as a
receipt or labeled as physical evidence.

Reproduce the positive structural check from the repository root using an
already available external JSON Schema validator; do not install it into the
product or add a runtime dependency. For the measured external referee:

```sh
python3 - <<'PY'
import json
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker
root = Path('docs/independent-verification')
schema = json.loads((root / 'manifest.schema.json').read_text())
template = json.loads((root / 'manifest.NOT_RUN.json').read_text())
Draft202012Validator.check_schema(schema)
formats = FormatChecker()
print('date-time format checker available:', 'date-time' in formats.checkers)
validator = Draft202012Validator(schema, format_checker=formats)
validator.validate(template)
print('EMPTY_TEMPLATE_STRUCTURALLY_VALID_NOT_PHYSICAL_PROOF')
PY
```

The measured negative checks copied the empty template in memory, applied
one change per copy, and required `list(validator.iter_errors(copy))` to be
nonempty. Each of these copies was rejected:

| Mutation | Refusal boundary |
| --- | --- |
| Delete `checks.no_submission` | Required control row |
| Template `checks.correct_kernel.status = COLLECTED`, evidence = `["fake"]` | Template cannot claim collection |
| `claim_status = PASS` | No physical verdict field |
| Add top-level `physical_proof = true` | Unknown field |
| Template `operator.identity = test-operator` | No template attribution |
| Set `record_kind = operator_report`, correct-kernel status COLLECTED, no evidence | Collection requires artifact references |
| Delete correct-kernel `observation` | Required observed meaning/blocker |

A separate synthetic operator-report shape added artifact `test-file`, path
`test.stdout`, SHA256 of 64 lowercase `a` characters, byte count zero, no
redaction, and null raw custody fields. Its correct-kernel row was COLLECTED
with evidence `["test-file"]`, empty commands and observation explicitly saying
synthetic. It was structurally accepted. **No such artifact or event existed.**
This control shows why structural acceptance cannot establish a real result.
Starting with that shape, these independent mutations were rejected:

- SHA256 `xyz`.
- An absolute public path or `../file` traversal.
- Byte count `-1`.
- Redaction `sanitized` without raw hash/custody/sanitization-report identity.

Independent review additionally found that `$` anchors accepted a final
newline in digests, commits, IDs and public paths, and date-time format alone
allowed non-UTC offsets. These inputs were reproduced as unexpectedly valid
before repair. Exact digest lengths and actual-end patterns now reject the
newline cases; timestamps have an explicit UTC lexical/range pattern ending
in `Z`. The local `FormatChecker` has no optional `date-time` implementation,
so a suffix alone also accepted invalid text. The new pattern rejects
`not-a-dateZ` and `2026-99-99T99:99:99Z`. Calendar-specific validity, such as
February having fewer than 31 days, is not established by this local referee;
it requires a format-capable validator or a separate calendar check. No
chronology check is claimed. The regression checks reject every case above
and accept a UTC-shaped timestamp.
See [review-refusals.log](review-refusals.log).

The referee does not check real file contents, ID resolution, time ordering,
independence, permission, computation or device execution. Those remain
manual review or future qualified .om checker obligations.

## Repository checks, unchanged failures retained

| Command | Exit | Observation |
| --- | --- | --- |
| `make test` | 2 | `src/omega_blackwell_engine.c:558` and `:393`: assembler operand mismatch for `ds`; aggregate suite did not complete |
| `make test-native-artifact` | 2 | Same unchanged engine compilation blocker; target did not complete |
| `make zero-cuda-gate` | 2 | Same unchanged engine compilation blocker; gate did not execute |
| `python3 tools/figures/make_figures.py --check` | 1 | Stale `stack.svg` |
| `bash tools/extract/audit-selftest.sh` | 0 | Scanner found all planted test secrets, including historical/deleted-file and commit-message cases |
| `git diff --check` | 0 | No whitespace errors |
| `crumb validate docs/independent-verification` | 0 | New crumb structurally valid |
| `crumb verify .` | 1 | Existing `docs/device-trace` and `tests/device-trace` crumbs are UNCOMPILED; no unrelated crumb repair included |

The external crumb tool was built from architecture revision
`d0f0b95d344b4c6628c3d868346932fd247ce168`. Its compile operation walks the
repository even when passed a directory. Generated changes to the unrelated
existing trace crumbs were discarded; the new directory reports CURRENT.
The remaining root verification failure is preserved in [crumb-verify.log](crumb-verify.log).

The same engine failure was measured on the original `b5e0da48` base before
this change. No barrier, target or gate was disabled. Raw final logs are
adjacent. `build/hello_gpu --sim` was not run because the build prerequisite
failed. Native host execution, all GPU arms, dynamic tracing, external
operator collection, second-machine and toolkit-absent experiments are
NOT_RUN. The schema is not a substitute for those tests.

Only new public content was scanned with the repository secret/private rules
and manually reviewed. This is not a whole-history audit; #89 owns that work.
