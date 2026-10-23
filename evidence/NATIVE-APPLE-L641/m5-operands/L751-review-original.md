# L751 independent review — G17 entry/header and immediate-index profiles

## Verdicts

- **Bounded-source verdict: ACCEPTABLE WITH EXPLICIT LIMITS; no correctness defect found in the reviewed frozen profiles.** The source is a read-only research checker for two captured G17g source families, not a decoder/emitter or target105 approval. Pressure and K-index interpretations are separately selected and bounded; unknown family, entry, context, malformed capture and out-of-range cases fail closed. Review covers `da94831331abd3c66d05541fc53017ce4ed30371`, with evidence head `ead7fc45fad3aa3f0786fa60f5d0698eb2adc88b`.
- **General-ISA/dependency verdict: OPEN / NOT QUALIFIED.** Actual captures and pinned M5 references resolve useful operand roles, but do not provide a general G17g load/ALU modifier, cache, dependency, interlock, scheduling, or control-flow model. Bounded sequential CPU interpretation does not establish those properties, target legality, or hardware execution.
- No hardware launch, production/shared edit, target105 enablement, service action, or parent/ledger change was made.

## Review findings

### F1 — Bounded admission and source-family separation (no defect observed)

Reviewed the baseline-to-pin changes in `native/gpulab/probes/m5-registers.om`, `test-m5-entry-mutants.om`, `m5-axis-input.om`, `test-m5-axis-input.om`, `test-m5-index-registers.om`, and `test-m5-index-mutants.om`.

- `m5-registers.om`: `entrylen` admits the captured four-byte entry or exact eight-byte x-entry block; `constok` pins the paired constant-program prefix before body interpretation. The wide form seeds the observed live gid into r1 and begins the body after that block. This is structural admission of the known block/context, not a general instruction decode. The probe checks family selector range and forbids nonzero K in pressure mode. Unknown x/y/z entry variants are not silently generalized.
- Bounds are separately constrained: capture length 72..4096, body scan bounded to 256 steps, load address range is profile-dependent, and stores must fit the seen table/domain. The index profile requires the six specified store offsets and exact observed operation counts. Relaxing the load envelope to gid+63 does not enlarge the store domain.
- The axis producer has selectors only 0..95, constructs x/y/z plus bounded pressure, and the tester checks exact source length and bytes. This supports source-golden checking, not axis semantics in the G17 interpreter; y/z remain refused.
- Mutation routines save and restore each changed byte/selector before rerunning the positive. The controls therefore test refusal and restoration rather than accumulating mutations. The K=0 immediate controls deliberately exercise the add-1 instruction because there is no encoded add-0; they do not mislabel the absent add-zero as a field.

The source-level bounds/admission review is consistent with the committed receipts. A remaining semantic assumption is intentional and documented: the interpreter follows a synchronous functional model and treats captured header/context bytes as an admission contract; it does not execute physical constant-program effects or model inter-instruction timing.

### F2 — Frozen-source focused native replay (independent confirmation)

I rebuilt the exact pinned core-plus-driver concatenations with the pinned Spark compiler (`omc` SHA256 `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`), verified nonempty statically linked AArch64 executables, and used the archived G17 captures. Rebuilt executable hashes exactly match the frozen replay hashes:

| Probe | Native executable SHA256 | Observed focused cases |
|---|---|---|
| index contract | `825eb77cb8fc2d0accd02df40705fed9a7011f05beec8977d525dee177d0b3df` | K=0,1,31,32,63 all exit 0 |
| index mutants | `cb122f3c750bd9c129ba769813ad751fee912be04d377cd3ad4142c8d5b9ac33` | K=1,32,63 mutation/restoration controls all exit 0 |
| entry/pressure contract | `b5e9241326491611e19356ee7c57584bb65c89306dcde6718b209ac5ec453a0d` | p=1,18,25 all exit 0 |
| entry mutants | `6d57bf29336d482e3d56afb84692e23be2ac1cfc5775973a4d979dae5b0477d5` | p=18 structural mutation/restoration controls exit 0 |

These are CPU executions of the pinned Omega checker on existing compiler captures. They are not GPU results. The wider committed replay also records K=0..63 and pressure/entry controls; I did not repeat the full matrix.

### F3 — Capture-to-source discriminating correlation (actual bytes, not only prior prose)

The archived `m5-index-corpus/applegpu_g17g-i1.text` and `...-i2.text` are 232- and 232-byte captured instruction texts with `_agc.main` at offset 64. Their original MSL differs at the source index expression: `a[gid+1]` / `offset=gid+1` versus `a[gid+2]` / `offset=gid+2`; all six output roles are retained. In the captured body, the corresponding 12-byte `0x27` integer-add instruction at offset `0x5c` changes from `27 01 04 1a 25 00 a3 02 ...` (K=1) to `27 02 04 1a 25 00 a3 02 ...` (K=2), while the surrounding source-family operations remain in the same role. The native checker accepts each only under its matching K selector, and the compiled `K xor 1` controls refuse after restoring the source selector. This is a useful dataflow/encoding correlation for the tested compiler family; it does not decode all `0x27` bits or prove a universal ISA rule. K=0's control is intentionally different: the captured program has no add-zero counterpart, and its bit perturbations target the observed add-one output address.

## Operand/dependency evidence table

| Region / field | Evidence-backed statement | Evidence type and boundary |
|---|---|---|
| G17g entry/context | Exact observed eight-byte **x** entry and paired constant-program prefix are admitted; observed live scalar is r1 in this constant-program context. | x/y/z minimal pairs differ only at byte 65; G16 disassembly corroborates axis role but is not a G17 decoder. No general single-instruction/modifier decode. y/z refused. |
| Split address generation `?f` | Byte0 high nibble is destination address register; byte+1 is buffer/base slot encoding; byte+2 `0x03` is the address-gen signature; byte+3 is index mode. | EXP-M5-07 splice-and-observe report. Exact emitted variants remain a bounded capture family. |
| Split loads | Load leader/component-count family and element-size role (+3) are supported; byte+5 index register is explicitly splice-supported in EXP-M5-11; the preceding integer add carries `gid+K` for the tested indexed-source family. | EXP-M5-07 and later EXP-M5-11; see evidence conflict F4 below. No arbitrary addressing-mode coverage. |
| Split stores | Distinct store leader, data-source class/format roles and positional/implicit data behavior are supported for named variants. | EXP-M5-07/11 own-MSL and splice evidence; does not give a general source-register interpretation for all store forms. |
| `0x27` general ALU / integer add | Selected opcode/signature and some operation-family roles exist; operand packing is intentionally raw in the pinned integration report, and index immediate interpretation is qualified only for the frozen profile/corpus. | EXP-M5-11 explicitly keeps general M5 ALU operand packing raw; L751’s interpreter semantics are not a replacement decoder. |
| Load/cache/control tails | Some tested bytes are inert in particular splice cases; this does not establish all load-tail, cache, discard, format, scope, or context bits as inert. | Negative results are case-specific. In particular, do not generalize an unaltered output into a global field-absence claim. |
| Dependency/interlock/wait/scheduling | Unknown for general G17g. No approved wait encoding, scoreboard rule, latency model, cross-lane ordering rule, cache/discard semantics, or arbitrary producer-consumer sequencing follows. | The available scoreboard item is labelled G17P and is not automatically G17g evidence. No wait is inferred or guessed here. |
| Control flow / broad combinations | Not qualified by these pressure/index source families or their synchronous evaluator. | Narrow contexts and compiler-generated captures only; no arbitrary instruction mixes or target105 integration. |

### F4 — Evidence reconciliation needed; not a demonstrated source bug (MEDIUM evidence-quality risk)

The pinned `agx3-reference/EXP-M5-07-isa-semantics-report.md` says for the 10-byte load that byte+5/+6/+7 are inert in its tested cases (line 25). The later `EXP-M5-11-isa-integration-report.md` says the M5 load byte+5 is the index register and reports a splice forcing it to zero to read `a[0]` (lines 28–31). These are conflicting descriptions if read as the same bytes/form. EXP-M5-11 is the more discriminating, explicit load-index splice and the actual L751 captures contain distinct byte+5 values in their load forms; I rely on the later positive result for the named forms, but the older negative must be scoped to its exact tested form or corrected by the owner. Neither statement justifies treating all other load tail/cache/control fields as solved. This is an evidence reconciliation item, not proof that L751’s bounded source is wrong.

## Severity and disposition

- **No High/Critical code defect observed** in the frozen G17 x-entry/constant-prefix and immediate-index admission changes.
- **Medium evidence/documentation issue:** reconcile the EXP-M5-07 byte+5..7 negative with EXP-M5-11’s byte+5 index-register splice and retain exact form/capture scope. Preserve the G17P scoreboard provenance warning.
- **General dependency/ISA gate remains OPEN.** Do not promote this bounded CPU interpretation, compile-only corpus, reference byte mapping, or green controls to M5 backend qualification, physical dependency evidence, target105 approval, or hardware qualification.

## Artifacts and limits

- Report: `/Users/drakestapleton/.hermes/cache/scratch/amd-agx-native/m5-L751-review.md`.
- Private Spark outputs/captures: `~/handoffs/native-apple-L641/L751-review/` (four focused static native executables and selected archive extracts; no shared source or evidence was modified).
- No new `.om` probe was needed; the exact pinned Omega probes were rebuilt and exercised. The first compile attempt used a driver path missing its `native/gpulab/probes/` prefix and failed before a build; the corrected pinned-path builds succeeded. One first replay lookup mistakenly assumed the Mac scratch corpus existed on Spark; it was replaced with extraction of the committed corpus archive into the private review directory.
