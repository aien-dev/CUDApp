# G17 axis minimal pairs and bounded pressure-entry contract

> Entry-only source snapshot `10c05b26a14e8a3623d6bf86a10ea30bb4803566`; reproduce this receipt from that pin. The subsequent shared pressure/index source and its separate results are documented in [M5-INDEX-SEMANTICS.md](M5-INDEX-SEMANTICS.md). Neither snapshot enables the production M5 backend.

## Verdict

**BOUNDED_PRESSURE_PASS_PRODUCTION_HOLD.** The research-only Omega interpreter now accepts all32 frozen pressure programs at gid0,7,31. This is not an M5 backend, complete ISA decoder, physical dependency model or hardware result. Target105 remains refused; M1–M4 functional files remain unchanged from4ff284212aad09ebf60832d83a25e2a6f8ef5ef3.

Final research source SHA256: `df07b719e3a6582f375e643042268dd1e4a49cef76db954a1cfca1f248e677cf`.

## Discriminating experiment

The previous p18/p25 eight-byte entry was not skipped or accepted on resemblance alone. An Omega source producer changed the original scalar thread-position parameter to `uint3 pos`, then selected `pos.x`, `pos.y` or `pos.z` into the existing scalar gid variable. Every pressure level0–31 was compiled for both G16g and G17g:96 source files,192 captures. Its byte selector is `axis * 32 + pressure`, with axis0/1/2 denoting x/y/z. Source files keep the historical `p<selector>` filename convention.

* Exact native source goldens for all three axes passed. The old pressure writer first failed the new x golden with exit1. Empty input, selector96 and extra input refused65 with zero output.
* All192 exact `_agc.main` symbols are at text offset64. Symbol matching must be exact: `_agc.main.constant_program` is a distinct symbol, not a second main.
* All64 new x captures are byte-identical to their original pressure captures across both generations.
* Across128 x/y and x/z minimal pairs, the **only changed byte in the entire extracted text is offset65**, incremented by1 or2. Complete per-pair observations are in `m5-axis-minimal-pairs.json`.
* Thus p18/p25 have x-entry `34 80 01 1a 40 00 a0 02`, y-entry `34 81 01 1a 40 00 a0 02`, and z-entry `34 82 01 1a 40 00 a0 02`. Their body and constant-program bytes remain identical across axes.
* The separately decoded G16 pressure18 axis cases read `thread_position_in_grid.x/y/z` into r1. This is corroborating G16 evidence, **not** a G17 decoder. The existing G17 scalar body consumes r1 for gid arithmetic; substituting r0 in the new entry contract makes both18/25 fail, while an unaffected short-entry case still passes.

## What changed, and what did not

The research model admits precisely the captured eight-byte **x-entry block** and seeds its live gid result into r1 before decoding the body at72. This is a bounded entry-block contract, not a claim that every bit of a general eight-byte instruction has been decoded. Other entry patterns remain refused. In particular, y/z are acquisition references, not newly enabled interpreter profiles.

Admission also pins the corresponding captured constant-program prefix. Previously the first12 bytes were unchecked; a new native header mutation test exposed that gap with exit2. The correction checks the exact normal prefix or the exact stop/padding prefix paired with the wide entry. These bytes are validated, not simulated as physical constant-program effects. Unknown header contexts cannot enter the model.

The original L742 audit and a separate disposition are retained unchanged/corrected separately. That review predates this source extension and does not approve it. Its warning about a G17P-labelled scoreboard reference still applies; no absence-of-wait inference, guessed wait instruction, memory-order claim, or target105 integration follows from these results.

## Actual native gates

`m5-entry-final-results.log` and `m5-entry-verdict.json` contain the final-source results:

* Original G17 pressure cases:32/32 accepted at all three existing gids.
* Newly acquired x cases:32/32 accepted; these are duplicate machine programs, not32 additional unique programs.
*448 header/entry single-bit refusals, each followed by a restored positive:128 for short-entry p1 and160 each for p18/p25. These are **structural admission controls**, not physical semantic splice experiments.
* Existing nine field-directed mutations still rejected, each restored positive passing.
* Compiled old-entry control:18/25 refuse66. Compiled wrong-r0 control:18/25 refuse66; unaffected p0 passes.
* Compiled unchecked-header control returns2, demonstrating that the new admission test detects the previously accepted malformed context.
* All96 G16 axis captures refuse66 in the G17 model. All64 non-x G17 captures refuse66 under the deliberately x-only profile.

No altered shared backend/test gate, production source, GPU launch, performance measurement, service change or proprietary compiler redistribution is involved. Native source generation, interpretation and semantic/mutation assertions are Omega. Shell/SSH orchestrate the pinned compiler, native programs and external Apple compilation; existing Python tools inspect archived data and disassemble G16 only.

## Reproduction boundary

Use pinned seeded AArch64 compiler SHA256 `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9` on the Spark. Compile these source concatenations, requiring successful compilation and a nonempty static AArch64 ELF before execution:

1. `native/gpulab/probes/m5-registers.om` + `test-m5-registers.om`: feed each original G17 pressure capture and each new x capture to stdin; expect0.
2. Same core + `test-m5-entry-mutants.om`: feed original p1,p18,p25; expect0.
3. Same core + `test-m5-register-mutants.om`: feed original p1; expect0.
4. Feed the96 G16 captures and64 G17 y/z captures to the first executable; expect66, not0.
5. Preserved `axis-entry-controls/*.all.om` bind the compiled old-entry, wrong-register and unchecked-header controls to the recorded native executable hashes. Their expected statuses are given above and in the log.

Compile `m5-axis-input.om` and `test-m5-axis-input.om` separately. The exact-source test receives one axis byte0–2 followed by pressure-one MSL; producer selectors1,33,65 are its x/y/z cases. `m5-axis-corpus.sh` performs the offline Apple compilation and native container extraction. These pressure kernels have overlapping stores and **must not be launched**.

The acquisition archive includes original sources, compiler outputs/logs, extracted text, symbols, hashes and final exit sentinel. Archive metadata xattrs are excluded; the earlier transfer emitted harmless provenance-xattr warnings, not capture or test failures. Final raw stderr is retained separately because expected refusing native controls emit their two-byte diagnostic offset.

## Frozen-source replay

A complete Git archive of source `10c05b26a14e8a3623d6bf86a10ea30bb4803566` was exported into fresh Spark output space. Its archived pressure/axis inputs passed their checksum manifests; all32 pressure and32 x replay inputs,448 structural controls, nine field controls,96 G16 refusals and64 non-x refusals passed again. All eight reconstructed native executables were byte-identical to the originally tested executables; compiled old-entry/wrong-register/header controls retained their expected failures. Exact receipts are `m5-entry-frozen-replay.log` and `.stderr`.

The first replay attempt stopped before building or testing because `archives.sha256` contains repository-root-relative paths but was invoked inside the evidence directory. That setup failure is retained as `m5-entry-frozen-replay-failed-cwd.*`. The corrected invocation runs that manifest from the exported repository root and `axis-archive.sha256` from the evidence directory; neither manifest nor any acceptance check was weakened. Old pressure archives emit provenance-xattr warnings on GNU tar; the raw stderr preserves those warnings and expected diagnostic bytes.

## Remaining work

The immediate0–63 corpus still needs its semantic gate. General memory-register roles, modifier/entry semantics and G17g dependency scheduling remain unresolved beyond the stated profiles. A production M5 emitter/interpreter and positive M1–M5 completion remain open; this result supplies evidence for those tasks, not a substitute definition of done.
