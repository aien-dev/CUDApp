# G17 address-family research: bounded native profile, production HOLD

L761 acquired the complete selector0..15 matrix for G16/G17, varying `gid`, `gid+5`, `j=b[gid]`, and `j+5` independently of retained-register pressure0..3. Parent verified exact membership, all273 capture-manifest hashes and all32 exact `_agc.main` symbols at0x40. The archive is `m5-address-corpus-L761.tar.gz`; the original worker reports, failures, source goldens and receipts are retained inside. The actual Mac output directory has an additional `lab/` component; no files were missing or regenerated to hide the path mismatch.

## Native functional extension

`native/gpulab/probes/m5-address-registers.om` is an experimental snapshot extending the reviewed `m5-registers.om` at da94831. The latter and every production backend remain unchanged. Experimental core SHA256: `06be6bf3e5c487f3dd8c77b2f657ca1550c20b645ffe8b73127b1644cb090a7b`.

The original core plus a new family2 driver compiled successfully and refused a genuine selector8 capture with66. The extension now accepts all16 G17 captures at gid0,7,31, checking every stored value against the original MSL. Its synthetic b input is `8 + ((index xor 21) and 31)`, so gathers are genuinely different from gid indexing while remaining in a bounded test address domain. It preserves the previous pressure/index input formulas outside family2.

No existing ALU/register field extraction changed. A new original-source contract supplies expected output values, source-dependent load/store/add counts, bounded addresses and output slots. A captured terminal tail byte is admitted exactly as32 for the a-load in the direct-j family, and0 otherwise. It is NOT executed as a wait or decoded as a cache/dependency instruction. This remains synchronous CPU evaluation, not GPU execution.

The first extension accepted12/16. The four pressure-zero cases refused at their first store because the compiler removes unused c and packs out into uniform slot2 instead of3. G16 independently decodes the matching packed uniform slot; the corrected `outbase` contract admits that precise source-dependent layout. The initial result is preserved. Two earlier compiler error12 failures from deep expressions were resolved by moving bounds and tail admission into small helpers; those failed builds were never counted as runtime tests.

## Discriminating observation and limits

Across direct-j selectors8..11, the inferred address register varies1,3,5,7 while the candidate load tail byte remains0x20. Across j+5 selectors12..15 that tail is0, while the index expression is still derived from j. The full changed spans are in the parent corpus-inspection JSON inside the archive. These are not whole-text single-byte minimal pairs.

A deliberately compiled alternative replaces the model's address register with the raw tail byte for the direct-j a-load. All four direct-j cases refuse, while the other12 controls remain positive. This rejects that literal substitution **within this effective register model**. It does not prove a physical register-number mapping, refute every possible interpretation of the external report, or establish that0x20 is a wait/cache bit. The tail's correlation with a direct load-to-load dependency is a hypothesis for further qualification, not an ISA fact.

## Executed controls and reproduction

- Six refusal/restoration controls per address capture: wrong family, wrong pressure, data-register bit, buffer-base bit, address-register bit, and exact tail-context bit. Totals:64 field-bit controls and32 wrong-selector controls, each followed by a restored positive.
- All16 G16 address captures refuse66 in the G17 address driver.
- Pressure32/32 and index64/64 regressions pass with the experimental snapshot, as do448 header/entry and nine previous pressure-field controls.
- Parent rebuilt the source producer byte-identically to the worker's final ELF; all16 outputs exactly match the captured MSL. Five exact source goldens pass. Empty/extra/16/255 producer inputs return65 with zero output. The worker's original extra-byte acceptance is preserved in the archive.

Concatenate `m5-address-registers.om` with `test-m5-address-registers.om` or `test-m5-address-mutants.om`, then compile with the existing seeded omc SHA256191b776e. The packet is one original selector byte followed by the independently compiled text. The driver itself asserts the captured program's source behavior at the three gids. Existing pressure/index/entry/field drivers also link against this experimental snapshot for regression. `address-controls/tail-as-index.all.om` preserves the compiled alternative. `address-controls/missing-contract.all.om` preserves the missing-family RED.

The producer and its exact goldens/refusal gate are under `native/gpulab/probes/l761-address/`. Host utilities handle build/transport and inspect archived bytes; all new behavioral implementation and assertions are Omega. Apple compiler executables are not redistributed. No foreign inference runtime or GPU dispatch is introduced.

## Frozen-source replay

The full source tree at `e9ff18accda8372584d20cd14bd9db3747b7e733` was exported into fresh private output space. Seven reconstructed model/control executables matched the author artifacts byte-for-byte. The producer also matched its prior binary and regenerated all16 captured sources exactly. The full address matrix, compiled alternative, prior pressure/index gates, header/entry controls and field controls passed. A freshly built index-mutation driver additionally passed all64 cases;64 G16 index,96 G16 axis and64 non-x G17 refusals were retained.

The first replay reached the earlier gates but stopped when its hash inventory expected an unstaged `address-tail-as-index.all.om` filename. No native assertion was waived: the corrected orchestration materializes that exact committed control source before compiling/hashing it. The first command/stdout/stderr are preserved as `m5-address-frozen-failed-*`; the complete fresh replay is `m5-address-frozen-command.txt` and `m5-address-frozen-replay.*`. Old tar provenance-xattr warnings and expected refusal traces remain in raw stderr.

## Acceptance still open

This is a new bounded research profile, not production target105 support or full M1–M5 completion. The final address extension needs independent source review. General G17g instruction boundaries, physical memory-register identities, modifier/cache/dependency semantics, arbitrary scheduling and hardware behavior remain unqualified. The original pressure/index core and M1–M4 production files remain unchanged.
