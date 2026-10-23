# L742 parent disposition

## Verdict

Accept the audit's **bounded research-only / production HOLD** conclusion. L742 has delivered its read-only evidence audit; this closes only that audit deliverable. L641 remains open. Target105 remains refused, and no M5 hardware, scheduling, emitter or complete-generation gate is approved.

The original report is preserved unchanged as `L742-review-original.md`, SHA256 `51022578bff698d3e07d22d190e2fbb6e62cfcb111b483d449113be90608511d`.

## Verified findings and corrections

1. **Generation mismatch is real.** The inspected `m5-db.json` SHA256 is `dabc69357dccc017ff0ea9d984d498f65a8ca8c03c897364343bb89fbc20159b`. Its `scoreboard_model.note` explicitly says G17P. Its hardware-interlock assertions cannot qualify this G17g emitter. The synchronous research interpreter has no hazard model and makes no such claim. Keep the dependency gate open; do not invent waits or inherit a different generation's model.
2. **Unknown entries remain unknown.** Pressure18 and25 have the independently established main64 boundary and the same eight-byte prefix. No source change bypasses them. The later immediate sweep has128 paired captures and recognizable four-byte G17 entries throughout, but acquisition alone does not resolve the two pressure cases.
3. **The memory-field discrepancy was already an explicit research question.** The audit confirms the conflict with the saved report; it does not demonstrate that the native interpretation is wrong or that the saved report is right. Our captured-code/output consistency check is narrower than a general physical register contract. No production field map is changed merely to agree with the report.
4. **Original report line31 has a factual transcription error:** pressure18 contains18 extra declarations, `v0` through `v17`, not `v0` through `v13`. Pressure25 contains25, `v0` through `v24`. Parent checked the actual saved MSL declarations. This correction does not change the unknown-entry verdict.
5. **Execution receipts now exist and are separate from source-writer gates.** `m5-register-results.log` and `m5-register-verdict.json`, published with f5c6118, record30 accepted /2 refused kernels, the nine field-mutation controls and G16 refusal. The reviewer did not rerun them and read only the source-writer gate log. Preserve that review limit rather than retroactively calling these independently replayed tests. The nine mutations test the software interpretation, not physical splice behavior.
6. **Reviewed source identity matches the parent-tested source:** `m5-registers.om` SHA256 `d4bfcb71aa2f1c48843c58442a6eb0f83a2948165719715a174334c21631413d`; `test-m5-registers.om` SHA256 `cd8729500cfe1124d0fac8f95925af88951fde96cbbb9bb5598e3efff4953bfb`. No runtime code changes are made by this disposition.

## Additional reference check

Parent searched for an independent G17g dependency/entry reference rather than treating the missing decode as settled impossibility. Two actual retrieved sources did not close the gate:

* [gpu-kernel README at019b71dccd7f32df99825d6fcc420dc79816a1fe](https://github.com/dwijenpatel/gpu-kernel/blob/019b71dccd7f32df99825d6fcc420dc79816a1fe/README.md), SHA256 `2fa922402d0a9260fcb2b23b908da75f9d922851e46ca0b73efbcd97ce5ed9cc`: measurement-level Apple GPU work, explicitly distinguished from ISA-level decoding. It supplies no exact decode for the disputed prefix or dependency fields.
* [AGX3 predication gist, pinned raw revision](https://gist.githubusercontent.com/zboralski/4b091ecd9c5edf04c93cac8aeb8d1465/raw/dc3fb1f9ff1d276c11b0ad818f35b5fdf7fded62/agx3_predicated_execution_model.md), SHA256 `5cb3357fc08663fa3feae3a035474ce20b3bd3f53b065fa52642aebee1e8c6a0`: reports control-flow observations, but the inspected text supplies neither the claimed full opcode table nor the required entry/dependency decoding. Those broad external claims are not adopted as qualification.

The configured web extractor failed before fetching these pages; parent retrieved the actual pinned public text with curl. Neither source is a new runtime dependency or implementation source.

## Post-audit entry experiment

This disposition records the reviewed pre-change source. Subsequent parent-owned axis minimal pairs and a separately tested research-only entry/context extension are documented in `M5-AXIS-ENTRY.md`. They bring the frozen pressure-family check to32/32, without treating this original review as review of the newer source. They do not close the general opcode, physical dependency, cache, register-contract or production gates.

## Remaining acceptance owner

L641 retains register-field independence, immediate-family interpretation, alternate-entry and G17g dependency investigation, then the actual Omega M5 emitter/interpreter integration. L634 remains the sole existing publication/integration intake. No replacement worker or duplicate implementation lane is created by this audit.
