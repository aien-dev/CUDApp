# Execution readiness

Inspection date: 2026-10-23. Implementation evidence and immutable branch
sources take precedence over an open issue label.

| Requirement | Evidence and disposition |
| --- | --- |
| Existing 27B model and fixed speculation | Historical sovereign-core `speed2/specdecode` at `68fa1683874899eb0b6b4ab68bb2b9b19fcbcbb1` contains the qwen38 model, MTP, verification and accepted-state work. This is existing Rust integration over native GPU code, not a seeded `.om` adapter. Reuse its contracts and references; do not introduce a second runtime |
| Whole-model correctness | Prior branch evidence contains complete-model checks beyond the two/four-layer evidence in Omega PR #559. This does not qualify a new build or this experiment's two context lengths. Requalification remains required |
| Canonical backend | [CUDApp #5](https://github.com/aien-dev/CUDApp/issues/5), owner L457, has an existing integrated candidate with host evidence. Its actual serving/device gates remain separate. [Omega #618](https://github.com/aien-dev/omega/pull/618) is open |
| Existing timing owner | [CUDApp #17](https://github.com/aien-dev/CUDApp/issues/17), L468, branch `roadmap/17-native-benchmark` at `65948c9c9420391782939e3492b9ed310ec774c7`: graph/request collector interface and raw observations. This change consumes the same timing concepts in `.om`, without copying its C/JS runner or inventing device times |
| Seeded compiler | [omega-bridge source](https://github.com/aien-dev/omega-bridge/tree/0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9) reconstructs compiler SHA-256 `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`. This component uses that immutable compiler |
| Real adapter and host facilities | [Omega #636](https://github.com/aien-dev/omega/issues/636), [#637](https://github.com/aien-dev/omega/issues/637), existing L408/L605 and experiment L576. Recent native device-call work exists, but no qualified complete seeded 27B attachment is established by this component. Do not describe all language facilities as absent based on issue labels |
| First-party language | The experiment requires handwritten seeded `.om` adapters, workload preparation, scheduling, validation, measurement and reporting. Legacy shell development reconstruction is disclosed; a foreign-language model runner cannot silently satisfy the experiment |

## Remaining acceptance, in order

1. Resolve the exact checkpoint/tokenizer/precision and all executable/backend
   identities locally. Freeze a single ordinary reference and a development-only
   fixed-budget choice. Record whether base and ordinary are identical.
2. Connect the existing 27B/MTP/complete-state owner to the qualified seeded
   adapter. Prove complete prefill/decode, correction-token consumption, EOS,
   cancellation and rollback, wrong-backend refusal, and first-divergence checks.
3. Feed completed native observations into the collector at 1K and 32K. Preserve
   real timestamps, stage coverage, raw samples, costs and memory per shape.
4. Run balanced held-out prose/code/edit/copy/Omega fixtures with separate
   development data, natural EOS, at most 256 new tokens, and at least five
   paired repetitions. Six or more are needed for the report's finite 95% median
   interval; eight gives a balanced four-arm order. Report cold/warm separately.
5. Compare engine and served endpoint separately; retain failed/contaminated
   groups. Publish public fixtures and reproducible results. Keep checkpoint
   paths, private traces and machine-specific configuration local.
6. Obtain independent final-source review and qualify the integrated artifact.
   Close #64/#65/#66 only when their remaining gates pass, then #63.

No production swap, driver change, extra model server or GPU trial is part of
the component receipt. No performance result is inferred from synthetic costs.
