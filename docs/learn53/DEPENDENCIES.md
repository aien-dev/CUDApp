# CUDA-LEARN dependency and evidence reconciliation

Historical inspection baseline: CUDApp public main `80f3a2492eef315106feebfdffcb32de39be7d95`.
All #52-60 issue bodies/comments, #15/#16/#17/#19/#22/#26/#35 addenda, and linked Omega/trainer/RSI requirements were fetched before editing. Only existing open CUDApp PR was #28 (LocalRouter); it does not implement these tasks. No active CUDA-LEARN implementation owner was displaced. Parent ledger L777; independent review L778. Canonical backend, compiler, runtime, source owners and Merge Desk remain unchanged.

## Preserved implementation and evidence

| Existing owner/input | Resolved head | What its evidence actually establishes |
|---|---|---|
| #15, L466 host qualification | `2bf8b781219ac179336d5e7a876d63e6fb5d1873` | executable1f281708;35 host targets,8 legacy OSC slices; x86 cross-compile only. Native .om replacement and final consumer not established |
| #16, L467 numerical contracts | `2da3fa1b2392bcab593bbb5f71a36b04a6d289b1` | source3565287250e4ebd40e79a23d978edd3c936c94f2; finite nonempty FP comparison,285 API/236 router-simulator/19 trace fixture tests. No full model or GB10 receipt |
| #17, L468 benchmark collector | `65948c9c9420391782939e3492b9ed310ec774c7` | source4fdc3dabc5a0243ad93b6106e76497d0ce428d8a;18 formatter/12 wrapper/62 JS tests; real graph API but fixture observations only, batch1, unknown counters omitted. No measured native model baseline |
| #24, L475 reservation | `65e7ae4b8b52d51664153e9cf02b752473165fac` | source264c7752bceb38e18b4f58125a07ad1719b1ad6b;228 parent checks, real process recovery,8 mutants. No actual device allocation/cleanup or disjoint memory overhead measurement |
| #48, L600 weight codec | `566dc53502321166f4a898a29f8e4f828a209239` | existing native/weightpack/weightpack.om,464 codec cases and selected five real-matrix exact replays. Not a full payload census, loader/model cutover or speed result |
| Recovery provenance | `22965981baf78b3ae8768444f071f9b2c5410a55` | inventory/recovery snapshot, not qualification of this source or final artifact |
| #5, L457 canonical backend | `349913f17c083996680c72b32f767b225005155d` | existing integration branch owns runtime/library changes. Not copied or overwritten by this adapter |

These refs and receipts are left immutable. No cherry-pick of the legacy runners is presented as seeded delivery. New code resides only under `native/learn53/` and tests under `tests/learn53/`. The scalar reference is not a second runtime, GPU IR, disassembler, schedule search, store, trainer or judge. It adds a GPU-domain public contract evaluator to be consumed by the existing framework.

## Finite remaining work, in required order

| Issue | Current deliverable/status | Exact remaining prerequisite and owner |
|---|---|---|
| #53 | Seeded native-host public u32 manifest/reference/output slice; three operation kinds, eight registered shape families, refusal/negative controls. Full issue OPEN | #653/L775-L776 qualified framework/profile and independent split custody/result accounting; #15/L466 seeded eight-slice/build path; #16/L467 qualified numerical modes; #17/L468 native .om collector and full-cost hot-operator baseline; #10/#35 byte/launch path. Host-only results cannot substitute |
| #54 | NOT_RUN | #53's frozen independent framework plus qualified .om microkernel/lowering records #35, legal action interface #22/L473 and independent decoder #10. No speculative competing registry/search created |
| #55 | NOT_RUN, not NO-GO | #53 corpus; #13 repeatability; #10/#22 qualified native perturbation instruction/control and device path. Existing runtime/session owner must admit bounded instrumented artifact; no delay opcode guessed, no live waits removed |
| #56 | NOT_RUN | one supported task + actual #55 scope, existing Omega #623/#624/#625 checker/semantic interface and #35 binary correspondence. No duplicate proof engine or claim from hashes/real algebra |
| #57 | NOT_RUN | validated candidate #54 or #22, #16/#17 numerical/full-graph costs, #19 dependency records/L470, #24/L475 native memory/session cutover; #5/L457 canonical consuming graph. No incumbent/default modified |
| #58 | NOT_RUN | real applicable #53-57 receipts, Omega #651/#658 existing semantic store and RSI #67 independent evaluation; matched total-cost experiment. No new store or claimed break-even |
| #59 | NOT_RUN parameter training; current public fixtures are development-only, not a trained corpus | #53/#54 validated task/action/feedback interface, native gradients/update/checkpoint/resume under Omega606 and existing trainer L491, then aien-train3/4. No inference-budget gain mislabelled training |
| #60 | NOT_RUN | qualified native rollout/update path and objective snapshot semantics from Omega606/aien-train3/4, #16/#17/#35; must compare exact logits/probabilities/KV and wrong-snapshot controls. No probability cache or optimizer consumer changed |

At the initial inspection Omega #653's existing L775/L776 work was stopped at an account cap (superseded by the continuation below). Its coordination record explicitly says TICK-OFF is set and continuation is queued/unacknowledged. This chat did not restart them or invent a replacement framework. A dependency request was appended to the existing L775 inbox naming the public domain adapter and requesting its qualified immutable interface. Missing process/deadline/isolation/build-host primitives remain with Omega #635/#636/L408/L605. Native library tests proceed independently; authenticated custody and whole-workload qualification do not.

## Identity and cost limits

This slice is not using a model/tokenizer, policy checkpoint, generated GPU binary, launch descriptor or driver. Their identities are NOT_APPLICABLE to native-host reference execution; actual GB10/consumer rows are NOT_RUN. The manifest itself fixes the host target and numerical mode and rejects a GB10 target request.

No candidates generated, no optimization/training arms, no model inference or GPU time requested. Full lifecycle baseline, uncertainty, performance, energy, and caller CPU/GPU/time/token budget enforcement are NOT_RUN. Output/input byte capacities and executed check counts are measured/verified separately. Existing host suite timing printouts are diagnostics, not performance evidence.

No superseded production code/dependency was removed because none has yet been qualified and cut over by this additive domain slice. Legacy C/.osc/build/runner presence remains unfinished work under its existing owners, never the accepted final state. Removing it now would discard un-replaced consumers. A future integration must qualify and delete those active paths while preserving history.

## Continuation: canonical component became available

Source94ada80cd8acf2c31441ab1dcee98f896ce884c4 adds the frozen public catalog and
an actual native composition with Omega653 core84d03c5523f39ca1045d1c313c5f62597779462b.
L775/L776 were resumed by their authorized owner, not this chat. Published
upstream headdd28ebc0a84abc4dab3a38bde49aeae98c846c78 carries the bounded185-check
component and public observation evidence. This supersedes the old stopped-owner
status, not the remaining process/custody/full-cost gates. The continuation uses
that exact core by external source lock, without copying it into active product
source or changing its owner. Catalog83 and adapter160 assertions pass; detailed
limits and receipts are in CATALOG.md and LEARN53-L777-CONTINUATION.

Canonical admission/accounting is now exercised through the domain adapter,
including complete membership, failed outcomes, identity mismatch and cost
rejection. Metadata/cost test fixtures are explicitly synthetic. Actual native
candidate supervision, independent source/structure/lineage metadata, protected
splits, real cost collection, hot operators, GB10 and integrated consumers remain
unqualified. No validated training records or complete baseline costs are claimed.
