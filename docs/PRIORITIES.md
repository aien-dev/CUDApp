# Integration and qualification first

Drake's execution order, 2026-10-22. Ledger L456. This orders the existing work; it is not a new GPU architecture or permission to interrupt running work.

| Order | Deliverable | Tracking | Acceptance gate | Existing ownership |
| --- | --- | --- | --- | --- |
| 1 | Complete serving path consumes the canonical pinned CUDApp backend | [Motor integration](https://github.com/aien-dev/CUDApp/issues/5) | MOTOR-CANONICAL-E2E: source/pin/library/binary provenance, shared train goldens, concurrent==alone, second turn, tool call, safe verified swap | L395 and motor/merge-train coordinator |
| 2 | Streams, events, isolation and same-process fault recovery | [Concurrency](https://github.com/aien-dev/CUDApp/issues/6), [recovery](https://github.com/aien-dev/CUDApp/issues/11), [chip faults](https://github.com/aien-dev/CUDApp/issues/12) | C4 hardware overlap and event data ordering; C5 isolation and endless-kernel timeout followed by successful work in the same process | L386; reuse omega #617 and #518 |
| 3 | General BF16/FP8 GEMM, reductions and scans | [Libraries](https://github.com/aien-dev/CUDApp/issues/9) | C3: 200+ GEMM shapes 1..8192 against reference and measured ceiling; reductions/scans through 2^28 with exact outputs and mutants | L386; reuse omega #615/#616 |
| 4 | Finish and integrate the independent disassembler | [Disassembler](https://github.com/aien-dev/CUDApp/issues/10) | C6: independent oracle agreement on every library kernel and no production/regular-test execution dependency on nvdisasm | L386; reuse omega #614 |
| 5 | Hardware qualification of the independent driver, kept isolated | [Independent driver](https://github.com/aien-dev/CUDApp/issues/7) | Simulator/allowlist first, then window 0, window 1, golden outputs, rollback and restored serving | L332, L390 windows, L401 packaging |

## Integration dependencies are not priority inversions

Omega [#618](https://github.com/aien-dev/omega-mv/pull/618) is the existing CUDApp-consumption PR. It removes GPU files that #614 through #617 modify. Preserve/carry those changes before extraction, or port them into CUDApp and update the pin. Do not blindly merge extraction first and discard their work. The goal at the front of the queue is a qualified canonical serving path; prerequisite code carries can precede that result without making a new feature project the priority.

Use the shared merge-train candidate and model load. Do not launch another per-feature full model copy, a duplicate worker, or another architectural redesign. Existing workers finish their current task, then take the ordered queue. This chat starts no further math expansion ahead of integration/qualification. Already-tested sparse code stays available on `feat/sparse-core`; its independent review L455 is still required and its host receipt is not GPU qualification.

## The immediate motor provenance gap

The read-only [live baseline](../evidence/integration-priorities/motor-baseline.json) records a healthy model listing on 18091, the actual running binary hash, and its build manifest. That manifest names Omega `ship/motor-l387` and its GPU archive hash, but does not name a CUDApp pin. Canonical consumption is therefore **NOT VERIFIED**, not disproven. No CUDA user-space library was mapped in the inspected process; this does not mean the NVIDIA kernel driver is gone. No process or driver was changed to take the snapshot.

## Completion means a receipt, not a checkbox

Every issue below has a mandatory exact-commit closure checklist. Keep it open until its final integrated source/pin/binary identities, commands, public raw evidence, gate exit codes, negative controls and required hardware tests are verified. Host/simulator evidence cannot close a chip gate. A push, merge, or agent summary is not a substitute. Do not put auto-closing keywords in a PR unless every closure gate is already met.

Supporting qualification: [memory budget](https://github.com/aien-dev/CUDApp/issues/8), [chip reproducibility](https://github.com/aien-dev/CUDApp/issues/13), [process overhead](https://github.com/aien-dev/CUDApp/issues/14). Coordinate their tests with the priority owner; do not deliberately exhaust memory or hang the chip while the working motor serves users.

## Safety boundary

Independent-driver experiments stay separate from the known working GPU path. No module unload, default-boot switch, live-server interruption or hardware reset follows from this priority update. The serving/window coordinator owns those actions after the relevant prerequisite receipts pass.
