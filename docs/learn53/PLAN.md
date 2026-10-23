# CUDA-LEARN public task adapter implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver the dependency-ready public native-host task-contract/output-evaluation part of CUDApp #53, not substitute it for final #53 qualification.
**Architecture:** A pure bounded CUDA++ domain module consumes a caller-owned frozen manifest and candidate output bytes. Omega #653 owns scheduling, custody, aggregate outcomes and budgets. CUDA++ #15/#16/#17 retain qualification, numerical policy and measurement ownership. No generated host code is executed by this module.
**Tech Stack:** Seeded Omega .om, compiler SHA256 191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9, Linux AArch64. Existing shell/Linux commands are disclosed development orchestration, not new delivered automation.
**Spec:** CUDApp #52/#53 and integration addenda #15/#16/#17; omega-mv #599/#653. NOT A MASTER PLAN.

## Global constraints
- New executable logic, tests and fixtures are .om. Preserve existing legacy code/history until its owning replacement and consumer cutover qualify; this additive slice does not retire the runtime.
- Own only native/learn53, tests/learn53, docs/learn53, evidence/LEARN53-L777. Do not resume existing prerequisite workers or create a second compiler/runtime/judge/store.
- Public development input only. Refuse validation/final splits at this entry; do not inspect sealed tasks or invent a seal.
- Exact modular u32 semantics only; no tolerance or FP support. No model inference, search, training, GPU execution, timing reward or performance claim.
- Bound manifest to 4096 bytes, each input/output extent to 256 u32 words, all loops by these bounds. Candidate output has exact padded extent plus two canaries, maximum1032 bytes. Registration limited to listed shape families.

## Review focus
- Length/count overflow before any read: exercise lengths0,95,4097,2^64-1 and forged u32 counts.
- Semantic-policy downgrade or false evidence: reject tolerance/mode, target and split substitutions before output comparison; candidate PASS text is not a result.
- Sparse output layout and ragged tails: compare every logical result and every padding/guard word; skipped/wrong-tail outputs must fail.
- Reuse after refusal: a malformed second record cannot reuse the previous success; valid third record must recover, inputs unchanged.
- Oracle correlation: literal Wolfram-checked add/reduce/matmul answers and compiled evaluator mutants, not expected values produced by this implementation.

### Task 1: Domain contract and independent evaluator
**Files:** native/learn53/task.om; tests/learn53/unit.om; docs/learn53/CONTRACT.md.
**Interfaces:** globals kt[4096] and ko[1032]; kvalid(task_bytes:ulong)->int; keval(task_bytes:ulong,output_bytes:ulong)->int. Caller fills manifest once from trusted pinned task storage; only ko receives candidate bytes. These are single-threaded scratch buffers, not an authority or sandbox.
- [ ] Write unit fixtures and kvalid/keval compilable stubs. Run native combined source; expect intended assertions to fail, not compile/setup errors.
- [ ] Implement canonical fixed little-endian manifest and exact shape/stride/count policy; prove structural/unsupported/split failures and no buffer mutation.
- [ ] Implement independent scalar u32 add, fixed-order sum and small matmul checking with guards; retain wrong values, skipped writes, sample memorization, strides/tails, weakened tolerance and spoofing controls.
- [ ] Execute complete native suite and compiled semantic mutants; keep every raw result and actual exit.
- [ ] Commit source freeze and reproduce from immutable Git export.

### Task 2: Data-only native entry point
**Files:** native/learn53/stream.om; tests/learn53/stream-input.om; tests/learn53/stream-check.om.
**Interfaces:** length-framed task then candidate bytes on stdin, no arguments/files/process launch. Result is authoritative process status; stdout contains only a fixed evidence-class/status record, never a candidate-provided result.
- [ ] Write native stream tests first; reject empty, truncated, extra and forged-length input; require no success output on failure.
- [ ] Implement bounded read/copy/check entry; run valid and wrong-output data through the real executable.
- [ ] Confirm static AArch64 ELF and no linked foreign runtime. Record build/input/binary identities and declared closure limits.

### Task 3: Review and dependency-safe publication
**Files:** docs/learn53/DEPENDENCIES.md; evidence/LEARN53-L777/; existing desk intake and issue-update packet.
- [ ] Preserve exact prerequisite branch/source/receipt pins and map every remaining #53-60 gate to its current owner. No hardware or proof credit from this unit.
- [ ] Run unchanged repository host suite separately. Preserve baseline/final failures without changing owned legacy gates.
- [ ] Independent reviewer checks exact final source, source closure, public oracle cases and mutants. Fix substantive findings with RED/GREEN.
- [ ] Publish scoped branch/evidence and request one reviewable PR through the sole desk; read back exact remote head and issue updates. Keep #52-60 open until their final acceptance is met.

## Pre-flight ruling
Omega653/L775/L776 currently stopped at account cap with TICK-OFF set; no published qualified adapter/profile. This module provides a non-speculative domain seam (trusted manifest plus observed output bytes) but does not impersonate the absent framework. #15 eight-slice .om runner, #16 FP comparator and #17 native graph collector migration remain owned prerequisites. #54-60 dependent experiments cannot qualify on this slice. No primitive/generator is promoted or timed here.
