# Apple M1–M5 native laboratory implementation plan

> Inline execution by the original author, using superpowers:executing-plans. No implementation subagents.

**Goal:** Extend the existing bounded Omega GEMV laboratory through explicit M1, M2, M3, M4 and M5 targets without substituting labels for ISA evidence.

**Architecture:** Retain G13/G14. Add a separate G15 byte emitter and independent decoder. Newer generations require independently supported operand layouts, not shared opcode names. A recognized but unqualified generation must refuse emission/execution.

**Tech stack:** Handwritten seed-native Omega; shell/Make construction glue. Pinned upstream assemblers are external validation tools only, never linked or shipped as runtime fallbacks.

**Spec:** User request: all first-party work in Omega; implement personally; cover M chips through M5. Existing native/gpulab/README.md supplies numerical bounds and transactional semantics.

## Constraints and review focus
- Keep the existing signed-byte-in-i32 GEMV contract: rows 1..32, K 1..64, whole-dispatch transactional output.
- A virtual pass is not real-chip compatibility, dispatch, or performance qualification.
- M3 base-chip wait workaround must not disappear silently.
- Truncated code, wrong synchronization, late-lane faults, unknown encodings and unsupported generation requests must preserve output.
- Do not copy or execute the GPL agx-re implementation. Its published captures are research evidence, not sufficient proof of arbitrary operand semantics.

## Task 1: G15 core
- [ ] Add compilable refusal stub and test-m3.om. Observe runtime RED, not a compiler error.
- [ ] Implement m3.om: m3_generate(depth)->bool; m3_run(rows,depth)->int; m3_lane(lane)->bool. Use 14-byte memory, 12-byte imadd, short immediate, long wait and stop.
- [ ] Check frozen literal instructions and complete K64 shader against pinned TellowKrinkle/applegpu M3 assembler bde26552eaca66a0ba2d5a6362e08b90693fa4cc.
- [ ] Test all supported dimensions, signed-byte products, truncation, mutations and final-lane fault atomicity. Run full existing suite.

## Task 2: family dispatch and later ISA
- [ ] Give M1..M5 distinct IDs, preserve legacy request IDs 1 and 2.
- [ ] Establish generation-specific byte/operand evidence before enabling each new family. Unknown semantics refuse, never alias G15/G14.
- [ ] Implement M4 and M5 only once the complete bounded kernel has independently checkable operand layouts. Record unresolved fields precisely.
- [ ] Exercise actual binary request and emitted-wire paths for every enabled family, and refusal for every disabled family.

## Task 3: frozen replay and publication
- [ ] Commit source, rebuild clean export, publish logs/source hashes and external-oracle provenance.
- [ ] Push one candidate, verify remote head, update desk intake with exact scope. Keep L641 open if any M1–M5 acceptance row is incomplete.

## Research ruling
TellowKrinkle's G15 reference encodes imadd with low opcode 0x1f (plus add bit), whereas ADevWithAnIdea's later captures use 0x2f and report M5 data-register packing as implicit/positional and m5_alu operands raw/unmapped. Opcode similarity is not enough to enable later-generation GEMV. M3 is implementable now; M4/M5 completion remains evidence-gated, not declared impossible.
