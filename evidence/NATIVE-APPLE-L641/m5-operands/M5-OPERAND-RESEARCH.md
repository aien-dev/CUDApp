# G17 operand investigation: historical partial result, HOLD

> Historical receipt for source `f5c6118e60ade1e09469ba78bbb003e02e0fc81f`. Reproduce the commands below from that pin, not current HEAD. The subsequent axis/entry experiment reaches32/32 within the bounded research profile; see [M5-AXIS-ENTRY.md](M5-AXIS-ENTRY.md), `m5-entry-verdict.json`, and the original L742 review plus its disposition. Production M5 remains refused. The historical observations below are retained rather than rewritten as a newer test run.

This is research evidence for L641, not an enabled M5 backend. **Target105 still refuses. No GPU execution, dependency-scheduling qualification, performance result, merge approval or full M1–M5 completion is claimed.** The existing M1–M4 implementation remains unchanged from functional source `4ff284212aad09ebf60832d83a25e2a6f8ef5ef3`; its frozen replay is recorded in the parent evidence directory.

## Actual new result

The native Omega `m5-registers.om` research interpreter reads independently Apple-compiled G17 text, decodes scalar register/address fields, and checks every store against the source-family expression. It does not consume a self-generated instruction fixture.

* 32 pressure kernels attempted. **30 accepted; cases18 and25 refused with exit66.** See `m5-register-results.log` and machine-readable `m5-register-verdict.json`. Full-corpus verdict is **HOLD_NOT_FULL_CORPUS_PASS**, not PASS.
* Each accepted kernel is evaluated at gid0,7,31. Inputs are distinct affine uint32 buffer values; all expected multiply-add and retained-value stores are checked. This is synchronous functional interpretation, not a model of physical execution or dependency timing.
* Nine field-directed byte mutations of pressure1 were rejected; the original positive input was restored and rerun after each. Fields: multiply sources A/B/C, result, memory data/base/index, address immediate, stop. Native mutation driver exits0 only when all nine controls and their restored positives behave correctly.
* G16 pressure1 is refused by the G17 checker (exit66).
* A successfully compiled missing-decoder control returns66 on actual G17 pressure0. It is not a compiler-error control.

All new implementation and executable tests are `.om`, built to static AArch64 ELF by pinned seeded compiler SHA256 `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`. Source and executable fingerprints are in the raw log. Shell commands only orchestrate the native tools and the external Apple oracle; Python was used interactively to inspect/aggregate tool output, not shipped as decoder, runtime or test harness.

## What changed in the interpretation

The fixed-register examples were insufficient. The retained-value and pressure cases expose explicit scalar load/store data and address-index registers, including register extensions. In this corpus:

* The first memory word's high nibble tracks the **data** register, not merely an address-register temporary. Register extension bits are separately present.
* Its byte3 tracks a register index in half-register units. It is not adequately described as a two-valued gid/direct-index mode.
* Stores can write values loaded much earlier, after unrelated address ALU instructions. A rule saying they always consume the immediately preceding producer is therefore insufficient.
* Destination and multiply operands are distributed across several bytes. Register32 and register64 transitions are exercised in the pressure series. The exact observed mapping is implemented in `dest`, `sourcea`, `sourceb`, `sourcec`, and `datareg`.

These conclusions come from our own MSL inputs, compiler-produced instruction bytes, retained-value relationships, and the native functional checks. The pinned external AGX3 report remains a source of hypotheses, not authority for unsupported bit semantics. Multiplication is commutative, so this family does not independently prove which multiplicand is conventionally named A versus B. Correlated prefix/control bits are not individually splice-proven.

## Unresolved cases and boundaries

Cases18 and25 have `_agc.main` at64, but begin with the eight-byte sequence `34 80 01 1a 40 00 a0 02` rather than the independently recognizable four-byte `get_sr` form. Their constant programs also differ. The interpreter deliberately refuses the unknown entry; no opcode is silently skipped and no presumed gid value is substituted. Independent evidence audit L742 investigates this and the remaining dependency/cache controls. Its report is not yet part of this receipt.

The helper only admits the frozen pressure-source family. It is not a general G17 decoder or an emitter, and no production target imports it. Sequentially evaluating loads and stores does not qualify scoreboard ordering, cache/discard behavior, inter-lane ordering, or a newly generated program's legality. Those remain prerequisites to enabling M5.

## Captures and reproduction

`m5-pressure-corpus.tar.gz` contains all64 paired G16/G17 captures (32 source pressure levels), source `.metal`, compiled own-shader outputs, symbol tables, acquisition logs, hash inventory, and selected G16 independent disassembly. `m5-live-corpus.tar.gz` retains the earlier live-operand minimal pairs. No Apple toolchain executable is redistributed.

Pressure acquisition uses `m5-pressure-input.om` -> external Apple `metal` -> `air-tt -arch applegpu_g16g|applegpu_g17g` -> native Omega Mach-O extraction -> independent Apple symbol listing. Main64 is established by all64 symbol tables; it is not inferred from a fixed-offset assumption. The root lab retains the pinned oracle/toolchain provenance.

From the repository, with `$OMC` bound to the pinned native compiler, build into fresh owned output files:

```sh
{ dd if=native/gpulab/probes/m5-registers.om status=none; dd if=native/gpulab/probes/test-m5-registers.om status=none; } | quiet-build "$OMC" > test-m5-registers
# Require compile exit0, a nonempty AArch64 ELF, then chmod +x.
./test-m5-registers < applegpu_g17g-p0.text
# p18 and p25 currently return66. Never label the full corpus PASS.

{ dd if=native/gpulab/probes/m5-registers.om status=none; dd if=native/gpulab/probes/test-m5-register-mutants.om status=none; } | quiet-build "$OMC" > test-m5-register-mutants
# Same build validation, then chmod +x.
./test-m5-register-mutants < applegpu_g17g-p1.text
```

The live/pressure source writers have exact-source goldens and empty/range/trailing-input refusal checks in `probe-gates.log`. The new immediate-index writer also passed its independently written index1 golden and refusal checks. Its distinct immediate0..63 compiler sweep is a continuation, **not included as a completed result here**.

The whole repository suite is not relabeled as newly run for these unintegrated research helpers. Existing broad M1–M4 receipts remain bound to their unchanged source. Source-reader/encoder integration, a full M5 functional gate, independent final-source review and physical qualification remain open.
