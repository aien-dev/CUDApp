# L742 — independent G17 scalar/dependency and entry-prefix audit

**Disposition:** scalar/register evidence is useful for a bounded corpus-specific synchronous model, but it does not yet qualify general G17 operand semantics or dependency scheduling. The eight-byte entry form remains **UNKNOWN** and correctly fails closed in the research interpreter. No implementation or hardware launch was performed. This report does not close L742.

## Scope and evidence inspected

Read the self-contained brief, research-only `m5-registers.om` and `test-m5-registers.om`, pressure MSL/capture/symbol files, saved M5 ISA reports and `agx3-reference/m5-db.json`; also read the parent L641 README and existing M5 tool-gate receipt over `ssh spark` (read-only). No parent source was changed. The local `m5-pressure-corpus/SHA256SUMS` verification completed with every listed entry `OK`; the corpus has 32 G17 text captures, 32 symbol tables and 32 MSL files. All G17 symbol tables place `_agc.main` at file offset `0x40`.

## Findings — scalar registers and dependencies

### What is supported (bounded PASS)

- `m5-registers.om` is explicitly labelled research-only and says it does not qualify dependency/cache/latency. It decodes a small accepted family, executes instructions synchronously, and compares output-store values to an expected function derived from the original MSL (`expected`, lines 48–55; `memory`, 57–93; `arithmetic`, 160–236). The outer check evaluates synthetic `gid` values 0, 7 and 31 (`lane`, 260–288; `probe`, 310–327), validates final store coverage/statistics, and rejects unrecognized forms. This is meaningful semantic cross-checking of the supported capture sequences, not hardware output evidence.
- The supplied brief reports the Spark-built probe accepted pressure inputs 0–31 except 18 and 25 (RC 66), and checked all three gids for the successful captures. I did not rerun that executable; the saved local tree contains the source/test but not its run receipt. The parent's separate `m5-probe-gates.log` concerns source-generator tests and explicitly ends `BACKEND_NOT_IMPLEMENTED; HARDWARE_NOT_RUN`; it is not evidence that this register interpreter passed.
- The capture family is broad enough to be useful: the original `pN.metal` files vary the number of extra live loads/stores while preserving `va * vb + vc` and observable stores. The exact-source probe checks output values rather than merely instruction bytes. It is therefore stronger than self-round-trip or a compiler-success check.

### Assumptions / unresolved scalar mapping

- The semantic check is still a *model of captured code*, not an independent hardware oracle. Register initialization, instruction recognition, operand extraction, register-file updates and expected-value mapping are implemented together in `m5-registers.om`; there is no independently decoded byte-to-register result file or hardware result buffer in this evidence. No field-by-field bit mutants or independent decoder comparison were run in this audit. The corpus proves the current interpretation is consistent with the selected original MSL cases, not that every scattered operand bit has the claimed role for all valid streams.
- There is a specific cross-source discrepancy to resolve before promoting the memory-register model. In `m5-registers.om`, `memory()` treats `byte+3 >> 1` as an index-GPR number and `datareg()` extracts a register from the address-generation leader, then the model assigns the loaded data to that register (`m5-registers.om` lines 16–21, 35–37, 70–79, 125–140). For example, the p0 capture at main offset `0x44` is `4f 00 03 0e`; the model's index extraction interprets `0x0e >> 1` as GPR 7. The pinned local M5 reference instead describes address-generation byte+3 as an access/index *mode* and byte0 high nibble as the destination address register; its enumerated modes include `0x02`, `0x00`, `0x04`, `0x06`, `0x08`, and `0x18`, not `0x0e`. It also describes the M5 load result as positional and the store data register as implicit/positional (see `agx3-reference/m5-db.json`, `m5_addr_gen`, `m5_load`, and `m5_store`). The reference is not gospel and may cover a different form, but this mismatch means the successful arithmetic-output check alone does not settle which physical register the address-gen/load fields name. Treat these specific p0 pressure encodings as a local empirical dialect until independently adjudicated; do not merge the interpretations by assumption.
- Arithmetic checking covers the one product-plus-sum expression in this pressure family and the values induced by the fixture's three input formulas. It does not establish all ALU opcodes, high-bit/wrap cases, immediate forms, register aliasing, predicates, or general source/destination field independence. The `dest`/`sourcea`/`sourceb`/`sourcec` scattered-field extraction in the model is not backed here by a separate minimal-pair splice for each bit.

### Dependency / scoreboard boundary

- The interpreter has no pending-write state, issue queue, wait/scoreboard representation, lane divergence, memory ordering, or asynchronous completion model: updates happen immediately in source iteration order. Passing it therefore says nothing about whether a newly emitted load/ALU/store sequence is hazard-safe on physical G17g.
- The saved G17g capture streams are compiler-generated schedules; observing adjacent producers/consumers or no visible wait instruction is not proof that a hand-generated sequence can omit waits. The M5-specific reports document selected instruction encodings and field splices, not a qualified dependency contract for new emitted programs.
- Do **not** import `m5-db.json`'s `scoreboard_model` as M5 evidence: that entry explicitly calls itself **G17P**, and claims a G17P hardware interlock/no-wait model. It is the wrong generation label for this G17g task. Conversely, absence of explicit waits in the inspected G17g captures does not prove an interlock. For new code, load-to-consumer latency/interlock, multi-outstanding behavior, reuse of a pending destination, write-after-write/read-after-write, ordering/fences, and any required safe scheduling discipline remain **UNKNOWN**. Until proven, conservative fail-closed refusal is warranted; no guessed waits or scoreboard bits.

## Eight-byte entry prefix — UNKNOWN (observed, not decoded)

- The 32 G17g `_agc.main` symbols are at offset `0x40`. At that offset, 30 pressure captures use the four-byte-shaped form `xx a0 10 06` (the first byte varies: `7c`, `9c`, `bc`, `dc`, or `1c`). Captures p18 and p25 instead begin with the same eight bytes: `34 80 01 1a 40 00 a0 02`. This is directly visible in `m5-pressure-corpus/applegpu_g17g-pN.text`; p18/p25 are not decoder truncations or an offset change. Their source files also confirm different register-pressure counts (p18 has `v0..v13`; p25 has `v0..v24`).
- The constant-program area differs too: p0 begins `23 00 07 02 42 20 a0 02 0e 00 00 00`, while p18 and p25 begin `0e 00 00 00` followed by `06` stop/padding bytes. This is correlation with the two refused cases, not proof that the prefix is a stack/spill/entry operation or that the changed constant program causes it.
- Current `m5-registers.om` hard-requires the ordinary entry bytes `byte+1..3 = a0 10 06` and a low-nibble `c` leader (lines 249–258); it seeds the gid GPR from the leader's high nibble, then starts decoding at byte offset 68 (lines 264–266). Therefore p18/p25 fail before any scalar instruction semantics are exercised. That is a correct fail-closed boundary; accepting them as if they had the ordinary prefix or moving the start offset would be unsupported.
- No public reference or inspected decoder/tool artifact found in this audit assigns semantics to the literal eight-byte G17g prefix. The M5 reference DB does not describe this entry sequence, and the available independent TellowKrinkle decoder artifacts in this corpus are G16/M3 material, not a G17g entry oracle. **Do not strip, skip, synthesize, or reinterpret these bytes.** Required evidence to close this item is a controlled compiler minimal pair that preserves the entry/constant-program context while varying one relevant source/ABI condition, plus an independent exact boundary/length determination and semantic decode for both p18 and p25. Those probes were not run here.

## Strict status

| Claim | Status | Boundary |
|---|---|---|
| Selected pressure-capture operand mapping agrees with the MSL outputs under the supplied synchronous interpreter | **PASS, bounded / parent-reported** | Only inputs 0–17, 19–24, 26–31 and synthetic gids 0/7/31; not rerun here; not hardware execution |
| General G17 scalar register/operand ABI | **UNKNOWN** | Memory-field/register role discrepancy and no independent per-bit field controls |
| M5 split-memory field facts in the saved M5 report/DB | **Evidence present, scoped** | Own MSL compiler captures/splices; do not treat every encoded pattern as same form |
| G17g RAW dependency, latency, scoreboard/interlock and memory-order contract | **UNKNOWN** | G17P scoreboard entry is generation-mismatched; synchronous model cannot establish it |
| p18/p25 eight-byte main-entry form and changed constant-program prefix | **UNKNOWN** | Bytes and offset verified; operation semantics and cause not decoded |
| Hardware qualification / native target105 integration | **NOT ESTABLISHED** | No launch; research interpreter is not integrated target105 |

## Checks run in this audit

- Read-only verification of the complete `m5-pressure-corpus/SHA256SUMS` list: all entries returned `OK`.
- Inspected all 32 G17g main-entry byte windows at offset `0x40`, plus the first 64 bytes of p0, p18 and p25; confirmed 30 ordinary-shaped entries and the repeated p18/p25 eight-byte sequence.
- Confirmed corpus membership counts (32 G17 text captures, 32 symbol tables, 32 MSL files) and main offset `0x40` in symbol tables.
- Reviewed, but did **not** rerun, the parent-reported interpreter result; no Omega executable logic, tests, backend, or hardware work was authored or run.

## Sources inspected

- `/Users/drakestapleton/.hermes/cache/scratch/amd-agx-native/L742-brief.md`
- `/Users/drakestapleton/.hermes/cache/scratch/amd-agx-native/m5-registers.om`
- `/Users/drakestapleton/.hermes/cache/scratch/amd-agx-native/test-m5-registers.om`
- `/Users/drakestapleton/.hermes/cache/scratch/amd-agx-native/m5-pressure-corpus/` (MSL, G17g captures, symbol tables, `SHA256SUMS`)
- `/Users/drakestapleton/.hermes/cache/scratch/amd-agx-native/agx3-reference/m5-db.json` (SHA-256 `dabc69357dccc017ff0ea9d984d498f65a8ca8c03c897364343bb89fbc20159b`)
- `/Users/drakestapleton/.hermes/cache/scratch/amd-agx-native/agx3-reference/EXP-M5-07-isa-semantics-report.md`
- `/Users/drakestapleton/.hermes/cache/scratch/amd-agx-native/agx3-reference/EXP-M5-11-isa-integration-report.md`
- Spark read-only: `~/workspace/cudapp-native-amd-agx/evidence/NATIVE-APPLE-L641/README.md`, `evidence/NATIVE-APPLE-L641/m5-operands/probe-gates.log`; the latter states `BACKEND_NOT_IMPLEMENTED; HARDWARE_NOT_RUN`.
