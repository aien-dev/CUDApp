# G17 immediate-index semantic profile:64/64, production HOLD

The existing independent Apple compiler sweep now has a native Omega semantic gate. **All64 G17 cases K=0..63 pass at gid0,7,31**, checking all six original-source stores: product-plus-sum, retained a/b/c, `a[gid+K]`, and the separately stored scalar `gid+K`. The prior32-case pressure profile still passes. No machine code was emitted by this helper, no GPU was launched, and target105 remains refused.

Shared research core SHA256: `2d8ffb719d95a6e66ce0611ffabd4fe29bc2b39f47bd20e5fc2d9699d234d24b`.

## Scope of the change

This reuses the existing G17 operand decoder. **No register field, immediate extraction or opcode packing was changed.** A second original-MSL checking contract was added: profile1 carries K, permits load indices through gid+63, checks exactly six output positions and verifies four loads, one multiply-add and the observed five/six additions. Profile0 preserves the previous pressure contract. Store offsets are separately checked before indexing the40-element seen table; allowing a higher load index does not expand the output-store domain.

The index test's input is one selector byte K followed by the original extracted `.text`. The selector names the original MSL source oracle; it is not used to manufacture the decoded instruction's answer. Native `probe` still executes the captured instructions, reads their encoded immediates and compares every store to the independent source contract.

The compiled prior pressure-only contract failed the valid index1 capture with exit66 (diagnostic offset220). After the source-contract extension, every index capture passed without changing any instruction decoder field. One intermediate build rejected the reserved identifier `bound` with compiler error16; it was renamed `addrmax`. That failed compile was not counted as a runtime test.

## Executed controls

See `m5-index-final-results.log`, `m5-index-envelope-results.json` and `m5-index-verdict.json`.

* 64 positive G17 index cases, each also exercised by the mutation driver.
* 384 single-bit immediate perturbations (six per capture), each rejected and followed by a restored positive. For K1..63 the selected add encodes K; K0 has no add-zero instruction, so its six controls target the output-address add1 instead. These are software interpretation/admission controls, not hardware splices.
* 64 deliberately wrong source-K controls (`K xor 1`) rejected, each restored positive passing. This ensures the oracle selector is meaningful rather than ignored.
* A compiled decoder fault discarding immediate bit5: K0..31 stay positive; all K32..63 refuse66. The source and native executable fingerprints are retained, distinguishing a compiled semantic failure from a build failure.
* All64 G16 index captures refuse66 in the G17 index profile.
* Both source-family boundaries checked: pressure mode rejects all64 index captures; index mode rejects all32 pressure captures under the independently specified index1 contract.
* All32 pressure programs,448 header/entry controls and nine pressure-field controls still pass with the same newly built shared core.
* Six input-envelope cases have the expected status and zero stdout: empty input and selector64/255 return65; missing, truncated and trailing-byte captures return66.

The index corpus itself is unchanged from the acquisition published at9b81e9d. The64 passing G17 captures are not recaptured or emitted by the interpreter. Original sources, binaries, symbols and hashes remain in `m5-index-corpus.tar.gz`.

## Reproduction

Use the pinned seeded compiler191b776e on the Spark and require compile0 plus a nonempty static AArch64 ELF before execution. Concatenate the shared `native/gpulab/probes/m5-registers.om` with one driver:

*  `test-m5-index-registers.om`: selector K followed by G17 index-K text ->0; same selector plus G16 text ->66.
*  `test-m5-index-mutants.om`: same positive packet ->0 only after six immediate-bit and one wrong-K rejection/restoration checks.
*  Existing `test-m5-registers.om`, `test-m5-entry-mutants.om` and `test-m5-register-mutants.om`: unchanged pressure-driver protocol and original pressure fixtures.

Preserved `index-controls/index-missing-contract.all.om` reproduces the missing-contract RED66. `index-controls/index-no-bit5.all.om` reproduces the lower-half positives/upper-half failures. All first-party executable semantics and assertions are Omega; shell/SSH perform host orchestration and existing utilities inspect outputs. The external Apple compiler supplies reference bytes only.

## Not closed

This is a bounded synchronous functional model of two compiler-source families, not a general ISA qualification. It does not settle physical memory-register identities, cache/discard flags, constant-program effects, scheduling/interlocks, memory ordering or arbitrary instruction combinations. The independent L742 warning about the G17P-labelled dependency reference remains applicable. The newer source needs its own independent review; the original L742 report is not reused as approval.

The production M5 emitter/interpreter and full positive M1–M5 coverage remain open. M1–M4 production sources are unchanged. No service, device, PR, merge or performance claim accompanies this research result.
