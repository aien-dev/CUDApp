# G16/M4 increment and G17 investigation (L641 remains OPEN)

This increment adds a bounded, offline Omega G16 emitter and independent byte
interpreter. It does not qualify physical hardware or complete the M1–M5 request.
All first-party implementation, source writers, binary readers and tests are
Omega. Shell/Make are host orchestration; Apple Metal and the pinned upstream
assembler/disassembler are external validation tools only.

## Observed results

- Compilable M4 refusal baseline: compiler exit 0, test exit 1, assertion 701.
- M4 native numerical/encoding gate: 140137 checks, 0 failures.
- K1 mutation classification: 463 refusals, 37 changed answers caught by the
  scalar oracle, 12 equivalent on the chosen probes (not universal equivalence).
- Explicit M4 family path: observed runtime RED exit 1 before dispatch enablement.
- Enabled family paths: M1, M2, M3, M4. M5 request refuses exit 65, zero stdout.
- Entire K64 G16-profile output: 2710 bytes, equal to the independently assembled
  shared-field fixture; first-byte corruption is rejected with exit 3, no stdout.
- Complete native suite passes; earlier AMD, G13/G14, G15 and cross-target gates
  remain in the runner. G15/G16 cross-profile refusals are checked explicitly.
- Hardware: NOT_RUN. Speed: NOT_MEASURED. Independent review: PENDING.

Frozen functional source: `4ff284212aad09ebf60832d83a25e2a6f8ef5ef3`.
A fresh `git archive` replay passed the complete native suite, the independent
Mach-O reader tests (291 checks, zero failures), all 24 frozen compiler-container
extractions, and source-writer valid/refusal checks. The replay command is
`bash m4-replay.sh 4ff284212aad09ebf60832d83a25e2a6f8ef5ef3`; the orchestration
script is included beside this receipt. `source-files.sha256` inventories the
complete frozen native/gpulab input set. `native-binaries.sha256` identifies
executable outputs, not hardware code.

## Correction to the earlier M4 premise

The earlier note generalized a later-ISA arithmetic observation to M4 and M5.
Direct compilation contradicts that generalization. Apple `applegpu_g16g`
produces the G15-style 12-byte `0x9f` integer multiply-add and 14-byte memory
forms. Its scalar, vector and volatile scalar probes decode under the pinned
G15 reference. `applegpu_g17g` really produces a different encoding.

`m4.om` reuses the established field layout, not the G15 interpreter entry point.
It explicitly emits/decodes short waits rather than the base-M3 long-wait profile.
The independent evidence is layered: Apple G16 compiler captures establish the
shared fields; the external G15 assembler checks the entire selected profile.
The full Omega K64 stream is NOT claimed byte-identical to Apple's optimized K64
kernel. Apple selects different registers, schedules and constant programs.
Physical legality of this exact stream remains untested.

## External toolchain acquisition

Apple MetalToolchain asset 17B54:

https://updates.cdn-apple.com/2025/mobileassets/022-20354/192ADD28-31EE-43C8-972D-E5BED5FB1845/com_apple_MobileAsset_MetalToolchain/637B0E26-E00A-488B-BA3B-CB50A8ECFD74.aar

Archive SHA256, independently matched to zml/zml's public MODULE.bazel:
`84f1cec3984878294777349e794f7bff0407e501a728bd600dc8605fa36abf6f`.
Compiler reports Apple Metal version 32023.830. Native-translator tool identity
is retained in compiler output metadata. The toolchain was extracted and mounted
read-only in an isolated scratch directory, not installed system-wide. Apple
toolchain binaries are not redistributed here.

Apple's `aea decrypt` needs `-key-value base64:<public-asset-key>`, not the bare
base64 text. The decrypted archive is an AppleArchive patch: extract with
`aa patch -src /var/empty -dst <isolated-directory>`, then mount its Restore DMG
read-only. Obtain the public asset key/recipe from zml/zml, not from user secrets.

## Reproduce the independent compiler probes

`native/gpulab/probes/` contains the Omega source writers and bounded read-only
Mach-O reader. Compile each standalone writer with the pinned seed compiler.
For `macho-extract` or `macho-inner`, concatenate `macho.om` and its entry point;
for the parser tests concatenate `macho.om` and `macho-test.om`. Compile only after
checking compiler identity; require compile exit 0 and a nonempty static AArch64
ELF before executing it. These Linux binaries run on the Spark, not macOS.

The source writers accept one selector byte followed by EOF:
- `apple-oracle-input`: K in 1..64, ordinary signed-i32 GEMV source.
- `apple-operand-input`: K in 1..64, volatile loads preventing uniform lifting.
- `m5-roles-input`: selectors 0..5, three fixed volatile input loads and controlled
  multiply/add/or changes. Selector 3 deliberately leaves one volatile read unused;
  the compiler warning is retained, not a failing compile.

Run these generated sources through the separately identified Apple oracle:

```sh
"$METAL/usr/bin/metal" -std=metal3.1 -target air64-apple-macosx26.0 \
  -o probe.metallib probe.metal
"$METAL/usr/bin/air-tt" -arch applegpu_g16g \
  -platform_version macos 26.0 26.0 -o probe.gpubin k.mtlp-json probe.metallib
# Repeat with applegpu_g17g, preserving separate outputs.
```

The Omega reader obtains nested `__compute`/`__text` spans with bounds checks;
Apple `air-objdump --macho --syms` independently identifies `_agc.main`.
Do not assume the main kernel starts at text offset zero. The ordinary G16 K64
capture has a constant program followed by main at 0x240; the small/volatile and
role probes inspected here have main at 0x40. The initial attempt to treat the
constant program as main was discarded. Only explicitly named `actual-main` or
symbol-confirmed main disassemblies are used as evidence.

The external disassembler/assembler remains pinned to
TellowKrinkle/applegpu@bde26552eaca66a0ba2d5a6362e08b90693fa4cc (BSD-3-Clause).
The ordinary suite uses frozen fixtures and does not require Apple tools/Python.

## M5: actual progress, still not enabled

The independently compiled G17 captures now cover ordinary/volatile K1, K4 and
K64 plus six operand-role probes. This replaces the prior missing-tool blocker.
The fixed-load role probes establish real byte changes when the accumulator and
multiplicands are exchanged. Their multiply-add is a 14-byte form; merely applying
a published 12-byte generic-ALU label would desynchronize the stream. The `0xa1`
byte occurs in this integer-multiply path, so a global operation-name table alone
is not sufficient to infer semantics.

The remaining unknowns are the full operand/control-bit packing and memory
index/dependency contract needed for arbitrary supported depths. The lab must
not substitute whole-blob recognition followed by a scalar answer for a genuine
instruction interpreter. M5 continues to refuse until those fields are supported
by independent evidence and implemented/tested in Omega.

Apple `air-objdump -d` reports no instruction printer for this GPU target.
`applegpu-nt -S` did not produce usable GPU assembly; IR input was refused as an
unsupported input kind. Environment logging and a tested translator-option route
did not expose the missing assembly. These are preserved failed approaches, not
fabricated disassembly and not proof that the ISA cannot be recovered.
